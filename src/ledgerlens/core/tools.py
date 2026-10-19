"""Tool definitions: pure compute functions behind strict JSON schemas (BUILD_PROMPT section 9.2).

One implementation per calculation. The agent loop, the MCP server and the API all call tools through
``execute_tool``, which validates input, writes the ``audit.tool_calls`` row and hashes the output.
Numbers leave a tool only as ``NumberRef`` objects created by ``ToolContext.number``, so every one has
a lineage row.
"""

import json
import re
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from ledgerlens.core.approvals import ApprovalError, ApprovalQueue
from ledgerlens.core.db import connect, utcnow
from ledgerlens.core.lineage import NumberRef, new_id, record_lineage, sha256_text

# Registry names are dotted ("rc.optimize_panel"); the Messages API only accepts [A-Za-z0-9_-],
# so dots become "__" on the wire.
TOOL_NAME = re.compile(r"^[a-z][a-z0-9_]*(\.[a-z][a-z0-9_]*)+$")
MAX_API_NAME = 64


class ToolResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    data: Any
    numbers: list[NumberRef] = Field(default_factory=list)


class ToolInputError(ValueError):
    """The model (or a client) sent arguments that fail the tool's input model."""


class OutputBlockedError(RuntimeError):
    """A tool's output guard refused its result; nothing is returned to the caller."""


@dataclass
class ToolContext:
    db_path: Path
    agent_run_id: str
    tool_call_id: str

    def number(
        self,
        *,
        label: str,
        value: float,
        unit: str,
        sql_text: str,
        source_tables: Sequence[str],
        row_ids: Sequence[str | int],
    ) -> NumberRef:
        return record_lineage(
            self.db_path,
            tool_call_id=self.tool_call_id,
            label=label,
            value=value,
            unit=unit,
            sql_text=sql_text,
            source_tables=source_tables,
            row_ids=row_ids,
        )


ToolFn = Callable[[Any, ToolContext], ToolResult]


def _assert_closed_objects(node: Any, path: str = "$") -> None:
    """Strict tools need additionalProperties: false on every object schema, including $defs."""
    if isinstance(node, dict):
        if node.get("type") == "object" and node.get("additionalProperties") is not False:
            raise ValueError(f"schema object at {path} must set additionalProperties: false")
        for key, child in node.items():
            _assert_closed_objects(child, f"{path}.{key}")
    elif isinstance(node, list):
        for i, child in enumerate(node):
            _assert_closed_objects(child, f"{path}[{i}]")


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    input_model: type[BaseModel]
    fn: ToolFn
    writes: bool = False
    # For write tools: state captured when the change is proposed, stored with the approval request.
    snapshot: Callable[[Path], dict[str, Any]] | None = None
    # Runs on every result before it leaves execute_tool; raise to block it (e.g. person-level data).
    output_guard: Callable[[ToolResult], None] | None = None
    _schema: dict[str, Any] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not TOOL_NAME.match(self.name):
            raise ValueError(f"tool name {self.name!r} must be dotted lowercase, e.g. 'rc.benchmark'")
        if len(self.api_name) > MAX_API_NAME:
            raise ValueError(f"tool name {self.name!r} is too long for the Messages API")
        if self.writes and "approval_id" not in self.input_model.model_fields:
            raise ValueError(f"write tool {self.name!r} needs an approval_id input")
        schema = self.input_model.model_json_schema()
        _assert_closed_objects(schema)
        object.__setattr__(self, "_schema", schema)

    @property
    def api_name(self) -> str:
        return self.name.replace(".", "__")

    @property
    def input_schema(self) -> dict[str, Any]:
        return self._schema

    def to_anthropic(self) -> dict[str, Any]:
        return {
            "name": self.api_name,
            "description": self.description,
            "input_schema": self.input_schema,
            "strict": True,
        }


def execute_tool(tool: Tool, raw_input: Mapping[str, Any], *, db_path: Path, agent_run_id: str) -> ToolResult:
    """Validate, run and audit one tool call."""
    tool_call_id = new_id("tc")
    input_json = json.dumps(dict(raw_input), sort_keys=True, default=str)
    started = time.perf_counter()
    with connect(db_path) as conn:
        conn.execute(
            "insert into audit.tool_calls values (?, ?, ?, ?, null, ?, null, 'running')",
            [tool_call_id, agent_run_id, tool.name, input_json, utcnow()],
        )

    def finish(status: str, output_hash: str | None = None) -> None:
        ms = (time.perf_counter() - started) * 1000
        with connect(db_path) as conn:
            conn.execute(
                "update audit.tool_calls set status = ?, output_hash = ?, ms = ? where tool_call_id = ?",
                [status, output_hash, ms, tool_call_id],
            )

    try:
        params = tool.input_model.model_validate(dict(raw_input))
    except ValidationError as exc:
        finish("invalid_input")
        raise ToolInputError(f"{tool.name}: {exc.errors(include_url=False)}") from exc
    if tool.writes:
        # The one gate every caller passes through (agents, API, evals): no approved record, no write.
        try:
            ApprovalQueue(db_path).require_approved(str(raw_input.get("approval_id") or ""), action=tool.name)
        except ApprovalError:
            finish("refused")
            raise
    try:
        result = tool.fn(params, ToolContext(db_path, agent_run_id, tool_call_id))
        if not isinstance(result, ToolResult):
            raise TypeError(f"{tool.name} must return ToolResult, got {type(result).__name__}")
    except Exception:
        finish("error")
        raise
    if tool.output_guard is not None:
        try:
            tool.output_guard(result)
        except Exception as exc:
            # The tool body already recorded lineage for its numbers; a blocked result leaves none behind.
            with connect(db_path) as conn:
                conn.execute("delete from audit.lineage where tool_call_id = ?", [tool_call_id])
            finish("blocked")
            raise OutputBlockedError(f"{tool.name}: result blocked by its output guard ({exc})") from exc
    finish("ok", sha256_text(result.model_dump_json()))
    return result


# --- core tools -----------------------------------------------------------------------------------


class LlmSpendInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    month: str | None = Field(
        default=None,
        pattern=r"^\d{4}-(0[1-9]|1[0-2])$",
        description="Calendar month YYYY-MM; omit for all time",
    )


LLM_SPEND_SQL = """
select coalesce(sum(cost_usd), 0) as cost_usd, count(*) as calls, coalesce(list(call_id), []) as call_ids
from audit.llm_calls
where ?::varchar is null or strftime(created_at, '%Y-%m') = ?
"""


def llm_spend(params: LlmSpendInput, ctx: ToolContext) -> ToolResult:
    with connect(ctx.db_path) as conn:
        row = conn.execute(LLM_SPEND_SQL, [params.month, params.month]).fetchone()
    cost, calls, call_ids = row if row else (0.0, 0, [])
    period = params.month or "all time"
    numbers = [
        ctx.number(
            label=f"LLM spend ({period})",
            value=float(cost),
            unit="USD",
            sql_text=LLM_SPEND_SQL,
            source_tables=["audit.llm_calls"],
            row_ids=call_ids,
        ),
        ctx.number(
            label=f"LLM calls ({period})",
            value=float(calls),
            unit="count",
            sql_text=LLM_SPEND_SQL,
            source_tables=["audit.llm_calls"],
            row_ids=call_ids,
        ),
    ]
    return ToolResult(data={"period": period, "cost_usd": float(cost), "calls": int(calls)}, numbers=numbers)


CORE_TOOLS: tuple[Tool, ...] = (
    Tool(
        name="core.llm_spend",
        description="Total Claude API spend and call count from the audit log, optionally for one month.",
        input_model=LlmSpendInput,
        fn=llm_spend,
    ),
)
