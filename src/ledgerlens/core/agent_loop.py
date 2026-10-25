"""Plan/act loop with mandate checks, approval proposals and the lineage check (sections 9.1 and 9.4).

The model plans tool calls and writes the narrative. Tools produce every number. A final answer is
accepted only if every number in its text traces to lineage recorded during this run; otherwise the
loop retries once, then falls back to the raw tool numbers without narration.
"""

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, ValidationError

from ledgerlens.core.approvals import ApprovalError, ApprovalQueue
from ledgerlens.core.db import connect, utcnow
from ledgerlens.core.guards import fence_untrusted
from ledgerlens.core.lineage import NumberRef, lineage_values_for_run, new_id, verify_answer
from ledgerlens.core.llm import BudgetExceededError, LLMClient
from ledgerlens.core.registry import (
    AgentSpec,
    Decision,
    Mandate,
    ToolNotPermittedError,
    ToolRegistry,
    authorize,
    missing_tools,
)
from ledgerlens.core.tools import Tool, ToolInputError, execute_tool

MAX_STRICT_TOOLS = 20  # Messages API limit on strict tools per request
MAX_PROPOSALS_PER_RUN = 5  # keeps one run from flooding the human review queue

AGENT_SYSTEM = """You are a LedgerLens analyst agent. You plan tool calls and explain their results.

Rules:
- You never compute numbers yourself. Every number in answer_text must be copied from a tool result,
  and listed in numbers[] with the exact value, unit and lineage_id the tool returned.
- Write counts up to ten in words ("three vendors"), write every larger number in digits, and keep
  dates and periods out of answer_text, because every number in answer_text is checked against
  numbers[]: a percentage must cite a ratio or share, and a dollar amount must cite a USD value.
- If the tools cannot answer the question, say so plainly instead of estimating.
- State assumptions in assumptions[]. Put any change you recommend in proposed_actions[]; you cannot
  make changes yourself, a human approves or rejects each proposal.
- Your final reply is the JSON object described by the output schema, with no other text."""


class ProposedAction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action: str
    payload_json: str
    rationale: str


class FinalAnswer(BaseModel):
    model_config = ConfigDict(extra="forbid")

    answer_text: str
    numbers: list[NumberRef]
    assumptions: list[str]
    proposed_actions: list[ProposedAction]


class RunStatus(StrEnum):
    COMPLETED = "completed"
    LINEAGE_FAILED = "lineage_failed"
    TOOL_BUDGET_EXCEEDED = "tool_budget_exceeded"
    BUDGET_EXCEEDED = "budget_exceeded"
    STOPPED = "stopped"
    ERROR = "error"


@dataclass
class AgentRunResult:
    agent_run_id: str
    status: RunStatus
    answer: FinalAnswer | None
    numbers: list[NumberRef] = field(default_factory=list)  # every tool number, the fallback table
    approval_ids: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    tool_calls: int = 0


@dataclass
class AgentDeps:
    tools: ToolRegistry
    llm: LLMClient
    approvals: ApprovalQueue
    db_path: Path


def _tool_result(tool_use_id: str, content: str, *, is_error: bool = False) -> dict[str, Any]:
    return {"type": "tool_result", "tool_use_id": tool_use_id, "content": content, "is_error": is_error}


def _first_text(content: Sequence[Any]) -> str:
    return next((block.text for block in content if block.type == "text"), "")


class _Run:
    def __init__(self, agent: AgentSpec, deps: AgentDeps, agent_run_id: str) -> None:
        self.agent = agent
        self.deps = deps
        self.run_id = agent_run_id
        self.result = AgentRunResult(agent_run_id=agent_run_id, status=RunStatus.STOPPED, answer=None)

    def call_tool(self, block: Any) -> dict[str, Any]:  # noqa: PLR0911 (one exit per refusal)
        tool: Tool | None = self.deps.tools.by_api_name(block.name)
        if tool is None:
            return _tool_result(block.id, f"Unknown tool {block.name}.", is_error=True)
        try:
            decision = authorize(self.agent, tool)
        except ToolNotPermittedError as exc:
            return _tool_result(block.id, f"Refused: {exc}.", is_error=True)

        raw_input = dict(block.input)
        if decision is Decision.PROPOSE:
            # Validate here too: strict mode is enforced by the API, and a reviewer should only ever see
            # arguments the tool would accept.
            try:
                tool.input_model.model_validate(raw_input)
            except ValidationError as exc:
                return _tool_result(
                    block.id, f"Invalid input: {exc.errors(include_url=False)}", is_error=True
                )
            return _tool_result(block.id, self._propose(tool, raw_input))
        if decision is Decision.EXECUTE_IF_APPROVED:
            try:
                self.deps.approvals.require_approved(str(raw_input.get("approval_id", "")), action=tool.name)
            except ApprovalError as exc:
                return _tool_result(block.id, f"Refused: {exc}.", is_error=True)

        try:
            output = execute_tool(tool, raw_input, db_path=self.deps.db_path, agent_run_id=self.run_id)
        except ToolInputError as exc:
            return _tool_result(block.id, f"Invalid input: {exc}", is_error=True)
        self.result.numbers.extend(output.numbers)
        return _tool_result(block.id, fence_untrusted(output.model_dump_json(), source=f"tool:{tool.name}"))

    def _propose(self, tool: Tool, arguments: dict[str, Any], rationale: str | None = None) -> str:
        """Create an approval request that records exactly what would change.

        The tool's snapshot (for example the panel's vendors and a hash) is stored with the request, so a
        human approves concrete state and the write can later refuse if that state has moved.
        """
        if len(self.result.approval_ids) >= MAX_PROPOSALS_PER_RUN:
            return f"Not proposed: this run already made {MAX_PROPOSALS_PER_RUN} proposals."
        payload: dict[str, Any] = {"arguments": arguments}
        if tool.snapshot is not None:
            payload["snapshot"] = tool.snapshot(self.deps.db_path)
        if rationale is not None:
            payload["rationale"] = rationale
        approval = self.deps.approvals.propose(agent_run_id=self.run_id, action=tool.name, payload=payload)
        self.result.approval_ids.append(approval.approval_id)
        return f"Not executed. Proposed for human approval as {approval.approval_id}."

    def accept(self, answer: FinalAnswer) -> None:
        self.result.answer = answer
        self.result.status = RunStatus.COMPLETED
        if not answer.proposed_actions:
            return
        if self.agent.mandate is Mandate.READ_ONLY:
            self.result.warnings.append("read-only agent proposed actions; they were discarded")
            return
        for proposal in answer.proposed_actions:
            # Only write tools this agent is registered for can be proposed, with arguments that validate.
            tool = self.deps.tools.get(proposal.action) if proposal.action in self.deps.tools else None
            if tool is None or not tool.writes or tool.name not in self.agent.tools:
                self.result.warnings.append(
                    f"proposal for {proposal.action!r} discarded: not a write tool of this agent"
                )
                continue
            try:
                arguments = json.loads(proposal.payload_json)
                tool.input_model.model_validate(arguments)
            except (json.JSONDecodeError, ValidationError, TypeError) as exc:
                self.result.warnings.append(
                    f"proposal for {proposal.action!r} discarded: invalid arguments ({exc})"
                )
                continue
            note = self._propose(tool, arguments, rationale=proposal.rationale)
            if note.startswith("Not proposed"):
                self.result.warnings.append(note)
                break


def run_agent(agent: AgentSpec, question: str, *, user: str, deps: AgentDeps) -> AgentRunResult:
    missing = missing_tools(agent, deps.tools)
    if missing:
        raise ValueError(f"agent {agent.id} cannot run: tools not registered yet: {missing}")
    tools = [deps.tools.get(name) for name in agent.tools]
    if len(tools) > MAX_STRICT_TOOLS:
        raise ValueError(f"agent {agent.id} exposes {len(tools)} tools; the API allows {MAX_STRICT_TOOLS}")

    run_id = new_id("run")
    with connect(deps.db_path) as conn:
        conn.execute(
            "insert into audit.agent_runs values (?, ?, ?, ?, null, 'running')",
            [run_id, agent.id, user, utcnow()],
        )
    run = _Run(agent, deps, run_id)
    try:
        _loop(run, question, [t.to_anthropic() for t in tools])
    except BudgetExceededError as exc:
        run.result.status = RunStatus.BUDGET_EXCEEDED
        run.result.warnings.append(str(exc))
    except Exception:
        run.result.status = RunStatus.ERROR
        raise
    finally:
        with connect(deps.db_path) as conn:
            conn.execute(
                "update audit.agent_runs set finished_at = ?, status = ? where agent_run_id = ?",
                [utcnow(), run.result.status.value, run_id],
            )
    if run.result.status is not RunStatus.COMPLETED:
        run.result.answer = None  # fall back to the tool numbers, without narration
    return run.result


def _loop(run: _Run, question: str, tool_defs: list[dict[str, Any]]) -> None:
    messages: list[dict[str, Any]] = [{"role": "user", "content": question}]
    schema = FinalAnswer.model_json_schema()
    retried = False
    max_turns = run.agent.max_tool_calls + 3
    brief = run.agent.brief.strip()
    system = f"{AGENT_SYSTEM}\n\n{brief}" if brief else AGENT_SYSTEM
    for _ in range(max_turns):
        response = run.deps.llm.create(
            task=run.agent.model_task,
            system=system,
            messages=messages,
            tools=tool_defs,
            output_schema=schema,
            agent_run_id=run.run_id,
        )
        message = response.message
        messages.append({"role": "assistant", "content": message.content})

        if message.stop_reason == "tool_use":
            results = []
            for block in (b for b in message.content if b.type == "tool_use"):
                run.result.tool_calls += 1
                if run.result.tool_calls > run.agent.max_tool_calls:
                    run.result.status = RunStatus.TOOL_BUDGET_EXCEEDED
                    return
                results.append(run.call_tool(block))
            messages.append({"role": "user", "content": results})
            continue

        if message.stop_reason != "end_turn":
            run.result.status = RunStatus.STOPPED
            run.result.warnings.append(f"model stopped with stop_reason={message.stop_reason}")
            return

        answer: FinalAnswer | None = None
        try:
            answer = FinalAnswer.model_validate_json(_first_text(message.content))
            failures = verify_answer(
                answer.answer_text, answer.numbers, lineage_values_for_run(run.deps.db_path, run.run_id)
            )
        except ValidationError as exc:
            failures = [f"final answer does not match the schema: {exc.error_count()} errors"]
        if answer is not None and not failures:
            run.accept(answer)
            return
        run.result.warnings.extend(failures)
        if retried:
            run.result.status = RunStatus.LINEAGE_FAILED
            return
        retried = True
        listed = "\n".join(f"- {f}" for f in failures)
        messages.append(
            {
                "role": "user",
                "content": f"Your answer failed the lineage check:\n{listed}\n"
                "Rewrite it so every number in answer_text is listed in numbers[] with the value and "
                "lineage_id a tool returned. Reply with the final answer JSON only.",
            }
        )
    run.result.status = RunStatus.TOOL_BUDGET_EXCEEDED
