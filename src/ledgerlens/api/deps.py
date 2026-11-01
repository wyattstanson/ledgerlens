"""Shared API plumbing: the tool registry, the run row every call gets, and error translation.

Every API read goes through ``execute_tool``, exactly as the agents and the MCP server do, so one
calculation has one implementation and every number it returns carries a lineage id.
"""

import threading
from collections.abc import Generator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from fastapi import HTTPException, status

from ledgerlens.core.db import connect, utcnow
from ledgerlens.core.lineage import new_id
from ledgerlens.core.registry import ToolRegistry
from ledgerlens.core.tools import OutputBlockedError, Tool, ToolInputError, ToolResult, execute_tool
from ledgerlens.packs import default_tool_registry
from ledgerlens.settings import Settings, get_settings

API_ACTOR = "api.ui"

# One warehouse file, two connection shapes: tools read it with external access off, while the audit
# writer opens it plainly. DuckDB refuses both at once in one process, and FastAPI runs sync endpoints
# on a threadpool, so a page that loads six panels would otherwise race itself and fail at random.
# Serialising costs latency on a page that asks for several panels, which is the trade: this is a
# single-warehouse analyst tool, so correctness is worth more than overlapping two reads.
_WAREHOUSE = threading.Lock()


@contextmanager
def warehouse() -> Generator[None]:
    """Hold the warehouse for one request's worth of reads and audit writes."""
    with _WAREHOUSE:
        yield


def registry() -> ToolRegistry:
    return default_tool_registry()


def settings() -> Settings:
    return get_settings()


def readable_tool(tools: ToolRegistry, name: str) -> Tool:
    """The named read-only tool, or a 404/403. Write tools are never reachable over HTTP."""
    try:
        tool = tools.get(name)
    except KeyError:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"no tool named {name!r}") from None
    if tool.writes:
        raise HTTPException(
            status.HTTP_403_FORBIDDEN,
            f"{name} changes state, so it runs only through an approved record, never over HTTP",
        )
    return tool


@contextmanager
def api_run(db_path: Path, actor: str = API_ACTOR) -> Generator[str]:
    """One audit.agent_runs row per API call, closed with the call's outcome."""
    run_id = new_id("run")
    with connect(db_path) as conn:
        conn.execute(
            "insert into audit.agent_runs values (?, ?, 'api', ?, null, 'running')",
            [run_id, actor, utcnow()],
        )
    status_text = "error"
    try:
        yield run_id
        status_text = "completed"
    finally:
        with connect(db_path) as conn:
            conn.execute(
                "update audit.agent_runs set finished_at = ?, status = ? where agent_run_id = ?",
                [utcnow(), status_text, run_id],
            )


def run_tool(tool: Tool, params: Mapping[str, Any], db_path: Path) -> ToolResult:
    """Execute one read-only tool, translating its refusals into HTTP status codes."""
    try:
        with warehouse(), api_run(db_path) as run_id:
            return execute_tool(tool, params, db_path=db_path, agent_run_id=run_id)
    except ToolInputError as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(exc)) from exc
    except OutputBlockedError as exc:
        # A guard refused the result (for example person-level interview data). Say so without the detail.
        raise HTTPException(status.HTTP_403_FORBIDDEN, str(exc)) from exc
    except RuntimeError as exc:
        # The pack's pipeline has not run on the loaded warehouse yet.
        raise HTTPException(status.HTTP_409_CONFLICT, str(exc)) from exc
