"""In-process MCP test (Phase 1 exit criterion): a real MCP client calls the governed tools."""

import asyncio
from pathlib import Path
from typing import Any

from mcp import Client
from pydantic import BaseModel, ConfigDict

from ledgerlens.core.db import connect
from ledgerlens.core.registry import ToolRegistry
from ledgerlens.core.tools import CORE_TOOLS, Tool, ToolContext, ToolResult
from ledgerlens.mcp.server import build_server


class In(BaseModel):
    model_config = ConfigDict(extra="forbid")


def never(params: In, ctx: ToolContext) -> ToolResult:
    raise AssertionError("write tools must not be reachable over MCP")


WRITE = Tool(name="rc.accept_panel", description="d", input_model=In, fn=never, writes=True)


def run(coro_fn) -> Any:
    return asyncio.run(coro_fn())


def test_mcp_lists_read_only_tools_with_flat_schemas(db: Path):
    server = build_server(ToolRegistry([*CORE_TOOLS, WRITE]), db)

    async def go():
        async with Client(server) as client:
            return (await client.list_tools()).tools

    tools = {t.name: t for t in run(go)}
    assert set(tools) == {"core.llm_spend"}
    assert "month" in tools["core.llm_spend"].input_schema["properties"]  # MCP SDK v2 is snake_case


def test_mcp_call_returns_numbers_with_lineage_in_the_audit_log(db: Path):
    with connect(db) as conn:
        conn.execute(
            "insert into audit.llm_calls values ('llm_a', null, 't', 'm', 'low', 1, 1, 0, 0, 0.75, 1, "
            "'2026-10-02 00:00:00')"
        )
    server = build_server(ToolRegistry(CORE_TOOLS), db)

    async def go():
        async with Client(server) as client:
            return await client.call_tool("core.llm_spend", {"month": "2026-10"})

    result = run(go)
    assert not result.is_error
    numbers = result.structured_content["numbers"]
    assert numbers[0]["value"] == 0.75
    with connect(db) as conn:
        recorded = conn.execute(
            "select l.value, r.agent_id, r.status from audit.lineage l "
            "join audit.tool_calls t using (tool_call_id) join audit.agent_runs r using (agent_run_id) "
            "where l.lineage_id = ?",
            [numbers[0]["lineage_id"]],
        ).fetchone()
    assert recorded == (0.75, "mcp.client", "completed")


def test_mcp_optional_arguments_can_be_omitted(db: Path):
    server = build_server(ToolRegistry(CORE_TOOLS), db)

    async def go():
        async with Client(server) as client:
            return await client.call_tool("core.llm_spend", {})

    result = run(go)
    assert not result.is_error
    assert result.structured_content["data"]["period"] == "all time"


def test_mcp_invalid_input_is_a_tool_error(db: Path):
    server = build_server(ToolRegistry(CORE_TOOLS), db)

    async def go():
        async with Client(server) as client:
            return await client.call_tool("core.llm_spend", {"month": "October"})

    assert run(go).is_error
