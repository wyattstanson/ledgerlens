"""MCP server exposing LedgerLens's read-only governed tools to any MCP client.

The same ``execute_tool`` path as the agents, so every call is audited and every number carries a
lineage id. Write tools are never exposed here: changes go through the approval queue.

Run with ``make mcp`` (stdio transport).
"""

import inspect
from pathlib import Path
from typing import Any

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

from ledgerlens.core.db import connect, ensure_audit_schema, utcnow
from ledgerlens.core.lineage import new_id
from ledgerlens.core.registry import ToolRegistry
from ledgerlens.core.tools import Tool, ToolInputError, execute_tool
from ledgerlens.settings import get_settings

MCP_AGENT_ID = "mcp.client"

INSTRUCTIONS = (
    "Governed LedgerLens analytics tools. Every number in a result carries a lineage_id that traces to "
    "the SQL and rows behind it; quote numbers only from results. Tools are read-only."
)


def _handler_for(tool: Tool, db_path: Path) -> Any:
    def handler(**kwargs: Any) -> dict[str, Any]:
        run_id = new_id("run")
        with connect(db_path) as conn:
            conn.execute(
                "insert into audit.agent_runs values (?, ?, 'mcp', ?, null, 'running')",
                [run_id, MCP_AGENT_ID, utcnow()],
            )
        status = "error"
        try:
            # Drop arguments the client left unset so the input model applies its own defaults.
            result = execute_tool(
                tool, {k: v for k, v in kwargs.items() if v is not None}, db_path=db_path, agent_run_id=run_id
            )
            status = "completed"
            return result.model_dump(mode="json")
        except ToolInputError as exc:
            raise ToolError(str(exc)) from exc
        finally:
            with connect(db_path) as conn:
                conn.execute(
                    "update audit.agent_runs set finished_at = ?, status = ? where agent_run_id = ?",
                    [utcnow(), status, run_id],
                )

    # Present the input model's fields as keyword parameters so MCP clients see a flat schema.
    params = [
        inspect.Parameter(
            name,
            inspect.Parameter.KEYWORD_ONLY,
            annotation=info.annotation,
            default=inspect.Parameter.empty if info.is_required() else info.default,
        )
        for name, info in tool.input_model.model_fields.items()
    ]
    handler.__signature__ = inspect.Signature(params, return_annotation=dict[str, Any])  # type: ignore[attr-defined]
    handler.__annotations__ = {p.name: p.annotation for p in params} | {"return": dict[str, Any]}
    handler.__name__ = tool.api_name
    return handler


def build_server(tools: ToolRegistry, db_path: Path) -> MCPServer:
    server = MCPServer(name="ledgerlens", instructions=INSTRUCTIONS)
    for tool in tools:
        if tool.writes:
            continue
        server.add_tool(
            _handler_for(tool, db_path), name=tool.name, description=tool.description, structured_output=True
        )
    return server


def main() -> None:
    settings = get_settings()
    ensure_audit_schema(settings.db)
    from ledgerlens.packs import default_tool_registry  # noqa: PLC0415 (composition root)

    build_server(default_tool_registry(), settings.db).run(transport="stdio")


if __name__ == "__main__":
    main()
