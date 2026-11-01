"""Tool routes: the read-only governed tools, as HTTP.

``GET /api/tools`` is the catalogue the UI builds its forms from; ``POST /api/tools/{name}`` runs one.
A tool's answer is returned verbatim (``data`` plus ``numbers``), so a page can show a number next to
the lineage id that proves where it came from.
"""

from typing import Annotated, Any

from fastapi import APIRouter, Body, Depends

from ledgerlens.api.deps import readable_tool, registry, run_tool, settings
from ledgerlens.core.registry import ToolRegistry
from ledgerlens.settings import Settings

router = APIRouter(prefix="/api/tools", tags=["tools"])

Tools = Annotated[ToolRegistry, Depends(registry)]
Config = Annotated[Settings, Depends(settings)]


@router.get("")
def list_tools(tools: Tools) -> dict[str, list[dict[str, Any]]]:
    """Every read-only tool, with the JSON schema of its inputs."""
    return {
        "tools": [
            {"name": t.name, "description": t.description, "input_schema": t.input_schema}
            for t in tools
            if not t.writes
        ]
    }


@router.post("/{name}")
def call_tool(
    name: str,
    tools: Tools,
    config: Config,
    params: Annotated[dict[str, Any], Body(default_factory=dict)],
) -> dict[str, Any]:
    """Run one read-only tool and return its data and its lineage-backed numbers."""
    tool = readable_tool(tools, name)
    result = run_tool(tool, params, config.db)
    return result.model_dump(mode="json")
