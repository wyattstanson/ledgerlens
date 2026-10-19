"""Agent and tool registry with mandates (BUILD_PROMPT section 9.1).

Like card-network "Know Your Agent" rules: every agent is registered, scoped to a tool list and bound
to a mandate. The loop asks ``authorize`` before every tool call.
"""

from collections.abc import Iterable, Iterator
from enum import StrEnum
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field

from ledgerlens.core.tools import Tool


class Mandate(StrEnum):
    READ_ONLY = "read_only"
    PROPOSE = "propose"
    WRITE_WITH_APPROVAL = "write_with_approval"


class Decision(StrEnum):
    EXECUTE = "execute"  # run the tool now
    PROPOSE = "propose"  # create an approval request instead of running it
    EXECUTE_IF_APPROVED = "execute_if_approved"  # run only against an approved approval record


class ToolNotPermittedError(PermissionError):
    pass


class AgentSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str
    owner: str
    model_task: str = "pack_agent"
    mandate: Mandate
    tools: tuple[str, ...]
    max_tool_calls: int = Field(default=8, ge=1, le=50)


class ToolRegistry:
    def __init__(self, tools: Iterable[Tool] = ()) -> None:
        self._by_name: dict[str, Tool] = {}
        self._by_api: dict[str, Tool] = {}
        for tool in tools:
            self.register(tool)

    def register(self, tool: Tool) -> None:
        if tool.name in self._by_name or tool.api_name in self._by_api:
            raise ValueError(f"tool {tool.name!r} is already registered")
        self._by_name[tool.name] = tool
        self._by_api[tool.api_name] = tool

    def get(self, name: str) -> Tool:
        return self._by_name[name]

    def by_api_name(self, api_name: str) -> Tool | None:
        return self._by_api.get(api_name)

    def __contains__(self, name: object) -> bool:
        return name in self._by_name

    def __iter__(self) -> Iterator[Tool]:
        return iter(self._by_name.values())


class AgentRegistry:
    def __init__(self, agents: Iterable[AgentSpec]) -> None:
        self._agents: dict[str, AgentSpec] = {}
        for agent in agents:
            if agent.id in self._agents:
                raise ValueError(f"agent {agent.id!r} is registered twice")
            self._agents[agent.id] = agent

    @classmethod
    def from_yaml(cls, path: Path) -> "AgentRegistry":
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        return cls(AgentSpec.model_validate(item) for item in raw.get("agents", []))

    def get(self, agent_id: str) -> AgentSpec:
        try:
            return self._agents[agent_id]
        except KeyError:
            raise KeyError(f"agent {agent_id!r} is not registered") from None

    def __iter__(self) -> Iterator[AgentSpec]:
        return iter(self._agents.values())


def missing_tools(agent: AgentSpec, tools: ToolRegistry) -> list[str]:
    """Tools the agent is mandated to use that no pack has registered yet."""
    return [name for name in agent.tools if name not in tools]


def authorize(agent: AgentSpec, tool: Tool) -> Decision:
    """Decide how (or whether) an agent may use a tool. Raises if it may not."""
    if tool.name not in agent.tools:
        raise ToolNotPermittedError(f"agent {agent.id} is not registered for tool {tool.name}")
    if not tool.writes:
        return Decision.EXECUTE
    match agent.mandate:
        case Mandate.READ_ONLY:
            raise ToolNotPermittedError(
                f"agent {agent.id} is read-only and cannot use write tool {tool.name}"
            )
        case Mandate.PROPOSE:
            return Decision.PROPOSE
        case Mandate.WRITE_WITH_APPROVAL:
            return Decision.EXECUTE_IF_APPROVED
