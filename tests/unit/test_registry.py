from pathlib import Path

import pytest
from pydantic import BaseModel, ConfigDict

from ledgerlens.core.registry import (
    AgentRegistry,
    AgentSpec,
    Decision,
    Mandate,
    ToolNotPermittedError,
    ToolRegistry,
    authorize,
    default_tool_registry,
    missing_tools,
)
from ledgerlens.core.tools import Tool, ToolContext, ToolResult

REPO_ROOT = Path(__file__).resolve().parents[2]


class In(BaseModel):
    model_config = ConfigDict(extra="forbid")


def noop(params: In, ctx: ToolContext) -> ToolResult:
    return ToolResult(data=None)


READ = Tool(name="t.read", description="d", input_model=In, fn=noop)
WRITE = Tool(name="t.write", description="d", input_model=In, fn=noop, writes=True)


def agent(mandate: Mandate, tools: tuple[str, ...] = ("t.read", "t.write")) -> AgentSpec:
    return AgentSpec(id="a", owner="o", mandate=mandate, tools=tools)


def test_registry_loads_the_shipped_agents_yaml():
    agents = AgentRegistry.from_yaml(REPO_ROOT / "config" / "agents.yaml")
    spec = agents.get("ratecards.panel")
    assert spec.mandate is Mandate.PROPOSE
    assert spec.tools == ("rc.benchmark", "rc.optimize_panel", "rc.savings_bridge")


def test_unknown_agent_and_duplicate_agents_fail_loudly():
    with pytest.raises(KeyError, match="not registered"):
        AgentRegistry([]).get("nope")
    with pytest.raises(ValueError, match="twice"):
        AgentRegistry([agent(Mandate.READ_ONLY), agent(Mandate.PROPOSE)])


def test_unknown_yaml_keys_are_rejected(tmp_path: Path):
    bad = tmp_path / "agents.yaml"
    bad.write_text("agents:\n  - {id: a, owner: o, mandate: read_only, tools: [], can_write: true}\n")
    with pytest.raises(ValueError, match="can_write"):
        AgentRegistry.from_yaml(bad)


def test_unlisted_tool_is_refused_for_every_mandate():
    for mandate in Mandate:
        with pytest.raises(ToolNotPermittedError, match="not registered for tool"):
            authorize(agent(mandate, tools=()), READ)


@pytest.mark.parametrize(
    ("mandate", "expected"),
    [(Mandate.PROPOSE, Decision.PROPOSE), (Mandate.WRITE_WITH_APPROVAL, Decision.EXECUTE_IF_APPROVED)],
)
def test_write_tools_never_execute_directly(mandate, expected):
    assert authorize(agent(mandate), WRITE) is expected
    assert authorize(agent(mandate), READ) is Decision.EXECUTE


def test_read_only_agents_cannot_touch_write_tools():
    with pytest.raises(ToolNotPermittedError, match="read-only"):
        authorize(agent(Mandate.READ_ONLY), WRITE)


def test_missing_tools_lists_unregistered_names():
    registry = ToolRegistry([READ])
    assert missing_tools(agent(Mandate.PROPOSE), registry) == ["t.write"]


def test_duplicate_tool_registration_fails():
    with pytest.raises(ValueError, match="already registered"):
        ToolRegistry([READ, READ])


def test_default_registry_ships_core_tools_by_api_name():
    registry = default_tool_registry()
    assert "core.llm_spend" in registry
    assert registry.by_api_name("core__llm_spend") is registry.get("core.llm_spend")
    assert registry.by_api_name("nope") is None
