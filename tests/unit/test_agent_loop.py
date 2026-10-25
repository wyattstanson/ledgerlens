from pathlib import Path

import pytest
from fakes import (
    FakeClient,
    answer_citing_last_tool,
    fetch_one,
    final_answer,
    last_tool_numbers,
    text_message,
    tool_use_message,
)
from pydantic import BaseModel, ConfigDict

from ledgerlens.core.agent_loop import MAX_PROPOSALS_PER_RUN, AgentDeps, RunStatus, run_agent
from ledgerlens.core.approvals import ApprovalQueue, ApprovalStatus
from ledgerlens.core.db import connect
from ledgerlens.core.llm import BudgetExceededError, LLMClient, ModelsConfig
from ledgerlens.core.registry import AgentSpec, Mandate, ToolRegistry
from ledgerlens.core.tools import CORE_TOOLS, Tool, ToolContext, ToolResult

SPEND = "core__llm_spend"
executed_writes: list[dict] = []


class WriteIn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    mapping_id: str
    approval_id: str | None = None


def approve_mapping(params: WriteIn, ctx: ToolContext) -> ToolResult:
    executed_writes.append(params.model_dump())
    return ToolResult(data={"approved": params.mapping_id})


WRITE_TOOL = Tool(
    name="rc.approve_mapping", description="d", input_model=WriteIn, fn=approve_mapping, writes=True
)


@pytest.fixture(autouse=True)
def _reset_writes() -> None:
    executed_writes.clear()


def spec(
    mandate: Mandate = Mandate.READ_ONLY, tools: tuple[str, ...] = ("core.llm_spend",), **kw
) -> AgentSpec:
    return AgentSpec(id="test.agent", owner="test", mandate=mandate, tools=tools, **kw)


def deps_for(db: Path, models_config: ModelsConfig, script: list) -> tuple[AgentDeps, FakeClient]:
    fake = FakeClient(script)
    deps = AgentDeps(
        tools=ToolRegistry([*CORE_TOOLS, WRITE_TOOL]),
        llm=LLMClient(models_config, db, client=fake, sleep=lambda _s: None),
        approvals=ApprovalQueue(db),
        db_path=db,
    )
    return deps, fake


def run_status(db: Path, run_id: str) -> str:
    with connect(db) as conn:
        status, finished = fetch_one(
            conn, "select status, finished_at from audit.agent_runs where agent_run_id = ?", [run_id]
        )
    assert finished is not None
    return status


def test_happy_path_cites_tool_numbers_and_passes_lineage(db: Path, models_config: ModelsConfig):
    script = [
        tool_use_message((SPEND, {})),
        answer_citing_last_tool("Spend so far is ${0:.2f} over {1:.0f} calls."),
    ]
    deps, fake = deps_for(db, models_config, script)
    result = run_agent(spec(), "How much have we spent?", user="priya", deps=deps)

    assert result.status is RunStatus.COMPLETED
    assert result.answer is not None
    # The planning call is logged before the tool runs, so the tool sees one call.
    assert result.answer.answer_text == "Spend so far is $0.00 over 1 calls."
    assert result.tool_calls == 1
    assert run_status(db, result.agent_run_id) == "completed"
    # The tool result reached the model fenced as untrusted data.
    tool_result = fake.messages.requests[1]["messages"][2]["content"][0]
    assert tool_result["content"].startswith('<data source="tool:core.llm_spend">')
    with connect(db) as conn:
        assert conn.execute("select count(*) from audit.llm_calls").fetchone() == (2,)


def test_planted_unsupported_number_fails_lineage_after_one_retry(db: Path, models_config: ModelsConfig):
    def invented(request: dict) -> object:
        return final_answer("Spend is $0.00 across 17 vendors.", last_tool_numbers(request))

    script = [tool_use_message((SPEND, {})), invented, invented]
    deps, fake = deps_for(db, models_config, script)
    result = run_agent(spec(), "q", user="priya", deps=deps)

    assert result.status is RunStatus.LINEAGE_FAILED
    assert result.answer is None  # no narration
    assert [n.unit for n in result.numbers] == ["USD", "count"]  # the fallback table
    assert "'17' in the answer does not match any cited number" in result.warnings
    assert run_status(db, result.agent_run_id) == "lineage_failed"
    retry_prompt = fake.messages.requests[2]["messages"][-1]["content"]
    assert "failed the lineage check" in retry_prompt and "'17'" in retry_prompt


def test_one_retry_can_recover(db: Path, models_config: ModelsConfig):
    def invented(request: dict) -> object:
        return final_answer("About 5 dollars.", last_tool_numbers(request))

    script = [tool_use_message((SPEND, {})), invented, answer_citing_last_tool("Spend is ${0:.2f}.")]
    deps, _ = deps_for(db, models_config, script)
    assert run_agent(spec(), "q", user="u", deps=deps).status is RunStatus.COMPLETED


def test_malformed_final_answer_counts_as_a_failed_attempt(db: Path, models_config: ModelsConfig):
    deps, _ = deps_for(db, models_config, [text_message("not json"), text_message("{}")])
    result = run_agent(spec(), "q", user="u", deps=deps)
    assert result.status is RunStatus.LINEAGE_FAILED
    assert any("schema" in w for w in result.warnings)


def test_unlisted_tool_is_refused_and_never_runs(db: Path, models_config: ModelsConfig):
    script = [
        tool_use_message(("rc__approve_mapping", {"mapping_id": "m1"})),
        final_answer("Nothing to report.", []),
    ]
    deps, fake = deps_for(db, models_config, script)
    result = run_agent(spec(), "q", user="u", deps=deps)
    refusal = fake.messages.requests[1]["messages"][2]["content"][0]
    assert refusal["is_error"] is True and "not registered for tool" in refusal["content"]
    assert executed_writes == []
    assert result.status is RunStatus.COMPLETED


def test_unknown_tool_name_is_an_error_result(db: Path, models_config: ModelsConfig):
    script = [tool_use_message(("ghost__tool", {})), final_answer("No data.", [])]
    deps, fake = deps_for(db, models_config, script)
    run_agent(spec(), "q", user="u", deps=deps)
    assert "Unknown tool" in fake.messages.requests[1]["messages"][2]["content"][0]["content"]


def test_propose_agent_creates_an_approval_instead_of_writing(db: Path, models_config: ModelsConfig):
    script = [tool_use_message(("rc__approve_mapping", {"mapping_id": "m1"})), final_answer("Proposed.", [])]
    deps, _ = deps_for(db, models_config, script)
    agent = spec(Mandate.PROPOSE, tools=("rc.approve_mapping",))
    result = run_agent(agent, "q", user="u", deps=deps)
    assert executed_writes == []
    [approval_id] = result.approval_ids
    approval = deps.approvals.get(approval_id)
    assert (approval.action, approval.status, approval.payload) == (
        "rc.approve_mapping",
        ApprovalStatus.PENDING,
        {"arguments": {"mapping_id": "m1"}},
    )


def test_read_only_agent_cannot_use_a_listed_write_tool(db: Path, models_config: ModelsConfig):
    script = [tool_use_message(("rc__approve_mapping", {"mapping_id": "m1"})), final_answer("Done.", [])]
    deps, fake = deps_for(db, models_config, script)
    run_agent(spec(Mandate.READ_ONLY, tools=("rc.approve_mapping",)), "q", user="u", deps=deps)
    assert "read-only" in fake.messages.requests[1]["messages"][2]["content"][0]["content"]
    assert executed_writes == []


def test_write_with_approval_runs_only_against_an_approved_record(db: Path, models_config: ModelsConfig):
    queue = ApprovalQueue(db)
    pending = queue.propose(agent_run_id="run_0", action="rc.approve_mapping", payload={})
    approved = queue.propose(agent_run_id="run_0", action="rc.approve_mapping", payload={})
    queue.decide(approved.approval_id, approve=True, decided_by="priya", reason="checked")
    script = [
        tool_use_message(("rc__approve_mapping", {"mapping_id": "m1", "approval_id": pending.approval_id})),
        tool_use_message(("rc__approve_mapping", {"mapping_id": "m2", "approval_id": approved.approval_id})),
        final_answer("Applied.", []),
    ]
    deps, _ = deps_for(db, models_config, script)
    run_agent(spec(Mandate.WRITE_WITH_APPROVAL, tools=("rc.approve_mapping",)), "q", user="u", deps=deps)
    assert [w["mapping_id"] for w in executed_writes] == ["m2"]


def test_tool_call_budget_stops_the_run(db: Path, models_config: ModelsConfig):
    script = [tool_use_message((SPEND, {})) for _ in range(3)]
    deps, _ = deps_for(db, models_config, script)
    result = run_agent(spec(max_tool_calls=2), "q", user="u", deps=deps)
    assert result.status is RunStatus.TOOL_BUDGET_EXCEEDED
    assert result.answer is None
    assert len(result.numbers) == 4  # two successful calls, two numbers each


def test_invalid_tool_input_is_returned_to_the_model(db: Path, models_config: ModelsConfig):
    script = [tool_use_message((SPEND, {"month": "bad"})), final_answer("Could not read spend.", [])]
    deps, fake = deps_for(db, models_config, script)
    run_agent(spec(), "q", user="u", deps=deps)
    result_block = fake.messages.requests[1]["messages"][2]["content"][0]
    assert result_block["is_error"] is True and result_block["content"].startswith("Invalid input")


def test_proposed_actions_become_approvals_only_for_the_agents_own_write_tools(
    db: Path, models_config: ModelsConfig
):
    good = {"action": "rc.approve_mapping", "payload_json": '{"mapping_id": "m9"}', "rationale": "matches"}
    # Registered and listed for this agent, but a read tool: proposals are only for writes.
    foreign = {"action": "core.llm_spend", "payload_json": "{}", "rationale": "not a write"}
    invalid = {"action": "rc.approve_mapping", "payload_json": '{"wrong": 1}', "rationale": "bad args"}
    deps, _ = deps_for(db, models_config, [final_answer("See proposals.", [], [good, foreign, invalid])])
    agent = spec(Mandate.PROPOSE, tools=("core.llm_spend", "rc.approve_mapping"))
    result = run_agent(agent, "q", user="u", deps=deps)
    [approval_id] = result.approval_ids
    assert deps.approvals.get(approval_id).payload == {
        "arguments": {"mapping_id": "m9"},
        "rationale": "matches",
    }
    assert any("not a write tool of this agent" in w for w in result.warnings)
    assert any("invalid arguments" in w for w in result.warnings)

    deps, _ = deps_for(db, models_config, [final_answer("See proposal.", [], [good])])
    read_only = run_agent(spec(Mandate.READ_ONLY), "q", user="u", deps=deps)
    assert read_only.approval_ids == []
    assert "discarded" in read_only.warnings[0]


def test_one_run_cannot_flood_the_review_queue(db: Path, models_config: ModelsConfig):
    proposal = {"action": "rc.approve_mapping", "payload_json": '{"mapping_id": "m"}', "rationale": "r"}
    deps, _ = deps_for(db, models_config, [final_answer("Many.", [], [proposal] * 8)])
    agent = spec(Mandate.PROPOSE, tools=("rc.approve_mapping",))
    result = run_agent(agent, "q", user="u", deps=deps)
    assert len(result.approval_ids) == MAX_PROPOSALS_PER_RUN
    assert any("already made" in w for w in result.warnings)


def test_agent_with_unregistered_tools_refuses_to_start(db: Path, models_config: ModelsConfig):
    deps, _ = deps_for(db, models_config, [])
    with pytest.raises(ValueError, match=r"rc\.benchmark"):
        run_agent(spec(tools=("rc.benchmark",)), "q", user="u", deps=deps)


def test_budget_exhaustion_ends_the_run_cleanly(db: Path, models_config: ModelsConfig, monkeypatch):
    deps, _ = deps_for(db, models_config, [])

    def broke(**_kwargs):
        raise BudgetExceededError("cap reached")

    monkeypatch.setattr(deps.llm, "create", broke)
    result = run_agent(spec(), "q", user="u", deps=deps)
    assert result.status is RunStatus.BUDGET_EXCEEDED
    assert run_status(db, result.agent_run_id) == "budget_exceeded"


def test_non_end_turn_stop_is_reported(db: Path, models_config: ModelsConfig):
    deps, _ = deps_for(db, models_config, [text_message("{", stop_reason="max_tokens")])
    result = run_agent(spec(), "q", user="u", deps=deps)
    assert result.status is RunStatus.STOPPED
    assert "max_tokens" in result.warnings[0]


def test_unexpected_errors_mark_the_run_and_propagate(db: Path, models_config: ModelsConfig):
    deps, _ = deps_for(db, models_config, [RuntimeError("network down")])
    with pytest.raises(RuntimeError, match="network down"):
        run_agent(spec(), "q", user="u", deps=deps)
    with connect(db) as conn:
        assert conn.execute("select status from audit.agent_runs").fetchone() == ("error",)


def test_proposals_with_invalid_arguments_never_reach_the_review_queue(db: Path, models_config: ModelsConfig):
    script = [
        tool_use_message(("rc__approve_mapping", {"mapping_id": 7, "extra": "x"})),
        final_answer("No.", []),
    ]
    deps, fake = deps_for(db, models_config, script)
    result = run_agent(spec(Mandate.PROPOSE, tools=("rc.approve_mapping",)), "q", user="u", deps=deps)
    assert result.approval_ids == []
    assert deps.approvals.pending() == []
    assert "Invalid input" in fake.messages.requests[1]["messages"][2]["content"][0]["content"]
