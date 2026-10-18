"""Approval hardening from the Phase 2 review: batch decisions, decider kinds, single use, the write gate."""

from pathlib import Path

import pytest
from pydantic import BaseModel, ConfigDict

from ledgerlens.core.approvals import ApprovalError, ApprovalQueue, ApprovalStatus, DecisionKind
from ledgerlens.core.db import connect
from ledgerlens.core.tools import Tool, ToolContext, ToolResult, execute_tool


def _two(approvals: ApprovalQueue, action: str = "a") -> tuple[str, str]:
    first = approvals.propose(agent_run_id="run_1", action=action, payload={}).approval_id
    second = approvals.propose(agent_run_id="run_1", action=action, payload={}).approval_id
    return first, second


def test_decide_many_is_all_or_nothing_on_action_mismatch(approvals: ApprovalQueue):
    first, _ = _two(approvals)
    other = approvals.propose(agent_run_id="run_1", action="b", payload={}).approval_id
    with pytest.raises(ApprovalError, match="covers 'b', not 'a'"):
        approvals.decide_many({first: "ok", other: "ok"}, approve=True, decided_by="priya", action="a")
    assert approvals.get(first).status is ApprovalStatus.PENDING  # rolled back


def test_decide_many_rolls_back_when_the_dependent_write_fails(approvals: ApprovalQueue):
    first, second = _two(approvals)

    def failing_write(conn) -> None:
        raise ValueError("title map update failed")

    with pytest.raises(ValueError, match="title map update failed"):
        approvals.decide_many(
            {first: "ok", second: "ok"},
            approve=True,
            decided_by="priya",
            action="a",
            in_transaction=failing_write,
        )
    assert [approvals.get(i).status for i in (first, second)] == [ApprovalStatus.PENDING] * 2


def test_decide_many_with_nothing_to_decide_is_a_no_op(approvals: ApprovalQueue):
    approvals.decide_many({}, approve=True, decided_by="priya", action="a")


@pytest.mark.parametrize(
    "who",
    ["pipeline.ratecards", "MCP.Client", "agent：ratecards.panel", "eval:offline"],  # noqa: RUF001 (deliberate look-alike colon)
)
def test_system_actors_and_lookalike_prefixes_cannot_decide(approvals: ApprovalQueue, who: str):
    first, _ = _two(approvals)
    with pytest.raises(ApprovalError, match="named human"):
        approvals.decide(first, approve=True, decided_by=who, reason="r")


def test_simulated_decisions_are_labelled_and_kept_apart(approvals: ApprovalQueue):
    first, second = _two(approvals)
    with pytest.raises(ApprovalError, match="simulated"):
        approvals.decide(first, approve=True, decided_by="simulated:truth", reason="r")  # human kind
    with pytest.raises(ApprovalError, match="simulated"):
        approvals.decide(first, approve=True, decided_by="priya", reason="r", kind=DecisionKind.SIMULATED)
    approvals.decide(
        first, approve=True, decided_by="simulated:truth", reason="r", kind=DecisionKind.SIMULATED
    )
    assert approvals.get(first).decided_kind is DecisionKind.SIMULATED
    with pytest.raises(ApprovalError, match="simulated reviewer"):
        approvals.require_approved(first, action="a")
    assert approvals.require_approved(first, action="a", allow_simulated=True).approval_id == first
    approvals.decide(second, approve=True, decided_by="priya", reason="r")
    assert approvals.get(second).decided_kind is DecisionKind.HUMAN


def test_an_approval_is_consumed_exactly_once(approvals: ApprovalQueue):
    first, _ = _two(approvals)
    approvals.decide(first, approve=True, decided_by="priya", reason="r")
    approvals.mark_executed(first)
    assert approvals.get(first).status is ApprovalStatus.EXECUTED
    with pytest.raises(ApprovalError, match="not an unused approved record"):
        approvals.mark_executed(first)
    with pytest.raises(ApprovalError, match="executed, not approved"):
        approvals.require_approved(first, action="a")


class WriteIn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    approval_id: str | None = None


class NoApprovalIn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    value: int


writes_done: list[str] = []


def do_write(params: WriteIn, ctx: ToolContext) -> ToolResult:
    writes_done.append(str(params.approval_id))
    return ToolResult(data=None)


WRITE = Tool(name="t.write", description="d", input_model=WriteIn, fn=do_write, writes=True)


def test_write_tools_must_take_an_approval_id():
    with pytest.raises(ValueError, match="approval_id"):
        Tool(name="t.bad", description="d", input_model=NoApprovalIn, fn=do_write, writes=True)


def test_execute_tool_refuses_writes_without_a_human_approval(db: Path, approvals: ApprovalQueue):
    writes_done.clear()
    with pytest.raises(ApprovalError, match="does not exist"):
        execute_tool(WRITE, {"approval_id": "ap_forged"}, db_path=db, agent_run_id="run_1")
    pending = approvals.propose(agent_run_id="run_1", action="t.write", payload={}).approval_id
    with pytest.raises(ApprovalError, match="pending, not approved"):
        execute_tool(WRITE, {"approval_id": pending}, db_path=db, agent_run_id="run_1")
    approvals.decide(pending, approve=True, decided_by="priya", reason="ok")
    execute_tool(WRITE, {"approval_id": pending}, db_path=db, agent_run_id="run_1")
    assert writes_done == [pending]
    with connect(db) as conn:
        statuses = [
            r[0] for r in conn.execute("select status from audit.tool_calls order by started_at").fetchall()
        ]
    assert statuses == ["refused", "refused", "ok"]
