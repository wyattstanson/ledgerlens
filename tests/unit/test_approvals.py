import pytest

from ledgerlens.core.approvals import ApprovalError, ApprovalQueue, ApprovalStatus


def test_proposal_starts_pending_and_appears_in_queue(approvals: ApprovalQueue):
    proposal = approvals.propose(agent_run_id="run_1", action="rc.accept_panel", payload={"vendors": ["v1"]})
    assert proposal.status is ApprovalStatus.PENDING
    assert proposal.payload == {"vendors": ["v1"]}
    assert [a.approval_id for a in approvals.pending()] == [proposal.approval_id]


def test_human_decision_is_recorded_once(approvals: ApprovalQueue):
    proposal = approvals.propose(agent_run_id="run_1", action="a", payload={})
    decided = approvals.decide(proposal.approval_id, approve=True, decided_by="priya", reason="matches brief")
    assert (decided.status, decided.decided_by, decided.reason) == (
        ApprovalStatus.APPROVED,
        "priya",
        "matches brief",
    )
    assert approvals.pending() == []
    with pytest.raises(ApprovalError, match="already approved"):
        approvals.decide(proposal.approval_id, approve=False, decided_by="sam", reason="changed mind")


@pytest.mark.parametrize("who", ["", "  ", "agent:ratecards.panel", "MCP:client", "system:cron"])
def test_agents_and_systems_cannot_decide(approvals: ApprovalQueue, who: str):
    proposal = approvals.propose(agent_run_id="run_1", action="a", payload={})
    with pytest.raises(ApprovalError, match="named human"):
        approvals.decide(proposal.approval_id, approve=True, decided_by=who, reason="r")


def test_decision_needs_a_reason(approvals: ApprovalQueue):
    proposal = approvals.propose(agent_run_id="run_1", action="a", payload={})
    with pytest.raises(ApprovalError, match="reason"):
        approvals.decide(proposal.approval_id, approve=False, decided_by="priya", reason=" ")


def test_require_approved_checks_status_and_action(approvals: ApprovalQueue):
    proposal = approvals.propose(agent_run_id="run_1", action="a", payload={})
    with pytest.raises(ApprovalError, match="pending"):
        approvals.require_approved(proposal.approval_id, action="a")
    approvals.decide(proposal.approval_id, approve=True, decided_by="priya", reason="ok")
    with pytest.raises(ApprovalError, match="covers 'a'"):
        approvals.require_approved(proposal.approval_id, action="b")
    assert approvals.require_approved(proposal.approval_id, action="a").status is ApprovalStatus.APPROVED


def test_rejected_approval_cannot_authorise(approvals: ApprovalQueue):
    proposal = approvals.propose(agent_run_id="run_1", action="a", payload={})
    approvals.decide(proposal.approval_id, approve=False, decided_by="priya", reason="no")
    with pytest.raises(ApprovalError, match="rejected"):
        approvals.require_approved(proposal.approval_id, action="a")


def test_unknown_approval_id(approvals: ApprovalQueue):
    with pytest.raises(ApprovalError, match="does not exist"):
        approvals.get("ap_missing")
    with pytest.raises(ApprovalError, match="does not exist"):
        approvals.decide("ap_missing", approve=True, decided_by="priya", reason="r")
