"""Approval queue (rule 3): agents propose state changes, humans decide them.

An approval moves pending -> approved | rejected exactly once, and only a named human can decide it.
Write tools run only against an approved record for the same action.
"""

import json
from enum import StrEnum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict

from ledgerlens.core.db import connect, utcnow
from ledgerlens.core.lineage import new_id

AGENT_PREFIXES = ("agent:", "mcp:", "system:")


class ApprovalStatus(StrEnum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"


class ApprovalError(RuntimeError):
    pass


class Approval(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    approval_id: str
    agent_run_id: str
    action: str
    payload: dict[str, Any]
    status: ApprovalStatus
    decided_by: str | None = None
    reason: str | None = None


_COLUMNS = "approval_id, agent_run_id, action, payload_json, status, decided_by, reason"


def _check_decider(decided_by: str, *reasons: str) -> str:
    who = decided_by.strip()
    if not who or who.lower().startswith(AGENT_PREFIXES):
        raise ApprovalError("approvals must be decided by a named human, not an agent or system")
    if any(not r.strip() for r in reasons):
        raise ApprovalError("a decision needs a reason")
    return who


def _row_to_approval(row: tuple[Any, ...]) -> Approval:
    approval_id, agent_run_id, action, payload_json, status, decided_by, reason = row
    return Approval(
        approval_id=approval_id,
        agent_run_id=agent_run_id,
        action=action,
        payload=json.loads(payload_json),
        status=ApprovalStatus(status),
        decided_by=decided_by,
        reason=reason,
    )


class ApprovalQueue:
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path

    def propose(self, *, agent_run_id: str, action: str, payload: dict[str, Any]) -> Approval:
        approval_id = new_id("ap")
        with connect(self.db_path) as conn:
            conn.execute(
                "insert into audit.approvals values (?, ?, ?, ?, 'pending', ?, null, null, null)",
                [
                    approval_id,
                    agent_run_id,
                    action,
                    json.dumps(payload, sort_keys=True, default=str),
                    utcnow(),
                ],
            )
        return self.get(approval_id)

    def get(self, approval_id: str) -> Approval:
        with connect(self.db_path) as conn:
            row = conn.execute(
                f"select {_COLUMNS} from audit.approvals where approval_id = ?",  # noqa: S608 (constant columns)
                [approval_id],
            ).fetchone()
        if row is None:
            raise ApprovalError(f"approval {approval_id} does not exist")
        return _row_to_approval(row)

    def pending(self) -> list[Approval]:
        with connect(self.db_path) as conn:
            rows = conn.execute(
                f"select {_COLUMNS} from audit.approvals where status = 'pending' order by created_at",  # noqa: S608
            ).fetchall()
        return [_row_to_approval(r) for r in rows]

    def decide(self, approval_id: str, *, approve: bool, decided_by: str, reason: str) -> Approval:
        who = _check_decider(decided_by, reason)
        status = ApprovalStatus.APPROVED if approve else ApprovalStatus.REJECTED
        with connect(self.db_path) as conn:
            # Conditional update so two concurrent deciders cannot both win.
            updated = conn.execute(
                """
                update audit.approvals
                set status = ?, decided_by = ?, decided_at = ?, reason = ?
                where approval_id = ? and status = 'pending'
                returning approval_id
                """,
                [status.value, who, utcnow(), reason.strip(), approval_id],
            ).fetchall()
        if not updated:
            current = self.get(approval_id)  # raises if missing
            raise ApprovalError(f"approval {approval_id} was already {current.status.value}")
        return self.get(approval_id)

    def decide_many(
        self, decisions: dict[str, str], *, approve: bool, decided_by: str, action: str
    ) -> list[str]:
        """Decide several pending approvals of one action in a single transaction.

        ``decisions`` maps approval_id -> reason. All succeed or none do; returns the decided ids.
        """
        who = _check_decider(decided_by, *decisions.values())
        status = ApprovalStatus.APPROVED if approve else ApprovalStatus.REJECTED
        now = utcnow()
        with connect(self.db_path) as conn:
            conn.execute("begin transaction")
            for approval_id, reason in decisions.items():
                updated = conn.execute(
                    """
                    update audit.approvals
                    set status = ?, decided_by = ?, decided_at = ?, reason = ?
                    where approval_id = ? and action = ? and status = 'pending'
                    returning approval_id
                    """,
                    [status.value, who, now, reason.strip(), approval_id, action],
                ).fetchall()
                if not updated:
                    conn.execute("rollback")
                    raise ApprovalError(f"approval {approval_id} is not a pending {action!r} request")
            conn.execute("commit")
        return list(decisions)

    def require_approved(self, approval_id: str, *, action: str) -> Approval:
        approval = self.get(approval_id)
        if approval.action != action:
            raise ApprovalError(f"approval {approval_id} covers {approval.action!r}, not {action!r}")
        if approval.status is not ApprovalStatus.APPROVED:
            raise ApprovalError(f"approval {approval_id} is {approval.status.value}, not approved")
        return approval
