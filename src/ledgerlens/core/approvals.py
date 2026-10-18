"""Approval queue (rule 3): agents propose state changes, humans decide them.

Lifecycle: pending -> approved | rejected (decided once), then approved -> executed (used once).

- Only a named human decides a real approval. Agent, MCP, pipeline and eval actors are refused, and
  names are NFKC-normalized first so look-alike characters cannot slip past the check.
- Demo and eval runs may decide with ``kind="simulated"`` (decider names start with ``simulated:``).
  Those decisions are labelled in the audit log and write tools refuse them by default.
- Write tools consume an approval exactly once (``mark_executed``).
"""

import json
import unicodedata
from collections.abc import Callable
from enum import StrEnum
from pathlib import Path
from typing import Any

import duckdb
from pydantic import BaseModel, ConfigDict

from ledgerlens.core.db import connect, utcnow
from ledgerlens.core.lineage import new_id

BLOCKED_PREFIXES = ("agent:", "mcp:", "system:", "pipeline:", "eval:")
SIMULATED_PREFIX = "simulated:"
# Internal actor ids that must never pass as a human decider.
SYSTEM_ACTORS = frozenset({"pipeline.ratecards", "mcp.client", "eval.offline", "eval", "pipeline", "mcp"})


class ApprovalStatus(StrEnum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    EXECUTED = "executed"


class DecisionKind(StrEnum):
    HUMAN = "human"
    SIMULATED = "simulated"


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
    decided_kind: DecisionKind | None = None


_COLUMNS = "approval_id, agent_run_id, action, payload_json, status, decided_by, reason, decided_kind"


def _check_decider(decided_by: str, kind: DecisionKind, reasons: list[str]) -> str:
    who = unicodedata.normalize("NFKC", decided_by).strip()
    lowered = who.lower()
    if not who or lowered.startswith(BLOCKED_PREFIXES) or lowered in SYSTEM_ACTORS:
        raise ApprovalError("approvals must be decided by a named human, not an agent or system")
    if (kind is DecisionKind.SIMULATED) != lowered.startswith(SIMULATED_PREFIX):
        raise ApprovalError(
            f"simulated decisions, and only those, use a decider named '{SIMULATED_PREFIX}...'"
        )
    if any(not r.strip() for r in reasons):
        raise ApprovalError("a decision needs a reason")
    return who


def _row_to_approval(row: tuple[Any, ...]) -> Approval:
    approval_id, agent_run_id, action, payload_json, status, decided_by, reason, kind = row
    return Approval(
        approval_id=approval_id,
        agent_run_id=agent_run_id,
        action=action,
        payload=json.loads(payload_json),
        status=ApprovalStatus(status),
        decided_by=decided_by,
        reason=reason,
        decided_kind=DecisionKind(kind) if kind else None,
    )


class ApprovalQueue:
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path

    def propose(self, *, agent_run_id: str, action: str, payload: dict[str, Any]) -> Approval:
        approval_id = new_id("ap")
        with connect(self.db_path) as conn:
            conn.execute(
                "insert into audit.approvals "
                "(approval_id, agent_run_id, action, payload_json, status, created_at) "
                "values (?, ?, ?, ?, 'pending', ?)",
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

    def decide(
        self,
        approval_id: str,
        *,
        approve: bool,
        decided_by: str,
        reason: str,
        kind: DecisionKind = DecisionKind.HUMAN,
    ) -> Approval:
        self.decide_many(
            {approval_id: reason}, approve=approve, decided_by=decided_by, action=None, kind=kind
        )
        return self.get(approval_id)

    def decide_many(
        self,
        decisions: dict[str, str],
        *,
        approve: bool,
        decided_by: str,
        action: str | None,
        kind: DecisionKind = DecisionKind.HUMAN,
        in_transaction: Callable[[duckdb.DuckDBPyConnection], None] | None = None,
    ) -> None:
        """Decide pending approvals (approval_id -> reason) in one transaction: all succeed or none do.

        ``action`` restricts every id to one action (None = any). ``in_transaction`` runs on the same
        connection before commit, so a caller's dependent write rolls back with the decisions.
        """
        if not decisions:
            return
        who = _check_decider(decided_by, kind, list(decisions.values()))
        status = ApprovalStatus.APPROVED if approve else ApprovalStatus.REJECTED
        now = utcnow()
        with connect(self.db_path) as conn:
            conn.execute("begin transaction")
            try:
                for approval_id, reason in decisions.items():
                    # Conditional update so two concurrent deciders cannot both win.
                    updated = conn.execute(
                        """
                        update audit.approvals
                        set status = ?, decided_by = ?, decided_at = ?, reason = ?, decided_kind = ?
                        where approval_id = ? and status = 'pending' and (?::varchar is null or action = ?)
                        returning approval_id
                        """,
                        [status.value, who, now, reason.strip(), kind.value, approval_id, action, action],
                    ).fetchall()
                    if not updated:
                        raise ApprovalError(self._why_not_pending(conn, approval_id, action))
                if in_transaction is not None:
                    in_transaction(conn)
            except Exception:
                conn.execute("rollback")
                raise
            conn.execute("commit")

    @staticmethod
    def _why_not_pending(conn: duckdb.DuckDBPyConnection, approval_id: str, action: str | None) -> str:
        row = conn.execute(
            "select status, action from audit.approvals where approval_id = ?", [approval_id]
        ).fetchone()
        if row is None:
            return f"approval {approval_id} does not exist"
        if action is not None and row[1] != action:
            return f"approval {approval_id} covers {row[1]!r}, not {action!r}"
        return f"approval {approval_id} was already {row[0]}"

    def require_approved(self, approval_id: str, *, action: str, allow_simulated: bool = False) -> Approval:
        approval = self.get(approval_id)
        if approval.action != action:
            raise ApprovalError(f"approval {approval_id} covers {approval.action!r}, not {action!r}")
        if approval.status is not ApprovalStatus.APPROVED:
            raise ApprovalError(f"approval {approval_id} is {approval.status.value}, not approved")
        if approval.decided_kind is DecisionKind.SIMULATED and not allow_simulated:
            raise ApprovalError(f"approval {approval_id} was decided by a simulated reviewer")
        return approval

    def mark_executed(self, approval_id: str, conn: duckdb.DuckDBPyConnection | None = None) -> None:
        """Consume an approved record exactly once (pass ``conn`` to join a caller's transaction)."""
        sql = (
            "update audit.approvals set status = 'executed' "
            "where approval_id = ? and status = 'approved' returning 1"
        )
        if conn is not None:
            used = conn.execute(sql, [approval_id]).fetchall()
        else:
            with connect(self.db_path) as own:
                used = own.execute(sql, [approval_id]).fetchall()
        if not used:
            raise ApprovalError(f"approval {approval_id} is not an unused approved record")
