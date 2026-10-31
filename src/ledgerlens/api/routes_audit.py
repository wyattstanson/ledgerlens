"""Audit, lineage and approval routes: the paper trail behind every number on a page.

Lineage is the point of the product, so it gets a route of its own: given the ``lineage_id`` that came
back beside a number, return the SQL, the source tables and the row ids that produced it.

Approvals are listed everywhere and decided only where a named human is at the keyboard
(``LEDGERLENS_ALLOW_APPROVALS``, off by default, so a shared deployment is read-only; ADR 0002).
"""

from pathlib import Path
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, ConfigDict, Field

from ledgerlens.api.deps import settings, warehouse
from ledgerlens.core.approvals import ApprovalError, ApprovalQueue
from ledgerlens.core.db import NO_EXTERNAL_ACCESS, connect, fetch_dicts
from ledgerlens.settings import Settings

router = APIRouter(prefix="/api", tags=["audit"])

Config = Annotated[Settings, Depends(settings)]

RECENT_RUNS = 50
RECENT_CALLS = 100

COST_SQL = """
select
    coalesce(sum(cost_usd), 0.0)           as total_usd,
    coalesce(sum(input_tokens), 0)         as input_tokens,
    coalesce(sum(output_tokens), 0)        as output_tokens,
    coalesce(sum(cache_read_tokens), 0)    as cache_read_tokens,
    count(*)                               as calls
from audit.llm_calls
"""

RUNS_SQL = """
select r.agent_run_id, r.agent_id, r."user", r.started_at, r.finished_at, r.status,
       count(t.tool_call_id) as tool_calls
from audit.agent_runs r
left join audit.tool_calls t using (agent_run_id)
group by all
order by r.started_at desc
limit ?
"""

CALLS_SQL = """
select t.tool_call_id, t.agent_run_id, t.tool_name, t.status, t.ms, t.started_at,
       count(l.lineage_id) as numbers
from audit.tool_calls t
left join audit.lineage l using (tool_call_id)
group by all
order by t.started_at desc
limit ?
"""

LINEAGE_SQL = """
select l.lineage_id, l.value, l.unit, l.sql_text, l.sql_hash, l.source_tables, l.row_ids,
       t.tool_name, t.agent_run_id, t.started_at
from audit.lineage l
join audit.tool_calls t using (tool_call_id)
where l.lineage_id = ?
"""

MODELS_SQL = """
select model, task, count(*) as calls, sum(cost_usd) as cost_usd
from audit.llm_calls group by all order by cost_usd desc
"""


def _read(db_path: Path, sql: str, params: list[Any] | None = None) -> list[dict[str, Any]]:
    with warehouse(), connect(db_path, read_only=True, config=NO_EXTERNAL_ACCESS) as conn:
        return fetch_dicts(conn, sql, params)


class Decision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    approve: bool
    decided_by: str = Field(min_length=1, max_length=120)
    reason: str = Field(min_length=1, max_length=2000)


@router.get("/lineage/{lineage_id}")
def lineage(lineage_id: str, config: Config) -> dict[str, Any]:
    """Where one number came from: its SQL, source tables and row ids."""
    rows = _read(config.db, LINEAGE_SQL, [lineage_id])
    if not rows:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"no lineage record {lineage_id!r}")
    return rows[0]


@router.get("/audit")
def audit(config: Config) -> dict[str, Any]:
    """Runs, tool calls and what the language model has cost so far."""
    cost = _read(config.db, COST_SQL)[0]
    return {
        "cost": cost,
        "by_model": _read(config.db, MODELS_SQL),
        "runs": _read(config.db, RUNS_SQL, [RECENT_RUNS]),
        "tool_calls": _read(config.db, CALLS_SQL, [RECENT_CALLS]),
    }


@router.get("/approvals")
def approvals(config: Config) -> dict[str, Any]:
    """The pending queue, and whether this deployment may decide any of it."""
    with warehouse():
        pending = [a.model_dump(mode="json") for a in ApprovalQueue(config.db).pending()]
    return {
        "can_decide": config.allow_approvals,
        "pending": pending,
    }


@router.post("/approvals/{approval_id}/decide")
def decide(approval_id: str, decision: Decision, config: Config) -> dict[str, Any]:
    """Approve or reject one pending record. Refused unless this deployment allows deciding."""
    if not config.allow_approvals:
        raise HTTPException(
            status.HTTP_403_FORBIDDEN,
            "this deployment shows approvals read-only; decide them where you run LedgerLens yourself",
        )
    try:
        with warehouse():
            approved = ApprovalQueue(config.db).decide(
                approval_id,
                approve=decision.approve,
                decided_by=decision.decided_by,
                reason=decision.reason,
            )
    except ApprovalError as exc:
        raise HTTPException(status.HTTP_409_CONFLICT, str(exc)) from exc
    return approved.model_dump(mode="json")
