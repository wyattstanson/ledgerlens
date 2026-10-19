"""Rate-card tools on a small world: infeasible-panel reporting, tail plans, writes behind approval."""

from pathlib import Path

import polars as pl
import pytest

from ledgerlens.core.approvals import ApprovalError, ApprovalQueue, ApprovalStatus, DecisionKind
from ledgerlens.core.db import connect, ensure_audit_schema
from ledgerlens.core.tools import execute_tool
from ledgerlens.packs import default_tool_registry
from ledgerlens.ratecards.benchmark import benchmarks
from ledgerlens.ratecards.config import OptimizerConfig, RateCardsConfig, WorldConfig
from ledgerlens.ratecards.generate import generate_world, write_world
from ledgerlens.ratecards.mapping import simulate_review_from_truth
from ledgerlens.ratecards.pipeline import run_ratecards

SMALL = RateCardsConfig(
    world=WorldConfig(vendors=24, months=14, duplicate_lines=10, distinct_titles=128),
    optimizer=OptimizerConfig(panel_size=24, critical_top_k=2),
)
TOOLS = default_tool_registry()


def call(db: Path, name: str, args: dict | None = None):
    with connect(db) as conn:
        conn.execute(
            "insert into audit.agent_runs values ('run_t', 'test', 'test', now(), now(), 'completed') "
            "on conflict do nothing"
        )
    return execute_tool(TOOLS.get(name), args or {}, db_path=db, agent_run_id="run_t")


@pytest.fixture(scope="module")
def small_db(tmp_path_factory: pytest.TempPathFactory) -> Path:
    db = tmp_path_factory.mktemp("rc_small") / "w.duckdb"
    ensure_audit_schema(db)
    write_world(db, generate_world(SMALL.world, 7))
    return db


def test_before_review_the_panel_tools_report_infeasible_with_a_reason(small_db: Path):
    tight = SMALL.model_copy(update={"optimizer": SMALL.optimizer.model_copy(update={"panel_size": 2})})
    assert run_ratecards(small_db, tight) is None
    panel = call(small_db, "rc.optimize_panel")
    assert panel.data["status"] == "infeasible" and "relax" in panel.data["message"]
    assert panel.numbers == []
    assert call(small_db, "rc.savings_bridge").data["status"] == "infeasible"
    assert call(small_db, "rc.tail_plan", {"vendor_id": "V01"}).data["status"] == "infeasible"


def test_after_review_tail_plans_and_vendor_rates_have_lineage(small_db: Path):
    simulate_review_from_truth(small_db, ApprovalQueue(small_db))
    solution = run_ratecards(small_db, SMALL)
    assert solution is not None
    outside = next(v for v in (f"V{i:02d}" for i in range(1, 17)) if v not in solution.kept)
    tail = call(small_db, "rc.tail_plan", {"vendor_id": outside})
    assert tail.data["proposal"].startswith(("exit", "negotiate"))
    assert tail.numbers[0].unit == "USD"
    on_panel = call(small_db, "rc.tail_plan", {"vendor_id": solution.kept[0]})
    assert on_panel.data["found"] is False

    with connect(small_db) as conn:
        role, tier = conn.execute(
            "select role_id, location_tier from ratecards.cards_normalized "
            "where effective_usd_per_hour is not null and role_id is not null limit 1"
        ).fetchone() or ("", "")
    rates = call(small_db, "rc.vendor_rates", {"role_id": role, "location_tier": tier})
    assert rates.data["vendors"][0]["rank_by_total_cost"] == 1
    assert {n.label.split()[0] for n in rates.numbers} == {"sticker", "effective"}
    missing = call(small_db, "rc.benchmark", {"role_id": "nope.junior", "location_tier": "onshore"})
    assert missing.data["found"] is False
    assert call(small_db, "rc.vendor_scorecard", {"vendor_id": "V99"}).data["found"] is False


def test_tail_negotiation_targets_never_exceed_the_current_rate(small_db: Path):
    with connect(small_db) as conn:
        briefs = conn.execute(
            "select effective_usd_per_hour, target_usd_per_hour, walk_away_usd_per_hour "
            "from ratecards.tail_briefs"
        ).fetchall()
    assert briefs
    assert all(target <= current and walk <= current for current, target, walk in briefs)


def _approved_panel_proposal(db: Path, *, simulated: bool = False) -> str:
    queue = ApprovalQueue(db)
    tool = TOOLS.get("rc.accept_panel")
    assert tool.snapshot is not None
    proposal = queue.propose(
        agent_run_id="run_t",
        action="rc.accept_panel",
        payload={"arguments": {}, "snapshot": tool.snapshot(db)},
    )
    if simulated:
        queue.decide(proposal.approval_id, approve=True, decided_by="simulated:x", reason="demo",
                     kind=DecisionKind.SIMULATED)  # fmt: skip
    else:
        queue.decide(proposal.approval_id, approve=True, decided_by="priya", reason="panel looks right")
    return proposal.approval_id


def test_accept_panel_needs_a_real_unused_human_approval_for_this_exact_panel(small_db: Path):
    with pytest.raises(ApprovalError, match="does not exist"):
        call(small_db, "rc.accept_panel", {"approval_id": "ap_forged"})
    with pytest.raises(ApprovalError, match="simulated reviewer"):
        call(small_db, "rc.accept_panel", {"approval_id": _approved_panel_proposal(small_db, simulated=True)})

    approval_id = _approved_panel_proposal(small_db)
    result = call(small_db, "rc.accept_panel", {"approval_id": approval_id})
    assert result.data["accepted"] is True
    assert ApprovalQueue(small_db).get(approval_id).status is ApprovalStatus.EXECUTED
    with pytest.raises(ApprovalError, match="executed, not approved"):
        call(small_db, "rc.accept_panel", {"approval_id": approval_id})  # single use


def test_accept_panel_refuses_when_the_panel_moved_after_approval(small_db: Path):
    approval_id = _approved_panel_proposal(small_db)
    with connect(small_db) as conn:
        conn.execute("update ratecards.panel_meta set optimizer_cost_usd = optimizer_cost_usd + 1000")
    with pytest.raises(ApprovalError, match="changed after it was approved"):
        call(small_db, "rc.accept_panel", {"approval_id": approval_id})
    assert ApprovalQueue(small_db).get(approval_id).status is ApprovalStatus.APPROVED  # not consumed


def test_external_benchmarks_are_labelled_with_their_source(tmp_path: Path):
    cards = pl.DataFrame(
        {
            "role_id": ["r.mid"] * 3,
            "location_tier": ["onshore"] * 3,
            "effective_usd_per_hour": [80.0, 90.0, 100.0],
        }
    )
    csv = tmp_path / "external_benchmarks.csv"
    csv.write_text(
        "role_id,location_tier,p25,median,p75,n,source_name,as_of\n"
        "r.mid,onshore,70,85,95,120,Survey X,2026-06\n"
    )
    out = benchmarks(cards, csv).sort("source")
    assert out["source"].to_list() == ["external", "internal"]
    assert out.filter(pl.col("source") == "internal")["median"].item() == 90.0
    assert benchmarks(cards, tmp_path / "absent.csv")["source"].to_list() == ["internal"]
    bad = tmp_path / "bad.csv"
    bad.write_text("role_id,median\nr.mid,1\n")
    with pytest.raises(ValueError, match="missing columns"):
        benchmarks(cards, bad)
