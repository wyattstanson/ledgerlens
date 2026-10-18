"""Rate-card tools on a small world: infeasible-panel reporting, tail plans, writes behind approval."""

from pathlib import Path

import polars as pl
import pytest

from ledgerlens.core.approvals import ApprovalQueue
from ledgerlens.core.db import connect, ensure_audit_schema
from ledgerlens.core.registry import default_tool_registry
from ledgerlens.core.tools import execute_tool
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


def test_accept_panel_refuses_without_a_feasible_panel(tmp_path: Path):
    db = tmp_path / "w.duckdb"
    ensure_audit_schema(db)
    with connect(db) as conn:
        conn.execute(
            "create schema ratecards; create table ratecards.panel_meta as select 'infeasible' as status"
        )
    with pytest.raises(RuntimeError, match="no feasible panel"):
        call(db, "rc.accept_panel", {"approval_id": "ap_x"})


def test_accept_panel_records_the_decision(small_db: Path):
    result = call(small_db, "rc.accept_panel", {"approval_id": "ap_test"})
    assert result.data["accepted"] is True
    with connect(small_db) as conn:
        assert conn.execute("select approval_id from ratecards.panel_decisions").fetchall() == [("ap_test",)]


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
