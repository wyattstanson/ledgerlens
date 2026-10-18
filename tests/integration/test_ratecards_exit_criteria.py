"""Phase 2 exit criteria (BUILD_PROMPT 12.4) on the full seed-42 world, end to end through the warehouse."""

import json
import os
import subprocess
import sys
from pathlib import Path

import polars as pl
import pytest

from ledgerlens.core.approvals import ApprovalQueue
from ledgerlens.core.db import connect, ensure_audit_schema
from ledgerlens.core.lineage import lineage_values_for_run
from ledgerlens.core.registry import default_tool_registry
from ledgerlens.core.tools import execute_tool
from ledgerlens.ratecards.config import RateCardsConfig
from ledgerlens.ratecards.drift import RATIO_EPSILON
from ledgerlens.ratecards.generate import World, generate_world, write_world
from ledgerlens.ratecards.golden import golden_cases
from ledgerlens.ratecards.mapping import (
    FuzzyPicker,
    evaluate_mapping,
    load_title_map,
    simulate_review_from_truth,
)
from ledgerlens.ratecards.pipeline import load_raw, run_ratecards, update_title_map

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG = RateCardsConfig.from_yaml(REPO_ROOT / "config" / "ratecards.yaml")
SEED = 42


@pytest.fixture(scope="module")
def built(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, World, dict[str, float]]:
    db = tmp_path_factory.mktemp("rc") / "warehouse.duckdb"
    ensure_audit_schema(db)
    world = generate_world(CONFIG.world, SEED)
    write_world(db, world)
    approvals = ApprovalQueue(db)
    update_title_map(db, load_raw(db), FuzzyPicker(CONFIG.mapping), CONFIG, approvals)
    before_review = evaluate_mapping(load_title_map(db), world.title_truth)
    simulate_review_from_truth(db, approvals)
    assert run_ratecards(db, CONFIG) is not None
    return db, world, before_review


def table(db: Path, sql: str) -> pl.DataFrame:
    with connect(db) as conn:
        return conn.execute(sql).pl()


def test_world_has_the_specified_shape(built):
    _, world, _ = built
    assert world.vendors.height == 40
    assert world.roles.height == 64  # "about 60 standard roles"
    assert world.invoice_lines["billed_title"].n_unique() == 600
    assert 45_000 <= world.invoice_lines.height <= 60_000  # "about 50,000 invoice lines"
    assert set(world.invoice_lines["currency"]) == {"USD", "MXN", "INR"}


def test_title_mapping_accuracy_after_review_with_review_rate_reported(built):
    db, world, before = built
    after = evaluate_mapping(load_title_map(db), world.title_truth)
    print(
        f"\nmapping: auto-accepted {before['auto_accepted']:.0f}/600, "
        f"auto accuracy {before['auto_accuracy']:.3f}, "
        f"review rate {before['review_rate']:.3f}, accuracy after review {after['accuracy_after_review']:.3f}"
    )
    assert before["auto_accuracy"] >= 0.95
    assert after["accuracy_after_review"] >= 0.95
    assert before["review_rate"] < 0.5


def test_every_planted_overbill_and_duplicate_is_found_and_traced(built):
    db, world, _ = built
    leakage = table(db, "select line_id, kind, leakage_usd from ratecards.leakage")
    planted = world.planted
    threshold = CONFIG.drift.threshold
    expected_over = planted.filter(
        pl.col("kind").is_in(["overbill", "drift"]) & (pl.col("excess_ratio") > threshold + RATIO_EPSILON)
    )
    found_over = leakage.filter(pl.col("kind") == "overbill")
    assert set(found_over["line_id"]) == set(expected_over["line_id"])  # all found, no false flags
    assert set(planted.filter(pl.col("kind") == "overbill")["line_id"]) <= set(found_over["line_id"])
    found_dup = leakage.filter(pl.col("kind") == "duplicate")
    assert set(found_dup["line_id"]) == set(planted.filter(pl.col("kind") == "duplicate")["line_id"])
    by_line = dict(found_over.select("line_id", "leakage_usd").iter_rows())
    assert all(
        abs(by_line[r["line_id"]] - r["excess_usd"]) < 0.005 for r in expected_over.iter_rows(named=True)
    )


def test_panel_honours_size_and_critical_coverage(built):
    db, _, _ = built
    meta = table(db, "select * from ratecards.panel_meta").row(0, named=True)
    kept = json.loads(meta["kept_vendors"])
    assert meta["status"] == "optimal" and meta["relaxed_constraint"] is None
    assert len(kept) <= CONFIG.optimizer.panel_size
    allocation = table(db, "select * from ratecards.panel_allocation")
    cards = table(
        db,
        "select vendor_id, role_id || '@' || location_tier as role_key from ratecards.cards_normalized "
        "where role_id is not null and effective_usd_per_hour is not null",
    )
    for role in json.loads(meta["critical_roles"]):
        suppliers = set(cards.filter(pl.col("role_key") == role)["vendor_id"]) & set(kept)
        assert len(suppliers) >= 2, role
        hours = allocation.filter(pl.col("role_key") == role)["hours"].to_list()
        assert max(hours) <= CONFIG.optimizer.max_share_critical * sum(hours) * (1 + 1e-6)
    assert set(allocation["vendor_id"]) <= set(kept)


def test_savings_bridge_sums_exactly(built):
    db, _, _ = built
    steps = dict(table(db, "select step, cents from ratecards.savings_bridge").iter_rows())
    middle = (
        steps["leakage_recovery"]
        + steps["rate_harmonization"]
        + steps["volume_shift"]
        + steps["vendor_exits"]
    )
    assert steps["current_cost"] - middle == steps["new_cost"]
    traced = dict(
        table(db, "select step, sum(cents) as cents from ratecards.savings_lines group by step").iter_rows()
    )
    assert all(
        traced.get(s, 0) == steps[s]
        for s in ("leakage_recovery", "rate_harmonization", "volume_shift", "vendor_exits")
    )
    assert (
        abs(table(db, "select optimizer_check_cents from ratecards.panel_meta").item()) < 1_000
    )  # < $10 rounding


def test_total_cost_normalization_ranks_the_cheap_per_hour_vendor_correctly(built):
    db, world, _ = built
    cheap = world.planted.filter(pl.col("kind") == "cheap_vendor")["vendor_id"][0]
    outcome = world.planted.filter(pl.col("kind") == "outcome_vendor")["vendor_id"][0]
    rates = table(
        db,
        f"select vendor_id, avg(sticker_usd_per_hour) as sticker, avg(effective_usd_per_hour) as eff "  # noqa: S608
        f"from ratecards.cards_normalized where vendor_id in ('{cheap}', '{outcome}') group by 1",
    )
    r = {row["vendor_id"]: row for row in rates.iter_rows(named=True)}
    assert r[cheap]["sticker"] < r[outcome]["sticker"]  # cheaper per hour...
    assert r[outcome]["eff"] < r[cheap]["eff"]  # ...but loses on total cost
    scores = dict(table(db, "select vendor_id, rate_index from ratecards.scorecard").iter_rows())
    assert scores[outcome] < scores[cheap]


def test_tools_return_numbers_with_lineage(built):
    db, _, _ = built
    tools = default_tool_registry()
    run_id = "run_tools"
    with connect(db) as conn:
        conn.execute(
            "insert into audit.agent_runs values (?, 'test', 'test', now(), now(), 'completed')", [run_id]
        )
    results = [
        execute_tool(tools.get(name), args, db_path=db, agent_run_id=run_id)
        for name, args in [
            ("rc.optimize_panel", {}),
            ("rc.savings_bridge", {}),
            ("rc.leakage", {}),
            ("rc.mapping_status", {}),
            ("rc.vendor_scorecard", {"vendor_id": "V01"}),
            ("rc.benchmark", {"role_id": "backend.senior", "location_tier": "onshore"}),
        ]
    ]
    recorded = lineage_values_for_run(db, run_id)
    numbers = [n for r in results for n in r.numbers]
    assert numbers and all(recorded[n.lineage_id] == n.value for n in numbers)
    reviewed = next(n for n in results[3].numbers if n.label == "titles reviewed")
    assert reviewed.value > 0


def test_committed_golden_set_matches_generator_truth():
    committed = (REPO_ROOT / "tests" / "evals" / "golden" / "ratecards.jsonl").read_text(encoding="utf-8")
    regenerated = "".join(
        json.dumps(c, sort_keys=True) + "\n" for c in golden_cases(generate_world(CONFIG.world, SEED), CONFIG)
    )
    assert committed == regenerated, "run `python -m ledgerlens.cli golden ratecards` and commit the result"
    assert len(committed.splitlines()) >= 30


def test_generation_is_identical_across_hash_seeds():
    code = (
        "import hashlib; from pathlib import Path; from ledgerlens.ratecards.config import RateCardsConfig; "
        "from ledgerlens.ratecards.generate import generate_world; "
        "w = generate_world(RateCardsConfig.from_yaml(Path(r'%s')).world, 42); "
        "print(hashlib.sha256((w.invoice_lines.write_csv() + w.planted.write_csv()).encode()).hexdigest())"
    ) % (REPO_ROOT / "config" / "ratecards.yaml")
    digests = {
        subprocess.run(
            [sys.executable, "-c", code],
            env={**os.environ, "PYTHONHASHSEED": seed},
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        for seed in ("1", "2")
    }
    assert len(digests) == 1
