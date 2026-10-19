"""Phase 3 exit criteria (BUILD_PROMPT 11.5) on three seeds: each planted problem flagged, nothing else."""

import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import polars as pl
import pytest

from ledgerlens.core.db import connect, ensure_audit_schema
from ledgerlens.core.evals.golden import load_golden, run_offline_case
from ledgerlens.core.plans import RegisteredPlan
from ledgerlens.interviews.config import InterviewsConfig
from ledgerlens.interviews.generate import InterviewWorld, generate_world, write_world
from ledgerlens.interviews.golden import golden_cases
from ledgerlens.interviews.pipeline import CELL_TABLES, TABLES, run_interviews
from ledgerlens.interviews.plan import AnalysisPlan
from ledgerlens.interviews.tagger import RuleTagger
from ledgerlens.packs import default_tool_registry
from ledgerlens.settings import REPO_ROOT

CONFIG = InterviewsConfig.from_yaml(REPO_ROOT / "config" / "interviews.yaml")
PLAN = AnalysisPlan.from_yaml(REPO_ROOT / "config" / "analysis_plan.yaml")
GOLDEN_DIR = REPO_ROOT / "tests" / "evals" / "golden"


@dataclass(frozen=True)
class Built:
    warehouse: Path
    world: InterviewWorld
    registered: RegisteredPlan


@pytest.fixture(scope="module", params=[42, 43, 44], ids=lambda s: f"seed{s}")
def built(request: pytest.FixtureRequest, tmp_path_factory: pytest.TempPathFactory) -> Built:
    root = tmp_path_factory.mktemp(f"iv{request.param}")
    warehouse, vault = root / "warehouse.duckdb", root / "vault.duckdb"
    ensure_audit_schema(warehouse)
    world = generate_world(CONFIG.world, request.param)
    write_world(warehouse, vault, world)
    return Built(warehouse, world, run_interviews(warehouse, vault, PLAN, RuleTagger()))


def rows(db: Path, sql: str) -> list[tuple[Any, ...]]:
    with connect(db) as conn:
        return conn.execute(sql).fetchall()


def planted(world: InterviewWorld, kind: str) -> dict[str, Any]:
    return world.truth_planted.filter(pl.col("kind") == kind).row(0, named=True)


def test_only_the_lenient_interviewer_is_flagged(built: Built):
    p = planted(built.world, "lenient_interviewer")
    expected = [(p["interviewer_id"], p["role_id"])]
    assert (
        rows(
            built.warehouse, "select interviewer_id, role_id from interviews.interviewer_flags where flagged"
        )
        == expected
    )
    assert set(
        rows(built.warehouse, "select interviewer_id, role_id from interviews.calibration where flagged")
    ) == set(expected)


def test_only_the_random_panel_is_flagged(built: Built):
    p = planted(built.world, "random_panel")
    assert rows(built.warehouse, "select panel_id from interviews.panel_agreement where flagged") == [
        (p["panel_id"],)
    ]
    clean = rows(built.warehouse, "select min(kappa_mean) from interviews.panel_agreement where not flagged")[
        0
    ][0]
    assert clean >= PLAN.panel_agreement.kappa_threshold


def test_only_the_planted_adverse_impact_cell_is_flagged(built: Built):
    p = planted(built.world, "adverse_impact")
    flagged = rows(
        built.warehouse,
        'select role_id, stage, dimension, "group", impact_ratio from interviews.fairness where flagged',
    )
    assert [r[:4] for r in flagged] == [(p["role_id"], p["stage"], p["dimension"], p["group_value"])]
    assert 0.5 < flagged[0][4] < PLAN.fairness.impact_ratio_threshold


def test_only_the_non_predictive_role_is_flagged_and_process_is_clean(built: Built):
    p = planted(built.world, "non_predictive_role")
    assert rows(built.warehouse, "select role_id from interviews.predictive_validity where flagged") == [
        (p["role_id"],)
    ]
    assert rows(built.warehouse, "select count(*) from interviews.process_quality where flagged") == [(0,)]


def test_one_bh_family_covers_every_calibration_and_tested_fairness_cell(built: Built):
    [(family,)] = rows(built.warehouse, "select tests_in_family from interviews.run_meta")
    [(cal,)] = rows(built.warehouse, "select count(*) from interviews.calibration")
    [(fair,)] = rows(built.warehouse, "select count(*) from interviews.fairness where tested")
    assert family == cal + fair
    assert rows(built.warehouse, "select count(*) from interviews.calibration where q < p_raw - 1e-12") == [
        (0,)
    ]


def test_every_result_row_cites_the_completed_registered_run(built: Built):
    [(run_id, status, sha)] = rows(
        built.warehouse, "select analysis_run_id, status, plan_sha256 from audit.analysis_runs"
    )
    assert status == "completed" and sha == built.registered.sha256
    for table in TABLES:
        stamps = rows(
            built.warehouse,
            f"select distinct analysis_run_id, plan_version, plan_sha256 from interviews.{table}",  # noqa: S608
        )
        assert stamps == [(run_id, built.registered.version, sha)], table


def test_no_published_cell_is_below_the_floor(built: Built):
    for table in CELL_TABLES:
        assert rows(
            built.warehouse,
            f"select count(*) from interviews.{table} where n < {PLAN.min_group_n} and not suppressed",  # noqa: S608
        ) == [(0,)]


def test_tagger_agreement_is_reported_with_its_caveat(built: Built):
    [(n, kappa, target)] = rows(
        built.warehouse,
        "select notes, kappa, target from interviews.tagger_quality "
        "where comparison = 'tagger_vs_generator_truth'",
    )
    assert n == PLAN.tagger.eval_sample and target == PLAN.tagger.kappa_target
    assert kappa > 0.4  # a sanity floor, not the target: the owner's labels are the real check


def test_committed_golden_cases_pass_through_the_tools(built: Built):
    if built.world.seed != 42:
        pytest.skip("the committed golden set is seed 42")
    tools = default_tool_registry()
    cases = [c for c in load_golden(GOLDEN_DIR, "offline") if c.pack == "interviews"]
    results = [run_offline_case(c, tools=tools, db_path=built.warehouse) for c in cases]
    assert len(cases) >= 30
    assert [(r.case_id, r.failures) for r in results if not r.passed] == []


def test_committed_golden_set_matches_generator_truth():
    committed = (GOLDEN_DIR / "interviews.jsonl").read_text(encoding="utf-8")
    regenerated = "".join(
        json.dumps(c, sort_keys=True) + "\n" for c in golden_cases(generate_world(CONFIG.world, 42), PLAN)
    )
    assert committed == regenerated, "run `python -m ledgerlens.cli golden interviews` and commit the result"


def test_generation_is_identical_across_hash_seeds():
    code = (
        "import hashlib; from pathlib import Path; "
        "from ledgerlens.interviews.config import InterviewsConfig; "
        "from ledgerlens.interviews.generate import generate_world; "
        "w = generate_world(InterviewsConfig.from_yaml(Path(r'%s')).world, 42); "
        "print(hashlib.sha256((w.ratings.write_csv() + w.notes.write_csv() + w.stage_outcomes.write_csv()"
        " + w.truth_planted.write_csv()).encode()).hexdigest())"
    ) % (REPO_ROOT / "config" / "interviews.yaml")
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
