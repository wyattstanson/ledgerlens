"""The interviews job on a small world: strict plans that leave nothing eligible, and a failing tagger."""

from collections.abc import Sequence
from pathlib import Path

import pytest

from ledgerlens.core.db import connect, ensure_audit_schema
from ledgerlens.interviews.config import WorldConfig
from ledgerlens.interviews.generate import generate_world, write_world
from ledgerlens.interviews.pipeline import TABLES, run_interviews
from ledgerlens.interviews.plan import AnalysisPlan
from ledgerlens.interviews.tagger import NoteToTag, RuleTagger
from ledgerlens.settings import REPO_ROOT

PLAN = AnalysisPlan.from_yaml(REPO_ROOT / "config" / "analysis_plan.yaml")


class BrokenTagger:
    """Stands in for a Haiku outage: every note falls back to "no evidence" and counts as a failure."""

    name = "claude"

    def __init__(self) -> None:
        self.failures = 0

    def tag(self, notes: Sequence[NoteToTag]) -> dict[str, bool]:
        self.failures = len(notes)
        return dict.fromkeys((n.note_id for n in notes), False)


@pytest.fixture(scope="module")
def small(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    root = tmp_path_factory.mktemp("iv_small")
    warehouse, vault = root / "warehouse.duckdb", root / "vault.duckdb"
    ensure_audit_schema(warehouse)
    write_world(warehouse, vault, generate_world(WorldConfig(interviewers=20, candidates=400), 3))
    return warehouse, vault


def test_a_plan_strict_enough_to_leave_nothing_eligible_still_completes(small: tuple[Path, Path]):
    strict = PLAN.model_copy(
        update={
            "calibration": PLAN.calibration.model_copy(update={"min_n": 100_000}),
            "predictive_validity": PLAN.predictive_validity.model_copy(update={"min_n": 100_000}),
        }
    )
    run_interviews(*small, strict, RuleTagger())
    with connect(small[0]) as conn:
        assert conn.execute("select count(*) from interviews.calibration").fetchone() == (0,)
        assert conn.execute(
            "select count(*) from interviews.predictive_validity where flagged"
        ).fetchone() == (0,)
        meta = conn.execute(
            "select flagged_interviewers, tagger_failures from interviews.run_meta"
        ).fetchone()
        tables = {
            r[0]
            for r in conn.execute(
                "select table_name from information_schema.tables where table_schema = 'interviews'"
            ).fetchall()
        }
    assert meta == (0, 0)
    assert set(TABLES) <= tables


def test_a_tagger_outage_fails_the_run_instead_of_flagging_everyone(small: tuple[Path, Path]):
    with pytest.raises(RuntimeError, match="tagger failed"):
        run_interviews(*small, PLAN, BrokenTagger())
    with connect(small[0]) as conn:
        status = conn.execute(
            "select status from audit.analysis_runs order by started_at desc limit 1"
        ).fetchone()
    assert status == ("failed",)
