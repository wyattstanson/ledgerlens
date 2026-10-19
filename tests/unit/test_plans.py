"""Plan registration (rule 7): hash before compute, version on change, log every run."""

from pathlib import Path

import pytest
from fakes import fetch_one
from pydantic import ValidationError

from ledgerlens.core.db import connect
from ledgerlens.core.plans import analysis_run, register_plan
from ledgerlens.interviews.config import InterviewsConfig
from ledgerlens.interviews.plan import AnalysisPlan
from ledgerlens.settings import REPO_ROOT

PLAN_YAML = REPO_ROOT / "config" / "analysis_plan.yaml"


def test_same_plan_keeps_its_version_and_any_change_bumps_it(db: Path):
    plan = AnalysisPlan.from_yaml(PLAN_YAML)
    first = register_plan(db, plan.name, plan)
    assert (first.version, register_plan(db, plan.name, plan).version) == (1, 1)
    tighter = plan.model_copy(update={"alpha": 0.01})
    second = register_plan(db, plan.name, tighter)
    assert second.version == 2 and second.sha256 != first.sha256
    assert register_plan(db, plan.name, plan).version == 1  # reverting finds the old version, no v3
    with connect(db) as conn:
        assert fetch_one(conn, "select count(*) from audit.analysis_plans") == (2,)


def test_yaml_comments_and_spacing_do_not_change_the_hash(db: Path, tmp_path: Path):
    noisy = tmp_path / "plan.yaml"
    noisy.write_text("# a reviewer's note\n\n" + PLAN_YAML.read_text(encoding="utf-8"), encoding="utf-8")
    a = register_plan(db, "interviews", AnalysisPlan.from_yaml(PLAN_YAML))
    b = register_plan(db, "interviews", AnalysisPlan.from_yaml(noisy))
    assert a.sha256 == b.sha256


def test_run_is_logged_before_compute_and_closed_after(db: Path):
    plan = AnalysisPlan.from_yaml(PLAN_YAML)
    registered = register_plan(db, plan.name, plan)
    with analysis_run(db, registered, pack="interviews", seed=42) as run_id, connect(db) as conn:
        during = fetch_one(
            conn,
            "select status, plan_sha256, seed from audit.analysis_runs where analysis_run_id = ?",
            [run_id],
        )
    assert during == ("running", registered.sha256, 42)
    with connect(db) as conn:
        assert fetch_one(
            conn, "select status from audit.analysis_runs where analysis_run_id = ?", [run_id]
        ) == ("completed",)


def test_a_failed_run_is_marked_failed_and_the_error_propagates(db: Path):
    plan = AnalysisPlan.from_yaml(PLAN_YAML)
    registered = register_plan(db, plan.name, plan)
    with pytest.raises(ZeroDivisionError), analysis_run(db, registered, pack="interviews", seed=1):
        _ = 1 / 0
    with connect(db) as conn:
        assert fetch_one(conn, "select status from audit.analysis_runs") == ("failed",)


@pytest.mark.parametrize(
    ("update", "message"),
    [
        ({"min_group_n": 4}, "min_group_n"),
        ({"min_group_share": 0.01}, "min_group_share"),
        ({"surprise": 1}, "surprise"),
    ],
)
def test_a_plan_can_only_tighten_the_privacy_floors(update: dict, message: str):
    raw = AnalysisPlan.from_yaml(PLAN_YAML).model_dump()
    with pytest.raises(ValidationError, match=message):
        AnalysisPlan.model_validate(raw | update)


def test_interview_code_defaults_match_the_shipped_yaml():
    assert InterviewsConfig.from_yaml(REPO_ROOT / "config" / "interviews.yaml") == InterviewsConfig()
