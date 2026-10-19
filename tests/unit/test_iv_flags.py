"""The pre-registered calibration flag rule, one condition at a time (ADR 0004 decision 2)."""

import polars as pl

from ledgerlens.interviews.pipeline import flag_calibration, flag_interviewers
from ledgerlens.interviews.plan import AnalysisPlan
from ledgerlens.settings import REPO_ROOT

PLAN = AnalysisPlan.from_yaml(REPO_ROOT / "config" / "analysis_plan.yaml")  # z 1.5, q 0.05, |d| 0.5, 2 cells


def test_a_cell_needs_z_bh_q_and_effect_size_together():
    cells = pl.DataFrame(
        {
            "case": ["all pass", "negative side", "z too small", "q too large", "effect too small"],
            "z": [2.0, -2.0, 1.4, 2.0, 2.0],
            "q": [0.01, 0.01, 0.01, 0.06, 0.01],
            "d": [0.8, -0.8, 0.8, 0.8, 0.4],
        }
    )
    flagged = dict(flag_calibration(cells, PLAN).select("case", "flagged").iter_rows())
    assert flagged == {
        "all pass": True,
        "negative side": True,
        "z too small": False,
        "q too large": False,
        "effect too small": False,
    }


def test_an_interviewer_needs_two_flagged_cells_in_one_role():
    cal = pl.DataFrame(
        {
            "interviewer_id": ["INT-01", "INT-01", "INT-01", "INT-01", "INT-02", "INT-02"],
            "role_id": ["backend", "backend", "backend", "qa", "backend", "backend"],
            "flagged": [True, True, False, True, True, False],
        }
    )
    out = {
        (r["interviewer_id"], r["role_id"]): (r["cells"], r["flagged_cells"], r["flagged"])
        for r in flag_interviewers(cal, PLAN).to_dicts()
    }
    assert out == {
        ("INT-01", "backend"): (3, 2, True),
        ("INT-01", "qa"): (1, 1, False),
        ("INT-02", "backend"): (2, 1, False),
    }
