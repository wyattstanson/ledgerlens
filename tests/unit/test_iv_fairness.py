"""Impact ratios on a hand-built stage: ratio vs the top group, test vs everyone else (ADR 0004)."""

import polars as pl
import pytest

from ledgerlens.core.stats import cohens_h, fisher_p
from ledgerlens.interviews.fairness import impact_ratios, intersectional
from ledgerlens.interviews.plan import AnalysisPlan
from ledgerlens.settings import REPO_ROOT

PLAN = AnalysisPlan.from_yaml(REPO_ROOT / "config" / "analysis_plan.yaml")


def stage(groups: dict[str, tuple[int, int]]) -> pl.DataFrame:
    """One role and stage; ``groups`` maps a sex value to (entered, passed). Race is constant."""
    rows = []
    for sex, (n, passed) in groups.items():
        base = {"role_id": "r", "stage": "screen", "sex": sex, "race_ethnicity": "White"}
        rows += [{**base, "passed": i < passed} for i in range(n)]
    return pl.DataFrame(rows)


def by_group(frame: pl.DataFrame) -> dict[str, dict]:
    return {r["group"]: r for r in frame.filter(pl.col("dimension") == "sex").to_dicts()}


def test_ratio_uses_the_top_group_and_the_test_uses_everyone_else():
    rows = by_group(
        impact_ratios(stage({"A": (100, 60), "B": (100, 40), "C": (100, 50), "Unknown": (40, 39)}), PLAN)
    )
    assert rows["B"]["reference_group"] == "A"  # Unknown has the top rate but is never the reference
    assert rows["B"]["impact_ratio"] == pytest.approx(0.4 / 0.6)
    # B's complement: A + C + Unknown = 240 entered, 149 passed. Never just A, the group picked after looking.
    assert rows["B"]["p_raw"] == pytest.approx(fisher_p(40, 100, 149, 240))
    assert rows["B"]["h"] == pytest.approx(cohens_h(0.4, 149 / 240))
    assert rows["B"]["h_ci_low"] < rows["B"]["h"] < rows["B"]["h_ci_high"]
    assert rows["Unknown"]["impact_ratio"] is None and not rows["Unknown"]["tested"]


def test_small_share_groups_are_excluded_and_small_groups_are_not_tested():
    rows = by_group(
        impact_ratios(stage({"A": (500, 300), "B": (480, 250), "X": (8, 1), "Y": (25, 10)}), PLAN)
    )
    assert rows["X"]["excluded_small_share"] and rows["X"]["impact_ratio"] is None  # 8 / 1013 < 2%
    assert not rows["Y"]["excluded_small_share"] and rows["Y"]["impact_ratio"] == pytest.approx(0.4 / 0.6)
    assert not rows["Y"]["tested"] and rows["Y"]["p_raw"] is None  # 25 < min_test_n
    assert rows["B"]["tested"]


def test_intersectional_ratios_skip_unknown_and_small_share_cells():
    frame = stage({"A": (300, 150), "B": (300, 120), "Unknown": (100, 90), "X": (5, 5)})
    rows = {r["sex"]: r for r in intersectional(frame, PLAN).to_dicts()}
    assert rows["B"]["impact_ratio"] == pytest.approx(0.4 / 0.5)
    assert rows["Unknown"]["impact_ratio"] is None
    assert rows["X"]["impact_ratio"] is None and rows["X"]["excluded_small_share"]
