"""Impact ratios on hand-built stages: analysis against the true top group, publishing against what shows."""

import polars as pl
import pytest

from ledgerlens.core.stats import cohens_h, fisher_p
from ledgerlens.interviews.fairness import (
    impact_ratios,
    intersectional,
    publish_impact,
    publish_intersectional,
    q_band,
)
from ledgerlens.interviews.plan import AnalysisPlan
from ledgerlens.settings import REPO_ROOT

PLAN = AnalysisPlan.from_yaml(REPO_ROOT / "config" / "analysis_plan.yaml")


def stage(groups: dict[str, tuple[int, int]], name: str = "screen") -> list[dict]:
    """One role and stage; ``groups`` maps a sex value to (entered, passed). Race is constant."""
    rows = []
    for sex, (n, passed) in groups.items():
        base = {"role_id": "r", "stage": name, "sex": sex, "race_ethnicity": "White"}
        rows += [{**base, "passed": i < passed} for i in range(n)]
    return rows


def by_group(frame: pl.DataFrame, at: str = "screen") -> dict[str, dict]:
    picked = frame.filter((pl.col("dimension") == "sex") & (pl.col("stage") == at))
    return {r["group"]: r for r in picked.to_dicts()}


def test_ratio_uses_the_top_group_and_the_test_uses_everyone_else():
    frame = pl.DataFrame(stage({"A": (100, 60), "B": (100, 40), "C": (100, 50), "Unknown": (40, 39)}))
    rows = by_group(impact_ratios(frame, PLAN))
    assert rows["B"]["reference_group"] == "A"  # Unknown has the top rate but is never the reference
    assert rows["B"]["impact_ratio"] == pytest.approx(0.4 / 0.6)
    # B's complement: A + C + Unknown = 240 entered, 149 passed. Never just A, the group picked after looking.
    assert rows["B"]["p_raw"] == pytest.approx(fisher_p(40, 100, 149, 240))
    assert rows["Unknown"]["impact_ratio"] is None and not rows["Unknown"]["tested"]


def test_small_share_groups_are_excluded_and_small_groups_are_not_tested():
    frame = pl.DataFrame(stage({"A": (500, 300), "B": (480, 250), "X": (8, 1), "Y": (25, 10)}))
    rows = by_group(impact_ratios(frame, PLAN))
    assert rows["X"]["excluded_small_share"] and rows["X"]["impact_ratio"] is None  # 8 / 1013 < 2%
    assert not rows["Y"]["excluded_small_share"] and rows["Y"]["impact_ratio"] == pytest.approx(0.4 / 0.6)
    assert not rows["Y"]["tested"] and rows["Y"]["p_raw"] is None  # 25 < min_test_n
    assert rows["B"]["tested"]


def test_a_group_with_no_one_left_keeps_its_row():
    # X's two applicants both fail the screen: a missing technical row would say so; an n = 0 row does not.
    rows = stage({"A": (100, 60), "X": (2, 0)}) + stage({"A": (60, 40)}, name="technical")
    technical = by_group(impact_ratios(pl.DataFrame(rows), PLAN), at="technical")
    assert set(technical) == {"A", "X"}
    assert technical["X"]["n"] == 0 and technical["X"]["rate"] is None


def test_intersectional_ratios_skip_unknown_and_small_share_cells():
    frame = pl.DataFrame(stage({"A": (300, 150), "B": (300, 120), "Unknown": (100, 90), "X": (5, 5)}))
    rows = {r["sex"]: r for r in intersectional(frame, PLAN).to_dicts()}
    assert rows["B"]["impact_ratio"] == pytest.approx(0.4 / 0.5)
    assert rows["Unknown"]["impact_ratio"] is None
    assert rows["X"]["impact_ratio"] is None and rows["X"]["excluded_small_share"]


def test_publishing_blanks_hidden_rows_and_compares_with_the_top_shown_group():
    frame = pl.DataFrame(stage({"A": (100, 60), "B": (100, 40), "C": (100, 50), "D": (3, 1)}))
    fair = impact_ratios(frame, PLAN).with_columns(
        pl.when(pl.col("tested")).then(0.004).alias("q"), pl.lit(False).alias("flagged")
    )
    hidden = {"ir|r|screen|sex|A", "ir|r|screen|sex|D"}
    rows = by_group(publish_impact(fair, hidden, PLAN))
    blank = rows["A"]
    assert blank["suppressed"] and all(blank[c] is None for c in ("n", "rate", "impact_ratio", "h", "q_band"))
    # A is hidden, so the published reference is C, and nothing on the page lets a reader recover A's rate.
    assert rows["B"]["reference_group"] == "C" and rows["B"]["impact_ratio"] == pytest.approx(0.4 / 0.5)
    assert rows["B"]["h"] == pytest.approx(round(cohens_h(0.4, 0.5), 2))
    assert rows["B"]["h_ci_low"] < rows["B"]["h"] < rows["B"]["h_ci_high"]
    assert rows["C"]["h"] is None  # the reference is not compared with itself
    assert rows["B"]["q_band"] == "<0.01"
    assert "share" not in rows["B"] and "p_raw" not in rows["B"] and "q" not in rows["B"]


def test_q_is_published_as_a_band():
    assert [q_band(q) for q in (None, 0.0009, 0.001, 0.0099, 0.01, 0.049, 0.05, 0.4)] == [
        None,
        "<0.001",
        "<0.01",
        "<0.01",
        "<0.05",
        "<0.05",
        ">=0.05",
        ">=0.05",
    ]


def test_intersectional_publishing_uses_the_top_shown_cell():
    frame = pl.DataFrame(stage({"A": (100, 60), "B": (100, 40), "C": (100, 50)}))
    cross = intersectional(frame, PLAN)
    rows = {r["sex"]: r for r in publish_intersectional(cross, {"ix|r|screen|A|White"}, PLAN).to_dicts()}
    assert rows["A"]["suppressed"] and rows["A"]["rate"] is None
    assert rows["B"]["impact_ratio"] == pytest.approx(0.4 / 0.5)
