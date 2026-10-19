"""Selection rates and impact ratios per stage (BUILD_PROMPT 11.2, NYC LL144-style; ADR 0004).

The only module that sees group attributes, and only inside the pipeline's audit step. It returns group-level
counts and rates, never people:

- selection rate = passed / entered, per role x stage x dimension x group;
- impact ratio = a group's rate / the highest eligible group's rate;
- groups under ``min_group_share`` of the stage's candidates are excluded from impact ratios (noted);
- the significance test compares a group with all other candidates at that stage (not with the highest
  group, which is picked after looking and would inflate false positives); both sides need >= min_test_n.

Every group that applied to a role gets a row at every stage (n may be 0), so a missing row never tells a
reader that a group had no one left.

Publishing (after the disclosure audit decides what is hidden) recomputes impact ratios and Cohen's h against
the highest-rate *published* group, so neither reveals anything about a hidden one. Statistics against
everyone else at the stage (h, exact q) would reveal the stage total and with it the small groups, so the
test's q is published only as a band.
"""

import numpy as np
import polars as pl

from ledgerlens.core.stats import Estimate, cohens_h, fisher_p, seed_for
from ledgerlens.interviews.disclosure import H_DECIMALS
from ledgerlens.interviews.generate import STAGES
from ledgerlens.interviews.plan import AnalysisPlan

IMPACT_SCHEMA = {
    "row_key": pl.String,
    "role_id": pl.String,
    "stage": pl.String,
    "dimension": pl.String,
    "group": pl.String,
    "n": pl.Int64,
    "selected": pl.Int64,
    "rate": pl.Float64,
    "share": pl.Float64,
    "excluded_small_share": pl.Boolean,
    "reference_group": pl.String,
    "impact_ratio": pl.Float64,
    "p_raw": pl.Float64,
    "tested": pl.Boolean,
}
INTERSECTIONAL_SCHEMA = {
    "row_key": pl.String,
    "role_id": pl.String,
    "stage": pl.String,
    "sex": pl.String,
    "race_ethnicity": pl.String,
    "n": pl.Int64,
    "selected": pl.Int64,
    "rate": pl.Float64,
    "excluded_small_share": pl.Boolean,
    "impact_ratio": pl.Float64,
}
IMPACT_KEYS = ("row_key", "role_id", "stage", "dimension", "group")
INTERSECTIONAL_KEYS = ("row_key", "role_id", "stage", "sex", "race_ethnicity")
PUBLISHED_IMPACT = (
    *IMPACT_KEYS,
    "n",
    "selected",
    "rate",
    "excluded_small_share",
    "reference_group",
    "impact_ratio",
    "h",
    "h_ci_low",
    "h_ci_high",
    "tested",
    "q_band",
    "flagged",
)
PUBLISHED_INTERSECTIONAL = (
    *INTERSECTIONAL_KEYS,
    "n",
    "selected",
    "rate",
    "excluded_small_share",
    "impact_ratio",
)
Q_BANDS = ((0.001, "<0.001"), (0.01, "<0.01"), (0.05, "<0.05"))


def _h_interval(sel_a: int, n_a: int, sel_b: int, n_b: int, rng: np.random.Generator, draws: int) -> Estimate:
    p_a, p_b = sel_a / n_a, sel_b / n_b
    sims_a = rng.binomial(n_a, p_a, draws) / n_a
    sims_b = rng.binomial(n_b, p_b, draws) / n_b
    hs = 2 * np.arcsin(np.sqrt(sims_a)) - 2 * np.arcsin(np.sqrt(sims_b))
    low, high = np.quantile(hs, [0.025, 0.975])
    return Estimate(*(round(float(x), H_DECIMALS) for x in (cohens_h(p_a, p_b), low, high)))


def selection_cells(outcomes: pl.DataFrame, group_cols: list[str]) -> pl.DataFrame:
    """Counts per role x stage x group, with a row at every stage for each group that applied to the role.

    A group that reaches no later stage keeps a row there (n = 0), so a missing row cannot reveal that its
    members all failed. A group that never applied has no rows: that says nothing about any person.
    ``outcomes`` needs role_id, stage, passed and the group columns.
    """
    keys = ["role_id", "stage", *group_cols]
    counts = outcomes.group_by(keys).agg(pl.len().alias("n"), pl.col("passed").sum().alias("selected"))
    applied = outcomes.filter(pl.col("stage") == STAGES[0]).select("role_id", *group_cols).unique()
    grid = applied.join(outcomes.select("stage").unique(), how="cross")
    return (
        grid.join(counts, on=keys, how="left")
        .with_columns(pl.col("n").fill_null(0).cast(pl.Int64), pl.col("selected").fill_null(0).cast(pl.Int64))
        .with_columns(pl.when(pl.col("n") > 0).then(pl.col("selected") / pl.col("n")).alias("rate"))
    )


def impact_ratios(outcomes: pl.DataFrame, plan: AnalysisPlan) -> pl.DataFrame:
    """Analysis rows per role x stage x dimension x group: rate, ratio to the true highest group, test."""
    rows = []
    for dimension in plan.fairness.dimensions:
        cells = selection_cells(outcomes, [dimension]).rename({dimension: "group"})
        for (role, stage), group in cells.group_by(["role_id", "stage"]):
            total = int(group["n"].sum())
            records = group.sort("group").to_dicts()
            eligible = [
                r
                for r in records
                if r["n"] / total >= plan.min_group_share
                and r["n"] >= plan.min_group_n
                and r["group"] != "Unknown"
            ]
            reference = max(eligible, key=lambda r: (r["rate"], r["n"])) if eligible else None
            for r in records:
                excluded = r["n"] / total < plan.min_group_share
                row = {
                    "row_key": f"ir|{role}|{stage}|{dimension}|{r['group']}",
                    "role_id": str(role),
                    "stage": str(stage),
                    "dimension": dimension,
                    "group": str(r["group"]),
                    "n": int(r["n"]),
                    "selected": int(r["selected"]),
                    "rate": r["rate"],
                    "share": r["n"] / total,
                    "excluded_small_share": excluded,
                    "reference_group": None,
                    "impact_ratio": None,
                    "p_raw": None,
                    "tested": False,
                }
                if reference is not None and not excluded and r["group"] != "Unknown":
                    row["reference_group"] = str(reference["group"])
                    row["impact_ratio"] = r["rate"] / reference["rate"] if reference["rate"] > 0 else None
                    rest_n = total - int(r["n"])
                    rest_sel = int(group["selected"].sum()) - int(r["selected"])
                    if r["n"] >= plan.fairness.min_test_n and rest_n >= plan.fairness.min_test_n:
                        row.update(
                            tested=True, p_raw=fisher_p(int(r["selected"]), int(r["n"]), rest_sel, rest_n)
                        )
                rows.append(row)
    return pl.DataFrame(rows, schema=IMPACT_SCHEMA).sort("role_id", "stage", "dimension", "group")


def intersectional(outcomes: pl.DataFrame, plan: AnalysisPlan) -> pl.DataFrame:
    """Sex x race/ethnicity selection rates and impact ratios per role x stage (LL144 audit table)."""
    cells = selection_cells(outcomes, ["sex", "race_ethnicity"])
    rows = []
    for (role, stage), group in cells.group_by(["role_id", "stage"]):
        total = int(group["n"].sum())
        records = group.to_dicts()
        eligible = [
            r
            for r in records
            if r["n"] / total >= plan.min_group_share
            and r["n"] >= plan.min_group_n
            and "Unknown" not in (r["sex"], r["race_ethnicity"])
        ]
        best = max((r["rate"] for r in eligible), default=0.0)
        for r in records:
            excluded = r["n"] / total < plan.min_group_share
            unknown = "Unknown" in (r["sex"], r["race_ethnicity"])
            rows.append(
                {
                    "row_key": f"ix|{role}|{stage}|{r['sex']}|{r['race_ethnicity']}",
                    "role_id": str(role),
                    "stage": str(stage),
                    "sex": r["sex"],
                    "race_ethnicity": r["race_ethnicity"],
                    "n": int(r["n"]),
                    "selected": int(r["selected"]),
                    "rate": r["rate"],
                    "excluded_small_share": excluded,
                    "impact_ratio": (r["rate"] / best)
                    if (best > 0 and not excluded and not unknown)
                    else None,
                }
            )
    return pl.DataFrame(rows, schema=INTERSECTIONAL_SCHEMA).sort("role_id", "stage", "sex", "race_ethnicity")


def q_band(q: float | None) -> str | None:
    if q is None:
        return None
    return next((label for limit, label in Q_BANDS if q < limit), ">=0.05")


def _blank(row: dict, keys: tuple[str, ...], columns: tuple[str, ...]) -> dict:
    """A hidden row keeps only its labels."""
    return {c: (row.get(c) if c in keys else None) for c in columns} | {"suppressed": True}


def publish_impact(fair: pl.DataFrame, hidden: set[str], plan: AnalysisPlan) -> pl.DataFrame:
    """The published one-way table: hidden rows blanked, ratios and h against the highest published group."""
    out = []
    for _, line in fair.group_by(["role_id", "stage", "dimension"], maintain_order=True):
        records = line.sort("group").to_dicts()
        shown = [r for r in records if r["row_key"] not in hidden]
        eligible = [
            r
            for r in shown
            if r["n"] >= plan.min_group_n and not r["excluded_small_share"] and r["group"] != "Unknown"
        ]
        reference = max(eligible, key=lambda r: (r["rate"], r["n"])) if eligible else None
        for r in records:
            if r["row_key"] in hidden:
                out.append(_blank(r, IMPACT_KEYS, PUBLISHED_IMPACT))
                continue
            row = {c: r.get(c) for c in PUBLISHED_IMPACT} | {"suppressed": False}
            row.update(
                reference_group=None,
                impact_ratio=None,
                h=None,
                h_ci_low=None,
                h_ci_high=None,
                q_band=q_band(r.get("q")) if r["tested"] else None,
            )
            if reference is not None and not r["excluded_small_share"] and r["group"] != "Unknown":
                row["reference_group"] = reference["group"]
                row["impact_ratio"] = r["rate"] / reference["rate"] if reference["rate"] > 0 else None
                both_large = min(r["n"], reference["n"]) >= plan.fairness.min_test_n
                if r is not reference and both_large:
                    est = _h_interval(
                        r["selected"],
                        r["n"],
                        reference["selected"],
                        reference["n"],
                        seed_for(plan.seed, f"{r['row_key']}|ref"),
                        plan.bootstrap_resamples,
                    )
                    row.update(h=est.value, h_ci_low=est.ci_low, h_ci_high=est.ci_high)
            out.append(row)
    schema = {c: IMPACT_SCHEMA.get(c, pl.Float64) for c in PUBLISHED_IMPACT} | {
        "h": pl.Float64,
        "h_ci_low": pl.Float64,
        "h_ci_high": pl.Float64,
        "q_band": pl.String,
        "flagged": pl.Boolean,
        "suppressed": pl.Boolean,
    }
    return pl.DataFrame(out, schema=schema).sort("role_id", "stage", "dimension", "group")


def publish_intersectional(cross: pl.DataFrame, hidden: set[str], plan: AnalysisPlan) -> pl.DataFrame:
    """The published intersectional table: hidden rows blanked, ratios against the highest published cell."""
    out = []
    for _, block in cross.group_by(["role_id", "stage"], maintain_order=True):
        records = block.to_dicts()
        shown = [r for r in records if r["row_key"] not in hidden]
        best = max(
            (
                r["rate"]
                for r in shown
                if r["n"] >= plan.min_group_n
                and not r["excluded_small_share"]
                and "Unknown" not in (r["sex"], r["race_ethnicity"])
            ),
            default=0.0,
        )
        for r in records:
            if r["row_key"] in hidden:
                out.append(_blank(r, INTERSECTIONAL_KEYS, PUBLISHED_INTERSECTIONAL))
                continue
            unknown = "Unknown" in (r["sex"], r["race_ethnicity"])
            ratio = r["rate"] / best if best > 0 and not r["excluded_small_share"] and not unknown else None
            out.append(
                {c: r[c] for c in PUBLISHED_INTERSECTIONAL} | {"impact_ratio": ratio, "suppressed": False}
            )
    schema = {c: INTERSECTIONAL_SCHEMA[c] for c in PUBLISHED_INTERSECTIONAL} | {"suppressed": pl.Boolean}
    return pl.DataFrame(out, schema=schema).sort("role_id", "stage", "sex", "race_ethnicity")
