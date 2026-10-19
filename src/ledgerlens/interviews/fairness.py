"""Selection rates and impact ratios per stage (BUILD_PROMPT 11.2, NYC LL144-style; ADR 0004).

The only module that sees group attributes, and only inside the pipeline's audit step. It returns group-level
counts and rates, never people:

- selection rate = passed / entered, per role x stage x dimension x group;
- impact ratio = a group's rate / the highest eligible group's rate;
- groups under ``min_group_share`` of the stage's candidates are excluded from impact ratios (noted);
- the significance test and Cohen's h compare a group with all other candidates at that stage (not with the
  highest group, which is picked after looking and would inflate false positives); both sides need
  >= min_test_n.
"""

import numpy as np
import polars as pl

from ledgerlens.core.guards import Funnel
from ledgerlens.core.stats import Estimate, cohens_h, fisher_p, seed_for
from ledgerlens.interviews.generate import STAGES
from ledgerlens.interviews.plan import AnalysisPlan

# Suppression policy for the published selection tables (ADR 0004). Group sizes in one stage add up to a
# known total, and a group's n at one stage is how many of it passed the stage before.
IMPACT_LINES = [("role_id", "stage", "dimension")]
IMPACT_FUNNEL = Funnel("stage", STAGES, ("role_id", "dimension", "group"))
INTERSECTIONAL_LINES = [("role_id", "stage", "sex"), ("role_id", "stage", "race_ethnicity")]
INTERSECTIONAL_FUNNEL = Funnel("stage", STAGES, ("role_id", "sex", "race_ethnicity"))

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
    "h": pl.Float64,
    "h_ci_low": pl.Float64,
    "h_ci_high": pl.Float64,
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


def _h_interval(sel_a: int, n_a: int, sel_b: int, n_b: int, rng: np.random.Generator, draws: int) -> Estimate:
    p_a, p_b = sel_a / n_a, sel_b / n_b
    sims_a = rng.binomial(n_a, p_a, draws) / n_a
    sims_b = rng.binomial(n_b, p_b, draws) / n_b
    hs = 2 * np.arcsin(np.sqrt(sims_a)) - 2 * np.arcsin(np.sqrt(sims_b))
    low, high = np.quantile(hs, [0.025, 0.975])
    return Estimate(cohens_h(p_a, p_b), float(low), float(high))


def selection_cells(outcomes: pl.DataFrame, group_cols: list[str]) -> pl.DataFrame:
    """Counts per role x stage x group. ``outcomes`` needs role_id, stage, passed and the group columns."""
    return (
        outcomes.group_by(["role_id", "stage", *group_cols])
        .agg(pl.len().alias("n"), pl.col("passed").sum().alias("selected"))
        .with_columns((pl.col("selected") / pl.col("n")).alias("rate"))
    )


def impact_ratios(outcomes: pl.DataFrame, plan: AnalysisPlan) -> pl.DataFrame:
    """One row per role x stage x dimension x group with rate, ratio, test and effect size."""
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
                key = f"ir|{role}|{stage}|{dimension}|{r['group']}"
                row = {
                    "row_key": key,
                    "role_id": str(role),
                    "stage": str(stage),
                    "dimension": dimension,
                    "group": str(r["group"]),
                    "n": int(r["n"]),
                    "selected": int(r["selected"]),
                    "rate": float(r["rate"]),
                    "share": r["n"] / total,
                    "excluded_small_share": excluded,
                    "reference_group": None,
                    "impact_ratio": None,
                    "h": None,
                    "h_ci_low": None,
                    "h_ci_high": None,
                    "p_raw": None,
                    "tested": False,
                }
                if reference is not None and not excluded and r["group"] != "Unknown":
                    row["reference_group"] = str(reference["group"])
                    row["impact_ratio"] = r["rate"] / reference["rate"] if reference["rate"] > 0 else None
                    rest_n = total - int(r["n"])
                    rest_sel = int(group["selected"].sum()) - int(r["selected"])
                    if r["n"] >= plan.fairness.min_test_n and rest_n >= plan.fairness.min_test_n:
                        est = _h_interval(
                            int(r["selected"]),
                            int(r["n"]),
                            rest_sel,
                            rest_n,
                            seed_for(plan.seed, key),
                            plan.bootstrap_resamples,
                        )
                        row.update(
                            h=est.value,
                            h_ci_low=est.ci_low,
                            h_ci_high=est.ci_high,
                            tested=True,
                            p_raw=fisher_p(int(r["selected"]), int(r["n"]), rest_sel, rest_n),
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
                    "rate": float(r["rate"]),
                    "excluded_small_share": excluded,
                    "impact_ratio": (r["rate"] / best)
                    if (best > 0 and not excluded and not unknown)
                    else None,
                }
            )
    return pl.DataFrame(rows, schema=INTERSECTIONAL_SCHEMA).sort("role_id", "stage", "sex", "race_ethnicity")
