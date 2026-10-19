"""The interviews audit job: the only code that reads the vault (ADR 0004).

Order: register the plan and log the run (before any compute) -> read the vault -> tag notes -> metrics and
fairness -> one Benjamini-Hochberg family across every calibration and fairness test -> pre-registered flags
-> small-cell suppression (a disclosure audit for the selection tables) -> write aggregates (with run id,
plan version and hash) to the warehouse.
"""

from dataclasses import dataclass
from pathlib import Path

import polars as pl

from ledgerlens.core.db import connect
from ledgerlens.core.guards import suppress_small_groups
from ledgerlens.core.plans import RegisteredPlan, analysis_run, register_plan
from ledgerlens.core.stats import benjamini_hochberg
from ledgerlens.interviews.disclosure import TOTAL_ROUNDING, Policy, plan_hidden
from ledgerlens.interviews.fairness import (
    impact_ratios,
    intersectional,
    publish_impact,
    publish_intersectional,
)
from ledgerlens.interviews.metrics import calibration, panel_agreement, predictive_validity, process_quality
from ledgerlens.interviews.plan import AnalysisPlan
from ledgerlens.interviews.tagger import (
    EvidenceTagger,
    NoteToTag,
    agreement,
    evaluation_sample,
    read_owner_labels,
)

TABLES = (
    "calibration",
    "interviewer_flags",
    "panel_agreement",
    "process_quality",
    "fairness",
    "intersectional",
    "predictive_validity",
    "tagger_quality",
    "run_meta",
)
# Tables whose rows are group cells with a size n (rule 2 applies to each row).
CELL_TABLES = (
    "calibration",
    "panel_agreement",
    "process_quality",
    "fairness",
    "intersectional",
    "predictive_validity",
)
# A tagger that fails on more notes than this would distort coverage metrics; the run fails instead.
MAX_TAGGER_FAILURE_SHARE = 0.05


@dataclass(frozen=True)
class VaultData:
    ratings: pl.DataFrame  # one row per rating, with interview context and note id
    outcomes: pl.DataFrame  # stage outcomes with group attributes
    hires: pl.DataFrame  # role_id, interview_score, performance_rating
    required: pl.DataFrame  # role_id, competency_id
    truth_labels: dict[str, bool]
    seed: int


def load_vault(vault: Path, warehouse: Path) -> VaultData:
    with connect(vault) as conn:
        ratings = conn.execute(
            """
            select r.rating_id, r.interview_id, r.competency_id, r.rating, r.question_asked,
                   n.note_id, n.text, i.candidate_id, i.role_id, i.stage, i.interviewer_id, i.panel_id
            from restricted.ratings r
            join restricted.interviews i using (interview_id)
            join restricted.notes n using (rating_id)
            order by r.rating_id
            """
        ).pl()
        outcomes = conn.execute(
            "select o.role_id, o.stage, o.passed, g.sex, g.race_ethnicity "
            "from restricted.stage_outcomes o join restricted.candidate_groups g using (candidate_id) "
            "order by o.candidate_id, o.stage"
        ).pl()
        hires = conn.execute(
            """
            select p.role_id, avg(r.rating) as interview_score,
                   any_value(p.performance_rating) as performance_rating
            from restricted.performance p
            join restricted.interviews i on i.candidate_id = p.candidate_id and i.stage = 'onsite'
            join restricted.ratings r using (interview_id)
            group by p.candidate_id, p.role_id
            order by p.candidate_id
            """
        ).pl()
        truth = dict(
            conn.execute("select note_id, has_evidence from restricted.truth_note_labels").fetchall()
        )
        seed_row = conn.execute("select seed from restricted.world_meta").fetchone()
    with connect(warehouse) as conn:
        required = conn.execute("select role_id, competency_id from raw_interviews.competencies").pl()
    return VaultData(ratings, outcomes, hires, required, truth, int(seed_row[0]) if seed_row else 0)


def flag_calibration(cal: pl.DataFrame, plan: AnalysisPlan) -> pl.DataFrame:
    """A cell is flagged only when |z|, the BH q-value and |Cohen's d| all pass the plan (ADR 0004)."""
    c = plan.calibration
    return cal.with_columns(
        (
            (pl.col("z").abs() > c.z_threshold)
            & (pl.col("q") < plan.alpha)
            & (pl.col("d").abs() >= c.min_effect_d)
        ).alias("flagged")
    )


def flag_interviewers(cal: pl.DataFrame, plan: AnalysisPlan) -> pl.DataFrame:
    """An interviewer is flagged in a role when at least ``min_flagged_cells`` of their cells there are."""
    return (
        cal.group_by(["interviewer_id", "role_id"])
        .agg(pl.len().alias("cells"), pl.col("flagged").sum().alias("flagged_cells"))
        .with_columns((pl.col("flagged_cells") >= plan.calibration.min_flagged_cells).alias("flagged"))
        .with_columns(
            pl.concat_str([pl.lit("ivf"), "interviewer_id", "role_id"], separator="|").alias("row_key")
        )
        .sort("interviewer_id", "role_id")
    )


def run_interviews(
    warehouse: Path,
    vault: Path,
    plan: AnalysisPlan,
    tagger: EvidenceTagger,
    *,
    owner_labels: Path | None = None,
) -> RegisteredPlan:
    registered = register_plan(warehouse, plan.name, plan)
    data = load_vault(vault, warehouse)
    with analysis_run(warehouse, registered, pack="interviews", seed=plan.seed) as run_id:
        notes = [
            NoteToTag(r["note_id"], r["text"], r["competency_id"])
            for r in data.ratings.select("note_id", "text", "competency_id").iter_rows(named=True)
        ]
        evidence = tagger.tag(notes)
        failures = int(getattr(tagger, "failures", 0))
        if failures > MAX_TAGGER_FAILURE_SHARE * len(notes):
            raise RuntimeError(f"the {tagger.name} tagger failed on {failures} of {len(notes)} notes")

        cal = calibration(data.ratings, plan)
        fair = impact_ratios(data.outcomes, plan)
        # One BH family per run: every calibration test plus every tested fairness cell.
        p_values = cal["p_raw"].to_list() + fair.filter(pl.col("tested"))["p_raw"].to_list()
        q_values = benjamini_hochberg(p_values)
        cal = flag_calibration(cal.with_columns(pl.Series("q", q_values[: cal.height])), plan)
        fair_q = iter(q_values[cal.height :])
        fair = fair.with_columns(
            pl.Series("q", [next(fair_q) if t else None for t in fair["tested"]], dtype=pl.Float64)
        ).with_columns(
            (
                pl.col("tested")
                & (pl.col("impact_ratio") < plan.fairness.impact_ratio_threshold)
                & (pl.col("q") < plan.alpha)
            )
            .fill_null(False)
            .alias("flagged")
        )
        interviewer_flags = flag_interviewers(cal, plan)
        panels = panel_agreement(data.ratings).with_columns(
            (
                (pl.col("n") >= plan.panel_agreement.min_candidates)
                & (pl.col("kappa_mean") < plan.panel_agreement.kappa_threshold)
            )
            .fill_null(False)
            .alias("flagged")
        )
        thresholds = {
            "rubric_coverage": plan.process.rubric_coverage_min,
            "evidence_quality": plan.process.evidence_quality_min,
            "loop_coverage": plan.process.loop_coverage_min,
        }
        process = (
            process_quality(data.ratings, evidence, data.required)
            .with_columns(
                pl.col("metric").replace_strict(thresholds, return_dtype=pl.Float64).alias("threshold")
            )
            .with_columns((pl.col("value") < pl.col("threshold")).alias("flagged"))
        )
        validity = predictive_validity(data.hires, plan).with_columns(
            (
                (pl.col("n") >= plan.predictive_validity.min_n)
                & (pl.col("ci_low") <= 0)
                & (pl.col("ci_high") >= 0)
            ).alias("flagged")
        )
        cross = intersectional(data.outcomes, plan)
        tagger_rows, stale_labels = _tagger_quality(evidence, data, plan, owner_labels)

        floor = plan.min_group_n
        # Hires per role are published exactly, so the audit treats them as known.
        hires = {r: n for r, n in validity.select("role_id", "n").iter_rows() if n >= floor}
        hidden = plan_hidden(fair, cross, hires, Policy(floor, floor))
        tables = {
            # Workload counts are rounded: exact ones would give away stage totals (ADR 0004).
            "calibration": round_counts(
                suppress(
                    cal,
                    ["mean", "peer_mean", "peer_sd", "z", "d", "d_ci_low", "d_ci_high", "p_raw", "q"],
                    floor,
                )
            ),
            "interviewer_flags": interviewer_flags,
            "panel_agreement": round_counts(suppress(panels, ["kappa_mean", "kappa_min"], floor)),
            "process_quality": round_counts(suppress(process, ["value", "flagged"], floor)),
            "fairness": publish_impact(fair, hidden, plan),
            "intersectional": publish_intersectional(cross, hidden, plan),
            "predictive_validity": suppress(validity, ["rho", "ci_low", "ci_high"], floor),
            "tagger_quality": tagger_rows,
        }
        tables["run_meta"] = pl.DataFrame(
            [
                {
                    "row_key": "meta",
                    "tests_in_family": len(p_values),
                    "fairness_tests": int(fair["tested"].sum()),
                    "hidden_selection_cells": len(hidden),
                    "tagger": tagger.name,
                    "tagger_failures": failures,
                    "owner_labels_ignored": stale_labels,
                    "world_seed": data.seed,
                    # The thresholds documents quote, so their text follows the plan.
                    "min_group_n": plan.min_group_n,
                    "min_group_share": plan.min_group_share,
                    "impact_ratio_threshold": plan.fairness.impact_ratio_threshold,
                    "alpha": plan.alpha,
                    "flagged_interviewers": interviewer_flags.filter(pl.col("flagged"))[
                        "interviewer_id"
                    ].n_unique(),
                    "flagged_panels": int(panels["flagged"].sum()),
                    "flagged_fairness_cells": int(fair["flagged"].sum()),
                    "nonpredictive_roles": int(validity["flagged"].sum()),
                    "process_flags": int(process["flagged"].sum()),
                }
            ]
        )
        stamp = [
            pl.lit(run_id).alias("analysis_run_id"),
            pl.lit(registered.version).alias("plan_version"),
            pl.lit(registered.sha256).alias("plan_sha256"),
        ]
        with connect(warehouse) as conn:
            conn.execute("begin transaction")  # readers never see a mix of two runs' tables
            conn.execute("create schema if not exists interviews")
            for name, frame in tables.items():
                stamped = frame.with_columns(stamp)  # noqa: F841 (read by DuckDB's replacement scan)
                conn.execute(f"create or replace table interviews.{name} as select * from stamped")  # noqa: S608
            conn.execute("commit")
    return registered


def suppress(frame: pl.DataFrame, value_cols: list[str], floor: int) -> pl.DataFrame:
    """Null every value (and n) for rows below the floor; adds ``suppressed``."""
    if frame.is_empty():
        return frame.with_columns(pl.lit(False).alias("suppressed"))
    rows = suppress_small_groups(frame.to_dicts(), value_keys=value_cols, min_n=floor)
    return pl.DataFrame(rows, schema=frame.schema | {"suppressed": pl.Boolean})


def round_counts(frame: pl.DataFrame, base: int = TOTAL_ROUNDING) -> pl.DataFrame:
    """Publish n to the nearest ``base`` (half up); suppressed rows stay null."""
    return frame.with_columns(((pl.col("n") + base // 2) // base * base).alias("n"))


def _tagger_quality(
    evidence: dict[str, bool], data: VaultData, plan: AnalysisPlan, owner_labels: Path | None
) -> tuple[pl.DataFrame, int]:
    """Agreement table, plus how many owner labels were ignored because their note text no longer matches
    (a sheet written for another world or seed)."""
    sample = evaluation_sample(list(data.truth_labels), plan.tagger.eval_sample, plan.seed)
    truth = {k: data.truth_labels[k] for k in sample}
    tagged = {k: evidence[k] for k in sample}
    texts = dict(data.ratings.select("note_id", "text").iter_rows())
    owner = read_owner_labels(owner_labels, notes=texts) if owner_labels else {}
    stale = len(read_owner_labels(owner_labels)) - len(owner) if owner_labels else 0
    comparisons = [("tagger_vs_generator_truth", *agreement(tagged, truth))]
    if owner:
        comparisons += [
            ("tagger_vs_owner", *agreement(tagged, owner)),
            ("owner_vs_generator_truth", *agreement(owner, truth)),
        ]
    frame = pl.DataFrame(
        [
            {
                "row_key": f"tq|{name}",
                "comparison": name,
                "notes": n,
                "kappa": kappa,
                "target": plan.tagger.kappa_target,
            }
            for name, n, kappa in comparisons
        ],
        schema={
            "row_key": pl.String,
            "comparison": pl.String,
            "notes": pl.Int64,
            "kappa": pl.Float64,
            "target": pl.Float64,
        },
    )
    return frame, stale


def spot_check_notes(vault: Path, plan: AnalysisPlan) -> list[NoteToTag]:
    """The blind labelling set: the first spot_check_size ids of the seeded evaluation sample."""
    with connect(vault) as conn:
        rows = conn.execute(
            "select n.note_id, n.text, r.competency_id "
            "from restricted.notes n join restricted.ratings r using (rating_id)"
        ).fetchall()
    by_id = {note_id: NoteToTag(note_id, text, comp) for note_id, text, comp in rows}
    sample = evaluation_sample(list(by_id), plan.tagger.eval_sample, plan.seed)
    return [by_id[k] for k in sample[: plan.tagger.spot_check_size]]
