"""Interview process metrics (BUILD_PROMPT 11.2): pure functions over vault frames that return aggregates.

Nothing here reads group attributes. Every output row describes an interviewer, panel, role or loop set,
never a candidate.
"""

from collections import defaultdict

import numpy as np
import polars as pl

from ledgerlens.core.stats import (
    bootstrap_cohens_d,
    fleiss_kappa,
    seed_for,
    spearman,
    welch_p,
)
from ledgerlens.interviews.plan import AnalysisPlan

MIN_PEERS = 2  # a z-score needs a spread of at least two peer means
MIN_KAPPA_SUBJECTS = 2
MIN_RANK_N = 3  # Spearman's rho on fewer points is meaningless

# Explicit schemas keep every column when a plan's floors leave no eligible rows.
CALIBRATION_SCHEMA = {
    "row_key": pl.String,
    "interviewer_id": pl.String,
    "role_id": pl.String,
    "competency_id": pl.String,
    "n": pl.Int64,
    "mean": pl.Float64,
    "peer_mean": pl.Float64,
    "peer_sd": pl.Float64,
    "z": pl.Float64,
    "d": pl.Float64,
    "d_ci_low": pl.Float64,
    "d_ci_high": pl.Float64,
    "p_raw": pl.Float64,
}
PANEL_SCHEMA = {
    "row_key": pl.String,
    "panel_id": pl.String,
    "role_id": pl.String,
    "n": pl.Int64,
    "members": pl.Int64,
    "kappa_mean": pl.Float64,
    "kappa_min": pl.Float64,
}
VALIDITY_SCHEMA = {
    "row_key": pl.String,
    "role_id": pl.String,
    "n": pl.Int64,
    "rho": pl.Float64,
    "ci_low": pl.Float64,
    "ci_high": pl.Float64,
}


def calibration(ratings: pl.DataFrame, plan: AnalysisPlan) -> pl.DataFrame:
    """Per interviewer x role x competency: mean vs the spread of peer interviewers' means (leave-one-out).

    ``ratings`` needs interviewer_id, role_id, competency_id, stage, rating. Ratings are first centred on
    their role x competency x stage mean: later stages only see candidates who passed earlier ones, so raw
    means would flag hiring managers for meeting stronger candidates. z compares this interviewer's centred
    mean to the mean and SD of the other interviewers' centred means; the Welch test and Cohen's d (with a
    bootstrap CI) compare this interviewer's centred ratings with the pooled peer ratings.
    """
    min_n = plan.calibration.min_n
    keys = ["interviewer_id", "role_id", "competency_id"]
    centred = ratings.with_columns(
        (pl.col("rating") - pl.col("rating").mean().over(["role_id", "competency_id", "stage"])).alias(
            "residual"
        )
    ).sort([*keys, "residual"])  # a canonical order, so the seeded bootstrap draws the same values every run
    cells: dict[tuple[str, str], dict[str, np.ndarray]] = defaultdict(dict)
    for (iid, role, comp), group in centred.group_by(keys, maintain_order=True):
        cells[(str(role), str(comp))][str(iid)] = group["residual"].to_numpy().astype(float)
    rows = []
    for (role, comp), by_interviewer in sorted(cells.items()):
        eligible = {i: v for i, v in by_interviewer.items() if len(v) >= min_n}
        for iid, own in sorted(eligible.items()):
            peers = {i: eligible[i] for i in sorted(eligible) if i != iid}
            if len(peers) < MIN_PEERS:
                continue
            peer_means = np.array([v.mean() for v in peers.values()])
            pooled = np.concatenate(list(peers.values()))
            peer_sd = float(peer_means.std(ddof=1))
            z = float((own.mean() - peer_means.mean()) / peer_sd) if peer_sd > 0 else 0.0
            key = f"cal|{iid}|{role}|{comp}"
            d = bootstrap_cohens_d(
                own, pooled, rng=seed_for(plan.seed, key), n_resamples=plan.bootstrap_resamples
            )
            rows.append(
                {
                    "row_key": key,
                    "interviewer_id": iid,
                    "role_id": role,
                    "competency_id": comp,
                    "n": len(own),
                    "mean": float(own.mean()),
                    "peer_mean": float(peer_means.mean()),
                    "peer_sd": peer_sd,
                    "z": z,
                    "d": d.value,
                    "d_ci_low": d.ci_low,
                    "d_ci_high": d.ci_high,
                    "p_raw": welch_p(own, pooled),
                }
            )
    return pl.DataFrame(rows, schema=CALIBRATION_SCHEMA)


def panel_agreement(ratings: pl.DataFrame) -> pl.DataFrame:
    """Mean Fleiss' kappa across competencies for each onsite panel (candidates rated by every member)."""
    onsite = ratings.filter(pl.col("panel_id").is_not_null())
    rows = []
    for (panel, role), group in onsite.group_by(["panel_id", "role_id"]):
        kappas, n_candidates = [], 0
        members = sorted(group["interviewer_id"].unique().to_list())
        for _, comp_group in group.group_by("competency_id"):
            wide = comp_group.pivot(on="interviewer_id", index="candidate_id", values="rating").drop_nulls()
            matrix = (
                wide.select(members).to_numpy()
                if all(m in wide.columns for m in members)
                else np.empty((0, 0))
            )
            if matrix.shape[0] >= MIN_KAPPA_SUBJECTS:
                kappas.append(fleiss_kappa(matrix))
                n_candidates = max(n_candidates, matrix.shape[0])
        rows.append(
            {
                "row_key": f"panel|{panel}",
                "panel_id": str(panel),
                "role_id": str(role),
                "n": n_candidates,
                "members": len(members),
                "kappa_mean": float(np.mean(kappas)) if kappas else None,
                "kappa_min": float(np.min(kappas)) if kappas else None,
            }
        )
    return pl.DataFrame(rows, schema=PANEL_SCHEMA).sort("panel_id")


def process_quality(ratings: pl.DataFrame, evidence: dict[str, bool], required: pl.DataFrame) -> pl.DataFrame:
    """Rubric coverage and evidence quality per interviewer, loop coverage per role.

    ``ratings`` needs rating_id, note_id, interview_id, candidate_id, interviewer_id, role_id, competency_id,
    question_asked. ``required`` maps role_id -> competency_id.
    """
    tagged = ratings.with_columns(
        pl.col("note_id")
        .replace_strict(evidence, default=False, return_dtype=pl.Boolean)
        .alias("has_evidence")
    )
    per_interview = tagged.group_by(["interview_id", "interviewer_id"]).agg(
        (pl.col("question_asked") & pl.col("has_evidence")).mean().alias("covered_share"),
    )
    rubric = per_interview.group_by("interviewer_id").agg(
        pl.len().alias("n"), pl.col("covered_share").mean().alias("value")
    )
    evidence_q = tagged.group_by("interviewer_id").agg(
        pl.len().alias("n"), pl.col("has_evidence").mean().alias("value")
    )
    required_n = required.group_by("role_id").agg(pl.len().alias("required"))
    loops = (
        tagged.filter(pl.col("question_asked"))
        .group_by(["candidate_id", "role_id"])
        .agg(pl.col("competency_id").n_unique().alias("covered"))
        .join(required_n, on="role_id")
        .group_by("role_id")
        .agg(pl.len().alias("n"), (pl.col("covered") >= pl.col("required")).mean().alias("value"))
    )
    frames = [
        rubric.select(
            pl.lit("rubric_coverage").alias("metric"),
            pl.lit("interviewer").alias("scope"),
            pl.col("interviewer_id").alias("subject_id"),
            "n",
            "value",
        ),
        evidence_q.select(
            pl.lit("evidence_quality").alias("metric"),
            pl.lit("interviewer").alias("scope"),
            pl.col("interviewer_id").alias("subject_id"),
            "n",
            "value",
        ),
        loops.select(
            pl.lit("loop_coverage").alias("metric"),
            pl.lit("role").alias("scope"),
            pl.col("role_id").alias("subject_id"),
            "n",
            "value",
        ),
    ]
    out = pl.concat(frames).with_columns(
        pl.concat_str([pl.lit("proc"), "metric", "subject_id"], separator="|").alias("row_key")
    )
    return out.sort("metric", "subject_id")


def predictive_validity(scores: pl.DataFrame, plan: AnalysisPlan) -> pl.DataFrame:
    """Spearman's rho between onsite interview score and later performance, per role, with a bootstrap CI.

    ``scores`` needs role_id, interview_score, performance_rating (one row per hire).
    """
    rows = []
    ordered = scores.sort("role_id", "interview_score", "performance_rating")
    for (role,), group in ordered.group_by(["role_id"], maintain_order=True):
        x = group["interview_score"].to_numpy().astype(float)
        y = group["performance_rating"].to_numpy().astype(float)
        key = f"pv|{role}"
        rng = seed_for(plan.seed, key)
        rho = spearman(x, y)
        low, high = np.nan, np.nan
        if len(x) >= MIN_RANK_N:
            draws = [
                spearman(x[idx], y[idx])
                for idx in rng.integers(0, len(x), (plan.bootstrap_resamples, len(x)))
            ]
            low, high = np.nanquantile(draws, [0.025, 0.975])
        rows.append(
            {
                "row_key": key,
                "role_id": str(role),
                "n": len(x),
                "rho": rho,
                "ci_low": float(low),
                "ci_high": float(high),
            }
        )
    return pl.DataFrame(rows, schema=VALIDITY_SCHEMA).sort("role_id")
