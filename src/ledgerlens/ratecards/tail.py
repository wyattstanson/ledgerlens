"""Tail plan (BUILD_PROMPT section 12.2, step 9): what to do with vendors that are not on the panel.

Rules, deterministic:
- Below the quality floor: exit.
- Any mapped role priced at or below the role/tier median: negotiate and keep as a backup, with a target
  rate (p25) and a walk-away rate (median) per role.
- Otherwise: exit, since panel vendors cover every role at lower cost.

The negotiation drafter agent writes the brief; these numbers are its only source.
"""

import polars as pl


def tail_plan(
    scorecard: pl.DataFrame,
    normalized_cards: pl.DataFrame,
    bench: pl.DataFrame,
    *,
    kept: set[str],
    min_quality: float,
) -> pl.DataFrame:
    internal = bench.filter(pl.col("source") == "internal").select(
        "role_id", "location_tier", "p25", "median"
    )
    priced = (
        normalized_cards.filter(
            ~pl.col("vendor_id").is_in(list(kept))
            & pl.col("role_id").is_not_null()
            & pl.col("effective_usd_per_hour").is_not_null()
        )
        .join(internal, on=["role_id", "location_tier"], how="inner")
        .select(
            "vendor_id",
            "role_id",
            "card_id",
            "effective_usd_per_hour",
            pl.col("p25").alias("target_usd_per_hour"),
            pl.col("median").alias("walk_away_usd_per_hour"),
        )
    )
    competitive = priced.filter(pl.col("effective_usd_per_hour") <= pl.col("walk_away_usd_per_hour"))
    vendors = scorecard.filter(~pl.col("vendor_id").is_in(list(kept))).select(
        "vendor_id", "name", "quality", "baseline_spend_usd", "roles_covered"
    )
    competitive_vendors = set(competitive["vendor_id"])
    decision = (
        pl.when(pl.col("quality") < min_quality)
        .then(pl.lit("exit: below quality floor"))
        .when(pl.col("vendor_id").is_in(list(competitive_vendors)))
        .then(pl.lit("negotiate: backup supplier"))
        .otherwise(pl.lit("exit: panel covers its roles at lower cost"))
    )
    plan = vendors.with_columns(decision.alias("proposal")).sort("baseline_spend_usd", descending=True)
    briefs = competitive.join(plan.select("vendor_id", "proposal"), on="vendor_id").filter(
        pl.col("proposal").str.starts_with("negotiate")
    )
    return plan.join(
        briefs.group_by("vendor_id").agg(pl.struct(pl.exclude("vendor_id", "proposal")).alias("briefs")),
        on="vendor_id",
        how="left",
    )
