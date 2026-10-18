"""Rate benchmarks and the vendor scorecard (BUILD_PROMPT section 12.2, steps 4 and 5).

Benchmarks are p25 / median / p75 of contracted effective USD per baseline hour, per role and location
tier, with ``n`` vendors and a source label. External benchmarks are used only when
``data/external_benchmarks.csv`` exists (with its source and date); otherwise everything is "internal".
"""

from pathlib import Path

import polars as pl

from ledgerlens.ratecards.config import ScorecardWeights

EXTERNAL_COLUMNS = ("role_id", "location_tier", "p25", "median", "p75", "n", "source_name", "as_of")


def internal_benchmarks(normalized_cards: pl.DataFrame) -> pl.DataFrame:
    rate = pl.col("effective_usd_per_hour")
    return (
        normalized_cards.filter(pl.col("role_id").is_not_null() & rate.is_not_null())
        .group_by("role_id", "location_tier")
        .agg(
            pl.len().alias("n"),
            rate.quantile(0.25, interpolation="linear").alias("p25"),
            rate.median().alias("median"),
            rate.quantile(0.75, interpolation="linear").alias("p75"),
        )
        .with_columns(pl.lit("internal").alias("source"), pl.lit(None, dtype=pl.String).alias("source_name"))
        .sort("role_id", "location_tier")
    )


def load_external(path: Path) -> pl.DataFrame | None:
    if not path.exists():
        return None
    external = pl.read_csv(path)
    missing = set(EXTERNAL_COLUMNS) - set(external.columns)
    if missing:
        raise ValueError(f"{path.name} is missing columns {sorted(missing)}")
    return external.select(
        "role_id",
        "location_tier",
        pl.col("n").cast(pl.UInt32),
        "p25",
        "median",
        "p75",
        pl.lit("external").alias("source"),
        "source_name",
    )


def benchmarks(normalized_cards: pl.DataFrame, external_csv: Path | None = None) -> pl.DataFrame:
    internal = internal_benchmarks(normalized_cards)
    external = load_external(external_csv) if external_csv else None
    if external is None:
        return internal
    return pl.concat([internal, external.select(internal.columns)], how="vertical_relaxed")


def scorecard(
    normalized_cards: pl.DataFrame,
    normalized_lines: pl.DataFrame,
    vendors: pl.DataFrame,
    bench: pl.DataFrame,
    *,
    total_roles: int,
    baseline_months: list[str],
    weights: ScorecardWeights,
) -> pl.DataFrame:
    """Rate index (vendor rate / role median), quality, SLA, role coverage and spend share, weighted."""
    medians = bench.filter(pl.col("source") == "internal").select("role_id", "location_tier", "median")
    rate_index = (
        normalized_cards.filter(
            pl.col("role_id").is_not_null() & pl.col("effective_usd_per_hour").is_not_null()
        )
        .join(medians, on=["role_id", "location_tier"], how="inner")
        .group_by("vendor_id")
        .agg(
            (pl.col("effective_usd_per_hour") / pl.col("median")).mean().alias("rate_index"),
            pl.col("role_id").n_unique().alias("roles_covered"),
        )
    )
    spend = (
        normalized_lines.filter(pl.col("month").is_in(baseline_months))
        .group_by(pl.col("entity").alias("vendor_id"))
        .agg(pl.col("amount").sum().alias("baseline_spend_usd"))
    )
    total_spend = spend["baseline_spend_usd"].sum()
    table = (
        vendors.select("vendor_id", "name", "location_tier", "quality", "sla")
        .join(rate_index, on="vendor_id", how="left")
        .join(spend, on="vendor_id", how="left")
        .with_columns(
            pl.col("baseline_spend_usd").fill_null(0.0),
            pl.col("roles_covered").fill_null(0),
        )
        .with_columns(
            (pl.col("baseline_spend_usd") / total_spend).alias("spend_share"),
            (pl.col("roles_covered") / total_roles).alias("coverage"),
            (1.5 - pl.col("rate_index")).clip(0.0, 1.0).fill_null(0.0).alias("rate_score"),
        )
    )
    max_share = table["spend_share"].max() or 1.0
    return table.with_columns(
        (
            weights.rate * pl.col("rate_score")
            + weights.quality * pl.col("quality")
            + weights.sla * pl.col("sla")
            + weights.coverage * pl.col("coverage")
            + weights.spend_share * pl.col("spend_share") / max_share
        ).alias("score")
    ).sort("score", descending=True)
