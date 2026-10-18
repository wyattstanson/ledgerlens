"""Total-cost normalization to USD per baseline hour (BUILD_PROMPT section 12.2, step 3).

- Time and materials: rate per hour.
- Monthly FTE: monthly rate / standard hours (default 160).
- Fixed-price SOW: amount / reported hours when reported; otherwise kept in spend but out of rate benchmarks.
- Outcome-based: rate x (1 - productivity commitment). $110 with a 20% commitment = $88 per baseline hour.

USD amounts on invoice lines use the exchange rate stamped at posting (ADR 0001); rate cards convert at the
as-of month's rate.
"""

from datetime import date

import polars as pl

from ledgerlens.ratecards.config import NormalizeConfig

MAPPED = ("accepted", "reviewed")


def _months_between(start: date, as_of_month: str) -> int:
    year, month = map(int, as_of_month.split("-"))
    return (year * 12 + month) - (start.year * 12 + start.month)


def role_lookup(title_map: pl.DataFrame) -> pl.DataFrame:
    """billed_title -> role_id for titles that are mapped (auto-accepted or reviewed)."""
    return title_map.filter(pl.col("status").is_in(MAPPED)).select("billed_title", "role_id")


def _effective(sticker: pl.Expr) -> pl.Expr:
    commitment = pl.col("productivity_commitment").fill_null(0.0)
    return pl.when(pl.col("pricing_model") == "outcome").then(sticker * (1 - commitment)).otherwise(sticker)


def normalize_cards(
    cards: pl.DataFrame, title_map: pl.DataFrame, fx: pl.DataFrame, as_of_month: str, config: NormalizeConfig
) -> pl.DataFrame:
    rates = fx.filter(pl.col("month") == as_of_month).select("currency", "usd_per_unit")
    out = (
        cards.join(rates, on="currency", how="left")
        .join(role_lookup(title_map).rename({"billed_title": "vendor_title"}), on="vendor_title", how="left")
        .with_columns(
            pl.when(pl.col("pricing_model").is_in(["tm", "outcome"]))
            .then(pl.col("rate") * pl.col("usd_per_unit"))
            .when(pl.col("pricing_model") == "fte")
            .then(pl.col("rate") * pl.col("usd_per_unit") / config.standard_hours)
            .otherwise(None)
            .alias("sticker_usd_per_hour")
        )
        .with_columns(_effective(pl.col("sticker_usd_per_hour")).alias("effective_usd_per_hour"))
        .with_columns(
            pl.col("contract_start")
            .map_elements(lambda d: _months_between(d, as_of_month), return_dtype=pl.Int64)
            .alias("contract_age_months")
        )
        .with_columns(
            (pl.col("contract_age_months") > config.renegotiation_age_months).alias("renegotiation_candidate")
        )
    )
    return out.drop("usd_per_unit")


def normalize_lines(
    lines: pl.DataFrame, cards: pl.DataFrame, title_map: pl.DataFrame, config: NormalizeConfig
) -> pl.DataFrame:
    card_cols = cards.select(
        pl.col("vendor_id").alias("entity"),
        pl.col("vendor_title").alias("billed_title"),
        "card_id",
        pl.col("rate").alias("card_rate"),
        "productivity_commitment",
    )
    quantity, amount = pl.col("quantity"), pl.col("amount")
    sticker = (
        pl.when(pl.col("pricing_model").is_in(["tm", "outcome"]))
        .then(amount / quantity)
        .when(pl.col("pricing_model") == "fte")
        .then(amount / (quantity * config.standard_hours))
        .otherwise(amount / pl.col("reported_hours"))  # null when hours are not reported
    )
    return (
        lines.join(card_cols, on=["entity", "billed_title"], how="left")
        .join(role_lookup(title_map), on="billed_title", how="left")
        .with_columns(sticker.alias("sticker_usd_per_hour"))
        .with_columns(_effective(pl.col("sticker_usd_per_hour")).alias("effective_usd_per_hour"))
        .with_columns((amount / pl.col("effective_usd_per_hour")).alias("baseline_hours"))
        .with_columns(
            ((pl.col("card_rate") * quantity).round(2) * pl.col("exchangerate"))
            .round(2)
            .alias("card_amount_usd")
        )
        .with_columns(
            (amount - pl.col("card_amount_usd")).round(2).alias("excess_usd"),
            (pl.col("rate") / pl.col("card_rate") - 1).alias("billed_vs_card"),
        )
    )
