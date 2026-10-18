"""Leakage detection (BUILD_PROMPT section 12.2, step 8): lines billed above the contracted card, and
duplicate invoice lines. Both feed the leakage-recovery step of the savings bridge.

- Drift / overbilling: billed rate more than ``threshold`` (default 2%) above the card, compared in the
  card's own currency so FX can never trigger a flag.
- Duplicates: the same vendor, bill, title, quantity and amount more than once; every copy after the
  first (by line id) is a duplicate. A duplicate's full amount is leakage, so it is never also counted as
  an overbill.
"""

import polars as pl

DUPLICATE_KEY = ("entity", "tranid", "billed_title", "quantity", "foreignamount")
# 102 / 100 - 1 is 0.020000000000000018 in binary floating point; exactly 2% must not count as "more than 2%".
RATIO_EPSILON = 1e-9


def find_duplicates(lines: pl.DataFrame) -> pl.DataFrame:
    """line_id of each duplicate copy, with the line it duplicates."""
    ordered = lines.sort("line_id").with_columns(
        pl.col("line_id").first().over(DUPLICATE_KEY).alias("original_line_id"),
        pl.int_range(pl.len()).over(DUPLICATE_KEY).alias("copy_index"),
    )
    return ordered.filter(pl.col("copy_index") > 0).select("line_id", "original_line_id")


def find_leakage(normalized_lines: pl.DataFrame, threshold: float) -> pl.DataFrame:
    """One row per leaking line: kind, vendor, excess USD (overbill) or full amount (duplicate)."""
    duplicates = find_duplicates(normalized_lines)
    dup_rows = normalized_lines.join(duplicates, on="line_id", how="inner").select(
        "line_id",
        pl.lit("duplicate").alias("kind"),
        pl.col("entity").alias("vendor_id"),
        "billed_title",
        "month",
        pl.col("amount").alias("leakage_usd"),
        "original_line_id",
        pl.lit(None, dtype=pl.Float64).alias("billed_vs_card"),
    )
    over_rows = (
        normalized_lines.join(duplicates, on="line_id", how="anti")
        .filter(pl.col("billed_vs_card") > threshold + RATIO_EPSILON)
        .select(
            "line_id",
            pl.lit("overbill").alias("kind"),
            pl.col("entity").alias("vendor_id"),
            "billed_title",
            "month",
            pl.col("excess_usd").alias("leakage_usd"),
            pl.lit(None, dtype=pl.String).alias("original_line_id"),
            "billed_vs_card",
        )
    )
    return pl.concat([over_rows, dup_rows]).sort("line_id")
