"""The processor's built-in scenario: send a share of agent-initiated attempts with network tokens.

Scenarios change driver inputs, never outputs (BUILD_PROMPT 10.6). The input is the share of agent-initiated
attempts sent with network tokens. The assumption, stated rather than invented: a tokenized agent attempt is
approved at the rate, and later disputed at the rate, observed for human-initiated purchases in the same
vertical over the same 13 calendar weeks (Monday to Sunday) up to the as-of date. Everything else stays as
observed, and each extra approval earns the margin (fees less interchange and network fees) of an average
agent-initiated purchase in its vertical. Outputs are expected values, so the result is deterministic:

- agent-initiated approvals and payment volume over the 13 weeks;
- processing margin, weekly and in total;
- expected chargebacks and their net cost after merchant recoveries.

Chargeback rates are chargebacks raised in the 13 weeks over purchases approved in them, as the card networks
count them, so the baseline expected chargebacks are the ones observed. The net cost of a chargeback is the
vertical's average dollars not repaid per chargeback (human and agent together), taken only from disputes old
enough for their recoveries to have landed (``recovery_days``): a repayment still in flight is not a loss.
"""

from datetime import date, timedelta

import polars as pl

from ledgerlens.fpa.scenario import ScenarioResult

WEEKS = 13


def _ratio(numerator: str | pl.Expr, denominator: str | pl.Expr) -> pl.Expr:
    num = pl.col(numerator) if isinstance(numerator, str) else numerator
    den = pl.col(denominator) if isinstance(denominator, str) else denominator
    return pl.when(den > 0).then(num / den).otherwise(0.0)


def _by_vertical(
    recent: pl.DataFrame, disputes: pl.DataFrame, settled: pl.DataFrame, share: float
) -> pl.DataFrame:
    """One row per vertical: human (_h) and agent (_a) totals, and what tokenizing ``share`` changes."""
    cells = recent.group_by("lob", "initiation").agg(
        pl.col("attempts").sum().cast(pl.Float64).alias("attempts"),
        pl.col("approvals").sum().cast(pl.Float64).alias("approvals"),
        pl.col("volume").sum(),
        (pl.col("fee_amount") - pl.col("interchange_amount")).sum().alias("margin"),
    )
    raised = disputes.group_by("lob", "initiation").agg(pl.len().cast(pl.Float64).alias("chargebacks"))
    both = cells.join(raised, on=["lob", "initiation"], how="left").fill_null(0.0)
    cost = settled.group_by("lob").agg(
        (pl.col("amount").filter(pl.col("recovered_date").is_null()).sum() / pl.len()).alias(
            "net_per_chargeback"
        )
    )

    def side(who: str, suffix: str) -> pl.DataFrame:
        frame = both.filter(pl.col("initiation") == who).drop("initiation")
        return frame.rename({c: f"{c}{suffix}" for c in frame.columns if c != "lob"})

    v = side("human", "_h").join(side("agent", "_a"), on="lob", how="full", coalesce=True).fill_null(0.0)
    v = v.with_columns(
        _ratio("approvals_h", "attempts_h").alias("auth_h"),
        _ratio("approvals_a", "attempts_a").alias("auth_a"),
        _ratio("chargebacks_h", "approvals_h").alias("cb_rate_h"),
    )
    v = v.join(cost, on="lob", how="left").fill_null(0.0)
    v = v.with_columns((share * pl.col("attempts_a") * (pl.col("auth_h") - pl.col("auth_a"))).alias("extra"))
    return v.with_columns(
        (pl.col("extra") * _ratio("volume_a", "approvals_a")).alias("extra_volume"),
        (pl.col("extra") * _ratio("margin_a", "approvals_a")).alias("extra_margin"),
        (share * (pl.col("auth_h") - pl.col("auth_a")) * _ratio("margin_a", "approvals_a")).alias(
            "margin_per_agent_attempt"
        ),
        (
            (1 - share) * pl.col("chargebacks_a")
            + share * pl.col("attempts_a") * pl.col("auth_h") * pl.col("cb_rate_h")
        ).alias("chargebacks_a_scenario"),
    ).sort("lob")


def agent_tokens(
    captures: pl.DataFrame, chargebacks: pl.DataFrame, as_of: date, share: float, recovery_days: int
) -> ScenarioResult:
    week0 = as_of - timedelta(days=as_of.weekday() + 7 * (WEEKS - 1))
    recent = captures.filter(pl.col("capture_date").is_between(week0, as_of))
    if recent.is_empty():
        raise RuntimeError("no captures in the latest 13 weeks")
    disputes = chargebacks.filter(pl.col("raised_date").is_between(week0, as_of))
    settled = disputes.filter(pl.col("raised_date") <= as_of - timedelta(days=recovery_days))
    v = _by_vertical(recent, disputes, settled, share)

    def total(expr: pl.Expr) -> float:
        return float(v.select(expr.sum()).item())

    approvals_a, attempts_a = total(pl.col("approvals_a")), total(pl.col("attempts_a"))
    extra = total(pl.col("extra"))
    cb_a, cb_a_scen = total(pl.col("chargebacks_a")), total(pl.col("chargebacks_a_scenario"))
    cb_base = total(pl.col("chargebacks_h") + pl.col("chargebacks_a"))
    cb_scen = total(pl.col("chargebacks_h") + pl.col("chargebacks_a_scenario"))
    volume = total(pl.col("volume_h") + pl.col("volume_a"))
    margin = total(pl.col("margin_h") + pl.col("margin_a"))
    summary = pl.DataFrame(
        [
            (f"agent-initiated approvals in {WEEKS} weeks", "count", approvals_a, approvals_a + extra),
            (f"payment volume in {WEEKS} weeks", "USD", volume, volume + total(pl.col("extra_volume"))),
            (f"processing margin in {WEEKS} weeks", "USD", margin, margin + total(pl.col("extra_margin"))),
            ("expected chargebacks", "count", cb_base, cb_scen),
            (
                "expected chargeback net cost",
                "USD",
                total((pl.col("chargebacks_h") + pl.col("chargebacks_a")) * pl.col("net_per_chargeback")),
                total(
                    (pl.col("chargebacks_h") + pl.col("chargebacks_a_scenario"))
                    * pl.col("net_per_chargeback")
                ),
            ),
            ("share of agent attempts tokenized", "share", 0.0, share),  # the input change, citable
        ],
        schema={"metric": pl.String, "unit": pl.String, "baseline": pl.Float64, "scenario": pl.Float64},
        orient="row",
    ).with_columns((pl.col("scenario") - pl.col("baseline")).alias("change"))

    gain = dict(v.select("lob", "margin_per_agent_attempt").iter_rows())
    by_week = (
        recent.with_columns(
            ((pl.col("capture_date") - pl.lit(week0)).dt.total_days() // 7 + 1).alias("week"),
            (pl.col("fee_amount") - pl.col("interchange_amount")).alias("margin"),
            pl.when(pl.col("initiation") == "agent")
            .then(
                pl.col("attempts") * pl.col("lob").replace_strict(gain, default=0.0, return_dtype=pl.Float64)
            )
            .otherwise(0.0)
            .alias("gain"),
        )
        .group_by("week")
        .agg(pl.col("margin").sum().alias("baseline_margin"), pl.col("gain").sum())
    )
    weekly = (
        pl.DataFrame({"week": list(range(1, WEEKS + 1))}, schema={"week": pl.Int64})
        .join(by_week, on="week", how="left")
        .fill_null(0.0)
        .select(
            "week",
            pl.col("week")
            .map_elements(lambda w: week0 + timedelta(days=7 * (w - 1)), return_dtype=pl.Date)
            .alias("week_start"),
            pl.col("baseline_margin").round(2),
            (pl.col("baseline_margin") + pl.col("gain")).round(2).alias("scenario_margin"),
        )
    )
    drivers = pl.DataFrame(
        {
            "driver": ["tokenized_share", "authorization_rate", "chargeback_rate"],
            "segment": ["agent", "agent", "agent"],
            "baseline": [
                0.0,
                approvals_a / attempts_a if attempts_a else 0.0,
                cb_a / approvals_a if approvals_a else 0.0,
            ],
            "scenario": [
                share,
                (approvals_a + extra) / attempts_a if attempts_a else 0.0,
                cb_a_scen / (approvals_a + extra) if approvals_a + extra else 0.0,
            ],
        }
    )
    pct = round(share * 100)
    return ScenarioResult(
        name=f"agent_tokens_{pct}",
        description=f"Send {pct}% of agent-initiated attempts with network tokens (latest {WEEKS} weeks)",
        drivers=drivers,
        summary=summary,
        weekly=weekly,
    )
