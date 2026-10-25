"""Scenarios change driver inputs, never outputs (BUILD_PROMPT 10.6).

The built-in scenario moves a share of claim payouts from check to instant payment. It takes the claim
payouts approved in the 13 calendar weeks (Monday to Sunday) up to the as-of date and replays them twice
through each channel's observed payout lag and handling cost: once as they are, once with the share moved.
Outputs are expected values, so the result is deterministic:

- weekly cash out for the 13 weeks (instant pays sooner, so cash leaves earlier);
- loss adjustment expense (instant is cheaper to handle than check);
- claimant days-to-pay (approval to money received).
"""

from dataclasses import dataclass
from datetime import date, timedelta

import numpy as np
import polars as pl

from ledgerlens.fpa.generate import month_bounds

WEEKS = 13
MAX_LAG_DAYS = 120
APPROVAL_HOUR = 9  # the generator stamps approvals at 09:00


@dataclass(frozen=True)
class ScenarioResult:
    name: str
    description: str
    drivers: pl.DataFrame  # driver, segment, baseline, scenario
    summary: pl.DataFrame  # metric, unit, baseline, scenario, change
    weekly: pl.DataFrame  # week, week_start, baseline_paid, scenario_paid


def lag_pmfs(payouts: pl.DataFrame) -> dict[str, np.ndarray]:
    """Per channel, the probability that a payout lands d whole days after approval (d = 0..MAX_LAG_DAYS)."""
    lags = payouts.filter(pl.col("paid_ts").is_not_null()).with_columns(
        ((pl.col("paid_ts") - pl.col("approved_date").cast(pl.Datetime("us"))).dt.total_seconds() / 86_400)
        .floor()
        .clip(0, MAX_LAG_DAYS)
        .cast(pl.Int64)
        .alias("lag_days")
    )
    out = {}
    for channel, g in lags.group_by("channel"):
        counts = np.bincount(g["lag_days"].to_numpy(), minlength=MAX_LAG_DAYS + 1)
        out[str(channel[0])] = counts / counts.sum()
    return out


def _mean_lag_days(payouts: pl.DataFrame) -> dict[str, float]:
    paid = payouts.filter(pl.col("paid_ts").is_not_null())
    hours = pl.col("paid_ts") - (
        pl.col("approved_date").cast(pl.Datetime("us")) + pl.duration(hours=APPROVAL_HOUR)
    )
    return dict(paid.group_by("channel").agg((hours.dt.total_seconds() / 86_400).mean()).iter_rows())


def check_to_instant(payouts: pl.DataFrame, as_of: date, share: float) -> ScenarioResult:
    # Thirteen Monday-to-Sunday weeks, the last one holding the as-of date, so every approval up to it counts.
    week0 = as_of - timedelta(days=as_of.weekday() + 7 * (WEEKS - 1))
    recent = payouts.filter((pl.col("approved_date") >= week0) & (pl.col("approved_date") <= as_of))
    if recent.is_empty():
        raise RuntimeError("no claim payouts approved in the latest 13 weeks")
    pmf = lag_pmfs(payouts)
    mean_lag = _mean_lag_days(payouts)
    cost = dict(recent.group_by("channel").agg(pl.col("handling_cost").median()).iter_rows())
    cost.setdefault(
        "instant",
        float(
            payouts.filter(pl.col("channel") == "instant").select(pl.col("handling_cost").median()).item()
            or 0
        ),
    )
    horizon_days = (as_of - week0).days + MAX_LAG_DAYS + 1

    def cash_by_day(moved: float) -> np.ndarray:
        cash = np.zeros(horizon_days)
        for r in recent.iter_rows(named=True):
            offset = (r["approved_date"] - week0).days
            shares = {r["channel"]: 1.0}
            if r["channel"] == "check" and moved:
                shares = {"check": 1 - moved, "instant": moved}
            for channel, weight in shares.items():
                cash[offset : offset + MAX_LAG_DAYS + 1] += r["amount"] * weight * pmf[channel]
        return cash

    base_days, scen_days = cash_by_day(0.0), cash_by_day(share)
    weekly = pl.DataFrame(
        {
            "week": list(range(1, WEEKS + 1)),
            "week_start": [week0 + timedelta(days=7 * w) for w in range(WEEKS)],
            "baseline_paid": [round(float(base_days[7 * w : 7 * w + 7].sum()), 2) for w in range(WEEKS)],
            "scenario_paid": [round(float(scen_days[7 * w : 7 * w + 7].sum()), 2) for w in range(WEEKS)],
        }
    )
    legs = recent.group_by("channel").agg(pl.len().alias("legs"))
    by_channel = dict(legs.iter_rows())
    total_legs = sum(by_channel.values())
    moved_legs = by_channel.get("check", 0) * share

    def lae(moved: float) -> float:
        return sum(n * cost[c] for c, n in by_channel.items()) - moved * (cost["check"] - cost["instant"])

    def days_to_pay(moved: float) -> float:
        total = sum(n * mean_lag[c] for c, n in by_channel.items())
        return (total - moved * (mean_lag["check"] - mean_lag["instant"])) / total_legs

    summary = pl.DataFrame(
        [
            (
                f"claims cash out in {WEEKS} weeks",
                "USD",
                float(weekly["baseline_paid"].sum()),
                float(weekly["scenario_paid"].sum()),
            ),
            ("loss adjustment expense", "USD", lae(0.0), lae(moved_legs)),
            ("claimant days to pay", "days", days_to_pay(0.0), days_to_pay(moved_legs)),
            ("payouts moved to instant", "count", 0.0, moved_legs),
            ("share of check payouts moved", "share", 0.0, share),  # the input change, citable
        ],
        schema={"metric": pl.String, "unit": pl.String, "baseline": pl.Float64, "scenario": pl.Float64},
        orient="row",
    ).with_columns((pl.col("scenario") - pl.col("baseline")).alias("change"))
    # Shares of payout legs (a multiparty claim pays two legs), not of claims as in fpa.drivers channel_share.
    check_share = by_channel.get("check", 0) / total_legs
    instant_share = by_channel.get("instant", 0) / total_legs
    drivers = pl.DataFrame(
        {
            "driver": ["payout_leg_share", "payout_leg_share"],
            "segment": ["check", "instant"],
            "baseline": [check_share, instant_share],
            "scenario": [check_share * (1 - share), instant_share + check_share * share],
        }
    )
    pct = round(share * 100)
    return ScenarioResult(
        name=f"check_to_instant_{pct}",
        description=f"Move {pct}% of claim payouts from check to instant (latest 13 weeks of approvals)",
        drivers=drivers,
        summary=summary,
        weekly=weekly,
    )


def as_of_date(periods: pl.DataFrame) -> date:
    """The last day of the latest accounting period."""
    return month_bounds(periods.sort("month")["month"][-1])[1]
