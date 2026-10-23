"""The check-to-instant scenario changes inputs only: hand-built payouts with known lags and costs."""

from datetime import date, datetime, timedelta

import polars as pl
import pytest

from ledgerlens.fpa.scenario import WEEKS, ScenarioResult, check_to_instant, lag_pmfs

AS_OF = date(2026, 9, 30)


def payouts(
    check_lag_days: int = 6, instant_lag_days: int = 0, n_check: int = 40, n_instant: int = 10
) -> pl.DataFrame:
    rows = []
    for i in range(n_check + n_instant):
        channel = "check" if i < n_check else "instant"
        approved = AS_OF - timedelta(days=3 + (i * 7) % 80)
        lag = check_lag_days if channel == "check" else instant_lag_days
        paid = datetime.combine(approved, datetime.min.time()) + timedelta(days=lag, hours=10)
        rows.append(
            {
                "payout_id": f"P{i}",
                "channel": channel,
                "amount": 1_000.0 + i,
                "handling_cost": 40.0 if channel == "check" else 5.0,
                "approved_date": approved,
                "paid_ts": paid,
            }
        )
    return pl.DataFrame(rows)


def summary(result: ScenarioResult, metric: str) -> dict:
    return result.summary.filter(pl.col("metric") == metric).row(0, named=True)


def test_lag_pmf_puts_all_mass_on_the_observed_lag():
    pmf = lag_pmfs(payouts(check_lag_days=6))
    assert pmf["check"][6] == pytest.approx(1.0) and pmf["instant"][0] == pytest.approx(1.0)


def test_moving_checks_to_instant_cuts_cost_and_days_and_pulls_cash_forward():
    result = check_to_instant(payouts(), AS_OF, share=0.5)
    assert summary(result, "payouts moved to instant")["scenario"] == pytest.approx(20.0)  # half of forty
    assert summary(result, "loss adjustment expense")["change"] == pytest.approx(-20 * (40.0 - 5.0))
    days = summary(result, "claimant days to pay")
    assert days["scenario"] < days["baseline"]
    cash = summary(result, "claims cash out in 13 weeks")
    assert cash["scenario"] >= cash["baseline"]  # instant money leaves sooner, inside the window
    assert result.weekly.height == WEEKS
    shares = dict(zip(result.drivers["segment"], result.drivers["scenario"], strict=True))
    assert shares["check"] + shares["instant"] == pytest.approx(1.0)


def test_with_identical_lags_cash_timing_does_not_move():
    result = check_to_instant(payouts(check_lag_days=2, instant_lag_days=2), AS_OF, share=0.3)
    assert result.weekly["baseline_paid"].to_list() == result.weekly["scenario_paid"].to_list()


def test_result_is_deterministic():
    a, b = check_to_instant(payouts(), AS_OF, 0.3), check_to_instant(payouts(), AS_OF, 0.3)
    assert a.summary.equals(b.summary) and a.weekly.equals(b.weekly)
    assert a.name == "check_to_instant_30"


def test_no_recent_approvals_is_an_error_not_a_zero():
    old = payouts().with_columns(pl.col("approved_date") - pl.duration(days=400))
    with pytest.raises(RuntimeError, match="latest 13 weeks"):
        check_to_instant(old, AS_OF, 0.3)


def test_approvals_up_to_the_as_of_date_are_in_the_window():
    # An instant payout approved and paid on the as-of date (a Wednesday) lands in week thirteen.
    paid = datetime.combine(AS_OF, datetime.min.time()) + timedelta(hours=10)
    last = pl.DataFrame(
        [
            {
                "payout_id": "P_last",
                "channel": "instant",
                "amount": 1_000_000.0,
                "handling_cost": 5.0,
                "approved_date": AS_OF,
                "paid_ts": paid,
            }
        ]
    )
    with_last = check_to_instant(pl.concat([payouts(), last]), AS_OF, 0.3)
    without = check_to_instant(payouts(), AS_OF, 0.3)
    added = with_last.weekly["baseline_paid"] - without.weekly["baseline_paid"]
    assert added.to_list()[:-1] == [0.0] * (WEEKS - 1)
    assert added.to_list()[-1] == pytest.approx(1_000_000.0)
