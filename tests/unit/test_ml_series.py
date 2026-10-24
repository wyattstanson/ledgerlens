"""Forecasting series helpers (ledgerlens.fpa.forecast_series): cleaning, test cutoff, statsforecast shape."""

import pandas as pd
import pytest

from ledgerlens.fpa import forecast_series as S


def raw(weeks: list[str], latest_closed: str = "2026-08", window_start: str = "2023-10-01"):
    weekly = pd.DataFrame({"week_start": pd.to_datetime(weeks), "collected_premium": 1.0, "claims_paid": 2.0})
    return S.Raw(pd.DataFrame(), weekly, S.timestamp(window_start), latest_closed, "2026-09")


def test_only_complete_weeks_inside_closed_months_are_kept():
    # 2023-09-25 starts before the window (partial); 2026-08-31 runs into the open month.
    weeks = ["2023-09-25", "2023-10-02", "2026-08-24", "2026-08-31", "2026-09-07"]
    kept = S.complete_weeks(raw(weeks))
    assert [d.strftime("%Y-%m-%d") for d in kept["week_start"]] == ["2023-10-02", "2026-08-24"]


def test_weekly_long_gives_one_series_per_measure():
    frame = S.weekly_long(raw(["2023-10-02", "2023-10-09"]))
    assert sorted(frame["series"].unique()) == ["cash_weekly|claims_paid", "cash_weekly|collected_premium"]
    assert set(frame["freq"]) == {"weekly"}


@pytest.mark.parametrize(("latest", "cutoff"), [("2026-08", "2026-03-01"), ("2026-02", "2025-09-01")])
def test_test_cutoff_is_the_first_of_the_last_six_closed_months(latest, cutoff):
    assert S.test_cutoff(latest) == pd.Timestamp(cutoff)


def test_monthly_long_drops_the_open_month():
    monthly = pd.DataFrame(
        {"month": ["2026-08", "2026-09"], "driver": "opex", "lob": "all", "value": [1.0, 2.0]}
    )
    r = S.Raw(monthly, pd.DataFrame(), S.timestamp("2023-10-01"), "2026-08", "2026-09")
    frame = S.monthly_long(r)
    assert frame["ds"].tolist() == [pd.Timestamp("2026-08-01")] and frame["series"].tolist() == ["opex|all"]


def test_spikes_are_flagged_by_robust_z_and_flat_series_are_not():
    values = [10.0, 11.0, 9.0, 10.0, 12.0, 10.0, 60.0]
    frame = pd.DataFrame(
        {
            "series": ["a"] * 7 + ["flat"] * 7,
            "ds": list(pd.date_range("2024-01-01", periods=7, freq="MS")) * 2,
            "value": values + [5.0] * 7,
        }
    )
    spikes = S.robust_spikes(frame)
    assert spikes["series"].tolist() == ["a"] and spikes["value"].tolist() == [60.0]


def test_statsforecast_shape_refuses_a_gap():
    months = pd.to_datetime(["2024-01-01", "2024-02-01", "2024-04-01"])
    frame = pd.DataFrame({"ds": months, "series": "a", "value": [1.0, 2.0, 3.0], "freq": "monthly"})
    with pytest.raises(ValueError, match="regular calendar"):
        S.to_statsforecast(frame, "monthly")
    ok = frame.iloc[:2]
    shaped = S.to_statsforecast(ok, "monthly")
    assert shaped.columns.tolist() == ["unique_id", "ds", "y"] and len(shaped) == 2
    with pytest.raises(ValueError, match="no weekly series"):
        S.to_statsforecast(ok, "weekly")
