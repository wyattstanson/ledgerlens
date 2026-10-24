"""Forecast scoring (ledgerlens.fpa.forecast_eval): per-fold scale, horizons, coverage, choice."""

import numpy as np
import pandas as pd
import pytest

from ledgerlens.fpa import forecast_eval as E


def test_seasonal_scale_uses_the_season_and_falls_back_to_lag_one():
    values = np.array([1.0, 2.0, 3.0, 5.0, 6.0, 7.0])
    assert E.seasonal_scale(values, 3) == pytest.approx(4.0)  # |5-1|, |6-2|, |7-3|
    assert E.seasonal_scale(values[:3], 3) == pytest.approx(1.0)  # too short for a season: lag 1


def cv_frame() -> tuple[pd.DataFrame, pd.DataFrame]:
    months = pd.date_range("2024-01-01", periods=8, freq="MS")
    history = pd.DataFrame({"unique_id": "a", "ds": months, "y": [10.0, 12, 11, 13, 12, 14, 13, 15]})
    cutoff = months[5]
    cv = pd.DataFrame(
        {
            "unique_id": "a",
            "ds": months[6:],
            "cutoff": cutoff,
            "y": [13.0, 15.0],
            "Good": [13.0, 14.0],
            "Bad": [10.0, 10.0],
            "Good-lo-80": [12.0, 13.5],
            "Good-hi-80": [14.0, 14.5],
            "Bad-lo-80": [9.0, 9.0],
            "Bad-hi-80": [11.0, 11.0],
        }
    )
    return cv, history


def test_metrics_use_each_folds_own_history_and_bucket_by_horizon():
    cv, history = cv_frame()
    metrics = E.cv_metrics(cv, history, ["Good", "Bad"], season=1, horizons={"1-1": 1, "1-2": 2})
    scale = np.abs(np.diff([10.0, 12, 11, 13, 12, 14])).mean()  # rows up to the cutoff only
    good = metrics.set_index(["model", "horizon"]).loc[("Good", "1-2")]
    assert good["mase"] == pytest.approx(0.5 / scale)  # errors 0 and 1
    assert good["mape"] == pytest.approx((0 / 13 + 1 / 15) / 2)
    assert metrics.set_index(["model", "horizon"]).loc[("Good", "1-1"), "mase"] == pytest.approx(0.0)


def test_coverage_counts_actuals_inside_the_interval():
    cv, _ = cv_frame()
    cov = E.coverage(cv, ["Good", "Bad"], [80]).set_index("model")["coverage"]
    assert cov["Good"] == pytest.approx(0.5) and cov["Bad"] == pytest.approx(0.0)  # 15 is above Good's 14.5


def test_the_baseline_stays_unless_a_candidate_beats_it():
    metrics = pd.DataFrame(
        {
            "unique_id": ["a", "a", "b", "b"],
            "model": ["SeasonalNaive", "AutoETS"] * 2,
            "horizon": "1-6",
            "mase": [1.0, 0.8, 0.9, 0.9],
            "mape": 0.1,
        }
    )
    chosen = E.choose(metrics, "SeasonalNaive", "1-6").set_index("unique_id")
    assert chosen.loc["a", "chosen"] == "AutoETS" and bool(chosen.loc["a", "beats_baseline"])
    # A tie keeps the baseline.
    assert chosen.loc["b", "chosen"] == "SeasonalNaive" and not bool(chosen.loc["b", "beats_baseline"])
