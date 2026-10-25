"""The forecast driver tree: totals rebuilt from driver values and lag profiles, by hand-worked cases."""

import json

import numpy as np
import pandas as pd
import pytest

from ledgerlens.fpa.forecast import LAGS, _model, load_choice, load_forms, rebuild_totals, reviewed_form
from ledgerlens.fpa.forecast_series import LINES, timestamp

MONTHS = [timestamp(m) for m in pd.date_range("2026-01-01", periods=6, freq="MS")]


def drivers(
    due: float = 100.0, claims: float = 2.0, severity: float = 50.0, opex: float = 30.0
) -> pd.DataFrame:
    rows = []
    for month in MONTHS:
        for lob in LINES:
            rows += [
                {"unique_id": f"due_premium|{lob}", "ds": month, "value": due},
                {"unique_id": f"approved_claims|{lob}", "ds": month, "value": claims},
                {"unique_id": f"severity|{lob}", "ds": month, "value": severity},
            ]
        rows.append({"unique_id": "opex|all", "ds": month, "value": opex})
    return pd.DataFrame(rows)


def profile(collection: dict[int, float], payout: dict[int, float], lae: dict[int, float]) -> pd.DataFrame:
    rows = []
    for lob in LINES:
        for kind, shares in (("collection", collection), ("payout", payout), ("lae", lae)):
            rows += [{"lob": lob, "kind": kind, "lag": lag, "share": share} for lag, share in shares.items()]
    return pd.DataFrame(rows)


def total(frame: pd.DataFrame, measure: str, month: pd.Timestamp) -> float:
    return float(frame.loc[(frame["measure"] == measure) & (frame["ds"] == month), "value"].iloc[0])


def test_steady_drivers_give_steady_totals_and_net_cash_adds_up():
    shares = profile({-1: 0.1, 0: 0.7, 1: 0.15}, {0: 0.5, 1: 0.4}, {0: 6.0, 1: 4.0})
    month = MONTHS[4]
    totals = rebuild_totals(drivers(), shares, [month])
    assert total(totals, "collected_premium", month) == pytest.approx(3 * 100 * 0.95)
    assert total(totals, "claims_paid", month) == pytest.approx(3 * 2 * 50 * 0.9)
    assert total(totals, "lae", month) == pytest.approx(3 * 2 * 10.0)
    net = 285.0 - 270.0 - 60.0 - 30.0
    assert total(totals, "net_operating_cash", month) == pytest.approx(net)


def test_cash_follows_the_lag_of_the_driver_it_came_from():
    frame = drivers(due=0.0)
    frame.loc[(frame["unique_id"] == "due_premium|homeowners") & (frame["ds"] == MONTHS[2]), "value"] = 1000.0
    shares = profile({0: 0.6, 2: 0.4}, {0: 1.0}, {0: 0.0})
    totals = rebuild_totals(frame, shares, MONTHS[2:5])
    assert [total(totals, "collected_premium", m) for m in MONTHS[2:5]] == pytest.approx([600.0, 0.0, 400.0])


def test_a_month_without_its_lagged_drivers_is_an_error_not_a_guess():
    shares = profile({3: 1.0}, {0: 1.0}, {0: 0.0})  # cash three months after billing
    with pytest.raises(KeyError, match="no due_premium value"):
        rebuild_totals(drivers(), shares, [MONTHS[1]])  # three months back is before the data
    early = profile({-1: 1.0}, {0: 1.0}, {0: 0.0})  # paid a month early: needs next month's billing
    with pytest.raises(KeyError, match="2026-07"):
        rebuild_totals(drivers(), early, [MONTHS[-1]])


def test_lags_cover_early_payment_and_late_cash():
    assert LAGS[0] == -1 and LAGS[-1] == 3


def test_every_committed_choice_names_a_known_model():
    choice = load_choice()
    assert len(choice) == 12
    assert {spec["chosen"] for spec in choice.values()} <= {"SeasonalNaive", "AutoETS", "AutoARIMA"}
    with pytest.raises(ValueError, match="unknown model"):
        _model("Prophet", "monthly")


def test_forecasts_keep_the_season_and_never_dip_below_zero():
    pytest.importorskip("statsforecast.models", reason="statsforecast cannot load here (ADR 0005)")
    from ledgerlens.fpa.forecast import forecast_chosen, predict_rows  # noqa: PLC0415

    months = pd.date_range("2023-01-01", periods=36, freq="MS")
    values = [10.0 + (i % 12) for i in range(36)]
    history = pd.DataFrame({"unique_id": "s", "ds": months, "y": values})
    out = forecast_chosen(history, "monthly", {"s": "SeasonalNaive"}, 3)
    assert out["mean"].tolist() == values[24:27]
    assert (out["lo-95"] >= 0).all()

    frame = pd.DataFrame({"ds": months, "series": "s", "value": values, "freq": "monthly"})
    rows = pd.DataFrame(
        {"ds": pd.date_range("2026-01-01", periods=2, freq="MS"), "series": "s", "freq": "monthly"}
    )
    predicted = predict_rows(frame, rows, {"s": "AutoETS"})
    assert predicted["unique_id"].tolist() == ["s", "s"] and predicted["baseline"].tolist() == values[24:26]
    with pytest.raises(ValueError, match="straight after the history"):
        predict_rows(frame, rows.assign(ds=rows["ds"] + pd.DateOffset(months=3)), {"s": "AutoETS"})


def test_each_series_must_restart_right_after_its_own_history_and_forms_are_recorded():
    pytest.importorskip("statsforecast.models", reason="statsforecast cannot load here (ADR 0005)")
    from ledgerlens.fpa.forecast import forecast_chosen, predict_rows  # noqa: PLC0415

    months = pd.date_range("2023-01-01", periods=36, freq="MS")
    values = [10.0 + (i % 12) for i in range(36)]
    frame = pd.concat(
        [pd.DataFrame({"ds": months, "series": s, "value": values, "freq": "monthly"}) for s in ("a", "b")]
    )
    rows = pd.DataFrame(
        {
            "ds": list(pd.date_range("2026-01-01", periods=3, freq="MS"))
            + list(pd.date_range("2026-02-01", periods=2, freq="MS")),  # b skips its first month
            "series": ["a", "a", "a", "b", "b"],
            "freq": "monthly",
        }
    )
    with pytest.raises(ValueError, match="rows for b do not continue"):
        predict_rows(frame, rows, {"a": "SeasonalNaive", "b": "SeasonalNaive"})

    history = pd.DataFrame({"unique_id": frame["series"], "ds": frame["ds"], "y": frame["value"]})
    out = forecast_chosen(history, "monthly", {"a": "AutoETS", "b": "AutoARIMA"}, 2)
    forms = dict(zip(out["unique_id"], out["form"], strict=True))
    assert str(forms["a"]).startswith("ETS(") and str(forms["b"]).startswith("ARIMA(")


def test_reviewed_arima_forms_are_written_like_deployed_ones():
    spec = {"chosen": "AutoARIMA", "forms_on_training_rows": {"AutoARIMA": "[3, 1, 0, 0, 52, 1, 1]"}}
    assert reviewed_form(spec) == "ARIMA(3,1,1)(0,1,0)[52]"  # statsforecast's arma is (p, q, P, Q, m, d, D)
    assert (
        reviewed_form({"chosen": "AutoETS", "forms_on_training_rows": {"AutoETS": "ETS(M,A,N)"}})
        == "ETS(M,A,N)"
    )
    assert reviewed_form({"chosen": "SeasonalNaive", "forms_on_training_rows": {}}) == "SeasonalNaive"


ARIMA_311 = {
    "model": "AutoARIMA",
    "form": "ARIMA(3,1,1)(0,1,0)[52]",
    "order": [3, 1, 1],
    "seasonal_order": [0, 1, 0],
    "include_mean": False,
    "include_drift": False,
}
REVIEWED = {
    "a": {"chosen": "AutoARIMA", "forms_on_training_rows": {"AutoARIMA": "[3, 1, 0, 0, 52, 1, 1]"}},
    "b": {"chosen": "SeasonalNaive", "forms_on_training_rows": {}},
}


def test_production_refuses_forms_other_than_the_reviewed_ones(tmp_path):
    path = tmp_path / "model_forms.json"
    forms = {"a": ARIMA_311, "b": {"model": "SeasonalNaive", "form": "SeasonalNaive"}}
    path.write_text(json.dumps(forms), encoding="utf-8")
    assert load_forms(REVIEWED, path) == forms

    for wrong in ({"form": "ARIMA(0,1,3)(0,0,0)[52]"}, {"model": "AutoETS"}):
        path.write_text(json.dumps(forms | {"a": ARIMA_311 | wrong}), encoding="utf-8")
        with pytest.raises(ValueError, match="reviewed form of a"):
            load_forms(REVIEWED, path)
    with pytest.raises(ValueError, match="reviewed form of a"):
        load_forms(REVIEWED, tmp_path / "missing.json")


def test_the_committed_forms_are_the_reviewed_ones():
    choice = load_choice()
    forms = load_forms(choice)
    assert {uid: form["form"] for uid, form in forms.items()} == {
        uid: reviewed_form(spec) for uid, spec in choice.items()
    }
    # The weekly collections form keeps its yearly difference (the reason forms are fixed at all).
    assert forms["cash_weekly|collected_premium"]["seasonal_order"] == [0, 1, 0]


def test_a_fixed_form_is_refit_as_it_is_and_a_fitted_form_refits_to_the_same_forecast():
    pytest.importorskip("statsforecast.models", reason="statsforecast cannot load here (ADR 0005)")
    from ledgerlens.fpa.forecast import fit_forms, forecast_chosen  # noqa: PLC0415

    months = pd.date_range("2020-01-01", periods=60, freq="MS")
    noise = np.random.default_rng(7).normal(0.0, 3.0, len(months))
    values = [100.0 + 10.0 * (i % 12) + i + e for i, e in enumerate(noise)]
    history = pd.DataFrame({"unique_id": "s", "ds": months, "y": values})

    ar1 = {
        "model": "AutoARIMA",
        "form": "ARIMA(1,0,0)(0,0,0)[12]",
        "order": [1, 0, 0],
        "seasonal_order": [0, 0, 0],
        "include_mean": True,
        "include_drift": False,
    }
    searched = forecast_chosen(history, "monthly", {"s": "AutoARIMA"}, 3)
    assert set(searched["form"]) != {ar1["form"]}  # left free, the search picks another form for this series
    fixed = forecast_chosen(history, "monthly", {"s": ar1}, 3)
    assert set(fixed["form"]) == {ar1["form"]} and set(fixed["model"]) == {"AutoARIMA"}
    damped = {"model": "AutoETS", "form": "ETS(A,Ad,N)", "ets": "AAN", "damped": True}
    assert set(forecast_chosen(history, "monthly", {"s": damped}, 3)["form"]) == {"ETS(A,Ad,N)"}

    forms = fit_forms(history, "monthly", {"s": "AutoARIMA"})
    again = forecast_chosen(history, "monthly", forms, 3)
    assert again["form"].tolist() == searched["form"].tolist()
    assert again["mean"].tolist() == pytest.approx(searched["mean"].tolist())
