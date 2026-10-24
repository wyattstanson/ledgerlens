import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _(mo):
    mo.md(r"""
    # FP&A forecasting, steps 13-16: error analysis, final test, deployment, monitoring

    Gate C (2026-10-24) accepted the cross-validated results. Step 13 still uses the training rows only; step 14
    touches the frozen test rows exactly once through `guard.final_test`; step 15 refits the reviewed models on
    every closed period and stores the forecasts (`ledgerlens.fpa.forecast.run_forecast`). Runs in WSL:

        UV_PROJECT_ENVIRONMENT=~/.venvs/ledgerlens ~/.local/bin/uv run python ml_pipeline/04_eval.py
    """)
    return


@app.cell
def _(mo):
    import json
    import sys
    from pathlib import Path

    import duckdb
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    from ledgerlens.fpa import forecast as F
    from ledgerlens.fpa import forecast_eval as E
    from ledgerlens.fpa.forecast_series import SEASON

    HERE = Path(mo.notebook_dir() or Path.cwd() / "ml_pipeline")
    sys.path.insert(0, str(HERE))
    import guard

    WAREHOUSE = HERE.parent / "data" / "warehouse.duckdb"
    SPLITS = HERE.parent / "data" / "ml_pipeline"
    train = pd.read_parquet(SPLITS / "train.parquet")
    choice = F.load_choice(HERE / "model_choice.json")
    picks = {uid: spec["chosen"] for uid, spec in choice.items()}
    history = train.rename(columns={"series": "unique_id", "value": "y"})
    CONTRACT_H = {"monthly": 6, "weekly": 13}
    return (
        CONTRACT_H,
        E,
        F,
        HERE,
        SEASON,
        SPLITS,
        WAREHOUSE,
        choice,
        duckdb,
        guard,
        history,
        json,
        np,
        pd,
        picks,
        plt,
        train,
    )


@app.cell
def _(mo):
    mo.md(r"""
    ## Step 13: error analysis (cross-validation, training rows only)
    """)
    return


@app.cell
def _(E, HERE, SEASON, SPLITS, choice, guard, history, np, pd, picks, plt):
    _rows = []
    _worst = []
    _cover = []
    for _freq in ("monthly", "weekly"):
        _cv = E.with_horizon(pd.read_parquet(SPLITS / f"cv_{_freq}.parquet"))
        for _uid in _cv["unique_id"].unique():
            _model = picks[_uid]
            for _cutoff in _cv.loc[_cv["unique_id"] == _uid, "cutoff"].unique():
                _fold = _cv.loc[(_cv["unique_id"] == _uid) & (_cv["cutoff"] == _cutoff)]
                _past = history.loc[(history["unique_id"] == _uid) & (history["ds"] <= _cutoff)].sort_values(
                    by="ds"
                )
                _scale = E.seasonal_scale(_past["y"].to_numpy(dtype=float), SEASON[_freq])
                _actual = _fold["y"].to_numpy(dtype=float)
                for _name, _col in (("chosen", _model), ("baseline", "SeasonalNaive")):
                    _err = np.abs(_actual - _fold[_col].to_numpy(dtype=float))
                    _rows += [
                        {"freq": _freq, "unique_id": _uid, "h": int(h), "kind": _name, "scaled": e / _scale}
                        for h, e in zip(_fold["h"], _err, strict=True)
                    ]
                _ape = np.abs(_actual - _fold[_model].to_numpy(dtype=float)) / np.abs(_actual)
                _worst += [
                    {"unique_id": _uid, "ds": d, "model": _model, "ape": a, "actual": y}
                    for d, a, y in zip(_fold["ds"], _ape, _actual, strict=True)
                ]
            _part = _cv.loc[_cv["unique_id"] == _uid]
            for _level in (80, 95):
                _inside = (_part["y"] >= _part[f"{_model}-lo-{_level}"]) & (
                    _part["y"] <= _part[f"{_model}-hi-{_level}"]
                )
                _cover.append({"unique_id": _uid, "level": _level, "coverage": float(_inside.mean())})
    by_h = pd.DataFrame(_rows).groupby(["freq", "kind", "h"])["scaled"].mean().unstack("kind")
    worst = pd.DataFrame(_worst).sort_values(by="ape", ascending=False).head(8)
    cover = pd.DataFrame(_cover).pivot(index="unique_id", columns="level", values="coverage")

    _fig, _axes = plt.subplots(1, 2, figsize=(12, 4))
    for _ax, _freq in zip(_axes, ("monthly", "weekly"), strict=True):
        by_h.loc[_freq].plot(ax=_ax, marker="o")
        _ax.set_title(f"{_freq}: scaled error by horizon (cross-validation)")
        _ax.set_xlabel("periods ahead")
        _ax.set_ylabel("absolute error / seasonal scale")
    _m = by_h.loc["monthly"]
    guard.fig(
        13,
        "error_by_horizon",
        _fig,
        f"Scaled error by how far ahead the forecast is. Monthly: the chosen models average "
        f"{_m['chosen'].mean():.2f} against SeasonalNaive's {_m['baseline'].mean():.2f}, and their error "
        f"grows from {_m['chosen'].iloc[0]:.2f} one month out to {_m['chosen'].iloc[-1]:.2f} six months out. "
        "Errors that grow with the horizon are expected; a flat line would mean the model ignores recent data.",
        state_dir=HERE,
    )

    _fig, _ax = plt.subplots(figsize=(12, 4.5))
    cover.plot.bar(ax=_ax)
    for _level in (80, 95):
        _ax.axhline(_level / 100, color="grey", linestyle="--", linewidth=0.8)
    _ax.set_ylabel("share of actuals inside the chosen model's interval")
    _ax.tick_params(axis="x", labelrotation=60, labelsize=8)
    _low = cover.index[cover[80] < 0.6].tolist()
    guard.fig(
        13,
        "coverage_by_series",
        _fig,
        f"Interval coverage of each series' chosen model in cross-validation (dashed: nominal 80% and 95%). "
        f"Series whose 80% interval held under 60% of actuals: {', '.join(_low) or 'none'}. Short histories "
        "and claim-size spikes make intervals too narrow; step 14 reports coverage on the test months.",
        state_dir=HERE,
    )
    _top = worst.iloc[0]
    error_notes = (
        f"Largest single miss: {_top['unique_id']} in {_top['ds']:%Y-%m} ({_top['ape']:.0%} off, model "
        f"{_top['model']}). {sum(1 for u in worst['unique_id'] if u.startswith('severity|'))} of the 8 largest "
        "percentage misses are claim size, where one large loss dominates a month; the models are not built to "
        "predict individual large losses, and no feature available at the forecast origin could."
    )
    _ = choice
    error_notes
    return by_h, cover, error_notes, worst


@app.cell
def _(mo):
    mo.md(r"""
    ## Step 14: final test (the frozen test rows, once)
    """)
    return


@app.cell
def _(CONTRACT_H, E, F, HERE, SEASON, SPLITS, guard, history, json, pd, picks, train):
    result_path = HERE / "final_test.json"
    state = json.loads((HERE / ".guard_state.json").read_text(encoding="utf-8"))
    if state["test_touches"] >= 1:
        # Already used once: never score again here; show what was recorded.
        if not result_path.is_file():
            raise RuntimeError(
                "the test set was used but final_test.json is missing; ask the owner before anything"
            )
        final = json.loads(result_path.read_text(encoding="utf-8"))
    else:
        holdout_rows = pd.read_parquet(SPLITS / "test.parquet")
        stash = {}

        def predict(features: pd.DataFrame):
            stash["pred"] = F.predict_rows(train, features, picks)
            return stash["pred"]["chosen"].to_numpy()

        def median_mase(y_true, y_pred):
            # Everything is scored and saved here, before the guard counts the touch: if any of it fails, the
            # test set is still unused.
            assert (stash["pred"]["chosen"].to_numpy() == y_pred).all()
            holdout = stash["pred"].assign(y=y_true)
            scored = E.holdout_scores(holdout, history, SEASON, CONTRACT_H)
            summary = {
                "median_chosen_mase": float(scored["chosen_mase"].median()),
                "median_baseline_mase": float(scored["baseline_mase"].median()),
                "beats_baseline": int(scored["beats_baseline"].sum()),
                "series_count": len(scored),
                "coverage_80": float(scored["coverage_80"].mean()),
                "coverage_95": float(scored["coverage_95"].mean()),
                "series": {
                    r["unique_id"]: {
                        k: (bool(v) if k == "beats_baseline" else v) for k, v in r.items() if k != "unique_id"
                    }
                    for r in scored.to_dict(orient="records")
                },
            }
            holdout.to_parquet(SPLITS / "final_test_predictions.parquet")
            result_path.write_text(json.dumps(summary, indent=2, default=float) + "\n", encoding="utf-8")
            return summary["median_chosen_mase"]

        guard.final_test(
            predict,
            holdout_rows,
            target="value",
            metric_fn=median_mase,
            state_dir=HERE,
            pipeline_md=HERE / "PIPELINE.md",
        )
        final = json.loads(result_path.read_text(encoding="utf-8"))
    final_table = pd.DataFrame.from_dict(final["series"], orient="index")[
        ["freq", "chosen_mase", "baseline_mase", "beats_baseline", "coverage_80", "coverage_95"]
    ]
    final_table
    return final, final_table


@app.cell
def _(HERE, SPLITS, final, guard, pd, plt):
    _pred = pd.read_parquet(SPLITS / "final_test_predictions.parquet")
    _picks = [
        "due_premium|personal_auto",
        "approved_claims|homeowners",
        "severity|homeowners",
        "cash_weekly|collected_premium",
    ]
    _fig, _axes = plt.subplots(2, 2, figsize=(13, 8))
    for _ax, _uid in zip(_axes.ravel(), _picks, strict=True):
        _p = _pred.loc[_pred["unique_id"] == _uid]
        _ax.plot(_p["ds"], _p["y"], color="black", label="actual (test months)")
        _ax.plot(_p["ds"], _p["chosen"], label="chosen model")
        _ax.plot(_p["ds"], _p["baseline"], label="SeasonalNaive", linestyle="--")
        _ax.fill_between(_p["ds"], _p["lo-80"], _p["hi-80"], alpha=0.2, label="80% interval")
        _ax.set_title(_uid, fontsize=9)
        _ax.legend(fontsize=6)
    guard.fig(
        14,
        "final_test",
        _fig,
        f"The one-time final test on March to August 2026. Median MASE: chosen models "
        f"{final['median_chosen_mase']:.2f}, SeasonalNaive {final['median_baseline_mase']:.2f}; the chosen model "
        f"beats SeasonalNaive on {final['beats_baseline']} of {final['series_count']} series. Average interval "
        f"coverage: {final['coverage_80']:.0%} at 80%, {final['coverage_95']:.0%} at 95%.",
        state_dir=HERE,
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Step 15: deployment
    The reviewed choice (`model_choice.json`) plus `ledgerlens.fpa.forecast_series` is the whole pipeline:
    `run_forecast` refits each chosen model on every closed period and stores forecasts, rebuilt totals and the
    profiles behind them in the warehouse. Refitting is deterministic, so running it twice must give identical
    forecasts; that is the reload check.
    """)
    return


@app.cell
def _(F, HERE, WAREHOUSE, duckdb, guard, pd, plt):
    first = F.run_forecast(WAREHOUSE, HERE / "model_choice.json", HERE / "final_test.json")
    second = F.run_forecast(WAREHOUSE, HERE / "model_choice.json", HERE / "final_test.json")
    _cols = ["unique_id", "ds", "mean", "lo-80", "hi-80", "lo-95", "hi-95"]
    reproducible = first.forecasts[_cols].equals(second.forecasts[_cols]) and first.totals.equals(
        second.totals
    )
    assert reproducible, "refitting the reviewed models did not reproduce the stored forecasts"

    # How well the driver tree rebuilds known totals: actual drivers in, compared with the ledger's cash. Out of
    # sample: the six months checked are the latest closed months whose early-payment source month is also
    # closed, and their lag profile is measured as of the month before them, on cohorts that end before any
    # cohort feeding a checked month. Only closed months are read.
    with duckdb.connect(str(WAREHOUSE), read_only=True) as _con:
        _drivers = _con.execute(
            "select driver || '|' || lob as unique_id, (month || '-01')::date as ds, value from fpa.drivers "
            "where segment = '' and (driver in ('due_premium', 'approved_claims', 'severity') or driver = 'opex') "
            "and month <= (select latest_closed from fpa.run_meta)"
        ).df()
        _cash = _con.execute(
            "select (month || '-01')::date as ds, collected_premium, claims_paid - coalesce(payouts_returned, 0) "
            "as claims_paid, lae from fpa.fpa_cash_monthly where closed"
        ).df()
    _drivers["ds"] = pd.to_datetime(_drivers["ds"])
    _cash["ds"] = pd.to_datetime(_cash["ds"])
    _months = sorted(_cash["ds"])[-7:-1]  # the latest closed month's early payments come from the open month
    _as_of = (_months[0] - pd.DateOffset(months=1)).strftime("%Y-%m")
    tree_profile = F.profiles(WAREHOUSE, _as_of)
    _rebuilt = F.rebuild_totals(_drivers, tree_profile, list(_months)).pivot(
        index="ds", columns="measure", values="value"
    )
    tree_error = {
        m: float(
            (
                (_rebuilt[m] - _cash.set_index("ds").loc[_months, m]).abs()
                / _cash.set_index("ds").loc[_months, m]
            ).mean()
        )
        for m in ("collected_premium", "claims_paid", "lae")
    }

    _fig, _axes = plt.subplots(1, 2, figsize=(13, 4.5))
    _tot = first.totals.pivot(index="ds", columns="measure", values="value")
    _tot[["collected_premium", "claims_paid", "net_operating_cash"]].plot(ax=_axes[0], marker="o")
    _axes[0].set_title(f"Monthly totals rebuilt from driver forecasts (origin {first.origin})")
    _wk = first.forecasts.loc[first.forecasts["freq"] == "weekly"]
    for _uid, _colour in (
        ("cash_weekly|collected_premium", "tab:blue"),
        ("cash_weekly|claims_paid", "tab:orange"),
    ):
        _p = _wk.loc[_wk["unique_id"] == _uid]
        _axes[1].plot(_p["ds"], _p["mean"], color=_colour, label=_uid.split("|")[1])
        _axes[1].fill_between(_p["ds"], _p["lo-80"], _p["hi-80"], color=_colour, alpha=0.2)
    _axes[1].set_title("13-week cash forecast with 80% intervals")
    _axes[1].legend(fontsize=7)
    guard.fig(
        15,
        "deployed_forecast",
        _fig,
        f"Left: collections, claims paid and net operating cash for the next six months, rebuilt from the driver "
        f"forecasts. On six closed months its lag profile never saw ({_months[0]:%Y-%m} to {_months[-1]:%Y-%m}), the "
        f"tree rebuilt from actual drivers is off by "
        f"{tree_error['collected_premium']:.1%} (collections), {tree_error['claims_paid']:.1%} (claims paid) "
        f"and {tree_error['lae']:.1%} (LAE) on average. Right: the 13-week weekly cash forecast. Refitting twice "
        "gave identical numbers.",
        state_dir=HERE,
    )
    tree_window = f"{_months[0]:%Y-%m} to {_months[-1]:%Y-%m}"
    _models = first.models
    form_changes = _models.loc[_models["reviewed_form"] != _models["deployed_form"], "unique_id"].tolist()
    return first, form_changes, reproducible, tree_error, tree_window


@app.cell
def _(mo):
    mo.md(r"""
    ## Step 16: monitoring and retraining plan
    - **Each month after close**: the Dagster check `forecast_actuals_within_interval` compares the newly closed
      month's actual driver values with the forecast made one month earlier (`fpa.forecast_history`). It warns
      when 3 or more of the 10 monthly drivers fall outside their 95% interval.
    - **Each quarter**: re-score the last 3 closed months per series; if a chosen model loses to SeasonalNaive on
      2 consecutive quarters, flag that series for review.
    - **Input drift**: a robust z-score above 5 on a new month (the step 4 spike rule) is flagged, not dropped.
    - **Retraining** reuses this pipeline from step 1 (new EDA, a new frozen test window, all four gates). The
      refit in step 15 keeps each series' reviewed model; AutoETS and AutoARIMA may settle on another form of
      it on the longer history, and `fpa.forecast_models` records both the reviewed and the deployed form.
    """)
    return


@app.cell
def _(
    HERE, by_h, cover, error_notes, final, first, form_changes, reproducible, tree_error, tree_window, worst
):
    _pipeline = HERE / "PIPELINE.md"
    _text = _pipeline.read_text(encoding="utf-8")
    _results = {
        13: f"Scaled error grows with the horizon (monthly chosen {by_h.loc['monthly']['chosen'].iloc[0]:.2f} at 1 "
        f"month to {by_h.loc['monthly']['chosen'].iloc[-1]:.2f} at 6); {int((cover[80] < 0.6).sum())} series hold "
        f"under 60% at the 80% level. {error_notes} Worst rows listed in the notebook ({len(worst)}).",
        14: f"Median MASE {final['median_chosen_mase']:.2f} (SeasonalNaive {final['median_baseline_mase']:.2f}); the "
        f"chosen model beats SeasonalNaive on {final['beats_baseline']} of {final['series_count']} series; coverage "
        f"{final['coverage_80']:.0%} at 80%, {final['coverage_95']:.0%} at 95%. Per series in final_test.json.",
        15: f"run_forecast refits the reviewed models on every closed period (origin {first.origin}): fpa.forecasts, "
        "fpa.forecast_totals, fpa.forecast_profiles, fpa.forecast_models, fpa.forecast_history; refit reproduces "
        f"exactly: {reproducible}; on all closed months AutoETS/AutoARIMA settled on another form of the same "
        f"model for {len(form_changes)} of {len(first.models)} series ({', '.join(form_changes) or 'none'}), "
        "both forms in fpa.forecast_models; "
        f"driver-tree rebuild error out of sample ({tree_window}) "
        f"{tree_error['collected_premium']:.1%} / {tree_error['claims_paid']:.1%} / {tree_error['lae']:.1%} "
        "(collections / claims paid / LAE). Inference input: closed months of fpa.drivers and complete weeks of "
        "fpa.fpa_cash_weekly, shaped by forecast_series.to_statsforecast.",
        16: "Plan in 04_eval.py: monthly interval check (Dagster), quarterly re-score against SeasonalNaive, spike "
        "flags, retraining from step 1 with all four gates.",
    }
    _names = {13: "Error analysis", 14: "Final test", 15: "Deployment", 16: "Monitoring and retraining plan"}
    for _step, _result in _results.items():
        _old = next(
            _line for _line in _text.splitlines() if _line.startswith(f"| {_step} | {_names[_step]} |")
        )
        _text = _text.replace(_old, f"| {_step} | {_names[_step]} | done | {_result} |")
    _pipeline.write_text(_text, encoding="utf-8")


if __name__ == "__main__":
    app.run()
