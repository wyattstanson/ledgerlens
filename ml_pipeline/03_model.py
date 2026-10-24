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
    # FP&A forecasting, steps 9-12: baseline, candidates, choice and evaluation

    Gate B (2026-10-23) accepted the model plan in `PIPELINE.md`. Everything here uses the training rows only:
    rolling-origin cross-validation inside them, never the frozen test rows. Windows App Control blocks a
    compiled statsmodels module that statsforecast needs, so this notebook runs in WSL (Linux):

        UV_PROJECT_ENVIRONMENT=~/.venvs/ledgerlens ~/.local/bin/uv run python ml_pipeline/03_model.py
    """)
    return


@app.cell
def _(mo):
    import json
    import sys
    from pathlib import Path

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd
    from statsforecast import StatsForecast
    from statsforecast.models import AutoARIMA, AutoETS, SeasonalNaive

    HERE = Path(mo.notebook_dir() or Path.cwd() / "ml_pipeline")
    sys.path.insert(0, str(HERE))
    import evaluation as E
    import guard
    import series as S

    SPLITS = HERE.parent / "data" / "ml_pipeline"
    train = pd.read_parquet(SPLITS / "train.parquet")
    assert len(train) == 542, "the training rows changed since the split; rerun 02_prep.py"
    LEVELS = [80, 95]
    MODELS = ["SeasonalNaive", "AutoETS", "AutoARIMA"]
    CHOOSE_ON = {"monthly": "1-6", "weekly": "1-13"}
    PLAN = {  # per frequency: season, horizon, origins apart, horizon buckets, models (Gate B rationale)
        "monthly": dict(
            season=12,
            h=6,
            step=3,
            buckets={"1-3": 3, "1-6": 6},
            models=lambda: [
                SeasonalNaive(season_length=12),
                AutoETS(season_length=12),
                AutoARIMA(season_length=12),
            ],
        ),
        "weekly": dict(
            season=52,
            h=13,
            step=13,
            buckets={"1-13": 13},
            # statsforecast's ETS supports seasons of at most 24 periods, so weekly ETS is non-seasonal.
            models=lambda: [
                SeasonalNaive(season_length=52),
                AutoETS(season_length=1),
                AutoARIMA(season_length=52),
            ],
        ),
    }
    return CHOOSE_ON, E, HERE, LEVELS, MODELS, PLAN, S, SPLITS, StatsForecast, guard, json, pd, plt, train


@app.cell
def _(mo):
    mo.md(r"""
    ## Steps 9-11: baseline, candidates and the choice per series
    Three cross-validation windows per frequency. AutoETS and AutoARIMA pick their own form (AICc) on each
    fold's training rows: that search is the tuning, and nothing else is tuned. A candidate replaces
    SeasonalNaive on a series only if its cross-validated MASE is lower.
    """)
    return


@app.cell
def _(CHOOSE_ON, E, LEVELS, MODELS, PLAN, S, SPLITS, StatsForecast, pd, train):
    results = {}
    for _freq, _cfg in PLAN.items():
        _history = S.to_statsforecast(train, _freq)
        _engine = StatsForecast(models=_cfg["models"](), freq=S.FREQ[_freq], n_jobs=1)
        _cv = _engine.cross_validation(
            df=_history, h=_cfg["h"], n_windows=3, step_size=_cfg["step"], level=LEVELS
        )
        _cv.to_parquet(SPLITS / f"cv_{_freq}.parquet", index=False)
        _metrics = E.cv_metrics(_cv, _history, MODELS, _cfg["season"], _cfg["buckets"])
        # The forms the Auto models pick on all training rows (step 14 refits these before forecasting).
        _engine.fit(df=_history)
        _forms = {}
        for _i, _uid in enumerate(_engine.uids):
            _ets, _arima = _engine.fitted_[_i, 1].model_, _engine.fitted_[_i, 2].model_
            _forms[str(_uid)] = {
                "AutoETS": str(_ets.get("method")),
                "AutoARIMA": str(list(_arima.get("arma"))),
            }
        results[_freq] = {
            "history": _history,
            "cv": _cv,
            "metrics": _metrics,
            "choice": E.choose(_metrics, "SeasonalNaive", CHOOSE_ON[_freq]).assign(freq=_freq),
            "coverage": E.coverage(_cv, MODELS, LEVELS),
            "forms": _forms,
        }
    choices = pd.concat([_r["choice"] for _r in results.values()], ignore_index=True)
    choices
    return choices, results


@app.cell
def _(mo):
    mo.md(r"""
    ## Step 12: evaluation against the baseline
    """)
    return


@app.cell
def _(CHOOSE_ON, HERE, LEVELS, MODELS, choices, guard, pd, plt, results):
    mase = pd.concat(
        [_r["metrics"][_r["metrics"]["horizon"] == CHOOSE_ON[_f]] for _f, _r in results.items()]
    ).pivot(index="unique_id", columns="model", values="mase")[MODELS]
    _fig, _ax = plt.subplots(figsize=(12, 5))
    mase.plot.bar(ax=_ax, width=0.8)
    _ax.axhline(1.0, color="grey", linewidth=0.8, linestyle="--")
    _ax.set_ylabel("cross-validated MASE (lower is better)")
    _ax.set_xlabel("")
    _ax.tick_params(axis="x", labelrotation=60, labelsize=8)
    _beaten = choices[choices["beats_baseline"]]
    guard.fig(
        12,
        "cv_mase",
        _fig,
        f"Cross-validated MASE per series (monthly over 1-6 months, weekly over 1-13 weeks); the dashed line is "
        f"the in-sample seasonal naive error. A candidate beats SeasonalNaive on {len(_beaten)} of "
        f"{len(choices)} series ({', '.join(_beaten['unique_id']) or 'none'}); the rest keep the baseline.",
        state_dir=HERE,
    )

    cov = (
        pd.concat([_r["coverage"] for _r in results.values()]).groupby(["model", "level"])["coverage"].mean()
    )
    _fig, _ax = plt.subplots(figsize=(7, 4))
    cov.unstack("level").loc[MODELS].plot.bar(ax=_ax)
    for _level in LEVELS:
        _ax.axhline(_level / 100, color="grey", linewidth=0.8, linestyle="--")
    _ax.set_ylabel("share of actuals inside the interval")
    _ax.tick_params(axis="x", labelrotation=0)
    _worst = (cov - cov.index.get_level_values("level") / 100).abs().idxmax()
    guard.fig(
        12,
        "interval_coverage",
        _fig,
        f"Average interval coverage across series and folds against the nominal 80% and 95% (dashed). Furthest "
        f"from nominal: {_worst[0]} at {_worst[1]}% ({cov[_worst]:.0%}). Under-coverage means the intervals are "
        "too narrow to plan cash against; over-coverage means they are wider than needed.",
        state_dir=HERE,
    )

    _picks = [
        "due_premium|personal_auto",
        "approved_claims|homeowners",
        "severity|homeowners",
        "cash_weekly|claims_paid",
    ]
    _fig, _axes = plt.subplots(2, 2, figsize=(13, 8))
    for _ax, _uid in zip(_axes.ravel(), _picks, strict=True):
        _f = "weekly" if _uid.startswith("cash_weekly") else "monthly"
        _hist = results[_f]["history"]
        _folds = results[_f]["cv"]
        _own = _hist[_hist["unique_id"] == _uid]
        _ax.plot(_own["ds"], _own["y"], color="black", linewidth=1, label="actual (training rows)")
        _last = _folds[(_folds["unique_id"] == _uid) & (_folds["cutoff"] == _folds["cutoff"].max())]
        for _model in MODELS:
            _ax.plot(_last["ds"], _last[_model], label=_model)
        _chosen = choices.set_index("unique_id").loc[_uid, "chosen"]
        _ax.fill_between(
            _last["ds"],
            _last[f"{_chosen}-lo-80"],
            _last[f"{_chosen}-hi-80"],
            alpha=0.2,
            label=f"{_chosen} 80%",
        )
        _ax.set_title(f"{_uid} (chosen: {_chosen})", fontsize=9)
        _ax.legend(fontsize=6)
    guard.fig(
        12,
        "cv_forecasts",
        _fig,
        "The last cross-validation window for four series: each model's forecast against what actually happened, "
        "with the chosen model's 80% interval. These are training months held back inside cross-validation, not "
        "the frozen test set.",
        state_dir=HERE,
    )
    return cov, mase


@app.cell
def _(HERE, choices, cov, json, mase, results):
    # Record the choice for step 14 and the results in PIPELINE.md.
    _record = {
        _row.unique_id: {
            "freq": _row.freq,
            "chosen": _row.chosen,
            "cv_mase": round(_row.chosen_mase, 4),
            "baseline_cv_mase": round(_row.baseline_mase, 4),
            "beats_baseline": bool(_row.beats_baseline),
            "forms_on_training_rows": results[_row.freq]["forms"][_row.unique_id],
        }
        for _row in choices.itertuples()
    }
    (HERE / "model_choice.json").write_text(json.dumps(_record, indent=2) + "\n", encoding="utf-8")

    _pipeline = HERE / "PIPELINE.md"
    _text = _pipeline.read_text(encoding="utf-8")
    _wins = int(choices["beats_baseline"].sum())
    _median = mase.median()
    _results = {
        9: f"SeasonalNaive(12) monthly and SeasonalNaive(52) weekly, cross-validated: median MASE "
        f"{_median['SeasonalNaive']:.2f} across the 12 series.",
        10: f"AutoETS (median MASE {_median['AutoETS']:.2f}) and AutoARIMA ({_median['AutoARIMA']:.2f}) in the "
        "same 3-window cross-validation; the forms picked on all training rows are in model_choice.json.",
        11: "No manual tuning: the Auto models' AICc search on each fold's training rows is the tuning. Choice per "
        f"series by cross-validated MASE, baseline kept unless beaten: a candidate wins on {_wins} of "
        f"{len(choices)} series.",
        12: "Figures 12_cv_mase, 12_interval_coverage and 12_cv_forecasts; mean 80% interval coverage "
        f"{cov.xs(80, level='level').mean():.0%}, mean 95% coverage {cov.xs(95, level='level').mean():.0%}.",
    }
    _names = {9: "Baseline model", 10: "Model training", 11: "Hyperparameter tuning", 12: "Model evaluation"}
    for _step, _result in _results.items():
        _old = next(
            _line for _line in _text.splitlines() if _line.startswith(f"| {_step} | {_names[_step]} |")
        )
        _text = _text.replace(_old, f"| {_step} | {_names[_step]} | done | {_result} |")
    _pipeline.write_text(_text, encoding="utf-8")


if __name__ == "__main__":
    app.run()
