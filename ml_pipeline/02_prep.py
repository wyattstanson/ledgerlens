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
    # FP&A forecasting, steps 4-8: cleaning, data engineering, split, features, preprocessing

    Gate A (2026-10-23) accepted the contract in `PIPELINE.md`. This notebook prepares the data and freezes the
    test set; it fits nothing. Run headless with `uv run python ml_pipeline/02_prep.py`.
    """)
    return


@app.cell
def _(mo):
    import sys
    from pathlib import Path

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    HERE = Path(mo.notebook_dir() or Path.cwd() / "ml_pipeline")
    sys.path.insert(0, str(HERE))
    import guard
    import series as S

    WAREHOUSE = HERE.parent / "data" / "warehouse.duckdb"
    SPLITS = HERE.parent / "data" / "ml_pipeline"  # gitignored: rebuilt from the seed-42 warehouse
    raw = S.load_raw(WAREHOUSE)
    return HERE, S, SPLITS, guard, pd, plt, raw


@app.cell
def _(mo):
    mo.md(r"""
    ## Step 4: data cleaning
    """)
    return


@app.cell
def _(HERE, S, guard, pd, plt, raw):
    monthly_all = raw.monthly
    monthly = S.monthly_long(raw)
    weekly = S.weekly_long(raw)
    kept_weeks = S.complete_weeks(raw)
    cutoff = S.test_cutoff(raw.latest_closed)

    # Spikes are flagged on the training months only, so the test rows never shape a cleaning decision.
    severity_train = monthly[monthly["series"].str.startswith("severity|") & (monthly["ds"] < cutoff)]
    spikes = S.robust_spikes(severity_train)
    cleaning = pd.DataFrame(
        [
            ("monthly rows, open month dropped", len(monthly_all), len(monthly)),
            ("weekly weeks, partial or open-month weeks dropped", len(raw.weekly), len(kept_weeks)),
            ("missing values", int(monthly_all["value"].isna().sum()), int(monthly["value"].isna().sum())),
            ("claim-size spikes flagged and kept (training months)", len(spikes), len(spikes)),
        ],
        columns=["rule", "before", "after"],
    )

    fig_c, (ax_w, ax_s) = plt.subplots(2, 1, figsize=(12, 7))
    all_weeks = pd.to_datetime(raw.weekly["week_start"])
    dropped = raw.weekly[~all_weeks.isin(pd.to_datetime(kept_weeks["week_start"]))]
    ax_w.plot(all_weeks, raw.weekly["claims_paid"], color="tab:orange", label="claims paid, every week")
    ax_w.scatter(
        pd.to_datetime(dropped["week_start"]),
        dropped["claims_paid"],
        color="black",
        zorder=3,
        label="dropped: partial or in the open month",
    )
    ax_w.legend(fontsize=8)
    ax_w.set_title("Weekly claims paid: which weeks are kept")
    for _name, _part in severity_train.groupby("series"):
        ax_s.plot(_part["ds"], _part["value"], label=_name)
    ax_s.scatter(
        spikes["ds"], spikes["value"], color="black", zorder=3, label=f"robust z > {S.SPIKE_Z:g} (kept)"
    )
    ax_s.set_title("Claim size per line, training months, with flagged spikes")
    ax_s.legend(fontsize=7)
    spike_list = ", ".join(
        f"{r.series.split('|')[1]} {r.ds:%Y-%m} (z={r.robust_z:.1f})" for r in spikes.itertuples()
    )
    guard.fig(
        4,
        "cleaning",
        fig_c,
        f"Top: {len(raw.weekly) - len(kept_weeks)} of {len(raw.weekly)} weeks are dropped because they are "
        f"partial or fall in the open month; {len(kept_weeks)} complete weeks remain. Bottom: claim size has "
        f"{len(spikes)} spikes above a robust z of {S.SPIKE_Z:g} in the training months ({spike_list or 'none'}). "
        "They are real large losses, so they stay in the data; the model choice must cope with them.",
        state_dir=HERE,
    )
    cleaning
    return cleaning, cutoff, monthly, spikes, weekly


@app.cell
def _(mo):
    mo.md(r"""
    ## Step 5: data engineering
    One row per series and period, monthly and weekly together, checked for a regular calendar.
    """)
    return


@app.cell
def _(S, monthly, pd, weekly):
    frame = pd.concat([monthly, weekly], ignore_index=True)
    quality = S.quality_report(frame)
    assert not quality[["gaps", "duplicates", "non_positive", "missing"]].to_numpy().any(), quality
    assert not frame.duplicated(["series", "ds"]).any()
    quality
    return frame, quality


@app.cell
def _(mo):
    mo.md(r"""
    ## Step 6: train / test split (frozen with the guard)
    Chronological: every row from the cutoff on is test, for monthly and weekly series alike. There is no
    separate validation slice: model choice uses rolling-origin cross-validation inside the training rows.
    """)
    return


@app.cell
def _(HERE, SPLITS, cutoff, frame, guard, pd, plt):
    ordered = frame.sort_values("ds", kind="mergesort", ignore_index=True)
    n_test = int((ordered["ds"] >= cutoff).sum())
    train, _val, test = guard.split(
        ordered, target="value", time_col="ds", test_size=n_test / len(ordered), val_size=0.0, state_dir=HERE
    )
    assert len(test) == n_test and _val.empty
    assert train["ds"].max() < cutoff <= test["ds"].min()
    SPLITS.mkdir(parents=True, exist_ok=True)
    train.to_parquet(SPLITS / "train.parquet", index=False)
    test.to_parquet(SPLITS / "test.parquet", index=False)  # read back only by guard.final_test at step 14
    split_counts = (
        pd.concat([train.assign(split="train"), test.assign(split="test")])
        .groupby(["freq", "series", "split"])
        .size()
        .unstack("split")
    )

    fig_split, axes_split = plt.subplots(2, 2, figsize=(12, 6))
    for ax, name in zip(
        axes_split.ravel(),
        [
            "due_premium|personal_auto",
            "approved_claims|homeowners",
            "severity|homeowners",
            "cash_weekly|claims_paid",
        ],
        strict=True,
    ):
        for part, colour in ((train, "tab:blue"), (test, "tab:red")):
            rows = part[part["series"] == name]
            ax.plot(rows["ds"], rows["value"], color=colour)
        ax.axvline(cutoff, color="grey", linestyle="--", linewidth=0.8)
        ax.set_title(name, fontsize=9)
    guard.fig(
        6,
        "split",
        fig_split,
        f"Blue is training, red is the frozen test set from {cutoff:%Y-%m-%d}: {len(train)} training rows and "
        f"{len(test)} test rows ({split_counts['test'].min()}-{split_counts['test'].max()} per series). The test "
        "rows are fingerprinted by the guard and are touched once, at step 14.",
        state_dir=HERE,
    )
    split_counts
    return split_counts, test, train


@app.cell
def _(mo):
    mo.md(r"""
    ## Step 7: feature engineering
    None. The candidates are univariate (SeasonalNaive, AutoETS, AutoARIMA): each series is forecast from its
    own past, with the season length (12 months, 52 weeks) as the only design choice. External regressors
    would need forecasts of their own, and lags or rolling features belong to models this little data cannot
    support. No transformation is chosen here: AutoETS picks additive or multiplicative components itself, on
    the training rows.

    ## Step 8: preprocessing
    `series.to_statsforecast` turns the training rows into `unique_id, ds, y` for each frequency and refuses
    any series with a gap, duplicate or missing value. Inference at step 14 and in `fpa/forecast.py` uses the
    same function, so the two cannot diverge. There is nothing to scale, encode or impute.
    """)
    return


@app.cell
def _(S, train):
    monthly_input = S.to_statsforecast(train, "monthly")
    weekly_input = S.to_statsforecast(train, "weekly")
    shapes = {
        "monthly": (
            monthly_input["unique_id"].nunique(),
            int(monthly_input.groupby("unique_id").size().min()),
        ),
        "weekly": (weekly_input["unique_id"].nunique(), int(weekly_input.groupby("unique_id").size().min())),
    }
    shapes
    return (shapes,)


@app.cell
def _(HERE, cleaning, cutoff, quality, shapes, spikes, split_counts, test, train):
    # Record steps 4-8 in PIPELINE.md (figure lines are appended by guard.fig).
    pipeline = HERE / "PIPELINE.md"
    text = pipeline.read_text(encoding="utf-8")
    rules = "; ".join(f"{r.rule}: {r.before} -> {r.after}" for r in cleaning.itertuples())
    results = {
        4: f"Rules (before -> after): {rules}. Nothing imputed or deleted otherwise; spikes kept.",
        5: f"{len(quality)} series (10 monthly, 2 weekly), one row per series and period; no gaps, duplicates, "
        "non-positive or missing values.",
        6: f"Chronological split at {cutoff:%Y-%m-%d} with guard.split: {len(train)} training rows, {len(test)} "
        f"test rows ({int(split_counts['test'].min())}-{int(split_counts['test'].max())} per series), test "
        "fingerprint frozen in .guard_state.json; no validation slice (rolling-origin CV inside training).",
        7: "No engineered features: univariate models; season lengths 12 (monthly) and 52 (weekly).",
        8: f"series.to_statsforecast: {shapes['monthly'][0]} monthly series x {shapes['monthly'][1]} months, "
        f"{shapes['weekly'][0]} weekly series x {shapes['weekly'][1]} weeks; no scaling, encoding or imputation.",
    }
    names = {
        4: "Data cleaning",
        5: "Data engineering",
        6: "Train / validation / test split",
        7: "Feature engineering",
        8: "Preprocessing",
    }
    for step, result in results.items():
        old = next(line for line in text.splitlines() if line.startswith(f"| {step} | {names[step]} |"))
        text = text.replace(old, f"| {step} | {names[step]} | done | {result} |")
    pipeline.write_text(text, encoding="utf-8")
    len(spikes)
    return


if __name__ == "__main__":
    app.run()
