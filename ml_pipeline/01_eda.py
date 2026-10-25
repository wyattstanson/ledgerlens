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
    # FP&A forecasting, steps 1-3: inspection, EDA, problem definition

    BUILD_PROMPT 10.5: forecast the **drivers**, not the totals, then rebuild the totals through the driver
    tree. This notebook only looks at the data and writes the contract for the owner's review. Nothing here
    trains or tunes a model.

    Run headless with `uv run python ml_pipeline/01_eda.py`, or open it with `uv run marimo edit ml_pipeline/01_eda.py`.
    """)


@app.cell
def _(mo):
    import sys
    from pathlib import Path

    import duckdb
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    HERE = Path(mo.notebook_dir() or Path.cwd() / "ml_pipeline")
    sys.path.insert(0, str(HERE))
    import guard

    WAREHOUSE = HERE.parent / "data" / "warehouse.duckdb"
    LINES = ("personal_auto", "homeowners", "small_commercial")
    TARGETS = ("due_premium", "approved_claims", "severity")  # per line; opex is company-wide
    return HERE, LINES, TARGETS, WAREHOUSE, duckdb, guard, np, pd, plt


@app.cell
def _(LINES, TARGETS, WAREHOUSE, duckdb, pd):
    # Read-only, closed periods only: the open month is preliminary and is never a training target.
    with duckdb.connect(str(WAREHOUSE), read_only=True) as con:
        latest_closed, open_period = con.execute(
            "select latest_closed, open_period from fpa.run_meta"
        ).fetchone()
        actual = con.execute(
            "select month, driver, lob, value from fpa.drivers "
            "where segment = '' and month <= ? and (driver in ('due_premium', 'approved_claims', 'severity') "
            "or driver = 'opex')",
            [latest_closed],
        ).df()
        plan = con.execute(
            """
            select month, 'due_premium' as driver, lob, sum(due_amount) as value
            from raw_fpa.plan_premium group by all
            union all
            select month, 'approved_claims', lob, sum(approved_count) from raw_fpa.plan_claims group by all
            union all
            select month, 'severity', lob, sum(approved_amount) / sum(approved_count)
            from raw_fpa.plan_claims group by all
            union all
            select month, 'opex', 'all', amount from raw_fpa.plan_opex
            """
        ).df()
        weekly = con.execute(
            "select week_start, collected_premium, claims_paid, events from fpa.fpa_cash_weekly order by week_start"
        ).df()

    def tidy(frame: pd.DataFrame) -> pd.DataFrame:
        out = frame.assign(
            month=pd.to_datetime(frame["month"] + "-01"),
            series=frame["driver"] + "|" + frame["lob"],
        )
        return out[["month", "series", "value"]].sort_values(["series", "month"]).reset_index(drop=True)

    series = tidy(actual)
    plan_series = tidy(plan[plan["month"] <= latest_closed])
    expected = {f"{t}|{lob}" for t in TARGETS for lob in LINES} | {"opex|all"}
    assert set(series["series"]) == expected, set(series["series"]) ^ expected
    return latest_closed, open_period, plan_series, series, weekly


@app.cell
def _(mo):
    mo.md(r"""
    ## Step 1: data inspection
    """)


@app.cell
def _(HERE, guard, mo, series):
    profile = guard.profile(series, target="value", time_col="month", state_dir=HERE)
    shape = profile["shape"]
    counts = series.groupby("series")["month"].agg(["count", "min", "max"])
    mo.vstack(
        [
            mo.md(
                f"**{shape['n_rows']} rows x {shape['n_cols']} columns**: {series['series'].nunique()} monthly series, "
                f"{counts['count'].min()}-{counts['count'].max()} months each. Missing values: "
                f"{int(series['value'].isna().sum())}. Exact duplicate rows: {profile['duplicates']['exact_row_frac']:.0%}. "
                f"Leakage suspects: {profile['leakage_suspects'] or 'none'}."
            ),
            counts,
        ]
    )
    return (profile,)


@app.cell
def _(mo):
    mo.md(r"""
    ## Step 2: exploratory data analysis
    """)


@app.cell
def _(HERE, guard, series):
    standard_figures = guard.eda_figures(series, target="value", time_col="month", state_dir=HERE)
    return (standard_figures,)


@app.cell
def _(HERE, guard, np, pd, plan_series, plt, series):
    # Actual against plan for every series. The plan is built from the same drivers with expected values, so a
    # lasting gap between the lines is a driver that moved, which a forecaster trained on all months will see.
    min_side = 6  # months each side: shorter edges let a few noisy months pass for a regime

    def change_point(ratio: pd.Series) -> tuple[int, float, float]:
        """Index, size and t statistic of the biggest shift in mean (a two-sample scan, no model).

        The scan picks the best split, so its t is inflated; |t| >= 3 is a screening rule, not a p-value."""
        values = ratio.to_numpy()
        best, at, t_best = 0.0, 0, 0.0
        for k in range(min_side, len(values) - min_side + 1):
            before, after = values[:k], values[k:]
            se = np.sqrt(before.var(ddof=1) / len(before) + after.var(ddof=1) / len(after))
            t = (after.mean() - before.mean()) / se if se > 0 else 0.0
            if abs(t) > abs(t_best):
                best, at, t_best = after.mean() - before.mean(), k, t
        return at, best, t_best

    merged = series.merge(plan_series, on=["month", "series"], suffixes=("", "_plan"))
    merged["ratio"] = merged["value"] / merged["value_plan"]
    names = sorted(merged["series"].unique())
    fig_vs, axes_vs = plt.subplots(4, 3, figsize=(13, 11), sharex=True)
    shifts = {}
    for ax, name in zip(axes_vs.ravel(), names, strict=False):
        part = merged[merged["series"] == name]
        ax.plot(part["month"], part["value"], label="actual")
        ax.plot(part["month"], part["value_plan"], label="plan", linestyle="--")
        at, gap, t = change_point(part["ratio"].reset_index(drop=True))
        shifts[name] = (part["month"].iloc[at], gap, t)
        if abs(t) >= 3 and abs(gap) >= 0.05:
            ax.axvline(part["month"].iloc[at], color="grey", linewidth=0.8)
        ax.set_title(name, fontsize=9)
        ax.tick_params(labelsize=7)
    for ax in axes_vs.ravel()[len(names) :]:
        ax.axis("off")
    axes_vs.ravel()[0].legend(fontsize=7)
    big = {n: s for n, s in shifts.items() if abs(s[2]) >= 3 and abs(s[1]) >= 0.05}
    listed = "; ".join(f"{n} {g:+.0%} from {m:%Y-%m} (t={t:.1f})" for n, (m, g, t) in sorted(big.items()))
    guard.fig(
        2,
        "driver_series_vs_plan",
        fig_vs,
        f"Each of the {len(names)} series against its plan. A grey line marks a lasting shift in actual/plan "
        f"(at least {min_side} months a side, |t| >= 3, 5% or more): {listed or 'none'}. A model fitted across "
        "a shift learns a mix of two regimes, so the training window matters for steps 4-8; series without "
        "a marked shift track the plan within noise.",
        state_dir=HERE,
    )

    month_of_year = series.assign(moy=series["month"].dt.month, year=series["month"].dt.year)
    month_of_year["index"] = month_of_year["value"] / month_of_year.groupby(["series", "year"])[
        "value"
    ].transform("mean")
    seasonal = month_of_year.groupby(["series", "moy"])["index"].mean().unstack(0)
    fig_s, ax_s = plt.subplots(figsize=(10, 4.5))
    for name in seasonal.columns:
        ax_s.plot(seasonal.index, seasonal[name], label=name, linewidth=1)
    ax_s.set_xticks(range(1, 13))
    ax_s.set_xlabel("calendar month")
    ax_s.set_ylabel("value / that year's mean")
    ax_s.legend(fontsize=6, ncol=2)
    amplitude = (seasonal.max() - seasonal.min()).sort_values(ascending=False)
    guard.fig(
        2,
        "seasonal_profile",
        fig_s,
        f"Average month-of-year index per series. Strongest seasonality: {amplitude.index[0]} (peak-to-trough "
        f"{amplitude.iloc[0]:.0%}), then {amplitude.index[1]} ({amplitude.iloc[1]:.0%}); weakest "
        f"{amplitude.index[-1]} ({amplitude.iloc[-1]:.0%}). With under three years of history each calendar "
        "month is seen two or three times, so SeasonalNaive(12) is a fair but noisy baseline.",
        state_dir=HERE,
    )

    def acf(values: np.ndarray, lags: int) -> np.ndarray:
        centred = values - values.mean()
        denom = float((centred**2).sum())
        return np.array([float((centred[k:] * centred[:-k]).sum()) / denom for k in range(1, lags + 1)])

    lags = 13
    fig_a, axes_a = plt.subplots(4, 3, figsize=(13, 9), sharex=True, sharey=True)
    lag12 = {}
    for ax, name in zip(axes_a.ravel(), names, strict=False):
        values = series.loc[series["series"] == name, "value"].to_numpy()
        r = acf(values, lags)
        lag12[name] = r[11]
        ax.bar(range(1, lags + 1), r)
        ax.axhline(1.96 / np.sqrt(len(values)), color="grey", linewidth=0.8, linestyle="--")
        ax.axhline(-1.96 / np.sqrt(len(values)), color="grey", linewidth=0.8, linestyle="--")
        ax.set_title(name, fontsize=9)
    for ax in axes_a.ravel()[len(names) :]:
        ax.axis("off")
    band = 1.96 / np.sqrt(series.groupby("series").size().min())
    seasonal_ac = sorted(n for n, v in lag12.items() if v > band)
    guard.fig(
        2,
        "autocorrelation",
        fig_a,
        f"Autocorrelation at lags 1-{lags}; dashed lines are the 95% noise band ({band:.2f}). "
        f"{len(seasonal_ac)} of {len(names)} series are significant at lag 12 ({', '.join(seasonal_ac) or 'none'}). "
        "Strong lag-1 autocorrelation with slow decay means trend or level shifts, which ETS handles with a "
        "trend term and ARIMA with differencing.",
        state_dir=HERE,
    )
    # Slow drift against plan: the last 12 months' actual/plan against the first 12.
    drift = {
        name: float(part["ratio"].tail(12).mean() / part["ratio"].head(12).mean() - 1)
        for name, part in merged.groupby("series")
    }
    return big, drift, lag12, shifts


@app.cell
def _(HERE, guard, plt, weekly):
    first, last = weekly["week_start"].iloc[0], weekly["week_start"].iloc[-1]
    fig_w, ax_w = plt.subplots(figsize=(12, 4))
    ax_w.plot(weekly["week_start"], weekly["collected_premium"], label="premium collected")
    ax_w.plot(weekly["week_start"], weekly["claims_paid"], label="claims paid (net of returns)")
    ax_w.legend(fontsize=8)
    ax_w.set_title("Weekly cash from the payments feed")
    cv_premium = weekly["collected_premium"].iloc[1:-1].std() / weekly["collected_premium"].iloc[1:-1].mean()
    cv_claims = weekly["claims_paid"].iloc[1:-1].std() / weekly["claims_paid"].iloc[1:-1].mean()
    guard.fig(
        2,
        "weekly_cash",
        fig_w,
        f"{len(weekly)} weeks from {first:%Y-%m-%d} to {last:%Y-%m-%d}. The first and last weeks are partial "
        f"(the window starts mid-week and the last month is still open), so they are dropped before modelling. "
        f"Week-to-week variation: premium {cv_premium:.0%} of its mean, claims {cv_claims:.0%}; claims are "
        "lumpier because a few large losses dominate a week, so their 13-week intervals will be wider.",
        state_dir=HERE,
    )


@app.cell
def _(mo):
    mo.md(r"""
    ## Step 3: the prediction problem (contract for the owner's review)

    - **Targets** (one model choice per series): monthly `due_premium` (USD billed), `approved_claims` (count)
      and `severity` (USD per approved claim) for each of the three lines, and company `opex` (USD): 10 series.
      Collections, claims paid, LAE and net cash are rebuilt from these through the driver tree, using the
      recent method mix, on-time rates, channel shares, payout lags and handling costs.
    - **13-week cash**: weekly premium collected and claims paid (net of returns) from the payments feed,
      partial first and last weeks dropped.
    - **Prediction unit and time**: one series at one forecast origin; origin = the latest closed month
      (weekly: the latest complete week). Only data up to the origin is used; the open month never is.
    - **Horizons**: 3 and 6 months; 13 weeks.
    - **Metric**: MASE against the seasonal naive forecast (primary), MAPE (secondary; all targets are
      strictly positive). Intervals: 80% and 95%, checked for coverage.
    - **Validation**: the last 6 closed months of every series are the frozen test set, touched once at step
      14. Model choice uses rolling-origin cross-validation (3 windows) inside the remaining months only.
    - **Baseline and candidates**: SeasonalNaive(12) (weekly: seasonal naive 52, or last value if a year of
      complete weeks is not left after the split) as the line to beat; AutoETS and AutoARIMA from statsforecast.
      If no candidate beats the baseline on a series, the baseline is used and the report says so.
    - **Constraints**: about 29 training months per series, so models stay small (no ML regressors, no
      neural nets); level shifts found above are reported, not hidden.
    """)


@app.cell
def _(HERE, big, drift, lag12, latest_closed, open_period, profile, series, shifts, standard_figures):
    # Record steps 1-3 in PIPELINE.md (the figure lines are appended by guard.fig).
    pipeline = HERE / "PIPELINE.md"
    text = pipeline.read_text(encoding="utf-8")
    months = series.groupby("series").size()
    step1 = (
        f"Profiled {profile['shape']['n_rows']} rows: {series['series'].nunique()} monthly driver series x "
        f"{months.min()} closed months (to {latest_closed}; {open_period} is open and excluded). No missing values, "
        f"no duplicate rows, no leakage suspects. Columns: month (datetime), series (categorical), value (target)."
    )
    shift_text = "; ".join(f"{n} {g:+.0%} from {m:%Y-%m}" for n, (m, g, _t) in sorted(big.items())) or "none"
    seasonal_series = sorted(n for n, v in lag12.items() if v > 0)
    ranked = sorted(drift.items(), key=lambda kv: abs(kv[1]), reverse=True)
    drift_text = "; ".join(f"{n} {d:+.1%}" for n, d in ranked[:3])
    shift_clause = (
        f"Lasting steps against plan: {shift_text}, so the training window matters"
        if big
        else "No series steps away from plan beyond noise, so the full history is usable"
    )
    step2 = (
        f"{len(standard_figures) + 4} figures. {shift_clause}. Largest slow drift in actual/plan (last 12 months "
        f"against the first 12): {drift_text}. Positive lag-12 autocorrelation in {len(seasonal_series)} of "
        f"{len(shifts)} series. Weekly cash has partial first and last weeks. Hypotheses: (1) series that drift "
        "away from plan favour a model with a trend or level term (ETS, ARIMA) over SeasonalNaive; (2) severity "
        "is the noisiest driver, so its baseline may be hard to beat; (3) opex is close to flat, so a naive model "
        "may win there."
    )
    # Step 3's contract is the owner-reviewed text in PIPELINE.md; a re-run never rewrites it.
    for header, body in (("## Step 1: data inspection", step1), ("## Step 2: EDA", step2)):
        start = text.index(header) + len(header)
        end = text.index("\n## ", start)
        text = text[:start] + "\n\n" + body + "\n" + text[end:]
    pipeline.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    app.run()
