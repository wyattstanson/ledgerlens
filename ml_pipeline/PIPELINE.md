# FP&A forecasting pipeline (BUILD_PROMPT 10.5)

The only place in LedgerLens that fits models. Data: the seed-42 synthetic insurer in
`data/warehouse.duckdb` (driver table `fpa.drivers`, weekly cash `fpa.fpa_cash_weekly`, plan `raw_fpa.plan_*`).
Workbench: marimo notebooks in this folder; figures in `figures/`; profile in `data_profile.json`. Splits and
cross-validation frames go to `data/ml_pipeline/` (gitignored, rebuilt from the seed-42 warehouse). From step 9
on the notebooks run in WSL (`UV_PROJECT_ENVIRONMENT=~/.venvs/ledgerlens`), because Windows App Control blocks a
compiled statsmodels module that statsforecast imports.

## Steps

| # | Step | Status | Result |
|---|------|--------|--------|
| 1 | Data inspection | approved-gate | see Step 1 below |
| 2 | Exploratory data analysis | approved-gate | see Step 2 below |
| 3 | Define the prediction problem | approved-gate | contract below |
| 4 | Data cleaning | approved-gate | Rules (before -> after): monthly rows, open month dropped: 360 -> 350; weekly weeks, partial or open-month weeks dropped: 158 -> 152; missing values: 0 -> 0; claim-size spikes flagged and kept (training months): 2 -> 2. Nothing imputed or deleted otherwise; spikes kept. |
| 5 | Data engineering | approved-gate | 12 series (10 monthly, 2 weekly), one row per series and period; no gaps, duplicates, non-positive or missing values. |
| 6 | Train / validation / test split | approved-gate | Chronological split at 2026-03-01 with guard.split: 542 training rows, 112 test rows (6-26 per series), test fingerprint frozen in .guard_state.json; no validation slice (rolling-origin CV inside training). |
| 7 | Feature engineering | approved-gate | No engineered features: univariate models; season lengths 12 (monthly) and 52 (weekly). |
| 8 | Preprocessing | approved-gate | series.to_statsforecast: 10 monthly series x 29 months, 2 weekly series x 126 weeks; no scaling, encoding or imputation. |
| 9 | Baseline model | approved-gate | SeasonalNaive(12) monthly and SeasonalNaive(52) weekly, cross-validated: median MASE 1.35 across the 12 series. |
| 10 | Model training | approved-gate | AutoETS (median MASE 1.00) and AutoARIMA (0.94) in the same 3-window cross-validation; the forms picked on all training rows are in model_choice.json. |
| 11 | Hyperparameter tuning | approved-gate | No manual tuning: the Auto models' AICc search on each fold's training rows is the tuning. Choice per series by cross-validated MASE, baseline kept unless beaten: a candidate wins on 9 of 12 series. |
| 12 | Model evaluation | approved-gate | Figures 12_cv_mase, 12_interval_coverage and 12_cv_forecasts; mean 80% interval coverage 71%, mean 95% coverage 90%. |
| 13 | Error analysis | todo | |
| 14 | Final test | todo | |
| 15 | Deployment | todo | |
| 16 | Monitoring and retraining plan | todo | |

## Phase reviews

- Gate A: approved 2026-10-23
- Gate B: approved 2026-10-23
- Gate C: approved 2026-10-24
- Gate D: pending

## Step 1: data inspection

Profiled 350 rows: 10 monthly driver series x 35 closed months (to 2026-08; 2026-09 is open and excluded). No missing values, no duplicate rows, no leakage suspects. Columns: month (datetime), series (categorical), value (target).

## Step 2: EDA

7 figures. No series steps away from plan beyond noise, so the full history is usable. Largest slow drift in actual/plan (last 12 months against the first 12): severity|homeowners +28.1%; approved_claims|small_commercial -12.9%; severity|small_commercial -7.7%. Positive lag-12 autocorrelation in 7 of 10 series. Weekly cash has partial first and last weeks. Hypotheses: (1) series that drift away from plan favour a model with a trend or level term (ETS, ARIMA) over SeasonalNaive; (2) severity is the noisiest driver, so its baseline may be hard to beat; (3) opex is close to flat, so a naive model may win there.

## Step 3: prediction problem (contract)

Accepted by the owner with Gate A (as written in `01_eda.py`):

- Targets, one model choice per series: monthly `due_premium` (USD billed), `approved_claims` (count) and
  `severity` (USD per approved claim) for each of the three lines, and company `opex` (USD): 10 series.
  Collections, claims paid, LAE and net cash are rebuilt from these through the driver tree.
- 13-week cash: weekly premium collected and claims paid (net of returns), partial weeks dropped.
- Prediction unit and time: one series at one forecast origin, the latest closed month (weekly: the latest
  complete week). Only data up to the origin is used; the open month never is.
- Horizons: 3 and 6 months; 13 weeks.
- Metric: MASE against the seasonal naive forecast (primary), MAPE (secondary). Intervals at 80% and 95%,
  checked for coverage.
- Validation: the last 6 closed months are the frozen test set, touched once at step 14. Model choice uses
  rolling-origin cross-validation (3 windows) inside the earlier months only.
- Baseline SeasonalNaive(12) (weekly: seasonal naive 52, or last value if a year of complete weeks is not
  left after the split); candidates AutoETS and AutoARIMA. If none beats the baseline on a series, the
  baseline is used and the report says so.
- Constraints: about 29 training months per series, so models stay small; level shifts are reported.

## Steps 4-8: preparation (for the Gate B review)

Written by `02_prep.py`; the per-step results are in the table above. The split is chronological at
2026-03-01 for every series, monthly and weekly alike, and the guard froze the test rows' fingerprint.

Model rationale:
- traits: univariate time-series regression; 10 monthly series x 29 training months (small data) and 2 weekly series x 126 training weeks; has_datetime; no groups (each series is forecast on its own); two claim-size spikes kept as real losses
- baseline: SeasonalNaive(12) for the monthly series and SeasonalNaive(52) for the weekly series, the contract's line to beat
- candidates: AutoETS and AutoARIMA from statsforecast, chosen per series by rolling-origin cross-validation inside the training rows (3 windows; monthly h=6 with origins 3 months apart, weekly h=13 with origins 13 weeks apart); weekly AutoETS is non-seasonal because statsforecast's ETS supports seasons of at most 24 periods, and AutoARIMA skips seasonal differencing in folds of 24 months or fewer
- ruled out: gradient boosting on lag features and neural nets (29 points per monthly series is too little); random or k-fold splits (temporal data); external regressors (they would need forecasts of their own); removing or capping the claim-size spikes (they are real losses)
- metric: MASE with a seasonal scale from each fold's training rows (primary), over horizons 1-3 and 1-6 months and 1-13 weeks; MAPE secondary; 80% and 95% interval coverage reported; a candidate replaces the baseline on a series only if its cross-validated MASE is lower

## Figures
- Figure 01_missingness.png: No column has missing values.
- Figure 02_target_balance.png: value ranges 6 to 1.61e+06 with median 1.79e+04; a mean-predictor baseline is the number to beat.
- Figure 02_temporal_coverage.png: month spans 2023-10-01 to 2026-08-01 over 35 months; a chronological split must hold out the latest period.
- Figure 02_driver_series_vs_plan.png: Each of the 10 series against its plan. A grey line marks a lasting shift in actual/plan (at least 6 months a side, |t| >= 3, 5% or more): none. A model fitted across a shift learns a mix of two regimes, so the training window matters for steps 4-8; series without a marked shift track the plan within noise.
- Figure 02_seasonal_profile.png: Average month-of-year index per series. Strongest seasonality: severity|homeowners (peak-to-trough 98%), then severity|small_commercial (85%); weakest opex|all (8%). With under three years of history each calendar month is seen two or three times, so SeasonalNaive(12) is a fair but noisy baseline.
- Figure 02_autocorrelation.png: Autocorrelation at lags 1-13; dashed lines are the 95% noise band (0.33). 2 of 10 series are significant at lag 12 (approved_claims|homeowners, approved_claims|personal_auto). Strong lag-1 autocorrelation with slow decay means trend or level shifts, which ETS handles with a trend term and ARIMA with differencing.
- Figure 02_weekly_cash.png: 158 weeks from 2023-09-25 to 2026-09-28. The first and last weeks are partial (the window starts mid-week and the last month is still open), so they are dropped before modelling. Week-to-week variation: premium 7% of its mean, claims 39%; claims are lumpier because a few large losses dominate a week, so their 13-week intervals will be wider.
- Figure 04_cleaning.png: Top: 6 of 158 weeks are dropped because they are partial or fall in the open month; 152 complete weeks remain. Bottom: claim size has 2 spikes above a robust z of 5 in the training months (homeowners 2025-11 (z=10.6), homeowners 2026-01 (z=5.2)). They are real large losses, so they stay in the data; the model choice must cope with them.
- Figure 06_split.png: Blue is training, red is the frozen test set from 2026-03-01: 542 training rows and 112 test rows (6-26 per series). The test rows are fingerprinted by the guard and are touched once, at step 14.
- Figure 12_cv_mase.png: Cross-validated MASE per series (monthly over 1-6 months, weekly over 1-13 weeks); the dashed line is the in-sample seasonal naive error. A candidate beats SeasonalNaive on 9 of 12 series (approved_claims|small_commercial, due_premium|homeowners, due_premium|personal_auto, due_premium|small_commercial, opex|all, severity|personal_auto, severity|small_commercial, cash_weekly|claims_paid, cash_weekly|collected_premium); the rest keep the baseline.
- Figure 12_interval_coverage.png: Average interval coverage across series and folds against the nominal 80% and 95% (dashed). Furthest from nominal: SeasonalNaive at 80% (62%). Under-coverage means the intervals are too narrow to plan cash against; over-coverage means they are wider than needed.
- Figure 12_cv_forecasts.png: The last cross-validation window for four series: each model's forecast against what actually happened, with the chosen model's 80% interval. These are training months held back inside cross-validation, not the frozen test set.
