# FP&A forecasting pipeline (BUILD_PROMPT 10.5)

The only place in LedgerLens that fits models. Data: the seed-42 synthetic insurer in
`data/warehouse.duckdb` (driver table `fpa.drivers`, weekly cash `fpa.fpa_cash_weekly`, plan `raw_fpa.plan_*`).
Workbench: marimo notebooks in this folder; figures in `figures/`; profile in `data_profile.json`.

## Steps

| # | Step | Status | Result |
|---|------|--------|--------|
| 1 | Data inspection | approved-gate | see Step 1 below |
| 2 | Exploratory data analysis | approved-gate | see Step 2 below |
| 3 | Define the prediction problem | approved-gate | contract below |
| 4 | Data cleaning | todo | |
| 5 | Data engineering | todo | |
| 6 | Train / validation / test split | todo | |
| 7 | Feature engineering | todo | |
| 8 | Preprocessing | todo | |
| 9 | Baseline model | todo | |
| 10 | Model training | todo | |
| 11 | Hyperparameter tuning | todo | |
| 12 | Model evaluation | todo | |
| 13 | Error analysis | todo | |
| 14 | Final test | todo | |
| 15 | Deployment | todo | |
| 16 | Monitoring and retraining plan | todo | |

## Phase reviews

- Gate A: approved 2026-10-23
- Gate B: pending
- Gate C: pending
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

## Figures
- Figure 01_missingness.png: No column has missing values.
- Figure 02_target_balance.png: value ranges 6 to 1.61e+06 with median 1.79e+04; a mean-predictor baseline is the number to beat.
- Figure 02_temporal_coverage.png: month spans 2023-10-01 to 2026-08-01 over 35 months; a chronological split must hold out the latest period.
- Figure 02_driver_series_vs_plan.png: Each of the 10 series against its plan. A grey line marks a lasting shift in actual/plan (at least 6 months a side, |t| >= 3, 5% or more): none. A model fitted across a shift learns a mix of two regimes, so the training window matters for steps 4-8; series without a marked shift track the plan within noise.
- Figure 02_seasonal_profile.png: Average month-of-year index per series. Strongest seasonality: severity|homeowners (peak-to-trough 98%), then severity|small_commercial (85%); weakest opex|all (8%). With under three years of history each calendar month is seen two or three times, so SeasonalNaive(12) is a fair but noisy baseline.
- Figure 02_autocorrelation.png: Autocorrelation at lags 1-13; dashed lines are the 95% noise band (0.33). 2 of 10 series are significant at lag 12 (approved_claims|homeowners, approved_claims|personal_auto). Strong lag-1 autocorrelation with slow decay means trend or level shifts, which ETS handles with a trend term and ARIMA with differencing.
- Figure 02_weekly_cash.png: 158 weeks from 2023-09-25 to 2026-09-28. The first and last weeks are partial (the window starts mid-week and the last month is still open), so they are dropped before modelling. Week-to-week variation: premium 7% of its mean, claims 39%; claims are lumpier because a few large losses dominate a week, so their 13-week intervals will be wider.
