"""Scoring FP&A forecasts: cross-validation (steps 9-13) and the one-time holdout score (step 14).

``cv`` is statsforecast's cross-validation frame: ``unique_id, ds, cutoff, y`` plus one column per model and
``<model>-lo-<level>`` / ``<model>-hi-<level>`` interval columns. ``history`` is the training frame the folds
were cut from (``unique_id, ds, y``). Each fold's MASE scale comes from that fold's own training rows (up to
its cutoff), so no fold borrows information from the months it forecasts.
"""

import numpy as np
import pandas as pd


def seasonal_scale(values: np.ndarray, season: int) -> float:
    """Mean absolute seasonal difference of the training values; lag 1 when a season does not fit."""
    lag = season if len(values) > season else 1
    diffs = np.abs(values[lag:] - values[:-lag])
    return float(diffs.mean()) if len(diffs) else float("nan")


def with_horizon(frame: pd.DataFrame, origin: str = "cutoff") -> pd.DataFrame:
    """Add ``h``: 1 for the first forecast month or week after each origin, 2 for the next, and so on."""
    out = frame.sort_values(by=["unique_id", origin, "ds"], kind="mergesort").copy()
    out["h"] = out.groupby(["unique_id", origin]).cumcount() + 1
    return out


def _abs_error(actual: pd.Series, forecast: pd.Series) -> np.ndarray:
    return np.abs(actual.to_numpy(dtype=float) - forecast.to_numpy(dtype=float))


def cv_metrics(
    cv: pd.DataFrame, history: pd.DataFrame, models: list[str], season: int, horizons: dict[str, int]
) -> pd.DataFrame:
    """MASE and MAPE per series, model and horizon bucket (e.g. {"1-3": 3, "1-6": 6}), averaged over folds."""
    frame = with_horizon(cv)
    folds = frame[["unique_id", "cutoff"]].drop_duplicates()
    scales = {}
    for uid, cutoff in folds.itertuples(index=False):
        past = history.loc[(history["unique_id"] == uid) & (history["ds"] <= cutoff)].sort_values(by="ds")
        scales[(uid, cutoff)] = seasonal_scale(past["y"].to_numpy(dtype=float), season)
    rows = []
    for bucket, max_h in horizons.items():
        for uid, cutoff in folds.itertuples(index=False):
            fold = frame.loc[
                (frame["unique_id"] == uid) & (frame["cutoff"] == cutoff) & (frame["h"] <= max_h)
            ]
            actual = fold["y"].to_numpy(dtype=float)
            for model in models:
                error = _abs_error(fold["y"], fold[model])
                rows.append(
                    {
                        "unique_id": uid,
                        "cutoff": cutoff,
                        "model": model,
                        "horizon": bucket,
                        "mase": float(error.mean() / scales[(uid, cutoff)]),
                        "mape": float((error / np.abs(actual)).mean()),
                    }
                )
    per_fold = pd.DataFrame(rows)
    means = per_fold.groupby(["unique_id", "model", "horizon"])[["mase", "mape"]].mean()
    return pd.DataFrame(means).reset_index()


def coverage(cv: pd.DataFrame, models: list[str], levels: list[int]) -> pd.DataFrame:
    """Share of actuals inside each model's prediction interval, per series and level."""
    rows = []
    for uid in cv["unique_id"].unique():
        part = cv.loc[cv["unique_id"] == uid]
        actual = part["y"].to_numpy(dtype=float)
        for model in models:
            for level in levels:
                low = part[f"{model}-lo-{level}"].to_numpy(dtype=float)
                high = part[f"{model}-hi-{level}"].to_numpy(dtype=float)
                inside = (actual >= low) & (actual <= high)
                rows.append(
                    {"unique_id": uid, "model": model, "level": level, "coverage": float(inside.mean())}
                )
    return pd.DataFrame(rows)


def choose(metrics: pd.DataFrame, baseline: str, horizon: str) -> pd.DataFrame:
    """Per series, the lowest cross-validated MASE over ``horizon``; the baseline unless it is beaten."""
    part = metrics.loc[metrics["horizon"] == horizon]
    rows = []
    for uid in part["unique_id"].unique():
        scores = part.loc[part["unique_id"] == uid]
        by_model = dict(zip(scores["model"], scores["mase"], strict=True))
        best = min(by_model, key=lambda m: by_model[m])
        beats = by_model[best] < by_model[baseline]
        rows.append(
            {
                "unique_id": uid,
                "chosen": best if beats else baseline,
                "chosen_mase": by_model[best] if beats else by_model[baseline],
                "baseline_mase": by_model[baseline],
                "beats_baseline": bool(beats),
            }
        )
    return pd.DataFrame(rows)


def holdout_scores(
    holdout: pd.DataFrame, history: pd.DataFrame, season: dict[str, int], horizon: dict[str, int]
) -> pd.DataFrame:
    """Per series MASE and MAPE of the chosen and baseline forecasts on held-out rows (step 14).

    ``holdout`` has ``unique_id, ds, freq, y, chosen, baseline`` and the chosen model's interval columns
    ``lo-80, hi-80, lo-95, hi-95``; ``history`` is the training frame. The scale comes from the training rows;
    ``horizon`` caps how many periods count per frequency (the contract: 6 months, 13 weeks).
    """
    frame = with_horizon(holdout.assign(origin=0), origin="origin")
    rows = []
    for uid in frame["unique_id"].unique():
        part = frame.loc[frame["unique_id"] == uid]
        freq = str(part["freq"].iloc[0])
        part = part.loc[part["h"] <= horizon[freq]]
        past = history.loc[history["unique_id"] == uid].sort_values(by="ds")
        scale = seasonal_scale(past["y"].to_numpy(dtype=float), season[freq])
        actual = part["y"].to_numpy(dtype=float)
        row: dict[str, object] = {"unique_id": uid, "freq": freq, "periods": len(part)}
        for name in ("chosen", "baseline"):
            error = _abs_error(part["y"], part[name])
            row[f"{name}_mase"] = float(error.mean() / scale)
            row[f"{name}_mape"] = float((error / np.abs(actual)).mean())
        for level in (80, 95):
            low, high = part[f"lo-{level}"].to_numpy(dtype=float), part[f"hi-{level}"].to_numpy(dtype=float)
            row[f"coverage_{level}"] = float(((actual >= low) & (actual <= high)).mean())
        rows.append(row)
    scores = pd.DataFrame(rows)
    scores["beats_baseline"] = scores["chosen_mase"] < scores["baseline_mase"]
    return scores
