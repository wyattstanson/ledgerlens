"""Scoring cross-validated forecasts for the FP&A pipeline (steps 9-12) without touching the frozen test rows.

``cv`` is statsforecast's cross-validation frame: ``unique_id, ds, cutoff, y`` plus one column per model and
``<model>-lo-<level>`` / ``<model>-hi-<level>`` interval columns. ``history`` is the training frame the folds
were cut from (``unique_id, ds, y``). Each fold's MASE scale comes from that fold's own training rows (up to
its cutoff), so no fold borrows information from the months it forecasts.
"""

import numpy as np
import pandas as pd


def seasonal_scale(values: np.ndarray, season: int) -> float:
    """Mean absolute seasonal difference of the training values; falls back to lag 1 if a season does not fit."""
    lag = season if len(values) > season else 1
    diffs = np.abs(values[lag:] - values[:-lag])
    return float(diffs.mean()) if len(diffs) else float("nan")


def with_horizon(cv: pd.DataFrame) -> pd.DataFrame:
    """Add ``h``: 1 for the first forecast month or week after each cutoff, 2 for the next, and so on."""
    out = cv.sort_values(["unique_id", "cutoff", "ds"], kind="mergesort").copy()
    out["h"] = out.groupby(["unique_id", "cutoff"]).cumcount() + 1
    return out


def cv_metrics(
    cv: pd.DataFrame, history: pd.DataFrame, models: list[str], season: int, horizons: dict[str, int]
) -> pd.DataFrame:
    """MASE and MAPE per series, model and horizon bucket (e.g. {"1-3": 3, "1-6": 6}), averaged over folds."""
    frame = with_horizon(cv)
    scales = {}
    for (uid, cutoff), _ in frame.groupby(["unique_id", "cutoff"]):
        past = history[(history["unique_id"] == uid) & (history["ds"] <= cutoff)].sort_values("ds")
        scales[(uid, cutoff)] = seasonal_scale(past["y"].to_numpy(dtype=float), season)
    rows = []
    for bucket, max_h in horizons.items():
        part = frame[frame["h"] <= max_h]
        for (uid, cutoff), fold in part.groupby(["unique_id", "cutoff"]):
            for model in models:
                error = (fold["y"] - fold[model]).abs()
                rows.append(
                    {
                        "unique_id": uid,
                        "cutoff": cutoff,
                        "model": model,
                        "horizon": bucket,
                        "mase": float(error.mean() / scales[(uid, cutoff)]),
                        "mape": float((error / fold["y"].abs()).mean()),
                    }
                )
    folds = pd.DataFrame(rows)
    return folds.groupby(["unique_id", "model", "horizon"], as_index=False)[["mase", "mape"]].mean()


def coverage(cv: pd.DataFrame, models: list[str], levels: list[int]) -> pd.DataFrame:
    """Share of actuals inside each model's prediction interval, per series and level."""
    rows = []
    for uid, part in cv.groupby("unique_id"):
        for model in models:
            for level in levels:
                inside = (part["y"] >= part[f"{model}-lo-{level}"]) & (
                    part["y"] <= part[f"{model}-hi-{level}"]
                )
                rows.append(
                    {"unique_id": uid, "model": model, "level": level, "coverage": float(inside.mean())}
                )
    return pd.DataFrame(rows)


def choose(metrics: pd.DataFrame, baseline: str, horizon: str) -> pd.DataFrame:
    """Per series, the model with the lowest cross-validated MASE over ``horizon``; the baseline unless beaten."""
    part = metrics[metrics["horizon"] == horizon]
    rows = []
    for uid, scores in part.groupby("unique_id"):
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
