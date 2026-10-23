"""The FP&A forecasting series: load them from the warehouse, clean them, and shape them for statsforecast.

Every step notebook (and later ``fpa/forecast.py``) prepares data through these functions, so training and
inference can never diverge (ml-pipeline step 8). Cleaning rules, all decided without looking at test rows:

- monthly driver series use closed months only; the open month is preliminary and never a target;
- weekly cash uses complete Monday-to-Sunday weeks that lie wholly inside closed months, so the partial first
  week (the window starts on a Sunday) and the weeks of the open month are dropped;
- nothing is imputed or deleted otherwise: claim-size spikes are real losses and are kept (flagged only).
"""

from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

LINES = ("personal_auto", "homeowners", "small_commercial")
MONTHLY_DRIVERS = ("due_premium", "approved_claims", "severity")  # per line; opex is company-wide
WEEKLY_MEASURES = ("collected_premium", "claims_paid")  # claims net of returned payouts
TEST_MONTHS = 6  # the contract: the last six closed months are the frozen test set
SEASON = {"monthly": 12, "weekly": 52}
FREQ = {"monthly": "MS", "weekly": "W-MON"}  # pandas offsets: month start; weeks starting Monday
SPIKE_Z = 5.0  # robust z-score above which a value is flagged as a spike (kept, never removed)


@dataclass(frozen=True)
class Raw:
    monthly: pd.DataFrame  # month (YYYY-MM), driver, lob, value: every period, open month included
    weekly: pd.DataFrame  # week_start, collected_premium, claims_paid: every week in the feed
    window_start: pd.Timestamp  # first day of the first accounting period
    latest_closed: str
    open_period: str


def load_raw(warehouse: Path) -> Raw:
    """Read-only: the driver table, weekly cash and accounting periods, before any cleaning."""
    with duckdb.connect(str(warehouse), read_only=True) as con:
        latest_closed, open_period = con.execute(
            "select latest_closed, open_period from fpa.run_meta"
        ).fetchone()
        monthly = con.execute(
            "select month, driver, lob, value from fpa.drivers where segment = '' "
            "and (driver in ('due_premium', 'approved_claims', 'severity') or driver = 'opex')"
        ).df()
        weekly = con.execute(
            "select week_start, collected_premium, claims_paid from fpa.fpa_cash_weekly order by week_start"
        ).df()
        (window_start,) = con.execute("select min(startdate) from fpa.stg_fpa__accountingperiod").fetchone()
    return Raw(monthly, weekly, pd.Timestamp(window_start), latest_closed, open_period)


def month_end(month: str) -> pd.Timestamp:
    return pd.Timestamp(f"{month}-01") + pd.offsets.MonthEnd(0)


def monthly_long(raw: Raw) -> pd.DataFrame:
    """Closed months only, one row per series and month: ds, series, value, freq."""
    closed = raw.monthly[raw.monthly["month"] <= raw.latest_closed]
    return pd.DataFrame(
        {
            "ds": pd.to_datetime(closed["month"] + "-01"),
            "series": closed["driver"] + "|" + closed["lob"],
            "value": closed["value"].astype(float),
            "freq": "monthly",
        }
    ).sort_values(["series", "ds"], kind="mergesort", ignore_index=True)


def complete_weeks(raw: Raw) -> pd.DataFrame:
    """Weeks wholly inside closed months: starting on or after the window start, ending by the last close."""
    start = pd.to_datetime(raw.weekly["week_start"])
    last_day = month_end(raw.latest_closed)
    keep = (start >= raw.window_start) & (start + timedelta(days=6) <= last_day)
    return raw.weekly[keep.to_numpy()].reset_index(drop=True)


def weekly_long(raw: Raw) -> pd.DataFrame:
    weeks = complete_weeks(raw)
    frames = [
        pd.DataFrame(
            {
                "ds": pd.to_datetime(weeks["week_start"]),
                "series": f"cash_weekly|{measure}",
                "value": weeks[measure].astype(float),
                "freq": "weekly",
            }
        )
        for measure in WEEKLY_MEASURES
    ]
    return pd.concat(frames, ignore_index=True).sort_values(
        ["series", "ds"], kind="mergesort", ignore_index=True
    )


def test_cutoff(latest_closed: str, months: int = TEST_MONTHS) -> pd.Timestamp:
    """First day of the frozen test window: the last ``months`` closed months (weekly: weeks starting then)."""
    return pd.Timestamp(f"{latest_closed}-01") - pd.DateOffset(months=months - 1)


def robust_spikes(frame: pd.DataFrame, z: float = SPIKE_Z) -> pd.DataFrame:
    """Rows whose value is more than ``z`` robust standard deviations (1.4826 x MAD) above the series median."""
    out = []
    for name, part in frame.groupby("series"):
        median = part["value"].median()
        mad = 1.4826 * (part["value"] - median).abs().median()
        if mad > 0:
            scores = (part["value"] - median) / mad
            out.append(part.assign(robust_z=scores)[scores > z].assign(series=name))
    return pd.concat(out, ignore_index=True) if out else frame.head(0).assign(robust_z=np.nan)


def quality_report(frame: pd.DataFrame) -> pd.DataFrame:
    """Per series: rows, first and last date, gaps in the regular calendar, duplicates, non-positive values."""
    rows = []
    for name, part in frame.groupby("series"):
        freq = FREQ[part["freq"].iloc[0]]
        expected = pd.date_range(part["ds"].min(), part["ds"].max(), freq=freq)
        rows.append(
            {
                "series": name,
                "rows": len(part),
                "first": part["ds"].min(),
                "last": part["ds"].max(),
                "gaps": len(expected.difference(pd.DatetimeIndex(part["ds"]))),
                "duplicates": int(part["ds"].duplicated().sum()),
                "non_positive": int((part["value"] <= 0).sum()),
                "missing": int(part["value"].isna().sum()),
            }
        )
    return pd.DataFrame(rows)


def to_statsforecast(frame: pd.DataFrame, freq: str) -> pd.DataFrame:
    """The ``unique_id, ds, y`` frame statsforecast expects, for one frequency, checked to be regular."""
    part = frame[frame["freq"] == freq]
    if part.empty:
        raise ValueError(f"no {freq} series in the frame")
    report = quality_report(part)
    if report[["gaps", "duplicates", "missing"]].to_numpy().any():
        raise ValueError(f"{freq} series are not on a regular calendar:\n{report}")
    out = part.rename(columns={"series": "unique_id", "value": "y"})[["unique_id", "ds", "y"]]
    return out.sort_values(["unique_id", "ds"], kind="mergesort", ignore_index=True)
