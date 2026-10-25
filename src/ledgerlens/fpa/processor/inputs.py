"""The processor's driver inputs, day by day, shared by the generator and the plan.

The generator draws actuals from these inputs with the planted misses switched on; the plan takes expected
values of the same inputs with them off. Sharing one definition keeps the two apart only by the plants and
by chance.

Days run from ``PRE_WINDOW_DAYS`` before the window to its last day. Days before the window carry the
window-start values (no plants, the starting agent share), so purchases made then can still be disputed, and
batches captured then can still settle, inside the window.
"""

import math
from dataclasses import dataclass

import numpy as np

from ledgerlens.fpa.config import CARD_TYPES, VERTICALS, ProcessorWorldConfig
from ledgerlens.fpa.gl import month_bounds, month_list

PRE_WINDOW_DAYS = 300  # covers the chargeback lag and the write-off age for anything still open at the start
DAYS_PER_MONTH = 365.25 / 12
UNIX_EPOCH_ORDINAL = 719_163  # date(1970, 1, 1).toordinal()


@dataclass(frozen=True)
class DayGrid:
    days: np.ndarray  # ordinal day numbers
    month: np.ndarray  # window month index, -1 before the window
    calendar_month: np.ndarray  # 1..12
    weekday: np.ndarray  # 0 = Monday
    months: tuple[str, ...]
    window_start: int
    window_end: int


def day_grid(config: ProcessorWorldConfig) -> DayGrid:
    months = month_list(config.end_month, config.months)
    window_start = month_bounds(months[0])[0].toordinal()
    window_end = month_bounds(months[-1])[1].toordinal()
    days = np.arange(window_start - PRE_WINDOW_DAYS, window_end + 1)
    starts = np.array([month_bounds(m)[0].toordinal() for m in months])
    month = np.searchsorted(starts, days, side="right") - 1
    dates = (days - UNIX_EPOCH_ORDINAL).astype("datetime64[D]")
    calendar_month = dates.astype("datetime64[M]").astype(int) % 12 + 1
    return DayGrid(
        days=days,
        month=month,
        calendar_month=calendar_month,
        weekday=(days - 1) % 7,  # date.fromordinal(1) is a Monday
        months=months,
        window_start=int(window_start),
        window_end=int(window_end),
    )


def opening_counts(config: ProcessorWorldConfig) -> list[int]:
    """The opening book by vertical: the configured shares, rounded so they add up to ``merchants``."""
    exact = [config.merchants * config.verticals[v].merchant_share for v in VERTICALS]
    counts = [math.floor(x) for x in exact]
    by_remainder = sorted(range(len(exact)), key=lambda k: (counts[k] - exact[k], k))
    for k in by_remainder[: config.merchants - sum(counts)]:
        counts[k] += 1
    return counts


def day_factor(config: ProcessorWorldConfig, grid: DayGrid) -> np.ndarray:
    """(day, vertical): weekday pattern times seasonality, each averaging one over a week or a year."""
    out = np.empty((len(grid.days), len(VERTICALS)))
    weekend = grid.weekday >= 5  # noqa: PLR2004 (Saturday and Sunday)
    for v, name in enumerate(VERTICALS):
        cfg = config.verticals[name]
        weekly = np.where(weekend, cfg.weekend_factor, 1.0) * 7 / (5 + 2 * cfg.weekend_factor)
        season = 1 + cfg.seasonal_amplitude * np.cos(
            2 * np.pi * (grid.calendar_month - cfg.seasonal_peak_month) / 12
        )
        out[:, v] = weekly * season
    return out


def card_mix(config: ProcessorWorldConfig, grid: DayGrid, *, planted: bool) -> np.ndarray:
    """(day, card type): share of attempts. The plant moves debit attempts to premium credit."""
    base = np.array([config.card_types[c].mix for c in CARD_TYPES])
    out = np.tile(base, (len(grid.days), 1))
    if planted:
        p = config.planted
        moved = grid.month >= p.card_mix_start_month
        shift = base[CARD_TYPES.index("debit")] * p.card_mix_debit_to_premium
        out[moved, CARD_TYPES.index("debit")] -= shift
        out[moved, CARD_TYPES.index("premium_credit")] += shift
    return out


def agent_share(config: ProcessorWorldConfig, grid: DayGrid, *, planted: bool) -> np.ndarray:
    """(day,): share of attempts started by an AI agent, growing linearly month by month."""
    end = config.planted.agent_share_actual_end if planted else config.agent_share_end
    step = (end - config.agent_share_start) / max(config.months - 1, 1)
    return config.agent_share_start + step * np.maximum(grid.month, 0)


def initiation_share(config: ProcessorWorldConfig, grid: DayGrid, *, planted: bool) -> np.ndarray:
    """(day, initiation): human and agent shares of attempts."""
    agent = agent_share(config, grid, planted=planted)
    return np.stack([1 - agent, agent], axis=1)


def authorization_rate(config: ProcessorWorldConfig, grid: DayGrid, *, planted: bool) -> np.ndarray:
    """(day, initiation). The plant makes issuers decline more agent-initiated attempts."""
    agent = np.full(len(grid.days), config.agent_authorization_rate)
    if planted:
        p = config.planted
        agent = np.where(grid.month >= p.agent_auth_start_month, agent - p.agent_auth_rate_drop, agent)
    human = np.full(len(grid.days), config.human_authorization_rate)
    return np.stack([human, agent], axis=1)


def interchange_pct(config: ProcessorWorldConfig, grid: DayGrid, *, planted: bool) -> np.ndarray:
    """(day, card type): interchange plus network fees as a share of volume."""
    base = np.array([config.card_types[c].interchange_pct for c in CARD_TYPES]) + config.network_fee_pct
    out = np.tile(base, (len(grid.days), 1))
    if planted:
        p = config.planted
        lifted = grid.month >= p.interchange_start_month
        out[lifted, CARD_TYPES.index(p.interchange_card_type)] += p.interchange_pct_lift
    return out


def chargeback_rate(config: ProcessorWorldConfig) -> np.ndarray:
    """(vertical, initiation): chargebacks per approved transaction."""
    human = np.array([config.verticals[v].chargeback_rate for v in VERTICALS])
    return np.stack([human, human * config.agent_chargeback_multiplier], axis=1)


def daily_hazard(config: ProcessorWorldConfig) -> float:
    """A merchant's chance of churning, per day, from the monthly churn rate."""
    return -math.log(1 - config.merchant_churn_monthly) / DAYS_PER_MONTH


def settlement_days(days: np.ndarray, business_days: int) -> np.ndarray:
    """The ordinal day each capture day settles: that many business days later (weekend captures count from
    the Friday before, so they settle on Monday at T+1)."""
    dates = (days - UNIX_EPOCH_ORDINAL).astype("datetime64[D]")
    settled = np.busday_offset(dates, business_days, roll="backward")
    return settled.astype(np.int64) + UNIX_EPOCH_ORDINAL


def recovery_window_days(config: ProcessorWorldConfig) -> int:
    """Days after a dispute by which all but one in a thousand merchant repayments have landed."""
    return math.ceil(config.recovery_lag_median_days * math.exp(3.09 * config.recovery_lag_sigma))


def lag_pmf(median: float, sigma: float, max_days: int) -> np.ndarray:
    """P(ceil(L) = k) for k = 0..max_days, L lognormal with this median: the whole-day lag the generator
    draws."""
    k = np.arange(max_days + 1, dtype=float)
    with np.errstate(divide="ignore"):
        z = (np.log(k) - math.log(median)) / (sigma * math.sqrt(2))
    cdf = 0.5 * (1 + np.vectorize(math.erf)(z))
    return np.diff(np.concatenate([[0.0], cdf]))
