"""The processor's budget: expected values of the same day-level drivers, with the planted misses off.

The plan knows nothing of the plants, so actuals beat or miss it for reasons the bridge can name: more
digital-goods merchants (fee revenue volume), debit traffic moving to premium credit cards (interchange mix),
a premium-credit interchange increase (interchange rate), agent-initiated traffic growing faster than planned
(more chargebacks) and issuers declining more of it (agent authorization rate).

Everything here is an exact expectation, computed in closed form rather than by simulation, so the plan is
deterministic and needs no seed: merchants survive with the generator's daily hazard, attempts follow the same
weekday and seasonal pattern, batches settle on the same business-day rule, and chargebacks and recoveries
arrive on the same whole-day lag distributions.
"""

import math

import numpy as np
import polars as pl

from ledgerlens.fpa.config import CARD_TYPES, INITIATIONS, VERTICALS, ProcessorWorldConfig
from ledgerlens.fpa.gl import month_bounds
from ledgerlens.fpa.processor.inputs import (
    DayGrid,
    authorization_rate,
    card_mix,
    chargeback_rate,
    daily_hazard,
    day_factor,
    day_grid,
    initiation_share,
    interchange_pct,
    lag_pmf,
    opening_counts,
    settlement_days,
)


def expected_merchants(config: ProcessorWorldConfig, grid: DayGrid) -> tuple[np.ndarray, np.ndarray]:
    """(day, vertical) expected active merchants, and their expected summed daily attempts."""
    survive = math.exp(-daily_hazard(config))
    n_days = len(grid.days)
    count = np.zeros((n_days, len(VERTICALS)))
    for v, (name, opening) in enumerate(zip(VERTICALS, opening_counts(config), strict=True)):
        share = config.verticals[name].merchant_share
        arrivals = 0.0
        for k, day in enumerate(grid.days):
            m = grid.month[k]
            if m < 0:
                count[k, v] = opening
                continue
            start, end = month_bounds(grid.months[m])
            per_day = config.new_merchants_per_month * share / ((end - start).days + 1)
            arrivals = arrivals * survive + per_day
            count[k, v] = opening * survive ** (int(day) - grid.window_start) + arrivals
    rates = np.array([config.verticals[v].attempts_per_merchant_day for v in VERTICALS])
    return count, count * rates  # merchant size multipliers average one


def _by_month(grid: DayGrid, day_index: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Sum values (day first) into window months by the month each day index falls in."""
    out = np.zeros((len(grid.months), *values.shape[1:]))
    month = (
        np.searchsorted(
            np.array([month_bounds(m)[0].toordinal() for m in grid.months]), day_index, side="right"
        )
        - 1
    )
    inside = (month >= 0) & (day_index <= grid.window_end)
    np.add.at(out, month[inside], values[inside])
    return out


def build_plan(config: ProcessorWorldConfig) -> dict[str, pl.DataFrame]:
    grid = day_grid(config)
    count, rate = expected_merchants(config, grid)
    attempts = (
        (rate * day_factor(config, grid))[:, :, None, None]
        * card_mix(config, grid, planted=False)[:, None, :, None]
        * initiation_share(config, grid, planted=False)[:, None, None, :]
    )
    approvals = attempts * authorization_rate(config, grid, planted=False)[:, None, None, :]
    ticket = np.array([config.verticals[v].avg_ticket for v in VERTICALS])[None, :, None, None]
    volume = approvals * ticket
    take = np.array([config.verticals[v].take_rate for v in VERTICALS])[None, :, None, None]
    fixed = np.array([config.verticals[v].fixed_fee for v in VERTICALS])[None, :, None, None]
    fees = take * volume + fixed * approvals
    ic_fixed = np.array([config.card_types[c].interchange_fixed for c in CARD_TYPES])[None, None, :, None]
    interchange = (
        interchange_pct(config, grid, planted=False)[:, None, :, None] * volume + ic_fixed * approvals
    )

    captured = {
        name: _by_month(grid, grid.days, x)
        for name, x in (
            ("attempts", attempts),
            ("transactions", approvals),
            ("payment_volume", volume),
            ("fees_captured", fees),
            ("interchange_captured", interchange),
        )
    }
    settle = settlement_days(grid.days, config.settlement_business_days)
    settled = {
        "fees_settled": _by_month(grid, settle, fees),
        "interchange_settled": _by_month(grid, settle, interchange),
    }
    processing = []
    for m, month in enumerate(grid.months):
        for v, lob in enumerate(VERTICALS):
            for c, card in enumerate(CARD_TYPES):
                for i, who in enumerate(INITIATIONS):
                    row: dict[str, str | float] = {
                        "month": month,
                        "lob": lob,
                        "card_type": card,
                        "initiation": who,
                    }
                    for name, values in (captured | settled).items():
                        row[name] = round(
                            float(values[m, v, c, i]), 4 if name in ("attempts", "transactions") else 2
                        )
                    processing.append(row)

    # Chargebacks: approvals (summed over card types) times the rate, spread by the dispute lag.
    n_days = len(grid.days)
    dispute = lag_pmf(config.chargeback_lag_median_days, config.chargeback_lag_sigma, n_days)
    repay = lag_pmf(config.recovery_lag_median_days, config.recovery_lag_sigma, n_days)
    rates = chargeback_rate(config)
    raised = np.zeros((n_days, len(VERTICALS), len(INITIATIONS)))
    recovered = np.zeros_like(raised)
    for v, lob in enumerate(VERTICALS):
        cfg = config.verticals[lob]
        for i in range(len(INITIATIONS)):
            raised[:, v, i] = np.convolve(approvals[:, v, :, i].sum(axis=1) * rates[v, i], dispute)[:n_days]
            recovered[:, v, i] = np.convolve(raised[:, v, i] * cfg.recovery_rate, repay)[:n_days]
    ticket_v = np.array([config.verticals[v].avg_ticket for v in VERTICALS])[None, :, None]
    recovery_v = np.array([config.verticals[v].recovery_rate for v in VERTICALS])[None, :, None]
    by_month = {
        "chargebacks": _by_month(grid, grid.days, raised),
        "disputed_amount": _by_month(grid, grid.days, raised * ticket_v),
        "net_amount": _by_month(grid, grid.days, raised * ticket_v * (1 - recovery_v)),
        "debited_amount": _by_month(grid, grid.days, raised * ticket_v),
        "recovered_amount": _by_month(grid, grid.days, recovered * ticket_v),
    }
    chargebacks = [
        {"month": month, "lob": lob, "initiation": who}
        | {
            name: round(float(values[m, v, i]), 4 if name == "chargebacks" else 2)
            for name, values in by_month.items()
        }
        for m, month in enumerate(grid.months)
        for v, lob in enumerate(VERTICALS)
        for i, who in enumerate(INITIATIONS)
    ]

    month_end = np.array([month_bounds(m)[1].toordinal() for m in grid.months]) - int(grid.days[0])
    merchants = [
        {
            "month": month,
            "lob": lob,
            "active_merchants": round(float(count[month_end[m], v]), 4),
            "new_merchants": round(config.new_merchants_per_month * config.verticals[lob].merchant_share, 4),
        }
        for m, month in enumerate(grid.months)
        for v, lob in enumerate(VERTICALS)
    ]
    return {
        "plan_processing": pl.DataFrame(processing),
        "plan_chargebacks": pl.DataFrame(chargebacks),
        "plan_merchants": pl.DataFrame(merchants),
        "plan_opex": pl.DataFrame(
            {"month": list(grid.months), "amount": [config.opex_monthly] * len(grid.months)}
        ),
    }
