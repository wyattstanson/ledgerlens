"""The budget: the same driver tree run on the base assumptions, as expected values (BUILD_PROMPT 10.1).

The plan knows nothing of the planted misses, so actuals beat or miss it for reasons the bridge can name:
more new business in one line (volume), heavier severity in another (rate), customers moving from ACH to
card (mix), slower checks (timing) and dearer check handling (LAE rate).

Lag profiles say what share of a month's due premium (or approved claims) turns into cash in the same
month, the next, and so on. They come from the same lateness and payout-lag distributions the generator
uses, estimated with a fixed-seed simulation so the plan is deterministic.
"""

import math

import numpy as np
import polars as pl

from ledgerlens.core.stats import seed_for
from ledgerlens.fpa.config import CHANNELS, LINES, METHODS, ChannelConfig, MethodConfig, WorldConfig
from ledgerlens.fpa.generate import MORTGAGEE_SHARE, PAY_PLAN_INSTALMENTS, month_bounds, month_list

DAYS_PER_MONTH = 365.25 / 12
PROFILE_DRAWS = 200_000
MAX_LAG_MONTHS = 6


def _profile(offsets_days: np.ndarray, start_day: np.ndarray, paid: np.ndarray) -> dict[int, float]:
    """Share of a month's amount that turns into cash k months later (k may be -1 for early payment)."""
    k = np.floor((start_day + offsets_days) / DAYS_PER_MONTH).astype(int)
    k = np.clip(k, -1, MAX_LAG_MONTHS)
    counts = np.bincount(k[paid] + 1, minlength=MAX_LAG_MONTHS + 2)
    return {i - 1: float(c) / len(paid) for i, c in enumerate(counts) if c}


def collection_profile(cfg: MethodConfig, seed: int, key: str) -> dict[int, float]:
    rng = seed_for(seed, f"plan|collect|{key}")
    day = rng.uniform(0, DAYS_PER_MONTH, PROFILE_DRAWS)
    on_time = rng.random(PROFILE_DRAWS) < cfg.on_time_rate
    early = -rng.integers(0, 6, PROFILE_DRAWS)
    late = np.ceil(cfg.late_median_days * rng.lognormal(0.0, 0.6, PROFILE_DRAWS))
    never = rng.random(PROFILE_DRAWS) < cfg.nonpayment_rate
    offset = np.where(on_time, early, late)
    return _profile(offset, day, on_time | ~never)


def payout_profile(cfg: ChannelConfig, seed: int, key: str, median: float | None = None) -> dict[int, float]:
    rng = seed_for(seed, f"plan|payout|{key}")
    day = rng.integers(0, int(DAYS_PER_MONTH), PROFILE_DRAWS) + 0.375
    lag_median = cfg.lag_median_days if median is None else median
    lag = (
        lag_median * rng.lognormal(0.0, cfg.lag_sigma, PROFILE_DRAWS)
        if lag_median > 0
        else rng.uniform(0, 0.2, PROFILE_DRAWS)
    )
    return _profile(lag, day, np.ones(PROFILE_DRAWS, dtype=bool))


def approval_profile(median_days: float, seed: int) -> dict[int, float]:
    rng = seed_for(seed, "plan|approval")
    day = rng.uniform(0, DAYS_PER_MONTH, PROFILE_DRAWS)
    lag = np.ceil(median_days * rng.lognormal(0.0, 0.5, PROFILE_DRAWS))
    return _profile(lag, day, np.ones(PROFILE_DRAWS, dtype=bool))


def _book(config: WorldConfig, months: int) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Expected policies in force and annual premium per line, month by month (index 0 = window start)."""
    out = {}
    stay = 1 - 1 / 12 + config.renewal_rate / 12
    raise_ = 1 - 1 / 12 + config.renewal_rate * (1 + config.annual_rate_change) / 12
    for line in LINES:
        cfg = config.lines[line]
        count, premium = np.empty(months), np.empty(months)
        count[0], premium[0] = config.policies * cfg.share, config.policies * cfg.share * cfg.annual_premium
        for m in range(1, months):
            new = config.new_business_per_month * cfg.share
            count[m] = count[m - 1] * stay + new
            premium[m] = premium[m - 1] * raise_ + new * cfg.annual_premium
        out[line] = (count, premium)
    return out


def _lagged(series: np.ndarray, profile: dict[int, float], steady: float) -> np.ndarray:
    """Cash per month: each month's amount spread forward by the lag profile; ``steady`` before the window."""
    out = np.zeros(len(series))
    for m in range(len(series)):
        for k, share in profile.items():
            source = m - k
            value = series[source] if 0 <= source < len(series) else (steady if source < 0 else series[-1])
            out[m] += value * share
    return out


def build_plan(config: WorldConfig, seed: int) -> dict[str, pl.DataFrame]:
    months = month_list(config.end_month, config.months)
    n = len(months)
    book = _book(config, n + 1)  # one month past the window: early payers of next month's dues
    per_policy = sum(share * PAY_PLAN_INSTALMENTS[plan] for plan, share in config.pay_plans.items()) / 12

    # Instalments fall due every 365/n days, so a month bills in proportion to its days (February is light).
    days = np.array([(month_bounds(m)[1] - month_bounds(m)[0]).days + 1 for m in months] + [DAYS_PER_MONTH])
    premium_rows = []
    for line in LINES:
        count, total = book[line]
        due = total / 365.25 * days
        for method in METHODS:
            mcfg = config.methods[method]
            due_m = due * mcfg.mix
            collected = _lagged(due_m, collection_profile(mcfg, seed, method), due_m[0])
            for m, month in enumerate(months):
                premium_rows.append(
                    {
                        "month": month,
                        "lob": line,
                        "method": method,
                        "instalments_due": round(
                            float(count[m] * per_policy * mcfg.mix * days[m] / DAYS_PER_MONTH), 2
                        ),
                        "due_amount": round(float(due_m[m]), 2),
                        "collected_amount": round(float(collected[m]), 2),
                    }
                )

    approval = approval_profile(config.approval_lag_median_days, seed)
    claim_rows = []
    for line in LINES:
        cfg = config.lines[line]
        count, _ = book[line]
        calendar = np.array([month_bounds(m)[0].month for m in months])
        season = 1 + cfg.seasonal_amplitude * np.cos(2 * np.pi * (calendar - cfg.seasonal_peak_month) / 12)
        occurrences = count[:n] * cfg.claim_frequency * season
        approved = _lagged(occurrences, approval, float(occurrences[0]))
        mean_severity = cfg.severity_median * math.exp(cfg.severity_sigma**2 / 2)
        mortgagee = cfg.multiparty_share * MORTGAGEE_SHARE
        for channel in CHANNELS:
            ccfg = config.channels[channel]
            claims = approved * ccfg.mix
            amount = claims * mean_severity
            # Leg amounts and counts paid through this channel: claimant legs, plus mortgagee legs by check.
            leg_amount = amount * (1 - mortgagee)
            leg_count = claims.copy()
            if channel == "check":
                leg_amount = leg_amount + approved * mean_severity * mortgagee
                leg_count = leg_count + approved * cfg.multiparty_share
            profile = payout_profile(ccfg, seed, channel)
            paid = _lagged(leg_amount, profile, float(leg_amount[0]))
            payouts = _lagged(leg_count, profile, float(leg_count[0]))
            for m, month in enumerate(months):
                claim_rows.append(
                    {
                        "month": month,
                        "lob": line,
                        "channel": channel,
                        "approved_count": round(float(claims[m]), 4),
                        "approved_amount": round(float(amount[m]), 2),
                        "payouts": round(float(payouts[m]), 4),
                        "paid_amount": round(float(paid[m]), 2),
                        "lae_amount": round(float(payouts[m] * ccfg.handling_cost), 2),
                    }
                )

    return {
        "plan_premium": pl.DataFrame(premium_rows),
        "plan_claims": pl.DataFrame(claim_rows),
        "plan_opex": pl.DataFrame({"month": list(months), "amount": [config.opex_monthly] * n}),
    }
