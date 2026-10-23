"""Seeded synthetic P&C insurer: a payments feed and a NetSuite-shaped ledger (BUILD_PROMPT 10.1, ADR 0001).

Policies renew yearly and bill instalments by pay plan; each instalment is paid by a method with its own
on-time rate and lateness. Claims arrive by line with seasonality, are approved after a lag and paid in one or
two legs through a channel with its own payout lag and handling cost; some checks bounce and are reissued.

The ledger is generated from those events, so it reconciles to them exactly: an invoice per instalment, a
customer payment per premium received, and journal entries for claim payouts, returned payouts, handling
costs (LAE) and operating expense. An opening entry carries receivables for instalments due just before the
window. The planted misses in ``config.planted`` move actuals away from the plan (see ``fpa/plan.py``).
"""

import calendar
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import polars as pl

from ledgerlens.core.db import connect
from ledgerlens.fpa.config import CHANNELS, LINES, METHODS, WorldConfig

PAY_PLAN_INSTALMENTS = {"annual": 1, "quarterly": 4, "monthly": 12}
PRE_WINDOW_DUE_DAYS = 92  # instalments due this long before the window are carried as opening receivables
PRE_WINDOW_CLAIM_DAYS = 45  # claims approved just before the window can still pay out inside it
MORTGAGEE_SHARE = 0.4  # a multiparty homeowners payout sends this share to the mortgagee by check
RETURN_DAYS, REISSUE_DAYS = 10, 3
ACCOUNTS = (
    (1000, "Cash - Operating", "Bank"),
    (1200, "Premiums Receivable", "AcctRec"),
    (3000, "Opening Balance Equity", "Equity"),
    (4000, "Premium Revenue", "Income"),
    (5000, "Claims Paid", "Expense"),
    (5100, "Loss Adjustment Expense", "Expense"),
    (6000, "Operating Expense", "Expense"),
)


@dataclass(frozen=True)
class FpaWorld:
    policies: pl.DataFrame
    instalments: pl.DataFrame
    claims: pl.DataFrame
    payouts: pl.DataFrame
    payment_events: pl.DataFrame
    opex: pl.DataFrame
    transaction: pl.DataFrame
    transactionline: pl.DataFrame
    account: pl.DataFrame
    accountingperiod: pl.DataFrame
    subsidiary: pl.DataFrame
    months: tuple[str, ...]
    seed: int
    meta: dict[str, float | str] = field(default_factory=dict)


def month_list(end_month: str, months: int) -> tuple[str, ...]:
    year, month = map(int, end_month.split("-"))
    out = []
    for _ in range(months):
        out.append(f"{year:04d}-{month:02d}")
        year, month = (year, month - 1) if month > 1 else (year - 1, 12)
    return tuple(reversed(out))


def month_bounds(month: str) -> tuple[date, date]:
    year, mon = map(int, month.split("-"))
    return date(year, mon, 1), date(year, mon, calendar.monthrange(year, mon)[1])


def _days(d: date) -> int:
    return d.toordinal()


def _to_date(ordinals: np.ndarray) -> list[date | None]:
    return [None if o < 0 else date.fromordinal(int(o)) for o in ordinals]


def _lognormal(rng: np.random.Generator, median: float, sigma: float, size: int) -> np.ndarray:
    return median * rng.lognormal(0.0, sigma, size)


def _month_index(ordinals: np.ndarray, starts: np.ndarray) -> np.ndarray:
    """Index of the window month each day falls in (-1 before the window)."""
    return np.searchsorted(starts, ordinals, side="right") - 1


def generate_world(config: WorldConfig, seed: int) -> FpaWorld:  # noqa: PLR0912, PLR0915 (one linear recipe)
    rng = np.random.default_rng(seed)
    months = month_list(config.end_month, config.months)
    bounds = [month_bounds(m) for m in months]
    starts = np.array([_days(s) for s, _ in bounds])
    window_start, window_end = starts[0], _days(bounds[-1][1])
    planted = config.planted

    # --- policies and terms ------------------------------------------------------------------------------
    lob_names = np.array(LINES)
    lob_p = np.array([config.lines[line].share for line in LINES])
    initial = config.policies
    lob = list(rng.choice(lob_names, initial, p=lob_p))
    inception = list(window_start - rng.integers(1, 366, initial))
    for m, (start, end) in enumerate(bounds):
        n = rng.poisson(config.new_business_per_month)
        lob += list(rng.choice(lob_names, n, p=lob_p))
        if m >= planted.volume_start_month:  # planted: extra new business in one line
            extra = rng.poisson(
                config.new_business_per_month
                * config.lines[planted.volume_line].share
                * planted.volume_new_business_lift
            )
            lob += [planted.volume_line] * extra
            n += extra
        inception += list(rng.integers(_days(start), _days(end) + 1, n))
    lob_arr, inception_arr = np.array(lob), np.array(inception)
    count = len(lob_arr)
    policy_ids = np.array([f"POL-{i + 1:06d}" for i in range(count)])
    premium = np.empty(count)
    for name in LINES:
        cfg = config.lines[name]
        mask = lob_arr == name
        mu = np.log(cfg.annual_premium) - cfg.premium_sigma**2 / 2
        premium[mask] = rng.lognormal(mu, cfg.premium_sigma, mask.sum())
    plans = np.array(list(config.pay_plans))
    plan_arr = rng.choice(plans, count, p=np.array(list(config.pay_plans.values())))
    method_arr = rng.choice(np.array(METHODS), count, p=np.array([config.methods[m].mix for m in METHODS]))
    switches = (method_arr == "ach") & (rng.random(count) < planted.mix_ach_to_card)  # planted mix shift

    terms_policy, terms_start, terms_premium = [], [], []
    for i in range(count):
        k, start = 0, int(inception_arr[i])
        while start <= window_end:
            terms_policy.append(i)
            terms_start.append(start)
            terms_premium.append(premium[i] * (1 + config.annual_rate_change) ** k)
            if rng.random() > config.renewal_rate:
                break
            k += 1
            start += 365
    t_policy, t_start, t_premium = np.array(terms_policy), np.array(terms_start), np.array(terms_premium)
    t_end = t_start + 365

    # --- instalments and premium payments ----------------------------------------------------------------
    rows_policy, rows_due, rows_amount = [], [], []
    for plan_name, n in PAY_PLAN_INSTALMENTS.items():
        sel = np.flatnonzero(plan_arr[t_policy] == plan_name)
        for k in range(n):
            due = t_start[sel] + round(k * 365 / n)
            keep = (due >= window_start - PRE_WINDOW_DUE_DAYS) & (due <= window_end)
            rows_policy.append(t_policy[sel][keep])
            rows_due.append(due[keep])
            rows_amount.append(np.round(t_premium[sel][keep] / n, 2))
    i_policy = np.concatenate(rows_policy)
    i_due = np.concatenate(rows_due)
    i_amount = np.concatenate(rows_amount)
    order = np.lexsort((i_policy, i_due))
    i_policy, i_due, i_amount = i_policy[order], i_due[order], i_amount[order]
    i_month = _month_index(i_due, starts)
    i_method = method_arr[i_policy].astype(object)
    i_method[switches[i_policy] & (i_month >= planted.mix_start_month)] = "card"
    n_inst = len(i_due)
    paid = np.full(n_inst, -1)
    for name in METHODS:
        cfg = config.methods[name]
        mask = i_method == name
        size = int(mask.sum())
        on_time = rng.random(size) < cfg.on_time_rate
        early = rng.integers(0, 6, size)
        late = np.ceil(_lognormal(rng, cfg.late_median_days, 0.6, size)).astype(int)
        never = rng.random(size) < cfg.nonpayment_rate
        due = i_due[mask]
        paid[mask] = np.where(on_time, due - early, np.where(never, -1, due + late))
    paid[paid > window_end] = -1  # paid after the window: still outstanding
    # An early payer of an instalment due in the window's first days pays when the window opens, so every
    # invoice in the window meets its payment inside the window too.
    early_payers = (paid >= 0) & (paid < window_start) & (i_due >= window_start)
    paid[early_payers] = window_start
    instalments = pl.DataFrame(
        {
            "instalment_id": [f"INS-{i + 1:07d}" for i in range(n_inst)],
            "policy_id": policy_ids[i_policy],
            "lob": lob_arr[i_policy],
            "method": i_method.astype(str),
            "due_date": _to_date(i_due),
            "amount": i_amount,
            "paid_date": _to_date(paid),
        }
    )

    # --- claims and payouts -------------------------------------------------------------------------------
    claim_rows: list[tuple] = []
    claim_months = [(m, s, e) for m, (s, e) in enumerate(bounds)]
    claim_months.insert(
        0, (-1, bounds[0][0] - timedelta(days=PRE_WINDOW_CLAIM_DAYS), bounds[0][0] - timedelta(days=1))
    )
    for m, start, end in claim_months:
        mid = _days(start) + (_days(end) - _days(start)) // 2
        active = (t_start <= mid) & (t_end > mid)
        calendar_month = start.month
        for name in LINES:
            cfg = config.lines[name]
            pool = t_policy[active & (lob_arr[t_policy] == name)]
            season = 1 + cfg.seasonal_amplitude * np.cos(
                2 * np.pi * (calendar_month - cfg.seasonal_peak_month) / 12
            )
            share = (
                (_days(end) - _days(start) + 1) / ((bounds[0][1] - bounds[0][0]).days + 1) if m < 0 else 1.0
            )
            n = rng.poisson(len(pool) * cfg.claim_frequency * season * share)
            if n == 0:
                continue
            who = rng.choice(pool, n)
            occurred = rng.integers(_days(start), _days(end) + 1, n)
            claim_rows += [(int(w), int(o), name) for w, o in zip(who, occurred, strict=True)]
    c_policy = np.array([r[0] for r in claim_rows])
    c_occurred = np.array([r[1] for r in claim_rows])
    c_lob = np.array([r[2] for r in claim_rows])
    c_approved = c_occurred + np.ceil(
        _lognormal(rng, config.approval_lag_median_days, 0.5, len(c_policy))
    ).astype(int)
    keep = (c_approved <= window_end) & (c_approved >= window_start - PRE_WINDOW_CLAIM_DAYS)
    c_policy, c_occurred, c_lob, c_approved = c_policy[keep], c_occurred[keep], c_lob[keep], c_approved[keep]
    order = np.lexsort((c_policy, c_approved))
    c_policy, c_occurred, c_lob, c_approved = (
        c_policy[order],
        c_occurred[order],
        c_lob[order],
        c_approved[order],
    )
    n_claims = len(c_policy)
    severity = np.empty(n_claims)
    for name in LINES:
        cfg = config.lines[name]
        mask = c_lob == name
        severity[mask] = _lognormal(rng, cfg.severity_median, cfg.severity_sigma, int(mask.sum()))
    lifted = (c_lob == planted.severity_line) & (
        _month_index(c_approved, starts) >= planted.severity_start_month
    )
    severity = np.round(np.where(lifted, severity * (1 + planted.severity_lift), severity), 2)
    c_channel = rng.choice(
        np.array(CHANNELS), n_claims, p=np.array([config.channels[c].mix for c in CHANNELS])
    )
    claim_ids = np.array([f"CLM-{i + 1:06d}" for i in range(n_claims)])
    claims = pl.DataFrame(
        {
            "claim_id": claim_ids,
            "policy_id": policy_ids[c_policy],
            "lob": c_lob,
            "channel": c_channel,
            "occurred_date": _to_date(c_occurred),
            "approved_date": _to_date(c_approved),
            "severity": severity,
        }
    )
    multiparty = np.array([rng.random() < config.lines[line].multiparty_share for line in c_lob])
    legs = []  # (claim index, leg, channel, amount)
    for i in range(n_claims):
        if multiparty[i]:
            mortgagee = round(severity[i] * MORTGAGEE_SHARE, 2)
            legs += [
                (i, "claimant", c_channel[i], round(severity[i] - mortgagee, 2)),
                (i, "mortgagee", "check", mortgagee),
            ]
        else:
            legs.append((i, "claimant", c_channel[i], severity[i]))
    l_claim = np.array([x[0] for x in legs])
    l_leg = np.array([x[1] for x in legs])
    l_channel = np.array([x[2] for x in legs])
    l_amount = np.array([x[3] for x in legs])
    paid_at = np.empty(len(legs))  # fractional ordinal days
    for name in CHANNELS:
        cfg = config.channels[name]
        mask = l_channel == name
        size = int(mask.sum())
        median = (
            planted.timing_actual_lag_median_days if name == planted.timing_channel else cfg.lag_median_days
        )
        lag = _lognormal(rng, median, cfg.lag_sigma, size) if median > 0 else rng.uniform(0, 0.2, size)
        paid_at[mask] = c_approved[l_claim[mask]] + 0.375 + lag  # approvals land at 09:00
    cost = np.array([config.channels[c].handling_cost for c in l_channel])
    cost[l_channel == planted.lae_channel] = planted.lae_actual_handling_cost
    returned = rng.random(len(legs)) < np.array([config.channels[c].return_rate for c in l_channel])
    returned_at = np.where(returned, paid_at + RETURN_DAYS, np.nan)
    reissue_at = (
        returned_at + REISSUE_DAYS + _lognormal(rng, config.channels["ach"].lag_median_days, 0.4, len(legs))
    )
    reissued = np.flatnonzero(returned)  # each bounced check is paid again by ACH
    n_legs = len(legs)
    payouts = _timestamped(
        pl.DataFrame(
            {
                "payout_id": [f"PAY-{i + 1:07d}" for i in range(n_legs + len(reissued))],
                "claim_id": np.concatenate([claim_ids[l_claim], claim_ids[l_claim[reissued]]]),
                "lob": np.concatenate([c_lob[l_claim], c_lob[l_claim[reissued]]]),
                "leg": np.concatenate([l_leg, np.full(len(reissued), "reissue")]),
                "channel": np.concatenate([l_channel, np.full(len(reissued), "ach")]),
                "amount": np.concatenate([l_amount, l_amount[reissued]]),
                "handling_cost": np.concatenate(
                    [cost, np.full(len(reissued), config.channels["ach"].handling_cost)]
                ),
                "approved_date": _to_date(
                    np.concatenate([c_approved[l_claim], c_approved[l_claim[reissued]]])
                ),
                "paid_ts": np.concatenate([paid_at, reissue_at[reissued]]),
                "returned_ts": np.concatenate([returned_at, np.full(len(reissued), np.nan)]),
            }
        ),
        window_end,
    )

    # --- payments feed: what a carrier payments network would report --------------------------------------
    premium_events = instalments.filter(pl.col("paid_date").is_not_null()).select(
        pl.lit("premium_received").alias("kind"),
        (pl.col("paid_date").cast(pl.Datetime("us")) + pl.duration(hours=12)).alias("event_ts"),
        "amount",
        pl.col("method").alias("instrument"),
        "lob",
        "policy_id",
        pl.lit(None, dtype=pl.String).alias("claim_id"),
        pl.col("instalment_id").alias("ref_id"),
    )
    paid_events = payouts.filter(pl.col("paid_ts").is_not_null()).select(
        pl.lit("claim_paid").alias("kind"),
        pl.col("paid_ts").alias("event_ts"),
        "amount",
        pl.col("channel").alias("instrument"),
        "lob",
        pl.lit(None, dtype=pl.String).alias("policy_id"),
        "claim_id",
        pl.col("payout_id").alias("ref_id"),
    )
    returned_events = payouts.filter(pl.col("returned_ts").is_not_null()).select(
        pl.lit("payout_returned").alias("kind"),
        pl.col("returned_ts").alias("event_ts"),
        "amount",
        pl.col("channel").alias("instrument"),
        "lob",
        pl.lit(None, dtype=pl.String).alias("policy_id"),
        "claim_id",
        pl.col("payout_id").alias("ref_id"),
    )
    window = (pl.col("event_ts").dt.date() >= date.fromordinal(int(window_start))) & (
        pl.col("event_ts").dt.date() <= date.fromordinal(int(window_end))
    )
    events = (
        pl.concat([premium_events, paid_events, returned_events])
        .filter(window)
        .sort("event_ts", "kind", "ref_id")
        .with_row_index("n", offset=1)
        .with_columns(pl.format("PE-{}", pl.col("n").cast(pl.String).str.zfill(7)).alias("event_id"))
        .drop("n")
        .with_columns(pl.col("event_ts").dt.date().alias("event_date"))
    )

    opex = pl.DataFrame(
        {
            "month": list(months),
            "amount": np.round(config.opex_monthly * (1 + rng.normal(0, config.opex_noise, len(months))), 2),
        }
    )
    ledger = _ledger(instalments, events, payouts, opex, months, window_start)
    in_window = instalments.filter(pl.col("due_date") >= date.fromordinal(int(window_start)))
    last_year = in_window.filter(pl.col("due_date") >= month_bounds(months[-12])[0])
    meta: dict[str, float | str] = {
        "start_month": months[0],
        "end_month": months[-1],
        "seed": float(seed),
        "annual_premium": round(float(last_year["amount"].sum()), 2),
        "policies_written": float(count),
        "claims": float(n_claims),
    }
    policies = pl.DataFrame(
        {
            "policy_id": policy_ids,
            "lob": lob_arr,
            "pay_plan": plan_arr,
            "method": method_arr,
            "inception_date": _to_date(inception_arr),
            "annual_premium": np.round(premium, 2),
        }
    )
    return FpaWorld(
        policies=policies,
        instalments=instalments,
        claims=claims,
        payouts=payouts,
        payment_events=events,
        opex=opex,
        months=months,
        seed=seed,
        meta=meta,
        **ledger,
    )


def _timestamped(payouts: pl.DataFrame, window_end: int) -> pl.DataFrame:
    """Fractional ordinal days to timestamps; anything after the window has not happened yet."""
    epoch = date.fromordinal(1)

    def ts(column: str) -> pl.Expr:
        days = pl.col(column)
        stamp = (
            pl.lit(epoch).cast(pl.Datetime("us"))
            + pl.duration(days=days.floor() - 1)
            + pl.duration(seconds=((days - days.floor()) * 86_400).round())
        )
        return pl.when(days.is_nan() | (days >= window_end + 1)).then(None).otherwise(stamp).alias(column)

    return payouts.with_columns(ts("paid_ts"), ts("returned_ts")).with_columns(
        pl.when(pl.col("paid_ts").is_null()).then(None).otherwise(pl.col("returned_ts")).alias("returned_ts")
    )


def _ledger(
    instalments: pl.DataFrame,
    events: pl.DataFrame,
    payouts: pl.DataFrame,
    opex: pl.DataFrame,
    months: tuple[str, ...],
    window_start: int,
) -> dict[str, pl.DataFrame]:
    """NetSuite-shaped GL (ADR 0001): every posting comes from an event, so the ledger reconciles to them."""
    first_day = date.fromordinal(int(window_start))
    periods = pl.DataFrame(
        {
            "id": list(range(1, len(months) + 1)),
            "periodname": [month_bounds(m)[0].strftime("%b %Y") for m in months],
            "startdate": [month_bounds(m)[0] for m in months],
            "enddate": [month_bounds(m)[1] for m in months],
            "closed": [i < len(months) - 1 for i in range(len(months))],  # the latest period is open
        }
    )
    # One header frame per posting family, each with its two lines: (debit account, credit account).
    opening = instalments.filter(
        (pl.col("due_date") < first_day)
        & (pl.col("paid_date").is_null() | (pl.col("paid_date") >= first_day))
    )
    lae = payouts.filter(pl.col("paid_ts").is_not_null() & (pl.col("paid_ts").dt.date() >= first_day))
    families = [
        (
            instalments.filter(pl.col("due_date") >= first_day).select(
                pl.lit("invoice").alias("recordtype"),
                pl.col("due_date").alias("trandate"),
                pl.col("policy_id").alias("entity"),
                "amount",
                "lob",
                pl.col("method").alias("cls"),
                pl.lit(None, dtype=pl.String).alias("event_id"),
                pl.lit("premium billed").alias("memo"),
            ),
            1200,
            4000,
        ),
        (
            opening.select(
                pl.lit("journalentry").alias("recordtype"),
                pl.lit(first_day).alias("trandate"),
                pl.col("policy_id").alias("entity"),
                "amount",
                "lob",
                pl.col("method").alias("cls"),
                pl.lit(None, dtype=pl.String).alias("event_id"),
                pl.lit("opening receivable").alias("memo"),
            ),
            1200,
            3000,
        ),
        (
            events.filter(pl.col("kind") == "premium_received").select(
                pl.lit("custpymt").alias("recordtype"),
                pl.col("event_date").alias("trandate"),
                pl.col("policy_id").alias("entity"),
                "amount",
                "lob",
                pl.col("instrument").alias("cls"),
                "event_id",
                pl.lit("premium received").alias("memo"),
            ),
            1000,
            1200,
        ),
        (
            events.filter(pl.col("kind") == "claim_paid").select(
                pl.lit("journalentry").alias("recordtype"),
                pl.col("event_date").alias("trandate"),
                pl.col("claim_id").alias("entity"),
                "amount",
                "lob",
                pl.col("instrument").alias("cls"),
                "event_id",
                pl.lit("claim payout").alias("memo"),
            ),
            5000,
            1000,
        ),
        (
            events.filter(pl.col("kind") == "payout_returned").select(
                pl.lit("journalentry").alias("recordtype"),
                pl.col("event_date").alias("trandate"),
                pl.col("claim_id").alias("entity"),
                "amount",
                "lob",
                pl.col("instrument").alias("cls"),
                "event_id",
                pl.lit("payout returned").alias("memo"),
            ),
            1000,
            5000,
        ),
        (
            lae.select(
                pl.lit("journalentry").alias("recordtype"),
                pl.col("paid_ts").dt.date().alias("trandate"),
                pl.col("claim_id").alias("entity"),
                pl.col("handling_cost").alias("amount"),
                "lob",
                pl.col("channel").alias("cls"),
                pl.lit(None, dtype=pl.String).alias("event_id"),
                pl.lit("claim handling cost").alias("memo"),
            ),
            5100,
            1000,
        ),
        (
            opex.select(
                pl.lit("journalentry").alias("recordtype"),
                pl.col("month")
                .map_elements(lambda m: month_bounds(m)[1], return_dtype=pl.Date)
                .alias("trandate"),
                pl.lit("OPEX").alias("entity"),
                "amount",
                pl.lit(None, dtype=pl.String).alias("lob"),
                pl.lit(None, dtype=pl.String).alias("cls"),
                pl.lit(None, dtype=pl.String).alias("event_id"),
                pl.lit("operating expense").alias("memo"),
            ),
            6000,
            1000,
        ),
    ]
    headers = []
    for frame, debit, credit in families:
        headers.append(frame.with_columns(pl.lit(debit).alias("debit"), pl.lit(credit).alias("credit")))
    tx = (
        pl.concat(headers)
        .sort("trandate", "recordtype", "entity", "amount")
        .with_row_index("id", offset=1)
        .with_columns(pl.col("id").cast(pl.Int64))
    )
    period_of = {row["startdate"].strftime("%Y-%m"): row["id"] for row in periods.iter_rows(named=True)}
    prefix = pl.col("recordtype").replace_strict({"invoice": "INV", "custpymt": "PYMT", "journalentry": "JE"})
    transaction = tx.select(
        "id",
        pl.format("{}-{}", prefix, pl.col("id").cast(pl.String).str.zfill(7)).alias("tranid"),
        "trandate",
        "recordtype",
        "entity",
        pl.lit(1).alias("subsidiary"),
        pl.col("trandate")
        .dt.strftime("%Y-%m")
        .replace_strict(period_of, return_dtype=pl.Int64)
        .alias("postingperiod"),
        pl.lit(True).alias("posting"),
        pl.lit("approved").alias("approvalstatus"),
        pl.lit("USD").alias("currency"),
        pl.lit(1.0).alias("exchangerate"),
        "amount",
        pl.col("amount").alias("foreignamount"),
        "memo",
        pl.col("event_id").alias("custbody_event_id"),  # links a posting to the payments-feed event behind it
    )
    side = [pl.col("id").alias("transaction"), "lob", "cls"]
    lines = pl.concat(
        [
            tx.select(*side, pl.col("debit").alias("account"), pl.col("amount")),
            tx.select(*side, pl.col("credit").alias("account"), -pl.col("amount")),
        ]
    ).sort("transaction", "account")
    transactionline = lines.with_row_index("id", offset=1).select(
        pl.col("id").cast(pl.Int64),
        "transaction",
        "account",
        "amount",
        pl.col("amount").alias("foreignamount"),
        pl.col("lob").alias("department"),
        pl.col("cls").alias("class"),
        pl.lit("US").alias("location"),
    )
    account = pl.DataFrame(
        {
            "id": [a[0] for a in ACCOUNTS],
            "acctnumber": [str(a[0]) for a in ACCOUNTS],
            "fullname": [a[1] for a in ACCOUNTS],
            "accttype": [a[2] for a in ACCOUNTS],
            "parent": [None] * len(ACCOUNTS),
        },
        schema_overrides={"parent": pl.Int64},
    )
    subsidiary = pl.DataFrame(
        {"id": [1], "name": ["LedgerLens Mutual (synthetic)"], "currency": ["USD"], "parent": [None]},
        schema_overrides={"parent": pl.Int64},
    )
    return {
        "transaction": transaction,
        "transactionline": transactionline,
        "account": account,
        "accountingperiod": periods,
        "subsidiary": subsidiary,
    }


RAW_TABLES = (
    "policies",
    "instalments",
    "claims",
    "payouts",
    "payment_events",
    "opex",
    "transaction",
    "transactionline",
    "account",
    "accountingperiod",
    "subsidiary",
)
PLAN_TABLES = ("plan_premium", "plan_claims", "plan_opex")  # what plan.build_plan returns


def write_world(warehouse: Path, world: FpaWorld, plan: dict[str, pl.DataFrame]) -> None:
    """Raw tables and the plan to schema ``raw_fpa``, where the dbt sources read them."""
    if set(plan) != set(PLAN_TABLES):
        raise ValueError(f"plan tables {sorted(plan)} differ from {PLAN_TABLES}")
    tables = (
        {name: getattr(world, name) for name in RAW_TABLES}
        | plan
        | {"world_meta": pl.DataFrame([world.meta])}
    )
    with connect(warehouse) as conn:
        conn.execute("create schema if not exists raw_fpa")
        for name, frame in tables.items():
            conn.register("_raw_frame", frame)  # an explicit name, never a lookup of a Python variable
            conn.execute(f'create or replace table raw_fpa."{name}" as select * from _raw_frame')  # noqa: S608
            conn.unregister("_raw_frame")
