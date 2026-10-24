"""Seeded synthetic payment processor: a payments feed and a NetSuite-shaped ledger (BUILD_PROMPT 10.1,
``--variant processor``; ADR 0005).

Merchants in three verticals onboard and churn; each sends card authorization attempts, by day, card type and
initiation (a person, or an AI agent buying for one). Issuers approve a share. An approved purchase earns the
processor its merchant discount (a percentage plus a fixed fee) and costs it interchange and network fees,
which depend on the card. Each day's captures settle with the card networks in batches (vertical by card
type) one business day later. Weeks later some purchases are disputed: the network debits the processor (a
chargeback), the processor recovers most of them from the merchant, and writes off what it has not recovered
after ``write_off_days``.

Counts are drawn per (day, vertical, card type, initiation) cell rather than per purchase, so the world stays
small; merchants count through how many are active and how big they are. Chargebacks are individual.

The ledger is generated from those records, so it reconciles to the payments feed exactly:

- on the capture date, fees (Dr settlement receivable, Cr processing fee revenue) and interchange (Dr
  interchange and network fees, Cr settlement receivable), one pair per batch;
- on the settlement date, a deposit of the batch's fees net of interchange (Dr cash, Cr settlement
  receivable);
- a chargeback debited (Dr merchant receivable, Cr cash), recovered (a customer payment: Dr cash, Cr merchant
  receivable) or written off (Dr chargeback losses, Cr merchant receivable);
- operating expense, and opening entries for batches and chargebacks still open when the window starts.

The planted misses in ``config.planted`` move actuals away from the plan (see ``processor/plan.py``).
"""

import math
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl

from ledgerlens.fpa.config import CARD_TYPES, INITIATIONS, VERTICALS, ProcessorWorldConfig
from ledgerlens.fpa.gl import Family, accounting_periods, month_bounds, post
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
    opening_counts,
    settlement_days,
)
from ledgerlens.fpa.raw import VARIANT_TABLES, write_raw

ACCOUNTS = (
    (1000, "Cash - Operating", "Bank"),
    (1100, "Settlement Receivable", "AcctRec"),
    (1300, "Merchant Receivables - Chargebacks", "AcctRec"),
    (3000, "Opening Balance Equity", "Equity"),
    (4100, "Processing Fee Revenue", "Income"),
    (5200, "Interchange and Network Fees", "Expense"),
    (5300, "Chargeback Losses", "Expense"),
    (6000, "Operating Expense", "Expense"),
)
SUBSIDIARY = "LedgerLens Payments (synthetic)"
OPENING_TENURE_DAYS = 1_000  # merchants active at the start onboarded up to this long before the earliest day
SETTLEMENT_HOUR, CHARGEBACK_HOUR, RECOVERY_HOUR = 6, 10, 15  # when each kind of event lands
PLAN_TABLES = ("plan_processing", "plan_chargebacks", "plan_merchants", "plan_opex")
RAW_TABLES = tuple(t for t in VARIANT_TABLES["processor"] if t not in PLAN_TABLES)


@dataclass(frozen=True)
class ProcessorWorld:
    merchants: pl.DataFrame
    captures: pl.DataFrame
    chargebacks: pl.DataFrame
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


def _dates(ordinals: np.ndarray) -> list[date | None]:
    return [None if o < 0 else date.fromordinal(int(o)) for o in ordinals]


def _merchants(config: ProcessorWorldConfig, grid: DayGrid, rng: np.random.Generator) -> pl.DataFrame:
    """Merchants active at the start and those onboarded each month; lifetimes are exponential.

    The plan starts from the book as it stood at the window start, so the opening merchants come in the
    configured shares and their sizes average exactly one in each vertical. Merchants onboarded later arrive
    at random and keep their random sizes."""
    shares = np.array([config.verticals[v].merchant_share for v in VERTICALS])
    hazard = daily_hazard(config)
    first_day = int(grid.days[0])
    lob = [name for name, n in zip(VERTICALS, opening_counts(config), strict=True) for _ in range(n)]
    onboard = list(first_day - rng.integers(1, OPENING_TENURE_DAYS, config.merchants))
    life_from = [grid.window_start] * config.merchants  # opening merchants start churning with the window
    planted = config.planted
    for m, month in enumerate(grid.months):
        start, end = (d.toordinal() for d in month_bounds(month))
        for v, name in enumerate(VERTICALS):
            lift = 1 + planted.volume_new_merchant_lift if (
                name == planted.volume_vertical and m >= planted.volume_start_month
            ) else 1.0  # fmt: skip
            n = int(rng.poisson(config.new_merchants_per_month * shares[v] * lift))
            days = list(rng.integers(start, end + 1, n))
            lob += [name] * n
            onboard += days
            life_from += days
    count = len(lob)
    lob_arr = np.array(lob)
    lifetime = rng.exponential(1 / hazard, count) if hazard > 0 else np.full(count, np.inf)
    churn = np.array(life_from, dtype=float) + np.ceil(lifetime)
    churn_ord = np.where(churn > grid.window_end, -1, churn).astype(np.int64)
    size = rng.lognormal(-(config.merchant_size_sigma**2) / 2, config.merchant_size_sigma, count)
    opening = np.arange(count) < config.merchants
    for name in VERTICALS:
        sel = opening & (lob_arr == name)
        if sel.any():
            size[sel] /= size[sel].mean()
    per_day = np.array([config.verticals[v].attempts_per_merchant_day for v in lob_arr]) * size
    return pl.DataFrame(
        {
            "merchant_id": [f"M-{i + 1:05d}" for i in range(count)],
            "lob": lob_arr,
            "onboard_date": _dates(np.array(onboard)),
            "churn_date": _dates(churn_ord),
            "attempts_per_day": np.round(per_day, 4),
        }
    )


def active_rate(merchants: pl.DataFrame, grid: DayGrid) -> np.ndarray:
    """(day, vertical): the sum of active merchants' expected daily attempts."""
    first = int(grid.days[0])
    n_days = len(grid.days)
    out = np.zeros((len(VERTICALS), n_days + 1))
    onboard = np.array([d.toordinal() for d in merchants["onboard_date"]])
    churn = np.array([n_days + first if d is None else d.toordinal() for d in merchants["churn_date"]])
    rate = merchants["attempts_per_day"].to_numpy()
    lob = merchants["lob"].to_numpy()
    for v, name in enumerate(VERTICALS):
        sel = lob == name
        np.add.at(out[v], np.clip(onboard[sel] - first, 0, n_days), rate[sel])
        np.add.at(out[v], np.clip(churn[sel] - first, 0, n_days), -rate[sel])
    return np.cumsum(out, axis=1)[:, :n_days].T


def _cells(config: ProcessorWorldConfig, grid: DayGrid, merchants: pl.DataFrame, rng: np.random.Generator):
    """Attempts, approvals, volume, fees and interchange per (day, vertical, card type, initiation)."""
    expected = active_rate(merchants, grid) * day_factor(config, grid)  # (D, V)
    mix = card_mix(config, grid, planted=True)  # (D, C)
    who = initiation_share(config, grid, planted=True)  # (D, I)
    lam = expected[:, :, None, None] * mix[:, None, :, None] * who[:, None, None, :]
    attempts = rng.poisson(lam)
    auth = authorization_rate(config, grid, planted=True)  # (D, I)
    approvals = rng.binomial(attempts, np.broadcast_to(auth[:, None, None, :], attempts.shape))
    mean = np.array([config.verticals[v].avg_ticket for v in VERTICALS])[None, :, None, None]
    sigma = np.array([config.verticals[v].ticket_sigma for v in VERTICALS])[None, :, None, None]
    spread = mean * np.sqrt(np.exp(sigma**2) - 1)
    # The sum of n lognormal tickets, by its normal approximation (n is in the hundreds).
    volume = np.clip(
        approvals * mean + np.sqrt(approvals) * spread * rng.standard_normal(attempts.shape), 0, None
    )
    volume = np.round(volume, 2)
    take = np.array([config.verticals[v].take_rate for v in VERTICALS])[None, :, None, None]
    fixed = np.array([config.verticals[v].fixed_fee for v in VERTICALS])[None, :, None, None]
    fees = np.round(take * volume + fixed * approvals, 2)
    ic_pct = interchange_pct(config, grid, planted=True)[:, None, :, None]
    ic_fixed = np.array([config.card_types[c].interchange_fixed for c in CARD_TYPES])[None, None, :, None]
    interchange = np.round(ic_pct * volume + ic_fixed * approvals, 2)
    return attempts, approvals, volume, fees, interchange


def _captures(grid: DayGrid, config: ProcessorWorldConfig, cells) -> pl.DataFrame:
    """Capture records for days in the window, and for earlier days whose batches settle inside it."""
    attempts, approvals, volume, fees, interchange = cells
    settle = settlement_days(grid.days, config.settlement_business_days)
    d, v, c, i = np.nonzero(attempts)
    keep = (grid.days[d] >= grid.window_start) | (settle[d] >= grid.window_start)
    d, v, c, i = d[keep], v[keep], c[keep], i[keep]
    frame = pl.DataFrame(
        {
            "capture_date": _dates(grid.days[d]),
            "settle_date": _dates(settle[d]),
            "lob": np.array(VERTICALS)[v],
            "card_type": np.array(CARD_TYPES)[c],
            "initiation": np.array(INITIATIONS)[i],
            "attempts": attempts[d, v, c, i].astype(np.int64),
            "approvals": approvals[d, v, c, i].astype(np.int64),
            "volume": volume[d, v, c, i],
            "fee_amount": fees[d, v, c, i],
            "interchange_amount": interchange[d, v, c, i],
        }
    )
    batches = (
        frame.select("capture_date", "lob", "card_type")
        .unique()
        .sort("capture_date", "lob", "card_type")
        .with_row_index("n", offset=1)
        .select(
            "capture_date",
            "lob",
            "card_type",
            pl.format("BAT-{}", pl.col("n").cast(pl.String).str.zfill(6)).alias("batch_id"),
        )
    )
    return frame.join(batches, on=["capture_date", "lob", "card_type"]).sort(
        "capture_date", "lob", "card_type", "initiation"
    )


def _chargebacks(
    config: ProcessorWorldConfig, grid: DayGrid, merchants: pl.DataFrame, approvals: np.ndarray, rng
) -> pl.DataFrame:
    """Disputes on approved purchases, with recovery from the merchant or a write-off."""
    rate = chargeback_rate(config)  # (V, I)
    counts = rng.poisson(approvals * rate[None, :, None, :])
    flat = np.repeat(np.arange(counts.size), counts.ravel())
    d, v, c, i = np.unravel_index(flat, counts.shape)
    n = len(flat)
    avg = np.array([config.verticals[x].avg_ticket for x in VERTICALS])[v]
    sigma = np.array([config.verticals[x].ticket_sigma for x in VERTICALS])[v]
    amount = np.maximum(np.round(avg * rng.lognormal(-(sigma**2) / 2, sigma), 2), 0.01)
    purchase = grid.days[d]
    raised = purchase + np.ceil(
        rng.lognormal(math.log(config.chargeback_lag_median_days), config.chargeback_lag_sigma, n)
    ).astype(np.int64)
    recovery = np.array([config.verticals[x].recovery_rate for x in VERTICALS])[v]
    recovered = rng.random(n) < recovery
    recovered_on = raised + np.ceil(
        rng.lognormal(math.log(config.recovery_lag_median_days), config.recovery_lag_sigma, n)
    ).astype(np.int64)
    written_off_on = raised + config.write_off_days
    merchant = _pick_merchants(merchants, grid, d, v, rng)
    ws, we = grid.window_start, grid.window_end
    open_at_start = np.where(recovered, recovered_on >= ws, written_off_on >= ws)
    keep = (raised <= we) & ((raised >= ws) | open_at_start)
    rec = np.where(recovered & (recovered_on <= we), recovered_on, -1)
    off = np.where(~recovered & (written_off_on <= we), written_off_on, -1)
    frame = pl.DataFrame(
        {
            "merchant_id": merchant[keep],
            "lob": np.array(VERTICALS)[v[keep]],
            "card_type": np.array(CARD_TYPES)[c[keep]],
            "initiation": np.array(INITIATIONS)[i[keep]],
            "purchase_date": _dates(purchase[keep]),
            "raised_date": _dates(raised[keep]),
            "amount": amount[keep],
            "recovered_date": _dates(rec[keep]),
            "written_off_date": _dates(off[keep]),
            "_order": np.arange(n)[keep],
        }
    )
    return (
        frame.sort("raised_date", "_order")
        .with_row_index("n", offset=1)
        .with_columns(pl.format("CB-{}", pl.col("n").cast(pl.String).str.zfill(6)).alias("chargeback_id"))
        .drop("n", "_order")
    )


def _pick_merchants(
    merchants: pl.DataFrame, grid: DayGrid, d: np.ndarray, v: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """A merchant for each chargeback, by size, among those active in its vertical on the purchase date."""
    ids = merchants["merchant_id"].to_numpy()
    lob = merchants["lob"].to_numpy()
    onboard = np.array([x.toordinal() for x in merchants["onboard_date"]])
    churn = np.array(
        [np.iinfo(np.int64).max if x is None else x.toordinal() for x in merchants["churn_date"]]
    )
    size = merchants["attempts_per_day"].to_numpy()
    out = np.empty(len(d), dtype=object)
    day = grid.days[d]
    for purchase in np.unique(day):
        for x, name in enumerate(VERTICALS):
            sel = np.flatnonzero((day == purchase) & (v == x))
            if not len(sel):
                continue
            active = (lob == name) & (onboard <= purchase) & (churn > purchase)
            if not active.any():
                raise RuntimeError(
                    f"no active {name} merchant on {date.fromordinal(int(purchase))} to carry a chargeback"
                )
            weights = size[active] / size[active].sum()
            out[sel] = rng.choice(ids[active], len(sel), p=weights)
    return out.astype(str)


def _events(batches: pl.DataFrame, chargebacks: pl.DataFrame, grid: DayGrid) -> pl.DataFrame:
    """The processor's side of the networks' feeds: settlements, chargebacks and merchant recoveries."""
    first, last = date.fromordinal(grid.window_start), date.fromordinal(grid.window_end)

    def at(column: str, hour: int) -> pl.Expr:
        return (pl.col(column).cast(pl.Datetime("us")) + pl.duration(hours=hour)).alias("event_ts")

    nulls = [pl.lit(None, dtype=pl.String).alias(c) for c in ("policy_id", "claim_id")]
    settled = batches.filter(pl.col("settle_date").is_between(first, last)).select(
        pl.lit("settlement").alias("kind"),
        at("settle_date", SETTLEMENT_HOUR),
        pl.col("net").abs().alias("amount"),
        pl.when(pl.col("net") >= 0).then(pl.lit("in")).otherwise(pl.lit("out")).alias("direction"),
        pl.col("card_type").alias("instrument"),
        "lob",
        *nulls,
        pl.lit(None, dtype=pl.String).alias("merchant_id"),
        pl.col("batch_id").alias("ref_id"),
    )
    debited = chargebacks.filter(pl.col("raised_date") >= first).select(
        pl.lit("chargeback").alias("kind"),
        at("raised_date", CHARGEBACK_HOUR),
        "amount",
        pl.lit("out").alias("direction"),
        pl.col("card_type").alias("instrument"),
        "lob",
        *nulls,
        "merchant_id",
        pl.col("chargeback_id").alias("ref_id"),
    )
    recovered = chargebacks.filter(pl.col("recovered_date") >= first).select(
        pl.lit("chargeback_recovered").alias("kind"),
        at("recovered_date", RECOVERY_HOUR),
        "amount",
        pl.lit("in").alias("direction"),
        pl.col("card_type").alias("instrument"),
        "lob",
        *nulls,
        "merchant_id",
        pl.col("chargeback_id").alias("ref_id"),
    )
    return (
        pl.concat([settled, debited, recovered])
        .sort("event_ts", "kind", "ref_id")
        .with_row_index("n", offset=1)
        .with_columns(
            pl.format("PE-{}", pl.col("n").cast(pl.String).str.zfill(7)).alias("event_id"),
            pl.col("event_ts").dt.date().alias("event_date"),
        )
        .drop("n")
    )


def _header(
    frame: pl.DataFrame, recordtype: str, trandate: str | date, entity: str, amount: str, memo: str
) -> pl.DataFrame:
    """Headers for one posting family, with lob, class (the card type) and event when the frame has them."""

    def column_or_null(name: str) -> pl.Expr:
        return pl.col(name) if name in frame.columns else pl.lit(None, dtype=pl.String)

    return frame.select(
        pl.lit(recordtype).alias("recordtype"),
        (pl.lit(trandate) if isinstance(trandate, date) else pl.col(trandate)).alias("trandate"),
        pl.col(entity).alias("entity"),
        pl.col(amount).alias("amount"),
        column_or_null("lob").alias("lob"),
        column_or_null("card_type").alias("cls"),
        column_or_null("event_id").alias("event_id"),
        pl.lit(memo).alias("memo"),
    )


def _ledger(
    batches: pl.DataFrame,
    chargebacks: pl.DataFrame,
    events: pl.DataFrame,
    opex: pl.DataFrame,
    grid: DayGrid,
) -> dict[str, pl.DataFrame]:
    first = date.fromordinal(grid.window_start)
    last = date.fromordinal(grid.window_end)
    batch_event = events.filter(pl.col("kind") == "settlement").select(
        pl.col("ref_id").alias("batch_id"), "event_id"
    )
    cb_event = events.filter(pl.col("kind") == "chargeback").select(
        pl.col("ref_id").alias("chargeback_id"), "event_id"
    )
    rec_event = events.filter(pl.col("kind") == "chargeback_recovered").select(
        pl.col("ref_id").alias("chargeback_id"), "event_id"
    )
    captured = batches.filter(pl.col("capture_date") >= first)
    settled = batches.filter(pl.col("settle_date").is_between(first, last)).join(batch_event, on="batch_id")
    opening_batches = batches.filter((pl.col("capture_date") < first) & (pl.col("settle_date") >= first))
    raised = chargebacks.filter(pl.col("raised_date") >= first).join(cb_event, on="chargeback_id")
    recovered = chargebacks.filter(pl.col("recovered_date") >= first).join(rec_event, on="chargeback_id")
    written_off = chargebacks.filter(pl.col("written_off_date").is_not_null())
    opening_cb = chargebacks.filter(pl.col("raised_date") < first)
    opex_rows = opex.with_columns(
        pl.col("month").map_elements(lambda m: month_bounds(m)[1], return_dtype=pl.Date).alias("trandate"),
        pl.lit("OPEX").alias("entity"),
    )
    je, deposit, payment = "journalentry", "deposit", "custpymt"
    specs = [  # frame, record type, date, entity, amount, memo, debit, credit
        (captured, je, "capture_date", "batch_id", "fee", "processing fees captured", 1100, 4100),
        (captured, je, "capture_date", "batch_id", "interchange", "interchange assessed", 5200, 1100),
        (settled, deposit, "settle_date", "batch_id", "net", "network settlement", 1000, 1100),
        (opening_batches, je, first, "batch_id", "net", "opening settlement receivable", 1100, 3000),
        (raised, je, "raised_date", "merchant_id", "amount", "chargeback debited", 1300, 1000),
        (recovered, payment, "recovered_date", "merchant_id", "amount", "chargeback recovered", 1000, 1300),
        (written_off, je, "written_off_date", "merchant_id", "amount", "chargeback written off", 5300, 1300),
        (opening_cb, je, first, "merchant_id", "amount", "opening merchant receivable", 1300, 3000),
        (opex_rows, je, "trandate", "entity", "amount", "operating expense", 6000, 1000),
    ]
    families: list[Family] = [
        (_header(frame, kind, trandate, entity, amount, memo), debit, credit)
        for frame, kind, trandate, entity, amount, memo, debit, credit in specs
    ]
    return post(families, accounting_periods(grid.months), ACCOUNTS, SUBSIDIARY)


def batch_totals(captures: pl.DataFrame) -> pl.DataFrame:
    """One row per settlement batch with approvals: fees, interchange and the net the network settles."""
    return (
        captures.group_by("batch_id", "capture_date", "settle_date", "lob", "card_type")
        .agg(
            pl.col("approvals").sum(),
            pl.col("fee_amount").sum().round(2).alias("fee"),
            pl.col("interchange_amount").sum().round(2).alias("interchange"),
        )
        .filter(pl.col("approvals") > 0)
        .with_columns((pl.col("fee") - pl.col("interchange")).round(2).alias("net"))
        .sort("batch_id")
    )


def generate_world(config: ProcessorWorldConfig, seed: int) -> ProcessorWorld:
    rng = np.random.default_rng(seed)
    grid = day_grid(config)
    merchants = _merchants(config, grid, rng)
    cells = _cells(config, grid, merchants, rng)
    captures = _captures(grid, config, cells)
    chargebacks = _chargebacks(config, grid, merchants, cells[1], rng)
    batches = batch_totals(captures)
    events = _events(batches, chargebacks, grid)
    months = grid.months
    opex = pl.DataFrame(
        {
            "month": list(months),
            "amount": np.round(config.opex_monthly * (1 + rng.normal(0, config.opex_noise, len(months))), 2),
        }
    )
    ledger = _ledger(batches, chargebacks, events, opex, grid)
    last_year = captures.filter(pl.col("capture_date") >= month_bounds(months[-12])[0])
    fees, interchange = float(last_year["fee_amount"].sum()), float(last_year["interchange_amount"].sum())
    meta: dict[str, float | str] = {
        "start_month": months[0],
        "end_month": months[-1],
        "seed": float(seed),
        "annual_revenue": round(fees, 2),  # processing fees captured, last 12 months: ADR 0001's band
        "annual_net_revenue": round(fees - interchange, 2),  # after interchange and network fees
        "annual_payment_volume": round(float(last_year["volume"].sum()), 2),
        "merchants": float(merchants.height),
        "chargebacks": float(chargebacks.height),
    }
    return ProcessorWorld(
        merchants=merchants,
        captures=captures.select(
            "batch_id",
            "capture_date",
            "settle_date",
            "lob",
            "card_type",
            "initiation",
            "attempts",
            "approvals",
            "volume",
            "fee_amount",
            "interchange_amount",
        ),
        chargebacks=chargebacks,
        payment_events=events,
        opex=opex,
        months=months,
        seed=seed,
        meta=meta,
        **ledger,
    )


def write_world(
    warehouse: Path, world: ProcessorWorld, plan: dict[str, pl.DataFrame]
) -> dict[str, pl.DataFrame]:
    """Raw tables and the plan to ``raw_fpa``; the insurer's own tables are left empty."""
    if set(plan) != set(PLAN_TABLES):
        raise ValueError(f"plan tables {sorted(plan)} differ from {PLAN_TABLES}")
    frames = {name: getattr(world, name) for name in RAW_TABLES} | plan
    return write_raw(warehouse, "processor", frames, world.meta)
