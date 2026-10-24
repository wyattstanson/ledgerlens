"""NetSuite-shaped general ledger assembly, shared by the insurer and the processor (ADR 0001).

A world describes its postings as families: a frame of transaction headers (record type, date, entity, amount,
line of business, class, the payments-feed event behind it, memo) plus the account each family debits and the
account it credits. Every transaction gets exactly two lines, a debit and an equal credit, so debits equal
credits by construction (and a dbt test checks it). Ids follow date, record type, entity and amount, so the
same postings always get the same ids.
"""

import calendar
from collections.abc import Sequence
from datetime import date

import polars as pl

RECORD_PREFIX = {"invoice": "INV", "custpymt": "PYMT", "journalentry": "JE", "deposit": "DEP"}
HEADER = ("recordtype", "trandate", "entity", "amount", "lob", "cls", "event_id", "memo")
Family = tuple[pl.DataFrame, int, int]  # (headers, debit account, credit account)
Account = tuple[int, str, str]  # (number, name, NetSuite account type)


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


def accounting_periods(months: Sequence[str]) -> pl.DataFrame:
    """One period per month; the latest is open, every older one closed."""
    return pl.DataFrame(
        {
            "id": list(range(1, len(months) + 1)),
            "periodname": [month_bounds(m)[0].strftime("%b %Y") for m in months],
            "startdate": [month_bounds(m)[0] for m in months],
            "enddate": [month_bounds(m)[1] for m in months],
            "closed": [i < len(months) - 1 for i in range(len(months))],
        }
    )


def post(
    families: Sequence[Family], periods: pl.DataFrame, accounts: Sequence[Account], subsidiary: str
) -> dict[str, pl.DataFrame]:
    """The ``transaction``, ``transactionline``, ``account``, ``accountingperiod`` and ``subsidiary``
    tables."""
    headers = [
        frame.select(HEADER).with_columns(pl.lit(debit).alias("debit"), pl.lit(credit).alias("credit"))
        for frame, debit, credit in families
    ]
    tx = (
        pl.concat(headers)
        .sort("trandate", "recordtype", "entity", "amount")
        .with_row_index("id", offset=1)
        .with_columns(pl.col("id").cast(pl.Int64))
    )
    period_of = {row["startdate"].strftime("%Y-%m"): row["id"] for row in periods.iter_rows(named=True)}
    prefix = pl.col("recordtype").replace_strict(RECORD_PREFIX)
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
            "id": [a[0] for a in accounts],
            "acctnumber": [str(a[0]) for a in accounts],
            "fullname": [a[1] for a in accounts],
            "accttype": [a[2] for a in accounts],
            "parent": [None] * len(accounts),
        },
        schema_overrides={"parent": pl.Int64},
    )
    subsidiaries = pl.DataFrame(
        {"id": [1], "name": [subsidiary], "currency": ["USD"], "parent": [None]},
        schema_overrides={"parent": pl.Int64},
    )
    return {
        "transaction": transaction,
        "transactionline": transactionline,
        "account": account,
        "accountingperiod": periods,
        "subsidiary": subsidiaries,
    }
