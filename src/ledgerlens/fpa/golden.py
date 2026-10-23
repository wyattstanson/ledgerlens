"""Golden eval cases for the FP&A pack, computed from the generator's own tables and the plan frames.

Expected values come from the world before it is written anywhere:

- payments feed totals from the events;
- income statement lines from instalments (billed), payouts (paid, net of returns), handling costs and opex;
- receivable aging from instalments still open on the as-of date;
- driver values (billing, instalments, approved claims, severity, opex) from instalments, claims and opex;
- bridge actual and plan totals per measure, for the latest closed month and fiscal year to date.

The tools reach the same numbers through dbt staging, the ledger postings, the marts and the pipeline, so a
case fails if any step in between drifts.
"""

import polars as pl

from ledgerlens.fpa.config import LINES, FpaConfig
from ledgerlens.fpa.generate import FpaWorld, month_bounds
from ledgerlens.fpa.ledger_adapter import AGING_NAMES
from ledgerlens.fpa.pipeline import fiscal_ytd

SOURCE = "generator:fpa.payment_events+instalments+claims+payouts+opex+plan"


def _case(case_id: str, tool: str, args: dict, expected: list[tuple[str, float]], tol: float = 0.01) -> dict:
    return {
        "id": case_id,
        "pack": "fpa",
        "suite": "offline",
        "expected_source": SOURCE,
        "tool": tool,
        "input": args,
        "expected": [{"label": label, "value": round(value, 6), "abs_tol": tol} for label, value in expected],
    }


def _in_months(column: str, months: list[str]) -> pl.Expr:
    return pl.col(column).dt.strftime("%Y-%m").is_in(months)


def _total(frame: pl.DataFrame, column: str) -> float:
    return float(frame[column].sum())


def measure_totals(
    world: FpaWorld, plan: dict[str, pl.DataFrame], months: list[str]
) -> dict[str, tuple[float, float]]:
    """(actual, plan) per bridge measure over ``months``: premium, net claims paid, LAE and opex."""
    events = world.payment_events.filter(_in_months("event_date", months))
    paid = world.payouts.filter(_in_months("paid_ts", months))
    returned = world.payouts.filter(_in_months("returned_ts", months))
    plan_premium = plan["plan_premium"].filter(pl.col("month").is_in(months))
    plan_claims = plan["plan_claims"].filter(pl.col("month").is_in(months))
    return {
        "collected_premium": (
            _total(events.filter(pl.col("kind") == "premium_received"), "amount"),
            _total(plan_premium, "collected_amount"),
        ),
        "claims_paid": (
            _total(paid, "amount") - _total(returned, "amount"),
            _total(plan_claims, "paid_amount"),
        ),
        "lae": (_total(paid, "handling_cost"), _total(plan_claims, "lae_amount")),
        "opex": (
            _total(world.opex.filter(pl.col("month").is_in(months)), "amount"),
            _total(plan["plan_opex"].filter(pl.col("month").is_in(months)), "amount"),
        ),
    }


def golden_cases(world: FpaWorld, plan: dict[str, pl.DataFrame], config: FpaConfig) -> list[dict]:
    months = sorted(world.months)
    closed = months[:-1]  # the generator leaves the latest period open
    latest = closed[-1]
    picks = [months[len(months) // 2], latest, months[-1]]  # mid-window, latest closed, open
    cases: list[dict] = []

    for month in picks:
        events = world.payment_events.filter(_in_months("event_date", [month]))
        groups = events.group_by("kind", "instrument").agg(
            pl.len().alias("n"), pl.col("amount").sum().alias("amt")
        )
        expected = []
        for kind, instrument, n, amount in groups.sort("kind", "instrument").iter_rows():
            expected += [
                (f"{kind} {instrument} {month} amount", amount),
                (f"{kind} {instrument} {month} events", n),
            ]
        cases.append(_case(f"fpa-payments-{month}", "payments.summary", {"month": month}, expected))

        billed = _total(world.instalments.filter(_in_months("due_date", [month])), "amount")
        totals = measure_totals(world, plan, [month])
        claims, lae, opex = totals["claims_paid"][0], totals["lae"][0], totals["opex"][0]
        cases.append(
            _case(
                f"fpa-income-{month}",
                "ledger.income_statement",
                {"period": month},
                [
                    (f"Premium revenue (billed) {month}", billed),
                    (f"Claims paid {month}", claims),
                    (f"Loss adjustment expense {month}", lae),
                    (f"Operating expense {month}", opex),
                    (f"Operating result {month}", billed - claims - lae - opex),
                ],
            )
        )

    for lob in LINES:
        due = world.instalments.filter(_in_months("due_date", [latest]) & (pl.col("lob") == lob))
        approved = world.claims.filter(_in_months("approved_date", [latest]) & (pl.col("lob") == lob))
        expected = [
            (f"due_premium {lob} {latest}", _total(due, "amount")),
            (f"instalments_due {lob} {latest}", float(due.height)),
            (f"approved_claims {lob} {latest}", float(approved.height)),
            (
                f"severity {lob} {latest}",
                _total(approved, "severity") / approved.height if approved.height else 0.0,
            ),
            (f"opex all {latest}", _total(world.opex.filter(pl.col("month") == latest), "amount")),
        ]
        cases.append(
            _case(f"fpa-drivers-{lob}-{latest}", "fpa.drivers", {"month": latest, "lob": lob}, expected)
        )

    for period, window in ((latest, [latest]), ("ytd", fiscal_ytd(closed, config.fiscal_year_start_month))):
        for measure, (actual, planned) in measure_totals(world, plan, window).items():
            cases.append(
                _case(
                    f"fpa-bridge-{measure}-{period}",
                    "fpa.bridge",
                    {"measure": measure, "period": period},
                    [(f"actual {measure} {period}", actual), (f"plan {measure} {period}", planned)],
                )
            )

    as_of = month_bounds(months[-1])[1]
    open_items = world.instalments.filter(
        (pl.col("due_date") <= as_of) & (pl.col("paid_date").is_null() | (pl.col("paid_date") > as_of))
    ).with_columns((pl.lit(as_of) - pl.col("due_date")).dt.total_days().alias("days"))
    bucket = pl.lit(AGING_NAMES[0])
    for edge, name in zip(config.aging_buckets_days, AGING_NAMES[1:], strict=True):
        bucket = pl.when(pl.col("days") > edge).then(pl.lit(name)).otherwise(bucket)
    aged = (
        open_items.with_columns(bucket.alias("bucket"))
        .group_by("bucket")
        .agg(pl.len().alias("n"), pl.col("amount").sum().alias("amt"))
    )
    by_bucket = {b: (n, amt) for b, n, amt in aged.iter_rows()}
    expected = []
    for name in AGING_NAMES:
        n, amount = by_bucket.get(name, (0, 0.0))
        expected += [(f"receivable {name} amount", amount), (f"receivable {name} items", n)]
    cases.append(_case("fpa-ar-aging", "ledger.ar_aging", {}, expected))
    return cases
