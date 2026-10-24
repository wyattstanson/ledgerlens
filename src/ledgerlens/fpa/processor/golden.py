"""Golden eval cases for the processor world, computed from the generator's own tables and the plan frames.

Expected values come from the world before it is written anywhere:

- payments feed totals from the events;
- income statement lines from captures (fees and interchange as captured), write-offs and opex;
- receivable aging from batches not yet settled and chargebacks still owed on the as-of date;
- driver values (attempts, approved transactions, volume, merchants, chargebacks raised, opex) from captures,
  merchants, chargebacks and opex;
- bridge actual and plan totals per measure, for the latest closed month and fiscal year to date.

The tools reach the same numbers through dbt staging, the ledger postings, the marts and the pipeline, so a
case fails if any step in between drifts. The cases live apart from the insurer's
(``tests/evals/golden/fpa_processor/``) because the warehouse holds one world at a time.
"""

from datetime import date

import polars as pl

from ledgerlens.fpa.config import VERTICALS, FpaConfig
from ledgerlens.fpa.gl import month_bounds
from ledgerlens.fpa.ledger_adapter import AGING_NAMES, aging_bucket
from ledgerlens.fpa.pipeline import fiscal_ytd
from ledgerlens.fpa.processor.generate import ProcessorWorld, batch_totals

SOURCE = "generator:fpa.processor captures+chargebacks+merchants+payment_events+opex+plan"


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
    world: ProcessorWorld, plan: dict[str, pl.DataFrame], months: list[str]
) -> dict[str, tuple[float, float]]:
    """(actual, plan) per bridge measure over ``months``: cash fees, interchange, net chargebacks, opex."""
    settled = world.captures.filter(_in_months("settle_date", months))
    raised = world.chargebacks.filter(_in_months("raised_date", months))
    recovered = world.chargebacks.filter(_in_months("recovered_date", months))
    plan_proc = plan["plan_processing"].filter(pl.col("month").is_in(months))
    plan_cb = plan["plan_chargebacks"].filter(pl.col("month").is_in(months))
    return {
        "fee_revenue": (_total(settled, "fee_amount"), _total(plan_proc, "fees_settled")),
        "interchange": (_total(settled, "interchange_amount"), _total(plan_proc, "interchange_settled")),
        "chargebacks": (
            _total(raised, "amount") - _total(recovered, "amount"),
            _total(plan_cb, "debited_amount") - _total(plan_cb, "recovered_amount"),
        ),
        "opex": (
            _total(world.opex.filter(pl.col("month").is_in(months)), "amount"),
            _total(plan["plan_opex"].filter(pl.col("month").is_in(months)), "amount"),
        ),
    }


def _active_merchants(world: ProcessorWorld, lob: str, day: date) -> int:
    return world.merchants.filter(
        (pl.col("lob") == lob)
        & (pl.col("onboard_date") <= day)
        & (pl.col("churn_date").is_null() | (pl.col("churn_date") > day))
    ).height


def golden_cases(world: ProcessorWorld, plan: dict[str, pl.DataFrame], config: FpaConfig) -> list[dict]:
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
        cases.append(_case(f"fpa-processor-payments-{month}", "payments.summary", {"month": month}, expected))

        captured = world.captures.filter(_in_months("capture_date", [month]))
        fees, interchange = _total(captured, "fee_amount"), _total(captured, "interchange_amount")
        losses = _total(world.chargebacks.filter(_in_months("written_off_date", [month])), "amount")
        opex = _total(world.opex.filter(pl.col("month") == month), "amount")
        cases.append(
            _case(
                f"fpa-processor-income-{month}",
                "ledger.income_statement",
                {"period": month},
                [
                    (f"Processing fee revenue (captured) {month}", fees),
                    (f"Interchange and network fees {month}", interchange),
                    (f"Chargeback losses (written off) {month}", losses),
                    (f"Operating expense {month}", opex),
                    (f"Operating result {month}", fees - interchange - losses - opex),
                ],
            )
        )

    month_end = month_bounds(latest)[1]
    for lob in VERTICALS:
        captured = world.captures.filter(_in_months("capture_date", [latest]) & (pl.col("lob") == lob))
        raised = world.chargebacks.filter(_in_months("raised_date", [latest]) & (pl.col("lob") == lob))
        expected = [
            (f"attempts {lob} {latest}", _total(captured, "attempts")),
            (f"transactions {lob} {latest}", _total(captured, "approvals")),
            (f"payment_volume {lob} {latest}", _total(captured, "volume")),
            (f"merchants {lob} {latest}", float(_active_merchants(world, lob, month_end))),
            (f"chargebacks_raised {lob} {latest}", float(raised.height)),
            (f"opex all {latest}", _total(world.opex.filter(pl.col("month") == latest), "amount")),
        ]
        cases.append(
            _case(
                f"fpa-processor-drivers-{lob}-{latest}",
                "fpa.drivers",
                {"month": latest, "lob": lob},
                expected,
            )
        )

    for period, window in ((latest, [latest]), ("ytd", fiscal_ytd(closed, config.fiscal_year_start_month))):
        for measure, (actual, planned) in measure_totals(world, plan, window).items():
            cases.append(
                _case(
                    f"fpa-processor-bridge-{measure}-{period}",
                    "fpa.bridge",
                    {"measure": measure, "period": period},
                    [(f"actual {measure} {period}", actual), (f"plan {measure} {period}", planned)],
                )
            )

    as_of = month_bounds(months[-1])[1]
    unsettled = batch_totals(world.captures).filter(pl.col("settle_date") > as_of)
    owed = world.chargebacks.filter(
        (pl.col("raised_date") <= as_of)
        & pl.col("recovered_date").is_null()
        & pl.col("written_off_date").is_null()
    )
    items = [((as_of - d).days, float(a)) for d, a in unsettled.select("settle_date", "net").iter_rows()] + [
        ((as_of - d).days, float(a)) for d, a in owed.select("raised_date", "amount").iter_rows()
    ]
    by_bucket = {name: [0, 0.0] for name in AGING_NAMES}
    for days, amount in items:
        bucket = by_bucket[aging_bucket(days, config.aging_buckets_days)]
        bucket[0] += 1
        bucket[1] += amount
    expected = []
    for name in AGING_NAMES:
        n, amount = by_bucket[name]
        expected += [(f"receivable {name} amount", amount), (f"receivable {name} items", n)]
    cases.append(_case("fpa-processor-ar-aging", "ledger.ar_aging", {}, expected))
    return cases
