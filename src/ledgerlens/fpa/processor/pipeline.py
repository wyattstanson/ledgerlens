"""The processor's side of the FP&A job: its measures, variance bridges, driver tree and scenario.

The bridge machinery is the insurer's (``fpa/bridge.py``), on the processor's own event and cash bases:

- fee revenue: segments vertical x card type; quantity payment volume, amount fees on purchases captured;
  cash, fees as settled. Volume is volume x the planned take rate, mix the shift between segments priced
  differently, rate a change in what a segment earns per dollar, timing settlement across month ends.
- interchange: the same segments; amount interchange and network fees; cash, as settled.
- chargebacks: segments vertical x initiation; quantity chargebacks raised, amount the net merchants have not
  repaid; cash, debited by the networks less recovered from merchants. Timing is recoveries landing in a later
  month than the dispute.
- opex, and net operating cash = fee revenue - interchange - chargebacks - opex.
"""

from datetime import date
from pathlib import Path

import polars as pl

from ledgerlens.core.db import NO_EXTERNAL_ACCESS, connect
from ledgerlens.fpa.bridge import Bridge, Segment, combine, variance_bridge
from ledgerlens.fpa.config import FpaConfig
from ledgerlens.fpa.materiality import Kind
from ledgerlens.fpa.processor.drivers import ProcessorMarts, driver_table, load_marts
from ledgerlens.fpa.processor.inputs import recovery_window_days
from ledgerlens.fpa.processor.scenario import agent_tokens
from ledgerlens.fpa.scenario import ScenarioResult
from ledgerlens.fpa.world import WorldView, segments

MEASURES: dict[str, tuple[Kind, int]] = {  # measure -> (kind, sign in net operating cash)
    "fee_revenue": ("revenue", 1),
    "interchange": ("expense", -1),
    "chargebacks": ("expense", -1),
    "opex": ("expense", -1),
}
NET = "net_operating_cash"
ENTITY = "LedgerLens Payments (synthetic), all verticals"
BOOK = "primary (cash basis: fees and interchange as settled, chargebacks net of merchant recoveries)"
INCOME_BASIS = (
    "processing fees and interchange as captured; chargeback losses as written off; operating expense as paid"
)


def _sum(frame: pl.DataFrame, column: str) -> float:
    return float(frame[column].sum())


def bridges_for(marts: ProcessorMarts, months: list[str]) -> dict[str, Bridge]:
    """The four measure bridges and net operating cash for a set of months (one month, or year to date)."""
    pick = pl.col("month").is_in(months)
    proc, plan_proc = marts.processing.filter(pick), marts.plan_processing.filter(pick)
    cb, plan_cb = marts.chargebacks.filter(pick), marts.plan_chargebacks.filter(pick)
    opex = float(marts.cash.filter(pick)["opex"].fill_null(0).sum())
    plan_opex = _sum(marts.plan_opex.filter(pick), "amount")
    by_card = ["lob", "card_type"]
    volume = ("payment_volume", "payment_volume")
    out = {
        "fee_revenue": variance_bridge(
            segments(proc, plan_proc, by_card, volume, ("fees_captured", "fees_captured")),
            _sum(proc, "fees_settled"),
            _sum(plan_proc, "fees_settled"),
        ),
        "interchange": variance_bridge(
            segments(proc, plan_proc, by_card, volume, ("interchange_captured", "interchange_captured")),
            _sum(proc, "interchange_settled"),
            _sum(plan_proc, "interchange_settled"),
        ),
        "chargebacks": variance_bridge(
            segments(
                cb,
                plan_cb,
                ["lob", "initiation"],
                ("chargebacks", "chargebacks"),
                ("net_amount", "net_amount"),
            ),
            _sum(cb, "debited_amount") - _sum(cb, "recovered_amount"),
            _sum(plan_cb, "debited_amount") - _sum(plan_cb, "recovered_amount"),
        ),
        "opex": variance_bridge([Segment("opex", 1.0, opex, 1.0, plan_opex)], opex, plan_opex),
    }
    out[NET] = combine(out, {k: sign for k, (_, sign) in MEASURES.items()})
    return out


def processor_view(warehouse: Path, config: FpaConfig) -> WorldView:
    marts = load_marts(warehouse)

    def scenario(as_of: date) -> ScenarioResult:
        with connect(warehouse, read_only=True, config=NO_EXTERNAL_ACCESS) as conn:
            captures = conn.execute("select * from fpa.stg_fpa__captures").pl()
            chargebacks = conn.execute("select * from fpa.stg_fpa__chargebacks").pl()
        share, world = config.processor.scenario_token_share, config.processor.world
        return agent_tokens(captures, chargebacks, as_of, share, recovery_window_days(world))

    return WorldView(
        variant="processor",
        entity=ENTITY,
        book=BOOK,
        income_basis=INCOME_BASIS,
        measures=MEASURES,
        drivers=driver_table(marts),
        periods=marts.periods,
        bridges_for=lambda months: bridges_for(marts, months),
        scenario=scenario,
    )
