"""The processor's driver tree, read from the dbt marts so every driver cites its mart rows.

- transactions       = attempts x authorization_rate
- payment_volume     = transactions x average_ticket
- fee_revenue        = payment_volume x take_rate + fee_timing
- interchange        = sum over card types of payment_volume x card_mix x interchange_rate
                       + interchange_timing
- chargebacks_raised = transactions x chargeback_rate
- chargebacks        = chargebacks_raised x average_chargeback x (1 - recovery_rate) + chargeback_timing
- net_operating_cash = fee_revenue - interchange - chargebacks - opex

fee_revenue, interchange and chargebacks are cash: fees and interchange as settled, chargebacks as the
networks debited them less what merchants repaid. The timing terms are cash less the event-basis amount
(fees and interchange on purchases captured that month; the net still owed on chargebacks raised that month),
so each identity holds exactly, and a test checks the tree's net cash against the ledger's. The chargeback
rate counts chargebacks raised against purchases approved in the same month, as the card networks count it.
``net_cash_from_drivers`` in ``fpa/drivers.py`` checks the identity with this world's measure signs.

The event basis of chargebacks is what merchants have not repaid as of the run, so in the open month a
recovery still in flight counts as unpaid until it lands; closed months have had it land.
"""

from dataclasses import dataclass
from pathlib import Path

import polars as pl

from ledgerlens.core.db import NO_EXTERNAL_ACCESS, connect
from ledgerlens.fpa.drivers import DRIVER_SCHEMA, cash_rows, driver_row

PROCESSING = "fpa_processing_monthly"
CHARGEBACKS = "fpa_chargebacks_monthly"
MERCHANTS = "fpa_merchants_monthly"


@dataclass(frozen=True)
class ProcessorMarts:
    processing: pl.DataFrame
    chargebacks: pl.DataFrame
    merchants: pl.DataFrame
    cash: pl.DataFrame
    plan_processing: pl.DataFrame
    plan_chargebacks: pl.DataFrame
    plan_merchants: pl.DataFrame
    plan_opex: pl.DataFrame
    periods: pl.DataFrame


def load_marts(db: Path) -> ProcessorMarts:
    with connect(db, read_only=True, config=NO_EXTERNAL_ACCESS) as conn:
        frames = {
            "processing": f"select * from fpa.{PROCESSING}",  # noqa: S608 (module constants)
            "chargebacks": f"select * from fpa.{CHARGEBACKS}",  # noqa: S608
            "merchants": f"select * from fpa.{MERCHANTS}",  # noqa: S608
            "cash": "select * from fpa.fpa_cash_monthly",
            "plan_processing": "select * from fpa.stg_fpa__plan_processing",
            "plan_chargebacks": "select * from fpa.stg_fpa__plan_chargebacks",
            "plan_merchants": "select * from fpa.stg_fpa__plan_merchants",
            "plan_opex": "select * from fpa.stg_fpa__plan_opex",
            "periods": "select id, periodname, strftime(startdate, '%Y-%m') as month, closed "
            "from fpa.stg_fpa__accountingperiod order by startdate",
        }
        return ProcessorMarts(**{name: conn.execute(sql).pl() for name, sql in frames.items()})


def _ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def _processing_rows(month: str, lob: str, g: pl.DataFrame) -> list[dict]:
    keys = sorted(g["row_key"].to_list())
    attempts, transactions = float(g["attempts"].sum()), float(g["transactions"].sum())
    volume = float(g["payment_volume"].sum())
    fees, fees_settled = float(g["fees_captured"].sum()), float(g["fees_settled"].sum())
    ic, ic_settled = float(g["interchange_captured"].sum()), float(g["interchange_settled"].sum())

    def row(driver: str, segment: str, value: float, unit: str, row_keys: list[str]) -> dict:
        return driver_row(month, lob, driver, segment, value, unit, PROCESSING, row_keys)

    rows = [
        row("attempts", "", attempts, "count", keys),
        row("transactions", "", transactions, "count", keys),
        row("authorization_rate", "", _ratio(transactions, attempts), "share", keys),
        row("payment_volume", "", volume, "USD", keys),
        row("average_ticket", "", _ratio(volume, transactions), "USD", keys),
        row("take_rate", "", _ratio(fees, volume), "share", keys),
        row("fee_revenue", "", fees_settled, "USD", keys),
        row("fee_timing", "", fees_settled - fees, "USD", keys),
        row("interchange", "", ic_settled, "USD", keys),
        row("interchange_timing", "", ic_settled - ic, "USD", keys),
    ]
    by_initiation = g.group_by("initiation").agg(
        pl.col("attempts").sum(), pl.col("transactions").sum(), pl.col("row_key").sort()
    )
    for who, a, t, k in by_initiation.sort("initiation").iter_rows():
        rows.append(row("authorization_rate", who, _ratio(float(t), float(a)), "share", k))
        if who == "agent":
            rows.append(row("agent_share", "", _ratio(float(a), attempts), "share", keys))
    by_card = g.group_by("card_type").agg(
        pl.col("payment_volume").sum(), pl.col("interchange_captured").sum(), pl.col("row_key").sort()
    )
    for card, v, i, k in by_card.sort("card_type").iter_rows():
        rows += [
            row("card_mix", card, _ratio(float(v), volume), "share", k),
            row("interchange_rate", card, _ratio(float(i), float(v)), "share", k),
        ]
    return rows


def _chargeback_rows(month: str, lob: str, g: pl.DataFrame) -> list[dict]:
    keys = sorted(g["row_key"].to_list())
    transactions, raised = float(g["transactions"].sum()), float(g["chargebacks"].sum())
    disputed, net = float(g["disputed_amount"].sum()), float(g["net_amount"].sum())
    cash = float(g["debited_amount"].sum()) - float(g["recovered_amount"].sum())

    def row(driver: str, segment: str, value: float, unit: str, row_keys: list[str]) -> dict:
        return driver_row(month, lob, driver, segment, value, unit, CHARGEBACKS, row_keys)

    rows = [
        row("chargebacks_raised", "", raised, "count", keys),
        row("chargeback_rate", "", _ratio(raised, transactions), "share", keys),
        row("average_chargeback", "", _ratio(disputed, raised), "USD", keys),
        row("recovery_rate", "", 1 - _ratio(net, disputed) if disputed else 0.0, "share", keys),
        row("chargebacks", "", cash, "USD", keys),
        row("chargeback_timing", "", cash - net, "USD", keys),
    ]
    for r in g.sort("initiation").iter_rows(named=True):
        rate = _ratio(float(r["chargebacks"]), float(r["transactions"]))
        rows.append(row("chargeback_rate", r["initiation"], rate, "share", [r["row_key"]]))
    return rows


def driver_table(marts: ProcessorMarts) -> pl.DataFrame:
    rows: list[dict] = []
    for (month, lob), g in marts.processing.group_by(["month", "lob"], maintain_order=True):
        rows += _processing_rows(str(month), str(lob), g)
    for (month, lob), g in marts.chargebacks.group_by(["month", "lob"], maintain_order=True):
        rows += _chargeback_rows(str(month), str(lob), g)
    for r in marts.merchants.iter_rows(named=True):
        keys = [r["row_key"]]
        rows += [
            driver_row(
                r["month"], r["lob"], "merchants", "", r["active_merchants"], "count", MERCHANTS, keys
            ),
            driver_row(
                r["month"], r["lob"], "new_merchants", "", r["new_merchants"], "count", MERCHANTS, keys
            ),
            driver_row(
                r["month"],
                r["lob"],
                "churned_merchants",
                "",
                r["churned_merchants"],
                "count",
                MERCHANTS,
                keys,
            ),
        ]
    rows += cash_rows(marts.cash)
    return pl.DataFrame(rows, schema=DRIVER_SCHEMA).sort("month", "lob", "driver", "segment")
