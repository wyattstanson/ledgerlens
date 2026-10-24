"""The driver tree (BUILD_PROMPT 10.3), read from the dbt marts so every driver cites its mart rows.

- collected_premium = sum over methods of due_premium x method_mix x on_time_rate + late_collections
- claims_paid       = cash out for approved claims (approved_claims x severity x channel_share, released by
                      each channel's payout lag), net of returned payouts
- lae               = sum over channels of payouts x handling_cost
- net_operating_cash = collected_premium - claims_paid - lae - opex

``late_collections`` is what the tree's on-time term does not explain: payments for earlier months' dues,
early payments of next month's, minus dues that went unpaid. The identities hold exactly by construction,
and a test checks the tree's net cash against the ledger's.
"""

from dataclasses import dataclass
from pathlib import Path

import polars as pl

from ledgerlens.core.db import NO_EXTERNAL_ACCESS, connect

DRIVER_SCHEMA = {
    "month": pl.String,
    "lob": pl.String,
    "driver": pl.String,
    "segment": pl.String,
    "value": pl.Float64,
    "unit": pl.String,
    "source_table": pl.String,
    "row_keys": pl.List(pl.String),
}


@dataclass(frozen=True)
class Marts:
    premium: pl.DataFrame
    claims: pl.DataFrame
    cash: pl.DataFrame
    plan_premium: pl.DataFrame
    plan_claims: pl.DataFrame
    plan_opex: pl.DataFrame
    periods: pl.DataFrame


def load_marts(db: Path) -> Marts:
    with connect(db, read_only=True, config=NO_EXTERNAL_ACCESS) as conn:
        frames = {
            "premium": "select * from fpa.fpa_premium_monthly",
            "claims": "select * from fpa.fpa_claims_monthly",
            "cash": "select * from fpa.fpa_cash_monthly",
            "plan_premium": "select * from fpa.stg_fpa__plan_premium",
            "plan_claims": "select * from fpa.stg_fpa__plan_claims",
            "plan_opex": "select * from fpa.stg_fpa__plan_opex",
            "periods": "select id, periodname, strftime(startdate, '%Y-%m') as month, closed "
            "from fpa.stg_fpa__accountingperiod order by startdate",
        }
        return Marts(**{name: conn.execute(sql).pl() for name, sql in frames.items()})


def driver_row(
    month: str, lob: str, driver: str, segment: str, value: float, unit: str, table: str, keys: list[str]
) -> dict:
    return {
        "month": month,
        "lob": lob,
        "driver": driver,
        "segment": segment,
        "value": float(value),
        "unit": unit,
        "source_table": f"fpa.{table}",
        "row_keys": keys,
    }


def driver_table(marts: Marts) -> pl.DataFrame:
    rows: list[dict] = []
    for (month, lob), g in marts.premium.group_by(["month", "lob"], maintain_order=True):
        keys = sorted(g["row_key"].to_list())
        due = float(g["due_amount"].sum())
        collected = float(g["collected_amount"].sum())
        on_time_part = 0.0
        rows += [
            driver_row(month, lob, "due_premium", "", due, "USD", "fpa_premium_monthly", keys),
            driver_row(
                month,
                lob,
                "instalments_due",
                "",
                float(g["instalments_due"].sum()),
                "count",
                "fpa_premium_monthly",
                keys,
            ),
            driver_row(month, lob, "collected_premium", "", collected, "USD", "fpa_premium_monthly", keys),
        ]
        for r in g.iter_rows(named=True):
            mix = r["due_amount"] / due if due else 0.0
            on_time = r["paid_on_time"] / r["instalments_due"] if r["instalments_due"] else 0.0
            on_time_part += due * mix * on_time
            rows += [
                driver_row(
                    month, lob, "method_mix", r["method"], mix, "share", "fpa_premium_monthly", [r["row_key"]]
                ),
                driver_row(
                    month,
                    lob,
                    "on_time_rate",
                    r["method"],
                    on_time,
                    "share",
                    "fpa_premium_monthly",
                    [r["row_key"]],
                ),
            ]
        rows.append(
            driver_row(
                month,
                lob,
                "late_collections",
                "",
                collected - on_time_part,
                "USD",
                "fpa_premium_monthly",
                keys,
            )
        )
    for (month, lob), g in marts.claims.group_by(["month", "lob"], maintain_order=True):
        keys = sorted(g["row_key"].to_list())
        count = float(g["approved_count"].sum())
        amount = float(g["approved_amount"].sum())
        payouts = float(g["payouts"].sum())
        lae = float(g["lae_amount"].sum())
        net_paid = float(g["paid_amount"].sum()) - float(g["returned_amount"].sum())
        rows += [
            driver_row(month, lob, "approved_claims", "", count, "count", "fpa_claims_monthly", keys),
            driver_row(
                month,
                lob,
                "severity",
                "",
                amount / count if count else 0.0,
                "USD",
                "fpa_claims_monthly",
                keys,
            ),
            driver_row(month, lob, "claims_paid", "", net_paid, "USD", "fpa_claims_monthly", keys),
            driver_row(month, lob, "payouts", "", payouts, "count", "fpa_claims_monthly", keys),
            driver_row(month, lob, "lae", "", lae, "USD", "fpa_claims_monthly", keys),
        ]
        for r in g.iter_rows(named=True):
            k = [r["row_key"]]
            rows.append(
                driver_row(
                    month,
                    lob,
                    "channel_share",
                    r["channel"],
                    r["approved_count"] / count if count else 0.0,
                    "share",
                    "fpa_claims_monthly",
                    k,
                )
            )
            if r["payouts"]:
                rows += [
                    driver_row(
                        month,
                        lob,
                        "handling_cost",
                        r["channel"],
                        r["lae_amount"] / r["payouts"],
                        "USD",
                        "fpa_claims_monthly",
                        k,
                    ),
                    driver_row(
                        month,
                        lob,
                        "days_to_pay",
                        r["channel"],
                        r["days_to_pay"],
                        "days",
                        "fpa_claims_monthly",
                        k,
                    ),
                ]
    rows += cash_rows(marts.cash)
    return pl.DataFrame(rows, schema=DRIVER_SCHEMA).sort("month", "lob", "driver", "segment")


def cash_rows(cash: pl.DataFrame) -> list[dict]:
    """Opex and net operating cash per month from the ledger's cash mart, for every world."""
    rows: list[dict] = []
    for r in cash.iter_rows(named=True):
        keys = [r["month"]]
        rows += [
            driver_row(r["month"], "all", "opex", "", r["opex"] or 0.0, "USD", "fpa_cash_monthly", keys),
            driver_row(
                r["month"],
                "all",
                "net_operating_cash",
                "",
                r["net_operating_cash"],
                "USD",
                "fpa_cash_monthly",
                keys,
            ),
        ]
    return rows


INSURER_SIGNS = {"collected_premium": 1, "claims_paid": -1, "lae": -1, "opex": -1}


def net_cash_from_drivers(drivers: pl.DataFrame, signs: dict[str, int] = INSURER_SIGNS) -> pl.DataFrame:
    """Recompute net operating cash per month through the tree, from the driver table alone; ``signs`` says
    how the world's measures combine (the insurer's by default)."""
    wide = drivers.filter(pl.col("segment") == "").group_by("month", "driver").agg(pl.col("value").sum())
    pivot = wide.pivot(on="driver", index="month", values="value").fill_null(0.0)
    tree = pl.sum_horizontal([sign * pl.col(measure) for measure, sign in signs.items()])
    return pivot.select("month", tree.alias("tree_net_cash")).sort("month")
