"""The raw ``raw_fpa`` schema that both synthetic worlds write and the dbt sources read.

The warehouse holds one FP&A world at a time, the insurer or the processor (``--variant processor``). One dbt
project serves both: shared tables (the payments feed, the NetSuite-shaped ledger, opex) carry either world,
and each world's own tables are written empty when the other world is loaded, so every model builds and
every test runs, and a model never reads the leftovers of a world generated earlier. ``raw_fpa.world_meta``
records which world is loaded. Column types are declared here, and a frame that drifts from them is refused.

Writing a world also drops ``fpa.run_meta``, in the same transaction: every FP&A tool reads it, so until
``run_fpa`` has run on the new world the tools refuse to answer, instead of labelling one world's numbers with
the other's scope (the staging views read ``raw_fpa`` directly; the marts and run tables do not).
"""

from pathlib import Path
from typing import Any

import polars as pl

from ledgerlens.core.db import NO_EXTERNAL_ACCESS, connect
from ledgerlens.fpa.config import Variant

S, F, D, B, I64 = pl.String, pl.Float64, pl.Date, pl.Boolean, pl.Int64
TS = pl.Datetime("us")

RAW_SCHEMAS: dict[str, dict[str, Any]] = {
    # Shared by both worlds.
    "payment_events": {
        "event_id": S, "kind": S, "event_ts": TS, "event_date": D, "amount": F,
        "direction": S,  # credit/debit indicator from the account holder's side: "in" or "out"
        "instrument": S, "lob": S, "policy_id": S, "claim_id": S, "merchant_id": S, "ref_id": S,
    },
    "opex": {"month": S, "amount": F},
    "transaction": {
        "id": I64, "tranid": S, "trandate": D, "recordtype": S, "entity": S, "subsidiary": pl.Int32,
        "postingperiod": I64, "posting": B, "approvalstatus": S, "currency": S, "exchangerate": F,
        "amount": F, "foreignamount": F, "memo": S, "custbody_event_id": S,
    },
    "transactionline": {
        "id": I64, "transaction": I64, "account": pl.Int32, "amount": F, "foreignamount": F,
        "department": S, "class": S, "location": S,
    },
    "account": {"id": I64, "acctnumber": S, "fullname": S, "accttype": S, "parent": I64},
    "accountingperiod": {"id": I64, "periodname": S, "startdate": D, "enddate": D, "closed": B},
    "subsidiary": {"id": I64, "name": S, "currency": S, "parent": I64},
    "plan_opex": {"month": S, "amount": F},
    # The insurer.
    "policies": {
        "policy_id": S, "lob": S, "pay_plan": S, "method": S, "inception_date": D, "annual_premium": F,
    },
    "instalments": {
        "instalment_id": S, "policy_id": S, "lob": S, "method": S, "due_date": D, "amount": F, "paid_date": D,
    },
    "claims": {
        "claim_id": S, "policy_id": S, "lob": S, "channel": S, "occurred_date": D, "approved_date": D,
        "severity": F,
    },
    "payouts": {
        "payout_id": S, "claim_id": S, "lob": S, "leg": S, "channel": S, "amount": F, "handling_cost": F,
        "approved_date": D, "paid_ts": TS, "returned_ts": TS,
    },
    "plan_premium": {
        "month": S, "lob": S, "method": S, "instalments_due": F, "due_amount": F, "collected_amount": F,
    },
    "plan_claims": {
        "month": S, "lob": S, "channel": S, "approved_count": F, "approved_amount": F, "payouts": F,
        "paid_amount": F, "lae_amount": F,
    },
    # The processor.
    "merchants": {"merchant_id": S, "lob": S, "onboard_date": D, "churn_date": D, "attempts_per_day": F},
    "captures": {
        "batch_id": S, "capture_date": D, "settle_date": D, "lob": S, "card_type": S, "initiation": S,
        "attempts": I64, "approvals": I64, "volume": F, "fee_amount": F, "interchange_amount": F,
    },
    "chargebacks": {
        "chargeback_id": S, "merchant_id": S, "lob": S, "card_type": S, "initiation": S, "purchase_date": D,
        "raised_date": D, "amount": F, "recovered_date": D, "written_off_date": D,
    },
    "plan_processing": {
        "month": S, "lob": S, "card_type": S, "initiation": S, "attempts": F, "transactions": F,
        "payment_volume": F, "fees_captured": F, "interchange_captured": F, "fees_settled": F,
        "interchange_settled": F,
    },
    "plan_chargebacks": {
        "month": S, "lob": S, "initiation": S, "chargebacks": F, "disputed_amount": F, "net_amount": F,
        "debited_amount": F, "recovered_amount": F,
    },
    "plan_merchants": {"month": S, "lob": S, "active_merchants": F, "new_merchants": F},
}  # fmt: skip
SHARED_TABLES = (
    "payment_events",
    "opex",
    "transaction",
    "transactionline",
    "account",
    "accountingperiod",
    "subsidiary",
    "plan_opex",
)
VARIANT_TABLES: dict[Variant, tuple[str, ...]] = {
    "insurer": (
        *SHARED_TABLES,
        "policies",
        "instalments",
        "claims",
        "payouts",
        "plan_premium",
        "plan_claims",
    ),
    "processor": (
        *SHARED_TABLES,
        "merchants",
        "captures",
        "chargebacks",
        "plan_processing",
        "plan_chargebacks",
        "plan_merchants",
    ),
}


class RawSchemaError(ValueError):
    pass


def raw_tables(variant: Variant, frames: dict[str, pl.DataFrame]) -> dict[str, pl.DataFrame]:
    """Every raw table: this world's frames in declared column order, the other world's tables empty."""
    expected = set(VARIANT_TABLES[variant])
    if set(frames) != expected:
        raise RawSchemaError(
            f"the {variant} world writes {sorted(expected)}; got {sorted(frames)} "
            f"(missing {sorted(expected - set(frames))}, unexpected {sorted(set(frames) - expected)})"
        )
    out = {}
    for name, schema in RAW_SCHEMAS.items():
        if name not in frames:
            out[name] = pl.DataFrame(schema=schema)
            continue
        got = dict(frames[name].schema)
        if got != schema:
            drift = sorted(c for c in set(got) | set(schema) if got.get(c) != schema.get(c))
            raise RawSchemaError(f"raw_fpa.{name}: columns {drift} differ from the declared schema")
        out[name] = frames[name].select(list(schema))
    return out


def write_raw(
    warehouse: Path, variant: Variant, frames: dict[str, pl.DataFrame], meta: dict[str, float | str]
) -> dict[str, pl.DataFrame]:
    """Write one world to ``raw_fpa`` (replacing any world there before) and return what was written."""
    tables = raw_tables(variant, frames) | {"world_meta": pl.DataFrame([{**meta, "variant": variant}])}
    with connect(warehouse) as conn:
        conn.execute("begin transaction")
        try:
            conn.execute("create schema if not exists raw_fpa")
            conn.execute("create schema if not exists fpa")
            conn.execute("drop table if exists fpa.run_meta")
            for name, frame in tables.items():
                conn.register("_raw_frame", frame)  # an explicit name, never a lookup of a Python variable
                conn.execute(f'create or replace table raw_fpa."{name}" as select * from _raw_frame')  # noqa: S608
                conn.unregister("_raw_frame")
        except Exception:
            conn.execute("rollback")
            raise
        conn.execute("commit")
    return tables


def world_meta(warehouse: Path) -> dict[str, Any]:
    """What ``raw_fpa.world_meta`` says about the loaded world, including its ``variant``."""
    with connect(warehouse, read_only=True, config=NO_EXTERNAL_ACCESS) as conn:
        cursor = conn.execute("select * from raw_fpa.world_meta")
        row = cursor.fetchone()
        columns = [d[0] for d in cursor.description or []]
    if row is None or "variant" not in columns:
        raise RuntimeError("raw_fpa.world_meta names no FP&A world; run `ledgerlens generate --pack fpa`")
    return dict(zip(columns, row, strict=True))
