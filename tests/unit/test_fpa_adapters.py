"""Ledger and payments adapters are read-only, and the raw query is fenced (ADR 0001, decision 2)."""

import inspect
from pathlib import Path

import duckdb
import pytest

from ledgerlens.core.db import connect
from ledgerlens.fpa.config import FpaConfig
from ledgerlens.fpa.ledger_adapter import (
    MAX_QUERY_ROWS,
    DuckDBLedger,
    LedgerAdapter,
    QueryRefused,
    aging_bucket,
)
from ledgerlens.fpa.payments_adapter import MAX_EVENTS, DuckDBPaymentsFeed, PaymentsFeed
from ledgerlens.fpa.pipeline import fiscal_ytd

REPO_ROOT = Path(__file__).resolve().parents[2]
WRITE_WORDS = ("write", "insert", "update", "delete", "create", "post", "save", "put")


@pytest.fixture
def ledger_db(tmp_path: Path) -> Path:
    db = tmp_path / "ledger.duckdb"
    with connect(db) as conn:
        conn.execute("create schema fpa")
        conn.execute(
            "create table fpa.fpa_ledger_monthly as select '2026-0' || (i % 9 + 1) as month, "
            "4000 as account, "
            "'Premium' as fullname, i * 1.5 as amount, 'k' || i as row_key from range(1500) t(i)"
        )
        conn.execute("create table fpa.secrets (s varchar)")
        conn.execute(
            "create table fpa.stg_fpa__payment_events as select 'PE-' || i as event_id, "
            "'claim_paid' as kind, "
            "timestamp '2026-08-01' + to_hours(i) as event_ts, (timestamp '2026-08-01' + to_hours(i))::date "
            "as event_date, 10.0 as amount, 'out' as direction, 'check' as instrument, 'auto' as lob, "
            "'C' || i as ref_id "
            "from range(300) t(i)"
        )
    return db


@pytest.mark.parametrize("protocol", [LedgerAdapter, PaymentsFeed, DuckDBLedger, DuckDBPaymentsFeed])
def test_adapters_expose_no_write_methods(protocol):
    names = [n for n, _ in inspect.getmembers(protocol, inspect.isfunction) if not n.startswith("_")]
    assert names and not [n for n in names if set(n.lower().split("_")) & set(WRITE_WORDS)]


@pytest.mark.parametrize(
    "sql",
    [
        "select * from fpa.fpa_ledger_monthly",
        "select month, fpa_ledger_monthly.* from fpa.fpa_ledger_monthly",
        "select s from fpa.secrets",
        "select month from fpa.fpa_ledger_monthly; drop table fpa.secrets",
        "delete from fpa.fpa_ledger_monthly",
        "with x as (select 1) select month from fpa.fpa_ledger_monthly",
        "select month from fpa.fpa_ledger_monthly join fpa.secrets on true",
        "select 1",
        "select month from fpa.fpa_ledger_monthly, fpa.secrets",
        "select month from fpa.fpa_ledger_monthly where amount in (select 1 from fpa.secrets)",
        "select month from read_csv('C:/anything.csv')",
        "select month from fpa.fpa_ledger_monthly union all select s from fpa.secrets",
        "select columns(*) from fpa.fpa_ledger_monthly",
    ],
)
def test_raw_query_refuses_anything_but_one_named_select_over_ledger_tables(ledger_db, sql):
    with pytest.raises(QueryRefused):
        DuckDBLedger(ledger_db).query(sql)


def test_raw_query_is_capped_and_runs_read_only(ledger_db):
    ledger = DuckDBLedger(ledger_db)
    result = ledger.query("select month, amount from fpa.fpa_ledger_monthly")
    assert len(result.rows) == MAX_QUERY_ROWS
    assert result.sql == "select month, amount from fpa.fpa_ledger_monthly"  # lineage cites what ran
    with pytest.raises(duckdb.Error):  # the connection itself refuses writes, whatever slips past the guard
        ledger._rows("insert into fpa.secrets values ('x')")
    with pytest.raises(duckdb.Error):  # and cannot read environment variables or files
        ledger._rows("select getenv('PATH')")
    sql = (
        "select count(*) as n, sum(amount * 2) as twice, round(max(amount), 1) as top "
        "from fpa.fpa_ledger_monthly"
    )
    assert ledger.query(sql).rows[0]["n"] == 1500


@pytest.mark.parametrize(
    "sql",
    [
        "select month, current_setting('temp_directory') as t from fpa.fpa_ledger_monthly",
        "select month, version() as v from fpa.fpa_ledger_monthly",
        "select month, getvariable('x') as v from fpa.fpa_ledger_monthly",
        "select month, current_schemas(true) as s from fpa.fpa_ledger_monthly",
        "select month, my_macro(amount) as m from fpa.fpa_ledger_monthly",
    ],
)
def test_raw_query_refuses_functions_that_describe_the_machine_or_are_not_built_in(ledger_db, sql):
    with pytest.raises(QueryRefused, match="function"):
        DuckDBLedger(ledger_db).query(sql)


def test_a_runaway_query_is_interrupted(ledger_db):
    with pytest.raises(QueryRefused, match="longer than"):
        DuckDBLedger(ledger_db)._rows("select count(*) from range(100000000000)", timeout=0.2)


def test_payments_feed_caps_events_newest_first_and_filters(ledger_db):
    feed = DuckDBPaymentsFeed(ledger_db)
    events = feed.events(limit=10_000).rows
    assert len(events) == MAX_EVENTS
    assert events[0]["event_id"] == "PE-299" and events[0]["event_ts"] > events[-1]["event_ts"]
    assert feed.events(kind="premium_received").rows == []
    summary = feed.summary("2026-08")
    assert sum(g["events"] for g in summary.rows) == 300 and summary.tables == ["fpa.stg_fpa__payment_events"]


def test_committed_config_is_the_documented_default():
    assert FpaConfig.from_yaml(REPO_ROOT / "config" / "fpa.yaml") == FpaConfig()


@pytest.mark.parametrize("edges", [(30, 30, 90), (60, 30, 90), (0, 60, 90)])
def test_aging_edges_must_be_positive_and_increasing(edges):
    with pytest.raises(ValueError, match="increasing"):
        FpaConfig(aging_buckets_days=edges)


@pytest.mark.parametrize(
    ("days", "bucket"),
    [
        (0, "current"),
        (30, "current"),
        (31, "monitor"),
        (60, "monitor"),
        (61, "escalate"),
        (90, "escalate"),
        (91, "stale"),
    ],
)
def test_aging_buckets_follow_adr_0001_edges(days, bucket):
    assert aging_bucket(days, (30, 60, 90)) == bucket


@pytest.mark.parametrize(("start", "expected_first"), [(1, "2026-01"), (4, "2026-04"), (10, "2025-10")])
def test_fiscal_year_to_date_starts_at_the_configured_month(start, expected_first):
    months = [f"{y}-{m:02d}" for y in (2025, 2026) for m in range(1, 13) if f"{y}-{m:02d}" <= "2026-08"]
    window = fiscal_ytd(months, start)
    assert window[0] == expected_first and window[-1] == "2026-08"
