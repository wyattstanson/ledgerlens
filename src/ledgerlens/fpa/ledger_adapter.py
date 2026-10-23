"""Read-only ledger adapter in the shape of the NetSuite AI connector rules (ADR 0001, decision 2).

Tool order mirrors the connector: report-shaped methods first (income statement, budget versus actual,
receivables aging), named saved searches next, and a raw query last. Every method returns ``Rows``: the data
plus the query and tables behind it, so lineage cites what actually ran, whichever adapter is plugged in.

The raw query is not an agent tool (ADR 0005). It is checked on DuckDB's own parse tree, not with
patterns: one SELECT, no ``*``, no table functions, only built-in functions that describe data (not
settings, secrets or sequences), and every table it reads must be a ledger table, comma joins and subqueries
included. It returns at most 1,000 rows and is interrupted after ten seconds. Every connection is read-only
with external access off, so nothing here can read files or environment variables. The protocol has no write
methods, and connecting a real NetSuite account is a decision for the owner (BUILD_PROMPT section 17), not
something an agent can do.
"""

import json
import threading
from collections.abc import Iterator
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any, Literal, Protocol

import duckdb

from ledgerlens.core.db import NO_EXTERNAL_ACCESS, connect, fetch_dicts
from ledgerlens.fpa.source import Rows

SavedSearch = Literal["premium_collected_by_method", "claims_paid_by_channel", "cash_by_month"]
MAX_QUERY_ROWS = 1000
QUERY_TIMEOUT_SECONDS = 10.0
LEDGER_TABLES = frozenset(
    {
        "fpa.fpa_ledger_monthly",
        "fpa.stg_fpa__transaction",
        "fpa.stg_fpa__transactionline",
        "fpa.stg_fpa__account",
        "fpa.stg_fpa__accountingperiod",
    }
)
# Built-in functions that describe the machine, not the ledger: paths, settings, secrets, sequences.
REFUSED_FUNCTIONS = frozenset(
    {"getvariable", "version", "which_secret", "nextval", "currval", "user", "session_user", "platform"}
)
REFUSED_PREFIXES = ("current_", "duckdb_", "pragma_")
AGING_NAMES = ("current", "monitor", "escalate", "stale")
REVENUE = 4000  # premium revenue; every other income statement line is an expense
ACCOUNT_LINES = (  # (account, label, sign: income is credit-positive)
    (REVENUE, "Premium revenue (billed)", -1),
    (5000, "Claims paid", 1),
    (5100, "Loss adjustment expense", 1),
    (6000, "Operating expense", 1),
)

INCOME_SQL = """
select account, fullname, amount, row_key
from fpa.fpa_ledger_monthly
where month = ? and account in (4000, 5000, 5100, 6000)
"""
BVA_SQL = """
select row_key, measure, actual_cents, plan_cents, variance_cents, pct_of_plan, status, favourable, kind,
       level, preliminary, confidence, confidence_reason
from fpa.bridge_totals
where period = ?
order by measure
"""
AGING_SQL = """
select instalment_id, amount, days_past_due
from fpa.fpa_ar_aging
"""


@dataclass(frozen=True)
class SavedSearchSpec:
    source: str  # the mart whose rows each total sums (row_keys cite them)
    sql: str
    units: dict[str, str]


SAVED_SEARCHES: dict[str, SavedSearchSpec] = {
    "premium_collected_by_method": SavedSearchSpec(
        "fpa.fpa_premium_monthly",
        "select month, method, sum(collected_amount) as collected_amount, "
        "list(row_key order by row_key) as row_keys "
        "from fpa.fpa_premium_monthly group by all order by month, method",
        {"collected_amount": "USD"},
    ),
    "claims_paid_by_channel": SavedSearchSpec(
        "fpa.fpa_claims_monthly",
        "select month, channel, sum(coalesce(paid_amount, 0) - coalesce(returned_amount, 0)) as claims_paid, "
        "sum(payouts) as payouts, list(row_key order by row_key) as row_keys "
        "from fpa.fpa_claims_monthly group by all order by month, channel",
        {"claims_paid": "USD", "payouts": "count"},
    ),
    # Claims net of returned payouts, as in the income statement, drivers and bridges.
    "cash_by_month": SavedSearchSpec(
        "fpa.fpa_cash_monthly",
        "select month, coalesce(collected_premium, 0) as collected_premium, "
        "coalesce(claims_paid, 0) - coalesce(payouts_returned, 0) as claims_paid, coalesce(lae, 0) as lae, "
        "coalesce(opex, 0) as opex, net_operating_cash, [month] as row_keys "
        "from fpa.fpa_cash_monthly order by month",
        {
            "collected_premium": "USD",
            "claims_paid": "USD",
            "lae": "USD",
            "opex": "USD",
            "net_operating_cash": "USD",
        },
    ),
}


class QueryRefused(ValueError):
    pass


def _nodes(node: Any) -> Iterator[dict[str, Any]]:
    if isinstance(node, dict):
        yield node
        for value in node.values():
            yield from _nodes(value)
    elif isinstance(node, list):
        for value in node:
            yield from _nodes(value)


@cache
def _builtin_functions() -> frozenset[str]:
    with duckdb.connect(config=NO_EXTERNAL_ACCESS) as conn:
        return frozenset(
            name
            for (name,) in conn.execute("select distinct function_name from duckdb_functions()").fetchall()
        )


def check_query(sql: str) -> str:
    """The SQL to run, or QueryRefused naming the rule it breaks."""
    text = sql.strip().rstrip(";").strip()
    if not text.lower().startswith("select"):
        raise QueryRefused("only a single SELECT statement is allowed")
    with duckdb.connect(config=NO_EXTERNAL_ACCESS) as parser:
        row = parser.execute("select json_serialize_sql(?)", [text]).fetchone()
    tree: dict[str, Any] = json.loads(row[0]) if row else {"error": True}
    if tree.get("error") or len(tree.get("statements", [])) != 1:
        raise QueryRefused("only a single SELECT statement is allowed")
    tables: set[str] = set()
    for node in _nodes(tree["statements"]):
        if node.get("class") == "STAR":
            raise QueryRefused("name the columns you need; SELECT * is not allowed")
        if node.get("type") == "TABLE_FUNCTION":
            raise QueryRefused("table functions are not allowed")
        if node.get("class") == "FUNCTION":
            name = str(node.get("function_name", "")).lower()
            if (
                node.get("schema")
                or name in REFUSED_FUNCTIONS
                or name.startswith(REFUSED_PREFIXES)
                or name not in _builtin_functions()
            ):
                raise QueryRefused(f"function {name!r} is not allowed")
        if node.get("type") == "BASE_TABLE":
            if node.get("catalog_name"):
                raise QueryRefused("catalog-qualified tables are not allowed")
            tables.add(f"{node.get('schema_name') or ''}.{node.get('table_name') or ''}".lower())
    if not tables or not tables <= LEDGER_TABLES:
        raise QueryRefused(f"only ledger tables may be read: {sorted(LEDGER_TABLES)}")
    return text


def aging_bucket(days_past_due: int, buckets: tuple[int, int, int]) -> str:
    """ADR 0001 aging: current up to the first edge, then monitor, escalate, and stale past the last."""
    return AGING_NAMES[sum(days_past_due > edge for edge in buckets)]


class LedgerAdapter(Protocol):
    def income_statement(self, period: str) -> Rows: ...
    def budget_vs_actual(self, period: str) -> Rows: ...
    def ar_aging(self, buckets: tuple[int, int, int]) -> Rows: ...
    def saved_search(self, name: SavedSearch) -> Rows: ...
    def query(self, sql: str) -> Rows: ...


class DuckDBLedger:
    """The synthetic NetSuite-shaped ledger in the warehouse."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path

    def _rows(
        self, sql: str, params: list[Any] | None = None, *, timeout: float | None = None
    ) -> list[dict[str, Any]]:
        with connect(self.db_path, read_only=True, config=NO_EXTERNAL_ACCESS) as conn:
            if timeout is None:
                return fetch_dicts(conn, sql, params)
            timer = threading.Timer(timeout, conn.interrupt)
            timer.start()
            try:
                return fetch_dicts(conn, sql, params)
            except duckdb.InterruptException as exc:
                raise QueryRefused(f"the query ran longer than {timeout:g} seconds") from exc
            finally:
                timer.cancel()

    def income_statement(self, period: str) -> Rows:
        by_account = {r["account"]: r for r in self._rows(INCOME_SQL, [period])}
        lines = [
            {
                "account": acct,
                "label": label,
                "amount": round(sign * float(by_account[acct]["amount"]), 2),
                "row_keys": [by_account[acct]["row_key"]],
            }
            for acct, label, sign in ACCOUNT_LINES
            if acct in by_account
        ]
        if lines:
            result = sum(line["amount"] * (1 if line["account"] == REVENUE else -1) for line in lines)
            keys = [key for line in lines for key in line["row_keys"]]
            lines.append(
                {"account": None, "label": "Operating result", "amount": round(result, 2), "row_keys": keys}
            )
        return Rows(lines, INCOME_SQL, ["fpa.fpa_ledger_monthly"])

    def budget_vs_actual(self, period: str) -> Rows:
        return Rows(self._rows(BVA_SQL, [period]), BVA_SQL, ["fpa.bridge_totals"])

    def ar_aging(self, buckets: tuple[int, int, int]) -> Rows:
        out = {
            name: {"bucket": name, "items": 0, "amount": 0.0, "instalment_ids": []} for name in AGING_NAMES
        }
        for r in self._rows(AGING_SQL):
            bucket = out[aging_bucket(r["days_past_due"], buckets)]
            bucket["items"] += 1
            bucket["amount"] += float(r["amount"])
            bucket["instalment_ids"].append(r["instalment_id"])
        for bucket in out.values():
            bucket["amount"] = round(bucket["amount"], 2)
        edges = ", ".join(f"{name} to {edge} days" for name, edge in zip(AGING_NAMES, buckets, strict=False))
        return Rows(list(out.values()), f"{AGING_SQL}-- buckets: {edges}, stale beyond", ["fpa.fpa_ar_aging"])

    def saved_search(self, name: SavedSearch) -> Rows:
        if name not in SAVED_SEARCHES:
            raise KeyError(f"unknown saved search {name!r}")
        spec = SAVED_SEARCHES[name]
        return Rows(self._rows(spec.sql), spec.sql, [spec.source], spec.units)

    def query(self, sql: str) -> Rows:
        """Raw query, last resort: one SELECT over ledger tables, no ``*``, at most 1,000 rows."""
        text = check_query(sql)
        wrapped = f"select q.* from ({text}) q limit {MAX_QUERY_ROWS}"  # noqa: S608 (checked above)
        tables = sorted(LEDGER_TABLES)
        return Rows(self._rows(wrapped, timeout=QUERY_TIMEOUT_SECONDS), text, tables)


def ledger(db_path: Path) -> LedgerAdapter:
    """The ledger the tools read. Swap the class here to point them at a different ledger."""
    return DuckDBLedger(db_path)
