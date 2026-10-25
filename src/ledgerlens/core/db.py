"""DuckDB access and the audit schema (BUILD_PROMPT section 9.3).

Connections are short-lived (open, work, close) so the single warehouse file is never held locked
while Dagster, the API or the MCP server also need it.
"""

from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import duckdb

DuckConfig = dict[str, str | bool | int | float | list[str]]
# For connections that only read the warehouse: no file reads, getenv(), ATTACH or extensions. DuckDB
# refuses a second connection to one file with a different configuration while the first is open, so
# readers that use this must not overlap a plain connection in the same process.
NO_EXTERNAL_ACCESS: DuckConfig = {"enable_external_access": False}

AUDIT_DDL = """
create schema if not exists audit;

create table if not exists audit.agent_runs (
    agent_run_id varchar primary key,
    agent_id     varchar not null,
    "user"       varchar not null,
    started_at   timestamp not null,
    finished_at  timestamp,
    status       varchar not null
);

create table if not exists audit.tool_calls (
    tool_call_id varchar primary key,
    agent_run_id varchar not null,
    tool_name    varchar not null,
    input_json   varchar not null,
    output_hash  varchar,
    started_at   timestamp not null,
    ms           double,
    status       varchar not null
);

create table if not exists audit.lineage (
    lineage_id    varchar primary key,
    tool_call_id  varchar not null,
    sql_hash      varchar not null,
    sql_text      varchar not null,
    source_tables varchar[] not null,
    row_ids       varchar[] not null,
    value         double not null,
    unit          varchar not null
);

create table if not exists audit.llm_calls (
    call_id            varchar primary key,
    agent_run_id       varchar,
    task               varchar not null,
    model              varchar not null,
    effort             varchar not null,
    input_tokens       bigint not null,
    output_tokens      bigint not null,
    cache_read_tokens  bigint not null,
    cache_write_tokens bigint not null,
    cost_usd           double not null,
    ms                 double not null,
    created_at         timestamp not null
);

create table if not exists audit.approvals (
    approval_id  varchar primary key,
    agent_run_id varchar not null,
    action       varchar not null,
    payload_json varchar not null,
    status       varchar not null,
    created_at   timestamp not null,
    decided_by   varchar,
    decided_at   timestamp,
    reason       varchar,
    decided_kind varchar
);

create table if not exists audit.analysis_plans (
    plan_name     varchar not null,
    plan_version  integer not null,
    sha256        varchar not null,
    plan_json     varchar not null,
    registered_at timestamp not null,
    primary key (plan_name, plan_version),
    unique (plan_name, sha256)
);

create table if not exists audit.analysis_runs (
    analysis_run_id varchar primary key,
    pack            varchar not null,
    plan_name       varchar not null,
    plan_version    integer not null,
    plan_sha256     varchar not null,
    seed            bigint not null,
    started_at      timestamp not null,
    finished_at     timestamp,
    status          varchar not null
);
"""


def utcnow() -> datetime:
    """Naive UTC timestamp (DuckDB TIMESTAMP columns hold UTC by convention here)."""
    return datetime.now(UTC).replace(tzinfo=None)


@contextmanager
def connect(
    db_path: Path, *, read_only: bool = False, config: DuckConfig | None = None
) -> Iterator[duckdb.DuckDBPyConnection]:
    if not read_only:
        db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = duckdb.connect(str(db_path), read_only=read_only, config=config or {})
    try:
        yield conn
    finally:
        conn.close()


def fetch_dicts(
    conn: duckdb.DuckDBPyConnection, sql: str, params: list[Any] | None = None
) -> list[dict[str, Any]]:
    """Rows of a query as dicts keyed by column name."""
    cursor = conn.execute(sql, params or [])
    columns = [d[0] for d in cursor.description]
    return [dict(zip(columns, row, strict=True)) for row in cursor.fetchall()]


def ensure_audit_schema(db_path: Path) -> None:
    with connect(db_path) as conn:
        conn.execute(AUDIT_DDL)
        # Warehouses created before decided_kind existed get the column added in place.
        conn.execute("alter table audit.approvals add column if not exists decided_kind varchar")
        # Phase 0-2 warehouses have an unused single-plan analysis_plans table: replace it while it is empty.
        old = conn.execute(
            "select count(*) from information_schema.columns "
            "where table_schema = 'audit' and table_name = 'analysis_plans' and column_name = 'plan_name'"
        ).fetchone()
        if old is not None and old[0] == 0:
            rows = conn.execute("select count(*) from audit.analysis_plans").fetchone()
            if rows is not None and rows[0] > 0:
                raise RuntimeError("audit.analysis_plans has rows in the pre-Phase-3 layout; migrate by hand")
            conn.execute("drop table audit.analysis_plans")
            conn.execute(AUDIT_DDL)
