"""DuckDB access and the audit schema (BUILD_PROMPT section 9.3).

Connections are short-lived (open, work, close) so the single warehouse file is never held locked
while Dagster, the API or the MCP server also need it.
"""

from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

import duckdb

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
    reason       varchar
);

create table if not exists audit.analysis_plans (
    plan_version  integer primary key,
    sha256        varchar not null unique,
    registered_at timestamp not null
);
"""


def utcnow() -> datetime:
    """Naive UTC timestamp (DuckDB TIMESTAMP columns hold UTC by convention here)."""
    return datetime.now(UTC).replace(tzinfo=None)


@contextmanager
def connect(db_path: Path, *, read_only: bool = False) -> Iterator[duckdb.DuckDBPyConnection]:
    if not read_only:
        db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = duckdb.connect(str(db_path), read_only=read_only)
    try:
        yield conn
    finally:
        conn.close()


def ensure_audit_schema(db_path: Path) -> None:
    with connect(db_path) as conn:
        conn.execute(AUDIT_DDL)
