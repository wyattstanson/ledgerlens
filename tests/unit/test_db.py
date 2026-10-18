from pathlib import Path

from ledgerlens.core.db import connect, ensure_audit_schema

AUDIT_TABLES = {"agent_runs", "tool_calls", "lineage", "llm_calls", "approvals", "analysis_plans"}


def test_audit_schema_has_every_table_and_is_idempotent(db: Path):
    ensure_audit_schema(db)  # second run must not fail
    with connect(db) as conn:
        tables = {
            r[0]
            for r in conn.execute(
                "select table_name from information_schema.tables where table_schema = 'audit'"
            ).fetchall()
        }
    assert tables == AUDIT_TABLES


def test_connect_creates_parent_folder(tmp_path: Path):
    path = tmp_path / "nested" / "w.duckdb"
    ensure_audit_schema(path)
    assert path.exists()
