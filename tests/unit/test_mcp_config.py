"""The generic DuckDB MCP server must not be able to reach files beyond the warehouse (ADR 0004)."""

import json

import duckdb
import pytest

from ledgerlens.settings import REPO_ROOT

LOCKDOWN = "SET enable_external_access=false; SET lock_configuration=true"


def test_the_duckdb_mcp_server_starts_locked_down():
    args = json.loads((REPO_ROOT / ".mcp.json").read_text(encoding="utf-8"))["mcpServers"]["duckdb"]["args"]
    assert args[args.index("--init-sql") + 1] == LOCKDOWN
    assert "--read-write" not in args


def test_the_lockdown_blocks_attaching_the_vault_and_reading_files(tmp_path):
    vault = tmp_path / "vault.duckdb"
    with duckdb.connect(vault) as conn:
        conn.execute("create schema restricted; create table restricted.notes as select 'x' as text")
    (tmp_path / "sheet.csv").write_text("note_id,note_text\nNT-1,hello\n", encoding="utf-8")
    with duckdb.connect(tmp_path / "warehouse.duckdb") as conn:
        conn.execute(LOCKDOWN)
        for sql in (
            f"attach '{vault}' as v (read_only)",
            f"select * from read_csv('{tmp_path / 'sheet.csv'}')",  # noqa: S608 (a test path)
            "set enable_external_access=true",
        ):
            with pytest.raises((duckdb.PermissionException, duckdb.InvalidInputException)):
                conn.execute(sql)
        assert conn.execute("select 1").fetchone() == (1,)
