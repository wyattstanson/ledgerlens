"""Export the warehouse the hosted demo serves, compressed, for Render to unpack at build time.

Render's free instance has 512 MB of memory. Building the world there needs more than that (the rate
card pipeline alone peaks around 520 MB), so the demo ships a warehouse built here instead. Serving it
needs about 215 MB, which fits comfortably.

What is left out: the raw and intermediate schemas the generators write, which only dbt reads. The
NetSuite-shaped GL staging tables stay, big as they are, because ``ledger.saved_search`` lists them in
its allowed set; dropping them would break that tool on the deployment alone. The test for this script
holds that line, comparing the skip list against every table the tools name.

    uv run python scripts/export_demo_warehouse.py

Rebuild it whenever the generators, the marts or the forecast change, and commit the result.
"""

import gzip
import shutil
import sys
from pathlib import Path

import duckdb

REPO_ROOT = Path(__file__).resolve().parents[1]
TARGET = REPO_ROOT / "deploy" / "warehouse.duckdb.gz"
SCHEMAS = ("fpa", "interviews", "ratecards", "audit")
# Nothing inside these schemas is dropped today: the two big GL staging tables looked droppable, but
# ledger.saved_search allows queries against them. The test keeps this honest if it ever grows.
SKIP: set[str] = set()


def tables(conn: duckdb.DuckDBPyConnection) -> list[tuple[str, str]]:
    placeholders = ", ".join("?" for _ in SCHEMAS)
    rows = conn.execute(
        f"select table_schema, table_name from information_schema.tables "  # noqa: S608 (fixed list)
        f"where table_schema in ({placeholders}) order by 1, 2",
        list(SCHEMAS),
    ).fetchall()
    return [(s, t) for s, t in rows if f"{s}.{t}" not in SKIP]


def export(source: Path, target: Path) -> tuple[int, int]:
    """Copy the demo tables into a fresh file, gzip it, and return (tables, compressed bytes)."""
    target.parent.mkdir(parents=True, exist_ok=True)
    plain = target.with_suffix("")  # .../warehouse.duckdb
    plain.unlink(missing_ok=True)
    # Read-write: a read-only connection cannot attach the new file. Nothing is written to the source.
    conn = duckdb.connect(source)
    try:
        wanted = tables(conn)
        # ATTACH takes a literal, not a parameter; the path is this script's own, not user input.
        conn.execute(f"attach '{plain.as_posix()}' as demo")
        for schema in SCHEMAS:
            conn.execute(f"create schema if not exists demo.{schema}")
        for schema, table in wanted:
            # Names come from information_schema on this same file, not from a caller.
            copy = f'create table demo."{schema}"."{table}" as select * from "{schema}"."{table}"'  # noqa: S608
            conn.execute(copy)
        conn.execute("detach demo")
    finally:
        conn.close()
    with plain.open("rb") as raw, gzip.open(target, "wb", compresslevel=9) as packed:
        shutil.copyfileobj(raw, packed)
    plain.unlink()
    return len(wanted), target.stat().st_size


def main() -> int:
    source = REPO_ROOT / "data" / "warehouse.duckdb"
    if not source.exists():
        print(f"no warehouse at {source}: run the generate and pipeline commands first", file=sys.stderr)
        return 1
    count, size = export(source, TARGET)
    print(f"{count} tables -> {TARGET.relative_to(REPO_ROOT)} ({size / 1048576:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
