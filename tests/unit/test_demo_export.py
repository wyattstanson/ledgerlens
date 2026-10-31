"""The demo warehouse the hosted deployment serves must still hold everything the tools read.

The export drops two staging tables to fit a free instance. If a tool ever starts reading one, the
deployment would break in a way no local run would show, so the skip list is checked against the SQL
the tools actually contain.
"""

import re
from pathlib import Path

from export_demo_warehouse import SKIP

REPO_ROOT = Path(__file__).resolve().parents[2]
TOOL_FILES = [
    REPO_ROOT / "src" / "ledgerlens" / "fpa" / f
    for f in ("tools.py", "ledger_adapter.py", "payments_adapter.py", "forecast.py")
] + [
    REPO_ROOT / "src" / "ledgerlens" / "interviews" / "tools.py",
    REPO_ROOT / "src" / "ledgerlens" / "ratecards" / "tools.py",
]
QUALIFIED = re.compile(r"\b(fpa|interviews|ratecards|audit)\.([a-z_][a-z0-9_]*)")


def tables_the_tools_read() -> set[str]:
    found: set[str] = set()
    for path in TOOL_FILES:
        pairs = QUALIFIED.findall(path.read_text(encoding="utf-8"))
        found |= {f"{schema}.{table}" for schema, table in pairs}
    return found


def test_the_export_keeps_every_table_a_tool_queries():
    read = tables_the_tools_read()
    assert read, "the scan found no table names, so it would pass for the wrong reason"
    assert not (SKIP & read), f"the demo export drops tables the tools read: {sorted(SKIP & read)}"


def test_anything_skipped_is_named_well_enough_to_check():
    # Empty today. If something is dropped later it must be schema-qualified, or the test above
    # compares names that can never match and silently passes.
    for name in SKIP:
        schema, table = name.split(".")
        assert schema in {"fpa", "interviews", "ratecards", "audit"}
        assert table
