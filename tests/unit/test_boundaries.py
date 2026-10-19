"""Static boundaries: core knows no packs; only the interviews job and generator read the vault."""

import ast
import re
from pathlib import Path

from ledgerlens.interviews.generate import VAULT_TABLES
from ledgerlens.settings import REPO_ROOT

SRC = REPO_ROOT / "src" / "ledgerlens"
PACKS = ("ledgerlens.ratecards", "ledgerlens.interviews", "ledgerlens.fpa", "ledgerlens.packs")
# The eval CLI is a composition root: it imports the pack registry lazily inside main().
COMPOSITION_ROOTS_IN_CORE = {"core/evals/run.py"}
# SQL naming an interview vault table (rate cards keep separate generator truth in the warehouse).
VAULT_TABLE = "|".join((*VAULT_TABLES, "world_meta"))
VAULT_SQL = re.compile(rf"\b(from|join|table)\s+restricted\.({VAULT_TABLE})\b", re.IGNORECASE)


def modules() -> dict[Path, ast.Module]:
    return {p: ast.parse(p.read_text(encoding="utf-8")) for p in SRC.rglob("*.py")}


def imported(tree: ast.Module) -> set[str]:
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
        elif isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
    return names


def rel(path: Path) -> str:
    return path.relative_to(SRC).as_posix()


def test_core_never_imports_a_pack():
    offenders = {
        rel(p): sorted(m for m in imported(t) if m.startswith(PACKS))
        for p, t in modules().items()
        if rel(p).startswith("core/") and rel(p) not in COMPOSITION_ROOTS_IN_CORE
    }
    assert {k: v for k, v in offenders.items() if v} == {}


def test_only_the_generator_and_pipeline_query_the_vault_schema():
    readers = {
        rel(p)
        for p, t in modules().items()
        if any(
            isinstance(n, ast.Constant) and isinstance(n.value, str) and VAULT_SQL.search(n.value)
            for n in ast.walk(t)
        )
    }
    assert readers == {"interviews/generate.py", "interviews/pipeline.py"}


def test_only_composition_roots_open_the_vault_path():
    users = {
        rel(p)
        for p, t in modules().items()
        if any(isinstance(n, ast.Attribute) and n.attr == "restricted_db" for n in ast.walk(t))
    }
    assert users == {"cli.py", "orchestration/interviews_assets.py"}


def test_tools_and_compliance_never_import_vault_readers():
    vault_readers = {"ledgerlens.interviews.pipeline", "ledgerlens.interviews.generate"}
    for name in ("interviews/tools.py", "interviews/compliance.py", "mcp/server.py", "core/agent_loop.py"):
        tree = ast.parse((SRC / name).read_text(encoding="utf-8"))
        assert not imported(tree) & vault_readers, name
