import subprocess
import sys
from pathlib import Path

import pytest

from ledgerlens.core.approvals import ApprovalQueue
from ledgerlens.core.db import ensure_audit_schema
from ledgerlens.core.llm import ModelsConfig

REPO_ROOT = Path(__file__).resolve().parents[1]
DBT_DIR = REPO_ROOT / "transform"
MANIFEST = DBT_DIR / "target" / "manifest.json"


@pytest.fixture(scope="session")
def dbt_manifest() -> Path:
    """Ensure the dbt manifest exists, parsing from inside transform/ (never the repo root)."""
    if not MANIFEST.exists():
        dbt = Path(sys.executable).with_name("dbt")
        subprocess.run([str(dbt), "parse", "--profiles-dir", "."], cwd=DBT_DIR, check=True)
    return MANIFEST


@pytest.fixture
def db(tmp_path: Path) -> Path:
    """A fresh warehouse file with the audit schema, isolated per test."""
    path = tmp_path / "warehouse.duckdb"
    ensure_audit_schema(path)
    return path


@pytest.fixture(scope="session")
def models_config() -> ModelsConfig:
    return ModelsConfig.from_yaml(REPO_ROOT / "config" / "models.yaml")


@pytest.fixture
def approvals(db: Path) -> ApprovalQueue:
    return ApprovalQueue(db)
