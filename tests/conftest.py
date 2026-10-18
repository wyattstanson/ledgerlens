import subprocess
import sys
from pathlib import Path

import pytest

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
