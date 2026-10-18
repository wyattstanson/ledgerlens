"""Phase 0 scaffold checks: package, dbt manifest, Dagster definitions, Streamlit app."""

import json
import tomllib
from pathlib import Path

import dagster as dg
from streamlit.testing.v1 import AppTest

import ledgerlens

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_version_matches_pyproject():
    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert ledgerlens.__version__ == pyproject["project"]["version"]


def test_dbt_manifest_paths_are_relative_to_transform(dbt_manifest: Path):
    # Parsing from the repo root bakes "transform/..." into paths and breaks seeds under Dagster.
    nodes = json.loads(dbt_manifest.read_text(encoding="utf-8"))["nodes"]
    model = nodes["model.ledgerlens.core_project_info"]
    assert model["original_file_path"].replace("\\", "/") == "models/core/core_project_info.sql"


def test_dagster_definitions_load_dbt_assets(dbt_manifest: Path):
    # Imported here, not at module top: defs reads the manifest at import time.
    from ledgerlens.orchestration.defs import defs  # noqa: PLC0415

    asset_keys = {key.to_user_string() for key in defs.resolve_asset_graph().get_all_asset_keys()}
    # dagster-dbt prefixes asset keys with the model's custom schema (+schema: core).
    assert "core/core_project_info" in asset_keys
    assert isinstance(defs, dg.Definitions)


def test_streamlit_home_renders_without_errors():
    at = AppTest.from_file(str(REPO_ROOT / "app" / "Home.py")).run(timeout=30)
    assert not at.exception
    assert at.title[0].value == "LedgerLens"


def test_streamlit_usage_stats_disabled():
    config = tomllib.loads((REPO_ROOT / ".streamlit" / "config.toml").read_text(encoding="utf-8"))
    assert config["browser"]["gatherUsageStats"] is False
