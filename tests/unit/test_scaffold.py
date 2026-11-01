"""Scaffold checks: package version, the dbt manifest and the Dagster definitions."""

import json
import tomllib
from pathlib import Path

import dagster as dg

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
    assert {"interviews_world", "interviews_pipeline"} <= asset_keys
    assert {"raw_fpa/transaction", "fpa/fpa_cash_monthly", "fpa_pipeline", "fpa_forecast"} <= asset_keys
    check_names = {key.name for key in defs.resolve_asset_graph().asset_check_keys}
    assert {
        "no_billing_drift",
        "plan_hash_logged",
        "no_small_cells",
        "bridge_sums_to_variance",
        "forecast_actuals_within_interval",
    } <= check_names
    assert isinstance(defs, dg.Definitions)
