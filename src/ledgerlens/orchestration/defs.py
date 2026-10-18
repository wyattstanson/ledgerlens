"""Dagster definitions: dbt models become assets. Python assets and asset checks join in later phases.

The dbt manifest must exist before import (``make dbt-parse``). ``prepare_if_dev`` only regenerates it
under ``dagster dev``; tests, CI and ``dg.materialize`` rely on the Makefile target.
"""

from pathlib import Path

import dagster as dg
from dagster_dbt import DbtCliResource, DbtProject, dbt_assets

REPO_ROOT = Path(__file__).resolve().parents[3]
DBT_PROJECT_DIR = REPO_ROOT / "transform"

dbt_project = DbtProject(project_dir=DBT_PROJECT_DIR, profiles_dir=DBT_PROJECT_DIR)
dbt_project.prepare_if_dev()


@dbt_assets(manifest=dbt_project.manifest_path, project=dbt_project)
def ledgerlens_dbt_assets(context: dg.AssetExecutionContext, dbt: DbtCliResource):
    yield from dbt.cli(["build"], context=context).stream()


defs = dg.Definitions(
    assets=[ledgerlens_dbt_assets],
    resources={"dbt": DbtCliResource(project_dir=dbt_project)},
)
