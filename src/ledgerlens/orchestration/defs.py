"""Dagster definitions: dbt models become assets, pack compute steps are Python assets, checks watch them.

The dbt manifest must exist before import (``make dbt-parse``). ``prepare_if_dev`` only regenerates it
under ``dagster dev``; tests, CI and ``dg.materialize`` rely on the Makefile target.
"""

import json
from pathlib import Path

import dagster as dg
from dagster_dbt import DbtCliResource, DbtProject, dbt_assets

from ledgerlens.core.db import connect, ensure_audit_schema
from ledgerlens.ratecards.config import RateCardsConfig
from ledgerlens.ratecards.generate import generate_world, write_world
from ledgerlens.ratecards.pipeline import run_ratecards
from ledgerlens.settings import get_settings

REPO_ROOT = Path(__file__).resolve().parents[3]
DBT_PROJECT_DIR = REPO_ROOT / "transform"

dbt_project = DbtProject(project_dir=DBT_PROJECT_DIR, profiles_dir=DBT_PROJECT_DIR)
dbt_project.prepare_if_dev()


@dbt_assets(manifest=dbt_project.manifest_path, project=dbt_project)
def ledgerlens_dbt_assets(context: dg.AssetExecutionContext, dbt: DbtCliResource):
    yield from dbt.cli(["build"], context=context).stream()


def _ratecards_config() -> RateCardsConfig:
    return RateCardsConfig.from_yaml(get_settings().config_dir / "ratecards.yaml")


RAW_RATECARD_KEYS = [dg.AssetKey(["raw_ratecards", t]) for t in ("invoice_lines", "rate_cards", "vendors")]


@dg.multi_asset(
    specs=[
        dg.AssetSpec(
            key, group_name="ratecards", description="Seeded synthetic rate-card data (and hidden truth)."
        )
        for key in RAW_RATECARD_KEYS
    ]
)
def ratecards_world():
    settings = get_settings()
    ensure_audit_schema(settings.db)
    world = generate_world(_ratecards_config().world, settings.seed)
    write_world(settings.db, world)
    sizes = {
        "invoice_lines": world.invoice_lines.height,
        "rate_cards": world.rate_cards.height,
        "vendors": world.vendors.height,
    }
    for key in RAW_RATECARD_KEYS:
        yield dg.MaterializeResult(
            asset_key=key, metadata={"rows": sizes[key.path[-1]], "seed": settings.seed}
        )


@dg.asset(
    group_name="ratecards",
    deps=[
        dg.AssetKey(["ratecards", "stg_rc__invoice_lines"]),
        dg.AssetKey(["ratecards", "stg_rc__rate_cards"]),
    ],
    description="Title mapping, normalization, leakage, benchmarks, panel, savings bridge, tail plan.",
)
def ratecards_pipeline() -> dg.MaterializeResult:
    settings = get_settings()
    solution = run_ratecards(settings.db, _ratecards_config())
    with connect(settings.db) as conn:
        status, pending = conn.execute(
            "select status, pending_reviews from ratecards.panel_meta"
        ).fetchone() or (
            "unknown",
            0,
        )
    return dg.MaterializeResult(
        metadata={
            "panel_status": status,
            "pending_title_reviews": pending,
            "kept_vendors": json.dumps(list(solution.kept)) if solution else "[]",
        }
    )


@dg.asset_check(asset=ratecards_pipeline, description="No invoice line billed more than 2% above its card.")
def no_billing_drift() -> dg.AssetCheckResult:
    with connect(get_settings().db) as conn:
        count, usd = conn.execute(
            "select count(*), coalesce(sum(leakage_usd), 0) from ratecards.leakage where kind = 'overbill'"
        ).fetchone() or (0, 0.0)
    return dg.AssetCheckResult(
        passed=count == 0,
        severity=dg.AssetCheckSeverity.WARN,
        metadata={"lines_over_card": int(count), "excess_usd": float(usd)},
    )


defs = dg.Definitions(
    assets=[ledgerlens_dbt_assets, ratecards_world, ratecards_pipeline],
    asset_checks=[no_billing_drift],
    resources={"dbt": DbtCliResource(project_dir=dbt_project)},
)
