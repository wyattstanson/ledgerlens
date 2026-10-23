"""Dagster assets and checks for the FP&A pack (BUILD_PROMPT section 10, ADR 0001).

``fpa_world`` writes the raw tables the dbt FP&A sources read, so dagster-dbt links them as upstream of the
staging models. ``fpa_pipeline`` runs after the marts; it skips its own dbt build because Dagster has just
run it (and its tests, which become asset checks).
"""

import dagster as dg

from ledgerlens.core.db import connect, ensure_audit_schema
from ledgerlens.fpa.config import FpaConfig
from ledgerlens.fpa.generate import PLAN_TABLES, RAW_TABLES, generate_world, write_world
from ledgerlens.fpa.pipeline import run_fpa
from ledgerlens.fpa.plan import build_plan
from ledgerlens.settings import get_settings

# Everything write_world writes, so every dbt source in transform/models/fpa/staging/_sources.yml has a key.
RAW_FPA_KEYS = [dg.AssetKey(["raw_fpa", t]) for t in (*RAW_TABLES, *PLAN_TABLES)]
MARTS = (
    "fpa_premium_monthly",
    "fpa_claims_monthly",
    "fpa_cash_monthly",
    "fpa_cash_weekly",
    "fpa_ledger_monthly",
    "fpa_ar_aging",
)
RESIDUAL_LIMIT_CENTS = {"net_operating_cash": 8}  # four measures' rounding; every other measure 2


def _config() -> FpaConfig:
    return FpaConfig.from_yaml(get_settings().config_dir / "fpa.yaml")


@dg.multi_asset(
    specs=[
        dg.AssetSpec(
            key, group_name="fpa", description="Seeded synthetic insurer: payments feed, ledger, plan."
        )
        for key in RAW_FPA_KEYS
    ]
)
def fpa_world():
    settings = get_settings()
    ensure_audit_schema(settings.db)
    config = _config()
    world = generate_world(config.world, settings.seed)
    plan = build_plan(config.world, settings.seed)
    write_world(settings.db, world, plan)
    frames = {name: getattr(world, name) for name in RAW_TABLES} | plan
    for key in RAW_FPA_KEYS:
        frame = frames[key.path[-1]]
        yield dg.MaterializeResult(asset_key=key, metadata={"rows": frame.height, "seed": settings.seed})


@dg.asset(
    group_name="fpa",
    deps=[dg.AssetKey(["fpa", m]) for m in MARTS],
    description="Driver tree, volume/mix/rate/timing bridges, ADR 0001 materiality flags and the scenario.",
)
def fpa_pipeline() -> dg.MaterializeResult:
    run = run_fpa(get_settings().db, _config(), build_dbt=False)
    material = run.totals.filter(run.totals["status"] == "material").height
    return dg.MaterializeResult(
        metadata={
            "as_of": run.as_of,
            "open_period": run.open_period,
            "material_variances": material,
            "scenario": run.scenario.name,
        }
    )


BRIDGE_GAP_SQL = """
select count(*) from (
    select t.row_key, t.variance_cents, coalesce(sum(b.cents), 0) as bridged
    from fpa.bridge_totals t left join fpa.bridge b on b.period = t.period and b.measure = t.measure
    group by all
) where bridged <> variance_cents
"""


RESIDUAL_SQL = "select measure, max(abs(cents)) from fpa.bridge where component = 'residual' group by measure"


@dg.asset_check(
    asset=fpa_pipeline,
    description="Every bridge sums to its variance to the cent, and the residual is only rounding.",
)
def bridge_sums_to_variance() -> dg.AssetCheckResult:
    # The residual absorbs whatever volume, mix, rate and timing miss, so the sum alone always holds; a
    # residual beyond rounding is what shows the decomposition broke.
    # Read-only: Dagster runs checks in parallel processes, and DuckDB allows one writer per file.
    with connect(get_settings().db, read_only=True) as conn:
        gaps = int((conn.execute(BRIDGE_GAP_SQL).fetchone() or (0,))[0])
        residuals = dict(conn.execute(RESIDUAL_SQL).fetchall())
    too_big = {m: int(c) for m, c in residuals.items() if c > RESIDUAL_LIMIT_CENTS.get(m, 2)}
    return dg.AssetCheckResult(
        passed=gaps == 0 and not too_big,
        severity=dg.AssetCheckSeverity.ERROR,
        metadata={"bridges_not_summing": gaps, "residuals_beyond_rounding_cents": str(too_big or "none")},
    )


FPA_ASSETS = [fpa_world, fpa_pipeline]
FPA_CHECKS = [bridge_sums_to_variance]
