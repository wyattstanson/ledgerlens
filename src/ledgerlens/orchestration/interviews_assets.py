"""Dagster assets and checks for the interviews pack (ADR 0004)."""

import dagster as dg

from ledgerlens.core.db import connect, ensure_audit_schema
from ledgerlens.interviews.config import InterviewsConfig
from ledgerlens.interviews.generate import generate_world, write_world
from ledgerlens.interviews.pipeline import CELL_TABLES, TABLES, run_interviews
from ledgerlens.interviews.plan import AnalysisPlan
from ledgerlens.interviews.tagger import RuleTagger
from ledgerlens.settings import get_settings


@dg.asset(
    group_name="interviews", description="Seeded synthetic interview world; person-level data to the vault."
)
def interviews_world() -> dg.MaterializeResult:
    settings = get_settings()
    ensure_audit_schema(settings.db)
    config = InterviewsConfig.from_yaml(settings.config_dir / "interviews.yaml")
    world = generate_world(config.world, settings.seed)
    write_world(settings.db, settings.restricted_db, world)
    return dg.MaterializeResult(metadata={"ratings": world.ratings.height, "seed": settings.seed})


@dg.asset(
    group_name="interviews",
    deps=[interviews_world],
    description="Plan-registered interview metrics and flags.",
)
def interviews_pipeline() -> dg.MaterializeResult:
    settings = get_settings()
    plan = AnalysisPlan.from_yaml(settings.config_dir / "analysis_plan.yaml")
    registered = run_interviews(settings.db, settings.restricted_db, plan, RuleTagger())
    return dg.MaterializeResult(
        metadata={"plan_version": registered.version, "plan_sha256": registered.sha256}
    )


ORPHAN_SQL = (
    "select count(*) from interviews.{table} x left join audit.analysis_runs r using (analysis_run_id) "
    "where r.analysis_run_id is null or r.plan_sha256 <> x.plan_sha256"
)
SMALL_CELL_SQL = "select count(*) from interviews.{table} where n < ? and not suppressed"


def _count(sql: str, tables: tuple[str, ...], params: list[object] | None = None) -> int:
    """Sum a count query over the pipeline's own table names (constants, never input)."""
    # Read-only: Dagster runs checks in parallel processes, and DuckDB allows one writer per file.
    with connect(get_settings().db, read_only=True) as conn:
        return sum(
            int((conn.execute(sql.format(table=t), params or []).fetchone() or (0,))[0]) for t in tables
        )


@dg.asset_check(asset=interviews_pipeline, description="Every result row cites a registered plan run.")
def plan_hash_logged() -> dg.AssetCheckResult:
    orphans = _count(ORPHAN_SQL, TABLES)
    return dg.AssetCheckResult(passed=orphans == 0, metadata={"rows_without_registered_plan": orphans})


@dg.asset_check(asset=interviews_pipeline, description="No published cell below the minimum group size.")
def no_small_cells() -> dg.AssetCheckResult:
    floor = AnalysisPlan.from_yaml(get_settings().config_dir / "analysis_plan.yaml").min_group_n
    leaks = _count(SMALL_CELL_SQL, CELL_TABLES, [floor])
    return dg.AssetCheckResult(
        passed=leaks == 0, severity=dg.AssetCheckSeverity.ERROR, metadata={"leaking_rows": leaks}
    )


INTERVIEW_ASSETS = [interviews_world, interviews_pipeline]
INTERVIEW_CHECKS = [plan_hash_logged, no_small_cells]
