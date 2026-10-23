"""The FP&A job: build the dbt marts, then drivers, variance bridges, materiality flags and the scenario.

Order: ``dbt build`` for the FP&A models and tests (the ledger must reconcile before anything is computed)
-> driver table -> one bridge per measure for every month and for fiscal year to date -> ADR 0001 flags ->
the built-in scenario -> tables in schema ``fpa``. Forecasting will live in ``fpa/forecast.py`` once the
ml-pipeline skill's phase reviews are approved (``ml_pipeline/PIPELINE.md``); nothing here fits a model.
"""

import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import polars as pl

from ledgerlens.core.db import connect
from ledgerlens.fpa.bridge import COMPONENTS, Bridge, Segment, combine, variance_bridge
from ledgerlens.fpa.config import FpaConfig
from ledgerlens.fpa.drivers import Marts, driver_table, load_marts
from ledgerlens.fpa.materiality import Kind, Variance, flag, level_for
from ledgerlens.fpa.scenario import ScenarioResult, as_of_date, check_to_instant
from ledgerlens.settings import get_settings

MEASURES: dict[str, tuple[Kind, int]] = {  # measure -> (kind, sign in net operating cash)
    "collected_premium": ("revenue", 1),
    "claims_paid": ("expense", -1),
    "lae": ("expense", -1),
    "opex": ("expense", -1),
}
NET = "net_operating_cash"
SECRET_ENV = re.compile(r"KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL", re.IGNORECASE)  # never passed to dbt
MATERIALITY_BAND = (10_000_000, 100_000_000)  # ADR 0001's thresholds are calibrated for this revenue range


class DbtBuildError(RuntimeError):
    pass


def run_dbt(warehouse: Path, select: str = "path:models/fpa") -> None:
    """Build and test the FP&A models against ``warehouse``; any failing model or test stops the job.

    dbt runs in its own process, as under Dagster: dbt-duckdb keeps its connection open after a build, and
    DuckDB will not then open the same file read-only (or with external access off) in this process.
    """
    project = get_settings().dbt_project_dir
    command = [
        sys.executable,
        "-m",
        "dbt.cli.main",
        "build",
        "--select",
        select,
        "--project-dir",
        str(project),
        "--profiles-dir",
        str(project),
        "--quiet",
    ]
    env = {k: v for k, v in os.environ.items() if not SECRET_ENV.search(k)} | {
        "LEDGERLENS_DBT_DB": str(warehouse)
    }
    result = subprocess.run(  # noqa: S603 (fixed argv, no shell)
        command, env=env, cwd=project, capture_output=True, text=True, check=False
    )
    if result.returncode != 0:
        output = (result.stdout + result.stderr).strip()[-2000:]
        raise DbtBuildError(f"dbt build failed for {select}:\n{output or 'a model or test failed'}")


def _segments(
    actual: pl.DataFrame, plan: pl.DataFrame, keys: list[str], q: tuple[str, str], a: tuple[str, str]
):
    """Segments joined on ``keys``; q and a name the (actual, plan) quantity and amount columns."""
    act = actual.group_by(keys).agg(pl.col(q[0]).sum().alias("qa"), pl.col(a[0]).sum().alias("aa"))
    pln = plan.group_by(keys).agg(pl.col(q[1]).sum().alias("qp"), pl.col(a[1]).sum().alias("ap"))
    joined = act.join(pln, on=keys, how="full", coalesce=True).fill_null(0.0)
    return [
        Segment("|".join(str(r[k]) for k in keys), r["qa"], r["aa"], r["qp"], r["ap"])
        for r in joined.sort(keys).iter_rows(named=True)
    ]


def bridges_for(marts: Marts, months: list[str]) -> dict[str, Bridge]:
    """The four measure bridges and net operating cash for a set of months (one month, or year to date)."""
    pick = pl.col("month").is_in(months)
    prem, plan_prem = marts.premium.filter(pick), marts.plan_premium.filter(pick)
    claims, plan_claims = marts.claims.filter(pick), marts.plan_claims.filter(pick)
    opex = float(marts.cash.filter(pick)["opex"].fill_null(0).sum())
    plan_opex = float(marts.plan_opex.filter(pick)["amount"].sum())
    out = {
        "collected_premium": variance_bridge(
            _segments(
                prem,
                plan_prem,
                ["lob", "method"],
                ("instalments_due", "instalments_due"),
                ("due_amount", "due_amount"),
            ),
            float(prem["collected_amount"].sum()),
            float(plan_prem["collected_amount"].sum()),
        ),
        "claims_paid": variance_bridge(
            _segments(
                claims,
                plan_claims,
                ["lob", "channel"],
                ("approved_count", "approved_count"),
                ("approved_amount", "approved_amount"),
            ),
            float(claims["paid_amount"].sum()) - float(claims["returned_amount"].sum()),
            float(plan_claims["paid_amount"].sum()),
        ),
        "lae": variance_bridge(
            _segments(
                claims, plan_claims, ["lob", "channel"], ("payouts", "payouts"), ("lae_amount", "lae_amount")
            ),
            float(claims["lae_amount"].sum()),
            float(plan_claims["lae_amount"].sum()),
        ),
        "opex": variance_bridge([Segment("opex", 1.0, opex, 1.0, plan_opex)], opex, plan_opex),
    }
    out[NET] = combine(out, {k: sign for k, (_, sign) in MEASURES.items()})
    return out


def fiscal_ytd(months: list[str], fiscal_start: int) -> list[str]:
    """Months of the fiscal year that contains the last month in ``months``, up to it."""
    last_year, last_month = map(int, months[-1].split("-"))
    start_year = last_year if last_month >= fiscal_start else last_year - 1
    start = f"{start_year:04d}-{fiscal_start:02d}"
    return [m for m in months if m >= start]


@dataclass(frozen=True)
class FpaRun:
    as_of: str
    open_period: str
    bridges: pl.DataFrame
    totals: pl.DataFrame
    scenario: ScenarioResult


def run_fpa(warehouse: Path, config: FpaConfig, *, build_dbt: bool = True) -> FpaRun:
    if build_dbt:
        run_dbt(warehouse)
    marts = load_marts(warehouse)
    drivers = driver_table(marts)
    periods = marts.periods
    months = periods["month"].to_list()
    closed = periods.filter(pl.col("closed"))["month"].to_list()
    open_period = periods.filter(~pl.col("closed"))["month"].to_list()[-1]
    windows = {m: [m] for m in months} | {"ytd": fiscal_ytd(closed, config.fiscal_year_start_month)}

    bridge_rows, total_rows, variances = [], [], []
    for period, window in windows.items():
        for measure, b in bridges_for(marts, window).items():
            kind = MEASURES[measure][0] if measure in MEASURES else "net"
            level = level_for(measure)
            for component in COMPONENTS:
                bridge_rows.append(
                    {
                        "row_key": f"{period}|{measure}|{component}",
                        "period": period,
                        "measure": measure,
                        "component": component,
                        "cents": b.components[component],
                    }
                )
            variances.append(Variance(period, measure, kind, level, b.actual_cents / 100, b.plan_cents / 100))
            total_rows.append(
                {
                    "row_key": f"{period}|{measure}",
                    "period": period,
                    "measure": measure,
                    "months": len(window),
                    "actual_cents": b.actual_cents,
                    "plan_cents": b.plan_cents,
                    "variance_cents": b.variance_cents,
                    "kind": kind,
                    "level": level,
                }
            )
    statuses = flag(variances, config.materiality, config.one_time_items)
    totals = (
        pl.DataFrame(total_rows)
        .with_columns(
            pl.Series("status", statuses),
            pl.Series("favourable", [v.favourable for v in variances]),
            pl.Series("pct_of_plan", [None if v.pct == float("inf") else round(v.pct, 6) for v in variances]),
            (pl.col("period") == open_period).alias("preliminary"),
        )
        .with_columns(
            # ADR 0001: high = closed and reconciled, medium = open with known gaps, low = missing data.
            pl.when((pl.col("actual_cents") == 0) | (pl.col("plan_cents") == 0))
            .then(pl.lit("low"))
            .when(pl.col("preliminary"))
            .then(pl.lit("medium"))
            .otherwise(pl.lit("high"))
            .alias("confidence"),
            pl.when(pl.col("plan_cents") == 0)
            .then(pl.lit("no plan for this period"))
            .when(pl.col("actual_cents") == 0)
            .then(pl.lit("no actual postings for this period"))
            .when(pl.col("preliminary"))
            .then(pl.lit("open period: late premium and pending payouts are still arriving"))
            .otherwise(pl.lit("closed period reconciled to the payments feed"))
            .alias("confidence_reason"),
        )
    )
    bridges = pl.DataFrame(bridge_rows)

    with connect(warehouse, read_only=True) as conn:
        payouts = conn.execute("select * from fpa.stg_fpa__payouts").pl()
        meta_row = conn.execute("select annual_premium from raw_fpa.world_meta").fetchone()
    as_of = as_of_date(periods)
    scenario = check_to_instant(payouts, as_of, config.scenario_shift_share)
    annual_premium = float(meta_row[0]) if meta_row else 0.0
    meta = pl.DataFrame(
        [
            {
                "row_key": "meta",
                "as_of": str(as_of),
                "open_period": open_period,
                "latest_closed": closed[-1],
                "annual_premium": annual_premium,
                "materiality_band_ok": MATERIALITY_BAND[0] <= annual_premium <= MATERIALITY_BAND[1],
                # The aging edges this run used, so the aging tool reads the same ones (ADR 0001 defaults).
                "aging_buckets_days": list(config.aging_buckets_days),
            }
        ]
    )

    def named(frame: pl.DataFrame, *keys: str) -> pl.DataFrame:
        cols = [pl.col("scenario_name"), *(pl.col(k).cast(pl.String) for k in keys)]
        return frame.with_columns(pl.lit(scenario.name).alias("scenario_name")).with_columns(
            pl.concat_str(cols, separator="|").alias("row_key")
        )

    scenario_tables = {
        "scenario_summary": named(scenario.summary, "metric"),
        "scenario_drivers": named(scenario.drivers, "driver", "segment"),
        "scenario_weekly": named(scenario.weekly, "week"),
    }
    tables = {
        "drivers": drivers.with_columns(
            pl.concat_str(["month", "lob", "driver", "segment"], separator="|").alias("row_key")
        ),
        "bridge": bridges,
        "bridge_totals": totals,
        "run_meta": meta,
    } | scenario_tables
    with connect(warehouse) as conn:
        conn.execute("begin transaction")
        try:
            conn.execute("create schema if not exists fpa")
            for name, frame in tables.items():
                conn.register("_fpa_frame", frame)  # an explicit name, never a lookup of a Python variable
                conn.execute(f"create or replace table fpa.{name} as select * from _fpa_frame")  # noqa: S608
                conn.unregister("_fpa_frame")
            conn.execute(
                "create table if not exists fpa.plan_versions (version_id varchar primary key, "
                "scenario varchar, approval_id varchar, drivers_json varchar, created_at timestamp)"
            )
        except Exception:
            conn.execute("rollback")
            raise
        conn.execute("commit")
    return FpaRun(
        as_of=str(as_of), open_period=open_period, bridges=bridges, totals=totals, scenario=scenario
    )
