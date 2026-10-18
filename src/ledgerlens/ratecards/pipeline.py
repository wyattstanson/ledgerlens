"""Run the rate-card steps against the warehouse and persist every result in schema ``ratecards``.

Order: title mapping (keeps earlier human decisions) -> normalization -> leakage -> benchmarks ->
scorecard -> panel optimization -> savings bridge -> tail plan. Dagster assets and the CLI both call
``run_ratecards``; tools only read the tables it writes.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import polars as pl

from ledgerlens.core.approvals import ApprovalQueue
from ledgerlens.core.db import connect, utcnow
from ledgerlens.core.lineage import new_id
from ledgerlens.ratecards.benchmark import benchmarks, scorecard
from ledgerlens.ratecards.bridge import build_bridge, harmonized_prices
from ledgerlens.ratecards.config import RateCardsConfig
from ledgerlens.ratecards.drift import find_leakage
from ledgerlens.ratecards.mapping import (
    FuzzyPicker,
    TitlePicker,
    map_titles,
    queue_reviews,
    save_title_map,
)
from ledgerlens.ratecards.normalize import normalize_cards, normalize_lines
from ledgerlens.ratecards.optimizer import InfeasiblePanelError, PanelProblem, PanelSolution, solve_panel
from ledgerlens.ratecards.tail import tail_plan

PIPELINE_AGENT_ID = "pipeline.ratecards"
PANEL_TABLES = ("panel_allocation", "savings_bridge", "savings_lines", "tail_plan", "tail_briefs")


@dataclass(frozen=True)
class RawWorld:
    vendors: pl.DataFrame
    roles: pl.DataFrame
    rate_cards: pl.DataFrame
    fx: pl.DataFrame
    invoice_lines: pl.DataFrame
    as_of_month: str


def load_raw(db_path: Path) -> RawWorld:
    with connect(db_path) as conn:
        frames = {
            name: conn.execute(f"select * from raw_ratecards.{name}").pl()  # noqa: S608 (fixed table names)
            for name in RawWorld.__annotations__
            if name != "as_of_month"
        }
        row = conn.execute("select as_of_month from raw_ratecards.world_meta").fetchone()
    if row is None:
        raise RuntimeError("raw_ratecards.world_meta is empty; run `ledgerlens generate` first")
    return RawWorld(as_of_month=row[0], **frames)


def _existing_title_map(db_path: Path) -> pl.DataFrame | None:
    with connect(db_path) as conn:
        exists = conn.execute(
            "select count(*) from information_schema.tables "
            "where table_schema = 'ratecards' and table_name = 'title_map'"
        ).fetchone()
        if not exists or exists[0] == 0:
            return None
        return conn.execute("select * from ratecards.title_map").pl()


def update_title_map(
    db_path: Path, raw: RawWorld, picker: TitlePicker, config: RateCardsConfig, approvals: ApprovalQueue
) -> pl.DataFrame:
    """Map titles not seen before; keep every earlier decision (auto, reviewed or still pending)."""
    titles = set(raw.invoice_lines["billed_title"].to_list()) | set(raw.rate_cards["vendor_title"].to_list())
    existing = _existing_title_map(db_path)
    known = set(existing["billed_title"].to_list()) if existing is not None else set()
    fresh = map_titles(sorted(titles - known), raw.roles, picker, config.mapping)
    if fresh.height:
        run_id = new_id("run")
        with connect(db_path) as conn:
            conn.execute(
                "insert into audit.agent_runs values (?, ?, 'pipeline', ?, ?, 'completed')",
                [run_id, PIPELINE_AGENT_ID, utcnow(), utcnow()],
            )
        fresh = queue_reviews(fresh, approvals, agent_run_id=run_id)
    combined = fresh if existing is None else pl.concat([existing, fresh], how="vertical_relaxed")
    save_title_map(db_path, combined.sort("billed_title"))
    return combined


ROLE_KEY = pl.concat_str([pl.col("role_id"), pl.col("location_tier")], separator="@").alias("role_key")


def build_problem(
    baseline_lines: pl.DataFrame, cards: pl.DataFrame, vendors: pl.DataFrame, config: RateCardsConfig
) -> PanelProblem:
    """Demand is per role *and* location tier ("backend.senior@nearshore"): onshore work does not move
    offshore just because offshore is cheaper. That is a sourcing decision, not a rate-card one."""
    usable = baseline_lines.filter(pl.col("role_id").is_not_null() & pl.col("baseline_hours").is_not_null())
    priced = cards.filter(pl.col("role_id").is_not_null() & pl.col("effective_usd_per_hour").is_not_null())
    price = {
        (v, k): float(p)
        for v, k, p in priced.select("vendor_id", ROLE_KEY, "effective_usd_per_hour").iter_rows()
    }
    suppliable = {k for _, k in price}
    by_role = (
        usable.with_columns(ROLE_KEY)
        .filter(pl.col("role_key").is_in(list(suppliable)))
        .group_by("role_key")
        .agg(pl.col("baseline_hours").sum().alias("hours"), pl.col("amount").sum().alias("spend"))
    )
    demand = {k: float(h) for k, h in by_role.select("role_key", "hours").iter_rows() if h > 0}
    top = by_role.sort("spend", descending=True).head(config.optimizer.critical_top_k)["role_key"].to_list()
    return PanelProblem(
        vendors=tuple(vendors["vendor_id"].to_list()),
        quality=dict(vendors.select("vendor_id", "quality").iter_rows()),
        demand=demand,
        price=price,
        panel_size=config.optimizer.panel_size,
        critical=frozenset(top),
        max_share_critical=config.optimizer.max_share_critical,
        min_quality=config.optimizer.min_quality,
    )


def _write(db_path: Path, tables: dict[str, pl.DataFrame]) -> None:
    with connect(db_path) as conn:
        conn.execute("create schema if not exists ratecards")
        for name, frame in tables.items():  # noqa: B007 (frame is read by DuckDB's replacement scan)
            conn.execute(f"create or replace table ratecards.{name} as select * from frame")  # noqa: S608


def run_ratecards(
    db_path: Path,
    config: RateCardsConfig,
    *,
    picker: TitlePicker | None = None,
    external_benchmarks: Path | None = None,
) -> PanelSolution | None:
    """Returns the panel, or None when it is infeasible (the reason is stored in ratecards.panel_meta)."""
    approvals = ApprovalQueue(db_path)
    raw = load_raw(db_path)
    title_map = update_title_map(db_path, raw, picker or FuzzyPicker(config.mapping), config, approvals)

    cards = normalize_cards(raw.rate_cards, title_map, raw.fx, raw.as_of_month, config.normalize)
    lines = normalize_lines(raw.invoice_lines, cards, title_map, config.normalize)
    leakage = find_leakage(lines, config.drift.threshold)
    bench = benchmarks(cards, external_benchmarks)
    months = sorted(raw.invoice_lines["month"].unique().to_list())[-config.optimizer.baseline_months :]
    baseline = lines.filter(pl.col("month").is_in(months)).sort("line_id")
    score = scorecard(
        cards,
        lines,
        raw.vendors,
        bench,
        total_roles=raw.roles.height,
        baseline_months=months,
        weights=config.scorecard.weights,
    )

    base_tables = {
        "cards_normalized": cards,
        "lines_normalized": lines,
        "leakage": leakage,
        "benchmarks": bench,
        "scorecard": score,
    }
    dup_ids = leakage.filter(pl.col("kind") == "duplicate")["line_id"].to_list()
    problem = build_problem(baseline.filter(~pl.col("line_id").is_in(dup_ids)), cards, raw.vendors, config)
    meta_common = {
        "as_of_month": raw.as_of_month,
        "baseline_months": json.dumps(months),
        "panel_size_limit": problem.panel_size,
        "critical_roles": json.dumps(sorted(problem.critical)),
        "pending_reviews": title_map.filter(pl.col("status") == "pending_review").height,
    }
    try:
        solution = solve_panel(problem, config.optimizer.relax_order)
    except InfeasiblePanelError as exc:
        # Recorded, never silent: the panel tools report the status and reason instead of stale results.
        _write(
            db_path,
            base_tables
            | {"panel_meta": pl.DataFrame([meta_common | {"status": "infeasible", "message": str(exc)}])},
        )
        with connect(db_path) as conn:
            for name in PANEL_TABLES:
                conn.execute(f"drop table if exists ratecards.{name}")
        return None
    internal = bench.filter(pl.col("source") == "internal")
    medians = {k: float(m) for k, m in internal.select(ROLE_KEY, "median").iter_rows()}
    harmonized = harmonized_prices(problem, solution, medians)
    savings = build_bridge(
        baseline.with_columns(ROLE_KEY),
        leakage.filter(pl.col("month").is_in(months)),
        problem,
        solution,
        harmonized,
    )
    tail = tail_plan(score, cards, bench, kept=set(solution.kept), min_quality=config.optimizer.min_quality)

    allocation = pl.DataFrame(
        [
            {
                "vendor_id": v,
                "role_key": k,
                "hours": h,
                "price_usd_per_hour": problem.price[(v, k)],
                "harmonized_usd_per_hour": harmonized[(v, k)],
            }
            for (v, k), h in sorted(solution.hours.items())
        ]
    )
    bridge_steps = pl.DataFrame(
        [{"step": "current_cost", "cents": savings.current_cents}]
        + [{"step": s, "cents": c} for s, c in savings.steps.items()]
        + [{"step": "new_cost", "cents": savings.new_cents}]
    )
    status = "optimal" if solution.relaxed is None else f"optimal after relaxing {solution.relaxed}"
    meta = pl.DataFrame(
        [
            meta_common
            | {
                "status": status,
                "message": None,
                "kept_vendors": json.dumps(list(solution.kept)),
                "relaxed_constraint": solution.relaxed,
                "optimizer_cost_usd": solution.cost,
                "optimizer_check_cents": savings.optimizer_cost_check_cents,
            }
        ]
    )
    _write(
        db_path,
        base_tables
        | {
            "panel_meta": meta,
            "panel_allocation": allocation,
            "savings_bridge": bridge_steps,
            "savings_lines": savings.lines,
            "tail_plan": tail.drop("briefs"),
            "tail_briefs": tail.select("vendor_id", "briefs").explode("briefs").drop_nulls().unnest("briefs"),
        },
    )
    return solution
