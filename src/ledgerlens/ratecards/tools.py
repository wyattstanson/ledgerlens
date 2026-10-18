"""Governed rate-card tools. They read the tables ``run_ratecards`` writes and return numbers with lineage.

Each number's lineage stores the exact SQL and the row ids (card ids, line ids or vendor ids) behind it.
"""

import json
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from ledgerlens.core.db import connect, utcnow
from ledgerlens.core.tools import Tool, ToolContext, ToolResult

Tier = Literal["onshore", "nearshore", "offshore"]


class _In(BaseModel):
    model_config = ConfigDict(extra="forbid")


def _rows(ctx: ToolContext, sql: str, params: list[Any] | None = None) -> list[dict[str, Any]]:
    with connect(ctx.db_path) as conn:
        cursor = conn.execute(sql, params or [])
        columns = [d[0] for d in cursor.description]
        return [dict(zip(columns, row, strict=True)) for row in cursor.fetchall()]


def _panel_status(ctx: ToolContext) -> dict[str, Any]:
    rows = _rows(ctx, "select * from ratecards.panel_meta")
    if not rows:
        raise RuntimeError("the rate-card pipeline has not run yet")
    return rows[0]


# --- rc.benchmark -----------------------------------------------------------------------------------


class BenchmarkIn(_In):
    role_id: str = Field(description="Standard role id, e.g. 'backend.senior'")
    location_tier: Tier


BENCHMARK_SQL = """
select b.p25, b.median, b.p75, b.n, b.source, list(c.card_id order by c.card_id) as card_ids
from ratecards.benchmarks b
join ratecards.cards_normalized c
  on c.role_id = b.role_id and c.location_tier = b.location_tier and c.effective_usd_per_hour is not null
where b.role_id = ? and b.location_tier = ? and b.source = 'internal'
group by all
"""


def benchmark(params: BenchmarkIn, ctx: ToolContext) -> ToolResult:
    rows = _rows(ctx, BENCHMARK_SQL, [params.role_id, params.location_tier])
    if not rows:
        return ToolResult(
            data={"found": False, "role_id": params.role_id, "location_tier": params.location_tier}
        )
    row = rows[0]
    key = f"{params.role_id}@{params.location_tier}"
    numbers = [
        ctx.number(
            label=f"{stat} USD/hour {key}",
            value=float(row[stat]),
            unit="USD/hour",
            sql_text=BENCHMARK_SQL,
            source_tables=["ratecards.benchmarks", "ratecards.cards_normalized"],
            row_ids=row["card_ids"],
        )
        for stat in ("p25", "median", "p75")
    ]
    numbers.append(
        ctx.number(
            label=f"vendors in benchmark {key}",
            value=float(row["n"]),
            unit="count",
            sql_text=BENCHMARK_SQL,
            source_tables=["ratecards.benchmarks"],
            row_ids=row["card_ids"],
        )
    )
    return ToolResult(data={"found": True, "source": row["source"], "role_key": key}, numbers=numbers)


# --- rc.vendor_rates --------------------------------------------------------------------------------


class VendorRatesIn(_In):
    role_id: str
    location_tier: Tier


VENDOR_RATES_SQL = """
select card_id, vendor_id, pricing_model, productivity_commitment,
       sticker_usd_per_hour, effective_usd_per_hour
from ratecards.cards_normalized
where role_id = ? and location_tier = ? and effective_usd_per_hour is not null
order by effective_usd_per_hour
"""


def vendor_rates(params: VendorRatesIn, ctx: ToolContext) -> ToolResult:
    rows = _rows(ctx, VENDOR_RATES_SQL, [params.role_id, params.location_tier])
    numbers = []
    for row in rows:
        for column, kind in (("sticker_usd_per_hour", "sticker"), ("effective_usd_per_hour", "effective")):
            numbers.append(
                ctx.number(
                    label=f"{kind} USD/hour {row['vendor_id']} {params.role_id}",
                    value=float(row[column]),
                    unit="USD/hour",
                    sql_text=VENDOR_RATES_SQL,
                    source_tables=["ratecards.cards_normalized"],
                    row_ids=[row["card_id"]],
                )
            )
    ranking = [
        {"vendor_id": r["vendor_id"], "pricing_model": r["pricing_model"], "rank_by_total_cost": i + 1}
        for i, r in enumerate(rows)
    ]
    return ToolResult(data={"vendors": ranking}, numbers=numbers)


# --- rc.leakage -------------------------------------------------------------------------------------


class LeakageIn(_In):
    vendor_id: str | None = Field(default=None, description="Omit for all vendors")


LEAKAGE_SQL = """
select kind, count(*) as line_count, sum(leakage_usd) as leakage_usd,
       list(line_id order by line_id) as line_ids
from ratecards.leakage
where ?::varchar is null or vendor_id = ?
group by kind
"""


def leakage(params: LeakageIn, ctx: ToolContext) -> ToolResult:
    rows = {r["kind"]: r for r in _rows(ctx, LEAKAGE_SQL, [params.vendor_id, params.vendor_id])}
    scope = params.vendor_id or "all vendors"
    numbers = []
    for kind in ("overbill", "duplicate"):
        row = rows.get(kind, {"line_count": 0, "leakage_usd": 0.0, "line_ids": []})
        numbers += [
            ctx.number(
                label=f"{kind} USD {scope}",
                value=round(float(row["leakage_usd"] or 0.0), 2),
                unit="USD",
                sql_text=LEAKAGE_SQL,
                source_tables=["ratecards.leakage"],
                row_ids=row["line_ids"],
            ),
            ctx.number(
                label=f"{kind} lines {scope}",
                value=float(row["line_count"]),
                unit="count",
                sql_text=LEAKAGE_SQL,
                source_tables=["ratecards.leakage"],
                row_ids=row["line_ids"],
            ),
        ]
    return ToolResult(
        data={"scope": scope, "threshold_rule": "billed more than 2% above card"}, numbers=numbers
    )


# --- rc.optimize_panel ------------------------------------------------------------------------------

PANEL_SQL = "select vendor_id, role_key, hours, harmonized_usd_per_hour from ratecards.panel_allocation"


def optimize_panel(params: _In, ctx: ToolContext) -> ToolResult:
    meta = _panel_status(ctx)
    if meta["status"] == "infeasible":
        return ToolResult(
            data={
                "status": "infeasible",
                "message": meta["message"],
                "pending_reviews": meta["pending_reviews"],
            }
        )
    kept = json.loads(meta["kept_vendors"])
    allocation = _rows(ctx, PANEL_SQL)
    keys = [f"{a['vendor_id']}|{a['role_key']}" for a in allocation]
    numbers = [
        ctx.number(
            label="panel vendors",
            value=float(len(kept)),
            unit="count",
            sql_text="select kept_vendors from ratecards.panel_meta",
            source_tables=["ratecards.panel_meta"],
            row_ids=kept,
        ),
        ctx.number(
            label="panel annual cost at contracted rates USD",
            value=round(float(meta["optimizer_cost_usd"]), 2),
            unit="USD",
            sql_text="select optimizer_cost_usd from ratecards.panel_meta",
            source_tables=["ratecards.panel_meta", "ratecards.panel_allocation"],
            row_ids=keys,
        ),
    ]
    return ToolResult(
        data={
            "status": meta["status"],
            "kept_vendors": kept,
            "critical_roles": json.loads(meta["critical_roles"]),
            "relaxed_constraint": meta["relaxed_constraint"],
            "panel_size_limit": meta["panel_size_limit"],
        },
        numbers=numbers,
    )


# --- rc.savings_bridge ------------------------------------------------------------------------------

BRIDGE_SQL = """
select b.step, b.cents,
       coalesce(list(l.line_id order by l.line_id) filter (where l.line_id is not null), []) as line_ids
from ratecards.savings_bridge b left join ratecards.savings_lines l using (step)
group by b.step, b.cents
"""
STEP_ORDER = (
    "current_cost",
    "leakage_recovery",
    "rate_harmonization",
    "volume_shift",
    "vendor_exits",
    "new_cost",
)


def savings_bridge(params: _In, ctx: ToolContext) -> ToolResult:
    meta = _panel_status(ctx)
    if meta["status"] == "infeasible":
        return ToolResult(data={"status": "infeasible", "message": meta["message"]})
    rows = {r["step"]: r for r in _rows(ctx, BRIDGE_SQL)}
    numbers = [
        ctx.number(
            label=f"{step.replace('_', ' ')} USD",
            value=rows[step]["cents"] / 100,
            unit="USD",
            sql_text=BRIDGE_SQL,
            source_tables=["ratecards.savings_bridge", "ratecards.savings_lines"],
            row_ids=rows[step]["line_ids"],
        )
        for step in STEP_ORDER
    ]
    return ToolResult(
        data={"baseline_months": json.loads(meta["baseline_months"]), "steps": list(STEP_ORDER)},
        numbers=numbers,
    )


# --- rc.vendor_scorecard ----------------------------------------------------------------------------


class ScorecardIn(_In):
    vendor_id: str


SCORECARD_SQL = "select * from ratecards.scorecard where vendor_id = ?"


def vendor_scorecard(params: ScorecardIn, ctx: ToolContext) -> ToolResult:
    rows = _rows(ctx, SCORECARD_SQL, [params.vendor_id])
    if not rows:
        return ToolResult(data={"found": False, "vendor_id": params.vendor_id})
    row = rows[0]
    fields = (
        ("score", "score"),
        ("rate_index", "ratio"),
        ("quality", "score"),
        ("sla", "ratio"),
        ("coverage", "ratio"),
        ("spend_share", "ratio"),
        ("baseline_spend_usd", "USD"),
    )
    numbers = [
        ctx.number(
            label=f"{name} {params.vendor_id}",
            value=float(row[name]),
            unit=unit,
            sql_text=SCORECARD_SQL,
            source_tables=["ratecards.scorecard"],
            row_ids=[params.vendor_id],
        )
        for name, unit in fields
        if row[name] is not None
    ]
    return ToolResult(
        data={"found": True, "vendor_id": params.vendor_id, "name": row["name"]}, numbers=numbers
    )


# --- rc.mapping_status ------------------------------------------------------------------------------

MAPPING_SQL = """
select status, count(*) as n, list(billed_title order by billed_title) as titles
from ratecards.title_map group by status
"""


def mapping_status(params: _In, ctx: ToolContext) -> ToolResult:
    rows = {r["status"]: r for r in _rows(ctx, MAPPING_SQL)}
    numbers = [
        ctx.number(
            label=f"titles {status.replace('_', ' ')}",
            value=float(rows.get(status, {"n": 0})["n"]),
            unit="count",
            sql_text=MAPPING_SQL,
            source_tables=["ratecards.title_map"],
            row_ids=rows.get(status, {"titles": []})["titles"],
        )
        for status in ("accepted", "pending_review", "reviewed")
    ]
    return ToolResult(data={"statuses": ["accepted", "pending_review", "reviewed"]}, numbers=numbers)


# --- rc.tail_plan -----------------------------------------------------------------------------------


class TailIn(_In):
    vendor_id: str


TAIL_SQL = "select * from ratecards.tail_plan where vendor_id = ?"
BRIEF_SQL = "select * from ratecards.tail_briefs where vendor_id = ? order by role_id"


def tail(params: TailIn, ctx: ToolContext) -> ToolResult:
    meta = _panel_status(ctx)
    if meta["status"] == "infeasible":
        return ToolResult(data={"status": "infeasible", "message": meta["message"]})
    plan = _rows(ctx, TAIL_SQL, [params.vendor_id])
    if not plan:
        return ToolResult(
            data={"found": False, "vendor_id": params.vendor_id, "note": "vendor is on the panel"}
        )
    numbers = [
        ctx.number(
            label=f"baseline spend {params.vendor_id}",
            value=float(plan[0]["baseline_spend_usd"]),
            unit="USD",
            sql_text=TAIL_SQL,
            source_tables=["ratecards.tail_plan"],
            row_ids=[params.vendor_id],
        )
    ]
    briefs = _rows(ctx, BRIEF_SQL, [params.vendor_id])
    for brief in briefs:
        for column, kind in (
            ("effective_usd_per_hour", "current"),
            ("target_usd_per_hour", "target"),
            ("walk_away_usd_per_hour", "walk-away"),
        ):
            numbers.append(
                ctx.number(
                    label=f"{kind} USD/hour {params.vendor_id} {brief['role_id']}",
                    value=float(brief[column]),
                    unit="USD/hour",
                    sql_text=BRIEF_SQL,
                    source_tables=["ratecards.tail_briefs"],
                    row_ids=[brief["card_id"]],
                )
            )
    return ToolResult(data={"vendor_id": params.vendor_id, "proposal": plan[0]["proposal"]}, numbers=numbers)


# --- rc.accept_panel (write) ------------------------------------------------------------------------


class AcceptPanelIn(_In):
    approval_id: str = Field(description="An approved rc.accept_panel approval record")


def accept_panel(params: AcceptPanelIn, ctx: ToolContext) -> ToolResult:
    """Record the panel as accepted. Runs only against an approved approval (enforced by the agent loop)."""
    meta = _panel_status(ctx)
    if meta["status"] == "infeasible":
        raise RuntimeError(
            "there is no feasible panel to accept; resolve the reason in rc.optimize_panel first"
        )
    with connect(ctx.db_path) as conn:
        conn.execute(
            "create table if not exists ratecards.panel_decisions "
            "(approval_id varchar primary key, kept_vendors varchar, as_of_month varchar, "
            "accepted_at timestamp)"
        )
        conn.execute(
            "insert into ratecards.panel_decisions values (?, ?, ?, ?)",
            [params.approval_id, meta["kept_vendors"], meta["as_of_month"], utcnow()],
        )
    return ToolResult(data={"accepted": True, "kept_vendors": json.loads(meta["kept_vendors"])})


RATECARD_TOOLS: tuple[Tool, ...] = (
    Tool(
        "rc.benchmark",
        "p25, median and p75 effective USD per baseline hour for a role and location tier.",
        BenchmarkIn,
        benchmark,
    ),
    Tool(
        "rc.vendor_rates",
        "Each vendor's sticker and total-cost (effective) hourly rate for a role and tier, ranked.",
        VendorRatesIn,
        vendor_rates,
    ),
    Tool(
        "rc.leakage",
        "Overbilled excess and duplicate invoice lines, in USD and line counts, traced to line ids.",
        LeakageIn,
        leakage,
    ),
    Tool(
        "rc.optimize_panel",
        "The optimized vendor panel: status, kept vendors, critical roles and annual cost.",
        _In,
        optimize_panel,
    ),
    Tool(
        "rc.savings_bridge",
        "Savings bridge from current to new cost, each step traced to invoice line ids.",
        _In,
        savings_bridge,
    ),
    Tool(
        "rc.vendor_scorecard",
        "A vendor's scorecard: rate index, quality, SLA, coverage, spend share and score.",
        ScorecardIn,
        vendor_scorecard,
    ),
    Tool(
        "rc.mapping_status",
        "How many billed titles are auto-mapped, pending human review, or reviewed.",
        _In,
        mapping_status,
    ),
    Tool(
        "rc.tail_plan",
        "Exit or negotiation plan for a vendor not on the panel, with target and walk-away rates.",
        TailIn,
        tail,
    ),
    Tool(
        "rc.accept_panel",
        "Record the proposed panel as accepted. Needs an approved approval record.",
        AcceptPanelIn,
        accept_panel,
        writes=True,
    ),
)
