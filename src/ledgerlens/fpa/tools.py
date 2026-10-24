"""Governed FP&A, payments and ledger tools. They read what ``run_fpa`` and the dbt marts wrote.

Every number cites its rows: driver values cite their ``fpa.drivers`` row and the dbt mart rows behind
it, bridge numbers cite ``fpa.bridge`` rows, and ledger and payments numbers cite whatever query their
adapter ran. FP&A outputs carry a scope (period, entity, currency, book, preliminary or final) and a "not
financial advice" note (ADR 0001). A period that is not in the ledger gets ``found: false``, never a number.
Saving a scenario as a plan version is the one write, and it needs a human approval of exactly the scenario
that is current when it runs. All reads use read-only connections with external access off.
"""

import json
from datetime import date
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from ledgerlens.core.approvals import ApprovalError, ApprovalQueue
from ledgerlens.core.db import NO_EXTERNAL_ACCESS, connect, fetch_dicts, utcnow
from ledgerlens.core.lineage import new_id, sha256_text
from ledgerlens.core.tools import Tool, ToolContext, ToolResult
from ledgerlens.fpa.config import Line
from ledgerlens.fpa.ledger_adapter import SavedSearch, ledger
from ledgerlens.fpa.materiality import risk_label
from ledgerlens.fpa.payments_adapter import MAX_EVENTS, EventKind, payments_feed
from ledgerlens.fpa.source import Rows

NOT_ADVICE = "Not financial advice: synthetic data, computed by LedgerLens to illustrate the analysis."
Measure = Literal["collected_premium", "claims_paid", "lae", "opex", "net_operating_cash"]
MONTH = r"^[0-9]{4}-(0[1-9]|1[0-2])$"  # [0-9], not \d: \d also matches other scripts' digits
PERIOD = r"^([0-9]{4}-(0[1-9]|1[0-2])|ytd)$"
RECENT_MONTHS = 6  # saved searches cite numbers for the latest months only


class _In(BaseModel):
    model_config = ConfigDict(extra="forbid")


def _rows(ctx: ToolContext | Path, sql: str, params: list[Any] | None = None) -> list[dict[str, Any]]:
    db_path = ctx if isinstance(ctx, Path) else ctx.db_path
    with connect(db_path, read_only=True, config=NO_EXTERNAL_ACCESS) as conn:
        return fetch_dicts(conn, sql, params)


def _meta(ctx: ToolContext) -> dict[str, Any]:
    rows = _rows(ctx, "select as_of, open_period, latest_closed, aging_buckets_days from fpa.run_meta")
    if not rows:
        raise RuntimeError("the FP&A pipeline has not run yet")
    return rows[0]


def _scope(ctx: ToolContext, period: str) -> dict[str, Any] | None:
    """ADR 0001 answer scope, or None when the ledger has no such period."""
    known = {r["period"] for r in _rows(ctx, "select distinct period from fpa.bridge_totals")}
    if period not in known:
        return None
    meta = _meta(ctx)
    return {
        "period": period,
        "entity": "LedgerLens Mutual (synthetic), all lines",
        "currency": "USD",
        "book": "primary (cash basis for claims and LAE)",
        "status": "preliminary" if period == meta["open_period"] else "final",
        "note": NOT_ADVICE,
    }


def _not_found(period: str) -> ToolResult:
    return ToolResult(data={"found": False, "period": period, "reason": "no such period in the ledger"})


def _cite(ctx: ToolContext, label: str, value: float, unit: str, source: Rows, row_ids: list[Any]):
    return ctx.number(
        label=label,
        value=float(value),
        unit=unit,
        sql_text=source.sql,
        source_tables=source.tables,
        row_ids=row_ids,
    )


# --- fpa.drivers -------------------------------------------------------------------------------------------


class DriversIn(_In):
    month: str = Field(pattern=MONTH, description="YYYY-MM")
    lob: Line | None = Field(default=None, description="Line of business; omit for every line")


DRIVERS_SQL = """
select row_key, lob, driver, segment, value, unit, source_table, row_keys
from fpa.drivers
where month = ? and (?::varchar is null or lob = ? or lob = 'all')
order by lob, driver, segment
"""


def drivers(params: DriversIn, ctx: ToolContext) -> ToolResult:
    scope = _scope(ctx, params.month)
    if scope is None:
        return _not_found(params.month)
    rows = _rows(ctx, DRIVERS_SQL, [params.month, params.lob, params.lob])
    numbers = []
    for r in rows:
        segment = f" {r['segment']}" if r["segment"] else ""
        numbers.append(
            ctx.number(
                label=f"{r['driver']}{segment} {r['lob']} {params.month}",
                value=float(r["value"]),
                unit=r["unit"],
                sql_text=DRIVERS_SQL,
                source_tables=["fpa.drivers", r["source_table"]],
                row_ids=[r["row_key"], *r["row_keys"]],
            )
        )
    data = [{k: r[k] for k in ("lob", "driver", "segment", "unit")} for r in rows]
    return ToolResult(data={"found": True, "scope": scope, "drivers": data}, numbers=numbers)


# --- fpa.bridge --------------------------------------------------------------------------------------------


class BridgeIn(_In):
    measure: Measure
    period: str = Field(pattern=PERIOD, description="YYYY-MM, or 'ytd' for fiscal year to date")


BRIDGE_TOTAL_SQL = """
select row_key, actual_cents, plan_cents, variance_cents, pct_of_plan, status, favourable, level, confidence,
       confidence_reason
from fpa.bridge_totals where period = ? and measure = ?
"""
BRIDGE_SQL = (
    "select row_key, component, cents from fpa.bridge where period = ? and measure = ? order by component"
)


def bridge(params: BridgeIn, ctx: ToolContext) -> ToolResult:
    scope = _scope(ctx, params.period)
    totals = _rows(ctx, BRIDGE_TOTAL_SQL, [params.period, params.measure]) if scope else []
    if scope is None or not totals:
        return _not_found(params.period)
    t = totals[0]
    tag = f"{params.measure} {params.period}"
    source = Rows([], BRIDGE_TOTAL_SQL, ["fpa.bridge_totals"])
    numbers = [
        _cite(ctx, f"{name} {tag}", t[col] / 100, "USD", source, [t["row_key"]])
        for name, col in (("actual", "actual_cents"), ("plan", "plan_cents"), ("variance", "variance_cents"))
    ]
    if t["pct_of_plan"] is not None:
        numbers.append(
            _cite(ctx, f"variance share of plan {tag}", t["pct_of_plan"], "ratio", source, [t["row_key"]])
        )
    components = Rows([], BRIDGE_SQL, ["fpa.bridge"])
    for r in _rows(ctx, BRIDGE_SQL, [params.period, params.measure]):
        numbers.append(
            _cite(ctx, f"{r['component']} {tag}", r["cents"] / 100, "USD", components, [r["row_key"]])
        )
    data = {
        "found": True,
        "scope": scope,
        "status": t["status"],
        "risk_label": risk_label(t["status"], t["level"]),
        "favourable": t["favourable"],
        "materiality_level": t["level"],
        "confidence": t["confidence"],
        "confidence_reason": t["confidence_reason"],
        "components": ["volume", "mix", "rate", "timing", "residual"],
    }
    return ToolResult(data=data, numbers=numbers)


# --- fpa.variance_flags ------------------------------------------------------------------------------------


class FlagsIn(_In):
    period: str = Field(pattern=PERIOD)


FLAGS_SQL = """
select row_key, measure, variance_cents, status, favourable, level
from fpa.bridge_totals where period = ? order by measure
"""


def variance_flags(params: FlagsIn, ctx: ToolContext) -> ToolResult:
    scope = _scope(ctx, params.period)
    if scope is None:
        return _not_found(params.period)
    rows = _rows(ctx, FLAGS_SQL, [params.period])
    source = Rows([], FLAGS_SQL, ["fpa.bridge_totals"])
    numbers = [
        _cite(
            ctx,
            f"variance {r['measure']} {params.period}",
            r["variance_cents"] / 100,
            "USD",
            source,
            [r["row_key"]],
        )
        for r in rows
    ]
    flags = [
        {
            **{k: r[k] for k in ("measure", "status", "favourable", "level")},
            "label": risk_label(r["status"], r["level"]),
        }
        for r in rows
    ]
    return ToolResult(data={"found": True, "scope": scope, "flags": flags}, numbers=numbers)


# --- fpa.scenario and fpa.save_plan_version ----------------------------------------------------------------

SCENARIO_SQL = (
    "select row_key, scenario_name, metric, unit, baseline, scenario, change from fpa.scenario_summary"
)
SCENARIO_DRIVERS_SQL = "select row_key, driver, segment, baseline, scenario from fpa.scenario_drivers"


def scenario(_params: _In, ctx: ToolContext) -> ToolResult:
    rows = _rows(ctx, SCENARIO_SQL)
    if not rows:
        raise RuntimeError("the FP&A pipeline has not run yet")
    summary = Rows([], SCENARIO_SQL, ["fpa.scenario_summary"])
    numbers = [
        _cite(ctx, f"{column} {r['metric']}", r[column], r["unit"], summary, [r["row_key"]])
        for r in rows
        for column in ("baseline", "scenario", "change")
    ]
    changed = _rows(ctx, SCENARIO_DRIVERS_SQL)
    driver_source = Rows([], SCENARIO_DRIVERS_SQL, ["fpa.scenario_drivers"])
    numbers += [
        _cite(
            ctx, f"{column} {r['driver']} {r['segment']}", r[column], "share", driver_source, [r["row_key"]]
        )
        for r in changed
        for column in ("baseline", "scenario")
    ]
    data = {
        "scenario": rows[0]["scenario_name"],
        "note": NOT_ADVICE,
        "changed_drivers": [{"driver": r["driver"], "segment": r["segment"]} for r in changed],
        "metrics": [r["metric"] for r in rows],
    }
    return ToolResult(data=data, numbers=numbers)


class SavePlanIn(_In):
    approval_id: str | None = Field(
        default=None, description="Omit when proposing; an approved record to execute."
    )


def _state_hash(state: dict[str, Any]) -> str:
    return sha256_text(json.dumps(state, sort_keys=True, default=str))


def scenario_snapshot(db_path: Path) -> dict[str, Any]:
    """What a human approves: the scenario's changed drivers and results, and a hash that moves with them."""
    summary = [
        list(r.values())
        for r in _rows(
            db_path,
            "select metric, round(baseline, 2), round(scenario, 2) from fpa.scenario_summary order by metric",
        )
    ]
    if not summary:
        raise RuntimeError("the FP&A pipeline has not run yet: there is no scenario to save")
    drivers_ = [
        list(r.values())
        for r in _rows(
            db_path,
            "select driver, segment, round(baseline, 6), round(scenario, 6) "
            "from fpa.scenario_drivers order by driver, segment",
        )
    ]
    name = _rows(db_path, "select any_value(scenario_name) as name from fpa.scenario_summary")[0]["name"]
    state = {"scenario": name, "summary": summary, "drivers": drivers_}
    return state | {"scenario_hash": _state_hash(state)}


def save_plan_version(params: SavePlanIn, ctx: ToolContext) -> ToolResult:
    queue = ApprovalQueue(ctx.db_path)
    approval = queue.require_approved(params.approval_id or "", action="fpa.save_plan_version")
    approved = approval.payload.get("snapshot", {})
    shown = {k: approved.get(k) for k in ("scenario", "summary", "drivers")}
    current = scenario_snapshot(ctx.db_path)
    # The reviewer saw `shown`: it must be what the stored hash describes and what is current now.
    if not (_state_hash(shown) == approved.get("scenario_hash") == current["scenario_hash"]):
        raise ApprovalError("the scenario changed after it was approved; propose it again")
    version_id = new_id("plan")
    with connect(ctx.db_path) as conn:
        conn.execute("begin transaction")
        try:
            conn.execute(
                "insert into fpa.plan_versions values (?, ?, ?, ?, ?)",
                [
                    version_id,
                    current["scenario"],
                    approval.approval_id,
                    json.dumps(current["drivers"]),
                    utcnow(),
                ],
            )
            queue.mark_executed(approval.approval_id, conn)
        except Exception:
            conn.execute("rollback")
            raise
        conn.execute("commit")
    return ToolResult(data={"saved": True, "version_id": version_id, "scenario": current["scenario"]})


# --- payments.events and payments.summary ------------------------------------------------------------------


class EventsIn(_In):
    kind: EventKind | None = None
    start: date | None = None
    end: date | None = None
    # No numeric bounds in the schema (strict tool schemas reject them); the feed caps the number it returns.
    limit: int = Field(default=50, description=f"How many events, most recent first; at most {MAX_EVENTS}.")


def payments_events(params: EventsIn, ctx: ToolContext) -> ToolResult:
    feed = payments_feed(ctx.db_path).events(
        kind=params.kind, start=params.start, end=params.end, limit=params.limit
    )
    ids = [e["event_id"] for e in feed.rows]
    numbers = []
    if feed.rows:
        total = round(sum(float(e["amount"]) for e in feed.rows), 2)
        numbers = [
            _cite(ctx, "events returned", len(feed.rows), "count", feed, ids),
            _cite(ctx, "amount of events returned", total, "USD", feed, ids),
        ]
    meta = _meta(ctx)
    data = {
        "events": [
            {k: (str(v) if k == "event_ts" else v) for k, v in e.items() if k != "amount"} for e in feed.rows
        ],
        "order": "most recent first",
        "as_of": meta["as_of"],
        "note": f"Events after {meta['latest_closed']} fall in the open period and are preliminary.",
    }
    return ToolResult(data=data, numbers=numbers)


class SummaryIn(_In):
    month: str = Field(pattern=MONTH)


def payments_summary(params: SummaryIn, ctx: ToolContext) -> ToolResult:
    scope = _scope(ctx, params.month)
    if scope is None:
        return _not_found(params.month)
    feed = payments_feed(ctx.db_path).summary(params.month)
    numbers = []
    for r in feed.rows:
        label = f"{r['kind']} {r['instrument']} {params.month}"
        numbers += [
            _cite(ctx, f"{label} amount", r["amount"], "USD", feed, r["event_ids"]),
            _cite(ctx, f"{label} events", r["events"], "count", feed, r["event_ids"]),
        ]
    groups = [{"kind": r["kind"], "instrument": r["instrument"]} for r in feed.rows]
    return ToolResult(data={"found": True, "scope": scope, "groups": groups}, numbers=numbers)


# --- ledger.* ----------------------------------------------------------------------------------------------


class PeriodIn(_In):
    period: str = Field(pattern=MONTH)


def income_statement(params: PeriodIn, ctx: ToolContext) -> ToolResult:
    scope = _scope(ctx, params.period)
    if scope is None:
        return _not_found(params.period)
    statement = ledger(ctx.db_path).income_statement(params.period)
    numbers = [
        _cite(ctx, f"{line['label']} {params.period}", line["amount"], "USD", statement, line["row_keys"])
        for line in statement.rows
    ]
    data = {
        "found": True,
        "scope": scope,
        "lines": [line["label"] for line in statement.rows],
        "basis": "premium revenue as billed; claims, LAE and operating expense as paid",
    }
    return ToolResult(data=data, numbers=numbers)


def budget_vs_actual(params: PeriodIn, ctx: ToolContext) -> ToolResult:
    scope = _scope(ctx, params.period)
    if scope is None:
        return _not_found(params.period)
    result = ledger(ctx.db_path).budget_vs_actual(params.period)
    numbers = [
        _cite(ctx, f"{name} {r['measure']} {params.period}", r[col] / 100, "USD", result, [r["row_key"]])
        for r in result.rows
        for name, col in (("actual", "actual_cents"), ("plan", "plan_cents"), ("variance", "variance_cents"))
    ]
    measures = [
        {
            **{k: r[k] for k in ("measure", "status", "favourable", "level", "confidence")},
            "risk_label": risk_label(r["status"], r["level"]),
        }
        for r in result.rows
    ]
    return ToolResult(data={"found": True, "scope": scope, "measures": measures}, numbers=numbers)


def ar_aging(_params: _In, ctx: ToolContext) -> ToolResult:
    meta = _meta(ctx)
    edges = [int(edge) for edge in meta["aging_buckets_days"]]  # the edges the pipeline run used
    buckets = (edges[0], edges[1], edges[2])
    aging = ledger(ctx.db_path).ar_aging(buckets)
    numbers = []
    for r in aging.rows:
        numbers += [
            _cite(ctx, f"receivable {r['bucket']} amount", r["amount"], "USD", aging, r["instalment_ids"]),
            _cite(ctx, f"receivable {r['bucket']} items", r["items"], "count", aging, r["instalment_ids"]),
        ]
    data = {
        "as_of": meta["as_of"],
        "status": "preliminary",  # aged to the end of the open period
        "buckets_days": list(buckets),
        "note": NOT_ADVICE,
        "stale_needs_review": any(r["bucket"] == "stale" and r["items"] for r in aging.rows),
    }
    return ToolResult(data=data, numbers=numbers)


class SavedSearchIn(_In):
    name: SavedSearch


def saved_search(params: SavedSearchIn, ctx: ToolContext) -> ToolResult:
    result = ledger(ctx.db_path).saved_search(params.name)
    recent = sorted({r["month"] for r in result.rows})[-RECENT_MONTHS:]
    numbers = []
    for r in result.rows:
        if r["month"] not in recent:
            continue
        key = " ".join(str(v) for v in r.values() if isinstance(v, str))
        for column, value in r.items():
            if column == "row_keys" or isinstance(value, str) or value is None:
                continue
            if column not in result.units:
                raise KeyError(f"saved search {params.name}: column {column!r} has no declared unit")
            numbers.append(_cite(ctx, f"{column} {key}", value, result.units[column], result, r["row_keys"]))
    return ToolResult(data={"name": params.name, "numbers_for_months": recent}, numbers=numbers)


# --- fpa.forecast ------------------------------------------------------------------------------------------

ForecastView = Literal["totals", "drivers", "weekly_cash"]
DRIVER_UNITS = {"due_premium": "USD", "approved_claims": "count", "severity": "USD", "opex": "USD"}
FORECAST_NOTE = (
    "Expected values from the models reviewed in ml_pipeline (Gate C); totals are rebuilt from driver "
    "forecasts through the driver tree and carry no interval."
)


class ForecastIn(_In):
    view: ForecastView = Field(
        default="totals",
        description="totals: monthly cash totals rebuilt from drivers; drivers: the monthly driver "
        "forecasts; weekly_cash: the 13-week cash forecast",
    )
    lob: Line | None = Field(
        default=None, description="For drivers: one line of business; omit for every line"
    )


FORECAST_TABLES_SQL = (
    "select count(*) as n from information_schema.tables "
    "where table_schema = 'fpa' and table_name = 'forecast_meta'"
)
FORECAST_META_SQL = "select origin, last_complete_week from fpa.forecast_meta"
FORECAST_TOTALS_SQL = "select row_key, ds, measure, value from fpa.forecast_totals order by measure, ds"
FORECAST_SQL = """
select row_key, unique_id, ds, model, mean, "lo-80" as lo80, "hi-80" as hi80
from fpa.forecasts
where in_horizon and freq = ? and (?::varchar is null or unique_id = 'opex|all' or unique_id like '%|' || ?)
order by unique_id, ds
"""
FORECAST_MODELS_SQL = (
    "select row_key, unique_id, chosen, cv_mase, test_mase, baseline_test_mase from fpa.forecast_models"
)


def forecast(params: ForecastIn, ctx: ToolContext) -> ToolResult:
    if not _rows(ctx, FORECAST_TABLES_SQL)[0]["n"]:
        return ToolResult(
            data={
                "found": False,
                "reason": "forecasts have not been generated; run "
                "`python -m ledgerlens.cli forecast fpa` on Linux (ADR 0005)",
            }
        )
    meta = _rows(ctx, FORECAST_META_SQL)[0]
    numbers = []
    if params.view == "totals":
        source = Rows([], FORECAST_TOTALS_SQL, ["fpa.forecast_totals"])
        for r in _rows(ctx, FORECAST_TOTALS_SQL):
            label = f"forecast {r['measure']} {r['ds']:%Y-%m}"
            numbers.append(_cite(ctx, label, r["value"], "USD", source, [r["row_key"]]))
        shown: set[str] = set()
    else:
        freq = "weekly" if params.view == "weekly_cash" else "monthly"
        source = Rows([], FORECAST_SQL, ["fpa.forecasts"])
        rows = _rows(ctx, FORECAST_SQL, [freq, params.lob, params.lob])
        shown = {r["unique_id"] for r in rows}
        for r in rows:
            unit = "USD" if freq == "weekly" else DRIVER_UNITS[r["unique_id"].split("|")[0]]
            when = f"{r['ds']:%Y-%m-%d}" if freq == "weekly" else f"{r['ds']:%Y-%m}"
            for name, column in (("forecast", "mean"), ("80% low", "lo80"), ("80% high", "hi80")):
                numbers.append(
                    _cite(ctx, f"{name} {r['unique_id']} {when}", r[column], unit, source, [r["row_key"]])
                )
    models = _rows(ctx, FORECAST_MODELS_SQL)
    quality = Rows([], FORECAST_MODELS_SQL, ["fpa.forecast_models"])
    for m in models:
        if m["unique_id"] not in shown:
            continue
        for name in ("cv_mase", "test_mase", "baseline_test_mase"):
            if m[name] is not None:
                numbers.append(
                    _cite(ctx, f"{name} {m['unique_id']}", m[name], "ratio", quality, [m["row_key"]])
                )
    data = {
        "found": True,
        "view": params.view,
        "origin": meta["origin"],
        "last_complete_week": meta["last_complete_week"],
        "models": {
            m["unique_id"]: m["chosen"] for m in models if params.view == "totals" or m["unique_id"] in shown
        },
        "note": f"{FORECAST_NOTE} {NOT_ADVICE}",
    }
    return ToolResult(data=data, numbers=numbers)


FPA_TOOLS: tuple[Tool, ...] = (
    Tool(
        name="fpa.drivers",
        description="Driver-tree values for a month (premium, collection, claims, LAE, opex), by line.",
        input_model=DriversIn,
        fn=drivers,
    ),
    Tool(
        name="fpa.bridge",
        description="Actual vs plan for a measure and period: volume, mix, rate, timing and residual.",
        input_model=BridgeIn,
        fn=bridge,
    ),
    Tool(
        name="fpa.variance_flags",
        description="ADR 0001 materiality status and risk label of every measure's variance for a period.",
        input_model=FlagsIn,
        fn=variance_flags,
    ),
    Tool(
        name="fpa.scenario",
        description="The built-in scenario (claims moved from check to instant), baseline vs scenario.",
        input_model=_In,
        fn=scenario,
    ),
    Tool(
        name="fpa.save_plan_version",
        description="Propose saving the scenario as a plan version (no arguments); a human approves it.",
        input_model=SavePlanIn,
        fn=save_plan_version,
        writes=True,
        snapshot=scenario_snapshot,
    ),
    Tool(
        name="payments.events",
        description="Payments feed events (premium received, claim paid, payout returned), newest first.",
        input_model=EventsIn,
        fn=payments_events,
    ),
    Tool(
        name="payments.summary",
        description="Payments feed totals for a month by event kind and instrument.",
        input_model=SummaryIn,
        fn=payments_summary,
    ),
    Tool(
        name="ledger.income_statement",
        description="Income statement lines for a period from the NetSuite-shaped ledger.",
        input_model=PeriodIn,
        fn=income_statement,
    ),
    Tool(
        name="ledger.budget_vs_actual",
        description="Budget vs actual for every measure in a period, with materiality and confidence.",
        input_model=PeriodIn,
        fn=budget_vs_actual,
    ),
    Tool(
        name="ledger.ar_aging",
        description="Premium receivables by aging bucket (current, monitor, escalate, stale).",
        input_model=_In,
        fn=ar_aging,
    ),
    Tool(
        name="fpa.forecast",
        description="Forecasts from the reviewed models: monthly cash totals, drivers, or 13-week cash.",
        input_model=ForecastIn,
        fn=forecast,
    ),
    Tool(
        name="ledger.saved_search",
        description="A named ledger search: premium by method, claims by channel, or cash by month.",
        input_model=SavedSearchIn,
        fn=saved_search,
    ),
)
