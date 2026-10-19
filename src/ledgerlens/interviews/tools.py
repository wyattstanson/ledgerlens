"""Governed interview tools (``iv.*``). Aggregates only, read from the warehouse, never from the vault.

Every tool carries ``aggregate_only`` as its output guard: a result is blocked if it contains a person-level
key, a vault id, or a cell below the minimum group size that is not marked suppressed (rule 2, ADR 0004).
"""

import numbers
import re
import shutil
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from ledgerlens.core.approvals import ApprovalError, ApprovalQueue
from ledgerlens.core.db import connect, utcnow
from ledgerlens.core.guards import MIN_GROUP_N
from ledgerlens.core.tools import Tool, ToolContext, ToolResult

Stage = Literal["screen", "technical", "onsite", "hiring_manager"]
Dimension = Literal["sex", "race_ethnicity"]
ProcessMetric = Literal["rubric_coverage", "evidence_quality", "loop_coverage"]

FORBIDDEN_KEYS = frozenset(
    {
        "candidate_id",
        "candidate",
        "candidate_name",
        "name",
        "email",
        "note_text",
        "text",
        "rating_id",
        "interview_id",
        "note_id",
        "performance_rating",
        "candidates",
        "candidate_ids",
        "rank",
        "ranking",
        "score",
        "scores",
    }
)
VAULT_ID = re.compile(r"\b(CAND|IV|RT|NT)-\d+", re.IGNORECASE)


class OutputPolicyError(ValueError):
    pass


def aggregate_only(result: ToolResult) -> None:
    """Refuse person-level content and unsuppressed small cells anywhere in a result."""

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            bad = FORBIDDEN_KEYS.intersection(str(k).lower() for k in node)
            if bad:
                raise OutputPolicyError(f"person-level keys {sorted(bad)}")
            n = node.get("n")
            small = isinstance(n, numbers.Real) and not isinstance(n, bool) and n < MIN_GROUP_N
            if small and not node.get("suppressed"):
                raise OutputPolicyError(f"cell with n={n} below {MIN_GROUP_N} is not suppressed")
            for key, value in node.items():
                walk(str(key))
                walk(value)
        elif isinstance(node, list | tuple):
            for value in node:
                walk(value)
        elif isinstance(node, str) and VAULT_ID.search(node):
            raise OutputPolicyError("vault identifier in output")

    walk(result.data)
    for number in result.numbers:
        if VAULT_ID.search(number.label):
            raise OutputPolicyError("vault identifier in a number label")


class _In(BaseModel):
    model_config = ConfigDict(extra="forbid")


def _rows(ctx: ToolContext, sql: str, params: list[Any] | None = None) -> list[dict[str, Any]]:
    with connect(ctx.db_path) as conn:
        cursor = conn.execute(sql, params or [])
        columns = [d[0] for d in cursor.description]
        return [dict(zip(columns, row, strict=True)) for row in cursor.fetchall()]


def _num(ctx: ToolContext, label: str, value: float | None, unit: str, sql: str, table: str, keys: list[str]):
    if value is None:
        return None
    return ctx.number(
        label=label,
        value=float(value),
        unit=unit,
        sql_text=sql,
        source_tables=[f"interviews.{table}"],
        row_ids=keys,
    )


# --- iv.plan_status -------------------------------------------------------------------------------

PLAN_SQL = """
select m.*, r.plan_name, r.plan_version, r.plan_sha256, r.status
from interviews.run_meta m join audit.analysis_runs r using (analysis_run_id)
"""


def plan_status(params: _In, ctx: ToolContext) -> ToolResult:
    rows = _rows(ctx, PLAN_SQL)
    if not rows:
        raise RuntimeError("the interviews pipeline has not run yet")
    m = rows[0]
    numbers = [
        _num(ctx, label, m[col], "count", PLAN_SQL, "run_meta", ["meta"])
        for label, col in (
            ("tests in BH family", "tests_in_family"),
            ("flagged interviewers", "flagged_interviewers"),
            ("flagged panels", "flagged_panels"),
            ("flagged fairness cells", "flagged_fairness_cells"),
            ("non-predictive roles", "nonpredictive_roles"),
            ("process flags", "process_flags"),
        )
    ]
    data = {
        "plan": m["plan_name"],
        "plan_version": m["plan_version"],
        "plan_sha256": m["plan_sha256"],
        "run_status": m["status"],
        "tagger": m["tagger"],
    }
    return ToolResult(data=data, numbers=[n for n in numbers if n])


# --- iv.calibration --------------------------------------------------------------------------------


class CalibrationIn(_In):
    interviewer_id: str | None = Field(
        default=None, description="e.g. INT-07; omit for flagged interviewers only"
    )


CAL_SQL = """
select row_key, interviewer_id, role_id, competency_id, n, z, d, d_ci_low, d_ci_high, q, flagged, suppressed
from interviews.calibration
where (?::varchar is null and flagged) or interviewer_id = ?
order by interviewer_id, role_id, competency_id
"""


def calibration(params: CalibrationIn, ctx: ToolContext) -> ToolResult:
    rows = _rows(ctx, CAL_SQL, [params.interviewer_id, params.interviewer_id])
    numbers = []
    for r in rows:
        tag = f"{r['interviewer_id']} {r['role_id']} {r['competency_id']}"
        numbers += [
            _num(ctx, f"calibration z {tag}", r["z"], "z", CAL_SQL, "calibration", [r["row_key"]]),
            _num(ctx, f"Cohen's d {tag}", r["d"], "d", CAL_SQL, "calibration", [r["row_key"]]),
        ]
    cells = [
        {
            k: r[k]
            for k in (
                "interviewer_id",
                "role_id",
                "competency_id",
                "n",
                "d_ci_low",
                "d_ci_high",
                "q",
                "flagged",
                "suppressed",
            )
        }
        for r in rows
    ]
    return ToolResult(data={"cells": cells}, numbers=[n for n in numbers if n])


# --- iv.interviewer_flags --------------------------------------------------------------------------

IVF_SQL = (
    "select row_key, interviewer_id, role_id, cells, flagged_cells "
    "from interviews.interviewer_flags where flagged"
)


def interviewer_flags(params: _In, ctx: ToolContext) -> ToolResult:
    rows = _rows(ctx, IVF_SQL)
    numbers = [
        _num(
            ctx,
            f"flagged cells {r['interviewer_id']} {r['role_id']}",
            r["flagged_cells"],
            "count",
            IVF_SQL,
            "interviewer_flags",
            [r["row_key"]],
        )
        for r in rows
    ]
    data = [
        {"interviewer_id": r["interviewer_id"], "role_id": r["role_id"], "cells": r["cells"]} for r in rows
    ]
    return ToolResult(data={"flagged": data}, numbers=[n for n in numbers if n])


# --- iv.panel_agreement ----------------------------------------------------------------------------

PANEL_SQL = (
    "select row_key, panel_id, role_id, n, members, kappa_mean, kappa_min, flagged, suppressed "
    "from interviews.panel_agreement order by panel_id"
)


def panel_agreement(params: _In, ctx: ToolContext) -> ToolResult:
    rows = _rows(ctx, PANEL_SQL)
    numbers = [
        _num(
            ctx,
            f"Fleiss kappa {r['panel_id']}",
            r["kappa_mean"],
            "kappa",
            PANEL_SQL,
            "panel_agreement",
            [r["row_key"]],
        )
        for r in rows
    ]
    data = [{k: r[k] for k in ("panel_id", "role_id", "n", "members", "flagged", "suppressed")} for r in rows]
    return ToolResult(data={"panels": data}, numbers=[n for n in numbers if n])


# --- iv.process_quality ----------------------------------------------------------------------------


class ProcessIn(_In):
    metric: ProcessMetric


PROC_SQL = (
    "select row_key, metric, scope, subject_id, n, value, threshold, flagged, suppressed "
    "from interviews.process_quality where metric = ? order by subject_id"
)


def process_quality(params: ProcessIn, ctx: ToolContext) -> ToolResult:
    rows = _rows(ctx, PROC_SQL, [params.metric])
    numbers = [
        _num(
            ctx,
            f"{params.metric} {r['subject_id']}",
            r["value"],
            "share",
            PROC_SQL,
            "process_quality",
            [r["row_key"]],
        )
        for r in rows
    ]
    data = [
        {k: r[k] for k in ("scope", "subject_id", "n", "threshold", "flagged", "suppressed")} for r in rows
    ]
    return ToolResult(data={"metric": params.metric, "rows": data}, numbers=[n for n in numbers if n])


# --- iv.impact_ratios -------------------------------------------------------------------------------


class ImpactIn(_In):
    role_id: str
    stage: Stage
    dimension: Dimension


IR_SQL = """
select row_key, "group", n, selected, rate, impact_ratio, reference_group, h, h_ci_low, h_ci_high, q, tested,
       excluded_small_share, flagged, suppressed
from interviews.fairness where role_id = ? and stage = ? and dimension = ? order by "group"
"""


def impact_ratios(params: ImpactIn, ctx: ToolContext) -> ToolResult:
    rows = _rows(ctx, IR_SQL, [params.role_id, params.stage, params.dimension])
    numbers = []
    scope = f"{params.role_id} {params.stage}"
    for r in rows:
        numbers += [
            _num(
                ctx,
                f"selection rate {r['group']} {scope}",
                r["rate"],
                "rate",
                IR_SQL,
                "fairness",
                [r["row_key"]],
            ),
            _num(
                ctx,
                f"impact ratio {r['group']} {scope}",
                r["impact_ratio"],
                "ratio",
                IR_SQL,
                "fairness",
                [r["row_key"]],
            ),
        ]
    groups = [
        {
            k: r[k]
            for k in (
                "group",
                "n",
                "reference_group",
                "h",
                "h_ci_low",
                "h_ci_high",
                "q",
                "tested",
                "excluded_small_share",
                "flagged",
                "suppressed",
            )
        }
        for r in rows
    ]
    excluded = [r["group"] for r in rows if r["excluded_small_share"]]
    note = None
    if excluded:
        share = _rows(ctx, "select min_group_share from interviews.run_meta")[0]["min_group_share"]
        note = f"Excluded from impact ratios as under {share:.0%} of candidates: {', '.join(excluded)}"
    return ToolResult(data={"groups": groups, "note": note}, numbers=[n for n in numbers if n])


# --- iv.predictive_validity -------------------------------------------------------------------------

PV_SQL = (
    "select row_key, role_id, n, rho, ci_low, ci_high, flagged, suppressed "
    "from interviews.predictive_validity order by role_id"
)


def predictive_validity(params: _In, ctx: ToolContext) -> ToolResult:
    rows = _rows(ctx, PV_SQL)
    numbers = []
    for r in rows:
        numbers += [
            _num(
                ctx,
                f"Spearman rho {r['role_id']}",
                r["rho"],
                "rho",
                PV_SQL,
                "predictive_validity",
                [r["row_key"]],
            ),
            _num(
                ctx, f"hires {r['role_id']}", r["n"], "count", PV_SQL, "predictive_validity", [r["row_key"]]
            ),
        ]
    data = [{k: r[k] for k in ("role_id", "n", "ci_low", "ci_high", "flagged", "suppressed")} for r in rows]
    return ToolResult(data={"roles": data}, numbers=[n for n in numbers if n])


# --- iv.tagger_quality ------------------------------------------------------------------------------

TQ_SQL = "select row_key, comparison, notes, kappa, target from interviews.tagger_quality order by comparison"


def tagger_quality(params: _In, ctx: ToolContext) -> ToolResult:
    rows = _rows(ctx, TQ_SQL)
    numbers = [
        _num(
            ctx,
            f"Cohen's kappa {r['comparison']}",
            r["kappa"],
            "kappa",
            TQ_SQL,
            "tagger_quality",
            [r["row_key"]],
        )
        for r in rows
    ]
    # "notes" counts labelled notes, not people, so the small-cell rule does not apply to it.
    data = [{"comparison": r["comparison"], "notes": r["notes"], "target": r["target"]} for r in rows]
    caveat = (
        "Agreement with generator truth overstates real accuracy: the same author wrote the synthetic notes "
        "and the rules. The owner's spot-check labels are the real check."
    )
    return ToolResult(data={"comparisons": data, "caveat": caveat}, numbers=[n for n in numbers if n])


# --- iv.export_compliance_pack (write) --------------------------------------------------------------


class ExportIn(_In):
    approval_id: str | None = Field(
        default=None, description="Omit when proposing; an approved record to execute."
    )


def compliance_snapshot(db_path: Path) -> dict[str, Any]:
    from ledgerlens.interviews.compliance import render_pack  # noqa: PLC0415 (heavy import only when used)

    pack = render_pack(db_path)
    return {"run_id": pack.run_id, "pack_hash": pack.digest}


def export_compliance_pack(params: ExportIn, ctx: ToolContext) -> ToolResult:
    """Write the approved compliance pack. The rendered pack must hash to what the human approved."""
    from ledgerlens.interviews.compliance import COMPLIANCE_DIR, render_pack, write_pack  # noqa: PLC0415

    queue = ApprovalQueue(ctx.db_path)
    approval = queue.require_approved(params.approval_id or "", action="iv.export_compliance_pack")
    pack = render_pack(ctx.db_path)
    if approval.payload.get("snapshot", {}).get("pack_hash") != pack.digest:
        raise ApprovalError("the compliance pack changed after it was approved; propose it again")
    name = f"{pack.run_id}_{approval.approval_id}"
    folder: Path | None = None
    with connect(ctx.db_path) as conn:
        # Consume the approval, record the export and write the files as one unit: a failure anywhere
        # rolls back the approval and removes the folder, so no files exist without an audit row.
        conn.execute("begin transaction")
        try:
            conn.execute("create schema if not exists interviews")
            conn.execute(
                "create table if not exists interviews.compliance_exports (approval_id varchar primary key, "
                "run_id varchar, pack_hash varchar, folder varchar, exported_at timestamp)"
            )
            conn.execute(
                "insert into interviews.compliance_exports values (?, ?, ?, ?, ?)",
                [approval.approval_id, pack.run_id, pack.digest, name, utcnow()],
            )
            queue.mark_executed(approval.approval_id, conn)
            folder = write_pack(pack, COMPLIANCE_DIR, name)
            conn.execute("commit")
        except Exception:
            conn.execute("rollback")
            if folder is not None:
                shutil.rmtree(folder, ignore_errors=True)
            raise
    return ToolResult(data={"exported": True, "files": sorted(pack.files), "folder": name})


def _tool(name: str, description: str, model: type[BaseModel], fn: Any, **extra: Any) -> Tool:
    return Tool(
        name=name, description=description, input_model=model, fn=fn, output_guard=aggregate_only, **extra
    )


INTERVIEW_TOOLS: tuple[Tool, ...] = (
    _tool(
        "iv.plan_status",
        "The pre-registered analysis plan version and hash, and flag counts from the latest run.",
        _In,
        plan_status,
    ),
    _tool(
        "iv.calibration",
        "Calibration cells (z, Cohen's d with CI, BH q): the flagged ones, or all cells for one interviewer.",
        CalibrationIn,
        calibration,
    ),
    _tool(
        "iv.interviewer_flags",
        "Interviewers flagged for calibration drift in a role, with the number of flagged cells.",
        _In,
        interviewer_flags,
    ),
    _tool(
        "iv.panel_agreement",
        "Mean Fleiss' kappa per onsite panel and whether it falls below the plan threshold.",
        _In,
        panel_agreement,
    ),
    _tool(
        "iv.process_quality",
        "Rubric coverage or evidence quality per interviewer, or loop coverage per role.",
        ProcessIn,
        process_quality,
    ),
    _tool(
        "iv.impact_ratios",
        "Selection rates and impact ratios by group for one role, stage and dimension (LL144-style).",
        ImpactIn,
        impact_ratios,
    ),
    _tool(
        "iv.predictive_validity",
        "Spearman correlation of interview score and later performance per role, with CI.",
        _In,
        predictive_validity,
    ),
    _tool(
        "iv.tagger_quality",
        "Agreement (Cohen's kappa) of the evidence tagger with generator truth and owner labels.",
        _In,
        tagger_quality,
    ),
    _tool(
        "iv.export_compliance_pack",
        "Propose exporting the compliance pack (call with no arguments); a human approves it.",
        ExportIn,
        export_compliance_pack,
        writes=True,
        snapshot=compliance_snapshot,
    ),
)
