"""Golden cases and scorers.

Expected values come from generator ground truth (``expected_source`` names it), never typed by hand.
The offline suite calls tools directly; the LLM suite asks an agent the case's question.
"""

import json
import time
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from ledgerlens.core.agent_loop import AgentDeps, RunStatus, run_agent
from ledgerlens.core.db import connect, utcnow
from ledgerlens.core.lineage import NumberRef, new_id
from ledgerlens.core.registry import AgentRegistry, ToolRegistry
from ledgerlens.core.tools import execute_tool


class ExpectedNumber(BaseModel):
    model_config = ConfigDict(extra="forbid")

    label: str
    value: float
    abs_tol: float = Field(default=0.005, ge=0)


class GoldenCase(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    pack: str
    suite: Literal["offline", "llm"]
    expected_source: str = Field(
        min_length=1, description="Generator ground-truth table that produced expected"
    )
    expected: list[ExpectedNumber] = Field(min_length=1)
    tool: str | None = None  # offline
    input: dict[str, Any] = Field(default_factory=dict)  # offline
    agent: str | None = None  # llm
    question: str | None = None  # llm


def load_golden(golden_dir: Path, suite: str) -> list[GoldenCase]:
    cases: list[GoldenCase] = []
    for path in sorted(golden_dir.glob("*.jsonl")):
        for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            try:
                case = GoldenCase.model_validate_json(line)
            except ValueError as exc:
                raise ValueError(f"{path.name}:{line_no}: {exc}") from exc
            if case.suite == suite:
                cases.append(case)
    ids = [c.id for c in cases]
    if len(ids) != len(set(ids)):
        raise ValueError("golden case ids must be unique")
    return cases


def score_numbers(expected: Iterable[ExpectedNumber], actual: Iterable[NumberRef]) -> list[str]:
    """Failures for expected numbers that are missing or out of tolerance (matched by label)."""
    by_label = {n.label: n.value for n in actual}
    failures = []
    for exp in expected:
        got = by_label.get(exp.label)
        if got is None:
            failures.append(f"missing number {exp.label!r}")
        elif abs(got - exp.value) > exp.abs_tol:
            failures.append(f"{exp.label!r}: expected {exp.value} +/- {exp.abs_tol}, got {got}")
    return failures


@dataclass
class CaseResult:
    case_id: str
    passed: bool
    failures: list[str] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)


def _start_run(db_path: Path, agent_id: str) -> str:
    run_id = new_id("run")
    with connect(db_path) as conn:
        conn.execute(
            "insert into audit.agent_runs values (?, ?, 'eval', ?, ?, 'completed')",
            [run_id, agent_id, utcnow(), utcnow()],
        )
    return run_id


def run_offline_case(case: GoldenCase, *, tools: ToolRegistry, db_path: Path) -> CaseResult:
    if case.tool is None or case.tool not in tools:
        return CaseResult(case.id, False, [f"offline case needs a registered tool, got {case.tool!r}"])
    run_id = _start_run(db_path, "eval.offline")
    try:
        result = execute_tool(tools.get(case.tool), case.input, db_path=db_path, agent_run_id=run_id)
    except Exception as exc:
        return CaseResult(case.id, False, [f"{type(exc).__name__}: {exc}"])
    failures = score_numbers(case.expected, result.numbers)
    return CaseResult(case.id, not failures, failures)


def run_llm_case(case: GoldenCase, *, agents: AgentRegistry, deps: AgentDeps) -> CaseResult:
    if case.agent is None or case.question is None:
        return CaseResult(case.id, False, ["llm case needs agent and question"])
    started = time.perf_counter()
    result = run_agent(agents.get(case.agent), case.question, user="eval", deps=deps)
    latency_ms = (time.perf_counter() - started) * 1000
    with connect(deps.db_path) as conn:
        row = conn.execute(
            "select coalesce(sum(cost_usd), 0) from audit.llm_calls where agent_run_id = ?",
            [result.agent_run_id],
        ).fetchone()
    cited = result.answer.numbers if result.answer else []
    failures = score_numbers(case.expected, cited)
    lineage_pass = result.status is RunStatus.COMPLETED
    if not lineage_pass:
        failures.append(f"run status {result.status.value}")
    metrics = {
        "numeric_accuracy": 1.0 - len(score_numbers(case.expected, cited)) / len(case.expected),
        "lineage_pass": float(lineage_pass),
        "tool_calls": float(result.tool_calls),
        "latency_ms": latency_ms,
        "cost_usd": float(row[0]) if row else 0.0,
    }
    return CaseResult(case.id, not failures, failures, metrics)


def summarize(results: list[CaseResult]) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "cases": len(results),
        "passed": sum(r.passed for r in results),
        "failed": [{"id": r.case_id, "failures": r.failures} for r in results if not r.passed],
    }
    metric_names = sorted({k for r in results for k in r.metrics})
    for name in metric_names:
        values = [r.metrics[name] for r in results if name in r.metrics]
        summary[f"mean_{name}"] = sum(values) / len(values)
    return summary


def dump_report(summary: dict[str, Any], out_dir: Path, suite: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{utcnow():%Y%m%dT%H%M%S}-{suite}.json"
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return path
