import json
from pathlib import Path

import pytest
from fakes import FakeClient, answer_citing_last_tool, fetch_one, tool_use_message

from ledgerlens.core.agent_loop import AgentDeps
from ledgerlens.core.approvals import ApprovalQueue
from ledgerlens.core.db import connect
from ledgerlens.core.evals import run as runner
from ledgerlens.core.evals.golden import (
    ExpectedNumber,
    GoldenCase,
    load_golden,
    run_llm_case,
    run_offline_case,
    score_numbers,
    summarize,
)
from ledgerlens.core.lineage import NumberRef
from ledgerlens.core.llm import LLMClient, ModelsConfig
from ledgerlens.core.registry import AgentRegistry, AgentSpec, Mandate, default_tool_registry
from ledgerlens.settings import Settings


def write_golden(path: Path, cases: list[dict]) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    (path / "core.jsonl").write_text("\n".join(json.dumps(c) for c in cases) + "\n", encoding="utf-8")
    return path


def offline_case(case_id: str, value: float) -> dict:
    return {
        "id": case_id,
        "pack": "core",
        "suite": "offline",
        "expected_source": "fixture:audit.llm_calls",
        "tool": "core.llm_spend",
        "input": {},
        "expected": [{"label": "LLM calls (all time)", "value": value, "abs_tol": 0}],
    }


def test_score_numbers_matches_by_label_within_tolerance():
    actual = [NumberRef(label="a", value=1.004, unit="USD", lineage_id="ln")]
    assert score_numbers([ExpectedNumber(label="a", value=1.0, abs_tol=0.005)], actual) == []
    assert score_numbers([ExpectedNumber(label="a", value=2.0)], actual)
    assert score_numbers([ExpectedNumber(label="b", value=1.0)], actual) == ["missing number 'b'"]


def test_offline_case_passes_and_fails_against_compute(db: Path):
    tools = default_tool_registry()
    assert run_offline_case(GoldenCase(**offline_case("ok", 0)), tools=tools, db_path=db).passed
    failed = run_offline_case(GoldenCase(**offline_case("bad", 3)), tools=tools, db_path=db)
    assert not failed.passed and "expected 3.0" in failed.failures[0]
    unknown = run_offline_case(
        GoldenCase(**{**offline_case("x", 0), "tool": "rc.nope"}), tools=tools, db_path=db
    )
    assert not unknown.passed


def test_offline_case_reports_tool_errors(db: Path):
    case = GoldenCase(**{**offline_case("x", 0), "input": {"month": "bad"}})
    result = run_offline_case(case, tools=default_tool_registry(), db_path=db)
    assert not result.passed and result.failures[0].startswith("ToolInputError")


def test_load_golden_filters_by_suite_and_validates(tmp_path: Path):
    llm = {**offline_case("q1", 0), "suite": "llm", "agent": "a", "question": "q"}
    golden = write_golden(tmp_path / "g", [offline_case("o1", 0), llm])
    assert [c.id for c in load_golden(golden, "offline")] == ["o1"]
    write_golden(tmp_path / "dup", [offline_case("o1", 0), offline_case("o1", 1)])
    with pytest.raises(ValueError, match="unique"):
        load_golden(tmp_path / "dup", "offline")
    write_golden(tmp_path / "bad", [{**offline_case("o1", 0), "expected_source": ""}])
    with pytest.raises(ValueError, match=r"core\.jsonl:1"):
        load_golden(tmp_path / "bad", "offline")


def test_summarize_means_metrics():
    from ledgerlens.core.evals.golden import CaseResult  # noqa: PLC0415

    summary = summarize(
        [CaseResult("a", True, [], {"cost_usd": 1.0}), CaseResult("b", False, ["x"], {"cost_usd": 3.0})]
    )
    assert summary["cases"] == 2 and summary["passed"] == 1
    assert summary["mean_cost_usd"] == 2.0
    assert summary["failed"] == [{"id": "b", "failures": ["x"]}]


def test_llm_case_scores_accuracy_lineage_and_cost(db: Path, models_config: ModelsConfig):
    fake = FakeClient(
        [tool_use_message(("core__llm_spend", {})), answer_citing_last_tool("Spend is ${0:.2f}.")]
    )
    deps = AgentDeps(
        tools=default_tool_registry(),
        llm=LLMClient(models_config, db, client=fake),
        approvals=ApprovalQueue(db),
        db_path=db,
    )
    agents = AgentRegistry(
        [AgentSpec(id="core.spend", owner="t", mandate=Mandate.READ_ONLY, tools=("core.llm_spend",))]
    )
    case = GoldenCase(
        id="spend",
        pack="core",
        suite="llm",
        expected_source="fixture",
        agent="core.spend",
        question="How much?",
        # The agent's planning call is already logged when the tool counts calls.
        expected=[ExpectedNumber(label="LLM calls (all time)", value=1.0, abs_tol=0)],
    )
    result = run_llm_case(case, agents=agents, deps=deps)
    assert result.passed, result.failures
    assert result.metrics["lineage_pass"] == 1.0 and result.metrics["tool_calls"] == 1.0
    with connect(db) as conn:
        (logged,) = fetch_one(conn, "select coalesce(sum(cost_usd), 0) from audit.llm_calls")
    assert result.metrics["cost_usd"] == pytest.approx(logged)


@pytest.fixture
def cli_settings(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Settings:
    settings = Settings(db=tmp_path / "w.duckdb", anthropic_api_key=None)
    monkeypatch.setattr(runner, "get_settings", lambda: settings)
    return settings


def test_cli_empty_golden_dir_needs_allow_empty(cli_settings: Settings, tmp_path: Path):
    empty = tmp_path / "empty"
    empty.mkdir()
    assert runner.main(["--suite", "offline", "--golden-dir", str(empty)]) == 1
    assert runner.main(["--suite", "offline", "--golden-dir", str(empty), "--allow-empty"]) == 0


def test_cli_offline_exit_codes(cli_settings: Settings, tmp_path: Path):
    passing = write_golden(tmp_path / "pass", [offline_case("o1", 0)])
    failing = write_golden(tmp_path / "fail", [offline_case("o1", 9)])
    assert runner.main(["--suite", "offline", "--golden-dir", str(passing)]) == 0
    assert runner.main(["--suite", "offline", "--golden-dir", str(failing)]) == 1


def test_cli_llm_suite_without_key_cannot_run(cli_settings: Settings, tmp_path: Path):
    llm = {**offline_case("q1", 0), "suite": "llm", "agent": "a", "question": "q"}
    golden = write_golden(tmp_path / "g", [llm])
    assert runner.main(["--suite", "llm", "--golden-dir", str(golden)]) == 2
