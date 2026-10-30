from pathlib import Path

import pytest
from fakes import fetch_one
from pydantic import BaseModel, ConfigDict

from ledgerlens.core.db import connect, utcnow
from ledgerlens.core.tools import CORE_TOOLS, Tool, ToolContext, ToolInputError, ToolResult, execute_tool

LLM_SPEND = CORE_TOOLS[0]


class Closed(BaseModel):
    model_config = ConfigDict(extra="forbid")
    value: int


class Open(BaseModel):
    value: int


def echo(params: Closed, ctx: ToolContext) -> ToolResult:
    return ToolResult(data={"value": params.value})


def boom(params: Closed, ctx: ToolContext) -> ToolResult:
    raise RuntimeError("compute failed")


def tool_call_rows(db: Path) -> list[tuple]:
    with connect(db) as conn:
        return conn.execute(
            "select tool_name, input_json, status, output_hash, ms from audit.tool_calls"
        ).fetchall()


def test_tool_exports_a_strict_closed_schema_under_a_wire_safe_name():
    tool = Tool(name="rc.optimize_panel", description="d", input_model=Closed, fn=echo)
    exported = tool.to_anthropic()
    assert exported["name"] == "rc__optimize_panel"
    assert exported["strict"] is True
    assert exported["input_schema"]["additionalProperties"] is False


def test_open_input_models_are_rejected_at_definition_time():
    with pytest.raises(ValueError, match="additionalProperties"):
        Tool(name="rc.bad", description="d", input_model=Open, fn=echo)


@pytest.mark.parametrize("name", ["nodots", "Rc.Bad", "rc.bad-name", "rc." + "x" * 70])
def test_tool_names_must_be_dotted_lowercase_and_short(name):
    with pytest.raises(ValueError):
        Tool(name=name, description="d", input_model=Closed, fn=echo)


def test_execute_tool_audits_the_call_with_an_output_hash(db: Path):
    tool = Tool(name="t.echo", description="d", input_model=Closed, fn=echo)
    result = execute_tool(tool, {"value": 3}, db_path=db, agent_run_id="run_1")
    assert result.data == {"value": 3}
    [(name, input_json, status, output_hash, ms)] = tool_call_rows(db)
    assert (name, input_json, status) == ("t.echo", '{"value": 3}', "ok")
    assert len(output_hash) == 64
    assert ms >= 0


def test_invalid_input_raises_and_is_audited(db: Path):
    tool = Tool(name="t.echo", description="d", input_model=Closed, fn=echo)
    with pytest.raises(ToolInputError):
        execute_tool(tool, {"value": "x", "extra": 1}, db_path=db, agent_run_id="run_1")
    assert tool_call_rows(db)[0][2] == "invalid_input"


def test_compute_failure_is_audited_and_reraised(db: Path):
    tool = Tool(name="t.boom", description="d", input_model=Closed, fn=boom)
    with pytest.raises(RuntimeError, match="compute failed"):
        execute_tool(tool, {"value": 1}, db_path=db, agent_run_id="run_1")
    assert tool_call_rows(db)[0][2] == "error"


def _seed_llm_calls(db: Path) -> None:
    rows = [
        ("llm_a", 0.25, "2026-09-03 10:00:00"),
        ("llm_b", 0.50, "2026-10-01 09:00:00"),
        ("llm_c", 1.00, "2026-10-15 12:00:00"),
    ]
    with connect(db) as conn:
        for call_id, cost, at in rows:
            conn.execute(
                "insert into audit.llm_calls values (?, null, 'pack_agent', 'm', 'low', 1, 1, 0, 0, ?, 1, ?)",
                [call_id, cost, at],
            )


def test_llm_spend_totals_with_lineage_to_the_call_rows(db: Path):
    _seed_llm_calls(db)
    result = execute_tool(LLM_SPEND, {"month": "2026-10"}, db_path=db, agent_run_id="run_1")
    cost, calls = result.numbers
    assert (cost.value, cost.unit, calls.value) == (1.5, "USD", 2.0)
    with connect(db) as conn:
        (row_ids,) = fetch_one(
            conn, "select row_ids from audit.lineage where lineage_id = ?", [cost.lineage_id]
        )
    assert sorted(row_ids) == ["llm_b", "llm_c"]


def test_llm_spend_all_time_and_empty_log(db: Path):
    empty = execute_tool(LLM_SPEND, {}, db_path=db, agent_run_id="run_1")
    assert [n.value for n in empty.numbers] == [0.0, 0.0]
    _seed_llm_calls(db)
    total = execute_tool(LLM_SPEND, {}, db_path=db, agent_run_id="run_1")
    assert total.numbers[0].value == pytest.approx(1.75)


def test_llm_spend_rejects_malformed_month(db: Path):
    with pytest.raises(ToolInputError):
        execute_tool(LLM_SPEND, {"month": "2026-13"}, db_path=db, agent_run_id="run_1")


def test_utcnow_is_naive():
    assert utcnow().tzinfo is None


def many_numbers(params: Closed, ctx: ToolContext) -> ToolResult:
    numbers = [
        ctx.number(
            label=f"n{i}",
            value=float(i),
            unit="count",
            sql_text=f"select {i}",
            source_tables=["t"],
            row_ids=[i],
        )
        for i in range(params.value)
    ]
    return ToolResult(data={"n": params.value}, numbers=numbers)


def test_every_number_a_tool_makes_is_written_once_and_resolves(db: Path):
    """Lineage is batched for speed, so prove the batch is complete and matches what was returned."""
    tool = Tool(name="t.many", description="d", input_model=Closed, fn=many_numbers)
    result = execute_tool(tool, {"value": 25}, db_path=db, agent_run_id="run_many")

    with connect(db) as conn:
        stored = dict(conn.execute("select lineage_id, value from audit.lineage").fetchall())
    assert len(stored) == 25, "every number must leave exactly one lineage row"
    assert {n.lineage_id: n.value for n in result.numbers} == stored


def test_a_tool_that_fails_after_citing_numbers_leaves_no_lineage(db: Path):
    def cite_then_fail(params: Closed, ctx: ToolContext) -> ToolResult:
        many_numbers(params, ctx)
        raise RuntimeError("compute failed after citing")

    tool = Tool(name="t.halfway", description="d", input_model=Closed, fn=cite_then_fail)
    with pytest.raises(RuntimeError, match="after citing"):
        execute_tool(tool, {"value": 5}, db_path=db, agent_run_id="run_fail")
    with connect(db) as conn:
        assert conn.execute("select count(*) from audit.lineage").fetchone() == (0,)
