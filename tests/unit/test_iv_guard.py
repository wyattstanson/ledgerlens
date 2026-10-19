"""The interviews output guard and the write gate in execute_tool (ADR 0004 decision 5)."""

from pathlib import Path

import pytest
from fakes import fetch_one
from pydantic import BaseModel, ConfigDict

from ledgerlens.core.approvals import ApprovalError
from ledgerlens.core.db import connect
from ledgerlens.core.lineage import NumberRef
from ledgerlens.core.tools import OutputBlockedError, Tool, ToolContext, ToolResult, execute_tool
from ledgerlens.interviews.tools import OutputPolicyError, aggregate_only


class In(BaseModel):
    model_config = ConfigDict(extra="forbid")


class WriteIn(In):
    approval_id: str | None = None


def result(data: object, label: str = "rate") -> ToolResult:
    return ToolResult(data=data, numbers=[NumberRef(label=label, value=0.5, unit="rate", lineage_id="ln")])


def test_aggregates_with_suppressed_small_cells_pass():
    aggregate_only(
        result(
            {
                "groups": [
                    {"group": "Female", "n": 120, "rate": 0.4},
                    {"group": "Asian", "n": 5, "rate": 0.6},  # exactly at the floor
                    {"group": "Unknown", "n": None, "suppressed": True},
                    {"group": "Two or More", "n": 3, "suppressed": True},
                ]
            }
        )
    )


@pytest.mark.parametrize(
    ("data", "label", "message"),
    [
        ({"rows": [{"candidate_id": "x"}]}, "rate", "person-level keys"),
        ({"deep": [[{"note_text": "said X"}]]}, "rate", "person-level keys"),
        ({"cells": [{"group": "A", "n": 4}]}, "rate", "n=4"),
        ({"cells": [{"group": "A", "n": 0, "suppressed": False}]}, "rate", "n=0"),
        ({"cells": [{"group": "A", "n": 3.0}]}, "rate", "n=3.0"),  # a float n is still a small cell
        ({"rows": [{"Candidate_ID": "x"}]}, "rate", "person-level keys"),  # key case does not matter
        ({"by_person": {"CAND-00017": 0.4}}, "rate", "vault identifier"),  # ids hidden in keys
        ({"comment": "see cand-00017"}, "rate", "vault identifier"),
        ({"comment": "see CAND-00017"}, "rate", "vault identifier"),
        (["IV-12 ran long"], "rate", "vault identifier"),
        ({"ok": True}, "selection rate NT-00004", "number label"),
    ],
)
def test_person_level_content_is_refused(data: object, label: str, message: str):
    with pytest.raises(OutputPolicyError, match=message):
        aggregate_only(result(data, label))


def test_interviewer_ids_are_aggregate_subjects_not_vault_ids():
    aggregate_only(result({"flagged": [{"interviewer_id": "INT-01", "role_id": "backend", "cells": 5}]}))


def _start_run(db: Path) -> None:
    with connect(db) as conn:
        conn.execute(
            "insert into audit.agent_runs values ('run_g', 'test', 'test', now(), now(), 'completed')"
        )


def _leaky(params: In, ctx: ToolContext) -> ToolResult:
    number = ctx.number(
        label="rate CAND-00001",
        value=0.5,
        unit="rate",
        sql_text="select 1",
        source_tables=["interviews.fairness"],
        row_ids=["x"],
    )
    return ToolResult(data={"candidate_id": "CAND-00001"}, numbers=[number])


def test_execute_tool_blocks_a_leaky_result_and_audits_it(db: Path):
    leaky = Tool(name="iv.leaky", description="d", input_model=In, output_guard=aggregate_only, fn=_leaky)
    _start_run(db)
    with pytest.raises(OutputBlockedError, match="output guard"):
        execute_tool(leaky, {}, db_path=db, agent_run_id="run_g")
    with connect(db) as conn:
        assert fetch_one(conn, "select status, output_hash from audit.tool_calls") == ("blocked", None)
        assert fetch_one(conn, "select count(*) from audit.lineage") == (0,)  # its label never persists


def test_write_tools_must_take_an_approval_id():
    with pytest.raises(ValueError, match="approval_id"):
        Tool(
            name="iv.write",
            description="d",
            input_model=In,
            fn=lambda p, ctx: ToolResult(data={}),
            writes=True,
        )


def test_write_tools_never_run_without_an_approved_record(db: Path):
    calls: list[ToolContext] = []
    tool = Tool(
        name="iv.write",
        description="d",
        input_model=WriteIn,
        writes=True,
        fn=lambda p, ctx: calls.append(ctx) or ToolResult(data={}),
    )
    _start_run(db)
    for args in ({}, {"approval_id": "ap_forged"}):
        with pytest.raises(ApprovalError):
            execute_tool(tool, args, db_path=db, agent_run_id="run_g")
    assert calls == []
    with connect(db) as conn:
        assert conn.execute("select distinct status from audit.tool_calls").fetchall() == [("refused",)]
