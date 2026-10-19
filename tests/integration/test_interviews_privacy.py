"""Privacy sweep (rule 2, ADR 0004): no person-level data leaves the vault via tools, MCP, exports or logs."""

import asyncio
import csv
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
from mcp import Client

from ledgerlens.core.approvals import ApprovalError, ApprovalQueue, ApprovalStatus, DecisionKind
from ledgerlens.core.db import connect, ensure_audit_schema
from ledgerlens.core.lineage import lineage_values_for_run
from ledgerlens.core.stats import cohens_h
from ledgerlens.core.tools import execute_tool
from ledgerlens.interviews import compliance
from ledgerlens.interviews.config import InterviewsConfig
from ledgerlens.interviews.disclosure import TOTAL_ROUNDING
from ledgerlens.interviews.generate import STAGES, InterviewWorld, generate_world, write_world
from ledgerlens.interviews.pipeline import run_interviews, spot_check_notes
from ledgerlens.interviews.plan import AnalysisPlan
from ledgerlens.interviews.tagger import RuleTagger, write_spot_check
from ledgerlens.interviews.tools import FORBIDDEN_KEYS, INTERVIEW_TOOLS, VAULT_ID
from ledgerlens.mcp.server import build_server
from ledgerlens.packs import default_tool_registry
from ledgerlens.settings import REPO_ROOT

CONFIG = InterviewsConfig.from_yaml(REPO_ROOT / "config" / "interviews.yaml")
PLAN = AnalysisPlan.from_yaml(REPO_ROOT / "config" / "analysis_plan.yaml")
TOOLS = default_tool_registry()
EXPORT = "iv.export_compliance_pack"
CALLS: list[tuple[str, dict[str, Any]]] = [
    ("iv.plan_status", {}),
    ("iv.calibration", {}),
    ("iv.calibration", {"interviewer_id": "INT-01"}),
    ("iv.interviewer_flags", {}),
    ("iv.panel_agreement", {}),
    ("iv.predictive_validity", {}),
    ("iv.tagger_quality", {}),
    *[
        ("iv.process_quality", {"metric": m})
        for m in ("rubric_coverage", "evidence_quality", "loop_coverage")
    ],
    # qa is a small role, so its race/ethnicity cells exercise suppression and the 2% exclusion.
    *[
        ("iv.impact_ratios", {"role_id": r, "stage": s, "dimension": d})
        for r in ("backend", "qa")
        for s in STAGES
        for d in ("sex", "race_ethnicity")
    ],
]


@dataclass(frozen=True)
class Built:
    warehouse: Path
    world: InterviewWorld
    sheet: Path


@pytest.fixture(scope="module")
def built(tmp_path_factory: pytest.TempPathFactory) -> Built:
    root = tmp_path_factory.mktemp("iv_privacy")
    warehouse, vault, sheet = root / "warehouse.duckdb", root / "vault.duckdb", root / "spot_check_50.csv"
    ensure_audit_schema(warehouse)
    world = generate_world(CONFIG.world, 42)
    write_world(warehouse, vault, world)
    write_spot_check(sheet, spot_check_notes(vault, PLAN))
    truth = dict(world.truth_note_labels.select("note_id", "has_evidence").iter_rows())
    with sheet.open(encoding="utf-8", newline="") as handle:
        labelled = [
            {**row, "owner_has_evidence": "yes" if truth[row["note_id"]] else "no"}
            for row in csv.DictReader(handle)
        ]
    with sheet.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["note_id", "note_text", "owner_has_evidence"])
        writer.writeheader()
        writer.writerows(labelled)
    run_interviews(warehouse, vault, PLAN, RuleTagger(), owner_labels=sheet)
    return Built(warehouse, world, sheet)


def long_note_texts(world: InterviewWorld) -> set[str]:
    return {t for t in world.notes["text"].to_list() if len(t) >= 30}


def assert_aggregate_only(text: str, world: InterviewWorld) -> None:
    assert not VAULT_ID.search(text), VAULT_ID.search(text)
    payload = text.lower()
    assert not any(f'"{key}"' in payload for key in FORBIDDEN_KEYS)
    assert not any(note in text for note in long_note_texts(world))


def start_run(db: Path, run_id: str) -> None:
    with connect(db) as conn:
        conn.execute(
            "insert into audit.agent_runs values (?, 'test', 'test', now(), now(), 'completed') "
            "on conflict do nothing",
            [run_id],
        )


@pytest.fixture(scope="module")
def swept(built: Built) -> list[str]:
    """Every read tool through execute_tool and over MCP; returns each serialized result."""
    start_run(built.warehouse, "run_sweep")
    outputs = []
    for name, args in CALLS:
        result = execute_tool(TOOLS.get(name), args, db_path=built.warehouse, agent_run_id="run_sweep")
        outputs.append(result.model_dump_json())
    server = build_server(TOOLS, built.warehouse)

    async def over_mcp() -> tuple[set[str], list[Any]]:
        async with Client(server) as client:
            listed = {t.name for t in (await client.list_tools()).tools}
            first_call = {name: args for name, args in reversed(CALLS)}
            return listed, [await client.call_tool(name, args) for name, args in first_call.items()]

    listed, mcp_results = asyncio.run(over_mcp())
    assert {t.name for t in INTERVIEW_TOOLS if not t.writes} <= listed and EXPORT not in listed
    assert not any(r.is_error for r in mcp_results)
    return outputs + [json.dumps(r.structured_content) for r in mcp_results]


def test_every_read_tool_was_swept(swept: list[str]):
    assert {name for name, _ in CALLS} == {t.name for t in INTERVIEW_TOOLS if not t.writes}


def test_tool_results_are_aggregate_only_and_traced(built: Built, swept: list[str]):
    for text in swept:
        assert_aggregate_only(text, built.world)
    recorded = lineage_values_for_run(built.warehouse, "run_sweep")
    numbers = [n for text in swept[: len(CALLS)] for n in json.loads(text)["numbers"]]
    assert numbers and all(recorded[n["lineage_id"]] == n["value"] for n in numbers)
    suppressed = [g for text in swept for g in json.loads(text)["data"].get("groups", []) if g["suppressed"]]
    assert suppressed and all(g["n"] is None and g["h"] is None for g in suppressed)


def test_warehouse_and_audit_log_hold_no_vault_ids(built: Built, swept: list[str]):
    with connect(built.warehouse) as conn:
        schemas = {
            r[0] for r in conn.execute("select schema_name from information_schema.schemata").fetchall()
        }
        columns = conn.execute(
            "select table_schema, table_name, column_name, data_type from information_schema.columns "
            "where table_catalog = current_database()"
        ).fetchall()
        assert "restricted" not in schemas
        assert not [c for c in columns if c[2] in FORBIDDEN_KEYS]
        leaks = {
            f"{s}.{t}.{c}": hits
            for s, t, c, kind in columns
            if kind == "VARCHAR"
            and (
                hits := conn.execute(
                    f'select count(*) from "{s}"."{t}" where regexp_matches("{c}", ?)',  # noqa: S608
                    [VAULT_ID.pattern],
                ).fetchone()[0]  # pyright: ignore[reportOptionalSubscript]
            )  # pyright: ignore[reportOptionalSubscript]
        }
    assert leaks == {}


def test_spot_check_sheet_carries_note_text_only(built: Built):
    with built.sheet.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == PLAN.tagger.spot_check_size
    assert list(rows[0]) == ["note_id", "note_text", "owner_has_evidence"]
    for row in rows:
        assert row["note_id"].startswith("NT-")
        assert not VAULT_ID.search(row["note_text"]) and "INT-" not in row["note_text"]


def test_owner_labels_reach_the_tagger_report(built: Built):
    with connect(built.warehouse) as conn:
        report = dict(conn.execute("select comparison, notes from interviews.tagger_quality").fetchall())
        kappa = conn.execute(
            "select kappa from interviews.tagger_quality where comparison = 'owner_vs_generator_truth'"
        ).fetchone()
    sample, sheet = PLAN.tagger.eval_sample, PLAN.tagger.spot_check_size
    assert report == {
        "tagger_vs_generator_truth": sample,
        "tagger_vs_owner": sheet,
        "owner_vs_generator_truth": sheet,
    }
    assert kappa == (pytest.approx(1.0),)  # the fixture labelled with the truth


def _approved_export(db: Path, *, kind: DecisionKind = DecisionKind.HUMAN) -> str:
    snapshot = TOOLS.get(EXPORT).snapshot
    assert snapshot is not None
    queue = ApprovalQueue(db)
    proposal = queue.propose(
        agent_run_id="run_sweep", action=EXPORT, payload={"arguments": {}, "snapshot": snapshot(db)}
    )
    who = "simulated:demo" if kind is DecisionKind.SIMULATED else "dana"
    queue.decide(proposal.approval_id, approve=True, decided_by=who, reason="reviewed", kind=kind)
    return proposal.approval_id


def test_compliance_export_needs_a_human_approval_of_this_exact_pack(
    built: Built, swept: list[str], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(compliance, "COMPLIANCE_DIR", tmp_path / "compliance")
    db, tool = built.warehouse, TOOLS.get(EXPORT)
    with pytest.raises(ApprovalError):
        execute_tool(tool, {}, db_path=db, agent_run_id="run_sweep")
    with pytest.raises(ApprovalError, match="simulated"):
        execute_tool(
            tool,
            {"approval_id": _approved_export(db, kind=DecisionKind.SIMULATED)},
            db_path=db,
            agent_run_id="run_sweep",
        )

    moved = _approved_export(db)
    with connect(db) as conn:
        conn.execute("update interviews.run_meta set tests_in_family = tests_in_family + 1")
    try:
        with pytest.raises(ApprovalError, match="changed after it was approved"):
            execute_tool(tool, {"approval_id": moved}, db_path=db, agent_run_id="run_sweep")
        assert ApprovalQueue(db).get(moved).status is ApprovalStatus.APPROVED  # not consumed
    finally:
        with connect(db) as conn:
            conn.execute("update interviews.run_meta set tests_in_family = tests_in_family - 1")

    approval_id = _approved_export(db)
    result = execute_tool(tool, {"approval_id": approval_id}, db_path=db, agent_run_id="run_sweep")
    folder = tmp_path / "compliance" / result.data["folder"]
    files = {p.name: p.read_text(encoding="utf-8") for p in folder.iterdir()}
    assert set(files) == set(result.data["files"]) and len(files) == 5
    for name, text in files.items():
        assert_aggregate_only(text, built.world)
        if name.endswith(".md"):
            assert compliance.BANNER in text and "## Sources" in text
    data = json.loads(files["compliance_data.json"])
    cells = data["fairness"] + data["intersectional"]
    assert all(c["n"] is None or c["n"] >= PLAN.min_group_n for c in cells)
    assert (
        "share" not in data["fairness"][0] and "p_raw" not in data["fairness"][0]
    )  # export is an allow-list
    assert ApprovalQueue(db).get(approval_id).status is ApprovalStatus.EXECUTED
    with pytest.raises(ApprovalError, match="executed, not approved"):
        execute_tool(tool, {"approval_id": approval_id}, db_path=db, agent_run_id="run_sweep")


def test_rerunning_the_same_plan_keeps_its_version_and_logs_a_new_run(built: Built, tmp_path: Path):
    warehouse, vault = (
        tmp_path / "warehouse.duckdb",
        tmp_path / "vault.duckdb",
    )  # a copy: the module's stays put
    shutil.copy(built.warehouse, warehouse)
    shutil.copy(built.warehouse.with_name("vault.duckdb"), vault)
    again = run_interviews(warehouse, vault, PLAN, RuleTagger())
    with connect(warehouse) as conn:
        runs = conn.execute(
            "select analysis_run_id, status from audit.analysis_runs order by started_at"
        ).fetchall()
        stamp = conn.execute(
            "select distinct analysis_run_id, plan_version from interviews.fairness"
        ).fetchall()
    assert again.version == 1 and [s for _, s in runs] == ["completed", "completed"]
    assert stamp == [(runs[-1][0], 1)]


KEY_COLUMNS = {
    "row_key",
    "role_id",
    "stage",
    "dimension",
    "group",
    "sex",
    "race_ethnicity",
    "suppressed",
    "analysis_run_id",
    "plan_version",
    "plan_sha256",
}


def test_hidden_rows_keep_only_their_labels(built: Built):
    with connect(built.warehouse) as conn:
        for table in ("fairness", "intersectional"):
            hidden = conn.execute(f"select * from interviews.{table} where suppressed").pl()  # noqa: S608
            assert hidden.height
            values = [c for c in hidden.columns if c not in KEY_COLUMNS]
            assert {c: hidden[c].null_count() for c in values} == dict.fromkeys(values, hidden.height), table


def test_every_group_that_applied_has_a_row_at_every_stage(built: Built):
    with connect(built.warehouse) as conn:
        counts = conn.execute(
            'select role_id, dimension, "group", count(*) from interviews.fairness group by all'
        ).fetchall()
    assert counts and {c for *_, c in counts} == {len(STAGES)}


def test_published_statistics_never_compare_against_everyone_else(built: Built):
    with connect(built.warehouse) as conn:
        columns = {r[0] for r in conn.execute("describe interviews.fairness").fetchall()}
        rows = conn.execute(
            "select f.rate, f.h, r.rate from interviews.fairness f join interviews.fairness r on "
            "f.role_id = r.role_id and f.stage = r.stage and f.dimension = r.dimension "
            'and r."group" = f.reference_group '
            "where f.h is not null"
        ).fetchall()
    assert not {"share", "p_raw", "q"} & columns
    assert rows and all(h == pytest.approx(round(cohens_h(a, b), 2)) for a, h, b in rows)


def test_workload_counts_are_rounded_so_they_cannot_give_stage_totals_away(built: Built):
    with connect(built.warehouse) as conn:
        for table in ("calibration", "panel_agreement", "process_quality"):
            counts = [
                r[0]
                for r in conn.execute(f"select n from interviews.{table} where n is not null").fetchall()  # noqa: S608
            ]
            assert counts and all(n % TOTAL_ROUNDING == 0 for n in counts), table
