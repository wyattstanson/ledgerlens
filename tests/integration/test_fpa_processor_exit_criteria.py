"""The processor variant (BUILD_PROMPT 10.1, ``--variant processor``; ADR 0005) on the full seed-42 world,
through dbt and the warehouse: the Phase 4 exit criteria again, on the processor's own drivers, with the same
tools and the same agents as the insurer.
"""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import polars as pl
import pytest
from fakes import FakeClient, answer_citing_last_tool, tool_use_message

from ledgerlens.core.agent_loop import AgentDeps, RunStatus, run_agent
from ledgerlens.core.approvals import ApprovalError, ApprovalQueue
from ledgerlens.core.db import connect, ensure_audit_schema
from ledgerlens.core.evals.golden import load_golden, run_offline_case
from ledgerlens.core.lineage import lineage_values_for_run
from ledgerlens.core.llm import LLMClient, ModelsConfig
from ledgerlens.core.registry import AgentRegistry
from ledgerlens.core.tools import execute_tool
from ledgerlens.fpa.build import build_world, golden_cases
from ledgerlens.fpa.config import FpaConfig
from ledgerlens.fpa.drivers import net_cash_from_drivers
from ledgerlens.fpa.pipeline import DbtBuildError, FpaRun, WorldMismatch, run_dbt, run_fpa
from ledgerlens.fpa.processor.drivers import load_marts
from ledgerlens.fpa.processor.pipeline import BOOK, ENTITY, MEASURES
from ledgerlens.fpa.tools import NOT_ADVICE, scenario_snapshot
from ledgerlens.orchestration.fpa_assets import BRIDGE_GAP_SQL
from ledgerlens.packs import default_tool_registry

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG = FpaConfig.from_yaml(REPO_ROOT / "config" / "fpa.yaml")
GOLDEN_DIR = REPO_ROOT / "tests" / "evals" / "golden" / "fpa_processor"
SEED = 42


@pytest.fixture(scope="module")
def built(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, FpaRun]:
    db = tmp_path_factory.mktemp("fpa_processor") / "warehouse.duckdb"
    ensure_audit_schema(db)
    build_world(db, CONFIG, "processor", SEED)
    run = run_fpa(db, CONFIG, variant="processor")  # dbt build first: a failing model or ledger test raises
    return db, run


@pytest.fixture
def scratch(built, tmp_path: Path) -> Path:
    """A private copy of the built warehouse, for tests that write or tamper with it."""
    copy = tmp_path / "warehouse.duckdb"
    shutil.copy(built[0], copy)
    return copy


def table(db: Path, sql: str, params: list[object] | None = None) -> pl.DataFrame:
    with connect(db) as conn:
        return conn.execute(sql, params or []).pl()


def start_run(db: Path, run_id: str) -> None:
    with connect(db) as conn:
        conn.execute(
            "insert into audit.agent_runs values (?, 'test', 'test', now(), now(), 'completed')", [run_id]
        )


def test_the_ledger_balances_and_reconciles_to_the_processor_feed(built):
    db, run = built
    assert run.variant == "processor"
    assert table(
        db,
        'select "transaction" from fpa.stg_fpa__transactionline group by 1 having abs(sum(amount)) > 0.005',
    ).is_empty()
    gaps = table(
        db,
        """
        select c.month
        from fpa.fpa_cash_monthly c
        join (select strftime(event_date, '%Y-%m') as month,
                     sum(case direction when 'out' then -amount else amount end) as net
              from fpa.stg_fpa__payment_events group by 1) f using (month)
        where abs(c.settlements - c.chargebacks_debited + c.chargebacks_recovered - f.net) > 0.005
        """,
    )
    assert gaps.is_empty()
    assert (
        table(db, "select count(*) from fpa.fpa_premium_monthly").item() == 0
    )  # the insurer's marts are empty


@pytest.mark.parametrize("test_name", ["fpa_settlements_tie_to_captures", "fpa_chargebacks_tie_to_disputes"])
def test_a_feed_that_drifts_from_its_records_fails_the_dbt_tie_out(scratch, test_name):
    with connect(scratch) as conn:
        kind = "settlement" if test_name == "fpa_settlements_tie_to_captures" else "chargeback"
        conn.execute(
            "update raw_fpa.payment_events set amount = amount + 1 "
            "where event_id = (select min(event_id) from raw_fpa.payment_events where kind = ?)",
            [kind],
        )
    with pytest.raises(DbtBuildError, match=test_name):
        run_dbt(scratch, select=test_name)


def test_every_bridge_sums_exactly_to_its_variance(built):
    db, run = built
    assert table(db, BRIDGE_GAP_SQL).item() == 0
    worst = (
        run.bridges.filter(pl.col("component") == "residual")
        .group_by("measure")
        .agg(pl.col("cents").abs().max())
    )
    assert all(cents <= (8 if m == "net_operating_cash" else 2) for m, cents in worst.iter_rows())


def test_the_driver_tree_reproduces_the_ledgers_net_cash(built):
    db, _ = built
    signs = {measure: sign for measure, (_, sign) in MEASURES.items()}
    tree = net_cash_from_drivers(table(db, "select * exclude (row_key) from fpa.drivers"), signs)
    ledger = load_marts(db).cash.select("month", "net_operating_cash")
    joined = tree.join(ledger, on="month")
    assert joined.height == ledger.height == 36
    assert joined.select((pl.col("tree_net_cash") - pl.col("net_operating_cash")).abs().max()).item() < 1e-6


def ytd_components(db: Path) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for m, c, cents in table(
        db, "select measure, component, cents from fpa.bridge where period = 'ytd'"
    ).iter_rows():
        out.setdefault(m, {})[c] = cents / 100
    return out


def largest(components: dict[str, float]) -> str:
    return max(components, key=lambda c: abs(components[c]))


def test_each_planted_miss_shows_up_in_its_bridge_column(built):
    ytd = ytd_components(built[0])
    assert largest(ytd["fee_revenue"]) == "volume" and ytd["fee_revenue"]["volume"] > 0  # digital merchants
    assert largest(ytd["interchange"]) == "mix" and ytd["interchange"]["mix"] > 0  # debit to premium credit
    assert ytd["interchange"]["rate"] > 0  # premium credit interchange up
    assert largest(ytd["chargebacks"]) == "volume" and ytd["chargebacks"]["volume"] > 0  # agent traffic
    assert abs(ytd["fee_revenue"]["rate"]) < 0.001 * abs(ytd["fee_revenue"]["volume"])  # blended pricing held


def test_each_planted_miss_moves_its_driver(built):
    db, _ = built
    drivers = table(db, "select month, lob, driver, segment, value from fpa.drivers where month = '2026-08'")

    def value(driver: str, segment: str = "", lob: str | None = None) -> float:
        """The driver for one vertical, or its average over the verticals."""
        picked = drivers.filter((pl.col("driver") == driver) & (pl.col("segment") == segment))
        if lob is not None:
            picked = picked.filter(pl.col("lob") == lob)
        return float(picked["value"].sum()) / picked.height

    plan = CONFIG.processor.world
    planted = plan.planted
    assert value("agent_share") == pytest.approx(0.02 + 0.13 * 34 / 35, abs=0.01)  # plan path: 0.02 -> 0.06
    assert value("authorization_rate", "agent") == pytest.approx(
        0.86 - planted.agent_auth_rate_drop, abs=0.01
    )
    assert value("card_mix", "debit") < 0.38 * (1 - planted.card_mix_debit_to_premium) + 0.01
    plan_merchants = table(
        db, "select active_merchants from fpa.stg_fpa__plan_merchants where month = '2026-08' and lob = ?",
        ["digital_goods"],
    ).item()  # fmt: skip
    assert value("merchants", lob="digital_goods") > 1.15 * plan_merchants


def test_material_variances_carry_adr_0001_statuses(built):
    db, _ = built
    rows = table(db, "select measure, status, favourable, level from fpa.bridge_totals where period = 'ytd'")
    ytd = {m: (s, f) for m, s, f, _ in rows.iter_rows()}
    assert ytd["fee_revenue"] == ("material", True)
    assert ytd["interchange"] == ("material", False)
    assert ytd["chargebacks"] == ("material", False)
    assert ytd["net_operating_cash"] == ("material", False)  # margin squeezed despite higher volume
    meta = table(db, "select variant, entity, book, open_period, materiality_band_ok from fpa.run_meta").row(
        0
    )
    assert meta == ("processor", ENTITY, BOOK, "2026-09", True)
    preliminary = table(
        db, "select distinct preliminary, confidence from fpa.bridge_totals where period = '2026-09'"
    )
    assert preliminary.rows() == [(True, "medium")]


def test_fiscal_ytd_covers_closed_months_only(built):
    db, _ = built
    totals = table(db, "select measure, months, actual_cents from fpa.bridge_totals where period = 'ytd'")
    assert set(totals["months"]) == {8}
    monthly = table(
        db,
        "select measure, sum(actual_cents) as cents from fpa.bridge_totals "
        "where period between '2026-01' and '2026-08' group by measure",
    )
    assert dict(totals.select("measure", "actual_cents").iter_rows()) == dict(monthly.iter_rows())


CALLS = [
    ("fpa.drivers", {"month": "2026-08", "lob": "digital_goods"}),
    ("fpa.bridge", {"measure": "interchange", "period": "ytd"}),
    ("fpa.variance_flags", {"period": "ytd"}),
    ("fpa.scenario", {}),
    ("payments.events", {"kind": "chargeback_recovered", "limit": 5}),
    ("payments.summary", {"month": "2026-08"}),
    ("ledger.income_statement", {"period": "2026-08"}),
    ("ledger.budget_vs_actual", {"period": "2026-08"}),
    ("ledger.ar_aging", {}),
    ("ledger.saved_search", {"name": "volume_by_card_type"}),
    ("ledger.saved_search", {"name": "chargebacks_by_initiation"}),
    ("ledger.saved_search", {"name": "cash_by_month"}),
]


def test_every_fpa_tool_answers_on_the_processor_world_with_lineage(built):
    db, _ = built
    tools = default_tool_registry()
    start_run(db, "run_proc_tools")
    results = [execute_tool(tools.get(n), a, db_path=db, agent_run_id="run_proc_tools") for n, a in CALLS]
    recorded = lineage_values_for_run(db, "run_proc_tools")
    assert all(r.numbers for r in results)
    assert all(recorded[n.lineage_id] == n.value for r in results for n in r.numbers)
    by_name = dict(zip([n for n, _ in CALLS], results, strict=False))
    scope = by_name["fpa.bridge"].data["scope"]
    assert (scope["entity"], scope["book"], scope["note"]) == (ENTITY, BOOK, NOT_ADVICE)
    flags = {f["measure"]: f["label"] for f in by_name["fpa.variance_flags"].data["flags"]}
    assert flags["net_operating_cash"] == "CRITICAL:" and flags["interchange"] == "FLAG:"
    lines = by_name["ledger.income_statement"].data["lines"]
    assert lines == [
        "Processing fee revenue (captured)",
        "Interchange and network fees",
        "Chargeback losses (written off)",
        "Operating expense",
        "Operating result",
    ]
    assert by_name["ledger.ar_aging"].data["stale_needs_review"] is True  # chargebacks owed 91 to 120 days


def test_the_insurers_measures_and_searches_are_not_found_on_the_processor_world(built):
    db, _ = built
    tools = default_tool_registry()
    start_run(db, "run_proc_missing")
    bridge = execute_tool(
        tools.get("fpa.bridge"), {"measure": "claims_paid", "period": "ytd"}, db_path=db, agent_run_id="x"
    )
    search = execute_tool(
        tools.get("ledger.saved_search"), {"name": "claims_paid_by_channel"}, db_path=db, agent_run_id="x"
    )
    drivers = execute_tool(
        tools.get("fpa.drivers"), {"month": "2026-08", "lob": "personal_auto"}, db_path=db, agent_run_id="x"
    )
    assert (bridge.data["found"], bridge.numbers) == (False, [])
    assert bridge.data["reason"] == "the processor world has no measure 'claims_paid'"
    assert (search.data["found"], search.numbers) == (False, [])
    assert (drivers.data["found"], drivers.numbers) == (False, [])
    assert drivers.data["reason"] == "the processor world has no line or vertical 'personal_auto'"


def test_lineage_row_ids_resolve_to_the_rows_that_make_each_number(built):
    db, _ = built
    start_run(db, "run_proc_rows")
    result = execute_tool(
        default_tool_registry().get("fpa.drivers"),
        {"month": "2026-08", "lob": "travel"},
        db_path=db,
        agent_run_id="run_proc_rows",
    )
    number = next(n for n in result.numbers if n.label == "payment_volume travel 2026-08")
    row_ids, tables = table(
        db, "select row_ids, source_tables from audit.lineage where lineage_id = ?", [number.lineage_id]
    ).row(0)
    assert tables == ["fpa.drivers", "fpa.fpa_processing_monthly"]
    recomputed = table(
        db,
        "select sum(payment_volume) from fpa.fpa_processing_monthly where list_contains(?, row_key)",
        [row_ids[1:]],
    ).item()
    assert recomputed == pytest.approx(number.value)


def agent_deps(db: Path, fake: FakeClient) -> AgentDeps:
    models = ModelsConfig.from_yaml(REPO_ROOT / "config" / "models.yaml")
    llm = LLMClient(models, db, client=fake, sleep=lambda _s: None)
    return AgentDeps(tools=default_tool_registry(), llm=llm, approvals=ApprovalQueue(db), db_path=db)


@pytest.mark.parametrize(
    ("agent_id", "tool", "args"),
    [
        ("fpa.driver", "fpa__drivers", {"month": "2026-08", "lob": "retail"}),
        ("fpa.variance", "fpa__bridge", {"measure": "fee_revenue", "period": "ytd"}),
        ("fpa.scenario", "fpa__scenario", {}),
    ],
)
def test_the_same_fpa_agents_pass_the_lineage_check_on_the_processor(built, agent_id, tool, args):
    # A scripted model (no API key in CI): what is under test is the loop's lineage verification.
    db, _ = built
    agents = AgentRegistry.from_yaml(REPO_ROOT / "config" / "agents.yaml")
    fake = FakeClient([tool_use_message((tool, args)), answer_citing_last_tool("Headline: {0} and {1}.")])
    result = run_agent(agents.get(agent_id), "What changed?", user="test", deps=agent_deps(db, fake))
    assert result.status is RunStatus.COMPLETED, result.warnings


def test_the_token_scenario_on_the_real_world_wins_approvals_and_cuts_chargebacks(built):
    _, run = built
    assert run.scenario.name == "agent_tokens_30"
    summary = {
        m: (b, s) for m, b, s in run.scenario.summary.select("metric", "baseline", "scenario").iter_rows()
    }
    approvals, margin = (
        summary["agent-initiated approvals in 13 weeks"],
        summary["processing margin in 13 weeks"],
    )
    chargebacks = summary["expected chargebacks"]
    assert approvals[1] > approvals[0] and margin[1] > margin[0] and chargebacks[1] < chargebacks[0]
    assert summary["share of agent attempts tokenized"] == (0.0, CONFIG.processor.scenario_token_share)


def test_saving_the_processor_scenario_needs_a_human_approval(scratch):
    save = default_tool_registry().get("fpa.save_plan_version")
    queue = ApprovalQueue(scratch)
    start_run(scratch, "run_proc_save")
    proposal = queue.propose(
        agent_run_id="run_proc_save", action=save.name, payload={"snapshot": scenario_snapshot(scratch)}
    )
    queue.decide(proposal.approval_id, approve=True, decided_by="controller", reason="reviewed the scenario")
    result = execute_tool(
        save, {"approval_id": proposal.approval_id}, db_path=scratch, agent_run_id="run_proc_save"
    )
    assert (result.data["saved"], result.data["scenario"]) == (True, "agent_tokens_30")


def test_an_approval_is_bound_to_the_world_its_scenario_came_from(scratch):
    save = default_tool_registry().get("fpa.save_plan_version")
    queue = ApprovalQueue(scratch)
    start_run(scratch, "run_proc_world")
    snapshot = scenario_snapshot(scratch)
    assert snapshot["variant"] == "processor"
    proposal = queue.propose(agent_run_id="run_proc_world", action=save.name, payload={"snapshot": snapshot})
    queue.decide(proposal.approval_id, approve=True, decided_by="controller", reason="reviewed")
    with connect(scratch) as conn:  # the run metadata now names another world, the scenario tables unchanged
        conn.execute("update fpa.run_meta set variant = 'insurer'")
    with pytest.raises(ApprovalError, match="changed after it was approved"):
        execute_tool(
            save, {"approval_id": proposal.approval_id}, db_path=scratch, agent_run_id="run_proc_world"
        )


def test_the_pipeline_refuses_to_run_the_insurer_on_the_processor_world(built):
    with pytest.raises(WorldMismatch, match="holds the processor world, not the insurer"):
        run_fpa(built[0], CONFIG, variant="insurer", build_dbt=False)


def test_committed_golden_set_matches_generator_truth():
    committed = (GOLDEN_DIR / "fpa.jsonl").read_text(encoding="utf-8")
    regenerated = "".join(
        json.dumps(c, sort_keys=True) + "\n" for c in golden_cases(CONFIG, "processor", SEED)
    )
    assert committed == regenerated, (
        "run `python -m ledgerlens.cli golden fpa --variant processor` and commit"
    )


def test_committed_golden_cases_pass_through_the_tools(built):
    db, _ = built
    cases = load_golden(GOLDEN_DIR, "offline")
    assert {c.tool for c in cases} >= {
        "fpa.drivers",
        "fpa.bridge",
        "ledger.income_statement",
        "ledger.ar_aging",
        "payments.summary",
    }
    results = [run_offline_case(c, tools=default_tool_registry(), db_path=db) for c in cases]
    assert [(r.case_id, r.failures) for r in results if not r.passed] == []


def test_generation_is_identical_across_hash_seeds():
    code = (
        "import hashlib; from pathlib import Path; from ledgerlens.fpa.config import FpaConfig; "
        "from ledgerlens.fpa.processor.generate import generate_world; "
        "w = generate_world(FpaConfig.from_yaml(Path(r'%s')).processor.world, 42); "
        "data = (w.payment_events.write_csv() + w.transactionline.write_csv() "
        "+ w.merchants.write_csv()).encode(); "
        "print(hashlib.sha256(data).hexdigest())"
    ) % (REPO_ROOT / "config" / "fpa.yaml")
    env = dict(os.environ)
    digests = set()
    for seed in ("1", "2"):
        env["PYTHONHASHSEED"] = seed
        out = subprocess.run(
            [sys.executable, "-c", code], env=env, capture_output=True, text=True, check=True
        )
        digests.add(out.stdout.strip())
    assert len(digests) == 1
