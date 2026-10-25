"""Phase 4 exit criteria (BUILD_PROMPT 10.8) on the full seed-42 world, through dbt and the warehouse.

The forecast criterion follows the ml-pipeline skill's phase reviews; it is tested with fpa/forecast.py.
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
from ledgerlens.core.lineage import lineage_values_for_run, verify_answer
from ledgerlens.core.llm import LLMClient, ModelsConfig
from ledgerlens.core.registry import AgentRegistry
from ledgerlens.core.tools import ToolInputError, execute_tool
from ledgerlens.fpa.config import FpaConfig
from ledgerlens.fpa.drivers import load_marts, net_cash_from_drivers
from ledgerlens.fpa.generate import FpaWorld, generate_world, write_world
from ledgerlens.fpa.golden import golden_cases
from ledgerlens.fpa.pipeline import FpaRun, run_fpa
from ledgerlens.fpa.plan import build_plan
from ledgerlens.fpa.tools import NOT_ADVICE, scenario_snapshot
from ledgerlens.orchestration.fpa_assets import BRIDGE_GAP_SQL
from ledgerlens.packs import default_tool_registry

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG = FpaConfig.from_yaml(REPO_ROOT / "config" / "fpa.yaml")
GOLDEN_DIR = REPO_ROOT / "tests" / "evals" / "golden"
SEED = 42


@pytest.fixture(scope="module")
def built(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, FpaWorld, FpaRun]:
    db = tmp_path_factory.mktemp("fpa") / "warehouse.duckdb"
    ensure_audit_schema(db)
    world = generate_world(CONFIG.world, SEED)
    write_world(db, world, build_plan(CONFIG.world, SEED))
    run = run_fpa(db, CONFIG)  # dbt build first: any failing model or reconciliation test raises here
    return db, world, run


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


def agent_deps(db: Path, fake: FakeClient) -> AgentDeps:
    models = ModelsConfig.from_yaml(REPO_ROOT / "config" / "models.yaml")
    llm = LLMClient(models, db, client=fake, sleep=lambda _s: None)
    return AgentDeps(tools=default_tool_registry(), llm=llm, approvals=ApprovalQueue(db), db_path=db)


def test_ledger_balances_and_reconciles_to_the_payments_feed_every_month(built):
    db, _, _ = built
    unbalanced = table(
        db,
        'select "transaction" from fpa.stg_fpa__transactionline group by 1 having abs(sum(amount)) > 0.005',
    )
    assert unbalanced.is_empty()
    gaps = table(
        db,
        """
        select c.month
        from fpa.fpa_cash_monthly c
        join (select strftime(event_date, '%Y-%m') as month, sum(amount) as premium
              from fpa.stg_fpa__payment_events where kind = 'premium_received' group by 1) f using (month)
        where abs(c.collected_premium - f.premium) > 0.005
        """,
    )
    assert gaps.is_empty()


def test_every_bridge_sums_exactly_to_its_variance(built):
    db, _, run = built
    assert table(db, BRIDGE_GAP_SQL).item() == 0
    worst = (
        run.bridges.filter(pl.col("component") == "residual")
        .group_by("measure")
        .agg(pl.col("cents").abs().max())
    )
    limits = {"net_operating_cash": 8}  # four measures' rounding combined
    assert all(cents <= limits.get(measure, 2) for measure, cents in worst.iter_rows())


def test_driver_tree_reproduces_ledger_net_cash(built):
    db, _, _ = built
    tree = net_cash_from_drivers(table(db, "select * exclude (row_key) from fpa.drivers"))
    ledger = load_marts(db).cash.select("month", "net_operating_cash")
    joined = tree.join(ledger, on="month")
    assert joined.height == ledger.height
    gap = joined.select((pl.col("tree_net_cash") - pl.col("net_operating_cash")).abs().max()).item()
    assert gap < 1e-6


def test_each_planted_deviation_shows_up_where_it_belongs(built):
    db, _, _ = built
    rows = table(db, "select measure, component, cents from fpa.bridge where period = 'ytd'").iter_rows()
    ytd = {(m, c): cents / 100 for m, c, cents in rows}
    premium = {c: v for (m, c), v in ytd.items() if m == "collected_premium"}
    assert max(premium, key=lambda c: abs(premium[c])) == "volume" and premium["volume"] > 0  # new business
    claims = {c: v for (m, c), v in ytd.items() if m == "claims_paid"}
    assert max(claims, key=lambda c: abs(claims[c])) == "rate" and claims["rate"] > 0  # homeowners severity
    assert ytd[("lae", "rate")] > 0  # check handling cost above plan
    assert ytd[("collected_premium", "timing")] > 0  # ACH moved to card, which collects sooner

    planted = CONFIG.world.planted
    mix = table(
        db,
        """
        select method, sum(instalments_due) / sum(sum(instalments_due)) over () as share, 'actual' as basis
        from fpa.fpa_premium_monthly where month >= '2026-01' group by method
        union all
        select method, sum(instalments_due) / sum(sum(instalments_due)) over (), 'plan'
        from raw_fpa.plan_premium where month >= '2026-01' group by method
        """,
    )
    share = {(b, m): s for m, s, b in mix.iter_rows()}
    assert share[("actual", "ach")] == pytest.approx(
        share[("plan", "ach")] * (1 - planted.mix_ach_to_card), abs=0.01
    )
    assert share[("actual", "card")] > share[("plan", "card")]

    check = table(
        db,
        """
        select driver, avg(value) as v from fpa.drivers
        where segment = ? and driver in ('days_to_pay', 'handling_cost')
          and month >= '2026-01'
        group by driver
        """,
        [planted.timing_channel],
    )
    values = dict(check.iter_rows())
    assert values["handling_cost"] == pytest.approx(planted.lae_actual_handling_cost)
    assert values["days_to_pay"] >= planted.timing_actual_lag_median_days - 1


def test_material_variances_carry_adr_0001_statuses(built):
    db, _, _ = built
    rows = table(
        db, "select measure, status, favourable from fpa.bridge_totals where period = 'ytd'"
    ).iter_rows()
    ytd = {m: (s, f) for m, s, f in rows}
    assert ytd["claims_paid"] == ("material", False)
    assert ytd["collected_premium"] == ("material", True)
    assert table(db, "select open_period, materiality_band_ok from fpa.run_meta").row(0) == ("2026-09", True)
    preliminary = table(
        db, "select distinct preliminary, confidence from fpa.bridge_totals where period = '2026-09'"
    )
    assert preliminary.rows() == [(True, "medium")]


def test_every_fpa_tool_returns_numbers_with_lineage(built):
    db, _, _ = built
    tools = default_tool_registry()
    start_run(db, "run_fpa_tools")
    calls = [
        ("fpa.drivers", {"month": "2026-08", "lob": "homeowners"}),
        ("fpa.bridge", {"measure": "claims_paid", "period": "ytd"}),
        ("fpa.variance_flags", {"period": "ytd"}),
        ("fpa.scenario", {}),
        ("payments.events", {"kind": "payout_returned", "limit": 5}),
        ("payments.summary", {"month": "2026-08"}),
        ("ledger.income_statement", {"period": "2026-08"}),
        ("ledger.budget_vs_actual", {"period": "2026-08"}),
        ("ledger.ar_aging", {}),
        ("ledger.saved_search", {"name": "claims_paid_by_channel"}),
    ]
    results = {n: execute_tool(tools.get(n), a, db_path=db, agent_run_id="run_fpa_tools") for n, a in calls}
    recorded = lineage_values_for_run(db, "run_fpa_tools")
    assert all(r.numbers for r in results.values())
    assert all(recorded[n.lineage_id] == n.value for r in results.values() for n in r.numbers)
    assert results["fpa.bridge"].data["scope"]["note"] == NOT_ADVICE
    flags = {f["measure"]: f["label"] for f in results["fpa.variance_flags"].data["flags"]}
    assert flags["net_operating_cash"] == "CRITICAL:" and flags["claims_paid"] == "FLAG:"


def test_saving_a_plan_version_needs_a_human_approval_of_the_current_scenario(scratch):
    db = scratch
    save = default_tool_registry().get("fpa.save_plan_version")
    queue = ApprovalQueue(db)
    start_run(db, "run_fpa_save")

    def execute(approval_id: str):
        return execute_tool(save, {"approval_id": approval_id}, db_path=db, agent_run_id="run_fpa_save")

    proposal = queue.propose(
        agent_run_id="run_fpa_save", action=save.name, payload={"snapshot": scenario_snapshot(db)}
    )
    with pytest.raises(ApprovalError):  # pending is not approved
        execute(proposal.approval_id)
    queue.decide(proposal.approval_id, approve=True, decided_by="controller", reason="reviewed the scenario")
    assert execute(proposal.approval_id).data["saved"] is True
    assert table(db, "select count(*) from fpa.plan_versions").item() == 1
    with pytest.raises(ApprovalError):  # an approval is consumed once
        execute(proposal.approval_id)

    stale = queue.propose(
        agent_run_id="run_fpa_save", action=save.name, payload={"snapshot": {"scenario_hash": "x"}}
    )
    queue.decide(stale.approval_id, approve=True, decided_by="controller", reason="ok")
    with pytest.raises(ApprovalError, match="changed after it was approved"):
        execute(stale.approval_id)
    assert table(db, "select count(*) from fpa.plan_versions").item() == 1


@pytest.mark.parametrize(
    ("agent_id", "tool", "args"),
    [
        ("fpa.driver", "fpa__drivers", {"month": "2026-08", "lob": "personal_auto"}),
        ("fpa.variance", "fpa__bridge", {"measure": "claims_paid", "period": "ytd"}),
        ("fpa.scenario", "fpa__scenario", {}),
    ],
)
def test_fpa_agent_answers_pass_the_lineage_check(built, agent_id, tool, args):
    # A scripted model (no API key in CI): what is under test is the loop's lineage verification.
    db, _, _ = built
    agents = AgentRegistry.from_yaml(REPO_ROOT / "config" / "agents.yaml")
    fake = FakeClient([tool_use_message((tool, args)), answer_citing_last_tool("Headline: {0} and {1}.")])
    result = run_agent(agents.get(agent_id), "What changed?", user="test", deps=agent_deps(db, fake))
    assert result.status is RunStatus.COMPLETED, result.warnings
    system = fake.messages.requests[0]["system"][0]["text"]
    assert "ADR 0001" in system and "Not financial advice" in system


def test_the_forecast_tool_says_when_no_forecast_has_been_generated(scratch):
    start_run(scratch, "run_fpa_fc_none")
    tool = default_tool_registry().get("fpa.forecast")
    result = execute_tool(tool, {}, db_path=scratch, agent_run_id="run_fpa_fc_none")
    assert result.data["found"] is False and result.numbers == []
    assert "forecast fpa" in result.data["reason"]


def test_a_forecast_from_a_form_nobody_reviewed_is_refused(scratch, tmp_path):
    pytest.importorskip("statsforecast.models", reason="statsforecast cannot load here")
    from ledgerlens.fpa.forecast import MODEL_FORMS, run_forecast  # noqa: PLC0415

    forms = json.loads(MODEL_FORMS.read_text(encoding="utf-8"))
    # The label still names the reviewed form, but these orders fit the one AutoARIMA found on more weeks.
    forms["cash_weekly|collected_premium"] |= {"order": [0, 1, 3], "seasonal_order": [0, 0, 0]}
    tampered = tmp_path / "model_forms.json"
    tampered.write_text(json.dumps(forms), encoding="utf-8")
    with pytest.raises(ValueError, match=r"nobody reviewed for \['cash_weekly\|collected_premium'\]"):
        run_forecast(scratch, forms_path=tampered)
    with connect(scratch, read_only=True) as conn:
        stored = conn.execute(
            "select count(*) from information_schema.tables "
            "where table_schema = 'fpa' and table_name = 'forecasts'"
        ).fetchone()
    assert stored == (0,)


def test_forecasts_from_the_reviewed_models_are_stored_and_cited(scratch):
    # statsforecast imports statsmodels, whose compiled module Windows App Control blocks on the dev machine;
    # CI runs this on Linux (ADR 0005).
    pytest.importorskip("statsforecast.models", reason="statsforecast cannot load here")
    from ledgerlens.fpa.forecast import HORIZON, run_forecast  # noqa: PLC0415

    run = run_forecast(scratch)
    assert run.forecasts["unique_id"].nunique() == 12
    assert run.models["deployed_form"].tolist() == run.models["reviewed_form"].tolist()
    counts = run.forecasts.loc[run.forecasts["in_horizon"]].groupby("freq")["unique_id"].value_counts()
    assert set(counts.loc["monthly"]) == {HORIZON["monthly"]} and set(counts.loc["weekly"]) == {
        HORIZON["weekly"]
    }
    assert (run.forecasts["lo-95"] >= 0).all() and (run.forecasts["lo-95"] <= run.forecasts["hi-95"]).all()
    wide = run.totals.pivot(index="ds", columns="measure", values="value")
    net = wide["collected_premium"] - wide["claims_paid"] - wide["lae"] - wide["opex"]
    assert (net - wide["net_operating_cash"]).abs().max() < 0.05  # each total is rounded to the cent
    collection = run.profile.loc[run.profile["kind"] == "collection"].groupby("lob")["share"].sum()
    assert collection.between(0.9, 1.0).all()  # nearly all billed premium is collected within the lags
    again = run_forecast(scratch)
    assert again.forecasts[["unique_id", "ds", "mean"]].equals(run.forecasts[["unique_id", "ds", "mean"]])

    start_run(scratch, "run_fpa_fc")
    result = execute_tool(
        default_tool_registry().get("fpa.forecast"),
        {"view": "totals"},
        db_path=scratch,
        agent_run_id="run_fpa_fc",
    )
    recorded = lineage_values_for_run(scratch, "run_fpa_fc")
    assert len(result.numbers) == 5 * HORIZON["monthly"]
    assert all(recorded[n.lineage_id] == n.value for n in result.numbers)

    agents = AgentRegistry.from_yaml(REPO_ROOT / "config" / "agents.yaml")
    fake = FakeClient(
        [
            tool_use_message(("fpa__forecast", {"view": "totals"})),
            answer_citing_last_tool("Headline: {0} next month."),
        ]
    )
    answer = run_agent(
        agents.get("fpa.forecast"), "What cash is coming?", user="test", deps=agent_deps(scratch, fake)
    )
    assert answer.status is RunStatus.COMPLETED, answer.warnings


def test_scenario_on_the_real_world_cuts_cost_and_days_to_pay(built):
    _, _, run = built
    summary = {
        m: (b, s) for m, b, s in run.scenario.summary.select("metric", "baseline", "scenario").iter_rows()
    }
    lae, days = summary["loss adjustment expense"], summary["claimant days to pay"]
    assert lae[1] < lae[0] and days[1] < days[0]
    assert summary["payouts moved to instant"][1] > 0


def test_committed_golden_set_matches_generator_truth():
    committed = (REPO_ROOT / "tests" / "evals" / "golden" / "fpa.jsonl").read_text(encoding="utf-8")
    cases = golden_cases(generate_world(CONFIG.world, SEED), build_plan(CONFIG.world, SEED), CONFIG)
    regenerated = "".join(json.dumps(c, sort_keys=True) + "\n" for c in cases)
    assert committed == regenerated, "run `python -m ledgerlens.cli golden fpa` and commit the result"


def test_generation_is_identical_across_hash_seeds():
    code = (
        "import hashlib; from pathlib import Path; from ledgerlens.fpa.config import FpaConfig; "
        "from ledgerlens.fpa.generate import generate_world; "
        "w = generate_world(FpaConfig.from_yaml(Path(r'%s')).world, 42); "
        "data = (w.payment_events.write_csv() + w.transactionline.write_csv()).encode(); "
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


def test_committed_golden_cases_pass_through_the_tools(built):
    db, _, _ = built
    cases = [c for c in load_golden(GOLDEN_DIR, "offline") if c.pack == "fpa"]
    results = [run_offline_case(c, tools=default_tool_registry(), db_path=db) for c in cases]
    assert {c.tool for c in cases} >= {
        "fpa.drivers",
        "fpa.bridge",
        "ledger.income_statement",
        "payments.summary",
    }
    assert [(r.case_id, r.failures) for r in results if not r.passed] == []


def test_an_approval_for_tampered_or_moved_scenario_state_is_refused(scratch):
    save = default_tool_registry().get("fpa.save_plan_version")
    queue = ApprovalQueue(scratch)
    start_run(scratch, "run_fpa_tamper")

    def approved(snapshot: dict) -> str:
        proposal = queue.propose(
            agent_run_id="run_fpa_tamper", action=save.name, payload={"snapshot": snapshot}
        )
        queue.decide(proposal.approval_id, approve=True, decided_by="controller", reason="reviewed")
        return proposal.approval_id

    real = scenario_snapshot(scratch)
    shown_wrong = real | {"drivers": [["payout_leg_share", "check", 0.1, 0.0]]}  # real hash, fake drivers
    with pytest.raises(ApprovalError, match="changed after it was approved"):
        execute_tool(
            save, {"approval_id": approved(shown_wrong)}, db_path=scratch, agent_run_id="run_fpa_tamper"
        )

    approval_id = approved(real)
    with connect(scratch) as conn:  # the scenario moves after the human approved it
        conn.execute("update fpa.scenario_drivers set scenario = scenario + 0.01 where segment = 'instant'")
    with pytest.raises(ApprovalError, match="changed after it was approved"):
        execute_tool(save, {"approval_id": approval_id}, db_path=scratch, agent_run_id="run_fpa_tamper")
    assert table(scratch, "select count(*) from fpa.plan_versions").item() == 0


def test_a_failed_consume_rolls_back_the_plan_version(scratch, monkeypatch):
    save = default_tool_registry().get("fpa.save_plan_version")
    queue = ApprovalQueue(scratch)
    start_run(scratch, "run_fpa_rollback")
    proposal = queue.propose(
        agent_run_id="run_fpa_rollback", action=save.name, payload={"snapshot": scenario_snapshot(scratch)}
    )
    queue.decide(proposal.approval_id, approve=True, decided_by="controller", reason="reviewed")

    def fail(self, approval_id, conn=None):
        raise ApprovalError("consumed elsewhere")

    monkeypatch.setattr(ApprovalQueue, "mark_executed", fail)
    with pytest.raises(ApprovalError, match="consumed elsewhere"):
        execute_tool(
            save, {"approval_id": proposal.approval_id}, db_path=scratch, agent_run_id="run_fpa_rollback"
        )
    assert table(scratch, "select count(*) from fpa.plan_versions").item() == 0


def test_lineage_row_ids_resolve_to_the_rows_that_make_each_number(built):
    db, _, _ = built
    start_run(db, "run_fpa_rows")
    result = execute_tool(
        default_tool_registry().get("fpa.drivers"),
        {"month": "2026-08", "lob": "homeowners"},
        db_path=db,
        agent_run_id="run_fpa_rows",
    )
    number = next(n for n in result.numbers if n.label == "due_premium homeowners 2026-08")
    row_ids, tables = table(
        db, "select row_ids, source_tables from audit.lineage where lineage_id = ?", [number.lineage_id]
    ).row(0)
    assert tables == ["fpa.drivers", "fpa.fpa_premium_monthly"]
    assert row_ids[0] == "2026-08|homeowners|due_premium|"
    recomputed = table(
        db,
        "select sum(due_amount) from fpa.fpa_premium_monthly where list_contains(?, row_key)",
        [row_ids[1:]],
    ).item()
    assert recomputed == pytest.approx(number.value)


def test_fiscal_ytd_covers_closed_months_only(built):
    db, _, _ = built
    totals = table(db, "select measure, months, actual_cents from fpa.bridge_totals where period = 'ytd'")
    assert set(totals["months"]) == {8}  # January to August; September is open
    monthly = table(
        db,
        "select measure, sum(actual_cents) as cents from fpa.bridge_totals "
        "where period between '2026-01' and '2026-08' group by measure",
    )
    assert dict(totals.select("measure", "actual_cents").iter_rows()) == dict(monthly.iter_rows())


def test_unknown_periods_get_found_false_and_odd_digits_are_rejected(built):
    db, _, _ = built
    tools = default_tool_registry()
    start_run(db, "run_fpa_period")
    result = execute_tool(
        tools.get("fpa.drivers"), {"month": "2099-12"}, db_path=db, agent_run_id="run_fpa_period"
    )
    assert result.data["found"] is False and result.numbers == []
    with pytest.raises(ToolInputError):
        execute_tool(
            tools.get("fpa.drivers"),
            {"month": "\uff12\uff10\uff12\uff16-08"},
            db_path=db,
            agent_run_id="run_fpa_period",
        )


def test_payments_events_and_aging_say_they_are_preliminary(built):
    db, _, _ = built
    tools = default_tool_registry()
    start_run(db, "run_fpa_prelim")
    events = execute_tool(
        tools.get("payments.events"), {"limit": 3}, db_path=db, agent_run_id="run_fpa_prelim"
    )
    assert events.data["order"] == "most recent first" and events.data["events"][0]["event_ts"] >= "2026-09"
    aging = execute_tool(tools.get("ledger.ar_aging"), {}, db_path=db, agent_run_id="run_fpa_prelim")
    assert aging.data["status"] == "preliminary" and aging.data["buckets_days"] == list(
        CONFIG.aging_buckets_days
    )


def test_the_scenario_input_change_can_be_cited_as_a_percentage(built):
    db, _, _ = built
    start_run(db, "run_fpa_share")
    tool = default_tool_registry().get("fpa.scenario")
    result = execute_tool(tool, {}, db_path=db, agent_run_id="run_fpa_share")
    moved = next(n for n in result.numbers if n.label == "scenario share of check payouts moved")
    assert (moved.value, moved.unit) == (CONFIG.scenario_shift_share, "share")
    recorded = lineage_values_for_run(db, "run_fpa_share")
    assert verify_answer("Moving 30% of check payouts to instant.", [moved], recorded) == []
