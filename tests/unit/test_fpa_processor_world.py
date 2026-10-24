"""The processor world (BUILD_PROMPT 10.1, ``--variant processor``): inputs, generator, plan, raw writes and
tools.

Everything here runs without dbt. The full seed-42 world through dbt and the warehouse is tested in
``tests/integration/test_fpa_processor_exit_criteria.py``.
"""

from datetime import date, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from ledgerlens.core.db import connect, ensure_audit_schema
from ledgerlens.core.tools import execute_tool
from ledgerlens.fpa.config import (
    CARD_TYPES,
    FpaConfig,
    ProcessorPlantedConfig,
    ProcessorWorldConfig,
    Variant,
)
from ledgerlens.fpa.gl import month_bounds
from ledgerlens.fpa.pipeline import ENTITY as INSURER_ENTITY
from ledgerlens.fpa.pipeline import WorldMismatch, revenue_in_band, run_fpa
from ledgerlens.fpa.processor.generate import ProcessorWorld, batch_totals, generate_world
from ledgerlens.fpa.processor.inputs import (
    daily_hazard,
    lag_pmf,
    opening_counts,
    recovery_window_days,
    settlement_days,
)
from ledgerlens.fpa.processor.pipeline import ENTITY
from ledgerlens.fpa.processor.plan import build_plan
from ledgerlens.fpa.processor.scenario import WEEKS, agent_tokens
from ledgerlens.fpa.raw import RAW_SCHEMAS, VARIANT_TABLES, RawSchemaError, raw_tables, world_meta, write_raw
from ledgerlens.packs import default_tool_registry

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG = FpaConfig.from_yaml(REPO_ROOT / "config" / "fpa.yaml")
WORLD = CONFIG.processor.world
NO_PLANTS = WORLD.model_copy(
    update={
        "planted": ProcessorPlantedConfig(
            volume_new_merchant_lift=0,
            card_mix_debit_to_premium=0,
            interchange_pct_lift=0,
            agent_share_actual_end=WORLD.agent_share_end,
            agent_auth_rate_drop=0,
        )
    }
)


@pytest.fixture(scope="module")
def world() -> ProcessorWorld:
    return generate_world(WORLD, 42)


def ordinal(day: str) -> int:
    return date.fromisoformat(day).toordinal()


def total(frame: pl.DataFrame, column: str) -> float:
    return float(frame[column].sum())


# --- inputs ------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("captured", "settled"),
    [
        ("2026-09-07", "2026-09-08"),  # Monday -> Tuesday
        ("2026-09-10", "2026-09-11"),  # Thursday -> Friday
        ("2026-09-11", "2026-09-14"),  # Friday -> Monday
        ("2026-09-12", "2026-09-14"),  # Saturday -> Monday
        ("2026-09-13", "2026-09-14"),  # Sunday -> Monday
    ],
)
def test_batches_settle_one_business_day_later(captured, settled):
    assert settlement_days(np.array([ordinal(captured)]), 1)[0] == ordinal(settled)


def test_two_business_days_from_a_friday_is_tuesday():
    assert settlement_days(np.array([ordinal("2026-09-11")]), 2)[0] == ordinal("2026-09-15")


def test_lag_pmf_is_a_distribution_over_whole_days():
    pmf = lag_pmf(38, 0.45, 1_000)
    assert pmf[0] == 0.0 and pmf.sum() == pytest.approx(1.0, abs=1e-9)
    assert lag_pmf(9.5, 0.001, 30)[10] == pytest.approx(1.0)  # a lag of 9.5 days is paid on day ten


def test_opening_book_follows_the_shares_and_adds_up():
    config = WORLD.model_copy(update={"merchants": 101})
    assert opening_counts(config) == [56, 15, 30]  # 55.55 / 15.15 / 30.3, the largest remainder rounds up


def test_config_rejects_shares_that_do_not_sum_to_one():
    bad = dict(WORLD.verticals) | {
        "travel": WORLD.verticals["travel"].model_copy(update={"merchant_share": 0.2})
    }
    with pytest.raises(ValueError, match="verticals shares must sum to 1"):
        ProcessorWorldConfig.model_validate({"verticals": {k: v.model_dump() for k, v in bad.items()}})


def test_config_needs_every_vertical():
    two = {k: v.model_dump() for k, v in WORLD.verticals.items() if k != "travel"}
    two["retail"]["merchant_share"] = 0.70  # retail and digital goods still add up to one
    with pytest.raises(ValueError, match="must list each of"):
        ProcessorWorldConfig.model_validate({"verticals": two})


def test_config_refuses_a_planted_drop_that_leaves_no_approvals():
    with pytest.raises(ValueError, match="leaves no approvals"):
        ProcessorWorldConfig(planted=ProcessorPlantedConfig(agent_auth_rate_drop=0.9))


# --- plan and plants ---------------------------------------------------------------------------------------


def test_the_plan_is_an_unbiased_expectation_of_the_world_without_plants():
    plan = build_plan(NO_PLANTS)
    proc, cb = plan["plan_processing"], plan["plan_chargebacks"]
    start, end = date(2023, 10, 1), date(2026, 9, 30)
    gaps = []
    for seed in range(8):
        w = generate_world(NO_PLANTS, seed)
        captured = w.captures.filter(pl.col("capture_date") >= start)
        settled = w.captures.filter(pl.col("settle_date").is_between(start, end))
        raised = w.chargebacks.filter(pl.col("raised_date") >= start)
        gaps.append(
            [
                total(captured, "attempts") / total(proc, "attempts") - 1,
                total(captured, "volume") / total(proc, "payment_volume") - 1,
                total(settled, "fee_amount") / total(proc, "fees_settled") - 1,
                total(settled, "interchange_amount") / total(proc, "interchange_settled") - 1,
                raised.height / total(cb, "chargebacks") - 1,
            ]
        )
    mean = np.array(gaps).mean(axis=0)
    # Eight seeds' spread is about two percent each, so the mean of an unbiased plan sits well inside this.
    assert np.abs(mean).max() < 0.02, mean


FLAT = WORLD.model_copy(
    update={
        "verticals": {
            k: v.model_copy(update={"seasonal_amplitude": 0.0, "weekend_factor": 1.0})
            for k, v in WORLD.verticals.items()
        },
        "new_merchants_per_month": 0.0,
    }
)


def test_the_plan_settles_each_days_fees_on_its_settlement_day():
    # With no seasonality, weekend pattern or churn, a month's fees are captured evenly over its days, so the
    # fees settled in a month can be recomputed day by day with the T+1 business-day rule written out here.
    proc = build_plan(FLAT.model_copy(update={"merchant_churn_monthly": 0.0}))["plan_processing"]
    captured = proc.group_by("month").agg(pl.col("fees_captured").sum(), pl.col("fees_settled").sum())
    per_day = {m: f / month_bounds(m)[1].day for m, f, _ in captured.iter_rows()}
    first = month_bounds(min(per_day))[0]
    expected: dict[str, float] = {}
    day = first - timedelta(days=7)
    while day <= month_bounds(max(per_day))[1]:
        rate = per_day.get(day.strftime("%Y-%m"), per_day[first.strftime("%Y-%m")])  # before: the start
        settles = day + timedelta(days={4: 3, 5: 2}.get(day.weekday(), 1))  # Fri +3, Sat +2, else +1
        if first <= settles:
            key = settles.strftime("%Y-%m")
            expected[key] = expected.get(key, 0.0) + rate
        day += timedelta(days=1)
    for month, _, settled in captured.iter_rows():
        assert settled == pytest.approx(expected[month], abs=1.0), month


def test_the_plan_keeps_merchants_on_the_generators_daily_survival():
    plan = build_plan(FLAT)["plan_merchants"].filter(pl.col("lob") == "travel")
    start = month_bounds("2023-10")[0]  # the window start; the opening travel book is 360
    for month, active in plan.select("month", "active_merchants").iter_rows():
        days = (month_bounds(month)[1] - start).days
        assert active == pytest.approx(360 * np.exp(-daily_hazard(FLAT) * days), abs=1e-3), month


def test_the_plan_needs_no_seed_and_is_deterministic():
    a, b = build_plan(WORLD), build_plan(WORLD)
    assert all(a[k].equals(b[k]) for k in a)
    assert set(a) == {"plan_processing", "plan_chargebacks", "plan_merchants", "plan_opex"}


def _share(frame: pl.DataFrame, where: pl.Expr, numerator: pl.Expr, denominator: str) -> float:
    picked = frame.filter(where)
    return total(picked.filter(numerator), denominator) / total(picked, denominator)


def test_debit_attempts_move_to_premium_credit_from_the_planted_month(world):
    month = pl.col("capture_date").dt.strftime("%Y-%m")
    before, after = month < "2024-10", month >= "2024-10"  # the window starts 2023-10; the plant at month 12
    debit = pl.col("card_type") == "debit"
    assert _share(world.captures, before, debit, "attempts") == pytest.approx(0.38, abs=0.005)
    assert _share(world.captures, after, debit, "attempts") == pytest.approx(0.38 * 0.8, abs=0.005)


def test_premium_interchange_rises_by_the_planted_lift(world):
    premium = world.captures.filter((pl.col("card_type") == "premium_credit") & (pl.col("approvals") > 0))
    pct = (pl.col("interchange_amount") - 0.10 * pl.col("approvals")).sum() / pl.col("volume").sum()
    month = pl.col("capture_date").dt.strftime("%Y-%m")
    before = premium.filter(month < "2026-01").select(pct).item()  # month 27 of the window is 2026-01
    after = premium.filter(month >= "2026-01").select(pct).item()
    assert before == pytest.approx(0.0210 + 0.0014, abs=2e-5)
    assert after == pytest.approx(0.0210 + 0.0014 + 0.0015, abs=2e-5)


def test_agent_traffic_grows_to_the_planted_share_and_issuers_decline_more_of_it(world):
    sept = world.captures.filter(pl.col("capture_date") >= date(2026, 9, 1))
    agent = pl.col("initiation") == "agent"
    assert _share(sept, pl.lit(True), agent, "attempts") == pytest.approx(0.15, abs=0.005)
    late = world.captures.filter(agent & (pl.col("capture_date") >= date(2025, 10, 1)))  # month 24
    early = world.captures.filter(agent & (pl.col("capture_date") < date(2025, 10, 1)))
    assert late["approvals"].sum() / late["attempts"].sum() == pytest.approx(0.86 - 0.08, abs=0.005)
    assert early["approvals"].sum() / early["attempts"].sum() == pytest.approx(0.86, abs=0.005)


def test_digital_goods_onboards_twice_as_many_merchants_from_the_planted_month(world):
    new = world.merchants.filter(
        (pl.col("lob") == "digital_goods") & (pl.col("onboard_date") >= date(2023, 10, 1))
    ).with_columns(pl.col("onboard_date").dt.strftime("%Y-%m").alias("month"))
    per_month = new.group_by("month").len()
    before = per_month.filter(pl.col("month") < "2025-01")["len"].mean()  # month 15 of the window
    after = per_month.filter(pl.col("month") >= "2025-01")["len"].mean()
    assert isinstance(before, float) and isinstance(after, float)
    assert after / before == pytest.approx(2.0, abs=0.35)  # Poisson months around 15 and 30


# --- ledger and feed, before dbt ---------------------------------------------------------------------------


def test_every_chargeback_belongs_to_a_merchant_active_on_the_purchase_date(world):
    joined = world.chargebacks.join(
        world.merchants.select("merchant_id", "lob", "onboard_date", "churn_date"), on="merchant_id"
    )
    assert joined.height == world.chargebacks.height
    assert (joined["lob"] == joined["lob_right"]).all()
    inactive = joined.filter(
        (pl.col("onboard_date") > pl.col("purchase_date"))
        | (pl.col("churn_date").is_not_null() & (pl.col("churn_date") <= pl.col("purchase_date")))
    )
    assert inactive.is_empty()


def test_every_transaction_balances_and_every_cash_posting_is_its_event(world):
    lines = world.transactionline
    imbalance = lines.group_by("transaction").agg(pl.col("amount").sum().abs())
    assert imbalance.filter(pl.col("amount") > 0.005).is_empty()
    cash = (
        lines.filter(pl.col("account") == 1000)
        .join(world.transaction.select("id", "custbody_event_id"), left_on="transaction", right_on="id")
        .filter(pl.col("custbody_event_id").is_not_null())
    )
    signed = world.payment_events.select(
        "event_id",
        pl.when(pl.col("direction") == "out")
        .then(-pl.col("amount"))
        .otherwise(pl.col("amount"))
        .alias("event_cash"),
    )
    joined = cash.join(signed, left_on="custbody_event_id", right_on="event_id", how="full", coalesce=True)
    assert joined.height == world.payment_events.height
    assert joined.select((pl.col("amount") - pl.col("event_cash")).abs().max()).item() < 0.005


def test_receivable_accounts_equal_the_open_items_at_the_end(world):
    as_of = date(2026, 9, 30)
    balance = dict(
        world.transactionline.filter(pl.col("account").is_in([1100, 1300]))
        .group_by("account")
        .agg(pl.col("amount").sum())
        .iter_rows()
    )
    unsettled = batch_totals(world.captures).filter(pl.col("settle_date") > as_of)["net"].sum()
    owed = world.chargebacks.filter(pl.col("recovered_date").is_null() & pl.col("written_off_date").is_null())
    assert balance[1100] == pytest.approx(unsettled, abs=0.005)
    assert balance[1300] == pytest.approx(owed["amount"].sum(), abs=0.005)


def test_unrecovered_chargebacks_are_written_off_at_the_configured_age(world):
    off = world.chargebacks.filter(pl.col("written_off_date").is_not_null())
    ages = off.select((pl.col("written_off_date") - pl.col("raised_date")).dt.total_days()).to_series()
    assert ages.unique().to_list() == [WORLD.write_off_days]
    assert off.filter(pl.col("recovered_date").is_not_null()).is_empty()


def test_revenue_sits_inside_the_adr_0001_band(world):
    revenue, net, ok = revenue_in_band(world.meta)
    assert ok and net is not None and 10e6 <= net < revenue <= 100e6  # gross and net, both inside the band
    assert revenue_in_band({"annual_revenue": 40e6}) == (40e6, None, True)  # the insurer reports no net
    assert revenue_in_band({"annual_revenue": 40e6, "annual_net_revenue": 4e6})[2] is False


# --- raw schema and switching worlds -----------------------------------------------------------------------


def empty_frames(variant: Variant) -> dict[str, pl.DataFrame]:
    return {name: pl.DataFrame(schema=RAW_SCHEMAS[name]) for name in VARIANT_TABLES[variant]}


def test_a_world_must_write_exactly_its_tables():
    frames = empty_frames("processor")
    frames.pop("captures")
    with pytest.raises(RawSchemaError, match=r"missing \['captures'\]"):
        raw_tables("processor", frames)


def test_a_frame_that_drifts_from_the_declared_schema_is_refused():
    frames = empty_frames("processor")
    frames["captures"] = frames["captures"].with_columns(pl.col("attempts").cast(pl.Float64))
    with pytest.raises(RawSchemaError, match=r"captures: columns \['attempts'\]"):
        raw_tables("processor", frames)


def test_loading_one_world_empties_the_other_worlds_tables(tmp_path):
    db = tmp_path / "w.duckdb"
    processor = empty_frames("processor")
    processor["merchants"] = pl.DataFrame(
        {"merchant_id": ["M-1"], "lob": ["retail"], "onboard_date": [date(2026, 1, 1)], "churn_date": [None],
         "attempts_per_day": [3.0]},
        schema=RAW_SCHEMAS["merchants"],
    )  # fmt: skip
    write_raw(db, "processor", processor, {"seed": 1.0})
    assert world_meta(db)["variant"] == "processor"
    write_raw(db, "insurer", empty_frames("insurer"), {"seed": 1.0})
    with connect(db) as conn:
        assert conn.execute("select count(*) from raw_fpa.merchants").fetchone() == (0,)
        assert conn.execute("select variant from raw_fpa.world_meta").fetchone() == ("insurer",)


def test_a_warehouse_without_a_world_says_so(tmp_path):
    db = tmp_path / "w.duckdb"
    with connect(db) as conn:
        conn.execute("create schema raw_fpa")
        conn.execute("create table raw_fpa.world_meta as select 42 as seed")
    with pytest.raises(RuntimeError, match="names no FP&A world"):
        world_meta(db)


def test_loading_a_world_makes_the_tools_wait_for_the_pipeline(tiny):
    # The tools answered for the processor run; a new world is loaded and the pipeline has not run on it.
    tools = default_tool_registry()
    write_raw(tiny, "insurer", empty_frames("insurer"), {"seed": 2.0})
    calls = [
        ("payments.summary", {"month": "2026-08"}),
        ("ledger.saved_search", {"name": "cash_by_month"}),
        ("fpa.scenario", {}),
    ]
    for name, args in calls:
        with pytest.raises(RuntimeError, match="has not run"):
            execute_tool(tools.get(name), args, db_path=tiny, agent_run_id="r")


def test_the_pipeline_stops_if_another_world_is_loaded_while_dbt_runs(tmp_path, monkeypatch):
    db = tmp_path / "w.duckdb"
    write_raw(db, "insurer", empty_frames("insurer"), {"seed": 1.0})

    def another_process_loads_the_processor(warehouse: Path, select: str = "") -> None:
        write_raw(warehouse, "processor", empty_frames("processor"), {"seed": 1.0})

    monkeypatch.setattr("ledgerlens.fpa.pipeline.run_dbt", another_process_loads_the_processor)
    with pytest.raises(WorldMismatch, match="changed from insurer to processor while dbt ran"):
        run_fpa(db, CONFIG)


def test_the_pipeline_refuses_to_run_on_another_world(tmp_path):
    db = tmp_path / "w.duckdb"
    write_raw(db, "insurer", empty_frames("insurer"), {"seed": 1.0})
    with pytest.raises(WorldMismatch, match="holds the insurer world, not the processor"):
        run_fpa(db, CONFIG, variant="processor")  # before dbt runs: nothing is built


# --- the scenario, on hand-built captures ------------------------------------------------------------------

AS_OF = date(2026, 9, 30)


def captures(agent_approvals: int = 70) -> pl.DataFrame:
    rows = []
    for day in range(7 * WEEKS):
        d = AS_OF - timedelta(days=day)
        for who, approvals in (("human", 90), ("agent", agent_approvals)):
            volume = approvals * 50.0
            rows.append(
                {
                    "capture_date": d,
                    "lob": "retail",
                    "card_type": "credit",
                    "initiation": who,
                    "attempts": 100,
                    "approvals": approvals,
                    "volume": volume,
                    "fee_amount": volume * 0.03,
                    "interchange_amount": volume * 0.018,
                }
            )
    return pl.DataFrame(rows)


def disputes(human: int = 18, agent: int = 42) -> pl.DataFrame:
    rows = [
        {"lob": "retail", "initiation": who, "raised_date": AS_OF - timedelta(days=i % 80), "amount": 50.0,
         "recovered_date": None if i % 5 == 0 else AS_OF}
        for who, n in (("human", human), ("agent", agent))
        for i in range(n)
    ]  # fmt: skip
    return pl.DataFrame(rows, schema_overrides={"recovered_date": pl.Date})


def metric(result, name: str) -> dict:
    return result.summary.filter(pl.col("metric") == name).row(0, named=True)


def test_tokenizing_agent_attempts_approves_and_disputes_them_like_human_ones():
    result = agent_tokens(captures(), disputes(), AS_OF, share=0.5, recovery_days=0)
    # Thirteen Monday-to-Sunday weeks ending in the as-of date's week: 87 days up to Wednesday 30 September.
    days = (AS_OF - (AS_OF - timedelta(days=AS_OF.weekday() + 7 * (WEEKS - 1)))).days + 1
    assert days == 87
    attempts = 100 * days
    extra = 0.5 * attempts * (0.9 - 0.7)
    assert metric(result, "agent-initiated approvals in 13 weeks")["change"] == pytest.approx(extra)
    assert metric(result, "processing margin in 13 weeks")["change"] == pytest.approx(extra * 50 * 0.012)
    human_rate, agent_approvals = 18 / (90 * days), 70 * days
    expected = 18 + 0.5 * 42 + 0.5 * attempts * 0.9 * human_rate
    assert metric(result, "expected chargebacks")["baseline"] == pytest.approx(60)  # as observed
    assert metric(result, "expected chargebacks")["scenario"] == pytest.approx(expected)
    unrepaid = sum(1 for n in (18, 42) for i in range(n) if i % 5 == 0)  # one in five not repaid
    net_per = unrepaid * 50 / 60
    assert metric(result, "expected chargeback net cost")["scenario"] == pytest.approx(expected * net_per)
    drivers = {r["driver"]: (r["baseline"], r["scenario"]) for r in result.drivers.iter_rows(named=True)}
    assert drivers["authorization_rate"] == pytest.approx((0.7, 0.7 + 0.5 * 0.2))
    assert drivers["chargeback_rate"][0] == pytest.approx(42 / agent_approvals)
    assert drivers["tokenized_share"] == (0.0, 0.5)


def test_weekly_margin_covers_thirteen_weeks_and_adds_the_gain():
    result = agent_tokens(captures(), disputes(), AS_OF, share=0.3, recovery_days=0)
    assert result.weekly.height == WEEKS and result.weekly["week_start"][0].weekday() == 0
    gain = total(result.weekly, "scenario_margin") - total(result.weekly, "baseline_margin")
    assert gain == pytest.approx(metric(result, "processing margin in 13 weeks")["change"], abs=0.1)


def test_when_agents_already_authorize_like_people_nothing_moves():
    result = agent_tokens(captures(agent_approvals=90), disputes(agent=18), AS_OF, share=0.3, recovery_days=0)
    unchanged = result.summary.filter(pl.col("metric") != "share of agent attempts tokenized")
    assert unchanged["change"].abs().max() == pytest.approx(0.0, abs=1e-9)
    assert result.name == "agent_tokens_30"


def test_a_repayment_still_in_flight_is_not_counted_as_a_loss():
    # Every dispute of the last ten days is still unrepaid; older ones were repaid four times in five.
    recent = pl.DataFrame(
        [{"lob": "retail", "initiation": "agent", "raised_date": AS_OF - timedelta(days=i), "amount": 50.0,
          "recovered_date": None} for i in range(10)],
        schema_overrides={"recovered_date": pl.Date},
    )  # fmt: skip
    both = pl.concat([disputes().filter(pl.col("raised_date") <= AS_OF - timedelta(days=10)), recent])
    settled = agent_tokens(captures(), both, AS_OF, 0.3, recovery_days=10)
    in_flight = agent_tokens(captures(), both, AS_OF, 0.3, recovery_days=0)
    count = metric(settled, "expected chargebacks")["baseline"]
    assert (
        count == metric(in_flight, "expected chargebacks")["baseline"] == both.height
    )  # all are chargebacks
    older = both.filter(pl.col("raised_date") <= AS_OF - timedelta(days=10))
    per_chargeback = older.filter(pl.col("recovered_date").is_null())["amount"].sum() / older.height
    assert metric(settled, "expected chargeback net cost")["baseline"] == pytest.approx(
        count * per_chargeback
    )
    assert metric(in_flight, "expected chargeback net cost")["baseline"] > count * per_chargeback


def test_the_recovery_window_covers_all_but_one_repayment_in_a_thousand():
    assert recovery_window_days(WORLD) == 26  # 4 days x exp(3.09 x 0.6) = 25.5


def test_no_recent_captures_is_an_error_not_a_zero():
    old = captures().with_columns(pl.col("capture_date") - pl.duration(days=400))
    with pytest.raises(RuntimeError, match="latest 13 weeks"):
        agent_tokens(old, disputes(), AS_OF, 0.3, recovery_days=0)


# --- tools on a hand-built processor warehouse -------------------------------------------------------------


@pytest.fixture
def tiny(tmp_path: Path) -> Path:
    db = tmp_path / "w.duckdb"
    ensure_audit_schema(db)
    write_raw(db, "processor", empty_frames("processor"), {"seed": 1.0})
    with connect(db) as conn:
        conn.execute("create schema if not exists fpa")
        conn.execute("create table fpa.bridge_totals as select '2026-08' as period")
        conn.execute(
            "create table fpa.run_meta as select 'processor' as variant, ? as entity, 'cash' as book, "
            "'as captured' as income_basis, '2026-09-30' as as_of, '2026-09' as open_period, "
            "'2026-08' as latest_closed, [30, 60, 90] as aging_buckets_days",
            [ENTITY],
        )
        conn.execute(
            "create table fpa.stg_fpa__payment_events as select * from (values "
            "('PE-1', 'settlement', 'in', 'debit', 100.0), ('PE-2', 'settlement', 'out', 'debit', 7.0), "
            "('PE-3', 'chargeback', 'out', 'credit', 50.0)) "
            "t(event_id, kind, direction, instrument, amount)"
        )
        conn.execute(
            "alter table fpa.stg_fpa__payment_events add column event_date date default date '2026-08-03'"
        )
    return db


def test_a_kind_moving_money_both_ways_labels_each_direction(tiny):
    result = execute_tool(
        default_tool_registry().get("payments.summary"), {"month": "2026-08"}, db_path=tiny, agent_run_id="r"
    )
    values = {n.label: n.value for n in result.numbers}
    assert values["settlement debit 2026-08 in amount"] == 100.0
    assert values["settlement debit 2026-08 out amount"] == 7.0
    assert values["chargeback credit 2026-08 amount"] == 50.0  # one direction: no suffix
    assert result.data["scope"]["entity"] == ENTITY != INSURER_ENTITY


def test_a_saved_search_from_the_other_world_is_not_found(tiny):
    result = execute_tool(
        default_tool_registry().get("ledger.saved_search"),
        {"name": "premium_collected_by_method"},
        db_path=tiny,
        agent_run_id="r",
    )
    assert result.data == {
        "found": False,
        "name": "premium_collected_by_method",
        "reason": "the processor world has no saved search 'premium_collected_by_method'",
    }
    assert result.numbers == []


def test_every_card_type_is_in_the_world(world):
    assert sorted(world.captures["card_type"].unique()) == sorted(CARD_TYPES)
