"""ADR 0001 materiality at its boundaries: reaching a threshold counts, and the screens apply in order."""

import pytest
from hypothesis import given
from hypothesis import strategies as st

from ledgerlens.fpa.config import MaterialityConfig
from ledgerlens.fpa.materiality import Variance, classify, flag, level_for, risk_label

CONFIG = MaterialityConfig()
LEVELS = {"financial_statement": CONFIG.financial_statement, "operating": CONFIG.operating}
plans = st.floats(1.0, 1e8, allow_nan=False)


@pytest.mark.parametrize("level", list(LEVELS))
def test_dollar_threshold_is_inclusive(level):
    usd = LEVELS[level].usd
    big_plan = usd * 1_000  # percentages far below every threshold
    assert classify(usd, big_plan, LEVELS[level], CONFIG.immaterial) == "material"
    assert classify(-usd, big_plan, LEVELS[level], CONFIG.immaterial) == "material"
    assert classify(usd - 0.01, big_plan, LEVELS[level], CONFIG.immaterial) == "watch"


@pytest.mark.parametrize("level", list(LEVELS))
@given(plan=plans)
def test_percentage_threshold_is_inclusive(level, plan):
    threshold = LEVELS[level]
    at = threshold.pct * plan
    if at < threshold.usd:  # otherwise the dollar rule decides
        assert classify(at, plan, threshold, CONFIG.immaterial) == "material"
        assert classify(at * (1 - 1e-9), plan, threshold, CONFIG.immaterial) != "material"


@given(plan=plans, variance=st.floats(-1e8, 1e8, allow_nan=False))
def test_status_matches_the_rule_as_written(plan, variance):
    level = CONFIG.operating
    status = classify(variance, plan, level, CONFIG.immaterial)
    size = abs(variance)
    if size >= level.usd or size >= level.pct * plan:
        assert status == "material"
    elif size < CONFIG.immaterial.usd and size < CONFIG.immaterial.pct * plan:
        assert status == "immaterial"
    else:
        assert status == "watch"


@given(plan=plans, a=st.floats(0, 1e8), b=st.floats(0, 1e8))
def test_a_bigger_variance_is_never_less_material(plan, a, b):
    rank = {"immaterial": 0, "watch": 1, "material": 2}
    small, large = sorted((a, b))
    level = CONFIG.financial_statement
    assert (
        rank[classify(small, plan, level, CONFIG.immaterial)]
        <= rank[classify(large, plan, level, CONFIG.immaterial)]
    )


def test_zero_plan_uses_the_dollar_rules_only():
    level = CONFIG.operating
    assert classify(0.0, 0.0, level, CONFIG.immaterial) == "immaterial"
    assert classify(5_000.0, 0.0, level, CONFIG.immaterial) == "immaterial"
    assert classify(12_000.0, 0.0, level, CONFIG.immaterial) == "watch"
    assert classify(25_000.0, 0.0, level, CONFIG.immaterial) == "material"


def test_documented_one_time_items_and_reclassifications_are_not_flagged():
    claims = Variance("2026-03", "claims_paid", "expense", "operating", 1_100_000, 1_000_000)
    lae = Variance("2026-03", "lae", "expense", "operating", 900_500, 1_000_000)  # offsets claims within 2%
    opex = Variance("2026-03", "opex", "expense", "operating", 1_200_000, 1_000_000)
    premium = Variance("2026-04", "collected_premium", "revenue", "operating", 1_100_000, 1_000_000)
    statuses = flag([claims, lae, opex, premium], CONFIG, one_time_items=("2026-03:opex",))
    assert statuses == ["reclass", "reclass", "one_time", "material"]


def test_offsets_in_another_month_or_against_revenue_do_not_count_as_reclass():
    claims = Variance("2026-03", "claims_paid", "expense", "operating", 1_100_000, 1_000_000)
    later = Variance("2026-04", "lae", "expense", "operating", 900_000, 1_000_000)
    revenue = Variance("2026-03", "collected_premium", "revenue", "operating", 900_000, 1_000_000)
    assert flag([claims, later, revenue], CONFIG, ())[0] == "material"


def test_favourable_follows_the_kind():
    assert Variance("m", "p", "revenue", "operating", 11, 10).favourable
    assert not Variance("m", "c", "expense", "operating", 11, 10).favourable
    assert Variance("m", "c", "expense", "operating", 9, 10).favourable


def test_only_material_variances_get_a_risk_label_and_net_cash_is_statement_level():
    assert level_for("net_operating_cash") == "financial_statement"
    assert level_for("claims_paid") == "operating"
    assert risk_label("material", "financial_statement") == "CRITICAL:"
    assert risk_label("material", "operating") == "FLAG:"
    assert [risk_label(s, "operating") for s in ("watch", "immaterial", "one_time", "reclass")] == [None] * 4
