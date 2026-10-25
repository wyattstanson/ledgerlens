import math
from pathlib import Path

import pytest

from ledgerlens.core.db import connect
from ledgerlens.core.lineage import (
    NumberRef,
    extract_numeric_tokens,
    lineage_values_for_run,
    record_lineage,
    verify_answer,
)


def ref(value: float, lineage_id: str = "ln_1", unit: str = "USD") -> NumberRef:
    return NumberRef(label="x", value=value, unit=unit, lineage_id=lineage_id)


@pytest.mark.parametrize(
    ("text", "value", "tolerance", "is_percent"),
    [
        ("$2.1M", 2_100_000, 50_000, False),
        ("1,234.56", 1234.56, 0.005, False),
        ("12.3%", 12.3, 0.05, True),
        ("-5", -5, 0.5, False),
        ("40k", 40_000, 500, False),
    ],
)
def test_extracts_value_and_display_precision(text, value, tolerance, is_percent):
    [token] = extract_numeric_tokens(f"Total was {text} this period.")
    assert math.isclose(token.value, value)
    assert math.isclose(token.tolerance, tolerance)
    assert token.is_percent is is_percent


def test_numbers_glued_to_words_are_not_tokens():
    assert extract_numeric_tokens("Q3 results for SE3 and v2") == []


def test_answer_with_only_cited_numbers_passes():
    numbers = [ref(2_134_567.0, "ln_a"), ref(0.123, "ln_b", unit="ratio")]
    lineage = {"ln_a": 2_134_567.0, "ln_b": 0.123}
    assert verify_answer("Savings of $2.1M, a 12.3% cut.", numbers, lineage) == []


def test_planted_unsupported_number_is_rejected():
    # Phase 1 exit criterion: the check rejects a number no tool produced.
    numbers = [ref(2_134_567.0, "ln_a")]
    failures = verify_answer("Savings of $2.1M across 17 vendors.", numbers, {"ln_a": 2_134_567.0})
    assert failures == ["'17' in the answer does not match any cited number"]


def test_rounding_outside_display_precision_is_rejected():
    numbers = [ref(2_160_000.0, "ln_a")]
    assert verify_answer("Savings of $2.1M.", numbers, {"ln_a": 2_160_000.0})


def test_lineage_id_from_another_run_is_rejected():
    failures = verify_answer("Spend was 5.", [ref(5.0, "ln_other")], {"ln_mine": 5.0})
    assert failures == ["x: lineage_id ln_other was not produced by this run"]


def test_real_lineage_id_cannot_vouch_for_an_invented_value():
    failures = verify_answer("Spend was 9.", [ref(9.0, "ln_a")], {"ln_a": 5.0})
    assert failures == ["x: cited value 9.0 differs from recorded value 5.0"]


def test_sign_dropped_in_narration_still_matches():
    assert verify_answer("Cash fell by $2.1M.", [ref(-2_100_000.0)], {"ln_1": -2_100_000.0}) == []


def test_record_lineage_writes_row_and_scopes_lookup_to_run(db: Path):
    with connect(db) as conn:
        conn.execute(
            "insert into audit.tool_calls values ('tc_1', 'run_1', 't', '{}', null, now(), null, 'ok')"
        )
        conn.execute(
            "insert into audit.tool_calls values ('tc_2', 'run_2', 't', '{}', null, now(), null, 'ok')"
        )
    mine = record_lineage(
        db,
        tool_call_id="tc_1",
        label="a",
        value=3.0,
        unit="USD",
        sql_text="select 3",
        source_tables=["t"],
        row_ids=[1, 2],
    )
    record_lineage(
        db,
        tool_call_id="tc_2",
        label="b",
        value=4.0,
        unit="USD",
        sql_text="select 4",
        source_tables=["t"],
        row_ids=[],
    )
    assert lineage_values_for_run(db, "run_1") == {mine.lineage_id: 3.0}
    with connect(db) as conn:
        row = conn.execute(
            "select row_ids, source_tables from audit.lineage where lineage_id = ?", [mine.lineage_id]
        )
        assert row.fetchone() == (["1", "2"], ["t"])


def test_record_lineage_rejects_non_finite_values(db: Path):
    with pytest.raises(ValueError, match="finite"):
        record_lineage(
            db,
            tool_call_id="tc",
            label="a",
            value=float("nan"),
            unit="USD",
            sql_text="s",
            source_tables=[],
            row_ids=[],
        )


@pytest.mark.parametrize(
    "text",
    ["Claims rose USD9400000 versus plan.", "See variance_9400000_ for detail.", "Claims rose 9.4 million."],
)
def test_numbers_glued_to_words_or_written_with_a_scale_word_are_checked(text):
    failures = verify_answer(text, [ref(0.0)], {"ln_1": 0.0})
    assert len(failures) == 1 and "does not match any cited number" in failures[0]


def test_scale_words_after_a_cited_number_pass():
    assert verify_answer("Claims rose 9.4 million.", [ref(9_400_000.0)], {"ln_1": 9_400_000.0}) == []


@pytest.mark.parametrize("text", ["Claims rose by twelve million dollars.", "Over a hundred claims moved."])
def test_numbers_above_ten_in_words_are_rejected(text):
    failures = verify_answer(text, [], {})
    assert failures and all("number in words" in f for f in failures)


def test_small_counts_in_words_still_pass():
    assert verify_answer("Three vendors and one panel.", [], {}) == []


def test_a_percentage_cannot_cite_a_count_or_a_dollar_amount():
    for unit in ("count", "USD"):
        failures = verify_answer("Up 6%.", [ref(6.0, unit=unit)], {"ln_1": 6.0})
        assert failures == ["'6%' in the answer does not match any cited number"], unit
    assert verify_answer("Up 6 percent.", [ref(0.06, unit="ratio")], {"ln_1": 0.06}) == []


def test_a_dollar_amount_must_cite_a_currency():
    failures = verify_answer("Saved $40.", [ref(40.0, unit="count")], {"ln_1": 40.0})
    assert failures == ["'$40' in the answer does not match any cited number"]
