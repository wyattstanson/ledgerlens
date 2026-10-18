"""Property tests: the lineage check accepts any honest rounding and rejects real discrepancies."""

from hypothesis import given
from hypothesis import strategies as st

from ledgerlens.core.lineage import NumberRef, verify_answer

money = st.floats(min_value=1e6, max_value=9e11, allow_nan=False, allow_infinity=False)
ratio = st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)


def cite(value: float, unit: str = "USD") -> tuple[list[NumberRef], dict[str, float]]:
    return [NumberRef(label="v", value=value, unit=unit, lineage_id="ln")], {"ln": value}


@given(money)
def test_millions_rounded_for_display_always_pass(value: float):
    numbers, lineage = cite(value)
    assert verify_answer(f"Total spend was ${value / 1e6:.1f}M.", numbers, lineage) == []


@given(money)
def test_comma_formatted_full_values_always_pass(value: float):
    numbers, lineage = cite(value)
    assert verify_answer(f"Total spend was ${value:,.2f}.", numbers, lineage) == []


@given(ratio)
def test_ratios_shown_as_percentages_always_pass(value: float):
    numbers, lineage = cite(value, unit="ratio")
    assert verify_answer(f"Share was {value * 100:.1f}%.", numbers, lineage) == []


@given(money, st.floats(min_value=1.2, max_value=10.0))
def test_values_off_by_more_than_display_precision_always_fail(value: float, factor: float):
    numbers, lineage = cite(value)
    assert verify_answer(f"Total spend was ${value * factor / 1e6:.1f}M.", numbers, lineage)
