"""Variance bridge: the parts sum to the variance to the cent, and each effect lands in its own component."""

from hypothesis import given
from hypothesis import strategies as st

from ledgerlens.fpa.bridge import COMPONENTS, Segment, combine, variance_bridge


def seg(key: str, qa: float, pa: float, qp: float, pp: float) -> Segment:
    return Segment(key, qa, qa * pa, qp, qp * pp)


segment = st.builds(
    seg,
    key=st.text(min_size=1, max_size=3),
    qa=st.integers(0, 5_000).map(float),
    pa=st.floats(0, 20_000, allow_nan=False),
    qp=st.integers(0, 5_000).map(float),
    pp=st.floats(0, 20_000, allow_nan=False),
)
cash = st.floats(-5e7, 5e7, allow_nan=False)


@given(st.lists(segment, min_size=1, max_size=12), cash, cash)
def test_bridge_sums_exactly_to_the_variance_and_the_residual_is_only_rounding(segments, cash_a, cash_p):
    bridge = variance_bridge(segments, cash_a, cash_p)
    assert set(bridge.components) == set(COMPONENTS)
    assert sum(bridge.components.values()) == bridge.variance_cents
    # Volume + mix + rate is an identity; four parts rounded to cents leave at most two cents over.
    assert abs(bridge.components["residual"]) <= 2


@given(
    st.lists(segment, min_size=1, max_size=6),
    cash,
    cash,
    st.lists(segment, min_size=1, max_size=6),
    cash,
    cash,
)
def test_signed_combinations_still_sum_exactly(seg_a, ca, pa, seg_b, cb, pb):
    a, b = variance_bridge(seg_a, ca, pa), variance_bridge(seg_b, cb, pb)
    net = combine({"in": a, "out": b}, {"in": 1, "out": -1})
    assert net.variance_cents == a.variance_cents - b.variance_cents
    assert sum(net.components.values()) == net.variance_cents


def event_basis(segments: list[Segment]) -> tuple[float, float]:
    return sum(s.amount_actual for s in segments), sum(s.amount_plan for s in segments)


def test_more_of_the_same_is_pure_volume():
    segments = [seg("a", 120, 100, 100, 100), seg("b", 60, 50, 50, 50)]
    bridge = variance_bridge(segments, *event_basis(segments))
    expected = (180 - 150) * (12_500 / 150)  # extra quantity at the plan's average price
    assert bridge.components["volume"] == round(expected * 100)
    assert bridge.components["rate"] == 0 and bridge.components["timing"] == 0
    assert abs(bridge.components["mix"] + bridge.components["residual"]) <= 1


def test_a_higher_price_is_pure_rate():
    segments = [seg("a", 100, 110, 100, 100)]
    bridge = variance_bridge(segments, *event_basis(segments))
    assert bridge.components["rate"] == 100_000
    assert bridge.components["volume"] == 0 and bridge.components["mix"] == 0


def test_a_shift_to_the_dearer_segment_is_pure_mix():
    segments = [seg("cheap", 50, 10, 100, 10), seg("dear", 150, 30, 100, 30)]
    bridge = variance_bridge(segments, *event_basis(segments))
    assert bridge.components["volume"] == 0 and bridge.components["rate"] == 0
    assert bridge.components["mix"] == (50 * 30 - 50 * 10) * 100


def test_cash_lagging_events_is_timing():
    segments = [seg("a", 100, 100, 100, 100)]
    bridge = variance_bridge(segments, cash_actual=9_000, cash_plan=9_500)  # events equal, collections lag
    assert bridge.components["timing"] == -50_000
    assert bridge.variance_cents == -50_000


def test_a_segment_without_plan_quantity_is_priced_at_actual_so_it_is_mix_not_rate():
    segments = [seg("old", 100, 100, 100, 100), seg("new", 10, 70, 0, 0)]
    bridge = variance_bridge(segments, *event_basis(segments))
    assert bridge.components["rate"] == 0
    assert sum(bridge.components.values()) == 70_000
