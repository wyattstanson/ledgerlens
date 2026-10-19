"""Property tests: BH never loosens a p-value or reorders it; suppression never leaks a small cell."""

import itertools

import polars as pl
from hypothesis import given
from hypothesis import strategies as st

from ledgerlens.core.stats import benjamini_hochberg, cohens_kappa
from ledgerlens.interviews.pipeline import suppress

p_values = st.lists(st.floats(min_value=0.0, max_value=1.0, allow_nan=False), min_size=1, max_size=60)


@given(p_values)
def test_bh_q_values_are_at_least_p_at_most_one_and_order_preserving(ps: list[float]):
    qs = benjamini_hochberg(ps)
    assert all(p <= q + 1e-12 and q <= 1.0 for p, q in zip(ps, qs, strict=True))
    ranked = sorted(zip(ps, qs, strict=True))
    assert all(a[1] <= b[1] + 1e-12 for a, b in itertools.pairwise(ranked))


@given(st.lists(st.tuples(st.booleans(), st.booleans()), min_size=1, max_size=80))
def test_cohens_kappa_is_bounded_when_defined(pairs: list[tuple[bool, bool]]):
    kappa = cohens_kappa([a for a, _ in pairs], [b for _, b in pairs])
    assert kappa is None or -1.0 - 1e-9 <= kappa <= 1.0 + 1e-9


@given(
    st.lists(st.tuples(st.integers(min_value=0, max_value=40), st.floats(0, 1, allow_nan=False)), max_size=40)
)
def test_suppress_nulls_every_value_below_the_floor(rows: list[tuple[int, float]]):
    frame = pl.DataFrame(
        {
            "row_key": [str(i) for i in range(len(rows))],
            "n": [n for n, _ in rows],
            "rate": [r for _, r in rows],
        },
        schema={"row_key": pl.String, "n": pl.Int64, "rate": pl.Float64},
    )
    out = suppress(frame, ["rate"], floor=5)
    assert out.height == frame.height
    small = out.filter(pl.col("suppressed"))
    assert small["rate"].null_count() == small.height
    assert all(n >= 5 for n in out.filter(~pl.col("suppressed"))["n"].to_list())
