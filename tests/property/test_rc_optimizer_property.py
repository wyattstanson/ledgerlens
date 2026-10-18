"""The MILP must match brute force on small instances and never break panel size or critical coverage.

Expected optima come from enumeration, never from hand arithmetic (BUILD_PROMPT gotcha 7).
"""

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from ledgerlens.ratecards.optimizer import (
    InfeasiblePanelError,
    PanelProblem,
    brute_force_panel,
    check_solution,
    solve_panel,
)

RELAX = ("panel_size", "critical_coverage", "concentration", "quality_floor")


@st.composite
def problems(draw: st.DrawFn) -> PanelProblem:
    n_vendors = draw(st.integers(2, 5))
    vendors = tuple(f"V{i}" for i in range(n_vendors))
    roles = [f"r{j}" for j in range(draw(st.integers(1, 3)))]
    price = {}
    for v in vendors:
        for r in roles:
            if draw(st.booleans()):
                price[(v, r)] = draw(st.floats(10, 200, allow_nan=False).map(lambda x: round(x, 2)))
    demand = {r: draw(st.floats(10, 500).map(lambda x: round(x, 1))) for r in roles}
    critical = frozenset(r for r in roles if draw(st.booleans()))
    return PanelProblem(
        vendors=vendors,
        quality={v: draw(st.sampled_from([0.4, 0.7, 0.9])) for v in vendors},
        demand=demand,
        price=price,
        panel_size=draw(st.integers(1, n_vendors)),
        critical=critical,
        max_share_critical=draw(st.sampled_from([0.6, 0.7, 1.0])),
        min_quality=0.6,
    )


@settings(max_examples=60, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(problems())
def test_milp_matches_brute_force_and_honours_constraints(problem: PanelProblem):
    expected = brute_force_panel(problem)
    try:
        solved = solve_panel(problem, relax_order=())  # no relaxation: must agree on feasibility too
    except InfeasiblePanelError:
        assert expected is None
        return
    assert expected is not None
    assert solved.cost == pytest.approx(expected.cost, rel=1e-6, abs=0.02)  # 0.02 covers the keep epsilon
    assert check_solution(problem, solved) == []


def _two_vendor_problem(**overrides) -> PanelProblem:
    base = {
        "vendors": ("A", "B"),
        "quality": {"A": 0.9, "B": 0.9},
        "demand": {"r": 100.0},
        "price": {("A", "r"): 50.0, ("B", "r"): 80.0},
        "panel_size": 1,
        "critical": frozenset({"r"}),
        "max_share_critical": 0.7,
        "min_quality": 0.6,
    }
    return PanelProblem(**{**base, **overrides})


def test_infeasibility_is_relaxed_and_the_binding_family_reported():
    # A critical role needs two suppliers but the panel allows one: relaxing panel size fixes it.
    solution = solve_panel(_two_vendor_problem(), RELAX)
    assert solution.relaxed == "panel_size"
    assert solution.cost == pytest.approx(70 * 50 + 30 * 80)  # concentration cap 70% still holds


def test_unfixable_panel_raises_with_a_reason():
    problem = _two_vendor_problem(quality={"A": 0.9, "B": 0.1}, panel_size=1)
    with pytest.raises(InfeasiblePanelError, match="fewer than two eligible suppliers"):
        solve_panel(problem, RELAX)
