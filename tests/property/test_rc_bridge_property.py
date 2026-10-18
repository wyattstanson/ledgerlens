"""The savings bridge sums exactly, to the cent, for any lines and panel."""

import polars as pl
from hypothesis import given, settings
from hypothesis import strategies as st

from ledgerlens.ratecards.bridge import STEPS, build_bridge, harmonized_prices
from ledgerlens.ratecards.optimizer import PanelProblem, PanelSolution

VENDORS = ("A", "B", "C")
ROLES = ("r1", "r2")


@st.composite
def scenario(draw: st.DrawFn):
    n = draw(st.integers(1, 25))
    lines = []
    for i in range(n):
        role = draw(st.sampled_from([*ROLES, None]))  # None = unmapped, passes through
        hours = draw(st.one_of(st.none(), st.floats(1, 200).map(lambda x: round(x, 2))))
        lines.append(
            {
                "line_id": f"L{i:03d}",
                "entity": draw(st.sampled_from(VENDORS)),
                "role_key": role,
                "baseline_hours": hours if role else None,
                "amount": draw(st.floats(10, 50_000).map(lambda x: round(x, 2))),
            }
        )
    frame = pl.DataFrame(
        lines,
        schema={
            "line_id": pl.String,
            "entity": pl.String,
            "role_key": pl.String,
            "baseline_hours": pl.Float64,
            "amount": pl.Float64,
        },
    )
    ids = frame["line_id"].to_list()
    dup = draw(st.lists(st.sampled_from(ids), max_size=3, unique=True))
    over = [i for i in draw(st.lists(st.sampled_from(ids), max_size=3, unique=True)) if i not in dup]
    leakage = pl.DataFrame(
        [{"line_id": i, "kind": "duplicate", "leakage_usd": 0.0} for i in dup]
        + [
            {
                "line_id": i,
                "kind": "overbill",
                "leakage_usd": draw(st.floats(0.01, 5.0).map(lambda x: round(x, 2))),
            }
            for i in over
        ],
        schema={"line_id": pl.String, "kind": pl.String, "leakage_usd": pl.Float64},
    )
    demand = {}
    for r in ROLES:
        hours = (
            frame.filter((pl.col("role_key") == r) & ~pl.col("line_id").is_in(dup))["baseline_hours"]
            .drop_nulls()
            .sum()
        )
        if hours > 0:
            demand[r] = float(hours)
    price = {(v, r): draw(st.floats(20, 150).map(lambda x: round(x, 2))) for v in VENDORS for r in demand}
    kept = tuple(draw(st.lists(st.sampled_from(VENDORS), min_size=1, max_size=3, unique=True)))
    hours = {}
    for r, d in demand.items():
        share = d / len(kept)
        hours.update({(v, r): share for v in kept})
    problem = PanelProblem(VENDORS, dict.fromkeys(VENDORS, 0.9), demand, price, 3, frozenset(), 1.0, 0.0)
    medians = {r: draw(st.floats(20, 150)) for r in demand}
    return frame, leakage, problem, PanelSolution(kept=kept, hours=hours, cost=0.0), medians


@settings(max_examples=150, deadline=None)
@given(scenario())
def test_bridge_steps_sum_exactly_and_trace_to_lines(case):
    lines, leakage, problem, solution, medians = case
    bridge = build_bridge(lines, leakage, problem, solution, harmonized_prices(problem, solution, medians))
    assert bridge.current_cents - sum(bridge.steps[s] for s in STEPS) == bridge.new_cents
    assert bridge.current_cents == sum(round(a * 100) for a in lines["amount"])
    assert set(bridge.lines["line_id"]).issubset(set(lines["line_id"]))
    assert int(bridge.lines["cents"].sum()) == sum(bridge.steps.values())  # every step dollar is on a line
