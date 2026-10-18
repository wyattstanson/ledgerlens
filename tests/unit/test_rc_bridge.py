"""Hand-worked savings bridge: each step lands where it should, in cents."""

import polars as pl

from ledgerlens.ratecards.bridge import build_bridge, harmonized_prices
from ledgerlens.ratecards.optimizer import PanelProblem, PanelSolution

EMPTY_LEAKAGE = pl.DataFrame(schema={"line_id": pl.String, "kind": pl.String, "leakage_usd": pl.Float64})


def test_fx_harmonization_and_exits_are_separated():
    # A is kept with a card of $50/h at today's FX (it was $52/h at posting); B exits at $60/h.
    problem = PanelProblem(
        vendors=("A", "B"),
        quality={"A": 0.9, "B": 0.9},
        demand={"r": 20.0},
        price={("A", "r"): 50.0, ("B", "r"): 60.0},
        panel_size=1,
        critical=frozenset(),
        max_share_critical=1.0,
        min_quality=0.0,
    )
    solution = PanelSolution(kept=("A",), hours={("A", "r"): 20.0}, cost=1000.0)
    harmonized = harmonized_prices(problem, solution, {"r": 45.0})  # median below A's card
    lines = pl.DataFrame(
        {
            "line_id": ["L1", "L2"],
            "entity": ["A", "B"],
            "role_key": ["r", "r"],
            "baseline_hours": [10.0, 10.0],
            "amount": [520.0, 600.0],
        }
    )
    bridge = build_bridge(lines, EMPTY_LEAKAGE, problem, solution, harmonized)
    assert bridge.steps == {
        "leakage_recovery": 0,
        "fx_retranslation": 2_000,  # $520 at posting FX vs $500 at today's FX
        "rate_harmonization": 5_000,  # $50 card down to the $45 median
        "volume_shift": 0,
        "vendor_exits": 15_000,  # B's $600 moves to the panel at $45/h
    }
    assert (bridge.current_cents, bridge.new_cents) == (112_000, 90_000)


def test_kept_vendor_without_an_hourly_card_books_volume_shift_not_exits():
    problem = PanelProblem(
        vendors=("A", "C"),
        quality={"A": 0.9, "C": 0.9},
        demand={"r": 10.0},
        price={("C", "r"): 40.0},  # A only has a fixed-price card for r
        panel_size=2,
        critical=frozenset(),
        max_share_critical=1.0,
        min_quality=0.0,
    )
    solution = PanelSolution(kept=("A", "C"), hours={("C", "r"): 10.0}, cost=400.0)
    lines = pl.DataFrame(
        {"line_id": ["L1"], "entity": ["A"], "role_key": ["r"], "baseline_hours": [10.0], "amount": [700.0]}
    )
    bridge = build_bridge(lines, EMPTY_LEAKAGE, problem, solution, harmonized_prices(problem, solution, {}))
    assert bridge.steps["vendor_exits"] == 0
    assert bridge.steps["volume_shift"] == 30_000  # $700 fixed-price work re-sourced at $40/h


def test_tail_negotiation_target_is_capped_at_the_current_rate():
    from ledgerlens.ratecards.tail import tail_plan  # noqa: PLC0415

    cards = pl.DataFrame(
        {"vendor_id": ["V9"], "role_id": ["r"], "card_id": ["C9"], "location_tier": ["onshore"],
         "effective_usd_per_hour": [40.0]}
    )  # fmt: skip
    bench = pl.DataFrame(
        {
            "role_id": ["r"],
            "location_tier": ["onshore"],
            "p25": [50.0],
            "median": [60.0],
            "source": ["internal"],
        }
    )
    score = pl.DataFrame(
        {
            "vendor_id": ["V9"],
            "name": ["Nine"],
            "quality": [0.9],
            "baseline_spend_usd": [1.0],
            "roles_covered": [1],
        }
    )
    plan = tail_plan(score, cards, bench, kept=set(), min_quality=0.6)
    [brief] = plan["briefs"][0]
    assert brief["target_usd_per_hour"] == 40.0  # never above today's $40, even though p25 is $50
    assert brief["walk_away_usd_per_hour"] == 40.0
