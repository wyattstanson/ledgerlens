"""The disclosure audit on hand-built roles (ADR 0004, decision 6).

Strategy: each piece of reader knowledge (rounded stage totals, hires, funnel steps, intersectional margins,
h against everyone else) gets a case where it alone narrows a small cell, so dropping it from the model fails
a test. The repair tests check that the audit hides only what a disclosure needs, and the cheapest cell that
works. The privacy integration test then runs the audit on generated worlds with a whole-number reader.
"""

import math

import pytest

from ledgerlens.core.stats import cohens_h
from ledgerlens.interviews.disclosure import (
    Cell,
    Policy,
    RateBound,
    RoleModel,
    cheapest,
    complement_rate_bounds,
    exposed,
    plan_role,
    rounded_interval,
)

STRICT = Policy(floor=5, min_outcome_values=5)


def one_way(stage: str, dimension: str, group: str, n: int, selected: int) -> Cell:
    tier = 1 if group == "Unknown" else 2
    return Cell(f"ir|r|{stage}|{dimension}|{group}", stage, dimension, (group,), n, selected, tier)


def cross(stage: str, sex: str, race: str, n: int, selected: int) -> Cell:
    return Cell(f"ix|r|{stage}|{sex}|{race}", stage, "cross", (sex, race), n, selected, 0)


def sex_line(
    stage: str, female: tuple[int, int], male: tuple[int, int], unknown: tuple[int, int]
) -> list[Cell]:
    return [
        one_way(stage, "sex", g, *v) for g, v in (("Female", female), ("Male", male), ("Unknown", unknown))
    ]


def key(cells: list[Cell], group: str, stage: str = "screen") -> str:
    return next(c.key for c in cells if c.group == (group,) and c.stage == stage)


def test_rounded_interval_is_what_a_reader_knows_about_a_total_shown_to_the_nearest_ten():
    assert rounded_interval(195) == (195, 204)  # shown as 200
    assert rounded_interval(194) == (185, 194)  # shown as 190


def test_a_small_group_is_safe_when_the_stage_total_is_only_known_to_the_nearest_ten():
    # The old neighbour rule hid Female here too; the audit sees that Unknown can be anything from 4 to 13.
    cells = sex_line("screen", (91, 69), (100, 73), (4, 4))
    unknown = key(cells, "Unknown")
    model = RoleModel(cells, {unknown}, None, [])
    assert model.exposure(unknown).values == {"n": 10, "selected": 14, "failed": 14}
    assert plan_role(cells, None, [], STRICT, True) == {unknown}


def test_a_total_at_the_edge_of_its_rounding_forces_the_cheapest_neighbour():
    cells = sex_line("screen", (91, 69), (100, 73), (1, 1))  # total 192: a reader knows 185..194, so 0..3
    unknown, female = key(cells, "Unknown"), key(cells, "Female")
    assert RoleModel(cells, {unknown}, None, []).exposure(unknown).values["n"] == 4
    assert plan_role(cells, None, [], STRICT, True) == {unknown, female}  # Female (91) is cheaper than Male


def test_exact_hires_pin_the_last_stage_passed_count():
    cells = sex_line("screen", (91, 69), (100, 73), (4, 4)) + sex_line(
        "technical", (69, 30), (73, 40), (4, 2)
    )
    hidden = {key(cells, "Unknown"), key(cells, "Unknown", "technical")}
    late = key(cells, "Unknown", "technical")
    assert (
        RoleModel(cells, hidden, 72, []).exposure(late).values["selected"] == 1
    )  # 72 - 30 - 40 = 2, exactly
    assert RoleModel(cells, hidden, None, []).exposure(late).values["selected"] > 5


def test_a_group_s_size_at_one_stage_is_its_passed_count_at_the_stage_before():
    # Unknown is published at the screen (8 entered, 3 passed), so its technical size is exactly 3, whatever
    # the technical total; that is why a group hidden later can stay exposed until the audit repairs it.
    # A second small group (X) keeps the screen's pass total, and so the technical total, uncertain.
    cells = [
        *sex_line("screen", (91, 69), (100, 73), (8, 3)),
        one_way("screen", "sex", "X", 2, 1),
        *sex_line("technical", (69, 30), (73, 40), (3, 1)),
        one_way("technical", "sex", "X", 1, 0),
    ]
    late = key(cells, "Unknown", "technical")
    model = RoleModel(cells, {late, key(cells, "X"), key(cells, "X", "technical")}, None, [])
    assert model.interval(late, "n")[:2] == pytest.approx((3, 3))
    hidden = plan_role(cells, None, [], STRICT, True)
    assert key(cells, "Unknown") in hidden  # the cheapest fix: stop publishing the screen row it comes from


def test_intersectional_margins_pin_a_cell_until_the_audit_repairs_it():
    one = [
        *sex_line("screen", (100, 50), (100, 50), (3, 2)),
        one_way("screen", "race_ethnicity", "White", 150, 76),
        one_way("screen", "race_ethnicity", "Black", 53, 26),
    ]
    inner = [
        cross("screen", "Female", "White", 75, 38),
        cross("screen", "Female", "Black", 25, 12),
        cross("screen", "Male", "White", 74, 37),
        cross("screen", "Male", "Black", 26, 13),
        cross("screen", "Unknown", "White", 1, 1),
        cross("screen", "Unknown", "Black", 2, 1),
    ]
    cells = one + inner
    small = {c.key for c in cells if c.n < 5}
    # White minus the other White cells gives Unknown x White away exactly.
    assert RoleModel(cells, small, None, []).exposure("ix|r|screen|Unknown|White").values["n"] == 1
    hidden = plan_role(cells, None, [], STRICT, True)
    assert hidden > small
    assert exposed(RoleModel(cells, hidden, None, [], integral=True), sorted(small), STRICT) == {}


def test_the_cheapest_repair_is_intersectional_then_unknown_or_tiny_then_smaller():
    cells = {
        c.key: c
        for c in (
            one_way("screen", "sex", "Male", 40, 20),
            one_way("screen", "sex", "Unknown", 90, 45),
            cross("screen", "Male", "White", 300, 150),
            one_way("screen", "sex", "Female", 30, 15),
        )
    }
    assert cheapest(cells, cells) == "ix|r|screen|Male|White"
    del cells["ix|r|screen|Male|White"]
    assert cheapest(cells, cells) == "ir|r|screen|sex|Unknown"  # tier beats size
    del cells["ir|r|screen|sex|Unknown"]
    assert cheapest(cells, cells) == "ir|r|screen|sex|Female"


def test_h_against_everyone_else_would_give_the_stage_total_away():
    # The safe case above, with the stage's passes known (hires) and Female's h against everyone else.
    cells = sex_line("screen", (91, 69), (100, 73), (4, 4))
    unknown = key(cells, "Unknown")
    h = round(cohens_h(69 / 91, 77 / 104), 2)
    bound = RateBound(key(cells, "Female"), "screen", 91, 69, *complement_rate_bounds(91, 69, h))
    assert RoleModel(cells, {unknown}, 146, []).exposure(unknown).values["n"] == 10
    assert RoleModel(cells, {unknown}, 146, [bound]).exposure(unknown).values["n"] < 5


def test_complement_rate_bounds_contain_the_true_rate():
    low, high = complement_rate_bounds(91, 69, round(cohens_h(69 / 91, 77 / 104), 2))
    assert low <= 77 / 104 <= high and high - low < 0.01


def test_a_hidden_row_publishes_no_h():
    cells = sex_line("screen", (91, 69), (100, 73), (4, 4))
    female, unknown = key(cells, "Female"), key(cells, "Unknown")
    bound = RateBound(female, "screen", 91, 69, *complement_rate_bounds(91, 69, 0.05))
    model = RoleModel(cells, {female, unknown}, None, [bound])
    assert not any(female in cells_ for _, cells_ in model.rows if len(cells_) == 1)


def test_the_whole_number_reader_agrees_on_the_safe_case():
    cells = sex_line("screen", (91, 69), (100, 73), (4, 4))
    unknown = key(cells, "Unknown")
    values = RoleModel(cells, {unknown}, None, [], integral=True).exposure(unknown).values
    assert values["n"] == 10 and not math.isinf(values["n"])


def test_a_group_cannot_be_protected_beyond_what_its_whole_stage_allows():
    # 14 entrants (known as 5..14) and 11 hires: at most 3 people failed, so 0..3 is all a group can get.
    cells = sex_line("screen", (6, 4), (7, 6), (1, 1))
    unknown = key(cells, "Unknown")
    model = RoleModel(cells, {c.key for c in cells}, 11, [])
    assert model.stage_ceiling("screen", "failed") == 3
    assert model.exposure(unknown).values["failed"] == 4
    assert exposed(model, [unknown], STRICT) == {}
