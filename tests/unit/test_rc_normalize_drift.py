"""Hand-worked normalization and leakage cases (BUILD_PROMPT 12.2 steps 3 and 8)."""

from datetime import date

import polars as pl
import pytest

from ledgerlens.ratecards.config import NormalizeConfig
from ledgerlens.ratecards.drift import find_duplicates, find_leakage
from ledgerlens.ratecards.normalize import normalize_cards, normalize_lines

CFG = NormalizeConfig(standard_hours=160, renegotiation_age_months=24)
TITLE_MAP = pl.DataFrame(
    {
        "billed_title": ["A", "B", "C", "D", "E"],
        "role_id": ["r.mid", "r.mid", "r.mid", "r.mid", None],
        "status": ["accepted", "reviewed", "accepted", "accepted", "pending_review"],
    }
)
FX = pl.DataFrame({"month": ["2026-09", "2026-09"], "currency": ["USD", "INR"], "usd_per_unit": [1.0, 0.012]})


def cards() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "card_id": ["C1", "C2", "C3", "C4", "C5"],
            "vendor_id": ["V1", "V2", "V3", "V4", "V5"],
            "vendor_title": ["A", "B", "C", "D", "E"],
            "location_tier": ["onshore"] * 4 + ["offshore"],
            "currency": ["USD"] * 4 + ["INR"],
            "pricing_model": ["tm", "outcome", "fte", "fixed", "tm"],
            "rate": [100.0, 110.0, 16000.0, 20000.0, 2500.0],
            "productivity_commitment": [None, 0.20, None, None, None],
            "contract_start": [
                date(2024, 9, 1),
                date(2025, 1, 1),
                date(2026, 1, 1),
                date(2020, 1, 1),
                date(2026, 6, 1),
            ],
        },
        schema_overrides={"productivity_commitment": pl.Float64},
    )


def test_outcome_based_rate_beats_cheaper_sticker_on_total_cost():
    # Spec example: $110 with a 20% productivity commitment is $88 per baseline hour, beating $100 T&M.
    out = normalize_cards(cards(), TITLE_MAP, FX, "2026-09", CFG).sort("card_id")
    sticker = dict(out.select("card_id", "sticker_usd_per_hour").iter_rows())
    effective = dict(out.select("card_id", "effective_usd_per_hour").iter_rows())
    assert (sticker["C1"], effective["C1"]) == (100.0, 100.0)
    assert sticker["C2"] == 110.0 and effective["C2"] == pytest.approx(88.0)
    assert effective["C3"] == pytest.approx(100.0)  # 16,000 a month / 160 hours
    assert effective["C4"] is None  # fixed price stays out of hourly benchmarks
    assert effective["C5"] == pytest.approx(30.0)  # INR 2,500 x 0.012
    assert out.filter(pl.col("card_id") == "C5")["role_id"][0] is None  # pending review: unmapped


def test_contracts_older_than_24_months_are_renegotiation_candidates():
    out = normalize_cards(cards(), TITLE_MAP, FX, "2026-09", CFG).sort("card_id")
    assert out["contract_age_months"].to_list() == [24, 20, 8, 80, 3]
    assert out["renegotiation_candidate"].to_list() == [False, False, False, True, False]


def lines() -> pl.DataFrame:
    base = {
        "tranid": "VB-1",
        "trandate": date(2026, 9, 28),
        "month": "2026-09",
        "subsidiary": "S",
        "currency": "USD",
        "exchangerate": 1.0,
        "location_tier": "onshore",
        "reported_hours": None,
    }
    rows = [
        {
            **base,
            "line_id": "L1",
            "entity": "V1",
            "billed_title": "A",
            "pricing_model": "tm",
            "quantity": 10.0,
            "unit": "hour",
            "rate": 100.0,
            "foreignamount": 1000.0,
            "amount": 1000.0,
        },
        {
            **base,
            "line_id": "L2",
            "entity": "V1",
            "billed_title": "A",
            "pricing_model": "tm",
            "quantity": 10.0,
            "unit": "hour",
            "rate": 102.0,
            "foreignamount": 1020.0,
            "amount": 1020.0,
            "tranid": "VB-2",
        },
        {
            **base,
            "line_id": "L3",
            "entity": "V1",
            "billed_title": "A",
            "pricing_model": "tm",
            "quantity": 10.0,
            "unit": "hour",
            "rate": 110.0,
            "foreignamount": 1100.0,
            "amount": 1100.0,
            "tranid": "VB-3",
        },
        {
            **base,
            "line_id": "L4",
            "entity": "V2",
            "billed_title": "B",
            "pricing_model": "outcome",
            "quantity": 10.0,
            "unit": "hour",
            "rate": 110.0,
            "foreignamount": 1100.0,
            "amount": 1100.0,
        },
        {
            **base,
            "line_id": "L5",
            "entity": "V4",
            "billed_title": "D",
            "pricing_model": "fixed",
            "quantity": 1.0,
            "unit": "milestone",
            "rate": 20000.0,
            "foreignamount": 20000.0,
            "amount": 20000.0,
            "reported_hours": 250.0,
        },
        {
            **base,
            "line_id": "L6",
            "entity": "V1",
            "billed_title": "A",
            "pricing_model": "tm",
            "quantity": 10.0,
            "unit": "hour",
            "rate": 110.0,
            "foreignamount": 1100.0,
            "amount": 1100.0,
            "tranid": "VB-3",
        },
    ]
    return pl.DataFrame(rows, schema_overrides={"reported_hours": pl.Float64})


def test_line_normalization_and_card_comparison():
    out = normalize_lines(lines(), cards(), TITLE_MAP, CFG).sort("line_id")
    row = {r["line_id"]: r for r in out.iter_rows(named=True)}
    assert row["L4"]["effective_usd_per_hour"] == pytest.approx(88.0)
    assert row["L4"]["baseline_hours"] == pytest.approx(12.5)  # 10 billed hours deliver 12.5 baseline hours
    assert row["L5"]["effective_usd_per_hour"] == pytest.approx(80.0)  # 20,000 / 250 reported hours
    assert row["L3"]["excess_usd"] == pytest.approx(100.0)
    assert row["L1"]["excess_usd"] == 0.0


def test_leakage_flags_more_than_two_percent_and_duplicates_without_double_counting():
    normalized = normalize_lines(lines(), cards(), TITLE_MAP, CFG)
    leak = find_leakage(normalized, threshold=0.02).sort("line_id")
    assert leak["line_id"].is_unique().all()  # each leaking line is counted once
    kinds = dict(leak.select("line_id", "kind").iter_rows())
    assert "L2" not in kinds  # exactly 2% above card is not "more than 2%"
    assert kinds == {"L3": "overbill", "L6": "duplicate"}  # L6 copies L3: full amount, not also an overbill
    assert dict(leak.select("line_id", "leakage_usd").iter_rows()) == {"L3": 100.0, "L6": 1100.0}


def test_duplicate_detection_keeps_the_first_line():
    assert find_duplicates(lines()).to_dicts() == [{"line_id": "L6", "original_line_id": "L3"}]
