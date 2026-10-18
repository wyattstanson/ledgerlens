"""Golden eval cases for the rate-card pack, computed from the generator's ground truth.

Expected values come from the world's hidden truth (true title -> role map, planted lines, card terms),
computed here independently of the pipeline modules under test.
"""

import json
from pathlib import Path
from statistics import median

import polars as pl

from ledgerlens.ratecards.config import RateCardsConfig
from ledgerlens.ratecards.drift import RATIO_EPSILON
from ledgerlens.ratecards.generate import World

MIN_BENCHMARK_VENDORS = 3
SOURCE = "generator:restricted.rc_title_truth+rc_planted+raw_ratecards.rate_cards"


def _truth_effective_rates(world: World, standard_hours: float) -> pl.DataFrame:
    fx = dict(
        world.fx.filter(pl.col("month") == world.as_of_month).select("currency", "usd_per_unit").iter_rows()
    )
    truth = dict(world.title_truth.select("billed_title", "role_id").iter_rows())
    rows = []
    for card in world.rate_cards.iter_rows(named=True):
        model, usd = card["pricing_model"], card["rate"] * fx[card["currency"]]
        if model == "fixed":
            continue
        sticker = usd / standard_hours if model == "fte" else usd
        effective = sticker * (1 - card["productivity_commitment"]) if model == "outcome" else sticker
        rows.append(
            {
                "vendor_id": card["vendor_id"],
                "role_id": truth[card["vendor_title"]],
                "location_tier": card["location_tier"],
                "sticker": sticker,
                "effective": effective,
            }
        )
    return pl.DataFrame(rows)


def _case(case_id: str, tool: str, args: dict, expected: list[tuple[str, float]], tol: float = 0.01) -> dict:
    return {
        "id": case_id,
        "pack": "ratecards",
        "suite": "offline",
        "expected_source": SOURCE,
        "tool": tool,
        "input": args,
        "expected": [{"label": label, "value": round(value, 6), "abs_tol": tol} for label, value in expected],
    }


def golden_cases(world: World, config: RateCardsConfig) -> list[dict]:
    rates = _truth_effective_rates(world, config.normalize.standard_hours)
    cases: list[dict] = []

    # 1. Benchmarks: median effective rate for 24 role/tier pairs with at least 3 vendors.
    groups = (
        rates.group_by("role_id", "location_tier").agg(pl.col("effective")).sort("role_id", "location_tier")
    )
    eligible = [g for g in groups.iter_rows(named=True) if len(g["effective"]) >= MIN_BENCHMARK_VENDORS]
    step = max(1, len(eligible) // 24)
    for g in eligible[::step][:24]:
        key = f"{g['role_id']}@{g['location_tier']}"
        cases.append(
            _case(
                f"rc-benchmark-{key}",
                "rc.benchmark",
                {"role_id": g["role_id"], "location_tier": g["location_tier"]},
                [(f"median USD/hour {key}", median(g["effective"]))],
            )
        )

    # 2. Leakage per planted vendor: lines billed more than the threshold above the card.
    planted = world.planted
    over = planted.filter(
        pl.col("kind").is_in(["overbill", "drift"])
        & (pl.col("excess_ratio") > config.drift.threshold + RATIO_EPSILON)
    )
    for vendor in sorted(over["vendor_id"].unique().to_list()):
        rows = over.filter(pl.col("vendor_id") == vendor)
        cases.append(
            _case(
                f"rc-leakage-{vendor}",
                "rc.leakage",
                {"vendor_id": vendor},
                [
                    (f"overbill USD {vendor}", float(rows["excess_usd"].sum())),
                    (f"overbill lines {vendor}", float(rows.height)),
                ],
            )
        )

    # 3. Duplicates across all vendors.
    dups = planted.filter(pl.col("kind") == "duplicate")
    cases.append(
        _case(
            "rc-duplicates-count", "rc.leakage", {}, [("duplicate lines all vendors", float(dups.height))], 0
        )
    )
    cases.append(
        _case(
            "rc-duplicates-usd",
            "rc.leakage",
            {},
            [("duplicate USD all vendors", float(dups["excess_usd"].sum()))],
        )
    )

    # 4. The planted pair: sticker vs total-cost rates on two shared roles.
    cheap = planted.filter(pl.col("kind") == "cheap_vendor")["vendor_id"][0]
    outcome = planted.filter(pl.col("kind") == "outcome_vendor")["vendor_id"][0]
    pair = rates.filter(pl.col("vendor_id").is_in([cheap, outcome]))
    shared = sorted(
        set(pair.filter(pl.col("vendor_id") == cheap)["role_id"])
        & set(pair.filter(pl.col("vendor_id") == outcome)["role_id"])
    )
    for role in shared[:2]:
        rows = {r["vendor_id"]: r for r in pair.filter(pl.col("role_id") == role).iter_rows(named=True)}
        cases.append(
            _case(
                f"rc-pair-{role}",
                "rc.vendor_rates",
                {"role_id": role, "location_tier": "onshore"},
                [
                    (f"sticker USD/hour {cheap} {role}", rows[cheap]["sticker"]),
                    (f"effective USD/hour {cheap} {role}", rows[cheap]["effective"]),
                    (f"sticker USD/hour {outcome} {role}", rows[outcome]["sticker"]),
                    (f"effective USD/hour {outcome} {role}", rows[outcome]["effective"]),
                ],
            )
        )
    return cases


def write_golden(cases: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(c, sort_keys=True) + "\n" for c in cases), encoding="utf-8")
