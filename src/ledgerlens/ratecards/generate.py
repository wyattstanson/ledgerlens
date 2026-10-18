"""Synthetic rate-card world with planted problems and hidden ground truth (BUILD_PROMPT section 12.1).

Everything is seeded. Raw tables go to schema ``raw_ratecards`` (NetSuite vendor-bill shape, ADR 0001);
the truth goes to ``restricted`` and is read only by evals and tests:

- ``restricted.rc_title_truth``: billed title -> standard role.
- ``restricted.rc_planted``: overbilled lines, drifting lines, duplicate lines, and the cheap-per-hour /
  outcome-based vendor pair.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl
from faker import Faker

from ledgerlens.core.db import connect
from ledgerlens.ratecards.config import WorldConfig

SENIORITIES = ("junior", "mid", "senior", "lead")
SENIORITY_MULT = {"junior": 0.7, "mid": 1.0, "senior": 1.3, "lead": 1.6}
SEN_WORD = {"junior": "Junior", "mid": "", "senior": "Senior", "lead": "Lead"}
SEN_ABBR = {"junior": "Jr.", "mid": "Mid", "senior": "Sr.", "lead": "Ld."}
ROMAN = {"junior": "I", "mid": "II", "senior": "III", "lead": "IV"}
LEVEL = {"junior": "L1", "mid": "L2", "senior": "L3", "lead": "L4"}


@dataclass(frozen=True)
class Specialization:
    key: str
    family: str
    name: str
    abbrev: str
    base_usd: float  # onshore mid-level hourly rate
    synonyms: tuple[str, ...]


SPECIALIZATIONS: tuple[Specialization, ...] = (
    Specialization(
        "backend",
        "engineering",
        "Backend Engineer",
        "Backend Eng",
        95,
        ("Server-side Developer", "API Developer"),
    ),
    Specialization(
        "frontend", "engineering", "Frontend Engineer", "Frontend Eng", 90, ("UI Developer", "Web Developer")
    ),
    Specialization(
        "mobile",
        "engineering",
        "Mobile Engineer",
        "Mobile Eng",
        95,
        ("App Developer", "iOS Android Developer"),
    ),
    Specialization(
        "fullstack",
        "engineering",
        "Full Stack Engineer",
        "FS Eng",
        100,
        ("Full-stack Developer", "Web Application Developer"),
    ),
    Specialization(
        "data_engineer",
        "data",
        "Data Engineer",
        "Data Eng",
        100,
        ("ETL Developer", "Data Pipeline Developer"),
    ),
    Specialization(
        "data_scientist",
        "data",
        "Data Scientist",
        "Data Sci",
        110,
        ("ML Scientist", "Machine Learning Specialist"),
    ),
    Specialization(
        "data_analyst", "data", "Data Analyst", "Data Anlst", 80, ("BI Analyst", "Reporting Analyst")
    ),
    Specialization("qa_manual", "qa", "QA Analyst", "QA Anlst", 65, ("Test Analyst", "Manual Tester")),
    Specialization(
        "qa_automation",
        "qa",
        "QA Automation Engineer",
        "QA Auto Eng",
        80,
        ("SDET", "Test Automation Developer"),
    ),
    Specialization(
        "project_manager",
        "pm",
        "Project Manager",
        "Proj Mgr",
        105,
        ("Delivery Manager", "Program Coordinator"),
    ),
    Specialization(
        "business_analyst",
        "ba",
        "Business Analyst",
        "Bus Anlst",
        90,
        ("Requirements Analyst", "Functional Consultant"),
    ),
    Specialization(
        "devops",
        "infrastructure",
        "DevOps Engineer",
        "DevOps Eng",
        100,
        ("Build Release Engineer", "CI CD Specialist"),
    ),
    Specialization(
        "cloud",
        "infrastructure",
        "Cloud Engineer",
        "Cloud Eng",
        105,
        ("Cloud Architect", "AWS Azure Specialist"),
    ),
    Specialization(
        "sre",
        "infrastructure",
        "Site Reliability Engineer",
        "SRE",
        110,
        ("Reliability Specialist", "Production Engineer"),
    ),
    Specialization(
        "service_desk",
        "support",
        "Service Desk Analyst",
        "Svc Desk Anlst",
        50,
        ("Help Desk Technician", "IT Support Agent"),
    ),
    Specialization(
        "app_support",
        "support",
        "Application Support Engineer",
        "App Support Eng",
        60,
        ("Production Support Analyst", "L2 Support Specialist"),
    ),
)

# tier -> (currency, rate multiplier vs onshore, USD per unit at month 0)
TIERS = {"onshore": ("USD", 1.0, 1.0), "nearshore": ("MXN", 0.6, 0.055), "offshore": ("INR", 0.32, 0.0119)}
PRICING_MODELS = ("tm", "fte", "fixed", "outcome")
PRICING_PROBS = (0.70, 0.12, 0.10, 0.08)
HOURS_REPORTED_SHARE = 0.7  # fixed-price lines that report hours
SUBSIDIARY = "SUB-US"


@dataclass
class World:
    vendors: pl.DataFrame
    roles: pl.DataFrame
    rate_cards: pl.DataFrame
    fx: pl.DataFrame
    invoice_lines: pl.DataFrame
    title_truth: pl.DataFrame
    planted: pl.DataFrame
    as_of_month: str
    seed: int


def role_id(spec: Specialization, seniority: str) -> str:
    return f"{spec.key}.{seniority}"


def standard_title(spec: Specialization, seniority: str) -> str:
    return f"{seniority.title()} {spec.name}"


def _month_label(start: str, offset: int) -> str:
    year, month = map(int, start.split("-"))
    total = year * 12 + (month - 1) + offset
    return f"{total // 12:04d}-{total % 12 + 1:02d}"


def _typo(text: str, rng: np.random.Generator) -> str:
    letters = [i for i in range(1, len(text) - 2) if text[i].isalpha() and text[i + 1].isalpha()]
    i = int(rng.choice(letters))
    return text[:i] + text[i + 1] + text[i] + text[i + 2 :]


def _title_candidates(spec: Specialization, sen: str, rng: np.random.Generator) -> list[str]:
    word = SEN_WORD[sen]
    candidates = [
        f"{word} {spec.name}".strip(),
        f"{SEN_ABBR[sen]} {spec.abbrev}",
        f"{spec.name} {ROMAN[sen]}",
        f"{spec.name} {LEVEL[sen]}",
        f"{spec.name} ({sen.title()})",
        f"{spec.name}, {sen.title()}",
        f"{sen.title()} {spec.abbrev}",
        _typo(f"{sen.title()} {spec.name}", rng),
        *(f"{syn} {ROMAN[sen]}" for syn in spec.synonyms),
        *(f"{sen.title()} {syn}" for syn in spec.synonyms),
    ]
    return list(dict.fromkeys(candidates))  # dedupe, keep order (canonical first)


def _title_pools(total: int, rng: np.random.Generator) -> dict[str, list[str]]:
    """Distinct billed-title strings per role, ``total`` across all roles, each string owned by one role."""
    roles = [(spec, sen) for spec in SPECIALIZATIONS for sen in SENIORITIES]
    per_role = np.full(len(roles), total // len(roles))
    per_role[rng.permutation(len(roles))[: total % len(roles)]] += 1
    used: set[str] = set()
    pools: dict[str, list[str]] = {}
    for (spec, sen), k in zip(roles, per_role, strict=True):
        candidates = [c for c in _title_candidates(spec, sen, rng) if c not in used]
        head, rest = candidates[:1], candidates[1:]
        rest = [rest[i] for i in rng.permutation(len(rest))]
        pool = (head + rest)[: int(k)]
        used.update(pool)
        pools[role_id(spec, sen)] = pool
    return pools


def generate_world(config: WorldConfig, seed: int) -> World:  # noqa: PLR0912, PLR0915 (one linear recipe)
    rng = np.random.default_rng(seed)
    fake = Faker()
    fake.seed_instance(seed)

    # --- roles -------------------------------------------------------------------------------
    roles = pl.DataFrame(
        [
            {
                "role_id": role_id(s, sen),
                "specialization": s.key,
                "family": s.family,
                "seniority": sen,
                "title": standard_title(s, sen),
            }
            for s in SPECIALIZATIONS
            for sen in SENIORITIES
        ]
    )
    spec_by_key = {s.key: s for s in SPECIALIZATIONS}

    # --- vendors -----------------------------------------------------------------------------
    n = config.vendors
    tier_names = ["onshore"] * round(n * 0.4) + ["nearshore"] * round(n * 0.3)
    tier_names += ["offshore"] * (n - len(tier_names))
    vendor_ids = [f"V{i + 1:02d}" for i in range(n)]
    onshore = [v for v, t in zip(vendor_ids, tier_names, strict=True) if t == "onshore"]
    cheap, outcome = onshore[0], onshore[1]
    others = [v for v in vendor_ids if v not in (cheap, outcome)]
    picks = rng.permutation(len(others))
    overbill = sorted(others[i] for i in picks[: config.overbill_vendors])
    drift = sorted(
        others[i] for i in picks[config.overbill_vendors : config.overbill_vendors + config.drift_vendors]
    )

    quality = rng.uniform(0.55, 0.95, n).round(3)
    premium = np.exp(rng.normal(0, 0.08, n))
    for vendor, q, p in ((cheap, 0.85, 0.90), (outcome, 0.85, 1.08)):
        i = vendor_ids.index(vendor)
        quality[i], premium[i] = q, p
    vendors = pl.DataFrame(
        {
            "vendor_id": vendor_ids,
            "name": [fake.unique.company() for _ in vendor_ids],
            "location_tier": tier_names,
            "currency": [TIERS[t][0] for t in tier_names],
            "quality": quality,
            "sla": rng.uniform(0.8, 0.99, n).round(3),
        }
    )

    # Balanced specializations: vendor i takes a contiguous block of a shuffled cycle.
    spec_keys = [SPECIALIZATIONS[i].key for i in rng.permutation(len(SPECIALIZATIONS))]
    k = min(config.specializations_per_vendor, len(spec_keys))
    vendor_specs = {
        v: [spec_keys[(i * k + j) % len(spec_keys)] for j in range(k)] for i, v in enumerate(vendor_ids)
    }
    vendor_specs[outcome] = list(vendor_specs[cheap])  # planted pair competes on the same roles

    # --- billed titles -------------------------------------------------------------------------
    pools = _title_pools(config.distinct_titles, rng)
    pairs_by_role: dict[str, list[str]] = {}
    for v in vendor_ids:
        for key in vendor_specs[v]:
            for sen in SENIORITIES:
                pairs_by_role.setdefault(role_id(spec_by_key[key], sen), []).append(v)
    vendor_title: dict[tuple[str, str], str] = {}
    for rid, vendors_for_role in pairs_by_role.items():
        pool = pools[rid]
        order = rng.permutation(len(vendors_for_role))
        for slot, idx in enumerate(order):
            vendor_title[(vendors_for_role[idx], rid)] = pool[slot % len(pool)]
    used_titles = set(vendor_title.values())
    title_truth = pl.DataFrame(
        [{"billed_title": t, "role_id": rid} for rid, pool in pools.items() for t in pool if t in used_titles]
    )

    # --- FX ------------------------------------------------------------------------------------
    months = [_month_label(config.start_month, m) for m in range(config.months)]
    fx_rows, fx = [], {}
    for currency, base in (("USD", 1.0), ("MXN", 0.055), ("INR", 0.0119)):
        walk = (
            np.ones(config.months)
            if currency == "USD"
            else np.cumprod(1 + rng.normal(0, 0.01, config.months))
        )
        walk[0] = 1.0
        series = (base * walk).round(6)
        fx[currency] = series
        fx_rows += [
            {"month": mo, "currency": currency, "usd_per_unit": float(r)}
            for mo, r in zip(months, series, strict=True)
        ]

    # --- rate cards ------------------------------------------------------------------------------
    start_year, start_mo = map(int, config.start_month.split("-"))
    cards, card_of = [], {}
    tier_of = dict(zip(vendor_ids, tier_names, strict=True))
    for (v, rid), title in sorted(vendor_title.items()):
        spec = spec_by_key[rid.split(".")[0]]
        sen = rid.split(".")[1]
        currency, tier_mult, fx0 = TIERS[tier_of[v]]
        usd_hourly = spec.base_usd * SENIORITY_MULT[sen] * tier_mult * premium[vendor_ids.index(v)]
        noise = float(np.exp(rng.normal(0, 0.03)))
        if v not in (cheap, outcome):  # keep the planted pair's ground truth crisp
            usd_hourly *= noise
        model = (
            "tm"
            if v == cheap
            else "outcome"
            if v == outcome
            else str(rng.choice(PRICING_MODELS, p=PRICING_PROBS))
        )
        commitment = None
        if model == "tm":
            rate = usd_hourly / fx0
        elif model == "outcome":
            uplift = 1.0 if v == outcome else float(rng.uniform(1.05, 1.15))
            commitment = 0.22 if v == outcome else round(float(rng.uniform(0.10, 0.25)), 3)
            rate = usd_hourly * uplift / fx0
        elif model == "fte":
            rate = usd_hourly * 160 / fx0
        else:
            rate = usd_hourly * float(rng.uniform(80, 200)) / fx0
        # Start of the current contract term, 1 to 48 months before the as-of month.
        age = int(rng.integers(1, 49))
        total = start_year * 12 + (start_mo - 1) + config.months - 1 - age
        card = {
            "card_id": f"C{len(cards) + 1:04d}",
            "vendor_id": v,
            "vendor_title": title,
            "location_tier": tier_of[v],
            "currency": currency,
            "pricing_model": model,
            "rate": round(rate, 2),
            "productivity_commitment": commitment,
            "contract_start": date(total // 12, total % 12 + 1, 1),
        }
        cards.append(card)
        card_of[(v, rid)] = card
    rate_cards = pl.DataFrame(cards, schema_overrides={"productivity_commitment": pl.Float64})

    # --- invoice lines -----------------------------------------------------------------------------
    overbill_roles = {
        v: {
            rid
            for rid in pairs_by_role
            if v in pairs_by_role[rid] and rng.random() < config.overbill_role_share
        }
        for v in overbill
    }
    # sorted(): set order depends on PYTHONHASHSEED, and rng draws must not.
    overbill_uplift = {
        (v, rid): float(rng.uniform(0.05, 0.15)) for v in overbill for rid in sorted(overbill_roles[v])
    }
    lines, planted = [], []
    vendor_roles = {v: sorted(rid for (vv, rid) in vendor_title if vv == v) for v in vendor_ids}
    for v in vendor_ids:
        n_projects = int(rng.integers(1, config.max_projects_per_vendor + 1))
        for m, month in enumerate(months):
            year, mo = map(int, month.split("-"))
            for p in range(n_projects):
                tranid = f"VB-{v}-{month.replace('-', '')}-{p + 1}"
                for rid in vendor_roles[v]:
                    if rng.random() >= config.line_probability:
                        continue
                    card = card_of[(v, rid)]
                    factor = 1.0
                    kind = None
                    if (v, rid) in overbill_uplift and m >= config.overbill_start_month:
                        factor, kind = 1 + overbill_uplift[(v, rid)], "overbill"
                    elif v in drift and m > config.drift_start_month:
                        factor, kind = 1 + config.drift_per_month * (m - config.drift_start_month), "drift"
                    model = card["pricing_model"]
                    reported_hours = None
                    if model in ("tm", "outcome"):
                        quantity, unit = float(rng.integers(8, 169)), "hour"
                    elif model == "fte":
                        quantity, unit = float(rng.integers(1, 3)), "fte_month"
                    else:
                        quantity, unit = 1.0, "milestone"
                        if rng.random() < HOURS_REPORTED_SHARE:
                            reported_hours = float(rng.integers(40, 200))
                    billed_rate = round(card["rate"] * factor, 2)
                    exch = float(fx[card["currency"]][m])
                    foreign = round(billed_rate * quantity, 2)
                    amount = round(foreign * exch, 2)
                    line_id = f"L{len(lines) + 1:06d}"
                    lines.append(
                        {
                            "line_id": line_id,
                            "tranid": tranid,
                            "trandate": date(year, mo, 28),
                            "month": month,
                            "entity": v,
                            "subsidiary": SUBSIDIARY,
                            "currency": card["currency"],
                            "exchangerate": exch,
                            "billed_title": card["vendor_title"],
                            "location_tier": card["location_tier"],
                            "pricing_model": model,
                            "quantity": quantity,
                            "unit": unit,
                            "reported_hours": reported_hours,
                            "rate": billed_rate,
                            "foreignamount": foreign,
                            "amount": amount,
                        }
                    )
                    if kind:
                        card_amount = round(round(card["rate"] * quantity, 2) * exch, 2)
                        # Truth is what the invoice actually says: the rate after rounding to cents.
                        planted.append(
                            {
                                "kind": kind,
                                "vendor_id": v,
                                "line_id": line_id,
                                "original_line_id": None,
                                "vendor_title": card["vendor_title"],
                                "excess_usd": round(amount - card_amount, 2),
                                "excess_ratio": billed_rate / card["rate"] - 1,
                            }
                        )

    planted_vendors = {cheap, outcome, *overbill, *drift}
    clean = [i for i, line in enumerate(lines) if line["entity"] not in planted_vendors]
    for i in rng.choice(clean, size=min(config.duplicate_lines, len(clean)), replace=False):
        copy = dict(lines[int(i)])
        original = copy["line_id"]
        copy["line_id"] = f"L{len(lines) + 1:06d}"
        lines.append(copy)
        planted.append(
            {
                "kind": "duplicate",
                "vendor_id": copy["entity"],
                "line_id": copy["line_id"],
                "original_line_id": original,
                "vendor_title": copy["billed_title"],
                "excess_usd": copy["amount"],
                "excess_ratio": None,
            }
        )
    for kind, v in (("cheap_vendor", cheap), ("outcome_vendor", outcome)):
        planted.append(
            {
                "kind": kind,
                "vendor_id": v,
                "line_id": None,
                "original_line_id": None,
                "vendor_title": None,
                "excess_usd": None,
                "excess_ratio": None,
            }
        )

    planted_schema = {
        "kind": pl.String,
        "vendor_id": pl.String,
        "line_id": pl.String,
        "original_line_id": pl.String,
        "vendor_title": pl.String,
        "excess_usd": pl.Float64,
        "excess_ratio": pl.Float64,
    }
    return World(
        vendors=vendors,
        roles=roles,
        rate_cards=rate_cards,
        fx=pl.DataFrame(fx_rows),
        invoice_lines=pl.DataFrame(
            lines, schema_overrides={"reported_hours": pl.Float64}, infer_schema_length=None
        ),
        title_truth=title_truth,
        planted=pl.DataFrame(planted, schema=planted_schema),
        as_of_month=months[-1],
        seed=seed,
    )


RAW_TABLES: Sequence[str] = ("vendors", "roles", "rate_cards", "fx", "invoice_lines")


def write_world(db_path: Path, world: World) -> None:
    """Replace the raw and restricted rate-card tables with this world."""
    with connect(db_path) as conn:
        conn.execute("create schema if not exists raw_ratecards; create schema if not exists restricted;")
        for name in RAW_TABLES:
            frame = getattr(world, name)
            conn.execute(f"create or replace table raw_ratecards.{name} as select * from frame")  # noqa: S608
        for name, frame_name in (("rc_title_truth", "title_truth"), ("rc_planted", "planted")):
            frame = getattr(world, frame_name)  # noqa: F841
            conn.execute(f"create or replace table restricted.{name} as select * from frame")  # noqa: S608
        conn.execute(
            "create or replace table raw_ratecards.world_meta as select ? as as_of_month, ? as seed",
            [world.as_of_month, world.seed],
        )
