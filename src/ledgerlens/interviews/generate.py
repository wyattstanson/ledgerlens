"""Seeded synthetic interview world with four planted problems (BUILD_PROMPT 11.1, ADR 0004).

Everything person-level (candidates, interviews, ratings, notes, outcomes, performance, group attributes)
and the hidden truth is written to the restricted vault. The warehouse gets reference data only: roles,
competencies with anchored descriptions, interviewer ids and panel membership.

Planted problems (``truth_planted``):
1. one lenient interviewer, +0.8 on every rating in one role;
2. one onsite panel whose ratings are random (Fleiss' kappa near 0);
3. one stage in the busiest role where one group's passes are kept at 65%, an impact ratio near 0.65;
4. one role whose later performance is unrelated to interview scores.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl

from ledgerlens.core.db import connect
from ledgerlens.core.stats import spearman
from ledgerlens.interviews.config import WorldConfig

STAGES = ("screen", "technical", "onsite", "hiring_manager")
SCREENERS, TECHNICAL = slice(0, 4), slice(4, 8)  # positions in a role's interviewer pool
QUESTION_ASKED_RATE = 0.98
# The non-predictive role resamples performance until its sample correlation is near zero.
NONPREDICTIVE_MAX_RHO, NONPREDICTIVE_ATTEMPTS, MIN_RANKABLE = 0.03, 200, 3
ROLES = (
    ("backend", "Backend Engineer"),
    ("frontend", "Frontend Engineer"),
    ("data_science", "Data Scientist"),
    ("data_eng", "Data Engineer"),
    ("product", "Product Manager"),
    ("qa", "QA Engineer"),
    ("devops", "DevOps Engineer"),
    ("eng_manager", "Engineering Manager"),
)
COMPETENCIES = {
    "problem_solving": (
        "restates the problem only after prompting",
        "breaks the problem down and tests ideas unprompted",
    ),
    "coding": ("code does not run or ignores edge cases", "clean, tested code that handles edge cases"),
    "system_design": ("cannot name the main components", "weighs trade-offs and failure modes with numbers"),
    "communication": ("answers are hard to follow", "explains clearly and checks understanding"),
    "collaboration": ("dismisses other views", "builds on others' ideas and credits them"),
    "ownership": ("blames others for outcomes", "describes owning a result end to end"),
    "data_modeling": ("no clear entities or keys", "normalized model with justified denormalization"),
    "product_sense": ("features without users", "ties choices to user problems and metrics"),
    "testing": ("no testing strategy", "risk-based test plan with automation"),
    "leadership": ("no examples of leading", "concrete examples of growing a team"),
}
ROLE_COMPETENCIES = {
    "backend": ("problem_solving", "coding", "system_design", "communication", "ownership"),
    "frontend": ("problem_solving", "coding", "communication", "collaboration"),
    "data_science": ("problem_solving", "coding", "data_modeling", "communication", "product_sense"),
    "data_eng": ("problem_solving", "coding", "data_modeling", "system_design", "ownership", "testing"),
    "product": ("product_sense", "communication", "collaboration", "leadership"),
    "qa": ("problem_solving", "testing", "coding", "communication"),
    "devops": ("problem_solving", "system_design", "ownership", "collaboration", "testing"),
    "eng_manager": (
        "leadership",
        "communication",
        "collaboration",
        "system_design",
        "ownership",
        "product_sense",
    ),
}
SEX = (("Male", 0.52), ("Female", 0.45), ("Unknown", 0.03))
RACE_ETHNICITY = (
    ("Hispanic or Latino", 0.18),
    ("White", 0.44),
    ("Black or African American", 0.13),
    ("Asian", 0.16),
    ("Native Hawaiian or Pacific Islander", 0.01),
    ("Native American or Alaska Native", 0.012),
    ("Two or More Races", 0.03),
    ("Unknown", 0.038),
)

EVIDENCE_NOTES = (
    'Said "{q}" when asked about {c}.',
    "Walked through {c} step by step: {d}.",
    "Gave the example of {d}, with the result and their own part in it.",
    "Observed them {d} without prompting.",
    'Quote: "{q}". Followed up with a concrete case on {c}.',
    "Described a time they {d}; named the trade-off they made.",
    "Their answer on {c} included a 40% latency cut after adding an index.",  # evidence with no cue word
    "Asked why, they explained they {d} and showed the numbers.",
)
VAGUE_NOTES = (
    "Seemed solid on {c}.",
    "Good vibe, would enjoy working with them.",
    "Strong {c}, I think.",
    "Not convinced about {c}.",
    "Explained that they are a great team player.",  # decoy: a cue word, no observed behaviour
    "Talked a lot about {c} in general terms.",
    "Mentioned a project they led on {c}, but gave no detail.",  # sounds concrete, is not
    "Clearly the strongest candidate I have seen for {c}.",
)
# Rare notes whose label a cue-based tagger gets wrong, so agreement stays realistic rather than perfect.
HARD_EVIDENCE_NOTES = ("Their whiteboard sketch for {c} matched what we actually run in production.",)
HARD_VAGUE_NOTES = ('Said "I really love clean code" about {c}.',)
HARD_NOTE_RATE = 0.04
EVIDENCE_DETAILS = (
    "split the service by tenant to cut lock contention",
    "wrote a failing test before the fix",
    "rolled back a release within ten minutes",
    "mapped the user journey before choosing features",
    "found the off-by-one in their own loop",
    "drew the read path and its cache",
)
QUOTES = (
    "I'd shard by tenant id first",
    "we measured p95 before and after",
    "I owned the rollback plan",
    "the metric we moved was activation",
    "I paired with the junior to unblock them",
)


@dataclass
class InterviewWorld:
    roles: pl.DataFrame
    competencies: pl.DataFrame
    interviewers: pl.DataFrame
    panels: pl.DataFrame
    candidates: pl.DataFrame
    candidate_groups: pl.DataFrame
    interviews: pl.DataFrame
    ratings: pl.DataFrame
    notes: pl.DataFrame
    stage_outcomes: pl.DataFrame
    performance: pl.DataFrame
    truth_planted: pl.DataFrame
    truth_note_labels: pl.DataFrame
    seed: int


def _draw(rng: np.random.Generator, options: tuple[tuple[str, float], ...], n: int) -> list[str]:
    labels, weights = zip(*options, strict=True)
    p = np.asarray(weights) / sum(weights)
    return [str(x) for x in rng.choice(labels, size=n, p=p)]


def generate_world(config: WorldConfig, seed: int) -> InterviewWorld:  # noqa: PLR0912, PLR0915 (one linear recipe)
    rng = np.random.default_rng(seed)
    role_ids = [r for r, _ in ROLES]
    planted_role, random_panel_role, nonpredictive_role = "backend", "frontend", "data_science"

    # --- interviewers, pools and panels ----------------------------------------------------------
    interviewer_ids = [f"INT-{i + 1:02d}" for i in range(config.interviewers)]
    pools: dict[str, list[str]] = {r: [] for r in role_ids}
    for i, iid in enumerate(interviewer_ids):  # each interviewer covers two or three roles, evenly
        for k in range(2 + (i % 2)):
            pools[role_ids[(i + k * 3) % len(role_ids)]].append(iid)
    lenient = pools[planted_role][0]
    screeners = {r: pools[r][SCREENERS] for r in role_ids}
    technical = {
        r: pools[r][TECHNICAL] if len(pools[r]) >= TECHNICAL.stop else pools[r][SCREENERS] for r in role_ids
    }
    managers = {r: pools[r][-1] for r in role_ids}
    panels: dict[str, list[str]] = {}
    for r in role_ids:
        # The lenient interviewer stays off panels so one planted problem does not leak into a clean panel.
        panel_pool = [x for x in pools[r] if x not in (managers[r], lenient)]
        for p in range(config.panels_per_role):
            size = 3 + (p % 2)
            start = (p * 2) % max(1, len(panel_pool) - size)
            panels[f"PNL-{r}-{p + 1}"] = panel_pool[start : start + size]
    random_panel = f"PNL-{random_panel_role}-1"
    bias = {iid: float(rng.normal(0, 0.06)) for iid in interviewer_ids}

    # --- candidates and groups -----------------------------------------------------------------------
    n = config.candidates
    other = (1 - config.planted_role_share) / (len(role_ids) - 1)
    cand_roles = rng.choice(
        role_ids, size=n, p=[config.planted_role_share if r == planted_role else other for r in role_ids]
    )
    cand_ids = [f"CAND-{i + 1:05d}" for i in range(n)]
    theta = rng.normal(0, 1, n)
    sex = _draw(rng, SEX, n)
    race = _draw(rng, RACE_ETHNICITY, n)
    evidence_rate = {iid: float(rng.uniform(*config.evidence_rate_range)) for iid in interviewer_ids}

    interviews, ratings, notes, labels, outcomes = [], [], [], [], []
    stage_scores: dict[tuple[str, str], dict[str, float]] = {}
    alive = dict.fromkeys(cand_ids, True)
    panel_cycle = {r: 0 for r in role_ids}
    competency_skill = {
        cid: {c: theta[i] + rng.normal(0, 0.5) for c in ROLE_COMPETENCIES[str(cand_roles[i])]}
        for i, cid in enumerate(cand_ids)
    }

    def interview(cid: str, role: str, stage: str, iid: str, panel_id: str | None) -> list[int]:
        interview_id = f"IV-{len(interviews) + 1:06d}"
        interviews.append(
            {
                "interview_id": interview_id,
                "candidate_id": cid,
                "role_id": role,
                "stage": stage,
                "interviewer_id": iid,
                "panel_id": panel_id,
            }
        )
        values = []
        for c in ROLE_COMPETENCIES[role]:
            if panel_id == random_panel:
                value = int(rng.integers(1, 5))
            else:
                shift = config.lenient_shift if (iid == lenient and role == planted_role) else 0.0
                raw = (
                    2.5
                    + 0.8 * competency_skill[cid][c]
                    + bias[iid]
                    + shift
                    + rng.normal(0, config.rating_noise)
                )
                value = int(np.clip(np.rint(raw), 1, 4))
            rating_id = f"RT-{len(ratings) + 1:07d}"
            ratings.append(
                {
                    "rating_id": rating_id,
                    "interview_id": interview_id,
                    "competency_id": c,
                    "rating": value,
                    "question_asked": bool(rng.random() < QUESTION_ASKED_RATE),
                }
            )
            has_evidence = bool(rng.random() < evidence_rate[iid])
            hard = rng.random() < HARD_NOTE_RATE
            options = (
                (HARD_EVIDENCE_NOTES if hard else EVIDENCE_NOTES)
                if has_evidence
                else (HARD_VAGUE_NOTES if hard else VAGUE_NOTES)
            )
            template = rng.choice(options)
            text = str(template).format(
                q=rng.choice(QUOTES), c=c.replace("_", " "), d=rng.choice(EVIDENCE_DETAILS)
            )
            note_id = f"NT-{len(notes) + 1:07d}"
            notes.append({"note_id": note_id, "rating_id": rating_id, "text": text})
            labels.append({"note_id": note_id, "has_evidence": has_evidence})
            values.append(value)
        return values

    index = {cid: k for k, cid in enumerate(cand_ids)}
    for stage in STAGES:
        for role in role_ids:
            members = [cid for i, cid in enumerate(cand_ids) if cand_roles[i] == role and alive[cid]]
            scores: dict[str, float] = {}
            for j, cid in enumerate(members):
                if stage == "onsite":
                    pid = f"PNL-{role}-{panel_cycle[role] % config.panels_per_role + 1}"
                    panel_cycle[role] += 1
                    vals = [v for iid in panels[pid] for v in interview(cid, role, stage, iid, pid)]
                else:
                    pool = {"screen": screeners, "technical": technical}.get(stage)
                    iid = managers[role] if pool is None else pool[role][j % len(pool[role])]
                    vals = interview(cid, role, stage, iid, None)
                scores[cid] = float(np.mean(vals)) + float(rng.normal(0, 0.2))
            stage_scores[(role, stage)] = scores
            if not scores:
                continue
            cut = float(np.quantile(list(scores.values()), 1 - getattr(config.pass_rates, stage)))
            for cid, score in scores.items():
                passed = score >= cut
                if passed and stage == "technical" and role == planted_role and sex[index[cid]] == "Female":
                    passed = bool(rng.random() < config.adverse_keep_rate)
                outcomes.append({"candidate_id": cid, "role_id": role, "stage": stage, "passed": passed})
                alive[cid] = passed

    # --- performance of hires 9-12 months later --------------------------------------------------------
    performance = []
    for role in role_ids:
        hires = [
            o["candidate_id"]
            for o in outcomes
            if o["role_id"] == role and o["stage"] == "hiring_manager" and o["passed"]
        ]
        onsite = stage_scores[(role, "onsite")]
        score = np.array([onsite[c] for c in hires])
        perf = np.zeros(len(hires))
        for attempt in range(NONPREDICTIVE_ATTEMPTS):
            noise = rng.normal(0, 1, len(hires))
            if role == nonpredictive_role:
                perf = noise  # unrelated to skill: interview scores cannot predict it
                near_zero = len(hires) < MIN_RANKABLE or abs(spearman(score, perf)) < NONPREDICTIVE_MAX_RHO
                if near_zero or attempt == NONPREDICTIVE_ATTEMPTS - 1:
                    break
            else:
                perf = 0.9 * np.array([theta[index[c]] for c in hires]) + 0.4 * noise
                break
        values = (
            np.clip(np.rint(3 + 0.9 * (perf - perf.mean()) / (perf.std() or 1.0)), 1, 5) if hires else perf
        )
        performance += [
            {"candidate_id": c, "role_id": role, "performance_rating": int(v)}
            for c, v in zip(hires, values, strict=True)
        ]

    truth = [
        {
            "kind": "lenient_interviewer",
            "role_id": planted_role,
            "stage": None,
            "interviewer_id": lenient,
            "panel_id": None,
            "dimension": None,
            "group_value": None,
        },
        {
            "kind": "random_panel",
            "role_id": random_panel_role,
            "stage": "onsite",
            "interviewer_id": None,
            "panel_id": random_panel,
            "dimension": None,
            "group_value": None,
        },
        {
            "kind": "adverse_impact",
            "role_id": planted_role,
            "stage": "technical",
            "interviewer_id": None,
            "panel_id": None,
            "dimension": "sex",
            "group_value": "Female",
        },
        {
            "kind": "non_predictive_role",
            "role_id": nonpredictive_role,
            "stage": None,
            "interviewer_id": None,
            "panel_id": None,
            "dimension": None,
            "group_value": None,
        },
    ]
    return InterviewWorld(
        roles=pl.DataFrame({"role_id": role_ids, "title": [t for _, t in ROLES]}),
        competencies=pl.DataFrame(
            [
                {
                    "role_id": r,
                    "competency_id": c,
                    "anchor_1": COMPETENCIES[c][0],
                    "anchor_4": COMPETENCIES[c][1],
                }
                for r in role_ids
                for c in ROLE_COMPETENCIES[r]
            ]
        ),
        interviewers=pl.DataFrame({"interviewer_id": interviewer_ids}),
        panels=pl.DataFrame(
            [
                {"panel_id": p, "role_id": p.split("-")[1], "interviewer_id": m}
                for p, ms in panels.items()
                for m in ms
            ]
        ),
        candidates=pl.DataFrame({"candidate_id": cand_ids, "role_id": [str(r) for r in cand_roles]}),
        candidate_groups=pl.DataFrame({"candidate_id": cand_ids, "sex": sex, "race_ethnicity": race}),
        interviews=pl.DataFrame(interviews, schema_overrides={"panel_id": pl.String}),
        ratings=pl.DataFrame(ratings),
        notes=pl.DataFrame(notes),
        stage_outcomes=pl.DataFrame(outcomes),
        performance=pl.DataFrame(performance),
        truth_planted=pl.DataFrame(truth, schema={k: pl.String for k in truth[0]}),
        truth_note_labels=pl.DataFrame(labels),
        seed=seed,
    )


VAULT_TABLES = (
    "candidates",
    "candidate_groups",
    "interviews",
    "ratings",
    "notes",
    "stage_outcomes",
    "performance",
    "truth_planted",
    "truth_note_labels",
)
REFERENCE_TABLES = ("roles", "competencies", "interviewers", "panels")


def write_world(warehouse: Path, vault: Path, world: InterviewWorld) -> None:
    """Person-level data and truth to the vault; reference data only to the warehouse."""
    with connect(vault) as conn:
        conn.execute("create schema if not exists restricted")
        for name in VAULT_TABLES:
            frame = getattr(world, name)
            conn.execute(f"create or replace table restricted.{name} as select * from frame")  # noqa: S608
        conn.execute("create or replace table restricted.world_meta as select ? as seed", [world.seed])
    with connect(warehouse) as conn:
        conn.execute("create schema if not exists raw_interviews")
        for name in REFERENCE_TABLES:
            frame = getattr(world, name)  # noqa: F841
            conn.execute(f"create or replace table raw_interviews.{name} as select * from frame")  # noqa: S608
