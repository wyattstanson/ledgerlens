"""Billed title -> standard role mapping with a human review queue (BUILD_PROMPT section 12.2, step 2).

1. Normalize the text (abbreviations, level codes).
2. Shortlist the top roles with rapidfuzz ``token_set_ratio``; rank the shortlist with ``token_sort_ratio``,
   which penalizes missing words (``token_set_ratio`` scores "backend engineer" 100 against every seniority).
3. Pick: offline, accept only a clear winner; with an LLM picker, accept only when its confidence is
   >= 0.85 and it agrees with the fuzzy top-1 (spec rule).
4. Everything else goes to the review queue as an approval record. Every mapping stores who or what
   decided it.
"""

import json
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import duckdb
import polars as pl
from rapidfuzz import fuzz, process
from rapidfuzz.distance import OSA

from ledgerlens.core.approvals import ApprovalQueue, DecisionKind
from ledgerlens.core.db import connect
from ledgerlens.core.guards import fence_untrusted
from ledgerlens.core.llm import LLMClient
from ledgerlens.ratecards.config import MappingConfig

MAPPING_ACTION = "rc.approve_mapping"
FUZZY_RULE = "rule:fuzzy-v1"

ABBREVIATIONS = {
    "jr": "junior",
    "sr": "senior",
    "ld": "lead",
    "snr": "senior",
    "eng": "engineer",
    "engr": "engineer",
    "mgr": "manager",
    "proj": "project",
    "anlst": "analyst",
    "sci": "scientist",
    "bus": "business",
    "svc": "service",
    "auto": "automation",
    "fs": "full stack",
    "sre": "site reliability engineer",
    "app": "application",
    "dev": "developer",
}
LEVEL_CODES = {
    "i": "junior",
    "ii": "mid",
    "iii": "senior",
    "iv": "lead",
    "l1": "junior",
    "l2": "mid",
    "l3": "senior",
    "l4": "lead",
}
SENIORITY_WORDS = {"junior", "mid", "senior", "lead"}
MIN_CORRECTABLE_LENGTH = 4  # shorter tokens are too ambiguous to spell-correct
_PUNCT = re.compile(r"[^a-z0-9 ]+")


def _expand(title: str) -> list[str]:
    text = _PUNCT.sub(" ", title.lower().replace("full-stack", "full stack"))
    words: list[str] = []
    for token in text.split():
        words.extend((LEVEL_CODES.get(token) or ABBREVIATIONS.get(token) or token).split())
    return words


def role_vocabulary(roles: pl.DataFrame) -> frozenset[str]:
    """Every word used in standard role titles, plus the seniority words."""
    return frozenset(SENIORITY_WORDS.union(*(_expand(t) for t in roles["title"].to_list())))


def _spell_correct(token: str, vocabulary: frozenset[str]) -> str:
    """Fix a one-edit typo (including adjacent swaps) against the role vocabulary, if unambiguous."""
    if token in vocabulary or len(token) < MIN_CORRECTABLE_LENGTH:
        return token
    close = [w for w in vocabulary if OSA.distance(token, w) <= 1]
    return close[0] if len(close) == 1 else token


def normalize_title(title: str, vocabulary: frozenset[str] = frozenset()) -> str:
    """Lowercase, expand abbreviations and level codes, fix one-edit typos against the role vocabulary.
    An unqualified title is read as mid-level, the usual HR convention ("Data Engineer" means
    "Data Engineer II")."""
    words = [_spell_correct(w, vocabulary) for w in _expand(title)] if vocabulary else _expand(title)
    if not SENIORITY_WORDS.intersection(words):
        words.insert(0, "mid")
    return " ".join(words)


def title_seniority(normalized: str) -> str:
    return next(w for w in normalized.split() if w in SENIORITY_WORDS)


@dataclass(frozen=True)
class Candidate:
    role_id: str
    shortlist_score: float  # token_set_ratio
    rank_score: float  # token_sort_ratio
    seniority_match: bool  # candidate's seniority equals the title's


def shortlist(
    title: str, roles: pl.DataFrame, k: int, vocabulary: frozenset[str] | None = None
) -> list[Candidate]:
    vocab = role_vocabulary(roles) if vocabulary is None else vocabulary
    normalized = normalize_title(title, vocab)
    choices = {row["role_id"]: normalize_title(row["title"]) for row in roles.iter_rows(named=True)}
    hits = process.extract(normalized, choices, scorer=fuzz.token_set_ratio, limit=k)
    # A title that names its seniority ("Senior ...", "... III") rules out roles of another seniority.
    seniority = title_seniority(normalized)
    ranked = [
        Candidate(
            rid,
            float(score),
            float(fuzz.token_sort_ratio(normalized, choices[rid])),
            rid.endswith(f".{seniority}"),
        )
        for _, score, rid in hits
    ]
    return sorted(ranked, key=lambda c: (not c.seniority_match, -c.rank_score, -c.shortlist_score, c.role_id))


@dataclass(frozen=True)
class Pick:
    role_id: str | None  # None = send to review
    suggested_role_id: str
    confidence: float
    decided_by: str
    reason: str


class TitlePicker(Protocol):
    def pick(self, title: str, candidates: Sequence[Candidate], roles: pl.DataFrame) -> Pick: ...


class FuzzyPicker:
    """Deterministic, no API needed: accept only a clear, strong winner."""

    def __init__(self, config: MappingConfig) -> None:
        self.config = config

    def pick(self, title: str, candidates: Sequence[Candidate], roles: pl.DataFrame) -> Pick:
        top = candidates[0]
        rivals = [c for c in candidates[1:] if c.seniority_match]
        margin = top.rank_score - (rivals[0].rank_score if rivals else 0.0)
        clear = (
            top.seniority_match
            and top.rank_score >= self.config.offline_accept_score
            and margin >= self.config.offline_accept_margin
        )
        return Pick(
            role_id=top.role_id if clear else None,
            suggested_role_id=top.role_id,
            confidence=round(top.rank_score / 100, 4),
            decided_by=FUZZY_RULE,
            reason=f"token_sort {top.rank_score:.0f}, margin {margin:.0f}",
        )


class LlmPicker:
    """Claude Haiku picks from the shortlist; accepted only when confident and agreeing with fuzzy top-1."""

    SYSTEM = (
        "You map a contractor's billed job title to one standard role from a shortlist. "
        "Pick the closest role, give a confidence between 0 and 1, and a one-sentence reason. "
        "If none fits, pick the closest and give low confidence."
    )

    def __init__(self, llm: LLMClient, config: MappingConfig) -> None:
        self.llm = llm
        self.config = config

    def pick(self, title: str, candidates: Sequence[Candidate], roles: pl.DataFrame) -> Pick:
        titles = dict(roles.select("role_id", "title").iter_rows())
        options = "\n".join(f"- {c.role_id}: {titles[c.role_id]}" for c in candidates)
        schema = {
            "type": "object",
            "properties": {
                "role_id": {"type": "string", "enum": [c.role_id for c in candidates]},
                "confidence": {"type": "number"},
                "reason": {"type": "string"},
            },
            "required": ["role_id", "confidence", "reason"],
            "additionalProperties": False,
        }
        prompt = f"Billed title:\n{fence_untrusted(title, 'invoice:billed_title')}\n\nShortlist:\n{options}"
        response = self.llm.create(
            task="title_mapping",
            system=self.SYSTEM,
            messages=[{"role": "user", "content": prompt}],
            output_schema=schema,
            max_tokens=1024,
        )
        text = next(b.text for b in response.message.content if b.type == "text")
        answer = json.loads(text)
        confidence = min(max(float(answer["confidence"]), 0.0), 1.0)
        agrees = answer["role_id"] == candidates[0].role_id
        accepted = confidence >= self.config.llm_accept_confidence and agrees
        return Pick(
            role_id=answer["role_id"] if accepted else None,
            suggested_role_id=answer["role_id"],
            confidence=confidence,
            decided_by=f"llm:{self.llm.config.route('title_mapping').model}",
            reason=str(answer["reason"])[:500],
        )


TITLE_MAP_SCHEMA = {
    "billed_title": pl.String,
    "role_id": pl.String,
    "suggested_role_id": pl.String,
    "status": pl.String,
    "confidence": pl.Float64,
    "decided_by": pl.String,
    "reason": pl.String,
    "shortlist": pl.String,
    "approval_id": pl.String,
}


def map_titles(
    titles: Sequence[str], roles: pl.DataFrame, picker: TitlePicker, config: MappingConfig
) -> pl.DataFrame:
    rows = []
    vocabulary = role_vocabulary(roles)
    for title in sorted(set(titles)):
        candidates = shortlist(title, roles, config.shortlist_size, vocabulary)
        pick = picker.pick(title, candidates, roles)
        rows.append(
            {
                "billed_title": title,
                "role_id": pick.role_id,
                "suggested_role_id": pick.suggested_role_id,
                "status": "accepted" if pick.role_id else "pending_review",
                "confidence": pick.confidence,
                "decided_by": pick.decided_by if pick.role_id else None,
                "reason": pick.reason,
                "shortlist": json.dumps([c.role_id for c in candidates]),
                "approval_id": None,
            }
        )
    return pl.DataFrame(rows, schema=TITLE_MAP_SCHEMA)


def queue_reviews(title_map: pl.DataFrame, approvals: ApprovalQueue, *, agent_run_id: str) -> pl.DataFrame:
    """Create one approval request per title that was not auto-accepted."""
    ids = []
    for row in title_map.iter_rows(named=True):
        if row["status"] != "pending_review":
            ids.append(row["approval_id"])
            continue
        payload = {
            "billed_title": row["billed_title"],
            "suggested_role_id": row["suggested_role_id"],
            "shortlist": json.loads(row["shortlist"]),
        }
        ids.append(
            approvals.propose(agent_run_id=agent_run_id, action=MAPPING_ACTION, payload=payload).approval_id
        )
    return title_map.with_columns(pl.Series("approval_id", ids, dtype=pl.String))


def save_title_map(db_path: Path, title_map: pl.DataFrame) -> None:
    with connect(db_path) as conn:
        conn.execute("create schema if not exists ratecards")
        conn.execute("create or replace table ratecards.title_map as select * from title_map")


def load_title_map(db_path: Path) -> pl.DataFrame:
    with connect(db_path) as conn:
        return conn.execute("select * from ratecards.title_map order by billed_title").pl()


def resolve_reviews(
    db_path: Path,
    approvals: ApprovalQueue,
    choices: dict[str, str],
    *,
    decided_by: str,
    reason: str,
    kind: DecisionKind = DecisionKind.HUMAN,
) -> None:
    """A reviewer picks the role for queued titles (approval_id -> role_id).

    The approvals and the title-map updates commit together or not at all, and every chosen role must
    exist in the standard role catalog.
    """
    if not choices:
        return
    with connect(db_path) as conn:
        known = {r[0] for r in conn.execute("select role_id from raw_ratecards.roles").fetchall()}
    unknown = sorted(set(choices.values()) - known)
    if unknown:
        raise ValueError(f"unknown role ids: {unknown}")

    def apply_mappings(conn: duckdb.DuckDBPyConnection) -> None:
        for aid, rid in choices.items():
            updated = conn.execute(
                "update ratecards.title_map set role_id = ?, status = 'reviewed', decided_by = ? "
                "where approval_id = ? and status = 'pending_review' returning 1",
                [rid, decided_by, aid],
            ).fetchall()
            if not updated:
                raise ValueError(f"approval {aid} has no pending title in ratecards.title_map")

    approvals.decide_many(
        {aid: f"{reason} (role {rid})" for aid, rid in choices.items()},
        approve=True,
        decided_by=decided_by,
        action=MAPPING_ACTION,
        kind=kind,
        in_transaction=apply_mappings,
    )


def resolve_review(
    db_path: Path, approvals: ApprovalQueue, approval_id: str, *, role_id: str, decided_by: str, reason: str
) -> None:
    resolve_reviews(db_path, approvals, {approval_id: role_id}, decided_by=decided_by, reason=reason)


SIMULATED_REVIEWER = "simulated:synthetic-truth"


def simulate_review_from_truth(db_path: Path, approvals: ApprovalQueue) -> int:
    """Demo and eval only: stand in for the human reviewer using the hidden truth. Audited as simulated."""
    with connect(db_path) as conn:
        pending = conn.execute(
            """
            select m.approval_id, t.role_id
            from ratecards.title_map m join restricted.rc_title_truth t using (billed_title)
            where m.status = 'pending_review'
            """
        ).fetchall()
    resolve_reviews(
        db_path,
        approvals,
        dict(pending),
        decided_by=SIMULATED_REVIEWER,
        reason="simulated review",
        kind=DecisionKind.SIMULATED,
    )
    return len(pending)


def evaluate_mapping(title_map: pl.DataFrame, truth: pl.DataFrame) -> dict[str, float]:
    """Accuracy against the hidden map. Reviewed rows are counted as the reviewer decided them."""
    joined = title_map.join(truth, on="billed_title", how="left", suffix="_true")
    total = joined.height
    auto = joined.filter(pl.col("status") == "accepted")
    review = joined.filter(pl.col("status") != "accepted")
    correct_auto = auto.filter(pl.col("role_id") == pl.col("role_id_true")).height
    correct_final = joined.filter(pl.col("role_id") == pl.col("role_id_true")).height
    return {
        "titles": float(total),
        "auto_accepted": float(auto.height),
        "auto_accuracy": correct_auto / auto.height if auto.height else 0.0,
        "review_rate": review.height / total if total else 0.0,
        "accuracy_after_review": correct_final / total if total else 0.0,
    }
