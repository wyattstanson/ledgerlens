"""Evidence tagging of interview notes (BUILD_PROMPT 11.3, ADR 0004).

- ``RuleTagger`` (default, offline): a note cites evidence if it quotes the candidate, reports a quantified
  result, or names an observed behaviour. The cue list was written from general interviewing guidance;
  because the same author wrote the synthetic note templates, agreement with generator truth overstates
  real accuracy. The owner's spot-check labels are the real check.
- ``ClaudeTagger`` (optional): Claude Haiku via the Message Batches API, structured output, notes fenced as
  untrusted data. Runs only when an API key is configured.
"""

import csv
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from ledgerlens.core.guards import fence_untrusted
from ledgerlens.core.llm import LLMClient
from ledgerlens.core.stats import cohens_kappa

QUOTED = re.compile(r'"[^"]{8,}"')
QUANTIFIED = re.compile(r"\b\d+(\.\d+)?\s?(%|percent|ms|x\b|hours?|days?|minutes?)", re.IGNORECASE)
BEHAVIOUR_CUES = (
    "walked through",
    "gave the example",
    "observed",
    "described a time",
    "step by step",
    "quote:",
    "wrote",
    "found the",
    "drew",
    "rolled back",
    "measured",
    "split the",
    "mapped",
    "paired",
)


@dataclass(frozen=True)
class NoteToTag:
    note_id: str
    text: str
    competency_id: str


class EvidenceTagger(Protocol):
    name: str

    def tag(self, notes: Sequence[NoteToTag]) -> dict[str, bool]: ...


class RuleTagger:
    name = "rules"

    def tag(self, notes: Sequence[NoteToTag]) -> dict[str, bool]:
        out = {}
        for note in notes:
            lowered = note.text.lower()
            cited = QUOTED.search(note.text) or QUANTIFIED.search(note.text)
            out[note.note_id] = bool(cited) or any(cue in lowered for cue in BEHAVIOUR_CUES)
        return out


class ClaudeTagger:
    """Haiku tags notes in one batch. A malformed or failed item becomes "no evidence" and is counted."""

    name = "claude"
    SYSTEM = (
        "You label interview notes. A note cites evidence when it quotes the candidate or describes "
        "a specific behaviour the interviewer observed. Opinions, impressions and generic praise "
        "are not evidence."
    )

    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm
        self.failures = 0

    def tag(self, notes: Sequence[NoteToTag]) -> dict[str, bool]:
        items = {}
        for note in notes:
            items[note.note_id] = [
                {
                    "role": "user",
                    "content": f"Competency: {note.competency_id}\n{fence_untrusted(note.text, 'note')}",
                }
            ]
        schema = {
            "type": "object",
            "properties": {
                "has_evidence": {"type": "boolean"},
                "evidence_span": {"type": ["string", "null"]},
                "competency_id": {"type": "string", "enum": sorted({n.competency_id for n in notes})},
            },
            "required": ["has_evidence", "evidence_span", "competency_id"],
            "additionalProperties": False,
        }
        results = self.llm.create_batch(
            task="evidence_tagging", system=self.SYSTEM, items=items, output_schema=schema, max_tokens=512
        )
        out = {}
        for note in notes:
            message = results.get(note.note_id)
            try:
                text = next(b.text for b in message.content if b.type == "text") if message else ""
                out[note.note_id] = bool(json.loads(text)["has_evidence"])
            except (StopIteration, json.JSONDecodeError, KeyError, TypeError):
                self.failures += 1
                out[note.note_id] = False
        return out


def evaluation_sample(note_ids: Sequence[str], size: int, seed: int) -> list[str]:
    """A seeded sample of note ids; the spot-check set is its first ``spot_check_size`` ids."""
    rng = np.random.default_rng(seed)
    ordered = sorted(note_ids)
    return [ordered[i] for i in rng.choice(len(ordered), size=min(size, len(ordered)), replace=False)]


def write_spot_check(path: Path, notes: Sequence[NoteToTag]) -> None:
    """Blind labelling sheet: note text only; no truth, tagger output, candidate or interviewer ids."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["note_id", "note_text", "owner_has_evidence"])
        for note in notes:
            writer.writerow([note.note_id, note.text, ""])


def read_owner_labels(path: Path, notes: Mapping[str, str] | None = None) -> dict[str, bool]:
    """Labels the owner filled in (yes/no, y/n, true/false, 1/0); blank rows are skipped.

    Pass ``notes`` (note id -> current text) to keep only labels whose sheet text still matches: note ids
    are positional, so a sheet written for another seed would otherwise label different notes.
    """
    if not path.exists():
        return {}
    truthy, falsy = {"yes", "y", "true", "1"}, {"no", "n", "false", "0"}
    labels = {}
    with path.open(newline="", encoding="utf-8-sig") as handle:  # Excel's "CSV UTF-8" adds a BOM
        for row in csv.DictReader(handle):
            note_id, value = row.get("note_id"), (row.get("owner_has_evidence") or "").strip().lower()
            if not note_id or value not in truthy | falsy:
                continue
            if notes is None or notes.get(note_id) == row.get("note_text"):
                labels[note_id] = value in truthy
    return labels


def agreement(a: dict[str, bool], b: dict[str, bool]) -> tuple[int, float | None]:
    """(n, Cohen's kappa) over the notes both label sets cover."""
    shared = sorted(set(a) & set(b))
    if not shared:
        return 0, None
    return len(shared), cohens_kappa([a[k] for k in shared], [b[k] for k in shared])
