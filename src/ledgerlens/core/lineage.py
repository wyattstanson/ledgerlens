"""Lineage ids, audit writers and the answer lineage check (BUILD_PROMPT sections 9.3 and 9.4).

Every number a user sees must trace to a lineage row: the SQL, source tables and row ids that produced
it. ``verify_answer`` enforces that for model-written text.
"""

import hashlib
import math
import re
import uuid
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict

from ledgerlens.core.db import connect


class NumberRef(BaseModel):
    """A number with provenance. Tools return these; final answers must cite them."""

    model_config = ConfigDict(extra="forbid")

    label: str
    value: float
    unit: str
    lineage_id: str


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex}"


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def pending_lineage(
    *,
    tool_call_id: str,
    label: str,
    value: float,
    unit: str,
    sql_text: str,
    source_tables: Sequence[str],
    row_ids: Sequence[str | int],
) -> tuple[NumberRef, list[Any]]:
    """The number and the row that will back it, without touching the database.

    Opening a DuckDB connection costs about a tenth of a second on a warehouse of any size, so a tool
    that cites a hundred numbers cannot afford one connection each. Callers collect these rows and
    write them together with ``write_lineage``; the number carries its id from the moment it is made.
    """
    if not math.isfinite(value):
        raise ValueError(f"lineage value for {label!r} must be finite, got {value!r}")
    lineage_id = new_id("ln")
    row = [
        lineage_id,
        tool_call_id,
        sha256_text(sql_text),
        sql_text,
        list(source_tables),
        [str(r) for r in row_ids],
        float(value),
        unit,
    ]
    return NumberRef(label=label, value=float(value), unit=unit, lineage_id=lineage_id), row


def write_lineage(db_path: Path, rows: Sequence[Sequence[Any]]) -> None:
    """Write a batch of lineage rows on one connection."""
    if not rows:
        return
    with connect(db_path) as conn:
        conn.executemany("insert into audit.lineage values (?, ?, ?, ?, ?, ?, ?, ?)", [list(r) for r in rows])


def record_lineage(
    db_path: Path,
    *,
    tool_call_id: str,
    label: str,
    value: float,
    unit: str,
    sql_text: str,
    source_tables: Sequence[str],
    row_ids: Sequence[str | int],
) -> NumberRef:
    """Persist one lineage row and return the number that cites it. Row ids are stored as text."""
    number, row = pending_lineage(
        tool_call_id=tool_call_id,
        label=label,
        value=value,
        unit=unit,
        sql_text=sql_text,
        source_tables=source_tables,
        row_ids=row_ids,
    )
    write_lineage(db_path, [row])
    return number


def lineage_values_for_run(db_path: Path, agent_run_id: str) -> dict[str, float]:
    """lineage_id -> recorded value, for every lineage row produced by tool calls in this run."""
    # Plain connection: DuckDB refuses mixed read-only/read-write configs on one file in one process.
    with connect(db_path) as conn:
        rows = conn.execute(
            """
            select l.lineage_id, l.value
            from audit.lineage l
            join audit.tool_calls t using (tool_call_id)
            where t.agent_run_id = ?
            """,
            [agent_run_id],
        ).fetchall()
    return {lineage_id: value for lineage_id, value in rows}


# --- numeric tokens -------------------------------------------------------------------------------

_SCALE = {"k": 1e3, "m": 1e6, "b": 1e9, "thousand": 1e3, "million": 1e6, "billion": 1e9}
# Not preceded by a digit or a decimal point. Digits glued to a word ("Q3", "USD9400000") are matched too;
# extract_numeric_tokens lets a single glued digit pass as an identifier and checks everything else.
_NUMBER = re.compile(
    r"(?<![\d.])(?P<neg>-)?(?P<cur>\$)?"
    r"(?P<num>\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)"
    r"(?P<suffix>[kKmMbB](?![A-Za-z])|\s(?:thousand|million|billion)\b)?"
    r"(?P<pct>%|\s?per\s?cent\b)?",
    re.IGNORECASE,
)
# Numbers spelled out cannot be checked against citations. Small counts ("three vendors") stay allowed.
_NUMBER_WORDS = re.compile(
    r"\b(?:eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|"
    r"fifty|sixty|seventy|eighty|ninety|hundred|thousand|million|billion|trillion|dozen)s?\b",
    re.IGNORECASE,
)
NOT_PERCENT_UNITS = frozenset({"USD", "USD/hour", "count", "days"})  # a "%" cannot cite these


@dataclass(frozen=True)
class NumericToken:
    text: str
    value: float
    tolerance: float
    is_percent: bool
    is_currency: bool = False


def extract_numeric_tokens(text: str) -> list[NumericToken]:
    """Every number written in ``text``, with the rounding tolerance its displayed precision implies.

    "$2.1M" is 2,100,000 give or take 50,000; "12.3%" is 12.3 give or take 0.05; "9.4 million" is
    9,400,000 give or take 50,000. A single digit glued to a word ("Q3", "v2") is an identifier, not a number.
    """
    tokens: list[NumericToken] = []
    for match in _NUMBER.finditer(text):
        raw = match.group("num").replace(",", "")
        glued = match.start() > 0 and (text[match.start() - 1].isalpha() or text[match.start() - 1] == "_")
        if glued and len(raw) == 1 and not (match.group("suffix") or match.group("pct")):
            continue
        decimals = len(raw.split(".")[1]) if "." in raw else 0
        suffix = (match.group("suffix") or "").strip().lower()
        scale = _SCALE[suffix] if suffix else 1.0
        value = float(raw) * scale * (-1 if match.group("neg") else 1)
        tolerance = 0.5 * 10.0**-decimals * scale
        tokens.append(
            NumericToken(
                text=match.group(0),
                value=value,
                tolerance=tolerance,
                is_percent=bool(match.group("pct")),
                is_currency=bool(match.group("cur")),
            )
        )
    return tokens


def spelled_out_numbers(text: str) -> list[str]:
    """Number words in ``text`` that are not the scale of a written number ("9.4 million" is fine)."""
    without_numbers = _NUMBER.sub(" ", text)
    return [m.group(0) for m in _NUMBER_WORDS.finditer(without_numbers)]


def _matches(token: NumericToken, ref: NumberRef) -> bool:
    if token.is_percent and ref.unit in NOT_PERCENT_UNITS:
        return False
    if token.is_currency and not ref.unit.startswith("USD"):
        return False
    value = ref.value
    candidates = [value, value * 100] if token.is_percent else [value]
    slack = token.tolerance + 1e-9 * max(1.0, abs(token.value))
    # Narration often drops the sign ("fell by $2.1M" for -2.1M), so compare magnitudes.
    return any(abs(abs(token.value) - abs(c)) <= slack for c in candidates)


def verify_answer(
    answer_text: str, numbers: Iterable[NumberRef], lineage_values: Mapping[str, float]
) -> list[str]:
    """Return lineage failures for an answer; an empty list means it passes.

    1. Every numeric token in the text must match a cited number within rounding tolerance, and a
       percentage or dollar amount must cite a number with a compatible unit.
    2. Every cited number's lineage_id must exist for this run.
    3. Every cited value must equal the value recorded in lineage, so a real id cannot vouch for
       an invented number.
    4. Numbers above ten may not be spelled out, because words cannot be checked.
    """
    cited = list(numbers)
    failures: list[str] = []
    for ref in cited:
        recorded = lineage_values.get(ref.lineage_id)
        if recorded is None:
            failures.append(f"{ref.label}: lineage_id {ref.lineage_id} was not produced by this run")
        elif not math.isclose(ref.value, recorded, rel_tol=1e-9, abs_tol=1e-9):
            failures.append(f"{ref.label}: cited value {ref.value} differs from recorded value {recorded}")
    for token in extract_numeric_tokens(answer_text):
        if not any(_matches(token, ref) for ref in cited):
            failures.append(f"'{token.text}' in the answer does not match any cited number")
    for word in spelled_out_numbers(answer_text):
        failures.append(f"'{word}' is a number in words; write it in digits and cite it")
    return failures
