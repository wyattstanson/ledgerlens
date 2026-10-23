"""What an FP&A adapter hands back: the rows, and the query and tables they came from, for lineage.

A real feed swapped in behind ``PaymentsFeed`` or ``LedgerAdapter`` reports its own source (an MCP call, a
saved-search id), so lineage always cites what actually ran.
"""

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Rows:
    rows: list[dict[str, Any]]
    sql: str
    tables: list[str]
    units: dict[str, str] = field(default_factory=dict)  # column -> unit, where the source declares them
