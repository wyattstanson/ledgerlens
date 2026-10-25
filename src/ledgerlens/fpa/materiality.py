"""Variance flags under the ADR 0001 materiality rules, applied by compute, never by the model.

- A variance is **material** at a level when it reaches the level's dollar amount or its percentage of plan
  (financial statement: 5% or $50,000; operating: 3% or $25,000). Reaching the threshold counts.
- It is **immaterial** when it is under both the immaterial-screen amount and percentage (2% and $10,000).
- Anything between is **watch**.

Before a material variance is flagged, compute checks the three things a reviewer would: a documented
one-time item (``one_time``), and a reclassification, an equal and opposite variance in another expense
measure of the same month (``reclass``). Seasonality is not a reason to flag: the plan already carries it
(seasonal claim frequency and billing by days in the month), so a variance is measured against a seasonal
plan (ADR 0005).

Net operating cash is the statement-level total and uses the financial statement thresholds; its components
are operating measures (the insurer's premium collected, claims paid, LAE and opex; the processor's fee
revenue, interchange, chargebacks and opex). The reclassification screen pairs any two expense measures of a
world, since a ledger reclass can move an amount between any two expense accounts. A material variance is
labelled ``CRITICAL:`` at the financial statement level and ``FLAG:`` at the operating level (ADR 0001 risks).
"""

import math
from dataclasses import dataclass
from typing import Literal

from ledgerlens.fpa.config import MaterialityConfig, MaterialityLevel

Status = Literal["material", "watch", "immaterial", "one_time", "reclass"]
Kind = Literal["revenue", "expense", "net"]
Level = Literal["financial_statement", "operating"]
RECLASS_TOLERANCE = 0.02  # an offset within 2% of the variance counts as the other half of a reclass
STATEMENT_LEVEL_MEASURES = frozenset({"net_operating_cash"})


def level_for(measure: str) -> Level:
    return "financial_statement" if measure in STATEMENT_LEVEL_MEASURES else "operating"


def risk_label(status: str, level: str) -> str | None:
    """The ADR 0001 risk label for a variance: only material ones are labelled."""
    if status != "material":
        return None
    return "CRITICAL:" if level == "financial_statement" else "FLAG:"


@dataclass(frozen=True)
class Variance:
    month: str
    measure: str
    kind: Kind
    level: Level
    actual: float
    plan: float

    @property
    def variance(self) -> float:
        return self.actual - self.plan

    @property
    def pct(self) -> float:
        return abs(self.variance) / abs(self.plan) if self.plan else math.inf

    @property
    def favourable(self) -> bool:
        """Revenue or net cash above plan is favourable; an expense below plan is favourable."""
        return self.variance < 0 if self.kind == "expense" else self.variance > 0


def classify(variance: float, plan: float, level: MaterialityLevel, immaterial: MaterialityLevel) -> Status:
    """Percentages compare by multiplication, so a variance of exactly pct x plan reaches the threshold.
    With no plan the percentage is undefined and only the dollar amounts apply."""
    size, base = abs(variance), abs(plan)
    if size >= level.usd or (base > 0 and size >= level.pct * base):
        return "material"
    if size < immaterial.usd and (base == 0 or size < immaterial.pct * base):
        return "immaterial"
    return "watch"


def flag(
    variances: list[Variance], config: MaterialityConfig, one_time_items: tuple[str, ...]
) -> list[Status]:
    """Status per variance, in input order."""
    out: list[Status] = []
    for v in variances:
        level = getattr(config, v.level)
        status = classify(v.variance, v.plan, level, config.immaterial)
        if status == "material" and f"{v.month}:{v.measure}" in one_time_items:
            status = "one_time"
        elif status == "material" and v.kind == "expense" and _has_offset(v, variances):
            status = "reclass"
        out.append(status)
    return out


def _has_offset(v: Variance, variances: list[Variance]) -> bool:
    for other in variances:
        if other is v or other.month != v.month or other.kind != "expense":
            continue
        if abs(v.variance + other.variance) <= RECLASS_TOLERANCE * abs(v.variance):
            return True
    return False
