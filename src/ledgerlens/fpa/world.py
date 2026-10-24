"""What the FP&A pipeline needs from a synthetic world, so the same steps run on the insurer and the
processor.

A world supplies its driver table, its accounting periods, a variance bridge per measure for any set of
months, its built-in scenario, and the words that label its answers (entity, book, income statement basis).
The pipeline does the rest the same way for both: materiality flags, confidence, run metadata and the tables
the tools read. Each world's ``measures`` say how its measures combine into net operating cash.
"""

from collections.abc import Callable
from dataclasses import dataclass
from datetime import date

import polars as pl

from ledgerlens.fpa.bridge import Bridge, Segment
from ledgerlens.fpa.config import Variant
from ledgerlens.fpa.materiality import Kind
from ledgerlens.fpa.scenario import ScenarioResult


@dataclass(frozen=True)
class WorldView:
    variant: Variant
    entity: str  # ADR 0001 answer scope
    book: str
    income_basis: str  # how the income statement recognises each line
    measures: dict[str, tuple[Kind, int]]  # measure -> (kind, sign in net operating cash)
    drivers: pl.DataFrame  # fpa.drivers.DRIVER_SCHEMA
    periods: pl.DataFrame  # id, periodname, month, closed
    bridges_for: Callable[[list[str]], dict[str, Bridge]]  # every measure and net operating cash
    scenario: Callable[[date], ScenarioResult]  # the built-in scenario as of a date


def segments(
    actual: pl.DataFrame, plan: pl.DataFrame, keys: list[str], q: tuple[str, str], a: tuple[str, str]
) -> list[Segment]:
    """Bridge segments joined on ``keys``; q and a name the (actual, plan) quantity and amount columns."""
    act = actual.group_by(keys).agg(pl.col(q[0]).sum().alias("qa"), pl.col(a[0]).sum().alias("aa"))
    pln = plan.group_by(keys).agg(pl.col(q[1]).sum().alias("qp"), pl.col(a[1]).sum().alias("ap"))
    joined = act.join(pln, on=keys, how="full", coalesce=True).fill_null(0.0)
    return [
        Segment("|".join(str(r[k]) for k in keys), r["qa"], r["aa"], r["qp"], r["ap"])
        for r in joined.sort(keys).iter_rows(named=True)
    ]
