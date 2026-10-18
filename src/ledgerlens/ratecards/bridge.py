"""Savings bridge (BUILD_PROMPT section 12.2, step 7): current cost -> new cost, every dollar traced to lines.

Steps, in order, all in integer cents so they sum exactly:

1. Leakage recovery: duplicate lines (full amount) and overbilled excess above the card.
2. Rate harmonization (kept vendors): from the line's clean cost to the vendor's harmonized panel rate,
   min(card rate, role/tier median), at the as-of exchange rate.
3. Volume shift (kept vendors): from that rate to the role's new average panel rate, as hours move to the
   optimal mix.
4. Vendor exits (vendors not kept): from the line's clean cost to the role's new average panel rate.

Lines that cannot be re-priced (unmapped titles, fixed-price lines without reported hours, roles with no
hourly supplier) pass through unchanged. Identity per line and in total:
current = leakage + harmonization + volume shift + exits + new.
"""

from dataclasses import dataclass

import polars as pl

from ledgerlens.ratecards.optimizer import PanelProblem, PanelSolution

STEPS = ("leakage_recovery", "rate_harmonization", "volume_shift", "vendor_exits")


def _cents(x: float) -> int:
    return round(x * 100)


@dataclass(frozen=True)
class SavingsBridge:
    current_cents: int
    new_cents: int
    steps: dict[str, int]  # step -> savings in cents (positive = saving)
    lines: pl.DataFrame  # line_id, step, cents
    optimizer_cost_check_cents: int  # new cost of re-priced lines from the allocation, for comparison


def harmonized_prices(
    problem: PanelProblem, solution: PanelSolution, medians: dict[str, float]
) -> dict[tuple[str, str], float]:
    """Kept vendors' prices after harmonization: never above the median for the same role key."""
    return {
        (v, r): min(p, medians.get(r, p))
        for (v, r), p in problem.price.items()
        if v in solution.kept and r in problem.demand
    }


def build_bridge(
    baseline_lines: pl.DataFrame,
    leakage: pl.DataFrame,
    problem: PanelProblem,
    solution: PanelSolution,
    harmonized: dict[tuple[str, str], float],
) -> SavingsBridge:
    duplicate_ids = set(leakage.filter(pl.col("kind") == "duplicate")["line_id"])
    overbill = dict(leakage.filter(pl.col("kind") == "overbill").select("line_id", "leakage_usd").iter_rows())

    new_role_cost = {r: 0.0 for r in problem.demand}
    for (v, r), h in solution.hours.items():
        new_role_cost[r] += h * harmonized[(v, r)]
    avg_price = {r: new_role_cost[r] / d for r, d in problem.demand.items() if d > 0}

    kept = set(solution.kept)
    attributions: list[tuple[str, str, int]] = []
    current = new = check = 0
    for line in baseline_lines.iter_rows(named=True):
        line_id, amount_c = line["line_id"], _cents(line["amount"])
        current += amount_c
        if line_id in duplicate_ids:
            attributions.append((line_id, "leakage_recovery", amount_c))
            continue
        leak_c = _cents(overbill.get(line_id, 0.0))
        if leak_c:
            attributions.append((line_id, "leakage_recovery", leak_c))
        clean_c = amount_c - leak_c
        role, vendor, hours = line["role_key"], line["entity"], line["baseline_hours"]
        if role not in avg_price or hours is None:
            new += clean_c  # pass-through
            continue
        target_c = _cents(hours * avg_price[role])
        check += target_c
        if vendor in kept and (vendor, role) in harmonized:
            harmonized_c = _cents(hours * harmonized[(vendor, role)])
            attributions.append((line_id, "rate_harmonization", clean_c - harmonized_c))
            attributions.append((line_id, "volume_shift", harmonized_c - target_c))
        else:
            attributions.append((line_id, "vendor_exits", clean_c - target_c))
        new += target_c

    lines = pl.DataFrame(
        attributions, schema={"line_id": pl.String, "step": pl.String, "cents": pl.Int64}, orient="row"
    )
    totals = dict(lines.group_by("step").agg(pl.col("cents").sum()).iter_rows()) if lines.height else {}
    steps = {s: int(totals.get(s, 0)) for s in STEPS}
    return SavingsBridge(
        current_cents=current,
        new_cents=new,
        steps=steps,
        lines=lines,
        optimizer_cost_check_cents=_cents(sum(new_role_cost.values())) - check,
    )
