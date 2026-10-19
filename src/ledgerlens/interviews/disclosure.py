"""Disclosure audit for the published selection tables (ADR 0004, decision 6).

Primary suppression hides every cell under the floor. A reader can still combine published numbers, so this
module writes down what a reader knows as a linear system and asks, for each small hidden cell, how far its
size and its passed and failed counts can be narrowed. A cell is disclosed when its size is confined to fewer
than ``floor`` possible values, or its passed or failed count to fewer than ``min_outcome_values``. Extra
cells are hidden only to repair a disclosure: the cheapest visible cell (an intersectional cell before a
one-way cell, Unknown or tiny groups before others, smaller before larger) among those in a constraint that
binds the exposed cell, found from the linear program's dual values.

What the reader is assumed to know (conservatively):
- every visible cell's n and selected count;
- each stage's total within the rounding of the published workload counts (nearest 10), each stage's selected
  total equals the next stage's total, and the hires per role exactly (predictive validity);
- each role x stage x dimension line sums to the stage total; intersectional cells sum to the one-way ones;
- a group's n at one stage equals its selected count at the stage before;
- for a visible tested group, h (published to 2 decimals) bounds everyone else's selection rate at that stage.
Not modelled (ADR 0004 residual risks): q values, the h intervals, flags and per-interviewer workloads.
"""

import math
import os
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

import polars as pl
from ortools.math_opt.python import mathopt

from ledgerlens.interviews.generate import STAGES

TOTAL_ROUNDING = 10  # stage totals leak only through counts published to the nearest 10
H_DECIMALS = 2
EPS = 1e-6
DUAL_EPS = 1e-9


@dataclass(frozen=True)
class Cell:
    key: str  # the published row_key
    stage: str
    dimension: str  # "sex", "race_ethnicity" or "cross"
    group: tuple[str, ...]  # (group,) for one-way cells, (sex, race_ethnicity) for intersectional cells
    n: int
    selected: int
    tier: int  # cost of hiding it: 0 intersectional, 1 Unknown or tiny one-way group, 2 other one-way group


@dataclass(frozen=True)
class RateBound:
    """A tested group at a stage: if its row is visible, the others' selection rate lies in [low, high]."""

    key: str
    stage: str
    n: int
    selected: int
    low: float
    high: float


@dataclass
class Expr:
    """A linear expression over the model's unknowns, remembering which visible cells fed its constant."""

    terms: dict[str, float] = field(default_factory=dict)
    const: float = 0.0
    cells: set[str] = field(default_factory=set)

    def plus(self, other: "Expr", weight: float = 1.0) -> "Expr":
        terms = dict(self.terms)
        for k, v in other.terms.items():
            terms[k] = terms.get(k, 0.0) + weight * v
        return Expr(terms, self.const + weight * other.const, self.cells | other.cells)


def rounded_interval(total: int, base: int = TOTAL_ROUNDING) -> tuple[int, int]:
    """What a reader knows about a total published to the nearest ``base`` (half up)."""
    shown = math.floor(total / base + 0.5) * base
    return shown - base // 2, shown + base // 2 - 1


def complement_rate_bounds(
    n: int, selected: int, h: float, decimals: int = H_DECIMALS
) -> tuple[float, float]:
    """Everyone else's selection rate implied by a group's rate and Cohen's h rounded to ``decimals``."""
    half = 0.5 * 10**-decimals
    base = math.asin(math.sqrt(selected / n))

    def rate(h_value: float) -> float:
        angle = min(max(base - h_value / 2, 0.0), math.pi / 2)
        return math.sin(angle) ** 2

    return rate(h + half), rate(h - half)  # the rate falls as h rises


def cells_from(fair: pl.DataFrame, cross: pl.DataFrame) -> dict[str, list[Cell]]:
    """Cells per role from the one-way and intersectional frames (true counts)."""
    by_role: dict[str, list[Cell]] = {}
    for r in fair.iter_rows(named=True):
        tier = 1 if r["group"] == "Unknown" or r["excluded_small_share"] else 2
        by_role.setdefault(r["role_id"], []).append(
            Cell(r["row_key"], r["stage"], r["dimension"], (r["group"],), r["n"], r["selected"], tier)
        )
    for r in cross.iter_rows(named=True):
        group = (r["sex"], r["race_ethnicity"])
        by_role.setdefault(r["role_id"], []).append(
            Cell(r["row_key"], r["stage"], "cross", group, r["n"], r["selected"], 0)
        )
    return by_role


def rate_bounds_from(fair: pl.DataFrame, decimals: int | None = H_DECIMALS) -> dict[str, list[RateBound]]:
    out: dict[str, list[RateBound]] = {}
    if decimals is None:
        return out
    for r in fair.filter(pl.col("tested") & pl.col("h").is_not_null()).iter_rows(named=True):
        low, high = complement_rate_bounds(r["n"], r["selected"], round(r["h"], decimals), decimals)
        out.setdefault(r["role_id"], []).append(
            RateBound(r["row_key"], r["stage"], r["n"], r["selected"], low, high)
        )
    return out


@dataclass(frozen=True)
class Exposure:
    values: dict[str, float]  # possible values a reader cannot rule out, per quantity
    binding: set[str]  # visible cells in constraints that bind the narrow quantities


class RoleModel:
    """The reader's linear system for one role, given which cells are hidden."""

    def __init__(
        self,
        cells: list[Cell],
        hidden: set[str],
        hires: int | None,
        rate_bounds: list[RateBound],
        *,
        integral: bool = False,
        known_totals: tuple[str, ...] = STAGES,
    ) -> None:
        self.cells = {c.key: c for c in cells}
        self.stages = [s for s in STAGES if any(c.stage == s for c in cells)]
        self.hidden = hidden
        self.hires = hires
        self.integral = integral
        self._ceilings: dict[tuple[str, str], float] = {}
        self.model = mathopt.Model(name="disclosure")
        self.vars: dict[str, mathopt.Variable] = {}
        self.rows: list[tuple[mathopt.LinearConstraint, set[str]]] = []
        for c in cells:
            if c.key in hidden:
                self._var(f"n|{c.key}", 0, math.inf)
                self._var(f"s|{c.key}", 0, math.inf)
        for s in self.stages:
            total = sum(c.n for c in cells if c.stage == s and c.dimension == "sex")
            self._var(f"T|{s}", *(rounded_interval(total) if s in known_totals else (0, math.inf)))
        if hires is None:
            self._var(f"S|{self.stages[-1]}", 0, math.inf)
        self._build(cells, rate_bounds)
        solver = mathopt.SolverType.GSCIP if integral else mathopt.SolverType.GLOP
        self.solver = mathopt.IncrementalSolver(self.model, solver)

    # --- expressions ------------------------------------------------------------------------------------

    def _var(self, name: str, low: float, high: float) -> None:
        self.vars[name] = self.model.add_variable(lb=low, ub=high, is_integer=self.integral, name=name)

    def n(self, c: Cell) -> Expr:
        return Expr({f"n|{c.key}": 1.0}) if c.key in self.hidden else Expr({}, float(c.n), {c.key})

    def s(self, c: Cell) -> Expr:
        return Expr({f"s|{c.key}": 1.0}) if c.key in self.hidden else Expr({}, float(c.selected), {c.key})

    def total(self, stage: str) -> Expr:
        return Expr({f"T|{stage}": 1.0})

    def selected_total(self, stage: str) -> Expr:
        later = self.stages.index(stage) + 1
        if later < len(self.stages):
            return self.total(self.stages[later])  # everyone who passes a stage enters the next one
        if self.hires is not None:
            return Expr({}, float(self.hires))
        return Expr({f"S|{stage}": 1.0})

    def _linear(self, e: Expr) -> Any:
        return sum((v * self.vars[k] for k, v in e.terms.items()), start=0.0)

    def _equal(self, e: Expr) -> None:
        if e.terms:
            row = self.model.add_linear_constraint(self._linear(e) == -e.const)
            self.rows.append((row, e.cells))

    def _at_most(self, e: Expr) -> None:
        if e.terms:
            row = self.model.add_linear_constraint(self._linear(e) <= -e.const)
            self.rows.append((row, e.cells))

    @staticmethod
    def _sum(parts: Iterable[Expr]) -> Expr:
        out = Expr()
        for p in parts:
            out = out.plus(p)
        return out

    def _build(self, cells: list[Cell], rate_bounds: list[RateBound]) -> None:
        self._build_lines(cells)
        self._build_funnels(cells)
        self._build_rate_bounds(rate_bounds)

    def _build_lines(self, cells: list[Cell]) -> None:
        one_way = {(c.stage, c.dimension, c.group[0]): c for c in cells if c.dimension != "cross"}
        lines: dict[tuple[str, str], list[Cell]] = {}
        for c in cells:
            lines.setdefault((c.stage, c.dimension), []).append(c)
            if c.key in self.hidden:
                self._at_most(self.s(c).plus(self.n(c), -1.0))  # selected <= n
        for (stage, dimension), members in lines.items():
            if dimension != "cross":
                self._equal(self._sum(self.n(m) for m in members).plus(self.total(stage), -1.0))
                self._equal(self._sum(self.s(m) for m in members).plus(self.selected_total(stage), -1.0))
                continue
            for axis, name in ((0, "sex"), (1, "race_ethnicity")):
                for value in sorted({m.group[axis] for m in members}):
                    margin = one_way.get((stage, name, value))
                    if margin is None:
                        continue
                    part = [m for m in members if m.group[axis] == value]
                    self._equal(self._sum(self.n(m) for m in part).plus(self.n(margin), -1.0))
                    self._equal(self._sum(self.s(m) for m in part).plus(self.s(margin), -1.0))

    def _build_funnels(self, cells: list[Cell]) -> None:
        chains: dict[tuple[str, tuple[str, ...]], dict[str, Cell]] = {}
        for c in cells:
            chains.setdefault((c.dimension, c.group), {})[c.stage] = c
        for chain in chains.values():
            for before, after in zip(self.stages, self.stages[1:], strict=False):
                if before in chain and after in chain:
                    self._equal(self.n(chain[after]).plus(self.s(chain[before]), -1.0))

    def _build_rate_bounds(self, rate_bounds: list[RateBound]) -> None:
        for b in rate_bounds:
            if b.key in self.hidden:
                continue  # a hidden row publishes no h
            visible = {b.key}
            rest_s = self.selected_total(b.stage).plus(Expr({}, float(b.selected), visible), -1.0)
            rest_n = self.total(b.stage).plus(Expr({}, float(b.n), visible), -1.0)
            self._at_most(rest_s.plus(rest_n, -b.high))  # rest_s <= high * rest_n
            self._at_most(Expr().plus(rest_n, b.low).plus(rest_s, -1.0))  # low * rest_n <= rest_s

    # --- what a reader can narrow ------------------------------------------------------------------------

    def _extreme(self, e: Expr, maximize: bool) -> tuple[float, set[str]]:
        objective = self._linear(e)
        if maximize:
            self.model.maximize(objective)
        else:
            self.model.minimize(objective)
        result = self.solver.solve()
        reason = result.termination.reason
        if reason == mathopt.TerminationReason.UNBOUNDED:
            return (math.inf if maximize else -math.inf), set()
        if reason != mathopt.TerminationReason.OPTIMAL:
            raise RuntimeError(f"disclosure audit model has no optimum: {result.termination}")
        binding: set[str] = set()
        if not self.integral:
            duals = result.dual_values()
            for row, cells in self.rows:
                if abs(duals[row]) > DUAL_EPS:
                    binding |= cells
        return result.objective_value() + e.const, binding

    def _quantity(self, key: str, name: str) -> Expr:
        c = self.cells[key]
        return {"n": self.n(c), "selected": self.s(c), "failed": self.n(c).plus(self.s(c), -1.0)}[name]

    def stage_ceiling(self, stage: str, name: str) -> float:
        """The most a whole stage can have of ``name`` (entrants, passes or failures), given what is known."""
        if (stage, name) not in self._ceilings:
            total, passed = self.total(stage), self.selected_total(stage)
            e = {"n": total, "selected": passed, "failed": total.plus(passed, -1.0)}[name]
            self._ceilings[(stage, name)] = self._extreme(e, maximize=True)[0] if e.terms else e.const
        return self._ceilings[(stage, name)]

    def interval(self, key: str, name: str) -> tuple[float, float, set[str]]:
        """The range a reader can confine ``name`` (n, selected or failed) of a cell to, and what binds it."""
        e = self._quantity(key, name)
        if not e.terms:
            return e.const, e.const, set()
        low, b_low = self._extreme(e, maximize=False)
        high, b_high = self._extreme(e, maximize=True)
        return low, high, b_low | b_high

    def exposure(self, key: str, need: dict[str, int] | None = None) -> Exposure:
        """Possible values per quantity; with ``need``, stops at the first quantity that falls short."""
        values: dict[str, float] = {}
        binding: set[str] = set()
        for name in ("n", "selected", "failed"):
            low, high, bound = self.interval(key, name)
            values[name] = math.inf if math.isinf(high) else math.floor(high + EPS) - math.ceil(low - EPS) + 1
            if need is not None and values[name] < need[name]:
                return Exposure(values, bound)
            binding |= bound
        return Exposure(values, binding)


def _neighbours(cells: list[Cell]) -> dict[str, set[str]]:
    """Cells sharing a line, a margin or a funnel step: hiding one adds an unknown to the other's relation."""
    groups: dict[tuple[Any, ...], list[str]] = {}
    for c in cells:
        if c.dimension == "cross":
            groups.setdefault(("sex", c.stage, c.group[0]), []).append(c.key)
            groups.setdefault(("race_ethnicity", c.stage, c.group[1]), []).append(c.key)
        else:
            groups.setdefault(("line", c.stage, c.dimension), []).append(c.key)
            groups.setdefault((c.dimension, c.stage, c.group[0]), []).append(c.key)
        groups.setdefault(("subject", c.dimension, c.group), []).append(c.key)
    out: dict[str, set[str]] = {c.key: set() for c in cells}
    for members in groups.values():
        for m in members:
            out[m].update(members)
    for k, v in out.items():
        v.discard(k)
    return out


def cheapest(keys: Iterable[str], cells: dict[str, Cell]) -> str | None:
    return min(keys, key=lambda k: (cells[k].tier, cells[k].n, k), default=None)


def _nearest_visible(
    start: str, cells: dict[str, Cell], graph: dict[str, set[str]], hidden: set[str]
) -> str | None:
    seen, layer = {start}, [start]
    while layer:
        nxt = sorted({n for k in layer for n in graph[k]} - seen)
        pick = cheapest((k for k in nxt if k not in hidden), cells)
        if pick is not None:
            return pick
        seen.update(nxt)
        layer = nxt
    return None


@dataclass(frozen=True)
class Policy:
    floor: int  # a small cell's size must keep at least this many possible values
    min_outcome_values: int  # ...and its passed and failed counts at least this many
    h_decimals: int | None = None  # precision of a published h against everyone else; None: not published
    known_totals: tuple[str, ...] = STAGES  # stages whose total a reader can bound to the rounding


def exposed(model: RoleModel, keys: Iterable[str], policy: Policy) -> dict[str, Exposure]:
    """Small cells a reader can narrow too far.

    A group's count can never have more possible values than its whole stage allows: if a reader can tell
    that at most 3 people failed a stage, every group's failed count is somewhere in 0..3, and that range
    is as protected as it gets.
    """
    want = {"n": policy.floor, "selected": policy.min_outcome_values, "failed": policy.min_outcome_values}
    out = {}
    for key in keys:
        stage = model.cells[key].stage
        need = {q: min(v, math.floor(model.stage_ceiling(stage, q) + EPS) + 1) for q, v in want.items()}
        e = model.exposure(key, need)
        if any(e.values.get(q, math.inf) < need[q] for q in need):
            out[key] = e
    return out


def plan_role(
    cells: list[Cell], hires: int | None, bounds: list[RateBound], policy: Policy, integral: bool
) -> set[str]:
    """Hidden row keys for one role (roles share no relation, so each is planned on its own)."""
    by_key = {c.key: c for c in cells}
    graph = _neighbours(cells)
    small = sorted(c.key for c in cells if c.n < policy.floor)
    hidden = set(small)
    for exact in (False, True) if integral else (False,):
        safe: set[str] = set()  # hiding more only widens what a reader can't rule out
        while True:
            todo = [k for k in small if k not in safe]
            model = RoleModel(cells, hidden, hires, bounds, integral=exact, known_totals=policy.known_totals)
            found = exposed(model, todo, policy)
            safe.update(k for k in todo if k not in found)
            if not found:
                break
            # Duals come from the relaxation; exposed cells pointing at the same neighbour share it.
            lp = RoleModel(cells, hidden, hires, bounds, known_totals=policy.known_totals) if exact else None
            picks = set()
            for key, e in found.items():
                binding = (lp.exposure(key).binding if lp else e.binding) - hidden
                pick = cheapest(binding, by_key) or _nearest_visible(key, by_key, graph, hidden)
                if pick is None:
                    raise RuntimeError(f"cannot protect {key}: every related cell is already hidden")
                picks.add(pick)
            hidden |= picks
    return hidden


def plan_hidden(
    fair: pl.DataFrame,
    cross: pl.DataFrame,
    hires: dict[str, int],
    policy: Policy,
    *,
    integral: bool = True,
    workers: int | None = None,
) -> set[str]:
    """Row keys to hide: every cell under the floor, plus the fewest cheap cells that stop any disclosure.

    The search runs on the linear relaxation; with ``integral`` the result is then checked, and repaired if
    needed, against a reader who reasons in whole numbers. Roles are planned in parallel threads.
    """
    bounds = rate_bounds_from(fair, policy.h_decimals)
    jobs = [
        (cells, hires.get(role), bounds.get(role, []), policy, integral)
        for role, cells in sorted(cells_from(fair, cross).items())
    ]
    workers = min(len(jobs), os.cpu_count() or 1) if workers is None else workers
    if workers <= 1:
        return set().union(*(plan_role(*job) for job in jobs))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return set().union(*pool.map(plan_role, *zip(*jobs, strict=True)))
