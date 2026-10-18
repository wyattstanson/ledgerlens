"""Vendor panel optimization with OR-Tools MathOpt + GSCIP (BUILD_PROMPT section 12.2, step 6).

    minimize   sum_vr p_vr * h_vr                     (effective USD per baseline hour x hours)
    subject to sum_v h_vr = D_r                       every role's demand is met
               h_vr <= D_r * keep_v                   only kept vendors supply hours
               sum_v keep_v <= N                      panel size
               sum_{v with a card for r} keep_v >= 2  critical roles keep two suppliers
               h_vr <= s_max * D_r                    no single supplier dominates a critical role
               keep_v = 0 if quality_v < q_min        quality floor

On infeasibility one constraint family at a time is relaxed, in configured order, and the result names the
family that had to give. It never returns silently: if no single relaxation works it raises.
"""

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, replace
from itertools import combinations

from ortools.math_opt.python import mathopt

from ledgerlens.ratecards.config import RelaxFamily

# A tiny cost per kept vendor makes the solver drop vendors that carry no hours and are not needed for
# coverage, so the reported panel is canonical. It is excluded from the reported cost.
KEEP_EPSILON = 1e-3
MIN_CRITICAL_SUPPLIERS = 2
HOURS_EPSILON = 1e-7  # solver noise below this is zero hours


@dataclass(frozen=True)
class PanelProblem:
    vendors: tuple[str, ...]
    quality: dict[str, float]
    demand: dict[str, float]  # role -> baseline hours
    price: dict[tuple[str, str], float]  # (vendor, role) -> effective USD per baseline hour
    panel_size: int
    critical: frozenset[str]
    max_share_critical: float
    min_quality: float

    def suppliers(self, role: str) -> list[str]:
        return [v for v in self.vendors if (v, role) in self.price]


@dataclass(frozen=True)
class PanelSolution:
    kept: tuple[str, ...]
    hours: dict[tuple[str, str], float]
    cost: float
    relaxed: RelaxFamily | None = None


class InfeasiblePanelError(RuntimeError):
    pass


def _solve(problem: PanelProblem, drop: frozenset[str]) -> PanelSolution | None:
    model = mathopt.Model(name="vendor_panel")
    keep = {v: model.add_binary_variable(name=f"keep[{v}]") for v in problem.vendors}
    hours = {
        (v, r): model.add_variable(lb=0.0, name=f"h[{v},{r}]")
        for (v, r) in problem.price
        if r in problem.demand
    }
    for role, demand in problem.demand.items():
        supply = [hours[(v, role)] for v in problem.suppliers(role)]
        if not supply:
            return None
        model.add_linear_constraint(sum(supply) == demand, name=f"demand[{role}]")
        cap = (
            problem.max_share_critical * demand
            if role in problem.critical and "concentration" not in drop
            else demand
        )
        for v in problem.suppliers(role):
            model.add_linear_constraint(hours[(v, role)] <= demand * keep[v])
            if cap < demand:
                model.add_linear_constraint(hours[(v, role)] <= cap)
    if "panel_size" not in drop:
        model.add_linear_constraint(sum(keep.values()) <= problem.panel_size, name="panel_size")
    if "critical_coverage" not in drop:
        for role in problem.critical:
            model.add_linear_constraint(
                sum(keep[v] for v in problem.suppliers(role)) >= MIN_CRITICAL_SUPPLIERS, name=f"cover[{role}]"
            )
    if "quality_floor" not in drop:
        for v in problem.vendors:
            if problem.quality[v] < problem.min_quality:
                model.add_linear_constraint(keep[v] == 0, name=f"quality[{v}]")

    cost_expr = sum(problem.price[k] * var for k, var in hours.items())
    model.minimize(cost_expr + KEEP_EPSILON * sum(keep.values()))
    params = mathopt.SolveParameters(relative_gap_tolerance=0.0, absolute_gap_tolerance=1e-7)
    result = mathopt.solve(model, mathopt.SolverType.GSCIP, params=params)
    reason = result.termination.reason
    if reason in (mathopt.TerminationReason.INFEASIBLE, mathopt.TerminationReason.INFEASIBLE_OR_UNBOUNDED):
        return None
    if reason != mathopt.TerminationReason.OPTIMAL:
        raise RuntimeError(f"panel solve ended with {reason.name}")
    values = result.variable_values()
    kept = tuple(v for v in problem.vendors if round(values[keep[v]]) == 1)
    allocation = {k: float(values[var]) for k, var in hours.items() if values[var] > HOURS_EPSILON}
    cost = sum(problem.price[k] * h for k, h in allocation.items())
    return PanelSolution(kept=kept, hours=allocation, cost=cost)


def solve_panel(problem: PanelProblem, relax_order: Sequence[RelaxFamily]) -> PanelSolution:
    solution = _solve(problem, frozenset())
    if solution is not None:
        return solution
    for family in relax_order:
        solution = _solve(problem, frozenset({family}))
        if solution is not None:
            return replace(solution, relaxed=family)
    eligible = {v for v in problem.vendors if problem.quality[v] >= problem.min_quality}
    thin = sorted(
        r
        for r in problem.critical
        if len(eligible.intersection(problem.suppliers(r))) < MIN_CRITICAL_SUPPLIERS
    )
    unsupplied = sorted(r for r in problem.demand if not problem.suppliers(r))
    details = [f"tried relaxing each of {list(relax_order)} alone"]
    if thin:
        details.append(f"critical roles with fewer than two eligible suppliers: {thin}")
    if unsupplied:
        details.append(f"roles with no hourly supplier: {unsupplied}")
    raise InfeasiblePanelError(
        "no single constraint relaxation makes the panel feasible; " + "; ".join(details)
    )


def check_solution(problem: PanelProblem, solution: PanelSolution, tol: float = 1e-6) -> list[str]:
    """Constraint violations in a solution (empty means it honours every constraint not relaxed)."""
    issues = []
    kept = set(solution.kept)
    if solution.relaxed != "panel_size" and len(kept) > problem.panel_size:
        issues.append(f"panel has {len(kept)} vendors, limit {problem.panel_size}")
    for role, demand in problem.demand.items():
        supplied = sum(h for (v, r), h in solution.hours.items() if r == role)
        if abs(supplied - demand) > tol * max(1.0, demand):
            issues.append(f"{role}: supplied {supplied:.4f} of {demand:.4f} hours")
        if role in problem.critical:
            if (
                solution.relaxed != "critical_coverage"
                and len(kept.intersection(problem.suppliers(role))) < MIN_CRITICAL_SUPPLIERS
            ):
                issues.append(f"{role}: critical role with fewer than two kept suppliers")
            if solution.relaxed != "concentration":
                for (v, r), h in solution.hours.items():
                    if r == role and h > problem.max_share_critical * demand * (1 + tol):
                        issues.append(f"{role}: {v} supplies more than the concentration cap")
    for (v, _r), h in solution.hours.items():
        if h > tol and v not in kept:
            issues.append(f"{v} supplies hours but is not kept")
    if solution.relaxed != "quality_floor":
        issues += [
            f"{v} is below the quality floor" for v in kept if problem.quality[v] < problem.min_quality
        ]
    return issues


def _greedy_allocation(problem: PanelProblem, subset: Iterable[str]) -> tuple[float, dict] | None:
    members = set(subset)
    cost, allocation = 0.0, {}
    for role, demand in problem.demand.items():
        cap = problem.max_share_critical * demand if role in problem.critical else demand
        remaining = demand
        for v in sorted(
            (v for v in problem.suppliers(role) if v in members), key=lambda v: problem.price[(v, role)]
        ):
            take = min(cap, remaining)
            if take > 0:
                allocation[(v, role)] = take
                cost += take * problem.price[(v, role)]
                remaining -= take
        if remaining > HOURS_EPSILON:
            return None
    return cost, allocation


def brute_force_panel(problem: PanelProblem) -> PanelSolution | None:
    """Exact answer by enumeration, for verifying the MILP on small instances (<= ~8 vendors).

    With the panel fixed, each role is an independent fractional fill: cheapest suppliers first, each
    up to its cap. So enumerating panels and filling greedily is exact.
    """
    eligible = [v for v in problem.vendors if problem.quality[v] >= problem.min_quality]
    best: PanelSolution | None = None
    for size in range(1, min(problem.panel_size, len(eligible)) + 1):
        for subset in combinations(eligible, size):
            if any(
                len(set(subset).intersection(problem.suppliers(r))) < MIN_CRITICAL_SUPPLIERS
                for r in problem.critical
            ):
                continue
            filled = _greedy_allocation(problem, subset)
            if filled is None:
                continue
            cost, allocation = filled
            if best is None or cost < best.cost - 1e-9:
                best = PanelSolution(kept=tuple(subset), hours=allocation, cost=cost)
    return best
