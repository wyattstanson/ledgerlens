"""Variance bridge: actual minus plan as volume, mix, rate and timing, plus a residual (BUILD_PROMPT 10.4).

For segments s with quantity Q and amount A = Q x P (actual a, plan p), on the event basis (billed premium,
approved claims, payouts):

- volume = (sum Qa - sum Qp) x Pp_avg, with Pp_avg = sum Ap / sum Qp
- mix    = sum over s of (share_a,s - share_p,s) x sum Qa x Pp,s
- rate   = sum over s of (Pa,s - Pp,s) x Qa,s

These three add up to sum Aa - sum Ap exactly. Cash differs from the event basis because collections and
payouts lag, so

- timing = (cash_a - event_a) - (cash_p - event_p)

and volume + mix + rate + timing = cash_a - cash_p. Each part is rounded to cents and the residual takes the
rounding, so the bridge sums to the variance to the cent. A segment with no plan quantity is priced at its
actual price, so all of it shows as mix and none as rate.
"""

from collections.abc import Iterable
from dataclasses import dataclass

COMPONENTS = ("volume", "mix", "rate", "timing", "residual")


@dataclass(frozen=True)
class Segment:
    key: str
    q_actual: float
    amount_actual: float  # event basis
    q_plan: float
    amount_plan: float


@dataclass(frozen=True)
class Bridge:
    actual_cents: int  # cash basis
    plan_cents: int
    components: dict[str, int]

    @property
    def variance_cents(self) -> int:
        return self.actual_cents - self.plan_cents


def cents(x: float) -> int:
    return round(x * 100)


def variance_bridge(segments: Iterable[Segment], cash_actual: float, cash_plan: float) -> Bridge:
    segs = list(segments)
    qa = sum(s.q_actual for s in segs)
    qp = sum(s.q_plan for s in segs)
    ea = sum(s.amount_actual for s in segs)
    ep = sum(s.amount_plan for s in segs)
    p_plan_avg = ep / qp if qp else 0.0

    def plan_price(s: Segment) -> float:
        if s.q_plan:
            return s.amount_plan / s.q_plan
        return s.amount_actual / s.q_actual if s.q_actual else 0.0

    volume = (qa - qp) * p_plan_avg
    mix = sum(
        ((s.q_actual / qa if qa else 0.0) - (s.q_plan / qp if qp else 0.0)) * qa * plan_price(s) for s in segs
    )
    rate = sum((s.amount_actual / s.q_actual - plan_price(s)) * s.q_actual for s in segs if s.q_actual)
    timing = (cash_actual - ea) - (cash_plan - ep)
    parts = {"volume": cents(volume), "mix": cents(mix), "rate": cents(rate), "timing": cents(timing)}
    actual, plan = cents(cash_actual), cents(cash_plan)
    parts["residual"] = (actual - plan) - sum(parts.values())
    return Bridge(actual_cents=actual, plan_cents=plan, components=parts)


def combine(bridges: dict[str, Bridge], signs: dict[str, int]) -> Bridge:
    """A signed sum of bridges, e.g. net operating cash = premium - claims - LAE - opex. Exact in cents."""
    return Bridge(
        actual_cents=sum(signs[k] * b.actual_cents for k, b in bridges.items()),
        plan_cents=sum(signs[k] * b.plan_cents for k, b in bridges.items()),
        components={c: sum(signs[k] * b.components[c] for k, b in bridges.items()) for c in COMPONENTS},
    )
