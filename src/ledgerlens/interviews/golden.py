"""Golden eval cases for the interviews pack, computed from the generator's ground truth.

Selection rates and hire counts come straight from the world's outcome tables; expected flag counts come from
the planted problems. Nothing here runs pipeline code, but rates are only expected for cells the published
suppression policy leaves visible (the same core rule, applied to the truth).
"""

import polars as pl

from ledgerlens.core.guards import suppress_small_groups
from ledgerlens.interviews.fairness import IMPACT_FUNNEL, IMPACT_LINES
from ledgerlens.interviews.generate import InterviewWorld
from ledgerlens.interviews.plan import AnalysisPlan

SOURCE = "generator:restricted.stage_outcomes+candidate_groups+performance+truth_planted"


def _case(case_id: str, tool: str, args: dict, expected: list[tuple[str, float]], tol: float = 1e-6) -> dict:
    return {
        "id": case_id,
        "pack": "interviews",
        "suite": "offline",
        "expected_source": SOURCE,
        "tool": tool,
        "input": args,
        "expected": [{"label": label, "value": round(value, 6), "abs_tol": tol} for label, value in expected],
    }


def golden_cases(world: InterviewWorld, plan: AnalysisPlan) -> list[dict]:
    outcomes = world.stage_outcomes.join(world.candidate_groups, on="candidate_id")
    cases: list[dict] = []
    cells = [
        {"role_id": r, "stage": s, "dimension": dim, "group": g, "n": n}
        for dim in plan.fairness.dimensions
        for r, s, g, n in outcomes.group_by(["role_id", "stage", dim]).len().iter_rows()
    ]
    published = suppress_small_groups(
        cells, value_keys=[], min_n=plan.min_group_n, lines=IMPACT_LINES, funnel=IMPACT_FUNNEL
    )
    visible = {
        (c["role_id"], c["stage"], c["dimension"], c["group"]) for c in published if not c["suppressed"]
    }

    def rate_cases(stage: str, dimension: str, roles: list[str]) -> None:
        for role in roles:
            cell = outcomes.filter((pl.col("role_id") == role) & (pl.col("stage") == stage))
            total = cell.height
            groups = (
                cell.group_by(dimension)
                .agg(pl.len().alias("n"), pl.col("passed").mean().alias("rate"))
                .sort(dimension)
            )
            expected = [
                (f"selection rate {g} {role} {stage}", float(r))
                for g, n, r in groups.iter_rows()
                if n >= plan.fairness.min_test_n
                and n / total >= plan.min_group_share
                and (role, stage, dimension, g) in visible
            ]
            if expected:
                cases.append(
                    _case(
                        f"iv-rate-{role}-{stage}-{dimension}",
                        "iv.impact_ratios",
                        {"role_id": role, "stage": stage, "dimension": dimension},
                        expected,
                    )
                )

    roles = world.roles["role_id"].to_list()
    rate_cases("screen", "sex", roles)
    rate_cases("technical", "sex", roles)
    rate_cases("screen", "race_ethnicity", roles[:5])

    planted = world.truth_planted["kind"].to_list()
    cases.append(
        _case(
            "iv-planted-flags",
            "iv.plan_status",
            {},
            [
                ("flagged interviewers", float(planted.count("lenient_interviewer"))),
                ("flagged panels", float(planted.count("random_panel"))),
                ("flagged fairness cells", float(planted.count("adverse_impact"))),
                ("non-predictive roles", float(planted.count("non_predictive_role"))),
                ("process flags", 0.0),
            ],
            tol=0,
        )
    )

    hires = world.performance.group_by("role_id").len().sort("role_id")
    for role, n in hires.iter_rows():
        cases.append(
            _case(f"iv-hires-{role}", "iv.predictive_validity", {}, [(f"hires {role}", float(n))], tol=0)
        )
    return cases
