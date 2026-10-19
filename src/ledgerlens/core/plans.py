"""Pre-registered analysis plans (rule 7): hash the validated plan before any compute, version it, log runs.

The hash covers the canonical JSON of the validated model, so comments and whitespace in the YAML never
create a new version, while any change to a threshold, test or exclusion does.
"""

import json
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel

from ledgerlens.core.db import connect, utcnow
from ledgerlens.core.lineage import new_id, sha256_text


@dataclass(frozen=True)
class RegisteredPlan:
    name: str
    version: int
    sha256: str
    plan_json: str


def canonical_json(plan: BaseModel) -> str:
    return json.dumps(plan.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))


def register_plan(db_path: Path, name: str, plan: BaseModel) -> RegisteredPlan:
    """Return the plan's version, creating version max+1 only if this exact plan is new."""
    body = canonical_json(plan)
    digest = sha256_text(body)
    with connect(db_path) as conn:
        row = conn.execute(
            "select plan_version from audit.analysis_plans where plan_name = ? and sha256 = ?", [name, digest]
        ).fetchone()
        if row is None:
            latest = conn.execute(
                "select coalesce(max(plan_version), 0) from audit.analysis_plans where plan_name = ?", [name]
            ).fetchone()
            version = int(latest[0] if latest else 0) + 1
            conn.execute(
                "insert into audit.analysis_plans "
                "(plan_name, plan_version, sha256, plan_json, registered_at) values (?, ?, ?, ?, ?)",
                [name, version, digest, body, utcnow()],
            )
        else:
            version = int(row[0])
    return RegisteredPlan(name=name, version=version, sha256=digest, plan_json=body)


@contextmanager
def analysis_run(db_path: Path, plan: RegisteredPlan, *, pack: str, seed: int) -> Iterator[str]:
    """Log a run against a registered plan before compute starts; mark it completed or failed at the end."""
    run_id = new_id("arun")
    with connect(db_path) as conn:
        conn.execute(
            "insert into audit.analysis_runs values (?, ?, ?, ?, ?, ?, ?, null, 'running')",
            [run_id, pack, plan.name, plan.version, plan.sha256, seed, utcnow()],
        )
    status = "failed"
    try:
        yield run_id
        status = "completed"
    finally:
        with connect(db_path) as conn:
            conn.execute(
                "update audit.analysis_runs set finished_at = ?, status = ? where analysis_run_id = ?",
                [utcnow(), status, run_id],
            )
