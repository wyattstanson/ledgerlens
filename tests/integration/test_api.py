"""The web service: tools over HTTP keep their governance, and the UI's own routes work.

Every assertion here is about a rule the API must not let slip: write tools stay unreachable, numbers
arrive with a lineage id that resolves, approvals need a named human, and a shared deployment cannot
decide at all.
"""

from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from ledgerlens.api import deps
from ledgerlens.api.main import app
from ledgerlens.core.approvals import ApprovalQueue
from ledgerlens.core.db import connect
from ledgerlens.core.lineage import new_id
from ledgerlens.settings import Settings


@pytest.fixture
def config(db: Path) -> Settings:
    return Settings(db=db, allow_approvals=False)


def _client(config: Settings) -> Iterator[TestClient]:
    """A client whose settings point at the test warehouse.

    Not a context manager: that would run the lifespan, which resolves the real settings and would
    touch the repository's own warehouse. The ``db`` fixture has already created the audit schema.
    """
    app.dependency_overrides[deps.settings] = lambda: config
    yield TestClient(app)
    app.dependency_overrides.clear()


@pytest.fixture
def client(config: Settings) -> Iterator[TestClient]:
    yield from _client(config)


@pytest.fixture
def deciding_client(db: Path) -> Iterator[TestClient]:
    yield from _client(Settings(db=db, allow_approvals=True))


def test_healthz_answers_without_touching_the_warehouse(client: TestClient):
    response = client.get("/healthz")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_the_catalogue_lists_read_tools_and_hides_write_tools(client: TestClient):
    names = {t["name"] for t in client.get("/api/tools").json()["tools"]}
    assert "core.llm_spend" in names
    assert "rc.accept_panel" not in names, "a write tool must never appear in the HTTP catalogue"


@pytest.mark.parametrize(
    "write_tool", ["rc.accept_panel", "fpa.save_plan_version", "iv.export_compliance_pack"]
)
def test_write_tools_are_refused_over_http(client: TestClient, write_tool: str):
    response = client.post(f"/api/tools/{write_tool}", json={"approval_id": "ap_whatever"})
    assert response.status_code == 403
    assert "never over HTTP" in response.json()["detail"]


def test_an_unknown_tool_is_a_404(client: TestClient):
    assert client.post("/api/tools/rc.not_a_tool", json={}).status_code == 404


def test_bad_arguments_are_a_400_not_a_crash(client: TestClient):
    response = client.post("/api/tools/core.llm_spend", json={"month": "not-a-month", "extra": 1})
    assert response.status_code == 400


def test_a_tool_call_returns_numbers_whose_lineage_resolves(client: TestClient, db: Path):
    # core.llm_spend reads the audit schema, so it works on a warehouse with no pack data in it.
    with connect(db) as conn:
        conn.execute(
            "insert into audit.llm_calls values (?, null, 'pack_agent', 'claude-sonnet-5-5', 'medium', "
            "100, 50, 0, 0, 0.0012, 420.0, now())",
            [new_id("call")],
        )
    body = client.post("/api/tools/core.llm_spend", json={}).json()
    assert body["numbers"], "a tool result must carry its numbers"
    lineage_id = body["numbers"][0]["lineage_id"]

    record = client.get(f"/api/lineage/{lineage_id}")
    assert record.status_code == 200
    assert record.json()["value"] == body["numbers"][0]["value"]
    assert record.json()["sql_text"].strip(), "lineage must name the query behind the number"


def test_an_unknown_lineage_id_is_a_404(client: TestClient):
    assert client.get("/api/lineage/ln_nope").status_code == 404


def test_requests_that_arrive_together_all_succeed(client: TestClient):
    """A page loads several panels at once, so the routes must not race each other for the warehouse.

    FastAPI runs sync endpoints on a threadpool, and DuckDB refuses a second connection to one file
    configured differently from the first. A tool call writes its audit rows through a plain
    connection while reading through a locked-down one, so mixing tool calls with the read-only
    audit and meta routes is what makes the clash certain rather than occasional.
    """
    work = [
        lambda: client.post("/api/tools/core.llm_spend", json={}),
        lambda: client.get("/api/audit"),
        lambda: client.get("/api/meta"),
        lambda: client.get("/api/approvals"),
    ] * 3
    with ThreadPoolExecutor(max_workers=len(work)) as pool:
        codes = [f.result().status_code for f in [pool.submit(job) for job in work]]
    assert codes == [200] * len(work)


def test_every_call_is_audited(client: TestClient):
    client.post("/api/tools/core.llm_spend", json={})
    audit = client.get("/api/audit").json()
    assert any(r["agent_id"] == "api.ui" for r in audit["runs"])
    assert any(c["tool_name"] == "core.llm_spend" and c["status"] == "ok" for c in audit["tool_calls"])


def test_a_shared_deployment_lists_approvals_but_cannot_decide(client: TestClient, db: Path):
    pending = ApprovalQueue(db).propose(agent_run_id=new_id("run"), action="rc.accept_panel", payload={})
    listed = client.get("/api/approvals").json()
    assert listed["can_decide"] is False
    assert [a["approval_id"] for a in listed["pending"]] == [pending.approval_id]

    refused = client.post(
        f"/api/approvals/{pending.approval_id}/decide",
        json={"approve": True, "decided_by": "Priya Raman", "reason": "panel reviewed"},
    )
    assert refused.status_code == 403
    assert ApprovalQueue(db).get(pending.approval_id).status == "pending"


def test_a_local_run_can_decide_and_the_decision_is_recorded(deciding_client: TestClient, db: Path):
    pending = ApprovalQueue(db).propose(agent_run_id=new_id("run"), action="rc.accept_panel", payload={})
    response = deciding_client.post(
        f"/api/approvals/{pending.approval_id}/decide",
        json={"approve": True, "decided_by": "Priya Raman", "reason": "panel reviewed with procurement"},
    )
    assert response.status_code == 200
    decided = ApprovalQueue(db).get(pending.approval_id)
    assert decided.status == "approved"
    assert decided.decided_by == "Priya Raman"


def test_an_agent_cannot_decide_even_where_deciding_is_allowed(deciding_client: TestClient, db: Path):
    pending = ApprovalQueue(db).propose(agent_run_id=new_id("run"), action="rc.accept_panel", payload={})
    response = deciding_client.post(
        f"/api/approvals/{pending.approval_id}/decide",
        json={"approve": True, "decided_by": "agent:ratecards.panel", "reason": "I propose and approve"},
    )
    assert response.status_code == 409
    assert ApprovalQueue(db).get(pending.approval_id).status == "pending"


def test_meta_reports_an_empty_warehouse_without_erroring(client: TestClient):
    body = client.get("/api/meta").json()
    assert body["synthetic"] is True
    assert body["can_decide_approvals"] is False
    assert body["packs"] == {"fpa": False, "interviews": False, "ratecards": False}
    assert body["fpa"] is None


def test_the_ui_is_served(client: TestClient):
    assert "LedgerLens" in client.get("/").text
    assert client.get("/static/app.js").status_code == 200
