"""The web service: tools over HTTP keep their governance, and the UI's own routes work.

Every assertion here is about a rule the API must not let slip: write tools stay unreachable, numbers
arrive with a lineage id that resolves, approvals need a named human, and a shared deployment cannot
decide at all.
"""

import html
import posixpath
import re
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from ledgerlens.api import deps
from ledgerlens.api.main import STATIC_DIR, app
from ledgerlens.api.offices import OFFICES
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


def test_the_landing_page_and_the_workspace_are_each_served(client: TestClient):
    landing = client.get("/").text
    assert "Ask about" in landing, "the landing page leads with the guide"
    workspace = client.get("/app").text
    assert 'id="view"' in workspace, "the workspace is the page with the pack tabs"
    assert landing != workspace
    for asset in ("/static/app.js", "/static/landing.js", "/static/landing.css"):
        assert client.get(asset).status_code == 200, asset


def test_a_missing_page_gets_the_styled_404_and_a_missing_api_path_gets_json(client: TestClient):
    page = client.get("/nope")
    assert page.status_code == 404
    assert "That page" in page.text, "a visitor should land somewhere that offers a way on"

    api = client.get("/api/nope")
    assert api.status_code == 404
    assert api.json()["detail"] == "not found", "a caller needs JSON, not a page"


@pytest.mark.parametrize("path", ["/static/app.js", "/", "/app", "/terms", "/privacy", "/for/fpa"])
def test_what_the_browser_caches_must_be_revalidated(client: TestClient, path: str):
    # Without this a returning visitor keeps running an old shell and old modules against a new API.
    assert client.get(path).headers["cache-control"] == "no-cache"


def test_every_page_the_tabs_offer_is_a_module_that_exists(client: TestClient):
    workspace = client.get("/app").text
    tabs = re.findall(r'data-tab="([a-z]+)"', workspace)
    assert set(tabs) == {"fpa", "interviews", "ratecards", "yours", "method", "approvals", "audit"}

    router = client.get("/static/app.js").text
    for tab in tabs:
        assert f"./pages/{tab}.js" in router, f"the {tab} tab has no module imported"
        assert client.get(f"/static/pages/{tab}.js").status_code == 200, tab


def test_every_module_a_module_imports_is_served(client: TestClient):
    """A relative import that points nowhere is a 404 the browser reports and the suite misses.

    The landing page and the workspace share the guide's intent table, so this is the seam where a
    renamed file would quietly break one of the two pages.
    """
    seen = 0
    for module in STATIC_DIR.rglob("*.js"):
        here = module.parent.relative_to(STATIC_DIR)
        for target in re.findall(r'from "(\.{1,2}/[^"]+)"', module.read_text(encoding="utf-8")):
            served = posixpath.normpath(posixpath.join(here.as_posix(), target))
            assert client.get(f"/static/{served}").status_code == 200, f"{module.name} imports {target}"
            seen += 1
    assert seen >= 2, "no relative imports found; this guard is checking nothing"


def test_the_shell_holds_every_element_the_modules_write_into(client: TestClient):
    """The modules reach for these by id, so one dropped from the page is a TypeError only a browser
    sees. Every module is scanned, not just the router, because the dock and the guide look things
    up too.
    """
    workspace = client.get("/app").text
    wanted: set[str] = set()
    for module in STATIC_DIR.rglob("*.js"):
        wanted |= set(re.findall(r'getElementById\("([\w-]+)"\)', module.read_text(encoding="utf-8")))
    assert wanted, "nothing is looked up by id any more; this guard needs rewriting"
    served = set(re.findall(r'id="([\w-]+)"', workspace))
    assert wanted <= served, f"the workspace is missing: {sorted(wanted - served)}"


def test_every_rail_link_carries_a_word_as_well_as_an_icon(client: TestClient):
    """Docked, the rail shows icons only and hides the words with CSS.

    The word has to stay in the DOM for that: swap it for a title attribute and every link in the
    collapsed rail loses its accessible name, which no screenshot would show.
    """
    workspace = client.get("/app").text
    links = re.findall(r'<a href="#/[a-z]+" data-tab="[a-z]+">(.*?)</a>', workspace)
    assert len(links) == 7, f"expected one link per page, found {len(links)}"
    for link in links:
        assert "rail-icon" in link, link
        assert re.search(r'class="rail-word">[^<]+</span>', link), f"this link has no word: {link}"


@pytest.mark.parametrize("slug", sorted(OFFICES))
def test_each_office_gets_its_own_page(client: TestClient, slug: str):
    office = OFFICES[slug]
    page = client.get(f"/for/{slug}")
    assert page.status_code == 200
    # The template escapes, so FP&A reaches the page as FP&amp;A; compare like for like.
    assert html.escape(office.title) in page.text, "the page is not the one the office asked for"
    assert office.workspace in page.text, "an office page with no way into its pack is a dead end"
    assert html.escape(office.problems[0].quote) in page.text


def test_an_office_that_does_not_exist_gets_the_styled_404(client: TestClient):
    page = client.get("/for/accounting")
    assert page.status_code == 404
    assert "That page" in page.text


@pytest.mark.parametrize("slug", sorted(OFFICES))
def test_an_office_page_carries_everything_landing_js_drives(client: TestClient, slug: str):
    """The office pages share landing.js, which reaches for these without checking.

    One missing node is a TypeError that stops the script before the reveals run, leaving the page
    blank in exactly the way a visitor would blame on the product.
    """
    page = client.get(f"/for/{slug}").text
    for hook in ("data-nav", "data-burger", "data-menu", "data-ask", "data-ask-input", "data-year"):
        assert hook in page, hook
    assert 'class="hero"' in page, "landing.js animates the hero by class"


def test_the_landing_page_links_only_to_offices_that_exist(client: TestClient):
    linked = set(re.findall(r'href="/for/([a-z]+)"', client.get("/").text))
    assert linked == set(OFFICES), f"the landing page and the office table disagree: {linked}"


@pytest.mark.parametrize("path", ["/terms", "/privacy"])
def test_the_legal_pages_are_served(client: TestClient, path: str):
    response = client.get(path)
    assert response.status_code == 200
    assert "synthetic" in response.text or "collects nothing" in response.text
