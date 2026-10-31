"""The LedgerLens web service: governed tools over HTTP, plus the static analyst UI (ADR 0002).

Run locally with ``make api``. On Render, ``uvicorn ledgerlens.api.main:app`` serves the same thing;
the synthetic warehouse is built during the build step, and the deployment shows approvals read-only.
"""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Any

import duckdb
from fastapi import Depends, FastAPI, Request, Response
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from ledgerlens import __version__
from ledgerlens.api import routes_audit, routes_compute, routes_tools
from ledgerlens.api.deps import settings, warehouse
from ledgerlens.core.db import NO_EXTERNAL_ACCESS, connect, ensure_audit_schema, fetch_dicts
from ledgerlens.settings import Settings

STATIC_DIR = Path(__file__).parent / "static"

DESCRIPTION = (
    "Governed analytics for FP&A, interview quality and rate cards. Every number comes from a tested "
    "function and carries a lineage id you can look up. All data is synthetic."
)


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncGenerator[None]:
    """The audit tables must exist before the first call writes its run row."""
    ensure_audit_schema(settings().db)
    yield


app = FastAPI(title="LedgerLens", version=__version__, description=DESCRIPTION, lifespan=lifespan)
app.include_router(routes_tools.router)
app.include_router(routes_audit.router)
app.include_router(routes_compute.router)

Config = Annotated[Settings, Depends(settings)]


@app.get("/healthz", tags=["meta"])
def healthz() -> dict[str, str]:
    """Liveness for Render. Deliberately does not touch the warehouse."""
    return {"status": "ok", "version": __version__}


@app.get("/api/meta", tags=["meta"])
def meta(config: Config) -> dict[str, Any]:
    """What this deployment is serving: packs with data, the FP&A world, and whether it can decide."""
    return {
        "version": __version__,
        "synthetic": True,
        "can_decide_approvals": config.allow_approvals,
        "packs": _packs_present(config.db),
        "fpa": _fpa_meta(config.db),
    }


def _schemas(db_path: Path) -> set[str]:
    with warehouse(), connect(db_path, read_only=True, config=NO_EXTERNAL_ACCESS) as conn:
        rows = conn.execute("select schema_name from information_schema.schemata").fetchall()
    return {r[0] for r in rows}


def _packs_present(db_path: Path) -> dict[str, bool]:
    """Which packs have run on the loaded warehouse, so the UI can say so instead of erroring."""
    try:
        schemas = _schemas(db_path)
    except duckdb.Error:
        # No warehouse yet, or one this build cannot open: either way no pack has data to show.
        return {"fpa": False, "interviews": False, "ratecards": False}
    return {name: name in schemas for name in ("fpa", "interviews", "ratecards")}


def _fpa_meta(db_path: Path) -> dict[str, Any] | None:
    """The FP&A run metadata (which world, which periods), or None when that pack has not run."""
    try:
        with warehouse(), connect(db_path, read_only=True, config=NO_EXTERNAL_ACCESS) as conn:
            rows = fetch_dicts(
                conn,
                "select variant, entity, book, as_of, open_period, latest_closed from fpa.run_meta",
            )
    except duckdb.Error:
        return None  # fpa.run_meta is written by the pipeline, so its absence means it has not run
    return rows[0] if rows else None


def page(name: str) -> FileResponse:
    """An HTML page the browser must revalidate.

    Same reasoning as RevalidatedStatic below, and the same bug if it is left off: the page names
    the modules, so a cached shell keeps loading the modules it knew about and the new ones are
    never asked for. A conditional request costs one 304.
    """
    return FileResponse(STATIC_DIR / name, headers={"cache-control": "no-cache"})


@app.get("/", include_in_schema=False)
def landing() -> FileResponse:
    """The public page. The workspace lives at /app so the two can look nothing alike."""
    return page("landing.html")


@app.get("/app", include_in_schema=False)
def workspace() -> FileResponse:
    return page("index.html")


@app.get("/terms", include_in_schema=False)
def terms() -> FileResponse:
    return page("terms.html")


@app.get("/privacy", include_in_schema=False)
def privacy() -> FileResponse:
    return page("privacy.html")


@app.exception_handler(404)
async def not_found(request: Request, _: Exception) -> Response:
    """A page gets the styled 404; an API path keeps its JSON so callers can read it."""
    if request.url.path.startswith("/api/"):
        return JSONResponse({"detail": "not found"}, status_code=404)
    return FileResponse(STATIC_DIR / "404.html", status_code=404, headers={"cache-control": "no-cache"})


class RevalidatedStatic(StaticFiles):
    """Static files a browser must revalidate before reusing.

    The pages are ES modules with stable names. With no Cache-Control a browser is free to invent a
    freshness lifetime, so after a deploy a returning visitor keeps running yesterday's app.js against
    today's API. `no-cache` still allows the cache, it just forces the conditional request, and the
    answer is almost always a 304.
    """

    def file_response(self, *args: Any, **kwargs: Any) -> Response:
        response = super().file_response(*args, **kwargs)
        response.headers["cache-control"] = "no-cache"
        return response


app.mount("/static", RevalidatedStatic(directory=STATIC_DIR), name="static")
