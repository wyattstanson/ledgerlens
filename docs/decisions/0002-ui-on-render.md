# ADR 0002: Replace Streamlit with a FastAPI-served web UI, hosted on Render

- Status: accepted (owner decision, 2026-10-19)
- Supersedes: the "UI: Streamlit with Plotly" row of BUILD_PROMPT section 3, and answers section 17's
  "deployment target beyond local".
- Affects: Phase 5 (API and UI), Phase 6 (demo).

## Decision

- **No Streamlit.** FastAPI serves both the JSON API and a static front end (HTML, CSS and plain
  JavaScript modules, Plotly.js from a pinned CDN version). There is no Node build step: Windows App
  Control blocks the native esbuild and rollup binaries on the owner's machine.
- **One Render web service** runs `uvicorn ledgerlens.api.main:app`. The UI follows
  `docs/ui_guardrails.md`.
- **Data on Render is synthetic and rebuilt at startup** (`ledgerlens generate` plus the pack pipelines,
  seeded), because the free instance's filesystem is ephemeral. The audit log and approvals therefore
  reset on each deploy or restart, which is acceptable for a demo. Revisit with a persistent disk or a
  managed database if real use appears.
- **Secrets** come only from Render environment variables (`ANTHROPIC_API_KEY` is optional; without it
  the app runs fully offline and agents are disabled).

## Consequences

- Phase 5 removes `streamlit`, `app/` and the Streamlit tests and Makefile target, and adds a
  `render.yaml` blueprint.
- Phase 5 adds the API routes the front end needs; every number the UI shows comes from a governed tool
  with lineage, exactly as for agents and MCP.

## Open for Phase 5's clarifying questions

- Who may approve on a public URL? Approvals need an authenticated human. Options include a proven
  auth library or provider, or keeping approvals local-only and showing them read-only on Render.
- Render plan and region, and whether a monthly hosting cost is acceptable.
