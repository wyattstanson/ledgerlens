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

## Settled at Phase 5 (owner decisions, 2026-10-31)

1. **Approvals are local only.** Deciding one needs a named human, and a shared URL cannot name
   anybody, so the queue is listed everywhere and decided only where `LEDGERLENS_ALLOW_APPROVALS` is
   true. `render.yaml` sets it to false. The gate sits in the route, above the queue's own rule that a
   decider must be a named human rather than an agent, so a deployment cannot decide even by mistake.
2. **The free Render plan.** No hosting cost. A free instance sleeps when idle, so the first visit
   after a pause is slow, and its disk is ephemeral: the audit log and approvals reset on each deploy.
   The synthetic warehouse is built during the build step rather than at startup, so a cold start
   serves a page instead of rebuilding one.
3. **No agent panel in the UI.** Pages call governed tools directly and render what they return, so
   the deployment makes no model calls and costs nothing to run. `ANTHROPIC_API_KEY` is absent from
   `render.yaml`; the audit page still shows the spend table, which reads zero there.

## Consequences of those three

- Every number on a page comes from a tool's `numbers[]`, so each one carries a `lineage_id`, and the
  UI turns it into a button that opens the query, the source tables and the row ids behind it.
- The API exposes read-only tools only. A write tool is refused with 403 whatever it is sent, because
  a write needs an approval record that only a human can create.
