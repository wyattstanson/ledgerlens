# Build prompt: LedgerLens, governed analytics agents for FP&A, interviews and rate cards

Versions and APIs in this prompt were checked against PyPI, npm, GitHub and the official Claude docs on 18 October 2026, and the core stack was installed and smoke-tested together on that date (details in section 6). "LedgerLens" is a working name; rename the package if you like.

---

## 0. How to use this prompt (for the human)

1. Install Claude Code (section 15.1) and the recommended plugins (section 15.2).
2. Create an empty folder, run `git init`, and save this file as `docs/BUILD_PROMPT.md`.
3. Export your API key in your shell, never in a file that gets committed: `export ANTHROPIC_API_KEY=...`
4. Start `claude` in the folder and say: **"Read docs/BUILD_PROMPT.md fully. Then execute Phase 0 only and stop for my review."**
5. After each phase passes its exit criteria, say "Execute Phase N".

Working phase by phase keeps each change reviewable. Don't ask the agent to build everything in one go.

---

## 1. Your role and mission (for the coding agent)

You are a senior Python data and AI engineer. Build **LedgerLens**: one shared, governed agent core and three domain packs.

1. **Payments-aware FP&A.** Forecasting and variance analysis for a P&C insurer, driven by how premiums are collected and how claims are paid out, not only by ledger totals.
2. **Interview Quality and Fairness Observatory.** Measures interviews, interviewers and loops, never candidates. Produces the bias audits that hiring laws require.
3. **Rate Card Consolidation Engine.** Maps billed job titles to standard roles, compares vendors on total cost, picks the cheapest vendor panel that still covers every role, and traces every dollar of savings back to invoice lines.

**The founding rule:** the language model plans and explains, but deterministic code produces every number. Every number shown to a user must carry lineage to the query and rows that produced it.

Work in phases (section 13). At the end of each phase, run the full test suite, show the output, report what is verified versus assumed, and stop.

---

## 2. Why the product looks like this (research context)

This came out of a teardown of Deloitte (Zora AI), Fractal (Cogentiq), Tredence (Milky Way), Tiger Analytics, Genpact and Accenture. It also covered the operators that actually run these domains: Anaplan, Workday, Oracle, OneStream, Fiserv, Stripe, Adyen, Intuit, Guidewire, Duck Creek, One Inc, Paymentus, HireVue, Eightfold, LinkedIn, Coupa, SAP Ariba, Keelvar, Pactum and Beeline. Six findings shape the design:

- **Agents alone are becoming free.** Oracle bundles its finance agents at no extra cost, and Workday pays for them from pooled credits. Value sits in the data model, the controls and measurable outcomes.
- **Leaders run one governance and evaluation layer for every agent.** Intuit's GenOS has a shared Evaluation Service, and Fiserv's agentOS uses identity-bound, policy-checked agent actions. We copy both.
- **Card networks register agents.** Visa and Mastercard require agents to be registered, scoped and logged ("Know Your Agent"). Our agent registry follows the same idea.
- **Interview analytics must not score candidates.** Mobley v. Workday treats the vendor as the employer's agent, and the Eightfold case argues AI candidate scores are consumer reports. We measure the process instead.
- **AI is rewriting rate cards.** IT buyers are reopening contracts for productivity commitments, so compare vendors on total cost, not hourly rate.
- **Most competitors publish self-reported results.** Reproducible evals on realistic data are a differentiator, so we build them in from day one.

---

## 3. Decisions already made (do not relitigate without asking)

| Area | Decision | Reason |
|---|---|---|
| Language | Python 3.13 (`requires-python = ">=3.13,<3.14"`) | dagster-dbt 0.29.26 requires Python below 3.14. Python 3.14.8 and 3.15.0 exist but would break the orchestrator integration |
| Packaging | uv 0.13.x, single `pyproject.toml`, committed `uv.lock` | Fast, reproducible, one tool for venv, deps, scripts and Python install |
| Repo shape | One monorepo, one Python package, three packs as subpackages, each demoable on its own (`make demo-ratecards` and so on) | Serves both "three portfolio projects" and "one platform" |
| Warehouse | DuckDB file `data/warehouse.duckdb` | Zero infrastructure, fast, works with dbt, Polars and MCP |
| Transformations | dbt-core with dbt-duckdb | Industry-standard SQL modelling with tests and lineage |
| Orchestration | Dagster with dagster-dbt (dbt models become assets) | Asset lineage UI, asset checks, schedules |
| Dataframes | Polars for Python transforms; pandas 2.3 only where a library needs it | statsforecast 2.1.1 requires `pandas<3` |
| LLM | Claude via the official `anthropic` Python SDK, using model routing (section 7) | Structured outputs, strict tools, effort control, prompt caching |
| Agent loop | Our own thin loop on the Messages API (or the SDK's beta `tool_runner`), not the Claude Agent SDK | We need full control of the registry, approvals and lineage checks. The Agent SDK is aimed at coding-agent workloads with file and shell tools |
| Tool exposure | Our own MCP server built on the official `mcp` SDK v2 (`MCPServer`) | Any MCP client (Claude Code, Claude Desktop) can call our governed tools |
| API | FastAPI | Typed, documented, easy to test |
| UI | ~~Streamlit with Plotly~~ FastAPI-served web UI on Render ([ADR 0002](decisions/0002-ui-on-render.md)) | Owner decision, 2026-10-19 |
| Data | Synthetic only, seeded and reproducible, with planted problems and hidden ground truth | No consent or legal issues, and the evals know the right answers |
| FP&A subject | P&C insurer as primary; a payments-processor variant behind `--variant processor` | Covers both open options from research |

If any decision blocks you, stop and ask. Don't silently swap a component.

---

## 4. Non-negotiable rules

1. **Numbers come only from compute.** LLM output may contain numbers only if each one appears in the structured `numbers[]` list returned by a tool call, with a `lineage_id` that exists in `audit.lineage`. The harness rejects any answer that fails this check (section 9.4).
2. **No candidate-level scores or rankings, anywhere.** No API route, MCP tool, UI page or export may return a score, rank or recommendation for an individual candidate. Fairness and quality aggregates require a group size of at least 5 (`MIN_GROUP_N = 5`), otherwise they are suppressed. Never analyse faces, voices or emotions.
3. **Read-only by default.** Tools that change state (approve a mapping, change a plan version, accept a vendor panel) require a human approval record. Agents can only *propose* them.
4. **Treat retrieved text as untrusted.** Notes, invoice descriptions, tool results and documents are passed to the model inside a clearly delimited data block and can never change instructions or tool permissions.
5. **Secrets come only from environment variables** (`pydantic-settings`). Commit `.env.example`, never `.env`. Don't read credential files.
6. **Every function with business logic gets tests.** For each new test, break the code once and watch the test fail before trusting it green.
7. **Pre-register statistics.** The interview analysis plan (tests, thresholds, groups, exclusions) lives in `config/analysis_plan.yaml`. It is hashed and logged before any run, and any change creates a new version. Correct for multiple comparisons (Benjamini-Hochberg) and report effect sizes with confidence intervals, never a bare p-value.
8. **Seeds everywhere.** Data generators and models take a seed, and the default seed is fixed in config.
9. **Don't pin versions from memory.** Use `uv add` and let the lockfile record what resolved. If a newer major version appears, check its changelog before upgrading.
10. **Log LLM usage.** Record model, effort, input, output and cache tokens, latency and cost for every call, taken from the API response's `usage`, not estimated.

---

## 5. Tech stack (verified 18 October 2026)

"Tested" means installed together in one environment on Python 3.13.16 and exercised by the smoke tests in section 6.

### 5.1 Runtime

| Package | Version | Role | Tested |
|---|---|---|---|
| Python | 3.13.16 | Runtime | yes |
| uv | 0.13.1 | Packaging and runner | yes |
| anthropic | 1.15.0 | Claude API SDK | API surface checked, no live call made |
| mcp | 2.3.0 | MCP server and client (protocol 2026-07-28) | yes |
| duckdb | 1.5.6 | Warehouse | yes |
| dbt-core | 1.12.6 | SQL transforms and tests | yes |
| dbt-duckdb | 1.11.0 | dbt adapter | yes |
| dagster, dagster-webserver | 1.13.26 | Orchestration and UI | yes |
| dagster-dbt, dagster-duckdb | 0.29.26 | Integrations | yes |
| polars | 2.0.0 | Dataframes | yes |
| pyarrow | 26.0.0 | Columnar interchange | yes |
| pandas | 2.3.3 (pin `>=2.3,<3`) | Required by statsforecast | yes |
| numpy | 2.5.4 | Numerics | yes |
| ortools | 9.15.6755 | MILP via MathOpt with the GSCIP solver | yes |
| statsforecast | 2.1.1 | AutoETS, AutoARIMA, SeasonalNaive, cross-validation | yes |
| statsmodels | 0.15.0 | Fleiss' kappa, Benjamini-Hochberg | yes |
| scipy | 1.18.1 | Fisher's exact test, bootstrap confidence intervals | yes |
| pandera[polars] | 0.34.1 | Dataframe schemas and checks | yes |
| rapidfuzz | 3.14.6 | Fuzzy title matching | installed |
| faker | 40.43.0 | Synthetic names and text | installed |
| fastapi | 0.146.0 | API | yes |
| uvicorn | 0.54.0 | ASGI server | installed |
| pydantic | 2.14.1 | Models and validation | yes |
| pydantic-settings | 2.16.0 | Configuration from environment | installed |
| streamlit | 1.66.0 | Analyst UI | yes (health endpoint) |
| plotly | 7.1.0 | Charts | yes |
| opentelemetry-sdk | 1.45.1 | Tracing | installed |

### 5.2 Development

| Package | Version | Role | Tested |
|---|---|---|---|
| pytest | 9.1.1 | Tests | yes |
| pytest-cov | 7.1.0 | Coverage | installed |
| hypothesis | 6.170.1 | Property-based tests | installed |
| ruff | 0.17.1 | Lint and format | yes |
| pyright | 1.1.414 | Type checking (standard mode) | yes |
| pre-commit | 4.7.0 | Git hooks | installed |
| inspect-ai | 0.3.281 | LLM eval framework for agent evals | import checked |

### 5.3 Optional extras (latest on PyPI, not smoke-tested)

| Package | Version | Use |
|---|---|---|
| fastembed | 0.9.0 | Light ONNX embeddings to shortlist title matches (avoids installing PyTorch) |
| langfuse | 5.0.2 | Hosted LLM tracing UI (our DuckDB audit log stays the source of truth) |
| mcp-server-motherduck | 1.1.0 | Ad hoc read-only SQL over the local DuckDB file from Claude Code (`--help` checked) |
| krippendorff | 0.9.0 | Krippendorff's alpha, if panels have missing ratings |
| marimo | 0.25.1 | Reactive notebooks for exploratory analysis |

### 5.4 Deliberately not used

- **pandas 3.x**, until statsforecast lifts its `pandas<3` pin.
- **Python 3.14 or later**, until dagster-dbt supports it.
- **sentence-transformers and PyTorch** in the core install, because they're too heavy. Use rapidfuzz, with fastembed as an option.
- **LangChain or LlamaIndex.** The agent loop is small, and owning it keeps lineage and approvals explicit.
- **Face, voice or emotion analysis libraries**, which are banned by rule 2.

---

## 6. Gotchas found during the smoke test (read before coding)

1. **MCP Python SDK v2 renamed FastMCP.** `from mcp.server.fastmcp import FastMCP` now fails. Use:
   ```python
   from mcp.server.mcpserver import MCPServer
   server = MCPServer(name="ledgerlens", instructions="...")   # keyword args only
   @server.tool()
   def role_rates(role_std: str) -> list[dict]: ...
   server.run(transport="stdio")   # host, port etc. go to run(), not the constructor
   ```
   Test servers in-process with `from mcp import Client`: `async with Client(server) as c: await c.call_tool(...)`. Results expose `.is_error` and `.structured_content` (snake_case). Raise `ToolError` for failures the model should see. `create_connected_server_and_client_session` was removed. `MCP_*` environment variables are no longer read automatically.
2. **Anthropic SDK structured outputs.** Use `output_config={"format": {...}}` on `messages.create`, or the `client.messages.parse(..., output_format=PydanticModel)` helper, which returns `.parsed_output`. The old `output_format` request parameter is deprecated and raises `TypeError` on `beta.messages.create`. The API does not enforce `minimum`, `maximum`, `minLength` and similar constraints (the `parse()` helper validates them locally). Objects need `additionalProperties: false`. Limits are 20 strict tools and 24 optional parameters per request.
3. **Thinking and effort.** Current models use adaptive thinking, controlled by `output_config={"effort": "low"|"medium"|"high"|"xhigh"|"max"}`. Don't send `thinking={"type":"enabled","budget_tokens":...}`, because current models reject it. Claude Opus 5.5 cannot disable thinking. Thinking tokens count against `max_tokens`, so leave room.
4. **Prompt caching** needs a prefix of at least 512 tokens on Opus, Sonnet and Haiku 5.5. Put static content (system prompt, tool definitions, schemas) first and per-request content last. Don't interpolate timestamps into the system prompt.
5. **dbt has to be parsed from inside its own folder.** Run `cd transform && uv run dbt parse --profiles-dir .`. Parsing from the repo root bakes `transform/seeds/...` paths into the manifest, and seeds then fail under Dagster with "No files found that match the pattern".
6. **Dagster's `DbtProject.prepare_if_dev()` only runs under `dagster dev`.** Tests, CI and `dg.materialize()` need the manifest generated by the Makefile target (step 5) first.
7. **The smoke test caught a wrong expectation in its own optimizer test.** With a critical-role constraint, the optimizer kept a different vendor pair than the hand-written answer assumed. Derive expected optimizer results from a brute-force enumeration in tests (feasible for 5 vendors or fewer), not from mental arithmetic.
8. **scipy and statsforecast return loosely typed results.** Use `_, p = fisher_exact(table)` and `isinstance(cv, pd.DataFrame)` guards. pyright flags `.pvalue` access otherwise.
9. **Streamlit phones home for usage stats.** Set `browser.gatherUsageStats = false` in `.streamlit/config.toml`.
10. **A transitive dependency emits a `sqlglotrs` deprecation warning,** and dagster-dbt emits a `context.op` deprecation warning internally. Both are harmless; filter them in the pytest config rather than chasing them.

---

## 7. Model routing

The model IDs below are current, dateless pinned snapshots (retirement no sooner than September 2027). Keep them in `config/models.yaml` so you can swap them without changing code.

| Task | Model | Effort | Notes |
|---|---|---|---|
| Title-to-role pick from a shortlist | `claude-haiku-5-5` | low | Structured output; batch it with the Message Batches API (50% cheaper) |
| Evidence tagging of interview notes | `claude-haiku-5-5` | low | Structured output; Batches API |
| Pack agents: plan tool calls, narrate results | `claude-sonnet-5-5` | medium | Strict tools plus a structured final answer |
| Negotiation briefs, variance narratives | `claude-sonnet-5-5` | medium | Numbers come only from tool outputs |
| LLM-as-judge in offline evals | `claude-opus-5-5` | medium | Offline only, never on the request path |

List prices per million tokens (input / output): Opus 5.5 $4 / $20, Sonnet 5.5 $2 / $10, Haiku 5.5 from $0.10 / $0.50. Cache reads cost 5% of base input on Opus and Sonnet 5.5, and 10% on Haiku. Write a `cost_usd` per call into `audit.llm_calls` and show the running total in the UI footer.

---

## 8. Repository layout

```
ledgerlens/
  pyproject.toml          uv.lock          Makefile          README.md
  CLAUDE.md               .mcp.json        .env.example      .pre-commit-config.yaml
  .streamlit/config.toml  .github/workflows/ci.yml           .github/workflows/llm-evals.yml
  config/
    settings.toml           # paths, seeds, thresholds (non-secret)
    models.yaml             # model routing (section 7)
    agents.yaml             # agent registry (section 9.1)
    analysis_plan.yaml      # pre-registered interview statistics
    ratecards.yaml          # optimizer constraints, scorecard weights
    fpa.yaml                # synthetic-world parameters for FP&A
  data/                     # gitignored: warehouse.duckdb, raw parquet
  transform/                # dbt project
    dbt_project.yml  profiles.yml
    models/
      core/        audit and lineage views
      fpa/         staging/ intermediate/ marts/
      interviews/  staging/ intermediate/ marts/
      ratecards/   staging/ intermediate/ marts/
    tests/         # singular dbt tests (reconciliations)
  src/ledgerlens/
    __init__.py
    settings.py             # pydantic-settings
    core/
      llm.py                # client wrapper: routing, caching, usage logging, retries
      registry.py           # agent and tool registry, mandates
      tools.py              # tool definitions (strict JSON schemas) -> compute functions
      agent_loop.py         # plan/act loop, lineage check, approval proposals
      lineage.py            # lineage ids, audit writers
      approvals.py          # approval queue
      guards.py             # untrusted-content fencing, MIN_GROUP_N suppression
      evals/                # golden sets, scorers, runner
    fpa/        generate.py drivers.py bridge.py forecast.py scenario.py payments_adapter.py agents.py
    interviews/ generate.py metrics.py fairness.py tagger.py compliance.py agents.py
    ratecards/  generate.py mapping.py normalize.py benchmark.py scorecard.py optimizer.py bridge.py drift.py tail.py agents.py
    orchestration/defs.py   # Dagster Definitions: dbt assets + python assets + asset checks
    mcp/server.py           # MCPServer exposing read-only governed tools
    api/main.py             # FastAPI
  app/                      # Streamlit multipage app
    Home.py
    pages/1_FPA.py  pages/2_Interviews.py  pages/3_Rate_cards.py  pages/4_Approvals.py  pages/5_Audit_and_cost.py
  tests/
    unit/  property/  integration/  evals/
  docs/
    BUILD_PROMPT.md  architecture.md  data_dictionary.md  demo_script.md  decisions/   # ADRs
```

---

## 9. Shared core specification

### 9.1 Agent registry (`config/agents.yaml`)

```yaml
agents:
  - id: fpa.variance
    owner: fpa
    model_task: pack_agent          # resolved via models.yaml
    mandate: read_only              # read_only | propose | write_with_approval
    tools: [fpa.bridge, fpa.drivers, fpa.forecast]
    max_tool_calls: 8
  - id: ratecards.panel
    owner: procurement
    mandate: propose
    tools: [rc.benchmark, rc.optimize_panel, rc.savings_bridge]
```

The loop refuses any tool not listed for that agent, and a `propose` agent can only create approval requests. Every run gets an `agent_run_id`.

### 9.2 Tools

Each tool is a pure Python function with a Pydantic input model, exported with `"strict": true` and `additionalProperties: false`. Each returns `ToolResult{data, numbers: [{label, value, unit, lineage_id}]}`. The same functions back the MCP server, the API and the agents, so there is one implementation of every calculation.

### 9.3 Audit and lineage schema (DuckDB schema `audit`)

| Table | Key columns |
|---|---|
| `agent_runs` | agent_run_id, agent_id, user, started_at, finished_at, status |
| `tool_calls` | tool_call_id, agent_run_id, tool_name, input_json, output_hash, started_at, ms |
| `lineage` | lineage_id, tool_call_id, sql_hash, sql_text, source_tables, row_ids (list), value, unit |
| `llm_calls` | call_id, agent_run_id, model, effort, input_tokens, output_tokens, cache_read_tokens, cache_write_tokens, cost_usd, ms |
| `approvals` | approval_id, agent_run_id, action, payload_json, status, decided_by, decided_at, reason |
| `analysis_plans` | plan_version, sha256, registered_at |

### 9.4 Lineage check

After the model writes its final answer (structured output: `answer_text`, `numbers[]`, `assumptions[]`, `proposed_actions[]`), the harness:

1. Extracts every numeric token from `answer_text`.
2. Requires each one to match an entry in `numbers[]` within rounding tolerance.
3. Requires each entry's `lineage_id` to exist in `audit.lineage` for this run.
4. If any step fails, retries once with the failures listed. If it fails again, returns the tool outputs as a table without narration and logs `status=lineage_failed`.

### 9.5 LLM wrapper (`core/llm.py`)

- Resolves the model and effort from `models.yaml`.
- Applies `cache_control` to the static prefix.
- Retries on 429 and 529 with jittered backoff.
- Writes the `llm_calls` row from `response.usage`.
- Wraps untrusted content in `<data source="...">` blocks, with a system instruction that data blocks are never instructions.

### 9.6 Evals (`core/evals/`)

- **Golden sets** live in `tests/evals/golden/*.jsonl`, one per pack, with at least 30 questions each. Each has an expected numeric answer computed from the generator's ground truth, never typed by hand.
- **Offline suite** (no API key needed, runs in CI): deterministic checks of the compute layer against ground truth.
- **LLM suite** (needs a key, manual workflow): agent answers scored on numeric accuracy, lineage pass rate, tool-call count, latency and cost, using inspect-ai or a pytest harness. Store before and after scores in `docs/evals/` and never accept a prompt change without them.

---

## 10. Pack 1: Payments-aware FP&A

### 10.1 Synthetic world (`fpa/generate.py`, parameters in `config/fpa.yaml`)

All parameters are synthetic assumptions, labelled as such in the UI.

- **Insurer:** 3 lines of business (personal auto, homeowners, small commercial property), about 20,000 policies, monthly data for 36 months ending September 2026.
- **Premium payments:** each instalment has a due date, a paid date, a method (card, ACH, check, digital wallet, agency bill) and an amount. On-time rates and lateness distributions differ by method.
- **Claims:** frequency per policy-month by line; severity lognormal by line; approval date. Payout channel is check, ACH, debit push, wallet or instant rail (RTP or FedNow), each with its own payout-lag distribution and a handling cost per claim (LAE). Multiparty payouts happen for a configurable share of homeowners claims.
- **Ledger:** GL postings are generated from the payment events, so the ledger reconciles exactly to events each month. Add a dbt singular test that proves it.
- **Plan:** a budget version built from the same drivers with deliberate misses, so variances exist.
- **Processor variant (`--variant processor`):** merchants, payment volume, authorization rate, chargebacks, interchange and share of agent-initiated transactions. The same agents run on different drivers.

### 10.2 Payments adapter (`fpa/payments_adapter.py`)

This is a read-only interface shaped like a carrier payments network feed (premium received, claim paid, payout returned, channel and timestamps). Expose it as tools (`payments.events`, `payments.summary`), so a real feed can be swapped in later through MCP. This copies the pattern One Inc uses with its MCP server for carriers.

### 10.3 Driver tree (`fpa/drivers.py`)

- `collected_premium[m] = sum over methods of due_premium x method_mix x on_time_rate + late_collections_from_prior_months`
- `claims_paid[m] = sum over channels of approved_claims x severity x channel_share, released by channel payout-lag distribution`
- `lae[m] = sum over channels of claims_count_channel x handling_cost_channel`
- `net_operating_cash[m] = collected_premium - claims_paid - lae - opex`

Every driver value carries lineage to its dbt mart rows.

### 10.4 Variance bridge (`fpa/bridge.py`)

Decompose actual minus plan into **volume, rate, mix and timing**, plus an explicit residual. For segments `s` with quantity Q and price or rate P (actual a, plan p):

- volume = (sum Q_a - sum Q_p) x P_p_avg
- mix = sum over s of (share_a,s - share_p,s) x sum Q_a x P_p,s
- rate = sum over s of (P_a,s - P_p,s) x Q_a,s
- timing = cash that moved between periods because payout or collection lags differed from plan

Property test: volume + mix + rate + timing + residual equals the total variance exactly (to the cent), for any generated data. Render it as a waterfall chart.

### 10.5 Forecasting (`fpa/forecast.py`)

- Forecast drivers, not totals, then recompute totals through the driver tree.
- Candidates per driver: AutoETS, AutoARIMA and SeasonalNaive(12) from statsforecast. Use rolling-origin cross-validation (`n_windows=3`, horizons 3 and 6 months) and score with MASE and MAPE.
- Pick the best model per driver. If none beats SeasonalNaive, use SeasonalNaive and say so.
- Produce a 13-week cash forecast from daily payment events aggregated weekly, with prediction intervals.
- **Run model selection through the `ml-pipeline` skill** (`ml_pipeline/PIPELINE.md`, marimo notebooks,
  figures under `ml_pipeline/figures/`). It is the only place in LedgerLens that fits models, and its
  hook blocks fitting until the owner approves each gate. Mapping:
  - Steps 1-3 (Gate A): profile and plot each monthly driver series. Contract: one target per driver,
    horizons 3 and 6 months, MASE primary and MAPE secondary, only data up to the forecast origin.
  - Steps 4-8 (Gate B): a temporal split with `guard.split(time_col=...)`, holding out the last 6 months as
    the frozen test set; rolling-origin CV (`n_windows=3`) runs inside the training months only.
  - Steps 9-12 (Gate C): SeasonalNaive(12) is the baseline; AutoETS and AutoARIMA are the candidates.
  - Steps 13-16 (Gate D): error analysis by driver, one `guard.final_test()` per driver, forecasts
    materialized as Dagster assets, and drift checks as asset checks.
- Rate cards (Phase 2) and interviews (Phase 3) train no models. Their thresholds (title-mapping
  confidence 0.85, tagger kappa 0.7, the analysis-plan flags) are fixed in config before any evaluation and
  are never tuned against the hidden ground truth.

### 10.6 Scenarios (`fpa/scenario.py`)

Scenarios change driver inputs only, never outputs. The built-in example is "move 30% of claims from check to instant payment", which shows the effect on weekly cash timing, LAE and claimant days-to-pay. The output is a side-by-side table that lists the changed drivers.

### 10.7 Agents

Each agent has its tools from 9.2 and a mandate from 9.1:
- **Driver agent** (read_only): explains the current drivers.
- **Variance agent** (read_only): narrates the bridge.
- **Forecast agent** (read_only): reports the chosen models and errors.
- **Scenario agent** (propose): can propose saving a scenario as a plan version, which needs approval.

### 10.7a NetSuite alignment (ADR 0001)

Follow [docs/decisions/0001-netsuite-alignment.md](decisions/0001-netsuite-alignment.md): NetSuite-shaped GL tables and
accounting periods, a read-only `fpa/ledger_adapter.py`, materiality and aging defaults in `config/fpa.yaml`,
preliminary labels for the open period, and the headline/scope/drivers/risks/action answer shape.

### 10.8 Exit criteria

- [ ] The ledger reconciles to payment events every month, and debits equal credits per transaction (dbt tests pass).
- [ ] Variance flags respect the ADR 0001 materiality thresholds at their boundaries (property test).
- [ ] The bridge sums exactly to the variance (hypothesis property test).
- [ ] The chosen forecast beats SeasonalNaive on held-out months for most drivers, and every driver's result is reported.
- [ ] 100% of agent answers pass the lineage check on the golden set.
- [ ] The scenario demo runs end to end in the UI.

---

## 11. Pack 2: Interview Quality and Fairness Observatory

### 11.1 Synthetic world (`interviews/generate.py`)

- 50 interviewers, 8 roles, 2,000 candidates. Stages: screen, technical, onsite panel (3 to 4 interviewers), hiring manager.
- 4 to 6 competencies per role, each with behaviourally anchored descriptions for 1 and 4. Ratings use a 4-point scale with no neutral middle.
- Notes are generated from templates, with and without cited evidence, so evidence labels are known.
- Group attributes (sex, race or ethnicity in EEO-1 style categories, plus Unknown) live in a separate table, `restricted.candidate_groups`. Only the audit job can read it.
- Hires get a performance rating 9 to 12 months later.
- **Planted problems (the hidden ground truth):**
  - One lenient interviewer, with a mean shift of +0.8 on one role.
  - One panel whose ratings are random (kappa near 0).
  - One stage on one role with an impact ratio near 0.65 for one group.
  - One role where interview scores do not predict performance.

### 11.2 Metrics (`interviews/metrics.py`, `interviews/fairness.py`)

Thresholds live in `analysis_plan.yaml`.

| Metric | Definition | Default flag |
|---|---|---|
| Calibration drift | z = (interviewer mean - peer mean) / peer SD, per role and competency, n of at least 10 | abs(z) > 1.5 |
| Panel agreement | Fleiss' kappa per panel and competency | kappa < 0.4 |
| Rubric coverage | Share of assigned competencies with at least one question and one evidence-backed rating | < 80% |
| Evidence quality | Share of ratings whose note cites a quote or observed behaviour | < 70% |
| Loop coverage | Share of loops covering every competency the role requires (the Bar Raiser idea) | < 90% |
| Impact ratio | Group selection rate / highest group's selection rate, per stage. For score stages, use the scoring rate (share above median) | < 0.80 |
| Significance | Fisher's exact test per comparison, Benjamini-Hochberg across every test in the run | adjusted p < 0.05 |
| Effect size | Cohen's h for rates, Cohen's d for scores, each with a 95% bootstrap CI | reported, not a flag |
| Predictive validity | Spearman correlation of interview score and later performance, 95% bootstrap CI, n of at least 30 per role | CI includes 0 |

Apply small-group rules everywhere: suppress any aggregate with n below 5, and exclude groups under 2% of the data from impact ratios, with a note (as NYC Local Law 144 allows).

### 11.3 Evidence tagger (`interviews/tagger.py`)

- Model: Haiku 5.5, effort low, structured output `{has_evidence: bool, evidence_span: str | null, competency_id: enum}`, run through the Batches API.
- Validation: compare against 200 labels (generator ground truth plus a human spot-check of 50). Target Cohen's kappa of at least 0.7, and report it next to HireVue's published r = 0.69 agreement with expert raters as a reference point, noting that the two statistics differ.

### 11.4 Compliance pack (`interviews/compliance.py`)

This produces documents, not legal advice. Each output carries a "not legal advice" banner.

- **NYC Local Law 144-style audit table:** selection or scoring rate and impact ratio by sex, by race or ethnicity, intersectional and Unknown, with the excluded small groups listed.
- **Colorado SB 26-189** (from 1 January 2027): a pre-use notice template, an adverse-decision explanation template (to be sent within 30 days) and the human review route.
- **Illinois HB 3773** (in force since 1 January 2026): an outcome monitoring summary.
- **EU AI Act** (high-risk recruitment duties apply from 2 December 2027): a readiness checklist covering logging, human oversight and transparency.

### 11.5 Agents

- **Observatory analyst** (read_only): explains which interviewers, panels and stages are flagged, using aggregates only.
- **Audit drafter** (propose): drafts the compliance pack and needs approval before export.

### 11.6 Exit criteria

- [ ] All 4 planted problems are found, with no false flags on clean interviewers (over 3 seeds).
- [ ] Impact ratio, kappa and BH outputs match hand-worked fixtures exactly.
- [ ] Evidence tagger kappa against labels is reported (target of at least 0.7).
- [ ] A test proves no endpoint, tool or page returns candidate-level scores, and that n below 5 is suppressed.
- [ ] The analysis plan hash is logged before every run.

---

## 12. Pack 3: Rate Card Consolidation Engine

### 12.1 Synthetic world (`ratecards/generate.py`)

- 40 vendors and about 60 standard roles: families (engineering, data, QA, PM, BA, infrastructure, support) times seniority (junior, mid, senior, lead).
- 600 distinct billed titles: abbreviations, level codes ("SE III"), typos and vendor-specific names. The hidden ground-truth title-to-role map is stored in `restricted.title_truth`.
- 3 location tiers (US onshore, nearshore, India offshore) with currencies USD, MXN and INR, plus a monthly FX table.
- About 50,000 invoice lines over 24 months.
- Pricing models: time and materials, fixed-price SOW, monthly FTE, and outcome-based with a productivity commitment percentage.
- Quality and SLA scores per vendor.
- **Planted problems:**
  - 3 vendors bill 5 to 15% above their contracted card on some roles after month 12.
  - A batch of duplicate invoice lines.
  - Rate drift on 2 vendors.
  - One vendor that is cheap per hour but loses on total cost once productivity commitments are applied.
- **Record shape (ADR 0001):** invoice lines are NetSuite-style `vendorbill` transactions and lines (`tranid`,
  `trandate`, `entity`, `subsidiary`, `currency`, `exchangerate`, `amount`, `foreignamount`), with the
  exchange rate stamped at posting.

### 12.2 Pipeline

1. **Ingest** invoices, rate cards and contracts into `raw`.
2. **Map titles** (`mapping.py`). Normalize the text, shortlist the top 5 roles with rapidfuzz `token_set_ratio` (fastembed similarity is an optional second signal), then Haiku 5.5 picks `{role_id, confidence, reason}` from the shortlist. Auto-accept only if confidence is at least 0.85 and agrees with the fuzzy top-1; everything else goes to the review queue (`app/pages/4_Approvals.py`). Store who or what approved each mapping.
3. **Normalize** (`normalize.py`) to USD per baseline hour:
   - FTE monthly rate / standard hours (default 160, configurable).
   - Fixed price uses reported hours if present. Otherwise it stays in spend but out of rate benchmarks.
   - Outcome-based deals get an effective rate of `rate x (1 - productivity_commitment)`. For example, $110 with a 20% commitment equals $88, which beats a $100 time-and-materials rate.
   - Flag contracts older than 24 months as AI-renegotiation candidates.
4. **Benchmark** (`benchmark.py`): p25, median and p75 per role and location tier, with `n` and a `source` label (`internal` or `external`). External benchmarks only come from an optional `data/external_benchmarks.csv` with its source and date. Otherwise, show "internal only".
5. **Score vendors** (`scorecard.py`): a rate index (vendor rate / role median), quality, SLA, role coverage and spend share, with weights in `ratecards.yaml`.
6. **Optimize the panel** (`optimizer.py`) with OR-Tools MathOpt and GSCIP (verified working):
   - Variables: `keep_v` binary, `h_vr` at least 0 (hours).
   - Minimize the sum of `p_vr x h_vr`, where `p` is the effective normalized rate.
   - Demand: the sum over v of `h_vr` equals `D_r` for every role.
   - Linking: `h_vr` is at most `D_r x keep_v`.
   - Panel size: the sum of `keep_v` is at most `N`.
   - Critical roles: at least 2 kept vendors with an active card for the role.
   - Concentration: `h_vr` is at most `s_max x D_r` on critical roles.
   - Quality floor: `keep_v = 0` if quality is below `q_min`.
   - On infeasibility, relax one constraint family at a time and report which one binds. Never return silently.
7. **Savings bridge** (`bridge.py`): current cost, then rate harmonization, volume shift, vendor exits and leakage recovery (overbilling plus duplicates), giving the new cost. Every step is traced to invoice line IDs, and a property test proves the steps sum exactly. An explicit FX re-translation step follows leakage recovery ([ADR 0003](decisions/0003-bridge-fx-and-approval-hardening.md)).
8. **Drift monitor** (`drift.py`): flag any invoice line billed more than 2% above its contracted card, as a Dagster asset check.
9. **Tail plan** (`tail.py`): for vendors not kept, list spend, roles and a proposed exit or a negotiation brief (target rate, walk-away rate). Sonnet writes the text, and the numbers come from compute.

### 12.3 Agents

- **Benchmark analyst** (read_only).
- **Panel planner** (propose): proposes a panel, which needs approval.
- **Negotiation drafter** (propose).

### 12.4 Exit criteria

- [ ] Title mapping reaches at least 95% accuracy against the hidden map after review, with the review rate reported.
- [ ] Every planted overbill and duplicate is found and traced to line IDs.
- [ ] The optimizer matches brute force on small instances (property test) and never breaks the panel size or critical-role coverage.
- [ ] The savings bridge sums exactly.
- [ ] Total-cost normalization ranks the planted "cheap per hour" vendor correctly.

---

## 13. Phased build plan

| Phase | Scope | Exit criteria |
|---|---|---|
| 0. Scaffold | uv project, Python 3.13 pin, deps via `uv add`, ruff, pyright, pre-commit, Makefile, dbt skeleton, Dagster defs, empty Streamlit app, CI workflow, CLAUDE.md, `.mcp.json` | `make check` (lint, types, tests) is green; `make dbt-parse` works; `dagster dev` loads; Streamlit `/_stcore/health` returns ok |
| 1. Core | Settings, LLM wrapper, registry, tools, audit schema, lineage check, approvals, MCP server, offline eval runner | Unit tests for every module; in-process MCP test passes; lineage check rejects a planted unsupported number |
| 2. Rate cards | Section 12 | Section 12.4 |
| 3. Interviews | Section 11 | Section 11.6 |
| 4. FP&A | Section 10 | Section 10.8 |
| 5. API and UI | FastAPI routes per pack; Streamlit pages; approvals page; audit and cost page | API tests; each page loads against a fresh seed; cost footer shows real usage |
| 6. Polish | README with architecture diagram, data dictionary, demo script, ADRs, LLM eval baseline committed | Fresh clone, then `make demo` works in under 10 minutes |

Rate cards come first because they have the cleanest data and the clearest headline number, and the optimizer stresses the compute layer hardest.

---

## 14. Configuration files to create

The concrete files from this section live in the repository root (`pyproject.toml`, `Makefile`, `.mcp.json`, `.streamlit/config.toml`, `.env.example`, `.pre-commit-config.yaml`, `.github/workflows/*.yml`, `CLAUDE.md`). Treat those files as the source of truth; deviations from the original prompt are noted in comments inside them.

Notes carried from the original prompt:
- `.PHONY` matters in the Makefile: without it, `make app` does nothing because an `app/` folder exists.
- The DuckDB MCP server runs read-only by default for local files, so don't pass `--read-write`. `--ephemeral-connections` keeps the file unlocked so Dagster can still write. Claude Code asks you to approve project-scoped servers on first use.
- `llm-evals.yml` runs on `workflow_dispatch` only, using an `ANTHROPIC_API_KEY` repository secret. LLM evals never run on every push.
- CI enforces `--cov-fail-under=85` and runs the offline eval suite (from Phase 1).

---

## 15. Recommended plugins and tools

### 15.1 Install Claude Code

- macOS, Linux or WSL: `curl -fsSL https://claude.ai/install.sh | bash`
- Windows PowerShell: `irm https://claude.ai/install.ps1 | iex`

### 15.2 Claude Code plugins (`claude-plugins-official`)

`pyright-lsp`, `security-guidance`, `duckdb-skills`, `mcp-server-dev`, `context7`, `feature-dev`, `code-review`, `pr-review-toolkit`, `commit-commands`, `github`, `claude-md-management`, `hookify`. Install with `/plugin install <name>@claude-plugins-official`. Optional: `langfuse-observability`, `altimate-code`, `agent-sdk-dev`.

### 15.3 Account skills worth using

- **engineering:** `architecture`, `system-design`, `testing-strategy`, `code-review`.
- **clawness:** `review`, `test`, `security-audit` (run before any demo).
- **ml-pipeline:** forecast model selection in Phase 4.
- **dataviz:** read before building any chart, especially the bridge waterfall and impact-ratio charts.
- **frontend-design:** only if Streamlit is later replaced with a custom front end.

### 15.4 Editor extensions

Claude Code (`anthropic.claude-code`), Ruff (`charliermarsh.ruff`), Python (`ms-python.python`), Even Better TOML (`tamasfe.even-better-toml`), dbt Power User (`innoverio.vscode-dbt-power-user`).

### 15.5 Useful DuckDB features

No extensions are needed for the core build. `read_parquet`, `read_csv` and `median()` are built in.

---

## 16. Definition of done and demo script

The project is done when a fresh clone runs `make demo` and shows the following:

1. **Rate cards:** 40 vendors cut to a proposed panel of N. The savings bridge waterfall is traced to invoice lines. The planted overbilling is caught. The "cheap per hour" vendor loses on total cost.
2. **Interviews:** the lenient interviewer, the random panel, the adverse-impact stage and the non-predictive role are all flagged. The LL144-style table is generated. No candidate-level data is visible anywhere.
3. **FP&A:** actual versus plan bridge, 13-week cash forecast against the naive baseline, and the "30% of claims to instant payment" scenario.
4. **Audit and cost page:** every agent run, tool call, lineage record, approval and dollar spent.
5. **MCP:** Claude Code or Claude Desktop can call `ledgerlens` tools and gets the same numbers as the UI.

---

## 17. Ask the human before deciding these

- Using any real data (candidate, payroll or invoice data needs consent and legal review).
- A deployment target beyond local, and any hosting cost.
- A monthly API budget cap. Default is a $20 soft cap enforced by the LLM wrapper, which refuses new calls once it's exceeded.
- Replacing any decision in section 3.
