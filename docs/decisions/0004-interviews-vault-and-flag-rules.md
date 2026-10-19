# ADR 0004: Interview data vault, pre-registered flag rules and plan registration

- Status: accepted (owner decisions, 2026-10-19)
- Affects: BUILD_PROMPT section 11 (Phase 3), section 9.3 (audit schema).

## Context

`.mcp.json` gives a generic DuckDB SQL server read access to `data/warehouse.duckdb`. Any person-level
interview table in the warehouse could therefore be queried into a candidate ranking, which rule 2
forbids. Threshold-only flags (impact ratio < 0.80, |z| > 1.5) also fire on noise across hundreds of
cells.

## Decisions

1. **Vault.** Everything person-level for interviews lives in `data/vault.duckdb`: candidates,
   interviews, ratings, notes, stage outcomes, performance, group attributes (sex, race or ethnicity in
   EEO-1 categories plus Unknown) and the hidden truth. Only the interviews generator and pipeline open
   it. The warehouse holds reference data (roles, competencies, interviewer ids) and aggregates only.
   Lineage `row_ids` name aggregate rows, never people.
2. **Flag rule.** A statistical flag needs both the plan threshold and a Benjamini-Hochberg-adjusted
   p < 0.05, with one BH family per run covering every calibration and fairness test:
   - calibration: ratings are first centred on their role x competency x stage mean (later stages only
     see candidates who passed earlier ones, so raw means would flag hiring managers for meeting stronger
     candidates). A cell is flagged when |z| > 1.5 against the spread of peer interviewers' centred means,
     q < 0.05 and |Cohen's d| >= 0.5, with n >= 10; an interviewer is flagged when at least two cells in
     one role are flagged;
   - impact ratio: ratio < 0.80 (against the highest group, as LL144 defines it) and q < 0.05, where the
     test compares the group with all other candidates at that stage, never with the highest group, which
     is picked after looking and would inflate false positives; both sides need n >= 30 (the published
     effect size is against the highest published group, see decision 6);
   - panel agreement: Fleiss' kappa < 0.4 (rule-based, no p-value);
   - coverage and evidence shares: rule-based thresholds;
   - predictive validity: bootstrap CI of Spearman's rho includes 0, with n >= 30.
   Every reported effect has a 95% bootstrap CI; no bare p-values.
3. **Plan registration.** `config/analysis_plan.yaml` is validated, serialized as canonical JSON and
   hashed before any compute. A new hash creates a new version in `audit.analysis_plans` (now keyed by
   plan name and version, with the plan JSON). Each run is logged in `audit.analysis_runs` and every
   result row carries the run id, plan version and hash. A plan can only tighten the floors
   (`min_group_n >= 5`, `min_group_share >= 0.02`).
4. **Tagger.** A deterministic rule tagger is the default and gets the kappa report. The optional Claude
   Haiku path uses the Message Batches API and runs only with an API key. The owner's 50 labels are the
   real check, since rules tested only against generator text overstate agreement.
5. **Output guard.** Tools may declare an `output_guard` that `execute_tool` runs on every result. The
   interviews guard refuses results with person-level keys (case-insensitive), vault ids (in keys, values
   and number labels) or cells with n < 5 of any numeric type. A blocked result's lineage rows are deleted.
   A sweep test calls every tool (directly and over MCP) and inspects exports and audit rows.
6. **Disclosure audit** (replaces the first-cut neighbour rule; owner decision 2026-10-19 to keep strict
   protection and win comparisons back safely). Primary suppression hides every cell under 5, but a reader
   can combine published numbers. `interviews/disclosure.py` writes down what a reader knows as a linear
   system per role and asks, for each small hidden cell, how far its size and its passed and failed counts
   can be narrowed. A cell is disclosed when any of the three has fewer than 5 possible values. Extra cells
   are hidden only to repair a disclosure: the cheapest visible cell (intersectional before one-way,
   Unknown or tiny groups before others, smaller before larger) among those in a constraint that binds the
   exposed cell, read from the linear program's dual values. The search uses the linear relaxation (GLOP);
   the result is then checked, and repaired if needed, against a whole-number reader (GSCIP).
   The reader is assumed to know, conservatively:
   - every visible cell's n and selected count;
   - every stage total to within the nearest 10, each stage's passes (the next stage's total), and the
     hires per role exactly;
   - that each role x stage x dimension line sums to its total and intersectional cells sum to the one-way
     cells;
   - that a group's n at one stage is its passed count at the stage before.
   Publication changes that make this assumption true:
   - every group that applied to a role has a row at every stage (n may be 0), so a missing row never
     reveals that a group had no one left;
   - hidden rows keep only their labels, and `share` and `p_raw` are not published;
   - impact ratios and Cohen's h (with its interval) compare a group with the highest *published* group.
     Statistics against everyone else at the stage give the stage total away: with h to 2 decimals for
     two groups, a reader solves for the stage's applicants and passes, and Unknown follows by
     subtraction (Female or Male would then be hidden in 30 of 32 comparisons). The flag still uses the
     pre-registered test against everyone else; its q is published only as a band (<0.001, <0.01, <0.05,
     >=0.05);
   - workload counts (calibration, process quality, panel sizes) are published to the nearest 10, so
     they cannot pin a stage total; hires stay exact.
   Result on the seed-42 world: one of Female or Male is hidden in 9 of 32 role x stage comparisons, down
   from 18 under the neighbour rule. A whole-number re-audit of the published tables finds no disclosed
   cell on seeds 42, 43 and 44.
7. **The generic DuckDB MCP server is locked down.** `.mcp.json` starts it with
   `SET enable_external_access=false; SET lock_configuration=true`, so a client can query the warehouse
   but cannot `ATTACH` the vault or read files such as the spot-check sheet (verified against
   `mcp-server-motherduck` 1.1.0, whose temporary connections each run the init SQL).

## Residual risks (accepted for now)

- A published cell of 5 or more where everyone (or no one) passed still tells you the outcome for each
  member of that group.
- Any MCP client can read every interviewer's calibration cells; who may see interviewer-level data is a
  Phase 5 authorization question.
- Not modelled by the audit: the q bands, the flags, h's interval and per-interviewer workload counts.
  Each is coarse or rounded, but their combination is not proven harmless.
- The audit assumes the reader knows every stage total to within 10. If another table ever publishes an
  exact stage total, the model must be updated or that table rounded.
