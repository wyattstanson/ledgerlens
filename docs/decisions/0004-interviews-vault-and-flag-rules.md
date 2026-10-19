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
     is picked after looking and would inflate false positives; both sides need n >= 30;
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
6. **Secondary suppression** (added after the Phase 3 security review). Primary suppression alone leaks:
   the stage total is known from the other dimension, so total minus the published cells gives back a
   lone hidden cell, and a group's n at one stage equals how many of it passed the stage before. So:
   - in each role x stage x dimension (and each sex and race line of the intersectional table), the hidden
     cells are either none or at least two covering at least 5 people; the smallest visible cell is hidden
     until that holds;
   - a group hidden at one stage stays hidden at every later stage of that role;
   - `share` and every other value column are nulled with n; the compliance JSON is an explicit column
     allow-list.
   Cost: sex has only three categories, so when "Unknown" has fewer than 5 people one of Female or Male is
   also hidden (18 of 32 role x stage comparisons in the seed-42 world). The planted cell stays visible.
7. **The generic DuckDB MCP server is locked down.** `.mcp.json` starts it with
   `SET enable_external_access=false; SET lock_configuration=true`, so a client can query the warehouse
   but cannot `ATTACH` the vault or read files such as the spot-check sheet (verified against
   `mcp-server-motherduck` 1.1.0, whose temporary connections each run the init SQL).

## Residual risks (accepted for now)

- A published cell of 5 or more where everyone (or no one) passed still tells you the outcome for each
  member of that group.
- Any MCP client can read every interviewer's calibration cells; who may see interviewer-level data is a
  Phase 5 authorization question.
- Suppression is checked against the additive relations listed above, not by a full linear disclosure
  audit across every published number.
