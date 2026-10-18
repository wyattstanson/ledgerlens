# ADR 0001: Align FP&A and rate-card data with NetSuite structure and finance-analyst conventions

- Status: accepted (owner approval, 2026-10-19)
- Affects: Phase 2 (rate cards), Phase 4 (FP&A), Phase 5 (UI). Interviews pack is out of scope.

## Context

Two NetSuite skills are installed in the owner's Claude account: the AI connector rules (tool order,
SuiteQL safety, domain knowledge) and the finance analyst playbook (materiality, variance method,
period status, narrative shape). No NetSuite connector is attached, and BUILD_PROMPT rules keep all
data synthetic (section 17: real data needs owner sign-off and legal review).

We still want the synthetic world to look like a real ERP so that (a) the demo reads as credible to
finance reviewers and (b) a real NetSuite MCP feed could later replace the synthetic one without
reshaping the packs. This is the same swap pattern as the payments adapter (section 10.2).

## Decision

### 1. Data shape (Phase 2 and Phase 4)

Synthetic GL and vendor-bill tables use NetSuite record and field names:

| Table | Fields |
|---|---|
| `transaction` | id, tranid, trandate, recordtype, entity, subsidiary, postingperiod, posting, approvalstatus, currency, exchangerate, amount, foreignamount, memo |
| `transactionline` | id, transaction, account, amount, foreignamount, department, class, location |
| `account` | id, acctnumber, fullname, accttype, parent |
| `accountingperiod` | id, periodname, startdate, enddate, closed |
| `subsidiary` | id, name, currency, parent |

- FP&A: premium receipts post as `custpymt`; claim payouts and other GL postings post as
  `journalentry` lines tagged with payout channel. Rate cards: invoice lines are `vendorbill` lines
  in USD, MXN or INR.
- `amount` is base currency and `foreignamount` is native currency. The exchange rate is stamped at
  posting and never recalculated.
- Fiscal year start is set in `config/fpa.yaml`; nothing assumes 1 January.
- The latest accounting period is open (`closed = false`). Older periods are closed.
- New dbt tests: debits equal credits for every transaction, alongside the ledger-to-events reconciliation.

### 2. Read-only ledger adapter (Phase 4)

`fpa/ledger_adapter.py` defines a read-only protocol with a synthetic DuckDB implementation. Its tool
order mirrors the connector rules:

1. Report-shaped tools first: income statement, budget versus actual, AR aging.
2. Saved-search-shaped views next.
3. Raw query last, always capped at 1,000 rows, no `SELECT *`.

The protocol has no write methods. Plugging in a real NetSuite connector is a section 17 decision for
the owner, not something an agent can do.

### 3. Analysis conventions (FP&A agents and compute)

Defaults live in `config/fpa.yaml` and are applied by compute, never by the model:

- Materiality: financial statement 5% or $50,000; operating 3% or $25,000; immaterial screen 2% and $10,000.
  These are calibrated for $10M to $100M revenue. The synthetic insurer (about 20,000 policies) is
  assumed to sit in that band; the generator must report its actual annual premium so this stays checked.
- Sign convention: revenue above plan is favourable; expense below plan is favourable.
- Before a variance is flagged, compute checks for known seasonality, documented one-time items and
  reclassifications with an offset elsewhere.
- Premium receivable aging buckets: 0 to 30 days current, 31 to 60 monitor, 61 to 90 escalate,
  over 90 stale (needs management review).
- Answers about the open period are labelled preliminary, with a confidence level (high: closed and
  reconciled; medium: open with known gaps; low: missing or inconsistent data) and the reason for it.

### 4. Narrative shape (FP&A agents' structured final answer)

The variance, forecast and scenario agents return, inside the existing structured answer (section 9.4):

- `headline`: one sentence with the key number.
- `scope`: period, entity scope, currency, book, preliminary or final.
- `drivers`: the top two or three, each quantified in dollars and percent.
- `risks`: items labelled `FLAG:` or `CRITICAL:`.
- `action`: task, owner (a role), due date; or an explicit "no action needed".

Every number in these fields still goes through `numbers[]` and the lineage check (rule 1). FP&A
outputs carry a "not financial advice" note, like the "not legal advice" banner on compliance outputs.

### 5. UI (Phase 5)

FP&A pages show period and entity scope in the header and use restrained semantic colours (green
positive, amber warning, red negative, deep blue headings) within `docs/ui_guardrails.md`.

## Not adopted

- Hyperlinks to `system.netsuite.com` records: there is no real account, so the links would be fake.
- Live SuiteQL or any NetSuite writes.
- Applying these rules to the interviews pack: the finance skill excludes HR workflows.

## Consequences

- More columns in the synthetic generators and marts, and one more dbt test family.
- The adapter protocol needs its own unit tests, including a test that it exposes no write methods.
- Materiality thresholds become an input to the variance flags, so they get property tests at the boundaries.
