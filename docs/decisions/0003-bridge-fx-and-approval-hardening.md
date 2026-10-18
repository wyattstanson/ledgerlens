# ADR 0003: An explicit FX step in the savings bridge, and a hardened approval gate

- Status: accepted (owner decision after the Phase 2 review, 2026-10-19)
- Affects: BUILD_PROMPT section 12.2 step 7 (savings bridge), section 4 rule 3 and section 9.1 (approvals).

## 1. Savings bridge: FX re-translation step

Invoice lines carry USD at the exchange rate stamped at posting (ADR 0001); panel prices use the as-of
rate. Without a separate step, currency movement showed up as "rate harmonization" savings that no
negotiation produced. The bridge is now:

current cost -> leakage recovery -> **FX re-translation** -> rate harmonization -> volume shift ->
vendor exits -> new cost

- FX re-translation is computed only where the vendor has an hourly card for the role.
- A kept vendor's work that has no hourly card (for example fixed price) is booked as volume shift,
  never as a vendor exit.
- Every line carries `current_cost` and `new_cost` rows, so the headline numbers have lineage too.

## 2. Approval gate

- **One gate for every caller.** `execute_tool` refuses any write tool unless `approval_id` names an
  approved, human-decided, unused record for that action. Agents, the API and evals cannot bypass it.
- **Approvals bind to state.** A write tool may define a `snapshot`. Proposals store it (for
  `rc.accept_panel`: kept vendors, as-of month, cost and a hash), and execution refuses if the state
  has changed since approval.
- **Single use.** Executing consumes the approval (`approved -> executed`) in the same transaction as
  the write.
- **Deciders.** Names are NFKC-normalized. Agent, MCP, pipeline and eval actors are refused. Demo and
  eval decisions use `kind="simulated"` with a `simulated:` name, are labelled in the audit log, and
  write tools refuse them.
- **Proposals.** A proposal from an agent's final answer must name a write tool on that agent's own
  list, with arguments that validate. One run can create at most five proposals.
- **Reviews are atomic.** A title-mapping decision and its mapping update commit together, and the
  chosen role must exist in the catalog.
