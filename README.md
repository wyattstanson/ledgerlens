# LedgerLens

Governed analytics agents for three domains, sharing one core:

- **Rate Card Consolidation**: map billed titles to standard roles, compare vendors on total cost,
  choose the cheapest vendor panel that covers every role, and trace savings to invoice lines.
- **Interview Quality and Fairness Observatory**: measures interviews, interviewers and loops,
  never candidates.
- **Payments-aware FP&A**: forecasting and variance analysis for a P&C insurer, driven by how
  premiums are collected and claims are paid.

The language model plans and explains. Deterministic code produces every number, and every number
carries lineage to the query and rows behind it. All data is synthetic.

Status: Phase 0 (scaffold). The full specification is in [docs/BUILD_PROMPT.md](docs/BUILD_PROMPT.md).

## Quick start

Requires Python 3.13 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync --all-groups
make dbt-parse
make check
```

Other targets: `make dagster` (Dagster UI), `make app` (Streamlit), `make build` (materialise all assets).
On Windows without `make`, run the commands listed in the [Makefile](Makefile) directly.
