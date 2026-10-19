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

Status: Phase 3 of 6 (rate cards and interviews built; FP&A, web UI and polish to come). The full
specification is in [docs/BUILD_PROMPT.md](docs/BUILD_PROMPT.md); design decisions are in
[docs/decisions](docs/decisions).

## Quick start

Requires Python 3.13 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync --all-groups
make dbt-parse
make check
```

Other targets: `make dagster` (Dagster UI), `make app` (Streamlit), `make build` (materialise all assets).
On Windows without `make`, run the commands listed in the [Makefile](Makefile) directly.

## Run the packs

Everything runs offline; no API key is needed.

```bash
uv run python -m ledgerlens.cli generate --seed 42
uv run python -m ledgerlens.cli pipeline ratecards --simulate-review
uv run python -m ledgerlens.cli pipeline interviews
uv run python -m ledgerlens.core.evals.run --suite offline
```

Interview data about people (candidates, ratings, notes, group attributes) is written only to
`data/vault.duckdb`. The warehouse and every tool see aggregates, and cells under five people are
suppressed ([ADR 0004](docs/decisions/0004-interviews-vault-and-flag-rules.md)). To check the evidence
tagger against your own judgement, run `uv run python -m ledgerlens.cli spotcheck interviews`, fill in
the `owner_has_evidence` column of `data/spot_check_50.csv` with yes or no, and rerun the interviews
pipeline.
