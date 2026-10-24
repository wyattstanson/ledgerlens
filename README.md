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

Status: Phase 4 of 6 (rate cards, interviews and FP&A built; FP&A forecasting waits on its first review gate;
web UI and polish to come). The full
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
uv run python -m ledgerlens.cli pipeline fpa
uv run python -m ledgerlens.core.evals.run --suite offline
```

The FP&A pipeline builds the dbt marts first (the ledger must reconcile to the payments feed every month, and
debits must equal credits), then the driver tree, volume/mix/rate/timing bridges for every month and fiscal
year to date, ADR 0001 materiality flags and the check-to-instant scenario
([ADR 0005](docs/decisions/0005-fpa-bridge-materiality-and-ledger-guard.md)). Forecasting goes through the
ml-pipeline workbench in [ml_pipeline/](ml_pipeline/PIPELINE.md): `uv run python ml_pipeline/01_eda.py`
redraws the exploration figures, and no model is fitted until each review gate is approved. The model notebooks
(`03_model.py` on) need statsforecast, whose statsmodels dependency loads a compiled module that Windows App
Control blocks on some machines; there, run them under Linux (WSL) with a separate environment:
`UV_PROJECT_ENVIRONMENT=~/.venvs/ledgerlens uv run python ml_pipeline/03_model.py`.

Interview data about people (candidates, ratings, notes, group attributes) is written only to
`data/vault.duckdb`. The warehouse and every tool see aggregates, and cells under five people are
suppressed ([ADR 0004](docs/decisions/0004-interviews-vault-and-flag-rules.md)). To check the evidence
tagger against your own judgement, run `uv run python -m ledgerlens.cli spotcheck interviews`, fill in
the `owner_has_evidence` column of `data/spot_check_50.csv` with yes or no, and rerun the interviews
pipeline.
