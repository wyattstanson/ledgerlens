.PHONY: sync dbt-parse generate build check dagster app api mcp demo

# build selects key:"*" rather than '*': Click expands a bare * into filenames on Windows.

sync:        ; uv sync --all-groups
dbt-parse:   ; cd transform && uv run dbt parse --profiles-dir .
generate:    ; uv run python -m ledgerlens.cli generate --seed 42
build:       dbt-parse ; uv run dagster asset materialize --select 'key:"*"' -m ledgerlens.orchestration.defs
check:       ; uv run ruff check . && uv run ruff format --check . && uv run pyright && uv run pytest -q
dagster:     dbt-parse ; uv run dagster dev -m ledgerlens.orchestration.defs
app:         ; uv run streamlit run app/Home.py
api:         ; uv run uvicorn ledgerlens.api.main:app --reload
mcp:         ; uv run python -m ledgerlens.mcp.server
demo:        sync generate build ; $(MAKE) app
