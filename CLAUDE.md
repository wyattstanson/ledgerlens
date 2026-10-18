# LedgerLens
Governed analytics agents: shared core + packs (fpa, interviews, ratecards). Spec: docs/BUILD_PROMPT.md.

## Commands
make sync | make check | make dbt-parse | make build | make dagster | make app | make mcp

## Rules
- LLM never computes numbers; compute functions return numbers with lineage_id. Lineage check must pass.
- No candidate-level scores or rankings anywhere. Aggregates need n >= 5.
- Write actions need an approval record. Agents only propose.
- MCP SDK is v2: from mcp.server.mcpserver import MCPServer (not fastmcp).
- Anthropic SDK: output_config for structured outputs and effort; no budget_tokens thinking.
- Run dbt parse from inside transform/. pandas stays <3; Python stays 3.13.
- New logic needs tests; watch each new test fail once before trusting it.
- UI work follows docs/ui_guardrails.md (no generic vibecoded tells, but keep depth and motion).

## Windows notes
- A *folder* named pyproject.toml sits in D:\Grey-HP-Laptop-By-Aryansh; [tool.uv.workspace] stops uv walking into it.
- Click expands a bare `*` into filenames on Windows: select all Dagster assets with `key:"*"`.
