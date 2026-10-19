"""Eval runner CLI.

    uv run python -m ledgerlens.core.evals.run --suite offline   # CI, no API key
    uv run python -m ledgerlens.core.evals.run --suite llm       # manual, needs ANTHROPIC_API_KEY

Exit codes: 0 all passed, 1 a case failed (or no cases without --allow-empty), 2 cannot run.
"""

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from ledgerlens.core.agent_loop import AgentDeps
from ledgerlens.core.approvals import ApprovalQueue
from ledgerlens.core.db import ensure_audit_schema
from ledgerlens.core.evals.golden import (
    dump_report,
    load_golden,
    run_llm_case,
    run_offline_case,
    summarize,
)
from ledgerlens.core.llm import LLMClient, ModelsConfig
from ledgerlens.core.registry import AgentRegistry
from ledgerlens.settings import REPO_ROOT, get_settings

DEFAULT_GOLDEN_DIR = REPO_ROOT / "tests" / "evals" / "golden"
REPORT_DIR = REPO_ROOT / "docs" / "evals"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run LedgerLens golden-set evals.")
    parser.add_argument("--suite", choices=["offline", "llm"], required=True)
    parser.add_argument("--golden-dir", type=Path, default=DEFAULT_GOLDEN_DIR)
    parser.add_argument("--db", type=Path, default=None, help="warehouse path (defaults to settings)")
    parser.add_argument("--allow-empty", action="store_true", help="pass when no golden cases exist yet")
    args = parser.parse_args(argv)

    settings = get_settings()
    db_path = args.db or settings.db
    cases = load_golden(args.golden_dir, args.suite)
    if not cases:
        print(f"No {args.suite} golden cases in {args.golden_dir}.")
        return 0 if args.allow_empty else 1

    ensure_audit_schema(db_path)
    from ledgerlens.packs import default_tool_registry  # noqa: PLC0415 (composition root)

    tools = default_tool_registry()
    if args.suite == "offline":
        results = [run_offline_case(c, tools=tools, db_path=db_path) for c in cases]
    else:
        if settings.anthropic_api_key is None:
            print("The llm suite needs ANTHROPIC_API_KEY in the environment.", file=sys.stderr)
            return 2
        config = ModelsConfig.from_yaml(settings.config_dir / "models.yaml")
        deps = AgentDeps(
            tools=tools,
            llm=LLMClient(config, db_path, api_key=settings.anthropic_api_key),
            approvals=ApprovalQueue(db_path),
            db_path=db_path,
        )
        agents = AgentRegistry.from_yaml(settings.config_dir / "agents.yaml")
        results = [run_llm_case(c, agents=agents, deps=deps) for c in cases]

    summary = summarize(results)
    if args.suite == "llm":
        print(f"Report written to {dump_report(summary, REPORT_DIR, args.suite)}")
    print(json.dumps(summary, indent=2))
    return 0 if summary["passed"] == summary["cases"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
