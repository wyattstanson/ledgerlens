"""LedgerLens command line.

python -m ledgerlens.cli generate --seed 42             # write the synthetic world(s) to the warehouse
python -m ledgerlens.cli pipeline ratecards [--simulate-review]
python -m ledgerlens.cli golden ratecards --seed 42     # regenerate tests/evals/golden/ratecards.jsonl
"""

import argparse
import json
from collections.abc import Sequence

from ledgerlens.core.approvals import ApprovalQueue
from ledgerlens.core.db import ensure_audit_schema
from ledgerlens.ratecards.config import RateCardsConfig
from ledgerlens.ratecards.generate import generate_world, write_world
from ledgerlens.ratecards.golden import golden_cases, write_golden
from ledgerlens.ratecards.mapping import simulate_review_from_truth
from ledgerlens.ratecards.pipeline import run_ratecards
from ledgerlens.settings import REPO_ROOT, get_settings

GOLDEN_RATECARDS = REPO_ROOT / "tests" / "evals" / "golden" / "ratecards.jsonl"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="ledgerlens")
    sub = parser.add_subparsers(dest="command", required=True)
    gen = sub.add_parser("generate", help="write synthetic data to the warehouse")
    gen.add_argument("--seed", type=int, default=None)
    pipe = sub.add_parser("pipeline", help="run a pack's compute steps")
    pipe.add_argument("pack", choices=["ratecards"])
    pipe.add_argument(
        "--simulate-review",
        action="store_true",
        help="demo/evals only: resolve pending title reviews from the hidden truth, audited as simulated",
    )
    gold = sub.add_parser("golden", help="regenerate a pack's golden eval set from generator truth")
    gold.add_argument("pack", choices=["ratecards"])
    gold.add_argument("--seed", type=int, default=None)
    args = parser.parse_args(argv)

    settings = get_settings()
    config = RateCardsConfig.from_yaml(settings.config_dir / "ratecards.yaml")
    seed = settings.seed if getattr(args, "seed", None) is None else args.seed
    if args.command == "generate":
        ensure_audit_schema(settings.db)
        world = generate_world(config.world, seed)
        write_world(settings.db, world)
        print(
            f"ratecards: {world.invoice_lines.height} invoice lines, {world.vendors.height} vendors"
            f" -> {settings.db}"
        )
    elif args.command == "pipeline":
        ensure_audit_schema(settings.db)
        solution = run_ratecards(settings.db, config)
        if args.simulate_review:
            reviewed = simulate_review_from_truth(settings.db, ApprovalQueue(settings.db))
            print(f"simulated review of {reviewed} titles")
            solution = run_ratecards(settings.db, config)
        status = (
            "infeasible (see ratecards.panel_meta)" if solution is None else json.dumps(list(solution.kept))
        )
        print(f"panel: {status}")
    else:
        cases = golden_cases(generate_world(config.world, seed), config)
        write_golden(cases, GOLDEN_RATECARDS)
        print(f"{len(cases)} golden cases -> {GOLDEN_RATECARDS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
