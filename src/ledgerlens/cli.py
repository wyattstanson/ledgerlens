"""LedgerLens command line.

python -m ledgerlens.cli generate [--pack ratecards|interviews|fpa|all] [--seed 42] [--variant processor]
python -m ledgerlens.cli pipeline ratecards [--simulate-review]
python -m ledgerlens.cli pipeline interviews
python -m ledgerlens.cli pipeline fpa [--variant processor]  # dbt build of the FP&A marts, bridges, scenario
python -m ledgerlens.cli forecast fpa                 # refit the reviewed forecast models (Linux: ADR 0005)
python -m ledgerlens.cli golden ratecards|interviews|fpa [--seed 42] [--variant processor]
python -m ledgerlens.cli spotcheck interviews        # writes the blind 50-note labelling sheet

--variant picks the FP&A world (BUILD_PROMPT 10.1, ADR 0005): the P&C insurer (default, from config/fpa.yaml)
or the payment processor. The warehouse holds one at a time; `pipeline fpa --variant X` refuses another world.
"""

import argparse
import json
from collections.abc import Sequence

from ledgerlens.core.approvals import ApprovalQueue
from ledgerlens.core.db import ensure_audit_schema
from ledgerlens.core.llm import LLMClient, ModelsConfig
from ledgerlens.fpa.build import build_world
from ledgerlens.fpa.build import golden_cases as fpa_golden_cases
from ledgerlens.fpa.config import VARIANTS, FpaConfig
from ledgerlens.fpa.pipeline import run_fpa
from ledgerlens.interviews import generate as iv_generate
from ledgerlens.interviews import golden as iv_golden
from ledgerlens.interviews.config import InterviewsConfig
from ledgerlens.interviews.pipeline import run_interviews, spot_check_notes
from ledgerlens.interviews.plan import AnalysisPlan
from ledgerlens.interviews.tagger import (
    ClaudeTagger,
    EvidenceTagger,
    RuleTagger,
    read_owner_labels,
    write_spot_check,
)
from ledgerlens.ratecards import generate as rc_generate
from ledgerlens.ratecards.config import RateCardsConfig
from ledgerlens.ratecards.golden import golden_cases, write_golden
from ledgerlens.ratecards.mapping import simulate_review_from_truth
from ledgerlens.ratecards.pipeline import run_ratecards
from ledgerlens.settings import REPO_ROOT, Settings, get_settings

GOLDEN_DIR = REPO_ROOT / "tests" / "evals" / "golden"
PROCESSOR_GOLDEN_DIR = GOLDEN_DIR / "fpa_processor"  # apart: an offline run reads one world's warehouse
SPOT_CHECK = REPO_ROOT / "data" / "spot_check_50.csv"


def _tagger(settings: Settings, config: InterviewsConfig) -> EvidenceTagger:
    if config.tagger.kind == "claude":
        if settings.anthropic_api_key is None:
            raise SystemExit("tagger.kind is 'claude' but ANTHROPIC_API_KEY is not set")
        models = ModelsConfig.from_yaml(settings.config_dir / "models.yaml")
        return ClaudeTagger(LLMClient(models, settings.db, api_key=settings.anthropic_api_key))
    return RuleTagger()


def main(argv: Sequence[str] | None = None) -> int:  # noqa: PLR0912, PLR0915 (one dispatcher)
    parser = argparse.ArgumentParser(prog="ledgerlens")
    sub = parser.add_subparsers(dest="command", required=True)
    gen = sub.add_parser(
        "generate", help="write synthetic data (warehouse; interview person data to the vault)"
    )
    gen.add_argument("--pack", choices=["ratecards", "interviews", "fpa", "all"], default="all")
    gen.add_argument("--seed", type=int, default=None)
    gen.add_argument(
        "--variant", choices=VARIANTS, default=None, help="FP&A world (default: config/fpa.yaml)"
    )
    pipe = sub.add_parser("pipeline", help="run a pack's compute steps")
    pipe.add_argument("pack", choices=["ratecards", "interviews", "fpa"])
    pipe.add_argument(
        "--simulate-review",
        action="store_true",
        help=(
            "rate cards, demo/evals only: resolve pending title reviews from the hidden truth "
            "(audited as simulated)"
        ),
    )
    pipe.add_argument(
        "--variant", choices=VARIANTS, default=None, help="FP&A: refuse to run on another world"
    )
    gold = sub.add_parser("golden", help="regenerate a pack's golden eval set from generator truth")
    gold.add_argument("pack", choices=["ratecards", "interviews", "fpa"])
    gold.add_argument("--seed", type=int, default=None)
    gold.add_argument(
        "--variant", choices=VARIANTS, default=None, help="FP&A world (default: config/fpa.yaml)"
    )
    fcst = sub.add_parser("forecast", help="refit the reviewed forecast models and store their forecasts")
    fcst.add_argument("pack", choices=["fpa"])
    spot = sub.add_parser("spotcheck", help="write the blind tagger spot-check sheet for a human to label")
    spot.add_argument("pack", choices=["interviews"])
    args = parser.parse_args(argv)

    settings = get_settings()
    rc_config = RateCardsConfig.from_yaml(settings.config_dir / "ratecards.yaml")
    iv_config = InterviewsConfig.from_yaml(settings.config_dir / "interviews.yaml")
    fpa_config = FpaConfig.from_yaml(settings.config_dir / "fpa.yaml")
    plan = AnalysisPlan.from_yaml(settings.config_dir / "analysis_plan.yaml")
    seed = settings.seed if getattr(args, "seed", None) is None else args.seed
    variant = getattr(args, "variant", None)
    if variant is not None and getattr(args, "pack", None) not in ("fpa", "all"):
        parser.error("--variant applies to the fpa pack")
    ensure_audit_schema(settings.db)

    if args.command == "generate":
        if args.pack in ("ratecards", "all"):
            world = rc_generate.generate_world(rc_config.world, seed)
            rc_generate.write_world(settings.db, world)
            print(f"ratecards: {world.invoice_lines.height} invoice lines, {world.vendors.height} vendors")
        if args.pack in ("interviews", "all"):
            iv_world = iv_generate.generate_world(iv_config.world, seed)
            iv_generate.write_world(settings.db, settings.restricted_db, iv_world)
            print(f"interviews: {iv_world.ratings.height} ratings; person data -> {settings.restricted_db}")
        if args.pack in ("fpa", "all"):
            world = variant or fpa_config.variant
            tables = build_world(settings.db, fpa_config, world, seed)
            print(
                f"fpa ({world}): {tables['payment_events'].height} payment events, "
                f"{tables['transaction'].height} transactions"
            )
    elif args.command == "pipeline" and args.pack == "ratecards":
        solution = run_ratecards(settings.db, rc_config)
        if args.simulate_review:
            reviewed = simulate_review_from_truth(settings.db, ApprovalQueue(settings.db))
            print(f"simulated review of {reviewed} titles")
            solution = run_ratecards(settings.db, rc_config)
        if solution is None:
            print("panel: infeasible (see ratecards.panel_meta)")
        else:
            print(f"panel: {json.dumps(list(solution.kept))}")
    elif args.command == "pipeline" and args.pack == "fpa":
        run = run_fpa(settings.db, fpa_config, variant=variant)
        print(
            f"fpa ({run.variant}): as of {run.as_of}; {run.totals.height} variance rows; "
            f"scenario {run.scenario.name}"
        )
    elif args.command == "pipeline":
        registered = run_interviews(
            settings.db, settings.restricted_db, plan, _tagger(settings, iv_config), owner_labels=SPOT_CHECK
        )
        print(f"interviews: plan v{registered.version} ({registered.sha256[:12]}) registered and run")
    elif args.command == "forecast":
        from ledgerlens.fpa.forecast import run_forecast  # noqa: PLC0415 (statsforecast loads only here)

        result = run_forecast(settings.db)
        print(
            f"fpa: forecasts from {result.origin}: {result.forecasts['unique_id'].nunique()} series, "
            f"{result.totals['ds'].nunique()} months of rebuilt totals"
        )
    elif args.command == "golden":
        if args.pack == "ratecards":
            cases = golden_cases(rc_generate.generate_world(rc_config.world, seed), rc_config)
        elif args.pack == "fpa":
            cases = fpa_golden_cases(fpa_config, variant or fpa_config.variant, seed)
        else:
            cases = iv_golden.golden_cases(iv_generate.generate_world(iv_config.world, seed), plan)
        folder = (
            PROCESSOR_GOLDEN_DIR
            if args.pack == "fpa" and (variant or fpa_config.variant) == "processor"
            else GOLDEN_DIR
        )
        path = folder / f"{args.pack}.jsonl"
        write_golden(cases, path)
        print(f"{len(cases)} golden cases -> {path}")
    else:
        if read_owner_labels(SPOT_CHECK):
            raise SystemExit(f"{SPOT_CHECK} already has your labels; move it before writing a new sheet")
        notes = spot_check_notes(settings.restricted_db, plan)
        write_spot_check(SPOT_CHECK, notes)
        print(f"{len(notes)} notes -> {SPOT_CHECK}.")
        print("Fill owner_has_evidence with yes/no, then rerun `pipeline interviews`.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
