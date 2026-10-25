"""Generate either synthetic FP&A world and write it, with its plan, to ``raw_fpa``.

The one entry point the CLI (``generate --pack fpa [--variant processor]``) and the Dagster asset share, so
both write the same world for the same variant and seed. Golden eval cases come from the same generators.
"""

from pathlib import Path

import polars as pl

from ledgerlens.fpa import generate as insurer_generate
from ledgerlens.fpa import golden as insurer_golden
from ledgerlens.fpa import plan as insurer_plan
from ledgerlens.fpa.config import FpaConfig, Variant
from ledgerlens.fpa.processor import generate as processor_generate
from ledgerlens.fpa.processor import golden as processor_golden
from ledgerlens.fpa.processor import plan as processor_plan


def build_world(warehouse: Path, config: FpaConfig, variant: Variant, seed: int) -> dict[str, pl.DataFrame]:
    """Write one world (replacing whichever was there) and return every raw table as written."""
    if variant == "processor":
        world = processor_generate.generate_world(config.processor.world, seed)
        plan = processor_plan.build_plan(config.processor.world)
        return processor_generate.write_world(warehouse, world, plan)
    insurer = insurer_generate.generate_world(config.world, seed)
    return insurer_generate.write_world(warehouse, insurer, insurer_plan.build_plan(config.world, seed))


def golden_cases(config: FpaConfig, variant: Variant, seed: int) -> list[dict]:
    """Golden cases for one world, from its generator's own tables and plan."""
    if variant == "processor":
        world = processor_generate.generate_world(config.processor.world, seed)
        return processor_golden.golden_cases(world, processor_plan.build_plan(config.processor.world), config)
    insurer = insurer_generate.generate_world(config.world, seed)
    return insurer_golden.golden_cases(insurer, insurer_plan.build_plan(config.world, seed), config)
