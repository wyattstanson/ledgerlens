"""Typed view of config/ratecards.yaml."""

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

RelaxFamily = Literal["panel_size", "critical_coverage", "concentration", "quality_floor"]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class WorldConfig(_Strict):
    vendors: int = Field(default=40, ge=6)
    specializations_per_vendor: int = Field(default=6, ge=1)
    months: int = Field(default=24, ge=13)
    start_month: str = Field(default="2024-10", pattern=r"^\d{4}-\d{2}$")
    max_projects_per_vendor: int = Field(default=4, ge=1)
    line_probability: float = Field(default=0.85, gt=0, le=1)
    distinct_titles: int = Field(default=600, ge=64)
    duplicate_lines: int = Field(default=150, ge=0)
    overbill_vendors: int = Field(default=3, ge=0)
    overbill_role_share: float = Field(default=0.3, gt=0, le=1)
    overbill_start_month: int = 12
    drift_vendors: int = Field(default=2, ge=0)
    drift_per_month: float = 0.004
    drift_start_month: int = 6


class NormalizeConfig(_Strict):
    standard_hours: float = Field(default=160, gt=0)
    renegotiation_age_months: int = 24


class MappingConfig(_Strict):
    shortlist_size: int = Field(default=5, ge=2)
    offline_accept_score: float = 90
    offline_accept_margin: float = 6
    llm_accept_confidence: float = 0.85


class DriftConfig(_Strict):
    threshold: float = Field(default=0.02, ge=0)


class OptimizerConfig(_Strict):
    panel_size: int = Field(default=18, ge=1)
    critical_top_k: int = Field(default=8, ge=0)
    max_share_critical: float = Field(default=0.7, gt=0, le=1)
    min_quality: float = Field(default=0.6, ge=0, le=1)
    baseline_months: int = Field(default=12, ge=1)
    relax_order: tuple[RelaxFamily, ...] = (
        "panel_size",
        "critical_coverage",
        "concentration",
        "quality_floor",
    )


class ScorecardWeights(_Strict):
    rate: float = 0.4
    quality: float = 0.25
    sla: float = 0.15
    coverage: float = 0.1
    spend_share: float = 0.1


class ScorecardConfig(_Strict):
    weights: ScorecardWeights = ScorecardWeights()


class RateCardsConfig(_Strict):
    world: WorldConfig = WorldConfig()
    normalize: NormalizeConfig = NormalizeConfig()
    mapping: MappingConfig = MappingConfig()
    drift: DriftConfig = DriftConfig()
    optimizer: OptimizerConfig = OptimizerConfig()
    scorecard: ScorecardConfig = ScorecardConfig()

    @classmethod
    def from_yaml(cls, path: Path) -> "RateCardsConfig":
        return cls.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")) or {})
