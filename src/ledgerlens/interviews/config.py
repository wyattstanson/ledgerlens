"""Typed view of config/interviews.yaml: the synthetic interview world (all values are assumptions)."""

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class StagePassRates(_Strict):
    screen: float = Field(default=0.75, gt=0, lt=1)
    technical: float = Field(default=0.70, gt=0, lt=1)
    onsite: float = Field(default=0.65, gt=0, lt=1)
    hiring_manager: float = Field(default=0.80, gt=0, lt=1)


class WorldConfig(_Strict):
    interviewers: int = Field(default=50, ge=20)
    candidates: int = Field(default=2000, ge=400)
    planted_role_share: float = Field(default=0.25, gt=0, lt=1)  # the adverse-impact role is the busiest
    panels_per_role: int = Field(default=3, ge=1)
    pass_rates: StagePassRates = StagePassRates()
    rating_noise: float = Field(default=0.22, gt=0)
    lenient_shift: float = 0.8
    adverse_keep_rate: float = Field(default=0.65, gt=0, le=1)  # passes kept for the planted group
    evidence_rate_range: tuple[float, float] = (0.88, 0.97)


class TaggerConfig(_Strict):
    kind: str = Field(default="rules", pattern="^(rules|claude)$")


class InterviewsConfig(_Strict):
    world: WorldConfig = WorldConfig()
    tagger: TaggerConfig = TaggerConfig()

    @classmethod
    def from_yaml(cls, path: Path) -> "InterviewsConfig":
        return cls.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")) or {})
