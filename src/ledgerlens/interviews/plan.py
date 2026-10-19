"""The pre-registered analysis plan as a validated model. A plan can only tighten the privacy floors."""

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field

from ledgerlens.core.guards import MIN_GROUP_N


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class CalibrationPlan(_Strict):
    min_n: int = Field(ge=MIN_GROUP_N)
    z_threshold: float = Field(gt=0)
    min_effect_d: float = Field(ge=0)
    min_flagged_cells: int = Field(ge=1)


class PanelPlan(_Strict):
    kappa_threshold: float
    min_candidates: int = Field(ge=MIN_GROUP_N)


class ProcessPlan(_Strict):
    rubric_coverage_min: float = Field(ge=0, le=1)
    evidence_quality_min: float = Field(ge=0, le=1)
    loop_coverage_min: float = Field(ge=0, le=1)


class FairnessPlan(_Strict):
    impact_ratio_threshold: float = Field(gt=0, le=1)
    min_test_n: int = Field(ge=MIN_GROUP_N)
    dimensions: tuple[str, ...]


class PredictivePlan(_Strict):
    min_n: int = Field(ge=MIN_GROUP_N)


class TaggerPlan(_Strict):
    kappa_target: float
    eval_sample: int = Field(ge=MIN_GROUP_N)
    spot_check_size: int = Field(ge=MIN_GROUP_N)


class AnalysisPlan(_Strict):
    name: str
    seed: int
    alpha: float = Field(gt=0, lt=1)
    bootstrap_resamples: int = Field(ge=200)
    min_group_n: int = Field(ge=MIN_GROUP_N)  # rule 2's floor; a plan may only raise it
    min_group_share: float = Field(ge=0.02)
    calibration: CalibrationPlan
    panel_agreement: PanelPlan
    process: ProcessPlan
    fairness: FairnessPlan
    predictive_validity: PredictivePlan
    tagger: TaggerPlan

    @classmethod
    def from_yaml(cls, path: Path) -> "AnalysisPlan":
        return cls.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
