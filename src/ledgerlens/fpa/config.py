"""Typed view of config/fpa.yaml: the synthetic insurer, its plan misses, and analysis conventions (ADR 0001).

Every value is a synthetic assumption. The code defaults equal the shipped YAML (a test keeps them in step).
"""

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

Line = Literal["personal_auto", "homeowners", "small_commercial"]
Method = Literal["card", "ach", "check", "wallet", "agency_bill"]
Channel = Literal["check", "ach", "debit_push", "wallet", "instant"]
LINES: tuple[Line, ...] = ("personal_auto", "homeowners", "small_commercial")
METHODS: tuple[Method, ...] = ("card", "ach", "check", "wallet", "agency_bill")
CHANNELS: tuple[Channel, ...] = ("check", "ach", "debit_push", "wallet", "instant")


SHARE_TOLERANCE = 1e-9  # mixes and shares must sum to one within this


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class LineConfig(_Strict):
    share: float = Field(gt=0, lt=1)  # of the starting book
    annual_premium: float = Field(gt=0)  # mean written premium per policy
    premium_sigma: float = Field(default=0.3, gt=0)  # lognormal spread
    claim_frequency: float = Field(gt=0)  # claims per policy-month
    seasonal_amplitude: float = Field(default=0.0, ge=0, lt=1)
    seasonal_peak_month: int = Field(default=1, ge=1, le=12)  # calendar month with the most claims
    severity_median: float = Field(gt=0)
    severity_sigma: float = Field(gt=0)
    multiparty_share: float = Field(default=0.0, ge=0, lt=1)  # claims paid in two legs (e.g. mortgagee)


class MethodConfig(_Strict):
    mix: float = Field(gt=0, lt=1)
    on_time_rate: float = Field(gt=0, le=1)
    late_median_days: float = Field(gt=0)
    nonpayment_rate: float = Field(ge=0, lt=1)  # share of late instalments never paid


class ChannelConfig(_Strict):
    mix: float = Field(gt=0, lt=1)
    lag_median_days: float = Field(ge=0)  # approval to money in the claimant's hands
    lag_sigma: float = Field(default=0.5, gt=0)
    handling_cost: float = Field(gt=0)  # loss adjustment expense per payout
    return_rate: float = Field(default=0.0, ge=0, lt=1)  # payouts that bounce and are reissued


def _lines() -> dict[Line, LineConfig]:
    return {
        "personal_auto": LineConfig(
            share=0.55,
            annual_premium=1400,
            claim_frequency=0.0105,
            seasonal_amplitude=0.18,
            seasonal_peak_month=1,
            severity_median=4200,
            severity_sigma=0.9,
        ),
        "homeowners": LineConfig(
            share=0.35,
            annual_premium=1650,
            claim_frequency=0.0055,
            seasonal_amplitude=0.30,
            seasonal_peak_month=5,
            severity_median=8500,
            severity_sigma=1.1,
            multiparty_share=0.15,
        ),
        "small_commercial": LineConfig(
            share=0.10,
            annual_premium=5200,
            premium_sigma=0.45,
            claim_frequency=0.006,
            severity_median=14000,
            severity_sigma=1.2,
        ),
    }


def _methods() -> dict[Method, MethodConfig]:
    return {
        "card": MethodConfig(mix=0.34, on_time_rate=0.93, late_median_days=6, nonpayment_rate=0.10),
        "ach": MethodConfig(mix=0.30, on_time_rate=0.97, late_median_days=4, nonpayment_rate=0.08),
        "check": MethodConfig(mix=0.12, on_time_rate=0.80, late_median_days=12, nonpayment_rate=0.12),
        "wallet": MethodConfig(mix=0.14, on_time_rate=0.95, late_median_days=5, nonpayment_rate=0.10),
        "agency_bill": MethodConfig(mix=0.10, on_time_rate=0.70, late_median_days=25, nonpayment_rate=0.06),
    }


def _channels() -> dict[Channel, ChannelConfig]:
    return {
        "check": ChannelConfig(mix=0.40, lag_median_days=6, handling_cost=38, return_rate=0.012),
        "ach": ChannelConfig(mix=0.30, lag_median_days=3, lag_sigma=0.4, handling_cost=14),
        "debit_push": ChannelConfig(mix=0.12, lag_median_days=1, lag_sigma=0.4, handling_cost=9),
        "wallet": ChannelConfig(mix=0.10, lag_median_days=1, lag_sigma=0.4, handling_cost=8),
        "instant": ChannelConfig(mix=0.08, lag_median_days=0, lag_sigma=0.3, handling_cost=6),
    }


class PlantedConfig(_Strict):
    """Where actuals deliberately leave the plan, and where each miss shows up.

    - Extra personal-auto new business: the premium bridge's volume column (and lob mix, so its mix column).
    - Homeowners severity: the claims bridge's rate column.
    - ACH to card: the method_mix driver and premium timing (card collects sooner); not the premium mix
      column, because an instalment's price does not depend on how it is paid.
    - Slower checks: the check days_to_pay driver; claims timing also carries ordinary month-end noise.
    - Dearer check handling: the LAE bridge's rate column and the check handling_cost driver.
    """

    volume_line: Line = "personal_auto"
    volume_start_month: int = Field(default=18, ge=0)  # months from the window start
    volume_new_business_lift: float = Field(default=0.35, ge=0)  # extra new business vs plan
    severity_line: Line = "homeowners"
    severity_start_month: int = Field(default=24, ge=0)
    severity_lift: float = Field(default=0.15, ge=0)
    mix_start_month: int = Field(default=12, ge=0)
    mix_ach_to_card: float = Field(default=0.25, ge=0, le=1)  # share of ACH instalments that move to card
    timing_channel: Channel = "check"
    timing_actual_lag_median_days: float = Field(default=9, ge=0)  # the plan assumes the channel's lag
    lae_channel: Channel = "check"
    lae_actual_handling_cost: float = Field(default=44, gt=0)


class WorldConfig(_Strict):
    policies: int = Field(default=20_000, ge=500)  # in force at the window start
    months: int = Field(default=36, ge=24)
    end_month: str = Field(default="2026-09", pattern=r"^\d{4}-\d{2}$")
    new_business_per_month: int = Field(default=230, ge=0)
    renewal_rate: float = Field(default=0.88, gt=0, le=1)
    annual_rate_change: float = Field(default=0.05, ge=-0.5, le=0.5)
    pay_plans: dict[Literal["annual", "quarterly", "monthly"], float] = {"annual": 0.25, "quarterly": 0.25,
                                                                          "monthly": 0.5}  # fmt: skip
    approval_lag_median_days: float = Field(default=18, gt=0)
    opex_monthly: float = Field(default=900_000, gt=0)
    opex_noise: float = Field(default=0.04, ge=0)
    lines: dict[Line, LineConfig] = Field(default_factory=_lines)
    methods: dict[Method, MethodConfig] = Field(default_factory=_methods)
    channels: dict[Channel, ChannelConfig] = Field(default_factory=_channels)
    planted: PlantedConfig = PlantedConfig()

    @model_validator(mode="after")
    def _mixes_sum_to_one(self) -> "WorldConfig":
        for name, mix in (("lines", {k: v.share for k, v in self.lines.items()}),
                          ("methods", {k: v.mix for k, v in self.methods.items()}),
                          ("channels", {k: v.mix for k, v in self.channels.items()}),
                          ("pay_plans", self.pay_plans)):  # fmt: skip
            if abs(sum(mix.values()) - 1) > SHARE_TOLERANCE:
                raise ValueError(f"{name} shares must sum to 1, got {sum(mix.values())}")
        return self


class MaterialityLevel(_Strict):
    pct: float = Field(gt=0, lt=1)
    usd: float = Field(gt=0)


class MaterialityConfig(_Strict):
    """ADR 0001: calibrated for $10M to $100M revenue."""

    financial_statement: MaterialityLevel = MaterialityLevel(pct=0.05, usd=50_000)
    operating: MaterialityLevel = MaterialityLevel(pct=0.03, usd=25_000)
    immaterial: MaterialityLevel = MaterialityLevel(pct=0.02, usd=10_000)


class FpaConfig(_Strict):
    world: WorldConfig = WorldConfig()
    fiscal_year_start_month: int = Field(default=1, ge=1, le=12)
    materiality: MaterialityConfig = MaterialityConfig()
    aging_buckets_days: tuple[int, int, int] = (30, 60, 90)  # current, monitor, escalate, then stale
    one_time_items: tuple[str, ...] = ()  # "YYYY-MM:measure" entries documented as one-off, never flagged
    scenario_shift_share: float = Field(default=0.30, gt=0, le=1)  # check -> instant in the built-in scenario

    @model_validator(mode="after")
    def _aging_edges_increase(self) -> "FpaConfig":
        a, b, c = self.aging_buckets_days
        if not 0 < a < b < c:
            raise ValueError(
                f"aging_buckets_days must be positive and increasing, got {self.aging_buckets_days}"
            )
        return self

    @classmethod
    def from_yaml(cls, path: Path) -> "FpaConfig":
        return cls.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")) or {})
