"""Your own numbers, through the same functions the packs use.

A client wants to know what the platform would say about their month before they load a warehouse
into it. These routes answer that: the figures they type go through the tested functions behind the
FP&A and rate-card pages, and come back with the same rules applied.

What they do not come back with is a lineage id. A lineage id resolves to a query and the rows it
read; a number computed from four figures someone typed has no rows behind it, so claiming one would
be the one thing this product is built not to do. The page says so next to every result, and nothing
here touches the warehouse or writes anything.
"""

import csv
import io
from typing import Any, Literal

from fastapi import APIRouter, HTTPException, UploadFile, status
from pydantic import BaseModel, ConfigDict, Field

from ledgerlens.fpa.bridge import Segment, variance_bridge
from ledgerlens.fpa.config import MaterialityConfig
from ledgerlens.fpa.materiality import Kind, Variance, classify, level_for, risk_label
from ledgerlens.ratecards.config import NormalizeConfig
from ledgerlens.ratecards.normalize import card_rate_usd_per_hour

router = APIRouter(prefix="/api/compute", tags=["compute"])

# A browser form, not a bulk loader: enough rows for a real segment list, few enough that a request
# cannot be used to make the free instance chew through its memory.
MAX_SEGMENTS = 500
MAX_CSV_BYTES = 256 * 1024

SEGMENT_COLUMNS = ("segment", "q_actual", "amount_actual", "q_plan", "amount_plan")

DISCLAIMER = (
    "Computed from the figures you entered. No warehouse rows back this, so it carries no lineage id."
)


class _Input(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SegmentInput(_Input):
    key: str = Field(min_length=1, max_length=120)
    q_actual: float = Field(ge=0)
    amount_actual: float
    q_plan: float = Field(ge=0)
    amount_plan: float


class BridgeInput(_Input):
    segments: list[SegmentInput] = Field(min_length=1, max_length=MAX_SEGMENTS)
    cash_actual: float
    cash_plan: float


class MaterialityInput(_Input):
    measure: str = Field(min_length=1, max_length=120)
    kind: Kind
    actual: float
    plan: float
    month: str = Field(default="", max_length=7)
    one_time: bool = False


class RateInput(_Input):
    pricing_model: Literal["tm", "fte", "outcome", "fixed"]
    rate: float = Field(gt=0)
    usd_per_unit: float = Field(default=1.0, gt=0)
    productivity_commitment: float = Field(default=0.0, ge=0, lt=1)
    standard_hours: float = Field(default=NormalizeConfig().standard_hours, gt=0)


def _bridge(payload: BridgeInput) -> dict[str, Any]:
    segments = [
        Segment(
            key=s.key,
            q_actual=s.q_actual,
            amount_actual=s.amount_actual,
            q_plan=s.q_plan,
            amount_plan=s.amount_plan,
        )
        for s in payload.segments
    ]
    bridge = variance_bridge(segments, payload.cash_actual, payload.cash_plan)
    return {
        "components": {name: value / 100 for name, value in bridge.components.items()},
        "actual": bridge.actual_cents / 100,
        "plan": bridge.plan_cents / 100,
        "variance": bridge.variance_cents / 100,
        "unit": "USD",
    }


def _materiality(payload: MaterialityInput) -> dict[str, Any]:
    config = MaterialityConfig()
    level = level_for(payload.measure)
    variance = Variance(
        month=payload.month,
        measure=payload.measure,
        kind=payload.kind,
        level=level,
        actual=payload.actual,
        plan=payload.plan,
    )
    thresholds = getattr(config, level)
    status_text = classify(variance.variance, variance.plan, thresholds, config.immaterial)
    # The reclassification screen needs the month's other expense measures, which one typed figure
    # does not have; a documented one-time item is the only override a single variance can carry.
    if status_text == "material" and payload.one_time:
        status_text = "one_time"
    return {
        "status": status_text,
        "level": level,
        "label": risk_label(status_text, level),
        "variance": variance.variance,
        "pct_of_plan": None if variance.plan == 0 else variance.pct,
        "favourable": variance.favourable,
        "threshold_usd": thresholds.usd,
        "threshold_pct": thresholds.pct,
        "unit": "USD",
    }


def _rate(payload: RateInput) -> dict[str, Any]:
    config = NormalizeConfig(standard_hours=payload.standard_hours)
    sticker, effective = card_rate_usd_per_hour(
        payload.pricing_model, payload.rate, payload.usd_per_unit, payload.productivity_commitment, config
    )
    return {
        "sticker_usd_per_hour": sticker,
        "effective_usd_per_hour": effective,
        "note": None
        if sticker is not None
        else "A fixed-price SOW has no rate per hour until the invoice reports the hours it covered.",
        "unit": "USD per hour",
    }


@router.post("/fpa.variance_bridge")
def compute_bridge(payload: BridgeInput) -> dict[str, Any]:
    """Volume, mix, rate and timing for segments you supply. The parts sum to the variance."""
    return {"result": _bridge(payload), "disclaimer": DISCLAIMER}


@router.post("/fpa.materiality")
def compute_materiality(payload: MaterialityInput) -> dict[str, Any]:
    """Whether one variance is material, on the ADR 0001 thresholds the packs use."""
    return {"result": _materiality(payload), "disclaimer": DISCLAIMER}


@router.post("/rc.card_rate")
def compute_rate(payload: RateInput) -> dict[str, Any]:
    """One vendor rate, normalised to USD per baseline hour."""
    return {"result": _rate(payload), "disclaimer": DISCLAIMER}


@router.post("/fpa.variance_bridge/csv")
async def bridge_from_csv(file: UploadFile, cash_actual: float, cash_plan: float) -> dict[str, Any]:
    """The same bridge, from a segment extract rather than a form.

    The file is read into memory, parsed and dropped. Nothing is stored, so an upload leaves no trace
    beyond the answer it produces.
    """
    raw = await file.read(MAX_CSV_BYTES + 1)
    if len(raw) > MAX_CSV_BYTES:
        raise HTTPException(
            status.HTTP_413_CONTENT_TOO_LARGE,
            f"the file is over {MAX_CSV_BYTES // 1024}KB; this takes a segment list, not a ledger export",
        )
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "the file is not UTF-8 text") from None

    reader = csv.DictReader(io.StringIO(text))
    missing = set(SEGMENT_COLUMNS) - set(reader.fieldnames or [])
    if missing:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST, f"the file needs these columns too: {', '.join(sorted(missing))}"
        )

    segments: list[SegmentInput] = []
    for number, row in enumerate(reader, start=2):  # row 1 is the header, so a complaint names the line
        if len(segments) >= MAX_SEGMENTS:
            raise HTTPException(status.HTTP_400_BAD_REQUEST, f"the file has over {MAX_SEGMENTS} segments")
        try:
            segments.append(
                SegmentInput(
                    key=row["segment"],
                    q_actual=float(row["q_actual"]),
                    amount_actual=float(row["amount_actual"]),
                    q_plan=float(row["q_plan"]),
                    amount_plan=float(row["amount_plan"]),
                )
            )
        except (TypeError, ValueError) as exc:
            raise HTTPException(status.HTTP_400_BAD_REQUEST, f"line {number}: {exc}") from exc

    if not segments:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "the file has a header but no segments")

    payload = BridgeInput(segments=segments, cash_actual=cash_actual, cash_plan=cash_plan)
    return {"result": _bridge(payload), "segments": len(segments), "disclaimer": DISCLAIMER}
