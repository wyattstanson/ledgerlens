import json
from pathlib import Path

import polars as pl
import pytest
from fakes import FakeClient, text_message

from ledgerlens.core.approvals import ApprovalQueue, ApprovalStatus
from ledgerlens.core.llm import LLMClient, ModelsConfig
from ledgerlens.ratecards.config import MappingConfig
from ledgerlens.ratecards.generate import SENIORITIES, SPECIALIZATIONS, role_id, standard_title
from ledgerlens.ratecards.mapping import (
    Candidate,
    FuzzyPicker,
    LlmPicker,
    evaluate_mapping,
    load_title_map,
    map_titles,
    normalize_title,
    queue_reviews,
    resolve_review,
    role_vocabulary,
    save_title_map,
    shortlist,
)

CFG = MappingConfig()
ROLES = pl.DataFrame(
    [
        {"role_id": role_id(s, sen), "title": standard_title(s, sen)}
        for s in SPECIALIZATIONS
        for sen in SENIORITIES
    ]
)
VOCAB = role_vocabulary(ROLES)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Sr. Backend Eng", "senior backend engineer"),
        ("Data Engineer III", "data engineer senior"),
        ("QA Analyst L1", "qa analyst junior"),
        ("Project Manager", "mid project manager"),  # unqualified = mid level
        ("Jnuior DevOps Engineer", "junior devops engineer"),  # adjacent-letter swap fixed
        ("FS Eng, Lead", "full stack engineer lead"),
    ],
)
def test_normalize_title(raw: str, expected: str):
    assert normalize_title(raw, VOCAB) == expected


def test_explicit_seniority_rules_out_other_seniorities():
    candidates = shortlist("Backend Engineer III", ROLES, 5, VOCAB)
    assert candidates[0].role_id == "backend.senior"
    assert all(not c.seniority_match for c in candidates[1:] if not c.role_id.endswith(".senior"))


def test_fuzzy_picker_accepts_clear_winners_and_queues_synonyms():
    out = map_titles(["Senior Data Scientist", "ML Scientist III"], ROLES, FuzzyPicker(CFG), CFG)
    rows = {r["billed_title"]: r for r in out.iter_rows(named=True)}
    assert rows["Senior Data Scientist"]["role_id"] == "data_scientist.senior"
    assert rows["Senior Data Scientist"]["decided_by"] == "rule:fuzzy-v1"
    assert rows["ML Scientist III"]["status"] == "pending_review"
    assert rows["ML Scientist III"]["role_id"] is None


def test_fuzzy_picker_rejects_a_near_tie_between_same_seniority_roles():
    near_tie = [
        Candidate("data_engineer.senior", 95.0, 95.0, True),
        Candidate("data_scientist.senior", 93.0, 93.0, True),
    ]
    pick = FuzzyPicker(CFG).pick("Senior Data Something", near_tie, ROLES)
    assert pick.role_id is None
    assert pick.suggested_role_id == "data_engineer.senior"
    clear = [near_tie[0], Candidate("data_scientist.senior", 80.0, 80.0, True)]
    assert FuzzyPicker(CFG).pick("Senior Data Something", clear, ROLES).role_id == "data_engineer.senior"


def test_fuzzy_picker_margin_boundary_is_inclusive():
    # margin == offline_accept_margin (6) accepts; one point less does not.
    at_margin = [Candidate("a.senior", 96.0, 96.0, True), Candidate("b.senior", 90.0, 90.0, True)]
    below = [Candidate("a.senior", 96.0, 96.0, True), Candidate("b.senior", 91.0, 91.0, True)]
    assert FuzzyPicker(CFG).pick("t", at_margin, ROLES).role_id == "a.senior"
    assert FuzzyPicker(CFG).pick("t", below, ROLES).role_id is None


def _llm(models_config: ModelsConfig, db: Path, answer: dict) -> LLMClient:
    return LLMClient(models_config, db, client=FakeClient([text_message(json.dumps(answer))]))


@pytest.mark.parametrize(
    ("answer_role", "confidence", "accepted"),
    [("backend.senior", 0.95, True), ("backend.senior", 0.80, False), ("backend.lead", 0.99, False)],
)
def test_llm_picker_needs_confidence_and_agreement_with_fuzzy(
    models_config: ModelsConfig, db: Path, answer_role: str, confidence: float, accepted: bool
):
    llm = _llm(models_config, db, {"role_id": answer_role, "confidence": confidence, "reason": "r"})
    candidates = shortlist("Backend Engineer III", ROLES, 5, VOCAB)
    pick = LlmPicker(llm, CFG).pick("Backend Engineer III", candidates, ROLES)
    assert (pick.role_id is not None) is accepted
    assert pick.decided_by == "llm:claude-haiku-5-5"
    request = llm.client.messages.requests[0]
    assert (
        '<data source="invoice:billed_title">' in request["messages"][0]["content"]
    )  # untrusted text fenced
    assert (
        request["output_config"]["format"]["schema"]["properties"]["role_id"]["enum"][0] == "backend.senior"
    )


def test_review_queue_and_human_resolution(db: Path):
    approvals = ApprovalQueue(db)
    mapped = map_titles(["ML Scientist III", "Senior Data Scientist"], ROLES, FuzzyPicker(CFG), CFG)
    queued = queue_reviews(mapped, approvals, agent_run_id="run_1")
    save_title_map(db, queued)
    [pending] = approvals.pending()
    assert pending.action == "rc.approve_mapping" and pending.payload["billed_title"] == "ML Scientist III"

    resolve_review(
        db,
        approvals,
        pending.approval_id,
        role_id="data_scientist.senior",
        decided_by="priya",
        reason="ML = DS",
    )
    row = load_title_map(db).filter(pl.col("billed_title") == "ML Scientist III").row(0, named=True)
    assert (row["role_id"], row["status"], row["decided_by"]) == (
        "data_scientist.senior",
        "reviewed",
        "priya",
    )
    assert approvals.get(pending.approval_id).status is ApprovalStatus.APPROVED


def test_evaluate_mapping_reports_review_rate():
    title_map = pl.DataFrame(
        {
            "billed_title": ["a", "b", "c", "d"],
            "role_id": ["x", "y", None, "z"],
            "status": ["accepted", "accepted", "pending_review", "reviewed"],
        }
    )
    truth = pl.DataFrame({"billed_title": ["a", "b", "c", "d"], "role_id": ["x", "q", "w", "z"]})
    result = evaluate_mapping(title_map, truth)
    assert result["auto_accuracy"] == 0.5
    assert result["review_rate"] == 0.5
    assert result["accuracy_after_review"] == 0.5
