"""Issues #31/#32: preflight gates + campaign closeout/reopen."""

from __future__ import annotations

import pytest

from nexus_learning.campaign_gates import (
    CLOSEOUT_CLAIM_CEILING,
    PREFLIGHT_CLAIM_CEILING,
    build_campaign_closeout,
    build_cohort_preflight,
    verify_closeout,
    verify_preflight,
)


def _preflight(**overrides):
    kw = {
        "cohort_id": "r2-local-assist",
        "incumbent_identity": {
            "deterministic_kind": "literal_extraction",
            "result_hash": "sha256:" + "a" * 64,
        },
        "incumbent_result": "PASS",
        "candidate_levels": [
            {
                "level_id": "2b",
                "mechanism": "generative_extract",
                "disposition": "FAILED",
                "failure_reason": "false_safe",
            },
            {"level_id": "4b", "mechanism": "generative_extract", "disposition": "NOT_EVALUATED"},
        ],
        "role_metrics": {"false_safe_rate": 0.12, "unknown_recall": 0.8, "accuracy": 0.9},
        "candidate_coverage": {"candidates_cover_reranker_claims": True},
    }
    kw.update(overrides)
    return build_cohort_preflight(**kw)


def test_preflight_eligible_with_incumbent_and_metrics():
    body = _preflight()
    assert body["eligible"] is True
    assert body["claim_ceiling"] == PREFLIGHT_CLAIM_CEILING
    assert verify_preflight(body)


def test_preflight_requires_role_metrics_not_accuracy_alone():
    body = _preflight(role_metrics={"accuracy": 0.95})
    assert body["role_metrics"] == {"accuracy": 0.95}
    # accuracy-only cohort stays ineligible until safety metrics arrive
    body2 = _preflight(role_metrics={})
    assert body2["eligible"] is False
    assert any("accuracy alone" in g for g in body2["evidence_gaps"])


def test_adjacent_same_reason_failure_triggers_stop_loss():
    body = _preflight(
        candidate_levels=[
            {
                "level_id": "2b",
                "mechanism": "m",
                "disposition": "FAILED",
                "failure_reason": "false_safe",
            },
            {
                "level_id": "4b",
                "mechanism": "m",
                "disposition": "FAILED",
                "failure_reason": "false_safe",
            },
        ]
    )
    assert body["stop_loss_triggered"] is True
    assert body["eligible"] is False


def test_candidate_coverage_precondition_blocks_reranker_claims():
    body = _preflight(candidate_coverage={"candidates_cover_reranker_claims": False})
    assert body["eligible"] is False
    assert any("coverage" in g for g in body["evidence_gaps"])


def test_tampered_preflight_fails():
    body = _preflight()
    body["content_sha256"] = "0" * 64
    assert not verify_preflight(body)
    assert not verify_preflight({"schema": "wrong"})


def _closeout():
    return build_campaign_closeout(
        campaign_id="r1-r2-retrieval-20260929",
        closed_experiments=[
            {
                "experiment_id": "R1",
                "disposition": "STOPPED_BY_GATE",
                "highest_safe_claim": "no general small-model decision layer",
                "receipt_sha256": "x",
            },
            {
                "experiment_id": "R2",
                "disposition": "STOPPED_BY_GATE",
                "highest_safe_claim": "deterministic literal extraction admitted",
                "receipt_sha256": "y",
            },
            {
                "experiment_id": "RetrievalV2",
                "disposition": "PASS",
                "highest_safe_claim": "sparse candidate generation improved; localization additive-only",
                "receipt_sha256": "8dc40b9a",
            },
        ],
        highest_safe_claims={"R1": "deterministic pre-gates carry safe portion"},
        not_proven=["generative local assist over deterministic", "dense default path"],
        reopen_triggers=[
            {
                "kind": "new_role_with_residual_gap",
                "condition": "deterministic incumbent has proven residual gap",
            },
            {
                "kind": "dense_index_economics_change",
                "condition": "persistent prebuilt index removes cold-index stop",
            },
        ],
    )


def test_closeout_round_trip():
    body = _closeout()
    assert body["claim_ceiling"] == CLOSEOUT_CLAIM_CEILING
    assert len(body["closed_experiments"]) == 3
    assert verify_closeout(body)


def test_closeout_rejects_unknown_reopen_kind():
    with pytest.raises(ValueError):
        build_campaign_closeout(
            campaign_id="c",
            closed_experiments=[{"experiment_id": "R1", "disposition": "PASS"}],
            highest_safe_claims={},
            not_proven=[],
            reopen_triggers=[{"kind": "new_model_appeared", "condition": "x"}],
        )


def test_closeout_tamper_fails():
    body = _closeout()
    body["campaign_id"] = "rewritten"
    assert not verify_closeout(body)
