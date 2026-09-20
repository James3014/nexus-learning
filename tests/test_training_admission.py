"""Acceptance tests for issue #21: evaluation-only vs training-eligible data.

Locks in the Semantic Reflex V2 Wave 1 lesson: third-party hosted model output
used as private evaluation evidence must never be converted into labels,
pseudo-labels, rewards, preference pairs, or hard negatives for training.
"""

from __future__ import annotations

import pytest

from nexus_learning.contracts import (
    EVALUATION_ONLY_PURPOSE,
    LEARNING_POLICY_EVIDENCE_PURPOSE,
    TRAINING_ADMISSION_FORBIDDEN,
    TRAINING_ADMISSION_QUALITY_GATED,
    TRAINING_CANDIDATE_PURPOSE,
    apply_autodata_quality_gate,
    build_learning_experience,
    learning_experience_from_dict,
    project_model_training,
    resolve_training_admission,
)


def _verified_experience(data_purpose: str = TRAINING_CANDIDATE_PURPOSE):
    exp = build_learning_experience(
        task_id="task-p21",
        task_type="bug",
        usage_trace={
            "phase_trace": {
                "S": "start",
                "P": "plan",
                "X": "context",
                "D": "design",
                "R": "repair",
                "A": "audit",
                "C": "close",
            },
            "capabilities": {
                "artifact_gate_passed": True,
                "artifact_refs": ["artifact:task-p21"],
                "claim_verified": True,
                "delivery_gate_passed": True,
                "delivery_refs": ["delivery:task-p21"],
            },
            "s2t": {"trace_path": ".nexus/reports/s2t/task-p21.jsonl"},
        },
        capability_receipts=[
            {
                "name": "autoreason",
                "selected": True,
                "invoked": True,
                "evidence_present": True,
                "gate_passed": True,
                "outcome_contributed": True,
            }
        ],
    )
    from dataclasses import replace

    return replace(exp, data_purpose=data_purpose)


def _quality_row(**overrides: object) -> dict:
    row: dict = {
        "task_id": "task-p21",
        "eligible_for_training": True,
        "leakage_risk": False,
        "reward_hacking_risk": False,
        "trajectory_steps": 12,
        "information_density": 0.85,
    }
    row.update(overrides)
    return row


def test_evaluation_only_evidence_produces_no_training_target():
    """Evaluation-only evidence yields an empty target set, never a training target."""
    exp = _verified_experience(EVALUATION_ONLY_PURPOSE)
    projection = project_model_training(exp)
    assert projection["training_eligible"] is False
    assert projection["training_admission"] == TRAINING_ADMISSION_FORBIDDEN
    assert projection["targets"] == []
    assert projection["exclusion_reason"] == "EVALUATION_ONLY_TRAINING_FORBIDDEN"


def test_evaluation_only_failed_behavior_never_becomes_hard_negative():
    """Failed eval-only outcomes must not be written back as hard negatives."""
    exp = _verified_experience(EVALUATION_ONLY_PURPOSE)
    projection = project_model_training(exp)
    assert "hard_negative" not in projection["targets"]
    assert "hard_negative" not in "%s" % projection


def test_policy_evidence_is_training_forbidden_and_quality_gated_is_forward_compatible():
    assert (
        resolve_training_admission(LEARNING_POLICY_EVIDENCE_PURPOSE)
        == TRAINING_ADMISSION_FORBIDDEN
    )
    assert resolve_training_admission(EVALUATION_ONLY_PURPOSE) == TRAINING_ADMISSION_FORBIDDEN
    assert (
        resolve_training_admission(TRAINING_CANDIDATE_PURPOSE)
        == TRAINING_ADMISSION_QUALITY_GATED
    )


def test_training_forbidden_survives_autodata_quality_gate():
    """A forbidden projection stays forbidden through the autodata quality gate."""
    exp = _verified_experience(EVALUATION_ONLY_PURPOSE)
    gated = apply_autodata_quality_gate(project_model_training(exp), _quality_row())
    assert gated["training_eligible"] is False
    assert gated["targets"] == []
    assert gated["training_admission"] == TRAINING_ADMISSION_FORBIDDEN
    assert "training_forbidden_by_data_purpose" in gated["model_training_gate"]["reasons"]
    assert gated["model_training_gate"]["status"] == "fail"


def test_positive_quality_row_cannot_override_training_prohibition():
    """Even a fully eligible quality row cannot re-admit evaluation-only data."""
    exp = _verified_experience(EVALUATION_ONLY_PURPOSE)
    gated = apply_autodata_quality_gate(
        project_model_training(exp),
        _quality_row(eligible_for_training=True, leakage_risk=False),
    )
    assert gated["training_eligible"] is False
    assert gated["targets"] == []
    # The autodata quality itself is fine; the prohibition is a data-purpose gate.
    assert gated["autodata_gate"]["status"] == "pass"


def test_legacy_training_candidate_keeps_quality_gated_behavior():
    """Ordinary training candidates (the legacy default) stay quality-gated."""
    exp = _verified_experience(TRAINING_CANDIDATE_PURPOSE)
    projection = project_model_training(exp)
    assert projection["training_eligible"] is True
    assert projection["training_admission"] == TRAINING_ADMISSION_QUALITY_GATED
    assert projection["targets"] == ["preference_pair", "reward_row"]

    gated = apply_autodata_quality_gate(
        projection,
        _quality_row(eligible_for_training=False, trajectory_steps=2),
    )
    assert gated["training_eligible"] is False
    assert gated["targets"] == ["hard_negative"]


def test_unknown_data_purpose_fails_closed():
    """Unknown or malformed data purposes hard-fail instead of downgrading."""
    with pytest.raises(ValueError, match="EXPERIENCE_UNKNOWN_DATA_PURPOSE"):
        resolve_training_admission("UNSPECIFIED_PURPOSE")
    with pytest.raises(ValueError, match="EXPERIENCE_UNKNOWN_DATA_PURPOSE"):
        resolve_training_admission("ORIGIN_SECRET/VENDOR_LLM_OUTPUT")
    # Empty/missing purpose is the legacy default (training candidate), not a downgrade.
    assert resolve_training_admission("") == TRAINING_ADMISSION_QUALITY_GATED


def test_projection_exposes_machine_readable_exclusion_reason():
    """Callers get a stable, machine-readable exclusion reason for forbidden data."""
    exp = _verified_experience(EVALUATION_ONLY_PURPOSE)
    projection = project_model_training(exp)
    assert projection["exclusion_reason"] == "EVALUATION_ONLY_TRAINING_FORBIDDEN"
    assert isinstance(projection["exclusion_reason"], str)


def test_learning_experience_round_trips_data_purpose():
    """data_purpose survives the dict round trip and defaults to candidate."""
    exp = _verified_experience(EVALUATION_ONLY_PURPOSE)
    rebuilt = learning_experience_from_dict(exp.to_dict())
    assert rebuilt.data_purpose == EVALUATION_ONLY_PURPOSE

    legacy = learning_experience_from_dict(exp.to_dict() | {"data_purpose": None})
    assert legacy.data_purpose == TRAINING_CANDIDATE_PURPOSE


def test_recommendation_adoption_unchanged_for_existing_memory_path():
    """Learning-policy recommendation/adoption behavior is unaffected by #21."""
    from nexus_learning.contracts import (
        build_learning_policy_recommendation,
        build_nexus_learning_episode,
        evaluate_learning_policy_recommendation,
    )

    ep = build_nexus_learning_episode(
        task_id="task_rec",
        source="runtime_closure",
        terminal_outcome="SUCCESS",
        terminal_evidence={"verifier": "pytest", "receipt": "rec_rec", "verifier_status": "passed"},
        qualification={
            "repeatability": True,
            "prevention_rule": "rule",
            "authority_qualification": True,
        },
        lesson_disposition="graduated",
        learning_write_succeeded=True,
    )
    rec = build_learning_policy_recommendation(
        source_episodes=[ep],
        source_evidence_refs=["rec_rec", "retrieval_receipt:g2", "physical_consumption:ollama"],
        source_revision="rev-p21",
        runtime_identity="local_model_executor",
        task_fingerprint="task_rec",
        off_arm={"task_id": "task_rec", "verifier_status": "failed", "receipt": "off"},
        on_arm={"task_id": "task_rec", "verifier_status": "passed", "receipt": "on"},
        applicable_scope={"task_family": "record_serialization"},
        recommended_policy_delta={"episodic_memory_injection": {"enabled": True}},
        current_policy={"episodic_memory_injection": {"enabled": False}},
        expected_effect="Improve pass rate",
        rollback_target={"target_state": {"episodic_memory_injection": {"enabled": False}}},
    )
    assert rec["observed_effect"] == "paired_memory_uplift_observed"
    validation = evaluate_learning_policy_recommendation(
        rec,
        validator_identity="validator:g6",
        current_workspace_revision="rev-p21",
        current_runtime_identity="local_model_executor",
    )
    assert validation["validation_disposition"] == "VALIDATED_FOR_ADOPTION_CONSIDERATION"

def test_policy_evidence_projection_has_specific_exclusion_reason():
    exp = _verified_experience(LEARNING_POLICY_EVIDENCE_PURPOSE)
    projection = project_model_training(exp)
    assert projection["training_eligible"] is False
    assert projection["targets"] == []
    assert projection["exclusion_reason"] == "LEARNING_POLICY_EVIDENCE_TRAINING_FORBIDDEN"


def test_quality_gate_clears_targets_from_inconsistent_forbidden_projection():
    forged = {
        "schema_version": "nexus_model_training_projection.v1",
        "experience_id": "exp-forged",
        "training_eligible": True,
        "training_admission": TRAINING_ADMISSION_FORBIDDEN,
        "exclusion_reason": "EVALUATION_ONLY_TRAINING_FORBIDDEN",
        "targets": ["preference_pair", "reward_row"],
        "source_trace_refs": [".nexus/reports/s2t/forged.jsonl"],
    }
    gated = apply_autodata_quality_gate(forged, _quality_row())
    assert gated["training_eligible"] is False
    assert gated["targets"] == []
    assert "training_forbidden_by_data_purpose" in gated["model_training_gate"]["reasons"]


def test_missing_quality_row_preserves_training_forbidden_reason():
    exp = _verified_experience(EVALUATION_ONLY_PURPOSE)
    gated = apply_autodata_quality_gate(project_model_training(exp), None)
    assert gated["training_eligible"] is False
    assert gated["targets"] == []
    assert "training_forbidden_by_data_purpose" in gated["model_training_gate"]["reasons"]
