from __future__ import annotations

import pytest

from nexus_learning.workflow_friction import (
    build_friction_observation,
    build_workflow_recommendation,
    validate_friction_observation,
    validate_workflow_recommendation,
)


def observation(attempt: str, *, outcome: str = "FRICTION", types=("receipt", "checkpoint")):
    return build_friction_observation(
        task_id="task-1",
        operation_id="op-1",
        attempt_id=attempt,
        source_revision="git:abc",
        runtime_identity="runtime:1",
        friction_kind="handoff-recovery-gap",
        evidence_refs=(f"receipt:{attempt}", f"checkpoint:{attempt}"),
        evidence_types=types,
        outcome=outcome,
    )


def test_repeated_friction_becomes_evidence_bounded_recommendation():
    rows = [observation("a1"), observation("a2")]
    rec = build_workflow_recommendation(
        rows,
        recommendation="persist exact attempt checkpoint before handoff",
        required_evidence_types=("receipt", "checkpoint"),
    )
    assert rec["repetition_count"] == 2
    assert rec["coverage"]["missing"] == []
    assert rec["uncertainty"] == []
    assert rec["direct_mutation_allowed"] is False
    assert rec["adoption_requires_external_authority"] is True
    validation = validate_workflow_recommendation(
        rec,
        validator_identity="independent-validator",
        current_source_revision="git:abc",
        current_runtime_identity="runtime:1",
    )
    assert validation["disposition"] == "VALIDATED_FOR_CONSIDERATION"
    assert validation["adoption_authorized"] is False


def test_missing_coverage_is_explicit_uncertainty():
    rec = build_workflow_recommendation(
        [observation("a1", types=("receipt",)), observation("a2", types=("receipt",))],
        recommendation="candidate",
        required_evidence_types=("receipt", "checkpoint"),
    )
    assert rec["coverage"]["missing"] == ["checkpoint"]
    assert "evidence_coverage_incomplete" in rec["uncertainty"]


def test_chat_or_transcript_is_not_admissible_evidence():
    with pytest.raises(ValueError, match="conversation memory"):
        build_friction_observation(
            task_id="t", operation_id="o", attempt_id="a",
            source_revision="s", runtime_identity="r", friction_kind="f",
            evidence_refs=("chat:message-1",), evidence_types=("chat_transcript",),
        )


def test_negative_experiment_is_preserved_not_promoted_away():
    negative = observation("a3", outcome="NEGATIVE_EXPERIMENT")
    rec = build_workflow_recommendation(
        [observation("a1"), observation("a2"), negative],
        recommendation="candidate",
        required_evidence_types=("receipt", "checkpoint"),
    )
    assert negative["observation_hash"] in rec["negative_evidence"]


def test_tamper_and_stale_identity_fail_closed():
    row = observation("a1")
    tampered = dict(row)
    tampered["attempt_id"] = "other"
    with pytest.raises(ValueError, match="tampered"):
        validate_friction_observation(tampered)
    rec = build_workflow_recommendation(
        [row, observation("a2")],
        recommendation="candidate",
        required_evidence_types=("receipt", "checkpoint"),
    )
    result = validate_workflow_recommendation(
        rec,
        validator_identity="v",
        current_source_revision="git:new",
        current_runtime_identity="runtime:1",
    )
    assert result["disposition"] == "INSUFFICIENT_EVIDENCE"
    assert "source_or_runtime_identity_stale" in result["blockers"]
