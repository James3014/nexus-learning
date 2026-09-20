"""Acceptance tests for issue #20: calibration / policy-freeze / held-out integrity.

These tests lock in the Semantic Reflex V2 Wave 1 lesson: an experiment must
freeze evaluation identity and truth, derive operating policy from a calibration
population only, freeze the policy hash before held-out evaluation, and then
preserve the PASS / STOP / DEFER / insufficient-calibration / reject-all result
instead of converting it into an adoptable positive claim.
"""

from __future__ import annotations

import pytest

from nexus_learning.contracts import (
    build_learning_policy_recommendation,
    build_nexus_learning_episode,
    evaluate_learning_policy_recommendation,
)
from nexus_learning.experiment_integrity import (
    CALIBRATED,
    FROZEN_REJECT_ALL,
    INDEPENDENCE_UNIT_BASE,
    INDEPENDENCE_UNIT_ROW,
    INSUFFICIENT_CALIBRATION,
    TERMINAL_DEFER,
    TERMINAL_NEGATIVE,
    TERMINAL_PASS,
    TERMINAL_STOP,
    build_experiment_integrity,
    validate_experiment_integrity,
)


@pytest.fixture()
def cal_heldout() -> tuple[list[dict], list[dict]]:
    calibration = [
        {"identity": "row:c1", "base_identity": "base:cal"},
        {"identity": "row:c2", "base_identity": "base:cal"},
    ]
    heldout = [
        {"identity": "row:h1", "base_identity": "base:heldout"},
        {"identity": "row:h2", "base_identity": "base:heldout"},
    ]
    return calibration, heldout


def _frozen_policy(**overrides: object) -> dict:
    policy = {
        "active_route": "direct_mode",
        "route_version": 3,
        "policy_hash_revision": "rev-frozen-1",
    }
    policy.update(overrides)
    return policy


def test_e1_disjoint_populations_with_prefrozen_policy_validate():
    """Disjoint calibration/evaluation with a pre-frozen policy validates."""
    calibration, heldout = _cal_heldout()
    integrity = build_experiment_integrity(
        experiment_id="exp:semantic-reflex-v2-w1",
        calibration_members=calibration,
        heldout_members=heldout,
        independence_unit=INDEPENDENCE_UNIT_ROW,
        policy_derivation_ref="calibration_analysis:w1",
        frozen_policy=_frozen_policy(),
        freeze_generation=10,
        heldout_evaluation_start_generation=20,
        calibration_status=CALIBRATED,
        terminal_outcome=TERMINAL_PASS,
    )
    validate_experiment_integrity(integrity)
    assert integrity["schema"] == "nexus.learning_experiment_integrity.v1"
    assert integrity["calibration"]["member_count"] == 2
    assert integrity["heldout"]["member_count"] == 2
    assert integrity["terminal"]["outcome"] == TERMINAL_PASS
    assert integrity["claim_ceiling"] == (
        "experiment-integrity evidence only; no model/route/production claim"
    )


def test_e2_direct_row_overlap_fails():
    """A member present in both calibration and held-out populations fails closed."""
    calibration, heldout = _cal_heldout()
    heldout.append({"identity": "row:c2", "base_identity": "base:cal"})
    with pytest.raises(ValueError, match="EXPERIMENT_POPULATION_OVERLAP:row_identity"):
        build_experiment_integrity(
            experiment_id="exp:overlap",
            calibration_members=calibration,
            heldout_members=heldout,
            independence_unit=INDEPENDENCE_UNIT_ROW,
            policy_derivation_ref="calibration_analysis:w1",
            frozen_policy=_frozen_policy(),
            freeze_generation=10,
            heldout_evaluation_start_generation=20,
        )


def test_e3_shared_base_identity_fails_at_base_independence_unit():
    """Distinct rows sharing a forbidden parent/base identity are not independent."""
    calibration, heldout = _cal_heldout()
    # Same base, distinct row identities -> overlap only visible at BASE unit.
    heldout.append({"identity": "row:h9", "base_identity": "base:cal"})
    with pytest.raises(ValueError, match="EXPERIMENT_POPULATION_OVERLAP:base_identity"):
        build_experiment_integrity(
            experiment_id="exp:base-overlap",
            calibration_members=calibration,
            heldout_members=heldout,
            independence_unit=INDEPENDENCE_UNIT_BASE,
            policy_derivation_ref="calibration_analysis:w1",
            frozen_policy=_frozen_policy(),
            freeze_generation=10,
            heldout_evaluation_start_generation=20,
        )


def test_e3b_switching_independence_unit_breaks_binding():
    """Rebinding identities at a different unit cannot hide overlap."""
    calibration, heldout = _cal_heldout()
    # Row unit hides the base overlap at build; validator must recompute and fail.
    integrity = build_experiment_integrity(
        experiment_id="exp:row-build",
        calibration_members=calibration,
        heldout_members=heldout,
        independence_unit=INDEPENDENCE_UNIT_ROW,
        policy_derivation_ref="calibration_analysis:w1",
        frozen_policy=_frozen_policy(),
        freeze_generation=10,
        heldout_evaluation_start_generation=20,
    )
    integrity["independence_unit"] = INDEPENDENCE_UNIT_BASE
    with pytest.raises(ValueError, match="EXPERIMENT_CALIBRATION_POPULATION_HASH_MISMATCH"):
        validate_experiment_integrity(integrity)


def test_e4_policy_frozen_after_evaluartion_start_hard_blocks():
    """Freezing policy after held-out evaluation started is forbidden."""
    calibration, heldout = _cal_heldout()
    with pytest.raises(
        ValueError, match="EXPERIMENT_POLICY_FROZEN_AFTER_EVALUATION_START"
    ):
        build_experiment_integrity(
            experiment_id="exp:late-freeze",
            calibration_members=calibration,
            heldout_members=heldout,
            independence_unit=INDEPENDENCE_UNIT_ROW,
            policy_derivation_ref="calibration_analysis:w1",
            frozen_policy=_frozen_policy(),
            freeze_generation=20,
            heldout_evaluation_start_generation=10,
        )


def test_e5_policy_hash_substitution_fails_validation():
    """Replacing the frozen policy content after the fact breaks the bound hash."""
    calibration, heldout = _cal_heldout()
    integrity = build_experiment_integrity(
        experiment_id="exp:hash-sub",
        calibration_members=calibration,
        heldout_members=heldout,
        independence_unit=INDEPENDENCE_UNIT_ROW,
        policy_derivation_ref="calibration_analysis:w1",
        frozen_policy=_frozen_policy(),
        freeze_generation=10,
        heldout_evaluation_start_generation=20,
    )
    integrity["frozen_policy"]["policy"] = {"active_route": "nightshift", "route_version": 99}
    with pytest.raises(ValueError, match="EXPERIMENT_POLICY_HASH_MISMATCH"):
        validate_experiment_integrity(integrity)


def test_e6_insufficient_calibration_remains_uncalibrated():
    """Insufficient calibration keeps its explicit status and cannot be promoted."""
    calibration, heldout = _cal_heldout()
    integrity = build_experiment_integrity(
        experiment_id="exp:insufficient",
        calibration_members=calibration,
        heldout_members=heldout,
        independence_unit=INDEPENDENCE_UNIT_ROW,
        policy_derivation_ref="calibration_analysis:w1",
        frozen_policy=_frozen_policy(),
        freeze_generation=10,
        heldout_evaluation_start_generation=20,
        calibration_status=INSUFFICIENT_CALIBRATION,
        terminal_outcome=TERMINAL_STOP,
        insufficient_calibration_reasons=[
            "calibration_n_below_minimum",
            "heldout_truth_ambiguous",
        ],
    )
    validate_experiment_integrity(integrity)
    assert integrity["calibration_status"] == INSUFFICIENT_CALIBRATION
    assert integrity["terminal"]["outcome"] == TERMINAL_STOP

    ep = _recommendation_episode()
    off_arm = {"task_id": "task_1", "verifier_status": "failed", "receipt": "rec_off"}
    on_arm = {"task_id": "task_1", "verifier_status": "passed", "receipt": "rec_on"}
    with pytest.raises(
        ValueError, match="RECOMMENDATION_NON_POSITIVE_EXPERIMENT_CANNOT_BE_RECOMMENDED"
    ):
        build_learning_policy_recommendation(
            source_episodes=[ep],
            source_evidence_refs=["rec_A", "retrieval_receipt:g2", "physical_consumption:ollama"],
            source_revision="rev1",
            runtime_identity="local_model_executor",
            task_fingerprint="task_1",
            off_arm=off_arm,
            on_arm=on_arm,
            applicable_scope={"task_family": "record_serialization"},
            recommended_policy_delta={"episodic_memory_injection": {"enabled": True}},
            current_policy={"episodic_memory_injection": {"enabled": False}},
            expected_effect="Improve pass rate",
            rollback_target={
                "target_state": {"episodic_memory_injection": {"enabled": False}},
                "trigger": "regression",
            },
            experiment_integrity=integrity,
        )


def test_e7_reject_all_policy_represents_no_model_success():
    """Reject-all keeps model_success_claim False and never passes an integrity build."""
    calibration, heldout = _cal_heldout()
    with pytest.raises(ValueError, match="EXPERIMENT_REJECT_ALL_CLAIMS_MODEL_SUCCESS"):
        build_experiment_integrity(
            experiment_id="exp:reject-all",
            calibration_members=calibration,
            heldout_members=heldout,
            independence_unit=INDEPENDENCE_UNIT_ROW,
            policy_derivation_ref="calibration_analysis:w1",
            frozen_policy=_frozen_policy(),
            freeze_generation=10,
            heldout_evaluation_start_generation=20,
            calibration_status=FROZEN_REJECT_ALL,
            terminal_outcome=TERMINAL_STOP,
            insufficient_calibration_reasons=["zero_family_success"],
            reject_all_policy={
                "target_policy_delta": {"default_decision": "REJECT"},
                "model_success_claim": True,
            },
        )

    integrity = build_experiment_integrity(
        experiment_id="exp:reject-all",
        calibration_members=calibration,
        heldout_members=heldout,
        independence_unit=INDEPENDENCE_UNIT_ROW,
        policy_derivation_ref="calibration_analysis:w1",
        frozen_policy=_frozen_policy(),
        freeze_generation=10,
        heldout_evaluation_start_generation=20,
        calibration_status=FROZEN_REJECT_ALL,
        terminal_outcome=TERMINAL_STOP,
        insufficient_calibration_reasons=["zero_family_success"],
        reject_all_policy={
            "target_policy_delta": {"default_decision": "REJECT"},
            "model_success_claim": False,
        },
    )
    validate_experiment_integrity(integrity)
    assert integrity["reject_all_policy"]["model_success_claim"] is False


def test_e8_negative_terminal_stays_negative_through_recommendation_lifecycle():
    """A NEGATIVE terminal experiment is never promoted nor evaluated as adoptable."""
    calibration, heldout = _cal_heldout()
    integrity = build_experiment_integrity(
        experiment_id="exp:negative",
        calibration_members=calibration,
        heldout_members=heldout,
        independence_unit=INDEPENDENCE_UNIT_ROW,
        policy_derivation_ref="calibration_analysis:w1",
        frozen_policy=_frozen_policy(),
        freeze_generation=10,
        heldout_evaluation_start_generation=20,
        calibration_status=CALIBRATED,
        terminal_outcome=TERMINAL_NEGATIVE,
        negative_terminal=True,
    )
    validate_experiment_integrity(integrity)
    assert integrity["terminal"]["negative_terminal"] is True

    ep = _recommendation_episode()
    off_arm = {"task_id": "task_1", "verifier_status": "failed", "receipt": "rec_off"}
    on_arm = {"task_id": "task_1", "verifier_status": "passed", "receipt": "rec_on"}
    with pytest.raises(
        ValueError, match="RECOMMENDATION_NEGATIVE_TERMINAL_CANNOT_PROMOTE"
    ):
        build_learning_policy_recommendation(
            source_episodes=[ep],
            source_evidence_refs=["rec_A", "retrieval_receipt:g2", "physical_consumption:ollama"],
            source_revision="rev1",
            runtime_identity="local_model_executor",
            task_fingerprint="task_1",
            off_arm=off_arm,
            on_arm=on_arm,
            applicable_scope={"task_family": "record_serialization"},
            recommended_policy_delta={"episodic_memory_injection": {"enabled": True}},
            current_policy={"episodic_memory_injection": {"enabled": False}},
            expected_effect="Improve pass rate",
            rollback_target={
                "target_state": {"episodic_memory_injection": {"enabled": False}},
                "trigger": "regression",
            },
            experiment_integrity=integrity,
        )


def test_calibrated_pass_experiment_builds_validation_and_evaluates():
    """A calibrated, pre-frozen, non-overlapping PASS experiment is adoptable-considerable."""
    calibration, heldout = _cal_heldout()
    integrity = build_experiment_integrity(
        experiment_id="exp:semantic-reflex-v2-w1",
        calibration_members=calibration,
        heldout_members=heldout,
        independence_unit=INDEPENDENCE_UNIT_ROW,
        policy_derivation_ref="calibration_analysis:w1",
        frozen_policy=_frozen_policy(),
        freeze_generation=10,
        heldout_evaluation_start_generation=20,
        calibration_status=CALIBRATED,
        terminal_outcome=TERMINAL_PASS,
    )
    ep = _recommendation_episode()
    off_arm = {"task_id": "task_1", "verifier_status": "failed", "receipt": "rec_off"}
    on_arm = {"task_id": "task_1", "verifier_status": "passed", "receipt": "rec_on"}
    rec = build_learning_policy_recommendation(
        source_episodes=[ep],
        source_evidence_refs=["rec_A", "retrieval_receipt:g2", "physical_consumption:ollama"],
        source_revision="rev1",
        runtime_identity="local_model_executor",
        task_fingerprint="task_1",
        off_arm=off_arm,
        on_arm=on_arm,
        applicable_scope={"task_family": "record_serialization"},
        recommended_policy_delta={"episodic_memory_injection": {"enabled": True}},
        current_policy={"episodic_memory_injection": {"enabled": False}},
        expected_effect="Improve pass rate",
        rollback_target={
            "target_state": {"episodic_memory_injection": {"enabled": False}},
            "trigger": "regression",
        },
        experiment_integrity=integrity,
    )
    assert rec["observed_effect"] == "experiment_heldout_evaluation_bound"
    assert rec["effect_measurement"] == "frozen_policy_relative_heldout_quality"
    assert rec["experiment_integrity"]["schema"] == "nexus.learning_experiment_integrity.v1"

    validation = evaluate_learning_policy_recommendation(
        rec,
        validator_identity="validator:g6",
        current_workspace_revision="rev1",
        current_runtime_identity="local_model_executor",
    )
    assert validation["hostile_probes"]["experiment_integrity"] == "PASS"
    assert validation["validation_disposition"] == "VALIDATED_FOR_ADOPTION_CONSIDERATION"


def _cal_heldout() -> tuple[list[dict], list[dict]]:
    return [
        {"identity": "row:c1", "base_identity": "base:cal"},
        {"identity": "row:c2", "base_identity": "base:cal"},
    ], [
        {"identity": "row:h1", "base_identity": "base:heldout"},
        {"identity": "row:h2", "base_identity": "base:heldout"},
    ]


def _recommendation_episode() -> dict:
    return build_nexus_learning_episode(
        task_id="task_A",
        source="runtime_closure",
        terminal_outcome="SUCCESS",
        terminal_evidence={"verifier": "pytest", "receipt": "rec_A", "verifier_status": "passed"},
        qualification={
            "repeatability": True,
            "prevention_rule": "rule",
            "authority_qualification": True,
        },
        lesson_disposition="graduated",
        learning_write_succeeded=True,
    )

def test_zero_generation_freeze_is_valid_when_evaluation_starts_later():
    calibration, heldout = _cal_heldout()
    integrity = build_experiment_integrity(
        experiment_id="exp:zero-gen",
        calibration_members=calibration,
        heldout_members=heldout,
        independence_unit=INDEPENDENCE_UNIT_ROW,
        policy_derivation_ref="calibration_analysis:zero",
        frozen_policy=_frozen_policy(),
        freeze_generation=0,
        heldout_evaluation_start_generation=1,
        calibration_status=CALIBRATED,
        terminal_outcome=TERMINAL_PASS,
    )
    validate_experiment_integrity(integrity)


def test_heldout_member_truth_is_hash_bound():
    calibration, heldout = _cal_heldout()
    heldout[0]["truth"] = "ALLOW"
    integrity = build_experiment_integrity(
        experiment_id="exp:truth-bound",
        calibration_members=calibration,
        heldout_members=heldout,
        independence_unit=INDEPENDENCE_UNIT_ROW,
        policy_derivation_ref="calibration_analysis:truth",
        frozen_policy=_frozen_policy(),
        freeze_generation=1,
        heldout_evaluation_start_generation=2,
        calibration_status=CALIBRATED,
        terminal_outcome=TERMINAL_PASS,
    )
    integrity["heldout"]["members"][0]["truth"] = "BLOCK"
    with pytest.raises(ValueError, match="EXPERIMENT_HELDOUT_MEMBERS_HASH_MISMATCH"):
        validate_experiment_integrity(integrity)


@pytest.mark.parametrize("terminal_outcome", [TERMINAL_STOP, TERMINAL_DEFER])
def test_non_pass_terminal_cannot_be_promoted(terminal_outcome):
    calibration, heldout = _cal_heldout()
    integrity = build_experiment_integrity(
        experiment_id=f"exp:{terminal_outcome.lower()}",
        calibration_members=calibration,
        heldout_members=heldout,
        independence_unit=INDEPENDENCE_UNIT_ROW,
        policy_derivation_ref="calibration_analysis:terminal",
        frozen_policy=_frozen_policy(),
        freeze_generation=1,
        heldout_evaluation_start_generation=2,
        calibration_status=CALIBRATED,
        terminal_outcome=terminal_outcome,
    )
    ep = build_nexus_learning_episode(
        task_id="task-non-pass",
        source="runtime_closure",
        terminal_outcome="SUCCESS",
        terminal_evidence={
            "verifier": "pytest",
            "receipt": "rec-non-pass",
            "verifier_status": "passed",
        },
        qualification={
            "repeatability": True,
            "prevention_rule": "rule",
            "authority_qualification": True,
        },
        lesson_disposition="graduated",
        learning_write_succeeded=True,
    )
    with pytest.raises(
        ValueError, match="RECOMMENDATION_NON_POSITIVE_EXPERIMENT_CANNOT_BE_RECOMMENDED"
    ):
        build_learning_policy_recommendation(
            source_episodes=[ep],
            source_evidence_refs=[
                "rec-non-pass",
                "retrieval_receipt:g2",
                "physical_consumption:local",
            ],
            source_revision="rev-terminal",
            runtime_identity="local_model_executor",
            task_fingerprint="task-non-pass",
            off_arm={"task_id": "task-non-pass", "verifier_status": "failed", "receipt": "off"},
            on_arm={"task_id": "task-non-pass", "verifier_status": "passed", "receipt": "on"},
            applicable_scope={"task_family": "experiment"},
            recommended_policy_delta={"experiment_policy": {"enabled": True}},
            current_policy={"experiment_policy": {"enabled": False}},
            expected_effect="Only PASS experiments may be promoted",
            rollback_target={"target_state": {"experiment_policy": {"enabled": False}}},
            experiment_integrity=integrity,
        )


def test_non_mapping_population_member_fails_closed():
    calibration, heldout = _cal_heldout()
    calibration.append("not-a-member")
    with pytest.raises(ValueError, match="EXPERIMENT_POPULATION_MEMBER_INVALID"):
        build_experiment_integrity(
            experiment_id="exp:bad-member",
            calibration_members=calibration,
            heldout_members=heldout,
            independence_unit=INDEPENDENCE_UNIT_ROW,
            policy_derivation_ref="calibration_analysis:bad-member",
            frozen_policy=_frozen_policy(),
            freeze_generation=1,
            heldout_evaluation_start_generation=2,
        )


def test_policy_derivation_reference_is_cross_bound():
    calibration, heldout = _cal_heldout()
    integrity = build_experiment_integrity(
        experiment_id="exp:derivation-bound",
        calibration_members=calibration,
        heldout_members=heldout,
        independence_unit=INDEPENDENCE_UNIT_ROW,
        policy_derivation_ref="calibration_analysis:bound",
        frozen_policy=_frozen_policy(),
        freeze_generation=1,
        heldout_evaluation_start_generation=2,
        calibration_status=CALIBRATED,
        terminal_outcome=TERMINAL_PASS,
    )
    integrity["policy_derivation"]["policy_derivation_ref"] = "calibration_analysis:other"
    with pytest.raises(ValueError, match="EXPERIMENT_POLICY_DERIVATION_REF_MISMATCH"):
        validate_experiment_integrity(integrity)
