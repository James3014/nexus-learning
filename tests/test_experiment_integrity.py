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
    EFFECT_OUTCOME_UNKNOWN,
    EFFECT_SUCCEEDED,
    FROZEN_REJECT_ALL,
    INDEPENDENCE_UNIT_BASE,
    INDEPENDENCE_UNIT_ROW,
    INSUFFICIENT_CALIBRATION,
    LOCAL_FAKE_PROVIDER,
    MOCK_TRANSPORT,
    PHYSICAL_LOCAL_MODEL,
    REMOTE_PROVIDER_OBSERVED,
    SIMULATION_ONLY,
    TERMINAL_DEFER,
    TERMINAL_NEGATIVE,
    TERMINAL_PASS,
    TERMINAL_STOP,
    UNKNOWN_ORIGIN,
    build_evidence_origin_provenance,
    build_experiment_integrity,
    require_remote_provider_observation,
    validate_evidence_origin_provenance,
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


def _physical_observation_receipt(
    *,
    transport_class: str = "REMOTE_PROVIDER",
    outcome: str = EFFECT_SUCCEEDED,
    provider: str | None = "jev",
    model: str = "jev-latest",
    revision: str | None = "jev-physical-r1",
) -> dict:
    return {
        "receipt_id": "receipt:provider:1",
        "effect_identity": "effect:provider:1",
        "operation_id": "operation:provider:1",
        "transport_class": transport_class,
        "external_effect_started": True,
        "outcome": outcome,
        "observed_provider": provider,
        "observed_model": model,
        "observed_revision": revision,
        "source_receipt_ref": "provider-journal:entry:1",
        "source_receipt_sha256": "sha256:" + ("a" * 64),
    }


def test_configured_jev_identity_does_not_self_prove_observed_execution():
    provenance = build_evidence_origin_provenance(
        origin_class=UNKNOWN_ORIGIN,
        requested_provider="jev",
        requested_model="jev-latest",
        configured_provider="jev",
        configured_model="jev-latest",
    )
    assert provenance["configured_identity"]["model"] == "jev-latest"
    assert provenance["observed_identity"]["model"] is None
    with pytest.raises(ValueError, match="LIVE_PROVIDER_REMOTE_OBSERVATION_REQUIRED"):
        require_remote_provider_observation(provenance)


def test_localhost_mock_self_declared_model_cannot_satisfy_remote_live_gate():
    provenance = build_evidence_origin_provenance(
        origin_class=MOCK_TRANSPORT,
        requested_provider="jev",
        configured_provider="jev",
        configured_model="jev-latest",
        observation_receipt={
            "transport": "http://127.0.0.1:8123",
            "response": {"model": "jev-latest"},
        },
    )
    validate_evidence_origin_provenance(provenance)
    with pytest.raises(ValueError, match="LIVE_PROVIDER_REMOTE_OBSERVATION_REQUIRED"):
        require_remote_provider_observation(provenance)


def test_simulation_and_local_fake_provider_remain_nonphysical():
    for origin in (SIMULATION_ONLY, LOCAL_FAKE_PROVIDER):
        provenance = build_evidence_origin_provenance(
            origin_class=origin,
            configured_model="jev-latest",
        )
        assert provenance["external_effect"]["started"] is False
        with pytest.raises(ValueError, match="LIVE_PROVIDER_REMOTE_OBSERVATION_REQUIRED"):
            require_remote_provider_observation(provenance)


def test_physical_local_model_is_distinct_from_remote_provider_observation():
    receipt = _physical_observation_receipt(
        transport_class="LOCAL_MODEL",
        provider=None,
        model="mlx-qwen-local",
        revision=None,
    )
    provenance = build_evidence_origin_provenance(
        origin_class=PHYSICAL_LOCAL_MODEL,
        configured_model="mlx-qwen-local",
        observed_model="mlx-qwen-local",
        external_effect_started=True,
        effect_outcome=EFFECT_SUCCEEDED,
        effect_identity=receipt["effect_identity"],
        operation_id=receipt["operation_id"],
        observation_receipt=receipt,
    )
    validate_evidence_origin_provenance(provenance)
    with pytest.raises(ValueError, match="LIVE_PROVIDER_REMOTE_OBSERVATION_REQUIRED"):
        require_remote_provider_observation(provenance)


def test_physical_remote_provider_with_bound_receipt_satisfies_live_gate():
    receipt = _physical_observation_receipt()
    provenance = build_evidence_origin_provenance(
        origin_class=REMOTE_PROVIDER_OBSERVED,
        requested_provider="jev",
        requested_model="jev-latest",
        configured_provider="jev",
        configured_model="jev-latest",
        observed_provider="jev",
        observed_model="jev-latest",
        observed_revision="jev-physical-r1",
        external_effect_started=True,
        effect_outcome=EFFECT_SUCCEEDED,
        effect_identity=receipt["effect_identity"],
        operation_id=receipt["operation_id"],
        observation_receipt=receipt,
    )
    proven = require_remote_provider_observation(provenance)
    assert proven["origin_class"] == REMOTE_PROVIDER_OBSERVED
    assert proven["observed_identity"]["model"] == "jev-latest"


def test_remote_observed_model_missing_is_not_filled_from_configured_model():
    receipt = _physical_observation_receipt()
    with pytest.raises(ValueError, match="EVIDENCE_ORIGIN_REMOTE_OBSERVED_IDENTITY_MISSING"):
        build_evidence_origin_provenance(
            origin_class=REMOTE_PROVIDER_OBSERVED,
            configured_provider="jev",
            configured_model="jev-latest",
            observed_provider="jev",
            observed_model=None,
            observed_revision="jev-physical-r1",
            external_effect_started=True,
            effect_outcome=EFFECT_SUCCEEDED,
            effect_identity=receipt["effect_identity"],
            operation_id=receipt["operation_id"],
            observation_receipt=receipt,
        )


def test_remote_outcome_unknown_cannot_be_promoted_to_successful_live_observation():
    receipt = _physical_observation_receipt(outcome=EFFECT_OUTCOME_UNKNOWN)
    provenance = build_evidence_origin_provenance(
        origin_class=REMOTE_PROVIDER_OBSERVED,
        configured_provider="jev",
        configured_model="jev-latest",
        observed_provider="jev",
        observed_model="jev-latest",
        observed_revision="jev-physical-r1",
        external_effect_started=True,
        effect_outcome=EFFECT_OUTCOME_UNKNOWN,
        effect_identity=receipt["effect_identity"],
        operation_id=receipt["operation_id"],
        observation_receipt=receipt,
    )
    validate_evidence_origin_provenance(provenance)
    with pytest.raises(ValueError, match="LIVE_PROVIDER_SUCCESSFUL_OUTCOME_REQUIRED"):
        require_remote_provider_observation(provenance)


def test_tampered_physical_receipt_fails_closed():
    receipt = _physical_observation_receipt()
    provenance = build_evidence_origin_provenance(
        origin_class=REMOTE_PROVIDER_OBSERVED,
        observed_provider="jev",
        observed_model="jev-latest",
        observed_revision="jev-physical-r1",
        external_effect_started=True,
        effect_outcome=EFFECT_SUCCEEDED,
        effect_identity=receipt["effect_identity"],
        operation_id=receipt["operation_id"],
        observation_receipt=receipt,
    )
    provenance["observation_receipt"]["payload"]["operation_id"] = "operation:substituted"
    with pytest.raises(ValueError, match="EVIDENCE_ORIGIN_RECEIPT_HASH_MISMATCH"):
        validate_evidence_origin_provenance(provenance)


def test_experiment_can_bind_remote_provenance_without_breaking_legacy_artifacts():
    calibration, heldout = _cal_heldout()
    legacy = build_experiment_integrity(
        experiment_id="exp:legacy-no-origin",
        calibration_members=calibration,
        heldout_members=heldout,
        independence_unit=INDEPENDENCE_UNIT_ROW,
        policy_derivation_ref="calibration_analysis:legacy",
        frozen_policy=_frozen_policy(),
        freeze_generation=1,
        heldout_evaluation_start_generation=2,
    )
    assert "evidence_origin" not in legacy
    validate_experiment_integrity(legacy)
    with pytest.raises(ValueError, match="LIVE_PROVIDER_PROVENANCE_MISSING"):
        require_remote_provider_observation(legacy)

    receipt = _physical_observation_receipt()
    provenance = build_evidence_origin_provenance(
        origin_class=REMOTE_PROVIDER_OBSERVED,
        observed_provider="jev",
        observed_model="jev-latest",
        observed_revision="jev-physical-r1",
        external_effect_started=True,
        effect_outcome=EFFECT_SUCCEEDED,
        effect_identity=receipt["effect_identity"],
        operation_id=receipt["operation_id"],
        observation_receipt=receipt,
    )
    bound = build_experiment_integrity(
        experiment_id="exp:remote-origin-bound",
        calibration_members=calibration,
        heldout_members=heldout,
        independence_unit=INDEPENDENCE_UNIT_ROW,
        policy_derivation_ref="calibration_analysis:origin",
        frozen_policy=_frozen_policy(),
        freeze_generation=1,
        heldout_evaluation_start_generation=2,
        evidence_origin=provenance,
    )
    assert require_remote_provider_observation(bound)["binding_hash"] == provenance["binding_hash"]
