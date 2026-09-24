"""Acceptance tests for non-rescuable necessary-condition early stop (#26)."""

from __future__ import annotations

import pytest

from nexus_learning.effectiveness_measurement import compare_workflows_at_required_quality
from nexus_learning.experiment_integrity import (
    CALIBRATED,
    INDEPENDENCE_UNIT_ROW,
    TERMINAL_DEFER,
    build_experiment_integrity,
)
from nexus_learning.necessary_condition_stop import (
    BOUND_EXACT,
    BOUND_LOWER,
    BOUND_UNKNOWN,
    BOUND_UPPER,
    CONTINUE,
    NECESSARY_CONDITION_STOP_CLAIM_CEILING,
    NECESSARY_CONDITION_STOP_SCHEMA,
    NEGATIVE_STOP,
    NecessaryConditionStopError,
    build_necessary_condition_stop_evidence,
    validate_necessary_condition_stop_evidence,
)


def _integrity() -> dict:
    return build_experiment_integrity(
        experiment_id="context-economy-v1",
        calibration_members=[
            {"identity": "cal:1", "base_identity": "base:cal"},
            {"identity": "cal:2", "base_identity": "base:cal"},
        ],
        heldout_members=[
            {"identity": "hold:1", "base_identity": "base:hold"},
            {"identity": "hold:2", "base_identity": "base:hold"},
        ],
        independence_unit=INDEPENDENCE_UNIT_ROW,
        policy_derivation_ref="calibration:context-economy-v1",
        frozen_policy={
            "selector": "frozen-policy-a2",
            "necessary_conditions": [
                {
                    "condition_id": "incremental-economy-min",
                    "metric": "incremental_visible_token_reduction",
                    "comparator": "GTE",
                    "threshold": 0.10,
                },
                {
                    "condition_id": "provider-cost-max",
                    "metric": "provider_cost_usd",
                    "comparator": "LTE",
                    "threshold": 1.0,
                },
            ],
        },
        freeze_generation=10,
        heldout_evaluation_start_generation=20,
        calibration_status=CALIBRATED,
        terminal_outcome=TERMINAL_DEFER,
    )


def test_non_rescuable_upper_bound_stops_without_opening_holdout():
    integrity = _integrity()
    evidence = build_necessary_condition_stop_evidence(
        experiment_integrity=integrity,
        necessary_condition_id="incremental-economy-min",
        observed_value_or_bound=0.06,
        observed_bound_kind=BOUND_UPPER,
        remaining_unknowns=("quality_score", "provider_latency"),
        heldout_opened=False,
    )

    assert evidence["schema"] == NECESSARY_CONDITION_STOP_SCHEMA
    assert evidence["claim_ceiling"] == NECESSARY_CONDITION_STOP_CLAIM_CEILING
    assert evidence["terminal_disposition"] == NEGATIVE_STOP
    assert evidence["rescuable"] is False
    assert evidence["remaining_unknowns_cannot_rescue_condition"] is True
    assert evidence["heldout_opened"] is False
    assert evidence["heldout_preserved_sealed"] is True
    assert evidence["quality_qualified_claim"] is False
    assert evidence["adoption_recommendation"] is None
    assert evidence["authority_effect"] is False


def test_rescuable_upper_bound_continues():
    evidence = build_necessary_condition_stop_evidence(
        experiment_integrity=_integrity(),
        necessary_condition_id="incremental-economy-min",
        observed_value_or_bound=0.12,
        observed_bound_kind=BOUND_UPPER,
        remaining_unknowns=("quality_score",),
    )

    assert evidence["terminal_disposition"] == CONTINUE
    assert evidence["rescuable"] is True
    assert evidence["remaining_unknowns_cannot_rescue_condition"] is False


def test_missing_evidence_remains_explicit_and_cannot_become_negative_stop():
    evidence = build_necessary_condition_stop_evidence(
        experiment_integrity=_integrity(),
        necessary_condition_id="incremental-economy-min",
        observed_value_or_bound=None,
        observed_bound_kind=BOUND_UNKNOWN,
        remaining_unknowns=("incremental_visible_token_reduction",),
        missing_evidence_reasons=("incremental_visible_token_reduction:not_measured",),
    )

    assert evidence["terminal_disposition"] == CONTINUE
    assert evidence["rescuable"] is True
    assert evidence["observation"]["observed_value_or_bound"] is None
    assert evidence["observation"]["missing_evidence_reasons"]


def test_exact_current_value_with_remaining_unknowns_is_not_terminal():
    evidence = build_necessary_condition_stop_evidence(
        experiment_integrity=_integrity(),
        necessary_condition_id="incremental-economy-min",
        observed_value_or_bound=0.02,
        observed_bound_kind=BOUND_EXACT,
        remaining_unknowns=("remaining_selection_cases",),
    )

    assert evidence["terminal_disposition"] == CONTINUE
    assert evidence["rescuable"] is True


def test_lte_condition_uses_lower_bound_to_prove_non_rescuable_failure():
    evidence = build_necessary_condition_stop_evidence(
        experiment_integrity=_integrity(),
        necessary_condition_id="provider-cost-max",
        observed_value_or_bound=1.25,
        observed_bound_kind=BOUND_LOWER,
        remaining_unknowns=("final_token_usage",),
        heldout_opened=False,
    )

    assert evidence["terminal_disposition"] == NEGATIVE_STOP
    assert evidence["necessary_condition"]["comparator"] == "LTE"
    assert evidence["necessary_condition"]["threshold"] == 1.0


def test_negative_stop_cannot_be_tampered_into_quality_or_adoption_success():
    integrity = _integrity()
    evidence = build_necessary_condition_stop_evidence(
        experiment_integrity=integrity,
        necessary_condition_id="incremental-economy-min",
        observed_value_or_bound=0.05,
        observed_bound_kind=BOUND_UPPER,
    )
    evidence["quality_qualified_claim"] = True

    with pytest.raises(
        NecessaryConditionStopError,
        match="cannot claim QUALITY_QUALIFIED",
    ):
        validate_necessary_condition_stop_evidence(
            evidence,
            experiment_integrity=integrity,
        )


def test_frozen_threshold_substitution_after_freeze_fails_closed():
    integrity = _integrity()
    integrity["frozen_policy"]["policy"]["necessary_conditions"][0]["threshold"] = 0.01

    with pytest.raises(ValueError, match="EXPERIMENT_POLICY_HASH_MISMATCH"):
        build_necessary_condition_stop_evidence(
            experiment_integrity=integrity,
            necessary_condition_id="incremental-economy-min",
            observed_value_or_bound=0.06,
            observed_bound_kind=BOUND_UPPER,
        )


def test_stop_evidence_condition_substitution_fails_closed():
    integrity = _integrity()
    evidence = build_necessary_condition_stop_evidence(
        experiment_integrity=integrity,
        necessary_condition_id="incremental-economy-min",
        observed_value_or_bound=0.05,
        observed_bound_kind=BOUND_UPPER,
    )
    evidence["necessary_condition"]["threshold"] = 0.01

    with pytest.raises(NecessaryConditionStopError, match="threshold mismatch"):
        validate_necessary_condition_stop_evidence(
            evidence,
            experiment_integrity=integrity,
        )


def test_unknown_observation_requires_explicit_missingness():
    with pytest.raises(
        NecessaryConditionStopError,
        match="explicit missing_evidence_reasons",
    ):
        build_necessary_condition_stop_evidence(
            experiment_integrity=_integrity(),
            necessary_condition_id="incremental-economy-min",
            observed_value_or_bound=None,
            observed_bound_kind=BOUND_UNKNOWN,
        )


def test_negative_stop_after_holdout_open_does_not_claim_sealed_holdout():
    evidence = build_necessary_condition_stop_evidence(
        experiment_integrity=_integrity(),
        necessary_condition_id="incremental-economy-min",
        observed_value_or_bound=0.05,
        observed_bound_kind=BOUND_UPPER,
        heldout_opened=True,
    )

    assert evidence["terminal_disposition"] == NEGATIVE_STOP
    assert evidence["heldout_opened"] is True
    assert evidence["heldout_preserved_sealed"] is False


def test_quality_qualified_positive_economics_remains_owned_by_existing_contract():
    rows = [
        {
            "workflow_identity": "baseline",
            "workflow_revision": "r1",
            "task_fingerprint": "task-a",
            "attempt_count": 10,
            "qualified_success_count": 10,
            "critical_failure_count": 0,
            "semantic_failure_count": 0,
            "provider_failure_count": 0,
            "false_allow_count": 0,
            "model_invocation_count": 10,
            "provider_invocation_count": 0,
            "fallback_count": 0,
            "token_usage": 100,
            "wall_time_seconds": 10.0,
            "monetary_cost_usd": 1.0,
            "human_intervention_count": 0,
            "missingness_reasons": [],
            "ineligibility_reasons": [],
        },
        {
            "workflow_identity": "cheaper",
            "workflow_revision": "r1",
            "task_fingerprint": "task-a",
            "attempt_count": 10,
            "qualified_success_count": 10,
            "critical_failure_count": 0,
            "semantic_failure_count": 0,
            "provider_failure_count": 0,
            "false_allow_count": 0,
            "model_invocation_count": 5,
            "provider_invocation_count": 0,
            "fallback_count": 0,
            "token_usage": 50,
            "wall_time_seconds": 5.0,
            "monetary_cost_usd": 0.5,
            "human_intervention_count": 0,
            "missingness_reasons": [],
            "ineligibility_reasons": [],
        },
    ]
    result = compare_workflows_at_required_quality(
        rows,
        required_quality_floor=0.9,
        critical_failure_ceiling=0,
        baseline_workflow="baseline",
        baseline_workflow_revision="r1",
    )
    cheaper = next(
        row for row in result["rows"] if row["workflow_identity"] == "cheaper"
    )

    assert cheaper["gate_status"] == "QUALITY_QUALIFIED"
    assert "COST_COMPARABLE" in cheaper["dispositions"]


def test_binding_hash_tamper_fails_closed():
    integrity = _integrity()
    evidence = build_necessary_condition_stop_evidence(
        experiment_integrity=integrity,
        necessary_condition_id="incremental-economy-min",
        observed_value_or_bound=0.05,
        observed_bound_kind=BOUND_UPPER,
    )
    evidence["binding_hash"] = "0" * 64

    with pytest.raises(NecessaryConditionStopError, match="binding hash mismatch"):
        validate_necessary_condition_stop_evidence(
            evidence,
            experiment_integrity=integrity,
        )


def test_missing_pre_registered_condition_fails_closed():
    integrity = _integrity()
    with pytest.raises(
        NecessaryConditionStopError,
        match="uniquely pre-registered",
    ):
        build_necessary_condition_stop_evidence(
            experiment_integrity=integrity,
            necessary_condition_id="not-registered",
            observed_value_or_bound=0.0,
            observed_bound_kind=BOUND_UPPER,
        )
