from __future__ import annotations

import copy

import pytest

from nexus_learning.experiment_integrity import (
    CALIBRATED,
    INDEPENDENCE_UNIT_ROW,
    build_experiment_integrity,
)
from nexus_learning.experiment_preflight import (
    ADMITTED,
    COMPARATOR_MATCHED,
    COMPARATOR_UNSUPPORTED,
    LEAKAGE_FAIL,
    LEAKAGE_NOT_EVALUATED,
    NOT_OBSERVED,
    PREFLIGHT_DEFER,
    PREFLIGHT_FAIL,
    PREFLIGHT_PASS,
    RECOVERY_NOT_OBSERVED,
    RECOVERY_YES,
    REFUSED,
    STAGE_CALIBRATION,
    STAGE_HARD_STOP,
    STAGE_HOLDOUT_COMPLETE,
    STAGE_HOLDOUT_ELIGIBLE,
    STAGE_SMOKE,
    build_cohort_preflight,
    build_comparator_preflight,
    build_resource_observation,
    build_staged_gate_evidence,
    compare_comparator_preflights,
)
from nexus_learning.necessary_condition_stop import (
    BOUND_UPPER,
    NEGATIVE_STOP,
    build_necessary_condition_stop_evidence,
)

HASH_A = "sha256:" + "a" * 64
HASH_B = "sha256:" + "b" * 64


def _comparator(**dimension_overrides):
    dimensions = {
        name: {"status": COMPARATOR_MATCHED, "value": {"frozen": True}, "evidence_hash": HASH_A}
        for name in (
            "sampling_fields", "reasoning_mode", "context_and_prefill_policy",
            "cache_semantics", "concurrency_and_admission_policy", "material_runtime_fields",
        )
    }
    dimensions.update(dimension_overrides)
    return build_comparator_preflight(
        comparator_identity="comparator:r1",
        dimension_observations=dimensions,
        sampling_request={"requested_fields": ["top_p"], "accepted_fields": ["top_p"]},
    )


def _resource(*, fail_at=None, recovery=RECOVERY_NOT_OBSERVED):
    rows = []
    for level in [1, 2, 4, 8]:
        failure = level == fail_at
        rows.append({
            "requested_concurrency": level,
            "warmup_policy": "one-warmup-batch",
            "measured_batch_count": 2,
            "headroom_state": "EXHAUSTED" if failure else "AVAILABLE",
            "headroom_observation": None if failure else {"free_memory_gb": 12},
            "admission_state": REFUSED if failure else ADMITTED,
            "error_count": 1 if failure else 0,
            "recovery_after_batch": recovery if failure else RECOVERY_NOT_OBSERVED,
        })
        if failure:
            break
    return build_resource_observation(
        planned_concurrency_levels=[1, 2, 4, 8],
        observations=rows,
        stop_on_resource_failure=True,
    )


def _cases(labels=("yes", "no", "yes", "no")):
    return [
        {"case_id": "c1", "split": "CALIBRATION", "source_group_id": "g1", "outcome": labels[0]},
        {"case_id": "c2", "split": "CALIBRATION", "source_group_id": "g2", "outcome": labels[1]},
        {"case_id": "h1", "split": "HOLDOUT", "source_group_id": "g3", "outcome": labels[2]},
        {"case_id": "h2", "split": "HOLDOUT", "source_group_id": "g4", "outcome": labels[3]},
    ]


def _gates():
    return [
        {"gate_id": "smoke:no-errors", "phase": "SMOKE", "metric": "error_count",
         "comparator": "LTE", "threshold": 0, "kind": "SAFETY", "hard": True},
        {"gate_id": "calibration:quality", "phase": "CALIBRATION", "metric": "quality",
         "comparator": "GTE", "threshold": 0.8, "kind": "CORRECTNESS", "hard": True},
        {"gate_id": "holdout:quality", "phase": "HOLDOUT", "metric": "quality",
         "comparator": "GTE", "threshold": 0.8, "kind": "QUALITY", "hard": True},
        {"gate_id": "holdout:latency", "phase": "HOLDOUT", "metric": "latency_ms",
         "comparator": "LTE", "threshold": 100, "kind": "PERFORMANCE", "hard": True},
    ]


def _preflight(**overrides):
    args = {
        "experiment_id": "exp:one",
        "cases": _cases(),
        "independence_unit": INDEPENDENCE_UNIT_ROW,
        "value_claim": True,
        "ground_truth_provenance": {"source": "reviewed-labels:v1", "artifact_sha256": HASH_A},
        "leakage_observation": {"status": "PASS", "evaluator": "prompt-audit:v1", "evidence_hash": HASH_B},
        "incumbent": {"identity": "baseline:v1", "revision": "r1"},
        "comparator_preflight": _comparator(),
        "resource_observation": _resource(),
        "resource_stop_policy": {"stop_on_resource_failure": True},
        "hard_gates": _gates(),
    }
    args.update(overrides)
    if "experiment_integrity" not in overrides:
        try:
            args["experiment_integrity"] = build_experiment_integrity(
                experiment_id=args["experiment_id"],
                calibration_members=[{"identity": row["case_id"], "base_identity": row["source_group_id"]}
                                     for row in args["cases"] if row["split"] == "CALIBRATION"],
                heldout_members=[{"identity": row["case_id"], "base_identity": row["source_group_id"]}
                                 for row in args["cases"] if row["split"] == "HOLDOUT"],
                independence_unit=args["independence_unit"],
                policy_derivation_ref="calibration-only",
                frozen_policy={"staged_hard_gates": args["hard_gates"]},
                freeze_generation=1,
                heldout_evaluation_start_generation=2,
                calibration_status=CALIBRATED,
            )
        except ValueError:
            args["experiment_integrity"] = None
    return build_cohort_preflight(**args)


def _stage(preflight, *, previous=None, observation=None, **extra):
    return build_staged_gate_evidence(
        preflight=preflight, previous_evidence=previous,
        stage_observation=observation, **extra,
    )


def test_duplicate_case_identity_fails_preflight():
    cases = _cases()
    cases[3]["case_id"] = "c1"
    result = _preflight(cases=cases)
    assert result["preflight_disposition"] == PREFLIGHT_FAIL
    assert "c1" in result["cohort"]["duplicate_case_ids"]


def test_source_group_overlap_fails_even_when_case_ids_differ():
    cases = _cases()
    cases[-1]["source_group_id"] = "g1"
    result = _preflight(cases=cases)
    assert result["preflight_disposition"] == PREFLIGHT_FAIL
    assert result["cohort"]["source_group_overlap"] == ["g1"]


def test_constant_answer_baseline_blocks_value_claim_but_mixed_cohort_does_not():
    constant = _preflight(cases=_cases(labels=("yes", "yes", "yes", "yes")))
    assert constant["preflight_disposition"] == PREFLIGHT_DEFER
    assert constant["outcome_distribution"]["constant_answer_accuracy"] == 1.0
    assert constant["value_claim_eligible"] is False

    mixed = _preflight()
    assert mixed["preflight_disposition"] == PREFLIGHT_PASS
    assert mixed["outcome_distribution"]["constant_outcome"] is False


def test_missing_ground_truth_provenance_is_explicit_defer():
    result = _preflight(ground_truth_provenance=None)
    assert result["preflight_disposition"] == PREFLIGHT_DEFER
    assert result["ground_truth_provenance"]["status"] == "MISSING"
    assert "ground_truth_provenance_missing" in result["defer_reasons"]


def test_missing_issue_20_integrity_binding_cannot_open_holdout():
    result = _preflight(experiment_integrity=None)
    assert result["preflight_disposition"] == PREFLIGHT_DEFER
    assert result["experiment_integrity_binding_hash"] is None
    assert "experiment_integrity_binding_missing" in result["defer_reasons"]


def test_leakage_failure_blocks_and_missing_leakage_is_not_silently_passed():
    failed = _preflight(leakage_observation={
        "status": LEAKAGE_FAIL, "evaluator": "prompt-audit:v1", "evidence_hash": HASH_A,
    })
    assert failed["preflight_disposition"] == PREFLIGHT_FAIL

    missing = _preflight(leakage_observation=None)
    assert missing["preflight_disposition"] == PREFLIGHT_DEFER
    assert missing["leakage_observation"]["status"] == LEAKAGE_NOT_EVALUATED


def test_missing_incumbent_blocks_value_comparison():
    result = _preflight(incumbent=None)
    assert result["preflight_disposition"] == PREFLIGHT_DEFER
    assert result["value_claim_eligible"] is False


def test_sampling_field_rejection_requires_frozen_compatibility_amendment():
    dimensions = {name: {"status": COMPARATOR_MATCHED, "value": {"frozen": True}, "evidence_hash": HASH_A}
                  for name in (
                      "sampling_fields", "reasoning_mode", "context_and_prefill_policy",
                      "cache_semantics", "concurrency_and_admission_policy", "material_runtime_fields",
                  )}
    rejected = build_comparator_preflight(
        comparator_identity="cmp:r1", dimension_observations=dimensions,
        sampling_request={"requested_fields": ["top_k", "top_p"], "accepted_fields": ["top_p"]},
    )
    assert rejected["formal_comparable"] is False
    amended = build_comparator_preflight(
        comparator_identity="cmp:r1", dimension_observations=dimensions,
        sampling_request={"requested_fields": ["top_k", "top_p"], "accepted_fields": ["top_p"]},
        compatibility_amendment={
            "amendment_id": "sampling-adjustment-1", "sha256": HASH_B,
            "frozen_before_formal_run": True, "amended_request_fields": ["top_p"],
        },
    )
    assert amended["formal_comparable"] is True
    assert amended["sampling_request"]["compatibility_amendment"]["sha256"] == HASH_B


def test_comparator_settings_and_cache_semantics_remain_explicit():
    unsupported = _comparator(cache_semantics={"status": COMPARATOR_UNSUPPORTED, "reason": "prefix cache cannot be disabled"})
    assert unsupported["formal_comparable"] is False
    assert unsupported["comparison_scope"] == "WHOLE_RUNTIME_STACK_ONLY"
    preflight = _preflight(comparator_preflight=unsupported)
    assert preflight["preflight_disposition"] == PREFLIGHT_DEFER


def test_auto_selected_prefill_changes_make_comparator_receipts_non_comparable():
    first = _comparator(context_and_prefill_policy={
        "status": COMPARATOR_MATCHED, "value": {"prefill_chunk": 512}, "evidence_hash": HASH_A,
    })
    second = _comparator(context_and_prefill_policy={
        "status": COMPARATOR_MATCHED, "value": {"prefill_chunk": 1024}, "evidence_hash": HASH_B,
    })
    pair = compare_comparator_preflights(first, second)
    assert pair["evidence_mixing_allowed"] is False
    assert "context_and_prefill_policy" in pair["changed_dimensions"]
    assert pair["comparison_scope"] == "NOT_COMPARABLE"


def test_resource_failure_preserves_receipt_and_skips_higher_concurrency():
    result = _resource(fail_at=2, recovery=RECOVERY_YES)
    assert result["first_failure_concurrency"] == 2
    assert result["skipped_concurrency_levels"] == [4, 8]
    assert result["observations"][-1]["admission_state"] == REFUSED
    assert result["observations"][-1]["recovery_after_batch"] == RECOVERY_YES
    with pytest.raises(ValueError, match="LEVEL_AFTER_FROZEN_STOP"):
        build_resource_observation(
            planned_concurrency_levels=[1, 2, 4],
            observations=[
                {"requested_concurrency": 2, "warmup_policy": "fixed", "measured_batch_count": 1,
                 "headroom_state": "EXHAUSTED", "headroom_observation": None,
                 "admission_state": REFUSED, "error_count": 1, "recovery_after_batch": RECOVERY_NOT_OBSERVED},
                {"requested_concurrency": 4, "warmup_policy": "fixed", "measured_batch_count": 1,
                 "headroom_state": "AVAILABLE", "headroom_observation": {},
                 "admission_state": ADMITTED, "error_count": 0, "recovery_after_batch": RECOVERY_NOT_OBSERVED},
            ],
            stop_on_resource_failure=True,
        )


def test_missing_resource_telemetry_is_explicitly_deferred_unless_inapplicable():
    resource = build_resource_observation(
        planned_concurrency_levels=[1],
        observations=[{"requested_concurrency": 1, "warmup_policy": "fixed", "measured_batch_count": 1,
                       "headroom_state": "NOT_OBSERVED", "headroom_observation": None,
                       "admission_state": NOT_OBSERVED, "error_count": 0,
                       "recovery_after_batch": RECOVERY_NOT_OBSERVED}],
        stop_on_resource_failure=True,
    )
    deferred = _preflight(resource_observation=resource)
    assert deferred["preflight_disposition"] == PREFLIGHT_DEFER
    assert "resource_headroom_telemetry_incomplete" in deferred["defer_reasons"]
    simulation = _preflight(
        resource_observation=resource,
        resource_stop_policy={"stop_on_resource_failure": True, "applicability": "NOT_APPLICABLE",
                              "inapplicability_reason": "simulation-only deterministic harness"},
    )
    assert simulation["preflight_disposition"] == PREFLIGHT_PASS


def test_calibration_hard_failure_stops_before_holdout_and_is_sticky():
    preflight = _preflight()
    state = _stage(preflight)
    assert state["stage"] == STAGE_SMOKE
    state = _stage(preflight, previous=state, observation={
        "observation_id": "smoke-1", "status": "PASS", "evidence_hash": HASH_A,
        "metrics": {"error_count": 0},
    })
    assert state["stage"] == STAGE_CALIBRATION
    state = _stage(preflight, previous=state, observation={
        "observation_id": "cal-1", "status": "PASS", "evidence_hash": HASH_B,
        "metrics": {"quality": 0.4, "latency_ms": 10},
    })
    assert state["stage"] == STAGE_HARD_STOP
    assert state["disposition"] == "STOP"
    assert state["heldout_opened"] is False
    assert state["heldout_preserved_sealed"] is True
    assert state["history"][-1]["performance_is_diagnostic_only"] is True
    old = copy.deepcopy(state)
    later_positive = _stage(preflight, previous=state, observation={
        "observation_id": "later", "status": "PASS", "evidence_hash": HASH_A,
        "metrics": {"quality": 1.0, "latency_ms": 1},
    })
    assert later_positive == old


def test_calibration_passes_before_holdout_can_become_eligible_or_complete():
    preflight = _preflight()
    state = _stage(preflight)
    state = _stage(preflight, previous=state, observation={
        "observation_id": "smoke-1", "status": "PASS", "evidence_hash": HASH_A,
        "metrics": {"error_count": 0},
    })
    state = _stage(preflight, previous=state, observation={
        "observation_id": "cal-1", "status": "PASS", "evidence_hash": HASH_B,
        "metrics": {"quality": 0.9},
    })
    assert state["stage"] == STAGE_HOLDOUT_ELIGIBLE
    assert state["heldout_opened"] is False
    state = _stage(preflight, previous=state, observation={
        "observation_id": "holdout-1", "status": "PASS", "evidence_hash": HASH_A,
        "metrics": {"quality": 0.9, "latency_ms": 90},
    })
    assert state["stage"] == STAGE_HOLDOUT_COMPLETE
    assert state["heldout_opened"] is True


def test_necessary_condition_stop_reuses_stronger_issue_26_proof():
    cases = _cases()
    integrity = build_experiment_integrity(
        experiment_id="exp:one",
        calibration_members=[{"identity": row["case_id"], "base_identity": row["source_group_id"]}
                             for row in cases if row["split"] == "CALIBRATION"],
        heldout_members=[{"identity": row["case_id"], "base_identity": row["source_group_id"]}
                         for row in cases if row["split"] == "HOLDOUT"],
        independence_unit=INDEPENDENCE_UNIT_ROW,
        policy_derivation_ref="calibration-only",
        frozen_policy={"necessary_conditions": [{"condition_id": "quality", "metric": "quality",
                                                   "comparator": "GTE", "threshold": 0.8}]},
        freeze_generation=1,
        heldout_evaluation_start_generation=2,
        calibration_status=CALIBRATED,
    )
    preflight = _preflight(cases=cases, experiment_integrity=integrity)
    stop = build_necessary_condition_stop_evidence(
        experiment_integrity=integrity,
        necessary_condition_id="quality",
        observed_value_or_bound=0.4,
        observed_bound_kind=BOUND_UPPER,
        remaining_unknowns=["quality_tail"],
        heldout_opened=False,
    )
    assert stop["terminal_disposition"] == NEGATIVE_STOP
    result = build_staged_gate_evidence(
        preflight=preflight,
        necessary_condition_stop_evidence=stop,
        experiment_integrity=integrity,
    )
    assert result["stage"] == STAGE_HARD_STOP
    assert result["heldout_preserved_sealed"] is True
