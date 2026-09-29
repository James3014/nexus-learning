from __future__ import annotations

import copy
import hashlib
import json

import pytest

from nexus_learning.campaign_closeout import (
    CAMPAIGN_COMPLETE,
    CAMPAIGN_INCOMPLETE,
    NO_GATE_DEFINED,
    build_campaign_closeout,
    evaluate_reopen_trigger,
    validate_campaign_closeout,
)
from nexus_learning.experiment_integrity import (
    CALIBRATED,
    INDEPENDENCE_UNIT_ROW,
    build_experiment_integrity,
)
from nexus_learning.experiment_preflight import (
    COMPARATOR_MATCHED,
    build_cohort_preflight,
    build_comparator_preflight,
    build_resource_observation,
    build_staged_gate_evidence,
)
from nexus_learning.experiment_run_identity import (
    COMPLETE,
    NOT_STARTED,
    build_experiment_run_identity,
)

HASH_A = "sha256:" + "a" * 64
HASH_B = "sha256:" + "b" * 64


def _preflight(experiment_id, **overrides):
    dimensions = {
        name: {"status": COMPARATOR_MATCHED, "value": {"frozen": True}, "evidence_hash": HASH_A}
        for name in (
            "sampling_fields", "reasoning_mode", "context_and_prefill_policy",
            "cache_semantics", "concurrency_and_admission_policy", "material_runtime_fields",
        )
    }
    comparator = build_comparator_preflight(
        comparator_identity="cmp:r1", dimension_observations=dimensions,
        sampling_request={"requested_fields": ["top_p"], "accepted_fields": ["top_p"]},
    )
    resource = build_resource_observation(
        planned_concurrency_levels=[1],
        observations=[{"requested_concurrency": 1, "warmup_policy": "fixed", "measured_batch_count": 1,
                       "headroom_state": "AVAILABLE", "headroom_observation": {"free_memory_gb": 8},
                       "admission_state": "ADMITTED", "error_count": 0,
                       "recovery_after_batch": "NOT_OBSERVED"}],
        stop_on_resource_failure=True,
    )
    cases = [
        {"case_id": "c1", "split": "CALIBRATION", "source_group_id": "g1", "outcome": "yes"},
        {"case_id": "c2", "split": "CALIBRATION", "source_group_id": "g2", "outcome": "no"},
        {"case_id": "h1", "split": "HOLDOUT", "source_group_id": "g3", "outcome": "yes"},
        {"case_id": "h2", "split": "HOLDOUT", "source_group_id": "g4", "outcome": "no"},
    ]
    gates = [
        {"gate_id": "smoke:safe", "phase": "SMOKE", "metric": "errors", "comparator": "LTE",
         "threshold": 0, "kind": "SAFETY", "hard": True},
        {"gate_id": "cal:quality", "phase": "CALIBRATION", "metric": "quality", "comparator": "GTE",
         "threshold": 0.8, "kind": "CORRECTNESS", "hard": True},
        {"gate_id": "holdout:quality", "phase": "HOLDOUT", "metric": "quality", "comparator": "GTE",
         "threshold": 0.8, "kind": "QUALITY", "hard": True},
    ]
    integrity = build_experiment_integrity(
        experiment_id=experiment_id,
        calibration_members=[{"identity": row["case_id"], "base_identity": row["source_group_id"]}
                             for row in cases if row["split"] == "CALIBRATION"],
        heldout_members=[{"identity": row["case_id"], "base_identity": row["source_group_id"]}
                         for row in cases if row["split"] == "HOLDOUT"],
        independence_unit=INDEPENDENCE_UNIT_ROW,
        policy_derivation_ref="calibration-only",
        frozen_policy={"staged_hard_gates": gates},
        freeze_generation=1,
        heldout_evaluation_start_generation=2,
        calibration_status=CALIBRATED,
    )
    args = {
        "experiment_id": experiment_id,
        "cases": cases,
        "independence_unit": INDEPENDENCE_UNIT_ROW,
        "value_claim": True,
        "ground_truth_provenance": {"source": "truth:v1", "sha256": HASH_A},
        "leakage_observation": {"status": "PASS", "evaluator": "audit:v1", "evidence_hash": HASH_B},
        "incumbent": {"identity": "incumbent:v1"},
        "comparator_preflight": comparator,
        "resource_observation": resource,
        "resource_stop_policy": {"stop_on_resource_failure": True},
        "hard_gates": gates,
        "experiment_integrity": integrity,
    }
    args.update(overrides)
    return build_cohort_preflight(**args)


def _delta(changes):
    raw = json.dumps(changes, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return {"changes": changes, "evidence_hash": "sha256:" + hashlib.sha256(raw.encode("utf-8")).hexdigest()}


def _rebind(record):
    unsigned = {key: value for key, value in record.items() if key != "binding_hash"}
    raw = json.dumps(unsigned, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    record["binding_hash"] = "sha256:" + hashlib.sha256(raw.encode("utf-8")).hexdigest()
    return record


def _stage_complete(preflight):
    stage = build_staged_gate_evidence(preflight=preflight)
    stage = build_staged_gate_evidence(
        preflight=preflight, previous_evidence=stage,
        stage_observation={"observation_id": "smoke", "status": "PASS", "evidence_hash": HASH_A,
                           "metrics": {"errors": 0}},
    )
    stage = build_staged_gate_evidence(
        preflight=preflight, previous_evidence=stage,
        stage_observation={"observation_id": "calibration", "status": "PASS", "evidence_hash": HASH_B,
                           "metrics": {"quality": 0.9}},
    )
    return build_staged_gate_evidence(
        preflight=preflight, previous_evidence=stage,
        stage_observation={"observation_id": "holdout", "status": "PASS", "evidence_hash": HASH_A,
                           "metrics": {"quality": 0.9}},
    )


def _run(experiment_id, preflight, *, generation=1, state=COMPLETE, cohort_hash=None, policy_hash=None):
    return build_experiment_run_identity(
        experiment_id=experiment_id,
        experiment_generation=generation,
        subject_repo="James3014/example",
        subject_commit="abc123",
        subject_tree="tree123",
        harness_identity="harness:v1",
        harness_hash=HASH_A,
        cohort_identity="cohort:v1",
        cohort_hash=cohort_hash or preflight["cohort"]["cohort_hash"],
        frozen_policy_hash=policy_hash or preflight["frozen_policy_hash"],
        observed_state=state,
        producer_kind="harness",
        last_observed_at="2026-09-29T10:00:00Z",
        started_at="2026-09-29T09:00:00Z" if state == COMPLETE else None,
        result_artifact={"artifact_id": f"receipt:{experiment_id}", "sha256": HASH_B} if state == COMPLETE else None,
    )


def _reference(experiment_id, *, disposition="PASS", not_proven=()):
    preflight = _preflight(experiment_id)
    return {
        "experiment_ref": f"ref:{experiment_id}",
        "run_evidence": _run(experiment_id, preflight),
        "preflight_evidence": preflight,
        "staged_gate_evidence": _stage_complete(preflight),
        "disposition": disposition,
        "highest_safe_claim": "experiment result identity and bounded quality evidence",
        "not_proven": list(not_proven),
    }


def _trigger():
    return {
        "trigger_id": "runtime-revision-changed",
        "experiment_id": "exp:occamy",
        "trigger_type": "MATERIAL_DELTA",
        "field_path": "runtime.revision",
        "expected_value": "runtime-r2",
        "materiality_rationale": "runtime revision can change the recorded resource failure",
    }


def _closeout(experiments=None, triggers=None):
    return build_campaign_closeout(
        campaign_id="campaign:local-calibration-2026-09",
        campaign_generation=1,
        experiments=experiments or [
            _reference("exp:occamy", disposition="CLOSE_FOR_NOW"),
            _reference("exp:splash", disposition="PROVISIONAL_DEFAULT"),
        ],
        reopen_triggers=triggers if triggers is not None else [_trigger()],
        next_gate=NO_GATE_DEFINED,
        closed_at="2026-09-29T11:00:00Z",
        evidence_watermark={"source_ref": "nexus-learning/main", "sha256": HASH_A},
    )


def test_two_complete_experiments_can_close_with_no_fabricated_successor():
    result = _closeout()
    validate_campaign_closeout(result)
    assert result["completion_status"] == CAMPAIGN_COMPLETE
    assert result["campaign_closed"] is True
    assert result["next_gate"] == "NONE_DEFINED"
    assert len(result["experiment_refs"]) == 2
    assert "runtime_admission" not in result
    assert "model_promotion" not in result


def test_closeout_rejects_run_evidence_bound_to_a_different_cohort_or_policy():
    ref = _reference("exp:occamy")
    ref["run_evidence"] = _run("exp:occamy", ref["preflight_evidence"], cohort_hash=HASH_A)
    try:
        _closeout(experiments=[ref])
    except ValueError as exc:
        assert "PREFLIGHT_COHORT_MISMATCH" in str(exc)
    else:
        raise AssertionError("cohort identity mismatch must fail closed")


def test_rejected_experiment_remains_terminal_after_session_restart_and_reopens_only_on_bound_delta():
    rejected = _reference("exp:occamy", disposition="CLOSE_FOR_NOW")
    # Produce a terminal early-stop record; lack of a later session does not alter its receipt.
    state = build_staged_gate_evidence(preflight=rejected["preflight_evidence"])
    state = build_staged_gate_evidence(
        preflight=rejected["preflight_evidence"], previous_evidence=state,
        stage_observation={"observation_id": "smoke", "status": "FAIL", "evidence_hash": HASH_A,
                           "metrics": {"errors": 2}},
    )
    rejected["staged_gate_evidence"] = state
    record = _closeout(experiments=[rejected])
    old_hash = record["binding_hash"]
    reopened = evaluate_reopen_trigger(
        record,
        experiment_id="exp:occamy",
        evidence_delta=_delta({"runtime": {"revision": "runtime-r2"}}),
        proposed_experiment_generation=2,
    )
    assert reopened["eligible"] is True
    assert reopened["new_experiment_generation"] == 2
    assert reopened["prior_closeout_binding_hash"] == old_hash
    assert record["binding_hash"] == old_hash


def test_unrelated_delta_does_not_reopen_closed_experiment():
    record = _closeout(experiments=[_reference("exp:occamy", disposition="CLOSE_FOR_NOW")])
    reopened = evaluate_reopen_trigger(
        record,
        experiment_id="exp:occamy",
        evidence_delta=_delta({"unrelated": {"docs": "updated"}}),
        proposed_experiment_generation=2,
    )
    assert reopened["eligible"] is False
    assert reopened["matched_trigger_id"] is None
    assert reopened["prior_receipt_hashes"] == sorted(
        item["result_receipt_sha256"] for item in record["experiment_refs"]
    )
    assert reopened["trigger_evidence_hash"] is None


def test_reopen_delta_hash_must_bind_the_exact_change_payload():
    record = _closeout(experiments=[_reference("exp:occamy", disposition="CLOSE_FOR_NOW")])
    changes = {"runtime": {"revision": "runtime-r2"}}
    with pytest.raises(ValueError, match="DELTA_EVIDENCE_INVALID"):
        evaluate_reopen_trigger(
            record,
            experiment_id="exp:occamy",
            evidence_delta={"changes": changes, "evidence_hash": HASH_B},
            proposed_experiment_generation=2,
        )


def test_owner_decision_must_match_recorded_reference_and_hash():
    trigger = {
        "trigger_id": "owner-approval-1", "experiment_id": "exp:occamy",
        "trigger_type": "OWNER_DECISION", "owner_decision_ref": "decision:reopen-1",
        "owner_decision_sha256": HASH_A, "materiality_rationale": "explicit owner decision",
    }
    record = _closeout(experiments=[_reference("exp:occamy", disposition="CLOSE_FOR_NOW")], triggers=[trigger])
    denied = evaluate_reopen_trigger(
        record, experiment_id="exp:occamy", owner_decision_ref="decision:other",
        owner_decision_sha256=HASH_A, proposed_experiment_generation=2,
    )
    allowed = evaluate_reopen_trigger(
        record, experiment_id="exp:occamy", owner_decision_ref="decision:reopen-1",
        owner_decision_sha256=HASH_A, proposed_experiment_generation=2,
    )
    assert denied["eligible"] is False
    assert allowed["eligible"] is True


def test_provisional_comparator_keeps_missing_correctness_claim_visible():
    record = _closeout()
    splash = next(item for item in record["experiment_refs"] if item["experiment_id"] == "exp:splash")
    assert "formal correctness certification" in splash["not_proven"]
    assert splash["correctness_certification"]["status"] == "NOT_EVALUATED"
    assert splash["performance_cannot_override_correctness"] is True


def test_missing_terminal_receipt_produces_explicit_incomplete_closeout():
    preflight = _preflight("exp:pending")
    pending = {
        "experiment_ref": "ref:pending",
        "run_evidence": _run("exp:pending", preflight, state=NOT_STARTED),
        "preflight_evidence": preflight,
        "staged_gate_evidence": build_staged_gate_evidence(preflight=preflight),
        "disposition": "DEFER",
        "highest_safe_claim": "cohort readiness only",
        "not_proven": ["terminal experiment outcome"],
    }
    record = _closeout(experiments=[pending], triggers=[])
    assert record["completion_status"] == CAMPAIGN_INCOMPLETE
    assert record["campaign_closed"] is False
    assert any("run_not_terminal" in reason for reason in record["incomplete_reasons"])
    assert any("result_receipt_missing" in reason for reason in record["incomplete_reasons"])


def test_deferred_staged_gate_cannot_close_campaign_as_complete():
    preflight = _preflight("exp:deferred", leakage_observation=None)
    stage = build_staged_gate_evidence(preflight=preflight)
    assert stage["stage"] == "DEFER"
    assert stage["terminal"] is False
    deferred = {
        "experiment_ref": "ref:deferred",
        "run_evidence": _run("exp:deferred", preflight),
        "preflight_evidence": preflight,
        "staged_gate_evidence": stage,
        "disposition": "DEFER",
        "highest_safe_claim": "cohort identity only",
        "not_proven": ["leakage check"],
    }
    record = _closeout(experiments=[deferred], triggers=[])
    assert record["completion_status"] == CAMPAIGN_INCOMPLETE
    assert record["campaign_closed"] is False
    assert any("staged_gate_not_terminal" in reason for reason in record["incomplete_reasons"])


def test_closeout_tampering_fails_binding():
    record = _closeout()
    changed = copy.deepcopy(record)
    changed["experiment_refs"][1]["not_proven"] = []
    try:
        validate_campaign_closeout(changed)
    except ValueError as exc:
        assert "BINDING_HASH_MISMATCH" in str(exc)
    else:
        raise AssertionError("tampered closeout must fail validation")


@pytest.mark.parametrize(
    ("mutation", "reason"),
    [
        (lambda record: record.update(campaign_id=""), "CAMPAIGN_CAMPAIGN_ID_INVALID"),
        (lambda record: record.update(campaign_generation=True), "CAMPAIGN_CAMPAIGN_GENERATION_INVALID"),
        (lambda record: record.update(closed_at="not-a-timestamp"), "CAMPAIGN_CLOSED_AT_INVALID"),
        (lambda record: record.update(reopen_triggers="corrupted"), "CAMPAIGN_REOPEN_TRIGGERS_INVALID"),
        (lambda record: record["experiment_refs"][0].update(disposition=[]), "CAMPAIGN_EXPERIMENT_DISPOSITION_INVALID"),
        (lambda record: record["experiment_refs"][0].update(experiment_generation=True), "CAMPAIGN_EXPERIMENT_GENERATION_INVALID"),
    ],
)
def test_rebound_closeout_rejects_malformed_top_level_and_reference_fields(mutation, reason):
    record = copy.deepcopy(_closeout())
    mutation(record)
    _rebind(record)
    with pytest.raises(ValueError, match=reason):
        validate_campaign_closeout(record)


def test_reopen_without_prior_generation_returns_the_reopen_schema_shape(monkeypatch):
    import nexus_learning.campaign_closeout as campaign_closeout

    record = _closeout()
    record["experiment_refs"][0]["experiment_generation"] = None
    monkeypatch.setattr(campaign_closeout, "validate_campaign_closeout", lambda _record: record)
    result = evaluate_reopen_trigger(
        record,
        experiment_id="exp:occamy",
        proposed_experiment_generation=2,
    )
    assert result["schema"] == "nexus.learning_calibration_campaign_reopen_evidence.v1"
    assert result["experiment_id"] == "exp:occamy"
    assert result["new_experiment_generation"] is None
    assert result["claim_ceiling"]
