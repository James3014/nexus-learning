from __future__ import annotations

import copy

import pytest

from nexus_learning.experiment_integrity import (
    EFFECT_SUCCEEDED,
    LOCAL_FAKE_PROVIDER,
    REMOTE_PROVIDER_OBSERVED,
    SIMULATION_ONLY,
    build_evidence_origin_provenance,
)
from nexus_learning.experiment_run_identity import (
    COMPLETE,
    OBSERVE_SAME_RUN,
    OUTCOME_UNKNOWN,
    RECONCILE_WITH_EFFECT_OWNER,
    RUNNING,
    build_experiment_run_identity,
    classify_run_observation,
    validate_experiment_run_identity,
)

HASH_A = "sha256:" + "a" * 64
HASH_B = "sha256:" + "b" * 64


def _run(**overrides):
    args = {
        "experiment_id": "exp:one",
        "experiment_generation": 1,
        "subject_repo": "James3014/example",
        "subject_commit": "abc123",
        "subject_tree": "tree123",
        "harness_identity": "harness:v1",
        "harness_hash": HASH_A,
        "cohort_identity": "cohort:v1",
        "cohort_hash": HASH_B,
        "frozen_policy_hash": HASH_A,
        "observed_state": RUNNING,
        "producer_kind": "local-experiment-harness",
        "last_observed_at": "2026-09-29T10:00:00Z",
        "started_at": "2026-09-29T09:00:00Z",
        "producer_run_id": "run-1",
        "operation_id": "op-1",
        "effect_identity": "effect-1",
        "requested_execution_identity": {"provider": "p", "model": "m", "runtime": "r", "revision": "rev1"},
        "configured_execution_identity": {"provider": "p", "model": "m", "runtime": "r", "revision": "rev1"},
        "observed_execution_identity": {"provider": "p", "model": "m", "runtime": "r", "revision": "rev1"},
        "producer_observations": {},
    }
    args.update(overrides)
    return build_experiment_run_identity(**args)


def test_running_same_generation_requires_observing_same_run_without_pid_or_session_index():
    previous = _run(producer_observations={})
    # A new coordinator has no local PID or session index to consult.
    current = _run(producer_observations={}, last_observed_at="2026-09-29T10:05:00Z")
    result = classify_run_observation(previous, current)
    assert result["classification"] == RUNNING
    assert result["requirement"] == OBSERVE_SAME_RUN
    assert result["retry_authorized"] is False
    assert result["new_effect_authorized"] is False


def test_outcome_unknown_requires_external_effect_reconciliation_and_never_authorizes_retry():
    previous = _run(observed_state=OUTCOME_UNKNOWN)
    current = _run(observed_state=OUTCOME_UNKNOWN, last_observed_at="2026-09-29T10:05:00Z")
    result = classify_run_observation(previous, current)
    assert result["requirement"] == RECONCILE_WITH_EFFECT_OWNER
    assert result["retry_authorized"] is False


def test_terminal_result_requires_and_binds_artifact_identity_and_hash():
    run = _run(
        observed_state=COMPLETE,
        result_artifact={"artifact_id": "receipt:one", "sha256": HASH_B},
    )
    validate_experiment_run_identity(run)
    tampered = copy.deepcopy(run)
    tampered["result_artifact"]["sha256"] = HASH_A
    with pytest.raises(ValueError, match="BINDING_HASH_MISMATCH"):
        validate_experiment_run_identity(tampered)
    with pytest.raises(ValueError, match="TERMINAL_REQUIRES_RESULT_ARTIFACT"):
        _run(observed_state=COMPLETE)


@pytest.mark.parametrize(
    "changes",
    [
        {"cohort_hash": HASH_A},
        {"frozen_policy_hash": HASH_B},
        {"subject_commit": "def456"},
        {"subject_tree": "tree456"},
        {"harness_hash": HASH_B},
    ],
)
def test_changed_source_or_cohort_binding_requires_new_generation(changes):
    previous = _run()
    current = _run(**changes)
    result = classify_run_observation(previous, current)
    assert result["classification"] == "GENERATION_IDENTITY_MISMATCH"
    assert result["requirement"] == "NEW_GENERATION_REQUIRED"


def test_generation_hash_distinguishes_requested_and_configured_identity_without_filling_observed():
    run = _run(
        requested_execution_identity={"provider": "p", "model": "expected", "runtime": "r", "revision": None},
        configured_execution_identity={"provider": "p", "model": "configured", "runtime": "r", "revision": "v1"},
        observed_execution_identity={"provider": None, "model": None, "runtime": None, "revision": None},
    )
    assert run["requested_execution_identity"]["model"] == "expected"
    assert run["configured_execution_identity"]["model"] == "configured"
    assert run["observed_execution_identity"]["model"] is None


@pytest.mark.parametrize("origin", [SIMULATION_ONLY, LOCAL_FAKE_PROVIDER])
def test_simulated_and_local_fake_provenance_remains_compatible(origin):
    provenance = build_evidence_origin_provenance(
        origin_class=origin,
        requested_provider="p",
        requested_model="m",
        configured_provider="p",
        configured_model="m",
        external_effect_started=False,
    )
    run = _run(evidence_origin_provenance=provenance)
    assert run["evidence_origin_provenance"]["origin_class"] == origin


def test_remote_observed_provenance_remains_compatible():
    provenance = build_evidence_origin_provenance(
        origin_class=REMOTE_PROVIDER_OBSERVED,
        requested_provider="requested-provider",
        requested_model="requested-model",
        configured_provider="configured-provider",
        configured_model="configured-model",
        observed_provider="observed-provider",
        observed_model="observed-model",
        observed_revision="revision-1",
        external_effect_started=True,
        effect_outcome=EFFECT_SUCCEEDED,
        effect_identity="effect-1",
        operation_id="op-1",
        observation_receipt={
            "effect_identity": "effect-1",
            "operation_id": "op-1",
            "external_effect_started": True,
            "outcome": EFFECT_SUCCEEDED,
            "source_receipt_ref": "receipt://remote/1",
            "source_receipt_sha256": HASH_A,
            "transport_class": "REMOTE_PROVIDER",
            "observed_provider": "observed-provider",
            "observed_model": "observed-model",
            "observed_revision": "revision-1",
        },
    )
    run = _run(
        observed_state=COMPLETE,
        result_artifact={"artifact_id": "result:remote", "sha256": HASH_B},
        evidence_origin_provenance=provenance,
    )
    assert run["evidence_origin_provenance"]["observed_identity"]["model"] == "observed-model"
