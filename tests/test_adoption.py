"""Tests for the governed adoption pipeline built from a memory on/off scorecard."""

import copy

import pytest

from nexus_learning.adoption import (
    ADOPTION_DECISION_ADOPT,
    ADOPTION_DECISION_DEFER,
    ADOPTION_DECISION_REJECT,
    ADOPTION_PIPELINE_SCHEMA,
    AdoptionStore,
    build_adoption_from_scorecard,
)
from nexus_learning.contracts import (
    project_adoption_into_planner_budget,
    validate_learning_policy_adoption,
)
from nexus_learning.effectiveness_measurement import normalize_attempt_row, replay_scorecard
from nexus_learning.state_root import LearningStateRoot

FINGERPRINTS = ("fp-1", "fp-2", "fp-3")
REVISION = "rev-1"
RUNTIME = "runtime-1"


def attempt(fingerprint, arm, passed, *, origin="physical", revision=REVISION, refs=None):
    on = arm == "memory_on"
    row = {
        "task_fingerprint": fingerprint,
        "task_id": f"task-{fingerprint}",
        "attempt_id": f"attempt-{fingerprint}-{arm}",
        "attempt_index": 0,
        "action_id": f"action-{fingerprint}-{arm}",
        "source_revision": revision,
        "source_tree": "tree-1",
        "verifier_status": "passed" if passed else "failed",
        "verifier_artifact": f"artifact-{fingerprint}-{arm}.json",
        "verifier_artifact_hash": f"hash-{fingerprint}-{arm}",
        "verifier_receipt": f"receipt-{fingerprint}-{arm}",
        "memory_arm": arm,
        "retrieved_lesson_ids": ["lesson-1"] if on else [],
        "applied_attributed_lesson_ids": ["lesson-1"] if on else [],
        "terminal_outcome": "SUCCEEDED" if passed else "FAILED",
        "measured_elapsed_seconds": 2.0,
        "intervention_events": [],
        "intervention_count": 0,
        "forbidden_strategy_identity": "forbidden-x",
        "forbidden_strategy_violation_event": False,
        "missingness_reasons": [],
        "ineligibility_reasons": [],
    }
    if origin is not None:
        row["evidence_origin"] = origin
    if refs is None:
        refs = ["retrieval_receipt:lesson-1", f"ollama_consumption:{fingerprint}"] if on else []
    if refs:
        row["evidence_refs"] = list(refs)
    return row


def scorecard_rows(on_passes, off_passes, **kwargs):
    rows = []
    for index, fingerprint in enumerate(FINGERPRINTS):
        rows.append(attempt(fingerprint, "memory_off", off_passes[index], **kwargs))
        rows.append(attempt(fingerprint, "memory_on", on_passes[index], **kwargs))
    return rows


def decide(scorecard, state_root=None, **overrides):
    options = {
        "source_revision": REVISION,
        "runtime_identity": RUNTIME,
        "required_quality_floor": 0.6,
        "critical_failure_ceiling": 0,
        "validator_identity": "validator-1",
        "owner_authority_reference": "owner:james",
        "state_root": state_root,
    }
    options.update(overrides)
    return build_adoption_from_scorecard(scorecard, **options)


# Memory-on passes every task; memory-off fails every task: a paired uplift.
UPLIFT = (True, True, True), (False, False, False)


def test_positive_uplift_with_quality_gate_adopts_and_persists(tmp_path):
    scorecard = replay_scorecard(scorecard_rows(*UPLIFT))
    state_root = LearningStateRoot(tmp_path)

    result = decide(scorecard, state_root=state_root)

    assert result["schema"] == ADOPTION_PIPELINE_SCHEMA
    assert result["decision"] == ADOPTION_DECISION_ADOPT, result["reason_codes"]
    assert result["reason_codes"] == []
    assert result["advisory_only"] is True
    assert result["authority_effect"] is False
    assert result["validation"]["validation_disposition"] == (
        "VALIDATED_FOR_ADOPTION_CONSIDERATION"
    )
    assert result["rollback"] is None

    store = AdoptionStore(state_root)
    stored = store.read_adoption()
    assert stored == result["adoption"]
    validate_learning_policy_adoption(stored)
    assert state_root.adoption_path.exists()
    assert not state_root.rollback_path.exists()


def test_adoption_is_accepted_by_planner_projection(tmp_path):
    result = decide(replay_scorecard(scorecard_rows(*UPLIFT)))
    adoption = result["adoption"]

    budget = project_adoption_into_planner_budget(adoption, task_desc="memory retrieval task")

    lineage = budget["learning_policy"]["adoption_lineage"]
    assert lineage["adoption_id"] == adoption["adoption_id"]
    # adopted_scope carries no task_family, so the projection reports OUT_OF_SCOPE
    # and keeps episodic injection disabled. Scope is a known gap, see report.
    assert lineage["status"] == "OUT_OF_SCOPE"
    assert budget["learning_policy"]["episodic_memory_injection"] == {"enabled": False}


def test_zero_uplift_defers_without_writing(tmp_path):
    # Both arms pass on every task: quality passes, but no failed-memory-off pair exists.
    scorecard = replay_scorecard(scorecard_rows((True, True, True), (True, True, True)))
    state_root = LearningStateRoot(tmp_path)

    result = decide(scorecard, state_root=state_root)

    assert result["decision"] == ADOPTION_DECISION_DEFER
    assert result["reason_codes"] == ["ADOPTION_NO_POSITIVE_UPLIFT"]
    assert result["adoption"] is None
    assert not state_root.adoption_path.exists()
    assert not state_root.rollback_path.exists()


def test_simulated_evidence_is_rejected(tmp_path):
    rows = scorecard_rows(*UPLIFT)
    rows[1]["evidence_origin"] = "simulated"
    state_root = LearningStateRoot(tmp_path)

    result = decide(replay_scorecard(rows), state_root=state_root)

    assert result["decision"] == ADOPTION_DECISION_REJECT
    assert result["reason_codes"] == ["ADOPTION_SIMULATED_EVIDENCE"]
    assert not state_root.adoption_path.exists()


def test_row_without_evidence_origin_is_rejected():
    rows = scorecard_rows(*UPLIFT)
    del rows[0]["evidence_origin"]

    result = decide(replay_scorecard(rows))

    assert result["decision"] == ADOPTION_DECISION_REJECT
    assert result["reason_codes"] == ["ADOPTION_SIMULATED_EVIDENCE"]


def test_on_arm_below_required_quality_defers(tmp_path):
    # Memory-on passes 2 of 3 tasks (rate 0.67) and has one critical failure.
    scorecard = replay_scorecard(scorecard_rows((True, True, False), (False, False, False)))
    state_root = LearningStateRoot(tmp_path)

    result = decide(
        scorecard,
        state_root=state_root,
        required_quality_floor=0.9,
        critical_failure_ceiling=3,
    )

    assert result["decision"] == ADOPTION_DECISION_DEFER
    assert result["reason_codes"] == ["ADOPTION_QUALITY_GATE_NOT_PASSED"]
    assert result["quality_gate"] is not None
    assert not state_root.adoption_path.exists()


@pytest.mark.parametrize(
    "scorecard",
    [
        None,
        {"schema": "wrong.schema", "rows": [], "paired_memory_uplift": {}},
        {
            "schema": "nexus.learning_effectiveness_measurement.v1",
            "rows": [],
            "paired_memory_uplift": {},
        },
    ],
)
def test_invalid_scorecard_is_rejected(scorecard):
    result = decide(scorecard)

    assert result["decision"] == ADOPTION_DECISION_REJECT
    assert result["reason_codes"] == ["ADOPTION_SCORECARD_INVALID"]


def test_scorecard_without_paired_payload_is_rejected():
    scorecard = copy.deepcopy(replay_scorecard(scorecard_rows(*UPLIFT)))
    del scorecard["paired_memory_uplift"]

    assert decide(scorecard)["reason_codes"] == ["ADOPTION_SCORECARD_INVALID"]


def test_stale_source_revision_defers():
    scorecard = replay_scorecard(scorecard_rows(*UPLIFT))

    result = decide(scorecard, source_revision="rev-2")

    assert result["decision"] == ADOPTION_DECISION_DEFER
    assert result["reason_codes"] == ["ADOPTION_SOURCE_REVISION_MISMATCH"]


def test_missing_consumption_evidence_blocks_validation_without_weakening():
    rows = scorecard_rows(*UPLIFT, refs=["retrieval_receipt:lesson-1"])

    result = decide(replay_scorecard(rows))

    assert result["decision"] == ADOPTION_DECISION_DEFER
    assert result["reason_codes"][0] == "ADOPTION_VALIDATION_NOT_PASSED"
    assert "missing_physical_consumption" in result["reason_codes"]
    assert result["adoption"] is None


def test_default_rollback_target_is_validator_compatible():
    result = decide(replay_scorecard(scorecard_rows(*UPLIFT)))

    assert result["recommendation"]["rollback_target"] == {
        "target_state": {"memory_retrieval": "shadow"}
    }
    assert result["adoption"]["rollback_target"] == result["recommendation"]["rollback_target"]


def test_rollback_target_without_target_state_defers():
    result = decide(
        replay_scorecard(scorecard_rows(*UPLIFT)),
        rollback_target={"memory_retrieval": "shadow"},
    )

    assert result["decision"] == ADOPTION_DECISION_DEFER
    assert "rollback_not_defined" in result["reason_codes"]


def test_store_writes_identical_bytes_when_repeated(tmp_path):
    result = decide(replay_scorecard(scorecard_rows(*UPLIFT)))
    state_root = LearningStateRoot(tmp_path)
    store = AdoptionStore(state_root)

    first = store.write_adoption(result["adoption"]).read_bytes()
    second = store.write_adoption(result["adoption"]).read_bytes()
    third = AdoptionStore(state_root).write_adoption(result["adoption"]).read_bytes()

    assert first == second == third
    assert store.read_adoption() == result["adoption"]
    assert not list(state_root.policy_dir.glob(".*.tmp")), "temp files must not linger"
    assert sorted(path.name for path in state_root.policy_dir.iterdir()) == [
        "governed_learning_policy_adoption.json"
    ]


def test_store_rejects_tampered_adoption(tmp_path):
    adoption = copy.deepcopy(decide(replay_scorecard(scorecard_rows(*UPLIFT)))["adoption"])
    adoption["adopted_scope"]["task_fingerprint"] = "fp-other"
    store = AdoptionStore(LearningStateRoot(tmp_path))

    with pytest.raises(ValueError):
        store.write_adoption(adoption)
    assert store.read_adoption() is None


def test_replay_keeps_evidence_origin_and_refs_only_when_supplied():
    row = attempt("fp-9", "memory_on", True)
    plain = dict(row)
    del plain["evidence_origin"], plain["evidence_refs"]

    with_fields = normalize_attempt_row(row).to_dict()
    without = normalize_attempt_row(plain).to_dict()

    assert with_fields["evidence_origin"] == "physical"
    assert with_fields["evidence_refs"] == sorted(row["evidence_refs"])
    assert "evidence_origin" not in without
    assert "evidence_refs" not in without
