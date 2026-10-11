"""Consumer-side enforcement: producer-only checks must hold where data is consumed."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
from test_adoption import attempt, decide

from nexus_learning.adoption import AdoptionStore
from nexus_learning.contracts import (
    canonical_adoption_identity,
    project_adoption_into_planner_budget,
    validate_learning_policy_adoption,
)
from nexus_learning.effectiveness_measurement import replay_scorecard
from nexus_learning.outcome_memory import EpisodeOutcomeRecord, OutcomeMemoryManager
from nexus_learning.state_root import LearningStateRoot


def _legit_failed_episode() -> EpisodeOutcomeRecord:
    return EpisodeOutcomeRecord.from_task(
        task_id="t-legit",
        task_type="repair",
        task_desc="d",
        solved=False,
        wall_duration_sec=1,
        total_tokens_used=1,
        trust_mismatch=False,
        attempt_id="a1",
        action_id="x1",
        terminal_outcome="FAILED",
        qualification_evidence_present=True,
        receipts=[
            {
                "name": "cap.real",
                "selected": True,
                "invoked": True,
                "evidence_present": True,
                "gate_passed": False,
            }
        ],
    )


def test_worker_write_cannot_qualify_or_preclaim_idempotency(tmp_path: Path) -> None:
    legit = _legit_failed_episode()
    forged = {
        "task_id": "t-forged",
        "worker_name": "w-untrusted",
        "qualification_status": "QUALIFIED",
        "qualification_evidence_present": True,
        "solved": True,
        "receipts": [
            {
                "name": "cap.evil",
                "selected": True,
                "invoked": True,
                "evidence_present": True,
                "gate_passed": True,
                "outcome_contributed": True,
            }
        ],
        "idempotency_key": legit.idempotency_key,
    }
    OutcomeMemoryManager.append_worker_write(forged, project_root=tmp_path)

    result = OutcomeMemoryManager.save_episode_and_tune_sync(legit, project_root=tmp_path)

    assert result["status"] == "PASS"
    policy = json.loads(
        (tmp_path / ".nexus/memory/dynamic_learning_policy.json").read_text(encoding="utf-8")
    )
    assert "cap.evil" not in policy["promoted_capabilities"]
    assert policy["penalized_capabilities"] == ["cap.real"]
    rows = (tmp_path / ".nexus/memory/outcome_history.jsonl").read_text(encoding="utf-8")
    worker_row = json.loads(rows.splitlines()[0])
    assert worker_row["schema_version"] == "nexus_outcome_memory_worker_write.v1"
    assert "qualification_status" not in worker_row
    assert "idempotency_key" not in worker_row


def _forged_adoption(**overrides: object) -> dict:
    adoption = {
        "schema": "nexus.learning_policy_adoption.v1",
        "owner_authority_reference": "owner",
        "recommendation_id": "lrec:x",
        "recommendation_hash": "0" * 64,
        "validation_id": "lval:x",
        "validation_hash": "0" * 64,
        "source_revision": "rev-x",
        "adopted_scope": {"task_family": "local_heal"},
        "target_policy_delta": {"episodic_memory_injection": {"enabled": True}},
        "target_policy_hash": "f" * 64,
        "previous_policy": {},
        "previous_policy_hash": "e" * 64,
        "rollback_target": {"target_state": {"memory_retrieval": "shadow"}},
        "adoption_status": "ACTIVE_CANDIDATE",
        "claim_ceiling": "SUPPORTED_POLICY_RECOMMENDATION",
        "route_truth_source": "CapabilityPlanner",
        "direct_route_mutation_allowed": False,
    }
    adoption.update(overrides)
    adoption["adoption_hash"], adoption["adoption_id"] = canonical_adoption_identity(adoption)
    return adoption


@pytest.mark.parametrize(
    ("overrides", "code"),
    [
        (
            {"target_policy_delta": {"CapabilityPlanner": {"force_route": "evil"}}},
            "ADOPTION_DIRECT_PLANNER_MUTATION_FORBIDDEN",
        ),
        (
            {"target_policy_delta": {"workforce_admission": "open"}},
            "ADOPTION_DIRECT_PLANNER_MUTATION_FORBIDDEN",
        ),
        ({}, "ADOPTION_TARGET_POLICY_HASH_MISMATCH"),
    ],
)
def test_rehashed_adoption_outside_authority_is_rejected_at_consumers(
    tmp_path: Path, overrides: dict, code: str
) -> None:
    adoption = _forged_adoption(**overrides)
    with pytest.raises(ValueError, match=code):
        validate_learning_policy_adoption(adoption)
    with pytest.raises(ValueError, match=code):
        AdoptionStore(LearningStateRoot(tmp_path)).write_adoption(adoption)
    with pytest.raises(ValueError, match=code):
        project_adoption_into_planner_budget(adoption, task_desc="local_heal repair")


def test_read_adoption_rejects_tampered_file(tmp_path: Path) -> None:
    root = LearningStateRoot(tmp_path)
    adoption = _forged_adoption(
        target_policy_delta={"route_authority": "x"},
    )
    path = root.adoption_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(adoption), encoding="utf-8")
    with pytest.raises(ValueError, match="ADOPTION_DIRECT_PLANNER_MUTATION_FORBIDDEN"):
        AdoptionStore(root).read_adoption()


def test_edited_paired_uplift_summary_cannot_cross_min_pairs() -> None:
    rows = [attempt("fp-1", "memory_off", False), attempt("fp-1", "memory_on", True)]
    genuine = replay_scorecard(rows)
    assert decide(genuine)["reason_codes"] == ["ADOPTION_INSUFFICIENT_PAIRS"]

    forged = copy.deepcopy(genuine)
    forged["paired_memory_uplift"].update(eligible=3, eligible_pairs=3, numerator=3, denominator=3)
    result = decide(forged)

    assert result["decision"] == "REJECT"
    assert result["reason_codes"] == ["ADOPTION_SCORECARD_INVALID"]
