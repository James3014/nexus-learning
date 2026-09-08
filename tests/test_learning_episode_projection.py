from __future__ import annotations

import json

from nexus_learning.contracts import build_nexus_learning_episode
from nexus_learning.episode_projection import project_learning_entries, write_learning_projection

_QUALIFICATION = {
    "repeatability": True,
    "prevention_rule": "verified repair rule",
    "authority_qualification": True,
}


def test_projection_deduplicates_canonical_and_aggregates_evidence() -> None:
    rows = [
        {
            "schema": "nexus.learning_episode.v1",
            "episode_id": "lep-1",
            "idempotency_key": "idem-1",
            "task_id": "t1",
            "terminal_evidence": {"receipt": "r1"},
            "stages": {"outcome_measured": True},
            "source_schema": "nexus.learning_episode.v1",
            "terminal_outcome": "SUCCEEDED",
            "retrieved_lesson_ids": ["l1"],
            "applied_lesson_ids": ["l1"],
            "created_at": "2026-01-01T00:00:00Z",
        },
        {
            "schema": "nexus.learning_episode.v1",
            "episode_id": "lep-1",
            "idempotency_key": "idem-1",
            "task_id": "t1",
            "terminal_evidence": {"receipt": "r2"},
            "stages": {"outcome_measured": True},
            "source_schema": "nexus.learning_episode.v1",
            "terminal_outcome": "SUCCEEDED",
            "created_at": "2026-01-02T00:00:00Z",
            "applied_lesson_ids": ["l1"],
        },
    ]
    projected = project_learning_entries(rows)
    assert len(projected) == 1
    assert projected[0]["occurrence_count"] == 1
    assert projected[0]["evidence_refs"] == ["r1"]


def test_legacy_patterns_are_scoped_and_correct_abstain_is_negative_only() -> None:
    rows = [
        {
            "source": "local_heal",
            "task_id": "t2",
            "classification": "correct_abstain",
            "summary": "no patch",
            "receipt": {"status": "pass"},
        },
        {
            "source": "learn_mode",
            "action": "verifier_pass",
            "status": "success",
            "reason": "fixed",
            "topic": "bug",
            "source_schema": "lesson_event.v1",
            "terminal_outcome": "SUCCEEDED",
            "terminal_evidence": {"receipt": "v1"},
            "qualification_status": "QUALIFIED",
            "qualification": _QUALIFICATION,
        },
    ]
    projected = project_learning_entries(rows)
    abstain = next(row for row in projected if row["pattern_type"] == "correct_abstain")
    assert abstain["retrieval_eligible"] is True
    assert abstain["qualification_reason"] == "negative_hint_only"
    assert abstain["lesson_disposition"] == "negative_hint"
    verifier = next(row for row in projected if row["pattern_type"] == "verifier_pass")
    assert verifier["retrieval_eligible"] is True


def test_verifier_pass_without_evidence_is_not_retrieval_eligible() -> None:
    projected = project_learning_entries(
        [
            {
                "source": "learn_mode",
                "action": "verifier_pass",
                "status": "success",
                "reason": "fixed",
            }
        ]
    )
    assert projected[0]["retrieval_eligible"] is False
    assert projected[0]["qualification_reason"] == "missing_terminal_evidence"


def test_verifier_pass_rejects_status_only_or_incomplete_qualification() -> None:
    base = {
        "source": "local_heal",
        "classification": "verifier_pass",
        "terminal_outcome": "SUCCEEDED",
        "qualification_status": "QUALIFIED",
    }

    status_only = project_learning_entries(
        [{**base, "terminal_evidence": {"verifier_status": "PASS"}}]
    )[0]
    incomplete = project_learning_entries(
        [
            {
                **base,
                "terminal_evidence": {"receipt": "receipt://verified"},
                "qualification": {"repeatability": True},
            }
        ]
    )[0]

    assert status_only["retrieval_eligible"] is False
    assert status_only["qualification_reason"] == "missing_terminal_evidence"
    assert incomplete["retrieval_eligible"] is False
    assert incomplete["qualification_reason"] == "incomplete_qualification"


def test_projection_write_is_atomic_and_preserves_raw(tmp_path) -> None:
    output = tmp_path / "projection.jsonl"
    raw = [{"schema": "legacy", "task_id": "t3", "summary": "x", "source": "learn_mode"}]
    result = write_learning_projection(raw, output)
    assert result["status"] == "PASS"
    assert json.loads(output.read_text().splitlines()[0])["task_ids"] == ["t3"]
    assert raw[0] == {"schema": "legacy", "task_id": "t3", "summary": "x", "source": "learn_mode"}


def test_source_shape_and_semantic_grouping_ignore_random_legacy_ids() -> None:
    rows = [
        {
            "task_id": "lh",
            "classification": "correct_abstain",
            "summary": "same",
            "findings_card_id": "random-a",
            "lesson_id": "a",
            "receipt": "receipt:pending",
        },
        {
            "task_id": "lh",
            "classification": "correct_abstain",
            "summary": "same",
            "findings_card_id": "random-b",
            "lesson_id": "b",
            "receipt": "receipt:pending",
        },
        {
            "episode_id": "random-1",
            "task_id": "canon",
            "classification": "verifier_pass",
            "summary": "same",
            "terminal_outcome": "SUCCEEDED",
            "terminal_evidence": {"receipt": "r"},
            "qualification_status": "QUALIFIED",
            "qualification": _QUALIFICATION,
        },
        {
            "episode_id": "random-2",
            "task_id": "canon",
            "classification": "verifier_pass",
            "summary": "same",
            "terminal_outcome": "SUCCEEDED",
            "terminal_evidence": {"receipt": "r"},
            "qualification_status": "QUALIFIED",
            "qualification": _QUALIFICATION,
        },
    ]
    projected = project_learning_entries(rows)
    abstain = next(row for row in projected if row["task_ids"] == ["lh"])
    assert abstain["occurrence_count"] == 2
    assert abstain["source"] == "local_heal"
    canonical = next(row for row in projected if row["task_ids"] == ["canon"])
    assert canonical["occurrence_count"] == 2
    assert canonical["episode_ids"] == ["random-1", "random-2"]


def test_semantic_group_aggregates_receipts_and_requires_qualified_success() -> None:
    rows = [
        {
            "source": "local_heal",
            "task_id": "repair-task",
            "classification": "verifier_pass",
            "summary": "same repair",
            "terminal_outcome": "SUCCEEDED",
            "terminal_evidence": {"receipt": "receipt://one"},
            "qualification_status": "UNQUALIFIED",
        },
        {
            "source": "local_heal",
            "task_id": "repair-task",
            "classification": "verifier_pass",
            "summary": "same repair",
            "terminal_outcome": "SUCCEEDED",
            "terminal_evidence": {"receipt": "receipt://two"},
            "qualification_status": "QUALIFIED",
            "qualification": _QUALIFICATION,
        },
    ]

    projected = project_learning_entries(rows)

    assert len(projected) == 1
    assert projected[0]["occurrence_count"] == 2
    assert projected[0]["retrieval_eligible"] is True
    assert projected[0]["evidence_refs"] == ["receipt://one", "receipt://two"]
    assert project_learning_entries([rows[0]])[0]["retrieval_eligible"] is False


def test_projection_does_not_merge_distinct_capability_events() -> None:
    projected = project_learning_entries(
        [
            {
                "capability_name": "repair_loop",
                "gate_passed": True,
                "outcome": {"status": "pass", "reason": "verified"},
            },
            {
                "capability_name": "claim_gate",
                "gate_passed": True,
                "outcome": {"status": "pass", "reason": "verified"},
            },
        ]
    )

    assert len(projected) == 2
    assert {row["classification"] for row in projected} == {"repair_loop", "claim_gate"}


def test_projection_preserves_occurrence_count_when_reprojected() -> None:
    projected = project_learning_entries(
        [
            {
                "source": "local_heal",
                "task_id": "reprojected",
                "classification": "correct_abstain",
                "summary": "same negative hint",
                "occurrence_count": 12,
            }
        ]
    )

    assert projected[0]["occurrence_count"] == 12


def _g3_episode(
    *, key: str, outcome: str = "SUCCEEDED", disposition: str = "reinforce", targets=()
):
    qualification = _QUALIFICATION if outcome != "RETIRED" else {}
    episode = build_nexus_learning_episode(
        task_id=f"task-{key}",
        attempt_id=f"attempt-{key}",
        action_id=f"action-{key}",
        source="local_heal",
        terminal_outcome=outcome,
        terminal_evidence={
            "receipt": f"receipt://{key}",
            "verifier_status": "PASS" if outcome == "SUCCEEDED" else "FAIL",
        },
        retrieved_lesson_ids=list(targets),
        applied_lesson_ids=list(targets),
        qualification=qualification,
        lesson_disposition=disposition,
        idempotency_key=f"g3-{key}",
    )
    episode["classification"] = "verifier_pass" if outcome == "SUCCEEDED" else "verifier_fail"
    episode["summary"] = "stable repair rule"
    return episode


def test_later_contradict_invalidates_prior_canonical_projection() -> None:
    prior = _g3_episode(key="prior")
    invalidator = _g3_episode(
        key="contradict",
        outcome="FAILED",
        disposition="contradict",
        targets=[prior["episode_id"]],
    )

    projected = project_learning_entries([prior, invalidator])
    prior_projection = next(row for row in projected if prior["episode_id"] in row["episode_ids"])

    assert prior_projection["retrieval_eligible"] is False
    assert prior_projection["validity_state"] == "invalidated"
    assert prior_projection["invalidated_episode_ids"] == [prior["episode_id"]]
    assert (
        prior_projection["invalidation_evidence"][0]["invalidated_by_episode_id"]
        == invalidator["episode_id"]
    )
    assert prior_projection["invalidation_evidence"][0]["invalidation_disposition"] == "contradict"


def test_later_retire_invalidates_but_preceding_control_cannot_preinvalidate() -> None:
    prior = _g3_episode(key="retired-prior")
    retire = _g3_episode(
        key="retire",
        outcome="RETIRED",
        disposition="retire",
        targets=[prior["episode_id"]],
    )

    later_projection = project_learning_entries([prior, retire])
    later_prior = next(row for row in later_projection if prior["episode_id"] in row["episode_ids"])
    assert later_prior["retrieval_eligible"] is False
    assert later_prior["validity_state"] == "invalidated"

    reversed_projection = project_learning_entries([retire, prior])
    reversed_prior = next(
        row for row in reversed_projection if prior["episode_id"] in row["episode_ids"]
    )
    assert reversed_prior["retrieval_eligible"] is True
    assert reversed_prior["validity_state"] == "active"


def test_tampered_invalidation_episode_cannot_mask_prior_projection() -> None:
    prior = _g3_episode(key="tamper-prior")
    invalidator = _g3_episode(
        key="tamper-control",
        outcome="FAILED",
        disposition="contradict",
        targets=[prior["episode_id"]],
    )
    invalidator["episode_id"] = "lep:000000000000000000000000"

    projected = project_learning_entries([prior, invalidator])
    prior_projection = next(row for row in projected if prior["episode_id"] in row["episode_ids"])

    assert prior_projection["retrieval_eligible"] is True
    assert prior_projection["validity_state"] == "active"
    assert prior_projection["invalidated_episode_ids"] == []
