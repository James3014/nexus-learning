from __future__ import annotations

import hashlib
from typing import Any

import pytest

from nexus_learning.contracts import (
    build_nexus_learning_episode,
    canonical_episode_identity,
    validate_nexus_learning_episode,
)
from nexus_learning.outcome_memory import EpisodeOutcomeRecord


def test_episode_identity_and_stages_are_stable_and_fail_closed() -> None:
    kwargs: dict[str, Any] = dict(
        task_id="task-1",
        attempt_id="attempt-1",
        action_id="action-1",
        source="learn_mode",
        terminal_outcome="SUCCEEDED",
        terminal_evidence={"verifier_status": "PASS"},
        retrieved_lesson_ids=["lesson-b", "lesson-a"],
        applied_lesson_ids=["lesson-a"],
        qualification={
            "repeatability": True,
            "prevention_rule": "rule",
            "authority_qualification": True,
        },
    )
    first = build_nexus_learning_episode(**kwargs)
    second = build_nexus_learning_episode(**kwargs)
    assert first["schema"] == "nexus.learning_episode.v1"
    assert first["source_schema"] == "nexus.learning_episode.v1"
    assert first["episode_id"] == second["episode_id"]
    assert first["idempotency_key"] == second["idempotency_key"]
    assert first["stages"]["outcome_measured"] is True
    assert first["stages"]["outcome_uplift_observed"] is False
    assert first["qualification_status"] == "QUALIFIED"


def test_episode_id_and_idempotency_key_tamper_fail_closed() -> None:
    episode = build_nexus_learning_episode(
        task_id="task-tamper",
        attempt_id="attempt-1",
        action_id="action-1",
        source="router",
        terminal_outcome="SUCCEEDED",
        terminal_evidence={"verifier_status": "PASS"},
    )
    validate_nexus_learning_episode(episode)

    tampered_id = dict(episode, episode_id=f"lep:{'0' * 24}")
    with pytest.raises(ValueError, match="IDENTITY_MISMATCH"):
        validate_nexus_learning_episode(tampered_id)

    tampered_key = dict(episode, idempotency_key="forged-custom-key")
    with pytest.raises(ValueError, match="IDENTITY_MISMATCH"):
        validate_nexus_learning_episode(tampered_key)

    swapped_id = dict(episode, episode_id=f"lep:{'a' * 24}", idempotency_key="replaced")
    with pytest.raises(ValueError, match="IDENTITY_MISMATCH"):
        validate_nexus_learning_episode(swapped_id)


def test_cross_task_identity_substitution_fails_closed() -> None:
    base: dict[str, Any] = dict(
        terminal_outcome="SUCCEEDED", terminal_evidence={"verifier_status": "PASS"}
    )
    task_a = build_nexus_learning_episode(
        task_id="task-a", attempt_id="attempt-1", action_id="action-1", source="router", **base
    )
    task_b = build_nexus_learning_episode(
        task_id="task-b", attempt_id="attempt-1", action_id="action-1", source="router", **base
    )
    assert task_a["episode_id"] != task_b["episode_id"]

    forged = dict(task_a, task_id="task-b", idempotency_key="task-b:attempt-1:action-1:router")
    with pytest.raises(ValueError, match="IDENTITY_MISMATCH"):
        validate_nexus_learning_episode(forged)


def test_empty_and_malformed_identity_keys_fail_closed() -> None:
    episode = build_nexus_learning_episode(
        task_id="task-empty",
        terminal_outcome="SUCCEEDED",
        terminal_evidence={"verifier_status": "PASS"},
    )
    with pytest.raises(ValueError, match="EMPTY_IDEMPOTENCY_KEY"):
        validate_nexus_learning_episode(dict(episode, idempotency_key=""))
    with pytest.raises(ValueError, match="EMPTY_IDEMPOTENCY_KEY"):
        validate_nexus_learning_episode(dict(episode, idempotency_key="   "))

    malformed = [
        dict(episode, episode_id="lep:nothex"),
        dict(episode, episode_id="lep:" + "z" * 24),
        dict(episode, episode_id="not-lep:" + "a" * 24),
        dict(episode, episode_id="lep:" + "a" * 25),
        dict(episode, episode_id="lep:" + "a" * 23),
    ]
    for tampered in malformed:
        with pytest.raises(ValueError, match="MALFORMED_EPISODE_ID"):
            validate_nexus_learning_episode(tampered)


def test_explicit_custom_idempotency_key_remains_compatible() -> None:
    base: dict[str, Any] = dict(
        task_id="task-custom",
        attempt_id="attempt-1",
        action_id="action-1",
        source="router",
        terminal_outcome="SUCCEEDED",
        terminal_evidence={"verifier_status": "PASS"},
    )
    first = build_nexus_learning_episode(**base, idempotency_key="custom-123")
    second = build_nexus_learning_episode(**base, idempotency_key="custom-123")
    assert first["idempotency_key"] == "custom-123"
    assert first["episode_id"] == second["episode_id"]
    assert first["episode_id"].startswith("lep:")
    assert len(first["episode_id"]) == 4 + 24
    validate_nexus_learning_episode(first)

    _, expected = canonical_episode_identity(
        task_id=base["task_id"],
        attempt_id=base["attempt_id"],
        action_id=base["action_id"],
        source=base["source"],
        idempotency_key="custom-123",
    )
    assert expected == first["episode_id"]
    assert expected == "lep:" + hashlib.sha256(b"custom-123").hexdigest()[:24]

    distinct = build_nexus_learning_episode(**base, idempotency_key="custom-456")
    assert distinct["episode_id"] != first["episode_id"]


def test_semantic_payload_fields_remain_distinct() -> None:
    base: dict[str, Any] = dict(
        terminal_outcome="SUCCEEDED", terminal_evidence={"verifier_status": "PASS"}
    )
    task = build_nexus_learning_episode(
        task_id="task-d1", attempt_id="a", action_id="c", source="router", **base
    )
    attempt = build_nexus_learning_episode(
        task_id="task-d1", attempt_id="b", action_id="c", source="router", **base
    )
    action = build_nexus_learning_episode(
        task_id="task-d1", attempt_id="a", action_id="d", source="router", **base
    )
    source = build_nexus_learning_episode(
        task_id="task-d1", attempt_id="a", action_id="c", source="learn_mode", **base
    )
    ids = {item["episode_id"] for item in (task, attempt, action, source)}
    assert len(ids) == 4


def test_lesson_ordering_normalization_is_deterministic() -> None:
    kwargs: dict[str, Any] = dict(
        task_id="task-order",
        terminal_outcome="SUCCEEDED",
        terminal_evidence={"verifier_status": "PASS"},
        retrieved_lesson_ids=["lesson-b", "lesson-a"],
        applied_lesson_ids=["lesson-a"],
    )
    reversed_kwargs: dict[str, Any] = dict(
        task_id="task-order",
        terminal_outcome="SUCCEEDED",
        terminal_evidence={"verifier_status": "PASS"},
        retrieved_lesson_ids=["lesson-a", "lesson-b"],
        applied_lesson_ids=["lesson-a"],
    )
    first = build_nexus_learning_episode(**kwargs)
    second = build_nexus_learning_episode(**reversed_kwargs)
    assert first["episode_id"] == second["episode_id"]
    assert first["retrieved_lesson_ids"] == ["lesson-a", "lesson-b"]
    assert first["applied_lesson_ids"] == second["applied_lesson_ids"] == ["lesson-a"]


def test_extra_authority_fields_fail_closed() -> None:
    episode = build_nexus_learning_episode(
        task_id="task-auth",
        terminal_outcome="SUCCEEDED",
        terminal_evidence={"verifier_status": "PASS"},
    )
    boosted = dict(episode, authority="approve", promotion_status="promote")
    validate_nexus_learning_episode(boosted)
    assert boosted["qualification_status"] == "UNQUALIFIED"
    assert boosted["stages"]["outcome_uplift_observed"] is False

    still_tampered = dict(boosted, episode_id=f"lep:{'1' * 24}")
    with pytest.raises(ValueError, match="IDENTITY_MISMATCH"):
        validate_nexus_learning_episode(still_tampered)


def test_applied_lessons_are_bounded_by_retrieval() -> None:
    episode = build_nexus_learning_episode(
        task_id="task-subset",
        retrieved_lesson_ids=["known"],
        applied_lesson_ids=["known", "forged"],
    )
    assert episode["applied_lesson_ids"] == ["known"]


def test_episode_without_terminal_evidence_cannot_claim_uplift() -> None:
    episode = build_nexus_learning_episode(
        task_id="task-2",
        attempt_id="attempt-1",
        action_id="action-1",
        source="router",
        terminal_outcome="SUCCEEDED",
        qualification={"uplift_observed": True},
    )
    assert episode["qualification_status"] == "UNQUALIFIED"
    assert episode["stages"]["outcome_uplift_observed"] is False

    status_only = build_nexus_learning_episode(
        task_id="task-status-only",
        terminal_outcome="SUCCEEDED",
        terminal_evidence={"status": "PASS"},
    )
    assert status_only["stages"]["outcome_measured"] is False
    episode["stages"]["outcome_uplift_observed"] = True
    with pytest.raises(ValueError, match="UPLIFT_WITHOUT_OUTCOME"):
        validate_nexus_learning_episode(episode)


def test_uplift_requires_paired_memory_verifiers_with_same_fingerprint() -> None:
    base = dict(
        task_id="paired",
        terminal_outcome="SUCCEEDED",
        qualification={
            "repeatability": True,
            "prevention_rule": "rule",
            "authority_qualification": True,
            "uplift_observed": True,
        },
    )
    missing = build_nexus_learning_episode(**base, terminal_evidence={"verifier_status": "missing"})
    assert missing["stages"]["outcome_uplift_observed"] is False
    paired = {
        "task_fingerprint": "fp-1",
        "memory_off": {
            "task_fingerprint": "fp-1",
            "verifier_status": "fail",
            "artifact": "off.json",
        },
        "memory_on": {"task_fingerprint": "fp-1", "verifier_status": "pass", "artifact": "on.json"},
    }
    observed = build_nexus_learning_episode(**base, terminal_evidence={"paired_verifier": paired})
    assert observed["stages"]["outcome_uplift_observed"] is True
    mismatched = dict(paired, memory_on=dict(paired["memory_on"], task_fingerprint="other"))
    rejected = build_nexus_learning_episode(
        **base, terminal_evidence={"paired_verifier": mismatched}
    )
    assert rejected["stages"]["outcome_uplift_observed"] is False


def test_outcome_memory_derives_stable_episode_id_and_keeps_evidence() -> None:
    record = EpisodeOutcomeRecord.from_task(
        task_id="task-3",
        task_type="bug",
        task_desc="desc",
        solved=True,
        wall_duration_sec=1,
        total_tokens_used=2,
        trust_mismatch=False,
        attempt_id="attempt-1",
        action_id="action-1",
        source_schema="nexus.learning_episode.v1",
        terminal_evidence={"verifier_status": "PASS"},
        receipts=[
            {
                "name": "repair",
                "selected": True,
                "invoked": True,
                "evidence_present": True,
                "gate_passed": True,
            }
        ],
    )
    assert record.episode_id.startswith("lep:")
    assert record.qualification_evidence_present is True
    assert record.stages["outcome_measured"] is True


def test_outcome_memory_bounds_applied_lessons_and_uplift_to_evidence() -> None:
    record = EpisodeOutcomeRecord.from_task(
        task_id="task-4",
        task_type="bug",
        task_desc="desc",
        solved=True,
        wall_duration_sec=1,
        total_tokens_used=2,
        trust_mismatch=False,
        retrieved_lesson_ids=["kept"],
        applied_lesson_ids=["kept", "forged"],
        terminal_evidence={"verifier_status": "pass"},
        stages={"outcome_uplift_observed": True},
    )
    assert record.applied_lesson_ids == ["kept"]
    assert record.stages["outcome_uplift_observed"] is False
