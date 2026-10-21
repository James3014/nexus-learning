"""#115 review: without ids, rebuilding the same outcome for a retry must still be a
duplicate, while two outcomes recorded at different times are kept apart."""

from __future__ import annotations

from pathlib import Path

from nexus_learning.outcome_memory import EpisodeOutcomeRecord, OutcomeMemoryManager


def _record(created_at: str, *, solved: bool = True) -> EpisodeOutcomeRecord:
    return EpisodeOutcomeRecord.from_task(
        task_id="retry-task",
        task_type="repair",
        task_desc="no attempt or action ids",
        solved=solved,
        wall_duration_sec=1.0,
        total_tokens_used=10,
        trust_mismatch=False,
        receipts=[{"name": "cap", "selected": True, "invoked": True, "gate_passed": solved}],
        created_at=created_at,
    )


def test_rebuilt_identical_outcome_is_a_duplicate(tmp_path: Path) -> None:
    at = "2026-01-01T12:00:00+00:00"
    first, retry = _record(at), _record(at)
    assert first.idempotency_key == retry.idempotency_key
    assert first.episode_id == retry.episode_id
    save = OutcomeMemoryManager.save_episode_and_tune_sync
    assert save(first, project_root=tmp_path)["status"] == "PASS"
    assert save(retry, project_root=tmp_path)["status"] == "IDEMPOTENT_DUPLICATE"


def test_same_content_at_another_time_is_another_outcome(tmp_path: Path) -> None:
    first = _record("2026-01-01T12:00:00+00:00")
    later = _record("2026-01-01T12:00:01+00:00")
    assert first.idempotency_key != later.idempotency_key
    save = OutcomeMemoryManager.save_episode_and_tune_sync
    assert save(first, project_root=tmp_path)["status"] == "PASS"
    assert save(later, project_root=tmp_path)["status"] == "PASS"
