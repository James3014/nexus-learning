from __future__ import annotations

import json
import threading
from pathlib import Path

import pytest

from nexus_learning.outcome_memory import EpisodeOutcomeRecord, OutcomeMemoryManager
from nexus_learning.state_root import LearningStateRoot


def _record(task_id: str, idempotency_key: str) -> EpisodeOutcomeRecord:
    return EpisodeOutcomeRecord.from_task(
        task_id=task_id,
        task_type="repair",
        task_desc=f"policy publication ordering {task_id}",
        solved=True,
        wall_duration_sec=1.0,
        total_tokens_used=10,
        trust_mismatch=False,
        attempt_id=f"attempt-{task_id}",
        action_id=f"action-{task_id}",
        idempotency_key=idempotency_key,
        terminal_outcome="SUCCEEDED",
        qualification_evidence_present=True,
    )


def test_concurrent_policy_publication_cannot_overwrite_newer_history_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An older autotune snapshot must not publish after a newer committed snapshot."""
    state_root = LearningStateRoot.from_project_root(tmp_path)
    original_load = OutcomeMemoryManager.load_recent_records.__func__
    a_snapshot_loaded = threading.Event()
    b_completed = threading.Event()
    errors: list[BaseException] = []

    def controlled_load(cls, **kwargs):
        records = original_load(cls, **kwargs)
        if threading.current_thread().name == "episode-A":
            a_snapshot_loaded.set()
            # Pre-fix B can complete while A is paused here, making A's snapshot stale.
            # With state-root ordering fixed, B is blocked until A publishes and releases.
            b_completed.wait(timeout=1.0)
        return records

    monkeypatch.setattr(
        OutcomeMemoryManager,
        "load_recent_records",
        classmethod(controlled_load),
    )

    record_a = _record("episode-A", "idem-policy-order-A")
    record_b = _record("episode-B", "idem-policy-order-B")

    def write(record: EpisodeOutcomeRecord) -> None:
        try:
            OutcomeMemoryManager.save_episode_and_tune_sync(record, project_root=state_root)
        except BaseException as exc:  # pragma: no cover - asserted below
            errors.append(exc)
        finally:
            if record is record_b:
                b_completed.set()

    thread_a = threading.Thread(target=write, args=(record_a,), name="episode-A")
    thread_b = threading.Thread(target=write, args=(record_b,), name="episode-B")

    thread_a.start()
    assert a_snapshot_loaded.wait(timeout=2.0), "episode A never reached autotune snapshot"
    thread_b.start()
    thread_a.join(timeout=4.0)
    thread_b.join(timeout=4.0)

    assert not thread_a.is_alive()
    assert not thread_b.is_alive()
    assert errors == []

    rows = [
        json.loads(line)
        for line in state_root.outcome_history_path.read_text(encoding="utf-8").splitlines()
    ]
    assert [row["task_id"] for row in rows] == ["episode-A", "episode-B"]

    policy = json.loads(state_root.dynamic_policy_path.read_text(encoding="utf-8"))
    assert policy["source_experiences_count"] == 2
    assert policy["source_experiences"] == ["episode-A", "episode-B"]
