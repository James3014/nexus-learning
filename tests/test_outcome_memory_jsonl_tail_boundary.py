from __future__ import annotations

import json
from pathlib import Path

import pytest

from nexus_learning.outcome_memory import EpisodeOutcomeRecord, OutcomeMemoryManager
from nexus_learning.state_root import LearningStateRoot


def _record(task_id: str, idempotency_key: str) -> EpisodeOutcomeRecord:
    return EpisodeOutcomeRecord.from_task(
        task_id=task_id,
        task_type="repair",
        task_desc="jsonl tail boundary",
        solved=True,
        wall_duration_sec=1.0,
        total_tokens_used=1,
        trust_mismatch=False,
        idempotency_key=idempotency_key,
        qualification_evidence_present=True,
    )


@pytest.mark.parametrize("tail", [b"", b"\n", b'{"task_id":"a"}\n', b'{"task_id":"a"}'])
def test_complete_jsonl_tail_allows_append(tmp_path: Path, tail: bytes) -> None:
    state_root = LearningStateRoot.from_project_root(tmp_path)
    storage = state_root.outcome_history_path
    storage.parent.mkdir(parents=True)
    storage.write_bytes(tail)

    result = OutcomeMemoryManager.save_episode_and_tune_sync(
        _record("new", "new-key"), project_root=state_root
    )

    assert result["status"] == "PASS"
    rows = [json.loads(line) for line in storage.read_text().splitlines() if line.strip()]
    if tail.strip():
        assert rows[0]["task_id"] == "a"
    assert rows[-1]["task_id"] == "new"


def test_valid_missing_newline_preserves_history_and_policy_contents(tmp_path: Path) -> None:
    state_root = LearningStateRoot.from_project_root(tmp_path)
    first = _record("A", "key-A")
    second = _record("B", "key-B")

    OutcomeMemoryManager.save_episode_and_tune_sync(first, project_root=state_root)
    storage = state_root.outcome_history_path
    storage.write_bytes(storage.read_bytes().rstrip(b"\n"))
    result = OutcomeMemoryManager.save_episode_and_tune_sync(second, project_root=state_root)

    rows = [json.loads(line) for line in storage.read_text().splitlines() if line.strip()]
    assert [row["task_id"] for row in rows] == ["A", "B"]
    assert result["policy"]["source_experiences"] == ["A", "B"]
    policy = json.loads(state_root.dynamic_policy_path.read_text())
    assert policy["source_experiences"] == ["A", "B"]


@pytest.mark.parametrize("tail", [b'{"task_id":"a"', b"not-json", b"[]\n"])
def test_partial_or_corrupt_tail_fails_closed_without_mutation_or_policy(
    tmp_path: Path, tail: bytes
) -> None:
    state_root = LearningStateRoot.from_project_root(tmp_path)
    storage = state_root.outcome_history_path
    storage.parent.mkdir(parents=True)
    storage.write_bytes(tail)
    before = storage.read_bytes()

    with pytest.raises(ValueError, match="OUTCOME_HISTORY_TAIL_INVALID"):
        OutcomeMemoryManager.save_episode_and_tune_sync(
            _record("new", "new-key"), project_root=state_root
        )

    assert storage.read_bytes() == before
    assert b'"new"' not in storage.read_bytes()
    assert not state_root.dynamic_policy_path.exists()


def test_worker_append_uses_same_tail_guard_and_lock(tmp_path: Path) -> None:
    state_root = LearningStateRoot.from_project_root(tmp_path)
    storage = state_root.outcome_history_path
    storage.parent.mkdir(parents=True)
    storage.write_bytes(b'{"task_id":"partial"')
    before = storage.read_bytes()

    with pytest.raises(ValueError, match="OUTCOME_HISTORY_TAIL_INVALID"):
        OutcomeMemoryManager.append_worker_write(
            {"task_id": "new", "worker_name": "w1"}, project_root=state_root
        )

    assert storage.read_bytes() == before
