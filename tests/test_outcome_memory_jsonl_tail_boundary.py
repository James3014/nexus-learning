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
    third = _record("C", "key-C")

    OutcomeMemoryManager.save_episode_and_tune_sync(first, project_root=state_root)
    storage = state_root.outcome_history_path
    storage.write_bytes(
        storage.read_bytes()
        + (json.dumps(second.to_dict(), sort_keys=True).encode() + b"\n").rstrip(b"\n")
    )
    result = OutcomeMemoryManager.save_episode_and_tune_sync(third, project_root=state_root)

    rows = OutcomeMemoryManager.load_recent_records(project_root=state_root)
    assert [row["task_id"] for row in rows] == ["A", "B", "C"]
    assert result["policy"]["source_experiences"] == ["A", "B", "C"]
    policy = json.loads(state_root.dynamic_policy_path.read_text())
    assert policy["source_experiences"] == ["A", "B", "C"]


def test_duplicate_retry_after_separator_does_not_append_second_record(tmp_path: Path) -> None:
    state_root = LearningStateRoot.from_project_root(tmp_path)
    record = _record("C", "key-C")
    storage = state_root.outcome_history_path
    storage.parent.mkdir(parents=True)
    storage.write_bytes(b'{"task_id":"A"}')

    OutcomeMemoryManager.save_episode_and_tune_sync(record, project_root=state_root)
    retry = OutcomeMemoryManager.save_episode_and_tune_sync(record, project_root=state_root)

    rows = OutcomeMemoryManager.load_recent_records(project_root=state_root)
    assert [row["task_id"] for row in rows] == ["A", "C"]
    assert retry["status"] == "IDEMPOTENT_DUPLICATE"


def test_worker_append_accepts_valid_unterminated_object_tail(tmp_path: Path) -> None:
    state_root = LearningStateRoot.from_project_root(tmp_path)
    storage = state_root.outcome_history_path
    storage.parent.mkdir(parents=True)
    storage.write_bytes(b'{"task_id":"A"}')

    result = OutcomeMemoryManager.append_worker_write(
        {"task_id": "B", "worker_name": "w1"}, project_root=state_root
    )

    assert result["status"] == "PASS"
    rows = OutcomeMemoryManager.load_recent_records(project_root=state_root)
    assert [row["task_id"] for row in rows] == ["A", "B"]


def test_worker_appends_are_serialized_by_state_root_lock(tmp_path: Path) -> None:
    import threading

    state_root = LearningStateRoot.from_project_root(tmp_path)
    receipts = [
        {"task_id": f"task-{index}", "worker_name": f"worker-{index}"} for index in range(8)
    ]
    errors: list[BaseException] = []

    def append(receipt: dict[str, str]) -> None:
        try:
            OutcomeMemoryManager.append_worker_write(receipt, project_root=state_root)
        except BaseException as exc:  # pragma: no cover - assertion below reports failures
            errors.append(exc)

    threads = [threading.Thread(target=append, args=(receipt,)) for receipt in receipts]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert errors == []
    rows = OutcomeMemoryManager.load_recent_records(project_root=state_root, limit=20)
    assert {row["task_id"] for row in rows} == {receipt["task_id"] for receipt in receipts}


@pytest.mark.parametrize("tail", [b'{"task_id":"a"', b"not-json", b"[]"])
def test_partial_or_corrupt_tail_fails_closed_without_mutation_or_policy(
    tmp_path: Path, tail: bytes
) -> None:
    state_root = LearningStateRoot.from_project_root(tmp_path)
    storage = state_root.outcome_history_path
    storage.parent.mkdir(parents=True)
    storage.write_bytes(tail)
    policy_before = b'{"status":"existing-policy"}\n'
    state_root.dynamic_policy_path.write_bytes(policy_before)
    before = storage.read_bytes()

    with pytest.raises(ValueError, match="OUTCOME_HISTORY_TAIL_INVALID"):
        OutcomeMemoryManager.save_episode_and_tune_sync(
            _record("new", "new-key"), project_root=state_root
        )

    assert storage.read_bytes() == before
    assert b'"new"' not in storage.read_bytes()
    assert state_root.dynamic_policy_path.read_bytes() == policy_before


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
