from __future__ import annotations

import json
from pathlib import Path

import pytest

from nexus_learning.outcome_memory import EpisodeOutcomeRecord, OutcomeMemoryManager


def _record() -> EpisodeOutcomeRecord:
    return EpisodeOutcomeRecord.from_task(
        task_id="partial-failure-task",
        task_type="repair",
        task_desc="policy reconciliation regression",
        solved=True,
        wall_duration_sec=1.0,
        total_tokens_used=10,
        trust_mismatch=False,
        attempt_id="attempt-1",
        action_id="action-1",
        idempotency_key="idem-partial-failure-1",
        terminal_outcome="SUCCEEDED",
        qualification_evidence_present=True,
    )


def test_duplicate_retry_reconciles_policy_after_post_append_failure(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """A durable episode must not block recovery of a failed derived-policy update."""
    original = OutcomeMemoryManager.run_dynamic_autotune_sync.__func__
    calls = 0

    def fail_once(cls, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise OSError("injected policy publication failure")
        return original(cls, **kwargs)

    monkeypatch.setattr(
        OutcomeMemoryManager,
        "run_dynamic_autotune_sync",
        classmethod(fail_once),
    )

    record = _record()
    with pytest.raises(OSError, match="injected policy publication failure"):
        OutcomeMemoryManager.save_episode_and_tune_sync(record, project_root=tmp_path)

    storage = tmp_path / ".nexus" / "memory" / "outcome_history.jsonl"
    policy_path = tmp_path / ".nexus" / "memory" / "dynamic_learning_policy.json"
    rows = [json.loads(line) for line in storage.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 1
    assert rows[0]["idempotency_key"] == record.idempotency_key
    assert not policy_path.exists()
    assert calls == 1

    retry = OutcomeMemoryManager.save_episode_and_tune_sync(record, project_root=tmp_path)

    assert retry["status"] == "IDEMPOTENT_DUPLICATE"
    rows_after = [json.loads(line) for line in storage.read_text(encoding="utf-8").splitlines()]
    assert len(rows_after) == 1
    assert calls == 2
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    assert policy["status"] == "PASS"
    assert policy["source_experiences"] == [record.task_id]
