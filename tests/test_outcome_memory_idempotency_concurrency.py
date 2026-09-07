from __future__ import annotations

import json
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import nexus_learning.outcome_memory as outcome_memory
from nexus_learning.outcome_memory import EpisodeOutcomeRecord, OutcomeMemoryManager


def _record() -> EpisodeOutcomeRecord:
    return EpisodeOutcomeRecord.from_task(
        task_id="concurrent-task",
        task_type="repair",
        task_desc="concurrent idempotency regression",
        solved=True,
        wall_duration_sec=1.0,
        total_tokens_used=10,
        trust_mismatch=False,
        attempt_id="attempt-1",
        action_id="action-1",
        idempotency_key="idem-concurrent-1",
        terminal_outcome="SUCCEEDED",
        qualification_evidence_present=True,
    )


def test_same_key_concurrent_writers_append_exactly_once(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """The idempotency read/check/append boundary must be atomic for one state root."""
    original_load = outcome_memory._load_idempotency_keys
    barrier = threading.Barrier(2)

    def synchronized_load(path: Path) -> set[str]:
        keys = original_load(path)
        try:
            barrier.wait(timeout=0.5)
        except threading.BrokenBarrierError:
            pass
        return keys

    monkeypatch.setattr(outcome_memory, "_load_idempotency_keys", synchronized_load)
    monkeypatch.setattr(
        OutcomeMemoryManager,
        "run_dynamic_autotune_sync",
        classmethod(lambda cls, **kwargs: {"status": "PASS"}),
    )

    record = _record()

    def write() -> dict[str, object]:
        return OutcomeMemoryManager.save_episode_and_tune_sync(record, project_root=tmp_path)

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: write(), range(2)))

    statuses = sorted(str(result["status"]) for result in results)
    assert statuses == ["IDEMPOTENT_DUPLICATE", "PASS"]

    storage = tmp_path / ".nexus" / "memory" / "outcome_history.jsonl"
    rows = [json.loads(line) for line in storage.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 1
    assert rows[0]["idempotency_key"] == record.idempotency_key
