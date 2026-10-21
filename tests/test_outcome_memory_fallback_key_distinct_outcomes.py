"""Outcomes recorded without attempt/action ids must not collapse per task.

Real callers (router, research flow, executor registry) record outcomes with no
attempt id, action id or explicit idempotency key. Each later outcome of the same
task is a new outcome and must be kept; only a true duplicate (the same record
written again, or a record with the same explicit key / attempt+action ids) may be
reported as IDEMPOTENT_DUPLICATE.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

import nexus_learning.outcome_memory as outcome_memory
from nexus_learning.outcome_memory import (
    EpisodeOutcomeRecord,
    OutcomeMemoryManager,
    build_episode_from_receipts,
)

_BASE_TIME = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)


@pytest.fixture(autouse=True)
def _stepping_clock(monkeypatch: pytest.MonkeyPatch) -> None:
    """Deterministic clock: every now() call is one second after the previous one."""
    ticks = {"n": 0}

    class _SteppingDatetime(datetime):
        @classmethod
        def now(cls, tz: Any = None) -> datetime:  # type: ignore[override]
            ticks["n"] += 1
            value = _BASE_TIME + timedelta(seconds=ticks["n"])
            return value if tz is not None else value.replace(tzinfo=None)

    monkeypatch.setattr(outcome_memory, "datetime", _SteppingDatetime)


def _rows(root: Path) -> list[dict[str, Any]]:
    storage = root / ".nexus" / "memory" / "outcome_history.jsonl"
    if not storage.exists():
        return []
    return [
        json.loads(line)
        for line in storage.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _episode_rows(root: Path, task_id: str) -> list[dict[str, Any]]:
    return [
        row
        for row in _rows(root)
        if row.get("task_id") == task_id
        and row.get("schema_version") == outcome_memory.OUTCOME_MEMORY_SCHEMA
    ]


def _no_id_record(*, solved: bool, tokens: int, receipt_name: str) -> EpisodeOutcomeRecord:
    # Mirrors research_flow_service / executor registry: no ids, no explicit key.
    return EpisodeOutcomeRecord.from_task(
        task_id="same-task",
        task_type="repair",
        task_desc="repeated task without attempt or action ids",
        solved=solved,
        wall_duration_sec=1.5 if solved else 3.0,
        total_tokens_used=tokens,
        trust_mismatch=False,
        receipts=[
            {
                "name": receipt_name,
                "selected": True,
                "invoked": True,
                "evidence_present": True,
                "gate_passed": solved,
            }
        ],
    )


def test_two_outcomes_of_same_task_without_ids_are_both_kept(tmp_path: Path) -> None:
    first = _no_id_record(solved=False, tokens=120, receipt_name="cap.alpha")
    second = _no_id_record(solved=True, tokens=80, receipt_name="cap.beta")

    first_result = OutcomeMemoryManager.save_episode_and_tune_sync(first, project_root=tmp_path)
    second_result = OutcomeMemoryManager.save_episode_and_tune_sync(second, project_root=tmp_path)

    assert first_result["status"] == "PASS"
    assert second_result["status"] == "PASS", (
        "a later outcome of the same task without ids was dropped as a duplicate"
    )
    rows = _episode_rows(tmp_path, "same-task")
    assert len(rows) == 2
    assert [row["solved"] for row in rows] == [False, True]
    assert [row["total_tokens_used"] for row in rows] == [120, 80]


def test_three_outcomes_of_same_task_without_ids_are_all_kept(tmp_path: Path) -> None:
    statuses = [
        OutcomeMemoryManager.save_episode_and_tune_sync(
            _no_id_record(solved=solved, tokens=tokens, receipt_name=name),
            project_root=tmp_path,
        )["status"]
        for solved, tokens, name in (
            (False, 300, "cap.one"),
            (False, 200, "cap.two"),
            (True, 100, "cap.three"),
        )
    ]

    assert statuses == ["PASS", "PASS", "PASS"]
    rows = _episode_rows(tmp_path, "same-task")
    assert [row["total_tokens_used"] for row in rows] == [300, 200, 100]


@dataclass
class _Receipt:
    capability_name: str
    gate_passed: bool
    evidence_alignment: bool = True
    telemetries: dict[str, Any] = field(default_factory=dict)


def test_router_style_receipt_episodes_of_same_task_are_both_kept(tmp_path: Path) -> None:
    # Mirrors the router: build_episode_from_receipts with no ids, then save.
    failed = build_episode_from_receipts(
        task_id="router-task",
        task_type="execute",
        task_desc="router learning loop",
        receipts=[
            _Receipt(
                "cap.search",
                gate_passed=False,
                telemetries={"wall_time_ms": 4000, "token_usage": 50},
            )
        ],
    )
    succeeded = build_episode_from_receipts(
        task_id="router-task",
        task_type="execute",
        task_desc="router learning loop",
        receipts=[
            _Receipt(
                "cap.search",
                gate_passed=True,
                telemetries={"wall_time_ms": 2000, "token_usage": 30},
            )
        ],
    )

    results = [
        OutcomeMemoryManager.save_episode_and_tune_sync(episode, project_root=tmp_path)["status"]
        for episode in (failed, succeeded)
    ]

    assert results == ["PASS", "PASS"]
    rows = _episode_rows(tmp_path, "router-task")
    assert [row["solved"] for row in rows] == [False, True]


def test_same_no_id_record_written_twice_is_still_deduplicated(tmp_path: Path) -> None:
    record = _no_id_record(solved=True, tokens=42, receipt_name="cap.retry")

    first = OutcomeMemoryManager.save_episode_and_tune_sync(record, project_root=tmp_path)
    retry = OutcomeMemoryManager.save_episode_and_tune_sync(record, project_root=tmp_path)

    assert [first["status"], retry["status"]] == ["PASS", "IDEMPOTENT_DUPLICATE"]
    assert len(_episode_rows(tmp_path, "same-task")) == 1


def test_explicit_idempotency_key_still_deduplicates(tmp_path: Path) -> None:
    def make(solved: bool, tokens: int) -> EpisodeOutcomeRecord:
        return EpisodeOutcomeRecord.from_task(
            task_id="keyed-task",
            task_type="repair",
            task_desc="explicit key",
            solved=solved,
            wall_duration_sec=1.0,
            total_tokens_used=tokens,
            trust_mismatch=False,
            idempotency_key="explicit-key-1",
        )

    first = OutcomeMemoryManager.save_episode_and_tune_sync(make(False, 10), project_root=tmp_path)
    again = OutcomeMemoryManager.save_episode_and_tune_sync(make(True, 20), project_root=tmp_path)

    assert [first["status"], again["status"]] == ["PASS", "IDEMPOTENT_DUPLICATE"]
    rows = _episode_rows(tmp_path, "keyed-task")
    assert len(rows) == 1
    assert rows[0]["total_tokens_used"] == 10


def test_same_attempt_and_action_ids_still_deduplicate_when_rebuilt(tmp_path: Path) -> None:
    def make() -> EpisodeOutcomeRecord:
        return EpisodeOutcomeRecord.from_task(
            task_id="attempt-task",
            task_type="repair",
            task_desc="attempt and action ids",
            solved=True,
            wall_duration_sec=1.0,
            total_tokens_used=10,
            trust_mismatch=False,
            attempt_id="attempt-7",
            action_id="action-3",
        )

    first = OutcomeMemoryManager.save_episode_and_tune_sync(make(), project_root=tmp_path)
    rebuilt = OutcomeMemoryManager.save_episode_and_tune_sync(make(), project_root=tmp_path)

    assert [first["status"], rebuilt["status"]] == ["PASS", "IDEMPOTENT_DUPLICATE"]
    assert len(_episode_rows(tmp_path, "attempt-task")) == 1
