"""Policy and state writes must replace files atomically; a failed replace keeps prior content."""

import json
import os
from pathlib import Path

import pytest

from nexus_learning._atomic_io import _atomic_write_text
from nexus_learning.adoption import AdoptionStore
from nexus_learning.contracts import (
    build_learning_experience,
    build_promoted_learning_policy,
    save_promoted_learning_policy,
)
from nexus_learning.outcome_memory import OutcomeMemoryManager
from nexus_learning.state_root import resolve_learning_state_root


def _experience(task_id: str, capability: str):
    return build_learning_experience(
        task_id=task_id,
        task_type="bug",
        usage_trace={
            "phase_trace": {"S": "start", "P": "plan", "X": "context", "D": "design",
                            "R": "repair", "A": "audit", "C": "close"},
            "capabilities": {
                "artifact_gate_passed": True,
                "artifact_refs": [f"artifact:{task_id}"],
                "claim_verified": True,
                "delivery_gate_passed": True,
                "delivery_refs": [f"delivery:{task_id}"],
            },
        },
        capability_receipts=[
            {
                "name": capability,
                "selected": True,
                "invoked": True,
                "evidence_present": True,
                "gate_passed": True,
                "outcome_contributed": True,
                "evidence_refs": [f"{capability}:{task_id}"],
            },
        ],
    )


def _fail_replace(monkeypatch):
    def boom(src, dst):
        raise OSError("simulated crash before replace")

    monkeypatch.setattr(os, "replace", boom)


def _leftover_temp_files(directory: Path, name: str) -> list[Path]:
    return sorted(directory.glob(f".{name}.*"))


def test_atomic_write_text_replaces_content_and_leaves_no_temp_files(tmp_path):
    target = tmp_path / "policy.json"
    target.write_text("old", encoding="utf-8")

    _atomic_write_text(target, "new\n")

    assert target.read_text(encoding="utf-8") == "new\n"
    assert _leftover_temp_files(tmp_path, "policy.json") == []


def test_atomic_write_text_keeps_prior_file_when_replace_fails(tmp_path, monkeypatch):
    target = tmp_path / "policy.json"
    target.write_text("old", encoding="utf-8")
    _fail_replace(monkeypatch)

    with pytest.raises(OSError):
        _atomic_write_text(target, "new")

    assert target.read_text(encoding="utf-8") == "old"
    assert _leftover_temp_files(tmp_path, "policy.json") == []


def test_outcome_memory_dynamic_policy_write_is_atomic(tmp_path, monkeypatch):
    OutcomeMemoryManager.run_dynamic_autotune_sync(project_root=tmp_path, allow_dev_cwd_fallback=False)
    policy_path = resolve_learning_state_root(tmp_path, allow_dev_cwd_fallback=False).dynamic_policy_path
    assert policy_path.is_file(), "dynamic policy file was not written"
    before = policy_path.read_bytes()
    _fail_replace(monkeypatch)

    with pytest.raises(OSError):
        OutcomeMemoryManager.run_dynamic_autotune_sync(project_root=tmp_path, allow_dev_cwd_fallback=False)

    assert policy_path.read_bytes() == before


def test_promoted_learning_policy_save_is_atomic(tmp_path, monkeypatch):
    first = [_experience("task-a", "codeintel")]
    second = [_experience("task-b", "swarm")]
    assert build_promoted_learning_policy(second) != build_promoted_learning_policy(first)
    path = tmp_path / "promoted_learning_policy.json"
    save_promoted_learning_policy(path, first)
    before = path.read_bytes()
    _fail_replace(monkeypatch)

    with pytest.raises(OSError):
        save_promoted_learning_policy(path, second)

    assert path.read_bytes() == before
    assert _leftover_temp_files(tmp_path, path.name) == []


def test_adoption_store_write_still_round_trips_with_shared_helper(tmp_path):
    target = tmp_path / "adoption.json"
    AdoptionStore._write(target, {"b": 2, "a": "x"})

    assert json.loads(target.read_text(encoding="utf-8")) == {"a": "x", "b": 2}
    assert _leftover_temp_files(tmp_path, "adoption.json") == []
