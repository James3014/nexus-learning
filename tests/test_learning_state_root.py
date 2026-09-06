from pathlib import Path

import pytest

from nexus_learning.outcome_memory import EpisodeOutcomeRecord, OutcomeMemoryManager
from nexus_learning.state_root import LearningStateRoot, resolve_learning_state_root


def test_state_root_paths_structure(tmp_path: Path):
    root = LearningStateRoot.from_project_root(tmp_path)
    assert root.nexus_dir == tmp_path / ".nexus"
    assert root.memory_dir == tmp_path / ".nexus" / "memory"
    assert root.audit_dir == tmp_path / ".nexus" / "audit"
    assert root.outcome_history_path == tmp_path / ".nexus" / "memory" / "outcome_history.jsonl"
    assert root.dynamic_policy_path == tmp_path / ".nexus" / "memory" / "dynamic_learning_policy.json"
    assert root.learning_episodes_path == tmp_path / ".nexus" / "memory" / "learning_episodes.jsonl"
    assert root.retrieval_log_path == tmp_path / ".nexus" / "audit" / "retrieval_log.jsonl"


def test_same_project_root_different_cwd_resolves_same_state(tmp_path: Path, monkeypatch):
    proj_root = tmp_path / "canonical_repo"
    proj_root.mkdir()
    cwd_a = tmp_path / "somewhere_a"
    cwd_a.mkdir()
    cwd_b = tmp_path / "somewhere_b"
    cwd_b.mkdir()

    record = EpisodeOutcomeRecord.from_task(
        task_id="t1",
        task_type="feature",
        task_desc="test description",
        solved=True,
        wall_duration_sec=1.0,
        total_tokens_used=100,
        trust_mismatch=False,
        terminal_outcome="SUCCEEDED",
        qualification_evidence_present=True,
    )

    # Run from cwd_a
    monkeypatch.chdir(cwd_a)
    assert Path.cwd() == cwd_a
    res_a = OutcomeMemoryManager.save_episode_and_tune_sync(record, project_root=proj_root)
    assert res_a["status"] == "PASS"

    # Verify written to proj_root/.nexus, NOT cwd_a
    expected_storage = proj_root / ".nexus" / "memory" / "outcome_history.jsonl"
    assert expected_storage.is_file()
    assert not (cwd_a / ".nexus").exists()

    # Read from cwd_b
    monkeypatch.chdir(cwd_b)
    assert Path.cwd() == cwd_b
    loaded = OutcomeMemoryManager.load_recent_records(project_root=proj_root)
    assert len(loaded) == 1
    assert loaded[0]["task_id"] == "t1"
    assert not (cwd_b / ".nexus").exists()


def test_different_project_roots_remain_isolated(tmp_path: Path):
    proj_1 = tmp_path / "proj1"
    proj_2 = tmp_path / "proj2"
    proj_1.mkdir()
    proj_2.mkdir()

    rec1 = EpisodeOutcomeRecord.from_task(
        task_id="proj1-task",
        task_type="feature",
        task_desc="desc1",
        solved=True,
        wall_duration_sec=1.0,
        total_tokens_used=100,
        trust_mismatch=False,
        terminal_outcome="SUCCEEDED",
        qualification_evidence_present=True,
    )
    rec2 = EpisodeOutcomeRecord.from_task(
        task_id="proj2-task",
        task_type="feature",
        task_desc="desc2",
        solved=True,
        wall_duration_sec=1.0,
        total_tokens_used=100,
        trust_mismatch=False,
        terminal_outcome="SUCCEEDED",
        qualification_evidence_present=True,
    )

    OutcomeMemoryManager.save_episode_and_tune_sync(rec1, project_root=proj_1)
    OutcomeMemoryManager.save_episode_and_tune_sync(rec2, project_root=proj_2)

    rows1 = OutcomeMemoryManager.load_recent_records(project_root=proj_1)
    rows2 = OutcomeMemoryManager.load_recent_records(project_root=proj_2)
    assert len(rows1) == 1 and rows1[0]["task_id"] == "proj1-task"
    assert len(rows2) == 1 and rows2[0]["task_id"] == "proj2-task"


def test_missing_project_root_fails_closed_without_dev_fallback(monkeypatch):
    monkeypatch.delenv("NEXUS_LEARNING_STATE_ROOT", raising=False)
    with pytest.raises(ValueError, match="Persistent Learning state requires an explicit project_root"):
        resolve_learning_state_root(None, allow_dev_cwd_fallback=False)


def test_environment_variable_override_state_root(tmp_path: Path, monkeypatch):
    configured = tmp_path / "configured_env_root"
    monkeypatch.setenv("NEXUS_LEARNING_STATE_ROOT", str(configured))
    root = resolve_learning_state_root(None)
    assert root.root == configured.resolve()


def test_explicit_dev_cwd_fallback_allowed(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("NEXUS_LEARNING_STATE_ROOT", raising=False)
    monkeypatch.chdir(tmp_path)
    root = resolve_learning_state_root(None, allow_dev_cwd_fallback=True)
    assert root.root == tmp_path.resolve()
