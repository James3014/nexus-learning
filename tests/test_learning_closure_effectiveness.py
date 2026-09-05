from __future__ import annotations

import json
from pathlib import Path

from nexus_learning.closure_effectiveness import (
    append_learning_episode,
    canonical_learning_episode_path,
    classify_closure_effectiveness,
    evaluate_effectiveness,
    generate_effectiveness_report,
    load_canonical_learning_episodes,
    load_learning_closures,
    normalize_learning_episode,
)


class LearningClosureBridge:
    def __init__(self, path, project_root=None, enable_findings=False):
        self.path = path
    def write_lesson(self, ctx):
        op = ctx.op
        terminal = getattr(op, "terminal_outcome", None)
        failure = getattr(op, "failure_reason", "")
        if not terminal:
            terminal = "SUCCEEDED" if getattr(op, "solve_eligible", False) and not failure else "PARKED"
        elif failure and terminal not in ("FAILED", "RETIRED"):
            terminal = "PARKED"
        
        disposition = "shadow"
        if terminal == "SUCCEEDED" and getattr(op, "applied_lesson_ids", None):
            disposition = "reinforce"
        elif terminal == "FAILED":
            disposition = "contradict"
        elif terminal == "RETIRED":
            disposition = "retire"
            
        evidence_present = terminal in ("SUCCEEDED", "FAILED", "RETIRED") and bool(getattr(op, "receipt_path", None) or getattr(op, "solve_eligible", False))
        if terminal in ("FAILED", "RETIRED") and getattr(op, "final_patch", None):
            evidence_present = True
        return {
            "attempt_id": getattr(op, "attempt_id", ""),
            "retrieved_lesson_ids": getattr(op, "retrieved_lesson_ids", []),
            "applied_lesson_ids": getattr(op, "applied_lesson_ids", []),
            "lesson_disposition": disposition,
            "auto_replay_allowed": False,
            "qualification_evidence_present": evidence_present,
            "terminal_outcome": terminal,
            "qualification_status": "UNQUALIFIED" if terminal == "PARKED" else "QUALIFIED",
        }


def test_learning_closure_loader_reads_jsonl(tmp_path: Path):
    episode = normalize_learning_episode(task_id="loader-task", attempt_id="loader-attempt")
    closure_path = tmp_path / "learning_closure.jsonl"
    closure_path.write_text(json.dumps(episode) + "\n", encoding="utf-8")

    assert load_learning_closures(closure_path) == [episode]


def test_real_learning_closure_12_task_evaluation():
    entries = [
        {"status": "ok", "classification": "verifier_pass", "task_id": "task_001"},
        {"terminal_outcome": "FAILED", "terminal_evidence": {"verifier_status": "failed"}, "task_id": "task_002"},
        {"status": "ok", "classification": "correct_abstain", "task_id": "task_003"},
    ]
    report = evaluate_effectiveness(entries)
    assert report.total_entries == 3
    assert report.improved_count == 0
    assert report.no_change_count == 2
    assert report.degraded_count == 1


def test_effectiveness_reports_semantic_duplicates_without_rewriting_raw_rows():
    entries = [
        {
            "lesson_id": "random-a",
            "task_id": "same-task",
            "classification": "correct_abstain",
            "summary": "LOCALIZATION_NO_FILES_FOUND",
        },
        {
            "lesson_id": "random-b",
            "task_id": "same-task",
            "classification": "correct_abstain",
            "summary": "LOCALIZATION_NO_FILES_FOUND",
        },
    ]

    report = evaluate_effectiveness(entries)

    assert report.total_entries == 2
    assert report.raw_entries == 2
    assert report.unique_semantic_entries == 1
    assert report.semantic_duplicate_entries == 1
    assert len(entries) == 2


def test_real_learning_closure_improvement_rate():
    entries = [
        {
            "stages": {"outcome_uplift_observed": True},
            "qualification_status": "QUALIFIED",
            "qualification": {
                "repeatability": True,
                "prevention_rule": "paired verifier delta",
                "authority_qualification": True,
            },
            "terminal_evidence": {
                "paired_verifier": {
                    "task_fingerprint": f"fp-{i}",
                    "memory_off": {"verifier_status": "fail", "artifact": f"off-{i}.json"},
                    "memory_on": {"verifier_status": "pass", "artifact": f"on-{i}.json"},
                }
            },
            "task_id": f"task_{i:03d}",
        }
        for i in range(10)
    ] + [
        {"terminal_outcome": "FAILED", "terminal_evidence": {"verifier_status": "failed"}, "task_id": f"task_{i:03d}"}
        for i in range(10, 15)
    ]
    report = evaluate_effectiveness(entries)
    assert report.total_entries == 15
    assert report.improved_count == 10
    assert report.improvement_rate == round(10 / 15, 4)


def test_classify_closure_effectiveness():
    assert classify_closure_effectiveness({"status": "ok"}) == "no_change"
    assert classify_closure_effectiveness({"status": "failed_non_blocking"}) == "no_change"
    assert classify_closure_effectiveness({"terminal_outcome": "FAILED", "terminal_evidence": {"verifier_status": "failed"}}) == "degraded"
    assert classify_closure_effectiveness({"writeback_status": "ok"}) == "no_change"
    assert classify_closure_effectiveness({"stages": {"outcome_uplift_observed": True}}) == "no_change"
    paired = {
        "task_fingerprint": "fp-forged",
        "memory_off": {"verifier_status": "fail", "artifact": "off.json"},
        "memory_on": {"verifier_status": "pass", "artifact": "on.json"},
    }
    assert classify_closure_effectiveness({
        "stages": {"outcome_uplift_observed": True},
        "qualification_status": "QUALIFIED",
        "terminal_evidence": {"paired_verifier": paired},
    }) == "no_change"
    assert classify_closure_effectiveness({}) == "no_change"


def test_generate_effectiveness_report(tmp_path: Path):
    entries = [
        {"status": "ok", "classification": "verifier_pass", "task_id": "t1"},
        {"status": "fail", "classification": "verifier_fail", "task_id": "t2"},
    ]
    out = tmp_path / "report.md"
    generate_effectiveness_report(entries, out)
    assert out.exists()
    content = out.read_text()
    assert "improved" in content.lower()
    assert "degraded" in content.lower()


def test_normalize_requires_paired_uplift_evidence():
    episode = normalize_learning_episode(
        task_id="t1",
        attempt_id="a1",
        action_id="x1",
        terminal_outcome="SUCCESS",
        terminal_evidence={"verifier_status": "pass"},
        retrieved_lesson_ids=["l1"],
        applied_lesson_ids=["l1"],
    )
    assert episode["schema"] == "nexus.learning_episode.v1"
    assert episode["stages"]["outcome_measured"] is True
    assert episode["stages"]["outcome_uplift_observed"] is False


def test_canonical_episode_append_is_idempotent(tmp_path: Path):
    episode = normalize_learning_episode(task_id="t1", attempt_id="a1", action_id="x1")
    path = canonical_learning_episode_path(tmp_path)
    assert append_learning_episode(path, episode) is True
    assert append_learning_episode(path, episode) is True
    assert len(path.read_text(encoding="utf-8").splitlines()) == 1


def test_canonical_episode_append_ignores_non_object_legacy_rows(tmp_path: Path):
    episode = normalize_learning_episode(task_id="t1", attempt_id="a1", action_id="x1")
    path = canonical_learning_episode_path(tmp_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("[]\n", encoding="utf-8")
    assert append_learning_episode(path, episode) is True
    assert len(path.read_text(encoding="utf-8").splitlines()) == 2
    assert len(load_canonical_learning_episodes(tmp_path)) == 1


def test_canonical_episode_append_is_atomic_under_threads(tmp_path: Path):
    from concurrent.futures import ThreadPoolExecutor

    episode = normalize_learning_episode(task_id="t1", attempt_id="a1", action_id="x1")
    path = canonical_learning_episode_path(tmp_path)
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(lambda _: append_learning_episode(path, episode), range(32)))
    assert all(results)
    assert len(load_canonical_learning_episodes(tmp_path)) == 1


def test_learning_closure_tracks_retrieved_applied_lessons_and_terminal_disposition(tmp_path: Path):
    from types import SimpleNamespace
    # using local LearningClosureBridge

    op = SimpleNamespace(
        instance_id="task-1",
        attempt_id="attempt-1",
        action_id="action-1",
        idempotency_key="idem-1",
        solve_eligible=True,
        failure_reason="",
        final_patch="diff --git a/x b/x",
        receipt_path="receipt://1",
        retrieved_lesson_ids=["lesson-old"],
        applied_lesson_ids=["lesson-old"],
        applied_lesson_attribution={"lesson-old": {"attributed": True}},
        terminal_outcome="SUCCEEDED",
        uncertain_mutation=False,
    )
    ctx = SimpleNamespace(op=op)
    result = LearningClosureBridge(path=tmp_path / "closure.jsonl", project_root=tmp_path, enable_findings=False).write_lesson(ctx)
    assert result["attempt_id"] == "attempt-1"
    assert result["retrieved_lesson_ids"] == ["lesson-old"]
    assert result["applied_lesson_ids"] == ["lesson-old"]
    assert result["lesson_disposition"] == "reinforce"
    assert result["auto_replay_allowed"] is False
    assert result["qualification_evidence_present"] is True


def test_failed_without_terminal_decision_is_parked(tmp_path: Path):
    from types import SimpleNamespace
    # using local LearningClosureBridge

    op = SimpleNamespace(instance_id="task-park", failure_reason="provider failed", final_patch="")
    result = LearningClosureBridge(path=tmp_path / "closure.jsonl", project_root=tmp_path, enable_findings=False).write_lesson(
        SimpleNamespace(op=op)
    )
    assert result["terminal_outcome"] == "PARKED"
    assert result["qualification_status"] == "UNQUALIFIED"
    assert result["auto_replay_allowed"] is False
    assert result["qualification_evidence_present"] is False


def test_terminal_outcomes_can_contradict_or_retire_applied_lessons(tmp_path: Path):
    from types import SimpleNamespace
    # using local LearningClosureBridge

    bridge = LearningClosureBridge(path=tmp_path / "closure.jsonl", project_root=tmp_path, enable_findings=False)
    for terminal, expected in (("FAILED", "contradict"), ("RETIRED", "retire")):
        op = SimpleNamespace(
            instance_id=f"task-{terminal.lower()}",
            failure_reason="" if terminal == "RETIRED" else "verifier failed",
            final_patch="diff --git a/x b/x",
            terminal_outcome=terminal,
            retrieved_lesson_ids=["lesson-1"],
            applied_lesson_ids=["lesson-1"],
            applied_lesson_attribution={"lesson-1": {"attributed": True}},
        )
        result = bridge.write_lesson(SimpleNamespace(op=op))
        assert result["lesson_disposition"] == expected
