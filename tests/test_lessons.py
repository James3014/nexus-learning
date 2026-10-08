from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from nexus_learning import (
    EVIDENCE_ORIGIN_PHYSICAL,
    EVIDENCE_ORIGIN_SIMULATED,
    LEARNING_LESSON_SCHEMA,
    LessonStore,
    build_lesson,
    build_reflection_prompt,
    load_lessons,
    reflect_episodes,
    retrieve_lessons,
    validate_lesson,
)
from nexus_learning.closure_effectiveness import append_learning_episode
from nexus_learning.contracts import build_nexus_learning_episode
from nexus_learning.state_root import LearningStateRoot

LESSON_KEYS = {
    "schema",
    "lesson_id",
    "title",
    "lesson_body",
    "applies_when",
    "avoid_when",
    "outcome_polarity",
    "source_episode_ids",
    "source_task_ids",
    "evidence_refs",
    "confidence",
    "created_at",
    "reflector",
    "evidence_origin",
    "retrieval_eligible",
    "tags",
}


def _base_lesson(**overrides: Any) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "title": "Pin fixture versions",
        "lesson_body": "Pin the pytest fixture plugin version before rerunning flaky tests.",
        "source_episode_ids": ["lep:aaa"],
        "outcome_polarity": "success",
        "applies_when": ["flaky pytest fixture"],
        "evidence_origin": EVIDENCE_ORIGIN_PHYSICAL,
        "created_at": "2026-01-01T00:00:00+00:00",
    }
    kwargs.update(overrides)
    return build_lesson(**kwargs)


def _episode(
    task_id: str,
    outcome: str = "SUCCEEDED",
    *,
    qualified: bool = True,
    evidence: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if evidence is None:
        evidence = {"receipt": f"rcpt-{task_id}", "verifier": "pytest"}
    return build_nexus_learning_episode(
        task_id=task_id,
        attempt_id=f"att-{task_id}",
        source="unit",
        terminal_outcome=outcome,
        terminal_evidence=evidence,
        qualification={"status": "QUALIFIED"} if qualified else None,
    )


def test_build_and_validate_happy_path() -> None:
    lesson = _base_lesson()
    assert set(lesson) == LESSON_KEYS
    assert lesson["schema"] == LEARNING_LESSON_SCHEMA
    assert lesson["lesson_id"].startswith("lsn:") and len(lesson["lesson_id"]) == 28
    assert lesson["retrieval_eligible"] is True
    assert lesson["reflector"] == {"kind": "deterministic", "model": "", "prompt_hash": ""}
    validate_lesson(lesson)


def test_build_normalizes_inputs() -> None:
    lesson = build_lesson(
        title="  " + "t" * 150 + "  ",
        lesson_body="  body  ",
        source_episode_ids=["lep:b", "lep:a", "lep:a", " "],
        outcome_polarity="failure",
        applies_when=["x", "", "x", "y"],
        avoid_when=["z", "z"],
        confidence=7.0,
        tags=["b", "a", "a"],
    )
    assert len(lesson["title"]) == 120
    assert lesson["lesson_body"] == "body"
    assert lesson["source_episode_ids"] == ["lep:a", "lep:b"]
    assert lesson["applies_when"] == ["x", "y"]
    assert lesson["avoid_when"] == ["z"]
    assert lesson["confidence"] == 1.0
    assert lesson["tags"] == ["a", "b"]
    assert lesson["created_at"]
    assert lesson["retrieval_eligible"] is False


def test_lesson_id_stable_across_created_at_and_confidence_and_reflector() -> None:
    first = _base_lesson(created_at="2026-01-01T00:00:00+00:00", confidence=0.2)
    second = _base_lesson(
        created_at="2026-06-01T00:00:00+00:00",
        confidence=0.9,
        reflector={"kind": "judge", "model": "m", "prompt_hash": "abc"},
    )
    assert first["lesson_id"] == second["lesson_id"]
    different = _base_lesson(lesson_body="A different insight.")
    assert different["lesson_id"] != first["lesson_id"]


@pytest.mark.parametrize(
    ("mutation", "reason"),
    [
        ({"schema": "other.v1"}, "LESSON_SCHEMA_MISMATCH"),
        ({"title": "   "}, "LESSON_CONTENT_REQUIRED"),
        ({"lesson_body": ""}, "LESSON_CONTENT_REQUIRED"),
        ({"source_episode_ids": []}, "LESSON_PROVENANCE_REQUIRED"),
        ({"outcome_polarity": "maybe"}, "LESSON_POLARITY_INVALID"),
        ({"evidence_origin": "synthetic"}, "LESSON_EVIDENCE_ORIGIN_INVALID"),
        (
            {"evidence_origin": EVIDENCE_ORIGIN_SIMULATED, "retrieval_eligible": True},
            "LESSON_SIMULATED_NOT_RETRIEVABLE",
        ),
        ({"confidence": 1.5}, "LESSON_CONFIDENCE_RANGE"),
        ({"confidence": -0.1}, "LESSON_CONFIDENCE_RANGE"),
        ({"lesson_body": "tampered body"}, "LESSON_ID_MISMATCH"),
    ],
)
def test_validate_reason_codes(mutation: dict[str, Any], reason: str) -> None:
    lesson = copy.deepcopy(_base_lesson())
    lesson.update(mutation)
    with pytest.raises(ValueError, match=reason):
        validate_lesson(lesson)


def test_build_rejects_invalid_polarity() -> None:
    with pytest.raises(ValueError, match="LESSON_POLARITY_INVALID"):
        build_lesson(
            title="t",
            lesson_body="b",
            source_episode_ids=["lep:a"],
            outcome_polarity="unknown",
        )


def test_simulated_lessons_are_not_retrieval_eligible() -> None:
    lesson = _base_lesson(evidence_origin=EVIDENCE_ORIGIN_SIMULATED)
    assert lesson["retrieval_eligible"] is False
    forged = copy.deepcopy(lesson)
    forged["retrieval_eligible"] = True
    with pytest.raises(ValueError, match="LESSON_SIMULATED_NOT_RETRIEVABLE"):
        validate_lesson(forged)


def test_store_append_is_idempotent(tmp_path: Path) -> None:
    store = LessonStore(LearningStateRoot(tmp_path))
    lesson = _base_lesson()
    assert store.append(lesson) is True
    assert store.append(lesson) is True
    lines = store.path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    assert store.load() == [lesson]


def test_store_path_under_state_root(tmp_path: Path) -> None:
    state_root = LearningStateRoot(tmp_path)
    expected = tmp_path.resolve() / ".nexus" / "memory" / "learning_lessons.jsonl"
    assert state_root.lessons_path == expected
    assert LessonStore(state_root).path == expected


def test_store_append_validates_before_writing(tmp_path: Path) -> None:
    store = LessonStore(LearningStateRoot(tmp_path))
    bad = _base_lesson()
    bad["confidence"] = 2.0
    with pytest.raises(ValueError, match="LESSON_CONFIDENCE_RANGE"):
        store.append(bad)
    assert not store.path.exists()


def test_load_skips_malformed_and_invalid_rows(tmp_path: Path) -> None:
    path = tmp_path / "learning_lessons.jsonl"
    good = _base_lesson()
    tampered = dict(good, lesson_body="changed")
    path.write_text(
        json.dumps(good) + "\n{not json\n" + json.dumps(tampered) + "\n[1, 2]\n",
        encoding="utf-8",
    )
    assert load_lessons(path) == [good]
    assert load_lessons(tmp_path / "missing.jsonl") == []


@pytest.mark.parametrize(
    ("tail", "expected"),
    [
        (b"", True),  # valid unterminated tail is framed with a separator
        (b"{garbage", False),  # invalid tail refuses the append
    ],
)
def test_unterminated_tail_policy_matches_closure_effectiveness(
    tmp_path: Path, tail: bytes, expected: bool
) -> None:
    episode = _episode("closure-new")
    lesson = _base_lesson()
    closure_path = tmp_path / "closure" / "learning_episodes.jsonl"
    closure_path.parent.mkdir(parents=True)
    store = LessonStore(LearningStateRoot(tmp_path / "lessons"))
    store.path.parent.mkdir(parents=True)

    if tail == b"":
        previous = _episode("closure-old")
        closure_path.write_bytes(json.dumps(previous).encode("utf-8"))
        store.path.write_bytes(json.dumps(_base_lesson(lesson_body="old body")).encode("utf-8"))
    else:
        closure_path.write_bytes(tail)
        store.path.write_bytes(tail)

    closure_result = append_learning_episode(closure_path, episode)
    lesson_result = store.append(lesson)
    assert closure_result is expected
    assert lesson_result is closure_result
    if expected:
        assert len(load_lessons(store.path)) == 2
    else:
        assert store.path.read_bytes() == tail


def _lesson_for(body: str, *, physical: bool = True, title: str = "Lesson") -> dict[str, Any]:
    return _base_lesson(
        title=title,
        lesson_body=body,
        evidence_origin=EVIDENCE_ORIGIN_PHYSICAL if physical else EVIDENCE_ORIGIN_SIMULATED,
        source_episode_ids=[f"lep:{body[:6]}"],
        applies_when=[],
    )


def test_retrieve_ranks_by_overlap_and_excludes_simulated() -> None:
    strong = _lesson_for("pin pytest fixture plugin version flaky", title="fixture pin")
    weak = _lesson_for("unrelated deployment checklist", title="deploy")
    simulated = _lesson_for("pytest fixture flaky simulated only", physical=False, title="sim")
    lessons = [weak, strong, simulated]

    rows = retrieve_lessons(lessons, query_text="flaky pytest fixture_plugin", limit=3)
    assert [row["title"] for row in rows] == ["fixture pin"]
    row = rows[0]
    assert row["lesson_id"] == strong["lesson_id"]
    assert row["summary"] == strong["lesson_body"]
    assert row["classification"] == row["pattern_type"] == "success"
    assert row["source"] == "nexus_learning.lessons"
    assert row["provenance"] == "canonical_lesson"
    assert row["relevance_score"] == pytest.approx(4 / 4)

    unrestricted = retrieve_lessons(
        lessons, query_text="flaky pytest fixture", limit=3, require_physical=False
    )
    assert [r["title"] for r in unrestricted] == ["sim", "fixture pin"]


def test_retrieve_empty_query_and_limit() -> None:
    lessons = [_lesson_for("alpha beta gamma", title="a"), _lesson_for("alpha delta", title="b")]
    assert retrieve_lessons(lessons, query_text="zz") == []
    # Equal overlap, confidence and created_at: lesson_id ascending breaks the tie.
    tie_winner = min(lessons, key=lambda lesson: lesson["lesson_id"])
    top = retrieve_lessons(lessons, query_text="alpha", limit=1)
    assert [row["lesson_id"] for row in top] == [tie_winner["lesson_id"]]
    assert retrieve_lessons(lessons, query_text="alpha", limit=0) == []


class _FakeJudge:
    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.prompts: list[str] = []

    def __call__(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.reply


def test_reflect_with_fenced_judge_output() -> None:
    reply = (
        "```json\n"
        '{"title": "Pin fixture", "lesson": "Pin the fixture version.", '
        '"applies_when": ["flaky"], "avoid_when": [], "confidence": 0.8}\n'
        "```"
    )
    judge = _FakeJudge(reply)
    episodes = [_episode("task-ok")]
    lessons = reflect_episodes(episodes, judge=judge, model_name="fake-model")
    assert len(lessons) == 1
    lesson = lessons[0]
    prompt = judge.prompts[0]
    assert lesson["title"] == "Pin fixture"
    assert lesson["confidence"] == pytest.approx(0.8)
    assert lesson["reflector"] == {
        "kind": "judge",
        "model": "fake-model",
        "prompt_hash": hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16],
    }
    assert lesson["retrieval_eligible"] is True
    assert lesson["source_episode_ids"] == [episodes[0]["episode_id"]]
    assert lesson["evidence_refs"] == ["rcpt-task-ok"]
    assert "task_id=task-ok" in prompt and "Output JSON only." in prompt


def test_reflect_judge_exception_falls_back_to_deterministic() -> None:
    def broken(prompt: str) -> str:
        raise RuntimeError("provider down")

    lessons = reflect_episodes([_episode("task-x")], judge=broken)
    assert len(lessons) == 1
    assert lessons[0]["reflector"]["kind"] == "deterministic"
    assert lessons[0]["title"].startswith("success: task-x")
    assert lessons[0]["reflector"]["prompt_hash"]


@pytest.mark.parametrize(
    "reply",
    [
        "I cannot produce JSON today.",
        '{"title": "only a title"}',
        '{"title": "t", "lesson": "b", "applies_when": "not-a-list"}',
        '{"title": "t", "lesson": "b", "confidence": "high"}',
        "",
    ],
)
def test_reflect_garbage_judge_output_falls_back(reply: str) -> None:
    lessons = reflect_episodes([_episode("task-g")], judge=_FakeJudge(reply))
    assert len(lessons) == 1
    assert lessons[0]["reflector"]["kind"] == "deterministic"


def test_reflect_failure_deterministic_avoid_clause() -> None:
    lessons = reflect_episodes([_episode("task-f", "FAILED")])
    assert lessons[0]["outcome_polarity"] == "failure"
    assert lessons[0]["avoid_when"] == [
        "repeat the same patch strategy without a new verifier signal"
    ]
    assert lessons[0]["confidence"] == pytest.approx(0.3)


def test_reflect_mixed_outcomes_produces_opposite_polarities() -> None:
    episodes = [_episode("ok-1"), _episode("bad-1", "FAILED")]
    lessons = reflect_episodes(episodes)
    polarities = sorted(lesson["outcome_polarity"] for lesson in lessons)
    assert polarities == ["failure", "success"]
    by_polarity = {lesson["outcome_polarity"]: lesson for lesson in lessons}
    assert by_polarity["success"]["source_task_ids"] == ["ok-1"]
    assert by_polarity["failure"]["source_task_ids"] == ["bad-1"]


def test_reflect_ignores_unverified_episodes() -> None:
    unverified = _episode("pending", "UNVERIFIED")
    assert reflect_episodes([unverified]) == []
    lessons = reflect_episodes([unverified, _episode("done")])
    assert len(lessons) == 1
    assert lessons[0]["source_task_ids"] == ["done"]
    assert lessons[0]["source_episode_ids"] == [_episode("done")["episode_id"]]


def test_reflect_physical_only_when_qualified_with_terminal_evidence() -> None:
    qualified = reflect_episodes([_episode("q1"), _episode("q2")])
    assert qualified[0]["evidence_origin"] == EVIDENCE_ORIGIN_PHYSICAL
    assert qualified[0]["retrieval_eligible"] is True

    unqualified = reflect_episodes([_episode("q3"), _episode("u1", qualified=False)])
    assert unqualified[0]["evidence_origin"] == EVIDENCE_ORIGIN_SIMULATED
    assert unqualified[0]["retrieval_eligible"] is False

    empty_evidence = _episode("e1", evidence={})
    empty_evidence["qualification"] = {"status": "QUALIFIED"}
    assert reflect_episodes([empty_evidence])[0]["evidence_origin"] == EVIDENCE_ORIGIN_SIMULATED

    via_status = _episode("s1", qualified=False)
    via_status["qualification_status"] = "QUALIFIED"
    assert reflect_episodes([via_status])[0]["evidence_origin"] == EVIDENCE_ORIGIN_PHYSICAL


def test_reflect_empty_and_max_lessons() -> None:
    assert reflect_episodes([]) == []
    assert reflect_episodes([_episode("m1")], max_lessons=0) == []


def test_build_reflection_prompt_truncates_evidence() -> None:
    episode = _episode("big", evidence={"receipt": "r", "blob": "x" * 2000})
    prompt = build_reflection_prompt([episode], "success")
    assert "terminal_outcome=SUCCEEDED" in prompt
    assert "x" * 601 not in prompt
    assert prompt.endswith("Output JSON only.")


def test_build_lesson_bounds_text_and_lists() -> None:
    from nexus_learning.lessons import build_lesson, validate_lesson

    lesson = build_lesson(
        title="t" * 500,
        lesson_body="b" * 5000,
        source_episode_ids=["lep:1"],
        outcome_polarity="success",
        applies_when=[f"cond-{i}-" + "x" * 400 for i in range(50)],
        avoid_when=["a" * 400],
    )
    assert len(lesson["title"]) == 120
    assert len(lesson["lesson_body"]) == 2000
    assert len(lesson["applies_when"]) == 20
    assert all(len(item) <= 300 for item in lesson["applies_when"] + lesson["avoid_when"])
    validate_lesson(lesson)

    forged = dict(lesson)
    forged["lesson_body"] = "b" * 2001
    with pytest.raises(ValueError, match="LESSON_CONTENT_TOO_LONG"):
        validate_lesson(forged)


def _parked_episode(
    task_id: str,
    *,
    terminal_evidence: dict[str, Any],
    qualification: dict[str, Any] | None,
) -> dict[str, Any]:
    # The contract validator refuses QUALIFIED on PARKED (PARKED is not a measured
    # outcome), so the qualification block is attached after the envelope is built.
    # _is_qualified reads qualification.status, which is what this exercises.
    episode = build_nexus_learning_episode(
        task_id=task_id,
        attempt_id=f"att-{task_id}",
        source="unit",
        terminal_outcome="PARKED",
        terminal_evidence=terminal_evidence,
    )
    if qualification is not None:
        episode["qualification"] = dict(qualification)
    return episode


def test_reflect_parked_with_qualified_repeatability_fail_is_physical_failure() -> None:
    episode = _parked_episode(
        "park-q",
        terminal_evidence={"receipt": "r1", "verifier": "pytest"},
        qualification={
            "status": "QUALIFIED",
            "repeatability": {"verifier_status": "fail"},
            "prevention_rule": "stop retrying the same patch",
            "authority_qualification": "local-heal",
        },
    )
    lessons = reflect_episodes([episode])
    assert len(lessons) == 1
    lesson = lessons[0]
    assert lesson["outcome_polarity"] == "failure"
    assert lesson["evidence_origin"] == EVIDENCE_ORIGIN_PHYSICAL
    assert lesson["retrieval_eligible"] is True
    assert lesson["title"].startswith("failure (parked):")
    assert lesson["source_episode_ids"] == [episode["episode_id"]]
    assert "failure (parked)" in lesson["lesson_body"]


def test_reflect_parked_with_terminal_verifier_fail_unqualified_is_simulated_failure() -> None:
    episode = _parked_episode(
        "park-s",
        terminal_evidence={"verifier": "fail", "receipt": "r1"},
        qualification=None,
    )
    lessons = reflect_episodes([episode])
    assert len(lessons) == 1
    assert lessons[0]["outcome_polarity"] == "failure"
    assert lessons[0]["evidence_origin"] == EVIDENCE_ORIGIN_SIMULATED
    assert lessons[0]["retrieval_eligible"] is False


def test_reflect_parked_without_verifier_fail_is_ignored() -> None:
    episode = _parked_episode(
        "park-n",
        terminal_evidence={"receipt": "r0", "verifier": "pytest"},
        qualification={"repeatability": {"verifier_status": "pass"}},
    )
    assert reflect_episodes([episode]) == []


def test_reflect_mixed_succeeded_and_parked_fail_yield_opposite_polarities() -> None:
    succeeded = _episode("ok-d")
    parked = _parked_episode(
        "park-d",
        terminal_evidence={"verifier": "fail", "receipt": "r2"},
        qualification={
            "status": "QUALIFIED",
            "repeatability": {"verifier_status": "fail"},
            "prevention_rule": "rule",
            "authority_qualification": "auth",
        },
    )
    lessons = reflect_episodes([succeeded, parked])
    assert len(lessons) == 2
    by_polarity = {lesson["outcome_polarity"]: lesson for lesson in lessons}
    assert set(by_polarity) == {"success", "failure"}
    assert by_polarity["success"]["source_task_ids"] == ["ok-d"]
    assert by_polarity["failure"]["source_task_ids"] == ["park-d"]
    assert by_polarity["failure"]["evidence_origin"] == EVIDENCE_ORIGIN_PHYSICAL
