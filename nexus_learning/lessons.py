"""Reflection stage of the learning loop: canonical lesson schema, lesson store, deterministic retrieval helpers and an injectable reflector. Lessons are advisory evidence; they never select routes, models or workers."""

from __future__ import annotations

import hashlib
import json
import math
import re
import threading
from collections.abc import Callable, Iterable, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:  # POSIX advisory lock; the thread lock remains the fallback.
    import fcntl
except ImportError:  # pragma: no cover
    fcntl = None  # type: ignore[assignment]

from nexus_learning.closure_effectiveness import _validate_unterminated_tail
from nexus_learning.episode_projection import project_learning_entries
from nexus_learning.state_root import LearningStateRoot

LEARNING_LESSON_SCHEMA = "nexus.learning_lesson.v1"
EVIDENCE_ORIGIN_PHYSICAL = "physical"
EVIDENCE_ORIGIN_SIMULATED = "simulated"
OUTCOME_POLARITY_SUCCESS = "success"
OUTCOME_POLARITY_FAILURE = "failure"
REFLECTOR_KIND_JUDGE = "judge"
REFLECTOR_KIND_DETERMINISTIC = "deterministic"

_EPISODE_SCHEMA = "nexus.learning_episode.v1"
_EVIDENCE_ORIGINS = frozenset({EVIDENCE_ORIGIN_PHYSICAL, EVIDENCE_ORIGIN_SIMULATED})
_POLARITIES = frozenset({OUTCOME_POLARITY_SUCCESS, OUTCOME_POLARITY_FAILURE})
_PARKED_OUTCOME = "PARKED"
_VERIFIER_FAIL_VALUES = frozenset({"fail", "failed"})
_EVIDENCE_REF_KEYS = ("receipt", "evidence_ref", "receipt_id")
_TITLE_MAX = 120
_BODY_MAX = 2000
_LIST_MAX = 20
_ITEM_MAX = 300
_EVIDENCE_PROMPT_MAX = 600
_DEFAULT_CONFIDENCE = 0.5
_DETERMINISTIC_CONFIDENCE = 0.3
_FAILURE_AVOID = "repeat the same patch strategy without a new verifier signal"
_APPEND_LOCK = threading.Lock()
_TOKEN_RE = re.compile(r"\w+")
_FENCE_RE = re.compile(r"^```[A-Za-z0-9_-]*\s*|\s*```$")


def _text(value: Any) -> str:
    return str(value).strip() if value is not None else ""


def _unique_sorted(values: Iterable[Any]) -> list[str]:
    return sorted({_text(v) for v in values if _text(v)})


def _unique_ordered(values: Iterable[Any]) -> list[str]:
    seen: dict[str, None] = {}
    for value in values:
        item = _text(value)
        if item:
            seen.setdefault(item, None)
    return list(seen)


def _tokens(text: str) -> set[str]:
    normalized = text.lower().replace("_", " ")
    return {token for token in _TOKEN_RE.findall(normalized) if len(token) >= 3}


def _clamp_confidence(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    if math.isnan(number):
        return 0.0
    return min(1.0, max(0.0, number))


def _in(value: Any, allowed: frozenset[str]) -> bool:
    return isinstance(value, str) and value in allowed


def _compute_lesson_id(
    *,
    title: Any,
    lesson_body: Any,
    applies_when: Any,
    avoid_when: Any,
    outcome_polarity: Any,
    source_episode_ids: Any,
) -> str:
    payload = {
        "title": title,
        "lesson_body": lesson_body,
        "applies_when": list(applies_when),
        "avoid_when": list(avoid_when),
        "outcome_polarity": outcome_polarity,
        "source_episode_ids": list(source_episode_ids),
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return "lsn:" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:24]


def build_lesson(
    *,
    title: str,
    lesson_body: str,
    source_episode_ids: Iterable[str],
    outcome_polarity: str,
    applies_when: Iterable[str] = (),
    avoid_when: Iterable[str] = (),
    source_task_ids: Iterable[str] = (),
    evidence_refs: Iterable[str] = (),
    confidence: float = _DEFAULT_CONFIDENCE,
    created_at: str = "",
    reflector: Mapping[str, Any] | None = None,
    evidence_origin: str = EVIDENCE_ORIGIN_SIMULATED,
    tags: Iterable[str] = (),
) -> dict[str, Any]:
    """Normalize lesson content into the canonical v1 shape and validate it."""
    clean_title = _text(title)[:_TITLE_MAX].strip()
    clean_body = _text(lesson_body)[:_BODY_MAX].strip()
    sources = _unique_sorted(source_episode_ids)
    applies = [item[:_ITEM_MAX] for item in _unique_ordered(applies_when)][:_LIST_MAX]
    avoid = [item[:_ITEM_MAX] for item in _unique_ordered(avoid_when)][:_LIST_MAX]
    polarity = _text(outcome_polarity)
    origin = _text(evidence_origin)
    lesson_id = _compute_lesson_id(
        title=clean_title,
        lesson_body=clean_body,
        applies_when=applies,
        avoid_when=avoid,
        outcome_polarity=polarity,
        source_episode_ids=sources,
    )
    reflector_block: dict[str, Any] = {
        "kind": REFLECTOR_KIND_DETERMINISTIC,
        "model": "",
        "prompt_hash": "",
    }
    if reflector:
        reflector_block.update({str(key): value for key, value in reflector.items()})
    lesson: dict[str, Any] = {
        "schema": LEARNING_LESSON_SCHEMA,
        "lesson_id": lesson_id,
        "title": clean_title,
        "lesson_body": clean_body,
        "applies_when": applies,
        "avoid_when": avoid,
        "outcome_polarity": polarity,
        "source_episode_ids": sources,
        "source_task_ids": _unique_sorted(source_task_ids),
        "evidence_refs": _unique_sorted(evidence_refs),
        "confidence": _clamp_confidence(confidence),
        "created_at": _text(created_at) or datetime.now(timezone.utc).isoformat(),
        "reflector": reflector_block,
        "evidence_origin": origin,
        "retrieval_eligible": origin == EVIDENCE_ORIGIN_PHYSICAL,
        "tags": _unique_sorted(tags),
    }
    validate_lesson(lesson)
    return lesson


def validate_lesson(lesson: Mapping[str, Any]) -> None:
    """Raise ValueError carrying a reason code when a lesson violates the contract."""
    if not isinstance(lesson, Mapping) or lesson.get("schema") != LEARNING_LESSON_SCHEMA:
        raise ValueError("LESSON_SCHEMA_MISMATCH")
    if not _text(lesson.get("title")) or not _text(lesson.get("lesson_body")):
        raise ValueError("LESSON_CONTENT_REQUIRED")
    if len(_text(lesson.get("lesson_body"))) > _BODY_MAX or len(_text(lesson.get("title"))) > _TITLE_MAX:
        raise ValueError("LESSON_CONTENT_TOO_LONG")
    for key in ("applies_when", "avoid_when"):
        items = lesson.get(key) or []
        if len(items) > _LIST_MAX or any(len(_text(item)) > _ITEM_MAX for item in items):
            raise ValueError("LESSON_CONTENT_TOO_LONG")
    sources = lesson.get("source_episode_ids")
    if not isinstance(sources, (list, tuple)) or not _unique_sorted(sources):
        raise ValueError("LESSON_PROVENANCE_REQUIRED")
    if not _in(lesson.get("outcome_polarity"), _POLARITIES):
        raise ValueError("LESSON_POLARITY_INVALID")
    origin = lesson.get("evidence_origin")
    if not _in(origin, _EVIDENCE_ORIGINS):
        raise ValueError("LESSON_EVIDENCE_ORIGIN_INVALID")
    if bool(lesson.get("retrieval_eligible")) and origin != EVIDENCE_ORIGIN_PHYSICAL:
        raise ValueError("LESSON_SIMULATED_NOT_RETRIEVABLE")
    try:
        confidence = float(lesson.get("confidence"))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        raise ValueError("LESSON_CONFIDENCE_RANGE") from None
    if not 0.0 <= confidence <= 1.0:
        raise ValueError("LESSON_CONFIDENCE_RANGE")
    try:
        expected = _compute_lesson_id(
            title=lesson.get("title"),
            lesson_body=lesson.get("lesson_body"),
            applies_when=list(lesson.get("applies_when") or []),
            avoid_when=list(lesson.get("avoid_when") or []),
            outcome_polarity=lesson.get("outcome_polarity"),
            source_episode_ids=list(sources),
        )
    except TypeError:
        raise ValueError("LESSON_ID_MISMATCH") from None
    if lesson.get("lesson_id") != expected:
        raise ValueError("LESSON_ID_MISMATCH")


def load_lessons(path: Path) -> list[dict[str, Any]]:
    """Load valid lessons; malformed lines and contract-violating rows are skipped."""
    if not path.exists():
        return []
    lessons: list[dict[str, Any]] = []
    for raw in path.read_bytes().splitlines():
        if not raw.strip():
            continue
        try:
            row = json.loads(raw)
        except ValueError:
            continue
        if not isinstance(row, dict):
            continue
        try:
            validate_lesson(row)
        except ValueError:
            continue
        lessons.append(row)
    return lessons


class LessonStore:
    """Append-only JSONL lesson store under the explicit learning state root."""

    def __init__(self, state_root: LearningStateRoot) -> None:
        self._state_root = state_root

    @property
    def path(self) -> Path:
        return self._state_root.lessons_path

    def append(self, lesson: dict[str, Any]) -> bool:
        """Append once keyed by lesson_id; returns False only on a refused tail boundary."""
        validate_lesson(lesson)
        lesson_id = str(lesson["lesson_id"])
        path = self.path
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with _APPEND_LOCK:
                with path.open("a+", encoding="utf-8") as handle:
                    if fcntl is not None:
                        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
                    try:
                        tail_state = _validate_unterminated_tail(path)
                        if tail_state is False:
                            return False
                        handle.seek(0)
                        for line in handle:
                            try:
                                row = json.loads(line)
                            except json.JSONDecodeError:
                                continue
                            if isinstance(row, dict) and row.get("lesson_id") == lesson_id:
                                return True
                        handle.seek(0, 2)
                        if tail_state is True:
                            handle.write("\n")
                        handle.write(json.dumps(lesson, ensure_ascii=False) + "\n")
                        handle.flush()
                        return True
                    finally:
                        if fcntl is not None:
                            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        except (OSError, TypeError, ValueError):
            return False

    def load(self) -> list[dict[str, Any]]:
        return load_lessons(self.path)


def _as_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def retrieve_lessons(
    lessons: Iterable[Mapping[str, Any]],
    *,
    query_text: str,
    limit: int = 3,
    require_physical: bool = True,
) -> list[dict[str, Any]]:
    """Deterministic keyword-overlap retrieval over canonical lessons."""
    query = _tokens(_text(query_text))
    if limit <= 0 or not query:
        return []
    matches: list[tuple[int, Mapping[str, Any]]] = []
    for lesson in lessons:
        if not isinstance(lesson, Mapping):
            continue
        if require_physical and not lesson.get("retrieval_eligible"):
            continue
        searchable = " ".join(
            [
                _text(lesson.get("title")),
                _text(lesson.get("lesson_body")),
                " ".join(_text(item) for item in lesson.get("applies_when") or []),
                " ".join(_text(item) for item in lesson.get("tags") or []),
            ]
        )
        overlap = len(query & _tokens(searchable))
        if overlap:
            matches.append((overlap, lesson))
    # Stable sorts applied from least to most significant key.
    matches.sort(key=lambda m: _text(m[1].get("lesson_id")))
    matches.sort(key=lambda m: _text(m[1].get("created_at")), reverse=True)
    matches.sort(key=lambda m: _as_float(m[1].get("confidence")), reverse=True)
    matches.sort(key=lambda m: m[0], reverse=True)
    rows: list[dict[str, Any]] = []
    for overlap, lesson in matches[:limit]:
        polarity = _text(lesson.get("outcome_polarity"))
        rows.append(
            {
                "lesson_id": _text(lesson.get("lesson_id")),
                "summary": _text(lesson.get("lesson_body")),
                "title": _text(lesson.get("title")),
                "classification": polarity,
                "pattern_type": polarity,
                "source": "nexus_learning.lessons",
                "relevance_score": overlap / max(1, len(query)),
                "applies_when": list(lesson.get("applies_when") or []),
                "avoid_when": list(lesson.get("avoid_when") or []),
                "evidence_refs": list(lesson.get("evidence_refs") or []),
                "source_episode_ids": list(lesson.get("source_episode_ids") or []),
                "confidence": _as_float(lesson.get("confidence")),
                "provenance": "canonical_lesson",
            }
        )
    return rows


def _episode_source(episode: Mapping[str, Any]) -> str:
    return _text(episode.get("source") or episode.get("producer"))


def build_reflection_prompt(episodes: Sequence[Mapping[str, Any]], polarity: str) -> str:
    lines = [f"Outcome polarity: {polarity}.", "Episodes:"]
    for episode in episodes:
        evidence = json.dumps(
            episode.get("terminal_evidence") or {},
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            default=str,
        )[:_EVIDENCE_PROMPT_MAX]
        retrieved = ",".join(_unique_sorted(episode.get("retrieved_lesson_ids") or []))
        task_id = _text(episode.get("task_id"))
        outcome = _text(episode.get("terminal_outcome")).upper()
        source = _episode_source(episode)
        lines.append(
            f"- task_id={task_id} terminal_outcome={outcome} source={source} "
            f"terminal_evidence={evidence} retrieved_lesson_ids=[{retrieved}]"
        )
    lines.append(
        "Write ONE reusable lesson as JSON with keys title, lesson, applies_when (list), "
        "avoid_when (list), confidence (0-1). For failures, state what to avoid and why. "
        "Output JSON only."
    )
    return "\n".join(lines)


def _is_qualified(episode: Mapping[str, Any]) -> bool:
    qualification = episode.get("qualification")
    status = qualification.get("status") if isinstance(qualification, Mapping) else None
    qualified = status == "QUALIFIED" or episode.get("qualification_status") == "QUALIFIED"
    evidence = episode.get("terminal_evidence")
    return bool(qualified) and isinstance(evidence, Mapping) and bool(evidence)


def _verifier_failed(episode: Mapping[str, Any]) -> bool:
    """True when the episode carries explicit verifier-fail evidence.

    Used only to decide whether a PARKED episode is reflected as a failure
    lesson. It does not change the episode's lifecycle state.
    """
    qualification = episode.get("qualification")
    repeatability = qualification.get("repeatability") if isinstance(qualification, Mapping) else None
    if isinstance(repeatability, Mapping) and _text(repeatability.get("verifier_status")).lower() == "fail":
        return True
    evidence = episode.get("terminal_evidence")
    if not isinstance(evidence, Mapping):
        return False
    return any(
        _text(evidence.get(key)).lower() in _VERIFIER_FAIL_VALUES
        for key in ("verifier_status", "verifier")
    )


def _reflection_polarity(episode: Mapping[str, Any]) -> str | None:
    """Map an episode to a lesson polarity, or None when it is not reflectable.

    PARKED is a lifecycle state (failure without an explicit terminal decision);
    it yields a failure lesson only when verifier-fail evidence is present.
    """
    outcome = _text(episode.get("terminal_outcome")).upper()
    if outcome == "SUCCEEDED":
        return OUTCOME_POLARITY_SUCCESS
    if outcome == "FAILED" or (outcome == _PARKED_OUTCOME and _verifier_failed(episode)):
        return OUTCOME_POLARITY_FAILURE
    return None


def _evidence_refs(episodes: Sequence[Mapping[str, Any]]) -> list[str]:
    refs: list[str] = []
    for episode in episodes:
        evidence = episode.get("terminal_evidence")
        if not isinstance(evidence, Mapping):
            continue
        for key in _EVIDENCE_REF_KEYS:
            value = evidence.get(key)
            if isinstance(value, str) and value.strip():
                refs.append(value)
    return refs


def _parse_judge_output(raw: Any) -> dict[str, Any] | None:
    if not isinstance(raw, str):
        return None
    text = _FENCE_RE.sub("", raw.strip()).strip()
    candidates = [text]
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        candidates.append(text[start : end + 1])
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except ValueError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def _optional_str_list(value: Any) -> list[str] | None:
    if value is None:
        return []
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        return None
    return value


def _judge_lesson(
    judge: Callable[[str], str],
    prompt: str,
    *,
    prompt_hash: str,
    model_name: str,
    shared: Mapping[str, Any],
) -> dict[str, Any] | None:
    try:
        raw = judge(prompt)
    except Exception:  # a failing judge must never break reflection
        return None
    payload = _parse_judge_output(raw)
    if payload is None:
        return None
    title = payload.get("title")
    body = payload.get("lesson")
    if not isinstance(title, str) or not isinstance(body, str):
        return None
    if not title.strip() or not body.strip():
        return None
    applies = _optional_str_list(payload.get("applies_when"))
    avoid = _optional_str_list(payload.get("avoid_when"))
    if applies is None or avoid is None:
        return None
    confidence = payload.get("confidence")
    if confidence is None:
        confidence = _DEFAULT_CONFIDENCE
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
        return None
    try:
        return build_lesson(
            title=title,
            lesson_body=body,
            applies_when=applies,
            avoid_when=avoid,
            confidence=float(confidence),
            reflector={
                "kind": REFLECTOR_KIND_JUDGE,
                "model": model_name,
                "prompt_hash": prompt_hash,
            },
            **shared,
        )
    except ValueError:
        return None


def _deterministic_lesson(
    group: Sequence[Mapping[str, Any]],
    *,
    polarity: str,
    prompt_hash: str,
    shared: Mapping[str, Any],
) -> dict[str, Any]:
    task_ids = _unique_ordered(episode.get("task_id") for episode in group)
    parked_only = polarity == OUTCOME_POLARITY_FAILURE and all(
        _text(ep.get("terminal_outcome")).upper() == _PARKED_OUTCOME for ep in group
    )
    label = f"{polarity} ({_PARKED_OUTCOME.lower()})" if parked_only else polarity
    title = f"{label}: {','.join(task_ids)}"[:_TITLE_MAX]
    summaries = [_text(entry.get("summary")) for entry in project_learning_entries(group)]
    body = "; ".join(summary for summary in summaries if summary)
    if not body:
        body = "; ".join(
            f"task {_text(ep.get('task_id'))} ended "
            f"{_text(ep.get('terminal_outcome')).upper()} via {_episode_source(ep)}"
            for ep in group
        )
    if parked_only:
        body = f"{label} with verifier-fail evidence: {body}"
    return build_lesson(
        title=title,
        lesson_body=body,
        applies_when=[f"task:{task_id}" for task_id in task_ids],
        avoid_when=[_FAILURE_AVOID] if polarity == OUTCOME_POLARITY_FAILURE else [],
        confidence=_DETERMINISTIC_CONFIDENCE,
        reflector={
            "kind": REFLECTOR_KIND_DETERMINISTIC,
            "model": "",
            "prompt_hash": prompt_hash,
        },
        **shared,
    )


def _reflect_group(
    group: Sequence[Mapping[str, Any]],
    *,
    polarity: str,
    origin: str,
    judge: Callable[[str], str] | None,
    model_name: str,
) -> dict[str, Any]:
    prompt = build_reflection_prompt(group, polarity)
    prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16]
    shared: dict[str, Any] = {
        "source_episode_ids": [_text(ep.get("episode_id")) for ep in group],
        "source_task_ids": _unique_ordered(ep.get("task_id") for ep in group),
        "evidence_refs": _evidence_refs(group),
        "outcome_polarity": polarity,
        "evidence_origin": origin,
    }
    if judge is not None:
        lesson = _judge_lesson(
            judge,
            prompt,
            prompt_hash=prompt_hash,
            model_name=model_name,
            shared=shared,
        )
        if lesson is not None:
            return lesson
    return _deterministic_lesson(group, polarity=polarity, prompt_hash=prompt_hash, shared=shared)


def reflect_episodes(
    episodes: Iterable[Mapping[str, Any]],
    *,
    judge: Callable[[str], str] | None = None,
    model_name: str = "",
    max_lessons: int = 1,
) -> list[dict[str, Any]]:
    """Reflect canonical episodes into validated lessons.

    The judge callable is the only injection point; failures fall back to a
    deterministic lesson. SUCCEEDED and FAILED episodes are reflected as before.
    PARKED episodes are reflected as failure lessons only when they carry
    verifier-fail evidence; the episode's lifecycle state is untouched and only
    the advisory lesson is produced. Other episodes, and episodes without an
    episode_id (provenance is required), are ignored.
    """
    if max_lessons < 1:
        return []
    tagged: list[tuple[str, dict[str, Any]]] = []
    for episode in episodes:
        if not isinstance(episode, Mapping) or not _text(episode.get("episode_id")):
            continue
        polarity = _reflection_polarity(episode)
        if polarity is not None:
            tagged.append((polarity, dict(episode)))
    if not tagged:
        return []
    kept = [episode for _, episode in tagged]
    groups: list[tuple[str, list[dict[str, Any]]]] = [
        (polarity, [episode for tag, episode in tagged if tag == polarity])
        for polarity in (OUTCOME_POLARITY_SUCCESS, OUTCOME_POLARITY_FAILURE)
    ]
    groups = [(polarity, group) for polarity, group in groups if group]
    origin = (
        EVIDENCE_ORIGIN_PHYSICAL
        if all(_is_qualified(episode) for episode in kept)
        else EVIDENCE_ORIGIN_SIMULATED
    )
    return [
        _reflect_group(
            group,
            polarity=polarity,
            origin=origin,
            judge=judge,
            model_name=model_name,
        )
        for polarity, group in groups
    ]
