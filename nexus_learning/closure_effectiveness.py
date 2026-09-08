from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:  # POSIX advisory lock; fallback remains safe for single-process callers.
    import fcntl
except ImportError:  # pragma: no cover
    fcntl = None

from nexus_learning.contracts import (
    build_nexus_learning_episode,
    paired_memory_uplift_observed,
)
from nexus_learning.episode_projection import project_learning_entries
from nexus_learning.state_root import LearningStateRoot, resolve_learning_state_root

NEXUS_LEARNING_EPISODES_RELATIVE = Path(".nexus/memory/learning_episodes.jsonl")
_APPEND_FALLBACK_LOCK = threading.Lock()


def canonical_learning_episode_path(project_root: Path | LearningStateRoot) -> Path:
    if isinstance(project_root, LearningStateRoot):
        return project_root.learning_episodes_path
    return resolve_learning_state_root(project_root).learning_episodes_path



@dataclass
class EffectivenessReport:
    total_entries: int
    improved_count: int
    no_change_count: int
    degraded_count: int
    improvement_rate: float
    details: list[dict[str, Any]] = field(default_factory=list)
    data_exists_count: int = 0
    retrieved_count: int = 0
    applied_count: int = 0
    outcome_measured_count: int = 0
    paired_uplift_count: int = 0
    raw_entries: int = 0
    unique_semantic_entries: int = 0
    semantic_duplicate_entries: int = 0


def normalize_learning_episode(
    *,
    task_id: str,
    attempt_id: str = "",
    action_id: str = "",
    source: str = "effectiveness",
    terminal_outcome: str = "UNVERIFIED",
    terminal_evidence: dict[str, Any] | None = None,
    phase_receipts: list[dict[str, Any]] | tuple[dict[str, Any], ...] = (),
    receipts: list[dict[str, Any]] | tuple[dict[str, Any], ...] = (),
    retrieved_lesson_ids: list[str] | tuple[str, ...] = (),
    applied_lesson_ids: list[str] | tuple[str, ...] = (),
    qualification: dict[str, Any] | None = None,
    lesson_disposition: str = "shadow",
    learning_write_succeeded: bool = True,
    idempotency_key: str = "",
) -> dict[str, Any]:
    """Normalize a producer event into the one Nexus learning authority."""
    return build_nexus_learning_episode(
        task_id=task_id,
        attempt_id=attempt_id,
        action_id=action_id,
        source=source,
        terminal_outcome=terminal_outcome,
        terminal_evidence=terminal_evidence,
        phase_receipts=phase_receipts,
        receipts=receipts,
        retrieved_lesson_ids=retrieved_lesson_ids,
        applied_lesson_ids=applied_lesson_ids,
        qualification=qualification,
        lesson_disposition=lesson_disposition,
        learning_write_succeeded=learning_write_succeeded,
        idempotency_key=idempotency_key,
    )


def append_learning_episode(path: Path, episode: dict[str, Any]) -> bool:
    """Atomically append once by episode id; duplicate is idempotent success."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        episode_id = str(episode.get("episode_id", ""))
        if not episode_id:
            return False
        with _APPEND_FALLBACK_LOCK:
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
                            if isinstance(row, dict) and str(row.get("episode_id", "")) == episode_id:
                                return True
                        except json.JSONDecodeError:
                            continue
                    handle.seek(0, 2)
                    if tail_state is True:
                        handle.write("\n")
                    handle.write(json.dumps(episode, ensure_ascii=False) + "\n")
                    handle.flush()
                    return True
                finally:
                    if fcntl is not None:
                        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    except (OSError, TypeError, ValueError):
        return False


def _validate_unterminated_tail(path: Path) -> bool | None:
    """Return valid-tail separator state, or invalid-tail/complete-boundary state."""
    raw = path.read_bytes()
    if not raw or not raw.strip() or raw.endswith(b"\n"):
        return None
    lines = [line for line in raw.splitlines() if line.strip()]
    if not lines:
        return None
    try:
        tail = json.loads(lines[-1])
    except json.JSONDecodeError:
        return False
    return isinstance(tail, dict)


def learning_episode_exists(path: Path, episode_id: str) -> bool:
    if not path.exists() or not episode_id:
        return False
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
            if isinstance(row, dict) and str(row.get("episode_id", "")) == episode_id:
                return True
        except json.JSONDecodeError:
            continue
    return False


def load_learning_closures(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    entries: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return entries


def load_canonical_learning_episodes(project_root: Path | LearningStateRoot) -> list[dict[str, Any]]:
    """Load only canonical episodes; legacy projections remain separate."""
    return [
        entry
        for entry in load_learning_closures(canonical_learning_episode_path(project_root))
        if isinstance(entry, dict) and entry.get("schema") == "nexus.learning_episode.v1"
    ]



def classify_closure_effectiveness(entry: dict[str, Any]) -> str:
    raw_stages = entry.get("stages")
    stages: dict[str, Any] = raw_stages if isinstance(raw_stages, dict) else {}
    raw_evidence = entry.get("terminal_evidence")
    evidence: dict[str, Any] = raw_evidence if isinstance(raw_evidence, dict) else {}
    raw_qual = entry.get("qualification")
    qualification: dict[str, Any] = raw_qual if isinstance(raw_qual, dict) else {}
    qualification_complete = bool(
        qualification.get("repeatability")
        and qualification.get("prevention_rule")
        and qualification.get("authority_qualification")
    )
    if (
        stages.get("outcome_uplift_observed") is True
        and entry.get("qualification_status") == "QUALIFIED"
        and qualification_complete
        and paired_memory_uplift_observed(evidence)
    ):
        return "improved"
    outcome = str(entry.get("terminal_outcome", "")).upper()
    verifier_status = str(evidence.get("verifier_status", evidence.get("verifier", ""))).lower()
    if outcome in {"FAILED", "REJECTED"} and verifier_status in {"fail", "failed", "rejected"}:
        return "degraded"
    return "no_change"


def evaluate_effectiveness(entries: list[dict[str, Any]]) -> EffectivenessReport:
    details: list[dict[str, Any]] = []
    improved = 0
    no_change = 0
    degraded = 0
    data_exists = retrieved = applied = measured = uplift = 0
    projected = project_learning_entries(entries)
    for entry in entries:
        raw_stages = entry.get("stages")
        stages: dict[str, Any] = raw_stages if isinstance(raw_stages, dict) else {}
        data_exists += int(bool(stages.get("recorded", entry.get("learning_write_succeeded", False))))
        retrieved += int(bool(stages.get("retrieved", entry.get("retrieved_lesson_ids"))))
        applied += int(bool(stages.get("applied", entry.get("applied_lesson_ids"))))
        measured += int(bool(stages.get("outcome_measured")))
        effect = classify_closure_effectiveness(entry)
        uplift += int(effect == "improved")
        if effect == "improved":
            improved += 1
        elif effect == "degraded":
            degraded += 1
        else:
            no_change += 1
        details.append({
            "effect": effect,
            "classification": entry.get("classification", entry.get("action", "unknown")),
            "task_id": entry.get("task_id", "unknown"),
            "status": entry.get("status", entry.get("writeback_status", entry.get("terminal_outcome", "unknown"))),
        })
    total = len(entries)
    return EffectivenessReport(
        total_entries=total,
        improved_count=improved,
        no_change_count=no_change,
        degraded_count=degraded,
        improvement_rate=round(improved / total, 4) if total > 0 else 0.0,
        details=details[:50],
        data_exists_count=data_exists,
        retrieved_count=retrieved,
        applied_count=applied,
        outcome_measured_count=measured,
        paired_uplift_count=uplift,
        raw_entries=total,
        unique_semantic_entries=len(projected),
        semantic_duplicate_entries=max(0, total - len(projected)),
    )


def generate_effectiveness_report(entries: list[dict[str, Any]], output_path: Path) -> None:
    report = evaluate_effectiveness(entries)
    lines = [
        "# Learning Closure Effectiveness Report",
        "",
        f"**Total entries**: {report.total_entries}",
        f"**Improved**: {report.improved_count}",
        f"**No change**: {report.no_change_count}",
        f"**Degraded**: {report.degraded_count}",
        f"**Improvement rate**: {report.improvement_rate}",
        f"**Data exists / retrieved / applied / measured / paired uplift**: {report.data_exists_count} / {report.retrieved_count} / {report.applied_count} / {report.outcome_measured_count} / {report.paired_uplift_count}",
        f"**Raw / unique semantic / duplicate entries**: {report.raw_entries} / {report.unique_semantic_entries} / {report.semantic_duplicate_entries}",
        "",
        "## Top 50 Details",
        "",
    ]
    for d in report.details:
        lines.append(f"- {d['effect']:>10} | {d['classification']:20} | task={d['task_id']:20} | status={d['status']}")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
