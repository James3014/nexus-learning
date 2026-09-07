from __future__ import annotations

import asyncio
import hashlib
import json
import threading
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping

try:
    import fcntl
except ImportError:  # pragma: no cover - non-POSIX thread fallback
    fcntl = None  # type: ignore[assignment]

from nexus_learning.state_root import LearningStateRoot, resolve_learning_state_root

OUTCOME_MEMORY_SCHEMA = "nexus_outcome_memory_episode.v1"
LEARNING_EPISODE_SCHEMA = "nexus.learning_episode.v1"
DYNAMIC_LEARNING_POLICY_SCHEMA = "nexus_dynamic_learning_policy.v1"
TERMINAL_OUTCOMES = frozenset({"SUCCEEDED", "FAILED", "CANCELLED", "PROCESS_LOST", "PARKED", "RETIRED"})
QUALIFIED_TERMINAL_OUTCOMES = frozenset({"SUCCEEDED", "FAILED", "CANCELLED"})

_OUTCOME_WRITE_LOCKS: dict[Path, threading.Lock] = {}
_OUTCOME_WRITE_LOCKS_GUARD = threading.Lock()


def _thread_lock_for(storage_path: Path) -> threading.Lock:
    key = storage_path.resolve()
    with _OUTCOME_WRITE_LOCKS_GUARD:
        return _OUTCOME_WRITE_LOCKS.setdefault(key, threading.Lock())


@contextmanager
def _locked_outcome_write(storage_path: Path) -> Iterator[None]:
    """Serialize one outcome history read/check/append boundary per state root."""
    with _thread_lock_for(storage_path):
        if fcntl is None:
            yield
            return
        lock_path = storage_path.with_name(storage_path.name + ".lock")
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        with lock_path.open("a+b") as lock_handle:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)


@dataclass(frozen=True)
class EpisodeOutcomeRecord:
    task_id: str
    task_type: str
    task_desc_hash: str
    solved: bool
    wall_duration_sec: float
    total_tokens_used: int
    trust_mismatch: bool
    receipts: list[dict[str, Any]] = field(default_factory=list)
    ab_lift_value: float = 0.0
    created_at: str = ""
    schema_version: str = OUTCOME_MEMORY_SCHEMA
    attempt_id: str = ""
    action_id: str = ""
    idempotency_key: str = ""
    terminal_outcome: str = "PARKED"
    auto_replay_allowed: bool = False
    qualification_status: str = "UNQUALIFIED"
    qualification_evidence_present: bool = False
    retrieved_lesson_ids: list[str] = field(default_factory=list)
    applied_lesson_ids: list[str] = field(default_factory=list)
    lesson_updates: list[dict[str, Any]] = field(default_factory=list)
    episode_id: str = ""
    source_schema: str = LEARNING_EPISODE_SCHEMA
    terminal_evidence: dict[str, Any] = field(default_factory=dict)
    stages: dict[str, bool] = field(default_factory=dict)
    qualification: dict[str, Any] = field(default_factory=dict)
    lesson_disposition: str = "shadow"

    @classmethod
    def from_task(
        cls,
        *,
        task_id: str,
        task_type: str,
        task_desc: str,
        solved: bool,
        wall_duration_sec: float,
        total_tokens_used: int,
        trust_mismatch: bool,
        receipts: Iterable[Mapping[str, Any]] = (),
        ab_lift_value: float = 0.0,
        created_at: str | None = None,
        attempt_id: str = "",
        action_id: str = "",
        idempotency_key: str = "",
        terminal_outcome: str | None = None,
        auto_replay_allowed: bool = False,
        retrieved_lesson_ids: Iterable[str] = (),
        applied_lesson_ids: Iterable[str] = (),
        lesson_updates: Iterable[Mapping[str, Any]] = (),
        qualification_evidence_present: bool = False,
        terminal_evidence: Mapping[str, Any] | None = None,
        stages: Mapping[str, bool] | None = None,
        qualification: Mapping[str, Any] | None = None,
        lesson_disposition: str = "shadow",
        source_schema: str = LEARNING_EPISODE_SCHEMA,
    ) -> "EpisodeOutcomeRecord":
        normalized_terminal = str(terminal_outcome or ("SUCCEEDED" if solved else "PARKED")).upper()
        if normalized_terminal not in TERMINAL_OUTCOMES:
            normalized_terminal = "PARKED"
        normalized_receipts = [dict(item) for item in receipts if isinstance(item, Mapping)]
        evidence = dict(terminal_evidence or {})
        inferred_evidence = bool(qualification_evidence_present) or bool(
            evidence.get("verifier") or evidence.get("verifier_status") or evidence.get("receipt")
            or any(
                isinstance(item, Mapping) and item.get("evidence_present") and item.get("gate_passed")
                for item in normalized_receipts
            )
        )
        qualified = (
            normalized_terminal in QUALIFIED_TERMINAL_OUTCOMES
            and not bool(trust_mismatch)
            and inferred_evidence
        )
        stable_identity = str(idempotency_key or f"{task_id}:{attempt_id}:{action_id}")
        episode_id = "lep:" + hashlib.sha256(stable_identity.encode("utf-8")).hexdigest()[:24]
        stage_map = dict(stages or {})
        stage_map.setdefault("recorded", True)
        stage_map.setdefault("outcome_measured", bool(evidence) and inferred_evidence)
        stage_map["outcome_uplift_observed"] = bool(
            stage_map.get("outcome_uplift_observed") and _has_paired_uplift_evidence(evidence)
        )
        retrieved_ids = {str(item) for item in retrieved_lesson_ids if str(item)}
        applied_ids = {str(item) for item in applied_lesson_ids if str(item)} & retrieved_ids
        return cls(
            task_id=str(task_id or "unknown"),
            task_type=str(task_type or "unknown"),
            task_desc_hash=_stable_hash(task_desc or ""),
            solved=bool(solved),
            wall_duration_sec=max(0.0, float(wall_duration_sec or 0.0)),
            total_tokens_used=max(0, int(total_tokens_used or 0)),
            trust_mismatch=bool(trust_mismatch),
            receipts=normalized_receipts,
            ab_lift_value=float(ab_lift_value or 0.0),
            created_at=created_at or datetime.now(timezone.utc).isoformat(),
            attempt_id=str(attempt_id or ""),
            action_id=str(action_id or ""),
            idempotency_key=str(idempotency_key or ""),
            terminal_outcome=normalized_terminal,
            auto_replay_allowed=bool(auto_replay_allowed) and qualified,
            qualification_status="QUALIFIED" if qualified else "UNQUALIFIED",
            qualification_evidence_present=inferred_evidence,
            retrieved_lesson_ids=sorted(retrieved_ids),
            applied_lesson_ids=sorted(applied_ids),
            lesson_updates=[dict(item) for item in lesson_updates if isinstance(item, Mapping)],
            episode_id=episode_id,
            source_schema=str(source_schema or LEARNING_EPISODE_SCHEMA),
            terminal_evidence=evidence,
            stages=stage_map,
            qualification=dict(qualification or {}),
            lesson_disposition=str(lesson_disposition or "shadow"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "task_id": self.task_id,
            "task_type": self.task_type,
            "task_desc_hash": self.task_desc_hash,
            "solved": self.solved,
            "wall_duration_sec": self.wall_duration_sec,
            "total_tokens_used": self.total_tokens_used,
            "trust_mismatch": self.trust_mismatch,
            "receipts": [dict(receipt) for receipt in self.receipts],
            "ab_lift_value": self.ab_lift_value,
            "created_at": self.created_at,
            "attempt_id": self.attempt_id,
            "action_id": self.action_id,
            "idempotency_key": self.idempotency_key,
            "terminal_outcome": self.terminal_outcome,
            "auto_replay_allowed": self.auto_replay_allowed,
            "qualification_status": self.qualification_status,
            "qualification_evidence_present": self.qualification_evidence_present,
            "retrieved_lesson_ids": list(self.retrieved_lesson_ids),
            "applied_lesson_ids": list(self.applied_lesson_ids),
            "lesson_updates": [dict(item) for item in self.lesson_updates],
            "episode_id": self.episode_id,
            "source_schema": self.source_schema,
            "terminal_evidence": dict(self.terminal_evidence),
            "stages": dict(self.stages),
            "qualification": dict(self.qualification),
            "lesson_disposition": self.lesson_disposition,
        }


class OutcomeMemoryManager:
    STORAGE_PATH = Path(".nexus") / "memory" / "outcome_history.jsonl"
    POLICY_PATH = Path(".nexus") / "memory" / "dynamic_learning_policy.json"
    RECENT_LIMIT = 20
    MIN_RECENCY_WEIGHT = 0.35

    @classmethod
    async def save_episode_and_tune(
        cls,
        record: EpisodeOutcomeRecord,
        *,
        project_root: Path | LearningStateRoot | None = None,
        allow_dev_cwd_fallback: bool = False,
    ) -> dict[str, Any]:
        return await asyncio.to_thread(
            cls.save_episode_and_tune_sync,
            record,
            project_root=project_root,
            allow_dev_cwd_fallback=allow_dev_cwd_fallback,
        )

    @classmethod
    def save_episode_and_tune_sync(
        cls,
        record: EpisodeOutcomeRecord,
        *,
        project_root: Path | LearningStateRoot | None = None,
        allow_dev_cwd_fallback: bool = False,
    ) -> dict[str, Any]:
        state_root = (
            project_root
            if isinstance(project_root, LearningStateRoot)
            else resolve_learning_state_root(
                project_root, allow_dev_cwd_fallback=allow_dev_cwd_fallback
            )
        )
        storage_path = state_root.outcome_history_path
        storage_path.parent.mkdir(parents=True, exist_ok=True)
        with _locked_outcome_write(storage_path):
            existing_keys = _load_idempotency_keys(storage_path)
            if record.idempotency_key and record.idempotency_key in existing_keys:
                return {
                    "schema_version": "nexus_outcome_memory_write.v1",
                    "status": "IDEMPOTENT_DUPLICATE",
                    "storage_path": str(cls.STORAGE_PATH),
                }
            with storage_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record.to_dict(), ensure_ascii=False, sort_keys=True) + "\n")
        policy = cls.run_dynamic_autotune_sync(
            project_root=state_root, allow_dev_cwd_fallback=allow_dev_cwd_fallback
        )
        return {
            "schema_version": "nexus_outcome_memory_write.v1",
            "status": "PASS",
            "storage_path": str(cls.STORAGE_PATH),
            "policy_path": str(cls.POLICY_PATH),
            "policy": policy,
        }

    @classmethod
    def run_dynamic_autotune_sync(
        cls,
        *,
        project_root: Path | LearningStateRoot | None = None,
        allow_dev_cwd_fallback: bool = False,
    ) -> dict[str, Any]:
        state_root = (
            project_root
            if isinstance(project_root, LearningStateRoot)
            else resolve_learning_state_root(
                project_root, allow_dev_cwd_fallback=allow_dev_cwd_fallback
            )
        )
        records = cls.load_recent_records(
            project_root=state_root,
            limit=cls.RECENT_LIMIT,
            allow_dev_cwd_fallback=allow_dev_cwd_fallback,
        )
        eligible_records = [
            record
            for record in records
            if not bool(record.get("trust_mismatch", False))
            and (
                (
                    str(record.get("qualification_status") or "UNQUALIFIED").upper() == "QUALIFIED"
                    and bool(record.get("qualification_evidence_present", False))
                )
                or any(
                    isinstance(item, Mapping)
                    and item.get("selected")
                    and not item.get("invoked")
                    for item in (record.get("receipts") or [])
                )
            )
            and not bool(record.get("auto_replay_allowed", False))
        ]
        promoted_scores: dict[str, float] = {}
        penalized_scores: dict[str, float] = {}
        weights = _recency_weights(len(eligible_records), minimum=cls.MIN_RECENCY_WEIGHT)
        for record, weight in zip(eligible_records, weights, strict=False):
            solved = bool(record.get("solved", False))
            for receipt in record.get("receipts", []) or []:
                if not isinstance(receipt, Mapping):
                    continue
                name = str(receipt.get("name") or "").strip()
                if not name:
                    continue
                if _receipt_promotes(record_solved=solved, receipt=receipt):
                    promoted_scores[name] = promoted_scores.get(name, 0.0) + weight
                if _receipt_penalizes(record_solved=solved, receipt=receipt):
                    penalized_scores[name] = penalized_scores.get(name, 0.0) + weight
        capability_scores = {
            name: round(promoted_scores.get(name, 0.0) - penalized_scores.get(name, 0.0), 4)
            for name in sorted(set(promoted_scores) | set(penalized_scores))
        }
        resolved_promoted = [name for name, score in capability_scores.items() if score > 0]
        resolved_penalized = [name for name, score in capability_scores.items() if score < 0]
        policy = {
            "schema_version": DYNAMIC_LEARNING_POLICY_SCHEMA,
            "status": "PASS",
            "source_experiences_count": len(records),
            "eligible_experiences_count": len(eligible_records),
            "source_experiences": [str(record.get("task_id") or "") for record in records if record.get("task_id")],
            "excluded_experiences": [
                {
                    "task_id": str(record.get("task_id") or ""),
                    "reason": "trust_mismatch",
                }
                for record in records
                if bool(record.get("trust_mismatch", False))
            ],
            "aging_window": {
                "recent_limit": cls.RECENT_LIMIT,
                "minimum_weight": cls.MIN_RECENCY_WEIGHT,
                "records_used": len(eligible_records),
            },
            "capability_scores": capability_scores,
            "promoted_capabilities": resolved_promoted,
            "penalized_capabilities": resolved_penalized,
            "enforce_penalties": False,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        policy_path = state_root.dynamic_policy_path
        policy_path.parent.mkdir(parents=True, exist_ok=True)
        policy_path.write_text(json.dumps(policy, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
        return policy

    @classmethod
    def append_worker_write(
        cls,
        receipt: dict[str, Any],
        *,
        project_root: Path | LearningStateRoot | None = None,
        allow_dev_cwd_fallback: bool = False,
    ) -> dict[str, Any]:
        if "task_id" not in receipt:
            raise ValueError("Missing required field: task_id")
        if "worker_name" not in receipt:
            raise ValueError("Missing required field: worker_name")
        state_root = (
            project_root
            if isinstance(project_root, LearningStateRoot)
            else resolve_learning_state_root(
                project_root, allow_dev_cwd_fallback=allow_dev_cwd_fallback
            )
        )
        storage_path = state_root.outcome_history_path
        storage_path.parent.mkdir(parents=True, exist_ok=True)
        with storage_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(receipt, ensure_ascii=False, sort_keys=True) + "\n")
        return {
            "status": "PASS",
            "storage_path": str(cls.STORAGE_PATH),
        }

    @classmethod
    def load_recent_records(
        cls,
        *,
        project_root: Path | LearningStateRoot | None = None,
        limit: int = RECENT_LIMIT,
        allow_dev_cwd_fallback: bool = False,
    ) -> list[dict[str, Any]]:
        state_root = (
            project_root
            if isinstance(project_root, LearningStateRoot)
            else resolve_learning_state_root(
                project_root, allow_dev_cwd_fallback=allow_dev_cwd_fallback
            )
        )
        storage_path = state_root.outcome_history_path
        if not storage_path.exists():
            return []
        rows: list[dict[str, Any]] = []
        for line in storage_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                rows.append(payload)
        return rows[-max(1, int(limit or 1)) :]


def _receipt_promotes(*, record_solved: bool, receipt: Mapping[str, Any]) -> bool:
    return bool(
        record_solved
        and receipt.get("public_claim_safe")
        and receipt.get("selected")
        and receipt.get("invoked")
        and receipt.get("evidence_present")
        and receipt.get("gate_passed")
        and receipt.get("outcome_contributed")
    )


def _receipt_penalizes(*, record_solved: bool, receipt: Mapping[str, Any]) -> bool:
    selected = bool(receipt.get("selected"))
    if not selected:
        return False
    if not receipt.get("invoked"):
        return True
    if receipt.get("evidence_present") and not receipt.get("gate_passed"):
        return True
    return bool(not record_solved and not receipt.get("gate_passed"))


def _recency_weights(count: int, *, minimum: float) -> list[float]:
    if count <= 0:
        return []
    if count == 1:
        return [1.0]
    floor = max(0.0, min(1.0, float(minimum)))
    step = (1.0 - floor) / float(count - 1)
    return [round(floor + (step * index), 4) for index in range(count)]


def _resolve(
    project_root: Path | LearningStateRoot | None,
    path: Path,
    *,
    allow_dev_cwd_fallback: bool = False,
) -> Path:
    if path.is_absolute():
        return path
    state_root = (
        project_root
        if isinstance(project_root, LearningStateRoot)
        else resolve_learning_state_root(
            project_root, allow_dev_cwd_fallback=allow_dev_cwd_fallback
        )
    )
    return state_root.root / path


def _load_idempotency_keys(storage_path: Path) -> set[str]:
    """Read every key so deduplication remains correct beyond the recent window."""
    if not storage_path.exists():
        return set()
    keys: set[str] = set()
    for line in storage_path.read_text(encoding="utf-8").splitlines():
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, Mapping) and payload.get("idempotency_key"):
            keys.add(str(payload["idempotency_key"]))
    return keys


def _has_paired_uplift_evidence(evidence: Mapping[str, Any]) -> bool:
    paired = evidence.get("paired_verifier")
    pair: Mapping[str, Any] = paired if isinstance(paired, Mapping) else evidence
    off_val = pair.get("memory_off")
    off: Mapping[str, Any] = off_val if isinstance(off_val, Mapping) else {}
    on_val = pair.get("memory_on")
    on: Mapping[str, Any] = on_val if isinstance(on_val, Mapping) else {}
    fingerprint = str(pair.get("task_fingerprint") or pair.get("task_id") or "")
    if not fingerprint or str(off.get("task_fingerprint") or off.get("task_id") or fingerprint) != fingerprint or str(on.get("task_fingerprint") or on.get("task_id") or fingerprint) != fingerprint:
        return False
    off_status = str(off.get("verifier_status") or off.get("status") or "").lower()
    on_status = str(on.get("verifier_status") or on.get("status") or "").lower()
    return (
        off_status in {"fail", "failed", "blocked", "rejected"}
        and on_status in {"pass", "passed", "success", "succeeded"}
        and bool(off.get("artifact") or off.get("artifact_ref") or off.get("receipt"))
        and bool(on.get("artifact") or on.get("artifact_ref") or on.get("receipt"))
    )


def build_episode_from_receipts(
    *,
    task_id: str,
    task_type: str,
    task_desc: str = "",
    plan: Any = None,
    receipts: list[Any] | None = None,
    attempt_id: str = "",
    action_id: str = "",
    idempotency_key: str = "",
    terminal_outcome: str | None = None,
    retrieved_lesson_ids: Iterable[str] = (),
        applied_lesson_ids: Iterable[str] = (),
        qualification_evidence_present: bool | None = None,
) -> EpisodeOutcomeRecord:
    solved = all(
        getattr(r, "gate_passed", False) for r in (receipts or []) if hasattr(r, "gate_passed")
    )
    trust_mismatch = any(
        not getattr(r, "evidence_alignment", True) for r in (receipts or []) if hasattr(r, "evidence_alignment")
    )
    wall_duration_sec = 0.0
    total_tokens_used = 0
    for r in (receipts or []):
        tel = getattr(r, "telemetries", None) or {}
        wall_duration_sec = max(wall_duration_sec, float(tel.get("wall_time_ms", 0) or 0) / 1000.0)
        total_tokens_used += int(tel.get("token_usage", 0) or 0)
    receipt_dicts: list[dict[str, Any]] = []
    for r in (receipts or []):
        if hasattr(r, "to_dict") and callable(r.to_dict):
            d = r.to_dict()
            receipt_dicts.append(dict(d) if isinstance(d, Mapping) else {"data": str(d)})
        elif hasattr(r, "__dataclass_fields__"):
            import dataclasses
            receipt_dicts.append(dataclasses.asdict(r))
        else:
            receipt_dicts.append({"capability_name": getattr(r, "capability_name", str(type(r).__name__))})
    return EpisodeOutcomeRecord.from_task(
        task_id=task_id,
        task_type=task_type,
        task_desc=task_desc,
        solved=solved,
        wall_duration_sec=wall_duration_sec,
        total_tokens_used=total_tokens_used,
        trust_mismatch=trust_mismatch,
        receipts=receipt_dicts,
        attempt_id=attempt_id,
        action_id=action_id,
        idempotency_key=idempotency_key,
        terminal_outcome=terminal_outcome,
        retrieved_lesson_ids=retrieved_lesson_ids,
        applied_lesson_ids=applied_lesson_ids,
        qualification_evidence_present=(bool(receipt_dicts) if qualification_evidence_present is None else bool(qualification_evidence_present)),
    )


def _stable_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]