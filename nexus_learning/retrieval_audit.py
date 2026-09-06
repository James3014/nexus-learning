from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from nexus_learning.state_root import LearningStateRoot, resolve_learning_state_root

logger = logging.getLogger(__name__)


@dataclass
class AuditEntry:
    """Retrieval audit entry."""
    query: str
    threshold: float
    top_k: int
    embedding_version: str
    hits: list[tuple[str, float]]
    task_type: str = ""
    task_id: str = ""
    trace_id: str = ""
    context: dict[str, Any] | None = None


class RetrievalAuditLogger:
    """Appends structured retrieval events to .nexus/audit/retrieval_log.jsonl"""
    def __init__(self, project_root: Path | LearningStateRoot):
        state_root = (
            project_root
            if isinstance(project_root, LearningStateRoot)
            else resolve_learning_state_root(project_root)
        )
        self.state_root = state_root
        self.log_dir = state_root.audit_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = state_root.retrieval_log_path


    def log(self, entry: AuditEntry) -> None:
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "task_id": entry.task_id,
            "trace_id": entry.trace_id,
            "query": entry.query,
            "task_type": entry.task_type,
            "threshold": entry.threshold,
            "top_k": entry.top_k,
            "embedding_version": entry.embedding_version,
            "hits": [{"skill_id": sid, "score": score} for sid, score in entry.hits],
            "context": entry.context or {}
        }
        try:
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, default=str) + "\n")
        except Exception as e:
            logger.warning("retrieval_audit_logger_failed task_id=%s trace_id=%s: %s", entry.task_id, entry.trace_id, e)


_global_auditor: RetrievalAuditLogger | None = None


def log_retrieval_audit(entry: AuditEntry, project_root: Path | LearningStateRoot) -> None:
    global _global_auditor
    resolved_root = (
        project_root.root
        if isinstance(project_root, LearningStateRoot)
        else resolve_learning_state_root(project_root).root
    )
    if not _global_auditor or _global_auditor.state_root.root != resolved_root:
        _global_auditor = RetrievalAuditLogger(resolved_root)
    _global_auditor.log(entry)
