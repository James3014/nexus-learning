"""Explicit state root contract for Nexus Learning persistent state.

Enforces that persistent writes require an explicit project root or explicit
configured state root, avoiding accidental dependence on process working directory.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class LearningStateRoot:
    """Explicit state root boundary for persistent learning stores and audits."""

    root: Path

    def __post_init__(self) -> None:
        resolved = Path(self.root).resolve()
        object.__setattr__(self, "root", resolved)

    @property
    def nexus_dir(self) -> Path:
        return self.root / ".nexus"

    @property
    def memory_dir(self) -> Path:
        return self.nexus_dir / "memory"

    @property
    def audit_dir(self) -> Path:
        return self.nexus_dir / "audit"

    @property
    def outcome_history_path(self) -> Path:
        return self.memory_dir / "outcome_history.jsonl"

    @property
    def dynamic_policy_path(self) -> Path:
        return self.memory_dir / "dynamic_learning_policy.json"

    @property
    def learning_episodes_path(self) -> Path:
        return self.memory_dir / "learning_episodes.jsonl"

    @property
    def retrieval_log_path(self) -> Path:
        return self.audit_dir / "retrieval_log.jsonl"

    @classmethod
    def from_project_root(cls, project_root: str | Path) -> LearningStateRoot:
        if not project_root:
            raise ValueError("project_root must be a non-empty path")
        return cls(root=Path(project_root))

    @classmethod
    def from_configured_root(cls, state_root: str | Path) -> LearningStateRoot:
        if not state_root:
            raise ValueError("state_root must be a non-empty path")
        return cls(root=Path(state_root))


def resolve_learning_state_root(
    project_root: str | Path | None = None,
    *,
    allow_dev_cwd_fallback: bool = False,
) -> LearningStateRoot:
    """Resolve explicit learning state root.

    Production and integration callers must provide an explicit project_root or
    set the NEXUS_LEARNING_STATE_ROOT environment variable.
    Process cwd fallback is strictly gated by allow_dev_cwd_fallback=True.
    """
    if project_root is not None:
        return LearningStateRoot.from_project_root(project_root)
    env_root = os.getenv("NEXUS_LEARNING_STATE_ROOT")
    if env_root:
        return LearningStateRoot.from_configured_root(env_root)
    if allow_dev_cwd_fallback:
        return LearningStateRoot.from_project_root(Path.cwd())
    raise ValueError(
        "Persistent Learning state requires an explicit project_root, "
        "configured NEXUS_LEARNING_STATE_ROOT, or explicit allow_dev_cwd_fallback=True."
    )
