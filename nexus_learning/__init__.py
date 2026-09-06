"""Nexus Learning: Evidence-bounded learning and adaptation system."""

from __future__ import annotations

from nexus_learning.closure_effectiveness import (
    EffectivenessReport,
    append_learning_episode,
    evaluate_effectiveness,
    generate_effectiveness_report,
)
from nexus_learning.contracts import (
    LEARNING_POLICY_ADOPTION_SCHEMA,
    LEARNING_POLICY_RECOMMENDATION_SCHEMA,
    LEARNING_POLICY_ROLLBACK_SCHEMA,
    LEARNING_POLICY_VALIDATION_SCHEMA,
    NEXUS_LEARNING_EPISODE_SCHEMA,
    RUNTIME_LEARNING_CLOSURE_SCHEMA,
    build_learning_policy_adoption,
    build_learning_policy_recommendation,
    build_learning_policy_rollback,
    build_nexus_learning_episode,
    build_runtime_learning_closure,
    evaluate_learning_policy_recommendation,
    validate_learning_policy_adoption,
    validate_learning_policy_recommendation,
    validate_nexus_learning_episode,
)
from nexus_learning.effectiveness_measurement import (
    normalize_attempt_row,
    paired_memory_uplift,
    replay_scorecard,
)
from nexus_learning.episode_projection import (
    project_learning_entries,
    reduce_learning_episode_validity,
    semantic_projection_key,
    write_learning_projection,
)
from nexus_learning.outcome_memory import (
    OutcomeMemoryManager,
)
from nexus_learning.retrieval_audit import (
    AuditEntry,
    log_retrieval_audit,
)
from nexus_learning.state_root import (
    LearningStateRoot,
    resolve_learning_state_root,
)

__all__ = [
    "NEXUS_LEARNING_EPISODE_SCHEMA",
    "RUNTIME_LEARNING_CLOSURE_SCHEMA",
    "LEARNING_POLICY_RECOMMENDATION_SCHEMA",
    "LEARNING_POLICY_VALIDATION_SCHEMA",
    "LEARNING_POLICY_ADOPTION_SCHEMA",
    "LEARNING_POLICY_ROLLBACK_SCHEMA",
    "build_nexus_learning_episode",
    "validate_nexus_learning_episode",
    "build_runtime_learning_closure",
    "build_learning_policy_recommendation",
    "validate_learning_policy_recommendation",
    "evaluate_learning_policy_recommendation",
    "build_learning_policy_adoption",
    "validate_learning_policy_adoption",
    "build_learning_policy_rollback",
    "project_learning_entries",
    "write_learning_projection",
    "semantic_projection_key",
    "reduce_learning_episode_validity",
    "EffectivenessReport",
    "append_learning_episode",
    "evaluate_effectiveness",
    "generate_effectiveness_report",
    "normalize_attempt_row",
    "paired_memory_uplift",
    "replay_scorecard",
    "OutcomeMemoryManager",
    "log_retrieval_audit",
    "AuditEntry",
    "LearningStateRoot",
    "resolve_learning_state_root",
]
