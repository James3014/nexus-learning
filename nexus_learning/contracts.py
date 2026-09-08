from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

LEARNING_EXPERIENCE_SCHEMA_VERSION = "nexus_learning_experience.v1"
RUNTIME_LEARNING_CLOSURE_SCHEMA = "nexus.runtime_learning_closure.v1"
NEXUS_LEARNING_EPISODE_SCHEMA = "nexus.learning_episode.v1"
LEARNING_PROVENANCE_CONTRACT_SCHEMA = "nexus.learning_provenance_contract.v1"
LEARNING_POLICY_RECOMMENDATION_SCHEMA = "nexus.learning_policy_recommendation.v1"
LEARNING_POLICY_VALIDATION_SCHEMA = "nexus.learning_policy_validation.v1"
LEARNING_POLICY_ADOPTION_SCHEMA = "nexus.learning_policy_adoption.v1"
LEARNING_POLICY_ROLLBACK_SCHEMA = "nexus.learning_policy_rollback.v1"
HISTORICAL_UNKNOWN = "HISTORICAL_UNKNOWN"
RUNTIME_LEARNING_PHASE_CHAIN = ("S", "P", "D", "X", "R", "A", "C")
PHASE_CHAIN = ("S", "P", "X", "D", "R", "A", "C")
HIGH_COST_CAPABILITIES = {
    "research",
    "external_doc_scout",
    "ultra_review",
    "sandbox",
    "swarm",
    "nightshift",
    "research_control_plane",
}


CAPABILITY_TAXONOMY: dict[str, dict[str, Any]] = {
    "direct_mode": {"category": "primary_execution", "phases": ("S", "P", "X", "D", "R", "A", "C")},
    "repair_loop": {"category": "primary_execution", "phases": ("R", "A")},
    "hyper": {"category": "primary_execution", "phases": ("P", "R", "A")},
    "nightshift": {"category": "primary_execution", "phases": ("D", "R", "C")},
    "codeintel": {"category": "recon_context", "phases": ("S", "P", "X", "A")},
    "research": {"category": "recon_context", "phases": ("X", "C")},
    "research_route": {"category": "recon_context", "phases": ("S", "P", "X")},
    "research_control_plane": {"category": "recon_context", "phases": ("X", "R", "A", "C")},
    "xray": {"category": "recon_context", "phases": ("X", "D")},
    "lancedb": {"category": "recon_context", "phases": ("X",)},
    "memory": {"category": "memory_learning", "phases": ("P", "X", "C")},
    "learn_mode": {"category": "memory_learning", "phases": ("X", "A", "C")},
    "learn_scheduler": {"category": "memory_learning", "phases": ("X", "C")},
    "learn_phase_slo": {"category": "memory_learning", "phases": ("P", "X", "D", "R", "A", "C")},
    "autoreason": {"category": "reasoning_acceleration", "phases": ("D", "R", "A")},
    "judge_panel": {"category": "reasoning_acceleration", "phases": ("D", "R", "A")},
    "llm_judge_panel": {"category": "reasoning_acceleration", "phases": ("D", "R", "A")},
    "ddtree": {"category": "reasoning_acceleration", "phases": ("X", "R", "A")},
    "belief": {"category": "reasoning_acceleration", "phases": ("D", "R")},
    "autonomic_router": {"category": "reasoning_acceleration", "phases": ("P", "D")},
    "forecast_gate": {"category": "reasoning_acceleration", "phases": ("P", "D")},
    "swarm": {"category": "collaboration", "phases": ("D", "R", "A")},
    "swarm_quiet_moment": {"category": "collaboration", "phases": ("D", "R", "A")},
    "drone": {"category": "collaboration", "phases": ("R", "A")},
    "multi_agent": {"category": "collaboration", "phases": ("P", "D", "R", "A", "C")},
    "file_lock": {"category": "collaboration", "phases": ("S", "P", "D")},
    "integration_manager": {"category": "collaboration", "phases": ("C",)},
    "mempalace_gate": {"category": "governance_risk", "phases": ("S", "D", "A")},
    "policy_gate": {"category": "governance_risk", "phases": ("S", "P", "D")},
    "capability_gate": {"category": "governance_risk", "phases": ("S", "P", "D")},
    "pregate": {"category": "governance_risk", "phases": ("S", "P")},
    "plan_quality_gate": {"category": "governance_risk", "phases": ("P", "D")},
    "ultra_review": {"category": "governance_risk", "phases": ("D", "A")},
    "artifact_gate": {"category": "validation_delivery", "phases": ("A", "C")},
    "claim_gate": {"category": "validation_delivery", "phases": ("A", "C")},
    "delivery_gate": {"category": "validation_delivery", "phases": ("A", "C")},
    "acceptance_check": {"category": "validation_delivery", "phases": ("A", "C")},
    "sandbox": {"category": "validation_delivery", "phases": ("R", "A")},
    "benchmark": {"category": "self_evolution", "phases": ("C",)},
    "meta_opt": {"category": "self_evolution", "phases": ("C",)},
    "regression_guard": {"category": "self_evolution", "phases": ("A", "C")},
    "registry_sync": {"category": "productization", "phases": ("S", "C")},
    "metabolism": {"category": "productization", "phases": ("C", "S")},
    "oracle_shadow": {"category": "productization", "phases": ("P", "R", "A", "C")},
    "federation": {"category": "productization", "phases": ("X", "R", "C")},
    "ui_validator": {"category": "productization", "phases": ("A",)},
    "stress_test": {"category": "productization", "phases": ("A", "C")},
}


@dataclass(frozen=True)
class CapabilityLifecycle:
    capability: str
    category: str
    phase: str
    selected: bool
    invoked: bool
    evidence: bool
    outcome: bool
    gate_passed: bool = False
    evidence_refs: tuple[str, ...] = ()
    failure_reason: str = ""

    @property
    def funnel_complete(self) -> bool:
        return bool(
            self.selected and self.invoked and self.evidence and self.outcome and self.gate_passed
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self) | {
            "evidence_refs": list(self.evidence_refs),
            "funnel_complete": self.funnel_complete,
        }


@dataclass(frozen=True)
class LearningExperience:
    experience_id: str
    task_id: str
    task_type: str
    phase_continuity: dict[str, Any]
    capability_lifecycle: tuple[CapabilityLifecycle, ...]
    gate_chain: dict[str, str]
    outcome: str
    route_decision_ref: str = ""
    s2t_trace_refs: tuple[str, ...] = ()
    learning_steward_decision: str = "shadow"
    nexus_policy_targets: tuple[str, ...] = ("route_weight", "capability_weight", "s2t_prior")
    model_training_targets: tuple[str, ...] = ("preference_pair", "reward_row")
    promotion_status: str = "shadow"
    schema_version: str = LEARNING_EXPERIENCE_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return asdict(self) | {
            "capability_lifecycle": [item.to_dict() for item in self.capability_lifecycle],
            "s2t_trace_refs": list(self.s2t_trace_refs),
            "nexus_policy_targets": list(self.nexus_policy_targets),
            "model_training_targets": list(self.model_training_targets),
        }


def build_runtime_learning_closure(
    *,
    task_id: str,
    attempt_id: str,
    action_id: str,
    phase_receipts: list[dict[str, Any]] | tuple[dict[str, Any], ...],
    candidate_ref: str = "",
    outcome: str,
    terminal_evidence: dict[str, Any],
    uncertain_mutation: bool = False,
    retrieved_lesson_ids: list[str] | tuple[str, ...] = (),
    applied_lesson_ids: list[str] | tuple[str, ...] = (),
    lesson_disposition: str = "shadow",
    qualification: dict[str, Any] | None = None,
    primary_task_success: bool = False,
    learning_write_succeeded: bool = True,
) -> dict[str, Any]:
    """Build one outcome-linked learning episode without creating new storage."""

    disposition = str(lesson_disposition or "shadow")
    write_ok = bool(learning_write_succeeded)
    canonical = build_nexus_learning_episode(
        task_id=task_id,
        attempt_id=attempt_id,
        action_id=action_id,
        source="runtime_closure",
        terminal_outcome=outcome,
        terminal_evidence=terminal_evidence,
        phase_receipts=phase_receipts,
        retrieved_lesson_ids=retrieved_lesson_ids,
        applied_lesson_ids=applied_lesson_ids,
        qualification=qualification,
        lesson_disposition=disposition,
        learning_write_succeeded=write_ok,
    )
    episode = {
        "schema": RUNTIME_LEARNING_CLOSURE_SCHEMA,
        "task_id": str(task_id),
        "attempt_id": str(attempt_id),
        "action_id": str(action_id),
        "phase_chain": list(RUNTIME_LEARNING_PHASE_CHAIN),
        "phase_receipts": list(phase_receipts),
        "candidate_ref": str(candidate_ref),
        "outcome": str(outcome),
        "terminal_evidence": dict(terminal_evidence or {}),
        "uncertain_mutation": bool(uncertain_mutation),
        "auto_replay_allowed": False,
        "retrieved_lesson_ids": list(canonical["retrieved_lesson_ids"]),
        "applied_lesson_ids": list(canonical["applied_lesson_ids"]),
        "lesson_disposition": disposition,
        "qualification": dict(qualification or {}),
        "learning_write_succeeded": write_ok,
        "primary_task_success": bool(primary_task_success and write_ok),
        "learning_blocker": "" if write_ok else "LEARNING_WRITE_FAILED",
        "episode_id": canonical["episode_id"],
        "idempotency_key": canonical["idempotency_key"],
        "source_schema": NEXUS_LEARNING_EPISODE_SCHEMA,
        "stages": canonical["stages"],
        "qualification_status": canonical["qualification_status"],
        "outcome_uplift_observed": canonical["stages"]["outcome_uplift_observed"],
    }
    validate_runtime_learning_closure(episode)
    return episode


def canonical_episode_identity(
    *,
    task_id: str,
    attempt_id: str = "",
    action_id: str = "",
    source: str = "unknown",
    idempotency_key: str = "",
) -> tuple[str, str]:
    """Resolve the canonical (idempotency_key, episode_id) pair for a payload.

    An explicit, non-blank producer idempotency_key is preserved verbatim;
    otherwise the identity falls back to task_id:attempt_id:action_id:source.
    The episode_id is always ``lep:`` plus the first 24 hex characters of
    SHA-256(identity). Builders and validators must use this one helper so the
    two mutable identity fields cannot be independently tampered without the
    recomputed digest changing.
    """
    task = str(task_id or "unknown")
    attempt = str(attempt_id or "")
    action = str(action_id or "")
    producer = str(source or "unknown")
    key = str(idempotency_key or "")
    identity = key if key.strip() else f"{task}:{attempt}:{action}:{producer}"
    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:24]
    return identity, f"lep:{digest}"


_HEX_24 = frozenset("0123456789abcdef")


def _is_canonical_episode_id(value: Any) -> bool:
    return (
        isinstance(value, str)
        and value.startswith("lep:")
        and len(value) == 4 + 24
        and all(char in _HEX_24 for char in value[4:])
    )


def build_nexus_learning_episode(
    *,
    task_id: str,
    attempt_id: str = "",
    action_id: str = "",
    source: str = "unknown",
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
    """Normalize every Nexus producer into one evidence-gated episode envelope."""
    task = str(task_id or "unknown")
    attempt = str(attempt_id or "")
    action = str(action_id or "")
    producer = str(source or "unknown")
    identity, episode_id = canonical_episode_identity(
        task_id=task,
        attempt_id=attempt,
        action_id=action,
        source=producer,
        idempotency_key=idempotency_key,
    )
    evidence = dict(terminal_evidence or {})
    phase = [dict(item) for item in phase_receipts if isinstance(item, dict)]
    raw_receipts = [dict(item) for item in receipts if isinstance(item, dict)]
    retrieved = sorted({str(item) for item in retrieved_lesson_ids if str(item)})
    # Adoption is only attributable to lessons that were actually retrieved.
    applied = sorted({str(item) for item in applied_lesson_ids if str(item)} & set(retrieved))
    verifier_status = str(evidence.get("verifier_status") or "").strip().lower()
    has_outcome_evidence = bool(evidence) and bool(
        evidence.get("verifier")
        or evidence.get("receipt")
        or evidence.get("paired_verifier")
        or (verifier_status and verifier_status not in {"missing", "unverified", "unknown"})
        # A generic status is not verifier evidence; producers must identify
        # the verifier/receipt that measured the terminal outcome.
    )
    qual = dict(qualification or {})
    qualified = bool(
        has_outcome_evidence
        and qual.get("repeatability")
        and qual.get("prevention_rule")
        and qual.get("authority_qualification")
    )
    outcome_measured = has_outcome_evidence and str(terminal_outcome).upper() in {
        "SUCCEEDED",
        "SUCCESS",
        "FAILED",
        "CANCELLED",
        "BLOCKED",
        "REJECTED",
    }
    stages = {
        "recorded": bool(learning_write_succeeded),
        "retrieved": bool(retrieved),
        "applied": bool(applied),
        "outcome_measured": outcome_measured,
        "outcome_uplift_observed": bool(qualified and paired_memory_uplift_observed(evidence)),
    }
    episode = {
        "schema": NEXUS_LEARNING_EPISODE_SCHEMA,
        "source_schema": NEXUS_LEARNING_EPISODE_SCHEMA,
        "episode_id": episode_id,
        "idempotency_key": identity,
        "task_id": task,
        "attempt_id": attempt,
        "action_id": action,
        "producer": producer,
        "terminal_outcome": str(terminal_outcome or "UNVERIFIED").upper(),
        "terminal_evidence": evidence,
        "phase_receipts": phase,
        "receipts": raw_receipts,
        "retrieved_lesson_ids": retrieved,
        "applied_lesson_ids": applied,
        "qualification": qual,
        "qualification_status": "QUALIFIED" if qualified else "UNQUALIFIED",
        "lesson_disposition": str(lesson_disposition or "shadow"),
        "stages": stages,
        "auto_replay_allowed": False,
        "learning_write_succeeded": bool(learning_write_succeeded),
    }
    validate_nexus_learning_episode(episode)
    return episode


def validate_nexus_learning_episode(episode: dict[str, Any]) -> None:
    required = (
        "schema",
        "source_schema",
        "episode_id",
        "idempotency_key",
        "task_id",
        "producer",
        "terminal_evidence",
        "stages",
    )
    missing = [name for name in required if name not in episode]
    if missing:
        raise ValueError(f"NEXUS_LEARNING_EPISODE_INCOMPLETE:{','.join(missing)}")
    if episode.get("schema") != NEXUS_LEARNING_EPISODE_SCHEMA:
        raise ValueError("NEXUS_LEARNING_EPISODE_SCHEMA_INVALID")
    if episode.get("auto_replay_allowed") is not False:
        raise ValueError("NEXUS_LEARNING_EPISODE_AUTO_REPLAY_FORBIDDEN")
    stored_key = episode.get("idempotency_key")
    if not isinstance(stored_key, str) or not stored_key.strip():
        raise ValueError("NEXUS_LEARNING_EPISODE_EMPTY_IDEMPOTENCY_KEY")
    stored_id = episode.get("episode_id")
    if not _is_canonical_episode_id(stored_id):
        raise ValueError("NEXUS_LEARNING_EPISODE_MALFORMED_EPISODE_ID")
    _, expected_id = canonical_episode_identity(
        task_id=str(episode.get("task_id") or "unknown"),
        attempt_id=str(episode.get("attempt_id") or ""),
        action_id=str(episode.get("action_id") or ""),
        source=str(episode.get("producer") or "unknown"),
        idempotency_key=stored_key,
    )
    if stored_id != expected_id:
        raise ValueError("NEXUS_LEARNING_EPISODE_IDENTITY_MISMATCH")
    stages = episode.get("stages") or {}
    if stages.get("outcome_uplift_observed") and not stages.get("outcome_measured"):
        raise ValueError("NEXUS_LEARNING_EPISODE_UPLIFT_WITHOUT_OUTCOME")
    if episode.get("qualification_status") == "QUALIFIED" and not stages.get("outcome_measured"):
        raise ValueError("NEXUS_LEARNING_EPISODE_QUALIFICATION_WITHOUT_EVIDENCE")


def paired_memory_uplift_observed(evidence: Mapping[str, Any]) -> bool:
    """Require a true memory_off/on paired verifier before claiming uplift."""
    paired = evidence.get("paired_verifier")
    pair: Mapping[str, Any] = paired if isinstance(paired, Mapping) else evidence
    off_val = pair.get("memory_off")
    off: Mapping[str, Any] = off_val if isinstance(off_val, Mapping) else {}
    on_val = pair.get("memory_on")
    on: Mapping[str, Any] = on_val if isinstance(on_val, Mapping) else {}
    fingerprint = str(pair.get("task_fingerprint") or pair.get("task_id") or "")
    off_fp = str(off.get("task_fingerprint") or off.get("task_id") or fingerprint)
    on_fp = str(on.get("task_fingerprint") or on.get("task_id") or fingerprint)
    if not fingerprint or off_fp != fingerprint or on_fp != fingerprint:
        return False
    off_status = str(off.get("verifier_status") or off.get("status") or "").lower()
    on_status = str(on.get("verifier_status") or on.get("status") or "").lower()
    if off_status not in {"failed", "fail", "blocked", "rejected"} or on_status not in {
        "passed",
        "pass",
        "success",
        "succeeded",
    }:
        return False
    return bool(
        (off.get("artifact") or off.get("artifact_ref") or off.get("receipt"))
        and (on.get("artifact") or on.get("artifact_ref") or on.get("receipt"))
    )


def canonical_recommendation_identity(payload: dict[str, Any]) -> tuple[str, str]:
    """Compute content-addressed idempotency key and recommendation_id."""
    clean_payload = {
        k: v for k, v in payload.items() if k not in {"recommendation_id", "recommendation_hash"}
    }
    canonical_repr = json.dumps(clean_payload, sort_keys=True, separators=(",", ":"))
    rec_hash = hashlib.sha256(canonical_repr.encode("utf-8")).hexdigest()
    rec_id = f"lrec:{rec_hash[:24]}"
    return rec_hash, rec_id


def build_learning_policy_recommendation(
    *,
    source_episodes: list[dict[str, Any]] | tuple[dict[str, Any], ...],
    source_evidence_refs: list[str] | tuple[str, ...],
    source_revision: str,
    runtime_identity: str,
    task_fingerprint: str,
    off_arm: dict[str, Any],
    on_arm: dict[str, Any],
    applicable_scope: dict[str, Any],
    recommended_policy_delta: dict[str, Any],
    current_policy: dict[str, Any],
    expected_effect: str,
    rollback_target: dict[str, Any],
    known_risks: list[str] | tuple[str, ...] = (),
    experiment_identity: str = "",
    contract_revision: str = NEXUS_LEARNING_EPISODE_SCHEMA,
    confidence: str = "high",
    claim_ceiling: str = "SUPPORTED_POLICY_RECOMMENDATION",
) -> dict[str, Any]:
    """Construct an exact content-addressed, evidence-bound learning policy recommendation."""
    if not source_episodes:
        raise ValueError("RECOMMENDATION_MISSING_SOURCE_EPISODES")
    for ep in source_episodes:
        validate_nexus_learning_episode(ep)
        if (
            ep.get("lineage_status") == HISTORICAL_UNKNOWN
            or ep.get("episode_id") == HISTORICAL_UNKNOWN
        ):
            raise ValueError("RECOMMENDATION_HISTORICAL_UNKNOWN_PROVENANCE_FORBIDDEN")

    episode_ids = [str(ep["episode_id"]) for ep in source_episodes]

    paired_payload = {
        "task_fingerprint": task_fingerprint,
        "memory_off": off_arm,
        "memory_on": on_arm,
    }
    if not paired_memory_uplift_observed(paired_payload):
        raise ValueError("RECOMMENDATION_PAIRED_UPLIFT_NOT_OBSERVED")

    # Scope validation: Scope must follow evidence and not claim global authority
    if (
        applicable_scope.get("universal_learning_claim")
        or applicable_scope.get("all_models")
        or applicable_scope.get("all_tasks")
    ):
        raise ValueError("RECOMMENDATION_OVERBROAD_SCOPE_FORBIDDEN")

    payload: dict[str, Any] = {
        "schema": LEARNING_POLICY_RECOMMENDATION_SCHEMA,
        "source_episode_ids": episode_ids,
        "source_evidence_refs": list(source_evidence_refs),
        "source_revision": str(source_revision).strip(),
        "runtime_identity": str(runtime_identity).strip(),
        "contract_revision": str(contract_revision).strip(),
        "experiment_identity": str(experiment_identity or f"exp:{task_fingerprint}"),
        "task_fingerprint": str(task_fingerprint).strip(),
        "off_arm": dict(off_arm),
        "on_arm": dict(on_arm),
        "observed_effect": "paired_memory_uplift_observed",
        "effect_measurement": "verifier_pass_uplift",
        "confidence": str(confidence),
        "applicable_scope": dict(applicable_scope),
        "recommended_policy_delta": dict(recommended_policy_delta),
        "current_policy": dict(current_policy),
        "expected_effect": str(expected_effect),
        "known_risks": list(known_risks),
        "rollback_target": dict(rollback_target),
        "claim_ceiling": str(claim_ceiling),
        "status": "PROPOSED",
        "direct_mutation_allowed": False,
        "route_mutation_allowed": False,
        "planner_mutation_allowed": False,
    }
    rec_hash, rec_id = canonical_recommendation_identity(payload)
    payload["recommendation_hash"] = rec_hash
    payload["recommendation_id"] = rec_id
    validate_learning_policy_recommendation(payload)
    return payload


def validate_learning_policy_recommendation(recommendation: dict[str, Any]) -> None:
    """Validate content address, authority boundaries, and evidence contracts."""
    if not isinstance(recommendation, dict):
        raise ValueError("RECOMMENDATION_NOT_A_MAPPING")
    if recommendation.get("schema") != LEARNING_POLICY_RECOMMENDATION_SCHEMA:
        raise ValueError("RECOMMENDATION_SCHEMA_INVALID")

    required = (
        "schema",
        "recommendation_id",
        "recommendation_hash",
        "source_episode_ids",
        "source_evidence_refs",
        "source_revision",
        "runtime_identity",
        "contract_revision",
        "task_fingerprint",
        "off_arm",
        "on_arm",
        "observed_effect",
        "applicable_scope",
        "recommended_policy_delta",
        "current_policy",
        "rollback_target",
        "claim_ceiling",
        "status",
    )
    missing = [k for k in required if k not in recommendation]
    if missing:
        raise ValueError(f"RECOMMENDATION_INCOMPLETE:{','.join(missing)}")

    # Content-address check
    stored_hash = recommendation.get("recommendation_hash")
    stored_id = recommendation.get("recommendation_id")
    expected_hash, expected_id = canonical_recommendation_identity(recommendation)
    if stored_hash != expected_hash or stored_id != expected_id:
        raise ValueError("RECOMMENDATION_CONTENT_ADDRESS_MISMATCH")

    # Authority and scope invariants
    if (
        recommendation.get("direct_mutation_allowed")
        or recommendation.get("route_mutation_allowed")
        or recommendation.get("planner_mutation_allowed")
    ):
        raise ValueError("RECOMMENDATION_CANNOT_AUTHORIZE_MUTATION")

    delta = recommendation.get("recommended_policy_delta", {})
    if any(
        k in delta
        for k in (
            "CapabilityPlanner",
            "route_authority",
            "workforce_admission",
            "active_route_override",
        )
    ):
        raise ValueError("RECOMMENDATION_DIRECT_PLANNER_MUTATION_FORBIDDEN")

    scope = recommendation.get("applicable_scope", {})
    if (
        not scope
        or scope.get("all_tasks")
        or scope.get("all_models")
        or scope.get("universal_learning_claim")
    ):
        raise ValueError("RECOMMENDATION_SCOPE_OVERBROAD")

    if not recommendation.get("rollback_target"):
        raise ValueError("RECOMMENDATION_MISSING_ROLLBACK_TARGET")

    # Paired verifier check
    paired = {
        "task_fingerprint": recommendation.get("task_fingerprint"),
        "memory_off": recommendation.get("off_arm"),
        "memory_on": recommendation.get("on_arm"),
    }
    if not paired_memory_uplift_observed(paired):
        raise ValueError("RECOMMENDATION_EVIDENCE_INVALID")


def canonical_validation_identity(payload: dict[str, Any]) -> tuple[str, str]:
    """Compute content-addressed idempotency key and validation_id."""
    clean_payload = {
        k: v for k, v in payload.items() if k not in {"validation_id", "validation_hash"}
    }
    canonical_repr = json.dumps(clean_payload, sort_keys=True, separators=(",", ":"))
    val_hash = hashlib.sha256(canonical_repr.encode("utf-8")).hexdigest()
    val_id = f"lval:{val_hash[:24]}"
    return val_hash, val_id


def evaluate_learning_policy_recommendation(
    recommendation: dict[str, Any],
    *,
    validator_identity: str,
    current_workspace_revision: str,
    current_runtime_identity: str,
) -> dict[str, Any]:
    """Independent G6 validator adjudicating recommendation admissibility."""
    hostile_probes: dict[str, str] = {}
    blockers: list[str] = []

    # 1. Structural & Content Address Verification
    try:
        validate_learning_policy_recommendation(recommendation)
        hostile_probes["tamper_check"] = "PASS"
    except Exception as exc:
        hostile_probes["tamper_check"] = f"FAIL:{exc.__class__.__name__}"
        blockers.append(f"recommendation_invalid:{str(exc)}")

    # 2. Freshness Check
    source_rev = str(recommendation.get("source_revision") or "").strip()
    runtime_id = str(recommendation.get("runtime_identity") or "").strip()
    fresh = True
    if source_rev != current_workspace_revision.strip():
        fresh = False
        hostile_probes["source_freshness"] = f"STALE:{source_rev}!={current_workspace_revision}"
        blockers.append("source_revision_stale")
    else:
        hostile_probes["source_freshness"] = "PASS"

    if runtime_id != current_runtime_identity.strip():
        fresh = False
        hostile_probes["runtime_identity_freshness"] = (
            f"MISMATCH:{runtime_id}!={current_runtime_identity}"
        )
        blockers.append("runtime_identity_mismatch")
    else:
        hostile_probes["runtime_identity_freshness"] = "PASS"

    # 3. Causal & Evidence Sufficiency Check
    evidence_refs = recommendation.get("source_evidence_refs") or []
    has_retrieval = any("retrieval_receipt" in str(r) or "receipt" in str(r) for r in evidence_refs)
    has_consumption = any(
        "consumption" in str(r) or "ollama" in str(r) or "metrics" in str(r) for r in evidence_refs
    )
    if not evidence_refs or not has_retrieval:
        blockers.append("missing_retrieval_receipt")
        hostile_probes["retrieval_receipt"] = "FAIL:missing_retrieval_receipt"
    else:
        hostile_probes["retrieval_receipt"] = "PASS"

    if not has_consumption:
        blockers.append("missing_physical_consumption")
        hostile_probes["physical_consumption"] = "FAIL:missing_physical_consumption"
    else:
        hostile_probes["physical_consumption"] = "PASS"

    # 4. Authority Boundary Enforcement
    if (
        recommendation.get("direct_mutation_allowed")
        or recommendation.get("route_mutation_allowed")
        or recommendation.get("planner_mutation_allowed")
    ):
        blockers.append("violates_authority_boundary")
        hostile_probes["authority_boundary"] = "FAIL"
    else:
        hostile_probes["authority_boundary"] = "PASS"

    # 5. Rollback Readiness
    rollback = recommendation.get("rollback_target")
    if not rollback or not isinstance(rollback, dict) or not rollback.get("target_state"):
        blockers.append("rollback_not_defined")
        hostile_probes["rollback_readiness"] = "FAIL"
    else:
        hostile_probes["rollback_readiness"] = "PASS"

    # 6. Scope Boundaries
    scope = recommendation.get("applicable_scope") or {}
    if not scope or scope.get("all_models") or scope.get("all_tasks"):
        blockers.append("scope_overbroad")
        hostile_probes["scope_conformance"] = "FAIL"
    else:
        hostile_probes["scope_conformance"] = "PASS"

    # Disposition Determination
    if blockers:
        disposition = (
            "REJECTED_RECOMMENDATION"
            if any("violates" in b or "tamper" in b or "invalid" in b for b in blockers)
            else "INSUFFICIENT_EVIDENCE"
        )
    else:
        disposition = "VALIDATED_FOR_ADOPTION_CONSIDERATION"

    validation_payload: dict[str, Any] = {
        "schema": LEARNING_POLICY_VALIDATION_SCHEMA,
        "recommendation_id": recommendation.get("recommendation_id", ""),
        "recommendation_hash": recommendation.get("recommendation_hash", ""),
        "validator_identity": validator_identity,
        "validation_source_revision": current_workspace_revision,
        "validated_evidence_refs": list(evidence_refs),
        "scope": dict(scope),
        "freshness_result": "FRESH" if fresh else "STALE",
        "tamper_check_result": hostile_probes.get("tamper_check", "FAIL"),
        "authority_boundary_result": hostile_probes.get("authority_boundary", "FAIL"),
        "rollback_readiness_result": hostile_probes.get("rollback_readiness", "FAIL"),
        "hostile_probes": hostile_probes,
        "validation_disposition": disposition,
        "blockers": blockers,
        "claim_ceiling": "SUPPORTED_POLICY_RECOMMENDATION",
    }
    val_hash, val_id = canonical_validation_identity(validation_payload)
    validation_payload["validation_hash"] = val_hash
    validation_payload["validation_id"] = val_id
    return validation_payload


def canonical_adoption_identity(payload: dict[str, Any]) -> tuple[str, str]:
    """Compute content-addressed idempotency key and adoption_id."""
    clean_payload = {k: v for k, v in payload.items() if k not in {"adoption_id", "adoption_hash"}}
    canonical_repr = json.dumps(clean_payload, sort_keys=True, separators=(",", ":"))
    ad_hash = hashlib.sha256(canonical_repr.encode("utf-8")).hexdigest()
    ad_id = f"ladopt:{ad_hash[:24]}"
    return ad_hash, ad_id


def build_learning_policy_adoption(
    *,
    owner_authority_reference: str,
    recommendation: dict[str, Any],
    validation: dict[str, Any],
    source_revision: str,
    adopted_scope: dict[str, Any],
    target_policy_delta: dict[str, Any],
    previous_policy: dict[str, Any],
    rollback_target: dict[str, Any],
    effective_candidate_generation: str = "candidate_r7_g7",
) -> dict[str, Any]:
    """Construct an exact Owner-bound governed policy adoption artifact."""
    if not owner_authority_reference or not str(owner_authority_reference).strip():
        raise ValueError("ADOPTION_OWNER_AUTHORITY_MISSING")

    # Validate incoming recommendation and validation
    validate_learning_policy_recommendation(recommendation)
    if validation.get("schema") != LEARNING_POLICY_VALIDATION_SCHEMA:
        raise ValueError("ADOPTION_VALIDATION_SCHEMA_INVALID")
    if validation.get("validation_disposition") != "VALIDATED_FOR_ADOPTION_CONSIDERATION":
        raise ValueError("ADOPTION_VALIDATION_DISPOSITION_INVALID")

    rec_id = recommendation.get("recommendation_id")
    rec_hash = recommendation.get("recommendation_hash")
    if (
        validation.get("recommendation_id") != rec_id
        or validation.get("recommendation_hash") != rec_hash
    ):
        raise ValueError("ADOPTION_RECOMMENDATION_VALIDATION_MISMATCH")

    # Recheck validation hash
    val_clean = {
        k: v for k, v in validation.items() if k not in {"validation_id", "validation_hash"}
    }
    val_canonical = json.dumps(val_clean, sort_keys=True, separators=(",", ":"))
    computed_val_hash = hashlib.sha256(val_canonical.encode("utf-8")).hexdigest()
    if validation.get("validation_hash") != computed_val_hash:
        raise ValueError("ADOPTION_VALIDATION_HASH_TAMPERED")

    # Scope confinement: Must not exceed recommendation scope
    rec_scope = recommendation.get("applicable_scope", {})
    if (
        adopted_scope.get("universal_learning_claim")
        or adopted_scope.get("all_models")
        or adopted_scope.get("all_tasks")
    ):
        raise ValueError("ADOPTION_SCOPE_OVERBROAD")
    if str(adopted_scope.get("task_family") or "") != str(rec_scope.get("task_family") or ""):
        raise ValueError("ADOPTION_SCOPE_TASK_FAMILY_MISMATCH")
    if str(adopted_scope.get("model_name") or "") != str(rec_scope.get("model_name") or ""):
        raise ValueError("ADOPTION_SCOPE_MODEL_MISMATCH")
    if str(adopted_scope.get("runtime_identity") or "") != str(
        rec_scope.get("runtime_identity") or ""
    ):
        raise ValueError("ADOPTION_SCOPE_RUNTIME_MISMATCH")

    # Recheck source revision freshness
    if str(source_revision).strip() != str(recommendation.get("source_revision") or "").strip():
        raise ValueError("ADOPTION_SOURCE_REVISION_STALE")

    # Rollback readiness
    if (
        not rollback_target
        or not isinstance(rollback_target, dict)
        or not rollback_target.get("target_state")
    ):
        raise ValueError("ADOPTION_MISSING_ROLLBACK_TARGET")

    # Authority invariant: No direct planner route override allowed
    if any(
        k in target_policy_delta
        for k in (
            "CapabilityPlanner",
            "route_authority",
            "workforce_admission",
            "active_route_override",
        )
    ):
        raise ValueError("ADOPTION_DIRECT_PLANNER_MUTATION_FORBIDDEN")

    prev_canonical = json.dumps(previous_policy, sort_keys=True, separators=(",", ":"))
    prev_hash = hashlib.sha256(prev_canonical.encode("utf-8")).hexdigest()

    delta_canonical = json.dumps(target_policy_delta, sort_keys=True, separators=(",", ":"))
    delta_hash = hashlib.sha256(delta_canonical.encode("utf-8")).hexdigest()

    payload: dict[str, Any] = {
        "schema": LEARNING_POLICY_ADOPTION_SCHEMA,
        "owner_authority_reference": str(owner_authority_reference).strip(),
        "recommendation_id": rec_id,
        "recommendation_hash": rec_hash,
        "validation_id": validation.get("validation_id"),
        "validation_hash": validation.get("validation_hash"),
        "source_revision": str(source_revision).strip(),
        "adopted_scope": dict(adopted_scope),
        "target_policy_delta": dict(target_policy_delta),
        "target_policy_hash": delta_hash,
        "previous_policy": dict(previous_policy),
        "previous_policy_hash": prev_hash,
        "effective_candidate_generation": str(effective_candidate_generation),
        "rollback_target": dict(rollback_target),
        "adoption_status": "ACTIVE_CANDIDATE",
        "claim_ceiling": "SUPPORTED_POLICY_RECOMMENDATION",
        "route_truth_source": "CapabilityPlanner",
        "direct_route_mutation_allowed": False,
    }
    ad_hash, ad_id = canonical_adoption_identity(payload)
    payload["adoption_hash"] = ad_hash
    payload["adoption_id"] = ad_id
    validate_learning_policy_adoption(payload)
    return payload


def validate_learning_policy_adoption(adoption: dict[str, Any]) -> None:
    """Validate adoption identity, content address, and fail-closed bounds."""
    if not isinstance(adoption, dict):
        raise ValueError("ADOPTION_NOT_A_MAPPING")
    if adoption.get("schema") != LEARNING_POLICY_ADOPTION_SCHEMA:
        raise ValueError("ADOPTION_SCHEMA_INVALID")

    required = (
        "schema",
        "adoption_id",
        "adoption_hash",
        "owner_authority_reference",
        "recommendation_id",
        "recommendation_hash",
        "validation_id",
        "validation_hash",
        "source_revision",
        "adopted_scope",
        "target_policy_delta",
        "target_policy_hash",
        "previous_policy",
        "previous_policy_hash",
        "rollback_target",
        "adoption_status",
        "claim_ceiling",
        "route_truth_source",
    )
    missing = [k for k in required if k not in adoption]
    if missing:
        raise ValueError(f"ADOPTION_INCOMPLETE:{','.join(missing)}")

    # Content-address check
    stored_hash = adoption.get("adoption_hash")
    stored_id = adoption.get("adoption_id")
    expected_hash, expected_id = canonical_adoption_identity(adoption)
    if stored_hash != expected_hash or stored_id != expected_id:
        raise ValueError("ADOPTION_CONTENT_ADDRESS_MISMATCH")

    # Owner authority check
    if not adoption.get("owner_authority_reference"):
        raise ValueError("ADOPTION_OWNER_AUTHORITY_MISSING")

    # Authority invariants
    if adoption.get("direct_route_mutation_allowed") is not False:
        raise ValueError("ADOPTION_DIRECT_ROUTE_MUTATION_FORBIDDEN")
    if adoption.get("route_truth_source") != "CapabilityPlanner":
        raise ValueError("ADOPTION_ROUTE_TRUTH_SOURCE_INVALID")

    # Scope check
    scope = adoption.get("adopted_scope", {})
    if (
        not scope
        or scope.get("all_tasks")
        or scope.get("all_models")
        or scope.get("universal_learning_claim")
    ):
        raise ValueError("ADOPTION_SCOPE_OVERBROAD")

    # Rollback check
    if not adoption.get("rollback_target"):
        raise ValueError("ADOPTION_MISSING_ROLLBACK_TARGET")


def canonical_rollback_identity(payload: dict[str, Any]) -> tuple[str, str]:
    """Compute content-addressed idempotency key and rollback_id."""
    clean_payload = {k: v for k, v in payload.items() if k not in {"rollback_id", "rollback_hash"}}
    canonical_repr = json.dumps(clean_payload, sort_keys=True, separators=(",", ":"))
    rb_hash = hashlib.sha256(canonical_repr.encode("utf-8")).hexdigest()
    rb_id = f"lroll:{rb_hash[:24]}"
    return rb_hash, rb_id


def build_learning_policy_rollback(
    *,
    adoption: dict[str, Any],
    reason: str,
    triggered_by: str = "governed_rollback_test",
) -> dict[str, Any]:
    """Execute a governed reversal of an active adoption artifact."""
    validate_learning_policy_adoption(adoption)
    if adoption.get("adoption_status") != "ACTIVE_CANDIDATE":
        raise ValueError("ROLLBACK_ADOPTION_NOT_ACTIVE")

    rollback_target = adoption.get("rollback_target", {})
    target_state = rollback_target.get("target_state", {})
    if not target_state:
        raise ValueError("ROLLBACK_TARGET_STATE_MISSING")

    payload: dict[str, Any] = {
        "schema": LEARNING_POLICY_ROLLBACK_SCHEMA,
        "adoption_id": adoption.get("adoption_id"),
        "adoption_hash": adoption.get("adoption_hash"),
        "recommendation_id": adoption.get("recommendation_id"),
        "recommendation_hash": adoption.get("recommendation_hash"),
        "previous_policy_hash": adoption.get("target_policy_hash"),
        "rolled_back_policy": dict(target_state),
        "rolled_back_policy_hash": hashlib.sha256(
            json.dumps(target_state, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        "rollback_reason": str(reason),
        "triggered_by": str(triggered_by),
        "rollback_status": "ROLLED_BACK",
        "route_truth_source": "CapabilityPlanner",
    }
    rb_hash, rb_id = canonical_rollback_identity(payload)
    payload["rollback_hash"] = rb_hash
    payload["rollback_id"] = rb_id
    validate_learning_policy_rollback(payload)
    return payload


def validate_learning_policy_rollback(rollback: dict[str, Any]) -> None:
    """Validate rollback artifact integrity and hash."""
    if not isinstance(rollback, dict):
        raise ValueError("ROLLBACK_NOT_A_MAPPING")
    if rollback.get("schema") != LEARNING_POLICY_ROLLBACK_SCHEMA:
        raise ValueError("ROLLBACK_SCHEMA_INVALID")

    required = (
        "schema",
        "rollback_id",
        "rollback_hash",
        "adoption_id",
        "adoption_hash",
        "recommendation_id",
        "recommendation_hash",
        "previous_policy_hash",
        "rolled_back_policy",
        "rolled_back_policy_hash",
        "rollback_reason",
        "rollback_status",
        "route_truth_source",
    )
    missing = [k for k in required if k not in rollback]
    if missing:
        raise ValueError(f"ROLLBACK_INCOMPLETE:{','.join(missing)}")

    stored_hash = rollback.get("rollback_hash")
    stored_id = rollback.get("rollback_id")
    expected_hash, expected_id = canonical_rollback_identity(rollback)
    if stored_hash != expected_hash or stored_id != expected_id:
        raise ValueError("ROLLBACK_CONTENT_ADDRESS_MISMATCH")


def project_adoption_into_planner_budget(
    adoption: dict[str, Any],
    *,
    task_desc: str,
    target_model: str = "qwen2.5-coder:7b",
    runtime_identity: str = "local_model_executor",
    rollback: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Pure, deterministic projection of an adoption artifact into CapabilityPlanner budget."""
    validate_learning_policy_adoption(adoption)
    if rollback is not None:
        validate_learning_policy_rollback(rollback)
        if rollback.get("adoption_id") != adoption.get("adoption_id"):
            raise ValueError("PROJECTION_ROLLBACK_ADOPTION_MISMATCH")
        # Rollback active: return clean inactive budget
        return {
            "learning_policy": {
                "episodic_memory_injection": {"enabled": False},
                "adoption_lineage": {
                    "adoption_id": adoption.get("adoption_id"),
                    "status": "ROLLED_BACK",
                    "rollback_id": rollback.get("rollback_id"),
                },
            }
        }

    if adoption.get("adoption_status") != "ACTIVE_CANDIDATE":
        return {
            "learning_policy": {
                "episodic_memory_injection": {"enabled": False},
                "adoption_lineage": {"status": "INACTIVE"},
            }
        }

    scope = adoption.get("adopted_scope", {})
    # Scope checking: task_family, model, runtime
    scope_family = str(scope.get("task_family") or "")
    scope_model = str(scope.get("model_name") or "")
    scope_runtime = str(scope.get("runtime_identity") or "")

    in_scope_task = bool(scope_family and scope_family in task_desc.lower())
    in_scope_model = bool(not scope_model or scope_model == target_model)
    in_scope_runtime = bool(not scope_runtime or scope_runtime == runtime_identity)

    if in_scope_task and in_scope_model and in_scope_runtime:
        delta = adoption.get("target_policy_delta", {})
        injection_spec = delta.get("episodic_memory_injection", {})
        enabled = bool(injection_spec.get("enabled", True))
        return {
            "learning_policy": {
                "episodic_memory_injection": {
                    "enabled": enabled,
                    "scope": scope_family,
                },
                "adoption_lineage": {
                    "adoption_id": adoption.get("adoption_id"),
                    "adoption_hash": adoption.get("adoption_hash"),
                    "recommendation_id": adoption.get("recommendation_id"),
                    "validation_id": adoption.get("validation_id"),
                    "policy_hash": adoption.get("target_policy_hash"),
                    "scope": dict(scope),
                    "status": "ACTIVE_CANDIDATE",
                },
            }
        }

    return {
        "learning_policy": {
            "episodic_memory_injection": {"enabled": False},
            "adoption_lineage": {
                "adoption_id": adoption.get("adoption_id"),
                "status": "OUT_OF_SCOPE",
                "in_scope_task": in_scope_task,
                "in_scope_model": in_scope_model,
                "in_scope_runtime": in_scope_runtime,
            },
        }
    }


def validate_runtime_learning_closure(episode: dict[str, Any]) -> None:
    required = (
        "schema",
        "task_id",
        "attempt_id",
        "action_id",
        "phase_receipts",
        "outcome",
        "terminal_evidence",
        "auto_replay_allowed",
        "lesson_disposition",
        "learning_write_succeeded",
        "primary_task_success",
    )
    missing = [item for item in required if item not in episode]
    if missing:
        raise ValueError(f"RUNTIME_LEARNING_CLOSURE_INCOMPLETE:{','.join(missing)}")
    if episode.get("schema") != RUNTIME_LEARNING_CLOSURE_SCHEMA:
        raise ValueError("RUNTIME_LEARNING_CLOSURE_SCHEMA_INVALID")
    if episode.get("auto_replay_allowed") is not False:
        raise ValueError("RUNTIME_LEARNING_AUTO_REPLAY_FORBIDDEN")
    if episode.get("uncertain_mutation") and episode.get("auto_replay_allowed"):
        raise ValueError("RUNTIME_LEARNING_UNCERTAIN_REPLAY_FORBIDDEN")
    if (
        str(episode.get("outcome") or "").upper() in {"FAILED", "BLOCKED", "REJECTED"}
        and episode.get("lesson_disposition") == "graduated"
    ):
        raise ValueError("RUNTIME_LEARNING_FAILED_ATTEMPT_CANNOT_GRADUATE")
    if episode.get("lesson_disposition") == "graduated":
        qualification = episode.get("qualification") or {}
        required_qualification = (
            "terminal_evidence",
            "repeatability",
            "prevention_rule",
            "authority_qualification",
        )
        if not episode.get("terminal_evidence") or any(
            not qualification.get(field) for field in required_qualification[1:]
        ):
            raise ValueError("RUNTIME_LEARNING_QUALIFICATION_INCOMPLETE")
    if episode.get("primary_task_success") and not episode.get("learning_write_succeeded"):
        raise ValueError("RUNTIME_LEARNING_WRITE_FAILURE_CANNOT_REPORT_SUCCESS")


def learning_experience_from_dict(payload: dict[str, Any]) -> LearningExperience:
    lifecycle = tuple(
        CapabilityLifecycle(
            capability=str(item.get("capability", "unknown")),
            category=str(item.get("category", "unknown")),
            phase=str(item.get("phase", "C")),
            selected=bool(item.get("selected", False)),
            invoked=bool(item.get("invoked", False)),
            evidence=bool(item.get("evidence", False)),
            outcome=bool(item.get("outcome", False)),
            gate_passed=bool(item.get("gate_passed", False)),
            evidence_refs=tuple(str(ref) for ref in item.get("evidence_refs", []) or []),
            failure_reason=str(item.get("failure_reason", "")),
        )
        for item in payload.get("capability_lifecycle", []) or []
        if isinstance(item, dict)
    )
    return LearningExperience(
        experience_id=str(payload.get("experience_id") or "exp:unknown"),
        task_id=str(payload.get("task_id") or "unknown"),
        task_type=str(payload.get("task_type") or ""),
        phase_continuity=dict(payload.get("phase_continuity", {}) or {}),
        capability_lifecycle=lifecycle,
        gate_chain=dict(payload.get("gate_chain", {}) or {}),
        outcome=str(payload.get("outcome") or "unverified"),
        route_decision_ref=str(payload.get("route_decision_ref") or ""),
        s2t_trace_refs=tuple(str(ref) for ref in payload.get("s2t_trace_refs", []) or []),
        learning_steward_decision=str(payload.get("learning_steward_decision") or "shadow"),
        nexus_policy_targets=tuple(
            str(item) for item in payload.get("nexus_policy_targets", []) or ()
        ),
        model_training_targets=tuple(
            str(item) for item in payload.get("model_training_targets", []) or ()
        ),
        promotion_status=str(payload.get("promotion_status") or "shadow"),
        schema_version=str(payload.get("schema_version") or LEARNING_EXPERIENCE_SCHEMA_VERSION),
    )


def build_learning_experience(
    *,
    task_id: str,
    task_type: str = "",
    usage_trace: dict[str, Any] | None = None,
    capability_receipts: list[dict[str, Any]] | None = None,
    route_decision_ref: str = "",
    learning_steward_decision: str = "shadow",
) -> LearningExperience:
    usage = usage_trace or {}
    receipts = capability_receipts or usage.get("capability_receipts", []) or []
    phase_trace = usage.get("phase_trace", {}) if isinstance(usage.get("phase_trace"), dict) else {}
    observed = [
        phase
        for phase in PHASE_CHAIN
        if phase in phase_trace or phase in usage.get("phase_wall_sec", {})
    ]
    capabilities = tuple(
        _lifecycle_from_receipt(item) for item in receipts if isinstance(item, dict)
    )
    gate_chain = _gate_chain(usage)
    outcome = (
        "verified_success"
        if all(gate_chain.get(key) == "pass" for key in ("artifact", "claim", "delivery"))
        else "unverified"
    )
    s2t = usage.get("s2t", {}) if isinstance(usage.get("s2t"), dict) else {}
    s2t_refs = tuple(str(ref) for ref in [s2t.get("trace_path", "")] if str(ref).strip())
    experience_id = f"exp:{task_id or 'unknown'}:{abs(hash((task_id, len(capabilities), outcome))) % 10_000_000}"
    return LearningExperience(
        experience_id=experience_id,
        task_id=task_id or "unknown",
        task_type=task_type,
        phase_continuity={
            "expected": list(PHASE_CHAIN),
            "observed": observed,
            "complete": observed == list(PHASE_CHAIN),
            "broken_at": _first_missing_phase(observed),
            "phase_wall_sec": dict(usage.get("phase_wall_sec", {}) or {}),
        },
        capability_lifecycle=capabilities,
        gate_chain=gate_chain,
        outcome=outcome,
        route_decision_ref=route_decision_ref,
        s2t_trace_refs=s2t_refs,
        learning_steward_decision=learning_steward_decision,
        promotion_status="shadow" if outcome == "verified_success" else "frozen",
    )


def project_nexus_policy(experience: LearningExperience) -> dict[str, Any]:
    complete = [item.capability for item in experience.capability_lifecycle if item.funnel_complete]
    unnecessary = [
        item.capability
        for item in experience.capability_lifecycle
        if item.selected and not item.invoked
    ]
    escalation = build_escalation_recommendations(experience)
    return {
        "schema_version": "nexus_policy_learning_projection.v1",
        "experience_id": experience.experience_id,
        "promotion_status": experience.promotion_status,
        "route_weight_updates": complete,
        "capability_penalties": unnecessary,
        "escalation_recommendations": escalation,
        "s2t_prior_eligible": bool(
            experience.s2t_trace_refs and experience.outcome == "verified_success"
        ),
    }


def project_model_training(experience: LearningExperience) -> dict[str, Any]:
    eligible = (
        experience.outcome == "verified_success" and experience.gate_chain.get("claim") == "pass"
    )
    return {
        "schema_version": "nexus_model_training_projection.v1",
        "experience_id": experience.experience_id,
        "training_eligible": eligible,
        "targets": list(experience.model_training_targets) if eligible else ["hard_negative"],
        "source_trace_refs": list(experience.s2t_trace_refs),
    }


def apply_autodata_quality_gate(
    projection: dict[str, Any], quality_row: dict[str, Any] | None
) -> dict[str, Any]:
    """Fail closed model export when trajectory quality is not training-grade."""
    gated = dict(projection)
    reasons: list[str] = []

    def fail_closed() -> None:
        gated["training_eligible"] = False
        gated["targets"] = ["hard_negative"]

    if not gated.get("source_trace_refs"):
        reasons.append("missing_s2t_trace_refs")
    if not quality_row:
        reasons.append("missing_autodata_quality_row")
        fail_closed()
        gated["autodata_gate"] = {"attached": False, "status": "not_attached"}
        gated["model_training_gate"] = {"status": "fail", "reasons": reasons}
        return gated

    eligible = bool(quality_row.get("eligible_for_training", False))
    if not eligible:
        reasons.append("autodata_not_training_eligible")
    if bool(quality_row.get("leakage_risk", False)):
        reasons.append("leakage_risk")
    if bool(quality_row.get("reward_hacking_risk", False)):
        reasons.append("reward_hacking_risk")
    quality_reasons = [
        str(reason) for reason in quality_row.get("reasons", []) or [] if str(reason).strip()
    ]
    if any("leakage" in reason for reason in quality_reasons):
        reasons.append("leakage_risk")
    if any("reward_hacking" in reason for reason in quality_reasons):
        reasons.append("reward_hacking_risk")

    gated["training_eligible"] = bool(gated.get("training_eligible") and eligible)
    if reasons:
        fail_closed()
    gated["autodata_gate"] = {
        "attached": True,
        "status": "pass" if eligible and not reasons else "fail",
        "reasons": quality_reasons,
        "trajectory_steps": int(
            quality_row.get("trajectory_steps", quality_row.get("trajectory_step_count", 0)) or 0
        ),
        "information_density": float(quality_row.get("information_density", 0.0) or 0.0),
    }
    gated["model_training_gate"] = {
        "status": "pass" if gated["training_eligible"] else "fail",
        "reasons": sorted(set(reasons)),
    }
    return gated


def build_escalation_recommendations(experience: LearningExperience) -> list[dict[str, str]]:
    recommendations: list[dict[str, str]] = []
    by_capability = {item.capability: item for item in experience.capability_lifecycle}
    hyper = by_capability.get("hyper")
    if hyper and hyper.selected and hyper.invoked and not hyper.outcome:
        recommendations.append(
            {
                "from": "hyper",
                "to": "nightshift",
                "reason": "hyper_invoked_without_outcome",
            }
        )
    autoreason = by_capability.get("autoreason")
    if autoreason and autoreason.selected and not autoreason.evidence:
        recommendations.append(
            {
                "from": "autoreason",
                "to": "judge_panel",
                "reason": "autoreason_selected_without_evidence",
            }
        )
    if (
        experience.gate_chain.get("delivery") != "pass"
        and experience.gate_chain.get("artifact") == "pass"
    ):
        recommendations.append(
            {
                "from": "artifact_gate",
                "to": "delivery_gate",
                "reason": "artifact_passed_delivery_not_verified",
            }
        )
    return recommendations


def build_promoted_learning_policy(experiences: list[LearningExperience]) -> dict[str, Any]:
    promoted: list[str] = []
    penalized: list[str] = []
    escalation: list[dict[str, str]] = []
    source_experiences: list[str] = []
    for exp in experiences:
        if exp.outcome != "verified_success":
            continue
        projection = project_nexus_policy(exp)
        source_experiences.append(exp.experience_id)
        promoted.extend(str(item) for item in projection["route_weight_updates"])
        penalized.extend(str(item) for item in projection["capability_penalties"])
        escalation.extend(projection["escalation_recommendations"])
    return {
        "schema_version": "nexus_promoted_learning_policy.v1",
        "source_experiences": source_experiences,
        "promoted_capabilities": sorted(set(promoted)),
        "penalized_capabilities": sorted(set(penalized)),
        "escalation_recommendations": escalation,
        "capability_roi": _aggregate_capability_roi(experiences),
        "penalty_candidates": _derive_penalty_candidates(_aggregate_capability_roi(experiences)),
        "enforce_penalties": False,
    }


def save_promoted_learning_policy(
    path: Path, experiences: list[LearningExperience]
) -> dict[str, Any]:
    current = build_promoted_learning_policy(experiences)
    prior = load_promoted_learning_policy(path)
    merged_roi = _merge_capability_roi(
        prior.get("capability_roi", {}) if isinstance(prior, dict) else {},
        current.get("capability_roi", {}),
    )
    penalty_candidates = _derive_penalty_candidates(merged_roi)
    policy = {
        "schema_version": "nexus_promoted_learning_policy.v1",
        "source_experiences": sorted(
            set(
                str(item)
                for item in (prior.get("source_experiences", []) if isinstance(prior, dict) else [])
                or []
            )
            | set(current.get("source_experiences", []) or [])
        ),
        "promoted_capabilities": sorted(
            set(
                str(item)
                for item in (
                    prior.get("promoted_capabilities", []) if isinstance(prior, dict) else []
                )
                or []
            )
            | set(current.get("promoted_capabilities", []) or [])
        ),
        "penalized_capabilities": sorted(
            set(
                str(item)
                for item in (
                    prior.get("penalized_capabilities", []) if isinstance(prior, dict) else []
                )
                or []
            )
            | set(current.get("penalized_capabilities", []) or [])
            | set(penalty_candidates)
        ),
        "escalation_recommendations": list(
            (prior.get("escalation_recommendations", []) if isinstance(prior, dict) else []) or []
        )
        + list(current.get("escalation_recommendations", []) or []),
        "capability_roi": merged_roi,
        "penalty_candidates": penalty_candidates,
        "enforce_penalties": bool(penalty_candidates),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(policy, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
    )
    return policy


def load_promoted_learning_policy(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {
            "schema_version": "nexus_promoted_learning_policy.v1",
            "source_experiences": [],
            "promoted_capabilities": [],
            "penalized_capabilities": [],
            "escalation_recommendations": [],
            "capability_roi": {},
            "penalty_candidates": [],
            "enforce_penalties": False,
        }
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _lifecycle_from_receipt(receipt: dict[str, Any]) -> CapabilityLifecycle:
    name = str(receipt.get("name") or receipt.get("capability") or "unknown")
    meta = CAPABILITY_TAXONOMY.get(name, {"category": "unknown", "phases": ("C",)})
    refs = tuple(str(ref) for ref in receipt.get("evidence_refs", []) or [] if str(ref).strip())
    return CapabilityLifecycle(
        capability=name,
        category=str(meta["category"]),
        phase=str(tuple(meta["phases"])[0]),
        selected=bool(receipt.get("selected", False)),
        invoked=bool(receipt.get("invoked", False)),
        evidence=bool(receipt.get("evidence_present", False) or refs),
        outcome=bool(receipt.get("outcome_contributed", False)),
        gate_passed=bool(receipt.get("gate_passed", False)),
        evidence_refs=refs,
        failure_reason=str(receipt.get("failure_reason") or ""),
    )


def _gate_chain(usage: dict[str, Any]) -> dict[str, str]:
    caps = usage.get("capabilities", {}) if isinstance(usage.get("capabilities"), dict) else {}

    def status(key: str) -> str:
        value = caps.get(f"{key}_gate_passed")
        if value is True:
            return "pass"
        if value is False and (caps.get(f"{key}_refs") or caps.get(f"{key}_invoked")):
            return "fail"
        return "not_run"

    return {
        "mempalace": status("mempalace"),
        "belief": status("belief"),
        "artifact": status("artifact"),
        "claim": "pass"
        if caps.get("claim_verified") or caps.get("claim_gate_passed")
        else "not_run",
        "delivery": status("delivery"),
    }


def _first_missing_phase(observed: list[str]) -> str:
    for phase in PHASE_CHAIN:
        if phase not in observed:
            return phase
    return ""


def _aggregate_capability_roi(experiences: list[LearningExperience]) -> dict[str, dict[str, int]]:
    roi: dict[str, dict[str, int]] = {}
    for exp in experiences:
        for item in exp.capability_lifecycle:
            entry = roi.setdefault(
                item.capability,
                {
                    "selected": 0,
                    "invoked": 0,
                    "evidence": 0,
                    "outcome": 0,
                    "gate_passed": 0,
                },
            )
            entry["selected"] += int(bool(item.selected))
            entry["invoked"] += int(bool(item.invoked))
            entry["evidence"] += int(bool(item.evidence))
            entry["outcome"] += int(bool(item.outcome))
            entry["gate_passed"] += int(bool(item.gate_passed))
    return roi


def _merge_capability_roi(
    prior: dict[str, Any],
    current: dict[str, dict[str, int]],
) -> dict[str, dict[str, int]]:
    merged: dict[str, dict[str, int]] = {}
    names = set(prior) | set(current)
    for name in names:
        base = prior.get(name, {}) if isinstance(prior.get(name, {}), dict) else {}
        now = current.get(name, {})
        merged[name] = {
            "selected": int(base.get("selected", 0) or 0) + int(now.get("selected", 0) or 0),
            "invoked": int(base.get("invoked", 0) or 0) + int(now.get("invoked", 0) or 0),
            "evidence": int(base.get("evidence", 0) or 0) + int(now.get("evidence", 0) or 0),
            "outcome": int(base.get("outcome", 0) or 0) + int(now.get("outcome", 0) or 0),
            "gate_passed": int(base.get("gate_passed", 0) or 0)
            + int(now.get("gate_passed", 0) or 0),
        }
    return merged


def _derive_penalty_candidates(roi: dict[str, dict[str, int]]) -> list[str]:
    penalty: list[str] = []
    for name, counts in roi.items():
        selected = int(counts.get("selected", 0) or 0)
        invoked = int(counts.get("invoked", 0) or 0)
        outcome = int(counts.get("outcome", 0) or 0)
        if name not in HIGH_COST_CAPABILITIES or selected < 2:
            continue
        if invoked * 2 < selected or outcome * 2 < selected:
            penalty.append(name)
    return sorted(set(penalty))
