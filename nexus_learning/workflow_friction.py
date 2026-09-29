"""Evidence-bounded workflow-friction recommendations (issue #34).

This module turns repeated, exact-identity workflow friction into an advisory
Learning recommendation. It never adopts a recommendation or mutates Runtime,
Planner, Workforce, merge, release, or deployment state.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any

WORKFLOW_FRICTION_SCHEMA = "nexus.learning.workflow_friction.v1"
WORKFLOW_RECOMMENDATION_SCHEMA = "nexus.learning.workflow_recommendation.v1"
WORKFLOW_VALIDATION_SCHEMA = "nexus.learning.workflow_recommendation_validation.v1"
CLAIM_CEILING = "WORKFLOW_RECOMMENDATION_ONLY"
_ALLOWED_OUTCOMES = {"FRICTION", "NO_FRICTION", "NEGATIVE_EXPERIMENT"}
_FORBIDDEN_EVIDENCE_WORDS = ("chat", "transcript", "chain_of_thought", "session_memory")


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _hash(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be non-empty")
    return value.strip()


def build_friction_observation(
    *,
    task_id: str,
    operation_id: str,
    attempt_id: str,
    source_revision: str,
    runtime_identity: str,
    friction_kind: str,
    evidence_refs: Sequence[str],
    evidence_types: Sequence[str],
    outcome: str = "FRICTION",
) -> dict[str, Any]:
    refs = sorted({_text(v, "evidence_ref") for v in evidence_refs})
    types = sorted({_text(v, "evidence_type") for v in evidence_types})
    if not refs or not types:
        raise ValueError("workflow friction requires durable evidence refs and types")
    lowered = " ".join(refs + types).lower()
    if any(word in lowered for word in _FORBIDDEN_EVIDENCE_WORDS):
        raise ValueError("conversation memory/transcript is not workflow evidence")
    normalized_outcome = _text(outcome, "outcome").upper()
    if normalized_outcome not in _ALLOWED_OUTCOMES:
        raise ValueError("unsupported workflow friction outcome")
    body: dict[str, Any] = {
        "schema": WORKFLOW_FRICTION_SCHEMA,
        "task_id": _text(task_id, "task_id"),
        "operation_id": _text(operation_id, "operation_id"),
        "attempt_id": _text(attempt_id, "attempt_id"),
        "source_revision": _text(source_revision, "source_revision"),
        "runtime_identity": _text(runtime_identity, "runtime_identity"),
        "friction_kind": _text(friction_kind, "friction_kind"),
        "evidence_refs": refs,
        "evidence_types": types,
        "outcome": normalized_outcome,
        "claim_ceiling": CLAIM_CEILING,
    }
    body["observation_hash"] = _hash(body)
    return body


def validate_friction_observation(value: Mapping[str, Any]) -> None:
    if value.get("schema") != WORKFLOW_FRICTION_SCHEMA:
        raise ValueError("workflow friction schema invalid")
    supplied = value.get("observation_hash")
    material = dict(value)
    material.pop("observation_hash", None)
    if supplied != _hash(material):
        raise ValueError("workflow friction observation tampered")
    if value.get("claim_ceiling") != CLAIM_CEILING:
        raise ValueError("workflow friction claim ceiling invalid")
    for field in ("task_id", "operation_id", "attempt_id", "source_revision",
                  "runtime_identity", "friction_kind"):
        _text(value.get(field), field)
    refs = value.get("evidence_refs")
    types = value.get("evidence_types")
    if not isinstance(refs, list) or not refs or not isinstance(types, list) or not types:
        raise ValueError("workflow friction durable evidence missing")
    lowered = " ".join(map(str, refs + types)).lower()
    if any(word in lowered for word in _FORBIDDEN_EVIDENCE_WORDS):
        raise ValueError("conversation memory/transcript is not workflow evidence")
    if value.get("outcome") not in _ALLOWED_OUTCOMES:
        raise ValueError("workflow friction outcome invalid")


def build_workflow_recommendation(
    observations: Sequence[Mapping[str, Any]],
    *,
    recommendation: str,
    required_evidence_types: Sequence[str],
    minimum_repetitions: int = 2,
) -> dict[str, Any]:
    if minimum_repetitions < 2:
        raise ValueError("minimum_repetitions must be >= 2")
    rows = [dict(row) for row in observations]
    if not rows:
        raise ValueError("workflow recommendation requires observations")
    for row in rows:
        validate_friction_observation(row)
    friction_kinds = {row["friction_kind"] for row in rows}
    if len(friction_kinds) != 1:
        raise ValueError("workflow recommendation must describe one friction kind")
    positive = [row for row in rows if row["outcome"] == "FRICTION"]
    negative = [row for row in rows if row["outcome"] == "NEGATIVE_EXPERIMENT"]
    required = sorted({_text(v, "required_evidence_type") for v in required_evidence_types})
    present = sorted({kind for row in rows for kind in row["evidence_types"]})
    missing = sorted(set(required) - set(present))
    uncertainty: list[str] = []
    if len(positive) < minimum_repetitions:
        uncertainty.append("insufficient_repetition")
    if missing:
        uncertainty.append("evidence_coverage_incomplete")
    body: dict[str, Any] = {
        "schema": WORKFLOW_RECOMMENDATION_SCHEMA,
        "friction_kind": next(iter(friction_kinds)),
        "recommendation": _text(recommendation, "recommendation"),
        "source_observation_hashes": sorted(row["observation_hash"] for row in rows),
        "source_identities": sorted(
            [
                {
                    "task_id": row["task_id"],
                    "operation_id": row["operation_id"],
                    "attempt_id": row["attempt_id"],
                    "source_revision": row["source_revision"],
                    "runtime_identity": row["runtime_identity"],
                }
                for row in rows
            ],
            key=_canonical,
        ),
        "coverage": {"required": required, "present": present, "missing": missing},
        "negative_evidence": sorted(row["observation_hash"] for row in negative),
        "repetition_count": len(positive),
        "minimum_repetitions": minimum_repetitions,
        "uncertainty": uncertainty,
        "status": "PROPOSED",
        "direct_mutation_allowed": False,
        "adoption_requires_external_authority": True,
        "claim_ceiling": CLAIM_CEILING,
    }
    body["recommendation_hash"] = _hash(body)
    body["recommendation_id"] = "wrec:" + body["recommendation_hash"][:24]
    return body


def validate_workflow_recommendation(
    recommendation: Mapping[str, Any],
    *,
    validator_identity: str,
    current_source_revision: str,
    current_runtime_identity: str,
) -> dict[str, Any]:
    if recommendation.get("schema") != WORKFLOW_RECOMMENDATION_SCHEMA:
        raise ValueError("workflow recommendation schema invalid")
    material = dict(recommendation)
    supplied_hash = material.pop("recommendation_hash", None)
    supplied_id = material.pop("recommendation_id", None)
    expected_hash = _hash(material)
    if supplied_hash != expected_hash or supplied_id != "wrec:" + expected_hash[:24]:
        raise ValueError("workflow recommendation tampered")
    if recommendation.get("direct_mutation_allowed") is not False:
        raise ValueError("workflow recommendation cannot authorize mutation")
    identities = recommendation.get("source_identities")
    if not isinstance(identities, list) or not identities:
        raise ValueError("workflow recommendation source identities missing")
    stale = any(
        row.get("source_revision") != current_source_revision
        or row.get("runtime_identity") != current_runtime_identity
        for row in identities
    )
    blockers = list(recommendation.get("uncertainty") or [])
    if stale:
        blockers.append("source_or_runtime_identity_stale")
    disposition = "VALIDATED_FOR_CONSIDERATION" if not blockers else "INSUFFICIENT_EVIDENCE"
    result: dict[str, Any] = {
        "schema": WORKFLOW_VALIDATION_SCHEMA,
        "recommendation_id": recommendation["recommendation_id"],
        "recommendation_hash": recommendation["recommendation_hash"],
        "validator_identity": _text(validator_identity, "validator_identity"),
        "current_source_revision": _text(current_source_revision, "current_source_revision"),
        "current_runtime_identity": _text(current_runtime_identity, "current_runtime_identity"),
        "disposition": disposition,
        "blockers": sorted(set(blockers)),
        "adoption_authorized": False,
        "claim_ceiling": CLAIM_CEILING,
    }
    result["validation_hash"] = _hash(result)
    return result


__all__ = [
    "WORKFLOW_FRICTION_SCHEMA",
    "WORKFLOW_RECOMMENDATION_SCHEMA",
    "WORKFLOW_VALIDATION_SCHEMA",
    "CLAIM_CEILING",
    "build_friction_observation",
    "validate_friction_observation",
    "build_workflow_recommendation",
    "validate_workflow_recommendation",
]
