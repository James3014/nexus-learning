"""Durable identity and reconciliation evidence for one experiment run.

This module records observations only.  It does not start, stop, resume, retry,
schedule, or reconcile an external effect; that remains the responsibility of
the producer/effect owner.

Frozen research-governance contract (BOUNDARY v2): bug fixes only; no consumer outside nexus-learning.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from typing import Any, Literal, Mapping, overload

from .experiment_integrity import (
    EFFECT_FAILED,
    EFFECT_NOT_STARTED,
    EFFECT_OUTCOME_UNKNOWN,
    EFFECT_SUCCEEDED,
    PHYSICAL_LOCAL_MODEL,
    REMOTE_PROVIDER_OBSERVED,
    validate_evidence_origin_provenance,
)

LEARNING_BOUNDARY_STATUS = "FROZEN_RESEARCH_GOVERNANCE"  # see docs/architecture/BOUNDARY.md

EXPERIMENT_RUN_IDENTITY_SCHEMA = "nexus.learning_experiment_run_identity.v1"
EXPERIMENT_RUN_IDENTITY_CLAIM_CEILING = (
    "LEARNING_EXPERIMENT_RUN_IDENTITY_RECONCILIATION_EVIDENCE_ONLY"
)

NOT_STARTED = "NOT_STARTED"
RUNNING = "RUNNING"
COMPLETE = "COMPLETE"
FAILED = "FAILED"
OUTCOME_UNKNOWN = "OUTCOME_UNKNOWN"
_RUN_STATES = frozenset({NOT_STARTED, RUNNING, COMPLETE, FAILED, OUTCOME_UNKNOWN})

OBSERVE_SAME_RUN = "OBSERVE_SAME_RUN"
RECONCILE_WITH_EFFECT_OWNER = "RECONCILE_WITH_EFFECT_OWNER"
CONSUME_BOUND_RESULT = "CONSUME_BOUND_RESULT"
RECORD_TERMINAL_FAILURE = "RECORD_TERMINAL_FAILURE"
NO_EFFECT_OBSERVED = "NO_EFFECT_OBSERVED"
GENERATION_IDENTITY_MISMATCH = "GENERATION_IDENTITY_MISMATCH"
DIFFERENT_LOGICAL_GENERATION = "DIFFERENT_LOGICAL_GENERATION"
EFFECT_IDENTITY_CONFLICT = "EFFECT_IDENTITY_CONFLICT"
TERMINAL_STATE_CONFLICT = "TERMINAL_STATE_CONFLICT"


def _canonical_hash(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return "sha256:" + hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _is_hash(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 71
        and value.startswith("sha256:")
        and all(char in "0123456789abcdef" for char in value[7:])
    )


@overload
def _text(value: Any, field: str, *, optional: Literal[False] = False) -> str: ...


@overload
def _text(value: Any, field: str, *, optional: Literal[True]) -> str | None: ...


def _text(value: Any, field: str, *, optional: bool = False) -> str | None:
    if value is None and optional:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"RUN_IDENTITY_{field.upper()}_INVALID")
    return value.strip()


@overload
def _timestamp(value: Any, field: str, *, optional: Literal[False] = False) -> str: ...


@overload
def _timestamp(value: Any, field: str, *, optional: Literal[True]) -> str | None: ...


def _timestamp(value: Any, field: str, *, optional: bool = False) -> str | None:
    if value is None and optional:
        return None
    text = _text(value, field)
    normalized = text[:-1] + "+00:00" if text.endswith("Z") else text
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError as exc:
        raise ValueError(f"RUN_IDENTITY_{field.upper()}_INVALID") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"RUN_IDENTITY_{field.upper()}_TIMEZONE_REQUIRED")
    return text


def _execution_identity(value: Any, field: str) -> dict[str, str | None]:
    if not isinstance(value, Mapping):
        raise ValueError(f"RUN_IDENTITY_{field.upper()}_INVALID")
    keys = {"provider", "model", "runtime", "revision"}
    if set(value) != keys:
        raise ValueError(f"RUN_IDENTITY_{field.upper()}_INVALID")
    return {
        key: _text(value[key], f"{field}_{key}", optional=True)
        for key in ("provider", "model", "runtime", "revision")
    }


def _identity_projection(run: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "experiment_id": run["experiment_id"],
        "experiment_generation": run["experiment_generation"],
        "subject_repo": run["subject_repo"],
        "subject_commit": run["subject_commit"],
        "subject_tree": run["subject_tree"],
        "harness_identity": run["harness_identity"],
        "harness_hash": run["harness_hash"],
        "cohort_identity": run["cohort_identity"],
        "cohort_hash": run["cohort_hash"],
        "frozen_policy_hash": run["frozen_policy_hash"],
        "requested_execution_identity": run["requested_execution_identity"],
        "configured_execution_identity": run["configured_execution_identity"],
    }


def _expected_requirement(state: str) -> str:
    return {
        NOT_STARTED: NO_EFFECT_OBSERVED,
        RUNNING: OBSERVE_SAME_RUN,
        OUTCOME_UNKNOWN: RECONCILE_WITH_EFFECT_OWNER,
        COMPLETE: CONSUME_BOUND_RESULT,
        FAILED: RECORD_TERMINAL_FAILURE,
    }[state]


def build_experiment_run_identity(
    *,
    experiment_id: str,
    experiment_generation: int,
    subject_repo: str,
    subject_commit: str,
    subject_tree: str,
    harness_identity: str,
    harness_hash: str,
    cohort_identity: str,
    cohort_hash: str,
    frozen_policy_hash: str,
    observed_state: str,
    producer_kind: str,
    last_observed_at: str,
    requested_execution_identity: Mapping[str, Any] | None = None,
    configured_execution_identity: Mapping[str, Any] | None = None,
    observed_execution_identity: Mapping[str, Any] | None = None,
    host_identity: str | None = None,
    producer_run_id: str | None = None,
    operation_id: str | None = None,
    effect_identity: str | None = None,
    started_at: str | None = None,
    result_artifact: Mapping[str, Any] | None = None,
    producer_observations: Mapping[str, Any] | None = None,
    evidence_origin_provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a hash-bound run observation without inferring absent values.

    Omitted optional identities are represented as explicit ``None`` fields.
    The generation binding hashes exact source/harness/cohort/policy identity;
    changing any of them under the same numeric generation is an identity
    conflict, never evidence that an old effect disappeared.
    """
    if isinstance(experiment_generation, bool) or not isinstance(experiment_generation, int):
        raise ValueError("RUN_IDENTITY_GENERATION_INVALID")
    if experiment_generation < 1:
        raise ValueError("RUN_IDENTITY_GENERATION_INVALID")
    state = _text(observed_state, "observed_state")
    if state not in _RUN_STATES:
        raise ValueError("RUN_IDENTITY_STATE_INVALID")
    harness_digest = _text(harness_hash, "harness_hash")
    cohort_digest = _text(cohort_hash, "cohort_hash")
    policy_digest = _text(frozen_policy_hash, "frozen_policy_hash")
    if not all(_is_hash(value) for value in (harness_digest, cohort_digest, policy_digest)):
        raise ValueError("RUN_IDENTITY_REQUIRED_HASH_INVALID")

    requested = _execution_identity(
        requested_execution_identity if requested_execution_identity is not None else {"provider": None, "model": None, "runtime": None, "revision": None},
        "requested_execution_identity",
    )
    configured = _execution_identity(
        configured_execution_identity if configured_execution_identity is not None else {"provider": None, "model": None, "runtime": None, "revision": None},
        "configured_execution_identity",
    )
    observed = _execution_identity(
        observed_execution_identity if observed_execution_identity is not None else {"provider": None, "model": None, "runtime": None, "revision": None},
        "observed_execution_identity",
    )
    started = _timestamp(started_at, "started_at", optional=True)
    observed_at = _timestamp(last_observed_at, "last_observed_at")
    if state in {RUNNING, COMPLETE, FAILED} and started is None:
        raise ValueError("RUN_IDENTITY_STARTED_AT_REQUIRED")
    if started is not None:
        start_dt = datetime.fromisoformat(started[:-1] + "+00:00" if started.endswith("Z") else started)
        observed_dt = datetime.fromisoformat(observed_at[:-1] + "+00:00" if observed_at.endswith("Z") else observed_at)
        if start_dt > observed_dt:
            raise ValueError("RUN_IDENTITY_OBSERVATION_PRECEDES_START")

    artifact: dict[str, str] | None = None
    if result_artifact is not None:
        if not isinstance(result_artifact, Mapping) or set(result_artifact) != {"artifact_id", "sha256"}:
            raise ValueError("RUN_IDENTITY_RESULT_ARTIFACT_INVALID")
        artifact_id = _text(result_artifact.get("artifact_id"), "result_artifact_id")
        artifact_hash = _text(result_artifact.get("sha256"), "result_artifact_hash")
        if not _is_hash(artifact_hash):
            raise ValueError("RUN_IDENTITY_RESULT_ARTIFACT_HASH_INVALID")
        artifact = {"artifact_id": artifact_id, "sha256": artifact_hash}
    if state in {COMPLETE, FAILED} and artifact is None:
        raise ValueError("RUN_IDENTITY_TERMINAL_REQUIRES_RESULT_ARTIFACT")
    if state in {RUNNING, OUTCOME_UNKNOWN, NOT_STARTED} and artifact is not None:
        raise ValueError("RUN_IDENTITY_NONTERMINAL_RESULT_ARTIFACT_FORBIDDEN")

    observations = dict(producer_observations or {})
    # Ensure optional local process/workspace observations cannot be used as
    # portable logical or effect identity.
    if any(key in observations for key in ("experiment_id", "effect_identity", "operation_id")):
        raise ValueError("RUN_IDENTITY_LOCAL_OBSERVATION_CANNOT_OVERRIDE_IDENTITY")
    origin = None
    if evidence_origin_provenance is not None:
        validate_evidence_origin_provenance(evidence_origin_provenance)
        origin = dict(evidence_origin_provenance)

    payload: dict[str, Any] = {
        "schema": EXPERIMENT_RUN_IDENTITY_SCHEMA,
        "experiment_id": _text(experiment_id, "experiment_id"),
        "experiment_generation": experiment_generation,
        "subject_repo": _text(subject_repo, "subject_repo"),
        "subject_commit": _text(subject_commit, "subject_commit"),
        "subject_tree": _text(subject_tree, "subject_tree"),
        "harness_identity": _text(harness_identity, "harness_identity"),
        "harness_hash": harness_digest,
        "cohort_identity": _text(cohort_identity, "cohort_identity"),
        "cohort_hash": cohort_digest,
        "frozen_policy_hash": policy_digest,
        "generation_binding_hash": "",
        "requested_execution_identity": requested,
        "configured_execution_identity": configured,
        "observed_execution_identity": observed,
        "host_identity": _text(host_identity, "host_identity", optional=True),
        "producer_kind": _text(producer_kind, "producer_kind"),
        "producer_run_id": _text(producer_run_id, "producer_run_id", optional=True),
        "operation_id": _text(operation_id, "operation_id", optional=True),
        "effect_identity": _text(effect_identity, "effect_identity", optional=True),
        "started_at": started,
        "observed_state": state,
        "last_observed_at": observed_at,
        "result_artifact": artifact,
        "evidence_origin_provenance": origin,
        "producer_observations": observations,
        "reconciliation_requirement": _expected_requirement(state),
        "claim_ceiling": EXPERIMENT_RUN_IDENTITY_CLAIM_CEILING,
    }
    payload["generation_binding_hash"] = _canonical_hash(_identity_projection(payload))
    payload["binding_hash"] = _canonical_hash(payload)
    validate_experiment_run_identity(payload)
    return payload


def validate_experiment_run_identity(run: Any) -> dict[str, Any]:
    """Validate the run binding and all state/result invariants fail-closed."""
    if not isinstance(run, Mapping):
        raise ValueError("RUN_IDENTITY_NOT_A_MAPPING")
    expected_keys = {
        "schema", "experiment_id", "experiment_generation", "subject_repo", "subject_commit",
        "subject_tree", "harness_identity", "harness_hash", "cohort_identity", "cohort_hash",
        "frozen_policy_hash", "generation_binding_hash", "requested_execution_identity",
        "configured_execution_identity", "observed_execution_identity", "host_identity",
        "producer_kind", "producer_run_id", "operation_id", "effect_identity", "started_at",
        "observed_state", "last_observed_at", "result_artifact", "producer_observations",
        "evidence_origin_provenance", "reconciliation_requirement", "claim_ceiling", "binding_hash",
    }
    if set(run) != expected_keys or run.get("schema") != EXPERIMENT_RUN_IDENTITY_SCHEMA:
        raise ValueError("RUN_IDENTITY_SCHEMA_INVALID")
    generation = run.get("experiment_generation")
    if isinstance(generation, bool) or not isinstance(generation, int) or generation < 1:
        raise ValueError("RUN_IDENTITY_GENERATION_INVALID")
    for field in ("experiment_id", "subject_repo", "subject_commit", "subject_tree", "harness_identity", "cohort_identity", "producer_kind"):
        _text(run.get(field), field)
    for field in ("harness_hash", "cohort_hash", "frozen_policy_hash"):
        if not _is_hash(run.get(field)):
            raise ValueError(f"RUN_IDENTITY_{field.upper()}_INVALID")
    for field in ("requested_execution_identity", "configured_execution_identity", "observed_execution_identity"):
        _execution_identity(run.get(field), field)
    for field in ("host_identity", "producer_run_id", "operation_id", "effect_identity"):
        _text(run.get(field), field, optional=True)
    state = _text(run.get("observed_state"), "observed_state")
    if state not in _RUN_STATES:
        raise ValueError("RUN_IDENTITY_STATE_INVALID")
    started = _timestamp(run.get("started_at"), "started_at", optional=True)
    observed_at = _timestamp(run.get("last_observed_at"), "last_observed_at")
    if state in {RUNNING, COMPLETE, FAILED} and started is None:
        raise ValueError("RUN_IDENTITY_STARTED_AT_REQUIRED")
    if started:
        def to_dt(value: str) -> datetime:
            return datetime.fromisoformat(value[:-1] + "+00:00" if value.endswith("Z") else value)

        if to_dt(started) > to_dt(observed_at):
            raise ValueError("RUN_IDENTITY_OBSERVATION_PRECEDES_START")
    artifact = run.get("result_artifact")
    if artifact is not None:
        if not isinstance(artifact, Mapping) or set(artifact) != {"artifact_id", "sha256"}:
            raise ValueError("RUN_IDENTITY_RESULT_ARTIFACT_INVALID")
        _text(artifact.get("artifact_id"), "result_artifact_id")
        if not _is_hash(artifact.get("sha256")):
            raise ValueError("RUN_IDENTITY_RESULT_ARTIFACT_HASH_INVALID")
    if state in {COMPLETE, FAILED} and artifact is None:
        raise ValueError("RUN_IDENTITY_TERMINAL_REQUIRES_RESULT_ARTIFACT")
    if state in {RUNNING, OUTCOME_UNKNOWN, NOT_STARTED} and artifact is not None:
        raise ValueError("RUN_IDENTITY_NONTERMINAL_RESULT_ARTIFACT_FORBIDDEN")
    if not isinstance(run.get("producer_observations"), Mapping):
        raise ValueError("RUN_IDENTITY_PRODUCER_OBSERVATIONS_INVALID")
    if run.get("evidence_origin_provenance") is not None:
        validate_evidence_origin_provenance(run["evidence_origin_provenance"])
        provenance = run["evidence_origin_provenance"]
        effect = provenance["external_effect"]
        for identity_field in ("effect_identity", "operation_id"):
            run_identity = run.get(identity_field)
            provenance_identity = effect.get(identity_field)
            if effect.get("started") is True and (
                not run_identity or not provenance_identity
            ):
                raise ValueError("RUN_IDENTITY_EVIDENCE_ORIGIN_EFFECT_IDENTITY_MISSING")
            if run_identity is not None and provenance_identity is not None and run_identity != provenance_identity:
                raise ValueError(f"RUN_IDENTITY_EVIDENCE_ORIGIN_{identity_field.upper()}_MISMATCH")
        identity_pairs = (
            ("requested_execution_identity", "requested_identity"),
            ("configured_execution_identity", "configured_identity"),
            ("observed_execution_identity", "observed_identity"),
        )
        for run_field, provenance_field in identity_pairs:
            run_identity = run[run_field]
            provenance_identity = provenance[provenance_field]
            for identity_field in ("provider", "model", "revision"):
                run_value = run_identity.get(identity_field)
                provenance_value = provenance_identity.get(identity_field)
                if run_value is not None and provenance_value is not None and run_value != provenance_value:
                    raise ValueError(
                        f"RUN_IDENTITY_EVIDENCE_ORIGIN_{provenance_field.upper()}_"
                        f"{identity_field.upper()}_MISMATCH"
                    )
        if provenance.get("origin_class") in {PHYSICAL_LOCAL_MODEL, REMOTE_PROVIDER_OBSERVED}:
            expected_outcomes = {
                NOT_STARTED: {EFFECT_NOT_STARTED},
                RUNNING: {EFFECT_OUTCOME_UNKNOWN},
                OUTCOME_UNKNOWN: {EFFECT_OUTCOME_UNKNOWN},
                COMPLETE: {EFFECT_SUCCEEDED},
                FAILED: {EFFECT_FAILED},
            }
            if effect.get("outcome") not in expected_outcomes[state]:
                raise ValueError("RUN_IDENTITY_EVIDENCE_ORIGIN_STATE_MISMATCH")
    if any(key in run["producer_observations"] for key in ("experiment_id", "effect_identity", "operation_id")):
        raise ValueError("RUN_IDENTITY_LOCAL_OBSERVATION_CANNOT_OVERRIDE_IDENTITY")
    if run.get("reconciliation_requirement") != _expected_requirement(state):
        raise ValueError("RUN_IDENTITY_RECONCILIATION_REQUIREMENT_MISMATCH")
    if run.get("claim_ceiling") != EXPERIMENT_RUN_IDENTITY_CLAIM_CEILING:
        raise ValueError("RUN_IDENTITY_CLAIM_CEILING_INVALID")
    if run.get("generation_binding_hash") != _canonical_hash(_identity_projection(run)):
        raise ValueError("RUN_IDENTITY_GENERATION_BINDING_MISMATCH")
    payload = dict(run)
    supplied_hash = payload.pop("binding_hash", None)
    if supplied_hash != _canonical_hash(payload):
        raise ValueError("RUN_IDENTITY_BINDING_HASH_MISMATCH")
    return dict(run)


def classify_run_observation(
    previous: Mapping[str, Any], current: Mapping[str, Any]
) -> dict[str, Any]:
    """Compare observations without turning missing local state into retry permission."""
    old = validate_experiment_run_identity(previous)
    new = validate_experiment_run_identity(current)
    if (old["experiment_id"], old["experiment_generation"]) != (
        new["experiment_id"], new["experiment_generation"]
    ):
        return {
            "classification": DIFFERENT_LOGICAL_GENERATION,
            "requirement": None,
            "retry_authorized": False,
            "new_effect_authorized": False,
            "claim_ceiling": EXPERIMENT_RUN_IDENTITY_CLAIM_CEILING,
        }
    if old["generation_binding_hash"] != new["generation_binding_hash"]:
        return {
            "classification": GENERATION_IDENTITY_MISMATCH,
            "requirement": "NEW_GENERATION_REQUIRED",
            "retry_authorized": False,
            "new_effect_authorized": False,
            "claim_ceiling": EXPERIMENT_RUN_IDENTITY_CLAIM_CEILING,
        }
    old_effect = old.get("effect_identity")
    new_effect = new.get("effect_identity")
    if old_effect and new_effect and old_effect != new_effect:
        return {
            "classification": EFFECT_IDENTITY_CONFLICT,
            "requirement": "RECONCILE_WITH_EFFECT_OWNER",
            "retry_authorized": False,
            "new_effect_authorized": False,
            "claim_ceiling": EXPERIMENT_RUN_IDENTITY_CLAIM_CEILING,
        }
    old_state = old["observed_state"]
    state = new["observed_state"]
    terminal_states = {COMPLETE, FAILED}
    if old_state in terminal_states:
        terminal_identity = (
            old_state == state
            and old.get("result_artifact") == new.get("result_artifact")
            and old.get("producer_run_id") == new.get("producer_run_id")
            and old.get("effect_identity") == new.get("effect_identity")
            and old.get("operation_id") == new.get("operation_id")
            and old.get("observed_execution_identity") == new.get("observed_execution_identity")
            and old.get("evidence_origin_provenance") == new.get("evidence_origin_provenance")
        )
        if not terminal_identity:
            return {
                "classification": TERMINAL_STATE_CONFLICT,
                "requirement": "RECONCILE_WITH_EFFECT_OWNER",
                "retry_authorized": False,
                "new_effect_authorized": False,
                "claim_ceiling": EXPERIMENT_RUN_IDENTITY_CLAIM_CEILING,
            }
    if old_state in {RUNNING, OUTCOME_UNKNOWN} and state == NOT_STARTED:
        # A missing process/session observation cannot erase a prior unresolved
        # external effect. Keep the strongest previously observed requirement.
        state = old_state
    return {
        "classification": state,
        "requirement": _expected_requirement(state),
        "retry_authorized": False,
        "new_effect_authorized": False,
        "claim_ceiling": EXPERIMENT_RUN_IDENTITY_CLAIM_CEILING,
    }
