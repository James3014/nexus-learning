"""Deterministic cohort/comparator preflight and staged-gate evidence.

The contracts here record and validate methodology evidence.  They do not run
benchmarks, call providers, allocate resources, or authorize holdout execution.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from math import isfinite
from typing import Any, Literal, Mapping, Sequence, overload

from .experiment_integrity import (
    CALIBRATED,
    INDEPENDENCE_UNIT_BASE,
    INDEPENDENCE_UNIT_ROW,
    TERMINAL_NEGATIVE,
    TERMINAL_PASS,
    TERMINAL_STOP,
    validate_experiment_integrity,
)
from .necessary_condition_stop import (
    CONTINUE,
    NEGATIVE_STOP,
    validate_necessary_condition_stop_evidence,
)

COHORT_PREFLIGHT_SCHEMA = "nexus.learning_experiment_cohort_preflight.v1"
STAGED_GATE_SCHEMA = "nexus.learning_experiment_staged_gate.v1"
PREFLIGHT_CLAIM_CEILING = "LEARNING_EXPERIMENT_PREFLIGHT_AND_STAGED_GATE_EVIDENCE_ONLY"

PREFLIGHT_PASS = "PASS"
PREFLIGHT_FAIL = "FAIL"
PREFLIGHT_DEFER = "DEFER"

LEAKAGE_PASS = "PASS"
LEAKAGE_FAIL = "FAIL"
LEAKAGE_NOT_EVALUATED = "NOT_EVALUATED"
LEAKAGE_UNSUPPORTED = "UNSUPPORTED"
_LEAKAGE_STATES = frozenset({LEAKAGE_PASS, LEAKAGE_FAIL, LEAKAGE_NOT_EVALUATED, LEAKAGE_UNSUPPORTED})

COMPARATOR_MATCHED = "MATCHED"
COMPARATOR_MISMATCHED = "MISMATCHED"
COMPARATOR_UNSUPPORTED = "UNSUPPORTED"
COMPARATOR_NOT_OBSERVED = "NOT_OBSERVED"
COMPARATOR_NOT_APPLICABLE = "NOT_APPLICABLE"
_COMPARATOR_STATES = frozenset(
    {COMPARATOR_MATCHED, COMPARATOR_MISMATCHED, COMPARATOR_UNSUPPORTED,
     COMPARATOR_NOT_OBSERVED, COMPARATOR_NOT_APPLICABLE}
)
COMPARATOR_DIMENSIONS = (
    "sampling_fields",
    "reasoning_mode",
    "context_and_prefill_policy",
    "cache_semantics",
    "concurrency_and_admission_policy",
    "material_runtime_fields",
)

ADMITTED = "ADMITTED"
HELD = "HELD"
REFUSED = "REFUSED"
NOT_OBSERVED = "NOT_OBSERVED"
_ADMISSION_STATES = frozenset({ADMITTED, HELD, REFUSED, NOT_OBSERVED})
RECOVERY_YES = "YES"
RECOVERY_NO = "NO"
RECOVERY_NOT_OBSERVED = "NOT_OBSERVED"
_RECOVERY_STATES = frozenset({RECOVERY_YES, RECOVERY_NO, RECOVERY_NOT_OBSERVED})

STAGE_PREFLIGHT = "PREFLIGHT"
STAGE_SMOKE = "SMOKE"
STAGE_CALIBRATION = "CALIBRATION"
STAGE_HARD_STOP = "HARD_STOP"
STAGE_HOLDOUT_ELIGIBLE = "HOLDOUT_ELIGIBLE"
STAGE_HOLDOUT_COMPLETE = "HOLDOUT_COMPLETE"
STAGE_STOP = "STOP"
STAGE_DEFER = "DEFER"
_STAGES = frozenset({STAGE_PREFLIGHT, STAGE_SMOKE, STAGE_CALIBRATION,
                     STAGE_HARD_STOP, STAGE_HOLDOUT_ELIGIBLE,
                     STAGE_HOLDOUT_COMPLETE, STAGE_STOP, STAGE_DEFER})

GATE_GTE = "GTE"
GATE_LTE = "LTE"
GATE_EQ = "EQ"
_GATE_COMPARATORS = frozenset({GATE_GTE, GATE_LTE, GATE_EQ})
_GATE_KINDS = frozenset({"CORRECTNESS", "SAFETY", "RESOURCE", "QUALITY", "PERFORMANCE"})


def _hash(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return "sha256:" + hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _is_hash(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 71 and value.startswith("sha256:") and all(
        c in "0123456789abcdef" for c in value[7:]
    )


def _is_sha256_digest(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        c in "0123456789abcdef" for c in value
    )


@overload
def _text(value: Any, field: str, *, optional: Literal[False] = False) -> str: ...


@overload
def _text(value: Any, field: str, *, optional: Literal[True]) -> str | None: ...


def _text(value: Any, field: str, *, optional: bool = False) -> str | None:
    if value is None and optional:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"PREFLIGHT_{field.upper()}_INVALID")
    return value.strip()


def _number(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(value):
        raise ValueError(f"PREFLIGHT_{field.upper()}_INVALID")
    return float(value)


def _hash_payload(payload: Mapping[str, Any], hash_field: str = "binding_hash") -> str:
    unsigned = dict(payload)
    unsigned.pop(hash_field, None)
    return _hash(unsigned)


def build_comparator_preflight(
    *,
    comparator_identity: str,
    dimension_observations: Mapping[str, Mapping[str, Any]] | None = None,
    sampling_request: Mapping[str, Any] | None = None,
    compatibility_amendment: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Bind comparability observations for material request/runtime settings.

    All six comparator dimensions remain explicit.  A rejected sampling field
    blocks formal comparability unless an exact, hash-bound amendment was
    frozen before the formal run and the amended fields are accepted.
    """
    supplied = dimension_observations or {}
    if not isinstance(supplied, Mapping):
        raise ValueError("PREFLIGHT_COMPARATOR_DIMENSIONS_INVALID")
    dimensions: dict[str, dict[str, Any]] = {}
    for name in COMPARATOR_DIMENSIONS:
        raw = supplied.get(name)
        if raw is None:
            dimensions[name] = {"status": COMPARATOR_NOT_OBSERVED, "value": None,
                                "evidence_hash": None, "reason": "not supplied"}
            continue
        if not isinstance(raw, Mapping):
            raise ValueError(f"PREFLIGHT_COMPARATOR_DIMENSION_INVALID:{name}")
        status = _text(raw.get("status"), f"{name}_status")
        if status not in _COMPARATOR_STATES:
            raise ValueError(f"PREFLIGHT_COMPARATOR_STATUS_INVALID:{name}")
        evidence_hash = raw.get("evidence_hash")
        if evidence_hash is not None and not _is_hash(evidence_hash):
            raise ValueError(f"PREFLIGHT_COMPARATOR_EVIDENCE_HASH_INVALID:{name}")
        reason = _text(raw.get("reason"), f"{name}_reason", optional=True)
        if status in {COMPARATOR_MATCHED, COMPARATOR_MISMATCHED} and evidence_hash is None:
            raise ValueError(f"PREFLIGHT_COMPARATOR_EVIDENCE_REQUIRED:{name}")
        if status in {COMPARATOR_MATCHED, COMPARATOR_MISMATCHED} and raw.get("value") is None:
            raise ValueError(f"PREFLIGHT_COMPARATOR_VALUE_REQUIRED:{name}")
        if status in {COMPARATOR_UNSUPPORTED, COMPARATOR_NOT_OBSERVED, COMPARATOR_NOT_APPLICABLE} and reason is None:
            raise ValueError(f"PREFLIGHT_COMPARATOR_REASON_REQUIRED:{name}")
        dimensions[name] = {
            "status": status,
            "value": raw.get("value"),
            "evidence_hash": evidence_hash,
            "reason": reason,
        }

    sampling: dict[str, Any] = {"requested_fields": [], "accepted_fields": [],
                                "semantic_equivalents": {}, "unmatched_fields": []}
    if sampling_request is not None:
        if not isinstance(sampling_request, Mapping):
            raise ValueError("PREFLIGHT_SAMPLING_REQUEST_INVALID")
        requested = sampling_request.get("requested_fields", [])
        accepted = sampling_request.get("accepted_fields", [])
        equivalents = sampling_request.get("semantic_equivalents", {})
        if not isinstance(requested, (list, tuple)) or not isinstance(accepted, (list, tuple)):
            raise ValueError("PREFLIGHT_SAMPLING_FIELDS_INVALID")
        if not isinstance(equivalents, Mapping):
            raise ValueError("PREFLIGHT_SAMPLING_EQUIVALENTS_INVALID")
        req = sorted({_text(item, "requested_sampling_field") for item in requested})
        acc = sorted({_text(item, "accepted_sampling_field") for item in accepted})
        eq = {str(k): _text(v, "semantic_equivalent") for k, v in equivalents.items()}
        unmatched = [field for field in req if field not in acc and eq.get(field) not in acc]
        sampling = {"requested_fields": req, "accepted_fields": acc,
                    "semantic_equivalents": eq, "unmatched_fields": unmatched}
        if unmatched:
            amended = None
            if compatibility_amendment is not None:
                if not isinstance(compatibility_amendment, Mapping):
                    raise ValueError("PREFLIGHT_COMPATIBILITY_AMENDMENT_INVALID")
                amendment_id = _text(compatibility_amendment.get("amendment_id"), "amendment_id")
                amendment_hash = compatibility_amendment.get("sha256")
                if not _is_hash(amendment_hash):
                    raise ValueError("PREFLIGHT_COMPATIBILITY_AMENDMENT_HASH_INVALID")
                if compatibility_amendment.get("frozen_before_formal_run") is not True:
                    raise ValueError("PREFLIGHT_COMPATIBILITY_AMENDMENT_NOT_PREFROZEN")
                amended_fields = compatibility_amendment.get("amended_request_fields")
                if not isinstance(amended_fields, (list, tuple)):
                    raise ValueError("PREFLIGHT_AMENDED_FIELDS_INVALID")
                amended_fields = sorted({_text(item, "amended_request_field") for item in amended_fields})
                if any(field not in acc for field in amended_fields):
                    raise ValueError("PREFLIGHT_AMENDED_FIELD_NOT_ACCEPTED")
                if not set(unmatched).issubset(set(req) - set(amended_fields)):
                    raise ValueError("PREFLIGHT_AMENDMENT_DOES_NOT_RESOLVE_FIELDS")
                amended = {"amendment_id": amendment_id, "sha256": amendment_hash,
                           "frozen_before_formal_run": True,
                           "amended_request_fields": amended_fields}
                sampling["unmatched_fields"] = []
                sampling["amended_from_fields"] = req
                sampling["requested_fields"] = amended_fields
                dimensions["sampling_fields"].update(
                    {"status": COMPARATOR_MATCHED, "evidence_hash": amendment_hash,
                     "value": {"requested_fields": amended_fields, "accepted_fields": acc,
                               "semantic_equivalents": eq},
                     "reason": "frozen compatibility amendment resolves rejected request fields"}
                )
            sampling["compatibility_amendment"] = amended
        elif compatibility_amendment is not None:
            raise ValueError("PREFLIGHT_UNNECESSARY_COMPATIBILITY_AMENDMENT")
    elif compatibility_amendment is not None:
        raise ValueError("PREFLIGHT_AMENDMENT_WITHOUT_SAMPLING_REQUEST")
    elif dimensions["sampling_fields"]["status"] != COMPARATOR_NOT_APPLICABLE:
        dimensions["sampling_fields"] = {
            "status": COMPARATOR_NOT_OBSERVED,
            "value": None,
            "evidence_hash": None,
            "reason": "sampling request configuration not supplied",
        }

    if sampling.get("unmatched_fields"):
        dimensions["sampling_fields"]["status"] = COMPARATOR_MISMATCHED
        dimensions["sampling_fields"]["reason"] = "requested fields rejected without a resolving frozen amendment"
    dimension_states = [item["status"] for item in dimensions.values()]
    formal_comparable = all(
        state in {COMPARATOR_MATCHED, COMPARATOR_NOT_APPLICABLE}
        for state in dimension_states
    )
    if sampling.get("unmatched_fields"):
        formal_comparable = False
    comparison_scope = _comparator_scope(dimensions, sampling)
    body = {
        "schema": "nexus.learning_comparator_preflight.v1",
        "comparator_identity": _text(comparator_identity, "comparator_identity"),
        "dimensions": dimensions,
        "sampling_request": sampling,
        "formal_comparable": formal_comparable,
        "comparison_scope": comparison_scope,
        "claim_ceiling": "comparator compatibility evidence only; no model or runtime claim",
    }
    body["binding_hash"] = _hash(body)
    validate_comparator_preflight(body)
    return body


def validate_comparator_preflight(evidence: Any) -> dict[str, Any]:
    if not isinstance(evidence, Mapping) or evidence.get("schema") != "nexus.learning_comparator_preflight.v1":
        raise ValueError("PREFLIGHT_COMPARATOR_SCHEMA_INVALID")
    if set(evidence) != {"schema", "comparator_identity", "dimensions", "sampling_request",
                         "formal_comparable", "comparison_scope", "claim_ceiling", "binding_hash"}:
        raise ValueError("PREFLIGHT_COMPARATOR_SCHEMA_INVALID")
    if not isinstance(evidence.get("dimensions"), Mapping) or set(evidence["dimensions"]) != set(COMPARATOR_DIMENSIONS):
        raise ValueError("PREFLIGHT_COMPARATOR_DIMENSIONS_INVALID")
    for name, item in evidence["dimensions"].items():
        if (
            not isinstance(item, Mapping)
            or set(item) != {"status", "value", "evidence_hash", "reason"}
            or not isinstance(item.get("status"), str)
            or item.get("status") not in _COMPARATOR_STATES
        ):
            raise ValueError(f"PREFLIGHT_COMPARATOR_STATUS_INVALID:{name}")
        if item.get("status") in {COMPARATOR_MATCHED, COMPARATOR_MISMATCHED} and not _is_hash(item.get("evidence_hash")):
            raise ValueError(f"PREFLIGHT_COMPARATOR_EVIDENCE_REQUIRED:{name}")
        if item.get("status") in {COMPARATOR_MATCHED, COMPARATOR_MISMATCHED} and item.get("value") is None:
            raise ValueError(f"PREFLIGHT_COMPARATOR_VALUE_REQUIRED:{name}")
        if item.get("status") in {COMPARATOR_UNSUPPORTED, COMPARATOR_NOT_OBSERVED, COMPARATOR_NOT_APPLICABLE}:
            reason = item.get("reason")
            if not isinstance(reason, str) or not reason.strip():
                raise ValueError(f"PREFLIGHT_COMPARATOR_REASON_REQUIRED:{name}")
    sampling = evidence.get("sampling_request")
    if not isinstance(sampling, Mapping):
        raise ValueError("PREFLIGHT_SAMPLING_REQUEST_INVALID")
    required_sampling_keys = {
        "requested_fields", "accepted_fields", "semantic_equivalents", "unmatched_fields"
    }
    if not required_sampling_keys.issubset(sampling) or set(sampling) - (
        required_sampling_keys | {"compatibility_amendment", "amended_from_fields"}
    ):
        raise ValueError("PREFLIGHT_SAMPLING_REQUEST_INVALID")
    requested = sampling.get("requested_fields")
    accepted = sampling.get("accepted_fields")
    equivalents = sampling.get("semantic_equivalents")
    unmatched = sampling.get("unmatched_fields")
    if (
        not isinstance(requested, list)
        or not isinstance(accepted, list)
        or not isinstance(equivalents, Mapping)
        or not isinstance(unmatched, list)
        or any(not isinstance(field, str) or not field.strip() for field in requested + accepted + unmatched)
        or requested != sorted(set(requested))
        or accepted != sorted(set(accepted))
        or unmatched != sorted(set(unmatched))
        or any(not isinstance(key, str) or not key.strip() or not isinstance(value, str) or not value.strip()
               for key, value in equivalents.items())
    ):
        raise ValueError("PREFLIGHT_SAMPLING_FIELDS_INVALID")
    expected_unmatched = sorted(
        field for field in requested
        if field not in accepted and equivalents.get(field) not in accepted
    )
    if unmatched != expected_unmatched:
        raise ValueError("PREFLIGHT_SAMPLING_UNMATCHED_FIELDS_MISMATCH")
    amendment = sampling.get("compatibility_amendment")
    amended_from = sampling.get("amended_from_fields")
    if amendment is not None:
        if (
            not isinstance(amendment, Mapping)
            or set(amendment) != {"amendment_id", "sha256", "frozen_before_formal_run", "amended_request_fields"}
            or not isinstance(amendment.get("amendment_id"), str)
            or not amendment["amendment_id"].strip()
            or not _is_hash(amendment.get("sha256"))
            or amendment.get("frozen_before_formal_run") is not True
            or amendment.get("amended_request_fields") != requested
            or amended_from is None
            or not isinstance(amended_from, list)
            or any(not isinstance(field, str) or not field.strip() for field in amended_from)
            or amended_from != sorted(set(amended_from))
            or not set(requested).issubset(amended_from)
            or not set(requested).issubset(accepted)
            or unmatched
        ):
            raise ValueError("PREFLIGHT_COMPATIBILITY_AMENDMENT_INVALID")
    elif amended_from is not None:
        raise ValueError("PREFLIGHT_COMPATIBILITY_AMENDMENT_INVALID")
    expected = all(item["status"] in {COMPARATOR_MATCHED, COMPARATOR_NOT_APPLICABLE}
                   for item in evidence["dimensions"].values()) and not unmatched
    if evidence.get("formal_comparable") is not expected:
        raise ValueError("PREFLIGHT_COMPARATOR_DISPOSITION_MISMATCH")
    expected_scope = _comparator_scope(evidence["dimensions"], sampling)
    if evidence.get("comparison_scope") != expected_scope:
        raise ValueError("PREFLIGHT_COMPARATOR_SCOPE_MISMATCH")
    if evidence.get("claim_ceiling") != "comparator compatibility evidence only; no model or runtime claim":
        raise ValueError("PREFLIGHT_COMPARATOR_CLAIM_CEILING_INVALID")
    if evidence.get("binding_hash") != _hash_payload(evidence):
        raise ValueError("PREFLIGHT_COMPARATOR_BINDING_HASH_MISMATCH")
    return dict(evidence)


def _comparator_scope(dimensions: Mapping[str, Mapping[str, Any]], sampling: Mapping[str, Any]) -> str:
    """Describe exactly what the receipt can compare without implying parity."""
    stack_dimensions = {"cache_semantics", "concurrency_and_admission_policy", "material_runtime_fields"}
    stack_difference = any(
        dimensions[name]["status"] in {COMPARATOR_MISMATCHED, COMPARATOR_UNSUPPORTED}
        for name in stack_dimensions
    )
    sampling_compatible = not sampling.get("unmatched_fields")
    non_stack_compatible = all(
        dimensions[name]["status"] in {COMPARATOR_MATCHED, COMPARATOR_NOT_APPLICABLE}
        for name in dimensions
        if name not in stack_dimensions
    )
    stack_observed = all(dimensions[name]["status"] != COMPARATOR_NOT_OBSERVED for name in stack_dimensions)
    if stack_difference and sampling_compatible and non_stack_compatible and stack_observed:
        return "WHOLE_RUNTIME_STACK_ONLY"
    if all(item["status"] in {COMPARATOR_MATCHED, COMPARATOR_NOT_APPLICABLE} for item in dimensions.values()) and sampling_compatible:
        return "CONFIGURATION_MATCHED"
    return "NOT_COMPARABLE"


def _comparator_ready_for_scope(evidence: Mapping[str, Any]) -> bool:
    """Allow explicit whole-stack observations without claiming parity."""
    if evidence.get("formal_comparable") is True:
        return True
    if evidence.get("comparison_scope") != "WHOLE_RUNTIME_STACK_ONLY":
        return False
    sampling = evidence.get("sampling_request")
    dimensions = evidence.get("dimensions")
    if not isinstance(sampling, Mapping) or sampling.get("unmatched_fields"):
        return False
    if not isinstance(dimensions, Mapping):
        return False
    stack_dimensions = {
        "cache_semantics", "concurrency_and_admission_policy", "material_runtime_fields"
    }
    for name, item in dimensions.items():
        status = item.get("status") if isinstance(item, Mapping) else None
        if status == COMPARATOR_NOT_OBSERVED:
            return False
        if name not in stack_dimensions and status not in {
            COMPARATOR_MATCHED, COMPARATOR_NOT_APPLICABLE
        }:
            return False
    return True


def compare_comparator_preflights(
    left: Mapping[str, Any], right: Mapping[str, Any]
) -> dict[str, Any]:
    """Detect material configuration drift before two comparator receipts are mixed."""
    first = validate_comparator_preflight(left)
    second = validate_comparator_preflight(right)
    changed = [
        name for name in COMPARATOR_DIMENSIONS
        if any(
            first["dimensions"][name][field] != second["dimensions"][name][field]
            for field in ("status", "value", "reason")
        )
    ]
    if first["sampling_request"] != second["sampling_request"]:
        changed.append("sampling_request")
    if first["comparator_identity"] != second["comparator_identity"]:
        changed.append("comparator_identity")
    stack_only = any(
        name in {"cache_semantics", "concurrency_and_admission_policy", "material_runtime_fields", "comparator_identity"}
        for name in changed
    ) or first["comparison_scope"] == "WHOLE_RUNTIME_STACK_ONLY" or second["comparison_scope"] == "WHOLE_RUNTIME_STACK_ONLY"
    blocking_changes = any(
        name in {"sampling_fields", "reasoning_mode", "context_and_prefill_policy", "sampling_request"}
        for name in changed
    )
    scope_ready = _comparator_ready_for_scope(first) and _comparator_ready_for_scope(second)
    configuration_matched = (
        not changed and first["formal_comparable"] and second["formal_comparable"]
    )
    stack_comparable = scope_ready and stack_only and not blocking_changes
    compatible = configuration_matched or stack_comparable
    return {
        "schema": "nexus.learning_comparator_pair_comparison.v1",
        "left_binding_hash": first["binding_hash"],
        "right_binding_hash": second["binding_hash"],
        "changed_dimensions": sorted(changed),
        "evidence_mixing_allowed": compatible,
        "comparison_scope": (
            "CONFIGURATION_MATCHED" if configuration_matched
            else "WHOLE_RUNTIME_STACK_ONLY" if stack_comparable
            else "NOT_COMPARABLE"
        ),
        "claim_ceiling": "comparator scope evidence only; no isolated runtime or scheduler claim",
    }


def build_resource_observation(
    *,
    planned_concurrency_levels: Sequence[int],
    observations: Sequence[Mapping[str, Any]],
    stop_on_resource_failure: bool,
) -> dict[str, Any]:
    """Bind per-level resource/admission evidence and any frozen skip decision."""
    if not isinstance(stop_on_resource_failure, bool):
        raise ValueError("PREFLIGHT_RESOURCE_STOP_POLICY_INVALID")
    planned = list(planned_concurrency_levels)
    if not planned or any(isinstance(n, bool) or not isinstance(n, int) or n < 1 for n in planned):
        raise ValueError("PREFLIGHT_PLANNED_CONCURRENCY_INVALID")
    if len(planned) != len(set(planned)) or planned != sorted(planned):
        raise ValueError("PREFLIGHT_PLANNED_CONCURRENCY_MUST_BE_SORTED_UNIQUE")
    normalized: list[dict[str, Any]] = []
    seen: set[int] = set()
    first_failure: int | None = None
    last_level = 0
    for raw in observations:
        if not isinstance(raw, Mapping):
            raise ValueError("PREFLIGHT_RESOURCE_OBSERVATION_INVALID")
        level = raw.get("requested_concurrency")
        if isinstance(level, bool) or not isinstance(level, int) or level not in planned or level in seen:
            raise ValueError("PREFLIGHT_RESOURCE_CONCURRENCY_INVALID")
        if level <= last_level:
            raise ValueError("PREFLIGHT_RESOURCE_LEVELS_MUST_BE_OBSERVED_IN_ORDER")
        last_level = level
        if first_failure is not None and stop_on_resource_failure and level > first_failure:
            raise ValueError("PREFLIGHT_RESOURCE_LEVEL_AFTER_FROZEN_STOP")
        seen.add(level)
        warmup = _text(raw.get("warmup_policy"), "warmup_policy")
        batches = raw.get("measured_batch_count")
        errors = raw.get("error_count")
        if isinstance(batches, bool) or not isinstance(batches, int) or batches < 0:
            raise ValueError("PREFLIGHT_RESOURCE_BATCH_COUNT_INVALID")
        if isinstance(errors, bool) or not isinstance(errors, int) or errors < 0:
            raise ValueError("PREFLIGHT_RESOURCE_ERROR_COUNT_INVALID")
        admission = _text(raw.get("admission_state"), "admission_state")
        recovery = _text(raw.get("recovery_after_batch"), "recovery_after_batch")
        if admission not in _ADMISSION_STATES or recovery not in _RECOVERY_STATES:
            raise ValueError("PREFLIGHT_RESOURCE_STATE_INVALID")
        headroom_state = _text(raw.get("headroom_state"), "headroom_state")
        if headroom_state not in {"AVAILABLE", "LOW", "EXHAUSTED", "NOT_OBSERVED"}:
            raise ValueError("PREFLIGHT_RESOURCE_HEADROOM_STATE_INVALID")
        headroom = raw.get("headroom_observation")
        if headroom_state == "NOT_OBSERVED" and headroom is not None:
            raise ValueError("PREFLIGHT_RESOURCE_HEADROOM_UNKNOWN_MUST_BE_NULL")
        failure = admission in {HELD, REFUSED} or headroom_state == "EXHAUSTED" or errors > 0
        item = {
            "requested_concurrency": level,
            "warmup_policy": warmup,
            "measured_batch_count": batches,
            "headroom_state": headroom_state,
            "headroom_observation": headroom,
            "admission_state": admission,
            "error_count": errors,
            "recovery_after_batch": recovery,
            "resource_failure_observed": failure,
        }
        normalized.append(item)
        if failure and first_failure is None:
            first_failure = level
    normalized.sort(key=lambda item: item["requested_concurrency"])
    skipped = [n for n in planned if first_failure is not None and n > first_failure] if stop_on_resource_failure else []
    unobserved = [n for n in planned if n not in seen and n not in skipped]
    payload = {
        "schema": "nexus.learning_resource_headroom_observation.v1",
        "planned_concurrency_levels": planned,
        "observations": normalized,
        "stop_on_resource_failure": stop_on_resource_failure,
        "first_failure_concurrency": first_failure,
        "skipped_concurrency_levels": skipped,
        "unobserved_concurrency_levels": unobserved,
        "claim_ceiling": "resource/admission observation only; no capacity or admission authority",
    }
    payload["binding_hash"] = _hash(payload)
    validate_resource_observation(payload)
    return payload


def validate_resource_observation(evidence: Any) -> dict[str, Any]:
    if not isinstance(evidence, Mapping) or evidence.get("schema") != "nexus.learning_resource_headroom_observation.v1":
        raise ValueError("PREFLIGHT_RESOURCE_SCHEMA_INVALID")
    if set(evidence) != {"schema", "planned_concurrency_levels", "observations", "stop_on_resource_failure",
                         "first_failure_concurrency", "skipped_concurrency_levels",
                         "unobserved_concurrency_levels", "claim_ceiling", "binding_hash"}:
        raise ValueError("PREFLIGHT_RESOURCE_SCHEMA_INVALID")
    planned = evidence.get("planned_concurrency_levels")
    observations = evidence.get("observations")
    stop = evidence.get("stop_on_resource_failure")
    if (not isinstance(planned, list) or not planned or planned != sorted(set(planned))
            or any(isinstance(item, bool) or not isinstance(item, int) or item < 1 for item in planned)
            or not isinstance(observations, list) or not isinstance(stop, bool)):
        raise ValueError("PREFLIGHT_RESOURCE_FIELDS_INVALID")
    seen: set[int] = set()
    first_failure: int | None = None
    last_level = 0
    for item in observations:
        if not isinstance(item, Mapping):
            raise ValueError("PREFLIGHT_RESOURCE_OBSERVATION_INVALID")
        level = item.get("requested_concurrency")
        if isinstance(level, bool) or not isinstance(level, int) or level not in planned or level in seen:
            raise ValueError("PREFLIGHT_RESOURCE_CONCURRENCY_INVALID")
        if level <= last_level:
            raise ValueError("PREFLIGHT_RESOURCE_LEVELS_MUST_BE_OBSERVED_IN_ORDER")
        last_level = level
        if first_failure is not None and stop and level > first_failure:
            raise ValueError("PREFLIGHT_RESOURCE_LEVEL_AFTER_FROZEN_STOP")
        seen.add(level)
        admission = item.get("admission_state")
        headroom = item.get("headroom_state")
        errors = item.get("error_count")
        if admission not in _ADMISSION_STATES or headroom not in {"AVAILABLE", "LOW", "EXHAUSTED", "NOT_OBSERVED"}:
            raise ValueError("PREFLIGHT_RESOURCE_STATE_INVALID")
        if isinstance(errors, bool) or not isinstance(errors, int) or errors < 0:
            raise ValueError("PREFLIGHT_RESOURCE_ERROR_COUNT_INVALID")
        failure = admission in {HELD, REFUSED} or headroom == "EXHAUSTED" or errors > 0
        if item.get("resource_failure_observed") is not failure:
            raise ValueError("PREFLIGHT_RESOURCE_FAILURE_PROOF_MISMATCH")
        if failure and first_failure is None:
            first_failure = level
    expected_skipped = [n for n in planned if first_failure is not None and n > first_failure] if stop else []
    expected_unobserved = [n for n in planned if n not in seen and n not in expected_skipped]
    if evidence.get("first_failure_concurrency") != first_failure or evidence.get("skipped_concurrency_levels") != expected_skipped:
        raise ValueError("PREFLIGHT_RESOURCE_STOP_PROOF_MISMATCH")
    if evidence.get("unobserved_concurrency_levels") != expected_unobserved:
        raise ValueError("PREFLIGHT_RESOURCE_UNOBSERVED_LEVELS_MISMATCH")
    if evidence.get("claim_ceiling") != "resource/admission observation only; no capacity or admission authority":
        raise ValueError("PREFLIGHT_RESOURCE_CLAIM_CEILING_INVALID")
    if evidence.get("binding_hash") != _hash_payload(evidence):
        raise ValueError("PREFLIGHT_RESOURCE_BINDING_HASH_MISMATCH")
    return dict(evidence)


def _normalize_gate_specs(raw_gates: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    gates: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw in raw_gates:
        if not isinstance(raw, Mapping):
            raise ValueError("PREFLIGHT_GATE_SPEC_INVALID")
        gate_id = _text(raw.get("gate_id"), "gate_id")
        if gate_id in seen:
            raise ValueError("PREFLIGHT_GATE_ID_DUPLICATE")
        seen.add(gate_id)
        phase = _text(raw.get("phase"), "gate_phase")
        if phase not in {STAGE_SMOKE, STAGE_CALIBRATION, "HOLDOUT"}:
            raise ValueError("PREFLIGHT_GATE_PHASE_INVALID")
        comparator = _text(raw.get("comparator"), "gate_comparator")
        if comparator not in _GATE_COMPARATORS:
            raise ValueError("PREFLIGHT_GATE_COMPARATOR_INVALID")
        kind = _text(raw.get("kind"), "gate_kind")
        if kind not in _GATE_KINDS:
            raise ValueError("PREFLIGHT_GATE_KIND_INVALID")
        if not isinstance(raw.get("hard"), bool) or raw.get("hard") is not True:
            raise ValueError("PREFLIGHT_GATE_MUST_BE_PREREGISTERED_HARD_GATE")
        gates.append({"gate_id": gate_id, "phase": phase,
                      "metric": _text(raw.get("metric"), "gate_metric"),
                      "comparator": comparator, "threshold": _number(raw.get("threshold"), "gate_threshold"),
                      "kind": kind, "hard": True})
    return sorted(gates, key=lambda item: item["gate_id"])


def _eval_gate(gate: Mapping[str, Any], observed: float) -> bool:
    comp = gate["comparator"]
    threshold = gate["threshold"]
    if comp == GATE_GTE:
        return observed >= threshold
    if comp == GATE_LTE:
        return observed <= threshold
    return observed == threshold


def build_cohort_preflight(
    *,
    experiment_id: str,
    cases: Sequence[Mapping[str, Any]],
    independence_unit: str,
    value_claim: bool,
    ground_truth_provenance: Mapping[str, Any] | None,
    leakage_observation: Mapping[str, Any] | None,
    incumbent: Mapping[str, Any] | None = None,
    incumbent_inapplicability_reason: str | None = None,
    comparator_preflight: Mapping[str, Any] | None = None,
    resource_observation: Mapping[str, Any] | None = None,
    hard_gates: Sequence[Mapping[str, Any]] = (),
    resource_stop_policy: Mapping[str, Any] | None = None,
    experiment_integrity: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Create hash-bound cohort readiness evidence, returning FAIL/DEFER explicitly."""
    if independence_unit not in {INDEPENDENCE_UNIT_ROW, INDEPENDENCE_UNIT_BASE}:
        raise ValueError("PREFLIGHT_INDEPENDENCE_UNIT_INVALID")
    if not isinstance(value_claim, bool):
        raise ValueError("PREFLIGHT_VALUE_CLAIM_INVALID")
    gates = _normalize_gate_specs(hard_gates)
    if not isinstance(cases, (list, tuple)) or not cases:
        raise ValueError("PREFLIGHT_COHORT_EMPTY")
    if experiment_integrity is not None:
        validate_experiment_integrity(experiment_integrity)
        if experiment_integrity.get("experiment_id") != experiment_id:
            raise ValueError("PREFLIGHT_EXPERIMENT_INTEGRITY_ID_MISMATCH")

    normalized_cases: list[dict[str, Any]] = []
    for raw in cases:
        if not isinstance(raw, Mapping):
            raise ValueError("PREFLIGHT_CASE_INVALID")
        split = _text(raw.get("split"), "case_split").upper()
        if split not in {"CALIBRATION", "HOLDOUT"}:
            raise ValueError("PREFLIGHT_CASE_SPLIT_INVALID")
        item = {
            "case_id": _text(raw.get("case_id"), "case_id"),
            "split": split,
            "source_group_id": _text(raw.get("source_group_id"), "source_group_id"),
            "outcome": _text(raw.get("outcome"), "outcome", optional=True),
        }
        normalized_cases.append(item)
    normalized_cases.sort(key=lambda item: (item["split"], item["case_id"], item["source_group_id"], item["outcome"] or ""))
    id_counts = Counter(item["case_id"] for item in normalized_cases)
    duplicates = sorted(ident for ident, count in id_counts.items() if count > 1)
    calibration = [item for item in normalized_cases if item["split"] == "CALIBRATION"]
    heldout = [item for item in normalized_cases if item["split"] == "HOLDOUT"]
    if not calibration or not heldout:
        raise ValueError("PREFLIGHT_BOTH_SPLITS_REQUIRED")
    direct_overlap = sorted(set(item["case_id"] for item in calibration) & set(item["case_id"] for item in heldout))
    group_overlap = sorted(set(item["source_group_id"] for item in calibration) & set(item["source_group_id"] for item in heldout))
    if experiment_integrity is not None:
        expected_cal = set(experiment_integrity["calibration"]["member_identities"])
        expected_hold = set(experiment_integrity["heldout"]["member_identities"])
        actual_cal = {item["case_id"] if independence_unit == INDEPENDENCE_UNIT_ROW else item["source_group_id"] for item in calibration}
        actual_hold = {item["case_id"] if independence_unit == INDEPENDENCE_UNIT_ROW else item["source_group_id"] for item in heldout}
        if (expected_cal, expected_hold) != (actual_cal, actual_hold):
            raise ValueError("PREFLIGHT_COHORT_EXPERIMENT_INTEGRITY_MISMATCH")

    labels = [item["outcome"] for item in normalized_cases if item["outcome"] is not None]
    label_counts: dict[str, int] = {}
    for label in labels:
        label_counts[label] = label_counts.get(label, 0) + 1
    label_count = len(labels)
    label_kinds = len(label_counts)
    top_label = sorted(label_counts, key=lambda label: (-label_counts[label], label))[0] if label_counts else None
    top_count = label_counts.get(top_label, 0) if top_label is not None else 0
    baseline = {
        "status": "COMPUTED" if labels else "NOT_APPLICABLE",
        "inapplicability_reason": None if labels else "no declared outcome labels",
        "label_counts": dict(sorted(label_counts.items())),
        "case_count_with_labels": label_count,
        "distinct_outcome_count": label_kinds,
        "constant_outcome": bool(label_count and label_kinds == 1),
        "constant_answer": top_label,
        "constant_answer_correct_count": top_count,
        "constant_answer_accuracy": (top_count / label_count) if label_count else None,
        "explains_perfect_nominal_accuracy": bool(label_count and label_kinds == 1),
    }

    truth: dict[str, Any]
    pending: list[str] = []
    failures: list[str] = []
    if ground_truth_provenance is None:
        truth = {"status": "MISSING", "payload": None, "payload_hash": None}
        pending.append("ground_truth_provenance_missing")
    else:
        if not isinstance(ground_truth_provenance, Mapping) or not ground_truth_provenance:
            raise ValueError("PREFLIGHT_GROUND_TRUTH_PROVENANCE_INVALID")
        truth_payload = dict(ground_truth_provenance)
        truth = {"status": "DECLARED", "payload": truth_payload, "payload_hash": _hash(truth_payload)}

    if leakage_observation is None:
        leakage = {"status": LEAKAGE_NOT_EVALUATED, "reason": "not supplied", "evidence_hash": None,
                   "evaluator": None}
        pending.append("leakage_not_evaluated")
    else:
        if not isinstance(leakage_observation, Mapping):
            raise ValueError("PREFLIGHT_LEAKAGE_OBSERVATION_INVALID")
        leakage_state = _text(leakage_observation.get("status"), "leakage_status").upper()
        if leakage_state not in _LEAKAGE_STATES:
            raise ValueError("PREFLIGHT_LEAKAGE_STATUS_INVALID")
        leakage_hash = leakage_observation.get("evidence_hash")
        leakage_reason = _text(leakage_observation.get("reason"), "leakage_reason", optional=True)
        evaluator = _text(leakage_observation.get("evaluator"), "leakage_evaluator", optional=True)
        if leakage_state in {LEAKAGE_PASS, LEAKAGE_FAIL} and (not _is_hash(leakage_hash) or not evaluator):
            raise ValueError("PREFLIGHT_LEAKAGE_EVIDENCE_REQUIRED")
        if leakage_state in {LEAKAGE_NOT_EVALUATED, LEAKAGE_UNSUPPORTED} and not leakage_reason:
            raise ValueError("PREFLIGHT_LEAKAGE_REASON_REQUIRED")
        leakage = {"status": leakage_state, "reason": leakage_reason,
                   "evidence_hash": leakage_hash, "evaluator": evaluator}
        if leakage_state == LEAKAGE_FAIL:
            failures.append("leakage_check_failed")
        elif leakage_state != LEAKAGE_PASS:
            pending.append("leakage_not_passed")

    if value_claim:
        if incumbent is None:
            pending.append("incumbent_identity_missing_for_value_claim")
            incumbent_projection = None
        elif not isinstance(incumbent, Mapping) or not incumbent:
            raise ValueError("PREFLIGHT_INCUMBENT_INVALID")
        else:
            incumbent_projection = dict(incumbent)
    else:
        if incumbent is None and not _text(incumbent_inapplicability_reason, "incumbent_inapplicability_reason", optional=True):
            pending.append("incumbent_inapplicability_not_explained")
        incumbent_projection = dict(incumbent) if isinstance(incumbent, Mapping) else None

    comparator: dict[str, Any] | None = None
    if comparator_preflight is None:
        pending.append("comparator_preflight_missing")
    else:
        comparator = validate_comparator_preflight(comparator_preflight)
        if not _comparator_ready_for_scope(comparator):
            blocking_mismatch = any(
                item["status"] == COMPARATOR_MISMATCHED
                and name not in {
                    "cache_semantics", "concurrency_and_admission_policy", "material_runtime_fields"
                }
                for name, item in comparator["dimensions"].items()
            ) or bool(comparator["sampling_request"].get("unmatched_fields"))
            if blocking_mismatch:
                failures.append("comparator_material_mismatch")
            else:
                pending.append("comparator_not_fully_observed")

    resource: dict[str, Any] | None = None
    if resource_observation is None:
        pending.append("resource_headroom_not_observed")
    else:
        resource = validate_resource_observation(resource_observation)
        if resource["unobserved_concurrency_levels"]:
            pending.append("resource_concurrency_levels_not_fully_observed")
        resource_not_applicable = (
            isinstance(resource_stop_policy, Mapping)
            and resource_stop_policy.get("applicability") == "NOT_APPLICABLE"
            and isinstance(resource_stop_policy.get("inapplicability_reason"), str)
            and bool(resource_stop_policy["inapplicability_reason"].strip())
        )
        if not resource_not_applicable and any(
            row["headroom_state"] == "NOT_OBSERVED" or row["admission_state"] == NOT_OBSERVED
            for row in resource["observations"]
        ):
            pending.append("resource_headroom_telemetry_incomplete")
        if resource["first_failure_concurrency"] is not None:
            if resource_stop_policy is None or not isinstance(resource_stop_policy, Mapping):
                pending.append("resource_failure_stop_policy_missing")
            else:
                if resource_stop_policy.get("stop_on_resource_failure") != resource["stop_on_resource_failure"]:
                    raise ValueError("PREFLIGHT_RESOURCE_STOP_POLICY_MISMATCH")
                if resource["stop_on_resource_failure"]:
                    failures.append("resource_admission_gate_failed")
        elif resource_stop_policy is not None and not isinstance(resource_stop_policy, Mapping):
            raise ValueError("PREFLIGHT_RESOURCE_STOP_POLICY_INVALID")

    if value_claim and baseline["constant_outcome"]:
        pending.append("constant_answer_baseline_saturates_cohort")
    if label_count < len(normalized_cases):
        pending.append("outcome_labels_incomplete")
    if not gates:
        pending.append("hard_gates_not_preregistered")
    if experiment_integrity is None:
        pending.append("experiment_integrity_binding_missing")
    else:
        if experiment_integrity.get("calibration_status") != CALIBRATED:
            pending.append("issue_20_calibration_not_calibrated")
        terminal_outcome = experiment_integrity.get("terminal", {}).get("outcome")
        if terminal_outcome in {TERMINAL_NEGATIVE, TERMINAL_STOP}:
            failures.append("issue_20_terminal_stop_preserved")
        elif terminal_outcome != TERMINAL_PASS:
            pending.append("issue_20_terminal_outcome_not_pass")
        frozen_policy = experiment_integrity.get("frozen_policy", {}).get("policy", {})
        frozen_gates = frozen_policy.get("staged_hard_gates") if isinstance(frozen_policy, Mapping) else None
        if frozen_gates is None:
            pending.append("hard_gates_not_bound_to_frozen_policy")
        elif not isinstance(frozen_gates, (list, tuple)) or _normalize_gate_specs(frozen_gates) != gates:
            failures.append("pre_registered_hard_gate_mismatch")
    if duplicates:
        failures.append("duplicate_case_identity")
    if direct_overlap:
        failures.append("calibration_holdout_case_overlap")
    if group_overlap:
        failures.append("calibration_holdout_source_group_overlap")

    disposition = PREFLIGHT_FAIL if failures else PREFLIGHT_DEFER if pending else PREFLIGHT_PASS
    value_claim_eligible = bool(disposition == PREFLIGHT_PASS and value_claim and incumbent_projection and not baseline["constant_outcome"])
    payload = {
        "schema": COHORT_PREFLIGHT_SCHEMA,
        "experiment_id": _text(experiment_id, "experiment_id"),
        "independence_unit": independence_unit,
        "cohort": {
            "cases": normalized_cases,
            "cohort_hash": _hash(normalized_cases),
            "case_count": len(normalized_cases),
            "calibration_case_count": len(calibration),
            "holdout_case_count": len(heldout),
            "duplicate_case_ids": duplicates,
            "direct_case_overlap": direct_overlap,
            "source_group_overlap": group_overlap,
        },
        "outcome_distribution": baseline,
        "ground_truth_provenance": truth,
        "leakage_observation": leakage,
        "incumbent": incumbent_projection,
        "value_claim_requested": value_claim,
        "value_claim_eligible": value_claim_eligible,
        "incumbent_inapplicability_reason": incumbent_inapplicability_reason if not value_claim else None,
        "comparator_preflight": comparator,
        "resource_observation": resource,
        "resource_stop_policy": dict(resource_stop_policy) if isinstance(resource_stop_policy, Mapping) else None,
        "hard_gates": gates,
        "hard_gates_hash": _hash(gates),
        "experiment_integrity_binding_hash": _hash(dict(experiment_integrity)) if experiment_integrity is not None else None,
        "frozen_policy_hash": (
            "sha256:" + experiment_integrity["frozen_policy"]["policy_hash"]
            if experiment_integrity is not None else None
        ),
        "preflight_disposition": disposition,
        "failure_reasons": sorted(set(failures)),
        "defer_reasons": sorted(set(pending)),
        "holdout_eligible": False,
        "claim_ceiling": PREFLIGHT_CLAIM_CEILING,
    }
    payload["binding_hash"] = _hash(payload)
    validate_cohort_preflight(payload)
    return payload


def validate_cohort_preflight(evidence: Any) -> dict[str, Any]:
    if not isinstance(evidence, Mapping) or evidence.get("schema") != COHORT_PREFLIGHT_SCHEMA:
        raise ValueError("PREFLIGHT_SCHEMA_INVALID")
    expected_keys = {
        "schema", "experiment_id", "independence_unit", "cohort", "outcome_distribution",
        "ground_truth_provenance", "leakage_observation", "incumbent", "value_claim_requested",
        "value_claim_eligible", "incumbent_inapplicability_reason", "comparator_preflight",
        "resource_observation", "resource_stop_policy", "hard_gates", "hard_gates_hash",
        "experiment_integrity_binding_hash", "frozen_policy_hash", "preflight_disposition",
        "failure_reasons", "defer_reasons", "holdout_eligible", "claim_ceiling", "binding_hash",
    }
    if set(evidence) != expected_keys:
        raise ValueError("PREFLIGHT_SCHEMA_INVALID")
    payload = dict(evidence)
    supplied_hash = payload.pop("binding_hash", None)
    if supplied_hash != _hash(payload):
        raise ValueError("PREFLIGHT_BINDING_HASH_MISMATCH")
    experiment_id = _text(evidence.get("experiment_id"), "experiment_id")
    if experiment_id != evidence.get("experiment_id"):
        raise ValueError("PREFLIGHT_EXPERIMENT_ID_INVALID")
    independence_unit = evidence.get("independence_unit")
    if not isinstance(independence_unit, str) or independence_unit not in {
        INDEPENDENCE_UNIT_ROW, INDEPENDENCE_UNIT_BASE
    }:
        raise ValueError("PREFLIGHT_INDEPENDENCE_UNIT_INVALID")
    if evidence.get("claim_ceiling") != PREFLIGHT_CLAIM_CEILING:
        raise ValueError("PREFLIGHT_CLAIM_CEILING_INVALID")
    cohort = evidence.get("cohort")
    cohort_keys = {
        "cases", "cohort_hash", "case_count", "calibration_case_count", "holdout_case_count",
        "duplicate_case_ids", "direct_case_overlap", "source_group_overlap",
    }
    if not isinstance(cohort, Mapping) or set(cohort) != cohort_keys or not isinstance(cohort.get("cases"), list):
        raise ValueError("PREFLIGHT_COHORT_INVALID")
    cases = cohort["cases"]
    if not cases or any(
        not isinstance(item, Mapping)
        or set(item) != {"case_id", "split", "source_group_id", "outcome"}
        or not isinstance(item.get("case_id"), str)
        or not item["case_id"].strip()
        or not isinstance(item.get("split"), str)
        or item["split"] not in {"CALIBRATION", "HOLDOUT"}
        or not isinstance(item.get("source_group_id"), str)
        or not item["source_group_id"].strip()
        or (item.get("outcome") is not None and (not isinstance(item["outcome"], str) or not item["outcome"].strip()))
        for item in cases
    ):
        raise ValueError("PREFLIGHT_COHORT_CASE_INVALID")
    expected_case_order = sorted(
        cases,
        key=lambda item: (item["split"], item["case_id"], item["source_group_id"], item["outcome"] or ""),
    )
    if cases != expected_case_order:
        raise ValueError("PREFLIGHT_COHORT_CASE_ORDER_INVALID")
    if cohort.get("case_count") != len(cases):
        raise ValueError("PREFLIGHT_CASE_COUNT_MISMATCH")
    if cohort.get("cohort_hash") != _hash(cases):
        raise ValueError("PREFLIGHT_COHORT_HASH_MISMATCH")
    case_id_counts = Counter(item["case_id"] for item in cases)
    duplicate_ids = sorted(case_id for case_id, count in case_id_counts.items() if count > 1)
    if duplicate_ids != cohort.get("duplicate_case_ids"):
        raise ValueError("PREFLIGHT_DUPLICATE_CASE_PROOF_MISMATCH")
    cal = [item for item in cases if item.get("split") == "CALIBRATION"]
    hold = [item for item in cases if item.get("split") == "HOLDOUT"]
    if not cal or not hold:
        raise ValueError("PREFLIGHT_BOTH_SPLITS_REQUIRED")
    direct = sorted({item["case_id"] for item in cal} & {item["case_id"] for item in hold})
    groups = sorted({item["source_group_id"] for item in cal} & {item["source_group_id"] for item in hold})
    if direct != cohort.get("direct_case_overlap") or groups != cohort.get("source_group_overlap"):
        raise ValueError("PREFLIGHT_OVERLAP_PROOF_MISMATCH")
    if cohort.get("calibration_case_count") != len(cal) or cohort.get("holdout_case_count") != len(hold):
        raise ValueError("PREFLIGHT_SPLIT_COUNT_MISMATCH")
    truth = evidence.get("ground_truth_provenance")
    if (
        not isinstance(truth, Mapping)
        or not isinstance(truth.get("status"), str)
        or truth.get("status") not in {"DECLARED", "MISSING"}
    ):
        raise ValueError("PREFLIGHT_GROUND_TRUTH_PROVENANCE_INVALID")
    if truth.get("status") == "DECLARED":
        if not isinstance(truth.get("payload"), Mapping) or truth.get("payload_hash") != _hash(dict(truth["payload"])):
            raise ValueError("PREFLIGHT_GROUND_TRUTH_PROVENANCE_HASH_MISMATCH")
    elif truth.get("payload") is not None or truth.get("payload_hash") is not None:
        raise ValueError("PREFLIGHT_MISSING_GROUND_TRUTH_MUST_REMAIN_EXPLICIT")
    leakage = evidence.get("leakage_observation")
    if (
        not isinstance(leakage, Mapping)
        or not isinstance(leakage.get("status"), str)
        or leakage.get("status") not in _LEAKAGE_STATES
    ):
        raise ValueError("PREFLIGHT_LEAKAGE_STATUS_INVALID")
    if leakage.get("status") in {LEAKAGE_PASS, LEAKAGE_FAIL}:
        if not _is_hash(leakage.get("evidence_hash")) or not isinstance(leakage.get("evaluator"), str):
            raise ValueError("PREFLIGHT_LEAKAGE_EVIDENCE_REQUIRED")
    elif not isinstance(leakage.get("reason"), str) or not leakage["reason"].strip():
        raise ValueError("PREFLIGHT_LEAKAGE_REASON_REQUIRED")
    comparator = evidence.get("comparator_preflight")
    if comparator is not None:
        validate_comparator_preflight(comparator)
    resource = evidence.get("resource_observation")
    if resource is not None:
        validate_resource_observation(resource)
    gates = evidence.get("hard_gates")
    if not isinstance(gates, list) or _normalize_gate_specs(gates) != gates:
        raise ValueError("PREFLIGHT_HARD_GATES_INVALID")
    if evidence.get("hard_gates_hash") != _hash(gates):
        raise ValueError("PREFLIGHT_HARD_GATES_HASH_MISMATCH")
    if (
        not isinstance(evidence.get("preflight_disposition"), str)
        or evidence.get("preflight_disposition") not in {PREFLIGHT_PASS, PREFLIGHT_FAIL, PREFLIGHT_DEFER}
    ):
        raise ValueError("PREFLIGHT_DISPOSITION_INVALID")
    if evidence.get("holdout_eligible") is not False:
        raise ValueError("PREFLIGHT_CANNOT_OPEN_HOLDOUT")
    if evidence.get("preflight_disposition") == PREFLIGHT_PASS and (evidence.get("failure_reasons") or evidence.get("defer_reasons")):
        raise ValueError("PREFLIGHT_PASS_WITH_BLOCKERS")
    if evidence.get("preflight_disposition") == PREFLIGHT_FAIL and not evidence.get("failure_reasons"):
        raise ValueError("PREFLIGHT_FAIL_WITHOUT_REASON")
    if evidence.get("preflight_disposition") == PREFLIGHT_PASS and (
        cohort.get("duplicate_case_ids") or cohort.get("direct_case_overlap") or cohort.get("source_group_overlap")
    ):
        raise ValueError("PREFLIGHT_PASS_WITH_COHORT_OVERLAP")
    if evidence.get("preflight_disposition") == PREFLIGHT_PASS and (
        truth.get("status") != "DECLARED"
        or leakage.get("status") != LEAKAGE_PASS
        or not isinstance(evidence.get("comparator_preflight"), Mapping)
        or not _comparator_ready_for_scope(evidence["comparator_preflight"])
        or not isinstance(evidence.get("resource_observation"), Mapping)
        or not evidence.get("hard_gates")
    ):
        raise ValueError("PREFLIGHT_PASS_WITH_INCOMPLETE_READINESS_EVIDENCE")
    if evidence.get("preflight_disposition") == PREFLIGHT_PASS and (
        not _is_hash(evidence.get("experiment_integrity_binding_hash"))
        or not _is_hash(evidence.get("frozen_policy_hash"))
    ):
        raise ValueError("PREFLIGHT_PASS_WITHOUT_ISSUE_20_POLICY_BINDING")
    return dict(evidence)


def _validate_stage_observation(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        raise ValueError("STAGED_GATE_OBSERVATION_INVALID")
    status = _text(raw.get("status"), "stage_observation_status").upper()
    if status not in {"PASS", "FAIL", "DEFER"}:
        raise ValueError("STAGED_GATE_OBSERVATION_STATUS_INVALID")
    evidence_hash = raw.get("evidence_hash")
    if not _is_hash(evidence_hash):
        raise ValueError("STAGED_GATE_OBSERVATION_HASH_REQUIRED")
    metrics = raw.get("metrics", {})
    if not isinstance(metrics, Mapping):
        raise ValueError("STAGED_GATE_METRICS_INVALID")
    normalized_metrics = {str(key): _number(value, "stage_metric") for key, value in metrics.items()}
    observation_id = _text(raw.get("observation_id"), "stage_observation_id")
    return {"observation_id": observation_id, "status": status,
            "evidence_hash": evidence_hash, "metrics": normalized_metrics}


def build_staged_gate_evidence(
    *,
    preflight: Mapping[str, Any],
    stage_observation: Mapping[str, Any] | None = None,
    previous_evidence: Mapping[str, Any] | None = None,
    previous_preflight: Mapping[str, Any] | None = None,
    necessary_condition_stop_evidence: Mapping[str, Any] | None = None,
    experiment_integrity: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Advance a deterministic evidence state machine; terminal negative evidence is sticky."""
    pre = validate_cohort_preflight(preflight)
    recovering_defer = False
    previous: dict[str, Any] | None = None
    if previous_evidence is not None:
        prior_pre = validate_cohort_preflight(previous_preflight if previous_preflight is not None else pre)
        previous = validate_staged_gate_evidence(previous_evidence, preflight=prior_pre)
        if previous["terminal"] is True:
            return dict(previous)
        history = list(previous["history"])
        observations = dict(previous["observations"])
        prior_hash = previous["binding_hash"]
        heldout_opened = previous["heldout_opened"]
        if previous["preflight_binding_hash"] != pre["binding_hash"]:
            same_frozen_identity = all(
                prior_pre.get(field) == pre.get(field)
                for field in (
                    "experiment_id", "hard_gates_hash", "experiment_integrity_binding_hash",
                    "frozen_policy_hash",
                )
            ) and prior_pre["cohort"]["cohort_hash"] == pre["cohort"]["cohort_hash"]
            if (
                previous_preflight is None
                or previous["stage"] != STAGE_DEFER
                or previous["terminal"] is True
                or prior_pre["preflight_disposition"] != PREFLIGHT_DEFER
                or pre["preflight_disposition"] != PREFLIGHT_PASS
                or not same_frozen_identity
                or observations
                or heldout_opened
            ):
                raise ValueError("STAGED_GATE_PREFLIGHT_BINDING_MISMATCH")
            history.append({
                "stage": STAGE_PREFLIGHT,
                "reason": "deferred preflight superseded by complete bound evidence",
                "prior_preflight_binding_hash": prior_pre["binding_hash"],
                "updated_preflight_binding_hash": pre["binding_hash"],
            })
            current_stage = STAGE_PREFLIGHT
        else:
            current_stage = previous["stage"]
            recovering_defer = current_stage == STAGE_DEFER
            if recovering_defer:
                last = history[-1] if history else {}
                previous_phase = last.get("stage")
                phase_map = {
                    STAGE_SMOKE: STAGE_SMOKE,
                    STAGE_CALIBRATION: STAGE_CALIBRATION,
                    "HOLDOUT": STAGE_HOLDOUT_ELIGIBLE,
                }
                if previous_phase in phase_map:
                    current_stage = phase_map[previous_phase]
                elif previous_phase == STAGE_DEFER and last.get("reason") == "preflight incomplete":
                    if stage_observation is not None:
                        raise ValueError("STAGED_GATE_PREFLIGHT_DEFER_REQUIRES_UPDATED_PREFLIGHT")
                    return dict(previous)
                else:
                    raise ValueError("STAGED_GATE_DEFER_RESUME_STATE_INVALID")
    else:
        current_stage = STAGE_PREFLIGHT
        history = []
        observations = {}
        prior_hash = None
        heldout_opened = False

    ncs_decision = None
    if necessary_condition_stop_evidence is not None:
        if experiment_integrity is None:
            raise ValueError("STAGED_GATE_NECESSARY_STOP_REQUIRES_EXPERIMENT_INTEGRITY")
        validate_experiment_integrity(experiment_integrity)
        if pre["experiment_integrity_binding_hash"] != _hash(dict(experiment_integrity)):
            raise ValueError("STAGED_GATE_NECESSARY_STOP_INTEGRITY_BINDING_MISMATCH")
        proof = validate_necessary_condition_stop_evidence(
            necessary_condition_stop_evidence, experiment_integrity=experiment_integrity
        )
        if proof["experiment_id"] != pre["experiment_id"]:
            raise ValueError("STAGED_GATE_NECESSARY_STOP_EXPERIMENT_MISMATCH")
        ncs_decision = proof["terminal_disposition"]
        if ncs_decision == NEGATIVE_STOP:
            current_stage = STAGE_HARD_STOP
            history.append({"stage": STAGE_HARD_STOP, "reason": "#26 non-rescuable necessary condition"})
            return _staged_payload(pre, current_stage, "STOP", True, heldout_opened,
                                   history, observations, prior_hash,
                                   necessary_condition_stop_evidence)
        if ncs_decision != CONTINUE:
            raise ValueError("STAGED_GATE_NECESSARY_STOP_DECISION_INVALID")

    if recovering_defer and stage_observation is None:
        if previous is None:
            raise ValueError("STAGED_GATE_DEFER_RESUME_STATE_INVALID")
        return dict(previous)

    if current_stage == STAGE_PREFLIGHT:
        if pre["preflight_disposition"] == PREFLIGHT_FAIL:
            current_stage = STAGE_STOP
            history.append({"stage": STAGE_STOP, "reason": "preflight failed"})
            return _staged_payload(pre, current_stage, "STOP", True, False, history,
                                   observations, prior_hash, necessary_condition_stop_evidence)
        if pre["preflight_disposition"] == PREFLIGHT_DEFER:
            current_stage = STAGE_DEFER
            history.append({"stage": STAGE_DEFER, "reason": "preflight incomplete"})
            return _staged_payload(pre, current_stage, "DEFER", False, False, history,
                                   observations, prior_hash, necessary_condition_stop_evidence)
        current_stage = STAGE_SMOKE
        history.append({"stage": STAGE_PREFLIGHT, "reason": "deterministic preflight passed"})
        if stage_observation is None:
            return _staged_payload(pre, current_stage, "IN_PROGRESS", False, False,
                                   history, observations, prior_hash, necessary_condition_stop_evidence)

    phase = current_stage
    if phase not in {STAGE_SMOKE, STAGE_CALIBRATION, STAGE_HOLDOUT_ELIGIBLE}:
        raise ValueError("STAGED_GATE_STATE_NOT_ADVANCEABLE")
    if stage_observation is None:
        return _staged_payload(pre, phase, "IN_PROGRESS", False, heldout_opened,
                               history, observations, prior_hash, necessary_condition_stop_evidence)
    observed = _validate_stage_observation(stage_observation)
    stage_key = "HOLDOUT" if phase == STAGE_HOLDOUT_ELIGIBLE else phase
    if stage_key in observations:
        if not recovering_defer or not history or history[-1].get("stage") != stage_key:
            raise ValueError("STAGED_GATE_STAGE_OBSERVATION_ALREADY_RECORDED")
        old_observation = observations[stage_key]
        new_observation_id = _text(stage_observation.get("observation_id"), "stage_observation_id")
        if old_observation.get("observation_id") == new_observation_id:
            raise ValueError("STAGED_GATE_DEFER_RESUME_REQUIRES_NEW_OBSERVATION")
        history.append({
            "stage": stage_key,
            "reason": "deferred observation superseded by later bound evidence",
            "superseded_observation_id": old_observation["observation_id"],
            "superseded_evidence_hash": old_observation["evidence_hash"],
        })
    observations[stage_key] = observed
    if phase == STAGE_HOLDOUT_ELIGIBLE:
        heldout_opened = True
    gates = [gate for gate in pre["hard_gates"] if gate["phase"] == stage_key]
    missing_metrics = sorted({gate["metric"] for gate in gates if gate["metric"] not in observed["metrics"]})
    failed_gates = [gate["gate_id"] for gate in gates if gate["metric"] in observed["metrics"] and
                    not _eval_gate(gate, observed["metrics"][gate["metric"]])]
    history.append({"stage": stage_key, "observation_id": observed["observation_id"],
                    "status": observed["status"], "evidence_hash": observed["evidence_hash"],
                    "failed_hard_gates": failed_gates, "missing_metrics": missing_metrics})
    if failed_gates or observed["status"] == "FAIL":
        history[-1]["performance_is_diagnostic_only"] = any(
            gate["kind"] == "PERFORMANCE" for gate in pre["hard_gates"]
        )
        return _staged_payload(pre, STAGE_HARD_STOP, "STOP", True, heldout_opened,
                               history, observations, prior_hash, necessary_condition_stop_evidence)
    if missing_metrics or observed["status"] == "DEFER":
        return _staged_payload(pre, STAGE_DEFER, "DEFER", False, heldout_opened,
                               history, observations, prior_hash, necessary_condition_stop_evidence)
    if phase == STAGE_SMOKE:
        next_stage, disposition = STAGE_CALIBRATION, "IN_PROGRESS"
    elif phase == STAGE_CALIBRATION:
        next_stage, disposition = STAGE_HOLDOUT_ELIGIBLE, "HOLDOUT_ELIGIBLE"
    else:
        next_stage, disposition = STAGE_HOLDOUT_COMPLETE, "COMPLETE"
    history.append({"stage": next_stage, "reason": "prior stage gates passed"})
    return _staged_payload(pre, next_stage, disposition, next_stage in {STAGE_HOLDOUT_COMPLETE},
                           heldout_opened, history, observations, prior_hash,
                           necessary_condition_stop_evidence)


def _staged_payload(
    preflight: Mapping[str, Any],
    stage: str,
    disposition: str,
    terminal: bool,
    heldout_opened: bool,
    history: list[dict[str, Any]],
    observations: dict[str, Any],
    prior_hash: str | None,
    necessary_stop: Mapping[str, Any] | None,
) -> dict[str, Any]:
    payload = {
        "schema": STAGED_GATE_SCHEMA,
        "experiment_id": preflight["experiment_id"],
        "preflight_binding_hash": preflight["binding_hash"],
        "hard_gates_hash": preflight["hard_gates_hash"],
        "stage": stage,
        "disposition": disposition,
        "terminal": terminal,
        "heldout_opened": heldout_opened,
        "heldout_preserved_sealed": not heldout_opened,
        "observations": observations,
        "history": history,
        "necessary_condition_stop_binding_hash": necessary_stop.get("binding_hash") if isinstance(necessary_stop, Mapping) else None,
        "previous_binding_hash": prior_hash,
        "claim_ceiling": PREFLIGHT_CLAIM_CEILING,
    }
    payload["binding_hash"] = _hash(payload)
    validate_staged_gate_evidence(payload, preflight=preflight)
    return payload


def validate_staged_gate_evidence(evidence: Any, *, preflight: Mapping[str, Any]) -> dict[str, Any]:
    pre = validate_cohort_preflight(preflight)
    if not isinstance(evidence, Mapping) or evidence.get("schema") != STAGED_GATE_SCHEMA:
        raise ValueError("STAGED_GATE_SCHEMA_INVALID")
    expected_keys = {
        "schema", "experiment_id", "preflight_binding_hash", "hard_gates_hash", "stage", "disposition",
        "terminal", "heldout_opened", "heldout_preserved_sealed", "observations", "history",
        "necessary_condition_stop_binding_hash", "previous_binding_hash", "claim_ceiling", "binding_hash",
    }
    if set(evidence) != expected_keys:
        raise ValueError("STAGED_GATE_SCHEMA_INVALID")
    payload = dict(evidence)
    supplied_hash = payload.pop("binding_hash", None)
    if supplied_hash != _hash(payload):
        raise ValueError("STAGED_GATE_BINDING_HASH_MISMATCH")
    staged_experiment_id = evidence.get("experiment_id")
    if not isinstance(staged_experiment_id, str) or not staged_experiment_id.strip():
        raise ValueError("STAGED_GATE_EXPERIMENT_ID_INVALID")
    if staged_experiment_id != pre["experiment_id"]:
        raise ValueError("STAGED_GATE_EXPERIMENT_ID_MISMATCH")
    if evidence.get("preflight_binding_hash") != pre["binding_hash"] or evidence.get("hard_gates_hash") != pre["hard_gates_hash"]:
        raise ValueError("STAGED_GATE_PREFLIGHT_BINDING_MISMATCH")
    stage = evidence.get("stage")
    if not isinstance(stage, str) or stage not in _STAGES:
        raise ValueError("STAGED_GATE_STAGE_INVALID")
    if not isinstance(evidence.get("heldout_opened"), bool):
        raise ValueError("STAGED_GATE_HELDOUT_STATE_INVALID")
    if not isinstance(evidence.get("terminal"), bool):
        raise ValueError("STAGED_GATE_TERMINAL_STATE_INVALID")
    if evidence.get("heldout_preserved_sealed") is not (not evidence.get("heldout_opened")):
        raise ValueError("STAGED_GATE_HELDOUT_SEALING_INVALID")
    if stage in {STAGE_STOP, STAGE_HARD_STOP} and evidence.get("disposition") != "STOP":
        raise ValueError("STAGED_GATE_STOP_DISPOSITION_INVALID")
    if stage == STAGE_DEFER and evidence.get("disposition") != "DEFER":
        raise ValueError("STAGED_GATE_DEFER_DISPOSITION_INVALID")
    if stage == STAGE_DEFER and evidence.get("terminal") is True:
        raise ValueError("STAGED_GATE_DEFER_CANNOT_BE_TERMINAL")
    if stage in {STAGE_STOP, STAGE_HARD_STOP, STAGE_HOLDOUT_COMPLETE} and evidence.get("terminal") is not True:
        raise ValueError("STAGED_GATE_TERMINAL_STAGE_REQUIRED")
    if evidence.get("terminal") is True and stage not in {STAGE_STOP, STAGE_HARD_STOP, STAGE_HOLDOUT_COMPLETE}:
        raise ValueError("STAGED_GATE_TERMINAL_STAGE_INVALID")
    if pre["preflight_disposition"] == PREFLIGHT_DEFER and stage not in {STAGE_DEFER, STAGE_HARD_STOP}:
        raise ValueError("STAGED_GATE_ADVANCED_FROM_DEFERRED_PREFLIGHT")
    if pre["preflight_disposition"] == PREFLIGHT_FAIL and stage not in {STAGE_STOP, STAGE_HARD_STOP}:
        raise ValueError("STAGED_GATE_ADVANCED_FROM_FAILED_PREFLIGHT")
    if pre["preflight_disposition"] == PREFLIGHT_PASS and stage == STAGE_STOP:
        raise ValueError("STAGED_GATE_STOP_WITHOUT_PREFLIGHT_FAILURE")
    necessary_stop_hash = evidence.get("necessary_condition_stop_binding_hash")
    if necessary_stop_hash is not None and not _is_sha256_digest(necessary_stop_hash):
        raise ValueError("STAGED_GATE_NECESSARY_STOP_BINDING_INVALID")
    if (
        pre["preflight_disposition"] == PREFLIGHT_DEFER
        and stage == STAGE_HARD_STOP
        and not _is_sha256_digest(necessary_stop_hash)
    ):
        raise ValueError("STAGED_GATE_DEFERRED_PREFLIGHT_STOP_REQUIRES_ISSUE_26_PROOF")
    if evidence.get("claim_ceiling") != PREFLIGHT_CLAIM_CEILING:
        raise ValueError("STAGED_GATE_CLAIM_CEILING_INVALID")
    if not isinstance(evidence.get("observations"), Mapping) or not isinstance(evidence.get("history"), list):
        raise ValueError("STAGED_GATE_EVIDENCE_INVALID")
    return dict(evidence)
