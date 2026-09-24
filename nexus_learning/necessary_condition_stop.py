"""Negative early-stop evidence for pre-registered necessary conditions.

This module is intentionally observational.  It consumes a validated
`nexus.learning_experiment_integrity.v1` artifact, requires the necessary
condition to already exist inside the frozen policy, and proves only whether
that one condition is mathematically non-rescuable.

It does not replace experiment integrity, quality-qualified economics,
verification, adoption, route/model selection, or production authority.
"""

from __future__ import annotations

import hashlib
import json
from math import isfinite
from typing import Any, Mapping

from .experiment_integrity import validate_experiment_integrity

NECESSARY_CONDITION_STOP_SCHEMA = (
    "nexus.learning_negative_necessary_condition_stop.v1"
)
NECESSARY_CONDITION_STOP_CLAIM_CEILING = (
    "LEARNING_NEGATIVE_NECESSARY_CONDITION_STOP_EVIDENCE_ONLY"
)

COMPARATOR_GTE = "GTE"
COMPARATOR_LTE = "LTE"
_COMPARATORS = frozenset({COMPARATOR_GTE, COMPARATOR_LTE})

BOUND_EXACT = "EXACT"
BOUND_UPPER = "UPPER_BOUND"
BOUND_LOWER = "LOWER_BOUND"
BOUND_UNKNOWN = "UNKNOWN"
_BOUND_KINDS = frozenset({BOUND_EXACT, BOUND_UPPER, BOUND_LOWER, BOUND_UNKNOWN})

CONTINUE = "CONTINUE"
NEGATIVE_STOP = "NEGATIVE_STOP"


class NecessaryConditionStopError(ValueError):
    """Raised when a negative-stop proof cannot be safely established."""


def _hash(payload: Any) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise NecessaryConditionStopError(f"{field} must be a non-empty string")
    return value.strip()


def _number(value: Any, field: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not isfinite(value)
    ):
        raise NecessaryConditionStopError(f"{field} must be a finite number")
    return float(value)


def _strings(value: Any, field: str) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        raise NecessaryConditionStopError(f"{field} must be a list or tuple")
    normalized = tuple(sorted(str(item).strip() for item in value if str(item).strip()))
    if len(normalized) != len(set(normalized)):
        raise NecessaryConditionStopError(f"{field} must not contain duplicates")
    return normalized


def _condition_from_frozen_policy(
    integrity: Mapping[str, Any],
    condition_id: str,
) -> dict[str, Any]:
    validate_experiment_integrity(integrity)
    frozen = integrity.get("frozen_policy") or {}
    policy = frozen.get("policy") or {}
    if not isinstance(policy, Mapping):
        raise NecessaryConditionStopError("frozen policy must be a mapping")
    raw_conditions = policy.get("necessary_conditions")
    if not isinstance(raw_conditions, (list, tuple)):
        raise NecessaryConditionStopError(
            "frozen policy must pre-register necessary_conditions"
        )

    target = _text(condition_id, "necessary_condition_id")
    matches: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw in raw_conditions:
        if not isinstance(raw, Mapping):
            raise NecessaryConditionStopError(
                "necessary condition declarations must be mappings"
            )
        cid = _text(raw.get("condition_id"), "condition_id")
        if cid in seen:
            raise NecessaryConditionStopError(
                f"duplicate necessary condition id: {cid}"
            )
        seen.add(cid)
        comparator = _text(raw.get("comparator"), "comparator").upper()
        if comparator not in _COMPARATORS:
            raise NecessaryConditionStopError(
                f"unsupported necessary condition comparator: {comparator}"
            )
        threshold = _number(raw.get("threshold"), "threshold")
        metric = _text(raw.get("metric"), "metric")
        normalized = dict(raw)
        normalized["condition_id"] = cid
        normalized["comparator"] = comparator
        normalized["threshold"] = threshold
        normalized["metric"] = metric
        if cid == target:
            matches.append(normalized)

    if len(matches) != 1:
        raise NecessaryConditionStopError(
            f"necessary condition must be uniquely pre-registered: {target}"
        )
    return matches[0]


def _non_rescuable(
    *,
    comparator: str,
    threshold: float,
    observed: float,
    bound_kind: str,
    remaining_unknowns: tuple[str, ...],
) -> bool:
    """Return true only when remaining measurements cannot rescue the condition."""
    if comparator == COMPARATOR_GTE:
        if bound_kind == BOUND_UPPER:
            return observed < threshold
        if bound_kind == BOUND_EXACT and not remaining_unknowns:
            return observed < threshold
        return False
    if comparator == COMPARATOR_LTE:
        if bound_kind == BOUND_LOWER:
            return observed > threshold
        if bound_kind == BOUND_EXACT and not remaining_unknowns:
            return observed > threshold
        return False
    return False


def build_necessary_condition_stop_evidence(
    *,
    experiment_integrity: Mapping[str, Any],
    necessary_condition_id: str,
    observed_value_or_bound: float | int | None,
    observed_bound_kind: str,
    remaining_unknowns: list[str] | tuple[str, ...] = (),
    missing_evidence_reasons: list[str] | tuple[str, ...] = (),
    heldout_opened: bool = False,
) -> dict[str, Any]:
    """Build a fail-closed proof for CONTINUE versus NEGATIVE_STOP.

    `EXACT` can only prove a terminal negative result when no remaining
    unknown metric can still alter the condition.  With unknown work remaining,
    early stop requires a mathematically final `UPPER_BOUND` for a `GTE`
    condition or `LOWER_BOUND` for an `LTE` condition.
    """
    if not isinstance(experiment_integrity, Mapping):
        raise NecessaryConditionStopError("experiment_integrity must be a mapping")
    validate_experiment_integrity(experiment_integrity)
    condition = _condition_from_frozen_policy(
        experiment_integrity, necessary_condition_id
    )
    unknowns = _strings(remaining_unknowns, "remaining_unknowns")
    missing = _strings(missing_evidence_reasons, "missing_evidence_reasons")
    if not isinstance(heldout_opened, bool):
        raise NecessaryConditionStopError("heldout_opened must be a bool")

    bound_kind = _text(observed_bound_kind, "observed_bound_kind").upper()
    if bound_kind not in _BOUND_KINDS:
        raise NecessaryConditionStopError(
            f"unsupported observed bound kind: {bound_kind}"
        )

    observed: float | None
    if observed_value_or_bound is None:
        observed = None
        if bound_kind != BOUND_UNKNOWN:
            raise NecessaryConditionStopError(
                "missing observed value requires UNKNOWN bound kind"
            )
        if not missing:
            raise NecessaryConditionStopError(
                "missing observed value requires explicit missing_evidence_reasons"
            )
        non_rescuable = False
    else:
        observed = _number(observed_value_or_bound, "observed_value_or_bound")
        if bound_kind == BOUND_UNKNOWN:
            raise NecessaryConditionStopError(
                "known observed value cannot use UNKNOWN bound kind"
            )
        non_rescuable = _non_rescuable(
            comparator=condition["comparator"],
            threshold=condition["threshold"],
            observed=observed,
            bound_kind=bound_kind,
            remaining_unknowns=unknowns,
        )

    frozen = experiment_integrity["frozen_policy"]
    condition_hash = _hash(condition)
    unsigned = {
        "schema": NECESSARY_CONDITION_STOP_SCHEMA,
        "experiment_id": experiment_integrity["experiment_id"],
        "frozen_policy_hash": frozen["policy_hash"],
        "freeze_generation": frozen["freeze_generation"],
        "necessary_condition": {
            "condition_id": condition["condition_id"],
            "metric": condition["metric"],
            "comparator": condition["comparator"],
            "threshold": condition["threshold"],
            "condition_hash": condition_hash,
            "pre_registered_in_frozen_policy": True,
        },
        "observation": {
            "bound_kind": bound_kind,
            "observed_value_or_bound": observed,
            "missing_evidence_reasons": list(missing),
        },
        "remaining_unknowns": list(unknowns),
        "rescuable": not non_rescuable,
        "remaining_unknowns_cannot_rescue_condition": non_rescuable,
        "terminal_disposition": NEGATIVE_STOP if non_rescuable else CONTINUE,
        "heldout_opened": heldout_opened,
        "heldout_preserved_sealed": bool(non_rescuable and not heldout_opened),
        "quality_qualified_claim": False,
        "adoption_recommendation": None,
        "observational_only": True,
        "authority_effect": False,
        "claim_ceiling": NECESSARY_CONDITION_STOP_CLAIM_CEILING,
    }
    evidence = {**unsigned, "binding_hash": _hash(unsigned)}
    validate_necessary_condition_stop_evidence(
        evidence,
        experiment_integrity=experiment_integrity,
    )
    return evidence


def validate_necessary_condition_stop_evidence(
    evidence: Any,
    *,
    experiment_integrity: Mapping[str, Any],
) -> dict[str, Any]:
    """Recompute the stop proof against the exact frozen experiment policy."""
    if not isinstance(evidence, Mapping):
        raise NecessaryConditionStopError("stop evidence must be a mapping")
    if evidence.get("schema") != NECESSARY_CONDITION_STOP_SCHEMA:
        raise NecessaryConditionStopError("stop evidence schema is invalid")
    if evidence.get("claim_ceiling") != NECESSARY_CONDITION_STOP_CLAIM_CEILING:
        raise NecessaryConditionStopError("stop evidence claim ceiling is invalid")
    if not isinstance(experiment_integrity, Mapping):
        raise NecessaryConditionStopError("experiment_integrity must be a mapping")
    validate_experiment_integrity(experiment_integrity)

    if evidence.get("experiment_id") != experiment_integrity.get("experiment_id"):
        raise NecessaryConditionStopError("experiment identity mismatch")
    frozen = experiment_integrity.get("frozen_policy") or {}
    if evidence.get("frozen_policy_hash") != frozen.get("policy_hash"):
        raise NecessaryConditionStopError("frozen policy hash mismatch")
    if evidence.get("freeze_generation") != frozen.get("freeze_generation"):
        raise NecessaryConditionStopError("freeze generation mismatch")

    condition_payload = evidence.get("necessary_condition")
    if not isinstance(condition_payload, Mapping):
        raise NecessaryConditionStopError("necessary condition evidence is invalid")
    condition = _condition_from_frozen_policy(
        experiment_integrity,
        _text(condition_payload.get("condition_id"), "condition_id"),
    )
    if condition_payload.get("condition_hash") != _hash(condition):
        raise NecessaryConditionStopError("necessary condition hash mismatch")
    if condition_payload.get("metric") != condition["metric"]:
        raise NecessaryConditionStopError("necessary condition metric mismatch")
    if condition_payload.get("comparator") != condition["comparator"]:
        raise NecessaryConditionStopError("necessary condition comparator mismatch")
    if condition_payload.get("threshold") != condition["threshold"]:
        raise NecessaryConditionStopError("necessary condition threshold mismatch")
    if condition_payload.get("pre_registered_in_frozen_policy") is not True:
        raise NecessaryConditionStopError(
            "necessary condition pre-registration proof is invalid"
        )

    observation = evidence.get("observation")
    if not isinstance(observation, Mapping):
        raise NecessaryConditionStopError("observation evidence is invalid")
    bound_kind = _text(observation.get("bound_kind"), "bound_kind").upper()
    if bound_kind not in _BOUND_KINDS:
        raise NecessaryConditionStopError("observation bound kind is invalid")
    missing = _strings(
        observation.get("missing_evidence_reasons") or (),
        "missing_evidence_reasons",
    )
    unknowns = _strings(evidence.get("remaining_unknowns") or (), "remaining_unknowns")
    observed_raw = observation.get("observed_value_or_bound")

    if observed_raw is None:
        if bound_kind != BOUND_UNKNOWN or not missing:
            raise NecessaryConditionStopError(
                "missing observation must remain explicit and non-terminal"
            )
        non_rescuable = False
    else:
        observed = _number(observed_raw, "observed_value_or_bound")
        if bound_kind == BOUND_UNKNOWN:
            raise NecessaryConditionStopError(
                "known observation cannot use UNKNOWN bound kind"
            )
        non_rescuable = _non_rescuable(
            comparator=condition["comparator"],
            threshold=condition["threshold"],
            observed=observed,
            bound_kind=bound_kind,
            remaining_unknowns=unknowns,
        )

    expected_disposition = NEGATIVE_STOP if non_rescuable else CONTINUE
    if evidence.get("rescuable") is not (not non_rescuable):
        raise NecessaryConditionStopError("rescuable flag mismatch")
    if evidence.get("remaining_unknowns_cannot_rescue_condition") is not non_rescuable:
        raise NecessaryConditionStopError("remaining-unknown rescue proof mismatch")
    if evidence.get("terminal_disposition") != expected_disposition:
        raise NecessaryConditionStopError("terminal disposition mismatch")

    heldout_opened = evidence.get("heldout_opened")
    if not isinstance(heldout_opened, bool):
        raise NecessaryConditionStopError("heldout_opened must be a bool")
    if evidence.get("heldout_preserved_sealed") is not bool(
        non_rescuable and not heldout_opened
    ):
        raise NecessaryConditionStopError("heldout sealing projection mismatch")

    if evidence.get("quality_qualified_claim") is not False:
        raise NecessaryConditionStopError(
            "negative stop cannot claim QUALITY_QUALIFIED"
        )
    if evidence.get("adoption_recommendation") is not None:
        raise NecessaryConditionStopError(
            "negative stop cannot emit an adoption recommendation"
        )
    if evidence.get("observational_only") is not True:
        raise NecessaryConditionStopError("stop evidence must remain observational")
    if evidence.get("authority_effect") is not False:
        raise NecessaryConditionStopError("stop evidence cannot carry authority")

    unsigned = dict(evidence)
    binding_hash = unsigned.pop("binding_hash", None)
    if binding_hash != _hash(unsigned):
        raise NecessaryConditionStopError("stop evidence binding hash mismatch")

    return {
        "schema": NECESSARY_CONDITION_STOP_SCHEMA,
        "experiment_id": evidence["experiment_id"],
        "terminal_disposition": expected_disposition,
        "rescuable": not non_rescuable,
        "heldout_preserved_sealed": bool(non_rescuable and not heldout_opened),
        "claim_ceiling": NECESSARY_CONDITION_STOP_CLAIM_CEILING,
        "binding_hash": binding_hash,
    }
