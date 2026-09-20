"""Experiment-integrity evidence for calibration / held-out separation.

This module binds a reusable experiment-integrity invariant that prevents a
decision experiment from tuning operating policy on the same population it
later uses to claim validation:

    freeze evaluation identity/truth
    -> choose an independent calibration population
    -> derive operating policy from calibration only
    -> freeze policy identity/hash
    -> only then inspect held-out evaluation output
    -> preserve PASS / STOP / DEFER / insufficient-calibration result

It is purely observational evidence.  It never routes, never mutates
CapabilityPlanner / Workforce / production state, and never claims any model,
route, capability, or policy is good enough to adopt.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping

EXPERIMENT_INTEGRITY_SCHEMA = "nexus.learning_experiment_integrity.v1"

INDEPENDENCE_UNIT_ROW = "row_identity"
INDEPENDENCE_UNIT_BASE = "base_identity"
_INDEPENDENCE_UNITS = frozenset({INDEPENDENCE_UNIT_ROW, INDEPENDENCE_UNIT_BASE})

CALIBRATED = "CALIBRATED"
INSUFFICIENT_CALIBRATION = "INSUFFICIENT_CALIBRATION"
EXPLORATORY_UNCALIBRATED = "EXPLORATORY_UNCALIBRATED"
FROZEN_REJECT_ALL = "FROZEN_REJECT_ALL"
_CALIBRATION_STATUSES = frozenset(
    {CALIBRATED, INSUFFICIENT_CALIBRATION, EXPLORATORY_UNCALIBRATED, FROZEN_REJECT_ALL}
)

TERMINAL_PASS = "PASS"
TERMINAL_STOP = "STOP"
TERMINAL_DEFER = "DEFER"
TERMINAL_NEGATIVE = "NEGATIVE"
_TERMINAL_OUTCOMES = frozenset({TERMINAL_PASS, TERMINAL_STOP, TERMINAL_DEFER, TERMINAL_NEGATIVE})


def _hash(payload: Any) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _member_identity(member: Mapping[str, Any], independence_unit: str) -> str:
    if independence_unit == INDEPENDENCE_UNIT_BASE:
        base = str(member.get("base_identity") or "").strip()
        if not base:
            raise ValueError("EXPERIMENT_MEMBER_MISSING_BASE_IDENTITY")
        return base
    identity = str(member.get("identity") or "").strip()
    if not identity:
        raise ValueError("EXPERIMENT_MEMBER_MISSING_IDENTITY")
    return identity


def _population_identities(
    members: tuple[Mapping[str, Any], ...], independence_unit: str
) -> tuple[str, ...]:
    identities: tuple[str, ...] = tuple(
        sorted({_member_identity(member, independence_unit) for member in members})
    )
    if not identities:
        raise ValueError("EXPERIMENT_EMPTY_POPULATION")
    return identities


def _overlap(
    calibration: tuple[str, ...], heldout: tuple[str, ...], independence_unit: str
) -> tuple[list[str], bool]:
    shared = sorted(set(calibration) & set(heldout))
    if shared:
        return shared, True
    return [], False


def build_experiment_integrity(
    *,
    experiment_id: str,
    calibration_members: list[Mapping[str, Any]] | tuple[Mapping[str, Any], ...],
    heldout_members: list[Mapping[str, Any]] | tuple[Mapping[str, Any], ...],
    independence_unit: str,
    policy_derivation_ref: str,
    frozen_policy: Mapping[str, Any],
    freeze_generation: int,
    heldout_evaluation_start_generation: int,
    calibration_status: str = CALIBRATED,
    terminal_outcome: str = TERMINAL_PASS,
    insufficient_calibration_reasons: list[str] | tuple[str, ...] = (),
    reject_all_policy: Mapping[str, Any] | None = None,
    negative_terminal: bool = False,
) -> dict[str, Any]:
    """Bind one experiment's populations, frozen policy, and terminal evidence."""
    if independence_unit not in _INDEPENDENCE_UNITS:
        raise ValueError("EXPERIMENT_INDEPENDENCE_UNIT_INVALID")
    if calibration_status not in _CALIBRATION_STATUSES:
        raise ValueError("EXPERIMENT_CALIBRATION_STATUS_INVALID")
    outcome = str(terminal_outcome).upper()
    if outcome not in _TERMINAL_OUTCOMES:
        raise ValueError("EXPERIMENT_TERMINAL_OUTCOME_INVALID")

    freeze_gen = int(freeze_generation)
    eval_start_gen = int(heldout_evaluation_start_generation)
    if freeze_gen < 0 or eval_start_gen < 0:
        raise ValueError("EXPERIMENT_GENERATION_NEGATIVE_FORBIDDEN")
    if not freeze_gen < eval_start_gen:
        raise ValueError(
            "EXPERIMENT_POLICY_FROZEN_AFTER_EVALUATION_START "
            f"({freeze_gen} !< {eval_start_gen})"
        )

    calibration = tuple(
        dict(member) for member in calibration_members if isinstance(member, Mapping)
    )
    heldout = tuple(dict(member) for member in heldout_members if isinstance(member, Mapping))
    cal_ids = _population_identities(calibration, independence_unit)
    heldout_ids = _population_identities(heldout, independence_unit)

    shared, has_overlap = _overlap(cal_ids, heldout_ids, independence_unit)
    if has_overlap:
        raise ValueError(
            f"EXPERIMENT_POPULATION_OVERLAP:{independence_unit}:{','.join(shared[:5])}"
        )

    policy_payload = {
        "policy": dict(frozen_policy),
        "policy_derivation_ref": str(policy_derivation_ref).strip(),
        "freeze_generation": int(freeze_generation),
    }
    policy_hash = _hash(policy_payload)

    integrity = {
        "schema": EXPERIMENT_INTEGRITY_SCHEMA,
        "experiment_id": str(experiment_id).strip(),
        "independence_unit": independence_unit,
        "calibration": {
            "population_hash": _hash({"identities": cal_ids, "unit": independence_unit}),
            "member_identities": list(cal_ids),
            "member_count": len(cal_ids),
            "members": [dict(item) for item in calibration],
        },
        "heldout": {
            "population_hash": _hash({"identities": heldout_ids, "unit": independence_unit}),
            "member_identities": list(heldout_ids),
            "member_count": len(heldout_ids),
            "members": [dict(item) for item in heldout],
        },
        "overlap_proof": {
            "basis": "semantic_" + independence_unit,
            "overlap": False,
            "shared_identities": [],
        },
        "policy_derivation": {
            "source": "calibration_only",
            "policy_derivation_ref": str(policy_derivation_ref).strip(),
        },
        "frozen_policy": {
            "policy": dict(frozen_policy),
            "policy_hash": policy_hash,
            "policy_derivation_ref": str(policy_derivation_ref).strip(),
            "freeze_generation": int(freeze_generation),
        },
        "freeze_order_proof": {
            "policy_frozen_before_heldout_evaluation": True,
            "freeze_generation": int(freeze_generation),
            "heldout_evaluation_start_generation": int(heldout_evaluation_start_generation),
            "calibration_status": calibration_status,
        },
        "calibration_status": calibration_status,
        "insufficient_calibration_reasons": sorted(
            {str(reason) for reason in insufficient_calibration_reasons if str(reason).strip()}
        ),
        "terminal": {
            "outcome": outcome,
            "negative_terminal": bool(negative_terminal) or outcome == TERMINAL_NEGATIVE,
        },
        "reject_all_policy": dict(reject_all_policy) if reject_all_policy is not None else None,
        "claim_ceiling": "experiment-integrity evidence only; no model/route/production claim",
    }
    validate_experiment_integrity(integrity)
    return integrity


def validate_experiment_integrity(integrity: Any) -> None:
    """Fail closed on population overlap, order/semantic tampering, or unsafe calibration."""
    if not isinstance(integrity, Mapping):
        raise ValueError("EXPERIMENT_INTEGRITY_NOT_A_MAPPING")
    if integrity.get("schema") != EXPERIMENT_INTEGRITY_SCHEMA:
        raise ValueError("EXPERIMENT_INTEGRITY_SCHEMA_INVALID")
    if not str(integrity.get("experiment_id") or "").strip():
        raise ValueError("EXPERIMENT_IDENTITY_MISSING")

    independence_unit = integrity.get("independence_unit")
    if independence_unit not in _INDEPENDENCE_UNITS:
        raise ValueError("EXPERIMENT_INDEPENDENCE_UNIT_INVALID")

    calibration = integrity.get("calibration") or {}
    heldout = integrity.get("heldout") or {}
    cal_ids = tuple(str(item) for item in calibration.get("member_identities", []) or [])
    heldout_ids = tuple(str(item) for item in heldout.get("member_identities", []) or [])
    if not cal_ids or not heldout_ids:
        raise ValueError("EXPERIMENT_EMPTY_POPULATION")

    expected_cal_hash = _hash({"identities": cal_ids, "unit": independence_unit})
    expected_heldout_hash = _hash({"identities": heldout_ids, "unit": independence_unit})
    if calibration.get("population_hash") != expected_cal_hash:
        raise ValueError("EXPERIMENT_CALIBRATION_POPULATION_HASH_MISMATCH")
    if heldout.get("population_hash") != expected_heldout_hash:
        raise ValueError("EXPERIMENT_HELDOUT_POPULATION_HASH_MISMATCH")

    # Recompute overlap from canonical member identities at the declared unit.
    cal_members = tuple(
        dict(item) for item in calibration.get("members", []) or [] if isinstance(item, Mapping)
    )
    heldout_members = tuple(
        dict(item) for item in heldout.get("members", []) or [] if isinstance(item, Mapping)
    )
    try:
        recomputed_cal = _population_identities(cal_members, independence_unit)
        recomputed_heldout = _population_identities(heldout_members, independence_unit)
    except ValueError as exc:
        raise ValueError(f"EXPERIMENT_POPULATION_MEMBER_INVALID:{exc}") from exc
    if list(recomputed_cal) != list(cal_ids) or list(recomputed_heldout) != list(heldout_ids):
        raise ValueError("EXPERIMENT_POPULATION_IDENTITY_MISMATCH")
    shared, has_overlap = _overlap(recomputed_cal, recomputed_heldout, independence_unit)
    if has_overlap:
        raise ValueError(
            f"EXPERIMENT_POPULATION_OVERLAP:{','.join(shared[:5])}"
        )
    overlap_proof = integrity.get("overlap_proof") or {}
    if overlap_proof.get("overlap") is not False or overlap_proof.get("shared_identities"):
        raise ValueError("EXPERIMENT_OVERLAP_PROOF_TAMPERED")

    frozen = integrity.get("frozen_policy") or {}
    policy_payload = {
        "policy": dict(frozen.get("policy") or {}),
        "policy_derivation_ref": str(frozen.get("policy_derivation_ref") or "").strip(),
        "freeze_generation": int(frozen.get("freeze_generation") or -1),
    }
    expected_policy_hash = _hash(policy_payload)
    if frozen.get("policy_hash") != expected_policy_hash:
        raise ValueError("EXPERIMENT_POLICY_HASH_MISMATCH")

    # Recompute freeze ordering from evidence, never trust an asserted flag.
    order_proof = integrity.get("freeze_order_proof") or {}
    stored_eval_start = order_proof.get("heldout_evaluation_start_generation")
    top_level_eval_start = integrity.get("heldout_evaluation_start_generation")
    derived_freeze_gen = int(frozen.get("freeze_generation") or -1)
    derived_eval_start = (
        int(top_level_eval_start)
        if isinstance(top_level_eval_start, int)
        else int(stored_eval_start)
        if isinstance(stored_eval_start, int)
        else -1
    )
    if derived_freeze_gen < 0 or derived_eval_start < 0:
        raise ValueError("EXPERIMENT_FREEZE_ORDER_INCOMPLETE")
    if not derived_freeze_gen < derived_eval_start:
        raise ValueError("EXPERIMENT_POLICY_FROZEN_AFTER_EVALUATION_START")
    if order_proof.get("policy_frozen_before_heldout_evaluation") is not True:
        raise ValueError("EXPERIMENT_FREEZE_ORDER_PROOF_TAMPERED")
    if (
        order_proof.get("freeze_generation") != derived_freeze_gen
        or order_proof.get("heldout_evaluation_start_generation") != derived_eval_start
    ):
        raise ValueError("EXPERIMENT_FREEZE_ORDER_PROOF_TAMPERED")

    derivation = integrity.get("policy_derivation") or {}
    if derivation.get("source") != "calibration_only":
        raise ValueError("EXPERIMENT_POLICY_SOURCE_NOT_CALIBRATION")
    if not str(derivation.get("policy_derivation_ref") or "").strip():
        raise ValueError("EXPERIMENT_POLICY_DERIVATION_REF_MISSING")

    calibration_status = integrity.get("calibration_status")
    if calibration_status not in _CALIBRATION_STATUSES:
        raise ValueError("EXPERIMENT_CALIBRATION_STATUS_INVALID")
    if calibration_status == CALIBRATED:
        reasons = tuple(integrity.get("insufficient_calibration_reasons", []) or [])
        if reasons:
            raise ValueError("EXPERIMENT_CALIBRATED_WITH_INSUFFICIENT_REASONS")
    if calibration_status in {INSUFFICIENT_CALIBRATION, EXPLORATORY_UNCALIBRATED}:
        reasons = tuple(integrity.get("insufficient_calibration_reasons", []) or [])
        if not reasons:
            raise ValueError("EXPERIMENT_UNCALIBRATED_WITHOUT_REASONS")

    reject_all = integrity.get("reject_all_policy")
    if reject_all is not None:
        if not isinstance(reject_all, Mapping) or not reject_all.get("target_policy_delta"):
            raise ValueError("EXPERIMENT_REJECT_ALL_POLICY_INVALID")
        if reject_all.get("model_success_claim") is not False:
            raise ValueError("EXPERIMENT_REJECT_ALL_CLAIMS_MODEL_SUCCESS")

    terminal = integrity.get("terminal") or {}
    outcome = str(terminal.get("outcome") or "").upper()
    if outcome not in _TERMINAL_OUTCOMES:
        raise ValueError("EXPERIMENT_TERMINAL_OUTCOME_INVALID")
    if bool(terminal.get("negative_terminal")) and outcome != TERMINAL_NEGATIVE:
        raise ValueError("EXPERIMENT_TERMINAL_NEGATIVE_FLAG_OUTCOME_MISMATCH")
    if outcome == TERMINAL_PASS and calibration_status != CALIBRATED:
        raise ValueError("EXPERIMENT_PASS_WITHOUT_PREFROZEN_CALIBRATION")