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

EVIDENCE_ORIGIN_SCHEMA = "nexus.learning_evidence_origin_provenance.v1"
SIMULATION_ONLY = "SIMULATION_ONLY"
MOCK_TRANSPORT = "MOCK_TRANSPORT"
LOCAL_FAKE_PROVIDER = "LOCAL_FAKE_PROVIDER"
PHYSICAL_LOCAL_MODEL = "PHYSICAL_LOCAL_MODEL"
REMOTE_PROVIDER_OBSERVED = "REMOTE_PROVIDER_OBSERVED"
UNKNOWN_ORIGIN = "UNKNOWN"
_EVIDENCE_ORIGINS = frozenset(
    {
        SIMULATION_ONLY,
        MOCK_TRANSPORT,
        LOCAL_FAKE_PROVIDER,
        PHYSICAL_LOCAL_MODEL,
        REMOTE_PROVIDER_OBSERVED,
        UNKNOWN_ORIGIN,
    }
)

EFFECT_NOT_STARTED = "NOT_STARTED"
EFFECT_SUCCEEDED = "SUCCEEDED"
EFFECT_FAILED = "FAILED"
EFFECT_OUTCOME_UNKNOWN = "OUTCOME_UNKNOWN"
_EFFECT_OUTCOMES = frozenset(
    {EFFECT_NOT_STARTED, EFFECT_SUCCEEDED, EFFECT_FAILED, EFFECT_OUTCOME_UNKNOWN}
)


def _hash(payload: Any) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _optional_text(value: Any, field: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"EVIDENCE_ORIGIN_{field.upper()}_INVALID")
    return value.strip()


def _identity(
    *,
    provider: str | None,
    model: str | None,
    revision: str | None = None,
) -> dict[str, str | None]:
    return {
        "provider": _optional_text(provider, "identity_provider"),
        "model": _optional_text(model, "identity_model"),
        "revision": _optional_text(revision, "identity_revision"),
    }


def _sha256_ref(value: Any) -> bool:
    if not isinstance(value, str) or not value.startswith("sha256:") or len(value) != 71:
        return False
    return all(char in "0123456789abcdef" for char in value[7:])


def _validate_identity(identity: Any, field: str) -> dict[str, Any]:
    if not isinstance(identity, Mapping):
        raise ValueError(f"EVIDENCE_ORIGIN_{field.upper()}_IDENTITY_INVALID")
    if set(identity) != {"provider", "model", "revision"}:
        raise ValueError(f"EVIDENCE_ORIGIN_{field.upper()}_IDENTITY_INVALID")
    return {
        key: _optional_text(identity.get(key), f"{field}_{key}")
        for key in ("provider", "model", "revision")
    }


def build_evidence_origin_provenance(
    *,
    origin_class: str,
    requested_provider: str | None = None,
    requested_model: str | None = None,
    configured_provider: str | None = None,
    configured_model: str | None = None,
    observed_provider: str | None = None,
    observed_model: str | None = None,
    observed_revision: str | None = None,
    external_effect_started: bool = False,
    effect_outcome: str = EFFECT_NOT_STARTED,
    effect_identity: str | None = None,
    operation_id: str | None = None,
    observation_receipt: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build provenance without inferring observed execution from configuration.

    The observation receipt is a public-safe projection of a durable physical
    receipt. Its source_receipt_sha256 binds that projection to the owning
    transport or effect journal. This module validates integrity and
    compatibility, not the external journal's authenticity.
    """
    origin = str(origin_class).strip().upper()
    outcome = str(effect_outcome).strip().upper()
    receipt = None
    if observation_receipt is not None:
        if not isinstance(observation_receipt, Mapping):
            raise ValueError("EVIDENCE_ORIGIN_RECEIPT_INVALID")
        payload = dict(observation_receipt)
        receipt = {"payload": payload, "payload_hash": _hash(payload)}

    provenance_without_hash = {
        "schema": EVIDENCE_ORIGIN_SCHEMA,
        "origin_class": origin,
        "requested_identity": _identity(
            provider=requested_provider,
            model=requested_model,
        ),
        "configured_identity": _identity(
            provider=configured_provider,
            model=configured_model,
        ),
        "observed_identity": _identity(
            provider=observed_provider,
            model=observed_model,
            revision=observed_revision,
        ),
        "external_effect": {
            "started": external_effect_started,
            "outcome": outcome,
            "effect_identity": _optional_text(effect_identity, "effect_identity"),
            "operation_id": _optional_text(operation_id, "operation_id"),
        },
        "observation_receipt": receipt,
        "claim_ceiling": (
            "evidence-origin provenance only; external receipt authenticity and "
            "provider/model success claims require the owning producer gate"
        ),
    }
    provenance = {
        **provenance_without_hash,
        "binding_hash": _hash(provenance_without_hash),
    }
    validate_evidence_origin_provenance(provenance)
    return provenance


def validate_evidence_origin_provenance(provenance: Any) -> None:
    """Fail closed on physical/provider classifications unsupported by receipts."""
    if not isinstance(provenance, Mapping):
        raise ValueError("EVIDENCE_ORIGIN_NOT_A_MAPPING")
    if provenance.get("schema") != EVIDENCE_ORIGIN_SCHEMA:
        raise ValueError("EVIDENCE_ORIGIN_SCHEMA_INVALID")

    origin = str(provenance.get("origin_class") or "").upper()
    if origin not in _EVIDENCE_ORIGINS:
        raise ValueError("EVIDENCE_ORIGIN_CLASS_INVALID")
    _validate_identity(provenance.get("requested_identity"), "requested")
    _validate_identity(provenance.get("configured_identity"), "configured")
    observed = _validate_identity(provenance.get("observed_identity"), "observed")

    effect = provenance.get("external_effect")
    if not isinstance(effect, Mapping):
        raise ValueError("EVIDENCE_ORIGIN_EFFECT_INVALID")
    if set(effect) != {"started", "outcome", "effect_identity", "operation_id"}:
        raise ValueError("EVIDENCE_ORIGIN_EFFECT_INVALID")
    if not isinstance(effect.get("started"), bool):
        raise ValueError("EVIDENCE_ORIGIN_EFFECT_STARTED_INVALID")
    outcome = str(effect.get("outcome") or "").upper()
    if outcome not in _EFFECT_OUTCOMES:
        raise ValueError("EVIDENCE_ORIGIN_EFFECT_OUTCOME_INVALID")
    effect_identity = _optional_text(effect.get("effect_identity"), "effect_identity")
    operation_id = _optional_text(effect.get("operation_id"), "operation_id")

    receipt_wrapper = provenance.get("observation_receipt")
    receipt_payload: Mapping[str, Any] | None = None
    if receipt_wrapper is not None:
        if not isinstance(receipt_wrapper, Mapping) or set(receipt_wrapper) != {
            "payload",
            "payload_hash",
        }:
            raise ValueError("EVIDENCE_ORIGIN_RECEIPT_INVALID")
        payload = receipt_wrapper.get("payload")
        if not isinstance(payload, Mapping):
            raise ValueError("EVIDENCE_ORIGIN_RECEIPT_INVALID")
        if receipt_wrapper.get("payload_hash") != _hash(dict(payload)):
            raise ValueError("EVIDENCE_ORIGIN_RECEIPT_HASH_MISMATCH")
        receipt_payload = payload

    expected_claim_ceiling = (
        "evidence-origin provenance only; external receipt authenticity and "
        "provider/model success claims require the owning producer gate"
    )
    if provenance.get("claim_ceiling") != expected_claim_ceiling:
        raise ValueError("EVIDENCE_ORIGIN_CLAIM_CEILING_TAMPERED")
    binding_payload = dict(provenance)
    supplied_binding_hash = binding_payload.pop("binding_hash", None)
    if supplied_binding_hash != _hash(binding_payload):
        raise ValueError("EVIDENCE_ORIGIN_BINDING_HASH_MISMATCH")

    if origin in {SIMULATION_ONLY, MOCK_TRANSPORT, LOCAL_FAKE_PROVIDER}:
        if effect.get("started") is not False:
            raise ValueError("EVIDENCE_ORIGIN_NONPHYSICAL_EXTERNAL_EFFECT_FORBIDDEN")
        return

    if origin not in {PHYSICAL_LOCAL_MODEL, REMOTE_PROVIDER_OBSERVED}:
        return

    if effect.get("started") is not True:
        raise ValueError("EVIDENCE_ORIGIN_PHYSICAL_EFFECT_NOT_STARTED")
    if not effect_identity or not operation_id:
        raise ValueError("EVIDENCE_ORIGIN_PHYSICAL_EFFECT_IDENTITY_MISSING")
    if receipt_payload is None:
        raise ValueError("EVIDENCE_ORIGIN_PHYSICAL_RECEIPT_MISSING")

    receipt_effect = _optional_text(
        receipt_payload.get("effect_identity"), "receipt_effect_identity"
    )
    receipt_operation = _optional_text(
        receipt_payload.get("operation_id"), "receipt_operation_id"
    )
    if receipt_effect != effect_identity or receipt_operation != operation_id:
        raise ValueError("EVIDENCE_ORIGIN_RECEIPT_EFFECT_IDENTITY_MISMATCH")
    if receipt_payload.get("external_effect_started") is not True:
        raise ValueError("EVIDENCE_ORIGIN_RECEIPT_EFFECT_NOT_STARTED")
    if str(receipt_payload.get("outcome") or "").upper() != outcome:
        raise ValueError("EVIDENCE_ORIGIN_RECEIPT_OUTCOME_MISMATCH")
    if _optional_text(receipt_payload.get("source_receipt_ref"), "source_receipt_ref") is None:
        raise ValueError("EVIDENCE_ORIGIN_SOURCE_RECEIPT_REF_MISSING")
    if not _sha256_ref(receipt_payload.get("source_receipt_sha256")):
        raise ValueError("EVIDENCE_ORIGIN_SOURCE_RECEIPT_HASH_INVALID")

    transport_class = str(receipt_payload.get("transport_class") or "").upper()
    if origin == PHYSICAL_LOCAL_MODEL:
        if transport_class != "LOCAL_MODEL":
            raise ValueError("EVIDENCE_ORIGIN_LOCAL_TRANSPORT_MISMATCH")
        if not observed["model"]:
            raise ValueError("EVIDENCE_ORIGIN_LOCAL_OBSERVED_MODEL_MISSING")
        if (
            _optional_text(receipt_payload.get("observed_model"), "receipt_observed_model")
            != observed["model"]
        ):
            raise ValueError("EVIDENCE_ORIGIN_RECEIPT_MODEL_MISMATCH")
        return

    if transport_class != "REMOTE_PROVIDER":
        raise ValueError("EVIDENCE_ORIGIN_REMOTE_TRANSPORT_MISMATCH")
    if not observed["provider"] or not observed["model"]:
        raise ValueError("EVIDENCE_ORIGIN_REMOTE_OBSERVED_IDENTITY_MISSING")
    if (
        _optional_text(receipt_payload.get("observed_provider"), "receipt_observed_provider")
        != observed["provider"]
    ):
        raise ValueError("EVIDENCE_ORIGIN_RECEIPT_PROVIDER_MISMATCH")
    if (
        _optional_text(receipt_payload.get("observed_model"), "receipt_observed_model")
        != observed["model"]
    ):
        raise ValueError("EVIDENCE_ORIGIN_RECEIPT_MODEL_MISMATCH")
    receipt_revision = _optional_text(
        receipt_payload.get("observed_revision"), "receipt_observed_revision"
    )
    if receipt_revision != observed["revision"]:
        raise ValueError("EVIDENCE_ORIGIN_RECEIPT_REVISION_MISMATCH")


def require_remote_provider_observation(value: Any) -> dict[str, Any]:
    """Return provenance only for a successful physical remote observation."""
    provenance = value
    if isinstance(value, Mapping) and value.get("schema") == EXPERIMENT_INTEGRITY_SCHEMA:
        provenance = value.get("evidence_origin")
    if not isinstance(provenance, Mapping):
        raise ValueError("LIVE_PROVIDER_PROVENANCE_MISSING")
    validate_evidence_origin_provenance(provenance)
    if provenance.get("origin_class") != REMOTE_PROVIDER_OBSERVED:
        raise ValueError("LIVE_PROVIDER_REMOTE_OBSERVATION_REQUIRED")
    effect = provenance["external_effect"]
    if effect.get("outcome") != EFFECT_SUCCEEDED:
        raise ValueError("LIVE_PROVIDER_SUCCESSFUL_OUTCOME_REQUIRED")
    return dict(provenance)


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


def _canonical_members(
    members: list[Mapping[str, Any]] | tuple[Mapping[str, Any], ...],
    independence_unit: str,
) -> tuple[dict[str, Any], ...]:
    """Validate and canonically order the full member records.

    Population identity hashes protect the declared independence unit; the
    full-member hash additionally binds all supplied member content (including
    held-out truth/evidence fields) so those bytes cannot drift independently.
    """
    if not isinstance(members, (list, tuple)):
        raise ValueError("EXPERIMENT_POPULATION_MUST_BE_SEQUENCE")
    canonical: list[dict[str, Any]] = []
    for member in members:
        if not isinstance(member, Mapping):
            raise ValueError("EXPERIMENT_POPULATION_MEMBER_INVALID")
        item = dict(member)
        _member_identity(item, independence_unit)
        _hash(item)
        canonical.append(item)
    if not canonical:
        raise ValueError("EXPERIMENT_EMPTY_POPULATION")
    return tuple(
        sorted(
            canonical,
            key=lambda item: (_member_identity(item, independence_unit), _hash(item)),
        )
    )


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
    evidence_origin: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Bind one experiment's populations, frozen policy, and terminal evidence."""
    if independence_unit not in _INDEPENDENCE_UNITS:
        raise ValueError("EXPERIMENT_INDEPENDENCE_UNIT_INVALID")
    if calibration_status not in _CALIBRATION_STATUSES:
        raise ValueError("EXPERIMENT_CALIBRATION_STATUS_INVALID")
    outcome = str(terminal_outcome).upper()
    if outcome not in _TERMINAL_OUTCOMES:
        raise ValueError("EXPERIMENT_TERMINAL_OUTCOME_INVALID")

    if (
        isinstance(freeze_generation, bool)
        or not isinstance(freeze_generation, int)
        or isinstance(heldout_evaluation_start_generation, bool)
        or not isinstance(heldout_evaluation_start_generation, int)
    ):
        raise ValueError("EXPERIMENT_GENERATION_MUST_BE_INTEGER")
    freeze_gen = freeze_generation
    eval_start_gen = heldout_evaluation_start_generation
    if freeze_gen < 0 or eval_start_gen < 0:
        raise ValueError("EXPERIMENT_GENERATION_NEGATIVE_FORBIDDEN")
    if not freeze_gen < eval_start_gen:
        raise ValueError(
            "EXPERIMENT_POLICY_FROZEN_AFTER_EVALUATION_START "
            f"({freeze_gen} !< {eval_start_gen})"
        )

    calibration = _canonical_members(calibration_members, independence_unit)
    heldout = _canonical_members(heldout_members, independence_unit)
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
            "members_hash": _hash({"unit": independence_unit, "members": calibration}),
            "member_identities": list(cal_ids),
            "member_count": len(cal_ids),
            "members": [dict(item) for item in calibration],
        },
        "heldout": {
            "population_hash": _hash({"identities": heldout_ids, "unit": independence_unit}),
            "members_hash": _hash({"unit": independence_unit, "members": heldout}),
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
    if evidence_origin is not None:
        validate_evidence_origin_provenance(evidence_origin)
        integrity["evidence_origin"] = dict(evidence_origin)
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
    if "evidence_origin" in integrity:
        validate_evidence_origin_provenance(integrity.get("evidence_origin"))

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
    if calibration.get("member_count") != len(cal_ids):
        raise ValueError("EXPERIMENT_CALIBRATION_MEMBER_COUNT_MISMATCH")
    if heldout.get("member_count") != len(heldout_ids):
        raise ValueError("EXPERIMENT_HELDOUT_MEMBER_COUNT_MISMATCH")

    # Recompute overlap and full member-content binding from canonical records.
    try:
        cal_members = _canonical_members(
            list(calibration.get("members", []) or []), independence_unit
        )
        heldout_members = _canonical_members(
            list(heldout.get("members", []) or []), independence_unit
        )
        recomputed_cal = _population_identities(cal_members, independence_unit)
        recomputed_heldout = _population_identities(heldout_members, independence_unit)
    except ValueError as exc:
        raise ValueError(f"EXPERIMENT_POPULATION_MEMBER_INVALID:{exc}") from exc
    expected_cal_members_hash = _hash({"unit": independence_unit, "members": cal_members})
    expected_heldout_members_hash = _hash({"unit": independence_unit, "members": heldout_members})
    if calibration.get("members_hash") != expected_cal_members_hash:
        raise ValueError("EXPERIMENT_CALIBRATION_MEMBERS_HASH_MISMATCH")
    if heldout.get("members_hash") != expected_heldout_members_hash:
        raise ValueError("EXPERIMENT_HELDOUT_MEMBERS_HASH_MISMATCH")
    if list(recomputed_cal) != list(cal_ids) or list(recomputed_heldout) != list(heldout_ids):
        raise ValueError("EXPERIMENT_POPULATION_IDENTITY_MISMATCH")
    shared, has_overlap = _overlap(recomputed_cal, recomputed_heldout, independence_unit)
    if has_overlap:
        raise ValueError(
            f"EXPERIMENT_POPULATION_OVERLAP:{','.join(shared[:5])}"
        )
    overlap_proof = integrity.get("overlap_proof") or {}
    if (
        overlap_proof.get("basis") != "semantic_" + independence_unit
        or overlap_proof.get("overlap") is not False
        or overlap_proof.get("shared_identities")
    ):
        raise ValueError("EXPERIMENT_OVERLAP_PROOF_TAMPERED")

    frozen = integrity.get("frozen_policy") or {}
    raw_freeze_generation = frozen.get("freeze_generation")
    if isinstance(raw_freeze_generation, bool) or not isinstance(raw_freeze_generation, int):
        raise ValueError("EXPERIMENT_FREEZE_GENERATION_INVALID")
    policy_payload = {
        "policy": dict(frozen.get("policy") or {}),
        "policy_derivation_ref": str(frozen.get("policy_derivation_ref") or "").strip(),
        "freeze_generation": raw_freeze_generation,
    }
    expected_policy_hash = _hash(policy_payload)
    if frozen.get("policy_hash") != expected_policy_hash:
        raise ValueError("EXPERIMENT_POLICY_HASH_MISMATCH")

    # Recompute freeze ordering from evidence, never trust an asserted flag.
    order_proof = integrity.get("freeze_order_proof") or {}
    stored_eval_start = order_proof.get("heldout_evaluation_start_generation")
    top_level_eval_start = integrity.get("heldout_evaluation_start_generation")
    derived_freeze_gen = raw_freeze_generation
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
    derivation_ref = str(derivation.get("policy_derivation_ref") or "").strip()
    frozen_derivation_ref = str(frozen.get("policy_derivation_ref") or "").strip()
    if not derivation_ref:
        raise ValueError("EXPERIMENT_POLICY_DERIVATION_REF_MISSING")
    if derivation_ref != frozen_derivation_ref:
        raise ValueError("EXPERIMENT_POLICY_DERIVATION_REF_MISMATCH")

    calibration_status = integrity.get("calibration_status")
    if calibration_status not in _CALIBRATION_STATUSES:
        raise ValueError("EXPERIMENT_CALIBRATION_STATUS_INVALID")
    if order_proof.get("calibration_status") != calibration_status:
        raise ValueError("EXPERIMENT_FREEZE_ORDER_CALIBRATION_STATUS_MISMATCH")
    if calibration_status == CALIBRATED:
        reasons = tuple(integrity.get("insufficient_calibration_reasons", []) or [])
        if reasons:
            raise ValueError("EXPERIMENT_CALIBRATED_WITH_INSUFFICIENT_REASONS")
    if calibration_status in {INSUFFICIENT_CALIBRATION, EXPLORATORY_UNCALIBRATED}:
        reasons = tuple(integrity.get("insufficient_calibration_reasons", []) or [])
        if not reasons:
            raise ValueError("EXPERIMENT_UNCALIBRATED_WITHOUT_REASONS")

    reject_all = integrity.get("reject_all_policy")
    if calibration_status == FROZEN_REJECT_ALL and reject_all is None:
        raise ValueError("EXPERIMENT_REJECT_ALL_POLICY_REQUIRED")
    if reject_all is not None:
        if calibration_status != FROZEN_REJECT_ALL:
            raise ValueError("EXPERIMENT_REJECT_ALL_STATUS_MISMATCH")
        if not isinstance(reject_all, Mapping) or not reject_all.get("target_policy_delta"):
            raise ValueError("EXPERIMENT_REJECT_ALL_POLICY_INVALID")
        if reject_all.get("model_success_claim") is not False:
            raise ValueError("EXPERIMENT_REJECT_ALL_CLAIMS_MODEL_SUCCESS")

    terminal = integrity.get("terminal") or {}
    outcome = str(terminal.get("outcome") or "").upper()
    if outcome not in _TERMINAL_OUTCOMES:
        raise ValueError("EXPERIMENT_TERMINAL_OUTCOME_INVALID")
    if bool(terminal.get("negative_terminal")) != (outcome == TERMINAL_NEGATIVE):
        raise ValueError("EXPERIMENT_TERMINAL_NEGATIVE_FLAG_OUTCOME_MISMATCH")
    if outcome == TERMINAL_PASS and calibration_status != CALIBRATED:
        raise ValueError("EXPERIMENT_PASS_WITHOUT_PREFROZEN_CALIBRATION")