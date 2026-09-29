"""Campaign-level closeout and evidence-bound reopen projection.

This is a small immutable index over run/preflight evidence.  It is not an
experiment registry, benchmark runner, model ranker, or runtime admission gate.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from typing import Any, Literal, Mapping, Sequence, overload

from .experiment_preflight import (
    PREFLIGHT_DEFER,
    PREFLIGHT_FAIL,
    PREFLIGHT_PASS,
    STAGE_CALIBRATION,
    STAGE_DEFER,
    STAGE_HARD_STOP,
    STAGE_HOLDOUT_COMPLETE,
    STAGE_HOLDOUT_ELIGIBLE,
    STAGE_PREFLIGHT,
    STAGE_SMOKE,
    STAGE_STOP,
    validate_cohort_preflight,
    validate_comparator_preflight,
    validate_staged_gate_evidence,
)
from .experiment_run_identity import (
    COMPLETE,
    FAILED,
    NOT_STARTED,
    OUTCOME_UNKNOWN,
    RUNNING,
    validate_experiment_run_identity,
)

CAMPAIGN_CLOSEOUT_SCHEMA = "nexus.learning_calibration_campaign_closeout.v1"
CAMPAIGN_REOPEN_SCHEMA = "nexus.learning_calibration_campaign_reopen_evidence.v1"
CAMPAIGN_CLOSEOUT_CLAIM_CEILING = "LEARNING_CALIBRATION_CAMPAIGN_CLOSEOUT_AND_REOPEN_EVIDENCE_ONLY"

CAMPAIGN_COMPLETE = "COMPLETE"
CAMPAIGN_INCOMPLETE = "INCOMPLETE"
NO_GATE_DEFINED = "NONE_DEFINED"
MATERIAL_DELTA = "MATERIAL_DELTA"
OWNER_DECISION = "OWNER_DECISION"


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
        raise ValueError(f"CAMPAIGN_{field.upper()}_INVALID")
    return value.strip()


def _generation(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"CAMPAIGN_{field.upper()}_INVALID")
    return value


def _timestamp(value: Any) -> str:
    text = _text(value, "closed_at")
    normalized = text[:-1] + "+00:00" if text.endswith("Z") else text
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError as exc:
        raise ValueError("CAMPAIGN_CLOSED_AT_INVALID") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("CAMPAIGN_CLOSED_AT_TIMEZONE_REQUIRED")
    return text


def _normalize_trigger(raw: Mapping[str, Any]) -> dict[str, Any]:
    trigger_id = _text(raw.get("trigger_id"), "trigger_id")
    experiment_id = _text(raw.get("experiment_id"), "trigger_experiment_id")
    trigger_type = _text(raw.get("trigger_type"), "trigger_type")
    if trigger_type == MATERIAL_DELTA:
        path = _text(raw.get("field_path"), "trigger_field_path")
        if path.startswith(".") or path.endswith(".") or any(not part for part in path.split(".")):
            raise ValueError("CAMPAIGN_REOPEN_TRIGGER_FIELD_PATH_INVALID")
        expected = raw.get("expected_value")
        rationale = _text(raw.get("materiality_rationale"), "trigger_materiality_rationale")
        if expected is None:
            raise ValueError("CAMPAIGN_REOPEN_TRIGGER_EXPECTED_VALUE_REQUIRED")
        return {"trigger_id": trigger_id, "experiment_id": experiment_id,
                "trigger_type": trigger_type, "field_path": path,
                "expected_value": expected, "materiality_rationale": rationale}
    if trigger_type == OWNER_DECISION:
        decision_ref = _text(raw.get("owner_decision_ref"), "owner_decision_ref")
        decision_hash = raw.get("owner_decision_sha256")
        rationale = _text(raw.get("materiality_rationale"), "trigger_materiality_rationale")
        if not _is_hash(decision_hash):
            raise ValueError("CAMPAIGN_OWNER_DECISION_HASH_INVALID")
        return {"trigger_id": trigger_id, "experiment_id": experiment_id,
                "trigger_type": trigger_type, "owner_decision_ref": decision_ref,
                "owner_decision_sha256": decision_hash, "materiality_rationale": rationale}
    raise ValueError("CAMPAIGN_REOPEN_TRIGGER_TYPE_INVALID")


def _read_field_path(value: Any, field_path: str) -> Any:
    current = value
    for part in field_path.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return _MISSING
        current = current[part]
    return current


_MISSING = object()


def _project_experiment_ref(raw: Mapping[str, Any]) -> tuple[dict[str, Any], list[str]]:
    if not isinstance(raw, Mapping):
        raise ValueError("CAMPAIGN_EXPERIMENT_REF_INVALID")
    ref_id = _text(raw.get("experiment_ref"), "experiment_ref")
    missing: list[str] = []
    result: dict[str, Any] = {"experiment_ref": ref_id}
    preflight_deferred_terminal_stop = False
    run_raw = raw.get("run_evidence")
    if run_raw is None:
        missing.append(f"{ref_id}:run_identity_evidence_missing")
        run = None
    else:
        run = validate_experiment_run_identity(run_raw)
        result.update({
            "experiment_id": run["experiment_id"],
            "experiment_generation": run["experiment_generation"],
            "run_state": run["observed_state"],
            "run_binding_hash": run["binding_hash"],
        })
        if run["observed_state"] not in {COMPLETE, FAILED}:
            missing.append(f"{ref_id}:run_not_terminal")
        artifact = run.get("result_artifact")
        if artifact is None:
            missing.append(f"{ref_id}:result_receipt_missing")
        else:
            result["result_receipt_identity"] = artifact["artifact_id"]
            result["result_receipt_sha256"] = artifact["sha256"]

    preflight_raw = raw.get("preflight_evidence")
    stage_raw = raw.get("staged_gate_evidence")
    if preflight_raw is None:
        missing.append(f"{ref_id}:preflight_evidence_missing")
        preflight = None
        comparator = None
    else:
        preflight = validate_cohort_preflight(preflight_raw)
        result["preflight_binding_hash"] = preflight["binding_hash"]
        comparator_raw = preflight.get("comparator_preflight")
        comparator = validate_comparator_preflight(comparator_raw) if isinstance(comparator_raw, Mapping) else None
    if stage_raw is None:
        missing.append(f"{ref_id}:staged_gate_evidence_missing")
        stage = None
    elif preflight is None:
        missing.append(f"{ref_id}:staged_gate_cannot_bind_without_preflight")
        stage = None
    else:
        stage = validate_staged_gate_evidence(stage_raw, preflight=preflight)
        result["staged_gate_binding_hash"] = stage["binding_hash"]
        result["staged_gate_stage"] = stage["stage"]
        result["staged_gate_terminal"] = stage["terminal"]
        result["necessary_condition_stop_binding_hash"] = stage.get(
            "necessary_condition_stop_binding_hash"
        )
        if stage["terminal"] is not True or stage["stage"] == STAGE_DEFER or stage["stage"] not in {
            STAGE_STOP, STAGE_HARD_STOP, STAGE_HOLDOUT_COMPLETE
        }:
            missing.append(f"{ref_id}:staged_gate_not_terminal")
    if preflight is not None:
        result["preflight_disposition"] = preflight["preflight_disposition"]
        if preflight["preflight_disposition"] == PREFLIGHT_DEFER:
            preflight_deferred_terminal_stop = bool(
                stage is not None
                and stage["stage"] == STAGE_HARD_STOP
                and _is_sha256_digest(stage.get("necessary_condition_stop_binding_hash"))
            )
            if not preflight_deferred_terminal_stop:
                missing.append(f"{ref_id}:preflight_deferred")
    if run is not None and preflight is not None and run["experiment_id"] != preflight["experiment_id"]:
        raise ValueError("CAMPAIGN_RUN_PREFLIGHT_EXPERIMENT_MISMATCH")
    if run is not None and preflight is not None:
        if run["cohort_hash"] != preflight["cohort"]["cohort_hash"]:
            raise ValueError("CAMPAIGN_RUN_PREFLIGHT_COHORT_MISMATCH")
        frozen_hash = preflight.get("frozen_policy_hash")
        if frozen_hash is None:
            missing.append(f"{ref_id}:issue_20_frozen_policy_binding_missing")
        elif run["frozen_policy_hash"] != frozen_hash:
            raise ValueError("CAMPAIGN_RUN_PREFLIGHT_POLICY_MISMATCH")
    if run is not None and stage is not None and run["experiment_id"] != stage["experiment_id"]:
        raise ValueError("CAMPAIGN_RUN_STAGE_EXPERIMENT_MISMATCH")

    disposition = _text(raw.get("disposition"), "experiment_disposition", optional=True)
    highest_claim = _text(raw.get("highest_safe_claim"), "highest_safe_claim", optional=True)
    not_proven = raw.get("not_proven")
    if disposition is None:
        missing.append(f"{ref_id}:disposition_missing")
    if highest_claim is None:
        missing.append(f"{ref_id}:highest_safe_claim_missing")
    if not isinstance(not_proven, (list, tuple)):
        missing.append(f"{ref_id}:not_proven_missing")
        not_proven_list: list[str] = []
    else:
        not_proven_list = sorted({_text(item, "not_proven_item") for item in not_proven})
    correctness_raw = raw.get("correctness_certification")
    if correctness_raw is None:
        correctness = {"status": "NOT_EVALUATED", "evidence_hash": None}
    elif not isinstance(correctness_raw, Mapping) or correctness_raw.get("status") not in {
        "PASS", "FAIL", "NOT_EVALUATED", "UNSUPPORTED"
    }:
        raise ValueError("CAMPAIGN_CORRECTNESS_CERTIFICATION_INVALID")
    else:
        correctness = {"status": correctness_raw["status"],
                       "evidence_hash": correctness_raw.get("evidence_hash")}
        if correctness["status"] == "PASS" and not _is_hash(correctness["evidence_hash"]):
            raise ValueError("CAMPAIGN_CORRECTNESS_CERTIFICATION_HASH_REQUIRED")
        if correctness["status"] != "PASS" and correctness["evidence_hash"] is not None and not _is_hash(correctness["evidence_hash"]):
            raise ValueError("CAMPAIGN_CORRECTNESS_CERTIFICATION_HASH_INVALID")
    if correctness["status"] != "PASS":
        not_proven_list.append("formal correctness certification")
    if preflight_deferred_terminal_stop:
        not_proven_list.append("full preflight readiness before #26 terminal negative stop")
    if stage is not None and stage["stage"] != STAGE_HOLDOUT_COMPLETE:
        not_proven_list.append("heldout evaluation completion")
    if comparator is not None:
        if comparator["formal_comparable"] is not True:
            not_proven_list.append("formal comparator compatibility")
        if comparator["comparison_scope"] == "WHOLE_RUNTIME_STACK_ONLY":
            not_proven_list.append("isolated runtime/cache/scheduler parity")
    if preflight is not None and isinstance(preflight.get("resource_observation"), Mapping):
        resource = preflight["resource_observation"]
        if resource.get("first_failure_concurrency") is not None:
            not_proven_list.append("resource/admission success above observed concurrency")
        if resource.get("unobserved_concurrency_levels"):
            not_proven_list.append("complete resource/headroom characterization")
    not_proven_list = sorted(set(not_proven_list))
    result["disposition"] = disposition
    result["highest_safe_claim"] = highest_claim
    result["not_proven"] = not_proven_list
    result["correctness_certification"] = correctness
    # Keep correctness/performance boundaries explicit in every projection.
    result["performance_cannot_override_correctness"] = True
    return result, missing


def build_campaign_closeout(
    *,
    campaign_id: str,
    campaign_generation: int,
    experiments: Sequence[Mapping[str, Any]],
    reopen_triggers: Sequence[Mapping[str, Any]],
    next_gate: str,
    closed_at: str,
    evidence_watermark: Mapping[str, Any],
) -> dict[str, Any]:
    """Build campaign closeout; missing receipts stay explicit INCOMPLETE evidence."""
    generation = _generation(campaign_generation, "campaign_generation")
    if not isinstance(experiments, (list, tuple)) or not experiments:
        raise ValueError("CAMPAIGN_EXPERIMENTS_REQUIRED")
    projected: list[dict[str, Any]] = []
    incomplete: list[str] = []
    for raw in experiments:
        item, missing = _project_experiment_ref(raw)
        projected.append(item)
        incomplete.extend(missing)
    refs = [item["experiment_ref"] for item in projected]
    if len(refs) != len(set(refs)):
        raise ValueError("CAMPAIGN_EXPERIMENT_REF_DUPLICATE")
    triggers = []
    seen_triggers: set[str] = set()
    known_experiment_ids = {item["experiment_id"] for item in projected if "experiment_id" in item}
    for raw in reopen_triggers:
        if not isinstance(raw, Mapping):
            raise ValueError("CAMPAIGN_REOPEN_TRIGGER_INVALID")
        item = _normalize_trigger(raw)
        if item["trigger_id"] in seen_triggers:
            raise ValueError("CAMPAIGN_REOPEN_TRIGGER_DUPLICATE")
        seen_triggers.add(item["trigger_id"])
        if item["experiment_id"] not in known_experiment_ids:
            raise ValueError("CAMPAIGN_REOPEN_TRIGGER_EXPERIMENT_UNKNOWN")
        triggers.append(item)
    triggers.sort(key=lambda item: item["trigger_id"])
    gate = _text(next_gate, "next_gate")
    if gate != NO_GATE_DEFINED and (len(gate) < 8 or " " in gate):
        raise ValueError("CAMPAIGN_NEXT_GATE_MUST_BE_EXACT_BOUNDED_IDENTIFIER")
    if not isinstance(evidence_watermark, Mapping) or not evidence_watermark:
        raise ValueError("CAMPAIGN_EVIDENCE_WATERMARK_INVALID")
    watermark = dict(evidence_watermark)
    if not _text(watermark.get("source_ref"), "evidence_watermark_source_ref"):
        raise ValueError("CAMPAIGN_EVIDENCE_WATERMARK_SOURCE_REQUIRED")
    if not _is_hash(watermark.get("sha256")):
        raise ValueError("CAMPAIGN_EVIDENCE_WATERMARK_HASH_INVALID")
    complete = not incomplete
    payload = {
        "schema": CAMPAIGN_CLOSEOUT_SCHEMA,
        "campaign_id": _text(campaign_id, "campaign_id"),
        "campaign_generation": generation,
        "experiment_refs": projected,
        "reopen_triggers": triggers,
        "next_gate": gate,
        "closed_at": _timestamp(closed_at),
        "evidence_watermark": watermark,
        "completion_status": CAMPAIGN_COMPLETE if complete else CAMPAIGN_INCOMPLETE,
        "incomplete_reasons": sorted(set(incomplete)),
        "campaign_closed": complete,
        "claim_ceiling": CAMPAIGN_CLOSEOUT_CLAIM_CEILING,
    }
    payload["binding_hash"] = _hash(payload)
    validate_campaign_closeout(payload)
    return payload


def validate_campaign_closeout(evidence: Any) -> dict[str, Any]:
    if not isinstance(evidence, Mapping) or evidence.get("schema") != CAMPAIGN_CLOSEOUT_SCHEMA:
        raise ValueError("CAMPAIGN_CLOSEOUT_SCHEMA_INVALID")
    expected_keys = {
        "schema", "campaign_id", "campaign_generation", "experiment_refs", "reopen_triggers",
        "next_gate", "closed_at", "evidence_watermark", "completion_status", "incomplete_reasons",
        "campaign_closed", "claim_ceiling", "binding_hash",
    }
    if set(evidence) != expected_keys:
        raise ValueError("CAMPAIGN_CLOSEOUT_SCHEMA_INVALID")
    payload = dict(evidence)
    supplied_hash = payload.pop("binding_hash", None)
    if supplied_hash != _hash(payload):
        raise ValueError("CAMPAIGN_CLOSEOUT_BINDING_HASH_MISMATCH")
    if evidence.get("claim_ceiling") != CAMPAIGN_CLOSEOUT_CLAIM_CEILING:
        raise ValueError("CAMPAIGN_CLOSEOUT_CLAIM_CEILING_INVALID")
    _text(evidence.get("campaign_id"), "campaign_id")
    _generation(evidence.get("campaign_generation"), "campaign_generation")
    _timestamp(evidence.get("closed_at"))
    completion_status = evidence.get("completion_status")
    if not isinstance(completion_status, str) or completion_status not in {CAMPAIGN_COMPLETE, CAMPAIGN_INCOMPLETE}:
        raise ValueError("CAMPAIGN_CLOSEOUT_COMPLETION_STATUS_INVALID")
    reasons = evidence.get("incomplete_reasons")
    if (
        not isinstance(reasons, list)
        or any(not isinstance(reason, str) or not reason.strip() for reason in reasons)
        or reasons != sorted(set(reasons))
    ):
        raise ValueError("CAMPAIGN_CLOSEOUT_INCOMPLETE_REASONS_INVALID")
    complete = not reasons
    if evidence.get("campaign_closed") is not complete:
        raise ValueError("CAMPAIGN_CLOSEOUT_STATE_MISMATCH")
    if evidence.get("completion_status") != (CAMPAIGN_COMPLETE if complete else CAMPAIGN_INCOMPLETE):
        raise ValueError("CAMPAIGN_CLOSEOUT_STATUS_MISMATCH")
    next_gate = evidence.get("next_gate")
    if next_gate == NO_GATE_DEFINED:
        pass
    elif not isinstance(next_gate, str) or not next_gate.strip() or len(next_gate) < 8 or " " in next_gate:
        raise ValueError("CAMPAIGN_NEXT_GATE_INVALID")
    refs = evidence.get("experiment_refs")
    if not isinstance(refs, list) or not refs:
        raise ValueError("CAMPAIGN_EXPERIMENTS_REQUIRED")
    required_for_complete = {
        "experiment_id", "experiment_generation", "run_state", "run_binding_hash",
        "result_receipt_identity", "result_receipt_sha256", "preflight_binding_hash",
        "preflight_disposition", "staged_gate_binding_hash", "staged_gate_stage",
        "staged_gate_terminal", "necessary_condition_stop_binding_hash", "disposition",
        "highest_safe_claim", "not_proven",
        "correctness_certification",
    }
    allowed_ref_keys = required_for_complete | {
        "experiment_ref", "result_receipt_sha256", "preflight_disposition", "staged_gate_binding_hash",
        "staged_gate_stage", "staged_gate_terminal", "necessary_condition_stop_binding_hash",
        "performance_cannot_override_correctness",
    }
    required_ref_keys = {
        "experiment_ref", "not_proven", "correctness_certification",
        "performance_cannot_override_correctness",
    }
    ref_ids: set[str] = set()
    for ref in refs:
        if (
            not isinstance(ref, Mapping)
            or not set(ref).issubset(allowed_ref_keys)
            or not required_ref_keys.issubset(ref)
            or not isinstance(ref.get("not_proven"), list)
            or any(not isinstance(item, str) or not item.strip() for item in ref["not_proven"])
            or ref.get("not_proven") != sorted(set(ref.get("not_proven", [])))
        ):
            raise ValueError("CAMPAIGN_EXPERIMENT_REF_INVALID")
        if not isinstance(ref.get("experiment_ref"), str) or not ref["experiment_ref"].strip() or ref["experiment_ref"] in ref_ids:
            raise ValueError("CAMPAIGN_EXPERIMENT_REF_INVALID")
        ref_ids.add(ref["experiment_ref"])
        if "experiment_id" in ref:
            _text(ref["experiment_id"], "experiment_id")
        if "experiment_generation" in ref and ref["experiment_generation"] is not None:
            _generation(ref["experiment_generation"], "experiment_generation")
        if "run_state" in ref and (
            not isinstance(ref["run_state"], str)
            or ref["run_state"] not in {NOT_STARTED, RUNNING, COMPLETE, FAILED, OUTCOME_UNKNOWN}
        ):
            raise ValueError("CAMPAIGN_EXPERIMENT_RUN_STATE_INVALID")
        if "run_binding_hash" in ref and ref["run_binding_hash"] is not None and not _is_hash(ref["run_binding_hash"]):
            raise ValueError("CAMPAIGN_RUN_BINDING_HASH_INVALID")
        if "result_receipt_identity" in ref and ref["result_receipt_identity"] is not None:
            _text(ref["result_receipt_identity"], "result_receipt_identity")
        if ref.get("result_receipt_sha256") is not None and not _is_hash(ref["result_receipt_sha256"]):
            raise ValueError("CAMPAIGN_RESULT_RECEIPT_HASH_INVALID")
        if "preflight_binding_hash" in ref and ref["preflight_binding_hash"] is not None and not _is_hash(ref["preflight_binding_hash"]):
            raise ValueError("CAMPAIGN_PREFLIGHT_BINDING_HASH_INVALID")
        if "preflight_disposition" in ref and (
            not isinstance(ref["preflight_disposition"], str)
            or ref["preflight_disposition"] not in {PREFLIGHT_PASS, PREFLIGHT_FAIL, PREFLIGHT_DEFER}
        ):
            raise ValueError("CAMPAIGN_PREFLIGHT_DISPOSITION_INVALID")
        if "staged_gate_binding_hash" in ref and ref["staged_gate_binding_hash"] is not None and not _is_hash(ref["staged_gate_binding_hash"]):
            raise ValueError("CAMPAIGN_STAGED_GATE_BINDING_HASH_INVALID")
        if "staged_gate_stage" in ref and ref["staged_gate_stage"] is not None and (
            not isinstance(ref["staged_gate_stage"], str)
            or ref["staged_gate_stage"] not in {
                STAGE_STOP, STAGE_HARD_STOP, STAGE_HOLDOUT_COMPLETE, STAGE_DEFER,
                STAGE_PREFLIGHT, STAGE_SMOKE, STAGE_CALIBRATION, STAGE_HOLDOUT_ELIGIBLE,
            }
        ):
            raise ValueError("CAMPAIGN_STAGED_GATE_STAGE_INVALID")
        if "staged_gate_terminal" in ref and not isinstance(ref["staged_gate_terminal"], bool):
            raise ValueError("CAMPAIGN_STAGED_GATE_TERMINAL_INVALID")
        if "necessary_condition_stop_binding_hash" in ref and ref["necessary_condition_stop_binding_hash"] is not None and not _is_sha256_digest(ref["necessary_condition_stop_binding_hash"]):
            raise ValueError("CAMPAIGN_NECESSARY_STOP_BINDING_INVALID")
        if "disposition" in ref:
            _text(ref["disposition"], "experiment_disposition")
        if "highest_safe_claim" in ref:
            _text(ref["highest_safe_claim"], "highest_safe_claim")
        correctness_ref = ref["correctness_certification"]
        if (
            not isinstance(correctness_ref, Mapping)
            or set(correctness_ref) != {"status", "evidence_hash"}
            or not isinstance(correctness_ref.get("status"), str)
            or correctness_ref.get("status") not in {"PASS", "FAIL", "NOT_EVALUATED", "UNSUPPORTED"}
            or (correctness_ref.get("evidence_hash") is not None and not _is_hash(correctness_ref.get("evidence_hash")))
            or (correctness_ref.get("status") == "PASS" and not _is_hash(correctness_ref.get("evidence_hash")))
        ):
            raise ValueError("CAMPAIGN_CORRECTNESS_CERTIFICATION_INVALID")
        if ref["performance_cannot_override_correctness"] is not True:
            raise ValueError("CAMPAIGN_CORRECTNESS_BOUNDARY_INVALID")
        if complete:
            if not required_for_complete.issubset(ref):
                raise ValueError("CAMPAIGN_COMPLETE_WITH_INCOMPLETE_EXPERIMENT_REF")
            if ref["run_state"] not in {COMPLETE, FAILED}:
                raise ValueError("CAMPAIGN_COMPLETE_WITH_NONTERMINAL_RUN")
            if not _is_hash(ref["run_binding_hash"]) or not _is_hash(ref["preflight_binding_hash"]):
                raise ValueError("CAMPAIGN_COMPLETE_WITH_UNBOUND_EVIDENCE")
            if not _is_hash(ref["staged_gate_binding_hash"]) or ref["staged_gate_terminal"] is not True:
                raise ValueError("CAMPAIGN_COMPLETE_WITH_NONTERMINAL_GATE")
            if ref["staged_gate_stage"] not in {STAGE_STOP, STAGE_HARD_STOP, STAGE_HOLDOUT_COMPLETE}:
                raise ValueError("CAMPAIGN_COMPLETE_WITH_NONTERMINAL_GATE")
            if ref["preflight_disposition"] == PREFLIGHT_DEFER and (
                ref["staged_gate_stage"] != STAGE_HARD_STOP
                or not _is_sha256_digest(ref.get("necessary_condition_stop_binding_hash"))
            ):
                raise ValueError("CAMPAIGN_COMPLETE_WITH_DEFERRED_PREFLIGHT")
            if (
                ref["necessary_condition_stop_binding_hash"] is not None
                and not _is_sha256_digest(ref["necessary_condition_stop_binding_hash"])
            ):
                raise ValueError("CAMPAIGN_NECESSARY_STOP_BINDING_INVALID")
            if not isinstance(ref.get("result_receipt_identity"), str) or not isinstance(ref.get("highest_safe_claim"), str):
                raise ValueError("CAMPAIGN_COMPLETE_WITHOUT_RESULT_OR_CLAIM_BOUNDARY")
            correctness = ref.get("correctness_certification")
            if not isinstance(correctness, Mapping) or correctness.get("status") not in {
                "PASS", "FAIL", "NOT_EVALUATED", "UNSUPPORTED"
            }:
                raise ValueError("CAMPAIGN_CORRECTNESS_CERTIFICATION_INVALID")
            if correctness.get("status") == "PASS" and not _is_hash(correctness.get("evidence_hash")):
                raise ValueError("CAMPAIGN_CORRECTNESS_CERTIFICATION_HASH_REQUIRED")
            if ref.get("preflight_disposition") not in {PREFLIGHT_PASS, PREFLIGHT_FAIL, PREFLIGHT_DEFER}:
                raise ValueError("CAMPAIGN_PREFLIGHT_DISPOSITION_INVALID")
            if not isinstance(ref["experiment_id"], str) or not ref["experiment_id"].strip():
                raise ValueError("CAMPAIGN_EXPERIMENT_ID_INVALID")
            _generation(ref["experiment_generation"], "experiment_generation")
    watermark = evidence.get("evidence_watermark")
    if (
        not isinstance(watermark, Mapping)
        or not isinstance(watermark.get("source_ref"), str)
        or not watermark["source_ref"].strip()
        or not _is_hash(watermark.get("sha256"))
    ):
        raise ValueError("CAMPAIGN_EVIDENCE_WATERMARK_INVALID")
    triggers = evidence.get("reopen_triggers")
    if not isinstance(triggers, list):
        raise ValueError("CAMPAIGN_REOPEN_TRIGGERS_INVALID")
    seen_trigger_ids: set[str] = set()
    known_experiment_ids = {ref["experiment_id"] for ref in refs if isinstance(ref.get("experiment_id"), str)}
    for trigger in triggers:
        if not isinstance(trigger, Mapping):
            raise ValueError("CAMPAIGN_REOPEN_TRIGGER_INVALID")
        normalized = _normalize_trigger(trigger)
        if dict(trigger) != normalized or normalized["trigger_id"] in seen_trigger_ids:
            raise ValueError("CAMPAIGN_REOPEN_TRIGGER_INVALID")
        if normalized["experiment_id"] not in known_experiment_ids:
            raise ValueError("CAMPAIGN_REOPEN_TRIGGER_EXPERIMENT_UNKNOWN")
        seen_trigger_ids.add(normalized["trigger_id"])
    return dict(evidence)


def evaluate_reopen_trigger(
    closeout: Mapping[str, Any],
    *,
    experiment_id: str,
    evidence_delta: Mapping[str, Any] | None = None,
    owner_decision_ref: str | None = None,
    owner_decision_sha256: str | None = None,
    proposed_experiment_generation: int,
) -> dict[str, Any]:
    """Match only an exact named delta/decision; historical closeout stays immutable."""
    record = validate_campaign_closeout(closeout)
    if record["campaign_closed"] is not True:
        raise ValueError("CAMPAIGN_REOPEN_CLOSEOUT_INCOMPLETE")
    proposed_generation = _generation(proposed_experiment_generation, "proposed_experiment_generation")
    refs = [item for item in record["experiment_refs"] if item.get("experiment_id") == experiment_id]
    if not refs:
        raise ValueError("CAMPAIGN_REOPEN_EXPERIMENT_UNKNOWN")
    previous_generations = [
        item.get("experiment_generation")
        for item in refs
        if isinstance(item.get("experiment_generation"), int)
        and not isinstance(item.get("experiment_generation"), bool)
    ]
    if not previous_generations:
        return {
            "schema": CAMPAIGN_REOPEN_SCHEMA,
            "eligible": False,
            "experiment_id": experiment_id,
            "matched_trigger_id": None,
            "prior_closeout_binding_hash": record["binding_hash"],
            "prior_receipt_hashes": sorted(
                item["result_receipt_sha256"] for item in record["experiment_refs"]
                if item.get("result_receipt_sha256")
            ),
            "new_experiment_generation": None,
            "trigger_evidence_hash": None,
            "reason": "prior run generation missing",
            "claim_ceiling": CAMPAIGN_CLOSEOUT_CLAIM_CEILING,
        }
    if proposed_generation <= max(previous_generations):
        raise ValueError("CAMPAIGN_REOPEN_REQUIRES_NEW_EXPERIMENT_GENERATION")
    changes = evidence_delta.get("changes") if isinstance(evidence_delta, Mapping) else None
    delta_hash = evidence_delta.get("evidence_hash") if isinstance(evidence_delta, Mapping) else None
    if evidence_delta is not None and (
        not isinstance(evidence_delta, Mapping)
        or set(evidence_delta) != {"changes", "evidence_hash"}
        or not isinstance(changes, Mapping)
        or not _is_hash(delta_hash)
        or delta_hash != _hash(dict(changes))
    ):
        raise ValueError("CAMPAIGN_REOPEN_DELTA_EVIDENCE_INVALID")
    for trigger in record["reopen_triggers"]:
        if trigger["experiment_id"] != experiment_id:
            continue
        matches = False
        if trigger["trigger_type"] == MATERIAL_DELTA and isinstance(changes, Mapping):
            actual = _read_field_path(changes, trigger["field_path"])
            matches = actual is not _MISSING and actual == trigger["expected_value"]
        elif trigger["trigger_type"] == OWNER_DECISION:
            matches = (owner_decision_ref == trigger["owner_decision_ref"] and
                       owner_decision_sha256 == trigger["owner_decision_sha256"] and
                       _is_hash(owner_decision_sha256))
        if matches:
            return {
                "schema": CAMPAIGN_REOPEN_SCHEMA,
                "eligible": True,
                "experiment_id": experiment_id,
                "matched_trigger_id": trigger["trigger_id"],
                "prior_closeout_binding_hash": record["binding_hash"],
                "prior_receipt_hashes": sorted(
                    item["result_receipt_sha256"] for item in record["experiment_refs"]
                    if item.get("result_receipt_sha256")
                ),
                "new_experiment_generation": proposed_generation,
                "trigger_evidence_hash": delta_hash if trigger["trigger_type"] == MATERIAL_DELTA else owner_decision_sha256,
                "claim_ceiling": CAMPAIGN_CLOSEOUT_CLAIM_CEILING,
            }
    return {
        "schema": CAMPAIGN_REOPEN_SCHEMA,
        "eligible": False,
        "experiment_id": experiment_id,
        "matched_trigger_id": None,
        "prior_closeout_binding_hash": record["binding_hash"],
        "new_experiment_generation": None,
        "reason": "no recorded evidence-bound reopen trigger matched",
        "claim_ceiling": CAMPAIGN_CLOSEOUT_CLAIM_CEILING,
    }
