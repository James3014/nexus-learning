"""Staged cohort preflight + campaign closeout/reopen evidence (#31/#32)."""

from __future__ import annotations

import hashlib
import json
from typing import Mapping

PREFLIGHT_SCHEMA = "nexus.learning_cohort_preflight.v1"
PREFLIGHT_CLAIM_CEILING = "LEARNING_COHORT_PREFLIGHT_STAGED_GATE_ONLY"
CLOSEOUT_SCHEMA = "nexus.learning_campaign_closeout.v1"
CLOSEOUT_CLAIM_CEILING = "LEARNING_CALIBRATION_CAMPAIGN_CLOSEOUT_AND_REOPEN_EVIDENCE_ONLY"

DISPOSITION_NOT_EVALUATED = "NOT_EVALUATED"
DISPOSITION_FAILED = "FAILED"
DISPOSITION_STOPPED_BY_GATE = "STOPPED_BY_GATE"
DISPOSITION_PASS = "PASS"
DISPOSITIONS = frozenset(
    {DISPOSITION_NOT_EVALUATED, DISPOSITION_FAILED, DISPOSITION_STOPPED_BY_GATE, DISPOSITION_PASS}
)

ROLE_SAFETY_METRICS = (
    "false_safe_rate",
    "unknown_recall",
    "escalate_recall",
    "no_gold_abstention_rate",
    "invented_reference_rate",
    "contract_valid_rate",
    "accuracy",
)

REOPEN_TRIGGER_KINDS = (
    "new_role_with_residual_gap",
    "new_mechanism_delta",
    "dense_index_economics_change",
    "richer_graph_evidence",
    "workload_distribution_shift",
)


def _hash(payload):
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _text(value, field):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value.strip()


def build_cohort_preflight(
    *,
    cohort_id,
    incumbent_identity,
    incumbent_result,
    candidate_levels,
    role_metrics,
    candidate_coverage=None,
):
    cohort_id = _text(cohort_id, "cohort_id")
    if not isinstance(incumbent_identity, Mapping):
        raise ValueError("incumbent_identity must be a mapping")
    incumbent_identity = dict(incumbent_identity)
    for key in ("deterministic_kind", "result_hash"):
        if key not in incumbent_identity:
            raise ValueError(f"incumbent_identity missing {key}")
    if incumbent_result not in DISPOSITIONS:
        raise ValueError("incumbent_result must be a known disposition")
    if not isinstance(candidate_levels, (list, tuple)) or not candidate_levels:
        raise ValueError("candidate_levels must be a non-empty list")
    levels = []
    for entry in candidate_levels:
        if not isinstance(entry, Mapping):
            raise ValueError("candidate_levels entries must be mappings")
        levels.append(
            {
                "level_id": _text(entry.get("level_id"), "level_id"),
                "mechanism": _text(entry.get("mechanism", "unspecified"), "mechanism"),
                "disposition": entry.get("disposition", DISPOSITION_NOT_EVALUATED),
                "failure_reason": entry.get("failure_reason"),
            }
        )
        if levels[-1]["disposition"] not in DISPOSITIONS:
            raise ValueError("candidate disposition unknown")
    if not isinstance(role_metrics, Mapping):
        raise ValueError("role_metrics must be a mapping")
    metrics = {}
    for key in ROLE_SAFETY_METRICS:
        if key in role_metrics:
            value = role_metrics[key]
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"role metric {key} must be numeric")
            metrics[key] = float(value)
    coverage = dict(candidate_coverage or {})
    stop_loss_triggered = False
    for first, second in zip(levels, levels[1:]):
        if (
            first["disposition"] == DISPOSITION_FAILED
            and second["disposition"] == DISPOSITION_FAILED
            and first["failure_reason"]
            and first["failure_reason"] == second["failure_reason"]
        ):
            stop_loss_triggered = True
    eligible = (
        incumbent_result in (DISPOSITION_PASS, DISPOSITION_FAILED, DISPOSITION_STOPPED_BY_GATE)
        and bool(metrics)
        and not stop_loss_triggered
    )
    gaps = []
    if not metrics:
        gaps.append("role safety metrics missing; accuracy alone is insufficient")
    if stop_loss_triggered:
        gaps.append("stop-loss: two adjacent levels failed for the same mechanism reason")
    if coverage and not coverage.get("candidates_cover_reranker_claims", True):
        gaps.append("candidate-coverage precondition unmet for reranker/model claims")
        eligible = False
    body = {
        "schema": PREFLIGHT_SCHEMA,
        "cohort_id": cohort_id,
        "incumbent_identity": incumbent_identity,
        "incumbent_result": incumbent_result,
        "candidate_levels": levels,
        "role_metrics": metrics,
        "candidate_coverage": coverage,
        "stop_loss_triggered": stop_loss_triggered,
        "eligible": eligible,
        "evidence_gaps": gaps,
        "claim_ceiling": PREFLIGHT_CLAIM_CEILING,
    }
    body["content_sha256"] = _hash({k: v for k, v in body.items() if k != "content_sha256"})
    return body


def build_campaign_closeout(
    *, campaign_id, closed_experiments, highest_safe_claims, not_proven, reopen_triggers
):
    campaign_id = _text(campaign_id, "campaign_id")
    if not isinstance(closed_experiments, (list, tuple)) or not closed_experiments:
        raise ValueError("closed_experiments must be a non-empty list")
    closed = []
    for entry in closed_experiments:
        if not isinstance(entry, Mapping):
            raise ValueError("closed_experiments entries must be mappings")
        exp_id = _text(entry.get("experiment_id"), "experiment_id")
        disp = entry.get("disposition", DISPOSITION_NOT_EVALUATED)
        if disp not in DISPOSITIONS:
            raise ValueError("experiment disposition unknown")
        closed.append(
            {
                "experiment_id": exp_id,
                "disposition": disp,
                "highest_safe_claim": entry.get("highest_safe_claim"),
                "receipt_sha256": entry.get("receipt_sha256"),
            }
        )
    triggers = []
    for entry in reopen_triggers or []:
        if not isinstance(entry, Mapping):
            raise ValueError("reopen_triggers entries must be mappings")
        kind = entry.get("kind")
        if kind not in REOPEN_TRIGGER_KINDS:
            raise ValueError(f"reopen trigger kind unknown: {kind!r}")
        triggers.append(
            {
                "kind": kind,
                "condition": _text(entry.get("condition", "unspecified"), "condition"),
                "evidence_required": entry.get("evidence_required", "named material delta"),
            }
        )
    body = {
        "schema": CLOSEOUT_SCHEMA,
        "campaign_id": campaign_id,
        "closed_experiments": closed,
        "highest_safe_claims": dict(highest_safe_claims or {}),
        "not_proven": list(not_proven or []),
        "reopen_triggers": triggers,
        "claim_ceiling": CLOSEOUT_CLAIM_CEILING,
    }
    body["content_sha256"] = _hash({k: v for k, v in body.items() if k != "content_sha256"})
    return body


def verify_preflight(body):
    if not isinstance(body, Mapping) or body.get("schema") != PREFLIGHT_SCHEMA:
        return False
    if body.get("claim_ceiling") != PREFLIGHT_CLAIM_CEILING:
        return False
    unsigned = {k: v for k, v in body.items() if k != "content_sha256"}
    return _hash(unsigned) == body.get("content_sha256")


def verify_closeout(body):
    if not isinstance(body, Mapping) or body.get("schema") != CLOSEOUT_SCHEMA:
        return False
    if body.get("claim_ceiling") != CLOSEOUT_CLAIM_CEILING:
        return False
    unsigned = {k: v for k, v in body.items() if k != "content_sha256"}
    return _hash(unsigned) == body.get("content_sha256")
