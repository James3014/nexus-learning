"""Staged cohort preflight + campaign closeout/reopen evidence (#31/#32)."""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Mapping, Sequence

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
        if not isinstance(incumbent_identity.get(key), str) or not incumbent_identity[key].strip():
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
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
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
        and bool(set(metrics) - {"accuracy"})
        and not stop_loss_triggered
        and all(level["disposition"] == DISPOSITION_PASS for level in levels)
    )
    gaps = []
    if not set(metrics) - {"accuracy"}:
        gaps.append("role safety metrics missing; accuracy alone is insufficient")
    if any(level["disposition"] != DISPOSITION_PASS for level in levels):
        gaps.append("candidate stage has not passed; no holdout admission")
    if stop_loss_triggered:
        gaps.append("stop-loss: two adjacent levels failed for the same mechanism reason")
    if (coverage or any("rerank" in level["mechanism"].lower() for level in levels)) and coverage.get("candidates_cover_reranker_claims") is not True:
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
        "eligibility_scope": "RECORDED_PREREQUISITES_ONLY_NOT_HOLDOUT_ADMISSION",
        "evidence_gaps": gaps,
        "claim_ceiling": PREFLIGHT_CLAIM_CEILING,
    }
    body["content_sha256"] = _hash({k: v for k, v in body.items() if k != "content_sha256"})
    return body


def build_campaign_closeout(
    *, campaign_id, closed_experiments, highest_safe_claims, not_proven, reopen_triggers,
    campaign_generation=1, next_gate="NONE_DEFINED", closed_at=None, evidence_watermark=None
):
    campaign_id = _text(campaign_id, "campaign_id")
    if type(campaign_generation) is not int or campaign_generation < 1:
        raise ValueError("campaign_generation must be a positive integer")
    next_gate = _text(next_gate, "next_gate")
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
        "campaign_generation": campaign_generation,
        "next_gate": next_gate,
        "closed_at": closed_at,
        "evidence_watermark": evidence_watermark,
        "closed_experiments": closed,
        "highest_safe_claims": dict(highest_safe_claims or {}),
        "not_proven": list(not_proven or []),
        "reopen_triggers": triggers,
        "claim_ceiling": CLOSEOUT_CLAIM_CEILING,
    }
    gaps = []
    for entry in closed:
        digest = entry.get("receipt_sha256")
        if not isinstance(digest, str) or len(digest.removeprefix("sha256:")) != 64 or any(c not in "0123456789abcdef" for c in digest.removeprefix("sha256:")):
            gaps.append(f"missing or invalid receipt hash: {entry['experiment_id']}")
        if entry["disposition"] == DISPOSITION_NOT_EVALUATED:
            gaps.append(f"experiment not terminal: {entry['experiment_id']}")
        if not entry.get("highest_safe_claim"):
            gaps.append(f"highest safe claim missing: {entry['experiment_id']}")
    if not closed_at or not evidence_watermark:
        gaps.append("closeout time or evidence watermark missing")
    body["evidence_gaps"] = gaps
    body["complete"] = not gaps
    body["content_sha256"] = _hash({k: v for k, v in body.items() if k != "content_sha256"})
    return body


def verify_preflight(body):
    if not isinstance(body, Mapping) or body.get("schema") != PREFLIGHT_SCHEMA:
        return False
    if body.get("claim_ceiling") != PREFLIGHT_CLAIM_CEILING:
        return False
    try:
        recomputed = build_cohort_preflight(**{k: body[k] for k in (
            "cohort_id", "incumbent_identity", "incumbent_result", "candidate_levels",
            "role_metrics", "candidate_coverage")})
        return dict(body) == recomputed
    except (KeyError, TypeError, ValueError, OverflowError):
        return False


def verify_closeout(body):
    if not isinstance(body, Mapping) or body.get("schema") != CLOSEOUT_SCHEMA:
        return False
    if body.get("claim_ceiling") != CLOSEOUT_CLAIM_CEILING:
        return False
    try:
        recomputed = build_campaign_closeout(**{k: body[k] for k in (
            "campaign_id", "closed_experiments", "highest_safe_claims", "not_proven",
            "reopen_triggers", "campaign_generation", "next_gate", "closed_at",
            "evidence_watermark")})
        return dict(body) == recomputed
    except (KeyError, TypeError, ValueError, OverflowError):
        return False


def project_frontier_from_gates(
    *,
    preflight_evidence: Mapping[str, Any],
    closeout_evidence: Mapping[str, Any],
    proposal_id: str,
    hypothesis: str,
    proposed_baseline: Mapping[str, Any],
    mechanism_family: str,
    donor_evidence: Mapping[str, Any] | None = None,
    residual_decision_delta: float | None = None,
    is_optional_branch: bool = False,
    reopen_trigger: Mapping[str, Any] | None = None,
    proposed_authorities: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Compose research frontier governance (#37) from #31 preflight and #32 closeout state."""
    from nexus_learning.research_frontier import build_research_frontier_governance

    if not verify_preflight(preflight_evidence):
        raise ValueError("preflight_evidence failed canonical #31 validation")
    if not verify_closeout(closeout_evidence):
        raise ValueError("closeout_evidence failed canonical #32 validation")

    incumbent = preflight_evidence.get("incumbent_identity")
    if not isinstance(incumbent, Mapping):
        raise ValueError("preflight_evidence missing incumbent_identity mapping")
    return build_research_frontier_governance(
        proposal_id=proposal_id,
        hypothesis=hypothesis,
        current_incumbent=incumbent,
        proposed_baseline=proposed_baseline,
        mechanism_family=mechanism_family,
        campaign_closeout=closeout_evidence,
        donor_evidence=donor_evidence,
        residual_decision_delta=residual_decision_delta,
        is_optional_branch=is_optional_branch,
        reopen_trigger=reopen_trigger,
        proposed_authorities=proposed_authorities,
    )
