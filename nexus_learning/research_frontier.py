"""Research governance: inherit latest incumbent and gate external-donor transfer (#37).

Forces every successor proposal to inherit the latest accepted incumbent,
requires evidence of experiment necessity before authorizing a new matrix,
treats closed negative mechanism families as closed unless an evidence-bound
reopen trigger is satisfied, classifies external donor evidence into reuse/
minimal-transfer validation vs full experiment, drops optional branches with
zero decision delta, and ensures candidate-generation misses cannot be blamed
on downstream models.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping, Sequence

RESEARCH_FRONTIER_SCHEMA = "nexus.learning_research_frontier_governance.v1"
RESEARCH_FRONTIER_CLAIM_CEILING = (
    "LEARNING_RESEARCH_FRONTIER_AND_TRANSFER_SCOPE_EVIDENCE_ONLY"
)

# Frontier Dispositions
DISPOSITION_INELIGIBLE_OBSOLETE_BASELINE = "INELIGIBLE_OBSOLETE_BASELINE"
DISPOSITION_REOPEN_TRIGGER_REQUIRED = "REOPEN_TRIGGER_REQUIRED"
DISPOSITION_ELIGIBLE_MINIMUM_TRANSFER_VALIDATION = "ELIGIBLE_MINIMUM_TRANSFER_VALIDATION"
DISPOSITION_ELIGIBLE_FULL_EXPERIMENT = "ELIGIBLE_FULL_EXPERIMENT"
DISPOSITION_ELIGIBLE_BOUNDED_EXPERIMENT = "ELIGIBLE_BOUNDED_EXPERIMENT"
DISPOSITION_DROP_NO_DECISION_DELTA = "DROP_NO_DECISION_DELTA"

FRONTIER_DISPOSITIONS = frozenset(
    {
        DISPOSITION_INELIGIBLE_OBSOLETE_BASELINE,
        DISPOSITION_REOPEN_TRIGGER_REQUIRED,
        DISPOSITION_ELIGIBLE_MINIMUM_TRANSFER_VALIDATION,
        DISPOSITION_ELIGIBLE_FULL_EXPERIMENT,
        DISPOSITION_ELIGIBLE_BOUNDED_EXPERIMENT,
        DISPOSITION_DROP_NO_DECISION_DELTA,
    }
)

# Attribution classes for failure isolation
ATTRIBUTION_CANDIDATE_GENERATION_MISS = "CANDIDATE_GENERATION_MISS"
ATTRIBUTION_MODEL_CAPABILITY = "MODEL_CAPABILITY"
ATTRIBUTION_POLICY_BINDING = "POLICY_BINDING"
ATTRIBUTION_PROVENANCE_COMPATIBILITY = "PROVENANCE_COMPATIBILITY"
ATTRIBUTION_WORKLOAD_DISTRIBUTION = "WORKLOAD_DISTRIBUTION"

ATTRIBUTION_CLASSES = frozenset(
    {
        ATTRIBUTION_CANDIDATE_GENERATION_MISS,
        ATTRIBUTION_MODEL_CAPABILITY,
        ATTRIBUTION_POLICY_BINDING,
        ATTRIBUTION_PROVENANCE_COMPATIBILITY,
        ATTRIBUTION_WORKLOAD_DISTRIBUTION,
    }
)

FORBIDDEN_AUTHORITIES = frozenset(
    {
        "production",
        "production_promotion",
        "route",
        "routing",
        "routing_authority",
        "admission",
        "workforce",
        "capability_planner",
        "merge",
        "release",
        "acceptance",
    }
)


def _hash(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value.strip()


def classify_failure_attribution(
    *,
    candidate_hit: bool,
    downstream_success: bool = True,
    requested_attribution: str | None = None,
    failure_reason: str | None = None,
) -> dict[str, Any]:
    """Isolate whether a failure originated in candidate generation or downstream model.

    Acceptance requirement: candidate-generation miss cannot be credited or blamed
    to downstream model capability.
    """
    if not candidate_hit:
        if requested_attribution in (
            ATTRIBUTION_MODEL_CAPABILITY,
            "model",
            "model_capability",
        ):
            raise ValueError(
                "candidate-generation miss cannot be credited or blamed to downstream model"
            )
        return {
            "attribution": ATTRIBUTION_CANDIDATE_GENERATION_MISS,
            "blame_downstream_model": False,
            "candidate_hit": False,
            "downstream_evaluated": False,
            "reason": failure_reason or "candidate generation missed required target",
        }

    if not downstream_success:
        attr = requested_attribution or ATTRIBUTION_MODEL_CAPABILITY
        if attr not in ATTRIBUTION_CLASSES:
            raise ValueError(f"unknown attribution class: {attr!r}")
        return {
            "attribution": attr,
            "blame_downstream_model": (attr == ATTRIBUTION_MODEL_CAPABILITY),
            "candidate_hit": True,
            "downstream_evaluated": True,
            "reason": failure_reason or "downstream execution failed with valid candidate",
        }

    return {
        "attribution": "NONE",
        "blame_downstream_model": False,
        "candidate_hit": True,
        "downstream_evaluated": True,
        "reason": "success; no failure observed",
    }


def build_research_frontier_governance(
    *,
    proposal_id: str,
    hypothesis: str,
    current_incumbent: Mapping[str, Any],
    proposed_baseline: Mapping[str, Any],
    mechanism_family: str,
    campaign_closeout: Mapping[str, Any] | None = None,
    donor_evidence: Mapping[str, Any] | None = None,
    residual_decision_delta: float | None = None,
    is_optional_branch: bool = False,
    reopen_trigger: Mapping[str, Any] | None = None,
    proposed_authorities: Sequence[str] | None = None,
    claim_ceiling: str = RESEARCH_FRONTIER_CLAIM_CEILING,
) -> dict[str, Any]:
    """Evaluate research frontier proposal necessity, incumbent binding, and transfer scope.

    Implements issue #37 governance acceptance criteria.
    """
    proposal_id = _text(proposal_id, "proposal_id")
    hypothesis = _text(hypothesis, "hypothesis")
    mechanism_family = _text(mechanism_family, "mechanism_family")

    # Enforce claim ceiling and strict non-authority
    if claim_ceiling != RESEARCH_FRONTIER_CLAIM_CEILING:
        raise ValueError(f"claim_ceiling must be {RESEARCH_FRONTIER_CLAIM_CEILING}")

    if proposed_authorities:
        for auth in proposed_authorities:
            normalized = str(auth).strip().lower()
            if normalized in FORBIDDEN_AUTHORITIES or any(
                f in normalized for f in FORBIDDEN_AUTHORITIES
            ):
                raise ValueError(
                    f"frontier governance strictly forbids authority claim: {auth}"
                )

    if not isinstance(current_incumbent, Mapping):
        raise ValueError("current_incumbent must be a mapping")
    if not isinstance(proposed_baseline, Mapping):
        raise ValueError("proposed_baseline must be a mapping")

    # Check incumbent identity keys
    cur_mech = _text(
        current_incumbent.get("mechanism")
        or current_incumbent.get("deterministic_kind"),
        "current_incumbent.mechanism",
    )
    cur_hash = _text(
        current_incumbent.get("result_hash")
        or current_incumbent.get("receipt_sha256"),
        "current_incumbent.result_hash",
    )

    base_mech = str(
        proposed_baseline.get("mechanism")
        or proposed_baseline.get("deterministic_kind")
        or ""
    ).strip()
    base_hash = str(
        proposed_baseline.get("result_hash")
        or proposed_baseline.get("receipt_sha256")
        or ""
    ).strip()

    evidence_gaps: list[str] = []
    disposition: str
    eligible: bool
    scope_recommendation: str

    # 1. Obsolete baseline check: proposal compares against obsolete baseline
    # when newer incumbent is bound
    if (
        base_mech.lower() != cur_mech.lower()
        or (base_hash and cur_hash and base_hash != cur_hash)
        or proposed_baseline.get("is_obsolete") is True
    ):
        disposition = DISPOSITION_INELIGIBLE_OBSOLETE_BASELINE
        eligible = False
        scope_recommendation = "REBASE_REQUIRED_AGAINST_LATEST_INCUMBENT"
        evidence_gaps.append(
            f"proposal compares against obsolete baseline {base_mech!r}; "
            f"must rebase against bound incumbent {cur_mech!r}"
        )

    # 2. Closed negative mechanism check: closed negative mechanism + new model version
    # but no reopen trigger -> REOPEN_TRIGGER_REQUIRED
    elif campaign_closeout is not None and _is_closed_negative(
        mechanism_family, campaign_closeout
    ):
        trigger_satisfied = _is_reopen_trigger_satisfied(
            mechanism_family, campaign_closeout, reopen_trigger
        )
        if not trigger_satisfied:
            disposition = DISPOSITION_REOPEN_TRIGGER_REQUIRED
            eligible = False
            scope_recommendation = "REOPEN_TRIGGER_REQUIRED_BEFORE_EXPERIMENT"
            evidence_gaps.append(
                f"mechanism family {mechanism_family!r} was closed negative; "
                "evidence-bound reopen trigger required before authorizing new experiment"
            )
        else:
            disposition = DISPOSITION_ELIGIBLE_BOUNDED_EXPERIMENT
            eligible = True
            scope_recommendation = "REOPEN_TRIGGER_SATISFIED_BOUNDED_EXPERIMENT"

    # 3. Optional branch check: optional branch with no measurable residual workload/decision delta
    elif is_optional_branch and (
        residual_decision_delta is not None and residual_decision_delta <= 0
    ):
        disposition = DISPOSITION_DROP_NO_DECISION_DELTA
        eligible = False
        scope_recommendation = "DROP_NO_DECISION_DELTA"
        evidence_gaps.append(
            "optional mechanism branch has zero or negative residual workload/decision delta; drop branch"
        )

    # 4. Donor evidence classification
    elif donor_evidence is not None:
        if not isinstance(donor_evidence, Mapping):
            raise ValueError("donor_evidence must be a mapping")

        auth_mismatch = bool(donor_evidence.get("authority_mismatch", False))
        fail_mismatch = bool(donor_evidence.get("failure_model_mismatch", False))
        aligned = (
            donor_evidence.get("alignment") in ("ALIGNED", True)
            and not auth_mismatch
            and not fail_mismatch
        )

        if aligned:
            # External donor with aligned proof obligations -> recommends minimum transfer validation, not full science rerun
            disposition = DISPOSITION_ELIGIBLE_MINIMUM_TRANSFER_VALIDATION
            eligible = True
            scope_recommendation = (
                "MINIMUM_TRANSFER_VALIDATION_ONLY_NO_FULL_SCIENCE_RERUN"
            )
        else:
            # Donor with unresolved authority/failure-model mismatch -> bounded/full Nexus experiment remains required
            disposition = DISPOSITION_ELIGIBLE_FULL_EXPERIMENT
            eligible = True
            mismatches = []
            if auth_mismatch:
                mismatches.append("authority_mismatch")
            if fail_mismatch:
                mismatches.append("failure_model_mismatch")
            scope_recommendation = f"FULL_EXPERIMENT_REQUIRED_DUE_TO_{'_AND_'.join(mismatches).upper()}"
            evidence_gaps.append(
                f"external donor has unresolved mismatch ({', '.join(mismatches)}); "
                "requires bounded/full Nexus experiment"
            )

    # 5. Default proposal with valid incumbent and no donor
    else:
        disposition = DISPOSITION_ELIGIBLE_BOUNDED_EXPERIMENT
        eligible = True
        scope_recommendation = "BOUNDED_EXPERIMENT_AUTHORIZED"

    body: dict[str, Any] = {
        "schema": RESEARCH_FRONTIER_SCHEMA,
        "proposal_id": proposal_id,
        "hypothesis": hypothesis,
        "mechanism_family": mechanism_family,
        "current_incumbent": {
            "mechanism": cur_mech,
            "result_hash": cur_hash,
        },
        "proposed_baseline": {
            "mechanism": base_mech,
            "result_hash": base_hash,
        },
        "disposition": disposition,
        "eligible": eligible,
        "scope_recommendation": scope_recommendation,
        "is_optional_branch": is_optional_branch,
        "residual_decision_delta": residual_decision_delta,
        "evidence_gaps": evidence_gaps,
        "claim_ceiling": claim_ceiling,
    }
    body["content_sha256"] = _hash(
        {k: v for k, v in body.items() if k != "content_sha256"}
    )
    return body


def _is_closed_negative(
    mechanism_family: str, campaign_closeout: Mapping[str, Any]
) -> bool:
    """Check whether mechanism family is recorded as closed negative in campaign closeout."""
    mech_norm = mechanism_family.strip().lower()

    # Check not_proven list
    for np in campaign_closeout.get("not_proven", []):
        if str(np).strip().lower() in mech_norm or mech_norm in str(np).strip().lower():
            return True

    # Check closed_experiments
    for exp in campaign_closeout.get("closed_experiments", []):
        if not isinstance(exp, Mapping):
            continue
        exp_id = str(exp.get("experiment_id", "")).strip().lower()
        disp = str(exp.get("disposition", "")).strip()
        if (exp_id in mech_norm or mech_norm in exp_id) and disp in (
            "FAILED",
            "STOPPED_BY_GATE",
            "TERMINAL_NEGATIVE",
            "TERMINAL_STOP",
        ):
            return True

    return False


def _is_reopen_trigger_satisfied(
    mechanism_family: str,
    campaign_closeout: Mapping[str, Any],
    reopen_trigger: Mapping[str, Any] | None,
) -> bool:
    """Verify whether a valid reopen trigger is satisfied for a closed mechanism family."""
    if not reopen_trigger or not isinstance(reopen_trigger, Mapping):
        return False

    if reopen_trigger.get("satisfied") is not True:
        return False

    trigger_kind = reopen_trigger.get("kind")
    if not trigger_kind:
        return False

    # Verify the trigger kind matches campaign closeout defined triggers
    known_triggers = [
        t.get("kind")
        for t in campaign_closeout.get("reopen_triggers", [])
        if isinstance(t, Mapping)
    ]
    if known_triggers and trigger_kind not in known_triggers:
        return False

    # Verify evidence_delta exists and is material
    delta = reopen_trigger.get("evidence_delta")
    if not delta or not str(delta).strip():
        return False

    return True


def validate_research_frontier_governance(evidence: Any) -> dict[str, Any]:
    """Validate research frontier evidence payload fail-closed."""
    if not isinstance(evidence, Mapping):
        raise ValueError("evidence must be a mapping")

    if evidence.get("schema") != RESEARCH_FRONTIER_SCHEMA:
        raise ValueError(
            f"invalid schema: expected {RESEARCH_FRONTIER_SCHEMA}, got {evidence.get('schema')!r}"
        )

    if evidence.get("claim_ceiling") != RESEARCH_FRONTIER_CLAIM_CEILING:
        raise ValueError(
            f"invalid claim_ceiling: expected {RESEARCH_FRONTIER_CLAIM_CEILING}, "
            f"got {evidence.get('claim_ceiling')!r}"
        )

    disp = evidence.get("disposition")
    if disp not in FRONTIER_DISPOSITIONS:
        raise ValueError(f"unknown frontier disposition: {disp!r}")

    expected_hash = _hash({k: v for k, v in evidence.items() if k != "content_sha256"})
    if evidence.get("content_sha256") != expected_hash:
        raise ValueError("content_sha256 mismatch: evidence payload has been tampered")

    return dict(evidence)


def verify_frontier_governance(body: Mapping[str, Any]) -> bool:
    """Convenience predicate to verify frontier evidence validity."""
    try:
        validate_research_frontier_governance(body)
        return True
    except (ValueError, TypeError, KeyError):
        return False
