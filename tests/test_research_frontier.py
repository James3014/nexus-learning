"""Acceptance tests for Issue #37: research governance and donor transfer scope."""

from __future__ import annotations

import pytest

from nexus_learning.research_frontier import (
    ATTRIBUTION_CANDIDATE_GENERATION_MISS,
    ATTRIBUTION_MODEL_CAPABILITY,
    DISPOSITION_DROP_NO_DECISION_DELTA,
    DISPOSITION_ELIGIBLE_BOUNDED_EXPERIMENT,
    DISPOSITION_ELIGIBLE_FULL_EXPERIMENT,
    DISPOSITION_ELIGIBLE_MINIMUM_TRANSFER_VALIDATION,
    DISPOSITION_INELIGIBLE_OBSOLETE_BASELINE,
    DISPOSITION_REOPEN_TRIGGER_REQUIRED,
    RESEARCH_FRONTIER_CLAIM_CEILING,
    RESEARCH_FRONTIER_SCHEMA,
    build_research_frontier_governance,
    classify_failure_attribution,
    validate_research_frontier_governance,
    verify_frontier_governance,
)


def _base_proposal(**overrides):
    kw = {
        "proposal_id": "prop-ctx-acq-01",
        "hypothesis": "test context acquisition proposal",
        "mechanism_family": "sparse_retrieval",
        "current_incumbent": {
            "mechanism": "deterministic-first",
            "result_hash": "sha256:" + "a" * 64,
        },
        "proposed_baseline": {
            "mechanism": "deterministic-first",
            "result_hash": "sha256:" + "a" * 64,
        },
    }
    kw.update(overrides)
    return build_research_frontier_governance(**kw)


def test_obsolete_baseline_ineligible_until_rebased():
    """Criterion 1: proposal compares against obsolete baseline when newer incumbent is bound -> ineligible."""
    body = _base_proposal(
        proposed_baseline={
            "mechanism": "online-only",
            "result_hash": "sha256:" + "b" * 64,
        }
    )
    assert body["schema"] == RESEARCH_FRONTIER_SCHEMA
    assert body["claim_ceiling"] == RESEARCH_FRONTIER_CLAIM_CEILING
    assert body["eligible"] is False
    assert body["disposition"] == DISPOSITION_INELIGIBLE_OBSOLETE_BASELINE
    assert body["scope_recommendation"] == "REBASE_REQUIRED_AGAINST_LATEST_INCUMBENT"
    assert any("rebase" in g.lower() for g in body["evidence_gaps"])
    assert verify_frontier_governance(body)


def test_obsolete_baseline_hash_mismatch():
    """Criterion 1: same mechanism name but obsolete receipt hash -> rebase required."""
    body = _base_proposal(
        proposed_baseline={
            "mechanism": "deterministic-first",
            "result_hash": "sha256:" + "0" * 64,
        }
    )
    assert body["eligible"] is False
    assert body["disposition"] == DISPOSITION_INELIGIBLE_OBSOLETE_BASELINE


def test_closed_negative_mechanism_requires_reopen_trigger():
    """Criterion 2: closed negative mechanism + new model version but no reopen trigger -> REOPEN_TRIGGER_REQUIRED."""
    closeout = {
        "closed_experiments": [
            {
                "experiment_id": "dense_embedding_experiment",
                "disposition": "FAILED",
            }
        ],
        "not_proven": ["dense_embedding"],
        "reopen_triggers": [
            {
                "kind": "dense_index_economics_change",
                "condition": "hardware-accelerated dense index reduces latency 10x",
            }
        ],
    }

    body = _base_proposal(
        mechanism_family="dense_embedding",
        campaign_closeout=closeout,
        # No satisfied reopen trigger provided; just a new model version
        reopen_trigger=None,
    )
    assert body["eligible"] is False
    assert body["disposition"] == DISPOSITION_REOPEN_TRIGGER_REQUIRED
    assert any("reopen trigger required" in g.lower() for g in body["evidence_gaps"])
    assert verify_frontier_governance(body)


def test_closed_negative_with_satisfied_reopen_trigger_becomes_eligible():
    """Criterion 2: satisfied reopen trigger admits bounded experiment."""
    closeout = {
        "closed_experiments": [
            {"experiment_id": "dense_embedding", "disposition": "FAILED"}
        ],
        "not_proven": ["dense_embedding"],
        "reopen_triggers": [
            {
                "kind": "dense_index_economics_change",
                "condition": "hardware-accelerated dense index",
            }
        ],
    }
    reopen_trig = {
        "kind": "dense_index_economics_change",
        "satisfied": True,
        "evidence_delta": "new M5 Pro hardware index achieves 2ms latency",
    }
    body = _base_proposal(
        mechanism_family="dense_embedding",
        campaign_closeout=closeout,
        reopen_trigger=reopen_trig,
    )
    assert body["eligible"] is True
    assert body["disposition"] == DISPOSITION_ELIGIBLE_BOUNDED_EXPERIMENT
    assert verify_frontier_governance(body)


def test_external_donor_aligned_proof_obligations_minimum_transfer_only():
    """Criterion 3: external donor with aligned proof obligations -> recommends minimum transfer validation, not full science rerun."""
    donor = {
        "donor_id": "Retrieval_V2",
        "receipt_sha256": "8dc40b9a9da06cc75e33ca064ca3258bf052f6a24ce6a6e6a2722c83421c1367",
        "alignment": "ALIGNED",
        "authority_mismatch": False,
        "failure_model_mismatch": False,
    }
    body = _base_proposal(donor_evidence=donor)
    assert body["eligible"] is True
    assert body["disposition"] == DISPOSITION_ELIGIBLE_MINIMUM_TRANSFER_VALIDATION
    assert body["scope_recommendation"] == "MINIMUM_TRANSFER_VALIDATION_ONLY_NO_FULL_SCIENCE_RERUN"
    assert verify_frontier_governance(body)


def test_external_donor_mismatch_requires_full_or_bounded_experiment():
    """Criterion 4: donor with unresolved authority/failure-model mismatch -> bounded/full Nexus experiment remains required."""
    donor = {
        "donor_id": "External_Unverified_System",
        "alignment": "ALIGNED",
        "authority_mismatch": True,
        "failure_model_mismatch": False,
    }
    body = _base_proposal(donor_evidence=donor)
    assert body["eligible"] is True
    assert body["disposition"] == DISPOSITION_ELIGIBLE_FULL_EXPERIMENT
    assert "AUTHORITY_MISMATCH" in body["scope_recommendation"]
    assert any("mismatch" in g.lower() for g in body["evidence_gaps"])
    assert verify_frontier_governance(body)


def test_optional_branch_zero_decision_delta_dropped():
    """Criterion 5: optional branch with no measurable residual workload/decision delta -> DROP_NO_DECISION_DELTA."""
    body = _base_proposal(
        is_optional_branch=True,
        residual_decision_delta=0.0,
    )
    assert body["eligible"] is False
    assert body["disposition"] == DISPOSITION_DROP_NO_DECISION_DELTA
    assert body["scope_recommendation"] == "DROP_NO_DECISION_DELTA"
    assert any("drop branch" in g.lower() for g in body["evidence_gaps"])
    assert verify_frontier_governance(body)


def test_candidate_generation_miss_cannot_blame_downstream_model():
    """Criterion 6: candidate-generation miss cannot be credited or blamed to downstream model."""
    # Attempting to blame downstream model when candidates missed gold must raise ValueError
    with pytest.raises(ValueError, match="candidate-generation miss cannot be credited or blamed"):
        classify_failure_attribution(
            candidate_hit=False,
            downstream_success=False,
            requested_attribution=ATTRIBUTION_MODEL_CAPABILITY,
        )

    # Valid classification isolates failure to candidate generation
    attr = classify_failure_attribution(
        candidate_hit=False,
        downstream_success=False,
        failure_reason="retriever Top-24 omitted target file",
    )
    assert attr["attribution"] == ATTRIBUTION_CANDIDATE_GENERATION_MISS
    assert attr["blame_downstream_model"] is False
    assert attr["candidate_hit"] is False
    assert attr["downstream_evaluated"] is False


def test_downstream_model_can_be_blamed_only_when_candidates_hit():
    """Criterion 6 (cont): model capability is evaluated only when candidates cover required target."""
    attr = classify_failure_attribution(
        candidate_hit=True,
        downstream_success=False,
        failure_reason="model output failed AST syntax parse",
    )
    assert attr["attribution"] == ATTRIBUTION_MODEL_CAPABILITY
    assert attr["blame_downstream_model"] is True
    assert attr["candidate_hit"] is True
    assert attr["downstream_evaluated"] is True


def test_component_pass_cannot_imply_production_or_routing_authority():
    """Criteria 7 & 8: component PASS cannot imply production or route/admission change; no forbidden authority."""
    for forbidden in (
        "production",
        "production_promotion",
        "route",
        "routing",
        "routing_authority",
        "workforce",
        "capability_planner",
        "merge",
        "release",
        "acceptance",
    ):
        with pytest.raises(ValueError, match="strictly forbids authority claim"):
            _base_proposal(proposed_authorities=[forbidden])

    with pytest.raises(ValueError, match="claim_ceiling must be"):
        _base_proposal(claim_ceiling="PRODUCTION_ROLLOUT_AUTHORITY")


def test_tamper_fails_validation():
    """Tampering with fields or hash fails closed."""
    body = _base_proposal()
    assert verify_frontier_governance(body)

    # Tamper with disposition
    tampered = dict(body)
    tampered["disposition"] = "ROGUE_AUTHORITY"
    with pytest.raises(ValueError, match="unknown frontier disposition"):
        validate_research_frontier_governance(tampered)
    assert not verify_frontier_governance(tampered)

    # Tamper with content hash
    tampered_hash = dict(body)
    tampered_hash["content_sha256"] = "sha256:" + "0" * 64
    with pytest.raises(ValueError, match="content_sha256 mismatch"):
        validate_research_frontier_governance(tampered_hash)
    assert not verify_frontier_governance(tampered_hash)


def test_project_frontier_from_gates_composition():
    """Verify composition from #31 preflight and #32 closeout without parallel registry."""
    from nexus_learning.campaign_gates import (
        build_campaign_closeout,
        build_cohort_preflight,
        project_frontier_from_gates,
    )

    preflight = build_cohort_preflight(
        cohort_id="c1",
        incumbent_identity={
            "deterministic_kind": "deterministic-first",
            "result_hash": "sha256:" + "c" * 64,
        },
        incumbent_result="PASS",
        candidate_levels=[
            {"level_id": "l1", "mechanism": "m", "disposition": "PASS"}
        ],
        role_metrics={"false_safe_rate": 0.0},
    )

    closeout = build_campaign_closeout(
        campaign_id="camp1",
        closed_experiments=[
            {
                "experiment_id": "exp1",
                "disposition": "PASS",
                "highest_safe_claim": "safe",
                "receipt_sha256": "sha256:" + "d" * 64,
            }
        ],
        highest_safe_claims={},
        not_proven=[],
        reopen_triggers=[],
        closed_at="2026-09-29T18:00:00Z",
        evidence_watermark="sha256:" + "e" * 64,
    )

    # Obsolete baseline check through projected gates
    projected = project_frontier_from_gates(
        preflight_evidence=preflight,
        closeout_evidence=closeout,
        proposal_id="prop1",
        hypothesis="successor test",
        proposed_baseline={
            "mechanism": "online-only",
            "result_hash": "sha256:" + "f" * 64,
        },
        mechanism_family="m",
    )
    assert projected["disposition"] == DISPOSITION_INELIGIBLE_OBSOLETE_BASELINE
    assert projected["eligible"] is False
    assert verify_frontier_governance(projected)

