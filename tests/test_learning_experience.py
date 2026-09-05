from types import SimpleNamespace

from nexus_learning.contracts import (
    CAPABILITY_TAXONOMY,
    apply_autodata_quality_gate,
    build_escalation_recommendations,
    build_learning_experience,
    build_promoted_learning_policy,
    load_promoted_learning_policy,
    project_model_training,
    project_nexus_policy,
    save_promoted_learning_policy,
)


class LearningSteward:
    def decide_experience(self, experience):
        lifecycle = getattr(experience, "capability_lifecycle", ()) or ()
        funnel_complete = any(getattr(item, "funnel_complete", False) for item in lifecycle)
        s2t_refs = getattr(experience, "s2t_trace_refs", ()) or ()
        promotion_ready = funnel_complete
        export_ready = bool(s2t_refs)
        nexus_action = "PROMOTE_NEXUS" if promotion_ready else "INGEST_SHADOW"
        model_action = "EXPORT_MODEL" if export_ready else "INGEST_SHADOW"
        reasons = []
        if not funnel_complete:
            reasons.append("no_complete_capability_funnel")
        if not export_ready:
            reasons.append("missing_s2t_trace_refs")
        return SimpleNamespace(
            nexus_action=nexus_action,
            model_action=model_action,
            reasons=reasons,
            promotion_ready=promotion_ready,
            export_ready=export_ready,
        )

class CapabilityPlanner:
    def plan(self, task_desc, task_type, route, budget=None):
        budget_dict = budget or {}
        lp = budget_dict.get("learning_policy", {})
        caps = ["memory"] if lp.get("episodic_memory_injection", {}).get("enabled") else []
        return SimpleNamespace(
            selected_capabilities=caps,
            signal_snapshot={
                "learning_policy": lp,
                "route_truth_source": "CapabilityPlanner",
            },
        )



def test_learning_experience_unifies_phase_capability_and_gate_chain() -> None:
    usage_trace = {
        "phase_trace": {"S": "start", "P": "plan", "X": "context", "D": "design", "R": "repair", "A": "audit", "C": "close"},
        "capabilities": {
            "artifact_gate_passed": True,
            "artifact_refs": ["artifact:task:tests_passed"],
            "claim_verified": True,
            "delivery_gate_passed": True,
            "delivery_refs": ["delivery:task:artifact_tests_passed"],
        },
        "s2t": {"trace_path": ".nexus/reports/s2t/runtime_trace.jsonl"},
    }
    receipts = [
        {
            "name": "codeintel",
            "selected": True,
            "invoked": True,
            "evidence_present": True,
            "gate_passed": True,
            "outcome_contributed": True,
            "evidence_refs": ["codeintel:impact"],
        },
        {
            "name": "swarm",
            "selected": True,
            "invoked": False,
            "evidence_present": False,
            "gate_passed": False,
            "outcome_contributed": False,
        },
    ]

    exp = build_learning_experience(
        task_id="task-1",
        task_type="bug",
        usage_trace=usage_trace,
        capability_receipts=receipts,
        route_decision_ref="route:task-1",
        learning_steward_decision="INGEST_SHADOW",
    )

    payload = exp.to_dict()
    assert payload["schema_version"] == "nexus_learning_experience.v1"
    assert payload["phase_continuity"]["complete"] is True
    assert payload["gate_chain"]["artifact"] == "pass"
    assert payload["gate_chain"]["claim"] == "pass"
    assert payload["outcome"] == "verified_success"
    assert payload["capability_lifecycle"][0]["category"] == "recon_context"
    assert payload["capability_lifecycle"][0]["funnel_complete"] is True
    assert payload["capability_lifecycle"][1]["capability"] == "swarm"

    nexus_projection = project_nexus_policy(exp)
    assert nexus_projection["route_weight_updates"] == ["codeintel"]
    assert nexus_projection["capability_penalties"] == ["swarm"]
    assert nexus_projection["escalation_recommendations"] == []
    assert nexus_projection["s2t_prior_eligible"] is True

    model_projection = project_model_training(exp)
    assert model_projection["training_eligible"] is True
    assert model_projection["targets"] == ["preference_pair", "reward_row"]

    decision = LearningSteward().decide_experience(exp)
    assert decision.nexus_action == "PROMOTE_NEXUS"
    assert decision.model_action == "EXPORT_MODEL"


def test_capability_taxonomy_covers_core_route_space() -> None:
    for name in (
        "codeintel",
        "research",
        "hyper",
        "nightshift",
        "swarm",
        "drone",
        "ultra_review",
        "autoreason",
        "ddtree",
        "memory",
        "lancedb",
        "mempalace_gate",
        "belief",
        "artifact_gate",
        "claim_gate",
        "delivery_gate",
        "benchmark",
        "meta_opt",
    ):
        assert name in CAPABILITY_TAXONOMY


def test_learning_experience_escalates_failed_hyper_and_gates_autodata_export() -> None:
    exp = build_learning_experience(
        task_id="task-2",
        task_type="bug",
        usage_trace={
            "phase_trace": {"S": "start", "P": "plan", "X": "context", "D": "design", "R": "repair", "A": "audit", "C": "close"},
            "capabilities": {
                "artifact_gate_passed": True,
                "artifact_refs": ["artifact:task-2"],
                "claim_verified": True,
                "delivery_gate_passed": True,
                "delivery_refs": ["delivery:task-2"],
            },
            "s2t": {"trace_path": ".nexus/reports/s2t/task-2.jsonl"},
        },
        capability_receipts=[
            {
                "name": "hyper",
                "selected": True,
                "invoked": True,
                "evidence_present": True,
                "gate_passed": True,
                "outcome_contributed": False,
                "evidence_refs": ["hyper:attempts"],
            }
        ],
    )

    assert build_escalation_recommendations(exp) == [
        {"from": "hyper", "to": "nightshift", "reason": "hyper_invoked_without_outcome"}
    ]
    decision = LearningSteward().decide_experience(exp)
    assert decision.nexus_action == "INGEST_SHADOW"
    assert decision.model_action == "EXPORT_MODEL"
    assert "no_complete_capability_funnel" in decision.reasons

    gated = apply_autodata_quality_gate(
        project_model_training(exp),
        {
            "task_id": "task-2",
            "eligible_for_training": False,
            "reasons": ["low_step_trajectory"],
            "trajectory_steps": 2,
            "information_density": 0.2,
        },
    )
    assert gated["training_eligible"] is False
    assert gated["targets"] == ["hard_negative"]
    assert gated["autodata_gate"]["status"] == "fail"


def test_model_training_gate_fails_closed_without_autodata_or_s2t_trace() -> None:
    exp = build_learning_experience(
        task_id="task-no-trace",
        task_type="bug",
        usage_trace={
            "phase_trace": {"S": "start", "P": "plan", "X": "context", "D": "design", "R": "repair", "A": "audit", "C": "close"},
            "capabilities": {
                "artifact_gate_passed": True,
                "claim_verified": True,
                "delivery_gate_passed": True,
            },
        },
        capability_receipts=[
            {
                "name": "autoreason",
                "selected": True,
                "invoked": True,
                "evidence_present": True,
                "gate_passed": True,
                "outcome_contributed": True,
            }
        ],
    )

    gated = apply_autodata_quality_gate(project_model_training(exp), None)

    assert gated["training_eligible"] is False
    assert gated["targets"] == ["hard_negative"]
    assert gated["autodata_gate"]["status"] == "not_attached"
    assert "missing_autodata_quality_row" in gated["model_training_gate"]["reasons"]
    assert "missing_s2t_trace_refs" in gated["model_training_gate"]["reasons"]


def test_model_training_gate_blocks_leakage_and_reward_hacking_risk() -> None:
    exp = build_learning_experience(
        task_id="task-risk",
        task_type="bug",
        usage_trace={
            "phase_trace": {"S": "start", "P": "plan", "X": "context", "D": "design", "R": "repair", "A": "audit", "C": "close"},
            "capabilities": {
                "artifact_gate_passed": True,
                "claim_verified": True,
                "delivery_gate_passed": True,
            },
            "s2t": {"trace_path": ".nexus/reports/s2t/task-risk.jsonl"},
        },
        capability_receipts=[
            {
                "name": "autoreason",
                "selected": True,
                "invoked": True,
                "evidence_present": True,
                "gate_passed": True,
                "outcome_contributed": True,
            }
        ],
    )

    gated = apply_autodata_quality_gate(
        project_model_training(exp),
        {
            "task_id": "task-risk",
            "eligible_for_training": True,
            "leakage_risk": True,
            "reward_hacking_risk": True,
            "trajectory_steps": 9,
            "information_density": 0.9,
        },
    )

    assert gated["training_eligible"] is False
    assert gated["targets"] == ["hard_negative"]
    assert gated["model_training_gate"]["status"] == "fail"
    assert "leakage_risk" in gated["model_training_gate"]["reasons"]
    assert "reward_hacking_risk" in gated["model_training_gate"]["reasons"]


def test_promoted_learning_policy_artifact_round_trips_verified_experience(tmp_path) -> None:
    exp = build_learning_experience(
        task_id="task-3",
        task_type="bug",
        usage_trace={
            "phase_trace": {"S": "start", "P": "plan", "X": "context", "D": "design", "R": "repair", "A": "audit", "C": "close"},
            "capabilities": {
                "artifact_gate_passed": True,
                "artifact_refs": ["artifact:task-3"],
                "claim_verified": True,
                "delivery_gate_passed": True,
                "delivery_refs": ["delivery:task-3"],
            },
        },
        capability_receipts=[
            {
                "name": "codeintel",
                "selected": True,
                "invoked": True,
                "evidence_present": True,
                "gate_passed": True,
                "outcome_contributed": True,
                "evidence_refs": ["codeintel:impact"],
            },
            {"name": "swarm", "selected": True, "invoked": False},
        ],
    )
    policy = build_promoted_learning_policy([exp])
    path = tmp_path / ".nexus" / "policy" / "promoted_learning_policy.json"

    saved = save_promoted_learning_policy(path, [exp])
    loaded = load_promoted_learning_policy(path)

    assert policy["promoted_capabilities"] == ["codeintel"]
    assert saved == loaded
    assert loaded["penalized_capabilities"] == ["swarm"]


def test_promoted_learning_policy_accumulates_recent_high_cost_roi_penalties(tmp_path) -> None:
    first = build_learning_experience(
        task_id="task-roi-1",
        task_type="bug",
        usage_trace={"phase_trace": {"S": "start", "P": "plan", "X": "context", "D": "design", "R": "repair", "A": "audit", "C": "close"}},
        capability_receipts=[
            {"name": "research", "selected": True, "invoked": False},
            {"name": "external_doc_scout", "selected": True, "invoked": False},
        ],
    )
    second = build_learning_experience(
        task_id="task-roi-2",
        task_type="bug",
        usage_trace={"phase_trace": {"S": "start", "P": "plan", "X": "context", "D": "design", "R": "repair", "A": "audit", "C": "close"}},
        capability_receipts=[
            {"name": "research", "selected": True, "invoked": True, "evidence_present": False, "outcome_contributed": False},
            {"name": "external_doc_scout", "selected": True, "invoked": False},
        ],
    )
    path = tmp_path / ".nexus" / "policy" / "promoted_learning_policy.json"

    save_promoted_learning_policy(path, [first])
    saved = save_promoted_learning_policy(path, [second])

    assert saved["capability_roi"]["research"]["selected"] == 2
    assert saved["capability_roi"]["research"]["invoked"] == 1
    assert saved["capability_roi"]["external_doc_scout"]["selected"] == 2
    assert "research" in saved["penalty_candidates"]
    assert "external_doc_scout" in saved["penalty_candidates"]
    assert saved["enforce_penalties"] is True


# ==============================================================================
# G5 & G6 Learning Policy Recommendation & Validation Tests
# ==============================================================================

def test_g5_recommendation_is_content_addressed_and_evidence_bound():
    from nexus_learning.contracts import (
        LEARNING_POLICY_RECOMMENDATION_SCHEMA,
        build_learning_policy_recommendation,
        build_nexus_learning_episode,
        canonical_recommendation_identity,
    )

    epA = build_nexus_learning_episode(
        task_id="task_A",
        source="runtime_closure",
        terminal_outcome="SUCCESS",
        terminal_evidence={"verifier": "pytest", "receipt": "rec_A", "verifier_status": "passed"},
        qualification={"repeatability": True, "prevention_rule": "rule", "authority_qualification": True},
        lesson_disposition="graduated",
        learning_write_succeeded=True,
    )
    off_arm = {"task_id": "task_1", "verifier_status": "failed", "receipt": "rec_off"}
    on_arm = {"task_id": "task_1", "verifier_status": "passed", "receipt": "rec_on"}

    rec = build_learning_policy_recommendation(
        source_episodes=[epA],
        source_evidence_refs=["rec_A", "retrieval_receipt:g2", "physical_consumption:ollama"],
        source_revision="7955b770b88e0aeb852e5e735de54346b98b6b2b",
        runtime_identity="local_model_executor",
        task_fingerprint="task_1",
        off_arm=off_arm,
        on_arm=on_arm,
        applicable_scope={"task_family": "record_serialization", "model_name": "qwen2.5-coder:7b"},
        recommended_policy_delta={"episodic_memory_injection": {"enabled": True}},
        current_policy={"episodic_memory_injection": {"enabled": False}},
        expected_effect="Improve pass rate",
        rollback_target={"target_state": {"episodic_memory_injection": {"enabled": False}}, "trigger": "regression"},
    )

    assert rec["schema"] == LEARNING_POLICY_RECOMMENDATION_SCHEMA
    rec_hash, rec_id = canonical_recommendation_identity(rec)
    assert rec["recommendation_hash"] == rec_hash
    assert rec["recommendation_id"] == rec_id
    assert rec["status"] == "PROPOSED"
    assert rec["direct_mutation_allowed"] is False
    assert rec["claim_ceiling"] == "SUPPORTED_POLICY_RECOMMENDATION"


def test_g5_negative_controls_fail_closed():
    import pytest

    from nexus_learning.contracts import (
        HISTORICAL_UNKNOWN,
        NEXUS_LEARNING_EPISODE_SCHEMA,
        build_learning_policy_recommendation,
        build_nexus_learning_episode,
    )

    epA = build_nexus_learning_episode(
        task_id="task_A",
        source="runtime_closure",
        terminal_outcome="SUCCESS",
        terminal_evidence={"verifier": "pytest", "receipt": "rec_A", "verifier_status": "passed"},
        qualification={"repeatability": True, "prevention_rule": "rule", "authority_qualification": True},
        lesson_disposition="graduated",
        learning_write_succeeded=True,
    )
    off_arm = {"task_id": "task_1", "verifier_status": "failed", "receipt": "rec_off"}
    on_arm = {"task_id": "task_1", "verifier_status": "passed", "receipt": "rec_on"}

    # G5-N1: tampered ON result
    with pytest.raises(ValueError, match="RECOMMENDATION_PAIRED_UPLIFT_NOT_OBSERVED"):
        build_learning_policy_recommendation(
            source_episodes=[epA],
            source_evidence_refs=["rec_A"],
            source_revision="rev1",
            runtime_identity="local_model_executor",
            task_fingerprint="task_1",
            off_arm=off_arm,
            on_arm={"task_id": "task_1", "verifier_status": "failed", "receipt": "rec_on"},
            applicable_scope={"task": "task_1"},
            recommended_policy_delta={"mem": True},
            current_policy={"mem": False},
            expected_effect="uplift",
            rollback_target={"target_state": {"mem": False}},
        )

    # G5-N2: tampered OFF result
    with pytest.raises(ValueError, match="RECOMMENDATION_PAIRED_UPLIFT_NOT_OBSERVED"):
        build_learning_policy_recommendation(
            source_episodes=[epA],
            source_evidence_refs=["rec_A"],
            source_revision="rev1",
            runtime_identity="local_model_executor",
            task_fingerprint="task_1",
            off_arm={"task_id": "task_1", "verifier_status": "passed", "receipt": "rec_off"},
            on_arm=on_arm,
            applicable_scope={"task": "task_1"},
            recommended_policy_delta={"mem": True},
            current_policy={"mem": False},
            expected_effect="uplift",
            rollback_target={"target_state": {"mem": False}},
        )

    # G5-N3: fingerprint mismatch
    with pytest.raises(ValueError, match="RECOMMENDATION_PAIRED_UPLIFT_NOT_OBSERVED"):
        build_learning_policy_recommendation(
            source_episodes=[epA],
            source_evidence_refs=["rec_A"],
            source_revision="rev1",
            runtime_identity="local_model_executor",
            task_fingerprint="task_1",
            off_arm=off_arm,
            on_arm={"task_id": "task_2", "verifier_status": "passed", "receipt": "rec_on"},
            applicable_scope={"task": "task_1"},
            recommended_policy_delta={"mem": True},
            current_policy={"mem": False},
            expected_effect="uplift",
            rollback_target={"target_state": {"mem": False}},
        )

    # G5-N10: direct Planner mutation
    with pytest.raises(ValueError, match="RECOMMENDATION_DIRECT_PLANNER_MUTATION_FORBIDDEN"):
        build_learning_policy_recommendation(
            source_episodes=[epA],
            source_evidence_refs=["rec_A"],
            source_revision="rev1",
            runtime_identity="local_model_executor",
            task_fingerprint="task_1",
            off_arm=off_arm,
            on_arm=on_arm,
            applicable_scope={"task": "task_1"},
            recommended_policy_delta={"CapabilityPlanner": {"override": True}},
            current_policy={"mem": False},
            expected_effect="uplift",
            rollback_target={"target_state": {"mem": False}},
        )

    # G5-N11: overbroad scope claim
    with pytest.raises(ValueError, match="RECOMMENDATION_OVERBROAD_SCOPE_FORBIDDEN"):
        build_learning_policy_recommendation(
            source_episodes=[epA],
            source_evidence_refs=["rec_A"],
            source_revision="rev1",
            runtime_identity="local_model_executor",
            task_fingerprint="task_1",
            off_arm=off_arm,
            on_arm=on_arm,
            applicable_scope={"universal_learning_claim": True},
            recommended_policy_delta={"mem": True},
            current_policy={"mem": False},
            expected_effect="uplift",
            rollback_target={"target_state": {"mem": False}},
        )

    # G5-N12: historical unknown provenance
    fake_episode = {
        "schema": NEXUS_LEARNING_EPISODE_SCHEMA,
        "source_schema": NEXUS_LEARNING_EPISODE_SCHEMA,
        "episode_id": HISTORICAL_UNKNOWN,
        "idempotency_key": HISTORICAL_UNKNOWN,
        "task_id": "task_unknown",
        "producer": "unknown",
        "terminal_evidence": {"receipt": "rec_1", "verifier_status": "passed"},
        "stages": {"outcome_measured": True},
    }
    with pytest.raises(ValueError):
        build_learning_policy_recommendation(
            source_episodes=[fake_episode],
            source_evidence_refs=["rec_1"],
            source_revision="rev1",
            runtime_identity="local_model_executor",
            task_fingerprint="task_1",
            off_arm=off_arm,
            on_arm=on_arm,
            applicable_scope={"task": "task_1"},
            recommended_policy_delta={"mem": True},
            current_policy={"mem": False},
            expected_effect="uplift",
            rollback_target={"target_state": {"mem": False}},
        )


def test_g6_independent_validation_and_hostile_probes():
    import copy

    from nexus_learning.contracts import (
        LEARNING_POLICY_VALIDATION_SCHEMA,
        build_learning_policy_recommendation,
        build_nexus_learning_episode,
        canonical_recommendation_identity,
        evaluate_learning_policy_recommendation,
    )

    epA = build_nexus_learning_episode(
        task_id="task_A",
        source="runtime_closure",
        terminal_outcome="SUCCESS",
        terminal_evidence={"verifier": "pytest", "receipt": "rec_A", "verifier_status": "passed"},
        qualification={"repeatability": True, "prevention_rule": "rule", "authority_qualification": True},
        lesson_disposition="graduated",
        learning_write_succeeded=True,
    )
    rec = build_learning_policy_recommendation(
        source_episodes=[epA],
        source_evidence_refs=["rec_A", "retrieval_receipt:g2", "physical_consumption:ollama"],
        source_revision="rev_current",
        runtime_identity="local_model_executor",
        task_fingerprint="task_1",
        off_arm={"task_id": "task_1", "verifier_status": "failed", "receipt": "rec_off"},
        on_arm={"task_id": "task_1", "verifier_status": "passed", "receipt": "rec_on"},
        applicable_scope={"task_family": "record_serialization"},
        recommended_policy_delta={"episodic_memory_injection": {"enabled": True}},
        current_policy={"episodic_memory_injection": {"enabled": False}},
        expected_effect="Improve pass rate",
        rollback_target={"target_state": {"episodic_memory_injection": {"enabled": False}}, "trigger": "regression"},
    )

    # Positive witness
    val = evaluate_learning_policy_recommendation(
        rec,
        validator_identity="claude_deep_review",
        current_workspace_revision="rev_current",
        current_runtime_identity="local_model_executor",
    )
    assert val["schema"] == LEARNING_POLICY_VALIDATION_SCHEMA
    assert val["validation_disposition"] == "VALIDATED_FOR_ADOPTION_CONSIDERATION"
    assert val["freshness_result"] == "FRESH"
    assert val["blockers"] == []

    # Hostile Probe 1: Stale source revision
    val_stale = evaluate_learning_policy_recommendation(
        rec,
        validator_identity="claude_deep_review",
        current_workspace_revision="rev_stale_999",
        current_runtime_identity="local_model_executor",
    )
    assert val_stale["validation_disposition"] == "INSUFFICIENT_EVIDENCE"
    assert "source_revision_stale" in val_stale["blockers"]

    # Hostile Probe 2: Runtime identity mismatch
    val_id_mismatch = evaluate_learning_policy_recommendation(
        rec,
        validator_identity="claude_deep_review",
        current_workspace_revision="rev_current",
        current_runtime_identity="other_engine",
    )
    assert val_id_mismatch["validation_disposition"] == "INSUFFICIENT_EVIDENCE"
    assert "runtime_identity_mismatch" in val_id_mismatch["blockers"]

    # Hostile Probe 3: Missing physical consumption witness
    tampered_rec = copy.deepcopy(rec)
    tampered_rec["source_evidence_refs"] = ["rec_A", "retrieval_receipt:g2"]
    h, i = canonical_recommendation_identity(tampered_rec)
    tampered_rec["recommendation_hash"] = h
    tampered_rec["recommendation_id"] = i
    val_noconsume = evaluate_learning_policy_recommendation(
        tampered_rec,
        validator_identity="claude_deep_review",
        current_workspace_revision="rev_current",
        current_runtime_identity="local_model_executor",
    )
    assert val_noconsume["validation_disposition"] == "INSUFFICIENT_EVIDENCE"
    assert "missing_physical_consumption" in val_noconsume["blockers"]

    # Hostile Probe 4: Authority boundary violation
    tampered_auth = copy.deepcopy(rec)
    tampered_auth["direct_mutation_allowed"] = True
    h, i = canonical_recommendation_identity(tampered_auth)
    tampered_auth["recommendation_hash"] = h
    tampered_auth["recommendation_id"] = i
    val_auth = evaluate_learning_policy_recommendation(
        tampered_auth,
        validator_identity="claude_deep_review",
        current_workspace_revision="rev_current",
        current_runtime_identity="local_model_executor",
    )
    assert val_auth["validation_disposition"] == "REJECTED_RECOMMENDATION"
    assert "violates_authority_boundary" in val_auth["blockers"]


# ==============================================================================
# G7, G8, G10 Learning Policy Adoption, Planner Seam & Rollback Tests
# ==============================================================================

def test_g7_adoption_and_hostile_controls():
    import copy

    import pytest

    from nexus_learning.contracts import (
        LEARNING_POLICY_ADOPTION_SCHEMA,
        build_learning_policy_adoption,
        build_learning_policy_recommendation,
        build_nexus_learning_episode,
        canonical_adoption_identity,
        evaluate_learning_policy_recommendation,
    )

    epA = build_nexus_learning_episode(
        task_id="task_A",
        source="runtime_closure",
        terminal_outcome="SUCCESS",
        terminal_evidence={"verifier": "pytest", "receipt": "rec_A", "verifier_status": "passed"},
        qualification={"repeatability": True, "prevention_rule": "rule", "authority_qualification": True},
        lesson_disposition="graduated",
        learning_write_succeeded=True,
    )
    rec = build_learning_policy_recommendation(
        source_episodes=[epA],
        source_evidence_refs=["rec_A", "retrieval_receipt:g2", "physical_consumption:ollama"],
        source_revision="rev_current",
        runtime_identity="local_model_executor",
        task_fingerprint="task_1",
        off_arm={"task_id": "task_1", "verifier_status": "failed", "receipt": "rec_off"},
        on_arm={"task_id": "task_1", "verifier_status": "passed", "receipt": "rec_on"},
        applicable_scope={"task_family": "record_serialization", "model_name": "qwen2.5-coder:7b", "runtime_identity": "local_model_executor"},
        recommended_policy_delta={"episodic_memory_injection": {"enabled": True, "scope": "record_serialization"}},
        current_policy={"episodic_memory_injection": {"enabled": False}},
        expected_effect="Improve pass rate",
        rollback_target={"target_state": {"episodic_memory_injection": {"enabled": False}}, "trigger": "regression"},
    )
    val = evaluate_learning_policy_recommendation(
        rec,
        validator_identity="claude_deep_review",
        current_workspace_revision="rev_current",
        current_runtime_identity="local_model_executor",
    )

    # Positive G7 adoption
    adopt = build_learning_policy_adoption(
        owner_authority_reference="MISSION_NEXUS_LEARNING_RUNTIME_G7_G10_LONG_MISSION_V1:JamesChen",
        recommendation=rec,
        validation=val,
        source_revision="rev_current",
        adopted_scope={"task_family": "record_serialization", "model_name": "qwen2.5-coder:7b", "runtime_identity": "local_model_executor"},
        target_policy_delta={"episodic_memory_injection": {"enabled": True, "scope": "record_serialization"}},
        previous_policy={"episodic_memory_injection": {"enabled": False}},
        rollback_target={"target_state": {"episodic_memory_injection": {"enabled": False}}, "trigger": "regression"},
    )
    assert adopt["schema"] == LEARNING_POLICY_ADOPTION_SCHEMA
    h, i = canonical_adoption_identity(adopt)
    assert adopt["adoption_hash"] == h
    assert adopt["adoption_id"] == i
    assert adopt["adoption_status"] == "ACTIVE_CANDIDATE"
    assert adopt["route_truth_source"] == "CapabilityPlanner"
    assert adopt["direct_route_mutation_allowed"] is False

    # G7-N1: recommendation hash substituted in validation
    with pytest.raises(ValueError, match="ADOPTION_RECOMMENDATION_VALIDATION_MISMATCH"):
        tampered_val = copy.deepcopy(val)
        tampered_val["recommendation_hash"] = "tampered_rec_hash_1111"
        build_learning_policy_adoption(
            owner_authority_reference="owner",
            recommendation=rec,
            validation=tampered_val,
            source_revision="rev_current",
            adopted_scope=rec["applicable_scope"],
            target_policy_delta={"mem": True},
            previous_policy={"mem": False},
            rollback_target={"target_state": {"mem": False}},
        )

    # G7-N2: validation hash substituted
    with pytest.raises(ValueError, match="ADOPTION_VALIDATION_HASH_TAMPERED"):
        tampered_val = copy.deepcopy(val)
        tampered_val["validation_hash"] = "tampered_hash_2222"
        build_learning_policy_adoption(
            owner_authority_reference="owner",
            recommendation=rec,
            validation=tampered_val,
            source_revision="rev_current",
            adopted_scope=rec["applicable_scope"],
            target_policy_delta={"mem": True},
            previous_policy={"mem": False},
            rollback_target={"target_state": {"mem": False}},
        )

    # G7-N4: Owner authority missing
    with pytest.raises(ValueError, match="ADOPTION_OWNER_AUTHORITY_MISSING"):
        build_learning_policy_adoption(
            owner_authority_reference="",
            recommendation=rec,
            validation=val,
            source_revision="rev_current",
            adopted_scope=rec["applicable_scope"],
            target_policy_delta={"mem": True},
            previous_policy={"mem": False},
            rollback_target={"target_state": {"mem": False}},
        )

    # G7-N6: scope wider than recommendation
    with pytest.raises(ValueError, match="ADOPTION_SCOPE_OVERBROAD"):
        build_learning_policy_adoption(
            owner_authority_reference="owner",
            recommendation=rec,
            validation=val,
            source_revision="rev_current",
            adopted_scope={"universal_learning_claim": True},
            target_policy_delta={"mem": True},
            previous_policy={"mem": False},
            rollback_target={"target_state": {"mem": False}},
        )

    # G7-N7: different model
    with pytest.raises(ValueError, match="ADOPTION_SCOPE_MODEL_MISMATCH"):
        build_learning_policy_adoption(
            owner_authority_reference="owner",
            recommendation=rec,
            validation=val,
            source_revision="rev_current",
            adopted_scope={"task_family": "record_serialization", "model_name": "different_model:70b", "runtime_identity": "local_model_executor"},
            target_policy_delta={"mem": True},
            previous_policy={"mem": False},
            rollback_target={"target_state": {"mem": False}},
        )

    # G7-N9: stale source revision
    with pytest.raises(ValueError, match="ADOPTION_SOURCE_REVISION_STALE"):
        build_learning_policy_adoption(
            owner_authority_reference="owner",
            recommendation=rec,
            validation=val,
            source_revision="stale_rev_999",
            adopted_scope=rec["applicable_scope"],
            target_policy_delta={"mem": True},
            previous_policy={"mem": False},
            rollback_target={"target_state": {"mem": False}},
        )

    # G7-N12: direct planner route mutation forbidden
    with pytest.raises(ValueError, match="ADOPTION_DIRECT_PLANNER_MUTATION_FORBIDDEN"):
        build_learning_policy_adoption(
            owner_authority_reference="owner",
            recommendation=rec,
            validation=val,
            source_revision="rev_current",
            adopted_scope=rec["applicable_scope"],
            target_policy_delta={"CapabilityPlanner": {"force_route": "bypass"}},
            previous_policy={"mem": False},
            rollback_target={"target_state": {"mem": False}},
        )


def test_g8_planner_consumption_and_negative_controls():
    from nexus_learning.contracts import (
        build_learning_policy_adoption,
        build_learning_policy_recommendation,
        build_nexus_learning_episode,
        evaluate_learning_policy_recommendation,
        project_adoption_into_planner_budget,
    )
    # planner mock fixture

    epA = build_nexus_learning_episode(
        task_id="task_A",
        source="runtime_closure",
        terminal_outcome="SUCCESS",
        terminal_evidence={"verifier": "pytest", "receipt": "rec_A", "verifier_status": "passed"},
        qualification={"repeatability": True, "prevention_rule": "rule", "authority_qualification": True},
        lesson_disposition="graduated",
        learning_write_succeeded=True,
    )
    rec = build_learning_policy_recommendation(
        source_episodes=[epA],
        source_evidence_refs=["rec_A", "retrieval_receipt:g2", "physical_consumption:ollama"],
        source_revision="rev_current",
        runtime_identity="local_model_executor",
        task_fingerprint="task_1",
        off_arm={"task_id": "task_1", "verifier_status": "failed", "receipt": "rec_off"},
        on_arm={"task_id": "task_1", "verifier_status": "passed", "receipt": "rec_on"},
        applicable_scope={"task_family": "record_serialization", "model_name": "qwen2.5-coder:7b", "runtime_identity": "local_model_executor"},
        recommended_policy_delta={"episodic_memory_injection": {"enabled": True, "scope": "record_serialization"}},
        current_policy={"episodic_memory_injection": {"enabled": False}},
        expected_effect="Improve pass rate",
        rollback_target={"target_state": {"episodic_memory_injection": {"enabled": False}}, "trigger": "regression"},
    )
    val = evaluate_learning_policy_recommendation(
        rec,
        validator_identity="claude_deep_review",
        current_workspace_revision="rev_current",
        current_runtime_identity="local_model_executor",
    )
    adopt = build_learning_policy_adoption(
        owner_authority_reference="owner",
        recommendation=rec,
        validation=val,
        source_revision="rev_current",
        adopted_scope={"task_family": "record_serialization", "model_name": "qwen2.5-coder:7b", "runtime_identity": "local_model_executor"},
        target_policy_delta={"episodic_memory_injection": {"enabled": True, "scope": "record_serialization"}},
        previous_policy={"episodic_memory_injection": {"enabled": False}},
        rollback_target={"target_state": {"episodic_memory_injection": {"enabled": False}}, "trigger": "regression"},
    )

    planner = CapabilityPlanner()

    # G8 Positive Witness: in-scope task
    budget_in_scope = project_adoption_into_planner_budget(
        adopt,
        task_desc="repair record_serialization for user profile",
        target_model="qwen2.5-coder:7b",
        runtime_identity="local_model_executor",
    )
    plan = planner.plan(
        task_desc="repair record_serialization for user profile",
        task_type="repair",
        route={"route_features": {"deterministic_verifier_available": True}, "workforce_admission_enabled": False},
        budget=budget_in_scope,
    )
    assert "memory" in plan.selected_capabilities
    assert plan.signal_snapshot["learning_policy"]["episodic_memory_injection"]["enabled"] is True
    assert plan.signal_snapshot["learning_policy"]["adoption_lineage"]["adoption_id"] == adopt["adoption_id"]
    assert plan.signal_snapshot["route_truth_source"] == "CapabilityPlanner"

    # G8-N1: unrelated task family
    budget_unrelated = project_adoption_into_planner_budget(
        adopt,
        task_desc="unrelated database optimization",
        target_model="qwen2.5-coder:7b",
        runtime_identity="local_model_executor",
    )
    assert budget_unrelated["learning_policy"]["episodic_memory_injection"]["enabled"] is False

    # G8-N2: wrong model
    budget_wrong_model = project_adoption_into_planner_budget(
        adopt,
        task_desc="repair record_serialization for user profile",
        target_model="different_model:70b",
        runtime_identity="local_model_executor",
    )
    assert budget_wrong_model["learning_policy"]["episodic_memory_injection"]["enabled"] is False

    # G8-N3: wrong runtime
    budget_wrong_runtime = project_adoption_into_planner_budget(
        adopt,
        task_desc="repair record_serialization for user profile",
        target_model="qwen2.5-coder:7b",
        runtime_identity="cloud_executor",
    )
    assert budget_wrong_runtime["learning_policy"]["episodic_memory_injection"]["enabled"] is False


def test_g10_rollback_and_hostile_controls():
    import copy

    import pytest

    from nexus_learning.contracts import (
        LEARNING_POLICY_ROLLBACK_SCHEMA,
        build_learning_policy_adoption,
        build_learning_policy_recommendation,
        build_learning_policy_rollback,
        build_nexus_learning_episode,
        canonical_rollback_identity,
        evaluate_learning_policy_recommendation,
        project_adoption_into_planner_budget,
        validate_learning_policy_rollback,
    )
    # planner mock fixture

    epA = build_nexus_learning_episode(
        task_id="task_A",
        source="runtime_closure",
        terminal_outcome="SUCCESS",
        terminal_evidence={"verifier": "pytest", "receipt": "rec_A", "verifier_status": "passed"},
        qualification={"repeatability": True, "prevention_rule": "rule", "authority_qualification": True},
        lesson_disposition="graduated",
        learning_write_succeeded=True,
    )
    rec = build_learning_policy_recommendation(
        source_episodes=[epA],
        source_evidence_refs=["rec_A", "retrieval_receipt:g2", "physical_consumption:ollama"],
        source_revision="rev_current",
        runtime_identity="local_model_executor",
        task_fingerprint="task_1",
        off_arm={"task_id": "task_1", "verifier_status": "failed", "receipt": "rec_off"},
        on_arm={"task_id": "task_1", "verifier_status": "passed", "receipt": "rec_on"},
        applicable_scope={"task_family": "record_serialization", "model_name": "qwen2.5-coder:7b", "runtime_identity": "local_model_executor"},
        recommended_policy_delta={"episodic_memory_injection": {"enabled": True, "scope": "record_serialization"}},
        current_policy={"episodic_memory_injection": {"enabled": False}},
        expected_effect="Improve pass rate",
        rollback_target={"target_state": {"episodic_memory_injection": {"enabled": False}}, "trigger": "regression"},
    )
    val = evaluate_learning_policy_recommendation(
        rec,
        validator_identity="claude_deep_review",
        current_workspace_revision="rev_current",
        current_runtime_identity="local_model_executor",
    )
    adopt = build_learning_policy_adoption(
        owner_authority_reference="owner",
        recommendation=rec,
        validation=val,
        source_revision="rev_current",
        adopted_scope={"task_family": "record_serialization", "model_name": "qwen2.5-coder:7b", "runtime_identity": "local_model_executor"},
        target_policy_delta={"episodic_memory_injection": {"enabled": True, "scope": "record_serialization"}},
        previous_policy={"episodic_memory_injection": {"enabled": False}},
        rollback_target={"target_state": {"episodic_memory_injection": {"enabled": False}}, "trigger": "regression"},
    )

    # Positive G10 Rollback
    rollback = build_learning_policy_rollback(
        adoption=adopt,
        reason="owner_requested_or_falsification_test",
    )
    assert rollback["schema"] == LEARNING_POLICY_ROLLBACK_SCHEMA
    h, i = canonical_rollback_identity(rollback)
    assert rollback["rollback_hash"] == h
    assert rollback["rollback_id"] == i
    assert rollback["rollback_status"] == "ROLLED_BACK"
    assert rollback["route_truth_source"] == "CapabilityPlanner"

    # Control-plane reversal in CapabilityPlanner
    budget_rb = project_adoption_into_planner_budget(
        adopt,
        task_desc="repair record_serialization for user profile",
        rollback=rollback,
    )
    planner = CapabilityPlanner()
    plan_rb = planner.plan(
        task_desc="repair record_serialization for user profile",
        task_type="repair",
        route={"route_features": {"deterministic_verifier_available": True}, "workforce_admission_enabled": False},
        budget=budget_rb,
    )
    assert plan_rb.signal_snapshot["learning_policy"]["episodic_memory_injection"]["enabled"] is False

    # G10-R1: rollback references wrong adoption
    with pytest.raises(ValueError, match="PROJECTION_ROLLBACK_ADOPTION_MISMATCH"):
        tampered_rb = copy.deepcopy(rollback)
        tampered_rb["adoption_id"] = "other_adoption_id"
        h2, i2 = canonical_rollback_identity(tampered_rb)
        tampered_rb["rollback_hash"] = h2
        tampered_rb["rollback_id"] = i2
        project_adoption_into_planner_budget(
            adopt,
            task_desc="repair record_serialization for user profile",
            rollback=tampered_rb,
        )

    # G10-R3: tampered rollback hash fails closed
    with pytest.raises(ValueError, match="ROLLBACK_CONTENT_ADDRESS_MISMATCH"):
        tampered_hash_rb = copy.deepcopy(rollback)
        tampered_hash_rb["rollback_hash"] = "tampered_hash_12345"
        validate_learning_policy_rollback(tampered_hash_rb)
