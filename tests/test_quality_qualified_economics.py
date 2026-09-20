"""Tests for cost-at-required-quality economics (Semantic Reflex V2 Wave 1 #22)."""

import pytest

from nexus_learning.effectiveness_measurement import (
    QUALITY_QUALIFIED_ECONOMICS_SCHEMA,
    ReplayContractError,
    compare_workflows_at_required_quality,
)


def _workflow(
    *,
    identity: str,
    revision: str = "rev-1",
    fingerprint: str = "task-1",
    attempts: int,
    qualified: int,
    critical: int = 0,
    semantic: int = 0,
    provider: int = 0,
    model_invocations: int | None,
    provider_invocations: int | None = 0,
    fallback: int | None = 0,
    token_usage: int | None = 0,
    wall_time: float | None = 0.0,
    monetary: float | None = 0.0,
    missing: tuple[str, ...] = (),
    ineligible: tuple[str, ...] = (),
) -> dict:
    row = {
        "workflow_identity": identity,
        "workflow_revision": revision,
        "task_fingerprint": fingerprint,
        "attempt_count": attempts,
        "qualified_success_count": qualified,
        "critical_failure_count": critical,
        "semantic_failure_count": semantic,
        "provider_failure_count": provider,
        "false_allow_count": 0,
        "model_invocation_count": model_invocations,
        "provider_invocation_count": provider_invocations,
        "fallback_count": fallback,
        "token_usage": token_usage,
        "wall_time_seconds": wall_time,
        "monetary_cost_usd": monetary,
        "human_intervention_count": 0,
        "missingness_reasons": list(missing),
        "ineligibility_reasons": list(ineligible),
    }
    return row


def _by_identity(result: dict, identity: str) -> dict:
    return next(row for row in result["rows"] if row["workflow_identity"] == identity)


# AC1: v1.0 donor lesson — cascade avoids expensive calls but finishes below the
# required quality floor. Quality is gated BEFORE cost, so it must not win.
def test_donor_lesson_cheaper_below_floor_never_wins():
    deterministic = _workflow(
        identity="deterministic-only",
        attempts=20,
        qualified=20,
        model_invocations=20,
    )
    cascade = _workflow(
        identity="cascade",
        attempts=20,
        qualified=11,
        semantic=9,
        model_invocations=5,
    )
    result = compare_workflows_at_required_quality(
        [deterministic, cascade],
        required_quality_floor=0.95,
        critical_failure_ceiling=0,
    )
    assert result["schema"] == QUALITY_QUALIFIED_ECONOMICS_SCHEMA
    assert result["observational_only"] is True
    assert result["adaptation_applied"] is False
    assert result["authority_effect"] is False
    assert _by_identity(result, "deterministic-only")["gate_status"] == "QUALITY_QUALIFIED"
    assert _by_identity(result, "cascade")["gate_status"] == "QUALITY_FLOOR_FAILED"
    assert "QUALITY_FLOOR_FAILED" in _by_identity(result, "cascade")["dispositions"]
    assert result["summary"]["cheapest_qualified_workflow"] == "deterministic-only"
    assert result["summary"]["below_floor_count"] == 1


# AC2: among quality-qualified workflows, a strictly cheaper one is comparable.
def test_cheaper_qualified_workflow_is_comparable():
    baseline = _workflow(
        identity="baseline",
        attempts=10,
        qualified=10,
        model_invocations=10,
    )
    cheaper = _workflow(
        identity="cheaper",
        attempts=10,
        qualified=10,
        model_invocations=6,
    )
    result = compare_workflows_at_required_quality(
        [baseline, cheaper],
        required_quality_floor=0.9,
        critical_failure_ceiling=1,
    )
    assert _by_identity(result, "baseline")["gate_status"] == "QUALITY_QUALIFIED"
    assert _by_identity(result, "cheaper")["gate_status"] == "QUALITY_QUALIFIED"
    assert "COST_COMPARABLE" in _by_identity(result, "cheaper")["dispositions"]
    assert result["summary"]["comparable_count"] == 1
    assert result["summary"]["cheapest_qualified_workflow"] == "cheaper"


# AC3: a more expensive workflow is represented as quality-superior when its
# qualified rate is materially higher; it is never forced into a cost loser bin.
def test_expensive_qualified_workflow_with_material_quality_gain_is_superior():
    baseline = _workflow(
        identity="baseline",
        attempts=20,
        qualified=17,
        semantic=3,
        model_invocations=20,
    )
    expensive = _workflow(
        identity="expensive",
        attempts=20,
        qualified=19,
        semantic=1,
        model_invocations=40,
    )
    result = compare_workflows_at_required_quality(
        [baseline, expensive],
        required_quality_floor=0.8,
        critical_failure_ceiling=0,
    )
    assert "QUALITY_SUPERIOR" in _by_identity(result, "expensive")["dispositions"]
    assert result["summary"]["superior_quality_count"] == 1


# AC4: missing cost telemetry is never treated as zero cost — it is an explicit
# insufficient-evidence disposition, never COST_COMPARABLE.
def test_missing_cost_telemetry_is_never_zero():
    baseline = _workflow(
        identity="baseline",
        attempts=10,
        qualified=10,
        model_invocations=10,
    )
    no_cost = _workflow(
        identity="no-cost-data",
        attempts=10,
        qualified=10,
        model_invocations=None,
        missing=("model_invocation_count",),
    )
    result = compare_workflows_at_required_quality(
        [baseline, no_cost],
        required_quality_floor=0.9,
        critical_failure_ceiling=0,
    )
    body = _by_identity(result, "no-cost-data")
    assert "INSUFFICIENT_COST_EVIDENCE" in body["dispositions"]
    assert "COST_COMPARABLE" not in body["dispositions"]
    assert result["summary"]["insufficient_cost_evidence_count"] == 1


# AC5: missing cost telemetry without an explicit missingness reason is a contract error.
def test_missing_cost_telemetry_without_reason_is_contract_error():
    row = _workflow(
        identity="bad",
        attempts=5,
        qualified=5,
        model_invocations=None,
    )
    with pytest.raises(ReplayContractError):
        compare_workflows_at_required_quality(
            [row],
            required_quality_floor=0.8,
            critical_failure_ceiling=0,
        )


# AC6: equal-cost, no material quality gain at or above floor is no incremental value.
def test_equal_quality_and_cost_is_no_incremental_value():
    baseline = _workflow(
        identity="baseline",
        attempts=10,
        qualified=10,
        model_invocations=10,
    )
    equal = _workflow(
        identity="equal",
        attempts=10,
        qualified=10,
        model_invocations=10,
    )
    result = compare_workflows_at_required_quality(
        [baseline, equal],
        required_quality_floor=0.9,
        critical_failure_ceiling=0,
    )
    assert "NO_INCREMENTAL_VALUE" in _by_identity(result, "equal")["dispositions"]
    assert "NO_INCREMENTAL_VALUE" in _by_identity(result, "baseline")["dispositions"]


# AC7: critical-failure ceiling is enforced; exceeding it fails the quality gate.
def test_critical_failure_ceiling_enforced():
    row = _workflow(
        identity="risky",
        attempts=10,
        qualified=10,
        critical=2,
        model_invocations=4,
    )
    result = compare_workflows_at_required_quality(
        [row],
        required_quality_floor=0.9,
        critical_failure_ceiling=1,
    )
    assert _by_identity(result, "risky")["gate_status"] == "QUALITY_FLOOR_FAILED"
    assert "QUALITY_FLOOR_FAILED" in _by_identity(result, "risky")["dispositions"]


# AC8: provider failure is a distinct telemetry axis from semantic failure.
def test_provider_and_semantic_failures_are_distinct():
    row = _workflow(
        identity="mixed",
        attempts=10,
        qualified=8,
        semantic=1,
        provider=1,
        model_invocations=9,
    )
    result = compare_workflows_at_required_quality(
        [row],
        required_quality_floor=0.7,
        critical_failure_ceiling=0,
    )
    quality = _by_identity(result, "mixed")["quality"]
    assert quality["semantic_failure_count"] == 1
    assert quality["provider_failure_count"] == 1


# AC9: ineligible rows are excluded with explicit reasons, never compared.
def test_ineligible_workflow_is_excluded():
    row = _workflow(
        identity="ineligible",
        attempts=10,
        qualified=10,
        model_invocations=2,
        ineligible=("no_source_identity",),
    )
    result = compare_workflows_at_required_quality(
        [row],
        required_quality_floor=0.9,
        critical_failure_ceiling=0,
    )
    body = _by_identity(result, "ineligible")
    assert body["gate_status"] == "INELIGIBLE"
    assert body["exclusions"] == ["ineligible:ineligible:no_source_identity"]


# AC10: explicit quality gate on non-negative finite floor.
def test_invalid_floor_rejected():
    row = _workflow(
        identity="w",
        attempts=1,
        qualified=1,
        model_invocations=1,
    )
    with pytest.raises(ReplayContractError):
        compare_workflows_at_required_quality(
            [row],
            required_quality_floor=-0.1,
            critical_failure_ceiling=0,
        )


# Regression: multi-row aggregation per (workflow, revision) sums counts and never
# drops telemetry across task fingerprint rows.
def test_aggregation_across_task_fingerprints():
    first = _workflow(
        identity="wf",
        revision="r1",
        fingerprint="task-a",
        attempts=4,
        qualified=4,
        model_invocations=4,
        token_usage=100,
    )
    second = _workflow(
        identity="wf",
        revision="r1",
        fingerprint="task-b",
        attempts=6,
        qualified=6,
        model_invocations=6,
        token_usage=200,
    )
    result = compare_workflows_at_required_quality(
        [first, second],
        required_quality_floor=0.9,
        critical_failure_ceiling=0,
    )
    body = next(row for row in result["rows"] if row["workflow_identity"] == "wf")
    assert body["workflow_revision"] == "r1"
    assert body["quality"]["attempt_count"] == 10
    assert body["quality"]["qualified_success_count"] == 10
    assert body["cost"]["model_invocation_count"] == 10
    assert body["cost"]["token_usage"] == 300