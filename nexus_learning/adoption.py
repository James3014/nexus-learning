"""Governed learning-policy adoption built from a memory on/off replay scorecard.

The pipeline gates a replay scorecard (``effectiveness_measurement.replay_scorecard``)
through a required-quality check, a paired memory uplift check, and the contract
validator, then emits ADOPT, DEFER or REJECT. An ADOPT produces an adoption
artifact the Nexus-new loader reads. This module is advisory: it never mutates
routes, planners, or workers, and every result carries ``authority_effect: False``.
"""

from __future__ import annotations

import contextlib
import json
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from nexus_learning.contracts import (
    build_learning_policy_adoption,
    build_learning_policy_recommendation,
    build_nexus_learning_episode,
    evaluate_learning_policy_recommendation,
    paired_memory_uplift_observed,
    validate_learning_policy_adoption,
    validate_learning_policy_rollback,
)
from nexus_learning.effectiveness_measurement import (
    SCHEMA as REPLAY_SCHEMA,
)
from nexus_learning.effectiveness_measurement import (
    compare_workflows_at_required_quality,
)
from nexus_learning.experiment_integrity import (
    CALIBRATED,
    TERMINAL_PASS,
    validate_experiment_integrity,
)
from nexus_learning.lessons import EVIDENCE_ORIGIN_PHYSICAL
from nexus_learning.state_root import LearningStateRoot

ADOPTION_DECISION_ADOPT = "ADOPT"
ADOPTION_DECISION_DEFER = "DEFER"
ADOPTION_DECISION_REJECT = "REJECT"
ADOPTION_PIPELINE_SCHEMA = "nexus.learning_adoption_pipeline.v1"

REASON_SCORECARD_INVALID = "ADOPTION_SCORECARD_INVALID"
REASON_SIMULATED_EVIDENCE = "ADOPTION_SIMULATED_EVIDENCE"
REASON_SOURCE_REVISION_MISMATCH = "ADOPTION_SOURCE_REVISION_MISMATCH"
REASON_QUALITY_GATE_NOT_PASSED = "ADOPTION_QUALITY_GATE_NOT_PASSED"
REASON_NO_POSITIVE_UPLIFT = "ADOPTION_NO_POSITIVE_UPLIFT"
REASON_VALIDATION_NOT_PASSED = "ADOPTION_VALIDATION_NOT_PASSED"

VALIDATION_PASS_DISPOSITION = "VALIDATED_FOR_ADOPTION_CONSIDERATION"

# Quality-gate workflow identities derived from the replay ``memory_arm`` key.
_OFF_WORKFLOW = "nexus_memory_off"
_ON_WORKFLOW = "nexus_memory_on"
_QUALITY_TASK_KEY = "memory_ab_scorecard"
_QUALITY_UNMEASURED_FIELDS = (
    "semantic_failure_count",
    "provider_failure_count",
    "false_allow_count",
    "model_invocation_count",
    "provider_invocation_count",
    "fallback_count",
    "token_usage",
    "human_intervention_count",
    "monetary_cost_usd",
    "wall_time_seconds",
)
_PASS_STATUSES = frozenset({"pass", "passed", "success", "succeeded", "qualified"})
_FAIL_STATUSES = frozenset({"fail", "failed", "failure", "blocked", "rejected"})
_DEFAULT_CURRENT_POLICY = {"memory_retrieval": "shadow"}
_DEFAULT_ROLLBACK_TARGET = {"target_state": {"memory_retrieval": "shadow"}}


def _is_qualified_pass(row: Mapping[str, Any]) -> bool:
    return (
        str(row.get("verifier_status")) in _PASS_STATUSES
        and bool(row.get("verifier_artifact"))
        and bool(row.get("verifier_artifact_hash"))
        and bool(row.get("verifier_receipt"))
    )


def _scorecard_problem(scorecard: Any) -> str | None:
    if not isinstance(scorecard, Mapping):
        return "not_a_mapping"
    if scorecard.get("schema") != REPLAY_SCHEMA:
        return "schema"
    rows = scorecard.get("rows")
    if not isinstance(rows, (list, tuple)) or not rows:
        return "rows"
    if not all(isinstance(row, Mapping) for row in rows):
        return "row_type"
    if not isinstance(scorecard.get("paired_memory_uplift"), Mapping):
        return "paired_memory_uplift"
    return None


def _result(
    decision: str,
    reason_codes: list[str],
    *,
    paired: Mapping[str, Any] | None = None,
    quality_gate: Mapping[str, Any] | None = None,
    recommendation: Mapping[str, Any] | None = None,
    validation: Mapping[str, Any] | None = None,
    adoption: Mapping[str, Any] | None = None,
    rollback: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "schema": ADOPTION_PIPELINE_SCHEMA,
        "decision": decision,
        "reason_codes": list(reason_codes),
        "quality_gate": dict(quality_gate) if quality_gate is not None else None,
        "paired_memory_uplift": dict(paired) if paired is not None else None,
        "recommendation": dict(recommendation) if recommendation is not None else None,
        "validation": dict(validation) if validation is not None else None,
        "adoption": dict(adoption) if adoption is not None else None,
        "rollback": dict(rollback) if rollback is not None else None,
        "advisory_only": True,
        "authority_effect": False,
    }


def _quality_workflow_rows(
    rows: list[dict[str, Any]], source_revision: str
) -> list[dict[str, Any]]:
    """Aggregate replay rows per memory arm into the quality-gate workflow shape.

    Qualified success = verifier pass with artifact, hash and receipt present.
    Critical failure = verifier fail/blocked/rejected. Cost and other quality
    telemetry the replay does not carry is declared missing, never zero.
    """
    units: list[dict[str, Any]] = []
    for arm, workflow in (("memory_off", _OFF_WORKFLOW), ("memory_on", _ON_WORKFLOW)):
        arm_rows = [row for row in rows if row.get("memory_arm") == arm]
        if not arm_rows:
            continue
        unit: dict[str, Any] = {
            "workflow_identity": workflow,
            "workflow_revision": source_revision,
            "task_fingerprint": _QUALITY_TASK_KEY,
            "attempt_count": len(arm_rows),
            "qualified_success_count": sum(1 for row in arm_rows if _is_qualified_pass(row)),
            "critical_failure_count": sum(
                1 for row in arm_rows if str(row.get("verifier_status")) in _FAIL_STATUSES
            ),
            "missingness_reasons": list(_QUALITY_UNMEASURED_FIELDS),
            "ineligibility_reasons": sorted(
                {
                    f"{row.get('task_fingerprint')}:{reason}"
                    for row in arm_rows
                    for reason in row.get("ineligibility_reasons", ())
                }
            ),
        }
        for field in _QUALITY_UNMEASURED_FIELDS:
            unit[field] = None
        units.append(unit)
    return units


def _arm_summary(
    arm_rows: list[dict[str, Any]], pair_row: Mapping[str, Any], fingerprint: str
) -> dict[str, Any]:
    elapsed = [
        float(row["measured_elapsed_seconds"])
        for row in arm_rows
        if row.get("measured_elapsed_seconds") is not None
    ]
    return {
        "task_fingerprint": fingerprint,
        "task_id": pair_row["task_id"],
        "attempt_id": pair_row["attempt_id"],
        "verifier_status": pair_row["verifier_status"],
        "artifact": pair_row["verifier_artifact"],
        "artifact_hash": pair_row["verifier_artifact_hash"],
        "receipt": pair_row["verifier_receipt"],
        "arm_attempt_count": len(arm_rows),
        "arm_success_count": sum(1 for row in arm_rows if _is_qualified_pass(row)),
        "arm_mean_elapsed_seconds": (round(sum(elapsed) / len(elapsed), 4) if elapsed else None),
    }


def _episode_for_row(row: Mapping[str, Any]) -> dict[str, Any]:
    key = "|".join(
        str(row[field])
        for field in (
            "task_fingerprint",
            "task_id",
            "attempt_index",
            "attempt_id",
            "action_id",
            "memory_arm",
        )
    )
    return build_nexus_learning_episode(
        task_id=str(row["task_id"]),
        attempt_id=str(row["attempt_id"]),
        action_id=str(row["action_id"]),
        source="memory_ab_scorecard",
        terminal_outcome=str(row["terminal_outcome"]),
        terminal_evidence={
            "verifier_status": row["verifier_status"],
            "verifier_artifact": row["verifier_artifact"],
            "verifier_artifact_hash": row["verifier_artifact_hash"],
            "receipt": row["verifier_receipt"],
        },
        retrieved_lesson_ids=row.get("retrieved_lesson_ids") or (),
        applied_lesson_ids=row.get("applied_attributed_lesson_ids") or (),
        lesson_disposition="shadow",
        idempotency_key=key,
    )


def _decide(
    scorecard: Any,
    *,
    source_revision: str,
    runtime_identity: str,
    required_quality_floor: float,
    critical_failure_ceiling: int,
    validator_identity: str,
    owner_authority_reference: str,
    experiment_integrity: dict[str, Any] | None,
    task_fingerprint: str,
    current_policy: dict[str, Any] | None,
    rollback_target: dict[str, Any] | None,
) -> dict[str, Any]:
    if _scorecard_problem(scorecard) is not None:
        return _result(ADOPTION_DECISION_REJECT, [REASON_SCORECARD_INVALID])
    paired = scorecard["paired_memory_uplift"]
    rows = [dict(row) for row in scorecard["rows"]]

    # 2. Only physically observed rows may ground a policy.
    if any(row.get("evidence_origin") != EVIDENCE_ORIGIN_PHYSICAL for row in rows):
        return _result(ADOPTION_DECISION_REJECT, [REASON_SIMULATED_EVIDENCE], paired=paired)
    if any(str(row.get("source_revision", "")).strip() != source_revision.strip() for row in rows):
        return _result(ADOPTION_DECISION_DEFER, [REASON_SOURCE_REVISION_MISMATCH], paired=paired)

    # 3. Required-quality gate on the memory-on workflow.
    try:
        quality = compare_workflows_at_required_quality(
            _quality_workflow_rows(rows, source_revision),
            required_quality_floor=required_quality_floor,
            critical_failure_ceiling=critical_failure_ceiling,
        )
    except ValueError:
        return _result(ADOPTION_DECISION_REJECT, [REASON_SCORECARD_INVALID], paired=paired)
    on_unit = next(
        (unit for unit in quality["rows"] if unit["workflow_identity"] == _ON_WORKFLOW),
        None,
    )
    if on_unit is None or on_unit["gate_status"] != "QUALITY_QUALIFIED":
        return _result(
            ADOPTION_DECISION_DEFER,
            [REASON_QUALITY_GATE_NOT_PASSED],
            paired=paired,
            quality_gate=quality,
        )

    # 4. Paired memory uplift on one task with a strictly positive pair count.
    eligible = [str(item) for item in paired.get("eligible_fingerprints") or []]
    if not eligible or int(paired.get("eligible") or 0) <= 0:
        return _result(
            ADOPTION_DECISION_DEFER,
            [REASON_NO_POSITIVE_UPLIFT],
            paired=paired,
            quality_gate=quality,
        )
    fingerprint = task_fingerprint.strip() or sorted(eligible)[0]
    if fingerprint not in eligible:
        return _result(
            ADOPTION_DECISION_DEFER,
            [REASON_NO_POSITIVE_UPLIFT],
            paired=paired,
            quality_gate=quality,
        )
    off_pair = [
        row
        for row in rows
        if row["task_fingerprint"] == fingerprint and row["memory_arm"] == "memory_off"
    ]
    on_pair = [
        row
        for row in rows
        if row["task_fingerprint"] == fingerprint and row["memory_arm"] == "memory_on"
    ]
    if len(off_pair) != 1 or len(on_pair) != 1:
        return _result(
            ADOPTION_DECISION_DEFER,
            [REASON_NO_POSITIVE_UPLIFT],
            paired=paired,
            quality_gate=quality,
        )
    off_row, on_row = off_pair[0], on_pair[0]
    off_arm = _arm_summary(
        [r for r in rows if r["memory_arm"] == "memory_off"], off_row, fingerprint
    )
    on_arm = _arm_summary([r for r in rows if r["memory_arm"] == "memory_on"], on_row, fingerprint)
    on_arm["paired_memory_uplift"] = dict(paired)
    if not paired_memory_uplift_observed(
        {"task_fingerprint": fingerprint, "memory_off": off_arm, "memory_on": on_arm}
    ):
        return _result(
            ADOPTION_DECISION_DEFER,
            [REASON_NO_POSITIVE_UPLIFT],
            paired=paired,
            quality_gate=quality,
        )

    # 5. Optional experiment-integrity binding must be a calibrated PASS.
    if experiment_integrity is not None:
        try:
            validate_experiment_integrity(experiment_integrity)
        except ValueError as exc:
            return _result(ADOPTION_DECISION_DEFER, [str(exc)], paired=paired, quality_gate=quality)
        terminal = experiment_integrity.get("terminal") or {}
        outcome = str(terminal.get("outcome") or "").upper()
        if outcome != TERMINAL_PASS or experiment_integrity.get("calibration_status") != CALIBRATED:
            return _result(
                ADOPTION_DECISION_DEFER,
                ["RECOMMENDATION_NON_POSITIVE_EXPERIMENT_CANNOT_BE_RECOMMENDED"],
                paired=paired,
                quality_gate=quality,
            )

    # 6. Build the content-addressed recommendation.
    evidence_refs = sorted(
        {str(ref) for row in (off_row, on_row) for ref in row.get("evidence_refs", ())}
    )
    uplift_rate = round(int(paired["numerator"]) / int(paired["denominator"]), 4)
    applicable_scope = {
        "memory_arm": "nexus_memory_on",
        "task_fingerprint": fingerprint,
        "source": "local_heal",
    }
    policy_delta = {"memory_retrieval": "enabled", "lesson_sources": ["canonical_lesson"]}
    rollback = dict(rollback_target or _DEFAULT_ROLLBACK_TARGET)
    try:
        recommendation = build_learning_policy_recommendation(
            source_episodes=[_episode_for_row(row) for row in rows],
            source_evidence_refs=evidence_refs,
            source_revision=source_revision,
            runtime_identity=runtime_identity,
            task_fingerprint=fingerprint,
            off_arm=off_arm,
            on_arm=on_arm,
            applicable_scope=applicable_scope,
            recommended_policy_delta=policy_delta,
            current_policy=dict(current_policy or _DEFAULT_CURRENT_POLICY),
            expected_effect=(
                f"paired memory uplift on {fingerprint}: memory_on passed where memory_off "
                f"failed; uplift rate {uplift_rate} across {paired['denominator']} paired tasks"
            ),
            rollback_target=rollback,
            experiment_integrity=experiment_integrity,
        )
    except ValueError as exc:
        return _result(ADOPTION_DECISION_DEFER, [str(exc)], paired=paired, quality_gate=quality)

    # 7. Independent validator.
    validation = evaluate_learning_policy_recommendation(
        recommendation,
        validator_identity=validator_identity,
        current_workspace_revision=source_revision,
        current_runtime_identity=runtime_identity,
    )
    if validation["validation_disposition"] != VALIDATION_PASS_DISPOSITION:
        return _result(
            ADOPTION_DECISION_DEFER,
            [REASON_VALIDATION_NOT_PASSED, *validation["blockers"]],
            paired=paired,
            quality_gate=quality,
            recommendation=recommendation,
            validation=validation,
        )

    # 8. Owner-bound adoption.
    try:
        adoption = build_learning_policy_adoption(
            owner_authority_reference=owner_authority_reference,
            recommendation=recommendation,
            validation=validation,
            source_revision=source_revision,
            adopted_scope=applicable_scope,
            target_policy_delta=policy_delta,
            previous_policy=dict(current_policy or _DEFAULT_CURRENT_POLICY),
            rollback_target=rollback,
        )
        validate_learning_policy_adoption(adoption)
    except ValueError as exc:
        return _result(
            ADOPTION_DECISION_DEFER,
            [str(exc)],
            paired=paired,
            quality_gate=quality,
            recommendation=recommendation,
            validation=validation,
        )
    return _result(
        ADOPTION_DECISION_ADOPT,
        [],
        paired=paired,
        quality_gate=quality,
        recommendation=recommendation,
        validation=validation,
        adoption=adoption,
    )


def build_adoption_from_scorecard(
    scorecard: Any,
    *,
    source_revision: str,
    runtime_identity: str,
    required_quality_floor: float,
    critical_failure_ceiling: int,
    validator_identity: str,
    owner_authority_reference: str,
    state_root: LearningStateRoot | None = None,
    experiment_integrity: dict[str, Any] | None = None,
    task_fingerprint: str = "",
    current_policy: dict[str, Any] | None = None,
    rollback_target: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Decide ADOPT / DEFER / REJECT for a memory on/off replay scorecard.

    Rows must carry ``evidence_origin == "physical"`` and ``memory_arm`` (the
    workflow identity is derived from it). Invalid inputs raise nothing for the
    decision itself; they yield REJECT. When ``state_root`` is given, an ADOPT
    writes the adoption file and a rollback record (if any) writes the rollback
    file; never both for one decision.
    """
    decision = _decide(
        scorecard,
        source_revision=source_revision,
        runtime_identity=runtime_identity,
        required_quality_floor=required_quality_floor,
        critical_failure_ceiling=critical_failure_ceiling,
        validator_identity=validator_identity,
        owner_authority_reference=owner_authority_reference,
        experiment_integrity=experiment_integrity,
        task_fingerprint=task_fingerprint,
        current_policy=current_policy,
        rollback_target=rollback_target,
    )
    if state_root is not None:
        store = AdoptionStore(state_root)
        if decision["adoption"] is not None:
            store.write_adoption(decision["adoption"])
        elif decision["rollback"] is not None:
            store.write_rollback(decision["rollback"])
    return decision


class AdoptionStore:
    """Atomic, idempotent persistence for adoption and rollback artifacts."""

    def __init__(self, state_root: LearningStateRoot) -> None:
        self._state_root = state_root

    def write_adoption(self, adoption: Mapping[str, Any]) -> Path:
        payload = dict(adoption)
        validate_learning_policy_adoption(payload)
        return self._write(self._state_root.adoption_path, payload)

    def write_rollback(self, rollback: Mapping[str, Any]) -> Path:
        payload = dict(rollback)
        validate_learning_policy_rollback(payload)
        return self._write(self._state_root.rollback_path, payload)

    def read_adoption(self) -> dict[str, Any] | None:
        return self._read(self._state_root.adoption_path)

    def read_rollback(self) -> dict[str, Any] | None:
        return self._read(self._state_root.rollback_path)

    @staticmethod
    def _read(path: Path) -> dict[str, Any] | None:
        if not path.exists():
            return None
        loaded = json.loads(path.read_text(encoding="utf-8"))
        return loaded if isinstance(loaded, dict) else None

    @staticmethod
    def _write(target: Path, payload: Mapping[str, Any]) -> Path:
        target.parent.mkdir(parents=True, exist_ok=True)
        data = json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
        fd, temporary = tempfile.mkstemp(prefix=f".{target.name}.", dir=str(target.parent))
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, target)
        except BaseException:
            with contextlib.suppress(OSError):
                os.unlink(temporary)
            raise
        return target


__all__ = [
    "ADOPTION_DECISION_ADOPT",
    "ADOPTION_DECISION_DEFER",
    "ADOPTION_DECISION_REJECT",
    "ADOPTION_PIPELINE_SCHEMA",
    "AdoptionStore",
    "build_adoption_from_scorecard",
]
