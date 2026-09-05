"""Deterministic, observational replay for learning-effectiveness scorecards.

This module deliberately has no producer, persistence, adaptation, or authority
dependencies.  It consumes identity-complete attempt rows and returns ordinary
Python data that can be independently inspected or serialized by a caller.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from math import isfinite
from typing import Any, Iterable, Mapping

SCHEMA = "nexus.learning_effectiveness_measurement.v1"
_IDENTITY = (
    "task_fingerprint",
    "task_id",
    "attempt_id",
    "attempt_index",
    "action_id",
)
_REQUIRED = _IDENTITY + (
    "source_revision",
    "source_tree",
    "verifier_status",
    "verifier_artifact",
    "verifier_artifact_hash",
    "verifier_receipt",
    "memory_arm",
    "retrieved_lesson_ids",
    "applied_attributed_lesson_ids",
    "terminal_outcome",
    "measured_elapsed_seconds",
    "intervention_events",
    "intervention_count",
    "forbidden_strategy_identity",
    "forbidden_strategy_violation_event",
    "missingness_reasons",
    "ineligibility_reasons",
)
_PASS_STATUSES = {"pass", "passed", "success", "succeeded", "qualified"}
_FAIL_STATUSES = {"fail", "failed", "failure", "blocked", "rejected"}
_PASS_OUTCOMES = {"SUCCESS", "SUCCEEDED", "PASSED", "QUALIFIED"}
_FAIL_OUTCOMES = {"FAIL", "FAILED", "FAILURE", "BLOCKED", "REJECTED"}


class ReplayContractError(ValueError):
    """Raised when rows cannot be safely replayed under the v1 contract."""


def _text(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def _string_sequence(field: str, value: Any) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        raise ReplayContractError(f"{field} must be a list or tuple of non-empty strings")
    if any(not isinstance(item, str) or not item.strip() for item in value):
        raise ReplayContractError(f"{field} must contain only non-empty strings")
    normalized = tuple(sorted(item.strip() for item in value))
    if len(normalized) != len(set(normalized)):
        raise ReplayContractError(f"{field} must not contain duplicates")
    return normalized


def _has_missing_reason(reasons: tuple[str, ...], field: str) -> bool:
    return any(reason == field or reason.startswith(f"{field}:") for reason in reasons)


@dataclass(frozen=True)
class AttemptRow:
    """An immutable, canonical copy of one identity-complete attempt row."""

    values: tuple[tuple[str, Any], ...]

    @classmethod
    def from_mapping(
        cls,
        raw: Mapping[str, Any],
        *,
        allow_missing_verifier_evidence: bool = False,
    ) -> "AttemptRow":
        if not isinstance(raw, Mapping):
            raise ReplayContractError("attempt row must be a mapping")
        missing = [key for key in _REQUIRED if key not in raw]
        if missing:
            raise ReplayContractError("missing required fields: " + ",".join(missing))
        identity = {key: raw[key] for key in _IDENTITY}
        if any(
            not isinstance(identity[key], str) or not identity[key].strip()
            for key in _IDENTITY
            if key != "attempt_index"
        ):
            raise ReplayContractError("identity fields must be non-empty strings")
        if (
            isinstance(identity["attempt_index"], bool)
            or not isinstance(identity["attempt_index"], int)
            or identity["attempt_index"] < 0
        ):
            raise ReplayContractError("attempt_index must be a non-negative integer")
        if _text(raw["memory_arm"]) not in {"memory_off", "memory_on"}:
            raise ReplayContractError("memory_arm must be memory_off or memory_on")
        missingness = _string_sequence("missingness_reasons", raw["missingness_reasons"])
        strict_text_fields = (
            "source_revision",
            "source_tree",
            "forbidden_strategy_identity",
        )
        verifier_evidence_fields = (
            "verifier_artifact",
            "verifier_artifact_hash",
            "verifier_receipt",
        )
        if any(
            not isinstance(raw[field], str) or not raw[field].strip()
            for field in strict_text_fields
        ):
            raise ReplayContractError(
                "source and forbidden-strategy identity fields must be non-empty strings"
            )
        for field in verifier_evidence_fields:
            if isinstance(raw[field], str) and raw[field].strip():
                continue
            if allow_missing_verifier_evidence and _has_missing_reason(missingness, field):
                continue
            raise ReplayContractError(f"{field} must be a non-empty string")
        status = _text(raw["verifier_status"]).lower()
        outcome = _text(raw["terminal_outcome"]).upper()
        if status not in _PASS_STATUSES | _FAIL_STATUSES:
            raise ReplayContractError("verifier_status is invalid")
        if outcome not in _PASS_OUTCOMES | _FAIL_OUTCOMES:
            raise ReplayContractError("terminal_outcome is invalid")
        if (status in _PASS_STATUSES) != (outcome in _PASS_OUTCOMES):
            raise ReplayContractError("verifier_status and terminal_outcome disagree")
        ineligibility = _string_sequence("ineligibility_reasons", raw["ineligibility_reasons"])
        elapsed = raw["measured_elapsed_seconds"]
        if elapsed is None:
            if not _has_missing_reason(missingness, "measured_elapsed_seconds"):
                raise ReplayContractError(
                    "missing elapsed time requires an explicit missingness reason"
                )
        elif (
            isinstance(elapsed, bool)
            or not isinstance(elapsed, (int, float))
            or not isfinite(elapsed)
            or elapsed < 0
        ):
            raise ReplayContractError("measured_elapsed_seconds must be finite and non-negative")
        count = raw["intervention_count"]
        events = raw["intervention_events"]
        if count is None or events is None:
            if not _has_missing_reason(
                missingness, "intervention_count"
            ) or not _has_missing_reason(missingness, "intervention_events"):
                raise ReplayContractError(
                    "missing intervention telemetry requires an explicit missingness reason"
                )
            if count is not None or events is not None:
                raise ReplayContractError(
                    "intervention_count and intervention_events must be missing together"
                )
            normalized_events: tuple[str, ...] | None = None
        else:
            if isinstance(count, bool) or not isinstance(count, int) or count < 0:
                raise ReplayContractError("intervention_count must be a non-negative integer")
            normalized_events = _string_sequence("intervention_events", events)
            if count != len(normalized_events):
                raise ReplayContractError(
                    "intervention_count must equal intervention_events length"
                )
        violation = raw["forbidden_strategy_violation_event"]
        if violation is None:
            if not _has_missing_reason(missingness, "forbidden_strategy_violation_event"):
                raise ReplayContractError(
                    "missing violation telemetry requires an explicit missingness reason"
                )
        elif not isinstance(violation, bool):
            raise ReplayContractError("forbidden_strategy_violation_event must be a bool")
        values = {key: deepcopy(raw[key]) for key in _REQUIRED}
        for field in (
            "task_fingerprint",
            "task_id",
            "attempt_id",
            "action_id",
        ) + strict_text_fields:
            values[field] = _text(values[field])
        for field in verifier_evidence_fields:
            values[field] = _text(values[field]) or None
        values["memory_arm"] = _text(values["memory_arm"])
        values["verifier_status"] = status
        values["terminal_outcome"] = outcome
        values["retrieved_lesson_ids"] = _string_sequence(
            "retrieved_lesson_ids", values["retrieved_lesson_ids"]
        )
        values["applied_attributed_lesson_ids"] = _string_sequence(
            "applied_attributed_lesson_ids", values["applied_attributed_lesson_ids"]
        )
        if not set(values["applied_attributed_lesson_ids"]).issubset(
            values["retrieved_lesson_ids"]
        ):
            raise ReplayContractError(
                "applied_attributed_lesson_ids must be a subset of retrieved_lesson_ids"
            )
        values["missingness_reasons"] = missingness
        values["ineligibility_reasons"] = ineligibility
        values["intervention_events"] = normalized_events
        return cls(tuple(sorted(values.items())))

    def to_dict(self) -> dict[str, Any]:
        result = deepcopy(dict(self.values))
        for field in (
            "retrieved_lesson_ids",
            "applied_attributed_lesson_ids",
            "missingness_reasons",
            "ineligibility_reasons",
            "intervention_events",
        ):
            if result[field] is not None:
                result[field] = list(result[field])
        return result

    def identity(self) -> tuple[Any, ...]:
        row = dict(self.values)
        return tuple(row[key] for key in _IDENTITY)


def normalize_attempt_row(row: Mapping[str, Any]) -> AttemptRow:
    return AttemptRow.from_mapping(row)


def _status(row: Mapping[str, Any]) -> str:
    return _text(row.get("verifier_status")).lower()


def _passed(row: Mapping[str, Any]) -> bool:
    return _status(row) in {"pass", "passed", "success", "succeeded", "qualified"}


def _qualified_pass(row: Mapping[str, Any]) -> bool:
    return _passed(row) and all(
        _text(row.get(field))
        for field in ("verifier_artifact", "verifier_artifact_hash", "verifier_receipt")
    )


def _failed(row: Mapping[str, Any]) -> bool:
    return _status(row) in {"fail", "failed", "failure", "blocked", "rejected"} or _text(
        row.get("terminal_outcome")
    ).upper() in {"FAILED", "BLOCKED", "REJECTED"}


def _row_eligible(row: Mapping[str, Any], *fields: str) -> bool:
    return not row.get("ineligibility_reasons") and all(
        row.get(field) is not None for field in fields
    )


def _row_exclusions(row: Mapping[str, Any], *fields: str) -> list[str]:
    prefix = f"{row['task_fingerprint']}:{row['attempt_id']}"
    reasons = [f"{prefix}:ineligible:{reason}" for reason in row.get("ineligibility_reasons", ())]
    reasons.extend(f"{prefix}:missing:{field}" for field in fields if row.get(field) is None)
    return reasons


def _metric(
    numerator: Any,
    denominator: int,
    eligible: int,
    missing: int,
    exclusions: list[str],
    ceiling: str,
) -> dict[str, Any]:
    return {
        "numerator": numerator,
        "denominator": denominator,
        "eligible": eligible,
        "missing": missing,
        "missing_telemetry": missing,
        "exclusions": sorted(set(exclusions)),
        "claim_ceiling": ceiling,
        "rate": (
            round(numerator / denominator, 4)
            if isinstance(numerator, (int, float)) and denominator
            else None
        ),
    }


def _deduplicate(
    rows: Iterable[Mapping[str, Any]],
    *,
    allow_missing_verifier_evidence: bool = False,
) -> list[AttemptRow]:
    found: dict[tuple[Any, ...], AttemptRow] = {}
    for raw in rows:
        item = AttemptRow.from_mapping(
            raw,
            allow_missing_verifier_evidence=allow_missing_verifier_evidence,
        )
        previous = found.get(item.identity())
        if previous is not None:
            raise ReplayContractError(
                "duplicate or colliding attempt identity: " + repr(item.identity())
            )
        found[item.identity()] = item
    return [found[key] for key in sorted(found, key=repr)]


def _base_metrics(rows: list[AttemptRow]) -> dict[str, Any]:
    data = [row.to_dict() for row in rows]
    task_groups: dict[str, list[dict[str, Any]]] = {}
    for row in data:
        task_groups.setdefault(row["task_fingerprint"], []).append(row)
    failed_candidates = {
        fingerprint: [row for row in group if _failed(row)]
        for fingerprint, group in task_groups.items()
    }
    failed_candidates = {
        fingerprint: group for fingerprint, group in failed_candidates.items() if group
    }
    failed_population = {
        fingerprint: group
        for fingerprint, group in failed_candidates.items()
        if all(_row_eligible(row, "verifier_status", "terminal_outcome") for row in group)
    }
    failed_missing = [
        row
        for fingerprint, group in failed_candidates.items()
        if fingerprint not in failed_population
        for row in group
        if not _row_eligible(row, "verifier_status", "terminal_outcome")
    ]
    recurrence = sum(len(group) > 1 for group in failed_population.values())
    first_pass_cohort = [row for row in data if row["attempt_index"] == 0]
    first_pass = [row for row in first_pass_cohort if _row_eligible(row, "verifier_status")]
    first_pass_missing = [row for row in first_pass_cohort if row not in first_pass]
    task_green = {
        key: min(
            (
                row
                for row in group
                if _qualified_pass(row) and _row_eligible(row, "verifier_status")
            ),
            key=lambda row: row["attempt_index"],
        )
        for key, group in task_groups.items()
        if any(_qualified_pass(row) and _row_eligible(row, "verifier_status") for row in group)
    }
    pass_missing = [
        row
        for fingerprint, group in task_groups.items()
        if fingerprint not in task_green
        for row in group
        if _passed(row) and not _row_eligible(row, "verifier_status")
    ]
    green_rows = list(task_green.values())
    elapsed = [row for row in green_rows if _row_eligible(row, "measured_elapsed_seconds")]
    elapsed_missing = [row for row in green_rows if row not in elapsed]
    time_missing = {row["task_fingerprint"]: row for row in elapsed_missing + pass_missing}
    intervention = [
        row for row in data if _row_eligible(row, "intervention_count", "intervention_events")
    ]
    intervention_missing = [row for row in data if row not in intervention]
    forbidden = [row for row in data if _row_eligible(row, "forbidden_strategy_violation_event")]
    forbidden_missing = [row for row in data if row not in forbidden]
    retrieval = [row for row in data if _row_eligible(row, "retrieved_lesson_ids")]
    retrieval_missing = [row for row in data if row not in retrieval]
    retrieved = [row for row in retrieval if row["retrieved_lesson_ids"]]
    applied = [row for row in retrieved if _row_eligible(row, "applied_attributed_lesson_ids")]
    applied_missing = [
        row for row in data if row.get("retrieved_lesson_ids") and row not in applied
    ]
    attributed = [row for row in applied if row["applied_attributed_lesson_ids"]]
    qualified = [row for row in attributed if _row_eligible(row, "verifier_status")]
    qualified_missing = [
        row for row in data if row.get("applied_attributed_lesson_ids") and row not in qualified
    ]

    def exclusions(items: list[dict[str, Any]], *fields: str) -> list[str]:
        return [reason for item in items for reason in _row_exclusions(item, *fields)]

    return {
        "failure_recurrence": _metric(
            recurrence,
            len(failed_population),
            len(failed_population),
            len({row["task_fingerprint"] for row in failed_missing}),
            exclusions(failed_missing, "verifier_status", "terminal_outcome"),
            "observational recurrence among eligible failed task fingerprints",
        ),
        "first_pass_qualification": _metric(
            sum(_qualified_pass(row) for row in first_pass),
            len(first_pass),
            len(first_pass),
            len(first_pass_missing),
            exclusions(first_pass_missing, "verifier_status"),
            "observational first-attempt qualified verifier outcome",
        ),
        "attempts_to_green": _metric(
            sum(row["attempt_index"] + 1 for row in green_rows),
            len(green_rows),
            len(green_rows),
            len({row["task_fingerprint"] for row in pass_missing}),
            exclusions(pass_missing, "verifier_status"),
            "descriptive attempts through the first complete passing verifier row",
        ),
        "time_to_green": _metric(
            round(sum(row["measured_elapsed_seconds"] for row in elapsed), 4) if elapsed else None,
            len(elapsed),
            len(elapsed),
            len(time_missing),
            exclusions(list(time_missing.values()), "measured_elapsed_seconds"),
            "selected first-qualified-attempt cumulative elapsed seconds",
        ),
        "intervention_rate": _metric(
            sum(row["intervention_count"] > 0 for row in intervention),
            len(intervention),
            len(intervention),
            len(intervention_missing),
            exclusions(intervention_missing, "intervention_count", "intervention_events"),
            "observational intervention incidence",
        ),
        "intervention_count": _metric(
            sum(row["intervention_count"] for row in intervention),
            len(intervention),
            len(intervention),
            len(intervention_missing),
            exclusions(intervention_missing, "intervention_count", "intervention_events"),
            "observational intervention total",
        ),
        "forbidden_strategy_violation_rate": _metric(
            sum(row["forbidden_strategy_violation_event"] for row in forbidden),
            len(forbidden),
            len(forbidden),
            len(forbidden_missing),
            exclusions(forbidden_missing, "forbidden_strategy_violation_event"),
            "observational forbidden-strategy violation incidence",
        ),
        "retrieved_to_applied_to_qualified_useful": {
            "retrieved": _metric(
                len(retrieved),
                len(retrieval),
                len(retrieval),
                len(retrieval_missing),
                exclusions(retrieval_missing, "retrieved_lesson_ids"),
                "observational retrieval among eligible attempts",
            ),
            "applied": _metric(
                len(attributed),
                len(applied),
                len(applied),
                len(applied_missing),
                exclusions(applied_missing, "applied_attributed_lesson_ids"),
                "attributed application among attempts that retrieved lessons",
            ),
            "qualified_useful": _metric(
                sum(_qualified_pass(row) for row in qualified),
                len(qualified),
                len(qualified),
                len(qualified_missing),
                exclusions(qualified_missing, "verifier_status"),
                "qualified verifier evidence among attributed applications",
            ),
        },
    }


def paired_memory_uplift(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Return a separate, fail-closed paired signal; never merge it into base metrics."""
    normalized = _deduplicate(rows, allow_missing_verifier_evidence=True)
    data = [row.to_dict() for row in normalized]
    pairs: dict[str, list[dict[str, Any]]] = {}
    for row in data:
        pairs.setdefault(row["task_fingerprint"], []).append(row)
    valid: list[str] = []
    exclusions: list[str] = []
    missing_telemetry: set[str] = set()
    for fingerprint, candidates in sorted(pairs.items()):
        off_rows = [row for row in candidates if row["memory_arm"] == "memory_off"]
        on_rows = [row for row in candidates if row["memory_arm"] == "memory_on"]
        if len(off_rows) != 1 or len(on_rows) != 1:
            reason = (
                "missing_arm"
                if not off_rows or not on_rows
                else "multiple_attempts_or_duplicate_arm"
            )
            exclusions.append(f"{fingerprint}:{reason}")
            if reason == "missing_arm":
                missing_telemetry.add(fingerprint)
            continue
        off, on = off_rows[0], on_rows[0]
        if off.get("ineligibility_reasons") or on.get("ineligibility_reasons"):
            exclusions.append(f"{fingerprint}:ineligible_arm")
            continue
        if off["task_id"] != on["task_id"]:
            exclusions.append(f"{fingerprint}:task_id_mismatch")
            continue
        if (off["source_revision"], off["source_tree"]) != (
            on["source_revision"],
            on["source_tree"],
        ):
            exclusions.append(f"{fingerprint}:source_identity_mismatch")
            continue
        if off["attempt_index"] != on["attempt_index"]:
            exclusions.append(f"{fingerprint}:attempt_index_mismatch")
            continue
        if any(
            row.get(field) is None
            for row in (off, on)
            for field in (
                "measured_elapsed_seconds",
                "intervention_events",
                "intervention_count",
                "forbidden_strategy_violation_event",
            )
        ):
            exclusions.append(f"{fingerprint}:missing_telemetry")
            missing_telemetry.add(fingerprint)
            continue
        if any(
            not _text(row.get(field))
            for row in (off, on)
            for field in ("verifier_artifact", "verifier_artifact_hash", "verifier_receipt")
        ):
            exclusions.append(f"{fingerprint}:missing_artifact_or_receipt")
            missing_telemetry.add(fingerprint)
            continue
        if not _failed(off) or not _passed(on):
            exclusions.append(f"{fingerprint}:off_fail_on_pass_required")
            continue
        valid.append(fingerprint)
    return {
        "schema": SCHEMA,
        "signal": "paired_memory_off_on_uplift",
        "base_metrics_separate": True,
        "eligible_pairs": len(valid),
        "eligible": len(valid),
        "numerator": len(valid),
        "denominator": len(pairs),
        "missing": len(missing_telemetry),
        "missing_telemetry": len(missing_telemetry),
        "exclusions": exclusions,
        "claim_ceiling": "observational paired replay eligibility only; no causal uplift or adaptation claim",
        "eligible_fingerprints": valid,
        "uplift_claimable": False,
    }


def replay_scorecard(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    normalized = _deduplicate(rows)
    return {
        "schema": SCHEMA,
        "row_count": len(normalized),
        "rows": [row.to_dict() for row in normalized],
        "metrics": _base_metrics(normalized),
        "paired_memory_uplift": paired_memory_uplift([row.to_dict() for row in normalized]),
        "observational_only": True,
        "adaptation_applied": False,
        "authority_effect": False,
        "claim_ceiling": "contract replay correctness over supplied identity-complete rows",
    }


reduce_attempts = replay_scorecard


__all__ = [
    "SCHEMA",
    "AttemptRow",
    "ReplayContractError",
    "normalize_attempt_row",
    "replay_scorecard",
    "reduce_attempts",
    "paired_memory_uplift",
]
