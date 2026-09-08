from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping


def semantic_projection_key(entry: Mapping[str, Any]) -> str:
    """Return a stable identity without mutating the source row."""
    source = _source(entry)
    classification = _semantic_classification(entry)
    summary = _semantic_summary(entry)
    parts = (
        source,
        entry.get("task_id"),
        classification,
        summary,
        entry.get("terminal_outcome"),
        tuple(sorted(str(x) for x in (entry.get("applied_lesson_ids") or []))),
    )
    identity = "semantic:" + "|".join(_norm(item) for item in parts)
    return "pattern:" + hashlib.sha256(identity.encode("utf-8")).hexdigest()[:24]


_INVALIDATING_DISPOSITIONS = frozenset({"contradict", "retire", "quarantine", "quarantined"})


def reduce_learning_episode_validity(
    entries: Iterable[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Reduce canonical episode validity in deterministic ledger/input order.

    Only a later, identity-valid canonical episode with terminal evidence may
    invalidate an already-observed episode that it actually applied.  Unknown
    targets and tampered/substituted invalidation rows are ignored so future
    rows cannot be pre-invalidated by forged evidence.
    """
    from nexus_learning.contracts import validate_nexus_learning_episode

    states: dict[str, dict[str, Any]] = {}
    for position, raw in enumerate(entries):
        if not isinstance(raw, Mapping):
            continue
        entry = dict(raw)
        if str(entry.get("schema") or "") != "nexus.learning_episode.v1":
            continue
        episode_id = str(entry.get("episode_id") or "").strip()
        if not episode_id:
            continue
        try:
            validate_nexus_learning_episode(entry)
        except Exception:
            continue

        states.setdefault(
            episode_id,
            {
                "validity_state": "active",
                "retrieval_eligible": True,
                "invalidated_by_episode_id": "",
                "invalidation_disposition": "",
                "invalidation_evidence_refs": [],
                "invalidation_position": None,
                "invalidation_events": [],
            },
        )
        disposition = str(entry.get("lesson_disposition") or "").strip().lower()
        if disposition not in _INVALIDATING_DISPOSITIONS or not _has_terminal_evidence(entry):
            continue
        targets = [
            str(item).strip()
            for item in (entry.get("applied_lesson_ids") or [])
            if str(item).strip()
        ]
        for target_id in targets:
            # Invalidation is strictly later-evidence-over-prior-evidence.  A
            # control row cannot poison an episode that has not appeared yet.
            if target_id == episode_id or target_id not in states:
                continue
            prior_events = [
                dict(item)
                for item in (states[target_id].get("invalidation_events") or [])
                if isinstance(item, Mapping)
            ]
            event = {
                "invalidated_by_episode_id": episode_id,
                "invalidation_disposition": disposition,
                "invalidation_evidence_refs": _evidence_refs(entry),
                "invalidation_position": position,
            }
            prior_events.append(event)
            states[target_id] = {
                "validity_state": "invalidated",
                "retrieval_eligible": False,
                **event,
                "invalidation_events": prior_events,
            }
    return states


def project_learning_entries(entries: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    materialized = [dict(raw) for raw in entries if isinstance(raw, Mapping)]
    validity = reduce_learning_episode_validity(materialized)
    grouped: dict[str, dict[str, Any]] = {}
    for raw in materialized:
        key = semantic_projection_key(raw)
        event_id = str(raw.get("episode_id") or raw.get("idempotency_key") or "")
        pattern, eligible, reason, disposition = _classify(raw)
        semantic_classification = _semantic_classification(raw)
        seen = str(raw.get("created_at") or raw.get("timestamp") or raw.get("timestamp_utc") or "")
        evidence_refs = _evidence_refs(raw)
        current = grouped.get(key)
        if current is None:
            current = grouped[key] = {
                "projection_key": key,
                "pattern_type": pattern,
                "retrieval_eligible": eligible,
                "qualification_reason": reason,
                "lesson_disposition": disposition,
                "occurrence_count": 0,
                "first_seen": seen,
                "last_seen": seen,
                "source_schemas": [],
                "evidence_refs": [],
                "task_ids": [],
                "episode_ids": [],
                "idempotency_keys": [],
                "_event_ids": set(),
                "task_id": str(raw.get("task_id") or ""),
                "classification": semantic_classification,
                "summary": _semantic_summary(raw),
                "source": _source(raw),
            }
        if event_id and event_id in current["_event_ids"]:
            continue
        if event_id:
            current["_event_ids"].add(event_id)
        try:
            occurrence_increment = max(1, int(raw.get("occurrence_count", 1) or 1))
        except (TypeError, ValueError):
            occurrence_increment = 1
        current["occurrence_count"] += occurrence_increment
        if seen and (not current["first_seen"] or seen < current["first_seen"]):
            current["first_seen"] = seen
        if seen > current["last_seen"]:
            current["last_seen"] = seen
        schema = str(raw.get("source_schema") or raw.get("schema") or "legacy")
        if schema not in current["source_schemas"]:
            current["source_schemas"].append(schema)
        for ref in evidence_refs:
            if ref not in current["evidence_refs"]:
                current["evidence_refs"].append(ref)
        task_id = str(raw.get("task_id") or "")
        if task_id and task_id not in current["task_ids"]:
            current["task_ids"].append(task_id)
        current["retrieval_eligible"] = bool(current["retrieval_eligible"] or eligible)
        if eligible and pattern == "verifier_pass":
            current["qualification_reason"] = "qualified"
            current["lesson_disposition"] = disposition
            current["qualification_status"] = str(raw.get("qualification_status") or "QUALIFIED")
        if raw.get("episode_id") and raw["episode_id"] not in current["episode_ids"]:
            current["episode_ids"].append(str(raw["episode_id"]))
        if raw.get("idempotency_key") and raw["idempotency_key"] not in current["idempotency_keys"]:
            current["idempotency_keys"].append(str(raw["idempotency_key"]))
        for field in (
            "task_id",
            "classification",
            "summary",
            "action",
            "status",
            "reason",
            "topic",
            "capability_name",
            "outcome",
            "gate_passed",
            "provenance",
            "receipt_id",
            "source",
            "producer",
            "terminal_evidence",
            "stages",
            "qualification_status",
        ):
            if field in raw and field not in current:
                current[field] = raw[field]
        current.setdefault("source", _source(raw))
    for row in grouped.values():
        row.pop("_event_ids", None)
        episode_ids = [str(item) for item in row.get("episode_ids", []) if str(item)]
        invalidated_ids = [
            episode_id
            for episode_id in episode_ids
            if validity.get(episode_id, {}).get("validity_state") == "invalidated"
        ]
        active_ids = [episode_id for episode_id in episode_ids if episode_id not in invalidated_ids]
        row["invalidated_episode_ids"] = invalidated_ids
        row["active_episode_ids"] = active_ids
        if invalidated_ids and not active_ids:
            row["retrieval_eligible"] = False
            row["qualification_reason"] = "invalidated_by_later_evidence"
            row["validity_state"] = "invalidated"
        elif invalidated_ids:
            row["validity_state"] = "active_with_invalidated_history"
        else:
            row["validity_state"] = "active" if episode_ids else "unversioned"
        row["invalidation_evidence"] = [
            {"episode_id": episode_id, **validity[episode_id]} for episode_id in invalidated_ids
        ]
    return list(grouped.values())


def write_learning_projection(
    entries: Iterable[Mapping[str, Any]], output_path: Path
) -> dict[str, Any]:
    """Atomically replace a projection file; never rewrites raw input."""
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    rows = project_learning_entries(entries)
    fd, temporary = tempfile.mkstemp(prefix=f".{target.name}.", dir=str(target.parent), text=True)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    except Exception as exc:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        return {"status": "ERROR", "error": str(exc), "path": str(target)}
    return {"status": "PASS", "path": str(target), "row_count": len(rows)}


def _classify(entry: Mapping[str, Any]) -> tuple[str, bool, str, str]:
    classification = _semantic_classification(entry).lower()
    if "correct_abstain" in classification:
        return "correct_abstain", True, "negative_hint_only", "negative_hint"
    if "verifier_pass" in classification:
        if not _has_terminal_evidence(entry):
            return "verifier_pass", False, "missing_terminal_evidence", "unqualified"
        terminal = str(entry.get("terminal_outcome") or "").upper()
        if terminal not in {"SUCCESS", "SUCCEEDED"}:
            return "verifier_pass", False, "terminal_outcome_not_success", "unqualified"
        if str(entry.get("qualification_status") or "").upper() != "QUALIFIED":
            return "verifier_pass", False, "unqualified_terminal_evidence", "unqualified"
        qualification = entry.get("qualification")
        if not isinstance(qualification, Mapping) or not all(
            qualification.get(field)
            for field in ("repeatability", "prevention_rule", "authority_qualification")
        ):
            return "verifier_pass", False, "incomplete_qualification", "unqualified"
        return "verifier_pass", True, "qualified", "shadow"
    # Older retrieval backends use a provenance-backed generic ``success``
    # classification without the canonical verifier fields.  Preserve those
    # records as legacy hints; only an explicit verifier_pass may enter the
    # stricter qualified-success path above.
    if classification in {"pass", "success", "succeeded"}:
        return "legacy_success", True, "legacy_provenance_only", "shadow"
    return classification or "unknown", False, "unclassified", "shadow"


def _evidence_refs(entry: Mapping[str, Any]) -> list[str]:
    refs = entry.get("evidence_refs") or entry.get("evidence_ref") or []
    if isinstance(refs, str):
        refs = [refs]
    evidence = entry.get("terminal_evidence")
    if isinstance(evidence, Mapping):
        refs = list(refs) + [
            evidence.get("receipt"),
            evidence.get("verifier"),
            evidence.get("receipt_id"),
            evidence.get("provenance"),
        ]
    refs = list(refs) + [entry.get("receipt_id"), entry.get("provenance")]
    return sorted(
        {
            str(item)
            for item in refs
            if item and str(item).lower() not in {"receipt:pending", "pending"}
        }
    )


def _has_terminal_evidence(entry: Mapping[str, Any]) -> bool:
    evidence = entry.get("terminal_evidence")
    if not isinstance(evidence, Mapping):
        return False
    receipt = _norm(evidence.get("receipt") or evidence.get("receipt_id"))
    verifier = _norm(evidence.get("verifier"))
    verifier_reference = verifier and verifier not in {
        "fail",
        "failed",
        "pass",
        "passed",
        "success",
        "succeeded",
        "missing",
        "unverified",
        "unknown",
    }
    return bool(
        (receipt and receipt not in {"receipt:pending", "pending"})
        or verifier_reference
        or evidence.get("artifact")
        or evidence.get("artifact_path")
        or evidence.get("evidence_ref")
    )


def _source(entry: Mapping[str, Any]) -> str:
    explicit = str(
        entry.get("source") or entry.get("producer") or entry.get("source_schema") or ""
    ).lower()
    if explicit:
        return explicit
    if any(key in entry for key in ("findings_card_id", "classification", "lesson_id")):
        return "local_heal"
    if "capability_name" in entry:
        return "skills_router"
    return "learn_mode"


def _semantic_classification(entry: Mapping[str, Any]) -> str:
    return str(
        entry.get("classification")
        or entry.get("action")
        or entry.get("capability_name")
        or entry.get("pattern_type")
        or ""
    )


def _semantic_summary(entry: Mapping[str, Any]) -> str:
    value = entry.get("summary") or entry.get("reason") or entry.get("topic")
    if value:
        return str(value)
    outcome = entry.get("outcome")
    if isinstance(outcome, Mapping):
        return json.dumps(dict(outcome), ensure_ascii=False, sort_keys=True)
    return str(outcome or "")


def _norm(value: Any) -> str:
    return " ".join(str(value or "").split()).lower()
