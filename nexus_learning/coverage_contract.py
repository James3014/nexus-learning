"""Fail-closed, exact-source-bound learning coverage classification."""

from __future__ import annotations

import hashlib
import inspect
import json
import re
from pathlib import Path
from typing import Any, Mapping, Protocol, TypeAlias, runtime_checkable

from nexus_learning import contracts as learning_experience
from nexus_learning.contracts import CAPABILITY_TAXONOMY

CLAIM_CEILING = "DETERMINISTIC_EXACT_SOURCE_TAXONOMY_EVIDENCE_CLASSIFICATION"
_LEVELS = frozenset({"observed", "missing", "not_applicable"})
_BOOL_FIELDS = (
    "selected",
    "invoked",
    "evidence_present",
    "outcome",
    "gate_passed",
    "persistence",
    "consumer_shadow_use",
    "verifier_proof",
)
_HANDLE_KINDS = frozenset({
    "artifact",
    "consumer_shadow",
    "invocation",
    "persistence",
    "selection",
    "verifier",
})
_HANDLE = re.compile(
    r"^(artifact|consumer_shadow|invocation|persistence|selection|verifier):sha256:[0-9a-f]{64}$"
)
_MISSINGNESS = re.compile(r"^[a-z_]+:(unreported|not_applicable)$")
_LEVEL_BINDING = {
    "W": ("selected", "selection"),
    "F": ("invoked", "invocation"),
    "P": ("verifier_proof", "verifier"),
    "S": ("persistence", "persistence"),
}
_FIELD_HANDLE_KIND = {
    "selected": "selection",
    "invoked": "invocation",
    "evidence_present": "artifact",
    "outcome": "artifact",
    "gate_passed": "verifier",
    "persistence": "persistence",
    "consumer_shadow_use": "consumer_shadow",
    "verifier_proof": "verifier",
}
_OBSERVATION_FIELDS = frozenset(_BOOL_FIELDS) | {
    "evidence_levels",
    "missingness",
    "source_handles",
}
_ROW_FIELDS = _OBSERVATION_FIELDS | {
    "capability",
    "category",
    "claim_ceiling",
    "phases",
    "taxonomy_source_handle",
}
_CONTRACT_FIELDS = {
    "claim_ceiling",
    "rows",
    "schema",
    "source_binding",
    "task_id",
}
_SNAPSHOT_DIGEST = re.compile(r"^[0-9a-f]{64}$")

EvidencePayload: TypeAlias = bytes | Mapping[str, Any]


@runtime_checkable
class EvidenceResolver(Protocol):
    """Injected read-only authority for content-addressed evidence.

    Callers authorize this resolver as the evidence source. State drift is
    rejected, but a malicious resolver that lies about its own snapshot remains
    outside this contract's claim boundary.
    """

    def resolve(self, handle: str) -> EvidencePayload | None:
        """Return exact content without changing resolver state."""

    def snapshot_digest(self) -> str:
        """Return a deterministic digest of all resolver-visible state."""


EvidenceResolverInput: TypeAlias = EvidenceResolver | Mapping[str, EvidencePayload]


class _MappingEvidenceResolver:
    def __init__(self, store: Mapping[str, EvidencePayload]) -> None:
        self._store = store

    def resolve(self, handle: str) -> EvidencePayload | None:
        return self._store.get(handle)

    def snapshot_digest(self) -> str:
        try:
            entries = [
                (handle, hashlib.sha256(_canonical_evidence(self._store[handle])).hexdigest())
                for handle in sorted(self._store)
            ]
            payload = json.dumps(entries, separators=(",", ":"), ensure_ascii=True).encode()
        except Exception as exc:
            raise CoverageContractError("mapping resolver snapshot failed") from exc
        return hashlib.sha256(payload).hexdigest()


class CoverageContractError(ValueError):
    """Raised when coverage evidence is incomplete, contradictory, or unbound."""


def _taxonomy_payload(taxonomy: Mapping[str, Mapping[str, Any]]) -> str:
    normalized = {
        str(name): {"category": str(meta["category"]), "phases": list(meta["phases"])}
        for name, meta in sorted(taxonomy.items())
    }
    return json.dumps(normalized, ensure_ascii=True, separators=(",", ":"), sort_keys=True)


def _source_binding() -> dict[str, str]:
    source = Path(inspect.getsourcefile(learning_experience) or "").resolve()
    return {
        "source_path": "nexus/contracts/learning_experience.py",
        "source_symbol": "CAPABILITY_TAXONOMY",
        "source_revision": hashlib.sha256(source.read_bytes()).hexdigest(),
        "taxonomy_sha256": hashlib.sha256(
            _taxonomy_payload(CAPABILITY_TAXONOMY).encode()
        ).hexdigest(),
    }


def _taxonomy_handle(capability: str) -> str:
    return f"nexus_learning.contracts:CAPABILITY_TAXONOMY[{capability}]"


def _handle_kind(handle: str) -> str:
    if not isinstance(handle, str) or not _HANDLE.fullmatch(handle):
        _fail("source handle authority invalid")
    kind = handle.split(":", 1)[0]
    if kind not in _HANDLE_KINDS:
        _fail("source handle authority invalid")
    return kind


def _canonical_evidence(value: bytes | Mapping[str, Any]) -> bytes:
    if isinstance(value, bytes):
        return value
    if not isinstance(value, Mapping):
        _fail("evidence resolver returned unsupported value")
    try:
        return json.dumps(
            dict(value),
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode()
    except (TypeError, ValueError) as exc:
        raise CoverageContractError("evidence resolver returned non-canonical record") from exc


def _resolver_snapshot(resolver: EvidenceResolver) -> str:
    try:
        digest = resolver.snapshot_digest()
    except Exception as exc:
        raise CoverageContractError("evidence resolver snapshot failed") from exc
    if not isinstance(digest, str) or not _SNAPSHOT_DIGEST.fullmatch(digest):
        _fail("evidence resolver snapshot digest invalid")
    return digest


def prepare_evidence_resolver(
    resolver: EvidenceResolverInput | None,
) -> tuple[EvidenceResolver | None, str | None]:
    if resolver is None:
        return None, None
    if isinstance(resolver, Mapping):
        active: EvidenceResolver = _MappingEvidenceResolver(resolver)
    elif isinstance(resolver, EvidenceResolver):
        active = resolver
    else:
        _fail("evidence resolver interface invalid")
    first = _resolver_snapshot(active)
    if _resolver_snapshot(active) != first:
        _fail("evidence resolver snapshot is nondeterministic")
    return active, first


def verify_evidence_resolver_unchanged(
    resolver: EvidenceResolver | None,
    before: str | None,
) -> None:
    if resolver is None:
        if before is not None:
            _fail("evidence resolver snapshot state invalid")
        return
    after = _resolver_snapshot(resolver)
    if _resolver_snapshot(resolver) != after:
        _fail("evidence resolver snapshot is nondeterministic")
    if before != after:
        _fail("evidence resolver state changed during validation")


def _resolver_value(resolver: EvidenceResolver | None, handle: str) -> EvidencePayload:
    if resolver is None:
        _fail("evidence resolver is required")
    try:
        value = resolver.resolve(handle)
    except Exception as exc:
        raise CoverageContractError("evidence resolver lookup failed") from exc
    if value is None:
        _fail("evidence resolver could not resolve handle")
    return value


def resolve_evidence_handle(
    resolver: EvidenceResolver | None,
    handle: str,
    *,
    expected_kind: str | None = None,
) -> bytes | Mapping[str, Any]:
    """Resolve without mutation and verify the handle against exact content bytes."""
    kind = _handle_kind(handle)
    if expected_kind is not None and kind != expected_kind:
        _fail("evidence handle kind mismatch")
    value = _resolver_value(resolver, handle)
    payload = _canonical_evidence(value)
    digest = handle.rsplit(":", 1)[1]
    if hashlib.sha256(payload).hexdigest() != digest:
        _fail("evidence resolver hash mismatch")
    return value


def build_coverage_contract(
    observations: Mapping[str, Mapping[str, Any]] | None = None,
    *,
    task_id: str | None = None,
    evidence_resolver: EvidenceResolverInput | None = None,
) -> dict[str, Any]:
    """Build one stable row for each capability currently in the source taxonomy."""
    if observations is None:
        observations = {}
    if not isinstance(observations, Mapping):
        raise CoverageContractError("observations must be a mapping")
    expected = set(CAPABILITY_TAXONOMY)
    unknown = set(observations) - expected
    if unknown:
        raise CoverageContractError(f"unknown capability: {sorted(unknown)}")
    rows = []
    for capability in sorted(expected):
        meta = CAPABILITY_TAXONOMY[capability]
        supplied = observations.get(capability) or {}
        if not isinstance(supplied, Mapping):
            raise CoverageContractError("capability observation must be a mapping")
        raw = dict(supplied)
        if set(raw) - _OBSERVATION_FIELDS:
            raise CoverageContractError("unbounded evidence fields are forbidden")
        values: dict[str, Any] = {field: raw.get(field) for field in _BOOL_FIELDS}
        missingness = list(raw.get("missingness") or [])
        for field in _BOOL_FIELDS:
            if values[field] is None and f"{field}:unreported" not in missingness:
                missingness.append(f"{field}:unreported")
        levels = dict(raw.get("evidence_levels") or {})
        levels = {key: levels.get(key, "missing") for key in ("W", "F", "P", "S")}
        handles = sorted(raw.get("source_handles") or [])
        rows.append({
            "capability": capability,
            "category": meta["category"],
            "phases": list(meta["phases"]),
            "taxonomy_source_handle": _taxonomy_handle(capability),
            **values,
            "evidence_levels": levels,
            "source_handles": handles,
            "missingness": sorted(set(str(item) for item in missingness)),
            "claim_ceiling": CLAIM_CEILING,
        })
    normalized_task_id = task_id.strip() if isinstance(task_id, str) else None
    if task_id is not None and not normalized_task_id:
        raise CoverageContractError("task identity invalid")
    contract = {
        "schema": "nexus.learning_coverage_contract.v1",
        "source_binding": _source_binding(),
        "task_id": normalized_task_id,
        "rows": rows,
        "claim_ceiling": CLAIM_CEILING,
    }
    validate_coverage_contract(contract, evidence_resolver=evidence_resolver)
    return contract


def _fail(message: str) -> None:
    raise CoverageContractError(message)


def validate_coverage_contract(
    contract: Mapping[str, Any],
    *,
    evidence_resolver: EvidenceResolverInput | None = None,
) -> None:
    active_resolver, resolver_snapshot = prepare_evidence_resolver(evidence_resolver)
    try:
        _validate_coverage_contract(contract, evidence_resolver=active_resolver)
    finally:
        verify_evidence_resolver_unchanged(active_resolver, resolver_snapshot)


def _validate_coverage_contract(
    contract: Mapping[str, Any],
    *,
    evidence_resolver: EvidenceResolver | None,
) -> None:
    if not isinstance(contract, Mapping) or set(contract) != _CONTRACT_FIELDS:
        _fail("contract shape invalid")
    if contract.get("schema") != "nexus.learning_coverage_contract.v1":
        _fail("schema invalid")
    if contract.get("claim_ceiling") != CLAIM_CEILING:
        _fail("claim ceiling invalid")
    binding = contract.get("source_binding")
    if not isinstance(binding, Mapping) or dict(binding) != _source_binding():
        _fail("source binding invalid or stale")
    task_id = contract.get("task_id")
    if task_id is not None and (
        not isinstance(task_id, str) or not task_id.strip() or len(task_id) > 256
    ):
        _fail("task identity invalid")
    rows = contract.get("rows")
    if not isinstance(rows, list):
        _fail("rows invalid")
    names = [row.get("capability") for row in rows if isinstance(row, Mapping)]
    if len(names) != len(set(names)):
        _fail("duplicate capability")
    if names != sorted(CAPABILITY_TAXONOMY) or set(names) != set(CAPABILITY_TAXONOMY):
        _fail("unknown or missing capability")
    for row in rows:
        if (
            not isinstance(row, Mapping)
            or set(row) != _ROW_FIELDS
            or row.get("claim_ceiling") != CLAIM_CEILING
        ):
            _fail("row malformed")
        capability = row["capability"]
        meta = CAPABILITY_TAXONOMY[capability]
        if row.get("category") != meta["category"] or tuple(row.get("phases", ())) != tuple(
            meta["phases"]
        ):
            _fail("source taxonomy mismatch")
        if row.get("taxonomy_source_handle") != _taxonomy_handle(capability):
            _fail("source taxonomy handle mismatch")
        if any(field not in row for field in _BOOL_FIELDS):
            _fail("boolean evidence missing")
        if any(not isinstance(row.get(field), (bool, type(None))) for field in _BOOL_FIELDS):
            _fail("boolean evidence malformed")
        levels = row.get("evidence_levels")
        if (
            not isinstance(levels, Mapping)
            or set(levels) != {"W", "F", "P", "S"}
            or any(level not in _LEVELS for level in levels.values())
        ):
            _fail("evidence level malformed")
        handles = row.get("source_handles")
        if not isinstance(handles, list) or handles != sorted(set(handles)):
            _fail("source handle list malformed")
        bound_fields: set[str] = set()
        for handle in handles:
            value = resolve_evidence_handle(evidence_resolver, handle)
            if not isinstance(value, Mapping) or set(value) != {
                "capability",
                "kind",
                "schema",
                "task_id",
                "values",
            }:
                _fail("evidence resolver record shape invalid")
            kind = _handle_kind(handle)
            if (
                value.get("schema") != "nexus.learning_coverage_evidence.v1"
                or value.get("kind") != kind
                or value.get("capability") != capability
            ):
                _fail("evidence capability or kind binding invalid")
            if task_id is None or value.get("task_id") != task_id:
                _fail("evidence task binding invalid")
            record_values = value.get("values")
            if not isinstance(record_values, Mapping) or not record_values:
                _fail("evidence lifecycle values invalid")
            for field, observed_value in record_values.items():
                if (
                    field not in _FIELD_HANDLE_KIND
                    or _FIELD_HANDLE_KIND[field] != kind
                    or not isinstance(observed_value, bool)
                    or row[field] is not observed_value
                    or field in bound_fields
                ):
                    _fail("evidence lifecycle binding invalid")
                bound_fields.add(field)
        missingness = row.get("missingness")
        if (
            not isinstance(missingness, list)
            or missingness != sorted(set(missingness))
            or any(
                not isinstance(item, str) or not _MISSINGNESS.fullmatch(item)
                for item in missingness
            )
        ):
            _fail("missingness malformed")
        if row["invoked"] is True and row["selected"] is not True:
            _fail("transition invoked without selected")
        if row["evidence_present"] is True and row["invoked"] is not True:
            _fail("transition evidence without invoked")
        if row["outcome"] is True and row["evidence_present"] is not True:
            _fail("outcome without evidence")
        if row["outcome"] is True and row["verifier_proof"] is not True:
            _fail("outcome without verifier proof")
        if row["verifier_proof"] is True and row["evidence_present"] is not True:
            _fail("verifier proof without evidence")
        if row["gate_passed"] is True and row["outcome"] is not True:
            _fail("gate passed without verified outcome")
        if row["persistence"] is True and row["evidence_present"] is not True:
            _fail("persistence without evidence")
        if row["consumer_shadow_use"] is True and row["persistence"] is not True:
            _fail("consumer shadow use without persistence")
        for field in _FIELD_HANDLE_KIND:
            if row[field] is not None and field not in bound_fields:
                _fail(f"evidence for {field} is not source bound")
            reason = f"{field}:unreported"
            if row[field] is None and reason not in missingness:
                _fail(f"missingness required for {field}")
            if row[field] is not None and reason in missingness:
                _fail(f"contradictory missingness for {field}")
        for level, (field, _handle_kind_name) in _LEVEL_BINDING.items():
            expected_level = "observed" if field in bound_fields else "missing"
            if levels[level] != expected_level:
                _fail(f"evidence level {level} is not lifecycle bound")
