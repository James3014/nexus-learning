"""Source-backed observational probes for the learning coverage contract."""

from __future__ import annotations

import re
from typing import Any, Mapping, NoReturn

from nexus_learning.coverage_contract import (
    CoverageContractError,
    EvidenceResolver,
    EvidenceResolverInput,
    prepare_evidence_resolver,
    resolve_evidence_handle,
    validate_coverage_contract,
    verify_evidence_resolver_unchanged,
)

CLAIM_CEILING = "OBSERVATIONAL_SOURCE_BACKED_PROBES_ONLY"
_ARTIFACT_HANDLE = re.compile(r"^artifact:sha256:[0-9a-f]{64}$")
_FINGERPRINT = re.compile(r"^[A-Za-z0-9_.:-]{1,128}$")
_MISSING_MEMORY_SIGNAL = {
    "status": "missing",
    "task_id": None,
    "task_fingerprint": None,
    "memory_off": None,
    "memory_on": None,
    "missingness": [
        "memory_off:unreported",
        "memory_on:unreported",
        "task_fingerprint:unreported",
    ],
}
_PROBE_FIELDS = {
    "claim_ceiling",
    "memory_uplift_signal",
    "probes",
    "schema",
    "source_binding",
}


def _probe_rows(contract: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "capability": row["capability"],
            "source_handles": list(row["source_handles"]),
            "observational_only": True,
            "evidence_levels": dict(row["evidence_levels"]),
        }
        for row in contract["rows"]
        if row["source_handles"] and "observed" in row["evidence_levels"].values()
    ]


def _memory_error(message: str, exc: Exception | None = None) -> NoReturn:
    error = CoverageContractError(f"memory pair {message}")
    if exc is None:
        raise error
    raise error from exc


def _memory_signal(
    evidence: Mapping[str, Any] | None,
    *,
    evidence_resolver: EvidenceResolver | None,
    expected_task_id: str | None,
) -> dict[str, Any]:
    if evidence is None:
        return {
            **_MISSING_MEMORY_SIGNAL,
            "task_id": expected_task_id,
            "missingness": list(_MISSING_MEMORY_SIGNAL["missingness"]),
        }
    if not isinstance(evidence, Mapping) or set(evidence) != {
        "task_id",
        "task_fingerprint",
        "memory_off",
        "memory_on",
    }:
        _memory_error("shape invalid")
    task_id = evidence.get("task_id")
    if expected_task_id is None or task_id != expected_task_id:
        _memory_error("task identity invalid")
    fingerprint = evidence.get("task_fingerprint")
    if not isinstance(fingerprint, str) or not _FINGERPRINT.fullmatch(fingerprint):
        _memory_error("fingerprint identity invalid")
    expected_status = {"memory_off": "fail", "memory_on": "pass"}
    normalized_arms = {}
    for arm_name, status in expected_status.items():
        arm = evidence.get(arm_name)
        if not isinstance(arm, Mapping) or set(arm) != {"receipt_handle"}:
            _memory_error(f"{arm_name} input shape invalid")
        receipt_handle = arm.get("receipt_handle")
        if not isinstance(receipt_handle, str):
            _memory_error(f"{arm_name} receipt handle invalid")
        try:
            receipt = resolve_evidence_handle(
                evidence_resolver,
                receipt_handle,
                expected_kind="verifier",
            )
        except CoverageContractError as exc:
            _memory_error(f"{arm_name} receipt resolution failed", exc)
        if not isinstance(receipt, Mapping) or set(receipt) != {
            "arm",
            "artifact_handle",
            "capability",
            "kind",
            "schema",
            "task_fingerprint",
            "task_id",
            "verifier_status",
        }:
            _memory_error(f"{arm_name} receipt shape invalid")
        if (
            receipt.get("schema") != "nexus.learning_coverage_memory_arm.v1"
            or receipt.get("kind") != "verifier"
            or receipt.get("capability") != "memory"
            or receipt.get("task_id") != task_id
            or receipt.get("task_fingerprint") != fingerprint
            or receipt.get("arm") != arm_name
            or receipt.get("verifier_status") != status
        ):
            _memory_error(f"{arm_name} receipt binding invalid")
        artifact_handle = receipt.get("artifact_handle")
        if not isinstance(artifact_handle, str) or not _ARTIFACT_HANDLE.fullmatch(artifact_handle):
            _memory_error(f"{arm_name} artifact handle invalid")
        try:
            artifact = resolve_evidence_handle(
                evidence_resolver,
                artifact_handle,
                expected_kind="artifact",
            )
        except CoverageContractError as exc:
            _memory_error(f"{arm_name} artifact resolution failed", exc)
        if not isinstance(artifact, bytes) or not artifact:
            _memory_error(f"{arm_name} artifact bytes invalid")
        normalized_arms[arm_name] = {
            "receipt_handle": receipt_handle,
            "artifact_handle": artifact_handle,
            "verifier_status": status,
        }
    if (
        normalized_arms["memory_off"]["artifact_handle"]
        == normalized_arms["memory_on"]["artifact_handle"]
    ):
        _memory_error("arm artifact evidence must be distinct")
    return {
        "status": "paired_eligibility_observed",
        "task_id": task_id,
        "task_fingerprint": fingerprint,
        **normalized_arms,
        "missingness": [],
    }


def build_observational_probes(
    contract: Mapping[str, Any],
    *,
    evidence_resolver: EvidenceResolverInput | None = None,
    paired_memory_evidence: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    active_resolver, resolver_snapshot = prepare_evidence_resolver(evidence_resolver)
    try:
        return _build_observational_probes(
            contract,
            evidence_resolver=active_resolver,
            paired_memory_evidence=paired_memory_evidence,
        )
    finally:
        verify_evidence_resolver_unchanged(active_resolver, resolver_snapshot)


def _build_observational_probes(
    contract: Mapping[str, Any],
    *,
    evidence_resolver: EvidenceResolver | None,
    paired_memory_evidence: Mapping[str, Any] | None,
) -> dict[str, Any]:
    validate_coverage_contract(contract, evidence_resolver=evidence_resolver)
    result = {
        "schema": "nexus.learning_coverage_probes.v1",
        "source_binding": dict(contract["source_binding"]),
        "probes": _probe_rows(contract),
        "memory_uplift_signal": _memory_signal(
            paired_memory_evidence,
            evidence_resolver=evidence_resolver,
            expected_task_id=contract["task_id"],
        ),
        "claim_ceiling": CLAIM_CEILING,
    }
    validate_observational_probes(
        result,
        contract,
        evidence_resolver=evidence_resolver,
    )
    return result


def validate_observational_probes(
    probes: Mapping[str, Any],
    contract: Mapping[str, Any],
    *,
    evidence_resolver: EvidenceResolverInput | None = None,
) -> None:
    active_resolver, resolver_snapshot = prepare_evidence_resolver(evidence_resolver)
    try:
        _validate_observational_probes(
            probes,
            contract,
            evidence_resolver=active_resolver,
        )
    finally:
        verify_evidence_resolver_unchanged(active_resolver, resolver_snapshot)


def _validate_observational_probes(
    probes: Mapping[str, Any],
    contract: Mapping[str, Any],
    *,
    evidence_resolver: EvidenceResolver | None,
) -> None:
    validate_coverage_contract(contract, evidence_resolver=evidence_resolver)
    if not isinstance(probes, Mapping) or set(probes) != _PROBE_FIELDS:
        raise CoverageContractError("probe contract shape invalid")
    if (
        probes.get("schema") != "nexus.learning_coverage_probes.v1"
        or probes.get("claim_ceiling") != CLAIM_CEILING
    ):
        raise CoverageContractError("probe contract invalid")
    if probes.get("source_binding") != contract.get("source_binding"):
        raise CoverageContractError("probe source binding invalid")
    signal = probes.get("memory_uplift_signal")
    if not isinstance(signal, Mapping):
        raise CoverageContractError("memory uplift signal must be structured")
    if signal.get("status") == "missing":
        expected_signal = _memory_signal(
            None,
            evidence_resolver=evidence_resolver,
            expected_task_id=contract["task_id"],
        )
    elif signal.get("status") == "paired_eligibility_observed":
        off_signal = signal.get("memory_off")
        on_signal = signal.get("memory_on")
        if not isinstance(off_signal, Mapping) or not isinstance(on_signal, Mapping):
            raise CoverageContractError("memory uplift signal arm shape invalid")
        expected_signal = _memory_signal(
            {
                "task_id": signal.get("task_id"),
                "task_fingerprint": signal.get("task_fingerprint"),
                "memory_off": {"receipt_handle": off_signal.get("receipt_handle")},
                "memory_on": {"receipt_handle": on_signal.get("receipt_handle")},
            },
            evidence_resolver=evidence_resolver,
            expected_task_id=contract["task_id"],
        )
    else:
        raise CoverageContractError("memory uplift signal status invalid")
    if dict(signal) != expected_signal:
        raise CoverageContractError("memory uplift signal is incomplete or contradictory")
    rows = probes.get("probes")
    if not isinstance(rows, list) or rows != _probe_rows(contract):
        raise CoverageContractError("probe capability coverage invalid")
