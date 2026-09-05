import hashlib
import json
from copy import deepcopy

import pytest

from nexus_learning.coverage_contract import (
    CoverageContractError,
    build_coverage_contract,
)
from nexus_learning.coverage_probes import (
    build_observational_probes,
    validate_observational_probes,
)


def _canonical(record):
    return json.dumps(record, ensure_ascii=True, separators=(",", ":"), sort_keys=True).encode()


def _put_record(store, kind, record):
    handle = f"{kind}:sha256:{hashlib.sha256(_canonical(record)).hexdigest()}"
    store[handle] = deepcopy(record)
    return handle


def _put_artifact(store, content):
    handle = f"artifact:sha256:{hashlib.sha256(content).hexdigest()}"
    store[handle] = content
    return handle


def _absent_handle(kind, content):
    return f"{kind}:sha256:{hashlib.sha256(content).hexdigest()}"


def _store_digest(store):
    entries = [
        (
            handle,
            hashlib.sha256(value if isinstance(value, bytes) else _canonical(value)).hexdigest(),
        )
        for handle, value in sorted(store.items())
    ]
    return hashlib.sha256(_canonical(entries)).hexdigest()


class _ImmutableResolver:
    def __init__(self, store):
        self._store = store

    def resolve(self, handle):
        return self._store.get(handle)

    def snapshot_digest(self):
        return _store_digest(self._store)


class _SideEffectfulResolver(_ImmutableResolver):
    def __init__(self, store):
        super().__init__(store)
        self.writes = 0

    def resolve(self, handle):
        self.writes += 1
        return self._store.get(handle)

    def snapshot_digest(self):
        payload = f"{_store_digest(self._store)}:writes:{self.writes}".encode()
        return hashlib.sha256(payload).hexdigest()


def _lifecycle_record(kind, values, *, capability="memory", task_id="task-1"):
    return {
        "schema": "nexus.learning_coverage_evidence.v1",
        "kind": kind,
        "capability": capability,
        "task_id": task_id,
        "values": values,
    }


def _observed_contract(value=True):
    capability = "memory"
    store = {}
    records = {
        "selection": {"selected": value},
        "invocation": {"invoked": value},
        "artifact": {"evidence_present": value, "outcome": value},
        "verifier": {"gate_passed": value, "verifier_proof": value},
        "persistence": {"persistence": value},
        "consumer_shadow": {"consumer_shadow_use": value},
    }
    handles = [
        _put_record(store, kind, _lifecycle_record(kind, values))
        for kind, values in records.items()
    ]
    contract = build_coverage_contract(
        {
            capability: {
                "selected": value,
                "invoked": value,
                "evidence_present": value,
                "outcome": value,
                "gate_passed": value,
                "persistence": value,
                "consumer_shadow_use": value,
                "verifier_proof": value,
                "evidence_levels": {level: "observed" for level in "WFPS"},
                "source_handles": handles,
            }
        },
        task_id="task-1",
        evidence_resolver=store,
    )
    return contract, store


def _paired_memory_evidence(store):
    pair = {"task_id": "task-1", "task_fingerprint": "fp-1"}
    for arm, status, content in (
        ("memory_off", "fail", b"off verifier artifact"),
        ("memory_on", "pass", b"on verifier artifact"),
    ):
        artifact_handle = _put_artifact(store, content)
        receipt = {
            "schema": "nexus.learning_coverage_memory_arm.v1",
            "kind": "verifier",
            "capability": "memory",
            "task_id": "task-1",
            "task_fingerprint": "fp-1",
            "arm": arm,
            "verifier_status": status,
            "artifact_handle": artifact_handle,
        }
        pair[arm] = {"receipt_handle": _put_record(store, "verifier", receipt)}
    return pair


def _replace_arm_receipt(pair, store, arm_name, **updates):
    receipt = deepcopy(store[pair[arm_name]["receipt_handle"]])
    receipt.update(updates)
    pair[arm_name] = {"receipt_handle": _put_record(store, "verifier", receipt)}


def test_probes_are_deterministic_source_backed_and_keep_memory_pair_strict():
    contract, store = _observed_contract()
    pair = _paired_memory_evidence(store)
    original_store = deepcopy(store)
    probes = build_observational_probes(
        contract,
        evidence_resolver=store,
        paired_memory_evidence=pair,
    )
    assert probes == build_observational_probes(
        contract,
        evidence_resolver=store,
        paired_memory_evidence=pair,
    )
    assert probes["claim_ceiling"] == "OBSERVATIONAL_SOURCE_BACKED_PROBES_ONLY"
    assert probes["memory_uplift_signal"]["status"] == "paired_eligibility_observed"
    assert set(probes["memory_uplift_signal"]) >= {"memory_off", "memory_on", "task_fingerprint"}
    assert all(probe["source_handles"] for probe in probes["probes"])
    assert all(probe["observational_only"] is True for probe in probes["probes"])
    validate_observational_probes(probes, contract, evidence_resolver=store)
    assert store == original_store


@pytest.mark.parametrize("value", [False, True])
def test_resolver_bound_boolean_observations_are_not_fabricated(value):
    contract, store = _observed_contract(value)
    probes = build_observational_probes(contract, evidence_resolver=store)
    assert [probe["capability"] for probe in probes["probes"]] == ["memory"]


def test_mapping_and_immutable_resolvers_are_accepted_without_state_change():
    contract, store = _observed_contract()
    immutable = _ImmutableResolver(store)
    probes = build_observational_probes(contract, evidence_resolver=immutable)
    validate_observational_probes(
        probes,
        contract,
        evidence_resolver=immutable,
    )


def test_side_effectful_and_arbitrary_callable_resolvers_fail_closed():
    contract, store = _observed_contract()
    side_effectful = _SideEffectfulResolver(store)
    with pytest.raises(CoverageContractError, match="resolver state changed"):
        build_observational_probes(contract, evidence_resolver=side_effectful)

    with pytest.raises(CoverageContractError, match="resolver interface"):
        build_observational_probes(contract, evidence_resolver=store.get)


def test_no_probe_is_fabricated_from_taxonomy_membership_alone():
    probes = build_observational_probes(build_coverage_contract())
    assert probes["probes"] == []
    assert probes["memory_uplift_signal"]["status"] == "missing"


def test_probe_wfps_must_equal_validated_contract_evidence():
    contract, store = _observed_contract()
    probes = build_observational_probes(contract, evidence_resolver=store)
    probes["probes"][0]["evidence_levels"]["P"] = "missing"
    with pytest.raises(CoverageContractError, match="probe"):
        validate_observational_probes(probes, contract, evidence_resolver=store)


@pytest.mark.parametrize(
    "mutator",
    [
        lambda pair, store: pair.pop("memory_off"),
        lambda pair, store: pair["memory_on"].__setitem__(
            "receipt_handle",
            _absent_handle("verifier", b"absent memory-on receipt"),
        ),
        lambda pair, store: store.__setitem__(
            pair["memory_on"]["receipt_handle"], b"tampered receipt bytes"
        ),
        lambda pair, store: _replace_arm_receipt(pair, store, "memory_on", arm="memory_off"),
        lambda pair, store: _replace_arm_receipt(pair, store, "memory_on", verifier_status="fail"),
        lambda pair, store: _replace_arm_receipt(pair, store, "memory_on", task_id="other-task"),
        lambda pair, store: _replace_arm_receipt(
            pair,
            store,
            "memory_on",
            artifact_handle=_absent_handle("artifact", b"absent bound artifact"),
        ),
    ],
)
def test_paired_memory_requires_exact_arms_identity_and_evidence(mutator):
    contract, store = _observed_contract()
    pair = _paired_memory_evidence(store)
    mutator(pair, store)
    with pytest.raises(CoverageContractError, match="memory"):
        build_observational_probes(
            contract,
            evidence_resolver=store,
            paired_memory_evidence=pair,
        )


def test_universal_receipts_cannot_fabricate_all_taxonomy_probes():
    _, store = _observed_contract()
    universal_handle = next(iter(store))
    observations = {
        row["capability"]: {
            "selected": True,
            "evidence_levels": {"W": "observed"},
            "source_handles": [universal_handle],
        }
        for row in build_coverage_contract()["rows"]
    }
    with pytest.raises(CoverageContractError, match="capability"):
        build_coverage_contract(
            observations,
            task_id="task-1",
            evidence_resolver=store,
        )


@pytest.mark.parametrize(
    "tamper",
    [
        lambda p: p["probes"].append(deepcopy(p["probes"][0])),
        lambda p: p["probes"].__setitem__(0, {**p["probes"][0], "observational_only": False}),
        lambda p: p.__setitem__("memory_uplift_signal", "uplift_verified"),
    ],
)
def test_probe_tampering_fails_closed(tamper):
    contract, store = _observed_contract()
    probes = build_observational_probes(contract, evidence_resolver=store)
    tamper(probes)
    with pytest.raises(CoverageContractError):
        validate_observational_probes(probes, contract, evidence_resolver=store)


def test_probe_contract_rejects_unbound_contract():
    contract = build_coverage_contract()
    probes = build_observational_probes(contract)
    probes["source_binding"] = {**probes["source_binding"], "taxonomy_sha256": "f" * 64}
    with pytest.raises(CoverageContractError, match="source"):
        validate_observational_probes(probes, contract)
