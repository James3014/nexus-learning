import hashlib
import json
from copy import deepcopy

import pytest

from nexus_learning.contracts import CAPABILITY_TAXONOMY
from nexus_learning.coverage_contract import (
    CoverageContractError,
    build_coverage_contract,
    validate_coverage_contract,
)


def _put_record(store, kind, record):
    payload = json.dumps(
        record,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    handle = f"{kind}:sha256:{hashlib.sha256(payload).hexdigest()}"
    store[handle] = deepcopy(record)
    return handle


def test_contract_is_exactly_one_deterministic_row_per_live_taxonomy_capability():
    contract = build_coverage_contract()
    again = build_coverage_contract()

    assert [row["capability"] for row in contract["rows"]] == sorted(CAPABILITY_TAXONOMY)
    assert contract == again
    assert len(contract["rows"]) == len(CAPABILITY_TAXONOMY)
    assert {"W", "F", "P", "S"} <= set(contract["rows"][0]["evidence_levels"])
    assert (
        contract["claim_ceiling"] == "DETERMINISTIC_EXACT_SOURCE_TAXONOMY_EVIDENCE_CLASSIFICATION"
    )


def test_missingness_is_explicit_and_unknown_evidence_cannot_be_fabricated_as_zero():
    contract = build_coverage_contract()
    row = deepcopy(contract["rows"][0])
    row["invoked"] = False
    row["missingness"] = []
    contract["rows"] = [row, *contract["rows"][1:]]

    with pytest.raises(CoverageContractError, match="missingness"):
        validate_coverage_contract(contract)


@pytest.mark.parametrize("fabricated", [False, True])
def test_all_boolean_values_require_bound_evidence_not_fabricated_truth(fabricated):
    values = {
        field: fabricated
        for field in (
            "selected",
            "invoked",
            "evidence_present",
            "outcome",
            "gate_passed",
            "persistence",
            "consumer_shadow_use",
            "verifier_proof",
        )
    }
    capability = sorted(CAPABILITY_TAXONOMY)[0]
    store = {}
    record_values = {
        "artifact": {
            "evidence_present": not fabricated,
            "outcome": not fabricated,
        },
        "consumer_shadow": {"consumer_shadow_use": not fabricated},
        "invocation": {"invoked": not fabricated},
        "persistence": {"persistence": not fabricated},
        "selection": {"selected": not fabricated},
        "verifier": {
            "gate_passed": not fabricated,
            "verifier_proof": not fabricated,
        },
    }
    source_handles = [
        _put_record(
            store,
            kind,
            {
                "schema": "nexus.learning_coverage_evidence.v1",
                "kind": kind,
                "capability": capability,
                "task_id": "task-1",
                "values": observed_values,
            },
        )
        for kind, observed_values in record_values.items()
    ]

    with pytest.raises(CoverageContractError, match="lifecycle binding"):
        build_coverage_contract(
            {
                capability: {
                    **values,
                    "evidence_levels": {level: "observed" for level in "WFPS"},
                    "missingness": [],
                    "source_handles": source_handles,
                }
            },
            task_id="task-1",
            evidence_resolver=store,
        )


def test_observed_wfps_requires_matching_lifecycle_evidence_and_receipt_kinds():
    capability = sorted(CAPABILITY_TAXONOMY)[0]
    with pytest.raises(CoverageContractError, match="evidence level"):
        build_coverage_contract(
            {
                capability: {
                    "evidence_levels": {level: "observed" for level in "WFPS"},
                }
            }
        )


def test_legal_looking_nonexistent_hash_is_not_source_authority():
    capability = sorted(CAPABILITY_TAXONOMY)[0]
    digest = hashlib.sha256(b"absent selection receipt").hexdigest()
    with pytest.raises(CoverageContractError, match="resolve"):
        build_coverage_contract(
            {
                capability: {
                    "selected": True,
                    "evidence_levels": {"W": "observed"},
                    "source_handles": [f"selection:sha256:{digest}"],
                }
            },
            task_id="task-1",
            evidence_resolver={},
        )


def test_forged_bounded_looking_source_handle_is_not_an_evidence_authority():
    contract = build_coverage_contract()
    contract["rows"][0]["source_handles"] = ["attacker:invented-proof"]
    with pytest.raises(CoverageContractError, match="source handle"):
        validate_coverage_contract(contract)


@pytest.mark.parametrize(
    "mutator",
    [
        lambda c: c["rows"].append(deepcopy(c["rows"][0])),
        lambda c: c["rows"].__setitem__(0, {**c["rows"][0], "capability": "forged"}),
        lambda c: c["rows"].__setitem__(0, {**c["rows"][0], "evidence_levels": {"W": "bogus"}}),
        lambda c: c["rows"].__setitem__(
            0, {**c["rows"][0], "source_handles": ["free text\nnot a handle"]}
        ),
        lambda c: c["rows"][0].__setitem__("evidence_note", "trust me"),
    ],
)
def test_duplicate_unknown_malformed_and_unbounded_evidence_fail_closed(mutator):
    contract = build_coverage_contract()
    mutator(contract)

    with pytest.raises(CoverageContractError):
        validate_coverage_contract(contract)


def test_source_binding_rejects_tampered_or_stale_contract():
    contract = build_coverage_contract()
    tampered = deepcopy(contract)
    tampered["source_binding"]["taxonomy_sha256"] = "0" * 64
    with pytest.raises(CoverageContractError, match="source"):
        validate_coverage_contract(tampered)

    stale = deepcopy(contract)
    stale["source_binding"]["source_revision"] = "stale"
    with pytest.raises(CoverageContractError, match="source"):
        validate_coverage_contract(stale)


def test_inconsistent_transitions_and_fabricated_outcome_fail_closed():
    contract = build_coverage_contract()
    row = {**contract["rows"][0], "selected": False, "invoked": True}
    contract["rows"] = [row, *contract["rows"][1:]]
    with pytest.raises(CoverageContractError, match="transition"):
        validate_coverage_contract(contract)

    contract = build_coverage_contract()
    contract["rows"] = [{**contract["rows"][0], "outcome": True}, *contract["rows"][1:]]
    with pytest.raises(CoverageContractError, match="outcome"):
        validate_coverage_contract(contract)
