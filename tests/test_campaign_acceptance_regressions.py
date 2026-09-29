"""False-green and semantic replay controls for campaign helper evidence."""
import hashlib
import json
import unittest

from nexus_learning.campaign_gates import (
    build_campaign_closeout,
    build_cohort_preflight,
    verify_closeout,
    verify_preflight,
)


def _input():
    return dict(cohort_id="c", incumbent_identity={"deterministic_kind": "literal", "result_hash": "a" * 64}, incumbent_result="PASS",
                candidate_levels=[{"level_id": "a", "mechanism": "extract", "disposition": "PASS"}], role_metrics={"false_safe_rate": 0.0})


def _rehash(body):
    body.pop("content_sha256")
    body["content_sha256"] = hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def test_accuracy_only_and_unpassed_stage_cannot_pass():
    data = _input()
    assert build_cohort_preflight(**data)["eligible"]
    assert not build_cohort_preflight(**dict(data, role_metrics={"accuracy": 1.0}))["eligible"]
    for disposition in ("FAILED", "NOT_EVALUATED", "STOPPED_BY_GATE"):
        data["candidate_levels"][0]["disposition"] = disposition
        assert not build_cohort_preflight(**data)["eligible"]


def test_invalid_rates_rejected():
    for rate in (float("nan"), float("inf"), -0.1, 1.1, True):
        with unittest.TestCase().assertRaises(ValueError):
            build_cohort_preflight(**dict(_input(), role_metrics={"false_safe_rate": rate}))


def test_rehashed_eligibility_tamper_fails():
    body = build_cohort_preflight(**dict(_input(), role_metrics={"accuracy": 1.0}))
    body["eligible"] = True
    _rehash(body)
    assert not verify_preflight(body)


def test_missing_closeout_receipt_explicit_and_not_rehashable_to_complete():
    body = build_campaign_closeout(campaign_id="c", closed_experiments=[{"experiment_id": "e", "disposition": "STOPPED_BY_GATE"}], highest_safe_claims={}, not_proven=["runtime"], reopen_triggers=[])
    assert body["next_gate"] == "NONE_DEFINED"
    assert not body["complete"]
    assert verify_closeout(body)
    body["complete"] = True
    _rehash(body)
    assert not verify_closeout(body)
