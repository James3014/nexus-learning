"""Consumer-side tamper tests for governed adoption artifacts (Issue #85)."""

import copy
import json

import pytest
from test_adoption import UPLIFT, decide, scorecard_rows

from nexus_learning.adoption import AdoptionStore
from nexus_learning.contracts import (
    _policy_hash,
    canonical_adoption_identity,
    project_adoption_into_planner_budget,
    validate_learning_policy_adoption,
)
from nexus_learning.effectiveness_measurement import replay_scorecard
from nexus_learning.state_root import LearningStateRoot

OK_DELTA = {"episodic_memory_injection": {"enabled": True}}


def seal(adoption, *, rehash=True):
    if rehash:
        adoption["target_policy_hash"] = _policy_hash(adoption["target_policy_delta"])
        adoption["previous_policy_hash"] = _policy_hash(adoption["previous_policy"])
    adoption_hash, adoption_id = canonical_adoption_identity(adoption)
    adoption["adoption_hash"] = adoption_hash
    adoption["adoption_id"] = adoption_id
    return adoption


def forged(delta=None, **overrides):
    adoption = {
        "schema": "nexus.learning_policy_adoption.v1",
        "owner_authority_reference": "anyone",
        "recommendation_id": "lrec:does-not-exist",
        "recommendation_hash": "0" * 64,
        "validation_id": "lval:does-not-exist",
        "validation_hash": "0" * 64,
        "source_revision": "rev-x",
        "adopted_scope": {"task_family": "local_heal"},
        "target_policy_delta": copy.deepcopy(OK_DELTA if delta is None else delta),
        "target_policy_hash": "f" * 64,
        "previous_policy": {},
        "previous_policy_hash": "e" * 64,
        "rollback_target": {"target_state": {"memory_retrieval": "shadow"}},
        "adoption_status": "ACTIVE_CANDIDATE",
        "claim_ceiling": "SUPPORTED_POLICY_RECOMMENDATION",
        "route_truth_source": "CapabilityPlanner",
        "direct_route_mutation_allowed": False,
    }
    adoption.update(overrides)
    return seal(adoption)


@pytest.fixture()
def genuine():
    result = decide(replay_scorecard(scorecard_rows(*UPLIFT)))
    return result["adoption"], result["recommendation"], result["validation"]


def project(adoption, **kwargs):
    return project_adoption_into_planner_budget(adoption, task_desc="local_heal task", **kwargs)


def assert_rejected_everywhere(adoption, tmp_path, code=None):
    match = code or ""
    with pytest.raises(ValueError, match=match):
        validate_learning_policy_adoption(adoption)
    store = AdoptionStore(LearningStateRoot(tmp_path))
    with pytest.raises(ValueError, match=match):
        store.write_adoption(adoption)
    path = store._state_root.adoption_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(adoption))
    with pytest.raises(ValueError, match=match):
        store.read_adoption()
    with pytest.raises(ValueError, match=match):
        project(adoption)


FORBIDDEN_VARIANTS = [
    {"capabilityplanner": "x", **OK_DELTA},
    {"CAPABILITYPLANNER": "x", **OK_DELTA},
    {"Route_Authority": "x", **OK_DELTA},
    {"ROUTE_AUTHORITY": "x", **OK_DELTA},
    {"Workforce_Admission": "x", **OK_DELTA},
    {"episodic_memory_injection": {"enabled": True, "CapabilityPlanner": {"force": "evil"}}},
    {"episodic_memory_injection": {"enabled": True, "route_authority": "x"}},
    {"nested": {"active_route_override": "evil"}},
    {"deep": [{"ok": {"Workforce_Admission": "open"}}]},
]


@pytest.mark.parametrize("delta", FORBIDDEN_VARIANTS)
def test_forbidden_delta_keys_rejected_recursively_and_case_insensitively(delta, tmp_path):
    assert_rejected_everywhere(forged(delta), tmp_path, "ADOPTION_DIRECT_PLANNER_MUTATION")


def test_whitespace_owner_authority_rejected(tmp_path):
    assert_rejected_everywhere(
        forged(owner_authority_reference="   "), tmp_path, "ADOPTION_OWNER_AUTHORITY_MISSING"
    )


@pytest.mark.parametrize("bad", ["yes", ["enabled"], 1, None])
def test_non_dict_episodic_injection_rejected(bad, tmp_path):
    assert_rejected_everywhere(
        forged({"episodic_memory_injection": bad}), tmp_path, "ADOPTION_POLICY_SHAPE_INVALID"
    )


def test_read_rollback_validates(tmp_path):
    store = AdoptionStore(LearningStateRoot(tmp_path))
    path = store._state_root.rollback_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"schema": "x", "adoption_id": "whatever"}))
    with pytest.raises(ValueError, match="ROLLBACK_SCHEMA_INVALID"):
        store.read_rollback()


# ---- R1: optional lineage verification -------------------------------------


def test_genuine_adoption_with_lineage_accepted(genuine):
    adoption, recommendation, validation = genuine
    validate_learning_policy_adoption(adoption, recommendation=recommendation, validation=validation)
    out = project(adoption, recommendation=recommendation, validation=validation)
    assert out["learning_policy"]["episodic_memory_injection"]["enabled"] is True


def test_lineage_is_optional_for_backward_compatibility(genuine):
    adoption, _, _ = genuine
    validate_learning_policy_adoption(adoption)
    project(adoption)


def test_forged_adoption_rejected_when_lineage_supplied(genuine):
    _, recommendation, validation = genuine
    fake = forged()
    with pytest.raises(ValueError, match="ADOPTION_SOURCE_REVISION_STALE"):
        validate_learning_policy_adoption(fake, recommendation=recommendation, validation=validation)
    with pytest.raises(ValueError, match="ADOPTION_SOURCE_REVISION_STALE"):
        project(fake, recommendation=recommendation, validation=validation)


def test_only_one_lineage_artifact_is_incomplete(genuine):
    adoption, recommendation, validation = genuine
    with pytest.raises(ValueError, match="ADOPTION_LINEAGE_INCOMPLETE"):
        validate_learning_policy_adoption(adoption, recommendation=recommendation)
    with pytest.raises(ValueError, match="ADOPTION_LINEAGE_INCOMPLETE"):
        validate_learning_policy_adoption(adoption, validation=validation)
    with pytest.raises(ValueError, match="ADOPTION_LINEAGE_INCOMPLETE"):
        project(adoption, validation=validation)


def _reseal(adoption, **changes):
    adoption = copy.deepcopy(adoption)
    adoption.update(changes)
    return seal(adoption)


def test_lineage_id_and_hash_mismatch_rejected(genuine):
    adoption, recommendation, validation = genuine
    for field, code in (
        ("recommendation_id", "ADOPTION_LINEAGE_RECOMMENDATION_MISMATCH"),
        ("recommendation_hash", "ADOPTION_LINEAGE_RECOMMENDATION_MISMATCH"),
        ("validation_id", "ADOPTION_LINEAGE_VALIDATION_MISMATCH"),
        ("validation_hash", "ADOPTION_LINEAGE_VALIDATION_MISMATCH"),
    ):
        bad = _reseal(adoption, **{field: "z" * 8})
        with pytest.raises(ValueError, match=code):
            validate_learning_policy_adoption(bad, recommendation=recommendation, validation=validation)


def test_lineage_scope_and_revision_mismatch_rejected(genuine):
    adoption, recommendation, validation = genuine
    cases = [
        ({"source_revision": "other-rev"}, "ADOPTION_SOURCE_REVISION_STALE"),
        ({"adopted_scope": {"task_family": "other"}}, "ADOPTION_SCOPE_TASK_FAMILY_MISMATCH"),
    ]
    for changes, code in cases:
        bad = _reseal(adoption, **changes)
        with pytest.raises(ValueError, match=code):
            validate_learning_policy_adoption(bad, recommendation=recommendation, validation=validation)


def test_tampered_or_rejected_validation_rejected(genuine):
    adoption, recommendation, validation = genuine
    bad_val = copy.deepcopy(validation)
    bad_val["blockers"] = ["x"]
    with pytest.raises(ValueError, match="ADOPTION_VALIDATION_HASH_TAMPERED"):
        validate_learning_policy_adoption(adoption, recommendation=recommendation, validation=bad_val)
    bad_val = copy.deepcopy(validation)
    bad_val["validation_disposition"] = "REJECTED_RECOMMENDATION"
    with pytest.raises(ValueError, match="ADOPTION_VALIDATION_DISPOSITION_INVALID"):
        validate_learning_policy_adoption(adoption, recommendation=recommendation, validation=bad_val)
    bad_rec = copy.deepcopy(recommendation)
    bad_rec["observed_effect"] = "forged"
    with pytest.raises(ValueError, match="RECOMMENDATION_CONTENT_ADDRESS_MISMATCH"):
        validate_learning_policy_adoption(adoption, recommendation=bad_rec, validation=validation)


# --- Codex adversarial review r1 (PR #108) -----------------------------------


def _deep(levels):
    value = {"leaf": True}
    for _ in range(levels):
        value = {"x": value}
    return value


def test_deeply_nested_delta_rejected_with_value_error(tmp_path):
    adoption = forged({"episodic_memory_injection": {"enabled": True}, "deep": _deep(200)})
    assert_rejected_everywhere(adoption, tmp_path, "ADOPTION_DIRECT_PLANNER_MUTATION_FORBIDDEN")


@pytest.mark.parametrize("enabled", ["false", 0, 1, None, "true"])
def test_non_bool_injection_enabled_rejected(tmp_path, enabled):
    adoption = forged({"episodic_memory_injection": {"enabled": enabled}})
    assert_rejected_everywhere(adoption, tmp_path, "ADOPTION_POLICY_SHAPE_INVALID")


def test_validation_id_not_derived_from_hash_rejected(genuine):
    adoption, recommendation, validation = genuine
    bad_val = copy.deepcopy(validation)
    bad_val["validation_id"] = "lval:forged"
    bad = _reseal(adoption, validation_id="lval:forged")
    with pytest.raises(ValueError, match="ADOPTION_VALIDATION_ID_TAMPERED"):
        validate_learning_policy_adoption(bad, recommendation=recommendation, validation=bad_val)


def _genuine_rollback(adoption):
    from nexus_learning.contracts import build_learning_policy_rollback

    return build_learning_policy_rollback(adoption=adoption, reason="regression", triggered_by="owner")


def _reseal_rollback(rollback, **changes):
    from nexus_learning.contracts import canonical_rollback_identity

    rollback = copy.deepcopy(rollback)
    rollback.update(changes)
    rollback["rollback_hash"], rollback["rollback_id"] = canonical_rollback_identity(rollback)
    return rollback


def test_genuine_rollback_still_rolls_back(genuine):
    adoption, _, _ = genuine
    budget = project(adoption, rollback=_genuine_rollback(adoption))
    assert budget["learning_policy"]["adoption_lineage"]["status"] == "ROLLED_BACK"


@pytest.mark.parametrize(
    "changes,code",
    [
        ({"rolled_back_policy_hash": "0" * 64}, "ROLLBACK_POLICY_HASH_MISMATCH"),
        ({"rollback_status": "ACTIVE"}, "ROLLBACK_STATUS_INVALID"),
        ({"adoption_hash": "0" * 64}, "PROJECTION_ROLLBACK_ADOPTION_MISMATCH"),
        ({"previous_policy_hash": "0" * 64}, "PROJECTION_ROLLBACK_ADOPTION_MISMATCH"),
        ({"recommendation_hash": "0" * 64}, "PROJECTION_ROLLBACK_ADOPTION_MISMATCH"),
    ],
)
def test_resealed_forged_rollback_rejected(genuine, changes, code):
    adoption, _, _ = genuine
    bad = _reseal_rollback(_genuine_rollback(adoption), **changes)
    with pytest.raises(ValueError, match=code):
        project(adoption, rollback=bad)


def test_non_mapping_rollback_file_is_not_treated_as_absent(tmp_path):
    store = AdoptionStore(LearningStateRoot(tmp_path))
    path = store._state_root.rollback_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("[]")
    with pytest.raises(ValueError, match="ROLLBACK_NOT_A_MAPPING"):
        store.read_rollback()


# --- Gemini cross-model review (PR #108) --------------------------------------


@pytest.mark.parametrize("validation", ["bad", [1, 2], 3])
def test_non_mapping_validation_lineage_rejected_with_value_error(genuine, validation):
    adoption, recommendation, _ = genuine
    with pytest.raises(ValueError, match="ADOPTION_VALIDATION_SCHEMA_INVALID"):
        validate_learning_policy_adoption(adoption, recommendation=recommendation, validation=validation)


@pytest.mark.parametrize("scope", ["local_heal", ["task_family"], 1])
def test_non_mapping_adopted_scope_rejected(tmp_path, scope):
    assert_rejected_everywhere(forged(adopted_scope=scope), tmp_path, "ADOPTION_SCOPE_OVERBROAD")


@pytest.mark.parametrize("target", ["shadow", {"notes": "no target"}, {"target_state": {}}, ["x"]])
def test_unrollbackable_adoption_rejected(tmp_path, target):
    adoption = forged(rollback_target=target)
    assert_rejected_everywhere(adoption, tmp_path, "ADOPTION_MISSING_ROLLBACK_TARGET")


def test_non_mapping_adoption_file_is_not_treated_as_absent(tmp_path):
    store = AdoptionStore(LearningStateRoot(tmp_path))
    path = store._state_root.adoption_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("[]")
    with pytest.raises(ValueError, match="ADOPTION_NOT_A_MAPPING"):
        store.read_adoption()
