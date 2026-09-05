import json
from copy import deepcopy

import pytest

from nexus_learning.effectiveness_measurement import (
    ReplayContractError,
    paired_memory_uplift,
    replay_scorecard,
)


def row(**overrides):
    value = {
        "task_fingerprint": "fp-1",
        "task_id": "task-1",
        "attempt_id": "attempt-1",
        "attempt_index": 0,
        "action_id": "action-1",
        "source_revision": "rev-1",
        "source_tree": "tree-1",
        "verifier_status": "pass",
        "verifier_artifact": "artifact.json",
        "verifier_artifact_hash": "hash-1",
        "verifier_receipt": "receipt.json",
        "memory_arm": "memory_on",
        "retrieved_lesson_ids": ["lesson-1"],
        "applied_attributed_lesson_ids": ["lesson-1"],
        "terminal_outcome": "SUCCEEDED",
        "measured_elapsed_seconds": 2.5,
        "intervention_events": [],
        "intervention_count": 0,
        "forbidden_strategy_identity": "forbidden-x",
        "forbidden_strategy_violation_event": False,
        "missingness_reasons": [],
        "ineligibility_reasons": [],
    }
    value.update(overrides)
    return value


def test_identity_missingness_and_duplicate_rejection():
    missing = row()
    del missing["task_fingerprint"]
    with pytest.raises(ReplayContractError):
        replay_scorecard([missing])
    with pytest.raises(ReplayContractError):
        replay_scorecard([row(), row()])


def test_replay_is_order_independent_and_does_not_mutate_inputs():
    rows = [
        row(),
        row(
            task_fingerprint="fp-2",
            task_id="task-2",
            attempt_id="attempt-2",
            action_id="action-2",
            attempt_index=1,
            verifier_status="failed",
            terminal_outcome="FAILED",
            retrieved_lesson_ids=[],
            applied_attributed_lesson_ids=[],
            intervention_events=["manual"],
            intervention_count=1,
            forbidden_strategy_violation_event=True,
        ),
    ]
    original = deepcopy(rows)
    assert replay_scorecard(rows) == replay_scorecard(list(reversed(rows)))
    assert rows == original


def test_missing_telemetry_is_not_fabricated_as_zero_or_success():
    result = replay_scorecard([
        row(
            measured_elapsed_seconds=None,
            intervention_events=None,
            intervention_count=None,
            missingness_reasons=[
                "measured_elapsed_seconds:unreported",
                "intervention_count:unreported",
                "intervention_events:unreported",
            ],
        )
    ])
    assert result["metrics"]["time_to_green"]["numerator"] is None
    assert result["metrics"]["intervention_rate"]["denominator"] == 0
    assert result["metrics"]["intervention_rate"]["missing"] == 1
    assert result["metrics"]["intervention_rate"]["rate"] is None


def test_first_green_uses_first_complete_passing_row_not_earlier_failure():
    failed = row(
        attempt_id="failed",
        action_id="failed-action",
        attempt_index=0,
        verifier_status="failed",
        terminal_outcome="FAILED",
        measured_elapsed_seconds=1.0,
    )
    passing = row(
        attempt_id="passing",
        action_id="passing-action",
        attempt_index=1,
        measured_elapsed_seconds=9.0,
    )

    metrics = replay_scorecard([passing, failed])["metrics"]

    assert metrics["attempts_to_green"]["numerator"] == 2
    assert metrics["attempts_to_green"]["denominator"] == 1
    assert metrics["time_to_green"]["numerator"] == 9.0
    assert metrics["time_to_green"]["denominator"] == 1


@pytest.mark.parametrize(
    ("field", "invalid"),
    [
        ("source_revision", ""),
        ("source_tree", " "),
        ("verifier_artifact", ""),
        ("verifier_artifact_hash", None),
        ("verifier_receipt", ""),
        ("forbidden_strategy_identity", ""),
        ("memory_arm", False),
        ("verifier_status", "maybe"),
        ("verifier_status", 1),
        ("terminal_outcome", "UNKNOWN"),
        ("terminal_outcome", False),
        ("measured_elapsed_seconds", -1.0),
        ("measured_elapsed_seconds", float("inf")),
        ("measured_elapsed_seconds", float("nan")),
        ("measured_elapsed_seconds", True),
        ("intervention_count", -1),
        ("intervention_count", 1.5),
        ("intervention_count", False),
        ("intervention_events", "event"),
        ("intervention_events", {"event": "manual"}),
        ("intervention_events", [""]),
        ("forbidden_strategy_violation_event", "false"),
    ],
)
def test_invalid_identity_evidence_and_telemetry_fail_closed(field, invalid):
    with pytest.raises(ReplayContractError):
        replay_scorecard([row(**{field: invalid})])


def test_verifier_receipt_is_required_and_identity_evidence_types_are_strict():
    missing_receipt = row()
    del missing_receipt["verifier_receipt"]
    with pytest.raises(ReplayContractError, match="verifier_receipt"):
        replay_scorecard([missing_receipt])
    for field in ("task_fingerprint", "task_id", "attempt_id", "action_id"):
        with pytest.raises(ReplayContractError):
            replay_scorecard([row(**{field: 7})])
        with pytest.raises(ReplayContractError):
            replay_scorecard([row(**{field: " "})])
    with pytest.raises(ReplayContractError):
        replay_scorecard([row(retrieved_lesson_ids="lesson-1")])
    with pytest.raises(ReplayContractError):
        replay_scorecard([row(applied_attributed_lesson_ids=["", 3])])
    with pytest.raises(ReplayContractError):
        replay_scorecard([row(verifier_status="pass", terminal_outcome="FAILED")])
    with pytest.raises(ReplayContractError):
        replay_scorecard([row(verifier_status="failed", terminal_outcome="SUCCEEDED")])
    with pytest.raises(ReplayContractError):
        replay_scorecard([row(measured_elapsed_seconds=None)])
    with pytest.raises(ReplayContractError):
        replay_scorecard([
            row(measured_elapsed_seconds=None, missingness_reasons=["other_field:unreported"])
        ])
    with pytest.raises(ReplayContractError):
        replay_scorecard([row(intervention_events=["manual"], intervention_count=0)])
    with pytest.raises(ReplayContractError):
        replay_scorecard([row(forbidden_strategy_violation_event=None)])


def test_metric_missingness_is_per_field_and_non_cohort_rows_are_not_missing():
    elapsed_missing = row(
        task_fingerprint="fp-elapsed",
        task_id="task-elapsed",
        attempt_id="elapsed",
        action_id="elapsed-action",
        measured_elapsed_seconds=None,
        missingness_reasons=["measured_elapsed_seconds:unreported"],
    )
    no_retrieval = row(
        task_fingerprint="fp-no-retrieval",
        task_id="task-no-retrieval",
        attempt_id="no-retrieval",
        action_id="no-retrieval-action",
        retrieved_lesson_ids=[],
        applied_attributed_lesson_ids=[],
    )

    metrics = replay_scorecard([elapsed_missing, no_retrieval])["metrics"]

    assert metrics["time_to_green"]["denominator"] == 1
    assert metrics["time_to_green"]["missing"] == 1
    assert metrics["intervention_rate"]["denominator"] == 2
    assert metrics["intervention_rate"]["missing"] == 0
    assert metrics["failure_recurrence"]["denominator"] == 0
    assert metrics["failure_recurrence"]["eligible"] == 0
    assert metrics["failure_recurrence"]["missing"] == 0


def test_green_metrics_mark_ineligible_passing_task_missing_without_using_it():
    ineligible_pass = row(ineligibility_reasons=["verifier_scope:mismatch"])

    metrics = replay_scorecard([ineligible_pass])["metrics"]

    assert (
        metrics["attempts_to_green"]["denominator"],
        metrics["attempts_to_green"]["missing"],
    ) == (0, 1)
    assert (metrics["time_to_green"]["denominator"], metrics["time_to_green"]["missing"]) == (0, 1)


def test_failure_recurrence_uses_one_task_fingerprint_population():
    rows = [
        row(
            task_fingerprint="fp-fail",
            task_id="task-fail",
            attempt_id=f"fail-{index}",
            action_id=f"fail-action-{index}",
            attempt_index=index,
            verifier_status="failed",
            terminal_outcome="FAILED",
        )
        for index in range(2)
    ] + [
        row(
            task_fingerprint="fp-pass",
            task_id="task-pass",
            attempt_id="pass",
            action_id="pass-action",
        )
    ]

    metric = replay_scorecard(rows)["metrics"]["failure_recurrence"]

    assert metric["numerator"] == 1
    assert metric["denominator"] == 1
    assert metric["eligible"] == 1
    assert metric["missing"] == 0


def test_failure_recurrence_excludes_a_task_with_ineligible_failure_evidence():
    eligible_failure = row(
        task_fingerprint="fp-partial-fail",
        task_id="task-partial-fail",
        attempt_id="eligible-fail",
        action_id="eligible-fail-action",
        verifier_status="failed",
        terminal_outcome="FAILED",
    )
    ineligible_failure = row(
        task_fingerprint="fp-partial-fail",
        task_id="task-partial-fail",
        attempt_id="ineligible-fail",
        action_id="ineligible-fail-action",
        attempt_index=1,
        verifier_status="failed",
        terminal_outcome="FAILED",
        ineligibility_reasons=["verifier_scope:mismatch"],
    )

    metric = replay_scorecard([eligible_failure, ineligible_failure])["metrics"][
        "failure_recurrence"
    ]

    assert metric["numerator"] == 0
    assert metric["denominator"] == 0
    assert metric["eligible"] == 0
    assert metric["missing"] == 1


def test_lesson_funnel_uses_stage_denominators_and_requires_attribution_subset():
    retrieved_only = row(applied_attributed_lesson_ids=[])
    applied_failed = row(
        task_fingerprint="fp-applied-failed",
        task_id="task-applied-failed",
        attempt_id="applied-failed",
        action_id="applied-failed-action",
        verifier_status="failed",
        terminal_outcome="FAILED",
    )
    no_retrieval = row(
        task_fingerprint="fp-none",
        task_id="task-none",
        attempt_id="none",
        action_id="none-action",
        retrieved_lesson_ids=[],
        applied_attributed_lesson_ids=[],
    )

    funnel = replay_scorecard([retrieved_only, applied_failed, no_retrieval])["metrics"][
        "retrieved_to_applied_to_qualified_useful"
    ]

    assert (funnel["retrieved"]["numerator"], funnel["retrieved"]["denominator"]) == (2, 3)
    assert (funnel["applied"]["numerator"], funnel["applied"]["denominator"]) == (1, 2)
    assert (funnel["qualified_useful"]["numerator"], funnel["qualified_useful"]["denominator"]) == (
        0,
        1,
    )
    with pytest.raises(ReplayContractError):
        replay_scorecard([
            row(retrieved_lesson_ids=["lesson-1"], applied_attributed_lesson_ids=["lesson-2"])
        ])


def test_every_metric_has_complete_denominator_contract():
    metrics = replay_scorecard([row()])["metrics"]
    for metric in metrics.values():
        if (
            metric["claim_ceiling"]
            if isinstance(metric, dict) and "claim_ceiling" in metric
            else False
        ):
            assert {
                "numerator",
                "denominator",
                "eligible",
                "missing",
                "exclusions",
                "claim_ceiling",
            } <= metric.keys()
        else:
            for stage in metric.values():
                assert {
                    "numerator",
                    "denominator",
                    "eligible",
                    "missing",
                    "exclusions",
                    "claim_ceiling",
                } <= stage.keys()


def test_paired_uplift_is_separate_and_requires_strict_evidence():
    off = row(
        task_fingerprint="fp-pair",
        task_id="task-pair",
        attempt_id="off",
        action_id="off",
        memory_arm="memory_off",
        verifier_status="failed",
        terminal_outcome="FAILED",
    )
    on = row(
        task_fingerprint="fp-pair",
        task_id="task-pair",
        attempt_id="on",
        action_id="on",
        memory_arm="memory_on",
    )
    result = paired_memory_uplift([on, off])
    assert result["eligible"] == result["eligible_pairs"] == result["numerator"] == 1
    assert result["denominator"] == 1
    assert result["uplift_claimable"] is False
    assert replay_scorecard([off, on])["metrics"] != result
    with pytest.raises(ReplayContractError):
        paired_memory_uplift([off, dict(on, verifier_receipt="")])


@pytest.mark.parametrize(
    "mutator",
    [
        lambda off, on: [off, dict(on, task_id="other-task")],
        lambda off, on: [off, dict(on, source_revision="other-revision")],
        lambda off, on: [off, dict(on, source_tree="other-tree")],
        lambda off, on: [off, dict(on, attempt_index=1)],
        lambda off, on: [dict(off, verifier_status="passed", terminal_outcome="SUCCEEDED"), on],
        lambda off, on: [off, dict(on, verifier_status="failed", terminal_outcome="FAILED")],
        lambda off, on: [
            off,
            on,
            dict(on, attempt_id="on-2", action_id="on-action-2", attempt_index=1),
        ],
        lambda off, on: [
            off,
            on,
            dict(off, attempt_id="off-2", action_id="off-action-2", attempt_index=1),
        ],
    ],
)
def test_paired_uplift_excludes_cross_identity_wrong_outcome_and_multiple_arms(mutator):
    off = row(
        task_fingerprint="fp-strict-pair",
        task_id="task-strict-pair",
        attempt_id="off",
        action_id="off-action",
        memory_arm="memory_off",
        verifier_status="failed",
        terminal_outcome="FAILED",
    )
    on = row(
        task_fingerprint="fp-strict-pair",
        task_id="task-strict-pair",
        attempt_id="on",
        action_id="on-action",
        memory_arm="memory_on",
    )

    result = paired_memory_uplift(mutator(off, on))

    assert result["eligible_pairs"] == 0
    assert result["numerator"] == 0
    assert result["uplift_claimable"] is False
    assert result["exclusions"]


def test_paired_structural_exclusion_is_not_missing_telemetry():
    off = row(
        task_fingerprint="fp-structural-pair",
        task_id="task-a",
        attempt_id="off",
        action_id="off-action",
        memory_arm="memory_off",
        verifier_status="failed",
        terminal_outcome="FAILED",
    )
    on = row(
        task_fingerprint="fp-structural-pair",
        task_id="task-b",
        attempt_id="on",
        action_id="on-action",
        memory_arm="memory_on",
    )

    structural = paired_memory_uplift([off, on])

    assert structural["denominator"] == 1
    assert structural["eligible"] == structural["eligible_pairs"] == 0
    assert structural["missing"] == structural["missing_telemetry"] == 0
    assert structural["exclusions"] == ["fp-structural-pair:task_id_mismatch"]

    missing_arm = paired_memory_uplift([off])
    assert missing_arm["denominator"] == 1
    assert missing_arm["eligible"] == missing_arm["eligible_pairs"] == 0
    assert missing_arm["missing"] == missing_arm["missing_telemetry"] == 1
    assert missing_arm["exclusions"] == ["fp-structural-pair:missing_arm"]


def test_paired_complete_ineligible_arm_is_excluded_without_missing_telemetry():
    off = row(
        task_fingerprint="fp-ineligible-pair",
        task_id="task-ineligible-pair",
        attempt_id="off",
        action_id="off-action",
        memory_arm="memory_off",
        verifier_status="failed",
        terminal_outcome="FAILED",
        ineligibility_reasons=["verifier_scope:mismatch"],
    )
    on = row(
        task_fingerprint="fp-ineligible-pair",
        task_id="task-ineligible-pair",
        attempt_id="on",
        action_id="on-action",
        memory_arm="memory_on",
    )

    result = paired_memory_uplift([off, on])

    assert result["eligible"] == result["eligible_pairs"] == 0
    assert result["missing"] == result["missing_telemetry"] == 0
    assert result["exclusions"] == ["fp-ineligible-pair:ineligible_arm"]


@pytest.mark.parametrize(
    ("on_overrides", "expected_reason"),
    [
        (
            {
                "task_id": "other-task",
                "measured_elapsed_seconds": None,
                "missingness_reasons": ["measured_elapsed_seconds:unreported"],
            },
            "task_id_mismatch",
        ),
        (
            {
                "source_revision": "other-revision",
                "source_tree": "other-tree",
                "verifier_artifact": None,
                "missingness_reasons": ["verifier_artifact:unreported"],
            },
            "source_identity_mismatch",
        ),
        (
            {
                "attempt_index": 1,
                "forbidden_strategy_violation_event": None,
                "missingness_reasons": ["forbidden_strategy_violation_event:unreported"],
            },
            "attempt_index_mismatch",
        ),
    ],
)
def test_paired_structural_mismatch_precedes_combined_missing(on_overrides, expected_reason):
    off = row(
        task_fingerprint="fp-precedence-pair",
        task_id="task-precedence-pair",
        attempt_id="off",
        action_id="off-action",
        memory_arm="memory_off",
        verifier_status="failed",
        terminal_outcome="FAILED",
    )
    on = dict(
        row(
            task_fingerprint="fp-precedence-pair",
            task_id="task-precedence-pair",
            attempt_id="on",
            action_id="on-action",
            memory_arm="memory_on",
        ),
        **on_overrides,
    )

    result = paired_memory_uplift([off, on])

    assert result["eligible"] == result["eligible_pairs"] == 0
    assert result["missing"] == result["missing_telemetry"] == 0
    assert result["exclusions"] == [f"fp-precedence-pair:{expected_reason}"]


def test_paired_ineligible_arm_precedes_combined_missing():
    off = row(
        task_fingerprint="fp-ineligible-missing-pair",
        task_id="task-ineligible-missing-pair",
        attempt_id="off",
        action_id="off-action",
        memory_arm="memory_off",
        verifier_status="failed",
        terminal_outcome="FAILED",
        measured_elapsed_seconds=None,
        verifier_receipt=None,
        missingness_reasons=[
            "measured_elapsed_seconds:unreported",
            "verifier_receipt:unreported",
        ],
        ineligibility_reasons=["verifier_scope:mismatch"],
    )
    on = row(
        task_fingerprint="fp-ineligible-missing-pair",
        task_id="task-ineligible-missing-pair",
        attempt_id="on",
        action_id="on-action",
        memory_arm="memory_on",
    )

    result = paired_memory_uplift([off, on])

    assert result["eligible"] == result["eligible_pairs"] == 0
    assert result["missing"] == result["missing_telemetry"] == 0
    assert result["exclusions"] == ["fp-ineligible-missing-pair:ineligible_arm"]


def test_paired_true_missing_telemetry_artifact_and_receipt_remain_missing():
    off = row(
        task_fingerprint="fp-missing-pair",
        task_id="task-missing-pair",
        attempt_id="off",
        action_id="off-action",
        memory_arm="memory_off",
        verifier_status="failed",
        terminal_outcome="FAILED",
    )
    on = row(
        task_fingerprint="fp-missing-pair",
        task_id="task-missing-pair",
        attempt_id="on",
        action_id="on-action",
        memory_arm="memory_on",
    )

    missing_elapsed = paired_memory_uplift([
        off,
        dict(
            on,
            measured_elapsed_seconds=None,
            missingness_reasons=["measured_elapsed_seconds:unreported"],
        ),
    ])
    assert missing_elapsed["missing"] == missing_elapsed["missing_telemetry"] == 1
    assert missing_elapsed["exclusions"] == ["fp-missing-pair:missing_telemetry"]

    missing_intervention = paired_memory_uplift([
        off,
        dict(
            on,
            intervention_events=None,
            intervention_count=None,
            missingness_reasons=[
                "intervention_events:unreported",
                "intervention_count:unreported",
            ],
        ),
    ])
    assert missing_intervention["missing"] == missing_intervention["missing_telemetry"] == 1
    assert missing_intervention["exclusions"] == ["fp-missing-pair:missing_telemetry"]

    missing_violation = paired_memory_uplift([
        off,
        dict(
            on,
            forbidden_strategy_violation_event=None,
            missingness_reasons=["forbidden_strategy_violation_event:unreported"],
        ),
    ])
    assert missing_violation["missing"] == missing_violation["missing_telemetry"] == 1
    assert missing_violation["exclusions"] == ["fp-missing-pair:missing_telemetry"]

    missing_evidence = paired_memory_uplift([
        off,
        dict(
            on,
            verifier_artifact=None,
            verifier_artifact_hash=None,
            verifier_receipt=None,
            missingness_reasons=[
                "verifier_artifact:unreported",
                "verifier_artifact_hash:unreported",
                "verifier_receipt:unreported",
            ],
        ),
    ])
    assert missing_evidence["missing"] == missing_evidence["missing_telemetry"] == 1
    assert missing_evidence["exclusions"] == ["fp-missing-pair:missing_artifact_or_receipt"]


def test_no_authority_or_adaptation_side_effect():
    result = replay_scorecard([row()])
    assert result["observational_only"] is True
    assert result["adaptation_applied"] is False
    assert result["authority_effect"] is False


def test_output_is_deterministic_json_and_duplicate_identity_collisions_reject():
    def contains_tuple(value):
        if isinstance(value, tuple):
            return True
        if isinstance(value, dict):
            return any(contains_tuple(item) for item in value.values())
        if isinstance(value, list):
            return any(contains_tuple(item) for item in value)
        return False

    first = row()
    second = row(
        task_fingerprint="fp-json",
        task_id="task-json",
        attempt_id="json",
        action_id="json-action",
        retrieved_lesson_ids=["lesson-b", "lesson-a"],
        applied_attributed_lesson_ids=["lesson-a"],
    )
    before = deepcopy([first, second])

    forward = replay_scorecard([first, second])
    reverse = replay_scorecard([second, first])

    assert json.dumps(forward, sort_keys=True, allow_nan=False) == json.dumps(
        reverse, sort_keys=True, allow_nan=False
    )
    assert [first, second] == before
    assert contains_tuple(forward) is False
    with pytest.raises(ReplayContractError):
        replay_scorecard([first, dict(first, measured_elapsed_seconds=99.0)])
