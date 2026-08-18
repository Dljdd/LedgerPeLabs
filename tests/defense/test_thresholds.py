"""Exhaustive matched-budget threshold selection tests."""

from __future__ import annotations

import hashlib
import json
import math
import time
from collections.abc import Callable

import numpy as np
import pytest
from pydantic import ValidationError

import apar.defense.thresholds as threshold_module
from apar.contracts.decisions import Action
from apar.defense.policy import OperatingBudget
from apar.defense.thresholds import (
    ThresholdContractError,
    ThresholdReport,
    select_policy_thresholds,
)
from apar.runs.wire import canonical_json_bytes


def _rechecksum_threshold_document(document: dict[str, object]) -> bytes:
    without_digest = dict(document)
    without_digest.pop("report_digest")
    document["report_digest"] = hashlib.sha256(
        canonical_json_bytes(without_digest)
    ).hexdigest()
    return canonical_json_bytes(document)


def _actions(*values: Action) -> np.ndarray:
    return np.array(values, dtype=object)


def _one_case_per_intervention(actions: np.ndarray) -> int:
    return int(sum(action is not Action.APPROVE for action in actions))


def _zero_cases(_actions: np.ndarray) -> int:
    return 0


def test_hand_oracle_exhaustive_candidate_count_inclusivity_and_tie_break() -> None:
    scores = np.array([0.2, 0.5, 0.8, 0.9], dtype=np.float64)
    labels = np.array([0, 0, 1, 1], dtype=np.int8)
    mandatory = _actions(Action.APPROVE, Action.APPROVE, Action.APPROVE, Action.APPROVE)
    budget = OperatingBudget(
        false_decline_rate_max=0.5,
        challenge_rate_max=0.25,
        review_case_rate_max=1.0,
    )

    report = select_policy_thresholds(
        scores,
        labels,
        mandatory,
        _zero_cases,
        budget,
    )

    # Six candidates (four scores plus 0 and 1) yield 6*7/2 ordered pairs.
    assert report.candidate_count == 21
    assert report.feasible_candidate_count > 0
    assert report.feasible is True
    assert report.thresholds is not None
    # Both fraud rows are intercepted; tie chooses no legitimate interventions,
    # then the highest decline and challenge boundaries. score == boundary acts.
    assert report.thresholds.decline == 0.9
    assert report.thresholds.challenge == 0.8
    assert report.objective_value == 1.0
    assert report.false_intervention_count == 0
    assert report.intervention_count == 2
    assert report.calibration_false_decline_rate == 0.0
    assert report.calibration_challenge_rate == 0.25


def test_threshold_one_is_disabled_while_mandatory_decline_has_precedence() -> None:
    scores = np.array([0.2, 0.8], dtype=np.float64)
    labels = np.array([0, 1], dtype=np.int8)
    mandatory = _actions(Action.APPROVE, Action.DECLINE)
    report = select_policy_thresholds(
        scores,
        labels,
        mandatory,
        _zero_cases,
        OperatingBudget(
            false_decline_rate_max=1.0,
            challenge_rate_max=0.0,
            review_case_rate_max=1.0,
        ),
        values=np.array([0.0, 10.0], dtype=np.float64),
    )

    assert report.feasible
    assert report.thresholds is not None
    assert report.thresholds.challenge == 1.0
    assert report.thresholds.decline == 1.0
    assert report.objective_value == 10.0
    assert report.calibration_false_decline_rate == 0.0
    assert report.false_intervention_count == 0


def test_raw_zero_and_one_scores_are_normalized_and_both_digests_are_bound() -> None:
    scores = np.array([0.0, 1.0], dtype=np.float64)
    normalized = threshold_module.normalize_operating_scores(scores)
    report = select_policy_thresholds(
        scores,
        np.array([0, 1], dtype=np.int8),
        _actions(Action.APPROVE, Action.APPROVE),
        _zero_cases,
        OperatingBudget(
            false_decline_rate_max=0.0,
            challenge_rate_max=0.0,
            review_case_rate_max=1.0,
        ),
    )

    np.testing.assert_array_equal(normalized, np.array([1e-8, 1.0 - 1e-8]))
    np.testing.assert_array_equal(scores, np.array([0.0, 1.0]))
    assert report.thresholds is not None
    assert report.thresholds.challenge == 1.0 - 1e-8
    assert report.thresholds.decline == 1.0 - 1e-8
    assert report.input_scores_digest != report.normalized_scores_digest


def test_mandatory_decline_can_make_budget_infeasible_without_relaxation() -> None:
    report = select_policy_thresholds(
        np.array([0.1, 0.9], dtype=np.float64),
        np.array([0, 1], dtype=np.int8),
        _actions(Action.DECLINE, Action.APPROVE),
        _one_case_per_intervention,
        OperatingBudget(
            false_decline_rate_max=0.0,
            challenge_rate_max=1.0,
            review_case_rate_max=1.0,
        ),
    )

    assert report.feasible is False
    assert report.thresholds is None
    assert report.reason == "no_candidate_satisfies_operating_budget"
    assert report.feasible_candidate_count == 0
    assert report.calibration_false_decline_rate is None
    assert report.calibration_challenge_rate is None
    assert report.calibration_review_case_rate is None
    assert report.selected_actions_digest is None
    assert report.minimum_false_decline_rate == 1.0


def test_challenge_budget_excludes_declines_and_false_tie_counts_both() -> None:
    scores = np.array([0.1, 0.4, 0.8, 0.9], dtype=np.float64)
    labels = np.array([0, 0, 1, 1], dtype=np.int8)
    report = select_policy_thresholds(
        scores,
        labels,
        _actions(*([Action.APPROVE] * 4)),
        _zero_cases,
        OperatingBudget(
            false_decline_rate_max=1.0,
            challenge_rate_max=0.0,
            review_case_rate_max=1.0,
        ),
    )

    assert report.feasible
    assert report.thresholds is not None
    assert report.thresholds.challenge == report.thresholds.decline
    assert report.calibration_challenge_rate == 0.0
    assert report.false_intervention_count == 0


def test_values_change_objective_from_recall_to_captured_value() -> None:
    scores = np.array([0.1, 0.6, 0.7], dtype=np.float64)
    labels = np.array([0, 1, 1], dtype=np.int8)
    mandatory = _actions(*([Action.APPROVE] * 3))
    budget = OperatingBudget(
        false_decline_rate_max=0.0,
        challenge_rate_max=1 / 3,
        review_case_rate_max=1.0,
    )
    recall = select_policy_thresholds(scores, labels, mandatory, _zero_cases, budget)
    value = select_policy_thresholds(
        scores,
        labels,
        mandatory,
        _zero_cases,
        budget,
        values=np.array([0.0, 0.0, 100.0], dtype=np.float64),
    )

    assert recall.objective_kind == "fraud_recall"
    assert recall.objective_value == 1.0
    assert value.objective_kind == "fraud_value_captured"
    assert value.objective_value == 100.0
    assert recall.thresholds is not None
    assert value.thresholds is not None
    assert recall.thresholds.challenge == 0.6
    assert value.thresholds.challenge == 0.7


def test_callback_receives_fresh_isolated_arrays_and_is_checked_for_determinism() -> None:
    seen: list[np.ndarray] = []

    def callback(actions: np.ndarray) -> int:
        assert actions.dtype == object
        assert not any(actions is prior for prior in seen)
        seen.append(actions)
        result = int(sum(action is not Action.APPROVE for action in actions))
        actions[:] = Action.DECLINE
        return result

    report = select_policy_thresholds(
        np.array([0.2, 0.8], dtype=np.float64),
        np.array([0, 1], dtype=np.int8),
        _actions(Action.APPROVE, Action.APPROVE),
        callback,
        OperatingBudget(
            false_decline_rate_max=1.0,
            challenge_rate_max=1.0,
            review_case_rate_max=1.0,
        ),
    )
    assert report.candidate_count == 10
    assert len(seen) == 3 * (report.candidate_threshold_count - 1)


def test_callback_must_be_invariant_to_challenge_decline_severity_for_same_mask() -> None:
    def severity_sensitive(actions: np.ndarray) -> int:
        return int(sum(action is Action.CHALLENGE for action in actions))

    with pytest.raises(ThresholdContractError, match="intervention-mask invariant"):
        select_policy_thresholds(
            np.array([0.2, 0.8], dtype=np.float64),
            np.array([0, 1], dtype=np.int8),
            _actions(Action.APPROVE, Action.APPROVE),
            severity_sensitive,
            OperatingBudget(
                false_decline_rate_max=1.0,
                challenge_rate_max=1.0,
                review_case_rate_max=1.0,
            ),
        )


@pytest.mark.parametrize(
    "callback",
    [
        lambda _actions: True,
        lambda _actions: 1.0,
        lambda _actions: -1,
        lambda actions: len(actions) + 1,
        lambda _actions: (_ for _ in ()).throw(RuntimeError("boom")),
    ],
)
def test_callback_invalid_results_or_exceptions_fail_closed(
    callback: Callable[[np.ndarray], int],
) -> None:
    with pytest.raises(ThresholdContractError, match="review_case_counter"):
        select_policy_thresholds(
            np.array([0.2, 0.8], dtype=np.float64),
            np.array([0, 1], dtype=np.int8),
            _actions(Action.APPROVE, Action.APPROVE),
            callback,
            OperatingBudget(
                false_decline_rate_max=1.0,
                challenge_rate_max=1.0,
                review_case_rate_max=1.0,
            ),
        )


def test_nondeterministic_callback_fails_closed() -> None:
    count = 0

    def callback(_actions: np.ndarray) -> int:
        nonlocal count
        count += 1
        return count % 2

    with pytest.raises(ThresholdContractError, match="deterministic"):
        select_policy_thresholds(
            np.array([0.2, 0.8], dtype=np.float64),
            np.array([0, 1], dtype=np.int8),
            _actions(Action.APPROVE, Action.APPROVE),
            callback,
            OperatingBudget(
                false_decline_rate_max=1.0,
                challenge_rate_max=1.0,
                review_case_rate_max=1.0,
            ),
        )


@pytest.mark.parametrize(
    ("argument", "value"),
    [
        ("scores", [0.2, 0.8]),
        ("scores", np.array([], dtype=np.float64)),
        ("scores", np.array([[0.2, 0.8]], dtype=np.float64)),
        ("scores", np.array([0.2, np.nan], dtype=np.float64)),
        ("labels", np.array([0, 2], dtype=np.int8)),
        ("mandatory_actions", np.array(["approve", "approve"], dtype=object)),
        (
            "mandatory_actions",
            _actions(Action.APPROVE, Action.CHALLENGE),
        ),
        ("values", np.array([0.0, -1.0], dtype=np.float64)),
        ("values", np.array([0.0, np.inf], dtype=np.float64)),
        ("values", np.array([[0.0, 1.0]], dtype=np.float64)),
    ],
)
def test_threshold_selection_rejects_invalid_exact_inputs(argument: str, value: object) -> None:
    arguments: dict[str, object] = {
        "scores": np.array([0.2, 0.8], dtype=np.float64),
        "labels": np.array([0, 1], dtype=np.int8),
        "mandatory_actions": _actions(Action.APPROVE, Action.APPROVE),
        "review_case_counter": _zero_cases,
        "budget": OperatingBudget(
            false_decline_rate_max=1.0,
            challenge_rate_max=1.0,
            review_case_rate_max=1.0,
        ),
        "values": None,
    }
    arguments[argument] = value
    with pytest.raises(ThresholdContractError):
        select_policy_thresholds(**arguments)  # type: ignore[arg-type]


@pytest.mark.parametrize("argument", ["scores", "labels", "values"])
def test_threshold_selection_rejects_complex_dtype_before_conversion(argument: str) -> None:
    arguments: dict[str, object] = {
        "scores": np.array([0.2, 0.8], dtype=np.float64),
        "labels": np.array([0, 1], dtype=np.int8),
        "mandatory_actions": _actions(Action.APPROVE, Action.APPROVE),
        "review_case_counter": _zero_cases,
        "budget": OperatingBudget(
            false_decline_rate_max=1.0,
            challenge_rate_max=1.0,
            review_case_rate_max=1.0,
        ),
        "values": np.array([0.0, 1.0], dtype=np.float64),
    }
    arguments[argument] = np.array([0.0 + 0.0j, 1.0 + 0.0j], dtype=np.complex128)
    with pytest.raises(ThresholdContractError, match="real"):
        select_policy_thresholds(**arguments)  # type: ignore[arg-type]


def test_threshold_selection_rejects_alignment_zero_denominators_and_wrong_budget() -> None:
    scores = np.array([0.2, 0.8], dtype=np.float64)
    labels = np.array([0, 1], dtype=np.int8)
    actions = _actions(Action.APPROVE, Action.APPROVE)
    budget = OperatingBudget(
        false_decline_rate_max=1.0,
        challenge_rate_max=1.0,
        review_case_rate_max=1.0,
    )
    with pytest.raises(ThresholdContractError, match="equal lengths"):
        select_policy_thresholds(scores, labels[:1], actions, _zero_cases, budget)
    with pytest.raises(ThresholdContractError, match="legitimate"):
        select_policy_thresholds(
            scores, np.array([1, 1], dtype=np.int8), actions, _zero_cases, budget
        )
    with pytest.raises(ThresholdContractError, match="fraud"):
        select_policy_thresholds(
            scores, np.array([0, 0], dtype=np.int8), actions, _zero_cases, budget
        )
    with pytest.raises(ThresholdContractError, match="budget"):
        select_policy_thresholds(scores, labels, actions, _zero_cases, {"x": 1})  # type: ignore[arg-type]


def test_aligned_permutation_preserves_selected_operating_point_and_rates() -> None:
    scores = np.array([0.2, 0.4, 0.8, 0.9], dtype=np.float64)
    labels = np.array([0, 0, 1, 1], dtype=np.int8)
    mandatory = _actions(*([Action.APPROVE] * 4))
    values = np.array([0.0, 0.0, 2.0, 3.0], dtype=np.float64)
    budget = OperatingBudget(
        false_decline_rate_max=0.5,
        challenge_rate_max=0.5,
        review_case_rate_max=1.0,
    )
    first = select_policy_thresholds(scores, labels, mandatory, _zero_cases, budget, values)
    permutation = np.array([2, 0, 3, 1])
    second = select_policy_thresholds(
        scores[permutation],
        labels[permutation],
        mandatory[permutation],
        _zero_cases,
        budget,
        values[permutation],
    )

    assert first.thresholds == second.thresholds
    assert first.objective_value == second.objective_value
    assert first.calibration_false_decline_rate == second.calibration_false_decline_rate
    assert first.calibration_challenge_rate == second.calibration_challenge_rate
    assert first.input_scores_digest != second.input_scores_digest


def test_report_roundtrip_is_canonical_and_binds_inputs_actions_budget_and_counts() -> None:
    report = select_policy_thresholds(
        np.array([0.2, 0.8], dtype=np.float64),
        np.array([0, 1], dtype=np.int8),
        _actions(Action.APPROVE, Action.APPROVE),
        _zero_cases,
        OperatingBudget(
            false_decline_rate_max=1.0,
            challenge_rate_max=1.0,
            review_case_rate_max=1.0,
        ),
        np.array([0.0, 4.0], dtype=np.float64),
    )
    payload = report.to_json()
    restored = ThresholdReport.from_json(payload)
    assert restored == report
    assert payload == canonical_json_bytes(json.loads(payload))
    assert report.input_scores_digest
    assert report.normalized_scores_digest
    assert report.input_labels_digest
    assert report.input_mandatory_actions_digest
    assert report.input_values_digest
    assert report.selected_actions_digest
    assert report.row_count == 2
    assert report.legitimate_count == 1
    assert report.fraud_count == 1
    assert report.budget == OperatingBudget(
        false_decline_rate_max=1.0,
        challenge_rate_max=1.0,
        review_case_rate_max=1.0,
    )


def test_report_rejects_noncanonical_extra_and_digest_tampering() -> None:
    report = select_policy_thresholds(
        np.array([0.2, 0.8], dtype=np.float64),
        np.array([0, 1], dtype=np.int8),
        _actions(Action.APPROVE, Action.APPROVE),
        _zero_cases,
        OperatingBudget(
            false_decline_rate_max=1.0,
            challenge_rate_max=1.0,
            review_case_rate_max=1.0,
        ),
    )
    document = json.loads(report.to_json())
    document["candidate_count"] += 1
    with pytest.raises(
        (ThresholdContractError, ValidationError), match="digest|triangular"
    ):
        ThresholdReport.from_json(canonical_json_bytes(document))

    document = json.loads(report.to_json())
    document["extra"] = "no"
    with pytest.raises((ThresholdContractError, ValidationError)):
        ThresholdReport.from_json(canonical_json_bytes(document))

    pretty = json.dumps(json.loads(report.to_json()), indent=2).encode()
    with pytest.raises(ThresholdContractError, match="canonical"):
        ThresholdReport.from_json(pretty)


def test_report_contract_rejects_inconsistent_feasible_fields() -> None:
    report = select_policy_thresholds(
        np.array([0.2, 0.8], dtype=np.float64),
        np.array([0, 1], dtype=np.int8),
        _actions(Action.APPROVE, Action.APPROVE),
        _zero_cases,
        OperatingBudget(
            false_decline_rate_max=1.0,
            challenge_rate_max=1.0,
            review_case_rate_max=1.0,
        ),
    )
    document = report.model_dump(mode="json")
    document["thresholds"] = None
    with pytest.raises(ValidationError, match="feasible report"):
        ThresholdReport.model_validate(document)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ({"objective_value": 2.0}, "recall objective"),
        ({"candidate_count": 1}, "triangular"),
        ({"input_values_digest": "0" * 64}, "values digest"),
        (
            {"objective_kind": "fraud_value_captured", "input_values_digest": None},
            "values digest",
        ),
        ({"minimum_false_decline_rate": 0.5}, "minimum"),
    ],
)
def test_rechecksummed_report_rejects_semantic_tampering(
    mutation: dict[str, object], message: str
) -> None:
    report = select_policy_thresholds(
        np.array([0.2, 0.8], dtype=np.float64),
        np.array([0, 1], dtype=np.int8),
        _actions(Action.APPROVE, Action.APPROVE),
        _zero_cases,
        OperatingBudget(
            false_decline_rate_max=1.0,
            challenge_rate_max=1.0,
            review_case_rate_max=1.0,
        ),
    )
    document = json.loads(report.to_json())
    document.update(mutation)
    with pytest.raises((ThresholdContractError, ValidationError), match=message):
        ThresholdReport.from_json(_rechecksum_threshold_document(document))


def test_rechecksummed_infeasible_report_rejects_claimed_realized_state() -> None:
    report = select_policy_thresholds(
        np.array([0.2, 0.8], dtype=np.float64),
        np.array([0, 1], dtype=np.int8),
        _actions(Action.DECLINE, Action.APPROVE),
        _zero_cases,
        OperatingBudget(
            false_decline_rate_max=0.0,
            challenge_rate_max=1.0,
            review_case_rate_max=1.0,
        ),
    )
    assert not report.feasible
    document = json.loads(report.to_json())
    document["objective_value"] = 0.0
    with pytest.raises((ThresholdContractError, ValidationError), match="infeasible report"):
        ThresholdReport.from_json(_rechecksum_threshold_document(document))


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            {"candidate_threshold_count": 2, "candidate_count": 3},
            "candidate_threshold_count",
        ),
        (
            {
                "candidate_threshold_count": 4099,
                "candidate_count": 4099 * 4100 // 2,
            },
            "candidate_threshold_count",
        ),
        (
            {
                "candidate_threshold_count": 100_000,
                "candidate_count": 100_000 * 100_001 // 2,
            },
            "candidate_threshold_count",
        ),
        (
            {"review_case_count": 1, "calibration_review_case_rate": 0.0},
            "review-case rate",
        ),
        (
            {
                "false_intervention_count": 0,
                "calibration_false_decline_rate": 1.0,
            },
            "false-decline count",
        ),
        ({"objective_value": 0.5}, "fraud-recall count"),
    ],
)
def test_rechecksummed_report_rejects_impossible_counts_and_candidate_space(
    mutation: dict[str, object], message: str
) -> None:
    report = select_policy_thresholds(
        np.array([0.2, 0.8], dtype=np.float64),
        np.array([0, 1], dtype=np.int8),
        _actions(Action.APPROVE, Action.APPROVE),
        _zero_cases,
        OperatingBudget(
            false_decline_rate_max=1.0,
            challenge_rate_max=1.0,
            review_case_rate_max=1.0,
        ),
    )
    document = json.loads(report.to_json())
    document.update(mutation)
    with pytest.raises((ThresholdContractError, ValidationError), match=message):
        ThresholdReport.from_json(_rechecksum_threshold_document(document))


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ({"objective_value": 2e-12}, "fraud-recall count"),
        ({"minimum_challenge_rate": 1e-6}, "minimum challenge rate"),
    ],
)
def test_rate_count_integrality_tolerance_rejects_material_near_integers(
    mutation: dict[str, object], message: str
) -> None:
    report = select_policy_thresholds(
        np.array([0.2, 0.8], dtype=np.float64),
        np.array([0, 1], dtype=np.int8),
        _actions(Action.APPROVE, Action.APPROVE),
        _zero_cases,
        OperatingBudget(
            false_decline_rate_max=1.0,
            challenge_rate_max=1.0,
            review_case_rate_max=1.0,
        ),
    )
    document = json.loads(report.to_json())
    document.update(mutation)
    with pytest.raises((ThresholdContractError, ValidationError), match=message):
        ThresholdReport.from_json(_rechecksum_threshold_document(document))


def _brute_force_thresholds(
    raw_scores: np.ndarray,
    labels: np.ndarray,
    mandatory: np.ndarray,
    budget: OperatingBudget,
    values: np.ndarray | None,
) -> tuple[bool, tuple[float, float] | None, float | None, int | None, int]:
    scores = np.clip(raw_scores, 1e-8, 1.0 - 1e-8)
    candidates = sorted({0.0, 1.0, *(float(value) for value in scores)})
    legitimate = labels == 0
    fraud = labels == 1
    selected: tuple[tuple[float, int, float, float], float, float, int] | None = None
    for decline in candidates:
        for challenge in candidates:
            if challenge > decline:
                continue
            actions = _actions(*([Action.APPROVE] * len(scores)))
            for index, score in enumerate(scores):
                if mandatory[index] is Action.DECLINE or score >= decline:
                    actions[index] = Action.DECLINE
                elif score >= challenge:
                    actions[index] = Action.CHALLENGE
            false_declines = int(
                sum(
                    legitimate[index] and action is Action.DECLINE
                    for index, action in enumerate(actions)
                )
            )
            challenges = int(sum(action is Action.CHALLENGE for action in actions))
            false_interventions = int(
                sum(
                    legitimate[index] and action is not Action.APPROVE
                    for index, action in enumerate(actions)
                )
            )
            if false_declines / int(legitimate.sum()) > budget.false_decline_rate_max:
                continue
            if challenges / len(scores) > budget.challenge_rate_max:
                continue
            intervened_fraud = np.array(
                [
                    fraud[index] and action is not Action.APPROVE
                    for index, action in enumerate(actions)
                ]
            )
            objective = (
                math.fsum(float(value) for value in values[intervened_fraud])
                if values is not None
                else float(intervened_fraud.sum() / fraud.sum())
            )
            ranking = (objective, -false_interventions, decline, challenge)
            if selected is None or ranking > selected[0]:
                selected = (ranking, challenge, decline, false_interventions)
    if selected is None:
        return False, None, None, None, len(candidates) * (len(candidates) + 1) // 2
    return (
        True,
        (selected[1], selected[2]),
        selected[0][0],
        selected[3],
        len(candidates) * (len(candidates) + 1) // 2,
    )


def test_value_objective_matches_literal_exhaustive_adversarial_fixture() -> None:
    scores = np.array([0.3, 0.5, 0.4, 0.5], dtype=np.float64)
    labels = np.array([0, 1, 1, 1], dtype=np.int8)
    values = np.array([0.1, 1.0, 0.2, 1e16], dtype=np.float64)
    mandatory = _actions(*([Action.APPROVE] * 4))
    budget = OperatingBudget(
        challenge_rate_max=0.0,
        false_decline_rate_max=1.0,
        review_case_rate_max=1.0,
    )

    expected = _brute_force_thresholds(scores, labels, mandatory, budget, values)
    report = select_policy_thresholds(
        scores, labels, mandatory, _zero_cases, budget, values
    )

    assert expected[1] == (0.4, 0.4)
    assert expected[2] == 1.0000000000000002e16
    assert report.thresholds is not None
    assert (report.thresholds.challenge, report.thresholds.decline) == expected[1]
    assert report.objective_value == expected[2]


def test_value_objective_matches_brute_force_for_mixed_significands_and_magnitudes() -> None:
    rng = np.random.default_rng(8260816)
    score_choices = np.array([0.0, 0.15, 0.35, 0.55, 0.75, 1.0])
    significands = np.array([0.1, 0.2, 1.0, 1.5, 3.7, 9.9])
    for _ in range(40):
        scores = rng.choice(score_choices, size=14).astype(np.float64)
        labels = rng.integers(0, 2, size=14, dtype=np.int8)
        labels[0] = 0
        labels[1] = 1
        exponents = rng.integers(-12, 17, size=14)
        values = rng.choice(significands, size=14) * np.power(10.0, exponents)
        mandatory = _actions(
            *(
                Action.DECLINE if value else Action.APPROVE
                for value in rng.integers(0, 7, size=14) == 0
            )
        )
        budget = OperatingBudget(
            challenge_rate_max=float(rng.choice([0.0, 0.25, 0.5, 1.0])),
            false_decline_rate_max=float(rng.choice([0.0, 0.5, 1.0])),
            review_case_rate_max=1.0,
        )
        expected = _brute_force_thresholds(scores, labels, mandatory, budget, values)
        report = select_policy_thresholds(
            scores, labels, mandatory, _zero_cases, budget, values
        )
        assert report.feasible == expected[0]
        assert report.candidate_count == expected[4]
        if report.feasible:
            assert report.thresholds is not None
            assert (report.thresholds.challenge, report.thresholds.decline) == expected[1]
            assert report.objective_value == expected[2]
            assert report.false_intervention_count == expected[3]


def test_optimized_selector_matches_brute_force_randomized_hand_oracle() -> None:
    rng = np.random.default_rng(260816)
    score_choices = np.array([0.0, 0.1, 0.3, 0.7, 1.0])
    for case in range(30):
        raw_scores = rng.choice(score_choices, size=9).astype(np.float64)
        labels = rng.integers(0, 2, size=9, dtype=np.int8)
        labels[0] = 0
        labels[1] = 1
        mandatory = _actions(
            *(
                Action.DECLINE if value else Action.APPROVE
                for value in rng.integers(0, 5, size=9) == 0
            )
        )
        budget = OperatingBudget(
            false_decline_rate_max=float(rng.choice([0.0, 0.25, 0.5, 1.0])),
            challenge_rate_max=float(rng.choice([0.0, 0.25, 0.5, 1.0])),
            review_case_rate_max=1.0,
        )
        values = (
            rng.integers(0, 20, size=9).astype(np.float64) if case % 2 else None
        )
        expected = _brute_force_thresholds(raw_scores, labels, mandatory, budget, values)
        actual = select_policy_thresholds(
            raw_scores, labels, mandatory, _zero_cases, budget, values
        )
        assert actual.feasible == expected[0]
        assert actual.candidate_count == expected[4]
        if actual.feasible:
            assert actual.thresholds is not None
            assert (actual.thresholds.challenge, actual.thresholds.decline) == expected[1]
            assert actual.objective_value == expected[2]
            assert actual.false_intervention_count == expected[3]


def test_callback_calls_scale_with_unique_challenge_masks_not_candidate_pairs() -> None:
    calls: list[int] = []

    def callback(actions: np.ndarray) -> int:
        calls.append(len(actions))
        return 0

    scores = np.linspace(0.0, 1.0, 400, dtype=np.float64)
    labels = np.tile(np.array([0, 1], dtype=np.int8), 200)
    report = select_policy_thresholds(
        scores,
        labels,
        _actions(*([Action.APPROVE] * len(scores))),
        callback,
        OperatingBudget(
            false_decline_rate_max=1.0,
            challenge_rate_max=1.0,
            review_case_rate_max=1.0,
        ),
    )
    assert report.candidate_count > 80_000
    assert len(calls) == 3 * (report.candidate_threshold_count - 1)


def test_threshold_selection_benchmark_scales_proportionately() -> None:
    durations: dict[int, float] = {}
    for size in (100, 200, 400):
        scores = np.linspace(0.0, 1.0, size, dtype=np.float64)
        labels = np.tile(np.array([0, 1], dtype=np.int8), size // 2)
        started = time.perf_counter()
        select_policy_thresholds(
            scores,
            labels,
            _actions(*([Action.APPROVE] * size)),
            _zero_cases,
            OperatingBudget(
                false_decline_rate_max=1.0,
                challenge_rate_max=1.0,
                review_case_rate_max=1.0,
            ),
        )
        durations[size] = time.perf_counter() - started
    assert durations[400] < 5.0
    assert durations[400] <= 6.0 * max(durations[200], 0.01)


def test_unique_score_cap_rejects_before_callback_or_quadratic_allocation() -> None:
    assert threshold_module.MAX_UNIQUE_OPERATING_SCORES == 4096
    callback_called = False

    def callback(_actions: np.ndarray) -> int:
        nonlocal callback_called
        callback_called = True
        return 0

    size = threshold_module.MAX_UNIQUE_OPERATING_SCORES + 1
    with pytest.raises(ThresholdContractError, match="4096"):
        select_policy_thresholds(
            np.linspace(0.0, 1.0, size, dtype=np.float64),
            np.tile(np.array([0, 1], dtype=np.int8), size // 2 + 1)[:size],
            _actions(*([Action.APPROVE] * size)),
            callback,
            OperatingBudget(
                false_decline_rate_max=1.0,
                challenge_rate_max=1.0,
                review_case_rate_max=1.0,
            ),
        )
    assert not callback_called
