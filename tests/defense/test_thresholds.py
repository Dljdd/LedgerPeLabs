"""Exhaustive matched-budget threshold selection tests."""

from __future__ import annotations

import json
from collections.abc import Callable

import numpy as np
import pytest
from pydantic import ValidationError

from apar.contracts.decisions import Action
from apar.defense.policy import OperatingBudget
from apar.defense.thresholds import (
    ThresholdContractError,
    ThresholdReport,
    select_policy_thresholds,
)
from apar.runs.wire import canonical_json_bytes


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
    assert len(seen) == report.candidate_count * 2


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
        ("scores", np.array([0.0, 0.8], dtype=np.float64)),
        ("scores", np.array([0.2, 1.0], dtype=np.float64)),
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
    with pytest.raises((ThresholdContractError, ValidationError), match="digest"):
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
