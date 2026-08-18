"""Exhaustive matched-budget action-threshold selection."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Callable
from typing import Literal, cast

import numpy as np
from numpy.typing import NDArray
from pydantic import Field, ValidationError, field_validator, model_validator

from apar.contracts._validation import ExternalContract
from apar.contracts.decisions import Action
from apar.defense.contracts import PolicyThresholds
from apar.defense.policy import OperatingBudget
from apar.runs.wire import WireContractError, canonical_json_bytes, strict_json_loads

_SCORE_MIN = 1e-8
_SCORE_MAX = 1.0 - 1e-8
_SHA256_LENGTH = 64
MAX_UNIQUE_OPERATING_SCORES = 4096


class ThresholdContractError(ValueError):
    """Threshold inputs or a serialized report violate the closed contract."""


class ThresholdReport(ExternalContract):
    """Frozen evidence for an exhaustive, never-relaxed operating-point search."""

    schema_version: Literal["1.0.0"] = "1.0.0"
    feasible: bool
    thresholds: PolicyThresholds | None
    budget: OperatingBudget
    objective_kind: Literal["fraud_recall", "fraud_value_captured"]
    objective_value: float | None = Field(default=None, ge=0.0)
    calibration_false_decline_rate: float | None = Field(default=None, ge=0.0, le=1.0)
    calibration_challenge_rate: float | None = Field(default=None, ge=0.0, le=1.0)
    calibration_review_case_rate: float | None = Field(default=None, ge=0.0, le=1.0)
    false_intervention_count: int | None = Field(default=None, ge=0)
    intervention_count: int | None = Field(default=None, ge=0)
    review_case_count: int | None = Field(default=None, ge=0)
    candidate_count: int = Field(ge=1)
    candidate_threshold_count: int = Field(
        ge=3, le=MAX_UNIQUE_OPERATING_SCORES + 2
    )
    feasible_candidate_count: int = Field(ge=0)
    row_count: int = Field(ge=1)
    legitimate_count: int = Field(ge=1)
    fraud_count: int = Field(ge=1)
    minimum_false_decline_rate: float = Field(ge=0.0, le=1.0)
    minimum_challenge_rate: float = Field(ge=0.0, le=1.0)
    minimum_review_case_rate: float = Field(ge=0.0, le=1.0)
    reason: Literal["selected", "no_candidate_satisfies_operating_budget"]
    input_scores_digest: str
    normalized_scores_digest: str
    input_labels_digest: str
    input_mandatory_actions_digest: str
    input_values_digest: str | None = None
    selected_actions_digest: str | None = None
    report_digest: str

    @field_validator(
        "input_scores_digest",
        "normalized_scores_digest",
        "input_labels_digest",
        "input_mandatory_actions_digest",
        "input_values_digest",
        "selected_actions_digest",
        "report_digest",
    )
    @classmethod
    def digests_are_sha256(cls, value: str | None) -> str | None:
        if value is None:
            return None
        if len(value) != _SHA256_LENGTH:
            raise ValueError("threshold digests must be lowercase SHA-256")
        try:
            int(value, 16)
        except ValueError as error:
            raise ValueError("threshold digests must be lowercase SHA-256") from error
        if value != value.lower():
            raise ValueError("threshold digests must be lowercase SHA-256")
        return value

    @field_validator(
        "objective_value",
        "calibration_false_decline_rate",
        "calibration_challenge_rate",
        "calibration_review_case_rate",
        "minimum_false_decline_rate",
        "minimum_challenge_rate",
        "minimum_review_case_rate",
    )
    @classmethod
    def metrics_are_finite(cls, value: float | None) -> float | None:
        if value is not None and not math.isfinite(value):
            raise ValueError("threshold report metrics must be finite")
        return value

    @model_validator(mode="after")
    def fields_are_consistent(self) -> ThresholdReport:
        realized = (
            self.objective_value,
            self.calibration_false_decline_rate,
            self.calibration_challenge_rate,
            self.calibration_review_case_rate,
            self.false_intervention_count,
            self.intervention_count,
            self.review_case_count,
        )
        if self.feasible:
            if self.thresholds is None or any(value is None for value in realized):
                raise ValueError("feasible report must contain thresholds and realized metrics")
            if self.selected_actions_digest is None or self.reason != "selected":
                raise ValueError("feasible report must bind selected actions")
            if self.feasible_candidate_count < 1:
                raise ValueError("feasible report must count at least one feasible candidate")
            assert self.calibration_false_decline_rate is not None
            assert self.calibration_challenge_rate is not None
            assert self.calibration_review_case_rate is not None
            if self.calibration_false_decline_rate > self.budget.false_decline_rate_max:
                raise ValueError("selected false-decline rate exceeds its frozen budget")
            if self.calibration_challenge_rate > self.budget.challenge_rate_max:
                raise ValueError("selected challenge rate exceeds its frozen budget")
            if self.calibration_review_case_rate > self.budget.review_case_rate_max:
                raise ValueError("selected review-case rate exceeds its frozen budget")
            if (
                self.minimum_false_decline_rate > self.calibration_false_decline_rate
                or self.minimum_challenge_rate > self.calibration_challenge_rate
                or self.minimum_review_case_rate > self.calibration_review_case_rate
            ):
                raise ValueError("minimum rates cannot exceed selected realized rates")
        else:
            if self.thresholds is not None or any(value is not None for value in realized):
                raise ValueError("infeasible report cannot claim thresholds or realized metrics")
            if self.selected_actions_digest is not None:
                raise ValueError("infeasible report cannot bind selected actions")
            if self.feasible_candidate_count != 0:
                raise ValueError("infeasible report cannot count feasible candidates")
            if self.reason != "no_candidate_satisfies_operating_budget":
                raise ValueError("infeasible report must declare the budget failure")
        expected_candidate_count = (
            self.candidate_threshold_count * (self.candidate_threshold_count + 1) // 2
        )
        if self.candidate_count != expected_candidate_count:
            raise ValueError("candidate count must be triangular over all threshold candidates")
        if self.feasible_candidate_count > self.candidate_count:
            raise ValueError("feasible candidate count exceeds exhaustive candidate count")
        if self.legitimate_count + self.fraud_count != self.row_count:
            raise ValueError("threshold report class counts must sum to row count")
        _integral_count(
            self.minimum_false_decline_rate,
            self.legitimate_count,
            label="minimum false-decline rate",
        )
        _integral_count(
            self.minimum_challenge_rate,
            self.row_count,
            label="minimum challenge rate",
        )
        _integral_count(
            self.minimum_review_case_rate,
            self.row_count,
            label="minimum review-case rate",
        )
        if (
            self.false_intervention_count is not None
            and self.false_intervention_count > self.legitimate_count
        ):
            raise ValueError("false intervention count exceeds legitimate rows")
        if self.review_case_count is not None and self.review_case_count > self.row_count:
            raise ValueError("review case count exceeds decision rows")
        if self.intervention_count is not None and self.intervention_count > self.row_count:
            raise ValueError("intervention count exceeds decision rows")
        if self.feasible:
            assert self.calibration_false_decline_rate is not None
            assert self.calibration_challenge_rate is not None
            assert self.calibration_review_case_rate is not None
            assert self.false_intervention_count is not None
            assert self.intervention_count is not None
            assert self.review_case_count is not None
            false_decline_count = _integral_count(
                self.calibration_false_decline_rate,
                self.legitimate_count,
                label="false-decline rate",
            )
            challenge_count = _integral_count(
                self.calibration_challenge_rate,
                self.row_count,
                label="challenge rate",
            )
            review_count = _integral_count(
                self.calibration_review_case_rate,
                self.row_count,
                label="review-case rate",
            )
            if review_count != self.review_case_count:
                raise ValueError("review-case rate does not match review_case_count")
            if false_decline_count > self.false_intervention_count:
                raise ValueError("false-decline count exceeds false interventions")
            if self.false_intervention_count > self.intervention_count:
                raise ValueError("false interventions exceed total interventions")
            if challenge_count > self.intervention_count:
                raise ValueError("challenge count exceeds total interventions")
            if review_count > self.intervention_count:
                raise ValueError("review-case count exceeds total interventions")
        if self.objective_kind == "fraud_recall":
            if self.input_values_digest is not None:
                raise ValueError("recall objective cannot bind a values digest")
            if self.objective_value is not None and self.objective_value > 1.0:
                raise ValueError("recall objective must be in [0, 1]")
            if self.objective_value is not None:
                fraud_intervention_count = _integral_count(
                    self.objective_value,
                    self.fraud_count,
                    label="fraud-recall count",
                )
                assert self.false_intervention_count is not None
                assert self.intervention_count is not None
                if (
                    self.false_intervention_count + fraud_intervention_count
                    != self.intervention_count
                ):
                    raise ValueError(
                        "fraud-recall count and false interventions must sum to interventions"
                    )
        elif self.input_values_digest is None:
            raise ValueError("fraud-value objective requires a values digest")
        elif self.intervention_count is not None and self.false_intervention_count is not None:
            if self.intervention_count - self.false_intervention_count > self.fraud_count:
                raise ValueError("fraud interventions exceed fraud rows")
        if self.report_digest != _report_digest(self):
            raise ValueError("threshold report digest is inconsistent")
        return self

    def to_json(self) -> bytes:
        """Return canonical threshold evidence bytes."""
        return canonical_json_bytes(self.model_dump(mode="json"))

    @classmethod
    def from_json(cls, payload: bytes) -> ThresholdReport:
        """Load canonical JSON and revalidate report semantics and its digest."""
        try:
            document = strict_json_loads(payload)
            if type(document) is not dict:
                raise ThresholdContractError("threshold JSON must contain an object")
            return cls.model_validate(document)
        except (WireContractError, ValidationError) as error:
            raise ThresholdContractError(str(error)) from error


def select_policy_thresholds(
    scores: NDArray[np.generic],
    labels: NDArray[np.generic],
    mandatory_actions: NDArray[np.object_],
    review_case_counter: Callable[[NDArray[np.object_]], int],
    budget: OperatingBudget,
    values: NDArray[np.generic] | None = None,
) -> ThresholdReport:
    """Exhaustively select thresholds without relaxing any frozen operating cap.

    ``review_case_counter`` is deliberately truth-blind: it receives only a fresh
    candidate action array. Task 10 supplies the production causal grouping adapter.
    """
    raw_score_values = _raw_scores(scores)
    score_values = _normalize_validated_scores(raw_score_values)
    label_values = _labels(labels)
    action_values = _mandatory_actions(mandatory_actions)
    value_values = _values(values) if values is not None else None
    lengths = {len(score_values), len(label_values), len(action_values)}
    if value_values is not None:
        lengths.add(len(value_values))
    if len(lengths) != 1:
        raise ThresholdContractError("scores, labels, actions, and values must have equal lengths")
    if type(budget) is not OperatingBudget:
        raise ThresholdContractError("budget must be an exact OperatingBudget")
    budget = OperatingBudget.model_validate(budget)
    if not callable(review_case_counter):
        raise ThresholdContractError("review_case_counter must be callable")
    unique_score_count = len(np.unique(raw_score_values))
    if unique_score_count > MAX_UNIQUE_OPERATING_SCORES:
        raise ThresholdContractError(
            "unique operating scores exceed the frozen maximum of "
            f"{MAX_UNIQUE_OPERATING_SCORES}"
        )

    row_count = len(score_values)
    legitimate_count = int(np.sum(label_values == 0))
    fraud_count = int(np.sum(label_values == 1))
    if legitimate_count == 0:
        raise ThresholdContractError("threshold selection requires legitimate rows")
    if fraud_count == 0:
        raise ThresholdContractError("threshold selection requires fraud rows")

    candidates = tuple(sorted({0.0, 1.0, *(float(value) for value in score_values)}))
    candidate_threshold_count = len(candidates)
    candidate_count = candidate_threshold_count * (candidate_threshold_count + 1) // 2
    (
        nonmandatory_ge,
        legitimate_nonmandatory_ge,
        fraud_nonmandatory_ge,
        mandatory_legitimate_count,
        mandatory_fraud_count,
    ) = _cumulative_statistics(
        score_values, label_values, action_values, candidates
    )
    false_decline_rates = tuple(
        (mandatory_legitimate_count + legitimate_nonmandatory_ge[index])
        / legitimate_count
        for index in range(candidate_threshold_count)
    )
    first_false_decline_feasible = next(
        (
            index
            for index, rate in enumerate(false_decline_rates)
            if rate <= budget.false_decline_rate_max
        ),
        candidate_threshold_count,
    )
    feasible_count = 0
    minimum_false_decline = false_decline_rates[-1]
    minimum_challenge = 0.0
    minimum_review = 1.0
    selected: tuple[
        tuple[float, int, float, float],
        float,
        float,
        float,
        float,
        float,
        int,
        int,
        int,
    ] | None = None
    review_cache: dict[bytes, int] = {}

    for challenge_index, challenge in enumerate(candidates):
        review_cases = _review_cases_for_challenge(
            review_case_counter,
            score_values,
            action_values,
            challenge,
            review_cache,
        )
        review_rate = review_cases / row_count
        minimum_review = min(minimum_review, review_rate)
        if review_rate > budget.review_case_rate_max:
            continue
        decline_start = max(challenge_index, first_false_decline_feasible)
        decline_end = _last_challenge_feasible_decline(
            nonmandatory_ge,
            challenge_index,
            row_count,
            budget.challenge_rate_max,
        )
        if decline_start > decline_end:
            continue
        feasible_count += decline_end - decline_start + 1
        decline = candidates[decline_end]
        false_decline_rate = false_decline_rates[decline_end]
        challenge_rate = (
            nonmandatory_ge[challenge_index] - nonmandatory_ge[decline_end]
        ) / row_count
        false_interventions = (
            mandatory_legitimate_count
            + legitimate_nonmandatory_ge[challenge_index]
        )
        intervention_count = (
            false_interventions
            + mandatory_fraud_count
            + fraud_nonmandatory_ge[challenge_index]
        )
        objective = (
            _captured_fraud_value(
                score_values,
                label_values,
                action_values,
                value_values,
                challenge,
            )
            if value_values is not None
            else (
                mandatory_fraud_count + fraud_nonmandatory_ge[challenge_index]
            )
            / fraud_count
        )
        objective = float(objective)
        if not math.isfinite(objective):
            raise ThresholdContractError("threshold objective must remain finite")
        ranking = (objective, -false_interventions, decline, challenge)
        candidate = (
            ranking,
            challenge,
            decline,
            false_decline_rate,
            challenge_rate,
            review_rate,
            false_interventions,
            intervention_count,
            review_cases,
        )
        if selected is None or ranking > selected[0]:
            selected = candidate

    base_document: dict[str, object] = {
        "schema_version": "1.0.0",
        "budget": budget.model_dump(mode="json"),
        "objective_kind": (
            "fraud_value_captured" if value_values is not None else "fraud_recall"
        ),
        "candidate_count": candidate_count,
        "candidate_threshold_count": candidate_threshold_count,
        "feasible_candidate_count": feasible_count,
        "row_count": row_count,
        "legitimate_count": legitimate_count,
        "fraud_count": fraud_count,
        "minimum_false_decline_rate": minimum_false_decline,
        "minimum_challenge_rate": minimum_challenge,
        "minimum_review_case_rate": minimum_review,
        "input_scores_digest": _array_digest(scores),
        "normalized_scores_digest": _array_digest(score_values),
        "input_labels_digest": _array_digest(labels),
        "input_mandatory_actions_digest": _actions_digest(action_values),
        "input_values_digest": _array_digest(values) if values is not None else None,
    }
    if selected is None:
        base_document.update(
            {
                "feasible": False,
                "thresholds": None,
                "objective_value": None,
                "calibration_false_decline_rate": None,
                "calibration_challenge_rate": None,
                "calibration_review_case_rate": None,
                "false_intervention_count": None,
                "intervention_count": None,
                "review_case_count": None,
                "reason": "no_candidate_satisfies_operating_budget",
                "selected_actions_digest": None,
            }
        )
    else:
        (
            ranking,
            challenge,
            decline,
            false_decline_rate,
            challenge_rate,
            review_rate,
            false_interventions,
            intervention_count,
            review_cases,
        ) = selected
        thresholds = PolicyThresholds(challenge=challenge, decline=decline)
        selected_actions = _apply_actions(
            score_values, action_values, challenge, decline
        )
        base_document.update(
            {
                "feasible": True,
                "thresholds": thresholds.model_dump(mode="json"),
                "objective_value": ranking[0],
                "calibration_false_decline_rate": false_decline_rate,
                "calibration_challenge_rate": challenge_rate,
                "calibration_review_case_rate": review_rate,
                "false_intervention_count": false_interventions,
                "intervention_count": intervention_count,
                "review_case_count": review_cases,
                "reason": "selected",
                "selected_actions_digest": _actions_digest(selected_actions),
            }
        )
    base_document["report_digest"] = _digest(canonical_json_bytes(base_document))
    return ThresholdReport.model_validate(base_document)


def normalize_operating_scores(scores: NDArray[np.generic]) -> NDArray[np.float64]:
    """Map finite raw arm scores from [0, 1] into the frozen interior score band.

    Task 12 must call this same function for every arm before action-policy replay,
    preserving exact ``1.0`` as the disabled threshold sentinel.
    """
    return _normalize_validated_scores(_raw_scores(scores))


def _raw_scores(values: object) -> NDArray[np.float64]:
    if type(values) is not np.ndarray:
        raise ThresholdContractError("scores must be an exact numpy array")
    array = cast(NDArray[np.generic], values)
    if array.ndim != 1 or array.size == 0:
        raise ThresholdContractError("scores must be a nonempty one-dimensional array")
    if np.issubdtype(array.dtype, np.bool_) or not _is_real_numeric_dtype(array.dtype):
        raise ThresholdContractError(
            "scores must have a non-boolean real integer or floating dtype"
        )
    result = np.asarray(array, dtype=np.float64).copy()
    if not np.isfinite(result).all():
        raise ThresholdContractError("scores must be finite")
    if np.any((result < 0.0) | (result > 1.0)):
        raise ThresholdContractError("raw operating scores must be in [0, 1]")
    return result


def _normalize_validated_scores(scores: NDArray[np.float64]) -> NDArray[np.float64]:
    return np.clip(scores, _SCORE_MIN, _SCORE_MAX)


def _labels(values: object) -> NDArray[np.int64]:
    if type(values) is not np.ndarray:
        raise ThresholdContractError("labels must be an exact numpy array")
    array = cast(NDArray[np.generic], values)
    if array.ndim != 1 or array.size == 0:
        raise ThresholdContractError("labels must be a nonempty one-dimensional array")
    if not (
        _is_real_numeric_dtype(array.dtype) or np.issubdtype(array.dtype, np.bool_)
    ):
        raise ThresholdContractError(
            "labels must have a real integer, floating, or boolean dtype"
        )
    numeric = np.asarray(array, dtype=np.float64)
    if not np.isfinite(numeric).all() or not np.all((numeric == 0.0) | (numeric == 1.0)):
        raise ThresholdContractError("labels must contain only binary values")
    return numeric.astype(np.int64, copy=True)


def _mandatory_actions(values: object) -> NDArray[np.object_]:
    if type(values) is not np.ndarray:
        raise ThresholdContractError("mandatory_actions must be an exact numpy array")
    array = cast(NDArray[np.object_], values)
    if array.ndim != 1 or array.size == 0:
        raise ThresholdContractError(
            "mandatory_actions must be a nonempty one-dimensional array"
        )
    result = np.empty(array.size, dtype=object)
    for index, action in enumerate(array):
        if type(action) is not Action or action not in {Action.APPROVE, Action.DECLINE}:
            raise ThresholdContractError(
                "mandatory_actions must contain only exact APPROVE or DECLINE Action values"
            )
        result[index] = action
    return result


def _values(values: object) -> NDArray[np.float64]:
    if type(values) is not np.ndarray:
        raise ThresholdContractError("values must be an exact numpy array")
    array = cast(NDArray[np.generic], values)
    if array.ndim != 1 or array.size == 0:
        raise ThresholdContractError("values must be a nonempty one-dimensional array")
    if np.issubdtype(array.dtype, np.bool_) or not _is_real_numeric_dtype(array.dtype):
        raise ThresholdContractError(
            "values must have a non-boolean real integer or floating dtype"
        )
    result = np.asarray(array, dtype=np.float64).copy()
    if not np.isfinite(result).all() or np.any(result < 0.0):
        raise ThresholdContractError("values must be finite and nonnegative")
    return result


def _cumulative_statistics(
    scores: NDArray[np.float64],
    labels: NDArray[np.int64],
    mandatory: NDArray[np.object_],
    candidates: tuple[float, ...],
) -> tuple[
    tuple[int, ...],
    tuple[int, ...],
    tuple[int, ...],
    int,
    int,
]:
    group_nonmandatory: dict[float, int] = {}
    group_legitimate: dict[float, int] = {}
    group_fraud: dict[float, int] = {}
    mandatory_legitimate = 0
    mandatory_fraud = 0
    for score, label, action in zip(scores, labels, mandatory, strict=True):
        numeric_score = float(score)
        if action is Action.DECLINE:
            if label == 0:
                mandatory_legitimate += 1
            else:
                mandatory_fraud += 1
            continue
        group_nonmandatory[numeric_score] = group_nonmandatory.get(numeric_score, 0) + 1
        if label == 0:
            group_legitimate[numeric_score] = group_legitimate.get(numeric_score, 0) + 1
        else:
            group_fraud[numeric_score] = group_fraud.get(numeric_score, 0) + 1

    nonmandatory_ge = [0] * len(candidates)
    legitimate_ge = [0] * len(candidates)
    fraud_ge = [0] * len(candidates)
    running_nonmandatory = 0
    running_legitimate = 0
    running_fraud = 0
    for index in range(len(candidates) - 1, -1, -1):
        threshold = candidates[index]
        running_nonmandatory += group_nonmandatory.get(threshold, 0)
        running_legitimate += group_legitimate.get(threshold, 0)
        running_fraud += group_fraud.get(threshold, 0)
        nonmandatory_ge[index] = running_nonmandatory
        legitimate_ge[index] = running_legitimate
        fraud_ge[index] = running_fraud
    return (
        tuple(nonmandatory_ge),
        tuple(legitimate_ge),
        tuple(fraud_ge),
        mandatory_legitimate,
        mandatory_fraud,
    )


def _captured_fraud_value(
    scores: NDArray[np.float64],
    labels: NDArray[np.int64],
    mandatory: NDArray[np.object_],
    values: NDArray[np.float64],
    challenge: float,
) -> float:
    try:
        total = math.fsum(
            float(value)
            for score, label, action, value in zip(
                scores, labels, mandatory, values, strict=True
            )
            if label == 1 and (action is Action.DECLINE or score >= challenge)
        )
    except OverflowError as error:
        raise ThresholdContractError(
            "captured fraud value aggregate must remain finite"
        ) from error
    if not math.isfinite(total):
        raise ThresholdContractError("captured fraud value aggregate must remain finite")
    return total


def _last_challenge_feasible_decline(
    nonmandatory_ge: tuple[int, ...],
    challenge_index: int,
    row_count: int,
    challenge_rate_max: float,
) -> int:
    low = challenge_index
    high = len(nonmandatory_ge) - 1
    result = challenge_index
    while low <= high:
        middle = (low + high) // 2
        challenge_rate = (
            nonmandatory_ge[challenge_index] - nonmandatory_ge[middle]
        ) / row_count
        if challenge_rate <= challenge_rate_max:
            result = middle
            low = middle + 1
        else:
            high = middle - 1
    return result


def _apply_actions(
    scores: NDArray[np.float64],
    mandatory: NDArray[np.object_],
    challenge: float,
    decline: float,
) -> NDArray[np.object_]:
    result = np.empty(len(scores), dtype=object)
    for index, score in enumerate(scores):
        result[index] = Action.APPROVE
        if mandatory[index] is Action.DECLINE or score >= decline:
            result[index] = Action.DECLINE
        elif score >= challenge:
            result[index] = Action.CHALLENGE
    return result


def _review_cases_for_challenge(
    callback: Callable[[NDArray[np.object_]], int],
    scores: NDArray[np.float64],
    mandatory: NDArray[np.object_],
    challenge: float,
    cache: dict[bytes, int],
) -> int:
    all_declined = _apply_actions(scores, mandatory, challenge, challenge)
    intervention_mask = ~_is_action(all_declined, Action.APPROVE)
    cache_key = intervention_mask.tobytes()
    if cache_key in cache:
        return cache[cache_key]
    first = _call_review_case_counter(callback, all_declined)
    repeated = _call_review_case_counter(callback, all_declined)
    if first != repeated:
        raise ThresholdContractError("review_case_counter must be deterministic")
    challenged = _apply_actions(scores, mandatory, challenge, 1.0)
    if not np.array_equal(
        intervention_mask, ~_is_action(challenged, Action.APPROVE)
    ):
        raise AssertionError("severity variants must preserve the intervention mask")
    changed_severity = _call_review_case_counter(callback, challenged)
    if first != changed_severity:
        raise ThresholdContractError(
            "review_case_counter must be intervention-mask invariant when "
            "CHALLENGE and DECLINE severities change"
        )
    cache[cache_key] = first
    return first


def _call_review_case_counter(
    callback: Callable[[NDArray[np.object_]], int], actions: NDArray[np.object_]
) -> int:
    interventions = int(np.sum(~_is_action(actions, Action.APPROVE)))
    try:
        result = callback(actions.copy())
    except Exception as error:
        raise ThresholdContractError("review_case_counter raised an exception") from error
    if type(result) is not int or result < 0 or result > interventions:
        raise ThresholdContractError(
            "review_case_counter must return an exact nonnegative int no greater "
            "than interventions"
        )
    return result


def _is_action(actions: NDArray[np.object_], action: Action) -> NDArray[np.bool_]:
    return np.fromiter(
        (value is action for value in actions),
        dtype=np.bool_,
        count=len(actions),
    )


def _is_real_numeric_dtype(dtype: np.dtype[np.generic]) -> bool:
    return bool(
        np.issubdtype(dtype, np.integer) or np.issubdtype(dtype, np.floating)
    )


def _integral_count(rate: float, denominator: int, *, label: str) -> int:
    """Accept only the canonical float emitted by exact integer division."""
    scaled = rate * denominator
    nearest = round(scaled)
    if nearest < 0 or nearest > denominator:
        raise ValueError(f"{label} reconstructs a count outside its denominator")
    canonical_rate = nearest / denominator
    if rate != canonical_rate:
        raise ValueError(f"{label} must reconstruct an integer count")
    return nearest


def _array_digest(array: NDArray[np.generic]) -> str:
    contiguous = np.ascontiguousarray(array)
    document = {
        "dtype": contiguous.dtype.str,
        "shape": list(contiguous.shape),
        "bytes_hex": contiguous.tobytes(order="C").hex(),
    }
    return _digest(canonical_json_bytes(document))


def _actions_digest(actions: NDArray[np.object_]) -> str:
    return _digest(canonical_json_bytes([cast(Action, action).value for action in actions]))


def _report_digest(report: ThresholdReport) -> str:
    document = report.model_dump(mode="json", exclude={"report_digest"})
    return _digest(canonical_json_bytes(document))


def _digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()
