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
    review_case_count: int | None = Field(default=None, ge=0)
    candidate_count: int = Field(ge=1)
    feasible_candidate_count: int = Field(ge=0)
    row_count: int = Field(ge=1)
    legitimate_count: int = Field(ge=1)
    fraud_count: int = Field(ge=1)
    minimum_false_decline_rate: float = Field(ge=0.0, le=1.0)
    minimum_challenge_rate: float = Field(ge=0.0, le=1.0)
    minimum_review_case_rate: float = Field(ge=0.0, le=1.0)
    reason: Literal["selected", "no_candidate_satisfies_operating_budget"]
    input_scores_digest: str
    input_labels_digest: str
    input_mandatory_actions_digest: str
    input_values_digest: str | None = None
    selected_actions_digest: str | None = None
    report_digest: str

    @field_validator(
        "input_scores_digest",
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
        else:
            if self.thresholds is not None or any(value is not None for value in realized):
                raise ValueError("infeasible report cannot claim thresholds or realized metrics")
            if self.selected_actions_digest is not None:
                raise ValueError("infeasible report cannot bind selected actions")
            if self.feasible_candidate_count != 0:
                raise ValueError("infeasible report cannot count feasible candidates")
            if self.reason != "no_candidate_satisfies_operating_budget":
                raise ValueError("infeasible report must declare the budget failure")
        if self.feasible_candidate_count > self.candidate_count:
            raise ValueError("feasible candidate count exceeds exhaustive candidate count")
        if self.legitimate_count + self.fraud_count != self.row_count:
            raise ValueError("threshold report class counts must sum to row count")
        if (
            self.false_intervention_count is not None
            and self.false_intervention_count > self.legitimate_count
        ):
            raise ValueError("false intervention count exceeds legitimate rows")
        if self.review_case_count is not None and self.review_case_count > self.row_count:
            raise ValueError("review case count exceeds decision rows")
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
    score_values = _scores(scores)
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

    row_count = len(score_values)
    legitimate = label_values == 0
    fraud = label_values == 1
    legitimate_count = int(legitimate.sum())
    fraud_count = int(fraud.sum())
    if legitimate_count == 0:
        raise ThresholdContractError("threshold selection requires legitimate rows")
    if fraud_count == 0:
        raise ThresholdContractError("threshold selection requires fraud rows")

    candidates = sorted({0.0, 1.0, *(float(value) for value in score_values)})
    candidate_count = len(candidates) * (len(candidates) + 1) // 2
    feasible_count = 0
    minimum_false_decline = 1.0
    minimum_challenge = 1.0
    minimum_review = 1.0
    selected: tuple[
        tuple[float, int, float, float],
        PolicyThresholds,
        NDArray[np.object_],
        float,
        float,
        float,
        int,
        int,
    ] | None = None

    for decline in candidates:
        for challenge in candidates:
            if challenge > decline:
                continue
            actions = _apply_actions(score_values, action_values, challenge, decline)
            review_cases = _review_cases(review_case_counter, actions)
            false_declines = int(np.sum(legitimate & _is_action(actions, Action.DECLINE)))
            challenges = int(np.sum(_is_action(actions, Action.CHALLENGE)))
            false_interventions = int(
                np.sum(legitimate & ~_is_action(actions, Action.APPROVE))
            )
            false_decline_rate = false_declines / legitimate_count
            challenge_rate = challenges / row_count
            review_rate = review_cases / row_count
            minimum_false_decline = min(minimum_false_decline, false_decline_rate)
            minimum_challenge = min(minimum_challenge, challenge_rate)
            minimum_review = min(minimum_review, review_rate)
            if (
                false_decline_rate > budget.false_decline_rate_max
                or challenge_rate > budget.challenge_rate_max
                or review_rate > budget.review_case_rate_max
            ):
                continue
            feasible_count += 1
            intervened_fraud = fraud & ~_is_action(actions, Action.APPROVE)
            objective = (
                float(value_values[intervened_fraud].sum())
                if value_values is not None
                else float(np.sum(intervened_fraud) / fraud_count)
            )
            if not math.isfinite(objective):
                raise ThresholdContractError("threshold objective must remain finite")
            ranking = (objective, -false_interventions, decline, challenge)
            thresholds = PolicyThresholds(challenge=challenge, decline=decline)
            candidate = (
                ranking,
                thresholds,
                actions,
                false_decline_rate,
                challenge_rate,
                review_rate,
                false_interventions,
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
        "feasible_candidate_count": feasible_count,
        "row_count": row_count,
        "legitimate_count": legitimate_count,
        "fraud_count": fraud_count,
        "minimum_false_decline_rate": minimum_false_decline,
        "minimum_challenge_rate": minimum_challenge,
        "minimum_review_case_rate": minimum_review,
        "input_scores_digest": _array_digest(scores),
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
                "review_case_count": None,
                "reason": "no_candidate_satisfies_operating_budget",
                "selected_actions_digest": None,
            }
        )
    else:
        (
            ranking,
            thresholds,
            selected_actions,
            false_decline_rate,
            challenge_rate,
            review_rate,
            false_interventions,
            review_cases,
        ) = selected
        base_document.update(
            {
                "feasible": True,
                "thresholds": thresholds.model_dump(mode="json"),
                "objective_value": ranking[0],
                "calibration_false_decline_rate": false_decline_rate,
                "calibration_challenge_rate": challenge_rate,
                "calibration_review_case_rate": review_rate,
                "false_intervention_count": false_interventions,
                "review_case_count": review_cases,
                "reason": "selected",
                "selected_actions_digest": _actions_digest(selected_actions),
            }
        )
    base_document["report_digest"] = _digest(canonical_json_bytes(base_document))
    return ThresholdReport.model_validate(base_document)


def _scores(values: object) -> NDArray[np.float64]:
    if type(values) is not np.ndarray:
        raise ThresholdContractError("scores must be an exact numpy array")
    array = cast(NDArray[np.generic], values)
    if array.ndim != 1 or array.size == 0:
        raise ThresholdContractError("scores must be a nonempty one-dimensional array")
    if np.issubdtype(array.dtype, np.bool_) or not np.issubdtype(array.dtype, np.number):
        raise ThresholdContractError("scores must have a non-boolean numeric dtype")
    result = np.asarray(array, dtype=np.float64).copy()
    if not np.isfinite(result).all():
        raise ThresholdContractError("scores must be finite")
    if np.any((result < _SCORE_MIN) | (result > _SCORE_MAX)):
        raise ThresholdContractError("calibrated scores must be in [1e-8, 1 - 1e-8]")
    return result


def _labels(values: object) -> NDArray[np.int64]:
    if type(values) is not np.ndarray:
        raise ThresholdContractError("labels must be an exact numpy array")
    array = cast(NDArray[np.generic], values)
    if array.ndim != 1 or array.size == 0:
        raise ThresholdContractError("labels must be a nonempty one-dimensional array")
    if not (
        np.issubdtype(array.dtype, np.number) or np.issubdtype(array.dtype, np.bool_)
    ):
        raise ThresholdContractError("labels must have a numeric or boolean dtype")
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
    if np.issubdtype(array.dtype, np.bool_) or not np.issubdtype(array.dtype, np.number):
        raise ThresholdContractError("values must have a non-boolean numeric dtype")
    result = np.asarray(array, dtype=np.float64).copy()
    if not np.isfinite(result).all() or np.any(result < 0.0):
        raise ThresholdContractError("values must be finite and nonnegative")
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


def _review_cases(
    callback: Callable[[NDArray[np.object_]], int], actions: NDArray[np.object_]
) -> int:
    interventions = int(np.sum(~_is_action(actions, Action.APPROVE)))
    results: list[int] = []
    for _ in range(2):
        try:
            result = callback(actions.copy())
        except Exception as error:
            raise ThresholdContractError("review_case_counter raised an exception") from error
        if type(result) is not int or result < 0 or result > interventions:
            raise ThresholdContractError(
                "review_case_counter must return an exact nonnegative int no greater "
                "than interventions"
            )
        results.append(result)
    if results[0] != results[1]:
        raise ThresholdContractError("review_case_counter must be deterministic")
    return results[0]


def _is_action(actions: NDArray[np.object_], action: Action) -> NDArray[np.bool_]:
    return np.fromiter(
        (value is action for value in actions),
        dtype=np.bool_,
        count=len(actions),
    )


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
