"""Chronological probability calibration with portable numeric artifacts."""

from __future__ import annotations

import hashlib
import math
from enum import StrEnum
from typing import Literal, cast

import numpy as np
import sklearn  # type: ignore[import-untyped]
from numpy.typing import NDArray
from pydantic import Field, ValidationError, field_validator, model_validator
from sklearn.isotonic import IsotonicRegression  # type: ignore[import-untyped]
from sklearn.linear_model import LogisticRegression  # type: ignore[import-untyped]

from apar.contracts._validation import ExternalContract
from apar.runs.wire import WireContractError, canonical_json_bytes, strict_json_loads

_CLIP_EPSILON = 1e-8
_SHA256_LENGTH = 64
_SIGMOID_C = 1e6
_SIGMOID_SOLVER = "lbfgs"
_SIGMOID_RANDOM_STATE = 260816


class CalibrationContractError(ValueError):
    """Calibration input or serialized state violates the closed contract."""


class CalibrationKind(StrEnum):
    SIGMOID = "sigmoid"
    ISOTONIC = "isotonic"


class CalibrationArtifact(ExternalContract):
    """JSON-safe selected calibrator and chronological window-content evidence.

    The two digests bind the complete array content supplied for each window.
    They do not claim row identity or independently prove chronological order;
    split manifests own that chronology.
    """

    schema_version: Literal["1.0.0"] = "1.0.0"
    kind: CalibrationKind
    fit_window_content_digest: str
    selection_window_content_digest: str
    selection_brier: float = Field(ge=0.0, le=1.0)
    sigmoid_candidate_brier: float = Field(ge=0.0, le=1.0)
    isotonic_candidate_brier: float | None = Field(default=None, ge=0.0, le=1.0)
    min_class_count: int = Field(ge=1)
    clip_epsilon: float = _CLIP_EPSILON
    sigmoid_c: float = _SIGMOID_C
    sigmoid_solver: Literal["lbfgs"] = "lbfgs"
    sigmoid_random_state: Literal[260816] = 260816
    isotonic_out_of_bounds: Literal["clip"] = "clip"
    sklearn_version: str
    numpy_version: str
    sigmoid_coefficient: float | None = None
    sigmoid_intercept: float | None = None
    isotonic_x: tuple[float, ...] = ()
    isotonic_y: tuple[float, ...] = ()
    artifact_digest: str

    @field_validator(
        "fit_window_content_digest",
        "selection_window_content_digest",
        "artifact_digest",
    )
    @classmethod
    def digests_are_sha256(cls, value: str) -> str:
        if len(value) != _SHA256_LENGTH:
            raise ValueError("calibration digests must be lowercase SHA-256")
        try:
            int(value, 16)
        except ValueError as error:
            raise ValueError("calibration digests must be lowercase SHA-256") from error
        if value != value.lower():
            raise ValueError("calibration digests must be lowercase SHA-256")
        return value

    @field_validator(
        "selection_brier",
        "sigmoid_candidate_brier",
        "isotonic_candidate_brier",
        "clip_epsilon",
        "sigmoid_c",
        "sigmoid_coefficient",
        "sigmoid_intercept",
    )
    @classmethod
    def scalar_state_is_finite(cls, value: float | None) -> float | None:
        if value is not None and not math.isfinite(value):
            raise ValueError("calibration numeric state must be finite")
        return value

    @model_validator(mode="after")
    def state_is_closed_and_consistent(self) -> CalibrationArtifact:
        if self.clip_epsilon != _CLIP_EPSILON:
            raise ValueError("calibration clip epsilon does not match the frozen contract")
        if self.sigmoid_c != _SIGMOID_C:
            raise ValueError("sigmoid C does not match the frozen contract")
        if not self.sklearn_version or not self.numpy_version:
            raise ValueError("calibration environment versions must not be blank")
        if self.kind is CalibrationKind.SIGMOID:
            if self.sigmoid_coefficient is None or self.sigmoid_intercept is None:
                raise ValueError("sigmoid artifact must populate sigmoid numeric state")
            if self.sigmoid_coefficient < 0.0:
                raise ValueError("sigmoid artifact must be nondecreasing")
            if self.isotonic_x or self.isotonic_y:
                raise ValueError("sigmoid artifact cannot populate isotonic numeric state")
            if self.selection_brier != self.sigmoid_candidate_brier:
                raise ValueError("selected sigmoid Brier score is inconsistent")
        else:
            if self.sigmoid_coefficient is not None or self.sigmoid_intercept is not None:
                raise ValueError("isotonic artifact cannot populate sigmoid numeric state")
            if self.isotonic_candidate_brier is None:
                raise ValueError("isotonic artifact must record its candidate Brier score")
            if self.selection_brier != self.isotonic_candidate_brier:
                raise ValueError("selected isotonic Brier score is inconsistent")
            if len(self.isotonic_x) < 2 or len(self.isotonic_x) != len(self.isotonic_y):
                raise ValueError("isotonic artifact must contain aligned threshold arrays")
        if self.isotonic_x or self.isotonic_y:
            if len(self.isotonic_x) != len(self.isotonic_y):
                raise ValueError("isotonic threshold arrays must be aligned")
            if not all(math.isfinite(value) for value in (*self.isotonic_x, *self.isotonic_y)):
                raise ValueError("isotonic threshold arrays must be finite")
            if any(
                left >= right
                for left, right in zip(self.isotonic_x, self.isotonic_x[1:], strict=False)
            ):
                raise ValueError("isotonic x thresholds must be strictly increasing")
            if any(not 0.0 <= value <= 1.0 for value in self.isotonic_x):
                raise ValueError("isotonic x thresholds must be in [0, 1]")
            if any(
                left > right
                for left, right in zip(self.isotonic_y, self.isotonic_y[1:], strict=False)
            ):
                raise ValueError("isotonic y thresholds must be nondecreasing")
            if any(not 0.0 <= value <= 1.0 for value in self.isotonic_y):
                raise ValueError("isotonic y thresholds must be in [0, 1]")
        if self.artifact_digest != _artifact_digest(self):
            raise ValueError("calibration artifact digest is inconsistent")
        return self


class ProbabilityCalibrator(ExternalContract):
    """Immutable calibrator performing inference only from stored numeric state."""

    artifact: CalibrationArtifact

    def predict(self, scores: NDArray[np.generic]) -> NDArray[np.float64]:
        """Calibrate an exact one-dimensional score array without mutating it."""
        checked = _scores(scores, label="prediction scores")
        if self.artifact.kind is CalibrationKind.SIGMOID:
            coefficient = self.artifact.sigmoid_coefficient
            intercept = self.artifact.sigmoid_intercept
            assert coefficient is not None
            assert intercept is not None
            linear = coefficient * _logit(checked) + intercept
            predictions = _expit(linear)
        else:
            predictions = np.interp(
                checked,
                np.asarray(self.artifact.isotonic_x, dtype=np.float64),
                np.asarray(self.artifact.isotonic_y, dtype=np.float64),
            )
        return np.clip(predictions, _CLIP_EPSILON, 1.0 - _CLIP_EPSILON)

    def to_json(self) -> bytes:
        """Return canonical non-executable JSON bytes."""
        return canonical_json_bytes(self.model_dump(mode="json"))

    @classmethod
    def from_json(cls, payload: bytes) -> ProbabilityCalibrator:
        """Load only canonical JSON and revalidate all numeric state and digests."""
        try:
            document = strict_json_loads(payload)
            if type(document) is not dict:
                raise CalibrationContractError("calibrator JSON must contain an object")
            return cls.model_validate(document)
        except (WireContractError, ValidationError) as error:
            raise CalibrationContractError(str(error)) from error


def select_calibrator(
    fit_scores: NDArray[np.generic],
    fit_labels: NDArray[np.generic],
    selection_scores: NDArray[np.generic],
    selection_labels: NDArray[np.generic],
    min_class_count: int = 50,
) -> ProbabilityCalibrator:
    """Fit candidates on one window and select by Brier on the separate later window."""
    if type(min_class_count) is not int or min_class_count < 1:
        raise CalibrationContractError("min_class_count must be an exact positive int")
    fit_score_values = _scores(fit_scores, label="fit scores")
    fit_label_values = _labels(fit_labels, label="fit labels")
    selection_score_values = _scores(selection_scores, label="selection scores")
    selection_label_values = _labels(selection_labels, label="selection labels")
    if len(fit_score_values) != len(fit_label_values):
        raise CalibrationContractError("fit scores and labels must have equal lengths")
    if len(selection_score_values) != len(selection_label_values):
        raise CalibrationContractError("selection scores and labels must have equal lengths")
    class_counts = np.bincount(fit_label_values, minlength=2)
    if np.any(class_counts == 0):
        raise CalibrationContractError("sigmoid calibration fit requires both classes")

    sigmoid = LogisticRegression(
        C=_SIGMOID_C,
        solver=_SIGMOID_SOLVER,
        random_state=_SIGMOID_RANDOM_STATE,
    )
    sigmoid.fit(_logit(fit_score_values).reshape(-1, 1), fit_label_values)
    coefficient = float(sigmoid.coef_[0, 0])
    intercept = float(sigmoid.intercept_[0])
    if coefficient < 0.0:
        raise CalibrationContractError(
            "sigmoid fit is decreasing; calibrated probabilities must be nondecreasing"
        )
    sigmoid_predictions = _expit(coefficient * _logit(selection_score_values) + intercept)
    sigmoid_brier = _brier(selection_label_values, sigmoid_predictions)

    isotonic_x: tuple[float, ...] = ()
    isotonic_y: tuple[float, ...] = ()
    isotonic_brier: float | None = None
    if int(class_counts.min()) >= min_class_count:
        isotonic = IsotonicRegression(out_of_bounds="clip")
        isotonic.fit(fit_score_values, fit_label_values)
        isotonic_x = tuple(float(value) for value in isotonic.X_thresholds_)
        isotonic_y = tuple(float(value) for value in isotonic.y_thresholds_)
        isotonic_predictions = np.interp(
            selection_score_values,
            np.asarray(isotonic_x, dtype=np.float64),
            np.asarray(isotonic_y, dtype=np.float64),
        )
        isotonic_brier = _brier(selection_label_values, isotonic_predictions)

    kind = (
        CalibrationKind.ISOTONIC
        if isotonic_brier is not None and isotonic_brier < sigmoid_brier
        else CalibrationKind.SIGMOID
    )
    document: dict[str, object] = {
        "schema_version": "1.0.0",
        "kind": kind.value,
        "fit_window_content_digest": _window_digest(fit_scores, fit_labels),
        "selection_window_content_digest": _window_digest(
            selection_scores, selection_labels
        ),
        "selection_brier": isotonic_brier if kind is CalibrationKind.ISOTONIC else sigmoid_brier,
        "sigmoid_candidate_brier": sigmoid_brier,
        "isotonic_candidate_brier": isotonic_brier,
        "min_class_count": min_class_count,
        "clip_epsilon": _CLIP_EPSILON,
        "sigmoid_c": _SIGMOID_C,
        "sigmoid_solver": _SIGMOID_SOLVER,
        "sigmoid_random_state": _SIGMOID_RANDOM_STATE,
        "isotonic_out_of_bounds": "clip",
        "sklearn_version": sklearn.__version__,
        "numpy_version": np.__version__,
        "sigmoid_coefficient": coefficient if kind is CalibrationKind.SIGMOID else None,
        "sigmoid_intercept": intercept if kind is CalibrationKind.SIGMOID else None,
        "isotonic_x": list(isotonic_x) if kind is CalibrationKind.ISOTONIC else [],
        "isotonic_y": list(isotonic_y) if kind is CalibrationKind.ISOTONIC else [],
    }
    document["artifact_digest"] = _digest(canonical_json_bytes(document))
    artifact = CalibrationArtifact.model_validate(document)
    return ProbabilityCalibrator(artifact=artifact)


def _scores(values: object, *, label: str) -> NDArray[np.float64]:
    if type(values) is not np.ndarray:
        raise CalibrationContractError(f"{label} must be an exact numpy array")
    array = cast(NDArray[np.generic], values)
    if array.ndim != 1 or array.size == 0:
        raise CalibrationContractError(f"{label} must be a nonempty one-dimensional array")
    if np.issubdtype(array.dtype, np.bool_) or not np.issubdtype(array.dtype, np.number):
        raise CalibrationContractError(f"{label} must have a non-boolean numeric dtype")
    result = np.asarray(array, dtype=np.float64).copy()
    if not np.isfinite(result).all():
        raise CalibrationContractError(f"{label} must be finite")
    if np.any((result < 0.0) | (result > 1.0)):
        raise CalibrationContractError(f"{label} must be in [0, 1]")
    return result


def _labels(values: object, *, label: str) -> NDArray[np.int64]:
    if type(values) is not np.ndarray:
        raise CalibrationContractError(f"{label} must be an exact numpy array")
    array = cast(NDArray[np.generic], values)
    if array.ndim != 1 or array.size == 0:
        raise CalibrationContractError(f"{label} must be a nonempty one-dimensional array")
    if not (
        np.issubdtype(array.dtype, np.number) or np.issubdtype(array.dtype, np.bool_)
    ):
        raise CalibrationContractError(f"{label} must have a numeric or boolean dtype")
    numeric = np.asarray(array, dtype=np.float64)
    if not np.isfinite(numeric).all() or not np.all((numeric == 0.0) | (numeric == 1.0)):
        raise CalibrationContractError(f"{label} must contain only binary labels")
    return numeric.astype(np.int64, copy=True)


def _logit(scores: NDArray[np.float64]) -> NDArray[np.float64]:
    clipped = np.clip(scores, _CLIP_EPSILON, 1.0 - _CLIP_EPSILON)
    return np.log(clipped / (1.0 - clipped))


def _expit(values: NDArray[np.float64]) -> NDArray[np.float64]:
    result = np.empty_like(values, dtype=np.float64)
    nonnegative = values >= 0.0
    result[nonnegative] = 1.0 / (1.0 + np.exp(-values[nonnegative]))
    exponential = np.exp(values[~nonnegative])
    result[~nonnegative] = exponential / (1.0 + exponential)
    return result


def _brier(labels: NDArray[np.int64], predictions: NDArray[np.float64]) -> float:
    return float(np.mean(np.square(predictions - labels)))


def _window_digest(scores: NDArray[np.generic], labels: NDArray[np.generic]) -> str:
    document = {
        "scores": _array_content(scores),
        "labels": _array_content(labels),
    }
    return _digest(canonical_json_bytes(document))


def _array_content(array: NDArray[np.generic]) -> dict[str, object]:
    contiguous = np.ascontiguousarray(array)
    return {
        "dtype": contiguous.dtype.str,
        "shape": list(contiguous.shape),
        "bytes_hex": contiguous.tobytes(order="C").hex(),
    }


def _artifact_digest(artifact: CalibrationArtifact) -> str:
    document = artifact.model_dump(mode="json", exclude={"artifact_digest"})
    return _digest(canonical_json_bytes(document))


def _digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()
