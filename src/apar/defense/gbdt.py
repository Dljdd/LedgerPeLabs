"""Deterministic CatBoost model selection, native serialization, and scoring."""

from __future__ import annotations

import hashlib
import json
import math
import platform as platform_module
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal, cast

import catboost  # type: ignore[import-untyped]
import numpy as np
import sklearn  # type: ignore[import-untyped]
from catboost import CatBoostClassifier, Pool
from numpy.typing import NDArray
from pydantic import Field, ValidationError, field_validator, model_validator
from sklearn.metrics import average_precision_score  # type: ignore[import-untyped]

from apar.contracts._validation import ExternalContract, validate_utc_timestamp
from apar.features.builders import FeatureMatrix
from apar.features.catalog import (
    EXPECTED_FEATURE_NAMES,
    FeatureCatalogError,
    audit_feature_catalog,
)
from apar.features.state import FeatureVector, feature_catalog_digest
from apar.runs.wire import canonical_json_bytes

_SHA256_LENGTH = 64
_COMPETITION_CATALOG_PATH = (
    Path(__file__).resolve().parents[3] / "config" / "defense" / "feature-catalog.json"
)


class ModelContractError(ValueError):
    """Training or scoring would violate the frozen model boundary."""


class HyperParameters(ExternalContract):
    """One fully specified CatBoost candidate."""

    depth: int = Field(ge=1, le=16)
    learning_rate: float = Field(gt=0.0, le=1.0)
    l2_leaf_reg: float = Field(ge=0.0)
    iterations: int = Field(ge=1)

    @field_validator("learning_rate", "l2_leaf_reg")
    @classmethod
    def floats_are_finite(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("hyperparameters must be finite")
        return value


class RollingFold(ExternalContract):
    """A declared past-only fit population and its later validation rows."""

    name: str
    fit_ids: tuple[str, ...]
    validation_ids: tuple[str, ...]

    @model_validator(mode="after")
    def ids_are_closed(self) -> RollingFold:
        if not self.name:
            raise ValueError("fold name must not be blank")
        if not self.fit_ids or not self.validation_ids:
            raise ValueError("fold fit and validation IDs must not be empty")
        if len(self.fit_ids) != len(set(self.fit_ids)):
            raise ValueError("fold fit IDs must be unique")
        if len(self.validation_ids) != len(set(self.validation_ids)):
            raise ValueError("fold validation IDs must be unique")
        if set(self.fit_ids) & set(self.validation_ids):
            raise ValueError("fold fit and validation IDs must not overlap")
        if any(not row_id for row_id in (*self.fit_ids, *self.validation_ids)):
            raise ValueError("fold row IDs must not be blank")
        return self


class GbdtTrainingConfig(ExternalContract):
    """Frozen bounded-search settings; production defaults evaluate eight candidates."""

    seed: int = Field(default=260816, ge=0)
    depths: tuple[int, ...] = (4, 6)
    learning_rates: tuple[float, ...] = (0.03, 0.08)
    l2_leaf_regs: tuple[float, ...] = (3.0, 8.0)
    iterations: int = Field(default=300, ge=1)
    fpr_probability_threshold: float = 0.5

    @field_validator("fpr_probability_threshold")
    @classmethod
    def fpr_threshold_is_frozen(cls, value: float) -> float:
        if value != 0.5:
            raise ValueError("FPR probability threshold must remain frozen at 0.5")
        return value

    @model_validator(mode="after")
    def grid_is_bounded_and_finite(self) -> GbdtTrainingConfig:
        for name, values in (
            ("depths", self.depths),
            ("learning_rates", self.learning_rates),
            ("l2_leaf_regs", self.l2_leaf_regs),
        ):
            if not values:
                raise ValueError(f"{name} must not be empty")
            if len(values) != len(set(values)):
                raise ValueError(f"{name} must be unique")
        if any(type(value) is not int or not 1 <= value <= 16 for value in self.depths):
            raise ValueError("depths must contain integers in [1, 16]")
        if any(not math.isfinite(value) or not 0.0 < value <= 1.0 for value in self.learning_rates):
            raise ValueError("learning rates must be finite and in (0, 1]")
        if any(not math.isfinite(value) or value < 0.0 for value in self.l2_leaf_regs):
            raise ValueError("L2 regularization values must be finite and non-negative")
        return self


class FoldResult(ExternalContract):
    """Immutable evidence for one candidate evaluated on one rolling fold."""

    fold_name: str
    params: HyperParameters
    average_precision: float = Field(ge=0.0, le=1.0)
    legitimate_fpr: float = Field(ge=0.0, le=1.0)
    fit_count: int = Field(ge=1)
    validation_count: int = Field(ge=1)
    class_weights: tuple[float, float]
    fit_ids_digest: str
    validation_ids_digest: str

    @field_validator("average_precision", "legitimate_fpr")
    @classmethod
    def metrics_are_finite(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("fold metrics must be finite")
        return value

    @field_validator("class_weights")
    @classmethod
    def weights_are_positive_finite(cls, value: tuple[float, float]) -> tuple[float, float]:
        if any(not math.isfinite(weight) or weight <= 0.0 for weight in value):
            raise ValueError("class weights must be finite and positive")
        return value

    @field_validator("fit_ids_digest", "validation_ids_digest")
    @classmethod
    def digests_are_sha256(cls, value: str) -> str:
        return _validate_digest(value)


class TrainingReceipt(ExternalContract):
    """JSON-safe provenance sufficient to validate and replay a native model."""

    schema_version: Literal["1.0.0"] = "1.0.0"
    model_payload_digest: str
    training_contract_digest: str
    feature_order: tuple[str, ...]
    catalog_digest: str
    requested_training_row_ids_digest: str
    final_training_row_ids_digest: str
    folds_digest: str
    requested_training_count: int = Field(ge=1)
    mandatory_excluded_count: int = Field(ge=0)
    final_training_count: int = Field(ge=1)
    training_cutoff: datetime
    seed: int = Field(ge=0)
    fpr_probability_threshold: float
    selected_params: HyperParameters
    class_weights: tuple[float, float]
    fold_results: tuple[FoldResult, ...]
    global_feature_importance: tuple[float, ...]
    catboost_version: str
    scikit_learn_version: str
    numpy_version: str
    python_version: str
    platform: str

    @field_validator(
        "model_payload_digest",
        "training_contract_digest",
        "catalog_digest",
        "requested_training_row_ids_digest",
        "final_training_row_ids_digest",
        "folds_digest",
    )
    @classmethod
    def digests_are_sha256(cls, value: str) -> str:
        return _validate_digest(value)

    @field_validator("training_cutoff")
    @classmethod
    def cutoff_is_utc(cls, value: datetime) -> datetime:
        return validate_utc_timestamp(value)

    @field_validator("fpr_probability_threshold")
    @classmethod
    def fpr_threshold_is_frozen(cls, value: float) -> float:
        if value != 0.5:
            raise ValueError("FPR probability threshold must remain frozen at 0.5")
        return value

    @field_validator("class_weights")
    @classmethod
    def weights_are_positive_finite(cls, value: tuple[float, float]) -> tuple[float, float]:
        if any(not math.isfinite(weight) or weight <= 0.0 for weight in value):
            raise ValueError("class weights must be finite and positive")
        return value

    @field_validator("global_feature_importance")
    @classmethod
    def importance_is_finite(cls, value: tuple[float, ...]) -> tuple[float, ...]:
        if len(value) != len(EXPECTED_FEATURE_NAMES):
            raise ValueError("global feature importance must match the feature order")
        if any(not math.isfinite(item) or item < 0.0 for item in value):
            raise ValueError("global feature importance must be finite and non-negative")
        return value

    @model_validator(mode="after")
    def receipt_is_consistent(self) -> TrainingReceipt:
        if self.feature_order != EXPECTED_FEATURE_NAMES:
            raise ValueError("receipt feature order must match the frozen catalog")
        if (
            self.requested_training_count - self.mandatory_excluded_count
            != self.final_training_count
        ):
            raise ValueError("receipt training counts are inconsistent")
        if not self.fold_results:
            raise ValueError("receipt must contain fold results")
        if any(
            result.params.iterations != self.selected_params.iterations
            for result in self.fold_results
        ):
            raise ValueError("fold and selected iteration contracts differ")
        if any(
            not value
            for value in (
                self.catboost_version,
                self.scikit_learn_version,
                self.numpy_version,
                self.python_version,
                self.platform,
            )
        ):
            raise ValueError("environment versions must not be blank")
        if self.training_contract_digest != _receipt_training_contract_digest(self):
            raise ValueError("receipt training contract digest is inconsistent")
        return self


@dataclass(frozen=True, slots=True)
class CatBoostScorer:
    """A receipt-bound, native CatBoost scorer for the frozen numeric matrix."""

    _model: CatBoostClassifier
    _payload: bytes
    receipt: TrainingReceipt

    def to_bytes(self) -> bytes:
        """Return the exact native CatBoost payload, never Python executable state."""
        return bytes(self._payload)

    @classmethod
    def from_bytes(cls, payload: bytes, receipt: TrainingReceipt) -> CatBoostScorer:
        """Load native model bytes after receipt, environment, and semantic checks."""
        if type(payload) is not bytes:
            raise ModelContractError("model payload must be exact bytes")
        if type(receipt) is not TrainingReceipt:
            raise ModelContractError("receipt must be an exact TrainingReceipt")
        try:
            validated = TrainingReceipt.model_validate(receipt)
        except ValidationError as error:
            raise ModelContractError("model receipt is invalid") from error
        if _digest_bytes(payload) != validated.model_payload_digest:
            raise ModelContractError("model payload digest does not match the receipt")
        _validate_environment(validated)
        expected_catalog = _competition_catalog_digest()
        if validated.catalog_digest != expected_catalog:
            raise ModelContractError("receipt catalog digest does not match the frozen catalog")
        model = _load_native_model(payload)
        _validate_loaded_model(model, validated)
        return cls(model, payload, validated)

    def predict(self, matrix: FeatureMatrix) -> NDArray[np.float64]:
        """Return finite uncalibrated fraud probabilities in the closed unit interval."""
        data = _scoring_data(matrix, self.receipt)
        probabilities = np.asarray(self._model.predict_proba(_pool(data))[:, 1], dtype=np.float64)
        if probabilities.shape != (data.shape[0],):
            raise ModelContractError("CatBoost returned an invalid probability shape")
        if not np.isfinite(probabilities).all() or np.any(
            (probabilities < 0.0) | (probabilities > 1.0)
        ):
            raise ModelContractError("CatBoost returned an invalid probability")
        return probabilities

    def predict_raw(self, matrix: FeatureMatrix) -> NDArray[np.float64]:
        """Return finite raw logits before calibration."""
        data = _scoring_data(matrix, self.receipt)
        raw = np.asarray(
            self._model.predict(_pool(data), prediction_type="RawFormulaVal"),
            dtype=np.float64,
        ).reshape(-1)
        if raw.shape != (data.shape[0],) or not np.isfinite(raw).all():
            raise ModelContractError("CatBoost returned invalid raw scores")
        return raw

    def contributions(self, matrix: FeatureMatrix) -> NDArray[np.float64]:
        """Return SHAP values plus expected value after verifying logit reconstruction."""
        data = _scoring_data(matrix, self.receipt)
        contributions = np.asarray(
            self._model.get_feature_importance(_pool(data), type="ShapValues"),
            dtype=np.float64,
        )
        expected_shape = (data.shape[0], len(EXPECTED_FEATURE_NAMES) + 1)
        if contributions.shape != expected_shape or not np.isfinite(contributions).all():
            raise ModelContractError("CatBoost returned invalid SHAP contributions")
        raw = np.asarray(
            self._model.predict(_pool(data), prediction_type="RawFormulaVal"),
            dtype=np.float64,
        ).reshape(-1)
        reconstructed = contributions[:, :-1].sum(axis=1) + contributions[:, -1]
        if not np.allclose(reconstructed, raw, rtol=0.0, atol=1e-12):
            raise ModelContractError("SHAP contributions do not reconstruct raw logits")
        return contributions

    def global_feature_importance(self) -> dict[str, float]:
        """Return deterministic training importance in frozen feature order."""
        return dict(
            zip(self.receipt.feature_order, self.receipt.global_feature_importance, strict=True)
        )


def train_gbdt(
    matrix: FeatureMatrix,
    labels: Mapping[str, int | bool],
    train_ids: Sequence[str],
    folds: Sequence[RollingFold],
    config: GbdtTrainingConfig,
    *,
    training_cutoff: datetime,
    mandatory_row_ids: Sequence[str] = (),
) -> CatBoostScorer:
    """Search rolling folds and fit only permitted, label-mature synthetic training rows.

    Mandatory-gated IDs must be declared inside ``train_ids``. They are removed
    deterministically from every fold and the final fit; they never reach CatBoost.
    """
    if type(matrix) is not FeatureMatrix:
        raise ModelContractError("matrix must be an exact FeatureMatrix")
    if type(config) is not GbdtTrainingConfig:
        raise ModelContractError("config must be an exact GbdtTrainingConfig")
    config = GbdtTrainingConfig.model_validate(config)
    try:
        training_cutoff = validate_utc_timestamp(training_cutoff)
    except (TypeError, ValueError) as error:
        raise ModelContractError("training cutoff must be timezone-aware UTC") from error

    rows, _ = _validated_matrix(matrix)
    requested_ids = _exact_unique_ids(train_ids, label="training")
    if not requested_ids:
        raise ModelContractError("training IDs must not be empty")
    if any(row_id not in rows for row_id in requested_ids):
        raise ModelContractError("training IDs must resolve to matrix rows")
    if requested_ids != tuple(sorted(requested_ids, key=lambda item: _row_order(rows[item]))):
        raise ModelContractError("training IDs must be in chronological order")
    if any(rows[row_id].decision_at > training_cutoff for row_id in requested_ids):
        raise ModelContractError("training row occurs after the declared cutoff")

    clean_labels = _validate_labels(labels, requested_ids)
    mandatory_ids = _exact_unique_ids(mandatory_row_ids, label="mandatory")
    if not set(mandatory_ids) <= set(requested_ids):
        raise ModelContractError("mandatory row IDs must be a subset of training IDs")
    mandatory = set(mandatory_ids)
    final_ids = tuple(row_id for row_id in requested_ids if row_id not in mandatory)
    if not final_ids:
        raise ModelContractError("mandatory exclusions removed every training row")
    final_weights = _balanced_class_weights(clean_labels, final_ids)

    clean_folds = _validate_folds(folds, rows, clean_labels, final_ids, mandatory)
    grid = _parameter_grid(config)
    fold_results: list[FoldResult] = []
    for params in grid:
        for fold in clean_folds:
            fit_weights = _balanced_class_weights(clean_labels, fold.fit_ids)
            model = _classifier(params, fit_weights, config.seed)
            fit_data = _data_for_ids(rows, fold.fit_ids)
            fit_labels = _labels_for_ids(clean_labels, fold.fit_ids)
            model.fit(_pool(fit_data, fit_labels))
            validation_data = _data_for_ids(rows, fold.validation_ids)
            validation_labels = _labels_for_ids(clean_labels, fold.validation_ids)
            probabilities = np.asarray(
                model.predict_proba(_pool(validation_data))[:, 1], dtype=np.float64
            )
            _validate_probabilities(probabilities, len(fold.validation_ids))
            average_precision = float(average_precision_score(validation_labels, probabilities))
            legitimate_fpr = _legitimate_fpr(
                validation_labels, probabilities, config.fpr_probability_threshold
            )
            fold_results.append(
                FoldResult(
                    fold_name=fold.name,
                    params=params,
                    average_precision=average_precision,
                    legitimate_fpr=legitimate_fpr,
                    fit_count=len(fold.fit_ids),
                    validation_count=len(fold.validation_ids),
                    class_weights=fit_weights,
                    fit_ids_digest=_ids_digest(fold.fit_ids),
                    validation_ids_digest=_ids_digest(fold.validation_ids),
                )
            )

    selected = _select_hyperparameters(tuple(fold_results), grid)
    requested_digest = _ids_digest(requested_ids)
    final_digest = _ids_digest(final_ids)
    folds_digest = _folds_digest(clean_folds)
    catboost_version = catboost.__version__
    scikit_learn_version = sklearn.__version__
    numpy_version = np.__version__
    python_version = platform_module.python_version()
    platform_version = platform_module.platform()
    training_contract_digest = _training_contract_digest(
        feature_order=EXPECTED_FEATURE_NAMES,
        catalog_digest=matrix.catalog_digest,
        requested_training_row_ids_digest=requested_digest,
        final_training_row_ids_digest=final_digest,
        folds_digest=folds_digest,
        requested_training_count=len(requested_ids),
        mandatory_excluded_count=len(mandatory_ids),
        final_training_count=len(final_ids),
        training_cutoff=training_cutoff,
        seed=config.seed,
        fpr_probability_threshold=config.fpr_probability_threshold,
        selected_params=selected,
        class_weights=final_weights,
        fold_results=tuple(fold_results),
        catboost_version=catboost_version,
        scikit_learn_version=scikit_learn_version,
        numpy_version=numpy_version,
        python_version=python_version,
        platform=platform_version,
    )
    final_model = _classifier(
        selected,
        final_weights,
        config.seed,
        metadata={
            "apar_training_contract_digest": training_contract_digest,
            "model_guid": training_contract_digest,
            "train_finish_time": training_cutoff.isoformat().replace(
                "+00:00", "Z"
            ),
        },
    )
    final_data = _data_for_ids(rows, final_ids)
    final_labels = _labels_for_ids(clean_labels, final_ids)
    final_model.fit(_pool(final_data, final_labels))
    importance = tuple(
        float(value)
        for value in np.asarray(
            final_model.get_feature_importance(type="PredictionValuesChange"),
            dtype=np.float64,
        )
    )
    if len(importance) != len(EXPECTED_FEATURE_NAMES) or any(
        not math.isfinite(value) or value < 0.0 for value in importance
    ):
        raise ModelContractError("CatBoost returned invalid global feature importance")
    payload = _save_native_model(final_model)
    receipt = TrainingReceipt(
        model_payload_digest=_digest_bytes(payload),
        training_contract_digest=training_contract_digest,
        feature_order=EXPECTED_FEATURE_NAMES,
        catalog_digest=matrix.catalog_digest,
        requested_training_row_ids_digest=requested_digest,
        final_training_row_ids_digest=final_digest,
        folds_digest=folds_digest,
        requested_training_count=len(requested_ids),
        mandatory_excluded_count=len(mandatory_ids),
        final_training_count=len(final_ids),
        training_cutoff=training_cutoff,
        seed=config.seed,
        fpr_probability_threshold=config.fpr_probability_threshold,
        selected_params=selected,
        class_weights=final_weights,
        fold_results=tuple(fold_results),
        global_feature_importance=importance,
        catboost_version=catboost_version,
        scikit_learn_version=scikit_learn_version,
        numpy_version=numpy_version,
        python_version=python_version,
        platform=platform_version,
    )
    _validate_loaded_model(final_model, receipt)
    return CatBoostScorer(final_model, payload, receipt)


def _parameter_grid(config: GbdtTrainingConfig) -> tuple[HyperParameters, ...]:
    return tuple(
        HyperParameters(
            depth=depth,
            learning_rate=learning_rate,
            l2_leaf_reg=l2_leaf_reg,
            iterations=config.iterations,
        )
        for depth in config.depths
        for learning_rate in config.learning_rates
        for l2_leaf_reg in config.l2_leaf_regs
    )


def _selection_key(
    params: HyperParameters, *, mean_average_precision: float, mean_legitimate_fpr: float
) -> tuple[float, float, int, float, float, int]:
    """Higher AP, lower FPR at frozen 0.5, then lexicographic parameters."""
    return (
        -mean_average_precision,
        mean_legitimate_fpr,
        params.depth,
        params.learning_rate,
        params.l2_leaf_reg,
        params.iterations,
    )


def _select_hyperparameters(
    results: tuple[FoldResult, ...], grid: tuple[HyperParameters, ...]
) -> HyperParameters:
    candidates: list[tuple[tuple[float, float, int, float, float, int], HyperParameters]] = []
    for params in grid:
        matching = tuple(result for result in results if result.params == params)
        if not matching:
            raise ModelContractError("candidate has no rolling-fold results")
        mean_ap = sum(result.average_precision for result in matching) / len(matching)
        mean_fpr = sum(result.legitimate_fpr for result in matching) / len(matching)
        candidates.append(
            (
                _selection_key(
                    params,
                    mean_average_precision=mean_ap,
                    mean_legitimate_fpr=mean_fpr,
                ),
                params,
            )
        )
    return min(candidates, key=lambda item: item[0])[1]


def _classifier(
    params: HyperParameters,
    class_weights: tuple[float, float],
    seed: int,
    *,
    metadata: Mapping[str, str] | None = None,
) -> CatBoostClassifier:
    settings: dict[str, object] = {
        "loss_function": "Logloss",
        "iterations": params.iterations,
        "depth": params.depth,
        "learning_rate": params.learning_rate,
        "l2_leaf_reg": params.l2_leaf_reg,
        "class_weights": list(class_weights),
        "random_seed": seed,
        "task_type": "CPU",
        "thread_count": 1,
        "allow_writing_files": False,
        "bootstrap_type": "No",
        "random_strength": 0,
        "verbose": False,
    }
    if metadata is not None:
        settings["metadata"] = dict(metadata)
    return CatBoostClassifier(**settings)


def _validated_matrix(
    matrix: FeatureMatrix,
) -> tuple[dict[str, FeatureVector], NDArray[np.float64]]:
    try:
        audit_feature_catalog(matrix.catalog)
        actual_digest = feature_catalog_digest(matrix.catalog)
    except (FeatureCatalogError, TypeError, ValueError) as error:
        raise ModelContractError(f"invalid feature catalog: {error}") from error
    if matrix.catalog.names != EXPECTED_FEATURE_NAMES:
        raise ModelContractError("feature order must match the frozen 48-column catalog")
    if matrix.catalog_digest != actual_digest:
        raise ModelContractError("matrix catalog digest does not match its catalog")
    if matrix.catalog_digest != _competition_catalog_digest():
        raise ModelContractError("feature matrix does not match the frozen competition catalog")
    event_ids = tuple(event.event_id for event in matrix.events)
    if len(event_ids) != len(set(event_ids)):
        raise ModelContractError("matrix contains duplicate event IDs")
    decision_events = {event.event_id: event for event in matrix.events if event.is_decision_point}
    row_ids = tuple(row.event_id for row in matrix.rows)
    if len(row_ids) != len(set(row_ids)):
        raise ModelContractError("matrix contains duplicate feature row IDs")
    if set(row_ids) != set(decision_events):
        raise ModelContractError("matrix decision event and feature row IDs do not match")
    if not row_ids:
        raise ModelContractError("feature matrix must not be empty")
    values: list[tuple[float, ...]] = []
    rows: dict[str, FeatureVector] = {}
    for row in matrix.rows:
        if row.catalog_digest != matrix.catalog_digest:
            raise ModelContractError("feature row catalog digest mismatch")
        if tuple(row.values) != EXPECTED_FEATURE_NAMES:
            raise ModelContractError("feature order must be exact for every row")
        ordered = tuple(row.values[name] for name in EXPECTED_FEATURE_NAMES)
        if any(type(value) not in {int, float} or not math.isfinite(value) for value in ordered):
            raise ModelContractError("feature values must be finite numeric scalars")
        event = decision_events[row.event_id]
        if event.decision_at != row.decision_at:
            raise ModelContractError("feature row decision time does not match its event")
        rows[row.event_id] = row
        values.append(ordered)
    data = np.asarray(values, dtype=np.float64)
    if data.shape != (len(rows), len(EXPECTED_FEATURE_NAMES)) or not np.isfinite(data).all():
        raise ModelContractError("feature matrix has invalid numeric shape or values")
    return rows, data


def _scoring_data(matrix: FeatureMatrix, receipt: TrainingReceipt) -> NDArray[np.float64]:
    if type(matrix) is not FeatureMatrix:
        raise ModelContractError("matrix must be an exact FeatureMatrix")
    if matrix.catalog_digest != receipt.catalog_digest:
        raise ModelContractError("scoring matrix catalog digest does not match the model")
    _, data = _validated_matrix(matrix)
    return data


def _exact_unique_ids(ids: Sequence[str], *, label: str) -> tuple[str, ...]:
    values = tuple(ids)
    if any(type(value) is not str or not value for value in values):
        raise ModelContractError(f"{label} IDs must be non-empty strings")
    if len(values) != len(set(values)):
        raise ModelContractError(f"{label} IDs contain duplicates")
    return values


def _validate_labels(
    labels: Mapping[str, int | bool], requested_ids: tuple[str, ...]
) -> dict[str, int]:
    if not isinstance(labels, Mapping):
        raise ModelContractError("labels must be a row-ID mapping")
    if set(labels) != set(requested_ids):
        raise ModelContractError("label IDs must match the requested training IDs exactly")
    clean: dict[str, int] = {}
    for row_id, value in labels.items():
        if type(row_id) is not str or type(value) not in {bool, int} or int(value) not in {0, 1}:
            raise ModelContractError("labels must be exact binary integers or booleans")
        clean[row_id] = int(value)
    return clean


def _validate_folds(
    folds: Sequence[RollingFold],
    rows: dict[str, FeatureVector],
    labels: dict[str, int],
    final_ids: tuple[str, ...],
    mandatory: set[str],
) -> tuple[RollingFold, ...]:
    supplied = tuple(folds)
    if not supplied:
        raise ModelContractError("at least one rolling fold is required")
    if any(type(fold) is not RollingFold for fold in supplied):
        raise ModelContractError("folds must be exact RollingFold contracts")
    if len({fold.name for fold in supplied}) != len(supplied):
        raise ModelContractError("fold names must be unique")
    allowed = set(final_ids)
    cleaned: list[RollingFold] = []
    prior_validation_ids: set[str] = set()
    prior_validation_start: datetime | None = None
    for raw in supplied:
        fit_ids = tuple(row_id for row_id in raw.fit_ids if row_id not in mandatory)
        validation_ids = tuple(row_id for row_id in raw.validation_ids if row_id not in mandatory)
        if not set((*fit_ids, *validation_ids)) <= allowed:
            raise ModelContractError("fold rows must be permitted training IDs")
        if not fit_ids or not validation_ids:
            raise ModelContractError("mandatory filtering emptied a rolling fold")
        fold = RollingFold(name=raw.name, fit_ids=fit_ids, validation_ids=validation_ids)
        if fit_ids != tuple(sorted(fit_ids, key=lambda item: _row_order(rows[item]))):
            raise ModelContractError("fold fit IDs must be chronological")
        if validation_ids != tuple(sorted(validation_ids, key=lambda item: _row_order(rows[item]))):
            raise ModelContractError("fold validation IDs must be chronological")
        if max(rows[row_id].decision_at for row_id in fit_ids) >= min(
            rows[row_id].decision_at for row_id in validation_ids
        ):
            raise ModelContractError("fold fit rows must be strictly earlier than validation")
        if prior_validation_ids & set(validation_ids):
            raise ModelContractError("rolling-fold validation populations overlap")
        validation_start = min(rows[row_id].decision_at for row_id in validation_ids)
        if prior_validation_start is not None and validation_start <= prior_validation_start:
            raise ModelContractError("rolling folds must be ordered by validation time")
        _require_both_classes(labels, fit_ids, label=f"fold {fold.name} fit")
        _require_both_classes(labels, validation_ids, label=f"fold {fold.name} validation")
        prior_validation_ids.update(validation_ids)
        prior_validation_start = validation_start
        cleaned.append(fold)
    return tuple(cleaned)


def _require_both_classes(labels: Mapping[str, int], ids: Sequence[str], *, label: str) -> None:
    if {labels[row_id] for row_id in ids} != {0, 1}:
        raise ModelContractError(f"{label} must contain both classes")


def _balanced_class_weights(labels: Mapping[str, int], ids: Sequence[str]) -> tuple[float, float]:
    _require_both_classes(labels, ids, label="training population")
    zeros = sum(labels[row_id] == 0 for row_id in ids)
    ones = len(ids) - zeros
    total = float(len(ids))
    return (total / (2.0 * zeros), total / (2.0 * ones))


def _row_order(row: FeatureVector) -> tuple[datetime, str]:
    return row.decision_at, row.event_id


def _data_for_ids(rows: Mapping[str, FeatureVector], ids: Sequence[str]) -> NDArray[np.float64]:
    data = np.asarray(
        [[rows[row_id].values[name] for name in EXPECTED_FEATURE_NAMES] for row_id in ids],
        dtype=np.float64,
    )
    if data.shape != (len(ids), len(EXPECTED_FEATURE_NAMES)) or not np.isfinite(data).all():
        raise ModelContractError("selected feature data is invalid")
    return data


def _labels_for_ids(labels: Mapping[str, int], ids: Sequence[str]) -> NDArray[np.int64]:
    return np.asarray([labels[row_id] for row_id in ids], dtype=np.int64)


def _pool(data: NDArray[np.float64], labels: NDArray[np.int64] | None = None) -> Pool:
    return Pool(data=data, label=labels, feature_names=list(EXPECTED_FEATURE_NAMES))


def _validate_probabilities(probabilities: NDArray[np.float64], expected: int) -> None:
    if probabilities.shape != (expected,) or not np.isfinite(probabilities).all():
        raise ModelContractError("CatBoost returned invalid validation probabilities")
    if np.any((probabilities < 0.0) | (probabilities > 1.0)):
        raise ModelContractError("CatBoost returned probabilities outside [0, 1]")


def _legitimate_fpr(
    labels: NDArray[np.int64], probabilities: NDArray[np.float64], threshold: float
) -> float:
    legitimate = labels == 0
    denominator = int(np.sum(legitimate))
    if denominator == 0:
        raise ModelContractError("legitimate FPR denominator is undefined")
    false_interventions = int(np.sum((probabilities >= threshold) & legitimate))
    return false_interventions / denominator


def _save_native_model(model: CatBoostClassifier) -> bytes:
    with tempfile.TemporaryDirectory(prefix="apar-catboost-") as directory:
        path = Path(directory) / "model.cbm"
        model.save_model(str(path), format="cbm")
        payload = path.read_bytes()
    if not payload:
        raise ModelContractError("CatBoost emitted an empty native model")
    return payload


def _load_native_model(payload: bytes) -> CatBoostClassifier:
    with tempfile.TemporaryDirectory(prefix="apar-catboost-") as directory:
        path = Path(directory) / "model.cbm"
        path.write_bytes(payload)
        path.chmod(0o600)
        model = CatBoostClassifier()
        try:
            model.load_model(str(path), format="cbm")
        except catboost.CatBoostError as error:
            raise ModelContractError("native CatBoost model could not be loaded") from error
    return model


def _validate_loaded_model(model: CatBoostClassifier, receipt: TrainingReceipt) -> None:
    try:
        _validate_loaded_model_inner(model, receipt)
    except ModelContractError:
        raise
    except (catboost.CatBoostError, json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
        raise ModelContractError("native model validation failed") from error


def _validate_loaded_model_inner(model: CatBoostClassifier, receipt: TrainingReceipt) -> None:
    if tuple(model.feature_names_) != receipt.feature_order:
        raise ModelContractError("native model feature order does not match the receipt")
    if int(model.tree_count_) != receipt.selected_params.iterations:
        raise ModelContractError("native model iteration count does not match the receipt")
    metadata = model.get_metadata()
    if metadata.get("apar_training_contract_digest") != receipt.training_contract_digest:
        raise ModelContractError("native model training contract does not match the receipt")
    if metadata.get("model_guid") != receipt.training_contract_digest:
        raise ModelContractError("native model GUID does not match the training contract")
    if metadata.get("train_finish_time") != receipt.training_cutoff.isoformat().replace(
        "+00:00", "Z"
    ):
        raise ModelContractError("native model finish time does not match the training cutoff")
    params = cast(dict[str, object], model.get_all_params())
    _validate_deterministic_native_settings(params, metadata, receipt)
    if int(cast(int, params.get("depth"))) != receipt.selected_params.depth:
        raise ModelContractError("native model depth does not match the receipt")
    learning_rate = float(cast(float, params.get("learning_rate")))
    # CatBoost stores this setting as a single-precision native model parameter.
    if not math.isclose(
        learning_rate,
        receipt.selected_params.learning_rate,
        rel_tol=0.0,
        abs_tol=1e-7,
    ):
        raise ModelContractError("native model learning rate does not match the receipt")
    l2 = float(cast(float, params.get("l2_leaf_reg")))
    if not math.isclose(l2, receipt.selected_params.l2_leaf_reg, rel_tol=0.0, abs_tol=1e-12):
        raise ModelContractError("native model regularization does not match the receipt")
    raw_weights = params.get("class_weights")
    if type(raw_weights) is not list or len(raw_weights) != 2:
        raise ModelContractError("native model class weights are unavailable")
    weight_values = cast(list[object], raw_weights)
    if any(type(value) not in {int, float} for value in weight_values):
        raise ModelContractError("native model class weights are invalid")
    weights = tuple(float(cast(int | float, value)) for value in weight_values)
    if not all(
        math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-7)
        for actual, expected in zip(weights, receipt.class_weights, strict=True)
    ):
        raise ModelContractError("native model class weights do not match the receipt")
    importance = np.asarray(
        model.get_feature_importance(type="PredictionValuesChange"), dtype=np.float64
    )
    expected_importance = np.asarray(receipt.global_feature_importance, dtype=np.float64)
    if importance.shape != expected_importance.shape or not np.allclose(
        importance, expected_importance, rtol=0.0, atol=1e-12
    ):
        raise ModelContractError(
            "native model global feature importance does not match the receipt"
        )


def _validate_deterministic_native_settings(
    params: Mapping[str, object], metadata: Mapping[str, str], receipt: TrainingReceipt
) -> None:
    exact_expected: dict[str, object] = {
        "loss_function": "Logloss",
        "random_seed": receipt.seed,
        "bootstrap_type": "No",
        "task_type": "CPU",
    }
    for name, expected in exact_expected.items():
        if params.get(name) != expected:
            raise ModelContractError(f"native model deterministic setting mismatch: {name}")
    random_strength = params.get("random_strength")
    if (
        type(random_strength) not in {int, float}
        or float(cast(int | float, random_strength)) != 0.0
    ):
        raise ModelContractError("native model deterministic setting mismatch: random_strength")

    raw_native_params = metadata.get("params")
    if type(raw_native_params) is not str:
        raise ModelContractError("native model persisted settings are unavailable")
    native_document = json.loads(raw_native_params)
    if type(native_document) is not dict:
        raise ModelContractError("native model persisted settings are invalid")
    flat_params = native_document.get("flat_params")
    if type(flat_params) is not dict:
        raise ModelContractError("native model flat settings are unavailable")
    flat = cast(dict[str, object], flat_params)
    flat_expected: dict[str, object] = {
        "loss_function": "Logloss",
        "random_seed": receipt.seed,
        "bootstrap_type": "No",
        "task_type": "CPU",
        "thread_count": 1,
        "allow_writing_files": False,
        "verbose": 0,
    }
    for name, expected in flat_expected.items():
        if flat.get(name) != expected:
            raise ModelContractError(f"native model deterministic setting mismatch: {name}")
    flat_random_strength = flat.get("random_strength")
    if (
        type(flat_random_strength) not in {int, float}
        or float(cast(int | float, flat_random_strength)) != 0.0
    ):
        raise ModelContractError("native model deterministic setting mismatch: random_strength")


def _validate_environment(receipt: TrainingReceipt) -> None:
    if receipt.catboost_version != catboost.__version__:
        raise ModelContractError("CatBoost version is incompatible with the receipt")
    if receipt.scikit_learn_version != sklearn.__version__:
        raise ModelContractError("scikit-learn version is incompatible with the receipt")
    if receipt.numpy_version != np.__version__:
        raise ModelContractError("NumPy version is incompatible with the receipt")
    if receipt.python_version != platform_module.python_version():
        raise ModelContractError("Python version is incompatible with the receipt")
    if receipt.platform != platform_module.platform():
        raise ModelContractError("platform is incompatible with the receipt")


def _competition_catalog_digest() -> str:
    from apar.features.catalog import load_feature_catalog

    try:
        return feature_catalog_digest(load_feature_catalog(_COMPETITION_CATALOG_PATH))
    except (OSError, FeatureCatalogError, ValueError) as error:
        raise ModelContractError("frozen feature catalog is unavailable") from error


def _ids_digest(ids: Sequence[str]) -> str:
    return _digest_bytes(canonical_json_bytes(list(ids)))


def _folds_digest(folds: Sequence[RollingFold]) -> str:
    return _digest_bytes(canonical_json_bytes([fold.model_dump(mode="json") for fold in folds]))


def _receipt_training_contract_digest(receipt: TrainingReceipt) -> str:
    return _training_contract_digest(
        feature_order=receipt.feature_order,
        catalog_digest=receipt.catalog_digest,
        requested_training_row_ids_digest=receipt.requested_training_row_ids_digest,
        final_training_row_ids_digest=receipt.final_training_row_ids_digest,
        folds_digest=receipt.folds_digest,
        requested_training_count=receipt.requested_training_count,
        mandatory_excluded_count=receipt.mandatory_excluded_count,
        final_training_count=receipt.final_training_count,
        training_cutoff=receipt.training_cutoff,
        seed=receipt.seed,
        fpr_probability_threshold=receipt.fpr_probability_threshold,
        selected_params=receipt.selected_params,
        class_weights=receipt.class_weights,
        fold_results=receipt.fold_results,
        catboost_version=receipt.catboost_version,
        scikit_learn_version=receipt.scikit_learn_version,
        numpy_version=receipt.numpy_version,
        python_version=receipt.python_version,
        platform=receipt.platform,
    )


def _training_contract_digest(
    *,
    feature_order: tuple[str, ...],
    catalog_digest: str,
    requested_training_row_ids_digest: str,
    final_training_row_ids_digest: str,
    folds_digest: str,
    requested_training_count: int,
    mandatory_excluded_count: int,
    final_training_count: int,
    training_cutoff: datetime,
    seed: int,
    fpr_probability_threshold: float,
    selected_params: HyperParameters,
    class_weights: tuple[float, float],
    fold_results: tuple[FoldResult, ...],
    catboost_version: str,
    scikit_learn_version: str,
    numpy_version: str,
    python_version: str,
    platform: str,
) -> str:
    document = {
        "feature_order": list(feature_order),
        "catalog_digest": catalog_digest,
        "requested_training_row_ids_digest": requested_training_row_ids_digest,
        "final_training_row_ids_digest": final_training_row_ids_digest,
        "folds_digest": folds_digest,
        "requested_training_count": requested_training_count,
        "mandatory_excluded_count": mandatory_excluded_count,
        "final_training_count": final_training_count,
        "training_cutoff": training_cutoff.isoformat(),
        "seed": seed,
        "fpr_probability_threshold": fpr_probability_threshold,
        "selected_params": selected_params.model_dump(mode="json"),
        "class_weights": list(class_weights),
        "fold_results": [result.model_dump(mode="json") for result in fold_results],
        "catboost_version": catboost_version,
        "scikit_learn_version": scikit_learn_version,
        "numpy_version": numpy_version,
        "python_version": python_version,
        "platform": platform,
    }
    return _digest_bytes(canonical_json_bytes(document))


def _digest_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _validate_digest(value: str) -> str:
    if len(value) != _SHA256_LENGTH or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError("digest must be lowercase SHA-256")
    return value
