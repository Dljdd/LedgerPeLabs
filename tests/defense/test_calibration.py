"""Chronological, portable calibration contract tests."""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest
from pydantic import ValidationError

from apar.defense.calibration import (
    CalibrationArtifact,
    CalibrationContractError,
    CalibrationKind,
    ProbabilityCalibrator,
    select_calibrator,
)
from apar.runs.wire import canonical_json_bytes


def _fit_window(repeats: int = 30) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.array([0.05, 0.2, 0.7, 0.9] * repeats, dtype=np.float64),
        np.array([0, 0, 1, 1] * repeats, dtype=np.int8),
    )


def _selection_window(repeats: int = 30) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.array([0.1, 0.3, 0.6, 0.8] * repeats, dtype=np.float64),
        np.array([0, 0, 1, 1] * repeats, dtype=np.int8),
    )


def _calibrator(*, min_class_count: int = 50) -> ProbabilityCalibrator:
    fit_scores, fit_labels = _fit_window()
    selection_scores, selection_labels = _selection_window()
    return select_calibrator(
        fit_scores,
        fit_labels,
        selection_scores,
        selection_labels,
        min_class_count=min_class_count,
    )


def test_calibrator_roundtrip_is_canonical_portable_and_monotone() -> None:
    calibrator = _calibrator()
    payload = calibrator.to_json()
    restored = ProbabilityCalibrator.from_json(payload)

    assert payload == canonical_json_bytes(json.loads(payload))
    grid = np.linspace(0.0, 1.0, 101, dtype=np.float64)
    original = calibrator.predict(grid)
    recovered = restored.predict(grid)
    np.testing.assert_allclose(recovered, original, rtol=0.0, atol=0.0)
    assert np.isfinite(recovered).all()
    assert np.all(np.diff(recovered) >= 0.0)
    assert np.all(recovered >= 1e-8)
    assert np.all(recovered <= 1.0 - 1e-8)
    assert calibrator.artifact.sklearn_version
    assert calibrator.artifact.numpy_version
    assert calibrator.artifact.sigmoid_c == 1e6
    assert calibrator.artifact.sigmoid_solver == "lbfgs"
    assert calibrator.artifact.sigmoid_random_state == 260816
    assert calibrator.artifact.clip_epsilon == 1e-8


def test_isotonic_kind_has_only_isotonic_numeric_state() -> None:
    fit_scores = np.repeat(np.array([0.1, 0.2, 0.8, 0.9]), 60)
    fit_labels = np.repeat(np.array([0, 0, 1, 1], dtype=np.int8), 60)
    selection_scores = np.repeat(np.array([0.1, 0.2, 0.8, 0.9]), 30)
    selection_labels = np.repeat(np.array([0, 0, 1, 1], dtype=np.int8), 30)
    calibrator = select_calibrator(
        fit_scores,
        fit_labels,
        selection_scores,
        selection_labels,
    )

    assert calibrator.artifact.kind is CalibrationKind.ISOTONIC
    assert calibrator.artifact.sigmoid_coefficient is None
    assert calibrator.artifact.sigmoid_intercept is None
    assert calibrator.artifact.isotonic_x
    assert len(calibrator.artifact.isotonic_x) == len(calibrator.artifact.isotonic_y)
    assert all(
        left < right
        for left, right in zip(
            calibrator.artifact.isotonic_x,
            calibrator.artifact.isotonic_x[1:],
            strict=False,
        )
    )
    assert all(
        left <= right
        for left, right in zip(
            calibrator.artifact.isotonic_y,
            calibrator.artifact.isotonic_y[1:],
            strict=False,
        )
    )


def test_sigmoid_kind_has_only_sigmoid_numeric_state_when_isotonic_ineligible() -> None:
    calibrator = _calibrator(min_class_count=10_000)

    assert calibrator.artifact.kind is CalibrationKind.SIGMOID
    assert calibrator.artifact.sigmoid_coefficient is not None
    assert calibrator.artifact.sigmoid_intercept is not None
    assert calibrator.artifact.isotonic_x == ()
    assert calibrator.artifact.isotonic_y == ()
    assert calibrator.artifact.isotonic_candidate_brier is None
    assert calibrator.artifact.selection_brier == calibrator.artifact.sigmoid_candidate_brier


def test_exact_selection_brier_tie_chooses_sigmoid(monkeypatch: pytest.MonkeyPatch) -> None:
    import apar.defense.calibration as module

    monkeypatch.setattr(module, "_brier", lambda _labels, _predictions: 0.25)
    calibrator = _calibrator()
    assert calibrator.artifact.kind is CalibrationKind.SIGMOID


def test_fit_and_selection_window_content_digests_bind_all_bytes() -> None:
    fit_scores, fit_labels = _fit_window()
    selection_scores, selection_labels = _selection_window()
    baseline = select_calibrator(
        fit_scores,
        fit_labels,
        selection_scores,
        selection_labels,
    ).artifact

    altered_fit_scores = fit_scores.copy()
    altered_fit_scores[0] = np.nextafter(altered_fit_scores[0], 1.0)
    fit_changed = select_calibrator(
        altered_fit_scores,
        fit_labels,
        selection_scores,
        selection_labels,
    ).artifact
    altered_selection_labels = selection_labels.copy()
    altered_selection_labels[0] = 1
    selection_changed = select_calibrator(
        fit_scores,
        fit_labels,
        selection_scores,
        altered_selection_labels,
    ).artifact

    assert baseline.fit_window_content_digest != fit_changed.fit_window_content_digest
    assert (
        baseline.selection_window_content_digest
        == fit_changed.selection_window_content_digest
    )
    assert (
        baseline.selection_window_content_digest
        != selection_changed.selection_window_content_digest
    )
    assert baseline.fit_window_content_digest == selection_changed.fit_window_content_digest


def test_selection_labels_cannot_affect_fitted_numeric_states() -> None:
    fit_scores, fit_labels = _fit_window()
    selection_scores, selection_labels = _selection_window()
    flipped = 1 - selection_labels
    original = select_calibrator(
        fit_scores,
        fit_labels,
        selection_scores,
        selection_labels,
        min_class_count=10_000,
    )
    changed = select_calibrator(
        fit_scores,
        fit_labels,
        selection_scores,
        flipped,
        min_class_count=10_000,
    )

    # Selection can choose a candidate, but it cannot refit either candidate.
    assert original.artifact.sigmoid_coefficient == changed.artifact.sigmoid_coefficient
    assert original.artifact.sigmoid_intercept == changed.artifact.sigmoid_intercept
    assert original.artifact.isotonic_x == changed.artifact.isotonic_x == ()
    assert original.artifact.isotonic_y == changed.artifact.isotonic_y == ()


def test_fit_label_perturbation_changes_fitted_state_but_not_selection_digest() -> None:
    fit_scores, fit_labels = _fit_window()
    selection_scores, selection_labels = _selection_window()
    changed_labels = fit_labels.copy()
    changed_labels[0], changed_labels[2] = changed_labels[2], changed_labels[0]
    original = select_calibrator(
        fit_scores,
        fit_labels,
        selection_scores,
        selection_labels,
    ).artifact
    changed = select_calibrator(
        fit_scores,
        changed_labels,
        selection_scores,
        selection_labels,
    ).artifact

    assert original.sigmoid_coefficient != changed.sigmoid_coefficient
    assert original.selection_window_content_digest == changed.selection_window_content_digest


@pytest.mark.parametrize(
    ("argument", "value"),
    [
        ("fit_scores", np.array([], dtype=np.float64)),
        ("fit_scores", np.array([[0.1, 0.9]], dtype=np.float64)),
        ("fit_scores", np.array([0.1, np.nan], dtype=np.float64)),
        ("fit_scores", np.array([0.1, np.inf], dtype=np.float64)),
        ("fit_scores", np.array([-1e-12, 0.9], dtype=np.float64)),
        ("fit_scores", np.array([0.1, 1.0 + 1e-12], dtype=np.float64)),
        ("fit_scores", np.array([True, False], dtype=np.bool_)),
        ("fit_scores", [0.1, 0.9]),
        ("fit_labels", np.array([0, 2], dtype=np.int8)),
        ("fit_labels", np.array([0.0, 1.5], dtype=np.float64)),
        ("fit_labels", np.array([[0, 1]], dtype=np.int8)),
        ("selection_scores", np.array([], dtype=np.float64)),
        ("selection_labels", np.array([0, 2], dtype=np.int8)),
    ],
)
def test_calibration_rejects_invalid_exact_arrays(argument: str, value: object) -> None:
    fit_scores = np.array([0.1, 0.9], dtype=np.float64)
    fit_labels = np.array([0, 1], dtype=np.int8)
    selection_scores = np.array([0.2, 0.8], dtype=np.float64)
    selection_labels = np.array([0, 1], dtype=np.int8)
    arguments: dict[str, object] = {
        "fit_scores": fit_scores,
        "fit_labels": fit_labels,
        "selection_scores": selection_scores,
        "selection_labels": selection_labels,
    }
    arguments[argument] = value

    with pytest.raises(CalibrationContractError):
        select_calibrator(**arguments)  # type: ignore[arg-type]


def test_calibration_rejects_misalignment_one_class_fit_and_bad_minimum() -> None:
    scores = np.array([0.1, 0.9], dtype=np.float64)
    labels = np.array([0, 1], dtype=np.int8)
    with pytest.raises(CalibrationContractError, match="equal lengths"):
        select_calibrator(scores, np.array([0], dtype=np.int8), scores, labels)
    with pytest.raises(CalibrationContractError, match="both classes"):
        select_calibrator(scores, np.array([1, 1], dtype=np.int8), scores, labels)
    for invalid in (True, 0, -1, 1.5):
        with pytest.raises(CalibrationContractError, match="min_class_count"):
            select_calibrator(scores, labels, scores, labels, min_class_count=invalid)  # type: ignore[arg-type]


def test_score_endpoints_are_only_clipped_at_documented_epsilon_without_mutation() -> None:
    fit_scores = np.array([0.0, 1.0, 0.25, 0.75], dtype=np.float64)
    fit_labels = np.array([0, 1, 0, 1], dtype=np.int8)
    selection_scores = fit_scores.copy()
    selection_labels = fit_labels.copy()
    fit_before = fit_scores.copy()
    selection_before = selection_scores.copy()
    calibrator = select_calibrator(
        fit_scores,
        fit_labels,
        selection_scores,
        selection_labels,
        min_class_count=10,
    )
    predict_scores = np.array([0.0, 1.0], dtype=np.float64)
    predict_before = predict_scores.copy()

    output = calibrator.predict(predict_scores)

    np.testing.assert_array_equal(fit_scores, fit_before)
    np.testing.assert_array_equal(selection_scores, selection_before)
    np.testing.assert_array_equal(predict_scores, predict_before)
    assert np.all(output >= 1e-8)
    assert np.all(output <= 1.0 - 1e-8)


def test_inverse_sigmoid_fit_is_rejected_instead_of_publishing_decreasing_probabilities() -> None:
    scores = np.array([0.1, 0.2, 0.8, 0.9], dtype=np.float64)
    inverse_labels = np.array([1, 1, 0, 0], dtype=np.int8)
    with pytest.raises(CalibrationContractError, match="nondecreasing"):
        select_calibrator(
            scores,
            inverse_labels,
            scores.copy(),
            inverse_labels.copy(),
            min_class_count=10_000,
        )


def test_serialization_rejects_noncanonical_extra_or_tampered_artifacts() -> None:
    calibrator = _calibrator(min_class_count=10_000)
    document = json.loads(calibrator.to_json())
    coefficient = document["artifact"]["sigmoid_coefficient"]
    assert isinstance(coefficient, float)
    document["artifact"]["sigmoid_coefficient"] = coefficient + 0.01
    tampered = canonical_json_bytes(document)
    with pytest.raises((CalibrationContractError, ValidationError), match="digest"):
        ProbabilityCalibrator.from_json(tampered)

    extra = json.loads(calibrator.to_json())
    extra["unexpected"] = True
    with pytest.raises((CalibrationContractError, ValidationError)):
        ProbabilityCalibrator.from_json(canonical_json_bytes(extra))

    pretty = json.dumps(json.loads(calibrator.to_json()), indent=2).encode()
    with pytest.raises(CalibrationContractError, match="canonical"):
        ProbabilityCalibrator.from_json(pretty)


def test_artifact_contract_rejects_wrong_kind_specific_fields_and_digest_shape() -> None:
    artifact = _calibrator(min_class_count=10_000).artifact
    document = artifact.model_dump(mode="json")
    document["isotonic_x"] = [0.1]
    document["isotonic_y"] = [0.1]
    with pytest.raises(ValidationError, match="sigmoid artifact"):
        CalibrationArtifact.model_validate(document)

    document = artifact.model_dump(mode="json")
    document["fit_window_content_digest"] = "bad"
    with pytest.raises(ValidationError, match="SHA-256"):
        CalibrationArtifact.model_validate(document)


def test_artifact_digest_is_sha256_and_stable() -> None:
    first = _calibrator(min_class_count=10_000)
    second = _calibrator(min_class_count=10_000)
    assert first.to_json() == second.to_json()
    assert len(first.artifact.artifact_digest) == 64
    int(first.artifact.artifact_digest, 16)
    assert hashlib.sha256(first.to_json()).hexdigest()
