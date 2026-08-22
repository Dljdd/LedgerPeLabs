"""Mixed-class calibration and threshold selection regression tests."""

from __future__ import annotations

import numpy as np
import pytest

from apar.defense.sentinel import train_sentinel_defender


def _mixed_matrix(n_benign: int, n_fraud: int, seed: int):
    rng = np.random.RandomState(seed)
    benign = rng.normal(0.0, 0.5, (n_benign, 7))
    fraud = rng.normal(3.0, 0.5, (n_fraud, 7))
    x = np.vstack([benign, fraud])
    y = np.array([0] * n_benign + [1] * n_fraud)
    return x, y


class TestOneClassCalibration:
    def test_one_class_calibration_raises(self) -> None:
        x_train, y_train = _mixed_matrix(200, 50, seed=42)
        # Calibration partition has only benign rows.
        x_cal_benign_only, _ = _mixed_matrix(30, 0, seed=43)
        y_cal_all_zero = np.zeros(30, dtype=int)
        x_threshold, y_threshold = _mixed_matrix(20, 5, seed=44)
        with pytest.raises(ValueError, match="one-class calibration"):
            train_sentinel_defender(
                x_train=x_train,
                y_train=y_train,
                x_calibration=x_cal_benign_only,
                y_calibration=y_cal_all_zero,
                x_threshold=x_threshold,
                y_threshold=y_threshold,
                catboost_seeds=(1001, 1002, 1003),
                bootstrap_seed=707,
            )

    def test_one_class_threshold_raises(self) -> None:
        x_train, y_train = _mixed_matrix(200, 50, seed=42)
        x_cal, y_cal = _mixed_matrix(30, 10, seed=43)
        x_thr_benign_only, _ = _mixed_matrix(20, 0, seed=44)
        y_thr_all_zero = np.zeros(20, dtype=int)
        with pytest.raises(ValueError, match="one-class threshold"):
            train_sentinel_defender(
                x_train=x_train,
                y_train=y_train,
                x_calibration=x_cal,
                y_calibration=y_cal,
                x_threshold=x_thr_benign_only,
                y_threshold=y_thr_all_zero,
                catboost_seeds=(1001, 1002, 1003),
                bootstrap_seed=707,
            )

    def test_mixed_calibration_preserves_nonconstant_probabilities(self) -> None:
        x_train, y_train = _mixed_matrix(200, 50, seed=42)
        x_cal, y_cal = _mixed_matrix(40, 15, seed=43)
        x_threshold, y_threshold = _mixed_matrix(30, 10, seed=44)
        defender = train_sentinel_defender(
            x_train=x_train,
            y_train=y_train,
            x_calibration=x_cal,
            y_calibration=y_cal,
            x_threshold=x_threshold,
            y_threshold=y_threshold,
            catboost_seeds=(1001, 1002, 1003),
            bootstrap_seed=707,
        )
        x_probe, _ = _mixed_matrix(20, 10, seed=99)
        decisions = defender.decide_batch(x_probe)
        probs = [d.ensemble_probability for d in decisions]
        assert len(set(probs)) > 1, "calibrated probabilities are constant"
