"""Trust runner integration regression tests."""

from __future__ import annotations

import numpy as np
import pytest

from apar.defense.sentinel import SentinelAction, train_sentinel_defender


def _make_defender():
    rng = np.random.RandomState(42)
    x_train = np.vstack([
        rng.normal(0.0, 0.5, (100, 7)),
        rng.normal(3.0, 0.5, (30, 7)),
    ])
    y_train = np.array([0] * 100 + [1] * 30)
    x_cal = np.vstack([x_train[:20], x_train[100:110]])
    y_cal = np.concatenate([y_train[:20], y_train[100:110]])
    x_thr = np.vstack([x_train[20:30], x_train[110:115]])
    y_thr = np.concatenate([y_train[20:30], y_train[110:115]])
    return train_sentinel_defender(
        x_train=x_train, y_train=y_train,
        x_calibration=x_cal, y_calibration=y_cal,
        x_threshold=x_thr, y_threshold=y_thr,
        catboost_seeds=(1, 2, 3), bootstrap_seed=99,
    )


class TestTrustRunnerIntegration:
    def test_decide_batch_with_trust_failures(self) -> None:
        """RED: decide_batch must accept per-row trust_failure sequence."""
        defender = _make_defender()
        features = np.zeros((3, 7))
        trust_failures = [True, False, False]

        decisions = defender.decide_batch(features, trust_failures=trust_failures)
        assert decisions[0].action == SentinelAction.DECLINE_HOLD
        assert decisions[1].action != SentinelAction.DECLINE_HOLD or decisions[1].trust_failure is False

    def test_decide_batch_rejects_mismatched_trust_length(self) -> None:
        defender = _make_defender()
        features = np.zeros((3, 7))
        with pytest.raises(ValueError, match="trust"):
            defender.decide_batch(features, trust_failures=[True])
