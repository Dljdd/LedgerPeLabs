"""Trust-binding and integrity decision regression tests."""

from __future__ import annotations

import numpy as np
import pytest

from apar.defense.sentinel import SentinelAction, SentinelDecision


class TestTrustBinding:
    def test_agentic_integrity_failure_declines(self) -> None:
        from apar.defense.sentinel import train_sentinel_defender

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
        defender = train_sentinel_defender(
            x_train=x_train, y_train=y_train,
            x_calibration=x_cal, y_calibration=y_cal,
            x_threshold=x_thr, y_threshold=y_thr,
            catboost_seeds=(1, 2, 3), bootstrap_seed=99,
        )
        low_risk = np.zeros(7)
        decision = defender.decide(low_risk, trust_failure=True)
        assert decision.action == SentinelAction.DECLINE_HOLD

    def test_valid_agentic_control_not_declined_for_rail_alone(self) -> None:
        from apar.defense.sentinel import train_sentinel_defender

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
        defender = train_sentinel_defender(
            x_train=x_train, y_train=y_train,
            x_calibration=x_cal, y_calibration=y_cal,
            x_threshold=x_thr, y_threshold=y_thr,
            catboost_seeds=(1, 2, 3), bootstrap_seed=99,
        )
        benign_agentic = rng.normal(0.0, 0.3, (7,))
        decision = defender.decide(benign_agentic, trust_failure=False)
        assert decision.action != SentinelAction.DECLINE_HOLD or decision.trust_failure
