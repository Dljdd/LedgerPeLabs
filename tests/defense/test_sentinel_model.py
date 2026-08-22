"""Calibrated Sentinel ensemble and action policy tests."""

from __future__ import annotations

import numpy as np
import pytest

from apar.defense.sentinel import (
    SentinelAction,
    SentinelDefender,
    train_sentinel_defender,
)


def _make_matrix(n_benign: int, n_fraud: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.RandomState(seed)
    benign = rng.normal(loc=0.0, scale=0.5, size=(n_benign, 7))
    fraud = rng.normal(loc=3.0, scale=0.5, size=(n_fraud, 7))
    x = np.vstack([benign, fraud])
    y = np.array([0] * n_benign + [1] * n_fraud)
    return x, y


@pytest.fixture(scope="module")
def defender() -> SentinelDefender:
    x_train, y_train = _make_matrix(200, 50, seed=42)
    x_cal, y_cal = _make_matrix(60, 15, seed=43)
    x_threshold, y_threshold = _make_matrix(40, 10, seed=44)
    return train_sentinel_defender(
        x_train=x_train,
        y_train=y_train,
        x_calibration=x_cal,
        y_calibration=y_cal,
        x_threshold=x_threshold,
        y_threshold=y_threshold,
        catboost_seeds=(1001, 1002, 1003),
        bootstrap_seed=707,
    )


class TestSentinelEnsemble:
    def test_three_seeds_trained(self, defender: SentinelDefender) -> None:
        assert len(defender.model_members) == 3

    def test_actions_are_valid(self, defender: SentinelDefender) -> None:
        x_test, _ = _make_matrix(20, 5, seed=45)
        decisions = defender.decide_batch(x_test)
        valid = {a.value for a in SentinelAction}
        assert all(d.action in valid for d in decisions)

    def test_novelty_never_declines_alone(self, defender: SentinelDefender) -> None:
        novel_low_score = np.array([[5.0, 5.0, 5.0, 5.0, 5.0, 5.0, -10.0]])
        decision = defender.decide(novel_low_score[0], novelty_score=1.0, trust_failure=False)
        assert decision.action != SentinelAction.DECLINE_HOLD

    def test_trust_failure_declines(self, defender: SentinelDefender) -> None:
        low_risk = np.zeros(7)
        decision = defender.decide(low_risk, novelty_score=0.0, trust_failure=True)
        assert decision.action == SentinelAction.DECLINE_HOLD

    def test_action_ordering(self, defender: SentinelDefender) -> None:
        order = [
            SentinelAction.APPROVE,
            SentinelAction.CHALLENGE,
            SentinelAction.REVIEW_HOLD,
            SentinelAction.DECLINE_HOLD,
        ]
        values = [a.severity for a in order]
        assert values == sorted(values)

    def test_deterministic(self, defender: SentinelDefender) -> None:
        x_test, _ = _make_matrix(10, 3, seed=46)
        d1 = defender.decide_batch(x_test)
        d2 = defender.decide_batch(x_test)
        assert [d.action for d in d1] == [d.action for d in d2]
