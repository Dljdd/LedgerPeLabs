"""Evaluation metrics and controls tests for Defend v5."""

from __future__ import annotations

import numpy as np
import pytest

from apar.defense.sentinel import SentinelAction
from apar.evaluation.v5_evaluation import (
    V5Arm,
    evaluate_v5_arm,
)


class TestMetrics:
    def test_perfect_separation(self) -> None:
        y_true = np.array([0] * 50 + [1] * 20)
        actions = [SentinelAction.APPROVE] * 50 + [SentinelAction.DECLINE_HOLD] * 20
        probs = np.concatenate([
            np.random.RandomState(1).uniform(0, 0.3, 50),
            np.random.RandomState(2).uniform(0.7, 1.0, 20),
        ])
        result = evaluate_v5_arm(
            arm=V5Arm.FULL_SENTINEL,
            y_true=y_true,
            actions=actions,
            probabilities=probs,
            campaign_ids=np.array([f"c{i}" for i in range(70)]),
            amounts=np.ones(70),
        )
        assert result.recall is not None and result.recall >= 0.95
        assert result.false_decline_rate == 0.0

    def test_false_decline_rate(self) -> None:
        y_true = np.array([0] * 10 + [1] * 10)
        actions = [SentinelAction.APPROVE] * 9 + [SentinelAction.DECLINE_HOLD] * 11
        probs = np.full(20, 0.5)
        result = evaluate_v5_arm(
            arm=V5Arm.RULES_ONLY,
            y_true=y_true,
            actions=actions,
            probabilities=probs,
            campaign_ids=np.array([f"c{i}" for i in range(20)]),
            amounts=np.ones(20),
        )
        assert result.false_decline_rate == pytest.approx(1 / 10)

    def test_non_finite_fails(self) -> None:
        with pytest.raises(ValueError):
            evaluate_v5_arm(
                arm=V5Arm.FULL_SENTINEL,
                y_true=np.array([]),
                actions=[],
                probabilities=np.array([]),
                campaign_ids=np.array([]),
                amounts=np.array([]),
            )

    def test_operational_rates_use_legitimate_denominators_and_retain_ece_p99(self) -> None:
        result = evaluate_v5_arm(
            arm=V5Arm.ENSEMBLE_NO_GRAPH,
            y_true=np.array([0, 0, 1]),
            actions=[
                SentinelAction.CHALLENGE,
                SentinelAction.APPROVE,
                SentinelAction.REVIEW_HOLD,
            ],
            probabilities=np.array([0.2, 0.1, 0.8]),
            campaign_ids=np.array(["c0", "c1", "c2"]),
            amounts=np.ones(3),
        )
        assert result.challenge_rate == pytest.approx(1 / 2)
        assert result.review_rate == 0.0
        assert result.expected_calibration_error is not None
        assert result.p99_latency_ms is None

    def test_one_class_evaluation_fails_closed(self) -> None:
        with pytest.raises(ValueError, match="both classes"):
            evaluate_v5_arm(
                arm=V5Arm.RULES_ONLY,
                y_true=np.array([0, 0]),
                actions=[SentinelAction.APPROVE, SentinelAction.APPROVE],
                probabilities=np.array([0.1, 0.2]),
                campaign_ids=np.array(["c0", "c1"]),
                amounts=np.ones(2),
            )
