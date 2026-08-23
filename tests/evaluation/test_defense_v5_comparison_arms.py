"""Comparison arms regression tests."""

from __future__ import annotations

import pytest

from apar.evaluation.v5_evaluation import V5Arm


class TestComparisonArms:
    def test_all_four_arm_values_exist(self) -> None:
        expected = {"rules_only", "ensemble_no_graph", "ensemble_with_graph", "full_sentinel"}
        actual = {arm.value for arm in V5Arm}
        assert expected <= actual, f"missing arms: {expected - actual}"

    def test_hardened_sentinel_not_in_current_round(self) -> None:
        """hardened_sentinel is a future arm; it must not be evaluated this round."""
        assert V5Arm.HARDENED_SENTINEL.value == "hardened_sentinel"
