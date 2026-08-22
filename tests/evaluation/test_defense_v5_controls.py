"""Baseline controls regression tests."""

from __future__ import annotations

import pytest

from apar.evaluation.v5_evaluation import run_v5_controls


class TestControls:
    def test_all_required_controls_present(self) -> None:
        results = run_v5_controls()
        names = {r.name for r in results}
        required = {
            "label_shuffle", "identity_rename",
            "future_causality", "equal_time_isolation",
            "benign_only", "fraud_only_diagnostic", "feature_leakage",
        }
        missing = required - names
        assert not missing, f"missing controls: {missing}"

    def test_fraud_only_diagnostic_never_passes(self) -> None:
        results = run_v5_controls()
        fraud_only = [r for r in results if r.name == "fraud_only_diagnostic"]
        assert len(fraud_only) == 1
        assert fraud_only[0].passed is False

    @pytest.mark.parametrize("control_name", ["label_shuffle"])
    def test_discrimination_controls_fail_on_placeholder(self, control_name: str) -> None:
        """Label shuffle must report failure until a real implementation exists."""
        results = run_v5_controls()
        match = [r for r in results if r.name == control_name]
        assert len(match) == 1
        # The control must be implemented (not just a placeholder string).
        assert "not yet implemented" not in match[0].detail, f"{control_name} is still a placeholder"
