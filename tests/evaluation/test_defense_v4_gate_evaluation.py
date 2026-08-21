"""Conservative gate evaluation tests for Defend v4."""

from __future__ import annotations

import pytest

from apar.evaluation.v4_gate_evaluation import (
    ArmGateEvidence,
    evaluate_gates,
)
from apar.v4_protocol import V4GateValues


def _passing_evidence(arm: str = "rules_only") -> ArmGateEvidence:
    return ArmGateEvidence(
        arm=arm,
        family_recall_min=0.8,
        calibration_ece_max=0.05,
        challenge_rate_max=0.01,
        false_decline_rate_max=0.0005,
        review_case_rate_max=0.005,
        p95_decision_latency_ms_max=10.0,
        captured_value_min=0.7,
        escaped_value_max=0.3,
        p95_time_to_alert_seconds_max=120.0,
    )


def test_all_gates_pass() -> None:
    result = evaluate_gates(_passing_evidence(), gates=V4GateValues())
    assert result.gate_outcome.passed is True
    assert result.gate_outcome.codes == ()


def test_family_recall_failure_detected() -> None:
    evidence = _passing_evidence().model_copy(update={"family_recall_min": 0.3})
    result = evaluate_gates(evidence, gates=V4GateValues())
    assert "FAMILY_COVERAGE" in result.gate_outcome.codes


def test_calibration_failure_detected() -> None:
    evidence = _passing_evidence().model_copy(update={"calibration_ece_max": 0.15})
    result = evaluate_gates(evidence, gates=V4GateValues())
    assert "CALIBRATION" in result.gate_outcome.codes


def test_challenge_budget_failure_detected() -> None:
    evidence = _passing_evidence().model_copy(update={"challenge_rate_max": 0.03})
    result = evaluate_gates(evidence, gates=V4GateValues())
    assert "CHALLENGE_BUDGET" in result.gate_outcome.codes


def test_undefined_metric_fails_closed() -> None:
    evidence = ArmGateEvidence(arm="gbdt_only")
    result = evaluate_gates(evidence, gates=V4GateValues())
    assert not result.gate_outcome.passed
    assert len(result.gate_outcome.codes) == 9


def test_multiple_failures_retained() -> None:
    evidence = _passing_evidence().model_copy(
        update={"family_recall_min": 0.2, "calibration_ece_max": 0.5}
    )
    result = evaluate_gates(evidence, gates=V4GateValues())
    assert "FAMILY_COVERAGE" in result.gate_outcome.codes
    assert "CALIBRATION" in result.gate_outcome.codes
    assert len(result.gate_outcome.codes) >= 2
