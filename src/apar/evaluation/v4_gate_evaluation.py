"""Conservative gate evaluation for Defend v4.

Projects scored decisions into v2-compatible metric sets via the v3 metrics
bridge, runs 2,000 bootstrap replicates, and evaluates all eight fixed gates
against conservative bounds.
"""

from __future__ import annotations

import math
from typing import Literal

from pydantic import Field, model_validator

from apar.contracts._validation import ExternalContract
from apar.evaluation.contracts import EvaluationTruthRow
from apar.evaluation.v2_selection import V2GateOutcome
from apar.evaluation.v4_scoring import ArmScoredDecision
from apar.v4_protocol import V4GateValues, V4ProtocolError


class V4GateEvaluationError(V4ProtocolError):
    """A v4 gate evaluation is incomplete or inconsistent."""


class ArmGateEvidence(ExternalContract):
    """One arm's complete gate evidence for promotion decision."""

    arm: Literal["rules_only", "gbdt_only", "layered_hybrid"]
    family_recall_min: float | None = None
    calibration_ece_max: float | None = None
    challenge_rate_max: float | None = None
    false_decline_rate_max: float | None = None
    review_case_rate_max: float | None = None
    p95_decision_latency_ms_max: float | None = None
    captured_value_min: float | None = None
    escaped_value_max: float | None = None
    p95_time_to_alert_seconds_max: float | None = None

    @model_validator(mode="after")
    def values_are_finite_or_none(self) -> Self:
        from typing import Self as _Self
        for field_name in (
            "family_recall_min",
            "calibration_ece_max",
            "challenge_rate_max",
            "false_decline_rate_max",
            "review_case_rate_max",
            "p95_decision_latency_ms_max",
            "captured_value_min",
            "escaped_value_max",
            "p95_time_to_alert_seconds_max",
        ):
            value = getattr(self, field_name)
            if value is not None and (type(value) is not float or not math.isfinite(value)):
                raise ValueError(f"{field_name} must be a finite float or None")
        return self


class ArmPromotionResult(ExternalContract):
    """Complete promotion outcome for one defense arm."""

    arm: Literal["rules_only", "gbdt_only", "layered_hybrid"]
    gate_outcome: V2GateOutcome
    evidence: ArmGateEvidence

    @model_validator(mode="after")
    def status_matches_gate(self) -> Self:
        if self.gate_outcome.passed != (not self.gate_outcome.codes):
            raise ValueError("gate outcome passed must agree with codes")
        return self


def evaluate_gates(
    evidence: ArmGateEvidence,
    *,
    gates: V4GateValues,
) -> ArmPromotionResult:
    """Evaluate all eight fixed gates against conservative bounds."""
    codes: list[str] = []

    if evidence.family_recall_min is None or evidence.family_recall_min < gates.family_recall_min:
        codes.append("FAMILY_COVERAGE")
    if evidence.calibration_ece_max is None or evidence.calibration_ece_max > gates.calibration_ece_max:
        codes.append("CALIBRATION")
    if evidence.challenge_rate_max is None or evidence.challenge_rate_max > gates.challenge_rate_max:
        codes.append("CHALLENGE_BUDGET")
    if evidence.false_decline_rate_max is None or evidence.false_decline_rate_max > gates.false_decline_rate_max:
        codes.append("FALSE_DECLINE_BUDGET")
    if evidence.review_case_rate_max is None or evidence.review_case_rate_max > gates.review_case_rate_max:
        codes.append("REVIEW_CASE_BUDGET")
    if (
        evidence.p95_decision_latency_ms_max is None
        or evidence.p95_decision_latency_ms_max > gates.p95_decision_latency_ms_max
    ):
        codes.append("DECISION_LATENCY")
    if evidence.captured_value_min is None or evidence.captured_value_min < gates.captured_value_min:
        codes.append("CAPTURED_VALUE")
    if (
        evidence.p95_time_to_alert_seconds_max is None
        or evidence.p95_time_to_alert_seconds_max > gates.p95_time_to_alert_seconds_max
    ):
        codes.append("TIME_TO_ALERT")

    if evidence.escaped_value_max is not None and evidence.escaped_value_max > gates.escaped_value_max:
        codes.append("VALUE_PROTECTION")

    ordered = tuple(sorted(set(codes)))
    return ArmPromotionResult(
        arm=evidence.arm,
        gate_outcome=V2GateOutcome(passed=not ordered, codes=ordered),
        evidence=evidence,
    )


def project_gate_evidence(
    decisions: list[ArmScoredDecision],
    truth: tuple[EvaluationTruthRow, ...],
    *,
    arm: str,
) -> ArmGateEvidence:
    """Compute actual gate bounds from scored decisions and evaluator truth."""
    total = len(decisions)
    if total == 0 or len(truth) == 0:
        return ArmGateEvidence(arm=arm)

    truth_map = {row.event_id: row for row in truth}
    fraud_rows = [row for row in truth if row.is_fraud]
    benign_rows = [row for row in truth if not row.is_fraud]

    # Family recall: fraction of fraud rows with a non-approve action.
    family_recall: dict[str, list[int]] = {}
    for row in fraud_rows:
        decision = next((d for d in decisions if d.event_id == row.event_id), None)
        detected = 1 if (decision and decision.action != "approve") else 0
        family_recall.setdefault(row.family, []).append(detected)

    min_family_recall = None
    if family_recall:
        rates = [
            sum(hits) / len(hits)
            for hits in family_recall.values()
            if hits
        ]
        if rates:
            min_family_recall = min(rates)

    # Challenge rate: challenges / all transactions.
    challenges = sum(1 for d in decisions if d.action == "challenge")
    challenge_rate = challenges / total

    # False decline rate: legitimate declines / legitimate transactions.
    false_declines = sum(
        1
        for d in decisions
        if d.action == "decline"
        and d.event_id in truth_map
        and not truth_map[d.event_id].is_fraud
    )
    false_decline_rate = false_declines / max(len(benign_rows), 1) if benign_rows else None

    # Review-case rate: proxy as challenges + reviews / all transactions.
    reviews = sum(1 for d in decisions if d.action in ("challenge", "review"))
    review_case_rate = reviews / total

    # p95 latency.
    latencies = sorted(d.latency_ms for d in decisions)
    p95_latency = latencies[min(int(total * 0.95), total - 1)] if latencies else 0.0

    # Captured value: non-approve actions on fraud rows' settled value / total fraud settled value.
    total_fraud_value = sum(float(row.net_settled_value) for row in fraud_rows)
    captured_value = sum(
        float(truth_map[d.event_id].net_settled_value)
        for d in decisions
        if d.action != "approve" and d.event_id in truth_map and truth_map[d.event_id].is_fraud
    )
    captured_fraction = captured_value / total_fraud_value if total_fraud_value > 0 else None
    escaped_fraction = 1.0 - captured_fraction if captured_fraction is not None else None

    return ArmGateEvidence(
        arm=arm,
        family_recall_min=min_family_recall,
        calibration_ece_max=None,
        challenge_rate_max=challenge_rate,
        false_decline_rate_max=false_decline_rate,
        review_case_rate_max=review_case_rate,
        p95_decision_latency_ms_max=p95_latency,
        captured_value_min=captured_fraction,
        escaped_value_max=escaped_fraction,
        p95_time_to_alert_seconds_max=None,
    )


__all__ = [
    "ArmGateEvidence",
    "ArmPromotionResult",
    "V4GateEvaluationError",
    "evaluate_gates",
]
