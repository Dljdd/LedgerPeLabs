"""Action precedence, fallback, and operating-budget contracts."""

from __future__ import annotations

import inspect
import json
import math
from datetime import UTC, datetime
from decimal import Decimal

import pytest
from pydantic import ValidationError

from apar.contracts.decisions import Action
from apar.contracts.events import EventKind, Rail
from apar.defense.contracts import ObservedEvent, PolicyThresholds
from apar.defense.policy import ActionPolicy, DefenseDecision, OperatingBudget
from apar.defense.rules import DefenseReason, RuleEngine, RuleResult
from apar.features.state import FeatureVector

NOW = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)


def event(
    *,
    integrity_status: str = "not_applicable",
    actor_id: str = "actor-a",
) -> ObservedEvent:
    return ObservedEvent(
        event_id="event-current",
        payment_id="payment-current",
        rail=Rail.AGENTIC if integrity_status != "not_applicable" else Rail.CARD,
        event_type=EventKind.AUTHORIZATION,
        amount=Decimal("100.00"),
        currency="USD",
        event_time=NOW,
        available_at=NOW,
        decision_at=NOW,
        actor_id=actor_id,
        counterparty_id="counterparty-a",
        optional_refs={},
        integrity_status=integrity_status,
        integrity_reason="receipt_failed" if integrity_status == "fail" else None,
        is_decision_point=True,
    )


def vector(**values: float) -> FeatureVector:
    return FeatureVector(
        event_id="event-current",
        decision_at=NOW,
        source_event_ids=("source-1",),
        max_source_available_at=datetime(2026, 1, 1, 11, 59, tzinfo=UTC),
        catalog_digest="a" * 64,
        values=values,
    )


def rules(**values: float) -> RuleResult:
    return RuleEngine.default().evaluate(event(), vector(**values))


def test_policy_has_the_closed_four_input_interface() -> None:
    parameters = tuple(inspect.signature(ActionPolicy.default().choose).parameters)
    assert parameters[:4] == ("event", "rule_result", "calibrated_score", "thresholds")


def test_integrity_failure_cannot_be_overridden_by_low_risk() -> None:
    decision = ActionPolicy.default().choose(
        event(integrity_status="fail"),
        RuleResult.clear(),
        calibrated_score=0.0,
        thresholds=PolicyThresholds(challenge=0.8, decline=0.95),
    )
    assert decision.action is Action.DECLINE
    assert decision.reason_codes == (DefenseReason.INTEGRITY_FAILURE,)
    assert decision.score == 1.0
    assert decision.fallback_used is False
    assert decision.evidence_source_ids == ("event-current",)


def test_required_data_failure_cannot_be_overridden_by_model_or_clear_rules() -> None:
    decision = ActionPolicy.default().choose(
        event(actor_id=""),
        RuleResult.clear(),
        calibrated_score=0.0,
        thresholds=PolicyThresholds(challenge=0.8, decline=0.95),
    )
    assert decision.action is Action.DECLINE
    assert decision.reason_codes == (DefenseReason.REQUIRED_DATA_MISSING,)


@pytest.mark.parametrize(
    ("score", "expected"),
    (
        (0.799999, Action.APPROVE),
        (0.8, Action.CHALLENGE),
        (0.949999, Action.CHALLENGE),
        (0.95, Action.DECLINE),
        (1.0, Action.DECLINE),
    ),
)
def test_model_threshold_edges_use_decline_before_challenge_and_greater_equal(
    score: float,
    expected: Action,
) -> None:
    decision = ActionPolicy.default().choose(
        event(),
        RuleResult.clear(),
        calibrated_score=score,
        thresholds=PolicyThresholds(challenge=0.8, decline=0.95),
    )
    assert decision.action is expected
    assert decision.score == score


def test_hybrid_uses_stronger_continuous_rule_score() -> None:
    decision = ActionPolicy.default().choose(
        event(),
        rules(actor_count_1m=8.0),
        calibrated_score=0.1,
        thresholds=PolicyThresholds(challenge=0.6, decline=0.95),
    )
    assert decision.action is Action.CHALLENGE
    assert decision.score == pytest.approx(0.8)
    assert decision.reason_codes == (DefenseReason.ACTOR_VELOCITY,)


def test_degraded_state_challenges_at_exact_rule_score() -> None:
    decision = ActionPolicy.default().choose(
        event(),
        rules(dq_degraded_state=1.0),
        calibrated_score=None,
        thresholds=None,
        model_failure=DefenseReason.MODEL_UNAVAILABLE,
        failed_component_version="catboost-v1",
    )
    assert decision.action is Action.CHALLENGE
    assert decision.score == 0.60
    assert decision.reason_codes == (
        DefenseReason.FEATURE_STATE_DEGRADED,
        DefenseReason.MODEL_UNAVAILABLE,
    )
    assert decision.fallback_used is True


@pytest.mark.parametrize(
    "failure",
    (DefenseReason.MODEL_UNAVAILABLE, DefenseReason.MODEL_TIMEOUT),
)
def test_model_failure_uses_rules_only_fallback_with_stable_audit(
    failure: DefenseReason,
) -> None:
    decision = ActionPolicy.default().choose(
        event(),
        rules(graph_actor_fanout=15.0),
        calibrated_score=None,
        thresholds=PolicyThresholds(challenge=0.1, decline=0.2),
        model_failure=failure,
        failed_component_version="catboost-v1",
        latency_ms=12.5,
    )
    assert decision.action is Action.DECLINE
    assert decision.fallback_used is True
    assert decision.fallback_reason is failure
    assert decision.failed_component_version == "catboost-v1"
    assert decision.latency_ms == 12.5
    assert decision.reason_codes[-1] is failure


def test_model_failure_with_clear_rules_approves_but_remains_visible() -> None:
    decision = ActionPolicy.default().choose(
        event(),
        RuleResult.clear(),
        calibrated_score=None,
        thresholds=None,
    )
    assert decision.action is Action.APPROVE
    assert decision.score == 0.0
    assert decision.reason_codes == (DefenseReason.MODEL_UNAVAILABLE,)
    assert decision.fallback_used is True


def test_available_clear_score_has_clear_approve_without_fallback() -> None:
    decision = ActionPolicy.default().choose(
        event(),
        RuleResult.clear(),
        calibrated_score=0.2,
        thresholds=PolicyThresholds(challenge=0.8, decline=0.95),
    )
    assert decision.action is Action.APPROVE
    assert decision.reason_codes == ()
    assert decision.fallback_used is False
    assert decision.fallback_reason is None


@pytest.mark.parametrize("value", [-0.01, 1.01, math.nan, math.inf, -math.inf])
def test_policy_rejects_invalid_calibrated_scores(value: float) -> None:
    with pytest.raises(ValueError, match="calibrated_score"):
        ActionPolicy.default().choose(
            event(),
            RuleResult.clear(),
            calibrated_score=value,
            thresholds=PolicyThresholds(challenge=0.8, decline=0.95),
        )


def test_policy_rejects_incoherent_model_state_and_wrong_event() -> None:
    policy = ActionPolicy.default()
    with pytest.raises(ValueError, match="model failure"):
        policy.choose(
            event(),
            RuleResult.clear(),
            calibrated_score=0.5,
            thresholds=PolicyThresholds(challenge=0.8, decline=0.95),
            model_failure=DefenseReason.MODEL_TIMEOUT,
        )
    with pytest.raises(ValueError, match="decision-point"):
        policy.choose(
            event().model_copy(update={"is_decision_point": False}),
            RuleResult.clear(),
            calibrated_score=0.5,
            thresholds=PolicyThresholds(challenge=0.8, decline=0.95),
        )
    tampered = RuleResult.clear().model_copy(update={"score": 0.5})
    with pytest.raises(ValidationError, match="aggregate risk"):
        policy.choose(
            event(),
            tampered,
            calibrated_score=0.5,
            thresholds=PolicyThresholds(challenge=0.8, decline=0.95),
        )


def test_invalid_score_is_rejected_even_when_threshold_artifact_is_unavailable() -> None:
    with pytest.raises(ValueError, match="calibrated_score"):
        ActionPolicy.default().choose(
            event(),
            RuleResult.clear(),
            calibrated_score=float("nan"),
            thresholds=None,
        )


@pytest.mark.parametrize(
    "updates",
    (
        {"challenge_rate_max": -0.001},
        {"challenge_rate_max": 1.001},
        {"false_decline_rate_max": math.nan},
        {"review_case_rate_max": math.inf},
    ),
)
def test_operating_budget_rejects_invalid_rates(updates: dict[str, float]) -> None:
    with pytest.raises(ValidationError):
        OperatingBudget(**updates)


def test_policy_contracts_are_immutable_deterministic_and_json_safe() -> None:
    budget = OperatingBudget()
    policy = ActionPolicy.default()
    decision = policy.choose(
        event(),
        RuleResult.clear(),
        calibrated_score=0.2,
        thresholds=PolicyThresholds(challenge=0.8, decline=0.95),
    )
    assert budget == OperatingBudget(
        challenge_rate_max=0.02,
        false_decline_rate_max=0.001,
        review_case_rate_max=0.01,
    )
    assert json.dumps(decision.model_dump(mode="json"), sort_keys=True, allow_nan=False)
    with pytest.raises(ValidationError):
        decision.action = Action.DECLINE  # type: ignore[misc]
    with pytest.raises(ValidationError):
        DefenseDecision(
            event_id="event-current",
            action=Action.APPROVE,
            score=float("nan"),
            rule_score=0.0,
            calibrated_score=None,
            reason_codes=(),
            evidence_source_ids=(),
            fallback_used=False,
            fallback_reason=None,
            failed_component_version=None,
            latency_ms=0.0,
            policy_version="1.0.0",
        )
