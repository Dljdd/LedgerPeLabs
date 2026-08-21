"""Action precedence, fallback, and operating-budget contracts."""

from __future__ import annotations

import inspect
import json
import math
from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest
from pydantic import ValidationError

from apar.contracts.decisions import Action
from apar.contracts.events import EventKind, Rail
from apar.defense.contracts import ObservedEvent, PolicyThresholds
from apar.defense.policy import ActionPolicy, DefenseDecision, OperatingBudget
from apar.defense.rules import (
    DefenseReason,
    RuleEngine,
    RuleHit,
    RuleManifest,
    RuleResult,
    RuleSeverity,
)
from apar.features.state import FeatureVector

NOW = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)


def event(
    *,
    integrity_status: str = "not_applicable",
    actor_id: str = "actor-a",
    event_id: str = "event-current",
    rail: Rail | None = None,
    event_type: EventKind = EventKind.AUTHORIZATION,
    decision_at: datetime | None = NOW,
    is_decision_point: bool = True,
) -> ObservedEvent:
    selected_rail = rail or (
        Rail.AGENTIC if integrity_status != "not_applicable" else Rail.CARD
    )
    return ObservedEvent(
        event_id=event_id,
        payment_id="payment-current",
        rail=selected_rail,
        event_type=event_type,
        amount=Decimal("100.00"),
        currency="USD",
        event_time=NOW,
        available_at=NOW,
        decision_at=decision_at,
        actor_id=actor_id,
        counterparty_id="counterparty-a",
        optional_refs={},
        integrity_status=integrity_status,
        integrity_reason="receipt_failed" if integrity_status == "fail" else None,
        is_decision_point=is_decision_point,
    )


def vector(
    *,
    event_id: str = "event-current",
    decision_at: datetime = NOW,
    catalog_digest: str = "a" * 64,
    **values: float,
) -> FeatureVector:
    return FeatureVector(
        event_id=event_id,
        decision_at=decision_at,
        source_event_ids=("source-1",),
        max_source_available_at=datetime(2026, 1, 1, 11, 59, tzinfo=UTC),
        catalog_digest=catalog_digest,
        values=values,
    )


def bound_clear() -> tuple[RuleResult, FeatureVector]:
    feature_vector = vector()
    return RuleEngine.default().evaluate(event(), feature_vector), feature_vector


def test_policy_has_the_closed_four_input_interface() -> None:
    parameters = inspect.signature(ActionPolicy.default().choose).parameters
    assert tuple(parameters)[:4] == (
        "event",
        "rule_result",
        "calibrated_score",
        "thresholds",
    )
    assert parameters["vector"].kind is inspect.Parameter.KEYWORD_ONLY
    assert parameters["vector"].default is None


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
    ("observed", "expected"),
    (
        (
            event(rail=Rail.CARD, integrity_status="fail"),
            (DefenseReason.REQUIRED_DATA_MISSING,),
        ),
        (
            event(rail=Rail.AGENTIC, integrity_status="not_applicable"),
            (DefenseReason.REQUIRED_DATA_MISSING,),
        ),
        (
            event(is_decision_point=False),
            (DefenseReason.REQUIRED_DATA_MISSING,),
        ),
        (
            event(rail=Rail.AGENTIC, integrity_status="fail"),
            (DefenseReason.INTEGRITY_FAILURE,),
        ),
    ),
)
def test_policy_reconstructs_exact_schema_mandatory_reasons(
    observed: ObservedEvent,
    expected: tuple[DefenseReason, ...],
) -> None:
    decision = ActionPolicy.default().choose(
        observed,
        RuleResult.clear(),
        calibrated_score=0.0,
        thresholds=PolicyThresholds(challenge=0.8, decline=0.95),
    )
    assert decision.action is Action.DECLINE
    assert decision.reason_codes == expected


def test_policy_rejects_cross_event_and_cross_decision_time_rule_replay() -> None:
    feature_vector = vector(actor_count_1m=4.0)
    result = RuleEngine.default().evaluate(event(), feature_vector)
    policy = ActionPolicy.default()
    other_time = NOW + timedelta(seconds=1)
    with pytest.raises(ValueError, match="event binding"):
        policy.choose(
            event(event_id="event-other"),
            result,
            0.1,
            PolicyThresholds(challenge=0.8, decline=0.95),
            vector=vector(event_id="event-other", actor_count_1m=4.0),
        )
    with pytest.raises(ValueError, match="decision-time binding"):
        policy.choose(
            event(decision_at=other_time),
            result,
            0.1,
            PolicyThresholds(challenge=0.8, decline=0.95),
            vector=vector(decision_at=other_time, actor_count_1m=4.0),
        )


def test_policy_rejects_vector_and_catalog_provenance_replay() -> None:
    original = vector(actor_count_1m=4.0)
    result = RuleEngine.default().evaluate(event(), original)
    policy = ActionPolicy.default()
    with pytest.raises(ValueError, match="vector provenance"):
        policy.choose(
            event(),
            result,
            0.1,
            PolicyThresholds(challenge=0.8, decline=0.95),
            vector=vector(actor_count_1m=5.0),
        )
    with pytest.raises(ValueError, match="catalog provenance"):
        policy.choose(
            event(),
            result,
            0.1,
            PolicyThresholds(challenge=0.8, decline=0.95),
            vector=vector(catalog_digest="b" * 64, actor_count_1m=4.0),
        )
    for replay in (
        original.model_copy(update={"source_event_ids": ("source-other",)}),
        original.model_copy(
            update={"max_source_available_at": NOW - timedelta(seconds=30)}
        ),
    ):
        with pytest.raises(ValueError, match="vector provenance"):
            policy.choose(
                event(),
                result,
                0.1,
                PolicyThresholds(challenge=0.8, decline=0.95),
                vector=replay,
            )


def test_bound_clear_result_still_requires_exact_vector() -> None:
    feature_vector = vector()
    bound_clear = RuleEngine.default().evaluate(event(), feature_vector)
    assert bound_clear.hits == ()
    with pytest.raises(ValueError, match="requires its feature vector"):
        ActionPolicy.default().choose(
            event(),
            bound_clear,
            0.1,
            PolicyThresholds(challenge=0.8, decline=0.95),
        )
    assert ActionPolicy.default().choose(
        event(),
        bound_clear,
        0.1,
        PolicyThresholds(challenge=0.8, decline=0.95),
        vector=feature_vector,
    ).action is Action.APPROVE


def test_unbound_neutral_sentinel_rejects_claimed_vector() -> None:
    with pytest.raises(ValueError, match="cannot claim vector provenance"):
        ActionPolicy.default().choose(
            event(),
            RuleResult.clear(),
            0.1,
            PolicyThresholds(challenge=0.8, decline=0.95),
            vector=vector(),
        )


def test_unbound_neutral_sentinel_is_only_for_current_mandatory_reconstruction() -> None:
    with pytest.raises(ValueError, match="only valid for mandatory reconstruction"):
        ActionPolicy.default().choose(
            event(),
            RuleResult.clear(),
            0.1,
            PolicyThresholds(challenge=0.8, decline=0.95),
        )


def test_policy_binds_complete_manifest_not_only_same_version() -> None:
    custom_manifest = RuleManifest(actor_count_1m=3.0)
    feature_vector = vector(actor_count_1m=3.0)
    result = RuleEngine(custom_manifest).evaluate(event(), feature_vector)
    with pytest.raises(ValueError, match="manifest provenance"):
        ActionPolicy.default().choose(
            event(),
            result,
            0.1,
            PolicyThresholds(challenge=0.8, decline=0.95),
            vector=feature_vector,
        )
    configured = ActionPolicy(rule_manifest=custom_manifest)
    assert configured.choose(
        event(),
        result,
        0.1,
        PolicyThresholds(challenge=0.8, decline=0.95),
        vector=feature_vector,
    ).action is Action.APPROVE


def test_policy_rejects_fabricated_nonmandatory_hit_with_real_bindings() -> None:
    feature_vector = vector()
    legitimate = RuleEngine.default().evaluate(event(), feature_vector)
    fabricated_hit = RuleHit(
        reason=DefenseReason.ACTOR_VELOCITY,
        score=0.95,
        severity=RuleSeverity.DECLINE,
        mandatory=False,
        evidence_source_ids=("event-current", "source-1"),
        rule_version="1.0.0",
    )
    fabricated = legitimate.model_copy(
        update={"hits": (fabricated_hit,), "score": 0.95}
    )
    with pytest.raises(ValueError, match="semantic re-evaluation"):
        ActionPolicy.default().choose(
            event(),
            fabricated,
            0.1,
            PolicyThresholds(challenge=0.8, decline=0.95),
            vector=feature_vector,
        )


def test_policy_rejects_omitted_legitimate_engine_hit() -> None:
    feature_vector = vector(actor_count_1m=4.0)
    legitimate = RuleEngine.default().evaluate(event(), feature_vector)
    omitted = legitimate.model_copy(update={"hits": (), "score": 0.0})
    with pytest.raises(ValueError, match="semantic re-evaluation"):
        ActionPolicy.default().choose(
            event(),
            omitted,
            0.1,
            PolicyThresholds(challenge=0.8, decline=0.95),
            vector=feature_vector,
        )


def test_policy_rejects_altered_legitimate_hit_score() -> None:
    feature_vector = vector(actor_count_1m=4.0)
    legitimate = RuleEngine.default().evaluate(event(), feature_vector)
    changed_hit = legitimate.hits[0].model_copy(update={"score": 0.7})
    changed = legitimate.model_copy(update={"hits": (changed_hit,), "score": 0.7})
    with pytest.raises(ValueError, match="semantic re-evaluation"):
        ActionPolicy.default().choose(
            event(),
            changed,
            0.1,
            PolicyThresholds(challenge=0.8, decline=0.95),
            vector=feature_vector,
        )


def test_policy_rejects_altered_legitimate_hit_reason() -> None:
    feature_vector = vector(actor_count_1m=4.0)
    legitimate = RuleEngine.default().evaluate(event(), feature_vector)
    changed_hit = legitimate.hits[0].model_copy(
        update={"reason": DefenseReason.GRAPH_FAN_IN}
    )
    changed = legitimate.model_copy(update={"hits": (changed_hit,)})
    with pytest.raises(ValueError, match="semantic re-evaluation"):
        ActionPolicy.default().choose(
            event(),
            changed,
            0.1,
            PolicyThresholds(challenge=0.8, decline=0.95),
            vector=feature_vector,
        )


def test_policy_rejects_altered_legitimate_hit_evidence() -> None:
    feature_vector = vector(actor_count_1m=4.0)
    legitimate = RuleEngine.default().evaluate(event(), feature_vector)
    changed_hit = legitimate.hits[0].model_copy(
        update={"evidence_source_ids": ("event-current",)}
    )
    changed = legitimate.model_copy(update={"hits": (changed_hit,)})
    with pytest.raises(ValueError, match="evidence"):
        ActionPolicy.default().choose(
            event(),
            changed,
            0.1,
            PolicyThresholds(challenge=0.8, decline=0.95),
            vector=feature_vector,
        )


def test_policy_rejects_manifest_version_change_with_unchanged_digest() -> None:
    feature_vector = vector(actor_count_1m=4.0)
    legitimate = RuleEngine.default().evaluate(event(), feature_vector)
    changed_hit = legitimate.hits[0].model_copy(update={"rule_version": "9.9.9"})
    changed = legitimate.model_copy(
        update={"hits": (changed_hit,), "manifest_version": "9.9.9"}
    )
    with pytest.raises(ValueError, match="semantic re-evaluation"):
        ActionPolicy.default().choose(
            event(),
            changed,
            0.1,
            PolicyThresholds(challenge=0.8, decline=0.95),
            vector=feature_vector,
        )


@pytest.mark.parametrize("kind", ["integrity", "required"])
def test_policy_rejects_fabricated_mandatory_hits_on_clean_event(kind: str) -> None:
    feature_vector = vector()
    source = (
        event(rail=Rail.AGENTIC, integrity_status="fail")
        if kind == "integrity"
        else event(actor_id="")
    )
    result = RuleEngine.default().evaluate(source, feature_vector)
    clean = (
        event(rail=Rail.AGENTIC, integrity_status="pass")
        if kind == "integrity"
        else event()
    )
    with pytest.raises(ValueError, match="unjustified mandatory"):
        ActionPolicy.default().choose(
            clean,
            result,
            0.1,
            PolicyThresholds(challenge=0.8, decline=0.95),
            vector=feature_vector,
        )


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
    rule_result, feature_vector = bound_clear()
    decision = ActionPolicy.default().choose(
        event(),
        rule_result,
        calibrated_score=score,
        thresholds=PolicyThresholds(challenge=0.8, decline=0.95),
        vector=feature_vector,
    )
    assert decision.action is expected
    assert decision.score == (1.0 - 1e-8 if score == 1.0 else score)


def test_hybrid_uses_stronger_continuous_rule_score() -> None:
    feature_vector = vector(actor_count_1m=8.0)
    decision = ActionPolicy.default().choose(
        event(),
        RuleEngine.default().evaluate(event(), feature_vector),
        calibrated_score=0.1,
        thresholds=PolicyThresholds(challenge=0.6, decline=0.95),
        vector=feature_vector,
    )
    assert decision.action is Action.CHALLENGE
    assert decision.score == pytest.approx(0.8)
    assert decision.reason_codes == (DefenseReason.ACTOR_VELOCITY,)


def test_degraded_state_challenges_at_exact_rule_score() -> None:
    feature_vector = vector(dq_degraded_state=1.0)
    decision = ActionPolicy.default().choose(
        event(),
        RuleEngine.default().evaluate(event(), feature_vector),
        calibrated_score=None,
        thresholds=None,
        model_failure=DefenseReason.MODEL_UNAVAILABLE,
        failed_component_version="catboost-v1",
        vector=feature_vector,
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
    feature_vector = vector(graph_actor_fanout=10.0)
    decision = ActionPolicy.default().choose(
        event(),
        RuleEngine.default().evaluate(event(), feature_vector),
        calibrated_score=None,
        thresholds=PolicyThresholds(challenge=0.1, decline=0.2),
        model_failure=failure,
        failed_component_version="catboost-v1",
        latency_ms=12.5,
        vector=feature_vector,
    )
    assert decision.action is Action.CHALLENGE
    assert decision.fallback_used is True
    assert decision.fallback_reason is failure
    assert decision.failed_component_version == "catboost-v1"
    assert decision.latency_ms == 12.5
    assert decision.reason_codes[-1] is failure


def test_model_failure_with_clear_rules_approves_but_remains_visible() -> None:
    rule_result, feature_vector = bound_clear()
    decision = ActionPolicy.default().choose(
        event(),
        rule_result,
        calibrated_score=None,
        thresholds=None,
        vector=feature_vector,
    )
    assert decision.action is Action.APPROVE
    assert decision.score == 1e-8
    assert decision.reason_codes == (DefenseReason.MODEL_UNAVAILABLE,)
    assert decision.fallback_used is True
    assert decision.failed_component_version == "unknown"


def test_rules_only_decline_uses_exact_unsaturated_edge_and_ignores_model_thresholds() -> None:
    policy = ActionPolicy.default()
    supplied = PolicyThresholds(challenge=0.99, decline=1.0)
    below_vector = vector(actor_count_1m=9.999)
    edge_vector = vector(actor_count_1m=10.0)
    below_rules = RuleEngine.default().evaluate(event(), below_vector)
    edge_rules = RuleEngine.default().evaluate(event(), edge_vector)
    assert below_rules.score < 0.90
    assert edge_rules.score == 0.90
    below = policy.choose(
        event(),
        below_rules,
        calibrated_score=None,
        thresholds=supplied,
        model_failure=DefenseReason.MODEL_UNAVAILABLE,
        failed_component_version="catboost-v1",
        vector=below_vector,
    )
    edge = policy.choose(
        event(),
        edge_rules,
        calibrated_score=None,
        thresholds=supplied,
        model_failure=DefenseReason.MODEL_UNAVAILABLE,
        failed_component_version="catboost-v1",
        vector=edge_vector,
    )
    assert below.action is Action.CHALLENGE
    assert edge.action is Action.DECLINE


def test_high_calibrated_score_cannot_influence_missing_threshold_fallback() -> None:
    rule_result, feature_vector = bound_clear()
    decision = ActionPolicy.default().choose(
        event(),
        rule_result,
        calibrated_score=0.99,
        thresholds=None,
        vector=feature_vector,
    )
    assert decision.action is Action.APPROVE
    assert decision.score == 1e-8
    assert decision.calibrated_score is None
    assert decision.failed_component_version == "unknown"


def test_available_clear_score_has_clear_approve_without_fallback() -> None:
    rule_result, feature_vector = bound_clear()
    decision = ActionPolicy.default().choose(
        event(),
        rule_result,
        calibrated_score=0.2,
        thresholds=PolicyThresholds(challenge=0.8, decline=0.95),
        vector=feature_vector,
    )
    assert decision.action is Action.APPROVE
    assert decision.reason_codes == ()
    assert decision.fallback_used is False
    assert decision.fallback_reason is None


@pytest.mark.parametrize("value", [-0.01, 1.01, math.nan, math.inf, -math.inf])
def test_policy_rejects_invalid_calibrated_scores(value: float) -> None:
    rule_result, feature_vector = bound_clear()
    with pytest.raises(ValueError, match="calibrated_score"):
        ActionPolicy.default().choose(
            event(),
            rule_result,
            calibrated_score=value,
            thresholds=PolicyThresholds(challenge=0.8, decline=0.95),
            vector=feature_vector,
        )


def test_policy_rejects_incoherent_model_state_and_wrong_event() -> None:
    policy = ActionPolicy.default()
    rule_result, feature_vector = bound_clear()
    with pytest.raises(ValueError, match="model failure"):
        policy.choose(
            event(),
            rule_result,
            calibrated_score=0.5,
            thresholds=PolicyThresholds(challenge=0.8, decline=0.95),
            model_failure=DefenseReason.MODEL_TIMEOUT,
            failed_component_version="catboost-v1",
            vector=feature_vector,
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
    rule_result, feature_vector = bound_clear()
    with pytest.raises(ValueError, match="calibrated_score"):
        ActionPolicy.default().choose(
            event(),
            rule_result,
            calibrated_score=float("nan"),
            thresholds=None,
            vector=feature_vector,
        )


@pytest.mark.parametrize(
    ("failure", "component"),
    (
        (DefenseReason.MODEL_UNAVAILABLE, None),
        (DefenseReason.MODEL_UNAVAILABLE, ""),
        (DefenseReason.MODEL_TIMEOUT, "   "),
    ),
)
def test_explicit_model_failure_requires_nonblank_component_identity(
    failure: DefenseReason,
    component: str | None,
) -> None:
    rule_result, feature_vector = bound_clear()
    with pytest.raises(ValueError, match="failed component"):
        ActionPolicy.default().choose(
            event(),
            rule_result,
            calibrated_score=None,
            thresholds=None,
            model_failure=failure,
            failed_component_version=component,
            vector=feature_vector,
        )


@pytest.mark.parametrize("component", [None, "", "   "])
def test_fallback_decision_contract_requires_nonblank_component_identity(
    component: str | None,
) -> None:
    with pytest.raises(ValidationError, match="failed component"):
        DefenseDecision(
            event_id="event-current",
            action=Action.APPROVE,
            score=0.0,
            rule_score=0.0,
            calibrated_score=None,
            reason_codes=(DefenseReason.MODEL_UNAVAILABLE,),
            evidence_source_ids=("event-current",),
            fallback_used=True,
            fallback_reason=DefenseReason.MODEL_UNAVAILABLE,
            failed_component_version=component,
            latency_ms=0.0,
            policy_version="1.0.0",
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
    rule_result, feature_vector = bound_clear()
    decision = policy.choose(
        event(),
        rule_result,
        calibrated_score=0.2,
        thresholds=PolicyThresholds(challenge=0.8, decline=0.95),
        vector=feature_vector,
    )
    assert budget == OperatingBudget(
        challenge_rate_max=0.02,
        false_decline_rate_max=0.001,
        review_case_rate_max=0.01,
    )
    assert policy.rule_manifest == RuleManifest.default()
    assert policy.model_dump(mode="json")["rule_manifest"] == RuleManifest.default().model_dump(
        mode="json"
    )
    assert json.dumps(decision.model_dump(mode="json"), sort_keys=True, allow_nan=False)
    with pytest.raises(ValidationError):
        policy.rule_manifest.actor_count_1m = 99  # type: ignore[misc]
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
