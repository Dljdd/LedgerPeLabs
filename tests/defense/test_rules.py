"""Behavior contract for the transparent, family-blind defense rules."""

from __future__ import annotations

import inspect
import json
import math
from datetime import UTC, datetime
from decimal import Decimal

import pytest
from pydantic import ValidationError

from apar.contracts.events import EventKind, Rail
from apar.defense.contracts import ObservedEvent
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
    integrity_reason: str | None = None,
    actor_id: str = "actor-a",
    counterparty_id: str = "counterparty-a",
    rail: Rail | None = None,
    event_type: EventKind = EventKind.AUTHORIZATION,
    decision_at: datetime | None = NOW,
    is_decision_point: bool = True,
) -> ObservedEvent:
    selected_rail = rail or (
        Rail.AGENTIC if integrity_status != "not_applicable" else Rail.CARD
    )
    return ObservedEvent(
        event_id="event-current",
        payment_id="payment-current",
        rail=selected_rail,
        event_type=event_type,
        amount=Decimal("100.00"),
        currency="USD",
        event_time=NOW,
        available_at=NOW,
        decision_at=decision_at,
        actor_id=actor_id,
        counterparty_id=counterparty_id,
        optional_refs={},
        integrity_status=integrity_status,
        integrity_reason=integrity_reason,
        is_decision_point=is_decision_point,
    )


def vector(**values: float) -> FeatureVector:
    return FeatureVector(
        event_id="event-current",
        decision_at=NOW,
        source_event_ids=("source-b", "source-a"),
        max_source_available_at=datetime(2026, 1, 1, 11, 59, tzinfo=UTC),
        catalog_digest="a" * 64,
        values=values,
    )


@pytest.fixture
def rule_engine() -> RuleEngine:
    return RuleEngine.default()


def test_rules_have_only_event_and_vector_inputs(rule_engine: RuleEngine) -> None:
    assert tuple(inspect.signature(rule_engine.evaluate).parameters) == ("event", "vector")
    forbidden = {"family", "truth", "label", "campaign", "scenario", "regime"}
    public_fields = set(RuleHit.model_fields) | set(RuleResult.model_fields)
    assert public_fields.isdisjoint(forbidden)


@pytest.mark.parametrize(
    ("feature_name", "below", "edge", "reason"),
    (
        ("actor_count_1m", 3.999, 4.0, DefenseReason.ACTOR_VELOCITY),
        ("actor_count_10m", 7.999, 8.0, DefenseReason.ACTOR_VELOCITY),
        ("graph_counterparty_fanin", 4.999, 5.0, DefenseReason.GRAPH_FAN_IN),
        ("graph_actor_fanout", 4.999, 5.0, DefenseReason.GRAPH_FAN_OUT),
        ("actor_amount_zscore_24h", 3.999, 4.0, DefenseReason.AMOUNT_DEVIATION),
        ("counterparty_amount_zscore_24h", 3.999, 4.0, DefenseReason.AMOUNT_DEVIATION),
        ("actor_amount_zscore_24h", -3.999, -4.0, DefenseReason.AMOUNT_DEVIATION),
        ("graph_shared_neighbor_count", 2.999, 3.0, DefenseReason.GRAPH_SHARED_NEIGHBOR),
        ("pair_prior_count", 3.999, 4.0, DefenseReason.COUNTERPARTY_VELOCITY),
    ),
)
def test_exact_rule_thresholds_and_below_edge_nonhits(
    rule_engine: RuleEngine,
    feature_name: str,
    below: float,
    edge: float,
    reason: DefenseReason,
) -> None:
    assert rule_engine.evaluate(event(), vector(**{feature_name: below})).hits == ()
    result = rule_engine.evaluate(event(), vector(**{feature_name: edge}))
    assert tuple(hit.reason for hit in result.hits) == (reason,)
    assert result.score == pytest.approx(0.60)


def test_degraded_state_has_exact_challenge_score(rule_engine: RuleEngine) -> None:
    below = rule_engine.evaluate(event(), vector(dq_degraded_state=0.999))
    edge = rule_engine.evaluate(event(), vector(dq_degraded_state=1.0))
    assert below.hits == ()
    assert edge.score == 0.60
    assert edge.hits[0].reason is DefenseReason.FEATURE_STATE_DEGRADED
    assert edge.hits[0].severity is RuleSeverity.CHALLENGE


def test_integrity_and_required_data_failures_are_mandatory(rule_engine: RuleEngine) -> None:
    integrity = rule_engine.evaluate(
        event(integrity_status="fail", integrity_reason="receipt_failed"), vector()
    )
    required = rule_engine.evaluate(event(actor_id=""), vector())
    assert integrity.hits[0].reason is DefenseReason.INTEGRITY_FAILURE
    assert required.hits[0].reason is DefenseReason.REQUIRED_DATA_MISSING
    assert integrity.hits[0].mandatory is required.hits[0].mandatory is True
    assert integrity.hits[0].severity is required.hits[0].severity is RuleSeverity.DECLINE
    assert integrity.score == required.score == 1.0


@pytest.mark.parametrize(
    ("rail", "event_type", "integrity_status", "expected"),
    (
        (Rail.CARD, EventKind.AUTHORIZATION, "not_applicable", ()),
        (Rail.CARD, EventKind.AUTHORIZATION_DECLINED, "not_applicable", ()),
        (Rail.A2A, EventKind.TRANSFER_INITIATED, "not_applicable", ()),
        (Rail.AGENTIC, EventKind.AUTHORIZATION, "pass", ()),
        (
            Rail.AGENTIC,
            EventKind.AUTHORIZATION,
            "fail",
            (DefenseReason.INTEGRITY_FAILURE,),
        ),
        (Rail.AGENTIC, EventKind.AUTHENTICATION_CHALLENGE, "pass", ()),
        (
            Rail.AGENTIC,
            EventKind.AUTHENTICATION_CHALLENGE,
            "fail",
            (DefenseReason.INTEGRITY_FAILURE,),
        ),
        (Rail.AGENTIC, EventKind.AUTHORIZATION_DECLINED, "pass", ()),
        (
            Rail.AGENTIC,
            EventKind.AUTHORIZATION_DECLINED,
            "fail",
            (DefenseReason.INTEGRITY_FAILURE,),
        ),
    ),
)
def test_valid_rail_event_integrity_combinations_have_exact_mandatory_reasons(
    rule_engine: RuleEngine,
    rail: Rail,
    event_type: EventKind,
    integrity_status: str,
    expected: tuple[DefenseReason, ...],
) -> None:
    result = rule_engine.evaluate(
        event(rail=rail, event_type=event_type, integrity_status=integrity_status),
        vector(),
    )
    assert tuple(hit.reason for hit in result.hits if hit.mandatory) == expected


@pytest.mark.parametrize(
    "observed",
    (
        event(
            rail=Rail.CARD,
            event_type=EventKind.TRANSFER_INITIATED,
            integrity_status="not_applicable",
        ),
        event(
            rail=Rail.A2A,
            event_type=EventKind.AUTHORIZATION,
            integrity_status="not_applicable",
        ),
        event(
            rail=Rail.AGENTIC,
            event_type=EventKind.TRANSFER_INITIATED,
            integrity_status="pass",
        ),
        event(rail=Rail.CARD, integrity_status="pass"),
        event(rail=Rail.CARD, integrity_status="fail"),
        event(rail=Rail.A2A, integrity_status="pass"),
        event(rail=Rail.AGENTIC, integrity_status="not_applicable"),
        event(is_decision_point=False),
    ),
)
def test_malformed_schema_combinations_emit_only_required_data_missing(
    rule_engine: RuleEngine,
    observed: ObservedEvent,
) -> None:
    result = rule_engine.evaluate(observed, vector())
    assert tuple(hit.reason for hit in result.hits if hit.mandatory) == (
        DefenseReason.REQUIRED_DATA_MISSING,
    )


def test_hits_have_stable_order_and_complete_evidence(rule_engine: RuleEngine) -> None:
    result = rule_engine.evaluate(
        event(integrity_status="fail", integrity_reason="receipt_failed"),
        vector(
            actor_count_1m=8.0,
            graph_actor_fanout=5.0,
            graph_counterparty_fanin=5.0,
        ),
    )
    assert tuple(hit.reason for hit in result.hits) == (
        DefenseReason.INTEGRITY_FAILURE,
        DefenseReason.ACTOR_VELOCITY,
        DefenseReason.GRAPH_FAN_IN,
        DefenseReason.GRAPH_FAN_OUT,
    )
    assert all(
        hit.evidence_source_ids == ("event-current", "source-a", "source-b")
        for hit in result.hits
    )
    assert all(hit.rule_version == result.manifest_version for hit in result.hits)


def test_tied_hits_are_ordered_by_reason_string(rule_engine: RuleEngine) -> None:
    result = rule_engine.evaluate(
        event(),
        vector(graph_actor_fanout=5.0, graph_counterparty_fanin=5.0),
    )
    assert tuple(hit.reason for hit in result.hits) == (
        DefenseReason.GRAPH_FAN_IN,
        DefenseReason.GRAPH_FAN_OUT,
    )


def test_rule_score_is_continuous_bounded_and_every_hit_contributes(
    rule_engine: RuleEngine,
) -> None:
    edge = rule_engine.evaluate(event(), vector(actor_count_1m=4.0))
    higher = rule_engine.evaluate(event(), vector(actor_count_1m=6.0))
    saturated = rule_engine.evaluate(event(), vector(actor_count_1m=80.0))
    combined = rule_engine.evaluate(
        event(), vector(actor_count_1m=6.0, graph_shared_neighbor_count=3.0)
    )
    assert 0.0 <= edge.score < higher.score < saturated.score == 1.0
    assert higher.score < combined.score < 1.0


def test_rule_manifest_is_stable_immutable_and_json_safe() -> None:
    first = RuleManifest.default()
    second = RuleManifest.default()
    assert first == second
    assert first.version == "1.0.0"
    assert json.dumps(first.model_dump(mode="json"), sort_keys=True, allow_nan=False)
    with pytest.raises(ValidationError):
        first.actor_count_1m = 99  # type: ignore[misc]
    with pytest.raises(ValidationError):
        RuleManifest(actor_count_1m=float("nan"))


@pytest.mark.parametrize("bad_score", [-0.01, 1.01, math.nan, math.inf, -math.inf])
def test_rule_hit_rejects_non_finite_or_out_of_range_scores(bad_score: float) -> None:
    with pytest.raises(ValidationError):
        RuleHit(
            reason=DefenseReason.ACTOR_VELOCITY,
            score=bad_score,
            severity=RuleSeverity.CHALLENGE,
            mandatory=False,
            evidence_source_ids=(),
            rule_version="1.0.0",
        )


def test_rule_result_rejects_noncanonical_hits_and_fabricated_score() -> None:
    high = RuleHit(
        reason=DefenseReason.ACTOR_VELOCITY,
        score=0.8,
        severity=RuleSeverity.CHALLENGE,
        mandatory=False,
        evidence_source_ids=(),
        rule_version="1.0.0",
    )
    low = RuleHit(
        reason=DefenseReason.GRAPH_FAN_IN,
        score=0.6,
        severity=RuleSeverity.CHALLENGE,
        mandatory=False,
        evidence_source_ids=(),
        rule_version="1.0.0",
    )
    bindings = {
        "event_id": "event-current",
        "decision_at": NOW,
        "catalog_digest": "a" * 64,
        "vector_digest": "b" * 64,
        "manifest_digest": "c" * 64,
    }
    with pytest.raises(ValidationError, match="canonical order"):
        RuleResult(
            hits=(low, high), score=0.8, manifest_version="1.0.0", **bindings
        )
    with pytest.raises(ValidationError, match="aggregate risk"):
        RuleResult(hits=(high,), score=0.7, manifest_version="1.0.0", **bindings)


def test_non_neutral_rule_result_requires_complete_provenance_bindings() -> None:
    hit = RuleHit(
        reason=DefenseReason.ACTOR_VELOCITY,
        score=0.6,
        severity=RuleSeverity.CHALLENGE,
        mandatory=False,
        evidence_source_ids=("event-current",),
        rule_version="1.0.0",
    )
    with pytest.raises(ValidationError, match="provenance binding"):
        RuleResult(hits=(hit,), score=0.6, manifest_version="1.0.0")


def test_rule_engine_binds_complete_vector_and_manifest_provenance(
    rule_engine: RuleEngine,
) -> None:
    result = rule_engine.evaluate(event(), vector(actor_count_1m=4.0))
    assert result.event_id == "event-current"
    assert result.decision_at == NOW
    assert result.catalog_digest == "a" * 64
    assert len(result.vector_digest or "") == 64
    assert len(result.manifest_digest) == 64


def test_neutral_clear_is_the_only_unbound_default_sentinel() -> None:
    clear = RuleResult.clear()
    assert clear.hits == ()
    assert clear.score == 0.0
    assert clear.event_id is None
    assert clear.decision_at is None
    assert clear.catalog_digest is None
    assert clear.vector_digest is None
    assert len(clear.manifest_digest) == 64


def test_rule_hit_rejects_noncanonical_evidence_and_mandatory_misclassification() -> None:
    common = {
        "score": 1.0,
        "severity": RuleSeverity.DECLINE,
        "mandatory": True,
        "rule_version": "1.0.0",
    }
    with pytest.raises(ValidationError, match="unique and sorted"):
        RuleHit(
            reason=DefenseReason.INTEGRITY_FAILURE,
            evidence_source_ids=("source-b", "source-a", "source-a"),
            **common,
        )
    with pytest.raises(ValidationError, match="mandatory"):
        RuleHit(
            reason=DefenseReason.ACTOR_VELOCITY,
            evidence_source_ids=(),
            **common,
        )


def test_rule_engine_rejects_mismatched_rows_and_nonfinite_rule_inputs(
    rule_engine: RuleEngine,
) -> None:
    mismatched = vector(actor_count_1m=4.0).model_copy(update={"event_id": "other-event"})
    with pytest.raises(ValueError, match="same event"):
        rule_engine.evaluate(event(), mismatched)
    with pytest.raises(ValueError, match="finite"):
        rule_engine.evaluate(event(), vector(actor_count_1m=float("nan")))
