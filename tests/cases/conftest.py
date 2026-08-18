"""Hand-checkable fixtures for defender-visible case tests."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Any

from apar.contracts.decisions import Action
from apar.contracts.events import EventKind, Rail
from apar.defense.contracts import ObservedEvent
from apar.defense.policy import DefenseDecision

NOW = datetime(2026, 8, 18, 12, 0, tzinfo=UTC)


def observation(
    event_id: str,
    *,
    actor_id: str,
    counterparty_id: str,
    decision_at: datetime = NOW,
    available_at: datetime | None = None,
    event_time: datetime | None = None,
    amount: str = "100.00",
    is_decision_point: bool = True,
    **updates: Any,
) -> ObservedEvent:
    available = available_at or decision_at - timedelta(seconds=1)
    occurred = event_time or available - timedelta(seconds=1)
    values: dict[str, Any] = {
        "event_id": event_id,
        "payment_id": f"payment-{event_id}",
        "rail": Rail.CARD,
        "event_type": EventKind.AUTHORIZATION,
        "amount": Decimal(amount),
        "currency": "USD",
        "event_time": occurred,
        "available_at": available,
        "decision_at": decision_at,
        "actor_id": actor_id,
        "counterparty_id": counterparty_id,
        "optional_refs": {},
        "integrity_status": "not_applicable",
        "integrity_reason": None,
        "is_decision_point": is_decision_point,
    }
    values.update(updates)
    return ObservedEvent(**values)


def decision(
    event_id: str,
    *,
    action: Action = Action.CHALLENGE,
    score: float = 0.8,
) -> DefenseDecision:
    return DefenseDecision(
        event_id=event_id,
        action=action,
        score=score,
        rule_score=0.0,
        calibrated_score=score,
        reason_codes=(),
        evidence_source_ids=(event_id,),
        fallback_used=False,
        fallback_reason=None,
        failed_component_version=None,
        latency_ms=1.0,
        policy_version="1.0.0",
    )
