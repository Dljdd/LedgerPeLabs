"""Contracts for data visible to the defense pipeline."""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import Literal

from pydantic import Field, model_validator

from apar.contracts._validation import ExternalContract
from apar.contracts.events import EventKind, PaymentEvent, Rail

_OPTIONAL_REFERENCE_NAMES = frozenset(
    {
        "merchant_id",
        "payee_id",
        "beneficiary_entity_id",
        "user_entity_id",
        "device_id",
        "institution_id",
        "agent_id",
    }
)


class ObservedEvent(ExternalContract):
    """A synthetic payment event with evaluator-only semantics removed."""

    schema_version: Literal["1.0.0"] = "1.0.0"
    event_id: str
    payment_id: str
    rail: Rail
    event_type: EventKind
    amount: Decimal
    currency: str
    event_time: datetime
    available_at: datetime
    decision_at: datetime | None
    actor_id: str
    counterparty_id: str
    optional_refs: dict[str, str] = Field(default_factory=dict)
    integrity_status: Literal["pass", "fail", "not_applicable"]
    integrity_reason: str | None = None
    is_decision_point: bool
    privacy_classification: Literal["synthetic"] = "synthetic"


class PolicyThresholds(ExternalContract):
    """Ordered policy thresholds shared by calibration and action selection."""

    challenge: float = Field(ge=0.0, le=1.0)
    decline: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def ordered(self) -> PolicyThresholds:
        if self.challenge > self.decline:
            raise ValueError("challenge threshold must not exceed decline threshold")
        return self


def scrub_event(event: PaymentEvent) -> ObservedEvent:
    """Project a payment event onto the closed defender-visible schema."""
    if type(event) is not PaymentEvent:
        raise TypeError("event must be an exact PaymentEvent")
    payment_id = event.rail_data.get("payment_id")
    if type(payment_id) is not str or not payment_id:
        raise ValueError("event rail_data must contain a non-empty payment_id")

    optional_refs = {
        name: value
        for name, value in event.party_refs.items()
        if name in _OPTIONAL_REFERENCE_NAMES and type(value) is str
    }
    integrity_status, integrity_reason = _integrity(event)
    return ObservedEvent(
        event_id=event.event_id,
        payment_id=payment_id,
        rail=event.rail,
        event_type=event.event_type,
        amount=event.amount,
        currency=event.currency,
        event_time=event.event_time,
        available_at=event.available_at,
        decision_at=event.decision_at,
        actor_id=event.actor_id,
        counterparty_id=event.counterparty_id,
        optional_refs=optional_refs,
        integrity_status=integrity_status,
        integrity_reason=integrity_reason,
        is_decision_point=_is_decision_point(event),
    )


def _integrity(event: PaymentEvent) -> tuple[Literal["pass", "fail", "not_applicable"], str | None]:
    if event.rail is not Rail.AGENTIC:
        return "not_applicable", None
    status = event.rail_data.get("integrity")
    reason = event.rail_data.get("reason_code")
    if status == "pass":
        return "pass", None
    if status == "fail":
        return "fail", reason if type(reason) is str and reason else "receipt_failed"
    return "fail", "missing_receipt_integrity"


def _is_decision_point(event: PaymentEvent) -> bool:
    if event.rail is Rail.CARD:
        return event.event_type in {EventKind.AUTHORIZATION, EventKind.AUTHORIZATION_DECLINED}
    if event.rail is Rail.A2A:
        return event.event_type is EventKind.TRANSFER_INITIATED
    return event.event_type in {
        EventKind.AUTHORIZATION,
        EventKind.AUTHENTICATION_CHALLENGE,
        EventKind.AUTHORIZATION_DECLINED,
    }
