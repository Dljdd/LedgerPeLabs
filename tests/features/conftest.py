"""Deterministic observations for causal feature behavior tests."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

import pytest

from apar.contracts.events import EventKind, Rail
from apar.defense.contracts import ObservedEvent
from apar.features.catalog import FeatureCatalog, load_feature_catalog

BASE_TIME = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)


def observation(
    sequence: int,
    *,
    seconds: int,
    actor: str = "actor-a",
    counterparty: str = "counterparty-a",
    amount: str = "10",
    event_type: EventKind = EventKind.AUTHORIZATION,
    rail: Rail = Rail.CARD,
    decision: bool = True,
    availability_seconds: int | None = None,
    payment: int | None = None,
    optional_refs: dict[str, str] | None = None,
    integrity_status: str = "not_applicable",
) -> ObservedEvent:
    event_time = BASE_TIME + timedelta(seconds=seconds)
    available_at = BASE_TIME + timedelta(
        seconds=seconds if availability_seconds is None else availability_seconds
    )
    return ObservedEvent(
        event_id=f"event-{sequence:03d}",
        payment_id=f"payment-{sequence if payment is None else payment:03d}",
        rail=rail,
        event_type=event_type,
        amount=Decimal(amount),
        currency="USD",
        event_time=event_time,
        available_at=available_at,
        decision_at=available_at if decision else None,
        actor_id=actor,
        counterparty_id=counterparty,
        optional_refs={} if optional_refs is None else optional_refs,
        integrity_status=integrity_status,
        integrity_reason=None,
        is_decision_point=decision,
    )


@pytest.fixture
def feature_catalog() -> FeatureCatalog:
    return load_feature_catalog(Path("config/defense/feature-catalog.json"))


@pytest.fixture
def equal_time_observations() -> tuple[ObservedEvent, ...]:
    return (
        observation(1, seconds=0, actor="actor-a", counterparty="counterparty-a"),
        observation(2, seconds=0, actor="actor-a", counterparty="counterparty-b"),
    )


@pytest.fixture
def observed_stream() -> tuple[ObservedEvent, ...]:
    """A chronological stream with openings and observable lifecycle outcomes."""
    return (
        observation(1, seconds=0, amount="10", payment=1),
        observation(
            2,
            seconds=5,
            event_type=EventKind.AUTHORIZATION_DECLINED,
            decision=False,
            payment=1,
        ),
        observation(3, seconds=30, amount="20", counterparty="counterparty-b", payment=2),
        observation(
            4,
            seconds=35,
            event_type=EventKind.AUTHENTICATION_CHALLENGE,
            decision=False,
            payment=2,
        ),
        observation(5, seconds=60, actor="actor-b", amount="15", payment=3),
        observation(
            6,
            seconds=65,
            event_type=EventKind.REFUND,
            decision=False,
            actor="actor-b",
            payment=3,
        ),
        observation(7, seconds=90, amount="30", counterparty="counterparty-c", payment=4),
        observation(
            8,
            seconds=95,
            event_type=EventKind.TRANSFER_RETURNED,
            decision=False,
            payment=4,
        ),
        observation(9, seconds=120, amount="40", counterparty="counterparty-d", payment=5),
        observation(10, seconds=150, actor="actor-c", amount="50", payment=6),
        observation(11, seconds=180, amount="60", counterparty="counterparty-e", payment=7),
        observation(12, seconds=210, actor="actor-d", amount="70", payment=8),
    )
