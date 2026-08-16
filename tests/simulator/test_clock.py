"""Behavioral coverage for the deterministic simulation clock."""

from collections.abc import Mapping
from dataclasses import FrozenInstanceError
from datetime import UTC, datetime, timedelta, timezone
from decimal import Decimal
from enum import Enum, StrEnum

import pytest

from apar.simulator.clock import Command, SimulationClock


@pytest.fixture
def now() -> datetime:
    return datetime(2026, 8, 16, 12, 0, tzinfo=UTC)


@pytest.fixture
def clock(now: datetime) -> SimulationClock:
    return SimulationClock(now)


def test_queue_orders_by_time_priority_then_sequence(clock: SimulationClock, now: datetime) -> None:
    """Catch heap ordering that ignores priority or insertion order ties."""
    clock.schedule(now, 2, Command("second"))
    clock.schedule(now, 1, Command("first"))
    clock.schedule(now, 1, Command("first-tie"))

    assert [clock.pop().command.name for _ in range(3)] == ["first", "first-tie", "second"]


def test_queue_orders_distinct_timestamps_before_priority(
    clock: SimulationClock, now: datetime
) -> None:
    """Catch priority incorrectly taking precedence over event time."""
    clock.schedule(now + timedelta(seconds=2), 0, Command("later-high"))
    clock.schedule(now + timedelta(seconds=1), 99, Command("earlier-low"))

    assert [clock.pop().command.name for _ in range(2)] == ["earlier-low", "later-high"]


def test_pop_advances_clock_to_popped_command_time(clock: SimulationClock, now: datetime) -> None:
    """Catch a clock that returns scheduled work without advancing its time."""
    due = now + timedelta(minutes=5)
    clock.schedule(due, 0, Command("advance"))

    assert clock.pop().at == due
    assert clock.now == due


def test_schedule_rejects_naive_timestamp(clock: SimulationClock) -> None:
    """Catch accepting an ambiguous local-time event."""
    with pytest.raises(ValueError, match="UTC"):
        clock.schedule(datetime(2026, 8, 16, 12, 0), 0, Command("naive"))


def test_schedule_rejects_non_utc_timestamp(clock: SimulationClock, now: datetime) -> None:
    """Catch accepting a non-UTC event timestamp."""
    non_utc = now.astimezone(timezone(timedelta(hours=5, minutes=30)))

    with pytest.raises(ValueError, match="UTC"):
        clock.schedule(non_utc, 0, Command("non-utc"))


def test_schedule_rejects_event_before_simulation_time(
    clock: SimulationClock, now: datetime
) -> None:
    """Catch time travel after simulation has begun."""
    clock.schedule(now + timedelta(seconds=1), 0, Command("first"))
    clock.pop()

    with pytest.raises(ValueError, match="earlier than the current simulation time"):
        clock.schedule(now, 0, Command("past"))


def test_pop_empty_queue_has_stable_error(clock: SimulationClock) -> None:
    """Catch leaking an implementation-dependent empty-heap error."""
    with pytest.raises(IndexError, match="simulation event queue is empty"):
        clock.pop()


def test_command_is_immutable(clock: SimulationClock, now: datetime) -> None:
    """Catch mutable commands that could change after scheduling."""
    command = Command("authorize", {"payment_id": "p1"})
    clock.schedule(now, 0, command)

    with pytest.raises(FrozenInstanceError):
        command.name = "settle"  # type: ignore[misc]
    with pytest.raises(TypeError):
        command.payload["payment_id"] = "p2"  # type: ignore[index]


def test_command_freezes_nested_payload_values() -> None:
    """Catch a frozen command retaining mutable nested payload state."""
    command = Command("authorize", {"payment": {"id": "p1"}})

    nested = command.payload["payment"]
    assert isinstance(nested, Mapping)
    with pytest.raises(TypeError):
        nested["id"] = "p2"  # type: ignore[index]


class PayloadKind(Enum):
    """A valid immutable payload scalar for command tests."""

    AUTHORIZE = "authorize"


class PayloadTextKind(StrEnum):
    """A valid text enum for command tests."""

    CAPTURE = "capture"


class MutableInt(int):
    """An int subclass that carries mutable state."""


class MutableStr(str):
    """A str subclass that carries mutable state."""


class MutableAttributeKind(Enum):
    """An enum whose members can carry mutable state."""

    REVIEW = "review"


class MutablePayload:
    """A deliberately unsupported mutable object for command tests."""

    value = "mutable"


@pytest.mark.parametrize("unsupported", [bytearray(b"p1"), MutablePayload()])
def test_command_rejects_unsupported_mutable_payload_values(unsupported: object) -> None:
    """Catch unsupported values being retained as mutable payload aliases."""
    with pytest.raises(TypeError, match="unsupported command payload value"):
        Command("authorize", {"payment": unsupported})


def test_command_freezes_supported_nested_payload_values(now: datetime) -> None:
    """Catch rejecting or mutating supported structured payload data."""
    command = Command(
        "authorize",
        {
            "amount": Decimal("10.00"),
            "created_at": now,
            "receipt": b"receipt",
            "kind": PayloadKind.AUTHORIZE,
            "text_kind": PayloadTextKind.CAPTURE,
            "nested": [{"payment_ids": {"p1", "p2"}}],
        },
    )

    assert command.payload["amount"] == Decimal("10.00")
    assert command.payload["created_at"] == now
    assert command.payload["receipt"] == b"receipt"
    assert command.payload["kind"] == "authorize"
    assert type(command.payload["kind"]) is str
    assert command.payload["text_kind"] == "capture"
    assert type(command.payload["text_kind"]) is str
    nested = command.payload["nested"]
    assert isinstance(nested, tuple)
    assert isinstance(nested[0], Mapping)
    assert nested[0]["payment_ids"] == frozenset({"p1", "p2"})


def test_command_rejects_mutable_scalar_subclasses() -> None:
    """Catch queued commands retaining mutable scalar-subclass identities."""
    number = MutableInt(7)
    number.state = []
    text = MutableStr("p1")
    text.state = []

    with pytest.raises(TypeError, match="unsupported command payload value: MutableInt"):
        Command("authorize", {"number": number})
    with pytest.raises(TypeError, match="unsupported command payload value: MutableStr"):
        Command("authorize", {"text": text})


def test_command_canonicalizes_enum_member_with_mutable_attribute() -> None:
    """Catch queued commands retaining enum identity or mutable member attributes."""
    kind = MutableAttributeKind.REVIEW
    kind.state = []
    command = Command("authorize", {"kind": kind})
    kind.state.append("changed")

    assert command.payload["kind"] == "review"
    assert type(command.payload["kind"]) is str
