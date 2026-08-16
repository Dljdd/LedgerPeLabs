"""A deterministic, UTC-only event clock for simulation commands."""

from __future__ import annotations

import heapq
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from enum import Enum, StrEnum
from types import MappingProxyType


def _immutable_mapping(values: Mapping[str, object]) -> Mapping[str, object]:
    """Copy a command payload so scheduled commands cannot be changed in place."""
    return _freeze_mapping(values)


def _freeze_mapping(values: Mapping[str, object]) -> Mapping[str, object]:
    """Copy and freeze a mapping with exact-string or canonicalized StrEnum keys."""
    frozen: dict[str, object] = {}
    for key, value in values.items():
        frozen[_freeze_key(key)] = _freeze(value)
    return MappingProxyType(frozen)


def _freeze_key(key: object) -> str:
    """Return a key detached from caller-owned string-like objects."""
    if type(key) is str:
        return key
    if isinstance(key, StrEnum):
        value = _freeze(key.value)
        if type(value) is str:
            return value
    raise TypeError("command payload mapping keys must be exact strings or StrEnum")


def _freeze(value: object) -> object:
    """Recursively freeze the explicit payload types supported by commands."""
    if isinstance(value, Enum):
        return _freeze(value.value)
    if type(value) is datetime:
        return normalize_utc_datetime(
            value,
            message="command payload datetime must be UTC",
        )
    if value is None or type(value) in (bool, bytes, Decimal, float, int, str):
        return value
    if isinstance(value, Mapping):
        return _freeze_mapping(value)
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, tuple):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, (frozenset, set)):
        raise TypeError("command payload contains unordered container")
    raise TypeError(f"unsupported command payload value: {type(value).__name__}")


def _is_utc(value: datetime) -> bool:
    """Return whether a timestamp is explicitly UTC, rather than merely offset-zero."""
    return (
        value.tzinfo is not None
        and value.utcoffset() == timedelta(0)
        and value.tzname() == "UTC"
    )


def normalize_utc_datetime(value: datetime, *, message: str) -> datetime:
    """Validate and detach one timestamp from caller-owned timezone identity."""
    if not _is_utc(value):
        raise ValueError(message)
    return datetime(
        value.year,
        value.month,
        value.day,
        value.hour,
        value.minute,
        value.second,
        value.microsecond,
        tzinfo=UTC,
        fold=value.fold,
    )


@dataclass(frozen=True, slots=True)
class Command:
    """An immutable, typed command envelope for a rail adapter to execute later."""

    name: str
    payload: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if type(self.name) is not str:
            raise TypeError("command name must be an exact string")
        if not self.name:
            raise ValueError("command name must not be empty")
        object.__setattr__(self, "payload", _immutable_mapping(self.payload))


@dataclass(frozen=True, slots=True)
class ScheduledCommand:
    """A command with its deterministic event ordering fields."""

    at: datetime
    priority: int
    sequence: int
    command: Command


class SimulationClock:
    """Own a stable priority queue ordered by time, priority, then insertion order."""

    def __init__(self, now: datetime) -> None:
        self._now = normalize_utc_datetime(
            now,
            message="simulation clock time must be a UTC timestamp",
        )
        self._sequence = 0
        self._queue: list[tuple[datetime, int, int, ScheduledCommand]] = []

    @property
    def now(self) -> datetime:
        """Return the current UTC simulation time."""
        return self._now

    def schedule(self, at: datetime, priority: int, command: Command) -> datetime:
        """Schedule one command and return its owned normalized UTC timestamp."""
        normalized_at = normalize_utc_datetime(
            at,
            message="scheduled time must be a UTC timestamp",
        )
        if normalized_at < self._now:
            raise ValueError("cannot schedule earlier than the current simulation time")
        if type(priority) is not int:
            raise TypeError("priority must be an exact integer")
        if not isinstance(command, Command):
            raise TypeError("command must be a Command")

        scheduled = ScheduledCommand(normalized_at, priority, self._sequence, command)
        heapq.heappush(
            self._queue,
            (normalized_at, priority, self._sequence, scheduled),
        )
        self._sequence += 1
        return normalized_at

    def pop(self) -> ScheduledCommand:
        """Pop the next command and advance the simulation time to its event time."""
        if not self._queue:
            raise IndexError("simulation event queue is empty")
        _, _, _, scheduled = heapq.heappop(self._queue)
        self._now = scheduled.at
        return scheduled
