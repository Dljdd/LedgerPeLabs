"""Deterministic, single-scenario discrete-event simulation engine."""

from __future__ import annotations

import heapq
import json
from collections.abc import Collection, Mapping
from datetime import datetime, timedelta
from decimal import Decimal
from types import MappingProxyType
from uuid import UUID

import numpy as np
from numpy.random import Generator

from apar.contracts.events import PaymentEvent, Rail
from apar.contracts.scenarios import ScenarioBundle
from apar.simulator.clock import Command, SimulationClock
from apar.simulator.ledger import AccountReference, Ledger, LedgerEntry
from apar.simulator.rails.base import RailAdapter


def _is_utc(value: datetime) -> bool:
    """Return whether a timestamp is explicitly UTC, rather than merely offset-zero."""
    return (
        value.tzinfo is not None
        and value.utcoffset() == timedelta(0)
        and value.tzname() == "UTC"
    )


def _validated_event(event: PaymentEvent) -> PaymentEvent:
    """Revalidate an event so unchecked Pydantic copies cannot enter a run."""
    if not isinstance(event, PaymentEvent):
        raise TypeError("rail adapters must emit PaymentEvent instances")
    return PaymentEvent.model_validate(
        event.model_dump(mode="python", round_trip=True, warnings=False)
    )


class LedgerView:
    """Read-only account and audit view exposed by the simulation engine."""

    __slots__ = ("__ledger",)

    def __init__(self, ledger: Ledger) -> None:
        self.__ledger = ledger

    @property
    def entries(self) -> tuple[LedgerEntry, ...]:
        """Return the immutable posting history."""
        return self.__ledger.entries

    def balance(self, account: str, currency: str = "USD") -> Decimal:
        """Return one current account balance."""
        return self.__ledger.balance(account, currency)

    def assert_conserved(self) -> None:
        """Assert value conservation without exposing the ledger's post method."""
        self.__ledger.assert_conserved()


class SimulationEngine:
    """Own simulation time, randomness, state, adapters, postings, and emitted events."""

    def __init__(
        self,
        bundle: ScenarioBundle,
        adapters: Mapping[Rail, RailAdapter],
        *,
        opening_balances: Mapping[AccountReference, Decimal] | None = None,
        allow_credit: Collection[AccountReference] | None = None,
    ) -> None:
        self._bundle = ScenarioBundle.model_validate(
            bundle.model_dump(mode="python", round_trip=True, warnings=False)
        )
        self._rng = np.random.default_rng(self._bundle.seed)
        self._clock = SimulationClock(self._bundle.replay_manifest.simulation_start)
        self._ledger = Ledger(opening_balances, allow_credit=allow_credit)
        self._ledger_view = LedgerView(self._ledger)
        self._adapters = MappingProxyType(dict(adapters))
        self._entity_state: dict[str, object] = {}
        self._events: list[PaymentEvent] = []
        self._seen_event_ids: set[str] = set()
        self._scheduled_times: list[datetime] = []
        self._initialized = False

    @property
    def bundle(self) -> ScenarioBundle:
        """Return the frozen scenario bundle driving this run."""
        return self._bundle

    @property
    def now(self) -> datetime:
        """Return current simulation time without exposing the event queue."""
        return self._clock.now

    @property
    def rng(self) -> Generator:
        """Return the engine-local generator seeded only from the scenario bundle."""
        return self._rng

    @property
    def ledger(self) -> LedgerView:
        """Return a read-only ledger view; adapters post through :meth:`post`."""
        return self._ledger_view

    @property
    def events(self) -> tuple[PaymentEvent, ...]:
        """Return an immutable snapshot of the append-only event stream."""
        return tuple(self._events)

    def schedule(self, at: datetime, priority: int, command: Command) -> None:
        """Request a future command without exposing queue internals."""
        self._clock.schedule(at, priority, command)
        heapq.heappush(self._scheduled_times, at)

    def post(self, entry: LedgerEntry) -> None:
        """Request one validated append-only double-entry ledger posting."""
        self._ledger.post(entry)

    def entity_state(self, entity_id: str) -> object:
        """Return engine-owned state for one entity or payment identifier."""
        return self._entity_state[entity_id]

    def set_entity_state(self, entity_id: str, state: object) -> None:
        """Set engine-owned state through a named public boundary."""
        if not entity_id:
            raise ValueError("entity_id must not be empty")
        self._entity_state[entity_id] = state

    def new_uuid(self) -> str:
        """Return a deterministic RFC 4122 UUID from the engine-local generator."""
        value = bytearray(self._rng.bytes(16))
        value[6] = (value[6] & 0x0F) | 0x40
        value[8] = (value[8] & 0x3F) | 0x80
        return str(UUID(bytes=bytes(value)))

    def emit(self, event: PaymentEvent) -> None:
        """Validate and append one event atomically."""
        self._append_batch([event])

    def run(self, until: datetime | None = None) -> tuple[PaymentEvent, ...]:
        """Run due commands through the scenario rail, optionally to a UTC cutoff."""
        if until is not None:
            if not _is_utc(until):
                raise ValueError("simulation cutoff must be a UTC timestamp")
            if until < self._clock.now:
                raise ValueError("simulation cutoff cannot precede current simulation time")

        adapter = self._adapters.get(self._bundle.rail)
        if adapter is None:
            raise ValueError(f"unsupported rail: {self._bundle.rail.value}")
        if not self._initialized:
            adapter.initialize(self)
            self._initialized = True

        while self._scheduled_times and (
            until is None or self._scheduled_times[0] <= until
        ):
            expected_at = heapq.heappop(self._scheduled_times)
            scheduled = self._clock.pop()
            if scheduled.at != expected_at:
                raise AssertionError("simulation clock and due-time index diverged")
            emitted = adapter.handle(scheduled.command, self)
            if not isinstance(emitted, list):
                raise TypeError("rail adapters must return a list of PaymentEvent instances")
            self._append_batch(emitted)

        return self.events

    def serialize_events(self) -> bytes:
        """Serialize emitted events to canonical, byte-stable JSON."""
        payload = [event.model_dump(mode="json") for event in self._events]
        return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()

    @staticmethod
    def replay_events(payload: bytes | str) -> tuple[PaymentEvent, ...]:
        """Revalidate a serialized event artifact and its stable stream invariants."""
        parsed: object = json.loads(payload)
        if not isinstance(parsed, list):
            raise ValueError("serialized events must contain a JSON array")

        events: list[PaymentEvent] = []
        seen: set[str] = set()
        last_time: datetime | None = None
        for raw_event in parsed:
            event = PaymentEvent.model_validate(raw_event)
            if event.event_id in seen:
                raise ValueError(f"duplicate event_id: {event.event_id}")
            if last_time is not None and event.event_time < last_time:
                raise ValueError("event_time must be monotonic")
            events.append(event)
            seen.add(event.event_id)
            last_time = event.event_time
        return tuple(events)

    def _append_batch(self, candidates: list[PaymentEvent]) -> None:
        """Validate a complete event batch before changing the append-only stream."""
        validated: list[PaymentEvent] = []
        batch_ids: set[str] = set()
        last_time = self._events[-1].event_time if self._events else None

        for candidate in candidates:
            event = _validated_event(candidate)
            if event.event_id in self._seen_event_ids or event.event_id in batch_ids:
                raise ValueError(f"duplicate event_id: {event.event_id}")
            if event.event_time < self._clock.now:
                raise ValueError("event_time cannot precede simulation time")
            if last_time is not None and event.event_time < last_time:
                raise ValueError("event_time must be monotonic")
            validated.append(event)
            batch_ids.add(event.event_id)
            last_time = event.event_time

        self._events.extend(validated)
        self._seen_event_ids.update(batch_ids)


__all__ = ["LedgerView", "SimulationEngine"]
