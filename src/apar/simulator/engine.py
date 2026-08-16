"""Deterministic, failure-atomic discrete-event simulation orchestration."""

from __future__ import annotations

import heapq
import json
from collections.abc import Collection, Mapping
from copy import deepcopy
from datetime import datetime, timedelta
from decimal import Decimal
from types import MappingProxyType
from typing import Any
from uuid import UUID

import numpy as np
from numpy.random import Generator
from pydantic_core import PydanticSerializationError

from apar.contracts.events import PaymentEvent, Rail
from apar.contracts.scenarios import ScenarioBundle
from apar.simulator.clock import Command, SimulationClock
from apar.simulator.ledger import AccountReference, Ledger, LedgerEntry
from apar.simulator.rails.base import (
    LedgerReader,
    RailAdapter,
    RailContext,
    RandomCapability,
)

_FAILED_MESSAGE = "simulation engine is terminally failed"


class SimulationFailedError(RuntimeError):
    """Raised when an operation targets an irrecoverably failed simulation."""


def _is_utc(value: datetime) -> bool:
    """Return whether a timestamp is explicitly UTC, rather than merely offset-zero."""
    return (
        value.tzinfo is not None
        and value.utcoffset() == timedelta(0)
        and value.tzname() == "UTC"
    )


def _reject_json_constant(value: str) -> object:
    """Reject Python's non-standard NaN and Infinity JSON extensions."""
    raise ValueError(f"non-standard JSON constant: {value}")


def _strict_json_loads(payload: bytes | str) -> object:
    """Decode only standards-compliant JSON values."""
    return json.loads(payload, parse_constant=_reject_json_constant)


def _event_json_bytes(event: PaymentEvent) -> bytes:
    """Return strict canonical JSON for one event or reject opaque/non-finite data."""
    if not isinstance(event, PaymentEvent):
        raise TypeError("rail adapters must emit PaymentEvent instances")
    try:
        raw = event.model_dump(mode="json", round_trip=True, warnings=False)
        return json.dumps(
            raw,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    except (PydanticSerializationError, TypeError, ValueError) as error:
        raise ValueError("event must be strict JSON") from error


def _validated_event(event: PaymentEvent) -> PaymentEvent:
    """Deeply detach and strictly revalidate an event at an ownership boundary."""
    payload = _event_json_bytes(event)
    return PaymentEvent.model_validate(_strict_json_loads(payload))


def _bundle_json_bytes(bundle: ScenarioBundle) -> bytes:
    """Return strict canonical JSON for one scenario bundle."""
    if not isinstance(bundle, ScenarioBundle):
        raise TypeError("bundle must be a ScenarioBundle")
    try:
        raw = bundle.model_dump(mode="json", round_trip=True, warnings=False)
        return json.dumps(
            raw,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    except (PydanticSerializationError, TypeError, ValueError) as error:
        raise ValueError("scenario bundle must be strict JSON") from error


def _validated_bundle(bundle: ScenarioBundle) -> tuple[ScenarioBundle, bytes]:
    """Deeply detach and strictly revalidate the scenario input."""
    payload = _bundle_json_bytes(bundle)
    return ScenarioBundle.model_validate(_strict_json_loads(payload)), payload


def _event_key(event: PaymentEvent) -> tuple[datetime, str]:
    """Return the replay manifest's declared total-order key."""
    return event.event_time, event.event_id


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


class _RandomCapability:
    """Guard every random draw with the adapter callback lifetime."""

    __slots__ = ("__engine",)

    def __init__(self, engine: SimulationEngine) -> None:
        self.__engine = engine

    def integers(
        self,
        low: int,
        high: int | None = None,
        size: None = None,
        *,
        endpoint: bool = False,
    ) -> np.integer[Any]:
        self.__engine._require_adapter_callback()
        return self.__engine._rng.integers(low, high, size, endpoint=endpoint)

    def uniform(
        self,
        low: float = 0.0,
        high: float = 1.0,
        size: int | tuple[int, ...] | None = None,
    ) -> object:
        self.__engine._require_adapter_callback()
        return self.__engine._rng.uniform(low, high, size)

    def random(self, size: int | tuple[int, ...] | None = None) -> object:
        self.__engine._require_adapter_callback()
        return self.__engine._rng.random(size)

    def bytes(self, length: int) -> bytes:
        self.__engine._require_adapter_callback()
        return self.__engine._rng.bytes(length)


class _RailContext:
    """Restricted facade passed to adapter callbacks instead of the owning engine."""

    __slots__ = ("__engine", "__rng")

    def __init__(self, engine: SimulationEngine) -> None:
        self.__engine = engine
        self.__rng = _RandomCapability(engine)

    @property
    def bundle(self) -> ScenarioBundle:
        return self.__engine.bundle

    @property
    def now(self) -> datetime:
        return self.__engine.now

    @property
    def rng(self) -> RandomCapability:
        return self.__rng

    @property
    def ledger(self) -> LedgerReader:
        return self.__engine.ledger

    def schedule(self, at: datetime, priority: int, command: Command) -> None:
        self.__engine._require_adapter_callback()
        self.__engine.schedule(at, priority, command)

    def post(self, entry: LedgerEntry) -> None:
        self.__engine._require_adapter_callback()
        self.__engine.post(entry)

    def entity_state(self, entity_id: str) -> object:
        return self.__engine.entity_state(entity_id)

    def set_entity_state(self, entity_id: str, state: object) -> None:
        self.__engine._require_adapter_callback()
        self.__engine.set_entity_state(entity_id, state)

    def new_uuid(self) -> str:
        self.__engine._require_adapter_callback()
        return self.__engine.new_uuid()


class SimulationEngine:
    """Own deterministic simulation state and fail closed after execution errors."""

    def __init__(
        self,
        bundle: ScenarioBundle,
        adapters: Mapping[Rail, RailAdapter],
        *,
        opening_balances: Mapping[AccountReference, Decimal] | None = None,
        allow_credit: Collection[AccountReference] | None = None,
    ) -> None:
        self._bundle, self._bundle_payload = _validated_bundle(bundle)
        self._rng: Generator = np.random.default_rng(self._bundle.seed)
        uuid_prefix = bytearray(self._rng.bytes(10))
        uuid_prefix[6] = (uuid_prefix[6] & 0x0F) | 0x40
        uuid_prefix[8] = (uuid_prefix[8] & 0x3F) | 0x80
        self._uuid_prefix = bytes(uuid_prefix)
        self._uuid_sequence = 0
        self._clock = SimulationClock(self._bundle.replay_manifest.simulation_start)
        self._ledger = Ledger(opening_balances, allow_credit=allow_credit)
        self._ledger_view = LedgerView(self._ledger)
        self._adapters = MappingProxyType(
            {rail: deepcopy(adapter) for rail, adapter in adapters.items()}
        )
        self._entity_state: dict[str, object] = {}
        self._events: list[PaymentEvent] = []
        self._seen_event_ids: set[str] = set()
        self._scheduled_times: list[datetime] = []
        self._initialized = False
        self._callback_active = False
        self._failure: Exception | None = None
        self._context: RailContext = _RailContext(self)

    @property
    def bundle(self) -> ScenarioBundle:
        """Return a defensive, deeply detached scenario snapshot."""
        return ScenarioBundle.model_validate(_strict_json_loads(self._bundle_payload))

    @property
    def now(self) -> datetime:
        """Return current simulation time without exposing the event queue."""
        return self._clock.now

    @property
    def ledger(self) -> LedgerView:
        """Return a read-only ledger view; adapters post through their context."""
        return self._ledger_view

    @property
    def events(self) -> tuple[PaymentEvent, ...]:
        """Return defensive event snapshots unless the run has failed."""
        self._ensure_healthy()
        return tuple(_validated_event(event) for event in self._events)

    def schedule(self, at: datetime, priority: int, command: Command) -> None:
        """Request a future command without exposing queue internals."""
        self._ensure_healthy()
        try:
            self._clock.schedule(at, priority, command)
            heapq.heappush(self._scheduled_times, at)
        except Exception as error:
            self._mark_failed(error)
            raise

    def post(self, entry: LedgerEntry) -> None:
        """Request one validated append-only double-entry ledger posting."""
        self._ensure_healthy()
        try:
            self._ledger.post(entry)
        except Exception as error:
            self._mark_failed(error)
            raise

    def entity_state(self, entity_id: str) -> object:
        """Return a defensive copy of engine-owned entity state."""
        self._ensure_healthy()
        return deepcopy(self._entity_state[entity_id])

    def set_entity_state(self, entity_id: str, state: object) -> None:
        """Deeply detach and store engine-owned state."""
        self._ensure_healthy()
        try:
            if not entity_id:
                raise ValueError("entity_id must not be empty")
            self._entity_state[entity_id] = deepcopy(state)
        except Exception as error:
            self._mark_failed(error)
            raise

    def new_uuid(self) -> str:
        """Return a deterministic, monotonic RFC 4122 UUID from the scenario seed."""
        self._ensure_healthy()
        if self._uuid_sequence >= 1 << 48:
            error = OverflowError("deterministic UUID sequence exhausted")
            self._mark_failed(error)
            raise error
        suffix = self._uuid_sequence.to_bytes(6, "big")
        self._uuid_sequence += 1
        return str(UUID(bytes=self._uuid_prefix + suffix))

    def emit(self, event: PaymentEvent) -> None:
        """Validate and append one same-time event or fail the engine terminally."""
        self._ensure_healthy()
        try:
            self._append_batch([event])
        except Exception as error:
            self._mark_failed(error)
            raise

    def run(self, until: datetime | None = None) -> tuple[PaymentEvent, ...]:
        """Run due commands through the scenario rail, optionally to a UTC cutoff."""
        self._ensure_healthy()
        if until is not None:
            if not _is_utc(until):
                raise ValueError("simulation cutoff must be a UTC timestamp")
            if until < self._clock.now:
                raise ValueError("simulation cutoff cannot precede current simulation time")

        try:
            adapter = self._adapters.get(self._bundle.rail)
            if adapter is None:
                raise ValueError(f"unsupported rail: {self._bundle.rail.value}")
            if not self._initialized:
                self._invoke_initialize(adapter)
                self._initialized = True

            while self._scheduled_times and (
                until is None or self._scheduled_times[0] <= until
            ):
                expected_at = heapq.heappop(self._scheduled_times)
                scheduled = self._clock.pop()
                if scheduled.at != expected_at:
                    raise AssertionError("simulation clock and due-time index diverged")
                emitted = self._invoke_handle(adapter, scheduled.command)
                if not isinstance(emitted, list):
                    raise TypeError("rail adapters must return a list of PaymentEvent instances")
                self._append_batch(emitted)
        except Exception as error:
            self._mark_failed(error)
            raise

        return self.events

    def serialize_events(self) -> bytes:
        """Serialize emitted events to canonical, byte-stable strict JSON."""
        self._ensure_healthy()
        payload = [event.model_dump(mode="json", warnings=False) for event in self._events]
        return json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()

    @staticmethod
    def replay_events(payload: bytes | str) -> tuple[PaymentEvent, ...]:
        """Revalidate a serialized event artifact and its declared total order."""
        parsed = _strict_json_loads(payload)
        if not isinstance(parsed, list):
            raise ValueError("serialized events must contain a JSON array")

        events: list[PaymentEvent] = []
        seen: set[str] = set()
        last_key: tuple[datetime, str] | None = None
        for raw_event in parsed:
            candidate = PaymentEvent.model_validate(raw_event)
            event = _validated_event(candidate)
            if event.event_id in seen:
                raise ValueError(f"duplicate event_id: {event.event_id}")
            key = _event_key(event)
            if last_key is not None and key < last_key:
                if event.event_time < last_key[0]:
                    raise ValueError("event_time must be monotonic")
                raise ValueError("event ordering must be canonical")
            events.append(event)
            seen.add(event.event_id)
            last_key = key
        return tuple(_validated_event(event) for event in events)

    def _invoke_initialize(self, adapter: RailAdapter) -> None:
        self._callback_active = True
        try:
            adapter.initialize(self._context)
        finally:
            self._callback_active = False

    def _invoke_handle(self, adapter: RailAdapter, command: Command) -> list[PaymentEvent]:
        self._callback_active = True
        try:
            return adapter.handle(command, self._context)
        finally:
            self._callback_active = False

    def _require_adapter_callback(self) -> None:
        """Reject retained mutation/random capabilities outside their callback."""
        self._ensure_healthy()
        if not self._callback_active:
            error = RuntimeError("adapter context is inactive")
            self._mark_failed(error)
            raise error

    def _append_batch(self, candidates: list[PaymentEvent]) -> None:
        """Validate and canonically order a whole batch before append."""
        validated: list[PaymentEvent] = []
        batch_ids: set[str] = set()
        for candidate in candidates:
            event = _validated_event(candidate)
            if event.rail is not self._bundle.rail:
                raise ValueError("event rail must match scenario rail")
            if event.event_time != self._clock.now:
                raise ValueError("event_time must equal simulation time")
            if event.event_id in self._seen_event_ids or event.event_id in batch_ids:
                raise ValueError(f"duplicate event_id: {event.event_id}")
            validated.append(event)
            batch_ids.add(event.event_id)

        validated.sort(key=_event_key)
        last_key = _event_key(self._events[-1]) if self._events else None
        if validated and last_key is not None and _event_key(validated[0]) < last_key:
            raise ValueError("event ordering must be canonical")

        self._events.extend(validated)
        self._seen_event_ids.update(batch_ids)

    def _ensure_healthy(self) -> None:
        if self._failure is not None:
            raise SimulationFailedError(_FAILED_MESSAGE) from self._failure

    def _mark_failed(self, error: Exception) -> None:
        if self._failure is None:
            self._failure = error
        self._callback_active = False


__all__ = ["LedgerView", "SimulationEngine", "SimulationFailedError"]
