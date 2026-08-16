"""Deterministic, failure-atomic discrete-event simulation orchestration."""

from __future__ import annotations

import heapq
import json
from collections.abc import Collection, Mapping
from contextvars import ContextVar
from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum
from math import isfinite
from threading import Lock
from types import MappingProxyType
from typing import Any, cast
from uuid import UUID
from weakref import ReferenceType, ref

import numpy as np
from numpy.random import Generator
from pydantic_core import PydanticSerializationError

from apar.contracts.events import PaymentEvent, Rail
from apar.contracts.scenarios import ScenarioBundle
from apar.simulator.clock import Command, SimulationClock
from apar.simulator.ledger import AccountReference, Ledger, LedgerEntry
from apar.simulator.rails.base import (
    AdapterFactory,
    FrozenState,
    LedgerReader,
    RailAdapter,
    RailContext,
    RandomCapability,
)

_FAILED_MESSAGE = "simulation engine is terminally failed"
_MAPPING_PROXY_TYPE: type[object] = type(MappingProxyType({}))
_ACTIVE_ENGINE: ContextVar[SimulationEngine | None] = ContextVar(
    "apar_active_simulation_engine",
    default=None,
)
_LIVE_ADAPTERS: dict[int, ReferenceType[object]] = {}
_LIVE_ADAPTERS_LOCK = Lock()


class SimulationFailedError(RuntimeError):
    """Raised when an operation targets an irrecoverably failed simulation."""


def _active_engine() -> SimulationEngine:
    """Resolve callback-local dispatch without retaining an owner on the facade."""
    engine = _ACTIVE_ENGINE.get()
    if engine is None:
        raise RuntimeError("adapter context is inactive")
    return engine


def _release_adapter(adapter_id: int, reference: ReferenceType[object]) -> None:
    """Remove a dead adapter identity without racing a newer object reusing its ID."""
    with _LIVE_ADAPTERS_LOCK:
        if _LIVE_ADAPTERS.get(adapter_id) is reference:
            _LIVE_ADAPTERS.pop(adapter_id, None)


def _claim_fresh_adapter(adapter: object) -> RailAdapter:
    """Operationally reject factories lending one live adapter to multiple engines."""
    if isinstance(adapter, type) or not callable(getattr(adapter, "initialize", None)):
        raise TypeError("adapter factory must return a RailAdapter instance")
    if not callable(getattr(adapter, "handle", None)):
        raise TypeError("adapter factory must return a RailAdapter instance")
    adapter_id = id(adapter)

    def release(dead: ReferenceType[object]) -> None:
        _release_adapter(adapter_id, dead)

    try:
        reference = ref(adapter, release)
    except TypeError as error:
        raise TypeError("adapter factory must return a weak-referenceable adapter") from error
    with _LIVE_ADAPTERS_LOCK:
        existing = _LIVE_ADAPTERS.get(adapter_id)
        if existing is not None and existing() is adapter:
            raise ValueError("adapter factory must return a fresh adapter instance")
        _LIVE_ADAPTERS[adapter_id] = reference
    return cast(RailAdapter, adapter)


def _construct_selected_adapter(
    rail: Rail,
    factories: Mapping[Rail, AdapterFactory],
) -> RailAdapter | None:
    """Validate factory inputs but construct only the scenario-selected rail."""
    for candidate_factory in factories.values():
        if not callable(candidate_factory):
            raise TypeError("adapter factory must be callable")
    selected_factory = factories.get(rail)
    if selected_factory is None:
        return None
    return _claim_fresh_adapter(selected_factory())


def _validate_finite_tree(value: object, *, label: str) -> None:
    """Reject non-finite Python numeric semantics before JSON-mode coercion."""
    if isinstance(value, Enum):
        _validate_finite_tree(value.value, label=label)
        return
    if isinstance(value, float):
        if not isfinite(value):
            raise ValueError(f"{label} contains non-finite number")
        return
    if isinstance(value, Decimal):
        if not value.is_finite():
            raise ValueError(f"{label} contains non-finite number")
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            _validate_finite_tree(key, label=label)
            _validate_finite_tree(item, label=label)
        return
    if isinstance(value, (list, tuple, set, frozenset)):
        for item in value:
            _validate_finite_tree(item, label=label)


def _freeze_state(value: object) -> FrozenState:
    """Canonicalize the closed entity-state algebra without invoking user hooks."""
    if isinstance(value, Enum):
        return _freeze_state(value.value)
    if value is None:
        return None
    if type(value) is bool:
        return value
    if type(value) is int:
        return value
    if type(value) is float:
        number = value
        if not isfinite(number):
            raise ValueError("entity state numbers must be finite")
        return number
    if type(value) is str:
        return value
    if type(value) is bytes:
        return value
    if type(value) is Decimal:
        amount = value
        if not amount.is_finite():
            raise ValueError("entity state numbers must be finite")
        return amount
    if type(value) is datetime:
        timestamp = value
        if not _is_utc(timestamp):
            raise ValueError("entity state datetime must be a UTC timestamp")
        return timestamp
    if type(value) in (dict, _MAPPING_PROXY_TYPE):
        mapping = cast(Mapping[object, object], value)
        frozen: dict[str, FrozenState] = {}
        for key, item in mapping.items():
            if type(key) is not str:
                raise TypeError("entity state mapping keys must be exact strings")
            frozen[key] = _freeze_state(item)
        return MappingProxyType(frozen)
    if type(value) in (list, tuple):
        sequence = cast(list[object] | tuple[object, ...], value)
        return tuple(_freeze_state(item) for item in sequence)
    if type(value) in (set, frozenset):
        members = cast(set[object] | frozenset[object], value)
        try:
            return frozenset(_freeze_state(item) for item in members)
        except TypeError as error:
            raise TypeError("entity state set members must be hashable") from error
    raise TypeError(f"unsupported entity state: {type(value).__name__}")


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
    python_value = event.model_dump(mode="python", round_trip=True, warnings=False)
    _validate_finite_tree(python_value, label="event")
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
    python_value = bundle.model_dump(mode="python", round_trip=True, warnings=False)
    _validate_finite_tree(python_value, label="scenario bundle")
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

    __slots__ = ()

    def integers(
        self,
        low: int,
        high: int | None = None,
        size: None = None,
        *,
        endpoint: bool = False,
    ) -> np.integer[Any]:
        return _active_engine()._rng.integers(low, high, size, endpoint=endpoint)

    def uniform(
        self,
        low: float = 0.0,
        high: float = 1.0,
        size: int | tuple[int, ...] | None = None,
    ) -> object:
        return _active_engine()._rng.uniform(low, high, size)

    def random(self, size: int | tuple[int, ...] | None = None) -> object:
        return _active_engine()._rng.random(size)

    def bytes(self, length: int) -> bytes:
        return _active_engine()._rng.bytes(length)


class _RailContext:
    """Restricted facade passed to adapter callbacks instead of the owning engine."""

    __slots__ = ()

    @property
    def bundle(self) -> ScenarioBundle:
        return _active_engine().bundle

    @property
    def now(self) -> datetime:
        return _active_engine().now

    @property
    def rng(self) -> RandomCapability:
        return _RANDOM_CAPABILITY

    @property
    def ledger(self) -> LedgerReader:
        return _active_engine().ledger

    def schedule(self, at: datetime, priority: int, command: Command) -> None:
        _active_engine().schedule(at, priority, command)

    def post(self, entry: LedgerEntry) -> None:
        _active_engine().post(entry)

    def entity_state(self, entity_id: str) -> FrozenState:
        return _active_engine().entity_state(entity_id)

    def set_entity_state(self, entity_id: str, state: object) -> None:
        _active_engine().set_entity_state(entity_id, state)

    def new_uuid(self) -> str:
        return _active_engine().new_uuid()


_RANDOM_CAPABILITY: RandomCapability = _RandomCapability()
_RAIL_CONTEXT: RailContext = _RailContext()


class SimulationEngine:
    """Own deterministic simulation state and fail closed after execution errors."""

    def __init__(
        self,
        bundle: ScenarioBundle,
        adapter_factories: Mapping[Rail, AdapterFactory],
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
        self._adapter = _construct_selected_adapter(self._bundle.rail, adapter_factories)
        self._entity_state: dict[str, FrozenState] = {}
        self._events: list[PaymentEvent] = []
        self._seen_event_ids: set[str] = set()
        self._scheduled_times: list[datetime] = []
        self._initialized = False
        self._failure: Exception | None = None

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

    def entity_state(self, entity_id: str) -> FrozenState:
        """Return recursively immutable, engine-owned entity state."""
        self._ensure_healthy()
        return self._entity_state[entity_id]

    def set_entity_state(self, entity_id: str, state: object) -> None:
        """Canonicalize and store only the closed immutable state algebra."""
        self._ensure_healthy()
        try:
            if not entity_id:
                raise ValueError("entity_id must not be empty")
            self._entity_state[entity_id] = _freeze_state(state)
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
            adapter = self._adapter
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
        token = _ACTIVE_ENGINE.set(self)
        try:
            adapter.initialize(_RAIL_CONTEXT)
        finally:
            _ACTIVE_ENGINE.reset(token)

    def _invoke_handle(self, adapter: RailAdapter, command: Command) -> list[PaymentEvent]:
        token = _ACTIVE_ENGINE.set(self)
        try:
            return adapter.handle(command, _RAIL_CONTEXT)
        finally:
            _ACTIVE_ENGINE.reset(token)

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


__all__ = ["LedgerView", "SimulationEngine", "SimulationFailedError"]
