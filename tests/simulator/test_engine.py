"""Behavioral coverage for deterministic simulation orchestration."""

from __future__ import annotations

import asyncio
import gc
import json
from collections.abc import Mapping
from contextvars import Context, copy_context
from datetime import UTC, datetime, timedelta, timezone, tzinfo
from decimal import Decimal
from enum import Enum
from functools import partial
from types import MappingProxyType
from typing import ClassVar
from uuid import UUID
from weakref import ReferenceType, ref

import numpy as np
import pytest

from apar.compiler.compiler import compile_scenario
from apar.contracts.events import EventKind, LifecycleState, PaymentEvent, Rail
from apar.contracts.scenarios import ReplayConfig, ScenarioBundle
from apar.simulator.clock import Command
from apar.simulator.engine import SimulationEngine
from apar.simulator.ledger import AccountReference, LedgerEntry
from apar.simulator.rails.base import AdapterFactory, RailAdapter, RailContext
from tests.factories import NOW, make_scenario_config, make_threat_card


def _bundle(*, seed: int = 260_816) -> ScenarioBundle:
    config = make_scenario_config()
    replay = ReplayConfig(
        random_seed=seed,
        simulation_start=config.replay.simulation_start,
        generator_version=config.replay.generator_version,
        event_ordering=config.replay.event_ordering,
    )
    return compile_scenario(
        make_threat_card(),
        make_scenario_config(seed=seed, replay=replay),
    )


def _event(
    *,
    event_id: str = "00000000-0000-4000-8000-000000000001",
    event_time: datetime = NOW,
    amount: Decimal = Decimal("10.00"),
    rail: Rail = Rail.A2A,
    rail_data: dict[str, str | int | float | bool] | None = None,
    lineage: dict[str, str | bool] | None = None,
    extensions: dict[str, object] | None = None,
) -> PaymentEvent:
    return PaymentEvent(
        schema_version="1.0.0",
        event_id=event_id,
        campaign_id="00000000-0000-4000-8000-000000000002",
        trace_id="00000000-0000-4000-8000-000000000003",
        rail=rail,
        viewpoint="network_with_bank_enrichment",
        event_type=EventKind.TRANSFER_INITIATED,
        amount=amount,
        currency="USD",
        event_time=event_time,
        ingested_at=event_time + timedelta(milliseconds=25),
        available_at=event_time + timedelta(milliseconds=25),
        decision_at=event_time + timedelta(milliseconds=40),
        actor_id="00000000-0000-4000-8000-000000000004",
        counterparty_id="00000000-0000-4000-8000-000000000005",
        rail_data=rail_data or {},
        lineage=lineage or {},
        extensions=extensions or {},
    )


class InitializingAdapter:
    """Schedule one deterministic event from the adapter initialization boundary."""

    def initialize(self, engine: RailContext) -> None:
        engine.schedule(NOW, 0, Command("emit"))

    def handle(self, command: Command, engine: RailContext) -> list[PaymentEvent]:
        assert command.name == "emit"
        return [
            _event(
                event_id=engine.new_uuid(),
                amount=Decimal(int(engine.rng.integers(1, 1000))) / 100,
            )
        ]


class DuplicateBatchAdapter:
    """Return a duplicate batch so event append atomicity can be observed."""

    def initialize(self, engine: RailContext) -> None:
        engine.schedule(NOW, 0, Command("duplicate"))

    def handle(self, command: Command, engine: RailContext) -> list[PaymentEvent]:
        event = _event()
        return [event, event]


class PastSchedulingAdapter:
    """Try to schedule work behind the simulation clock while handling a command."""

    def initialize(self, engine: RailContext) -> None:
        engine.schedule(NOW + timedelta(seconds=1), 0, Command("time-travel"))

    def handle(self, command: Command, engine: RailContext) -> list[PaymentEvent]:
        engine.schedule(NOW, 0, Command("past"))
        return []


class PastEventAdapter:
    """Return an event whose event time predates the command being handled."""

    def initialize(self, engine: RailContext) -> None:
        engine.schedule(NOW + timedelta(seconds=1), 0, Command("late"))

    def handle(self, command: Command, engine: RailContext) -> list[PaymentEvent]:
        return [_event(event_time=NOW)]


class TwoEventAdapter:
    """Schedule two events to make cutoff and resumed-drain behavior visible."""

    def initialize(self, engine: RailContext) -> None:
        engine.schedule(NOW + timedelta(seconds=1), 0, Command("first"))
        engine.schedule(NOW + timedelta(seconds=2), 0, Command("second"))

    def handle(self, command: Command, engine: RailContext) -> list[PaymentEvent]:
        suffix = 11 if command.name == "first" else 12
        return [
            _event(
                event_id=f"00000000-0000-4000-8000-{suffix:012d}",
                event_time=engine.now,
            )
        ]


class PostingAdapter:
    """Exercise adapter access only through the engine's public mutation methods."""

    def initialize(self, engine: RailContext) -> None:
        engine.set_entity_state("payment", "created")
        engine.schedule(NOW, 0, Command("post"))

    def handle(self, command: Command, engine: RailContext) -> list[PaymentEvent]:
        assert engine.entity_state("payment") == "created"
        engine.post(
            LedgerEntry(
                "posting",
                debit={"payer": Decimal("2.00")},
                credit={"payee": Decimal("2.00")},
            )
        )
        engine.set_entity_state("payment", "posted")
        return [_event()]


class PartialInitializeFailureAdapter:
    """Schedule a command before initialization raises its original error."""

    def initialize(self, engine: RailContext) -> None:
        engine.schedule(NOW, 0, Command("must-never-run"))
        raise RuntimeError("initialize exploded")

    def handle(self, command: Command, engine: RailContext) -> list[PaymentEvent]:
        raise AssertionError("partially initialized work was retried")


class OneShotHandleFailureAdapter:
    """Raise after a command is popped so retry cannot look like successful completion."""

    def initialize(self, engine: RailContext) -> None:
        engine.schedule(NOW, 0, Command("one-shot"))

    def handle(self, command: Command, engine: RailContext) -> list[PaymentEvent]:
        raise RuntimeError("handle exploded")


class SideEffectThenInvalidBatchAdapter:
    """Mutate every adapter-visible subsystem before returning an invalid batch."""

    def initialize(self, engine: RailContext) -> None:
        engine.schedule(NOW, 0, Command("invalid-batch"))

    def handle(self, command: Command, engine: RailContext) -> list[PaymentEvent]:
        engine.post(
            LedgerEntry(
                "partial-post",
                debit={"payer": Decimal("1.00")},
                credit={"payee": Decimal("1.00")},
            )
        )
        engine.set_entity_state("payment", {"status": ["partially-mutated"]})
        engine.schedule(NOW + timedelta(seconds=1), 0, Command("must-never-resume"))
        duplicate = _event()
        return [duplicate, duplicate]


class FutureEventAdapter:
    """Emit future event time from a command at the simulation start."""

    def initialize(self, engine: RailContext) -> None:
        engine.schedule(NOW, 0, Command("future-event"))

    def handle(self, command: Command, engine: RailContext) -> list[PaymentEvent]:
        return [_event(event_time=NOW + timedelta(seconds=10))]


class UnsortedBatchAdapter:
    """Return equal-time events in reverse canonical event-ID order."""

    def initialize(self, engine: RailContext) -> None:
        engine.schedule(NOW, 0, Command("unsorted"))

    def handle(self, command: Command, engine: RailContext) -> list[PaymentEvent]:
        return [
            _event(event_id="00000000-0000-4000-8000-000000000020"),
            _event(event_id="00000000-0000-4000-8000-000000000010"),
        ]


class CrossBatchOrderAdapter:
    """Return a lower equal-time ID only after a higher ID was appended."""

    def initialize(self, engine: RailContext) -> None:
        engine.schedule(NOW, 0, Command("high"))
        engine.schedule(NOW, 1, Command("low"))

    def handle(self, command: Command, engine: RailContext) -> list[PaymentEvent]:
        suffix = 20 if command.name == "high" else 10
        return [_event(event_id=f"00000000-0000-4000-8000-{suffix:012d}")]


class DeterministicIdAdapter:
    """Issue three IDs across equal-time batches to prove monotonic generation."""

    def initialize(self, engine: RailContext) -> None:
        for priority in range(3):
            engine.schedule(NOW, priority, Command(f"event-{priority}"))

    def handle(self, command: Command, engine: RailContext) -> list[PaymentEvent]:
        return [_event(event_id=engine.new_uuid())]


class ContextBoundaryAdapter:
    """Verify callbacks receive a capability context rather than engine internals."""

    def initialize(self, engine: RailContext) -> None:
        forbidden = ("_clock", "_ledger", "_events", "_scheduled_times", "artifact_store")
        assert all(not hasattr(engine, name) for name in forbidden)
        assert engine.bundle.rail is Rail.A2A
        engine.set_entity_state("context", {"phase": ["initialized"]})
        engine.schedule(NOW, 0, Command("context"))

    def handle(self, command: Command, engine: RailContext) -> list[PaymentEvent]:
        assert engine.entity_state("context") == MappingProxyType(
            {"phase": ("initialized",)}
        )
        assert engine.ledger.balance("payer") == Decimal("5.00")
        amount = Decimal(int(engine.rng.integers(1, 2)))
        return [_event(event_id=engine.new_uuid(), amount=amount)]


class StatefulAdapter:
    """Expose adapter-instance reuse through an output-affecting callback counter."""

    def __init__(self) -> None:
        self.handled = 0

    def initialize(self, engine: RailContext) -> None:
        engine.schedule(NOW, 0, Command("stateful"))

    def handle(self, command: Command, engine: RailContext) -> list[PaymentEvent]:
        self.handled += 1
        return [_event(event_id=engine.new_uuid(), amount=Decimal(self.handled))]


class RetainingEventAdapter:
    """Keep the source event so caller mutation can challenge engine ownership."""

    def __init__(self) -> None:
        self.source = _event(
            rail_data={"risk": 3.5},
            lineage={"generated": True},
            extensions={"nested": {"values": ["original"]}},
        )

    def initialize(self, engine: RailContext) -> None:
        engine.schedule(NOW, 0, Command("emit-retained"))
        engine.schedule(NOW + timedelta(seconds=1), 0, Command("mutate-retained"))

    def handle(self, command: Command, engine: RailContext) -> list[PaymentEvent]:
        if command.name == "emit-retained":
            return [self.source]
        self.source.extensions["nested"]["values"].append("adapter")  # type: ignore[index]
        return []


class AliasState:
    """Hostile state whose deepcopy deliberately preserves mutable identity."""

    def __init__(self) -> None:
        self.values = ["original"]
        self.deepcopy_called = False

    def __deepcopy__(self, memo: dict[int, object]) -> AliasState:
        self.deepcopy_called = True
        return self


class MutableStateInt(int):
    """Scalar subclass with caller-owned mutable attributes."""


class MutableStateStr(str):
    """String subclass used as a mutable mapping key."""


class MutableFloat(float):
    """Float subclass that must not enter open contract fields."""


class LyingDecimal(Decimal):
    """Decimal subclass that lies about a non-finite underlying value."""

    def is_finite(self) -> bool:
        return True


class MutableUtcTz(tzinfo):
    """Caller-owned UTC tzinfo whose semantics can change after admission."""

    def __init__(self) -> None:
        self.offset = timedelta(0)
        self.name = "UTC"

    def utcoffset(self, value: datetime | None) -> timedelta:
        return self.offset

    def dst(self, value: datetime | None) -> timedelta:
        return timedelta(0)

    def tzname(self, value: datetime | None) -> str:
        return self.name


class MutableScheduledTimeAdapter:
    """Mutate a scheduled timestamp's tzinfo after handing it to the engine."""

    def __init__(self) -> None:
        self.mutable_utc = MutableUtcTz()

    def initialize(self, engine: RailContext) -> None:
        due = datetime(
            2026,
            8,
            16,
            12,
            0,
            1,
            123456,
            tzinfo=self.mutable_utc,
            fold=1,
        )
        engine.schedule(due, 0, Command("mutable-time"))
        self.mutable_utc.offset = timedelta(hours=5, minutes=30)
        self.mutable_utc.name = "IST"

    def handle(self, command: Command, engine: RailContext) -> list[PaymentEvent]:
        assert engine.now.tzinfo is UTC
        return [_event(event_id=engine.new_uuid(), event_time=engine.now)]


class StatePhase(Enum):
    """Enum canonicalization case for the closed state algebra."""

    READY = "ready"


class RefusesDeepcopyAdapter(StatefulAdapter):
    """A valid stateful adapter that cannot be borrowed through deepcopy."""

    def __deepcopy__(self, memo: dict[int, object]) -> RefusesDeepcopyAdapter:
        raise AssertionError("adapter must be factory-owned, not deepcopied")


class ConfiguredAdapter(StatefulAdapter):
    """Adapter constructed by a configurable partial factory."""

    def __init__(self, amount: int) -> None:
        super().__init__()
        self.amount = amount

    def handle(self, command: Command, engine: RailContext) -> list[PaymentEvent]:
        self.handled += 1
        return [_event(event_id=engine.new_uuid(), amount=Decimal(self.amount))]


class CaptureContextAdapter:
    """Capture callback capabilities so their retained shape can be probed."""

    captured_context: ClassVar[RailContext | None] = None
    captured_rng: ClassVar[object | None] = None

    def initialize(self, engine: RailContext) -> None:
        type(self).captured_context = engine
        type(self).captured_rng = engine.rng
        engine.schedule(NOW, 0, Command("capture"))

    def handle(self, command: Command, engine: RailContext) -> list[PaymentEvent]:
        return [_event(event_id=engine.new_uuid())]


class CaptureContextThenFailAdapter:
    """Capture the context before raising to prove exception cleanup."""

    captured_context: ClassVar[RailContext | None] = None

    def initialize(self, engine: RailContext) -> None:
        type(self).captured_context = engine
        raise RuntimeError("capture initialize failed")

    def handle(self, command: Command, engine: RailContext) -> list[PaymentEvent]:
        return []


class CopyContextRetentionAdapter:
    """Copy an active execution context so callback authority can be challenged."""

    captured_execution_context: ClassVar[Context | None] = None
    captured_rail_context: ClassVar[RailContext | None] = None

    def initialize(self, engine: RailContext) -> None:
        type(self).captured_execution_context = copy_context()
        type(self).captured_rail_context = engine

    def handle(self, command: Command, engine: RailContext) -> list[PaymentEvent]:
        return []


class AsyncTaskRetentionAdapter:
    """Create a child task while callback authority is dynamically active."""

    gate: ClassVar[asyncio.Event | None] = None
    task: ClassVar[asyncio.Task[Exception | None] | None] = None

    def initialize(self, engine: RailContext) -> None:
        engine.schedule(NOW, 0, Command("spawn-retained-task"))

    def handle(self, command: Command, engine: RailContext) -> list[PaymentEvent]:
        gate = asyncio.Event()
        type(self).gate = gate

        async def mutate_after_callback() -> Exception | None:
            await gate.wait()
            try:
                engine.set_entity_state("late", "mutation")
            except Exception as error:
                return error
            return None

        type(self).task = asyncio.create_task(mutate_after_callback())
        return []


class RunNestedEngineAdapter:
    """Run a nested engine and then reuse the restored outer callback context."""

    def __init__(self, nested: SimulationEngine) -> None:
        self.nested = nested

    def initialize(self, engine: RailContext) -> None:
        engine.schedule(NOW, 0, Command("nested"))

    def handle(self, command: Command, engine: RailContext) -> list[PaymentEvent]:
        assert len(self.nested.run()) == 1
        engine.set_entity_state("outer", {"callback": ["restored"]})
        return [_event(event_id=engine.new_uuid())]


def _engine(
    adapter_factory: AdapterFactory,
    *,
    seed: int = 260_816,
    opening_balances: Mapping[AccountReference, Decimal] | None = None,
) -> SimulationEngine:
    return SimulationEngine(
        _bundle(seed=seed),
        {Rail.A2A: adapter_factory},
        opening_balances=opening_balances,
    )


def _event_bytes(events: tuple[PaymentEvent, ...]) -> bytes:
    payload = [event.model_dump(mode="json") for event in events]
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()


def _assert_terminal_failure(engine: SimulationEngine) -> None:
    """Assert every run/artifact surface rejects a terminally failed engine."""
    with pytest.raises(RuntimeError, match="simulation engine is terminally failed"):
        engine.run()
    with pytest.raises(RuntimeError, match="simulation engine is terminally failed"):
        _ = engine.events
    with pytest.raises(RuntimeError, match="simulation engine is terminally failed"):
        engine.serialize_events()


def test_same_seed_produces_byte_identical_events() -> None:
    """Catch engine IDs or sampling depending on global or process-local state."""
    first = _engine(InitializingAdapter, seed=260_816).run()
    second = _engine(InitializingAdapter, seed=260_816).run()

    assert _event_bytes(first) == _event_bytes(second)


def test_engine_randomness_does_not_advance_numpy_global_state() -> None:
    """Catch an adapter accidentally receiving NumPy's global random stream."""
    np.random.seed(91)
    expected = int(np.random.randint(0, 1_000_000))
    np.random.seed(91)

    _engine(InitializingAdapter).run()

    assert int(np.random.randint(0, 1_000_000)) == expected


def test_duplicate_event_id_is_rejected_atomically() -> None:
    """Catch a duplicate event changing the append-only emitted-event snapshot."""
    engine = _engine(InitializingAdapter)
    event = _event()
    engine.emit(event)

    with pytest.raises(ValueError, match="duplicate event_id"):
        engine.emit(event)

    _assert_terminal_failure(engine)


def test_duplicate_event_inside_adapter_batch_rejects_the_whole_batch() -> None:
    """Catch validating event IDs after a prefix of the batch was already appended."""
    engine = _engine(DuplicateBatchAdapter)

    with pytest.raises(ValueError, match="duplicate event_id"):
        engine.run()

    _assert_terminal_failure(engine)


def test_adapter_cannot_schedule_behind_current_queue_time() -> None:
    """Catch adapter callbacks reintroducing already elapsed simulation work."""
    with pytest.raises(ValueError, match="earlier than the current simulation time"):
        _engine(PastSchedulingAdapter).run()


def test_event_time_cannot_precede_the_command_that_emits_it() -> None:
    """Catch an emitted stream moving backwards relative to the simulation queue."""
    engine = _engine(PastEventAdapter)

    with pytest.raises(ValueError, match="event_time must equal simulation time"):
        engine.run()

    _assert_terminal_failure(engine)


def test_run_rejects_an_unsupported_scenario_rail() -> None:
    """Catch silently dropping a scenario when no adapter implements its rail."""
    engine = SimulationEngine(_bundle(), {})

    with pytest.raises(ValueError, match="unsupported rail: a2a"):
        engine.run()


@pytest.mark.parametrize(
    "cutoff",
    [
        datetime(2026, 8, 16, 12, 0),
        NOW.astimezone(timezone(timedelta(hours=5, minutes=30))),
    ],
)
def test_run_rejects_a_cutoff_that_is_not_explicit_utc(cutoff: datetime) -> None:
    """Catch ambiguous or merely timezone-aware replay boundaries."""
    engine = _engine(TwoEventAdapter)

    with pytest.raises(ValueError, match="cutoff must be a UTC timestamp"):
        engine.run(until=cutoff)


def test_run_until_stops_before_future_work_and_can_resume() -> None:
    """Catch a bounded run draining or losing commands beyond its UTC cutoff."""
    engine = _engine(TwoEventAdapter)

    first = engine.run(until=NOW + timedelta(seconds=1))
    all_events = engine.run()

    assert [event.event_time for event in first] == [NOW + timedelta(seconds=1)]
    assert [event.event_time for event in all_events] == [
        NOW + timedelta(seconds=1),
        NOW + timedelta(seconds=2),
    ]


def test_event_serialization_replays_the_validated_event_tuple() -> None:
    """Catch serialization changing decimal, timestamp, enum, or field values."""
    engine = _engine(InitializingAdapter)
    expected = engine.run()

    payload = engine.serialize_events()
    replayed = SimulationEngine.replay_events(payload)

    assert payload == _event_bytes(expected)
    assert replayed == expected


def test_adapter_uses_public_engine_methods_without_heap_or_store_access() -> None:
    """Catch exposing mutable clock/store internals instead of the adapter facade."""
    engine = _engine(PostingAdapter, opening_balances={"payer": Decimal("5.00")})

    events = engine.run()

    assert len(events) == 1
    assert engine.entity_state("payment") == "posted"
    assert engine.ledger.balance("payer") == Decimal("3.00")
    assert engine.ledger.balance("payee") == Decimal("2.00")
    assert not hasattr(engine, "clock")
    assert not hasattr(engine, "artifact_store")
    with pytest.raises(AttributeError):
        _ = engine.ledger.post  # type: ignore[attr-defined]


def test_rail_adapter_protocol_accepts_the_declared_boundary() -> None:
    """Catch the public adapter interface drifting from initialize and handle."""
    adapter: RailAdapter = InitializingAdapter()
    factory: AdapterFactory = InitializingAdapter
    assert adapter
    engine = _engine(factory)

    assert len(engine.run()) == 1


def test_replay_rejects_duplicate_or_non_monotonic_event_streams() -> None:
    """Catch replay accepting artifacts the live engine would refuse to emit."""
    later = _event(event_id="00000000-0000-4000-8000-000000000010", event_time=NOW)
    earlier = _event(
        event_id="00000000-0000-4000-8000-000000000011",
        event_time=NOW - timedelta(seconds=1),
    )
    def encode(events: list[PaymentEvent]) -> bytes:
        return json.dumps(
            [event.model_dump(mode="json") for event in events],
            sort_keys=True,
            separators=(",", ":"),
        ).encode()

    with pytest.raises(ValueError, match="duplicate event_id"):
        SimulationEngine.replay_events(encode([later, later]))
    with pytest.raises(ValueError, match="event_time must be monotonic"):
        SimulationEngine.replay_events(encode([later, earlier]))


def test_returned_events_and_adapter_source_cannot_mutate_engine_artifact() -> None:
    """Catch internal event dictionaries escaping through adapter or return aliases."""
    engine = _engine(RetainingEventAdapter)
    returned = engine.run()[0]
    baseline = engine.serialize_events()

    returned.rail_data["risk"] = 99.0
    returned.lineage["generated"] = False
    returned.extensions["nested"]["values"].append("returned")  # type: ignore[index]

    assert engine.serialize_events() == baseline
    fresh = engine.events[0]
    assert fresh.rail_data == {"risk": 3.5}
    assert fresh.lineage == {"generated": True}
    assert fresh.extensions == {"nested": {"values": ["original"]}}


def test_partial_initialize_failure_is_terminal_and_preserves_original_error() -> None:
    """Catch retrying an adapter after initialization scheduled partial work."""
    engine = _engine(PartialInitializeFailureAdapter)

    with pytest.raises(RuntimeError, match="initialize exploded"):
        engine.run()

    _assert_terminal_failure(engine)


def test_popped_handle_failure_cannot_be_retried_as_a_successful_run() -> None:
    """Catch a lost popped command making the next run appear successfully empty."""
    engine = _engine(OneShotHandleFailureAdapter)

    with pytest.raises(RuntimeError, match="handle exploded"):
        engine.run()

    _assert_terminal_failure(engine)


def test_side_effects_followed_by_invalid_batch_make_engine_terminal() -> None:
    """Catch finalizing a run after ledger, state, and queue effects partially applied."""
    engine = _engine(
        SideEffectThenInvalidBatchAdapter,
        opening_balances={"payer": Decimal("5.00")},
    )

    with pytest.raises(ValueError, match="duplicate event_id"):
        engine.run()

    assert engine.ledger.balance("payer") == Decimal("4.00")
    _assert_terminal_failure(engine)


def test_cutoff_cannot_admit_an_event_after_the_executed_command_time() -> None:
    """Catch a command under the cutoff emitting an event beyond that cutoff."""
    engine = _engine(FutureEventAdapter)

    with pytest.raises(ValueError, match="event_time must equal simulation time"):
        engine.run(until=NOW)

    _assert_terminal_failure(engine)


def test_direct_emit_requires_the_current_simulation_timestamp() -> None:
    """Catch direct admission of a future event that breaks later queue causality."""
    engine = _engine(InitializingAdapter)

    with pytest.raises(ValueError, match="event_time must equal simulation time"):
        engine.emit(_event(event_time=NOW + timedelta(seconds=1)))

    _assert_terminal_failure(engine)


def test_live_equal_time_batch_is_canonicalized_by_event_id() -> None:
    """Catch preserving adapter list order instead of replay-manifest order."""
    events = _engine(UnsortedBatchAdapter).run()

    assert [event.event_id for event in events] == [
        "00000000-0000-4000-8000-000000000010",
        "00000000-0000-4000-8000-000000000020",
    ]


def test_cross_batch_equal_time_order_regression_is_terminal() -> None:
    """Catch appending a lower event ID after a higher equal-time event ID."""
    engine = _engine(CrossBatchOrderAdapter)

    with pytest.raises(ValueError, match="event ordering must be canonical"):
        engine.run()

    _assert_terminal_failure(engine)


def test_engine_generated_ids_are_rfc_valid_and_monotonic_for_equal_time_batches() -> None:
    """Catch random UUID order violating event-time-then-ID replay ordering."""
    events = _engine(DeterministicIdAdapter).run()
    event_ids = [event.event_id for event in events]

    assert event_ids == sorted(event_ids)
    assert all(UUID(event_id).version == 4 for event_id in event_ids)


def test_replay_rejects_noncanonical_equal_time_event_id_order() -> None:
    """Catch replay accepting an artifact live emission would not produce."""
    high = _event(event_id="00000000-0000-4000-8000-000000000020")
    low = _event(event_id="00000000-0000-4000-8000-000000000010")

    with pytest.raises(ValueError, match="event ordering must be canonical"):
        SimulationEngine.replay_events(_event_bytes((high, low)))


def test_bundle_and_entity_state_are_deep_engine_owned_snapshots() -> None:
    """Catch nested caller aliases mutating bundle or entity state after admission."""
    bundle = _bundle()
    bundle.extensions["nested"] = {"values": ["original"]}
    engine = SimulationEngine(bundle, {Rail.A2A: InitializingAdapter})
    bundle.extensions["nested"]["values"].append("source")  # type: ignore[index]

    first_bundle = engine.bundle
    first_bundle.extensions["nested"]["values"].append("returned")  # type: ignore[index]
    assert engine.bundle.extensions == {"nested": {"values": ["original"]}}

    history = [{"status": "created"}]
    state = {"history": history}
    engine.set_entity_state("payment", state)
    history[0]["status"] = "source-mutated"
    returned_state = engine.entity_state("payment")
    assert isinstance(returned_state, Mapping)
    returned_history = returned_state["history"]
    assert isinstance(returned_history, tuple)
    returned_item = returned_history[0]
    assert isinstance(returned_item, Mapping)
    with pytest.raises(TypeError):
        returned_item["status"] = "returned-mutated"  # type: ignore[index]
    assert engine.entity_state("payment") == MappingProxyType(
        {"history": (MappingProxyType({"status": "created"}),)}
    )


def test_event_rail_must_match_scenario_rail() -> None:
    """Catch a rail adapter atomically admitting an event for another rail."""
    engine = _engine(InitializingAdapter)

    with pytest.raises(ValueError, match="event rail must match scenario rail"):
        engine.emit(_event(rail=Rail.CARD))

    _assert_terminal_failure(engine)


@pytest.mark.parametrize(
    ("invalid_event", "message"),
    [
        (_event(rail_data={"risk": float("nan")}), "event contains non-finite number"),
        (_event(rail_data={"risk": float("inf")}), "event contains non-finite number"),
        (_event(rail_data={"risk": float("-inf")}), "event contains non-finite number"),
        (_event(extensions={"opaque": object()}), "event must be strict JSON"),
    ],
)
def test_event_admission_rejects_non_strict_json(
    invalid_event: PaymentEvent, message: str
) -> None:
    """Catch NaN or opaque nested values entering supposedly canonical artifacts."""
    engine = _engine(InitializingAdapter)

    with pytest.raises(ValueError, match=message):
        engine.emit(invalid_event)

    _assert_terminal_failure(engine)


def test_replay_rejects_nonstandard_json_constants() -> None:
    """Catch Python's permissive JSON parser accepting NaN in a stored artifact."""
    payload = _event_bytes((_event(),)).replace(b'"10.00"', b"NaN", 1)

    with pytest.raises(ValueError, match="non-standard JSON constant: NaN"):
        SimulationEngine.replay_events(payload)


def test_adapter_receives_restricted_capability_context() -> None:
    """Catch passing the owning engine and its direct private mutation paths to adapters."""
    engine = _engine(
        ContextBoundaryAdapter,
        opening_balances={"payer": Decimal("5.00")},
    )

    assert len(engine.run()) == 1


def test_factory_created_adapter_instances_are_owned_independently() -> None:
    """Catch cross-engine adapter callback state leaking into same-seed output."""
    created: list[StatefulAdapter] = []

    def factory() -> RailAdapter:
        adapter = StatefulAdapter()
        created.append(adapter)
        return adapter

    first_engine = SimulationEngine(_bundle(), {Rail.A2A: factory})
    second_engine = SimulationEngine(_bundle(), {Rail.A2A: factory})

    first = first_engine.run()
    second = second_engine.run()

    assert _event_bytes(first) == _event_bytes(second)
    assert len(created) == 2
    assert all(adapter.handled == 1 for adapter in created)


def test_entity_state_freezes_a_closed_nested_lifecycle_algebra() -> None:
    """Catch retaining mutable containers or enum identities in lifecycle state."""
    engine = _engine(InitializingAdapter)
    history = [{"phase": LifecycleState.INITIATED}]
    source = {
        "phase": LifecycleState.AUTHORIZED,
        "custom_phase": StatePhase.READY,
        "history": history,
        "flags": ["trusted", "reviewed"],
        "amount": Decimal("12.50"),
        "at": NOW,
        "metadata": None,
    }

    engine.set_entity_state("payment", source)
    history[0]["phase"] = LifecycleState.REJECTED
    frozen = engine.entity_state("payment")

    assert isinstance(frozen, Mapping)
    assert frozen["phase"] == "authorized"
    assert type(frozen["phase"]) is str
    assert frozen["custom_phase"] == "ready"
    assert frozen["history"] == (MappingProxyType({"phase": "initiated"}),)
    assert frozen["flags"] == ("trusted", "reviewed")
    with pytest.raises(TypeError):
        frozen["phase"] = "rejected"  # type: ignore[index]


@pytest.mark.parametrize(
    ("unsafe", "message"),
    [
        (MutableStateInt(3), "unsupported entity state"),
        (float("nan"), "finite"),
        (float("inf"), "finite"),
        (Decimal("NaN"), "finite"),
        (datetime(2026, 8, 16, 12), "UTC"),
        ({MutableStateStr("key"): "value"}, "exact strings"),
        ({"unordered"}, "unsupported entity state"),
        (frozenset({"unordered"}), "unsupported entity state"),
    ],
)
def test_entity_state_rejects_values_outside_the_closed_algebra(
    unsafe: object, message: str
) -> None:
    """Catch aliases, ambiguous time, mutable keys, or non-finite state values."""
    engine = _engine(InitializingAdapter)

    with pytest.raises((TypeError, ValueError), match=message):
        engine.set_entity_state("payment", unsafe)


def test_entity_state_rejects_hostile_deepcopy_without_invoking_it() -> None:
    """Catch arbitrary state objects defeating ownership through __deepcopy__."""
    engine = _engine(InitializingAdapter)
    hostile = AliasState()

    with pytest.raises(TypeError, match="unsupported entity state: AliasState"):
        engine.set_entity_state("payment", hostile)

    assert hostile.deepcopy_called is False


def test_factory_constructs_deepcopy_refusing_adapter() -> None:
    """Catch borrowing/deepcopying an instance instead of owning factory output."""
    engine = SimulationEngine(_bundle(), {Rail.A2A: RefusesDeepcopyAdapter})

    assert len(engine.run()) == 1


def test_same_factory_produces_isolated_same_seed_engines() -> None:
    """Catch state sharing when two engines use the same trusted factory."""
    first_engine = SimulationEngine(_bundle(), {Rail.A2A: StatefulAdapter})
    second_engine = SimulationEngine(_bundle(), {Rail.A2A: StatefulAdapter})

    assert _event_bytes(first_engine.run()) == _event_bytes(second_engine.run())


def test_factory_returning_a_live_shared_instance_is_rejected() -> None:
    """Catch a misconfigured factory lending one stateful adapter to two engines."""
    shared = StatefulAdapter()

    def shared_factory() -> RailAdapter:
        return shared

    first_engine = SimulationEngine(_bundle(), {Rail.A2A: shared_factory})
    assert first_engine
    with pytest.raises(ValueError, match="fresh adapter instance"):
        SimulationEngine(_bundle(), {Rail.A2A: shared_factory})


def test_unused_broken_factory_is_not_constructed() -> None:
    """Catch eagerly constructing adapters for rails outside the scenario."""
    def broken_factory() -> RailAdapter:
        raise AssertionError("unused card factory was called")

    engine = SimulationEngine(
        _bundle(),
        {Rail.A2A: InitializingAdapter, Rail.CARD: broken_factory},
    )

    assert len(engine.run()) == 1


def test_raw_adapter_instance_is_not_an_adapter_factory() -> None:
    """Catch reintroducing borrowed mutable adapter instances at construction."""
    with pytest.raises(TypeError, match="adapter factory must be callable"):
        SimulationEngine(_bundle(), {Rail.A2A: InitializingAdapter()})  # type: ignore[dict-item]


def test_partial_factory_can_capture_trusted_adapter_configuration() -> None:
    """Catch a factory contract too narrow for configurable Task 4 adapters."""
    factory = partial(ConfiguredAdapter, amount=7)

    events = SimulationEngine(_bundle(), {Rail.A2A: factory}).run()

    assert events[0].amount == Decimal("7")


def test_retained_capability_facade_contains_no_engine_reference() -> None:
    """Catch direct or name-mangled owner references on context/random facades."""
    CaptureContextAdapter.captured_context = None
    CaptureContextAdapter.captured_rng = None
    engine = _engine(CaptureContextAdapter)
    assert len(engine.run()) == 1
    context = CaptureContextAdapter.captured_context
    rng = CaptureContextAdapter.captured_rng
    assert context is not None
    assert rng is not None

    assert not hasattr(context, "_RailContext__engine")
    assert not hasattr(rng, "_RandomCapability__engine")
    assert vars(type(context)).get("__slots__") == ()
    assert vars(type(rng)).get("__slots__") == ()
    assert not hasattr(context, "_clock")
    assert not hasattr(rng, "_rng")
    with pytest.raises(RuntimeError, match="adapter context is inactive"):
        context.schedule(NOW, 0, Command("retained"))
    with pytest.raises(RuntimeError, match="adapter context is inactive"):
        rng.integers(0, 2)
    assert len(engine.events) == 1


def test_contextvar_cleanup_runs_when_adapter_raises() -> None:
    """Catch an exception leaving callback capabilities active."""
    CaptureContextThenFailAdapter.captured_context = None
    engine = _engine(CaptureContextThenFailAdapter)

    with pytest.raises(RuntimeError, match="capture initialize failed"):
        engine.run()
    context = CaptureContextThenFailAdapter.captured_context
    assert context is not None
    with pytest.raises(RuntimeError, match="adapter context is inactive"):
        context.set_entity_state("late", "mutation")


def test_nested_engine_callback_restores_outer_context() -> None:
    """Catch nested callbacks clearing rather than restoring the outer dispatch."""
    nested = SimulationEngine(_bundle(), {Rail.A2A: InitializingAdapter})
    outer = SimulationEngine(
        _bundle(),
        {Rail.A2A: lambda: RunNestedEngineAdapter(nested)},
    )

    assert len(outer.run()) == 1
    assert outer.entity_state("outer") == MappingProxyType(
        {"callback": ("restored",)}
    )


def test_copied_execution_context_cannot_reuse_revoked_callback_authority() -> None:
    """Catch copy_context retaining a live engine token after callback return."""
    CopyContextRetentionAdapter.captured_execution_context = None
    CopyContextRetentionAdapter.captured_rail_context = None
    engine = _engine(CopyContextRetentionAdapter)
    engine.run()
    execution_context = CopyContextRetentionAdapter.captured_execution_context
    rail_context = CopyContextRetentionAdapter.captured_rail_context
    assert execution_context is not None
    assert rail_context is not None

    with pytest.raises(RuntimeError, match="adapter context is inactive"):
        execution_context.run(
            rail_context.set_entity_state,
            "late",
            "copied-context-mutation",
        )

    retained_engine: ReferenceType[SimulationEngine] = ref(engine)
    del engine
    gc.collect()
    assert retained_engine() is None


def test_child_async_task_cannot_reuse_inherited_callback_authority() -> None:
    """Catch create_task inheriting a usable engine token beyond callback return."""

    async def exercise() -> None:
        AsyncTaskRetentionAdapter.gate = None
        AsyncTaskRetentionAdapter.task = None
        engine = _engine(AsyncTaskRetentionAdapter)
        engine.run()
        gate = AsyncTaskRetentionAdapter.gate
        task = AsyncTaskRetentionAdapter.task
        assert gate is not None
        assert task is not None

        gate.set()
        error = await task

        assert isinstance(error, RuntimeError)
        assert str(error) == "adapter context is inactive"
        with pytest.raises(KeyError, match="late"):
            engine.entity_state("late")

    asyncio.run(exercise())


def test_entity_state_detaches_caller_owned_utc_tzinfo() -> None:
    """Catch a mutable custom tzinfo changing an admitted entity timestamp."""
    mutable_utc = MutableUtcTz()
    source = datetime(
        2026,
        8,
        16,
        12,
        34,
        56,
        123456,
        tzinfo=mutable_utc,
        fold=1,
    )
    engine = _engine(InitializingAdapter)

    engine.set_entity_state("timestamp", source)
    mutable_utc.offset = timedelta(hours=5, minutes=30)
    mutable_utc.name = "IST"
    owned = engine.entity_state("timestamp")

    assert type(owned) is datetime
    assert owned is not source
    assert owned.tzinfo is UTC
    assert owned == datetime(
        2026,
        8,
        16,
        12,
        34,
        56,
        123456,
        tzinfo=UTC,
        fold=1,
    )
    assert owned.fold == 1


@pytest.mark.parametrize(
    "unsafe",
    [
        MutableFloat("0.5"),
        MutableFloat("NaN"),
        MutableFloat("Infinity"),
        MutableStateInt(3),
        LyingDecimal("Infinity"),
    ],
)
def test_open_fields_reject_numeric_scalar_subclasses(unsafe: object) -> None:
    """Catch numeric subclasses overriding or carrying mutable scalar semantics."""
    engine = _engine(InitializingAdapter)

    with pytest.raises(TypeError, match="unsupported numeric subclass"):
        engine.emit(_event(extensions={"unsafe": unsafe}))


def test_open_fields_keep_exact_finite_numbers_and_infinity_text() -> None:
    """Catch subclass hardening accidentally rejecting safe exact scalar values."""
    engine = _engine(InitializingAdapter)

    engine.emit(
        _event(
            extensions={
                "float": 0.5,
                "integer": 3,
                "decimal": Decimal("1.25"),
                "literal": "Infinity",
            }
        )
    )

    assert engine.events[0].extensions == {
        "float": 0.5,
        "integer": 3,
        "decimal": "1.25",
        "literal": "Infinity",
    }


def test_engine_owns_the_normalized_clock_timestamp_in_both_queues() -> None:
    """Catch the engine due-time mirror retaining a mutable scheduled datetime."""
    engine = _engine(MutableScheduledTimeAdapter)
    event = engine.run()[0]
    expected = datetime(
        2026,
        8,
        16,
        12,
        0,
        1,
        123456,
        tzinfo=UTC,
        fold=1,
    )

    assert type(engine.now) is datetime
    assert engine.now.tzinfo is UTC
    assert engine.now == expected
    assert engine.now.fold == 1
    assert event.event_time == expected


@pytest.mark.parametrize(
    "unordered",
    [{"first", "second"}, frozenset({"first", "second"})],
)
def test_event_admission_rejects_nested_unordered_artifact_values(
    unordered: object,
) -> None:
    """Catch set iteration order entering canonical event JSON artifacts."""
    engine = _engine(InitializingAdapter)

    with pytest.raises(TypeError, match="event contains unordered container"):
        engine.emit(_event(extensions={"nested": [{"values": unordered}]}))

    _assert_terminal_failure(engine)


@pytest.mark.parametrize(
    "unordered",
    [{"first", "second"}, frozenset({"first", "second"})],
)
def test_bundle_admission_rejects_nested_unordered_artifact_values(
    unordered: object,
) -> None:
    """Catch set iteration order entering canonical scenario JSON artifacts."""
    bundle = _bundle()
    bundle.extensions["nested"] = {"values": unordered}

    with pytest.raises(TypeError, match="scenario bundle contains unordered container"):
        SimulationEngine(bundle, {Rail.A2A: InitializingAdapter})


def test_artifact_admission_preserves_ordered_list_and_tuple_values() -> None:
    """Catch unordered-input hardening rejecting deterministic sequences."""
    engine = _engine(InitializingAdapter)

    engine.emit(
        _event(
            extensions={
                "list": ["first", "second"],
                "tuple": ("third", "fourth"),
            }
        )
    )

    assert engine.events[0].extensions == {
        "list": ["first", "second"],
        "tuple": ["third", "fourth"],
    }


@pytest.mark.parametrize(
    "non_finite",
    [Decimal("NaN"), Decimal("Infinity"), Decimal("-Infinity")],
)
def test_python_mode_event_input_rejects_nested_non_finite_decimal(
    non_finite: Decimal,
) -> None:
    """Catch JSON-mode serialization converting Decimal non-finite values to strings."""
    engine = _engine(InitializingAdapter)
    invalid = _event(extensions={"nested": [{"value": non_finite}]})

    with pytest.raises(ValueError, match="event contains non-finite number"):
        engine.emit(invalid)


def test_python_mode_bundle_input_rejects_nested_non_finite_decimal() -> None:
    """Catch scenario admission stringifying non-finite Decimal extension values."""
    bundle = _bundle()
    bundle.extensions["nested"] = {"value": Decimal("Infinity")}

    with pytest.raises(ValueError, match="scenario bundle contains non-finite number"):
        SimulationEngine(bundle, {Rail.A2A: InitializingAdapter})


def test_exact_nan_text_remains_valid_text_in_open_contract_fields() -> None:
    """Catch finite validation confusing a literal string with a non-finite number."""
    engine = _engine(InitializingAdapter)

    engine.emit(_event(extensions={"literal": "NaN"}))

    assert engine.events[0].extensions == {"literal": "NaN"}
