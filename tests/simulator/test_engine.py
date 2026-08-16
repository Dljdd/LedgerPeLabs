"""Behavioral coverage for deterministic simulation orchestration."""

from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import datetime, timedelta, timezone
from decimal import Decimal

import numpy as np
import pytest

from apar.compiler.compiler import compile_scenario
from apar.contracts.events import EventKind, PaymentEvent, Rail
from apar.contracts.scenarios import ReplayConfig, ScenarioBundle
from apar.simulator.clock import Command
from apar.simulator.engine import SimulationEngine
from apar.simulator.ledger import AccountReference, LedgerEntry
from apar.simulator.rails.base import RailAdapter
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
) -> PaymentEvent:
    return PaymentEvent(
        schema_version="1.0.0",
        event_id=event_id,
        campaign_id="00000000-0000-4000-8000-000000000002",
        trace_id="00000000-0000-4000-8000-000000000003",
        rail=Rail.A2A,
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
    )


class InitializingAdapter:
    """Schedule one deterministic event from the adapter initialization boundary."""

    def initialize(self, engine: SimulationEngine) -> None:
        engine.schedule(NOW, 0, Command("emit"))

    def handle(self, command: Command, engine: SimulationEngine) -> list[PaymentEvent]:
        assert command.name == "emit"
        return [
            _event(
                event_id=engine.new_uuid(),
                amount=Decimal(int(engine.rng.integers(1, 1000))) / 100,
            )
        ]


class DuplicateBatchAdapter:
    """Return a duplicate batch so event append atomicity can be observed."""

    def initialize(self, engine: SimulationEngine) -> None:
        engine.schedule(NOW, 0, Command("duplicate"))

    def handle(self, command: Command, engine: SimulationEngine) -> list[PaymentEvent]:
        event = _event()
        return [event, event]


class PastSchedulingAdapter:
    """Try to schedule work behind the simulation clock while handling a command."""

    def initialize(self, engine: SimulationEngine) -> None:
        engine.schedule(NOW + timedelta(seconds=1), 0, Command("time-travel"))

    def handle(self, command: Command, engine: SimulationEngine) -> list[PaymentEvent]:
        engine.schedule(NOW, 0, Command("past"))
        return []


class PastEventAdapter:
    """Return an event whose event time predates the command being handled."""

    def initialize(self, engine: SimulationEngine) -> None:
        engine.schedule(NOW + timedelta(seconds=1), 0, Command("late"))

    def handle(self, command: Command, engine: SimulationEngine) -> list[PaymentEvent]:
        return [_event(event_time=NOW)]


class TwoEventAdapter:
    """Schedule two events to make cutoff and resumed-drain behavior visible."""

    def initialize(self, engine: SimulationEngine) -> None:
        engine.schedule(NOW + timedelta(seconds=1), 0, Command("first"))
        engine.schedule(NOW + timedelta(seconds=2), 0, Command("second"))

    def handle(self, command: Command, engine: SimulationEngine) -> list[PaymentEvent]:
        suffix = 11 if command.name == "first" else 12
        return [
            _event(
                event_id=f"00000000-0000-4000-8000-{suffix:012d}",
                event_time=engine.now,
            )
        ]


class PostingAdapter:
    """Exercise adapter access only through the engine's public mutation methods."""

    def initialize(self, engine: SimulationEngine) -> None:
        engine.set_entity_state("payment", "created")
        engine.schedule(NOW, 0, Command("post"))

    def handle(self, command: Command, engine: SimulationEngine) -> list[PaymentEvent]:
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


def _engine(
    adapter: RailAdapter,
    *,
    seed: int = 260_816,
    opening_balances: Mapping[AccountReference, Decimal] | None = None,
) -> SimulationEngine:
    return SimulationEngine(
        _bundle(seed=seed),
        {Rail.A2A: adapter},
        opening_balances=opening_balances,
    )


def _event_bytes(events: tuple[PaymentEvent, ...]) -> bytes:
    payload = [event.model_dump(mode="json") for event in events]
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()


def test_same_seed_produces_byte_identical_events() -> None:
    """Catch engine IDs or sampling depending on global or process-local state."""
    first = _engine(InitializingAdapter(), seed=260_816).run()
    second = _engine(InitializingAdapter(), seed=260_816).run()

    assert _event_bytes(first) == _event_bytes(second)


def test_engine_randomness_does_not_advance_numpy_global_state() -> None:
    """Catch an adapter accidentally receiving NumPy's global random stream."""
    np.random.seed(91)
    expected = int(np.random.randint(0, 1_000_000))
    np.random.seed(91)

    _engine(InitializingAdapter()).run()

    assert int(np.random.randint(0, 1_000_000)) == expected


def test_duplicate_event_id_is_rejected_atomically() -> None:
    """Catch a duplicate event changing the append-only emitted-event snapshot."""
    engine = _engine(InitializingAdapter())
    event = _event()
    engine.emit(event)

    with pytest.raises(ValueError, match="duplicate event_id"):
        engine.emit(event)

    assert engine.events == (event,)


def test_duplicate_event_inside_adapter_batch_rejects_the_whole_batch() -> None:
    """Catch validating event IDs after a prefix of the batch was already appended."""
    engine = _engine(DuplicateBatchAdapter())

    with pytest.raises(ValueError, match="duplicate event_id"):
        engine.run()

    assert engine.events == ()


def test_adapter_cannot_schedule_behind_current_queue_time() -> None:
    """Catch adapter callbacks reintroducing already elapsed simulation work."""
    with pytest.raises(ValueError, match="earlier than the current simulation time"):
        _engine(PastSchedulingAdapter()).run()


def test_event_time_cannot_precede_the_command_that_emits_it() -> None:
    """Catch an emitted stream moving backwards relative to the simulation queue."""
    engine = _engine(PastEventAdapter())

    with pytest.raises(ValueError, match="event_time cannot precede simulation time"):
        engine.run()

    assert engine.events == ()


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
    engine = _engine(TwoEventAdapter())

    with pytest.raises(ValueError, match="cutoff must be a UTC timestamp"):
        engine.run(until=cutoff)


def test_run_until_stops_before_future_work_and_can_resume() -> None:
    """Catch a bounded run draining or losing commands beyond its UTC cutoff."""
    engine = _engine(TwoEventAdapter())

    first = engine.run(until=NOW + timedelta(seconds=1))
    all_events = engine.run()

    assert [event.event_time for event in first] == [NOW + timedelta(seconds=1)]
    assert [event.event_time for event in all_events] == [
        NOW + timedelta(seconds=1),
        NOW + timedelta(seconds=2),
    ]


def test_event_serialization_replays_the_validated_event_tuple() -> None:
    """Catch serialization changing decimal, timestamp, enum, or field values."""
    engine = _engine(InitializingAdapter())
    expected = engine.run()

    payload = engine.serialize_events()
    replayed = SimulationEngine.replay_events(payload)

    assert payload == _event_bytes(expected)
    assert replayed == expected


def test_adapter_uses_public_engine_methods_without_heap_or_store_access() -> None:
    """Catch exposing mutable clock/store internals instead of the adapter facade."""
    engine = _engine(PostingAdapter(), opening_balances={"payer": Decimal("5.00")})

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
    engine = _engine(adapter)

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
