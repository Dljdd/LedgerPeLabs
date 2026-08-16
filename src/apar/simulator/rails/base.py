"""Capability protocols implemented by payment-rail simulators."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
from decimal import Decimal
from typing import Any, Protocol

import numpy as np

from apar.contracts.events import PaymentEvent
from apar.contracts.scenarios import ScenarioBundle
from apar.simulator.clock import Command
from apar.simulator.ledger import LedgerEntry

type FrozenState = (
    None
    | bool
    | int
    | float
    | str
    | bytes
    | Decimal
    | datetime
    | Mapping[str, FrozenState]
    | tuple[FrozenState, ...]
    | frozenset[FrozenState]
)


class RandomCapability(Protocol):
    """Callback-scoped access to the engine-local NumPy generator."""

    def integers(
        self,
        low: int,
        high: int | None = None,
        size: None = None,
        *,
        endpoint: bool = False,
    ) -> np.integer[Any]:
        """Sample integer values from the local generator."""
        ...

    def uniform(
        self,
        low: float = 0.0,
        high: float = 1.0,
        size: int | tuple[int, ...] | None = None,
    ) -> object:
        """Sample uniform values from the local generator."""
        ...

    def random(self, size: int | tuple[int, ...] | None = None) -> object:
        """Sample values on the half-open unit interval."""
        ...

    def bytes(self, length: int) -> bytes:
        """Sample deterministic bytes from the local generator."""
        ...


class LedgerReader(Protocol):
    """Read-only ledger operations available to rail adapters."""

    @property
    def entries(self) -> tuple[LedgerEntry, ...]:
        """Return the append-only posting history."""
        ...

    def balance(self, account: str, currency: str = "USD") -> Decimal:
        """Return one account balance."""
        ...

    def assert_conserved(self) -> None:
        """Assert value conservation."""
        ...


class RailContext(Protocol):
    """Restricted capabilities supplied during adapter callbacks."""

    @property
    def bundle(self) -> ScenarioBundle:
        """Return a defensive scenario snapshot."""
        ...

    @property
    def now(self) -> datetime:
        """Return current simulation time."""
        ...

    @property
    def rng(self) -> RandomCapability:
        """Return callback-scoped local randomness."""
        ...

    @property
    def ledger(self) -> LedgerReader:
        """Return the read-only ledger view."""
        ...

    def schedule(self, at: datetime, priority: int, command: Command) -> None:
        """Request a future command."""
        ...

    def post(self, entry: LedgerEntry) -> None:
        """Request one validated ledger posting."""
        ...

    def entity_state(self, entity_id: str) -> FrozenState:
        """Read recursively immutable, engine-owned entity state."""
        ...

    def set_entity_state(self, entity_id: str, state: object) -> None:
        """Store engine-owned entity state."""
        ...

    def new_uuid(self) -> str:
        """Return a deterministic, monotonic RFC UUID."""
        ...


class RailAdapter(Protocol):
    """Initialize and handle commands through restricted engine capabilities."""

    def initialize(self, context: RailContext) -> None:
        """Register the adapter's initial future commands."""
        ...

    def handle(self, command: Command, context: RailContext) -> list[PaymentEvent]:
        """Handle one due command and return its emitted payment events."""
        ...


class AdapterFactory(Protocol):
    """Trusted constructor for one fresh, selected-rail adapter instance."""

    def __call__(self) -> RailAdapter:
        """Construct a fresh adapter; closures and partials are supported."""
        ...


__all__ = [
    "AdapterFactory",
    "FrozenState",
    "LedgerReader",
    "RailAdapter",
    "RailContext",
    "RandomCapability",
]
