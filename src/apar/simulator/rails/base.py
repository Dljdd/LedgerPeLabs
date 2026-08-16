"""Structural protocol implemented by payment-rail simulators."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

from apar.contracts.events import PaymentEvent
from apar.simulator.clock import Command

if TYPE_CHECKING:
    from apar.simulator.engine import SimulationEngine


class RailAdapter(Protocol):
    """Initialize and handle commands through the engine's public facade."""

    def initialize(self, engine: SimulationEngine) -> None:
        """Register the adapter's initial future commands."""
        ...

    def handle(self, command: Command, engine: SimulationEngine) -> list[PaymentEvent]:
        """Handle one due command and return its emitted payment events."""
        ...
