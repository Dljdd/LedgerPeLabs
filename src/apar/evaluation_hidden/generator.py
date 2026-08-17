"""Independent hidden motifs, schedules, and leaf distributions.

This implementation intentionally shares neither population nor campaign-generation code
with the production range.  It builds a causal entity graph and lifecycle schedule before
sampling bounded leaf amounts with Python's independent ``random`` implementation.
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from decimal import ROUND_HALF_EVEN, Decimal
from uuid import NAMESPACE_URL, uuid5

from apar.contracts.events import EventKind, PaymentEvent, Rail

_FAMILIES = frozenset(
    {
        "agentic_intent_abuse",
        "app_scam_mule",
        "card_testing_cnp",
        "synthetic_merchant_refund",
    }
)
_CENT = Decimal("0.01")
_START = datetime(2026, 8, 16, tzinfo=UTC)


@dataclass(frozen=True, slots=True)
class _HiddenFlow:
    payment_index: int
    rail: Rail
    actor: str
    counterparty: str
    actor_role: str
    counterparty_role: str
    actor_opening: Decimal
    counterparty_opening: Decimal
    amount: Decimal
    lifecycle: tuple[EventKind, ...]
    integrity_reason: str = ""


class _IndependentSchedule:
    """Generate a causal schedule from hidden, independently named timing controls."""

    __slots__ = ("_cursor", "_rng")

    def __init__(self, rng: random.Random, seed: int) -> None:
        self._rng = rng
        self._cursor = _START + timedelta(minutes=seed % 43)

    def next_stage(self, *, burst: bool) -> datetime:
        floor, ceiling = (2, 13) if burst else (17, 71)
        self._cursor += timedelta(seconds=self._rng.randint(floor, ceiling))
        return self._cursor


class HiddenCampaignGenerator:
    """Produce an independent deterministic event corpus for four hidden families."""

    def generate(self, family: str, seed: int, count: int) -> tuple[PaymentEvent, ...]:
        """Generate one closed hidden campaign without production-generator dependencies."""
        if type(family) is not str or family not in _FAMILIES:
            raise ValueError("unsupported hidden campaign family")
        if type(seed) is not int:
            raise TypeError("seed must be an exact integer")
        if not 0 <= seed < 2**63:
            raise ValueError("seed must be in [0, 2**63)")
        if type(count) is not int:
            raise TypeError("count must be an exact integer")
        if not 4 <= count <= 128:
            raise ValueError("count must be in [4, 128]")

        digest = hashlib.sha256(f"hidden-v1:{family}:{seed}".encode()).digest()
        rng = random.Random(int.from_bytes(digest, "big"))
        namespace = uuid5(NAMESPACE_URL, f"apar:hidden:v1:{family}:{seed}")
        campaign_id = str(uuid5(namespace, "campaign"))
        actors = tuple(str(uuid5(namespace, f"actor:{index}")) for index in range(count + 4))
        flows = self._motif(family, count, actors, rng)
        schedule = _IndependentSchedule(rng, seed)
        events: list[PaymentEvent] = []
        for flow in flows:
            trace_id = str(uuid5(namespace, f"trace:{flow.payment_index}"))
            payment_id = f"hidden:{family}:{flow.payment_index}"
            previous = ""
            for stage_index, event_type in enumerate(flow.lifecycle):
                event_time = schedule.next_stage(
                    burst=family in {"agentic_intent_abuse", "card_testing_cnp"}
                )
                event_id = str(
                    uuid5(namespace, f"event:{flow.payment_index}:{stage_index}:{event_type.value}")
                )
                lineage: dict[str, str | bool] = {
                    "synthetic": True,
                    "hidden_generator": "independent-v1",
                }
                if previous:
                    lineage["previous_event_id"] = previous
                rail_data: dict[str, str | int | float | bool] = {
                    "payment_id": payment_id,
                    "hidden_family": family,
                    "hidden_stage": stage_index,
                }
                if flow.integrity_reason:
                    rail_data["integrity"] = (
                        "pass" if event_type is EventKind.AUTHORIZATION else "fail"
                    )
                    rail_data["reason_code"] = flow.integrity_reason
                events.append(
                    PaymentEvent(
                        schema_version="1.0.0",
                        event_id=event_id,
                        campaign_id=campaign_id,
                        trace_id=trace_id,
                        rail=flow.rail,
                        viewpoint="hidden_synthetic_evaluator",
                        event_type=event_type,
                        amount=flow.amount,
                        currency="USD",
                        event_time=event_time,
                        ingested_at=event_time + timedelta(milliseconds=11),
                        available_at=event_time + timedelta(milliseconds=17),
                        decision_at=event_time + timedelta(milliseconds=23),
                        actor_id=flow.actor,
                        counterparty_id=flow.counterparty,
                        party_refs={
                            "actor_role": flow.actor_role,
                            "counterparty_role": flow.counterparty_role,
                            "actor_opening_balance": str(flow.actor_opening),
                            "counterparty_opening_balance": str(flow.counterparty_opening),
                        },
                        rail_data=rail_data,
                        lineage=lineage,
                        privacy={"classification": "synthetic", "retention": "restricted"},
                    )
                )
                previous = event_id
        return tuple(events)

    def _motif(
        self,
        family: str,
        count: int,
        actors: tuple[str, ...],
        rng: random.Random,
    ) -> tuple[_HiddenFlow, ...]:
        if family == "app_scam_mule":
            return self._app_motif(count, actors, rng)
        if family == "card_testing_cnp":
            return self._card_testing_motif(count, actors, rng)
        if family == "synthetic_merchant_refund":
            return self._refund_motif(count, actors, rng)
        return self._agentic_motif(count, actors, rng)

    @staticmethod
    def _amount(rng: random.Random, low: int, high: int) -> Decimal:
        sampled = Decimal(str(low + (high - low) * rng.betavariate(2.1, 3.4)))
        return sampled.quantize(_CENT, rounding=ROUND_HALF_EVEN)

    def _app_motif(
        self,
        count: int,
        actors: tuple[str, ...],
        rng: random.Random,
    ) -> tuple[_HiddenFlow, ...]:
        root_mule, second_mule, cash_endpoint = actors[0], actors[1], actors[2]
        incoming_count = count - 2
        flows: list[_HiddenFlow] = []
        incoming_total = Decimal("0.00")
        for index in range(incoming_count):
            amount = self._amount(rng, 42, 210)
            incoming_total += amount
            flows.append(
                _HiddenFlow(
                    index,
                    Rail.A2A,
                    actors[index + 3],
                    root_mule,
                    "victim",
                    "mule",
                    Decimal("1000.00"),
                    Decimal("0.00"),
                    amount,
                    (
                        EventKind.TRANSFER_INITIATED,
                        EventKind.TRANSFER_ACCEPTED,
                        EventKind.TRANSFER_POSTED,
                    ),
                )
            )
        layer_amount = (incoming_total * Decimal("0.54")).quantize(_CENT)
        flows.append(
            _HiddenFlow(
                incoming_count,
                Rail.A2A,
                root_mule,
                second_mule,
                "mule",
                "mule",
                Decimal("0.00"),
                Decimal("0.00"),
                layer_amount,
                (
                    EventKind.TRANSFER_INITIATED,
                    EventKind.TRANSFER_ACCEPTED,
                    EventKind.TRANSFER_POSTED,
                ),
            )
        )
        cash_amount = (layer_amount * Decimal("0.67")).quantize(_CENT)
        flows.append(
            _HiddenFlow(
                incoming_count + 1,
                Rail.A2A,
                second_mule,
                cash_endpoint,
                "mule",
                "attacker",
                Decimal("0.00"),
                Decimal("0.00"),
                cash_amount,
                (
                    EventKind.TRANSFER_INITIATED,
                    EventKind.TRANSFER_ACCEPTED,
                    EventKind.TRANSFER_POSTED,
                    EventKind.FRAUD_REPORTED,
                    EventKind.FUNDS_FROZEN,
                    EventKind.RECOVERY,
                ),
            )
        )
        return tuple(flows)

    def _card_testing_motif(
        self,
        count: int,
        actors: tuple[str, ...],
        rng: random.Random,
    ) -> tuple[_HiddenFlow, ...]:
        shared_actor = actors[0]
        probes = max(2, count // 2)
        flows: list[_HiddenFlow] = []
        for index in range(count):
            declined = index < probes
            lifecycle = (
                (EventKind.AUTHORIZATION_DECLINED,)
                if declined
                else (
                    EventKind.AUTHORIZATION,
                    EventKind.CLEARING,
                    EventKind.SETTLEMENT,
                )
            )
            if index == count - 1:
                lifecycle = (
                    *lifecycle,
                    EventKind.FRAUD_REPORTED,
                    EventKind.DISPUTE_OPENED,
                    EventKind.CHARGEBACK,
                    EventKind.RECOVERY,
                )
            flows.append(
                _HiddenFlow(
                    index,
                    Rail.CARD,
                    shared_actor,
                    actors[1 + index % 3],
                    "compromised_consumer",
                    "merchant",
                    Decimal("2500.00"),
                    Decimal("500.00"),
                    self._amount(rng, 1, 14) if declined else self._amount(rng, 24, 118),
                    lifecycle,
                )
            )
        return tuple(flows)

    def _refund_motif(
        self,
        count: int,
        actors: tuple[str, ...],
        rng: random.Random,
    ) -> tuple[_HiddenFlow, ...]:
        merchant = actors[0]
        flows: list[_HiddenFlow] = []
        for index in range(count):
            refunded = index < max(2, count // 2)
            lifecycle = (
                EventKind.AUTHORIZATION,
                EventKind.CLEARING,
                EventKind.SETTLEMENT,
                *((EventKind.REFUND,) if refunded else ()),
            )
            flows.append(
                _HiddenFlow(
                    index,
                    Rail.CARD,
                    actors[1 + index % 3],
                    merchant,
                    "consumer",
                    "synthetic_merchant",
                    Decimal("1800.00"),
                    Decimal("400.00"),
                    self._amount(rng, 18, 155),
                    lifecycle,
                )
            )
        return tuple(flows)

    def _agentic_motif(
        self,
        count: int,
        actors: tuple[str, ...],
        rng: random.Random,
    ) -> tuple[_HiddenFlow, ...]:
        agent, merchant = actors[0], actors[1]
        reasons = (
            "SIGNATURE_INVALID",
            "MANDATE_EXPIRED",
            "CART_HASH_MISMATCH",
            "MERCHANT_BINDING_MISMATCH",
            "NONCE_REPLAYED",
            "CREDENTIAL_SCOPE_INVALID",
        )
        flows: list[_HiddenFlow] = []
        for index in range(count):
            allowed = index >= count - 2
            flows.append(
                _HiddenFlow(
                    index,
                    Rail.AGENTIC,
                    agent,
                    merchant,
                    "agent",
                    "merchant",
                    Decimal("2500.00"),
                    Decimal("500.00"),
                    self._amount(rng, 12, 96),
                    (
                        EventKind.AUTHORIZATION
                        if allowed
                        else EventKind.AUTHORIZATION_DECLINED,
                    ),
                    "" if allowed else reasons[index % len(reasons)],
                )
            )
        return tuple(flows)


__all__ = ["HiddenCampaignGenerator"]
