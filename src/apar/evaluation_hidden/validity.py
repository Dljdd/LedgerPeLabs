"""Boolean-only independent campaign validity with restricted post-run detail."""

from __future__ import annotations

import math
from collections import defaultdict, deque
from datetime import timedelta
from decimal import Decimal, InvalidOperation

from pydantic import field_validator

from apar.contracts._validation import ExternalContract
from apar.contracts.events import EventKind, PaymentEvent, Rail

_POSITIVE = frozenset({EventKind.SETTLEMENT, EventKind.TRANSFER_POSTED})
_NEGATIVE = frozenset(
    {
        EventKind.CHARGEBACK,
        EventKind.RECOVERY,
        EventKind.REFUND,
        EventKind.TRANSFER_RETURNED,
    }
)
_ROLE_PAIRS: dict[Rail, frozenset[tuple[str, str]]] = {
    Rail.A2A: frozenset(
        {
            ("consumer", "beneficiary"),
            ("consumer", "merchant"),
            ("consumer", "mule"),
            ("mule", "attacker"),
            ("mule", "mule"),
            ("organization", "beneficiary"),
            ("organization", "merchant"),
            ("organization", "mule"),
            ("victim", "beneficiary"),
            ("victim", "mule"),
        }
    ),
    Rail.CARD: frozenset(
        {
            ("compromised_consumer", "merchant"),
            ("consumer", "merchant"),
            ("consumer", "synthetic_merchant"),
            ("victim", "merchant"),
            ("victim", "synthetic_merchant"),
        }
    ),
    Rail.AGENTIC: frozenset({("agent", "merchant"), ("agent", "synthetic_merchant")}),
}


class HiddenValidityResult(ExternalContract):
    """The complete policy-visible hidden-oracle response."""

    valid: bool

    @field_validator("valid", mode="before")
    @classmethod
    def valid_is_exact(cls, value: object) -> object:
        if type(value) is not bool:
            raise TypeError("valid must be an exact boolean")
        return value


class RestrictedValidityReport(ExternalContract):
    """Evaluator-only post-run reasons stored in a restricted artifact."""

    schema_version: str = "1.0.0"
    valid: bool
    reason_codes: tuple[str, ...]
    metrics: dict[str, str | int | bool]


def _payment_id(event: PaymentEvent) -> str | None:
    value = event.rail_data.get("payment_id")
    return value if type(value) is str and value else None


def _owned_events(events: tuple[PaymentEvent, ...]) -> tuple[PaymentEvent, ...]:
    if type(events) is not tuple or not events:
        raise TypeError("events must be a non-empty exact tuple")
    owned: list[PaymentEvent] = []
    for event in events:
        if type(event) is not PaymentEvent:
            raise TypeError("events must contain exact PaymentEvent records")
        owned.append(PaymentEvent.model_validate(event.model_dump(mode="json", round_trip=True)))
    return tuple(owned)


class HiddenValidityOracle:
    """Apply independent economic, lifecycle, graph, role, and distance checks."""

    def evaluate(self, events: tuple[PaymentEvent, ...]) -> HiddenValidityResult:
        """Return one bit and deliberately discard all hidden rejection details."""
        return HiddenValidityResult(valid=self._report(events).valid)

    def evaluate_restricted(
        self,
        events: tuple[PaymentEvent, ...],
        *,
        run_complete: bool,
    ) -> RestrictedValidityReport:
        """Return detailed reasons only for a completed evaluator-owned run."""
        if type(run_complete) is not bool or not run_complete:
            raise PermissionError("restricted validity is available only after run completion")
        return self._report(events)

    def _report(self, events: tuple[PaymentEvent, ...]) -> RestrictedValidityReport:
        try:
            checked = _owned_events(events)
        except (TypeError, ValueError):
            return RestrictedValidityReport(
                valid=False,
                reason_codes=("MALFORMED_EVENT_INPUT",),
                metrics={"event_count": 0, "payment_count": 0},
            )

        reasons: set[str] = set()
        identifiers = tuple(event.event_id for event in checked)
        if len(set(identifiers)) != len(identifiers):
            reasons.add("DUPLICATE_EVENT_ID")
        if len({event.campaign_id for event in checked}) != 1:
            reasons.add("MULTIPLE_CAMPAIGNS")
        if tuple((event.event_time, event.event_id) for event in checked) != tuple(
            sorted((event.event_time, event.event_id) for event in checked)
        ):
            reasons.add("NON_CANONICAL_ORDER")

        grouped: dict[str, list[PaymentEvent]] = defaultdict(list)
        for event in checked:
            payment_id = _payment_id(event)
            if payment_id is None:
                reasons.add("LIFECYCLE_IDENTITY_MISSING")
                continue
            grouped[payment_id].append(event)
        self._check_lifecycles(grouped, reasons)
        self._check_roles(checked, reasons)
        self._check_balances(checked, reasons)
        self._check_bounds(checked, grouped, reasons)
        self._check_connectivity(checked, reasons)
        distance = self._benign_distance(checked)
        if distance > Decimal("0.85"):
            reasons.add("BENIGN_DISTANCE_EXCEEDED")
        return RestrictedValidityReport(
            valid=not reasons,
            reason_codes=tuple(sorted(reasons)),
            metrics={
                "event_count": len(checked),
                "payment_count": len(grouped),
                "campaign_count": len({event.campaign_id for event in checked}),
                "benign_distance": str(distance),
                "value_conserved": "VALUE_CONSERVATION_FAILED" not in reasons,
            },
        )

    @staticmethod
    def _check_lifecycles(
        grouped: dict[str, list[PaymentEvent]],
        reasons: set[str],
    ) -> None:
        legal_card_prefixes = (
            (EventKind.AUTHORIZATION_DECLINED,),
            (EventKind.AUTHORIZATION, EventKind.CLEARING, EventKind.SETTLEMENT),
            (
                EventKind.AUTHORIZATION,
                EventKind.CLEARING,
                EventKind.SETTLEMENT,
                EventKind.REFUND,
            ),
            (
                EventKind.AUTHORIZATION,
                EventKind.CLEARING,
                EventKind.SETTLEMENT,
                EventKind.FRAUD_REPORTED,
                EventKind.DISPUTE_OPENED,
                EventKind.CHARGEBACK,
            ),
            (
                EventKind.AUTHORIZATION,
                EventKind.CLEARING,
                EventKind.SETTLEMENT,
                EventKind.FRAUD_REPORTED,
                EventKind.DISPUTE_OPENED,
                EventKind.CHARGEBACK,
                EventKind.RECOVERY,
            ),
        )
        legal_a2a = (
            (EventKind.TRANSFER_INITIATED, EventKind.TRANSFER_REJECTED),
            (
                EventKind.TRANSFER_INITIATED,
                EventKind.TRANSFER_ACCEPTED,
                EventKind.TRANSFER_POSTED,
            ),
            (
                EventKind.TRANSFER_INITIATED,
                EventKind.TRANSFER_ACCEPTED,
                EventKind.TRANSFER_POSTED,
                EventKind.TRANSFER_RETURNED,
            ),
            (
                EventKind.TRANSFER_INITIATED,
                EventKind.TRANSFER_ACCEPTED,
                EventKind.TRANSFER_POSTED,
                EventKind.FRAUD_REPORTED,
                EventKind.FUNDS_FROZEN,
                EventKind.RECOVERY,
            ),
        )
        legal_agentic = (
            (EventKind.AUTHORIZATION,),
            (EventKind.AUTHORIZATION_DECLINED,),
            (EventKind.AUTHENTICATION_CHALLENGE,),
        )
        legal = {Rail.CARD: legal_card_prefixes, Rail.A2A: legal_a2a, Rail.AGENTIC: legal_agentic}
        for lifecycle in grouped.values():
            first = lifecycle[0]
            sequence = tuple(event.event_type for event in lifecycle)
            if sequence not in legal[first.rail] or any(
                event.rail is not first.rail for event in lifecycle
            ):
                reasons.add("ILLEGAL_LIFECYCLE")
            if any(
                event.amount != first.amount
                or event.currency != first.currency
                or event.actor_id != first.actor_id
                or event.counterparty_id != first.counterparty_id
                for event in lifecycle
            ):
                reasons.add("VALUE_CONSERVATION_FAILED")
            previous = ""
            for index, event in enumerate(lifecycle):
                declared = event.lineage.get("previous_event_id", "")
                if index == 0 and declared:
                    reasons.add("LINEAGE_INVALID")
                if index > 0 and declared != previous:
                    reasons.add("LINEAGE_INVALID")
                previous = event.event_id

    @staticmethod
    def _check_roles(events: tuple[PaymentEvent, ...], reasons: set[str]) -> None:
        for event in events:
            actor_role = event.party_refs.get("actor_role")
            counterparty_role = event.party_refs.get("counterparty_role")
            if (
                type(actor_role) is not str
                or type(counterparty_role) is not str
                or (actor_role, counterparty_role) not in _ROLE_PAIRS[event.rail]
            ):
                reasons.add("ACTOR_ROLE_INVALID")

    @staticmethod
    def _opening(event: PaymentEvent, name: str) -> Decimal | None:
        raw = event.party_refs.get(name)
        if type(raw) is not str:
            return None
        try:
            value = Decimal(raw)
        except InvalidOperation:
            return None
        if not value.is_finite() or value < 0 or str(value) != raw:
            return None
        return value

    def _check_balances(self, events: tuple[PaymentEvent, ...], reasons: set[str]) -> None:
        balances: dict[str, Decimal] = {}
        declared_openings: dict[str, Decimal] = {}
        outstanding: dict[str, Decimal] = {}
        reversed_principal: set[str] = set()
        for event in events:
            actor_opening = self._opening(event, "actor_opening_balance")
            counterparty_opening = self._opening(event, "counterparty_opening_balance")
            if actor_opening is None or counterparty_opening is None:
                reasons.add("BALANCE_EVIDENCE_MISSING")
                continue
            for entity_id, opening in (
                (event.actor_id, actor_opening),
                (event.counterparty_id, counterparty_opening),
            ):
                if (
                    entity_id in declared_openings
                    and declared_openings[entity_id] != opening
                ):
                    reasons.add("BALANCE_EVIDENCE_INCONSISTENT")
                declared_openings.setdefault(entity_id, opening)
            balances.setdefault(event.actor_id, actor_opening)
            balances.setdefault(event.counterparty_id, counterparty_opening)
            payment_id = _payment_id(event)
            if payment_id is None:
                continue
            positive = event.event_type in _POSITIVE or (
                event.rail is Rail.AGENTIC and event.event_type is EventKind.AUTHORIZATION
            )
            if positive:
                if balances[event.actor_id] < event.amount:
                    reasons.add("BALANCE_INFEASIBLE")
                    continue
                balances[event.actor_id] -= event.amount
                balances[event.counterparty_id] += event.amount
                outstanding[payment_id] = event.amount
            elif event.event_type in _NEGATIVE and payment_id not in reversed_principal:
                amount = outstanding.get(payment_id, Decimal("0.00"))
                if amount != event.amount or balances[event.counterparty_id] < amount:
                    reasons.add("VALUE_CONSERVATION_FAILED")
                    continue
                balances[event.counterparty_id] -= amount
                balances[event.actor_id] += amount
                reversed_principal.add(payment_id)
        if any(balance < 0 for balance in balances.values()):
            reasons.add("BALANCE_INFEASIBLE")

    @staticmethod
    def _check_bounds(
        events: tuple[PaymentEvent, ...],
        grouped: dict[str, list[PaymentEvent]],
        reasons: set[str],
    ) -> None:
        if not 4 <= len(grouped) <= 128:
            reasons.add("PARAMETER_OUT_OF_BOUNDS")
        if any(
            event.currency != "USD"
            or event.amount <= 0
            or event.amount > Decimal("5000.00")
            or event.amount != event.amount.quantize(Decimal("0.01"))
            for event in events
        ):
            reasons.add("PARAMETER_OUT_OF_BOUNDS")
        if events[-1].event_time - events[0].event_time > timedelta(hours=72):
            reasons.add("PARAMETER_OUT_OF_BOUNDS")

    @staticmethod
    def _check_connectivity(events: tuple[PaymentEvent, ...], reasons: set[str]) -> None:
        attack_events = tuple(
            event for event in events if event.lineage.get("campaign_role") == "attack"
        )
        scoped_events = attack_events or events
        graph: dict[str, set[str]] = defaultdict(set)
        for event in scoped_events:
            graph[event.actor_id].add(event.counterparty_id)
            graph[event.counterparty_id].add(event.actor_id)
        if not graph:
            reasons.add("CAMPAIGN_DISCONNECTED")
            return
        start = min(graph)
        seen = {start}
        queue = deque([start])
        while queue:
            current = queue.popleft()
            for neighbor in graph[current]:
                if neighbor not in seen:
                    seen.add(neighbor)
                    queue.append(neighbor)
        if seen != set(graph):
            reasons.add("CAMPAIGN_DISCONNECTED")

    @staticmethod
    def _benign_distance(events: tuple[PaymentEvent, ...]) -> Decimal:
        openings = tuple(
            event
            for event in events
            if event.event_type
            in {
                EventKind.AUTHORIZATION,
                EventKind.AUTHORIZATION_DECLINED,
                EventKind.TRANSFER_INITIATED,
            }
        )
        if not openings:
            return Decimal("9.999")
        mean_amount = sum((event.amount for event in openings), Decimal(0)) / Decimal(
            len(openings)
        )
        gaps = tuple(
            (right.event_time - left.event_time).total_seconds()
            for left, right in zip(openings, openings[1:], strict=False)
        )
        mean_gap = sum(gaps) / len(gaps) if gaps else 30.0
        amount_component = abs(math.log1p(float(mean_amount)) - math.log1p(100.0)) / 4.0
        gap_component = abs(math.log1p(mean_gap) - math.log1p(30.0)) / 4.0
        role_pairs = {
            (event.party_refs.get("actor_role"), event.party_refs.get("counterparty_role"))
            for event in openings
        }
        diversity_component = 0.0 if len(role_pairs) >= 1 else 1.0
        return Decimal(str((amount_component + gap_component + diversity_component) / 3)).quantize(
            Decimal("0.001")
        )


__all__ = ["HiddenValidityOracle", "HiddenValidityResult", "RestrictedValidityReport"]
