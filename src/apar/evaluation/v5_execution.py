"""Validated real-execution evidence and decision-row projection for Sentinel v5."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from datetime import datetime
from decimal import Decimal
from enum import Enum
from types import MappingProxyType
from typing import cast

from apar.contracts.decisions import Action
from apar.contracts.events import EventKind, PaymentEvent, Rail
from apar.evaluation.v5_population import V5DecisionRow
from apar.generators.campaigns import CampaignEvidence
from apar.simulator.clock import Command
from apar.simulator.ledger import AccountReference, Ledger, LedgerEntry
from apar.simulator.rails import AgenticPaymentCommand
from apar.trust.verifier import (
    AgentPaymentRequest,
    AuthenticationEvidence,
    IntegrityReceipt,
    ReceiptOutcome,
    TrustCommitPlan,
    TrustVerifier,
)

_FRAUD_FAMILIES = frozenset(
    {
        "agentic_intent_abuse",
        "app_scam_mule",
        "card_testing_cnp",
        "synthetic_merchant_refund",
    }
)
_EXECUTION_FAMILIES = _FRAUD_FAMILIES | frozenset({"legitimate"})
_COMMAND_EVENT_KINDS: Mapping[str, EventKind] = MappingProxyType(
    {
        "a2a.initiate": EventKind.TRANSFER_INITIATED,
        "a2a.accept": EventKind.TRANSFER_ACCEPTED,
        "a2a.reject": EventKind.TRANSFER_REJECTED,
        "a2a.post": EventKind.TRANSFER_POSTED,
        "a2a.report": EventKind.FRAUD_REPORTED,
        "a2a.freeze": EventKind.FUNDS_FROZEN,
        "a2a.recover": EventKind.RECOVERY,
        "a2a.return": EventKind.TRANSFER_RETURNED,
        "card.authorize": EventKind.AUTHORIZATION,
        "card.decline": EventKind.AUTHORIZATION_DECLINED,
        "card.clear": EventKind.CLEARING,
        "card.settle": EventKind.SETTLEMENT,
        "card.reverse": EventKind.REVERSAL,
        "card.report": EventKind.FRAUD_REPORTED,
        "card.dispute": EventKind.DISPUTE_OPENED,
        "card.chargeback": EventKind.CHARGEBACK,
        "card.recover": EventKind.RECOVERY,
        "card.refund": EventKind.REFUND,
    }
)
_EVENT_LIFECYCLE: Mapping[EventKind, str] = MappingProxyType(
    {
        EventKind.AUTHORIZATION: "authorized",
        EventKind.AUTHORIZATION_DECLINED: "declined",
        EventKind.AUTHENTICATION_CHALLENGE: "challenged",
        EventKind.CLEARING: "cleared",
        EventKind.SETTLEMENT: "settled",
        EventKind.REVERSAL: "reversed",
        EventKind.TRANSFER_INITIATED: "initiated",
        EventKind.TRANSFER_ACCEPTED: "accepted",
        EventKind.TRANSFER_REJECTED: "rejected",
        EventKind.TRANSFER_POSTED: "posted",
        EventKind.TRANSFER_RETURNED: "returned",
        EventKind.FUNDS_FROZEN: "frozen",
        EventKind.REFUND: "refunded",
        EventKind.FRAUD_REPORTED: "reported",
        EventKind.DISPUTE_OPENED: "disputed",
        EventKind.CHARGEBACK: "chargeback",
        EventKind.RECOVERY: "recovered",
    }
)


@dataclass(frozen=True, slots=True)
class V5CommandEventLineage:
    """One exact generated-command to emitted-event association."""

    command_id: str
    command_name: str
    event_id: str
    campaign_id: str
    payment_id: str
    actor_id: str
    counterparty_id: str
    rail: Rail
    scheduled_at: datetime
    lifecycle_position: int
    is_fraud: bool


@dataclass(frozen=True, slots=True)
class V5AgenticVerifierEvidence:
    """Actual request, referenced authentication evidence, and verifier verdict."""

    command_id: str
    event_id: str
    request: AgentPaymentRequest
    authentication_evidence: AuthenticationEvidence | None
    receipt: IntegrityReceipt


@dataclass(frozen=True, slots=True)
class V5ExecutionEvidence:
    """Immutable evidence accepted only after full execution reconciliation."""

    family: str
    campaign_id: str
    rail: Rail
    commands: tuple[Command, ...]
    campaign_evidence: CampaignEvidence
    events: tuple[PaymentEvent, ...]
    ledger_entries: tuple[LedgerEntry, ...]
    opening_balances: tuple[tuple[AccountReference, Decimal], ...]
    lineage: tuple[V5CommandEventLineage, ...]
    trust_evidence: tuple[V5AgenticVerifierEvidence, ...]
    evidence_sha256: str

    def __post_init__(self) -> None:
        if type(self.opening_balances) is not tuple:
            raise TypeError("opening balances must be an exact tuple")
        opening_map = dict(self.opening_balances)
        canonical_opening = tuple(
            sorted(opening_map.items(), key=lambda item: str(item[0]))
        )
        if (
            len(opening_map) != len(self.opening_balances)
            or self.opening_balances != canonical_opening
        ):
            raise ValueError("opening ledger accounts must be unique and canonical")
        expected = _validate_and_derive(
            family=self.family,
            commands=self.commands,
            campaign_evidence=self.campaign_evidence,
            events=self.events,
            ledger_entries=self.ledger_entries,
            opening_balances=opening_map,
        )
        expected_rail, expected_lineage, expected_trust, expected_digest = expected
        if self.campaign_id != self.campaign_evidence.campaign_id:
            raise ValueError("campaign_id does not match campaign execution evidence")
        if self.rail is not expected_rail:
            raise ValueError("rail does not match generated commands")
        if self.lineage != expected_lineage:
            raise ValueError("command_id or command-to-event lineage was tampered")
        if self.trust_evidence != expected_trust:
            raise ValueError("agentic verifier evidence was tampered")
        if self.evidence_sha256 != expected_digest:
            raise ValueError("execution evidence digest was tampered")


def _canonical_value(value: object) -> object:
    if isinstance(value, Enum):
        return _canonical_value(value.value)
    if value is None or type(value) in (bool, int, str):
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError("command payload contains non-finite float")
        return {"float": repr(value)}
    if type(value) is Decimal:
        if not value.is_finite():
            raise ValueError("command payload contains non-finite amount")
        return {"decimal": str(value)}
    if type(value) is bytes:
        return {"bytes": value.hex()}
    if type(value) is datetime:
        return {"datetime": value.isoformat().replace("+00:00", "Z")}
    if type(value) is tuple:
        return [_canonical_value(item) for item in value]
    if isinstance(value, Mapping):
        canonical: dict[str, object] = {}
        for key, item in sorted(value.items()):
            if type(key) is not str:
                raise TypeError("command payload keys must be exact strings")
            canonical[key] = _canonical_value(item)
        return canonical
    raise TypeError(f"unsupported command payload value: {type(value).__name__}")


def _command_id_from_facts(
    *,
    command_type: str,
    command_name: str,
    command_payload: Mapping[str, object],
) -> str:
    """Derive one command ID from the complete canonical retained envelope."""
    if type(command_type) is not str or not command_type:
        raise ValueError("canonical command type must be a non-empty exact string")
    if type(command_name) is not str or not command_name:
        raise ValueError("canonical command name must be a non-empty exact string")
    document = {
        "domain": "apar.sentinel-v5.generated-command.v1",
        "type": command_type,
        "name": command_name,
        "payload": _canonical_value(command_payload),
    }
    encoded = json.dumps(
        document,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def _command_id(command: Command) -> str:
    if not isinstance(command, Command):
        raise TypeError("execution evidence commands must be generated Commands")
    return _command_id_from_facts(
        command_type=type(command).__qualname__,
        command_name=command.name,
        command_payload=command.payload,
    )


def _rail_for_command(command: Command) -> Rail:
    if command.name.startswith("card."):
        return Rail.CARD
    if command.name.startswith("a2a."):
        return Rail.A2A
    if command.name == "agentic.pay":
        return Rail.AGENTIC
    raise ValueError(f"unknown lifecycle command type: {command.name}")


def _event_payment_id(event: PaymentEvent) -> str:
    payment_id = event.rail_data.get("payment_id")
    if type(payment_id) is not str or not payment_id:
        raise ValueError("event payment_id is missing or invalid")
    return payment_id


def _opening_command(command: Command) -> bool:
    return command.name in {
        "a2a.initiate",
        "card.authorize",
        "card.decline",
        "agentic.pay",
    }


def _expected_kind(command: Command, event: PaymentEvent) -> EventKind:
    if command.name != "agentic.pay":
        try:
            return _COMMAND_EVENT_KINDS[command.name]
        except KeyError as error:
            raise ValueError(f"unknown lifecycle command type: {command.name}") from error
    integrity = event.rail_data.get("integrity")
    action = event.rail_data.get("action")
    if integrity == "fail" or action == Action.DECLINE.value:
        return EventKind.AUTHORIZATION_DECLINED
    if integrity == "pass" and action == Action.CHALLENGE.value:
        return EventKind.AUTHENTICATION_CHALLENGE
    if integrity == "pass" and action == Action.APPROVE.value:
        return EventKind.AUTHORIZATION
    raise ValueError("agentic event contains an unknown integrity/action lifecycle")


def _expected_ledger_entry(
    event: PaymentEvent,
    state: Mapping[str, object],
) -> LedgerEntry | None:
    amount = cast(Decimal, state["amount"])
    currency = cast(str, state["currency"])
    payer = cast(str, state["payer_account"])
    payee = cast(str, state["payee_account"])
    event_id = event.event_id

    if event.rail is Rail.CARD:
        fee = cast(Decimal, state["fee"])
        hold = cast(str, state["hold_account"])
        fee_account = cast(str, state["fee_account"])
        chargeback = cast(str, state["chargeback_account"])
        if event.event_type is EventKind.AUTHORIZATION:
            return LedgerEntry(f"{event_id}:hold", {payer: amount}, {hold: amount}, currency)
        if event.event_type is EventKind.REVERSAL:
            return LedgerEntry(f"{event_id}:release", {hold: amount}, {payer: amount}, currency)
        if event.event_type is EventKind.SETTLEMENT:
            return LedgerEntry(
                f"{event_id}:settle",
                {hold: amount},
                {payee: amount - fee, fee_account: fee},
                currency,
            )
        if event.event_type is EventKind.REFUND:
            return LedgerEntry(
                f"{event_id}:refund",
                {payee: amount - fee, fee_account: fee},
                {payer: amount},
                currency,
            )
        if event.event_type is EventKind.CHARGEBACK:
            return LedgerEntry(
                f"{event_id}:chargeback",
                {payee: amount - fee, fee_account: fee},
                {chargeback: amount},
                currency,
            )
        if event.event_type is EventKind.RECOVERY:
            return LedgerEntry(
                f"{event_id}:recovery",
                {chargeback: amount},
                {payer: amount},
                currency,
            )
        return None

    if event.rail is Rail.A2A:
        fee = cast(Decimal, state["fee"])
        fee_account = cast(str, state["fee_account"])
        frozen = cast(str, state["frozen_account"])
        if event.event_type is EventKind.TRANSFER_POSTED:
            return LedgerEntry(
                f"{event_id}:post",
                {payer: amount + fee},
                {payee: amount, fee_account: fee},
                currency,
            )
        if event.event_type is EventKind.TRANSFER_RETURNED:
            return LedgerEntry(f"{event_id}:return", {payee: amount}, {payer: amount}, currency)
        if event.event_type is EventKind.FUNDS_FROZEN:
            return LedgerEntry(f"{event_id}:freeze", {payee: amount}, {frozen: amount}, currency)
        if event.event_type is EventKind.RECOVERY:
            return LedgerEntry(
                f"{event_id}:recovery",
                {frozen: amount},
                {payer: amount},
                currency,
            )
        return None

    if event.rail is Rail.AGENTIC and event.event_type is EventKind.AUTHORIZATION:
        return LedgerEntry(
            f"{event_id}:agentic-payment",
            {payer: amount},
            {payee: amount},
            currency,
        )
    return None


def _validate_ledger(
    opening_balances: Mapping[AccountReference, Decimal],
    entries: tuple[LedgerEntry, ...],
    expected: tuple[LedgerEntry, ...],
) -> None:
    if entries != expected:
        raise ValueError("ledger entries do not reconcile to emitted events")
    try:
        replay = Ledger(opening_balances)
        for entry in entries:
            replay.post(entry)
        replay.assert_conserved()
    except (AssertionError, TypeError, ValueError) as error:
        raise ValueError("ledger replay or conservation failed") from error


def _validate_trust(
    campaign_evidence: CampaignEvidence,
    lineage: tuple[V5CommandEventLineage, ...],
    commands: tuple[Command, ...],
    events: tuple[PaymentEvent, ...],
) -> tuple[V5AgenticVerifierEvidence, ...]:
    fixture = campaign_evidence.agentic_fixture
    if fixture is None:
        raise ValueError("agentic execution is missing verifier inputs")
    verifier = TrustVerifier(
        registered_agents={(fixture.agent_id, fixture.key_id): fixture.public_key},
        mandates={fixture.mandate.mandate_id: fixture.mandate},
        authentication_evidence={
            evidence.evidence_id: evidence for evidence in fixture.authentication_evidence
        },
    )
    registry = {
        evidence.evidence_id: evidence for evidence in fixture.authentication_evidence
    }
    records: list[V5AgenticVerifierEvidence] = []
    for link, command, event in zip(lineage, commands, events, strict=True):
        if type(command) is not AgenticPaymentCommand:
            raise TypeError("agentic execution requires exact AgenticPaymentCommand values")
        request = command.request
        preview = verifier.preview(request, link.scheduled_at)
        receipt: IntegrityReceipt
        if preview.allowed:
            action_value = event.rail_data.get("action")
            try:
                outcome = {
                    Action.APPROVE.value: ReceiptOutcome.APPROVE,
                    Action.CHALLENGE.value: ReceiptOutcome.CHALLENGE,
                    Action.DECLINE.value: ReceiptOutcome.DECLINE,
                }[cast(str, action_value)]
            except KeyError as error:
                raise ValueError("agentic verifier event has unknown action") from error
            prepared = verifier.prepare_commit(request, preview, outcome, link.scheduled_at)
            if type(prepared) is IntegrityReceipt:
                receipt = prepared
            else:
                if type(prepared) is not TrustCommitPlan:
                    raise TypeError("TrustVerifier returned an unknown verdict")
                receipt = verifier.apply_commit(prepared)
        else:
            receipt = preview

        reason = receipt.reason_code.value if receipt.reason_code is not None else ""
        expected_fields = {
            "request_id": request.request_id,
            "integrity": "pass" if receipt.allowed else "fail",
            "reason_code": reason,
            "receipt_hash": receipt.receipt_hash,
            "receipt_outcome": receipt.outcome.value,
        }
        for field, expected_value in expected_fields.items():
            if event.rail_data.get(field) != expected_value:
                raise ValueError(f"agentic verifier {field} does not match emitted event")
        records.append(
            V5AgenticVerifierEvidence(
                command_id=link.command_id,
                event_id=link.event_id,
                request=request,
                authentication_evidence=registry.get(
                    request.authentication_evidence_ref or ""
                ),
                receipt=receipt,
            )
        )
    return tuple(records)


def _digest_evidence(
    *,
    family: str,
    campaign_id: str,
    rail: Rail,
    lineage: tuple[V5CommandEventLineage, ...],
    events: tuple[PaymentEvent, ...],
    ledger_entries: tuple[LedgerEntry, ...],
    opening_balances: Mapping[AccountReference, Decimal],
    trust_evidence: tuple[V5AgenticVerifierEvidence, ...],
) -> str:
    document = {
        "domain": "apar.sentinel-v5.execution-evidence.v1",
        "family": family,
        "campaign_id": campaign_id,
        "rail": rail.value,
        "lineage": [
            {
                **asdict(link),
                "rail": link.rail.value,
                "scheduled_at": link.scheduled_at.isoformat().replace("+00:00", "Z"),
            }
            for link in lineage
        ],
        "events": [event.model_dump(mode="json", warnings=False) for event in events],
        "ledger_entries": [
            {
                "entry_id": entry.entry_id,
                "debit": {key: str(value) for key, value in sorted(entry.debit.items())},
                "credit": {key: str(value) for key, value in sorted(entry.credit.items())},
                "currency": entry.currency,
            }
            for entry in ledger_entries
        ],
        "opening_balances": [
            {
                "account": account if type(account) is str else list(account),
                "amount": str(amount),
            }
            for account, amount in sorted(
                opening_balances.items(),
                key=lambda item: str(item[0]),
            )
        ],
        "trust": [
            {
                "command_id": record.command_id,
                "event_id": record.event_id,
                "request_id": record.request.request_id,
                "authentication_evidence_id": (
                    record.authentication_evidence.evidence_id
                    if record.authentication_evidence is not None
                    else None
                ),
                "authentication_evidence": (
                    _canonical_value(asdict(record.authentication_evidence))
                    if record.authentication_evidence is not None
                    else None
                ),
                "receipt_hash": record.receipt.receipt_hash,
                "allowed": record.receipt.allowed,
                "reason_code": (
                    record.receipt.reason_code.value
                    if record.receipt.reason_code is not None
                    else None
                ),
                "outcome": record.receipt.outcome.value,
            }
            for record in trust_evidence
        ],
    }
    encoded = json.dumps(
        document,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _validate_and_derive(
    *,
    family: str,
    commands: tuple[Command, ...],
    campaign_evidence: CampaignEvidence,
    events: tuple[PaymentEvent, ...],
    ledger_entries: tuple[LedgerEntry, ...],
    opening_balances: Mapping[AccountReference, Decimal],
) -> tuple[
    Rail,
    tuple[V5CommandEventLineage, ...],
    tuple[V5AgenticVerifierEvidence, ...],
    str,
]:
    if type(family) is not str or family not in _EXECUTION_FAMILIES:
        raise ValueError("family is not part of the configured ground-truth contract")
    if type(campaign_evidence) is not CampaignEvidence:
        raise TypeError("campaign_evidence must be exact Task 5 evidence")
    if campaign_evidence.family != family:
        raise ValueError("family does not match campaign ground truth")
    if family == "legitimate" and any(campaign_evidence.class_labels):
        raise ValueError("legitimate execution evidence cannot contain fraud labels")
    if type(commands) is not tuple or not commands:
        raise ValueError("execution commands must be a non-empty exact tuple")
    if type(events) is not tuple or len(events) != len(commands):
        raise ValueError("lifecycle requires exactly one emitted event per generated command")
    if type(ledger_entries) is not tuple:
        raise TypeError("ledger entries must be an exact tuple")
    if len(campaign_evidence.schedule) != len(commands):
        raise ValueError("lifecycle schedule does not match generated commands")
    rails = {_rail_for_command(command) for command in commands}
    if len(rails) != 1:
        raise ValueError("one execution evidence contract cannot span multiple rails")
    rail = next(iter(rails))
    if any(event.rail is not rail for event in events):
        raise ValueError("event rail does not match generated commands")

    opening_commands = tuple(command for command in commands if _opening_command(command))
    if len(opening_commands) != len(campaign_evidence.class_labels):
        raise ValueError("ground-truth labels do not match payment openings")
    labels_by_payment = {
        cast(str, command.payload["payment_id"]): label
        for command, label in zip(
            opening_commands,
            campaign_evidence.class_labels,
            strict=True,
        )
    }
    if len(labels_by_payment) != len(opening_commands):
        raise ValueError("ground-truth payment IDs are duplicated")

    seen_commands: set[str] = set()
    seen_events: set[str] = set()
    payment_state: dict[str, Mapping[str, object]] = {}
    previous_event: dict[str, str] = {}
    lifecycle_positions: dict[str, int] = {}
    lineage: list[V5CommandEventLineage] = []
    expected_ledger: list[LedgerEntry] = []

    for command, scheduled_at, event in zip(
        commands,
        campaign_evidence.schedule,
        events,
        strict=True,
    ):
        command_id = _command_id(command)
        if command_id in seen_commands:
            raise ValueError("duplicate canonical command_id in execution evidence")
        if event.event_id in seen_events:
            raise ValueError("duplicate event_id in execution evidence")
        seen_commands.add(command_id)
        seen_events.add(event.event_id)
        if event.event_time != scheduled_at:
            raise ValueError("event order or scheduled lifecycle timestamp differs")
        if event.campaign_id != campaign_evidence.campaign_id:
            raise ValueError("event campaign_id does not match command campaign_id")
        command_campaign = getattr(command, "campaign_id", None)
        if command_campaign != campaign_evidence.campaign_id:
            raise ValueError("command campaign_id does not match ground truth")
        payment_id = cast(str, command.payload.get("payment_id"))
        if not payment_id:
            raise ValueError("command payment_id is missing")
        if _event_payment_id(event) != payment_id:
            raise ValueError("event payment_id does not match source command")
        if _opening_command(command):
            payment_state[payment_id] = command.payload
        try:
            state = payment_state[payment_id]
        except KeyError as error:
            raise ValueError("lifecycle event has no opening source lineage") from error

        expected_kind = _expected_kind(command, event)
        if event.event_type is not expected_kind:
            raise ValueError("event lifecycle type does not match generated command")
        for field in ("actor_id", "counterparty_id", "trace_id"):
            if getattr(event, field) != state.get(field):
                raise ValueError(f"event {field} does not match source command")
        amount = state.get("amount")
        currency = state.get("currency")
        if (
            type(amount) is not Decimal
            or not amount.is_finite()
            or amount <= 0
            or event.amount != amount
        ):
            raise ValueError("event amount is invalid or does not reconcile")
        if type(currency) is not str or not currency or event.currency != currency:
            raise ValueError("event currency is invalid or does not reconcile")
        if event.decision_at is None or event.decision_at != event.available_at:
            raise ValueError("event decision timestamp is missing or non-canonical")

        expected_previous = previous_event.get(payment_id, "")
        actual_previous = event.lineage.get("previous_event_id", "")
        if actual_previous != expected_previous:
            raise ValueError("event lifecycle source lineage is missing or out of order")
        position = lifecycle_positions.get(payment_id, 0)
        lifecycle_positions[payment_id] = position + 1
        previous_event[payment_id] = event.event_id
        ledger_entry = _expected_ledger_entry(event, state)
        if ledger_entry is not None:
            expected_ledger.append(ledger_entry)
        lineage.append(
            V5CommandEventLineage(
                command_id=command_id,
                command_name=command.name,
                event_id=event.event_id,
                campaign_id=event.campaign_id,
                payment_id=payment_id,
                actor_id=event.actor_id,
                counterparty_id=event.counterparty_id,
                rail=rail,
                scheduled_at=scheduled_at,
                lifecycle_position=position,
                is_fraud=labels_by_payment[payment_id],
            )
        )

    _validate_ledger(opening_balances, ledger_entries, tuple(expected_ledger))
    lineage_tuple = tuple(lineage)
    if rail is Rail.AGENTIC:
        trust = _validate_trust(campaign_evidence, lineage_tuple, commands, events)
    else:
        if campaign_evidence.agentic_fixture is not None:
            raise ValueError("non-agentic execution unexpectedly contains verifier inputs")
        trust = ()
    digest = _digest_evidence(
        family=family,
        campaign_id=campaign_evidence.campaign_id,
        rail=rail,
        lineage=lineage_tuple,
        events=events,
        ledger_entries=ledger_entries,
        opening_balances=opening_balances,
        trust_evidence=trust,
    )
    return rail, lineage_tuple, trust, digest


def build_execution_evidence(
    *,
    family: str,
    commands: tuple[Command, ...],
    campaign_evidence: CampaignEvidence,
    events: tuple[PaymentEvent, ...],
    ledger_entries: tuple[LedgerEntry, ...],
    opening_balances: Mapping[AccountReference, Decimal],
) -> V5ExecutionEvidence:
    """Validate real commands, events, ledger, labels, and trust into one contract."""
    rail, lineage, trust, digest = _validate_and_derive(
        family=family,
        commands=commands,
        campaign_evidence=campaign_evidence,
        events=events,
        ledger_entries=ledger_entries,
        opening_balances=opening_balances,
    )
    frozen_opening = tuple(sorted(opening_balances.items(), key=lambda item: str(item[0])))
    return V5ExecutionEvidence(
        family=family,
        campaign_id=campaign_evidence.campaign_id,
        rail=rail,
        commands=commands,
        campaign_evidence=campaign_evidence,
        events=events,
        ledger_entries=ledger_entries,
        opening_balances=frozen_opening,
        lineage=lineage,
        trust_evidence=trust,
        evidence_sha256=digest,
    )


def project_execution_evidence(evidence: V5ExecutionEvidence) -> list[V5DecisionRow]:
    """Project only a fully validated execution-evidence contract into decision rows."""
    if type(evidence) is not V5ExecutionEvidence:
        raise TypeError("projection requires exact V5ExecutionEvidence")
    trust_by_event = {record.event_id: record for record in evidence.trust_evidence}
    rows: list[V5DecisionRow] = []
    for event, link in zip(evidence.events, evidence.lineage, strict=True):
        try:
            lifecycle_state = _EVENT_LIFECYCLE[event.event_type]
        except KeyError as error:
            raise ValueError("unknown lifecycle/event type") from error
        trust_record = trust_by_event.get(event.event_id)
        if event.rail is Rail.AGENTIC:
            if trust_record is None:
                raise ValueError("agentic event is missing verifier result")
            integrity_status = "pass" if trust_record.receipt.allowed else "fail"
            integrity_pass = float(trust_record.receipt.allowed)
        else:
            if trust_record is not None:
                raise ValueError("non-agentic event contains verifier result")
            integrity_status = "not_applicable"
            integrity_pass = 0.0
        decision_at = event.decision_at
        if decision_at is None:
            raise ValueError("event is missing decision timestamp")
        hour = decision_at.hour
        rows.append(
            V5DecisionRow(
                event_id=event.event_id,
                payment_id=link.payment_id,
                campaign_id=event.campaign_id,
                family=evidence.family,
                actor_id=event.actor_id,
                counterparty_id=event.counterparty_id,
                amount=event.amount,
                currency=event.currency,
                decision_at=decision_at,
                is_fraud=link.is_fraud,
                rail=event.rail.value,
                integrity_status=integrity_status,
                lifecycle_state=lifecycle_state,
                source_command_id=link.command_id,
                source_event_id=event.event_id,
                execution_evidence_sha256=evidence.evidence_sha256,
                predictive_features={
                    "amount": float(event.amount),
                    "rail_card": float(event.rail is Rail.CARD),
                    "rail_a2a": float(event.rail is Rail.A2A),
                    "rail_agentic": float(event.rail is Rail.AGENTIC),
                    "integrity_pass": integrity_pass,
                    "txn_hour_sin": math.sin(2 * math.pi * hour / 24),
                    "txn_hour_cos": math.cos(2 * math.pi * hour / 24),
                },
            )
        )
    return rows


__all__ = [
    "V5AgenticVerifierEvidence",
    "V5CommandEventLineage",
    "V5ExecutionEvidence",
    "build_execution_evidence",
    "project_execution_evidence",
]
