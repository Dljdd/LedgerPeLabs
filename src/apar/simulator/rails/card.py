"""Deterministic card authorization-to-recovery lifecycle adapter."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from decimal import ROUND_HALF_EVEN, Decimal
from enum import StrEnum
from types import MappingProxyType
from typing import ClassVar, cast
from uuid import UUID

from apar.contracts.events import EventKind, PaymentEvent, Rail
from apar.simulator.clock import Command
from apar.simulator.ledger import LedgerEntry
from apar.simulator.rails.base import FrozenState, LifecycleError, RailContext

_EXPONENTS = {"EUR": 2, "JPY": 0, "KWD": 3, "USD": 2}
_MAPPING_PROXY_TYPE: type[object] = type(MappingProxyType({}))


class CardState(StrEnum):
    """Internal card state, separate from the frozen public lifecycle vocabulary."""

    CREATED = "created"
    AUTHORIZED = "authorized"
    DECLINED = "declined"
    CLEARED = "cleared"
    SETTLED = "settled"
    REPORTED = "reported"
    DISPUTE = "dispute"
    CHARGEBACK = "chargeback"
    RECOVERY = "recovery"
    REVERSED = "reversed"
    REFUNDED = "refunded"


def _text(label: str, value: object) -> str:
    if type(value) is not str:
        raise TypeError(f"{label} must be an exact string")
    if not value:
        raise ValueError(f"{label} must not be empty")
    return value


def _uuid_text(label: str, value: object) -> str:
    text = _text(label, value)
    try:
        UUID(text)
    except ValueError as error:
        raise ValueError(f"{label} must be a UUID string") from error
    return text


def _money(label: str, value: object, currency: str, *, positive: bool) -> Decimal:
    if type(value) is not Decimal:
        raise TypeError(f"{label} must be an exact Decimal")
    amount = value
    if not amount.is_finite() or (amount <= 0 if positive else amount < 0):
        qualifier = "positive" if positive else "non-negative"
        raise ValueError(f"{label} must be finite and {qualifier}")
    try:
        exponent = _EXPONENTS[currency]
    except KeyError as error:
        raise ValueError(f"unsupported currency: {currency}") from error
    quantized = amount.quantize(
        Decimal(1).scaleb(-exponent),
        rounding=ROUND_HALF_EVEN,
    )
    if positive and quantized <= 0:
        raise ValueError(f"{label} must be finite and positive")
    return quantized


def _require_distinct_accounts(*accounts: str) -> None:
    if len(set(accounts)) != len(accounts):
        raise ValueError("card account roles must be pairwise distinct")


def _payload_fingerprint(payload: Mapping[str, object]) -> str:
    tagged: list[list[str]] = []
    for key in sorted(payload):
        if type(key) is not str:
            raise TypeError("command payload keys must be exact strings")
        value = payload[key]
        if type(value) is str:
            tagged.append([key, "str", value])
        elif type(value) is Decimal:
            tagged.append([key, "decimal", str(value)])
        else:
            raise TypeError("card command payload contains unsupported value type")
    encoded = json.dumps(tagged, separators=(",", ":"), ensure_ascii=True).encode()
    return hashlib.sha256(encoded).hexdigest()


class CardCommand(Command):
    """Immutable public command base exposing campaign metadata to generators."""

    __slots__ = ("_operation_identity", "_request_fingerprint")

    _operation_identity: str
    _request_fingerprint: str

    def _seal(self, operation_identity: str) -> None:
        object.__setattr__(self, "_operation_identity", operation_identity)
        object.__setattr__(self, "_request_fingerprint", _payload_fingerprint(self.payload))

    @property
    def payment_id(self) -> str:
        return cast(str, self.payload["payment_id"])

    @property
    def idempotency_key(self) -> str:
        return cast(str, self.payload["idempotency_key"])

    @property
    def campaign_id(self) -> str:
        return cast(str, self.payload.get("campaign_id", ""))


class _OpenCardCommand(CardCommand):
    __slots__ = ()
    _NAME: ClassVar[str]

    def __init__(
        self,
        payment_id: str,
        *,
        amount: Decimal,
        currency: str,
        payer_account: str,
        payee_account: str,
        actor_id: str,
        counterparty_id: str,
        campaign_id: str,
        trace_id: str,
        fee: Decimal = Decimal("0"),
        hold_account: str = "card:holds",
        fee_account: str = "card:fees",
        chargeback_account: str = "card:chargebacks",
        idempotency_key: str | None = None,
    ) -> None:
        checked_currency = _text("currency", currency)
        checked_amount = _money("amount", amount, checked_currency, positive=True)
        checked_fee = _money("fee", fee, checked_currency, positive=False)
        if checked_fee > checked_amount:
            raise ValueError("fee must not exceed amount")
        checked_payment_id = _text("payment_id", payment_id)
        checked_payer = _text("payer_account", payer_account)
        checked_payee = _text("payee_account", payee_account)
        checked_hold = _text("hold_account", hold_account)
        checked_fee_account = _text("fee_account", fee_account)
        checked_chargeback = _text("chargeback_account", chargeback_account)
        _require_distinct_accounts(
            checked_payer,
            checked_payee,
            checked_hold,
            checked_fee_account,
            checked_chargeback,
        )
        super().__init__(
            self._NAME,
            {
                "payment_id": checked_payment_id,
                "amount": checked_amount,
                "currency": checked_currency,
                "payer_account": checked_payer,
                "payee_account": checked_payee,
                "actor_id": _uuid_text("actor_id", actor_id),
                "counterparty_id": _uuid_text("counterparty_id", counterparty_id),
                "campaign_id": _uuid_text("campaign_id", campaign_id),
                "trace_id": _uuid_text("trace_id", trace_id),
                "fee": checked_fee,
                "hold_account": checked_hold,
                "fee_account": checked_fee_account,
                "chargeback_account": checked_chargeback,
                "idempotency_key": _text(
                    "idempotency_key",
                    idempotency_key
                    if idempotency_key is not None
                    else f"{self._NAME}:{checked_payment_id}",
                ),
            },
        )
        self._seal(self._NAME)


class AuthorizeCard(_OpenCardCommand):
    """Request an approved card authorization and place an unsettled hold."""

    _NAME = "card.authorize"


class DeclineCardAuthorization(_OpenCardCommand):
    """Record an observable issuer decline without moving value."""

    _NAME = "card.decline"

    @classmethod
    def from_authorization(cls, command: AuthorizeCard) -> DeclineCardAuthorization:
        payload = command.payload
        return cls(
            command.payment_id,
            amount=cast(Decimal, payload["amount"]),
            currency=cast(str, payload["currency"]),
            payer_account=cast(str, payload["payer_account"]),
            payee_account=cast(str, payload["payee_account"]),
            actor_id=cast(str, payload["actor_id"]),
            counterparty_id=cast(str, payload["counterparty_id"]),
            campaign_id=cast(str, payload["campaign_id"]),
            trace_id=cast(str, payload["trace_id"]),
            fee=cast(Decimal, payload["fee"]),
            hold_account=cast(str, payload["hold_account"]),
            fee_account=cast(str, payload["fee_account"]),
            chargeback_account=cast(str, payload["chargeback_account"]),
            idempotency_key=f"card.decline:{command.payment_id}",
        )


class _FollowupCardCommand(CardCommand):
    __slots__ = ()
    _NAME: ClassVar[str]

    def __init__(self, payment_id: str, *, idempotency_key: str | None = None) -> None:
        checked_payment_id = _text("payment_id", payment_id)
        super().__init__(
            self._NAME,
            {
                "payment_id": checked_payment_id,
                "idempotency_key": _text(
                    "idempotency_key",
                    idempotency_key
                    if idempotency_key is not None
                    else f"{self._NAME}:{checked_payment_id}",
                ),
            },
        )
        self._seal(self._NAME)


class ClearCard(_FollowupCardCommand):
    _NAME = "card.clear"


class SettleCard(_FollowupCardCommand):
    _NAME = "card.settle"


class ReverseCardAuthorization(_FollowupCardCommand):
    _NAME = "card.reverse"


class ReportCardFraud(_FollowupCardCommand):
    _NAME = "card.report"


class OpenCardDispute(_FollowupCardCommand):
    _NAME = "card.dispute"


class ChargebackCard(_FollowupCardCommand):
    _NAME = "card.chargeback"


class RecoverCard(_FollowupCardCommand):
    _NAME = "card.recover"


class RefundCard(_FollowupCardCommand):
    _NAME = "card.refund"


type _Transition = tuple[CardState, EventKind, str]

_TRANSITIONS: Mapping[tuple[CardState, type[Command]], _Transition] = {
    (CardState.CREATED, AuthorizeCard): (
        CardState.AUTHORIZED,
        EventKind.AUTHORIZATION,
        "hold",
    ),
    (CardState.CREATED, DeclineCardAuthorization): (
        CardState.DECLINED,
        EventKind.AUTHORIZATION_DECLINED,
        "none",
    ),
    (CardState.AUTHORIZED, ClearCard): (CardState.CLEARED, EventKind.CLEARING, "none"),
    (CardState.AUTHORIZED, ReverseCardAuthorization): (
        CardState.REVERSED,
        EventKind.REVERSAL,
        "release_hold",
    ),
    (CardState.CLEARED, ReverseCardAuthorization): (
        CardState.REVERSED,
        EventKind.REVERSAL,
        "release_hold",
    ),
    (CardState.CLEARED, SettleCard): (CardState.SETTLED, EventKind.SETTLEMENT, "settle"),
    (CardState.SETTLED, ReportCardFraud): (
        CardState.REPORTED,
        EventKind.FRAUD_REPORTED,
        "none",
    ),
    (CardState.SETTLED, RefundCard): (CardState.REFUNDED, EventKind.REFUND, "refund"),
    (CardState.REPORTED, OpenCardDispute): (
        CardState.DISPUTE,
        EventKind.DISPUTE_OPENED,
        "none",
    ),
    (CardState.DISPUTE, ChargebackCard): (
        CardState.CHARGEBACK,
        EventKind.CHARGEBACK,
        "chargeback",
    ),
    (CardState.CHARGEBACK, RecoverCard): (
        CardState.RECOVERY,
        EventKind.RECOVERY,
        "recovery",
    ),
}

_MISSING_CODES: Mapping[type[Command], str] = {
    ClearCard: "CARD_CLEAR_BEFORE_AUTHORIZE",
    SettleCard: "CARD_SETTLE_BEFORE_CLEAR",
    ReverseCardAuthorization: "CARD_REVERSE_BEFORE_AUTHORIZE",
    ReportCardFraud: "CARD_REPORT_BEFORE_SETTLEMENT",
    OpenCardDispute: "CARD_DISPUTE_BEFORE_REPORT",
    ChargebackCard: "CARD_CHARGEBACK_BEFORE_DISPUTE",
    RecoverCard: "CARD_RECOVERY_BEFORE_CHARGEBACK",
    RefundCard: "CARD_REFUND_BEFORE_SETTLEMENT",
}

_COMMAND_OPERATIONS: Mapping[type[Command], str] = {
    AuthorizeCard: "card.authorize",
    DeclineCardAuthorization: "card.decline",
    ClearCard: "card.clear",
    SettleCard: "card.settle",
    ReverseCardAuthorization: "card.reverse",
    ReportCardFraud: "card.report",
    OpenCardDispute: "card.dispute",
    ChargebackCard: "card.chargeback",
    RecoverCard: "card.recover",
    RefundCard: "card.refund",
}
_KNOWN_COMMAND_TYPES = frozenset(_COMMAND_OPERATIONS)
_OPEN_PAYLOAD_KEYS = frozenset(
    {
        "payment_id",
        "amount",
        "currency",
        "payer_account",
        "payee_account",
        "actor_id",
        "counterparty_id",
        "campaign_id",
        "trace_id",
        "fee",
        "hold_account",
        "fee_account",
        "chargeback_account",
        "idempotency_key",
    }
)
_FOLLOWUP_PAYLOAD_KEYS = frozenset({"payment_id", "idempotency_key"})


def _canonical_command(command: Command) -> CardCommand:
    command_type = type(command)
    if not isinstance(command, CardCommand) or command_type not in _KNOWN_COMMAND_TYPES:
        raise LifecycleError("CARD_UNKNOWN_COMMAND")
    operation = _COMMAND_OPERATIONS[command_type]
    try:
        if type(command.name) is not str or command.name != operation:
            raise ValueError("command name does not match concrete operation")
        if type(command.payload) is not _MAPPING_PROXY_TYPE:
            raise TypeError("command payload must be an owned immutable mapping")
        payload = command.payload
        expected_keys = (
            _OPEN_PAYLOAD_KEYS
            if command_type in {AuthorizeCard, DeclineCardAuthorization}
            else _FOLLOWUP_PAYLOAD_KEYS
        )
        if set(payload) != expected_keys:
            raise ValueError("command payload fields do not match concrete operation")
        incoming_fingerprint = _payload_fingerprint(payload)
        if command._operation_identity != operation:
            raise ValueError("command operation attestation does not match")
        if command._request_fingerprint != incoming_fingerprint:
            raise ValueError("command payload attestation does not match")

        if command_type in {AuthorizeCard, DeclineCardAuthorization}:
            opening_type = cast(type[_OpenCardCommand], command_type)
            canonical: CardCommand = opening_type(
                cast(str, payload["payment_id"]),
                amount=cast(Decimal, payload["amount"]),
                currency=cast(str, payload["currency"]),
                payer_account=cast(str, payload["payer_account"]),
                payee_account=cast(str, payload["payee_account"]),
                actor_id=cast(str, payload["actor_id"]),
                counterparty_id=cast(str, payload["counterparty_id"]),
                campaign_id=cast(str, payload["campaign_id"]),
                trace_id=cast(str, payload["trace_id"]),
                fee=cast(Decimal, payload["fee"]),
                hold_account=cast(str, payload["hold_account"]),
                fee_account=cast(str, payload["fee_account"]),
                chargeback_account=cast(str, payload["chargeback_account"]),
                idempotency_key=cast(str, payload["idempotency_key"]),
            )
        else:
            followup_type = cast(type[_FollowupCardCommand], command_type)
            canonical = followup_type(
                cast(str, payload["payment_id"]),
                idempotency_key=cast(str, payload["idempotency_key"]),
            )
        if canonical._request_fingerprint != incoming_fingerprint:
            raise ValueError("command payload is not canonical")
        return canonical
    except (AttributeError, KeyError, TypeError, ValueError) as error:
        raise LifecycleError("CARD_COMMAND_INVALID") from error


def _state_value(
    state: Mapping[str, FrozenState],
    key: str,
    expected: type[object],
) -> object:
    value = state[key]
    if type(value) is not expected:
        raise LifecycleError("CARD_STATE_CORRUPT")
    return value


_STATE_KEYS = frozenset(
    {
        "state",
        "payment_id",
        "amount",
        "currency",
        "payer_account",
        "payee_account",
        "actor_id",
        "counterparty_id",
        "campaign_id",
        "trace_id",
        "fee",
        "hold_account",
        "fee_account",
        "chargeback_account",
        "last_event_id",
        "idempotency_records",
    }
)


def _validated_state(value: object) -> Mapping[str, FrozenState]:
    try:
        if type(value) not in (dict, _MAPPING_PROXY_TYPE):
            raise TypeError("state must be an owned mapping")
        state = cast(Mapping[str, FrozenState], value)
        if set(state) != _STATE_KEYS or any(type(key) is not str for key in state):
            raise ValueError("state fields do not match card schema")

        CardState(_text("state", state["state"]))
        _text("payment_id", state["payment_id"])
        currency = _text("currency", state["currency"])
        amount_value = state["amount"]
        fee_value = state["fee"]
        if type(amount_value) is not Decimal or type(fee_value) is not Decimal:
            raise TypeError("state money must be exact Decimal")
        amount = amount_value
        fee = fee_value
        if amount.as_tuple() != _money(
            "amount", amount, currency, positive=True
        ).as_tuple():
            raise ValueError("state amount must be canonically quantized")
        if fee.as_tuple() != _money("fee", fee, currency, positive=False).as_tuple():
            raise ValueError("state fee must be canonically quantized")
        if fee > amount:
            raise ValueError("state fee must not exceed amount")

        payer = _text("payer_account", state["payer_account"])
        payee = _text("payee_account", state["payee_account"])
        hold = _text("hold_account", state["hold_account"])
        fee_account = _text("fee_account", state["fee_account"])
        chargeback = _text("chargeback_account", state["chargeback_account"])
        _require_distinct_accounts(payer, payee, hold, fee_account, chargeback)
        _uuid_text("actor_id", state["actor_id"])
        _uuid_text("counterparty_id", state["counterparty_id"])
        _uuid_text("campaign_id", state["campaign_id"])
        _uuid_text("trace_id", state["trace_id"])

        last_event_id = state["last_event_id"]
        if type(last_event_id) is not str:
            raise TypeError("last_event_id must be an exact string")
        if last_event_id:
            _uuid_text("last_event_id", last_event_id)

        records = state["idempotency_records"]
        if type(records) is not tuple:
            raise TypeError("idempotency_records must be an exact tuple")
        seen_keys: set[str] = set()
        operations = frozenset(_COMMAND_OPERATIONS.values())
        for record in records:
            if type(record) is not tuple or len(record) != 3:
                raise ValueError("idempotency record must contain three fields")
            key, operation, fingerprint = record
            if type(key) is not str or not key:
                raise ValueError("idempotency record key must be a non-empty string")
            if type(operation) is not str or operation not in operations:
                raise ValueError("idempotency record operation is invalid")
            if (
                type(fingerprint) is not str
                or len(fingerprint) != 64
                or any(character not in "0123456789abcdef" for character in fingerprint)
            ):
                raise ValueError("idempotency record fingerprint is invalid")
            if key in seen_keys:
                raise ValueError("idempotency record keys must be unique")
            seen_keys.add(key)
        return state
    except (KeyError, TypeError, ValueError) as error:
        raise LifecycleError("CARD_STATE_CORRUPT") from error


def _opening_state(command: _OpenCardCommand) -> dict[str, object]:
    payload = command.payload
    return {
        "state": CardState.CREATED.value,
        "payment_id": command.payment_id,
        "amount": payload["amount"],
        "currency": payload["currency"],
        "payer_account": payload["payer_account"],
        "payee_account": payload["payee_account"],
        "actor_id": payload["actor_id"],
        "counterparty_id": payload["counterparty_id"],
        "campaign_id": payload["campaign_id"],
        "trace_id": payload["trace_id"],
        "fee": payload["fee"],
        "hold_account": payload["hold_account"],
        "fee_account": payload["fee_account"],
        "chargeback_account": payload["chargeback_account"],
        "last_event_id": "",
        "idempotency_records": (),
    }


def _invalid_code(current: CardState, command_type: type[Command]) -> str:
    if command_type is ReverseCardAuthorization and current in {
        CardState.SETTLED,
        CardState.REPORTED,
        CardState.DISPUTE,
        CardState.CHARGEBACK,
        CardState.RECOVERY,
    }:
        return "CARD_REVERSE_AFTER_SETTLEMENT"
    return _MISSING_CODES.get(command_type, "CARD_INVALID_TRANSITION")


def _is_idempotent_retry(
    state: Mapping[str, FrozenState],
    command: CardCommand,
) -> tuple[bool, tuple[FrozenState, ...]]:
    records = cast(
        tuple[FrozenState, ...],
        _state_value(state, "idempotency_records", tuple),
    )
    operation = _COMMAND_OPERATIONS[type(command)]
    fingerprint = command._request_fingerprint
    for record in records:
        key, recorded_operation, recorded_fingerprint = cast(
            tuple[str, str, str],
            record,
        )
        if key == command.idempotency_key:
            if recorded_operation == operation and recorded_fingerprint == fingerprint:
                return True, records
            raise LifecycleError("CARD_IDEMPOTENCY_KEY_COLLISION")
    return False, records


def _post_effect(
    context: RailContext,
    state: Mapping[str, FrozenState],
    effect: str,
    event_id: str,
) -> None:
    amount = cast(Decimal, _state_value(state, "amount", Decimal))
    fee = cast(Decimal, _state_value(state, "fee", Decimal))
    currency = cast(str, _state_value(state, "currency", str))
    payer = cast(str, _state_value(state, "payer_account", str))
    payee = cast(str, _state_value(state, "payee_account", str))
    hold = cast(str, _state_value(state, "hold_account", str))
    fee_account = cast(str, _state_value(state, "fee_account", str))
    chargeback = cast(str, _state_value(state, "chargeback_account", str))
    if effect == "none":
        return
    if effect == "hold":
        entry = LedgerEntry(f"{event_id}:hold", {payer: amount}, {hold: amount}, currency)
    elif effect == "release_hold":
        entry = LedgerEntry(f"{event_id}:release", {hold: amount}, {payer: amount}, currency)
    elif effect == "settle":
        entry = LedgerEntry(
            f"{event_id}:settle",
            {hold: amount},
            {payee: amount - fee, fee_account: fee},
            currency,
        )
    elif effect == "refund":
        entry = LedgerEntry(
            f"{event_id}:refund",
            {payee: amount - fee, fee_account: fee},
            {payer: amount},
            currency,
        )
    elif effect == "chargeback":
        entry = LedgerEntry(
            f"{event_id}:chargeback",
            {payee: amount - fee, fee_account: fee},
            {chargeback: amount},
            currency,
        )
    elif effect == "recovery":
        entry = LedgerEntry(
            f"{event_id}:recovery",
            {chargeback: amount},
            {payer: amount},
            currency,
        )
    else:
        raise AssertionError(f"unknown card ledger effect: {effect}")
    context.post(entry)


def _event(
    context: RailContext,
    state: Mapping[str, FrozenState],
    kind: EventKind,
    event_id: str,
    previous_event_id: str,
) -> PaymentEvent:
    lineage: dict[str, str | bool] = {"synthetic": True}
    if previous_event_id:
        lineage["previous_event_id"] = previous_event_id
    return PaymentEvent(
        schema_version="1.0.0",
        event_id=event_id,
        campaign_id=cast(str, state["campaign_id"]),
        trace_id=cast(str, state["trace_id"]),
        rail=Rail.CARD,
        viewpoint=context.bundle.viewpoint,
        event_type=kind,
        amount=cast(Decimal, state["amount"]),
        currency=cast(str, state["currency"]),
        event_time=context.now,
        ingested_at=context.now,
        available_at=context.now,
        decision_at=context.now,
        actor_id=cast(str, state["actor_id"]),
        counterparty_id=cast(str, state["counterparty_id"]),
        rail_data={
            "payment_id": cast(str, state["payment_id"]),
            "lifecycle_state": cast(str, state["state"]),
        },
        lineage=lineage,
        privacy={"classification": "synthetic"},
    )


class CardRailAdapter:
    """Execute legal card transitions through a restricted rail context."""

    def initialize(self, context: RailContext) -> None:
        if context.bundle.rail is not Rail.CARD:
            raise ValueError("card adapter requires a card scenario")

    def handle(self, command: Command, context: RailContext) -> list[PaymentEvent]:
        canonical = _canonical_command(command)
        payment_id = canonical.payment_id
        try:
            raw_state: object = context.entity_state(payment_id)
        except KeyError:
            if type(canonical) not in {AuthorizeCard, DeclineCardAuthorization}:
                raise LifecycleError(
                    _MISSING_CODES.get(type(canonical), "CARD_INVALID_TRANSITION")
                ) from None
            raw_state = _opening_state(cast(_OpenCardCommand, canonical))
        state = _validated_state(raw_state)

        is_retry, records = _is_idempotent_retry(state, canonical)
        if is_retry:
            return []
        current = CardState(cast(str, _state_value(state, "state", str)))
        try:
            next_state, kind, effect = _TRANSITIONS[(current, type(canonical))]
        except KeyError as error:
            raise LifecycleError(_invalid_code(current, type(canonical))) from error

        event_id = context.new_uuid()
        previous_event_id = cast(str, _state_value(state, "last_event_id", str))
        updated = dict(state)
        updated["state"] = next_state.value
        updated["last_event_id"] = event_id
        updated["idempotency_records"] = (
            *records,
            (
                canonical.idempotency_key,
                _COMMAND_OPERATIONS[type(canonical)],
                canonical._request_fingerprint,
            ),
        )
        validated_updated = _validated_state(updated)
        prospective_event = _event(
            context,
            validated_updated,
            kind,
            event_id,
            previous_event_id,
        )
        _post_effect(context, validated_updated, effect, event_id)
        context.set_entity_state(payment_id, validated_updated)
        return [prospective_event]


__all__ = [
    "AuthorizeCard",
    "CardCommand",
    "CardRailAdapter",
    "CardState",
    "ChargebackCard",
    "ClearCard",
    "DeclineCardAuthorization",
    "LifecycleError",
    "OpenCardDispute",
    "RecoverCard",
    "RefundCard",
    "ReportCardFraud",
    "ReverseCardAuthorization",
    "SettleCard",
]
