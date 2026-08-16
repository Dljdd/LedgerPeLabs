"""Deterministic account-to-account initiation-to-recovery lifecycle adapter."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from decimal import ROUND_HALF_EVEN, Decimal, DecimalException
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


class A2AState(StrEnum):
    """Internal A2A state, separate from the frozen public lifecycle vocabulary."""

    CREATED = "created"
    INITIATED = "initiated"
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    POSTED = "posted"
    REPORTED = "reported"
    FROZEN = "frozen"
    RECOVERED = "recovered"
    RETURNED = "returned"


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


def _history_event_position(value: object) -> tuple[str, bytes, int]:
    event_id = _text("history event_id", value)
    try:
        identifier = UUID(event_id)
    except ValueError as error:
        raise ValueError("history event_id must be a UUID string") from error
    if str(identifier) != event_id or identifier.version != 4:
        raise ValueError("history event_id must be a canonical version-4 UUID")
    return event_id, identifier.bytes[:10], int.from_bytes(identifier.bytes[10:], "big")


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
    try:
        quantized = amount.quantize(
            Decimal(1).scaleb(-exponent),
            rounding=ROUND_HALF_EVEN,
        )
    except DecimalException as error:
        raise ValueError(f"{label} cannot be represented in {currency}") from error
    if positive and quantized <= 0:
        raise ValueError(f"{label} must be finite and positive")
    return quantized


def _require_distinct_accounts(*accounts: str) -> None:
    if len(set(accounts)) != len(accounts):
        raise ValueError("A2A account roles must be pairwise distinct")


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
            raise TypeError("A2A command payload contains unsupported value type")
    encoded = json.dumps(tagged, separators=(",", ":"), ensure_ascii=True).encode()
    return hashlib.sha256(encoded).hexdigest()


class A2ACommand(Command):
    """Immutable public command base exposing campaign metadata to generators."""

    __slots__ = ()

    @property
    def payment_id(self) -> str:
        return cast(str, self.payload["payment_id"])

    @property
    def idempotency_key(self) -> str:
        return cast(str, self.payload["idempotency_key"])

    @property
    def campaign_id(self) -> str:
        return cast(str, self.payload.get("campaign_id", ""))


class InitiateA2A(A2ACommand):
    """Open an A2A transfer without moving value before posting."""

    __slots__ = ()

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
        fee_account: str = "a2a:fees",
        frozen_account: str = "a2a:frozen",
        idempotency_key: str | None = None,
    ) -> None:
        checked_currency = _text("currency", currency)
        checked_amount = _money("amount", amount, checked_currency, positive=True)
        checked_fee = _money("fee", fee, checked_currency, positive=False)
        checked_payment_id = _text("payment_id", payment_id)
        checked_payer = _text("payer_account", payer_account)
        checked_payee = _text("payee_account", payee_account)
        checked_fee_account = _text("fee_account", fee_account)
        checked_frozen = _text("frozen_account", frozen_account)
        _require_distinct_accounts(
            checked_payer,
            checked_payee,
            checked_fee_account,
            checked_frozen,
        )
        super().__init__(
            "a2a.initiate",
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
                "fee_account": checked_fee_account,
                "frozen_account": checked_frozen,
                "idempotency_key": _text(
                    "idempotency_key",
                    idempotency_key
                    if idempotency_key is not None
                    else f"a2a.initiate:{checked_payment_id}",
                ),
            },
        )


class _FollowupA2ACommand(A2ACommand):
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


class AcceptA2A(_FollowupA2ACommand):
    _NAME = "a2a.accept"


class RejectA2A(_FollowupA2ACommand):
    _NAME = "a2a.reject"


class PostA2A(_FollowupA2ACommand):
    _NAME = "a2a.post"


class ReportA2AFraud(_FollowupA2ACommand):
    _NAME = "a2a.report"


class FreezeA2AFunds(_FollowupA2ACommand):
    _NAME = "a2a.freeze"


class RecoverA2A(_FollowupA2ACommand):
    _NAME = "a2a.recover"


class ReturnA2A(_FollowupA2ACommand):
    _NAME = "a2a.return"


type _Transition = tuple[A2AState, EventKind, str]

_TRANSITIONS: Mapping[tuple[A2AState, type[Command]], _Transition] = {
    (A2AState.CREATED, InitiateA2A): (
        A2AState.INITIATED,
        EventKind.TRANSFER_INITIATED,
        "none",
    ),
    (A2AState.INITIATED, AcceptA2A): (
        A2AState.ACCEPTED,
        EventKind.TRANSFER_ACCEPTED,
        "none",
    ),
    (A2AState.INITIATED, RejectA2A): (
        A2AState.REJECTED,
        EventKind.TRANSFER_REJECTED,
        "none",
    ),
    (A2AState.ACCEPTED, PostA2A): (
        A2AState.POSTED,
        EventKind.TRANSFER_POSTED,
        "post",
    ),
    (A2AState.POSTED, ReportA2AFraud): (
        A2AState.REPORTED,
        EventKind.FRAUD_REPORTED,
        "none",
    ),
    (A2AState.POSTED, ReturnA2A): (
        A2AState.RETURNED,
        EventKind.TRANSFER_RETURNED,
        "return",
    ),
    (A2AState.REPORTED, FreezeA2AFunds): (
        A2AState.FROZEN,
        EventKind.FUNDS_FROZEN,
        "freeze",
    ),
    (A2AState.FROZEN, RecoverA2A): (
        A2AState.RECOVERED,
        EventKind.RECOVERY,
        "recovery",
    ),
}

_MISSING_CODES: Mapping[type[Command], str] = {
    AcceptA2A: "A2A_ACCEPT_BEFORE_INITIATE",
    RejectA2A: "A2A_REJECT_BEFORE_INITIATE",
    PostA2A: "A2A_POST_BEFORE_ACCEPT",
    ReportA2AFraud: "A2A_REPORT_BEFORE_POST",
    FreezeA2AFunds: "A2A_FREEZE_BEFORE_REPORT",
    RecoverA2A: "A2A_RECOVERY_BEFORE_FREEZE",
    ReturnA2A: "A2A_RETURN_BEFORE_POST",
}

_COMMAND_OPERATIONS: Mapping[type[Command], str] = {
    InitiateA2A: "a2a.initiate",
    AcceptA2A: "a2a.accept",
    RejectA2A: "a2a.reject",
    PostA2A: "a2a.post",
    ReportA2AFraud: "a2a.report",
    FreezeA2AFunds: "a2a.freeze",
    RecoverA2A: "a2a.recover",
    ReturnA2A: "a2a.return",
}
_KNOWN_COMMAND_TYPES = frozenset(_COMMAND_OPERATIONS)
_OPERATION_TYPES = {
    operation: command_type
    for command_type, operation in _COMMAND_OPERATIONS.items()
}
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
        "fee_account",
        "frozen_account",
        "idempotency_key",
    }
)
_FOLLOWUP_PAYLOAD_KEYS = frozenset({"payment_id", "idempotency_key"})


def _canonical_command(command: Command) -> A2ACommand:
    command_type = type(command)
    if not isinstance(command, A2ACommand) or command_type not in _KNOWN_COMMAND_TYPES:
        raise LifecycleError("A2A_UNKNOWN_COMMAND")
    operation = _COMMAND_OPERATIONS[command_type]
    try:
        if type(command.name) is not str or command.name != operation:
            raise ValueError("command name does not match concrete operation")
        if type(command.payload) is not _MAPPING_PROXY_TYPE:
            raise TypeError("command payload must be an owned immutable mapping")
        payload = dict(command.payload)
        expected_keys = (
            _OPEN_PAYLOAD_KEYS
            if command_type is InitiateA2A
            else _FOLLOWUP_PAYLOAD_KEYS
        )
        if set(payload) != expected_keys:
            raise ValueError("command payload fields do not match concrete operation")
        incoming_fingerprint = _payload_fingerprint(payload)

        if command_type is InitiateA2A:
            canonical: A2ACommand = InitiateA2A(
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
                fee_account=cast(str, payload["fee_account"]),
                frozen_account=cast(str, payload["frozen_account"]),
                idempotency_key=cast(str, payload["idempotency_key"]),
            )
        else:
            followup_type = cast(type[_FollowupA2ACommand], command_type)
            canonical = followup_type(
                cast(str, payload["payment_id"]),
                idempotency_key=cast(str, payload["idempotency_key"]),
            )
        if _payload_fingerprint(canonical.payload) != incoming_fingerprint:
            raise ValueError("command payload is not canonical")
        return canonical
    except (
        AttributeError,
        DecimalException,
        KeyError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as error:
        raise LifecycleError("A2A_COMMAND_INVALID") from error


def _state_value(
    state: Mapping[str, FrozenState],
    key: str,
    expected: type[object],
) -> object:
    value = state[key]
    if type(value) is not expected:
        raise LifecycleError("A2A_STATE_CORRUPT")
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
        "fee_account",
        "frozen_account",
        "last_event_id",
        "idempotency_records",
    }
)


def _command_from_record(
    operation: str,
    idempotency_key: str,
    state: Mapping[str, FrozenState],
) -> A2ACommand:
    command_type = _OPERATION_TYPES[operation]
    if command_type is InitiateA2A:
        return InitiateA2A(
            cast(str, state["payment_id"]),
            amount=cast(Decimal, state["amount"]),
            currency=cast(str, state["currency"]),
            payer_account=cast(str, state["payer_account"]),
            payee_account=cast(str, state["payee_account"]),
            actor_id=cast(str, state["actor_id"]),
            counterparty_id=cast(str, state["counterparty_id"]),
            campaign_id=cast(str, state["campaign_id"]),
            trace_id=cast(str, state["trace_id"]),
            fee=cast(Decimal, state["fee"]),
            fee_account=cast(str, state["fee_account"]),
            frozen_account=cast(str, state["frozen_account"]),
            idempotency_key=idempotency_key,
        )
    followup_type = cast(type[_FollowupA2ACommand], command_type)
    return followup_type(
        cast(str, state["payment_id"]),
        idempotency_key=idempotency_key,
    )


def _validated_state(
    value: object,
    expected_payment_id: str,
) -> Mapping[str, FrozenState]:
    try:
        if type(value) not in (dict, _MAPPING_PROXY_TYPE):
            raise TypeError("state must be an owned mapping")
        state = cast(Mapping[str, FrozenState], value)
        if set(state) != _STATE_KEYS or any(type(key) is not str for key in state):
            raise ValueError("state fields do not match A2A schema")

        stored_state = A2AState(_text("state", state["state"]))
        payment_id = _text("payment_id", state["payment_id"])
        if payment_id != expected_payment_id:
            raise ValueError("state payment_id does not match entity key")
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

        payer = _text("payer_account", state["payer_account"])
        payee = _text("payee_account", state["payee_account"])
        fee_account = _text("fee_account", state["fee_account"])
        frozen = _text("frozen_account", state["frozen_account"])
        _require_distinct_accounts(payer, payee, fee_account, frozen)
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
        seen_event_ids: set[str] = set()
        event_prefix: bytes | None = None
        previous_event_sequence = -1
        operations = frozenset(_COMMAND_OPERATIONS.values())
        replayed_state = A2AState.CREATED
        for index, record in enumerate(records):
            if type(record) is not tuple or len(record) != 4:
                raise ValueError("idempotency record must contain four fields")
            key, operation, fingerprint, record_event_id = record
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
            checked_event_id, current_prefix, current_sequence = _history_event_position(
                record_event_id
            )
            if checked_event_id in seen_event_ids:
                raise ValueError("idempotency record event IDs must be unique")
            seen_event_ids.add(checked_event_id)
            if event_prefix is None:
                event_prefix = current_prefix
            elif current_prefix != event_prefix:
                raise ValueError("idempotency record event IDs must share an engine prefix")
            if current_sequence <= previous_event_sequence:
                raise ValueError("idempotency record event IDs must increase")
            previous_event_sequence = current_sequence
            command = _command_from_record(operation, key, state)
            if _payload_fingerprint(command.payload) != fingerprint:
                raise ValueError("idempotency record fingerprint does not match request")
            command_type = type(command)
            if index == 0 and command_type is not InitiateA2A:
                raise ValueError("A2A history must begin with initiation")
            if index > 0 and command_type is InitiateA2A:
                raise ValueError("A2A history contains a repeated initiation")
            try:
                replayed_state = _TRANSITIONS[(replayed_state, command_type)][0]
            except KeyError as error:
                raise ValueError("A2A history contains an illegal transition") from error
        if records:
            final_record = cast(tuple[str, str, str, str], records[-1])
            if last_event_id != final_record[3]:
                raise ValueError("last_event_id does not match final A2A history record")
        elif last_event_id:
            raise ValueError("last_event_id presence does not match A2A history")
        if replayed_state is not stored_state:
            raise ValueError("stored A2A state does not match replayed history")
        return state
    except (DecimalException, KeyError, RuntimeError, TypeError, ValueError) as error:
        raise LifecycleError("A2A_STATE_CORRUPT") from error


def _opening_state(command: InitiateA2A) -> dict[str, object]:
    payload = command.payload
    return {
        "state": A2AState.CREATED.value,
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
        "fee_account": payload["fee_account"],
        "frozen_account": payload["frozen_account"],
        "last_event_id": "",
        "idempotency_records": (),
    }


def _invalid_code(current: A2AState, command_type: type[Command]) -> str:
    if command_type is RejectA2A and current in {
        A2AState.ACCEPTED,
        A2AState.POSTED,
        A2AState.REPORTED,
        A2AState.FROZEN,
        A2AState.RECOVERED,
    }:
        return "A2A_REJECT_AFTER_ACCEPT"
    return _MISSING_CODES.get(command_type, "A2A_INVALID_TRANSITION")


def _is_idempotent_retry(
    state: Mapping[str, FrozenState],
    command: A2ACommand,
) -> tuple[bool, tuple[FrozenState, ...]]:
    records = cast(
        tuple[FrozenState, ...],
        _state_value(state, "idempotency_records", tuple),
    )
    operation = _COMMAND_OPERATIONS[type(command)]
    fingerprint = _payload_fingerprint(command.payload)
    for record in records:
        key, recorded_operation, recorded_fingerprint, _ = cast(
            tuple[str, str, str, str],
            record,
        )
        if key == command.idempotency_key:
            if recorded_operation == operation and recorded_fingerprint == fingerprint:
                return True, records
            raise LifecycleError("A2A_IDEMPOTENCY_KEY_COLLISION")
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
    fee_account = cast(str, _state_value(state, "fee_account", str))
    frozen = cast(str, _state_value(state, "frozen_account", str))
    if effect == "none":
        return
    if effect == "post":
        entry = LedgerEntry(
            f"{event_id}:post",
            {payer: amount + fee},
            {payee: amount, fee_account: fee},
            currency,
        )
    elif effect == "return":
        entry = LedgerEntry(f"{event_id}:return", {payee: amount}, {payer: amount}, currency)
    elif effect == "freeze":
        entry = LedgerEntry(f"{event_id}:freeze", {payee: amount}, {frozen: amount}, currency)
    elif effect == "recovery":
        entry = LedgerEntry(
            f"{event_id}:recovery",
            {frozen: amount},
            {payer: amount},
            currency,
        )
    else:
        raise AssertionError(f"unknown A2A ledger effect: {effect}")
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
        rail=Rail.A2A,
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


class A2ARailAdapter:
    """Execute legal A2A transitions through a restricted rail context."""

    def initialize(self, context: RailContext) -> None:
        if context.bundle.rail is not Rail.A2A:
            raise ValueError("A2A adapter requires an A2A scenario")

    def handle(self, command: Command, context: RailContext) -> list[PaymentEvent]:
        canonical = _canonical_command(command)
        payment_id = canonical.payment_id
        try:
            raw_state: object = context.entity_state(payment_id)
        except KeyError:
            if type(canonical) is not InitiateA2A:
                raise LifecycleError(
                    _MISSING_CODES.get(type(canonical), "A2A_INVALID_TRANSITION")
                ) from None
            raw_state = _opening_state(canonical)
        state = _validated_state(raw_state, payment_id)

        is_retry, records = _is_idempotent_retry(state, canonical)
        if is_retry:
            return []
        current = A2AState(cast(str, _state_value(state, "state", str)))
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
                _payload_fingerprint(canonical.payload),
                event_id,
            ),
        )
        validated_updated = _validated_state(updated, payment_id)
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
    "A2ACommand",
    "A2ARailAdapter",
    "A2AState",
    "AcceptA2A",
    "FreezeA2AFunds",
    "InitiateA2A",
    "LifecycleError",
    "PostA2A",
    "RecoverA2A",
    "RejectA2A",
    "ReportA2AFraud",
    "ReturnA2A",
]
