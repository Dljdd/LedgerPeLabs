"""Deterministic account-to-account initiation-to-recovery lifecycle adapter."""

from __future__ import annotations

from collections.abc import Mapping
from decimal import ROUND_HALF_EVEN, Decimal
from enum import StrEnum
from typing import ClassVar, cast
from uuid import UUID

from apar.contracts.events import EventKind, PaymentEvent, Rail
from apar.simulator.clock import Command
from apar.simulator.ledger import LedgerEntry
from apar.simulator.rails.base import FrozenState, LifecycleError, RailContext

_EXPONENTS = {"EUR": 2, "JPY": 0, "KWD": 3, "USD": 2}


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
        raise ValueError("A2A account roles must be pairwise distinct")


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

_KNOWN_COMMAND_TYPES = frozenset(
    {
        InitiateA2A,
        AcceptA2A,
        RejectA2A,
        PostA2A,
        ReportA2AFraud,
        FreezeA2AFunds,
        RecoverA2A,
        ReturnA2A,
    }
)


def _mapping_state(value: FrozenState) -> Mapping[str, FrozenState]:
    if not isinstance(value, Mapping):
        raise LifecycleError("A2A_STATE_CORRUPT")
    return value


def _state_value(
    state: Mapping[str, FrozenState],
    key: str,
    expected: type[object],
) -> object:
    value = state[key]
    if type(value) is not expected:
        raise LifecycleError("A2A_STATE_CORRUPT")
    return value


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
    for record in records:
        if type(record) is not tuple or len(record) != 2:
            raise LifecycleError("A2A_STATE_CORRUPT")
        key, command_name = record
        if type(key) is not str or type(command_name) is not str:
            raise LifecycleError("A2A_STATE_CORRUPT")
        if key == command.idempotency_key:
            if command_name == command.name:
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
        if not isinstance(command, A2ACommand) or type(command) not in _KNOWN_COMMAND_TYPES:
            raise LifecycleError("A2A_UNKNOWN_COMMAND")
        payment_id = command.payment_id
        try:
            state = _mapping_state(context.entity_state(payment_id))
        except KeyError:
            if not isinstance(command, InitiateA2A):
                raise LifecycleError(
                    _MISSING_CODES.get(type(command), "A2A_INVALID_TRANSITION")
                ) from None
            state = _mapping_state(cast(FrozenState, _opening_state(command)))

        is_retry, records = _is_idempotent_retry(state, command)
        if is_retry:
            return []
        current = A2AState(cast(str, _state_value(state, "state", str)))
        try:
            next_state, kind, effect = _TRANSITIONS[(current, type(command))]
        except KeyError as error:
            raise LifecycleError(_invalid_code(current, type(command))) from error

        event_id = context.new_uuid()
        previous_event_id = cast(str, _state_value(state, "last_event_id", str))
        _post_effect(context, state, effect, event_id)
        updated = dict(state)
        updated["state"] = next_state.value
        updated["last_event_id"] = event_id
        updated["idempotency_records"] = (
            *records,
            (command.idempotency_key, command.name),
        )
        context.set_entity_state(payment_id, updated)
        return [_event(context, updated, kind, event_id, previous_event_id)]


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
