"""Executable A2A lifecycle, accounting, and event-contract tests."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from datetime import timedelta
from decimal import Decimal
from types import MappingProxyType
from typing import cast

import pytest

from apar.compiler.compiler import compile_scenario
from apar.contracts.events import EventKind, Rail
from apar.simulator.clock import Command
from apar.simulator.engine import SimulationEngine, SimulationFailedError
from apar.simulator.rails.a2a import (
    A2ACommand,
    A2ARailAdapter,
    AcceptA2A,
    FreezeA2AFunds,
    InitiateA2A,
    LifecycleError,
    PostA2A,
    RecoverA2A,
    RejectA2A,
    ReportA2AFraud,
    ReturnA2A,
)
from apar.simulator.rails.base import FrozenState
from tests.factories import NOW, make_scenario_config, make_threat_card

PAYMENT_ID = "a2a-payment-1"
CAMPAIGN_ID = "00000000-0000-4000-8000-000000000201"
TRACE_ID = "00000000-0000-4000-8000-000000000202"
PAYER_ID = "00000000-0000-4000-8000-000000000203"
PAYEE_ID = "00000000-0000-4000-8000-000000000204"


def _a2a_bundle(*, seed: int = 260_816):  # type: ignore[no-untyped-def]
    config = make_scenario_config(
        seed=seed,
        replay=make_scenario_config().replay.model_copy(update={"random_seed": seed}),
    )
    return compile_scenario(make_threat_card(default_config=config), config)


def _engine(
    *,
    seed: int = 260_816,
    payer_balance: Decimal = Decimal("100.00"),
    currency: str = "USD",
) -> SimulationEngine:
    return SimulationEngine(
        _a2a_bundle(seed=seed),
        {Rail.A2A: A2ARailAdapter},
        opening_balances={("payer", currency): payer_balance},
    )


def _initiate(
    *,
    amount: Decimal = Decimal("10.00"),
    fee: Decimal = Decimal("0.00"),
    currency: str = "USD",
    idempotency_key: str = "a2a-initiate-1",
    payer_account: str = "payer",
    payee_account: str = "payee",
    fee_account: str = "a2a:fees",
    frozen_account: str = "a2a:frozen",
) -> InitiateA2A:
    return InitiateA2A(
        PAYMENT_ID,
        amount=amount,
        currency=currency,
        payer_account=payer_account,
        payee_account=payee_account,
        actor_id=PAYER_ID,
        counterparty_id=PAYEE_ID,
        campaign_id=CAMPAIGN_ID,
        trace_id=TRACE_ID,
        fee=fee,
        fee_account=fee_account,
        frozen_account=frozen_account,
        idempotency_key=idempotency_key,
    )


def _schedule(engine: SimulationEngine, *commands: Command) -> None:
    for priority, command in enumerate(commands):
        engine.schedule(NOW, priority, command)


def test_a2a_commands_are_public_immutable_command_subclasses() -> None:
    """Catch rail builders bypassing the engine's immutable Command boundary."""
    command = _initiate()

    assert isinstance(command, Command)
    assert command.campaign_id == CAMPAIGN_ID
    assert command.payload["amount"] == Decimal("10.00")


def test_a2a_default_posting_moves_value_once() -> None:
    """Catch acceptance posting early or posting debiting the sender twice."""
    engine = _engine()
    _schedule(engine, _initiate(), AcceptA2A(PAYMENT_ID), PostA2A(PAYMENT_ID))

    events = engine.run()

    assert [event.event_type.value for event in events] == [
        "transfer_initiated",
        "transfer_accepted",
        "transfer_posted",
    ]
    assert engine.ledger.balance("payer") == Decimal("90.00")
    assert engine.ledger.balance("payee") == Decimal("10.00")
    assert len(engine.ledger.entries) == 1
    engine.ledger.assert_conserved()


def test_a2a_full_recovery_path_emits_linked_events_and_conserves_value() -> None:
    """Catch a missing A2A step, mutation, wrong freeze, or broken lineage."""
    engine = _engine()
    _schedule(
        engine,
        _initiate(fee=Decimal("1.00")),
        AcceptA2A(PAYMENT_ID),
        PostA2A(PAYMENT_ID),
        ReportA2AFraud(PAYMENT_ID),
        FreezeA2AFunds(PAYMENT_ID),
        RecoverA2A(PAYMENT_ID),
    )

    events = engine.run()

    assert [event.event_type for event in events] == [
        EventKind.TRANSFER_INITIATED,
        EventKind.TRANSFER_ACCEPTED,
        EventKind.TRANSFER_POSTED,
        EventKind.FRAUD_REPORTED,
        EventKind.FUNDS_FROZEN,
        EventKind.RECOVERY,
    ]
    assert [event.lineage.get("previous_event_id") for event in events] == [
        None,
        *[event.event_id for event in events[:-1]],
    ]
    assert all(event.rail is Rail.A2A for event in events)
    assert all(event.viewpoint == "network_with_bank_enrichment" for event in events)
    assert all(event.campaign_id == CAMPAIGN_ID for event in events)
    assert all(event.trace_id == TRACE_ID for event in events)
    assert all(event.actor_id == PAYER_ID for event in events)
    assert all(event.counterparty_id == PAYEE_ID for event in events)
    assert engine.ledger.balance("payer") == Decimal("99.00")
    assert engine.ledger.balance("payee") == Decimal("0.00")
    assert engine.ledger.balance("a2a:fees") == Decimal("1.00")
    assert engine.ledger.balance("a2a:frozen") == Decimal("0.00")
    engine.ledger.assert_conserved()


def test_a2a_rejection_is_preposting_terminal_branch_without_ledger() -> None:
    """Catch rejected transfers moving value or coexisting with posting."""
    engine = _engine()
    _schedule(engine, _initiate(), RejectA2A(PAYMENT_ID))

    events = engine.run()

    assert [event.event_type for event in events] == [
        EventKind.TRANSFER_INITIATED,
        EventKind.TRANSFER_REJECTED,
    ]
    assert engine.ledger.balance("payer") == Decimal("100.00")
    assert engine.ledger.balance("payee") == Decimal("0.00")
    assert engine.ledger.entries == ()


def test_a2a_return_is_separate_posting_branch_and_retains_fee() -> None:
    """Catch returns mutating the posted event or silently refunding a fee."""
    engine = _engine()
    _schedule(
        engine,
        _initiate(fee=Decimal("1.00")),
        AcceptA2A(PAYMENT_ID),
        PostA2A(PAYMENT_ID),
        ReturnA2A(PAYMENT_ID),
    )

    events = engine.run()

    assert [event.event_type for event in events] == [
        EventKind.TRANSFER_INITIATED,
        EventKind.TRANSFER_ACCEPTED,
        EventKind.TRANSFER_POSTED,
        EventKind.TRANSFER_RETURNED,
    ]
    assert engine.ledger.balance("payer") == Decimal("99.00")
    assert engine.ledger.balance("payee") == Decimal("0.00")
    assert engine.ledger.balance("a2a:fees") == Decimal("1.00")
    engine.ledger.assert_conserved()


def test_a2a_fee_is_explicit_and_duplicate_post_is_noop() -> None:
    """Catch hidden fee loss or duplicate idempotency posting value twice."""
    engine = _engine()
    post = PostA2A(PAYMENT_ID, idempotency_key="post-once")
    _schedule(engine, _initiate(fee=Decimal("1.00")), AcceptA2A(PAYMENT_ID), post, post)

    events = engine.run()

    assert [event.event_type for event in events].count(EventKind.TRANSFER_POSTED) == 1
    assert engine.ledger.balance("payer") == Decimal("89.00")
    assert engine.ledger.balance("payee") == Decimal("10.00")
    assert engine.ledger.balance("a2a:fees") == Decimal("1.00")
    assert len(engine.ledger.entries) == 1
    engine.ledger.assert_conserved()


def test_a2a_idempotency_key_collision_with_different_command_fails_closed() -> None:
    """Catch one key silently suppressing a different A2A lifecycle command."""
    engine = _engine()
    _schedule(
        engine,
        _initiate(idempotency_key="shared-key"),
        AcceptA2A(PAYMENT_ID, idempotency_key="shared-key"),
    )

    with pytest.raises(LifecycleError) as error:
        engine.run()

    assert error.value.code == "A2A_IDEMPOTENCY_KEY_COLLISION"
    assert engine.ledger.entries == ()
    assert engine.ledger.balance("payer") == Decimal("100.00")


def test_a2a_duplicate_constructor_with_identical_request_is_noop() -> None:
    """Catch request fingerprints depending on object identity rather than payload."""
    engine = _engine()
    _schedule(engine, _initiate(idempotency_key="same"), _initiate(idempotency_key="same"))

    events = engine.run()

    assert len(events) == 1
    assert engine.ledger.entries == ()


@pytest.mark.parametrize(
    "second",
    [
        lambda: _initiate(idempotency_key="same", amount=Decimal("11.00")),
        lambda: _initiate(idempotency_key="same", fee=Decimal("1.00")),
        lambda: _initiate(idempotency_key="same", payee_account="payee-2"),
        lambda: InitiateA2A(
            PAYMENT_ID,
            amount=Decimal("10.00"),
            currency="USD",
            payer_account="payer",
            payee_account="payee",
            actor_id="00000000-0000-4000-8000-000000000999",
            counterparty_id=PAYEE_ID,
            campaign_id=CAMPAIGN_ID,
            trace_id=TRACE_ID,
            idempotency_key="same",
        ),
    ],
)
def test_a2a_same_key_changed_opening_request_collides(
    second: Callable[[], InitiateA2A],
) -> None:
    """Catch amount, fee, account, or identity changes masquerading as retries."""
    engine = _engine()
    _schedule(engine, _initiate(idempotency_key="same"), second())

    with pytest.raises(LifecycleError) as error:
        engine.run()

    assert error.value.code == "A2A_IDEMPOTENCY_KEY_COLLISION"
    assert engine.ledger.entries == ()


def test_a2a_same_command_same_key_remains_complete_noop() -> None:
    """Catch collision detection breaking legitimate A2A retry idempotency."""
    engine = _engine()
    command = _initiate(idempotency_key="same-request")
    _schedule(engine, command, command)

    events = engine.run()

    assert [event.event_type for event in events] == [EventKind.TRANSFER_INITIATED]
    assert engine.ledger.entries == ()


def test_a2a_acceptance_does_not_move_value_before_posting() -> None:
    """Catch acceptance being conflated with irrevocable posting."""
    engine = _engine()
    engine.schedule(NOW, 0, _initiate())
    engine.schedule(NOW + timedelta(seconds=1), 0, AcceptA2A(PAYMENT_ID))
    engine.schedule(NOW + timedelta(seconds=2), 0, PostA2A(PAYMENT_ID))

    engine.run(until=NOW + timedelta(seconds=1))

    assert engine.ledger.balance("payer") == Decimal("100.00")
    assert engine.ledger.balance("payee") == Decimal("0.00")
    assert engine.ledger.entries == ()


def test_a2a_rejects_overdraft_atomically() -> None:
    """Catch posting bypassing the ledger's no-overdraft invariant."""
    engine = _engine(payer_balance=Decimal("5.00"))
    _schedule(engine, _initiate(), AcceptA2A(PAYMENT_ID), PostA2A(PAYMENT_ID))

    with pytest.raises(ValueError, match="overdraw account: payer"):
        engine.run()

    assert engine.ledger.balance("payer") == Decimal("5.00")
    assert engine.ledger.entries == ()
    with pytest.raises(SimulationFailedError):
        engine.run()


def test_a2a_quantizes_supported_currency_before_events_and_postings() -> None:
    """Catch payment events disagreeing with exponent-aware ledger amounts."""
    engine = _engine(currency="JPY", payer_balance=Decimal("100"))
    _schedule(
        engine,
        _initiate(amount=Decimal("10.5"), currency="JPY"),
        AcceptA2A(PAYMENT_ID),
        PostA2A(PAYMENT_ID),
    )

    events = engine.run()

    assert all(event.amount == Decimal("10") for event in events)
    assert engine.ledger.balance("payer", "JPY") == Decimal("90")
    assert engine.ledger.balance("payee", "JPY") == Decimal("10")
    engine.ledger.assert_conserved()


@pytest.mark.parametrize(
    ("currency", "amount"),
    [
        ("USD", Decimal("0.004")),
        ("JPY", Decimal("0.4")),
        ("KWD", Decimal("0.0004")),
    ],
)
def test_a2a_rejects_principal_that_quantizes_to_zero(
    currency: str,
    amount: Decimal,
) -> None:
    """Catch a nominally positive transfer becoming a zero-value posting."""
    with pytest.raises(ValueError, match="amount must be finite and positive"):
        _initiate(currency=currency, amount=amount)


def test_a2a_zero_fee_remains_valid_after_quantization() -> None:
    """Catch the principal fix accidentally rejecting an explicit zero fee."""
    command = _initiate(fee=Decimal("0"))

    assert command.payload["fee"] == Decimal("0.00")


_A2A_ACCOUNT_ROLES = (
    "payer_account",
    "payee_account",
    "fee_account",
    "frozen_account",
)


@pytest.mark.parametrize(
    ("first", "second"),
    [
        (first, second)
        for index, first in enumerate(_A2A_ACCOUNT_ROLES)
        for second in _A2A_ACCOUNT_ROLES[index + 1 :]
    ],
)
def test_a2a_rejects_every_account_role_alias(first: str, second: str) -> None:
    """Catch dictionary-leg collapse for any pair of A2A accounting roles."""
    accounts = {
        "payer_account": "payer",
        "payee_account": "payee",
        "fee_account": "a2a:fees",
        "frozen_account": "a2a:frozen",
    }
    accounts[second] = accounts[first]

    with pytest.raises(ValueError, match="A2A account roles must be pairwise distinct"):
        _initiate(**accounts)  # type: ignore[arg-type]


def test_a2a_frozen_account_cannot_alias_payee_and_engine_remains_unchanged() -> None:
    """Catch a fictitious freeze that debits and credits the same account."""
    engine = _engine()

    with pytest.raises(ValueError, match="A2A account roles must be pairwise distinct"):
        _initiate(frozen_account="payee")

    assert engine.ledger.entries == ()
    assert engine.ledger.balance("payer") == Decimal("100.00")
    assert engine.run() == ()


@pytest.mark.parametrize(
    ("command", "code"),
    [
        (AcceptA2A(PAYMENT_ID), "A2A_ACCEPT_BEFORE_INITIATE"),
        (RejectA2A(PAYMENT_ID), "A2A_REJECT_BEFORE_INITIATE"),
        (PostA2A(PAYMENT_ID), "A2A_POST_BEFORE_ACCEPT"),
        (ReportA2AFraud(PAYMENT_ID), "A2A_REPORT_BEFORE_POST"),
        (FreezeA2AFunds(PAYMENT_ID), "A2A_FREEZE_BEFORE_REPORT"),
        (RecoverA2A(PAYMENT_ID), "A2A_RECOVERY_BEFORE_FREEZE"),
        (ReturnA2A(PAYMENT_ID), "A2A_RETURN_BEFORE_POST"),
    ],
)
def test_a2a_illegal_shortcuts_fail_with_stable_code(command: Command, code: str) -> None:
    """Catch transition-table shortcuts or unstable lifecycle diagnostics."""
    engine = _engine()
    _schedule(engine, command)

    with pytest.raises(LifecycleError) as error:
        engine.run()

    assert error.value.code == code
    with pytest.raises(SimulationFailedError):
        engine.run()


def test_a2a_rejection_cannot_coexist_with_acceptance() -> None:
    """Catch incompatible acceptance and rejection terminal paths coexisting."""
    engine = _engine()
    _schedule(engine, _initiate(), AcceptA2A(PAYMENT_ID), RejectA2A(PAYMENT_ID))

    with pytest.raises(LifecycleError) as error:
        engine.run()

    assert error.value.code == "A2A_REJECT_AFTER_ACCEPT"


def test_a2a_state_is_closed_ordered_primitives() -> None:
    """Catch adapters storing mutable models, sets, or unordered lifecycle history."""
    engine = _engine()
    _schedule(engine, _initiate(), AcceptA2A(PAYMENT_ID))
    engine.run()

    state = engine.entity_state(PAYMENT_ID)
    assert isinstance(state, Mapping)
    assert state["state"] == "accepted"
    records = cast(tuple[tuple[str, str, str], ...], state["idempotency_records"])
    assert [(record[0], record[1]) for record in records] == [
        ("a2a-initiate-1", "a2a.initiate"),
        ("a2a.accept:a2a-payment-1", "a2a.accept"),
    ]
    assert all(len(record) == 3 and len(record[2]) == 64 for record in records)


@pytest.mark.parametrize(
    "bad_value",
    [Decimal("NaN"), Decimal("Infinity"), Decimal("-1.00")],
)
def test_a2a_command_rejects_nonfinite_or_negative_amount(bad_value: Decimal) -> None:
    """Catch invalid money entering state before the ledger boundary."""
    with pytest.raises(ValueError, match="finite and positive"):
        _initiate(amount=bad_value)


@pytest.mark.parametrize(
    "command",
    [
        A2ACommand("a2a.accept"),
        type("UnknownA2ACommand", (A2ACommand,), {})(
            "a2a.accept",
            {"payment_id": PAYMENT_ID, "idempotency_key": "unknown"},
        ),
        type("UnknownAcceptA2A", (AcceptA2A,), {})(PAYMENT_ID),
    ],
)
def test_a2a_rejects_unknown_concrete_command_before_payload_access(
    command: Command,
) -> None:
    """Catch public rail base commands leaking KeyError or spoofing known names."""
    engine = _engine()
    _schedule(engine, command)

    with pytest.raises(LifecycleError) as error:
        engine.run()

    assert error.value.code == "A2A_UNKNOWN_COMMAND"


def _forge_a2a_initiation(
    change: Callable[[dict[str, object]], None] | None = None,
    *,
    mutable_payload: bool = False,
    name: str | None = None,
) -> InitiateA2A:
    command = _initiate()
    payload = dict(command.payload)
    if change is not None:
        change(payload)
    forged_payload = payload if mutable_payload else MappingProxyType(payload)
    object.__setattr__(command, "payload", forged_payload)
    if name is not None:
        object.__setattr__(command, "name", name)
    return command


def _drop_a2a_field(payload: dict[str, object], field: str) -> None:
    payload.pop(field)


@pytest.mark.parametrize(
    "forge",
    [
        lambda: _forge_a2a_initiation(name="a2a.accept"),
        lambda: _forge_a2a_initiation(mutable_payload=True),
        lambda: _forge_a2a_initiation(
            lambda payload: _drop_a2a_field(payload, "payment_id")
        ),
        lambda: _forge_a2a_initiation(
            lambda payload: payload.__setitem__("frozen_account", "payee")
        ),
        lambda: _forge_a2a_initiation(
            lambda payload: payload.__setitem__("actor_id", "not-a-uuid")
        ),
        lambda: _forge_a2a_initiation(
            lambda payload: payload.__setitem__("amount", Decimal("NaN"))
        ),
        lambda: _forge_a2a_initiation(
            lambda payload: payload.__setitem__("amount", Decimal("0.004"))
        ),
        lambda: _forge_a2a_initiation(
            lambda payload: payload.__setitem__("idempotency_key", "altered")
        ),
        lambda: _forge_a2a_initiation(
            lambda payload: payload.__setitem__("extra", "forged")
        ),
    ],
)
def test_a2a_forged_known_command_fails_before_any_effect(
    forge: Callable[[], InitiateA2A],
) -> None:
    """Catch forged exact-type commands bypassing canonical constructor admission."""
    engine = _engine()
    _schedule(engine, forge())

    with pytest.raises(LifecycleError) as error:
        engine.run()

    assert error.value.code == "A2A_COMMAND_INVALID"
    assert engine.ledger.entries == ()
    assert engine.ledger.balance("payer") == Decimal("100.00")


@pytest.mark.parametrize(
    "build",
    [
        lambda: _initiate(idempotency_key=""),
        lambda: AcceptA2A(PAYMENT_ID, idempotency_key=""),
    ],
)
def test_a2a_rejects_explicit_empty_idempotency_key(
    build: Callable[[], Command],
) -> None:
    """Catch an explicitly invalid key being replaced with a generated default."""
    with pytest.raises(ValueError, match="idempotency_key must not be empty"):
        build()


def test_a2a_same_seed_replays_byte_identically() -> None:
    """Catch rail-local nondeterminism in IDs, event ordering, or lineage."""
    first = _engine(seed=123)
    second = _engine(seed=123)
    commands = (_initiate(), AcceptA2A(PAYMENT_ID), PostA2A(PAYMENT_ID))
    _schedule(first, *commands)
    _schedule(second, *commands)

    first.run()
    second.run()

    assert first.serialize_events() == second.serialize_events()


def _corrupt_a2a_state(change: Callable[[dict[str, object]], None]) -> SimulationEngine:
    engine = _engine()
    _schedule(engine, _initiate())
    engine.run()
    state = engine.entity_state(PAYMENT_ID)
    assert isinstance(state, Mapping)
    corrupted = cast(dict[str, object], dict(state))
    change(corrupted)
    engine._entity_state[PAYMENT_ID] = cast(FrozenState, MappingProxyType(corrupted))
    engine.schedule(NOW, 0, AcceptA2A(PAYMENT_ID))
    return engine


@pytest.mark.parametrize(
    "change",
    [
        lambda state: _drop_a2a_field(state, "idempotency_records"),
        lambda state: state.__setitem__("state", "invented"),
        lambda state: state.__setitem__("idempotency_records", (("bad",),)),
        lambda state: state.__setitem__("frozen_account", "payee"),
        lambda state: state.__setitem__("amount", Decimal("NaN")),
        lambda state: state.__setitem__("extra", "unexpected"),
    ],
)
def test_a2a_corrupt_state_maps_to_stable_error_before_new_effect(
    change: Callable[[dict[str, object]], None],
) -> None:
    """Catch malformed closed state leaking raw errors or posting new value."""
    engine = _corrupt_a2a_state(change)

    with pytest.raises(LifecycleError) as error:
        engine.run()

    assert error.value.code == "A2A_STATE_CORRUPT"
    assert engine.ledger.entries == ()
