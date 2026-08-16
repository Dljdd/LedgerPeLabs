"""Executable card-rail lifecycle, accounting, and event-contract tests."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from datetime import timedelta
from decimal import Decimal

import pytest

from apar.compiler.compiler import compile_scenario
from apar.contracts.events import EventKind, Rail
from apar.simulator.clock import Command
from apar.simulator.engine import SimulationEngine, SimulationFailedError
from apar.simulator.rails.card import (
    AuthorizeCard,
    CardCommand,
    CardRailAdapter,
    ChargebackCard,
    ClearCard,
    DeclineCardAuthorization,
    LifecycleError,
    OpenCardDispute,
    RecoverCard,
    RefundCard,
    ReportCardFraud,
    ReverseCardAuthorization,
    SettleCard,
)
from tests.factories import NOW, make_scenario_config, make_threat_card

PAYMENT_ID = "card-payment-1"
CAMPAIGN_ID = "00000000-0000-4000-8000-000000000101"
TRACE_ID = "00000000-0000-4000-8000-000000000102"
PAYER_ID = "00000000-0000-4000-8000-000000000103"
PAYEE_ID = "00000000-0000-4000-8000-000000000104"


def _card_bundle(*, seed: int = 260_816):  # type: ignore[no-untyped-def]
    config = make_scenario_config(
        rail=Rail.CARD,
        viewpoint="network_native",
        seed=seed,
        replay=make_scenario_config().replay.model_copy(update={"random_seed": seed}),
    )
    card = make_threat_card(
        rails=[Rail.CARD],
        viewpoint="network_native",
        default_config=config,
    )
    return compile_scenario(card, config)


def _engine(
    *,
    seed: int = 260_816,
    payer_balance: Decimal = Decimal("100.00"),
    currency: str = "USD",
    factories: Mapping[Rail, Callable[[], object]] | None = None,
) -> SimulationEngine:
    return SimulationEngine(
        _card_bundle(seed=seed),
        factories or {Rail.CARD: CardRailAdapter},  # type: ignore[arg-type]
        opening_balances={("payer", currency): payer_balance},
    )


def _authorize(
    *,
    amount: Decimal = Decimal("10.00"),
    fee: Decimal = Decimal("0.00"),
    currency: str = "USD",
    idempotency_key: str = "card-authorize-1",
    payer_account: str = "payer",
    payee_account: str = "merchant",
    hold_account: str = "card:holds",
    fee_account: str = "card:fees",
    chargeback_account: str = "card:chargebacks",
) -> AuthorizeCard:
    return AuthorizeCard(
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
        hold_account=hold_account,
        fee_account=fee_account,
        chargeback_account=chargeback_account,
        idempotency_key=idempotency_key,
    )


def _schedule(engine: SimulationEngine, *commands: Command) -> None:
    for priority, command in enumerate(commands):
        engine.schedule(NOW, priority, command)


def test_card_commands_are_public_immutable_command_subclasses() -> None:
    """Catch rail builders bypassing the engine's immutable Command boundary."""
    command = _authorize()

    assert isinstance(command, Command)
    assert command.campaign_id == CAMPAIGN_ID
    assert command.payload["amount"] == Decimal("10.00")


def test_rail_package_exports_both_payment_adapters() -> None:
    """Catch public adapter imports disappearing behind implementation modules."""
    from apar.simulator.rails import A2ARailAdapter as PublicA2AAdapter
    from apar.simulator.rails import CardRailAdapter as PublicCardAdapter

    assert PublicA2AAdapter.__name__ == "A2ARailAdapter"
    assert PublicCardAdapter is CardRailAdapter


def test_card_full_recovery_path_emits_linked_events_and_conserves_value() -> None:
    """Catch a missing lifecycle step, mutation, wrong posting, or broken lineage."""
    engine = _engine()
    _schedule(
        engine,
        _authorize(fee=Decimal("1.00")),
        ClearCard(PAYMENT_ID),
        SettleCard(PAYMENT_ID),
        ReportCardFraud(PAYMENT_ID),
        OpenCardDispute(PAYMENT_ID),
        ChargebackCard(PAYMENT_ID),
        RecoverCard(PAYMENT_ID),
    )

    events = engine.run()

    assert [event.event_type for event in events] == [
        EventKind.AUTHORIZATION,
        EventKind.CLEARING,
        EventKind.SETTLEMENT,
        EventKind.FRAUD_REPORTED,
        EventKind.DISPUTE_OPENED,
        EventKind.CHARGEBACK,
        EventKind.RECOVERY,
    ]
    assert [event.lineage.get("previous_event_id") for event in events] == [
        None,
        *[event.event_id for event in events[:-1]],
    ]
    assert all(event.rail is Rail.CARD for event in events)
    assert all(event.viewpoint == "network_native" for event in events)
    assert all(event.campaign_id == CAMPAIGN_ID for event in events)
    assert all(event.trace_id == TRACE_ID for event in events)
    assert all(event.actor_id == PAYER_ID for event in events)
    assert all(event.counterparty_id == PAYEE_ID for event in events)
    assert all(event.rail_data["payment_id"] == PAYMENT_ID for event in events)
    assert engine.ledger.balance("payer") == Decimal("100.00")
    assert engine.ledger.balance("merchant") == Decimal("0.00")
    assert engine.ledger.balance("card:fees") == Decimal("0.00")
    assert engine.ledger.balance("card:chargebacks") == Decimal("0.00")
    engine.ledger.assert_conserved()


def test_card_authorization_holds_then_reversal_releases_without_settlement() -> None:
    """Catch authorization posting settlement value or reversal creating value."""
    engine = _engine()
    engine.schedule(NOW, 0, _authorize())
    engine.schedule(NOW + timedelta(seconds=1), 0, ReverseCardAuthorization(PAYMENT_ID))

    first = engine.run(until=NOW)
    assert [event.event_type for event in first] == [EventKind.AUTHORIZATION]
    assert engine.ledger.balance("payer") == Decimal("90.00")
    assert engine.ledger.balance("card:holds") == Decimal("10.00")
    assert engine.ledger.balance("merchant") == Decimal("0.00")

    events = engine.run()
    assert [event.event_type for event in events] == [
        EventKind.AUTHORIZATION,
        EventKind.REVERSAL,
    ]
    assert engine.ledger.balance("payer") == Decimal("100.00")
    assert engine.ledger.balance("card:holds") == Decimal("0.00")
    assert engine.ledger.balance("merchant") == Decimal("0.00")
    assert len(engine.ledger.entries) == 2
    engine.ledger.assert_conserved()


def test_card_settlement_consumes_hold_once_and_posts_fee_explicitly() -> None:
    """Catch repeated hold consumption or a fee disappearing from double entry."""
    engine = _engine()
    settle = SettleCard(PAYMENT_ID, idempotency_key="settle-once")
    _schedule(engine, _authorize(fee=Decimal("1.00")), ClearCard(PAYMENT_ID), settle, settle)

    events = engine.run()

    assert [event.event_type for event in events].count(EventKind.SETTLEMENT) == 1
    assert engine.ledger.balance("payer") == Decimal("90.00")
    assert engine.ledger.balance("card:holds") == Decimal("0.00")
    assert engine.ledger.balance("merchant") == Decimal("9.00")
    assert engine.ledger.balance("card:fees") == Decimal("1.00")
    assert len(engine.ledger.entries) == 2
    engine.ledger.assert_conserved()


def test_card_refund_is_a_separate_terminal_branch_and_unwinds_fee() -> None:
    """Catch refunds mutating settlement history or leaving value unreconciled."""
    engine = _engine()
    _schedule(
        engine,
        _authorize(fee=Decimal("1.00")),
        ClearCard(PAYMENT_ID),
        SettleCard(PAYMENT_ID),
        RefundCard(PAYMENT_ID),
    )

    events = engine.run()

    assert [event.event_type for event in events] == [
        EventKind.AUTHORIZATION,
        EventKind.CLEARING,
        EventKind.SETTLEMENT,
        EventKind.REFUND,
    ]
    assert engine.ledger.balance("payer") == Decimal("100.00")
    assert engine.ledger.balance("merchant") == Decimal("0.00")
    assert engine.ledger.balance("card:fees") == Decimal("0.00")
    engine.ledger.assert_conserved()


def test_card_decline_emits_observable_event_without_ledger_posting() -> None:
    """Catch declined authorizations moving or holding value."""
    engine = _engine()
    decline = DeclineCardAuthorization.from_authorization(_authorize())
    _schedule(engine, decline)

    events = engine.run()

    assert [event.event_type for event in events] == [EventKind.AUTHORIZATION_DECLINED]
    assert engine.ledger.balance("payer") == Decimal("100.00")
    assert engine.ledger.balance("card:holds") == Decimal("0.00")
    assert engine.ledger.entries == ()


def test_card_duplicate_authorization_key_is_a_complete_noop() -> None:
    """Catch retries duplicating an authorization event or hold posting."""
    engine = _engine()
    command = _authorize(idempotency_key="same-request")
    _schedule(engine, command, command)

    events = engine.run()

    assert len(events) == 1
    assert len(engine.ledger.entries) == 1
    assert engine.ledger.balance("card:holds") == Decimal("10.00")


def test_card_idempotency_key_collision_with_different_command_fails_closed() -> None:
    """Catch one key silently suppressing a different card lifecycle command."""
    engine = _engine()
    _schedule(
        engine,
        _authorize(idempotency_key="shared-key"),
        ClearCard(PAYMENT_ID, idempotency_key="shared-key"),
    )

    with pytest.raises(LifecycleError) as error:
        engine.run()

    assert error.value.code == "CARD_IDEMPOTENCY_KEY_COLLISION"
    assert len(engine.ledger.entries) == 1
    assert engine.ledger.balance("payer") == Decimal("90.00")
    assert engine.ledger.balance("card:holds") == Decimal("10.00")


def test_card_idempotency_state_records_key_and_command_identity() -> None:
    """Catch closed state retaining keys without the command identity they retry."""
    engine = _engine()
    _schedule(engine, _authorize(), ClearCard(PAYMENT_ID))
    engine.run()

    state = engine.entity_state(PAYMENT_ID)
    assert isinstance(state, Mapping)
    assert state["idempotency_records"] == (
        ("card-authorize-1", "card.authorize"),
        ("card.clear:card-payment-1", "card.clear"),
    )


def test_card_rejects_overdraft_atomically() -> None:
    """Catch card holds bypassing the ledger's no-overdraft invariant."""
    engine = _engine(payer_balance=Decimal("5.00"))
    _schedule(engine, _authorize())

    with pytest.raises(ValueError, match="overdraw account: payer"):
        engine.run()

    assert engine.ledger.balance("payer") == Decimal("5.00")
    assert engine.ledger.entries == ()
    with pytest.raises(SimulationFailedError):
        engine.run()


def test_card_quantizes_supported_currency_before_events_and_postings() -> None:
    """Catch payment events disagreeing with exponent-aware ledger amounts."""
    engine = _engine(currency="KWD", payer_balance=Decimal("100.000"))
    _schedule(engine, _authorize(amount=Decimal("10.5555"), currency="KWD"))

    event = engine.run()[0]

    assert event.amount == Decimal("10.556")
    assert engine.ledger.balance("payer", "KWD") == Decimal("89.444")
    assert engine.ledger.balance("card:holds", "KWD") == Decimal("10.556")
    engine.ledger.assert_conserved()


@pytest.mark.parametrize(
    ("currency", "amount"),
    [
        ("USD", Decimal("0.004")),
        ("JPY", Decimal("0.4")),
        ("KWD", Decimal("0.0004")),
    ],
)
def test_card_rejects_principal_that_quantizes_to_zero(
    currency: str,
    amount: Decimal,
) -> None:
    """Catch a nominally positive card amount becoming a zero-value hold."""
    with pytest.raises(ValueError, match="amount must be finite and positive"):
        _authorize(currency=currency, amount=amount)


def test_card_zero_fee_remains_valid_after_quantization() -> None:
    """Catch the principal fix accidentally rejecting an explicit zero fee."""
    command = _authorize(fee=Decimal("0"))

    assert command.payload["fee"] == Decimal("0.00")


_CARD_ACCOUNT_ROLES = (
    "payer_account",
    "payee_account",
    "hold_account",
    "fee_account",
    "chargeback_account",
)


@pytest.mark.parametrize(
    ("first", "second"),
    [
        (first, second)
        for index, first in enumerate(_CARD_ACCOUNT_ROLES)
        for second in _CARD_ACCOUNT_ROLES[index + 1 :]
    ],
)
def test_card_rejects_every_account_role_alias(first: str, second: str) -> None:
    """Catch dictionary-leg collapse for any pair of card accounting roles."""
    accounts = {
        "payer_account": "payer",
        "payee_account": "merchant",
        "hold_account": "card:holds",
        "fee_account": "card:fees",
        "chargeback_account": "card:chargebacks",
    }
    accounts[second] = accounts[first]

    with pytest.raises(ValueError, match="card account roles must be pairwise distinct"):
        _authorize(**accounts)  # type: ignore[arg-type]


def test_card_hold_cannot_alias_payer_and_engine_remains_unchanged() -> None:
    """Catch a fictitious hold that debits and credits the same account."""
    engine = _engine()

    with pytest.raises(ValueError, match="card account roles must be pairwise distinct"):
        _authorize(hold_account="payer")

    assert engine.ledger.entries == ()
    assert engine.ledger.balance("payer") == Decimal("100.00")
    assert engine.run() == ()


@pytest.mark.parametrize(
    ("command", "code"),
    [
        (ClearCard(PAYMENT_ID), "CARD_CLEAR_BEFORE_AUTHORIZE"),
        (SettleCard(PAYMENT_ID), "CARD_SETTLE_BEFORE_CLEAR"),
        (ReverseCardAuthorization(PAYMENT_ID), "CARD_REVERSE_BEFORE_AUTHORIZE"),
        (ReportCardFraud(PAYMENT_ID), "CARD_REPORT_BEFORE_SETTLEMENT"),
        (OpenCardDispute(PAYMENT_ID), "CARD_DISPUTE_BEFORE_REPORT"),
        (ChargebackCard(PAYMENT_ID), "CARD_CHARGEBACK_BEFORE_DISPUTE"),
        (RecoverCard(PAYMENT_ID), "CARD_RECOVERY_BEFORE_CHARGEBACK"),
        (RefundCard(PAYMENT_ID), "CARD_REFUND_BEFORE_SETTLEMENT"),
    ],
)
def test_card_illegal_shortcuts_fail_with_stable_code(command: Command, code: str) -> None:
    """Catch transition-table shortcuts or unstable lifecycle diagnostics."""
    engine = _engine()
    _schedule(engine, command)

    with pytest.raises(LifecycleError) as error:
        engine.run()

    assert error.value.code == code
    with pytest.raises(SimulationFailedError):
        engine.run()


def test_card_reversal_is_illegal_after_settlement() -> None:
    """Catch the pre-settlement reversal branch coexisting with settlement."""
    engine = _engine()
    _schedule(
        engine,
        _authorize(),
        ClearCard(PAYMENT_ID),
        SettleCard(PAYMENT_ID),
        ReverseCardAuthorization(PAYMENT_ID),
    )

    with pytest.raises(LifecycleError) as error:
        engine.run()

    assert error.value.code == "CARD_REVERSE_AFTER_SETTLEMENT"


@pytest.mark.parametrize(
    "bad_value",
    [Decimal("NaN"), Decimal("Infinity"), Decimal("-1.00")],
)
def test_card_command_rejects_nonfinite_or_negative_amount(bad_value: Decimal) -> None:
    """Catch invalid money entering state before the ledger boundary."""
    with pytest.raises(ValueError, match="finite and positive"):
        _authorize(amount=bad_value)


def test_card_command_rejects_mutable_string_subclass() -> None:
    """Catch rail command builders retaining string-subclass aliases."""
    class MutableStr(str):
        pass

    payment_id = MutableStr(PAYMENT_ID)
    payment_id.audit = []  # type: ignore[attr-defined]

    with pytest.raises(TypeError, match="payment_id must be an exact string"):
        AuthorizeCard(
            payment_id,
            amount=Decimal("10.00"),
            currency="USD",
            payer_account="payer",
            payee_account="merchant",
            actor_id=PAYER_ID,
            counterparty_id=PAYEE_ID,
            campaign_id=CAMPAIGN_ID,
            trace_id=TRACE_ID,
        )


@pytest.mark.parametrize(
    "command",
    [
        CardCommand("card.clear"),
        type("UnknownCardCommand", (CardCommand,), {})(
            "card.clear",
            {"payment_id": PAYMENT_ID, "idempotency_key": "unknown"},
        ),
        type("UnknownClearCard", (ClearCard,), {})(PAYMENT_ID),
    ],
)
def test_card_rejects_unknown_concrete_command_before_payload_access(
    command: Command,
) -> None:
    """Catch public rail base commands leaking KeyError or spoofing known names."""
    engine = _engine()
    _schedule(engine, command)

    with pytest.raises(LifecycleError) as error:
        engine.run()

    assert error.value.code == "CARD_UNKNOWN_COMMAND"


@pytest.mark.parametrize(
    "build",
    [
        lambda: _authorize(idempotency_key=""),
        lambda: ClearCard(PAYMENT_ID, idempotency_key=""),
    ],
)
def test_card_rejects_explicit_empty_idempotency_key(
    build: Callable[[], Command],
) -> None:
    """Catch an explicitly invalid key being replaced with a generated default."""
    with pytest.raises(ValueError, match="idempotency_key must not be empty"):
        build()


def test_card_same_seed_replays_byte_identically() -> None:
    """Catch rail-local nondeterminism in IDs, event ordering, or lineage."""
    first = _engine(seed=99)
    second = _engine(seed=99)
    commands = (_authorize(), ClearCard(PAYMENT_ID), SettleCard(PAYMENT_ID))
    _schedule(first, *commands)
    _schedule(second, *commands)

    first.run()
    second.run()

    assert first.serialize_events() == second.serialize_events()
