"""Agentic rail integration tests for trust-before-risk and trust-before-value."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from types import MappingProxyType
from typing import cast
from unittest.mock import Mock

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat

from apar.compiler.compiler import compile_scenario
from apar.contracts.decisions import Action, ReasonCode
from apar.contracts.events import EventKind, Rail
from apar.simulator.clock import Command
from apar.simulator.engine import SimulationEngine
from apar.simulator.rails.agentic import (
    AGENTIC_TRUST_STATE_ID,
    AgenticPaymentCommand,
    AgenticRailAdapter,
)
from apar.simulator.rails.base import FrozenState, LifecycleError
from apar.trust.verifier import (
    AgentMandate,
    AgentPaymentRequest,
    AuthenticationEvidence,
    AuthenticationOutcome,
    AuthenticationRequirement,
    IntegrityReceipt,
    ReceiptOutcome,
    TrustVerifier,
)
from tests.factories import make_scenario_config, make_threat_card

NOW = datetime(2026, 8, 16, 12, tzinfo=UTC)
AGENT_ID = "agent-registered-1"
KEY_ID = "key-1"
MANDATE_ID = "mandate-1"
PAYEE_ID = "merchant-1"
MERCHANT_ID = "merchant-entity-1"
CART_HASH = "1" * 64
PAYMENT_INTENT_HASH = "3" * 64
CREDENTIAL_ID = "synthetic-token-1"
CREDENTIAL_SCOPE = "single_merchant_single_use"
CATEGORY = "TRAVEL"
PRODUCT_ID = "flight-1"
CAMPAIGN_ID = "00000000-0000-4000-8000-000000000301"
TRACE_ID = "00000000-0000-4000-8000-000000000302"
ACTOR_ID = "00000000-0000-4000-8000-000000000303"
COUNTERPARTY_ID = "00000000-0000-4000-8000-000000000304"
USER_REF = "user-1"


def _private_key() -> Ed25519PrivateKey:
    return Ed25519PrivateKey.from_private_bytes(bytes(range(32)))


def _mandate() -> AgentMandate:
    return AgentMandate(
        mandate_id=MANDATE_ID,
        version=1,
        agent_id=AGENT_ID,
        user_ref="user-1",
        consent_ref="consent-1",
        user_entity_id=ACTOR_ID,
        beneficiary_entity_id=COUNTERPARTY_ID,
        merchant_id=MERCHANT_ID,
        payee_id=PAYEE_ID,
        cart_hash=CART_HASH,
        payment_intent_hash=PAYMENT_INTENT_HASH,
        permitted_categories=(CATEGORY,),
        permitted_products=(PRODUCT_ID,),
        credential_id=CREDENTIAL_ID,
        credential_scope=CREDENTIAL_SCOPE,
        required_authentication=AuthenticationRequirement.STEP_UP,
        max_amount=Decimal("100.00"),
        currency="USD",
        issued_at=NOW - timedelta(minutes=5),
        expires_at=NOW + timedelta(hours=1),
    )


def _request(**updates: object) -> AgentPaymentRequest:
    mandate = cast(AgentMandate, updates.pop("mandate", _mandate()))
    values: dict[str, object] = {
        "request_id": "request-1",
        "payment_id": "agentic-payment-1",
        "agent_id": AGENT_ID,
        "key_id": KEY_ID,
        "mandate": mandate,
        "amount": Decimal("10.00"),
        "currency": "USD",
        "merchant_id": MERCHANT_ID,
        "payee_id": PAYEE_ID,
        "cart_hash": CART_HASH,
        "payment_intent_hash": PAYMENT_INTENT_HASH,
        "category": CATEGORY,
        "product_id": PRODUCT_ID,
        "credential_id": CREDENTIAL_ID,
        "credential_scope": CREDENTIAL_SCOPE,
        "consent_ref": "consent-1",
        "authentication_evidence_ref": "auth-evidence-1",
        "nonce": "nonce-1",
        "created_at": NOW,
        "expires_at": NOW + timedelta(minutes=5),
        "prior_receipt_hash": "",
        "campaign_id": CAMPAIGN_ID,
        "trace_id": TRACE_ID,
        "actor_id": ACTOR_ID,
        "counterparty_id": COUNTERPARTY_ID,
        "signature": b"",
    }
    values.update(updates)
    unsigned = AgentPaymentRequest(**values)  # type: ignore[arg-type]
    return unsigned.model_copy(
        update={"signature": _private_key().sign(unsigned.signing_bytes())}
    )


def _verifier() -> TrustVerifier:
    public_key = _private_key().public_key().public_bytes(Encoding.Raw, PublicFormat.Raw)
    return TrustVerifier(
        registered_agents={(AGENT_ID, KEY_ID): public_key},
        mandates={MANDATE_ID: _mandate()},
        authentication_evidence={
            "auth-evidence-1": AuthenticationEvidence(
                evidence_id="auth-evidence-1",
                agent_id=AGENT_ID,
                user_ref=USER_REF,
                mandate_id=MANDATE_ID,
                nonce="nonce-1",
                payment_intent_hash=PAYMENT_INTENT_HASH,
                request_id="request-1",
                outcome=AuthenticationOutcome.STEP_UP_VERIFIED,
                issued_at=NOW - timedelta(seconds=5),
                expires_at=NOW + timedelta(minutes=2),
            )
        },
    )


def _bundle(seed: int = 260_816):  # type: ignore[no-untyped-def]
    config = make_scenario_config(
        rail=Rail.AGENTIC,
        viewpoint="agentic_commerce_gateway",
        seed=seed,
        replay=make_scenario_config().replay.model_copy(update={"random_seed": seed}),
    )
    card = make_threat_card(
        rails=[Rail.AGENTIC],
        viewpoint="agentic_commerce_gateway",
        default_config=config,
    )
    return compile_scenario(card, config)


def _engine(
    scorer: Callable[[AgentPaymentRequest, object], Action],
    *,
    seed: int = 260_816,
) -> SimulationEngine:
    return SimulationEngine(
        _bundle(seed),
        {Rail.AGENTIC: lambda: AgenticRailAdapter(_verifier(), scorer)},
        opening_balances={USER_REF: Decimal("100.00")},
    )


def _command(request: AgentPaymentRequest | None = None) -> AgenticPaymentCommand:
    effective_request = request or _request()
    return AgenticPaymentCommand(
        effective_request,
        payer_account=effective_request.mandate.user_ref,
        payee_account=effective_request.payee_id,
    )


def test_agentic_command_is_public_immutable_engine_command() -> None:
    command = _command()

    assert isinstance(command, Command)
    assert command.payment_id == "agentic-payment-1"
    assert command.campaign_id == CAMPAIGN_ID
    assert isinstance(command.payload, Mapping)


def test_payee_substitution_declines_before_scorer_and_post() -> None:
    scorer = Mock(return_value=Action.APPROVE)
    request = _request(payee_id="attacker")
    engine = _engine(scorer)
    engine.schedule(NOW, 0, _command(request))

    events = engine.run()

    assert [event.event_type for event in events] == [EventKind.AUTHORIZATION_DECLINED]
    assert events[0].rail_data["reason_code"] == "PAYEE_BINDING_MISMATCH"
    scorer.assert_not_called()
    assert engine.ledger.entries == ()
    assert engine.ledger.balance(USER_REF) == Decimal("100.00")


def test_success_scores_once_posts_once_and_persists_closed_nonce_state() -> None:
    scorer = Mock(return_value=Action.APPROVE)
    engine = _engine(scorer)
    engine.schedule(NOW, 0, _command())

    events = engine.run()

    scorer.assert_called_once()
    assert [event.event_type for event in events] == [EventKind.AUTHORIZATION]
    assert events[0].rail is Rail.AGENTIC
    assert events[0].rail_data["integrity"] == "pass"
    assert len(cast(str, events[0].rail_data["receipt_hash"])) == 64
    assert engine.ledger.balance(USER_REF) == Decimal("90.00")
    assert engine.ledger.balance(PAYEE_ID) == Decimal("10.00")
    engine.ledger.assert_conserved()
    state = engine.entity_state(AGENTIC_TRUST_STATE_ID)
    assert isinstance(state, Mapping)
    assert state["version"] == 1
    assert isinstance(state["records"], tuple)
    assert events[0].actor_id == ACTOR_ID
    assert events[0].counterparty_id == COUNTERPARTY_ID
    assert events[0].party_refs == {
        "user_ref": USER_REF,
        "merchant_id": MERCHANT_ID,
        "payee_id": PAYEE_ID,
        "user_entity_id": ACTOR_ID,
        "beneficiary_entity_id": COUNTERPARTY_ID,
    }


def test_nonce_replay_declines_without_second_score_or_post() -> None:
    scorer = Mock(return_value=Action.APPROVE)
    engine = _engine(scorer)
    command = _command()
    engine.schedule(NOW, 0, command)
    engine.schedule(NOW, 1, command)

    events = engine.run()

    assert [event.event_type for event in events] == [
        EventKind.AUTHORIZATION,
        EventKind.AUTHORIZATION_DECLINED,
    ]
    assert events[1].rail_data["reason_code"] == "NONCE_REPLAY"
    scorer.assert_called_once()
    assert len(engine.ledger.entries) == 1


def test_risk_decline_occurs_only_after_integrity_and_does_not_post() -> None:
    scorer = Mock(return_value=Action.DECLINE)
    engine = _engine(scorer)
    engine.schedule(NOW, 0, _command())

    event = engine.run()[0]

    scorer.assert_called_once()
    assert event.event_type is EventKind.AUTHORIZATION_DECLINED
    assert event.rail_data["integrity"] == "pass"
    assert event.rail_data["action"] == "decline"
    assert engine.ledger.entries == ()
    assert isinstance(engine.entity_state(AGENTIC_TRUST_STATE_ID), Mapping)


def test_challenge_emits_semantic_challenge_event_and_outcome_receipt() -> None:
    scorer = Mock(return_value=Action.CHALLENGE)
    engine = _engine(scorer)
    engine.schedule(NOW, 0, _command())

    event = engine.run()[0]

    assert event.event_type is EventKind.AUTHENTICATION_CHALLENGE
    assert event.rail_data["action"] == "challenge"
    assert event.rail_data["receipt_outcome"] == ReceiptOutcome.CHALLENGE.value
    assert event.rail_data["integrity"] == "pass"
    assert engine.ledger.entries == ()


@pytest.mark.parametrize(
    "identity_update",
    [
        {"actor_id": "00000000-0000-4000-8000-000000000399"},
        {"counterparty_id": "00000000-0000-4000-8000-000000000398"},
    ],
)
def test_substituted_graph_identity_fails_before_score_or_effect(
    identity_update: dict[str, object],
) -> None:
    scorer = Mock(return_value=Action.APPROVE)
    request = _request(**identity_update)
    engine = _engine(scorer)
    engine.schedule(NOW, 0, _command(request))

    event = engine.run()[0]

    assert event.rail_data["reason_code"] == "AUTHORITY_IDENTITY_MISMATCH"
    scorer.assert_not_called()
    assert engine.ledger.entries == ()
    with pytest.raises(KeyError):
        engine.entity_state(AGENTIC_TRUST_STATE_ID)


def test_scorer_exception_leaves_nonce_state_unchanged_and_retry_succeeds() -> None:
    verifier = _verifier()
    state_before = verifier.dump_state()
    captured_receipts: list[object] = []

    def fail_after_capture(_request: AgentPaymentRequest, receipt: object) -> Action:
        captured_receipts.append(receipt)
        raise RuntimeError("score failed")

    failing = AgenticRailAdapter(
        verifier,
        fail_after_capture,
    )

    with pytest.raises(RuntimeError, match="score failed"):
        failing.process(_request(), now=NOW)

    assert verifier.dump_state() == state_before
    abandoned = cast(IntegrityReceipt, captured_receipts[0])
    rejected_commit = verifier.commit(
        _request(), abandoned, ReceiptOutcome.APPROVE, NOW
    )
    assert rejected_commit.reason_code is ReasonCode.EXECUTION_RECEIPT_MISMATCH
    retry = AgenticRailAdapter(
        verifier,
        lambda _request, _receipt: Action.APPROVE,
    ).process(_request(), now=NOW)
    assert retry.action is Action.APPROVE
    assert retry.integrity_receipt.outcome is ReceiptOutcome.PREVIEW
    assert verifier.dump_state() == state_before


@pytest.mark.parametrize("bad_output", [None, "approve", 0.5, float("nan")])
def test_malformed_or_nonfinite_scorer_output_is_failure_atomic(
    bad_output: object,
) -> None:
    verifier = _verifier()
    state_before = verifier.dump_state()
    adapter = AgenticRailAdapter(
        verifier,
        lambda _request, _receipt: cast(Action, bad_output),
    )

    with pytest.raises(TypeError, match="exact Action"):
        adapter.process(_request(), now=NOW)

    assert verifier.dump_state() == state_before


def test_approve_overdraft_leaves_verifier_ledger_event_and_entity_state_empty() -> None:
    scorer = Mock(return_value=Action.APPROVE)
    verifier = _verifier()
    adapter = AgenticRailAdapter(verifier, scorer)
    engine = SimulationEngine(
        _bundle(),
        {Rail.AGENTIC: lambda: adapter},
        opening_balances={USER_REF: Decimal("5.00")},
    )
    engine.schedule(NOW, 0, _command())
    state_before = verifier.dump_state()

    with pytest.raises(ValueError, match="overdraw"):
        engine.run()

    assert verifier.dump_state() == state_before
    assert engine.ledger.entries == ()
    assert engine._events == []
    assert AGENTIC_TRUST_STATE_ID not in engine._entity_state

    retry = SimulationEngine(
        _bundle(),
        {Rail.AGENTIC: lambda: AgenticRailAdapter(verifier, scorer)},
        opening_balances={USER_REF: Decimal("100.00")},
    )
    retry.schedule(NOW, 0, _command())

    event = retry.run()[0]

    assert event.rail_data["receipt_outcome"] == ReceiptOutcome.APPROVE.value
    assert len(retry.ledger.entries) == 1


def test_unexpected_ledger_post_failure_does_not_commit_receipt_or_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scorer = Mock(return_value=Action.APPROVE)
    verifier = _verifier()
    adapter = AgenticRailAdapter(verifier, scorer)
    engine = SimulationEngine(
        _bundle(),
        {Rail.AGENTIC: lambda: adapter},
        opening_balances={USER_REF: Decimal("100.00")},
    )
    engine.schedule(NOW, 0, _command())
    state_before = verifier.dump_state()

    def reject_post(_entry: object) -> None:
        raise ValueError("synthetic posting validation error")

    monkeypatch.setattr(engine._ledger, "post", reject_post)
    with pytest.raises(ValueError, match="posting validation"):
        engine.run()

    assert verifier.dump_state() == state_before
    assert engine._events == []
    assert AGENTIC_TRUST_STATE_ID not in engine._entity_state

    retry = SimulationEngine(
        _bundle(),
        {Rail.AGENTIC: lambda: AgenticRailAdapter(verifier, scorer)},
        opening_balances={USER_REF: Decimal("100.00")},
    )
    retry.schedule(NOW, 0, _command())

    event = retry.run()[0]

    assert event.rail_data["receipt_outcome"] == ReceiptOutcome.APPROVE.value
    assert len(retry.ledger.entries) == 1


def test_agentic_receipts_and_events_replay_deterministically() -> None:
    first = _engine(lambda _request, _receipt: Action.APPROVE, seed=99)
    second = _engine(lambda _request, _receipt: Action.APPROVE, seed=99)
    first.schedule(NOW, 0, _command())
    second.schedule(NOW, 0, _command())

    first.run()
    second.run()

    assert first.serialize_events() == second.serialize_events()
    assert first.entity_state(AGENTIC_TRUST_STATE_ID) == second.entity_state(
        AGENTIC_TRUST_STATE_ID
    )


def test_corrupt_adapter_state_fails_before_score_and_new_post() -> None:
    scorer = Mock(return_value=Action.APPROVE)
    engine = _engine(scorer)
    engine._entity_state[AGENTIC_TRUST_STATE_ID] = cast(
        FrozenState,
        MappingProxyType({"version": 2, "records": ()}),
    )
    engine.schedule(NOW, 0, _command())

    with pytest.raises(LifecycleError) as error:
        engine.run()

    assert error.value.code == "AGENTIC_STATE_CORRUPT"
    scorer.assert_not_called()
    assert engine.ledger.entries == ()


def test_forged_known_command_fails_before_score_and_post() -> None:
    scorer = Mock(return_value=Action.APPROVE)
    command = _command()
    object.__setattr__(command, "payload", MappingProxyType({"payment_id": "forged"}))
    engine = _engine(scorer)
    engine.schedule(NOW, 0, command)

    with pytest.raises(LifecycleError) as error:
        engine.run()

    assert error.value.code == "AGENTIC_COMMAND_INVALID"
    scorer.assert_not_called()
    assert engine.ledger.entries == ()


@pytest.mark.parametrize(
    "command",
    [
        Command("agentic.pay"),
        type("UnknownAgenticPayment", (AgenticPaymentCommand,), {})(
            _request(), payer_account=USER_REF, payee_account=PAYEE_ID
        ),
    ],
)
def test_unknown_agentic_command_fails_closed(command: Command) -> None:
    scorer = Mock(return_value=Action.APPROVE)
    engine = _engine(scorer)
    engine.schedule(NOW, 0, command)

    with pytest.raises(LifecycleError) as error:
        engine.run()

    assert error.value.code == "AGENTIC_UNKNOWN_COMMAND"
    scorer.assert_not_called()


def test_invalid_signature_does_not_create_nonce_state() -> None:
    scorer = Mock(return_value=Action.APPROVE)
    request = _request().model_copy(update={"signature": bytes(64)})
    engine = _engine(scorer)
    engine.schedule(NOW, 0, _command(request))

    event = engine.run()[0]

    assert event.rail_data["reason_code"] == ReasonCode.SIGNATURE_INVALID.value
    scorer.assert_not_called()
    assert engine.ledger.entries == ()
    with pytest.raises(KeyError):
        engine.entity_state(AGENTIC_TRUST_STATE_ID)


@pytest.mark.parametrize(
    ("payer_account", "payee_account"),
    [("substituted-payer", PAYEE_ID), (USER_REF, "substituted-payee")],
)
def test_agentic_command_rejects_ledger_account_substitution(
    payer_account: str,
    payee_account: str,
) -> None:
    with pytest.raises(ValueError, match="must match signed request binding"):
        AgenticPaymentCommand(
            _request(),
            payer_account=payer_account,
            payee_account=payee_account,
        )


def test_agentic_payer_account_is_bound_to_mandated_user() -> None:
    with pytest.raises(ValueError, match="must match signed request binding"):
        AgenticPaymentCommand(
            _request(),
            payer_account=ACTOR_ID,
            payee_account=PAYEE_ID,
        )
