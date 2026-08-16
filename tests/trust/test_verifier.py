"""Ordered fail-closed checks for delegated agent payment integrity."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from datetime import UTC, datetime, timedelta, timezone
from decimal import Decimal
from types import MappingProxyType

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat

from apar.contracts.decisions import ReasonCode
from apar.trust.verifier import (
    AgentMandate,
    AgentPaymentRequest,
    IntegrityReceipt,
    TrustVerifier,
    TrustVerifierStateError,
)

NOW = datetime(2026, 8, 16, 12, tzinfo=UTC)
AGENT_ID = "agent-registered-1"
KEY_ID = "key-1"
MANDATE_ID = "mandate-1"
USER_REF = "user-1"
CONSENT_REF = "consent-1"
PAYEE_ID = "merchant-1"
CART_HASH = "1" * 64


@pytest.fixture
def private_key() -> Ed25519PrivateKey:
    return Ed25519PrivateKey.from_private_bytes(bytes(range(32)))


@pytest.fixture
def mandate() -> AgentMandate:
    return AgentMandate(
        mandate_id=MANDATE_ID,
        version=3,
        agent_id=AGENT_ID,
        user_ref=USER_REF,
        consent_ref=CONSENT_REF,
        payee_id=PAYEE_ID,
        cart_hash=CART_HASH,
        max_amount=Decimal("150.00"),
        currency="USD",
        issued_at=NOW - timedelta(hours=1),
        expires_at=NOW + timedelta(hours=1),
    )


def _unsigned_request(mandate: AgentMandate, **updates: object) -> AgentPaymentRequest:
    values: dict[str, object] = {
        "request_id": "request-1",
        "payment_id": "agentic-payment-1",
        "agent_id": AGENT_ID,
        "key_id": KEY_ID,
        "mandate": mandate,
        "amount": Decimal("120.00"),
        "currency": "USD",
        "payee_id": PAYEE_ID,
        "cart_hash": CART_HASH,
        "nonce": "nonce-1",
        "created_at": NOW,
        "expires_at": NOW + timedelta(minutes=5),
        "prior_receipt_hash": "",
        "campaign_id": "00000000-0000-4000-8000-000000000301",
        "trace_id": "00000000-0000-4000-8000-000000000302",
        "actor_id": "00000000-0000-4000-8000-000000000303",
        "counterparty_id": "00000000-0000-4000-8000-000000000304",
        "signature": b"",
    }
    values.update(updates)
    return AgentPaymentRequest(**values)  # type: ignore[arg-type]


def _sign(request: AgentPaymentRequest, private_key: Ed25519PrivateKey) -> AgentPaymentRequest:
    return request.model_copy(
        update={"signature": private_key.sign(request.signing_bytes())}
    )


@pytest.fixture
def valid_request(
    mandate: AgentMandate,
    private_key: Ed25519PrivateKey,
) -> AgentPaymentRequest:
    return _sign(_unsigned_request(mandate), private_key)


def _verifier(
    mandate: AgentMandate,
    private_key: Ed25519PrivateKey,
) -> TrustVerifier:
    public_key = private_key.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw)
    return TrustVerifier(
        registered_agents={(AGENT_ID, KEY_ID): public_key},
        mandates={MANDATE_ID: mandate},
    )


def test_valid_request_returns_deterministic_synthetic_receipt(
    mandate: AgentMandate,
    private_key: Ed25519PrivateKey,
    valid_request: AgentPaymentRequest,
) -> None:
    first = _verifier(mandate, private_key).verify(valid_request, NOW)
    second = _verifier(mandate, private_key).verify(valid_request, NOW)

    assert first == second
    assert first.allowed is True
    assert first.reason_code is None
    assert len(first.receipt_hash) == 64
    assert first.previous_receipt_hash == ""
    assert first.receipt_hash != valid_request.signature.hex()


def test_public_trust_records_are_immutable(
    mandate: AgentMandate,
    valid_request: AgentPaymentRequest,
) -> None:
    receipt = IntegrityReceipt(
        request_id=valid_request.request_id,
        allowed=False,
        reason_code=ReasonCode.SIGNATURE_INVALID,
        receipt_hash="2" * 64,
        previous_receipt_hash="",
    )

    with pytest.raises(FrozenInstanceError):
        mandate.currency = "EUR"  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        valid_request.amount = Decimal("1.00")  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        receipt.allowed = True  # type: ignore[misc]


def test_identity_spoof_fails_before_signature(
    mandate: AgentMandate,
    private_key: Ed25519PrivateKey,
    valid_request: AgentPaymentRequest,
) -> None:
    spoofed = valid_request.model_copy(update={"agent_id": "attacker"})

    receipt = _verifier(mandate, private_key).verify(spoofed, NOW)

    assert receipt.reason_code is ReasonCode.AGENT_IDENTITY_MISMATCH


@pytest.mark.parametrize("signature", [b"bad", bytes(64)])
def test_invalid_or_malformed_signature_has_stable_reason(
    signature: bytes,
    mandate: AgentMandate,
    private_key: Ed25519PrivateKey,
    valid_request: AgentPaymentRequest,
) -> None:
    request = valid_request.model_copy(update={"signature": signature})

    receipt = _verifier(mandate, private_key).verify(request, NOW)

    assert receipt.reason_code is ReasonCode.SIGNATURE_INVALID


def test_mandate_escalation_fails_after_valid_signature(
    mandate: AgentMandate,
    private_key: Ed25519PrivateKey,
) -> None:
    escalated = mandate.model_copy(update={"max_amount": Decimal("999.00")})
    request = _sign(_unsigned_request(escalated), private_key)

    receipt = _verifier(mandate, private_key).verify(request, NOW)

    assert receipt.reason_code is ReasonCode.MANDATE_SCOPE_VIOLATION


@pytest.mark.parametrize(
    ("updates", "reason"),
    [
        ({"amount": Decimal("150.01")}, ReasonCode.AMOUNT_LIMIT_EXCEEDED),
        ({"currency": "EUR"}, ReasonCode.CURRENCY_MISMATCH),
        ({"payee_id": "attacker"}, ReasonCode.PAYEE_BINDING_MISMATCH),
        ({"cart_hash": "2" * 64}, ReasonCode.CART_HASH_MISMATCH),
    ],
)
def test_bound_request_fields_fail_with_ordered_reason(
    updates: dict[str, object],
    reason: ReasonCode,
    mandate: AgentMandate,
    private_key: Ed25519PrivateKey,
) -> None:
    request = _sign(_unsigned_request(mandate, **updates), private_key)

    receipt = _verifier(mandate, private_key).verify(request, NOW)

    assert receipt.reason_code is reason


def test_expiry_boundary_is_closed(
    mandate: AgentMandate,
    private_key: Ed25519PrivateKey,
) -> None:
    request = _sign(
        _unsigned_request(mandate, expires_at=NOW + timedelta(seconds=1)),
        private_key,
    )

    receipt = _verifier(mandate, private_key).verify(
        request,
        NOW + timedelta(seconds=1),
    )

    assert receipt.reason_code is ReasonCode.MANDATE_EXPIRED


@pytest.mark.parametrize("future_field", ["request", "mandate"])
def test_future_issued_request_or_mandate_fails_time_validity(
    future_field: str,
    mandate: AgentMandate,
    private_key: Ed25519PrivateKey,
) -> None:
    if future_field == "mandate":
        effective_mandate = mandate.model_copy(
            update={
                "issued_at": NOW + timedelta(seconds=1),
                "expires_at": NOW + timedelta(hours=1),
            }
        )
        verifier_mandate = effective_mandate
        created_at = NOW
    else:
        effective_mandate = mandate
        verifier_mandate = mandate
        created_at = NOW + timedelta(seconds=1)
    request = _sign(
        _unsigned_request(
            effective_mandate,
            created_at=created_at,
            expires_at=NOW + timedelta(minutes=5),
        ),
        private_key,
    )

    receipt = _verifier(verifier_mandate, private_key).verify(request, NOW)

    assert receipt.reason_code is ReasonCode.MANDATE_EXPIRED


def test_nonce_replay_rejects_without_changing_state(
    mandate: AgentMandate,
    private_key: Ed25519PrivateKey,
    valid_request: AgentPaymentRequest,
) -> None:
    verifier = _verifier(mandate, private_key)
    assert verifier.verify(valid_request, NOW).allowed
    accepted_state = verifier.dump_state()

    second = verifier.verify(valid_request, NOW)

    assert second.reason_code is ReasonCode.NONCE_REPLAY
    assert verifier.dump_state() == accepted_state


def test_broken_receipt_chain_does_not_consume_nonce(
    mandate: AgentMandate,
    private_key: Ed25519PrivateKey,
    valid_request: AgentPaymentRequest,
) -> None:
    verifier = _verifier(mandate, private_key)
    first = verifier.verify(valid_request, NOW)
    broken = _sign(
        _unsigned_request(
            mandate,
            request_id="request-2",
            payment_id="agentic-payment-2",
            nonce="nonce-2",
            prior_receipt_hash="f" * 64,
        ),
        private_key,
    )

    rejected = verifier.verify(broken, NOW)
    corrected = _sign(
        broken.model_copy(update={"prior_receipt_hash": first.receipt_hash}),
        private_key,
    )
    accepted = verifier.verify(corrected, NOW)

    assert rejected.reason_code is ReasonCode.RECEIPT_CHAIN_BROKEN
    assert accepted.allowed is True
    assert accepted.previous_receipt_hash == first.receipt_hash


@pytest.mark.parametrize(
    ("updates", "reason"),
    [
        (
            {"amount": Decimal("151.00"), "payee_id": "attacker"},
            ReasonCode.AMOUNT_LIMIT_EXCEEDED,
        ),
        (
            {"payee_id": "attacker", "cart_hash": "2" * 64},
            ReasonCode.PAYEE_BINDING_MISMATCH,
        ),
    ],
)
def test_check_order_precedence_is_stable(
    updates: dict[str, object],
    reason: ReasonCode,
    mandate: AgentMandate,
    private_key: Ed25519PrivateKey,
) -> None:
    request = _sign(_unsigned_request(mandate, **updates), private_key)

    receipt = _verifier(mandate, private_key).verify(request, NOW)

    assert receipt.reason_code is reason


def test_expiry_precedes_nonce_replay(
    mandate: AgentMandate,
    private_key: Ed25519PrivateKey,
) -> None:
    verifier = _verifier(mandate, private_key)
    request = _sign(
        _unsigned_request(mandate, expires_at=NOW + timedelta(seconds=1)),
        private_key,
    )
    assert verifier.verify(request, NOW).allowed

    replay = verifier.verify(request, NOW + timedelta(seconds=1))

    assert replay.reason_code is ReasonCode.MANDATE_EXPIRED


def test_nonce_replay_precedes_receipt_chain(
    mandate: AgentMandate,
    private_key: Ed25519PrivateKey,
    valid_request: AgentPaymentRequest,
) -> None:
    verifier = _verifier(mandate, private_key)
    assert verifier.verify(valid_request, NOW).allowed
    changed = _sign(
        valid_request.model_copy(update={"prior_receipt_hash": "f" * 64}),
        private_key,
    )

    replay = verifier.verify(changed, NOW)

    assert replay.reason_code is ReasonCode.NONCE_REPLAY


def test_verifier_state_round_trips_and_continues_receipt_chain(
    mandate: AgentMandate,
    private_key: Ed25519PrivateKey,
    valid_request: AgentPaymentRequest,
) -> None:
    first_verifier = _verifier(mandate, private_key)
    first = first_verifier.verify(valid_request, NOW)
    state = first_verifier.dump_state()
    second_verifier = _verifier(mandate, private_key)
    second_verifier.load_state(state)
    request = _sign(
        _unsigned_request(
            mandate,
            request_id="request-2",
            payment_id="agentic-payment-2",
            nonce="nonce-2",
            prior_receipt_hash=first.receipt_hash,
        ),
        private_key,
    )

    second = second_verifier.verify(request, NOW)

    assert second.allowed
    assert second.previous_receipt_hash == first.receipt_hash


def test_well_shaped_but_unverifiable_receipt_record_is_state_corrupt(
    mandate: AgentMandate,
    private_key: Ed25519PrivateKey,
) -> None:
    verifier = _verifier(mandate, private_key)
    forged_state = MappingProxyType(
        {
            "version": 1,
            "records": ((AGENT_ID, "forged-nonce", "", "2" * 64),),
        }
    )

    with pytest.raises(TrustVerifierStateError) as error:
        verifier.load_state(forged_state)

    assert error.value.code == "AGENTIC_STATE_CORRUPT"


@pytest.mark.parametrize(
    "corrupt",
    [
        {"version": 2, "records": ()},
        {"version": 1, "records": (("agent", "nonce", "", "bad"),)},
        {"version": 1, "records": (("unknown", "nonce", "", "2" * 64),)},
        {
            "version": 1,
            "records": (
                (AGENT_ID, "nonce", "", "2" * 64),
                (AGENT_ID, "nonce", "2" * 64, "3" * 64),
            ),
        },
    ],
)
def test_corrupt_verifier_state_fails_with_stable_error(corrupt: dict[str, object]) -> None:
    verifier = TrustVerifier(registered_agents={}, mandates={})

    with pytest.raises(TrustVerifierStateError) as error:
        verifier.load_state(MappingProxyType(corrupt))

    assert error.value.code == "AGENTIC_STATE_CORRUPT"


def test_boundaries_reject_subclasses_nonfinite_and_non_utc(
    mandate: AgentMandate,
) -> None:
    class MutableStr(str):
        pass

    class DecimalSubclass(Decimal):
        pass

    with pytest.raises(TypeError, match="agent_id must be an exact string"):
        mandate.model_copy(update={"agent_id": MutableStr(AGENT_ID)})
    with pytest.raises(TypeError, match="max_amount must be an exact Decimal"):
        mandate.model_copy(update={"max_amount": DecimalSubclass("1.00")})
    with pytest.raises(ValueError, match="finite and positive"):
        mandate.model_copy(update={"max_amount": Decimal("NaN")})
    with pytest.raises(ValueError, match="UTC"):
        mandate.model_copy(
            update={"issued_at": NOW.replace(tzinfo=timezone(timedelta(hours=1)))}
        )


def test_verifier_rejects_malformed_registered_public_key(
    mandate: AgentMandate,
) -> None:
    with pytest.raises(ValueError, match="Ed25519 public key"):
        TrustVerifier(
            registered_agents={(AGENT_ID, KEY_ID): b"short"},
            mandates={MANDATE_ID: mandate},
        )
