"""Ordered fail-closed checks for delegated agent payment integrity."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, fields, replace
from datetime import UTC, datetime, timedelta, timezone
from decimal import Decimal
from types import MappingProxyType
from typing import cast

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat

from apar.contracts.decisions import ReasonCode
from apar.trust.verifier import (
    AgentMandate,
    AgentPaymentRequest,
    AuthenticationEvidence,
    AuthenticationOutcome,
    AuthenticationRequirement,
    IntegrityReceipt,
    ReceiptOutcome,
    TrustCommitPlan,
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
MERCHANT_ID = "merchant-entity-1"
CART_HASH = "1" * 64
PAYMENT_INTENT_HASH = "3" * 64
CREDENTIAL_ID = "synthetic-token-1"
CREDENTIAL_SCOPE = "single_merchant_single_use"
CATEGORY = "TRAVEL"
PRODUCT_ID = "flight-1"
USER_ENTITY_ID = "00000000-0000-4000-8000-000000000303"
BENEFICIARY_ENTITY_ID = "00000000-0000-4000-8000-000000000304"
AUTH_EVIDENCE_ID = "auth-evidence-1"


class AgentPaymentRequestSubclass(AgentPaymentRequest):
    pass


class IntegrityReceiptSubclass(IntegrityReceipt):
    pass


class DatetimeSubclass(datetime):
    pass


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
        user_entity_id=USER_ENTITY_ID,
        beneficiary_entity_id=BENEFICIARY_ENTITY_ID,
        merchant_id=MERCHANT_ID,
        payee_id=PAYEE_ID,
        cart_hash=CART_HASH,
        payment_intent_hash=PAYMENT_INTENT_HASH,
        permitted_categories=(CATEGORY,),
        permitted_products=(PRODUCT_ID,),
        credential_id=CREDENTIAL_ID,
        credential_scope=CREDENTIAL_SCOPE,
        required_authentication=AuthenticationRequirement.STEP_UP,
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
        "merchant_id": MERCHANT_ID,
        "payee_id": PAYEE_ID,
        "cart_hash": CART_HASH,
        "payment_intent_hash": PAYMENT_INTENT_HASH,
        "category": CATEGORY,
        "product_id": PRODUCT_ID,
        "credential_id": CREDENTIAL_ID,
        "credential_scope": CREDENTIAL_SCOPE,
        "consent_ref": CONSENT_REF,
        "authentication_evidence_ref": AUTH_EVIDENCE_ID,
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


def _request_subclass(request: AgentPaymentRequest) -> AgentPaymentRequestSubclass:
    values = {field.name: getattr(request, field.name) for field in fields(request)}
    return AgentPaymentRequestSubclass(**values)


def _receipt_subclass(receipt: IntegrityReceipt) -> IntegrityReceiptSubclass:
    values = {field.name: getattr(receipt, field.name) for field in fields(receipt)}
    return IntegrityReceiptSubclass(**values)


def _datetime_subclass(value: datetime) -> DatetimeSubclass:
    return DatetimeSubclass(
        value.year,
        value.month,
        value.day,
        value.hour,
        value.minute,
        value.second,
        value.microsecond,
        tzinfo=UTC,
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
    *,
    authentication_evidence: dict[str, AuthenticationEvidence] | None = None,
) -> TrustVerifier:
    public_key = private_key.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw)
    return TrustVerifier(
        registered_agents={(AGENT_ID, KEY_ID): public_key},
        mandates={MANDATE_ID: mandate},
        authentication_evidence=(
            authentication_evidence
            if authentication_evidence is not None
            else {AUTH_EVIDENCE_ID: _authentication_evidence()}
        ),
    )


def _authentication_evidence(**updates: object) -> AuthenticationEvidence:
    values: dict[str, object] = {
        "evidence_id": AUTH_EVIDENCE_ID,
        "agent_id": AGENT_ID,
        "user_ref": USER_REF,
        "mandate_id": MANDATE_ID,
        "nonce": "nonce-1",
        "payment_intent_hash": PAYMENT_INTENT_HASH,
        "request_id": "request-1",
        "outcome": AuthenticationOutcome.STEP_UP_VERIFIED,
        "issued_at": NOW - timedelta(seconds=5),
        "expires_at": NOW + timedelta(minutes=2),
    }
    values.update(updates)
    return AuthenticationEvidence(**values)  # type: ignore[arg-type]


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
        request_hash="3" * 64,
        signature_hash="4" * 64,
        outcome=ReceiptOutcome.REJECTED,
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


@pytest.mark.parametrize(
    ("updates", "reason"),
    [
        ({"merchant_id": "substituted-merchant"}, ReasonCode.MERCHANT_BINDING_MISMATCH),
        ({"payee_id": "substituted-payee"}, ReasonCode.PAYEE_BINDING_MISMATCH),
        ({"category": "GAMBLING"}, ReasonCode.CATEGORY_SCOPE_VIOLATION),
        ({"product_id": "unapproved-product"}, ReasonCode.PRODUCT_SCOPE_VIOLATION),
        (
            {"payment_intent_hash": "4" * 64},
            ReasonCode.PAYMENT_INTENT_HASH_MISMATCH,
        ),
        ({"credential_id": "substituted-token"}, ReasonCode.CREDENTIAL_BINDING_MISMATCH),
        ({"credential_scope": "multi_merchant"}, ReasonCode.TOKEN_SCOPE_VIOLATION),
        ({"consent_ref": "different-consent"}, ReasonCode.CONSENT_BINDING_MISMATCH),
        (
            {"authentication_evidence_ref": "missing-evidence"},
            ReasonCode.AUTHENTICATION_EVIDENCE_MISSING,
        ),
    ],
)
def test_solution_level_agentic_bindings_reject_deterministically(
    updates: dict[str, object],
    reason: ReasonCode,
    mandate: AgentMandate,
    private_key: Ed25519PrivateKey,
) -> None:
    request = _sign(_unsigned_request(mandate, **updates), private_key)

    receipt = _verifier(mandate, private_key).verify(request, NOW)

    assert receipt.reason_code is reason


def test_missing_consent_is_rejected_at_canonical_boundary(
    mandate: AgentMandate,
) -> None:
    with pytest.raises(ValueError, match="consent_ref must not be empty"):
        _unsigned_request(mandate, consent_ref="")


def test_missing_authentication_reference_is_rejected_at_canonical_boundary(
    mandate: AgentMandate,
) -> None:
    with pytest.raises(ValueError, match="authentication_evidence_ref must not be empty"):
        _unsigned_request(mandate, authentication_evidence_ref="")


@pytest.mark.parametrize(
    ("evidence", "reason"),
    [
        (
            _authentication_evidence(user_ref="different-user"),
            ReasonCode.AUTHENTICATION_EVIDENCE_MISMATCH,
        ),
        (
            _authentication_evidence(
                issued_at=NOW - timedelta(minutes=2),
                expires_at=NOW,
            ),
            ReasonCode.AUTHENTICATION_EVIDENCE_EXPIRED,
        ),
    ],
)
def test_trusted_authentication_evidence_mismatch_or_expiry_rejects(
    evidence: AuthenticationEvidence,
    reason: ReasonCode,
    mandate: AgentMandate,
    private_key: Ed25519PrivateKey,
    valid_request: AgentPaymentRequest,
) -> None:
    verifier = _verifier(
        mandate,
        private_key,
        authentication_evidence={AUTH_EVIDENCE_ID: evidence},
    )

    receipt = verifier.verify(valid_request, NOW)

    assert receipt.reason_code is reason


def test_no_step_up_mandate_needs_no_evidence_registry_entry(
    mandate: AgentMandate,
    private_key: Ed25519PrivateKey,
) -> None:
    no_step_up = mandate.model_copy(
        update={"required_authentication": AuthenticationRequirement.NONE}
    )
    request = _sign(
        _unsigned_request(
            no_step_up,
            authentication_evidence_ref=None,
        ),
        private_key,
    )
    public_key = private_key.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw)
    verifier = TrustVerifier(
        registered_agents={(AGENT_ID, KEY_ID): public_key},
        mandates={MANDATE_ID: no_step_up},
        authentication_evidence={},
    )
    receipt = verifier.verify(request, NOW)
    restored = TrustVerifier(
        registered_agents={(AGENT_ID, KEY_ID): public_key},
        mandates={MANDATE_ID: no_step_up},
        authentication_evidence={},
    )

    restored.load_state(verifier.dump_state())

    assert receipt.allowed
    assert restored.dump_state() == verifier.dump_state()


def test_no_step_up_state_rejects_mutable_empty_reference_alias(
    mandate: AgentMandate,
    private_key: Ed25519PrivateKey,
) -> None:
    class MutableStr(str):
        pass

    no_step_up = mandate.model_copy(
        update={"required_authentication": AuthenticationRequirement.NONE}
    )
    request = _sign(
        _unsigned_request(no_step_up, authentication_evidence_ref=None), private_key
    )
    public_key = private_key.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw)
    verifier = TrustVerifier(
        registered_agents={(AGENT_ID, KEY_ID): public_key},
        mandates={MANDATE_ID: no_step_up},
        authentication_evidence={},
    )
    verifier.verify(request, NOW)
    record = list(cast(tuple[tuple[str, ...], ...], verifier.dump_state()["records"])[0])
    record[2] = MutableStr("")
    restored = TrustVerifier(
        registered_agents={(AGENT_ID, KEY_ID): public_key},
        mandates={MANDATE_ID: no_step_up},
        authentication_evidence={},
    )

    with pytest.raises(TrustVerifierStateError):
        restored.load_state({"version": 1, "records": (tuple(record),)})


@pytest.mark.parametrize("reference", ["arbitrary-reference", AUTH_EVIDENCE_ID])
def test_no_step_up_request_rejects_any_authentication_evidence_reference(
    reference: str,
    mandate: AgentMandate,
) -> None:
    no_step_up = mandate.model_copy(
        update={"required_authentication": AuthenticationRequirement.NONE}
    )

    with pytest.raises(ValueError, match="must be None"):
        _unsigned_request(no_step_up, authentication_evidence_ref=reference)


def test_no_step_up_activity_cannot_poison_other_agents_step_up_evidence(
    mandate: AgentMandate,
    private_key: Ed25519PrivateKey,
    valid_request: AgentPaymentRequest,
) -> None:
    no_step_key = Ed25519PrivateKey.from_private_bytes(bytes(reversed(range(32))))
    no_step_agent = "agent-no-step"
    no_step_mandate = mandate.model_copy(
        update={
            "mandate_id": "mandate-no-step",
            "agent_id": no_step_agent,
            "user_ref": "user-no-step",
            "user_entity_id": "00000000-0000-4000-8000-000000000351",
            "beneficiary_entity_id": "00000000-0000-4000-8000-000000000352",
            "required_authentication": AuthenticationRequirement.NONE,
        }
    )
    no_step_request = _sign(
        _unsigned_request(
            no_step_mandate,
            request_id="request-no-step",
            payment_id="payment-no-step",
            agent_id=no_step_agent,
            key_id="key-no-step",
            nonce="nonce-no-step",
            authentication_evidence_ref=None,
            actor_id=no_step_mandate.user_entity_id,
            counterparty_id=no_step_mandate.beneficiary_entity_id,
        ),
        no_step_key,
    )
    verifier = TrustVerifier(
        registered_agents={
            (AGENT_ID, KEY_ID): private_key.public_key().public_bytes(
                Encoding.Raw, PublicFormat.Raw
            ),
            (no_step_agent, "key-no-step"): no_step_key.public_key().public_bytes(
                Encoding.Raw, PublicFormat.Raw
            ),
        },
        mandates={
            MANDATE_ID: mandate,
            no_step_mandate.mandate_id: no_step_mandate,
        },
        authentication_evidence={AUTH_EVIDENCE_ID: _authentication_evidence()},
    )

    no_step_receipt = verifier.verify(no_step_request, NOW)
    step_up_receipt = verifier.verify(valid_request, NOW)

    assert no_step_receipt.allowed
    assert step_up_receipt.allowed


def test_authentication_evidence_is_single_use_across_new_request_and_nonce(
    mandate: AgentMandate,
    private_key: Ed25519PrivateKey,
    valid_request: AgentPaymentRequest,
) -> None:
    verifier = _verifier(mandate, private_key)
    first = verifier.verify(valid_request, NOW)
    replayed = _sign(
        _unsigned_request(
            mandate,
            request_id="request-2",
            payment_id="agentic-payment-2",
            nonce="nonce-2",
            prior_receipt_hash=first.receipt_hash,
            authentication_evidence_ref=AUTH_EVIDENCE_ID,
        ),
        private_key,
    )

    receipt = verifier.verify(replayed, NOW)

    assert receipt.reason_code is ReasonCode.AUTHENTICATION_EVIDENCE_REPLAY


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


@pytest.mark.parametrize(
    ("created_at", "expires_at", "reason"),
    [
        (
            NOW - timedelta(hours=1, microseconds=1),
            NOW + timedelta(minutes=5),
            ReasonCode.MANDATE_TIME_SCOPE_VIOLATION,
        ),
        (
            NOW,
            NOW + timedelta(hours=1, microseconds=1),
            ReasonCode.MANDATE_TIME_SCOPE_VIOLATION,
        ),
    ],
)
def test_request_window_must_be_nested_inside_mandate(
    created_at: datetime,
    expires_at: datetime,
    reason: ReasonCode,
    mandate: AgentMandate,
    private_key: Ed25519PrivateKey,
) -> None:
    request = _sign(
        _unsigned_request(mandate, created_at=created_at, expires_at=expires_at),
        private_key,
    )

    receipt = _verifier(mandate, private_key).verify(request, NOW)

    assert receipt.reason_code is reason


def test_request_window_accepts_exact_mandate_boundaries(
    mandate: AgentMandate,
    private_key: Ed25519PrivateKey,
) -> None:
    request = _sign(
        _unsigned_request(
            mandate,
            created_at=mandate.issued_at,
            expires_at=mandate.expires_at,
        ),
        private_key,
    )

    receipt = _verifier(mandate, private_key).verify(request, NOW)

    assert receipt.allowed


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

    expected = (
        ReasonCode.MANDATE_TIME_SCOPE_VIOLATION
        if future_field == "mandate"
        else ReasonCode.MANDATE_EXPIRED
    )
    assert receipt.reason_code is expected


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
    second_evidence = _authentication_evidence(
        evidence_id="auth-evidence-2",
        nonce="nonce-2",
        request_id="request-2",
    )
    verifier = _verifier(
        mandate,
        private_key,
        authentication_evidence={
            AUTH_EVIDENCE_ID: _authentication_evidence(),
            second_evidence.evidence_id: second_evidence,
        },
    )
    first = verifier.verify(valid_request, NOW)
    broken = _sign(
        _unsigned_request(
            mandate,
            request_id="request-2",
            payment_id="agentic-payment-2",
            nonce="nonce-2",
            authentication_evidence_ref=second_evidence.evidence_id,
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
    second_evidence = _authentication_evidence(
        evidence_id="auth-evidence-2",
        nonce="nonce-2",
        request_id="request-2",
    )
    evidence = {
        AUTH_EVIDENCE_ID: _authentication_evidence(),
        second_evidence.evidence_id: second_evidence,
    }
    first_verifier = _verifier(
        mandate, private_key, authentication_evidence=evidence
    )
    first = first_verifier.verify(valid_request, NOW)
    state = first_verifier.dump_state()
    second_verifier = _verifier(
        mandate, private_key, authentication_evidence=evidence
    )
    second_verifier.load_state(state)
    request = _sign(
        _unsigned_request(
            mandate,
            request_id="request-2",
            payment_id="agentic-payment-2",
            nonce="nonce-2",
            authentication_evidence_ref=second_evidence.evidence_id,
            prior_receipt_hash=first.receipt_hash,
        ),
        private_key,
    )

    second = second_verifier.verify(request, NOW)

    assert second.allowed
    assert second.previous_receipt_hash == first.receipt_hash


def test_nonce_only_state_record_relabel_breaks_receipt_commitment(
    mandate: AgentMandate,
    private_key: Ed25519PrivateKey,
    valid_request: AgentPaymentRequest,
) -> None:
    source = _verifier(mandate, private_key)
    source.verify(valid_request, NOW)
    state = source.dump_state()
    records = cast(tuple[tuple[str, ...], ...], state["records"])
    changed = list(records[0])
    changed[1] = "relabeled-nonce"
    corrupted = MappingProxyType({"version": 1, "records": (tuple(changed),)})

    with pytest.raises(TrustVerifierStateError) as error:
        _verifier(mandate, private_key).load_state(corrupted)

    assert "not reproducible" in str(error.value.__cause__)


def test_agent_only_state_record_relabel_breaks_receipt_commitment(
    mandate: AgentMandate,
    private_key: Ed25519PrivateKey,
    valid_request: AgentPaymentRequest,
) -> None:
    second_key = Ed25519PrivateKey.from_private_bytes(bytes(reversed(range(32))))
    second_agent = "agent-registered-2"
    public_keys = {
        (AGENT_ID, KEY_ID): private_key.public_key().public_bytes(
            Encoding.Raw, PublicFormat.Raw
        ),
        (second_agent, "key-2"): second_key.public_key().public_bytes(
            Encoding.Raw, PublicFormat.Raw
        ),
    }
    source = _verifier(mandate, private_key)
    source.verify(valid_request, NOW)
    records = cast(tuple[tuple[str, ...], ...], source.dump_state()["records"])
    changed = list(records[0])
    changed[0] = second_agent
    verifier = TrustVerifier(
        registered_agents=public_keys,
        mandates={MANDATE_ID: mandate},
        authentication_evidence={AUTH_EVIDENCE_ID: _authentication_evidence()},
    )

    with pytest.raises(TrustVerifierStateError) as error:
        verifier.load_state({"version": 1, "records": (tuple(changed),)})

    assert "not reproducible" in str(error.value.__cause__)


def test_legitimate_multi_agent_interleaved_receipt_chains_round_trip(
    mandate: AgentMandate,
    private_key: Ed25519PrivateKey,
    valid_request: AgentPaymentRequest,
) -> None:
    second_key = Ed25519PrivateKey.from_private_bytes(bytes(reversed(range(32))))
    second_agent = "agent-registered-2"
    second_mandate = mandate.model_copy(
        update={
            "mandate_id": "mandate-2",
            "agent_id": second_agent,
            "user_ref": "user-2",
            "user_entity_id": "00000000-0000-4000-8000-000000000305",
            "beneficiary_entity_id": "00000000-0000-4000-8000-000000000306",
        }
    )
    evidence_a2 = _authentication_evidence(
        evidence_id="auth-evidence-a2",
        nonce="nonce-a2",
        request_id="request-a2",
    )
    evidence_b1 = _authentication_evidence(
        evidence_id="auth-evidence-b1",
        agent_id=second_agent,
        user_ref="user-2",
        mandate_id="mandate-2",
        nonce="nonce-b1",
        request_id="request-b1",
    )
    public_keys = {
        (AGENT_ID, KEY_ID): private_key.public_key().public_bytes(
            Encoding.Raw, PublicFormat.Raw
        ),
        (second_agent, "key-2"): second_key.public_key().public_bytes(
            Encoding.Raw, PublicFormat.Raw
        ),
    }
    evidence = {
        AUTH_EVIDENCE_ID: _authentication_evidence(),
        evidence_a2.evidence_id: evidence_a2,
        evidence_b1.evidence_id: evidence_b1,
    }
    verifier = TrustVerifier(
        registered_agents=public_keys,
        mandates={MANDATE_ID: mandate, second_mandate.mandate_id: second_mandate},
        authentication_evidence=evidence,
    )
    first_a = verifier.verify(valid_request, NOW)
    unsigned_b = _unsigned_request(
        second_mandate,
        request_id="request-b1",
        payment_id="payment-b1",
        agent_id=second_agent,
        key_id="key-2",
        nonce="nonce-b1",
        authentication_evidence_ref=evidence_b1.evidence_id,
        actor_id=second_mandate.user_entity_id,
        counterparty_id=second_mandate.beneficiary_entity_id,
    )
    first_b = _sign(unsigned_b, second_key)
    receipt_b = verifier.verify(first_b, NOW)
    request_a2 = _sign(
        _unsigned_request(
            mandate,
            request_id="request-a2",
            payment_id="payment-a2",
            nonce="nonce-a2",
            authentication_evidence_ref=evidence_a2.evidence_id,
            prior_receipt_hash=first_a.receipt_hash,
        ),
        private_key,
    )
    second_a = verifier.verify(request_a2, NOW)
    frozen = verifier.dump_state()
    restored = TrustVerifier(
        registered_agents=public_keys,
        mandates={MANDATE_ID: mandate, second_mandate.mandate_id: second_mandate},
        authentication_evidence=evidence,
    )

    restored.load_state(frozen)

    assert receipt_b.allowed and second_a.allowed
    assert restored.dump_state() == frozen


def test_well_shaped_but_unverifiable_receipt_record_is_state_corrupt(
    mandate: AgentMandate,
    private_key: Ed25519PrivateKey,
) -> None:
    verifier = _verifier(mandate, private_key)
    request_hash = "3" * 64
    signature_hash = "4" * 64
    forged_state = MappingProxyType(
        {
            "version": 1,
            "records": (
                (
                    AGENT_ID,
                    "forged-nonce",
                    "auth-forged",
                    "",
                    request_hash,
                    signature_hash,
                    ReceiptOutcome.APPROVE.value,
                    "2" * 64,
                ),
            ),
        }
    )

    with pytest.raises(TrustVerifierStateError) as error:
        verifier.load_state(forged_state)

    assert error.value.code == "AGENTIC_STATE_CORRUPT"
    assert "not reproducible" in str(error.value.__cause__)


def test_outcome_receipt_tamper_fails_without_consuming_nonce(
    mandate: AgentMandate,
    private_key: Ed25519PrivateKey,
    valid_request: AgentPaymentRequest,
) -> None:
    verifier = _verifier(mandate, private_key)
    preview = verifier.preview(valid_request, NOW)
    tampered = preview.model_copy(update={"receipt_hash": "f" * 64})
    state_before = verifier.dump_state()

    rejected = verifier.commit(valid_request, tampered, ReceiptOutcome.APPROVE, NOW)
    state_after_rejection = verifier.dump_state()
    accepted = verifier.commit(valid_request, preview, ReceiptOutcome.APPROVE, NOW)

    assert rejected.reason_code is ReasonCode.EXECUTION_RECEIPT_MISMATCH
    assert state_after_rejection == state_before
    assert verifier.dump_state() == state_before
    assert accepted.reason_code is ReasonCode.EXECUTION_RECEIPT_MISMATCH


def test_commit_rejects_preview_not_issued_by_same_verifier_instance(
    mandate: AgentMandate,
    private_key: Ed25519PrivateKey,
    valid_request: AgentPaymentRequest,
) -> None:
    issuer = _verifier(mandate, private_key)
    committing = _verifier(mandate, private_key)
    foreign_preview = issuer.preview(valid_request, NOW)
    state_before = committing.dump_state()

    receipt = committing.commit(
        valid_request,
        foreign_preview,
        ReceiptOutcome.APPROVE,
        NOW,
    )

    assert receipt.reason_code is ReasonCode.EXECUTION_RECEIPT_MISMATCH
    assert committing.dump_state() == state_before


def test_prepared_commit_is_single_use_and_discardable(
    mandate: AgentMandate,
    private_key: Ed25519PrivateKey,
    valid_request: AgentPaymentRequest,
) -> None:
    verifier = _verifier(mandate, private_key)
    preview = verifier.preview(valid_request, NOW)
    prepared = verifier.prepare_commit(
        valid_request, preview, ReceiptOutcome.APPROVE, NOW
    )
    assert not isinstance(prepared, IntegrityReceipt)
    projected = verifier.projected_state(prepared)

    verifier.discard_commit(prepared)

    assert verifier.dump_state()["records"] == ()
    with pytest.raises(ValueError, match="not issued or was already consumed"):
        verifier.apply_commit(prepared)
    assert projected["records"] != ()

    retry_preview = verifier.preview(valid_request, NOW)
    receipt = verifier.commit(
        valid_request, retry_preview, ReceiptOutcome.APPROVE, NOW
    )

    assert receipt.allowed
    assert receipt.outcome is ReceiptOutcome.APPROVE


def test_equality_equivalent_reconstructed_plan_is_not_an_issued_capability(
    mandate: AgentMandate,
    private_key: Ed25519PrivateKey,
    valid_request: AgentPaymentRequest,
) -> None:
    verifier = _verifier(mandate, private_key)
    preview = verifier.preview(valid_request, NOW)
    prepared = verifier.prepare_commit(
        valid_request, preview, ReceiptOutcome.APPROVE, NOW
    )
    assert isinstance(prepared, TrustCommitPlan)
    reconstructed = replace(prepared)

    with pytest.raises(ValueError, match="not issued or was already consumed"):
        verifier.apply_commit(reconstructed)
    with pytest.raises(ValueError, match="not issued or was already consumed"):
        verifier.apply_commit(prepared)

    assert verifier.dump_state()["records"] == ()


def test_projected_state_requires_exact_issued_plan_instance(
    mandate: AgentMandate,
    private_key: Ed25519PrivateKey,
    valid_request: AgentPaymentRequest,
) -> None:
    verifier = _verifier(mandate, private_key)
    preview = verifier.preview(valid_request, NOW)
    prepared = verifier.prepare_commit(
        valid_request, preview, ReceiptOutcome.APPROVE, NOW
    )
    assert isinstance(prepared, TrustCommitPlan)

    with pytest.raises(ValueError, match="not issued or was already consumed"):
        verifier.projected_state(replace(prepared))

    assert verifier.projected_state(prepared)["records"] != ()


def test_discard_requires_exact_plan_identity_and_revokes_before_rejection(
    mandate: AgentMandate,
    private_key: Ed25519PrivateKey,
    valid_request: AgentPaymentRequest,
) -> None:
    verifier = _verifier(mandate, private_key)
    preview = verifier.preview(valid_request, NOW)
    prepared = verifier.prepare_commit(
        valid_request, preview, ReceiptOutcome.APPROVE, NOW
    )
    assert isinstance(prepared, TrustCommitPlan)

    with pytest.raises(ValueError, match="not issued or was already consumed"):
        verifier.discard_commit(replace(prepared))
    with pytest.raises(ValueError, match="not issued or was already consumed"):
        verifier.discard_commit(prepared)

    assert verifier.dump_state()["records"] == ()


def test_cross_verifier_plan_rejects_and_revokes_receiving_verifier_plan(
    mandate: AgentMandate,
    private_key: Ed25519PrivateKey,
    valid_request: AgentPaymentRequest,
) -> None:
    first = _verifier(mandate, private_key)
    second = _verifier(mandate, private_key)
    first_preview = first.preview(valid_request, NOW)
    second_preview = second.preview(valid_request, NOW)
    first_plan = first.prepare_commit(
        valid_request, first_preview, ReceiptOutcome.APPROVE, NOW
    )
    second_plan = second.prepare_commit(
        valid_request, second_preview, ReceiptOutcome.APPROVE, NOW
    )
    assert isinstance(first_plan, TrustCommitPlan)
    assert isinstance(second_plan, TrustCommitPlan)

    with pytest.raises(ValueError, match="not issued or was already consumed"):
        second.apply_commit(first_plan)
    with pytest.raises(ValueError, match="not issued or was already consumed"):
        second.apply_commit(second_plan)

    assert second.dump_state()["records"] == ()
    assert first.apply_commit(first_plan).allowed


def test_apply_uses_internal_canonical_plan_not_mutated_public_view(
    mandate: AgentMandate,
    private_key: Ed25519PrivateKey,
    valid_request: AgentPaymentRequest,
) -> None:
    verifier = _verifier(mandate, private_key)
    preview = verifier.preview(valid_request, NOW)
    prepared = verifier.prepare_commit(
        valid_request, preview, ReceiptOutcome.APPROVE, NOW
    )
    assert isinstance(prepared, TrustCommitPlan)
    canonical_receipt = prepared.receipt
    object.__setattr__(
        prepared,
        "receipt",
        canonical_receipt.model_copy(update={"receipt_hash": "f" * 64}),
    )

    applied = verifier.apply_commit(prepared)
    restored = _verifier(mandate, private_key)
    restored.load_state(verifier.dump_state())

    assert applied == canonical_receipt
    assert restored.dump_state() == verifier.dump_state()


def test_trust_commit_plan_rejects_noncanonical_receipt_and_subclasses(
    mandate: AgentMandate,
    private_key: Ed25519PrivateKey,
    valid_request: AgentPaymentRequest,
) -> None:
    verifier = _verifier(mandate, private_key)
    preview = verifier.preview(valid_request, NOW)
    prepared = verifier.prepare_commit(
        valid_request, preview, ReceiptOutcome.APPROVE, NOW
    )
    assert isinstance(prepared, TrustCommitPlan)

    with pytest.raises(TypeError, match="exact IntegrityReceipt"):
        TrustCommitPlan(cast(IntegrityReceipt, object()))
    with pytest.raises(TypeError, match="exact IntegrityReceipt"):
        TrustCommitPlan(_receipt_subclass(prepared.receipt))

    class MutableStr(str):
        pass

    aliased_receipt = object.__new__(IntegrityReceipt)
    for field in fields(prepared.receipt):
        value = getattr(prepared.receipt, field.name)
        if field.name == "receipt_hash":
            value = MutableStr(value)
        object.__setattr__(aliased_receipt, field.name, value)

    with pytest.raises(ValueError, match="receipt_hash"):
        TrustCommitPlan(aliased_receipt)


@pytest.mark.parametrize("malformed", ["object", "subclass"])
def test_malformed_apply_attempt_revokes_plan_before_validation(
    malformed: str,
    mandate: AgentMandate,
    private_key: Ed25519PrivateKey,
    valid_request: AgentPaymentRequest,
) -> None:
    verifier = _verifier(mandate, private_key)
    preview = verifier.preview(valid_request, NOW)
    prepared = verifier.prepare_commit(
        valid_request, preview, ReceiptOutcome.APPROVE, NOW
    )
    assert isinstance(prepared, TrustCommitPlan)
    candidate: object = object()
    if malformed == "subclass":
        class PlanSubclass(TrustCommitPlan):
            pass

        candidate = PlanSubclass(prepared.receipt)

    with pytest.raises((TypeError, ValueError)):
        verifier.apply_commit(cast(TrustCommitPlan, candidate))
    with pytest.raises(ValueError, match="not issued or was already consumed"):
        verifier.apply_commit(prepared)

    assert verifier.dump_state()["records"] == ()


def test_valid_plan_applies_once_and_round_trips_state(
    mandate: AgentMandate,
    private_key: Ed25519PrivateKey,
    valid_request: AgentPaymentRequest,
) -> None:
    verifier = _verifier(mandate, private_key)
    preview = verifier.preview(valid_request, NOW)
    prepared = verifier.prepare_commit(
        valid_request, preview, ReceiptOutcome.APPROVE, NOW
    )
    assert isinstance(prepared, TrustCommitPlan)

    receipt = verifier.apply_commit(prepared)
    restored = _verifier(mandate, private_key)
    restored.load_state(verifier.dump_state())

    assert receipt.outcome is ReceiptOutcome.APPROVE
    assert restored.dump_state() == verifier.dump_state()
    with pytest.raises(ValueError, match="not issued or was already consumed"):
        verifier.apply_commit(prepared)


def test_new_preview_attempt_revokes_abandoned_preview_even_when_it_rejects(
    mandate: AgentMandate,
    private_key: Ed25519PrivateKey,
    valid_request: AgentPaymentRequest,
) -> None:
    verifier = _verifier(mandate, private_key)
    abandoned = verifier.preview(valid_request, NOW)
    invalid = _sign(
        _unsigned_request(mandate, payee_id="substituted-payee"), private_key
    )

    rejected = verifier.preview(invalid, NOW)
    stale_commit = verifier.commit(
        valid_request, abandoned, ReceiptOutcome.APPROVE, NOW
    )

    assert rejected.reason_code is ReasonCode.PAYEE_BINDING_MISMATCH
    assert stale_commit.reason_code is ReasonCode.EXECUTION_RECEIPT_MISMATCH
    assert verifier.dump_state()["records"] == ()


@pytest.mark.parametrize(
    "malformed",
    ["request_object", "request_subclass", "now_object", "now_subclass"],
)
def test_every_malformed_preview_attempt_revokes_existing_capability(
    malformed: str,
    mandate: AgentMandate,
    private_key: Ed25519PrivateKey,
    valid_request: AgentPaymentRequest,
) -> None:
    verifier = _verifier(mandate, private_key)
    existing = verifier.preview(valid_request, NOW)
    request: object = valid_request
    now: object = NOW
    if malformed == "request_object":
        request = object()
    elif malformed == "request_subclass":
        request = _request_subclass(valid_request)
    elif malformed == "now_object":
        now = object()
    else:
        now = _datetime_subclass(NOW)

    with pytest.raises((TypeError, ValueError)):
        verifier.preview(
            cast(AgentPaymentRequest, request),
            cast(datetime, now),
        )

    stale = verifier.commit(valid_request, existing, ReceiptOutcome.APPROVE, NOW)

    assert stale.reason_code is ReasonCode.EXECUTION_RECEIPT_MISMATCH
    assert verifier.dump_state()["records"] == ()


@pytest.mark.parametrize(
    "malformed",
    [
        "request_object",
        "request_subclass",
        "receipt_object",
        "receipt_subclass",
        "outcome_object",
        "now_object",
        "now_subclass",
    ],
)
def test_every_malformed_prepare_attempt_revokes_existing_capability(
    malformed: str,
    mandate: AgentMandate,
    private_key: Ed25519PrivateKey,
    valid_request: AgentPaymentRequest,
) -> None:
    verifier = _verifier(mandate, private_key)
    existing = verifier.preview(valid_request, NOW)
    request: object = valid_request
    preview: object = existing
    outcome: object = ReceiptOutcome.APPROVE
    now: object = NOW
    if malformed == "request_object":
        request = object()
    elif malformed == "request_subclass":
        request = _request_subclass(valid_request)
    elif malformed == "receipt_object":
        preview = object()
    elif malformed == "receipt_subclass":
        preview = _receipt_subclass(existing)
    elif malformed == "outcome_object":
        outcome = "approve"
    elif malformed == "now_object":
        now = object()
    else:
        now = _datetime_subclass(NOW)

    with pytest.raises((TypeError, ValueError)):
        verifier.prepare_commit(
            cast(AgentPaymentRequest, request),
            cast(IntegrityReceipt, preview),
            cast(ReceiptOutcome, outcome),
            cast(datetime, now),
        )

    stale = verifier.prepare_commit(
        valid_request, existing, ReceiptOutcome.APPROVE, NOW
    )

    assert isinstance(stale, IntegrityReceipt)
    assert stale.reason_code is ReasonCode.EXECUTION_RECEIPT_MISMATCH
    assert verifier.dump_state()["records"] == ()


@pytest.mark.parametrize(
    "malformed",
    [
        "request_object",
        "request_subclass",
        "receipt_object",
        "receipt_subclass",
        "outcome_object",
        "now_object",
        "now_subclass",
    ],
)
def test_every_malformed_commit_attempt_revokes_existing_capability(
    malformed: str,
    mandate: AgentMandate,
    private_key: Ed25519PrivateKey,
    valid_request: AgentPaymentRequest,
) -> None:
    verifier = _verifier(mandate, private_key)
    existing = verifier.preview(valid_request, NOW)
    request: object = valid_request
    preview: object = existing
    outcome: object = ReceiptOutcome.APPROVE
    now: object = NOW
    if malformed == "request_object":
        request = object()
    elif malformed == "request_subclass":
        request = _request_subclass(valid_request)
    elif malformed == "receipt_object":
        preview = object()
    elif malformed == "receipt_subclass":
        preview = _receipt_subclass(existing)
    elif malformed == "outcome_object":
        outcome = "approve"
    elif malformed == "now_object":
        now = object()
    else:
        now = _datetime_subclass(NOW)

    with pytest.raises((TypeError, ValueError)):
        verifier.commit(
            cast(AgentPaymentRequest, request),
            cast(IntegrityReceipt, preview),
            cast(ReceiptOutcome, outcome),
            cast(datetime, now),
        )

    stale = verifier.commit(valid_request, existing, ReceiptOutcome.APPROVE, NOW)

    assert stale.reason_code is ReasonCode.EXECUTION_RECEIPT_MISMATCH
    assert verifier.dump_state()["records"] == ()


def test_malformed_commit_attempt_still_consumes_ephemeral_preview(
    mandate: AgentMandate,
    private_key: Ed25519PrivateKey,
    valid_request: AgentPaymentRequest,
) -> None:
    verifier = _verifier(mandate, private_key)
    preview = verifier.preview(valid_request, NOW)

    with pytest.raises(TypeError, match="final ReceiptOutcome"):
        verifier.commit(
            valid_request,
            preview,
            cast(ReceiptOutcome, "approve"),
            NOW,
        )

    stale = verifier.commit(valid_request, preview, ReceiptOutcome.APPROVE, NOW)

    assert stale.reason_code is ReasonCode.EXECUTION_RECEIPT_MISMATCH
    assert verifier.dump_state()["records"] == ()


def test_commit_cannot_precede_preview_issuance(
    mandate: AgentMandate,
    private_key: Ed25519PrivateKey,
) -> None:
    request = _sign(
        _unsigned_request(mandate, created_at=NOW - timedelta(minutes=1)),
        private_key,
    )
    verifier = _verifier(mandate, private_key)
    preview = verifier.preview(request, NOW)

    receipt = verifier.commit(
        request,
        preview,
        ReceiptOutcome.APPROVE,
        NOW - timedelta(seconds=1),
    )

    assert receipt.reason_code is ReasonCode.EXECUTION_RECEIPT_MISMATCH
    assert verifier.dump_state()["records"] == ()


def test_commit_rejects_after_request_or_authentication_expiry_without_retention(
    mandate: AgentMandate,
    private_key: Ed25519PrivateKey,
    valid_request: AgentPaymentRequest,
) -> None:
    verifier = _verifier(mandate, private_key)
    preview = verifier.preview(valid_request, NOW)

    expired = verifier.commit(
        valid_request,
        preview,
        ReceiptOutcome.APPROVE,
        NOW + timedelta(minutes=5),
    )
    retry = verifier.commit(
        valid_request,
        preview,
        ReceiptOutcome.APPROVE,
        NOW,
    )

    assert expired.reason_code is ReasonCode.AUTHENTICATION_EVIDENCE_EXPIRED
    assert retry.reason_code is ReasonCode.EXECUTION_RECEIPT_MISMATCH
    assert verifier.dump_state()["records"] == ()


def test_failure_receipt_hash_binds_failed_canonical_request_and_signature(
    mandate: AgentMandate,
    private_key: Ed25519PrivateKey,
) -> None:
    first = _sign(
        _unsigned_request(mandate, payee_id="attacker-a"),
        private_key,
    )
    second = _sign(
        _unsigned_request(mandate, payee_id="attacker-b"),
        private_key,
    )
    verifier = _verifier(mandate, private_key)

    first_receipt = verifier.verify(first, NOW)
    second_receipt = verifier.verify(second, NOW)

    assert first_receipt.reason_code is ReasonCode.PAYEE_BINDING_MISMATCH
    assert second_receipt.reason_code is ReasonCode.PAYEE_BINDING_MISMATCH
    assert first.request_id == second.request_id
    assert first_receipt.receipt_hash != second_receipt.receipt_hash
    assert first_receipt.request_hash != second_receipt.request_hash


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
    verifier = TrustVerifier(
        registered_agents={}, mandates={}, authentication_evidence={}
    )

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
            authentication_evidence={},
        )
