"""Deterministic Ed25519 verification for synthetic agentic payments."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from decimal import ROUND_HALF_EVEN, Decimal, DecimalException
from enum import StrEnum
from types import MappingProxyType
from typing import Any, Self, cast
from uuid import UUID

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

from apar.contracts.decisions import ReasonCode

_EXPONENTS = {"EUR": 2, "JPY": 0, "KWD": 3, "USD": 2}
_MAPPING_PROXY_TYPE: type[object] = type(MappingProxyType({}))


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


def _digest(label: str, value: object, *, allow_empty: bool = False) -> str:
    if allow_empty and value == "":
        return ""
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _utc(label: str, value: object) -> datetime:
    if type(value) is not datetime or value.tzinfo is not UTC:
        raise ValueError(f"{label} must be an exact UTC datetime")
    timestamp = value
    return datetime(
        timestamp.year,
        timestamp.month,
        timestamp.day,
        timestamp.hour,
        timestamp.minute,
        timestamp.second,
        timestamp.microsecond,
        tzinfo=UTC,
        fold=timestamp.fold,
    )


def _money(label: str, value: object, currency: str) -> Decimal:
    if type(value) is not Decimal:
        raise TypeError(f"{label} must be an exact Decimal")
    amount = value
    if not amount.is_finite() or amount <= 0:
        raise ValueError(f"{label} must be finite and positive")
    try:
        exponent = _EXPONENTS[currency]
        quantized = amount.quantize(
            Decimal(1).scaleb(-exponent),
            rounding=ROUND_HALF_EVEN,
        )
    except KeyError as error:
        raise ValueError(f"unsupported currency: {currency}") from error
    except DecimalException as error:
        raise ValueError(f"{label} cannot be represented in {currency}") from error
    if quantized <= 0:
        raise ValueError(f"{label} must be finite and positive")
    if amount.as_tuple() != quantized.as_tuple():
        raise ValueError(f"{label} must be canonically quantized for {currency}")
    return amount


def _timestamp_text(value: datetime) -> str:
    return value.isoformat().replace("+00:00", "Z")


def _canonical_bytes(domain: str, values: list[list[str]]) -> bytes:
    return json.dumps(
        [["domain", domain], *values],
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode()


def _scope(label: str, value: object) -> tuple[str, ...]:
    if type(value) is not tuple or not value:
        raise TypeError(f"{label} must be a non-empty exact tuple")
    checked = tuple(_text(label, item) for item in value)
    if len(set(checked)) != len(checked) or checked != tuple(sorted(checked)):
        raise ValueError(f"{label} must be unique and sorted")
    return checked


class AuthenticationRequirement(StrEnum):
    NONE = "none"
    STEP_UP = "step_up"


class AuthenticationOutcome(StrEnum):
    STEP_UP_VERIFIED = "step_up_verified"


class ReceiptOutcome(StrEnum):
    REJECTED = "rejected"
    PREVIEW = "preview"
    VERIFIED = "verified"
    APPROVE = "approve"
    CHALLENGE = "challenge"
    DECLINE = "decline"


class _CopyableRecord:
    """Small Pydantic-like copy helper that retains frozen dataclass validation."""

    def model_copy(self, *, update: Mapping[str, object] | None = None) -> Self:
        changes = dict(update or {})
        names = set(cast(dict[str, object], vars(type(self))["__dataclass_fields__"]))
        unknown = set(changes) - names
        if unknown:
            raise ValueError(f"unknown fields: {sorted(unknown)}")
        return cast(Self, replace(cast(Any, self), **changes))


@dataclass(frozen=True, slots=True)
class AuthenticationEvidence(_CopyableRecord):
    """Trusted synthetic authentication result held outside the signed request."""

    evidence_id: str
    agent_id: str
    user_ref: str
    mandate_id: str
    nonce: str
    payment_intent_hash: str
    request_id: str
    outcome: AuthenticationOutcome
    issued_at: datetime
    expires_at: datetime

    def __post_init__(self) -> None:
        object.__setattr__(self, "evidence_id", _text("evidence_id", self.evidence_id))
        object.__setattr__(self, "agent_id", _text("agent_id", self.agent_id))
        object.__setattr__(self, "user_ref", _text("user_ref", self.user_ref))
        object.__setattr__(self, "mandate_id", _text("mandate_id", self.mandate_id))
        object.__setattr__(self, "nonce", _text("nonce", self.nonce))
        object.__setattr__(
            self,
            "payment_intent_hash",
            _digest("payment_intent_hash", self.payment_intent_hash),
        )
        object.__setattr__(self, "request_id", _text("request_id", self.request_id))
        if type(self.outcome) is not AuthenticationOutcome:
            raise TypeError("outcome must be an exact AuthenticationOutcome")
        issued_at = _utc("issued_at", self.issued_at)
        expires_at = _utc("expires_at", self.expires_at)
        if expires_at <= issued_at:
            raise ValueError("authentication evidence expires_at must be after issued_at")
        object.__setattr__(self, "issued_at", issued_at)
        object.__setattr__(self, "expires_at", expires_at)


@dataclass(frozen=True, slots=True)
class AgentMandate(_CopyableRecord):
    """Canonical user consent and delegated-purchase bounds."""

    mandate_id: str
    version: int
    agent_id: str
    user_ref: str
    user_entity_id: str
    beneficiary_entity_id: str
    consent_ref: str
    merchant_id: str
    payee_id: str
    cart_hash: str
    payment_intent_hash: str
    permitted_categories: tuple[str, ...]
    permitted_products: tuple[str, ...]
    credential_id: str
    credential_scope: str
    required_authentication: AuthenticationRequirement
    max_amount: Decimal
    currency: str
    issued_at: datetime
    expires_at: datetime

    def __post_init__(self) -> None:
        object.__setattr__(self, "mandate_id", _text("mandate_id", self.mandate_id))
        if type(self.version) is not int or self.version <= 0:
            raise TypeError("version must be a positive exact integer")
        object.__setattr__(self, "agent_id", _text("agent_id", self.agent_id))
        object.__setattr__(self, "user_ref", _text("user_ref", self.user_ref))
        object.__setattr__(
            self, "user_entity_id", _uuid_text("user_entity_id", self.user_entity_id)
        )
        object.__setattr__(
            self,
            "beneficiary_entity_id",
            _uuid_text("beneficiary_entity_id", self.beneficiary_entity_id),
        )
        object.__setattr__(self, "consent_ref", _text("consent_ref", self.consent_ref))
        object.__setattr__(self, "merchant_id", _text("merchant_id", self.merchant_id))
        object.__setattr__(self, "payee_id", _text("payee_id", self.payee_id))
        object.__setattr__(self, "cart_hash", _digest("cart_hash", self.cart_hash))
        object.__setattr__(
            self,
            "payment_intent_hash",
            _digest("payment_intent_hash", self.payment_intent_hash),
        )
        object.__setattr__(
            self,
            "permitted_categories",
            _scope("permitted_categories", self.permitted_categories),
        )
        object.__setattr__(
            self,
            "permitted_products",
            _scope("permitted_products", self.permitted_products),
        )
        object.__setattr__(
            self,
            "credential_id",
            _text("credential_id", self.credential_id),
        )
        object.__setattr__(
            self,
            "credential_scope",
            _text("credential_scope", self.credential_scope),
        )
        if type(self.required_authentication) is not AuthenticationRequirement:
            raise TypeError(
                "required_authentication must be an exact AuthenticationRequirement"
            )
        currency = _text("currency", self.currency)
        object.__setattr__(self, "currency", currency)
        object.__setattr__(
            self,
            "max_amount",
            _money("max_amount", self.max_amount, currency),
        )
        issued_at = _utc("issued_at", self.issued_at)
        expires_at = _utc("expires_at", self.expires_at)
        if expires_at <= issued_at:
            raise ValueError("mandate expires_at must be after issued_at")
        object.__setattr__(self, "issued_at", issued_at)
        object.__setattr__(self, "expires_at", expires_at)

    def canonical_bytes(self) -> bytes:
        return _canonical_bytes(
            "apar.agent-mandate.v1",
            [
                ["mandate_id", self.mandate_id],
                ["version", str(self.version)],
                ["agent_id", self.agent_id],
                ["user_ref", self.user_ref],
                ["user_entity_id", self.user_entity_id],
                ["beneficiary_entity_id", self.beneficiary_entity_id],
                ["consent_ref", self.consent_ref],
                ["merchant_id", self.merchant_id],
                ["payee_id", self.payee_id],
                ["cart_hash", self.cart_hash],
                ["payment_intent_hash", self.payment_intent_hash],
                ["permitted_categories", json.dumps(self.permitted_categories)],
                ["permitted_products", json.dumps(self.permitted_products)],
                ["credential_id", self.credential_id],
                ["credential_scope", self.credential_scope],
                ["required_authentication", self.required_authentication.value],
                ["max_amount", str(self.max_amount)],
                ["currency", self.currency],
                ["issued_at", _timestamp_text(self.issued_at)],
                ["expires_at", _timestamp_text(self.expires_at)],
            ],
        )


@dataclass(frozen=True, slots=True)
class AgentPaymentRequest(_CopyableRecord):
    """Canonical signed request whose bindings are checked before risk scoring."""

    request_id: str
    payment_id: str
    agent_id: str
    key_id: str
    mandate: AgentMandate
    amount: Decimal
    currency: str
    merchant_id: str
    payee_id: str
    cart_hash: str
    payment_intent_hash: str
    category: str
    product_id: str
    credential_id: str
    credential_scope: str
    consent_ref: str
    authentication_evidence_ref: str | None
    nonce: str
    created_at: datetime
    expires_at: datetime
    prior_receipt_hash: str
    campaign_id: str
    trace_id: str
    actor_id: str
    counterparty_id: str
    signature: bytes

    def __post_init__(self) -> None:
        object.__setattr__(self, "request_id", _text("request_id", self.request_id))
        object.__setattr__(self, "payment_id", _text("payment_id", self.payment_id))
        object.__setattr__(self, "agent_id", _text("agent_id", self.agent_id))
        object.__setattr__(self, "key_id", _text("key_id", self.key_id))
        if type(self.mandate) is not AgentMandate:
            raise TypeError("mandate must be an exact AgentMandate")
        currency = _text("currency", self.currency)
        object.__setattr__(self, "currency", currency)
        object.__setattr__(self, "amount", _money("amount", self.amount, currency))
        object.__setattr__(self, "merchant_id", _text("merchant_id", self.merchant_id))
        object.__setattr__(self, "payee_id", _text("payee_id", self.payee_id))
        object.__setattr__(self, "cart_hash", _digest("cart_hash", self.cart_hash))
        object.__setattr__(
            self,
            "payment_intent_hash",
            _digest("payment_intent_hash", self.payment_intent_hash),
        )
        object.__setattr__(self, "category", _text("category", self.category))
        object.__setattr__(self, "product_id", _text("product_id", self.product_id))
        object.__setattr__(
            self,
            "credential_id",
            _text("credential_id", self.credential_id),
        )
        object.__setattr__(
            self,
            "credential_scope",
            _text("credential_scope", self.credential_scope),
        )
        object.__setattr__(self, "consent_ref", _text("consent_ref", self.consent_ref))
        if self.mandate.required_authentication is AuthenticationRequirement.NONE:
            if self.authentication_evidence_ref is not None:
                raise ValueError(
                    "authentication_evidence_ref must be None when step-up is not required"
                )
        else:
            object.__setattr__(
                self,
                "authentication_evidence_ref",
                _text("authentication_evidence_ref", self.authentication_evidence_ref),
            )
        object.__setattr__(self, "nonce", _text("nonce", self.nonce))
        created_at = _utc("created_at", self.created_at)
        expires_at = _utc("expires_at", self.expires_at)
        if expires_at <= created_at:
            raise ValueError("request expires_at must be after created_at")
        object.__setattr__(self, "created_at", created_at)
        object.__setattr__(self, "expires_at", expires_at)
        object.__setattr__(
            self,
            "prior_receipt_hash",
            _digest("prior_receipt_hash", self.prior_receipt_hash, allow_empty=True),
        )
        object.__setattr__(self, "campaign_id", _uuid_text("campaign_id", self.campaign_id))
        object.__setattr__(self, "trace_id", _uuid_text("trace_id", self.trace_id))
        object.__setattr__(self, "actor_id", _uuid_text("actor_id", self.actor_id))
        object.__setattr__(
            self,
            "counterparty_id",
            _uuid_text("counterparty_id", self.counterparty_id),
        )
        if type(self.signature) is not bytes:
            raise TypeError("signature must be exact bytes")

    def signing_bytes(self) -> bytes:
        mandate_hash = hashlib.sha256(self.mandate.canonical_bytes()).hexdigest()
        return _canonical_bytes(
            "apar.agent-payment-request.v1",
            [
                ["request_id", self.request_id],
                ["payment_id", self.payment_id],
                ["agent_id", self.agent_id],
                ["key_id", self.key_id],
                ["mandate_hash", mandate_hash],
                ["amount", str(self.amount)],
                ["currency", self.currency],
                ["merchant_id", self.merchant_id],
                ["payee_id", self.payee_id],
                ["cart_hash", self.cart_hash],
                ["payment_intent_hash", self.payment_intent_hash],
                ["category", self.category],
                ["product_id", self.product_id],
                ["credential_id", self.credential_id],
                ["credential_scope", self.credential_scope],
                ["consent_ref", self.consent_ref],
                [
                    "authentication_evidence_ref",
                    self.authentication_evidence_ref or "",
                ],
                ["nonce", self.nonce],
                ["created_at", _timestamp_text(self.created_at)],
                ["expires_at", _timestamp_text(self.expires_at)],
                ["prior_receipt_hash", self.prior_receipt_hash],
                ["campaign_id", self.campaign_id],
                ["trace_id", self.trace_id],
                ["actor_id", self.actor_id],
                ["counterparty_id", self.counterparty_id],
            ],
        )


@dataclass(frozen=True, slots=True)
class IntegrityReceipt(_CopyableRecord):
    """Synthetic integrity result; its digest is not a payment credential."""

    request_id: str
    allowed: bool
    reason_code: ReasonCode | None
    receipt_hash: str
    previous_receipt_hash: str
    request_hash: str
    signature_hash: str
    outcome: ReceiptOutcome

    def __post_init__(self) -> None:
        object.__setattr__(self, "request_id", _text("request_id", self.request_id))
        if type(self.allowed) is not bool:
            raise TypeError("allowed must be an exact boolean")
        if self.reason_code is not None and type(self.reason_code) is not ReasonCode:
            raise TypeError("reason_code must be an exact ReasonCode or None")
        if self.allowed == (self.reason_code is not None):
            raise ValueError("allowed and reason_code are inconsistent")
        object.__setattr__(self, "receipt_hash", _digest("receipt_hash", self.receipt_hash))
        object.__setattr__(
            self,
            "previous_receipt_hash",
            _digest("previous_receipt_hash", self.previous_receipt_hash, allow_empty=True),
        )
        object.__setattr__(self, "request_hash", _digest("request_hash", self.request_hash))
        object.__setattr__(
            self,
            "signature_hash",
            _digest("signature_hash", self.signature_hash),
        )
        if type(self.outcome) is not ReceiptOutcome:
            raise TypeError("outcome must be an exact ReceiptOutcome")
        if self.allowed and self.outcome is ReceiptOutcome.REJECTED:
            raise ValueError("allowed receipt cannot have rejected outcome")
        if not self.allowed and self.outcome is not ReceiptOutcome.REJECTED:
            raise ValueError("rejected receipt must have rejected outcome")


class TrustVerifierStateError(RuntimeError):
    """Stable failure for malformed run-local nonce and receipt state."""

    def __init__(self) -> None:
        self.code = "AGENTIC_STATE_CORRUPT"
        super().__init__(self.code)


type _StateRecord = tuple[str, str, str, str, str, str, str, str]


def _receipt_digest(
    agent_id: str,
    nonce: str,
    authentication_evidence_ref: str,
    request_hash: str,
    signature_hash: str,
    previous_receipt_hash: str,
    outcome: ReceiptOutcome,
) -> str:
    return hashlib.sha256(
        _canonical_bytes(
            "apar.synthetic-integrity-receipt.v1",
            [
                ["agent_id", agent_id],
                ["nonce", nonce],
                ["authentication_evidence_ref", authentication_evidence_ref],
                ["request_hash", request_hash],
                ["signature_hash", signature_hash],
                ["previous_receipt_hash", previous_receipt_hash],
                ["outcome", outcome.value],
            ],
        )
    ).hexdigest()


def _owned_receipt(receipt: IntegrityReceipt) -> IntegrityReceipt:
    if type(receipt) is not IntegrityReceipt:
        raise TypeError("receipt must be an exact IntegrityReceipt")
    return IntegrityReceipt(
        request_id=receipt.request_id,
        allowed=receipt.allowed,
        reason_code=receipt.reason_code,
        receipt_hash=receipt.receipt_hash,
        previous_receipt_hash=receipt.previous_receipt_hash,
        request_hash=receipt.request_hash,
        signature_hash=receipt.signature_hash,
        outcome=receipt.outcome,
    )


@dataclass(frozen=True, slots=True, eq=False)
class TrustCommitPlan:
    """Verifier-issued, single-use final receipt plan for atomic rail execution."""

    receipt: IntegrityReceipt

    def __post_init__(self) -> None:
        object.__setattr__(self, "receipt", _owned_receipt(self.receipt))


@dataclass(frozen=True, slots=True)
class _PendingCommit:
    capability: TrustCommitPlan
    receipt: IntegrityReceipt
    record: _StateRecord


class TrustVerifier:
    """Verify ordered integrity claims and own deterministic replay state."""

    def __init__(
        self,
        *,
        registered_agents: Mapping[tuple[str, str], bytes],
        mandates: Mapping[str, AgentMandate],
        authentication_evidence: Mapping[str, AuthenticationEvidence],
    ) -> None:
        keys: dict[tuple[str, str], Ed25519PublicKey] = {}
        for identity, raw_key in registered_agents.items():
            if type(identity) is not tuple or len(identity) != 2:
                raise TypeError("registered agent identity must be an exact pair")
            agent_id = _text("registered agent_id", identity[0])
            key_id = _text("registered key_id", identity[1])
            if type(raw_key) is not bytes or len(raw_key) != 32:
                raise ValueError("registered key must be a 32-byte Ed25519 public key")
            try:
                keys[(agent_id, key_id)] = Ed25519PublicKey.from_public_bytes(raw_key)
            except ValueError as error:
                raise ValueError("registered key must be a valid Ed25519 public key") from error
        approved: dict[str, AgentMandate] = {}
        for mandate_id, mandate in mandates.items():
            key = _text("mandate mapping key", mandate_id)
            if type(mandate) is not AgentMandate or mandate.mandate_id != key:
                raise TypeError("mandates must contain exact AgentMandate values keyed by ID")
            approved[key] = mandate
        evidence_registry: dict[str, AuthenticationEvidence] = {}
        for evidence_id, evidence in authentication_evidence.items():
            key = _text("authentication evidence mapping key", evidence_id)
            if type(evidence) is not AuthenticationEvidence or evidence.evidence_id != key:
                raise TypeError(
                    "authentication_evidence must contain exact AuthenticationEvidence "
                    "values keyed by ID"
                )
            evidence_registry[key] = evidence
        self._keys = MappingProxyType(keys)
        self._agent_ids = frozenset(agent_id for agent_id, _ in keys)
        self._mandates = MappingProxyType(approved)
        self._authentication_evidence = MappingProxyType(evidence_registry)
        self._records: tuple[_StateRecord, ...] = ()
        self._pending_preview: tuple[IntegrityReceipt, datetime] | None = None
        self._pending_commit: _PendingCommit | None = None

    def verify(self, request: AgentPaymentRequest, now: datetime) -> IntegrityReceipt:
        preview = self.preview(request, now)
        if not preview.allowed:
            return preview
        return self.commit(request, preview, ReceiptOutcome.VERIFIED, now)

    def preview(self, request: AgentPaymentRequest, now: datetime) -> IntegrityReceipt:
        """Evaluate without consuming a nonce so downstream scoring stays atomic."""
        self._revoke_ephemeral()
        if type(request) is not AgentPaymentRequest:
            raise TypeError("request must be an exact AgentPaymentRequest")
        checked_now = _utc("now", now)

        public_key = self._keys.get((request.agent_id, request.key_id))
        if public_key is None:
            return self._failure(request, ReasonCode.AGENT_IDENTITY_MISMATCH)

        try:
            public_key.verify(request.signature, request.signing_bytes())
        except (InvalidSignature, ValueError):
            return self._failure(request, ReasonCode.SIGNATURE_INVALID)

        approved = self._mandates.get(request.mandate.mandate_id)
        if (
            approved is None
            or request.mandate.canonical_bytes() != approved.canonical_bytes()
            or request.mandate.agent_id != request.agent_id
        ):
            return self._failure(request, ReasonCode.MANDATE_SCOPE_VIOLATION)

        if (
            request.actor_id != approved.user_entity_id
            or request.counterparty_id != approved.beneficiary_entity_id
        ):
            return self._failure(request, ReasonCode.AUTHORITY_IDENTITY_MISMATCH)

        if request.amount > approved.max_amount:
            return self._failure(request, ReasonCode.AMOUNT_LIMIT_EXCEEDED)
        if request.currency != approved.currency:
            return self._failure(request, ReasonCode.CURRENCY_MISMATCH)
        if request.merchant_id != approved.merchant_id:
            return self._failure(request, ReasonCode.MERCHANT_BINDING_MISMATCH)
        if request.payee_id != approved.payee_id:
            return self._failure(request, ReasonCode.PAYEE_BINDING_MISMATCH)
        if request.category not in approved.permitted_categories:
            return self._failure(request, ReasonCode.CATEGORY_SCOPE_VIOLATION)
        if request.product_id not in approved.permitted_products:
            return self._failure(request, ReasonCode.PRODUCT_SCOPE_VIOLATION)
        if request.cart_hash != approved.cart_hash:
            return self._failure(request, ReasonCode.CART_HASH_MISMATCH)
        if request.payment_intent_hash != approved.payment_intent_hash:
            return self._failure(request, ReasonCode.PAYMENT_INTENT_HASH_MISMATCH)
        if request.credential_id != approved.credential_id:
            return self._failure(request, ReasonCode.CREDENTIAL_BINDING_MISMATCH)
        if request.credential_scope != approved.credential_scope:
            return self._failure(request, ReasonCode.TOKEN_SCOPE_VIOLATION)
        if request.consent_ref != approved.consent_ref:
            return self._failure(request, ReasonCode.CONSENT_BINDING_MISMATCH)
        if (
            request.created_at < approved.issued_at
            or request.expires_at > approved.expires_at
        ):
            return self._failure(request, ReasonCode.MANDATE_TIME_SCOPE_VIOLATION)
        if (
            checked_now < request.created_at
            or checked_now < approved.issued_at
            or checked_now >= request.expires_at
            or checked_now >= approved.expires_at
        ):
            return self._failure(request, ReasonCode.MANDATE_EXPIRED)

        if any(
            agent_id == request.agent_id and nonce == request.nonce
            for agent_id, nonce, _, _, _, _, _, _ in self._records
        ):
            return self._failure(request, ReasonCode.NONCE_REPLAY)

        previous = self._last_receipt(request.agent_id)
        if request.prior_receipt_hash != previous:
            return self._failure(request, ReasonCode.RECEIPT_CHAIN_BROKEN)

        # Trusted step-up evidence follows the invariant base ordering above. It is
        # registry-owned, so the agent signs only its synthetic reference.
        if approved.required_authentication is AuthenticationRequirement.STEP_UP:
            evidence_ref = request.authentication_evidence_ref
            assert evidence_ref is not None
            if any(
                committed_ref == evidence_ref
                for _, _, committed_ref, _, _, _, _, _ in self._records
                if committed_ref
            ):
                return self._failure(
                    request, ReasonCode.AUTHENTICATION_EVIDENCE_REPLAY
                )
            evidence = self._authentication_evidence.get(
                evidence_ref
            )
            if evidence is None:
                return self._failure(
                    request, ReasonCode.AUTHENTICATION_EVIDENCE_MISSING
                )
            if (
                evidence.agent_id != request.agent_id
                or evidence.user_ref != approved.user_ref
                or evidence.mandate_id != approved.mandate_id
                or evidence.nonce != request.nonce
                or evidence.payment_intent_hash != request.payment_intent_hash
                or evidence.request_id != request.request_id
                or evidence.outcome is not AuthenticationOutcome.STEP_UP_VERIFIED
            ):
                return self._failure(
                    request, ReasonCode.AUTHENTICATION_EVIDENCE_MISMATCH
                )
            if checked_now < evidence.issued_at or checked_now >= evidence.expires_at:
                return self._failure(
                    request, ReasonCode.AUTHENTICATION_EVIDENCE_EXPIRED
                )

        preview = self._allowed_receipt(request, previous, ReceiptOutcome.PREVIEW)
        self._pending_preview = (preview, checked_now)
        self._pending_commit = None
        return preview

    def commit(
        self,
        request: AgentPaymentRequest,
        preview: IntegrityReceipt,
        outcome: ReceiptOutcome,
        now: datetime,
    ) -> IntegrityReceipt:
        """Atomically consume replay state only after a valid final outcome exists."""
        issued_preview = self._take_pending_preview()
        prepared = self._prepare_commit(
            request, preview, outcome, now, issued_preview
        )
        if type(prepared) is IntegrityReceipt:
            return prepared
        pending = self._pending_commit
        assert pending is not None and pending.capability is prepared
        return self._apply_pending(pending)

    def prepare_commit(
        self,
        request: AgentPaymentRequest,
        preview: IntegrityReceipt,
        outcome: ReceiptOutcome,
        now: datetime,
    ) -> TrustCommitPlan | IntegrityReceipt:
        """Validate a final result without consuming persistent replay state."""
        issued_preview = self._take_pending_preview()
        return self._prepare_commit(request, preview, outcome, now, issued_preview)

    def _prepare_commit(
        self,
        request: AgentPaymentRequest,
        preview: IntegrityReceipt,
        outcome: ReceiptOutcome,
        now: datetime,
        issued_preview: tuple[IntegrityReceipt, datetime] | None,
    ) -> TrustCommitPlan | IntegrityReceipt:
        if type(request) is not AgentPaymentRequest:
            raise TypeError("request must be an exact AgentPaymentRequest")
        if type(preview) is not IntegrityReceipt:
            raise TypeError("preview must be an exact IntegrityReceipt")
        if type(outcome) is not ReceiptOutcome or outcome not in {
            ReceiptOutcome.VERIFIED,
            ReceiptOutcome.APPROVE,
            ReceiptOutcome.CHALLENGE,
            ReceiptOutcome.DECLINE,
        }:
            raise TypeError("outcome must be an exact final ReceiptOutcome")
        checked_now = _utc("now", now)
        previous = self._last_receipt(request.agent_id)
        expected_preview = self._allowed_receipt(
            request,
            previous,
            ReceiptOutcome.PREVIEW,
        )
        if preview != expected_preview or issued_preview is None or issued_preview[0] != preview:
            return self._failure(request, ReasonCode.EXECUTION_RECEIPT_MISMATCH)
        if checked_now < issued_preview[1]:
            return self._failure(request, ReasonCode.EXECUTION_RECEIPT_MISMATCH)
        if request.mandate.required_authentication is AuthenticationRequirement.STEP_UP:
            evidence_ref = request.authentication_evidence_ref
            assert evidence_ref is not None
            evidence = self._authentication_evidence.get(evidence_ref)
            if evidence is not None and (
                checked_now < evidence.issued_at or checked_now >= evidence.expires_at
            ):
                return self._failure(
                    request, ReasonCode.AUTHENTICATION_EVIDENCE_EXPIRED
                )
        if (
            checked_now < request.created_at
            or checked_now < request.mandate.issued_at
            or checked_now >= request.expires_at
            or checked_now >= request.mandate.expires_at
        ):
            return self._failure(request, ReasonCode.MANDATE_EXPIRED)
        if any(
            agent_id == request.agent_id and nonce == request.nonce
            for agent_id, nonce, _, _, _, _, _, _ in self._records
        ):
            return self._failure(request, ReasonCode.NONCE_REPLAY)
        if request.prior_receipt_hash != previous:
            return self._failure(request, ReasonCode.RECEIPT_CHAIN_BROKEN)

        final = self._allowed_receipt(request, previous, outcome)
        record: _StateRecord = (
            request.agent_id,
            request.nonce,
            request.authentication_evidence_ref or "",
            final.previous_receipt_hash,
            final.request_hash,
            final.signature_hash,
            final.outcome.value,
            final.receipt_hash,
        )
        plan = TrustCommitPlan(final)
        self._pending_commit = _PendingCommit(plan, final, record)
        return plan

    def projected_state(self, plan: TrustCommitPlan) -> Mapping[str, object]:
        """Return the closed state that one currently issued plan will commit."""
        pending = self._require_pending_plan(plan)
        return MappingProxyType(
            {"version": 1, "records": (*self._records, pending.record)}
        )

    def apply_commit(self, plan: TrustCommitPlan) -> IntegrityReceipt:
        """Apply one verifier-issued plan through a total, single-thread operation."""
        pending = self._pending_commit
        self._pending_preview = None
        self._pending_commit = None
        if type(plan) is not TrustCommitPlan:
            raise TypeError("plan must be an exact TrustCommitPlan")
        if pending is None or pending.capability is not plan:
            raise ValueError("commit plan was not issued or was already consumed")
        return self._apply_pending(pending)

    def discard_commit(self, plan: TrustCommitPlan) -> None:
        """Discard a prepared outcome after external execution cannot complete."""
        pending = self._pending_commit
        self._pending_preview = None
        self._pending_commit = None
        if type(plan) is not TrustCommitPlan:
            raise TypeError("plan must be an exact TrustCommitPlan")
        if pending is not None and pending.capability is plan:
            return
        raise ValueError("commit plan was not issued or was already consumed")

    def _require_pending_plan(self, plan: TrustCommitPlan) -> _PendingCommit:
        pending = self._pending_commit
        if (
            type(plan) is not TrustCommitPlan
            or pending is None
            or pending.capability is not plan
        ):
            raise ValueError("commit plan was not issued or was already consumed")
        return pending

    def _take_pending_preview(self) -> tuple[IntegrityReceipt, datetime] | None:
        preview = self._pending_preview
        self._pending_preview = None
        self._pending_commit = None
        return preview

    def _revoke_ephemeral(self) -> None:
        self._pending_preview = None
        self._pending_commit = None

    def _apply_pending(self, pending: _PendingCommit) -> IntegrityReceipt:
        self._records = (*self._records, pending.record)
        self._pending_commit = None
        return pending.receipt

    def discard_preview(self, preview: IntegrityReceipt) -> None:
        """Revoke one issued preview after downstream scoring cannot finalize it."""
        if type(preview) is not IntegrityReceipt:
            raise TypeError("preview must be an exact IntegrityReceipt")
        if self._pending_preview is not None and self._pending_preview[0] == preview:
            self._pending_preview = None

    def dump_state(self) -> Mapping[str, object]:
        return MappingProxyType({"version": 1, "records": self._records})

    def load_state(self, value: object) -> None:
        try:
            if type(value) not in (dict, _MAPPING_PROXY_TYPE):
                raise TypeError("verifier state must be an owned mapping")
            state = cast(Mapping[str, object], value)
            if set(state) != {"version", "records"} or any(
                type(key) is not str for key in state
            ):
                raise ValueError("verifier state fields are invalid")
            if type(state["version"]) is not int or state["version"] != 1:
                raise ValueError("verifier state version is invalid")
            records_value = state["records"]
            if type(records_value) is not tuple:
                raise TypeError("verifier state records must be an exact tuple")
            records: list[_StateRecord] = []
            seen_nonces: set[tuple[str, str]] = set()
            seen_evidence_refs: set[str] = set()
            last_receipts: dict[str, str] = {}
            for raw_record in records_value:
                if type(raw_record) is not tuple or len(raw_record) != 8:
                    raise ValueError("verifier state record is malformed")
                agent_id = _text("state agent_id", raw_record[0])
                if agent_id not in self._agent_ids:
                    raise ValueError("verifier state agent is not registered")
                nonce = _text("state nonce", raw_record[1])
                raw_evidence_ref = raw_record[2]
                if type(raw_evidence_ref) is not str:
                    raise TypeError(
                        "state authentication evidence ref must be an exact string"
                    )
                evidence_ref = raw_evidence_ref
                previous = _digest("state previous receipt", raw_record[3], allow_empty=True)
                request_hash = _digest("state request hash", raw_record[4])
                signature_hash = _digest("state signature hash", raw_record[5])
                outcome = ReceiptOutcome(_text("state outcome", raw_record[6]))
                if outcome not in {
                    ReceiptOutcome.VERIFIED,
                    ReceiptOutcome.APPROVE,
                    ReceiptOutcome.CHALLENGE,
                    ReceiptOutcome.DECLINE,
                }:
                    raise ValueError("verifier state outcome is not final")
                receipt = _digest("state receipt", raw_record[7])
                nonce_key = (agent_id, nonce)
                if nonce_key in seen_nonces:
                    raise ValueError("verifier state nonce is duplicated")
                if evidence_ref and evidence_ref in seen_evidence_refs:
                    raise ValueError(
                        "verifier state authentication evidence is duplicated"
                    )
                if previous != last_receipts.get(agent_id, ""):
                    raise ValueError("verifier state receipt chain is broken")
                if receipt != _receipt_digest(
                    agent_id,
                    nonce,
                    evidence_ref,
                    request_hash,
                    signature_hash,
                    previous,
                    outcome,
                ):
                    raise ValueError("verifier state receipt is not reproducible")
                seen_nonces.add(nonce_key)
                if evidence_ref:
                    seen_evidence_refs.add(evidence_ref)
                last_receipts[agent_id] = receipt
                records.append(
                    (
                        agent_id,
                        nonce,
                        evidence_ref,
                        previous,
                        request_hash,
                        signature_hash,
                        outcome.value,
                        receipt,
                    )
                )
            self._records = tuple(records)
            self._pending_preview = None
            self._pending_commit = None
        except (KeyError, RuntimeError, TypeError, ValueError) as error:
            raise TrustVerifierStateError from error

    def _last_receipt(self, agent_id: str) -> str:
        for record_agent, _, _, _, _, _, _, receipt in reversed(self._records):
            if record_agent == agent_id:
                return receipt
        return ""

    @staticmethod
    def _failure(
        request: AgentPaymentRequest,
        reason: ReasonCode,
    ) -> IntegrityReceipt:
        request_hash = hashlib.sha256(request.signing_bytes()).hexdigest()
        signature_hash = hashlib.sha256(request.signature).hexdigest()
        failure_hash = hashlib.sha256(
            _canonical_bytes(
                "apar.synthetic-integrity-failure.v1",
                [
                    ["request_id", request.request_id],
                    ["request_hash", request_hash],
                    ["signature_hash", signature_hash],
                    ["reason_code", reason.value],
                    ["prior_receipt_hash", request.prior_receipt_hash],
                    ["outcome", ReceiptOutcome.REJECTED.value],
                ],
            )
        ).hexdigest()
        return IntegrityReceipt(
            request_id=request.request_id,
            allowed=False,
            reason_code=reason,
            receipt_hash=failure_hash,
            previous_receipt_hash=request.prior_receipt_hash,
            request_hash=request_hash,
            signature_hash=signature_hash,
            outcome=ReceiptOutcome.REJECTED,
        )

    @staticmethod
    def _allowed_receipt(
        request: AgentPaymentRequest,
        previous: str,
        outcome: ReceiptOutcome,
    ) -> IntegrityReceipt:
        request_hash = hashlib.sha256(request.signing_bytes()).hexdigest()
        signature_hash = hashlib.sha256(request.signature).hexdigest()
        return IntegrityReceipt(
            request_id=request.request_id,
            allowed=True,
            reason_code=None,
            receipt_hash=_receipt_digest(
                request.agent_id,
                request.nonce,
                request.authentication_evidence_ref or "",
                request_hash,
                signature_hash,
                previous,
                outcome,
            ),
            previous_receipt_hash=previous,
            request_hash=request_hash,
            signature_hash=signature_hash,
            outcome=outcome,
        )


__all__ = [
    "AgentMandate",
    "AgentPaymentRequest",
    "AuthenticationEvidence",
    "AuthenticationOutcome",
    "AuthenticationRequirement",
    "IntegrityReceipt",
    "ReceiptOutcome",
    "TrustVerifier",
    "TrustCommitPlan",
    "TrustVerifierStateError",
]
