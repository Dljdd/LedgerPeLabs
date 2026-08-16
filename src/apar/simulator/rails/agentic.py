"""Agentic-commerce rail with integrity verification before risk and value."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal, DecimalException
from types import MappingProxyType
from typing import cast

from apar.contracts.decisions import Action, ReasonCode
from apar.contracts.events import EventKind, PaymentEvent, Rail
from apar.simulator.clock import Command
from apar.simulator.ledger import LedgerEntry
from apar.simulator.rails.base import LifecycleError, RailContext
from apar.trust.verifier import (
    AgentMandate,
    AgentPaymentRequest,
    AuthenticationRequirement,
    IntegrityReceipt,
    ReceiptOutcome,
    TrustCommitPlan,
    TrustVerifier,
    TrustVerifierStateError,
)

AGENTIC_TRUST_STATE_ID = "agentic:trust"
_MAPPING_PROXY_TYPE: type[object] = type(MappingProxyType({}))


def _text(label: str, value: object) -> str:
    if type(value) is not str:
        raise TypeError(f"{label} must be an exact string")
    if not value:
        raise ValueError(f"{label} must not be empty")
    return value


def _request_payload(request: AgentPaymentRequest) -> dict[str, object]:
    if type(request) is not AgentPaymentRequest:
        raise TypeError("request must be an exact AgentPaymentRequest")
    mandate = request.mandate
    return {
        "request_id": request.request_id,
        "payment_id": request.payment_id,
        "agent_id": request.agent_id,
        "key_id": request.key_id,
        "mandate_id": mandate.mandate_id,
        "mandate_version": mandate.version,
        "mandate_agent_id": mandate.agent_id,
        "user_ref": mandate.user_ref,
        "user_entity_id": mandate.user_entity_id,
        "beneficiary_entity_id": mandate.beneficiary_entity_id,
        "mandate_consent_ref": mandate.consent_ref,
        "mandate_merchant_id": mandate.merchant_id,
        "mandate_payee_id": mandate.payee_id,
        "mandate_cart_hash": mandate.cart_hash,
        "mandate_payment_intent_hash": mandate.payment_intent_hash,
        "permitted_categories": mandate.permitted_categories,
        "permitted_products": mandate.permitted_products,
        "mandate_credential_id": mandate.credential_id,
        "mandate_credential_scope": mandate.credential_scope,
        "required_authentication": mandate.required_authentication.value,
        "max_amount": mandate.max_amount,
        "mandate_currency": mandate.currency,
        "mandate_issued_at": mandate.issued_at,
        "mandate_expires_at": mandate.expires_at,
        "amount": request.amount,
        "currency": request.currency,
        "merchant_id": request.merchant_id,
        "payee_id": request.payee_id,
        "cart_hash": request.cart_hash,
        "payment_intent_hash": request.payment_intent_hash,
        "category": request.category,
        "product_id": request.product_id,
        "credential_id": request.credential_id,
        "credential_scope": request.credential_scope,
        "consent_ref": request.consent_ref,
        "authentication_evidence_ref": request.authentication_evidence_ref,
        "nonce": request.nonce,
        "created_at": request.created_at,
        "expires_at": request.expires_at,
        "prior_receipt_hash": request.prior_receipt_hash,
        "campaign_id": request.campaign_id,
        "trace_id": request.trace_id,
        "actor_id": request.actor_id,
        "counterparty_id": request.counterparty_id,
        "signature": request.signature,
    }


def _request_from_payload(payload: Mapping[str, object]) -> AgentPaymentRequest:
    mandate = AgentMandate(
        mandate_id=cast(str, payload["mandate_id"]),
        version=cast(int, payload["mandate_version"]),
        agent_id=cast(str, payload["mandate_agent_id"]),
        user_ref=cast(str, payload["user_ref"]),
        user_entity_id=cast(str, payload["user_entity_id"]),
        beneficiary_entity_id=cast(str, payload["beneficiary_entity_id"]),
        consent_ref=cast(str, payload["mandate_consent_ref"]),
        merchant_id=cast(str, payload["mandate_merchant_id"]),
        payee_id=cast(str, payload["mandate_payee_id"]),
        cart_hash=cast(str, payload["mandate_cart_hash"]),
        payment_intent_hash=cast(str, payload["mandate_payment_intent_hash"]),
        permitted_categories=cast(tuple[str, ...], payload["permitted_categories"]),
        permitted_products=cast(tuple[str, ...], payload["permitted_products"]),
        credential_id=cast(str, payload["mandate_credential_id"]),
        credential_scope=cast(str, payload["mandate_credential_scope"]),
        required_authentication=AuthenticationRequirement(
            cast(str, payload["required_authentication"])
        ),
        max_amount=cast(Decimal, payload["max_amount"]),
        currency=cast(str, payload["mandate_currency"]),
        issued_at=cast(datetime, payload["mandate_issued_at"]),
        expires_at=cast(datetime, payload["mandate_expires_at"]),
    )
    return AgentPaymentRequest(
        request_id=cast(str, payload["request_id"]),
        payment_id=cast(str, payload["payment_id"]),
        agent_id=cast(str, payload["agent_id"]),
        key_id=cast(str, payload["key_id"]),
        mandate=mandate,
        amount=cast(Decimal, payload["amount"]),
        currency=cast(str, payload["currency"]),
        merchant_id=cast(str, payload["merchant_id"]),
        payee_id=cast(str, payload["payee_id"]),
        cart_hash=cast(str, payload["cart_hash"]),
        payment_intent_hash=cast(str, payload["payment_intent_hash"]),
        category=cast(str, payload["category"]),
        product_id=cast(str, payload["product_id"]),
        credential_id=cast(str, payload["credential_id"]),
        credential_scope=cast(str, payload["credential_scope"]),
        consent_ref=cast(str, payload["consent_ref"]),
        authentication_evidence_ref=cast(
            str | None, payload["authentication_evidence_ref"]
        ),
        nonce=cast(str, payload["nonce"]),
        created_at=cast(datetime, payload["created_at"]),
        expires_at=cast(datetime, payload["expires_at"]),
        prior_receipt_hash=cast(str, payload["prior_receipt_hash"]),
        campaign_id=cast(str, payload["campaign_id"]),
        trace_id=cast(str, payload["trace_id"]),
        actor_id=cast(str, payload["actor_id"]),
        counterparty_id=cast(str, payload["counterparty_id"]),
        signature=cast(bytes, payload["signature"]),
    )


_REQUEST_KEYS = frozenset(
    {
        "request_id",
        "payment_id",
        "agent_id",
        "key_id",
        "mandate_id",
        "mandate_version",
        "mandate_agent_id",
        "user_ref",
        "user_entity_id",
        "beneficiary_entity_id",
        "mandate_consent_ref",
        "mandate_merchant_id",
        "mandate_payee_id",
        "mandate_cart_hash",
        "mandate_payment_intent_hash",
        "permitted_categories",
        "permitted_products",
        "mandate_credential_id",
        "mandate_credential_scope",
        "required_authentication",
        "max_amount",
        "mandate_currency",
        "mandate_issued_at",
        "mandate_expires_at",
        "amount",
        "currency",
        "merchant_id",
        "payee_id",
        "cart_hash",
        "payment_intent_hash",
        "category",
        "product_id",
        "credential_id",
        "credential_scope",
        "consent_ref",
        "authentication_evidence_ref",
        "nonce",
        "created_at",
        "expires_at",
        "prior_receipt_hash",
        "campaign_id",
        "trace_id",
        "actor_id",
        "counterparty_id",
        "signature",
    }
)
_PAYLOAD_KEYS = _REQUEST_KEYS | {"payer_account", "payee_account"}


def _payload_fingerprint(payload: Mapping[str, object]) -> str:
    tagged: list[list[str]] = []
    for key in sorted(payload):
        if type(key) is not str:
            raise TypeError("agentic command payload keys must be exact strings")
        value = payload[key]
        if value is None:
            kind, text = "none", ""
        elif type(value) is str:
            kind, text = "str", value
        elif type(value) is int:
            kind, text = "int", str(value)
        elif type(value) is Decimal:
            kind, text = "decimal", str(value)
        elif type(value) is datetime:
            kind, text = "datetime", value.isoformat()
        elif type(value) is bytes:
            kind, text = "bytes", value.hex()
        elif type(value) is tuple and all(type(item) is str for item in value):
            kind, text = "tuple[str]", json.dumps(value, separators=(",", ":"))
        else:
            raise TypeError("agentic command payload contains unsupported value type")
        tagged.append([key, kind, text])
    return hashlib.sha256(
        json.dumps(tagged, separators=(",", ":"), ensure_ascii=True).encode()
    ).hexdigest()


class AgenticPaymentCommand(Command):
    """Public immutable request command consumed by campaign generators."""

    __slots__ = ()

    def __init__(
        self,
        request: AgentPaymentRequest,
        *,
        payer_account: str,
        payee_account: str,
    ) -> None:
        if type(request) is not AgentPaymentRequest:
            raise TypeError("request must be an exact AgentPaymentRequest")
        payer = _text("payer_account", payer_account)
        payee = _text("payee_account", payee_account)
        if payer != request.mandate.user_ref or payee != request.payee_id:
            raise ValueError("ledger accounts must match signed request binding")
        if payer == payee:
            raise ValueError("agentic account roles must be pairwise distinct")
        super().__init__(
            "agentic.pay",
            {
                **_request_payload(request),
                "payer_account": payer,
                "payee_account": payee,
            },
        )

    @property
    def request(self) -> AgentPaymentRequest:
        return _request_from_payload(self.payload)

    @property
    def payment_id(self) -> str:
        return cast(str, self.payload["payment_id"])

    @property
    def campaign_id(self) -> str:
        return cast(str, self.payload["campaign_id"])


def _canonical_command(command: Command) -> AgenticPaymentCommand:
    if not isinstance(command, AgenticPaymentCommand) or type(command) is not AgenticPaymentCommand:
        raise LifecycleError("AGENTIC_UNKNOWN_COMMAND")
    try:
        if type(command.name) is not str or command.name != "agentic.pay":
            raise ValueError("command name does not match agentic payment operation")
        if type(command.payload) is not _MAPPING_PROXY_TYPE:
            raise TypeError("command payload must be an owned immutable mapping")
        payload = dict(command.payload)
        if set(payload) != _PAYLOAD_KEYS:
            raise ValueError("agentic command payload fields are invalid")
        incoming_fingerprint = _payload_fingerprint(payload)
        request = _request_from_payload(payload)
        canonical = AgenticPaymentCommand(
            request,
            payer_account=cast(str, payload["payer_account"]),
            payee_account=cast(str, payload["payee_account"]),
        )
        if _payload_fingerprint(canonical.payload) != incoming_fingerprint:
            raise ValueError("agentic command payload is not canonical")
        return canonical
    except (
        AttributeError,
        DecimalException,
        KeyError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as error:
        raise LifecycleError("AGENTIC_COMMAND_INVALID") from error


@dataclass(frozen=True, slots=True)
class AgenticDecision:
    """Small synchronous result separating integrity evidence from risk action."""

    action: Action
    reason_codes: tuple[ReasonCode, ...]
    integrity_receipt: IntegrityReceipt


type RiskScorer = Callable[[AgentPaymentRequest, IntegrityReceipt], Action]


class AgenticRailAdapter:
    """Verify delegated intent before invoking risk controls or moving value."""

    def __init__(self, verifier: TrustVerifier, scorer: RiskScorer) -> None:
        if type(verifier) is not TrustVerifier:
            raise TypeError("verifier must be an exact TrustVerifier")
        if not callable(scorer):
            raise TypeError("scorer must be callable")
        self._verifier = verifier
        self._scorer = scorer

    def initialize(self, context: RailContext) -> None:
        if context.bundle.rail is not Rail.AGENTIC:
            raise ValueError("agentic adapter requires an agentic scenario")

    def process(self, request: AgentPaymentRequest, *, now: datetime) -> AgenticDecision:
        preview = self._verifier.preview(request, now)
        if not preview.allowed:
            assert preview.reason_code is not None
            return AgenticDecision(Action.DECLINE, (preview.reason_code,), preview)
        try:
            action = self._scorer(request, preview)
        except Exception:
            self._verifier.discard_preview(preview)
            raise
        if type(action) is not Action:
            self._verifier.discard_preview(preview)
            raise TypeError("agentic risk scorer must return an exact Action")
        return AgenticDecision(action, (), preview)

    def handle(self, command: Command, context: RailContext) -> list[PaymentEvent]:
        canonical = _canonical_command(command)
        try:
            state = context.entity_state(AGENTIC_TRUST_STATE_ID)
        except KeyError:
            state = {"version": 1, "records": ()}
        try:
            self._verifier.load_state(state)
        except TrustVerifierStateError as error:
            raise LifecycleError(error.code) from error

        request = canonical.request
        decision = self.process(request, now=context.now)
        if not decision.integrity_receipt.allowed:
            event_id = context.new_uuid()
            return [self._event(context, canonical, decision, event_id)]

        outcome = {
            Action.APPROVE: ReceiptOutcome.APPROVE,
            Action.CHALLENGE: ReceiptOutcome.CHALLENGE,
            Action.DECLINE: ReceiptOutcome.DECLINE,
        }[decision.action]
        prepared = self._verifier.prepare_commit(
            request,
            decision.integrity_receipt,
            outcome,
            context.now,
        )
        if type(prepared) is IntegrityReceipt:
            assert prepared.reason_code is not None
            rejected = AgenticDecision(
                Action.DECLINE,
                (prepared.reason_code,),
                prepared,
            )
            event_id = context.new_uuid()
            return [self._event(context, canonical, rejected, event_id)]

        plan = cast(TrustCommitPlan, prepared)
        final_decision = AgenticDecision(decision.action, (), plan.receipt)
        event_id = context.new_uuid()
        event = self._event(context, canonical, final_decision, event_id)
        projected_state = self._verifier.projected_state(plan)

        try:
            if final_decision.action is Action.APPROVE:
                context.post(
                    LedgerEntry(
                        f"{event_id}:agentic-payment",
                        {cast(str, canonical.payload["payer_account"]): request.amount},
                        {cast(str, canonical.payload["payee_account"]): request.amount},
                        request.currency,
                    )
                )
            context.set_entity_state(AGENTIC_TRUST_STATE_ID, projected_state)
        except Exception:
            self._verifier.discard_commit(plan)
            raise
        self._verifier.apply_commit(plan)
        return [event]

    @staticmethod
    def _event(
        context: RailContext,
        command: AgenticPaymentCommand,
        decision: AgenticDecision,
        event_id: str,
    ) -> PaymentEvent:
        request = command.request
        receipt = decision.integrity_receipt
        reason = receipt.reason_code.value if receipt.reason_code is not None else ""
        return PaymentEvent(
            schema_version="1.0.0",
            event_id=event_id,
            campaign_id=request.campaign_id,
            trace_id=request.trace_id,
            rail=Rail.AGENTIC,
            viewpoint=context.bundle.viewpoint,
            event_type=(
                EventKind.AUTHORIZATION
                if decision.action is Action.APPROVE and receipt.allowed
                else EventKind.AUTHENTICATION_CHALLENGE
                if decision.action is Action.CHALLENGE and receipt.allowed
                else EventKind.AUTHORIZATION_DECLINED
            ),
            amount=request.amount,
            currency=request.currency,
            event_time=context.now,
            ingested_at=context.now,
            available_at=context.now,
            decision_at=context.now,
            actor_id=request.actor_id,
            counterparty_id=request.counterparty_id,
            party_refs={
                "user_ref": request.mandate.user_ref,
                "merchant_id": request.merchant_id,
                "payee_id": request.payee_id,
                "user_entity_id": request.mandate.user_entity_id,
                "beneficiary_entity_id": request.mandate.beneficiary_entity_id,
            },
            rail_data={
                "payment_id": request.payment_id,
                "request_id": request.request_id,
                "integrity": "pass" if receipt.allowed else "fail",
                "reason_code": reason,
                "action": decision.action.value,
                "receipt_hash": receipt.receipt_hash,
                "receipt_outcome": receipt.outcome.value,
            },
            lineage={"synthetic": True},
            privacy={"classification": "synthetic"},
        )


__all__ = [
    "AGENTIC_TRUST_STATE_ID",
    "AgenticDecision",
    "AgenticPaymentCommand",
    "AgenticRailAdapter",
    "RiskScorer",
]
