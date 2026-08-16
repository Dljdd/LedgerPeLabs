"""Decision records with event-time source constraints."""

from datetime import datetime
from enum import StrEnum

from pydantic import Field, field_validator, model_validator

from apar.contracts._validation import (
    ExternalContract,
    validate_schema_version,
    validate_utc_timestamp,
    validate_uuid,
)


class Action(StrEnum):
    APPROVE = "approve"
    CHALLENGE = "challenge"
    DECLINE = "decline"


class ReasonCode(StrEnum):
    VELOCITY_1M = "VELOCITY_1M"
    BENEFICIARY_RECENTLY_ADDED = "BENEFICIARY_RECENTLY_ADDED"
    DEVICE_FAN_OUT = "DEVICE_FAN_OUT"
    AMOUNT_OUTLIER = "AMOUNT_OUTLIER"
    MANDATE_SCOPE_VIOLATION = "MANDATE_SCOPE_VIOLATION"
    INTEGRITY_FAILURE = "INTEGRITY_FAILURE"
    AGENT_IDENTITY_MISMATCH = "AGENT_IDENTITY_MISMATCH"
    SIGNATURE_INVALID = "SIGNATURE_INVALID"
    AMOUNT_LIMIT_EXCEEDED = "AMOUNT_LIMIT_EXCEEDED"
    CURRENCY_MISMATCH = "CURRENCY_MISMATCH"
    PAYEE_BINDING_MISMATCH = "PAYEE_BINDING_MISMATCH"
    CART_HASH_MISMATCH = "CART_HASH_MISMATCH"
    MANDATE_EXPIRED = "MANDATE_EXPIRED"
    NONCE_REPLAY = "NONCE_REPLAY"
    RECEIPT_CHAIN_BROKEN = "RECEIPT_CHAIN_BROKEN"
    MERCHANT_BINDING_MISMATCH = "MERCHANT_BINDING_MISMATCH"
    CATEGORY_SCOPE_VIOLATION = "CATEGORY_SCOPE_VIOLATION"
    PRODUCT_SCOPE_VIOLATION = "PRODUCT_SCOPE_VIOLATION"
    PAYMENT_INTENT_HASH_MISMATCH = "PAYMENT_INTENT_HASH_MISMATCH"
    CREDENTIAL_BINDING_MISMATCH = "CREDENTIAL_BINDING_MISMATCH"
    TOKEN_SCOPE_VIOLATION = "TOKEN_SCOPE_VIOLATION"
    CONSENT_BINDING_MISMATCH = "CONSENT_BINDING_MISMATCH"
    AUTHENTICATION_REQUIRED = "AUTHENTICATION_REQUIRED"
    MANDATE_TIME_SCOPE_VIOLATION = "MANDATE_TIME_SCOPE_VIOLATION"
    EXECUTION_RECEIPT_MISMATCH = "EXECUTION_RECEIPT_MISMATCH"


class Decision(ExternalContract):
    """An immutable decision made using only strictly earlier source timestamps."""

    schema_version: str = "1.0.0"
    decision_id: str
    event_id: str
    decision_time: datetime
    max_source_timestamp: datetime
    score: float = Field(ge=0, le=1)
    action: Action
    reason_codes: list[ReasonCode] = Field(default_factory=list)
    model_version: str
    policy_version: str | None = None
    evidence_refs: list[str] = Field(default_factory=list)
    extensions: dict[str, object] = Field(default_factory=dict)

    @field_validator("schema_version")
    @classmethod
    def schema_version_is_supported(cls, value: str) -> str:
        return validate_schema_version(value)

    @field_validator("decision_id", "event_id")
    @classmethod
    def identifiers_are_uuid_strings(cls, value: str) -> str:
        return validate_uuid(value)

    @field_validator("decision_time", "max_source_timestamp")
    @classmethod
    def timestamps_are_utc(cls, value: datetime) -> datetime:
        return validate_utc_timestamp(value)

    @model_validator(mode="after")
    def validate_source_time_and_reason_codes(self) -> "Decision":
        if self.max_source_timestamp >= self.decision_time:
            raise ValueError("max_source_timestamp must be strictly before decision_time")
        if self.action is not Action.APPROVE and not self.reason_codes:
            raise ValueError("reason_codes must be non-empty for non-approve decisions")
        return self
