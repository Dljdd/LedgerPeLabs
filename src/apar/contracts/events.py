"""Payment event contract and rail-neutral event vocabulary."""

from datetime import datetime
from decimal import Decimal
from enum import StrEnum

from pydantic import Field, field_validator, model_validator

from apar.contracts._validation import (
    ExternalContract,
    validate_schema_version,
    validate_utc_timestamp,
    validate_uuid,
)


class Rail(StrEnum):
    CARD = "card"
    A2A = "a2a"
    AGENTIC = "agentic"


class EventKind(StrEnum):
    AUTHORIZATION = "authorization"
    CLEARING = "clearing"
    SETTLEMENT = "settlement"
    REVERSAL = "reversal"
    TRANSFER_INITIATED = "transfer_initiated"
    TRANSFER_POSTED = "transfer_posted"
    REFUND = "refund"
    FRAUD_REPORTED = "fraud_reported"
    DISPUTE_OPENED = "dispute_opened"
    CHARGEBACK = "chargeback"
    RECOVERY = "recovery"


class LifecycleState(StrEnum):
    INITIATED = "initiated"
    AUTHENTICATED = "authenticated"
    AUTHORIZED = "authorized"
    REJECTED = "rejected"
    DECLINED = "declined"
    CLEARED = "cleared"
    SETTLED = "settled"
    REPORTED = "reported"
    RECOVERED = "recovered"
    LOSS_FINAL = "loss_final"
    REVERSED = "reversed"
    DISPUTED = "disputed"
    CHARGEBACK = "chargeback"


class PaymentEvent(ExternalContract):
    """An immutable payment lifecycle event, as visible to a declared viewpoint."""

    schema_version: str
    event_id: str
    campaign_id: str
    trace_id: str
    rail: Rail
    viewpoint: str
    event_type: EventKind
    amount: Decimal
    currency: str
    event_time: datetime
    ingested_at: datetime
    available_at: datetime
    decision_at: datetime | None = None
    actor_id: str
    counterparty_id: str
    party_refs: dict[str, str] = Field(default_factory=dict)
    rail_data: dict[str, str | int | float | bool] = Field(default_factory=dict)
    lineage: dict[str, str | bool] = Field(default_factory=dict)
    privacy: dict[str, str] = Field(default_factory=dict)
    extensions: dict[str, object] = Field(default_factory=dict)

    @field_validator("schema_version")
    @classmethod
    def schema_version_is_supported(cls, value: str) -> str:
        return validate_schema_version(value)

    @field_validator("event_id", "campaign_id", "trace_id", "actor_id", "counterparty_id")
    @classmethod
    def identifiers_are_uuid_strings(cls, value: str) -> str:
        return validate_uuid(value)

    @field_validator("event_time", "ingested_at", "available_at", "decision_at")
    @classmethod
    def timestamps_are_utc(cls, value: datetime | None) -> datetime | None:
        return None if value is None else validate_utc_timestamp(value)

    @model_validator(mode="after")
    def validate_times_and_amount(self) -> "PaymentEvent":
        if self.ingested_at < self.event_time:
            raise ValueError("ingested_at must be at or after event_time")
        if self.available_at < self.ingested_at:
            raise ValueError("available_at must be at or after ingested_at")
        if self.decision_at is not None and self.decision_at < self.available_at:
            raise ValueError("decision_at must be at or after available_at")
        if self.amount < 0:
            raise ValueError("amount must be non-negative")
        return self
