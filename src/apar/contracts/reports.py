"""Evaluation and promotion-review report contracts."""

from datetime import datetime
from decimal import Decimal
from enum import StrEnum

from pydantic import Field, field_validator

from apar.contracts._validation import (
    ExternalContract,
    validate_schema_version,
    validate_utc_timestamp,
    validate_uuid,
)


class PromotionDecision(StrEnum):
    PROMOTE = "promote"
    HOLD = "hold"
    REJECT = "reject"


class EvaluationReport(ExternalContract):
    """A reproducible evaluation record with explicit human promotion review."""

    schema_version: str = "1.0.0"
    run_id: str
    scenario_id: str
    generator_hash: str
    model_hash: str
    policy_hash: str
    evaluator_hash: str
    dataset_partitions: dict[str, int] = Field(default_factory=dict)
    sample_counts: dict[str, int] = Field(default_factory=dict)
    fraud_prevalence: dict[str, float] = Field(default_factory=dict)
    fraud_value_distribution: dict[str, Decimal] = Field(default_factory=dict)
    family_metrics: dict[str, dict[str, float]] = Field(default_factory=dict)
    segment_metrics: dict[str, dict[str, float]] = Field(default_factory=dict)
    operational_action_rates: dict[str, float] = Field(default_factory=dict)
    operational_budgets: dict[str, Decimal] = Field(default_factory=dict)
    calibration: dict[str, float] = Field(default_factory=dict)
    latency: dict[str, float] = Field(default_factory=dict)
    leakage_tests: dict[str, bool] = Field(default_factory=dict)
    metamorphic_tests: dict[str, bool] = Field(default_factory=dict)
    adaptive_search_ablation: dict[str, float] = Field(default_factory=dict)
    hidden_evaluation_results: dict[str, str | float | bool] = Field(default_factory=dict)
    failed_gates: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    reviewer_id: str
    promotion_decision: PromotionDecision
    reviewed_at: datetime
    extensions: dict[str, object] = Field(default_factory=dict)

    @field_validator("schema_version")
    @classmethod
    def schema_version_is_supported(cls, value: str) -> str:
        return validate_schema_version(value)

    @field_validator("reviewer_id")
    @classmethod
    def reviewer_identifier_is_a_uuid_string(cls, value: str) -> str:
        return validate_uuid(value)

    @field_validator("reviewed_at")
    @classmethod
    def reviewed_at_is_utc(cls, value: datetime) -> datetime:
        return validate_utc_timestamp(value)
