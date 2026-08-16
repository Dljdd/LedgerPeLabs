"""Frozen scenario configuration contracts for simulator and evaluator inputs."""

from enum import StrEnum

from pydantic import Field, field_validator

from apar.contracts._validation import (
    ExternalContract,
    validate_schema_version,
    validate_semantic_version,
)
from apar.contracts.events import Rail


class AttackerMode(StrEnum):
    STATIC = "static"
    ADAPTIVE = "adaptive"
    DECISION_ONLY = "decision_only"


class FeedbackField(StrEnum):
    APPROVE = "approve"
    CHALLENGE = "challenge"
    DECLINE = "decline"
    REALIZED_VALUE = "realized_value"


class _ScenarioParameters(ExternalContract):
    """Validated parameters shared by scenario requests and compiled bundles."""

    schema_version: str = "1.0.0"
    scenario_id: str
    version: str
    rail: Rail
    viewpoint: str
    attacker_mode: AttackerMode
    attacker_objective: str
    query_budget: int = Field(gt=0)
    feedback: list[FeedbackField] = Field(min_length=1)
    benign_entity_count: int = Field(gt=0)
    illicit_entity_count: int = Field(gt=0)
    duration_hours: int = Field(gt=0)
    seed: int = Field(strict=True)
    economics: dict[str, str] = Field(default_factory=dict)
    lifecycle: dict[str, str | int] = Field(default_factory=dict)
    hidden_validity: dict[str, str] = Field(default_factory=dict)
    extensions: dict[str, object] = Field(default_factory=dict)

    @field_validator("schema_version")
    @classmethod
    def schema_version_is_supported(cls, value: str) -> str:
        return validate_schema_version(value)

    @field_validator("version")
    @classmethod
    def version_is_semantic(cls, value: str) -> str:
        return validate_semantic_version(value, field_name="version")


class ScenarioConfig(_ScenarioParameters):
    """A bounded, deterministic request to compile an approved threat card."""

    export_level: str


class ScenarioBundle(_ScenarioParameters):
    """A declared synthetic scenario with bounded attacker capabilities."""

    threat_card_ref: str
    genai_capability: dict[str, bool] = Field(default_factory=dict)
    safety: dict[str, str | bool] = Field(default_factory=dict)
