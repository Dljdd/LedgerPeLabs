"""Frozen scenario configuration contracts for simulator and evaluator inputs."""

from enum import StrEnum

from pydantic import Field, field_validator

from apar.contracts._validation import ExternalContract, validate_schema_version
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


class ScenarioBundle(ExternalContract):
    """A declared synthetic scenario with bounded attacker capabilities."""

    schema_version: str = "1.0.0"
    scenario_id: str
    version: str
    threat_card_ref: str
    rail: Rail
    viewpoint: str
    genai_capability: dict[str, bool] = Field(default_factory=dict)
    attacker_mode: AttackerMode
    attacker_objective: str
    query_budget: int = Field(gt=0)
    feedback: list[FeedbackField] = Field(min_length=1)
    economics: dict[str, str] = Field(default_factory=dict)
    lifecycle: dict[str, str | int] = Field(default_factory=dict)
    hidden_validity: dict[str, str] = Field(default_factory=dict)
    safety: dict[str, str | bool] = Field(default_factory=dict)
    extensions: dict[str, object] = Field(default_factory=dict)

    @field_validator("schema_version", "version")
    @classmethod
    def versions_are_supported(cls, value: str) -> str:
        return validate_schema_version(value)
