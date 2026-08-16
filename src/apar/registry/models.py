"""Validated evidence and threat-card records at the registry boundary."""

from datetime import date
from typing import Literal

from pydantic import Field, field_validator

from apar.contracts._validation import ExternalContract, validate_schema_version
from apar.contracts.events import Rail
from apar.contracts.scenarios import ScenarioConfig


class EvidenceRecord(ExternalContract):
    """A reviewed source claim with an explicit fact-versus-inference label."""

    schema_version: str = "1.0.0"
    evidence_id: str = Field(pattern=r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
    direct_source_url: str
    source_type: str
    publisher: str
    published_on: date
    accessed_on: date
    claim: str
    is_project_inference: bool
    quality_grade: Literal["A", "B", "C", "D"]
    reviewer_notes: str
    extensions: dict[str, object] = Field(default_factory=dict)

    @field_validator("schema_version")
    @classmethod
    def schema_version_is_supported(cls, value: str) -> str:
        return validate_schema_version(value)


class ThreatCard(ExternalContract):
    """An evidence-backed, reviewed hypothesis eligible for scenario compilation."""

    schema_version: str = "1.0.0"
    threat_id: str = Field(pattern=r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
    title: str
    version: int = Field(gt=0)
    status: str
    family: str
    confidence: float = Field(ge=0, le=1)
    implementation_status: str
    rails: list[Rail]
    viewpoint: str
    channels: list[str] = Field(default_factory=list)
    lifecycle_stages: list[str] = Field(default_factory=list)
    decision_owner: str = "network"
    genai_capability: dict[str, bool]
    baseline_behavior: str = "matched non-GenAI baseline"
    attacker_objective: str
    attacker_costs: dict[str, str] = Field(default_factory=dict)
    observables: list[str]
    defender_knowledge_boundary: str = Field(min_length=1)
    simulation_status: str = "simulatable"
    safety_class: str | None
    evidence: list[EvidenceRecord]
    default_config: ScenarioConfig
    extensions: dict[str, object] = Field(default_factory=dict)

    @field_validator("schema_version")
    @classmethod
    def schema_version_is_supported(cls, value: str) -> str:
        return validate_schema_version(value)
