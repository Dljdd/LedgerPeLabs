"""Frozen scenario configuration contracts for simulator and evaluator inputs."""

from datetime import datetime
from enum import StrEnum

from pydantic import Field, field_validator, model_validator

from apar.contracts._validation import (
    ExternalContract,
    validate_schema_version,
    validate_semantic_version,
    validate_utc_timestamp,
)
from apar.contracts.events import Rail


class AttackerMode(StrEnum):
    STATIC = "static"
    ADAPTIVE = "adaptive"
    DECISION_ONLY = "decision_only"


class FeedbackField(StrEnum):
    ACTION = "action"
    REASON_FAMILY = "reason_family"
    APPROVE = "approve"
    CHALLENGE = "challenge"
    DECLINE = "decline"
    REALIZED_VALUE = "realized_value"


class ReplayOrdering(StrEnum):
    """Stable total ordering used when replay timestamps are equal."""

    EVENT_TIME_THEN_EVENT_ID = "event_time_then_event_id"


class CampaignStage(ExternalContract):
    """One declared phase in a synthetic campaign."""

    stage_id: str = Field(pattern=r"^[a-z0-9]+(?:_[a-z0-9]+)*$")
    description: str = Field(min_length=1)


class StageTransition(ExternalContract):
    """A deterministic rule connecting two declared campaign stages."""

    from_stage: str = Field(pattern=r"^[a-z0-9]+(?:_[a-z0-9]+)*$")
    to_stage: str = Field(pattern=r"^[a-z0-9]+(?:_[a-z0-9]+)*$")
    condition: str = Field(min_length=1)


class ReplayConfig(ExternalContract):
    """Inputs required to reproduce simulator ordering and randomness."""

    random_seed: int = Field(strict=True)
    simulation_start: datetime
    generator_version: str
    event_ordering: ReplayOrdering

    @field_validator("simulation_start")
    @classmethod
    def simulation_start_is_utc(cls, value: datetime) -> datetime:
        return validate_utc_timestamp(value)

    @field_validator("generator_version")
    @classmethod
    def generator_version_is_semantic(cls, value: str) -> str:
        return validate_semantic_version(value, field_name="generator_version")


class ReplayManifest(ReplayConfig):
    """Self-contained deterministic replay identity emitted by the compiler."""

    scenario_id: str
    scenario_version: str
    threat_card_ref: str

    @field_validator("scenario_version")
    @classmethod
    def scenario_version_is_semantic(cls, value: str) -> str:
        return validate_semantic_version(value, field_name="scenario_version")


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
    campaign_stages: list[CampaignStage] = Field(min_length=1)
    transition_rules: list[StageTransition] = Field(min_length=1)
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

    @model_validator(mode="after")
    def campaign_graph_is_closed(self) -> "_ScenarioParameters":
        stage_ids = [stage.stage_id for stage in self.campaign_stages]
        if len(stage_ids) != len(set(stage_ids)):
            raise ValueError("campaign stage IDs must be unique")
        declared = set(stage_ids)
        for transition in self.transition_rules:
            if transition.from_stage not in declared or transition.to_stage not in declared:
                raise ValueError("transition rules must reference a declared campaign stage")
        return self


class ScenarioConfig(_ScenarioParameters):
    """A bounded, deterministic request to compile an approved threat card."""

    export_level: str
    replay: ReplayConfig

    @model_validator(mode="after")
    def replay_seed_matches_execution_seed(self) -> "ScenarioConfig":
        if self.replay.random_seed != self.seed:
            raise ValueError("replay random_seed must equal scenario seed")
        return self


class ScenarioBundle(_ScenarioParameters):
    """A declared synthetic scenario with bounded attacker capabilities."""

    threat_card_ref: str
    defender_knowledge_boundary: str = Field(min_length=1)
    replay_manifest: ReplayManifest
    genai_capability: dict[str, bool]
    safety: dict[str, str | bool]
