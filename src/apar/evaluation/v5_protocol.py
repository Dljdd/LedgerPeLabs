"""Sealed Defend v5 development protocol contracts."""

from __future__ import annotations

import hashlib
import json
import math
from enum import StrEnum
from pathlib import Path
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class V5Partition(StrEnum):
    TRAIN = "train"
    CALIBRATION = "calibration"
    THRESHOLD = "threshold"
    DEVELOPMENT_TEST = "development_test"
    HARDENING_TRAIN = "hardening_train"
    ADAPTIVE_HOLDOUT = "adaptive_holdout"


class V5Profile(StrEnum):
    SMOKE = "smoke"
    PRODUCTION = "production"


class V5Family(StrEnum):
    AGENTIC_INTENT_ABUSE = "agentic_intent_abuse"
    APP_SCAM_MULE = "app_scam_mule"
    CARD_TESTING_CNP = "card_testing_cnp"
    SYNTHETIC_MERCHANT_REFUND = "synthetic_merchant_refund"


class V5ReadinessTargets(BaseModel):
    model_config = ConfigDict(frozen=True)

    family_recall_min: float = 0.75
    false_decline_rate_max: float = 0.001
    manual_review_rate_max: float = 0.01
    challenge_rate_max: float = 0.02
    captured_value_fraction_min: float = 0.70
    expected_calibration_error_max: float = 0.10
    p95_decision_latency_ms_max: float = 50.0

    @field_validator("*", mode="before")
    @classmethod
    def values_are_finite(cls, value: object) -> object:
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError("readiness target must be finite")
        return value


class V5SeedSets(BaseModel):
    model_config = ConfigDict(frozen=True)

    train: int = Field(gt=0)
    calibration: int = Field(gt=0)
    threshold: int = Field(gt=0)
    development_test: int = Field(gt=0)
    hardening_train: int = Field(gt=0)
    adaptive_holdout: int = Field(gt=0)
    bootstrap: int = Field(gt=0)
    catboost_seeds: tuple[int, ...] = Field(min_length=3, max_length=5)

    @model_validator(mode="after")
    def seeds_are_unique(self) -> Self:
        all_seeds = [
            self.train, self.calibration, self.threshold,
            self.development_test, self.hardening_train,
            self.adaptive_holdout, self.bootstrap, *self.catboost_seeds,
        ]
        if len(all_seeds) != len(set(all_seeds)):
            raise ValueError("seed sets must be unique")
        return self


class V5PopulationCounts(BaseModel):
    model_config = ConfigDict(frozen=True)

    legitimate_decisions: int = Field(gt=0)
    campaigns_per_family: dict[str, int]

    @model_validator(mode="after")
    def families_are_complete(self) -> Self:
        expected = {member.value for member in V5Family}
        if set(self.campaigns_per_family.keys()) != expected:
            raise ValueError("campaign counts must contain exactly the four v5 families")
        if any(v < 1 for v in self.campaigns_per_family.values()):
            raise ValueError("each family must have at least one campaign")
        return self


class V5DevelopmentProtocol(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    protocol_id: str
    schema_version: str = "1.0.0"
    development_only: bool = True
    sealed_evaluation_allowed: bool = False
    readiness: V5ReadinessTargets = V5ReadinessTargets()
    seeds: V5SeedSets
    smoke_profile: V5PopulationCounts
    production_profile: V5PopulationCounts
    bootstrap_replicates: int = Field(ge=2000)
    feature_catalog_path: str = "config/defense/feature-catalog-v5.json"
    protocol_sha256: str = ""

    @model_validator(mode="after")
    def validate_protocol(self) -> Self:
        if not self.development_only:
            raise ValueError("v5 protocol is development-only")
        if self.sealed_evaluation_allowed:
            raise ValueError("sealed evaluation must be forbidden in v5")
        return self


def _canonical_json_bytes(value: object) -> bytes:
    def check(item: object) -> None:
        if isinstance(item, float) and (math.isnan(item) or math.isinf(item)):
            raise ValueError("non-finite number in protocol document")
        if isinstance(item, dict):
            for v in item.values():
                check(v)
        elif isinstance(item, list):
            for v in item:
                check(v)

    check(value)
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def load_v5_development_protocol(path: Path) -> V5DevelopmentProtocol:
    """Load, validate, freeze, and digest the development protocol."""
    raw = path.read_bytes()
    document = json.loads(raw)
    digest_input = {k: v for k, v in document.items() if k != "protocol_sha256"}
    digest = hashlib.sha256(_canonical_json_bytes(digest_input)).hexdigest()
    document["protocol_sha256"] = digest
    return V5DevelopmentProtocol.model_validate(document)


__all__ = [
    "V5DevelopmentProtocol",
    "V5Family",
    "V5Partition",
    "V5Profile",
    "V5ReadinessTargets",
    "load_v5_development_protocol",
]
