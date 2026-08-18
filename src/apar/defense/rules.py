"""Transparent deterministic rules over defender-visible decision data only."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import Literal

from pydantic import Field, field_validator, model_validator

from apar.contracts._validation import (
    ExternalContract,
    validate_semantic_version,
    validate_utc_timestamp,
)
from apar.contracts.events import EventKind, Rail
from apar.defense.contracts import ObservedEvent
from apar.features.state import FeatureVector


class DefenseReason(StrEnum):
    """Stable public reasons emitted by the deterministic defense layer."""

    INTEGRITY_FAILURE = "INTEGRITY_FAILURE"
    REQUIRED_DATA_MISSING = "REQUIRED_DATA_MISSING"
    ACTOR_VELOCITY = "ACTOR_VELOCITY"
    COUNTERPARTY_VELOCITY = "COUNTERPARTY_VELOCITY"
    AMOUNT_DEVIATION = "AMOUNT_DEVIATION"
    NEW_COUNTERPARTY = "NEW_COUNTERPARTY"
    GRAPH_FAN_IN = "GRAPH_FAN_IN"
    GRAPH_FAN_OUT = "GRAPH_FAN_OUT"
    GRAPH_SHARED_NEIGHBOR = "GRAPH_SHARED_NEIGHBOR"
    FEATURE_STATE_DEGRADED = "FEATURE_STATE_DEGRADED"
    MODEL_UNAVAILABLE = "MODEL_UNAVAILABLE"
    MODEL_TIMEOUT = "MODEL_TIMEOUT"


class RuleSeverity(StrEnum):
    """A compact intervention-oriented severity vocabulary."""

    CHALLENGE = "challenge"
    DECLINE = "decline"


class RuleManifest(ExternalContract):
    """Frozen rule version and explicit initial domain thresholds."""

    schema_version: Literal["1.0.0"] = "1.0.0"
    version: str = "1.0.0"
    actor_count_1m: float = Field(default=4.0, gt=0.0)
    actor_count_10m: float = Field(default=8.0, gt=0.0)
    counterparty_fanin: float = Field(default=5.0, gt=0.0)
    actor_fanout: float = Field(default=5.0, gt=0.0)
    amount_zscore: float = Field(default=4.0, gt=0.0)
    shared_neighbors: float = Field(default=3.0, gt=0.0)
    repeated_pair_count: float = Field(default=4.0, gt=0.0)
    degraded_state: float = Field(default=1.0, gt=0.0)
    threshold_score: float = Field(default=0.60, ge=0.0, le=1.0)

    @field_validator("version")
    @classmethod
    def version_is_semantic(cls, value: str) -> str:
        return validate_semantic_version(value, field_name="version")

    @field_validator(
        "actor_count_1m",
        "actor_count_10m",
        "counterparty_fanin",
        "actor_fanout",
        "amount_zscore",
        "shared_neighbors",
        "repeated_pair_count",
        "degraded_state",
        "threshold_score",
    )
    @classmethod
    def thresholds_are_finite(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("rule thresholds must be finite")
        return value

    @classmethod
    def default(cls) -> RuleManifest:
        """Return the versioned competition rule manifest."""
        return cls()


def rule_manifest_digest(manifest: RuleManifest) -> str:
    """Hash every rule-manifest field in canonical JSON form."""
    if type(manifest) is not RuleManifest:
        raise TypeError("manifest must be an exact RuleManifest")
    validated = RuleManifest.model_validate(manifest)
    return _digest(validated.model_dump(mode="json"))


def feature_vector_digest(vector: FeatureVector) -> str:
    """Hash complete feature values and decision/provenance metadata canonically."""
    if type(vector) is not FeatureVector:
        raise TypeError("vector must be an exact FeatureVector")
    validated = FeatureVector.model_validate(vector)
    ordered_values = tuple(sorted(validated.values.items()))
    if any(not math.isfinite(value) for _, value in ordered_values):
        raise ValueError("feature vector values must be finite")
    document = validated.model_dump(mode="json", exclude={"values"})
    document["ordered_values"] = ordered_values
    return _digest(document)


class RuleHit(ExternalContract):
    """One deterministic rule signal and its complete observation evidence."""

    schema_version: Literal["1.0.0"] = "1.0.0"
    reason: DefenseReason
    score: float = Field(ge=0.0, le=1.0)
    severity: RuleSeverity
    mandatory: bool
    evidence_source_ids: tuple[str, ...]
    rule_version: str

    @field_validator("score")
    @classmethod
    def score_is_finite(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("rule score must be finite")
        return value

    @field_validator("evidence_source_ids")
    @classmethod
    def evidence_is_canonical(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if value != tuple(sorted(set(value))):
            raise ValueError("evidence source IDs must be unique and sorted")
        return value

    @field_validator("rule_version")
    @classmethod
    def rule_version_is_semantic(cls, value: str) -> str:
        return validate_semantic_version(value, field_name="rule_version")

    @model_validator(mode="after")
    def classification_is_consistent(self) -> RuleHit:
        mandatory_reasons = {
            DefenseReason.INTEGRITY_FAILURE,
            DefenseReason.REQUIRED_DATA_MISSING,
        }
        if self.mandatory != (self.reason in mandatory_reasons):
            raise ValueError("mandatory classification must match the rule reason")
        expected_severity = (
            RuleSeverity.DECLINE
            if self.mandatory or self.score >= 0.90
            else RuleSeverity.CHALLENGE
        )
        if self.severity is not expected_severity:
            raise ValueError("rule severity must match mandatory status and score")
        if self.mandatory and self.score != 1.0:
            raise ValueError("mandatory rules must have score 1.0")
        return self


class RuleResult(ExternalContract):
    """Canonical deterministic hits bound to one rule evaluation context."""

    schema_version: Literal["1.0.0"] = "1.0.0"
    hits: tuple[RuleHit, ...] = ()
    score: float = Field(default=0.0, ge=0.0, le=1.0)
    manifest_version: str = "1.0.0"
    event_id: str | None = None
    decision_at: datetime | None = None
    catalog_digest: str | None = None
    vector_digest: str | None = None
    manifest_digest: str = Field(
        default_factory=lambda: rule_manifest_digest(RuleManifest.default())
    )

    @field_validator("score")
    @classmethod
    def score_is_finite(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("rule result score must be finite")
        return value

    @field_validator("manifest_version")
    @classmethod
    def manifest_version_is_semantic(cls, value: str) -> str:
        return validate_semantic_version(value, field_name="manifest_version")

    @field_validator("decision_at")
    @classmethod
    def decision_at_is_utc(cls, value: datetime | None) -> datetime | None:
        return None if value is None else validate_utc_timestamp(value)

    @field_validator("catalog_digest", "vector_digest", "manifest_digest")
    @classmethod
    def digests_are_sha256(cls, value: str | None) -> str | None:
        if value is not None and (
            len(value) != 64 or any(character not in "0123456789abcdef" for character in value)
        ):
            raise ValueError("rule provenance digests must be lowercase SHA-256")
        return value

    @model_validator(mode="after")
    def result_is_canonical(self) -> RuleResult:
        if self.hits != tuple(sorted(self.hits, key=_hit_order)):
            raise ValueError("rule hits must be in canonical order")
        reasons = tuple(hit.reason for hit in self.hits)
        if len(reasons) != len(set(reasons)):
            raise ValueError("rule reasons must be unique")
        if any(hit.rule_version != self.manifest_version for hit in self.hits):
            raise ValueError("rule hit version must match result manifest version")
        expected = _aggregate_risk(self.hits)
        if self.score != expected:
            raise ValueError("rule result score must equal the aggregate risk")
        bindings = (
            self.event_id,
            self.decision_at,
            self.catalog_digest,
            self.vector_digest,
        )
        bound = all(value is not None for value in bindings)
        unbound = all(value is None for value in bindings)
        if not bound and not unbound:
            raise ValueError("rule result must have a complete provenance binding")
        if unbound and (
            self.hits
            or self.score != 0.0
            or self.manifest_version != RuleManifest.default().version
            or self.manifest_digest != rule_manifest_digest(RuleManifest.default())
        ):
            raise ValueError("only the neutral default sentinel may omit provenance binding")
        if bound and (not self.event_id or not self.catalog_digest):
            raise ValueError("rule result provenance binding must not be blank")
        return self

    @classmethod
    def clear(cls) -> RuleResult:
        """Return the sole unbound, score-zero, default-manifest sentinel."""
        return cls()


@dataclass(frozen=True, slots=True)
class RuleEngine:
    """Evaluate fixed rules without evaluator grouping or target information."""

    manifest: RuleManifest

    @classmethod
    def default(cls) -> RuleEngine:
        """Build the competition rule engine from its fixed manifest."""
        return cls(RuleManifest.default())

    def evaluate(self, event: ObservedEvent, vector: FeatureVector) -> RuleResult:
        """Evaluate one observation and its strictly causal feature vector."""
        if type(event) is not ObservedEvent:
            raise TypeError("event must be an exact ObservedEvent")
        if type(vector) is not FeatureVector:
            raise TypeError("vector must be an exact FeatureVector")
        if event.event_id != vector.event_id:
            raise ValueError("event and vector must describe the same event")
        if event.decision_at is not None and event.decision_at != vector.decision_at:
            raise ValueError("event and vector must have the same decision time")

        evidence = tuple(sorted({event.event_id, *vector.source_event_ids}))
        hits = [self._hit(reason, 1.0, True, evidence) for reason in mandatory_reasons(event)]

        actor_velocity = _strongest(
            _rule_score(self._value(vector, "actor_count_1m"), self.manifest.actor_count_1m),
            _rule_score(
                self._value(vector, "actor_count_10m"), self.manifest.actor_count_10m
            ),
        )
        self._append(hits, DefenseReason.ACTOR_VELOCITY, actor_velocity, evidence)
        self._append(
            hits,
            DefenseReason.GRAPH_FAN_IN,
            _rule_score(
                self._value(vector, "graph_counterparty_fanin"),
                self.manifest.counterparty_fanin,
            ),
            evidence,
        )
        self._append(
            hits,
            DefenseReason.GRAPH_FAN_OUT,
            _rule_score(
                self._value(vector, "graph_actor_fanout"), self.manifest.actor_fanout
            ),
            evidence,
        )
        amount_deviation = max(
            abs(self._value(vector, "actor_amount_zscore_24h")),
            abs(self._value(vector, "counterparty_amount_zscore_24h")),
        )
        self._append(
            hits,
            DefenseReason.AMOUNT_DEVIATION,
            _rule_score(amount_deviation, self.manifest.amount_zscore),
            evidence,
        )
        self._append(
            hits,
            DefenseReason.GRAPH_SHARED_NEIGHBOR,
            _rule_score(
                self._value(vector, "graph_shared_neighbor_count"),
                self.manifest.shared_neighbors,
            ),
            evidence,
        )
        self._append(
            hits,
            DefenseReason.COUNTERPARTY_VELOCITY,
            _rule_score(
                self._value(vector, "pair_prior_count"), self.manifest.repeated_pair_count
            ),
            evidence,
        )
        degraded = self._value(vector, "dq_degraded_state")
        if degraded >= self.manifest.degraded_state:
            hits.append(
                self._hit(
                    DefenseReason.FEATURE_STATE_DEGRADED,
                    self.manifest.threshold_score,
                    False,
                    evidence,
                )
            )

        ordered = tuple(sorted(hits, key=_hit_order))
        return RuleResult(
            hits=ordered,
            score=_aggregate_risk(ordered),
            manifest_version=self.manifest.version,
            event_id=event.event_id,
            decision_at=vector.decision_at,
            catalog_digest=vector.catalog_digest,
            vector_digest=feature_vector_digest(vector),
            manifest_digest=rule_manifest_digest(self.manifest),
        )

    def _value(self, vector: FeatureVector, name: str) -> float:
        value = vector.values.get(name, 0.0)
        if not math.isfinite(value):
            raise ValueError(f"rule feature {name} must be finite")
        return value

    def _append(
        self,
        hits: list[RuleHit],
        reason: DefenseReason,
        score: float | None,
        evidence: tuple[str, ...],
    ) -> None:
        if score is not None:
            hits.append(self._hit(reason, score, False, evidence))

    def _hit(
        self,
        reason: DefenseReason,
        score: float,
        mandatory: bool,
        evidence: tuple[str, ...],
    ) -> RuleHit:
        return RuleHit(
            reason=reason,
            score=score,
            severity=(
                RuleSeverity.DECLINE if mandatory or score >= 0.90 else RuleSeverity.CHALLENGE
            ),
            mandatory=mandatory,
            evidence_source_ids=evidence,
            rule_version=self.manifest.version,
        )


def mandatory_reasons(event: ObservedEvent) -> tuple[DefenseReason, ...]:
    """Classify mandatory schema and integrity gates from the current event only."""
    if type(event) is not ObservedEvent:
        raise TypeError("event must be an exact ObservedEvent")
    allowed_kinds = {
        Rail.CARD: {EventKind.AUTHORIZATION, EventKind.AUTHORIZATION_DECLINED},
        Rail.A2A: {EventKind.TRANSFER_INITIATED},
        Rail.AGENTIC: {
            EventKind.AUTHORIZATION,
            EventKind.AUTHENTICATION_CHALLENGE,
            EventKind.AUTHORIZATION_DECLINED,
        },
    }
    expected_integrity = (
        {"pass", "fail"} if event.rail is Rail.AGENTIC else {"not_applicable"}
    )
    coherent_schema = (
        event.is_decision_point
        and event.decision_at is not None
        and event.event_type in allowed_kinds[event.rail]
        and event.integrity_status in expected_integrity
    )
    required_missing = (
        not event.event_id
        or not event.payment_id
        or not event.actor_id
        or not event.counterparty_id
        or not event.currency
        or event.decision_at is None
        or not event.amount.is_finite()
        or event.amount < 0
        or not coherent_schema
    )
    reasons: list[DefenseReason] = []
    if (
        coherent_schema
        and event.rail is Rail.AGENTIC
        and event.integrity_status == "fail"
    ):
        reasons.append(DefenseReason.INTEGRITY_FAILURE)
    if required_missing:
        reasons.append(DefenseReason.REQUIRED_DATA_MISSING)
    return tuple(reasons)


def _rule_score(value: float, threshold: float) -> float | None:
    if value < threshold:
        return None
    ratio = value / threshold
    return min(1.0, 0.60 + 0.20 * (ratio - 1.0))


def _strongest(*scores: float | None) -> float | None:
    present = tuple(score for score in scores if score is not None)
    return max(present, default=None)


def _aggregate_risk(hits: tuple[RuleHit, ...]) -> float:
    return 1.0 - math.prod(1.0 - hit.score for hit in hits)


def _hit_order(hit: RuleHit) -> tuple[bool, float, str]:
    return (not hit.mandatory, -hit.score, hit.reason.value)


def _digest(document: object) -> str:
    payload = json.dumps(
        document,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
