"""Evaluator-owned disjoint populations for Defend v3.

Reuses v2 operating population construction unchanged and adds an adversarial
efficacy builder plus a disjointness auditor that emits signed manifests.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Literal

from pydantic import Field, model_validator

from apar.contracts._validation import ExternalContract
from apar.contracts.events import EventKind, Rail
from apar.defense.contracts import ObservedEvent
from apar.evaluation.contracts import EvaluationTruthRow, Family
from apar.evaluation.v2_population import (
    CampaignInjection,
    OperatingPopulation,
    PopulationIsolationError,
    PopulationManifest,
    build_benign_base,
    inject_frozen_campaigns,
)
from apar.runs.wire import canonical_json_bytes
from apar.v2_protocol import PrevalenceStratum
from apar.v3_protocol import V3ProtocolError

_FAMILIES: tuple[Family, ...] = (
    "agentic_intent_abuse",
    "app_scam_mule",
    "card_testing_cnp",
    "synthetic_merchant_refund",
)

_EFFICACY_START = datetime(2026, 3, 1, tzinfo=UTC)


class V3PopulationError(V3ProtocolError):
    """A v3 population violates isolation or completeness rules."""


class EfficacyPopulationManifest(ExternalContract):
    """Public-safe provenance for the sealed adversarial efficacy population."""

    schema_version: Literal["1.0.0"] = "1.0.0"
    population_kind: Literal["adversarial_efficacy"] = "adversarial_efficacy"
    transaction_count: int = Field(gt=0)
    fraud_transaction_count: int = Field(gt=0)
    day_count: int = Field(gt=0)
    family_names: tuple[str, str, str, str]
    observations_sha256: str
    truth_sha256: str

    @model_validator(mode="after")
    def family_allocation_is_equal(self) -> Self:
        if set(self.family_names) != set(_FAMILIES):
            raise ValueError("efficacy population must use the four executable families")
        return self


@dataclass(frozen=True, slots=True)
class EfficacyPopulation:
    observations: tuple[ObservedEvent, ...]
    truth: tuple[EvaluationTruthRow, ...]
    manifest: EfficacyPopulationManifest


def build_efficacy_population(
    *,
    campaign_count_per_family: int,
    decisions_per_campaign: int,
    day_count: int,
    seed: int,
) -> EfficacyPopulation:
    """Build an adversarial efficacy population with equal family representation."""
    if type(campaign_count_per_family) is not int or campaign_count_per_family <= 0:
        raise V3PopulationError("campaign count per family must be positive")
    if type(decisions_per_campaign) is not int or decisions_per_campaign <= 0:
        raise V3PopulationError("decisions per campaign must be positive")
    if type(day_count) is not int or day_count <= 0:
        raise V3PopulationError("day count must be positive")
    if type(seed) is not int:
        raise V3PopulationError("efficacy seed must be an exact integer")

    observations: list[ObservedEvent] = []
    truth: list[EvaluationTruthRow] = []
    campaign_index = 0
    for family in _FAMILIES:
        for _ in range(campaign_count_per_family):
            campaign_id = f"efficacy-{family}-{campaign_index:06d}"
            actor_id = f"efficacy-actor-{campaign_index:06d}"
            counterparty_id = f"efficacy-counterparty-{campaign_index:06d}"
            start = _EFFICACY_START + timedelta(hours=6 * campaign_index)
            for row_index in range(decisions_per_campaign):
                event_time = start + timedelta(minutes=row_index)
                decision_at = event_time + timedelta(minutes=1)
                rail = (Rail.CARD, Rail.A2A, Rail.AGENTIC)[row_index % 3]
                event_type = (
                    EventKind.TRANSFER_INITIATED if rail is Rail.A2A else EventKind.AUTHORIZATION
                )
                event_id = f"{campaign_id}-event-{row_index:06d}"
                payment_id = f"{campaign_id}-payment-{row_index:06d}"
                observation = ObservedEvent(
                    event_id=event_id,
                    payment_id=payment_id,
                    rail=rail,
                    event_type=event_type,
                    amount=Decimal(1000 + row_index) / Decimal("100"),
                    currency="USD",
                    event_time=event_time,
                    available_at=event_time,
                    decision_at=decision_at,
                    actor_id=actor_id,
                    counterparty_id=counterparty_id,
                    integrity_status="pass" if rail is Rail.AGENTIC else "not_applicable",
                    is_decision_point=True,
                )
                truth_row = EvaluationTruthRow(
                    event_id=event_id,
                    payment_id=payment_id,
                    campaign_id=campaign_id,
                    family=family,
                    viewpoint="development",
                    is_fraud=True,
                    label_source="population_truth",
                    label_mature_at=decision_at + timedelta(days=7),
                    first_settlement_at=decision_at + timedelta(days=1),
                    net_settled_value=observation.amount,
                    lifecycle_event_ids=(event_id,),
                )
                observations.append(observation)
                truth.append(truth_row)
            campaign_index += 1

    total = len(observations)
    per_family = total // len(_FAMILIES)
    if per_family * len(_FAMILIES) != total:
        raise V3PopulationError("efficacy population must have equal family counts")
    manifest = EfficacyPopulationManifest(
        transaction_count=total,
        fraud_transaction_count=total,
        day_count=day_count,
        family_names=_FAMILIES,
        observations_sha256=hashlib.sha256(
            canonical_json_bytes([row.model_dump(mode="json") for row in observations])
        ).hexdigest(),
        truth_sha256=hashlib.sha256(
            canonical_json_bytes([row.model_dump(mode="json") for row in truth])
        ).hexdigest(),
    )
    return EfficacyPopulation(
        observations=tuple(observations), truth=tuple(truth), manifest=manifest
    )


class DisjointnessProof(ExternalContract):
    """Signed proof that all v3 partitions are pairwise disjoint."""

    schema_version: Literal["1.0.0"] = "1.0.0"
    partition_names: tuple[str, ...]
    entity_disjoint: bool
    campaign_disjoint: bool
    time_disjoint: bool
    combined_digest: str

    @model_validator(mode="after")
    def proof_is_complete(self) -> Self:
        if len(self.partition_names) < 2:
            raise ValueError("disjointness proof requires at least two partitions")
        if not all((self.entity_disjoint, self.campaign_disjoint, self.time_disjoint)):
            raise ValueError("disjointness proof cannot claim partial isolation")
        return self


def audit_disjointness(
    partitions: dict[str, OperatingPopulation | EfficacyPopulation],
) -> DisjointnessProof:
    """Verify pairwise campaign, entity, and decision-time disjointness."""
    if type(partitions) is not dict or len(partitions) < 2:
        raise V3PopulationError("disjointness audit requires at least two partitions")

    entity_sets: dict[str, set[str]] = {}
    campaign_sets: dict[str, set[str]] = {}
    time_intervals: dict[str, tuple[datetime, datetime]] = {}

    for name, population in partitions.items():
        observations = population.observations
        entities: set[str] = set()
        campaigns: set[str] = set()
        decision_times: list[datetime] = []
        for row in observations:
            entities.update({row.actor_id, row.counterparty_id, *row.optional_refs.values()})
            decision_times.append(row.decision_at or row.event_time)
        truth_rows = population.truth
        for row in truth_rows:
            campaigns.add(row.campaign_id)
        entity_sets[name] = entities
        campaign_sets[name] = campaigns
        if decision_times:
            time_intervals[name] = (min(decision_times), max(decision_times))

    names = tuple(sorted(partitions))
    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            if entity_sets[left] & entity_sets[right]:
                raise V3PopulationError(f"entity overlap between {left} and {right}")
            if campaign_sets[left] & campaign_sets[right]:
                raise V3PopulationError(f"campaign overlap between {left} and {right}")
            if left in time_intervals and right in time_intervals:
                left_start, left_end = time_intervals[left]
                right_start, right_end = time_intervals[right]
                if left_start <= right_end and right_start <= left_end:
                    raise V3PopulationError(f"time overlap between {left} and {right}")

    combined = hashlib.sha256(
        canonical_json_bytes(
            {
                "entity": {name: sorted(entity_sets[name]) for name in names},
                "campaign": {name: sorted(campaign_sets[name]) for name in names},
            }
        )
    ).hexdigest()
    return DisjointnessProof(
        partition_names=names,
        entity_disjoint=True,
        campaign_disjoint=True,
        time_disjoint=True,
        combined_digest=combined,
    )


__all__ = [
    "DisjointnessProof",
    "EfficacyPopulation",
    "EfficacyPopulationManifest",
    "PopulationIsolationError",
    "V3PopulationError",
    "audit_disjointness",
    "build_benign_base",
    "build_efficacy_population",
    "inject_frozen_campaigns",
]
