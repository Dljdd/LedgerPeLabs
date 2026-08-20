"""Evaluator-owned, in-memory construction of synthetic Defend v2 populations."""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Literal

from pydantic import Field, model_validator

from apar.contracts._validation import ExternalContract
from apar.contracts.events import EventKind, Rail
from apar.defense.contracts import ObservedEvent
from apar.evaluation.contracts import EvaluationTruthRow, Family
from apar.evaluation.v2_protocol import PrevalenceStratum, SeedCommitment, V2Protocol
from apar.runs.wire import canonical_json_bytes

_BASE_START = datetime(2026, 1, 1, tzinfo=UTC)
_INJECTION_START = datetime(2026, 2, 1, tzinfo=UTC)
_FAMILIES: tuple[Family, ...] = (
    "agentic_intent_abuse",
    "app_scam_mule",
    "card_testing_cnp",
    "synthetic_merchant_refund",
)


class PopulationIsolationError(ValueError):
    """Raised when a proposed population violates evaluator isolation rules."""


class PopulationManifest(ExternalContract):
    """Public-safe provenance for an in-memory operating population."""

    schema_version: Literal["1.0.0"] = "1.0.0"
    population_kind: Literal["benign_base", "injected"]
    transaction_count: int = Field(ge=0)
    fraud_transaction_count: int = Field(ge=0)
    day_count: int = Field(gt=0)
    stratum_name: Literal["low", "medium", "high"] | None = None
    seed_commitments: tuple[SeedCommitment, ...]
    observations_sha256: str
    truth_sha256: str


class CampaignInjection(ExternalContract):
    """A frozen evaluator campaign whose observation and truth channels agree."""

    campaign_id: str = Field(min_length=1)
    family: Family
    entity_ids: tuple[str, ...]
    observations: tuple[ObservedEvent, ...]
    truth: tuple[EvaluationTruthRow, ...]

    @model_validator(mode="after")
    def declared_entities_are_nonempty_and_unique(self) -> "CampaignInjection":
        if not self.entity_ids or any(not item for item in self.entity_ids):
            raise ValueError("campaign entities must be non-empty")
        if len(self.entity_ids) != len(set(self.entity_ids)):
            raise ValueError("campaign entity ids must be unique")
        return self

    @classmethod
    def fixture(
        cls,
        *,
        campaign_id: str = "fixture-campaign",
        family: Family = "card_testing_cnp",
        decision_count: int = 1,
        entity_ids: tuple[str, ...] | None = None,
        start_at: datetime = _INJECTION_START,
    ) -> "CampaignInjection":
        """Create an in-memory synthetic campaign for fixture-only tests."""
        if type(decision_count) is not int or decision_count <= 0:
            raise ValueError("fixture decision count must be positive")
        if start_at.tzinfo is None or start_at.utcoffset() is None:
            raise ValueError("fixture start time must be timezone-aware")
        entities = entity_ids or (f"{campaign_id}-actor", f"{campaign_id}-counterparty")
        actor_id = entities[0]
        counterparty_id = entities[1] if len(entities) > 1 else entities[0]
        observations: list[ObservedEvent] = []
        truth: list[EvaluationTruthRow] = []
        for index in range(decision_count):
            event_time = start_at + timedelta(minutes=index)
            decision_at = event_time + timedelta(minutes=1)
            event_id = f"{campaign_id}-event-{index:06d}"
            payment_id = f"{campaign_id}-payment-{index:06d}"
            observations.append(
                ObservedEvent(
                    event_id=event_id,
                    payment_id=payment_id,
                    rail=Rail.CARD,
                    event_type=EventKind.AUTHORIZATION,
                    amount=Decimal("10.00"),
                    currency="USD",
                    event_time=event_time,
                    available_at=event_time,
                    decision_at=decision_at,
                    actor_id=actor_id,
                    counterparty_id=counterparty_id,
                    integrity_status="not_applicable",
                    is_decision_point=True,
                )
            )
            truth.append(
                EvaluationTruthRow(
                    event_id=event_id,
                    payment_id=payment_id,
                    campaign_id=campaign_id,
                    family=family,
                    viewpoint="development",
                    is_fraud=True,
                    label_source="population_truth",
                    label_mature_at=decision_at + timedelta(days=7),
                    first_settlement_at=decision_at + timedelta(days=1),
                    net_settled_value=Decimal("10.00"),
                    lifecycle_event_ids=(event_id,),
                )
            )
        return cls(
            campaign_id=campaign_id,
            family=family,
            entity_ids=entities,
            observations=tuple(observations),
            truth=tuple(truth),
        )


@dataclass(frozen=True, slots=True)
class OperatingPopulation:
    """Separately addressed evaluator observations, truth, and provenance."""

    observations: tuple[ObservedEvent, ...]
    truth: tuple[EvaluationTruthRow, ...]
    manifest: PopulationManifest


def build_benign_base(protocol: V2Protocol, *, seed: int) -> OperatingPopulation:
    """Create an independent benign decision population from a sealed profile."""
    _require_seed(seed)
    if type(protocol) is not V2Protocol:
        raise TypeError("protocol must be an exact V2Protocol")
    count = protocol.operating.transaction_count
    days = protocol.operating.day_count
    seed_commitment = _seed_commitment("operating_population", seed)
    identity_prefix = seed_commitment.commitment_sha256[:16]
    rng = random.Random(seed)
    observations: list[ObservedEvent] = []
    truth: list[EvaluationTruthRow] = []
    for index in range(count):
        day = index * days // count
        event_time = _BASE_START + timedelta(days=day, seconds=index % max(1, count // days))
        decision_at = event_time + timedelta(minutes=1)
        rail = (Rail.CARD, Rail.A2A, Rail.AGENTIC)[index % 3]
        event_type = EventKind.TRANSFER_INITIATED if rail is Rail.A2A else EventKind.AUTHORIZATION
        event_id = f"benign-{identity_prefix}-event-{index:06d}"
        payment_id = f"benign-{identity_prefix}-payment-{index:06d}"
        observation = ObservedEvent(
            event_id=event_id,
            payment_id=payment_id,
            rail=rail,
            event_type=event_type,
            amount=Decimal(rng.randint(100, 10_000)) / Decimal("100"),
            currency="USD",
            event_time=event_time,
            available_at=event_time,
            decision_at=decision_at,
            actor_id=f"benign-{identity_prefix}-actor-{index:06d}",
            counterparty_id=f"benign-{identity_prefix}-counterparty-{index:06d}",
            integrity_status="pass" if rail is Rail.AGENTIC else "not_applicable",
            is_decision_point=True,
        )
        observations.append(observation)
        truth.append(
            EvaluationTruthRow(
                event_id=event_id,
                payment_id=payment_id,
                campaign_id=f"benign-base-{identity_prefix}",
                family="card_testing_cnp",
                viewpoint="development",
                is_fraud=False,
                label_source="population_truth",
                label_mature_at=decision_at + timedelta(days=7),
                first_settlement_at=decision_at + timedelta(days=1),
                net_settled_value=observation.amount,
                lifecycle_event_ids=(event_id,),
            )
        )
    return _population(
        observations=tuple(observations),
        truth=tuple(truth),
        population_kind="benign_base",
        day_count=days,
        stratum_name=None,
        seed_commitments=(seed_commitment,),
    )


def inject_frozen_campaigns(
    base: OperatingPopulation,
    injections: tuple[CampaignInjection, ...],
    stratum: PrevalenceStratum,
    *,
    seed: int,
) -> OperatingPopulation:
    """Replace exactly ``stratum.fraud_transaction_count`` benign decisions."""
    _require_seed(seed)
    _validate_benign_base(base)
    if type(stratum) is not PrevalenceStratum:
        raise TypeError("stratum must be an exact PrevalenceStratum")
    if len(base.observations) != stratum.transaction_count:
        raise PopulationIsolationError("stratum denominator does not match benign base")
    if not injections:
        raise PopulationIsolationError("frozen campaigns are required")

    flattened_observations = tuple(row for injection in injections for row in injection.observations)
    flattened_truth = tuple(row for injection in injections for row in injection.truth)
    _validate_injection_identity(base, injections, flattened_observations, flattened_truth)
    _validate_injection_context(base, injections, flattened_observations)
    _validate_injection_pairs(injections)
    _validate_frozen_allocation(flattened_truth, stratum)

    replacement_indices = sorted(random.Random(seed).sample(range(len(base.observations)), len(flattened_observations)))
    observations = list(base.observations)
    truth = list(base.truth)
    for index, observation, truth_row in zip(
        replacement_indices, flattened_observations, flattened_truth, strict=True
    ):
        observations[index] = observation
        truth[index] = truth_row
    seed_commitments = (*base.manifest.seed_commitments, _seed_commitment("campaign_injection", seed))
    return _population(
        observations=tuple(observations),
        truth=tuple(truth),
        population_kind="injected",
        day_count=base.manifest.day_count,
        stratum_name=stratum.name,
        seed_commitments=seed_commitments,
    )


def _population(
    *,
    observations: tuple[ObservedEvent, ...],
    truth: tuple[EvaluationTruthRow, ...],
    population_kind: Literal["benign_base", "injected"],
    day_count: int,
    stratum_name: Literal["low", "medium", "high"] | None,
    seed_commitments: tuple[SeedCommitment, ...],
) -> OperatingPopulation:
    return OperatingPopulation(
        observations=observations,
        truth=truth,
        manifest=PopulationManifest(
            population_kind=population_kind,
            transaction_count=len(observations),
            fraud_transaction_count=sum(row.is_fraud for row in truth),
            day_count=day_count,
            stratum_name=stratum_name,
            seed_commitments=seed_commitments,
            observations_sha256=_rows_digest(observations),
            truth_sha256=_rows_digest(truth),
        ),
    )


def _validate_benign_base(base: OperatingPopulation) -> None:
    if type(base) is not OperatingPopulation:
        raise TypeError("base must be an exact OperatingPopulation")
    if base.manifest.population_kind != "benign_base" or any(row.is_fraud for row in base.truth):
        raise PopulationIsolationError("non-benign base")
    if len(base.observations) != len(base.truth):
        raise PopulationIsolationError("base observations and truth have different lengths")
    if base.manifest.transaction_count != len(base.observations):
        raise PopulationIsolationError("base manifest denominator mismatch")
    _validate_row_pairs(base.observations, base.truth, label="base")
    _reject_duplicates(base.observations, base.truth, label="base")
    if any(
        row.decision_at is None or row.available_at >= row.decision_at or not row.is_decision_point
        for row in base.observations
    ):
        raise PopulationIsolationError("base decisions must be strict past-only decision rows")


def _validate_injection_identity(
    base: OperatingPopulation,
    injections: tuple[CampaignInjection, ...],
    observations: tuple[ObservedEvent, ...],
    truth: tuple[EvaluationTruthRow, ...],
) -> None:
    if any(type(injection) is not CampaignInjection for injection in injections):
        raise TypeError("campaign injections must be exact CampaignInjection values")
    _reject_duplicates(observations, truth, label="campaign")
    base_event_ids = {row.event_id for row in base.observations}
    if base_event_ids & {row.event_id for row in observations}:
        raise PopulationIsolationError("duplicate event id")
    base_payment_ids = {row.payment_id for row in base.observations}
    if base_payment_ids & {row.payment_id for row in observations}:
        raise PopulationIsolationError("duplicate payment id")


def _validate_injection_context(
    base: OperatingPopulation,
    injections: tuple[CampaignInjection, ...],
    observations: tuple[ObservedEvent, ...],
) -> None:
    base_campaigns = {row.campaign_id for row in base.truth}
    campaign_ids = tuple(injection.campaign_id for injection in injections)
    if len(campaign_ids) != len(set(campaign_ids)) or set(campaign_ids) & base_campaigns:
        raise PopulationIsolationError("campaign overlap")
    base_entities = _entities(base.observations)
    injection_entities = set().union(*(set(injection.entity_ids) for injection in injections))
    if base_entities & injection_entities:
        raise PopulationIsolationError("entity overlap")
    base_start, base_end = _decision_interval(base.observations)
    for injection in injections:
        start, end = _decision_interval(injection.observations)
        if start <= base_end and base_start <= end:
            raise PopulationIsolationError("time overlap")
    for index, left in enumerate(injections):
        left_start, left_end = _decision_interval(left.observations)
        for right in injections[index + 1 :]:
            right_start, right_end = _decision_interval(right.observations)
            if left_start <= right_end and right_start <= left_end:
                raise PopulationIsolationError("time overlap")
    if any(
        row.decision_at is None or row.available_at >= row.decision_at or not row.is_decision_point
        for row in observations
    ):
        raise PopulationIsolationError("campaign decisions must be strict past-only decision rows")


def _validate_injection_pairs(injections: tuple[CampaignInjection, ...]) -> None:
    for injection in injections:
        _validate_row_pairs(injection.observations, injection.truth, label="campaign")
        if any(not row.is_fraud for row in injection.truth):
            raise PopulationIsolationError("campaign injections must be fraud-only")
        if any(
            row.campaign_id != injection.campaign_id or row.family != injection.family
            for row in injection.truth
        ):
            raise PopulationIsolationError("campaign truth does not preserve context")
        observed_entities = _entities(injection.observations)
        if not observed_entities <= set(injection.entity_ids):
            raise PopulationIsolationError("campaign entity context is incomplete")


def _validate_frozen_allocation(
    truth: tuple[EvaluationTruthRow, ...], stratum: PrevalenceStratum
) -> None:
    if len(truth) != stratum.fraud_transaction_count:
        raise PopulationIsolationError("frozen campaign count does not match stratum")
    allocated = tuple(sum(row.family == family for row in truth) for family in _FAMILIES)
    if allocated != stratum.family_transaction_counts:
        raise PopulationIsolationError("wrong frozen family allocation")


def _validate_row_pairs(
    observations: tuple[ObservedEvent, ...], truth: tuple[EvaluationTruthRow, ...], *, label: str
) -> None:
    if len(observations) != len(truth):
        raise PopulationIsolationError(f"{label} observations and truth have different lengths")
    for observation, truth in zip(observations, truth, strict=True):
        if observation.event_id != truth.event_id or observation.payment_id != truth.payment_id:
            raise PopulationIsolationError(f"{label} observation and truth ids differ")


def _reject_duplicates(
    observations: tuple[ObservedEvent, ...], truth: tuple[EvaluationTruthRow, ...], *, label: str
) -> None:
    event_ids = tuple(row.event_id for row in observations)
    payment_ids = tuple(row.payment_id for row in observations)
    truth_event_ids = tuple(row.event_id for row in truth)
    if len(event_ids) != len(set(event_ids)) or len(truth_event_ids) != len(set(truth_event_ids)):
        raise PopulationIsolationError(f"duplicate event id in {label}")
    if len(payment_ids) != len(set(payment_ids)):
        raise PopulationIsolationError(f"duplicate payment id in {label}")


def _entities(rows: tuple[ObservedEvent, ...]) -> set[str]:
    return {
        entity
        for row in rows
        for entity in (row.actor_id, row.counterparty_id, *row.optional_refs.values())
    }


def _decision_interval(rows: tuple[ObservedEvent, ...]) -> tuple[datetime, datetime]:
    decision_times = tuple(row.decision_at for row in rows)
    if not decision_times or any(value is None for value in decision_times):
        raise PopulationIsolationError("campaign decisions must have decision timestamps")
    exact = tuple(value for value in decision_times if value is not None)
    return min(exact), max(exact)


def _seed_commitment(name: str, seed: int) -> SeedCommitment:
    return SeedCommitment(
        name=name,
        commitment_sha256=hashlib.sha256(
            canonical_json_bytes({"name": name, "seed": seed})
        ).hexdigest(),
    )


def _rows_digest(rows: tuple[ObservedEvent, ...] | tuple[EvaluationTruthRow, ...]) -> str:
    return hashlib.sha256(
        canonical_json_bytes([row.model_dump(mode="json") for row in rows])
    ).hexdigest()


def _require_seed(seed: int) -> None:
    if type(seed) is not int:
        raise TypeError("seed must be an exact integer")


__all__ = [
    "CampaignInjection",
    "OperatingPopulation",
    "PopulationIsolationError",
    "PopulationManifest",
    "build_benign_base",
    "inject_frozen_campaigns",
]
