"""Disjoint efficacy and operating population tests for Defend v3."""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime, timedelta

import pytest

from apar.evaluation.v2_population import (
    CampaignInjection,
    OperatingPopulation,
    build_benign_base,
    inject_frozen_campaigns,
)
from apar.evaluation.v3_population import (
    V3PopulationError,
    audit_disjointness,
    build_efficacy_population,
)
from apar.runs.wire import canonical_json_bytes
from apar.v2_protocol import PrevalenceStratum, SeedCommitment, V2Protocol

FAMILIES = (
    "agentic_intent_abuse",
    "app_scam_mule",
    "card_testing_cnp",
    "synthetic_merchant_refund",
)
CAMPAIGN_START = datetime(2026, 1, 1, 1, tzinfo=UTC)


def fixture_seed_commitment(name: str, seed: int) -> SeedCommitment:
    return SeedCommitment(
        name=name,
        commitment_sha256=hashlib.sha256(
            canonical_json_bytes({"name": name, "seed": seed})
        ).hexdigest(),
    )


def fixture_protocol(
    *,
    transaction_count: int = 100,
    day_count: int = 2,
    operating_seed: int = 11,
    injection_seed: int = 17,
) -> V2Protocol:
    protocol = V2Protocol.fixture(transaction_count=transaction_count)
    return protocol.model_copy(
        update={
            "operating": protocol.operating.model_copy(update={"day_count": day_count}),
            "seed_commitments": (
                fixture_seed_commitment("operating_population", operating_seed),
                fixture_seed_commitment("campaign_injection", injection_seed),
            ),
        }
    )


def campaign_injections(
    *, total_decisions: int, start_at: datetime = CAMPAIGN_START
) -> tuple[CampaignInjection, ...]:
    assert total_decisions % len(FAMILIES) == 0
    per_family = total_decisions // len(FAMILIES)
    return tuple(
        CampaignInjection.fixture(
            campaign_id=f"campaign-{family}",
            family=family,
            decision_count=per_family,
            entity_ids=(f"actor-{family}", f"counterparty-{family}"),
            start_at=start_at + timedelta(hours=4 * index),
        )
        for index, family in enumerate(FAMILIES)
    )


def build_operating(
    *, transaction_count: int = 100, seed: int = 11, injection_seed: int = 17
) -> OperatingPopulation:
    base = build_benign_base(fixture_protocol(transaction_count=transaction_count), seed=seed)
    fraud = transaction_count // 5
    return inject_frozen_campaigns(
        base,
        campaign_injections(total_decisions=fraud),
        PrevalenceStratum.fixture(transaction_count, fraud),
        seed=injection_seed,
    )


def test_efficacy_population_has_equal_family_representation() -> None:
    population = build_efficacy_population(
        campaign_count_per_family=2, decisions_per_campaign=3, day_count=28, seed=7
    )
    counts = {family: 0 for family in FAMILIES}
    for row in population.truth:
        counts[row.family] += 1
    assert all(value == 6 for value in counts.values())
    assert population.manifest.fraud_transaction_count == 24


def test_efficacy_population_preserves_past_only_causality() -> None:
    population = build_efficacy_population(
        campaign_count_per_family=1, decisions_per_campaign=2, day_count=1, seed=7
    )
    for row in population.observations:
        assert row.available_at < row.decision_at
        assert row.is_decision_point


def test_efficacy_population_manifest_digests_are_stable() -> None:
    first = build_efficacy_population(
        campaign_count_per_family=1, decisions_per_campaign=1, day_count=1, seed=7
    )
    second = build_efficacy_population(
        campaign_count_per_family=1, decisions_per_campaign=1, day_count=1, seed=7
    )
    assert first.manifest.observations_sha256 == second.manifest.observations_sha256
    assert first.manifest.truth_sha256 == second.manifest.truth_sha256


def test_disjointness_audit_accepts_separate_partitions() -> None:
    efficacy = build_efficacy_population(
        campaign_count_per_family=1, decisions_per_campaign=1, day_count=1, seed=7
    )
    operating = build_operating()
    proof = audit_disjointness({"efficacy": efficacy, "operating": operating})
    assert proof.entity_disjoint and proof.campaign_disjoint and proof.time_disjoint


def test_disjointness_audit_rejects_entity_overlap() -> None:
    efficacy = build_efficacy_population(
        campaign_count_per_family=1, decisions_per_campaign=1, day_count=1, seed=7
    )
    operating = build_operating()
    shared_row = operating.observations[0].model_copy(
        update={
            "actor_id": efficacy.observations[0].actor_id,
            "event_id": "shared-event",
            "payment_id": "shared-payment",
        }
    )
    tampered = OperatingPopulation(
        observations=(shared_row, *operating.observations[1:]),
        truth=operating.truth,
        manifest=operating.manifest,
    )
    with pytest.raises(V3PopulationError, match="entity overlap"):
        audit_disjointness({"efficacy": efficacy, "operating": tampered})


def test_disjointness_audit_rejects_campaign_overlap() -> None:
    efficacy = build_efficacy_population(
        campaign_count_per_family=1, decisions_per_campaign=1, day_count=1, seed=7
    )
    operating = build_operating()
    tampered_truth = operating.truth[0].model_copy(
        update={"campaign_id": efficacy.truth[0].campaign_id}
    )
    tampered = OperatingPopulation(
        observations=operating.observations,
        truth=(tampered_truth, *operating.truth[1:]),
        manifest=operating.manifest,
    )
    with pytest.raises(V3PopulationError, match="campaign overlap"):
        audit_disjointness({"efficacy": efficacy, "operating": tampered})


def test_disjointness_audit_requires_two_partitions() -> None:
    efficacy = build_efficacy_population(
        campaign_count_per_family=1, decisions_per_campaign=1, day_count=1, seed=7
    )
    with pytest.raises(V3PopulationError, match="at least two"):
        audit_disjointness({"efficacy": efficacy})
