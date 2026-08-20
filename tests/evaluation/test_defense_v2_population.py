"""Fixture-only contracts for independently constructed Defend v2 populations."""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime, timedelta

import pytest

from apar.evaluation.v2_population import (
    CampaignInjection,
    PopulationIsolationError,
    build_benign_base,
    inject_frozen_campaigns,
)
from apar.evaluation.v2_protocol import PrevalenceStratum, SeedCommitment, V2Protocol
from apar.runs.wire import canonical_json_bytes


FAMILIES = (
    "agentic_intent_abuse",
    "app_scam_mule",
    "card_testing_cnp",
    "synthetic_merchant_refund",
)
CAMPAIGN_START = datetime(2026, 2, 1, tzinfo=UTC)


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


def campaign_injections(*, total_decisions: int, start_at: datetime = CAMPAIGN_START) -> tuple[CampaignInjection, ...]:
    assert total_decisions % len(FAMILIES) == 0
    per_family = total_decisions // len(FAMILIES)
    return tuple(
        CampaignInjection.fixture(
            campaign_id=f"campaign-{family}",
            family=family,
            decision_count=per_family,
            entity_ids=(f"actor-{family}", f"counterparty-{family}"),
            start_at=start_at + timedelta(days=index),
        )
        for index, family in enumerate(FAMILIES)
    )


def test_injection_keeps_exact_denominator() -> None:
    base = build_benign_base(fixture_protocol(transaction_count=100), seed=11)

    result = inject_frozen_campaigns(
        base,
        campaign_injections(total_decisions=20),
        PrevalenceStratum.fixture(100, 20),
        seed=17,
    )

    assert len(result.observations) == 100
    assert len(result.truth) == 100
    assert sum(row.is_fraud for row in result.truth) == 20


def test_entity_overlap_is_rejected() -> None:
    base = build_benign_base(fixture_protocol(transaction_count=100), seed=11)
    injection = CampaignInjection.fixture(entity_ids=(base.observations[0].actor_id,))

    with pytest.raises(PopulationIsolationError, match="entity overlap"):
        inject_frozen_campaigns(base, (injection,), PrevalenceStratum.fixture(), seed=17)


def test_entity_overlap_between_injected_campaigns_is_rejected() -> None:
    base = build_benign_base(fixture_protocol(transaction_count=100), seed=11)
    shared_entity = "injected-shared-entity"
    first = CampaignInjection.fixture(
        campaign_id="campaign-first",
        entity_ids=(shared_entity, "injected-counterparty-one"),
        start_at=CAMPAIGN_START,
    )
    second = CampaignInjection.fixture(
        campaign_id="campaign-second",
        entity_ids=(shared_entity, "injected-counterparty-two"),
        start_at=CAMPAIGN_START + timedelta(days=1),
    )

    with pytest.raises(PopulationIsolationError, match="entity overlap"):
        inject_frozen_campaigns(base, (first, second), PrevalenceStratum.fixture(), seed=17)


def test_operating_seed_must_match_the_protocol_commitment() -> None:
    protocol = fixture_protocol(transaction_count=100, operating_seed=12)

    with pytest.raises(PopulationIsolationError, match="operating_population seed commitment"):
        build_benign_base(protocol, seed=11)


def test_injection_seed_must_match_the_protocol_commitment() -> None:
    base = build_benign_base(fixture_protocol(transaction_count=100), seed=11)

    with pytest.raises(PopulationIsolationError, match="campaign_injection seed commitment"):
        inject_frozen_campaigns(
            base,
            campaign_injections(total_decisions=20),
            PrevalenceStratum.fixture(),
            seed=18,
        )


@pytest.mark.parametrize(
    ("stratum", "fraud_count"),
    (
        ("low", 4),
        ("medium", 8),
        ("high", 12),
    ),
)
def test_each_fixture_stratum_has_equal_fraud_family_allocation(
    stratum: str, fraud_count: int
) -> None:
    protocol = fixture_protocol(transaction_count=100)
    selected = next(item for item in protocol.strata if item.name == stratum)
    assert selected.fraud_transaction_count == fraud_count

    result = inject_frozen_campaigns(
        build_benign_base(protocol, seed=11),
        campaign_injections(total_decisions=fraud_count),
        selected,
        seed=17,
    )

    by_family = {
        family: sum(row.is_fraud and row.family == family for row in result.truth)
        for family in FAMILIES
    }
    assert by_family == dict(zip(FAMILIES, selected.family_transaction_counts, strict=True))


def test_benign_base_spans_exactly_28_synthetic_days() -> None:
    base = build_benign_base(fixture_protocol(transaction_count=280, day_count=28), seed=11)

    assert {row.decision_at.date() for row in base.observations if row.decision_at} == {
        (datetime(2026, 1, 1, tzinfo=UTC) + timedelta(days=offset)).date()
        for offset in range(28)
    }
    assert all(row.available_at < row.decision_at for row in base.observations if row.decision_at)


def test_population_construction_is_deterministic_without_exposing_seeds() -> None:
    protocol = fixture_protocol(transaction_count=100)
    first_base = build_benign_base(protocol, seed=11)
    second_base = build_benign_base(protocol, seed=11)
    injections = campaign_injections(total_decisions=20)

    first = inject_frozen_campaigns(first_base, injections, PrevalenceStratum.fixture(), seed=17)
    second = inject_frozen_campaigns(second_base, injections, PrevalenceStratum.fixture(), seed=17)

    assert first == second
    assert "seed" not in first.manifest.model_dump(mode="json")
    assert {item.name for item in first.manifest.seed_commitments} == {
        "campaign_injection",
        "operating_population",
    }


def test_injected_campaigns_are_disjoint_from_the_benign_base() -> None:
    base = build_benign_base(fixture_protocol(transaction_count=100), seed=11)
    injections = campaign_injections(total_decisions=20)
    result = inject_frozen_campaigns(base, injections, PrevalenceStratum.fixture(), seed=17)

    base_entities = {
        entity
        for row in base.observations
        for entity in (row.actor_id, row.counterparty_id, *row.optional_refs.values())
    }
    injected_rows = tuple(row for row, truth in zip(result.observations, result.truth, strict=True) if truth.is_fraud)
    assert base_entities.isdisjoint(
        {
            entity
            for row in injected_rows
            for entity in (row.actor_id, row.counterparty_id, *row.optional_refs.values())
        }
    )
    assert max(row.decision_at for row in base.observations if row.decision_at) < min(
        row.decision_at for row in injected_rows if row.decision_at
    )


def test_non_benign_base_is_rejected() -> None:
    base = build_benign_base(fixture_protocol(transaction_count=100), seed=11)
    non_benign = base.__class__(
        observations=base.observations,
        truth=(base.truth[0].model_copy(update={"is_fraud": True}), *base.truth[1:]),
        manifest=base.manifest,
    )

    with pytest.raises(PopulationIsolationError, match="non-benign base"):
        inject_frozen_campaigns(
            non_benign, campaign_injections(total_decisions=20), PrevalenceStratum.fixture(), seed=17
        )


def test_wrong_frozen_family_allocation_is_rejected() -> None:
    base = build_benign_base(fixture_protocol(transaction_count=100), seed=11)
    injections = campaign_injections(total_decisions=20)
    displaced = injections[-1]
    wrong_family = "card_testing_cnp"
    malformed = displaced.model_copy(
        update={
            "family": wrong_family,
            "truth": tuple(row.model_copy(update={"family": wrong_family}) for row in displaced.truth),
        }
    )

    with pytest.raises(PopulationIsolationError, match="family allocation"):
        inject_frozen_campaigns(
            base,
            (*injections[:-1], malformed),
            PrevalenceStratum.fixture(100, 20),
            seed=17,
        )


def test_campaign_and_duplicate_id_overlap_are_rejected() -> None:
    base = build_benign_base(fixture_protocol(transaction_count=100), seed=11)
    duplicate_campaign = CampaignInjection.fixture(campaign_id=base.truth[0].campaign_id)

    with pytest.raises(PopulationIsolationError, match="campaign overlap"):
        inject_frozen_campaigns(base, (duplicate_campaign,), PrevalenceStratum.fixture(), seed=17)

    injection = CampaignInjection.fixture(campaign_id="campaign-unique", decision_count=20)
    duplicate_event = injection.observations[0].model_copy(
        update={"event_id": base.observations[0].event_id}
    )
    duplicated = injection.model_copy(update={"observations": (duplicate_event, *injection.observations[1:])})
    with pytest.raises(PopulationIsolationError, match="duplicate event id"):
        inject_frozen_campaigns(base, (duplicated,), PrevalenceStratum.fixture(), seed=17)
