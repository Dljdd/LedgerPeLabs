"""Deterministic derived-regime behavior."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest
from pydantic import ValidationError

from apar.contracts.events import EventKind, Rail
from apar.defense.contracts import ObservedEvent
from apar.evaluation.contracts import CorpusManifest, EvaluationTruthRow, FrozenCorpus
from apar.evaluation.regimes import RegimeKind, RegimeSpec, derive_regime, frozen_corpus_digest

T0 = datetime(2026, 1, 1, 12, tzinfo=UTC)


def _observation(
    event_id: str,
    payment_id: str,
    when: datetime,
    *,
    amount: str,
    actor: str,
    counterparty: str,
    decision: bool,
    optional_refs: dict[str, str] | None = None,
) -> ObservedEvent:
    available = when + timedelta(seconds=4)
    return ObservedEvent(
        event_id=event_id,
        payment_id=payment_id,
        rail=Rail.CARD,
        event_type=EventKind.AUTHORIZATION if decision else EventKind.SETTLEMENT,
        amount=Decimal(amount),
        currency="USD",
        event_time=when,
        available_at=available,
        decision_at=available + timedelta(seconds=2) if decision else None,
        actor_id=actor,
        counterparty_id=counterparty,
        optional_refs=optional_refs or {},
        integrity_status="not_applicable",
        is_decision_point=decision,
    )


def _truth(
    event_id: str,
    payment_id: str,
    campaign_id: str,
    *,
    fraud: bool,
    settlement_id: str,
    net: str,
) -> EvaluationTruthRow:
    return EvaluationTruthRow(
        event_id=event_id,
        payment_id=payment_id,
        campaign_id=campaign_id,
        family="card_testing_cnp",
        viewpoint="development",
        is_fraud=fraud,
        label_source="population_truth",
        label_mature_at=T0 + timedelta(days=7, microseconds=3),
        first_settlement_at=T0 + timedelta(seconds=10, microseconds=2),
        net_settled_value=Decimal(net),
        lifecycle_event_ids=(event_id, settlement_id),
    )


def _corpus() -> FrozenCorpus:
    observations = (
        _observation(
            "benign-open",
            "benign-pay",
            T0,
            amount="10.00",
            actor="actor-a",
            counterparty="cp-a",
            decision=True,
            optional_refs={"device_id": "shared-id", "merchant_id": "merchant-a"},
        ),
        _observation(
            "benign-settle",
            "benign-pay",
            T0 + timedelta(seconds=10, microseconds=2),
            amount="8.00",
            actor="actor-a",
            counterparty="cp-a",
            decision=False,
            optional_refs={"device_id": "shared-id"},
        ),
        _observation(
            "fraud-open",
            "fraud-pay",
            T0 + timedelta(seconds=20),
            amount="20.00",
            actor="shared-id",
            counterparty="cp-f",
            decision=True,
        ),
        _observation(
            "fraud-settle",
            "fraud-pay",
            T0 + timedelta(seconds=30),
            amount="20.00",
            actor="shared-id",
            counterparty="cp-f",
            decision=False,
        ),
    )
    truth = (
        _truth(
            "benign-open",
            "benign-pay",
            "campaign-base",
            fraud=False,
            settlement_id="benign-settle",
            net="8.00",
        ),
        _truth(
            "fraud-open",
            "fraud-pay",
            "campaign-base",
            fraud=True,
            settlement_id="fraud-settle",
            net="20.00",
        ),
    )
    return FrozenCorpus(
        observations=observations,
        truth=truth,
        manifest=CorpusManifest(
            profile_id="regime-fixture",
            run_ids=("run-base",),
            run_lineage_digests=("a" * 64,),
            observation_count=4,
            truth_count=2,
        ),
    )


def _control_corpus(*, fraud: bool = False, event_id: str = "control-open") -> FrozenCorpus:
    observations = (
        _observation(
            event_id,
            "control-pay",
            T0 + timedelta(days=20),
            amount="5.00",
            actor="actor-control",
            counterparty="cp-control",
            decision=True,
        ),
        _observation(
            "control-settle",
            "control-pay",
            T0 + timedelta(days=20, seconds=5),
            amount="5.00",
            actor="actor-control",
            counterparty="cp-control",
            decision=False,
        ),
    )
    truth = (
        EvaluationTruthRow(
            event_id=event_id,
            payment_id="control-pay",
            campaign_id="campaign-control",
            family="card_testing_cnp",
            viewpoint="development",
            is_fraud=fraud,
            label_source="population_truth",
            label_mature_at=T0 + timedelta(days=27),
            first_settlement_at=T0 + timedelta(days=20, seconds=5),
            net_settled_value=Decimal("5.00"),
            lifecycle_event_ids=(event_id, "control-settle"),
        ),
    )
    return FrozenCorpus(
        observations=observations,
        truth=truth,
        manifest=CorpusManifest(
            profile_id="control-fixture",
            run_ids=("run-control",),
            run_lineage_digests=("b" * 64,),
            observation_count=2,
            truth_count=1,
        ),
    )


def _truth_bytes(corpus: FrozenCorpus) -> bytes:
    return json.dumps(
        [row.model_dump(mode="json") for row in corpus.truth],
        sort_keys=True,
        separators=(",", ":"),
    ).encode()


def test_specs_have_closed_kind_specific_defaults_and_validation() -> None:
    assert RegimeSpec.availability_delay().delay_seconds == 300
    compressed = RegimeSpec.compressed_bursts()
    assert (compressed.compression_numerator, compressed.compression_denominator) == (1, 4)
    assert RegimeSpec.benign_amount_shift().scale == Decimal("1.25")
    assert RegimeSpec.cold_id_remap().salt == "defense-v1-cold-remap"

    with pytest.raises(ValidationError, match="sorted"):
        RegimeSpec.prevalence_dilution(("z", "a"))
    with pytest.raises(ValidationError, match="exact tuple"):
        RegimeSpec(kind=RegimeKind.PREVALENCE_DILUTION, control_campaign_ids=["a"])
    with pytest.raises(ValidationError, match="positive"):
        RegimeSpec.availability_delay(0)
    with pytest.raises(ValidationError, match="less than"):
        RegimeSpec.compressed_bursts(4, 4)
    with pytest.raises(ValidationError, match="greater than zero"):
        RegimeSpec.benign_amount_shift(Decimal("0"))
    with pytest.raises(ValidationError, match="irrelevant"):
        RegimeSpec(kind=RegimeKind.MISSING_OPTIONAL, delay_seconds=20)


def test_missing_optional_changes_only_observation_refs_and_not_truth() -> None:
    corpus = _corpus()

    changed, manifest = derive_regime(corpus, RegimeSpec.missing_optional())

    assert changed.truth == corpus.truth
    assert _truth_bytes(changed) == _truth_bytes(corpus)
    assert all(not observation.optional_refs for observation in changed.observations)
    assert manifest.parent_corpus_digest == frozen_corpus_digest(corpus)
    assert manifest.truth_bytes_unchanged is True


def test_availability_delay_moves_only_nondecision_availability() -> None:
    corpus = _corpus()

    changed, manifest = derive_regime(corpus, RegimeSpec.availability_delay(17))

    by_id = {event.event_id: event for event in changed.observations}
    original = {event.event_id: event for event in corpus.observations}
    assert by_id["benign-open"] == original["benign-open"]
    assert by_id["fraud-open"] == original["fraud-open"]
    assert by_id["benign-settle"].available_at == (
        original["benign-settle"].available_at + timedelta(seconds=17)
    )
    assert by_id["benign-settle"].event_time == original["benign-settle"].event_time
    assert changed.truth == corpus.truth
    assert manifest.truth_bytes_unchanged is True


def test_compressed_bursts_transform_related_times_with_half_even_microseconds() -> None:
    corpus = _corpus()

    changed, manifest = derive_regime(corpus, RegimeSpec.compressed_bursts(1, 4))

    by_id = {event.event_id: event for event in changed.observations}
    assert by_id["benign-open"].event_time == T0
    # 10,000,002 microseconds / 4 = 2,500,000.5, rounded half-even downward.
    assert by_id["benign-settle"].event_time == T0 + timedelta(microseconds=2_500_000)
    assert by_id["benign-settle"].available_at == T0 + timedelta(microseconds=3_500_000)
    assert by_id["fraud-open"].event_time == T0 + timedelta(seconds=5)
    assert changed.truth[0].first_settlement_at == T0 + timedelta(microseconds=2_500_000)
    assert changed.truth[0].is_fraud == corpus.truth[0].is_fraud
    assert changed.truth[0].label_source == corpus.truth[0].label_source
    assert changed.truth[0].net_settled_value == corpus.truth[0].net_settled_value
    assert manifest.truth_bytes_unchanged is False


def test_benign_amount_shift_preserves_fraud_and_scales_all_benign_economics() -> None:
    corpus = _corpus()

    changed, manifest = derive_regime(corpus, RegimeSpec.benign_amount_shift(Decimal("1.255")))

    by_id = {event.event_id: event for event in changed.observations}
    original = {event.event_id: event for event in corpus.observations}
    assert by_id["benign-open"].amount == Decimal("12.55")
    assert by_id["benign-settle"].amount == Decimal("10.04")
    assert changed.truth[0].net_settled_value == Decimal("10.04")
    assert by_id["fraud-open"] == original["fraud-open"]
    assert by_id["fraud-settle"] == original["fraud-settle"]
    assert changed.truth[1] == corpus.truth[1]
    assert manifest.truth_bytes_unchanged is False


def test_cold_id_remap_is_bijective_and_preserves_cross_field_graph_structure() -> None:
    corpus = _corpus()

    changed, manifest = derive_regime(corpus, RegimeSpec.cold_id_remap())
    repeated, repeated_manifest = derive_regime(corpus, RegimeSpec.cold_id_remap())

    by_id = {event.event_id: event for event in changed.observations}
    assert by_id["benign-open"].optional_refs["device_id"] == by_id["fraud-open"].actor_id
    originals = {
        value
        for event in corpus.observations
        for value in (
            event.actor_id,
            event.counterparty_id,
            *event.optional_refs.values(),
        )
    }
    remapped = {
        value
        for event in changed.observations
        for value in (
            event.actor_id,
            event.counterparty_id,
            *event.optional_refs.values(),
        )
    }
    assert len(remapped) == len(originals)
    assert remapped.isdisjoint(originals)
    assert changed.truth == corpus.truth
    assert changed == repeated
    assert manifest == repeated_manifest
    assert manifest.truth_bytes_unchanged is True


def test_prevalence_dilution_requires_separate_all_benign_collision_free_controls() -> None:
    corpus = _corpus()
    spec = RegimeSpec.prevalence_dilution(("campaign-control",))

    with pytest.raises(ValueError, match="control_corpus"):
        derive_regime(corpus, spec)
    with pytest.raises(ValueError, match="benign"):
        derive_regime(corpus, spec, control_corpus=_control_corpus(fraud=True))
    with pytest.raises(ValueError, match="event ID collision"):
        derive_regime(
            corpus,
            spec,
            control_corpus=_control_corpus(event_id="benign-open"),
        )

    changed, manifest = derive_regime(corpus, spec, control_corpus=_control_corpus())

    assert tuple(row.event_id for row in changed.truth) == (
        "benign-open",
        "control-open",
        "fraud-open",
    )
    assert {event.event_id for event in changed.observations} == {
        "benign-open",
        "benign-settle",
        "control-open",
        "control-settle",
        "fraud-open",
        "fraud-settle",
    }
    assert changed.manifest.observation_count == 6
    assert changed.manifest.truth_count == 3
    assert manifest.truth_bytes_unchanged is False


def test_regime_digests_are_exact_and_parameters_are_canonical() -> None:
    corpus = _corpus()
    changed, manifest = derive_regime(corpus, RegimeSpec.availability_delay(17))

    assert frozen_corpus_digest(corpus) == (
        "e61548a363a3f02ffdd11d90b0c6a031b4f70e2c9a4387397608cebf7e035079"
    )
    assert manifest.parameters == {"delay_seconds": 17}
    assert manifest.output_corpus_digest == frozen_corpus_digest(changed)
    assert manifest.transformer == "availability_delay"
    assert manifest.transformer_version == "1.0.0"


def test_frozen_corpus_digest_covers_observations_truth_and_manifest() -> None:
    corpus = _corpus()
    digest = frozen_corpus_digest(corpus)
    changed_observation = FrozenCorpus(
        observations=(
            corpus.observations[0].model_copy(update={"amount": Decimal("10.01")}),
            *corpus.observations[1:],
        ),
        truth=corpus.truth,
        manifest=corpus.manifest,
    )
    changed_truth = FrozenCorpus(
        observations=corpus.observations,
        truth=(
            corpus.truth[0].model_copy(update={"net_settled_value": Decimal("8.01")}),
            corpus.truth[1],
        ),
        manifest=corpus.manifest,
    )
    changed_manifest = FrozenCorpus(
        observations=corpus.observations,
        truth=corpus.truth,
        manifest=corpus.manifest.model_copy(update={"profile_id": "changed"}),
    )

    assert digest != frozen_corpus_digest(changed_observation)
    assert digest != frozen_corpus_digest(changed_truth)
    assert digest != frozen_corpus_digest(changed_manifest)
    assert len(digest) == hashlib.sha256().digest_size * 2


def test_evaluation_package_exports_regime_contracts() -> None:
    from apar.evaluation import DerivedRegimeManifest as ExportedManifest
    from apar.evaluation import RegimeSpec as ExportedRegimeSpec

    assert ExportedManifest is not None
    assert ExportedRegimeSpec is RegimeSpec
