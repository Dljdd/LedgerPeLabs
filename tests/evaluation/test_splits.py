"""Chronological evaluation split behavior."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest
from pydantic import ValidationError

from apar.contracts.events import EventKind, Rail
from apar.defense.contracts import ObservedEvent
from apar.evaluation.contracts import CorpusManifest, EvaluationTruthRow, FrozenCorpus
from apar.evaluation.splits import (
    EntityCohort,
    SplitConfig,
    make_evaluation_split,
    make_leave_one_family_out,
)

T0 = datetime(2026, 1, 1, tzinfo=UTC)


def _observation(
    event_id: str,
    payment_id: str,
    when: datetime,
    actor: str,
    counterparty: str,
) -> ObservedEvent:
    return ObservedEvent(
        event_id=event_id,
        payment_id=payment_id,
        rail=Rail.CARD,
        event_type=EventKind.AUTHORIZATION,
        amount=Decimal("10.00"),
        currency="USD",
        event_time=when,
        available_at=when,
        decision_at=when,
        actor_id=actor,
        counterparty_id=counterparty,
        integrity_status="not_applicable",
        is_decision_point=True,
    )


def _truth(
    event_id: str,
    payment_id: str,
    campaign_id: str,
    mature_at: datetime,
    *,
    family: str = "card_testing_cnp",
    fraud: bool = False,
) -> EvaluationTruthRow:
    return EvaluationTruthRow(
        event_id=event_id,
        payment_id=payment_id,
        campaign_id=campaign_id,
        family=family,
        viewpoint="development",
        is_fraud=fraud,
        label_source="population_truth",
        label_mature_at=mature_at,
        first_settlement_at=None,
        net_settled_value=Decimal("10.00"),
        lifecycle_event_ids=(event_id,),
    )


def _split_corpus() -> FrozenCorpus:
    rows = (
        ("train-boundary", "p1", "c-train", T0, "actor-old", "cp-old", T0),
        # This row follows the cutoff but remains in train because its campaign began there.
        (
            "train-late",
            "p2",
            "c-train",
            T0 + timedelta(days=2),
            "actor-x",
            "cp-x",
            T0 + timedelta(days=3),
        ),
        (
            "fit-boundary",
            "p3",
            "c-fit",
            T0 + timedelta(days=1),
            "actor-fit",
            "cp-fit",
            T0 + timedelta(days=2),
        ),
        (
            "threshold-boundary",
            "p4",
            "c-threshold",
            T0 + timedelta(days=2),
            "actor-threshold",
            "cp-threshold",
            T0 + timedelta(days=3),
        ),
        (
            "dev-returning",
            "p5",
            "c-dev",
            T0 + timedelta(days=3),
            "actor-old",
            "cp-new",
            T0 + timedelta(days=4),
        ),
        (
            "dev-warm",
            "p6",
            "c-dev",
            T0 + timedelta(days=3, hours=1),
            "actor-old",
            "cp-new",
            T0 + timedelta(days=4),
        ),
        (
            "dev-cold-actor",
            "p7",
            "c-dev",
            T0 + timedelta(days=3, hours=2),
            "actor-new",
            "cp-old",
            T0 + timedelta(days=4),
        ),
    )
    observations = tuple(
        _observation(event_id, payment_id, when, actor, counterparty)
        for event_id, payment_id, _campaign, when, actor, counterparty, _mature in rows
    )
    families = (
        "agentic_intent_abuse",
        "app_scam_mule",
        "card_testing_cnp",
        "synthetic_merchant_refund",
        "card_testing_cnp",
        "app_scam_mule",
        "synthetic_merchant_refund",
    )
    truth = tuple(
        _truth(event_id, payment_id, campaign, mature, family=family)
        for (
            event_id,
            payment_id,
            campaign,
            _when,
            _actor,
            _counterparty,
            mature,
        ), family in zip(rows, families, strict=True)
    )
    return FrozenCorpus(
        observations=observations,
        truth=truth,
        manifest=CorpusManifest(
            profile_id="split-fixture",
            run_ids=("run-1",),
            run_lineage_digests=("1" * 64,),
            observation_count=len(observations),
            truth_count=len(truth),
        ),
    )


def _config() -> SplitConfig:
    return SplitConfig(
        train_end=T0,
        calibrator_fit_end=T0 + timedelta(days=1),
        threshold_end=T0 + timedelta(days=2),
        development_end=T0 + timedelta(days=3),
    )


def test_cutoffs_are_utc_strictly_increasing_and_campaigns_must_fit() -> None:
    with pytest.raises(ValidationError, match="strictly increasing"):
        SplitConfig(
            train_end=T0,
            calibrator_fit_end=T0,
            threshold_end=T0 + timedelta(days=2),
            development_end=T0 + timedelta(days=3),
        )
    with pytest.raises(ValidationError, match="UTC"):
        SplitConfig(
            train_end=datetime(2026, 1, 1),
            calibrator_fit_end=T0 + timedelta(days=1),
            threshold_end=T0 + timedelta(days=2),
            development_end=T0 + timedelta(days=3),
        )

    corpus = _split_corpus()
    too_short = _config().model_copy(update={"development_end": T0 + timedelta(days=2, hours=23)})
    with pytest.raises(ValueError, match="beyond development_end"):
        make_evaluation_split(corpus, too_short)


def test_exact_cutoff_inclusivity_assigns_whole_campaigns() -> None:
    split = make_evaluation_split(_split_corpus(), _config())

    assert split.partition_names == (
        "train",
        "calibrator_fit",
        "threshold",
        "development",
    )
    assert split.campaigns == {
        "train": ("c-train",),
        "calibrator_fit": ("c-fit",),
        "threshold": ("c-threshold",),
        "development": ("c-dev",),
    }
    assert split.row_ids["train"] == ("train-boundary", "train-late")
    campaign_sets = [set(split.campaigns[name]) for name in split.partition_names]
    assert all(
        left.isdisjoint(right)
        for index, left in enumerate(campaign_sets)
        for right in campaign_sets[index + 1 :]
    )


def test_training_targets_require_labels_mature_by_train_end() -> None:
    split = make_evaluation_split(_split_corpus(), _config())

    assert split.row_ids["train"] == ("train-boundary", "train-late")
    assert split.training_row_ids == ("train-boundary",)


def test_entity_cohorts_are_ordered_multilabel_and_use_strictly_earlier_history() -> None:
    corpus = _split_corpus()
    equal_time = _observation(
        "dev-equal-time",
        "p8",
        T0 + timedelta(days=3),
        "actor-old",
        "cp-new",
    )
    equal_truth = _truth(
        "dev-equal-time",
        "p8",
        "c-dev-2",
        T0 + timedelta(days=4),
    )
    corpus = FrozenCorpus(
        observations=(*corpus.observations, equal_time),
        truth=(*corpus.truth, equal_truth),
        manifest=corpus.manifest.model_copy(update={"observation_count": 8, "truth_count": 8}),
    )

    split = make_evaluation_split(corpus, _config())

    assert split.entity_cohorts["dev-returning"] == (
        EntityCohort.COLD_COUNTERPARTY,
        EntityCohort.COLD_PAIR,
        EntityCohort.RETURNING_PRIOR_CAMPAIGN,
    )
    assert split.entity_cohorts["dev-equal-time"] == (
        EntityCohort.COLD_COUNTERPARTY,
        EntityCohort.COLD_PAIR,
        EntityCohort.RETURNING_PRIOR_CAMPAIGN,
    )
    assert split.entity_cohorts["dev-warm"] == (
        EntityCohort.WARM_WITHIN_CAMPAIGN,
        EntityCohort.RETURNING_PRIOR_CAMPAIGN,
    )
    assert split.entity_cohorts["dev-cold-actor"] == (
        EntityCohort.COLD_ACTOR,
        EntityCohort.COLD_PAIR,
        EntityCohort.RETURNING_PRIOR_CAMPAIGN,
    )


def test_entity_cohorts_include_strictly_earlier_nondecision_observations() -> None:
    corpus = _split_corpus()
    prior_source = ObservedEvent(
        event_id="prior-source",
        payment_id="p1",
        rail=Rail.CARD,
        event_type=EventKind.SETTLEMENT,
        amount=Decimal("10.00"),
        currency="USD",
        event_time=T0 + timedelta(hours=1),
        available_at=T0 + timedelta(hours=1),
        decision_at=None,
        actor_id="actor-source",
        counterparty_id="cp-source",
        integrity_status="not_applicable",
        is_decision_point=False,
    )
    target = _observation(
        "dev-source-return",
        "p9",
        T0 + timedelta(days=3),
        "actor-source",
        "cp-brand-new",
    )
    target_truth = _truth(
        "dev-source-return",
        "p9",
        "c-dev-source",
        T0 + timedelta(days=4),
    )
    corpus = FrozenCorpus(
        observations=(*corpus.observations, prior_source, target),
        truth=(*corpus.truth, target_truth),
        manifest=corpus.manifest.model_copy(
            update={"observation_count": 9, "truth_count": 8}
        ),
    )

    split = make_evaluation_split(corpus, _config())

    assert split.entity_cohorts["dev-source-return"] == (
        EntityCohort.COLD_COUNTERPARTY,
        EntityCohort.COLD_PAIR,
        EntityCohort.RETURNING_PRIOR_CAMPAIGN,
    )


@pytest.mark.parametrize(
    "family",
    [
        "agentic_intent_abuse",
        "app_scam_mule",
        "card_testing_cnp",
        "synthetic_merchant_refund",
    ],
)
def test_leave_one_family_out_excludes_fit_populations_and_keeps_evaluation_rows(
    family: str,
) -> None:
    split = make_evaluation_split(_split_corpus(), _config())

    held = make_leave_one_family_out(split, family)

    fit_ids = {
        event_id
        for name in ("train", "calibrator_fit", "threshold")
        for event_id in held.row_ids[name]
    }
    assert all(held.row_families[event_id] != family for event_id in fit_ids)
    assert held.row_ids["development"] == split.row_ids["development"]
    assert held.held_out_evaluation_row_ids == tuple(
        event_id
        for event_id in split.row_ids["development"]
        if split.row_families[event_id] == family
    )
    assert held.held_out_family == family


def test_held_out_family_in_split_config_applies_leave_one_family_out() -> None:
    family = "card_testing_cnp"
    config = _config().model_copy(update={"held_out_family": family})

    configured = make_evaluation_split(_split_corpus(), config)

    assert configured == make_leave_one_family_out(
        make_evaluation_split(_split_corpus(), _config()), family
    )


def test_evaluation_package_exports_split_contracts() -> None:
    from apar.evaluation import EvaluationSplit as ExportedEvaluationSplit
    from apar.evaluation import SplitConfig as ExportedSplitConfig

    assert ExportedEvaluationSplit is not None
    assert ExportedSplitConfig is SplitConfig
