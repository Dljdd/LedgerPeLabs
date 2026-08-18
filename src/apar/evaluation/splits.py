"""Chronological evaluator-owned partitions and cold-entity annotations."""

from __future__ import annotations

import hashlib
from collections import defaultdict
from datetime import datetime
from decimal import Decimal
from enum import StrEnum
from typing import Literal

from pydantic import model_validator

from apar.contracts._validation import ExternalContract, validate_utc_timestamp
from apar.defense.contracts import ObservedEvent
from apar.evaluation.contracts import EvaluationTruthRow, Family, FrozenCorpus
from apar.runs.wire import canonical_json_bytes

PartitionName = Literal["train", "calibrator_fit", "threshold", "development"]
_PARTITION_NAMES: tuple[PartitionName, ...] = (
    "train",
    "calibrator_fit",
    "threshold",
    "development",
)


class SplitConfig(ExternalContract):
    """Inclusive upper cutoffs for four strictly ordered chronological partitions."""

    train_end: datetime
    calibrator_fit_end: datetime
    threshold_end: datetime
    development_end: datetime
    held_out_family: Family | None = None

    @model_validator(mode="after")
    def cutoffs_are_strict_utc(self) -> SplitConfig:
        cutoffs = (
            self.train_end,
            self.calibrator_fit_end,
            self.threshold_end,
            self.development_end,
        )
        for cutoff in cutoffs:
            validate_utc_timestamp(cutoff)
        if any(left >= right for left, right in zip(cutoffs, cutoffs[1:], strict=False)):
            raise ValueError("split cutoffs must be strictly increasing")
        return self


class EntityCohort(StrEnum):
    COLD_ACTOR = "cold_actor"
    COLD_COUNTERPARTY = "cold_counterparty"
    COLD_PAIR = "cold_pair"
    WARM_WITHIN_CAMPAIGN = "warm_within_campaign"
    RETURNING_PRIOR_CAMPAIGN = "returning_prior_campaign"


class EvaluationSplit(ExternalContract):
    """A complete evaluator-side split manifest; none of its labels are model features."""

    schema_version: Literal["1.0.0"] = "1.0.0"
    config: SplitConfig
    partition_names: tuple[PartitionName, ...] = _PARTITION_NAMES
    campaigns: dict[str, tuple[str, ...]]
    row_ids: dict[str, tuple[str, ...]]
    training_row_ids: tuple[str, ...]
    entity_cohorts: dict[str, tuple[EntityCohort, ...]]
    row_families: dict[str, Family]
    row_campaigns: dict[str, str]
    row_is_fraud: dict[str, bool]
    row_net_settled_values: dict[str, Decimal]
    label_maturity_cutoff: datetime
    sample_counts: dict[str, int]
    fraud_prevalence: dict[str, Decimal]
    net_settled_value_totals: dict[str, Decimal]
    held_out_family: Family | None = None
    held_out_evaluation_row_ids: tuple[str, ...] = ()
    split_digest: str


def make_evaluation_split(corpus: FrozenCorpus, config: SplitConfig) -> EvaluationSplit:
    """Assign whole campaigns by first decision time and annotate causal entity cohorts."""
    if type(corpus) is not FrozenCorpus:
        raise TypeError("corpus must be an exact FrozenCorpus")
    if type(config) is not SplitConfig:
        raise TypeError("config must be an exact SplitConfig")

    decisions, truth_by_event = _decision_rows(corpus)
    campaign_decisions: dict[str, list[ObservedEvent]] = defaultdict(list)
    for decision in decisions:
        campaign_decisions[truth_by_event[decision.event_id].campaign_id].append(decision)

    partition_by_campaign: dict[str, PartitionName] = {}
    for campaign_id, campaign_rows in campaign_decisions.items():
        first_decision = min(_decision_time(row) for row in campaign_rows)
        partition_by_campaign[campaign_id] = _partition_for(first_decision, config)

    ordered_decisions = tuple(
        sorted(decisions, key=lambda row: (_decision_time(row), row.event_id))
    )
    campaigns: dict[str, tuple[str, ...]] = {
        name: tuple(
            campaign_id
            for campaign_id, _first in sorted(
                (
                    (campaign_id, min(_decision_time(row) for row in rows))
                    for campaign_id, rows in campaign_decisions.items()
                    if partition_by_campaign[campaign_id] == name
                ),
                key=lambda item: (item[1], item[0]),
            )
        )
        for name in _PARTITION_NAMES
    }
    row_ids: dict[str, tuple[str, ...]] = {
        name: tuple(
            row.event_id
            for row in ordered_decisions
            if partition_by_campaign[truth_by_event[row.event_id].campaign_id] == name
        )
        for name in _PARTITION_NAMES
    }
    row_families = {event_id: truth_by_event[event_id].family for event_id in truth_by_event}
    row_campaigns = {event_id: truth_by_event[event_id].campaign_id for event_id in truth_by_event}
    row_is_fraud = {event_id: truth_by_event[event_id].is_fraud for event_id in truth_by_event}
    row_values = {
        event_id: truth_by_event[event_id].net_settled_value for event_id in truth_by_event
    }
    training_row_ids = tuple(
        event_id
        for event_id in row_ids["train"]
        if truth_by_event[event_id].label_mature_at <= config.train_end
    )
    cohorts = _entity_cohorts(corpus.observations, ordered_decisions, truth_by_event)
    held_rows = (
        tuple(
            event_id
            for event_id in row_ids["development"]
            if row_families[event_id] == config.held_out_family
        )
        if config.held_out_family is not None
        else ()
    )
    split = _build_split(
        config=config,
        campaigns=campaigns,
        row_ids=row_ids,
        training_row_ids=training_row_ids,
        entity_cohorts=cohorts,
        row_families=row_families,
        row_campaigns=row_campaigns,
        row_is_fraud=row_is_fraud,
        row_net_settled_values=row_values,
        held_out_evaluation_row_ids=held_rows,
    )
    if config.held_out_family is not None:
        return make_leave_one_family_out(split, config.held_out_family)
    return split


def make_leave_one_family_out(split: EvaluationSplit, family: Family) -> EvaluationSplit:
    """Exclude one family from all fitting populations and expose its untouched test rows."""
    if type(split) is not EvaluationSplit:
        raise TypeError("split must be an exact EvaluationSplit")
    if family not in {
        "agentic_intent_abuse",
        "app_scam_mule",
        "card_testing_cnp",
        "synthetic_merchant_refund",
    }:
        raise ValueError("held-out family is not declared")

    filtered_rows = dict(split.row_ids)
    for name in ("train", "calibrator_fit", "threshold"):
        filtered_rows[name] = tuple(
            event_id for event_id in split.row_ids[name] if split.row_families[event_id] != family
        )
    filtered_campaigns = dict(split.campaigns)
    for name in ("train", "calibrator_fit", "threshold"):
        retained = {split.row_campaigns[event_id] for event_id in filtered_rows[name]}
        filtered_campaigns[name] = tuple(
            campaign_id for campaign_id in split.campaigns[name] if campaign_id in retained
        )
    config = split.config.model_copy(update={"held_out_family": family})
    return _build_split(
        config=config,
        campaigns=filtered_campaigns,
        row_ids=filtered_rows,
        training_row_ids=tuple(
            event_id
            for event_id in split.training_row_ids
            if split.row_families[event_id] != family
        ),
        entity_cohorts=split.entity_cohorts,
        row_families=split.row_families,
        row_campaigns=split.row_campaigns,
        row_is_fraud=split.row_is_fraud,
        row_net_settled_values=split.row_net_settled_values,
        held_out_evaluation_row_ids=tuple(
            event_id
            for event_id in split.row_ids["development"]
            if split.row_families[event_id] == family
        ),
    )


def _decision_rows(
    corpus: FrozenCorpus,
) -> tuple[tuple[ObservedEvent, ...], dict[str, EvaluationTruthRow]]:
    truth_by_event: dict[str, EvaluationTruthRow] = {}
    for truth in corpus.truth:
        if truth.event_id in truth_by_event:
            raise ValueError("corpus contains duplicate truth event IDs")
        truth_by_event[truth.event_id] = truth
    decisions = tuple(row for row in corpus.observations if row.is_decision_point)
    decision_ids = {row.event_id for row in decisions}
    if len(decision_ids) != len(decisions):
        raise ValueError("corpus contains duplicate observation event IDs")
    if decision_ids != set(truth_by_event):
        raise ValueError("decision observations and evaluator truth must correspond exactly")
    return decisions, truth_by_event


def _decision_time(row: ObservedEvent) -> datetime:
    if row.decision_at is None:
        raise ValueError("decision observation has no decision_at")
    return row.decision_at


def _partition_for(decision_at: datetime, config: SplitConfig) -> PartitionName:
    if decision_at <= config.train_end:
        return "train"
    if decision_at <= config.calibrator_fit_end:
        return "calibrator_fit"
    if decision_at <= config.threshold_end:
        return "threshold"
    if decision_at <= config.development_end:
        return "development"
    raise ValueError("campaign begins beyond development_end")


def _entity_cohorts(
    observations: tuple[ObservedEvent, ...],
    decisions: tuple[ObservedEvent, ...],
    truth_by_event: dict[str, EvaluationTruthRow],
) -> dict[str, tuple[EntityCohort, ...]]:
    truth_by_payment: dict[str, EvaluationTruthRow] = {}
    for truth in truth_by_event.values():
        if truth.payment_id in truth_by_payment:
            raise ValueError("corpus truth contains duplicate payment IDs")
        truth_by_payment[truth.payment_id] = truth
    for observation in observations:
        if observation.payment_id not in truth_by_payment:
            raise ValueError("observation does not resolve to evaluator truth")
    history = tuple(
        sorted(observations, key=lambda row: (_observation_time(row), row.event_id))
    )
    actor_campaigns: dict[str, set[str]] = defaultdict(set)
    counterparty_campaigns: dict[str, set[str]] = defaultdict(set)
    pair_campaigns: dict[tuple[str, str], set[str]] = defaultdict(set)
    result: dict[str, tuple[EntityCohort, ...]] = {}
    index = 0
    history_index = 0
    while index < len(decisions):
        decision_at = _decision_time(decisions[index])
        while (
            history_index < len(history)
            and _observation_time(history[history_index]) < decision_at
        ):
            observed = history[history_index]
            campaign = truth_by_payment[observed.payment_id].campaign_id
            actor_campaigns[observed.actor_id].add(campaign)
            counterparty_campaigns[observed.counterparty_id].add(campaign)
            pair_campaigns[(observed.actor_id, observed.counterparty_id)].add(campaign)
            history_index += 1
        end = index
        while end < len(decisions) and _decision_time(decisions[end]) == decision_at:
            end += 1
        cohort = decisions[index:end]
        for row in cohort:
            campaign = truth_by_event[row.event_id].campaign_id
            pair = (row.actor_id, row.counterparty_id)
            labels: list[EntityCohort] = []
            if not actor_campaigns[row.actor_id]:
                labels.append(EntityCohort.COLD_ACTOR)
            if not counterparty_campaigns[row.counterparty_id]:
                labels.append(EntityCohort.COLD_COUNTERPARTY)
            if not pair_campaigns[pair]:
                labels.append(EntityCohort.COLD_PAIR)
            if (
                campaign in actor_campaigns[row.actor_id]
                or campaign in counterparty_campaigns[row.counterparty_id]
                or campaign in pair_campaigns[pair]
            ):
                labels.append(EntityCohort.WARM_WITHIN_CAMPAIGN)
            if (
                actor_campaigns[row.actor_id] - {campaign}
                or counterparty_campaigns[row.counterparty_id] - {campaign}
                or pair_campaigns[pair] - {campaign}
            ):
                labels.append(EntityCohort.RETURNING_PRIOR_CAMPAIGN)
            result[row.event_id] = tuple(labels)
        index = end
    return result


def _observation_time(row: ObservedEvent) -> datetime:
    return _decision_time(row) if row.is_decision_point else row.available_at


def _build_split(
    *,
    config: SplitConfig,
    campaigns: dict[str, tuple[str, ...]],
    row_ids: dict[str, tuple[str, ...]],
    training_row_ids: tuple[str, ...],
    entity_cohorts: dict[str, tuple[EntityCohort, ...]],
    row_families: dict[str, Family],
    row_campaigns: dict[str, str],
    row_is_fraud: dict[str, bool],
    row_net_settled_values: dict[str, Decimal],
    held_out_evaluation_row_ids: tuple[str, ...],
) -> EvaluationSplit:
    sample_counts: dict[str, int] = {name: len(row_ids[name]) for name in _PARTITION_NAMES}
    prevalence: dict[str, Decimal] = {
        name: (
            Decimal(sum(row_is_fraud[event_id] for event_id in row_ids[name]))
            / Decimal(len(row_ids[name]))
            if row_ids[name]
            else Decimal("0")
        )
        for name in _PARTITION_NAMES
    }
    values: dict[str, Decimal] = {
        name: sum((row_net_settled_values[event_id] for event_id in row_ids[name]), Decimal("0"))
        for name in _PARTITION_NAMES
    }
    document = {
        "config": config.model_dump(mode="json"),
        "partition_names": _PARTITION_NAMES,
        "campaigns": campaigns,
        "row_ids": row_ids,
        "training_row_ids": training_row_ids,
        "entity_cohorts": {
            event_id: tuple(label.value for label in labels)
            for event_id, labels in entity_cohorts.items()
        },
        "row_families": row_families,
        "row_campaigns": row_campaigns,
        "row_is_fraud": {
            event_id: row_is_fraud[event_id] for event_id in sorted(row_is_fraud)
        },
        "row_net_settled_values": {
            event_id: _canonical_decimal(row_net_settled_values[event_id])
            for event_id in sorted(row_net_settled_values)
        },
        "label_maturity_cutoff": config.train_end.isoformat(),
        "sample_counts": sample_counts,
        "fraud_prevalence": {
            name: _canonical_decimal(value) for name, value in prevalence.items()
        },
        "net_settled_value_totals": {
            name: _canonical_decimal(value) for name, value in values.items()
        },
        "held_out_evaluation_row_ids": held_out_evaluation_row_ids,
    }
    digest = hashlib.sha256(canonical_json_bytes(_json_tree(document))).hexdigest()
    return EvaluationSplit(
        config=config,
        campaigns=campaigns,
        row_ids=row_ids,
        training_row_ids=training_row_ids,
        entity_cohorts=entity_cohorts,
        row_families=row_families,
        row_campaigns=row_campaigns,
        row_is_fraud=row_is_fraud,
        row_net_settled_values=row_net_settled_values,
        label_maturity_cutoff=config.train_end,
        sample_counts=sample_counts,
        fraud_prevalence=prevalence,
        net_settled_value_totals=values,
        held_out_family=config.held_out_family,
        held_out_evaluation_row_ids=held_out_evaluation_row_ids,
        split_digest=digest,
    )


def _json_tree(value: object) -> object:
    if type(value) is tuple:
        return [_json_tree(item) for item in value]
    if type(value) is list:
        return [_json_tree(item) for item in value]
    if type(value) is dict:
        return {str(key): _json_tree(item) for key, item in value.items()}
    return value


def _canonical_decimal(value: Decimal) -> str:
    if not value.is_finite():
        raise ValueError("split decimal values must be finite")
    if value == 0:
        return "0"
    text = format(value, "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text


__all__ = [
    "EntityCohort",
    "EvaluationSplit",
    "SplitConfig",
    "make_evaluation_split",
    "make_leave_one_family_out",
]
