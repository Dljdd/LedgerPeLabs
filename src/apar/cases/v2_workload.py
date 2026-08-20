"""Synthetic Defend v2 case grouping and action-specific workload metrics."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta

from pydantic import Field, field_validator, model_validator

from apar.contracts._validation import ExternalContract, validate_utc_timestamp
from apar.contracts.decisions import Action
from apar.defense.contracts import ObservedEvent
from apar.defense.policy import DefenseDecision
from apar.evaluation.contracts import EvaluationTruthRow

_REVIEW_WINDOW = timedelta(hours=24)


class ReviewCase(ExternalContract):
    """A frozen actor/day review unit built solely from visible observations."""

    entity_key: str = Field(min_length=1)
    window_start: datetime
    event_ids: tuple[str, ...] = Field(min_length=1)
    integrity_failure_event_ids: tuple[str, ...] = ()

    @field_validator("window_start")
    @classmethod
    def window_start_is_utc_midnight(cls, value: datetime) -> datetime:
        value = validate_utc_timestamp(value)
        if value.time() != datetime.min.time():
            raise ValueError("review-case window_start must be UTC midnight")
        return value

    @field_validator("event_ids", "integrity_failure_event_ids")
    @classmethod
    def event_ids_are_canonical(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if any(type(event_id) is not str or not event_id for event_id in value):
            raise ValueError("review-case event IDs must be non-empty strings")
        if value != tuple(sorted(set(value))):
            raise ValueError("review-case event IDs must be sorted and unique")
        return value

    @model_validator(mode="after")
    def integrity_failures_are_case_events(self) -> ReviewCase:
        if not set(self.integrity_failure_event_ids).issubset(self.event_ids):
            raise ValueError("integrity failure event IDs must belong to the review case")
        return self


class ActionWorkload(ExternalContract):
    """Explicit case and transaction workload denominators for one action vector."""

    total_transaction_count: int = Field(gt=0)
    legitimate_transaction_count: int = Field(ge=0)
    review_case_count: int = Field(ge=0)
    reviewed_transaction_count: int = Field(ge=0)
    challenge_count: int = Field(ge=0)
    automatic_integrity_decline_count: int = Field(ge=0)
    false_decline_count: int = Field(ge=0)
    false_intervention_count: int = Field(ge=0)
    review_case_rate: float = Field(ge=0.0)
    challenge_rate: float = Field(ge=0.0)
    false_decline_rate: float | None = Field(default=None, ge=0.0)
    false_interventions_per_10k: float = Field(ge=0.0)

    @model_validator(mode="after")
    def counts_and_denominators_are_consistent(self) -> ActionWorkload:
        if self.legitimate_transaction_count > self.total_transaction_count:
            raise ValueError("legitimate transactions cannot exceed total transactions")
        if self.reviewed_transaction_count > self.challenge_count:
            raise ValueError("reviewed transactions cannot exceed challenged transactions")
        if self.false_decline_count > self.legitimate_transaction_count:
            raise ValueError("false declines cannot exceed legitimate transactions")
        if self.false_intervention_count < self.false_decline_count:
            raise ValueError("false interventions must include false declines")
        if self.legitimate_transaction_count == 0 and self.false_decline_rate is not None:
            raise ValueError("false decline rate is undefined without legitimate transactions")
        return self


def group_review_cases(
    events: Sequence[ObservedEvent], *, window: timedelta = _REVIEW_WINDOW
) -> tuple[ReviewCase, ...]:
    """Group synthetic decision transactions by actor and fixed UTC calendar day."""
    if window != _REVIEW_WINDOW:
        raise ValueError("review-case window must be exactly 24 hours")

    grouped: dict[tuple[str, datetime], list[ObservedEvent]] = defaultdict(list)
    seen_event_ids: set[str] = set()
    for event in events:
        _validate_observation(event)
        if event.event_id in seen_event_ids:
            raise ValueError("review-case observations must have unique event IDs")
        seen_event_ids.add(event.event_id)
        assert event.decision_at is not None
        window_start = event.decision_at.astimezone(UTC).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        grouped[(event.actor_id, window_start)].append(event)

    return tuple(
        ReviewCase(
            entity_key=entity_key,
            window_start=window_start,
            event_ids=tuple(sorted(event.event_id for event in rows)),
            integrity_failure_event_ids=tuple(
                sorted(event.event_id for event in rows if event.integrity_status == "fail")
            ),
        )
        for (entity_key, window_start), rows in sorted(grouped.items())
    )


def aggregate_action_workload(
    total_transaction_count: int,
    review_cases: Sequence[ReviewCase],
    decisions: Sequence[DefenseDecision],
    truth: Sequence[EvaluationTruthRow],
) -> ActionWorkload:
    """Aggregate case review and transaction interventions with separate denominators."""
    if type(total_transaction_count) is not int or total_transaction_count <= 0:
        raise ValueError("total transaction count must be a positive integer")

    case_event_ids, integrity_failure_event_ids = _case_event_sets(review_cases)
    action_by_event_id = _actions_by_event_id(decisions)
    truth_by_event_id = _truth_by_event_id(truth)
    unknown_action_ids = set(action_by_event_id).difference(truth_by_event_id)
    if unknown_action_ids:
        raise ValueError("action decisions must reference operating truth rows")

    challenge_ids = {
        event_id for event_id, action in action_by_event_id.items() if action is Action.CHALLENGE
    }
    decline_ids = {
        event_id for event_id, action in action_by_event_id.items() if action is Action.DECLINE
    }
    reviewed_ids = challenge_ids.intersection(case_event_ids)
    reviewed_case_count = sum(
        bool(challenge_ids.intersection(case.event_ids)) for case in review_cases
    )
    legitimate_ids = {
        event_id for event_id, row in truth_by_event_id.items() if not row.is_fraud
    }
    false_decline_count = len(decline_ids.intersection(legitimate_ids))
    false_intervention_count = len((challenge_ids | decline_ids).intersection(legitimate_ids))
    legitimate_transaction_count = len(legitimate_ids)

    return ActionWorkload(
        total_transaction_count=total_transaction_count,
        legitimate_transaction_count=legitimate_transaction_count,
        review_case_count=reviewed_case_count,
        reviewed_transaction_count=len(reviewed_ids),
        challenge_count=len(challenge_ids),
        automatic_integrity_decline_count=len(decline_ids.intersection(integrity_failure_event_ids)),
        false_decline_count=false_decline_count,
        false_intervention_count=false_intervention_count,
        review_case_rate=reviewed_case_count / total_transaction_count,
        challenge_rate=len(challenge_ids) / total_transaction_count,
        false_decline_rate=(
            false_decline_count / legitimate_transaction_count
            if legitimate_transaction_count
            else None
        ),
        false_interventions_per_10k=false_intervention_count * 10_000 / total_transaction_count,
    )


def _validate_observation(event: ObservedEvent) -> None:
    if type(event) is not ObservedEvent:
        raise TypeError("review-case events must be exact ObservedEvent instances")
    ObservedEvent.model_validate(event.model_dump())
    if not event.is_decision_point or event.decision_at is None:
        raise ValueError("review-case events must be decision-point observations")
    validate_utc_timestamp(event.decision_at)


def _case_event_sets(review_cases: Sequence[ReviewCase]) -> tuple[set[str], set[str]]:
    event_ids: set[str] = set()
    integrity_failure_event_ids: set[str] = set()
    for case in review_cases:
        if type(case) is not ReviewCase:
            raise TypeError("review cases must be exact ReviewCase instances")
        ReviewCase.model_validate(case.model_dump())
        duplicate_ids = event_ids.intersection(case.event_ids)
        if duplicate_ids:
            raise ValueError("review cases must not share event IDs")
        event_ids.update(case.event_ids)
        integrity_failure_event_ids.update(case.integrity_failure_event_ids)
    return event_ids, integrity_failure_event_ids


def _actions_by_event_id(decisions: Sequence[DefenseDecision]) -> dict[str, Action]:
    actions: dict[str, Action] = {}
    for decision in decisions:
        if type(decision) is not DefenseDecision:
            raise TypeError("workload decisions must be exact DefenseDecision instances")
        DefenseDecision.model_validate(decision.model_dump())
        if decision.event_id in actions:
            raise ValueError("workload decisions must have unique event IDs")
        actions[decision.event_id] = decision.action
    return actions


def _truth_by_event_id(truth: Sequence[EvaluationTruthRow]) -> dict[str, EvaluationTruthRow]:
    rows: dict[str, EvaluationTruthRow] = {}
    for row in truth:
        if type(row) is not EvaluationTruthRow:
            raise TypeError("workload truth must contain exact EvaluationTruthRow instances")
        EvaluationTruthRow.model_validate(row.model_dump())
        if row.event_id in rows:
            raise ValueError("workload truth must have unique event IDs")
        rows[row.event_id] = row
    return rows


__all__ = ["ActionWorkload", "ReviewCase", "aggregate_action_workload", "group_review_cases"]
