"""Case-aware synthetic workload contracts for Defend v2."""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal

from apar.cases.v2_workload import aggregate_action_workload, group_review_cases
from apar.contracts.decisions import Action
from apar.defense.policy import DefenseDecision
from apar.evaluation.contracts import EvaluationTruthRow
from tests.cases.conftest import decision, observation


def test_two_transactions_in_one_utc_window_are_one_review_case() -> None:
    events = (
        _observed("a", actor="actor-1", at="2026-01-01T10:00:00Z"),
        _observed("b", actor="actor-1", at="2026-01-01T10:05:00Z"),
    )

    workload = aggregate_action_workload(
        100,
        group_review_cases(events),
        (decision("a"), decision("b")),
        _truth_for(events),
    )

    assert (
        workload.review_case_count,
        workload.reviewed_transaction_count,
        workload.review_case_rate,
    ) == (1, 2, 0.01)


def test_utc_midnight_starts_a_new_review_case_and_records_day_volume() -> None:
    events = (
        _observed("late", actor="actor-1", at="2026-01-01T23:59:59Z"),
        _observed("early", actor="actor-1", at="2026-01-02T00:00:00Z"),
    )

    cases = group_review_cases(events)

    assert [(case.window_start, case.event_ids) for case in cases] == [
        (datetime(2026, 1, 1, tzinfo=UTC), ("late",)),
        (datetime(2026, 1, 2, tzinfo=UTC), ("early",)),
    ]


def test_false_decline_rate_uses_legitimate_denominator() -> None:
    truth = _truth_rows(
        "legitimate-1",
        *[f"legitimate-{index}" for index in range(2, 81)],
        *[f"fraud-{index}" for index in range(1, 21)],
        fraudulent_ids=tuple(f"fraud-{index}" for index in range(1, 21)),
    )

    workload = aggregate_action_workload(
        100,
        (),
        (_decision("legitimate-1", Action.DECLINE),),
        truth,
    )

    assert workload.legitimate_transaction_count == 80
    assert workload.false_decline_count == 1
    assert workload.false_decline_rate == 1 / 80
    assert workload.false_interventions_per_10k == 100.0


def test_decline_precedence_excludes_case_review_and_counts_integrity_decline() -> None:
    event = _observed(
        "integrity-failure",
        actor="actor-1",
        at="2026-01-01T10:00:00Z",
        integrity_status="fail",
        integrity_reason="receipt_failed",
    )

    workload = aggregate_action_workload(
        1,
        group_review_cases((event,)),
        (_decision(event.event_id, Action.DECLINE),),
        _truth_for((event,), fraudulent_ids=(event.event_id,)),
    )

    assert (
        workload.review_case_count,
        workload.reviewed_transaction_count,
        workload.challenge_count,
        workload.automatic_integrity_decline_count,
    ) == (0, 0, 0, 1)


def test_zero_legitimate_denominator_keeps_false_decline_rate_undefined() -> None:
    event = _observed("fraud-1", actor="actor-1", at="2026-01-01T10:00:00Z")

    workload = aggregate_action_workload(
        1,
        (),
        (_decision(event.event_id, Action.DECLINE),),
        _truth_for((event,), fraudulent_ids=(event.event_id,)),
    )

    assert workload.legitimate_transaction_count == 0
    assert workload.false_decline_rate is None


def _observed(event_id: str, *, actor: str, at: str, **updates: object):
    decision_at = datetime.fromisoformat(at.replace("Z", "+00:00"))
    return observation(
        event_id,
        actor_id=actor,
        counterparty_id="merchant-1",
        decision_at=decision_at,
        **updates,
    )


def _decision(event_id: str, action: Action) -> DefenseDecision:
    return decision(event_id, action=action)


def _truth_for(
    events: tuple[object, ...], *, fraudulent_ids: tuple[str, ...] = ()
) -> tuple[EvaluationTruthRow, ...]:
    return _truth_rows(
        *(event.event_id for event in events), fraudulent_ids=fraudulent_ids
    )


def _truth_rows(
    *event_ids: str, fraudulent_ids: tuple[str, ...] = ()
) -> tuple[EvaluationTruthRow, ...]:
    fraudulent = set(fraudulent_ids)
    timestamp = datetime(2026, 1, 8, tzinfo=UTC)
    return tuple(
        EvaluationTruthRow(
            event_id=event_id,
            payment_id=f"payment-{event_id}",
            campaign_id="synthetic-test-campaign",
            family="card_testing_cnp",
            viewpoint="development",
            is_fraud=event_id in fraudulent,
            label_source="population_truth",
            label_mature_at=timestamp,
            first_settlement_at=timestamp,
            net_settled_value=Decimal("1.00"),
            lifecycle_event_ids=(event_id,),
        )
        for event_id in event_ids
    )
