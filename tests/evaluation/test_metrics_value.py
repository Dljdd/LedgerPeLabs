"""Hand-calculated lifecycle-value metric oracles."""

from __future__ import annotations

from datetime import timedelta
from decimal import Decimal

import pytest

from apar.contracts.decisions import Action
from apar.contracts.events import EventKind
from apar.evaluation.metrics import MetricContractError, compute_metric_report
from tests.evaluation.test_metrics_classification import (
    NOW,
    decision,
    make_inputs,
    observed,
    truth,
)


def _settled_inputs(
    *,
    first_action: Action = Action.DECLINE,
    first_decision_at=NOW,
    first_settlement_at=NOW + timedelta(seconds=10),
):
    opening_a = observed("open-a", amount="60.00", decision_at=first_decision_at)
    settle_a = observed(
        "settle-a",
        payment_id=opening_a.payment_id,
        event_type=EventKind.SETTLEMENT,
        amount="60.00",
        event_time=first_settlement_at,
        available_at=first_settlement_at,
        decision_at=None,
        is_decision_point=False,
        actor_id=opening_a.actor_id,
        counterparty_id=opening_a.counterparty_id,
    )
    opening_b = observed("open-b", amount="100.00")
    settle_b = observed(
        "settle-b",
        payment_id=opening_b.payment_id,
        event_type=EventKind.SETTLEMENT,
        amount="100.00",
        event_time=NOW + timedelta(seconds=20),
        available_at=NOW + timedelta(seconds=20),
        decision_at=None,
        is_decision_point=False,
        actor_id=opening_b.actor_id,
        counterparty_id=opening_b.counterparty_id,
    )
    recovery_b = observed(
        "recovery-b",
        payment_id=opening_b.payment_id,
        event_type=EventKind.RECOVERY,
        amount="60.00",
        event_time=NOW + timedelta(seconds=30),
        available_at=NOW + timedelta(seconds=30),
        decision_at=None,
        is_decision_point=False,
        actor_id=opening_b.actor_id,
        counterparty_id=opening_b.counterparty_id,
    )
    truth_rows = (
        truth(
            "open-a",
            is_fraud=True,
            payment_id=opening_a.payment_id,
            first_settlement_at=first_settlement_at,
            net_settled_value="60.00",
            lifecycle_event_ids=("open-a", "settle-a"),
        ),
        truth(
            "open-b",
            is_fraud=True,
            payment_id=opening_b.payment_id,
            first_settlement_at=NOW + timedelta(seconds=20),
            net_settled_value="40.00",
            lifecycle_event_ids=("open-b", "settle-b", "recovery-b"),
        ),
    )
    decisions = (
        decision("open-a", action=first_action, score=0.9),
        decision("open-b", action=Action.APPROVE, score=0.1),
    )
    return make_inputs(
        truth_rows,
        (opening_a, settle_a, opening_b, settle_b, recovery_b),
        decisions,
    )


def test_preventable_value_requires_a_strictly_pre_settlement_decline() -> None:
    report = compute_metric_report(_settled_inputs())
    assert report.value.currency == "USD"
    assert report.value.fraudulent_net_settled_value == Decimal("100.00")
    assert report.value.preventable_settled_value == Decimal("60.00")
    assert report.value.value_escaped == Decimal("40.00")
    assert report.value.challenge_credited_as_prevented == Decimal("0.00")
    assert report.value.value_before_first_alert == Decimal("0.00")
    assert report.value.remaining_preventable_at_alert == Decimal("60.00")
    assert report.value.captured_value_per_review_case.value == Decimal("60.00")
    assert report.value.captured_value_per_analyst_hour.value == Decimal("180.00")


@pytest.mark.parametrize("action", [Action.APPROVE, Action.CHALLENGE])
def test_approve_and_challenge_receive_no_counterfactual_prevention_credit(action: Action) -> None:
    report = compute_metric_report(_settled_inputs(first_action=action))
    assert report.value.preventable_settled_value == Decimal("0.00")
    assert report.value.value_escaped == Decimal("100.00")
    assert report.value.challenge_credited_as_prevented == Decimal("0.00")


def test_decline_at_or_after_exact_settlement_receives_no_credit() -> None:
    settlement = NOW + timedelta(seconds=10)
    exact = compute_metric_report(
        _settled_inputs(first_decision_at=settlement, first_settlement_at=settlement)
    )
    after = compute_metric_report(
        _settled_inputs(
            first_decision_at=settlement + timedelta(microseconds=1),
            first_settlement_at=settlement,
        )
    )
    assert exact.value.preventable_settled_value == Decimal("0.00")
    assert after.value.preventable_settled_value == Decimal("0.00")


def test_no_settlement_has_zero_value_without_inventing_prevention() -> None:
    opening = observed("open")
    inputs = make_inputs(
        (truth("open", is_fraud=True),),
        (opening,),
        (decision("open", action=Action.DECLINE, score=0.9),),
    )
    report = compute_metric_report(inputs)
    assert report.value.fraudulent_net_settled_value == Decimal("0.00")
    assert report.value.preventable_settled_value == Decimal("0.00")
    assert report.value.value_escaped == Decimal("0.00")


def test_truth_net_value_first_settlement_currency_and_lifecycle_are_reconstructed() -> None:
    inputs = _settled_inputs()
    wrong_total = inputs.truth[0].model_copy(update={"net_settled_value": Decimal("59.99")})
    with pytest.raises(MetricContractError, match="net settlement"):
        compute_metric_report(inputs.model_copy(update={"truth": (wrong_total, inputs.truth[1])}))
    wrong_time = inputs.truth[0].model_copy(
        update={"first_settlement_at": NOW + timedelta(seconds=9)}
    )
    with pytest.raises(MetricContractError, match="first settlement"):
        compute_metric_report(inputs.model_copy(update={"truth": (wrong_time, inputs.truth[1])}))
    mixed = inputs.observations[1].model_copy(update={"currency": "EUR"})
    with pytest.raises(MetricContractError, match="currency"):
        compute_metric_report(
            inputs.model_copy(
                update={"observations": (inputs.observations[0], mixed) + inputs.observations[2:]}
            )
        )
    with pytest.raises(MetricContractError, match="lifecycle"):
        compute_metric_report(inputs.model_copy(update={"observations": inputs.observations[:-1]}))


def test_post_alert_movement_is_not_counted_as_value_before_alert() -> None:
    report = compute_metric_report(_settled_inputs(first_action=Action.CHALLENGE))
    assert report.value.value_before_first_alert == Decimal("0.00")
    assert report.value.remaining_preventable_at_alert == Decimal("60.00")


def test_movement_strictly_before_later_alert_is_netted_once() -> None:
    opening = observed("open", decision_at=NOW + timedelta(seconds=30))
    settlement = observed(
        "settle",
        payment_id=opening.payment_id,
        event_type=EventKind.SETTLEMENT,
        amount="100.00",
        event_time=NOW + timedelta(seconds=10),
        available_at=NOW + timedelta(seconds=10),
        decision_at=None,
        is_decision_point=False,
        actor_id=opening.actor_id,
        counterparty_id=opening.counterparty_id,
    )
    refund = observed(
        "refund",
        payment_id=opening.payment_id,
        event_type=EventKind.REFUND,
        amount="25.00",
        event_time=NOW + timedelta(seconds=20),
        available_at=NOW + timedelta(seconds=20),
        decision_at=None,
        is_decision_point=False,
        actor_id=opening.actor_id,
        counterparty_id=opening.counterparty_id,
    )
    inputs = make_inputs(
        (
            truth(
                "open",
                is_fraud=True,
                payment_id=opening.payment_id,
                first_settlement_at=NOW + timedelta(seconds=10),
                net_settled_value="75.00",
                lifecycle_event_ids=("open", "settle", "refund"),
            ),
        ),
        (opening, settlement, refund),
        (decision("open", action=Action.CHALLENGE, score=0.9),),
    )
    report = compute_metric_report(inputs)
    assert report.value.value_before_first_alert == Decimal("75.00")
    assert report.value.remaining_preventable_at_alert == Decimal("0.00")


def test_extreme_decimal_is_normalized_to_metric_contract_error() -> None:
    inputs = _settled_inputs()
    huge = inputs.observations[1].model_copy(update={"amount": Decimal("1E+1000")})
    with pytest.raises(MetricContractError, match="numeric resource"):
        compute_metric_report(
            inputs.model_copy(
                update={"observations": (inputs.observations[0], huge) + inputs.observations[2:]}
            )
        )
