"""Past-only grouping and threshold-callback behavioral contracts."""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from time import perf_counter
from typing import cast

import numpy as np
import pytest
from pydantic import ValidationError

from apar.cases import (
    CaseContractError,
    CaseMotif,
    ReviewCaseCounter,
    bind_review_case_counter,
    group_cases,
)
from apar.contracts.decisions import Action
from apar.defense.policy import DefenseDecision, OperatingBudget
from apar.defense.thresholds import select_policy_thresholds
from apar.runs.wire import canonical_json_bytes
from tests.cases.conftest import NOW, decision, observation


def test_shared_counterparty_groups_alerts_without_future_evidence() -> None:
    first_time = NOW
    second_time = NOW + timedelta(minutes=1)
    observations = (
        observation(
            "event-a",
            actor_id="actor-a",
            counterparty_id="merchant",
            decision_at=first_time,
        ),
        observation(
            "event-b",
            actor_id="actor-b",
            counterparty_id="merchant",
            decision_at=second_time,
        ),
    )

    cases = group_cases(
        observations,
        (decision("event-a", score=0.7), decision("event-b", score=0.9)),
        as_of=second_time,
    )

    assert len(cases) == 1
    assert cases[0].event_ids == ("event-a", "event-b")
    assert cases[0].actor_ids == ("actor-a", "actor-b")
    assert cases[0].counterparty_ids == ("merchant",)


def test_isolated_alerts_open_distinct_cases() -> None:
    observations = (
        observation("event-a", actor_id="actor-a", counterparty_id="merchant-a"),
        observation("event-b", actor_id="actor-b", counterparty_id="merchant-b"),
    )
    decisions = (decision("event-a"), decision("event-b"))

    cases = group_cases(observations, decisions, as_of=NOW)

    assert len(cases) == 2
    assert tuple(item.event_ids for item in cases) == (("event-a",), ("event-b",))
    assert cases[0].case_id != cases[1].case_id


def test_shared_actor_and_approved_transitive_bridge_are_graph_evidence() -> None:
    bridge_time = NOW + timedelta(minutes=1)
    alert_time = NOW + timedelta(minutes=2)
    observations = (
        observation("alert-a", actor_id="actor-a", counterparty_id="merchant-a", decision_at=NOW),
        observation(
            "bridge",
            actor_id="actor-b",
            counterparty_id="merchant-a",
            decision_at=bridge_time,
        ),
        observation(
            "alert-b",
            actor_id="actor-b",
            counterparty_id="merchant-b",
            decision_at=alert_time,
        ),
    )
    decisions = (
        decision("alert-a"),
        decision("bridge", action=Action.APPROVE, score=0.1),
        decision("alert-b"),
    )

    cases = group_cases(observations, decisions, as_of=alert_time)

    assert len(cases) == 1
    assert cases[0].event_ids == ("alert-a", "alert-b")
    assert "bridge" not in cases[0].event_ids


def test_approve_rows_create_no_alert_or_case() -> None:
    row = observation("approved", actor_id="actor", counterparty_id="merchant")
    assert group_cases(
        (row,),
        (decision("approved", action=Action.APPROVE, score=0.1),),
        as_of=NOW,
    ) == ()


def test_future_extension_retains_first_evidence_id_and_priority() -> None:
    initial_observation = observation(
        "event-a", actor_id="actor-a", counterparty_id="merchant", decision_at=NOW
    )
    initial_decision = decision("event-a", score=0.8)
    before = group_cases((initial_observation,), (initial_decision,), as_of=NOW)
    later_time = NOW + timedelta(hours=1)
    later_observation = observation(
        "event-b", actor_id="actor-b", counterparty_id="merchant", decision_at=later_time
    )
    after = group_cases(
        (later_observation, initial_observation),
        (decision("event-b", score=1.0), initial_decision),
        as_of=later_time,
    )

    assert before[0].event_ids == ("event-a",)
    assert after[0].event_ids == ("event-a", "event-b")
    assert after[0].case_id == before[0].case_id
    assert after[0].priority == before[0].priority


def test_later_bridge_merges_only_after_a_subsequent_decision() -> None:
    first = observation(
        "event-a", actor_id="actor-a", counterparty_id="merchant-a", decision_at=NOW
    )
    second = observation(
        "event-b", actor_id="actor-b", counterparty_id="merchant-b", decision_at=NOW
    )
    before = group_cases(
        (first, second), (decision("event-a"), decision("event-b")), as_of=NOW
    )
    later = NOW + timedelta(minutes=1)
    bridge = observation(
        "event-bridge",
        actor_id="actor-a",
        counterparty_id="merchant-b",
        decision_at=later,
    )
    after = group_cases(
        (first, second, bridge),
        (decision("event-a"), decision("event-b"), decision("event-bridge")),
        as_of=later,
    )
    anchor = min(before, key=lambda item: (item.opened_at, item.case_id))

    assert len(before) == 2
    assert len(after) == 2
    bridge_case = next(item for item in after if "event-bridge" in item.event_ids)
    assert bridge_case.case_id == anchor.case_id
    assert bridge_case.priority == anchor.priority

    final_time = later + timedelta(minutes=1)
    trigger = observation(
        "event-trigger",
        actor_id="actor-b",
        counterparty_id="merchant-a",
        decision_at=final_time,
    )
    final = group_cases(
        (trigger, bridge, second, first),
        (
            decision("event-trigger"),
            decision("event-bridge"),
            decision("event-b"),
            decision("event-a"),
        ),
        as_of=final_time,
    )

    assert len(final) == 1
    assert final[0].case_id == anchor.case_id
    assert final[0].priority == anchor.priority
    assert final[0].event_ids == (
        "event-a",
        "event-b",
        "event-bridge",
        "event-trigger",
    )


def test_grouping_is_permutation_stable() -> None:
    observations = (
        observation("event-a", actor_id="actor-a", counterparty_id="shared"),
        observation("event-b", actor_id="actor-b", counterparty_id="shared"),
        observation("event-c", actor_id="actor-c", counterparty_id="isolated"),
    )
    decisions = tuple(decision(row.event_id) for row in observations)

    canonical = group_cases(observations, decisions, as_of=NOW)
    permuted = group_cases(
        tuple(reversed(observations)),
        (decisions[1], decisions[2], decisions[0]),
        as_of=NOW,
    )

    assert permuted == canonical


def test_equal_time_batch_is_symmetric_and_canonical() -> None:
    observations = (
        observation("z-event", actor_id="actor-z", counterparty_id="shared"),
        observation("a-event", actor_id="actor-a", counterparty_id="shared"),
    )
    decisions = (decision("z-event", score=0.6), decision("a-event", score=0.9))

    forward = group_cases(observations, decisions, as_of=NOW)
    reverse = group_cases(tuple(reversed(observations)), tuple(reversed(decisions)), as_of=NOW)

    assert forward == reverse
    assert tuple(item.event_ids for item in forward) == (("a-event",), ("z-event",))


def test_current_alert_value_is_excluded_from_historical_evidence() -> None:
    row = observation(
        "current",
        actor_id="actor",
        counterparty_id="merchant",
        amount="100.00",
    )

    grouped = group_cases((row,), (decision("current", score=0.8),), as_of=NOW)

    evidence = grouped[0].alert_evidence[0]
    assert evidence.visible_value_before_alert == Decimal("0")
    assert evidence.latest_graph_evidence_at is None
    assert evidence.evidence_source_ids == ()
    assert grouped[0].priority == 38.222222


def test_same_decision_peer_cannot_rewrite_first_alert_evidence_or_identity() -> None:
    first = observation(
        "first",
        actor_id="actor-a",
        counterparty_id="shared",
        amount="10.00",
    )
    peer = observation(
        "peer",
        actor_id="actor-b",
        counterparty_id="shared",
        amount="999.00",
    )
    before = group_cases((first,), (decision("first"),), as_of=NOW)
    after = group_cases(
        (peer, first),
        (decision("peer"), decision("first")),
        as_of=NOW,
    )
    first_after = next(item for item in after if "first" in item.event_ids)

    assert len(after) == 2
    assert first_after.case_id == before[0].case_id
    assert first_after.priority == before[0].priority
    assert canonical_json_bytes(
        first_after.alert_evidence[0].model_dump(mode="json")
    ) == canonical_json_bytes(before[0].alert_evidence[0].model_dump(mode="json"))


def test_current_bridge_does_not_merge_prior_cases_until_a_later_decision() -> None:
    bridge_at = NOW + timedelta(minutes=1)
    later_at = NOW + timedelta(minutes=2)
    first = observation(
        "historical-a",
        actor_id="actor-a",
        counterparty_id="merchant-a",
        amount="10.00",
    )
    second = observation(
        "historical-b",
        actor_id="actor-b",
        counterparty_id="merchant-b",
        amount="20.00",
    )
    bridge = observation(
        "current-bridge",
        actor_id="actor-a",
        counterparty_id="merchant-b",
        amount="999.00",
        decision_at=bridge_at,
    )
    at_bridge = group_cases(
        (first, second, bridge),
        (decision("historical-a"), decision("historical-b"), decision("current-bridge")),
        as_of=bridge_at,
    )
    bridge_case = next(item for item in at_bridge if "current-bridge" in item.event_ids)
    bridge_evidence = next(
        item for item in bridge_case.alert_evidence if item.event_id == "current-bridge"
    )

    assert len(at_bridge) == 2
    assert bridge_evidence.visible_value_before_alert == Decimal("30.00")
    assert bridge_evidence.latest_graph_evidence_at < bridge_at

    later = observation(
        "later",
        actor_id="actor-b",
        counterparty_id="merchant-a",
        decision_at=later_at,
    )
    after = group_cases(
        (later, bridge, second, first),
        (
            decision("later"),
            decision("current-bridge"),
            decision("historical-b"),
            decision("historical-a"),
        ),
        as_of=later_at,
    )

    assert len(after) == 1
    assert after[0].event_ids == (
        "current-bridge",
        "historical-a",
        "historical-b",
        "later",
    )


def test_current_batch_edge_is_excluded_regardless_of_early_availability() -> None:
    later = NOW + timedelta(minutes=1)
    first = observation(
        "alert-a", actor_id="actor-a", counterparty_id="merchant-x", decision_at=NOW
    )
    boundary_bridge = observation(
        "bridge",
        actor_id="actor-b",
        counterparty_id="merchant-x",
        decision_at=later,
        available_at=later,
    )
    second = observation(
        "alert-b",
        actor_id="actor-b",
        counterparty_id="merchant-y",
        decision_at=later,
        available_at=later,
    )
    decisions = (
        decision("alert-a"),
        decision("bridge", action=Action.APPROVE, score=0.1),
        decision("alert-b"),
    )

    boundary = group_cases((first, boundary_bridge, second), decisions, as_of=later)
    visible_bridge = boundary_bridge.model_copy(
        update={"available_at": later - timedelta(microseconds=1)}
    )
    visible = group_cases((first, visible_bridge, second), decisions, as_of=later)

    assert len(boundary) == 2
    assert len(visible) == 2
    boundary_first = next(
        case for case in boundary if "alert-a" in case.event_ids
    ).alert_evidence[0]
    visible_first = visible[0].alert_evidence[0]
    assert canonical_json_bytes(
        boundary_first.model_dump(mode="json")
    ) == canonical_json_bytes(visible_first.model_dump(mode="json"))


def test_future_decision_bridge_is_admitted_at_knowledge_time() -> None:
    alert_at = NOW + timedelta(minutes=1)
    bridge_decision_at = NOW + timedelta(minutes=2)
    historical = observation(
        "historical",
        actor_id="actor-a",
        counterparty_id="merchant-a",
        amount="10.00",
    )
    bridge = observation(
        "approved-bridge",
        actor_id="actor-b",
        counterparty_id="merchant-a",
        amount="20.00",
        event_time=NOW + timedelta(seconds=20),
        available_at=NOW + timedelta(seconds=30),
        decision_at=bridge_decision_at,
    )
    alert = observation(
        "alert",
        actor_id="actor-b",
        counterparty_id="merchant-b",
        decision_at=alert_at,
    )
    decisions = (
        decision("historical"),
        decision("alert"),
        decision("approved-bridge", action=Action.APPROVE, score=0.1),
    )

    first = group_cases((bridge, alert, historical), decisions, as_of=bridge_decision_at)
    shifted_bridge = bridge.model_copy(
        update={"decision_at": bridge_decision_at + timedelta(minutes=1)}
    )
    shifted = group_cases(
        (alert, historical, shifted_bridge),
        decisions,
        as_of=bridge_decision_at + timedelta(minutes=1),
    )

    assert len(first) == 1
    assert first[0].event_ids == ("alert", "historical")
    assert first[0].alert_evidence == shifted[0].alert_evidence


def test_nondecision_bridge_uses_availability_and_strict_equal_time_boundary() -> None:
    alert_at = NOW + timedelta(minutes=1)
    historical = observation(
        "historical",
        actor_id="actor-a",
        counterparty_id="merchant-a",
    )
    bridge = observation(
        "context-bridge",
        actor_id="actor-b",
        counterparty_id="merchant-a",
        event_time=NOW + timedelta(seconds=20),
        available_at=NOW + timedelta(seconds=30),
        decision_at=alert_at,
    ).model_copy(update={"decision_at": None, "is_decision_point": False})
    alert = observation(
        "alert",
        actor_id="actor-b",
        counterparty_id="merchant-b",
        decision_at=alert_at,
    )
    decisions = (decision("historical"), decision("alert"))

    visible = group_cases((bridge, alert, historical), decisions, as_of=alert_at)
    boundary_bridge = bridge.model_copy(update={"available_at": alert_at})
    boundary = group_cases(
        (alert, boundary_bridge, historical), decisions, as_of=alert_at
    )

    assert len(visible) == 1
    assert len(boundary) == 2
    assert visible == group_cases(
        (historical, alert, bridge), tuple(reversed(decisions)), as_of=alert_at
    )


def test_current_batch_temporarily_excludes_previously_known_own_edge() -> None:
    trigger_at = NOW + timedelta(minutes=1)
    current_at = NOW + timedelta(minutes=2)
    later_at = NOW + timedelta(minutes=3)
    historical = observation(
        "historical", actor_id="actor-a", counterparty_id="merchant-a"
    )
    early_bridge = observation(
        "early-bridge",
        actor_id="actor-b",
        counterparty_id="merchant-a",
        event_time=NOW + timedelta(seconds=20),
        available_at=NOW + timedelta(seconds=30),
        decision_at=current_at,
    )
    trigger = observation(
        "trigger",
        actor_id="actor-trigger",
        counterparty_id="merchant-trigger",
        decision_at=trigger_at,
    )
    current_peer = observation(
        "current-peer",
        actor_id="actor-b",
        counterparty_id="merchant-b",
        decision_at=current_at,
    )
    decisions = (
        decision("historical"),
        decision("trigger", action=Action.APPROVE),
        decision("early-bridge", action=Action.APPROVE),
        decision("current-peer"),
    )

    at_current = group_cases(
        (current_peer, early_bridge, historical, trigger),
        decisions,
        as_of=current_at,
    )

    assert len(at_current) == 2
    peer_case = next(item for item in at_current if "current-peer" in item.event_ids)
    assert peer_case.alert_evidence[0].motif is CaseMotif.ISOLATED

    later = observation(
        "later",
        actor_id="actor-b",
        counterparty_id="merchant-b",
        decision_at=later_at,
    )
    after = group_cases(
        (later, current_peer, early_bridge, historical, trigger),
        decisions + (decision("later"),),
        as_of=later_at,
    )

    assert len(after) == 1


def test_isolated_priority_includes_current_expected_value_not_visible_history() -> None:
    small = observation(
        "small",
        actor_id="actor-small",
        counterparty_id="merchant-small",
        amount="1.00",
    )
    large = observation(
        "large",
        actor_id="actor-large",
        counterparty_id="merchant-large",
        amount="1000000.00",
    )

    small_case = group_cases((small,), (decision("small", score=0.8),), as_of=NOW)[0]
    large_case = group_cases((large,), (decision("large", score=0.8),), as_of=NOW)[0]

    assert small_case.alert_evidence[0].visible_value_before_alert == Decimal("0")
    assert large_case.alert_evidence[0].visible_value_before_alert == Decimal("0")
    assert small_case.priority == 36.023981
    assert large_case.priority == 65.962547
    assert small_case.priority < large_case.priority


def test_current_expected_value_overflow_is_a_typed_case_error() -> None:
    row = observation(
        "overflow",
        actor_id="actor",
        counterparty_id="merchant",
    ).model_copy(update={"amount": Decimal("1e1000000")})

    with pytest.raises(CaseContractError, match="resource|priority|bounds"):
        group_cases((row,), (decision("overflow", score=0.8),), as_of=NOW)


@pytest.mark.parametrize(
    ("observations", "decisions", "match"),
    [
        (
            lambda: (
                observation("same", actor_id="a", counterparty_id="x"),
                observation("same", actor_id="b", counterparty_id="y"),
            ),
            lambda: (decision("same"), decision("other")),
            "duplicate observation event_id",
        ),
        (
            lambda: (
                observation("one", actor_id="a", counterparty_id="x"),
                observation("two", actor_id="b", counterparty_id="y"),
            ),
            lambda: (decision("one"), decision("one")),
            "duplicate decision event_id",
        ),
        (
            lambda: (observation("one", actor_id="a", counterparty_id="x"),),
            lambda: (decision("other"),),
            "event IDs must match",
        ),
        (
            lambda: (observation("one", actor_id="a", counterparty_id="x"),),
            lambda: (),
            "decision event IDs",
        ),
        (
            lambda: (
                observation(
                    "one", actor_id="a", counterparty_id="x", is_decision_point=False
                ),
            ),
            lambda: (decision("one"),),
            "nondecision",
        ),
    ],
)
def test_grouping_rejects_non_bijective_inputs(
    observations: object, decisions: object, match: str
) -> None:
    with pytest.raises((TypeError, ValueError), match=match):
        group_cases(
            cast("object", observations)(),  # type: ignore[operator]
            cast("object", decisions)(),  # type: ignore[operator]
            as_of=NOW,
        )


def test_grouping_rejects_non_exact_row_types() -> None:
    class DecisionSubclass(DefenseDecision):
        pass

    row = observation("one", actor_id="a", counterparty_id="x")
    bad = DecisionSubclass.model_validate(decision("one").model_dump())
    with pytest.raises(TypeError, match="exact DefenseDecision"):
        group_cases((row,), (bad,), as_of=NOW)
    with pytest.raises(TypeError, match="exact ObservedEvent"):
        group_cases((cast("object", row.model_dump()),), (decision("one"),), as_of=NOW)


@pytest.mark.parametrize(
    ("row", "as_of", "match"),
    [
        (
            observation(
                "one",
                actor_id="a",
                counterparty_id="x",
                event_time=NOW,
                available_at=NOW - timedelta(seconds=1),
            ),
            NOW,
            "event_time",
        ),
        (
            observation(
                "one",
                actor_id="a",
                counterparty_id="x",
                decision_at=NOW,
                available_at=NOW + timedelta(seconds=1),
            ),
            NOW + timedelta(seconds=1),
            "available_at",
        ),
        (
            observation(
                "one",
                actor_id="a",
                counterparty_id="x",
                decision_at=NOW + timedelta(seconds=1),
            ),
            NOW,
            "after as_of",
        ),
    ],
)
def test_grouping_rejects_incoherent_or_future_timestamps(
    row: object, as_of: datetime, match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        group_cases((cast("object", row),), (decision("one"),), as_of=as_of)


@pytest.mark.parametrize(
    "bad_as_of",
    [
        datetime(2026, 8, 18, 12, 0),
        datetime(
            2026,
            8,
            18,
            17,
            30,
            tzinfo=timezone(timedelta(hours=5, minutes=30)),
        ),
    ],
)
def test_grouping_requires_exact_utc_as_of(bad_as_of: datetime) -> None:
    with pytest.raises(ValueError, match="UTC"):
        group_cases((), (), as_of=bad_as_of)


def test_callback_requires_exact_object_action_vector_and_preserves_severity_mask() -> None:
    observations = (
        observation("event-a", actor_id="actor-a", counterparty_id="shared"),
        observation("event-b", actor_id="actor-b", counterparty_id="shared"),
    )
    decisions = (decision("event-a"), decision("event-b"))
    callback = bind_review_case_counter(observations, decisions, as_of=NOW)

    challenges = np.asarray([Action.CHALLENGE, Action.CHALLENGE], dtype=object)
    declines = np.asarray([Action.DECLINE, Action.DECLINE], dtype=object)
    assert callback(challenges) == callback(declines) == 2
    assert challenges.tolist() == [Action.CHALLENGE, Action.CHALLENGE]
    with pytest.raises(TypeError, match="exact numpy.ndarray"):
        callback(cast("object", [Action.CHALLENGE, Action.CHALLENGE]))
    with pytest.raises(TypeError, match="dtype object"):
        callback(np.asarray([1, 1], dtype=np.int64).astype(object).astype(np.int64))
    with pytest.raises(ValueError, match="length"):
        callback(np.asarray([Action.CHALLENGE], dtype=object))
    with pytest.raises(TypeError, match="exact Action"):
        callback(np.asarray(["challenge", "challenge"], dtype=object))


def test_callback_binding_rejects_noncanonical_or_misaligned_rows() -> None:
    observations = (
        observation("event-a", actor_id="actor-a", counterparty_id="shared"),
        observation("event-b", actor_id="actor-b", counterparty_id="shared"),
    )
    decisions = (decision("event-a"), decision("event-b"))
    with pytest.raises(ValueError, match="canonical availability order"):
        bind_review_case_counter(tuple(reversed(observations)), decisions, as_of=NOW)
    with pytest.raises(ValueError, match="canonical decision order"):
        bind_review_case_counter(observations, tuple(reversed(decisions)), as_of=NOW)


def test_callback_contract_rejects_pydantic_row_coercion() -> None:
    row = observation("event-a", actor_id="actor-a", counterparty_id="shared")
    with pytest.raises(ValidationError, match="exact ObservedEvent"):
        ReviewCaseCounter(
            observations=(row.model_dump(),),  # type: ignore[arg-type]
            decisions=(decision("event-a"),),
            as_of=NOW,
        )


def test_threshold_selection_uses_causal_callback_with_all_frozen_budgets() -> None:
    row_count = 1_000
    observations = tuple(
        observation(
            f"event-{index:04d}",
            actor_id=f"actor-{index:04d}",
            counterparty_id="shared-counterparty",
            decision_at=NOW + timedelta(microseconds=index),
        )
        for index in range(row_count)
    )
    decisions = tuple(
        decision(row.event_id, action=Action.APPROVE, score=0.1)
        for row in observations
    )
    callback = bind_review_case_counter(
        observations,
        decisions,
        as_of=cast(datetime, observations[-1].decision_at),
    )
    labels = np.zeros(row_count, dtype=np.int8)
    labels[:20] = 1
    scores = np.full(row_count, 0.1, dtype=np.float64)
    scores[:20] = 0.9
    mandatory = np.empty(row_count, dtype=object)
    mandatory[:] = [Action.APPROVE] * row_count
    budget = OperatingBudget()

    report = select_policy_thresholds(scores, labels, mandatory, callback, budget)

    assert report.feasible is True
    assert report.calibration_false_decline_rate == 0.0
    assert report.calibration_challenge_rate == 0.02
    assert report.calibration_review_case_rate == 0.001
    assert report.calibration_false_decline_rate <= budget.false_decline_rate_max
    assert report.calibration_challenge_rate <= budget.challenge_rate_max
    assert report.calibration_review_case_rate <= budget.review_case_rate_max
    assert math.isfinite(cast(float, report.objective_value))


def test_grouping_revalidates_model_constructed_decision_action() -> None:
    row = observation("forged", actor_id="actor", counterparty_id="merchant")
    valid = decision("forged", action=Action.APPROVE, score=0.1)
    forged = DefenseDecision.model_construct(
        **{**valid.model_dump(mode="python"), "action": "approve"}
    )

    with pytest.raises(CaseContractError, match="DefenseDecision|Action|semantic"):
        group_cases((row,), (forged,), as_of=NOW)


def test_grouping_revalidates_model_constructed_observation_and_numeric_decision() -> None:
    valid_row = observation("forged", actor_id="actor", counterparty_id="merchant")
    forged_row = type(valid_row).model_construct(
        **{**valid_row.model_dump(mode="python"), "amount": "100.00"}
    )
    valid_decision = decision("forged")
    forged_decision = DefenseDecision.model_construct(
        **{**valid_decision.model_dump(mode="python"), "score": 1}
    )

    with pytest.raises(CaseContractError, match="ObservedEvent|exact|semantic"):
        group_cases((forged_row,), (valid_decision,), as_of=NOW)
    with pytest.raises(CaseContractError, match="DefenseDecision|exact|semantic"):
        group_cases((valid_row,), (forged_decision,), as_of=NOW)


def test_per_alert_evidence_freezes_causal_motif_chronology_and_value() -> None:
    shared_actor_at = NOW + timedelta(minutes=1)
    bridge_at = NOW + timedelta(minutes=2)
    transitive_at = NOW + timedelta(minutes=3)
    rows = (
        observation(
            "isolated",
            actor_id="actor-a",
            counterparty_id="merchant-a",
            amount="10.00",
        ),
        observation(
            "shared-actor",
            actor_id="actor-a",
            counterparty_id="merchant-b",
            amount="20.00",
            decision_at=shared_actor_at,
        ),
        observation(
            "bridge",
            actor_id="actor-b",
            counterparty_id="merchant-b",
            amount="30.00",
            decision_at=bridge_at,
        ),
        observation(
            "transitive",
            actor_id="actor-b",
            counterparty_id="merchant-c",
            amount="40.00",
            decision_at=transitive_at,
        ),
    )
    decisions = (
        decision("isolated"),
        decision("shared-actor"),
        decision("bridge", action=Action.APPROVE, score=0.1),
        decision("transitive"),
    )

    grouped = group_cases(rows, decisions, as_of=transitive_at)

    assert len(grouped) == 1
    evidence = grouped[0].alert_evidence
    assert tuple(item.event_id for item in evidence) == (
        "isolated",
        "shared-actor",
        "transitive",
    )
    assert tuple(item.motif for item in evidence) == (
        CaseMotif.ISOLATED,
        CaseMotif.SHARED_ACTOR,
        CaseMotif.TRANSITIVE,
    )
    assert tuple(item.decision_at for item in evidence) == (
        NOW,
        shared_actor_at,
        transitive_at,
    )
    assert evidence[-1].visible_value_before_alert == Decimal("60.00")
    assert grouped[0].first_evidence_ids == ("isolated",)


def test_shared_counterparty_motif_is_stable_without_future_rejoin() -> None:
    later = NOW + timedelta(minutes=1)
    future = NOW + timedelta(hours=1)
    first = observation("first", actor_id="actor-a", counterparty_id="shared")
    second = observation(
        "second",
        actor_id="actor-b",
        counterparty_id="shared",
        decision_at=later,
    )
    before = group_cases(
        (first, second),
        (decision("first"), decision("second")),
        as_of=later,
    )
    future_row = observation(
        "future",
        actor_id="actor-c",
        counterparty_id="shared",
        decision_at=future,
        amount="999999.00",
    )
    after = group_cases(
        (future_row, second, first),
        (decision("future"), decision("second"), decision("first")),
        as_of=future,
    )

    assert before[0].alert_evidence[1].motif is CaseMotif.SHARED_COUNTERPARTY
    assert after[0].alert_evidence[:2] == before[0].alert_evidence
    assert after[0].first_evidence_ids == before[0].first_evidence_ids


def test_grouping_visits_each_graph_observation_once(monkeypatch: pytest.MonkeyPatch) -> None:
    from apar.cases import grouping

    row_count = 400
    rows = tuple(
        observation(
            f"event-{index:04d}",
            actor_id=f"actor-{index:04d}",
            counterparty_id=f"merchant-{index:04d}",
            decision_at=NOW + timedelta(microseconds=index),
        )
        for index in range(row_count)
    )
    decisions = tuple(decision(row.event_id) for row in rows)
    visits = 0
    original = grouping._IncrementalGraph.add_observation

    def counted(graph: object, row: object) -> None:
        nonlocal visits
        visits += 1
        original(graph, row)  # type: ignore[arg-type]

    monkeypatch.setattr(grouping._IncrementalGraph, "add_observation", counted)

    assert len(group_cases(rows, decisions, as_of=rows[-1].decision_at)) == row_count
    assert visits == row_count


def test_merged_case_index_does_not_rescan_stale_aliases_per_later_batch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from apar.cases import grouping

    isolated_count = 100
    bridge_at = NOW + timedelta(minutes=1)
    merge_at = NOW + timedelta(minutes=2)
    later_at = NOW + timedelta(minutes=3)
    isolated = tuple(
        observation(
            f"isolated-{index:03d}",
            actor_id=f"actor-{index:03d}",
            counterparty_id=f"merchant-{index:03d}",
        )
        for index in range(isolated_count)
    )
    bridges = tuple(
        observation(
            f"bridge-{index:03d}",
            actor_id=f"actor-{index:03d}",
            counterparty_id=f"merchant-{index + 1:03d}",
            decision_at=bridge_at,
        )
        for index in range(isolated_count - 1)
    )
    merger = observation(
        "merger",
        actor_id="actor-000",
        counterparty_id="merchant-000",
        decision_at=merge_at,
    )
    later = tuple(
        observation(
            f"later-{index:03d}",
            actor_id="actor-000",
            counterparty_id="merchant-000",
            decision_at=later_at + timedelta(microseconds=index),
        )
        for index in range(isolated_count)
    )
    rows = isolated + bridges + (merger,) + later
    decisions = (
        tuple(decision(row.event_id) for row in isolated)
        + tuple(decision(row.event_id, action=Action.APPROVE) for row in bridges)
        + (decision("merger"),)
        + tuple(decision(row.event_id) for row in later)
    )
    lookup_ids = 0
    original = grouping._IncrementalGraph.case_ids

    def counted(
        graph: object, roots: set[str] | frozenset[str]
    ) -> set[str]:
        nonlocal lookup_ids
        result = original(graph, roots)  # type: ignore[arg-type]
        lookup_ids += len(result)
        return result

    monkeypatch.setattr(grouping._IncrementalGraph, "case_ids", counted)

    grouped = group_cases(rows, decisions, as_of=later[-1].decision_at)

    assert len(grouped) == 1
    assert lookup_ids <= isolated_count * 3


def test_1500_isolated_grouping_meets_frozen_benchmark_ceiling() -> None:
    row_count = 1_500
    rows = tuple(
        observation(
            f"event-{index:04d}",
            actor_id=f"actor-{index:04d}",
            counterparty_id=f"merchant-{index:04d}",
        )
        for index in range(row_count)
    )
    decisions = tuple(decision(row.event_id) for row in rows)

    started = perf_counter()
    grouped = group_cases(rows, decisions, as_of=NOW)
    elapsed = perf_counter() - started

    assert len(grouped) == row_count
    assert elapsed < 2.5


def test_review_case_counter_random_masks_match_full_grouping() -> None:
    row_count = 48
    rows = tuple(
        observation(
            f"parity-{index:03d}",
            actor_id=f"actor-{index % 12:02d}",
            counterparty_id=f"merchant-{(index * 5) % 17:02d}",
            decision_at=NOW + timedelta(seconds=index // 4),
        )
        for index in range(row_count)
    )
    templates = tuple(
        decision(row.event_id, action=Action.APPROVE, score=(index + 1) / 50)
        for index, row in enumerate(rows)
    )
    callback = bind_review_case_counter(rows, templates, as_of=rows[-1].decision_at)
    generator = np.random.default_rng(260816)

    for _ in range(64):
        numeric = generator.integers(0, 3, size=row_count)
        actions = np.asarray(
            tuple((Action.APPROVE, Action.CHALLENGE, Action.DECLINE)[value] for value in numeric),
            dtype=object,
        )
        candidates = tuple(
            decision(row.event_id, action=cast(Action, action), score=templates[index].score)
            for index, (row, action) in enumerate(zip(rows, actions, strict=True))
        )
        expected = len(group_cases(rows, candidates, as_of=rows[-1].decision_at))

        assert callback(actions) == expected
        assert callback(actions) == expected


def test_review_case_counter_matches_full_grouping_with_context_rows() -> None:
    alert_at = NOW + timedelta(minutes=1)
    historical = observation(
        "historical", actor_id="actor-a", counterparty_id="merchant-a"
    )
    context = observation(
        "context",
        actor_id="actor-b",
        counterparty_id="merchant-a",
        available_at=NOW + timedelta(seconds=30),
        event_time=NOW + timedelta(seconds=20),
        decision_at=alert_at,
    ).model_copy(update={"decision_at": None, "is_decision_point": False})
    alert = observation(
        "alert",
        actor_id="actor-b",
        counterparty_id="merchant-b",
        decision_at=alert_at,
    )
    rows = (historical, context, alert)
    templates = (decision("historical"), decision("alert"))
    callback = bind_review_case_counter(rows, templates, as_of=alert_at)
    actions = np.asarray([Action.CHALLENGE, Action.CHALLENGE], dtype=object)

    assert callback(actions) == len(group_cases(rows, templates, as_of=alert_at)) == 1


def test_review_case_counter_has_no_reachable_result_cache_to_poison() -> None:
    rows = (
        observation("first", actor_id="actor-a", counterparty_id="merchant-a"),
        observation("second", actor_id="actor-b", counterparty_id="merchant-b"),
    )
    decisions = tuple(decision(row.event_id) for row in rows)
    callback = bind_review_case_counter(rows, decisions, as_of=NOW)
    actions = np.asarray([Action.CHALLENGE, Action.CHALLENGE], dtype=object)

    assert callback(actions) == 2
    assert not hasattr(callback, "_mask_cache")
    with pytest.raises(TypeError, match="immutable"):
        callback._topology = callback._topology
    assert callback(actions) == 2


def test_high_cardinality_threshold_callback_meets_frozen_ceiling() -> None:
    row_count = 300
    rows = tuple(
        observation(
            f"threshold-{index:03d}",
            actor_id=f"actor-{index:03d}",
            counterparty_id=f"merchant-{index:03d}",
        )
        for index in range(row_count)
    )
    templates = tuple(
        decision(row.event_id, action=Action.APPROVE, score=(index + 1) / (row_count + 1))
        for index, row in enumerate(rows)
    )
    callback = bind_review_case_counter(rows, templates, as_of=NOW)
    scores = np.linspace(0.001, 0.999, row_count, dtype=np.float64)
    labels = np.zeros(row_count, dtype=np.int8)
    labels[-30:] = 1
    mandatory = np.empty(row_count, dtype=object)
    mandatory[:] = [Action.APPROVE] * row_count

    started = perf_counter()
    report = select_policy_thresholds(
        scores,
        labels,
        mandatory,
        callback,
        OperatingBudget(),
    )
    elapsed = perf_counter() - started

    assert report.candidate_threshold_count == row_count + 2
    assert elapsed < 3.0


def test_review_case_counter_accepts_more_rows_than_unique_score_limit() -> None:
    row_count = 4_097
    rows = tuple(
        observation(
            f"cap-{index:04d}",
            actor_id=f"actor-{index:04d}",
            counterparty_id=f"merchant-{index:04d}",
        )
        for index in range(row_count)
    )
    decisions = tuple(decision(row.event_id, action=Action.APPROVE) for row in rows)
    callback = bind_review_case_counter(rows, decisions, as_of=NOW)
    scores = np.full(row_count, 0.1, dtype=np.float64)
    scores[-1] = 0.9
    labels = np.zeros(row_count, dtype=np.int8)
    labels[-1] = 1
    mandatory = np.empty(row_count, dtype=object)
    mandatory[:] = [Action.APPROVE] * row_count

    report = select_policy_thresholds(
        scores, labels, mandatory, callback, OperatingBudget()
    )

    assert report.row_count == row_count
    assert report.candidate_threshold_count == 4


def test_review_case_counter_uses_grouping_row_cap_not_score_cap() -> None:
    from apar.cases import grouping

    row = observation("repeated", actor_id="actor", counterparty_id="merchant")
    template = decision("repeated", action=Action.APPROVE)

    with pytest.raises(CaseContractError, match="duplicate observation"):
        bind_review_case_counter(
            (row,) * grouping._MAX_GROUPING_ROWS,
            (template,) * grouping._MAX_GROUPING_ROWS,
            as_of=NOW,
        )
    with pytest.raises(CaseContractError, match="resource cap"):
        bind_review_case_counter(
            (row,) * (grouping._MAX_GROUPING_ROWS + 1),
            (template,) * (grouping._MAX_GROUPING_ROWS + 1),
            as_of=NOW,
        )
