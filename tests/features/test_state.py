"""Behavior tests for strict knowledge-time feature state."""

from __future__ import annotations

import hashlib
import json
from datetime import timedelta

import pytest

from apar.contracts.events import EventKind
from apar.defense.contracts import ObservedEvent
from apar.features.catalog import FeatureCatalog
from apar.features.state import CausalFeatureState, FeatureStateError
from tests.features.conftest import BASE_TIME, observation


def _checkpoint_bytes(document: dict[str, object]) -> bytes:
    unsigned = {key: value for key, value in document.items() if key != "self_digest"}
    unsigned_bytes = json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode()
    document["self_digest"] = hashlib.sha256(unsigned_bytes).hexdigest()
    return json.dumps(document, sort_keys=True, separators=(",", ":")).encode()


def test_available_at_equal_to_decision_is_not_historical_source(
    feature_catalog: FeatureCatalog,
) -> None:
    """Catches replacing the strict availability comparison with <=."""
    equal = observation(1, seconds=0, decision=False)
    decision = observation(2, seconds=30, availability_seconds=30)
    equal = equal.model_copy(
        update={
            "event_time": BASE_TIME - timedelta(seconds=10),
            "available_at": decision.decision_at,
        }
    )

    row = CausalFeatureState(feature_catalog).process((equal, decision))[0]

    assert row.values["actor_count_1m"] == 0.0
    assert row.source_event_ids == ()
    assert row.max_source_available_at is None


def test_source_one_microsecond_before_decision_is_admitted(
    feature_catalog: FeatureCatalog,
) -> None:
    """Catches an overly conservative boundary that discards strictly prior knowledge."""
    decision = observation(2, seconds=30, availability_seconds=30)
    prior = observation(1, seconds=20, decision=False).model_copy(
        update={"available_at": decision.decision_at - timedelta(microseconds=1)}
    )

    row = CausalFeatureState(feature_catalog).process((decision, prior))[0]

    assert row.values["actor_count_1m"] == 1.0
    assert row.source_event_ids == (prior.event_id,)
    assert row.max_source_available_at == prior.available_at


def test_equal_time_decisions_do_not_observe_one_another(
    equal_time_observations: tuple[ObservedEvent, ...],
    feature_catalog: FeatureCatalog,
) -> None:
    """Catches mutating state between decisions in the same timestamp batch."""
    rows = CausalFeatureState(feature_catalog).process(equal_time_observations)

    assert [row.values["actor_count_1m"] for row in rows] == [0.0, 0.0]
    assert all(row.max_source_available_at is None for row in rows)


def test_unseen_equal_time_decision_is_rejected_after_timestamp_is_closed(
    feature_catalog: FeatureCatalog,
) -> None:
    """Catches call partitioning changing an equal-time cohort's pre-batch state."""
    first = observation(1, seconds=30, actor="actor-a", counterparty="counterparty-a")
    second = observation(2, seconds=30, actor="actor-a", counterparty="counterparty-b")
    newly_arrived_prior = observation(
        3,
        seconds=20,
        actor="actor-a",
        counterparty="counterparty-c",
        decision=False,
    )
    state = CausalFeatureState(feature_catalog)
    assert state.process((first,))[0].values["actor_count_1m"] == 0.0

    with pytest.raises(FeatureStateError, match="closed decision timestamp"):
        state.process((newly_arrived_prior, second))

    assert state.process((first,)) == ()


def test_amount_statistics_and_outcomes_use_hand_derived_history(
    feature_catalog: FeatureCatalog,
) -> None:
    """Catches sample deviation, lifecycle omission, or amount double-counting."""
    events = (
        observation(1, seconds=0, amount="10", payment=1),
        observation(
            2,
            seconds=1,
            amount="10",
            event_type=EventKind.AUTHORIZATION_DECLINED,
            decision=False,
            payment=1,
        ),
        observation(3, seconds=10, amount="20", counterparty="counterparty-b", payment=2),
        observation(4, seconds=20, amount="30", counterparty="counterparty-c", payment=3),
    )

    row = CausalFeatureState(feature_catalog).process(events)[-1]

    assert row.values["actor_count_1m"] == 3.0
    assert row.values["actor_amount_1h"] == 30.0
    assert row.values["actor_amount_zscore_24h"] == 3.0
    assert row.values["actor_prior_decline_1h"] == 1.0
    assert row.values["pair_prior_count"] == 0.0


def test_graph_uses_each_payment_opening_once(feature_catalog: FeatureCatalog) -> None:
    """Catches lifecycle events becoming duplicate graph edges."""
    events = (
        observation(1, seconds=0, payment=1),
        observation(
            2,
            seconds=2,
            event_type=EventKind.CLEARING,
            decision=False,
            payment=1,
        ),
        observation(3, seconds=10, counterparty="counterparty-b", payment=2),
        observation(4, seconds=20, counterparty="counterparty-a", payment=3),
    )

    row = CausalFeatureState(feature_catalog).process(events)[-1]

    assert row.values["graph_actor_fanout"] == 2.0
    assert row.values["graph_repeated_edge"] == 1.0
    assert row.values["graph_burst_motif"] == 2.0


def test_null_history_uses_explicit_sentinels_and_quality_indicator(
    feature_catalog: FeatureCatalog,
) -> None:
    """Catches unavailable history silently looking like a genuine zero-duration history."""
    row = CausalFeatureState(feature_catalog).process((observation(1, seconds=0),))[0]

    assert row.values["actor_seconds_since_first"] == -1.0
    assert row.values["actor_amount_zscore_24h"] == -1.0
    assert row.values["dq_mean_history_lag_ms"] == -1.0
    assert row.values["dq_history_age_seconds"] == -1.0
    assert row.values["dq_degraded_state"] == 1.0


def test_checkpoint_restore_reproduces_future_vectors(
    observed_stream: tuple[ObservedEvent, ...],
    feature_catalog: FeatureCatalog,
) -> None:
    """Catches checkpoint omission of pending or admitted causal state."""
    first = CausalFeatureState(feature_catalog)
    before = first.process(observed_stream[:8])
    restored = CausalFeatureState.restore(first.checkpoint(), feature_catalog)

    assert restored.process(observed_stream[8:]) == first.process(observed_stream[8:])
    assert before


def test_checkpoint_is_canonical_and_rejects_corruption_or_catalog_mismatch(
    feature_catalog: FeatureCatalog,
) -> None:
    """Catches accepting mutable, corrupt, or catalog-incompatible state bytes."""
    state = CausalFeatureState(feature_catalog)
    state.process((observation(1, seconds=0),))
    payload = state.checkpoint()
    decoded = json.loads(payload)

    assert payload == json.dumps(decoded, sort_keys=True, separators=(",", ":")).encode()
    with pytest.raises(FeatureStateError, match="checkpoint"):
        CausalFeatureState.restore(payload[:-1] + b"0", feature_catalog)

    changed = feature_catalog.model_copy(update={"version": "1.0.1"})
    with pytest.raises(FeatureStateError, match="catalog"):
        CausalFeatureState.restore(payload, changed)


def test_checkpoint_rejects_internally_inconsistent_state_with_valid_self_digest(
    feature_catalog: FeatureCatalog,
) -> None:
    """Catches trusting the envelope digest without validating state references."""
    state = CausalFeatureState(feature_catalog)
    state.process((observation(1, seconds=0),))
    document = json.loads(state.checkpoint())
    document["late_event_ids"] = ["unknown-event"]
    document["late_event_watermarks"] = {"unknown-event": document["watermark"]}

    with pytest.raises(FeatureStateError, match="checkpoint state"):
        CausalFeatureState.restore(_checkpoint_bytes(document), feature_catalog)


def test_checkpoint_rejects_impossible_late_timing_with_valid_self_digest(
    feature_catalog: FeatureCatalog,
) -> None:
    """Catches accepting a known on-time event falsely relabeled as a late arrival."""
    decision = observation(1, seconds=30)
    on_time = observation(2, seconds=40, decision=False)
    state = CausalFeatureState(feature_catalog)
    state.process((decision, on_time))
    document = json.loads(state.checkpoint())
    document["late_event_ids"] = [on_time.event_id]
    document["late_event_watermarks"] = {on_time.event_id: document["watermark"]}

    with pytest.raises(FeatureStateError, match="checkpoint state"):
        CausalFeatureState.restore(_checkpoint_bytes(document), feature_catalog)


def test_checkpoint_rejects_late_watermark_between_emitted_decisions(
    feature_catalog: FeatureCatalog,
) -> None:
    """Catches accepting an arrival watermark that no represented decision established."""
    first = observation(1, seconds=30)
    late = observation(2, seconds=20, decision=False)
    second = observation(3, seconds=60)
    state = CausalFeatureState(feature_catalog)
    state.process((first,))
    state.process((late, second))
    document = json.loads(state.checkpoint())
    assert document["late_event_watermarks"] == {
        late.event_id: first.decision_at.isoformat()
    }
    document["late_event_watermarks"] = {
        late.event_id: (BASE_TIME + timedelta(seconds=45)).isoformat()
    }

    with pytest.raises(FeatureStateError, match="checkpoint state"):
        CausalFeatureState.restore(_checkpoint_bytes(document), feature_catalog)


def test_checkpoint_rejects_missing_emitted_decision_with_valid_self_digest(
    feature_catalog: FeatureCatalog,
) -> None:
    """Catches a checkpoint that can re-emit an earlier decision after restore."""
    first = observation(1, seconds=0)
    second = observation(2, seconds=30)
    state = CausalFeatureState(feature_catalog)
    state.process((first, second))
    document = json.loads(state.checkpoint())
    document["emitted_decision_ids"] = [second.event_id]

    with pytest.raises(FeatureStateError, match="checkpoint state"):
        CausalFeatureState.restore(_checkpoint_bytes(document), feature_catalog)


def test_repeated_and_permuted_incremental_calls_are_idempotent(
    observed_stream: tuple[ObservedEvent, ...],
    feature_catalog: FeatureCatalog,
) -> None:
    """Catches input-order dependence and double-admission across calls."""
    prefix = tuple(reversed(observed_stream[:8]))
    state = CausalFeatureState(feature_catalog)

    first = state.process(prefix)
    repeated = state.process(prefix)
    future = state.process(tuple(reversed(observed_stream[8:])))

    reference = CausalFeatureState(feature_catalog)
    assert first == reference.process(observed_stream[:8])
    assert repeated == ()
    assert future == reference.process(observed_stream[8:])


def test_all_48_features_match_hand_derived_family_oracles(
    feature_catalog: FeatureCatalog,
    feature_family_scenario: tuple[
        tuple[ObservedEvent, ...], ObservedEvent, ObservedEvent
    ],
) -> None:
    """Catches an untested catalog column drifting from its declared family semantics."""
    history, late, target = feature_family_scenario
    state = CausalFeatureState(feature_catalog)
    state.process(history)

    row = state.process((late, target))[0]

    expected = {
        "txn_log_amount": 4.2626798770413155,
        "txn_rail_card": 0.0,
        "txn_rail_a2a": 0.0,
        "txn_rail_agentic": 1.0,
        "txn_hour_sin": -1.0,
        "txn_hour_cos": 0.0,
        "txn_integrity_pass": 1.0,
        "txn_optional_ref_count": 2.0,
        "actor_count_1m": 0.0,
        "actor_count_10m": 5.0,
        "actor_count_1h": 5.0,
        "actor_count_24h": 5.0,
        "actor_amount_1h": 90.0,
        "actor_amount_24h": 90.0,
        "counterparty_count_1h": 6.0,
        "counterparty_count_24h": 6.0,
        "counterparty_amount_24h": 90.0,
        "actor_prior_decline_1h": 1.0,
        "actor_prior_challenge_1h": 1.0,
        "actor_prior_return_24h": 1.0,
        "counterparty_prior_refund_24h": 1.0,
        "actor_seconds_since_first": 200.0,
        "actor_seconds_since_last": 100.0,
        "counterparty_seconds_since_first": 180.0,
        "counterparty_seconds_since_last": 120.0,
        "pair_seconds_since_first": 160.0,
        "pair_seconds_since_last": 120.0,
        "actor_distinct_counterparties_24h": 3.0,
        "counterparty_distinct_actors_24h": 3.0,
        "actor_amount_zscore_24h": 2.449489742783178,
        "counterparty_amount_zscore_24h": 4.898979485566356,
        "pair_prior_count": 3.0,
        "graph_actor_fanout": 3.0,
        "graph_counterparty_fanin": 3.0,
        "graph_shared_neighbor_count": 1.0,
        "graph_two_hop_reach": 1.0,
        "graph_component_size": 5.0,
        "graph_edge_density": 0.5,
        "graph_repeated_edge": 1.0,
        "graph_burst_motif": 5.0,
        "graph_prior_suspicious_count": 1.0,
        "dq_missing_optional_count": 5.0,
        "dq_current_availability_lag_ms": 0.0,
        "dq_mean_history_lag_ms": 0.0,
        "dq_late_event_count": 1.0,
        "dq_history_count": 8.0,
        "dq_history_age_seconds": 200.0,
        "dq_degraded_state": 1.0,
    }
    assert tuple(expected) == feature_catalog.names
    assert row.values == pytest.approx(expected, abs=1e-12)


def test_rolling_window_includes_exact_cutoff_and_expires_older_history(
    feature_catalog: FeatureCatalog,
) -> None:
    """Catches off-by-one pruning at a rolling window's lower timestamp boundary."""
    target = observation(203, seconds=100)
    exact_cutoff = observation(201, seconds=40, decision=False)
    expired = observation(202, seconds=40, decision=False).model_copy(
        update={
            "event_id": "event-202-expired",
            "event_time": exact_cutoff.event_time - timedelta(microseconds=1),
            "available_at": exact_cutoff.available_at - timedelta(microseconds=1),
        }
    )

    row = CausalFeatureState(feature_catalog).process((expired, target, exact_cutoff))[0]

    assert row.values["actor_count_1m"] == 1.0
    assert row.values["actor_count_10m"] == 2.0
    assert row.values["counterparty_count_1h"] == 2.0
