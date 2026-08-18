"""Independent feature-matrix provenance and train-serving parity tests."""

from __future__ import annotations

from datetime import timedelta

import pytest

from apar.defense.contracts import ObservedEvent
from apar.features.builders import FeatureMatrix, build_feature_matrix
from apar.features.catalog import FeatureCatalog
from apar.features.parity import (
    FeatureLeakageError,
    assert_online_offline_parity,
    audit_feature_matrix,
)
from tests.features.conftest import observation


def test_clean_matrix_passes_independent_catalog_and_provenance_audit(
    observed_stream: tuple[ObservedEvent, ...],
    feature_catalog: FeatureCatalog,
) -> None:
    """Catches an audit that cannot validate a real causal matrix end to end."""
    matrix = build_feature_matrix(observed_stream, feature_catalog)

    report = audit_feature_matrix(observed_stream, matrix, feature_catalog)

    assert report.passed
    assert report.catalog_valid
    assert report.strictly_past_only
    assert report.source_ids_resolve
    assert report.feature_order_matches
    assert report.forbidden_sources == ()


def test_audit_rebuilds_source_availability_from_supplied_observations(
    observed_stream: tuple[ObservedEvent, ...],
    feature_catalog: FeatureCatalog,
) -> None:
    """Catches trusting feature-state provenance without resolving it independently."""
    matrix = build_feature_matrix(observed_stream, feature_catalog)
    historical_row = next(row for row in matrix.rows if row.source_event_ids)
    missing_source = historical_row.source_event_ids[0]
    incomplete_events = tuple(
        event for event in observed_stream if event.event_id != missing_source
    )

    with pytest.raises(FeatureLeakageError, match="source event IDs do not resolve"):
        audit_feature_matrix(incomplete_events, matrix, feature_catalog)


def test_audit_rejects_equal_time_historical_provenance(
    feature_catalog: FeatureCatalog,
) -> None:
    """Catches weakening strict past-only provenance to a less-than-or-equal boundary."""
    event = observation(401, seconds=30)
    matrix = build_feature_matrix((event,), feature_catalog)
    row = matrix.rows[0].model_copy(
        update={
            "source_event_ids": (event.event_id,),
            "max_source_available_at": event.decision_at,
        }
    )
    invalid = matrix.model_copy(update={"rows": (row,)})

    with pytest.raises(FeatureLeakageError, match="strictly before"):
        audit_feature_matrix((event,), invalid, feature_catalog)


def test_audit_rejects_forged_catalog_digest_and_column_order(
    observed_stream: tuple[ObservedEvent, ...],
    feature_catalog: FeatureCatalog,
) -> None:
    """Catches accepting a matrix whose digest or ordered model contract was changed."""
    matrix = build_feature_matrix(observed_stream, feature_catalog)
    first = matrix.rows[0]
    reordered_values = dict(reversed(tuple(first.values.items())))
    invalid = FeatureMatrix(
        events=matrix.events,
        catalog=matrix.catalog,
        catalog_digest="0" * 64,
        rows=(first.model_copy(update={"values": reordered_values}), *matrix.rows[1:]),
    )

    with pytest.raises(FeatureLeakageError, match="catalog digest or feature order"):
        audit_feature_matrix(observed_stream, invalid, feature_catalog)


def test_online_offline_parity_batches_complete_equal_time_cohorts(
    feature_catalog: FeatureCatalog,
) -> None:
    """Catches online replay splitting a timestamp cohort or admitting its peers."""
    events = (
        observation(410, seconds=0, decision=False),
        observation(411, seconds=30, counterparty="counterparty-b"),
        observation(412, seconds=30, counterparty="counterparty-c"),
        observation(413, seconds=60),
    )

    assert_online_offline_parity(tuple(reversed(events)), feature_catalog)

    rows = build_feature_matrix(events, feature_catalog).rows
    assert [row.values["actor_count_1m"] for row in rows] == [1.0, 1.0, 3.0]
    assert rows[0].decision_at == rows[1].decision_at
    assert rows[0].source_event_ids == ("event-410",)
    assert rows[1].source_event_ids == ("event-410",)
    assert rows[2].max_source_available_at == rows[0].decision_at


def test_audit_rejects_a_forged_maximum_source_timestamp(
    feature_catalog: FeatureCatalog,
) -> None:
    """Catches provenance summaries that do not match independently resolved sources."""
    prior = observation(420, seconds=0, decision=False)
    decision = observation(421, seconds=30)
    matrix = build_feature_matrix((prior, decision), feature_catalog)
    row = matrix.rows[0].model_copy(
        update={"max_source_available_at": prior.available_at + timedelta(seconds=1)}
    )

    with pytest.raises(FeatureLeakageError, match="maximum source availability"):
        audit_feature_matrix(
            (prior, decision), matrix.model_copy(update={"rows": (row,)}), feature_catalog
        )


def test_audit_rejects_matrix_event_binding_drift(
    feature_catalog: FeatureCatalog,
) -> None:
    """Catches a matrix envelope claiming observations other than the audited inputs."""
    prior = observation(430, seconds=0, decision=False)
    decision = observation(431, seconds=30)
    matrix = build_feature_matrix((prior, decision), feature_catalog)
    invalid = matrix.model_copy(update={"events": (decision,)})

    with pytest.raises(FeatureLeakageError, match="catalog digest or feature order"):
        audit_feature_matrix((prior, decision), invalid, feature_catalog)
