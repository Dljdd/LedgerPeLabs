"""Behavioral metamorphic gates for causal feature construction."""

from __future__ import annotations

import json
import math
from decimal import Decimal

from apar.defense.contracts import ObservedEvent
from apar.features.builders import build_feature_matrix
from apar.features.catalog import FeatureCatalog
from apar.features.parity import (
    assert_checkpoint_restart_invariant,
    assert_duplicate_event_ids_invariant,
    assert_economic_scaling_invariant,
    assert_equal_time_permutation_invariant,
    assert_future_append_invariant,
    assert_missing_optional_references_invariant,
    assert_row_permutation_invariant,
    assert_synthetic_id_bijection_invariant,
)
from apar.features.state import FeatureVector
from tests.features.conftest import observation


def _value_bytes(event: FeatureVector) -> bytes:
    return json.dumps(event.values, separators=(",", ":"), allow_nan=False).encode()


def _provenance_bytes(event: FeatureVector) -> bytes:
    payload = {
        "source_event_ids": event.source_event_ids,
        "max_source_available_at": (
            event.max_source_available_at.isoformat()
            if event.max_source_available_at is not None
            else None
        ),
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()


def test_future_append_leaves_prior_model_and_provenance_bytes_identical(
    observed_stream: tuple[ObservedEvent, ...],
    feature_catalog: FeatureCatalog,
) -> None:
    """Catches a future append mutating an already materialized feature or lineage byte."""
    prefix = observed_stream[:8]
    future = observed_stream[8:]
    before = build_feature_matrix(prefix, feature_catalog)
    after = build_feature_matrix((*prefix, *future), feature_catalog)

    assert_future_append_invariant(prefix, future, feature_catalog)
    after_by_id = {row.event_id: row for row in after.rows}
    assert tuple(_value_bytes(after_by_id[row.event_id]) for row in before.rows) == tuple(
        _value_bytes(row) for row in before.rows
    )
    assert tuple(_provenance_bytes(after_by_id[row.event_id]) for row in before.rows) == tuple(
        _provenance_bytes(row) for row in before.rows
    )


def test_row_and_equal_time_permutations_preserve_stably_ordered_rows(
    observed_stream: tuple[ObservedEvent, ...],
    feature_catalog: FeatureCatalog,
) -> None:
    """Catches input position or peer order affecting a causal output row."""
    equal_time = (
        observation(601, seconds=300, counterparty="counterparty-x"),
        observation(602, seconds=300, counterparty="counterparty-y"),
    )

    assert_row_permutation_invariant(observed_stream, feature_catalog)
    assert_equal_time_permutation_invariant(equal_time, feature_catalog)

    rows = build_feature_matrix(tuple(reversed(equal_time)), feature_catalog).rows
    assert tuple(row.event_id for row in rows) == ("event-601", "event-602")
    assert [row.values["actor_count_1m"] for row in rows] == [0.0, 0.0]


def test_consistent_synthetic_id_bijection_preserves_values_and_maps_provenance(
    observed_stream: tuple[ObservedEvent, ...],
    feature_catalog: FeatureCatalog,
) -> None:
    """Catches raw ID values entering the model or provenance failing to rename consistently."""
    assert_synthetic_id_bijection_invariant(observed_stream, feature_catalog)


def test_duplicate_event_ids_are_idempotent(
    observed_stream: tuple[ObservedEvent, ...],
    feature_catalog: FeatureCatalog,
) -> None:
    """Catches duplicate delivery double-counting historical observations."""
    assert_duplicate_event_ids_invariant(observed_stream, feature_catalog)

    baseline = build_feature_matrix(observed_stream, feature_catalog)
    duplicated = build_feature_matrix(
        (*observed_stream, observed_stream[0], observed_stream[1]), feature_catalog
    )
    assert duplicated.rows == baseline.rows


def test_missing_optional_references_change_only_hand_derived_quality_columns(
    feature_catalog: FeatureCatalog,
) -> None:
    """Catches optional enrichment leaking into unrelated model families."""
    event = observation(
        610,
        seconds=0,
        optional_refs={"device_id": "device-1", "merchant_id": "merchant-1"},
    )
    without = event.model_copy(update={"optional_refs": {}})

    assert_missing_optional_references_invariant((event,), feature_catalog)
    with_values = build_feature_matrix((event,), feature_catalog).rows[0].values
    without_values = build_feature_matrix((without,), feature_catalog).rows[0].values
    assert with_values["txn_optional_ref_count"] == 2.0
    assert with_values["dq_missing_optional_count"] == 5.0
    assert without_values["txn_optional_ref_count"] == 0.0
    assert without_values["dq_missing_optional_count"] == 7.0


def test_consistent_economic_scaling_has_hand_derived_feature_relationships(
    feature_catalog: FeatureCatalog,
) -> None:
    """Catches inconsistent scaling of current, aggregate, or standardized amounts."""
    events = (
        observation(620, seconds=0, amount="10"),
        observation(621, seconds=30, amount="20"),
    )
    scaled = tuple(
        event.model_copy(update={"amount": event.amount * Decimal("10")})
        for event in events
    )

    assert_economic_scaling_invariant(events, feature_catalog, factor=Decimal("10"))
    original_row = build_feature_matrix(events, feature_catalog).rows[-1]
    scaled_row = build_feature_matrix(scaled, feature_catalog).rows[-1]
    assert original_row.values["actor_amount_1h"] == 10.0
    assert scaled_row.values["actor_amount_1h"] == 100.0
    assert original_row.values["actor_amount_zscore_24h"] == 0.0
    assert scaled_row.values["actor_amount_zscore_24h"] == 0.0
    assert original_row.values["txn_log_amount"] == math.log1p(20.0)
    assert scaled_row.values["txn_log_amount"] == math.log1p(200.0)


def test_checkpoint_restart_matches_uninterrupted_complete_cohorts(
    observed_stream: tuple[ObservedEvent, ...],
    feature_catalog: FeatureCatalog,
) -> None:
    """Catches restart omission while respecting the closed decision-time policy."""
    assert_checkpoint_restart_invariant(
        observed_stream[:8], observed_stream[8:], feature_catalog
    )
