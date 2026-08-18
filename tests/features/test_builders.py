"""Behavior tests for one-shot feature matrix construction."""

from __future__ import annotations

import pytest

from apar.contracts.events import Rail
from apar.defense.contracts import ObservedEvent
from apar.features.builders import build_feature_matrix
from apar.features.catalog import FeatureCatalog
from tests.features.conftest import observation


def test_matrix_has_stable_decision_order_and_exact_catalog_columns(
    observed_stream: tuple[ObservedEvent, ...],
    feature_catalog: FeatureCatalog,
) -> None:
    """Catches input permutation or ad-hoc feature derivation changing model bytes."""
    matrix = build_feature_matrix(tuple(reversed(observed_stream)), feature_catalog)

    expected_ids = tuple(
        event.event_id
        for event in sorted(
            (event for event in observed_stream if event.is_decision_point),
            key=lambda event: (event.decision_at, event.event_id),
        )
    )
    assert tuple(row.event_id for row in matrix.rows) == expected_ids
    assert all(tuple(row.values) == feature_catalog.names for row in matrix.rows)
    assert matrix.catalog == feature_catalog
    assert matrix.events == tuple(sorted(observed_stream, key=lambda event: event.event_id))


def test_current_request_identity_never_enters_model_values(
    feature_catalog: FeatureCatalog,
    equal_time_observations: tuple[ObservedEvent, ...],
) -> None:
    """Catches raw provenance keys accidentally emitted as feature columns."""
    matrix = build_feature_matrix(equal_time_observations, feature_catalog)

    forbidden = {"event_id", "payment_id", "actor_id", "counterparty_id"}
    assert all(forbidden.isdisjoint(row.values) for row in matrix.rows)
    assert all(len(row.values) == 48 for row in matrix.rows)


@pytest.mark.parametrize(
    ("rail", "integrity", "expected"),
    (
        (Rail.CARD, "not_applicable", (1.0, 0.0, 0.0, 0.0)),
        (Rail.A2A, "not_applicable", (0.0, 1.0, 0.0, 0.0)),
        (Rail.AGENTIC, "pass", (0.0, 0.0, 1.0, 1.0)),
        (Rail.AGENTIC, "fail", (0.0, 0.0, 1.0, 0.0)),
    ),
)
def test_rail_and_integrity_columns_have_hand_derived_indicators(
    feature_catalog: FeatureCatalog,
    rail: Rail,
    integrity: str,
    expected: tuple[float, float, float, float],
) -> None:
    """Catches one-hot or agentic-integrity semantics being inferred from another rail."""
    event = observation(
        301,
        seconds=21_600,
        rail=rail,
        integrity_status=integrity,
    )

    values = build_feature_matrix((event,), feature_catalog).rows[0].values

    assert (
        values["txn_rail_card"],
        values["txn_rail_a2a"],
        values["txn_rail_agentic"],
        values["txn_integrity_pass"],
    ) == expected
    assert values["txn_hour_sin"] == pytest.approx(-1.0)
    assert values["txn_hour_cos"] == pytest.approx(0.0, abs=1e-12)
