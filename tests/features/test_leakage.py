"""Behavior tests for forbidden semantic and identity feature provenance."""

from __future__ import annotations

import pytest

from apar.contracts.events import Rail
from apar.features.builders import FeatureMatrix, build_feature_matrix
from apar.features.catalog import FeatureCatalog, FeatureDefinition
from apar.features.parity import FeatureLeakageError, audit_feature_matrix
from tests.features.conftest import observation


@pytest.fixture
def clean_matrix(feature_catalog: FeatureCatalog) -> FeatureMatrix:
    """Return a valid deterministic matrix before any deliberate test-only corruption."""
    events = (
        observation(501, seconds=0, decision=False),
        observation(502, seconds=30),
    )
    return build_feature_matrix(events, feature_catalog)


def matrix_with_injected_feature(
    clean: FeatureMatrix, *, name: str, source_path: str
) -> FeatureMatrix:
    """Construct an invalid matrix directly, without invoking a production audit."""
    injected = FeatureDefinition(
        name=name,
        family="data_quality",
        rails=(Rail.CARD, Rail.A2A, Rail.AGENTIC),
        source_paths=(source_path,),
        missing_behavior="zero",
    )
    invalid_catalog = clean.catalog.model_copy(
        update={"features": (*clean.catalog.features, injected)}
    )
    invalid_rows = tuple(
        row.model_copy(update={"values": {**row.values, name: 1.0}}) for row in clean.rows
    )
    return FeatureMatrix(
        events=clean.events,
        catalog=invalid_catalog,
        catalog_digest=clean.catalog_digest,
        rows=invalid_rows,
    )


@pytest.mark.parametrize(
    "source",
    (
        "truth.is_fraud",
        "lineage.campaign_role",
        "rail_data.hidden_family",
        "party_refs.actor_role",
        "scenario.seed",
        "event.viewpoint",
        "truth.disposition",
        "attack.objective",
        "state.post_decision_outcome",
    ),
)
def test_forbidden_provenance_is_rejected_even_with_safe_name(
    clean_matrix: FeatureMatrix, source: str
) -> None:
    """Catches semantic leakage smuggled behind an innocuous feature name."""
    matrix = matrix_with_injected_feature(
        clean_matrix, name="safe_metric", source_path=source
    )

    with pytest.raises(FeatureLeakageError, match="forbidden feature provenance"):
        audit_feature_matrix(matrix.events, matrix, matrix.catalog)


@pytest.mark.parametrize(
    ("name", "source"),
    (
        ("safe_metric", "observed.actor_id"),
        ("counterparty_id", "observed.amount"),
    ),
)
def test_raw_ids_remain_state_or_provenance_keys_never_model_columns(
    clean_matrix: FeatureMatrix, name: str, source: str
) -> None:
    """Catches raw identities promoted into model values by name or source."""
    matrix = matrix_with_injected_feature(clean_matrix, name=name, source_path=source)

    with pytest.raises(FeatureLeakageError):
        audit_feature_matrix(matrix.events, matrix, matrix.catalog)


def test_audit_uses_the_external_event_argument_not_matrix_embedded_events(
    feature_catalog: FeatureCatalog,
) -> None:
    """Catches an audit resolving provenance against the matrix's trusted internals."""
    prior = observation(510, seconds=0, decision=False)
    decision = observation(511, seconds=30)
    matrix = build_feature_matrix((prior, decision), feature_catalog)

    with pytest.raises(FeatureLeakageError, match="source event IDs do not resolve"):
        audit_feature_matrix((decision,), matrix, feature_catalog)
