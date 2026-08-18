"""Behavioral tests for the closed decision-time feature catalog."""

from pathlib import Path

import pytest

from apar.contracts.events import Rail
from apar.features.catalog import (
    FeatureCatalog,
    FeatureCatalogError,
    FeatureDefinition,
    audit_feature_catalog,
    load_feature_catalog,
)

FEATURE_CATALOG_PATH = Path("config/defense/feature-catalog.json")
EXPECTED_FEATURE_NAMES = (
    "txn_log_amount",
    "txn_rail_card",
    "txn_rail_a2a",
    "txn_rail_agentic",
    "txn_hour_sin",
    "txn_hour_cos",
    "txn_integrity_pass",
    "txn_optional_ref_count",
    "actor_count_1m",
    "actor_count_10m",
    "actor_count_1h",
    "actor_count_24h",
    "actor_amount_1h",
    "actor_amount_24h",
    "counterparty_count_1h",
    "counterparty_count_24h",
    "counterparty_amount_24h",
    "actor_prior_decline_1h",
    "actor_prior_challenge_1h",
    "actor_prior_return_24h",
    "counterparty_prior_refund_24h",
    "actor_seconds_since_first",
    "actor_seconds_since_last",
    "counterparty_seconds_since_first",
    "counterparty_seconds_since_last",
    "pair_seconds_since_first",
    "pair_seconds_since_last",
    "actor_distinct_counterparties_24h",
    "counterparty_distinct_actors_24h",
    "actor_amount_zscore_24h",
    "counterparty_amount_zscore_24h",
    "pair_prior_count",
    "graph_actor_fanout",
    "graph_counterparty_fanin",
    "graph_shared_neighbor_count",
    "graph_two_hop_reach",
    "graph_component_size",
    "graph_edge_density",
    "graph_repeated_edge",
    "graph_burst_motif",
    "graph_prior_suspicious_count",
    "dq_missing_optional_count",
    "dq_current_availability_lag_ms",
    "dq_mean_history_lag_ms",
    "dq_late_event_count",
    "dq_history_count",
    "dq_history_age_seconds",
    "dq_degraded_state",
)


def _definition(
    *, name: str, source_paths: tuple[str, ...], state_keys: tuple[str, ...] = ()
) -> FeatureDefinition:
    return FeatureDefinition(
        name=name,
        family="temporal",
        rails=(Rail.CARD,),
        source_paths=source_paths,
        state_keys=state_keys,
        window_seconds=3600,
        missing_behavior="zero",
    )


def test_catalog_has_the_exact_ordered_competition_features() -> None:
    """Catches order drift that would change the trained model's column contract."""
    catalog = load_feature_catalog(FEATURE_CATALOG_PATH)

    assert catalog.names == EXPECTED_FEATURE_NAMES
    assert len(catalog.names) == len(set(catalog.names)) == 48


def test_catalog_declarations_are_audited_as_observed_or_strictly_past_state() -> None:
    """Catches a catalog row whose declared origin escapes the observation boundary."""
    catalog = load_feature_catalog(FEATURE_CATALOG_PATH)

    audit_feature_catalog(catalog)
    assert all(definition.rails for definition in catalog.features)
    assert all(definition.source_paths for definition in catalog.features)
    assert all(definition.missing_behavior for definition in catalog.features)


@pytest.mark.parametrize(
    "source",
    (
        "truth.is_fraud",
        "lineage.campaign_role",
        "rail_data.hidden_family",
        "party_refs.actor_role",
        "scenario.seed",
        "event.viewpoint",
    ),
)
def test_semantically_forbidden_source_is_rejected_after_an_innocent_rename(source: str) -> None:
    """Catches truth or generator provenance smuggled behind a harmless feature name."""
    definition = _definition(name="ordinary_count", source_paths=(source,))

    with pytest.raises(FeatureCatalogError, match="forbidden source"):
        audit_feature_catalog(FeatureCatalog(version="1.0.0", features=(definition,)))


def test_raw_entity_id_cannot_become_a_model_column() -> None:
    """Catches an identity column promoted from an observation into a model feature."""
    definition = _definition(name="ordinary_count", source_paths=("observed.actor_id",))

    with pytest.raises(FeatureCatalogError, match="raw ID"):
        audit_feature_catalog(FeatureCatalog(version="1.0.0", features=(definition,)))


def test_ids_are_permitted_only_for_state_keys_or_provenance_references() -> None:
    """Catches an audit that blocks safe identity use needed for history and traceability."""
    definition = _definition(
        name="ordinary_count",
        source_paths=("state.past_observed_events", "provenance.source_event_ids"),
        state_keys=("observed.actor_id", "observed.counterparty_id"),
    )

    audit_feature_catalog(FeatureCatalog(version="1.0.0", features=(definition,)))


def test_unknown_state_path_is_rejected_as_not_provably_past_observed_state() -> None:
    """Catches an undeclared state origin that cannot establish its causal source."""
    definition = _definition(name="ordinary_count", source_paths=("state.external_enrichment",))

    with pytest.raises(FeatureCatalogError, match="unapproved source"):
        audit_feature_catalog(FeatureCatalog(version="1.0.0", features=(definition,)))


@pytest.mark.parametrize("source", ("observed.decision_at", "observed.currency"))
def test_unlisted_current_observation_source_is_rejected(source: str) -> None:
    """Catches a visible field becoming a model input without catalog justification."""
    definition = _definition(name="ordinary_count", source_paths=(source,))

    with pytest.raises(FeatureCatalogError, match="unapproved source"):
        audit_feature_catalog(FeatureCatalog(version="1.0.0", features=(definition,)))
