"""The closed, decision-time feature allowlist and semantic audit."""

from __future__ import annotations

import json
from enum import StrEnum
from pathlib import Path
from typing import Literal

from pydantic import Field, ValidationError, field_validator

from apar.contracts._validation import ExternalContract, validate_semantic_version
from apar.contracts.events import Rail
from apar.defense.contracts import ObservedEvent

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

_FORBIDDEN_SEMANTICS = (
    "fraud",
    "illicit",
    "label",
    "target",
    "campaign",
    "family",
    "scenario",
    "regime",
    "seed",
    "generator",
    "hidden",
    "policy",
    "role",
    "viewpoint",
    "chargeback_truth",
    "post_decision",
)
_RAW_ID_FIELDS = frozenset({"event_id", "payment_id", "actor_id", "counterparty_id"})
_ALLOWED_STATE_PATHS = frozenset(
    {"state.past_observed_events", "state.past_observed_graph", "state.late_observed_events"}
)
_ALLOWED_PROVENANCE_PATHS = frozenset({"provenance.source_event_ids"})


class FeatureCatalogError(ValueError):
    """Raised when a catalog reaches outside the defender-visible feature boundary."""


class FeatureSource(StrEnum):
    """The only declared origins from which a feature can be derived."""

    OBSERVED = "observed"
    STATE = "state"
    PROVENANCE = "provenance"


class FeatureDefinition(ExternalContract):
    """One ordered model column and the inputs used to derive it."""

    name: str
    family: Literal["transaction", "temporal", "entity", "graph", "data_quality"]
    rails: tuple[Rail, ...]
    source_paths: tuple[str, ...]
    state_keys: tuple[str, ...] = ()
    window_seconds: int | None = Field(default=None, ge=1)
    missing_behavior: Literal["zero", "sentinel", "indicator"]


class FeatureCatalog(ExternalContract):
    """Versioned, ordered model-feature contract."""

    version: str
    features: tuple[FeatureDefinition, ...]

    @field_validator("version")
    @classmethod
    def version_is_semantic(cls, value: str) -> str:
        return validate_semantic_version(value, field_name="version")

    @property
    def names(self) -> tuple[str, ...]:
        """Return the immutable model-column order."""
        return tuple(feature.name for feature in self.features)


def load_feature_catalog(path: Path) -> FeatureCatalog:
    """Load the sole competition catalog and enforce its frozen model-column order."""
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
        catalog = FeatureCatalog.model_validate(document)
    except (OSError, json.JSONDecodeError, ValidationError) as error:
        raise FeatureCatalogError(f"unable to load feature catalog: {error}") from error

    audit_feature_catalog(catalog)
    if catalog.names != EXPECTED_FEATURE_NAMES:
        raise FeatureCatalogError("catalog feature names must match the frozen ordered allowlist")
    return catalog


def audit_feature_catalog(catalog: FeatureCatalog) -> None:
    """Reject semantics or identities that cannot safely enter a model feature column."""
    if not catalog.features:
        raise FeatureCatalogError("catalog must contain at least one feature")
    if len(catalog.names) != len(set(catalog.names)):
        raise FeatureCatalogError("catalog feature names must be unique")

    for definition in catalog.features:
        _audit_definition(definition)


def _audit_definition(definition: FeatureDefinition) -> None:
    if not definition.name:
        raise FeatureCatalogError("feature name must not be empty")
    _reject_forbidden(definition.name, kind="feature name")
    if _is_raw_id_path(definition.name):
        raise FeatureCatalogError(f"raw ID feature name is not permitted: {definition.name}")
    if not definition.rails:
        raise FeatureCatalogError(f"feature {definition.name} must declare at least one rail")
    if len(definition.rails) != len(set(definition.rails)):
        raise FeatureCatalogError(f"feature {definition.name} declares a rail more than once")
    if not definition.source_paths:
        raise FeatureCatalogError(f"feature {definition.name} must declare at least one source")

    for source_path in definition.source_paths:
        _audit_source_path(source_path)
    for state_key in definition.state_keys:
        if state_key not in {f"observed.{name}" for name in _RAW_ID_FIELDS}:
            raise FeatureCatalogError(f"unapproved state key: {state_key}")


def _audit_source_path(source_path: str) -> None:
    _reject_forbidden(source_path, kind="source")
    if source_path in _ALLOWED_PROVENANCE_PATHS:
        return
    if _is_raw_id_path(source_path):
        raise FeatureCatalogError(f"raw ID source is not permitted: {source_path}")

    if source_path.startswith(f"{FeatureSource.OBSERVED}."):
        field_name = source_path.removeprefix(f"{FeatureSource.OBSERVED}.")
        if field_name in ObservedEvent.model_fields and field_name not in _RAW_ID_FIELDS:
            return
    elif source_path in _ALLOWED_STATE_PATHS:
        return
    raise FeatureCatalogError(f"unapproved source path: {source_path}")


def _reject_forbidden(value: str, *, kind: str) -> None:
    normalized = value.lower()
    if any(forbidden in normalized for forbidden in _FORBIDDEN_SEMANTICS):
        raise FeatureCatalogError(f"forbidden {kind}: {value}")


def _is_raw_id_path(value: str) -> bool:
    normalized = value.lower().replace(".", "_")
    return any(
        normalized == identifier or normalized.endswith(f"_{identifier}")
        for identifier in _RAW_ID_FIELDS
    )
