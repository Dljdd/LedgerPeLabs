"""One-shot construction of deterministic causal feature matrices."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

from apar.contracts._validation import ExternalContract
from apar.defense.contracts import ObservedEvent
from apar.features.catalog import FeatureCatalog
from apar.features.state import CausalFeatureState, FeatureVector, feature_catalog_digest


class FeatureMatrix(ExternalContract):
    """Observations, catalog, and rows bound into one auditable matrix."""

    schema_version: Literal["1.0.0"] = "1.0.0"
    events: tuple[ObservedEvent, ...]
    catalog: FeatureCatalog
    catalog_digest: str
    rows: tuple[FeatureVector, ...]


def build_feature_matrix(
    events: Sequence[ObservedEvent], catalog: FeatureCatalog
) -> FeatureMatrix:
    """Build a stable one-shot matrix from arbitrarily ordered observations."""
    ordered_events = tuple(sorted(events, key=lambda event: event.event_id))
    state = CausalFeatureState(catalog)
    rows = state.process(ordered_events)
    return FeatureMatrix(
        events=ordered_events,
        catalog=catalog,
        catalog_digest=feature_catalog_digest(catalog),
        rows=rows,
    )
