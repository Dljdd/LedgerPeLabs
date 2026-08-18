"""Closed feature definitions for the defense pipeline."""

from apar.features.builders import FeatureMatrix, build_feature_matrix
from apar.features.catalog import (
    FeatureCatalog,
    FeatureCatalogError,
    FeatureDefinition,
    FeatureSource,
    audit_feature_catalog,
    load_feature_catalog,
)
from apar.features.state import (
    CausalFeatureState,
    FeatureStateError,
    FeatureVector,
    feature_catalog_digest,
)

__all__ = [
    "FeatureCatalog",
    "FeatureCatalogError",
    "FeatureDefinition",
    "FeatureMatrix",
    "FeatureSource",
    "FeatureStateError",
    "FeatureVector",
    "CausalFeatureState",
    "audit_feature_catalog",
    "build_feature_matrix",
    "feature_catalog_digest",
    "load_feature_catalog",
]
