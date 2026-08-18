"""Signed, content-addressed, native-only frozen defender bundles."""

from __future__ import annotations

import base64
import binascii
import hashlib
import importlib.metadata
import math
import os
import platform as platform_module
import re
import stat
import sys
import sysconfig
import unicodedata
from collections.abc import Callable, Sequence
from contextlib import suppress
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from pathlib import Path, PurePosixPath
from typing import Literal, NamedTuple, cast
from uuid import UUID

import catboost  # type: ignore[import-untyped]
import cryptography
import numpy as np
import pandas  # type: ignore[import-untyped]
import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]
import pydantic
import sklearn  # type: ignore[import-untyped]
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
from pydantic import Field, ValidationError, field_validator, model_validator

from apar.contracts._validation import ExternalContract, validate_utc_timestamp
from apar.contracts.decisions import Action
from apar.defense.calibration import ProbabilityCalibrator, select_calibrator
from apar.defense.contracts import ObservedEvent
from apar.defense.gbdt import CatBoostScorer, TrainingReceipt
from apar.defense.rules import RuleManifest, rule_manifest_digest
from apar.defense.thresholds import ThresholdReport, select_policy_thresholds
from apar.evaluation.splits import EvaluationSplit
from apar.features.builders import FeatureMatrix
from apar.features.catalog import (
    EXPECTED_FEATURE_NAMES,
    FeatureCatalog,
    audit_feature_catalog,
)
from apar.features.state import FeatureVector, feature_catalog_digest
from apar.runs.runner import RunSigningIdentity
from apar.runs.wire import canonical_json_bytes, strict_json_loads
from apar.storage.artifacts import ArtifactRef, ArtifactStore

GENESIS_ROLLBACK_REF = "genesis"
_SCHEMA_VERSION = "1.0.0"
_SHA256_LENGTH = 64
_MAX_ROLLBACK_DEPTH = 32
_MAX_ROLLBACK_MANIFEST_BYTES = 8 * 1024 * 1024
_MAX_BUNDLE_BYTES = 1024 * 1024
_MAX_JSON_BYTES = 16 * 1024 * 1024
_MAX_MODEL_BYTES = 128 * 1024 * 1024
_MAX_PARQUET_BYTES = 128 * 1024 * 1024
_MAX_PARQUET_DECODED_BYTES = 256 * 1024 * 1024
_MAX_PARQUET_ROWS = 1_000_000
_MAX_PARQUET_ROW_GROUPS = 1

_BUNDLE_MEDIA = "application/vnd.apar.defender-bundle+json"
_MODEL_MEDIA = "application/vnd.apar.catboost-model"
_PARQUET_MEDIA = "application/vnd.apache.parquet"
_CATALOG_MEDIA = "application/vnd.apar.feature-catalog+json"
_RULE_MEDIA = "application/vnd.apar.rule-manifest+json"
_RECEIPT_MEDIA = "application/vnd.apar.training-receipt+json"
_CALIBRATION_MEDIA = "application/vnd.apar.calibration+json"
_THRESHOLD_MEDIA = "application/vnd.apar.threshold-report+json"
_ENVIRONMENT_MEDIA = "application/vnd.apar.environment-lock+json"
_SOURCE_MEDIA = "application/vnd.apar.source-inventory+json"
_RELOAD_MEDIA = "application/vnd.apar.reload-fixture+json"
_SPLIT_MEDIA = "application/vnd.apar.evaluation-split+json"
_TRAINING_BINDING_MEDIA = "application/vnd.apar.training-binding+json"
_CALIBRATION_BINDING_MEDIA = "application/vnd.apar.calibration-binding+json"
_THRESHOLD_BINDING_MEDIA = "application/vnd.apar.threshold-binding+json"
_CALLBACK_CONTRACT_VERSION = "truth-blind-intervention-mask-v1"
_COMPONENT_FIELD_MEDIA: dict[str, tuple[str, str]] = {
    "calibration": ("calibration_digest", _CALIBRATION_MEDIA),
    "calibration_binding": ("calibration_binding_digest", _CALIBRATION_BINDING_MEDIA),
    "calibration_fit_matrix": ("calibration_fit_matrix_digest", _PARQUET_MEDIA),
    "calibration_selection_matrix": ("calibration_selection_matrix_digest", _PARQUET_MEDIA),
    "catalog": ("feature_catalog_digest", _CATALOG_MEDIA),
    "environment": ("environment_digest", _ENVIRONMENT_MEDIA),
    "model": ("model_digest", _MODEL_MEDIA),
    "receipt": ("training_receipt_digest", _RECEIPT_MEDIA),
    "reload_fixture": ("reload_fixture_digest", _RELOAD_MEDIA),
    "reload_matrix": ("reload_matrix_digest", _PARQUET_MEDIA),
    "rules": ("rule_manifest_digest", _RULE_MEDIA),
    "source_inventory": ("source_inventory_digest", _SOURCE_MEDIA),
    "split": ("split_artifact_digest", _SPLIT_MEDIA),
    "threshold": ("threshold_digest", _THRESHOLD_MEDIA),
    "threshold_binding": ("threshold_binding_digest", _THRESHOLD_BINDING_MEDIA),
    "threshold_matrix": ("threshold_matrix_digest", _PARQUET_MEDIA),
    "training_binding": ("training_binding_digest", _TRAINING_BINDING_MEDIA),
    "training_matrix": ("training_matrix_digest", _PARQUET_MEDIA),
}


class BundleContractError(ValueError):
    """A bundle cannot be safely published, verified, or loaded."""


def _digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _validate_digest(value: object, *, label: str = "digest") -> str:
    if type(value) is not str or len(value) != _SHA256_LENGTH:
        raise ValueError(f"{label} must be lowercase SHA-256")
    if value != value.lower() or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{label} must be lowercase SHA-256")
    return value


class BundleLineage(ExternalContract):
    """Upstream immutable evidence required by the frozen defender."""

    schema_version: Literal["1.0.0"] = "1.0.0"
    corpus_digest: str
    observation_dataset_digest: str
    evaluator_truth_digest: str
    split_manifest_digest: str
    feature_provenance_digest: str
    hyperparameter_digest: str
    reason_code_mapping_digest: str

    @field_validator(
        "corpus_digest",
        "observation_dataset_digest",
        "evaluator_truth_digest",
        "split_manifest_digest",
        "feature_provenance_digest",
        "hyperparameter_digest",
        "reason_code_mapping_digest",
    )
    @classmethod
    def digests_are_sha256(cls, value: str) -> str:
        return _validate_digest(value, label="lineage digest")

    @model_validator(mode="after")
    def lineage_is_distinct(self) -> BundleLineage:
        values = tuple(
            value for name, value in self.model_dump().items() if name != "schema_version"
        )
        if len(values) != len(set(values)):
            raise ValueError("lineage digests must identify distinct artifacts")
        return self


class InstalledDistribution(ExternalContract):
    """One normalized installed Python distribution identity."""

    name: str
    version: str

    @field_validator("name")
    @classmethod
    def name_is_normalized(cls, value: str) -> str:
        if type(value) is not str or value != _normalized_distribution_name(value):
            raise ValueError("distribution name must be normalized")
        return value

    @field_validator("version")
    @classmethod
    def version_is_exact_text(cls, value: str) -> str:
        if type(value) is not str or not value or value.strip() != value:
            raise ValueError("distribution version must be exact nonblank text")
        return value


class EnvironmentLock(ExternalContract):
    """Exact loader environment; portability outside it is not claimed."""

    schema_version: Literal["1.0.0"] = "1.0.0"
    python_version: str
    python_implementation: str
    python_cache_tag: str
    python_soabi: str
    byteorder: Literal["little", "big"]
    machine: str
    platform: str
    catboost_version: str
    scikit_learn_version: str
    numpy_version: str
    pyarrow_version: str
    pydantic_version: str
    cryptography_version: str
    pandas_version: str
    installed_distributions: tuple[InstalledDistribution, ...]
    apar_schema_version: Literal["1.0.0"] = "1.0.0"

    @field_validator(
        "python_version",
        "python_implementation",
        "python_cache_tag",
        "python_soabi",
        "machine",
        "platform",
        "catboost_version",
        "scikit_learn_version",
        "numpy_version",
        "pyarrow_version",
        "pydantic_version",
        "cryptography_version",
        "pandas_version",
    )
    @classmethod
    def versions_are_exact_text(cls, value: str) -> str:
        if type(value) is not str or not value or value.strip() != value:
            raise ValueError("environment identity fields must be exact nonblank text")
        return value

    @field_validator("installed_distributions")
    @classmethod
    def distributions_are_sorted_unique(
        cls, value: tuple[InstalledDistribution, ...]
    ) -> tuple[InstalledDistribution, ...]:
        if not value:
            raise ValueError("installed distribution inventory must not be empty")
        names = tuple(item.name for item in value)
        if names != tuple(sorted(names)) or len(names) != len(set(names)):
            raise ValueError("installed distributions must be normalized, unique, and sorted")
        return value


def _normalized_distribution_name(value: str) -> str:
    return re.sub(r"[-_.]+", "-", value).lower()


def _installed_distributions() -> tuple[InstalledDistribution, ...]:
    versions: dict[str, str] = {}
    for distribution in importlib.metadata.distributions():
        raw_name = distribution.metadata.get("Name")
        if type(raw_name) is not str or not raw_name:
            raise BundleContractError("installed distribution has no canonical name")
        name = _normalized_distribution_name(raw_name)
        version = distribution.version
        if name in versions:
            raise BundleContractError(f"duplicate installed distribution identity: {name}")
        versions[name] = version
    return tuple(
        InstalledDistribution(name=name, version=versions[name]) for name in sorted(versions)
    )


def current_environment_lock() -> EnvironmentLock:
    """Return the exact runtime, ABI, and dependency inventory accepted by this loader."""
    cache_tag = sys.implementation.cache_tag
    soabi = sysconfig.get_config_var("SOABI")
    if type(cache_tag) is not str or not cache_tag or type(soabi) is not str or not soabi:
        raise BundleContractError("Python ABI identity is unavailable")
    return EnvironmentLock(
        python_version=platform_module.python_version(),
        python_implementation=platform_module.python_implementation(),
        python_cache_tag=cache_tag,
        python_soabi=soabi,
        byteorder=sys.byteorder,
        machine=platform_module.machine(),
        platform=platform_module.platform(),
        catboost_version=catboost.__version__,
        scikit_learn_version=sklearn.__version__,
        numpy_version=np.__version__,
        pyarrow_version=pa.__version__,
        pydantic_version=pydantic.__version__,
        cryptography_version=cryptography.__version__,
        pandas_version=pandas.__version__,
        installed_distributions=_installed_distributions(),
        apar_schema_version="1.0.0",
    )


class SourceInventoryEntry(ExternalContract):
    """One public source path and its immutable content digest."""

    path: str
    sha256: str

    @field_validator("path")
    @classmethod
    def path_is_public_relative_posix(cls, value: str) -> str:
        if (
            type(value) is not str
            or not value
            or "\\" in value
            or unicodedata.normalize("NFC", value) != value
        ):
            raise ValueError("source path must be nonempty relative POSIX text")
        path = PurePosixPath(value)
        if (
            path.is_absolute()
            or str(path) != value
            or any(part in {"", ".", ".."} for part in path.parts)
        ):
            raise ValueError("source path must be canonical and relative")
        lowered = value.lower()
        if any(
            token in lowered for token in ("private", "hidden", "restricted", "evaluation_hidden")
        ):
            raise ValueError("private, hidden, or restricted evaluator paths are forbidden")
        return value

    @field_validator("sha256")
    @classmethod
    def digest_is_sha256(cls, value: str) -> str:
        return _validate_digest(value, label="source digest")


class SourceInventory(ExternalContract):
    """Closed, sorted source inventory."""

    schema_version: Literal["1.0.0"] = "1.0.0"
    entries: tuple[SourceInventoryEntry, ...]

    @field_validator("entries")
    @classmethod
    def entries_are_closed_and_sorted(
        cls, value: tuple[SourceInventoryEntry, ...]
    ) -> tuple[SourceInventoryEntry, ...]:
        if not value:
            raise ValueError("source inventory must not be empty")
        paths = tuple(entry.path for entry in value)
        if paths != tuple(sorted(set(paths))):
            raise ValueError("source inventory paths must be unique and sorted")
        aliases = tuple(unicodedata.normalize("NFC", path).casefold() for path in paths)
        if len(aliases) != len(set(aliases)):
            raise ValueError("source inventory paths contain a Unicode/casefold alias")
        return value


def build_source_inventory(source_root: Path, paths: Sequence[str]) -> SourceInventory:
    """Hash an explicit public path allowlist without following any symlink."""
    raw_paths = tuple(paths)
    if any(type(path) is not str for path in raw_paths):
        raise BundleContractError("source inventory paths must be exact strings")
    root_fd, root_identity = _open_source_root(source_root)
    try:
        entries: list[SourceInventoryEntry] = []
        for raw_path in sorted(raw_paths):
            provisional = SourceInventoryEntry(path=raw_path, sha256="0" * 64)
            digest = _read_verified_source(root_fd, root_identity, provisional.path)
            entries.append(SourceInventoryEntry(path=provisional.path, sha256=digest))
        try:
            return SourceInventory(entries=tuple(entries))
        except ValidationError as error:
            raise BundleContractError("source inventory paths are invalid") from error
    finally:
        os.close(root_fd)


class _ReloadFixture(ExternalContract):
    schema_version: Literal["1.0.0"] = "1.0.0"
    matrix_semantic_digest: str
    raw_scores: tuple[float, ...]
    probability_scores: tuple[float, ...]
    calibrated_scores: tuple[float, ...]

    @field_validator("matrix_semantic_digest")
    @classmethod
    def digest_is_sha256(cls, value: str) -> str:
        return _validate_digest(value, label="reload matrix semantic digest")

    @field_validator("raw_scores", "probability_scores", "calibrated_scores")
    @classmethod
    def scores_are_finite(cls, value: tuple[float, ...]) -> tuple[float, ...]:
        if not value or any(type(item) is not float or not math.isfinite(item) for item in value):
            raise ValueError("reload scores must be nonempty finite floats")
        return value

    @model_validator(mode="after")
    def score_shapes_and_probabilities_are_valid(self) -> _ReloadFixture:
        lengths = {len(self.raw_scores), len(self.probability_scores), len(self.calibrated_scores)}
        if len(lengths) != 1:
            raise ValueError("reload score arrays must have equal lengths")
        for values in (self.probability_scores, self.calibrated_scores):
            if any(not 0.0 <= value <= 1.0 for value in values):
                raise ValueError("reload probabilities must be in [0, 1]")
        return self


class TrainingBindingReceipt(ExternalContract):
    """Signed requested, mandatory-excluded, and final-fit training lineage."""

    schema_version: Literal["1.0.0"] = "1.0.0"
    split_artifact_digest: str
    split_semantic_digest: str
    training_matrix_digest: str
    training_matrix_semantic_digest: str
    training_receipt_digest: str
    requested_row_ids: tuple[str, ...]
    excluded_row_ids: tuple[str, ...]
    final_fit_row_ids: tuple[str, ...]
    requested_count: int = Field(ge=1)
    excluded_count: int = Field(ge=0)
    final_fit_count: int = Field(ge=1)
    requested_row_ids_digest: str
    excluded_row_ids_digest: str
    final_fit_row_ids_digest: str

    @field_validator(
        "split_artifact_digest",
        "split_semantic_digest",
        "training_matrix_digest",
        "training_matrix_semantic_digest",
        "training_receipt_digest",
        "requested_row_ids_digest",
        "excluded_row_ids_digest",
        "final_fit_row_ids_digest",
    )
    @classmethod
    def digest_fields_are_sha256(cls, value: str) -> str:
        return _validate_digest(value, label="training binding digest")

    @field_validator("requested_row_ids", "excluded_row_ids", "final_fit_row_ids")
    @classmethod
    def row_ids_are_exact(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if any(type(row_id) is not str or not row_id for row_id in value):
            raise ValueError("training binding row IDs must be exact nonblank strings")
        if len(value) != len(set(value)):
            raise ValueError("training binding row IDs must be unique")
        return value

    @model_validator(mode="after")
    def partitions_and_digests_are_exact(self) -> TrainingBindingReceipt:
        if not self.requested_row_ids or not self.final_fit_row_ids:
            raise ValueError("requested and final training rows must not be empty")
        excluded = set(self.excluded_row_ids)
        expected_excluded = tuple(
            row_id for row_id in self.requested_row_ids if row_id in excluded
        )
        expected_final = tuple(
            row_id for row_id in self.requested_row_ids if row_id not in excluded
        )
        if expected_excluded != self.excluded_row_ids:
            raise ValueError("mandatory excluded rows must be a requested-order subset")
        if expected_final != self.final_fit_row_ids:
            raise ValueError("final-fit rows must equal requested rows minus exclusions")
        if (
            self.requested_count != len(self.requested_row_ids)
            or self.excluded_count != len(self.excluded_row_ids)
            or self.final_fit_count != len(self.final_fit_row_ids)
        ):
            raise ValueError("training binding counts are inconsistent")
        if (
            self.requested_row_ids_digest != _row_ids_digest(self.requested_row_ids)
            or self.excluded_row_ids_digest != _row_ids_digest(self.excluded_row_ids)
            or self.final_fit_row_ids_digest != _row_ids_digest(self.final_fit_row_ids)
        ):
            raise ValueError("training binding row-ID digest is inconsistent")
        return self


class CalibrationBindingReceipt(ExternalContract):
    """Signed proof that a calibrator was rederived on declared split rows."""

    schema_version: Literal["1.0.0"] = "1.0.0"
    split_artifact_digest: str
    split_semantic_digest: str
    model_digest: str
    fit_matrix_digest: str
    fit_matrix_semantic_digest: str
    fit_row_ids_digest: str
    fit_probability_scores_digest: str
    fit_labels_digest: str
    selection_matrix_digest: str
    selection_matrix_semantic_digest: str
    selection_row_ids_digest: str
    selection_probability_scores_digest: str
    selection_labels_digest: str
    calibration_artifact_digest: str
    calibration_state_digest: str

    @field_validator(
        "split_artifact_digest",
        "split_semantic_digest",
        "model_digest",
        "fit_matrix_digest",
        "fit_matrix_semantic_digest",
        "fit_row_ids_digest",
        "fit_probability_scores_digest",
        "fit_labels_digest",
        "selection_matrix_digest",
        "selection_matrix_semantic_digest",
        "selection_row_ids_digest",
        "selection_probability_scores_digest",
        "selection_labels_digest",
        "calibration_artifact_digest",
        "calibration_state_digest",
    )
    @classmethod
    def digest_fields_are_sha256(cls, value: str) -> str:
        return _validate_digest(value, label="calibration binding digest")


class ThresholdBindingReceipt(ExternalContract):
    """Signed proof that thresholds were reselected on declared later rows."""

    schema_version: Literal["1.0.0"] = "1.0.0"
    split_artifact_digest: str
    split_semantic_digest: str
    model_digest: str
    matrix_digest: str
    matrix_semantic_digest: str
    row_ids_digest: str
    model_probability_scores_digest: str
    calibrated_scores_digest: str
    labels_digest: str
    mandatory_actions: tuple[Action, ...]
    mandatory_actions_digest: str
    values_digest: str | None
    threshold_artifact_digest: str
    threshold_report_digest: str
    callback_contract_version: Literal["truth-blind-intervention-mask-v1"] = (
        "truth-blind-intervention-mask-v1"
    )

    @field_validator(
        "split_artifact_digest",
        "split_semantic_digest",
        "model_digest",
        "matrix_digest",
        "matrix_semantic_digest",
        "row_ids_digest",
        "model_probability_scores_digest",
        "calibrated_scores_digest",
        "labels_digest",
        "mandatory_actions_digest",
        "values_digest",
        "threshold_artifact_digest",
        "threshold_report_digest",
    )
    @classmethod
    def digests_are_sha256(cls, value: str | None) -> str | None:
        return None if value is None else _validate_digest(value, label="threshold binding digest")

    @field_validator("mandatory_actions")
    @classmethod
    def actions_are_nonempty_and_exact(
        cls, value: tuple[Action, ...]
    ) -> tuple[Action, ...]:
        if not value or any(type(action) is not Action for action in value):
            raise ValueError("threshold binding actions must be nonempty exact Action values")
        return value


class BundleComponent(ExternalContract):
    """Signed media type and size bound for one content-addressed component."""

    name: str
    sha256: str
    media_type: str
    size_bytes: int = Field(ge=0)

    @field_validator("name", "media_type")
    @classmethod
    def text_is_exact_nonblank(cls, value: str) -> str:
        if type(value) is not str or not value or value.strip() != value:
            raise ValueError("component identity must be exact nonblank text")
        return value

    @field_validator("sha256")
    @classmethod
    def digest_is_sha256(cls, value: str) -> str:
        return _validate_digest(value, label="component digest")


class DefenderBundleManifest(ExternalContract):
    """Signed frozen lineage and component content addresses."""

    schema_version: Literal["1.0.0"] = "1.0.0"
    bundle_id: str
    corpus_digest: str
    observation_dataset_digest: str
    evaluator_truth_digest: str
    split_manifest_digest: str
    feature_provenance_digest: str
    hyperparameter_digest: str
    reason_code_mapping_digest: str
    split_artifact_digest: str
    feature_catalog_digest: str
    feature_semantic_digest: str
    training_matrix_digest: str
    training_matrix_semantic_digest: str
    calibration_fit_matrix_digest: str
    calibration_fit_matrix_semantic_digest: str
    calibration_selection_matrix_digest: str
    calibration_selection_matrix_semantic_digest: str
    threshold_matrix_digest: str
    threshold_matrix_semantic_digest: str
    rule_manifest_digest: str
    rule_semantic_digest: str
    model_digest: str
    training_receipt_digest: str
    training_binding_digest: str
    calibration_digest: str
    calibration_binding_digest: str
    threshold_digest: str
    threshold_binding_digest: str
    environment_digest: str
    source_inventory_digest: str
    reload_matrix_digest: str
    reload_matrix_semantic_digest: str
    reload_fixture_digest: str
    components: tuple[BundleComponent, ...]
    fallback_mode: Literal["rules_only"] = "rules_only"
    rollback_ref: str
    rollback_size_bytes: int = Field(ge=0, le=_MAX_BUNDLE_BYTES)
    signer_key_id: str
    public_key_base64: str
    signature_base64: str
    frozen_at: datetime

    @field_validator("bundle_id")
    @classmethod
    def bundle_id_is_canonical_uuid(cls, value: str) -> str:
        if type(value) is not str:
            raise ValueError("bundle ID must be a canonical UUID")
        try:
            parsed = UUID(value)
        except (TypeError, ValueError) as error:
            raise ValueError("bundle ID must be a canonical UUID") from error
        if str(parsed) != value:
            raise ValueError("bundle ID must be a canonical UUID")
        return value

    @field_validator(
        "corpus_digest",
        "observation_dataset_digest",
        "evaluator_truth_digest",
        "split_manifest_digest",
        "feature_provenance_digest",
        "hyperparameter_digest",
        "reason_code_mapping_digest",
        "split_artifact_digest",
        "feature_catalog_digest",
        "feature_semantic_digest",
        "training_matrix_digest",
        "training_matrix_semantic_digest",
        "calibration_fit_matrix_digest",
        "calibration_fit_matrix_semantic_digest",
        "calibration_selection_matrix_digest",
        "calibration_selection_matrix_semantic_digest",
        "threshold_matrix_digest",
        "threshold_matrix_semantic_digest",
        "rule_manifest_digest",
        "rule_semantic_digest",
        "model_digest",
        "training_receipt_digest",
        "training_binding_digest",
        "calibration_digest",
        "calibration_binding_digest",
        "threshold_digest",
        "threshold_binding_digest",
        "environment_digest",
        "source_inventory_digest",
        "reload_matrix_digest",
        "reload_matrix_semantic_digest",
        "reload_fixture_digest",
        "signer_key_id",
    )
    @classmethod
    def digests_are_sha256(cls, value: str) -> str:
        return _validate_digest(value)

    @field_validator("rollback_ref")
    @classmethod
    def rollback_is_genesis_or_digest(cls, value: str) -> str:
        if value == GENESIS_ROLLBACK_REF:
            return value
        return _validate_digest(value, label="rollback reference")

    @model_validator(mode="after")
    def rollback_size_matches_reference(self) -> DefenderBundleManifest:
        if (self.rollback_ref == GENESIS_ROLLBACK_REF) != (self.rollback_size_bytes == 0):
            raise ValueError("rollback size must be zero exactly for genesis")
        return self

    @model_validator(mode="after")
    def lineage_digests_are_distinct(self) -> DefenderBundleManifest:
        values = (
            self.corpus_digest,
            self.observation_dataset_digest,
            self.evaluator_truth_digest,
            self.split_manifest_digest,
            self.feature_provenance_digest,
            self.hyperparameter_digest,
            self.reason_code_mapping_digest,
        )
        if len(values) != len(set(values)):
            raise ValueError("manifest lineage digests must identify distinct artifacts")
        return self

    @field_validator("public_key_base64")
    @classmethod
    def public_key_is_raw_ed25519(cls, value: str) -> str:
        _validated_base64(value, 32, label="public key")
        return value

    @field_validator("signature_base64")
    @classmethod
    def signature_is_raw_ed25519(cls, value: str) -> str:
        _validated_base64(value, 64, label="signature")
        return value

    @field_validator("frozen_at")
    @classmethod
    def frozen_time_is_utc(cls, value: datetime) -> datetime:
        return validate_utc_timestamp(value)

    def unsigned_document(self) -> dict[str, object]:
        """Return every security and lineage field except the signature."""
        return self.model_dump(mode="json", exclude={"signature_base64"})

    @model_validator(mode="after")
    def components_are_exact_and_bound(self) -> DefenderBundleManifest:
        names = tuple(component.name for component in self.components)
        if names != tuple(sorted(_COMPONENT_FIELD_MEDIA)):
            raise ValueError("bundle component descriptor names are not exact and sorted")
        for component in self.components:
            field_name, media_type = _COMPONENT_FIELD_MEDIA[component.name]
            if component.sha256 != getattr(self, field_name) or component.media_type != media_type:
                raise ValueError("bundle component descriptor does not match the manifest")
            if component.size_bytes > _media_size_limit(media_type):
                raise ValueError("bundle component exceeds its signed media size limit")
        return self

    def component_digests(self) -> tuple[str, ...]:
        """Return all directly stored component addresses in stable order."""
        return tuple(component.sha256 for component in self.components)

    def component(self, name: str) -> BundleComponent:
        """Return one signed component descriptor by its exact stable name."""
        matching = tuple(component for component in self.components if component.name == name)
        if len(matching) != 1:
            raise BundleContractError(f"bundle component is unavailable: {name}")
        return matching[0]


def _validated_base64(value: object, size: int, *, label: str) -> bytes:
    if type(value) is not str:
        raise ValueError(f"{label} must be canonical base64")
    try:
        decoded = base64.b64decode(value, validate=True)
    except (binascii.Error, ValueError) as error:
        raise ValueError(f"{label} must be canonical base64") from error
    if len(decoded) != size or base64.b64encode(decoded).decode("ascii") != value:
        raise ValueError(f"{label} must be canonical base64")
    return decoded


class _LoadedSnapshot(NamedTuple):
    manifest_bytes: bytes
    model_bytes: bytes
    receipt_bytes: bytes
    catalog_bytes: bytes
    training_bytes: bytes
    calibration_fit_bytes: bytes
    calibration_selection_bytes: bytes
    threshold_matrix_bytes: bytes
    training_binding_bytes: bytes
    reload_bytes: bytes
    rules_bytes: bytes
    calibration_bytes: bytes
    threshold_bytes: bytes
    environment_bytes: bytes
    source_inventory_bytes: bytes
    calibration_binding_bytes: bytes
    threshold_binding_bytes: bytes
    reload_fixture_bytes: bytes


class LoadedDefenderBundle:
    """Sealed verified runtime rebuilding mutable public contracts from safe bytes."""

    __slots__ = ("_snapshot",)
    _snapshot: _LoadedSnapshot

    def __init__(
        self,
        *,
        manifest: DefenderBundleManifest,
        component_bytes: dict[str, bytes],
    ) -> None:
        try:
            object.__getattribute__(self, "_snapshot")
        except AttributeError:
            pass
        else:
            raise BundleContractError("loaded defender bundle is already initialized")
        snapshot = _LoadedSnapshot(
            manifest_bytes=canonical_json_bytes(manifest.model_dump(mode="json")),
            model_bytes=bytes(component_bytes["model"]),
            receipt_bytes=bytes(component_bytes["receipt"]),
            catalog_bytes=bytes(component_bytes["catalog"]),
            training_bytes=bytes(component_bytes["training_matrix"]),
            calibration_fit_bytes=bytes(component_bytes["calibration_fit_matrix"]),
            calibration_selection_bytes=bytes(
                component_bytes["calibration_selection_matrix"]
            ),
            threshold_matrix_bytes=bytes(component_bytes["threshold_matrix"]),
            training_binding_bytes=bytes(component_bytes["training_binding"]),
            reload_bytes=bytes(component_bytes["reload_matrix"]),
            rules_bytes=bytes(component_bytes["rules"]),
            calibration_bytes=bytes(component_bytes["calibration"]),
            threshold_bytes=bytes(component_bytes["threshold"]),
            environment_bytes=bytes(component_bytes["environment"]),
            source_inventory_bytes=bytes(component_bytes["source_inventory"]),
            calibration_binding_bytes=bytes(component_bytes["calibration_binding"]),
            threshold_binding_bytes=bytes(component_bytes["threshold_binding"]),
            reload_fixture_bytes=bytes(component_bytes["reload_fixture"]),
        )
        object.__setattr__(self, "_snapshot", snapshot)

    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError(f"{type(self).__name__} is read-only: {name}")

    def __delattr__(self, name: str) -> None:
        raise AttributeError(f"{type(self).__name__} is read-only: {name}")

    @property
    def manifest(self) -> DefenderBundleManifest:
        return _manifest_from_document(strict_json_loads(self._snapshot.manifest_bytes))

    @property
    def scorer(self) -> CatBoostScorer:
        receipt = TrainingReceipt.model_validate(
            strict_json_loads(self._snapshot.receipt_bytes)
        )
        return CatBoostScorer.from_bytes(self._snapshot.model_bytes, receipt)

    @property
    def catalog(self) -> FeatureCatalog:
        return FeatureCatalog.model_validate(strict_json_loads(self._snapshot.catalog_bytes))

    @property
    def training_matrix(self) -> FeatureMatrix:
        return _matrix_from_parquet(self._snapshot.training_bytes, self.catalog)

    @property
    def training_binding(self) -> TrainingBindingReceipt:
        return TrainingBindingReceipt.model_validate(
            strict_json_loads(self._snapshot.training_binding_bytes)
        )

    @property
    def calibration_fit_matrix(self) -> FeatureMatrix:
        return _matrix_from_parquet(self._snapshot.calibration_fit_bytes, self.catalog)

    @property
    def calibration_selection_matrix(self) -> FeatureMatrix:
        return _matrix_from_parquet(self._snapshot.calibration_selection_bytes, self.catalog)

    @property
    def threshold_matrix(self) -> FeatureMatrix:
        return _matrix_from_parquet(self._snapshot.threshold_matrix_bytes, self.catalog)

    @property
    def reload_matrix(self) -> FeatureMatrix:
        return _matrix_from_parquet(self._snapshot.reload_bytes, self.catalog)

    @property
    def rule_manifest(self) -> RuleManifest:
        return RuleManifest.model_validate(strict_json_loads(self._snapshot.rules_bytes))

    @property
    def calibrator(self) -> ProbabilityCalibrator:
        return ProbabilityCalibrator.from_json(self._snapshot.calibration_bytes)

    @property
    def threshold_report(self) -> ThresholdReport:
        return ThresholdReport.from_json(self._snapshot.threshold_bytes)

    @property
    def environment_lock(self) -> EnvironmentLock:
        return EnvironmentLock.model_validate(
            strict_json_loads(self._snapshot.environment_bytes)
        )

    @property
    def source_inventory(self) -> SourceInventory:
        return SourceInventory.model_validate(
            strict_json_loads(self._snapshot.source_inventory_bytes)
        )

    @property
    def calibration_binding(self) -> CalibrationBindingReceipt:
        return CalibrationBindingReceipt.model_validate(
            strict_json_loads(self._snapshot.calibration_binding_bytes)
        )

    @property
    def threshold_binding(self) -> ThresholdBindingReceipt:
        return ThresholdBindingReceipt.model_validate(
            strict_json_loads(self._snapshot.threshold_binding_bytes)
        )

    def verify_reload(self) -> None:
        """Re-run signed reload parity from private immutable component bytes."""
        fixture = _ReloadFixture.model_validate(
            strict_json_loads(self._snapshot.reload_fixture_bytes)
        )
        _verify_reload_parity(self.scorer, self.calibrator, self.reload_matrix, fixture)


class DefenderBundlePublisher:
    """Publish and load defender bundles under one exact store and signing authority."""

    __slots__ = ("_closed", "_source_root_fd", "_source_root_identity", "_store", "_signer")

    def __init__(self, store: ArtifactStore, signer: RunSigningIdentity, source_root: Path) -> None:
        if type(store) is not ArtifactStore:
            raise BundleContractError("publisher requires an exact ArtifactStore")
        if type(signer) is not RunSigningIdentity:
            raise BundleContractError("publisher requires an exact RunSigningIdentity")
        self._store = store
        self._signer = signer
        self._source_root_fd, self._source_root_identity = _open_source_root(source_root)
        self._closed = False

    def __enter__(self) -> DefenderBundlePublisher:
        self._require_open()
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        self.close()

    def __del__(self) -> None:
        self.close()

    def close(self) -> None:
        """Close the pinned source-root descriptor; repeated calls are harmless."""
        if not getattr(self, "_closed", True):
            descriptor = self._source_root_fd
            self._closed = True
            with suppress(OSError):
                os.close(descriptor)

    def _require_open(self) -> None:
        if getattr(self, "_closed", True):
            raise BundleContractError("defender bundle publisher is closed")

    def freeze(
        self,
        *,
        scorer: CatBoostScorer,
        catalog: FeatureCatalog,
        split: EvaluationSplit,
        training_matrix: FeatureMatrix,
        mandatory_excluded_row_ids: tuple[str, ...] = (),
        calibration_fit_matrix: FeatureMatrix,
        calibration_fit_labels: np.ndarray,
        calibration_selection_matrix: FeatureMatrix,
        calibration_selection_labels: np.ndarray,
        threshold_matrix: FeatureMatrix,
        threshold_labels: np.ndarray,
        threshold_mandatory_actions: np.ndarray,
        threshold_values: np.ndarray | None,
        review_case_counter: Callable[[np.ndarray], int],
        rule_manifest: RuleManifest,
        calibrator: ProbabilityCalibrator,
        threshold_report: ThresholdReport,
        lineage: BundleLineage,
        environment_lock: EnvironmentLock,
        source_inventory: SourceInventory,
        reload_matrix: FeatureMatrix,
        bundle_id: str,
        frozen_at: datetime,
        rollback_ref: str = GENESIS_ROLLBACK_REF,
    ) -> tuple[DefenderBundleManifest, ArtifactRef]:
        """Validate every binding, then atomically expose a signed top manifest."""
        self._require_open()
        try:
            components = self._prepare_components(
                scorer=scorer,
                catalog=catalog,
                split=split,
                training_matrix=training_matrix,
                mandatory_excluded_row_ids=mandatory_excluded_row_ids,
                calibration_fit_matrix=calibration_fit_matrix,
                calibration_fit_labels=calibration_fit_labels,
                calibration_selection_matrix=calibration_selection_matrix,
                calibration_selection_labels=calibration_selection_labels,
                threshold_matrix=threshold_matrix,
                threshold_labels=threshold_labels,
                threshold_mandatory_actions=threshold_mandatory_actions,
                threshold_values=threshold_values,
                review_case_counter=review_case_counter,
                rule_manifest=rule_manifest,
                calibrator=calibrator,
                threshold_report=threshold_report,
                lineage=lineage,
                environment_lock=environment_lock,
                source_inventory=source_inventory,
                reload_matrix=reload_matrix,
                bundle_id=bundle_id,
                rollback_ref=rollback_ref,
                frozen_at=frozen_at,
            )
            refs = {
                name: self._store.put_bytes(payload, media_type)
                for name, (payload, media_type) in components.payloads.items()
            }
            rollback_size_bytes = (
                0
                if rollback_ref == GENESIS_ROLLBACK_REF
                else self._component_ref_for_digest(
                    rollback_ref, _BUNDLE_MEDIA, _MAX_BUNDLE_BYTES
                ).size_bytes
            )
            document: dict[str, object] = {
                "schema_version": _SCHEMA_VERSION,
                "bundle_id": bundle_id,
                **lineage.model_dump(mode="json", exclude={"schema_version"}),
                "split_artifact_digest": refs["split"].sha256,
                "feature_catalog_digest": refs["catalog"].sha256,
                "feature_semantic_digest": components.feature_semantic_digest,
                "training_matrix_digest": refs["training_matrix"].sha256,
                "training_matrix_semantic_digest": components.training_semantic_digest,
                "training_binding_digest": refs["training_binding"].sha256,
                "calibration_fit_matrix_digest": refs["calibration_fit_matrix"].sha256,
                "calibration_fit_matrix_semantic_digest": (
                    components.calibration_fit_semantic_digest
                ),
                "calibration_selection_matrix_digest": refs["calibration_selection_matrix"].sha256,
                "calibration_selection_matrix_semantic_digest": (
                    components.calibration_selection_semantic_digest
                ),
                "threshold_matrix_digest": refs["threshold_matrix"].sha256,
                "threshold_matrix_semantic_digest": components.threshold_semantic_digest,
                "rule_manifest_digest": refs["rules"].sha256,
                "rule_semantic_digest": components.rule_semantic_digest,
                "model_digest": refs["model"].sha256,
                "training_receipt_digest": refs["receipt"].sha256,
                "calibration_digest": refs["calibration"].sha256,
                "calibration_binding_digest": refs["calibration_binding"].sha256,
                "threshold_digest": refs["threshold"].sha256,
                "threshold_binding_digest": refs["threshold_binding"].sha256,
                "environment_digest": refs["environment"].sha256,
                "source_inventory_digest": refs["source_inventory"].sha256,
                "reload_matrix_digest": refs["reload_matrix"].sha256,
                "reload_matrix_semantic_digest": components.reload_semantic_digest,
                "reload_fixture_digest": refs["reload_fixture"].sha256,
                "components": [
                    {
                        "name": name,
                        "sha256": refs[name].sha256,
                        "media_type": refs[name].media_type,
                        "size_bytes": refs[name].size_bytes,
                    }
                    for name in sorted(refs)
                ],
                "fallback_mode": "rules_only",
                "rollback_ref": rollback_ref,
                "rollback_size_bytes": rollback_size_bytes,
                "signer_key_id": self._signer.key_id,
                "public_key_base64": self._signer.public_key_base64,
                "signature_base64": base64.b64encode(b"\x00" * 64).decode("ascii"),
                "frozen_at": frozen_at,
            }
            unsigned = DefenderBundleManifest.model_validate(document)
            manifest = unsigned.model_copy(
                update={"signature_base64": self._signer.sign(unsigned.unsigned_document())}
            )
            manifest = DefenderBundleManifest.model_validate(manifest)
            if not self._verify_signature(manifest):
                raise BundleContractError("new bundle signature did not verify")
            payload = canonical_json_bytes(manifest.model_dump(mode="json"))
            ref = self._store.put_bytes(payload, _BUNDLE_MEDIA)
            return manifest, ref
        except BundleContractError:
            raise
        except (TypeError, ValueError, ValidationError, OSError, pa.ArrowException) as error:
            raise BundleContractError(f"bundle publication failed: {error}") from error

    def verify(self, manifest: object) -> bool:
        """Return false, never raise, for any hostile manifest or component state."""
        try:
            self._require_open()
            if type(manifest) is not DefenderBundleManifest:
                return False
            validated = DefenderBundleManifest.model_validate(manifest.model_dump(mode="python"))
            self._load_validated(validated, ancestry_bytes=0)
            return True
        except Exception:
            return False

    def load(self, ref: ArtifactRef) -> LoadedDefenderBundle:
        """Load only after top-ref, signature, chain, component, and parity checks."""
        self._require_open()
        try:
            if type(ref) is not ArtifactRef or ref.media_type != _BUNDLE_MEDIA:
                raise BundleContractError("bundle reference has an invalid type or media type")
            if ref.size_bytes > _MAX_BUNDLE_BYTES:
                raise BundleContractError("bundle reference exceeds its size limit")
            payload = self._store.read(ref)
            document = strict_json_loads(payload)
            manifest = _manifest_from_document(document)
            if canonical_json_bytes(manifest.model_dump(mode="json")) != payload:
                raise BundleContractError("bundle manifest is not canonical")
            return self._load_validated(manifest, ancestry_bytes=ref.size_bytes)
        except BundleContractError:
            raise
        except Exception as error:
            raise BundleContractError(f"bundle load failed: {error}") from error

    def _prepare_components(
        self,
        *,
        scorer: CatBoostScorer,
        catalog: FeatureCatalog,
        split: EvaluationSplit,
        training_matrix: FeatureMatrix,
        mandatory_excluded_row_ids: tuple[str, ...],
        calibration_fit_matrix: FeatureMatrix,
        calibration_fit_labels: np.ndarray,
        calibration_selection_matrix: FeatureMatrix,
        calibration_selection_labels: np.ndarray,
        threshold_matrix: FeatureMatrix,
        threshold_labels: np.ndarray,
        threshold_mandatory_actions: np.ndarray,
        threshold_values: np.ndarray | None,
        review_case_counter: Callable[[np.ndarray], int],
        rule_manifest: RuleManifest,
        calibrator: ProbabilityCalibrator,
        threshold_report: ThresholdReport,
        lineage: BundleLineage,
        environment_lock: EnvironmentLock,
        source_inventory: SourceInventory,
        reload_matrix: FeatureMatrix,
        bundle_id: str,
        rollback_ref: str,
        frozen_at: datetime,
    ) -> _PreparedComponents:
        _require_exact_types(
            scorer=scorer,
            catalog=catalog,
            split=split,
            training_matrix=training_matrix,
            calibration_fit_matrix=calibration_fit_matrix,
            calibration_selection_matrix=calibration_selection_matrix,
            threshold_matrix=threshold_matrix,
            rule_manifest=rule_manifest,
            calibrator=calibrator,
            threshold_report=threshold_report,
            lineage=lineage,
            environment_lock=environment_lock,
            source_inventory=source_inventory,
            reload_matrix=reload_matrix,
        )
        try:
            catalog = FeatureCatalog.model_validate(catalog)
            split = EvaluationSplit.model_validate(split)
            training_matrix = FeatureMatrix.model_validate(training_matrix)
            calibration_fit_matrix = FeatureMatrix.model_validate(calibration_fit_matrix)
            calibration_selection_matrix = FeatureMatrix.model_validate(
                calibration_selection_matrix
            )
            threshold_matrix = FeatureMatrix.model_validate(threshold_matrix)
            rule_manifest = RuleManifest.model_validate(rule_manifest)
            calibrator = ProbabilityCalibrator.model_validate(calibrator)
            threshold_report = ThresholdReport.model_validate(threshold_report)
            lineage = BundleLineage.model_validate(lineage)
            environment_lock = EnvironmentLock.model_validate(environment_lock)
            source_inventory = SourceInventory.model_validate(source_inventory)
            reload_matrix = FeatureMatrix.model_validate(reload_matrix)
        except ValidationError as error:
            raise BundleContractError("bundle component contract is invalid") from error
        try:
            validate_utc_timestamp(frozen_at)
        except (TypeError, ValueError) as error:
            raise BundleContractError("frozen_at must be timezone-aware UTC") from error
        try:
            canonical_bundle_id = str(UUID(bundle_id))
        except (TypeError, ValueError) as error:
            raise BundleContractError("bundle ID must be a canonical UUID") from error
        if type(bundle_id) is not str or bundle_id != canonical_bundle_id:
            raise BundleContractError("bundle ID must be a canonical UUID")
        _validate_environment(environment_lock, scorer, calibrator)
        _validate_source_inventory(
            self._source_root_fd, self._source_root_identity, source_inventory
        )
        if not threshold_report.feasible or threshold_report.thresholds is None:
            raise BundleContractError("only a feasible threshold report may be frozen")
        if rollback_ref != GENESIS_ROLLBACK_REF:
            _validate_digest(rollback_ref, label="rollback reference")
            predecessor_ref = self._component_ref_for_digest(
                rollback_ref, _BUNDLE_MEDIA, _MAX_BUNDLE_BYTES
            )
            predecessor, predecessor_size = self._load_manifest_ref_with_size(
                rollback_ref, predecessor_ref.size_bytes
            )
            if predecessor.frozen_at >= frozen_at:
                raise BundleContractError(
                    "rollback predecessor must be earlier than the new bundle"
                )
            if predecessor.bundle_id == bundle_id:
                raise BundleContractError("rollback predecessor must have a distinct bundle ID")
            self._verify_ancestry(
                predecessor,
                visited={rollback_ref},
                depth=1,
                cumulative_bytes=predecessor_size,
            )

        feature_semantic = _validate_catalog(catalog)
        split_semantic = _validate_split(split)
        if lineage.split_manifest_digest != split.split_digest:
            raise BundleContractError("lineage split digest does not match the supplied split")
        matrices = {
            "training": training_matrix,
            "calibration fit": calibration_fit_matrix,
            "calibration selection": calibration_selection_matrix,
            "threshold": threshold_matrix,
            "reload": reload_matrix,
        }
        for label, matrix in matrices.items():
            _validate_matrix(matrix, catalog, label=label)
        _require_matrix_rows(training_matrix, split.training_row_ids, label="training")
        _require_matrix_rows(
            calibration_fit_matrix,
            split.row_ids["calibrator_fit"],
            label="calibration fit",
        )
        _require_matrix_rows(
            calibration_selection_matrix,
            split.row_ids["threshold"],
            label="calibration selection",
        )
        _require_matrix_rows(threshold_matrix, split.row_ids["threshold"], label="threshold")
        _require_matrix_rows(reload_matrix, split.row_ids["development"], label="reload")
        receipt = scorer.receipt
        expected_hyperparameters = _digest(
            canonical_json_bytes(receipt.selected_params.model_dump(mode="json"))
        )
        if lineage.hyperparameter_digest != expected_hyperparameters:
            raise BundleContractError("lineage hyperparameters do not match the model receipt")
        if receipt.catalog_digest != feature_semantic:
            raise BundleContractError("model receipt does not match the feature catalog")
        if receipt.model_payload_digest != _digest(scorer.to_bytes()):
            raise BundleContractError("model payload does not match its receipt")
        training_ids = tuple(row.event_id for row in training_matrix.rows)
        excluded_ids, final_fit_ids = _training_partitions(
            training_ids, mandatory_excluded_row_ids
        )
        if len(training_ids) != receipt.requested_training_count:
            raise BundleContractError("training matrix count does not match the receipt")
        if (
            _digest(canonical_json_bytes(list(training_ids)))
            != receipt.requested_training_row_ids_digest
        ):
            raise BundleContractError("training matrix row IDs do not match the receipt")
        if (
            receipt.mandatory_excluded_count != len(excluded_ids)
            or receipt.final_training_count != len(final_fit_ids)
            or receipt.final_training_row_ids_digest != _row_ids_digest(final_fit_ids)
        ):
            raise BundleContractError(
                "mandatory excluded rows do not match the model receipt"
            )
        if any(row.decision_at > receipt.training_cutoff for row in training_matrix.rows):
            raise BundleContractError("training matrix contains a row after its cutoff")
        reloaded_scorer = CatBoostScorer.from_bytes(scorer.to_bytes(), receipt)

        matrix_payloads = {label: _matrix_to_parquet(matrix) for label, matrix in matrices.items()}
        for label, matrix in matrices.items():
            if _matrix_from_parquet(matrix_payloads[label], catalog) != matrix:
                raise BundleContractError(f"{label} Parquet does not reconstruct exactly")
        training_semantic = _matrix_semantic_digest(training_matrix)
        calibration_fit_semantic = _matrix_semantic_digest(calibration_fit_matrix)
        calibration_selection_semantic = _matrix_semantic_digest(calibration_selection_matrix)
        threshold_semantic = _matrix_semantic_digest(threshold_matrix)
        reload_semantic = _matrix_semantic_digest(reload_matrix)

        fit_labels = _binary_array(calibration_fit_labels, label="calibration fit labels")
        selection_labels = _binary_array(
            calibration_selection_labels, label="calibration selection labels"
        )
        checked_threshold_labels = _binary_array(threshold_labels, label="threshold labels")
        _require_split_labels(split, calibration_fit_matrix, fit_labels, label="calibration fit")
        _require_split_labels(
            split,
            calibration_selection_matrix,
            selection_labels,
            label="calibration selection",
        )
        _require_split_labels(split, threshold_matrix, checked_threshold_labels, label="threshold")
        fit_probabilities = reloaded_scorer.predict(calibration_fit_matrix)
        selection_probabilities = reloaded_scorer.predict(calibration_selection_matrix)
        derived_calibrator = select_calibrator(
            fit_probabilities,
            fit_labels,
            selection_probabilities,
            selection_labels,
            min_class_count=calibrator.artifact.min_class_count,
        )
        if derived_calibrator != calibrator:
            raise BundleContractError(
                "calibrator is not derived from the declared model and split windows"
            )
        threshold_probabilities = reloaded_scorer.predict(threshold_matrix)
        threshold_calibrated = calibrator.predict(threshold_probabilities)
        if type(threshold_mandatory_actions) is not np.ndarray:
            raise BundleContractError("threshold mandatory actions must be an exact array")
        actions = threshold_mandatory_actions.copy()
        checked_values = None if threshold_values is None else threshold_values.copy()
        if checked_values is not None:
            _require_split_values(split, threshold_matrix, checked_values)
        derived_threshold = select_policy_thresholds(
            threshold_calibrated,
            checked_threshold_labels,
            actions,
            review_case_counter,
            threshold_report.budget,
            checked_values,
        )
        if derived_threshold != threshold_report:
            raise BundleContractError(
                "threshold report is not derived from the declared calibrated split rows"
            )

        probability = reloaded_scorer.predict(reload_matrix)
        raw = reloaded_scorer.predict_raw(reload_matrix)
        if not np.array_equal(probability, scorer.predict(reload_matrix)) or not np.allclose(
            raw,
            scorer.predict_raw(reload_matrix),
            rtol=0.0,
            atol=1e-12,
        ):
            raise BundleContractError("provided scorer does not match its native payload")
        calibrated = calibrator.predict(probability)
        fixture = _ReloadFixture(
            matrix_semantic_digest=reload_semantic,
            raw_scores=tuple(float(value) for value in raw),
            probability_scores=tuple(float(value) for value in probability),
            calibrated_scores=tuple(float(value) for value in calibrated),
        )
        rule_semantic = rule_manifest_digest(rule_manifest)
        split_payload = _split_to_bytes(split)
        calibration_payload = calibrator.to_json()
        threshold_payload = threshold_report.to_json()
        model_payload = scorer.to_bytes()
        receipt_payload = canonical_json_bytes(receipt.model_dump(mode="json"))
        training_binding = TrainingBindingReceipt(
            split_artifact_digest=_digest(split_payload),
            split_semantic_digest=split_semantic,
            training_matrix_digest=_digest(matrix_payloads["training"]),
            training_matrix_semantic_digest=training_semantic,
            training_receipt_digest=_digest(receipt_payload),
            requested_row_ids=training_ids,
            excluded_row_ids=excluded_ids,
            final_fit_row_ids=final_fit_ids,
            requested_count=len(training_ids),
            excluded_count=len(excluded_ids),
            final_fit_count=len(final_fit_ids),
            requested_row_ids_digest=_row_ids_digest(training_ids),
            excluded_row_ids_digest=_row_ids_digest(excluded_ids),
            final_fit_row_ids_digest=_row_ids_digest(final_fit_ids),
        )
        calibration_binding = CalibrationBindingReceipt(
            split_artifact_digest=_digest(split_payload),
            split_semantic_digest=split_semantic,
            model_digest=_digest(model_payload),
            fit_matrix_digest=_digest(matrix_payloads["calibration fit"]),
            fit_matrix_semantic_digest=calibration_fit_semantic,
            fit_row_ids_digest=_row_ids_digest(
                tuple(row.event_id for row in calibration_fit_matrix.rows)
            ),
            fit_probability_scores_digest=_numeric_array_digest(fit_probabilities),
            fit_labels_digest=_numeric_array_digest(fit_labels),
            selection_matrix_digest=_digest(matrix_payloads["calibration selection"]),
            selection_matrix_semantic_digest=calibration_selection_semantic,
            selection_row_ids_digest=_row_ids_digest(
                tuple(row.event_id for row in calibration_selection_matrix.rows)
            ),
            selection_probability_scores_digest=_numeric_array_digest(selection_probabilities),
            selection_labels_digest=_numeric_array_digest(selection_labels),
            calibration_artifact_digest=_digest(calibration_payload),
            calibration_state_digest=calibrator.artifact.artifact_digest,
        )
        threshold_binding = ThresholdBindingReceipt(
            split_artifact_digest=_digest(split_payload),
            split_semantic_digest=split_semantic,
            model_digest=_digest(model_payload),
            matrix_digest=_digest(matrix_payloads["threshold"]),
            matrix_semantic_digest=threshold_semantic,
            row_ids_digest=_row_ids_digest(tuple(row.event_id for row in threshold_matrix.rows)),
            model_probability_scores_digest=_numeric_array_digest(threshold_probabilities),
            calibrated_scores_digest=_numeric_array_digest(threshold_calibrated),
            labels_digest=_numeric_array_digest(checked_threshold_labels),
            mandatory_actions=tuple(cast(Action, action) for action in actions),
            mandatory_actions_digest=_actions_digest(actions),
            values_digest=(
                None if checked_values is None else _numeric_array_digest(checked_values)
            ),
            threshold_artifact_digest=_digest(threshold_payload),
            threshold_report_digest=threshold_report.report_digest,
            callback_contract_version="truth-blind-intervention-mask-v1",
        )
        payloads: dict[str, tuple[bytes, str]] = {
            "catalog": (canonical_json_bytes(catalog.model_dump(mode="json")), _CATALOG_MEDIA),
            "split": (split_payload, _SPLIT_MEDIA),
            "training_matrix": (matrix_payloads["training"], _PARQUET_MEDIA),
            "calibration_fit_matrix": (
                matrix_payloads["calibration fit"],
                _PARQUET_MEDIA,
            ),
            "calibration_selection_matrix": (
                matrix_payloads["calibration selection"],
                _PARQUET_MEDIA,
            ),
            "threshold_matrix": (matrix_payloads["threshold"], _PARQUET_MEDIA),
            "rules": (canonical_json_bytes(rule_manifest.model_dump(mode="json")), _RULE_MEDIA),
            "model": (model_payload, _MODEL_MEDIA),
            "receipt": (receipt_payload, _RECEIPT_MEDIA),
            "training_binding": (
                canonical_json_bytes(training_binding.model_dump(mode="json")),
                _TRAINING_BINDING_MEDIA,
            ),
            "calibration": (calibration_payload, _CALIBRATION_MEDIA),
            "calibration_binding": (
                canonical_json_bytes(calibration_binding.model_dump(mode="json")),
                _CALIBRATION_BINDING_MEDIA,
            ),
            "threshold": (threshold_payload, _THRESHOLD_MEDIA),
            "threshold_binding": (
                canonical_json_bytes(threshold_binding.model_dump(mode="json")),
                _THRESHOLD_BINDING_MEDIA,
            ),
            "environment": (
                canonical_json_bytes(environment_lock.model_dump(mode="json")),
                _ENVIRONMENT_MEDIA,
            ),
            "source_inventory": (
                canonical_json_bytes(source_inventory.model_dump(mode="json")),
                _SOURCE_MEDIA,
            ),
            "reload_matrix": (matrix_payloads["reload"], _PARQUET_MEDIA),
            "reload_fixture": (
                canonical_json_bytes(fixture.model_dump(mode="json")),
                _RELOAD_MEDIA,
            ),
        }
        return _PreparedComponents(
            payloads,
            feature_semantic,
            training_semantic,
            calibration_fit_semantic,
            calibration_selection_semantic,
            threshold_semantic,
            reload_semantic,
            rule_semantic,
        )

    def _load_validated(
        self,
        manifest: DefenderBundleManifest,
        *,
        ancestry_bytes: int,
    ) -> LoadedDefenderBundle:
        self._verify_ancestry(
            manifest,
            visited=set(),
            depth=0,
            cumulative_bytes=ancestry_bytes,
        )
        component_bytes = {
            name: self._read_named_component(manifest, name)
            for name in sorted(_COMPONENT_FIELD_MEDIA)
        }
        catalog = FeatureCatalog.model_validate(strict_json_loads(component_bytes["catalog"]))
        semantic = _validate_catalog(catalog)
        if semantic != manifest.feature_semantic_digest:
            raise BundleContractError("feature catalog semantic digest mismatch")
        receipt = TrainingReceipt.model_validate(strict_json_loads(component_bytes["receipt"]))
        model_payload = component_bytes["model"]
        scorer = CatBoostScorer.from_bytes(model_payload, receipt)
        if receipt.catalog_digest != semantic:
            raise BundleContractError("training receipt catalog mismatch")
        rules = RuleManifest.model_validate(strict_json_loads(component_bytes["rules"]))
        if rule_manifest_digest(rules) != manifest.rule_semantic_digest:
            raise BundleContractError("rule manifest semantic digest mismatch")
        calibrator = ProbabilityCalibrator.from_json(component_bytes["calibration"])
        threshold = ThresholdReport.from_json(component_bytes["threshold"])
        if not threshold.feasible or threshold.thresholds is None:
            raise BundleContractError("frozen threshold report is infeasible")
        environment = EnvironmentLock.model_validate(
            strict_json_loads(component_bytes["environment"])
        )
        source_inventory = SourceInventory.model_validate(
            strict_json_loads(component_bytes["source_inventory"])
        )
        _validate_environment(environment, scorer, calibrator)
        _validate_source_inventory(
            self._source_root_fd, self._source_root_identity, source_inventory
        )
        split = _split_from_bytes(component_bytes["split"])
        split_semantic = _validate_split(split)
        if split.split_digest != manifest.split_manifest_digest:
            raise BundleContractError("loaded split does not match manifest lineage")
        training = _matrix_from_parquet(component_bytes["training_matrix"], catalog)
        calibration_fit = _matrix_from_parquet(component_bytes["calibration_fit_matrix"], catalog)
        calibration_selection = _matrix_from_parquet(
            component_bytes["calibration_selection_matrix"], catalog
        )
        threshold_matrix = _matrix_from_parquet(component_bytes["threshold_matrix"], catalog)
        reload_matrix = _matrix_from_parquet(component_bytes["reload_matrix"], catalog)
        if _matrix_semantic_digest(training) != manifest.training_matrix_semantic_digest:
            raise BundleContractError("training matrix semantic digest mismatch")
        if (
            _matrix_semantic_digest(calibration_fit)
            != manifest.calibration_fit_matrix_semantic_digest
            or _matrix_semantic_digest(calibration_selection)
            != manifest.calibration_selection_matrix_semantic_digest
            or _matrix_semantic_digest(threshold_matrix)
            != manifest.threshold_matrix_semantic_digest
        ):
            raise BundleContractError("calibration or threshold matrix semantic digest mismatch")
        if _matrix_semantic_digest(reload_matrix) != manifest.reload_matrix_semantic_digest:
            raise BundleContractError("reload matrix semantic digest mismatch")
        _require_matrix_rows(training, split.training_row_ids, label="training")
        _require_matrix_rows(
            calibration_fit, split.row_ids["calibrator_fit"], label="calibration fit"
        )
        _require_matrix_rows(
            calibration_selection,
            split.row_ids["threshold"],
            label="calibration selection",
        )
        _require_matrix_rows(threshold_matrix, split.row_ids["threshold"], label="threshold")
        _require_matrix_rows(reload_matrix, split.row_ids["development"], label="reload")
        training_binding = TrainingBindingReceipt.model_validate(
            strict_json_loads(component_bytes["training_binding"])
        )
        _verify_training_binding(
            training_binding,
            manifest,
            split_semantic,
            receipt,
            training,
            component_bytes,
        )
        calibration_binding = CalibrationBindingReceipt.model_validate(
            strict_json_loads(component_bytes["calibration_binding"])
        )
        threshold_binding = ThresholdBindingReceipt.model_validate(
            strict_json_loads(component_bytes["threshold_binding"])
        )
        _verify_calibration_binding(
            calibration_binding,
            manifest,
            split_semantic,
            scorer,
            calibrator,
            calibration_fit,
            calibration_selection,
            split,
            component_bytes,
        )
        _verify_threshold_binding(
            threshold_binding,
            manifest,
            split_semantic,
            scorer,
            calibrator,
            threshold,
            threshold_matrix,
            split,
            component_bytes,
        )
        fixture = _ReloadFixture.model_validate(
            strict_json_loads(component_bytes["reload_fixture"])
        )
        if fixture.matrix_semantic_digest != manifest.reload_matrix_semantic_digest:
            raise BundleContractError("reload fixture matrix binding mismatch")
        _verify_reload_parity(scorer, calibrator, reload_matrix, fixture)
        return LoadedDefenderBundle(
            manifest=manifest,
            component_bytes=component_bytes,
        )

    def _verify_ancestry(
        self,
        manifest: DefenderBundleManifest,
        *,
        visited: set[str],
        depth: int,
        cumulative_bytes: int,
    ) -> None:
        if depth > _MAX_ROLLBACK_DEPTH:
            raise BundleContractError("rollback chain exceeds its maximum depth")
        if cumulative_bytes > _MAX_ROLLBACK_MANIFEST_BYTES:
            raise BundleContractError("rollback manifests exceed their cumulative byte budget")
        if not self._verify_signature(manifest):
            raise BundleContractError("bundle signature or pinned authority is invalid")
        if manifest.rollback_ref == GENESIS_ROLLBACK_REF:
            return
        if manifest.rollback_ref in visited:
            raise BundleContractError("rollback chain contains a cycle")
        predecessor, size_bytes = self._load_manifest_ref_with_size(
            manifest.rollback_ref, manifest.rollback_size_bytes
        )
        if predecessor.frozen_at >= manifest.frozen_at:
            raise BundleContractError("rollback predecessor is not earlier")
        if predecessor.bundle_id == manifest.bundle_id:
            raise BundleContractError("rollback predecessor reuses the current bundle ID")
        self._verify_ancestry(
            predecessor,
            visited={*visited, manifest.rollback_ref},
            depth=depth + 1,
            cumulative_bytes=cumulative_bytes + size_bytes,
        )

    def _verify_signature(self, manifest: DefenderBundleManifest) -> bool:
        if (
            manifest.signer_key_id != self._signer.key_id
            or manifest.public_key_base64 != self._signer.public_key_base64
        ):
            return False
        try:
            public_key = _validated_base64(manifest.public_key_base64, 32, label="public key")
            signature = _validated_base64(manifest.signature_base64, 64, label="signature")
            Ed25519PublicKey.from_public_bytes(public_key).verify(
                signature, canonical_json_bytes(manifest.unsigned_document())
            )
        except (InvalidSignature, TypeError, ValueError):
            return False
        return self._signer.verify(manifest.unsigned_document(), manifest.signature_base64)

    def _read_named_component(self, manifest: DefenderBundleManifest, name: str) -> bytes:
        descriptor = manifest.component(name)
        limit = _media_size_limit(descriptor.media_type)
        if descriptor.size_bytes > limit:
            raise BundleContractError("component exceeds its media size limit")
        ref = ArtifactRef(
            descriptor.sha256,
            descriptor.media_type,
            descriptor.size_bytes,
            f"{descriptor.sha256}/payload",
        )
        payload = self._store.read(ref)
        if len(payload) != ref.size_bytes or _digest(payload) != descriptor.sha256:
            raise BundleContractError("component size or payload digest mismatch")
        return payload

    def _component_ref_for_digest(
        self, digest: str, media_type: str, size_limit: int
    ) -> ArtifactRef:
        ref = self._store.resolve(digest)
        if ref.media_type != media_type or ref.sha256 != digest or ref.size_bytes > size_limit:
            raise BundleContractError("artifact reference violates its media or size contract")
        return ref

    def _load_manifest_ref_with_size(
        self, digest: str, expected_size: int
    ) -> tuple[DefenderBundleManifest, int]:
        if expected_size <= 0 or expected_size > _MAX_BUNDLE_BYTES:
            raise BundleContractError("rollback manifest size is invalid")
        ref = ArtifactRef(
            digest,
            _BUNDLE_MEDIA,
            expected_size,
            f"{digest}/payload",
        )
        payload = self._store.read(ref)
        document = strict_json_loads(payload)
        manifest = _manifest_from_document(document)
        if canonical_json_bytes(manifest.model_dump(mode="json")) != payload:
            raise BundleContractError("rollback manifest is not canonical")
        return manifest, len(payload)


@dataclass(frozen=True, slots=True)
class _PreparedComponents:
    payloads: dict[str, tuple[bytes, str]]
    feature_semantic_digest: str
    training_semantic_digest: str
    calibration_fit_semantic_digest: str
    calibration_selection_semantic_digest: str
    threshold_semantic_digest: str
    reload_semantic_digest: str
    rule_semantic_digest: str


def _manifest_from_document(document: object) -> DefenderBundleManifest:
    if type(document) is not dict:
        raise BundleContractError("bundle manifest must be an exact object")
    raw = cast(dict[str, object], document)
    expected = set(DefenderBundleManifest.model_fields)
    if set(raw) != expected:
        raise BundleContractError("bundle manifest field set is not exact")
    frozen_at = raw.get("frozen_at")
    if type(frozen_at) is not str:
        raise BundleContractError("bundle frozen_at must be canonical timestamp text")
    try:
        parsed = datetime.fromisoformat(frozen_at)
    except ValueError as error:
        raise BundleContractError("bundle frozen_at is invalid") from error
    return DefenderBundleManifest.model_validate({**raw, "frozen_at": parsed})


def _require_exact_types(**values: object) -> None:
    expected: dict[str, type[object]] = {
        "scorer": CatBoostScorer,
        "catalog": FeatureCatalog,
        "split": EvaluationSplit,
        "training_matrix": FeatureMatrix,
        "calibration_fit_matrix": FeatureMatrix,
        "calibration_selection_matrix": FeatureMatrix,
        "threshold_matrix": FeatureMatrix,
        "rule_manifest": RuleManifest,
        "calibrator": ProbabilityCalibrator,
        "threshold_report": ThresholdReport,
        "lineage": BundleLineage,
        "environment_lock": EnvironmentLock,
        "source_inventory": SourceInventory,
        "reload_matrix": FeatureMatrix,
    }
    for name, value in values.items():
        if type(value) is not expected[name]:
            raise BundleContractError(f"{name} must be an exact {expected[name].__name__}")


def _media_size_limit(media_type: str) -> int:
    if media_type == _MODEL_MEDIA:
        return _MAX_MODEL_BYTES
    if media_type == _PARQUET_MEDIA:
        return _MAX_PARQUET_BYTES
    if media_type == _BUNDLE_MEDIA:
        return _MAX_BUNDLE_BYTES
    return _MAX_JSON_BYTES


def _open_source_root(source_root: Path) -> tuple[int, tuple[int, int, int]]:
    if not isinstance(source_root, Path):
        raise BundleContractError("source root must be a Path")
    try:
        lexical = source_root.lstat()
    except OSError as error:
        raise BundleContractError("source root is unavailable") from error
    if stat.S_ISLNK(lexical.st_mode) or not stat.S_ISDIR(lexical.st_mode):
        raise BundleContractError("source root must be a non-symlink directory")
    try:
        descriptor = os.open(
            source_root,
            os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
        )
    except OSError as error:
        raise BundleContractError("source root must be a non-symlink directory") from error
    try:
        opened = os.fstat(descriptor)
        if (
            not stat.S_ISDIR(opened.st_mode)
            or (opened.st_dev, opened.st_ino, opened.st_mode)
            != (lexical.st_dev, lexical.st_ino, lexical.st_mode)
        ):
            raise BundleContractError("source root changed while it was pinned")
        return descriptor, (opened.st_dev, opened.st_ino, opened.st_mode)
    except Exception:
        os.close(descriptor)
        raise


def _source_fingerprint(metadata: os.stat_result) -> tuple[int, int, int, int, int, int]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mode,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def _read_verified_source(
    source_root_fd: int,
    source_root_identity: tuple[int, int, int],
    relative_path: str,
) -> str:
    entry = SourceInventoryEntry(path=relative_path, sha256="0" * 64)
    parts = PurePosixPath(entry.path).parts
    parent_descriptors: list[tuple[int, tuple[int, int, int, int, int, int]]] = []
    final_descriptor: int | None = None
    try:
        root_before = os.fstat(source_root_fd)
        if (
            not stat.S_ISDIR(root_before.st_mode)
            or (root_before.st_dev, root_before.st_ino, root_before.st_mode)
            != source_root_identity
        ):
            raise BundleContractError("pinned source root identity changed")
        root_fingerprint = _source_fingerprint(root_before)
        current_parent_fd = source_root_fd
        for part in parts[:-1]:
            parent_fd = os.open(
                part,
                os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                dir_fd=current_parent_fd,
            )
            try:
                parent_metadata = os.fstat(parent_fd)
                if not stat.S_ISDIR(parent_metadata.st_mode):
                    raise BundleContractError(
                        f"source parent is not a directory: {entry.path}"
                    )
            except Exception:
                with suppress(OSError):
                    os.close(parent_fd)
                raise
            parent_descriptors.append((parent_fd, _source_fingerprint(parent_metadata)))
            current_parent_fd = parent_fd
        final_descriptor = os.open(
            parts[-1],
            os.O_RDONLY | os.O_NOFOLLOW,
            dir_fd=current_parent_fd,
        )
        opened = os.fstat(final_descriptor)
        opened_fingerprint = _source_fingerprint(opened)
        if not stat.S_ISREG(opened.st_mode) or opened.st_size > _MAX_JSON_BYTES:
            raise BundleContractError("source file type or size is invalid")
        hasher = hashlib.sha256()
        remaining = opened.st_size
        while remaining:
            chunk = os.read(final_descriptor, min(remaining, 1024 * 1024))
            if not chunk:
                raise BundleContractError("source file changed while being read")
            hasher.update(chunk)
            remaining -= len(chunk)
        if _source_fingerprint(os.fstat(final_descriptor)) != opened_fingerprint:
            raise BundleContractError("source file changed while being hashed")
        if any(
            _source_fingerprint(os.fstat(parent_fd)) != before
            for parent_fd, before in parent_descriptors
        ):
            raise BundleContractError("source parent changed while file was being hashed")
        if _source_fingerprint(os.fstat(source_root_fd)) != root_fingerprint:
            raise BundleContractError("source root changed while file was being hashed")
        return hasher.hexdigest()
    except BundleContractError:
        raise
    except OSError as error:
        raise BundleContractError(
            f"source path contains a symlink or invalid component: {entry.path}"
        ) from error
    finally:
        if final_descriptor is not None:
            with suppress(OSError):
                os.close(final_descriptor)
        for parent_fd, _ in reversed(parent_descriptors):
            with suppress(OSError):
                os.close(parent_fd)


def _validate_source_inventory(
    source_root_fd: int,
    source_root_identity: tuple[int, int, int],
    inventory: SourceInventory,
) -> None:
    for entry in inventory.entries:
        if _read_verified_source(source_root_fd, source_root_identity, entry.path) != entry.sha256:
            raise BundleContractError(f"source inventory hash mismatch: {entry.path}")


def _canonical_decimal(value: Decimal) -> str:
    if not value.is_finite():
        raise BundleContractError("split decimal is not finite")
    if value == 0:
        return "0"
    text = format(value, "f")
    return text.rstrip("0").rstrip(".") if "." in text else text


def _split_digest_document(split: EvaluationSplit) -> dict[str, object]:
    return {
        "config": split.config.model_dump(mode="json"),
        "partition_names": list(split.partition_names),
        "campaigns": {name: list(values) for name, values in split.campaigns.items()},
        "row_ids": {name: list(values) for name, values in split.row_ids.items()},
        "training_row_ids": list(split.training_row_ids),
        "entity_cohorts": {
            event_id: [label.value for label in labels]
            for event_id, labels in split.entity_cohorts.items()
        },
        "row_families": split.row_families,
        "row_campaigns": split.row_campaigns,
        "row_is_fraud": {
            event_id: split.row_is_fraud[event_id] for event_id in sorted(split.row_is_fraud)
        },
        "row_net_settled_values": {
            event_id: _canonical_decimal(split.row_net_settled_values[event_id])
            for event_id in sorted(split.row_net_settled_values)
        },
        "label_maturity_cutoff": split.label_maturity_cutoff.isoformat(),
        "sample_counts": split.sample_counts,
        "fraud_prevalence": {
            name: _canonical_decimal(value) for name, value in split.fraud_prevalence.items()
        },
        "net_settled_value_totals": {
            name: _canonical_decimal(value)
            for name, value in split.net_settled_value_totals.items()
        },
        "held_out_evaluation_row_ids": list(split.held_out_evaluation_row_ids),
    }


def _validate_split(split: EvaluationSplit) -> str:
    expected_partitions = ("train", "calibrator_fit", "threshold", "development")
    if split.partition_names != expected_partitions:
        raise BundleContractError("split partition names are not exact")
    if set(split.row_ids) != set(expected_partitions) or set(split.campaigns) != set(
        expected_partitions
    ):
        raise BundleContractError("split partition maps are not exact")
    all_rows = tuple(row for name in expected_partitions for row in split.row_ids[name])
    if len(all_rows) != len(set(all_rows)):
        raise BundleContractError("split rows overlap across partitions")
    expected_digest = _digest(canonical_json_bytes(_split_digest_document(split)))
    if split.split_digest != expected_digest:
        raise BundleContractError("split semantic digest is inconsistent")
    return _digest(canonical_json_bytes(split.model_dump(mode="json", exclude={"split_digest"})))


def _split_to_bytes(split: EvaluationSplit) -> bytes:
    _validate_split(split)
    return canonical_json_bytes(split.model_dump(mode="json"))


def _split_from_bytes(payload: bytes) -> EvaluationSplit:
    document = strict_json_loads(payload)
    try:
        split = EvaluationSplit.model_validate(document)
    except ValidationError as error:
        raise BundleContractError("split artifact is invalid") from error
    _validate_split(split)
    return split


def _row_ids_digest(row_ids: Sequence[str]) -> str:
    return _digest(canonical_json_bytes(list(row_ids)))


def _training_partitions(
    requested_row_ids: tuple[str, ...],
    mandatory_excluded_row_ids: tuple[str, ...],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    if type(mandatory_excluded_row_ids) is not tuple or any(
        type(row_id) is not str or not row_id for row_id in mandatory_excluded_row_ids
    ):
        raise BundleContractError("mandatory excluded row IDs must be an exact string tuple")
    if len(mandatory_excluded_row_ids) != len(set(mandatory_excluded_row_ids)):
        raise BundleContractError("mandatory excluded row IDs must be unique")
    excluded = set(mandatory_excluded_row_ids)
    if not excluded <= set(requested_row_ids):
        raise BundleContractError("mandatory excluded rows must be requested training rows")
    expected_excluded = tuple(row_id for row_id in requested_row_ids if row_id in excluded)
    if mandatory_excluded_row_ids != expected_excluded:
        raise BundleContractError("mandatory exclusions must follow requested row order")
    final_fit = tuple(row_id for row_id in requested_row_ids if row_id not in excluded)
    if not final_fit:
        raise BundleContractError("mandatory exclusions cannot remove every training row")
    return expected_excluded, final_fit


def _require_matrix_rows(matrix: FeatureMatrix, expected: Sequence[str], *, label: str) -> None:
    actual = tuple(row.event_id for row in matrix.rows)
    if actual != tuple(expected):
        raise BundleContractError(f"{label} matrix rows do not match the declared split")


def _numeric_array_digest(values: np.ndarray) -> str:
    if type(values) is not np.ndarray or values.ndim != 1 or not values.size:
        raise BundleContractError("binding array must be exact, nonempty, and one-dimensional")
    if np.issubdtype(values.dtype, np.complexfloating) or np.issubdtype(values.dtype, np.object_):
        raise BundleContractError("binding numeric array has an unsupported dtype")
    contiguous = np.ascontiguousarray(values)
    numeric = np.asarray(contiguous, dtype=np.float64)
    if not np.isfinite(numeric).all():
        raise BundleContractError("binding numeric array must be finite")
    return _digest(
        canonical_json_bytes(
            {
                "dtype": contiguous.dtype.str,
                "shape": list(contiguous.shape),
                "bytes_hex": contiguous.tobytes(order="C").hex(),
            }
        )
    )


def _calibration_window_digest(scores: np.ndarray, labels: np.ndarray) -> str:
    return _digest(
        canonical_json_bytes(
            {
                "scores": {
                    "dtype": np.ascontiguousarray(scores).dtype.str,
                    "shape": list(scores.shape),
                    "bytes_hex": np.ascontiguousarray(scores).tobytes(order="C").hex(),
                },
                "labels": {
                    "dtype": np.ascontiguousarray(labels).dtype.str,
                    "shape": list(labels.shape),
                    "bytes_hex": np.ascontiguousarray(labels).tobytes(order="C").hex(),
                },
            }
        )
    )


def _binary_array(values: np.ndarray, *, label: str) -> np.ndarray:
    if type(values) is not np.ndarray or values.ndim != 1 or not values.size:
        raise BundleContractError(f"{label} must be an exact nonempty one-dimensional array")
    numeric = np.asarray(values, dtype=np.float64)
    if not np.isfinite(numeric).all() or not np.all((numeric == 0.0) | (numeric == 1.0)):
        raise BundleContractError(f"{label} must contain exact binary values")
    return numeric.astype(np.int64, copy=True)


def _actions_digest(actions: np.ndarray) -> str:
    if type(actions) is not np.ndarray or actions.ndim != 1 or not actions.size:
        raise BundleContractError("mandatory actions must be an exact nonempty array")
    values: list[str] = []
    for action in actions:
        if type(action) is not Action:
            raise BundleContractError("mandatory actions must contain exact Action values")
        values.append(action.value)
    return _digest(canonical_json_bytes(values))


def _require_split_labels(
    split: EvaluationSplit,
    matrix: FeatureMatrix,
    labels: np.ndarray,
    *,
    label: str,
) -> None:
    expected = np.asarray(
        [int(split.row_is_fraud[row.event_id]) for row in matrix.rows], dtype=np.int64
    )
    if labels.shape != expected.shape or not np.array_equal(labels, expected):
        raise BundleContractError(f"{label} labels do not match evaluator split truth")


def _require_split_values(
    split: EvaluationSplit, matrix: FeatureMatrix, values: np.ndarray
) -> None:
    if type(values) is not np.ndarray or values.ndim != 1:
        raise BundleContractError("threshold values must be an exact one-dimensional array")
    expected = np.asarray(
        [float(split.row_net_settled_values[row.event_id]) for row in matrix.rows],
        dtype=np.float64,
    )
    actual = np.asarray(values, dtype=np.float64)
    if actual.shape != expected.shape or not np.array_equal(actual, expected):
        raise BundleContractError("threshold values do not match evaluator split truth")


def _verify_training_binding(
    binding: TrainingBindingReceipt,
    manifest: DefenderBundleManifest,
    split_semantic: str,
    receipt: TrainingReceipt,
    matrix: FeatureMatrix,
    component_bytes: dict[str, bytes],
) -> None:
    requested = tuple(row.event_id for row in matrix.rows)
    excluded, final_fit = _training_partitions(requested, binding.excluded_row_ids)
    expected = {
        "split_artifact_digest": manifest.split_artifact_digest,
        "split_semantic_digest": split_semantic,
        "training_matrix_digest": manifest.training_matrix_digest,
        "training_matrix_semantic_digest": manifest.training_matrix_semantic_digest,
        "training_receipt_digest": _digest(component_bytes["receipt"]),
        "requested_row_ids": requested,
        "excluded_row_ids": excluded,
        "final_fit_row_ids": final_fit,
    }
    document = binding.model_dump(mode="python")
    if any(document[name] != value for name, value in expected.items()):
        raise BundleContractError("training binding receipt is inconsistent")
    if (
        receipt.requested_training_count != len(requested)
        or receipt.requested_training_row_ids_digest != _row_ids_digest(requested)
        or receipt.mandatory_excluded_count != len(excluded)
        or receipt.final_training_count != len(final_fit)
        or receipt.final_training_row_ids_digest != _row_ids_digest(final_fit)
        or any(row.decision_at > receipt.training_cutoff for row in matrix.rows)
    ):
        raise BundleContractError("training matrix does not match the model receipt")


def _verify_calibration_binding(
    binding: CalibrationBindingReceipt,
    manifest: DefenderBundleManifest,
    split_semantic: str,
    scorer: CatBoostScorer,
    calibrator: ProbabilityCalibrator,
    fit_matrix: FeatureMatrix,
    selection_matrix: FeatureMatrix,
    split: EvaluationSplit,
    component_bytes: dict[str, bytes],
) -> None:
    fit_probabilities = scorer.predict(fit_matrix)
    selection_probabilities = scorer.predict(selection_matrix)
    fit_labels = np.asarray(
        [int(split.row_is_fraud[row.event_id]) for row in fit_matrix.rows], dtype=np.int64
    )
    selection_labels = np.asarray(
        [int(split.row_is_fraud[row.event_id]) for row in selection_matrix.rows],
        dtype=np.int64,
    )
    expected = {
        "split_artifact_digest": manifest.split_artifact_digest,
        "split_semantic_digest": split_semantic,
        "model_digest": manifest.model_digest,
        "fit_matrix_digest": manifest.calibration_fit_matrix_digest,
        "fit_matrix_semantic_digest": manifest.calibration_fit_matrix_semantic_digest,
        "fit_row_ids_digest": _row_ids_digest(tuple(row.event_id for row in fit_matrix.rows)),
        "fit_probability_scores_digest": _numeric_array_digest(fit_probabilities),
        "fit_labels_digest": _numeric_array_digest(fit_labels),
        "selection_matrix_digest": manifest.calibration_selection_matrix_digest,
        "selection_matrix_semantic_digest": (manifest.calibration_selection_matrix_semantic_digest),
        "selection_row_ids_digest": _row_ids_digest(
            tuple(row.event_id for row in selection_matrix.rows)
        ),
        "selection_probability_scores_digest": _numeric_array_digest(selection_probabilities),
        "selection_labels_digest": _numeric_array_digest(selection_labels),
        "calibration_artifact_digest": _digest(component_bytes["calibration"]),
        "calibration_state_digest": calibrator.artifact.artifact_digest,
    }
    document = binding.model_dump(mode="python")
    if any(document[name] != value for name, value in expected.items()):
        raise BundleContractError("calibration binding receipt is inconsistent")
    if (
        calibrator.artifact.fit_window_content_digest
        != _calibration_window_digest(fit_probabilities, fit_labels)
        or calibrator.artifact.selection_window_content_digest
        != _calibration_window_digest(selection_probabilities, selection_labels)
    ):
        raise BundleContractError("calibration artifact window binding is inconsistent")


def _verify_threshold_binding(
    binding: ThresholdBindingReceipt,
    manifest: DefenderBundleManifest,
    split_semantic: str,
    scorer: CatBoostScorer,
    calibrator: ProbabilityCalibrator,
    threshold: ThresholdReport,
    matrix: FeatureMatrix,
    split: EvaluationSplit,
    component_bytes: dict[str, bytes],
) -> None:
    probabilities = scorer.predict(matrix)
    calibrated = calibrator.predict(probabilities)
    labels = np.asarray(
        [int(split.row_is_fraud[row.event_id]) for row in matrix.rows], dtype=np.int64
    )
    values = np.asarray(
        [float(split.row_net_settled_values[row.event_id]) for row in matrix.rows],
        dtype=np.float64,
    )
    values_digest = (
        None if threshold.input_values_digest is None else _numeric_array_digest(values)
    )
    expected = {
        "split_artifact_digest": manifest.split_artifact_digest,
        "split_semantic_digest": split_semantic,
        "model_digest": manifest.model_digest,
        "matrix_digest": manifest.threshold_matrix_digest,
        "matrix_semantic_digest": manifest.threshold_matrix_semantic_digest,
        "row_ids_digest": _row_ids_digest(tuple(row.event_id for row in matrix.rows)),
        "model_probability_scores_digest": _numeric_array_digest(probabilities),
        "calibrated_scores_digest": _numeric_array_digest(calibrated),
        "labels_digest": _numeric_array_digest(labels),
        "values_digest": values_digest,
        "threshold_artifact_digest": _digest(component_bytes["threshold"]),
        "threshold_report_digest": threshold.report_digest,
        "callback_contract_version": _CALLBACK_CONTRACT_VERSION,
    }
    document = binding.model_dump(mode="python")
    if any(document[name] != value for name, value in expected.items()):
        raise BundleContractError("threshold binding receipt is inconsistent")
    if (
        threshold.input_scores_digest != binding.calibrated_scores_digest
        or threshold.input_labels_digest != binding.labels_digest
        or _actions_digest(np.asarray(binding.mandatory_actions, dtype=object))
        != binding.mandatory_actions_digest
        or threshold.input_mandatory_actions_digest != binding.mandatory_actions_digest
        or threshold.input_values_digest != binding.values_digest
    ):
        raise BundleContractError("threshold report input binding is inconsistent")


def _validate_environment(
    lock: EnvironmentLock,
    scorer: CatBoostScorer,
    calibrator: ProbabilityCalibrator,
) -> None:
    if lock != current_environment_lock():
        raise BundleContractError("bundle environment lock is incompatible with this loader")
    receipt = scorer.receipt
    artifact = calibrator.artifact
    if (
        receipt.python_version != lock.python_version
        or receipt.platform != lock.platform
        or receipt.catboost_version != lock.catboost_version
        or receipt.scikit_learn_version != lock.scikit_learn_version
        or receipt.numpy_version != lock.numpy_version
        or artifact.sklearn_version != lock.scikit_learn_version
        or artifact.numpy_version != lock.numpy_version
    ):
        raise BundleContractError("model or calibrator environment differs from the bundle lock")


def _validate_catalog(catalog: FeatureCatalog) -> str:
    try:
        audit_feature_catalog(catalog)
        digest = feature_catalog_digest(catalog)
    except (TypeError, ValueError) as error:
        raise BundleContractError("feature catalog is invalid") from error
    if catalog.names != EXPECTED_FEATURE_NAMES:
        raise BundleContractError("feature catalog order is not the competition order")
    return digest


def _validate_matrix(matrix: FeatureMatrix, catalog: FeatureCatalog, *, label: str) -> None:
    if matrix.catalog != catalog or matrix.catalog_digest != feature_catalog_digest(catalog):
        raise BundleContractError(f"{label} matrix catalog binding is invalid")
    event_ids = tuple(event.event_id for event in matrix.events)
    row_ids = tuple(row.event_id for row in matrix.rows)
    if not row_ids or event_ids != row_ids or len(row_ids) != len(set(row_ids)):
        raise BundleContractError(f"{label} matrix must contain one ordered decision event per row")
    for event, row in zip(matrix.events, matrix.rows, strict=True):
        if (
            not event.is_decision_point
            or event.decision_at is None
            or event.decision_at != row.decision_at
            or row.catalog_digest != matrix.catalog_digest
            or tuple(row.values) != EXPECTED_FEATURE_NAMES
        ):
            raise BundleContractError(f"{label} matrix row/event binding is invalid")
        values = tuple(row.values[name] for name in EXPECTED_FEATURE_NAMES)
        if any(type(value) not in {int, float} or not math.isfinite(value) for value in values):
            raise BundleContractError(f"{label} matrix feature values must be finite")


def _matrix_semantic_digest(matrix: FeatureMatrix) -> str:
    document: dict[str, object] = {
        "schema_version": matrix.schema_version,
        "catalog": matrix.catalog.model_dump(mode="json"),
        "catalog_digest": matrix.catalog_digest,
        "events": [event.model_dump(mode="json") for event in matrix.events],
        "rows": [
            {
                **row.model_dump(mode="json", exclude={"values"}),
                "ordered_values": [row.values[name] for name in EXPECTED_FEATURE_NAMES],
            }
            for row in matrix.rows
        ],
    }
    return _digest(canonical_json_bytes(document))


def _matrix_schema() -> pa.Schema:
    fields = [
        pa.field("event_id", pa.string(), nullable=False),
        pa.field("payment_id", pa.string(), nullable=False),
        pa.field("rail", pa.string(), nullable=False),
        pa.field("event_type", pa.string(), nullable=False),
        pa.field("amount", pa.string(), nullable=False),
        pa.field("currency", pa.string(), nullable=False),
        pa.field("event_time", pa.timestamp("us", tz="UTC"), nullable=False),
        pa.field("available_at", pa.timestamp("us", tz="UTC"), nullable=False),
        pa.field("decision_at", pa.timestamp("us", tz="UTC"), nullable=False),
        pa.field("actor_id", pa.string(), nullable=False),
        pa.field("counterparty_id", pa.string(), nullable=False),
        pa.field("optional_refs_json", pa.string(), nullable=False),
        pa.field("integrity_status", pa.string(), nullable=False),
        pa.field("integrity_reason", pa.string(), nullable=True),
        pa.field("is_decision_point", pa.bool_(), nullable=False),
        pa.field("privacy_classification", pa.string(), nullable=False),
        pa.field(
            "source_event_ids",
            pa.list_(pa.field("element", pa.string(), nullable=True)),
            nullable=False,
        ),
        pa.field("max_source_available_at", pa.timestamp("us", tz="UTC"), nullable=True),
        pa.field("catalog_digest", pa.string(), nullable=False),
    ]
    fields.extend(pa.field(name, pa.float64(), nullable=False) for name in EXPECTED_FEATURE_NAMES)
    return pa.schema(fields, metadata=None)


def _matrix_to_parquet(matrix: FeatureMatrix) -> bytes:
    data: dict[str, list[object]] = {name: [] for name in _matrix_schema().names}
    for event, row in zip(matrix.events, matrix.rows, strict=True):
        data["event_id"].append(event.event_id)
        data["payment_id"].append(event.payment_id)
        data["rail"].append(event.rail.value)
        data["event_type"].append(event.event_type.value)
        data["amount"].append(str(event.amount))
        data["currency"].append(event.currency)
        data["event_time"].append(event.event_time)
        data["available_at"].append(event.available_at)
        data["decision_at"].append(event.decision_at)
        data["actor_id"].append(event.actor_id)
        data["counterparty_id"].append(event.counterparty_id)
        data["optional_refs_json"].append(canonical_json_bytes(event.optional_refs).decode("ascii"))
        data["integrity_status"].append(event.integrity_status)
        data["integrity_reason"].append(event.integrity_reason)
        data["is_decision_point"].append(event.is_decision_point)
        data["privacy_classification"].append(event.privacy_classification)
        data["source_event_ids"].append(list(row.source_event_ids))
        data["max_source_available_at"].append(row.max_source_available_at)
        data["catalog_digest"].append(row.catalog_digest)
        for name in EXPECTED_FEATURE_NAMES:
            data[name].append(float(row.values[name]))
    table = pa.Table.from_pydict(data, schema=_matrix_schema())
    sink = pa.BufferOutputStream()
    pq.write_table(
        table,
        sink,
        compression="NONE",
        use_dictionary=False,
        write_statistics=False,
        version="2.6",
        data_page_version="1.0",
        use_deprecated_int96_timestamps=False,
        store_schema=True,
        row_group_size=_MAX_PARQUET_ROWS,
    )
    return bytes(sink.getvalue().to_pybytes())


def _matrix_from_parquet(payload: bytes, catalog: FeatureCatalog) -> FeatureMatrix:
    if type(payload) is not bytes:
        raise BundleContractError("matrix Parquet payload must be exact bytes")
    if len(payload) > _MAX_PARQUET_BYTES:
        raise BundleContractError("matrix Parquet exceeds its encoded byte limit")
    try:
        parquet = pq.ParquetFile(pa.BufferReader(payload))
        metadata = parquet.metadata
        expected = _matrix_schema()
        if (
            metadata.num_rows < 1
            or metadata.num_rows > _MAX_PARQUET_ROWS
            or metadata.num_row_groups != _MAX_PARQUET_ROW_GROUPS
            or metadata.num_columns != len(expected)
            or not parquet.schema_arrow.equals(expected, check_metadata=True)
        ):
            raise BundleContractError("matrix Parquet metadata exceeds its closed bounds")
        decoded_bytes = 0
        encoded_bytes = 0
        for row_group_index in range(metadata.num_row_groups):
            row_group = metadata.row_group(row_group_index)
            for column_index in range(row_group.num_columns):
                column = row_group.column(column_index)
                if column.compression != "UNCOMPRESSED":
                    raise BundleContractError("matrix Parquet compression is not permitted")
                decoded_bytes += column.total_uncompressed_size
                encoded_bytes += column.total_compressed_size
        if decoded_bytes > _MAX_PARQUET_DECODED_BYTES or encoded_bytes > _MAX_PARQUET_BYTES:
            raise BundleContractError("matrix Parquet decoded size exceeds its budget")
        table = parquet.read()
    except BundleContractError:
        raise
    except (MemoryError, OSError, pa.ArrowException) as error:
        raise BundleContractError("matrix Parquet could not be loaded") from error
    if not table.schema.equals(expected, check_metadata=True):
        raise BundleContractError("matrix Parquet schema, order, types, or metadata differ")
    columns = table.to_pydict()
    events: list[ObservedEvent] = []
    rows: list[FeatureVector] = []
    for index in range(table.num_rows):
        optional_raw = cast(str, columns["optional_refs_json"][index]).encode("ascii")
        optional = strict_json_loads(optional_raw)
        if type(optional) is not dict or any(
            type(key) is not str or type(value) is not str
            for key, value in cast(dict[object, object], optional).items()
        ):
            raise BundleContractError("matrix optional references are invalid")
        decision_at = cast(datetime, columns["decision_at"][index])
        event = ObservedEvent(
            event_id=columns["event_id"][index],
            payment_id=columns["payment_id"][index],
            rail=columns["rail"][index],
            event_type=columns["event_type"][index],
            amount=Decimal(cast(str, columns["amount"][index])),
            currency=columns["currency"][index],
            event_time=columns["event_time"][index],
            available_at=columns["available_at"][index],
            decision_at=decision_at,
            actor_id=columns["actor_id"][index],
            counterparty_id=columns["counterparty_id"][index],
            optional_refs=cast(dict[str, str], optional),
            integrity_status=columns["integrity_status"][index],
            integrity_reason=columns["integrity_reason"][index],
            is_decision_point=columns["is_decision_point"][index],
            privacy_classification=columns["privacy_classification"][index],
        )
        values = {name: cast(float, columns[name][index]) for name in EXPECTED_FEATURE_NAMES}
        rows.append(
            FeatureVector(
                event_id=event.event_id,
                decision_at=decision_at,
                source_event_ids=tuple(columns["source_event_ids"][index]),
                max_source_available_at=columns["max_source_available_at"][index],
                catalog_digest=columns["catalog_digest"][index],
                values=values,
            )
        )
        events.append(event)
    matrix = FeatureMatrix(
        events=tuple(events),
        catalog=catalog,
        catalog_digest=feature_catalog_digest(catalog),
        rows=tuple(rows),
    )
    _validate_matrix(matrix, catalog, label="loaded")
    return matrix


def _verify_reload_parity(
    scorer: CatBoostScorer,
    calibrator: ProbabilityCalibrator,
    matrix: FeatureMatrix,
    fixture: _ReloadFixture,
) -> None:
    raw = scorer.predict_raw(matrix)
    probability = scorer.predict(matrix)
    calibrated = calibrator.predict(probability)
    expected_raw = np.asarray(fixture.raw_scores, dtype=np.float64)
    expected_probability = np.asarray(fixture.probability_scores, dtype=np.float64)
    expected_calibrated = np.asarray(fixture.calibrated_scores, dtype=np.float64)
    if not np.allclose(raw, expected_raw, rtol=0.0, atol=1e-12):
        raise BundleContractError("reload raw-score parity failed")
    if not np.array_equal(probability, expected_probability):
        raise BundleContractError("reload probability-score parity failed")
    if not np.array_equal(calibrated, expected_calibrated):
        raise BundleContractError("reload calibrated-score parity failed")
