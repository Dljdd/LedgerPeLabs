"""Signed, content-addressed, native-only frozen defender bundles."""

from __future__ import annotations

import base64
import binascii
import hashlib
import math
import platform as platform_module
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from pathlib import PurePosixPath
from typing import Literal, cast
from uuid import UUID

import catboost  # type: ignore[import-untyped]
import numpy as np
import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]
import sklearn  # type: ignore[import-untyped]
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
from pydantic import ValidationError, field_validator, model_validator

from apar.contracts._validation import ExternalContract, validate_utc_timestamp
from apar.defense.calibration import ProbabilityCalibrator
from apar.defense.contracts import ObservedEvent
from apar.defense.gbdt import CatBoostScorer, TrainingReceipt
from apar.defense.rules import RuleManifest, rule_manifest_digest
from apar.defense.thresholds import ThresholdReport
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


class EnvironmentLock(ExternalContract):
    """Exact loader environment; portability outside it is not claimed."""

    schema_version: Literal["1.0.0"] = "1.0.0"
    python_version: str
    platform: str
    catboost_version: str
    scikit_learn_version: str
    numpy_version: str
    pyarrow_version: str
    apar_schema_version: Literal["1.0.0"] = "1.0.0"

    @field_validator(
        "python_version",
        "platform",
        "catboost_version",
        "scikit_learn_version",
        "numpy_version",
        "pyarrow_version",
    )
    @classmethod
    def versions_are_exact_text(cls, value: str) -> str:
        if type(value) is not str or not value or value.strip() != value:
            raise ValueError("environment versions must be exact nonblank text")
        return value


def current_environment_lock() -> EnvironmentLock:
    """Return the exact environment accepted by this loader."""
    return EnvironmentLock(
        python_version=platform_module.python_version(),
        platform=platform_module.platform(),
        catboost_version=catboost.__version__,
        scikit_learn_version=sklearn.__version__,
        numpy_version=np.__version__,
        pyarrow_version=pa.__version__,
        apar_schema_version="1.0.0",
    )


class SourceInventoryEntry(ExternalContract):
    """One public source path and its immutable content digest."""

    path: str
    sha256: str

    @field_validator("path")
    @classmethod
    def path_is_public_relative_posix(cls, value: str) -> str:
        if type(value) is not str or not value or "\\" in value:
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
        return value


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
    feature_catalog_digest: str
    feature_semantic_digest: str
    training_matrix_digest: str
    training_matrix_semantic_digest: str
    rule_manifest_digest: str
    rule_semantic_digest: str
    model_digest: str
    training_receipt_digest: str
    calibration_digest: str
    threshold_digest: str
    environment_digest: str
    source_inventory_digest: str
    reload_matrix_digest: str
    reload_matrix_semantic_digest: str
    reload_fixture_digest: str
    fallback_mode: Literal["rules_only"] = "rules_only"
    rollback_ref: str
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
        "feature_catalog_digest",
        "feature_semantic_digest",
        "training_matrix_digest",
        "training_matrix_semantic_digest",
        "rule_manifest_digest",
        "rule_semantic_digest",
        "model_digest",
        "training_receipt_digest",
        "calibration_digest",
        "threshold_digest",
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

    def component_digests(self) -> tuple[str, ...]:
        """Return all directly stored component addresses in stable order."""
        return (
            self.feature_catalog_digest,
            self.training_matrix_digest,
            self.rule_manifest_digest,
            self.model_digest,
            self.training_receipt_digest,
            self.calibration_digest,
            self.threshold_digest,
            self.environment_digest,
            self.source_inventory_digest,
            self.reload_matrix_digest,
            self.reload_fixture_digest,
        )


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


@dataclass(frozen=True, slots=True)
class LoadedDefenderBundle:
    """Fully verified runtime components, released only after reload parity."""

    manifest: DefenderBundleManifest
    scorer: CatBoostScorer
    catalog: FeatureCatalog
    training_matrix: FeatureMatrix
    rule_manifest: RuleManifest
    calibrator: ProbabilityCalibrator
    threshold_report: ThresholdReport
    environment_lock: EnvironmentLock
    source_inventory: SourceInventory
    reload_matrix: FeatureMatrix


class DefenderBundlePublisher:
    """Publish and load defender bundles under one exact store and signing authority."""

    __slots__ = ("_store", "_signer")

    def __init__(self, store: ArtifactStore, signer: RunSigningIdentity) -> None:
        if type(store) is not ArtifactStore:
            raise BundleContractError("publisher requires an exact ArtifactStore")
        if type(signer) is not RunSigningIdentity:
            raise BundleContractError("publisher requires an exact RunSigningIdentity")
        self._store = store
        self._signer = signer

    def freeze(
        self,
        *,
        scorer: CatBoostScorer,
        catalog: FeatureCatalog,
        training_matrix: FeatureMatrix,
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
        try:
            components = self._prepare_components(
                scorer=scorer,
                catalog=catalog,
                training_matrix=training_matrix,
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
            document: dict[str, object] = {
                "schema_version": _SCHEMA_VERSION,
                "bundle_id": bundle_id,
                **lineage.model_dump(mode="json", exclude={"schema_version"}),
                "feature_catalog_digest": refs["catalog"].sha256,
                "feature_semantic_digest": components.feature_semantic_digest,
                "training_matrix_digest": refs["training_matrix"].sha256,
                "training_matrix_semantic_digest": components.training_semantic_digest,
                "rule_manifest_digest": refs["rules"].sha256,
                "rule_semantic_digest": components.rule_semantic_digest,
                "model_digest": refs["model"].sha256,
                "training_receipt_digest": refs["receipt"].sha256,
                "calibration_digest": refs["calibration"].sha256,
                "threshold_digest": refs["threshold"].sha256,
                "environment_digest": refs["environment"].sha256,
                "source_inventory_digest": refs["source_inventory"].sha256,
                "reload_matrix_digest": refs["reload_matrix"].sha256,
                "reload_matrix_semantic_digest": components.reload_semantic_digest,
                "reload_fixture_digest": refs["reload_fixture"].sha256,
                "fallback_mode": "rules_only",
                "rollback_ref": rollback_ref,
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
            if type(manifest) is not DefenderBundleManifest:
                return False
            validated = DefenderBundleManifest.model_validate(manifest.model_dump(mode="python"))
            self._load_validated(validated, visited=set(), depth=0)
            return True
        except Exception:
            return False

    def load(self, ref: ArtifactRef) -> LoadedDefenderBundle:
        """Load only after top-ref, signature, chain, component, and parity checks."""
        try:
            if type(ref) is not ArtifactRef or ref.media_type != _BUNDLE_MEDIA:
                raise BundleContractError("bundle reference has an invalid type or media type")
            payload = self._store.read(ref)
            document = strict_json_loads(payload)
            manifest = _manifest_from_document(document)
            if canonical_json_bytes(manifest.model_dump(mode="json")) != payload:
                raise BundleContractError("bundle manifest is not canonical")
            return self._load_validated(manifest, visited={ref.sha256}, depth=0)
        except BundleContractError:
            raise
        except Exception as error:
            raise BundleContractError(f"bundle load failed: {error}") from error

    def _prepare_components(
        self,
        *,
        scorer: CatBoostScorer,
        catalog: FeatureCatalog,
        training_matrix: FeatureMatrix,
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
            training_matrix=training_matrix,
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
            training_matrix = FeatureMatrix.model_validate(training_matrix)
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
        if not threshold_report.feasible or threshold_report.thresholds is None:
            raise BundleContractError("only a feasible threshold report may be frozen")
        if rollback_ref != GENESIS_ROLLBACK_REF:
            _validate_digest(rollback_ref, label="rollback reference")
            predecessor = self._load_manifest_ref(rollback_ref)
            if predecessor.frozen_at >= frozen_at:
                raise BundleContractError(
                    "rollback predecessor must be earlier than the new bundle"
                )
            if predecessor.bundle_id == bundle_id:
                raise BundleContractError("rollback predecessor must have a distinct bundle ID")
            self._load_validated(predecessor, visited={rollback_ref}, depth=1)

        feature_semantic = _validate_catalog(catalog)
        _validate_matrix(training_matrix, catalog, label="training")
        _validate_matrix(reload_matrix, catalog, label="reload")
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
        if len(training_ids) != receipt.requested_training_count:
            raise BundleContractError("training matrix count does not match the receipt")
        if (
            _digest(canonical_json_bytes(list(training_ids)))
            != receipt.requested_training_row_ids_digest
        ):
            raise BundleContractError("training matrix row IDs do not match the receipt")
        if any(row.decision_at > receipt.training_cutoff for row in training_matrix.rows):
            raise BundleContractError("training matrix contains a row after its cutoff")
        reloaded_scorer = CatBoostScorer.from_bytes(scorer.to_bytes(), receipt)

        training_payload = _matrix_to_parquet(training_matrix)
        reload_payload = _matrix_to_parquet(reload_matrix)
        if _matrix_from_parquet(training_payload, catalog) != training_matrix:
            raise BundleContractError("training Parquet does not reconstruct exactly")
        if _matrix_from_parquet(reload_payload, catalog) != reload_matrix:
            raise BundleContractError("reload Parquet does not reconstruct exactly")
        training_semantic = _matrix_semantic_digest(training_matrix)
        reload_semantic = _matrix_semantic_digest(reload_matrix)
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
        payloads: dict[str, tuple[bytes, str]] = {
            "catalog": (canonical_json_bytes(catalog.model_dump(mode="json")), _CATALOG_MEDIA),
            "training_matrix": (training_payload, _PARQUET_MEDIA),
            "rules": (canonical_json_bytes(rule_manifest.model_dump(mode="json")), _RULE_MEDIA),
            "model": (scorer.to_bytes(), _MODEL_MEDIA),
            "receipt": (canonical_json_bytes(receipt.model_dump(mode="json")), _RECEIPT_MEDIA),
            "calibration": (calibrator.to_json(), _CALIBRATION_MEDIA),
            "threshold": (threshold_report.to_json(), _THRESHOLD_MEDIA),
            "environment": (
                canonical_json_bytes(environment_lock.model_dump(mode="json")),
                _ENVIRONMENT_MEDIA,
            ),
            "source_inventory": (
                canonical_json_bytes(source_inventory.model_dump(mode="json")),
                _SOURCE_MEDIA,
            ),
            "reload_matrix": (reload_payload, _PARQUET_MEDIA),
            "reload_fixture": (
                canonical_json_bytes(fixture.model_dump(mode="json")),
                _RELOAD_MEDIA,
            ),
        }
        return _PreparedComponents(
            payloads,
            feature_semantic,
            training_semantic,
            reload_semantic,
            rule_semantic,
        )

    def _load_validated(
        self,
        manifest: DefenderBundleManifest,
        *,
        visited: set[str],
        depth: int,
    ) -> LoadedDefenderBundle:
        if depth > _MAX_ROLLBACK_DEPTH:
            raise BundleContractError("rollback chain exceeds its maximum depth")
        if not self._verify_signature(manifest):
            raise BundleContractError("bundle signature or pinned authority is invalid")
        if manifest.rollback_ref != GENESIS_ROLLBACK_REF:
            if manifest.rollback_ref in visited:
                raise BundleContractError("rollback chain contains a cycle")
            predecessor = self._load_manifest_ref(manifest.rollback_ref)
            if predecessor.frozen_at >= manifest.frozen_at:
                raise BundleContractError("rollback predecessor is not earlier")
            if predecessor.bundle_id == manifest.bundle_id:
                raise BundleContractError("rollback predecessor reuses the current bundle ID")
            self._load_validated(
                predecessor,
                visited={*visited, manifest.rollback_ref},
                depth=depth + 1,
            )

        catalog = FeatureCatalog.model_validate(
            self._read_json_component(manifest.feature_catalog_digest, _CATALOG_MEDIA)
        )
        semantic = _validate_catalog(catalog)
        if semantic != manifest.feature_semantic_digest:
            raise BundleContractError("feature catalog semantic digest mismatch")
        receipt = TrainingReceipt.model_validate(
            self._read_json_component(manifest.training_receipt_digest, _RECEIPT_MEDIA)
        )
        model_payload = self._read_component(manifest.model_digest, _MODEL_MEDIA)
        scorer = CatBoostScorer.from_bytes(model_payload, receipt)
        if receipt.catalog_digest != semantic:
            raise BundleContractError("training receipt catalog mismatch")
        rules = RuleManifest.model_validate(
            self._read_json_component(manifest.rule_manifest_digest, _RULE_MEDIA)
        )
        if rule_manifest_digest(rules) != manifest.rule_semantic_digest:
            raise BundleContractError("rule manifest semantic digest mismatch")
        calibrator = ProbabilityCalibrator.from_json(
            self._read_component(manifest.calibration_digest, _CALIBRATION_MEDIA)
        )
        threshold = ThresholdReport.from_json(
            self._read_component(manifest.threshold_digest, _THRESHOLD_MEDIA)
        )
        if not threshold.feasible or threshold.thresholds is None:
            raise BundleContractError("frozen threshold report is infeasible")
        environment = EnvironmentLock.model_validate(
            self._read_json_component(manifest.environment_digest, _ENVIRONMENT_MEDIA)
        )
        source_inventory = SourceInventory.model_validate(
            self._read_json_component(manifest.source_inventory_digest, _SOURCE_MEDIA)
        )
        _validate_environment(environment, scorer, calibrator)
        training = _matrix_from_parquet(
            self._read_component(manifest.training_matrix_digest, _PARQUET_MEDIA), catalog
        )
        reload_matrix = _matrix_from_parquet(
            self._read_component(manifest.reload_matrix_digest, _PARQUET_MEDIA), catalog
        )
        if _matrix_semantic_digest(training) != manifest.training_matrix_semantic_digest:
            raise BundleContractError("training matrix semantic digest mismatch")
        if _matrix_semantic_digest(reload_matrix) != manifest.reload_matrix_semantic_digest:
            raise BundleContractError("reload matrix semantic digest mismatch")
        training_ids = tuple(row.event_id for row in training.rows)
        if (
            _digest(canonical_json_bytes(list(training_ids)))
            != receipt.requested_training_row_ids_digest
            or len(training_ids) != receipt.requested_training_count
            or any(row.decision_at > receipt.training_cutoff for row in training.rows)
        ):
            raise BundleContractError("training matrix does not match the model receipt")
        fixture = _ReloadFixture.model_validate(
            self._read_json_component(manifest.reload_fixture_digest, _RELOAD_MEDIA)
        )
        if fixture.matrix_semantic_digest != manifest.reload_matrix_semantic_digest:
            raise BundleContractError("reload fixture matrix binding mismatch")
        _verify_reload_parity(scorer, calibrator, reload_matrix, fixture)
        return LoadedDefenderBundle(
            manifest=manifest,
            scorer=scorer,
            catalog=catalog,
            training_matrix=training,
            rule_manifest=rules,
            calibrator=calibrator,
            threshold_report=threshold,
            environment_lock=environment,
            source_inventory=source_inventory,
            reload_matrix=reload_matrix,
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

    def _read_component(self, digest: str, media_type: str) -> bytes:
        ref = self._store.resolve(digest)
        if ref.media_type != media_type or ref.sha256 != digest:
            raise BundleContractError("component media type or digest mismatch")
        payload = self._store.read(ref)
        if len(payload) != ref.size_bytes or _digest(payload) != digest:
            raise BundleContractError("component size or payload digest mismatch")
        return payload

    def _read_json_component(self, digest: str, media_type: str) -> object:
        return strict_json_loads(self._read_component(digest, media_type))

    def _load_manifest_ref(self, digest: str) -> DefenderBundleManifest:
        document = self._read_json_component(digest, _BUNDLE_MEDIA)
        return _manifest_from_document(document)


@dataclass(frozen=True, slots=True)
class _PreparedComponents:
    payloads: dict[str, tuple[bytes, str]]
    feature_semantic_digest: str
    training_semantic_digest: str
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
        "training_matrix": FeatureMatrix,
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
    )
    return bytes(sink.getvalue().to_pybytes())


def _matrix_from_parquet(payload: bytes, catalog: FeatureCatalog) -> FeatureMatrix:
    if type(payload) is not bytes:
        raise BundleContractError("matrix Parquet payload must be exact bytes")
    try:
        table = pq.read_table(pa.BufferReader(payload))
    except (OSError, pa.ArrowException) as error:
        raise BundleContractError("matrix Parquet could not be loaded") from error
    expected = _matrix_schema()
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
