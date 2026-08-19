"""Neutral signed-defender attestation without importing defender implementation."""

from __future__ import annotations

import base64
import binascii
import hashlib
from datetime import datetime
from typing import NamedTuple, Never, cast
from uuid import UUID

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

from apar.contracts._validation import validate_utc_timestamp
from apar.runs.wire import WireContractError, canonical_json_bytes, strict_json_loads
from apar.storage.artifacts import ArtifactRef, ArtifactStore

_ATTESTATION_TOKEN = object()
_BUNDLE_MEDIA = "application/vnd.apar.defender-bundle+json"
_MAX_BUNDLE_BYTES = 1024 * 1024
_MAX_ROLLBACK_BYTES = 8 * 1024 * 1024
_MAX_ROLLBACK_DEPTH = 32
_MAX_ATTESTATION_BYTES = 64 * 1024
_GENESIS = "genesis"

_COMPONENT_FIELD_MEDIA: dict[str, tuple[str, str, int]] = {
    "calibration": ("calibration_digest", "application/vnd.apar.calibration+json", 16 << 20),
    "calibration_binding": (
        "calibration_binding_digest",
        "application/vnd.apar.calibration-binding+json",
        16 << 20,
    ),
    "calibration_fit_matrix": (
        "calibration_fit_matrix_digest",
        "application/vnd.apache.parquet",
        128 << 20,
    ),
    "calibration_selection_matrix": (
        "calibration_selection_matrix_digest",
        "application/vnd.apache.parquet",
        128 << 20,
    ),
    "catalog": ("feature_catalog_digest", "application/vnd.apar.feature-catalog+json", 16 << 20),
    "environment": ("environment_digest", "application/vnd.apar.environment-lock+json", 16 << 20),
    "model": ("model_digest", "application/vnd.apar.catboost-model", 128 << 20),
    "receipt": (
        "training_receipt_digest",
        "application/vnd.apar.training-receipt+json",
        16 << 20,
    ),
    "reload_fixture": (
        "reload_fixture_digest",
        "application/vnd.apar.reload-fixture+json",
        16 << 20,
    ),
    "reload_matrix": ("reload_matrix_digest", "application/vnd.apache.parquet", 128 << 20),
    "rules": ("rule_manifest_digest", "application/vnd.apar.rule-manifest+json", 16 << 20),
    "source_inventory": (
        "source_inventory_digest",
        "application/vnd.apar.source-inventory+json",
        16 << 20,
    ),
    "split": ("split_artifact_digest", "application/vnd.apar.evaluation-split+json", 16 << 20),
    "threshold": ("threshold_digest", "application/vnd.apar.threshold-report+json", 16 << 20),
    "threshold_binding": (
        "threshold_binding_digest",
        "application/vnd.apar.threshold-binding+json",
        16 << 20,
    ),
    "threshold_matrix": ("threshold_matrix_digest", "application/vnd.apache.parquet", 128 << 20),
    "training_binding": (
        "training_binding_digest",
        "application/vnd.apar.training-binding+json",
        16 << 20,
    ),
    "training_matrix": ("training_matrix_digest", "application/vnd.apache.parquet", 128 << 20),
}

_DIGEST_FIELDS = {
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
}

_MANIFEST_FIELDS = {
    "schema_version",
    "bundle_id",
    *_DIGEST_FIELDS,
    "components",
    "fallback_mode",
    "rollback_ref",
    "rollback_size_bytes",
    "public_key_base64",
    "signature_base64",
    "frozen_at",
}


class DefenderAttestationError(ValueError):
    """A bundle could not be authenticated by the neutral pinned verifier."""


class _VerifierState(NamedTuple):
    store: ArtifactStore
    signer_key_id: str
    public_key_base64: str
    public_key: bytes


class VerifiedDefenderAttestation(tuple[bytes]):
    """Intrinsically immutable canonical proof of one authenticated bundle."""

    __slots__ = ()

    def __new__(
        cls, payload: bytes, token: object = None
    ) -> VerifiedDefenderAttestation:
        if token is not _ATTESTATION_TOKEN or type(payload) is not bytes:
            raise DefenderAttestationError(
                "attestations must come from the exact neutral verifier"
            )
        _attestation_document(payload)
        return tuple.__new__(cls, (bytes(payload),))

    def __init__(self, payload: bytes, token: object = None) -> None:
        del payload
        if token is not _ATTESTATION_TOKEN:
            raise DefenderAttestationError("attestation cannot be reinitialized")

    def __setattr__(self, name: str, value: object) -> None:
        del name, value
        raise TypeError("verified defender attestation is immutable")

    def __delattr__(self, name: str) -> None:
        del name
        raise TypeError("verified defender attestation is immutable")

    @classmethod
    def model_construct(cls, **values: object) -> Never:
        del values
        raise DefenderAttestationError("attestation model construction is forbidden")

    @classmethod
    def from_json(
        cls, payload: bytes, *, verifier: DefenderBundleVerifier
    ) -> VerifiedDefenderAttestation:
        if type(verifier) is not DefenderBundleVerifier:
            raise DefenderAttestationError("attestation requires the exact verifier")
        return verifier.attestation_from_json(payload)

    def to_json(self) -> bytes:
        return bytes(tuple.__getitem__(self, 0))

    def __reduce__(self) -> Never:
        raise TypeError("attestations must be reloaded through their pinned verifier")

    @property
    def top_ref(self) -> ArtifactRef:
        document = cast(dict[str, object], _attestation_document(self.to_json())["top_ref"])
        return ArtifactRef(
            sha256=cast(str, document["sha256"]),
            media_type=cast(str, document["media_type"]),
            size_bytes=cast(int, document["size_bytes"]),
            relative_path=cast(str, document["relative_path"]),
        )

    @property
    def bundle_manifest_digest(self) -> str:
        return cast(str, _attestation_document(self.to_json())["bundle_manifest_digest"])

    @property
    def bundle_id(self) -> str:
        return cast(str, _attestation_document(self.to_json())["bundle_id"])

    @property
    def threshold_digest(self) -> str:
        return cast(str, _attestation_document(self.to_json())["threshold_digest"])

    @property
    def rollback_available(self) -> bool:
        return cast(bool, _attestation_document(self.to_json())["rollback_available"])

    @property
    def rollback_predecessor_digest(self) -> str | None:
        return cast(
            str | None,
            _attestation_document(self.to_json())["rollback_predecessor_digest"],
        )

    @property
    def frozen_at(self) -> datetime:
        value = cast(str, _attestation_document(self.to_json())["frozen_at"])
        return _parse_utc(value, label="attestation frozen time")

    @property
    def attestation_digest(self) -> str:
        return cast(str, _attestation_document(self.to_json())["attestation_digest"])


class DefenderBundleVerifier(tuple[_VerifierState]):
    """Concrete store-and-Ed25519-rooted verifier for signed bundle manifests."""

    __slots__ = ()

    def __new__(
        cls,
        store: ArtifactStore,
        *,
        signer_key_id: str,
        public_key_base64: str,
    ) -> DefenderBundleVerifier:
        if type(store) is not ArtifactStore:
            raise DefenderAttestationError("neutral verifier requires an exact ArtifactStore")
        public = _canonical_base64(public_key_base64, 32, label="pinned public key")
        _digest(signer_key_id, label="pinned signer key ID")
        if hashlib.sha256(public).hexdigest() != signer_key_id:
            raise DefenderAttestationError("pinned signer identity is inconsistent")
        return tuple.__new__(
            cls,
            (
                _VerifierState(
                    store=store,
                    signer_key_id=signer_key_id,
                    public_key_base64=public_key_base64,
                    public_key=public,
                ),
            ),
        )

    def __init__(
        self,
        store: ArtifactStore,
        *,
        signer_key_id: str,
        public_key_base64: str,
    ) -> None:
        del store, signer_key_id, public_key_base64

    def __setattr__(self, name: str, value: object) -> None:
        del name, value
        raise TypeError("neutral defender verifier is immutable")

    def __delattr__(self, name: str) -> None:
        del name
        raise TypeError("neutral defender verifier is immutable")

    def __reduce__(self) -> Never:
        raise TypeError("neutral defender verifier cannot be serialized")

    def attest(self, top_ref: ArtifactRef) -> VerifiedDefenderAttestation:
        """Authenticate a top ref, every component, and its rollback ancestry."""
        state = _state(self)
        if type(top_ref) is not ArtifactRef:
            raise DefenderAttestationError("bundle top reference must be exact")
        if top_ref.media_type != _BUNDLE_MEDIA or top_ref.size_bytes > _MAX_BUNDLE_BYTES:
            raise DefenderAttestationError("bundle top reference violates its media contract")
        try:
            document, chain = _verified_manifest(
                state,
                top_ref,
                visited=frozenset(),
                depth=0,
                cumulative_bytes=0,
            )
            fields: dict[str, object] = {
                "schema_version": "1.0.0",
                "top_ref": _ref_document(top_ref),
                "bundle_manifest_digest": top_ref.sha256,
                "bundle_id": document["bundle_id"],
                "threshold_digest": document["threshold_digest"],
                "component_manifest_digest": hashlib.sha256(
                    canonical_json_bytes(document["components"])
                ).hexdigest(),
                "rollback_available": document["rollback_ref"] != _GENESIS,
                "rollback_predecessor_digest": (
                    None if document["rollback_ref"] == _GENESIS else document["rollback_ref"]
                ),
                "rollback_chain_digest": hashlib.sha256(
                    canonical_json_bytes(list(chain))
                ).hexdigest(),
                "signer_key_id": state.signer_key_id,
                "public_key_base64": state.public_key_base64,
                "frozen_at": document["frozen_at"],
            }
            digest = hashlib.sha256(canonical_json_bytes(fields)).hexdigest()
            return VerifiedDefenderAttestation(
                canonical_json_bytes({**fields, "attestation_digest": digest}),
                _ATTESTATION_TOKEN,
            )
        except DefenderAttestationError:
            raise
        except Exception as error:
            raise DefenderAttestationError("signed defender attestation failed") from error

    def verify(self, attestation: object) -> bool:
        if type(attestation) is not VerifiedDefenderAttestation:
            return False
        try:
            return self.attest(attestation.top_ref).to_json() == attestation.to_json()
        except Exception:
            return False

    def attestation_from_json(self, payload: bytes) -> VerifiedDefenderAttestation:
        if type(payload) is not bytes or len(payload) > _MAX_ATTESTATION_BYTES:
            raise DefenderAttestationError("attestation payload is invalid")
        candidate = VerifiedDefenderAttestation(payload, _ATTESTATION_TOKEN)
        actual = self.attest(candidate.top_ref)
        if actual.to_json() != payload:
            raise DefenderAttestationError("attestation does not match pinned authority")
        return actual


def _state(verifier: DefenderBundleVerifier) -> _VerifierState:
    if type(verifier) is not DefenderBundleVerifier or tuple.__len__(verifier) != 1:
        raise DefenderAttestationError("neutral verifier is not initialized")
    state = tuple.__getitem__(verifier, 0)
    if type(state) is not _VerifierState or type(state.store) is not ArtifactStore:
        raise DefenderAttestationError("neutral verifier snapshot is invalid")
    public = _canonical_base64(
        state.public_key_base64, 32, label="pinned public key"
    )
    if public != state.public_key or hashlib.sha256(public).hexdigest() != state.signer_key_id:
        raise DefenderAttestationError("neutral verifier snapshot is inconsistent")
    return state


def _verified_manifest(
    state: _VerifierState,
    ref: ArtifactRef,
    *,
    visited: frozenset[str],
    depth: int,
    cumulative_bytes: int,
) -> tuple[dict[str, object], tuple[str, ...]]:
    if depth > _MAX_ROLLBACK_DEPTH or cumulative_bytes + ref.size_bytes > _MAX_ROLLBACK_BYTES:
        raise DefenderAttestationError("rollback ancestry exceeds its resource cap")
    if ref.sha256 in visited:
        raise DefenderAttestationError("rollback ancestry contains a cycle")
    payload = state.store.read(ref)
    try:
        parsed = strict_json_loads(payload)
    except WireContractError as error:
        raise DefenderAttestationError("bundle manifest JSON is invalid") from error
    if type(parsed) is not dict or set(parsed) != _MANIFEST_FIELDS:
        raise DefenderAttestationError("bundle manifest fields are not exact")
    document = cast(dict[str, object], parsed)
    if (
        canonical_json_bytes(document) != payload
        or hashlib.sha256(payload).hexdigest() != ref.sha256
    ):
        raise DefenderAttestationError("bundle top reference is not canonical")
    _verify_manifest_document(state, document)
    _verify_components(state.store, document)
    rollback = document["rollback_ref"]
    rollback_size = document["rollback_size_bytes"]
    if rollback == _GENESIS:
        if rollback_size != 0:
            raise DefenderAttestationError("genesis rollback size is invalid")
        return document, (ref.sha256,)
    _digest(rollback, label="rollback reference")
    if type(rollback_size) is not int or rollback_size <= 0 or rollback_size > _MAX_BUNDLE_BYTES:
        raise DefenderAttestationError("rollback predecessor size is invalid")
    try:
        predecessor_ref = state.store.resolve(cast(str, rollback))
    except Exception as error:
        raise DefenderAttestationError("rollback predecessor is not loadable") from error
    if predecessor_ref.media_type != _BUNDLE_MEDIA or predecessor_ref.size_bytes != rollback_size:
        raise DefenderAttestationError("rollback predecessor reference is inconsistent")
    predecessor, chain = _verified_manifest(
        state,
        predecessor_ref,
        visited=visited | {ref.sha256},
        depth=depth + 1,
        cumulative_bytes=cumulative_bytes + ref.size_bytes,
    )
    if _parse_utc(cast(str, predecessor["frozen_at"]), label="rollback time") >= _parse_utc(
        cast(str, document["frozen_at"]), label="bundle time"
    ):
        raise DefenderAttestationError("rollback predecessor is not earlier")
    if predecessor["bundle_id"] == document["bundle_id"]:
        raise DefenderAttestationError("rollback predecessor reuses bundle identity")
    return document, (ref.sha256, *chain)


def _verify_manifest_document(state: _VerifierState, document: dict[str, object]) -> None:
    if document["schema_version"] != "1.0.0" or document["fallback_mode"] != "rules_only":
        raise DefenderAttestationError("bundle manifest schema is unsupported")
    bundle_id = document["bundle_id"]
    if type(bundle_id) is not str:
        raise DefenderAttestationError("bundle ID is invalid")
    try:
        if str(UUID(bundle_id)) != bundle_id:
            raise ValueError
    except ValueError as error:
        raise DefenderAttestationError("bundle ID is invalid") from error
    for name in _DIGEST_FIELDS:
        _digest(document[name], label=name)
    if (
        document["signer_key_id"] != state.signer_key_id
        or document["public_key_base64"] != state.public_key_base64
    ):
        raise DefenderAttestationError("bundle signer is not the pinned authority")
    signature = _canonical_base64(document["signature_base64"], 64, label="bundle signature")
    _parse_utc(document["frozen_at"], label="bundle frozen time")
    unsigned = {key: value for key, value in document.items() if key != "signature_base64"}
    try:
        Ed25519PublicKey.from_public_bytes(state.public_key).verify(
            signature, canonical_json_bytes(unsigned)
        )
    except InvalidSignature as error:
        raise DefenderAttestationError("bundle signature is invalid") from error


def _verify_components(store: ArtifactStore, manifest: dict[str, object]) -> None:
    components = manifest["components"]
    if type(components) is not list or len(components) != len(_COMPONENT_FIELD_MEDIA):
        raise DefenderAttestationError("bundle component descriptors are incomplete")
    names: list[str] = []
    for raw in components:
        if type(raw) is not dict or set(raw) != {"name", "sha256", "media_type", "size_bytes"}:
            raise DefenderAttestationError("bundle component descriptor is invalid")
        descriptor = cast(dict[str, object], raw)
        name = descriptor["name"]
        if type(name) is not str or name not in _COMPONENT_FIELD_MEDIA:
            raise DefenderAttestationError("bundle component name is invalid")
        names.append(name)
        field, media_type, size_cap = _COMPONENT_FIELD_MEDIA[name]
        sha256 = descriptor["sha256"]
        size = descriptor["size_bytes"]
        if (
            sha256 != manifest[field]
            or descriptor["media_type"] != media_type
            or type(size) is not int
            or size < 0
            or size > size_cap
        ):
            raise DefenderAttestationError("bundle component descriptor is inconsistent")
        digest = _digest(sha256, label="component digest")
        ref = ArtifactRef(digest, media_type, size, f"{digest}/payload")
        try:
            store.read(ref)
        except Exception as error:
            raise DefenderAttestationError("bundle component is not loadable") from error
    if names != sorted(_COMPONENT_FIELD_MEDIA):
        raise DefenderAttestationError("bundle component descriptors are not canonical")


def _attestation_document(payload: bytes) -> dict[str, object]:
    if type(payload) is not bytes or not payload or len(payload) > _MAX_ATTESTATION_BYTES:
        raise DefenderAttestationError("attestation payload is invalid")
    try:
        parsed = strict_json_loads(payload)
    except WireContractError as error:
        raise DefenderAttestationError("attestation JSON is invalid") from error
    expected = {
        "schema_version",
        "top_ref",
        "bundle_manifest_digest",
        "bundle_id",
        "threshold_digest",
        "component_manifest_digest",
        "rollback_available",
        "rollback_predecessor_digest",
        "rollback_chain_digest",
        "signer_key_id",
        "public_key_base64",
        "frozen_at",
        "attestation_digest",
    }
    if (
        type(parsed) is not dict
        or set(parsed) != expected
        or canonical_json_bytes(parsed) != payload
    ):
        raise DefenderAttestationError("attestation fields are not exact and canonical")
    document = cast(dict[str, object], parsed)
    if document["schema_version"] != "1.0.0":
        raise DefenderAttestationError("attestation schema is unsupported")
    for name in (
        "bundle_manifest_digest",
        "threshold_digest",
        "component_manifest_digest",
        "rollback_chain_digest",
        "signer_key_id",
        "attestation_digest",
    ):
        _digest(document[name], label=name)
    if type(document["rollback_available"]) is not bool:
        raise DefenderAttestationError("attestation rollback state is invalid")
    predecessor = document["rollback_predecessor_digest"]
    if predecessor is not None:
        _digest(predecessor, label="rollback predecessor digest")
    if document["rollback_available"] != (predecessor is not None):
        raise DefenderAttestationError("attestation rollback state is inconsistent")
    top_ref = document["top_ref"]
    if type(top_ref) is not dict:
        raise DefenderAttestationError("attestation top reference is invalid")
    ref_document = cast(dict[str, object], top_ref)
    if set(ref_document) != {"sha256", "media_type", "size_bytes", "relative_path"}:
        raise DefenderAttestationError("attestation top reference fields are invalid")
    _digest(ref_document["sha256"], label="attestation top digest")
    if (
        ref_document["sha256"] != document["bundle_manifest_digest"]
        or ref_document["media_type"] != _BUNDLE_MEDIA
        or type(ref_document["size_bytes"]) is not int
        or ref_document["size_bytes"] < 1
        or ref_document["size_bytes"] > _MAX_BUNDLE_BYTES
        or ref_document["relative_path"] != f"{ref_document['sha256']}/payload"
    ):
        raise DefenderAttestationError("attestation top reference is inconsistent")
    _parse_utc(document["frozen_at"], label="attestation frozen time")
    unsigned = {key: value for key, value in document.items() if key != "attestation_digest"}
    if hashlib.sha256(canonical_json_bytes(unsigned)).hexdigest() != document["attestation_digest"]:
        raise DefenderAttestationError("attestation digest is inconsistent")
    return document


def _ref_document(ref: ArtifactRef) -> dict[str, object]:
    return {
        "sha256": ref.sha256,
        "media_type": ref.media_type,
        "size_bytes": ref.size_bytes,
        "relative_path": ref.relative_path,
    }


def _canonical_base64(value: object, size: int, *, label: str) -> bytes:
    if type(value) is not str:
        raise DefenderAttestationError(f"{label} must be canonical base64")
    try:
        decoded = base64.b64decode(value, validate=True)
    except (binascii.Error, ValueError) as error:
        raise DefenderAttestationError(f"{label} must be canonical base64") from error
    if len(decoded) != size or base64.b64encode(decoded).decode("ascii") != value:
        raise DefenderAttestationError(f"{label} must be canonical base64")
    return decoded


def _digest(value: object, *, label: str) -> str:
    if type(value) is not str or len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise DefenderAttestationError(f"{label} must be lowercase SHA-256")
    return value


def _parse_utc(value: object, *, label: str) -> datetime:
    if type(value) is not str:
        raise DefenderAttestationError(f"{label} must be an exact UTC timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        validate_utc_timestamp(parsed)
    except (TypeError, ValueError) as error:
        raise DefenderAttestationError(f"{label} must be an exact UTC timestamp") from error
    return parsed


__all__ = [
    "DefenderAttestationError",
    "DefenderBundleVerifier",
    "VerifiedDefenderAttestation",
]
