"""Sealed authenticated inputs admitted to judge-report publication."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import NamedTuple, Never, cast

from pydantic import ValidationError

from apar.defense.bundle import EnvironmentLock
from apar.defense.thresholds import ThresholdReport
from apar.evaluation.defender_attestation import (
    DefenderAttestationError,
    DefenderBundleVerifier,
    VerifiedDefenderAttestation,
)
from apar.evaluation.gates import EvaluatorReplayVerifier, EvaluatorSigningIdentity
from apar.evaluation.replay import ReplayCorpusEvidence
from apar.features.catalog import (
    EXPECTED_FEATURE_NAMES,
    FeatureCatalog,
    audit_feature_catalog,
)
from apar.runs.wire import WireContractError, canonical_json_bytes, strict_json_loads
from apar.storage.artifacts import ArtifactRef, ArtifactStore

CORPUS_ATTESTATION_MEDIA_TYPE = "application/vnd.apar.verified-corpus-attestation+json"
CORPUS_EVIDENCE_MEDIA_TYPE = "application/vnd.apar.replay-corpus-evidence+json"
_BUNDLE_MEDIA_TYPE = "application/vnd.apar.defender-bundle+json"
_MAX_ATTESTATION_BYTES = 64 * 1024
_MAX_CORPUS_EVIDENCE_BYTES = 64 * 1024 * 1024
_TOKEN = object()


class PublicationInputError(ValueError):
    """A corpus or defender failed its pinned authentication boundary."""


def _digest(value: object, *, label: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or value != value.lower()
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise PublicationInputError(f"{label} must be lowercase SHA-256")
    return value


def _ref_document(ref: ArtifactRef) -> dict[str, object]:
    return {
        "media_type": ref.media_type,
        "relative_path": ref.relative_path,
        "sha256": ref.sha256,
        "size_bytes": ref.size_bytes,
    }


def _ref_from_document(value: object, *, media_type: str, size_cap: int) -> ArtifactRef:
    if type(value) is not dict or set(value) != {
        "media_type",
        "relative_path",
        "sha256",
        "size_bytes",
    }:
        raise PublicationInputError("authenticated input reference fields differ")
    document = cast(dict[str, object], value)
    digest = _digest(document["sha256"], label="authenticated input digest")
    size = document["size_bytes"]
    if (
        document["media_type"] != media_type
        or document["relative_path"] != f"{digest}/payload"
        or type(size) is not int
        or not 0 < size <= size_cap
    ):
        raise PublicationInputError("authenticated input reference is invalid")
    return ArtifactRef(digest, media_type, size, f"{digest}/payload")


class _CorpusState(NamedTuple):
    payload: bytes
    evidence_ref: ArtifactRef
    corpus_digest: str
    split_digest: str
    observation_count: int
    truth_count: int
    signer_key_id: str


class VerifiedCorpusAttestation(tuple[_CorpusState]):
    """Immutable evaluator-authenticated handle to restricted frozen-corpus evidence."""

    __slots__ = ()

    def __new__(cls, state: _CorpusState, token: object = None) -> VerifiedCorpusAttestation:
        if token is not _TOKEN or type(state) is not _CorpusState:
            raise PublicationInputError("corpus attestations require pinned verification")
        return tuple.__new__(cls, (state,))

    def __init__(self, state: _CorpusState, token: object = None) -> None:
        del state
        if token is not _TOKEN:
            raise PublicationInputError("corpus attestation cannot be reinitialized")

    def __setattr__(self, name: str, value: object) -> None:
        del name, value
        raise TypeError("verified corpus attestation is immutable")

    def __delattr__(self, name: str) -> None:
        del name
        raise TypeError("verified corpus attestation is immutable")

    def __reduce__(self) -> Never:
        raise TypeError("verified corpus attestations must be reloaded")

    @classmethod
    def model_construct(cls, **values: object) -> Never:
        del values
        raise PublicationInputError("corpus attestation construction is forbidden")

    @property
    def _state(self) -> _CorpusState:
        if type(self) is not VerifiedCorpusAttestation or tuple.__len__(self) != 1:
            raise PublicationInputError("corpus attestation state is invalid")
        state = tuple.__getitem__(self, 0)
        if type(state) is not _CorpusState:
            raise PublicationInputError("corpus attestation state is invalid")
        return state

    def to_json(self) -> bytes:
        return bytes(self._state.payload)

    @property
    def evidence_ref(self) -> ArtifactRef:
        return self._state.evidence_ref

    @property
    def corpus_digest(self) -> str:
        return self._state.corpus_digest

    @property
    def split_digest(self) -> str:
        return self._state.split_digest

    @property
    def observation_count(self) -> int:
        return self._state.observation_count

    @property
    def truth_count(self) -> int:
        return self._state.truth_count

    @property
    def signer_key_id(self) -> str:
        return self._state.signer_key_id


def publish_corpus_attestation(
    evidence: ReplayCorpusEvidence,
    *,
    artifact_store: ArtifactStore,
    signer: EvaluatorSigningIdentity,
) -> tuple[VerifiedCorpusAttestation, ArtifactRef]:
    """Store restricted corpus evidence and publish its signed compact top attestation."""
    if type(artifact_store) is not ArtifactStore or not EvaluatorSigningIdentity.is_exact(signer):
        raise PublicationInputError("corpus attestation dependencies are not exact")
    if type(evidence) is not ReplayCorpusEvidence:
        raise PublicationInputError("corpus evidence must have its exact type")
    verifier = EvaluatorReplayVerifier.from_signer(signer)
    try:
        checked = ReplayCorpusEvidence.model_validate(
            {
                **evidence.model_dump(mode="python", warnings=False, exclude={"corpus"}),
                "corpus": evidence.corpus,
            },
            strict=True,
        )
    except ValidationError as error:
        raise PublicationInputError("corpus evidence failed semantic revalidation") from error
    if not checked.verify(verifier):
        raise PublicationInputError("corpus evidence signature is invalid")
    evidence_payload = canonical_json_bytes(checked.model_dump(mode="json"))
    if not 0 < len(evidence_payload) <= _MAX_CORPUS_EVIDENCE_BYTES:
        raise PublicationInputError("corpus evidence exceeds its resource cap")
    evidence_ref = artifact_store.put_bytes(evidence_payload, CORPUS_EVIDENCE_MEDIA_TYPE)
    fields: dict[str, object] = {
        "schema_version": "1.0.0",
        "evidence_ref": _ref_document(evidence_ref),
        "corpus_digest": checked.corpus_digest,
        "split_digest": checked.split_digest,
        "observation_count": checked.corpus.manifest.observation_count,
        "truth_count": checked.corpus.manifest.truth_count,
        "synthetic_only": True,
        "signer_key_id": signer.key_id,
        "public_key_base64": signer.public_key_base64,
    }
    signature = signer._sign(fields)
    signed = {**fields, "signature_base64": signature}
    document = {
        **signed,
        "attestation_digest": hashlib.sha256(canonical_json_bytes(signed)).hexdigest(),
    }
    payload = canonical_json_bytes(document)
    attestation = _load_corpus_attestation_payload(
        payload,
        artifact_store=artifact_store,
        verifier=verifier,
    )
    top_ref = artifact_store.put_bytes(payload, CORPUS_ATTESTATION_MEDIA_TYPE)
    return attestation, top_ref


def load_corpus_attestation(
    ref: ArtifactRef,
    *,
    artifact_store: ArtifactStore,
    verifier: EvaluatorReplayVerifier,
) -> VerifiedCorpusAttestation:
    if (
        type(ref) is not ArtifactRef
        or ref.media_type != CORPUS_ATTESTATION_MEDIA_TYPE
        or not 0 < ref.size_bytes <= _MAX_ATTESTATION_BYTES
    ):
        raise PublicationInputError("corpus attestation top reference is invalid")
    try:
        payload = artifact_store.read(ref)
    except (TypeError, ValueError) as error:
        raise PublicationInputError("corpus attestation is not loadable") from error
    return _load_corpus_attestation_payload(
        payload, artifact_store=artifact_store, verifier=verifier
    )


def _load_corpus_attestation_payload(
    payload: bytes,
    *,
    artifact_store: ArtifactStore,
    verifier: EvaluatorReplayVerifier,
) -> VerifiedCorpusAttestation:
    if type(verifier) is not EvaluatorReplayVerifier:
        raise PublicationInputError("corpus attestation verifier is not exact")
    if type(payload) is not bytes or not 0 < len(payload) <= _MAX_ATTESTATION_BYTES:
        raise PublicationInputError("corpus attestation payload is invalid")
    try:
        parsed = strict_json_loads(payload)
    except WireContractError as error:
        raise PublicationInputError("corpus attestation JSON is invalid") from error
    expected = {
        "schema_version",
        "evidence_ref",
        "corpus_digest",
        "split_digest",
        "observation_count",
        "truth_count",
        "synthetic_only",
        "signer_key_id",
        "public_key_base64",
        "signature_base64",
        "attestation_digest",
    }
    if (
        type(parsed) is not dict
        or set(parsed) != expected
        or canonical_json_bytes(parsed) != payload
    ):
        raise PublicationInputError("corpus attestation fields are not exact")
    document = cast(dict[str, object], parsed)
    if document["schema_version"] != "1.0.0" or document["synthetic_only"] is not True:
        raise PublicationInputError("corpus attestation schema is unsupported")
    if (
        document["signer_key_id"] != verifier.key_id
        or document["public_key_base64"] != verifier.public_key_base64
    ):
        raise PublicationInputError("corpus attestation authority differs")
    unsigned = {
        key: value
        for key, value in document.items()
        if key not in {"signature_base64", "attestation_digest"}
    }
    signature = document["signature_base64"]
    if type(signature) is not str or not verifier.verify_document(unsigned, signature):
        raise PublicationInputError("corpus attestation signature is invalid")
    signed = {**unsigned, "signature_base64": signature}
    if document["attestation_digest"] != hashlib.sha256(canonical_json_bytes(signed)).hexdigest():
        raise PublicationInputError("corpus attestation digest is inconsistent")
    evidence_ref = _ref_from_document(
        document["evidence_ref"],
        media_type=CORPUS_EVIDENCE_MEDIA_TYPE,
        size_cap=_MAX_CORPUS_EVIDENCE_BYTES,
    )
    try:
        evidence_payload = artifact_store.read(evidence_ref)
        evidence_document = strict_json_loads(evidence_payload)
        evidence = ReplayCorpusEvidence.model_validate(evidence_document)
    except (TypeError, ValueError, ValidationError, WireContractError) as error:
        raise PublicationInputError("restricted corpus evidence is invalid") from error
    if (
        canonical_json_bytes(evidence.model_dump(mode="json")) != evidence_payload
        or not evidence.verify(verifier)
        or evidence.corpus_digest != document["corpus_digest"]
        or evidence.split_digest != document["split_digest"]
        or evidence.corpus.manifest.observation_count != document["observation_count"]
        or evidence.corpus.manifest.truth_count != document["truth_count"]
    ):
        raise PublicationInputError("corpus attestation evidence binding differs")
    state = _CorpusState(
        payload=payload,
        evidence_ref=evidence_ref,
        corpus_digest=document["corpus_digest"],
        split_digest=document["split_digest"],
        observation_count=document["observation_count"],
        truth_count=document["truth_count"],
        signer_key_id=document["signer_key_id"],
    )
    return VerifiedCorpusAttestation(state, _TOKEN)


@dataclass(frozen=True, slots=True)
class _VerifiedInputState:
    corpus: VerifiedCorpusAttestation
    defender: VerifiedDefenderAttestation
    catalog: FeatureCatalog
    thresholds: ThresholdReport
    environment: EnvironmentLock


class VerifiedEvaluationInputs(tuple[_VerifiedInputState]):
    """Sealed exact corpus/defender capability passed to the isolated executor."""

    __slots__ = ()

    def __new__(cls, state: _VerifiedInputState, token: object = None) -> VerifiedEvaluationInputs:
        if token is not _TOKEN or type(state) is not _VerifiedInputState:
            raise PublicationInputError("evaluation inputs require pinned verification")
        return tuple.__new__(cls, (state,))

    def __init__(self, state: _VerifiedInputState, token: object = None) -> None:
        del state
        if token is not _TOKEN:
            raise PublicationInputError("verified evaluation inputs cannot be reinitialized")

    def __setattr__(self, name: str, value: object) -> None:
        del name, value
        raise TypeError("verified evaluation inputs are immutable")

    def __delattr__(self, name: str) -> None:
        del name
        raise TypeError("verified evaluation inputs are immutable")

    def __reduce__(self) -> Never:
        raise TypeError("verified evaluation inputs cannot be serialized")

    @property
    def _state(self) -> _VerifiedInputState:
        if type(self) is not VerifiedEvaluationInputs or tuple.__len__(self) != 1:
            raise PublicationInputError("verified evaluation input state is invalid")
        state = tuple.__getitem__(self, 0)
        if type(state) is not _VerifiedInputState:
            raise PublicationInputError("verified evaluation input state is invalid")
        return state

    @property
    def corpus(self) -> VerifiedCorpusAttestation:
        return self._state.corpus

    @property
    def defender(self) -> VerifiedDefenderAttestation:
        return self._state.defender

    @property
    def catalog(self) -> FeatureCatalog:
        return self._state.catalog

    @property
    def thresholds(self) -> ThresholdReport:
        return self._state.thresholds

    @property
    def environment(self) -> EnvironmentLock:
        return self._state.environment


def verify_evaluation_inputs(
    *,
    corpus_ref: ArtifactRef,
    defender_ref: ArtifactRef,
    artifact_store: ArtifactStore,
    evaluator_verifier: EvaluatorReplayVerifier,
    defender_verifier: DefenderBundleVerifier,
) -> VerifiedEvaluationInputs:
    """Authenticate both top refs and derive typed public facts from signed components."""
    if type(artifact_store) is not ArtifactStore:
        raise PublicationInputError("input verification requires the exact artifact store")
    corpus = load_corpus_attestation(
        corpus_ref, artifact_store=artifact_store, verifier=evaluator_verifier
    )
    try:
        defender = defender_verifier.attest(defender_ref)
        manifest_payload = artifact_store.read(defender.top_ref)
        manifest = strict_json_loads(manifest_payload)
    except (DefenderAttestationError, TypeError, ValueError, WireContractError) as error:
        raise PublicationInputError("defender bundle authentication failed") from error
    if type(manifest) is not dict or canonical_json_bytes(manifest) != manifest_payload:
        raise PublicationInputError("authenticated defender manifest is invalid")
    document = cast(dict[str, object], manifest)
    if corpus.split_digest != document.get("split_manifest_digest"):
        raise PublicationInputError("corpus and defender split lineage differ")
    components = document.get("components")
    if type(components) is not list:
        raise PublicationInputError("defender components are unavailable")
    refs: dict[str, ArtifactRef] = {}
    for item in components:
        if type(item) is not dict:
            raise PublicationInputError("defender component descriptor is invalid")
        raw = cast(dict[str, object], item)
        name = raw.get("name")
        if type(name) is not str:
            raise PublicationInputError("defender component name is invalid")
        refs[name] = _ref_from_document(
            {
                "media_type": raw.get("media_type"),
                "relative_path": f"{raw.get('sha256')}/payload",
                "sha256": raw.get("sha256"),
                "size_bytes": raw.get("size_bytes"),
            },
            media_type=cast(str, raw.get("media_type")),
            size_cap=128 * 1024 * 1024,
        )
    try:
        catalog_payload = artifact_store.read(refs["catalog"])
        threshold_payload = artifact_store.read(refs["threshold"])
        environment_payload = artifact_store.read(refs["environment"])
        catalog = FeatureCatalog.model_validate(strict_json_loads(catalog_payload))
        audit_feature_catalog(catalog)
        thresholds = ThresholdReport.from_json(threshold_payload)
        environment = EnvironmentLock.model_validate(strict_json_loads(environment_payload))
    except (
        KeyError,
        TypeError,
        ValueError,
        ValidationError,
        WireContractError,
    ) as error:
        raise PublicationInputError("authenticated defender public facts are invalid") from error
    if catalog.names != EXPECTED_FEATURE_NAMES or not thresholds.feasible:
        raise PublicationInputError("defender public facts are not competition-ready")
    return VerifiedEvaluationInputs(
        _VerifiedInputState(corpus, defender, catalog, thresholds, environment), _TOKEN
    )


__all__ = [
    "CORPUS_ATTESTATION_MEDIA_TYPE",
    "CORPUS_EVIDENCE_MEDIA_TYPE",
    "PublicationInputError",
    "VerifiedCorpusAttestation",
    "VerifiedEvaluationInputs",
    "load_corpus_attestation",
    "publish_corpus_attestation",
    "verify_evaluation_inputs",
]
