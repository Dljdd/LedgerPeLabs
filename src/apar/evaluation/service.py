"""Artifact-backed, idempotent defense evaluation publication service."""

from __future__ import annotations

import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass

from apar.evaluation.gates import EvaluatorReplayVerifier
from apar.evaluation.reporting import (
    PUBLIC_ARTIFACT_MEDIA_TYPES,
    SCORECARD_ARTIFACT_NAME,
    DefenseScorecard,
    EvaluationArtifactBundle,
    PublicArtifactReference,
    ReportingContractError,
    ScorecardPublicationRequest,
    load_evaluation_bundle,
    publish_scorecard,
)
from apar.runs.runner import RunSigningIdentity
from apar.runs.wire import WireContractError, canonical_json_bytes, strict_json_loads
from apar.storage.artifacts import ArtifactRef, ArtifactStore

CORPUS_MEDIA_TYPE = "application/vnd.apar.frozen-corpus+json"
DEFENDER_MEDIA_TYPE = "application/vnd.apar.defender-bundle+json"
MAX_EVALUATIONS = 10_000
MAX_EXECUTION_SECONDS = 900.0
MAX_CORPUS_BYTES = 512 * 1024 * 1024
MAX_DEFENDER_BYTES = 256 * 1024 * 1024


class DefenseServiceError(ValueError):
    """Base class for stable service error normalization."""


class DefenseResourceNotFound(DefenseServiceError):
    """An input or published evaluation cannot be resolved."""


class DefenseArtifactInvalid(DefenseServiceError):
    """A resolved immutable artifact fails semantic verification."""


class DefenseExecutionConflict(DefenseServiceError):
    """Evaluation cannot execute or pass its publication gates."""


class EvaluationExecutor(ABC):
    """Injected Task14 orchestration boundary returning exact frozen evidence."""

    @abstractmethod
    def execute(
        self,
        *,
        corpus_ref: ArtifactRef,
        defender_ref: ArtifactRef,
        timeout_seconds: float,
    ) -> ScorecardPublicationRequest:
        """Produce independently verified evaluation evidence for exact inputs."""


class UnavailableEvaluationExecutor(EvaluationExecutor):
    """Fail-closed placeholder until Task14 supplies real fixture orchestration."""

    def execute(
        self,
        *,
        corpus_ref: ArtifactRef,
        defender_ref: ArtifactRef,
        timeout_seconds: float,
    ) -> ScorecardPublicationRequest:
        del corpus_ref, defender_ref, timeout_seconds
        raise DefenseExecutionConflict("defense evaluation executor is unavailable")


@dataclass(frozen=True, slots=True)
class PublishedArtifact:
    """One freshly verified public response payload."""

    reference: PublicArtifactReference
    payload: bytes


class DefenseEvaluationService:
    """Serialize exact input pairs and publish only complete signed bundles."""

    __slots__ = (
        "_artifact_store",
        "_by_evaluation_id",
        "_by_inputs",
        "_evaluator_verifier",
        "_executor",
        "_hidden_proof_verifier",
        "_lock",
        "_signer",
        "_signer_key_id",
        "_signer_public_key",
    )

    def __init__(
        self,
        *,
        artifact_store: ArtifactStore,
        signer: RunSigningIdentity,
        evaluator_verifier: EvaluatorReplayVerifier,
        hidden_proof_verifier: EvaluatorReplayVerifier,
        executor: EvaluationExecutor,
    ) -> None:
        if type(artifact_store) is not ArtifactStore:
            raise TypeError("defense service requires an exact ArtifactStore")
        if type(signer) is not RunSigningIdentity:
            raise TypeError("defense service requires an exact RunSigningIdentity")
        if type(evaluator_verifier) is not EvaluatorReplayVerifier or type(
            hidden_proof_verifier
        ) is not EvaluatorReplayVerifier:
            raise TypeError("defense service requires exact pinned evaluator verifiers")
        if not isinstance(executor, EvaluationExecutor):
            raise TypeError("defense service requires an EvaluationExecutor")
        self._artifact_store = artifact_store
        self._signer = signer
        self._signer_key_id = signer.key_id
        self._signer_public_key = signer.public_key_base64
        self._evaluator_verifier = evaluator_verifier
        self._hidden_proof_verifier = hidden_proof_verifier
        self._executor = executor
        self._lock = threading.RLock()
        self._by_inputs: dict[tuple[str, str], ArtifactRef] = {}
        self._by_evaluation_id: dict[str, ArtifactRef] = {}

    def create(
        self, *, corpus_artifact_digest: str, defender_artifact_digest: str
    ) -> DefenseScorecard:
        """Resolve exact inputs, execute once, and expose only an atomic publication."""
        _validate_digest(corpus_artifact_digest)
        _validate_digest(defender_artifact_digest)
        key = (corpus_artifact_digest, defender_artifact_digest)
        with self._lock:
            self._verify_signer_identity()
            existing = self._by_inputs.get(key)
            if existing is not None:
                return self._load(existing).scorecard(
                    artifact_store=self._artifact_store, signer=self._signer
                )
            if len(self._by_inputs) >= MAX_EVALUATIONS:
                raise DefenseExecutionConflict("evaluation capacity is exhausted")
            corpus_ref = self._resolve_input(
                corpus_artifact_digest, expected_media_type=CORPUS_MEDIA_TYPE
            )
            defender_ref = self._resolve_input(
                defender_artifact_digest, expected_media_type=DEFENDER_MEDIA_TYPE
            )
            try:
                request = self._executor.execute(
                    corpus_ref=corpus_ref,
                    defender_ref=defender_ref,
                    timeout_seconds=MAX_EXECUTION_SECONDS,
                )
            except DefenseServiceError:
                raise
            except (MemoryError, TimeoutError) as error:
                raise DefenseExecutionConflict("defense evaluation did not complete") from error
            except Exception as error:
                raise DefenseExecutionConflict("defense evaluation was rejected") from error
            if type(request) is not ScorecardPublicationRequest:
                raise DefenseArtifactInvalid("evaluation executor returned invalid evidence")
            if (
                request.corpus_artifact_digest != corpus_ref.sha256
                or request.defender_artifact_digest != defender_ref.sha256
            ):
                raise DefenseArtifactInvalid("evaluation evidence input lineage differs")
            try:
                scorecard, bundle = publish_scorecard(
                    request,
                    artifact_store=self._artifact_store,
                    signer=self._signer,
                    evaluator_verifier=self._evaluator_verifier,
                    hidden_proof_verifier=self._hidden_proof_verifier,
                )
            except ReportingContractError as error:
                raise DefenseExecutionConflict(
                    "defense publication gates rejected evidence"
                ) from error
            bundle_ref = bundle.bundle_ref()
            loaded = self._load(bundle_ref)
            if loaded != bundle:
                raise DefenseArtifactInvalid("published evaluation did not reload exactly")
            self._by_inputs[key] = bundle_ref
            self._by_evaluation_id[scorecard.evaluation_id] = bundle_ref
            return scorecard

    def get(self, evaluation_id: str) -> DefenseScorecard:
        """Re-read every signed artifact before returning a public scorecard."""
        _validate_digest(evaluation_id)
        with self._lock:
            self._verify_signer_identity()
            ref = self._by_evaluation_id.get(evaluation_id)
            if ref is None:
                raise DefenseResourceNotFound("defense evaluation not found")
            bundle = self._load(ref)
            scorecard = bundle.scorecard(
                artifact_store=self._artifact_store, signer=self._signer
            )
            if scorecard.evaluation_id != evaluation_id:
                raise DefenseArtifactInvalid("evaluation identity differs")
            return scorecard

    def get_artifact(self, evaluation_id: str, name: str) -> PublishedArtifact:
        """Return one exact allowlisted artifact after full bundle revalidation."""
        _validate_digest(evaluation_id)
        if type(name) is not str or name not in {
            *PUBLIC_ARTIFACT_MEDIA_TYPES,
            SCORECARD_ARTIFACT_NAME,
        }:
            raise DefenseResourceNotFound("public artifact not found")
        with self._lock:
            self._verify_signer_identity()
            ref = self._by_evaluation_id.get(evaluation_id)
            if ref is None:
                raise DefenseResourceNotFound("defense evaluation not found")
            bundle = self._load(ref)
            reference = bundle.public_artifacts[name]
            try:
                payload = self._artifact_store.read(reference.as_artifact_ref())
            except (TypeError, ValueError) as error:
                raise DefenseArtifactInvalid(
                    "public artifact failed integrity validation"
                ) from error
            return PublishedArtifact(reference=reference, payload=payload)

    def _resolve_input(self, digest: str, *, expected_media_type: str) -> ArtifactRef:
        try:
            ref = self._artifact_store.resolve(digest)
        except ValueError as error:
            raise DefenseResourceNotFound("evaluation input artifact not found") from error
        if ref.media_type != expected_media_type:
            raise DefenseArtifactInvalid("evaluation input artifact type is invalid")
        size_cap = (
            MAX_CORPUS_BYTES
            if expected_media_type == CORPUS_MEDIA_TYPE
            else MAX_DEFENDER_BYTES
        )
        if not 0 < ref.size_bytes <= size_cap:
            raise DefenseArtifactInvalid("evaluation input artifact exceeds its cap")
        try:
            payload = self._artifact_store.read(ref)
            document = strict_json_loads(payload)
            if type(document) is not dict or canonical_json_bytes(document) != payload:
                raise DefenseArtifactInvalid("evaluation input artifact is not canonical")
            if document.get("schema_version") != "1.0.0":
                raise DefenseArtifactInvalid("evaluation input schema is unsupported")
            if (
                expected_media_type == CORPUS_MEDIA_TYPE
                and document.get("synthetic_only") is not True
            ):
                raise DefenseArtifactInvalid("evaluation corpus is not synthetic-only")
        except DefenseArtifactInvalid:
            raise
        except (TypeError, ValueError, WireContractError) as error:
            raise DefenseArtifactInvalid("evaluation input artifact is invalid") from error
        return ref

    def _load(self, ref: ArtifactRef) -> EvaluationArtifactBundle:
        try:
            return load_evaluation_bundle(
                ref, artifact_store=self._artifact_store, signer=self._signer
            )
        except ReportingContractError as error:
            raise DefenseArtifactInvalid("published defense artifacts are invalid") from error

    def _verify_signer_identity(self) -> None:
        if (
            self._signer.key_id != self._signer_key_id
            or self._signer.public_key_base64 != self._signer_public_key
        ):
            raise DefenseArtifactInvalid("pinned publication authority changed")


def _validate_digest(value: object) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or value != value.lower()
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise DefenseResourceNotFound("resource not found")
    return value


__all__ = [
    "CORPUS_MEDIA_TYPE",
    "DEFENDER_MEDIA_TYPE",
    "DefenseArtifactInvalid",
    "DefenseEvaluationService",
    "DefenseExecutionConflict",
    "DefenseResourceNotFound",
    "DefenseServiceError",
    "EvaluationExecutor",
    "MAX_CORPUS_BYTES",
    "MAX_DEFENDER_BYTES",
    "MAX_EXECUTION_SECONDS",
    "PublishedArtifact",
    "UnavailableEvaluationExecutor",
]
