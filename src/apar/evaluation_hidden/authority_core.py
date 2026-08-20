"""Opaque authority-owned lifecycle for restricted defense evaluation."""

from __future__ import annotations

import base64
import binascii
import hashlib
import secrets
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Literal, NamedTuple, Never, Protocol, cast

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
from pydantic import ValidationError, field_validator, model_validator

from apar.contracts._validation import ExternalContract, validate_utc_timestamp
from apar.evaluation.defender_attestation import (
    DefenderBundleVerifier,
    VerifiedDefenderAttestation,
)
from apar.evaluation.gates import (
    EvaluatorReplayVerifier,
    EvaluatorSigningIdentity,
    HiddenPublicProof,
    ReplayResult,
    VerifiedReplayBatch,
)
from apar.evaluation.hidden_source import HiddenSourceWorkerBinding
from apar.evaluation_hidden.worker_client import (
    EvaluatorWorkerClient,
    EvaluatorWorkerManifest,
    HiddenWorkerError,
)
from apar.runs.wire import WireContractError, canonical_json_bytes, strict_json_loads
from apar.storage.artifacts import ArtifactRef, ArtifactStore

HIDDEN_CONTEXT_MEDIA_TYPE = "application/vnd.apar.hidden-evaluation-context+json"
HIDDEN_FREEZE_RECEIPT_MEDIA_TYPE = (
    "application/vnd.apar.hidden-decision-freeze-receipt+json"
)
HIDDEN_EVALUATION_RECEIPT_MEDIA_TYPE = (
    "application/vnd.apar.hidden-evaluation-receipt+json"
)
_MAX_HIDDEN_BYTES = 64 * 1024 * 1024
_MAX_RECEIPT_BYTES = 256 * 1024
_MAX_RELEASES = 64
_OBJECT_TOKEN = object()


class HiddenBoundaryError(ValueError):
    """A hidden-evaluation caller violated the sealed lifecycle."""


class HiddenReplayOutcome(NamedTuple):
    """Public aggregate-only result of one isolated hidden evaluation."""

    batch: VerifiedReplayBatch
    public_proof: HiddenPublicProof

    @property
    def results(self) -> tuple[ReplayResult, ...]:
        return self.batch.results


class HiddenDecisionBinding(ExternalContract):
    """Truth-blind digest of one completely materialized replay arm."""

    schema_version: Literal["1.0.0"] = "1.0.0"
    arm: Literal["rules_only", "gbdt_only", "layered_hybrid"]
    decision_event_ids_digest: str
    decision_artifact_digest: str
    action_digest: str
    score_digest: str
    common_integrity_digest: str
    threshold_report_digest: str
    threshold_set_digest: str
    case_callback_digest: str

    @field_validator(
        "decision_event_ids_digest",
        "decision_artifact_digest",
        "action_digest",
        "score_digest",
        "common_integrity_digest",
        "threshold_report_digest",
        "threshold_set_digest",
        "case_callback_digest",
    )
    @classmethod
    def digests_are_sha256(cls, value: str) -> str:
        _validate_digest(value)
        return value


class HiddenArmEvidenceBinding(ExternalContract):
    """One exact restricted Task 11 evaluator derivation binding."""

    schema_version: Literal["1.0.0"] = "1.0.0"
    arm: Literal["rules_only", "gbdt_only", "layered_hybrid"]
    evaluator_input_digest: str
    derivation_evidence_digest: str
    metric_report_digest: str

    @field_validator(
        "evaluator_input_digest", "derivation_evidence_digest", "metric_report_digest"
    )
    @classmethod
    def evidence_digests_are_sha256(cls, value: str) -> str:
        _validate_digest(value)
        return value


class HiddenDecisionFreezeReceipt(ExternalContract):
    """Signed persistent proof that all truth-blind decisions froze first."""

    schema_version: Literal["1.0.0"] = "1.0.0"
    capability_digest: str
    defender_attestation_digest: str
    defender_top_ref_digest: str
    bundle_manifest_digest: str
    restricted_ref_digest: str
    invocation_digest: str
    release_sequence: int
    released_at: datetime
    frozen_at: datetime
    decisions: tuple[HiddenDecisionBinding, ...]
    signer_key_id: str
    public_key_base64: str
    signature_base64: str
    receipt_digest: str

    @field_validator(
        "capability_digest",
        "defender_attestation_digest",
        "defender_top_ref_digest",
        "bundle_manifest_digest",
        "restricted_ref_digest",
        "invocation_digest",
        "signer_key_id",
        "receipt_digest",
    )
    @classmethod
    def digests_are_sha256(cls, value: str) -> str:
        _validate_digest(value)
        return value

    @field_validator("release_sequence", mode="before")
    @classmethod
    def sequence_is_bounded(cls, value: object) -> object:
        if type(value) is not int or not 1 <= value <= _MAX_RELEASES:
            raise ValueError("hidden release sequence is invalid")
        return value

    @field_validator("released_at", "frozen_at")
    @classmethod
    def timestamps_are_utc(cls, value: datetime) -> datetime:
        if type(value) is not datetime:
            raise ValueError("hidden freeze timestamps must be exact datetimes")
        return validate_utc_timestamp(value)

    @field_validator("decisions", mode="before")
    @classmethod
    def decisions_are_tuple(cls, value: object) -> object:
        if type(value) is not tuple:
            raise ValueError("hidden decision bindings must be an exact tuple")
        return value

    @model_validator(mode="after")
    def receipt_is_closed_and_signed(self) -> HiddenDecisionFreezeReceipt:
        if self.frozen_at < self.released_at:
            raise ValueError("hidden decision freeze cannot precede release authorization")
        if tuple(item.arm for item in self.decisions) != _ARM_ORDER:
            raise ValueError("hidden decision freeze must bind all arms in order")
        _validate_signed_receipt(self)
        return self

    def unsigned_document(self) -> dict[str, object]:
        return self.model_dump(
            mode="json",
            exclude={"signature_base64", "receipt_digest"},
        )

    def to_json(self) -> bytes:
        return _receipt_to_json(self, HiddenDecisionFreezeReceipt)

    @classmethod
    def from_json(cls, payload: bytes) -> HiddenDecisionFreezeReceipt:
        return cast(HiddenDecisionFreezeReceipt, _receipt_from_json(cls, payload))


class HiddenEvaluationReceipt(ExternalContract):
    """Signed aggregate-only proof of evaluator use after the decision freeze."""

    schema_version: Literal["3.1.0"] = "3.1.0"
    capability_digest: str
    defender_attestation_digest: str
    defender_top_ref_digest: str
    bundle_manifest_digest: str
    restricted_ref_digest: str
    restricted_artifact_digest: str
    canonical_content_digest: str
    evaluator_context_digest: str
    restricted_cohort_mapping_digest: str
    descriptor_lineage_digest: str
    decision_freeze_receipt_digest: str
    decision_freeze_ref_digest: str
    decision_bindings_digest: str
    replay_batch_digest: str
    worker_manifest_digest: str
    public_proof_id: str
    release_sequence: int
    released_at: datetime
    sealed_at: datetime
    arm_evidence: tuple[HiddenArmEvidenceBinding, ...]
    signer_key_id: str
    public_key_base64: str
    signature_base64: str
    receipt_digest: str

    @field_validator(
        "capability_digest",
        "defender_attestation_digest",
        "defender_top_ref_digest",
        "bundle_manifest_digest",
        "restricted_ref_digest",
        "restricted_artifact_digest",
        "canonical_content_digest",
        "evaluator_context_digest",
        "restricted_cohort_mapping_digest",
        "descriptor_lineage_digest",
        "decision_freeze_receipt_digest",
        "decision_freeze_ref_digest",
        "decision_bindings_digest",
        "replay_batch_digest",
        "worker_manifest_digest",
        "signer_key_id",
        "receipt_digest",
    )
    @classmethod
    def digests_are_sha256(cls, value: str) -> str:
        _validate_digest(value)
        return value

    @field_validator("release_sequence", mode="before")
    @classmethod
    def sequence_is_bounded(cls, value: object) -> object:
        if type(value) is not int or not 1 <= value <= _MAX_RELEASES:
            raise ValueError("hidden release sequence is invalid")
        return value

    @field_validator("released_at", "sealed_at")
    @classmethod
    def timestamps_are_utc(cls, value: datetime) -> datetime:
        if type(value) is not datetime:
            raise ValueError("hidden receipt timestamps must be exact datetimes")
        return validate_utc_timestamp(value)

    @field_validator("arm_evidence", mode="before")
    @classmethod
    def evidence_is_tuple(cls, value: object) -> object:
        if type(value) is not tuple:
            raise ValueError("hidden arm evidence must be an exact tuple")
        return value

    @field_validator("public_proof_id")
    @classmethod
    def proof_id_is_opaque(cls, value: str) -> str:
        if (
            type(value) is not str
            or not value.startswith("hpf_")
            or len(value) != 36
            or any(character not in "0123456789abcdef" for character in value[4:])
        ):
            raise ValueError("hidden receipt public proof ID is invalid")
        return value

    @model_validator(mode="after")
    def receipt_is_closed_and_signed(self) -> HiddenEvaluationReceipt:
        if self.sealed_at < self.released_at:
            raise ValueError("hidden receipt cannot precede release authorization")
        if tuple(item.arm for item in self.arm_evidence) != _ARM_ORDER:
            raise ValueError("hidden receipt must bind all arms in order")
        _validate_signed_receipt(self)
        return self

    def unsigned_document(self) -> dict[str, object]:
        return self.model_dump(
            mode="json",
            exclude={"signature_base64", "receipt_digest"},
        )

    def to_json(self) -> bytes:
        return _receipt_to_json(self, HiddenEvaluationReceipt)

    @classmethod
    def from_json(cls, payload: bytes) -> HiddenEvaluationReceipt:
        return cast(HiddenEvaluationReceipt, _receipt_from_json(cls, payload))


class _SealedType(type):
    def __setattr__(cls, name: str, value: object) -> None:
        del cls, name, value
        raise TypeError("hidden authority types are sealed")

    def __delattr__(cls, name: str) -> None:
        del cls, name
        raise TypeError("hidden authority types are sealed")


class HiddenEvaluationCapability(metaclass=_SealedType):
    """Base marker for a unique, exact-identity authority capability."""

    __slots__ = ("__weakref__",)

    def __new__(cls, token: object = None) -> HiddenEvaluationCapability:
        del token
        raise HiddenBoundaryError("hidden capability cannot be constructed externally")

    def __setattr__(self, name: str, value: object) -> None:
        del name, value
        raise TypeError("hidden capability is immutable")

    def __delattr__(self, name: str) -> None:
        del name
        raise TypeError("hidden capability is immutable")

    def __copy__(self) -> Never:
        raise TypeError("hidden capability identity cannot be copied")

    def __deepcopy__(self, memo: object) -> Never:
        del memo
        raise TypeError("hidden capability identity cannot be copied")

    def __reduce__(self) -> Never:
        raise TypeError("hidden capability identity cannot be serialized")


class HiddenEvaluationAuthority(metaclass=_SealedType):
    """Base marker for one closure-owned hidden evaluation authority."""

    __slots__ = ("__weakref__",)

    def __new__(
        cls,
        verifier: DefenderBundleVerifier,
        restricted_store: ArtifactStore,
        evaluator_signer: EvaluatorSigningIdentity,
        hidden_source_binding: HiddenSourceWorkerBinding | None = None,
    ) -> HiddenEvaluationAuthority:
        if cls is not HiddenEvaluationAuthority:
            raise HiddenBoundaryError("hidden authority cannot be constructed externally")
        return _new_authority(
            verifier, restricted_store, evaluator_signer, hidden_source_binding
        )

    def __setattr__(self, name: str, value: object) -> None:
        del name, value
        raise TypeError("hidden evaluation authority is immutable")

    def __delattr__(self, name: str) -> None:
        del name
        raise TypeError("hidden evaluation authority is immutable")

    def __copy__(self) -> Never:
        raise TypeError("hidden authority cannot be copied")

    def __deepcopy__(self, memo: object) -> Never:
        del memo
        raise TypeError("hidden authority cannot be copied")

    def __reduce__(self) -> Never:
        raise TypeError("hidden authority cannot be serialized")


class _HiddenProductView(Protocol):
    evaluator_context_digest: str
    restricted_cohort_mapping_digest: str
    evaluation_lineage_digest: str


_HiddenSourceSnapshot = tuple[
    tuple[str, str, int, str],
    str,
    str,
    tuple[str, ...],
    tuple[str, ...],
    tuple[str, ...],
    tuple[str, ...],
]


def _snapshot_hidden_source_binding(
    binding: HiddenSourceWorkerBinding | None,
) -> _HiddenSourceSnapshot | None:
    if binding is None:
        return None
    if type(binding) is not HiddenSourceWorkerBinding:
        raise HiddenBoundaryError("hidden authority source binding is invalid")
    source_ref = binding.receipt_ref
    return (
        (
            source_ref.sha256,
            source_ref.media_type,
            source_ref.size_bytes,
            source_ref.relative_path,
        ),
        binding.source_signer_key_id,
        binding.source_public_key_base64,
        tuple(binding.development_run_ids),
        tuple(binding.development_event_ids),
        tuple(binding.development_payment_ids),
        tuple(binding.development_campaign_ids),
    )


def _new_authority(
    verifier: DefenderBundleVerifier,
    restricted_store: ArtifactStore,
    evaluator_signer: EvaluatorSigningIdentity,
    hidden_source_binding: HiddenSourceWorkerBinding | None,
) -> HiddenEvaluationAuthority:
    if type(verifier) is not DefenderBundleVerifier:
        raise HiddenBoundaryError("hidden authority requires the exact neutral verifier")
    if type(restricted_store) is not ArtifactStore:
        raise HiddenBoundaryError("hidden authority requires an exact restricted store")
    if not EvaluatorSigningIdentity.is_exact(evaluator_signer):
        raise HiddenBoundaryError("hidden authority requires an exact evaluator signer")
    if (
        hidden_source_binding is not None
        and type(hidden_source_binding) is not HiddenSourceWorkerBinding
    ):
        raise HiddenBoundaryError("hidden authority source binding is invalid")

    trusted_verifier = verifier
    trusted_store = restricted_store
    trusted_signer = evaluator_signer
    trusted_hidden_source = hidden_source_binding
    hidden_source_snapshot = _snapshot_hidden_source_binding(hidden_source_binding)
    evaluator_verifier = EvaluatorReplayVerifier.from_signer(trusted_signer)
    worker_manifest = EvaluatorWorkerManifest.create(trusted_signer)
    worker_client = EvaluatorWorkerClient(worker_manifest, evaluator_verifier)
    public_key_base64 = trusted_signer.public_key_base64
    signer_key_id = trusted_signer.key_id
    initialized = False
    authority_instance: HiddenEvaluationAuthority | None = None
    active_capability: HiddenEvaluationCapability | None = None
    active_attestation: VerifiedDefenderAttestation | None = None
    capability_digest: str | None = None
    capability_issued_at: datetime | None = None
    release_sequence = 0
    lifecycle = "READY"
    issuance_state = "UNISSUED"
    lifecycle_lock = threading.Lock()

    class _BoundCapability(HiddenEvaluationCapability):
        __slots__ = ()

        def __new__(cls, token: object = None) -> _BoundCapability:
            if token is not _OBJECT_TOKEN:
                raise HiddenBoundaryError("hidden capability cannot be constructed externally")
            return object.__new__(cls)

        def __init__(self, token: object = None) -> None:
            if token is not _OBJECT_TOKEN:
                raise HiddenBoundaryError("hidden capability cannot be reinitialized")

        @property
        def bundle_manifest_digest(self) -> str:
            assert active_attestation is not None
            return active_attestation.bundle_manifest_digest

        @property
        def bundle_id(self) -> str:
            assert active_attestation is not None
            return active_attestation.bundle_id

        @property
        def issued_at(self) -> datetime:
            assert capability_issued_at is not None
            return capability_issued_at

        def __repr__(self) -> str:
            return (
                "HiddenEvaluationCapability("
                f"bundle_manifest_digest={self.bundle_manifest_digest!r}, "
                f"bundle_id={self.bundle_id!r}, issued_at={self.issued_at.isoformat()!r})"
            )

        __str__ = __repr__

    class _BoundAuthority(HiddenEvaluationAuthority):
        __slots__ = ()

        def __new__(
            cls,
            verifier: DefenderBundleVerifier,
            restricted_store: ArtifactStore,
            evaluator_signer: EvaluatorSigningIdentity,
            hidden_source_binding: HiddenSourceWorkerBinding | None = None,
        ) -> _BoundAuthority:
            del cls, verifier, restricted_store, evaluator_signer, hidden_source_binding
            raise HiddenBoundaryError("hidden authority cannot be constructed externally")

        def __init__(
            self,
            verifier: DefenderBundleVerifier,
            restricted_store: ArtifactStore,
            evaluator_signer: EvaluatorSigningIdentity,
            hidden_source_binding: HiddenSourceWorkerBinding | None = None,
        ) -> None:
            nonlocal initialized
            if (
                initialized
                or self is not authority_instance
                or verifier is not trusted_verifier
                or restricted_store is not trusted_store
                or evaluator_signer is not trusted_signer
                or hidden_source_binding is not trusted_hidden_source
            ):
                raise HiddenBoundaryError("hidden authority is already initialized")
            initialized = True

        def freeze_and_issue(
            self,
            attestation: VerifiedDefenderAttestation,
            *,
            issued_at: datetime,
        ) -> HiddenEvaluationCapability:
            nonlocal active_capability, active_attestation
            nonlocal capability_digest, capability_issued_at
            nonlocal issuance_state
            if self is not authority_instance:
                raise HiddenBoundaryError("hidden authority instance identity is invalid")
            with lifecycle_lock:
                if issuance_state != "UNISSUED" or active_capability is not None:
                    raise HiddenBoundaryError("hidden authority is already frozen")
                issuance_state = "ISSUING"
            try:
                if (
                    type(attestation) is not VerifiedDefenderAttestation
                    or not trusted_verifier.verify(attestation)
                ):
                    raise HiddenBoundaryError(
                        "hidden release requires an exact verified signed defender attestation"
                    )
                checked_at = _utc(issued_at, label="hidden capability issue time")
                if attestation.frozen_at > checked_at:
                    raise HiddenBoundaryError(
                        "defender freeze follows hidden capability issue"
                    )
                next_digest = _digest_document(
                    {
                        "schema_version": "2.0.0",
                        "attestation_digest": attestation.attestation_digest,
                        "bundle_manifest_digest": attestation.bundle_manifest_digest,
                        "bundle_id": attestation.bundle_id,
                        "issued_at": _time_wire(checked_at),
                        "nonce": secrets.token_hex(32),
                    }
                )
                next_capability = _BoundCapability(_OBJECT_TOKEN)
            except BaseException:
                with lifecycle_lock:
                    if issuance_state == "ISSUING":
                        issuance_state = "UNISSUED"
                raise
            with lifecycle_lock:
                if issuance_state != "ISSUING" or active_capability is not None:
                    raise HiddenBoundaryError("hidden capability issuance was lost")
                active_attestation = attestation
                capability_issued_at = checked_at
                capability_digest = next_digest
                active_capability = next_capability
                issuance_state = "FROZEN"
                return active_capability

        def evaluate_hidden_replay(
            self,
            invocation: object,
            *,
            capability: HiddenEvaluationCapability,
            restricted_ref: ArtifactRef,
            released_at: datetime,
            sealed_at: datetime,
        ) -> object:
            nonlocal release_sequence, lifecycle
            if self is not authority_instance:
                raise HiddenBoundaryError("hidden authority instance identity is invalid")
            reserved_here = False
            try:
                with lifecycle_lock:
                    if lifecycle != "READY":
                        raise HiddenBoundaryError(
                            "hidden replay capability is already reserved or consumed"
                        )
                    lifecycle = "RESERVED"
                    reserved_here = True
                    sequence = release_sequence + 1
                if active_capability is None or active_attestation is None:
                    raise HiddenBoundaryError("restricted refs require a frozen defender")
                if capability is not active_capability or type(capability) is not _BoundCapability:
                    raise HiddenBoundaryError("hidden capability identity is invalid")
                if not trusted_verifier.verify(active_attestation):
                    raise HiddenBoundaryError("active defender attestation no longer verifies")
                ref = _restricted_ref(restricted_ref)
                release_time = _utc(released_at, label="hidden release time")
                seal_time = _utc(sealed_at, label="hidden seal time")
                assert capability_issued_at is not None
                if release_time < capability_issued_at or seal_time < release_time:
                    raise HiddenBoundaryError("hidden release timestamps are out of order")
                if sequence > _MAX_RELEASES:
                    raise HiddenBoundaryError("hidden release cap is exhausted")

                from apar.evaluation import replay as replay_module

                frozen = replay_module._freeze_hidden_invocation(
                    invocation,
                    pinned_verifier=trusted_verifier,
                    pinned_attestation=active_attestation,
                )
                proof_id = "hpf_" + secrets.token_hex(16)
                frozen_document = replay_module._hidden_worker_frozen_document(
                    frozen, proof_id=proof_id
                )
                invocation_digest = _digest_document(frozen_document)
                decisions = replay_module._hidden_decision_bindings(frozen)
                decision_bindings_digest = _digest_document(
                    [item.model_dump(mode="json") for item in decisions]
                )
                freeze_receipt = _sign_freeze_receipt(
                    trusted_signer,
                    signer_key_id=signer_key_id,
                    public_key_base64=public_key_base64,
                    capability_digest=cast(str, capability_digest),
                    attestation=active_attestation,
                    restricted_ref=ref,
                    invocation_digest=invocation_digest,
                    release_sequence=sequence,
                    released_at=release_time,
                    frozen_at=seal_time,
                    decisions=decisions,
                )
                freeze_ref = trusted_store.put_bytes(
                    freeze_receipt.to_json(), HIDDEN_FREEZE_RECEIPT_MEDIA_TYPE
                )
                with lifecycle_lock:
                    if lifecycle != "RESERVED":
                        raise HiddenBoundaryError("hidden lifecycle reservation was lost")
                    lifecycle = "CONSUMED"
                    release_sequence = sequence
                request = {
                    "schema_version": "1.0.0",
                    "store_root": str(trusted_store.validated_worker_root()),
                    "restricted_ref": _ref_document(ref),
                    "frozen": frozen_document,
                    "freeze_receipt_base64": base64.b64encode(
                        freeze_receipt.to_json()
                    ).decode("ascii"),
                    "freeze_ref": _ref_document(freeze_ref),
                    "signer_private_seed_base64": base64.b64encode(
                        trusted_signer._worker_private_bytes()
                    ).decode("ascii"),
                    "worker_manifest_digest": worker_manifest.manifest_digest,
                    "decision_bindings_digest": decision_bindings_digest,
                    "capability_digest": cast(str, capability_digest),
                    "defender_attestation_digest": active_attestation.attestation_digest,
                    "defender_top_ref_digest": active_attestation.top_ref.sha256,
                    "bundle_manifest_digest": active_attestation.bundle_manifest_digest,
                    "release_sequence": sequence,
                    "released_at": _time_wire(release_time),
                    "sealed_at": _time_wire(seal_time),
                    "hidden_source": _hidden_source_request_document(
                        hidden_source_snapshot
                    ),
                }
                response = worker_client.invoke(request)
                outcome = _public_outcome_from_worker(
                    response,
                    verifier=evaluator_verifier,
                    proof_id=proof_id,
                    worker_manifest_digest=worker_manifest.manifest_digest,
                    decision_bindings_digest=decision_bindings_digest,
                    attestation=active_attestation,
                )
                return outcome
            except HiddenBoundaryError:
                raise
            except (HiddenWorkerError, TypeError) as error:
                raise HiddenBoundaryError("isolated hidden evaluation failed closed") from error
            finally:
                with lifecycle_lock:
                    if reserved_here and lifecycle == "RESERVED":
                        lifecycle = "CONSUMED"
                        release_sequence = max(release_sequence, 1)

    authority_instance = cast(
        HiddenEvaluationAuthority, object.__new__(_BoundAuthority)
    )
    return authority_instance


class _WorkerAttestationView(NamedTuple):
    attestation_digest: str
    top_ref: ArtifactRef
    bundle_manifest_digest: str


def _hidden_source_request_document(
    snapshot: _HiddenSourceSnapshot | None,
) -> dict[str, object] | None:
    """Materialize a worker document only from construction-time primitives."""
    if snapshot is None:
        return None
    reference, key_id, public_key, run_ids, event_ids, payment_ids, campaign_ids = (
        snapshot
    )
    sha256, media_type, size_bytes, relative_path = reference
    return {
        "development_campaign_ids": list(campaign_ids),
        "development_event_ids": list(event_ids),
        "development_payment_ids": list(payment_ids),
        "development_run_ids": list(run_ids),
        "receipt_ref": {
            "media_type": media_type,
            "relative_path": relative_path,
            "sha256": sha256,
            "size_bytes": size_bytes,
        },
        "source_public_key_base64": public_key,
        "source_signer_key_id": key_id,
    }


def _verify_hidden_source_worker(
    *,
    store: ArtifactStore,
    restricted_ref: ArtifactRef,
    restricted_payload: bytes,
    source_document: object,
    sealed_at_wire: str,
) -> None:
    """Reassemble independent runs only inside the isolated restricted worker."""
    if hashlib.sha256(restricted_payload).hexdigest() != restricted_ref.sha256:
        raise HiddenBoundaryError("hidden restricted context digest differs")
    expected = {
        "development_campaign_ids",
        "development_event_ids",
        "development_payment_ids",
        "development_run_ids",
        "receipt_ref",
        "source_public_key_base64",
        "source_signer_key_id",
    }
    if type(source_document) is not dict or set(source_document) != expected:
        raise HiddenBoundaryError("hidden source worker binding fields differ")
    source = cast(dict[str, object], source_document)
    collections: dict[str, tuple[str, ...]] = {}
    for name in (
        "development_campaign_ids",
        "development_event_ids",
        "development_payment_ids",
        "development_run_ids",
    ):
        value = source[name]
        if type(value) is not list or any(type(item) is not str for item in value):
            raise HiddenBoundaryError("hidden source identity binding differs")
        collections[name] = tuple(cast(list[str], value))
    if (
        len(collections["development_run_ids"]) != 200
        or len(set(collections["development_run_ids"])) != 200
    ):
        raise HiddenBoundaryError("hidden source development runs differ")
    from apar.evaluation.hidden_source import (
        HIDDEN_SOURCE_RECEIPT_MEDIA_TYPE,
        HiddenSourceReceipt,
        ordered_ids_digest,
    )
    receipt_ref = _ref_from_document(source["receipt_ref"])
    if receipt_ref.media_type != HIDDEN_SOURCE_RECEIPT_MEDIA_TYPE:
        raise HiddenBoundaryError("hidden source receipt reference differs")
    try:
        source_key_id = _exact_digest_field(source["source_signer_key_id"])
        public_key_base64 = cast(str, source["source_public_key_base64"])
        public_key = base64.b64decode(public_key_base64, validate=True)
        if len(public_key) != 32 or hashlib.sha256(public_key).hexdigest() != source_key_id:
            raise HiddenBoundaryError("hidden source authority identity differs")
    except (TypeError, ValueError, binascii.Error) as error:
        raise HiddenBoundaryError("hidden source authority identity differs") from error
    receipt_payload = store.read(receipt_ref)
    try:
        receipt = HiddenSourceReceipt.model_validate_json(receipt_payload)
        receipt_signature = base64.b64decode(
            receipt.signature_base64, validate=True
        )
        Ed25519PublicKey.from_public_bytes(public_key).verify(
            receipt_signature,
            canonical_json_bytes(receipt.unsigned_document()),
        )
    except (InvalidSignature, TypeError, ValueError, binascii.Error) as error:
        raise HiddenBoundaryError("hidden source receipt is invalid") from error
    if (
        canonical_json_bytes(receipt.model_dump(mode="json")) != receipt_payload
        or receipt.signer_key_id != source_key_id
        or receipt.public_key_base64 != public_key_base64
        or receipt.hidden_context_digest != restricted_ref.sha256
        or _time_wire(receipt.authority_as_of) != sealed_at_wire
        or receipt.development_run_ids_digest
        != ordered_ids_digest(collections["development_run_ids"])
        or receipt.development_event_ids_digest
        != ordered_ids_digest(collections["development_event_ids"])
        or receipt.development_payment_ids_digest
        != ordered_ids_digest(collections["development_payment_ids"])
        or receipt.development_campaign_ids_digest
        != ordered_ids_digest(collections["development_campaign_ids"])
        or set(receipt.run_ids).intersection(collections["development_run_ids"])
    ):
        raise HiddenBoundaryError("hidden source receipt worker lineage differs")
    from apar.evaluation.contracts import CorpusManifest, FrozenCorpus
    from apar.evaluation.regimes import frozen_corpus_digest
    from apar.evaluation.replay import ReplayEvaluationContext

    context = ReplayEvaluationContext.from_json(restricted_payload)
    corpus = FrozenCorpus(
        observations=context.observations,
        truth=context.truth,
        manifest=CorpusManifest(
            profile_id="defense-hidden-authority-v1",
            run_ids=receipt.run_ids,
            run_lineage_digests=receipt.run_lineage_digests,
            observation_count=len(context.observations),
            truth_count=len(context.truth),
        ),
    )
    campaigns_by_family = {
        family: {row.campaign_id for row in context.truth if row.family == family}
        for family in receipt.families
    }
    if (
        frozen_corpus_digest(corpus) != receipt.hidden_corpus_digest
        or {row.event_id for row in corpus.observations}.intersection(
            collections["development_event_ids"]
        )
        or {row.payment_id for row in corpus.truth}.intersection(
            collections["development_payment_ids"]
        )
        or {row.campaign_id for row in corpus.truth}.intersection(
            collections["development_campaign_ids"]
        )
        or {row.family for row in corpus.truth} != set(receipt.families)
        or any(len(campaigns_by_family[family]) != 1 for family in receipt.families)
        or len({row.campaign_id for row in context.truth}) != 4
        or any(
            row.viewpoint != "hidden" or row.label_source != "hidden_truth"
            for row in context.truth
        )
        or {row.event_id for row in context.truth}
        != {row.event_id for row in context.observations if row.is_decision_point}
    ):
        raise HiddenBoundaryError("hidden source corpus worker lineage differs")
    if (
        _time_wire(context.as_of) != sealed_at_wire
        or context.evaluation.kind.value != "hidden"
        or context.evaluation.value != "hidden"
    ):
        raise HiddenBoundaryError("hidden restricted context differs from source corpus")


def _isolated_worker_main(document: object) -> dict[str, object]:
    """Resolve and evaluate restricted truth only inside the pinned worker process."""
    if type(document) is not dict:
        raise HiddenBoundaryError("isolated worker request must be an exact object")
    request = cast(dict[str, object], document)
    expected = {
        "schema_version",
        "store_root",
        "restricted_ref",
        "frozen",
        "freeze_receipt_base64",
        "freeze_ref",
        "signer_private_seed_base64",
        "worker_manifest_digest",
        "decision_bindings_digest",
        "capability_digest",
        "defender_attestation_digest",
        "defender_top_ref_digest",
        "bundle_manifest_digest",
        "release_sequence",
        "released_at",
        "sealed_at",
        "hidden_source",
    }
    if set(request) != expected or request["schema_version"] != "1.0.0":
        raise HiddenBoundaryError("isolated worker request field set is invalid")
    capability_digest = _exact_digest_field(request["capability_digest"])
    attestation_digest = _exact_digest_field(
        request["defender_attestation_digest"]
    )
    top_ref_digest = _exact_digest_field(request["defender_top_ref_digest"])
    bundle_manifest_digest = _exact_digest_field(
        request["bundle_manifest_digest"]
    )
    worker_manifest_digest = _exact_digest_field(request["worker_manifest_digest"])
    release_sequence = _exact_release_sequence(request["release_sequence"])
    released_at_wire = _exact_time_wire(request["released_at"])
    sealed_at_wire = _exact_time_wire(request["sealed_at"])
    seed = _decode_base64_exact(
        request["signer_private_seed_base64"], expected_size=32
    )
    signer = EvaluatorSigningIdentity.from_private_bytes(seed)
    verifier = EvaluatorReplayVerifier.from_signer(signer)
    freeze_payload = _decode_base64_bounded(
        request["freeze_receipt_base64"], maximum=_MAX_RECEIPT_BYTES
    )
    freeze_receipt = HiddenDecisionFreezeReceipt.from_json(freeze_payload)
    if (
        freeze_receipt.signer_key_id != signer.key_id
        or freeze_receipt.public_key_base64 != signer.public_key_base64
    ):
        raise HiddenBoundaryError("isolated freeze receipt signer is invalid")
    restricted_ref = _restricted_ref(_ref_from_document(request["restricted_ref"]))
    freeze_ref = _ref_from_document(request["freeze_ref"])
    if (
        freeze_ref.media_type != HIDDEN_FREEZE_RECEIPT_MEDIA_TYPE
        or freeze_ref.size_bytes != len(freeze_payload)
        or freeze_ref.sha256 != hashlib.sha256(freeze_payload).hexdigest()
        or freeze_receipt.restricted_ref_digest != _ref_digest(restricted_ref)
        or freeze_receipt.capability_digest != capability_digest
        or freeze_receipt.defender_attestation_digest
        != attestation_digest
        or freeze_receipt.defender_top_ref_digest != top_ref_digest
        or freeze_receipt.bundle_manifest_digest != bundle_manifest_digest
        or freeze_receipt.release_sequence != release_sequence
        or _time_wire(freeze_receipt.released_at) != released_at_wire
        or _time_wire(freeze_receipt.frozen_at) != sealed_at_wire
        or freeze_receipt.invocation_digest != _digest_document(request["frozen"])
    ):
        raise HiddenBoundaryError("isolated freeze receipt bindings are invalid")
    from apar.evaluation import replay as replay_module

    decisions = replay_module._hidden_worker_decision_bindings_from_document(
        request["frozen"]
    )
    decision_bindings_digest = _digest_document(
        [item.model_dump(mode="json") for item in decisions]
    )
    if (
        decisions != freeze_receipt.decisions
        or decision_bindings_digest != request["decision_bindings_digest"]
    ):
        raise HiddenBoundaryError("isolated decision freeze does not verify")

    store_root = request["store_root"]
    if type(store_root) is not str or not store_root or len(store_root) > 4096:
        raise HiddenBoundaryError("isolated artifact root is invalid")
    store = ArtifactStore(Path(store_root))
    if store.read(freeze_ref) != freeze_payload:
        raise HiddenBoundaryError("persisted decision freeze receipt is inconsistent")
    payload = store.read(restricted_ref)
    if type(payload) is not bytes or not 0 < len(payload) <= _MAX_HIDDEN_BYTES:
        raise HiddenBoundaryError("restricted hidden payload violates resource limits")
    try:
        parsed = strict_json_loads(payload)
    except WireContractError as error:
        raise HiddenBoundaryError("restricted hidden payload is not canonical JSON") from error
    if canonical_json_bytes(parsed) != payload:
        raise HiddenBoundaryError("restricted hidden payload is not canonical JSON")
    if request["hidden_source"] is not None:
        _verify_hidden_source_worker(
            store=store,
            restricted_ref=restricted_ref,
            restricted_payload=payload,
            source_document=request["hidden_source"],
            sealed_at_wire=sealed_at_wire,
        )
    product = replay_module._evaluate_hidden_worker_document(request["frozen"], payload)
    if _time_wire(product.evaluator_as_of) != sealed_at_wire:
        raise HiddenBoundaryError("hidden evaluator time differs from sealed release time")
    results = tuple(item.result for item in product.evaluated)
    batch = signer.sign_batch(results)
    proof_id = results[0].hidden_public_proof_id
    if proof_id is None:
        raise HiddenBoundaryError("hidden evaluator omitted its public proof ID")
    proof = HiddenPublicProof.create(
        proof_id=proof_id,
        batch_content_digest=batch.batch_content_digest,
        decision_bindings_digest=decision_bindings_digest,
        bundle_manifest_digest=bundle_manifest_digest,
        defender_top_ref_digest=top_ref_digest,
        worker_manifest_digest=worker_manifest_digest,
        evaluator_context_token=results[0].evaluation_context_digest,
        cohort_mapping_token=(
            results[0].evaluation_lineage.cohort_mapping_digest
        ),
        issued_at=sealed_at_wire,
        signer=signer,
    )
    attestation_view = _WorkerAttestationView(
        attestation_digest=attestation_digest,
        top_ref=ArtifactRef(
            sha256=top_ref_digest,
            media_type="application/vnd.apar.defender-bundle+json",
            size_bytes=1,
            relative_path=f"{top_ref_digest}/payload",
        ),
        bundle_manifest_digest=bundle_manifest_digest,
    )
    receipt = _sign_evaluation_receipt(
        signer,
        signer_key_id=signer.key_id,
        public_key_base64=signer.public_key_base64,
        capability_digest=capability_digest,
        attestation=cast(VerifiedDefenderAttestation, attestation_view),
        restricted_ref=restricted_ref,
        payload=payload,
        product=product,
        freeze_receipt=freeze_receipt,
        freeze_ref=freeze_ref,
        release_sequence=release_sequence,
        released_at=freeze_receipt.released_at,
        sealed_at=product.evaluator_as_of,
        arm_evidence=tuple(item.hidden_evidence for item in product.evaluated),
        decision_bindings_digest=decision_bindings_digest,
        replay_batch_digest=batch.batch_digest,
        worker_manifest_digest=worker_manifest_digest,
        public_proof_id=proof.proof_id,
    )
    store.put_bytes(receipt.to_json(), HIDDEN_EVALUATION_RECEIPT_MEDIA_TYPE)
    if not verifier.verify_batch(batch) or not verifier.verify_public_proof(proof):
        raise HiddenBoundaryError("isolated aggregate signatures are invalid")
    return {
        "schema_version": "1.0.0",
        "batch_base64": base64.b64encode(batch.to_json()).decode("ascii"),
        "public_proof_base64": base64.b64encode(proof.to_json()).decode("ascii"),
    }


def _public_outcome_from_worker(
    response: object,
    *,
    verifier: EvaluatorReplayVerifier,
    proof_id: str,
    worker_manifest_digest: str,
    decision_bindings_digest: str,
    attestation: VerifiedDefenderAttestation,
) -> HiddenReplayOutcome:
    if type(response) is not dict or set(response) != {
        "schema_version",
        "batch_base64",
        "public_proof_base64",
    } or response["schema_version"] != "1.0.0":
        raise HiddenBoundaryError("isolated evaluator response field set is invalid")
    batch = VerifiedReplayBatch.from_json(
        _decode_base64_bounded(response["batch_base64"], maximum=32_000_000)
    )
    proof = HiddenPublicProof.from_json(
        _decode_base64_bounded(response["public_proof_base64"], maximum=64_000)
    )
    if (
        not verifier.verify_batch(batch)
        or not verifier.verify_public_proof(proof)
        or proof.proof_id != proof_id
        or proof.batch_content_digest != batch.batch_content_digest
        or proof.worker_manifest_digest != worker_manifest_digest
        or proof.decision_bindings_digest != decision_bindings_digest
        or proof.bundle_manifest_digest != attestation.bundle_manifest_digest
        or proof.defender_top_ref_digest != attestation.top_ref.sha256
        or tuple(row.arm.value for row in batch.results) != _ARM_ORDER
        or any(
            row.hidden_public_proof_id != proof.proof_id
            or row.evaluation.kind.value != "hidden"
            or not row.assurance.hidden_access_clean
            or row.evaluation_context_digest != proof.evaluator_context_token
            or row.evaluation_lineage.cohort_mapping_digest
            != proof.cohort_mapping_token
            for row in batch.results
        )
    ):
        raise HiddenBoundaryError("isolated evaluator aggregate proof is invalid")
    return HiddenReplayOutcome(batch=batch, public_proof=proof)


def _ref_document(ref: ArtifactRef) -> dict[str, object]:
    return {
        "sha256": ref.sha256,
        "media_type": ref.media_type,
        "size_bytes": ref.size_bytes,
        "relative_path": ref.relative_path,
    }


def _ref_from_document(document: object) -> ArtifactRef:
    if type(document) is not dict or set(document) != {
        "sha256",
        "media_type",
        "size_bytes",
        "relative_path",
    }:
        raise HiddenBoundaryError("isolated artifact reference is invalid")
    values = cast(dict[str, object], document)
    return ArtifactRef(
        sha256=cast(str, values["sha256"]),
        media_type=cast(str, values["media_type"]),
        size_bytes=cast(int, values["size_bytes"]),
        relative_path=cast(str, values["relative_path"]),
    )


def _decode_base64_exact(value: object, *, expected_size: int) -> bytes:
    payload = _decode_base64_bounded(value, maximum=expected_size)
    if len(payload) != expected_size:
        raise HiddenBoundaryError("isolated base64 field has invalid size")
    return payload


def _decode_base64_bounded(value: object, *, maximum: int) -> bytes:
    if type(value) is not str or not value or len(value) > maximum * 2:
        raise HiddenBoundaryError("isolated base64 field violates bounds")
    try:
        payload = base64.b64decode(value, validate=True)
    except (TypeError, ValueError) as error:
        raise HiddenBoundaryError("isolated base64 field is invalid") from error
    if not payload or len(payload) > maximum:
        raise HiddenBoundaryError("isolated base64 field violates bounds")
    return payload


def _restricted_ref(ref: ArtifactRef) -> ArtifactRef:
    if type(ref) is not ArtifactRef:
        raise HiddenBoundaryError("restricted ref must be exact")
    if (
        ref.media_type != HIDDEN_CONTEXT_MEDIA_TYPE
        or not 0 < ref.size_bytes <= _MAX_HIDDEN_BYTES
    ):
        raise HiddenBoundaryError("restricted ref violates hidden context media limits")
    return ref


def _sign_freeze_receipt(
    signer: EvaluatorSigningIdentity,
    *,
    signer_key_id: str,
    public_key_base64: str,
    capability_digest: str,
    attestation: VerifiedDefenderAttestation,
    restricted_ref: ArtifactRef,
    invocation_digest: str,
    release_sequence: int,
    released_at: datetime,
    frozen_at: datetime,
    decisions: tuple[HiddenDecisionBinding, ...],
) -> HiddenDecisionFreezeReceipt:
    fields: dict[str, object] = {
        "capability_digest": capability_digest,
        "defender_attestation_digest": attestation.attestation_digest,
        "defender_top_ref_digest": attestation.top_ref.sha256,
        "bundle_manifest_digest": attestation.bundle_manifest_digest,
        "restricted_ref_digest": _ref_digest(restricted_ref),
        "invocation_digest": invocation_digest,
        "release_sequence": release_sequence,
        "released_at": released_at,
        "frozen_at": frozen_at,
        "decisions": decisions,
        "signer_key_id": signer_key_id,
        "public_key_base64": public_key_base64,
    }
    return cast(
        HiddenDecisionFreezeReceipt,
        _signed_contract(HiddenDecisionFreezeReceipt, fields, signer),
    )


def _sign_evaluation_receipt(
    signer: EvaluatorSigningIdentity,
    *,
    signer_key_id: str,
    public_key_base64: str,
    capability_digest: str,
    attestation: VerifiedDefenderAttestation,
    restricted_ref: ArtifactRef,
    payload: bytes,
    product: object,
    freeze_receipt: HiddenDecisionFreezeReceipt,
    freeze_ref: ArtifactRef,
    release_sequence: int,
    released_at: datetime,
    sealed_at: datetime,
    arm_evidence: tuple[HiddenArmEvidenceBinding, ...],
    decision_bindings_digest: str,
    replay_batch_digest: str,
    worker_manifest_digest: str,
    public_proof_id: str,
) -> HiddenEvaluationReceipt:
    product_view = cast(_HiddenProductView, product)
    context_digest = product_view.evaluator_context_digest
    lineage_digest = product_view.evaluation_lineage_digest
    fields: dict[str, object] = {
        "capability_digest": capability_digest,
        "defender_attestation_digest": attestation.attestation_digest,
        "defender_top_ref_digest": attestation.top_ref.sha256,
        "bundle_manifest_digest": attestation.bundle_manifest_digest,
        "restricted_ref_digest": _ref_digest(restricted_ref),
        "restricted_artifact_digest": restricted_ref.sha256,
        "canonical_content_digest": hashlib.sha256(payload).hexdigest(),
        "evaluator_context_digest": context_digest,
        "restricted_cohort_mapping_digest": (
            product_view.restricted_cohort_mapping_digest
        ),
        "descriptor_lineage_digest": lineage_digest,
        "decision_freeze_receipt_digest": freeze_receipt.receipt_digest,
        "decision_freeze_ref_digest": freeze_ref.sha256,
        "decision_bindings_digest": decision_bindings_digest,
        "replay_batch_digest": replay_batch_digest,
        "worker_manifest_digest": worker_manifest_digest,
        "public_proof_id": public_proof_id,
        "release_sequence": release_sequence,
        "released_at": released_at,
        "sealed_at": sealed_at,
        "arm_evidence": arm_evidence,
        "signer_key_id": signer_key_id,
        "public_key_base64": public_key_base64,
    }
    return cast(
        HiddenEvaluationReceipt,
        _signed_contract(HiddenEvaluationReceipt, fields, signer),
    )


def _signed_contract(
    contract_type: type[HiddenDecisionFreezeReceipt] | type[HiddenEvaluationReceipt],
    fields: dict[str, object],
    signer: EvaluatorSigningIdentity,
) -> HiddenDecisionFreezeReceipt | HiddenEvaluationReceipt:
    unsigned = cast(Any, contract_type).model_construct(
        **fields,
        signature_base64="",
        receipt_digest="0" * 64,
    ).model_dump(mode="json", exclude={"signature_base64", "receipt_digest"})
    signature = signer._sign(unsigned)
    digest = _digest_document({**unsigned, "signature_base64": signature})
    return contract_type.model_validate(
        {**fields, "signature_base64": signature, "receipt_digest": digest}
    )


def _validate_signed_receipt(
    receipt: HiddenDecisionFreezeReceipt | HiddenEvaluationReceipt,
) -> None:
    try:
        public = base64.b64decode(receipt.public_key_base64, validate=True)
        signature = base64.b64decode(receipt.signature_base64, validate=True)
    except (ValueError, TypeError) as error:
        raise ValueError("hidden receipt signature encoding is invalid") from error
    if len(public) != 32 or len(signature) != 64:
        raise ValueError("hidden receipt signature length is invalid")
    if hashlib.sha256(public).hexdigest() != receipt.signer_key_id:
        raise ValueError("hidden receipt signer identity is inconsistent")
    try:
        Ed25519PublicKey.from_public_bytes(public).verify(
            signature, canonical_json_bytes(receipt.unsigned_document())
        )
    except (InvalidSignature, ValueError) as error:
        raise ValueError("hidden receipt signature is invalid") from error
    expected = _digest_document(
        {**receipt.unsigned_document(), "signature_base64": receipt.signature_base64}
    )
    if receipt.receipt_digest != expected:
        raise ValueError("hidden receipt digest is inconsistent")


def _receipt_to_json(
    receipt: HiddenDecisionFreezeReceipt | HiddenEvaluationReceipt,
    exact_type: type[HiddenDecisionFreezeReceipt] | type[HiddenEvaluationReceipt],
) -> bytes:
    if type(receipt) is not exact_type:
        raise HiddenBoundaryError("hidden receipt must have its exact type")
    checked = exact_type.model_validate(
        receipt.model_dump(mode="python", warnings=False), strict=True
    )
    payload = canonical_json_bytes(checked.model_dump(mode="json"))
    if len(payload) > _MAX_RECEIPT_BYTES:
        raise HiddenBoundaryError("hidden receipt exceeds its resource cap")
    return payload


def _receipt_from_json(
    contract_type: type[HiddenDecisionFreezeReceipt] | type[HiddenEvaluationReceipt],
    payload: bytes,
) -> HiddenDecisionFreezeReceipt | HiddenEvaluationReceipt:
    if type(payload) is not bytes or not 0 < len(payload) <= _MAX_RECEIPT_BYTES:
        raise HiddenBoundaryError("hidden receipt payload is invalid")
    try:
        document = strict_json_loads(payload)
        if type(document) is not dict:
            raise HiddenBoundaryError("hidden receipt must be a JSON object")
        for field in ("decisions", "arm_evidence"):
            if type(document.get(field)) is list:
                document[field] = tuple(document[field])
        receipt = contract_type.model_validate(document)
        if receipt.to_json() != payload:
            raise HiddenBoundaryError("hidden receipt JSON is not canonical")
        return receipt
    except (ValidationError, WireContractError) as error:
        raise HiddenBoundaryError("hidden receipt failed canonical validation") from error


def _ref_digest(ref: ArtifactRef) -> str:
    return _digest_document(
        {
            "sha256": ref.sha256,
            "media_type": ref.media_type,
            "size_bytes": ref.size_bytes,
            "relative_path": ref.relative_path,
        }
    )


def _digest_document(document: object) -> str:
    return hashlib.sha256(canonical_json_bytes(document)).hexdigest()


def _utc(value: datetime, *, label: str) -> datetime:
    if type(value) is not datetime:
        raise HiddenBoundaryError(f"{label} must be exact")
    try:
        return validate_utc_timestamp(value)
    except ValueError as error:
        raise HiddenBoundaryError(f"{label} is invalid") from error


def _time_wire(value: datetime) -> str:
    return value.isoformat().replace("+00:00", "Z")


def _validate_digest(value: str) -> None:
    if type(value) is not str or len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError("hidden evidence digest must be lowercase SHA-256")


def _exact_digest_field(value: object) -> str:
    if type(value) is not str:
        raise HiddenBoundaryError("isolated digest field must be an exact string")
    try:
        _validate_digest(value)
    except ValueError as error:
        raise HiddenBoundaryError("isolated digest field is invalid") from error
    return value


def _exact_release_sequence(value: object) -> int:
    if type(value) is not int or not 1 <= value <= _MAX_RELEASES:
        raise HiddenBoundaryError("isolated release sequence is invalid")
    return value


def _exact_time_wire(value: object) -> str:
    if (
        type(value) is not str
        or not value.endswith("Z")
        or "T" not in value
        or len(value) > 40
    ):
        raise HiddenBoundaryError("isolated timestamp wire value is invalid")
    return value


_ARM_ORDER = ("rules_only", "gbdt_only", "layered_hybrid")


__all__ = [
    "HIDDEN_CONTEXT_MEDIA_TYPE",
    "HIDDEN_EVALUATION_RECEIPT_MEDIA_TYPE",
    "HIDDEN_FREEZE_RECEIPT_MEDIA_TYPE",
    "HiddenArmEvidenceBinding",
    "HiddenBoundaryError",
    "HiddenDecisionBinding",
    "HiddenDecisionFreezeReceipt",
    "HiddenEvaluationAuthority",
    "HiddenEvaluationCapability",
    "HiddenEvaluationReceipt",
    "HiddenReplayOutcome",
]
