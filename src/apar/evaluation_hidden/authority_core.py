"""Opaque authority-owned lifecycle for restricted defense evaluation."""

from __future__ import annotations

import base64
import hashlib
import secrets
from datetime import datetime
from typing import Any, Literal, Never, Protocol, cast

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)
from pydantic import ValidationError, field_validator, model_validator

from apar.contracts._validation import ExternalContract, validate_utc_timestamp
from apar.evaluation.defender_attestation import (
    DefenderBundleVerifier,
    VerifiedDefenderAttestation,
)
from apar.runs.wire import WireContractError, canonical_json_bytes, strict_json_loads
from apar.storage.artifacts import ArtifactRef, ArtifactStore

HIDDEN_CONTEXT_MEDIA_TYPE = "application/vnd.apar.hidden-evaluation-context+json"
HIDDEN_FREEZE_RECEIPT_MEDIA_TYPE = (
    "application/vnd.apar.hidden-decision-freeze-receipt+json"
)
_MAX_HIDDEN_BYTES = 64 * 1024 * 1024
_MAX_RECEIPT_BYTES = 256 * 1024
_MAX_RELEASES = 64
_OBJECT_TOKEN = object()


class HiddenBoundaryError(ValueError):
    """A hidden-evaluation caller violated the sealed lifecycle."""


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

    schema_version: Literal["2.0.0"] = "2.0.0"
    capability_digest: str
    defender_attestation_digest: str
    defender_top_ref_digest: str
    bundle_manifest_digest: str
    restricted_ref_digest: str
    restricted_artifact_digest: str
    canonical_content_digest: str
    evaluator_context_digest: str
    descriptor_lineage_digest: str
    decision_freeze_receipt_digest: str
    decision_freeze_ref_digest: str
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
        "descriptor_lineage_digest",
        "decision_freeze_receipt_digest",
        "decision_freeze_ref_digest",
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
    ) -> HiddenEvaluationAuthority:
        if cls is not HiddenEvaluationAuthority:
            raise HiddenBoundaryError("hidden authority cannot be constructed externally")
        return _new_authority(verifier, restricted_store)

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
    evaluation_lineage_digest: str


def _new_authority(
    verifier: DefenderBundleVerifier,
    restricted_store: ArtifactStore,
) -> HiddenEvaluationAuthority:
    if type(verifier) is not DefenderBundleVerifier:
        raise HiddenBoundaryError("hidden authority requires the exact neutral verifier")
    if type(restricted_store) is not ArtifactStore:
        raise HiddenBoundaryError("hidden authority requires an exact restricted store")

    trusted_verifier = verifier
    trusted_store = restricted_store
    signing_key = Ed25519PrivateKey.generate()
    public_key = signing_key.public_key().public_bytes(
        serialization.Encoding.Raw, serialization.PublicFormat.Raw
    )
    public_key_base64 = base64.b64encode(public_key).decode("ascii")
    signer_key_id = hashlib.sha256(public_key).hexdigest()
    initialized = False
    active_capability: HiddenEvaluationCapability | None = None
    active_attestation: VerifiedDefenderAttestation | None = None
    capability_digest: str | None = None
    capability_issued_at: datetime | None = None
    release_sequence = 0
    consumed_invocations: tuple[str, ...] = ()
    issued_receipts: tuple[HiddenEvaluationReceipt, ...] = ()

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
        ) -> _BoundAuthority:
            del cls, verifier, restricted_store
            raise HiddenBoundaryError("hidden authority cannot be constructed externally")

        def __init__(
            self,
            verifier: DefenderBundleVerifier,
            restricted_store: ArtifactStore,
        ) -> None:
            nonlocal initialized
            if (
                initialized
                or verifier is not trusted_verifier
                or restricted_store is not trusted_store
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
            if active_capability is not None:
                raise HiddenBoundaryError("hidden authority is already frozen")
            if (
                type(attestation) is not VerifiedDefenderAttestation
                or not trusted_verifier.verify(attestation)
            ):
                raise HiddenBoundaryError(
                    "hidden release requires an exact verified signed defender attestation"
                )
            checked_at = _utc(issued_at, label="hidden capability issue time")
            if attestation.frozen_at > checked_at:
                raise HiddenBoundaryError("defender freeze follows hidden capability issue")
            active_attestation = attestation
            capability_issued_at = checked_at
            capability_digest = _digest_document(
                {
                    "schema_version": "2.0.0",
                    "attestation_digest": attestation.attestation_digest,
                    "bundle_manifest_digest": attestation.bundle_manifest_digest,
                    "bundle_id": attestation.bundle_id,
                    "issued_at": _time_wire(checked_at),
                    "nonce": secrets.token_hex(32),
                }
            )
            active_capability = _BoundCapability(_OBJECT_TOKEN)
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
            nonlocal release_sequence, consumed_invocations, issued_receipts
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
            if release_sequence >= _MAX_RELEASES:
                raise HiddenBoundaryError("hidden release cap is exhausted")

            # Importing the evaluator surface here preserves the package boundary and
            # gives callers no callback through which restricted bytes could escape.
            from apar.evaluation import replay as replay_module

            frozen = replay_module._freeze_hidden_invocation(
                invocation,
                pinned_verifier=trusted_verifier,
                pinned_attestation=active_attestation,
            )
            invocation_digest = replay_module._hidden_invocation_digest(frozen)
            if invocation_digest in consumed_invocations:
                raise HiddenBoundaryError("hidden replay invocation is already consumed")
            decisions = replay_module._hidden_decision_bindings(frozen)
            sequence = release_sequence + 1
            freeze_receipt = _sign_freeze_receipt(
                signing_key,
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
            # Persistence is part of the release condition and happens before the
            # first restricted read. A failed write therefore cannot release truth.
            freeze_ref = trusted_store.put_bytes(
                freeze_receipt.to_json(), HIDDEN_FREEZE_RECEIPT_MEDIA_TYPE
            )
            try:
                payload = trusted_store.read(ref)
            except Exception as error:
                raise HiddenBoundaryError(
                    "restricted reference is invalid for the pinned store"
                ) from error
            if type(payload) is not bytes or not 0 < len(payload) <= _MAX_HIDDEN_BYTES:
                raise HiddenBoundaryError("restricted hidden payload violates resource limits")
            try:
                document = strict_json_loads(payload)
            except WireContractError as error:
                raise HiddenBoundaryError(
                    "restricted hidden payload is not canonical JSON"
                ) from error
            if canonical_json_bytes(document) != payload:
                raise HiddenBoundaryError("restricted hidden payload is not canonical JSON")

            product = replay_module._evaluate_hidden_frozen(frozen, payload)
            evidence = replay_module._hidden_product_evidence(product)
            receipt = _sign_evaluation_receipt(
                signing_key,
                signer_key_id=signer_key_id,
                public_key_base64=public_key_base64,
                capability_digest=cast(str, capability_digest),
                attestation=active_attestation,
                restricted_ref=ref,
                payload=payload,
                product=product,
                freeze_receipt=freeze_receipt,
                freeze_ref=freeze_ref,
                release_sequence=sequence,
                released_at=release_time,
                sealed_at=seal_time,
                arm_evidence=evidence,
            )
            outcome = replay_module._finalize_hidden_product(
                product,
                receipt,
                freeze_receipt=freeze_receipt,
                freeze_ref=freeze_ref,
            )
            consumed_invocations = (*consumed_invocations, invocation_digest)
            issued_receipts = (*issued_receipts, receipt)
            release_sequence = sequence
            return outcome

        def receipt_from_json(self, payload: bytes) -> HiddenEvaluationReceipt:
            candidate = HiddenEvaluationReceipt.from_json(payload)
            if (
                candidate.signer_key_id != signer_key_id
                or candidate.public_key_base64 != public_key_base64
            ):
                raise HiddenBoundaryError("hidden receipt signer is not this authority")
            for receipt in issued_receipts:
                if candidate.receipt_digest == receipt.receipt_digest:
                    return receipt
            raise HiddenBoundaryError("hidden receipt is not active for this authority")

    return cast(HiddenEvaluationAuthority, object.__new__(_BoundAuthority))


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
    key: Ed25519PrivateKey,
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
        _signed_contract(HiddenDecisionFreezeReceipt, fields, key),
    )


def _sign_evaluation_receipt(
    key: Ed25519PrivateKey,
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
        "descriptor_lineage_digest": lineage_digest,
        "decision_freeze_receipt_digest": freeze_receipt.receipt_digest,
        "decision_freeze_ref_digest": freeze_ref.sha256,
        "release_sequence": release_sequence,
        "released_at": released_at,
        "sealed_at": sealed_at,
        "arm_evidence": arm_evidence,
        "signer_key_id": signer_key_id,
        "public_key_base64": public_key_base64,
    }
    return cast(
        HiddenEvaluationReceipt,
        _signed_contract(HiddenEvaluationReceipt, fields, key),
    )


def _signed_contract(
    contract_type: type[HiddenDecisionFreezeReceipt] | type[HiddenEvaluationReceipt],
    fields: dict[str, object],
    key: Ed25519PrivateKey,
) -> HiddenDecisionFreezeReceipt | HiddenEvaluationReceipt:
    unsigned = cast(Any, contract_type).model_construct(
        **fields,
        signature_base64="",
        receipt_digest="0" * 64,
    ).model_dump(mode="json", exclude={"signature_base64", "receipt_digest"})
    signature = base64.b64encode(key.sign(canonical_json_bytes(unsigned))).decode("ascii")
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


_ARM_ORDER = ("rules_only", "gbdt_only", "layered_hybrid")


__all__ = [
    "HIDDEN_CONTEXT_MEDIA_TYPE",
    "HIDDEN_FREEZE_RECEIPT_MEDIA_TYPE",
    "HiddenArmEvidenceBinding",
    "HiddenBoundaryError",
    "HiddenDecisionBinding",
    "HiddenDecisionFreezeReceipt",
    "HiddenEvaluationAuthority",
    "HiddenEvaluationCapability",
    "HiddenEvaluationReceipt",
]
