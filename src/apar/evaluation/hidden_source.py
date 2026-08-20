"""Neutral contracts for independently authenticated hidden-run source evidence."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime
from typing import Literal, cast

from pydantic import field_validator, model_validator

from apar.contracts._validation import ExternalContract, validate_utc_timestamp
from apar.evaluation.contracts import Family
from apar.storage.artifacts import ArtifactRef

HIDDEN_SOURCE_RECEIPT_MEDIA_TYPE = (
    "application/vnd.apar.restricted-hidden-source-receipt+json"
)
_HEX = frozenset("0123456789abcdef")
_FAMILIES: tuple[Family, ...] = (
    "agentic_intent_abuse",
    "app_scam_mule",
    "card_testing_cnp",
    "synthetic_merchant_refund",
)


def _digest(value: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or value != value.lower()
        or any(character not in _HEX for character in value)
    ):
        raise ValueError("hidden source digest must be lowercase SHA-256")
    return value


def _reference(document: object) -> ArtifactRef:
    if type(document) is not dict or set(document) != {
        "media_type",
        "relative_path",
        "sha256",
        "size_bytes",
    }:
        raise ValueError("hidden source reference fields differ")
    checked = cast(dict[str, object], document)
    reference = ArtifactRef(
        sha256=cast(str, checked["sha256"]),
        media_type=cast(str, checked["media_type"]),
        size_bytes=cast(int, checked["size_bytes"]),
        relative_path=cast(str, checked["relative_path"]),
    )
    if (
        _digest(reference.sha256) != reference.sha256
        or reference.media_type != "application/json"
        or type(reference.size_bytes) is not int
        or not 0 < reference.size_bytes <= 64 * 1024 * 1024
        or reference.relative_path != f"{reference.sha256}/payload"
    ):
        raise ValueError("hidden source reference differs")
    return reference


class HiddenSourceReceipt(ExternalContract):
    """Restricted source-authority receipt for four independent synthetic runs."""

    schema_version: Literal["1.0.0"] = "1.0.0"
    kind: Literal["authenticated_independent_hidden_runs"] = (
        "authenticated_independent_hidden_runs"
    )
    profile_sha256: str
    ensemble_ref_sha256: str
    development_corpus_digest: str
    development_run_ids_digest: str
    development_event_ids_digest: str
    development_payment_ids_digest: str
    development_campaign_ids_digest: str
    manifest_refs: tuple[dict[str, object], ...]
    run_ids: tuple[str, ...]
    run_lineage_digests: tuple[str, ...]
    families: tuple[Family, ...]
    hidden_corpus_digest: str
    hidden_context_digest: str
    minimum_simulation_start: datetime
    authority_as_of: datetime
    signer_key_id: str
    public_key_base64: str
    signature_base64: str

    @field_validator(
        "profile_sha256",
        "ensemble_ref_sha256",
        "development_corpus_digest",
        "development_run_ids_digest",
        "development_event_ids_digest",
        "development_payment_ids_digest",
        "development_campaign_ids_digest",
        "hidden_corpus_digest",
        "hidden_context_digest",
        "signer_key_id",
    )
    @classmethod
    def digests_are_exact(cls, value: str) -> str:
        return _digest(value)

    @model_validator(mode="after")
    def source_is_closed(self) -> HiddenSourceReceipt:
        references = tuple(_reference(item) for item in self.manifest_refs)
        if (
            len(references) != 4
            or len({item.sha256 for item in references}) != 4
            or len(self.run_ids) != 4
            or len(set(self.run_ids)) != 4
            or len(self.run_lineage_digests) != 4
            or self.families != _FAMILIES
            or validate_utc_timestamp(self.minimum_simulation_start)
            != self.minimum_simulation_start
            or validate_utc_timestamp(self.authority_as_of) != self.authority_as_of
            or self.authority_as_of <= self.minimum_simulation_start
        ):
            raise ValueError("hidden source receipt differs")
        for value in self.run_lineage_digests:
            _digest(value)
        return self

    @property
    def references(self) -> tuple[ArtifactRef, ...]:
        return tuple(_reference(item) for item in self.manifest_refs)

    def unsigned_document(self) -> dict[str, object]:
        return self.model_dump(mode="json", exclude={"signature_base64"})

    def to_json(self) -> bytes:
        from apar.runs.wire import canonical_json_bytes

        return canonical_json_bytes(self.model_dump(mode="json"))


@dataclass(frozen=True, slots=True, repr=False)
class HiddenSourceWorkerBinding:
    """Ephemeral worker-only binding; no restricted values enter public output."""

    receipt_ref: ArtifactRef
    source_signer_key_id: str
    source_public_key_base64: str
    development_run_ids: tuple[str, ...]
    development_event_ids: tuple[str, ...]
    development_payment_ids: tuple[str, ...]
    development_campaign_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if (
            type(self.receipt_ref) is not ArtifactRef
            or self.receipt_ref.media_type != HIDDEN_SOURCE_RECEIPT_MEDIA_TYPE
            or _digest(self.source_signer_key_id) != self.source_signer_key_id
            or type(self.source_public_key_base64) is not str
            or not self.source_public_key_base64
            or len(self.development_run_ids) != 200
            or len(set(self.development_run_ids)) != 200
        ):
            raise ValueError("hidden source worker binding differs")

    def __repr__(self) -> str:
        return "HiddenSourceWorkerBinding(<restricted>)"


def ordered_ids_digest(values: tuple[str, ...]) -> str:
    """Bind an ordered identity collection without exposing it in source receipts."""
    from apar.runs.wire import canonical_json_bytes

    if type(values) is not tuple or any(type(item) is not str for item in values):
        raise ValueError("hidden source identity collection differs")
    return hashlib.sha256(canonical_json_bytes(list(values))).hexdigest()


__all__ = [
    "HIDDEN_SOURCE_RECEIPT_MEDIA_TYPE",
    "HiddenSourceReceipt",
    "HiddenSourceWorkerBinding",
    "ordered_ids_digest",
]
