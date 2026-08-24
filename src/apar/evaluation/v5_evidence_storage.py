"""Exclusive manifest-last chunk storage for locked Sentinel v5 evidence."""

from __future__ import annotations

import hashlib
import json
import os
import stat
from pathlib import Path
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode()


def _digest(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


class V5EvidenceChunk(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    index: int = Field(ge=0)
    filename: str
    bytes: int = Field(gt=0)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class V5LockedStartReceipt(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal["apar-sentinel-v5-locked-start-receipt/1"]
    run_binding_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    authorization_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    started_at_utc: str
    receipt_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def digest_matches(self) -> Self:
        if self.receipt_sha256 != _digest(
            self.model_dump(mode="json", exclude={"receipt_sha256"})
        ):
            raise ValueError("locked start receipt digest mismatch")
        return self


class V5LockedCompletionReceipt(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal["apar-sentinel-v5-locked-completion-receipt/1"]
    start_receipt_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    completed_at_utc: str
    elapsed_ms: float = Field(gt=0.0)
    payload_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    payload_bytes: int = Field(gt=0)
    receipt_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def digest_matches(self) -> Self:
        if self.receipt_sha256 != _digest(
            self.model_dump(mode="json", exclude={"receipt_sha256"})
        ):
            raise ValueError("locked completion receipt digest mismatch")
        return self


class V5ChunkedEvidenceManifest(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal["apar-sentinel-v5-chunked-evidence/1"]
    content_encoding: Literal["opaque-locked-evidence-bytes"]
    publication: Literal["content_chunks_then_atomic_exclusive_manifest"]
    chunks_directory: str
    chunk_size_bytes: int = Field(gt=0)
    maximum_envelope_bytes: int = Field(gt=0)
    maximum_chunk_count: int = Field(gt=0)
    normal_git_blob_limit_bytes: int = Field(gt=0)
    payload_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    payload_bytes: int = Field(gt=0)
    run_binding_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    start_receipt: V5LockedStartReceipt
    completion_receipt: V5LockedCompletionReceipt
    chunks: tuple[V5EvidenceChunk, ...]
    manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def manifest_reconciles(self) -> Self:
        if not self.chunks or len(self.chunks) > self.maximum_chunk_count:
            raise ValueError("locked evidence chunk count differs")
        if self.chunk_size_bytes >= self.normal_git_blob_limit_bytes:
            raise ValueError("locked evidence chunk reaches the Git blob limit")
        if self.payload_bytes > self.maximum_envelope_bytes:
            raise ValueError("locked evidence exceeds the maximum envelope")
        if sum(chunk.bytes for chunk in self.chunks) != self.payload_bytes:
            raise ValueError("locked evidence chunk byte total differs")
        expected_names = tuple(
            f"part-{index:04d}.bin" for index in range(len(self.chunks))
        )
        if tuple(chunk.index for chunk in self.chunks) != tuple(
            range(len(self.chunks))
        ) or tuple(chunk.filename for chunk in self.chunks) != expected_names:
            raise ValueError("locked evidence chunk order differs")
        if any(
            chunk.bytes != self.chunk_size_bytes for chunk in self.chunks[:-1]
        ) or self.chunks[-1].bytes > self.chunk_size_bytes:
            raise ValueError("locked evidence chunk size differs")
        if (
            self.completion_receipt.start_receipt_sha256
            != self.start_receipt.receipt_sha256
            or self.completion_receipt.payload_sha256 != self.payload_sha256
            or self.completion_receipt.payload_bytes != self.payload_bytes
            or self.start_receipt.run_binding_sha256 != self.run_binding_sha256
        ):
            raise ValueError("locked evidence receipt lineage differs")
        if self.manifest_sha256 != _digest(
            self.model_dump(mode="json", exclude={"manifest_sha256"})
        ):
            raise ValueError("locked evidence manifest digest mismatch")
        return self


def _write_exclusive(path: Path, content: bytes) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o600)
    try:
        view = memoryview(content)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("exclusive evidence write made no progress")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _ensure_absent(path: Path, label: str) -> None:
    if os.path.lexists(path):
        raise FileExistsError(f"{label} already exists")


def publish_v5_chunked_evidence(
    *,
    payload_bytes: bytes,
    target_manifest: Path,
    run_binding_sha256: str,
    authorization_sha256: str,
    started_at_utc: str,
    completed_at_utc: str,
    elapsed_ms: float,
    chunk_size_bytes: int,
    maximum_envelope_bytes: int,
    maximum_chunk_count: int,
    normal_git_blob_limit_bytes: int,
) -> V5ChunkedEvidenceManifest:
    """Write chunks first and atomically publish one exclusive manifest last."""
    if type(payload_bytes) is not bytes or not payload_bytes:
        raise ValueError("locked evidence payload bytes are empty")
    if len(payload_bytes) > maximum_envelope_bytes:
        raise ValueError("locked evidence exceeds the maximum envelope")
    if not 0 < chunk_size_bytes < normal_git_blob_limit_bytes:
        raise ValueError("locked evidence chunk size reaches the Git blob limit")
    chunk_count = (len(payload_bytes) + chunk_size_bytes - 1) // chunk_size_bytes
    if not 0 < chunk_count <= maximum_chunk_count:
        raise ValueError("locked evidence exceeds the maximum chunk count")
    target_manifest = target_manifest.absolute()
    chunks_directory = target_manifest.with_name(f"{target_manifest.name}.chunks")
    _ensure_absent(target_manifest, "candidate manifest target")
    _ensure_absent(chunks_directory, "candidate chunk directory")
    target_manifest.parent.mkdir(parents=True, exist_ok=True)
    chunks_directory.mkdir(mode=0o700)

    chunks: list[V5EvidenceChunk] = []
    for index in range(chunk_count):
        content = payload_bytes[
            index * chunk_size_bytes : (index + 1) * chunk_size_bytes
        ]
        filename = f"part-{index:04d}.bin"
        path = chunks_directory / filename
        _write_exclusive(path, content)
        chunks.append(
            V5EvidenceChunk(
                index=index,
                filename=filename,
                bytes=len(content),
                sha256=hashlib.sha256(content).hexdigest(),
            )
        )

    start_values = {
        "schema_version": "apar-sentinel-v5-locked-start-receipt/1",
        "run_binding_sha256": run_binding_sha256,
        "authorization_sha256": authorization_sha256,
        "started_at_utc": started_at_utc,
    }
    start_values["receipt_sha256"] = _digest(start_values)
    start = V5LockedStartReceipt.model_validate(start_values)
    payload_sha256 = hashlib.sha256(payload_bytes).hexdigest()
    completion_values = {
        "schema_version": "apar-sentinel-v5-locked-completion-receipt/1",
        "start_receipt_sha256": start.receipt_sha256,
        "completed_at_utc": completed_at_utc,
        "elapsed_ms": elapsed_ms,
        "payload_sha256": payload_sha256,
        "payload_bytes": len(payload_bytes),
    }
    completion_values["receipt_sha256"] = _digest(completion_values)
    completion = V5LockedCompletionReceipt.model_validate(completion_values)
    manifest_values = {
        "schema_version": "apar-sentinel-v5-chunked-evidence/1",
        "content_encoding": "opaque-locked-evidence-bytes",
        "publication": "content_chunks_then_atomic_exclusive_manifest",
        "chunks_directory": chunks_directory.name,
        "chunk_size_bytes": chunk_size_bytes,
        "maximum_envelope_bytes": maximum_envelope_bytes,
        "maximum_chunk_count": maximum_chunk_count,
        "normal_git_blob_limit_bytes": normal_git_blob_limit_bytes,
        "payload_sha256": payload_sha256,
        "payload_bytes": len(payload_bytes),
        "run_binding_sha256": run_binding_sha256,
        "start_receipt": start,
        "completion_receipt": completion,
        "chunks": tuple(chunks),
    }
    manifest_values["manifest_sha256"] = _digest(
        {
            key: (
                value.model_dump(mode="json")
                if isinstance(value, BaseModel)
                else [item.model_dump(mode="json") for item in value]
                if isinstance(value, tuple)
                else value
            )
            for key, value in manifest_values.items()
        }
    )
    manifest = V5ChunkedEvidenceManifest.model_validate(manifest_values)
    serialized = _canonical_bytes(manifest.model_dump(mode="json"))

    temporary = target_manifest.with_name(
        f".{target_manifest.name}.{os.getpid()}.{manifest.manifest_sha256[:12]}.tmp"
    )
    _ensure_absent(temporary, "temporary candidate manifest")
    _write_exclusive(temporary, serialized)
    try:
        os.link(temporary, target_manifest, follow_symlinks=False)
    except FileExistsError as error:
        raise FileExistsError("candidate manifest target already exists") from error
    finally:
        temporary.unlink(missing_ok=True)
    directory_descriptor = os.open(target_manifest.parent, os.O_RDONLY)
    try:
        os.fsync(directory_descriptor)
    finally:
        os.close(directory_descriptor)
    return manifest


def _regular_single_link(path: Path, label: str) -> os.stat_result:
    metadata = path.lstat()
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
        raise ValueError(f"{label} must be a single-link regular file")
    return metadata


def reconstruct_v5_chunked_evidence(
    *,
    target_manifest: Path,
    chunk_size_bytes: int,
    maximum_envelope_bytes: int,
    maximum_chunk_count: int,
    normal_git_blob_limit_bytes: int,
) -> bytes:
    """Verify exact storage topology and reconstruct the original opaque bytes."""
    target_manifest = target_manifest.absolute()
    metadata = _regular_single_link(target_manifest, "candidate manifest")
    if metadata.st_size > 1_048_576:
        raise ValueError("candidate manifest exceeds its byte bound")
    raw_manifest = target_manifest.read_bytes()
    try:
        document = json.loads(raw_manifest)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("candidate manifest is not JSON") from error
    if raw_manifest != _canonical_bytes(document):
        raise ValueError("candidate manifest is not canonical JSON")
    manifest = V5ChunkedEvidenceManifest.model_validate(document)
    if (
        manifest.chunk_size_bytes != chunk_size_bytes
        or manifest.maximum_envelope_bytes != maximum_envelope_bytes
        or manifest.maximum_chunk_count != maximum_chunk_count
        or manifest.normal_git_blob_limit_bytes != normal_git_blob_limit_bytes
    ):
        raise ValueError("candidate manifest storage limits differ")
    chunks_directory = target_manifest.parent / manifest.chunks_directory
    directory_metadata = chunks_directory.lstat()
    if not stat.S_ISDIR(directory_metadata.st_mode):
        raise ValueError("candidate chunk directory is not a real directory")
    observed_names = {path.name for path in chunks_directory.iterdir()}
    expected_names = {chunk.filename for chunk in manifest.chunks}
    if observed_names != expected_names:
        raise ValueError("candidate chunk set differs")
    content = bytearray()
    for chunk in manifest.chunks:
        path = chunks_directory / chunk.filename
        chunk_metadata = _regular_single_link(path, "candidate chunk")
        if chunk_metadata.st_size != chunk.bytes or chunk.bytes > chunk_size_bytes:
            raise ValueError("candidate chunk size differs")
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != chunk.sha256:
            raise ValueError("candidate chunk digest differs")
        content.extend(raw)
        if len(content) > maximum_envelope_bytes:
            raise ValueError("reconstructed evidence exceeds the maximum envelope")
    payload = bytes(content)
    if (
        len(payload) != manifest.payload_bytes
        or hashlib.sha256(payload).hexdigest() != manifest.payload_sha256
    ):
        raise ValueError("reconstructed evidence digest/size differs")
    return payload


__all__ = [
    "V5ChunkedEvidenceManifest",
    "V5EvidenceChunk",
    "V5LockedCompletionReceipt",
    "V5LockedStartReceipt",
    "publish_v5_chunked_evidence",
    "reconstruct_v5_chunked_evidence",
]
