"""Streaming manifest-last checkpoint storage for staged Sentinel v5 evidence."""

from __future__ import annotations

import hashlib
import json
import os
import stat
import time
import zlib
from collections.abc import Callable, Iterable, Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from pydantic.dataclasses import dataclass

from apar.evaluation.v5_kaggle_protocol import (
    V5KaggleEnvironmentBinding,
    V5KaggleResourceGates,
    V5KaggleStage,
    resolve_next_v5_kaggle_stage,
)

_MAX_MANIFEST_BYTES = 1_048_576
_MAX_OBSERVATION_BYTES = 4_194_304
_MAX_RECORD_HEADER_BYTES = 65_536
_MAX_RECORD_BYTES = 536_870_912
_DECOMPRESS_BLOCK_BYTES = 1_048_576
_EMPTY_LAYER_MARKER = "empty-layer.json"


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


def _empty_observational_layer_bytes() -> bytes:
    return _canonical_bytes(
        {
            "schema_version": "apar-sentinel-v5-kaggle-empty-checkpoint-layer/1",
            "layer": "observational",
            "record_count": 0,
            "record_stream_sha256": hashlib.sha256(b"").hexdigest(),
        }
    )


@dataclass(frozen=True, slots=True)
class V5CheckpointInput:
    kind: Annotated[str, Field(pattern=r"^[a-z][a-z0-9_]{0,63}$")]
    key: Annotated[
        str,
        Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.:@-]{0,511}$"),
    ]
    canonical_bytes: Annotated[bytes, Field(min_length=1, max_length=_MAX_RECORD_BYTES)]
    layer: Literal["deterministic", "observational"] = "deterministic"


class _FrozenModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class V5CheckpointObservation(_FrozenModel):
    schema_version: Literal["apar-sentinel-v5-kaggle-observation/1"]
    started_at_utc: str
    completed_at_utc: str
    wall_seconds: float = Field(gt=0.0)
    rss_samples_bytes: tuple[int, ...] = Field(min_length=1)
    host_available_samples_bytes: tuple[int, ...] = Field(min_length=1)
    peak_rss_bytes: int = Field(gt=0)
    environment: V5KaggleEnvironmentBinding
    checkpoint_output_bytes: int = Field(default=0, ge=0)
    pre_manifest_publication_seconds: float = Field(default=0.0, ge=0.0)
    observation_sha256: str = Field(default="", pattern=r"^[0-9a-f]{0}$|^[0-9a-f]{64}$")

    @field_validator("rss_samples_bytes", "host_available_samples_bytes")
    @classmethod
    def samples_are_positive(cls, value: tuple[int, ...]) -> tuple[int, ...]:
        if any(item <= 0 for item in value):
            raise ValueError("resource samples must be positive")
        return value

    @model_validator(mode="after")
    def observation_reconciles(self) -> Self:
        for label, value in (
            ("started", self.started_at_utc),
            ("completed", self.completed_at_utc),
        ):
            try:
                parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
            except ValueError as error:
                raise ValueError(f"observation {label} timestamp is invalid") from error
            if not value.endswith("Z") or parsed.tzinfo != UTC:
                raise ValueError(f"observation {label} timestamp is not UTC")
        if self.peak_rss_bytes != max(self.rss_samples_bytes):
            raise ValueError("observation peak RSS differs from retained samples")
        if len(self.rss_samples_bytes) != len(self.host_available_samples_bytes):
            raise ValueError("observation sample series lengths differ")
        if self.observation_sha256:
            expected = _digest(self.model_dump(mode="json", exclude={"observation_sha256"}))
            if self.observation_sha256 != expected:
                raise ValueError("checkpoint observation digest differs")
        return self


class V5CheckpointChunk(_FrozenModel):
    index: int = Field(ge=0)
    filename: str = Field(pattern=r"^part-[0-9]{4}\.bin$")
    bytes: int = Field(gt=0)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class V5CheckpointManifest(_FrozenModel):
    schema_version: Literal["apar-sentinel-v5-kaggle-checkpoint-manifest/1"]
    record_stream_schema_version: Literal["apar-sentinel-v5-kaggle-record-stream/1"]
    compression: Literal["gzip-zlib-level-9"]
    publication: Literal["chunks_observation_then_atomic_exclusive_manifest"]
    stage: V5KaggleStage
    run_binding_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    attempt_receipt_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    predecessor_stage: V5KaggleStage | None
    predecessor_manifest_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    predecessor_deterministic_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    record_count: int = Field(gt=0)
    uncompressed_record_bytes: int = Field(gt=0)
    record_stream_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    deterministic_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    observational_record_count: int = Field(ge=0)
    observational_uncompressed_record_bytes: int = Field(ge=0)
    observational_record_stream_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    observation_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    environment_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    max_stage_output_bytes: int = Field(gt=0)
    max_checkpoint_chunk_bytes: int = Field(gt=0)
    max_checkpoint_chunks: int = Field(gt=0)
    chunks: tuple[V5CheckpointChunk, ...] = Field(min_length=1)
    observational_chunks: tuple[V5CheckpointChunk, ...] = ()
    manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def manifest_reconciles(self) -> Self:
        if self.stage is V5KaggleStage.AUTHORIZE:
            if any(
                value is not None
                for value in (
                    self.predecessor_stage,
                    self.predecessor_manifest_sha256,
                    self.predecessor_deterministic_sha256,
                )
            ):
                raise ValueError("authorization checkpoint cannot have a predecessor")
        else:
            if (
                self.predecessor_stage is None
                or self.predecessor_manifest_sha256 is None
                or self.predecessor_deterministic_sha256 is None
                or resolve_next_v5_kaggle_stage(_ManifestStage(self.predecessor_stage))
                is not self.stage
            ):
                raise ValueError("checkpoint predecessor stage differs")
        if len(self.chunks) > self.max_checkpoint_chunks:
            raise ValueError("checkpoint chunk count exceeds bound")
        self._validate_chunks(self.chunks, label="deterministic")
        if len(self.observational_chunks) > self.max_checkpoint_chunks:
            raise ValueError("observational checkpoint chunk count exceeds bound")
        if self.observational_record_count == 0:
            if (
                self.observational_uncompressed_record_bytes != 0
                or self.observational_record_stream_sha256 != hashlib.sha256(b"").hexdigest()
                or self.observational_chunks
            ):
                raise ValueError("empty observational record stream differs")
        else:
            if self.observational_uncompressed_record_bytes <= 0:
                raise ValueError("observational record stream byte count differs")
            self._validate_chunks(self.observational_chunks, label="observational")
        if (
            sum(item.bytes for item in self.chunks)
            + sum(item.bytes for item in self.observational_chunks)
            > self.max_stage_output_bytes
        ):
            raise ValueError("checkpoint output exceeds stage bound")
        deterministic_document = {
            "schema_version": "apar-sentinel-v5-kaggle-deterministic-stage/1",
            "stage": self.stage,
            "run_binding_sha256": self.run_binding_sha256,
            "predecessor_deterministic_sha256": (self.predecessor_deterministic_sha256),
            "record_count": self.record_count,
            "uncompressed_record_bytes": self.uncompressed_record_bytes,
            "record_stream_sha256": self.record_stream_sha256,
        }
        if self.deterministic_sha256 != _digest(deterministic_document):
            raise ValueError("checkpoint deterministic digest differs")
        if self.manifest_sha256 != _digest(
            self.model_dump(mode="json", exclude={"manifest_sha256"})
        ):
            raise ValueError("checkpoint manifest digest differs")
        return self

    def _validate_chunks(self, chunks: tuple[V5CheckpointChunk, ...], *, label: str) -> None:
        if not chunks:
            raise ValueError(f"{label} checkpoint chunks are missing")
        expected_names = tuple(f"part-{index:04d}.bin" for index in range(len(chunks)))
        if tuple(item.index for item in chunks) != tuple(range(len(chunks))):
            raise ValueError(f"{label} checkpoint chunk indexes differ")
        if tuple(item.filename for item in chunks) != expected_names:
            raise ValueError(f"{label} checkpoint chunk names differ")
        if (
            any(item.bytes != self.max_checkpoint_chunk_bytes for item in chunks[:-1])
            or chunks[-1].bytes > self.max_checkpoint_chunk_bytes
        ):
            raise ValueError(f"{label} checkpoint chunk sizes differ")


class _ManifestStage:
    def __init__(self, stage: V5KaggleStage) -> None:
        self.stage = stage


class _ChunkWriter:
    def __init__(self, *, directory: Path, chunk_bytes: int, max_chunks: int) -> None:
        self._directory = directory
        self._chunk_bytes = chunk_bytes
        self._max_chunks = max_chunks
        self._buffer = bytearray()
        self._chunks: list[V5CheckpointChunk] = []

    def write(self, content: bytes) -> None:
        view = memoryview(content)
        while view:
            remaining = self._chunk_bytes - len(self._buffer)
            consumed = min(remaining, len(view))
            self._buffer.extend(view[:consumed])
            view = view[consumed:]
            if len(self._buffer) == self._chunk_bytes:
                self._flush_chunk()

    def finish(self, *, allow_empty: bool = False) -> tuple[V5CheckpointChunk, ...]:
        if self._buffer:
            self._flush_chunk()
        if not self._chunks and not allow_empty:
            raise ValueError("checkpoint compressed stream is empty")
        _fsync_directory(self._directory)
        return tuple(self._chunks)

    def _flush_chunk(self) -> None:
        if len(self._chunks) >= self._max_chunks:
            raise ValueError("checkpoint exceeds maximum chunk count")
        index = len(self._chunks)
        filename = f"part-{index:04d}.bin"
        content = bytes(self._buffer)
        self._buffer.clear()
        _write_exclusive(self._directory / filename, content)
        self._chunks.append(
            V5CheckpointChunk(
                index=index,
                filename=filename,
                bytes=len(content),
                sha256=hashlib.sha256(content).hexdigest(),
            )
        )


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
                raise OSError("exclusive checkpoint write made no progress")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _publish_no_replace(path: Path, content: bytes, digest: str) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{digest[:12]}.tmp")
    if os.path.lexists(temporary):
        raise FileExistsError("temporary checkpoint publication path exists")
    _write_exclusive(temporary, content)
    try:
        os.link(temporary, path, follow_symlinks=False)
    except FileExistsError as error:
        raise FileExistsError("checkpoint publication target already exists") from error
    finally:
        temporary.unlink(missing_ok=True)
    _fsync_directory(path.parent)


def _regular_single_link(path: Path, label: str) -> os.stat_result:
    metadata = path.lstat()
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
        raise ValueError(f"{label} must be a single-link regular file")
    return metadata


def _read_canonical_object(path: Path, *, label: str, max_bytes: int) -> dict[str, object]:
    metadata = _regular_single_link(path, label)
    if metadata.st_size > max_bytes:
        raise ValueError(f"{label} exceeds byte bound")
    raw = path.read_bytes()
    try:
        document = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is not JSON") from error
    if not isinstance(document, dict) or raw != _canonical_bytes(document):
        raise ValueError(f"{label} is not canonical JSON")
    return document


def _validate_publication_inputs(
    *,
    output_root: Path,
    stage: V5KaggleStage,
    run_binding_sha256: str,
    attempt_receipt_sha256: str,
    predecessor: V5CheckpointManifest | None,
    environment: V5KaggleEnvironmentBinding,
    observation: V5CheckpointObservation | None,
    limits: V5KaggleResourceGates,
) -> None:
    if os.path.lexists(output_root):
        raise FileExistsError("checkpoint output root already exists")
    if len(run_binding_sha256) != 64 or len(attempt_receipt_sha256) != 64:
        raise ValueError("checkpoint run or attempt digest is malformed")
    int(run_binding_sha256, 16)
    int(attempt_receipt_sha256, 16)
    if stage is V5KaggleStage.AUTHORIZE:
        if predecessor is not None:
            raise ValueError("authorization checkpoint cannot have a predecessor")
    elif (
        predecessor is None
        or predecessor.run_binding_sha256 != run_binding_sha256
        or predecessor.attempt_receipt_sha256 != attempt_receipt_sha256
        or resolve_next_v5_kaggle_stage(predecessor) is not stage
    ):
        raise ValueError("checkpoint predecessor lineage differs")
    if observation is not None:
        _validate_observation(
            observation=observation,
            environment=environment,
            limits=limits,
        )


def _validate_observation(
    *,
    observation: V5CheckpointObservation,
    environment: V5KaggleEnvironmentBinding,
    limits: V5KaggleResourceGates,
) -> None:
    if observation.environment != environment:
        raise ValueError("checkpoint observation environment differs")
    if (
        observation.wall_seconds > limits.max_stage_seconds
        or observation.peak_rss_bytes >= limits.max_peak_rss_bytes
    ):
        raise ValueError("checkpoint resource gate exceeded")


def publish_v5_checkpoint(
    *,
    output_root: Path,
    stage: V5KaggleStage,
    run_binding_sha256: str,
    attempt_receipt_sha256: str,
    predecessor: V5CheckpointManifest | None,
    records: Iterable[V5CheckpointInput],
    environment: V5KaggleEnvironmentBinding,
    observation: V5CheckpointObservation | None = None,
    observation_factory: Callable[[], V5CheckpointObservation] | None = None,
    limits: V5KaggleResourceGates,
) -> V5CheckpointManifest:
    """Stream records into bounded chunks and publish the manifest last."""
    output_root = output_root.absolute()
    if (observation is None) == (observation_factory is None):
        raise ValueError("checkpoint publication requires exactly one observation source")
    _validate_publication_inputs(
        output_root=output_root,
        stage=stage,
        run_binding_sha256=run_binding_sha256,
        attempt_receipt_sha256=attempt_receipt_sha256,
        predecessor=predecessor,
        environment=environment,
        observation=observation,
        limits=limits,
    )
    output_root.mkdir(mode=0o700, parents=True)
    chunks_directory = output_root / "chunks"
    chunks_directory.mkdir(mode=0o700)
    observational_chunks_directory = output_root / "observational-chunks"
    observational_chunks_directory.mkdir(mode=0o700)
    deterministic_writer = _ChunkWriter(
        directory=chunks_directory,
        chunk_bytes=limits.max_checkpoint_chunk_bytes,
        max_chunks=limits.max_checkpoint_chunks,
    )
    observational_writer = _ChunkWriter(
        directory=observational_chunks_directory,
        chunk_bytes=limits.max_checkpoint_chunk_bytes,
        max_chunks=limits.max_checkpoint_chunks,
    )
    deterministic_compressor = zlib.compressobj(level=9, wbits=31)
    observational_compressor = zlib.compressobj(level=9, wbits=31)
    deterministic_digest = hashlib.sha256()
    observational_digest = hashlib.sha256()
    deterministic_bytes = 0
    observational_bytes = 0
    deterministic_count = 0
    observational_count = 0
    seen: set[tuple[str, str, str]] = set()
    publication_started = time.perf_counter()
    for record in records:
        if type(record) is not V5CheckpointInput:
            raise TypeError("checkpoint record must be an exact V5CheckpointInput")
        identity = (record.layer, record.kind, record.key)
        if identity in seen:
            raise ValueError("checkpoint record identity is duplicated")
        seen.add(identity)
        header = _canonical_bytes(
            {
                "bytes": len(record.canonical_bytes),
                "key": record.key,
                "kind": record.kind,
            }
        )
        framed = len(header).to_bytes(8, "big") + header + record.canonical_bytes
        if record.layer == "deterministic":
            deterministic_digest.update(framed)
            deterministic_bytes += len(framed)
            deterministic_count += 1
            deterministic_writer.write(deterministic_compressor.compress(framed))
        else:
            observational_digest.update(framed)
            observational_bytes += len(framed)
            observational_count += 1
            observational_writer.write(observational_compressor.compress(framed))
    if deterministic_count == 0:
        raise ValueError("checkpoint record stream is empty")
    deterministic_writer.write(deterministic_compressor.flush())
    chunks = deterministic_writer.finish()
    if observational_count:
        observational_writer.write(observational_compressor.flush())
    observational_chunks = observational_writer.finish(allow_empty=True)
    if not observational_count:
        _write_exclusive(
            observational_chunks_directory / _EMPTY_LAYER_MARKER,
            _empty_observational_layer_bytes(),
        )
        _fsync_directory(observational_chunks_directory)
    compressed_bytes = sum(item.bytes for item in chunks) + sum(
        item.bytes for item in observational_chunks
    )
    elapsed = time.perf_counter() - publication_started
    if observation is not None:
        durable_source = observation
    elif observation_factory is not None:
        durable_source = observation_factory()
    else:  # guarded before any output directory is created
        raise AssertionError("checkpoint observation source disappeared")
    _validate_observation(
        observation=durable_source,
        environment=environment,
        limits=limits,
    )
    observation_values = durable_source.model_dump(mode="json", exclude={"observation_sha256"})
    observation_values["checkpoint_output_bytes"] = compressed_bytes
    observation_values["pre_manifest_publication_seconds"] = elapsed
    observation_values["observation_sha256"] = _digest(observation_values)
    durable_observation = V5CheckpointObservation.model_validate(observation_values)
    observation_bytes = _canonical_bytes(durable_observation.model_dump(mode="json"))
    if len(observation_bytes) > _MAX_OBSERVATION_BYTES:
        raise ValueError("checkpoint observation exceeds byte bound")
    _write_exclusive(output_root / "observational.json", observation_bytes)

    deterministic_values = {
        "schema_version": "apar-sentinel-v5-kaggle-deterministic-stage/1",
        "stage": stage,
        "run_binding_sha256": run_binding_sha256,
        "predecessor_deterministic_sha256": (
            None if predecessor is None else predecessor.deterministic_sha256
        ),
        "record_count": deterministic_count,
        "uncompressed_record_bytes": deterministic_bytes,
        "record_stream_sha256": deterministic_digest.hexdigest(),
    }
    manifest_values: dict[str, object] = {
        "schema_version": "apar-sentinel-v5-kaggle-checkpoint-manifest/1",
        "record_stream_schema_version": "apar-sentinel-v5-kaggle-record-stream/1",
        "compression": "gzip-zlib-level-9",
        "publication": "chunks_observation_then_atomic_exclusive_manifest",
        "stage": stage,
        "run_binding_sha256": run_binding_sha256,
        "attempt_receipt_sha256": attempt_receipt_sha256,
        "predecessor_stage": None if predecessor is None else predecessor.stage,
        "predecessor_manifest_sha256": (
            None if predecessor is None else predecessor.manifest_sha256
        ),
        "predecessor_deterministic_sha256": (
            None if predecessor is None else predecessor.deterministic_sha256
        ),
        "record_count": deterministic_count,
        "uncompressed_record_bytes": deterministic_bytes,
        "record_stream_sha256": deterministic_digest.hexdigest(),
        "deterministic_sha256": _digest(deterministic_values),
        "observational_record_count": observational_count,
        "observational_uncompressed_record_bytes": observational_bytes,
        "observational_record_stream_sha256": observational_digest.hexdigest(),
        "observation_sha256": durable_observation.observation_sha256,
        "environment_sha256": environment.environment_sha256,
        "max_stage_output_bytes": limits.max_stage_output_bytes,
        "max_checkpoint_chunk_bytes": limits.max_checkpoint_chunk_bytes,
        "max_checkpoint_chunks": limits.max_checkpoint_chunks,
        "chunks": tuple(item.model_dump(mode="json") for item in chunks),
        "observational_chunks": tuple(
            item.model_dump(mode="json") for item in observational_chunks
        ),
    }
    manifest_values["manifest_sha256"] = _digest(manifest_values)
    manifest = V5CheckpointManifest.model_validate(manifest_values)
    manifest_bytes = _canonical_bytes(manifest.model_dump(mode="json"))
    total_output = compressed_bytes + len(observation_bytes) + len(manifest_bytes)
    if total_output > limits.max_stage_output_bytes:
        raise ValueError("checkpoint resource gate exceeded: output bytes")
    _publish_no_replace(
        output_root / "checkpoint.manifest.json",
        manifest_bytes,
        manifest.manifest_sha256,
    )
    return manifest


def _validate_storage_topology(
    *,
    output_root: Path,
    manifest: V5CheckpointManifest,
    limits: V5KaggleResourceGates,
) -> V5CheckpointObservation:
    if (
        manifest.max_stage_output_bytes != limits.max_stage_output_bytes
        or manifest.max_checkpoint_chunk_bytes != limits.max_checkpoint_chunk_bytes
        or manifest.max_checkpoint_chunks != limits.max_checkpoint_chunks
    ):
        raise ValueError("checkpoint storage limits differ")
    root_metadata = output_root.lstat()
    if not stat.S_ISDIR(root_metadata.st_mode):
        raise ValueError("checkpoint root is not a real directory")
    chunks_directory = output_root / "chunks"
    chunks_metadata = chunks_directory.lstat()
    if not stat.S_ISDIR(chunks_metadata.st_mode):
        raise ValueError("checkpoint chunk root is not a real directory")
    observed_names = {item.name for item in chunks_directory.iterdir()}
    expected_names = {item.filename for item in manifest.chunks}
    if observed_names != expected_names:
        raise ValueError("checkpoint chunk set differs")
    observational_chunks_directory = output_root / "observational-chunks"
    observational_chunks_metadata = observational_chunks_directory.lstat()
    if not stat.S_ISDIR(observational_chunks_metadata.st_mode):
        raise ValueError("observational checkpoint chunk root is not a real directory")
    observed_observational_names = {item.name for item in observational_chunks_directory.iterdir()}
    expected_observational_names = (
        {_EMPTY_LAYER_MARKER}
        if manifest.observational_record_count == 0
        else {item.filename for item in manifest.observational_chunks}
    )
    if observed_observational_names != expected_observational_names:
        raise ValueError("observational checkpoint chunk set differs")
    if manifest.observational_record_count == 0:
        marker_path = observational_chunks_directory / _EMPTY_LAYER_MARKER
        _regular_single_link(marker_path, "empty observational checkpoint marker")
        if marker_path.read_bytes() != _empty_observational_layer_bytes():
            raise ValueError("empty observational checkpoint marker differs")
    observation_document = _read_canonical_object(
        output_root / "observational.json",
        label="checkpoint observation",
        max_bytes=_MAX_OBSERVATION_BYTES,
    )
    observation = V5CheckpointObservation.model_validate(observation_document)
    if not observation.observation_sha256:
        raise ValueError("checkpoint observation digest is absent")
    if observation.observation_sha256 != manifest.observation_sha256:
        raise ValueError("checkpoint observation binding differs")
    if observation.environment.environment_sha256 != manifest.environment_sha256:
        raise ValueError("checkpoint environment binding differs")
    if (
        observation.wall_seconds > limits.max_stage_seconds
        or observation.peak_rss_bytes >= limits.max_peak_rss_bytes
    ):
        raise ValueError("checkpoint resource gate exceeded")
    return observation


def read_v5_checkpoint_manifest(
    *, output_root: Path, limits: V5KaggleResourceGates
) -> V5CheckpointManifest:
    """Read and validate a manifest and all non-record checkpoint topology."""
    output_root = output_root.absolute()
    document = _read_canonical_object(
        output_root / "checkpoint.manifest.json",
        label="checkpoint manifest",
        max_bytes=_MAX_MANIFEST_BYTES,
    )
    manifest = V5CheckpointManifest.model_validate(document)
    _validate_storage_topology(
        output_root=output_root,
        manifest=manifest,
        limits=limits,
    )
    return manifest


def read_v5_checkpoint_observation(
    *, output_root: Path, limits: V5KaggleResourceGates
) -> V5CheckpointObservation:
    """Read the authenticated observational layer for one checkpoint."""
    manifest = read_v5_checkpoint_manifest(output_root=output_root, limits=limits)
    return _validate_storage_topology(
        output_root=output_root.absolute(), manifest=manifest, limits=limits
    )


def _iter_compressed_payload(
    *,
    output_root: Path,
    chunks_directory: str,
    chunks: tuple[V5CheckpointChunk, ...],
    label: str,
) -> Iterator[bytes]:
    for chunk in chunks:
        chunk_path = output_root / chunks_directory / chunk.filename
        metadata = _regular_single_link(chunk_path, f"{label} checkpoint chunk")
        if metadata.st_size != chunk.bytes:
            raise ValueError(f"{label} checkpoint chunk size differs")
        content = chunk_path.read_bytes()
        if hashlib.sha256(content).hexdigest() != chunk.sha256:
            raise ValueError(f"{label} checkpoint chunk digest differs")
        yield content


def _iter_decompressed_payload(
    *,
    output_root: Path,
    chunks_directory: str,
    chunks: tuple[V5CheckpointChunk, ...],
    label: str,
) -> Iterator[bytes]:
    decompressor = zlib.decompressobj(wbits=31)
    for compressed in _iter_compressed_payload(
        output_root=output_root,
        chunks_directory=chunks_directory,
        chunks=chunks,
        label=label,
    ):
        pending = compressed
        while pending:
            block = decompressor.decompress(pending, _DECOMPRESS_BLOCK_BYTES)
            pending = decompressor.unconsumed_tail
            if block:
                yield block
            if not pending:
                break
    tail = decompressor.flush()
    if tail:
        yield tail
    if not decompressor.eof or decompressor.unused_data:
        raise ValueError(f"{label} checkpoint compressed stream is incomplete or has trailing data")


def _iter_v5_checkpoint_record_layer(
    *,
    output_root: Path,
    limits: V5KaggleResourceGates,
    layer: Literal["deterministic", "observational"],
) -> Iterator[V5CheckpointInput]:
    output_root = output_root.absolute()
    manifest = read_v5_checkpoint_manifest(output_root=output_root, limits=limits)
    if layer == "deterministic":
        chunks_directory = "chunks"
        chunks = manifest.chunks
        expected_count = manifest.record_count
        expected_bytes = manifest.uncompressed_record_bytes
        expected_digest = manifest.record_stream_sha256
    else:
        chunks_directory = "observational-chunks"
        chunks = manifest.observational_chunks
        expected_count = manifest.observational_record_count
        expected_bytes = manifest.observational_uncompressed_record_bytes
        expected_digest = manifest.observational_record_stream_sha256
    if expected_count == 0:
        if chunks or expected_bytes != 0 or expected_digest != hashlib.sha256(b"").hexdigest():
            raise ValueError(f"{layer} checkpoint empty stream differs")
        return
    buffer = bytearray()
    stream_digest = hashlib.sha256()
    stream_bytes = 0
    count = 0
    expected_payload_bytes: int | None = None
    current_header: dict[str, object] | None = None
    seen: set[tuple[str, str]] = set()
    for block in _iter_decompressed_payload(
        output_root=output_root,
        chunks_directory=chunks_directory,
        chunks=chunks,
        label=layer,
    ):
        stream_digest.update(block)
        stream_bytes += len(block)
        if stream_bytes > expected_bytes:
            raise ValueError(f"{layer} checkpoint record stream exceeds declared bytes")
        buffer.extend(block)
        while True:
            if current_header is None:
                if len(buffer) < 8:
                    break
                header_bytes = int.from_bytes(buffer[:8], "big")
                if not 0 < header_bytes <= _MAX_RECORD_HEADER_BYTES:
                    raise ValueError("checkpoint record header length differs")
                if len(buffer) < 8 + header_bytes:
                    break
                raw_header = bytes(buffer[8 : 8 + header_bytes])
                del buffer[: 8 + header_bytes]
                try:
                    parsed_header = json.loads(raw_header)
                except (UnicodeDecodeError, json.JSONDecodeError) as error:
                    raise ValueError("checkpoint record header is not JSON") from error
                if (
                    not isinstance(parsed_header, dict)
                    or raw_header != _canonical_bytes(parsed_header)
                    or set(parsed_header) != {"bytes", "key", "kind"}
                ):
                    raise ValueError("checkpoint record header differs")
                payload_bytes = parsed_header["bytes"]
                kind = parsed_header["kind"]
                key = parsed_header["key"]
                if type(payload_bytes) is not int or not 0 < payload_bytes <= _MAX_RECORD_BYTES:
                    raise ValueError("checkpoint record payload length differs")
                if type(kind) is not str or type(key) is not str:
                    raise ValueError("checkpoint record identity types differ")
                current_header = parsed_header
                expected_payload_bytes = payload_bytes
            if expected_payload_bytes is None or len(buffer) < expected_payload_bytes:
                break
            payload = bytes(buffer[:expected_payload_bytes])
            del buffer[:expected_payload_bytes]
            assert current_header is not None
            kind = current_header["kind"]
            key = current_header["key"]
            if type(kind) is not str or type(key) is not str:
                raise ValueError("checkpoint record identity types differ")
            record = V5CheckpointInput(
                kind=kind,
                key=key,
                canonical_bytes=payload,
                layer=layer,
            )
            identity = (record.kind, record.key)
            if identity in seen:
                raise ValueError("checkpoint record identity is duplicated")
            seen.add(identity)
            yield record
            count += 1
            current_header = None
            expected_payload_bytes = None
    if buffer or current_header is not None or expected_payload_bytes is not None:
        raise ValueError("checkpoint record stream ends mid-record")
    if (
        count != expected_count
        or stream_bytes != expected_bytes
        or stream_digest.hexdigest() != expected_digest
    ):
        raise ValueError(f"{layer} checkpoint record stream digest or support differs")


def iter_v5_checkpoint_records(
    *, output_root: Path, limits: V5KaggleResourceGates
) -> Iterator[V5CheckpointInput]:
    """Verify and stream deterministic records from one completed checkpoint."""
    yield from _iter_v5_checkpoint_record_layer(
        output_root=output_root,
        limits=limits,
        layer="deterministic",
    )


def iter_v5_checkpoint_observational_records(
    *, output_root: Path, limits: V5KaggleResourceGates
) -> Iterator[V5CheckpointInput]:
    """Verify and stream the exact authenticated observational record layer."""
    yield from _iter_v5_checkpoint_record_layer(
        output_root=output_root,
        limits=limits,
        layer="observational",
    )


__all__ = [
    "V5CheckpointChunk",
    "V5CheckpointInput",
    "V5CheckpointManifest",
    "V5CheckpointObservation",
    "iter_v5_checkpoint_observational_records",
    "iter_v5_checkpoint_records",
    "publish_v5_checkpoint",
    "read_v5_checkpoint_manifest",
    "read_v5_checkpoint_observation",
]
