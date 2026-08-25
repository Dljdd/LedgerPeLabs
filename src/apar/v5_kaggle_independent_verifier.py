"""Offline verifier for staged Sentinel v5 Kaggle checkpoint evidence.

This module deliberately owns its storage reader and uses only the earlier
independent evidence verifier for semantic replay.  It never imports production
execution, checkpoint, feature, control, metric, simulator, rail, ledger, or
TrustVerifier implementations.
"""

from __future__ import annotations

import hashlib
import json
import math
import stat
import struct
import zlib
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, NoReturn, Self, cast

from pydantic import BaseModel, ConfigDict, Field, model_validator

from apar import v5_independent_verifier as semantic

_STAGES = (
    "00_authorize",
    "10_corpus",
    "20_features",
    "30_arms",
    "40_label_shuffle",
    "50_invariance_controls",
    "60_single_class_controls",
    "70_metrics",
    "80_finalize",
)
_ARMS = (
    "rules_only",
    "ensemble_no_graph",
    "ensemble_with_graph",
    "full_sentinel",
)
_MANIFEST_FIELDS = {
    "schema_version",
    "record_stream_schema_version",
    "compression",
    "publication",
    "stage",
    "run_binding_sha256",
    "attempt_receipt_sha256",
    "predecessor_stage",
    "predecessor_manifest_sha256",
    "predecessor_deterministic_sha256",
    "record_count",
    "uncompressed_record_bytes",
    "record_stream_sha256",
    "deterministic_sha256",
    "observational_record_count",
    "observational_uncompressed_record_bytes",
    "observational_record_stream_sha256",
    "observation_sha256",
    "environment_sha256",
    "max_stage_output_bytes",
    "max_checkpoint_chunk_bytes",
    "max_checkpoint_chunks",
    "chunks",
    "observational_chunks",
    "manifest_sha256",
}
_OBSERVATION_FIELDS = {
    "schema_version",
    "started_at_utc",
    "completed_at_utc",
    "wall_seconds",
    "rss_samples_bytes",
    "host_available_samples_bytes",
    "peak_rss_bytes",
    "environment",
    "checkpoint_output_bytes",
    "pre_manifest_publication_seconds",
    "observation_sha256",
}
_ENVIRONMENT_FIELDS = {
    "schema_version",
    "provider",
    "image",
    "image_sha256",
    "python_version",
    "architecture",
    "cpu_count",
    "dependency_manifest_sha256",
    "source_archive_sha256",
    "notebook_sha256",
    "internet_enabled",
    "accelerator",
    "file_fsync_supported",
    "directory_fsync_supported",
    "hardlink_no_replace_supported",
    "environment_sha256",
}
_AUTHORIZATION_FIELDS = {
    "schema_version",
    "stage",
    "mode",
    "profile",
    "development_test_seed",
    "repeatable",
    "authorization_required",
    "run_binding_sha256",
    "attempt_receipt_sha256",
    "execution_manifest_sha256",
    "protocol_sha256",
    "source_bindings",
    "recovery",
    "support_plan",
    "resources",
    "checkpoint",
}
_MAX_MANIFEST_BYTES = 1_048_576
_MAX_OBSERVATION_BYTES = 4_194_304
_MAX_RECORD_HEADER_BYTES = 65_536
_MAX_RECORD_BYTES = 1_073_741_824
_READ_BLOCK_BYTES = 1_048_576


class V5KaggleIndependentVerificationError(ValueError):
    """A staged artifact failed independent replay."""


def _fail(message: str) -> NoReturn:
    raise V5KaggleIndependentVerificationError(message)


def _canonical_bytes(document: object) -> bytes:
    return json.dumps(
        document,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode()


def _digest(document: object) -> str:
    return hashlib.sha256(_canonical_bytes(document)).hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and value == value.lower()
        and all(character in "0123456789abcdef" for character in value)
    )


def _mapping(value: object, label: str) -> dict[str, Any]:
    if type(value) is not dict or any(type(key) is not str for key in value):
        _fail(f"{label} must be an object")
    return cast(dict[str, Any], value)


def _sequence(value: object, label: str) -> list[Any]:
    if type(value) is not list:
        _fail(f"{label} must be an array")
    return value


def _exact(value: Mapping[str, Any], fields: set[str], label: str) -> None:
    if set(value) != fields:
        _fail(f"{label} schema differs")


def _read_canonical_file(
    path: Path, *, limit: int, label: str, require_canonical: bool = True
) -> dict[str, Any]:
    try:
        metadata = path.lstat()
    except OSError as error:
        raise V5KaggleIndependentVerificationError(f"{label} is missing") from error
    if (
        not stat.S_ISREG(metadata.st_mode)
        or metadata.st_nlink != 1
        or metadata.st_size <= 0
        or metadata.st_size > limit
    ):
        _fail(f"{label} topology or size differs")
    raw = path.read_bytes()
    try:
        document = _mapping(json.loads(raw), label)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise V5KaggleIndependentVerificationError(f"{label} is not JSON") from error
    if require_canonical and raw != _canonical_bytes(document):
        _fail(f"{label} is not canonical JSON")
    return document


@dataclass(frozen=True, slots=True)
class _Record:
    layer: Literal["deterministic", "observational"]
    kind: str
    key: str
    payload: bytes


class _RecordStream:
    def __init__(
        self,
        *,
        root: Path,
        chunks: Sequence[Mapping[str, Any]],
        layer: Literal["deterministic", "observational"],
        max_chunks: int,
        max_chunk_bytes: int,
    ) -> None:
        self.root = root
        self.chunks = tuple(chunks)
        self.layer = layer
        self.max_chunks = max_chunks
        self.max_chunk_bytes = max_chunk_bytes
        self.count = 0
        self.uncompressed_bytes = 0
        self.stream_sha256 = hashlib.sha256(b"").hexdigest()

    def _decompressed_blocks(self) -> Iterator[bytes]:
        directory_name = "chunks" if self.layer == "deterministic" else "observational-chunks"
        directory = self.root / directory_name
        try:
            metadata = directory.lstat()
        except OSError as error:
            raise V5KaggleIndependentVerificationError(
                f"{self.layer} chunk directory is missing"
            ) from error
        if not stat.S_ISDIR(metadata.st_mode):
            _fail(f"{self.layer} chunk directory topology differs")
        if len(self.chunks) > self.max_chunks:
            _fail(f"{self.layer} chunk count exceeds bound")
        expected_names = {str(item.get("filename")) for item in self.chunks}
        if {item.name for item in directory.iterdir()} != expected_names:
            _fail(f"{self.layer} chunk set differs")
        if not self.chunks:
            return
        decompressor = zlib.decompressobj(wbits=31)
        for expected_index, descriptor in enumerate(self.chunks):
            chunk = _mapping(descriptor, f"{self.layer} chunk descriptor")
            _exact(chunk, {"index", "filename", "bytes", "sha256"}, "chunk descriptor")
            filename = f"part-{expected_index:04d}.bin"
            if chunk["index"] != expected_index or chunk["filename"] != filename:
                _fail(f"{self.layer} chunk order differs")
            path = directory / filename
            metadata = path.lstat()
            if (
                not stat.S_ISREG(metadata.st_mode)
                or metadata.st_nlink != 1
                or metadata.st_size != chunk["bytes"]
                or not 0 < metadata.st_size <= self.max_chunk_bytes
            ):
                _fail(f"{self.layer} chunk topology or size differs")
            digest = hashlib.sha256()
            with path.open("rb") as handle:
                while block := handle.read(_READ_BLOCK_BYTES):
                    digest.update(block)
                    try:
                        expanded = decompressor.decompress(block)
                    except zlib.error as error:
                        raise V5KaggleIndependentVerificationError(
                            f"{self.layer} compressed stream is invalid"
                        ) from error
                    if expanded:
                        yield expanded
            if digest.hexdigest() != chunk["sha256"]:
                _fail(f"{self.layer} chunk digest differs")
        tail = decompressor.flush()
        if tail:
            yield tail
        if not decompressor.eof or decompressor.unused_data or decompressor.unconsumed_tail:
            _fail(f"{self.layer} compressed stream is incomplete or has trailing bytes")

    def __iter__(self) -> Iterator[_Record]:
        buffer = bytearray()
        stream_digest = hashlib.sha256()
        count = 0
        total = 0
        blocks = self._decompressed_blocks()

        def fill(size: int) -> bool:
            while len(buffer) < size:
                try:
                    buffer.extend(next(blocks))
                except StopIteration:
                    return False
            return True

        while fill(8):
            header_size_bytes = bytes(buffer[:8])
            del buffer[:8]
            header_size = int.from_bytes(header_size_bytes, "big")
            if not 1 <= header_size <= _MAX_RECORD_HEADER_BYTES or not fill(header_size):
                _fail(f"{self.layer} record header is truncated or oversized")
            header_bytes = bytes(buffer[:header_size])
            del buffer[:header_size]
            try:
                header = _mapping(json.loads(header_bytes), "record header")
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise V5KaggleIndependentVerificationError(
                    f"{self.layer} record header is invalid"
                ) from error
            if header_bytes != _canonical_bytes(header) or set(header) != {
                "bytes",
                "key",
                "kind",
            }:
                _fail(f"{self.layer} record header schema differs")
            payload_size = header["bytes"]
            if type(payload_size) is not int or not 1 <= payload_size <= _MAX_RECORD_BYTES:
                _fail(f"{self.layer} record size differs")
            if not fill(payload_size):
                _fail(f"{self.layer} record payload is truncated")
            payload = bytes(buffer[:payload_size])
            del buffer[:payload_size]
            frame = header_size_bytes + header_bytes + payload
            stream_digest.update(frame)
            total += len(frame)
            count += 1
            kind = header["kind"]
            key = header["key"]
            if type(kind) is not str or type(key) is not str:
                _fail(f"{self.layer} record identity differs")
            yield _Record(layer=self.layer, kind=kind, key=key, payload=payload)
        for extra in blocks:
            buffer.extend(extra)
        if buffer:
            _fail(f"{self.layer} record stream has trailing bytes")
        self.count = count
        self.uncompressed_bytes = total
        self.stream_sha256 = stream_digest.hexdigest()


@dataclass(frozen=True, slots=True)
class _VerifiedCheckpoint:
    root: Path
    manifest: dict[str, Any]
    observation: dict[str, Any]
    deterministic_records: tuple[_Record, ...]
    observational_records: tuple[_Record, ...]


def _read_checkpoint(root: Path) -> _VerifiedCheckpoint:
    try:
        metadata = root.lstat()
    except OSError as error:
        raise V5KaggleIndependentVerificationError("checkpoint root is missing") from error
    if not stat.S_ISDIR(metadata.st_mode):
        _fail("checkpoint root topology differs")
    manifest = _read_canonical_file(
        root / "checkpoint.manifest.json",
        limit=_MAX_MANIFEST_BYTES,
        label="checkpoint manifest",
    )
    _exact(manifest, _MANIFEST_FIELDS, "checkpoint manifest")
    if (
        manifest["schema_version"] != "apar-sentinel-v5-kaggle-checkpoint-manifest/1"
        or manifest["record_stream_schema_version"] != "apar-sentinel-v5-kaggle-record-stream/1"
        or manifest["compression"] != "gzip-zlib-level-9"
        or manifest["publication"] != "chunks_observation_then_atomic_exclusive_manifest"
    ):
        _fail("checkpoint manifest frozen schema differs")
    if manifest["manifest_sha256"] != _digest(
        {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    ):
        _fail("checkpoint manifest digest differs")
    max_output = manifest["max_stage_output_bytes"]
    max_chunk = manifest["max_checkpoint_chunk_bytes"]
    max_chunks = manifest["max_checkpoint_chunks"]
    if (max_output, max_chunk, max_chunks) != (10_000_000_000, 67_108_864, 160):
        _fail("checkpoint resource limits differ")

    retained_deterministic: list[_Record] = []
    retained_observational: list[_Record] = []
    retained_kinds = {
        "00_authorize": {"authorization"},
        "10_corpus": {
            "corpus_header",
            "partition_header",
            "decision_row",
            "execution_manifest",
        },
        "20_features": {
            "feature_header",
            "prepared_partition",
            "feature_matrix",
            "labels",
            "amounts",
            "trust_failures",
            "feature_batch",
            "training_evidence",
        },
        "30_arms": {"arm_header", "arm_result", "arm_latency"},
        "40_label_shuffle": {"control_group", "control_observation"},
        "50_invariance_controls": {"control_group", "control_observation"},
        "60_single_class_controls": {"control_group", "control_observation"},
        "70_metrics": {"metric_evidence", "metric_observation"},
        "80_finalize": {"final_core", "final_payload"},
    }.get(str(manifest["stage"]))
    if retained_kinds is None:
        _fail("checkpoint stage is unknown")
    streams = (
        _RecordStream(
            root=root,
            chunks=_sequence(manifest["chunks"], "deterministic chunks"),
            layer="deterministic",
            max_chunks=max_chunks,
            max_chunk_bytes=max_chunk,
        ),
        _RecordStream(
            root=root,
            chunks=_sequence(manifest["observational_chunks"], "observational chunks"),
            layer="observational",
            max_chunks=max_chunks,
            max_chunk_bytes=max_chunk,
        ),
    )
    for stream, retained in zip(
        streams, (retained_deterministic, retained_observational), strict=True
    ):
        for record in stream:
            if record.kind in retained_kinds:
                retained.append(record)
        prefix = "" if stream.layer == "deterministic" else "observational_"
        if (
            stream.count != manifest[f"{prefix}record_count"]
            or stream.uncompressed_bytes != manifest[f"{prefix}uncompressed_record_bytes"]
            or stream.stream_sha256 != manifest[f"{prefix}record_stream_sha256"]
        ):
            _fail(f"{stream.layer} record stream binding differs")
    deterministic_values = {
        "schema_version": "apar-sentinel-v5-kaggle-deterministic-stage/1",
        "stage": manifest["stage"],
        "run_binding_sha256": manifest["run_binding_sha256"],
        "predecessor_deterministic_sha256": manifest["predecessor_deterministic_sha256"],
        "record_count": manifest["record_count"],
        "uncompressed_record_bytes": manifest["uncompressed_record_bytes"],
        "record_stream_sha256": manifest["record_stream_sha256"],
    }
    if manifest["deterministic_sha256"] != _digest(deterministic_values):
        _fail("checkpoint deterministic stage digest differs")
    observation = _read_canonical_file(
        root / "observational.json",
        limit=_MAX_OBSERVATION_BYTES,
        label="checkpoint observation",
    )
    _exact(observation, _OBSERVATION_FIELDS, "checkpoint observation")
    environment = _mapping(observation["environment"], "checkpoint environment")
    _exact(environment, _ENVIRONMENT_FIELDS, "checkpoint environment")
    if environment["environment_sha256"] != _digest(
        {key: value for key, value in environment.items() if key != "environment_sha256"}
    ):
        _fail("checkpoint environment digest differs")
    if observation["observation_sha256"] != _digest(
        {key: value for key, value in observation.items() if key != "observation_sha256"}
    ):
        _fail("checkpoint observation digest differs")
    rss = _sequence(observation["rss_samples_bytes"], "RSS samples")
    available = _sequence(observation["host_available_samples_bytes"], "available-memory samples")
    if (
        not rss
        or len(rss) != len(available)
        or observation["peak_rss_bytes"] != max(rss)
        or observation["peak_rss_bytes"] >= 19_327_352_832
        or not 0 < observation["wall_seconds"] <= 21_600
    ):
        _fail("checkpoint resource observation differs or exceeds gate")
    if (
        manifest["observation_sha256"] != observation["observation_sha256"]
        or manifest["environment_sha256"] != environment["environment_sha256"]
    ):
        _fail("checkpoint observation/environment manifest binding differs")
    return _VerifiedCheckpoint(
        root=root,
        manifest=manifest,
        observation=observation,
        deterministic_records=tuple(retained_deterministic),
        observational_records=tuple(retained_observational),
    )


def _protocol_and_run_binding(
    *, root: Path, expected_mode: str
) -> tuple[dict[str, Any], dict[str, Any], str]:
    protocol_path = root / "config/defense/defense-v5-kaggle-recovery.json"
    protocol = _read_canonical_file(
        protocol_path,
        limit=1_048_576,
        label="Kaggle recovery protocol",
        require_canonical=False,
    )
    if "protocol_sha256" in protocol:
        _fail("Kaggle recovery source carries a self-referential digest")
    protocol_sha256 = _digest(protocol)
    modes = {
        "kaggle_capacity_validation": "capacity",
        "kaggle_locked_successor": "locked",
    }
    slot = modes.get(expected_mode)
    if slot is None:
        _fail("expected Kaggle mode is unknown")
    run = _mapping(protocol[slot], "Kaggle run mode")
    expected_seed = 404 if expected_mode == "kaggle_capacity_validation" else 2404
    if (
        run.get("mode") != expected_mode
        or run.get("profile") != "production"
        or run.get("development_test_seed") != expected_seed
    ):
        _fail("Kaggle mode/profile/seed differs")
    sources = _mapping(protocol["source_bindings"], "Kaggle source bindings")
    for path_field, digest_field in (
        ("base_protocol_path", "base_protocol_sha256"),
        ("evidence_protocol_path", "evidence_protocol_sha256"),
        ("arm_protocol_path", "arm_protocol_sha256"),
        ("feature_catalog_path", "feature_catalog_sha256"),
    ):
        path = root / str(sources[path_field])
        if hashlib.sha256(path.read_bytes()).hexdigest() != sources[digest_field]:
            _fail("Kaggle source file digest differs")
    run_binding = _digest(
        {
            "schema_version": "apar-sentinel-v5-kaggle-run-binding/1",
            "protocol_sha256": protocol_sha256,
            "run": run,
            "source_bindings": sources,
            "recovery": protocol["recovery"],
        }
    )
    return protocol, run, run_binding


def _unpack_payload_semantics(
    *, payload: dict[str, Any], root: Path, enforce_production_support: bool
) -> tuple[str, str, str]:
    proxy = dict(payload)
    proxy["safe_seed"] = 404
    protocol = semantic._verify_protocol(proxy, root)  # noqa: SLF001
    if enforce_production_support:
        base = _mapping(
            json.loads((root / str(protocol["base_protocol_path"])).read_bytes()),
            "production base protocol",
        )
        expected_support = semantic._independent_locked_support_plan(base)  # noqa: SLF001
        expected_support["mode"] = payload["mode"]
        expected_support["support_plan_sha256"] = _digest(
            {key: value for key, value in expected_support.items() if key != "support_plan_sha256"}
        )
        if payload["support_plan"] != expected_support:
            _fail("staged production support plan differs")
    artifact_pool: dict[str, dict[str, Any]] = {}
    for item in _sequence(payload["execution_artifact_pool"], "execution artifact pool"):
        artifact = semantic._unpack_document(  # noqa: SLF001
            item,
            expected_kind="execution_artifact",
            max_uncompressed_bytes=16_777_216,
        )
        evidence_id = artifact.get("evidence_sha256")
        if type(evidence_id) is not str or evidence_id in artifact_pool:
            _fail("staged execution artifact identifiers differ")
        artifact_pool[evidence_id] = artifact
    results: list[dict[str, Any]] = []
    retained_results: list[dict[str, Any]] = []
    used: set[str] = set()
    for item in _sequence(payload["arm_results"], "staged arm results"):
        retained = semantic._unpack_document(item, expected_kind="arm_result")  # noqa: SLF001
        expanded, references = semantic._expand_retained_result(  # noqa: SLF001
            retained, artifact_pool
        )
        retained_results.append(retained)
        results.append(expanded)
        used.update(references)
    if used != set(artifact_pool):
        _fail("staged artifact pool contains unused or missing support")
    complete = [
        semantic._unpack_document(item, expected_kind="complete_metrics")  # noqa: SLF001
        for item in _sequence(payload["complete_metrics"], "complete metrics")
    ]
    if [item["arm"] for item in results] != list(_ARMS) or [
        item["arm"] for item in complete
    ] != list(_ARMS):
        _fail("staged payload arm order differs")
    verified_rows: list[list[dict[str, Any]]] = []
    verified_manifests: list[dict[str, dict[str, Any]]] = []
    for result in results:
        rows, _artifacts, manifests = semantic._verify_arm_result(  # noqa: SLF001
            result, str(payload["catalog_sha256"]), protocol
        )
        verified_rows.append(rows)
        verified_manifests.append(manifests)
    reference_support = [row["support"] for row in verified_rows[0]]
    reference_features = [row["catalog_feature_values"] for row in verified_rows[0]]
    if any(
        [row["support"] for row in rows] != reference_support
        or [row["catalog_feature_values"] for row in rows] != reference_features
        for rows in verified_rows[1:]
    ):
        _fail("staged arms use different ordered support or features")
    if any(item != verified_manifests[0] for item in verified_manifests[1:]):
        _fail("staged arms retain different execution artifacts")
    semantic._verify_locked_support(  # noqa: SLF001
        expected_plan=_mapping(payload["support_plan"], "support plan"),
        result=results[0],
        rows=verified_rows[0],
        artifact_count=len(artifact_pool),
    )
    for result, metrics, rows, manifests in zip(
        results, complete, verified_rows, verified_manifests, strict=True
    ):
        semantic._verify_complete_metrics(  # noqa: SLF001
            metrics, result=result, rows=rows, manifests=manifests
        )
    controls = semantic._unpack_document(  # noqa: SLF001
        payload["controls"], expected_kind="executed_controls"
    )
    semantic._verify_controls(  # noqa: SLF001
        controls,
        protocol=protocol,
        support_ids=[str(item["event_id"]) for item in reference_support],
        execution_manifests=verified_manifests[0],
        reference_rows=verified_rows[0],
    )
    readiness = _mapping(payload["readiness"], "readiness")
    semantic._verify_readiness(  # noqa: SLF001
        readiness, complete_metrics=complete[-1], controls=controls
    )
    mode = str(payload["mode"])
    if mode == "kaggle_capacity_validation":
        semantic_core = semantic._independent_core_document(  # noqa: SLF001
            payload=proxy,
            artifacts=list(artifact_pool.values()),
            retained_results=retained_results,
            complete=complete,
            controls=controls,
            readiness=readiness,
        )
    else:
        locked_proxy = dict(payload)
        locked_proxy["run_binding"] = {
            "mode": "locked_development",
            "profile": "production",
            "development_test_seed": 2404,
            "kaggle_run_binding_sha256": payload["run_binding_sha256"],
            "support_plan_sha256": _mapping(payload["support_plan"], "support plan")[
                "support_plan_sha256"
            ],
        }
        semantic_core = semantic._independent_locked_core_document(  # noqa: SLF001
            payload=locked_proxy,
            artifacts=list(artifact_pool.values()),
            retained_results=retained_results,
            complete=complete,
            controls=controls,
            readiness=readiness,
        )
    support_plan = _mapping(payload["support_plan"], "support plan")
    staged_core = {
        "schema_version": "apar-sentinel-v5-kaggle-deterministic-core/1",
        "exclusion_schema": semantic._DETERMINISTIC_CORE_EXCLUSION_SCHEMA,  # noqa: SLF001
        "mode": mode,
        "profile": "production",
        "development_test_seed": payload["development_test_seed"],
        "run_binding_sha256": payload["run_binding_sha256"],
        "support_plan_sha256": support_plan["support_plan_sha256"],
        "semantic_core": semantic_core,
    }
    core_sha = _digest(staged_core)
    core_binding = _mapping(payload["deterministic_core"], "deterministic core")
    if (
        core_binding.get("schema_version") != "apar-sentinel-v5-kaggle-deterministic-core/1"
        or core_binding.get("exclusion_schema")
        != json.loads(
            _canonical_bytes(semantic._DETERMINISTIC_CORE_EXCLUSION_SCHEMA)  # noqa: SLF001
        )
        or core_binding.get("core_sha256") != core_sha
    ):
        _fail("staged deterministic core differs")
    observational = semantic._unpack_document(  # noqa: SLF001
        payload["observational_latency"], expected_kind="observational_latency"
    )
    expected_observational = semantic._independent_observational_document(  # noqa: SLF001
        core_sha256=core_sha,
        retained_results=retained_results,
        complete=complete,
        controls=controls,
        readiness=readiness,
    )
    if observational != expected_observational:
        _fail("staged observational latency differs")
    return (
        str(readiness["status"]),
        core_sha,
        str(observational["observational_latency_sha256"]),
    )


def _verify_final_payload(
    *,
    payload_bytes: bytes,
    root: Path,
    mode: str,
    run_binding: str,
    attempt: str,
    checkpoints: Sequence[_VerifiedCheckpoint],
    enforce_production_support: bool,
) -> tuple[dict[str, Any], str, str, str]:
    try:
        payload = _mapping(json.loads(payload_bytes), "staged final payload")
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise V5KaggleIndependentVerificationError("staged final payload is not JSON") from error
    if payload_bytes != _canonical_bytes(payload):
        _fail("staged final payload is not canonical JSON")
    expected_fields = {
        "schema_version",
        "mode",
        "profile",
        "development_test_seed",
        "run_binding_sha256",
        "support_plan",
        "attempt_receipt_sha256",
        "checkpoint_chain",
        "evidence_protocol",
        "catalog_sha256",
        "execution_artifact_pool",
        "arm_results",
        "complete_metrics",
        "controls",
        "readiness",
        "deterministic_core",
        "observational_latency",
        "payload_sha256",
    }
    _exact(payload, expected_fields, "staged final payload")
    if (
        payload.get("schema_version") != "apar-sentinel-v5-kaggle-staged-payload/1"
        or payload.get("mode") != mode
        or payload.get("profile") != "production"
        or payload.get("run_binding_sha256") != run_binding
        or payload.get("attempt_receipt_sha256") != attempt
    ):
        _fail("staged final payload mode/run/attempt binding differs")
    expected_seed = 404 if mode == "kaggle_capacity_validation" else 2404
    if payload.get("development_test_seed") != expected_seed:
        _fail("staged final payload seed differs")
    if payload["payload_sha256"] != _digest(
        {key: value for key, value in payload.items() if key != "payload_sha256"}
    ):
        _fail("staged final payload digest differs")
    chain = _mapping(payload["checkpoint_chain"], "checkpoint chain")
    chain_values = {
        "schema_version": "apar-sentinel-v5-checkpoint-chain/1",
        "attempt_receipt_sha256": attempt,
        "predecessor_stage_manifest_sha256": [
            [item.manifest["stage"], item.manifest["manifest_sha256"]] for item in checkpoints[:-1]
        ],
    }
    chain_values["predecessor_chain_root_sha256"] = _digest(chain_values)
    if chain != chain_values:
        _fail("staged payload predecessor chain differs")
    status, core_sha, observational_sha = _unpack_payload_semantics(
        payload=payload,
        root=root,
        enforce_production_support=enforce_production_support,
    )
    return payload, status, core_sha, observational_sha


def _json_record(record: _Record, label: str) -> dict[str, Any]:
    try:
        document = _mapping(json.loads(record.payload), label)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise V5KaggleIndependentVerificationError(f"{label} is not JSON") from error
    if record.payload != _canonical_bytes(document):
        _fail(f"{label} is not canonical JSON")
    return document


def _verify_corpus_records(
    *, records: Sequence[_Record], payload: Mapping[str, Any] | None
) -> dict[str, dict[str, list[dict[str, Any]]]]:
    if not records or records[0].kind != "corpus_header":
        _fail("corpus checkpoint header is missing")
    header = _json_record(records[0], "corpus header")
    order = _sequence(header.get("partition_order"), "corpus partition order")
    if tuple(order) != (
        "train",
        "calibration",
        "threshold",
        "development_test",
        "hardening_train",
        "adaptive_holdout",
    ):
        _fail("corpus partition order differs")
    support = _sequence(header.get("partition_support"), "corpus partition support")
    if len(support) != len(order):
        _fail("corpus partition support index differs")
    retained: dict[str, dict[str, list[dict[str, Any]]]] = {}
    cursor = 1
    isolation: dict[str, dict[str, set[str]]] = {}
    for expected_partition, declared_support in zip(order, support, strict=True):
        partition = str(expected_partition)
        declared = _mapping(declared_support, "corpus partition support")
        if cursor >= len(records) or records[cursor].kind != "partition_header":
            _fail("corpus partition header is missing or reordered")
        partition_header = _json_record(records[cursor], "corpus partition header")
        cursor += 1
        if (
            records[cursor - 1].key != partition
            or partition_header.get("partition") != partition
            or declared.get("partition") != partition
            or partition_header.get("decisions") != declared.get("decisions")
            or partition_header.get("executions") != declared.get("executions")
        ):
            _fail("corpus partition header/support binding differs")
        decision_count = int(partition_header["decisions"])
        execution_count = int(partition_header["executions"])
        decisions: list[dict[str, Any]] = []
        executions: list[dict[str, Any]] = []
        for _ in range(decision_count):
            if cursor >= len(records) or records[cursor].kind != "decision_row":
                _fail("corpus decision row is missing or reordered")
            wrapper = _json_record(records[cursor], "corpus decision row")
            decision = _mapping(wrapper.get("decision"), "corpus decision")
            if (
                wrapper.get("partition") != partition
                or records[cursor].key != f"{partition}:{decision.get('event_id')}"
            ):
                _fail("corpus decision identity binding differs")
            decisions.append(decision)
            cursor += 1
        for _ in range(execution_count):
            if cursor >= len(records) or records[cursor].kind != "execution_manifest":
                _fail("corpus execution manifest is missing or reordered")
            wrapper = _json_record(records[cursor], "corpus execution manifest")
            execution = _mapping(wrapper.get("execution"), "corpus execution")
            payload_json = _canonical_bytes(execution).decode()
            artifact = {
                "evidence_sha256": execution.get("evidence_sha256"),
                "artifact_sha256": execution.get("artifact_sha256"),
                "payload_sha256": hashlib.sha256(payload_json.encode()).hexdigest(),
                "payload_json": payload_json,
            }
            semantic._verify_manifest(artifact)  # noqa: SLF001
            if (
                wrapper.get("partition") != partition
                or records[cursor].key
                != f"{partition}:{execution.get('evidence_sha256')}"
            ):
                _fail("corpus execution identity binding differs")
            executions.append(execution)
            cursor += 1
        if len({str(item.get("event_id")) for item in decisions}) != len(decisions):
            _fail("corpus decision event IDs are duplicated")
        retained[partition] = {"decisions": decisions, "executions": executions}
        isolation[partition] = {
            "event": {str(item.get("event_id")) for item in decisions},
            "payment": {str(item.get("payment_id")) for item in decisions},
            "campaign": {str(item.get("campaign_id")) for item in decisions},
            "actor": {str(item.get("actor_id")) for item in decisions},
            "counterparty": {str(item.get("counterparty_id")) for item in decisions},
            "time": {str(item.get("decision_at")) for item in decisions},
            "account": {
                str(account)
                for execution in executions
                for account in _sequence(execution.get("account_ids"), "account IDs")
            },
        }
    if cursor != len(records):
        _fail("corpus checkpoint contains extra records")
    for index, left_name in enumerate(order):
        for right_name in order[index + 1 :]:
            for domain in isolation[str(left_name)]:
                if isolation[str(left_name)][domain] & isolation[str(right_name)][domain]:
                    _fail(f"corpus partition {domain} domains overlap")
    if header.get("corpus_sha256") != _digest(retained):
        _fail("corpus content digest differs")
    if payload is None:
        return retained
    final_pool = {
        str(document["evidence_sha256"]): document
        for document in (
            semantic._unpack_document(  # noqa: SLF001
                item,
                expected_kind="execution_artifact",
                max_uncompressed_bytes=16_777_216,
            )
            for item in _sequence(
                payload["execution_artifact_pool"], "execution artifact pool"
            )
        )
    }
    model_executions = {
        str(item["evidence_sha256"]): item
        for partition in ("train", "calibration", "threshold", "development_test")
        for item in retained[partition]["executions"]
    }
    if set(final_pool) != set(model_executions):
        _fail("corpus model-partition execution support differs from final payload")
    for evidence_id, execution in model_executions.items():
        artifact = final_pool[evidence_id]
        if json.loads(str(artifact["payload_json"])) != execution:
            _fail("corpus execution manifest differs from final artifact pool")
    return retained


def _array_values(
    record: _Record,
    metadata: object,
    *,
    expected_dtype: str,
) -> tuple[int | float, ...]:
    values = _mapping(metadata, "feature array metadata")
    if set(values) != {"bytes", "dtype", "sha256", "shape"}:
        _fail("feature array metadata fields differ")
    if (
        values.get("dtype") != expected_dtype
        or values.get("bytes") != len(record.payload)
        or values.get("sha256") != hashlib.sha256(record.payload).hexdigest()
    ):
        _fail("feature array byte binding differs")
    shape = _sequence(values.get("shape"), "feature array shape")
    count = 1
    for dimension in shape:
        if type(dimension) is not int or dimension < 0:
            _fail("feature array shape is invalid")
        count *= dimension
    width = {"<f8": 8, "<i8": 8, "|u1": 1}[expected_dtype]
    if count * width != len(record.payload):
        _fail("feature array shape/size binding differs")
    if expected_dtype == "|u1":
        decoded: tuple[int | float, ...] = tuple(record.payload)
    else:
        code = "<d" if expected_dtype == "<f8" else "<q"
        decoded = tuple(item[0] for item in struct.iter_unpack(code, record.payload))
    if any(isinstance(item, float) and not math.isfinite(item) for item in decoded):
        _fail("feature array contains a non-finite value")
    return decoded


def _verify_feature_records(
    *,
    records: Sequence[_Record],
    payload: Mapping[str, Any],
    corpus: Mapping[str, Mapping[str, list[dict[str, Any]]]],
    expanded_results: Sequence[Mapping[str, Any]] | None,
) -> None:
    if not records or records[0].kind != "feature_header":
        _fail("feature checkpoint header is missing")
    header = _json_record(records[0], "feature header")
    names = tuple(str(item) for item in _sequence(header.get("feature_names"), "feature names"))
    if (
        header.get("schema_version") != "apar-sentinel-v5-kaggle-features/1"
        or header.get("catalog_sha256") != payload["catalog_sha256"]
        or tuple(header.get("partition_order", ()))
        != ("train", "calibration", "threshold", "development_test")
        or not names
        or any(
            forbidden in name
            for name in names
            for forbidden in (
                "is_fraud",
                "family",
                "campaign",
                "seed",
                "split",
                "generator",
                "final_outcome",
            )
        )
    ):
        _fail("feature catalog/header binding differs")
    training_by_partition: dict[str, Mapping[str, Any]] = {}
    if expanded_results is not None:
        for result in expanded_results:
            spec = _mapping(result.get("arm_spec"), "arm specification")
            for item in _sequence(spec.get("training_partitions"), "training partitions"):
                training = _mapping(item, "training partition")
                partition = str(training.get("partition"))
                prior = training_by_partition.setdefault(partition, training)
                if prior != training:
                    _fail("arms use different training partition evidence")
        evaluation_rows: Sequence[Any] = _sequence(
            expanded_results[0].get("row_evidence"), "evaluation row evidence"
        )
    else:
        evaluation_rows = ()
    cursor = 1
    for partition in ("train", "calibration", "threshold", "development_test"):
        if cursor >= len(records) or records[cursor].kind != "prepared_partition":
            _fail("prepared feature partition is missing or reordered")
        metadata_record = records[cursor]
        metadata = _json_record(metadata_record, "prepared feature partition")
        cursor += 1
        event_ids = tuple(
            str(item) for item in _sequence(metadata.get("event_ids"), "feature event IDs")
        )
        campaign_ids = tuple(
            str(item)
            for item in _sequence(metadata.get("campaign_ids"), "feature campaign IDs")
        )
        if (
            metadata_record.key != partition
            or metadata.get("partition") != partition
            or len(event_ids) != len(set(event_ids))
            or len(campaign_ids) != len(event_ids)
        ):
            _fail("prepared feature partition identity differs")
        expected_kinds = (
            ("feature_matrix", "<f8", "matrix"),
            ("labels", "<i8", "labels"),
            ("amounts", "<f8", "amounts"),
            ("trust_failures", "|u1", "trust_failures"),
        )
        arrays: dict[str, tuple[int | float, ...]] = {}
        for kind, dtype, metadata_name in expected_kinds:
            if cursor >= len(records):
                _fail("feature array is missing")
            record = records[cursor]
            cursor += 1
            if record.kind != kind or record.key != f"{partition}:{kind}":
                _fail("feature array order differs")
            arrays[kind] = _array_values(
                record,
                metadata.get(metadata_name),
                expected_dtype=dtype,
            )
        rows = len(event_ids)
        if len(arrays["feature_matrix"]) != rows * len(names):
            _fail("feature matrix shape differs from catalog/support")
        matrix = tuple(
            tuple(
                float(value)
                for value in arrays["feature_matrix"][
                    index * len(names) : (index + 1) * len(names)
                ]
            )
            for index in range(rows)
        )
        labels = tuple(int(value) for value in arrays["labels"])
        amounts = tuple(float(value) for value in arrays["amounts"])
        trust = tuple(int(value) for value in arrays["trust_failures"])
        if (
            len(labels) != rows
            or len(amounts) != rows
            or len(trust) != rows
            or set(labels) - {0, 1}
            or set(trust) - {0, 1}
        ):
            _fail("feature support arrays differ")
        if cursor >= len(records) or records[cursor].kind != "feature_batch":
            _fail("feature batch is missing or reordered")
        batch = _json_record(records[cursor], "feature batch")
        cursor += 1
        provenance = _sequence(batch.get("provenance"), "feature provenance")
        if (
            records[cursor - 1].key != f"{partition}:feature_batch"
            or batch.get("catalog_sha256") != payload["catalog_sha256"]
            or batch.get("matrix") != [list(row) for row in matrix]
            or tuple(str(_mapping(item, "provenance").get("event_id")) for item in provenance)
            != event_ids
            or batch.get("batch_sha256")
            != hashlib.sha256(
                json.dumps({"names": list(names), "rows": matrix}, sort_keys=True).encode()
            ).hexdigest()
            or batch.get("batch_sha256") != metadata.get("feature_batch_sha256")
        ):
            _fail("feature batch/matrix/provenance binding differs")
        corpus_rows = corpus[partition]["decisions"]
        corpus_amounts: list[float] = []
        for item in corpus_rows:
            amount = item.get("amount")
            if not isinstance(amount, (int, float, str)):
                _fail("corpus decision amount is malformed")
            corpus_amounts.append(float(amount))
        if (
            tuple(str(item.get("event_id")) for item in corpus_rows) != event_ids
            or tuple(str(item.get("campaign_id")) for item in corpus_rows) != campaign_ids
            or tuple(int(bool(item.get("is_fraud"))) for item in corpus_rows) != labels
            or tuple(corpus_amounts) != amounts
            or tuple(int(item.get("integrity_status") == "fail") for item in corpus_rows)
            != trust
        ):
            _fail("feature support differs from executed corpus")
        if partition == "development_test":
            if metadata.get("has_training_evidence") is not False:
                _fail("development feature partition claims training evidence")
            if expanded_results is not None and (
                tuple(
                    str(_mapping(item, "evaluation row")["support"]["event_id"])
                    for item in evaluation_rows
                )
                != event_ids
                or tuple(
                    tuple(
                        float(value)
                        for value in _mapping(item, "evaluation row")[
                            "catalog_feature_values"
                        ]
                    )
                    for item in evaluation_rows
                )
                != matrix
            ):
                _fail("development features differ from scored arm evidence")
        else:
            if cursor >= len(records) or records[cursor].kind != "training_evidence":
                _fail("training feature evidence is missing or reordered")
            training = _json_record(records[cursor], "training feature evidence")
            cursor += 1
            if (
                records[cursor - 1].key != f"{partition}:training_evidence"
                or metadata.get("has_training_evidence") is not True
                or (
                    expanded_results is not None
                    and training_by_partition.get(partition) != training
                )
                or tuple(training.get("ordered_event_ids", ())) != event_ids
                or tuple(
                    tuple(float(value) for value in row)
                    for row in training.get("feature_matrix", ())
                )
                != matrix
            ):
                _fail("training feature evidence differs from arm specification")
            semantic._verify_training_partition(training)  # noqa: SLF001
    if cursor != len(records):
        _fail("feature checkpoint contains extra records")


def _verify_stage_record_bindings(
    *, checkpoints: Sequence[_VerifiedCheckpoint], payload: Mapping[str, Any]
) -> None:
    corpus_records = checkpoints[1].deterministic_records
    feature_records = checkpoints[2].deterministic_records
    corpus = _verify_corpus_records(records=corpus_records, payload=payload)
    corpus_header = _json_record(corpus_records[0], "corpus header")
    if (
        corpus_header.get("schema_version") != "apar-sentinel-v5-kaggle-corpus/1"
        or corpus_header.get("mode") != payload["mode"]
        or corpus_header.get("development_test_seed") != payload["development_test_seed"]
        or corpus_header.get("support_plan_sha256")
        != _mapping(payload["support_plan"], "support plan")["support_plan_sha256"]
    ):
        _fail("corpus checkpoint binding differs")

    arm_records = checkpoints[3].deterministic_records
    arm_observations = checkpoints[3].observational_records
    if len(arm_records) != 5 or arm_records[0].kind != "arm_header":
        _fail("arm checkpoint header differs")
    arm_header = _json_record(arm_records[0], "arm header")
    retained_results = [
        semantic._unpack_document(item, expected_kind="arm_result")  # noqa: SLF001
        for item in _sequence(payload["arm_results"], "arm results")
    ]
    expanded_results: list[dict[str, Any]] = []
    artifact_pool = {
        str(document["evidence_sha256"]): document
        for document in (
            semantic._unpack_document(  # noqa: SLF001
                item,
                expected_kind="execution_artifact",
                max_uncompressed_bytes=16_777_216,
            )
            for item in _sequence(payload["execution_artifact_pool"], "execution artifact pool")
        )
    }
    expected_result_digests = []
    expected_cores: list[dict[str, Any]] = []
    expected_observations: list[dict[str, Any]] = []
    for retained in retained_results:
        expanded, _used = semantic._expand_retained_result(  # noqa: SLF001
            retained, artifact_pool
        )
        expanded_results.append(expanded)
        core = json.loads(_canonical_bytes(expanded))
        latency_samples: list[dict[str, Any]] = []
        for row in core["row_evidence"]:
            latency = row.pop("latency_ms")
            row_output = row.pop("row_output_sha256")
            latency_samples.append(
                {
                    "event_id": row["support"]["event_id"],
                    "latency_ms": latency,
                    "row_output_sha256": row_output,
                }
            )
            row["deterministic_row_sha256"] = _digest(row)
        observation = {
            "schema_version": "apar-sentinel-v5-kaggle-arm-latency/1",
            "arm": core["arm"],
            "deterministic_result_sha256": "",
            "samples": latency_samples,
            "p50_latency_ms": core["p50_latency_ms"],
            "p95_latency_ms": core["p95_latency_ms"],
            "p99_latency_ms": core["p99_latency_ms"],
            "score_sha256": core["score_sha256"],
            "result_sha256": core["result_sha256"],
        }
        for field in (
            "p50_latency_ms",
            "p95_latency_ms",
            "p99_latency_ms",
            "score_sha256",
            "result_sha256",
        ):
            core.pop(field)
        core["deterministic_result_sha256"] = _digest(core)
        observation["deterministic_result_sha256"] = core[
            "deterministic_result_sha256"
        ]
        observation["observational_sha256"] = _digest(observation)
        expected_result_digests.append(core["deterministic_result_sha256"])
        expected_cores.append(core)
        expected_observations.append(observation)
    _verify_feature_records(
        records=feature_records,
        payload=payload,
        corpus=corpus,
        expanded_results=expanded_results,
    )
    feature_header = _json_record(feature_records[0], "feature header")
    if (
        feature_header.get("corpus_sha256") != corpus_header.get("corpus_sha256")
        or checkpoints[2].manifest["record_count"] != 28
    ):
        _fail("feature checkpoint corpus/count binding differs")
    if (
        arm_header.get("schema_version") != "apar-sentinel-v5-kaggle-arms/1"
        or arm_header.get("arm_order") != list(_ARMS)
        or arm_header.get("deterministic_result_sha256") != expected_result_digests
        or arm_header.get("support_event_ids")
        != [
            row["support"]["event_id"]
            for row in expanded_results[0]["row_evidence"]
        ]
        or arm_header.get("support_sha256")
        != expanded_results[0]["support_sha256"]
        or checkpoints[3].manifest["record_count"] != 5
        or checkpoints[3].manifest["observational_record_count"] != 4
    ):
        _fail("arm checkpoint/final payload binding differs")
    for index, arm in enumerate(_ARMS):
        if (
            arm_records[index + 1].kind != "arm_result"
            or arm_records[index + 1].key != arm
            or _json_record(arm_records[index + 1], "arm deterministic result")
            != expected_cores[index]
            or arm_observations[index].kind != "arm_latency"
            or arm_observations[index].key != arm
            or _json_record(arm_observations[index], "arm latency result")
            != expected_observations[index]
        ):
            _fail("arm checkpoint result differs from final payload")

    final_controls = semantic._unpack_document(  # noqa: SLF001
        payload["controls"], expected_kind="executed_controls"
    )
    observed_controls: dict[str, dict[str, Any]] = {}
    expected_groups = (
        ("label_shuffle", ("label_shuffle",)),
        (
            "invariance",
            (
                "identity_rename",
                "future_causality",
                "equal_time_isolation",
                "feature_leakage",
            ),
        ),
        ("single_class", ("benign_only", "fraud_only_diagnostic")),
    )
    for checkpoint, (group_name, control_names) in zip(
        checkpoints[4:7], expected_groups, strict=True
    ):
        if (
            len(checkpoint.deterministic_records) != 1
            or len(checkpoint.observational_records) != 1
            or checkpoint.deterministic_records[0].kind != "control_group"
            or checkpoint.observational_records[0].kind != "control_observation"
        ):
            _fail("control checkpoint record identity differs")
        observation = _json_record(checkpoint.observational_records[0], "control observation")
        claimed = observation.pop("observational_group_sha256", None)
        if claimed != _digest(observation):
            _fail("control observational group digest differs")
        group = _mapping(observation.get("executed_group"), "executed control group")
        if (
            group.get("group") != group_name
            or tuple(
                item.get("name")
                for item in _sequence(group.get("controls"), "group controls")
                if isinstance(item, dict)
            )
            != control_names
        ):
            _fail("control group membership differs")
        for item in _sequence(group["controls"], "group controls"):
            control = _mapping(item, "group control")
            observed_controls[str(control["name"])] = control
    final_control_sequence = _sequence(final_controls["controls"], "final controls")
    if any(
        observed_controls.get(str(item["name"])) != item for item in final_control_sequence
    ) or set(observed_controls) != {str(item["name"]) for item in final_control_sequence}:
        _fail("control checkpoints differ from final control suite")

    metric_checkpoint = checkpoints[7]
    if (
        len(metric_checkpoint.deterministic_records) != 1
        or len(metric_checkpoint.observational_records) != 1
        or metric_checkpoint.deterministic_records[0].kind != "metric_evidence"
        or metric_checkpoint.observational_records[0].kind != "metric_observation"
    ):
        _fail("metric checkpoint record identity differs")
    metric_core = _json_record(metric_checkpoint.deterministic_records[0], "metric core")
    claimed_metric_core = metric_core.pop("deterministic_metric_stage_sha256", None)
    if claimed_metric_core != _digest(metric_core):
        _fail("metric deterministic core digest differs")
    metric_observation = _json_record(
        metric_checkpoint.observational_records[0], "metric observation"
    )
    claimed_metric_observation = metric_observation.pop("observational_metric_stage_sha256", None)
    if (
        claimed_metric_observation != _digest(metric_observation)
        or metric_observation.get("deterministic_metric_stage_sha256") != claimed_metric_core
    ):
        _fail("metric observational layer binding differs")
    final_metrics = [
        semantic._unpack_document(item, expected_kind="complete_metrics")  # noqa: SLF001
        for item in _sequence(payload["complete_metrics"], "final metrics")
    ]
    if (
        metric_observation.get("complete_metrics") != final_metrics
        or metric_observation.get("controls") != final_controls
        or metric_observation.get("readiness") != payload["readiness"]
    ):
        _fail("metric checkpoint differs from final payload")


class V5KaggleVerificationReport(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal["apar-sentinel-v5-kaggle-verification/1"]
    valid: Literal[True]
    mode: Literal["kaggle_capacity_validation", "kaggle_locked_successor"]
    verified_stage_ids: tuple[str, ...]
    chain_root_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    execution_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    deterministic_stage_sha256: tuple[tuple[str, str], ...]
    observational_stage_sha256: tuple[tuple[str, str], ...]
    final_payload_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    deterministic_core_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    observational_latency_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    readiness_status: Literal["ready", "not_ready"]
    verifier_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    report_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def report_is_bound(self) -> Self:
        if self.verified_stage_ids != _STAGES:
            raise ValueError("verification report stage order differs")
        if self.report_sha256 != _digest(self.model_dump(mode="json", exclude={"report_sha256"})):
            raise ValueError("verification report digest differs")
        return self


class V5KagglePrefixVerificationReport(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal["apar-sentinel-v5-kaggle-prefix-verification/1"]
    valid: Literal[True]
    mode: Literal["kaggle_capacity_validation", "kaggle_locked_successor"]
    verified_stage_ids: tuple[str, ...]
    last_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    execution_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    deterministic_stage_sha256: tuple[tuple[str, str], ...]
    observational_stage_sha256: tuple[tuple[str, str], ...]
    verifier_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    report_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def report_is_bound(self) -> Self:
        if (
            not self.verified_stage_ids
            or self.verified_stage_ids != _STAGES[: len(self.verified_stage_ids)]
            or len(self.verified_stage_ids) >= len(_STAGES)
        ):
            raise ValueError("prefix verification stage order differs")
        if self.report_sha256 != _digest(
            self.model_dump(mode="json", exclude={"report_sha256"})
        ):
            raise ValueError("prefix verification report digest differs")
        return self


def verify_v5_kaggle_prefix(
    *,
    root: Path,
    checkpoint_roots: Sequence[Path],
    expected_mode: Literal["kaggle_capacity_validation", "kaggle_locked_successor"],
) -> V5KagglePrefixVerificationReport:
    """Verify one completed Stage-00 through Stage-70 prefix offline."""
    try:
        roots = tuple(checkpoint_roots)
        if not 1 <= len(roots) < len(_STAGES):
            _fail("prefix verification requires one through eight checkpoint roots")
        protocol, run, run_binding = _protocol_and_run_binding(
            root=root.resolve(), expected_mode=expected_mode
        )
        checkpoints = tuple(_read_checkpoint(path) for path in roots)
        if tuple(item.manifest["stage"] for item in checkpoints) != _STAGES[: len(roots)]:
            _fail("checkpoint prefix stage order differs")
        attempt: str | None = None
        common_environment: bytes | None = None
        for index, checkpoint in enumerate(checkpoints):
            manifest = checkpoint.manifest
            if manifest["run_binding_sha256"] != run_binding:
                _fail("checkpoint prefix run binding differs")
            if index == 0:
                if manifest["predecessor_stage"] is not None:
                    _fail("authorization prefix unexpectedly has a predecessor")
                attempt = str(manifest["attempt_receipt_sha256"])
            else:
                previous = checkpoints[index - 1].manifest
                if (
                    manifest["predecessor_stage"] != previous["stage"]
                    or manifest["predecessor_manifest_sha256"]
                    != previous["manifest_sha256"]
                    or manifest["predecessor_deterministic_sha256"]
                    != previous["deterministic_sha256"]
                    or manifest["attempt_receipt_sha256"] != attempt
                ):
                    _fail("checkpoint prefix predecessor/attempt binding differs")
            environment = _mapping(
                checkpoint.observation["environment"], "checkpoint environment"
            )
            projection = _canonical_bytes(
                {
                    key: value
                    for key, value in environment.items()
                    if key not in {"environment_sha256", "notebook_sha256"}
                }
            )
            if common_environment is None:
                common_environment = projection
            elif projection != common_environment:
                _fail("checkpoint prefix mixes runtime environments")
            notebook = (
                root.resolve()
                / "kaggle/defense_v5"
                / f"{manifest['stage']}.ipynb"
            )
            if (
                not notebook.is_file()
                or notebook.is_symlink()
                or environment["notebook_sha256"]
                != hashlib.sha256(notebook.read_bytes()).hexdigest()
            ):
                _fail("checkpoint prefix notebook binding differs")
        authorization_records = checkpoints[0].deterministic_records
        if len(authorization_records) != 1:
            _fail("authorization prefix record differs")
        authorization = _json_record(authorization_records[0], "authorization record")
        _exact(authorization, _AUTHORIZATION_FIELDS, "authorization record")
        base = _mapping(
            json.loads(
                (root / str(protocol["source_bindings"]["base_protocol_path"])).read_bytes()
            ),
            "production base protocol",
        )
        expected_support = semantic._independent_locked_support_plan(base)  # noqa: SLF001
        expected_support["mode"] = expected_mode
        expected_support["support_plan_sha256"] = _digest(
            {
                key: value
                for key, value in expected_support.items()
                if key != "support_plan_sha256"
            }
        )
        if (
            authorization.get("schema_version")
            != "apar-sentinel-v5-kaggle-authorization/1"
            or authorization.get("mode") != expected_mode
            or authorization.get("profile") != "production"
            or authorization.get("development_test_seed")
            != (404 if expected_mode == "kaggle_capacity_validation" else 2404)
            or authorization.get("run_binding_sha256") != run_binding
            or authorization.get("attempt_receipt_sha256") != attempt
            or not _is_sha256(authorization.get("execution_manifest_sha256"))
            or authorization.get("protocol_sha256") != _digest(protocol)
            or authorization.get("support_plan") != expected_support
            or authorization.get("source_bindings") != protocol["source_bindings"]
            or authorization.get("recovery") != protocol["recovery"]
            or authorization.get("resources") != protocol["resources"]
            or authorization.get("checkpoint") != protocol["checkpoint"]
            or authorization.get("repeatable") != run["repeatable"]
            or authorization.get("authorization_required")
            != run["authorization_required"]
        ):
            _fail("authorization prefix binding differs")
        corpus: dict[str, dict[str, list[dict[str, Any]]]] | None = None
        if len(checkpoints) >= 2:
            corpus = _verify_corpus_records(
                records=checkpoints[1].deterministic_records,
                payload=None,
            )
            corpus_header = _json_record(
                checkpoints[1].deterministic_records[0], "corpus header"
            )
            if (
                corpus_header.get("mode") != expected_mode
                or corpus_header.get("development_test_seed")
                != (404 if expected_mode == "kaggle_capacity_validation" else 2404)
                or corpus_header.get("support_plan_sha256")
                != expected_support["support_plan_sha256"]
            ):
                _fail("corpus prefix authorization binding differs")
        if len(checkpoints) >= 3:
            if corpus is None:
                raise AssertionError("feature prefix lacks verified corpus")
            catalog_path = root / str(
                protocol["source_bindings"]["feature_catalog_path"]
            )
            catalog_sha256 = hashlib.sha256(catalog_path.read_bytes()).hexdigest()
            _verify_feature_records(
                records=checkpoints[2].deterministic_records,
                payload={"catalog_sha256": catalog_sha256},
                corpus=corpus,
                expanded_results=None,
            )
        verifier_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        values = {
            "schema_version": "apar-sentinel-v5-kaggle-prefix-verification/1",
            "valid": True,
            "mode": expected_mode,
            "verified_stage_ids": _STAGES[: len(checkpoints)],
            "last_manifest_sha256": checkpoints[-1].manifest["manifest_sha256"],
            "execution_manifest_sha256": authorization["execution_manifest_sha256"],
            "deterministic_stage_sha256": tuple(
                (item.manifest["stage"], item.manifest["deterministic_sha256"])
                for item in checkpoints
            ),
            "observational_stage_sha256": tuple(
                (item.manifest["stage"], item.manifest["observation_sha256"])
                for item in checkpoints
            ),
            "verifier_sha256": verifier_sha,
        }
        values["report_sha256"] = _digest(values)
        return V5KagglePrefixVerificationReport.model_validate(values)
    except V5KaggleIndependentVerificationError:
        raise
    except (KeyError, IndexError, OSError, TypeError, ValueError) as error:
        raise V5KaggleIndependentVerificationError(
            "staged prefix is malformed or internally inconsistent"
        ) from error


def _verify_v5_kaggle_evidence(
    *,
    root: Path,
    checkpoint_roots: Sequence[Path],
    final_root: Path,
    expected_mode: Literal["kaggle_capacity_validation", "kaggle_locked_successor"],
    enforce_production_support: bool,
) -> V5KaggleVerificationReport:
    """Independently verify the complete nine-stage chain and final payload."""
    try:
        roots = tuple(checkpoint_roots)
        if len(roots) != 8:
            _fail("verification requires exact Stage 00-70 checkpoint roots")
        _protocol, _run, run_binding = _protocol_and_run_binding(
            root=root.resolve(), expected_mode=expected_mode
        )
        checkpoints = tuple(_read_checkpoint(path) for path in (*roots, final_root))
        if tuple(item.manifest["stage"] for item in checkpoints) != _STAGES:
            _fail("checkpoint stage order differs")
        for index, checkpoint in enumerate(checkpoints):
            manifest = checkpoint.manifest
            if manifest["run_binding_sha256"] != run_binding:
                _fail("checkpoint run binding differs")
            if index == 0:
                if any(
                    manifest[field] is not None
                    for field in (
                        "predecessor_stage",
                        "predecessor_manifest_sha256",
                        "predecessor_deterministic_sha256",
                    )
                ):
                    _fail("authorization checkpoint has a predecessor")
            else:
                previous = checkpoints[index - 1].manifest
                if (
                    manifest["predecessor_stage"] != previous["stage"]
                    or manifest["predecessor_manifest_sha256"] != previous["manifest_sha256"]
                    or manifest["predecessor_deterministic_sha256"]
                    != previous["deterministic_sha256"]
                ):
                    _fail("checkpoint predecessor lineage differs")
        attempts = {item.manifest["attempt_receipt_sha256"] for item in checkpoints}
        if len(attempts) != 1:
            _fail("checkpoint chain mixes attempt receipts")
        environment_documents = tuple(
            _mapping(item.observation["environment"], "checkpoint environment")
            for item in checkpoints
        )
        common_environments = {
            _canonical_bytes(
                {
                    key: value
                    for key, value in environment.items()
                    if key not in {"environment_sha256", "notebook_sha256"}
                }
            )
            for environment in environment_documents
        }
        if len(common_environments) != 1:
            _fail("checkpoint chain mixes runtime environments")
        attempt = str(next(iter(attempts)))
        authorization = checkpoints[0].deterministic_records
        if len(authorization) != 1 or authorization[0].kind != "authorization":
            _fail("authorization checkpoint record differs")
        authorization_document = _mapping(
            json.loads(authorization[0].payload), "authorization record"
        )
        _exact(authorization_document, _AUTHORIZATION_FIELDS, "authorization record")
        if (
            authorization[0].payload != _canonical_bytes(authorization_document)
            or authorization_document.get("mode") != expected_mode
            or authorization_document.get("run_binding_sha256") != run_binding
            or authorization_document.get("attempt_receipt_sha256") != attempt
            or not _is_sha256(
                authorization_document.get("execution_manifest_sha256")
            )
        ):
            _fail("authorization evidence binding differs")
        final = checkpoints[-1]
        if (
            len(final.deterministic_records) != 1
            or len(final.observational_records) != 1
            or final.deterministic_records[0].kind != "final_core"
            or final.observational_records[0].kind != "final_payload"
        ):
            _fail("final checkpoint records differ")
        payload, status, core_sha, latency_sha = _verify_final_payload(
            payload_bytes=final.observational_records[0].payload,
            root=root.resolve(),
            mode=expected_mode,
            run_binding=run_binding,
            attempt=attempt,
            checkpoints=checkpoints,
            enforce_production_support=enforce_production_support,
        )
        if enforce_production_support:
            for checkpoint, environment in zip(
                checkpoints, environment_documents, strict=True
            ):
                notebook = (
                    root.resolve()
                    / "kaggle/defense_v5"
                    / f"{checkpoint.manifest['stage']}.ipynb"
                )
                if (
                    not notebook.is_file()
                    or notebook.is_symlink()
                    or environment["notebook_sha256"]
                    != hashlib.sha256(notebook.read_bytes()).hexdigest()
                ):
                    _fail("checkpoint notebook source binding differs")
        _verify_stage_record_bindings(
            checkpoints=checkpoints,
            payload=payload,
        )
        final_core = _mapping(json.loads(final.deterministic_records[0].payload), "final core")
        claimed_final_core = final_core.pop("final_core_sha256", None)
        if (
            claimed_final_core != _digest(final_core)
            or final_core.get("deterministic_core_sha256") != core_sha
        ):
            _fail("final deterministic core binding differs")
        verifier_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        values = {
            "schema_version": "apar-sentinel-v5-kaggle-verification/1",
            "valid": True,
            "mode": expected_mode,
            "verified_stage_ids": _STAGES,
            "chain_root_sha256": final.manifest["manifest_sha256"],
            "execution_manifest_sha256": authorization_document[
                "execution_manifest_sha256"
            ],
            "deterministic_stage_sha256": tuple(
                (item.manifest["stage"], item.manifest["deterministic_sha256"])
                for item in checkpoints
            ),
            "observational_stage_sha256": tuple(
                (item.manifest["stage"], item.manifest["observation_sha256"])
                for item in checkpoints
            ),
            "final_payload_sha256": payload["payload_sha256"],
            "deterministic_core_sha256": core_sha,
            "observational_latency_sha256": latency_sha,
            "readiness_status": status,
            "verifier_sha256": verifier_sha,
        }
        values["report_sha256"] = _digest(values)
        return V5KaggleVerificationReport.model_validate(values)
    except V5KaggleIndependentVerificationError:
        raise
    except (KeyError, IndexError, OSError, TypeError, ValueError) as error:
        raise V5KaggleIndependentVerificationError(
            "staged evidence is malformed or internally inconsistent"
        ) from error


def verify_v5_kaggle_evidence(
    *,
    root: Path,
    checkpoint_roots: Sequence[Path],
    final_root: Path,
    expected_mode: Literal["kaggle_capacity_validation", "kaggle_locked_successor"],
) -> V5KaggleVerificationReport:
    """Verify a real production-sized capacity or locked-successor chain."""
    return _verify_v5_kaggle_evidence(
        root=root,
        checkpoint_roots=checkpoint_roots,
        final_root=final_root,
        expected_mode=expected_mode,
        enforce_production_support=True,
    )


def _verify_v5_kaggle_test_fixture(
    *,
    root: Path,
    checkpoint_roots: Sequence[Path],
    final_root: Path,
) -> V5KaggleVerificationReport:
    """Replay bounded safe test evidence; the public CLI cannot select this path."""
    return _verify_v5_kaggle_evidence(
        root=root,
        checkpoint_roots=checkpoint_roots,
        final_root=final_root,
        expected_mode="kaggle_capacity_validation",
        enforce_production_support=False,
    )


__all__ = [
    "V5KaggleIndependentVerificationError",
    "V5KagglePrefixVerificationReport",
    "V5KaggleVerificationReport",
    "verify_v5_kaggle_evidence",
    "verify_v5_kaggle_prefix",
]
