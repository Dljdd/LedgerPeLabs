"""Streaming, immutable checkpoint storage for staged Sentinel v5 execution."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from pydantic import ValidationError

from apar.evaluation.v5_checkpoint_storage import (
    V5CheckpointInput,
    V5CheckpointObservation,
    iter_v5_checkpoint_observational_records,
    iter_v5_checkpoint_records,
    publish_v5_checkpoint,
    read_v5_checkpoint_manifest,
    read_v5_checkpoint_observation,
)
from apar.evaluation.v5_kaggle_protocol import (
    V5KaggleEnvironmentBinding,
    V5KaggleResourceGates,
    V5KaggleStage,
)


def _limits() -> V5KaggleResourceGates:
    return V5KaggleResourceGates.model_construct(
        max_peak_rss_bytes=1_000_000,
        max_stage_seconds=60,
        max_stage_output_bytes=1_000_000,
        max_checkpoint_chunk_bytes=64,
        max_checkpoint_chunks=100,
    )


def _environment(*, notebook: str = "a" * 64) -> V5KaggleEnvironmentBinding:
    return V5KaggleEnvironmentBinding.bind(
        provider="kaggle",
        image="python-cpu-test",
        image_sha256="1" * 64,
        python_version="3.12.5",
        architecture="x86_64",
        cpu_count=4,
        dependency_manifest_sha256="2" * 64,
        source_archive_sha256="3" * 64,
        notebook_sha256=notebook,
        internet_enabled=False,
        accelerator="none",
        file_fsync_supported=True,
        directory_fsync_supported=True,
        hardlink_no_replace_supported=True,
    )


def _observation(
    *,
    environment: V5KaggleEnvironmentBinding | None = None,
    started: str = "2026-08-24T00:00:00Z",
    completed: str = "2026-08-24T00:00:01Z",
    wall_seconds: float = 1.0,
    rss: tuple[int, ...] = (100_000, 200_000),
) -> V5CheckpointObservation:
    return V5CheckpointObservation(
        schema_version="apar-sentinel-v5-kaggle-observation/1",
        started_at_utc=started,
        completed_at_utc=completed,
        wall_seconds=wall_seconds,
        rss_samples_bytes=rss,
        host_available_samples_bytes=tuple(8_000_000 for _ in rss),
        peak_rss_bytes=max(rss),
        environment=environment or _environment(),
    )


def _publish(
    output_root: Path,
    *,
    stage: V5KaggleStage = V5KaggleStage.AUTHORIZE,
    predecessor: object | None = None,
    observation: V5CheckpointObservation | None = None,
    records: tuple[V5CheckpointInput, ...] | None = None,
) -> object:
    return publish_v5_checkpoint(
        output_root=output_root,
        stage=stage,
        run_binding_sha256="4" * 64,
        attempt_receipt_sha256="5" * 64,
        predecessor=predecessor,
        records=records
        or (
            V5CheckpointInput("row", "event-1", b'{"x":1}'),
            V5CheckpointInput("row", "event-2", b'{"x":2}'),
        ),
        environment=(observation.environment if observation else _environment()),
        observation=observation or _observation(),
        limits=_limits(),
    )


def test_checkpoint_is_manifest_last_content_addressed_and_reconstructable(
    tmp_path: Path,
) -> None:
    """A visible manifest must authenticate all durable record and observation bytes."""
    output = tmp_path / "out"
    manifest = _publish(output)

    assert (output / "checkpoint.manifest.json").is_file()
    assert manifest.stage is V5KaggleStage.AUTHORIZE
    assert manifest.record_count == 2
    assert manifest.predecessor_manifest_sha256 is None
    assert manifest.predecessor_deterministic_sha256 is None
    assert tuple(
        (item.kind, item.key, item.canonical_bytes)
        for item in iter_v5_checkpoint_records(output_root=output, limits=_limits())
    ) == (
        ("row", "event-1", b'{"x":1}'),
        ("row", "event-2", b'{"x":2}'),
    )
    assert read_v5_checkpoint_manifest(
        output_root=output, limits=_limits()
    ) == manifest
    assert read_v5_checkpoint_observation(
        output_root=output, limits=_limits()
    ).peak_rss_bytes == 200_000


def test_empty_observational_layer_survives_file_only_checkpoint_transport(
    tmp_path: Path,
) -> None:
    """A file-only notebook output must retain an authenticated empty layer."""
    published = tmp_path / "published"
    manifest = _publish(published)
    marker = published / "observational-chunks" / "empty-layer.json"

    assert manifest.observational_record_count == 0
    assert marker.read_bytes() == (
        b'{"layer":"observational","record_count":0,'
        b'"record_stream_sha256":'
        b'"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",'
        b'"schema_version":"apar-sentinel-v5-kaggle-empty-checkpoint-layer/1"}'
    )

    transported = tmp_path / "transported"
    for source in published.rglob("*"):
        if source.is_file():
            destination = transported / source.relative_to(published)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(source.read_bytes())

    assert read_v5_checkpoint_manifest(
        output_root=transported,
        limits=_limits(),
    ) == manifest


@pytest.mark.parametrize("mutation", ["tampered", "symlink", "hardlink"])
def test_empty_observational_layer_marker_rejects_substitution(
    tmp_path: Path,
    mutation: str,
) -> None:
    """The portability marker cannot become an unauthenticated substitute."""
    output = tmp_path / "out"
    _publish(output)
    marker = output / "observational-chunks" / "empty-layer.json"
    if mutation == "tampered":
        marker.write_bytes(b"{}")
    else:
        real = output / "real-empty-layer.json"
        marker.rename(real)
        if mutation == "symlink":
            marker.symlink_to(real)
        else:
            os.link(real, marker)

    with pytest.raises(ValueError, match="empty observational checkpoint marker"):
        read_v5_checkpoint_manifest(output_root=output, limits=_limits())


def test_deterministic_digest_excludes_only_authenticated_observation(
    tmp_path: Path,
) -> None:
    """Real timings may vary while deterministic evaluation evidence stays identical."""
    first = _publish(tmp_path / "first")
    second = _publish(
        tmp_path / "second",
        observation=_observation(
            started="2026-08-24T00:10:00Z",
            completed="2026-08-24T00:10:02Z",
            wall_seconds=2.0,
            rss=(300_000, 400_000),
        ),
    )

    assert first.deterministic_sha256 == second.deterministic_sha256
    assert first.observation_sha256 != second.observation_sha256
    assert first.manifest_sha256 != second.manifest_sha256


def test_observational_record_stream_is_authenticated_but_not_deterministic(
    tmp_path: Path,
) -> None:
    """Real row timings cannot perturb deterministic actions or model evidence."""
    first = _publish(
        tmp_path / "first",
        records=(
            V5CheckpointInput("arm_core", "full_sentinel", b'{"action":"allow"}'),
            V5CheckpointInput(
                "arm_latency",
                "full_sentinel",
                b'{"latency_ms":1.25}',
                layer="observational",
            ),
        ),
    )
    second = _publish(
        tmp_path / "second",
        records=(
            V5CheckpointInput("arm_core", "full_sentinel", b'{"action":"allow"}'),
            V5CheckpointInput(
                "arm_latency",
                "full_sentinel",
                b'{"latency_ms":2.5}',
                layer="observational",
            ),
        ),
    )

    assert first.deterministic_sha256 == second.deterministic_sha256
    assert first.observational_record_stream_sha256 != (
        second.observational_record_stream_sha256
    )
    assert [
        item.canonical_bytes
        for item in iter_v5_checkpoint_records(
            output_root=tmp_path / "first", limits=_limits()
        )
    ] == [b'{"action":"allow"}']
    assert [
        item.canonical_bytes
        for item in iter_v5_checkpoint_observational_records(
            output_root=tmp_path / "first", limits=_limits()
        )
    ] == [b'{"latency_ms":1.25}']


def test_observational_record_chunk_tampering_is_rejected(tmp_path: Path) -> None:
    """Separating latency from deterministic evidence must not make it optional."""
    output = tmp_path / "out"
    manifest = _publish(
        output,
        records=(
            V5CheckpointInput("arm_core", "full_sentinel", b'{"action":"allow"}'),
            V5CheckpointInput(
                "arm_latency",
                "full_sentinel",
                b'{"latency_ms":1.25}',
                layer="observational",
            ),
        ),
    )
    target = output / "observational-chunks" / (
        manifest.observational_chunks[0].filename
    )
    target.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="observational|chunk"):
        tuple(
            iter_v5_checkpoint_observational_records(
                output_root=output, limits=_limits()
            )
        )


def test_predecessor_lineage_binds_both_manifest_and_deterministic_evidence(
    tmp_path: Path,
) -> None:
    """A later stage cannot detach from or disguise its immediate predecessor."""
    authorization = _publish(tmp_path / "authorization")
    corpus = _publish(
        tmp_path / "corpus",
        stage=V5KaggleStage.CORPUS,
        predecessor=authorization,
    )
    assert corpus.predecessor_stage is V5KaggleStage.AUTHORIZE
    assert corpus.predecessor_manifest_sha256 == authorization.manifest_sha256
    assert corpus.predecessor_deterministic_sha256 == (
        authorization.deterministic_sha256
    )


@pytest.mark.parametrize(
    "mutation",
    [
        "missing_chunk",
        "mutated_chunk",
        "extra_chunk",
        "missing_observation",
        "mutated_observation",
        "mutated_manifest",
    ],
)
def test_checkpoint_reader_rejects_deleted_extra_or_tampered_evidence(
    tmp_path: Path,
    mutation: str,
) -> None:
    """Every published byte is mandatory and authenticated."""
    output = tmp_path / "out"
    manifest = _publish(output)
    chunk = output / "chunks" / manifest.chunks[0].filename
    observation = output / "observational.json"
    manifest_path = output / "checkpoint.manifest.json"
    if mutation == "missing_chunk":
        chunk.unlink()
    elif mutation == "mutated_chunk":
        chunk.write_bytes(b"tampered")
    elif mutation == "extra_chunk":
        (output / "chunks" / "part-9999.bin").write_bytes(b"extra")
    elif mutation == "missing_observation":
        observation.unlink()
    elif mutation == "mutated_observation":
        document = json.loads(observation.read_bytes())
        document["peak_rss_bytes"] += 1
        observation.write_text(json.dumps(document), encoding="utf-8")
    elif mutation == "mutated_manifest":
        document = json.loads(manifest_path.read_bytes())
        document["attempt_receipt_sha256"] = "9" * 64
        manifest_path.write_text(json.dumps(document), encoding="utf-8")
    else:
        raise AssertionError(mutation)
    with pytest.raises((ValueError, FileNotFoundError)):
        tuple(iter_v5_checkpoint_records(output_root=output, limits=_limits()))


@pytest.mark.parametrize("existing", ["file", "directory", "symlink"])
def test_checkpoint_publication_refuses_any_existing_output_root(
    tmp_path: Path,
    existing: str,
) -> None:
    """Publication never repairs, replaces, or writes through an existing target."""
    output = tmp_path / "out"
    if existing == "file":
        output.write_bytes(b"occupied")
    elif existing == "directory":
        output.mkdir()
        (output / "partial").write_bytes(b"partial")
    else:
        source = tmp_path / "source"
        source.mkdir()
        output.symlink_to(source, target_is_directory=True)
    with pytest.raises((FileExistsError, ValueError)):
        _publish(output)


@pytest.mark.parametrize("target_kind", ["symlink", "hardlink"])
def test_checkpoint_reader_rejects_linked_manifest(
    tmp_path: Path,
    target_kind: str,
) -> None:
    """A manifest alias cannot substitute for the one saved checkpoint object."""
    output = tmp_path / "out"
    _publish(output)
    manifest = output / "checkpoint.manifest.json"
    real = output / "real-manifest.json"
    manifest.rename(real)
    if target_kind == "symlink":
        manifest.symlink_to(real.name)
    else:
        os.link(real, manifest)
    with pytest.raises(ValueError, match="single-link regular file"):
        read_v5_checkpoint_manifest(output_root=output, limits=_limits())


@pytest.mark.parametrize(
    "observation",
    [
        _observation(wall_seconds=61.0),
        _observation(rss=(1_000_001,)),
    ],
)
def test_checkpoint_publication_rejects_resource_gate_violation(
    tmp_path: Path,
    observation: V5CheckpointObservation,
) -> None:
    """A stage exceeding a frozen time or memory gate cannot publish completion."""
    with pytest.raises(ValueError, match="resource gate"):
        _publish(tmp_path / "out", observation=observation)


def test_checkpoint_input_rejects_empty_or_noncanonical_identity() -> None:
    """Malformed record identities cannot create ambiguous stream entries."""
    with pytest.raises(ValidationError):
        V5CheckpointInput("", "event-1", b"{}")
    with pytest.raises(ValidationError):
        V5CheckpointInput("row", "../event-1", b"{}")
    with pytest.raises(ValidationError):
        V5CheckpointInput("row", "event-1", b"")
