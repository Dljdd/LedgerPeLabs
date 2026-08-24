"""Durable chunked storage contracts for one locked development artifact."""

from __future__ import annotations

import ast
import hashlib
import json
import os
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def _limits() -> dict[str, int]:
    return {
        "chunk_size_bytes": 8,
        "maximum_envelope_bytes": 32,
        "maximum_chunk_count": 4,
        "normal_git_blob_limit_bytes": 100,
    }


def _publish(target: Path, payload: bytes = b"0123456789abcdef") -> object:
    from apar.evaluation.v5_evidence_storage import publish_v5_chunked_evidence

    return publish_v5_chunked_evidence(
        payload_bytes=payload,
        target_manifest=target,
        run_binding_sha256="1" * 64,
        authorization_sha256="2" * 64,
        started_at_utc="2026-08-24T00:00:00Z",
        completed_at_utc="2026-08-24T00:00:01Z",
        elapsed_ms=1000.0,
        **_limits(),
    )


def _rebind_manifest(path: Path, mutate: object) -> None:
    document = json.loads(path.read_text())
    assert isinstance(mutate, tuple)
    operation, value = mutate
    if operation == "reverse":
        document["chunks"] = list(reversed(document["chunks"]))
    elif operation == "payload_sha":
        document["payload_sha256"] = value
    else:
        raise AssertionError(operation)
    document["manifest_sha256"] = hashlib.sha256(
        json.dumps(
            {key: item for key, item in document.items() if key != "manifest_sha256"},
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode()
    ).hexdigest()
    path.write_text(
        json.dumps(document, sort_keys=True, separators=(",", ":"))
    )


def test_chunked_publication_is_exclusive_manifest_last_and_reconstructable(
    tmp_path: Path,
) -> None:
    """Writing the manifest before complete chunks would expose a partial result."""
    from apar.evaluation.v5_evidence_storage import reconstruct_v5_chunked_evidence

    target = tmp_path / "candidate.manifest.json"
    manifest = _publish(target)
    assert target.is_file()
    assert target.stat().st_nlink == 1
    assert manifest.publication == "content_chunks_then_atomic_exclusive_manifest"
    assert manifest.chunk_size_bytes == 8
    assert tuple(chunk.index for chunk in manifest.chunks) == (0, 1)
    assert reconstruct_v5_chunked_evidence(
        target_manifest=target, **_limits()
    ) == b"0123456789abcdef"


@pytest.mark.parametrize("existing_kind", ["file", "directory", "symlink", "hardlink"])
def test_chunked_publication_rejects_existing_or_linked_target(
    tmp_path: Path, existing_kind: str
) -> None:
    """Removing exclusive target checks would permit overwrite or alias publication."""
    target = tmp_path / "candidate.manifest.json"
    if existing_kind == "file":
        target.write_text("preserve")
    elif existing_kind == "directory":
        target.mkdir()
    elif existing_kind == "symlink":
        source = tmp_path / "source"
        source.write_text("preserve")
        target.symlink_to(source)
    else:
        source = tmp_path / "source"
        source.write_text("preserve")
        os.link(source, target)
    with pytest.raises((FileExistsError, ValueError), match="exists|target|link"):
        _publish(target)


def test_chunked_publication_rejects_partial_chunk_directory(tmp_path: Path) -> None:
    """A stale partial attempt must block any implicit resume or retry path."""
    target = tmp_path / "candidate.manifest.json"
    chunks = tmp_path / "candidate.manifest.json.chunks"
    chunks.mkdir()
    (chunks / "part-0000.bin").write_bytes(b"partial")
    with pytest.raises((FileExistsError, ValueError), match="chunk|partial|exists"):
        _publish(target)


def test_chunked_publication_rejects_oversized_artifact(tmp_path: Path) -> None:
    """Dropping the hard envelope bound would allow unreviewed storage growth."""
    target = tmp_path / "candidate.manifest.json"
    with pytest.raises(ValueError, match="maximum envelope"):
        _publish(target, payload=b"x" * 33)
    assert not target.exists()


@pytest.mark.parametrize("mutation", ["missing", "reordered", "content", "manifest"])
def test_chunked_reconstruction_rejects_missing_reordered_or_tampered_data(
    tmp_path: Path, mutation: str
) -> None:
    """A reconstructed result must preserve exact chunk order and every digest."""
    from apar.evaluation.v5_evidence_storage import reconstruct_v5_chunked_evidence

    target = tmp_path / "candidate.manifest.json"
    _publish(target)
    chunks = tmp_path / "candidate.manifest.json.chunks"
    if mutation == "missing":
        (chunks / "part-0001.bin").unlink()
    elif mutation == "reordered":
        _rebind_manifest(target, ("reverse", None))
    elif mutation == "content":
        (chunks / "part-0000.bin").write_bytes(b"tampered")
    else:
        _rebind_manifest(target, ("payload_sha", "f" * 64))
    with pytest.raises(ValueError):
        reconstruct_v5_chunked_evidence(target_manifest=target, **_limits())


def test_chunked_reconstruction_rejects_extra_or_oversized_chunk(tmp_path: Path) -> None:
    """Unmanifested or oversized chunk bytes must never enter reconstruction."""
    from apar.evaluation.v5_evidence_storage import reconstruct_v5_chunked_evidence

    target = tmp_path / "candidate.manifest.json"
    _publish(target)
    chunks = tmp_path / "candidate.manifest.json.chunks"
    (chunks / "part-9999.bin").write_bytes(b"extra")
    with pytest.raises(ValueError, match="chunk set"):
        reconstruct_v5_chunked_evidence(target_manifest=target, **_limits())
    (chunks / "part-9999.bin").unlink()
    (chunks / "part-0000.bin").write_bytes(b"x" * 9)
    with pytest.raises(ValueError, match="chunk.*size|digest"):
        reconstruct_v5_chunked_evidence(target_manifest=target, **_limits())


def test_independent_verifier_reconstructs_without_production_storage_import(
    tmp_path: Path,
) -> None:
    """Importing the writer/reconstructor would make storage verification circular."""
    from apar.v5_independent_verifier import read_locked_evidence_storage_bytes

    target = tmp_path / "candidate.manifest.json"
    _publish(target)
    assert read_locked_evidence_storage_bytes(
        target_manifest=target, **_limits()
    ) == b"0123456789abcdef"

    tree = ast.parse((ROOT / "src/apar/v5_independent_verifier.py").read_text())
    imported = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    } | {
        node.module or ""
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
    }
    assert "apar.evaluation.v5_evidence_storage" not in imported


def test_independent_storage_reconstruction_rejects_tampered_chunk(
    tmp_path: Path,
) -> None:
    """The independent reader must recompute chunk bytes rather than trust the manifest."""
    from apar.v5_independent_verifier import (
        IndependentVerificationError,
        read_locked_evidence_storage_bytes,
    )

    target = tmp_path / "candidate.manifest.json"
    _publish(target)
    (tmp_path / "candidate.manifest.json.chunks/part-0000.bin").write_bytes(
        b"tampered"
    )
    with pytest.raises(IndependentVerificationError, match="chunk"):
        read_locked_evidence_storage_bytes(target_manifest=target, **_limits())
