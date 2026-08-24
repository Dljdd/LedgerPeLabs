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
    from apar.evaluation.v5_evidence_storage import (
        build_v5_locked_attempt_receipt,
        publish_v5_chunked_evidence,
        publish_v5_locked_attempt_receipt,
    )

    attempt = build_v5_locked_attempt_receipt(
        run_binding_sha256="1" * 64,
        preregistration_commit="2" * 40,
        preregistration_sha256="3" * 64,
        source_commit="4" * 40,
        source_tree_oid="5" * 40,
        approved_safe_deterministic_core_sha256="6" * 64,
        approved_safe_observational_environment_sha256="7" * 64,
        authorization_sha256="8" * 64,
        exact_command="locked-test-command",
        started_at_utc="2026-08-24T00:00:00Z",
    )
    attempt_path = target.with_name("attempt.json")
    publish_v5_locked_attempt_receipt(
        target=attempt_path, receipt=attempt
    )

    return publish_v5_chunked_evidence(
        payload_bytes=payload,
        target_manifest=target,
        run_binding_sha256="1" * 64,
        attempt_receipt=attempt,
        attempt_receipt_path=attempt_path.name,
        attempt_receipt_target=attempt_path,
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


def test_chunked_publication_requires_visible_durable_attempt(
    tmp_path: Path,
) -> None:
    """A valid in-memory receipt cannot substitute for its durable target."""
    from apar.evaluation.v5_evidence_storage import (
        build_v5_locked_attempt_receipt,
        publish_v5_chunked_evidence,
    )

    attempt = build_v5_locked_attempt_receipt(
        run_binding_sha256="1" * 64,
        preregistration_commit="2" * 40,
        preregistration_sha256="3" * 64,
        source_commit="4" * 40,
        source_tree_oid="5" * 40,
        approved_safe_deterministic_core_sha256="6" * 64,
        approved_safe_observational_environment_sha256="7" * 64,
        authorization_sha256="8" * 64,
        exact_command="locked-test-command",
        started_at_utc="2026-08-24T00:00:00Z",
    )
    target = tmp_path / "candidate.manifest.json"
    with pytest.raises(FileNotFoundError):
        publish_v5_chunked_evidence(
            payload_bytes=b"payload",
            target_manifest=target,
            run_binding_sha256="1" * 64,
            attempt_receipt=attempt,
            attempt_receipt_path="attempt.json",
            attempt_receipt_target=tmp_path / "attempt.json",
            completed_at_utc="2026-08-24T00:00:01Z",
            elapsed_ms=1000.0,
            **_limits(),
        )
    assert not target.exists()
    assert not target.with_name(f"{target.name}.chunks").exists()


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
        target_manifest=target,
        attempt_receipt_path=tmp_path / "attempt.json",
        **_limits(),
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
        read_locked_evidence_storage_bytes(
            target_manifest=target,
            attempt_receipt_path=tmp_path / "attempt.json",
            **_limits(),
        )


@pytest.mark.parametrize("mutation", ["missing", "different_valid_receipt"])
def test_independent_storage_rejects_missing_or_substituted_attempt(
    tmp_path: Path, mutation: str
) -> None:
    from apar.evaluation.v5_evidence_storage import (
        build_v5_locked_attempt_receipt,
        publish_v5_locked_attempt_receipt,
    )
    from apar.v5_independent_verifier import (
        IndependentVerificationError,
        read_locked_evidence_storage_bytes,
    )

    target = tmp_path / "candidate.manifest.json"
    _publish(target)
    attempt_path = tmp_path / "attempt.json"
    attempt_path.unlink()
    if mutation == "different_valid_receipt":
        replacement = build_v5_locked_attempt_receipt(
            run_binding_sha256="1" * 64,
            preregistration_commit="2" * 40,
            preregistration_sha256="3" * 64,
            source_commit="4" * 40,
            source_tree_oid="5" * 40,
            approved_safe_deterministic_core_sha256="6" * 64,
            approved_safe_observational_environment_sha256="7" * 64,
            authorization_sha256="8" * 64,
            exact_command="locked-test-command",
            started_at_utc="2026-08-24T00:00:02Z",
        )
        publish_v5_locked_attempt_receipt(
            target=attempt_path, receipt=replacement
        )
    with pytest.raises(
        IndependentVerificationError, match="attempt receipt|receipt lineage"
    ):
        read_locked_evidence_storage_bytes(
            target_manifest=target,
            attempt_receipt_path=attempt_path,
            **_limits(),
        )


@pytest.mark.parametrize(
    "mutation", ["attempt", "manifest", "payload", "verification"]
)
def test_independent_verifier_rejects_rebound_judge_summary(
    tmp_path: Path, mutation: str
) -> None:
    from apar.v5_independent_verifier import (
        IndependentVerificationError,
        verify_locked_judge_summary,
    )
    from scripts.run_defense_v5_locked_development import (
        _write_summary_exclusive,
    )

    target = tmp_path / "candidate.manifest.json"
    manifest = _publish(target)
    verification = {"verified": True, "status": "test_only"}
    values: dict[str, object] = {
        "schema_version": "apar-sentinel-v5-locked-judge-summary/2",
        "candidate_manifest_path": target.name,
        "manifest_sha256": manifest.manifest_sha256,
        "payload_sha256": manifest.payload_sha256,
        "run_binding_sha256": manifest.run_binding_sha256,
        "attempt_receipt_path": "attempt.json",
        "attempt_receipt_sha256": manifest.attempt_receipt_sha256,
        "verification": verification,
    }
    values["summary_sha256"] = hashlib.sha256(
        json.dumps(
            values, sort_keys=True, separators=(",", ":")
        ).encode()
    ).hexdigest()
    summary = tmp_path / "summary.json"
    _write_summary_exclusive(summary, values)
    verify_locked_judge_summary(
        summary_path=summary,
        target_manifest=target,
        attempt_receipt_path=tmp_path / "attempt.json",
        verification=verification,
        candidate_manifest_path=target.name,
        declared_attempt_receipt_path="attempt.json",
    )

    document = json.loads(summary.read_bytes())
    field = {
        "attempt": "attempt_receipt_sha256",
        "manifest": "manifest_sha256",
        "payload": "payload_sha256",
        "verification": "verification",
    }[mutation]
    document[field] = (
        {"verified": False, "status": "rebound"}
        if mutation == "verification"
        else "9" * 64
    )
    document["summary_sha256"] = hashlib.sha256(
        json.dumps(
            {
                key: value
                for key, value in document.items()
                if key != "summary_sha256"
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    summary.write_text(
        json.dumps(document, sort_keys=True, separators=(",", ":"))
    )
    with pytest.raises(
        IndependentVerificationError, match="summary evidence binding"
    ):
        verify_locked_judge_summary(
            summary_path=summary,
            target_manifest=target,
            attempt_receipt_path=tmp_path / "attempt.json",
            verification=verification,
            candidate_manifest_path=target.name,
            declared_attempt_receipt_path="attempt.json",
        )
