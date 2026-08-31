from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.submission.model import ReleaseError, canonical_json, sha256_bytes
from scripts.submission.runtime_verify import verify_payload_manifest


def _write_release(root: Path) -> None:
    (root / "payload.txt").write_text("bound\n")
    payload = (root / "payload.txt").read_bytes()
    manifest = {
        "accepted_model": {"arm": "ensemble_with_graph"},
        "evidence_authority": {
            "accepted_capacity_evidence": False,
            "authoritative": False,
            "official_chain_complete": False,
            "production_ready": False,
            "real_cardholder_data": False,
        },
        "files": [
            {"path": "payload.txt", "sha256": sha256_bytes(payload), "size": len(payload)}
        ],
        "schema_version": "apar-submission-manifest/1",
        "web": {"status": "pending"},
    }
    manifest["deterministic_core_sha256"] = sha256_bytes(canonical_json(manifest))
    (root / "SUBMISSION_MANIFEST.json").write_bytes(canonical_json(manifest))


def test_runtime_verifier_checks_manifest_and_payload_hashes(tmp_path: Path) -> None:
    """A judge-side verifier that trusts extracted bytes would miss archive tampering."""
    _write_release(tmp_path)

    manifest = verify_payload_manifest(tmp_path)

    assert manifest["accepted_model"]["arm"] == "ensemble_with_graph"
    (tmp_path / "payload.txt").write_text("tampered\n")
    with pytest.raises(ReleaseError, match="payload digest differs"):
        verify_payload_manifest(tmp_path)


def test_runtime_verifier_rejects_unsafe_authority_flags(tmp_path: Path) -> None:
    """A promoted authority flag must invalidate the release even if hashes are recomputed."""
    _write_release(tmp_path)
    manifest_path = tmp_path / "SUBMISSION_MANIFEST.json"
    manifest = json.loads(manifest_path.read_bytes())
    manifest.pop("deterministic_core_sha256")
    manifest["evidence_authority"]["production_ready"] = True
    manifest["deterministic_core_sha256"] = sha256_bytes(canonical_json(manifest))
    manifest_path.write_bytes(canonical_json(manifest))

    with pytest.raises(ReleaseError, match="unsafe evidence authority flag"):
        verify_payload_manifest(tmp_path)
