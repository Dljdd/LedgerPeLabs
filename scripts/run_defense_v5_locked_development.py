"""Build and atomically publish the one-time locked Sentinel v5 evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from datetime import UTC, datetime
from importlib import import_module
from pathlib import Path
from typing import Protocol

from apar.evaluation.v5_evidence_protocol import (
    V5LockedArtifactStorage,
    load_v5_evidence_protocol,
)
from apar.evaluation.v5_evidence_storage import (
    V5ChunkedEvidenceManifest,
    publish_v5_chunked_evidence,
)
from apar.evaluation.v5_locked_evidence import build_v5_locked_evidence_payload
from apar.evaluation.v5_run_mode import V5LockedEvidenceRunBinding, V5RunMode
from apar.v5_independent_verifier import (
    read_locked_evidence_storage_bytes,
    verify_locked_evidence_payload_bytes,
)

_complete_module = import_module(
    f"{__package__}.v5_complete_evidence_execution"
    if __package__
    else "v5_complete_evidence_execution"
)


class _LockedExecutionAuthority(Protocol):
    def preflight(
        self, *, root: Path, safe_evidence: Path, approved_commit: str
    ) -> V5LockedEvidenceRunBinding: ...

    def build_payload(
        self, *, root: Path, binding: V5LockedEvidenceRunBinding
    ) -> bytes: ...

    def verify_payload(self, *, root: Path, payload: bytes) -> dict[str, object]: ...

    def storage(self, *, root: Path) -> V5LockedArtifactStorage: ...

    def verify_published(
        self,
        *,
        root: Path,
        target: Path,
        storage: V5LockedArtifactStorage,
    ) -> dict[str, object]: ...


class _ProductionLockedExecutionAuthority:
    def preflight(
        self, *, root: Path, safe_evidence: Path, approved_commit: str
    ) -> V5LockedEvidenceRunBinding:
        preexecution = import_module(
            f"{__package__}.verify_defense_v5_locked_preexecution"
            if __package__
            else "verify_defense_v5_locked_preexecution"
        )
        report = preexecution.verify_locked_preexecution(
            root=root,
            safe_evidence=safe_evidence,
            approved_commit=approved_commit,
        )
        return V5LockedEvidenceRunBinding.model_validate(report["run_binding"])

    def build_payload(
        self, *, root: Path, binding: V5LockedEvidenceRunBinding
    ) -> bytes:
        executed = _complete_module.execute_v5_complete_evidence(
            root=root,
            mode=V5RunMode.LOCKED_DEVELOPMENT,
            locked_capability=_complete_module._issue_locked_execution_capability(
                binding
            ),
        )
        return build_v5_locked_evidence_payload(
            run_binding=binding,
            evidence_protocol=executed.evidence_protocol,
            catalog_sha256=executed.catalog.catalog_sha256,
            arm_results=executed.arm_results,
            controls=executed.controls,
        )

    def verify_payload(self, *, root: Path, payload: bytes) -> dict[str, object]:
        return verify_locked_evidence_payload_bytes(payload, root=root)

    def storage(self, *, root: Path) -> V5LockedArtifactStorage:
        return load_v5_evidence_protocol(
            root / "config/defense/defense-v5-evidence.json", root=root
        ).locked_artifact_storage

    def verify_published(
        self,
        *,
        root: Path,
        target: Path,
        storage: V5LockedArtifactStorage,
    ) -> dict[str, object]:
        payload = read_locked_evidence_storage_bytes(
            target_manifest=target,
            chunk_size_bytes=storage.chunk_size_bytes,
            maximum_envelope_bytes=storage.maximum_envelope_bytes,
            maximum_chunk_count=storage.maximum_chunk_count,
            normal_git_blob_limit_bytes=storage.normal_git_blob_limit_bytes,
        )
        return verify_locked_evidence_payload_bytes(payload, root=root)


def _utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _write_summary_exclusive(path: Path, document: dict[str, object]) -> None:
    serialized = json.dumps(
        document,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode()
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o600)
    try:
        view = memoryview(serialized)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("exclusive summary write made no progress")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _execute_locked_development_once(
    *,
    root: Path,
    safe_evidence: Path,
    approved_commit: str,
    authorization_granted: bool,
    authority: _LockedExecutionAuthority,
) -> V5ChunkedEvidenceManifest:
    """Execute one authorized run; the public CLI supplies only production authority."""
    if not authorization_granted:
        raise PermissionError("explicit one-time authorization is required")
    root = root.resolve()
    binding = authority.preflight(
        root=root,
        safe_evidence=safe_evidence.resolve(),
        approved_commit=approved_commit,
    )
    if binding.mode is not V5RunMode.LOCKED_DEVELOPMENT:
        raise ValueError("execution authority did not bind locked development mode")
    target = root / binding.candidate_manifest_path
    chunks = target.with_name(f"{target.name}.chunks")
    if os.path.lexists(target):
        raise FileExistsError("candidate manifest already exists")
    if os.path.lexists(chunks):
        raise FileExistsError("candidate chunk directory already exists")
    storage = authority.storage(root=root)
    summary_target = root / storage.judge_summary_path
    if os.path.lexists(summary_target):
        raise FileExistsError("judge summary already exists")
    started_at = _utc_now()
    start_ns = time.perf_counter_ns()
    payload = authority.build_payload(root=root, binding=binding)
    verification = authority.verify_payload(root=root, payload=payload)
    if verification.get("verified") is not True:
        raise ValueError("independent locked payload verification failed")
    completed_at = _utc_now()
    elapsed_ms = max((time.perf_counter_ns() - start_ns) / 1_000_000.0, 0.000001)
    authorization_sha256 = hashlib.sha256(
        json.dumps(
            {
                "authorization": "execute-exactly-once-locked-development",
                "approved_commit": approved_commit,
                "run_binding_sha256": binding.run_binding_sha256,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    manifest = publish_v5_chunked_evidence(
        payload_bytes=payload,
        target_manifest=target,
        run_binding_sha256=binding.run_binding_sha256,
        authorization_sha256=authorization_sha256,
        started_at_utc=started_at,
        completed_at_utc=completed_at,
        elapsed_ms=elapsed_ms,
        chunk_size_bytes=storage.chunk_size_bytes,
        maximum_envelope_bytes=storage.maximum_envelope_bytes,
        maximum_chunk_count=storage.maximum_chunk_count,
        normal_git_blob_limit_bytes=storage.normal_git_blob_limit_bytes,
    )
    published = authority.verify_published(
        root=root, target=target, storage=storage
    )
    if published.get("verified") is not True:
        raise ValueError("published locked evidence verification failed")
    summary: dict[str, object] = {
        "schema_version": "apar-sentinel-v5-locked-judge-summary/1",
        "candidate_manifest_path": binding.candidate_manifest_path,
        "manifest_sha256": manifest.manifest_sha256,
        "payload_sha256": manifest.payload_sha256,
        "run_binding_sha256": binding.run_binding_sha256,
        "verification": published,
    }
    summary["summary_sha256"] = hashlib.sha256(
        json.dumps(summary, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    _write_summary_exclusive(summary_target, summary)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--safe-evidence", type=Path, required=True)
    parser.add_argument("--approved-commit", required=True)
    parser.add_argument("--authorize-exactly-once", action="store_true")
    args = parser.parse_args()
    manifest = _execute_locked_development_once(
        root=args.root,
        safe_evidence=args.safe_evidence,
        approved_commit=args.approved_commit,
        authorization_granted=args.authorize_exactly_once,
        authority=_ProductionLockedExecutionAuthority(),
    )
    print(
        json.dumps(
            {
                "published": True,
                "manifest_sha256": manifest.manifest_sha256,
                "payload_sha256": manifest.payload_sha256,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
