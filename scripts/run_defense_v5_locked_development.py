"""Build and atomically publish the one-time locked Sentinel v5 evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from dataclasses import dataclass
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
    V5LockedAttemptReceipt,
    build_v5_locked_attempt_receipt,
    publish_v5_chunked_evidence,
    publish_v5_locked_attempt_receipt,
)
from apar.evaluation.v5_locked_evidence import build_v5_locked_evidence_payload
from apar.evaluation.v5_run_mode import V5LockedEvidenceRunBinding, V5RunMode
from apar.v5_independent_verifier import (
    read_locked_evidence_storage_bytes,
    verify_locked_evidence_payload_bytes,
    verify_locked_judge_summary,
)

_complete_module = import_module(
    f"{__package__}.v5_complete_evidence_execution"
    if __package__
    else "v5_complete_evidence_execution"
)


@dataclass(frozen=True, slots=True)
class _LockedPreexecutionAuthorization:
    binding: V5LockedEvidenceRunBinding
    approved_safe_deterministic_core_sha256: str
    approved_safe_observational_environment_sha256: str
    exact_command: str

    def __post_init__(self) -> None:
        for value in (
            self.approved_safe_deterministic_core_sha256,
            self.approved_safe_observational_environment_sha256,
        ):
            if len(value) != 64 or any(
                character not in "0123456789abcdef" for character in value
            ):
                raise ValueError("preexecution safe evidence digest is invalid")
        if not self.exact_command:
            raise ValueError("preexecution exact command is empty")


class _LockedExecutionAuthority(Protocol):
    def preflight(
        self, *, root: Path, safe_evidence: Path, approved_commit: str
    ) -> _LockedPreexecutionAuthorization: ...

    def build_payload(
        self,
        *,
        root: Path,
        binding: V5LockedEvidenceRunBinding,
        attempt_receipt: V5LockedAttemptReceipt,
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

    def verify_summary(
        self,
        *,
        root: Path,
        summary_target: Path,
        target: Path,
        storage: V5LockedArtifactStorage,
        published: dict[str, object],
    ) -> dict[str, object]: ...


class _ProductionLockedExecutionAuthority:
    def preflight(
        self, *, root: Path, safe_evidence: Path, approved_commit: str
    ) -> _LockedPreexecutionAuthorization:
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
        return _LockedPreexecutionAuthorization(
            binding=V5LockedEvidenceRunBinding.model_validate(
                report["run_binding"]
            ),
            approved_safe_deterministic_core_sha256=str(
                report["safe_deterministic_core_sha256"]
            ),
            approved_safe_observational_environment_sha256=str(
                report["safe_observational_environment_sha256"]
            ),
            exact_command=str(report["exact_command"]),
        )

    def build_payload(
        self,
        *,
        root: Path,
        binding: V5LockedEvidenceRunBinding,
        attempt_receipt: V5LockedAttemptReceipt,
    ) -> bytes:
        executed = _complete_module.execute_v5_complete_evidence(
            root=root,
            mode=V5RunMode.LOCKED_DEVELOPMENT,
            locked_capability=_complete_module._issue_locked_execution_capability(
                binding, attempt_receipt, root=root
            ),
        )
        return build_v5_locked_evidence_payload(
            run_binding=binding,
            attempt_receipt_sha256=attempt_receipt.receipt_sha256,
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
            attempt_receipt_path=root / storage.attempt_receipt_path,
            chunk_size_bytes=storage.chunk_size_bytes,
            maximum_envelope_bytes=storage.maximum_envelope_bytes,
            maximum_chunk_count=storage.maximum_chunk_count,
            normal_git_blob_limit_bytes=storage.normal_git_blob_limit_bytes,
        )
        return verify_locked_evidence_payload_bytes(payload, root=root)

    def verify_summary(
        self,
        *,
        root: Path,
        summary_target: Path,
        target: Path,
        storage: V5LockedArtifactStorage,
        published: dict[str, object],
    ) -> dict[str, object]:
        verify_locked_judge_summary(
            summary_path=summary_target,
            target_manifest=target,
            attempt_receipt_path=root / storage.attempt_receipt_path,
            verification=published,
            candidate_manifest_path=storage.candidate_manifest_path,
            declared_attempt_receipt_path=storage.attempt_receipt_path,
        )
        return {"verified": True, "status": "locked_summary_verified"}


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
    directory_descriptor = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory_descriptor)
    finally:
        os.close(directory_descriptor)


def _authorization_digest(
    *, binding: V5LockedEvidenceRunBinding, exact_command: str
) -> str:
    return hashlib.sha256(
        json.dumps(
            {
                "authorization": "execute-exactly-once-locked-development",
                "preregistration_commit": binding.preregistration_commit,
                "run_binding_sha256": binding.run_binding_sha256,
                "exact_command": exact_command,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()


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
    preexecution = authority.preflight(
        root=root,
        safe_evidence=safe_evidence.resolve(),
        approved_commit=approved_commit,
    )
    binding = preexecution.binding
    if binding.mode is not V5RunMode.LOCKED_DEVELOPMENT:
        raise ValueError("execution authority did not bind locked development mode")
    storage = authority.storage(root=root)
    target = root / binding.candidate_manifest_path
    chunks = target.with_name(f"{target.name}.chunks")
    attempt_target = root / storage.attempt_receipt_path
    if os.path.lexists(attempt_target):
        raise FileExistsError("locked attempt receipt already exists")
    if os.path.lexists(target):
        raise FileExistsError("candidate manifest already exists")
    if os.path.lexists(chunks):
        raise FileExistsError("candidate chunk directory already exists")
    summary_target = root / storage.judge_summary_path
    if os.path.lexists(summary_target):
        raise FileExistsError("judge summary already exists")
    started_at = _utc_now()
    authorization_sha256 = _authorization_digest(
        binding=binding, exact_command=preexecution.exact_command
    )
    attempt_receipt = build_v5_locked_attempt_receipt(
        run_binding_sha256=binding.run_binding_sha256,
        preregistration_commit=binding.preregistration_commit,
        preregistration_sha256=binding.preregistration_sha256,
        source_commit=binding.source_commit,
        source_tree_oid=binding.source_tree_oid,
        approved_safe_deterministic_core_sha256=(
            preexecution.approved_safe_deterministic_core_sha256
        ),
        approved_safe_observational_environment_sha256=(
            preexecution.approved_safe_observational_environment_sha256
        ),
        authorization_sha256=authorization_sha256,
        exact_command=preexecution.exact_command,
        started_at_utc=started_at,
    )
    publish_v5_locked_attempt_receipt(
        target=attempt_target, receipt=attempt_receipt
    )
    start_ns = time.perf_counter_ns()
    payload = authority.build_payload(
        root=root,
        binding=binding,
        attempt_receipt=attempt_receipt,
    )
    verification = authority.verify_payload(root=root, payload=payload)
    if verification.get("verified") is not True:
        raise ValueError("independent locked payload verification failed")
    completed_at = _utc_now()
    elapsed_ms = max((time.perf_counter_ns() - start_ns) / 1_000_000.0, 0.000001)
    manifest = publish_v5_chunked_evidence(
        payload_bytes=payload,
        target_manifest=target,
        run_binding_sha256=binding.run_binding_sha256,
        attempt_receipt=attempt_receipt,
        attempt_receipt_path=storage.attempt_receipt_path,
        attempt_receipt_target=attempt_target,
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
        "schema_version": "apar-sentinel-v5-locked-judge-summary/2",
        "candidate_manifest_path": binding.candidate_manifest_path,
        "manifest_sha256": manifest.manifest_sha256,
        "payload_sha256": manifest.payload_sha256,
        "run_binding_sha256": binding.run_binding_sha256,
        "attempt_receipt_path": storage.attempt_receipt_path,
        "attempt_receipt_sha256": attempt_receipt.receipt_sha256,
        "verification": published,
    }
    summary["summary_sha256"] = hashlib.sha256(
        json.dumps(summary, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    _write_summary_exclusive(summary_target, summary)
    summary_verification = authority.verify_summary(
        root=root,
        summary_target=summary_target,
        target=target,
        storage=storage,
        published=published,
    )
    if summary_verification.get("verified") is not True:
        raise ValueError("published locked summary verification failed")
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
                "attempt_receipt_sha256": (
                    manifest.attempt_receipt_sha256
                ),
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
