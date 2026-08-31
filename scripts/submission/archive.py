"""Deterministic archive construction and self-contained manifest verification."""

from __future__ import annotations

import json
import os
import stat
import subprocess
import tempfile
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any, cast

from scripts.submission.model import (
    BuildResult,
    PolicyEntry,
    ReleaseError,
    canonical_json,
    require_safe_relative_path,
    sha256_bytes,
)
from scripts.submission.policy import load_policy
from scripts.submission.scan import scan_payloads

_MANIFEST = "SUBMISSION_MANIFEST.json"
_SCHEMA = "apar-submission-manifest/1"
_ZIP_TIME = (1980, 1, 1, 0, 0, 0)


def _git(root: Path, *args: str, text: bool = True) -> str | bytes:
    completed = subprocess.run(
        ("git", *args),
        cwd=root,
        check=False,
        capture_output=True,
        text=text,
    )
    if completed.returncode != 0:
        stderr = (
            completed.stderr
            if isinstance(completed.stderr, str)
            else completed.stderr.decode("utf-8", "replace")
        )
        raise ReleaseError(f"git command failed: {stderr.strip()}")
    stdout = completed.stdout
    if isinstance(stdout, (str, bytes)):
        return stdout
    raise ReleaseError("git command returned an unsupported payload")


def _index(root: Path) -> dict[str, tuple[str, str]]:
    raw = cast(bytes, _git(root, "ls-files", "--stage", "-z", text=False))
    result: dict[str, tuple[str, str]] = {}
    for record in raw.split(b"\x00"):
        if not record:
            continue
        try:
            metadata, raw_path = record.split(b"\t", 1)
            mode, oid, stage = metadata.decode("ascii").split(" ")
            path = raw_path.decode("utf-8")
        except (UnicodeDecodeError, ValueError) as error:
            raise ReleaseError("git index contains an unreadable entry") from error
        if stage != "0":
            raise ReleaseError(f"git index contains an unresolved entry: {path}")
        result[path] = (mode, oid)
    return result


def _payloads(
    root: Path,
    entries: tuple[PolicyEntry, ...],
    index: dict[str, tuple[str, str]],
) -> dict[str, bytes]:
    payloads: dict[str, bytes] = {}
    for entry in entries:
        tracked = index.get(entry.source)
        if tracked is None:
            if entry.required:
                raise ReleaseError(f"required tracked file is missing: {entry.source}")
            continue
        mode, oid = tracked
        if mode == "120000":
            raise ReleaseError(f"tracked symlink is forbidden: {entry.source}")
        if mode not in {"100644", "100755"}:
            raise ReleaseError(f"tracked entry is not a regular file: {entry.source}")
        payload = cast(bytes, _git(root, "cat-file", "blob", oid, text=False))
        payloads[entry.archive] = payload
    return payloads


def _validate_extensions(
    payloads: dict[str, bytes],
    *,
    allowed_extensions: frozenset[str],
    extensionless_paths: frozenset[str],
) -> None:
    for path in payloads:
        suffix = PurePosixPath(path).suffix.lower()
        if not suffix and path in extensionless_paths:
            continue
        if suffix not in allowed_extensions:
            raise ReleaseError(f"archive file extension is not allowlisted: {path}")


def _manifest(
    *,
    payloads: dict[str, bytes],
    release: dict[str, Any],
    source_commit: str,
    source_tree: str,
    web_status: str,
    scan_files: int,
    scan_total_bytes: int,
    scan_exemptions: int,
) -> dict[str, Any]:
    evidence_authority = release.get("evidence_authority")
    runtime = release.get("runtime")
    if not isinstance(evidence_authority, dict) or not isinstance(runtime, dict):
        raise ReleaseError("release authority or runtime metadata is absent")
    files = [
        {"path": path, "sha256": sha256_bytes(payload), "size": len(payload)}
        for path, payload in sorted(payloads.items())
    ]
    manifest: dict[str, Any] = {
        "accepted_model": {
            "arm": release.get("accepted_demo_arm"),
            "frozen_arm_spec_sha256": release.get("frozen_arm_spec_sha256"),
            "member_model_sha256": release.get("member_model_sha256"),
            "portable_bundle_manifest_sha256": release.get(
                "portable_bundle_manifest_sha256"
            ),
            "source_checkpoint_manifest_sha256": release.get(
                "source_checkpoint_manifest_sha256"
            ),
            "stage": release.get("accepted_stage"),
        },
        "build_command": release.get("build_command"),
        "evidence": {
            "baseline_commit": release.get("evidence_baseline_commit"),
            "first_missing_official_stage": release.get("first_missing_official_stage"),
            "locked_development_attempt": release.get("locked_development_attempt"),
            "recovered_metrics": release.get("recovered_metrics"),
        },
        "evidence_authority": evidence_authority,
        "files": files,
        "license": release.get("license"),
        "runtime": runtime,
        "runtime_verification": release.get("runtime_verification"),
        "scan": {
            "exemptions_applied": scan_exemptions,
            "files_scanned": scan_files,
            "total_bytes": scan_total_bytes,
        },
        "schema_version": _SCHEMA,
        "source": {"commit": source_commit, "tree": source_tree},
        "web": {
            "cold_start_verified": False,
            "included": web_status == "ready",
            "status": web_status,
        },
    }
    manifest["deterministic_core_sha256"] = sha256_bytes(canonical_json(manifest))
    return manifest


def _zip_info(path: str) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(path, _ZIP_TIME)
    info.compress_type = zipfile.ZIP_STORED
    info.create_system = 3
    info.external_attr = (stat.S_IFREG | 0o644) << 16
    info.flag_bits = 0x800
    return info


def build_archive(
    repo_root: Path,
    policy_path: Path,
    output_path: Path,
    *,
    include_web: bool = False,
) -> BuildResult:
    """Build a timestamp-free archive from exact regular files in the git index."""
    root = repo_root.resolve()
    output = output_path.resolve()
    if output.exists():
        raise ReleaseError(f"release output already exists: {output}")
    policy = load_policy(policy_path)
    index = _index(root)
    try:
        policy_relative = policy_path.resolve().relative_to(root).as_posix()
    except ValueError as error:
        raise ReleaseError("submission policy must be inside the repository") from error
    tracked_policy = index.get(policy_relative)
    if tracked_policy is None:
        raise ReleaseError("submission policy is not tracked")
    if policy_path.read_bytes() != cast(
        bytes, _git(root, "cat-file", "blob", tracked_policy[1], text=False)
    ):
        raise ReleaseError("submission policy differs from the tracked index")
    protected_prefixes = policy.release.get("protected_output_prefixes", [])
    if not isinstance(protected_prefixes, list) or any(
        not isinstance(item, str) for item in protected_prefixes
    ):
        raise ReleaseError("protected output prefixes must be a string list")
    for raw_prefix in protected_prefixes:
        prefix = require_safe_relative_path(raw_prefix, label="protected output prefix")
        protected = root.joinpath(*PurePosixPath(prefix).parts).resolve()
        if output == protected or protected in output.parents:
            raise ReleaseError(f"release output is under a protected canonical path: {prefix}")
    entries = policy.entries
    web_status = policy.web_status
    if include_web:
        if web_status != "ready":
            raise ReleaseError("web artifact integration is pending; no web files were packaged")
        entries += policy.web_entries
    elif web_status == "ready":
        web_status = "ready_not_requested"
    payloads = _payloads(root, entries, index)
    _validate_extensions(
        payloads,
        allowed_extensions=policy.allowed_extensions,
        extensionless_paths=policy.extensionless_paths,
    )
    scan = scan_payloads(
        payloads,
        allowed_emails=policy.scan_allowed_emails,
        exemptions=policy.scan_exemptions,
        max_file_bytes=policy.max_file_bytes,
        max_total_bytes=policy.max_total_bytes,
    )
    source_commit = cast(str, _git(root, "rev-parse", "HEAD")).strip()
    source_tree = cast(str, _git(root, "write-tree")).strip()
    manifest = _manifest(
        payloads=payloads,
        release=policy.release,
        source_commit=source_commit,
        source_tree=source_tree,
        web_status=web_status,
        scan_files=scan.files_scanned,
        scan_total_bytes=scan.total_bytes,
        scan_exemptions=scan.exemption_count,
    )
    manifest_bytes = canonical_json(manifest)
    archive_payloads = dict(payloads)
    archive_payloads[_MANIFEST] = manifest_bytes
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=f".{output.name}.", suffix=".tmp", dir=output.parent, delete=False
        ) as temporary:
            temporary_path = Path(temporary.name)
        with zipfile.ZipFile(temporary_path, "w", allowZip64=False) as archive:
            for relative, payload in sorted(archive_payloads.items()):
                archive.writestr(_zip_info(f"{policy.archive_root}/{relative}"), payload)
        os.replace(temporary_path, output)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    return BuildResult(
        archive_path=str(output),
        archive_sha256=sha256_bytes(output.read_bytes()),
        deterministic_core_sha256=cast(str, manifest["deterministic_core_sha256"]),
        source_commit=source_commit,
        source_tree=source_tree,
    )


def _safe_member(name: str) -> None:
    require_safe_relative_path(name, label="archive member")


def verify_archive(archive_path: Path) -> dict[str, Any]:
    """Verify canonical ZIP metadata, member set, manifest core, sizes, and hashes."""
    try:
        archive = zipfile.ZipFile(archive_path)
    except (OSError, zipfile.BadZipFile) as error:
        raise ReleaseError("submission archive is not a valid ZIP") from error
    with archive:
        infos = archive.infolist()
        names = [info.filename for info in infos]
        if len(names) != len(set(names)):
            raise ReleaseError("duplicate archive member")
        if names != sorted(names):
            raise ReleaseError("archive members are not sorted")
        for info in infos:
            _safe_member(info.filename)
            mode = (info.external_attr >> 16) & 0o177777
            if (
                info.date_time != _ZIP_TIME
                or info.compress_type != zipfile.ZIP_STORED
                or not stat.S_ISREG(mode)
                or stat.S_IMODE(mode) != 0o644
            ):
                raise ReleaseError(f"archive metadata differs: {info.filename}")
        roots = {PurePosixPath(name).parts[0] for name in names}
        if len(roots) != 1:
            raise ReleaseError("archive must contain exactly one root directory")
        root = next(iter(roots))
        manifest_name = f"{root}/{_MANIFEST}"
        if manifest_name not in names:
            raise ReleaseError("submission manifest is missing")
        manifest_bytes = archive.read(manifest_name)
        try:
            document = json.loads(manifest_bytes)
        except json.JSONDecodeError as error:
            raise ReleaseError("submission manifest is not valid JSON") from error
        if not isinstance(document, dict):
            raise ReleaseError("submission manifest must be a JSON object")
        manifest = cast(dict[str, Any], document)
        if canonical_json(manifest) != manifest_bytes:
            raise ReleaseError("submission manifest is not canonically encoded")
        claimed_core = manifest.pop("deterministic_core_sha256", None)
        actual_core = sha256_bytes(canonical_json(manifest))
        manifest["deterministic_core_sha256"] = claimed_core
        if claimed_core != actual_core:
            raise ReleaseError("submission manifest deterministic core differs")
        if manifest.get("schema_version") != _SCHEMA:
            raise ReleaseError("submission manifest schema differs")
        raw_files = manifest.get("files")
        if not isinstance(raw_files, list):
            raise ReleaseError("submission file inventory is absent")
        expected_names = {manifest_name}
        for raw_file in raw_files:
            if not isinstance(raw_file, dict):
                raise ReleaseError("submission file inventory is malformed")
            item = cast(dict[str, Any], raw_file)
            relative = require_safe_relative_path(item.get("path"), label="manifest file path")
            name = f"{root}/{relative}"
            if name in expected_names:
                raise ReleaseError("submission file inventory contains duplicates")
            expected_names.add(name)
            if name not in names:
                raise ReleaseError(f"manifest file is missing: {relative}")
            payload = archive.read(name)
            if item.get("size") != len(payload) or item.get("sha256") != sha256_bytes(payload):
                raise ReleaseError(f"manifest file digest differs: {relative}")
        if set(names) != expected_names:
            raise ReleaseError("archive contains a member absent from the manifest")
        authority = manifest.get("evidence_authority")
        if not isinstance(authority, dict):
            raise ReleaseError("evidence authority flags are absent")
        for false_flag in (
            "accepted_capacity_evidence",
            "authoritative",
            "official_chain_complete",
            "production_ready",
            "real_cardholder_data",
        ):
            if authority.get(false_flag) is not False:
                raise ReleaseError(f"unsafe evidence authority flag: {false_flag}")
        return manifest
