"""Judge-side payload, model replay, and CLI-fallback verifier.

This module is shipped in the archive. It does not build releases, train models,
or write into evidence paths.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, cast

from scripts.submission.model import (
    ReleaseError,
    canonical_json,
    require_safe_relative_path,
    sha256_bytes,
)
from scripts.submission.replay import build_fallback_trace

_SCHEMA = "apar-submission-manifest/1"
_FALSE_AUTHORITY_FLAGS = (
    "accepted_capacity_evidence",
    "authoritative",
    "official_chain_complete",
    "production_ready",
    "real_cardholder_data",
)


def _load_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        payload = path.read_bytes()
        document = json.loads(payload)
    except (OSError, json.JSONDecodeError) as error:
        raise ReleaseError(f"{label} is not valid JSON") from error
    if not isinstance(document, dict):
        raise ReleaseError(f"{label} must be a JSON object")
    return cast(dict[str, Any], document)


def verify_payload_manifest(root: Path) -> dict[str, Any]:
    """Verify canonical manifest bytes, deterministic core, and every listed payload."""
    release_root = root.resolve()
    manifest_path = release_root / "SUBMISSION_MANIFEST.json"
    manifest_bytes = manifest_path.read_bytes()
    manifest = _load_object(manifest_path, label="submission manifest")
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
    seen: set[str] = set()
    for raw_file in raw_files:
        if not isinstance(raw_file, dict):
            raise ReleaseError("submission file inventory is malformed")
        item = cast(dict[str, Any], raw_file)
        relative = require_safe_relative_path(item.get("path"), label="manifest file path")
        if relative in seen:
            raise ReleaseError("submission file inventory contains duplicates")
        seen.add(relative)
        path = release_root.joinpath(*Path(relative).parts)
        if path.is_symlink() or not path.is_file():
            raise ReleaseError(f"payload is missing or not regular: {relative}")
        payload = path.read_bytes()
        if item.get("size") != len(payload) or item.get("sha256") != sha256_bytes(payload):
            raise ReleaseError(f"payload digest differs: {relative}")
    authority = manifest.get("evidence_authority")
    if not isinstance(authority, dict):
        raise ReleaseError("evidence authority flags are absent")
    for flag in _FALSE_AUTHORITY_FLAGS:
        if authority.get(flag) is not False:
            raise ReleaseError(f"unsafe evidence authority flag: {flag}")
    return manifest


def verify_release_runtime(root: Path) -> dict[str, Any]:
    """Run all 12 hash-bound scenarios and validate the deterministic fallback trace."""
    release_root = root.resolve()
    manifest = verify_payload_manifest(release_root)
    source_root = release_root / "src"
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))
    try:
        from apar.demo.sentinel_v5_portable import run_portable_scenarios
    except ImportError as error:
        raise ReleaseError("documented portable runtime dependencies are not installed") from error
    bundle_root = release_root / "demo" / "sentinel-v5"
    try:
        report = run_portable_scenarios(
            bundle_root=bundle_root,
            scenario_path=bundle_root / "scenarios.json",
        )
    except (OSError, TypeError, ValueError) as error:
        raise ReleaseError(f"portable Sentinel replay failed: {error}") from error
    web = manifest.get("web")
    accepted_model = manifest.get("accepted_model")
    expected = manifest.get("runtime_verification")
    if not isinstance(web, dict) or not isinstance(accepted_model, dict) or not isinstance(
        expected, dict
    ):
        raise ReleaseError("runtime verification bindings are incomplete")
    web_status = web.get("status")
    if not isinstance(web_status, str):
        raise ReleaseError("web release status is absent")
    fallback = build_fallback_trace(report, web_status=web_status)
    if report.get("bundle_manifest_sha256") != accepted_model.get(
        "portable_bundle_manifest_sha256"
    ):
        raise ReleaseError("portable bundle identity differs from the submission manifest")
    checks = {
        "fallback_trace_sha256": fallback.get("trace_sha256"),
        "prediction_sha256": fallback.get("prediction_sha256"),
        "scenario_count": fallback.get("scenario_count"),
    }
    for name, actual in checks.items():
        if expected.get(name) != actual:
            raise ReleaseError(f"runtime verification binding differs: {name}")
    return {
        "accepted_model": accepted_model,
        "evidence_authority": manifest["evidence_authority"],
        "fallback_trace": fallback,
        "payload_core_sha256": manifest["deterministic_core_sha256"],
        "replay_verified": True,
        "schema_version": "apar-submission-clean-room-verification/1",
    }


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify an extracted APAR submission release.")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        result = verify_release_runtime(args.root)
        payload = canonical_json(result)
        if args.output is not None:
            if args.output.exists():
                raise ReleaseError(f"verification output already exists: {args.output}")
            args.output.write_bytes(payload)
        print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    except ReleaseError as error:
        print(f"release verification failed: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
