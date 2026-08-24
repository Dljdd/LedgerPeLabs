"""Fail-closed pre-execution audit for the one-time Sentinel v5 development run."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

from apar.v5_independent_verifier import (
    IndependentVerificationError,
    verify_evidence_bytes,
)

_FREEZE_PATH = "config/defense/defense-v5-safe-core-freeze.json"
_SOURCE_PATHS = (
    "config/defense/defense-v5-arms.json",
    "config/defense/defense-v5-development.json",
    "config/defense/defense-v5-evidence.json",
    "config/defense/feature-catalog-v5.json",
)
_BINDING_FIELDS = (
    "evidence_protocol_sha256",
    "base_protocol_sha256",
    "arm_protocol_sha256",
    "implementation_sha256",
    "catalog_sha256",
)


def _git(root: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise ValueError(completed.stderr.strip() or "git command failed")
    return completed.stdout.strip()


def _canonical_digest(document: object) -> str:
    return hashlib.sha256(
        json.dumps(
            document,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode()
    ).hexdigest()


def validate_safe_core_freeze(
    *,
    root: Path,
    document: dict[str, object],
    verification: dict[str, object],
    source_commit: str,
    source_tree_oid: str,
) -> dict[str, object]:
    """Validate the manifest against independently recomputed safe evidence."""
    expected_fields = {
        "schema_version",
        "source_commit",
        "source_tree_oid",
        "freeze_path",
        "approved_deterministic_core_sha256",
        "approved_observational_environment_sha256",
        "source_bindings",
        "source_files",
        "manifest_sha256",
    }
    if set(document) != expected_fields:
        raise ValueError("safe-core freeze manifest schema differs")
    if document["schema_version"] != "apar-sentinel-v5-safe-core-freeze/1":
        raise ValueError("safe-core freeze manifest version differs")
    if document["freeze_path"] != _FREEZE_PATH:
        raise ValueError("safe-core freeze path differs")
    if document["source_commit"] != source_commit or len(source_commit) != 40:
        raise ValueError("safe-core freeze source commit differs")
    if document["source_tree_oid"] != source_tree_oid or len(source_tree_oid) != 40:
        raise ValueError("safe-core freeze source tree differs")
    expected_manifest_sha256 = _canonical_digest(
        {key: value for key, value in document.items() if key != "manifest_sha256"}
    )
    if document["manifest_sha256"] != expected_manifest_sha256:
        raise ValueError("safe-core freeze manifest digest differs")
    bindings = document["source_bindings"]
    if type(bindings) is not dict or set(bindings) != set(_BINDING_FIELDS):
        raise ValueError("safe-core freeze source bindings differ")
    for field in _BINDING_FIELDS:
        if bindings[field] != verification.get(field):
            raise ValueError(f"safe-core freeze {field} differs")
    source_files = document["source_files"]
    if type(source_files) is not list:
        raise ValueError("safe-core freeze source file set/order differs")
    parsed_source_files: list[tuple[str, str]] = []
    for item in source_files:
        if (
            type(item) is not list
            or len(item) != 2
            or type(item[0]) is not str
            or type(item[1]) is not str
        ):
            raise ValueError("safe-core freeze source file record differs")
        relative = str(item[0])
        digest = str(item[1])
        parsed_source_files.append((relative, digest))
        source = root / relative
        if not source.is_file() or hashlib.sha256(source.read_bytes()).hexdigest() != digest:
            raise ValueError("safe-core freeze source file digest differs")
    if tuple(relative for relative, _digest_value in parsed_source_files) != _SOURCE_PATHS:
        raise ValueError("safe-core freeze source file set/order differs")
    if document["approved_deterministic_core_sha256"] != verification.get(
        "deterministic_core_sha256"
    ):
        raise ValueError("safe evidence does not match the approved deterministic core")
    if document["approved_observational_environment_sha256"] != verification.get(
        "observational_environment_sha256"
    ):
        raise ValueError("safe evidence observational environment differs")
    return document


def verify_freeze_commit_chronology(
    *,
    root: Path,
    approved_commit: str,
    source_commit: str,
    source_tree_oid: str,
    freeze_path: str,
) -> None:
    """Require one manifest-only child commit over the exact SOURCE commit."""
    head = _git(root, "rev-parse", "HEAD")
    if head != approved_commit or len(approved_commit) != 40:
        raise ValueError("HEAD does not equal the exact approved FREEZE commit")
    parent_line = _git(root, "rev-list", "--parents", "-n", "1", head).split()
    if len(parent_line) != 2 or parent_line[1] != source_commit:
        raise ValueError("FREEZE commit is not the single child of the SOURCE commit")
    if _git(root, "rev-parse", f"{source_commit}^{{tree}}") != source_tree_oid:
        raise ValueError("SOURCE commit tree differs from the freeze manifest")
    changed = tuple(
        line
        for line in _git(
            root,
            "diff-tree",
            "--no-commit-id",
            "--name-only",
            "-r",
            head,
        ).splitlines()
        if line
    )
    if changed != (freeze_path,):
        raise ValueError("FREEZE commit must be manifest-only")


def build_safe_core_freeze_manifest(
    *, root: Path, safe_evidence: Path, source_commit: str
) -> dict[str, object]:
    """Build the sole manifest document to add in the manifest-only FREEZE commit."""
    root = root.resolve()
    if _git(root, "status", "--porcelain", "--untracked-files=all"):
        raise ValueError("SOURCE worktree is not clean")
    if _git(root, "rev-parse", "HEAD") != source_commit or len(source_commit) != 40:
        raise ValueError("HEAD does not equal the exact SOURCE commit")
    verification = verify_evidence_bytes(safe_evidence.read_bytes(), root=root)
    values: dict[str, object] = {
        "schema_version": "apar-sentinel-v5-safe-core-freeze/1",
        "source_commit": source_commit,
        "source_tree_oid": _git(root, "rev-parse", f"{source_commit}^{{tree}}"),
        "freeze_path": _FREEZE_PATH,
        "approved_deterministic_core_sha256": verification[
            "deterministic_core_sha256"
        ],
        "approved_observational_environment_sha256": verification[
            "observational_environment_sha256"
        ],
        "source_bindings": {
            field: verification[field] for field in _BINDING_FIELDS
        },
        "source_files": [
            [relative, hashlib.sha256((root / relative).read_bytes()).hexdigest()]
            for relative in _SOURCE_PATHS
        ],
    }
    values["manifest_sha256"] = _canonical_digest(values)
    return values


def verify_preexecution(
    *, root: Path, safe_evidence: Path, approved_commit: str
) -> dict[str, object]:
    """Verify immutable inputs without invoking any experiment workload."""
    root = root.resolve()
    if _git(root, "status", "--porcelain", "--untracked-files=all"):
        raise ValueError("worktree is not clean")
    head = _git(root, "rev-parse", "HEAD")
    if head != approved_commit or len(approved_commit) != 40:
        raise ValueError("HEAD does not equal the exact approved FREEZE commit")

    freeze_path = root / _FREEZE_PATH
    freeze_document = json.loads(freeze_path.read_bytes())
    if type(freeze_document) is not dict:
        raise ValueError("safe-core freeze manifest must be a JSON object")
    source_commit = str(freeze_document.get("source_commit", ""))
    source_tree_oid = str(freeze_document.get("source_tree_oid", ""))
    verify_freeze_commit_chronology(
        root=root,
        approved_commit=approved_commit,
        source_commit=source_commit,
        source_tree_oid=source_tree_oid,
        freeze_path=_FREEZE_PATH,
    )

    evidence_config_path = root / "config/defense/defense-v5-evidence.json"
    development_config_path = root / "config/defense/defense-v5-development.json"
    evidence_config = json.loads(evidence_config_path.read_bytes())
    development_config = json.loads(development_config_path.read_bytes())
    if (
        evidence_config["safe_development_test_seed"] != 404
        or evidence_config["locked_development_test_seed"] != 2404
        or development_config["seeds"]["development_test"] != 2404
    ):
        raise ValueError("safe/locked seed bindings differ from 404/2404")

    result_path = root / evidence_config["existing_development_result_path"]
    if not result_path.is_file():
        raise ValueError("frozen existing development result is absent")
    result_sha256 = hashlib.sha256(result_path.read_bytes()).hexdigest()
    if result_sha256 != evidence_config["existing_development_result_sha256"]:
        raise ValueError("existing development result bytes changed")

    verification = verify_evidence_bytes(
        safe_evidence.read_bytes(), root=root
    )
    validate_safe_core_freeze(
        root=root,
        document=freeze_document,
        verification=verification,
        source_commit=source_commit,
        source_tree_oid=source_tree_oid,
    )
    return {
        "verified": True,
        "approved_commit": approved_commit,
        "worktree_clean": True,
        "safe_evidence_verified": verification["verified"],
        "safe_evidence_status": verification["status"],
        "safe_seed_executed": verification["safe_seed"],
        "locked_seed_asserted_only": 2404,
        "existing_result_sha256": result_sha256,
        "evidence_config_sha256": hashlib.sha256(
            evidence_config_path.read_bytes()
        ).hexdigest(),
        "development_config_sha256": hashlib.sha256(
            development_config_path.read_bytes()
        ).hexdigest(),
        "payload_sha256": verification["payload_sha256"],
        "envelope_sha256": verification["envelope_sha256"],
        "deterministic_core_sha256": verification[
            "deterministic_core_sha256"
        ],
        "observational_latency_sha256": verification[
            "observational_latency_sha256"
        ],
        "observational_environment_sha256": verification[
            "observational_environment_sha256"
        ],
        "source_commit": source_commit,
        "freeze_manifest_sha256": freeze_document["manifest_sha256"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--safe-evidence", type=Path, required=True)
    parser.add_argument("--approved-commit", required=True)
    args = parser.parse_args()
    try:
        report = verify_preexecution(
            root=args.root,
            safe_evidence=args.safe_evidence,
            approved_commit=args.approved_commit,
        )
    except (IndependentVerificationError, OSError, ValueError) as error:
        print(
            json.dumps(
                {"verified": False, "error": str(error)},
                sort_keys=True,
                separators=(",", ":"),
            ),
            file=sys.stderr,
        )
        return 1
    print(json.dumps(report, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
