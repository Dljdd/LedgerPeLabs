"""Pinned safe-core and two-commit chronology contracts for v5 pre-execution."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import pytest

from scripts import verify_defense_v5_preexecution as preexecution

_SOURCE_PATHS = (
    "config/defense/defense-v5-arms.json",
    "config/defense/defense-v5-development.json",
    "config/defense/defense-v5-evidence.json",
    "config/defense/feature-catalog-v5.json",
)


def _digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode()
    ).hexdigest()


def _manifest(root: Path) -> tuple[dict[str, object], dict[str, object]]:
    for index, relative in enumerate(_SOURCE_PATHS):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"source-{index}\n")
    bindings = {
        "evidence_protocol_sha256": "1" * 64,
        "base_protocol_sha256": "2" * 64,
        "arm_protocol_sha256": "3" * 64,
        "implementation_sha256": "4" * 64,
        "catalog_sha256": "5" * 64,
    }
    document: dict[str, object] = {
        "schema_version": "apar-sentinel-v5-safe-core-freeze/1",
        "source_commit": "a" * 40,
        "source_tree_oid": "b" * 40,
        "freeze_path": "config/defense/defense-v5-safe-core-freeze.json",
        "approved_deterministic_core_sha256": "c" * 64,
        "approved_observational_environment_sha256": "d" * 64,
        "source_bindings": bindings,
        "source_files": [
            [relative, hashlib.sha256((root / relative).read_bytes()).hexdigest()]
            for relative in _SOURCE_PATHS
        ],
    }
    document["manifest_sha256"] = _digest(document)
    verification = {
        **bindings,
        "deterministic_core_sha256": "c" * 64,
        "observational_environment_sha256": "d" * 64,
        "observational_latency_sha256": "e" * 64,
    }
    return document, verification


def test_safe_core_freeze_accepts_fresh_observation_but_rejects_other_core(
    tmp_path: Path,
) -> None:
    """Pinning an envelope instead of the core must reject valid fresh timing evidence."""
    validator = getattr(preexecution, "validate_safe_core_freeze", None)
    assert callable(validator), "safe-core freeze validator is missing"
    manifest, verification = _manifest(tmp_path)
    validated = validator(
        root=tmp_path,
        document=manifest,
        verification={**verification, "observational_latency_sha256": "f" * 64},
        source_commit="a" * 40,
        source_tree_oid="b" * 40,
    )
    assert validated["approved_deterministic_core_sha256"] == "c" * 64
    with pytest.raises(ValueError, match="approved deterministic core"):
        validator(
            root=tmp_path,
            document=manifest,
            verification={**verification, "deterministic_core_sha256": "0" * 64},
            source_commit="a" * 40,
            source_tree_oid="b" * 40,
        )


def _git(root: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


@pytest.mark.parametrize("extra_path", [False, True])
def test_freeze_commit_chronology_allows_only_the_manifest(
    tmp_path: Path, extra_path: bool
) -> None:
    """Dropping the changed-path audit must accept behavior mixed into the freeze commit."""
    verifier = getattr(preexecution, "verify_freeze_commit_chronology", None)
    assert callable(verifier), "freeze-commit chronology verifier is missing"
    _git(tmp_path, "init", "-q")
    source = tmp_path / "source.txt"
    source.write_text("source\n")
    _git(tmp_path, "add", "source.txt")
    _git(
        tmp_path,
        "-c",
        "user.name=Dylan Moraes",
        "-c",
        "user.email=dylanmoraesdljdd@gmail.com",
        "commit",
        "-q",
        "-m",
        "test: source",
    )
    source_commit = _git(tmp_path, "rev-parse", "HEAD")
    source_tree_oid = _git(tmp_path, "rev-parse", "HEAD^{tree}")
    freeze_path = Path("config/defense/defense-v5-safe-core-freeze.json")
    absolute_freeze = tmp_path / freeze_path
    absolute_freeze.parent.mkdir(parents=True)
    absolute_freeze.write_text("{}\n")
    _git(tmp_path, "add", str(freeze_path))
    if extra_path:
        unexpected = tmp_path / "unexpected.txt"
        unexpected.write_text("behavior\n")
        _git(tmp_path, "add", "unexpected.txt")
    _git(
        tmp_path,
        "-c",
        "user.name=Dylan Moraes",
        "-c",
        "user.email=dylanmoraesdljdd@gmail.com",
        "commit",
        "-q",
        "-m",
        "test: freeze",
    )
    approved_commit = _git(tmp_path, "rev-parse", "HEAD")
    if extra_path:
        with pytest.raises(ValueError, match="manifest-only"):
            verifier(
                root=tmp_path,
                approved_commit=approved_commit,
                source_commit=source_commit,
                source_tree_oid=source_tree_oid,
                freeze_path=str(freeze_path),
            )
    else:
        verifier(
            root=tmp_path,
            approved_commit=approved_commit,
            source_commit=source_commit,
            source_tree_oid=source_tree_oid,
            freeze_path=str(freeze_path),
        )
