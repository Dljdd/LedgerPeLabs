from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from apar.evaluation.v5_kaggle_protocol import V5KaggleMode, V5KaggleStage
from scripts.verify_defense_v5_kaggle_preexecution import (
    V5KagglePreexecutionPhase,
    build_v5_safe_evidence_input_manifest,
    build_v5_source_archive,
    build_v5_wheelhouse_manifest,
    verify_v5_kaggle_preexecution,
)

ROOT = Path(__file__).resolve().parents[2]


def _git(root: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _canonical(document: object) -> bytes:
    return json.dumps(
        document,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode()


def _fake_wheel(path: Path, payload: bytes = b"fixture-wheel") -> None:
    path.write_bytes(payload)


def _complete_fake_wheelhouse(path: Path) -> None:
    for distribution in (
        "apar",
        "catboost",
        "cryptography",
        "fastapi",
        "numpy",
        "pandas",
        "pydantic",
        "pyarrow",
        "scikit_learn",
    ):
        _fake_wheel(path / f"{distribution}-1.0-py3-none-any.whl")


def test_source_archive_is_canonical_and_uses_only_commit_tree(tmp_path: Path) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    _git(repository, "init", "-q")
    _git(repository, "config", "user.name", "Dylan Moraes")
    _git(repository, "config", "user.email", "dylanmoraesdljdd@gmail.com")
    (repository / "plain.txt").write_text("frozen\n")
    executable = repository / "run.sh"
    executable.write_text("#!/bin/sh\nexit 0\n")
    executable.chmod(0o755)
    _git(repository, "add", "plain.txt", "run.sh")
    _git(repository, "commit", "-q", "-m", "fixture")
    commit = _git(repository, "rev-parse", "HEAD")
    (repository / "plain.txt").write_text("dirty working tree\n")

    first = tmp_path / "first" / "apar-v5-source3.tar.gz"
    second = tmp_path / "second" / "apar-v5-source3.tar.gz"
    manifest_a = build_v5_source_archive(root=repository, commit=commit, output=first)
    manifest_b = build_v5_source_archive(root=repository, commit=commit, output=second)
    assert first.read_bytes() == second.read_bytes()
    assert manifest_a == manifest_b
    assert manifest_a.approved_commit == commit
    assert manifest_a.artifact_sha256 == hashlib.sha256(first.read_bytes()).hexdigest()
    assert tuple(item.path for item in manifest_a.files) == ("plain.txt", "run.sh")
    assert tuple(item.mode for item in manifest_a.files) == ("100644", "100755")
    assert b"dirty working tree" not in first.read_bytes()


def test_source_archive_rejects_overwrite_link_and_noncommit(tmp_path: Path) -> None:
    target = tmp_path / "apar-v5-source3.tar.gz"
    target.write_bytes(b"existing")
    with pytest.raises(FileExistsError):
        build_v5_source_archive(root=ROOT, commit="HEAD", output=target)
    target.unlink()
    linked = tmp_path / "linked" / "apar-v5-source3.tar.gz"
    linked.parent.mkdir()
    linked.symlink_to(tmp_path / "missing")
    with pytest.raises(FileExistsError):
        build_v5_source_archive(root=ROOT, commit="HEAD", output=linked)
    with pytest.raises(ValueError):
        build_v5_source_archive(root=ROOT, commit="not-a-commit", output=target)


def test_wheelhouse_manifest_is_sorted_self_digesting_and_platform_closed(
    tmp_path: Path,
) -> None:
    wheelhouse = tmp_path / "wheels"
    wheelhouse.mkdir()
    _complete_fake_wheelhouse(wheelhouse)
    manifest = build_v5_wheelhouse_manifest(wheelhouse=wheelhouse, write=True)
    assert tuple(item.filename for item in manifest.wheels) == tuple(
        sorted(item.filename for item in manifest.wheels)
    )
    document = json.loads((wheelhouse / "wheelhouse-manifest.json").read_bytes())
    claimed = document.pop("manifest_sha256")
    assert claimed == hashlib.sha256(_canonical(document)).hexdigest()
    assert build_v5_wheelhouse_manifest(wheelhouse=wheelhouse, write=False) == manifest

    (wheelhouse / "wheelhouse-manifest.json").unlink()
    _fake_wheel(wheelhouse / "bad-1.0-cp312-cp312-macosx_11_0_x86_64.whl")
    with pytest.raises(ValueError, match="Linux"):
        build_v5_wheelhouse_manifest(wheelhouse=wheelhouse, write=False)


def test_wheelhouse_manifest_rejects_missing_project_dependency(tmp_path: Path) -> None:
    wheelhouse = tmp_path / "wheels"
    wheelhouse.mkdir()
    _complete_fake_wheelhouse(wheelhouse)
    (wheelhouse / "catboost-1.0-py3-none-any.whl").unlink()
    with pytest.raises(ValueError, match="required distributions"):
        build_v5_wheelhouse_manifest(wheelhouse=wheelhouse, write=False)


def test_safe_evidence_input_manifest_binds_bytes_without_modifying_them(
    tmp_path: Path,
) -> None:
    safe = tmp_path / "safe-evidence.json"
    safe.write_bytes(b'{"safe":true}')
    before = safe.read_bytes()
    manifest = build_v5_safe_evidence_input_manifest(
        safe_evidence=safe,
        mode=V5KaggleMode.CAPACITY_VALIDATION,
        approved_commit="a" * 40,
        protocol_sha256="b" * 64,
        run_binding_sha256="c" * 64,
    )
    assert safe.read_bytes() == before
    assert manifest.artifact_sha256 == hashlib.sha256(before).hexdigest()
    assert manifest.artifact_name == "safe-evidence.json"
    assert manifest.execution_mode is V5KaggleMode.CAPACITY_VALIDATION
    assert manifest.development_test_seed == 404


def test_safe_input_manifest_rejects_mode_seed_or_authorization_relabel(
    tmp_path: Path,
) -> None:
    safe = tmp_path / "safe-evidence.json"
    safe.write_bytes(b'{"safe":true}')
    manifest = build_v5_safe_evidence_input_manifest(
        safe_evidence=safe,
        mode=V5KaggleMode.CAPACITY_VALIDATION,
        approved_commit="a" * 40,
        protocol_sha256="b" * 64,
        run_binding_sha256="c" * 64,
    )
    document = manifest.model_dump(mode="json")
    document["execution_mode"] = V5KaggleMode.LOCKED_SUCCESSOR.value
    document["development_test_seed"] = 2404
    document["authorization_required"] = True
    unsigned = dict(document)
    unsigned.pop("manifest_sha256")
    document["manifest_sha256"] = hashlib.sha256(_canonical(unsigned)).hexdigest()
    with pytest.raises(ValueError, match="authorization"):
        type(manifest).model_validate(document)


def test_source_preexecution_accepts_exact_clean_linear_repairs_and_rejects_dirty_tree(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "repository"
    shutil.copytree(
        ROOT,
        repository,
        ignore=shutil.ignore_patterns(
            ".git",
            ".venv",
            ".mypy_cache",
            ".pytest_cache",
            ".ruff_cache",
            "__pycache__",
        ),
    )
    _git(repository, "init", "-q")
    _git(repository, "config", "user.name", "Dylan Moraes")
    _git(repository, "config", "user.email", "dylanmoraesdljdd@gmail.com")
    _git(repository, "add", ".")
    _git(repository, "commit", "-q", "-m", "recovery fixture")
    recovery = _git(repository, "rev-parse", "HEAD")
    marker = repository / "SOURCE3"
    marker.write_text("source\n")
    _git(repository, "add", "SOURCE3")
    _git(repository, "commit", "-q", "-m", "source fixture")
    repair = repository / "SOURCE4"
    repair.write_text("browser-derived repair\n")
    _git(repository, "add", "SOURCE4")
    _git(repository, "commit", "-q", "-m", "source repair fixture")
    source = _git(repository, "rev-parse", "HEAD")

    archive = tmp_path / "apar-v5-source3.tar.gz"
    build_v5_source_archive(root=repository, commit=source, output=archive)
    wheelhouse = tmp_path / "wheelhouse"
    wheelhouse.mkdir()
    _complete_fake_wheelhouse(wheelhouse)
    build_v5_wheelhouse_manifest(wheelhouse=wheelhouse, write=True)

    report = verify_v5_kaggle_preexecution(
        root=repository,
        phase=V5KagglePreexecutionPhase.SOURCE,
        expected_head=source,
        source_archive=archive,
        wheelhouse=wheelhouse,
        rehearsal_chain_roots=(),
        expected_recovery_commit=recovery,
    )
    assert report.valid is True
    assert report.source_commit == source
    assert report.seed_2404_boundary == "asserted_only"

    marker.write_text("dirty\n")
    with pytest.raises(ValueError, match="clean"):
        verify_v5_kaggle_preexecution(
            root=repository,
            phase=V5KagglePreexecutionPhase.SOURCE,
            expected_head=source,
            source_archive=archive,
            wheelhouse=wheelhouse,
            rehearsal_chain_roots=(),
            expected_recovery_commit=recovery,
        )


def test_source_lineage_rejects_merge_ancestry(tmp_path: Path) -> None:
    from scripts import verify_defense_v5_kaggle_preexecution as module

    repository = tmp_path / "repository"
    repository.mkdir()
    _git(repository, "init", "-q")
    _git(repository, "config", "user.name", "Dylan Moraes")
    _git(repository, "config", "user.email", "dylanmoraesdljdd@gmail.com")
    (repository / "recovery").write_text("preserved\n")
    _git(repository, "add", "recovery")
    _git(repository, "commit", "-q", "-m", "recovery")
    recovery = _git(repository, "rev-parse", "HEAD")
    primary = _git(repository, "branch", "--show-current")
    _git(repository, "checkout", "-q", "-b", "repair-a")
    (repository / "repair-a").write_text("a\n")
    _git(repository, "add", "repair-a")
    _git(repository, "commit", "-q", "-m", "repair a")
    _git(repository, "checkout", "-q", primary)
    (repository / "repair-b").write_text("b\n")
    _git(repository, "add", "repair-b")
    _git(repository, "commit", "-q", "-m", "repair b")
    _git(repository, "merge", "-q", "--no-ff", "repair-a", "-m", "merge repairs")

    with pytest.raises(ValueError, match="linear descendant"):
        module._linear_source_lineage(  # type: ignore[attr-defined]
            repository,
            source_commit=_git(repository, "rev-parse", "HEAD"),
            recovery_commit=recovery,
        )


def test_source_audit_rejects_successor_output_and_notebook_privacy_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts import verify_defense_v5_kaggle_preexecution as module

    root = tmp_path / "root"
    root.mkdir()
    output = root / "docs/experiments/defense-v5-kaggle-successor-attempt.json"
    output.parent.mkdir(parents=True)
    output.write_text("partial")
    with pytest.raises(FileExistsError):
        module._verify_successor_outputs_absent(root)  # type: ignore[attr-defined]

    metadata = json.loads((ROOT / "kaggle/defense_v5/00_authorize-metadata.json").read_bytes())
    metadata["is_private"] = False
    with pytest.raises(ValueError, match="private"):
        module._verify_notebook_metadata_document(  # type: ignore[attr-defined]
            metadata,
            stage=V5KaggleStage.AUTHORIZE,
            predecessor=None,
            owner="dylanmoraes",
            source_slug="apar-sentinel-v5-source3",
            wheelhouse_slug="apar-sentinel-v5-wheelhouse-py312-linux-x86-64",
            safe_slug="apar-sentinel-v5-safe-evidence",
        )
    monkeypatch.setenv("APAR_SHOULD_NOT_BE_READ", "2404")


def test_public_audit_has_no_execution_or_arbitrary_seed_surface() -> None:
    source = (ROOT / "scripts/verify_defense_v5_kaggle_preexecution.py").read_text()
    forbidden = (
        "build_v5_corpus",
        "execute_v5_complete_evidence",
        "execute_v5_controls",
        "evaluate_v5_complete_result",
        "SentinelDefender",
        "SimulationEngine",
    )
    assert all(token not in source for token in forbidden)
    completed = subprocess.run(
        [
            os.fspath(Path(os.sys.executable)),
            "-m",
            "scripts.verify_defense_v5_kaggle_preexecution",
            "--root",
            str(ROOT),
            "--phase",
            "source",
            "--seed",
            "404",
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode != 0


def test_frozen_phase_requires_two_capacity_rehearsals() -> None:
    with pytest.raises(ValueError, match="two"):
        verify_v5_kaggle_preexecution(
            root=ROOT,
            phase=V5KagglePreexecutionPhase.FROZEN,
            expected_head="HEAD",
            source_archive=None,
            wheelhouse=None,
            rehearsal_chain_roots=(),
        )
    assert V5KaggleMode.LOCKED_SUCCESSOR.value == "kaggle_locked_successor"
