"""Closed next-stage runner and continuation semantics for Sentinel v5."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import stat
import subprocess
import sys
from pathlib import Path

import pytest

from apar.evaluation.v5_checkpoint_storage import V5CheckpointInput
from apar.evaluation.v5_kaggle_protocol import (
    V5KaggleEnvironmentBinding,
    V5KaggleMode,
    V5KaggleStage,
    load_v5_kaggle_protocol,
)
from apar.evaluation.v5_staged_evidence import V5StageCapability
from scripts import run_defense_v5_kaggle_stage as runner_module
from scripts.run_defense_v5_kaggle_stage import execute_next_v5_kaggle_stage
from scripts.verify_defense_v5_kaggle_preexecution import (
    build_v5_safe_evidence_input_manifest,
)

ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "scripts/run_defense_v5_kaggle_stage.py"


class _SyntheticAuthority:
    def __init__(self, *, fail_once_at: V5KaggleStage | None = None) -> None:
        self.fail_once_at = fail_once_at
        self.executed: list[V5KaggleStage] = []

    def preflight(self, **kwargs: object) -> None:
        return None

    def environment(self, *, root: Path) -> V5KaggleEnvironmentBinding:
        del root
        return V5KaggleEnvironmentBinding.bind(
            provider="kaggle",
            image="python-cpu-test",
            image_sha256="1" * 64,
            python_version="3.12.5",
            architecture="x86_64",
            cpu_count=4,
            dependency_manifest_sha256="2" * 64,
            source_archive_sha256="3" * 64,
            notebook_sha256="4" * 64,
            internet_enabled=False,
            accelerator="none",
            file_fsync_supported=True,
            directory_fsync_supported=True,
            hardlink_no_replace_supported=True,
        )

    def attempt_receipt_sha256(self, **kwargs: object) -> str:
        return "5" * 64

    def records(
        self,
        *,
        root: Path,
        capability: V5StageCapability,
        stage_roots: object,
    ) -> tuple[V5CheckpointInput, ...]:
        del root, stage_roots
        self.executed.append(capability.stage)
        if self.fail_once_at is capability.stage:
            self.fail_once_at = None
            raise RuntimeError("simulated interruption")
        document = {"stage": capability.stage, "mode": capability.mode}
        return (
            V5CheckpointInput(
                kind="synthetic_stage",
                key=capability.stage,
                canonical_bytes=json.dumps(
                    document, sort_keys=True, separators=(",", ":")
                ).encode(),
            ),
        )


def _advance(
    *,
    tmp_path: Path,
    authority: _SyntheticAuthority,
    mode: V5KaggleMode = V5KaggleMode.CAPACITY_VALIDATION,
) -> object:
    input_root = tmp_path / "chain"
    input_root.mkdir(exist_ok=True)
    next_stage = tuple(V5KaggleStage)[len(tuple(input_root.iterdir()))]
    safe = tmp_path / "safe-evidence.json"
    if not safe.exists():
        safe.write_text("{}")
    execution_manifest = tmp_path / "safe-evidence-manifest.json"
    if not execution_manifest.exists():
        protocol = load_v5_kaggle_protocol(
            ROOT / "config/defense/defense-v5-kaggle-recovery.json", root=ROOT
        )
        manifest = build_v5_safe_evidence_input_manifest(
            safe_evidence=safe,
            mode=mode,
            approved_commit="a" * 40,
            protocol_sha256=protocol.protocol_sha256,
            run_binding_sha256=protocol.run_binding_sha256(mode),
            successor_authorization_sha256=(
                "6" * 64 if mode is V5KaggleMode.LOCKED_SUCCESSOR else None
            ),
        )
        execution_manifest.write_text(
            json.dumps(
                manifest.model_dump(mode="json"),
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        )
    return execute_next_v5_kaggle_stage(
        root=ROOT,
        input_root=input_root,
        output_root=input_root / next_stage.value,
        safe_evidence=safe,
        execution_manifest=execution_manifest,
        approved_commit="a" * 40,
        authority=authority,
    )


def test_staged_runner_module_and_legacy_boundary_are_explicit() -> None:
    """The legacy monolithic command cannot be mistaken for a checkpoint runner."""
    assert RUNNER.is_file()
    assert importlib.util.spec_from_file_location("v5_kaggle_runner", RUNNER) is not None
    legacy = (ROOT / "scripts/run_defense_v5_locked_development.py").read_text()
    assert "--predecessor" not in legacy
    assert "--checkpoint" not in legacy


def test_completed_stage_advances_and_incomplete_stage_can_repeat(
    tmp_path: Path,
) -> None:
    """Only an absent current-stage manifest permits the same stage to execute again."""
    authority = _SyntheticAuthority(fail_once_at=V5KaggleStage.FEATURES)
    first = _advance(tmp_path=tmp_path, authority=authority)
    second = _advance(tmp_path=tmp_path, authority=authority)
    assert first.stage is V5KaggleStage.AUTHORIZE
    assert second.stage is V5KaggleStage.CORPUS

    with pytest.raises(RuntimeError, match="simulated interruption"):
        _advance(tmp_path=tmp_path, authority=authority)
    assert not (tmp_path / "chain" / V5KaggleStage.FEATURES.value).exists()
    replay = _advance(tmp_path=tmp_path, authority=authority)

    assert replay.stage is V5KaggleStage.FEATURES
    assert authority.executed == [
        V5KaggleStage.AUTHORIZE,
        V5KaggleStage.CORPUS,
        V5KaggleStage.FEATURES,
        V5KaggleStage.FEATURES,
    ]


def test_locked_execution_manifest_cannot_relabel_a_started_chain(tmp_path: Path) -> None:
    authority = _SyntheticAuthority()
    manifest = _advance(
        tmp_path=tmp_path,
        authority=authority,
        mode=V5KaggleMode.LOCKED_SUCCESSOR,
    )
    assert manifest.stage is V5KaggleStage.AUTHORIZE
    assert manifest.run_binding_sha256 != ""

    (tmp_path / "safe-evidence-manifest.json").unlink()
    with pytest.raises(ValueError, match="mode differs"):
        _advance(tmp_path=tmp_path, authority=authority)


def test_malformed_visible_checkpoint_is_terminal(tmp_path: Path) -> None:
    chain = tmp_path / "chain"
    malformed = chain / V5KaggleStage.AUTHORIZE.value
    malformed.mkdir(parents=True)
    (malformed / "checkpoint.manifest.json").write_text("{}")
    safe = tmp_path / "safe-evidence.json"
    safe.write_text("{}")
    protocol = load_v5_kaggle_protocol(
        ROOT / "config/defense/defense-v5-kaggle-recovery.json", root=ROOT
    )
    execution = build_v5_safe_evidence_input_manifest(
        safe_evidence=safe,
        mode=V5KaggleMode.CAPACITY_VALIDATION,
        approved_commit="a" * 40,
        protocol_sha256=protocol.protocol_sha256,
        run_binding_sha256=protocol.run_binding_sha256(
            V5KaggleMode.CAPACITY_VALIDATION
        ),
    )
    execution_path = tmp_path / "safe-evidence-manifest.json"
    execution_path.write_text(
        json.dumps(execution.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
        + "\n"
    )

    with pytest.raises(ValueError):
        execute_next_v5_kaggle_stage(
            root=ROOT,
            input_root=chain,
            output_root=chain / V5KaggleStage.CORPUS.value,
            safe_evidence=safe,
            execution_manifest=execution_path,
            approved_commit="a" * 40,
            authority=_SyntheticAuthority(),
        )


def test_unknown_later_stage_and_existing_output_fail_before_execution(
    tmp_path: Path,
) -> None:
    chain = tmp_path / "chain"
    chain.mkdir()
    (chain / V5KaggleStage.FEATURES.value).mkdir()
    safe = tmp_path / "safe-evidence.json"
    safe.write_text("{}")
    protocol = load_v5_kaggle_protocol(
        ROOT / "config/defense/defense-v5-kaggle-recovery.json", root=ROOT
    )
    execution = build_v5_safe_evidence_input_manifest(
        safe_evidence=safe,
        mode=V5KaggleMode.CAPACITY_VALIDATION,
        approved_commit="a" * 40,
        protocol_sha256=protocol.protocol_sha256,
        run_binding_sha256=protocol.run_binding_sha256(
            V5KaggleMode.CAPACITY_VALIDATION
        ),
    )
    execution_path = tmp_path / "safe-evidence-manifest.json"
    execution_path.write_text(
        json.dumps(execution.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
        + "\n"
    )
    authority = _SyntheticAuthority()
    with pytest.raises(ValueError, match="missing or reordered"):
        execute_next_v5_kaggle_stage(
            root=ROOT,
            input_root=chain,
            output_root=tmp_path / "output",
            safe_evidence=safe,
            execution_manifest=execution_path,
            approved_commit="a" * 40,
            authority=authority,
        )
    assert authority.executed == []

    empty = tmp_path / "empty"
    empty.mkdir()
    output = tmp_path / "existing"
    output.mkdir()
    with pytest.raises(FileExistsError, match="already exists"):
        execute_next_v5_kaggle_stage(
            root=ROOT,
            input_root=empty,
            output_root=output,
            safe_evidence=safe,
            execution_manifest=execution_path,
            approved_commit="a" * 40,
            authority=authority,
        )
    assert authority.executed == []


def test_direct_and_module_cli_expose_only_the_closed_surface() -> None:
    expected = {
        "--root",
        "--input-root",
        "--output-root",
        "--safe-evidence",
        "--execution-manifest",
        "--approved-commit",
    }
    outputs = []
    for command in (
        [sys.executable, str(RUNNER), "--help"],
        [sys.executable, "-m", "scripts.run_defense_v5_kaggle_stage", "--help"],
    ):
        completed = subprocess.run(
            command,
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        outputs.append(completed.stdout)
    for output in outputs:
        assert all(option in output for option in expected)
        assert all(
            forbidden not in output
            for forbidden in (
                "--seed",
                "--profile",
                "--stage",
                "--resume",
                "--retry",
                "--force",
                "--delete",
                "--test-authority",
                "--model",
            )
        )


def test_archive_preflight_recomputes_exact_source_tree_without_git(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    entrypoint = source / "entry.py"
    entrypoint.write_bytes(b"print('bound')\n")
    entrypoint.chmod(0o755)
    safe = tmp_path / "safe.json"
    safe.write_bytes(b"{}")
    commit = "a" * 40
    values: dict[str, object] = {
        "schema_version": "apar-sentinel-v5-source-archive/1",
        "artifact_name": "apar-v5-source3.tar.gz",
        "artifact_sha256": "b" * 64,
        "artifact_size_bytes": 1,
        "archive_prefix": "apar-v5-source/",
        "approved_commit": commit,
        "source_tree": "c" * 40,
        "files": [
            {
                "path": "entry.py",
                "mode": "100755",
                "size_bytes": entrypoint.stat().st_size,
                "sha256": hashlib.sha256(entrypoint.read_bytes()).hexdigest(),
            }
        ],
    }
    values["manifest_sha256"] = hashlib.sha256(
        json.dumps(values, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    manifest = tmp_path / "source-manifest.json"
    manifest.write_text(json.dumps(values, sort_keys=True, separators=(",", ":")))
    monkeypatch.setenv("APAR_V5_SOURCE_MANIFEST_PATH", os.fspath(manifest))
    monkeypatch.setattr(runner_module, "verify_evidence_bytes", lambda *_args, **_kwargs: {})
    protocol = runner_module.load_v5_kaggle_protocol(
        ROOT / "config/defense/defense-v5-kaggle-recovery.json",
        root=ROOT,
    )
    authority = runner_module._FrozenRepositoryAuthority()  # type: ignore[attr-defined]
    authority.preflight(
        root=source,
        approved_commit=commit,
        safe_evidence=safe,
        protocol=protocol,
        mode=runner_module.V5KaggleMode.CAPACITY_VALIDATION,
        stage=V5KaggleStage.AUTHORIZE,
    )

    entrypoint.write_bytes(b"print('tampered')\n")
    with pytest.raises(ValueError, match="source"):
        authority.preflight(
            root=source,
            approved_commit=commit,
            safe_evidence=safe,
            protocol=protocol,
            mode=runner_module.V5KaggleMode.CAPACITY_VALIDATION,
            stage=V5KaggleStage.AUTHORIZE,
        )
    entrypoint.write_bytes(b"print('bound')\n")
    extra = source / "extra.py"
    extra.write_text("extra\n")
    with pytest.raises(ValueError, match="source"):
        authority.preflight(
            root=source,
            approved_commit=commit,
            safe_evidence=safe,
            protocol=protocol,
            mode=runner_module.V5KaggleMode.CAPACITY_VALIDATION,
            stage=V5KaggleStage.AUTHORIZE,
        )
    assert stat.S_IMODE(entrypoint.stat().st_mode) == 0o755
