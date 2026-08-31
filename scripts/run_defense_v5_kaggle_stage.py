#!/usr/bin/env python3
"""Run exactly the next verified Sentinel v5 Kaggle checkpoint stage."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import stat
import subprocess
import sys
import time
from collections.abc import Iterable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from apar.evaluation.v5_checkpoint_storage import (  # noqa: E402
    V5CheckpointInput,
    V5CheckpointManifest,
    V5CheckpointObservation,
    publish_v5_checkpoint,
    read_v5_checkpoint_manifest,
)
from apar.evaluation.v5_kaggle_protocol import (  # noqa: E402
    V5KaggleEnvironmentBinding,
    V5KaggleMode,
    V5KaggleProtocol,
    V5KaggleStage,
    load_v5_kaggle_execution_manifest,
    load_v5_kaggle_protocol,
)
from apar.evaluation.v5_staged_evidence import (  # noqa: E402
    V5StageCapability,
    _issue_stage_capability,
    execute_v5_arm_stage,
    execute_v5_authorization_stage,
    execute_v5_control_stage,
    execute_v5_corpus_stage,
    execute_v5_feature_stage,
    execute_v5_finalize_stage,
    execute_v5_metric_arm_worker,
    execute_v5_metric_stage,
)
from apar.v5_independent_verifier import (  # noqa: E402
    verify_portable_evidence_bytes,
)

_PROTOCOL_PATH = Path("config/defense/defense-v5-kaggle-recovery.json")


def _canonical_bytes(document: object) -> bytes:
    return json.dumps(
        document,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode()


def _digest(document: object) -> str:
    return hashlib.sha256(_canonical_bytes(document)).hexdigest()


class V5KaggleStageAuthority(Protocol):
    """Injectable execution boundary; the public CLI uses only the frozen authority."""

    def preflight(
        self,
        *,
        root: Path,
        approved_commit: str,
        safe_evidence: Path,
        safe_deterministic_core_sha256: str,
        safe_observational_environment_sha256: str,
        protocol: V5KaggleProtocol,
        mode: V5KaggleMode,
        stage: V5KaggleStage,
    ) -> None: ...

    def environment(self, *, root: Path) -> V5KaggleEnvironmentBinding: ...

    def attempt_receipt_sha256(
        self,
        *,
        mode: V5KaggleMode,
        run_binding_sha256: str,
        approved_commit: str,
        safe_evidence: Path,
        execution_manifest_sha256: str,
    ) -> str: ...

    def records(
        self,
        *,
        root: Path,
        capability: V5StageCapability,
        stage_roots: Mapping[V5KaggleStage, Path],
    ) -> Iterable[V5CheckpointInput]: ...


class _FrozenRepositoryAuthority:
    def preflight(
        self,
        *,
        root: Path,
        approved_commit: str,
        safe_evidence: Path,
        safe_deterministic_core_sha256: str,
        safe_observational_environment_sha256: str,
        protocol: V5KaggleProtocol,
        mode: V5KaggleMode,
        stage: V5KaggleStage,
    ) -> None:
        del protocol, mode, stage
        if len(approved_commit) != 40 or any(
            character not in "0123456789abcdef" for character in approved_commit
        ):
            raise ValueError("approved commit must be an exact lowercase SHA-1")
        if (root / ".git").exists():
            head = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=root,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
            if head != approved_commit:
                raise ValueError("repository HEAD differs from approved commit")
            status = subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=root,
                check=True,
                capture_output=True,
                text=True,
            ).stdout
            if status:
                raise ValueError("repository worktree is not clean")
        else:
            self._verify_archived_source(root=root, approved_commit=approved_commit)
        if safe_evidence.is_symlink() or not safe_evidence.is_file():
            raise ValueError("approved safe evidence is missing or linked")
        verify_portable_evidence_bytes(
            safe_evidence.read_bytes(),
            root=root,
            expected_deterministic_core_sha256=(
                safe_deterministic_core_sha256
            ),
            expected_observational_environment_sha256=(
                safe_observational_environment_sha256
            ),
        )

    @staticmethod
    def _verify_archived_source(*, root: Path, approved_commit: str) -> None:
        manifest_value = os.environ.get("APAR_V5_SOURCE_MANIFEST_PATH")
        if not manifest_value:
            raise ValueError("archived source manifest path is absent")
        manifest_path = Path(manifest_value)
        if (
            manifest_path.is_symlink()
            or not manifest_path.is_file()
            or manifest_path.stat().st_nlink != 1
        ):
            raise ValueError("archived source manifest is missing or linked")
        try:
            document = json.loads(manifest_path.read_bytes())
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError("archived source manifest is malformed") from error
        if not isinstance(document, dict):
            raise ValueError("archived source manifest is not an object")
        claimed = document.pop("manifest_sha256", None)
        if (
            document.get("schema_version") != "apar-sentinel-v5-source-archive/1"
            or document.get("approved_commit") != approved_commit
            or claimed != _digest(document)
        ):
            raise ValueError("archived source manifest binding differs")
        entries = document.get("files")
        if not isinstance(entries, list) or not entries:
            raise ValueError("archived source file manifest is empty")
        expected_paths: list[str] = []
        for entry in entries:
            if not isinstance(entry, dict) or set(entry) != {
                "mode",
                "path",
                "sha256",
                "size_bytes",
            }:
                raise ValueError("archived source file entry is malformed")
            relative = entry["path"]
            if (
                not isinstance(relative, str)
                or Path(relative).is_absolute()
                or ".." in Path(relative).parts
                or "\\" in relative
            ):
                raise ValueError("archived source path is unsafe")
            path = root / relative
            if (
                path.is_symlink()
                or not path.is_file()
                or path.stat().st_nlink != 1
                or path.stat().st_size != entry["size_bytes"]
                or hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]
            ):
                raise ValueError("archived source file binding differs")
            expected_mode = 0o755 if entry["mode"] == "100755" else 0o644
            if entry["mode"] not in {"100644", "100755"} or (
                stat.S_IMODE(path.stat().st_mode) != expected_mode
            ):
                raise ValueError("archived source file mode differs")
            expected_paths.append(relative)
        if expected_paths != sorted(set(expected_paths)):
            raise ValueError("archived source paths are duplicated or reordered")
        observed_paths: list[str] = []
        for path in root.rglob("*"):
            if path.is_symlink():
                raise ValueError("archived source contains a linked entry")
            if path.is_file():
                observed_paths.append(path.relative_to(root).as_posix())
        if sorted(observed_paths) != expected_paths:
            raise ValueError("archived source path set differs")

    def environment(self, *, root: Path) -> V5KaggleEnvironmentBinding:
        del root
        required = {
            "image": "APAR_V5_KAGGLE_IMAGE",
            "image_sha256": "APAR_V5_KAGGLE_IMAGE_SHA256",
            "dependency_manifest_sha256": "APAR_V5_DEPENDENCY_MANIFEST_SHA256",
            "source_archive_sha256": "APAR_V5_SOURCE_ARCHIVE_SHA256",
            "notebook_sha256": "APAR_V5_NOTEBOOK_SHA256",
        }
        values: dict[str, str] = {}
        for field, variable in required.items():
            value = os.environ.get(variable)
            if not value:
                raise ValueError(f"frozen Kaggle environment variable is absent: {variable}")
            values[field] = value
        return V5KaggleEnvironmentBinding.bind(
            provider="kaggle",
            image=values["image"],
            image_sha256=values["image_sha256"],
            python_version=".".join(str(item) for item in sys.version_info[:3]),
            architecture="x86_64",
            cpu_count=os.cpu_count() or 1,
            dependency_manifest_sha256=values["dependency_manifest_sha256"],
            source_archive_sha256=values["source_archive_sha256"],
            notebook_sha256=values["notebook_sha256"],
            internet_enabled=False,
            accelerator="none",
            file_fsync_supported=True,
            directory_fsync_supported=True,
            hardlink_no_replace_supported=True,
        )

    def attempt_receipt_sha256(
        self,
        *,
        mode: V5KaggleMode,
        run_binding_sha256: str,
        approved_commit: str,
        safe_evidence: Path,
        execution_manifest_sha256: str,
    ) -> str:
        return _digest(
            {
                "schema_version": "apar-sentinel-v5-kaggle-stage-attempt/1",
                "mode": mode,
                "run_binding_sha256": run_binding_sha256,
                "approved_commit": approved_commit,
                "safe_evidence_sha256": hashlib.sha256(safe_evidence.read_bytes()).hexdigest(),
                "execution_manifest_sha256": execution_manifest_sha256,
            }
        )

    def records(
        self,
        *,
        root: Path,
        capability: V5StageCapability,
        stage_roots: Mapping[V5KaggleStage, Path],
    ) -> Iterable[V5CheckpointInput]:
        stage = capability.stage
        if stage is V5KaggleStage.AUTHORIZE:
            return execute_v5_authorization_stage(root=root, capability=capability)
        if stage is V5KaggleStage.CORPUS:
            return execute_v5_corpus_stage(root=root, capability=capability)
        if stage is V5KaggleStage.FEATURES:
            return execute_v5_feature_stage(
                root=root,
                capability=capability,
                corpus_checkpoint_root=stage_roots[V5KaggleStage.CORPUS],
            )
        if stage is V5KaggleStage.ARMS:
            return execute_v5_arm_stage(
                root=root,
                capability=capability,
                corpus_checkpoint_root=stage_roots[V5KaggleStage.CORPUS],
                feature_checkpoint_root=stage_roots[V5KaggleStage.FEATURES],
            )
        if stage in {
            V5KaggleStage.LABEL_SHUFFLE,
            V5KaggleStage.IDENTITY_RENAME,
            V5KaggleStage.FUTURE_CAUSALITY,
            V5KaggleStage.EQUAL_TIME_ISOLATION,
            V5KaggleStage.FEATURE_LEAKAGE,
            V5KaggleStage.SINGLE_CLASS_CONTROLS,
        }:
            return execute_v5_control_stage(
                root=root,
                capability=capability,
                corpus_checkpoint_root=stage_roots[V5KaggleStage.CORPUS],
            )
        if stage is V5KaggleStage.METRICS:
            return execute_v5_metric_stage(
                root=root,
                capability=capability,
                corpus_checkpoint_root=stage_roots[V5KaggleStage.CORPUS],
                arm_checkpoint_root=stage_roots[V5KaggleStage.ARMS],
                control_checkpoint_roots=tuple(
                    stage_roots[item]
                    for item in (
                        V5KaggleStage.LABEL_SHUFFLE,
                        V5KaggleStage.IDENTITY_RENAME,
                        V5KaggleStage.FUTURE_CAUSALITY,
                        V5KaggleStage.EQUAL_TIME_ISOLATION,
                        V5KaggleStage.FEATURE_LEAKAGE,
                        V5KaggleStage.SINGLE_CLASS_CONTROLS,
                    )
                ),
            )
        return execute_v5_finalize_stage(
            root=root,
            capability=capability,
            predecessor_checkpoint_roots=tuple(
                stage_roots[item] for item in tuple(V5KaggleStage)[:-1]
            ),
        )


def _read_input_chain(
    *, input_root: Path, protocol: V5KaggleProtocol
) -> tuple[dict[V5KaggleStage, Path], tuple[V5CheckpointManifest, ...]]:
    if input_root.is_symlink() or not input_root.is_dir():
        raise ValueError("checkpoint input root is not a real directory")
    observed = {item.name: item for item in input_root.iterdir()}
    known = {stage.value for stage in V5KaggleStage}
    if set(observed) - known:
        raise ValueError("checkpoint input root contains unknown entries")
    present = tuple(stage for stage in V5KaggleStage if stage.value in observed)
    if present != tuple(V5KaggleStage)[: len(present)]:
        raise ValueError("checkpoint input stages are missing or reordered")
    roots: dict[V5KaggleStage, Path] = {}
    manifests: list[V5CheckpointManifest] = []
    for stage in present:
        path = observed[stage.value]
        manifest = read_v5_checkpoint_manifest(output_root=path, limits=protocol.resources)
        if manifest.stage is not stage:
            raise ValueError("checkpoint directory/stage binding differs")
        if manifests and (manifest.predecessor_manifest_sha256 != manifests[-1].manifest_sha256):
            raise ValueError("checkpoint predecessor chain differs")
        roots[stage] = path
        manifests.append(manifest)
    return roots, tuple(manifests)


def _mode_from_predecessor(
    *, protocol: V5KaggleProtocol, predecessor: V5CheckpointManifest
) -> V5KaggleMode:
    matches = tuple(
        mode
        for mode in V5KaggleMode
        if protocol.run_binding_sha256(mode) == predecessor.run_binding_sha256
    )
    if len(matches) != 1:
        raise ValueError("checkpoint predecessor run mode is unknown")
    return matches[0]


def _rss_bytes() -> int:
    maximum = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if sys.platform != "darwin":
        maximum *= 1024
    return max(maximum, 1)


def _host_available_bytes() -> int:
    memory = Path("/proc/meminfo")
    if memory.is_file():
        for line in memory.read_text().splitlines():
            if line.startswith("MemAvailable:"):
                return max(int(line.split()[1]) * 1024, 1)
    page_size = int(os.sysconf("SC_PAGE_SIZE"))
    try:
        pages = int(os.sysconf("SC_AVPHYS_PAGES"))
    except ValueError:
        pages = int(os.sysconf("SC_PHYS_PAGES"))
    return max(page_size * pages, 1)


class _StageTelemetry:
    def __init__(self, environment: V5KaggleEnvironmentBinding) -> None:
        self._environment = environment
        self._started_wall = time.perf_counter()
        self._started_at = datetime.now(UTC)
        self._rss_start = _rss_bytes()
        self._available_start = _host_available_bytes()

    def observation(self) -> V5CheckpointObservation:
        completed = datetime.now(UTC)
        rss_end = _rss_bytes()
        available_end = _host_available_bytes()
        return V5CheckpointObservation(
            schema_version="apar-sentinel-v5-kaggle-observation/1",
            started_at_utc=self._started_at.isoformat().replace("+00:00", "Z"),
            completed_at_utc=completed.isoformat().replace("+00:00", "Z"),
            wall_seconds=max(time.perf_counter() - self._started_wall, 1e-9),
            rss_samples_bytes=(self._rss_start, rss_end),
            host_available_samples_bytes=(
                self._available_start,
                available_end,
            ),
            peak_rss_bytes=max(self._rss_start, rss_end),
            environment=self._environment,
        )


def execute_next_v5_kaggle_stage(
    *,
    root: Path,
    input_root: Path,
    output_root: Path,
    safe_evidence: Path,
    execution_manifest: Path,
    approved_commit: str,
    authority: V5KaggleStageAuthority,
) -> V5CheckpointManifest:
    """Infer, execute, and exclusively publish only the next valid stage."""
    root = root.resolve()
    protocol = load_v5_kaggle_protocol(root / _PROTOCOL_PATH, root=root)
    execution = load_v5_kaggle_execution_manifest(
        execution_manifest,
        safe_evidence=safe_evidence,
        approved_commit=approved_commit,
        protocol=protocol,
    )
    mode = execution.execution_mode
    stage_roots, manifests = _read_input_chain(input_root=input_root, protocol=protocol)
    predecessor = manifests[-1] if manifests else None
    if predecessor is None:
        stage = V5KaggleStage.AUTHORIZE
    else:
        if len(manifests) == len(V5KaggleStage):
            raise FileExistsError("checkpoint chain is already complete")
        stage = tuple(V5KaggleStage)[len(manifests)]
        predecessor_mode = _mode_from_predecessor(
            protocol=protocol, predecessor=predecessor
        )
        if mode is not predecessor_mode:
            raise ValueError("execution manifest mode differs from predecessor chain")
    if os.path.lexists(output_root):
        raise FileExistsError("checkpoint output root already exists")
    authority.preflight(
        root=root,
        approved_commit=approved_commit,
        safe_evidence=safe_evidence,
        safe_deterministic_core_sha256=(
            execution.safe_deterministic_core_sha256
        ),
        safe_observational_environment_sha256=(
            execution.safe_observational_environment_sha256
        ),
        protocol=protocol,
        mode=mode,
        stage=stage,
    )
    run_binding = protocol.run_binding_sha256(mode)
    current_attempt_receipt = authority.attempt_receipt_sha256(
        mode=mode,
        run_binding_sha256=run_binding,
        approved_commit=approved_commit,
        safe_evidence=safe_evidence,
        execution_manifest_sha256=execution.manifest_sha256,
    )
    if (
        predecessor is not None
        and predecessor.attempt_receipt_sha256 != current_attempt_receipt
    ):
        raise ValueError("execution manifest differs from predecessor attempt")
    attempt_receipt = current_attempt_receipt
    capability = _issue_stage_capability(
        protocol=protocol,
        mode=mode,
        attempt_receipt_sha256=attempt_receipt,
        predecessor=predecessor,
        execution_manifest_sha256=execution.manifest_sha256,
    )
    if capability.stage is not stage:
        raise PermissionError("issued capability differs from inferred next stage")
    environment = authority.environment(root=root)
    telemetry = _StageTelemetry(environment)
    records = authority.records(
        root=root,
        capability=capability,
        stage_roots=stage_roots,
    )
    manifest = publish_v5_checkpoint(
        output_root=output_root,
        stage=stage,
        run_binding_sha256=run_binding,
        attempt_receipt_sha256=attempt_receipt,
        predecessor=predecessor,
        records=records,
        environment=environment,
        observation_factory=telemetry.observation,
        limits=protocol.resources,
    )
    replay = read_v5_checkpoint_manifest(output_root=output_root, limits=protocol.resources)
    if replay != manifest:
        raise ValueError("published checkpoint replay differs")
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--safe-evidence", type=Path, required=True)
    parser.add_argument("--execution-manifest", type=Path, required=True)
    parser.add_argument("--approved-commit", required=True)
    return parser


def _internal_metric_worker_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--mode", type=V5KaggleMode, required=True)
    parser.add_argument("--run-binding-sha256", required=True)
    parser.add_argument("--attempt-receipt-sha256", required=True)
    parser.add_argument("--execution-manifest-sha256", required=True)
    parser.add_argument("--arm-checkpoint-root", type=Path, required=True)
    parser.add_argument("--arm-manifest-sha256", required=True)
    parser.add_argument("--arm-manifest-deterministic-sha256", required=True)
    parser.add_argument("--target-arm", required=True)
    parser.add_argument("--receipt-path", type=Path, required=True)
    parser.add_argument("--max-address-space-bytes", type=int, required=True)
    return parser


def _write_exclusive_canonical(path: Path, document: object) -> None:
    if path.parent.is_symlink() or not path.parent.is_dir() or os.path.lexists(path):
        raise FileExistsError("metric worker receipt path is unsafe")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        content = _canonical_bytes(document)
        view = memoryview(content)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("metric worker receipt write did not advance")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _run_internal_metric_worker(argv: Sequence[str]) -> int:
    arguments = _internal_metric_worker_parser().parse_args(argv)
    if arguments.max_address_space_bytes <= 0:
        return 2
    if sys.platform.startswith("linux") and hasattr(resource, "RLIMIT_AS"):
        _soft, hard = resource.getrlimit(resource.RLIMIT_AS)
        requested = arguments.max_address_space_bytes
        if hard != resource.RLIM_INFINITY:
            requested = min(requested, hard)
        resource.setrlimit(resource.RLIMIT_AS, (requested, hard))
    receipt = execute_v5_metric_arm_worker(
        root=arguments.root.resolve(),
        mode=arguments.mode,
        run_binding_sha256=arguments.run_binding_sha256,
        attempt_receipt_sha256=arguments.attempt_receipt_sha256,
        execution_manifest_sha256=arguments.execution_manifest_sha256,
        arm_checkpoint_root=arguments.arm_checkpoint_root.resolve(),
        arm_manifest_sha256=arguments.arm_manifest_sha256,
        arm_manifest_deterministic_sha256=(
            arguments.arm_manifest_deterministic_sha256
        ),
        target_arm=arguments.target_arm,
    )
    _write_exclusive_canonical(
        arguments.receipt_path,
        receipt.model_dump(mode="json"),
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    raw_arguments = tuple(sys.argv[1:] if argv is None else argv)
    if raw_arguments and raw_arguments[0] == "--internal-metric-worker":
        try:
            return _run_internal_metric_worker(raw_arguments[1:])
        except Exception:  # noqa: BLE001 - child failures cross only as a status code
            return 2
    arguments = _parser().parse_args(raw_arguments)
    manifest = execute_next_v5_kaggle_stage(
        root=arguments.root,
        input_root=arguments.input_root,
        output_root=arguments.output_root,
        safe_evidence=arguments.safe_evidence,
        execution_manifest=arguments.execution_manifest,
        approved_commit=arguments.approved_commit,
        authority=_FrozenRepositoryAuthority(),
    )
    print(
        json.dumps(
            {
                "stage": manifest.stage,
                "manifest_sha256": manifest.manifest_sha256,
                "deterministic_sha256": manifest.deterministic_sha256,
                "observation_sha256": manifest.observation_sha256,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "V5KaggleStageAuthority",
    "execute_next_v5_kaggle_stage",
    "main",
]
