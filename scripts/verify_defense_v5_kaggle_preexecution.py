#!/usr/bin/env python3
"""Verify the closed linear-source or frozen Kaggle Sentinel v5 boundary."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import os
import re
import subprocess
import sys
import tarfile
import tempfile
from collections.abc import Sequence
from enum import StrEnum
from pathlib import Path
from typing import Literal, Self, cast

from pydantic import BaseModel, ConfigDict, Field, model_validator

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from apar.evaluation.v5_kaggle_protocol import (  # noqa: E402
    V5KaggleExecutionManifest,
    V5KaggleMode,
    V5KaggleProtocol,
    V5KaggleStage,
    load_v5_kaggle_protocol,
)
from apar.v5_kaggle_independent_verifier import (  # noqa: E402
    V5KaggleVerificationReport,
    verify_v5_kaggle_evidence,
)
from scripts.build_defense_v5_kaggle_notebooks import (  # noqa: E402
    build_v5_kaggle_notebooks,
)

_PROTOCOL_PATH = Path("config/defense/defense-v5-kaggle-recovery.json")
_PREREGISTRATION_PATH = Path("config/defense/defense-v5-kaggle-preregistration.json")
_RECOVERY_COMMIT = "7318d425032b55d31a57bb1a00a880a4468bce89"
_ARCHIVE_PREFIX = "apar-v5-source"
_HEX_64 = re.compile(r"^[0-9a-f]{64}$")
_WHEEL_NAME = re.compile(
    r"^(?P<distribution>[A-Za-z0-9_.]+)-(?P<version>[A-Za-z0-9_.!+]+)-"
    r"(?:(?P<build>[A-Za-z0-9_.]+)-)?(?P<python>[A-Za-z0-9_.]+)-"
    r"(?P<abi>[A-Za-z0-9_.]+)-(?P<platform>[A-Za-z0-9_.]+)\.whl$"
)
_REQUIRED_WHEEL_DISTRIBUTIONS = {
    "apar",
    "catboost",
    "cryptography",
    "fastapi",
    "numpy",
    "pandas",
    "pydantic",
    "pyarrow",
    "scikit-learn",
}


class _FrozenModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class V5KagglePreexecutionPhase(StrEnum):
    SOURCE = "source"
    FROZEN = "frozen"


class V5SourceFile(_FrozenModel):
    path: str = Field(min_length=1, max_length=1024)
    mode: Literal["100644", "100755"]
    size_bytes: int = Field(ge=0)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class V5SourceArchiveManifest(_FrozenModel):
    schema_version: Literal["apar-sentinel-v5-source-archive/1"]
    artifact_name: Literal["apar-v5-source3.tar.gz"]
    artifact_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    artifact_size_bytes: int = Field(gt=0)
    archive_prefix: Literal["apar-v5-source/"]
    approved_commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    source_tree: str = Field(pattern=r"^[0-9a-f]{40}$")
    files: tuple[V5SourceFile, ...]
    manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def exact_order_and_digest(self) -> Self:
        if tuple(item.path for item in self.files) != tuple(
            sorted(item.path for item in self.files)
        ):
            raise ValueError("source manifest paths are not canonical")
        if self.manifest_sha256 != _digest(
            self.model_dump(mode="json", exclude={"manifest_sha256"})
        ):
            raise ValueError("source manifest self-digest differs")
        return self


class V5WheelFile(_FrozenModel):
    filename: str = Field(min_length=1, max_length=512)
    distribution: str = Field(min_length=1, max_length=256)
    version: str = Field(min_length=1, max_length=128)
    python_tag: str = Field(min_length=1, max_length=128)
    abi_tag: str = Field(min_length=1, max_length=128)
    platform_tag: str = Field(min_length=1, max_length=256)
    size_bytes: int = Field(gt=0)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class V5WheelhouseManifest(_FrozenModel):
    schema_version: Literal["apar-sentinel-v5-wheelhouse/1"]
    python_version: Literal["3.12"]
    architecture: Literal["x86_64"]
    wheels: tuple[V5WheelFile, ...]
    manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def exact_order_and_digest(self) -> Self:
        if not self.wheels or tuple(item.filename for item in self.wheels) != tuple(
            sorted(item.filename for item in self.wheels)
        ):
            raise ValueError("wheelhouse filenames are empty or noncanonical")
        if self.manifest_sha256 != _digest(
            self.model_dump(mode="json", exclude={"manifest_sha256"})
        ):
            raise ValueError("wheelhouse manifest self-digest differs")
        return self


class V5KagglePreexecutionReport(_FrozenModel):
    schema_version: Literal["apar-sentinel-v5-kaggle-preexecution/1"]
    valid: Literal[True]
    phase: V5KagglePreexecutionPhase
    source_commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    source_tree: str = Field(pattern=r"^[0-9a-f]{40}$")
    recovery_commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    protocol_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_archive_sha256: str | None
    wheelhouse_manifest_sha256: str | None
    execution_manifest_sha256: str | None
    rehearsal_chain_roots: tuple[str, ...]
    deterministic_stage_sha256: tuple[tuple[str, str], ...]
    observed_max_peak_rss_bytes: int | None = Field(default=None, gt=0)
    observed_max_stage_seconds: float | None = Field(default=None, gt=0)
    observed_max_stage_output_bytes: int | None = Field(default=None, gt=0)
    seed_2404_boundary: Literal["asserted_only"]
    successor_outputs_absent: Literal[True]


def _canonical_bytes(document: object, *, newline: bool = False) -> bytes:
    encoded = json.dumps(
        document,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode()
    return encoded + (b"\n" if newline else b"")


def _digest(document: object) -> str:
    return hashlib.sha256(_canonical_bytes(document)).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _git(root: Path, *arguments: str, binary: bool = False) -> str | bytes:
    try:
        completed = subprocess.run(
            ["git", *arguments],
            cwd=root,
            check=True,
            capture_output=True,
            text=not binary,
        )
    except subprocess.CalledProcessError as error:
        raise ValueError("Git source binding cannot be resolved") from error
    if binary:
        return cast(bytes, completed.stdout)
    return cast(str, completed.stdout).strip()


def _resolve_commit(root: Path, commit: str) -> str:
    resolved = _git(root, "rev-parse", "--verify", f"{commit}^{{commit}}")
    assert isinstance(resolved, str)
    if not re.fullmatch(r"[0-9a-f]{40}", resolved):
        raise ValueError("resolved source commit is malformed")
    return resolved


def _source_files(root: Path, commit: str) -> tuple[tuple[V5SourceFile, bytes], ...]:
    raw = _git(root, "ls-tree", "-rz", "--full-tree", commit, binary=True)
    assert isinstance(raw, bytes)
    records: list[tuple[V5SourceFile, bytes]] = []
    for item in raw.split(b"\0"):
        if not item:
            continue
        header, raw_path = item.split(b"\t", 1)
        mode, kind, object_id = header.decode("ascii").split(" ")
        try:
            relative = raw_path.decode("utf-8")
        except UnicodeDecodeError as error:
            raise ValueError("source tree contains a non-UTF-8 path") from error
        if kind != "blob" or mode not in {"100644", "100755"}:
            raise ValueError("source tree contains a non-regular tracked entry")
        path = Path(relative)
        if path.is_absolute() or ".." in path.parts or "\\" in relative:
            raise ValueError("source tree path is unsafe")
        content = _git(root, "cat-file", "blob", object_id, binary=True)
        assert isinstance(content, bytes)
        canonical_mode: Literal["100644", "100755"] = (
            "100755" if mode == "100755" else "100644"
        )
        records.append(
            (
                V5SourceFile(
                    path=relative,
                    mode=canonical_mode,
                    size_bytes=len(content),
                    sha256=hashlib.sha256(content).hexdigest(),
                ),
                content,
            )
        )
    records.sort(key=lambda pair: pair[0].path)
    if not records:
        raise ValueError("source tree is empty")
    return tuple(records)


def _archive_bytes(records: tuple[tuple[V5SourceFile, bytes], ...]) -> bytes:
    uncompressed = io.BytesIO()
    with tarfile.open(fileobj=uncompressed, mode="w", format=tarfile.USTAR_FORMAT) as archive:
        for source, content in records:
            info = tarfile.TarInfo(f"{_ARCHIVE_PREFIX}/{source.path}")
            info.size = len(content)
            info.mode = 0o755 if source.mode == "100755" else 0o644
            info.mtime = 0
            info.uid = 0
            info.gid = 0
            info.uname = ""
            info.gname = ""
            archive.addfile(info, io.BytesIO(content))
    compressed = io.BytesIO()
    with gzip.GzipFile(
        filename="", mode="wb", fileobj=compressed, compresslevel=9, mtime=0
    ) as stream:
        stream.write(uncompressed.getvalue())
    return compressed.getvalue()


def _exclusive_write(path: Path, content: bytes) -> None:
    if os.path.lexists(path):
        raise FileExistsError(f"output already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        raise
    directory = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def build_v5_source_archive(*, root: Path, commit: str, output: Path) -> V5SourceArchiveManifest:
    """Build a byte-reproducible archive solely from an immutable Git tree."""
    if os.path.lexists(output):
        raise FileExistsError(f"source archive output already exists: {output}")
    resolved = _resolve_commit(root.resolve(), commit)
    tree = _git(root.resolve(), "rev-parse", f"{resolved}^{{tree}}")
    assert isinstance(tree, str)
    records = _source_files(root.resolve(), resolved)
    content = _archive_bytes(records)
    values: dict[str, object] = {
        "schema_version": "apar-sentinel-v5-source-archive/1",
        "artifact_name": "apar-v5-source3.tar.gz",
        "artifact_sha256": hashlib.sha256(content).hexdigest(),
        "artifact_size_bytes": len(content),
        "archive_prefix": f"{_ARCHIVE_PREFIX}/",
        "approved_commit": resolved,
        "source_tree": tree,
        "files": [item.model_dump(mode="json") for item, _ in records],
    }
    values["manifest_sha256"] = _digest(values)
    manifest = V5SourceArchiveManifest.model_validate(values)
    _exclusive_write(output, content)
    return manifest


def _wheel_file(path: Path) -> V5WheelFile:
    match = _WHEEL_NAME.fullmatch(path.name)
    if match is None:
        raise ValueError(f"wheel filename is not canonical: {path.name}")
    platform_tag = match.group("platform")
    python_tag = match.group("python")
    if any(token in platform_tag.lower() for token in ("macosx", "win", "aarch64")):
        raise ValueError("wheelhouse must contain only Linux x86_64 or universal wheels")
    if platform_tag != "any" and not any(
        token in platform_tag.lower() for token in ("linux", "manylinux", "musllinux")
    ):
        raise ValueError("wheelhouse platform is not Linux x86_64")
    if "x86_64" not in platform_tag and platform_tag != "any":
        raise ValueError("wheelhouse architecture is not x86_64")
    if not any(token in python_tag.split(".") for token in ("cp312", "py3", "py312")):
        raise ValueError("wheelhouse Python tag is not CPython 3.12 compatible")
    if path.is_symlink() or not path.is_file() or path.stat().st_nlink != 1:
        raise ValueError("wheelhouse contains a linked or non-file entry")
    return V5WheelFile(
        filename=path.name,
        distribution=match.group("distribution").replace("_", "-").lower(),
        version=match.group("version"),
        python_tag=python_tag,
        abi_tag=match.group("abi"),
        platform_tag=platform_tag,
        size_bytes=path.stat().st_size,
        sha256=_sha256(path),
    )


def build_v5_wheelhouse_manifest(*, wheelhouse: Path, write: bool) -> V5WheelhouseManifest:
    """Validate and optionally publish the canonical offline wheelhouse manifest."""
    if wheelhouse.is_symlink() or not wheelhouse.is_dir():
        raise ValueError("wheelhouse is not a real directory")
    unknown = tuple(
        item
        for item in wheelhouse.iterdir()
        if item.name != "wheelhouse-manifest.json" and item.suffix != ".whl"
    )
    if unknown:
        raise ValueError("wheelhouse contains a non-wheel entry")
    wheels = tuple(
        _wheel_file(item) for item in sorted(wheelhouse.glob("*.whl"), key=lambda path: path.name)
    )
    distributions = {item.distribution for item in wheels}
    missing = sorted(_REQUIRED_WHEEL_DISTRIBUTIONS - distributions)
    if missing:
        raise ValueError(
            "wheelhouse is missing required distributions: " + ",".join(missing)
        )
    values: dict[str, object] = {
        "schema_version": "apar-sentinel-v5-wheelhouse/1",
        "python_version": "3.12",
        "architecture": "x86_64",
        "wheels": [item.model_dump(mode="json") for item in wheels],
    }
    values["manifest_sha256"] = _digest(values)
    manifest = V5WheelhouseManifest.model_validate(values)
    manifest_path = wheelhouse / "wheelhouse-manifest.json"
    if manifest_path.exists():
        if manifest_path.is_symlink() or manifest_path.stat().st_nlink != 1:
            raise ValueError("wheelhouse manifest is linked")
        retained = V5WheelhouseManifest.model_validate_json(manifest_path.read_bytes())
        if retained != manifest or manifest_path.read_bytes() != _canonical_bytes(
            manifest.model_dump(mode="json"), newline=True
        ):
            raise ValueError("retained wheelhouse manifest differs")
    elif write:
        _exclusive_write(
            manifest_path,
            _canonical_bytes(manifest.model_dump(mode="json"), newline=True),
        )
    return manifest


def build_v5_safe_evidence_input_manifest(
    *,
    safe_evidence: Path,
    mode: V5KaggleMode,
    approved_commit: str,
    protocol_sha256: str,
    run_binding_sha256: str,
    successor_authorization_sha256: str | None = None,
) -> V5KaggleExecutionManifest:
    """Bind an existing approved safe artifact without changing its bytes."""
    if (
        safe_evidence.name != "safe-evidence.json"
        or safe_evidence.is_symlink()
        or not safe_evidence.is_file()
        or safe_evidence.stat().st_nlink != 1
    ):
        raise ValueError("safe evidence input path is not an exact unlinked file")
    values: dict[str, object] = {
        "schema_version": "apar-sentinel-v5-kaggle-execution-input/1",
        "execution_mode": mode,
        "profile": "production",
        "development_test_seed": (
            404 if mode is V5KaggleMode.CAPACITY_VALIDATION else 2404
        ),
        "authorization_required": mode is V5KaggleMode.LOCKED_SUCCESSOR,
        "successor_authorization_sha256": successor_authorization_sha256,
        "approved_commit": approved_commit,
        "protocol_sha256": protocol_sha256,
        "run_binding_sha256": run_binding_sha256,
        "artifact_name": "safe-evidence.json",
        "artifact_size_bytes": safe_evidence.stat().st_size,
        "artifact_sha256": _sha256(safe_evidence),
    }
    values["manifest_sha256"] = _digest(values)
    return V5KaggleExecutionManifest.model_validate(values)


def _verify_clean_head(root: Path, expected_head: str) -> tuple[str, str]:
    status = _git(root, "status", "--porcelain=v1", "--untracked-files=all")
    assert isinstance(status, str)
    if status:
        raise ValueError("repository worktree is not clean")
    actual = _resolve_commit(root, "HEAD")
    expected = _resolve_commit(root, expected_head)
    if actual != expected:
        raise ValueError("repository HEAD differs from the approved head")
    tree = _git(root, "rev-parse", f"{actual}^{{tree}}")
    assert isinstance(tree, str)
    return actual, tree


def _sole_parent(root: Path, commit: str) -> str:
    line = _git(root, "rev-list", "--parents", "-n", "1", commit)
    assert isinstance(line, str)
    values = line.split()
    if len(values) != 2:
        raise ValueError("approved commit does not have exactly one parent")
    return values[1]


def _linear_source_lineage(
    root: Path, *, source_commit: str, recovery_commit: str
) -> tuple[str, ...]:
    """Return the exact no-merge source series after the preserved recovery commit."""
    source = _resolve_commit(root, source_commit)
    recovery = _resolve_commit(root, recovery_commit)
    if source == recovery:
        raise ValueError("source lineage contains no source commit")
    reverse_lineage: list[str] = []
    visited: set[str] = set()
    current = source
    while current != recovery:
        if current in visited:
            raise ValueError("source lineage contains a cycle")
        visited.add(current)
        reverse_lineage.append(current)
        try:
            current = _sole_parent(root, current)
        except ValueError as error:
            raise ValueError(
                "source is not a linear descendant of the preserved recovery commit"
            ) from error
    return tuple(reversed(reverse_lineage))


def _verify_successor_outputs_absent(root: Path) -> None:
    paths = (
        "docs/experiments/defense-v5-kaggle-successor-attempt.json",
        "docs/experiments/defense-v5-kaggle-successor-checkpoints",
        "docs/experiments/defense-v5-kaggle-development-candidate.manifest.json",
        "docs/experiments/defense-v5-kaggle-development-candidate.manifest.json.chunks",
        "docs/experiments/defense-v5-kaggle-development-summary.json",
    )
    for relative in paths:
        if os.path.lexists(root / relative):
            raise FileExistsError(f"successor output is already present: {relative}")


def _verify_notebook_metadata_document(
    document: object,
    *,
    stage: V5KaggleStage,
    predecessor: V5KaggleStage | None,
    owner: str,
    source_slug: str,
    wheelhouse_slug: str,
    safe_slug: str,
) -> None:
    if not isinstance(document, dict):
        raise ValueError("notebook metadata is not an object")
    if document.get("is_private") is not True:
        raise ValueError("notebook metadata is not private")
    if (
        document.get("enable_internet") is not False
        or document.get("enable_gpu") is not False
        or document.get("enable_tpu") is not False
        or document.get("machine_shape") != ""
        or document.get("accelerator") != "none"
    ):
        raise ValueError("notebook metadata is not offline CPU-only")
    expected_id = f"{owner}/apar-sentinel-v5-{stage.value.replace('_', '-')}"
    if document.get("id") != expected_id:
        raise ValueError("notebook identity differs")
    expected_datasets = [
        f"{owner}/{source_slug}",
        f"{owner}/{wheelhouse_slug}",
        f"{owner}/{safe_slug}",
    ]
    if document.get("dataset_sources") != expected_datasets:
        raise ValueError("notebook private inputs differ")
    expected_predecessors = (
        []
        if predecessor is None
        else [f"{owner}/apar-sentinel-v5-{predecessor.value.replace('_', '-')}"]
    )
    if document.get("kernel_sources") != expected_predecessors:
        raise ValueError("notebook predecessor input differs")


def _verify_notebooks(root: Path, protocol: V5KaggleProtocol) -> None:
    private = protocol.private_inputs
    with tempfile.TemporaryDirectory(prefix="apar-v5-notebooks-") as temporary:
        output = Path(temporary) / "generated"
        generated = build_v5_kaggle_notebooks(
            root=root,
            output_dir=output,
            owner_slug=private.owner_slug,
            source_dataset_slug=private.source_dataset_slug,
            wheelhouse_dataset_slug=private.wheelhouse_dataset_slug,
            safe_evidence_dataset_slug=private.safe_evidence_dataset_slug,
        )
        for index, item in enumerate(generated):
            retained_notebook = root / "kaggle/defense_v5" / item.notebook_path.name
            retained_metadata = root / "kaggle/defense_v5" / item.metadata_path.name
            if (
                retained_notebook.is_symlink()
                or retained_metadata.is_symlink()
                or retained_notebook.read_bytes() != item.notebook_path.read_bytes()
                or retained_metadata.read_bytes() != item.metadata_path.read_bytes()
            ):
                raise ValueError("generated notebook bytes differ from committed bytes")
            predecessor = None if index == 0 else tuple(V5KaggleStage)[index - 1]
            _verify_notebook_metadata_document(
                json.loads(retained_metadata.read_bytes()),
                stage=item.stage,
                predecessor=predecessor,
                owner=private.owner_slug,
                source_slug=private.source_dataset_slug,
                wheelhouse_slug=private.wheelhouse_dataset_slug,
                safe_slug=private.safe_evidence_dataset_slug,
            )


def _verify_source_archive(
    *, root: Path, source_commit: str, archive: Path
) -> V5SourceArchiveManifest:
    if (
        archive.name != "apar-v5-source3.tar.gz"
        or archive.is_symlink()
        or not archive.is_file()
        or archive.stat().st_nlink != 1
    ):
        raise ValueError("source archive is not an exact unlinked canonical file")
    with tempfile.TemporaryDirectory(prefix="apar-v5-source-audit-") as temporary:
        rebuilt = Path(temporary) / "apar-v5-source3.tar.gz"
        manifest = build_v5_source_archive(
            root=root,
            commit=source_commit,
            output=rebuilt,
        )
        if rebuilt.read_bytes() != archive.read_bytes():
            raise ValueError("source archive bytes differ from the approved Git tree")
    return manifest


def _rehearsal_observations(
    roots: Sequence[Path], protocol: V5KaggleProtocol
) -> tuple[int, float, int]:
    maximum_rss = 0
    maximum_seconds = 0.0
    maximum_output = 0
    for chain_root in roots:
        for stage in V5KaggleStage:
            checkpoint = chain_root / stage.value
            observation = json.loads((checkpoint / "observational.json").read_bytes())
            manifest = json.loads((checkpoint / "checkpoint.manifest.json").read_bytes())
            maximum_rss = max(maximum_rss, int(observation["peak_rss_bytes"]))
            maximum_seconds = max(maximum_seconds, float(observation["wall_seconds"]))
            size = sum(
                int(item["compressed_size_bytes"])
                for item in (*manifest["deterministic_chunks"], *manifest["observational_chunks"])
            )
            size += (checkpoint / "observational.json").stat().st_size
            size += (checkpoint / "checkpoint.manifest.json").stat().st_size
            maximum_output = max(maximum_output, size)
    if maximum_rss >= protocol.resources.max_peak_rss_bytes:
        raise ValueError("capacity rehearsal peak RSS exceeds the frozen gate")
    if maximum_seconds >= protocol.resources.max_stage_seconds:
        raise ValueError("capacity rehearsal wall time exceeds the frozen gate")
    if maximum_output >= protocol.resources.max_stage_output_bytes:
        raise ValueError("capacity rehearsal output exceeds the frozen gate")
    return maximum_rss, maximum_seconds, maximum_output


def _verify_rehearsals(
    *, root: Path, roots: Sequence[Path], protocol: V5KaggleProtocol
) -> tuple[
    tuple[V5KaggleVerificationReport, V5KaggleVerificationReport],
    tuple[int, float, int],
]:
    if len(roots) != 2:
        raise ValueError("frozen preexecution requires exactly two rehearsals")
    reports = tuple(
        verify_v5_kaggle_evidence(
            root=root,
            expected_mode="kaggle_capacity_validation",
            checkpoint_roots=tuple(chain / stage.value for stage in tuple(V5KaggleStage)[:-1]),
            final_root=chain / V5KaggleStage.FINALIZE.value,
        )
        for chain in roots
    )
    if len(reports) != 2:
        raise AssertionError("two rehearsal reports were not produced")
    first, second = reports
    if first.deterministic_stage_sha256 != second.deterministic_stage_sha256:
        raise ValueError("capacity rehearsal deterministic stage digests differ")
    if first.execution_manifest_sha256 != second.execution_manifest_sha256:
        raise ValueError("capacity rehearsal execution manifests differ")
    return (first, second), _rehearsal_observations(roots, protocol)


def _read_preregistration(root: Path) -> dict[str, object]:
    path = root / _PREREGISTRATION_PATH
    try:
        document = json.loads(path.read_bytes())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("Kaggle preregistration is absent or malformed") from error
    if not isinstance(document, dict):
        raise ValueError("Kaggle preregistration is not an object")
    claimed = document.get("preregistration_sha256")
    unsigned = dict(document)
    unsigned.pop("preregistration_sha256", None)
    if claimed != _digest(unsigned):
        raise ValueError("Kaggle preregistration self-digest differs")
    return document


def verify_v5_kaggle_preexecution(
    *,
    root: Path,
    phase: V5KagglePreexecutionPhase,
    expected_head: str,
    source_archive: Path | None,
    wheelhouse: Path | None,
    rehearsal_chain_roots: Sequence[Path],
    expected_recovery_commit: str = _RECOVERY_COMMIT,
) -> V5KagglePreexecutionReport:
    """Verify linear source or frozen topology without issuing a capability."""
    root = root.resolve()
    if phase is V5KagglePreexecutionPhase.FROZEN and len(rehearsal_chain_roots) != 2:
        raise ValueError("frozen preexecution requires exactly two rehearsals")
    head, tree = _verify_clean_head(root, expected_head)
    protocol = load_v5_kaggle_protocol(root / _PROTOCOL_PATH, root=root)
    if protocol.locked.development_test_seed != 2404:
        raise ValueError("locked successor seed differs from the asserted binding")
    _verify_successor_outputs_absent(root)
    _verify_notebooks(root, protocol)

    archive_digest: str | None = None
    wheel_digest: str | None = None
    rehearsal_roots: tuple[str, ...] = ()
    deterministic: tuple[tuple[str, str], ...] = ()
    execution_manifest_digest: str | None = None
    observed: tuple[int, float, int] | None = None
    source_commit = head
    recovery_commit = expected_recovery_commit

    if phase is V5KagglePreexecutionPhase.SOURCE:
        if rehearsal_chain_roots:
            raise ValueError("source preexecution cannot accept rehearsals")
        _linear_source_lineage(
            root,
            source_commit=head,
            recovery_commit=expected_recovery_commit,
        )
        if os.path.lexists(root / _PREREGISTRATION_PATH):
            raise FileExistsError("Kaggle preregistration exists before source freeze")
        if source_archive is None or wheelhouse is None:
            raise ValueError("source audit requires archive and wheelhouse inputs")
        source_manifest = _verify_source_archive(
            root=root,
            source_commit=head,
            archive=source_archive,
        )
        wheel_manifest = build_v5_wheelhouse_manifest(
            wheelhouse=wheelhouse,
            write=False,
        )
        archive_digest = source_manifest.artifact_sha256
        wheel_digest = wheel_manifest.manifest_sha256
    else:
        preregistration = _read_preregistration(root)
        source = preregistration.get("source")
        if not isinstance(source, dict):
            raise ValueError("preregistration source binding is absent")
        source_commit = str(source.get("commit"))
        recovery_commit = str(source.get("recovery_commit"))
        source_lineage = _linear_source_lineage(
            root,
            source_commit=source_commit,
            recovery_commit=recovery_commit,
        )
        if (
            _sole_parent(root, head) != source_commit
            or recovery_commit != expected_recovery_commit
            or source.get("lineage") != list(source_lineage)
            or _git(root, "rev-parse", f"{source_commit}^{{tree}}") != source.get("tree")
        ):
            raise ValueError("frozen source/preregistration chronology differs")
        changed = _git(root, "diff-tree", "--no-commit-id", "--name-only", "-r", head)
        assert isinstance(changed, str)
        if changed.splitlines() != [_PREREGISTRATION_PATH.as_posix()]:
            raise ValueError("preregistration commit changed more than its one path")
        reports, observed = _verify_rehearsals(
            root=root,
            roots=rehearsal_chain_roots,
            protocol=protocol,
        )
        rehearsal_roots = tuple(item.chain_root_sha256 for item in reports)
        deterministic = reports[0].deterministic_stage_sha256
        expected_rehearsals = preregistration.get("capacity_rehearsals")
        if not isinstance(expected_rehearsals, dict):
            raise ValueError("preregistration capacity evidence is absent")
        if expected_rehearsals.get("chain_roots") != list(rehearsal_roots):
            raise ValueError("preregistration rehearsal chain roots differ")
        if expected_rehearsals.get("deterministic_stage_sha256") != [
            list(item) for item in deterministic
        ]:
            raise ValueError("preregistration deterministic stage evidence differs")
        execution_manifest_digest = reports[0].execution_manifest_sha256
        if expected_rehearsals.get("execution_manifest_sha256") != (
            execution_manifest_digest
        ):
            raise ValueError("preregistration capacity execution manifest differs")
        private_inputs = preregistration.get("private_inputs")
        if not isinstance(private_inputs, dict):
            raise ValueError("preregistration private input evidence is absent")
        archive_digest = str(private_inputs.get("source_archive_sha256"))
        wheel_digest = str(private_inputs.get("wheelhouse_manifest_sha256"))
        if _HEX_64.fullmatch(archive_digest) is None or _HEX_64.fullmatch(wheel_digest) is None:
            raise ValueError("preregistration private input digest is malformed")

    return V5KagglePreexecutionReport(
        schema_version="apar-sentinel-v5-kaggle-preexecution/1",
        valid=True,
        phase=phase,
        source_commit=source_commit,
        source_tree=tree
        if phase is V5KagglePreexecutionPhase.SOURCE
        else str(_git(root, "rev-parse", f"{source_commit}^{{tree}}")),
        recovery_commit=recovery_commit,
        protocol_sha256=protocol.protocol_sha256,
        source_archive_sha256=archive_digest,
        wheelhouse_manifest_sha256=wheel_digest,
        execution_manifest_sha256=execution_manifest_digest,
        rehearsal_chain_roots=rehearsal_roots,
        deterministic_stage_sha256=deterministic,
        observed_max_peak_rss_bytes=None if observed is None else observed[0],
        observed_max_stage_seconds=None if observed is None else observed[1],
        observed_max_stage_output_bytes=None if observed is None else observed[2],
        seed_2404_boundary="asserted_only",
        successor_outputs_absent=True,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--phase", choices=tuple(V5KagglePreexecutionPhase), required=True)
    parser.add_argument("--expected-head")
    parser.add_argument("--expected-head-from-preregistration", action="store_true")
    parser.add_argument("--source-archive", type=Path)
    parser.add_argument("--wheelhouse", type=Path)
    parser.add_argument("--rehearsal-a", type=Path)
    parser.add_argument("--rehearsal-b", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    phase = V5KagglePreexecutionPhase(arguments.phase)
    if phase is V5KagglePreexecutionPhase.SOURCE:
        if (
            arguments.expected_head is None
            or arguments.expected_head_from_preregistration
            or arguments.source_archive is None
            or arguments.wheelhouse is None
            or arguments.rehearsal_a is not None
            or arguments.rehearsal_b is not None
        ):
            raise SystemExit("source phase arguments differ from the closed interface")
        expected_head = arguments.expected_head
        rehearsals: tuple[Path, ...] = ()
    else:
        if (
            arguments.expected_head is not None
            or not arguments.expected_head_from_preregistration
            or arguments.source_archive is not None
            or arguments.wheelhouse is not None
            or arguments.rehearsal_a is None
            or arguments.rehearsal_b is None
        ):
            raise SystemExit("frozen phase arguments differ from the closed interface")
        expected_head = "HEAD"
        rehearsals = (arguments.rehearsal_a, arguments.rehearsal_b)
    try:
        report = verify_v5_kaggle_preexecution(
            root=arguments.root,
            phase=phase,
            expected_head=expected_head,
            source_archive=arguments.source_archive,
            wheelhouse=arguments.wheelhouse,
            rehearsal_chain_roots=rehearsals,
        )
    except (OSError, ValueError, FileExistsError, PermissionError):
        print('{"error":"preexecution_failed","valid":false}')
        return 1
    print(_canonical_bytes(report.model_dump(mode="json")).decode())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "V5KagglePreexecutionPhase",
    "V5KagglePreexecutionReport",
    "V5KaggleExecutionManifest",
    "V5SourceArchiveManifest",
    "V5SourceFile",
    "V5WheelFile",
    "V5WheelhouseManifest",
    "build_v5_safe_evidence_input_manifest",
    "build_v5_source_archive",
    "build_v5_wheelhouse_manifest",
    "main",
    "verify_v5_kaggle_preexecution",
]
