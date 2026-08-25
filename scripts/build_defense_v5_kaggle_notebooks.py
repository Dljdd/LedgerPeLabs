#!/usr/bin/env python3
"""Generate the nine private, offline Sentinel v5 Kaggle notebooks."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from apar.evaluation.v5_kaggle_protocol import (  # noqa: E402
    V5KaggleStage,
    load_v5_kaggle_protocol,
)

_PROTOCOL_PATH = Path("config/defense/defense-v5-kaggle-recovery.json")
_GENERATOR_SCHEMA = "apar-sentinel-v5-kaggle-notebook-generator/1"


@dataclass(frozen=True)
class V5GeneratedNotebook:
    """Paths and digest for one deterministically generated Kaggle notebook."""

    stage: V5KaggleStage
    kernel_id: str
    notebook_path: Path
    metadata_path: Path
    notebook_sha256: str


def _canonical_bytes(document: object) -> bytes:
    return (
        json.dumps(
            document,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode()


def _bootstrap_cell(
    *, stage: V5KaggleStage, source_slug: str, wheelhouse_slug: str, safe_slug: str
) -> str:
    return f'''from __future__ import annotations

import hashlib
import json
import os
import sys
import tarfile
from pathlib import Path, PurePosixPath


def _canonical(document: object) -> bytes:
    return json.dumps(
        document,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _manifest(path: Path, *, schema: str) -> dict[str, object]:
    document = json.loads(path.read_bytes())
    if not isinstance(document, dict) or document.get("schema_version") != schema:
        raise RuntimeError("private input manifest schema differs")
    claimed = document.pop("manifest_sha256", None)
    if claimed != hashlib.sha256(_canonical(document)).hexdigest():
        raise RuntimeError("private input manifest digest differs")
    return document


SOURCE_INPUT = Path("/kaggle/input/{source_slug}")
WHEELHOUSE_INPUT = Path("/kaggle/input/{wheelhouse_slug}")
SAFE_INPUT = Path("/kaggle/input/{safe_slug}")
SOURCE_MANIFEST = _manifest(
    SOURCE_INPUT / "source-manifest.json",
    schema="apar-sentinel-v5-source-archive/1",
)
WHEELHOUSE_MANIFEST = _manifest(
    WHEELHOUSE_INPUT / "wheelhouse-manifest.json",
    schema="apar-sentinel-v5-wheelhouse/1",
)
SAFE_MANIFEST = _manifest(
    SAFE_INPUT / "safe-evidence-manifest.json",
    schema="apar-sentinel-v5-kaggle-execution-input/1",
)

SOURCE_ARCHIVE = SOURCE_INPUT / "apar-v5-source3.tar.gz"
SAFE_EVIDENCE = SAFE_INPUT / "safe-evidence.json"
if (
    SOURCE_MANIFEST.get("artifact_name") != SOURCE_ARCHIVE.name
    or SOURCE_MANIFEST.get("artifact_sha256") != _sha256(SOURCE_ARCHIVE)
):
    raise RuntimeError("source archive binding differs")
if (
    SAFE_MANIFEST.get("artifact_name") != SAFE_EVIDENCE.name
    or SAFE_MANIFEST.get("artifact_sha256") != _sha256(SAFE_EVIDENCE)
):
    raise RuntimeError("safe evidence binding differs")

wheel_entries = WHEELHOUSE_MANIFEST.get("wheels")
if not isinstance(wheel_entries, list) or not wheel_entries:
    raise RuntimeError("wheelhouse manifest is empty")
for entry in wheel_entries:
    if not isinstance(entry, dict):
        raise RuntimeError("wheelhouse entry is malformed")
    wheel = WHEELHOUSE_INPUT / str(entry.get("filename"))
    if (
        entry.get("size_bytes") != wheel.stat().st_size
        or entry.get("sha256") != _sha256(wheel)
    ):
        raise RuntimeError("wheelhouse file binding differs")

EXTRACT_ROOT = Path("/kaggle/working/apar-v5-source-extract")
if EXTRACT_ROOT.exists():
    raise RuntimeError("source extraction root already exists")
EXTRACT_ROOT.mkdir(parents=True, mode=0o700)
with tarfile.open(SOURCE_ARCHIVE, "r:gz") as archive:
    members = archive.getmembers()
    if not members:
        raise RuntimeError("source archive is empty")
    for member in members:
        relative = PurePosixPath(member.name)
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or relative.parts[0] != "apar-v5-source"
        ):
            raise RuntimeError("source archive path is unsafe")
        if not (member.isfile() or member.isdir()):
            raise RuntimeError("source archive contains a non-file entry")
    archive.extractall(EXTRACT_ROOT, members=members, filter="data")
SOURCE_ROOT = EXTRACT_ROOT / "apar-v5-source"
if not (SOURCE_ROOT / "scripts/run_defense_v5_kaggle_stage.py").is_file():
    raise RuntimeError("closed stage entrypoint is absent")

NOTEBOOK_SOURCE = SOURCE_ROOT / "kaggle/defense_v5/{stage.value}.ipynb"
if not NOTEBOOK_SOURCE.is_file():
    raise RuntimeError("approved notebook source is absent")
OS_RELEASE = Path("/etc/os-release")
if not OS_RELEASE.is_file():
    raise RuntimeError("Kaggle OS release binding is absent")
runtime_image_facts = {{
    "schema_version": "apar-sentinel-v5-kaggle-runtime-image/1",
    "os_release_sha256": _sha256(OS_RELEASE),
    "python_executable_sha256": _sha256(Path(sys.executable)),
    "python_version": ".".join(str(item) for item in sys.version_info[:3]),
}}
os.environ.update(
    {{
        "APAR_V5_KAGGLE_IMAGE": "kaggle-cpu-runtime-fingerprint/1",
        "APAR_V5_KAGGLE_IMAGE_SHA256": hashlib.sha256(
            _canonical(runtime_image_facts)
        ).hexdigest(),
        "APAR_V5_DEPENDENCY_MANIFEST_SHA256": hashlib.sha256(
            (WHEELHOUSE_INPUT / "wheelhouse-manifest.json").read_bytes()
        ).hexdigest(),
        "APAR_V5_SOURCE_ARCHIVE_SHA256": _sha256(SOURCE_ARCHIVE),
        "APAR_V5_SOURCE_MANIFEST_PATH": str(
            SOURCE_INPUT / "source-manifest.json"
        ),
        "APAR_V5_NOTEBOOK_SHA256": _sha256(NOTEBOOK_SOURCE),
    }}
)
'''


def _install_cell() -> str:
    return '''import subprocess
import sys

install = subprocess.run(
    [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--disable-pip-version-check",
        "--no-input",
        "--no-index",
        "--find-links",
        str(WHEELHOUSE_INPUT),
        "--force-reinstall",
        "--no-build-isolation",
        "apar==0.1.0",
    ],
    check=False,
    capture_output=True,
    text=True,
)
if install.returncode != 0:
    raise RuntimeError("offline dependency installation failed")
'''


def _invoke_cell(
    *, stage: V5KaggleStage, predecessor_kernel_slug: str | None
) -> str:
    predecessor_setup = (
        '''CHAIN_ROOT = Path("/kaggle/working/apar-v5-chain")
CHAIN_ROOT.mkdir(mode=0o700)
'''
        if predecessor_kernel_slug is None
        else f'''PREDECESSOR_CHAIN = Path("/kaggle/input/{predecessor_kernel_slug}/apar-v5-chain")
CHAIN_ROOT = Path("/kaggle/working/apar-v5-chain")
if not PREDECESSOR_CHAIN.is_dir():
    raise RuntimeError("exact predecessor checkpoint chain is absent")
shutil.copytree(PREDECESSOR_CHAIN, CHAIN_ROOT, copy_function=shutil.copy2)
'''
    )
    return f'''import json
import shutil
import subprocess
import sys
from pathlib import Path

{predecessor_setup}
OUTPUT_ROOT = CHAIN_ROOT / "{stage.value}"
execution_mode = SAFE_MANIFEST.get("execution_mode")
if execution_mode not in (
    "kaggle_capacity_validation",
    "kaggle_locked_successor",
):
    raise RuntimeError("closed execution mode is absent")
command = [
    sys.executable,
    str(SOURCE_ROOT / "scripts/run_defense_v5_kaggle_stage.py"),
    "--root",
    str(SOURCE_ROOT),
    "--input-root",
    str(CHAIN_ROOT),
    "--output-root",
    str(OUTPUT_ROOT),
    "--safe-evidence",
    str(SAFE_EVIDENCE),
    "--execution-manifest",
    str(SAFE_INPUT / "safe-evidence-manifest.json"),
    "--approved-commit",
    str(SOURCE_MANIFEST.get("approved_commit")),
]
completed = subprocess.run(
    command,
    cwd=SOURCE_ROOT,
    check=False,
    capture_output=True,
    text=True,
)
if completed.returncode != 0:
    raise RuntimeError("closed checkpoint stage failed")
receipt = json.loads(completed.stdout)
expected_receipt_keys = {{
    "deterministic_sha256",
    "manifest_sha256",
    "observation_sha256",
    "stage",
}}
if set(receipt) != expected_receipt_keys or receipt.get("stage") != "{stage.value}":
    raise RuntimeError("redacted stage receipt differs")
print(json.dumps(receipt, sort_keys=True, separators=(",", ":")))
'''


def _notebook_document(
    *,
    stage: V5KaggleStage,
    source_slug: str,
    wheelhouse_slug: str,
    safe_slug: str,
    predecessor_kernel_slug: str | None,
) -> dict[str, Any]:
    sources = (
        _bootstrap_cell(
            stage=stage,
            source_slug=source_slug,
            wheelhouse_slug=wheelhouse_slug,
            safe_slug=safe_slug,
        ),
        _install_cell(),
        _invoke_cell(stage=stage, predecessor_kernel_slug=predecessor_kernel_slug),
    )
    cells: list[dict[str, Any]] = []
    for index, source in enumerate(sources):
        cells.append(
            {
                "cell_type": "code",
                "execution_count": None,
                "id": f"apar-{stage.value.replace('_', '-')}-{index}",
                "metadata": {"apar_generated_role": ("bootstrap", "install", "invoke")[index]},
                "outputs": [],
                "source": source,
            }
        )
    return {
        "cells": cells,
        "metadata": {
            "apar": {"generator_schema": _GENERATOR_SCHEMA, "stage": stage.value},
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.12"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def build_v5_kaggle_notebooks(
    *,
    root: Path,
    output_dir: Path,
    owner_slug: str,
    source_dataset_slug: str,
    wheelhouse_dataset_slug: str,
    safe_evidence_dataset_slug: str,
) -> tuple[V5GeneratedNotebook, ...]:
    """Generate the exact frozen private notebooks without external side effects."""
    root = root.resolve()
    if not root.is_dir():
        raise ValueError("repository root is not a directory")
    protocol = load_v5_kaggle_protocol(root / _PROTOCOL_PATH, root=root)
    private = protocol.private_inputs
    supplied = (
        owner_slug,
        source_dataset_slug,
        wheelhouse_dataset_slug,
        safe_evidence_dataset_slug,
    )
    frozen = (
        private.owner_slug,
        private.source_dataset_slug,
        private.wheelhouse_dataset_slug,
        private.safe_evidence_dataset_slug,
    )
    slug_pattern = __import__("re").compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
    if supplied != frozen or any(slug_pattern.fullmatch(value) is None for value in supplied):
        raise ValueError("Kaggle owner and dataset identifiers must equal frozen literals")
    if os.path.lexists(output_dir):
        raise FileExistsError("notebook output directory already exists")
    output_dir.mkdir(parents=True, mode=0o700)

    generated: list[V5GeneratedNotebook] = []
    for index, stage in enumerate(V5KaggleStage):
        kernel_slug = f"{private.notebook_slug_prefix}-{stage.value.replace('_', '-')}"
        kernel_id = f"{owner_slug}/{kernel_slug}"
        predecessor = None if index == 0 else generated[index - 1]
        notebook_path = output_dir / f"{stage.value}.ipynb"
        metadata_path = output_dir / f"{stage.value}-metadata.json"
        notebook_bytes = _canonical_bytes(
            _notebook_document(
                stage=stage,
                source_slug=source_dataset_slug,
                wheelhouse_slug=wheelhouse_dataset_slug,
                safe_slug=safe_evidence_dataset_slug,
                predecessor_kernel_slug=(
                    None if predecessor is None else predecessor.kernel_id.split("/", 1)[1]
                ),
            )
        )
        notebook_sha256 = hashlib.sha256(notebook_bytes).hexdigest()
        notebook_path.write_bytes(notebook_bytes)
        metadata = {
            "accelerator": "none",
            "apar_generator_schema": _GENERATOR_SCHEMA,
            "apar_notebook_sha256": notebook_sha256,
            "code_file": notebook_path.name,
            "competition_sources": [],
            "dataset_sources": [
                f"{owner_slug}/{source_dataset_slug}",
                f"{owner_slug}/{wheelhouse_dataset_slug}",
                f"{owner_slug}/{safe_evidence_dataset_slug}",
            ],
            "enable_gpu": False,
            "enable_internet": False,
            "enable_tpu": False,
            "id": kernel_id,
            "is_private": True,
            "kernel_sources": [] if predecessor is None else [predecessor.kernel_id],
            "kernel_type": "notebook",
            "keywords": ["apar-sentinel-v5-private"],
            "language": "python",
            "machine_shape": "",
            "model_sources": [],
            "title": f"APAR Sentinel v5 {stage.value}",
        }
        metadata_path.write_bytes(_canonical_bytes(metadata))
        generated.append(
            V5GeneratedNotebook(
                stage=stage,
                kernel_id=kernel_id,
                notebook_path=notebook_path,
                metadata_path=metadata_path,
                notebook_sha256=notebook_sha256,
            )
        )
    return tuple(generated)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    root = arguments.root.resolve()
    protocol = load_v5_kaggle_protocol(root / _PROTOCOL_PATH, root=root)
    private = protocol.private_inputs
    build_v5_kaggle_notebooks(
        root=root,
        output_dir=arguments.output,
        owner_slug=private.owner_slug,
        source_dataset_slug=private.source_dataset_slug,
        wheelhouse_dataset_slug=private.wheelhouse_dataset_slug,
        safe_evidence_dataset_slug=private.safe_evidence_dataset_slug,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["V5GeneratedNotebook", "build_v5_kaggle_notebooks", "main"]
