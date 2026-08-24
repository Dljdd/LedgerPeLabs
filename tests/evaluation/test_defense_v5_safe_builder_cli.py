"""Fresh-process contracts for the Sentinel v5 safe-evidence builder."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from apar.v5_independent_verifier import verify_evidence_bytes

ROOT = Path(__file__).resolve().parents[2]


def _builder_command(mode: str, *, root: Path, output: Path) -> list[str]:
    entrypoint = (
        [str(ROOT / "scripts/build_defense_v5_safe_evidence.py")]
        if mode == "direct"
        else ["-m", "scripts.build_defense_v5_safe_evidence"]
    )
    return [
        sys.executable,
        *entrypoint,
        "--root",
        str(root),
        "--output",
        str(output),
    ]


@pytest.fixture(scope="module")
def independently_built_cli_evidence(
    tmp_path_factory: pytest.TempPathFactory,
) -> dict[str, tuple[Path, dict[str, object]]]:
    directory = tmp_path_factory.mktemp("v5-safe-builder-cli")
    built: dict[str, tuple[Path, dict[str, object]]] = {}
    for mode in ("direct", "module"):
        output = directory / f"safe-{mode}.json"
        completed = subprocess.run(
            _builder_command(mode, root=ROOT, output=output),
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        assert completed.returncode == 0, completed.stderr
        built[mode] = (output, verify_evidence_bytes(output.read_bytes(), root=ROOT))
    return built


@pytest.mark.parametrize("mode", ["direct", "module"])
def test_safe_builder_cli_modes_create_independently_verified_evidence(
    independently_built_cli_evidence: dict[str, tuple[Path, dict[str, object]]],
    mode: str,
) -> None:
    """Removing invocation-aware imports must break one supported CLI mode."""
    output, report = independently_built_cli_evidence[mode]
    assert output.is_file()
    assert report["verified"] is True


def test_two_cli_builds_share_core_but_retain_distinct_real_latency(
    independently_built_cli_evidence: dict[str, tuple[Path, dict[str, object]]],
) -> None:
    """Reintroducing timing into the core must make these independent builds differ."""
    direct = independently_built_cli_evidence["direct"][1]
    module = independently_built_cli_evidence["module"][1]
    assert direct["deterministic_core_sha256"] == module[
        "deterministic_core_sha256"
    ]
    assert direct["observational_latency_sha256"] != module[
        "observational_latency_sha256"
    ]
    assert direct["envelope_sha256"] != module["envelope_sha256"]


@pytest.mark.parametrize("mode", ["direct", "module"])
def test_safe_builder_cli_refuses_to_overwrite(mode: str, tmp_path: Path) -> None:
    """Removing the pre-write existence check must overwrite this sentinel file."""
    output = tmp_path / "existing.json"
    output.write_bytes(b"preserve-me")
    completed = subprocess.run(
        _builder_command(mode, root=ROOT, output=output),
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode != 0
    assert "refusing to overwrite" in completed.stderr
    assert output.read_bytes() == b"preserve-me"


@pytest.mark.parametrize("mode", ["direct", "module"])
def test_safe_builder_cli_rejects_malformed_root(
    mode: str, tmp_path: Path
) -> None:
    """Removing early repository-root validation must expose an internal traceback."""
    malformed_root = tmp_path / "not-a-checkout"
    malformed_root.mkdir()
    output = tmp_path / "must-not-exist.json"
    completed = subprocess.run(
        _builder_command(mode, root=malformed_root, output=output),
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode != 0
    assert "repository root is missing the Sentinel v5 evidence configuration" in completed.stderr
    assert not output.exists()
