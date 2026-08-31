"""Fresh-directory extraction and exact-dependency portable replay."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import zipfile
from pathlib import Path
from typing import Any, cast

from scripts.submission.archive import verify_archive
from scripts.submission.model import ReleaseError


def extract_verified_archive(archive_path: Path, destination: Path) -> Path:
    """Verify first, then materialize regular payloads under a new directory."""
    verify_archive(archive_path)
    if destination.exists():
        raise ReleaseError(f"clean-room extraction destination already exists: {destination}")
    destination.mkdir(parents=True)
    with zipfile.ZipFile(archive_path) as archive:
        names = archive.namelist()
        root_name = Path(names[0]).parts[0]
        for info in archive.infolist():
            target = destination.joinpath(*Path(info.filename).parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(archive.read(info.filename))
            target.chmod(0o644)
    return destination / root_name


def _run(command: list[str], *, cwd: Path | None = None, env: dict[str, str] | None = None) -> str:
    completed = subprocess.run(
        command,
        cwd=cwd,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout).strip()
        raise ReleaseError(f"clean-room command failed ({command[0]}): {detail}")
    return completed.stdout


def run_clean_room(archive_path: Path, *, python_executable: str) -> dict[str, Any]:
    """Install exact locks, replay the model, and build the extracted console."""
    uv = shutil.which("uv")
    with tempfile.TemporaryDirectory(prefix="apar-submission-clean-room-") as temporary:
        temporary_root = Path(temporary)
        release_root = extract_verified_archive(archive_path, temporary_root / "extracted")
        venv = temporary_root / "venv"
        if uv is not None:
            _run([uv, "venv", "--no-project", "--python", python_executable, str(venv)])
        else:
            _run([python_executable, "-m", "venv", str(venv)])
        venv_python = venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        requirements = release_root / "release" / "requirements-judge.txt"
        if uv is not None:
            _run(
                [
                    uv,
                    "pip",
                    "install",
                    "--python",
                    str(venv_python),
                    "--requirement",
                    str(requirements),
                ]
            )
        else:
            _run(
                [
                    str(venv_python),
                    "-m",
                    "pip",
                    "install",
                    "--disable-pip-version-check",
                    "--no-input",
                    "--requirement",
                    str(requirements),
                ]
            )
        environment = dict(os.environ)
        environment.pop("PYTHONPATH", None)
        environment.update(
            {
                "PYTHONDONTWRITEBYTECODE": "1",
                "PYTHONNOUSERSITE": "1",
                "UV_OFFLINE": "1",
            }
        )
        output = _run(
            [
                str(venv_python),
                "-m",
                "scripts.submission.runtime_verify",
                "--root",
                ".",
            ],
            cwd=release_root,
            env=environment,
        )
        try:
            document = json.loads(output)
        except json.JSONDecodeError as error:
            raise ReleaseError("clean-room verifier did not emit JSON") from error
        if not isinstance(document, dict) or document.get("replay_verified") is not True:
            raise ReleaseError("clean-room verifier did not confirm replay")
        manifest = document.get("accepted_model")
        if not isinstance(manifest, dict):
            raise ReleaseError("clean-room accepted model identity is absent")
        npm = shutil.which("npm")
        if npm is None:
            raise ReleaseError("clean-room web verification requires npm")
        web_root = release_root / "web"
        if not (web_root / "package-lock.json").is_file():
            raise ReleaseError("clean-room web dependency lock is absent")
        _run([npm, "ci", "--no-audit", "--no-fund"], cwd=web_root)
        _run([npm, "run", "build"], cwd=web_root)
        if not (web_root / "dist" / "index.html").is_file():
            raise ReleaseError("clean-room web build did not produce index.html")
        health_output = _run(
            [str(venv_python), "scripts/run_apar_console.py", "health"],
            cwd=release_root,
            env=environment,
        )
        try:
            health = json.loads(health_output)
        except json.JSONDecodeError as error:
            raise ReleaseError("clean-room console health did not emit JSON") from error
        if not isinstance(health, dict) or health.get("status") != "ready":
            raise ReleaseError("clean-room console health did not confirm readiness")
        document["web_build_verified"] = True
        document["web_health"] = health
        return cast(dict[str, Any], document)
