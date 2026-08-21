#!/usr/bin/env python3
"""Run the reproducible APAR Defend G3 acceptance gate."""

from __future__ import annotations

import hashlib
import os
import signal
import stat
import subprocess
import sys
import threading
from pathlib import Path
from typing import BinaryIO, Literal, TextIO

ROOT = Path(__file__).resolve().parents[1]
COMPETITION_PROFILE = ROOT / "config" / "defense" / "competition-profile.json"
COMPETITION_PROFILE_CANONICAL_SHA256 = (
    "f91c36e0329ef46631826a84d33b46282567069410cc9dc2c17694fe7463d7b1"
)
CHECK_TIMEOUT_SECONDS = 900.0
MAX_CAPTURE_BYTES = 8 * 1024 * 1024
CAPTURE_CHUNK_BYTES = 64 * 1024
KILL_GRACE_SECONDS = 5.0

EXECUTION_ERROR_EXIT = 70
OUTPUT_LIMIT_EXIT = 74
TIMEOUT_EXIT = 124

PASS_LINE = (
    "G3 PASS: causal features, rules/GBDT/hybrid, matched budgets, "
    "frozen hidden evaluation, and judge scorecards"
)

CHECKS = (
    ("G0", [sys.executable, "scripts/verify_g0.py"]),
    ("G1_G2", [sys.executable, "scripts/verify_g1_g2.py"]),
    ("FEATURES", [sys.executable, "-m", "pytest", "tests/features", "-q"]),
    ("DEFENSE", [sys.executable, "-m", "pytest", "tests/defense", "-q"]),
    ("CASES", [sys.executable, "-m", "pytest", "tests/cases", "-q"]),
    ("EVALUATION", [sys.executable, "-m", "pytest", "tests/evaluation", "-q"]),
    (
        "G3",
        [
            sys.executable,
            "-m",
            "pytest",
            "tests/integration/test_g3_defense.py",
            "-q",
        ],
    ),
)

_PASSTHROUGH_ENVIRONMENT = (
    "HOME",
    "LANG",
    "LC_ALL",
    "LC_CTYPE",
    "SSL_CERT_FILE",
    "SYSTEMROOT",
    "TEMP",
    "TERM",
    "TMP",
    "TMPDIR",
    "TZ",
    "WINDIR",
)


def _child_environment() -> dict[str, str]:
    """Return a minimal deterministic environment without Python/pytest hooks."""
    environment = {
        name: value
        for name in _PASSTHROUGH_ENVIRONMENT
        if (value := os.environ.get(name)) is not None
    }
    environment.update(
        {
            "PATH": os.defpath,
            "PYTHONHASHSEED": "0",
            "PYTHONNOUSERSITE": "1",
        }
    )
    return environment


def _terminate_process(process: subprocess.Popen[bytes]) -> None:
    """Terminate the isolated check and any descendants it created."""
    if process.poll() is not None:
        return
    try:
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGKILL)
        else:
            process.kill()
    except ProcessLookupError:
        return
    except OSError:
        try:
            process.kill()
        except OSError:
            return


class _BoundedCapture:
    """Thread-safe, combined stdout/stderr capture with a strict byte ceiling."""

    def __init__(self, process: subprocess.Popen[bytes]) -> None:
        self._process = process
        self._lock = threading.Lock()
        self._stdout = bytearray()
        self._stderr = bytearray()
        self._captured_bytes = 0
        self.output_exceeded = threading.Event()
        self.reader_failed = threading.Event()

    @property
    def stdout(self) -> bytes:
        return bytes(self._stdout)

    @property
    def stderr(self) -> bytes:
        return bytes(self._stderr)

    def add(self, stream: Literal["stdout", "stderr"], chunk: bytes) -> None:
        newly_exceeded = False
        with self._lock:
            remaining = max(0, MAX_CAPTURE_BYTES - self._captured_bytes)
            accepted = chunk[:remaining]
            if stream == "stdout":
                self._stdout.extend(accepted)
            else:
                self._stderr.extend(accepted)
            self._captured_bytes += len(accepted)
            if len(chunk) > len(accepted) and not self.output_exceeded.is_set():
                self.output_exceeded.set()
                newly_exceeded = True
        if newly_exceeded:
            _terminate_process(self._process)


def _drain_pipe(
    pipe: BinaryIO,
    stream: Literal["stdout", "stderr"],
    capture: _BoundedCapture,
) -> None:
    try:
        while chunk := os.read(pipe.fileno(), CAPTURE_CHUNK_BYTES):
            capture.add(stream, chunk)
    except (OSError, ValueError):
        capture.reader_failed.set()
        _terminate_process(capture._process)
    finally:
        pipe.close()


def _write_bytes(stream: TextIO, payload: bytes) -> None:
    if not payload:
        return
    binary_stream = getattr(stream, "buffer", None)
    if binary_stream is not None:
        binary_stream.write(payload)
        binary_stream.flush()
        return
    stream.write(payload.decode("utf-8", errors="replace"))
    stream.flush()


def _diagnostic(label: str, message: str) -> None:
    print(f"G3 ERROR [{label}]: {message}", file=sys.stderr, flush=True)


def _validate_competition_profile() -> bool:
    """Attest the one canonical semantic document before any child can run.

    The repository text file may contain one POSIX trailing newline.  All signed
    defense lineage uses the canonical JSON bytes without that transport newline,
    so the verifier deliberately pins that same semantic digest.
    """
    try:
        info = COMPETITION_PROFILE.lstat()
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
            return False
        payload = COMPETITION_PROFILE.read_bytes()
    except OSError:
        return False
    if payload.endswith(b"\n"):
        payload = payload[:-1]
    return (
        bool(payload)
        and not payload.endswith(b"\n")
        and hashlib.sha256(payload).hexdigest()
        == COMPETITION_PROFILE_CANONICAL_SHA256
    )


def _run_check(label: str, argv: list[str]) -> int:
    """Run one fixed check with bounded resources and forward its two streams."""
    if not argv or any(not isinstance(argument, str) for argument in argv):
        _diagnostic(label, "malformed check arguments")
        return EXECUTION_ERROR_EXIT

    try:
        process = subprocess.Popen(
            argv,
            cwd=ROOT,
            env=_child_environment(),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=os.name == "posix",
        )
    except (OSError, TypeError, ValueError) as error:
        _diagnostic(label, f"could not start check ({type(error).__name__})")
        return EXECUTION_ERROR_EXIT

    if process.stdout is None or process.stderr is None:
        _terminate_process(process)
        _diagnostic(label, "child process did not expose bounded output pipes")
        return EXECUTION_ERROR_EXIT

    capture = _BoundedCapture(process)
    readers = (
        threading.Thread(
            target=_drain_pipe,
            args=(process.stdout, "stdout", capture),
            daemon=True,
        ),
        threading.Thread(
            target=_drain_pipe,
            args=(process.stderr, "stderr", capture),
            daemon=True,
        ),
    )
    for reader in readers:
        reader.start()

    timed_out = False
    try:
        returncode = process.wait(timeout=CHECK_TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired:
        timed_out = True
        _terminate_process(process)
        try:
            returncode = process.wait(timeout=KILL_GRACE_SECONDS)
        except subprocess.TimeoutExpired:
            returncode = -signal.SIGKILL

    for reader in readers:
        reader.join(timeout=KILL_GRACE_SECONDS)
    if any(reader.is_alive() for reader in readers):
        capture.reader_failed.set()

    _write_bytes(sys.stdout, capture.stdout)
    _write_bytes(sys.stderr, capture.stderr)

    if capture.output_exceeded.is_set():
        _diagnostic(label, f"output exceeded {MAX_CAPTURE_BYTES} bytes")
        return OUTPUT_LIMIT_EXIT
    if timed_out:
        _diagnostic(label, f"timed out after {CHECK_TIMEOUT_SECONDS:g} seconds")
        return TIMEOUT_EXIT
    if capture.reader_failed.is_set():
        _diagnostic(label, "failed while reading child output")
        return EXECUTION_ERROR_EXIT
    if returncode < 0:
        signum = -returncode
        _diagnostic(label, f"terminated by signal {signum}")
        return 128 + signum
    if returncode != 0:
        _diagnostic(label, f"exited with status {returncode}")
    return returncode


def main() -> int:
    """Run all preregistered checks in order and stop at the first failure."""
    if not _validate_competition_profile():
        _diagnostic("PROFILE", "competition profile attestation failed")
        return EXECUTION_ERROR_EXIT
    for label, argv in CHECKS:
        try:
            returncode = _run_check(label, argv)
        except Exception as error:  # Fail closed on a malformed execution result.
            _diagnostic(label, f"verifier execution failed ({type(error).__name__})")
            return EXECUTION_ERROR_EXIT
        if type(returncode) is not int:
            _diagnostic(label, "verifier execution returned a malformed status")
            return EXECUTION_ERROR_EXIT
        if returncode != 0:
            return returncode
    print(PASS_LINE, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
