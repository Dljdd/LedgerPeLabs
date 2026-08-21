"""Pinned, bounded client for the evaluator-owned hidden-metrics worker."""

from __future__ import annotations

import contextlib
import ctypes
import hashlib
import os
import selectors
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import BinaryIO, Literal, cast

from pydantic import field_validator, model_validator

from apar.contracts._validation import ExternalContract
from apar.evaluation.gates import EvaluatorReplayVerifier, EvaluatorSigningIdentity
from apar.runs.wire import WireContractError, canonical_json_bytes, strict_json_loads

_SOURCE_ROOT = Path(__file__).resolve().parents[2]
_WORKER_PATH = (Path(__file__).resolve().parent / "worker.py").resolve()
_MAX_REQUEST_BYTES = 32_000_000
_MAX_OUTPUT_BYTES = 32_000_000
_MAX_SOURCE_FILES = 1_000
_MAX_SOURCE_BYTES = 4_000_000
_MAX_RSS_BYTES = 1_610_612_736


class _DarwinRusageInfoV2(ctypes.Structure):
    _fields_ = [
        ("uuid", ctypes.c_ubyte * 16),
        ("user_time", ctypes.c_uint64),
        ("system_time", ctypes.c_uint64),
        ("package_idle_wakeups", ctypes.c_uint64),
        ("interrupt_wakeups", ctypes.c_uint64),
        ("pageins", ctypes.c_uint64),
        ("wired_size", ctypes.c_uint64),
        ("resident_size", ctypes.c_uint64),
        ("physical_footprint", ctypes.c_uint64),
        ("process_start", ctypes.c_uint64),
        ("process_exit", ctypes.c_uint64),
        ("child_user_time", ctypes.c_uint64),
        ("child_system_time", ctypes.c_uint64),
        ("child_package_idle_wakeups", ctypes.c_uint64),
        ("child_interrupt_wakeups", ctypes.c_uint64),
        ("child_pageins", ctypes.c_uint64),
        ("child_elapsed", ctypes.c_uint64),
        ("disk_bytes_read", ctypes.c_uint64),
        ("disk_bytes_written", ctypes.c_uint64),
    ]


class HiddenWorkerError(RuntimeError):
    """The isolated hidden evaluator failed closed without exposing restricted data."""


class WorkerSourceEntry(ExternalContract):
    path: str
    sha256: str
    size_bytes: int

    @field_validator("path")
    @classmethod
    def path_is_canonical(cls, value: str) -> str:
        if (
            type(value) is not str
            or not value.startswith("apar/")
            or not value.endswith(".py")
            or ".." in value.split("/")
            or len(value) > 512
        ):
            raise ValueError("worker source path is invalid")
        return value

    @field_validator("sha256")
    @classmethod
    def digest_is_sha256(cls, value: str) -> str:
        _validate_digest(value)
        return value

    @field_validator("size_bytes", mode="before")
    @classmethod
    def size_is_exact(cls, value: object) -> object:
        if type(value) is not int or not 0 < value <= _MAX_SOURCE_BYTES:
            raise ValueError("worker source size is invalid")
        return value


class EvaluatorWorkerManifest(ExternalContract):
    """Evaluator-signed exact Python source inventory for one worker launch."""

    schema_version: Literal["1.0.0"] = "1.0.0"
    entrypoint: str
    entries: tuple[WorkerSourceEntry, ...]
    signer_key_id: str
    signature_base64: str
    manifest_digest: str

    @field_validator("entries", mode="before")
    @classmethod
    def entries_are_tuple(cls, value: object) -> object:
        if type(value) is not tuple:
            raise ValueError("worker inventory entries must be an exact tuple")
        return value

    @field_validator("signer_key_id", "manifest_digest")
    @classmethod
    def digests_are_sha256(cls, value: str) -> str:
        _validate_digest(value)
        return value

    @model_validator(mode="after")
    def inventory_is_closed(self) -> EvaluatorWorkerManifest:
        paths = tuple(item.path for item in self.entries)
        if (
            not paths
            or len(paths) > _MAX_SOURCE_FILES
            or paths != tuple(sorted(set(paths)))
            or self.entrypoint not in paths
        ):
            raise ValueError("worker source inventory is invalid")
        expected = _digest(
            {**self.unsigned_document(), "signature_base64": self.signature_base64}
        )
        if self.manifest_digest != expected:
            raise ValueError("worker manifest digest is inconsistent")
        return self

    @classmethod
    def create(cls, signer: EvaluatorSigningIdentity) -> EvaluatorWorkerManifest:
        if not EvaluatorSigningIdentity.is_exact(signer):
            raise HiddenWorkerError("worker manifest requires exact evaluator signer")
        entries = _source_inventory()
        fields: dict[str, object] = {
            "schema_version": "1.0.0",
            "entrypoint": _WORKER_PATH.relative_to(_SOURCE_ROOT).as_posix(),
            "entries": entries,
            "signer_key_id": signer.key_id,
        }
        unsigned = {
            **fields,
            "entries": [item.model_dump(mode="json") for item in entries],
        }
        signature = signer._sign(unsigned)
        return cls(
            schema_version="1.0.0",
            entrypoint=_WORKER_PATH.relative_to(_SOURCE_ROOT).as_posix(),
            entries=entries,
            signer_key_id=signer.key_id,
            signature_base64=signature,
            manifest_digest=_digest({**unsigned, "signature_base64": signature}),
        )

    def unsigned_document(self) -> dict[str, object]:
        return self.model_dump(
            mode="json", exclude={"signature_base64", "manifest_digest"}
        )


class EvaluatorWorkerClient:
    """Launch one isolated, one-shot evaluator after rechecking signed source bytes."""

    __slots__ = ("_manifest", "_verifier")

    def __init__(
        self,
        manifest: EvaluatorWorkerManifest,
        verifier: EvaluatorReplayVerifier,
    ) -> None:
        if type(manifest) is not EvaluatorWorkerManifest:
            raise HiddenWorkerError("worker manifest must have its exact type")
        if type(verifier) is not EvaluatorReplayVerifier:
            raise HiddenWorkerError("worker verifier must have its exact type")
        if (
            manifest.signer_key_id != verifier.key_id
            or not verifier.verify_document(
                manifest.unsigned_document(), manifest.signature_base64
            )
        ):
            raise HiddenWorkerError("worker manifest signature is invalid")
        self._manifest = manifest
        self._verifier = verifier
        self._verify_inventory()

    @property
    def manifest_digest(self) -> str:
        return self._manifest.manifest_digest

    def invoke(
        self, document: dict[str, object], *, timeout_ms: int = 30_000
    ) -> dict[str, object]:
        if type(document) is not dict:
            raise HiddenWorkerError("hidden worker request must be an exact object")
        if type(timeout_ms) is not int or not 100 <= timeout_ms <= 30_000:
            raise HiddenWorkerError("hidden worker timeout is invalid")
        self._verify_inventory()
        request = canonical_json_bytes(document)
        if not request or len(request) > _MAX_REQUEST_BYTES:
            raise HiddenWorkerError("hidden worker request exceeds its resource cap")
        environment = {
            "LANG": "C.UTF-8",
            "LC_ALL": "C.UTF-8",
            "PATH": "/usr/bin:/bin",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONHASHSEED": "0",
        }
        deadline = time.monotonic() + timeout_ms / 1000
        with tempfile.TemporaryDirectory(prefix="apar-hidden-evaluator-") as directory:
            temporary_root = Path(directory)
            snapshot_root = temporary_root / "source"
            work_root = temporary_root / "work"
            _create_verified_snapshot(snapshot_root, self._manifest)
            work_root.mkdir(mode=0o700)
            worker_path = snapshot_root / self._manifest.entrypoint
            try:
                process = subprocess.Popen(
                    [sys.executable, "-I", str(worker_path)],
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    cwd=work_root,
                    env=environment,
                    close_fds=True,
                    start_new_session=True,
                )
            except OSError as error:
                _make_snapshot_removable(snapshot_root)
                raise HiddenWorkerError("hidden evaluator could not start") from error
            try:
                stdout, stderr = self._collect_bounded(
                    process,
                    request=request,
                    deadline=deadline,
                )
            finally:
                _make_snapshot_removable(snapshot_root)
        if (
            process.returncode != 0
            or stderr
            or not stdout
            or len(stdout) > _MAX_OUTPUT_BYTES
        ):
            raise HiddenWorkerError("hidden evaluator failed closed")
        try:
            loaded = strict_json_loads(stdout)
        except WireContractError as error:
            raise HiddenWorkerError("hidden evaluator output is invalid") from error
        if type(loaded) is not dict or canonical_json_bytes(loaded) != stdout:
            raise HiddenWorkerError("hidden evaluator output is not canonical")
        return cast(dict[str, object], loaded)

    def _collect_bounded(
        self,
        process: subprocess.Popen[bytes],
        *,
        request: bytes,
        deadline: float,
    ) -> tuple[bytes, bytes]:
        """Bound deadline, RSS, and pipe bytes for the one-shot evaluator child."""
        del self
        assert process.stdin is not None
        assert process.stdout is not None
        assert process.stderr is not None
        output = bytearray()
        errors = bytearray()
        request_offset = 0
        selector = selectors.DefaultSelector()
        os.set_blocking(process.stdin.fileno(), False)
        selector.register(process.stdin, selectors.EVENT_WRITE, data="stdin")
        for stream, label in ((process.stdout, "stdout"), (process.stderr, "stderr")):
            os.set_blocking(stream.fileno(), False)
            selector.register(stream, selectors.EVENT_READ, data=label)
        next_memory_check = 0.0
        try:
            while selector.get_map() or process.poll() is None:
                now = time.monotonic()
                if now >= deadline:
                    _kill_group(process)
                    raise HiddenWorkerError(
                        "hidden evaluator deadline exceeded and process was killed"
                    )
                if now >= next_memory_check and process.poll() is None:
                    next_memory_check = now + 0.05
                    resident = _resident_bytes(process.pid)
                    if resident is None and process.poll() is None:
                        _kill_group(process)
                        raise HiddenWorkerError(
                            "hidden evaluator memory watchdog unavailable; process was killed"
                        )
                    if resident is not None and resident > _MAX_RSS_BYTES:
                        _kill_group(process)
                        raise HiddenWorkerError(
                            "hidden evaluator memory limit exceeded; process was killed"
                        )
                wait = max(0.0, min(0.05, deadline - now))
                for key, _ in selector.select(timeout=wait):
                    stream = cast(BinaryIO, key.fileobj)
                    if key.data == "stdin":
                        try:
                            written = os.write(
                                stream.fileno(),
                                request[request_offset : request_offset + 65_536],
                            )
                        except (BlockingIOError, BrokenPipeError):
                            written = 0
                        request_offset += written
                        if request_offset == len(request) or process.poll() is not None:
                            selector.unregister(key.fileobj)
                            stream.close()
                        continue
                    chunk = os.read(stream.fileno(), 65_536)
                    if not chunk:
                        selector.unregister(key.fileobj)
                        stream.close()
                        continue
                    target = output if key.data == "stdout" else errors
                    target.extend(chunk)
                    if len(output) + len(errors) > _MAX_OUTPUT_BYTES:
                        _kill_group(process)
                        raise HiddenWorkerError(
                            "hidden evaluator output limit exceeded; process was killed"
                        )
        finally:
            selector.close()
            for stream in (process.stdin, process.stdout, process.stderr):
                if not stream.closed:
                    stream.close()
        process.wait(timeout=1)
        return bytes(output), bytes(errors)

    def _verify_inventory(self) -> None:
        if (
            self._manifest.signer_key_id != self._verifier.key_id
            or not self._verifier.verify_document(
                self._manifest.unsigned_document(), self._manifest.signature_base64
            )
            or _source_inventory() != self._manifest.entries
        ):
            raise HiddenWorkerError("hidden evaluator source inventory changed")


def _source_inventory() -> tuple[WorkerSourceEntry, ...]:
    paths = tuple(
        sorted(
            path
            for path in (_SOURCE_ROOT / "apar").rglob("*.py")
            if path.is_file() and not path.is_symlink()
        )
    )
    if not paths or len(paths) > _MAX_SOURCE_FILES:
        raise HiddenWorkerError("hidden evaluator source inventory is invalid")
    entries: list[WorkerSourceEntry] = []
    for path in paths:
        raw = path.read_bytes()
        if not raw or len(raw) > _MAX_SOURCE_BYTES:
            raise HiddenWorkerError("hidden evaluator source file violates bounds")
        entries.append(
            WorkerSourceEntry(
                path=path.relative_to(_SOURCE_ROOT).as_posix(),
                sha256=hashlib.sha256(raw).hexdigest(),
                size_bytes=len(raw),
            )
        )
    return tuple(entries)


def _create_verified_snapshot(
    snapshot_root: Path, manifest: EvaluatorWorkerManifest
) -> None:
    snapshot_root.mkdir(mode=0o700)
    for entry in manifest.entries:
        source = _SOURCE_ROOT / entry.path
        raw = source.read_bytes()
        if (
            len(raw) != entry.size_bytes
            or hashlib.sha256(raw).hexdigest() != entry.sha256
        ):
            raise HiddenWorkerError("hidden evaluator source snapshot changed")
        destination = snapshot_root / entry.path
        destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        with destination.open("xb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        destination.chmod(0o400)
    snapshot_entries = tuple(
        WorkerSourceEntry(
            path=path.relative_to(snapshot_root).as_posix(),
            sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            size_bytes=path.stat().st_size,
        )
        for path in sorted(snapshot_root.rglob("*.py"))
    )
    if snapshot_entries != manifest.entries:
        raise HiddenWorkerError("hidden evaluator source snapshot is inconsistent")
    directories = sorted(
        (path for path in snapshot_root.rglob("*") if path.is_dir()),
        key=lambda path: len(path.parts),
        reverse=True,
    )
    for directory in directories:
        directory.chmod(0o500)
    snapshot_root.chmod(0o500)


def _make_snapshot_removable(snapshot_root: Path) -> None:
    if not snapshot_root.exists():
        return
    with contextlib.suppress(OSError):
        snapshot_root.chmod(0o700)
    for directory in snapshot_root.rglob("*"):
        if directory.is_dir():
            with contextlib.suppress(OSError):
                directory.chmod(0o700)


def _kill_group(process: subprocess.Popen[bytes]) -> None:
    with contextlib.suppress(ProcessLookupError):
        os.killpg(process.pid, signal.SIGKILL)
    with contextlib.suppress(subprocess.TimeoutExpired):
        process.wait(timeout=1)
    if process.poll() is None:
        with contextlib.suppress(ProcessLookupError):
            process.kill()
        process.wait(timeout=1)


def _resident_bytes(process_id: int) -> int | None:
    """Read live child RSS without granting process inspection to the child."""
    if sys.platform == "darwin":
        try:
            library = ctypes.CDLL("/usr/lib/libproc.dylib", use_errno=True)
            function = library.proc_pid_rusage
        except (AttributeError, OSError):
            return None
        function.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
        function.restype = ctypes.c_int
        info = _DarwinRusageInfoV2()
        if function(process_id, 2, ctypes.byref(info)) != 0:
            return None
        return int(info.resident_size)
    if sys.platform.startswith("linux"):
        try:
            fields = Path(f"/proc/{process_id}/status").read_text(
                encoding="ascii"
            ).splitlines()
        except OSError:
            return None
        for field in fields:
            if field.startswith("VmRSS:"):
                try:
                    return int(field.split()[1]) * 1024
                except (IndexError, ValueError):
                    return None
        return None
    return None


def _validate_digest(value: str) -> None:
    if type(value) is not str or len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError("worker evidence digest must be lowercase SHA-256")


def _digest(document: object) -> str:
    return hashlib.sha256(canonical_json_bytes(document)).hexdigest()


__all__ = [
    "EvaluatorWorkerClient",
    "EvaluatorWorkerManifest",
    "HiddenWorkerError",
    "WorkerSourceEntry",
]
