"""Isolated policy orchestration and authenticated immutable run manifests."""

from __future__ import annotations

import base64
import binascii
import contextlib
import ctypes
import hashlib
import os
import resource
import selectors
import signal
import stat
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, BinaryIO, cast
from uuid import NAMESPACE_URL, uuid5

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat
from pydantic import Field, field_validator, model_validator

from apar.contracts._validation import ExternalContract
from apar.contracts.events import PaymentEvent, Rail
from apar.contracts.scenarios import ScenarioBundle
from apar.redteam.policies import (
    PUBLIC_CAMPAIGN_FAMILIES,
    AttackCandidate,
    ParameterBounds,
    VisibleTrial,
)
from apar.runs.wire import (
    bounds_to_wire,
    candidate_from_wire,
    canonical_json_bytes,
    history_to_wire,
    strict_json_loads,
)
from apar.storage.artifacts import ArtifactRef, ArtifactStore

if TYPE_CHECKING:
    from apar.generators import CampaignParams
    from apar.redteam.benchmark import BenchmarkObservation

_HEX = frozenset("0123456789abcdef")
_WORKER_PATH = Path(__file__).with_name("policy_worker.py").resolve()
_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
_TASK6_RESULT = "docs/experiments/task6-v3.4-holdout-result.json"
_TASK6_RESULT_COMMIT = "d6d3eecbfe2d871af8375e1455814cb5c48f2928"
_TASK6_RESULT_SHA256 = "f82981a987651a7f7ebb10a9011df063b2dc54a56181cae5b838e31de5e658db"
_TASK6_CURRENT_MODE = 0o600
_TASK6_HISTORICAL_MODE = "100644"
_RUN_INDEX_MAX_BYTES = 4_096
_PINNED_CURRENT_FILES = {
    "docs/experiments/task6-v3.4-holdout-preregistration.json": (
        "12bf24e081e97f3222bf1fc92fb1d441c36bba548184c6b503519590efc649a4",
        0o644,
    ),
    "src/apar/generators/campaigns.py": (
        "670b4a3ec358f82d88f9655bd41d878fbee11d4841ff264655554bae31c3b31a",
        0o644,
    ),
    "src/apar/redteam/benchmark.py": (
        "7996fcf20c85547a861afcfeb9da132dad534ad30f7fe1a07618e90e2faef519",
        0o644,
    ),
    "src/apar/redteam/policies.py": (
        "c97ab7b263a493978cf901140a97f15874a34f8ff2ce54c84253e7baa998fb82",
        0o644,
    ),
}


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


def _resident_bytes(process_id: int) -> int | None:
    """Read live child RSS without granting the worker a process-inspection capability."""
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
            fields = Path(f"/proc/{process_id}/status").read_text(encoding="ascii").splitlines()
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


def _digest_document(document: object) -> str:
    return hashlib.sha256(canonical_json_bytes(document)).hexdigest()


def _artifact_digest_document(artifacts: dict[str, ArtifactRef]) -> dict[str, str]:
    return {name: reference.sha256 for name, reference in sorted(artifacts.items())}


def _strict_regular_file(path: Path, *, expected_mode: int, expected_sha256: str) -> bytes:
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    except FileNotFoundError as error:
        raise RunExecutionError(f"pinned provenance file is absent: {path}") from error
    except OSError as error:
        raise RunExecutionError(
            f"pinned provenance path is not a regular file: {path}"
        ) from error
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise RunExecutionError(f"pinned provenance path is not a regular file: {path}")
        if os.name == "posix" and stat.S_IMODE(metadata.st_mode) != expected_mode:
            raise RunExecutionError(f"pinned provenance filesystem mode changed: {path}")
        chunks: list[bytes] = []
        while chunk := os.read(descriptor, 1024 * 1024):
            chunks.append(chunk)
        raw = b"".join(chunks)
    finally:
        os.close(descriptor)
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise RunExecutionError(f"pinned provenance bytes changed: {path}")
    return raw


def _read_run_index_file(path: Path) -> bytes:
    """Read one index entry through the descriptor whose invariants were checked."""
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    except FileNotFoundError:
        raise
    except OSError as error:
        raise RunExecutionError("run index entry is not a regular mode-0600 file") from error
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or stat.S_IMODE(before.st_mode) != 0o600
            or before.st_size > _RUN_INDEX_MAX_BYTES
        ):
            raise RunExecutionError("run index entry is not a regular mode-0600 file")
        raw = os.read(descriptor, _RUN_INDEX_MAX_BYTES + 1)
        after = os.fstat(descriptor)
        if (
            len(raw) > _RUN_INDEX_MAX_BYTES
            or len(raw) != after.st_size
            or before.st_dev != after.st_dev
            or before.st_ino != after.st_ino
            or before.st_size != after.st_size
            or before.st_mtime_ns != after.st_mtime_ns
            or before.st_ctime_ns != after.st_ctime_ns
            or stat.S_IMODE(after.st_mode) != 0o600
        ):
            raise RunExecutionError("run index entry changed while it was read")
        return raw
    finally:
        os.close(descriptor)


def _git(*arguments: str, binary: bool = False) -> bytes | str:
    result = subprocess.run(
        ["git", *arguments],
        cwd=_REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=not binary,
        env={"LANG": "C", "LC_ALL": "C", "PATH": "/usr/bin:/bin"},
    )
    if result.returncode != 0:
        raise RunExecutionError("Git provenance verification failed")
    output: object = result.stdout
    if binary:
        if type(output) is not bytes:
            raise RunExecutionError("Git binary provenance output has the wrong type")
        return output
    if type(output) is not str:
        raise RunExecutionError("Git text provenance output has the wrong type")
    return output


def _provenance_document() -> dict[str, object]:
    """Verify current regular-file modes and separately pin historical Git mode."""
    result_path = _REPOSITORY_ROOT / _TASK6_RESULT
    result_raw = _strict_regular_file(
        result_path,
        expected_mode=_TASK6_CURRENT_MODE,
        expected_sha256=_TASK6_RESULT_SHA256,
    )
    current_files: dict[str, object] = {}
    for relative, (digest, mode) in sorted(_PINNED_CURRENT_FILES.items()):
        raw = _strict_regular_file(
            _REPOSITORY_ROOT / relative,
            expected_mode=mode,
            expected_sha256=digest,
        )
        current_files[relative] = {
            "filesystem_mode": format(mode, "04o"),
            "regular_non_symlink": True,
            "sha256": digest,
            "size_bytes": len(raw),
        }
    tree_line = cast(
        str,
        _git("ls-tree", _TASK6_RESULT_COMMIT, "--", _TASK6_RESULT),
    ).rstrip("\n")
    parts = tree_line.split(maxsplit=3)
    if (
        len(parts) != 4
        or parts[0] != _TASK6_HISTORICAL_MODE
        or parts[1] != "blob"
        or parts[3] != _TASK6_RESULT
    ):
        raise RunExecutionError("historical Task 6 Git mode or object type changed")
    historical_raw = cast(
        bytes,
        _git("show", f"{_TASK6_RESULT_COMMIT}:{_TASK6_RESULT}", binary=True),
    )
    if hashlib.sha256(historical_raw).hexdigest() != _TASK6_RESULT_SHA256:
        raise RunExecutionError("historical Task 6 result bytes changed")
    head = cast(str, _git("rev-parse", "HEAD")).strip()
    return {
        "current_source_head": head,
        "current_task6_files": current_files,
        "policy_worker": {
            "clean_process_per_proposal": True,
            "import_restrictions": True,
            "isolated_interpreter": True,
            "memory_watchdog_bytes": 805_306_368,
            "network_audit_denial": True,
            "output_cap_bytes": 1_048_576,
            "resource_limits": ["core", "cpu", "file_size", "file_descriptors", "processes"],
        },
        "python": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "task6_result": {
            "current_filesystem_mode": format(_TASK6_CURRENT_MODE, "04o"),
            "current_regular_non_symlink": True,
            "historical_commit": _TASK6_RESULT_COMMIT,
            "historical_git_mode": _TASK6_HISTORICAL_MODE,
            "historical_object_type": "blob",
            "path": _TASK6_RESULT,
            "sha256": _TASK6_RESULT_SHA256,
            "size_bytes": len(result_raw),
        },
    }


def _campaign_template_document(params: object) -> dict[str, object]:
    fields = getattr(type(params), "__dataclass_fields__", None)
    if type(fields) is not dict:
        raise TypeError("hidden campaign template is not a declared dataclass")
    document: dict[str, object] = {}
    for name in fields:
        value = getattr(params, name)
        if type(value) is Decimal:
            document[name] = str(value)
        elif type(value) is tuple:
            document[name] = list(value)
        elif type(value) in {bool, int, str} or value is None:
            document[name] = value
        else:
            raise TypeError(f"hidden campaign template field is unsupported: {name}")
    return document


def _event_bytes(events: tuple[PaymentEvent, ...]) -> bytes:
    return canonical_json_bytes(
        [event.model_dump(mode="json", round_trip=True, warnings=False) for event in events]
    )


class RunExecutionError(RuntimeError):
    """A run failed closed before an authenticated completion manifest existed."""


class PolicyWorkerError(RunExecutionError):
    """The disposable policy worker failed its strict process contract."""


class AttackerPolicyKind(StrEnum):
    """Closed policy implementations available in the clean worker."""

    FIXED = "fixed"
    RANDOM = "random"
    ADAPTIVE = "adaptive"
    CACHED_LLM = "cached_llm"


class AttackerPolicy(ExternalContract):
    """Typed selection containing budgets, never code, callables, or paths."""

    schema_version: str = "1.0.0"
    family: str
    kind: AttackerPolicyKind
    query_budget: int = Field(ge=1, le=64, strict=True)
    worker_timeout_ms: int = Field(ge=50, le=30_000, strict=True)
    expose_realized_value: bool = False

    @field_validator("family", mode="before")
    @classmethod
    def family_is_public(cls, value: object) -> object:
        if type(value) is not str or value not in PUBLIC_CAMPAIGN_FAMILIES:
            raise ValueError("policy family is unsupported")
        return value

    @field_validator("expose_realized_value", mode="before")
    @classmethod
    def disclosure_is_exact(cls, value: object) -> object:
        if type(value) is not bool:
            raise TypeError("expose_realized_value must be an exact boolean")
        return value


class PolicyWorkerBoundaryReport(ExternalContract):
    """Boolean self-test results from a fresh disposable worker."""

    clean_start: bool
    filesystem_blocked: bool
    forbidden_import_blocked: bool
    reflection_import_blocked: bool
    network_blocked: bool
    hidden_modules_absent: bool
    input_hidden_fields_absent: bool
    orchestrator_modules_absent: bool


def _set_limit(kind: int, maximum: int) -> None:
    soft, hard = resource.getrlimit(kind)
    bounded = maximum if hard == resource.RLIM_INFINITY else min(maximum, hard)
    resource.setrlimit(kind, (bounded, bounded))


def _limit_policy_child() -> None:
    """Apply hard POSIX resource limits before the isolated interpreter starts."""
    _set_limit(resource.RLIMIT_CORE, 0)
    _set_limit(resource.RLIMIT_CPU, 3)
    _set_limit(resource.RLIMIT_FSIZE, 1_048_576)
    _set_limit(resource.RLIMIT_NOFILE, 64)
    if hasattr(resource, "RLIMIT_NPROC"):
        _set_limit(resource.RLIMIT_NPROC, 1)


class PolicyWorkerClient:
    """Launch exactly one clean, killable process for each public proposal."""

    def __init__(self, worker_path: Path = _WORKER_PATH) -> None:
        resolved = worker_path.resolve()
        if resolved != _WORKER_PATH or resolved.is_symlink() or not resolved.is_file():
            raise ValueError("policy worker path is not the reviewed regular entry point")
        self._worker_path = resolved

    def propose(
        self,
        *,
        kind: AttackerPolicyKind,
        bounds: ParameterBounds,
        history: tuple[VisibleTrial, ...],
        seed: int,
        timeout_ms: int,
    ) -> AttackCandidate:
        if type(kind) is not AttackerPolicyKind:
            raise TypeError("policy kind must be exact")
        if type(seed) is not int or not 0 <= seed < 2**63:
            raise TypeError("worker seed must be an exact bounded integer")
        document = {
            "bounds": bounds_to_wire(bounds),
            "history": history_to_wire(history),
            "operation": "propose",
            "policy_kind": kind.value,
            "schema_version": "1.0.0",
            "seed": seed,
        }
        response = self._invoke(document, timeout_ms=timeout_ms)
        if type(response) is not dict or set(response) != {"candidate", "ok"}:
            raise PolicyWorkerError("policy worker response field set is invalid")
        if response["ok"] is not True:
            raise PolicyWorkerError("policy worker failed closed")
        return candidate_from_wire(response["candidate"], history=history, bounds=bounds)

    def probe(self, *, timeout_ms: int) -> PolicyWorkerBoundaryReport:
        response = self._invoke(
            {"operation": "probe", "schema_version": "1.0.0"},
            timeout_ms=timeout_ms,
        )
        if type(response) is not dict or set(response) != {"ok", "probe"}:
            raise PolicyWorkerError("policy worker probe response is invalid")
        if response["ok"] is not True:
            raise PolicyWorkerError("policy worker probe failed closed")
        return PolicyWorkerBoundaryReport.model_validate(response["probe"])

    def probe_hang(self, *, timeout_ms: int) -> None:
        self._invoke(
            {"operation": "probe_hang", "schema_version": "1.0.0"},
            timeout_ms=timeout_ms,
        )

    def _invoke(self, document: dict[str, object], *, timeout_ms: int) -> dict[str, object]:
        if type(timeout_ms) is not int or not 50 <= timeout_ms <= 30_000:
            raise ValueError("worker timeout must be an exact integer in [50, 30000]")
        request = canonical_json_bytes(document)
        environment = {
            "LANG": "C.UTF-8",
            "LC_ALL": "C.UTF-8",
            "PATH": "/usr/bin:/bin",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONHASHSEED": "0",
        }
        with tempfile.TemporaryDirectory(prefix="apar-policy-worker-") as worker_directory:
            process = subprocess.Popen(
                [sys.executable, "-I", str(self._worker_path)],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=worker_directory,
                env=environment,
                close_fds=True,
                start_new_session=True,
                preexec_fn=_limit_policy_child,
            )
            stdout, stderr = self._collect_bounded(
                process,
                request=request,
                timeout_ms=timeout_ms,
            )
        if process.returncode != 0:
            raise PolicyWorkerError("policy worker failed closed")
        if stderr or len(stdout) > 1_048_576:
            raise PolicyWorkerError("policy worker emitted invalid output")
        try:
            loaded = strict_json_loads(stdout)
        except ValueError as error:
            raise PolicyWorkerError("policy worker output is not canonical strict JSON") from error
        if type(loaded) is not dict:
            raise PolicyWorkerError("policy worker output must be an object")
        return cast(dict[str, object], loaded)

    @staticmethod
    def _kill_group(process: subprocess.Popen[bytes]) -> None:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGKILL)
        process.wait()

    def _collect_bounded(
        self,
        process: subprocess.Popen[bytes],
        *,
        request: bytes,
        timeout_ms: int,
    ) -> tuple[bytes, bytes]:
        """Bound deadline, resident memory, and pipe bytes while the child executes."""
        assert process.stdin is not None
        assert process.stdout is not None
        assert process.stderr is not None
        process.stdin.write(request)
        process.stdin.close()
        output = bytearray()
        errors = bytearray()
        selector = selectors.DefaultSelector()
        for stream, label in ((process.stdout, "stdout"), (process.stderr, "stderr")):
            os.set_blocking(stream.fileno(), False)
            selector.register(stream, selectors.EVENT_READ, data=label)
        deadline = time.monotonic() + timeout_ms / 1000
        next_memory_check = 0.0
        while selector.get_map() or process.poll() is None:
            now = time.monotonic()
            if now >= deadline:
                self._kill_group(process)
                selector.close()
                raise PolicyWorkerError("policy worker deadline exceeded and process killed")
            if now >= next_memory_check and process.poll() is None:
                next_memory_check = now + 0.05
                resident = _resident_bytes(process.pid)
                if resident is None and process.poll() is None:
                    self._kill_group(process)
                    selector.close()
                    raise PolicyWorkerError(
                        "policy worker memory watchdog unavailable; process killed"
                    )
                if resident is not None and resident > 805_306_368:
                    self._kill_group(process)
                    selector.close()
                    raise PolicyWorkerError(
                        "policy worker resident-memory limit exceeded and process killed"
                    )
            for key, _ in selector.select(timeout=min(0.05, deadline - now)):
                selected_stream = cast(BinaryIO, key.fileobj)
                chunk = os.read(selected_stream.fileno(), 65_536)
                if not chunk:
                    selector.unregister(key.fileobj)
                    continue
                target = output if key.data == "stdout" else errors
                target.extend(chunk)
                if len(output) + len(errors) > 1_048_576:
                    self._kill_group(process)
                    selector.close()
                    raise PolicyWorkerError(
                        "policy worker output limit exceeded and process killed"
                    )
        selector.close()
        process.wait()
        return bytes(output), bytes(errors)


def _private_key_bytes(key: Ed25519PrivateKey) -> bytes:
    from cryptography.hazmat.primitives.serialization import NoEncryption, PrivateFormat

    return key.private_bytes(Encoding.Raw, PrivateFormat.Raw, NoEncryption())


def _write_all(descriptor: int, content: bytes) -> None:
    offset = 0
    while offset < len(content):
        written = os.write(descriptor, content[offset:])
        if written <= 0:
            raise OSError("durable low-level write made no progress")
        offset += written


class RunSigningIdentity:
    """Durable or injected Ed25519 authority; private bytes never leave this object."""

    __slots__ = ("_key", "key_id", "public_key_base64")

    def __init__(self, key: Ed25519PrivateKey) -> None:
        if not isinstance(key, Ed25519PrivateKey):
            raise TypeError("run signer must be an Ed25519 private key")
        self._key = key
        public = key.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw)
        self.key_id = hashlib.sha256(public).hexdigest()
        self.public_key_base64 = base64.b64encode(public).decode("ascii")

    @classmethod
    def from_private_bytes(cls, private_bytes: bytes) -> RunSigningIdentity:
        if type(private_bytes) is not bytes or len(private_bytes) != 32:
            raise ValueError("Ed25519 private seed must be exactly 32 bytes")
        return cls(Ed25519PrivateKey.from_private_bytes(private_bytes))

    @classmethod
    def load_or_create(cls, path: Path) -> RunSigningIdentity:
        if not isinstance(path, Path):
            raise TypeError("signing key path must be an exact Path")
        path = Path(path)
        parent = path.parent
        parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        if parent.is_symlink() or not parent.is_dir():
            raise ValueError("signing key parent must be a regular directory")
        try:
            descriptor = os.open(
                path,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                0o600,
            )
        except FileExistsError:
            descriptor = -1
        if descriptor >= 0:
            key = Ed25519PrivateKey.generate()
            raw = _private_key_bytes(key)
            try:
                _write_all(descriptor, raw)
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
            directory = os.open(parent, os.O_RDONLY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        try:
            descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
        except OSError as error:
            raise ValueError("signing key is not a regular non-symlink file") from error
        try:
            metadata = os.fstat(descriptor)
            if not stat.S_ISREG(metadata.st_mode) or stat.S_IMODE(metadata.st_mode) != 0o600:
                raise ValueError("signing key must be a mode-0600 regular file")
            chunks: list[bytes] = []
            while chunk := os.read(descriptor, 64):
                chunks.append(chunk)
            raw = b"".join(chunks)
        finally:
            os.close(descriptor)
        return cls.from_private_bytes(raw)

    def sign(self, document: object) -> str:
        return base64.b64encode(self._key.sign(canonical_json_bytes(document))).decode("ascii")

    def verify(self, document: object, signature_base64: str) -> bool:
        try:
            signature = base64.b64decode(signature_base64, validate=True)
            public = base64.b64decode(self.public_key_base64, validate=True)
            Ed25519PublicKey.from_public_bytes(public).verify(
                signature,
                canonical_json_bytes(document),
            )
        except (InvalidSignature, TypeError, ValueError, binascii.Error):
            return False
        return True


class SignedRunReceipt(ExternalContract):
    """Authenticated authorization or completion record stored append-only."""

    schema_version: str = "1.0.0"
    receipt_kind: str
    run_id: str
    signer_key_id: str
    public_key_base64: str
    previous_receipt_sha256: str | None
    artifact_digests: dict[str, str]
    subject_sha256: str
    signature_base64: str

    @model_validator(mode="after")
    def receipt_is_closed(self) -> SignedRunReceipt:
        if self.receipt_kind not in {"authorization", "completion"}:
            raise ValueError("receipt kind is undeclared")
        for digest in (
            self.signer_key_id,
            self.subject_sha256,
            *self.artifact_digests.values(),
        ):
            if len(digest) != 64 or not set(digest) <= _HEX:
                raise ValueError("receipt digest is not lowercase SHA-256")
        if self.previous_receipt_sha256 is not None and (
            len(self.previous_receipt_sha256) != 64
            or not set(self.previous_receipt_sha256) <= _HEX
        ):
            raise ValueError("previous receipt digest is not lowercase SHA-256")
        try:
            public = base64.b64decode(self.public_key_base64, validate=True)
            base64.b64decode(self.signature_base64, validate=True)
        except (ValueError, binascii.Error) as error:
            raise ValueError("receipt signature material is not canonical base64") from error
        if len(public) != 32 or hashlib.sha256(public).hexdigest() != self.signer_key_id:
            raise ValueError("receipt key ID does not match public verification material")
        if self.subject_sha256 != _digest_document(
            dict(sorted(self.artifact_digests.items()))
        ):
            raise ValueError("receipt subject digest does not match artifact lineage")
        return self

    def unsigned_document(self) -> dict[str, object]:
        return {
            "artifact_digests": dict(sorted(self.artifact_digests.items())),
            "previous_receipt_sha256": self.previous_receipt_sha256,
            "public_key_base64": self.public_key_base64,
            "receipt_kind": self.receipt_kind,
            "run_id": self.run_id,
            "schema_version": self.schema_version,
            "signer_key_id": self.signer_key_id,
            "subject_sha256": self.subject_sha256,
        }


class RunManifest(ExternalContract):
    """Typed API-safe run lineage containing references, never hidden reasons."""

    schema_version: str = "1.0.0"
    run_id: str
    scenario_id: str
    policy_kind: AttackerPolicyKind
    completed_at: datetime
    artifacts: dict[str, ArtifactRef]
    signer_key_id: str
    public_key_base64: str
    lineage_digest: str
    signature_base64: str

    @model_validator(mode="after")
    def manifest_is_closed(self) -> RunManifest:
        if len(self.lineage_digest) != 64 or not set(self.lineage_digest) <= _HEX:
            raise ValueError("manifest lineage digest must be lowercase SHA-256")
        if self.signer_key_id != hashlib.sha256(
            base64.b64decode(self.public_key_base64, validate=True)
        ).hexdigest():
            raise ValueError("manifest key ID does not match public verification material")
        if self.completed_at.tzinfo is None or self.completed_at.utcoffset() != timedelta(0):
            raise ValueError("manifest completion timestamp must be UTC")
        if not self.artifacts or any(
            type(name) is not str or type(reference) is not ArtifactRef
            for name, reference in self.artifacts.items()
        ):
            raise ValueError("manifest artifacts must be exact named references")
        return self

    def unsigned_document(self) -> dict[str, object]:
        return {
            "artifacts": {
                name: asdict(reference) for name, reference in sorted(self.artifacts.items())
            },
            "completed_at": self.completed_at.isoformat().replace("+00:00", "Z"),
            "lineage_digest": self.lineage_digest,
            "policy_kind": self.policy_kind.value,
            "public_key_base64": self.public_key_base64,
            "run_id": self.run_id,
            "scenario_id": self.scenario_id,
            "schema_version": self.schema_version,
            "signer_key_id": self.signer_key_id,
        }


class RunRunner:
    """Freeze inputs, isolate policy proposals, execute real rails, and sign outputs."""

    def __init__(
        self,
        artifact_store: ArtifactStore,
        signer: RunSigningIdentity,
        run_index_root: Path | None = None,
    ) -> None:
        if type(artifact_store) is not ArtifactStore:
            raise TypeError("artifact_store must be an exact ArtifactStore")
        if type(signer) is not RunSigningIdentity:
            raise TypeError("signer must be an exact RunSigningIdentity")
        self.artifact_store = artifact_store
        self._signer = signer
        if run_index_root is not None and not isinstance(run_index_root, Path):
            raise TypeError("run_index_root must be a Path or None")
        self._run_index_root = None if run_index_root is None else Path(run_index_root)
        self._memory_index: dict[str, ArtifactRef] = {}
        if self._run_index_root is not None:
            self._run_index_root.mkdir(mode=0o700, parents=True, exist_ok=True)
            if self._run_index_root.is_symlink() or not self._run_index_root.is_dir():
                raise ValueError("run index root must be a regular directory")

    def execute(self, bundle: ScenarioBundle, policy: AttackerPolicy) -> RunManifest:
        """Execute one deterministic evaluator-owned run around disposable policies."""
        checked_bundle = self._validated_bundle(bundle)
        checked_policy = self._validated_policy(policy)
        if checked_policy.query_budget > checked_bundle.query_budget:
            raise RunExecutionError("policy query budget exceeds the compiled scenario")
        expected_rail = {
            "agentic_intent_abuse": Rail.AGENTIC,
            "app_scam_mule": Rail.A2A,
            "card_testing_cnp": Rail.CARD,
            "synthetic_merchant_refund": Rail.CARD,
        }[checked_policy.family]
        if checked_bundle.rail is not expected_rail:
            raise RunExecutionError("policy family does not match the compiled scenario rail")

        from apar.generators import PopulationGenerator
        from apar.redteam.benchmark import CampaignBenchmark, default_defender_rules
        from apar.redteam.search import DisclosureProfile

        population = PopulationGenerator(seed=checked_bundle.seed).generate(checked_bundle)
        template = self._template(checked_bundle, checked_policy)
        benchmark = CampaignBenchmark(
            family=checked_policy.family,
            population=population,
            hidden_template=template,
            defender=default_defender_rules(),
            disclosure_profile=DisclosureProfile(
                profile_id=(
                    "run-decision-value-v1"
                    if checked_policy.expose_realized_value
                    else "run-decision-only-v1"
                ),
                expose_realized_value=checked_policy.expose_realized_value,
            ),
            generator_seed=checked_bundle.seed,
        )

        inputs = {
            "policy": self.artifact_store.put_json(checked_policy),
            "population": self.artifact_store.put_bytes(
                population.canonical_bytes(), "application/vnd.apar.population+json"
            ),
            "provenance": self.artifact_store.put_json(_provenance_document()),
            "restricted_evaluation_input": self.artifact_store.put_json(
                {
                    "bounds": bounds_to_wire(benchmark.public_bounds),
                    "evaluation_contract": benchmark.evaluation_contract.model_dump(mode="json"),
                    "hidden_template": _campaign_template_document(template),
                    "restriction": "evaluator_only",
                }
            ),
            "scenario": self.artifact_store.put_json(checked_bundle),
        }
        input_lineage = _artifact_digest_document(inputs)
        run_digest = _digest_document(
            {
                "inputs": input_lineage,
                "signer_key_id": self._signer.key_id,
                "version": "run-v1",
            }
        )
        run_id = f"run-{run_digest[:32]}"
        authorization, authorization_ref = self._store_receipt(
            kind="authorization",
            run_id=run_id,
            artifacts=inputs,
            previous=None,
        )
        if not self._verify_receipt(authorization):
            raise RunExecutionError("authorization receipt failed immediate verification")

        worker = PolicyWorkerClient()
        boundary_report = worker.probe(timeout_ms=checked_policy.worker_timeout_ms)
        if not all(boundary_report.model_dump().values()):
            raise RunExecutionError("policy worker boundary self-test failed closed")
        history, observations = self._search(
            worker=worker,
            benchmark=benchmark,
            policy=checked_policy,
            run_id=run_id,
        )
        winner = min(
            history,
            key=lambda trial: (-trial.objective_value, trial.candidate.candidate_id),
        ).candidate
        events, event_source = self._final_events(
            bundle=checked_bundle,
            policy=checked_policy,
            population=population,
            benchmark=benchmark,
            winner=winner,
        )

        from apar.evaluation_hidden import HiddenValidityOracle

        oracle = HiddenValidityOracle()
        public_validity = oracle.evaluate(events)
        restricted_validity = oracle.evaluate_restricted(events, run_complete=True)
        if public_validity.valid != restricted_validity.valid:
            raise RunExecutionError("hidden validity views disagree")
        outputs = {
            "events": self.artifact_store.put_bytes(
                _event_bytes(events), "application/vnd.apar.events+json"
            ),
            "feedback": self.artifact_store.put_json(
                {
                    "history": history_to_wire(history),
                    "logical_time_used": len(history),
                    "proposal_budget": checked_policy.query_budget,
                    "proposals_used": len(history),
                    "queries_used": len(history),
                    "query_budget": checked_policy.query_budget,
                }
            ),
            "restricted_evaluation_audit": self.artifact_store.put_json(
                {
                    "observations": [observation.document() for observation in observations],
                    "policy_worker_boundary": boundary_report.model_dump(mode="json"),
                    "restriction": "evaluator_only",
                }
            ),
            "restricted_validity": self.artifact_store.put_json(restricted_validity),
            "summary": self.artifact_store.put_json(
                {
                    "adaptive_claim": (
                        "not_supported"
                        if checked_policy.kind is AttackerPolicyKind.ADAPTIVE
                        else "not_applicable"
                    ),
                    "event_count": len(events),
                    "event_source": event_source,
                    "family": checked_policy.family,
                    "hidden_valid": public_validity.valid,
                    "matched_budget": True,
                    "payment_count": len(
                        {
                            cast(str, event.rail_data["payment_id"])
                            for event in events
                            if type(event.rail_data.get("payment_id")) is str
                        }
                    ),
                    "policy_kind": checked_policy.kind.value,
                    "proposals_used": len(history),
                    "run_id": run_id,
                }
            ),
        }
        completion, completion_ref = self._store_receipt(
            kind="completion",
            run_id=run_id,
            artifacts=outputs,
            previous=authorization_ref.sha256,
        )
        if not self._verify_receipt(completion):
            raise RunExecutionError("completion receipt failed immediate verification")
        artifacts = {
            **inputs,
            "authorization_receipt": authorization_ref,
            **outputs,
            "completion_receipt": completion_ref,
        }
        lineage_digest = self._lineage_digest(artifacts)
        completed_at = max(event.event_time for event in events)
        manifest_values: dict[str, object] = {
            "artifacts": artifacts,
            "completed_at": completed_at,
            "lineage_digest": lineage_digest,
            "policy_kind": checked_policy.kind,
            "public_key_base64": self._signer.public_key_base64,
            "run_id": run_id,
            "scenario_id": checked_bundle.scenario_id,
            "schema_version": "1.0.0",
            "signer_key_id": self._signer.key_id,
        }
        unsigned = self._manifest_document(manifest_values)
        manifest = RunManifest(
            artifacts=artifacts,
            completed_at=completed_at,
            lineage_digest=lineage_digest,
            policy_kind=checked_policy.kind,
            public_key_base64=self._signer.public_key_base64,
            run_id=run_id,
            scenario_id=checked_bundle.scenario_id,
            schema_version="1.0.0",
            signer_key_id=self._signer.key_id,
            signature_base64=self._signer.sign(unsigned),
        )
        if not self.verify_run(manifest):
            raise RunExecutionError("completed run failed immediate authenticated verification")
        manifest_ref = self.artifact_store.put_json(manifest)
        self._publish_index(run_id, manifest_ref)
        return manifest

    def verify_manifest(self, manifest: RunManifest) -> bool:
        if type(manifest) is not RunManifest:
            return False
        return self._signer.verify(manifest.unsigned_document(), manifest.signature_base64)

    def verify_run(self, manifest: RunManifest) -> bool:
        try:
            if not self.verify_manifest(manifest):
                return False
            if manifest.lineage_digest != self._lineage_digest(manifest.artifacts):
                return False
            payloads = {
                name: self.artifact_store.read(reference)
                for name, reference in manifest.artifacts.items()
            }
            authorization = SignedRunReceipt.model_validate_json(
                payloads["authorization_receipt"]
            )
            completion = SignedRunReceipt.model_validate_json(payloads["completion_receipt"])
            input_names = {
                "policy",
                "population",
                "provenance",
                "restricted_evaluation_input",
                "scenario",
            }
            output_names = {
                "events",
                "feedback",
                "restricted_evaluation_audit",
                "restricted_validity",
                "summary",
            }
            if set(manifest.artifacts) != input_names | output_names | {
                "authorization_receipt",
                "completion_receipt",
            }:
                return False
            if (
                authorization.run_id != manifest.run_id
                or completion.run_id != manifest.run_id
                or authorization.previous_receipt_sha256 is not None
            ):
                return False
            if authorization.receipt_kind != "authorization" or (
                authorization.artifact_digests
                != {
                    name: manifest.artifacts[name].sha256 for name in sorted(input_names)
                }
            ):
                return False
            if completion.receipt_kind != "completion" or (
                completion.artifact_digests
                != {
                    name: manifest.artifacts[name].sha256 for name in sorted(output_names)
                }
            ):
                return False
            if completion.previous_receipt_sha256 != manifest.artifacts[
                "authorization_receipt"
            ].sha256:
                return False
            if not self._verify_receipt(authorization) or not self._verify_receipt(completion):
                return False
            stored_provenance = strict_json_loads(payloads["provenance"])
            current_provenance = _provenance_document()
            if type(stored_provenance) is not dict:
                return False
            stored = cast(dict[str, object], stored_provenance)
            if set(stored) != set(current_provenance):
                return False
            for key in set(stored) - {"current_source_head"}:
                if stored[key] != current_provenance[key]:
                    return False
            return True
        except (KeyError, TypeError, ValueError, RunExecutionError):
            return False

    def get(self, run_id: str) -> RunManifest:
        if type(run_id) is not str or not run_id.startswith("run-") or len(run_id) != 36:
            raise KeyError("run does not exist")
        reference: ArtifactRef | None
        if self._run_index_root is None:
            reference = self._memory_index.get(run_id)
        else:
            reference = None
            path = self._run_index_root / f"{run_id}.json"
            try:
                raw = _read_run_index_file(path)
            except FileNotFoundError as error:
                raise KeyError("run does not exist") from error
            try:
                loaded = strict_json_loads(raw)
                if type(loaded) is not dict or set(loaded) != {
                    "media_type",
                    "relative_path",
                    "sha256",
                    "size_bytes",
                }:
                    raise RunExecutionError("run index entry is invalid")
                document = cast(dict[str, object], loaded)
                if (
                    type(document["sha256"]) is not str
                    or type(document["media_type"]) is not str
                    or type(document["size_bytes"]) is not int
                    or type(document["relative_path"]) is not str
                ):
                    raise RunExecutionError("run index artifact reference is invalid")
                reference = ArtifactRef(
                    sha256=document["sha256"],
                    media_type=document["media_type"],
                    size_bytes=document["size_bytes"],
                    relative_path=document["relative_path"],
                )
            except (KeyError, TypeError, ValueError) as error:
                raise RunExecutionError("run index entry is invalid") from error
        if reference is None:
            raise KeyError("run does not exist")
        try:
            manifest = RunManifest.model_validate_json(self.artifact_store.read(reference))
        except (TypeError, ValueError) as error:
            raise RunExecutionError("stored run manifest is invalid") from error
        if manifest.run_id != run_id or not self.verify_run(manifest):
            raise RunExecutionError("stored run manifest failed authenticated verification")
        return manifest

    @staticmethod
    def _validated_bundle(bundle: ScenarioBundle) -> ScenarioBundle:
        if type(bundle) is not ScenarioBundle:
            raise TypeError("bundle must be an exact ScenarioBundle")
        return ScenarioBundle.model_validate(bundle.model_dump(mode="json", round_trip=True))

    @staticmethod
    def _validated_policy(policy: AttackerPolicy) -> AttackerPolicy:
        if type(policy) is not AttackerPolicy:
            raise TypeError("policy must be an exact AttackerPolicy")
        return AttackerPolicy.model_validate(policy.model_dump(mode="json", round_trip=True))

    @staticmethod
    def _template(bundle: ScenarioBundle, policy: AttackerPolicy) -> CampaignParams:
        from apar.generators import (
            AGENTIC_INTENT_ABUSE_MOTIF,
            APP_SCAM_MULE_MOTIF,
            CARD_TESTING_CNP_MOTIF,
            SYNTHETIC_MERCHANT_REFUND_MOTIF,
            CampaignParams,
        )

        motifs = {
            "agentic_intent_abuse": AGENTIC_INTENT_ABUSE_MOTIF,
            "app_scam_mule": APP_SCAM_MULE_MOTIF,
            "card_testing_cnp": CARD_TESTING_CNP_MOTIF,
            "synthetic_merchant_refund": SYNTHETIC_MERCHANT_REFUND_MOTIF,
        }
        agentic = policy.family == "agentic_intent_abuse"
        return CampaignParams(
            campaign_id=str(
                uuid5(
                    NAMESPACE_URL,
                    f"apar:run:{bundle.scenario_id}:{policy.family}:{bundle.seed}",
                )
            ),
            seed=bundle.seed,
            payment_count=25 if agentic else 10,
            target_illicit_rate=Decimal("0.92") if agentic else Decimal("0.70"),
            class_rate_tolerance=Decimal("0.01"),
            target_value_total=Decimal("500.00"),
            value_tolerance=Decimal("0.01"),
            min_amount=Decimal("10.00"),
            max_amount=Decimal("90.00"),
            currency="USD",
            duration_hours=min(bundle.duration_hours, 12),
            query_budget=policy.query_budget,
            min_delay_seconds=1,
            max_delay_seconds=60,
            expected_motif=motifs[policy.family],
            cash_out_delay_seconds=60,
            agentic_attack_mix=Decimal("0.92"),
        )

    @staticmethod
    def _search(
        *,
        worker: PolicyWorkerClient,
        benchmark: object,
        policy: AttackerPolicy,
        run_id: str,
    ) -> tuple[tuple[VisibleTrial, ...], tuple[BenchmarkObservation, ...]]:
        from apar.redteam.benchmark import CampaignBenchmark
        from apar.redteam.policies import Feedback, VisibleTrial, visible_objective

        if type(benchmark) is not CampaignBenchmark:
            raise TypeError("benchmark must be exact")
        history: list[VisibleTrial] = []
        observations: list[BenchmarkObservation] = []
        for generation in range(policy.query_budget):
            seed = int.from_bytes(
                hashlib.sha256(f"{run_id}:proposal:{generation}".encode()).digest()[:8],
                "big",
            ) & (2**63 - 1)
            candidate = worker.propose(
                kind=policy.kind,
                bounds=benchmark.public_bounds,
                history=tuple(history),
                seed=seed,
                timeout_ms=policy.worker_timeout_ms,
            )
            returned, observation = benchmark.evaluate_with_observation(candidate)
            feedback = Feedback(
                action=returned.action,
                reason_family=returned.reason_family,
                realized_value=(returned.realized_value if policy.expose_realized_value else None),
            )
            history.append(
                VisibleTrial(
                    candidate=candidate,
                    feedback=feedback,
                    objective_value=visible_objective(feedback),
                )
            )
            observations.append(observation)
        return tuple(history), tuple(observations)

    @staticmethod
    def _final_events(
        *,
        bundle: ScenarioBundle,
        policy: AttackerPolicy,
        population: object,
        benchmark: object,
        winner: AttackCandidate,
    ) -> tuple[tuple[PaymentEvent, ...], str]:
        from apar.generators import CampaignGenerator, Population
        from apar.redteam.benchmark import CampaignBenchmark
        from apar.simulator.engine import SimulationEngine
        from apar.simulator.ledger import AccountReference
        from apar.simulator.rails.a2a import A2ARailAdapter
        from apar.simulator.rails.base import AdapterFactory
        from apar.simulator.rails.card import CardRailAdapter

        if type(population) is not Population or type(benchmark) is not CampaignBenchmark:
            raise TypeError("final replay requires exact evaluator inputs")
        if policy.family == "agentic_intent_abuse":
            from apar.evaluation_hidden import HiddenCampaignGenerator

            return (
                HiddenCampaignGenerator().generate(
                    policy.family,
                    seed=bundle.seed,
                    count=25,
                ),
                "independent_hidden_evaluation_corpus_after_production_trial_replay",
            )
        params = benchmark.compose(winner)
        commands = CampaignGenerator(seed=bundle.seed).generate(
            policy.family,
            population,
            params,
        )
        factory: AdapterFactory
        if bundle.rail is Rail.A2A:

            def a2a_factory() -> A2ARailAdapter:
                return A2ARailAdapter()

            factory = a2a_factory
        else:

            def card_factory() -> CardRailAdapter:
                return CardRailAdapter()

            factory = card_factory
        engine = SimulationEngine(
            bundle,
            {bundle.rail: factory},
            opening_balances=cast(
                dict[AccountReference, Decimal],
                dict(population.opening_balances),
            ),
        )
        started = bundle.replay_manifest.simulation_start
        for priority, command in enumerate(commands):
            engine.schedule(started + timedelta(seconds=priority + 1), priority, command)
        raw_events = engine.run()
        engine.ledger.assert_conserved()
        entities = {entity.entity_id: entity for entity in population.entities}
        balance_by_entity = {
            account.owner_entity_id: account.opening_balance for account in population.accounts
        }
        enriched: list[PaymentEvent] = []
        for event in raw_events:
            actor = entities[event.actor_id]
            counterparty = entities[event.counterparty_id]
            party_refs = {
                **event.party_refs,
                "actor_opening_balance": str(balance_by_entity[event.actor_id]),
                "actor_role": actor.role,
                "counterparty_opening_balance": str(balance_by_entity[event.counterparty_id]),
                "counterparty_role": counterparty.role,
            }
            lineage = {
                **event.lineage,
                "campaign_role": (
                    "attack" if actor.illicit or counterparty.illicit else "control"
                ),
            }
            enriched.append(
                PaymentEvent.model_validate(
                    event.model_copy(
                        update={"lineage": lineage, "party_refs": party_refs}
                    ).model_dump(mode="json", round_trip=True)
                )
            )
        return tuple(enriched), "task5_commands_production_rail_replay"

    def _store_receipt(
        self,
        *,
        kind: str,
        run_id: str,
        artifacts: dict[str, ArtifactRef],
        previous: str | None,
    ) -> tuple[SignedRunReceipt, ArtifactRef]:
        digests = _artifact_digest_document(artifacts)
        subject = _digest_document(digests)
        unsigned: dict[str, object] = {
            "artifact_digests": digests,
            "previous_receipt_sha256": previous,
            "public_key_base64": self._signer.public_key_base64,
            "receipt_kind": kind,
            "run_id": run_id,
            "schema_version": "1.0.0",
            "signer_key_id": self._signer.key_id,
            "subject_sha256": subject,
        }
        receipt = SignedRunReceipt(
            artifact_digests=digests,
            previous_receipt_sha256=previous,
            public_key_base64=self._signer.public_key_base64,
            receipt_kind=kind,
            run_id=run_id,
            schema_version="1.0.0",
            signer_key_id=self._signer.key_id,
            subject_sha256=subject,
            signature_base64=self._signer.sign(unsigned),
        )
        return receipt, self.artifact_store.put_json(receipt)

    def _verify_receipt(self, receipt: SignedRunReceipt) -> bool:
        return (
            type(receipt) is SignedRunReceipt
            and receipt.signer_key_id == self._signer.key_id
            and receipt.public_key_base64 == self._signer.public_key_base64
            and self._signer.verify(receipt.unsigned_document(), receipt.signature_base64)
        )

    @staticmethod
    def _lineage_digest(artifacts: dict[str, ArtifactRef]) -> str:
        return _digest_document(
            {
                "artifacts": _artifact_digest_document(artifacts),
                "authorization_receipt": artifacts["authorization_receipt"].sha256,
                "completion_receipt": artifacts["completion_receipt"].sha256,
            }
        )

    @staticmethod
    def _manifest_document(values: dict[str, object]) -> dict[str, object]:
        artifacts = cast(dict[str, ArtifactRef], values["artifacts"])
        completed = cast(datetime, values["completed_at"])
        policy_kind = cast(AttackerPolicyKind, values["policy_kind"])
        return {
            "artifacts": {
                name: asdict(reference) for name, reference in sorted(artifacts.items())
            },
            "completed_at": completed.astimezone(UTC).isoformat().replace("+00:00", "Z"),
            "lineage_digest": values["lineage_digest"],
            "policy_kind": policy_kind.value,
            "public_key_base64": values["public_key_base64"],
            "run_id": values["run_id"],
            "scenario_id": values["scenario_id"],
            "schema_version": values["schema_version"],
            "signer_key_id": values["signer_key_id"],
        }

    def _publish_index(self, run_id: str, manifest_ref: ArtifactRef) -> None:
        self._memory_index[run_id] = manifest_ref
        if self._run_index_root is None:
            return
        path = self._run_index_root / f"{run_id}.json"
        raw = canonical_json_bytes(asdict(manifest_ref))
        try:
            descriptor = os.open(
                path,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                0o600,
            )
        except FileExistsError:
            try:
                existing = _read_run_index_file(path)
            except FileNotFoundError as error:
                raise RunExecutionError("append-only run index collision") from error
            if existing != raw:
                raise RunExecutionError("append-only run index collision") from None
            return
        try:
            _write_all(descriptor, raw)
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        directory = os.open(
            self._run_index_root,
            os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
        )
        try:
            os.fsync(directory)
        finally:
            os.close(directory)


__all__ = [
    "AttackerPolicy",
    "AttackerPolicyKind",
    "PolicyWorkerBoundaryReport",
    "PolicyWorkerClient",
    "PolicyWorkerError",
    "RunExecutionError",
    "RunManifest",
    "RunRunner",
    "RunSigningIdentity",
    "SignedRunReceipt",
]
