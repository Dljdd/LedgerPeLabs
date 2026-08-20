"""Authenticated, isolated, durable defense-evaluation publication service."""

from __future__ import annotations

import ast
import fcntl
import hashlib
import multiprocessing
import os
import resource
import select
import socket
import stat
import struct
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager, suppress
from multiprocessing.connection import Connection
from multiprocessing.process import BaseProcess
from pathlib import Path
from typing import NamedTuple, Never, Protocol, cast
from weakref import WeakKeyDictionary

from apar.evaluation.defender_attestation import DefenderBundleVerifier
from apar.evaluation.gates import (
    EvaluatorReplayVerifier,
    EvaluatorSigningIdentity,
)
from apar.evaluation.publication_inputs import (
    PublicationInputError,
    VerifiedEvaluationInputs,
    verify_evaluation_inputs,
)
from apar.evaluation.replay import ReplayThresholdSet
from apar.evaluation.reporting import (
    PUBLIC_ARTIFACT_MEDIA_TYPES,
    RESTRICTED_PUBLICATION_RECEIPT_MEDIA_TYPE,
    SCORECARD_ARTIFACT_NAME,
    DefenseScorecard,
    EvaluationArtifactBundle,
    PublicArtifactReference,
    PublicArtifactVerifier,
    ReportingContractError,
    ScorecardPublicationRequest,
    load_evaluation_bundle,
    publish_scorecard,
    store_restricted_publication_receipt,
)
from apar.runs.runner import RunSigningIdentity
from apar.runs.wire import WireContractError, canonical_json_bytes, strict_json_loads
from apar.storage.artifacts import ArtifactRef, ArtifactStore

MAX_EVALUATIONS = 10_000
MAX_EXECUTION_SECONDS = 900.0
MAX_EXECUTOR_RESULT_BYTES = 128 * 1024 * 1024
MAX_INDEX_BYTES = 64 * 1024
INDEX_MEDIA_TYPE = "application/vnd.apar.defense-evaluation-index+json"
_INDEX_DIRECTORY = ".defense-evaluation-index-v1"
_INDEX_LOCK = "index.lock"


class DefenseServiceError(ValueError):
    """Base class for stable service error normalization."""


class DefenseResourceNotFound(DefenseServiceError):
    """An input or published evaluation cannot be resolved."""


class DefenseArtifactInvalid(DefenseServiceError):
    """A resolved immutable artifact fails semantic verification."""


class DefenseExecutionConflict(DefenseServiceError):
    """Evaluation cannot execute or pass its publication gates."""


class DefenseServiceUnavailable(DefenseServiceError):
    """The independent defense trust roots are not configured."""


class _ExecutorState(NamedTuple):
    source: bytes
    config: bytes
    capability: bytes
    timeout_seconds: float


class DefenseEvaluationExecutor(Protocol):
    @property
    def timeout_seconds(self) -> float: ...

    def execute(
        self,
        inputs: VerifiedEvaluationInputs,
        *,
        artifact_root: Path,
        evaluator_verifier: EvaluatorReplayVerifier,
    ) -> ScorecardPublicationRequest: ...


class _SealedExecutorType(type):
    def __setattr__(cls, name: str, value: object) -> None:
        del cls, name, value
        raise TypeError("evaluation executor type is sealed")

    def __delattr__(cls, name: str) -> None:
        del cls, name
        raise TypeError("evaluation executor type is sealed")


def _build_executor_type() -> type:
    states: WeakKeyDictionary[object, _ExecutorState] = WeakKeyDictionary()
    lock = threading.RLock()
    exact_type: type | None = None

    def state_for(instance: object) -> _ExecutorState:
        if exact_type is None or type(instance) is not exact_type:
            raise DefenseExecutionConflict("evaluation executor state is invalid")
        with lock:
            try:
                state = states[instance]
            except KeyError as error:
                raise DefenseExecutionConflict(
                    "evaluation executor state is invalid"
                ) from error
        if type(state) is not _ExecutorState:
            raise DefenseExecutionConflict("evaluation executor state is invalid")
        return state

    class _EvaluationExecutor(metaclass=_SealedExecutorType):
        __slots__ = ("__weakref__",)

        def __new__(cls, *args: object, **kwargs: object) -> Never:
            del cls, args, kwargs
            raise TypeError("evaluation executors require the sealed factory")

        def __init__(self, *args: object, **kwargs: object) -> None:
            del self, args, kwargs
            raise TypeError("evaluation executor cannot be reinitialized")

        def __setattr__(self, name: str, value: object) -> None:
            del self, name, value
            raise TypeError("evaluation executor is immutable")

        def __delattr__(self, name: str) -> None:
            del self, name
            raise TypeError("evaluation executor is immutable")

        def __reduce__(self) -> Never:
            raise TypeError("evaluation executor cannot be serialized")

        def __copy__(self) -> Never:
            raise TypeError("evaluation executor cannot be copied")

        def __deepcopy__(self, memo: object) -> Never:
            del memo
            raise TypeError("evaluation executor cannot be copied")

        def __repr__(self) -> str:
            return "<sealed evaluator worker capability>"

        @classmethod
        def from_signed_source(
            cls,
            *,
            source_path: Path,
            callable_qualname: str,
            version: str,
            config: dict[str, object],
            signer: EvaluatorSigningIdentity,
            timeout_seconds: float = MAX_EXECUTION_SECONDS,
        ) -> object:
            if cls is not exact_type or not EvaluatorSigningIdentity.is_exact(signer):
                raise TypeError("evaluation worker requires the exact evaluator authority")
            if (
                not isinstance(source_path, Path)
                or source_path.is_symlink()
                or not source_path.is_file()
            ):
                raise TypeError("evaluation worker source must be one exact regular file")
            source = source_path.read_bytes()
            _audit_worker_source(source)
            if (
                type(callable_qualname) is not str
                or not callable_qualname.isidentifier()
                or type(version) is not str
                or not _semantic_version(version)
            ):
                raise ValueError("evaluation worker identity is invalid")
            if type(config) is not dict:
                raise TypeError("evaluation worker config must be an exact object")
            config_payload = canonical_json_bytes(config)
            if not 0 < len(config_payload) <= MAX_EXECUTOR_RESULT_BYTES:
                raise ValueError("evaluation worker config exceeds its cap")
            if (
                type(timeout_seconds) is not float
                or not 0.01 <= timeout_seconds <= MAX_EXECUTION_SECONDS
            ):
                raise ValueError("evaluation executor timeout is outside its cap")
            source_digest = hashlib.sha256(source).hexdigest()
            inventory = [
                {
                    "module_id": "evaluator_worker.py",
                    "sha256": source_digest,
                    "size_bytes": len(source),
                }
            ]
            fields: dict[str, object] = {
                "schema_version": "1.0.0",
                "callable_qualname": callable_qualname,
                "worker_version": version,
                "input_schema": "apar.verified-evaluation-inputs.v1",
                "output_schema": "apar.scorecard-publication-request.v1",
                "source_inventory": inventory,
                "source_inventory_digest": _digest_document(inventory),
                "config_digest": hashlib.sha256(config_payload).hexdigest(),
                "signer_key_id": signer.key_id,
                "public_key_base64": signer.public_key_base64,
            }
            signature = signer._sign(fields)
            capability = canonical_json_bytes(
                {
                    **fields,
                    "signature_base64": signature,
                    "capability_digest": _digest_document(
                        {**fields, "signature_base64": signature}
                    ),
                }
            )
            instance: _EvaluationExecutor = object.__new__(cast(type, exact_type))
            with lock:
                states[instance] = _ExecutorState(
                    bytes(source), bytes(config_payload), capability, timeout_seconds
                )
            return instance

        @property
        def timeout_seconds(self) -> float:
            return state_for(self).timeout_seconds

        def execute(
            self,
            inputs: VerifiedEvaluationInputs,
            *,
            artifact_root: Path,
            evaluator_verifier: EvaluatorReplayVerifier,
        ) -> ScorecardPublicationRequest:
            if type(inputs) is not VerifiedEvaluationInputs:
                raise DefenseExecutionConflict("executor input capability is invalid")
            if not isinstance(artifact_root, Path) or type(
                evaluator_verifier
            ) is not EvaluatorReplayVerifier:
                raise DefenseExecutionConflict("executor trust capability is invalid")
            state = state_for(self)
            capability = _verify_worker_capability(state, evaluator_verifier)
            snapshot = _snapshot_worker_source(
                artifact_root, state.source, cast(str, capability["source_inventory_digest"])
            )
            try:
                defender_document = strict_json_loads(inputs.defender.to_json())
            except WireContractError as error:
                raise DefenseExecutionConflict("verified defender input is invalid") from error
            if type(defender_document) is not dict:
                raise DefenseExecutionConflict("verified defender input is invalid")
            defender_identity = cast(dict[str, object], defender_document)
            defender_key_id = defender_identity.get("signer_key_id")
            defender_public_key = defender_identity.get("public_key_base64")
            if type(defender_key_id) is not str or type(defender_public_key) is not str:
                raise DefenseExecutionConflict("verified defender identity is invalid")
            deadline = time.monotonic() + state.timeout_seconds
            try:
                context = multiprocessing.get_context("spawn")
                parent, child = context.Pipe(duplex=False)
                process = context.Process(
                    target=_executor_child,
                    args=(
                        child,
                        str(snapshot),
                        state.source,
                        state.config,
                        state.capability,
                        str(artifact_root),
                        inputs.corpus.top_ref,
                        inputs.defender.top_ref,
                        evaluator_verifier.key_id,
                        evaluator_verifier.public_key_base64,
                        defender_key_id,
                        defender_public_key,
                    ),
                    daemon=True,
                )
                process.start()
                child.close()
            except Exception as error:
                raise DefenseExecutionConflict("defense evaluation could not start") from error
            try:
                payload = _read_framed_result(parent, process, deadline)
                remaining = max(0.0, deadline - time.monotonic())
                process.join(timeout=min(1.0, remaining))
                if process.is_alive():
                    _terminate_process(process)
                    raise DefenseExecutionConflict("defense evaluation did not exit")
                if process.exitcode != 0 or not payload.startswith(b"O"):
                    raise DefenseExecutionConflict("defense evaluation was rejected")
                raw = payload[1:]
                if not 0 < len(raw) <= MAX_EXECUTOR_RESULT_BYTES:
                    raise DefenseExecutionConflict(
                        "defense evaluation result exceeds its cap"
                    )
                try:
                    return ScorecardPublicationRequest.from_worker_json(raw)
                except (ReportingContractError, TypeError, ValueError) as error:
                    raise DefenseExecutionConflict(
                        "defense evaluation result is invalid"
                    ) from error
            finally:
                parent.close()
                if process.is_alive():
                    _terminate_process(process)

    exact_type = _EvaluationExecutor
    return _EvaluationExecutor


EvaluationExecutor = _build_executor_type()


def _semantic_version(value: str) -> bool:
    parts = value.split(".")
    return len(parts) == 3 and all(part.isascii() and part.isdigit() for part in parts)


def _audit_worker_source(source: bytes) -> None:
    if type(source) is not bytes or not 0 < len(source) <= 2 * 1024 * 1024:
        raise ValueError("evaluation worker source exceeds its cap")
    try:
        text = source.decode("utf-8")
        tree = ast.parse(text, filename="evaluator_worker.py", mode="exec")
    except (UnicodeDecodeError, SyntaxError) as error:
        raise ValueError("evaluation worker source is invalid") from error
    banned_modules = {
        "asyncio",
        "ctypes",
        "ftplib",
        "http",
        "importlib",
        "multiprocessing",
        "requests",
        "socket",
        "subprocess",
        "sys",
        "telnetlib",
        "urllib",
    }
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = (
                tuple(alias.name for alias in node.names)
                if isinstance(node, ast.Import)
                else ((node.module or ""),)
            )
            if any(name.split(".", 1)[0] in banned_modules for name in names):
                raise ValueError("evaluation worker imports a forbidden capability")
            if any(
                alias.name.split(".", 1)[0] in banned_modules
                or any(
                    token in alias.name.casefold()
                    for token in ("socket", "subprocess", "urlopen", "network")
                )
                for alias in node.names
            ):
                raise ValueError("evaluation worker imports a prebound capability")
        if isinstance(node, (ast.Name, ast.Attribute)):
            identifier = (
                node.id if isinstance(node, ast.Name) else node.attr
            ).casefold()
            if any(
                token in identifier
                for token in (
                    "create_connection",
                    "fork",
                    "popen",
                    "socket",
                    "spawn",
                    "subprocess",
                    "urlopen",
                )
            ):
                raise ValueError("evaluation worker references a forbidden capability")
        if isinstance(node, ast.Call):
            target = node.func
            if isinstance(target, ast.Name) and target.id in {
                "__import__",
                "compile",
                "eval",
                "exec",
            }:
                raise ValueError("evaluation worker uses dynamic code loading")
            if (
                isinstance(target, ast.Attribute)
                and target.attr in {"import_module", "exec_module", "load_module"}
            ):
                raise ValueError("evaluation worker uses dynamic module loading")


def _verify_worker_capability(
    state: _ExecutorState, verifier: EvaluatorReplayVerifier
) -> dict[str, object]:
    try:
        raw = strict_json_loads(state.capability)
    except WireContractError as error:
        raise DefenseExecutionConflict("evaluation worker capability is invalid") from error
    fields = {
        "schema_version",
        "callable_qualname",
        "worker_version",
        "input_schema",
        "output_schema",
        "source_inventory",
        "source_inventory_digest",
        "config_digest",
        "signer_key_id",
        "public_key_base64",
        "signature_base64",
        "capability_digest",
    }
    if type(raw) is not dict or set(raw) != fields or canonical_json_bytes(raw) != state.capability:
        raise DefenseExecutionConflict("evaluation worker capability fields differ")
    document = cast(dict[str, object], raw)
    signed = {key: value for key, value in document.items() if key != "capability_digest"}
    unsigned = {key: value for key, value in signed.items() if key != "signature_base64"}
    signature = document["signature_base64"]
    inventory = document["source_inventory"]
    source_digest = hashlib.sha256(state.source).hexdigest()
    if (
        document["schema_version"] != "1.0.0"
        or document["input_schema"] != "apar.verified-evaluation-inputs.v1"
        or document["output_schema"] != "apar.scorecard-publication-request.v1"
        or document["signer_key_id"] != verifier.key_id
        or document["public_key_base64"] != verifier.public_key_base64
        or type(signature) is not str
        or not verifier.verify_document(unsigned, signature)
        or document["capability_digest"] != _digest_document(signed)
        or document["config_digest"] != hashlib.sha256(state.config).hexdigest()
        or type(inventory) is not list
        or inventory
        != [
            {
                "module_id": "evaluator_worker.py",
                "sha256": source_digest,
                "size_bytes": len(state.source),
            }
        ]
        or document["source_inventory_digest"] != _digest_document(inventory)
    ):
        raise DefenseExecutionConflict("evaluation worker capability signature is invalid")
    _audit_worker_source(state.source)
    return document


def _snapshot_worker_source(root: Path, source: bytes, inventory_digest: str) -> Path:
    directory = root / ".defense-evaluator-workers-v1"
    directory.mkdir(mode=0o700, exist_ok=True)
    if directory.is_symlink() or not directory.is_dir():
        raise DefenseExecutionConflict("evaluation worker snapshot root is invalid")
    directory.chmod(0o700)
    snapshot_directory = directory / inventory_digest
    snapshot_directory.mkdir(mode=0o700, exist_ok=True)
    if snapshot_directory.is_symlink() or not snapshot_directory.is_dir():
        raise DefenseExecutionConflict("evaluation worker snapshot is invalid")
    snapshot_directory.chmod(0o700)
    path = snapshot_directory / "worker.py"
    if path.exists():
        if path.is_symlink() or path.read_bytes() != source:
            raise DefenseExecutionConflict("evaluation worker snapshot differs")
    else:
        temporary = snapshot_directory / f".tmp-{os.getpid()}-{threading.get_ident()}"
        descriptor = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
            0o600,
        )
        try:
            offset = 0
            while offset < len(source):
                written = os.write(descriptor, source[offset:])
                if written <= 0:
                    raise OSError("worker snapshot write made no progress")
                offset += written
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        try:
            os.link(temporary, path, follow_symlinks=False)
        except FileExistsError:
            if path.is_symlink() or path.read_bytes() != source:
                raise DefenseExecutionConflict("concurrent worker snapshot differs") from None
        finally:
            with suppress(FileNotFoundError):
                temporary.unlink()
    path.chmod(0o400)
    if path.read_bytes() != source:
        raise DefenseExecutionConflict("evaluation worker snapshot digest differs")
    return path


def _read_framed_result(
    pipe: Connection, process: BaseProcess, deadline: float
) -> bytes:
    descriptor = pipe.fileno()
    os.set_blocking(descriptor, False)
    buffer = bytearray()
    expected: int | None = None
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            _terminate_process(process)
            raise DefenseExecutionConflict("defense evaluation timed out")
        readable, _, _ = select.select((descriptor,), (), (), remaining)
        if not readable:
            _terminate_process(process)
            raise DefenseExecutionConflict("defense evaluation timed out")
        try:
            chunk = os.read(descriptor, min(64 * 1024, MAX_EXECUTOR_RESULT_BYTES + 9))
        except BlockingIOError:
            continue
        if not chunk:
            raise DefenseExecutionConflict("defense evaluation returned no result")
        buffer.extend(chunk)
        if expected is None and len(buffer) >= 8:
            expected = struct.unpack(">Q", bytes(buffer[:8]))[0]
            del buffer[:8]
            if not 0 < expected <= MAX_EXECUTOR_RESULT_BYTES + 1:
                _terminate_process(process)
                raise DefenseExecutionConflict("defense evaluation frame exceeds its cap")
        if expected is not None and len(buffer) == expected:
            return bytes(buffer)
        if expected is not None and len(buffer) > expected:
            _terminate_process(process)
            raise DefenseExecutionConflict("defense evaluation frame is malformed")


def _executor_child(
    pipe: Connection,
    snapshot_path: str,
    expected_source: bytes,
    config_payload: bytes,
    capability_payload: bytes,
    artifact_root: str,
    corpus_ref: ArtifactRef,
    defender_ref: ArtifactRef,
    evaluator_key_id: str,
    evaluator_public_key: str,
    defender_key_id: str,
    defender_public_key: str,
) -> None:
    try:
        _apply_child_limits()
        _close_inherited_fds(pipe.fileno())
        _disable_child_network()
        store = ArtifactStore(Path(artifact_root))
        evaluator_verifier = EvaluatorReplayVerifier(
            signer_key_id=evaluator_key_id,
            public_key_base64=evaluator_public_key,
        )
        state = _ExecutorState(expected_source, config_payload, capability_payload, 1.0)
        capability = _verify_worker_capability(state, evaluator_verifier)
        path = Path(snapshot_path)
        source = path.read_bytes()
        if source != expected_source or path.is_symlink():
            raise ValueError("worker source snapshot changed")
        defender_verifier = DefenderBundleVerifier(
            store,
            signer_key_id=defender_key_id,
            public_key_base64=defender_public_key,
        )
        inputs = verify_evaluation_inputs(
            corpus_ref=corpus_ref,
            defender_ref=defender_ref,
            artifact_store=store,
            evaluator_verifier=evaluator_verifier,
            defender_verifier=defender_verifier,
        )
        config = strict_json_loads(config_payload)
        if type(config) is not dict:
            raise ValueError("worker config is not an object")
        namespace: dict[str, object] = {
            "__builtins__": __builtins__,
            "__file__": str(path),
            "__name__": "_apar_verified_evaluator_worker",
        }
        code = compile(source, "evaluator_worker.py", "exec")
        exec(code, namespace)
        _audit_worker_namespace(namespace)
        worker = namespace.get(cast(str, capability["callable_qualname"]))
        if (
            type(worker).__name__ != "function"
            or getattr(worker, "__module__", None) != namespace["__name__"]
        ):
            raise TypeError("worker callable identity differs")
        payload = worker(inputs, config)  # type: ignore[operator]
        if type(payload) is not bytes:
            raise TypeError("executor output must be exact canonical request bytes")
        ScorecardPublicationRequest.from_worker_json(payload)
        if len(payload) > MAX_EXECUTOR_RESULT_BYTES:
            raise ValueError("executor output is too large")
        _write_frame(pipe.fileno(), b"O" + payload)
    except BaseException:
        with suppress(Exception):
            _write_frame(pipe.fileno(), b"E")
        raise SystemExit(1) from None
    finally:
        pipe.close()


def _write_frame(descriptor: int, payload: bytes) -> None:
    framed = struct.pack(">Q", len(payload)) + payload
    offset = 0
    while offset < len(framed):
        written = os.write(descriptor, framed[offset : offset + 64 * 1024])
        if written <= 0:
            raise OSError("executor frame write made no progress")
        offset += written


def _audit_worker_namespace(namespace: dict[str, object]) -> None:
    banned = ("socket", "subprocess", "requests", "urllib", "http.client", "ftplib")
    for name, value in namespace.items():
        if name.startswith("__"):
            continue
        origins = (
            getattr(value, "__module__", ""),
            getattr(value, "__name__", ""),
            type(value).__module__,
        )
        if any(
            type(origin) is str
            and any(
                origin == token or origin.startswith(f"{token}.")
                for token in banned
            )
            for origin in origins
        ):
            raise TypeError("worker namespace contains a prebound forbidden capability")


def _close_inherited_fds(preserve: int) -> None:
    try:
        maximum = min(4096, int(resource.getrlimit(resource.RLIMIT_NOFILE)[0]))
    except (OSError, ValueError):
        maximum = 256
    for descriptor in range(3, maximum):
        if descriptor != preserve:
            with suppress(OSError):
                os.close(descriptor)


def _apply_child_limits() -> None:
    """Apply portable hard limits independent of caller-controlled values."""
    limits = (
        (resource.RLIMIT_CPU, 901),
        (resource.RLIMIT_AS, 2 * 1024 * 1024 * 1024),
        (resource.RLIMIT_DATA, 2 * 1024 * 1024 * 1024),
        (resource.RLIMIT_FSIZE, MAX_EXECUTOR_RESULT_BYTES + 1),
        (resource.RLIMIT_NOFILE, 64),
        (resource.RLIMIT_NPROC, 0),
        (resource.RLIMIT_CORE, 0),
    )
    for kind, desired in limits:
        with suppress(OSError, ValueError):
            _soft, hard = resource.getrlimit(kind)
            cap = desired if hard == resource.RLIM_INFINITY else min(desired, hard)
            resource.setrlimit(kind, (cap, cap))


def _disable_child_network() -> None:
    def blocked(*args: object, **kwargs: object) -> Never:
        del args, kwargs
        raise PermissionError("evaluation workers have no network capability")

    socket.socket = blocked  # type: ignore[assignment,misc]
    socket.create_connection = blocked


def _terminate_process(process: BaseProcess) -> None:
    process.terminate()
    process.join(timeout=1.0)
    if process.is_alive():
        process.kill()
        process.join(timeout=1.0)


class _IndexRecord(NamedTuple):
    input_key: str
    corpus_digest: str
    defender_digest: str
    evaluation_id: str
    bundle_ref: ArtifactRef
    receipt_ref: ArtifactRef


class _IndexRepository:
    """One signed atomic pointer per input pair beneath a private index directory."""

    __slots__ = ("_directory_fd", "_signer", "_verifier")

    def __init__(
        self,
        *,
        root: Path,
        signer: RunSigningIdentity,
        verifier: PublicArtifactVerifier,
    ) -> None:
        if not isinstance(root, Path):
            raise TypeError("index root must be a Path")
        root_fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            _validate_directory(root_fd, "artifact root")
            with suppress(FileExistsError):
                os.mkdir(_INDEX_DIRECTORY, 0o700, dir_fd=root_fd)
            directory_fd = os.open(
                _INDEX_DIRECTORY,
                os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                dir_fd=root_fd,
            )
            _validate_directory(directory_fd, "defense index")
        finally:
            os.close(root_fd)
        self._directory_fd = directory_fd
        self._signer = signer
        self._verifier = verifier
        lock_fd = os.open(
            _INDEX_LOCK,
            os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW,
            0o600,
            dir_fd=directory_fd,
        )
        os.fchmod(lock_fd, 0o600)
        os.close(lock_fd)

    @contextmanager
    def locked(self) -> Iterator[None]:
        lock_fd = os.open(_INDEX_LOCK, os.O_RDWR | os.O_NOFOLLOW, dir_fd=self._directory_fd)
        try:
            _validate_regular(lock_fd, "defense index lock", max_bytes=0)
            fcntl.flock(lock_fd, fcntl.LOCK_EX)
            yield
        finally:
            with suppress(OSError):
                fcntl.flock(lock_fd, fcntl.LOCK_UN)
            os.close(lock_fd)

    def all(self) -> tuple[_IndexRecord, ...]:
        names: list[str] = []
        for name in sorted(os.listdir(self._directory_fd)):
            if name == _INDEX_LOCK:
                continue
            if _valid_temporary_name(name):
                self._discard_interrupted_temporary(name)
                continue
            names.append(name)
        if len(names) > MAX_EVALUATIONS:
            raise DefenseArtifactInvalid("defense evaluation index exceeds its cap")
        records: list[_IndexRecord] = []
        for name in names:
            if not _valid_index_name(name):
                raise DefenseArtifactInvalid("defense evaluation index contains an alias")
            record = self._read(name)
            if name != f"{record.input_key}.json":
                raise DefenseArtifactInvalid("defense evaluation index name differs")
            records.append(record)
        if len({record.input_key for record in records}) != len(records):
            raise DefenseArtifactInvalid("defense evaluation index is duplicated")
        if len({record.evaluation_id for record in records}) != len(records):
            raise DefenseArtifactInvalid("defense evaluation index is ambiguous")
        return tuple(records)

    def _discard_interrupted_temporary(self, name: str) -> None:
        """Remove only this index's exact private temp namespace while locked."""
        try:
            fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=self._directory_fd)
        except OSError as error:
            raise DefenseArtifactInvalid("defense index temporary is invalid") from error
        try:
            metadata = os.fstat(fd)
            if (
                not stat.S_ISREG(metadata.st_mode)
                or stat.S_IMODE(metadata.st_mode) != 0o600
                or metadata.st_uid != os.geteuid()
                or metadata.st_nlink not in {1, 2}
                or not 0 <= metadata.st_size <= MAX_INDEX_BYTES
            ):
                raise DefenseArtifactInvalid("defense index temporary is invalid")
        finally:
            os.close(fd)
        os.unlink(name, dir_fd=self._directory_fd)
        os.fsync(self._directory_fd)

    def publish(self, record: _IndexRecord) -> None:
        name = f"{record.input_key}.json"
        payload = _index_payload(record, self._signer)
        for existing_record in self.all():
            if (
                existing_record.evaluation_id == record.evaluation_id
                and existing_record != record
            ):
                raise DefenseExecutionConflict(
                    "evaluation identity is already bound to another input"
                )
        existing = self._try_read(name)
        if existing is not None:
            if existing != record:
                raise DefenseExecutionConflict("evaluation input pair is already published")
            return
        temporary = f".tmp-{os.getpid()}-{threading.get_ident()}-{record.input_key}"
        fd = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
            0o600,
            dir_fd=self._directory_fd,
        )
        try:
            os.fchmod(fd, 0o600)
            offset = 0
            while offset < len(payload):
                written = os.write(fd, payload[offset:])
                if written <= 0:
                    raise OSError("index write made no progress")
                offset += written
            os.fsync(fd)
            _validate_regular(fd, "new defense index", max_bytes=MAX_INDEX_BYTES)
        finally:
            os.close(fd)
        try:
            os.link(
                temporary,
                name,
                src_dir_fd=self._directory_fd,
                dst_dir_fd=self._directory_fd,
                follow_symlinks=False,
            )
            os.fsync(self._directory_fd)
        except FileExistsError:
            existing = self._read(name)
            if existing != record:
                raise DefenseExecutionConflict(
                    "concurrent evaluation publication differs"
                ) from None
        finally:
            with suppress(FileNotFoundError):
                os.unlink(temporary, dir_fd=self._directory_fd)

    def _try_read(self, name: str) -> _IndexRecord | None:
        try:
            return self._read(name)
        except FileNotFoundError:
            return None

    def _read(self, name: str) -> _IndexRecord:
        try:
            fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=self._directory_fd)
        except FileNotFoundError:
            raise
        except OSError as error:
            raise DefenseArtifactInvalid("defense evaluation index is invalid") from error
        try:
            before = os.fstat(fd)
            _validate_regular(fd, "defense evaluation index", max_bytes=MAX_INDEX_BYTES)
            chunks: list[bytes] = []
            while chunk := os.read(fd, 16 * 1024):
                chunks.append(chunk)
            after = os.fstat(fd)
            if _stable_stat(before) != _stable_stat(after):
                raise DefenseArtifactInvalid("defense evaluation index changed while read")
        finally:
            os.close(fd)
        return _parse_index_payload(b"".join(chunks), self._verifier)


def _stable_stat(value: os.stat_result) -> tuple[int, ...]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_nlink,
        value.st_uid,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _validate_directory(fd: int, label: str) -> None:
    metadata = os.fstat(fd)
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or stat.S_IMODE(metadata.st_mode) != 0o700
        or metadata.st_uid != os.geteuid()
        or metadata.st_nlink < 1
    ):
        raise DefenseArtifactInvalid(f"{label} is invalid")


def _validate_regular(fd: int, label: str, *, max_bytes: int) -> None:
    metadata = os.fstat(fd)
    if (
        not stat.S_ISREG(metadata.st_mode)
        or stat.S_IMODE(metadata.st_mode) != 0o600
        or metadata.st_uid != os.geteuid()
        or metadata.st_nlink != 1
        or not 0 <= metadata.st_size <= max_bytes
    ):
        raise DefenseArtifactInvalid(f"{label} is invalid")


def _input_key(corpus_digest: str, defender_digest: str) -> str:
    return hashlib.sha256(
        canonical_json_bytes(
            {
                "corpus_artifact_digest": corpus_digest,
                "defender_artifact_digest": defender_digest,
            }
        )
    ).hexdigest()


def _valid_index_name(name: str) -> bool:
    return (
        type(name) is str
        and len(name) == 69
        and name.endswith(".json")
        and all(character in "0123456789abcdef" for character in name[:-5])
    )


def _valid_temporary_name(name: str) -> bool:
    parts = name.split("-")
    return (
        len(parts) == 4
        and parts[0] == ".tmp"
        and parts[1].isascii()
        and parts[1].isdigit()
        and parts[2].isascii()
        and parts[2].isdigit()
        and len(parts[3]) == 64
        and all(character in "0123456789abcdef" for character in parts[3])
    )


def _ref_document(ref: ArtifactRef) -> dict[str, object]:
    return {
        "media_type": ref.media_type,
        "relative_path": ref.relative_path,
        "sha256": ref.sha256,
        "size_bytes": ref.size_bytes,
    }


def _index_payload(record: _IndexRecord, signer: RunSigningIdentity) -> bytes:
    fields: dict[str, object] = {
        "schema_version": "1.0.0",
        "input_key": record.input_key,
        "corpus_artifact_digest": record.corpus_digest,
        "defender_artifact_digest": record.defender_digest,
        "evaluation_id": record.evaluation_id,
        "bundle_ref": _ref_document(record.bundle_ref),
        "restricted_receipt_ref": _ref_document(record.receipt_ref),
        "signer_key_id": signer.key_id,
        "public_key_base64": signer.public_key_base64,
    }
    signature = signer.sign(fields)
    signed = {**fields, "signature_base64": signature}
    return canonical_json_bytes({**signed, "index_digest": _digest_document(signed)})


def _parse_index_payload(payload: bytes, verifier: PublicArtifactVerifier) -> _IndexRecord:
    if type(payload) is not bytes or not 0 < len(payload) <= MAX_INDEX_BYTES:
        raise DefenseArtifactInvalid("defense evaluation index payload is invalid")
    try:
        raw = strict_json_loads(payload)
    except WireContractError as error:
        raise DefenseArtifactInvalid("defense evaluation index JSON is invalid") from error
    fields = {
        "schema_version",
        "input_key",
        "corpus_artifact_digest",
        "defender_artifact_digest",
        "evaluation_id",
        "bundle_ref",
        "restricted_receipt_ref",
        "signer_key_id",
        "public_key_base64",
        "signature_base64",
        "index_digest",
    }
    if type(raw) is not dict or set(raw) != fields or canonical_json_bytes(raw) != payload:
        raise DefenseArtifactInvalid("defense evaluation index fields differ")
    document = cast(dict[str, object], raw)
    signed = {key: value for key, value in document.items() if key != "index_digest"}
    unsigned = {key: value for key, value in signed.items() if key != "signature_base64"}
    signature = document["signature_base64"]
    if (
        document["schema_version"] != "1.0.0"
        or document["signer_key_id"] != verifier.key_id
        or document["public_key_base64"] != verifier.public_key_base64
        or type(signature) is not str
        or not verifier.verify(unsigned, signature)
        or document["index_digest"] != _digest_document(signed)
    ):
        raise DefenseArtifactInvalid("defense evaluation index signature is invalid")
    input_key = _validate_digest(document["input_key"])
    corpus = _validate_digest(document["corpus_artifact_digest"])
    defender = _validate_digest(document["defender_artifact_digest"])
    evaluation = _validate_digest(document["evaluation_id"])
    if input_key != _input_key(corpus, defender):
        raise DefenseArtifactInvalid("defense evaluation index input binding differs")
    return _IndexRecord(
        input_key,
        corpus,
        defender,
        evaluation,
        _artifact_ref(
            document["bundle_ref"],
            media_type="application/vnd.apar.evaluation-artifact-bundle+json",
        ),
        _artifact_ref(
            document["restricted_receipt_ref"],
            media_type=RESTRICTED_PUBLICATION_RECEIPT_MEDIA_TYPE,
        ),
    )


def _artifact_ref(value: object, *, media_type: str) -> ArtifactRef:
    if type(value) is not dict or set(value) != {
        "media_type",
        "relative_path",
        "sha256",
        "size_bytes",
    }:
        raise DefenseArtifactInvalid("defense evaluation index reference fields differ")
    raw = cast(dict[str, object], value)
    digest = _validate_digest(raw["sha256"])
    size = raw["size_bytes"]
    if (
        raw["media_type"] != media_type
        or raw["relative_path"] != f"{digest}/payload"
        or type(size) is not int
        or not 0 < size <= MAX_EXECUTOR_RESULT_BYTES
    ):
        raise DefenseArtifactInvalid("defense evaluation index reference is invalid")
    return ArtifactRef(digest, media_type, size, f"{digest}/payload")


class PublishedArtifact(NamedTuple):
    """One freshly verified public response payload."""

    reference: PublicArtifactReference
    payload: bytes


class DefenseEvaluationService:
    """Authenticate inputs, isolate execution, and expose durable publication only."""

    __slots__ = (
        "_artifact_store",
        "_defender_verifier",
        "_evaluator_verifier",
        "_executor",
        "_hidden_proof_verifier",
        "_index",
        "_lock",
        "_publication_signer",
        "_publication_verifier",
    )

    def __init__(
        self,
        *,
        artifact_store: ArtifactStore,
        publication_signer: RunSigningIdentity,
        publication_verifier: PublicArtifactVerifier,
        evaluator_verifier: EvaluatorReplayVerifier,
        hidden_proof_verifier: EvaluatorReplayVerifier,
        defender_verifier: DefenderBundleVerifier,
        executor: DefenseEvaluationExecutor,
    ) -> None:
        if type(artifact_store) is not ArtifactStore:
            raise TypeError("defense service requires an exact ArtifactStore")
        if (
            type(publication_signer) is not RunSigningIdentity
            or type(publication_verifier) is not PublicArtifactVerifier
        ):
            raise TypeError("defense service requires exact publication authorities")
        if (
            type(evaluator_verifier) is not EvaluatorReplayVerifier
            or type(hidden_proof_verifier) is not EvaluatorReplayVerifier
        ):
            raise TypeError("defense service requires exact evaluator authorities")
        if (
            type(defender_verifier) is not DefenderBundleVerifier
            or type(executor) is not EvaluationExecutor
        ):
            raise TypeError("defense service dependencies must be sealed exact capabilities")
        identities = {
            publication_verifier.key_id,
            evaluator_verifier.key_id,
            hidden_proof_verifier.key_id,
        }
        if len(identities) != 3 or publication_signer.key_id != publication_verifier.key_id:
            raise TypeError("defense service requires three independent pinned identities")
        self._artifact_store = artifact_store
        self._publication_signer = publication_signer
        self._publication_verifier = publication_verifier
        self._evaluator_verifier = evaluator_verifier
        self._hidden_proof_verifier = hidden_proof_verifier
        self._defender_verifier = defender_verifier
        self._executor = executor
        self._lock = threading.RLock()
        self._index = _IndexRepository(
            root=artifact_store.validated_worker_root(),
            signer=publication_signer,
            verifier=publication_verifier,
        )
        with self._index.locked():
            self._load_records(validate_publications=True)

    def create(
        self, *, corpus_artifact_digest: str, defender_artifact_digest: str
    ) -> DefenseScorecard:
        """Resolve authenticated refs, execute once, and publish one atomic pointer."""
        corpus_digest = _validate_digest(corpus_artifact_digest)
        defender_digest = _validate_digest(defender_artifact_digest)
        key = _input_key(corpus_digest, defender_digest)
        with self._lock, self._index.locked():
            records = self._load_records(validate_publications=True)
            existing = records.get(key)
            if existing is not None:
                return self._load_record(existing)[1]
            if len(records) >= MAX_EVALUATIONS:
                raise DefenseExecutionConflict("evaluation capacity is exhausted")
            corpus_ref = self._resolve(corpus_digest)
            defender_ref = self._resolve(defender_digest)
            try:
                verified_inputs = verify_evaluation_inputs(
                    corpus_ref=corpus_ref,
                    defender_ref=defender_ref,
                    artifact_store=self._artifact_store,
                    evaluator_verifier=self._evaluator_verifier,
                    defender_verifier=self._defender_verifier,
                )
            except PublicationInputError as error:
                raise DefenseArtifactInvalid("evaluation inputs failed authentication") from error
            request = self._executor.execute(
                verified_inputs,
                artifact_root=self._artifact_store.validated_worker_root(),
                evaluator_verifier=self._evaluator_verifier,
            )
            try:
                scorecard, bundle = publish_scorecard(
                    request,
                    verified_inputs=verified_inputs,
                    artifact_store=self._artifact_store,
                    signer=self._publication_signer,
                    publication_verifier=self._publication_verifier,
                    evaluator_verifier=self._evaluator_verifier,
                    hidden_proof_verifier=self._hidden_proof_verifier,
                )
                receipt_ref = store_restricted_publication_receipt(
                    request,
                    verified_inputs=verified_inputs,
                    scorecard=scorecard,
                    bundle=bundle,
                    artifact_store=self._artifact_store,
                    signer=self._publication_signer,
                )
            except ReportingContractError as error:
                raise DefenseExecutionConflict(
                    "defense publication gates rejected evidence"
                ) from error
            record = _IndexRecord(
                key,
                corpus_digest,
                defender_digest,
                scorecard.evaluation_id,
                bundle.bundle_ref(),
                receipt_ref,
            )
            self._validate_record(record)
            self._index.publish(record)
            return self._load_record(record)[1]

    def get(self, evaluation_id: str) -> DefenseScorecard:
        digest = _validate_digest(evaluation_id)
        with self._lock, self._index.locked():
            records = self._load_records(validate_publications=True)
            by_evaluation = {item.evaluation_id: item for item in records.values()}
            record = by_evaluation.get(digest)
            if record is None:
                raise DefenseResourceNotFound("defense evaluation not found")
            return self._load_record(record)[1]

    def get_artifact(self, evaluation_id: str, name: str) -> PublishedArtifact:
        if type(name) is not str or name not in {
            *PUBLIC_ARTIFACT_MEDIA_TYPES,
            SCORECARD_ARTIFACT_NAME,
        }:
            raise DefenseResourceNotFound("public artifact not found")
        digest = _validate_digest(evaluation_id)
        with self._lock, self._index.locked():
            records = self._load_records(validate_publications=True)
            by_evaluation = {item.evaluation_id: item for item in records.values()}
            record = by_evaluation.get(digest)
            if record is None:
                raise DefenseResourceNotFound("defense evaluation not found")
            bundle, _scorecard = self._load_record(record)
            reference = bundle.public_artifacts[name]
            try:
                payload = self._artifact_store.read(reference.as_artifact_ref())
            except (TypeError, ValueError) as error:
                raise DefenseArtifactInvalid(
                    "public artifact failed integrity validation"
                ) from error
            return PublishedArtifact(reference, payload)

    def _resolve(self, digest: str) -> ArtifactRef:
        try:
            return self._artifact_store.resolve(digest)
        except ValueError as error:
            raise DefenseResourceNotFound("evaluation input artifact not found") from error

    def _load_records(self, *, validate_publications: bool = False) -> dict[str, _IndexRecord]:
        records = self._index.all()
        if len({record.evaluation_id for record in records}) != len(records):
            raise DefenseArtifactInvalid("defense evaluation index is ambiguous")
        output = {record.input_key: record for record in records}
        if validate_publications:
            for record in records:
                self._validate_record(record)
        return output

    def _validate_record(self, record: _IndexRecord) -> None:
        bundle, scorecard = self._load_record(record)
        if (
            bundle.evaluation_id != record.evaluation_id
            or scorecard.evaluation_id != record.evaluation_id
        ):
            raise DefenseArtifactInvalid("evaluation index cross-link differs")
        try:
            receipt_payload = self._artifact_store.read(record.receipt_ref)
            receipt = strict_json_loads(receipt_payload)
        except (TypeError, ValueError, WireContractError) as error:
            raise DefenseArtifactInvalid("restricted publication receipt is invalid") from error
        if type(receipt) is not dict or canonical_json_bytes(receipt) != receipt_payload:
            raise DefenseArtifactInvalid("restricted publication receipt is invalid")
        document = cast(dict[str, object], receipt)
        signed = {key: value for key, value in document.items() if key != "receipt_digest"}
        unsigned = {key: value for key, value in signed.items() if key != "signature_base64"}
        signature = signed.get("signature_base64")
        public_ref = document.get("public_bundle_ref")
        expected_fields = {
            "schema_version",
            "privacy_classification",
            "evaluation_id",
            "corpus_attestation_ref",
            "corpus_evidence_ref",
            "corpus_content_digest",
            "split_digest",
            "defender_attestation",
            "promotion_envelope_digest",
            "champion_decision_digest",
            "threshold_set_digest",
            "threshold_set",
            "metric_lineage",
            "public_bundle_digest",
            "public_bundle_ref",
            "signer_key_id",
            "public_key_base64",
            "signature_base64",
            "receipt_digest",
        }
        if (
            set(document) != expected_fields
            or document.get("evaluation_id") != record.evaluation_id
            or document.get("signer_key_id") != self._publication_verifier.key_id
            or document.get("public_key_base64") != self._publication_verifier.public_key_base64
            or type(signature) is not str
            or not self._publication_verifier.verify(unsigned, signature)
            or document.get("receipt_digest") != _digest_document(signed)
            or type(public_ref) is not dict
            or cast(dict[str, object], public_ref).get("sha256") != record.bundle_ref.sha256
            or document.get("public_bundle_digest") != bundle.bundle_digest
        ):
            raise DefenseArtifactInvalid("restricted publication receipt cross-link differs")
        try:
            corpus_ref = self._artifact_store.resolve(record.corpus_digest)
            defender_ref = self._artifact_store.resolve(record.defender_digest)
            verified = verify_evaluation_inputs(
                corpus_ref=corpus_ref,
                defender_ref=defender_ref,
                artifact_store=self._artifact_store,
                evaluator_verifier=self._evaluator_verifier,
                defender_verifier=self._defender_verifier,
            )
        except (PublicationInputError, TypeError, ValueError) as error:
            raise DefenseArtifactInvalid(
                "original evaluation inputs failed reauthentication"
            ) from error
        corpus_attestation_ref = document.get("corpus_attestation_ref")
        corpus_evidence_ref = document.get("corpus_evidence_ref")
        defender_attestation = document.get("defender_attestation")
        threshold_document = document.get("threshold_set")
        try:
            if type(threshold_document) is not dict or type(
                threshold_document.get("reports")
            ) is not list:
                raise ValueError("threshold receipt fields differ")
            threshold_document = dict(threshold_document)
            threshold_document["reports"] = tuple(threshold_document["reports"])
            threshold_set = ReplayThresholdSet.model_validate(threshold_document)
        except (TypeError, ValueError) as error:
            raise DefenseArtifactInvalid(
                "restricted threshold receipt is invalid"
            ) from error
        if (
            type(corpus_attestation_ref) is not dict
            or cast(dict[str, object], corpus_attestation_ref).get("sha256")
            != record.corpus_digest
            or type(corpus_evidence_ref) is not dict
            or cast(dict[str, object], corpus_evidence_ref).get("sha256")
            != verified.corpus.evidence_ref.sha256
            or document.get("corpus_content_digest") != verified.corpus.corpus_digest
            or document.get("split_digest") != verified.corpus.split_digest
            or type(defender_attestation) is not dict
            or canonical_json_bytes(defender_attestation) != verified.defender.to_json()
            or verified.defender.top_ref.sha256 != record.defender_digest
            or document.get("threshold_set_digest")
            != threshold_set.threshold_set_digest
            or threshold_set.bundle_manifest_digest != record.defender_digest
            or threshold_set.reports[-1].report != verified.thresholds
        ):
            raise DefenseArtifactInvalid(
                "restricted publication input lineage differs"
            )

    def _load_record(
        self, record: _IndexRecord
    ) -> tuple[EvaluationArtifactBundle, DefenseScorecard]:
        try:
            bundle = load_evaluation_bundle(
                record.bundle_ref,
                artifact_store=self._artifact_store,
                verifier=self._publication_verifier,
            )
            scorecard = bundle.scorecard(
                artifact_store=self._artifact_store,
                verifier=self._publication_verifier,
            )
        except ReportingContractError as error:
            raise DefenseArtifactInvalid("published defense artifacts are invalid") from error
        return bundle, scorecard


def _digest_document(document: object) -> str:
    return hashlib.sha256(canonical_json_bytes(document)).hexdigest()


def _validate_digest(value: object) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or value != value.lower()
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise DefenseResourceNotFound("resource not found")
    return value


__all__ = [
    "DefenseArtifactInvalid",
    "DefenseEvaluationExecutor",
    "DefenseEvaluationService",
    "DefenseExecutionConflict",
    "DefenseResourceNotFound",
    "DefenseServiceError",
    "DefenseServiceUnavailable",
    "EvaluationExecutor",
    "INDEX_MEDIA_TYPE",
    "MAX_EVALUATIONS",
    "MAX_EXECUTION_SECONDS",
    "PublishedArtifact",
]
