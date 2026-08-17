"""Disposable, public-only attacker-policy worker entry point.

The parent always starts this file with Python isolated mode.  Every request is one-shot:
the process reconstructs public bounds/history, emits one candidate, and exits.
"""

from __future__ import annotations

import resource
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import ModuleType


def _set_limit(kind: int, maximum: int) -> None:
    _, hard = resource.getrlimit(kind)
    bounded = maximum if hard == resource.RLIM_INFINITY else min(maximum, hard)
    resource.setrlimit(kind, (bounded, bounded))


def _install_resource_limits() -> None:
    """Install child-owned hard limits before loading package or numeric code."""
    _set_limit(resource.RLIMIT_CORE, 0)
    _set_limit(resource.RLIMIT_CPU, 3)
    _set_limit(resource.RLIMIT_FSIZE, 1_048_576)
    _set_limit(resource.RLIMIT_NOFILE, 64)
    if hasattr(resource, "RLIMIT_NPROC"):
        _set_limit(resource.RLIMIT_NPROC, 1)


_install_resource_limits()
if sys.argv[1:] == ["startup-hang-probe"]:
    while True:
        pass
if sys.argv[1:]:
    raise SystemExit(2)

# ``python -I`` intentionally omits the source checkout.  Add only the reviewed package
# root, never a caller-controlled path or environment value.
_SOURCE_ROOT = str(Path(__file__).resolve().parents[2])
if sys.path[0:1] != [_SOURCE_ROOT]:
    sys.path.insert(0, _SOURCE_ROOT)

import builtins  # noqa: E402
import ctypes as _native_probe  # noqa: E402
import os as _os_probe  # noqa: E402
import socket  # noqa: E402
from typing import cast  # noqa: E402

from numpy.random import default_rng as _default_rng  # noqa: E402

from apar.redteam.llm_policy import LLMPlannerPolicy  # noqa: E402
from apar.redteam.policies import (  # noqa: E402
    AdaptiveTournamentPolicy,
    AttackCandidate,
    FixedPolicy,
    ParameterBounds,
    Policy,
    RandomPolicy,
    VisibleTrial,
)
from apar.runs.wire import (  # noqa: E402
    bounds_from_wire,
    candidate_to_wire,
    canonical_json_bytes,
    feedback_fields_from_wire,
    history_from_wire,
    strict_json_loads,
)

_FORBIDDEN_IMPORTS = (
    "apar.defense",
    "apar.evaluation_hidden",
    "apar.generators",
    "apar.redteam.benchmark",
    "apar.simulator",
    "apar.trust",
    "ctypes",
    "gc",
    "inspect",
    "multiprocessing",
    "resource",
    "subprocess",
)
_NETWORK_EVENTS = frozenset(
    {
        "socket.__new__",
        "socket.bind",
        "socket.connect",
        "socket.getaddrinfo",
        "socket.gethostbyaddr",
        "socket.gethostbyname",
        "socket.gethostname",
        "socket.sendto",
    }
)
_PROCESS_EVENTS = frozenset(
    {
        "os.kill",
        "os.killpg",
        "os.system",
        "pty.spawn",
    }
)
_PROCESS_PREFIXES = ("os.exec", "os.fork", "os.posix_spawn", "os.spawn", "subprocess")


def _is_forbidden(name: object) -> bool:
    return type(name) is str and any(
        name == prefix or name.startswith(f"{prefix}.") for prefix in _FORBIDDEN_IMPORTS
    )


def _audit(event: str, args: tuple[object, ...]) -> None:
    if event == "open":
        raise PermissionError("policy worker filesystem capability denied")
    if event in _NETWORK_EVENTS or event in _PROCESS_EVENTS or event.startswith(
        _PROCESS_PREFIXES
    ):
        raise PermissionError("policy worker capability denied")
    if event == "import" and args and _is_forbidden(args[0]):
        raise ImportError("policy worker import denied")
    if event.startswith("ctypes."):
        raise PermissionError("policy worker native loading denied")


sys.addaudithook(_audit)
_ORIGINAL_IMPORT = builtins.__import__


def _restricted_import(
    name: str,
    globals: Mapping[str, object] | None = None,  # noqa: A002
    locals: Mapping[str, object] | None = None,  # noqa: A002
    fromlist: Sequence[str] | None = (),
    level: int = 0,
) -> ModuleType:
    if _is_forbidden(name):
        raise ImportError("policy worker import denied")
    return _ORIGINAL_IMPORT(name, globals, locals, fromlist, level)


builtins.__import__ = _restricted_import


class _NoNetworkClient:
    """A transport sentinel proving cache-only replay never reaches completion."""

    provider = "fixture"
    model_id = "cached-default-v1"

    def complete(self, _request: dict[str, object]) -> dict[str, object]:
        raise AssertionError("cache-only planner attempted a network transport")


def _cached_llm_proposal(
    history: tuple[VisibleTrial, ...],
    bounds: ParameterBounds,
    replay_cache: object,
) -> AttackCandidate:
    """Replay the exact parent-pinned Task6 cache with no transport fallback."""
    if type(replay_cache) is not dict or any(
        type(key) is not str or type(value) is not dict
        for key, value in replay_cache.items()
    ):
        raise ValueError("frozen replay cache is invalid")
    cached = LLMPlannerPolicy(
        _NoNetworkClient(),
        replay_cache=cast(dict[str, dict[str, object]], replay_cache),
        require_cached_replay=True,
        clock_ns=lambda: 0,
    )
    candidate = cached.propose(history, bounds)
    records = cached.take_audit_records()
    if len(records) != 1 or not records[0].cache_hit:
        raise RuntimeError("cache-only planner did not produce one authenticated cache hit")
    return candidate


def _object(value: object, fields: set[str], *, label: str) -> dict[str, object]:
    if type(value) is not dict or set(value) != fields:
        raise ValueError(f"{label} field set is not exact")
    return cast(dict[str, object], value)


def _integer(value: object, *, label: str, maximum: int) -> int:
    if type(value) is not int or not 0 <= value <= maximum:
        raise ValueError(f"{label} is invalid")
    return value


def _probe(request: dict[str, object]) -> dict[str, object]:
    if request["operation"] == "probe_hang":
        while True:
            pass
    hidden_absent = not any(
        _is_forbidden(name)
        for name in tuple(sys.modules)
        if name.startswith("apar.")
    )
    forbidden_import_blocked = False
    exec_blocked = False
    filesystem_blocked = False
    fork_blocked = False
    native_blocked = False
    process_signal_blocked = False
    reflection_import_blocked = False
    network_blocked = False
    spawn_blocked = False
    try:
        builtins.__import__("apar.evaluation_hidden")
    except (ImportError, PermissionError):
        forbidden_import_blocked = True
    try:
        builtins.__import__("inspect")
    except (ImportError, PermissionError):
        reflection_import_blocked = True
    try:
        socket.socket()
    except PermissionError:
        network_blocked = True
    try:
        open(__file__, "rb")  # noqa: PTH123, SIM115
    except PermissionError:
        filesystem_blocked = True
    try:
        _os_probe.execv("/definitely/missing/apar-worker", ("apar-worker",))
    except PermissionError:
        exec_blocked = True
    except OSError:
        pass
    try:
        child = _os_probe.fork()
    except PermissionError:
        fork_blocked = True
    except OSError:
        pass
    else:
        if child == 0:
            _os_probe._exit(97)
        _os_probe.waitpid(child, 0)
    try:
        _os_probe.posix_spawn(
            "/definitely/missing/apar-worker", ("apar-worker",), {}
        )
    except PermissionError:
        spawn_blocked = True
    except OSError:
        pass
    try:
        _os_probe.kill(_os_probe.getpid(), 0)
    except PermissionError:
        process_signal_blocked = True
    try:
        _native_probe.CDLL(None)
    except PermissionError:
        native_blocked = True
    orchestrator_absent = "apar.runs.runner" not in sys.modules
    return {
        "clean_start": hidden_absent and orchestrator_absent,
        "exec_blocked": exec_blocked,
        "filesystem_blocked": filesystem_blocked,
        "fork_blocked": fork_blocked,
        "forbidden_import_blocked": forbidden_import_blocked,
        "hidden_modules_absent": hidden_absent,
        "input_hidden_fields_absent": set(request) == {"operation", "schema_version"},
        "native_blocked": native_blocked,
        "network_blocked": network_blocked,
        "orchestrator_modules_absent": orchestrator_absent,
        "process_signal_blocked": process_signal_blocked,
        "reflection_import_blocked": reflection_import_blocked,
        "spawn_blocked": spawn_blocked,
    }


def _main() -> int:
    try:
        raw = sys.stdin.buffer.read(8 * 1024 * 1024 + 1)
        if len(raw) > 8 * 1024 * 1024:
            raise ValueError("worker request exceeds the byte cap")
        loaded = strict_json_loads(raw)
        if type(loaded) is not dict:
            raise ValueError("worker request must be an object")
        request = cast(dict[str, object], loaded)
        operation = request.get("operation")
        if operation in {"probe", "probe_hang"}:
            _object(request, {"operation", "schema_version"}, label="probe request")
            response = {"ok": True, "probe": _probe(request)}
        else:
            requested_kind = request.get("policy_kind")
            proposal_fields = {
                "bounds",
                "feedback_fields",
                "history",
                "operation",
                "policy_kind",
                "schema_version",
                "seed",
            }
            if requested_kind == "cached_llm":
                proposal_fields.add("llm_replay_cache")
            document = _object(
                request,
                proposal_fields,
                label="proposal request",
            )
            if document["schema_version"] != "1.0.0" or document["operation"] != "propose":
                raise ValueError("worker request version or operation is unsupported")
            bounds = bounds_from_wire(document["bounds"])
            feedback_fields = feedback_fields_from_wire(document["feedback_fields"])
            history = history_from_wire(
                document["history"], bounds, feedback_fields=feedback_fields
            )
            seed = _integer(document["seed"], label="worker seed", maximum=2**63 - 1)
            policy_kind = document["policy_kind"]
            if type(policy_kind) is not str or policy_kind not in {
                "adaptive",
                "cached_llm",
                "fixed",
                "random",
            }:
                raise ValueError("policy kind is unsupported")
            policy: Policy
            if policy_kind == "adaptive":
                policy = AdaptiveTournamentPolicy()
            elif policy_kind == "fixed":
                policy = FixedPolicy()
            else:
                policy = RandomPolicy()
            candidate = (
                _cached_llm_proposal(history, bounds, document["llm_replay_cache"])
                if policy_kind == "cached_llm"
                else policy.propose(history, bounds, _default_rng(seed))
            )
            response = {
                "audit": {
                    "cache_hit": policy_kind == "cached_llm",
                    "cache_source": (
                        "task6-v3-frozen-replay"
                        if policy_kind == "cached_llm"
                        else None
                    ),
                    "network_call_count": 0,
                    "policy_kind": policy_kind,
                },
                "candidate": candidate_to_wire(candidate),
                "ok": True,
            }
        sys.stdout.buffer.write(canonical_json_bytes(response))
        sys.stdout.buffer.flush()
        return 0
    except BaseException as error:
        failure = {
            "error": type(error).__name__,
            "message": "policy worker failed closed",
            "ok": False,
        }
        sys.stderr.buffer.write(canonical_json_bytes(failure))
        sys.stderr.buffer.flush()
        return 2


if __name__ == "__main__":
    raise SystemExit(_main())
