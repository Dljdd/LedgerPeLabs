"""Disposable, public-only attacker-policy worker entry point.

The parent always starts this file with Python isolated mode.  Every request is one-shot:
the process reconstructs public bounds/history, emits one candidate, and exits.
"""

from __future__ import annotations

import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import ModuleType

# ``python -I`` intentionally omits the source checkout.  Add only the reviewed package
# root, never a caller-controlled path or environment value.
_SOURCE_ROOT = str(Path(__file__).resolve().parents[2])
if sys.path[0:1] != [_SOURCE_ROOT]:
    sys.path.insert(0, _SOURCE_ROOT)

import builtins  # noqa: E402
import socket  # noqa: E402
from typing import cast  # noqa: E402

import numpy as np  # noqa: E402
from numpy.random import default_rng as _default_rng  # noqa: E402

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


def _is_forbidden(name: object) -> bool:
    return type(name) is str and any(
        name == prefix or name.startswith(f"{prefix}.") for prefix in _FORBIDDEN_IMPORTS
    )


def _audit(event: str, args: tuple[object, ...]) -> None:
    if event == "open":
        raise PermissionError("policy worker filesystem capability denied")
    if event in _NETWORK_EVENTS or event.startswith("subprocess") or event in {
        "os.posix_spawn",
        "os.spawn",
        "os.system",
        "pty.spawn",
    }:
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


class _CachedFixturePolicy:
    """Deterministic zero-network cached-planner fixture for local orchestration."""

    policy_name = "cached_llm"
    policy_version = "1.0.0"

    def propose(
        self,
        history: tuple[VisibleTrial, ...],
        bounds: ParameterBounds,
        rng: np.random.Generator,
    ) -> AttackCandidate:
        del rng
        visible = tuple(history)
        vector = bounds.feasible_vectors[len(visible) % len(bounds.feasible_vectors)]
        return AttackCandidate(
            params=vector,
            parent_id=None if not visible else visible[-1].candidate.candidate_id,
            generation=len(visible),
        )


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
    filesystem_blocked = False
    reflection_import_blocked = False
    network_blocked = False
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
    orchestrator_absent = "apar.runs.runner" not in sys.modules
    return {
        "clean_start": hidden_absent and orchestrator_absent,
        "filesystem_blocked": filesystem_blocked,
        "forbidden_import_blocked": forbidden_import_blocked,
        "hidden_modules_absent": hidden_absent,
        "input_hidden_fields_absent": set(request) == {"operation", "schema_version"},
        "network_blocked": network_blocked,
        "orchestrator_modules_absent": orchestrator_absent,
        "reflection_import_blocked": reflection_import_blocked,
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
            document = _object(
                request,
                {"bounds", "history", "operation", "policy_kind", "schema_version", "seed"},
                label="proposal request",
            )
            if document["schema_version"] != "1.0.0" or document["operation"] != "propose":
                raise ValueError("worker request version or operation is unsupported")
            bounds = bounds_from_wire(document["bounds"])
            history = history_from_wire(document["history"], bounds)
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
            elif policy_kind == "cached_llm":
                policy = _CachedFixturePolicy()
            elif policy_kind == "fixed":
                policy = FixedPolicy()
            else:
                policy = RandomPolicy()
            candidate = policy.propose(history, bounds, _default_rng(seed))
            response = {"candidate": candidate_to_wire(candidate), "ok": True}
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
