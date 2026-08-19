"""One-shot isolated worker that alone resolves restricted evaluation artifacts."""

from __future__ import annotations

import resource
import sys
from pathlib import Path


def _set_limit(kind: int, maximum: int) -> None:
    _, hard = resource.getrlimit(kind)
    bounded = maximum if hard == resource.RLIM_INFINITY else min(maximum, hard)
    resource.setrlimit(kind, (bounded, bounded))


def _limits() -> None:
    _set_limit(resource.RLIMIT_CORE, 0)
    _set_limit(resource.RLIMIT_CPU, 15)
    _set_limit(resource.RLIMIT_FSIZE, 1_048_576)
    _set_limit(resource.RLIMIT_NOFILE, 64)
    if hasattr(resource, "RLIMIT_NPROC"):
        _set_limit(resource.RLIMIT_NPROC, 1)


_limits()
if sys.argv[1:]:
    raise SystemExit(2)
_SOURCE_ROOT = str(Path(__file__).resolve().parents[2])
if sys.path[0:1] != [_SOURCE_ROOT]:
    sys.path.insert(0, _SOURCE_ROOT)

import os  # noqa: E402

from apar.evaluation_hidden.authority_core import _isolated_worker_main  # noqa: E402
from apar.runs.wire import canonical_json_bytes, strict_json_loads  # noqa: E402


def _audit(event: str, args: tuple[object, ...]) -> None:
    del args
    if event.startswith(("socket.", "subprocess", "os.exec", "os.fork", "os.spawn")):
        raise PermissionError("hidden evaluator capability denied")
    if event in {"os.system", "pty.spawn"}:
        raise PermissionError("hidden evaluator capability denied")


sys.addaudithook(_audit)


def _main() -> int:
    try:
        raw = sys.stdin.buffer.read(32_000_001)
        if not raw or len(raw) > 32_000_000:
            raise ValueError("hidden worker request violates byte cap")
        document = strict_json_loads(raw)
        if type(document) is not dict or canonical_json_bytes(document) != raw:
            raise ValueError("hidden worker request is not canonical")
        response = _isolated_worker_main(document)
        output = canonical_json_bytes(response)
        if len(output) > 32_000_000:
            raise ValueError("hidden worker output violates byte cap")
        sys.stdout.buffer.write(output)
        sys.stdout.buffer.flush()
        return 0
    except BaseException:
        # Never include exception text, artifact identifiers, or payload fragments.
        os.write(2, b'{"error":"hidden evaluator failed closed"}')
        return 2


if __name__ == "__main__":
    raise SystemExit(_main())
