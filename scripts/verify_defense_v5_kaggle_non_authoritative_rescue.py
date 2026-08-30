#!/usr/bin/env python3
"""Verify and summarize a downloaded non-authoritative Sentinel v5 rescue."""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Sequence
from pathlib import Path

from apar.evaluation.v5_rescue_verifier import verify_v5_rescue_artifacts


def _canonical(document: object) -> bytes:
    return json.dumps(
        document,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode()


def _write_atomic(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    report = verify_v5_rescue_artifacts(arguments.artifact_root)
    payload = _canonical(report)
    if arguments.report is not None:
        _write_atomic(arguments.report, payload)
    print(payload.decode())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["main"]
