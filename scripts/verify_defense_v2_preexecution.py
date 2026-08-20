"""Render the public Defend v2 pre-execution status without starting an evaluation."""

from __future__ import annotations

import argparse
from pathlib import Path

from apar.evaluation.v2_preexecution import verify_v2_preexecution
from apar.evaluation.v2_preregistration import V2Preregistration
from apar.runs.wire import canonical_json_bytes

_PUBLIC_PREREGISTRATION_PATH = Path("config/defense/competition-v2-preregistration.json")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--preregistration", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    path = (
        root / _PUBLIC_PREREGISTRATION_PATH
        if args.preregistration is None
        else args.preregistration
    )
    payload = path.read_bytes()
    preregistration = V2Preregistration.from_json(
        payload[:-1] if payload.endswith(b"\n") else payload
    )
    report = verify_v2_preexecution(root, preregistration)
    print(canonical_json_bytes(report.model_dump(mode="json")).decode("utf-8"))
    return 0 if report.admissible else 1


if __name__ == "__main__":
    raise SystemExit(main())
