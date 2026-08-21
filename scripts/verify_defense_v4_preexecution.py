"""Render the public Defend v4 pre-execution status without starting an evaluation."""

from __future__ import annotations

import argparse
from pathlib import Path

from apar.evaluation.v4_preexecution import verify_v4_preexecution
from apar.runs.wire import canonical_json_bytes


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    root = args.root.resolve()
    report = verify_v4_preexecution(root)
    print(canonical_json_bytes(report.model_dump(mode="json")).decode("utf-8"))
    return 0 if report.admissible else 1


if __name__ == "__main__":
    raise SystemExit(main())
