"""Independently verify one serialized Sentinel v5 evidence envelope offline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from apar.v5_independent_verifier import (
    IndependentVerificationError,
    verify_evidence_bytes,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evidence", type=Path)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    try:
        report = verify_evidence_bytes(
            args.evidence.read_bytes(), root=args.root.resolve()
        )
    except (IndependentVerificationError, OSError, ValueError) as error:
        print(
            json.dumps(
                {"verified": False, "error": str(error)},
                sort_keys=True,
                separators=(",", ":"),
            ),
            file=sys.stderr,
        )
        return 1
    print(json.dumps(report, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
