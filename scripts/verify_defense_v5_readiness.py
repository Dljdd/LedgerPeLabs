"""Independently verify a Sentinel v5 development result artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

_VALID_STATUSES = {"development_ready", "development_not_ready", "invalid_corpus"}
_FORBIDDEN_CLAIMS = {"winner", "production_ready", "competition_validated"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_path", type=Path)
    args = parser.parse_args()

    document = json.loads(args.result_path.read_bytes())
    status = document.get("status", "")

    if status == "smoke":
        print("VERIFIED: smoke evidence; not production-ready")
        return 0

    if status not in _VALID_STATUSES:
        print(f"INVALID: status '{status}' is not recognized", file=__import__("sys").stderr)
        return 1

    if any(claim in str(document).lower() for claim in _FORBIDDEN_CLAIMS):
        print("INVALID: forbidden claim detected", file=__import__("sys").stderr)
        return 1

    print(f"VERIFIED: {status}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
