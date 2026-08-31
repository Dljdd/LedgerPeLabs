"""Command-line interface for deterministic APAR submission releases."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from scripts.submission.archive import build_archive, verify_archive
from scripts.submission.clean_room import run_clean_room
from scripts.submission.model import ReleaseError


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build and verify the APAR submission archive.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build", help="build an allowlist-only deterministic archive")
    build.add_argument("--repo", type=Path, default=Path.cwd())
    build.add_argument(
        "--policy",
        type=Path,
        default=Path("scripts/submission/submission-policy.json"),
    )
    build.add_argument("--output", type=Path, required=True)
    build.add_argument("--include-web", action="store_true")
    verify = subparsers.add_parser("verify-archive", help="verify ZIP metadata and manifest")
    verify.add_argument("--archive", type=Path, required=True)
    clean_room = subparsers.add_parser(
        "clean-room", help="extract, install the exact lock, and replay the portable model"
    )
    clean_room.add_argument("--archive", type=Path, required=True)
    clean_room.add_argument("--python", default=sys.executable)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    response: dict[str, object]
    try:
        if args.command == "build":
            result = build_archive(
                args.repo,
                args.policy,
                args.output,
                include_web=args.include_web,
            )
            verify_archive(args.output)
            response = {
                "archive": result.archive_path,
                "archive_sha256": result.archive_sha256,
                "deterministic_core_sha256": result.deterministic_core_sha256,
                "source_commit": result.source_commit,
                "source_tree": result.source_tree,
            }
        elif args.command == "verify-archive":
            manifest = verify_archive(args.archive)
            response = {
                "archive": str(args.archive.resolve()),
                "deterministic_core_sha256": manifest["deterministic_core_sha256"],
                "source": manifest["source"],
                "verified": True,
            }
        else:
            response = run_clean_room(args.archive, python_executable=args.python)
    except ReleaseError as error:
        print(f"submission release failed: {error}", file=sys.stderr)
        return 2
    print(json.dumps(response, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
