#!/usr/bin/env python3
"""Run the hash-bound, demo-only Sentinel v5 portable model on accepted rows."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from apar.demo.sentinel_v5_portable import run_portable_scenarios  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=("Verify and run the accepted-checkpoint Sentinel v5 portable demo model.")
    )
    parser.add_argument(
        "--scenario",
        type=Path,
        required=True,
        help="Path to the hash-bound scenarios.json file.",
    )
    parser.add_argument(
        "--bundle",
        type=Path,
        help="Bundle root; defaults to the scenario file directory.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Trace output; defaults to the operating-system temporary directory.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    scenario_path = args.scenario.resolve()
    bundle_root = (args.bundle or scenario_path.parent).resolve()
    output_path = (
        args.output.resolve()
        if args.output is not None
        else Path(tempfile.gettempdir()) / "sentinel-v5-demo-trace.json"
    )
    report = run_portable_scenarios(
        bundle_root=bundle_root,
        scenario_path=scenario_path,
        output_path=output_path,
    )
    summary = {
        "arm": report["traces"][0]["arm"],
        "bundle_manifest_sha256": report["bundle_manifest_sha256"],
        "metrics": report["metrics"],
        "model_load_ms": report["model_load_ms"],
        "replay_verified": report["replay_verified"],
        "scenario_count": len(report["traces"]),
        "scoring_wall_ms": report["scoring_wall_ms"],
        "trace_path": str(output_path),
        "trace_sha256": report["trace_sha256"],
    }
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
