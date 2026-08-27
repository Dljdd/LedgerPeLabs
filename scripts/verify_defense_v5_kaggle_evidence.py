#!/usr/bin/env python3
"""Independently verify a completed Sentinel v5 Kaggle prefix or full chain."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from apar.v5_kaggle_independent_verifier import (  # noqa: E402
    V5KaggleIndependentVerificationError,
    V5KagglePrefixVerificationReport,
    V5KaggleVerificationReport,
    verify_v5_kaggle_evidence,
    verify_v5_kaggle_prefix,
)

_STAGES = (
    "00_authorize",
    "10_corpus",
    "20_features",
    "30_arms",
    "40_label_shuffle",
    "50_identity_rename",
    "51_future_causality",
    "52_equal_time_isolation",
    "53_feature_leakage",
    "60_single_class_controls",
    "70_metrics",
    "80_finalize",
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument(
        "--mode",
        required=True,
        choices=("kaggle_capacity_validation", "kaggle_locked_successor"),
    )
    parser.add_argument("--chain-root", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    report: V5KaggleVerificationReport | V5KagglePrefixVerificationReport
    try:
        if arguments.chain_root.is_symlink() or not arguments.chain_root.is_dir():
            raise V5KaggleIndependentVerificationError("chain root is not a directory")
        entries = {item.name for item in arguments.chain_root.iterdir()}
        if entries - set(_STAGES):
            raise V5KaggleIndependentVerificationError("chain root has unknown entries")
        present = tuple(stage for stage in _STAGES if stage in entries)
        if not present or present != _STAGES[: len(present)]:
            raise V5KaggleIndependentVerificationError("chain prefix is missing or reordered")
        roots = tuple(arguments.chain_root / stage for stage in present)
        if len(roots) == len(_STAGES):
            report = verify_v5_kaggle_evidence(
                root=arguments.root,
                checkpoint_roots=roots[:-1],
                final_root=roots[-1],
                expected_mode=arguments.mode,
            )
        else:
            report = verify_v5_kaggle_prefix(
                root=arguments.root,
                checkpoint_roots=roots,
                expected_mode=arguments.mode,
            )
    except (OSError, V5KaggleIndependentVerificationError):
        print('{"error":"verification_failed","valid":false}', file=sys.stderr)
        return 1
    print(
        json.dumps(
            report.model_dump(mode="json"),
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
