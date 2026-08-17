#!/usr/bin/env python3
"""Run the deterministic production-rail and adaptive-red-team acceptance gates."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _run(test_path: str) -> None:
    completed = subprocess.run(
        [sys.executable, "-m", "pytest", test_path, "-q"],
        cwd=ROOT,
        check=False,
        text=True,
        capture_output=True,
    )
    sys.stdout.write(completed.stdout)
    sys.stderr.write(completed.stderr)
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)


def _verify_task6_postcommit() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/run_task6_holdout.py",
            "--verify-postcommit",
            "--approved-result-commit",
            "d6d3eecbfe2d871af8375e1455814cb5c48f2928",
            "--approved-result-sha256",
            "f82981a987651a7f7ebb10a9011df063b2dc54a56181cae5b838e31de5e658db",
        ],
        cwd=ROOT,
        check=False,
        text=True,
        capture_output=True,
    )
    sys.stdout.write(completed.stdout)
    sys.stderr.write(completed.stderr)
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)


def main() -> None:
    """Fail closed unless both focused suites complete successfully."""
    _run("tests/integration/test_g1_simulation.py")
    print(
        "G1 PASS: card and A2A report/recovery conserve value; "
        "agentic 23-attack matrix fails closed with 2 controls"
    )
    _run("tests/integration/test_g2_adaptation.py")
    _verify_task6_postcommit()
    print(
        "G2 PASS: 4 hidden families; fixed/random/adaptive/cached-LLM matched budgets; "
        "boolean-only validity; byte-identical seeded reruns; exact Task6 historical "
        "postcommit verification-only recomputation"
    )


if __name__ == "__main__":
    main()
