"""Verifier forgery regression tests."""

from __future__ import annotations

import copy
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
VERIFIER = ROOT / "scripts" / "verify_defense_v5_readiness.py"


def _run(document: dict) -> int:
    tmp = Path("/tmp/v5-forgery-test.json")
    tmp.write_text(json.dumps(document, allow_nan=True))
    return subprocess.run(
        [sys.executable, str(VERIFIER), str(tmp)],
        capture_output=True, text=True, timeout=30,
    ).returncode


_BASE = {
    "status": "development_not_ready",
    "profile": "production",
    "protocol_sha256": "a" * 64,
    "corpus_sha256": "b" * 64,
    "fidelity_status": "pass",
    "failed_gates": ["family_recall_min"],
    "arms": {
        "full_sentinel": {
            "arm": "full_sentinel",
            "recall": 0.0,
            "false_decline_rate": 0.0,
            "challenge_rate": 0.0,
            "captured_value_fraction": 0.0,
            "support_total": 100,
            "support_fraud": 30,
            "support_legitimate": 70,
            "p50_latency_ms": None,
            "p95_latency_ms": None,
            "p99_latency_ms": None,
        }
    },
}


class TestVerifierForgery:
    def test_ready_with_null_latency_rejected(self) -> None:
        tampered = copy.deepcopy(_BASE)
        tampered["status"] = "development_ready"
        tampered["failed_gates"] = []
        assert _run(tampered) != 0, "ready with null latency accepted"

    def test_ready_with_missing_arm_rejected(self) -> None:
        tampered = copy.deepcopy(_BASE)
        tampered["status"] = "development_ready"
        tampered["failed_gates"] = []
        tampered["arms"] = {}
        assert _run(tampered) != 0, "ready with no arms accepted"

    def test_ready_with_failed_fidelity_rejected(self) -> None:
        tampered = copy.deepcopy(_BASE)
        tampered["status"] = "development_ready"
        tampered["failed_gates"] = []
        tampered["fidelity_status"] = "fail"
        assert _run(tampered) != 0, "ready with failed fidelity accepted"
