"""Readiness verifier tamper regression tests."""

from __future__ import annotations

import copy
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
VERIFIER = ROOT / "scripts" / "verify_defense_v5_readiness.py"


def _run_verifier(document: dict) -> tuple[int, str]:
    tmp = Path("/tmp/v5-tamper-test.json")
    tmp.write_text(json.dumps(document))
    proc = subprocess.run(
        [sys.executable, str(VERIFIER), str(tmp)],
        capture_output=True, text=True, timeout=30,
    )
    return proc.returncode, proc.stdout + proc.stderr


@pytest.fixture(scope="module")
def valid_result() -> dict:
    return {
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
                "pr_auc": 0.4,
                "roc_auc": 0.5,
                "false_decline_rate": 0.0,
                "challenge_rate": 0.0,
                "captured_value_fraction": 0.0,
                "escaped_value_fraction": 1.0,
                "support_total": 100,
                "support_fraud": 30,
                "support_legitimate": 70,
            }
        },
    }


class TestVerifierTamper:
    def test_valid_result_passes(self, valid_result: dict) -> None:
        code, output = _run_verifier(valid_result)
        assert code == 0, f"valid result rejected: {output}"

    def test_forbidden_status_rejected(self, valid_result: dict) -> None:
        tampered = copy.deepcopy(valid_result)
        tampered["status"] = "winner"
        code, _ = _run_verifier(tampered)
        assert code != 0, "forbidden status accepted"

    def test_nan_metric_rejected(self, valid_result: dict) -> None:
        tampered = copy.deepcopy(valid_result)
        tampered["arms"]["full_sentinel"]["recall"] = float("nan")
        code, _ = _run_verifier(tampered)
        assert code != 0, "NaN recall accepted"

    def test_infinity_metric_rejected(self, valid_result: dict) -> None:
        tampered = copy.deepcopy(valid_result)
        tampered["arms"]["full_sentinel"]["roc_auc"] = float("inf")
        code, _ = _run_verifier(tampered)
        assert code != 0, "infinity ROC-AUC accepted"

    def test_missing_families_rejected(self, valid_result: dict) -> None:
        tampered = copy.deepcopy(valid_result)
        del tampered["arms"]["full_sentinel"]["recall"]
        code, _ = _run_verifier(tampered)
        assert code != 0, "missing recall accepted"

    def test_failed_fidelity_with_ready_rejected(self, valid_result: dict) -> None:
        tampered = copy.deepcopy(valid_result)
        tampered["status"] = "development_ready"
        tampered["failed_gates"] = []
        tampered["fidelity_status"] = "fail"
        code, _ = _run_verifier(tampered)
        assert code != 0, "ready verdict with failed fidelity accepted"

    def test_missing_economics_is_valid_only_as_an_explicit_not_ready_gate(
        self, valid_result: dict
    ) -> None:
        incomplete = copy.deepcopy(valid_result)
        incomplete["failed_gates"] = ["economics_missing"]
        incomplete["arms"]["full_sentinel"]["captured_value_fraction"] = None
        incomplete["arms"]["full_sentinel"]["escaped_value_fraction"] = None
        code, output = _run_verifier(incomplete)
        assert code == 0, output

        incomplete["status"] = "development_ready"
        incomplete["failed_gates"] = []
        code, _ = _run_verifier(incomplete)
        assert code != 0
