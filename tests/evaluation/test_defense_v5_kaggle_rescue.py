"""Regression tests for the explicitly non-authoritative Kaggle rescue."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "scripts/run_defense_v5_kaggle_non_authoritative_rescue.py"
NOTEBOOK_BUILDER = ROOT / "scripts/build_defense_v5_kaggle_non_authoritative_rescue_notebook.py"


def _load_runner_module() -> object:
    spec = importlib.util.spec_from_file_location("apar_v5_rescue_runner_test", RUNNER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_rescue_runner_import_keeps_the_coordinator_memory_lean() -> None:
    """Importing the coordinator must not load arm-evaluation dependencies."""
    probe = """
import importlib.util
import json
import sys

runner = sys.argv[1]
spec = importlib.util.spec_from_file_location("apar_v5_rescue_runner_probe", runner)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
heavy_modules = [
    name
    for name in (
        "multiprocessing",
        "numpy",
        "scipy",
        "apar.evaluation.v5_metrics",
        "apar.evaluation.v5_staged_evidence",
    )
    if name in sys.modules
]
print(json.dumps(heavy_modules, sort_keys=True))
"""
    completed = subprocess.run(
        [sys.executable, "-c", probe, str(RUNNER)],
        check=True,
        capture_output=True,
        text=True,
    )

    assert json.loads(completed.stdout) == []


def test_rescue_action_executes_the_bound_runner_in_a_fresh_interpreter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Each heavy rescue action must cross an exec boundary, never a fork boundary."""
    runner = _load_runner_module()
    observed: list[tuple[list[str], bool, dict[str, str], bool]] = []

    def fake_run(
        command: list[str],
        *,
        check: bool,
        env: dict[str, str],
        text: bool,
    ) -> subprocess.CompletedProcess[str]:
        observed.append((command, check, env, text))
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(runner.subprocess, "run", fake_run)
    environment = {"APAR_V5_RESCUE_TEST": "bound"}

    runner._run_fresh_action(action="arm", environment=environment)

    assert observed == [
        (
            [sys.executable, str(RUNNER), "--internal-action", "arm"],
            True,
            environment,
            True,
        )
    ]


def test_compact_arm_receipt_is_self_bound_without_a_metric_payload(tmp_path: Path) -> None:
    """Coordinator validation must consume only the compact receipt."""
    runner = _load_runner_module()
    deterministic_result_sha256 = "1" * 64
    metric_core = {
        "arm": "rules_only",
        "deterministic_result_sha256": deterministic_result_sha256,
        "support_sha256": "2" * 64,
        "aggregate": {},
        "calibration_sha256": "3" * 64,
        "economics_sha256": "4" * 64,
        "family_sha256": [],
        "bootstrap_sha256": "5" * 64,
    }
    metric_core["deterministic_complete_metrics_sha256"] = runner._sha256_bytes(
        runner._canonical(metric_core)
    )
    metric_observation = {
        "schema_version": "apar-sentinel-v5-non-authoritative-compact-arm-metric/1",
        "arm": "rules_only",
        "support_sha256": "2" * 64,
        "complete_metrics_sha256": "6" * 64,
    }
    metric_observation["compact_observation_sha256"] = runner._sha256_bytes(
        runner._canonical(metric_observation)
    )
    receipt = {
        "schema_version": "apar-sentinel-v5-non-authoritative-compact-arm-receipt/1",
        "authoritative": False,
        "accepted_capacity_evidence": False,
        "arm": "rules_only",
        "deterministic_result_sha256": deterministic_result_sha256,
        "metric_core": metric_core,
        "metric_observation": metric_observation,
        "readiness_bundle": None,
    }
    receipt["receipt_sha256"] = runner._sha256_bytes(runner._canonical(receipt))
    receipt_path = tmp_path / "00-rules_only.json"
    receipt_path.write_bytes(runner._canonical(receipt))

    assert (
        runner._load_bound_arm_receipt(
            path=receipt_path,
            expected_arm="rules_only",
        )
        == receipt
    )


def test_rescue_notebook_uses_the_bound_capacity_validation_mode() -> None:
    """The notebook launcher must pass the exact mode accepted by the runner."""
    builder_source = NOTEBOOK_BUILDER.read_text()

    assert 'execution_mode="kaggle_capacity_validation"' in builder_source
    assert 'execution_mode="capacity_validation"' not in builder_source


def test_rescue_worker_uses_the_frozen_arms_stage_directory() -> None:
    """Arm restoration must resolve the protocol's Stage 30 checkpoint directory."""
    runner_source = RUNNER.read_text()

    assert "checkpoint_root=predecessor_chain / V5KaggleStage.ARMS.value" in runner_source
    assert 'checkpoint_root=predecessor_chain / "20_arms"' not in runner_source
