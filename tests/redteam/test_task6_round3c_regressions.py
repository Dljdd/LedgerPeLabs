"""Task 6 round-three Phase C dependency, durability, and claim regressions."""

from __future__ import annotations

import errno
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest

import apar.redteam.policies as policy_module
import scripts.run_task6_holdout as holdout_runner
from apar.redteam import AdaptiveSearch, AdaptiveTournamentPolicy, AttackCandidate
from tests.redteam.test_task6_round3_regressions import _registered

ROOT = Path(__file__).resolve().parents[2]
V31_CANCELLATION = ROOT / "docs/experiments/task6-v3.1-cancellation.json"
V31_RESULT = ROOT / "docs/experiments/task6-v3.1-holdout-result.json"
V32_PREREGISTRATION = ROOT / "docs/experiments/task6-v3.2-holdout-preregistration.json"
V32_RESULT = ROOT / "docs/experiments/task6-v3.2-holdout-result.json"


def _candidate(history, bounds, *, changed: bool) -> AttackCandidate:  # type: ignore[no-untyped-def]
    vector = bounds.defaults
    if changed:
        vector = next(
            candidate
            for candidate in bounds.feasible_vectors
            if candidate.fingerprint != bounds.defaults.fingerprint
        )
    return AttackCandidate(
        params=vector,
        parent_id=None if not history else history[-1].candidate.candidate_id,
        generation=len(history),
    )


def test_same_named_module_replacement_rejects_before_policy_behavior(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy = AdaptiveTournamentPolicy()
    authority, evaluator, registered = _registered(policy)
    replacement = ModuleType("math")

    def raising(_value):  # type: ignore[no-untyped-def]
        raise AssertionError("replacement module behavior executed")

    replacement.sqrt = raising  # type: ignore[attr-defined]
    replacement.log = raising  # type: ignore[attr-defined]
    monkeypatch.setattr(policy_module, "math", replacement)

    with pytest.raises(ValueError, match="module|implementation|integrity"):
        AdaptiveSearch(
            evaluator_capability=evaluator,
            policy_capability=registered,
            run_group=authority.issue_run_group("module-object-replacement"),
        )


def test_same_module_attribute_mutation_rejects_before_policy_behavior(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy = AdaptiveTournamentPolicy()
    authority, evaluator, registered = _registered(policy)

    def raising(_value):  # type: ignore[no-untyped-def]
        raise AssertionError("mutated math.sqrt executed")

    monkeypatch.setattr(math, "sqrt", raising)

    with pytest.raises(ValueError, match="module|implementation|integrity"):
        AdaptiveSearch(
            evaluator_capability=evaluator,
            policy_capability=registered,
            run_group=authority.issue_run_group("module-attribute-mutation"),
        )


def test_closure_cell_mutation_rejects_before_policy_behavior() -> None:
    def initial_selector(history, bounds):  # type: ignore[no-untyped-def]
        return _candidate(history, bounds, changed=False)

    selector = [initial_selector]

    class _ClosurePolicy:
        def propose(self, history, bounds, _rng):  # type: ignore[no-untyped-def]
            return selector[0](history, bounds)

    policy = _ClosurePolicy()
    authority, evaluator, registered = _registered(policy)

    def replacement(history, bounds):  # type: ignore[no-untyped-def]
        return _candidate(history, bounds, changed=True)

    selector[0] = replacement

    with pytest.raises(ValueError, match="closure|implementation|integrity"):
        AdaptiveSearch(
            evaluator_capability=evaluator,
            policy_capability=registered,
            run_group=authority.issue_run_group("closure-mutation"),
        )


def test_callable_default_mutation_rejects_before_policy_behavior(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def default_selector(history, bounds):  # type: ignore[no-untyped-def]
        return _candidate(history, bounds, changed=False)

    class _DefaultPolicy:
        def propose(  # type: ignore[no-untyped-def]
            self,
            history,
            bounds,
            _rng,
            selector=default_selector,
        ):
            return selector(history, bounds)

    policy = _DefaultPolicy()
    authority, evaluator, registered = _registered(policy)
    def replacement(history, bounds):  # type: ignore[no-untyped-def]
        return _candidate(history, bounds, changed=True)

    monkeypatch.setattr(_DefaultPolicy.propose, "__defaults__", (replacement,))

    with pytest.raises(ValueError, match="default|implementation|integrity"):
        AdaptiveSearch(
            evaluator_capability=evaluator,
            policy_capability=registered,
            run_group=authority.issue_run_group("default-mutation"),
        )


def test_callable_object_state_mutation_rejects_before_policy_behavior() -> None:
    class _Selector:
        def __init__(self) -> None:
            self.changed = False

        def __call__(self, history, bounds):  # type: ignore[no-untyped-def]
            return _candidate(history, bounds, changed=self.changed)

    class _CallablePolicy:
        def __init__(self) -> None:
            self.selector = _Selector()

        def propose(self, history, bounds, _rng):  # type: ignore[no-untyped-def]
            return self.selector(history, bounds)

    policy = _CallablePolicy()
    authority, evaluator, registered = _registered(policy)
    policy.selector.changed = True

    with pytest.raises(ValueError, match="state|implementation|integrity"):
        AdaptiveSearch(
            evaluator_capability=evaluator,
            policy_capability=registered,
            run_group=authority.issue_run_group("callable-state-mutation"),
        )


def test_directory_fsync_eio_reports_published_recovery_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    error_type = getattr(holdout_runner, "ResultPublicationDurabilityError", RuntimeError)
    original_fsync = holdout_runner.os.fsync
    calls = 0

    def fail_directory_fsync(descriptor: int) -> None:
        nonlocal calls
        calls += 1
        if calls == 1:
            original_fsync(descriptor)
            return
        raise OSError(errno.EIO, "injected directory durability failure")

    monkeypatch.setattr(holdout_runner.os, "fsync", fail_directory_fsync)
    target = tmp_path / "result.json"
    payload = b'{"result":"complete"}\n'

    with pytest.raises(error_type) as captured:
        holdout_runner._atomic_publish_result(target, payload)

    assert target.read_bytes() == payload
    assert getattr(captured.value, "target_published", False) is True
    assert "inspect the existing result" in str(captured.value)
    assert tuple(path for path in tmp_path.iterdir() if path != target) == ()


def test_directory_fsync_explicit_unsupported_error_is_tolerated(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_fsync = holdout_runner.os.fsync
    calls = 0

    def unsupported_directory_fsync(descriptor: int) -> None:
        nonlocal calls
        calls += 1
        if calls == 1:
            original_fsync(descriptor)
            return
        raise OSError(errno.EINVAL, "directory fsync unsupported")

    monkeypatch.setattr(holdout_runner.os, "fsync", unsupported_directory_fsync)
    target = tmp_path / "result.json"
    payload = b'{"result":"complete"}\n'

    holdout_runner._atomic_publish_result(target, payload)

    assert target.read_bytes() == payload


def _negative_control(**updates: object) -> dict[str, object]:
    document: dict[str, object] = {
        "matched_budgets": True,
        "network_call_count": 0,
        "observed_valid_yield_delta": "0",
        "supported": False,
    }
    document.update(updates)
    return document


@pytest.mark.parametrize(
    ("gate_updates", "control_updates"),
    (
        ({"target_cells_bound": False}, {}),
        ({"target_matched_budgets": False}, {}),
        ({"target_network_call_count": 1}, {}),
        ({}, {"matched_budgets": False}),
        ({}, {"network_call_count": 1}),
        ({}, {"observed_valid_yield_delta": "0.1"}),
        ({}, {"supported": True}),
    ),
)
def test_invalid_confirmatory_control_suppresses_every_support_claim(
    gate_updates: dict[str, object],
    control_updates: dict[str, object],
) -> None:
    gate = getattr(holdout_runner, "_confirmatory_gate", None)
    assert callable(gate), "runner must derive claims through a confirmatory hard gate"
    arguments: dict[str, object] = {
        "target_cells_bound": True,
        "target_matched_budgets": True,
        "target_network_call_count": 0,
        "target_supported_family_count": 2,
        "target_adaptive_claim": "supported",
        "negative_control": _negative_control(**control_updates),
    }
    arguments.update(gate_updates)

    claim = gate(**arguments)

    assert claim == {
        "confirmatory_valid": False,
        "criterion_met": False,
        "supported_family_count": 0,
        "adaptive_claim": "not_supported",
    }


def test_valid_confirmatory_control_preserves_preregistered_support_claim() -> None:
    gate = getattr(holdout_runner, "_confirmatory_gate", None)
    assert callable(gate), "runner must derive claims through a confirmatory hard gate"

    claim = gate(
        target_cells_bound=True,
        target_matched_budgets=True,
        target_network_call_count=0,
        target_supported_family_count=2,
        target_adaptive_claim="supported",
        negative_control=_negative_control(),
    )

    assert claim == {
        "confirmatory_valid": True,
        "criterion_met": True,
        "supported_family_count": 2,
        "adaptive_claim": "supported",
    }


def test_v31_is_canonically_cancelled_without_execution_or_result() -> None:
    assert V31_CANCELLATION.exists()
    cancellation = json.loads(V31_CANCELLATION.read_text(encoding="utf-8"))

    assert cancellation["status"] == "cancelled_before_execution"
    assert cancellation["cancelled_preregistration_commit"] == (
        "6ad59cdf178f23be8c7ad4a54a7b565e9bc5bf39"
    )
    assert cancellation["confirmatory_execution_invoked"] is False
    assert cancellation["seeds_used"] is False
    assert cancellation["result_created"] is False
    assert cancellation["replacement_preregistration_path"] == (
        "docs/experiments/task6-v3.2-holdout-preregistration.json"
    )
    assert not V31_RESULT.exists()
    assert not V32_RESULT.exists()


def test_v32_source_only_verification_is_direct_and_executes_no_search() -> None:
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/run_task6_holdout.py"),
            "--verify-source-only",
        ],
        cwd=ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert "v3.2 source" in completed.stdout
    assert "no holdout trial executed" in completed.stdout
    assert not V32_RESULT.exists()


def test_v32_behavior_manifest_covers_every_tracked_executable_input() -> None:
    head = holdout_runner._head_commit()
    freeze = holdout_runner._source_freeze_document(head)
    entries = freeze["behavior_manifest"]["entries"]
    tracked = set(holdout_runner._tracked_paths(head))
    expected = (
        {path for path in tracked if path.startswith("src/")}
        | {path for path in tracked if path.startswith("scripts/")}
        | {path for path in tracked if path.startswith("fixtures/")}
        | {
            "pyproject.toml",
            "docs/experiments/task6-v3-cached-llm-replay.json",
            "docs/experiments/task6-v3-cancellation.json",
        }
    )
    if "docs/experiments/task6-v3.1-cancellation.json" in tracked:
        expected.add("docs/experiments/task6-v3.1-cancellation.json")
    if "docs/experiments/task6-v3.2-cancellation.json" in tracked:
        expected.add("docs/experiments/task6-v3.2-cancellation.json")

    assert set(entries) == expected
    assert "scripts/run_task6_holdout.py" in entries
    assert "pyproject.toml" in entries
    assert "docs/experiments/task6-v3-cached-llm-replay.json" in entries
    assert freeze["lock_file"] == {"path": None, "status": "explicitly_absent"}
    assert freeze["git_tree"]
    assert freeze["behavior_manifest"]["digest"]
    installed = freeze["environment"]["installed_distributions"]
    assert len(installed) >= 10
    assert freeze["environment"]["installed_distributions_digest"] == (
        holdout_runner._canonical_digest(installed)
    )


def test_v32_preregistration_is_absent_until_separate_freeze_commit() -> None:
    if not V32_PREREGISTRATION.exists():
        assert not V32_RESULT.exists()
        return
    artifact = json.loads(V32_PREREGISTRATION.read_text(encoding="utf-8"))
    source = artifact["source_freeze"]
    assert artifact["status"] == "final_v3_2_frozen_before_confirmatory_execution"
    assert source["source_commit"]
    assert source["source_commit"] != holdout_runner._head_commit()
    assert artifact["holdout"]["seeds"] == [503, 607, 709, 811, 907, 1009, 1103, 1201]
    assert artifact["holdout"]["budget"] == 24
    assert artifact["holdout"]["maximum_additional_confirmatory_attempts"] == 1
    assert artifact["negative_control"]["confirmatory_execution_required"] is True
    assert not V32_RESULT.exists()
