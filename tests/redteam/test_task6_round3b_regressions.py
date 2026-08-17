"""Task 6 round-three Phase B authority, publication, and freeze regressions."""

from __future__ import annotations

import copy
import json
import sys
from dataclasses import fields
from pathlib import Path
from types import MethodType

import pytest

import scripts.run_task6_holdout as holdout_runner
from apar.redteam import (
    AdaptiveSearch,
    AdaptiveTournamentPolicy,
    AttackCandidate,
    FixedPolicy,
    LLMPlannerPolicy,
    PolicyCapability,
    RandomPolicy,
    SearchAuthority,
)
from apar.redteam.task6_experiment import build_task6_experiment
from tests.redteam.test_round1_regressions import _tiny_bounds
from tests.redteam.test_task6_round3_regressions import _registered

ROOT = Path(__file__).resolve().parents[2]
V3_PREREGISTRATION = ROOT / "docs/experiments/task6-v3-holdout-preregistration.json"
V3_CANCELLATION = ROOT / "docs/experiments/task6-v3-cancellation.json"
V31_PREREGISTRATION = ROOT / "docs/experiments/task6-v3.1-holdout-preregistration.json"
V3_RESULT = ROOT / "docs/experiments/task6-v3-holdout-result.json"
V31_RESULT = ROOT / "docs/experiments/task6-v3.1-holdout-result.json"


def _changed_candidate(history, bounds) -> AttackCandidate:  # type: ignore[no-untyped-def]
    changed = next(
        vector
        for vector in bounds.feasible_vectors
        if vector.fingerprint != bounds.defaults.fingerprint
    )
    return AttackCandidate(
        params=changed,
        parent_id=None if not history else history[-1].candidate.candidate_id,
        generation=len(history),
    )


class _HelperPolicy:
    def _select(self, history, bounds):  # type: ignore[no-untyped-def]
        return AttackCandidate(
            params=bounds.defaults,
            parent_id=None if not history else history[-1].candidate.candidate_id,
            generation=len(history),
        )

    def propose(self, history, bounds, _rng):  # type: ignore[no-untyped-def]
        return self._select(history, bounds)


def _global_select(history, bounds) -> AttackCandidate:  # type: ignore[no-untyped-def]
    return AttackCandidate(
        params=bounds.defaults,
        parent_id=None if not history else history[-1].candidate.candidate_id,
        generation=len(history),
    )


class _GlobalPolicy:
    def propose(self, history, bounds, _rng):  # type: ignore[no-untyped-def]
        return _global_select(history, bounds)


def test_policy_capability_is_only_an_opaque_nonce_and_coupled_tamper_rejects() -> None:
    authority = SearchAuthority()
    original = authority.register_policy(
        FixedPolicy(), name="fixed", version="opaque-v1"
    )
    replacement = authority.register_policy(
        AdaptiveTournamentPolicy(), name="adaptive", version="opaque-v1"
    )

    assert tuple(field.name for field in fields(PolicyCapability)) == ("capability_id",)
    assert not hasattr(original, "_propose")
    assert not hasattr(original, "_policy")
    assert not hasattr(original, "policy_code_digest")
    object.__setattr__(original, "capability_id", replacement.capability_id)

    with pytest.raises(ValueError, match="issued|opaque|capability"):
        authority.policy_binding(original)


def test_copied_synthetic_subclass_and_cross_authority_policy_handles_reject() -> None:
    first = SearchAuthority()
    second = SearchAuthority()
    issued = first.register_policy(FixedPolicy(), name="fixed", version="opaque-v1")
    foreign = second.register_policy(FixedPolicy(), name="fixed", version="opaque-v1")

    class _Subclass(PolicyCapability):
        pass

    for forged in (
        copy.copy(issued),
        PolicyCapability(capability_id=issued.capability_id),
        _Subclass(capability_id=issued.capability_id),
        foreign,
    ):
        with pytest.raises(ValueError, match="issued|authority|capability"):
            first.policy_binding(forged)
    assert not hasattr(issued, "model_copy")


def test_runtime_integrity_rejects_instance_helper_replacement() -> None:
    policy = _HelperPolicy()
    authority, evaluator, registered = _registered(policy)
    policy._select = MethodType(  # type: ignore[method-assign]
        lambda self, history, bounds: _changed_candidate(history, bounds),
        policy,
    )

    with pytest.raises(ValueError, match="implementation|integrity"):
        AdaptiveSearch(
            evaluator_capability=evaluator,
            policy_capability=registered,
            run_group=authority.issue_run_group("instance-helper-substitution"),
        )


def test_runtime_integrity_rejects_class_helper_replacement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy = _HelperPolicy()
    authority, evaluator, registered = _registered(policy)
    monkeypatch.setattr(
        _HelperPolicy,
        "_select",
        lambda self, history, bounds: _changed_candidate(history, bounds),
    )

    with pytest.raises(ValueError, match="implementation|integrity"):
        AdaptiveSearch(
            evaluator_capability=evaluator,
            policy_capability=registered,
            run_group=authority.issue_run_group("class-helper-substitution"),
        )


def test_runtime_integrity_rejects_module_global_helper_replacement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy = _GlobalPolicy()
    authority, evaluator, registered = _registered(policy)
    monkeypatch.setattr(
        sys.modules[__name__],
        "_global_select",
        lambda history, bounds: _changed_candidate(history, bounds),
    )

    with pytest.raises(ValueError, match="implementation|integrity"):
        AdaptiveSearch(
            evaluator_capability=evaluator,
            policy_capability=registered,
            run_group=authority.issue_run_group("global-helper-substitution"),
        )


def test_llm_mutable_audit_and_cache_state_preserve_registered_integrity() -> None:
    class _Client:
        provider = "fixture"
        model_id = "runtime-integrity-v1"

        def __init__(self) -> None:
            self.calls = 0

        def complete(self, request):  # type: ignore[no-untyped-def]
            self.calls += 1
            history = request["history"]
            return {
                "output": {
                    "params": _tiny_bounds().defaults_document(),
                    "parent_id": None if not history else history[-1]["candidate_id"],
                    "generation": len(history),
                },
                "latency_ms": 0,
                "input_tokens": 1,
                "output_tokens": 1,
            }

    policy = LLMPlannerPolicy(_Client())
    authority, evaluator, registered = _registered(policy)
    result = AdaptiveSearch(
        evaluator_capability=evaluator,
        policy_capability=registered,
        run_group=authority.issue_run_group("llm-mutable-state"),
    ).search(seed=2, budget=2, wall_time_budget_ms=1_000)

    assert result.proposals_used == 2
    assert len(policy.take_audit_records()) == 2


def test_llm_client_identity_and_transport_callable_substitution_reject() -> None:
    class _Client:
        provider = "fixture"
        model_id = "pinned-client-v1"

        def complete(self, request):  # type: ignore[no-untyped-def]
            history = request["history"]
            return {
                "output": {
                    "params": _tiny_bounds().defaults_document(),
                    "parent_id": None if not history else history[-1]["candidate_id"],
                    "generation": len(history),
                },
                "latency_ms": 0,
                "input_tokens": 1,
                "output_tokens": 1,
            }

    client = _Client()
    policy = LLMPlannerPolicy(client)
    authority, evaluator, registered = _registered(policy)
    client.provider = "forged"  # type: ignore[misc]
    client.complete = MethodType(  # type: ignore[method-assign]
        lambda self, request: _Client.complete(self, request),
        client,
    )

    with pytest.raises(ValueError, match="implementation|integrity"):
        AdaptiveSearch(
            evaluator_capability=evaluator,
            policy_capability=registered,
            run_group=authority.issue_run_group("llm-client-substitution"),
        )


def test_atomic_publication_never_overwrites_a_racing_result(tmp_path: Path) -> None:
    publisher = getattr(holdout_runner, "_atomic_publish_result", None)
    assert callable(publisher), "runner must expose atomic exclusive publication"
    target = tmp_path / "result.json"
    sentinel = b'{"winner":"external"}\n'

    def create_racing_result() -> None:
        target.write_bytes(sentinel)

    with pytest.raises(FileExistsError):
        publisher(target, b'{"winner":"runner"}\n', before_publish=create_racing_result)

    assert target.read_bytes() == sentinel
    assert tuple(path for path in tmp_path.iterdir() if path != target) == ()


def test_atomic_publication_is_exact_and_exclusive(tmp_path: Path) -> None:
    publisher = getattr(holdout_runner, "_atomic_publish_result", None)
    assert callable(publisher), "runner must expose atomic exclusive publication"
    target = tmp_path / "result.json"
    expected = b'{"supported_family_count":1}\n'

    publisher(target, expected)

    assert target.read_bytes() == expected
    with pytest.raises(FileExistsError):
        publisher(target, b"replacement")
    assert target.read_bytes() == expected


def test_confirmatory_negative_control_is_run_with_matched_budgets_and_excluded() -> None:
    experiment = build_task6_experiment(ROOT)
    negative_benchmark = getattr(experiment, "negative_control", None)
    runner = getattr(holdout_runner, "_run_negative_control", None)
    assert negative_benchmark is not None, "experiment must expose the frozen control"
    assert callable(runner), "runner must execute the frozen negative-control cell"
    authority = SearchAuthority()
    evaluator = negative_benchmark.issue_evaluator_capability(authority)
    group = authority.issue_run_group("negative-control-regression")
    policies = {
        "fixed": authority.register_policy(FixedPolicy(), name="fixed", version="1"),
        "random": authority.register_policy(RandomPolicy(), name="random", version="1"),
        "adaptive": authority.register_policy(
            AdaptiveTournamentPolicy(), name="adaptive", version="3.0.0"
        ),
    }

    document = runner(
        authority=authority,
        run_group=group,
        evaluator=evaluator,
        policies=policies,
        seeds=(1, 2),
        budget=2,
        wall_time_budget_ms=60_000,
    )

    assert document["family"] == "agentic_intent_abuse"
    assert document["included_in_supported_family_count"] is False
    assert document["matched_budgets"] is True
    assert document["network_call_count"] == 0
    assert document["observed_valid_yield_delta"] == "0"
    assert document["supported"] is False


def test_v3_is_cancelled_and_distinct_v31_freeze_remains_unexecuted() -> None:
    assert V3_PREREGISTRATION.exists()
    cancellation = json.loads(V3_CANCELLATION.read_text(encoding="utf-8"))
    frozen = json.loads(V31_PREREGISTRATION.read_text(encoding="utf-8"))

    assert cancellation["cancelled_preregistration_commit"] == (
        "cbeaeea3e0a98a86cb22673a42abd652a2d586a9"
    )
    assert cancellation["status"] == "cancelled_before_execution"
    assert cancellation["result_created"] is False
    assert cancellation["seeds_used"] is False
    assert frozen["supersedes_cancelled_commit"] == cancellation[
        "cancelled_preregistration_commit"
    ]
    assert frozen["holdout"]["seeds"] == [503, 607, 709, 811, 907, 1009, 1103, 1201]
    assert frozen["holdout"]["result_path"] == (
        "docs/experiments/task6-v3.1-holdout-result.json"
    )
    assert frozen["negative_control"]["confirmatory_execution_required"] is True
    assert frozen["negative_control"]["included_in_supported_family_count"] is False
    assert frozen["negative_control"]["seeds"] == "same_as_target_holdout"
    assert not V3_RESULT.exists()
    assert not V31_RESULT.exists()
