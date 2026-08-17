"""Task 6 round-four raw-evidence and frozen-behavior regressions."""

from __future__ import annotations

import hashlib
import inspect
import json
from copy import deepcopy
from pathlib import Path

import pytest

import scripts.run_task6_holdout as holdout_runner
from apar.redteam import (
    AdaptiveSearch,
    AdaptiveTournamentPolicy,
    FixedPolicy,
    LLMPlannerPolicy,
    RandomPolicy,
    SearchAuthority,
)
from apar.redteam.benchmark import default_defender_rules
from apar.redteam.policies import AttackCandidate
from apar.redteam.task6_experiment import build_task6_experiment
from apar.redteam.task6_verifier import (
    EvidenceVerificationError,
    build_result_bundle_document,
    build_search_cell_document,
    canonical_digest,
    canonical_json_bytes,
    strict_json_loads,
    verify_result_bundle,
    verify_search_cell,
)

ROOT = Path(__file__).resolve().parents[2]
V33_RESULT = ROOT / "docs/experiments/task6-v3.3-holdout-result.json"
V33_REJECTION = ROOT / "docs/experiments/task6-v3.3-postexecution-rejection.json"
V34_PREREGISTRATION = ROOT / "docs/experiments/task6-v3.4-holdout-preregistration.json"
V34_RESULT = ROOT / "docs/experiments/task6-v3.4-holdout-result.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_v33_result_is_preserved_but_canonically_rejected() -> None:
    rejection = strict_json_loads(V33_REJECTION.read_bytes(), require_canonical=True)

    assert V33_RESULT.exists()
    assert _sha256(V33_RESULT) == "78cfa7a8352b41c2f4ca34b67cde939d2e9ffdefd8f8f3f91ccb24ee1e05d7fd"
    assert rejection == {
        "schema_version": "1.0.0",
        "status": "rejected_unverifiable",
        "result_commit": "c513f263536330e7104c8b6eb1c0e5da4ccba0b4",
        "result_path": "docs/experiments/task6-v3.3-holdout-result.json",
        "result_sha256": "78cfa7a8352b41c2f4ca34b67cde939d2e9ffdefd8f8f3f91ccb24ee1e05d7fd",
        "result_preserved_byte_for_byte": True,
        "reviewer_reason": (
            "The result omits raw SearchResult documents, evaluator-owned per-trial "
            "execution traces, and individual cached-LLM audit attempts, so its "
            "aggregate capability claims cannot be independently reconstructed."
        ),
        "rejected_claims": [
            "adaptive_claim",
            "confirmatory_valid",
            "criterion_met",
            "supported_family_count",
        ],
        "replacement_preregistration_path": (
            "docs/experiments/task6-v3.4-holdout-preregistration.json"
        ),
        "replacement_result_path": "docs/experiments/task6-v3.4-holdout-result.json",
        "replacement_kind": "instrumentation_only_evidence_replication",
    }


def test_v34_protocol_is_instrumentation_only_evidence_replication() -> None:
    protocol = holdout_runner._expected_protocol()

    assert protocol["experiment_id"] == "task6-v3.4-evidence-replication"
    assert protocol["seeds"] == [2601, 2707, 2801, 2903, 3001, 3109, 3203, 3301]
    assert protocol["budgets"] == {
        "proposal": 24,
        "query": 24,
        "logical_time": 24,
        "wall_time_ms": 120000,
    }
    assert protocol["maximum_confirmatory_attempts"] == 1
    assert protocol["replication_kind"] == "instrumentation_only_evidence_replication"
    assert protocol["algorithm_retuned_after_v3_3"] is False
    assert protocol["network"] == {"allowed_calls": 0, "cached_llm_required": True}
    assert protocol["evidence_limits"] == {
        "expected_cell_count": 88,
        "trials_per_complete_cell": 24,
        "maximum_total_trials": 2112,
        "maximum_cached_llm_attempts": 384,
        "maximum_bundle_bytes": 33554432,
        "lossless": True,
    }
    assert "no further Task 6 confirmatory attempt" in protocol["stopping_rule"]
    assert not V34_RESULT.exists()


def test_policy_and_defender_behavior_equivalence_is_bound_to_v33() -> None:
    behavior_document = getattr(holdout_runner, "_behavior_equivalence_document", None)
    assert callable(behavior_document)

    document = behavior_document()

    assert document["baseline_commit"] == "c513f263536330e7104c8b6eb1c0e5da4ccba0b4"
    assert document["equivalent"] is True
    assert document["proposal_implementation_sha256"] == {
        "src/apar/redteam/llm_policy.py": (
            "8105a6788041f7d73b1afa571482f4b0ff3b15980f6c28769b701ab350936622"
        ),
        "src/apar/redteam/policies.py": (
            "c97ab7b263a493978cf901140a97f15874a34f8ff2ce54c84253e7baa998fb82"
        ),
        "src/apar/redteam/search.py": (
            "ee05348ab07a9852a68a3f6a477eeec7ad6837d94f187e6fbb97767220f60e89"
        ),
    }
    assert document["generator_implementation_sha256"] == {
        "src/apar/generators/__init__.py": (
            "c4b3cdb979f1ec154cd6d55b40317495f024be7e09f7d9b6f7a93101b99d2886"
        ),
        "src/apar/generators/campaigns.py": (
            "670b4a3ec358f82d88f9655bd41d878fbee11d4841ff264655554bae31c3b31a"
        ),
        "src/apar/generators/population.py": (
            "2e54862322980414098c17930ec95bd268372da8968a78384a5bd661bfdaa2e5"
        ),
        "src/apar/redteam/task6_experiment.py": (
            "a1367a8bb4310eeea2812a7d118ccb738ae1d9c32bfbc21c87413b1a869ce056"
        ),
    }
    assert document["generator_implementation_sha256"] == (
        document["v3_3_generator_implementation_sha256"]
    )
    assert document["defender_rules"] == [
        rule.document() for rule in default_defender_rules().rules
    ]
    assert document["defender_ast_sha256"] == document["v3_3_defender_ast_sha256"]


def test_evaluator_trace_is_lossless_and_does_not_change_feedback() -> None:
    experiment = build_task6_experiment(ROOT)
    benchmark = experiment.benchmarks["app_scam_mule"]
    control = build_task6_experiment(ROOT).benchmarks["app_scam_mule"]
    candidate = AttackCandidate(
        params=benchmark.public_bounds.defaults,
        parent_id=None,
        generation=0,
    )
    expected, _observation = control.evaluate_with_observation(candidate)

    actual = benchmark.evaluate(candidate)
    take_traces = getattr(benchmark, "take_evaluation_traces", None)
    assert callable(take_traces)
    traces = take_traces()

    assert actual == expected
    assert len(traces) == 1
    trace = traces[0].document()
    assert trace["family"] == "app_scam_mule"
    assert trace["candidate_id"] == candidate.candidate_id
    assert trace["command_digest"] != "0" * 64
    assert trace["command_count"] > 0
    assert trace["event_digest"] != "0" * 64
    assert trace["event_count"] == sum(
        item["count"] for item in trace["event_type_counts"]
    )
    assert trace["ledger_digest"] != "0" * 64
    assert trace["ledger_conserved"] is True
    assert trace["derived_feature_vector"]
    assert trace["decision"] == {
        "action": actual.action.value,
        "reason_family": actual.reason_family,
    }
    assert trace["executed_role_bound_value"] == "300.00"
    assert trace["feedback_realized_value"] == "300.00"
    assert sum(
        (int(item["outstanding_minor_units"]) for item in trace["role_bound_value_components"]),
        0,
    ) == 30000
    assert take_traces() == ()


def test_v34_result_stays_absent_while_source_or_preregistration_is_prepared() -> None:
    assert not V34_RESULT.exists()
    if V34_PREREGISTRATION.exists():
        artifact = json.loads(V34_PREREGISTRATION.read_text(encoding="utf-8"))
        assert artifact["protocol"] == holdout_runner._expected_protocol()


@pytest.fixture(scope="module")
def fixed_evidence_cell() -> dict[str, object]:
    experiment = build_task6_experiment(ROOT)
    benchmark = experiment.benchmarks["app_scam_mule"]
    authority = SearchAuthority()
    evaluator = benchmark.issue_evaluator_capability(authority)
    policy = authority.register_policy(FixedPolicy(), name="fixed", version="1.0.0")
    result = AdaptiveSearch(
        evaluator_capability=evaluator,
        policy_capability=policy,
        run_group=authority.issue_run_group("task6-v3.4-test-cell"),
    ).search(seed=4, budget=2, wall_time_budget_ms=60_000)
    traces = benchmark.take_evaluation_traces()

    return build_search_cell_document(
        cell_kind="target",
        result=result,
        public_bounds=benchmark.public_bounds,
        evaluation_contract=benchmark.evaluation_contract,
        policy_binding=authority.policy_binding(policy),
        defender=default_defender_rules(),
        evaluation_traces=traces,
        llm_audit_records=(),
    )


def _refresh_public_cell_digests(cell: dict[str, object]) -> None:
    search_result = cell["search_result"]
    assert type(search_result) is dict
    search_result["canonical_document_digest"] = canonical_digest(
        search_result["document"]
    )
    traces = cell["evaluation_traces"]
    assert type(traces) is list
    cell["evaluation_trace_digest"] = canonical_digest(traces)
    attempts = cell["llm_audit_attempts"]
    assert type(attempts) is list
    cell["llm_audit_digest"] = canonical_digest(attempts)
    core = {key: value for key, value in cell.items() if key != "cell_digest"}
    cell["cell_digest"] = canonical_digest(core)


def test_search_cell_persists_complete_public_result_and_raw_trace(
    fixed_evidence_cell: dict[str, object],
) -> None:
    cell = deepcopy(fixed_evidence_cell)
    metrics = verify_search_cell(
        cell,
        expected_cell_kind="target",
        expected_family="app_scam_mule",
        expected_policy="fixed",
        expected_seed=4,
        expected_budget=2,
        expected_wall_time_budget_ms=60_000,
    )

    search = cell["search_result"]
    assert type(search) is dict
    document = search["document"]
    assert type(document) is dict
    assert len(document["proposals"]) == len(document["trials"]) == 2
    assert len(cell["candidate_sequence"]) == 2
    assert len(cell["evaluation_traces"]) == 2
    assert cell["llm_audit_attempts"] == []
    assert search["seal_scope"] == "process_local_nonportable_hmac"
    assert metrics["proposal_count"] == 2
    assert metrics["approved_count"] == 2
    assert metrics["net_settled_value"] == "600.00"


def test_canonical_evidence_json_rejects_duplicate_keys_and_noncanonical_bytes(
    fixed_evidence_cell: dict[str, object],
) -> None:
    encoded = canonical_json_bytes(fixed_evidence_cell)
    assert encoded.endswith(b"\n")
    assert strict_json_loads(encoded) == fixed_evidence_cell

    with pytest.raises(EvidenceVerificationError, match="duplicate"):
        strict_json_loads(b'{"a":1,"a":2}\n')
    with pytest.raises(EvidenceVerificationError, match="canonical"):
        strict_json_loads(b'{ "a": 1 }\n', require_canonical=True)


def test_search_cell_rejects_objective_forgery_even_with_refreshed_public_digests(
    fixed_evidence_cell: dict[str, object],
) -> None:
    cell = deepcopy(fixed_evidence_cell)
    search = cell["search_result"]
    assert type(search) is dict
    document = search["document"]
    assert type(document) is dict
    document["objective_values"][0] = "999"
    document["trials"][0]["objective_value"] = "999"
    _refresh_public_cell_digests(cell)

    with pytest.raises(EvidenceVerificationError, match="objective"):
        verify_search_cell(cell)


def test_search_cell_rejects_budget_and_provenance_forgery(
    fixed_evidence_cell: dict[str, object],
) -> None:
    budget = deepcopy(fixed_evidence_cell)
    budget_search = budget["search_result"]
    assert type(budget_search) is dict
    budget_document = budget_search["document"]
    assert type(budget_document) is dict
    budget_document["proposals_used"] = 1
    _refresh_public_cell_digests(budget)
    with pytest.raises(EvidenceVerificationError, match="budget|usage|proposal"):
        verify_search_cell(budget)

    provenance = deepcopy(fixed_evidence_cell)
    provenance_search = provenance["search_result"]
    assert type(provenance_search) is dict
    provenance_document = provenance_search["document"]
    assert type(provenance_document) is dict
    provenance_document["defender_digest"] = "0" * 64
    _refresh_public_cell_digests(provenance)
    with pytest.raises(EvidenceVerificationError, match="defender|provenance"):
        verify_search_cell(provenance)


def test_search_cell_rejects_trace_forgery_even_with_refreshed_public_digests(
    fixed_evidence_cell: dict[str, object],
) -> None:
    cell = deepcopy(fixed_evidence_cell)
    traces = cell["evaluation_traces"]
    assert type(traces) is list
    trace = traces[0]
    assert type(trace) is dict
    trace["decision"]["reason_family"] = "velocity"
    trace_core = {
        key: value for key, value in trace.items() if key not in {"trial_index", "trace_digest"}
    }
    trace["trace_digest"] = canonical_digest(trace_core)
    _refresh_public_cell_digests(cell)

    with pytest.raises(EvidenceVerificationError, match="trace|decision|feedback|rule"):
        verify_search_cell(cell)


@pytest.fixture(scope="module")
def cached_llm_evidence_cell() -> tuple[dict[str, object], dict[str, object]]:
    cache = json.loads(
        (ROOT / "docs/experiments/task6-v3-cached-llm-replay.json").read_text(
            encoding="utf-8"
        )
    )
    experiment = build_task6_experiment(ROOT)
    benchmark = experiment.benchmarks["app_scam_mule"]
    authority = SearchAuthority()
    evaluator = benchmark.issue_evaluator_capability(authority)
    client = holdout_runner._NoNetworkClient()
    planner = LLMPlannerPolicy(
        client,
        replay_cache=cache["records"],
        require_cached_replay=True,
    )
    policy = authority.register_policy(
        planner,
        name="cached_llm",
        version="1.0.0",
    )
    result = AdaptiveSearch(
        evaluator_capability=evaluator,
        policy_capability=policy,
        run_group=authority.issue_run_group("task6-v3.4-test-cached-cell"),
    ).search(seed=4, budget=2, wall_time_budget_ms=60_000)
    assert client.calls == 0

    cell = build_search_cell_document(
        cell_kind="target",
        result=result,
        public_bounds=benchmark.public_bounds,
        evaluation_contract=benchmark.evaluation_contract,
        policy_binding=authority.policy_binding(policy),
        defender=default_defender_rules(),
        evaluation_traces=benchmark.take_evaluation_traces(),
        llm_audit_records=planner.take_audit_records(),
    )
    return cell, cache


def test_cached_llm_attempts_are_individually_bound_to_cache_and_trials(
    cached_llm_evidence_cell: tuple[dict[str, object], dict[str, object]],
) -> None:
    cell, cache = deepcopy(cached_llm_evidence_cell)

    verify_search_cell(
        cell,
        expected_cell_kind="target",
        expected_family="app_scam_mule",
        expected_policy="cached_llm",
        expected_seed=4,
        expected_budget=2,
        expected_wall_time_budget_ms=60_000,
        expected_llm_cache=cache["records"],
    )
    attempts = cell["llm_audit_attempts"]
    assert type(attempts) is list
    assert len(attempts) == 2
    assert all(attempt["call_status"] == "cache_success" for attempt in attempts)
    assert all(attempt["cache_hit"] is True for attempt in attempts)


def test_cached_llm_digest_forgery_is_rejected_against_frozen_cache(
    cached_llm_evidence_cell: tuple[dict[str, object], dict[str, object]],
) -> None:
    cell, cache = deepcopy(cached_llm_evidence_cell)
    attempts = cell["llm_audit_attempts"]
    assert type(attempts) is list
    attempt = attempts[0]
    assert type(attempt) is dict
    attempt["prompt_digest"] = "0" * 64
    core = {
        key: value
        for key, value in attempt.items()
        if key not in {"attempt_index", "trial_index", "record_digest"}
    }
    attempt["record_digest"] = canonical_digest(core)
    _refresh_public_cell_digests(cell)

    with pytest.raises(EvidenceVerificationError, match="cache|prompt|LLM"):
        verify_search_cell(cell, expected_llm_cache=cache["records"])


@pytest.fixture(scope="module")
def miniature_result_bundle() -> tuple[
    dict[str, object],
    dict[str, object],
    dict[str, object],
    dict[str, object],
    dict[str, object],
]:
    cache = json.loads(
        (ROOT / "docs/experiments/task6-v3-cached-llm-replay.json").read_text(
            encoding="utf-8"
        )
    )
    protocol = deepcopy(holdout_runner._expected_protocol())
    protocol["seeds"] = [4]
    protocol["budgets"] = {
        "proposal": 2,
        "query": 2,
        "logical_time": 2,
        "wall_time_ms": 60_000,
    }
    protocol["evidence_limits"] = {
        "expected_cell_count": 11,
        "trials_per_complete_cell": 2,
        "maximum_total_trials": 22,
        "maximum_cached_llm_attempts": 4,
        "maximum_bundle_bytes": 8_388_608,
        "lossless": True,
    }
    experiment = build_task6_experiment(ROOT)
    authority = SearchAuthority()
    group = authority.issue_run_group("task6-v3.4-miniature-verifier-fixture")
    evaluators = {
        family: benchmark.issue_evaluator_capability(authority)
        for family, benchmark in experiment.benchmarks.items()
    }
    negative_evaluator = experiment.negative_control.issue_evaluator_capability(authority)
    client = holdout_runner._NoNetworkClient()
    planner = LLMPlannerPolicy(
        client,
        replay_cache=cache["records"],
        require_cached_replay=True,
    )
    policy_objects = {
        "fixed": FixedPolicy(),
        "random": RandomPolicy(),
        "adaptive": AdaptiveTournamentPolicy(),
        "cached_llm": planner,
    }
    policies = {
        name: authority.register_policy(
            policy,
            name=name,
            version=protocol["policies"][name],
        )
        for name, policy in policy_objects.items()
    }
    bindings = {
        name: authority.policy_binding(policy).model_dump(mode="json")
        for name, policy in policies.items()
    }
    cells: list[dict[str, object]] = []
    contexts: dict[str, object] = {"targets": {}, "negative_control": None}
    for family in sorted(evaluators):
        benchmark = experiment.benchmarks[family]
        for name in ("fixed", "random", "adaptive", "cached_llm"):
            result = AdaptiveSearch(
                evaluator_capability=evaluators[family],
                policy_capability=policies[name],
                run_group=group,
            ).search(seed=4, budget=2, wall_time_budget_ms=60_000)
            cell = build_search_cell_document(
                cell_kind="target",
                result=result,
                public_bounds=benchmark.public_bounds,
                evaluation_contract=benchmark.evaluation_contract,
                policy_binding=authority.policy_binding(policies[name]),
                defender=default_defender_rules(),
                evaluation_traces=benchmark.take_evaluation_traces(),
                llm_audit_records=(
                    planner.take_audit_records() if name == "cached_llm" else ()
                ),
            )
            public_context = deepcopy(cell["public_context"])
            assert type(public_context) is dict
            public_context.pop("policy_binding")
            target_contexts = contexts["targets"]
            assert type(target_contexts) is dict
            target_contexts.setdefault(family, public_context)
            cells.append(cell)
    negative = experiment.negative_control
    for name in ("fixed", "random", "adaptive"):
        result = AdaptiveSearch(
            evaluator_capability=negative_evaluator,
            policy_capability=policies[name],
            run_group=group,
        ).search(seed=4, budget=2, wall_time_budget_ms=60_000)
        cell = build_search_cell_document(
            cell_kind="negative_control",
            result=result,
            public_bounds=negative.public_bounds,
            evaluation_contract=negative.evaluation_contract,
            policy_binding=authority.policy_binding(policies[name]),
            defender=default_defender_rules(),
            evaluation_traces=negative.take_evaluation_traces(),
            llm_audit_records=(),
        )
        public_context = deepcopy(cell["public_context"])
        assert type(public_context) is dict
        public_context.pop("policy_binding")
        contexts["negative_control"] = public_context
        cells.append(cell)
    assert client.calls == 0
    external_approval = {
        "approved_freeze_commit": "a" * 40,
        "approved_prereg_sha256": "b" * 64,
    }
    bundle = build_result_bundle_document(
        protocol=protocol,
        cells=cells,
        expected_contexts=contexts,
        expected_policy_bindings=bindings,
        expected_llm_cache=cache["records"],
        external_approval=external_approval,
        preregistration_canonical_digest="c" * 64,
        network_call_count=client.calls,
    )
    return bundle, protocol, contexts, bindings, cache["records"]


def _refresh_bundle_digest(bundle: dict[str, object]) -> None:
    core = {key: value for key, value in bundle.items() if key != "bundle_digest"}
    bundle["bundle_digest"] = canonical_digest(core)


def test_independent_bundle_verifier_recomputes_all_claims_from_raw_cells(
    miniature_result_bundle: tuple[
        dict[str, object],
        dict[str, object],
        dict[str, object],
        dict[str, object],
        dict[str, object],
    ],
) -> None:
    bundle, protocol, contexts, bindings, cache = deepcopy(miniature_result_bundle)

    summary = verify_result_bundle(
        bundle,
        expected_protocol=protocol,
        expected_contexts=contexts,
        expected_policy_bindings=bindings,
        expected_llm_cache=cache,
        expected_external_approval={
            "approved_freeze_commit": "a" * 40,
            "approved_prereg_sha256": "b" * 64,
        },
        expected_preregistration_canonical_digest="c" * 64,
    )

    evidence = bundle["evidence"]
    assert type(evidence) is dict
    assert evidence["cell_count"] == 11
    assert evidence["total_trial_count"] == 22
    assert evidence["cached_llm_attempt_count"] == 4
    assert summary == bundle["summary"]
    assert summary["negative_control"]["observed_valid_yield_delta"] == "0"
    source = inspect.getsource(verify_result_bundle)
    assert "AdaptiveSearch" not in source
    assert ".search(" not in source


def test_bundle_verifier_rejects_missing_raw_cell_and_aggregate_forgery(
    miniature_result_bundle: tuple[
        dict[str, object],
        dict[str, object],
        dict[str, object],
        dict[str, object],
        dict[str, object],
    ],
) -> None:
    original, protocol, contexts, bindings, cache = deepcopy(miniature_result_bundle)
    missing = deepcopy(original)
    evidence = missing["evidence"]
    assert type(evidence) is dict
    cells = evidence["cells"]
    assert type(cells) is list
    cells.pop()
    evidence["cell_count"] = len(cells)
    evidence["total_trial_count"] = sum(
        len(cell["evaluation_traces"]) for cell in cells
    )
    evidence["cached_llm_attempt_count"] = sum(
        len(cell["llm_audit_attempts"]) for cell in cells
    )
    evidence["evidence_digest"] = canonical_digest(cells)
    _refresh_bundle_digest(missing)
    with pytest.raises(EvidenceVerificationError, match="cell|evidence|count"):
        verify_result_bundle(
            missing,
            expected_protocol=protocol,
            expected_contexts=contexts,
            expected_policy_bindings=bindings,
            expected_llm_cache=cache,
        )

    forged = deepcopy(original)
    summary = forged["summary"]
    assert type(summary) is dict
    families = summary["families"]
    assert type(families) is dict
    families["app_scam_mule"]["observed_delta"] = "999"
    _refresh_bundle_digest(forged)
    with pytest.raises(EvidenceVerificationError, match="summary|aggregate|claim"):
        verify_result_bundle(
            forged,
            expected_protocol=protocol,
            expected_contexts=contexts,
            expected_policy_bindings=bindings,
            expected_llm_cache=cache,
        )


def test_bundle_verifier_rejects_negative_control_claim_tamper(
    miniature_result_bundle: tuple[
        dict[str, object],
        dict[str, object],
        dict[str, object],
        dict[str, object],
        dict[str, object],
    ],
) -> None:
    bundle, protocol, contexts, bindings, cache = deepcopy(miniature_result_bundle)
    summary = bundle["summary"]
    assert type(summary) is dict
    negative = summary["negative_control"]
    assert type(negative) is dict
    negative["observed_valid_yield_delta"] = "1"
    negative["supported"] = True
    _refresh_bundle_digest(bundle)

    with pytest.raises(EvidenceVerificationError, match="negative|summary|claim"):
        verify_result_bundle(
            bundle,
            expected_protocol=protocol,
            expected_contexts=contexts,
            expected_policy_bindings=bindings,
            expected_llm_cache=cache,
        )


def test_postexecution_mode_allows_exact_untracked_result_and_nothing_else() -> None:
    validate = getattr(
        holdout_runner,
        "_validate_postexecution_worktree_status",
        None,
    )
    assert callable(validate)
    exact = b"?? docs/experiments/task6-v3.4-holdout-result.json\0"

    validate(exact)
    for invalid in (
        b"",
        b" M scripts/run_task6_holdout.py\0" + exact,
        exact + b"?? unrelated.txt\0",
        b"A  docs/experiments/task6-v3.4-holdout-result.json\0",
    ):
        with pytest.raises(RuntimeError, match="post-execution|result|worktree"):
            validate(invalid)


def test_postcommit_chronology_requires_exact_prereg_parent_path_mode_and_sha(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    validate = getattr(holdout_runner, "_validate_postcommit_chronology", None)
    assert callable(validate)
    result_commit = "d" * 40
    prereg_commit = "e" * 40
    result_sha = "f" * 64
    result_path = "docs/experiments/task6-v3.4-holdout-result.json"
    approved = {result_path: result_sha}

    def git_output(arguments: list[str], *, text: bool = True) -> str | bytes:
        if arguments == ["rev-parse", "HEAD"]:
            return result_commit + "\n"
        if arguments == ["rev-list", "--parents", "-n", "1", result_commit]:
            return f"{result_commit} {prereg_commit}\n"
        if arguments == [
            "diff-tree",
            "--no-commit-id",
            "--name-status",
            "-r",
            "-z",
            result_commit,
        ]:
            return b"A\0" + result_path.encode() + b"\0"
        if arguments == ["cat-file", "blob", "a" * 40]:
            return b"result-bytes"
        raise AssertionError(arguments)

    monkeypatch.setattr(holdout_runner, "_git_output", git_output)
    monkeypatch.setattr(
        holdout_runner,
        "_git_tree_records",
        lambda _commit: {
            result_path: {
                "git_mode": "100644",
                "object_type": "blob",
                "git_object_id": "a" * 40,
            }
        },
    )
    monkeypatch.setattr(
        holdout_runner,
        "_sha256_bytes",
        lambda value: result_sha if value == b"result-bytes" else "0" * 64,
    )

    validate(
        approved_result_commit=result_commit,
        approved_result_sha256=result_sha,
        preregistration_commit=prereg_commit,
        approved_artifacts=approved,
    )

    with pytest.raises(RuntimeError, match="parent|preregistration|chronology"):
        validate(
            approved_result_commit=result_commit,
            approved_result_sha256=result_sha,
            preregistration_commit="1" * 40,
            approved_artifacts=approved,
        )
    with pytest.raises(RuntimeError, match="path|artifact|SHA"):
        validate(
            approved_result_commit=result_commit,
            approved_result_sha256="2" * 64,
            preregistration_commit=prereg_commit,
            approved_artifacts=approved,
        )


def test_cli_declares_prepare_postexecution_and_postcommit_as_exclusive_modes() -> None:
    parser = holdout_runner._parse_args

    assert parser(["--prepare-preregistration"]).prepare_preregistration is True
    assert parser(["--verify-postexecution"]).verify_postexecution is True
    postcommit = parser(
        [
            "--verify-postcommit",
            "--approved-result-commit",
            "a" * 40,
            "--approved-result-sha256",
            "b" * 64,
        ]
    )
    assert postcommit.verify_postcommit is True
    with pytest.raises(SystemExit):
        parser(["--verify-postcommit"])
    with pytest.raises(SystemExit):
        parser(["--verify-postexecution", "--verify-only"])


def test_v34_preregistration_builder_freezes_instrumentation_and_full_contexts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_commit = "9" * 40
    source_freeze = {
        "source_commit": source_commit,
        "git_tree": "8" * 40,
        "behavior_manifest": {"entries": {}, "digest": canonical_digest({})},
        "lock_file": {"path": None, "status": "explicitly_absent"},
        "environment": {
            "python_version": "3.12.0",
            "python_implementation": "CPython",
            "python_cache_tag": "cpython-312",
            "platform": "test",
            "installed_distributions": [],
            "installed_distributions_digest": canonical_digest([]),
        },
    }
    monkeypatch.setattr(
        holdout_runner,
        "_source_freeze_document",
        lambda _commit: source_freeze,
    )

    artifact = holdout_runner._build_preregistration_document(source_commit)

    holdout_runner._validate_preregistration_schema(artifact)
    assert artifact["source_freeze"] == source_freeze
    assert artifact["protocol"] == holdout_runner._expected_protocol()
    assert artifact["behavior_equivalence"]["equivalent"] is True
    assert artifact["predecessor_evidence"]["v3_3_result"]["status"] == (
        "preserved_but_rejected_unverifiable"
    )
    assert set(artifact["evidence_contexts"]["targets"]) == {
        "app_scam_mule",
        "card_testing_cnp",
    }
    for context in artifact["evidence_contexts"]["targets"].values():
        assert set(context) == {
            "public_bounds",
            "evaluation_contract",
            "evaluator_code_digest",
            "defender",
        }
        assert context["public_bounds"]["feasible_vectors"]
        assert context["defender"]["rules"]
    publication = artifact["result_publication"]
    assert publication["approved_artifact_paths"] == [
        "docs/experiments/task6-v3.4-holdout-result.json"
    ]
    assert publication["postexecution_mode"] == (
        "exactly_one_untracked_result_and_no_other_change"
    )
    assert publication["postcommit_mode"] == (
        "result_commit_parent_is_exact_preregistration_commit"
    )
    assert publication["verifier_calls_policy_search"] is False
    assert not V34_RESULT.exists()

    tampered = deepcopy(artifact)
    tampered["behavior_equivalence"]["equivalent"] = False
    with pytest.raises(RuntimeError, match="behavior|equivalence"):
        holdout_runner._validate_preregistration_schema(tampered)
