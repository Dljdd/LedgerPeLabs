"""Execute the Task 6 preregistered holdout exactly once.

The default mode refuses a dirty worktree or an existing result artifact.  Use
``--verify-only`` before the preregistration commit to validate all frozen digests
without evaluating any holdout seed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from decimal import Decimal
from pathlib import Path
from typing import Any

from apar.redteam import (
    AdaptiveSearch,
    AdaptiveTournamentPolicy,
    FamilyThreshold,
    FixedPolicy,
    LLMPlannerPolicy,
    PrimaryOutcome,
    RandomPolicy,
    SearchAuthority,
    capability_delta_report,
)
from tests.redteam.conftest import benchmark_population, campaign_benchmark

ROOT = Path(__file__).resolve().parents[1]
PREREGISTRATION_PATH = ROOT / "docs/experiments/task6-holdout-preregistration.json"
CACHE_PATH = ROOT / "docs/experiments/task6-cached-llm-replay.json"
RESULT_PATH = ROOT / "docs/experiments/task6-holdout-result.json"


class _NoNetworkClient:
    provider = "fixture"
    model_id = "cached-default-v1"

    def __init__(self) -> None:
        self.calls = 0

    def complete(self, _request: dict[str, object]) -> dict[str, object]:
        self.calls += 1
        raise AssertionError("holdout cached planner attempted network transport")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _load_exact_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if type(value) is not dict or any(type(key) is not str for key in value):
        raise TypeError(f"{path.name} must contain an exact JSON object")
    return value


def _canonical_digest(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode()
    return _sha256_bytes(encoded)


def _head_commit() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _require_clean_worktree() -> None:
    completed = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    if completed.stdout:
        raise RuntimeError("holdout requires the clean preregistration commit")


def _runtime() -> tuple[
    dict[str, Any],
    SearchAuthority,
    object,
    dict[str, object],
    dict[str, object],
    LLMPlannerPolicy,
    _NoNetworkClient,
]:
    preregistration = _load_exact_json(PREREGISTRATION_PATH)
    cache_artifact = _load_exact_json(CACHE_PATH)
    if _sha256_file(CACHE_PATH) != preregistration["cached_replay"]["file_sha256"]:
        raise RuntimeError("cached replay artifact digest changed")
    if cache_artifact["development_seed"] in preregistration["holdout"]["seeds"]:
        raise RuntimeError("cache preparation seed overlaps holdout")

    population = benchmark_population.__wrapped__()
    benchmarks = {
        family: campaign_benchmark(family, population, expose_realized_value=True)
        for family in ("app_scam_mule", "card_testing_cnp")
    }
    authority = SearchAuthority()
    run_group = authority.issue_run_group("task6-preregistered-holdout")
    evaluators = {
        family: benchmark.issue_evaluator_capability(authority)
        for family, benchmark in benchmarks.items()
    }
    no_network = _NoNetworkClient()
    cached_policy = LLMPlannerPolicy(
        no_network,
        replay_cache=cache_artifact["records"],
        require_cached_replay=True,
    )
    policy_objects = {
        "fixed": FixedPolicy(),
        "random": RandomPolicy(),
        "adaptive": AdaptiveTournamentPolicy(),
        "cached_llm": cached_policy,
    }
    versions = {
        "fixed": "1.0.0",
        "random": "1.0.0",
        "adaptive": "2.0.0",
        "cached_llm": "1.0.0",
    }
    policies = {
        name: authority.register_policy(policy, name=name, version=versions[name])
        for name, policy in policy_objects.items()
    }
    return (
        preregistration,
        authority,
        run_group,
        evaluators,
        policies,
        cached_policy,
        no_network,
    )


def _verify_frozen_bindings(
    artifact: dict[str, Any],
    evaluators: dict[str, object],
    policies: dict[str, object],
) -> None:
    for relative_path, expected in artifact["source_files"].items():
        if _sha256_file(ROOT / relative_path) != expected:
            raise RuntimeError(f"frozen source digest changed: {relative_path}")
    adaptive = policies["adaptive"]
    if adaptive.policy_code_digest != artifact["adaptive_policy"]["code_digest"]:
        raise RuntimeError("adaptive registered implementation digest changed")
    for family, expected in artifact["families"].items():
        evaluator = evaluators[family]
        contract = evaluator.evaluation_contract
        observed = {
            "evaluator_code_digest": evaluator.evaluator_code_digest,
            "contract_digest": contract.contract_digest,
            "bounds_digest": contract.bounds_digest,
            "hidden_template_digest": contract.hidden_template_digest,
            "background_digest": contract.background_digest,
            "population_digest": contract.population_digest,
            "evaluator_digest": contract.evaluator_digest,
            "defender_digest": contract.defender_digest,
            "disclosure_profile_digest": contract.disclosure_profile_digest,
        }
        if observed != expected["provenance"]:
            raise RuntimeError(f"frozen evaluator provenance changed: {family}")


def _metrics_document(metrics: object) -> dict[str, object]:
    return {
        "proposal_count": metrics.proposal_count,
        "approved_count": metrics.approved_count,
        "valid_yield": str(metrics.valid_yield),
        "net_settled_value": str(metrics.net_settled_value),
        "adaptation_speed": str(metrics.adaptation_speed),
        "campaign_scale": metrics.campaign_scale,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    artifact, authority, group, evaluators, policies, cached_policy, no_network = (
        _runtime()
    )
    _verify_frozen_bindings(artifact, evaluators, policies)
    if args.verify_only:
        print("verified frozen Task 6 holdout bindings; no holdout trial executed")
        return
    _require_clean_worktree()
    if RESULT_PATH.exists():
        raise RuntimeError("holdout result already exists; refusing a second execution")

    holdout = artifact["holdout"]
    seeds = tuple(holdout["seeds"])
    budget = holdout["budget"]
    wall_budget = holdout["wall_time_budget_ms"]
    thresholds = tuple(
        FamilyThreshold(
            family=family,
            primary_outcome=PrimaryOutcome(details["primary_outcome"]),
            minimum_delta=Decimal(details["minimum_delta"]),
            evaluation_contract=evaluators[family].evaluation_contract,
            evaluator_capability_id=evaluators[family].capability_id,
            evaluator_code_digest=evaluators[family].evaluator_code_digest,
        )
        for family, details in sorted(artifact["families"].items())
    )
    issued_preregistration = authority.issue_preregistration(
        run_group=group,
        seeds=seeds,
        budget=budget,
        wall_time_budget_ms=wall_budget,
        thresholds=thresholds,
        policies=tuple(policies[name] for name in sorted(policies)),
    )
    results = {
        family: {
            name: tuple(
                AdaptiveSearch(
                    evaluator_capability=evaluators[family],
                    policy_capability=policies[name],
                    run_group=group,
                ).search(
                    seed=seed,
                    budget=budget,
                    wall_time_budget_ms=wall_budget,
                )
                for seed in seeds
            )
            for name in ("fixed", "random", "adaptive", "cached_llm")
        }
        for family in sorted(evaluators)
    }
    report = capability_delta_report(
        issued_preregistration,
        results,
        authority=authority,
    )
    audit = cached_policy.take_audit_records()
    if no_network.calls != 0 or len(audit) != 2 * len(seeds) * budget:
        raise RuntimeError("cached LLM zero-network audit is incomplete")
    if any(record.call_status != "cache_success" for record in audit):
        raise RuntimeError("cached LLM holdout contains a replay miss")

    document = {
        "schema_version": "1.0.0",
        "preregistration_commit": _head_commit(),
        "preregistration_file_sha256": _sha256_file(PREREGISTRATION_PATH),
        "preregistration_canonical_digest": _canonical_digest(artifact),
        "holdout": holdout,
        "matched_budgets": report.matched_budgets,
        "supported_family_count": report.supported_family_count,
        "adaptive_claim": report.adaptive_claim,
        "families": {
            metric.family: {
                "primary_outcome": metric.primary_outcome.value,
                "minimum_delta": str(metric.minimum_delta),
                "observed_delta": str(metric.observed_delta),
                "supported": metric.supported,
                "fixed": _metrics_document(metric.fixed),
                "random": _metrics_document(metric.random),
                "adaptive": _metrics_document(metric.adaptive),
                "cached_llm": _metrics_document(metric.cached_llm),
            }
            for metric in report.family_metrics
        },
        "cached_llm_audit": {
            "attempt_count": len(audit),
            "cache_success_count": sum(
                record.call_status == "cache_success" for record in audit
            ),
            "network_call_count": no_network.calls,
            "audit_digest": _canonical_digest(
                [record.model_dump(mode="json") for record in audit]
            ),
        },
        "result_bindings": {
            family: {
                name: [
                    {
                        "seed": result.seed,
                        "result_id": result.result_id,
                        "result_seal": result.result_seal,
                        "canonical_document_digest": _canonical_digest(
                            result.canonical_document()
                        ),
                    }
                    for result in runs
                ]
                for name, runs in cells.items()
            }
            for family, cells in results.items()
        },
    }
    RESULT_PATH.write_text(
        json.dumps(document, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(document["families"], sort_keys=True, indent=2))
    print(f"supported_family_count={report.supported_family_count}")


if __name__ == "__main__":
    main()
