"""Verify or explicitly execute the frozen Task 6 v3.1 confirmatory experiment.

The local result-file check is an accidental-rerun guard, not cryptographic exactly-once
enforcement. Durable append-only execution receipts and cross-process verification remain
a Task 7 responsibility. The historical v2 runner is preserved at commit ``10bb4c4``.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import tempfile
from collections.abc import Callable
from contextlib import suppress
from decimal import Decimal, localcontext
from pathlib import Path
from typing import Any, cast

ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = ROOT / "src"
for import_root in (SOURCE_ROOT, ROOT):
    rendered = str(import_root)
    if rendered not in sys.path:
        sys.path.insert(0, rendered)

from apar.contracts.decisions import Action  # noqa: E402
from apar.redteam import (  # noqa: E402
    AdaptiveSearch,
    AdaptiveTournamentPolicy,
    EvaluatorCapability,
    FamilyThreshold,
    FixedPolicy,
    LLMPlannerPolicy,
    Policy,
    PolicyCapability,
    PolicyMetrics,
    PrimaryOutcome,
    RandomPolicy,
    RunGroupCapability,
    SearchAuthority,
    SearchResult,
    capability_delta_report,
)
from apar.redteam.task6_experiment import (  # noqa: E402
    Task6Experiment,
    build_task6_experiment,
)

PREREGISTRATION_PATH = ROOT / "docs/experiments/task6-v3.1-holdout-preregistration.json"
CACHE_PATH = ROOT / "docs/experiments/task6-v3-cached-llm-replay.json"
CANCELLATION_PATH = ROOT / "docs/experiments/task6-v3-cancellation.json"
CANCELLED_RESULT_PATH = ROOT / "docs/experiments/task6-v3-holdout-result.json"
RESULT_PATH = ROOT / "docs/experiments/task6-v3.1-holdout-result.json"
_PACKAGES = (
    "cryptography",
    "fastapi",
    "hypothesis",
    "mypy",
    "numpy",
    "pandas",
    "pyarrow",
    "pydantic",
    "pytest",
    "ruff",
)


class _NoNetworkClient:
    provider = "fixture"
    model_id = "cached-default-v1"

    def __init__(self) -> None:
        self.calls = 0

    def complete(self, _request: dict[str, object]) -> dict[str, object]:
        self.calls += 1
        raise AssertionError("v3.1 cached planner attempted network transport")


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


def _atomic_publish_result(
    path: Path,
    payload: bytes,
    *,
    before_publish: Callable[[], None] | None = None,
) -> None:
    """Durably publish once without replacing a concurrently-created result."""
    if not isinstance(path, Path):
        raise TypeError("result path must be a pathlib.Path")
    if type(payload) is not bytes:
        raise TypeError("result payload must be exact bytes")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    descriptor_open = True
    try:
        with os.fdopen(descriptor, "wb") as stream:
            descriptor_open = False
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        if before_publish is not None:
            before_publish()
        # A same-directory hard link is atomic and fails with EEXIST instead of replacing.
        os.link(temporary, path)
        temporary.unlink()
        directory_descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            # Some filesystems do not expose directory fsync.
            with suppress(OSError):
                os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        if descriptor_open:
            os.close(descriptor)
        temporary.unlink(missing_ok=True)


def _environment_document() -> dict[str, object]:
    return {
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "python_cache_tag": sys.implementation.cache_tag,
        "platform": platform.platform(),
        "pyproject_sha256": _sha256_file(ROOT / "pyproject.toml"),
        "lock_file": None,
        "lock_file_sha256": None,
        "packages": {
            package: importlib.metadata.version(package) for package in _PACKAGES
        },
    }


def _head_commit() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _require_recording_preconditions() -> None:
    completed = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    if completed.stdout:
        raise RuntimeError("v3.1 execution requires a clean preregistration commit")
    if RESULT_PATH.exists():
        raise RuntimeError("local v3.1 result exists; refusing an accidental rerun")


def _runtime() -> tuple[
    dict[str, Any],
    Task6Experiment,
    SearchAuthority,
    RunGroupCapability,
    dict[str, EvaluatorCapability],
    EvaluatorCapability,
    dict[str, PolicyCapability],
    LLMPlannerPolicy,
    _NoNetworkClient,
]:
    artifact = _load_exact_json(PREREGISTRATION_PATH)
    cache_artifact = _load_exact_json(CACHE_PATH)
    if _sha256_file(CACHE_PATH) != artifact["cached_replay"]["file_sha256"]:
        raise RuntimeError("v3.1 cached replay artifact digest changed")
    if cache_artifact["development_seed"] in artifact["holdout"]["seeds"]:
        raise RuntimeError("cache preparation seed overlaps the v3.1 holdout")

    experiment = build_task6_experiment(ROOT)
    authority = SearchAuthority()
    run_group = authority.issue_run_group("task6-v3.1-confirmatory")
    evaluators = {
        family: benchmark.issue_evaluator_capability(authority)
        for family, benchmark in experiment.benchmarks.items()
    }
    negative_evaluator = experiment.negative_control.issue_evaluator_capability(authority)
    no_network = _NoNetworkClient()
    cached_policy = LLMPlannerPolicy(
        no_network,
        replay_cache=cache_artifact["records"],
        require_cached_replay=True,
    )
    policy_objects: dict[str, Policy] = {
        "fixed": FixedPolicy(),
        "random": RandomPolicy(),
        "adaptive": AdaptiveTournamentPolicy(),
        "cached_llm": cast(Policy, cached_policy),
    }
    versions = {
        "fixed": "1.0.0",
        "random": "1.0.0",
        "adaptive": "3.0.0",
        "cached_llm": "1.0.0",
    }
    policies = {
        name: authority.register_policy(policy, name=name, version=versions[name])
        for name, policy in policy_objects.items()
    }
    return (
        artifact,
        experiment,
        authority,
        run_group,
        evaluators,
        negative_evaluator,
        policies,
        cached_policy,
        no_network,
    )


def _verify_frozen_bindings(
    artifact: dict[str, Any],
    experiment: Task6Experiment,
    authority: SearchAuthority,
    evaluators: dict[str, EvaluatorCapability],
    negative_evaluator: EvaluatorCapability,
    policies: dict[str, PolicyCapability],
) -> None:
    for relative_path, expected in artifact["source_files"].items():
        if _sha256_file(ROOT / relative_path) != expected:
            raise RuntimeError(f"frozen v3.1 source digest changed: {relative_path}")
    if _environment_document() != artifact["environment"]:
        raise RuntimeError("frozen v3.1 Python environment changed")
    if experiment.population_digest != artifact["frozen_benchmark"]["population_digest"]:
        raise RuntimeError("frozen v3.1 population changed")
    for name, expected in artifact["policy_bindings"].items():
        policy = authority.policy_binding(policies[name])
        observed = {
            "version": policy.version,
            "code_digest": policy.code_digest,
            "callable_digest": policy.callable_digest,
        }
        if observed != expected:
            raise RuntimeError(
                f"frozen v3.1 policy binding changed: {name}; "
                f"expected={expected!r}; observed={observed!r}"
            )
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
            raise RuntimeError(f"frozen v3.1 evaluator provenance changed: {family}")
    negative_contract = negative_evaluator.evaluation_contract
    negative_observed = {
        "evaluator_code_digest": negative_evaluator.evaluator_code_digest,
        "contract_digest": negative_contract.contract_digest,
        "bounds_digest": negative_contract.bounds_digest,
        "hidden_template_digest": negative_contract.hidden_template_digest,
        "background_digest": negative_contract.background_digest,
        "population_digest": negative_contract.population_digest,
        "evaluator_digest": negative_contract.evaluator_digest,
        "defender_digest": negative_contract.defender_digest,
        "disclosure_profile_digest": negative_contract.disclosure_profile_digest,
    }
    if negative_observed != artifact["negative_control"]["provenance"]:
        raise RuntimeError("frozen v3.1 negative-control provenance changed")


def _valid_yield(results: tuple[SearchResult, ...]) -> Decimal:
    proposal_count = sum(len(result.trials) for result in results)
    if proposal_count == 0:
        return Decimal(0)
    approved_count = sum(
        trial.feedback.action is Action.APPROVE
        for result in results
        for trial in result.trials
    )
    return Decimal(approved_count) / Decimal(proposal_count)


def _run_negative_control(
    *,
    authority: SearchAuthority,
    run_group: RunGroupCapability,
    evaluator: EvaluatorCapability,
    policies: dict[str, PolicyCapability],
    seeds: tuple[int, ...],
    budget: int,
    wall_time_budget_ms: int,
) -> dict[str, object]:
    policy_names = ("fixed", "random", "adaptive")
    results = {
        name: tuple(
            AdaptiveSearch(
                evaluator_capability=evaluator,
                policy_capability=policies[name],
                run_group=run_group,
            ).search(
                seed=seed,
                budget=budget,
                wall_time_budget_ms=wall_time_budget_ms,
            )
            for seed in seeds
        )
        for name in policy_names
    }
    all_results = tuple(
        result for name in policy_names for result in results[name]
    )
    matched = bool(all_results) and all(
        result.proposal_budget == budget
        and result.query_budget == budget
        and result.logical_time_budget == budget
        and result.wall_time_budget_ms == wall_time_budget_ms
        and result.proposals_used == budget
        and result.queries_used == budget
        and result.logical_time_used == budget
        and not result.wall_time_exhausted
        and result.wall_time_overrun_ms == 0
        for result in all_results
    )
    random_yield = _valid_yield(results["random"])
    adaptive_yield = _valid_yield(results["adaptive"])
    observed_delta = adaptive_yield - random_yield
    return {
        "family": evaluator.evaluation_contract.family,
        "primary_outcome": "valid_yield",
        "minimum_delta": "0.10",
        "included_in_supported_family_count": False,
        "matched_budgets": matched,
        "network_call_count": 0,
        "random_valid_yield": str(random_yield),
        "adaptive_valid_yield": str(adaptive_yield),
        "observed_valid_yield_delta": str(observed_delta),
        "supported": matched and observed_delta >= Decimal("0.10"),
        "result_bindings": {
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
            for name, runs in results.items()
        },
    }


def _metrics_document(metrics: PolicyMetrics) -> dict[str, object]:
    return {
        "proposal_count": metrics.proposal_count,
        "approved_count": metrics.approved_count,
        "valid_yield": str(metrics.valid_yield),
        "net_settled_value": str(metrics.net_settled_value),
        "adaptation_speed": str(metrics.adaptation_speed),
        "campaign_scale": metrics.campaign_scale,
    }


def _primary_value(result: SearchResult, outcome: PrimaryOutcome) -> Decimal:
    approved = tuple(
        trial for trial in result.trials if trial.feedback.action is Action.APPROVE
    )
    if outcome is PrimaryOutcome.VALID_YIELD:
        if not result.trials:
            return Decimal(0)
        return Decimal(len(approved)) / Decimal(len(result.trials))
    return sum(
        (trial.feedback.realized_value or Decimal(0) for trial in approved),
        Decimal(0),
    )


def _paired_delta(
    adaptive: Decimal,
    random: Decimal,
    outcome: PrimaryOutcome,
) -> Decimal:
    if outcome is not PrimaryOutcome.NET_SETTLED_VALUE_RATE:
        return adaptive - random
    if random == 0:
        return Decimal(0) if adaptive == 0 else Decimal(1)
    with localcontext() as context:
        context.prec = 28
        return (adaptive - random) / random


def _descriptive_uncertainty(
    adaptive: tuple[SearchResult, ...],
    random: tuple[SearchResult, ...],
    outcome: PrimaryOutcome,
) -> dict[str, object]:
    deltas = tuple(
        _paired_delta(
            _primary_value(adaptive_run, outcome),
            _primary_value(random_run, outcome),
            outcome,
        )
        for adaptive_run, random_run in zip(adaptive, random, strict=True)
    )
    with localcontext() as context:
        context.prec = 28
        observed_mean = sum(deltas, Decimal(0)) / Decimal(len(deltas))
        sign_means = sorted(
            sum(
                (
                    delta if mask & (1 << index) else -delta
                    for index, delta in enumerate(deltas)
                ),
                Decimal(0),
            )
            / Decimal(len(deltas))
            for mask in range(2 ** len(deltas))
        )
    lower = sign_means[int(Decimal("0.025") * Decimal(len(sign_means) - 1))]
    upper = sign_means[int(Decimal("0.975") * Decimal(len(sign_means) - 1))]
    return {
        "method": "exact_paired_sign_resampling_reference_interval",
        "role": "descriptive_only",
        "per_seed_deltas": [str(delta) for delta in deltas],
        "mean_per_seed_delta": str(observed_mean),
        "reference_interval_95": [str(lower), str(upper)],
    }


def _execute(
    artifact: dict[str, Any],
    authority: SearchAuthority,
    group: RunGroupCapability,
    evaluators: dict[str, EvaluatorCapability],
    negative_evaluator: EvaluatorCapability,
    policies: dict[str, PolicyCapability],
    cached_policy: LLMPlannerPolicy,
    no_network: _NoNetworkClient,
) -> dict[str, object]:
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
        raise RuntimeError("v3.1 cached LLM zero-network audit is incomplete")
    if any(record.call_status != "cache_success" for record in audit):
        raise RuntimeError("v3.1 cached LLM contains a replay miss")
    negative_control = _run_negative_control(
        authority=authority,
        run_group=group,
        evaluator=negative_evaluator,
        policies=policies,
        seeds=seeds,
        budget=budget,
        wall_time_budget_ms=wall_budget,
    )

    return {
        "schema_version": "1.0.0",
        "preregistration_commit": _head_commit(),
        "preregistration_file_sha256": _sha256_file(PREREGISTRATION_PATH),
        "preregistration_canonical_digest": _canonical_digest(artifact),
        "holdout": holdout,
        "matched_budgets": report.matched_budgets,
        "supported_family_count": report.supported_family_count,
        "adaptive_claim": report.adaptive_claim,
        "negative_control": negative_control,
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
                "uncertainty": _descriptive_uncertainty(
                    results[metric.family]["adaptive"],
                    results[metric.family]["random"],
                    metric.primary_outcome,
                ),
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


def main() -> None:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--verify-only", action="store_true")
    mode.add_argument("--execute-confirmatory", action="store_true")
    args = parser.parse_args()
    (
        artifact,
        experiment,
        authority,
        group,
        evaluators,
        negative_evaluator,
        policies,
        cached_policy,
        no_network,
    ) = _runtime()
    _verify_frozen_bindings(
        artifact,
        experiment,
        authority,
        evaluators,
        negative_evaluator,
        policies,
    )
    if args.verify_only:
        cancellation = _load_exact_json(CANCELLATION_PATH)
        if cancellation["status"] != "cancelled_before_execution":
            raise RuntimeError("v3 cancellation record is not canonical")
        if CANCELLED_RESULT_PATH.exists():
            raise RuntimeError("cancelled v3 result must remain absent")
        if RESULT_PATH.exists():
            raise RuntimeError("v3.1 result must be absent during Phase B verification")
        print("verified frozen Task 6 v3.1 bindings; no holdout trial executed")
        return

    _require_recording_preconditions()
    document = _execute(
        artifact,
        authority,
        group,
        evaluators,
        negative_evaluator,
        policies,
        cached_policy,
        no_network,
    )
    _atomic_publish_result(
        RESULT_PATH,
        (json.dumps(document, sort_keys=True, indent=2) + "\n").encode("utf-8"),
    )
    print(json.dumps(document["families"], sort_keys=True, indent=2))
    print(f"supported_family_count={document['supported_family_count']}")


if __name__ == "__main__":
    main()
