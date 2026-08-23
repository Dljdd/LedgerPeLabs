"""Run the Sentinel v5 development pipeline."""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Sequence
from pathlib import Path
from typing import TypedDict

import numpy as np
from numpy.typing import NDArray

from apar.defense.sentinel import (
    SentinelAction,
    SentinelDecision,
    SentinelDefender,
    train_sentinel_defender,
)
from apar.evaluation.v5_evaluation import V5Arm, evaluate_v5_arm
from apar.evaluation.v5_population import V5DecisionRow, build_v5_corpus
from apar.evaluation.v5_protocol import V5Profile, load_v5_development_protocol
from apar.evaluation.v5_reporting import build_v5_development_result
from apar.features.sentinel import (
    SentinelFeatureBatch,
    SentinelFeatureCatalog,
    build_sentinel_features,
)


class _ScoringOutput(TypedDict, total=False):
    error: str
    arm_result: dict[str, object]


def _build_partition_matrix(
    partition_decisions: Sequence[V5DecisionRow],
    catalog: SentinelFeatureCatalog,
) -> tuple[
    NDArray[np.float64],
    NDArray[np.int_],
    list[str],
    list[str],
    list[float],
    SentinelFeatureBatch,
]:
    """Build features for one partition and return (matrix, labels, metadata)."""
    batch = build_sentinel_features(partition_decisions, catalog=catalog)
    matrix = np.array(batch.matrix, dtype=np.float64)
    labels = np.array([1 if row.is_fraud else 0 for row in partition_decisions], dtype=int)
    event_ids = [row.event_id for row in partition_decisions]
    campaign_ids = [row.campaign_id for row in partition_decisions]
    amounts = [float(row.amount) for row in partition_decisions]
    return matrix, labels, event_ids, campaign_ids, amounts, batch


def _derive_trust_failures(
    ordered_rows: Sequence[V5DecisionRow],
) -> list[bool]:
    """Return one explicit verifier-failure value in exact feature-row order."""
    failures: list[bool] = []
    for row in ordered_rows:
        if row.rail == "agentic":
            if not (
                row.execution_evidence_sha256
                and row.source_command_id
                and row.source_event_id
            ):
                raise ValueError("agentic row lacks real verifier execution evidence")
            if row.integrity_status not in {"pass", "fail"}:
                raise ValueError("agentic row is missing a validated verifier outcome")
            failures.append(row.integrity_status == "fail")
        else:
            if row.integrity_status != "not_applicable":
                raise ValueError("non-agentic row cannot contain a verifier outcome")
            failures.append(False)
    return failures


def _decide_with_trust(
    defender: SentinelDefender,
    features_matrix: np.ndarray,
    ordered_rows: Sequence[V5DecisionRow],
) -> list[SentinelDecision]:
    """Score an ordered feature matrix with its exact real-verifier outcomes."""
    if len(features_matrix) != len(ordered_rows):
        raise ValueError("feature rows and trust evidence rows must align exactly")
    return defender.decide_batch(
        features_matrix,
        trust_failures=_derive_trust_failures(ordered_rows),
    )


def _score_and_evaluate(
    *,
    train_decisions: Sequence[V5DecisionRow],
    calibration_decisions: Sequence[V5DecisionRow],
    threshold_decisions: Sequence[V5DecisionRow],
    dev_test_decisions: Sequence[V5DecisionRow],
    catalog: SentinelFeatureCatalog,
    protocol_seeds: tuple[int, ...],
    bootstrap_seed: int,
) -> _ScoringOutput:
    """Build features via the catalog, train, score, and evaluate."""
    x_train, y_train, _, _, _, _ = _build_partition_matrix(train_decisions, catalog)
    x_cal, y_cal, _, _, _, _ = _build_partition_matrix(calibration_decisions, catalog)
    x_threshold, y_threshold, _, _, _, _ = _build_partition_matrix(threshold_decisions, catalog)
    x_test, y_test, _test_event_ids, test_campaign_ids, test_amounts, _ = _build_partition_matrix(
        dev_test_decisions, catalog
    )
    test_trust_failures = _derive_trust_failures(dev_test_decisions)

    train_fraud = int(y_train.sum())
    train_benign = len(y_train) - train_fraud
    if not train_fraud or not train_benign:
        return {"error": "one-class training partition"}
    cal_fraud = int(y_cal.sum())
    cal_benign = len(y_cal) - cal_fraud
    if not cal_fraud or not cal_benign:
        return {"error": "one-class calibration partition"}
    thr_fraud = int(y_threshold.sum())
    thr_benign = len(y_threshold) - thr_fraud
    if not thr_fraud or not thr_benign:
        return {"error": "one-class threshold partition"}

    defender = train_sentinel_defender(
        x_train=x_train,
        y_train=y_train,
        x_calibration=x_cal,
        y_calibration=y_cal,
        x_threshold=x_threshold,
        y_threshold=y_threshold,
        catboost_seeds=protocol_seeds,
        bootstrap_seed=bootstrap_seed,
    )

    # Real inference latency: measure each row individually after warm-up.
    warmup_size = min(5, len(x_test))
    defender.decide_batch(
        x_test[:warmup_size],
        trust_failures=test_trust_failures[:warmup_size],
    )
    latencies_ns: list[int] = []
    for i in range(len(x_test)):
        start = time.perf_counter_ns()
        defender.decide(
            x_test[i],
            trust_failure=test_trust_failures[i],
        )
        latencies_ns.append(time.perf_counter_ns() - start)
    decisions = _decide_with_trust(defender, x_test, dev_test_decisions)

    actions = [SentinelAction(d.action) for d in decisions]
    probs = np.array([d.ensemble_probability for d in decisions])
    campaign_ids_arr = np.array(test_campaign_ids)
    amounts_arr = np.array(test_amounts)

    arm_result = evaluate_v5_arm(
        arm=V5Arm.FULL_SENTINEL,
        y_true=y_test,
        actions=actions,
        probabilities=probs,
        campaign_ids=campaign_ids_arr,
        amounts=amounts_arr,
    )

    latencies_ms = [ns / 1_000_000 for ns in latencies_ns]
    arm_dict = arm_result.model_dump(mode="json")
    arm_dict["p50_latency_ms"] = float(np.percentile(latencies_ms, 50))
    arm_dict["p95_latency_ms"] = float(np.percentile(latencies_ms, 95))
    arm_dict["p99_latency_ms"] = float(np.percentile(latencies_ms, 99))
    arm_dict["catalog_sha256"] = catalog.catalog_sha256

    return {"arm_result": arm_dict}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=["smoke", "production"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    protocol = load_v5_development_protocol(root / "config/defense/defense-v5-development.json")
    catalog = SentinelFeatureCatalog.from_config(
        root / protocol.feature_catalog_path
    )
    profile = V5Profile(args.profile)
    corpus = build_v5_corpus(protocol, profile=profile)

    arm_metrics: dict[str, dict[str, object]] = {}
    train_partition = corpus.partitions["train"]
    calibration_partition = corpus.partitions["calibration"]
    threshold_partition = corpus.partitions["threshold"]
    dev_test_partition = corpus.partitions["development_test"]
    scoring_output = _score_and_evaluate(
        train_decisions=train_partition.decisions,
        calibration_decisions=calibration_partition.decisions,
        threshold_decisions=threshold_partition.decisions,
        dev_test_decisions=dev_test_partition.decisions,
        catalog=catalog,
        protocol_seeds=protocol.seeds.catboost_seeds,
        bootstrap_seed=protocol.seeds.bootstrap,
    )
    if "arm_result" in scoring_output:
        arm_metrics["full_sentinel"] = scoring_output["arm_result"]

    result = build_v5_development_result(
        protocol=protocol,
        corpus=corpus,
        arms=arm_metrics,
        catalog_sha256=catalog.catalog_sha256,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(json.loads(result.model_dump_json()), indent=2) + "\n")
    print(f"status={result.status} profile={result.profile} output={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
