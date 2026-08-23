"""Run the Sentinel v5 development pipeline."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import TypedDict

import numpy as np
from numpy.typing import NDArray

from apar.defense.sentinel import (
    SentinelDecision,
    SentinelDefender,
)
from apar.evaluation.v5_arms import score_v5_arm_set, train_v5_arm_set
from apar.evaluation.v5_evaluation import (
    V5ArmConfiguration,
    V5ArmSupportRow,
    bind_v5_evaluation_result,
    build_v5_arm_support_rows,
    build_v5_execution_artifacts,
    build_v5_training_partition_evidence,
    evaluate_v5_arm,
    load_v5_arm_configuration,
)
from apar.evaluation.v5_population import (
    V5DecisionRow,
    V5ExecutionManifest,
    build_v5_corpus,
)
from apar.evaluation.v5_protocol import V5Profile, load_v5_development_protocol
from apar.evaluation.v5_reporting import build_v5_development_result
from apar.features.sentinel import (
    SentinelFeatureBatch,
    SentinelFeatureCatalog,
    build_sentinel_features,
)


class _ScoringOutput(TypedDict, total=False):
    error: str
    arm_results: dict[str, dict[str, object]]


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
    canonical = sorted(partition_decisions, key=lambda row: (row.decision_at, row.event_id))
    if list(partition_decisions) != canonical:
        raise ValueError("partition decisions must use canonical feature-row order")
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


def _arm_support(rows: Sequence[V5DecisionRow]) -> tuple[V5ArmSupportRow, ...]:
    return build_v5_arm_support_rows(rows)


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


def _score_all_arms_and_evaluate(
    *,
    train_decisions: Sequence[V5DecisionRow],
    train_executions: Sequence[V5ExecutionManifest],
    calibration_decisions: Sequence[V5DecisionRow],
    calibration_executions: Sequence[V5ExecutionManifest],
    threshold_decisions: Sequence[V5DecisionRow],
    threshold_executions: Sequence[V5ExecutionManifest],
    dev_test_decisions: Sequence[V5DecisionRow],
    dev_test_executions: Sequence[V5ExecutionManifest],
    catalog: SentinelFeatureCatalog,
    configuration: V5ArmConfiguration,
    bootstrap_seed: int,
) -> _ScoringOutput:
    """Train and independently score all frozen arms over identical support."""
    x_train, y_train, train_event_ids, _, _, train_batch = _build_partition_matrix(
        train_decisions, catalog
    )
    x_cal, y_cal, cal_event_ids, _, _, cal_batch = _build_partition_matrix(
        calibration_decisions, catalog
    )
    x_threshold, y_threshold, threshold_event_ids, _, _, threshold_batch = (
        _build_partition_matrix(threshold_decisions, catalog)
    )
    (
        x_test,
        y_test,
        test_event_ids,
        test_campaign_ids,
        test_amounts,
        test_batch,
    ) = _build_partition_matrix(dev_test_decisions, catalog)
    test_trust_failures = _derive_trust_failures(dev_test_decisions)

    train_fraud = int(y_train.sum())
    train_benign = len(y_train) - train_fraud
    if not train_fraud or not train_benign:
        raise ValueError("one-class training partition")
    cal_fraud = int(y_cal.sum())
    cal_benign = len(y_cal) - cal_fraud
    if not cal_fraud or not cal_benign:
        raise ValueError("one-class calibration partition")
    thr_fraud = int(y_threshold.sum())
    thr_benign = len(y_threshold) - thr_fraud
    if not thr_fraud or not thr_benign:
        raise ValueError("one-class threshold partition")

    train_support = _arm_support(train_decisions)
    calibration_support = _arm_support(calibration_decisions)
    threshold_support = _arm_support(threshold_decisions)
    train_evidence = build_v5_training_partition_evidence(
        partition="train",
        event_ids=train_event_ids,
        labels=y_train,
        support=train_support,
        feature_batch_sha256=train_batch.batch_sha256,
        feature_matrix=x_train,
        feature_names=catalog.feature_names,
        catalog_sha256=catalog.catalog_sha256,
        execution_manifests=train_executions,
        feature_batch_source_matrix=train_batch.matrix,
    )
    calibration_evidence = build_v5_training_partition_evidence(
        partition="calibration",
        event_ids=cal_event_ids,
        labels=y_cal,
        support=calibration_support,
        feature_batch_sha256=cal_batch.batch_sha256,
        feature_matrix=x_cal,
        feature_names=catalog.feature_names,
        catalog_sha256=catalog.catalog_sha256,
        execution_manifests=calibration_executions,
        feature_batch_source_matrix=cal_batch.matrix,
    )
    threshold_evidence = build_v5_training_partition_evidence(
        partition="threshold",
        event_ids=threshold_event_ids,
        labels=y_threshold,
        support=threshold_support,
        feature_batch_sha256=threshold_batch.batch_sha256,
        feature_matrix=x_threshold,
        feature_names=catalog.feature_names,
        catalog_sha256=catalog.catalog_sha256,
        execution_manifests=threshold_executions,
        feature_batch_source_matrix=threshold_batch.matrix,
    )

    trained = train_v5_arm_set(
        configuration=configuration,
        catalog=catalog,
        x_train=x_train,
        y_train=y_train,
        x_calibration=x_cal,
        y_calibration=y_cal,
        x_threshold=x_threshold,
        y_threshold=y_threshold,
        bootstrap_seed=bootstrap_seed,
        train_evidence=train_evidence,
        calibration_evidence=calibration_evidence,
        threshold_evidence=threshold_evidence,
    )
    support = _arm_support(dev_test_decisions)
    scores = score_v5_arm_set(
        trained=trained,
        catalog=catalog,
        features_matrix=x_test,
        support=support,
        execution_artifacts=build_v5_execution_artifacts(dev_test_executions),
        trust_failures=test_trust_failures,
        feature_provenance=test_batch.provenance,
    )
    if tuple(item.event_id for item in support) != tuple(test_event_ids):
        raise ValueError("evaluation support order disagrees with feature metadata")
    campaign_ids_arr = np.array(test_campaign_ids)
    amounts_arr = np.array(test_amounts)
    arm_results: dict[str, dict[str, object]] = {}
    for arm, score in scores.by_arm.items():
        base = evaluate_v5_arm(
            arm=arm,
            y_true=y_test,
            actions=[row.action for row in score.rows],
            probabilities=np.array([row.probability for row in score.rows]),
            campaign_ids=campaign_ids_arr,
            amounts=amounts_arr,
        )
        result = bind_v5_evaluation_result(base=base, score=score)
        arm_results[arm.value] = result.model_dump(mode="json")
    return {"arm_results": arm_results}


_score_and_evaluate = _score_all_arms_and_evaluate


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
    arm_configuration = load_v5_arm_configuration(
        root / "config/defense/defense-v5-arms.json",
        catalog=catalog,
        protocol=protocol,
    )
    profile = V5Profile(args.profile)
    corpus = build_v5_corpus(protocol, profile=profile)

    arm_metrics: dict[str, dict[str, object]] = {}
    train_partition = corpus.partitions["train"]
    calibration_partition = corpus.partitions["calibration"]
    threshold_partition = corpus.partitions["threshold"]
    dev_test_partition = corpus.partitions["development_test"]
    scoring_output = _score_all_arms_and_evaluate(
        train_decisions=train_partition.decisions,
        train_executions=train_partition.executions,
        calibration_decisions=calibration_partition.decisions,
        calibration_executions=calibration_partition.executions,
        threshold_decisions=threshold_partition.decisions,
        threshold_executions=threshold_partition.executions,
        dev_test_decisions=dev_test_partition.decisions,
        dev_test_executions=dev_test_partition.executions,
        catalog=catalog,
        configuration=arm_configuration,
        bootstrap_seed=protocol.seeds.bootstrap,
    )
    if "arm_results" in scoring_output:
        arm_metrics.update(scoring_output["arm_results"])

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
