"""Run the Sentinel v5 development pipeline."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from apar.defense.sentinel import SentinelAction, train_sentinel_defender
from apar.evaluation.v5_population import build_v5_corpus
from apar.evaluation.v5_protocol import V5Profile, load_v5_development_protocol
from apar.evaluation.v5_reporting import build_v5_development_result

_FORBIDDEN_FEATURE_NAMES = {
    "family", "campaign_id", "is_fraud", "seed", "split", "scenario_id",
}


def _score_and_evaluate(
    *,
    train_decisions,
    calibration_decisions,
    threshold_decisions,
    dev_test_decisions,
    protocol_seeds: tuple[int, ...],
    bootstrap_seed: int,
) -> dict:
    """Score the partition with a trained Sentinel defender and evaluate arms."""
    from apar.evaluation.v5_evaluation import V5Arm, evaluate_v5_arm

    train_fraud = [r for r in train_decisions if r.is_fraud]
    train_benign = [r for r in train_decisions if not r.is_fraud]

    cal_fraud = [r for r in calibration_decisions if r.is_fraud]
    cal_benign = [r for r in calibration_decisions if not r.is_fraud]
    thr_fraud = [r for r in threshold_decisions if r.is_fraud]
    thr_benign = [r for r in threshold_decisions if not r.is_fraud]

    if not train_fraud or not train_benign:
        return {"error": "insufficient mixed data for training"}
    if not cal_fraud or not cal_benign:
        return {"error": "one-class calibration partition"}
    if not thr_fraud or not thr_benign:
        return {"error": "one-class threshold partition"}

    feature_names = sorted(
        {key for row in train_decisions for key in row.predictive_features}
        - _FORBIDDEN_FEATURE_NAMES
    )

    def _matrix(selected_rows):
        return np.array([
            [row.predictive_features.get(name, 0.0) for name in feature_names]
            for row in selected_rows
        ], dtype=np.float64)

    x_train = np.vstack([_matrix(train_benign), _matrix(train_fraud)])
    y_train = np.concatenate([
        np.zeros(len(train_benign), dtype=int),
        np.ones(len(train_fraud), dtype=int),
    ])
    x_cal = _matrix(calibration_decisions)
    y_cal = np.array([0 if not r.is_fraud else 1 for r in calibration_decisions])
    x_threshold = _matrix(threshold_decisions)
    y_threshold = np.array([0 if not r.is_fraud else 1 for r in threshold_decisions])
    x_test = _matrix(dev_test_decisions)
    y_test = np.array([0 if not r.is_fraud else 1 for r in dev_test_decisions])

    defender = train_sentinel_defender(
        x_train=x_train,
        y_train=y_train,
        x_calibration=x_cal if len(x_cal) > 1 else x_threshold,
        y_calibration=y_cal if len(y_cal) > 1 else y_threshold,
        x_threshold=x_threshold,
        y_threshold=y_threshold,
        catboost_seeds=protocol_seeds,
        bootstrap_seed=bootstrap_seed,
    )

    start = time.perf_counter()
    decisions = defender.decide_batch(x_test)
    elapsed_ms = [(time.perf_counter() - start) / max(len(x_test), 1) * 1000] * len(x_test)

    actions = [SentinelAction(d.action) for d in decisions]
    probs = np.array([d.ensemble_probability for d in decisions])
    campaign_ids = np.array([r.campaign_id for r in dev_test_decisions])
    amounts = np.array([float(r.amount) for r in dev_test_decisions])

    arm_result = evaluate_v5_arm(
        arm=V5Arm.FULL_SENTINEL,
        y_true=y_test,
        actions=actions,
        probabilities=probs,
        campaign_ids=campaign_ids,
        amounts=amounts,
    )

    return {
        "arm_result": arm_result.model_dump(mode="json"),
        "latencies": elapsed_ms,
    }




def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=["smoke", "production"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    protocol = load_v5_development_protocol(root / "config/defense/defense-v5-development.json")
    profile = V5Profile(args.profile)
    corpus = build_v5_corpus(protocol, profile=profile)

    train_partition = corpus.partitions.get("train")
    calibration_partition = corpus.partitions.get("calibration")
    threshold_partition = corpus.partitions.get("threshold")
    dev_test_partition = corpus.partitions.get("development_test")

    arm_metrics: dict[str, dict] = {}
    partitions_ready = all(
        p is not None
        for p in (train_partition, calibration_partition, threshold_partition, dev_test_partition)
    )
    if partitions_ready:
        scoring_output = _score_and_evaluate(
            train_decisions=train_partition.decisions,
            calibration_decisions=calibration_partition.decisions,
            threshold_decisions=threshold_partition.decisions,
            dev_test_decisions=dev_test_partition.decisions,
            protocol_seeds=protocol.seeds.catboost_seeds,
            bootstrap_seed=protocol.seeds.bootstrap,
        )
        if "arm_result" in scoring_output:
            arm_metrics["full_sentinel"] = scoring_output["arm_result"]

    result = build_v5_development_result(
        protocol=protocol, corpus=corpus, arms=arm_metrics,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(json.loads(result.model_dump_json()), indent=2) + "\n")
    print(f"status={result.status} profile={result.profile} output={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
