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
    corpus_partition_decisions,
    protocol_seeds: tuple[int, ...],
    bootstrap_seed: int,
) -> dict:
    """Score the partition with a trained Sentinel defender and evaluate arms."""
    from apar.evaluation.v5_evaluation import V5Arm, evaluate_v5_arm

    rows = corpus_partition_decisions
    fraud_rows = [r for r in rows if r.is_fraud]
    benign_rows = [r for r in rows if not r.is_fraud]

    if not fraud_rows or not benign_rows:
        return {"error": "insufficient mixed data for training"}

    feature_names = sorted(
        {key for row in rows for key in row.predictive_features}
        - _FORBIDDEN_FEATURE_NAMES
)

    def _matrix(selected_rows):
        return np.array([
            [row.predictive_features.get(name, 0.0) for name in feature_names]
            for row in selected_rows
        ], dtype=np.float64)

    x_benign = _matrix(benign_rows)
    x_fraud = _matrix(fraud_rows)
    y_benign = np.zeros(len(benign_rows), dtype=int)
    y_fraud = np.ones(len(fraud_rows), dtype=int)

    n_benign = len(benign_rows)
    train_end = int(n_benign * 0.4)
    cal_end = int(n_benign * 0.55)
    threshold_end = int(n_benign * 0.7)

    x_train = np.vstack([x_benign[:train_end], x_fraud])
    y_train = np.concatenate([y_benign[:train_end], y_fraud])
    x_cal = x_benign[train_end:cal_end]
    y_cal = y_benign[train_end:cal_end]
    x_threshold = x_benign[cal_end:threshold_end]
    y_threshold = y_benign[cal_end:threshold_end]
    x_test = np.vstack([x_benign[threshold_end:], x_fraud])
    y_test = np.concatenate([y_benign[threshold_end:], y_fraud])

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
    test_rows = x_test_rows_for_campaigns(
        rows, benign_rows, fraud_rows, threshold_end
    )
    campaign_ids = np.array([r.campaign_id for r in test_rows])
    amounts = np.array([float(r.amount) for r in test_rows])

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


def x_test_rows_for_campaigns(all_rows, benign_rows, fraud_rows, threshold_end):
    test_benign = benign_rows[threshold_end:]
    return test_benign + fraud_rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=["smoke", "production"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    protocol = load_v5_development_protocol(root / "config/defense/defense-v5-development.json")
    profile = V5Profile(args.profile)
    corpus = build_v5_corpus(protocol, profile=profile)
    dev_test = corpus.partitions.get("development_test")

    arm_metrics = {}
    if dev_test is not None:
        scoring_output = _score_and_evaluate(
            dev_test.decisions,
            protocol.seeds.catboost_seeds,
            protocol.seeds.bootstrap,
        )
        arm_metrics["full_sentinel"] = scoring_output.get("arm_result", {})

    result = build_v5_development_result(
        protocol=protocol, corpus=corpus, arms=arm_metrics,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(json.loads(result.model_dump_json()), indent=2) + "\n")
    print(f"status={result.status} profile={result.profile} output={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
