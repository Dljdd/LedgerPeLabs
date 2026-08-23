"""Real-corpus integration for the four frozen Sentinel v5 arms."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path

import pytest

from apar.evaluation.v5_evaluation import (
    V5Arm,
    V5EvaluationResult,
    load_v5_arm_configuration,
)
from apar.evaluation.v5_population import build_v5_corpus
from apar.evaluation.v5_protocol import V5Profile
from apar.evaluation.v5_reporting import (
    V5DevelopmentResult,
    build_v5_development_result,
)
from apar.features.sentinel import SentinelFeatureCatalog
from scripts import run_defense_v5_development as runner
from tests.evaluation.v5_safe_protocol import load_safe_v5_test_protocol

ROOT = Path(__file__).resolve().parents[2]


def test_runner_scores_four_arms_over_identical_real_execution_support() -> None:
    """A full-only or cloned runner cannot produce independent bound row evidence."""
    score_all = getattr(runner, "_score_all_arms_and_evaluate", None)
    assert callable(score_all), "runner four-arm integration is missing"

    protocol = load_safe_v5_test_protocol(ROOT)
    catalog = SentinelFeatureCatalog.default()
    configuration = load_v5_arm_configuration(
        ROOT / "config/defense/defense-v5-arms.json",
        catalog=catalog,
        protocol=protocol,
    )
    corpus = build_v5_corpus(protocol, profile=V5Profile.SMOKE)
    output = score_all(
        train_decisions=corpus.partitions["train"].decisions,
        train_executions=corpus.partitions["train"].executions,
        calibration_decisions=corpus.partitions["calibration"].decisions,
        calibration_executions=corpus.partitions["calibration"].executions,
        threshold_decisions=corpus.partitions["threshold"].decisions,
        threshold_executions=corpus.partitions["threshold"].executions,
        dev_test_decisions=corpus.partitions["development_test"].decisions,
        dev_test_executions=corpus.partitions["development_test"].executions,
        catalog=catalog,
        configuration=configuration,
        bootstrap_seed=protocol.seeds.bootstrap,
    )

    results = output["arm_results"]
    assert tuple(results) == (
        V5Arm.RULES_ONLY.value,
        V5Arm.ENSEMBLE_NO_GRAPH.value,
        V5Arm.ENSEMBLE_WITH_GRAPH.value,
        V5Arm.FULL_SENTINEL.value,
    )
    assert len({result["support_sha256"] for result in results.values()}) == 1
    expected_event_ids = tuple(
        row.event_id for row in corpus.partitions["development_test"].decisions
    )
    assert all(
        tuple(item["support"]["event_id"] for item in result["row_evidence"])
        == expected_event_ids
        for result in results.values()
    )
    assert any(
        row["trust_routed"]
        for row in results[V5Arm.FULL_SENTINEL.value]["row_evidence"]
    )
    assert not any(
        row["trust_routed"]
        for arm in (V5Arm.ENSEMBLE_NO_GRAPH, V5Arm.ENSEMBLE_WITH_GRAPH)
        for row in results[arm.value]["row_evidence"]
    )

    development = build_v5_development_result(
        protocol=protocol,
        corpus=corpus,
        arms=results,
        catalog_sha256=catalog.catalog_sha256,
    )
    assert len(development.result_sha256) == 64

    def independent_digest(document: object) -> str:
        return hashlib.sha256(
            json.dumps(
                document,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            ).encode()
        ).hexdigest()

    assert development.result_sha256 == independent_digest(
        development.model_dump(mode="json", exclude={"result_sha256"})
    )
    metric_tamper = deepcopy(results[V5Arm.FULL_SENTINEL.value])
    metric_tamper["recall"] = 0.0 if metric_tamper["recall"] != 0.0 else 1.0
    metric_tamper["result_sha256"] = independent_digest(
        {key: value for key, value in metric_tamper.items() if key != "result_sha256"}
    )
    with pytest.raises(ValueError, match="metric recall"):
        V5EvaluationResult.model_validate(metric_tamper)

    with pytest.raises(ValueError, match="exact four|complete"):
        build_v5_development_result(
            protocol=protocol,
            corpus=corpus,
            arms={V5Arm.FULL_SENTINEL.value: results[V5Arm.FULL_SENTINEL.value]},
            catalog_sha256=catalog.catalog_sha256,
        )
    cloned = development.model_dump(mode="json")
    cloned["arms"][V5Arm.ENSEMBLE_NO_GRAPH.value] = deepcopy(
        cloned["arms"][V5Arm.FULL_SENTINEL.value]
    )
    cloned["result_sha256"] = independent_digest(
        {key: value for key, value in cloned.items() if key != "result_sha256"}
    )
    with pytest.raises(ValueError, match="key|cloned"):
        V5DevelopmentResult.model_validate(cloned)

    mixed = development.model_dump(mode="json")
    mixed_arm = mixed["arms"][V5Arm.ENSEMBLE_NO_GRAPH.value]
    mixed_arm["arm_spec"]["training_partitions"][0]["feature_batch_sha256"] = "1" * 64
    mixed_spec_digest = independent_digest(
        {
            key: value
            for key, value in mixed_arm["arm_spec"].items()
            if key != "spec_sha256"
        }
    )
    mixed_arm["arm_spec"]["spec_sha256"] = mixed_spec_digest
    mixed_arm["arm_spec_sha256"] = mixed_spec_digest
    for row in mixed_arm["row_evidence"]:
        row["arm_spec_sha256"] = mixed_spec_digest
        row["row_output_sha256"] = independent_digest(
            {key: value for key, value in row.items() if key != "row_output_sha256"}
        )
    mixed_arm["score_sha256"] = independent_digest(
        {
            "spec": mixed_arm["arm_spec"],
            "support_sha256": mixed_arm["support_sha256"],
            "rows": mixed_arm["row_evidence"],
            "execution_artifacts": mixed_arm["execution_artifacts"],
        }
    )
    mixed_arm["result_sha256"] = independent_digest(
        {key: value for key, value in mixed_arm.items() if key != "result_sha256"}
    )
    mixed["result_sha256"] = independent_digest(
        {key: value for key, value in mixed.items() if key != "result_sha256"}
    )
    with pytest.raises(
        ValueError, match="mixed|training provenance|feature batch digest"
    ):
        V5DevelopmentResult.model_validate(mixed)

    divergent_features = development.model_dump(mode="json")
    divergent_arm = divergent_features["arms"][V5Arm.RULES_ONLY.value]
    non_rule_index = next(
        index
        for index, name in enumerate(
            divergent_arm["arm_spec"]["catalog_feature_names"]
        )
        if name not in divergent_arm["arm_spec"]["feature_names"]
    )
    divergent_row = divergent_arm["row_evidence"][0]
    divergent_row["catalog_feature_values"][non_rule_index] += 0.25
    divergent_row["catalog_feature_sha256"] = independent_digest(
        divergent_row["catalog_feature_values"]
    )
    divergent_row["row_output_sha256"] = independent_digest(
        {key: value for key, value in divergent_row.items() if key != "row_output_sha256"}
    )
    divergent_arm["score_sha256"] = independent_digest(
        {
            "spec": divergent_arm["arm_spec"],
            "support_sha256": divergent_arm["support_sha256"],
            "rows": divergent_arm["row_evidence"],
            "execution_artifacts": divergent_arm["execution_artifacts"],
        }
    )
    divergent_arm["result_sha256"] = independent_digest(
        {key: value for key, value in divergent_arm.items() if key != "result_sha256"}
    )
    divergent_features["result_sha256"] = independent_digest(
        {
            key: value
            for key, value in divergent_features.items()
            if key != "result_sha256"
        }
    )
    with pytest.raises(ValueError, match="catalog feature|feature stream"):
        V5DevelopmentResult.model_validate(divergent_features)
