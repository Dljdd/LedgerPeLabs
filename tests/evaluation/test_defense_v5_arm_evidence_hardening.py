"""Tamper-evident provenance contracts for Sentinel v5 comparison arms."""

from __future__ import annotations

import hashlib
import inspect
import json
import subprocess
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
from catboost import CatBoostClassifier

import apar.evaluation.v5_evaluation as v5_evaluation
from apar.evaluation.v5_evaluation import (
    V5Arm,
    V5ArmSpecification,
    V5CalibratorManifest,
    V5IsolationForestManifest,
    V5IsolationTreeManifest,
    V5TrainingPartitionEvidence,
    discover_v5_implementation_paths,
    load_v5_arm_configuration,
)
from apar.features import sentinel as sentinel_features
from apar.features.sentinel import SentinelFeatureCatalog
from tests.evaluation.v5_safe_protocol import load_safe_v5_test_protocol

ROOT = Path(__file__).resolve().parents[2]


def test_isolation_tree_replay_preserves_float64_threshold_boundary() -> None:
    """A float32 feature just above a retained threshold must take the right branch."""
    threshold = 5.148292989539119
    rounded_feature = float(np.float32(threshold))
    assert rounded_feature > threshold
    tree = V5IsolationTreeManifest(
        children_left=(1, -1, -1),
        children_right=(2, -1, -1),
        feature=(0, -2, -2),
        threshold=(threshold, -2.0, -2.0),
        decision_path_lengths=(1.0, 2.0, 2.0),
        average_path_lengths=(0.0, 0.0, 0.0),
        estimator_features=(0,),
    )

    assert tree.leaf_index((rounded_feature,)) == 2


def test_v5_implementation_paths_use_tracked_canonical_case() -> None:
    """Case aliases cannot be bound reproducibly on a case-sensitive verifier."""
    discovered = discover_v5_implementation_paths(ROOT)
    tracked = set(
        subprocess.run(
            ["git", "ls-files"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.splitlines()
    )
    assert "src/apar/generators/Population.py" not in discovered
    assert set(discovered) <= tracked


def _independent_digest(document: object) -> str:
    return hashlib.sha256(
        json.dumps(
            document,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode()
    ).hexdigest()


def test_arm_evidence_contracts_exist_before_training() -> None:
    """A matrix-only trainer cannot bind partition provenance or artifact identity."""
    assert getattr(v5_evaluation, "V5TrainingPartitionEvidence", None) is not None
    assert callable(getattr(v5_evaluation, "build_v5_training_partition_evidence", None))
    assert "score_sha256" in v5_evaluation.V5ArmScore.model_fields
    assert "row_output_sha256" in v5_evaluation.V5ArmRowEvidence.model_fields
    assert "result_sha256" in v5_evaluation.V5EvaluationResult.model_fields
    required_row_fields = {
        "catalog_feature_values",
        "subset_feature_values",
        "catalog_feature_sha256",
        "subset_feature_sha256",
        "model_raw_scores",
        "model_calibrated_scores",
        "threshold_trace",
        "rule_components",
        "probability_action",
        "model_action",
        "rule_action",
        "trust_action",
        "novelty_raw_score",
        "novelty_routed",
        "disagreement_routed",
    }
    assert required_row_fields <= set(v5_evaluation.V5ArmRowEvidence.model_fields)
    assert {
        "feature_names",
        "feature_matrix",
        "support_records",
        "execution_artifacts",
    } <= set(v5_evaluation.V5TrainingPartitionEvidence.model_fields)
    assert {
        "model_artifacts",
        "novelty_manifest",
    } <= set(v5_evaluation.V5ArmSpecification.model_fields)
    assert "execution_artifacts" in v5_evaluation.V5ArmScore.model_fields
    assert getattr(v5_evaluation, "V5SerializedModelArtifact", None) is not None
    assert getattr(v5_evaluation, "V5IsolationForestManifest", None) is not None
    assert getattr(v5_evaluation, "V5ExecutionArtifact", None) is not None


def test_catboost_evidence_serialization_excludes_only_volatile_metadata() -> None:
    """Restoring CBM bytes must make equal seeded models receive different identities."""
    from apar.evaluation.v5_arms import _model_artifact

    features = [[0.0, 1.0], [1.0, 0.0], [0.1, 0.9], [0.9, 0.1]]
    labels = [0, 1, 0, 1]
    models = []
    for _index in range(2):
        model = CatBoostClassifier(
            iterations=2,
            depth=2,
            random_seed=123,
            allow_writing_files=False,
            verbose=False,
        )
        model.fit(features, labels)
        models.append(model)
    left, right = (_model_artifact(model) for model in models)
    assert left.serialization == "catboost-json-canonical-v1"
    assert left == right
    assert left.load_model().predict_proba(features).tolist() == models[0].predict_proba(
        features
    ).tolist()


def test_trainer_requires_exact_partition_evidence_arguments() -> None:
    from apar.evaluation import v5_arms

    parameters = inspect.signature(v5_arms.train_v5_arm_set).parameters
    assert {"train_evidence", "calibration_evidence", "threshold_evidence"} <= set(
        parameters
    )
    assert callable(getattr(v5_arms, "route_full_sentinel_components", None))


def test_rules_only_frozen_spec_truthfully_binds_rule_engine_graph_inputs() -> None:
    """The real RuleEngine graph rules must be declared in the rules arm."""
    protocol = load_safe_v5_test_protocol(ROOT)
    configuration = load_v5_arm_configuration(
        ROOT / "config/defense/defense-v5-arms.json",
        catalog=SentinelFeatureCatalog.default(),
        protocol=protocol,
    )
    rules = next(spec for spec in configuration.arms if spec.arm is V5Arm.RULES_ONLY)
    assert rules.graph is True
    assert rules.graph_feature_names == (
        "graph_counterparty_fanin",
        "graph_actor_fanout",
        "graph_shared_neighbor_count",
    )
    assert set(rules.graph_feature_names) <= set(rules.feature_names)


def test_safe_404_protocol_copy_has_its_own_verified_digest() -> None:
    from apar.evaluation import v5_protocol

    digest = getattr(v5_protocol, "v5_protocol_digest", None)
    assert callable(digest), "protocol digest recomputation API is missing"
    safe = load_safe_v5_test_protocol(ROOT)
    assert safe.seeds.development_test == 404
    assert safe.protocol_sha256 == digest(safe)


def test_arm_implementation_inventory_is_recursive_and_exact() -> None:
    document = json.loads(
        (ROOT / "config/defense/defense-v5-arms.json").read_text()
    )
    required = {
        "src/apar/__init__.py",
        "src/apar/contracts/__init__.py",
        "src/apar/contracts/_validation.py",
        "src/apar/contracts/decisions.py",
        "src/apar/contracts/events.py",
        "src/apar/contracts/reports.py",
        "src/apar/contracts/scenarios.py",
        "src/apar/defense/__init__.py",
        "src/apar/defense/contracts.py",
        "src/apar/defense/rules.py",
        "src/apar/defense/sentinel.py",
        "src/apar/evaluation/__init__.py",
        "src/apar/evaluation/contracts.py",
        "src/apar/evaluation/corpus.py",
        "src/apar/evaluation/regimes.py",
        "src/apar/evaluation/splits.py",
        "src/apar/evaluation/v5_arms.py",
        "src/apar/evaluation/v5_evaluation.py",
        "src/apar/evaluation/v5_execution.py",
        "src/apar/evaluation/v5_fidelity.py",
        "src/apar/evaluation/v5_hardening.py",
        "src/apar/evaluation/v5_population.py",
        "src/apar/evaluation/v5_protocol.py",
        "src/apar/evaluation/v5_reporting.py",
        "src/apar/features/__init__.py",
        "src/apar/features/builders.py",
        "src/apar/features/catalog.py",
        "src/apar/features/sentinel.py",
        "src/apar/features/state.py",
        "src/apar/generators/__init__.py",
        "src/apar/generators/campaigns.py",
        "src/apar/generators/population.py",
        "src/apar/redteam/__init__.py",
        "src/apar/redteam/llm_policy.py",
        "src/apar/redteam/policies.py",
        "src/apar/redteam/search.py",
        "src/apar/runs/__init__.py",
        "src/apar/runs/runner.py",
        "src/apar/runs/wire.py",
        "src/apar/simulator/engine.py",
        "src/apar/simulator/ledger.py",
        "src/apar/simulator/clock.py",
        "src/apar/simulator/__init__.py",
        "src/apar/simulator/rails/__init__.py",
        "src/apar/simulator/rails/agentic.py",
        "src/apar/simulator/rails/a2a.py",
        "src/apar/simulator/rails/base.py",
        "src/apar/simulator/rails/card.py",
        "src/apar/storage/artifacts.py",
        "src/apar/trust/__init__.py",
        "src/apar/trust/verifier.py",
        "scripts/run_defense_v5_development.py",
    }
    declared = tuple(document["implementation_paths"])
    discovered = v5_evaluation.discover_v5_implementation_paths(ROOT)
    assert required <= set(discovered)
    assert declared == discovered
    with pytest.raises(ValueError, match="exact dependency closure"):
        v5_evaluation.validate_v5_implementation_paths(ROOT, declared[:-1])
    with pytest.raises(ValueError, match="exact dependency closure"):
        v5_evaluation.validate_v5_implementation_paths(
            ROOT,
            (*declared, "src/apar/evaluation/unexecuted_extra.py"),
        )


@pytest.mark.parametrize(
    "forbidden_name",
    (
        "is_fraud",
        "training_label",
        "campaign_family",
        "generator_seed",
        "development_split",
        "future_outcome_probability",
    ),
)
def test_public_sentinel_semantic_validator_rejects_predictive_leaks(
    forbidden_name: str,
) -> None:
    validator = getattr(
        sentinel_features,
        "validate_sentinel_predictive_feature_names",
        None,
    )
    assert callable(validator), "public Sentinel feature semantic validator is missing"
    with pytest.raises(ValueError, match="forbidden predictive feature semantics"):
        validator(("txn_log_amount", forbidden_name))


def test_arm_spec_rejects_consistently_rehashed_forbidden_catalog_and_subsets() -> None:
    protocol = load_safe_v5_test_protocol(ROOT)
    configuration = load_v5_arm_configuration(
        ROOT / "config/defense/defense-v5-arms.json",
        catalog=SentinelFeatureCatalog.default(),
        protocol=protocol,
    )
    full = next(spec for spec in configuration.arms if spec.arm is V5Arm.FULL_SENTINEL)
    forged = deepcopy(full.model_dump(mode="json"))
    original = "amount"
    assert original in full.catalog_feature_names
    replacement = "future_outcome_probability"
    for field in (
        "catalog_feature_names",
        "feature_names",
        "graph_feature_names",
        "non_graph_feature_names",
    ):
        forged[field] = [
            replacement if name == original else name for name in forged[field]
        ]
    forged["spec_sha256"] = _independent_digest(
        {key: value for key, value in forged.items() if key != "spec_sha256"}
    )
    with pytest.raises(ValueError, match="forbidden predictive feature semantics"):
        V5ArmSpecification.model_validate(forged)


def test_arm_spec_rejects_forbidden_names_in_all_bound_training_evidence() -> None:
    protocol = load_safe_v5_test_protocol(ROOT)
    configuration = load_v5_arm_configuration(
        ROOT / "config/defense/defense-v5-arms.json",
        catalog=SentinelFeatureCatalog.default(),
        protocol=protocol,
    )
    rules = next(spec for spec in configuration.arms if spec.arm is V5Arm.RULES_ONLY)
    original = "amount"
    assert original in rules.catalog_feature_names
    replacement = "is_fraud"
    forged_catalog = tuple(
        replacement if name == original else name
        for name in rules.catalog_feature_names
    )

    def evidence(partition: str, suffix: str) -> V5TrainingPartitionEvidence:
        return V5TrainingPartitionEvidence.model_construct(
            partition=partition,
            ordered_event_ids=(f"{suffix}-0", f"{suffix}-1"),
            labels=(0, 1),
            feature_names=forged_catalog,
            feature_matrix=(
                tuple(0.0 for _ in forged_catalog),
                tuple(1.0 for _ in forged_catalog),
            ),
            support_records=(),
            execution_artifacts=(),
            catalog_sha256=rules.catalog_sha256,
            feature_batch_sha256="1" * 64,
            feature_batch_payload_json="{}",
            feature_matrix_sha256="2" * 64,
            ordered_rows_sha256="3" * 64,
            ordered_support_sha256="4" * 64,
        )

    partitions = (
        evidence("train", "train"),
        evidence("calibration", "calibration"),
        evidence("threshold", "threshold"),
    )
    threshold_values = (("rules_challenge", 0.6), ("rules_decline", 0.9))
    threshold_digest = _independent_digest(
        {
            "source_partition": "threshold",
            "method": "rules_v1_fixed",
            "threshold_ordered_rows_sha256": partitions[2].ordered_rows_sha256,
            "threshold_support_sha256": partitions[2].ordered_support_sha256,
            "threshold_feature_batch_sha256": partitions[2].feature_batch_sha256,
            "threshold_feature_matrix_sha256": partitions[2].feature_matrix_sha256,
            "threshold_values": threshold_values,
        }
    )
    values = {name: getattr(rules, name) for name in V5ArmSpecification.model_fields}
    values.update(
        catalog_feature_names=forged_catalog,
        execution_bound=True,
        training_partitions=partitions,
        threshold_values=threshold_values,
        threshold_digest=threshold_digest,
        spec_sha256="",
    )
    unchecked = V5ArmSpecification.model_construct(**values)
    unchecked = unchecked.model_copy(
        update={"spec_sha256": unchecked.computed_digest()}
    )
    with pytest.raises(ValueError, match="forbidden predictive feature semantics"):
        unchecked.specification_is_bound()


def test_calibrator_rejects_more_knots_than_one_production_partition() -> None:
    knot_count = 100_001
    values = {
        "x_thresholds": tuple(index / (knot_count - 1) for index in range(knot_count)),
        "y_thresholds": tuple(index / (knot_count - 1) for index in range(knot_count)),
        "out_of_bounds": "clip",
    }
    with pytest.raises(ValueError, match="calibrator.*production profile limit"):
        V5CalibratorManifest(
            **values,
            artifact_sha256=_independent_digest(values),
        )


def test_isolation_tree_rejects_unbounded_estimator_feature_indices() -> None:
    with pytest.raises(ValueError, match="estimator feature.*production profile limit"):
        V5IsolationTreeManifest(
            children_left=(-1,),
            children_right=(-1,),
            feature=(-2,),
            threshold=(-2.0,),
            decision_path_lengths=(1.0,),
            average_path_lengths=(0.0,),
            estimator_features=tuple(0 for _ in range(10_000)),
        )


def test_isolation_forest_rejects_rehashed_appended_unused_estimator_feature() -> None:
    tree = V5IsolationTreeManifest(
        children_left=(-1,),
        children_right=(-1,),
        feature=(-2,),
        threshold=(-2.0,),
        decision_path_lengths=(1.0,),
        average_path_lengths=(0.0,),
        estimator_features=(0, 1, 0),
    )
    values = {
        "serialization": "sklearn-isolation-forest-tree-arrays-v1",
        "feature_count": 2,
        "max_samples": 2,
        "offset": -0.5,
        "trees": (tree,),
    }
    with pytest.raises(ValueError, match="exact ordered estimator features"):
        V5IsolationForestManifest(
            **values,
            artifact_sha256=_independent_digest(
                {
                    **values,
                    "trees": [tree.model_dump(mode="json")],
                }
            ),
        )
