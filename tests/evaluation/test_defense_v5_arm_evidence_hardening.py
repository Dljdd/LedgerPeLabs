"""Tamper-evident provenance contracts for Sentinel v5 comparison arms."""

from __future__ import annotations

import inspect
from pathlib import Path

import apar.evaluation.v5_evaluation as v5_evaluation
from apar.evaluation.v5_evaluation import V5Arm, load_v5_arm_configuration
from apar.features.sentinel import SentinelFeatureCatalog
from tests.evaluation.v5_safe_protocol import load_safe_v5_test_protocol

ROOT = Path(__file__).resolve().parents[2]


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


def test_trainer_requires_exact_partition_evidence_arguments() -> None:
    from apar.evaluation import v5_arms

    parameters = inspect.signature(v5_arms.train_v5_arm_set).parameters
    assert {"train_evidence", "calibration_evidence", "threshold_evidence"} <= set(
        parameters
    )
    assert callable(getattr(v5_arms, "route_full_sentinel_components", None))


def test_rules_only_frozen_spec_is_graph_free() -> None:
    """A graph=false rules arm cannot retain graph feature inputs."""
    protocol = load_safe_v5_test_protocol(ROOT)
    configuration = load_v5_arm_configuration(
        ROOT / "config/defense/defense-v5-arms.json",
        catalog=SentinelFeatureCatalog.default(),
        protocol=protocol,
    )
    rules = next(spec for spec in configuration.arms if spec.arm is V5Arm.RULES_ONLY)
    assert rules.graph is False
    assert rules.graph_feature_names == ()
    assert all(not name.startswith("graph_") for name in rules.feature_names)


def test_safe_404_protocol_copy_has_its_own_verified_digest() -> None:
    from apar.evaluation import v5_protocol

    digest = getattr(v5_protocol, "v5_protocol_digest", None)
    assert callable(digest), "protocol digest recomputation API is missing"
    safe = load_safe_v5_test_protocol(ROOT)
    assert safe.seeds.development_test == 404
    assert safe.protocol_sha256 == digest(safe)
