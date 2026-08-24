"""Complete, denominator-explicit Sentinel v5 metrics."""

from __future__ import annotations

from pathlib import Path

import pytest

from apar.defense.sentinel import SentinelAction
from apar.evaluation.v5_evaluation import (
    V5Arm,
    V5EvaluationResult,
    build_v5_arm_support_rows,
    build_v5_execution_artifacts,
    load_v5_arm_configuration,
)
from apar.evaluation.v5_evidence_protocol import (
    V5MetricApplicability,
    load_v5_evidence_protocol,
)
from apar.evaluation.v5_metrics import (
    compute_v5_binary_metrics,
    compute_v5_calibration,
    evaluate_v5_complete_result,
    reconcile_v5_economics,
)
from apar.evaluation.v5_population import build_v5_corpus
from apar.evaluation.v5_protocol import V5Profile
from apar.features.sentinel import SentinelFeatureCatalog
from scripts import run_defense_v5_development as runner
from tests.evaluation.v5_safe_protocol import load_safe_v5_test_protocol

ROOT = Path(__file__).resolve().parents[2]


def test_binary_metrics_record_exact_numerators_denominators_and_undefined_states() -> None:
    """Zero, undefined, and not-applicable cannot collapse into favorable zeroes."""
    support = ("e0", "e1", "e2", "e3")
    metrics = compute_v5_binary_metrics(
        labels=(0, 0, 1, 1),
        actions=(
            SentinelAction.APPROVE,
            SentinelAction.DECLINE_HOLD,
            SentinelAction.CHALLENGE,
            SentinelAction.APPROVE,
        ),
        probabilities=(0.1, 0.8, 0.7, 0.4),
        support_ids=support,
        probability_applicable=True,
    )
    assert metrics["recall"].value == pytest.approx(0.5)
    assert (metrics["recall"].numerator, metrics["recall"].denominator) == (1.0, 2.0)
    assert metrics["precision"].value == pytest.approx(0.5)
    assert metrics["f1"].value == pytest.approx(0.5)
    assert metrics["false_decline_rate"].value == pytest.approx(0.5)
    assert metrics["decline_rate"].value == pytest.approx(0.25)
    assert metrics["brier"].applicability is V5MetricApplicability.DEFINED

    benign_only = compute_v5_binary_metrics(
        labels=(0, 0),
        actions=(SentinelAction.APPROVE, SentinelAction.CHALLENGE),
        probabilities=(0.1, 0.2),
        support_ids=("b0", "b1"),
        probability_applicable=True,
    )
    assert benign_only["recall"].applicability is V5MetricApplicability.UNDEFINED
    assert benign_only["recall"].value is None
    assert benign_only["recall"].denominator == 0
    assert benign_only["roc_auc"].applicability is V5MetricApplicability.UNDEFINED
    assert benign_only["brier"].value is not None

    rules = compute_v5_binary_metrics(
        labels=(0, 1),
        actions=(SentinelAction.APPROVE, SentinelAction.DECLINE_HOLD),
        probabilities=None,
        support_ids=("r0", "r1"),
        probability_applicable=False,
    )
    for name in ("pr_auc", "roc_auc", "brier", "expected_calibration_error"):
        assert rules[name].applicability is V5MetricApplicability.NOT_APPLICABLE
        assert rules[name].value is None
        assert rules[name].numerator is None
        assert rules[name].denominator is None


def test_calibration_uses_frozen_bins_and_explicit_rules_only_semantics() -> None:
    """Changing bin edges or inventing rules probabilities must fail this contract."""
    protocol = load_v5_evidence_protocol(
        ROOT / "config/defense/defense-v5-evidence.json",
        root=ROOT,
    )
    calibration = compute_v5_calibration(
        labels=(0, 1),
        probabilities=(0.05, 0.95),
        boundaries=protocol.calibration.bin_boundaries,
        support_ids=("e0", "e1"),
        applicable=True,
    )
    assert tuple(bin_.lower for bin_ in calibration.bins) == tuple(
        index / 10 for index in range(10)
    )
    assert sum(bin_.count for bin_ in calibration.bins) == 2
    assert calibration.expected_calibration_error.value == pytest.approx(0.05)
    assert calibration.maximum_calibration_error.value == pytest.approx(0.05)

    rules = compute_v5_calibration(
        labels=(0, 1),
        probabilities=None,
        boundaries=protocol.calibration.bin_boundaries,
        support_ids=("e0", "e1"),
        applicable=False,
    )
    assert rules.bins == ()
    assert (
        rules.expected_calibration_error.applicability
        is V5MetricApplicability.NOT_APPLICABLE
    )


def test_economics_reconciles_real_lifecycles_once_per_payment_across_all_families() -> None:
    """Repeating row amounts or skipping ledger lifecycles must break reconciliation."""
    protocol = load_safe_v5_test_protocol(ROOT)
    evidence_protocol = load_v5_evidence_protocol(
        ROOT / "config/defense/defense-v5-evidence.json",
        root=ROOT,
    )
    partition = build_v5_corpus(protocol, profile=V5Profile.SMOKE).partitions[
        "development_test"
    ]
    support = build_v5_arm_support_rows(partition.decisions)
    economics = reconcile_v5_economics(
        support=support,
        execution_artifacts=build_v5_execution_artifacts(partition.executions),
        actions=tuple(SentinelAction.DECLINE_HOLD for _row in support),
        protocol=evidence_protocol.economics,
    )

    assert economics.ledger_conserved is True
    assert economics.payment_count < len(support)
    assert set(economics.currencies) == {"EUR", "USD"}
    assert sum(item.payment_count for item in economics.by_currency) == economics.payment_count
    assert all(item.attempted_amount > 0 for item in economics.by_currency)
    assert sum(item.authorized_amount for item in economics.by_currency) > 0
    assert sum(item.settled_or_posted_amount for item in economics.by_currency) > 0
    assert sum(
        item.returned_refunded_recovered_amount for item in economics.by_currency
    ) > 0
    assert all(
        item.captured_amount + item.escaped_amount == item.malicious_amount
        for item in economics.by_currency
    )
    assert all(item.escaped_amount == 0 for item in economics.by_currency)
    assert {item.family for item in economics.by_family} == {
        "agentic_intent_abuse",
        "app_scam_mule",
        "card_testing_cnp",
        "synthetic_merchant_refund",
    }
    assert {item.rail for item in economics.by_rail} == {"card", "a2a", "agentic"}
    assert len(economics.economics_sha256) == 64


def test_real_four_arm_metrics_include_families_calibration_economics_and_bootstrap() -> None:
    """Every complete metric must replay from one exact real safe-seed score stream."""
    protocol = load_safe_v5_test_protocol(ROOT)
    evidence_protocol = load_v5_evidence_protocol(
        ROOT / "config/defense/defense-v5-evidence.json",
        root=ROOT,
    )
    catalog = SentinelFeatureCatalog.from_config(ROOT / protocol.feature_catalog_path)
    configuration = load_v5_arm_configuration(
        ROOT / "config/defense/defense-v5-arms.json",
        catalog=catalog,
        protocol=protocol,
    )
    corpus = build_v5_corpus(protocol, profile=V5Profile.SMOKE)
    output = runner._score_all_arms_and_evaluate(
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
    arm_results = output["arm_results"]
    assert arm_results is not None
    complete = {
        arm: evaluate_v5_complete_result(
            result=V5EvaluationResult.model_validate(document),
            protocol=evidence_protocol,
        )
        for arm, document in arm_results.items()
    }
    assert set(complete) == {
        V5Arm.RULES_ONLY.value,
        V5Arm.ENSEMBLE_NO_GRAPH.value,
        V5Arm.ENSEMBLE_WITH_GRAPH.value,
        V5Arm.FULL_SENTINEL.value,
    }
    for arm, metrics in complete.items():
        assert {family.family for family in metrics.by_family} == {
            "agentic_intent_abuse",
            "app_scam_mule",
            "card_testing_cnp",
            "synthetic_merchant_refund",
        }
        assert all(family.campaign_count > 0 for family in metrics.by_family)
        assert all(family.campaign_alerts for family in metrics.by_family)
        assert metrics.aggregate["p50_latency_ms"].value is not None
        assert metrics.aggregate["p95_latency_ms"].value is not None
        assert metrics.aggregate["p99_latency_ms"].value is not None
        assert metrics.economics.payment_count > 0
        assert (
            metrics.aggregate["captured_value_fraction"].applicability
            is V5MetricApplicability.DEFINED
        )
        assert (
            metrics.aggregate["escaped_value_fraction"].applicability
            is V5MetricApplicability.DEFINED
        )
        assert metrics.bootstrap.replicates == 2000
        assert len(metrics.bootstrap.samples) == 2000
        assert {
            interval.metric
            for interval in metrics.bootstrap.intervals
        } >= {
            "recall",
            "false_decline_rate",
            "captured_value_fraction",
            "campaign_detection_rate",
        }
        if arm == V5Arm.RULES_ONLY.value:
            assert (
                metrics.calibration.applicability
                is V5MetricApplicability.NOT_APPLICABLE
            )
        else:
            assert metrics.calibration.applicability is V5MetricApplicability.DEFINED
            assert any(
                interval.metric == "expected_calibration_error"
                for interval in metrics.bootstrap.intervals
            )
        replay = evaluate_v5_complete_result(
            result=V5EvaluationResult.model_validate(arm_results[arm]),
            protocol=evidence_protocol,
        )
        assert replay.complete_metrics_sha256 == metrics.complete_metrics_sha256
