"""Frozen Round 3B evidence-protocol contracts."""

from __future__ import annotations

from pathlib import Path

import pytest

from apar.evaluation.v5_evidence_protocol import (
    V5MetricApplicability,
    load_v5_evidence_protocol,
)

ROOT = Path(__file__).resolve().parents[2]


def test_evidence_protocol_freezes_controls_metrics_economics_and_bootstrap() -> None:
    """A missing declarative contract would permit post-result criteria changes."""
    protocol = load_v5_evidence_protocol(
        ROOT / "config/defense/defense-v5-evidence.json",
        root=ROOT,
    )

    assert protocol.schema_version == "1.2.0"
    assert protocol.protocol_id == "apar-sentinel-v5-development-evidence"
    assert protocol.base_protocol_path == "config/defense/defense-v5-development.json"
    assert protocol.arm_protocol_path == "config/defense/defense-v5-arms.json"
    assert protocol.safe_development_test_seed == 404
    assert protocol.locked_development_test_seed == 2404
    assert (
        protocol.existing_development_result_path
        == "docs/experiments/defense-v5-development-result.json"
    )
    assert (
        protocol.existing_development_result_sha256
        == "af326f3a0fcbbe12c9b8623fc7d82a1ba6d0f327ec9a80f462cacd4bea1dd185"
    )
    assert protocol.controls.label_shuffle.permutation_seed == 1707
    assert protocol.controls.label_shuffle.max_roc_auc == pytest.approx(0.70)
    assert protocol.controls.label_shuffle.max_pr_auc_excess_over_prevalence == pytest.approx(0.20)
    assert protocol.controls.label_shuffle.min_roc_auc_delta == pytest.approx(0.05)
    assert protocol.controls.fraud_only.qualifies_for_readiness is False
    assert protocol.controls.identity_rename.require_exact_numeric_invariance is True
    assert protocol.controls.future_causality.require_exact_invariance is True
    assert protocol.controls.equal_time.require_exact_invariance is True

    assert protocol.calibration.bin_boundaries == tuple(index / 10 for index in range(11))
    assert protocol.calibration.rules_only is V5MetricApplicability.NOT_APPLICABLE
    assert protocol.bootstrap.replicates == 2000
    assert protocol.bootstrap.seed == 707
    assert protocol.bootstrap.confidence_level == pytest.approx(0.95)
    assert protocol.bootstrap.interval_method == "percentile"
    assert protocol.bootstrap.fraud_unit == "campaign"
    assert protocol.bootstrap.legitimate_unit == "campaign"
    assert protocol.bootstrap.stratification == "legitimate_and_each_fraud_family"

    assert protocol.economics.intervention_actions == (
        "challenge",
        "review_hold",
        "decline_hold",
    )
    assert protocol.economics.rail_movement_events == {
        "a2a": ("transfer_posted",),
        "agentic": ("authorization",),
        "card": ("settlement",),
    }
    assert protocol.economics.value_reversal_events == (
        "transfer_returned",
        "refund",
        "recovery",
    )

    assert protocol.bounds.max_rows == 100_000
    assert protocol.bounds.max_execution_artifacts == 4_096
    assert protocol.bounds.max_single_execution_bytes == 16 * 1024 * 1024
    assert protocol.bounds.max_aggregate_execution_bytes == 512 * 1024 * 1024
    assert len(protocol.evidence_protocol_sha256) == 64
    assert len(protocol.base_protocol_sha256) == 64
    assert len(protocol.arm_protocol_sha256) == 64
    assert len(protocol.implementation_sha256) == 64


def test_evidence_protocol_rejects_criterion_or_seed_mutation(tmp_path: Path) -> None:
    """A caller cannot weaken controls or substitute the locked execution seed."""
    source = ROOT / "config/defense/defense-v5-evidence.json"
    document = source.read_text()

    weakened = tmp_path / "weakened.json"
    weakened.write_text(document.replace('"max_roc_auc": 0.70', '"max_roc_auc": 0.95'))
    with pytest.raises(ValueError, match="label-shuffle|criterion|frozen"):
        load_v5_evidence_protocol(weakened, root=ROOT)

    unsafe = tmp_path / "unsafe.json"
    unsafe.write_text(
        document.replace('"safe_development_test_seed": 404', '"safe_development_test_seed": 2404')
    )
    with pytest.raises(ValueError, match="safe.*seed|locked"):
        load_v5_evidence_protocol(unsafe, root=ROOT)
