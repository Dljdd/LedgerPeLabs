"""Residual fail-closed evidence and artifact-cap contracts for Sentinel v5."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

import pytest

import apar.evaluation.v5_evaluation as v5_evaluation
from apar.evaluation.v5_population import V5DecisionRow, build_v5_corpus
from apar.evaluation.v5_protocol import V5Profile
from apar.features.sentinel import (
    SentinelFeatureCatalog,
    SentinelFeatureProvenance,
    build_sentinel_features,
)
from tests.evaluation.v5_safe_protocol import load_safe_v5_test_protocol

ROOT = Path(__file__).resolve().parents[2]
BASE = datetime(2026, 6, 1, 12, 0, tzinfo=UTC)


def _row(
    event_id: str,
    at: datetime,
    *,
    actor_id: str = "actor-shared",
    counterparty_id: str = "counterparty-shared",
) -> V5DecisionRow:
    return V5DecisionRow(
        event_id=event_id,
        payment_id=f"payment-{event_id}",
        campaign_id=f"campaign-{event_id}",
        family="legitimate",
        actor_id=actor_id,
        counterparty_id=counterparty_id,
        amount=Decimal("10.00"),
        decision_at=at,
        is_fraud=False,
        rail="card",
        integrity_status="not_applicable",
        lifecycle_state="authorization",
        source_command_id=f"command-{event_id}",
        source_event_id=event_id,
        execution_evidence_sha256="1" * 64,
        predictive_features={"amount": 10.0},
    )


def test_feature_batch_retains_exact_strictly_prior_source_provenance() -> None:
    """Dropping source IDs or admitting equal-time rows must change this contract."""
    catalog = SentinelFeatureCatalog(
        feature_names=("actor_count_1m",),
        feature_groups=("temporal",),
        catalog_sha256="2" * 64,
    )
    prior = _row("prior", BASE - timedelta(seconds=30))
    unrelated = _row(
        "unrelated",
        BASE - timedelta(seconds=20),
        actor_id="other-actor",
        counterparty_id="other-counterparty",
    )
    current = _row("current", BASE)
    equal_time = _row("equal-time", BASE)

    batch = build_sentinel_features(
        (prior, unrelated, current, equal_time),
        catalog=catalog,
    )

    by_event = {item.event_id: item for item in batch.provenance}
    assert by_event["current"].source_event_ids == ("prior",)
    assert by_event["current"].max_source_available_at == prior.decision_at
    assert by_event["equal-time"].source_event_ids == ("prior",)
    assert "current" not in by_event["equal-time"].source_event_ids
    assert batch.provenance == tuple(batch.provenance)


def test_arm_row_contract_retains_exact_rule_feature_provenance() -> None:
    """A row digest without the feature builder's sources cannot replay RuleEngine."""
    assert {
        "rule_source_event_ids",
        "rule_max_source_available_at",
    } <= set(v5_evaluation.V5ArmRowEvidence.model_fields)


def test_feature_provenance_rejects_non_utc_availability() -> None:
    """A timezone-free source maximum cannot prove knowledge-time ordering."""
    with pytest.raises(ValueError, match="UTC"):
        SentinelFeatureProvenance(
            event_id="target",
            decision_at=BASE,
            source_event_ids=("source",),
            max_source_available_at=datetime(2026, 6, 1, 11, 59),
        )


def test_retained_provenance_validator_rejects_omitted_forged_and_future_sources() -> None:
    """Rehashing a false source list must not make it acceptable evidence."""
    validator = getattr(v5_evaluation, "validate_v5_rule_feature_provenance", None)
    assert callable(validator), "retained RuleEngine provenance validator is missing"

    protocol = load_safe_v5_test_protocol(ROOT)
    corpus = build_v5_corpus(protocol, profile=V5Profile.SMOKE)
    partition = corpus.partitions["development_test"]
    catalog = SentinelFeatureCatalog.default()
    batch = build_sentinel_features(partition.decisions, catalog=catalog)
    support = v5_evaluation.build_v5_arm_support_rows(partition.decisions)
    artifacts = v5_evaluation.build_v5_execution_artifacts(partition.executions)

    validator(
        support=support,
        provenance=batch.provenance,
        catalog=catalog,
        artifacts=artifacts,
    )
    index = next(
        index
        for index, item in enumerate(batch.provenance)
        if item.source_event_ids
    )
    original = batch.provenance[index]
    mutations = (
        original.model_copy(
            update={"source_event_ids": (), "max_source_available_at": None}
        ),
        original.model_copy(
            update={"source_event_ids": ("forged-event",)}
        ),
        original.model_copy(
            update={
                "source_event_ids": (original.event_id,),
                "max_source_available_at": original.decision_at,
            }
        ),
    )
    for mutation in mutations:
        forged = (*batch.provenance[:index], mutation, *batch.provenance[index + 1 :])
        with pytest.raises(ValueError, match="provenance|source|strictly before"):
            validator(
                support=support,
                provenance=forged,
                catalog=catalog,
                artifacts=artifacts,
            )


def test_static_50k_plan_accounts_for_every_declared_execution_artifact() -> None:
    """A filler-only byte sum must not certify the complete production support."""
    from apar.evaluation import v5_population

    protocol = load_safe_v5_test_protocol(ROOT)
    planner = getattr(
        v5_population,
        "_plan_production_development_execution_artifacts",
        None,
    )
    assert callable(planner), "complete development-test artifact planner is missing"

    plan = planner(protocol)
    by_category = plan.artifact_counts_by_category
    assert by_category["legitimate_base"] == 3
    assert by_category["legitimate_filler"] > 0
    for family in (
        "agentic_intent_abuse",
        "app_scam_mule",
        "card_testing_cnp",
        "synthetic_merchant_refund",
    ):
        assert by_category[family] == 100
    assert plan.legitimate_event_count == 50_000
    assert plan.artifact_count == sum(by_category.values())
    assert plan.max_artifact_payload_bytes < 16 * 1024 * 1024
    assert plan.aggregate_payload_bytes < 256 * 1024 * 1024


def test_static_estimators_upper_bound_safe_real_execution_payloads() -> None:
    """Under-estimating any real family/base/filler artifact invalidates the cap proof."""
    from apar.evaluation import v5_population

    estimate = getattr(v5_population, "_estimate_execution_artifact_payload_bytes", None)
    assert callable(estimate), "execution artifact byte estimator is missing"

    protocol = load_safe_v5_test_protocol(ROOT)
    _rows, legitimate = v5_population._execute_legitimate_traffic(
        partition_name="development_test",
        partition_seed=404,
        requested_decisions=512,
    )
    samples = list(legitimate)
    fraud_samples = []
    for family in (
        "agentic_intent_abuse",
        "app_scam_mule",
        "card_testing_cnp",
        "synthetic_merchant_refund",
    ):
        for campaign_index in range(
            protocol.smoke_profile.campaigns_per_family[family]
        ):
            _projected, manifest = v5_population._execute_campaign(
                partition_name="development_test",
                family=family,
                campaign_index=campaign_index,
                partition_seed=404,
            )
            samples.append(manifest)
            fraud_samples.append(manifest)

    production_plan = v5_population._plan_production_development_execution_artifacts(
        protocol
    )
    planned_family_event_counts = {
        artifact.category: artifact.event_count
        for artifact in production_plan.artifacts
        if artifact.category not in {"legitimate_base", "legitimate_filler"}
    }
    assert all(
        len(manifest.lineage) == planned_family_event_counts[manifest.family]
        for manifest in fraud_samples
    )

    for manifest in samples:
        actual = len(manifest.model_dump_json().encode("utf-8"))
        predicted = estimate(manifest)
        assert predicted >= actual, (
            manifest.family,
            len(manifest.lineage),
            predicted,
            actual,
        )
