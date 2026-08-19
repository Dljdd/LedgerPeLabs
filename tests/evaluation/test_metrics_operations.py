"""Hand-calculated workload, alert-time, latency, and bootstrap oracles."""

from __future__ import annotations

import hashlib
import json
from datetime import timedelta
from decimal import Decimal

import numpy as np
import pytest

import apar.evaluation.metrics as metric_module
from apar.cases import QueueConfig, simulate_case_queue
from apar.contracts.decisions import Action
from apar.contracts.events import Rail
from apar.evaluation.metrics import (
    BOOTSTRAP_REPLICATES,
    BOOTSTRAP_SEED,
    AlertMetrics,
    ConfidenceIntervals,
    MetricContractError,
    MetricReportInputs,
    MetricValue,
    campaign_bootstrap,
    compute_metric_report,
)
from apar.runs.wire import canonical_json_bytes
from tests.evaluation.test_metrics_classification import (
    AS_OF,
    NOW,
    decision,
    four_row_inputs,
    latency,
    make_inputs,
    observed,
    truth,
)


def test_workload_queue_fallback_and_latency_metrics_are_hand_calculated() -> None:
    report = compute_metric_report(four_row_inputs())
    assert report.operations.false_challenges_per_10k.value == 0.0
    assert report.operations.false_declines_per_10k.value == 5000.0
    assert report.operations.total_challenges_per_10k.value == 2500.0
    assert report.operations.review_cases_per_100k.value == 75000.0
    assert report.operations.transactions_per_case.value == 1.0
    assert report.operations.entities_per_case.value == 2.0
    assert report.operations.analyst_minutes == 60
    assert report.operations.sla_breaches == 0
    assert report.operations.fallback_count == 1
    assert report.operations.fallback_rate.value == 0.25
    assert report.engineering.end_to_end_ms.p50.value == 10.0
    assert report.engineering.end_to_end_ms.p99.value == 10.0


def test_metric_contracts_reject_reversed_quantiles_and_impossible_queue_counts() -> None:
    alert_document = AlertMetrics(
        campaign_count=20,
        detected_campaigns=20,
        undetected_campaigns=0,
        p50_seconds=MetricValue(value=4.0, numerator=20.0, denominator=1.0),
        p90_seconds=MetricValue(value=5.0, numerator=20.0, denominator=10.0),
        p95_seconds=MetricValue(value=6.0, numerator=20.0, denominator=20.0),
        p99_seconds=MetricValue(
            value=None,
            numerator=20.0,
            denominator=100.0,
            undefined_reason="insufficient_detected_campaigns",
        ),
    ).model_dump(mode="python")
    alert_document["p50_seconds"]["value"] = 10.0
    with pytest.raises(ValueError, match="quantiles.*ordered"):
        AlertMetrics.model_validate(alert_document)

    report = compute_metric_report(four_row_inputs())
    latency_document = report.engineering.end_to_end_ms.model_dump(mode="python")
    latency_document["p50"].update({"value": 11.0, "numerator": 44.0})
    with pytest.raises(ValueError, match="quantiles.*ordered"):
        type(report.engineering.end_to_end_ms).model_validate(latency_document)

    for field_name in ("peak_backlog_count", "sla_breaches"):
        operations_document = report.operations.model_dump(mode="python")
        operations_document[field_name] = report.operations.review_case_count + 1
        with pytest.raises(ValueError, match="review cases"):
            type(report.operations).model_validate(operations_document)


def test_entities_per_case_uses_actor_counterparty_union() -> None:
    same = make_inputs(
        (truth("same", is_fraud=True),),
        (observed("same", actor_id="entity", counterparty_id="entity"),),
        (decision("same", action=Action.CHALLENGE, score=0.9),),
    )
    same_report = compute_metric_report(same)
    assert same_report.operations.case_entity_count == 1
    assert same_report.operations.entities_per_case.value == 1.0

    mixed = make_inputs(
        (
            truth("same", is_fraud=True, campaign_id="campaign-a"),
            truth("distinct", is_fraud=True, campaign_id="campaign-b"),
        ),
        (
            observed("same", actor_id="entity", counterparty_id="entity"),
            observed("distinct", actor_id="actor-b", counterparty_id="counterparty-b"),
        ),
        (
            decision("same", action=Action.CHALLENGE, score=0.9),
            decision("distinct", action=Action.CHALLENGE, score=0.9),
        ),
    )
    mixed_report = compute_metric_report(mixed)
    assert mixed_report.operations.review_case_count == 2
    assert mixed_report.operations.case_entity_count == 3
    assert mixed_report.operations.entities_per_case.value == 1.5


def test_empty_report_keeps_denominator_and_latency_quantiles_undefined() -> None:
    queue = simulate_case_queue((), QueueConfig())
    inputs = MetricReportInputs(
        truth=(),
        observations=(),
        decisions=(),
        cases=(),
        queue_report=queue,
        latency_samples=(),
        as_of=AS_OF,
        slice_assignments=(),
    )
    report = compute_metric_report(inputs)
    assert report.classification.precision.undefined_reason == "no_predicted_positives"
    assert report.operations.total_challenges_per_10k.undefined_reason == "no_decisions"
    assert report.engineering.feature_ms.p50.undefined_reason == "empty_latency_samples"
    assert report.engineering.feature_ms.p99.undefined_reason == "empty_latency_samples"


def test_count_contracts_reject_bool_coercion() -> None:
    undefined = MetricValue(
        value=None,
        numerator=0.0,
        denominator=1.0,
        undefined_reason="insufficient_detected_campaigns",
    )
    with pytest.raises(ValueError, match="exact integer"):
        AlertMetrics(
            campaign_count=True,  # type: ignore[arg-type]
            detected_campaigns=0,
            undetected_campaigns=1,
            p50_seconds=undefined,
            p90_seconds=undefined.model_copy(update={"denominator": 10.0}),
            p95_seconds=undefined.model_copy(update={"denominator": 20.0}),
            p99_seconds=undefined.model_copy(update={"denominator": 100.0}),
        )


def test_alert_quantile_eligibility_and_stable_linear_interpolation() -> None:
    truth_rows = []
    observations = []
    decisions = []
    for index in range(20):
        event_id = f"event-{index:02d}"
        anchor = NOW + timedelta(seconds=index * 100)
        observations.append(observed(event_id, decision_at=anchor))
        truth_rows.append(
            truth(event_id, is_fraud=True, campaign_id=f"campaign-{index:02d}")
        )
        decisions.append(decision(event_id, action=Action.CHALLENGE, score=0.9))
    report = compute_metric_report(
        make_inputs(tuple(truth_rows), tuple(observations), tuple(decisions))
    )
    assert report.alerts.detected_campaigns == 20
    assert report.alerts.p50_seconds.value == 0.0
    assert report.alerts.p90_seconds.value == 0.0
    assert report.alerts.p95_seconds.value == 0.0
    assert report.alerts.p99_seconds.undefined_reason == "insufficient_detected_campaigns"
    assert report.alerts.p99_seconds.numerator == 20.0
    assert report.alerts.p99_seconds.denominator == 100.0


def test_time_to_alert_uses_later_nonapprove_and_linear_percentiles() -> None:
    truth_rows = []
    observations = []
    decisions = []
    for index in range(10):
        campaign = f"campaign-{index:02d}"
        fraud_id = f"fraud-{index:02d}"
        alert_id = f"alert-{index:02d}"
        anchor = NOW + timedelta(minutes=index)
        truth_rows.extend(
            (
                truth(fraud_id, is_fraud=True, campaign_id=campaign),
                truth(alert_id, is_fraud=False, campaign_id=campaign),
            )
        )
        observations.extend(
            (
                observed(fraud_id, decision_at=anchor),
                observed(alert_id, decision_at=anchor + timedelta(seconds=index)),
            )
        )
        decisions.extend(
            (
                decision(fraud_id, action=Action.APPROVE, score=0.1),
                decision(alert_id, action=Action.CHALLENGE, score=0.9),
            )
        )
    report = compute_metric_report(
        make_inputs(tuple(truth_rows), tuple(observations), tuple(decisions))
    )
    assert report.alerts.p50_seconds.value == 4.5
    assert report.alerts.p90_seconds.value == pytest.approx(8.1)
    assert report.alerts.p95_seconds.undefined_reason == "insufficient_detected_campaigns"


def test_campaign_alert_preparation_comparisons_scale_linearly() -> None:
    class CountingCampaignId(str):
        comparisons = 0

        def __eq__(self, other: object) -> bool:
            type(self).comparisons += 1
            return super().__eq__(other)

        def __lt__(self, other: str) -> bool:
            type(self).comparisons += 1
            return super().__lt__(other)

        __hash__ = str.__hash__

    previous: int | None = None
    for size in (1_000, 2_000, 4_000):
        CountingCampaignId.comparisons = 0
        rows = tuple(
            metric_module._MetricRow(  # type: ignore[attr-defined]
                event_id=f"event-{index:04d}",
                campaign_id=CountingCampaignId(f"campaign-{index:04d}"),
                family="card_testing_cnp",
                rail=Rail.CARD,
                is_fraud=True,
                action=Action.APPROVE,
                score=0.1,
                decision_at=NOW + timedelta(seconds=index),
                net_value=Decimal("0.00"),
                first_settlement_at=None,
            )
            for index in range(size)
        )
        metric_module._campaign_alert_times(rows)  # type: ignore[attr-defined]
        comparisons = CountingCampaignId.comparisons
        assert comparisons > 0
        if previous is not None:
            assert comparisons <= previous * 3
        previous = comparisons


def test_undetected_campaign_remains_right_censored_without_artificial_duration() -> None:
    inputs = make_inputs(
        (truth("event", is_fraud=True),),
        (observed("event"),),
        (decision("event", action=Action.APPROVE, score=0.1),),
    )
    report = compute_metric_report(inputs)
    assert report.alerts.campaign_count == 1
    assert report.alerts.detected_campaigns == 0
    assert report.alerts.undetected_campaigns == 1
    assert report.alerts.p50_seconds.value is None
    assert report.alerts.p95_seconds.undefined_reason == "insufficient_detected_campaigns"


def test_latency_requires_exact_decision_bijection_and_coherent_end_to_end() -> None:
    inputs = four_row_inputs()
    with pytest.raises(MetricContractError, match="latency.*bijective"):
        compute_metric_report(
            inputs.model_copy(update={"latency_samples": inputs.latency_samples[:-1]})
        )
    broken = latency("event-a").model_copy(update={"end_to_end_ms": 9.0})
    with pytest.raises(MetricContractError, match="end-to-end"):
        compute_metric_report(
            inputs.model_copy(
                update={"latency_samples": (broken,) + inputs.latency_samples[1:]}
            )
        )


def test_supplied_cases_and_queue_must_exactly_match_causal_nonapprovals() -> None:
    inputs = four_row_inputs()
    with pytest.raises(MetricContractError, match="cases.*queue"):
        compute_metric_report(inputs.model_copy(update={"cases": inputs.cases[:-1]}))
    fewer_decisions = tuple(
        row.model_copy(update={"action": Action.APPROVE})
        if row.event_id == "event-c"
        else row
        for row in inputs.decisions
    )
    with pytest.raises(MetricContractError, match="causal"):
        compute_metric_report(inputs.model_copy(update={"decisions": fewer_decisions}))


def test_campaign_bootstrap_is_clustered_frozen_and_repeatable() -> None:
    first = campaign_bootstrap(four_row_inputs())
    second = campaign_bootstrap(four_row_inputs())
    assert first == second
    assert ConfidenceIntervals.from_json(first.to_json()) == first
    assert first.seed == BOOTSTRAP_SEED
    assert first.replicates == BOOTSTRAP_REPLICATES
    assert {item.metric_name for item in first.intervals} == {
        "precision",
        "recall",
        "f1",
        "false_positive_rate",
        "campaign_recall",
        "fraudulent_net_settled_value",
        "preventable_settled_value",
        "value_escaped",
    }
    assert all(
        item.valid_replicates + item.undefined_replicates == 1000
        for item in first.intervals
    )


def test_recomputed_confidence_checksum_cannot_hide_invalid_interval() -> None:
    document = json.loads(campaign_bootstrap(four_row_inputs()).to_json())
    document["intervals"][0]["lower"] = 2.0
    unsigned = {key: value for key, value in document.items() if key != "intervals_digest"}
    document["intervals_digest"] = hashlib.sha256(
        canonical_json_bytes(unsigned)
    ).hexdigest()
    with pytest.raises(MetricContractError, match="bounds"):
        ConfidenceIntervals.from_json(canonical_json_bytes(document))


def test_recomputed_confidence_checksum_cannot_hide_a_missing_interval() -> None:
    document = json.loads(campaign_bootstrap(four_row_inputs()).to_json())
    document["intervals"] = document["intervals"][:-1]
    unsigned = {key: value for key, value in document.items() if key != "intervals_digest"}
    document["intervals_digest"] = hashlib.sha256(
        canonical_json_bytes(unsigned)
    ).hexdigest()
    with pytest.raises(MetricContractError, match="complete"):
        ConfidenceIntervals.from_json(canonical_json_bytes(document))


def test_recomputed_confidence_checksum_cannot_hide_out_of_domain_bounds() -> None:
    document = json.loads(campaign_bootstrap(four_row_inputs()).to_json())
    precision = next(
        item for item in document["intervals"] if item["metric_name"] == "precision"
    )
    precision.update({"lower": 2.0, "median": 2.0, "upper": 2.0})
    unsigned = {key: value for key, value in document.items() if key != "intervals_digest"}
    document["intervals_digest"] = hashlib.sha256(
        canonical_json_bytes(unsigned)
    ).hexdigest()
    with pytest.raises(MetricContractError, match=r"\[0, 1\]"):
        ConfidenceIntervals.from_json(canonical_json_bytes(document))


def test_bootstrap_intervals_are_rederived_from_campaign_contributions() -> None:
    document = json.loads(campaign_bootstrap(four_row_inputs()).to_json())
    precision = next(
        item for item in document["intervals"] if item["metric_name"] == "precision"
    )
    precision.update({"lower": 0.25, "median": 0.25, "upper": 0.25})
    unsigned = {key: value for key, value in document.items() if key != "intervals_digest"}
    document["intervals_digest"] = hashlib.sha256(
        canonical_json_bytes(unsigned)
    ).hexdigest()
    with pytest.raises(MetricContractError, match="bootstrap derivation"):
        ConfidenceIntervals.from_json(canonical_json_bytes(document))


def test_model_construct_confidence_tamper_is_rederived_before_serialization() -> None:
    confidence = campaign_bootstrap(four_row_inputs())
    changed = confidence.intervals[0].model_copy(
        update={"lower": 0.25, "median": 0.25, "upper": 0.25}
    )
    poisoned = ConfidenceIntervals.model_construct(
        **{
            **confidence.model_dump(mode="python"),
            "intervals": (changed,) + confidence.intervals[1:],
        }
    )
    with pytest.raises(MetricContractError, match="semantic revalidation"):
        poisoned.to_json()


def test_campaign_bootstrap_small_oracle_matches_independent_pcg64_sampling() -> None:
    inputs = make_inputs(
        (
            truth("fraud", is_fraud=True, campaign_id="campaign-a"),
            truth("legit", is_fraud=False, campaign_id="campaign-b"),
        ),
        (observed("fraud"), observed("legit")),
        (
            decision("fraud", action=Action.CHALLENGE, score=0.9),
            decision("legit", action=Action.APPROVE, score=0.1),
        ),
    )
    result = campaign_bootstrap(inputs)
    precision = next(item for item in result.intervals if item.metric_name == "precision")
    generator = np.random.Generator(np.random.PCG64(260816))
    sampled = generator.integers(0, 2, size=(1000, 2), endpoint=False)
    valid = [1.0 for row in sampled if 0 in row]
    assert precision.valid_replicates == len(valid)
    assert precision.undefined_replicates == 1000 - len(valid)
    assert precision.lower == 1.0
    assert precision.median == 1.0
    assert precision.upper == 1.0


@pytest.mark.parametrize(
    ("seed", "replicates", "message"),
    [(1, 1000, "seed"), (260816, 999, "replicate")],
)
def test_campaign_bootstrap_rejects_nonfrozen_parameters(
    seed: int, replicates: int, message: str
) -> None:
    with pytest.raises(MetricContractError, match=message):
        campaign_bootstrap(four_row_inputs(), seed=seed, replicates=replicates)


def test_campaign_bootstrap_is_input_permutation_stable() -> None:
    inputs = four_row_inputs()
    permuted = inputs.model_copy(
        update={
            "truth": tuple(reversed(inputs.truth)),
            "observations": tuple(reversed(inputs.observations)),
            "decisions": tuple(reversed(inputs.decisions)),
            "latency_samples": tuple(reversed(inputs.latency_samples)),
            "slice_assignments": tuple(reversed(inputs.slice_assignments)),
        }
    )
    assert campaign_bootstrap(permuted) == campaign_bootstrap(inputs)


def test_row_resource_cap_is_checked_before_duplicate_expansion() -> None:
    inputs = four_row_inputs()
    oversized = inputs.model_copy(update={"truth": (inputs.truth[0],) * 100_001})
    with pytest.raises(MetricContractError, match="row resource cap"):
        compute_metric_report(oversized)


def test_campaign_resource_cap_is_checked_before_alignment_work() -> None:
    inputs = four_row_inputs()
    campaign_rows = tuple(
        inputs.truth[0].model_copy(
            update={
                "event_id": f"event-{index:05d}",
                "payment_id": f"payment-{index:05d}",
                "campaign_id": f"campaign-{index:05d}",
                "lifecycle_event_ids": (f"event-{index:05d}",),
            }
        )
        for index in range(10_001)
    )
    with pytest.raises(MetricContractError, match="campaign.*resource cap"):
        compute_metric_report(inputs.model_copy(update={"truth": campaign_rows}))
