"""Hand-calculated classification, calibration, and report-contract oracles."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest
from pydantic import ValidationError

from apar.cases import QueueConfig, group_cases, simulate_case_queue
from apar.contracts.decisions import Action
from apar.contracts.events import EventKind, Rail
from apar.defense.contracts import ObservedEvent
from apar.defense.policy import DefenseDecision
from apar.evaluation.contracts import EvaluationTruthRow
from apar.evaluation.metrics import (
    LatencySample,
    MetricContractError,
    MetricReport,
    MetricReportInputs,
    MetricValue,
    SliceAssignment,
    compute_metric_report,
)
from apar.runs.wire import canonical_json_bytes

NOW = datetime(2026, 8, 18, 12, 0, tzinfo=UTC)
AS_OF = NOW + timedelta(days=30)
FAMILIES = (
    "agentic_intent_abuse",
    "app_scam_mule",
    "card_testing_cnp",
    "synthetic_merchant_refund",
)


def observed(
    event_id: str,
    *,
    payment_id: str | None = None,
    rail: Rail = Rail.CARD,
    event_type: EventKind = EventKind.AUTHORIZATION,
    amount: str = "0.00",
    event_time: datetime | None = None,
    available_at: datetime | None = None,
    decision_at: datetime | None = NOW,
    is_decision_point: bool = True,
    currency: str = "USD",
    actor_id: str | None = None,
    counterparty_id: str | None = None,
) -> ObservedEvent:
    occurred = event_time or NOW - timedelta(seconds=2)
    available = available_at or occurred + timedelta(seconds=1)
    return ObservedEvent(
        event_id=event_id,
        payment_id=payment_id or f"payment-{event_id}",
        rail=rail,
        event_type=event_type,
        amount=Decimal(amount),
        currency=currency,
        event_time=occurred,
        available_at=available,
        decision_at=decision_at,
        actor_id=actor_id or f"actor-{event_id}",
        counterparty_id=counterparty_id or f"counterparty-{event_id}",
        optional_refs={},
        integrity_status="not_applicable",
        integrity_reason=None,
        is_decision_point=is_decision_point,
    )


def truth(
    event_id: str,
    *,
    is_fraud: bool,
    campaign_id: str | None = None,
    family: str = "card_testing_cnp",
    payment_id: str | None = None,
    first_settlement_at: datetime | None = None,
    net_settled_value: str = "0.00",
    lifecycle_event_ids: tuple[str, ...] | None = None,
    label_mature_at: datetime = NOW + timedelta(days=7),
) -> EvaluationTruthRow:
    return EvaluationTruthRow(
        event_id=event_id,
        payment_id=payment_id or f"payment-{event_id}",
        campaign_id=campaign_id or f"campaign-{event_id}",
        family=family,
        viewpoint="development",
        is_fraud=is_fraud,
        label_source="population_truth",
        label_mature_at=label_mature_at,
        first_settlement_at=first_settlement_at,
        net_settled_value=Decimal(net_settled_value),
        lifecycle_event_ids=lifecycle_event_ids or (event_id,),
    )


def decision(
    event_id: str,
    *,
    action: Action,
    score: float,
    calibrated_score: float | None = None,
    fallback_used: bool = False,
) -> DefenseDecision:
    from apar.defense.rules import DefenseReason

    return DefenseDecision(
        event_id=event_id,
        action=action,
        score=score,
        rule_score=0.0,
        calibrated_score=(
            None if fallback_used else score if calibrated_score is None else calibrated_score
        ),
        reason_codes=(DefenseReason.MODEL_UNAVAILABLE,) if fallback_used else (),
        evidence_source_ids=(event_id,),
        fallback_used=fallback_used,
        fallback_reason=DefenseReason.MODEL_UNAVAILABLE if fallback_used else None,
        failed_component_version="gbdt-1.0.0" if fallback_used else None,
        latency_ms=5.0,
        policy_version="1.0.0",
    )


def latency(event_id: str, *, scale: float = 1.0) -> LatencySample:
    return LatencySample(
        event_id=event_id,
        feature_ms=1.0 * scale,
        rules_ms=2.0 * scale,
        model_ms=3.0 * scale,
        calibration_policy_ms=4.0 * scale,
        end_to_end_ms=10.0 * scale,
    )


def make_inputs(
    truth_rows: tuple[EvaluationTruthRow, ...],
    observations: tuple[ObservedEvent, ...],
    decisions: tuple[DefenseDecision, ...],
    *,
    as_of: datetime = AS_OF,
    slice_assignments: tuple[SliceAssignment, ...] = (),
    latency_samples: tuple[LatencySample, ...] | None = None,
) -> MetricReportInputs:
    causal_cases = group_cases(observations, decisions, as_of=as_of)
    queue_report = simulate_case_queue(causal_cases, QueueConfig())
    return MetricReportInputs(
        truth=truth_rows,
        observations=observations,
        decisions=decisions,
        cases=queue_report.case_inputs,
        queue_report=queue_report,
        latency_samples=latency_samples
        or tuple(latency(row.event_id) for row in decisions),
        as_of=as_of,
        slice_assignments=slice_assignments,
    )


def four_row_inputs() -> MetricReportInputs:
    event_ids = ("event-a", "event-b", "event-c", "event-d")
    truth_rows = tuple(
        truth(event_id, is_fraud=index < 2, family=FAMILIES[index])
        for index, event_id in enumerate(event_ids)
    )
    observations = tuple(
        observed(event_id, rail=(Rail.CARD, Rail.A2A, Rail.AGENTIC, Rail.CARD)[index])
        for index, event_id in enumerate(event_ids)
    )
    decisions = (
        decision("event-a", action=Action.CHALLENGE, score=0.9),
        decision("event-b", action=Action.DECLINE, score=0.8),
        decision("event-c", action=Action.DECLINE, score=0.7, fallback_used=True),
        decision("event-d", action=Action.APPROVE, score=0.1),
    )
    assignments = tuple(
        SliceAssignment(
            event_id=event_id,
            regime="baseline" if index < 3 else "shifted",
            entity_cohort="cold" if index % 2 == 0 else "returning",
        )
        for index, event_id in enumerate(event_ids)
    )
    return make_inputs(
        truth_rows,
        observations,
        decisions,
        slice_assignments=assignments,
    )


def test_metric_value_preserves_an_explicit_undefined_denominator() -> None:
    metric = MetricValue(
        value=None,
        numerator=0.0,
        denominator=0.0,
        undefined_reason="absent_positive_class",
    )
    assert metric.value is None
    assert metric.denominator == 0.0
    assert metric.undefined_reason == "absent_positive_class"


def test_public_numeric_contracts_reject_bool_and_integer_coercion() -> None:
    with pytest.raises(ValidationError, match="exact finite float"):
        MetricValue(value=1, numerator=1.0, denominator=1.0)  # type: ignore[arg-type]
    with pytest.raises(ValidationError, match="exact finite float"):
        LatencySample(
            event_id="event",
            feature_ms=True,  # type: ignore[arg-type]
            rules_ms=0.0,
            model_ms=0.0,
            calibration_policy_ms=0.0,
            end_to_end_ms=1.0,
        )


def test_hand_calculated_classification_and_decline_only_views() -> None:
    report = compute_metric_report(four_row_inputs())
    assert report.classification.precision.value == pytest.approx(2 / 3)
    assert report.classification.recall.value == 1.0
    assert report.classification.f1.value == pytest.approx(0.8)
    assert report.classification.false_positive_rate.value == 0.5
    assert report.classification.decline_precision.value == 0.5
    assert report.classification.decline_recall.value == 0.5
    assert report.classification.campaign_recall.value == 1.0
    assert report.classification.pr_auc.value == 1.0
    assert report.classification.roc_auc.value == 1.0
    assert report.operations.false_interventions_per_10k.value == 5000.0


def test_calibration_is_hand_calculated_with_frozen_equal_frequency_bins() -> None:
    report = compute_metric_report(four_row_inputs())
    assert report.calibration.brier_score.value == pytest.approx(0.1375)
    assert report.calibration.ece.value == pytest.approx(0.275)
    assert tuple(item.count for item in report.calibration.reliability_bins) == (1, 1, 1, 1)
    assert tuple(item.mean_prediction for item in report.calibration.reliability_bins) == (
        0.1,
        0.7,
        0.8,
        0.9,
    )
    assert tuple(item.observed_frequency for item in report.calibration.reliability_bins) == (
        0.0,
        0.0,
        1.0,
        1.0,
    )


def test_calibration_endpoints_and_score_ties_use_stable_event_id_order() -> None:
    truth_rows = (
        truth("event-a", is_fraud=False),
        truth("event-b", is_fraud=True),
        truth("event-c", is_fraud=False),
        truth("event-d", is_fraud=True),
    )
    observations = tuple(observed(row.event_id) for row in truth_rows)
    decisions = (
        decision("event-a", action=Action.APPROVE, score=0.0),
        decision("event-b", action=Action.CHALLENGE, score=0.0),
        decision("event-c", action=Action.APPROVE, score=1.0),
        decision("event-d", action=Action.DECLINE, score=1.0),
    )
    inputs = make_inputs(truth_rows, observations, decisions)
    forward = compute_metric_report(inputs)
    reverse = compute_metric_report(
        inputs.model_copy(
            update={
                "truth": tuple(reversed(inputs.truth)),
                "observations": tuple(reversed(inputs.observations)),
                "decisions": tuple(reversed(inputs.decisions)),
                "latency_samples": tuple(reversed(inputs.latency_samples)),
            }
        )
    )
    assert forward == reverse
    assert tuple(item.lower_score for item in forward.calibration.reliability_bins) == (
        0.0,
        0.0,
        1.0,
        1.0,
    )
    assert tuple(
        item.observed_frequency for item in forward.calibration.reliability_bins
    ) == (0.0, 1.0, 0.0, 1.0)
    assert forward.calibration.slope.value is not None


def test_calibration_slope_uses_the_frozen_unpenalized_logit_fit() -> None:
    event_ids = ("event-a", "event-b", "event-c", "event-d")
    labels = (False, True, False, True)
    scores = (0.1, 0.4, 0.6, 0.9)
    inputs = make_inputs(
        tuple(
            truth(event_id, is_fraud=label)
            for event_id, label in zip(event_ids, labels, strict=True)
        ),
        tuple(observed(event_id) for event_id in event_ids),
        tuple(
            decision(
                event_id,
                action=Action.CHALLENGE if label else Action.APPROVE,
                score=score,
            )
            for event_id, label, score in zip(event_ids, labels, scores, strict=True)
        ),
    )
    report = compute_metric_report(inputs)
    assert report.calibration.slope.value == pytest.approx(
        0.952184776502211, abs=1e-10
    )
    assert report.calibration.intercept.value == pytest.approx(0.0, abs=1e-10)


def test_absent_classes_and_degenerate_logits_are_explicitly_undefined() -> None:
    inputs = make_inputs(
        (truth("legitimate", is_fraud=False),),
        (observed("legitimate"),),
        (decision("legitimate", action=Action.APPROVE, score=0.5),),
    )
    report = compute_metric_report(inputs)
    assert report.classification.precision.undefined_reason == "no_predicted_positives"
    assert report.classification.recall.undefined_reason == "absent_positive_class"
    assert report.classification.pr_auc.undefined_reason == "absent_positive_class"
    assert report.classification.roc_auc.undefined_reason == "absent_positive_class"
    assert report.calibration.slope.undefined_reason == "absent_class"
    assert report.calibration.intercept.undefined_reason == "absent_class"


def test_family_rail_regime_and_entity_cohort_slices_are_frozen() -> None:
    report = compute_metric_report(four_row_inputs())
    slice_keys = {(item.kind, item.value) for item in report.classification.slices}
    assert {("family", family) for family in FAMILIES} <= slice_keys
    assert {("rail", rail.value) for rail in Rail} <= slice_keys
    assert ("regime", "baseline") in slice_keys
    assert ("entity_cohort", "cold") in slice_keys
    cold = next(
        item
        for item in report.classification.slices
        if item.kind == "entity_cohort" and item.value == "cold"
    )
    assert cold.row_count == 2
    assert cold.precision.value == 0.5


def test_score_not_optional_calibrated_metadata_drives_auc_and_is_permutation_stable() -> None:
    inputs = four_row_inputs()
    altered = tuple(
        row.model_copy(
            update={"calibrated_score": None if row.fallback_used else 1.0 - row.score}
        )
        for row in inputs.decisions
    )
    left = compute_metric_report(inputs.model_copy(update={"decisions": altered}))
    right = compute_metric_report(
        inputs.model_copy(
            update={
                "truth": tuple(reversed(inputs.truth)),
                "observations": tuple(reversed(inputs.observations)),
                "decisions": tuple(reversed(altered)),
                "latency_samples": tuple(reversed(inputs.latency_samples)),
                "slice_assignments": tuple(reversed(inputs.slice_assignments)),
            }
        )
    )
    assert left.classification.pr_auc.value == 1.0
    assert right == left


def test_alignment_maturity_and_optional_slice_assignments_fail_closed() -> None:
    inputs = four_row_inputs()
    future_truth = inputs.truth[0].model_copy(
        update={"label_mature_at": inputs.as_of + timedelta(microseconds=1)}
    )
    with pytest.raises(MetricContractError, match="label.*mature"):
        compute_metric_report(
            inputs.model_copy(update={"truth": (future_truth,) + inputs.truth[1:]})
        )
    with pytest.raises(MetricContractError, match="bijective"):
        compute_metric_report(inputs.model_copy(update={"decisions": inputs.decisions[:-1]}))
    with pytest.raises(MetricContractError, match="slice assignments"):
        compute_metric_report(
            inputs.model_copy(update={"slice_assignments": inputs.slice_assignments[:-1]})
        )


def test_contextual_nondecision_observation_is_allowed_but_lifecycle_overlap_is_not() -> None:
    inputs = four_row_inputs()
    context = observed(
        "context",
        event_type=EventKind.FRAUD_REPORTED,
        decision_at=None,
        is_decision_point=False,
    )
    report = compute_metric_report(
        inputs.model_copy(update={"observations": inputs.observations + (context,)})
    )
    assert report.classification.row_count == 4
    overlapping = inputs.truth[1].model_copy(
        update={"lifecycle_event_ids": (inputs.truth[1].event_id, inputs.truth[0].event_id)}
    )
    with pytest.raises(MetricContractError, match="lifecycle.*overlap"):
        compute_metric_report(
            inputs.model_copy(
                update={"truth": (inputs.truth[0], overlapping) + inputs.truth[2:]}
            )
        )


def test_model_construct_bypass_and_nonfinite_metric_contracts_are_revalidated() -> None:
    inputs = four_row_inputs()
    poisoned = DefenseDecision.model_construct(
        **{**inputs.decisions[0].model_dump(), "score": float("nan")}
    )
    with pytest.raises(MetricContractError, match="semantic revalidation"):
        compute_metric_report(
            inputs.model_copy(update={"decisions": (poisoned,) + inputs.decisions[1:]})
        )
    with pytest.raises(ValueError, match="finite"):
        MetricValue(
            value=float("nan"),
            numerator=0.0,
            denominator=1.0,
            undefined_reason=None,
        )


def test_model_construct_integer_decision_numerics_are_not_silently_coerced() -> None:
    inputs = four_row_inputs()
    poisoned = DefenseDecision.model_construct(
        **{**inputs.decisions[0].model_dump(), "rule_score": 0}
    )
    with pytest.raises(MetricContractError, match="decision numeric fields"):
        compute_metric_report(
            inputs.model_copy(update={"decisions": (poisoned,) + inputs.decisions[1:]})
        )


def test_report_json_is_canonical_digest_bound_and_rejects_recomputed_semantic_tamper() -> None:
    report = compute_metric_report(four_row_inputs())
    payload = report.to_json()
    assert MetricReport.from_json(payload) == report
    assert hashlib.sha256(payload).hexdigest() == report.canonical_digest
    document = json.loads(payload)
    document["classification"]["precision"]["value"] = 0.25
    unsigned = {key: value for key, value in document.items() if key != "report_digest"}
    document["report_digest"] = hashlib.sha256(canonical_json_bytes(unsigned)).hexdigest()
    with pytest.raises(MetricContractError, match="precision"):
        MetricReport.from_json(canonical_json_bytes(document))


def test_recomputed_report_checksum_cannot_hide_calibration_ratio_tamper() -> None:
    document = json.loads(compute_metric_report(four_row_inputs()).to_json())
    document["calibration"]["brier_score"]["value"] = 0.5
    unsigned = {key: value for key, value in document.items() if key != "report_digest"}
    document["report_digest"] = hashlib.sha256(canonical_json_bytes(unsigned)).hexdigest()
    with pytest.raises(MetricContractError, match="Brier"):
        MetricReport.from_json(canonical_json_bytes(document))


def test_recomputed_report_checksum_binds_auc_and_calibration_fit_evidence() -> None:
    complete = json.loads(compute_metric_report(four_row_inputs()).to_json())

    auc = json.loads(json.dumps(complete))
    auc["classification"]["pr_auc"]["numerator"] = 0.0
    unsigned = {key: value for key, value in auc.items() if key != "report_digest"}
    auc["report_digest"] = hashlib.sha256(canonical_json_bytes(unsigned)).hexdigest()
    with pytest.raises(MetricContractError, match="PR-AUC numerator"):
        MetricReport.from_json(canonical_json_bytes(auc))

    fit = json.loads(json.dumps(complete))
    fit["calibration"]["slope"]["denominator"] = 2.0
    unsigned = {key: value for key, value in fit.items() if key != "report_digest"}
    fit["report_digest"] = hashlib.sha256(canonical_json_bytes(unsigned)).hexdigest()
    with pytest.raises(MetricContractError, match="calibration slope"):
        MetricReport.from_json(canonical_json_bytes(fit))


def test_recomputed_report_checksum_cannot_hide_incomplete_or_unbalanced_slices() -> None:
    complete = json.loads(compute_metric_report(four_row_inputs()).to_json())

    incomplete = json.loads(json.dumps(complete))
    incomplete["classification"]["slices"] = [
        item
        for item in incomplete["classification"]["slices"]
        if not (item["kind"] == "family" and item["value"] == FAMILIES[0])
    ]
    unsigned = {key: value for key, value in incomplete.items() if key != "report_digest"}
    incomplete["report_digest"] = hashlib.sha256(
        canonical_json_bytes(unsigned)
    ).hexdigest()
    with pytest.raises(MetricContractError, match="family slices"):
        MetricReport.from_json(canonical_json_bytes(incomplete))

    unbalanced = json.loads(json.dumps(complete))
    family = next(
        item
        for item in unbalanced["classification"]["slices"]
        if item["kind"] == "family" and item["value"] == FAMILIES[0]
    )
    family.update(
        {
            "row_count": 0,
            "fraud_count": 0,
            "legitimate_count": 0,
            "true_positives": 0,
            "false_positives": 0,
            "false_negatives": 0,
            "true_negatives": 0,
            "precision": {
                "value": None,
                "numerator": 0.0,
                "denominator": 0.0,
                "undefined_reason": "no_predicted_positives",
            },
            "recall": {
                "value": None,
                "numerator": 0.0,
                "denominator": 0.0,
                "undefined_reason": "absent_positive_class",
            },
            "f1": {
                "value": None,
                "numerator": 0.0,
                "denominator": 0.0,
                "undefined_reason": "no_positive_truth_or_predictions",
            },
            "false_positive_rate": {
                "value": None,
                "numerator": 0.0,
                "denominator": 0.0,
                "undefined_reason": "absent_legitimate_class",
            },
        }
    )
    unsigned = {key: value for key, value in unbalanced.items() if key != "report_digest"}
    unbalanced["report_digest"] = hashlib.sha256(
        canonical_json_bytes(unsigned)
    ).hexdigest()
    with pytest.raises(MetricContractError, match="family slice rollup"):
        MetricReport.from_json(canonical_json_bytes(unbalanced))


def test_recomputed_report_checksum_cannot_hide_cross_section_count_tamper() -> None:
    document = json.loads(compute_metric_report(four_row_inputs()).to_json())
    document["operations"]["legitimate_count"] = 1
    for name, numerator in (
        ("false_interventions_per_10k", 1.0),
        ("false_challenges_per_10k", 0.0),
        ("false_declines_per_10k", 1.0),
    ):
        document["operations"][name].update(
            {
                "value": numerator * 10_000.0,
                "numerator": numerator,
                "denominator": 1.0,
            }
        )
    unsigned = {key: value for key, value in document.items() if key != "report_digest"}
    document["report_digest"] = hashlib.sha256(canonical_json_bytes(unsigned)).hexdigest()
    with pytest.raises(MetricContractError, match="legitimate counts"):
        MetricReport.from_json(canonical_json_bytes(document))


def test_recomputed_report_checksum_binds_calibration_alert_and_latency_counts() -> None:
    complete = json.loads(compute_metric_report(four_row_inputs()).to_json())

    calibration = json.loads(json.dumps(complete))
    calibration["calibration"]["positive_count"] = 1
    calibration["calibration"]["reliability_bins"][2]["observed_frequency"] = 0.0
    ece_numerator = sum(
        item["count"]
        * abs(item["mean_prediction"] - item["observed_frequency"])
        for item in calibration["calibration"]["reliability_bins"]
    )
    calibration["calibration"]["ece"].update(
        {"value": ece_numerator / 4, "numerator": ece_numerator}
    )
    unsigned = {key: value for key, value in calibration.items() if key != "report_digest"}
    calibration["report_digest"] = hashlib.sha256(
        canonical_json_bytes(unsigned)
    ).hexdigest()
    with pytest.raises(MetricContractError, match="positive counts"):
        MetricReport.from_json(canonical_json_bytes(calibration))

    alert = json.loads(json.dumps(complete))
    alert["alerts"].update(
        {"campaign_count": 1, "detected_campaigns": 1, "undetected_campaigns": 0}
    )
    for name in ("p50_seconds", "p90_seconds", "p95_seconds", "p99_seconds"):
        alert["alerts"][name]["numerator"] = 1.0
    unsigned = {key: value for key, value in alert.items() if key != "report_digest"}
    alert["report_digest"] = hashlib.sha256(canonical_json_bytes(unsigned)).hexdigest()
    with pytest.raises(MetricContractError, match="campaign counts"):
        MetricReport.from_json(canonical_json_bytes(alert))

    latency_counts = json.loads(json.dumps(complete))
    for stage in latency_counts["engineering"].values():
        stage["sample_count"] = 1
        for name in ("p50", "p90", "p95", "p99"):
            stage[name]["denominator"] = 1.0
            stage[name]["numerator"] = stage[name]["value"]
    unsigned = {
        key: value for key, value in latency_counts.items() if key != "report_digest"
    }
    latency_counts["report_digest"] = hashlib.sha256(
        canonical_json_bytes(unsigned)
    ).hexdigest()
    with pytest.raises(MetricContractError, match="latency.*row counts"):
        MetricReport.from_json(canonical_json_bytes(latency_counts))


def test_recomputed_report_checksum_cannot_hide_invalid_reliability_range() -> None:
    document = json.loads(compute_metric_report(four_row_inputs()).to_json())
    first_bin = document["calibration"]["reliability_bins"][0]
    first_bin["lower_score"] = 0.0
    first_bin["upper_score"] = 0.0
    unsigned = {key: value for key, value in document.items() if key != "report_digest"}
    document["report_digest"] = hashlib.sha256(canonical_json_bytes(unsigned)).hexdigest()
    with pytest.raises(MetricContractError, match="mean prediction"):
        MetricReport.from_json(canonical_json_bytes(document))


def test_report_from_json_rejects_noncanonical_and_oversized_payloads() -> None:
    report = compute_metric_report(four_row_inputs())
    pretty = json.dumps(json.loads(report.to_json()), indent=2).encode()
    with pytest.raises(MetricContractError, match="canonical"):
        MetricReport.from_json(pretty)
    with pytest.raises(MetricContractError, match="resource cap"):
        MetricReport.from_json(b" " * (64 * 1024 * 1024 + 1))


def test_exact_public_container_contracts_reject_dict_and_complex_latency() -> None:
    inputs = four_row_inputs()
    with pytest.raises(MetricContractError, match="exact MetricReportInputs"):
        compute_metric_report(inputs.model_dump())  # type: ignore[arg-type]
    bad_latency = LatencySample.model_construct(
        event_id="event-a",
        feature_ms=1 + 2j,
        rules_ms=2.0,
        model_ms=3.0,
        calibration_policy_ms=4.0,
        end_to_end_ms=10.0,
    )
    with pytest.raises(MetricContractError, match="semantic revalidation"):
        compute_metric_report(
            inputs.model_copy(
                update={"latency_samples": (bad_latency,) + inputs.latency_samples[1:]}
            )
        )
