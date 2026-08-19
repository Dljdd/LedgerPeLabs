"""Exact, deterministic metrics for synthetic APAR defense evaluations.

These metrics measure the frozen synthetic evaluation corpus. They do not establish
external validity, real-world prevalence, or a production operating recommendation.
"""

from __future__ import annotations

import hashlib
import math
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime
from decimal import ROUND_HALF_EVEN, Context, Decimal, DecimalException, localcontext
from typing import Literal, cast

import numpy as np
from numpy.typing import NDArray
from pydantic import Field, ValidationError, field_validator, model_validator
from sklearn.linear_model import LogisticRegression  # type: ignore[import-untyped]
from sklearn.metrics import average_precision_score, roc_auc_score  # type: ignore[import-untyped]

from apar.cases import InvestigationCase, QueueReport, group_cases
from apar.contracts._validation import ExternalContract, validate_utc_timestamp
from apar.contracts.decisions import Action
from apar.contracts.events import EventKind, Rail
from apar.defense.contracts import ObservedEvent
from apar.defense.policy import DefenseDecision
from apar.evaluation.contracts import EvaluationTruthRow
from apar.evaluation.regimes import RegimeKind
from apar.evaluation.splits import EntityCohort
from apar.runs.wire import WireContractError, canonical_json_bytes, strict_json_loads

BOOTSTRAP_SEED = 260816
BOOTSTRAP_REPLICATES = 1000
MAX_METRIC_ROWS = 100_000
MAX_METRIC_CAMPAIGNS = 10_000
MAX_LIFECYCLE_REFERENCES = 1_000_000
MAX_BOOTSTRAP_WORK = 10_000_000
MAX_METRIC_PAYLOAD_BYTES = 64 * 1024 * 1024

_MAX_IDENTIFIER_LENGTH = 4_096
_MAX_DECIMAL_ADJUSTED = 308
_MONEY_QUANTUM = Decimal("0.01")
_DECIMAL128_CONTEXT = Context(
    prec=34,
    rounding=ROUND_HALF_EVEN,
    Emin=-6143,
    Emax=6144,
    capitals=1,
    clamp=1,
)
_CALIBRATION_BINS = 10
_LOGIT_EPSILON = 1e-8
_SETTLEMENT_EVENTS = frozenset({EventKind.SETTLEMENT, EventKind.TRANSFER_POSTED})
_REVERSING_EVENTS = frozenset(
    {
        EventKind.REVERSAL,
        EventKind.TRANSFER_RETURNED,
        EventKind.REFUND,
        EventKind.CHARGEBACK,
        EventKind.RECOVERY,
    }
)
_FAMILIES = (
    "agentic_intent_abuse",
    "app_scam_mule",
    "card_testing_cnp",
    "synthetic_merchant_refund",
)
_REGIME_VALUES = tuple(sorted(("baseline", *(kind.value for kind in RegimeKind))))
_ENTITY_COHORT_VALUES = tuple(sorted(cohort.value for cohort in EntityCohort))
_LIMITATIONS = (
    "Results use authorized synthetic APAR fixtures only.",
    "Campaign-clustered intervals describe variation in this synthetic corpus, "
    "not real populations.",
    "Operating thresholds and workload assumptions are competition evidence, "
    "not production advice.",
    "Value evidence uses one synthetic currency, cent-denominated inputs, and a "
    "frozen Decimal128-style arithmetic context.",
)

MetricName = Literal[
    "precision",
    "recall",
    "f1",
    "false_positive_rate",
    "campaign_recall",
    "fraudulent_net_settled_value",
    "preventable_settled_value",
    "value_escaped",
]
_BOOTSTRAP_METRICS: tuple[MetricName, ...] = (
    "campaign_recall",
    "f1",
    "false_positive_rate",
    "fraudulent_net_settled_value",
    "precision",
    "preventable_settled_value",
    "recall",
    "value_escaped",
)


class MetricContractError(ValueError):
    """Metric inputs, computations, or serialized evidence failed closed."""


class MetricValue(ExternalContract):
    """A finite metric with its evidence counts, or an explicit undefined result."""

    value: float | None
    numerator: float
    denominator: float
    undefined_reason: str | None = None

    @field_validator("value", "numerator", "denominator", mode="before")
    @classmethod
    def values_are_exact_finite_floats(cls, value: object) -> object:
        if value is not None and (type(value) is not float or not math.isfinite(value)):
            raise ValueError("metric values must be exact finite floats")
        return value

    @model_validator(mode="after")
    def defined_state_is_explicit(self) -> MetricValue:
        if self.denominator < 0.0:
            raise ValueError("metric denominator must be nonnegative")
        if (self.value is None) != (self.undefined_reason is not None):
            raise ValueError("undefined metrics require exactly one reason")
        if self.undefined_reason is not None and not self.undefined_reason:
            raise ValueError("undefined metric reason must be nonempty")
        return self


class DecimalMetricValue(ExternalContract):
    """A Decimal-valued ratio for exact synthetic-money aggregates."""

    value: Decimal | None
    numerator: Decimal
    denominator: Decimal
    undefined_reason: str | None = None

    @field_validator("value", "numerator", "denominator")
    @classmethod
    def decimals_are_finite(cls, value: Decimal | None) -> Decimal | None:
        if value is not None:
            _validate_decimal128_number(value, label="Decimal metric value")
        return value

    @model_validator(mode="after")
    def defined_state_is_explicit(self) -> DecimalMetricValue:
        if self.denominator < 0:
            raise ValueError("Decimal metric denominator must be nonnegative")
        if (self.value is None) != (self.undefined_reason is not None):
            raise ValueError("undefined Decimal metrics require exactly one reason")
        if self.undefined_reason is not None and not self.undefined_reason:
            raise ValueError("undefined Decimal metric reason must be nonempty")
        if self.value is not None:
            if self.denominator == 0:
                raise ValueError("defined Decimal metric requires a positive denominator")
            if self.value != _decimal_divide(self.numerator, self.denominator):
                raise ValueError("Decimal metric value must equal numerator / denominator")
        return self


class SliceAssignment(ExternalContract):
    """Evaluator-owned optional regime and entity-cohort labels for one opening."""

    event_id: str
    regime: str
    entity_cohort: str

    @field_validator("event_id", "regime", "entity_cohort")
    @classmethod
    def labels_are_bounded_exact_text(cls, value: str) -> str:
        if type(value) is not str or not value or len(value) > _MAX_IDENTIFIER_LENGTH:
            raise ValueError("slice labels must be bounded nonempty text")
        return value

    @model_validator(mode="after")
    def labels_use_closed_competition_vocabularies(self) -> SliceAssignment:
        if self.regime not in _REGIME_VALUES:
            raise ValueError("slice regime is outside the closed manifest vocabulary")
        if self.entity_cohort not in _ENTITY_COHORT_VALUES:
            raise ValueError("slice entity cohort is outside the closed manifest vocabulary")
        return self


class SliceManifest(ExternalContract):
    """Complete closed evaluator slice vocabularies for one metric report."""

    schema_version: Literal["1.0.0"] = "1.0.0"
    regimes: tuple[str, ...] = _REGIME_VALUES
    entity_cohorts: tuple[str, ...] = _ENTITY_COHORT_VALUES

    @field_validator("regimes", "entity_cohorts", mode="before")
    @classmethod
    def vocabularies_are_exact_tuples(cls, value: object) -> object:
        if type(value) is not tuple:
            raise ValueError("slice manifest vocabularies must be exact tuples")
        return value

    @model_validator(mode="after")
    def vocabularies_are_complete(self) -> SliceManifest:
        if self.regimes != _REGIME_VALUES:
            raise ValueError("slice manifest regimes must be complete and sorted")
        if self.entity_cohorts != _ENTITY_COHORT_VALUES:
            raise ValueError("slice manifest entity cohorts must be complete and sorted")
        return self

    @classmethod
    def closed(cls) -> SliceManifest:
        """Return the frozen competition slice vocabulary."""
        return cls()


class ClassificationSlice(ExternalContract):
    """Promotion-gate classification evidence for one exact slice."""

    kind: Literal["family", "rail", "regime", "entity_cohort"]
    value: str
    row_count: int = Field(ge=0)
    fraud_count: int = Field(ge=0)
    legitimate_count: int = Field(ge=0)
    true_positives: int = Field(ge=0)
    false_positives: int = Field(ge=0)
    false_negatives: int = Field(ge=0)
    true_negatives: int = Field(ge=0)
    decline_true_positives: int = Field(ge=0)
    decline_false_positives: int = Field(ge=0)
    fraud_campaigns: int = Field(ge=0)
    detected_fraud_campaigns: int = Field(ge=0)
    precision: MetricValue
    recall: MetricValue
    f1: MetricValue
    false_positive_rate: MetricValue
    decline_precision: MetricValue
    decline_recall: MetricValue
    campaign_recall: MetricValue
    pr_auc: MetricValue
    roc_auc: MetricValue

    @field_validator("value")
    @classmethod
    def value_is_bounded_nonempty_text(cls, value: str) -> str:
        if type(value) is not str or not value or len(value) > _MAX_IDENTIFIER_LENGTH:
            raise ValueError("classification slice values must be bounded nonempty text")
        return value

    @field_validator(
        "row_count",
        "fraud_count",
        "legitimate_count",
        "true_positives",
        "false_positives",
        "false_negatives",
        "true_negatives",
        "decline_true_positives",
        "decline_false_positives",
        "fraud_campaigns",
        "detected_fraud_campaigns",
        mode="before",
    )
    @classmethod
    def counts_are_exact_integers(cls, value: object) -> object:
        return _exact_integer(value)

    @model_validator(mode="after")
    def counts_and_rates_are_exact(self) -> ClassificationSlice:
        _validate_confusion_fields(self, label="slice")
        _validate_extended_classification_fields(self, label="slice")
        return self


class ClassificationMetrics(ExternalContract):
    """Hand-counted action metrics plus sklearn continuous-score AUCs."""

    row_count: int = Field(ge=0)
    fraud_count: int = Field(ge=0)
    legitimate_count: int = Field(ge=0)
    true_positives: int = Field(ge=0)
    false_positives: int = Field(ge=0)
    false_negatives: int = Field(ge=0)
    true_negatives: int = Field(ge=0)
    decline_true_positives: int = Field(ge=0)
    decline_false_positives: int = Field(ge=0)
    fraud_campaigns: int = Field(ge=0)
    detected_fraud_campaigns: int = Field(ge=0)
    precision: MetricValue
    recall: MetricValue
    f1: MetricValue
    false_positive_rate: MetricValue
    decline_precision: MetricValue
    decline_recall: MetricValue
    campaign_recall: MetricValue
    pr_auc: MetricValue
    roc_auc: MetricValue
    slices: tuple[ClassificationSlice, ...]

    @field_validator(
        "row_count",
        "fraud_count",
        "legitimate_count",
        "true_positives",
        "false_positives",
        "false_negatives",
        "true_negatives",
        "decline_true_positives",
        "decline_false_positives",
        "fraud_campaigns",
        "detected_fraud_campaigns",
        mode="before",
    )
    @classmethod
    def counts_are_exact_integers(cls, value: object) -> object:
        return _exact_integer(value)

    @model_validator(mode="after")
    def counts_and_rates_are_exact(self) -> ClassificationMetrics:
        _validate_confusion_fields(self, label="classification")
        _validate_extended_classification_fields(self, label="classification")
        expected = tuple(sorted(self.slices, key=_slice_key))
        if self.slices != expected or len(set(_slice_key(item) for item in self.slices)) != len(
            self.slices
        ):
            raise ValueError("classification slices must be sorted and unique")
        family_values = tuple(
            item.value for item in self.slices if item.kind == "family"
        )
        if family_values != tuple(sorted(_FAMILIES)):
            raise ValueError("classification family slices must be complete and exact")
        rail_values = tuple(item.value for item in self.slices if item.kind == "rail")
        if rail_values != tuple(sorted(rail.value for rail in Rail)):
            raise ValueError("classification rail slices must be complete and exact")
        for kind in ("family", "rail", "regime", "entity_cohort"):
            selected = tuple(item for item in self.slices if item.kind == kind)
            if selected:
                _validate_slice_rollup(self, selected, kind=kind)
        return self


class ReliabilityBin(ExternalContract):
    """One stable score/event-ID ordered equal-frequency calibration bin."""

    bin_index: int = Field(ge=0, lt=_CALIBRATION_BINS)
    start_rank: int = Field(ge=0)
    stop_rank: int = Field(ge=1)
    count: int = Field(ge=1)
    lower_score: float = Field(ge=0.0, le=1.0)
    upper_score: float = Field(ge=0.0, le=1.0)
    mean_prediction: float = Field(ge=0.0, le=1.0)
    observed_frequency: float = Field(ge=0.0, le=1.0)

    @field_validator("bin_index", "start_rank", "stop_rank", "count", mode="before")
    @classmethod
    def counts_are_exact_integers(cls, value: object) -> object:
        return _exact_integer(value)

    @field_validator(
        "lower_score",
        "upper_score",
        "mean_prediction",
        "observed_frequency",
        mode="before",
    )
    @classmethod
    def values_are_exact_finite_floats(cls, value: object) -> object:
        if type(value) is not float or not math.isfinite(value):
            raise ValueError("reliability values must be exact finite floats")
        return value

    @model_validator(mode="after")
    def rank_range_is_exact(self) -> ReliabilityBin:
        if self.stop_rank - self.start_rank != self.count:
            raise ValueError("reliability-bin ranks must equal count")
        if self.lower_score > self.upper_score:
            raise ValueError("reliability-bin score range is reversed")
        if not self.lower_score <= self.mean_prediction <= self.upper_score:
            raise ValueError("reliability-bin mean prediction must lie within its score range")
        positive_count = self.observed_frequency * self.count
        if not math.isclose(positive_count, round(positive_count), abs_tol=1e-12):
            raise ValueError("reliability-bin observed frequency must encode an exact count")
        return self


class CalibrationMetrics(ExternalContract):
    """Brier, frozen ten-bin ECE, reliability, and calibration fit evidence."""

    row_count: int = Field(ge=0)
    positive_count: int = Field(ge=0)
    brier_score: MetricValue
    ece: MetricValue
    reliability_bins: tuple[ReliabilityBin, ...]
    slope: MetricValue
    intercept: MetricValue

    @field_validator("row_count", "positive_count", mode="before")
    @classmethod
    def counts_are_exact_integers(cls, value: object) -> object:
        return _exact_integer(value)

    @model_validator(mode="after")
    def evidence_is_consistent(self) -> CalibrationMetrics:
        if self.positive_count > self.row_count:
            raise ValueError("calibration positives exceed row count")
        for label, metric in (("Brier", self.brier_score), ("ECE", self.ece)):
            if metric.denominator != float(self.row_count):
                raise ValueError(f"{label} denominator must equal row count")
            if metric.value is not None and not 0.0 <= metric.value <= 1.0:
                raise ValueError(f"{label} must be in [0, 1]")
            if self.row_count == 0:
                if metric.value is not None or metric.undefined_reason != "no_decisions":
                    raise ValueError(f"empty {label} must be explicitly undefined")
            elif (
                metric.value is None
                or metric.undefined_reason is not None
                or not math.isclose(
                    metric.value,
                    metric.numerator / metric.denominator,
                    abs_tol=1e-15,
                )
            ):
                raise ValueError(f"{label} value must equal numerator / denominator")
        if self.row_count == 0:
            if self.reliability_bins:
                raise ValueError("empty calibration cannot contain reliability bins")
        else:
            if sum(item.count for item in self.reliability_bins) != self.row_count:
                raise ValueError("reliability-bin counts must cover all rows")
            if tuple(item.bin_index for item in self.reliability_bins) != tuple(
                range(len(self.reliability_bins))
            ):
                raise ValueError("reliability bins must have contiguous indices")
            if tuple(item.start_rank for item in self.reliability_bins) != (
                0,
                *(item.stop_rank for item in self.reliability_bins[:-1]),
            ):
                raise ValueError("reliability bins must cover contiguous stable ranks")
            if self.reliability_bins[-1].stop_rank != self.row_count:
                raise ValueError("reliability bins must end at row count")
            if any(
                left.upper_score > right.lower_score
                for left, right in zip(
                    self.reliability_bins,
                    self.reliability_bins[1:],
                    strict=False,
                )
            ):
                raise ValueError("reliability-bin score ranges must be nondecreasing")
            positives = sum(
                item.observed_frequency * item.count for item in self.reliability_bins
            )
            if not math.isclose(positives, self.positive_count, abs_tol=1e-12):
                raise ValueError("reliability bins must preserve positive count")
            expected_ece_numerator = math.fsum(
                item.count * abs(item.mean_prediction - item.observed_frequency)
                for item in self.reliability_bins
            )
            if not math.isclose(
                self.ece.numerator, expected_ece_numerator, abs_tol=1e-12
            ):
                raise ValueError("ECE numerator must match reliability bins")
        if self.positive_count in {0, self.row_count}:
            fit_reason = "absent_class"
        elif self.reliability_bins and len(
            {
                score
                for item in self.reliability_bins
                for score in (item.lower_score, item.upper_score)
            }
        ) == 1:
            fit_reason = "degenerate_logits"
        else:
            fit_reason = None
        for label, metric in (("slope", self.slope), ("intercept", self.intercept)):
            if fit_reason is not None:
                if (
                    metric.value is not None
                    or metric.numerator != 0.0
                    or metric.denominator != float(self.row_count)
                    or metric.undefined_reason != fit_reason
                ):
                    raise ValueError(
                        f"calibration {label} must expose its undefined fit evidence"
                    )
            elif (
                metric.value is None
                or metric.numerator != metric.value
                or metric.denominator != 1.0
                or metric.undefined_reason is not None
            ):
                raise ValueError(f"calibration {label} must bind its fitted coefficient")
        return self


class ValueMetrics(ExternalContract):
    """Exact lifecycle-netted synthetic value and workload-normalized value."""

    currency: str | None
    fraudulent_net_settled_value: Decimal
    preventable_settled_value: Decimal
    value_escaped: Decimal
    value_before_first_alert: Decimal
    remaining_preventable_at_alert: Decimal
    challenge_credited_as_prevented: Decimal = Decimal("0.00")
    review_case_count: int = Field(ge=0)
    analyst_minutes: int = Field(ge=0)
    captured_value_per_review_case: DecimalMetricValue
    captured_value_per_analyst_hour: DecimalMetricValue

    @field_validator("review_case_count", "analyst_minutes", mode="before")
    @classmethod
    def counts_are_exact_integers(cls, value: object) -> object:
        return _exact_integer(value)

    @field_validator(
        "fraudulent_net_settled_value",
        "preventable_settled_value",
        "value_escaped",
        "value_before_first_alert",
        "remaining_preventable_at_alert",
        "challenge_credited_as_prevented",
    )
    @classmethod
    def money_is_finite(cls, value: Decimal) -> Decimal:
        _validate_cent_amount(value, label="value metric amount")
        return value

    @model_validator(mode="after")
    def value_identity_is_exact(self) -> ValueMetrics:
        if self.currency is not None and (type(self.currency) is not str or not self.currency):
            raise ValueError("metric currency must be nonempty text or None")
        if min(
            self.fraudulent_net_settled_value,
            self.preventable_settled_value,
            self.value_escaped,
            self.value_before_first_alert,
            self.remaining_preventable_at_alert,
        ) < 0:
            raise ValueError("value metrics must be nonnegative")
        if self.preventable_settled_value > self.fraudulent_net_settled_value:
            raise ValueError("preventable value exceeds fraudulent net settled value")
        if self.value_escaped != _money_subtract(
            self.fraudulent_net_settled_value, self.preventable_settled_value
        ):
            raise ValueError("value escaped must equal total less prevented")
        if self.challenge_credited_as_prevented != 0:
            raise ValueError("unsupported challenge counterfactual must receive zero credit")
        _assert_decimal_metric(
            self.captured_value_per_review_case,
            self.preventable_settled_value,
            Decimal(self.review_case_count),
            "no_review_cases",
            label="captured value per review case",
        )
        _assert_decimal_metric(
            self.captured_value_per_analyst_hour,
            _decimal_multiply(self.preventable_settled_value, Decimal(60)),
            Decimal(self.analyst_minutes),
            "no_analyst_minutes",
            label="captured value per analyst hour",
        )
        return self


class AlertMetrics(ExternalContract):
    """Right-censored campaign alert-time evidence with frozen eligibility."""

    campaign_count: int = Field(ge=0)
    detected_campaigns: int = Field(ge=0)
    undetected_campaigns: int = Field(ge=0)
    p50_seconds: MetricValue
    p90_seconds: MetricValue
    p95_seconds: MetricValue
    p99_seconds: MetricValue

    @field_validator(
        "campaign_count", "detected_campaigns", "undetected_campaigns", mode="before"
    )
    @classmethod
    def counts_are_exact_integers(cls, value: object) -> object:
        return _exact_integer(value)

    @model_validator(mode="after")
    def campaign_counts_and_quantiles_are_exact(self) -> AlertMetrics:
        if self.campaign_count != self.detected_campaigns + self.undetected_campaigns:
            raise ValueError("alert campaign counts must balance")
        for label, metric, minimum in (
            ("p50", self.p50_seconds, 1),
            ("p90", self.p90_seconds, 10),
            ("p95", self.p95_seconds, 20),
            ("p99", self.p99_seconds, 100),
        ):
            if metric.numerator != float(self.detected_campaigns):
                raise ValueError(f"alert {label} numerator must count detected campaigns")
            if metric.denominator != float(minimum):
                raise ValueError(f"alert {label} denominator must record eligibility")
            if self.detected_campaigns < minimum:
                if metric.undefined_reason != "insufficient_detected_campaigns":
                    raise ValueError(f"alert {label} must be explicitly ineligible")
            elif metric.value is None or metric.value < 0.0:
                raise ValueError(f"eligible alert {label} must be nonnegative")
        defined_quantiles = tuple(
            metric.value
            for metric in (
                self.p50_seconds,
                self.p90_seconds,
                self.p95_seconds,
                self.p99_seconds,
            )
            if metric.value is not None
        )
        if any(
            left > right
            for left, right in zip(defined_quantiles, defined_quantiles[1:], strict=False)
        ):
            raise ValueError("alert quantiles must be ordered")
        return self


class OperationalMetrics(ExternalContract):
    """Exact customer-friction and deterministic review-workload evidence."""

    decision_count: int = Field(ge=0)
    legitimate_count: int = Field(ge=0)
    false_intervention_count: int = Field(ge=0)
    false_challenge_count: int = Field(ge=0)
    false_decline_count: int = Field(ge=0)
    challenge_count: int = Field(ge=0)
    review_case_count: int = Field(ge=0)
    case_transaction_count: int = Field(ge=0)
    case_entity_count: int = Field(ge=0)
    analyst_minutes: int = Field(ge=0)
    peak_backlog_count: int = Field(ge=0)
    sla_breaches: int = Field(ge=0)
    fallback_count: int = Field(ge=0)
    false_interventions_per_10k: MetricValue
    false_challenges_per_10k: MetricValue
    false_declines_per_10k: MetricValue
    total_challenges_per_10k: MetricValue
    review_cases_per_100k: MetricValue
    transactions_per_case: MetricValue
    entities_per_case: MetricValue
    fallback_rate: MetricValue

    @field_validator(
        "decision_count",
        "legitimate_count",
        "false_intervention_count",
        "false_challenge_count",
        "false_decline_count",
        "challenge_count",
        "review_case_count",
        "case_transaction_count",
        "case_entity_count",
        "analyst_minutes",
        "peak_backlog_count",
        "sla_breaches",
        "fallback_count",
        mode="before",
    )
    @classmethod
    def counts_are_exact_integers(cls, value: object) -> object:
        return _exact_integer(value)

    @model_validator(mode="after")
    def workload_rates_are_exact(self) -> OperationalMetrics:
        if self.false_challenge_count + self.false_decline_count != self.false_intervention_count:
            raise ValueError("false intervention severities must sum to all false interventions")
        if self.false_intervention_count > self.legitimate_count:
            raise ValueError("false interventions exceed legitimate rows")
        if self.challenge_count > self.decision_count:
            raise ValueError("challenges exceed decisions")
        if self.fallback_count > self.decision_count:
            raise ValueError("fallbacks exceed decisions")
        if self.peak_backlog_count > self.review_case_count:
            raise ValueError("peak backlog exceeds review cases")
        if self.sla_breaches > self.review_case_count:
            raise ValueError("SLA breaches exceed review cases")
        if self.review_case_count == 0 and any(
            (
                self.case_transaction_count,
                self.case_entity_count,
                self.analyst_minutes,
                self.peak_backlog_count,
                self.sla_breaches,
            )
        ):
            raise ValueError("empty review workload must have zero direct counts")
        if self.review_case_count and self.case_transaction_count < self.review_case_count:
            raise ValueError("case transactions cannot be fewer than review cases")
        _assert_scaled_metric(
            self.false_interventions_per_10k,
            self.false_intervention_count,
            self.legitimate_count,
            10_000,
            "absent_legitimate_class",
            label="false interventions per 10k",
        )
        _assert_scaled_metric(
            self.false_challenges_per_10k,
            self.false_challenge_count,
            self.legitimate_count,
            10_000,
            "absent_legitimate_class",
            label="false challenges per 10k",
        )
        _assert_scaled_metric(
            self.false_declines_per_10k,
            self.false_decline_count,
            self.legitimate_count,
            10_000,
            "absent_legitimate_class",
            label="false declines per 10k",
        )
        _assert_scaled_metric(
            self.total_challenges_per_10k,
            self.challenge_count,
            self.decision_count,
            10_000,
            "no_decisions",
            label="total challenges per 10k",
        )
        _assert_scaled_metric(
            self.review_cases_per_100k,
            self.review_case_count,
            self.decision_count,
            100_000,
            "no_decisions",
            label="review cases per 100k",
        )
        _assert_metric(
            self.transactions_per_case,
            self.case_transaction_count,
            self.review_case_count,
            "no_review_cases",
            label="transactions per case",
        )
        _assert_metric(
            self.entities_per_case,
            self.case_entity_count,
            self.review_case_count,
            "no_review_cases",
            label="entities per case",
        )
        _assert_metric(
            self.fallback_rate,
            self.fallback_count,
            self.decision_count,
            "no_decisions",
            label="fallback rate",
        )
        return self


class LatencySample(ExternalContract):
    """One exact synchronous replay latency breakdown in milliseconds."""

    event_id: str
    feature_ms: float = Field(ge=0.0)
    rules_ms: float = Field(ge=0.0)
    model_ms: float = Field(ge=0.0)
    calibration_policy_ms: float = Field(ge=0.0)
    end_to_end_ms: float = Field(ge=0.0)

    @field_validator("event_id")
    @classmethod
    def event_id_is_bounded_text(cls, value: str) -> str:
        if type(value) is not str or not value or len(value) > _MAX_IDENTIFIER_LENGTH:
            raise ValueError("latency event ID must be bounded nonempty text")
        return value

    @field_validator(
        "feature_ms",
        "rules_ms",
        "model_ms",
        "calibration_policy_ms",
        "end_to_end_ms",
        mode="before",
    )
    @classmethod
    def stages_are_exact_finite_floats(cls, value: object) -> object:
        if type(value) is not float or not math.isfinite(value):
            raise ValueError("latency stages must be exact finite floats")
        return value

    @model_validator(mode="after")
    def end_to_end_covers_all_stages(self) -> LatencySample:
        stage_total = self.feature_ms + self.rules_ms + self.model_ms + self.calibration_policy_ms
        if self.end_to_end_ms < stage_total:
            raise ValueError("end-to-end latency must cover every sequential stage")
        return self


class LatencyQuantiles(ExternalContract):
    """Frozen linear-interpolation latency quantiles."""

    sample_count: int = Field(ge=0)
    p50: MetricValue
    p90: MetricValue
    p95: MetricValue
    p99: MetricValue

    @field_validator("sample_count", mode="before")
    @classmethod
    def sample_count_is_an_exact_integer(cls, value: object) -> object:
        return _exact_integer(value)

    @model_validator(mode="after")
    def quantiles_have_exact_evidence(self) -> LatencyQuantiles:
        for label, metric in (
            ("p50", self.p50),
            ("p90", self.p90),
            ("p95", self.p95),
            ("p99", self.p99),
        ):
            if self.sample_count == 0:
                if (
                    metric.value is not None
                    or metric.undefined_reason != "empty_latency_samples"
                    or metric.numerator != 0.0
                    or metric.denominator != 0.0
                ):
                    raise ValueError(f"empty latency {label} must be explicitly undefined")
            elif (
                metric.value is None
                or metric.value < 0.0
                or metric.numerator != metric.value * self.sample_count
                or metric.denominator != float(self.sample_count)
            ):
                raise ValueError(f"latency {label} must bind its sample count")
        if self.sample_count and not (
            cast(float, self.p50.value)
            <= cast(float, self.p90.value)
            <= cast(float, self.p95.value)
            <= cast(float, self.p99.value)
        ):
            raise ValueError("latency quantiles must be ordered")
        return self


class EngineeringMetrics(ExternalContract):
    """Per-stage and end-to-end replay latency evidence."""

    feature_ms: LatencyQuantiles
    rules_ms: LatencyQuantiles
    model_ms: LatencyQuantiles
    calibration_policy_ms: LatencyQuantiles
    end_to_end_ms: LatencyQuantiles

    @model_validator(mode="after")
    def stage_counts_match(self) -> EngineeringMetrics:
        counts = {
            self.feature_ms.sample_count,
            self.rules_ms.sample_count,
            self.model_ms.sample_count,
            self.calibration_policy_ms.sample_count,
            self.end_to_end_ms.sample_count,
        }
        if len(counts) != 1:
            raise ValueError("all latency stages must bind the same samples")
        return self


class MetricReportInputs(ExternalContract):
    """Closed evaluator inputs retained as canonical metric derivation evidence."""

    truth: tuple[EvaluationTruthRow, ...]
    observations: tuple[ObservedEvent, ...]
    decisions: tuple[DefenseDecision, ...]
    cases: tuple[InvestigationCase, ...]
    queue_report: QueueReport
    latency_samples: tuple[LatencySample, ...]
    as_of: datetime
    slice_assignments: tuple[SliceAssignment, ...] = ()
    slice_manifest: SliceManifest | None = None

    @field_validator(
        "truth",
        "observations",
        "decisions",
        "cases",
        "latency_samples",
        "slice_assignments",
        mode="before",
    )
    @classmethod
    def containers_are_exact_tuples(cls, value: object) -> object:
        if type(value) is not tuple:
            raise ValueError("metric input collections must be exact tuples")
        return value

    @field_validator("as_of")
    @classmethod
    def as_of_is_utc(cls, value: datetime) -> datetime:
        if type(value) is not datetime:
            raise ValueError("metric as_of must be an exact datetime")
        return validate_utc_timestamp(value)


class MetricDerivationEvidence(ExternalContract):
    """Exact truth-bearing evaluator inputs used to rederive every report section."""

    schema_version: Literal["1.0.0"] = "1.0.0"
    inputs: MetricReportInputs


class MetricReport(ExternalContract):
    """Canonical truth-bearing evaluator evidence requiring redaction for publication."""

    schema_version: Literal["1.0.0"] = "1.0.0"
    evaluator_as_of: datetime
    evaluator_input_digest: str
    classification: ClassificationMetrics
    calibration: CalibrationMetrics
    value: ValueMetrics
    alerts: AlertMetrics
    operations: OperationalMetrics
    engineering: EngineeringMetrics
    derivation_evidence: MetricDerivationEvidence
    synthetic_only: Literal[True] = True
    external_validity_claimed: Literal[False] = False
    limitations: tuple[str, ...] = _LIMITATIONS
    report_digest: str

    @field_validator("evaluator_as_of")
    @classmethod
    def as_of_is_utc(cls, value: datetime) -> datetime:
        return validate_utc_timestamp(value)

    @field_validator("evaluator_input_digest", "report_digest")
    @classmethod
    def digests_are_sha256(cls, value: str) -> str:
        _validate_sha256(value)
        return value

    @model_validator(mode="after")
    def report_is_semantically_bound(self) -> MetricReport:
        if self.limitations != _LIMITATIONS:
            raise ValueError("metric limitations must remain frozen")
        expected = _digest_document(self.model_dump(mode="json", exclude={"report_digest"}))
        if self.report_digest != expected:
            raise ValueError("metric report digest is inconsistent")
        if self.classification.row_count != self.operations.decision_count:
            raise ValueError("classification and operations row counts differ")
        if self.calibration.row_count != self.classification.row_count:
            raise ValueError("calibration and classification row counts differ")
        if self.calibration.positive_count != self.classification.fraud_count:
            raise ValueError("calibration and classification positive counts differ")
        if self.operations.legitimate_count != self.classification.legitimate_count:
            raise ValueError("operations and classification legitimate counts differ")
        if self.operations.false_intervention_count != self.classification.false_positives:
            raise ValueError("operations and classification false interventions differ")
        if self.alerts.campaign_count != self.classification.fraud_campaigns:
            raise ValueError("alert and classification campaign counts differ")
        if self.alerts.detected_campaigns != self.classification.detected_fraud_campaigns:
            raise ValueError("alert and classification detected campaigns differ")
        if self.operations.review_case_count != self.value.review_case_count:
            raise ValueError("value and operations review-case counts differ")
        if self.operations.analyst_minutes != self.value.analyst_minutes:
            raise ValueError("value and operations analyst minutes differ")
        if self.engineering.end_to_end_ms.sample_count != self.classification.row_count:
            raise ValueError("latency and classification row counts differ")
        try:
            derived = _derive_metric_sections(self.derivation_evidence.inputs)
        except (ArithmeticError, MemoryError, OverflowError, TypeError, ValueError) as error:
            raise ValueError("metric report derivation evidence is invalid") from error
        if self.evaluator_as_of != derived.inputs.as_of:
            raise ValueError("metric report derivation as_of differs")
        expected_input_digest = _input_digest(derived.inputs)
        if self.evaluator_input_digest != expected_input_digest:
            raise ValueError("metric report derivation input digest differs")
        for label, actual, expected_section in (
            ("classification", self.classification, derived.classification),
            ("calibration", self.calibration, derived.calibration),
            ("value", self.value, derived.value),
            ("alerts", self.alerts, derived.alerts),
            ("operations", self.operations, derived.operations),
            ("engineering", self.engineering, derived.engineering),
        ):
            if actual != expected_section:
                raise ValueError(f"metric report derivation differs for {label}")
        return self

    def to_json(self) -> bytes:
        """Return canonical report bytes after fresh semantic revalidation."""
        if type(self) is not MetricReport:
            raise MetricContractError("metric report must be an exact MetricReport")
        try:
            checked = MetricReport.model_validate(
                self.model_dump(mode="python", warnings=False), strict=True
            )
            payload = canonical_json_bytes(checked.model_dump(mode="json"))
        except MetricContractError:
            raise
        except (
            ArithmeticError,
            AttributeError,
            MemoryError,
            OverflowError,
            TypeError,
            ValidationError,
            ValueError,
        ) as error:
            raise MetricContractError(
                "metric report failed semantic revalidation or derivation recheck"
            ) from error
        if len(payload) > MAX_METRIC_PAYLOAD_BYTES:
            raise MetricContractError("metric report payload exceeds resource cap")
        return payload

    @classmethod
    def from_json(cls, payload: bytes) -> MetricReport:
        """Load only bounded canonical JSON with complete semantic validation."""
        if type(payload) is not bytes:
            raise MetricContractError("metric report payload must be exact bytes")
        if len(payload) > MAX_METRIC_PAYLOAD_BYTES:
            raise MetricContractError("metric report payload exceeds resource cap")
        try:
            document = strict_json_loads(payload)
            if type(document) is not dict:
                raise MetricContractError("metric report JSON must contain an object")
            _restore_metric_report_json_tuples(document)
            return cls.model_validate(document)
        except MetricContractError:
            raise
        except (
            ArithmeticError,
            MemoryError,
            OverflowError,
            ValidationError,
            WireContractError,
        ) as error:
            raise MetricContractError(str(error)) from error

    @property
    def canonical_digest(self) -> str:
        """Hash the complete canonical report envelope."""
        return hashlib.sha256(self.to_json()).hexdigest()


class ConfidenceInterval(ExternalContract):
    """Frozen 95% percentile interval over whole-campaign bootstrap replicates."""

    metric_name: MetricName
    lower: float | None
    median: float | None
    upper: float | None
    valid_replicates: int = Field(ge=0, le=BOOTSTRAP_REPLICATES)
    undefined_replicates: int = Field(ge=0, le=BOOTSTRAP_REPLICATES)
    undefined_reason: str | None = None

    @field_validator("valid_replicates", "undefined_replicates", mode="before")
    @classmethod
    def counts_are_exact_integers(cls, value: object) -> object:
        return _exact_integer(value)

    @field_validator("lower", "median", "upper", mode="before")
    @classmethod
    def bounds_are_finite(cls, value: object) -> object:
        if value is not None and (type(value) is not float or not math.isfinite(value)):
            raise ValueError("confidence bounds must be exact finite floats")
        return value

    @model_validator(mode="after")
    def interval_is_coherent(self) -> ConfidenceInterval:
        if self.valid_replicates + self.undefined_replicates != BOOTSTRAP_REPLICATES:
            raise ValueError("bootstrap replicate counts must sum to the frozen total")
        bounds = (self.lower, self.median, self.upper)
        if self.valid_replicates == 0:
            if any(value is not None for value in bounds):
                raise ValueError("undefined bootstrap interval cannot claim bounds")
            if self.undefined_reason != "no_defined_bootstrap_replicates":
                raise ValueError("undefined bootstrap interval requires its exact reason")
        else:
            if any(value is None for value in bounds) or self.undefined_reason is not None:
                raise ValueError("defined bootstrap interval requires all three bounds")
            lower, median, upper = cast(tuple[float, float, float], bounds)
            if not lower <= median <= upper:
                raise ValueError("confidence interval bounds must be ordered")
            if self.metric_name in {
                "precision",
                "recall",
                "f1",
                "false_positive_rate",
                "campaign_recall",
            } and not (0.0 <= lower <= median <= upper <= 1.0):
                raise ValueError("rate confidence interval bounds must be in [0, 1]")
            if self.metric_name in {
                "fraudulent_net_settled_value",
                "preventable_settled_value",
                "value_escaped",
            } and lower < 0.0:
                raise ValueError("value confidence interval bounds must be nonnegative")
        return self


class CampaignBootstrapContribution(ExternalContract):
    """Compact exact contribution for one whole-campaign bootstrap unit."""

    campaign_id: str
    true_positives: int = Field(ge=0)
    false_positives: int = Field(ge=0)
    false_negatives: int = Field(ge=0)
    true_negatives: int = Field(ge=0)
    fraud_campaign: int = Field(ge=0, le=1)
    detected_campaign: int = Field(ge=0, le=1)
    fraudulent_value: Decimal
    preventable_value: Decimal

    @field_validator("campaign_id")
    @classmethod
    def campaign_id_is_bounded_text(cls, value: str) -> str:
        if type(value) is not str or not value or len(value) > _MAX_IDENTIFIER_LENGTH:
            raise ValueError("bootstrap campaign ID must be bounded nonempty text")
        return value

    @field_validator(
        "true_positives",
        "false_positives",
        "false_negatives",
        "true_negatives",
        "fraud_campaign",
        "detected_campaign",
        mode="before",
    )
    @classmethod
    def counts_are_exact_integers(cls, value: object) -> object:
        return _exact_integer(value)

    @field_validator("fraudulent_value", "preventable_value")
    @classmethod
    def values_are_cent_denominated(cls, value: Decimal) -> Decimal:
        _validate_cent_amount(value, label="bootstrap campaign value")
        return value

    @model_validator(mode="after")
    def contribution_is_coherent(self) -> CampaignBootstrapContribution:
        if self.detected_campaign > self.fraud_campaign:
            raise ValueError("detected bootstrap campaign must be fraudulent")
        if self.preventable_value > self.fraudulent_value:
            raise ValueError("preventable bootstrap value exceeds fraudulent value")
        return self


class ConfidenceIntervals(ExternalContract):
    """Canonical frozen campaign-clustered uncertainty evidence."""

    schema_version: Literal["1.0.0"] = "1.0.0"
    resampling_unit: Literal["campaign"] = "campaign"
    generator: Literal["numpy.Generator(PCG64)"] = "numpy.Generator(PCG64)"
    percentile_method: Literal["linear"] = "linear"
    confidence_level: float = Field(default=0.95, ge=0.0, le=1.0)
    seed: Literal[260816] = 260816
    replicates: Literal[1000] = 1000
    evaluator_input_digest: str
    intervals: tuple[ConfidenceInterval, ...]
    campaign_contributions: tuple[CampaignBootstrapContribution, ...]
    synthetic_only: Literal[True] = True
    external_validity_claimed: Literal[False] = False
    intervals_digest: str

    @field_validator("evaluator_input_digest", "intervals_digest")
    @classmethod
    def digests_are_sha256(cls, value: str) -> str:
        _validate_sha256(value)
        return value

    @model_validator(mode="after")
    def intervals_are_complete_and_bound(self) -> ConfidenceIntervals:
        if type(self.confidence_level) is not float or self.confidence_level != 0.95:
            raise ValueError("confidence level must equal the frozen 0.95")
        names = tuple(item.metric_name for item in self.intervals)
        if names != _BOOTSTRAP_METRICS:
            raise ValueError("confidence intervals must be complete, sorted, and unique")
        campaign_ids = tuple(item.campaign_id for item in self.campaign_contributions)
        if campaign_ids != tuple(sorted(campaign_ids)) or len(set(campaign_ids)) != len(
            campaign_ids
        ):
            raise ValueError("bootstrap campaign contributions must be sorted and unique")
        if len(campaign_ids) > MAX_METRIC_CAMPAIGNS:
            raise ValueError("bootstrap campaign contribution count exceeds resource cap")
        if len(campaign_ids) * self.replicates > MAX_BOOTSTRAP_WORK:
            raise ValueError("bootstrap contribution replay exceeds frozen work cap")
        try:
            expected_intervals = _bootstrap_intervals(
                self.campaign_contributions,
                seed=self.seed,
                replicates=self.replicates,
            )
        except (ArithmeticError, MemoryError, OverflowError, TypeError, ValueError) as error:
            raise ValueError("bootstrap derivation evidence is invalid") from error
        if self.intervals != expected_intervals:
            raise ValueError("bootstrap derivation differs from stored intervals")
        expected = _digest_document(self.model_dump(mode="json", exclude={"intervals_digest"}))
        if self.intervals_digest != expected:
            raise ValueError("confidence interval digest is inconsistent")
        return self

    def to_json(self) -> bytes:
        """Return canonical confidence-interval bytes after semantic revalidation."""
        if type(self) is not ConfidenceIntervals:
            raise MetricContractError("confidence evidence must be exact")
        try:
            checked = ConfidenceIntervals.model_validate(
                self.model_dump(mode="python", warnings=False), strict=True
            )
            payload = canonical_json_bytes(checked.model_dump(mode="json"))
        except (
            ArithmeticError,
            AttributeError,
            MemoryError,
            OverflowError,
            TypeError,
            ValidationError,
            ValueError,
        ) as error:
            raise MetricContractError("confidence evidence failed semantic revalidation") from error
        if len(payload) > MAX_METRIC_PAYLOAD_BYTES:
            raise MetricContractError("confidence payload exceeds resource cap")
        return payload

    @classmethod
    def from_json(cls, payload: bytes) -> ConfidenceIntervals:
        """Load bounded canonical confidence-interval evidence."""
        if type(payload) is not bytes:
            raise MetricContractError("confidence payload must be exact bytes")
        if len(payload) > MAX_METRIC_PAYLOAD_BYTES:
            raise MetricContractError("confidence payload exceeds resource cap")
        try:
            document = strict_json_loads(payload)
            if type(document) is not dict:
                raise MetricContractError("confidence JSON must contain an object")
            return cls.model_validate(document)
        except MetricContractError:
            raise
        except (
            ArithmeticError,
            MemoryError,
            OverflowError,
            ValidationError,
            WireContractError,
        ) as error:
            raise MetricContractError(str(error)) from error


@dataclass(frozen=True, slots=True)
class _MetricRow:
    event_id: str
    campaign_id: str
    family: str
    rail: Rail
    is_fraud: bool
    action: Action
    score: float
    decision_at: datetime
    net_value: Decimal
    first_settlement_at: datetime | None


@dataclass(frozen=True, slots=True)
class _CampaignGroup:
    campaign_id: str
    rows: tuple[_MetricRow, ...]
    fraud_rows: tuple[_MetricRow, ...]
    fraud_anchor_at: datetime | None
    alert_at: datetime | None


@dataclass(frozen=True, slots=True)
class _CampaignIndex:
    groups: tuple[_CampaignGroup, ...]


@dataclass(frozen=True, slots=True)
class _ValidatedInputs:
    inputs: MetricReportInputs
    rows: tuple[_MetricRow, ...]
    campaign_index: _CampaignIndex
    observation_by_id: dict[str, ObservedEvent]
    truth_by_id: dict[str, EvaluationTruthRow]
    decision_by_id: dict[str, DefenseDecision]
    currency: str | None


@dataclass(frozen=True, slots=True)
class _DerivedMetricSections:
    inputs: MetricReportInputs
    classification: ClassificationMetrics
    calibration: CalibrationMetrics
    value: ValueMetrics
    alerts: AlertMetrics
    operations: OperationalMetrics
    engineering: EngineeringMetrics


@dataclass(frozen=True, slots=True)
class _CampaignBootstrapUnit:
    true_positives: int
    false_positives: int
    false_negatives: int
    true_negatives: int
    fraud_campaign: int
    detected_campaign: int
    fraudulent_value: Decimal
    preventable_value: Decimal


def compute_metric_report(report_inputs: MetricReportInputs) -> MetricReport:
    """Compute exact metrics from mature, bijectively aligned synthetic evidence."""
    if type(report_inputs) is not MetricReportInputs:
        raise MetricContractError("report_inputs must be an exact MetricReportInputs")
    try:
        derived = _derive_metric_sections(report_inputs)
        evidence = MetricDerivationEvidence(inputs=derived.inputs)
        input_digest = _input_digest(derived.inputs)
        document: dict[str, object] = {
            "schema_version": "1.0.0",
            "evaluator_as_of": derived.inputs.as_of,
            "evaluator_input_digest": input_digest,
            "classification": derived.classification,
            "calibration": derived.calibration,
            "value": derived.value,
            "alerts": derived.alerts,
            "operations": derived.operations,
            "engineering": derived.engineering,
            "derivation_evidence": evidence,
            "synthetic_only": True,
            "external_validity_claimed": False,
            "limitations": _LIMITATIONS,
        }
        digest = _digest_document(_json_tree(document))
        report = MetricReport(
            evaluator_as_of=derived.inputs.as_of,
            evaluator_input_digest=input_digest,
            classification=derived.classification,
            calibration=derived.calibration,
            value=derived.value,
            alerts=derived.alerts,
            operations=derived.operations,
            engineering=derived.engineering,
            derivation_evidence=evidence,
            limitations=_LIMITATIONS,
            report_digest=digest,
        )
        if len(report.to_json()) > MAX_METRIC_PAYLOAD_BYTES:
            raise MetricContractError("metric report payload exceeds resource cap")
        return report
    except MetricContractError:
        raise
    except (
        ArithmeticError,
        AttributeError,
        MemoryError,
        OverflowError,
        TypeError,
        ValidationError,
        ValueError,
    ) as error:
        raise MetricContractError("metric computation failed deterministically") from error


def _derive_metric_sections(report_inputs: MetricReportInputs) -> _DerivedMetricSections:
    validated = _validate_inputs(report_inputs)
    classification = _classification_metrics(validated)
    calibration = _calibration_metrics(validated.rows)
    alerts, alert_times = _alert_metrics(validated.campaign_index)
    value = _value_metrics(validated, alert_times)
    operations = _operational_metrics(validated)
    engineering = _engineering_metrics(validated.inputs.latency_samples)
    return _DerivedMetricSections(
        inputs=validated.inputs,
        classification=classification,
        calibration=calibration,
        value=value,
        alerts=alerts,
        operations=operations,
        engineering=engineering,
    )


def campaign_bootstrap(
    report_inputs: MetricReportInputs,
    *,
    seed: int = BOOTSTRAP_SEED,
    replicates: int = BOOTSTRAP_REPLICATES,
) -> ConfidenceIntervals:
    """Bootstrap whole synthetic campaigns with the frozen PCG64 contract.

    Undefined ratio replicates are excluded from that metric's percentile interval
    and counted explicitly. Money metrics are always defined. The 2.5%, 50%, and
    97.5% bounds use NumPy's stable ``linear`` percentile method.
    """
    if type(report_inputs) is not MetricReportInputs:
        raise MetricContractError("report_inputs must be an exact MetricReportInputs")
    if type(seed) is not int or seed != BOOTSTRAP_SEED:
        raise MetricContractError("campaign bootstrap seed must equal 260816")
    if type(replicates) is not int or replicates != BOOTSTRAP_REPLICATES:
        raise MetricContractError("campaign bootstrap replicate count must equal 1000")
    try:
        validated = _validate_inputs(report_inputs)
        campaign_groups = validated.campaign_index.groups
        campaign_ids = tuple(group.campaign_id for group in campaign_groups)
        if len(campaign_ids) > MAX_METRIC_CAMPAIGNS:
            raise MetricContractError("campaign count exceeds metric resource cap")
        if len(campaign_ids) * replicates > MAX_BOOTSTRAP_WORK:
            raise MetricContractError("campaign bootstrap exceeds frozen work cap")
        contributions = tuple(
            _campaign_bootstrap_contribution(group.campaign_id, group.rows)
            for group in campaign_groups
        )
        intervals = _bootstrap_intervals(
            contributions,
            seed=seed,
            replicates=replicates,
        )
        document: dict[str, object] = {
            "schema_version": "1.0.0",
            "resampling_unit": "campaign",
            "generator": "numpy.Generator(PCG64)",
            "percentile_method": "linear",
            "confidence_level": 0.95,
            "seed": seed,
            "replicates": replicates,
            "evaluator_input_digest": _input_digest(validated.inputs),
            "intervals": intervals,
            "campaign_contributions": contributions,
            "synthetic_only": True,
            "external_validity_claimed": False,
        }
        return ConfidenceIntervals(
            evaluator_input_digest=cast(str, document["evaluator_input_digest"]),
            intervals=intervals,
            campaign_contributions=contributions,
            intervals_digest=_digest_document(_json_tree(document)),
        )
    except MetricContractError:
        raise
    except (
        ArithmeticError,
        AttributeError,
        MemoryError,
        OverflowError,
        TypeError,
        ValidationError,
        ValueError,
    ) as error:
        raise MetricContractError("campaign bootstrap failed deterministically") from error


def _validate_inputs(inputs: MetricReportInputs) -> _ValidatedInputs:
    collections = (
        ("truth", inputs.truth),
        ("observations", inputs.observations),
        ("decisions", inputs.decisions),
        ("cases", inputs.cases),
        ("latency samples", inputs.latency_samples),
        ("slice assignments", inputs.slice_assignments),
    )
    for label, collection_rows in collections:
        if type(collection_rows) is not tuple:
            raise MetricContractError(f"{label} must be an exact tuple")
        if len(collection_rows) > MAX_METRIC_ROWS:
            raise MetricContractError("metric input row resource cap exceeded")
    raw_campaign_ids = {
        row.campaign_id
        for row in inputs.truth
        if type(row) is EvaluationTruthRow and type(row.campaign_id) is str
    }
    if len(raw_campaign_ids) > MAX_METRIC_CAMPAIGNS:
        raise MetricContractError("metric campaign resource cap exceeded")
    lifecycle_reference_count = sum(
        len(row.lifecycle_event_ids)
        for row in inputs.truth
        if type(row) is EvaluationTruthRow and type(row.lifecycle_event_ids) is tuple
    )
    if lifecycle_reference_count > MAX_LIFECYCLE_REFERENCES:
        raise MetricContractError("lifecycle reference resource cap exceeded")
    if type(inputs.as_of) is not datetime:
        raise MetricContractError("metric as_of must be an exact datetime")
    try:
        as_of = validate_utc_timestamp(inputs.as_of)
    except ValueError as error:
        raise MetricContractError("metric as_of must be timezone-aware UTC") from error

    truth_rows = _revalidate_rows(
        inputs.truth, EvaluationTruthRow, label="EvaluationTruthRow"
    )
    observation_rows = _revalidate_rows(
        inputs.observations, ObservedEvent, label="ObservedEvent"
    )
    for decision_row in inputs.decisions:
        if type(decision_row) is DefenseDecision:
            _validate_decision_numeric_types(decision_row)
    decision_rows = _revalidate_rows(
        inputs.decisions, DefenseDecision, label="DefenseDecision"
    )
    case_rows = _revalidate_rows(inputs.cases, InvestigationCase, label="InvestigationCase")
    latency_rows = _revalidate_rows(
        inputs.latency_samples, LatencySample, label="LatencySample"
    )
    slice_rows = _revalidate_rows(
        inputs.slice_assignments, SliceAssignment, label="SliceAssignment"
    )
    if inputs.slice_manifest is None:
        slice_manifest = None
    elif type(inputs.slice_manifest) is not SliceManifest:
        raise MetricContractError("slice manifest must be an exact SliceManifest")
    else:
        try:
            slice_manifest = SliceManifest.model_validate(
                inputs.slice_manifest.model_dump(mode="python", warnings=False),
                strict=True,
            )
        except (AttributeError, TypeError, ValidationError, ValueError) as error:
            raise MetricContractError("slice manifest failed semantic revalidation") from error
    if type(inputs.queue_report) is not QueueReport:
        raise MetricContractError("queue report must be an exact QueueReport")
    try:
        queue_report = QueueReport.from_json(inputs.queue_report.to_json())
    except (AttributeError, TypeError, ValueError) as error:
        raise MetricContractError("queue report failed semantic revalidation") from error
    if tuple(sorted(case_rows, key=lambda row: row.case_id)) != tuple(
        sorted(queue_report.case_inputs, key=lambda row: row.case_id)
    ):
        raise MetricContractError("supplied cases must exactly match queue case inputs")

    truth_ids = _unique_ids(truth_rows, label="truth")
    _unique_ids(observation_rows, label="observation")
    decision_ids = _unique_ids(decision_rows, label="decision")
    latency_ids = _unique_ids(latency_rows, label="latency")
    truth_by_id = {row.event_id: row for row in truth_rows}
    observation_by_id = {row.event_id: row for row in observation_rows}
    decision_by_id = {row.event_id: row for row in decision_rows}
    decision_point_ids = {
        row.event_id for row in observation_rows if row.is_decision_point
    }
    if truth_ids != decision_ids or truth_ids != decision_point_ids:
        raise MetricContractError(
            "truth, decisions, and decision-point observations must align bijectively"
        )
    if latency_ids != decision_ids:
        raise MetricContractError("latency samples must align bijectively with decisions")
    for observation in observation_rows:
        _validate_observation(observation, as_of)
    for row in truth_rows:
        _validate_truth(row, as_of)
        opening = observation_by_id[row.event_id]
        if opening.payment_id != row.payment_id:
            raise MetricContractError("truth opening payment binding differs")
    _validate_slice_assignments(slice_rows, truth_ids, slice_manifest)
    currency = _validate_lifecycles(truth_rows, observation_by_id)
    try:
        with localcontext(_DECIMAL128_CONTEXT):
            causal_cases = group_cases(observation_rows, decision_rows, as_of=as_of)
    except (ArithmeticError, MemoryError, OverflowError, TypeError, ValueError) as error:
        raise MetricContractError("causal case reconstruction failed") from error
    if tuple(sorted(case_rows, key=lambda row: row.case_id)) != tuple(
        sorted(causal_cases, key=lambda row: row.case_id)
    ):
        raise MetricContractError("supplied cases do not match causal nonapprove decisions")

    metric_rows = tuple(
        _MetricRow(
            event_id=event_id,
            campaign_id=truth_by_id[event_id].campaign_id,
            family=truth_by_id[event_id].family,
            rail=observation_by_id[event_id].rail,
            is_fraud=truth_by_id[event_id].is_fraud,
            action=decision_by_id[event_id].action,
            score=decision_by_id[event_id].score,
            decision_at=cast(datetime, observation_by_id[event_id].decision_at),
            net_value=truth_by_id[event_id].net_settled_value,
            first_settlement_at=truth_by_id[event_id].first_settlement_at,
        )
        for event_id in sorted(truth_ids)
    )
    checked_inputs = MetricReportInputs(
        truth=tuple(sorted(truth_rows, key=lambda row: row.event_id)),
        observations=tuple(sorted(observation_rows, key=lambda row: row.event_id)),
        decisions=tuple(sorted(decision_rows, key=lambda row: row.event_id)),
        cases=tuple(sorted(case_rows, key=lambda row: row.case_id)),
        queue_report=queue_report,
        latency_samples=tuple(sorted(latency_rows, key=lambda row: row.event_id)),
        as_of=as_of,
        slice_assignments=tuple(sorted(slice_rows, key=lambda row: row.event_id)),
        slice_manifest=slice_manifest,
    )
    return _ValidatedInputs(
        inputs=checked_inputs,
        rows=metric_rows,
        campaign_index=_prepare_campaign_index(metric_rows),
        observation_by_id=observation_by_id,
        truth_by_id=truth_by_id,
        decision_by_id=decision_by_id,
        currency=currency,
    )


def _classification_metrics(validated: _ValidatedInputs) -> ClassificationMetrics:
    rows = validated.rows
    counts = _confusion_counts(rows)
    fraud_campaigns, alert_times = _campaign_alert_times(validated.campaign_index)
    labels = np.asarray([int(row.is_fraud) for row in rows], dtype=np.int8)
    scores = np.asarray([row.score for row in rows], dtype=np.float64)
    pr_auc, roc_auc = _auc_metrics(labels, scores)
    assignments = {row.event_id: row for row in validated.inputs.slice_assignments}
    family_rows: dict[str, list[_MetricRow]] = defaultdict(list)
    rail_rows: dict[str, list[_MetricRow]] = defaultdict(list)
    regime_rows: dict[str, list[_MetricRow]] = defaultdict(list)
    cohort_rows: dict[str, list[_MetricRow]] = defaultdict(list)
    for row in rows:
        family_rows[row.family].append(row)
        rail_rows[row.rail.value].append(row)
        if assignments:
            assignment = assignments[row.event_id]
            regime_rows[assignment.regime].append(row)
            cohort_rows[assignment.entity_cohort].append(row)
    slices: list[ClassificationSlice] = []
    for family in _FAMILIES:
        slices.append(_classification_slice("family", family, tuple(family_rows[family])))
    for rail in Rail:
        slices.append(_classification_slice("rail", rail.value, tuple(rail_rows[rail.value])))
    manifest = validated.inputs.slice_manifest
    if assignments and manifest is not None:
        for regime in manifest.regimes:
            slices.append(
                _classification_slice(
                    "regime",
                    regime,
                    tuple(regime_rows[regime]),
                )
            )
        for cohort in manifest.entity_cohorts:
            slices.append(
                _classification_slice(
                    "entity_cohort",
                    cohort,
                    tuple(cohort_rows[cohort]),
                )
            )
    return ClassificationMetrics(
        **counts,
        decline_true_positives=sum(
            row.is_fraud and row.action is Action.DECLINE for row in rows
        ),
        decline_false_positives=sum(
            not row.is_fraud and row.action is Action.DECLINE for row in rows
        ),
        fraud_campaigns=len(fraud_campaigns),
        detected_fraud_campaigns=len(alert_times),
        precision=_ratio(
            counts["true_positives"],
            counts["true_positives"] + counts["false_positives"],
            "no_predicted_positives",
        ),
        recall=_ratio(
            counts["true_positives"], counts["fraud_count"], "absent_positive_class"
        ),
        f1=_ratio(
            2 * counts["true_positives"],
            2 * counts["true_positives"]
            + counts["false_positives"]
            + counts["false_negatives"],
            "no_positive_truth_or_predictions",
        ),
        false_positive_rate=_ratio(
            counts["false_positives"],
            counts["legitimate_count"],
            "absent_legitimate_class",
        ),
        decline_precision=_ratio(
            sum(row.is_fraud and row.action is Action.DECLINE for row in rows),
            sum(row.action is Action.DECLINE for row in rows),
            "no_declines",
        ),
        decline_recall=_ratio(
            sum(row.is_fraud and row.action is Action.DECLINE for row in rows),
            counts["fraud_count"],
            "absent_positive_class",
        ),
        campaign_recall=_ratio(
            len(alert_times), len(fraud_campaigns), "absent_fraud_campaigns"
        ),
        pr_auc=pr_auc,
        roc_auc=roc_auc,
        slices=tuple(sorted(slices, key=_slice_key)),
    )


def _calibration_metrics(rows: tuple[_MetricRow, ...]) -> CalibrationMetrics:
    ordered = tuple(sorted(rows, key=lambda row: (row.score, row.event_id)))
    count = len(ordered)
    positive_count = sum(row.is_fraud for row in ordered)
    if count == 0:
        undefined = MetricValue(
            value=None, numerator=0.0, denominator=0.0, undefined_reason="no_decisions"
        )
        absent = MetricValue(
            value=None, numerator=0.0, denominator=0.0, undefined_reason="absent_class"
        )
        return CalibrationMetrics(
            row_count=0,
            positive_count=0,
            brier_score=undefined,
            ece=undefined,
            reliability_bins=(),
            slope=absent,
            intercept=absent,
        )
    squared_error_sum = math.fsum(
        (row.score - float(row.is_fraud)) ** 2 for row in ordered
    )
    bin_count = min(_CALIBRATION_BINS, count)
    bins: list[ReliabilityBin] = []
    ece_numerator = 0.0
    for bin_index in range(bin_count):
        start = bin_index * count // bin_count
        stop = (bin_index + 1) * count // bin_count
        selected = ordered[start:stop]
        mean_prediction = math.fsum(row.score for row in selected) / len(selected)
        observed_frequency = sum(row.is_fraud for row in selected) / len(selected)
        ece_numerator += len(selected) * abs(mean_prediction - observed_frequency)
        bins.append(
            ReliabilityBin(
                bin_index=bin_index,
                start_rank=start,
                stop_rank=stop,
                count=len(selected),
                lower_score=selected[0].score,
                upper_score=selected[-1].score,
                mean_prediction=mean_prediction,
                observed_frequency=observed_frequency,
            )
        )
    slope, intercept = _calibration_fit(ordered)
    return CalibrationMetrics(
        row_count=count,
        positive_count=positive_count,
        brier_score=MetricValue(
            value=squared_error_sum / count,
            numerator=squared_error_sum,
            denominator=float(count),
        ),
        ece=MetricValue(
            value=ece_numerator / count,
            numerator=ece_numerator,
            denominator=float(count),
        ),
        reliability_bins=tuple(bins),
        slope=slope,
        intercept=intercept,
    )


def _alert_metrics(
    campaign_index: _CampaignIndex,
) -> tuple[AlertMetrics, dict[str, datetime]]:
    fraud_campaigns, alert_times = _campaign_alert_times(campaign_index)
    anchors = {
        group.campaign_id: group.fraud_anchor_at
        for group in campaign_index.groups
        if group.fraud_anchor_at is not None
    }
    durations = tuple(
        sorted(
            (alert_times[campaign] - anchors[campaign]).total_seconds()
            for campaign in alert_times
        )
    )
    detected = len(durations)
    return (
        AlertMetrics(
            campaign_count=len(fraud_campaigns),
            detected_campaigns=detected,
            undetected_campaigns=len(fraud_campaigns) - detected,
            p50_seconds=_eligible_alert_quantile(durations, 50.0, 1),
            p90_seconds=_eligible_alert_quantile(durations, 90.0, 10),
            p95_seconds=_eligible_alert_quantile(durations, 95.0, 20),
            p99_seconds=_eligible_alert_quantile(durations, 99.0, 100),
        ),
        alert_times,
    )


def _value_metrics(
    validated: _ValidatedInputs, alert_times: dict[str, datetime]
) -> ValueMetrics:
    fraudulent_total = _money_sum(
        row.net_value for row in validated.rows if row.is_fraud
    )
    prevented = _money_sum(
        row.net_value
        for row in validated.rows
        if row.is_fraud
        and row.action is Action.DECLINE
        and row.first_settlement_at is not None
        and row.decision_at < row.first_settlement_at
    )
    value_before_alert = Decimal("0.00")
    remaining_at_alert = Decimal("0.00")
    fraud_groups = {
        group.campaign_id: group.fraud_rows
        for group in validated.campaign_index.groups
        if group.fraud_rows
    }
    for campaign_id, alert_at in alert_times.items():
        campaign_rows = fraud_groups[campaign_id]
        campaign_total = _money_sum(row.net_value for row in campaign_rows)
        before = Decimal("0.00")
        for row in campaign_rows:
            truth = validated.truth_by_id[row.event_id]
            for lifecycle_id in truth.lifecycle_event_ids:
                event = validated.observation_by_id[lifecycle_id]
                if event.event_time >= alert_at:
                    continue
                if event.event_type in _SETTLEMENT_EVENTS:
                    before = _money_add(before, event.amount)
                elif event.event_type in _REVERSING_EVENTS:
                    before = _money_subtract(before, event.amount, allow_negative=True)
        value_before_alert = _money_add(
            value_before_alert, max(before, Decimal("0.00"))
        )
        remaining_at_alert = _money_add(
            remaining_at_alert,
            max(
                _money_subtract(campaign_total, before, allow_negative=True),
                Decimal("0.00"),
            ),
        )
    case_count = len(validated.inputs.cases)
    analyst_minutes = validated.inputs.queue_report.analyst_minutes
    return ValueMetrics(
        currency=validated.currency,
        fraudulent_net_settled_value=fraudulent_total,
        preventable_settled_value=prevented,
        value_escaped=_money_subtract(fraudulent_total, prevented),
        value_before_first_alert=value_before_alert,
        remaining_preventable_at_alert=remaining_at_alert,
        challenge_credited_as_prevented=Decimal("0.00"),
        review_case_count=case_count,
        analyst_minutes=analyst_minutes,
        captured_value_per_review_case=_decimal_ratio(
            prevented, Decimal(case_count), "no_review_cases"
        ),
        captured_value_per_analyst_hour=_decimal_ratio(
            _decimal_multiply(prevented, Decimal(60)),
            Decimal(analyst_minutes),
            "no_analyst_minutes",
        ),
    )


def _operational_metrics(validated: _ValidatedInputs) -> OperationalMetrics:
    rows = validated.rows
    legitimate = tuple(row for row in rows if not row.is_fraud)
    false_challenges = sum(row.action is Action.CHALLENGE for row in legitimate)
    false_declines = sum(row.action is Action.DECLINE for row in legitimate)
    false_interventions = false_challenges + false_declines
    challenge_count = sum(row.action is Action.CHALLENGE for row in rows)
    cases = validated.inputs.cases
    case_count = len(cases)
    case_transactions = sum(len(row.event_ids) for row in cases)
    case_entities = sum(
        len(set(row.actor_ids) | set(row.counterparty_ids)) for row in cases
    )
    fallback_count = sum(row.fallback_used for row in validated.inputs.decisions)
    return OperationalMetrics(
        decision_count=len(rows),
        legitimate_count=len(legitimate),
        false_intervention_count=false_interventions,
        false_challenge_count=false_challenges,
        false_decline_count=false_declines,
        challenge_count=challenge_count,
        review_case_count=case_count,
        case_transaction_count=case_transactions,
        case_entity_count=case_entities,
        analyst_minutes=validated.inputs.queue_report.analyst_minutes,
        peak_backlog_count=validated.inputs.queue_report.peak_backlog_count,
        sla_breaches=validated.inputs.queue_report.sla_breach_count,
        fallback_count=fallback_count,
        false_interventions_per_10k=_scaled_ratio(
            false_interventions,
            len(legitimate),
            10_000,
            "absent_legitimate_class",
        ),
        false_challenges_per_10k=_scaled_ratio(
            false_challenges,
            len(legitimate),
            10_000,
            "absent_legitimate_class",
        ),
        false_declines_per_10k=_scaled_ratio(
            false_declines,
            len(legitimate),
            10_000,
            "absent_legitimate_class",
        ),
        total_challenges_per_10k=_scaled_ratio(
            challenge_count, len(rows), 10_000, "no_decisions"
        ),
        review_cases_per_100k=_scaled_ratio(
            case_count, len(rows), 100_000, "no_decisions"
        ),
        transactions_per_case=_ratio(
            case_transactions, case_count, "no_review_cases"
        ),
        entities_per_case=_ratio(case_entities, case_count, "no_review_cases"),
        fallback_rate=_ratio(fallback_count, len(rows), "no_decisions"),
    )


def _engineering_metrics(samples: tuple[LatencySample, ...]) -> EngineeringMetrics:
    return EngineeringMetrics(
        feature_ms=_latency_quantiles(tuple(row.feature_ms for row in samples)),
        rules_ms=_latency_quantiles(tuple(row.rules_ms for row in samples)),
        model_ms=_latency_quantiles(tuple(row.model_ms for row in samples)),
        calibration_policy_ms=_latency_quantiles(
            tuple(row.calibration_policy_ms for row in samples)
        ),
        end_to_end_ms=_latency_quantiles(tuple(row.end_to_end_ms for row in samples)),
    )


def _latency_quantiles(values: tuple[float, ...]) -> LatencyQuantiles:
    if not values:
        def undefined() -> MetricValue:
            return MetricValue(
                value=None,
                numerator=0.0,
                denominator=0.0,
                undefined_reason="empty_latency_samples",
            )

        return LatencyQuantiles(
            sample_count=0,
            p50=undefined(),
            p90=undefined(),
            p95=undefined(),
            p99=undefined(),
        )
    count = len(values)

    def quantile(percentile: float) -> MetricValue:
        value = float(np.percentile(values, percentile, method="linear"))
        return MetricValue(
            value=value,
            numerator=value * count,
            denominator=float(count),
        )

    return LatencyQuantiles(
        sample_count=count,
        p50=quantile(50.0),
        p90=quantile(90.0),
        p95=quantile(95.0),
        p99=quantile(99.0),
    )


def _classification_slice(
    kind: Literal["family", "rail", "regime", "entity_cohort"],
    value: str,
    selected: tuple[_MetricRow, ...],
) -> ClassificationSlice:
    counts = _confusion_counts(selected)
    decline_true_positives = sum(
        row.is_fraud and row.action is Action.DECLINE for row in selected
    )
    decline_false_positives = sum(
        not row.is_fraud and row.action is Action.DECLINE for row in selected
    )
    fraud_campaigns, alert_times = _campaign_alert_times(selected)
    labels = np.asarray([int(row.is_fraud) for row in selected], dtype=np.int8)
    scores = np.asarray([row.score for row in selected], dtype=np.float64)
    pr_auc, roc_auc = _auc_metrics(labels, scores)
    return ClassificationSlice(
        kind=kind,
        value=value,
        **counts,
        decline_true_positives=decline_true_positives,
        decline_false_positives=decline_false_positives,
        fraud_campaigns=len(fraud_campaigns),
        detected_fraud_campaigns=len(alert_times),
        precision=_ratio(
            counts["true_positives"],
            counts["true_positives"] + counts["false_positives"],
            "no_predicted_positives",
        ),
        recall=_ratio(
            counts["true_positives"], counts["fraud_count"], "absent_positive_class"
        ),
        f1=_ratio(
            2 * counts["true_positives"],
            2 * counts["true_positives"]
            + counts["false_positives"]
            + counts["false_negatives"],
            "no_positive_truth_or_predictions",
        ),
        false_positive_rate=_ratio(
            counts["false_positives"],
            counts["legitimate_count"],
            "absent_legitimate_class",
        ),
        decline_precision=_ratio(
            decline_true_positives,
            decline_true_positives + decline_false_positives,
            "no_declines",
        ),
        decline_recall=_ratio(
            decline_true_positives,
            counts["fraud_count"],
            "absent_positive_class",
        ),
        campaign_recall=_ratio(
            len(alert_times), len(fraud_campaigns), "absent_fraud_campaigns"
        ),
        pr_auc=pr_auc,
        roc_auc=roc_auc,
    )


def _confusion_counts(rows: Sequence[_MetricRow]) -> dict[str, int]:
    true_positives = sum(row.is_fraud and row.action is not Action.APPROVE for row in rows)
    false_positives = sum(
        not row.is_fraud and row.action is not Action.APPROVE for row in rows
    )
    false_negatives = sum(row.is_fraud and row.action is Action.APPROVE for row in rows)
    true_negatives = sum(
        not row.is_fraud and row.action is Action.APPROVE for row in rows
    )
    return {
        "row_count": len(rows),
        "fraud_count": true_positives + false_negatives,
        "legitimate_count": false_positives + true_negatives,
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "true_negatives": true_negatives,
    }


def _auc_metrics(
    labels: NDArray[np.int8],
    scores: NDArray[np.float64],
) -> tuple[MetricValue, MetricValue]:
    count = len(labels)
    positives = int(labels.sum())
    if count == 0 or positives == 0:
        reason = "absent_positive_class"
        undefined = MetricValue(
            value=None,
            numerator=float(positives),
            denominator=float(count),
            undefined_reason=reason,
        )
        return undefined, undefined.model_copy()
    if positives == count:
        undefined = MetricValue(
            value=None,
            numerator=float(positives),
            denominator=float(count),
            undefined_reason="absent_negative_class",
        )
        return undefined, undefined.model_copy()
    try:
        pr_value = float(average_precision_score(labels, scores))
        roc_value = float(roc_auc_score(labels, scores))
    except (FloatingPointError, MemoryError, OverflowError, RuntimeError, ValueError) as error:
        raise MetricContractError("sklearn AUC computation failed deterministically") from error
    return (
        MetricValue(
            value=pr_value,
            numerator=pr_value * count,
            denominator=float(count),
        ),
        MetricValue(
            value=roc_value,
            numerator=roc_value * count,
            denominator=float(count),
        ),
    )


def _calibration_fit(
    ordered: tuple[_MetricRow, ...],
) -> tuple[MetricValue, MetricValue]:
    count = len(ordered)
    positives = sum(row.is_fraud for row in ordered)
    if positives == 0 or positives == count:
        return _undefined_fit(count, "absent_class")
    scores = np.asarray(
        [min(max(row.score, _LOGIT_EPSILON), 1.0 - _LOGIT_EPSILON) for row in ordered],
        dtype=np.float64,
    )
    logits = np.log(scores / (1.0 - scores))
    if float(np.ptp(logits)) == 0.0:
        return _undefined_fit(count, "degenerate_logits")
    labels = np.asarray([int(row.is_fraud) for row in ordered], dtype=np.int8)
    try:
        fitted = LogisticRegression(
            C=np.inf,
            solver="lbfgs",
            fit_intercept=True,
            random_state=BOOTSTRAP_SEED,
            max_iter=10_000,
            tol=1e-12,
        ).fit(logits.reshape(-1, 1), labels)
    except (FloatingPointError, MemoryError, OverflowError, RuntimeError, ValueError) as error:
        raise MetricContractError("sklearn calibration fit failed deterministically") from error
    slope = float(fitted.coef_[0, 0])
    intercept = float(fitted.intercept_[0])
    if not math.isfinite(slope) or not math.isfinite(intercept):
        raise MetricContractError("calibration fit returned non-finite coefficients")
    return (
        MetricValue(value=slope, numerator=slope, denominator=1.0),
        MetricValue(value=intercept, numerator=intercept, denominator=1.0),
    )


def _undefined_fit(count: int, reason: str) -> tuple[MetricValue, MetricValue]:
    return (
        MetricValue(
            value=None,
            numerator=0.0,
            denominator=float(count),
            undefined_reason=reason,
        ),
        MetricValue(
            value=None,
            numerator=0.0,
            denominator=float(count),
            undefined_reason=reason,
        ),
    )


def _prepare_campaign_index(rows: tuple[_MetricRow, ...]) -> _CampaignIndex:
    grouped: dict[str, list[_MetricRow]] = defaultdict(list)
    for row in rows:
        grouped[row.campaign_id].append(row)
    groups: list[_CampaignGroup] = []
    for campaign_id in sorted(grouped):
        campaign_rows = tuple(
            sorted(grouped[campaign_id], key=lambda row: (row.decision_at, row.event_id))
        )
        fraud_rows = tuple(row for row in campaign_rows if row.is_fraud)
        fraud_anchor_at = fraud_rows[0].decision_at if fraud_rows else None
        eligible_alerts = (
            tuple(
                row.decision_at
                for row in campaign_rows
                if row.decision_at >= fraud_anchor_at
                and row.action is not Action.APPROVE
            )
            if fraud_anchor_at is not None
            else ()
        )
        groups.append(
            _CampaignGroup(
                campaign_id=campaign_id,
                rows=campaign_rows,
                fraud_rows=fraud_rows,
                fraud_anchor_at=fraud_anchor_at,
                alert_at=eligible_alerts[0] if eligible_alerts else None,
            )
        )
    return _CampaignIndex(groups=tuple(groups))


def _campaign_alert_times(
    rows_or_index: tuple[_MetricRow, ...] | _CampaignIndex,
) -> tuple[tuple[str, ...], dict[str, datetime]]:
    campaign_index = (
        rows_or_index
        if isinstance(rows_or_index, _CampaignIndex)
        else _prepare_campaign_index(rows_or_index)
    )
    fraud_campaigns = tuple(
        group.campaign_id
        for group in campaign_index.groups
        if group.fraud_anchor_at is not None
    )
    alerts = {
        group.campaign_id: group.alert_at
        for group in campaign_index.groups
        if group.fraud_anchor_at is not None and group.alert_at is not None
    }
    return fraud_campaigns, alerts


def _eligible_alert_quantile(
    durations: tuple[float, ...], percentile: float, minimum: int
) -> MetricValue:
    detected = len(durations)
    if detected < minimum:
        return MetricValue(
            value=None,
            numerator=float(detected),
            denominator=float(minimum),
            undefined_reason="insufficient_detected_campaigns",
        )
    value = float(np.percentile(durations, percentile, method="linear"))
    return MetricValue(
        value=value,
        numerator=float(detected),
        denominator=float(minimum),
    )


def _validate_observation(row: ObservedEvent, as_of: datetime) -> None:
    for label, timestamp in (
        ("event_time", row.event_time),
        ("available_at", row.available_at),
    ):
        if type(timestamp) is not datetime:
            raise MetricContractError(f"observation {label} must be an exact datetime")
        try:
            validate_utc_timestamp(timestamp)
        except ValueError as error:
            raise MetricContractError(f"observation {label} must be UTC") from error
        if timestamp > as_of:
            raise MetricContractError("observation occurs after metric as_of")
    if row.event_time > row.available_at:
        raise MetricContractError("observation event_time exceeds availability")
    if row.is_decision_point:
        if type(row.decision_at) is not datetime:
            raise MetricContractError("decision-point observation requires exact decision_at")
        try:
            validate_utc_timestamp(row.decision_at)
        except ValueError as error:
            raise MetricContractError("observation decision_at must be UTC") from error
        if row.available_at > row.decision_at or row.decision_at > as_of:
            raise MetricContractError("observation decision time is outside its causal window")
    elif row.decision_at is not None:
        raise MetricContractError("contextual observation cannot declare decision_at")
    _validate_cent_amount(row.amount, label="observation amount")
    for label, identifier in (
        ("event_id", row.event_id),
        ("payment_id", row.payment_id),
        ("currency", row.currency),
    ):
        if (
            type(identifier) is not str
            or not identifier
            or len(identifier) > _MAX_IDENTIFIER_LENGTH
        ):
            raise MetricContractError(f"observation {label} must be bounded nonempty text")


def _validate_decision_numeric_types(row: DefenseDecision) -> None:
    values = (row.score, row.rule_score, row.latency_ms)
    if any(type(value) is not float or not math.isfinite(value) for value in values):
        raise MetricContractError(
            "DefenseDecision failed semantic revalidation: "
            "decision numeric fields must be exact finite floats"
        )
    if row.calibrated_score is not None and (
        type(row.calibrated_score) is not float or not math.isfinite(row.calibrated_score)
    ):
        raise MetricContractError(
            "DefenseDecision failed semantic revalidation: "
            "decision numeric fields must be exact finite floats"
        )


def _validate_truth(row: EvaluationTruthRow, as_of: datetime) -> None:
    for label, timestamp in (
        ("label_mature_at", row.label_mature_at),
        ("first_settlement_at", row.first_settlement_at),
    ):
        if timestamp is None:
            continue
        if type(timestamp) is not datetime:
            raise MetricContractError(f"truth {label} must be an exact datetime")
        try:
            validate_utc_timestamp(timestamp)
        except ValueError as error:
            raise MetricContractError(f"truth {label} must be UTC") from error
    if row.label_mature_at > as_of:
        raise MetricContractError("truth label is not mature at evaluation as_of")
    if row.first_settlement_at is not None and row.first_settlement_at > as_of:
        raise MetricContractError("truth first settlement occurs after evaluation as_of")
    if type(row.is_fraud) is not bool:
        raise MetricContractError("truth is_fraud must be an exact bool")
    _validate_cent_amount(row.net_settled_value, label="truth net settlement")
    if type(row.lifecycle_event_ids) is not tuple or not row.lifecycle_event_ids:
        raise MetricContractError("truth lifecycle IDs must be a nonempty exact tuple")
    if (
        any(
            type(identifier) is not str
            or not identifier
            or len(identifier) > _MAX_IDENTIFIER_LENGTH
            for identifier in row.lifecycle_event_ids
        )
        or len(set(row.lifecycle_event_ids)) != len(row.lifecycle_event_ids)
        or row.event_id not in row.lifecycle_event_ids
    ):
        raise MetricContractError("truth lifecycle IDs must be unique and contain the opening")


def _validate_lifecycles(
    truth_rows: tuple[EvaluationTruthRow, ...],
    observation_by_id: dict[str, ObservedEvent],
) -> str | None:
    payment_ids = tuple(row.payment_id for row in truth_rows)
    if len(set(payment_ids)) != len(payment_ids):
        raise MetricContractError("truth payment IDs must be unique")
    observations_by_payment: dict[str, list[ObservedEvent]] = defaultdict(list)
    for observation in observation_by_id.values():
        observations_by_payment[observation.payment_id].append(observation)
    currencies: set[str] = set()
    for truth in truth_rows:
        lifecycle = tuple(
            sorted(
                observations_by_payment[truth.payment_id],
                key=lambda event: (event.event_time, event.event_id),
            )
        )
        expected_ids = tuple(event.event_id for event in lifecycle)
        if set(truth.lifecycle_event_ids) != set(expected_ids):
            raise MetricContractError(
                "truth lifecycle IDs must exactly cover every observation for its payment"
            )
        if truth.lifecycle_event_ids != expected_ids:
            raise MetricContractError("truth lifecycle IDs must use canonical lifecycle order")
        lifecycle_currencies = {event.currency for event in lifecycle}
        if len(lifecycle_currencies) != 1:
            raise MetricContractError("mixed currency within a payment lifecycle")
        currencies.update(lifecycle_currencies)
        settlements = tuple(
            sorted(
                (event for event in lifecycle if event.event_type in _SETTLEMENT_EVENTS),
                key=lambda event: (event.event_time, event.event_id),
            )
        )
        first_settlement = settlements[0].event_time if settlements else None
        if first_settlement != truth.first_settlement_at:
            raise MetricContractError("truth first settlement does not match lifecycle evidence")
        net = _money_subtract(
            _money_sum(event.amount for event in settlements),
            _money_sum(
                event.amount
                for event in lifecycle
                if event.event_type in _REVERSING_EVENTS
            ),
            allow_negative=True,
        )
        if net < 0:
            raise MetricContractError("lifecycle net settlement must be nonnegative")
        if net != truth.net_settled_value:
            raise MetricContractError("truth net settlement does not match lifecycle evidence")
    if len(currencies) > 1:
        raise MetricContractError("mixed-currency metric aggregation is forbidden")
    return next(iter(currencies)) if currencies else None


def _validate_slice_assignments(
    assignments: tuple[SliceAssignment, ...],
    truth_ids: set[str],
    manifest: SliceManifest | None,
) -> None:
    if not truth_ids:
        if assignments:
            raise MetricContractError("empty metric inputs cannot contain slice assignments")
        return
    if manifest is None:
        raise MetricContractError("nonempty metric inputs require a slice manifest")
    if not assignments:
        raise MetricContractError("slice assignments are required for nonempty metric inputs")
    identifiers = tuple(row.event_id for row in assignments)
    if len(set(identifiers)) != len(identifiers) or set(identifiers) != truth_ids:
        raise MetricContractError("slice assignments must align bijectively with truth rows")
    if any(row.regime not in manifest.regimes for row in assignments):
        raise MetricContractError("slice assignment regime is absent from its manifest")
    if any(row.entity_cohort not in manifest.entity_cohorts for row in assignments):
        raise MetricContractError("slice assignment entity cohort is absent from its manifest")


def _revalidate_rows[T: ExternalContract](
    rows: tuple[T, ...], expected_type: type[T], *, label: str
) -> tuple[T, ...]:
    checked: list[T] = []
    for row in rows:
        if type(row) is not expected_type:
            raise MetricContractError(f"{label} rows must use the exact contract class")
        try:
            checked.append(
                expected_type.model_validate(
                    row.model_dump(mode="python", warnings=False),
                    strict=True,
                )
            )
        except (AttributeError, TypeError, ValidationError, ValueError) as error:
            raise MetricContractError(
                f"{label} failed semantic revalidation: {error}"
            ) from error
    return tuple(checked)


def _unique_ids(rows: Sequence[object], *, label: str) -> set[str]:
    identifiers: list[str] = []
    for row in rows:
        identifier = getattr(row, "event_id", None)
        if type(identifier) is not str or not identifier:
            raise MetricContractError(f"{label} event IDs must be exact nonempty text")
        identifiers.append(identifier)
    if len(set(identifiers)) != len(identifiers):
        raise MetricContractError(f"duplicate {label} event ID")
    return set(identifiers)


def _bootstrap_unit(rows: tuple[_MetricRow, ...]) -> _CampaignBootstrapUnit:
    counts = _confusion_counts(rows)
    fraud_campaigns, alerts = _campaign_alert_times(rows)
    fraudulent_value = _money_sum(row.net_value for row in rows if row.is_fraud)
    preventable_value = _money_sum(
        row.net_value
        for row in rows
        if row.is_fraud
        and row.action is Action.DECLINE
        and row.first_settlement_at is not None
        and row.decision_at < row.first_settlement_at
    )
    return _CampaignBootstrapUnit(
        true_positives=counts["true_positives"],
        false_positives=counts["false_positives"],
        false_negatives=counts["false_negatives"],
        true_negatives=counts["true_negatives"],
        fraud_campaign=len(fraud_campaigns),
        detected_campaign=len(alerts),
        fraudulent_value=fraudulent_value,
        preventable_value=preventable_value,
    )


def _campaign_bootstrap_contribution(
    campaign_id: str, rows: tuple[_MetricRow, ...]
) -> CampaignBootstrapContribution:
    unit = _bootstrap_unit(rows)
    return CampaignBootstrapContribution(
        campaign_id=campaign_id,
        true_positives=unit.true_positives,
        false_positives=unit.false_positives,
        false_negatives=unit.false_negatives,
        true_negatives=unit.true_negatives,
        fraud_campaign=unit.fraud_campaign,
        detected_campaign=unit.detected_campaign,
        fraudulent_value=unit.fraudulent_value,
        preventable_value=unit.preventable_value,
    )


def _bootstrap_intervals(
    contributions: tuple[CampaignBootstrapContribution, ...],
    *,
    seed: int,
    replicates: int,
) -> tuple[ConfidenceInterval, ...]:
    if len(contributions) > MAX_METRIC_CAMPAIGNS:
        raise MetricContractError("campaign count exceeds metric resource cap")
    if len(contributions) * replicates > MAX_BOOTSTRAP_WORK:
        raise MetricContractError("campaign bootstrap exceeds frozen work cap")
    units = tuple(
        _CampaignBootstrapUnit(
            true_positives=row.true_positives,
            false_positives=row.false_positives,
            false_negatives=row.false_negatives,
            true_negatives=row.true_negatives,
            fraud_campaign=row.fraud_campaign,
            detected_campaign=row.detected_campaign,
            fraudulent_value=row.fraudulent_value,
            preventable_value=row.preventable_value,
        )
        for row in contributions
    )
    values: dict[MetricName, list[float]] = {
        name: [] for name in _BOOTSTRAP_METRICS
    }
    generator = np.random.Generator(np.random.PCG64(seed))
    for _ in range(replicates):
        selected = (
            generator.integers(0, len(units), size=len(units), endpoint=False)
            if units
            else np.asarray([], dtype=np.int64)
        )
        totals = _sum_bootstrap_units(tuple(units[int(index)] for index in selected))
        _append_ratio(
            values["precision"],
            totals.true_positives,
            totals.true_positives + totals.false_positives,
        )
        _append_ratio(
            values["recall"],
            totals.true_positives,
            totals.true_positives + totals.false_negatives,
        )
        _append_ratio(
            values["f1"],
            2 * totals.true_positives,
            2 * totals.true_positives + totals.false_positives + totals.false_negatives,
        )
        _append_ratio(
            values["false_positive_rate"],
            totals.false_positives,
            totals.false_positives + totals.true_negatives,
        )
        _append_ratio(
            values["campaign_recall"],
            totals.detected_campaign,
            totals.fraud_campaign,
        )
        values["fraudulent_net_settled_value"].append(
            _finite_decimal_float(totals.fraudulent_value)
        )
        values["preventable_settled_value"].append(
            _finite_decimal_float(totals.preventable_value)
        )
        values["value_escaped"].append(
            _finite_decimal_float(
                _money_subtract(totals.fraudulent_value, totals.preventable_value)
            )
        )
    return tuple(
        _confidence_interval(name, values[name], replicates)
        for name in _BOOTSTRAP_METRICS
    )


def _sum_bootstrap_units(
    units: tuple[_CampaignBootstrapUnit, ...],
) -> _CampaignBootstrapUnit:
    return _CampaignBootstrapUnit(
        true_positives=sum(row.true_positives for row in units),
        false_positives=sum(row.false_positives for row in units),
        false_negatives=sum(row.false_negatives for row in units),
        true_negatives=sum(row.true_negatives for row in units),
        fraud_campaign=sum(row.fraud_campaign for row in units),
        detected_campaign=sum(row.detected_campaign for row in units),
        fraudulent_value=_money_sum(row.fraudulent_value for row in units),
        preventable_value=_money_sum(row.preventable_value for row in units),
    )


def _append_ratio(values: list[float], numerator: int, denominator: int) -> None:
    if denominator:
        values.append(numerator / denominator)


def _confidence_interval(
    metric_name: MetricName, values: list[float], replicates: int
) -> ConfidenceInterval:
    if not values:
        return ConfidenceInterval(
            metric_name=metric_name,
            lower=None,
            median=None,
            upper=None,
            valid_replicates=0,
            undefined_replicates=replicates,
            undefined_reason="no_defined_bootstrap_replicates",
        )
    array = np.asarray(values, dtype=np.float64)
    lower, median, upper = np.percentile(array, [2.5, 50.0, 97.5], method="linear")
    return ConfidenceInterval(
        metric_name=metric_name,
        lower=float(lower),
        median=float(median),
        upper=float(upper),
        valid_replicates=len(values),
        undefined_replicates=replicates - len(values),
    )


def _ratio(numerator: int, denominator: int, reason: str) -> MetricValue:
    if denominator == 0:
        return MetricValue(
            value=None,
            numerator=float(numerator),
            denominator=0.0,
            undefined_reason=reason,
        )
    return MetricValue(
        value=numerator / denominator,
        numerator=float(numerator),
        denominator=float(denominator),
    )


def _scaled_ratio(
    numerator: int, denominator: int, scale: int, reason: str
) -> MetricValue:
    if denominator == 0:
        return MetricValue(
            value=None,
            numerator=float(numerator),
            denominator=0.0,
            undefined_reason=reason,
        )
    return MetricValue(
        value=numerator * scale / denominator,
        numerator=float(numerator),
        denominator=float(denominator),
    )


def _validate_decimal128_number(value: Decimal, *, label: str) -> None:
    if type(value) is not Decimal or not value.is_finite():
        raise MetricContractError(f"{label} must be an exact finite Decimal")
    if value and value.adjusted() > _MAX_DECIMAL_ADJUSTED:
        raise MetricContractError(f"{label} exceeds numeric resource bounds")
    try:
        with localcontext(_DECIMAL128_CONTEXT) as context:
            context.plus(value)
    except DecimalException as error:
        raise MetricContractError(f"{label} exceeds Decimal128 bounds") from error


def _validate_cent_amount(value: Decimal, *, label: str) -> None:
    _validate_decimal128_number(value, label=label)
    if value < 0:
        raise MetricContractError(f"{label} must be nonnegative")
    exponent = value.as_tuple().exponent
    if type(exponent) is not int or exponent < -2:
        raise MetricContractError(f"{label} must have at most two fractional digits")
    try:
        with localcontext(_DECIMAL128_CONTEXT) as context:
            quantized = value.quantize(_MONEY_QUANTUM, context=context)
    except DecimalException as error:
        raise MetricContractError(f"{label} exceeds Decimal128 bounds") from error
    if quantized != value:
        raise MetricContractError(f"{label} must have at most two fractional digits")


def _money_add(left: Decimal, right: Decimal) -> Decimal:
    try:
        with localcontext(_DECIMAL128_CONTEXT) as context:
            return context.add(left, right).quantize(_MONEY_QUANTUM, context=context)
    except DecimalException as error:
        raise MetricContractError("money arithmetic exceeds Decimal128 bounds") from error


def _money_subtract(
    left: Decimal, right: Decimal, *, allow_negative: bool = False
) -> Decimal:
    try:
        with localcontext(_DECIMAL128_CONTEXT) as context:
            result = context.subtract(left, right).quantize(
                _MONEY_QUANTUM, context=context
            )
    except DecimalException as error:
        raise MetricContractError("money arithmetic exceeds Decimal128 bounds") from error
    if result < 0 and not allow_negative:
        raise MetricContractError("money subtraction produced a negative result")
    return result


def _money_sum(values: Iterable[Decimal]) -> Decimal:
    result = Decimal("0.00")
    for value in values:
        result = _money_add(result, value)
    return result


def _decimal_multiply(left: Decimal, right: Decimal) -> Decimal:
    try:
        with localcontext(_DECIMAL128_CONTEXT) as context:
            result = context.multiply(left, right)
    except DecimalException as error:
        raise MetricContractError("Decimal ratio arithmetic exceeds Decimal128 bounds") from error
    _validate_decimal128_number(result, label="Decimal ratio result")
    return result


def _decimal_divide(numerator: Decimal, denominator: Decimal) -> Decimal:
    try:
        with localcontext(_DECIMAL128_CONTEXT) as context:
            result = context.divide(numerator, denominator)
    except DecimalException as error:
        raise MetricContractError("Decimal ratio arithmetic exceeds Decimal128 bounds") from error
    _validate_decimal128_number(result, label="Decimal ratio result")
    return result


def _decimal_ratio(
    numerator: Decimal, denominator: Decimal, reason: str
) -> DecimalMetricValue:
    if denominator == 0:
        return DecimalMetricValue(
            value=None,
            numerator=numerator,
            denominator=denominator,
            undefined_reason=reason,
        )
    return DecimalMetricValue(
        value=_decimal_divide(numerator, denominator),
        numerator=numerator,
        denominator=denominator,
    )


def _assert_metric(
    metric: MetricValue,
    numerator: int,
    denominator: int,
    reason: str,
    *,
    label: str,
) -> None:
    if metric.numerator != float(numerator) or metric.denominator != float(denominator):
        raise ValueError(f"{label} numerator or denominator differs")
    if denominator == 0:
        if metric.value is not None or metric.undefined_reason != reason:
            raise ValueError(f"{label} must be explicitly undefined")
    elif (
        metric.value is None
        or metric.undefined_reason is not None
        or not math.isclose(metric.value, numerator / denominator, abs_tol=1e-15)
    ):
        raise ValueError(f"{label} value differs from its counts")


def _assert_scaled_metric(
    metric: MetricValue,
    numerator: int,
    denominator: int,
    scale: int,
    reason: str,
    *,
    label: str,
) -> None:
    if metric.numerator != float(numerator) or metric.denominator != float(denominator):
        raise ValueError(f"{label} numerator or denominator differs")
    if denominator == 0:
        if metric.value is not None or metric.undefined_reason != reason:
            raise ValueError(f"{label} must be explicitly undefined")
    elif (
        metric.value is None
        or metric.undefined_reason is not None
        or not math.isclose(metric.value, numerator * scale / denominator, abs_tol=1e-12)
    ):
        raise ValueError(f"{label} value differs from its counts")


def _assert_decimal_metric(
    metric: DecimalMetricValue,
    numerator: Decimal,
    denominator: Decimal,
    reason: str,
    *,
    label: str,
) -> None:
    if metric.numerator != numerator or metric.denominator != denominator:
        raise ValueError(f"{label} numerator or denominator differs")
    if denominator == 0:
        if metric.value is not None or metric.undefined_reason != reason:
            raise ValueError(f"{label} must be explicitly undefined")
    elif (
        metric.value != _decimal_divide(numerator, denominator)
        or metric.undefined_reason is not None
    ):
        raise ValueError(f"{label} value differs from its exact ratio")


def _validate_confusion_fields(
    value: ClassificationMetrics | ClassificationSlice, *, label: str
) -> None:
    if value.fraud_count + value.legitimate_count != value.row_count:
        raise ValueError(f"{label} class counts must sum to rows")
    if (
        value.true_positives
        + value.false_positives
        + value.false_negatives
        + value.true_negatives
        != value.row_count
    ):
        raise ValueError(f"{label} confusion counts must sum to rows")
    if value.true_positives + value.false_negatives != value.fraud_count:
        raise ValueError(f"{label} fraud confusion counts differ")
    if value.false_positives + value.true_negatives != value.legitimate_count:
        raise ValueError(f"{label} legitimate confusion counts differ")
    _assert_metric(
        value.precision,
        value.true_positives,
        value.true_positives + value.false_positives,
        "no_predicted_positives",
        label=f"{label} precision",
    )
    _assert_metric(
        value.recall,
        value.true_positives,
        value.fraud_count,
        "absent_positive_class",
        label=f"{label} recall",
    )
    _assert_metric(
        value.f1,
        2 * value.true_positives,
        2 * value.true_positives + value.false_positives + value.false_negatives,
        "no_positive_truth_or_predictions",
        label=f"{label} F1",
    )
    _assert_metric(
        value.false_positive_rate,
        value.false_positives,
        value.legitimate_count,
        "absent_legitimate_class",
        label=f"{label} false-positive rate",
    )


def _validate_extended_classification_fields(
    value: ClassificationMetrics | ClassificationSlice, *, label: str
) -> None:
    _assert_metric(
        value.decline_precision,
        value.decline_true_positives,
        value.decline_true_positives + value.decline_false_positives,
        "no_declines",
        label=f"{label} decline precision",
    )
    _assert_metric(
        value.decline_recall,
        value.decline_true_positives,
        value.fraud_count,
        "absent_positive_class",
        label=f"{label} decline recall",
    )
    _assert_metric(
        value.campaign_recall,
        value.detected_fraud_campaigns,
        value.fraud_campaigns,
        "absent_fraud_campaigns",
        label=f"{label} campaign recall",
    )
    if value.detected_fraud_campaigns > value.fraud_campaigns:
        raise ValueError(f"{label} detected campaigns exceed fraud campaigns")
    for auc_label, metric in (("PR-AUC", value.pr_auc), ("ROC-AUC", value.roc_auc)):
        if metric.value is not None and not 0.0 <= metric.value <= 1.0:
            raise ValueError(f"{label} {auc_label} must be in [0, 1]")
        if metric.denominator != float(value.row_count):
            raise ValueError(f"{label} {auc_label} denominator must equal row count")
        if value.fraud_count == 0:
            if (
                metric.value is not None
                or metric.numerator != 0.0
                or metric.undefined_reason != "absent_positive_class"
            ):
                raise ValueError(
                    f"{label} {auc_label} must expose the absent positive class"
                )
        elif value.legitimate_count == 0:
            if (
                metric.value is not None
                or metric.numerator != float(value.fraud_count)
                or metric.undefined_reason != "absent_negative_class"
            ):
                raise ValueError(
                    f"{label} {auc_label} must expose the absent negative class"
                )
        elif (
            metric.value is None
            or metric.undefined_reason is not None
            or not math.isclose(
                metric.numerator,
                metric.value * value.row_count,
                abs_tol=1e-12,
            )
        ):
            raise ValueError(
                f"{label} {auc_label} numerator must bind its value and row count"
            )


def _validate_slice_rollup(
    classification: ClassificationMetrics,
    slices: tuple[ClassificationSlice, ...],
    *,
    kind: str,
) -> None:
    for field_name in (
        "row_count",
        "fraud_count",
        "legitimate_count",
        "true_positives",
        "false_positives",
        "false_negatives",
        "true_negatives",
    ):
        expected = getattr(classification, field_name)
        actual = sum(getattr(item, field_name) for item in slices)
        if actual != expected:
            raise ValueError(f"classification {kind} slice rollup differs for {field_name}")


def _slice_key(row: ClassificationSlice) -> tuple[int, str]:
    order = {"family": 0, "rail": 1, "regime": 2, "entity_cohort": 3}
    return order[row.kind], row.value


def _finite_decimal_float(value: Decimal) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise MetricContractError("Decimal aggregate exceeds finite bootstrap bounds")
    return result


def _input_digest(inputs: MetricReportInputs) -> str:
    document = {
        "as_of": inputs.as_of,
        "truth": tuple(sorted(inputs.truth, key=lambda row: row.event_id)),
        "observations": tuple(sorted(inputs.observations, key=lambda row: row.event_id)),
        "decisions": tuple(sorted(inputs.decisions, key=lambda row: row.event_id)),
        "cases": tuple(sorted(inputs.cases, key=lambda row: row.case_id)),
        "queue_report_digest": inputs.queue_report.report_digest,
        "latency_samples": tuple(
            sorted(inputs.latency_samples, key=lambda row: row.event_id)
        ),
        "slice_assignments": tuple(
            sorted(inputs.slice_assignments, key=lambda row: row.event_id)
        ),
        "slice_manifest": inputs.slice_manifest,
    }
    return _digest_document(_json_tree(document))


def _restore_metric_report_json_tuples(document: dict[str, object]) -> None:
    """Restore only public exact-tuple fields represented as canonical JSON arrays."""
    evidence = document.get("derivation_evidence")
    if type(evidence) is not dict:
        return
    inputs = cast(dict[str, object], evidence).get("inputs")
    if type(inputs) is not dict:
        return
    input_document = cast(dict[str, object], inputs)
    for field_name in (
        "truth",
        "observations",
        "decisions",
        "cases",
        "latency_samples",
        "slice_assignments",
    ):
        value = input_document.get(field_name)
        if type(value) is list:
            input_document[field_name] = tuple(cast(list[object], value))
    manifest = input_document.get("slice_manifest")
    if type(manifest) is not dict:
        return
    manifest_document = cast(dict[str, object], manifest)
    for field_name in ("regimes", "entity_cohorts"):
        value = manifest_document.get(field_name)
        if type(value) is list:
            manifest_document[field_name] = tuple(cast(list[object], value))


def _json_tree(value: object) -> object:
    if isinstance(value, ExternalContract):
        return value.model_dump(mode="json")
    if type(value) is dict:
        return {str(key): _json_tree(item) for key, item in value.items()}
    if type(value) is tuple:
        return [_json_tree(item) for item in value]
    if type(value) is datetime:
        return value.isoformat().replace("+00:00", "Z")
    return value


def _digest_document(document: object) -> str:
    return hashlib.sha256(canonical_json_bytes(document)).hexdigest()


def _validate_sha256(value: str) -> None:
    if type(value) is not str or len(value) != 64 or value != value.lower():
        raise ValueError("metric digests must be lowercase SHA-256")
    try:
        int(value, 16)
    except ValueError as error:
        raise ValueError("metric digests must be lowercase SHA-256") from error


def _exact_integer(value: object) -> object:
    if type(value) is not int:
        raise ValueError("metric counts must be exact integers")
    return value
