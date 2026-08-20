"""Fixture-safe, conservative constrained selection contracts for Defend v2.

This module only evaluates supplied metric evidence.  It neither constructs an
operating population nor runs a defender or an evaluator replay.
"""

from __future__ import annotations

import math
import random
from collections.abc import Mapping, Sequence
from datetime import date
from typing import TYPE_CHECKING, Literal

from pydantic import Field, field_validator, model_validator

from apar.contracts._validation import ExternalContract
from apar.evaluation.v2_controls import ControlValidity, V2ControlContext
from apar.evaluation.v2_preregistration import V2Preregistration
from apar.evaluation.v2_protocol import V2Protocol

if TYPE_CHECKING:
    from apar.cases.v2_workload import ActionWorkload
    from apar.evaluation.metrics import MetricReport, MetricValue

V2_BOOTSTRAP_REPLICATES = 2_000
_STRATA: tuple[Literal["low"], Literal["medium"], Literal["high"]] = (
    "low",
    "medium",
    "high",
)
_METRIC_FIELDS = (
    "precision",
    "recall",
    "f1",
    "pr_auc",
    "roc_auc",
    "ece",
    "fpr",
    "challenge_rate",
    "false_decline_rate",
    "review_case_rate",
    "false_interventions_per_10k",
    "preventable_settled_value_fraction",
    "escaped_value_fraction",
    "time_to_alert_p95_seconds",
    "p95_decision_latency_ms",
)


class BoundedMetric(ExternalContract):
    """A point metric, its conservative interval, and retained derivation counts."""

    point: float | None
    lower: float | None
    upper: float | None
    numerator: float | None = None
    denominator: float | None = None
    bootstrap_replicates: int = Field(default=0, ge=0)
    valid_replicates: int = Field(default=0, ge=0)
    undefined_replicates: int = Field(default=0, ge=0)

    @field_validator("point", "lower", "upper", "numerator", "denominator", mode="before")
    @classmethod
    def values_are_finite_nonnegative_floats_or_none(cls, value: object) -> object:
        if value is None:
            return value
        if type(value) is not float or not math.isfinite(value) or value < 0.0:
            raise ValueError("bounded metric values must be finite nonnegative floats or None")
        return value

    @field_validator(
        "bootstrap_replicates", "valid_replicates", "undefined_replicates", mode="before"
    )
    @classmethod
    def counts_are_exact_nonnegative_integers(cls, value: object) -> object:
        if type(value) is not int or value < 0:
            raise ValueError("bootstrap counts must be exact nonnegative integers")
        return value

    @model_validator(mode="after")
    def interval_and_evidence_are_coherent(self) -> BoundedMetric:
        values = (self.point, self.lower, self.upper)
        if any(value is None for value in values):
            if any(value is not None for value in values):
                raise ValueError("undefined bounded metric cannot claim partial bounds")
        else:
            assert self.point is not None and self.lower is not None and self.upper is not None
            if not self.lower <= self.point <= self.upper:
                raise ValueError("bounded metric bounds must contain the point estimate")
        if (self.numerator is None) != (self.denominator is None):
            raise ValueError("metric numerator and denominator must be supplied together")
        if self.bootstrap_replicates == 0:
            if self.valid_replicates or self.undefined_replicates:
                raise ValueError("unbootstrapped metrics cannot claim replicate counts")
        elif self.valid_replicates + self.undefined_replicates != self.bootstrap_replicates:
            raise ValueError("bootstrap replicate counts must equal the declared total")
        return self

    @property
    def defined(self) -> bool:
        return self.point is not None

    @classmethod
    def undefined(
        cls, *, numerator: float | None = None, denominator: float | None = None
    ) -> BoundedMetric:
        """Construct explicit undefined evidence without silently substituting zero."""
        return cls(
            point=None,
            lower=None,
            upper=None,
            numerator=numerator,
            denominator=denominator,
        )


class V2MetricSet(ExternalContract):
    """Classification, calibration, workload, value, and alert metric evidence."""

    precision: BoundedMetric
    recall: BoundedMetric
    f1: BoundedMetric
    pr_auc: BoundedMetric
    roc_auc: BoundedMetric
    ece: BoundedMetric
    fpr: BoundedMetric
    challenge_rate: BoundedMetric
    false_decline_rate: BoundedMetric
    review_case_rate: BoundedMetric
    false_interventions_per_10k: BoundedMetric
    preventable_settled_value_fraction: BoundedMetric
    escaped_value_fraction: BoundedMetric
    time_to_alert_p95_seconds: BoundedMetric
    p95_decision_latency_ms: BoundedMetric

    @property
    def metrics(self) -> tuple[BoundedMetric, ...]:
        return tuple(getattr(self, field) for field in _METRIC_FIELDS)

    @property
    def all_defined(self) -> bool:
        return all(metric.defined for metric in self.metrics)

    @classmethod
    def from_metric_report(cls, report: MetricReport, workload: ActionWorkload) -> V2MetricSet:
        """Project precomputed evidence without recalculating or executing a replay.

        Existing reports contain point estimates and exact numerator/denominator
        evidence. Until two-level bootstrap bounds are supplied, the projected
        metrics remain reporting-only evidence: their zero bootstrap count makes
        them ineligible for promotion.
        """
        from apar.cases.v2_workload import ActionWorkload
        from apar.evaluation.metrics import MetricReport

        if type(report) is not MetricReport or type(workload) is not ActionWorkload:
            raise TypeError("metric projection requires exact MetricReport and ActionWorkload")
        classification = report.classification
        value_total = float(report.value.fraudulent_net_settled_value)
        captured = float(report.value.preventable_settled_value)
        escaped = float(report.value.value_escaped)
        return cls(
            precision=_from_metric_value(classification.precision),
            recall=_from_metric_value(classification.recall),
            f1=_from_metric_value(classification.f1),
            pr_auc=_from_metric_value(classification.pr_auc),
            roc_auc=_from_metric_value(classification.roc_auc),
            ece=_from_metric_value(report.calibration.ece),
            fpr=_from_metric_value(classification.false_positive_rate),
            challenge_rate=_ratio_metric(
                workload.challenge_count, workload.total_transaction_count
            ),
            false_decline_rate=_ratio_metric(
                workload.false_decline_count, workload.legitimate_transaction_count
            ),
            review_case_rate=_ratio_metric(
                workload.review_case_count, workload.total_transaction_count
            ),
            false_interventions_per_10k=_scaled_metric(
                workload.false_intervention_count, workload.total_transaction_count, 10_000.0
            ),
            preventable_settled_value_fraction=_ratio_metric(captured, value_total),
            escaped_value_fraction=_ratio_metric(escaped, value_total),
            time_to_alert_p95_seconds=_from_metric_value(report.alerts.p95_seconds),
            p95_decision_latency_ms=_from_metric_value(report.engineering.end_to_end_ms.p95),
        )


class ArmThresholdCandidate(ExternalContract):
    """One arm's precomputed threshold candidate over matched metric scopes."""

    candidate_id: str = Field(min_length=1)
    threshold_tuple: tuple[float, ...] = Field(min_length=1)
    metrics: V2MetricSet
    strata: dict[Literal["low", "medium", "high"], V2MetricSet]
    families: dict[str, V2MetricSet]
    control: object

    @field_validator("candidate_id", mode="before")
    @classmethod
    def candidate_id_is_bounded_text(cls, value: object) -> object:
        if type(value) is not str or not value or len(value) > 256:
            raise ValueError("candidate ID must be bounded nonempty text")
        return value

    @field_validator("threshold_tuple", mode="before")
    @classmethod
    def thresholds_are_exact_finite_tuple(cls, value: object) -> object:
        if type(value) is not tuple or not value:
            raise ValueError("threshold tuple must be nonempty exact tuple")
        if any(type(item) is not float or not math.isfinite(item) for item in value):
            raise ValueError("thresholds must be exact finite floats")
        return value

    @model_validator(mode="after")
    def scopes_are_complete(self) -> ArmThresholdCandidate:
        if set(self.strata) != set(_STRATA):
            raise ValueError("candidate strata must be complete and exact")
        if len(self.families) != 4 or any(not name for name in self.families):
            raise ValueError("candidate families must contain four nonempty names")
        if type(self.control) is not ControlValidity:
            raise ValueError("candidate requires exact mandatory control evidence")
        return self

    @property
    def minimum_family_captured_value_lower_bound(self) -> float:
        return min(
            _required_bound(metric.preventable_settled_value_fraction.lower)
            for metric in self.families.values()
        )

    @property
    def maximum_review_case_rate_upper_bound(self) -> float:
        return max(
            _required_bound(metric.review_case_rate.upper) for metric in self.strata.values()
        )

    @property
    def maximum_false_decline_rate_upper_bound(self) -> float:
        return max(
            _required_bound(metric.false_decline_rate.upper) for metric in self.strata.values()
        )

    @property
    def maximum_challenge_rate_upper_bound(self) -> float:
        return max(_required_bound(metric.challenge_rate.upper) for metric in self.strata.values())

    @property
    def p95_decision_latency_upper_bound(self) -> float:
        assert self.metrics.p95_decision_latency_ms.upper is not None
        return self.metrics.p95_decision_latency_ms.upper


class V2GateOutcome(ExternalContract):
    """The complete fail-closed outcome for one supplied metric scope."""

    passed: bool
    codes: tuple[str, ...] = ()

    @field_validator("passed", mode="before")
    @classmethod
    def passed_is_exact_bool(cls, value: object) -> object:
        if type(value) is not bool:
            raise ValueError("gate outcome passed must be an exact bool")
        return value

    @field_validator("codes", mode="before")
    @classmethod
    def codes_are_exact_tuple(cls, value: object) -> object:
        if type(value) is not tuple or any(type(code) is not str or not code for code in value):
            raise ValueError("gate codes must be an exact tuple of nonempty strings")
        return value

    @model_validator(mode="after")
    def status_matches_codes(self) -> V2GateOutcome:
        if self.codes != tuple(sorted(set(self.codes))):
            raise ValueError("gate codes must be sorted and unique")
        if self.passed != (not self.codes):
            raise ValueError("gate status must agree with codes")
        return self


class V2SelectionReport(ExternalContract):
    """Selection result with every candidate's gate outcome retained."""

    status: Literal["selected", "no_promotion"]
    selected_candidate_id: str | None
    selected_threshold_tuple: tuple[float, ...] | None
    reason: str | None = None
    gate_outcomes: dict[str, V2GateOutcome]

    @model_validator(mode="after")
    def result_is_coherent(self) -> V2SelectionReport:
        if self.status == "selected":
            if self.selected_candidate_id is None or self.selected_threshold_tuple is None:
                raise ValueError("selected result requires candidate identity and thresholds")
            if self.reason is not None:
                raise ValueError("selected result cannot carry a no-promotion reason")
        elif (
            self.selected_candidate_id is not None
            or self.selected_threshold_tuple is not None
            or type(self.reason) is not str
            or not self.reason
        ):
            raise ValueError("no-promotion result requires only a nonempty reason")
        return self

    @classmethod
    def no_promotion(
        cls, reason: str, *, gate_outcomes: Mapping[str, V2GateOutcome] | None = None
    ) -> V2SelectionReport:
        return cls(
            status="no_promotion",
            selected_candidate_id=None,
            selected_threshold_tuple=None,
            reason=reason,
            gate_outcomes=dict(gate_outcomes or {}),
        )


class BootstrapMetricContribution(ExternalContract):
    """One case-block contribution to a ratio or p95 metric bootstrap."""

    kind: Literal["ratio", "p95"]
    numerator: float = Field(default=0.0, ge=0.0)
    denominator: float = Field(default=0.0, ge=0.0)
    samples: tuple[float, ...] = ()

    @field_validator("numerator", "denominator", mode="before")
    @classmethod
    def contribution_counts_are_finite_floats(cls, value: object) -> object:
        if type(value) is not float or not math.isfinite(value) or value < 0.0:
            raise ValueError("bootstrap contributions must be finite nonnegative floats")
        return value

    @field_validator("samples", mode="before")
    @classmethod
    def samples_are_exact_finite_tuple(cls, value: object) -> object:
        if type(value) is not tuple or any(
            type(item) is not float or not math.isfinite(item) or item < 0.0 for item in value
        ):
            raise ValueError("bootstrap samples must be an exact finite nonnegative float tuple")
        return value

    @model_validator(mode="after")
    def shape_matches_kind(self) -> BootstrapMetricContribution:
        if self.kind == "ratio" and self.samples:
            raise ValueError("ratio bootstrap contribution cannot carry samples")
        if self.kind == "p95" and (not self.samples or self.numerator or self.denominator):
            raise ValueError("p95 bootstrap contribution requires only nonempty samples")
        return self


class V2BootstrapBlock(ExternalContract):
    """One campaign/entity case block nested in one synthetic day."""

    day: date
    block_id: str = Field(min_length=1)
    metrics: dict[str, BootstrapMetricContribution]

    @field_validator("block_id", mode="before")
    @classmethod
    def block_id_is_bounded_text(cls, value: object) -> object:
        if type(value) is not str or not value or len(value) > 256:
            raise ValueError("bootstrap block ID must be bounded nonempty text")
        return value

    @model_validator(mode="after")
    def metrics_are_nonempty_and_named(self) -> V2BootstrapBlock:
        if not self.metrics or any(type(name) is not str or not name for name in self.metrics):
            raise ValueError("bootstrap block metrics must be a nonempty named mapping")
        return self


def bootstrap_v2_metrics(
    blocks: Sequence[V2BootstrapBlock], *, seed: int
) -> Mapping[str, BoundedMetric]:
    """Bootstrap supplied metric contributions by day, then by case block.

    The input is evaluator-supplied fixture evidence.  The function does not
    produce observations or decisions and always performs the sealed 2,000
    resamples.  Percentiles use deterministic linear interpolation.
    """
    if type(seed) is not int:
        raise TypeError("bootstrap seed must be an exact integer")
    checked = tuple(blocks)
    if not checked:
        raise ValueError("bootstrap requires at least one case block")
    if any(type(block) is not V2BootstrapBlock for block in checked):
        raise TypeError("bootstrap blocks must be exact V2BootstrapBlock values")
    if len({(block.day, block.block_id) for block in checked}) != len(checked):
        raise ValueError("bootstrap day/case blocks must be unique")
    names = tuple(sorted(checked[0].metrics))
    if any(tuple(sorted(block.metrics)) != names for block in checked[1:]):
        raise ValueError("bootstrap blocks must carry the same metric names")
    if any(len({block.metrics[name].kind for block in checked}) != 1 for name in names):
        raise ValueError("bootstrap metric kinds must agree across blocks")

    by_day: dict[date, tuple[V2BootstrapBlock, ...]] = {
        day: tuple(block for block in checked if block.day == day)
        for day in sorted({block.day for block in checked})
    }
    point = {name: _aggregate_metric(name, checked) for name in names}
    samples: dict[str, list[float]] = {name: [] for name in names}
    undefined: dict[str, int] = {name: 0 for name in names}
    generator = random.Random(seed)
    days = tuple(by_day)
    for _ in range(V2_BOOTSTRAP_REPLICATES):
        resampled: list[V2BootstrapBlock] = []
        for day in (generator.choice(days) for _ in days):
            options = by_day[day]
            resampled.extend(generator.choice(options) for _ in options)
        for name in names:
            value = _aggregate_metric(name, resampled)
            if value is None:
                undefined[name] += 1
            else:
                samples[name].append(value)
    return {
        name: _bounded_bootstrap_metric(point[name], samples[name], undefined[name], checked, name)
        for name in names
    }


def evaluate_v2_gates(
    evidence: V2MetricSet | ArmThresholdCandidate,
    protocol: V2Protocol,
    *,
    sealed_preregistration: V2Preregistration | None = None,
    control_context: V2ControlContext | None = None,
) -> V2GateOutcome:
    """Apply conservative all-scope gates to already-derived metric evidence."""
    if type(protocol) is not V2Protocol:
        raise TypeError("protocol must be an exact V2Protocol")
    strata: tuple[V2MetricSet, ...]
    families: tuple[V2MetricSet, ...]
    if type(evidence) is V2MetricSet:
        aggregate = evidence
        strata = (evidence,)
        families = (evidence,)
        control = None
    elif type(evidence) is ArmThresholdCandidate:
        aggregate = evidence.metrics
        strata = tuple(evidence.strata[name] for name in _STRATA)
        families = tuple(evidence.families.values())
        control = evidence.control if type(evidence.control) is ControlValidity else None
    else:
        raise TypeError("gates require exact V2MetricSet or ArmThresholdCandidate evidence")

    all_sets = (aggregate, *strata, *families)
    mandatory_metrics = tuple(metric for metric_set in all_sets for metric in metric_set.metrics)
    codes: set[str] = set()
    if type(evidence) is ArmThresholdCandidate and set(evidence.families) != set(
        protocol.operating.family_names
    ):
        codes.add("FAMILY_SCOPE_INVALID")
    if not all(metric.defined for metric in mandatory_metrics):
        codes.add("METRIC_UNDEFINED")
    if any(metric.bootstrap_replicates != V2_BOOTSTRAP_REPLICATES for metric in mandatory_metrics):
        codes.add("BOOTSTRAP_REPLICATES")
    if any(metric.undefined_replicates != 0 for metric in mandatory_metrics):
        codes.add("BOOTSTRAP_UNDEFINED")
    if type(evidence) is ArmThresholdCandidate and (
        control is None
        or type(sealed_preregistration) is not V2Preregistration
        or type(control_context) is not V2ControlContext
        or control_context.candidate_id != evidence.candidate_id
        or not control.valid_for(
            sealed_preregistration=sealed_preregistration,
            expected_context=control_context,
        )
    ):
        codes.add("CONTROL_INVALID")

    if any(not _at_least(metric.recall, 0.50) for metric in families):
        codes.add("FAMILY_RECALL")
    if not _at_most(aggregate.ece, 0.10):
        codes.add("CALIBRATION_ECE")
    if any(
        not _at_most(metric.challenge_rate, protocol.budgets.challenge_rate_max)
        for metric in strata
    ):
        codes.add("CHALLENGE_BUDGET")
    if any(
        not _at_most(metric.false_decline_rate, protocol.budgets.false_decline_rate_max)
        for metric in strata
    ):
        codes.add("FALSE_DECLINE_BUDGET")
    if any(
        not _at_most(metric.review_case_rate, protocol.budgets.review_case_rate_max)
        for metric in strata
    ):
        codes.add("REVIEW_CASE_BUDGET")
    if not _at_most(aggregate.p95_decision_latency_ms, 50.0):
        codes.add("DECISION_LATENCY")
    if any(not _at_least(metric.preventable_settled_value_fraction, 0.50) for metric in families):
        codes.add("CAPTURED_VALUE")
    if any(not _at_most(metric.escaped_value_fraction, 0.50) for metric in families):
        codes.add("ESCAPED_VALUE")
    if any(not _at_most(metric.time_to_alert_p95_seconds, 300.0) for metric in families):
        codes.add("TIME_TO_ALERT")
    ordered = tuple(sorted(codes))
    return V2GateOutcome(passed=not ordered, codes=ordered)


def _required_bound(value: float | None) -> float:
    if value is None:
        raise ValueError("selection tie-break requires defined metric bounds")
    return value


def select_v2_thresholds(
    candidates: Sequence[ArmThresholdCandidate],
    protocol: V2Protocol,
    *,
    sealed_preregistration: V2Preregistration,
    control_contexts: Mapping[str, V2ControlContext],
) -> V2SelectionReport:
    """Select only a candidate that passes every matched conservative gate."""
    if type(protocol) is not V2Protocol:
        raise TypeError("protocol must be an exact V2Protocol")
    if (
        type(sealed_preregistration) is not V2Preregistration
        or not sealed_preregistration.verify_signature()
        or not sealed_preregistration.verify_manifest_bindings()
    ):
        raise TypeError("selection requires an exact sealed preregistration")
    if type(control_contexts) is not dict:
        raise TypeError("selection requires exact independent control contexts")
    checked = tuple(candidates)
    if any(type(candidate) is not ArmThresholdCandidate for candidate in checked):
        raise TypeError("candidates must be exact ArmThresholdCandidate values")
    ids = tuple(candidate.candidate_id for candidate in checked)
    if len(ids) != len(set(ids)):
        raise ValueError("candidate IDs must be unique")
    if set(control_contexts) != set(ids) or any(
        type(context) is not V2ControlContext or context.candidate_id != candidate_id
        for candidate_id, context in control_contexts.items()
    ):
        raise ValueError("selection control contexts must match every exact candidate")
    outcomes = {
        candidate.candidate_id: evaluate_v2_gates(
            candidate,
            protocol,
            sealed_preregistration=sealed_preregistration,
            control_context=control_contexts[candidate.candidate_id],
        )
        for candidate in checked
    }
    feasible = tuple(candidate for candidate in checked if outcomes[candidate.candidate_id].passed)
    if not feasible:
        return V2SelectionReport.no_promotion(
            "no_candidate_satisfies_v2_constraints", gate_outcomes=outcomes
        )
    selected = min(
        feasible,
        key=lambda candidate: (
            -candidate.minimum_family_captured_value_lower_bound,
            candidate.maximum_review_case_rate_upper_bound,
            candidate.maximum_false_decline_rate_upper_bound,
            candidate.maximum_challenge_rate_upper_bound,
            candidate.p95_decision_latency_upper_bound,
            candidate.threshold_tuple,
        ),
    )
    return V2SelectionReport(
        status="selected",
        selected_candidate_id=selected.candidate_id,
        selected_threshold_tuple=selected.threshold_tuple,
        gate_outcomes=outcomes,
    )


def _from_metric_value(metric: MetricValue) -> BoundedMetric:
    if metric.value is None:
        return BoundedMetric.undefined(numerator=metric.numerator, denominator=metric.denominator)
    return BoundedMetric(
        point=metric.value,
        lower=metric.value,
        upper=metric.value,
        numerator=metric.numerator,
        denominator=metric.denominator,
    )


def _ratio_metric(numerator: float | int, denominator: float | int) -> BoundedMetric:
    numerator_value = float(numerator)
    denominator_value = float(denominator)
    if denominator_value == 0.0:
        return BoundedMetric.undefined(numerator=numerator_value, denominator=denominator_value)
    value = numerator_value / denominator_value
    return BoundedMetric(
        point=value,
        lower=value,
        upper=value,
        numerator=numerator_value,
        denominator=denominator_value,
    )


def _scaled_metric(numerator: int, denominator: int, multiplier: float) -> BoundedMetric:
    if denominator == 0:
        return BoundedMetric.undefined(numerator=float(numerator), denominator=0.0)
    value = numerator * multiplier / denominator
    return BoundedMetric(
        point=value,
        lower=value,
        upper=value,
        numerator=float(numerator),
        denominator=float(denominator),
    )


def _aggregate_metric(name: str, blocks: Sequence[V2BootstrapBlock]) -> float | None:
    contributions = tuple(block.metrics[name] for block in blocks)
    if contributions[0].kind == "ratio":
        numerator = math.fsum(item.numerator for item in contributions)
        denominator = math.fsum(item.denominator for item in contributions)
        return None if denominator == 0.0 else numerator / denominator
    values = tuple(value for item in contributions for value in item.samples)
    return _percentile(values, 95.0) if values else None


def _bounded_bootstrap_metric(
    point: float | None,
    samples: Sequence[float],
    undefined: int,
    blocks: Sequence[V2BootstrapBlock],
    name: str,
) -> BoundedMetric:
    contribution = blocks[0].metrics[name]
    if point is None or not samples:
        return BoundedMetric(
            point=None,
            lower=None,
            upper=None,
            numerator=(
                math.fsum(block.metrics[name].numerator for block in blocks)
                if contribution.kind == "ratio"
                else None
            ),
            denominator=(
                math.fsum(block.metrics[name].denominator for block in blocks)
                if contribution.kind == "ratio"
                else None
            ),
            bootstrap_replicates=V2_BOOTSTRAP_REPLICATES,
            valid_replicates=len(samples),
            undefined_replicates=undefined,
        )
    numerator = (
        math.fsum(block.metrics[name].numerator for block in blocks)
        if contribution.kind == "ratio"
        else point * float(sum(len(block.metrics[name].samples) for block in blocks))
    )
    denominator = (
        math.fsum(block.metrics[name].denominator for block in blocks)
        if contribution.kind == "ratio"
        else float(sum(len(block.metrics[name].samples) for block in blocks))
    )
    return BoundedMetric(
        point=point,
        lower=_percentile(samples, 2.5),
        upper=_percentile(samples, 97.5),
        numerator=numerator,
        denominator=denominator,
        bootstrap_replicates=V2_BOOTSTRAP_REPLICATES,
        valid_replicates=len(samples),
        undefined_replicates=undefined,
    )


def _percentile(values: Sequence[float], percentage: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("cannot calculate percentile of empty values")
    position = (len(ordered) - 1) * percentage / 100.0
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _at_least(metric: BoundedMetric, limit: float) -> bool:
    return metric.lower is not None and metric.lower >= limit


def _at_most(metric: BoundedMetric, limit: float) -> bool:
    return metric.upper is not None and metric.upper <= limit


__all__ = [
    "ArmThresholdCandidate",
    "BootstrapMetricContribution",
    "BoundedMetric",
    "ControlValidity",
    "V2ControlContext",
    "V2BootstrapBlock",
    "V2GateOutcome",
    "V2MetricSet",
    "V2SelectionReport",
    "V2_BOOTSTRAP_REPLICATES",
    "bootstrap_v2_metrics",
    "evaluate_v2_gates",
    "select_v2_thresholds",
]
