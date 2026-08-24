"""Complete denominator-explicit metrics and ledger economics for Sentinel v5."""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from datetime import datetime
from decimal import Decimal
from types import MappingProxyType
from typing import Self

import numpy as np
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_serializer,
    field_validator,
    model_validator,
)
from sklearn.metrics import (  # type: ignore[import-untyped]
    average_precision_score,
    roc_auc_score,
)

from apar.contracts.events import PaymentEvent
from apar.defense.sentinel import SentinelAction
from apar.evaluation.v5_evaluation import (
    V5Arm,
    V5ArmRowEvidence,
    V5ArmSupportRow,
    V5EvaluationResult,
    V5ExecutionArtifact,
)
from apar.evaluation.v5_evidence_protocol import (
    V5BootstrapProtocol,
    V5EconomicProtocol,
    V5EvidenceProtocol,
    V5MetricApplicability,
)

_MAX_ROWS = 100_000
_DETECTION_ACTIONS = {
    SentinelAction.CHALLENGE,
    SentinelAction.REVIEW_HOLD,
    SentinelAction.DECLINE_HOLD,
}


def _digest(document: object) -> str:
    return hashlib.sha256(
        json.dumps(
            _jsonable(document),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode()
    ).hexdigest()


def _jsonable(value: object) -> object:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, datetime):
        rendered = value.isoformat()
        return f"{rendered[:-6]}Z" if rendered.endswith("+00:00") else rendered
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    return value


class V5MetricEstimate(BaseModel):
    """One metric with explicit mathematical state and denominator evidence."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    applicability: V5MetricApplicability
    value: float | None
    numerator: float | None
    denominator: float | None
    support_count: int = Field(ge=0, le=_MAX_ROWS)
    support_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    formula: str
    metric_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def state_and_digest_are_consistent(self) -> Self:
        numeric = (self.value, self.numerator, self.denominator)
        if any(item is not None and not math.isfinite(item) for item in numeric):
            raise ValueError("metric values must be finite")
        if self.applicability is V5MetricApplicability.DEFINED:
            if self.value is None or self.numerator is None or self.denominator is None:
                raise ValueError("defined metric requires value, numerator, and denominator")
            if self.denominator <= 0:
                raise ValueError("defined metric denominator must be positive")
        elif self.applicability is V5MetricApplicability.UNDEFINED:
            if self.value is not None or self.numerator is None or self.denominator != 0:
                raise ValueError("undefined metric requires numerator and zero denominator")
        elif any(item is not None for item in numeric):
            raise ValueError("unavailable/not-applicable metric cannot claim numeric values")
        expected = _digest(self.model_dump(mode="json", exclude={"metric_sha256"}))
        if self.metric_sha256 != expected:
            raise ValueError("metric digest mismatch")
        return self


def _metric(
    *,
    name: str,
    applicability: V5MetricApplicability,
    value: float | None,
    numerator: float | None,
    denominator: float | None,
    support_ids: Sequence[str],
    formula: str,
) -> V5MetricEstimate:
    values = {
        "name": name,
        "applicability": applicability.value,
        "value": value,
        "numerator": numerator,
        "denominator": denominator,
        "support_count": len(support_ids),
        "support_sha256": _digest(tuple(support_ids)),
        "formula": formula,
    }
    values["metric_sha256"] = _digest(values)
    return V5MetricEstimate.model_validate(values)


def _ratio_metric(
    *,
    name: str,
    numerator: float,
    denominator: float,
    support_ids: Sequence[str],
    formula: str,
) -> V5MetricEstimate:
    if denominator == 0:
        return _metric(
            name=name,
            applicability=V5MetricApplicability.UNDEFINED,
            value=None,
            numerator=numerator,
            denominator=0.0,
            support_ids=support_ids,
            formula=formula,
        )
    return _metric(
        name=name,
        applicability=V5MetricApplicability.DEFINED,
        value=numerator / denominator,
        numerator=numerator,
        denominator=denominator,
        support_ids=support_ids,
        formula=formula,
    )


def _not_applicable_metric(
    *, name: str, support_ids: Sequence[str], formula: str
) -> V5MetricEstimate:
    return _metric(
        name=name,
        applicability=V5MetricApplicability.NOT_APPLICABLE,
        value=None,
        numerator=None,
        denominator=None,
        support_ids=support_ids,
        formula=formula,
    )


class V5CalibrationBin(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    lower: float
    upper: float
    final_closed: bool
    count: int = Field(ge=0)
    mean_probability: float | None
    empirical_rate: float | None
    absolute_gap: float | None
    event_ids: tuple[str, ...] = Field(max_length=_MAX_ROWS)
    bin_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def bin_is_complete(self) -> Self:
        if self.count != len(self.event_ids):
            raise ValueError("calibration bin count disagrees with event IDs")
        statistics = (self.mean_probability, self.empirical_rate, self.absolute_gap)
        if self.count == 0 and any(value is not None for value in statistics):
            raise ValueError("empty calibration bin cannot claim statistics")
        if self.count > 0 and any(value is None for value in statistics):
            raise ValueError("nonempty calibration bin requires statistics")
        if self.bin_sha256 != _digest(self.model_dump(mode="json", exclude={"bin_sha256"})):
            raise ValueError("calibration bin digest mismatch")
        return self


class V5CalibrationEvidence(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    applicability: V5MetricApplicability
    boundaries: tuple[float, ...]
    bins: tuple[V5CalibrationBin, ...]
    expected_calibration_error: V5MetricEstimate
    maximum_calibration_error: V5MetricEstimate
    calibration_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def calibration_is_bound(self) -> Self:
        if self.applicability is V5MetricApplicability.DEFINED:
            if len(self.boundaries) != len(self.bins) + 1:
                raise ValueError("calibration boundaries and bins disagree")
        elif self.bins:
            raise ValueError("non-applicable calibration cannot retain bins")
        if self.calibration_sha256 != _digest(
            self.model_dump(mode="json", exclude={"calibration_sha256"})
        ):
            raise ValueError("calibration evidence digest mismatch")
        return self


def compute_v5_calibration(
    *,
    labels: Sequence[int],
    probabilities: Sequence[float] | None,
    boundaries: Sequence[float],
    support_ids: Sequence[str],
    applicable: bool,
) -> V5CalibrationEvidence:
    """Compute fixed-bin ECE/MCE with exact bin membership evidence."""
    if len(labels) != len(support_ids) or not labels:
        raise ValueError("calibration labels and support must align and be nonempty")
    canonical_boundaries = tuple(float(value) for value in boundaries)
    if canonical_boundaries != tuple(index / 10 for index in range(11)):
        raise ValueError("calibration requires the frozen ten-bin boundaries")
    if not applicable:
        ece = _not_applicable_metric(
            name="expected_calibration_error",
            support_ids=support_ids,
            formula="not applicable to deterministic rules",
        )
        mce = _not_applicable_metric(
            name="maximum_calibration_error",
            support_ids=support_ids,
            formula="not applicable to deterministic rules",
        )
        values = {
            "applicability": V5MetricApplicability.NOT_APPLICABLE.value,
            "boundaries": canonical_boundaries,
            "bins": (),
            "expected_calibration_error": ece,
            "maximum_calibration_error": mce,
        }
        values["calibration_sha256"] = _digest(
            {
                **values,
                "expected_calibration_error": ece.model_dump(mode="json"),
                "maximum_calibration_error": mce.model_dump(mode="json"),
            }
        )
        return V5CalibrationEvidence.model_validate(values)
    if probabilities is None or len(probabilities) != len(labels):
        raise ValueError("applicable calibration requires aligned probabilities")
    if any(
        not math.isfinite(float(value)) or not 0 <= float(value) <= 1 for value in probabilities
    ):
        raise ValueError("calibration probabilities must be finite in [0, 1]")
    bins: list[V5CalibrationBin] = []
    weighted_gap = 0.0
    maximum_gap = 0.0
    for index, (lower, upper) in enumerate(
        zip(canonical_boundaries, canonical_boundaries[1:], strict=False)
    ):
        final = index == len(canonical_boundaries) - 2
        indices = tuple(
            position
            for position, probability in enumerate(probabilities)
            if probability >= lower and (probability <= upper if final else probability < upper)
        )
        mean_probability: float | None
        empirical_rate: float | None
        gap: float | None
        if indices:
            mean_probability = sum(float(probabilities[item]) for item in indices) / len(indices)
            empirical_rate = sum(int(labels[item]) for item in indices) / len(indices)
            gap = abs(mean_probability - empirical_rate)
            weighted_gap += len(indices) / len(labels) * gap
            maximum_gap = max(maximum_gap, gap)
        else:
            mean_probability = empirical_rate = gap = None
        values = {
            "lower": lower,
            "upper": upper,
            "final_closed": final,
            "count": len(indices),
            "mean_probability": mean_probability,
            "empirical_rate": empirical_rate,
            "absolute_gap": gap,
            "event_ids": tuple(support_ids[item] for item in indices),
        }
        values["bin_sha256"] = _digest(values)
        bins.append(V5CalibrationBin.model_validate(values))
    ece = _metric(
        name="expected_calibration_error",
        applicability=V5MetricApplicability.DEFINED,
        value=weighted_gap,
        numerator=weighted_gap * len(labels),
        denominator=float(len(labels)),
        support_ids=support_ids,
        formula="sum(bin_count/total*absolute_gap)",
    )
    mce = _metric(
        name="maximum_calibration_error",
        applicability=V5MetricApplicability.DEFINED,
        value=maximum_gap,
        numerator=maximum_gap,
        denominator=1.0,
        support_ids=support_ids,
        formula="max(bin_absolute_gap)",
    )
    values = {
        "applicability": V5MetricApplicability.DEFINED.value,
        "boundaries": canonical_boundaries,
        "bins": tuple(bins),
        "expected_calibration_error": ece,
        "maximum_calibration_error": mce,
    }
    values["calibration_sha256"] = _digest(
        {
            **values,
            "bins": [item.model_dump(mode="json") for item in bins],
            "expected_calibration_error": ece.model_dump(mode="json"),
            "maximum_calibration_error": mce.model_dump(mode="json"),
        }
    )
    return V5CalibrationEvidence.model_validate(values)


def compute_v5_binary_metrics(
    *,
    labels: Sequence[int],
    actions: Sequence[SentinelAction],
    probabilities: Sequence[float] | None,
    support_ids: Sequence[str],
    probability_applicable: bool,
) -> Mapping[str, V5MetricEstimate]:
    """Compute exact aggregate classification and workload metrics."""
    if not labels or not (len(labels) == len(actions) == len(support_ids)):
        raise ValueError("binary metric inputs must align and be nonempty")
    if any(label not in {0, 1} for label in labels):
        raise ValueError("binary labels must be zero or one")
    detected = tuple(action in _DETECTION_ACTIONS for action in actions)
    fraud_count = sum(labels)
    legitimate_count = len(labels) - fraud_count
    true_positive = sum(
        bool(flag) and label == 1 for flag, label in zip(detected, labels, strict=True)
    )
    false_positive = sum(
        bool(flag) and label == 0 for flag, label in zip(detected, labels, strict=True)
    )
    false_negative = fraud_count - true_positive
    metrics: dict[str, V5MetricEstimate] = {
        "recall": _ratio_metric(
            name="recall",
            numerator=float(true_positive),
            denominator=float(fraud_count),
            support_ids=support_ids,
            formula="detected_fraud_rows/fraud_rows",
        ),
        "precision": _ratio_metric(
            name="precision",
            numerator=float(true_positive),
            denominator=float(true_positive + false_positive),
            support_ids=support_ids,
            formula="detected_fraud_rows/all_detected_rows",
        ),
        "f1": _ratio_metric(
            name="f1",
            numerator=float(2 * true_positive),
            denominator=float(2 * true_positive + false_positive + false_negative),
            support_ids=support_ids,
            formula="2*tp/(2*tp+fp+fn)",
        ),
        "false_decline_rate": _ratio_metric(
            name="false_decline_rate",
            numerator=float(
                sum(
                    action is SentinelAction.DECLINE_HOLD and label == 0
                    for action, label in zip(actions, labels, strict=True)
                )
            ),
            denominator=float(legitimate_count),
            support_ids=support_ids,
            formula="declined_legitimate_rows/legitimate_rows",
        ),
        "challenge_rate": _ratio_metric(
            name="challenge_rate",
            numerator=float(
                sum(
                    action is SentinelAction.CHALLENGE and label == 0
                    for action, label in zip(actions, labels, strict=True)
                )
            ),
            denominator=float(legitimate_count),
            support_ids=support_ids,
            formula="challenged_legitimate_rows/legitimate_rows",
        ),
        "review_rate": _ratio_metric(
            name="review_rate",
            numerator=float(
                sum(
                    action is SentinelAction.REVIEW_HOLD and label == 0
                    for action, label in zip(actions, labels, strict=True)
                )
            ),
            denominator=float(legitimate_count),
            support_ids=support_ids,
            formula="reviewed_legitimate_rows/legitimate_rows",
        ),
        "decline_rate": _ratio_metric(
            name="decline_rate",
            numerator=float(sum(action is SentinelAction.DECLINE_HOLD for action in actions)),
            denominator=float(len(labels)),
            support_ids=support_ids,
            formula="declined_rows/all_rows",
        ),
    }
    if not probability_applicable:
        for name in (
            "pr_auc",
            "roc_auc",
            "brier",
            "expected_calibration_error",
            "maximum_calibration_error",
        ):
            metrics[name] = _not_applicable_metric(
                name=name,
                support_ids=support_ids,
                formula="not applicable to deterministic rules",
            )
        return metrics
    if probabilities is None or len(probabilities) != len(labels):
        raise ValueError("probability metrics require aligned probabilities")
    probability_values = tuple(float(value) for value in probabilities)
    if any(not math.isfinite(value) or not 0 <= value <= 1 for value in probability_values):
        raise ValueError("probabilities must be finite values in [0, 1]")
    if fraud_count and legitimate_count:
        pr_auc = float(average_precision_score(labels, probability_values))
        roc_auc = float(roc_auc_score(labels, probability_values))
        metrics["pr_auc"] = _metric(
            name="pr_auc",
            applicability=V5MetricApplicability.DEFINED,
            value=pr_auc,
            numerator=pr_auc,
            denominator=1.0,
            support_ids=support_ids,
            formula="average_precision(labels,probabilities)",
        )
        metrics["roc_auc"] = _metric(
            name="roc_auc",
            applicability=V5MetricApplicability.DEFINED,
            value=roc_auc,
            numerator=roc_auc,
            denominator=1.0,
            support_ids=support_ids,
            formula="roc_auc(labels,probabilities)",
        )
    else:
        for name in ("pr_auc", "roc_auc"):
            metrics[name] = _metric(
                name=name,
                applicability=V5MetricApplicability.UNDEFINED,
                value=None,
                numerator=float(fraud_count),
                denominator=0.0,
                support_ids=support_ids,
                formula=f"{name} requires both classes",
            )
    squared_error = sum(
        (probability - label) ** 2
        for probability, label in zip(probability_values, labels, strict=True)
    )
    metrics["brier"] = _metric(
        name="brier",
        applicability=V5MetricApplicability.DEFINED,
        value=squared_error / len(labels),
        numerator=squared_error,
        denominator=float(len(labels)),
        support_ids=support_ids,
        formula="sum((probability-label)^2)/all_rows",
    )
    calibration = compute_v5_calibration(
        labels=labels,
        probabilities=probability_values,
        boundaries=tuple(index / 10 for index in range(11)),
        support_ids=support_ids,
        applicable=True,
    )
    metrics["expected_calibration_error"] = calibration.expected_calibration_error
    metrics["maximum_calibration_error"] = calibration.maximum_calibration_error
    return metrics


class V5PaymentEconomics(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    payment_id: str
    campaign_id: str
    family: str
    rail: str
    currency: str
    is_fraud: bool
    amount: Decimal
    authorized: bool
    moved: bool
    reversed_or_recovered: bool
    intervened_before_movement: bool
    captured: bool
    escaped: bool
    event_ids: tuple[str, ...]
    payment_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def payment_is_reconciled(self) -> Self:
        if self.captured and self.escaped:
            raise ValueError("payment value cannot be both captured and escaped")
        if self.is_fraud and not (self.captured or self.escaped):
            raise ValueError("malicious payment must reconcile to captured or escaped")
        if not self.is_fraud and (self.captured or self.escaped):
            raise ValueError("legitimate payment cannot claim malicious economics")
        if self.payment_sha256 != _digest(self.model_dump(mode="json", exclude={"payment_sha256"})):
            raise ValueError("payment economics digest mismatch")
        return self


class V5EconomicStratum(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    family: str | None = None
    rail: str | None = None
    currency: str
    payment_count: int = Field(ge=0)
    attempted_amount: Decimal
    malicious_amount: Decimal
    authorized_amount: Decimal
    settled_or_posted_amount: Decimal
    returned_refunded_recovered_amount: Decimal
    prevented_amount: Decimal
    captured_amount: Decimal
    escaped_amount: Decimal
    payment_ids: tuple[str, ...]
    stratum_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def stratum_is_bound(self) -> Self:
        if self.payment_count != len(self.payment_ids):
            raise ValueError("economic stratum payment count mismatch")
        if self.captured_amount + self.escaped_amount != self.malicious_amount:
            raise ValueError("economic stratum malicious value does not reconcile")
        if self.stratum_sha256 != _digest(self.model_dump(mode="json", exclude={"stratum_sha256"})):
            raise ValueError("economic stratum digest mismatch")
        return self


class V5EconomicEvidence(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    currencies: tuple[str, ...]
    payment_count: int
    ledger_debit_by_currency: tuple[tuple[str, Decimal], ...]
    ledger_credit_by_currency: tuple[tuple[str, Decimal], ...]
    ledger_conserved: bool
    payments: tuple[V5PaymentEconomics, ...]
    by_currency: tuple[V5EconomicStratum, ...]
    by_campaign: tuple[V5EconomicStratum, ...]
    by_family: tuple[V5EconomicStratum, ...]
    by_rail: tuple[V5EconomicStratum, ...]
    economics_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def totals_are_reconciled(self) -> Self:
        if self.payment_count != len(self.payments):
            raise ValueError("economic payment count mismatch")
        if self.currencies != tuple(sorted(set(self.currencies))):
            raise ValueError("economic currencies must be unique and canonical")
        if self.ledger_debit_by_currency != self.ledger_credit_by_currency:
            raise ValueError("economic ledger totals do not conserve")
        if not self.ledger_conserved:
            raise ValueError("economic ledger is not conserved")
        currency_payments = {item.currency: item.payment_count for item in self.by_currency}
        if currency_payments != {
            currency: sum(payment.currency == currency for payment in self.payments)
            for currency in self.currencies
        }:
            raise ValueError("currency economics do not cover aggregate payments")
        if self.economics_sha256 != _digest(
            self.model_dump(mode="json", exclude={"economics_sha256"})
        ):
            raise ValueError("economic evidence digest mismatch")
        return self


def _economic_stratum(
    *,
    payments: Sequence[V5PaymentEconomics],
    family: str | None = None,
    rail: str | None = None,
) -> V5EconomicStratum:
    selected = tuple(payments)
    if not selected:
        raise ValueError("economic stratum must contain payments")
    currencies = {item.currency for item in selected}
    if len(currencies) != 1:
        raise ValueError("economic stratum cannot combine currencies")
    values = {
        "family": family,
        "rail": rail,
        "currency": next(iter(currencies)),
        "payment_count": len(selected),
        "attempted_amount": sum((item.amount for item in selected), Decimal(0)),
        "malicious_amount": sum((item.amount for item in selected if item.is_fraud), Decimal(0)),
        "authorized_amount": sum((item.amount for item in selected if item.authorized), Decimal(0)),
        "settled_or_posted_amount": sum(
            (item.amount for item in selected if item.moved), Decimal(0)
        ),
        "returned_refunded_recovered_amount": sum(
            (item.amount for item in selected if item.reversed_or_recovered), Decimal(0)
        ),
        "prevented_amount": sum(
            (
                item.amount
                for item in selected
                if item.is_fraud and (not item.moved or item.intervened_before_movement)
            ),
            Decimal(0),
        ),
        "captured_amount": sum((item.amount for item in selected if item.captured), Decimal(0)),
        "escaped_amount": sum((item.amount for item in selected if item.escaped), Decimal(0)),
        "payment_ids": tuple(item.payment_id for item in selected),
    }
    digest_values = {
        key: str(value) if isinstance(value, Decimal) else value for key, value in values.items()
    }
    values["stratum_sha256"] = _digest(digest_values)
    return V5EconomicStratum.model_validate(values)


def reconcile_v5_economics(
    *,
    support: Sequence[V5ArmSupportRow],
    execution_artifacts: Sequence[V5ExecutionArtifact],
    actions: Sequence[SentinelAction],
    protocol: V5EconomicProtocol,
) -> V5EconomicEvidence:
    """Reconcile lifecycle, action, and double-entry facts once per payment."""
    if not support or len(support) != len(actions):
        raise ValueError("economic support and actions must align and be nonempty")
    if len({row.event_id for row in support}) != len(support):
        raise ValueError("economic support event IDs must be unique")
    action_by_event = {row.event_id: action for row, action in zip(support, actions, strict=True)}
    support_by_event = {row.event_id: row for row in support}
    artifact_events: set[str] = set()
    event_records: dict[str, tuple[PaymentEvent, str, str, str, bool, int]] = {}
    ledger_ids: set[str] = set()
    ledger_debit: dict[str, Decimal] = defaultdict(Decimal)
    ledger_credit: dict[str, Decimal] = defaultdict(Decimal)
    currencies: set[str] = set()
    for artifact in execution_artifacts:
        manifest = artifact.manifest()
        event_by_id = {
            record.event_id: PaymentEvent.model_validate_json(record.event_json)
            for record in manifest.event_records
        }
        for link in manifest.lineage:
            if link.event_id in artifact_events:
                raise ValueError("economic execution artifacts contain duplicate events")
            artifact_events.add(link.event_id)
            event = event_by_id[link.event_id]
            currencies.add(event.currency)
            event_records[link.event_id] = (
                event,
                link.payment_id,
                manifest.family,
                manifest.rail,
                link.is_fraud,
                link.lifecycle_position,
            )
        for posting in manifest.ledger_postings:
            if posting.entry_id in ledger_ids:
                raise ValueError("economic ledger entry IDs must be unique")
            ledger_ids.add(posting.entry_id)
            debit = sum((amount for _account, amount in posting.debit), Decimal(0))
            credit = sum((amount for _account, amount in posting.credit), Decimal(0))
            if debit != credit or posting.currency not in currencies:
                raise ValueError("economic ledger posting is not currency-conserving")
            ledger_debit[posting.currency] += debit
            ledger_credit[posting.currency] += credit
    if set(support_by_event) != artifact_events:
        raise ValueError("economic support must exactly equal execution lineage")
    by_payment: dict[str, list[tuple[PaymentEvent, str, str, str, bool, int]]] = defaultdict(list)
    for event_id, facts in event_records.items():
        support_row = support_by_event[event_id]
        event, payment_id, family, rail, is_fraud, position = facts
        if (
            support_row.payment_id != payment_id
            or support_row.family != family
            or support_row.rail != rail
            or bool(support_row.label) is not is_fraud
            or Decimal(str(support_row.amount)) != event.amount
            or support_row.currency != event.currency
        ):
            raise ValueError("economic support disagrees with canonical event facts")
        by_payment[payment_id].append(facts)
    payments: list[V5PaymentEconomics] = []
    movement_by_rail = {rail: set(kinds) for rail, kinds in protocol.rail_movement_events.items()}
    authorization_events = set(protocol.authorization_events)
    reversal_events = set(protocol.value_reversal_events)
    for payment_id, payment_facts in sorted(by_payment.items()):
        ordered = sorted(payment_facts, key=lambda item: item[5])
        if tuple(item[5] for item in ordered) != tuple(range(len(ordered))):
            raise ValueError("economic payment lifecycle positions are incomplete")
        events = tuple(item[0] for item in ordered)
        families = {item[2] for item in ordered}
        rails = {item[3] for item in ordered}
        labels = {item[4] for item in ordered}
        amounts = {item.amount for item in events}
        payment_currencies = {item.currency for item in events}
        campaigns = {item.campaign_id for item in events}
        if any(
            len(values) != 1
            for values in (families, rails, labels, amounts, payment_currencies, campaigns)
        ):
            raise ValueError("economic payment facts are inconsistent across lifecycle")
        family = next(iter(families))
        rail = next(iter(rails))
        is_fraud = next(iter(labels))
        amount = next(iter(amounts))
        movement_events = movement_by_rail[rail]
        movement_times = [
            event.decision_at
            for event in events
            if event.event_type.value in movement_events and event.decision_at is not None
        ]
        movement_time = min(movement_times) if movement_times else None
        intervention_times = [
            event.decision_at
            for event in events
            if action_by_event[event.event_id] in _DETECTION_ACTIONS
            and event.decision_at is not None
        ]
        first_intervention = min(intervention_times) if intervention_times else None
        intervened_before = first_intervention is not None and (
            movement_time is None or first_intervention <= movement_time
        )
        moved = movement_time is not None
        reversed_or_recovered = any(event.event_type.value in reversal_events for event in events)
        captured = bool(is_fraud and (not moved or reversed_or_recovered or intervened_before))
        escaped = bool(is_fraud and moved and not captured)
        values = {
            "payment_id": payment_id,
            "campaign_id": next(iter(campaigns)),
            "family": family,
            "rail": rail,
            "currency": next(iter(payment_currencies)),
            "is_fraud": is_fraud,
            "amount": amount,
            "authorized": any(event.event_type.value in authorization_events for event in events),
            "moved": moved,
            "reversed_or_recovered": reversed_or_recovered,
            "intervened_before_movement": intervened_before,
            "captured": captured,
            "escaped": escaped,
            "event_ids": tuple(event.event_id for event in events),
        }
        digest_values = {
            key: str(value) if isinstance(value, Decimal) else value
            for key, value in values.items()
        }
        values["payment_sha256"] = _digest(digest_values)
        payments.append(V5PaymentEconomics.model_validate(values))
    payment_tuple = tuple(payments)
    by_currency = tuple(
        _economic_stratum(
            payments=tuple(item for item in payment_tuple if item.currency == currency)
        )
        for currency in sorted(currencies)
    )
    by_campaign = tuple(
        _economic_stratum(
            payments=tuple(
                item
                for item in payment_tuple
                if item.campaign_id == campaign and item.currency == currency
            )
        )
        for campaign, currency in sorted(
            {(item.campaign_id, item.currency) for item in payment_tuple}
        )
    )
    by_family = tuple(
        _economic_stratum(
            payments=tuple(
                item
                for item in payment_tuple
                if item.family == family and item.currency == currency
            ),
            family=family,
        )
        for family, currency in sorted(
            {(item.family, item.currency) for item in payment_tuple if item.family != "legitimate"}
        )
    )
    by_rail = tuple(
        _economic_stratum(
            payments=tuple(
                item for item in payment_tuple if item.rail == rail and item.currency == currency
            ),
            rail=rail,
        )
        for rail, currency in sorted({(item.rail, item.currency) for item in payment_tuple})
    )
    values = {
        "currencies": tuple(sorted(currencies)),
        "payment_count": len(payment_tuple),
        "ledger_debit_by_currency": tuple(sorted(ledger_debit.items())),
        "ledger_credit_by_currency": tuple(sorted(ledger_credit.items())),
        "ledger_conserved": ledger_debit == ledger_credit,
        "payments": payment_tuple,
        "by_currency": by_currency,
        "by_campaign": by_campaign,
        "by_family": by_family,
        "by_rail": by_rail,
    }
    digest_values = {
        key: (
            [
                item.model_dump(mode="json")
                if isinstance(item, BaseModel)
                else [item[0], str(item[1])]
                if isinstance(item, tuple) and len(item) == 2 and isinstance(item[1], Decimal)
                else item
                for item in value
            ]
            if isinstance(value, tuple)
            else value
        )
        for key, value in values.items()
    }
    values["economics_sha256"] = _digest(digest_values)
    return V5EconomicEvidence.model_validate(values)


class V5CampaignAlert(BaseModel):
    """One fraud campaign's exact first-alert evidence."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    campaign_id: str
    family: str
    event_ids: tuple[str, ...]
    first_decision_at: datetime
    first_alert_event_id: str | None
    first_alert_at: datetime | None
    time_to_first_alert_seconds: float | None = Field(default=None, ge=0.0)
    detected: bool
    alert_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def alert_is_consistent(self) -> Self:
        if self.detected != (self.first_alert_event_id is not None):
            raise ValueError("campaign detection and first-alert evidence disagree")
        if self.detected:
            if self.first_alert_at is None or self.time_to_first_alert_seconds is None:
                raise ValueError("detected campaign requires alert time and latency")
            expected_seconds = (self.first_alert_at - self.first_decision_at).total_seconds()
            if not math.isclose(
                self.time_to_first_alert_seconds,
                expected_seconds,
                rel_tol=0.0,
                abs_tol=1e-9,
            ):
                raise ValueError("campaign time-to-first-alert is inconsistent")
        elif self.first_alert_at is not None or self.time_to_first_alert_seconds is not None:
            raise ValueError("undetected campaign cannot claim alert evidence")
        if self.alert_sha256 != _digest(self.model_dump(mode="json", exclude={"alert_sha256"})):
            raise ValueError("campaign alert digest mismatch")
        return self


class V5FamilyMetrics(BaseModel):
    """Complete metrics for one preregistered fraud family."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    family: str
    support_count: int = Field(gt=0)
    campaign_count: int = Field(gt=0)
    event_ids: tuple[str, ...]
    campaign_ids: tuple[str, ...]
    recall: V5MetricEstimate
    precision: V5MetricEstimate
    campaign_detection_rate: V5MetricEstimate
    time_to_first_alert: V5MetricEstimate
    action_distribution: tuple[tuple[str, int], ...]
    economic_strata: tuple[V5EconomicStratum, ...]
    campaign_alerts: tuple[V5CampaignAlert, ...]
    family_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def family_is_complete(self) -> Self:
        if self.support_count != len(self.event_ids):
            raise ValueError("family support count mismatch")
        if self.campaign_count != len(self.campaign_ids):
            raise ValueError("family campaign count mismatch")
        if self.campaign_ids != tuple(sorted(set(self.campaign_ids))):
            raise ValueError("family campaign IDs must be unique and canonical")
        if tuple(alert.campaign_id for alert in self.campaign_alerts) != self.campaign_ids:
            raise ValueError("family campaign alerts do not cover campaign IDs")
        if sum(count for _action, count in self.action_distribution) != self.support_count:
            raise ValueError("family action distribution does not cover support")
        if self.family_sha256 != _digest(self.model_dump(mode="json", exclude={"family_sha256"})):
            raise ValueError("family metric digest mismatch")
        return self


class V5BootstrapSample(BaseModel):
    """One campaign-stratified resample, retaining the complete draw."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    replicate: int = Field(ge=0, lt=10_000)
    campaign_ids: tuple[str, ...] = Field(max_length=_MAX_ROWS)
    event_ids: tuple[str, ...] = Field(max_length=_MAX_ROWS)
    sample_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def sample_is_bound(self) -> Self:
        if not self.campaign_ids or not self.event_ids:
            raise ValueError("bootstrap sample cannot be empty")
        if self.sample_sha256 != _digest(self.model_dump(mode="json", exclude={"sample_sha256"})):
            raise ValueError("bootstrap sample digest mismatch")
        return self


class V5BootstrapInterval(BaseModel):
    """One percentile interval with explicit undefined semantics."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    metric: str
    family: str | None = None
    applicability: V5MetricApplicability
    point: float | None
    lower: float | None
    upper: float | None
    defined_replicates: int = Field(ge=0, le=10_000)
    sample_values_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    interval_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def interval_is_consistent(self) -> Self:
        values = (self.point, self.lower, self.upper)
        if self.applicability is V5MetricApplicability.DEFINED:
            if any(value is None for value in values) or self.defined_replicates == 0:
                raise ValueError("defined interval requires point, bounds, and samples")
            assert self.lower is not None and self.point is not None and self.upper is not None
            if self.lower > self.upper:
                raise ValueError("bootstrap lower bound exceeds upper bound")
        elif any(value is not None for value in values):
            raise ValueError("non-defined interval cannot claim numeric bounds")
        if self.interval_sha256 != _digest(
            self.model_dump(mode="json", exclude={"interval_sha256"})
        ):
            raise ValueError("bootstrap interval digest mismatch")
        return self


class V5BootstrapEvidence(BaseModel):
    """Frozen campaign-level bootstrap design and complete deterministic draws."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    seed: int
    replicates: int = Field(gt=0, le=10_000)
    confidence_level: float
    interval_method: str
    resampling_unit: str
    stratification: str
    strata: tuple[tuple[str, tuple[str, ...]], ...]
    samples: tuple[V5BootstrapSample, ...] = Field(max_length=10_000)
    intervals: tuple[V5BootstrapInterval, ...]
    bootstrap_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def bootstrap_is_complete(self) -> Self:
        if self.replicates != len(self.samples):
            raise ValueError("bootstrap replicate count disagrees with samples")
        if tuple(sample.replicate for sample in self.samples) != tuple(range(self.replicates)):
            raise ValueError("bootstrap sample indices are incomplete")
        if self.bootstrap_sha256 != _digest(
            self.model_dump(mode="json", exclude={"bootstrap_sha256"})
        ):
            raise ValueError("bootstrap evidence digest mismatch")
        return self


class V5CompleteArmMetrics(BaseModel):
    """Complete independently replayable metric evidence for one arm."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    arm: V5Arm
    arm_result_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    support_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    aggregate: Mapping[str, V5MetricEstimate]
    calibration: V5CalibrationEvidence
    economics: V5EconomicEvidence
    by_family: tuple[V5FamilyMetrics, ...]
    bootstrap: V5BootstrapEvidence
    complete_metrics_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @field_validator("aggregate", mode="after")
    @classmethod
    def aggregate_is_immutable(
        cls, value: Mapping[str, V5MetricEstimate]
    ) -> Mapping[str, V5MetricEstimate]:
        return MappingProxyType(dict(sorted(value.items())))

    @field_serializer("aggregate")
    def serialize_aggregate(self, value: Mapping[str, V5MetricEstimate]) -> dict[str, object]:
        return {name: metric.model_dump(mode="json") for name, metric in value.items()}

    @model_validator(mode="after")
    def complete_evidence_is_bound(self) -> Self:
        required = {
            "recall",
            "precision",
            "f1",
            "pr_auc",
            "roc_auc",
            "brier",
            "false_decline_rate",
            "challenge_rate",
            "review_rate",
            "decline_rate",
            "captured_value_fraction",
            "escaped_value_fraction",
            "p50_latency_ms",
            "p95_latency_ms",
            "p99_latency_ms",
        }
        if not required.issubset(self.aggregate):
            raise ValueError("complete aggregate metric set is incomplete")
        if self.complete_metrics_sha256 != _digest(
            self.model_dump(mode="json", exclude={"complete_metrics_sha256"})
        ):
            raise ValueError("complete metric digest mismatch")
        return self


def _latency_metric(
    *,
    name: str,
    percentile: float,
    rows: Sequence[V5ArmRowEvidence],
    support_ids: Sequence[str],
) -> V5MetricEstimate:
    values = np.asarray([float(row.latency_ms) for row in rows], dtype=float)
    value = float(np.percentile(values, percentile))
    return _metric(
        name=name,
        applicability=V5MetricApplicability.DEFINED,
        value=value,
        numerator=value,
        denominator=1.0,
        support_ids=support_ids,
        formula=f"percentile_{percentile:g}(per_row_scoring_latency_ms)",
    )


def _economic_fraction_metrics(
    economics: V5EconomicEvidence, support_ids: Sequence[str]
) -> tuple[V5MetricEstimate, V5MetricEstimate]:
    malicious = tuple(item for item in economics.by_currency if item.malicious_amount > 0)
    if not malicious:
        return (
            _metric(
                name="captured_value_fraction",
                applicability=V5MetricApplicability.UNDEFINED,
                value=None,
                numerator=0.0,
                denominator=0.0,
                support_ids=support_ids,
                formula="mean(currency_captured_malicious/currency_malicious)",
            ),
            _metric(
                name="escaped_value_fraction",
                applicability=V5MetricApplicability.UNDEFINED,
                value=None,
                numerator=0.0,
                denominator=0.0,
                support_ids=support_ids,
                formula="mean(currency_escaped_malicious/currency_malicious)",
            ),
        )
    captured_fractions = tuple(
        float(stratum.captured_amount / stratum.malicious_amount) for stratum in malicious
    )
    escaped_fractions = tuple(
        float(stratum.escaped_amount / stratum.malicious_amount) for stratum in malicious
    )
    return (
        _ratio_metric(
            name="captured_value_fraction",
            numerator=sum(captured_fractions),
            denominator=float(len(captured_fractions)),
            support_ids=support_ids,
            formula="mean(currency_captured_malicious/currency_malicious)",
        ),
        _ratio_metric(
            name="escaped_value_fraction",
            numerator=sum(escaped_fractions),
            denominator=float(len(escaped_fractions)),
            support_ids=support_ids,
            formula="mean(currency_escaped_malicious/currency_malicious)",
        ),
    )


def _event_times(
    artifacts: Sequence[V5ExecutionArtifact],
) -> dict[str, datetime]:
    times: dict[str, datetime] = {}
    for artifact in artifacts:
        manifest = artifact.manifest()
        for record in manifest.event_records:
            event = PaymentEvent.model_validate_json(record.event_json)
            if event.event_id in times or event.decision_at is None:
                raise ValueError("metric event time evidence is missing or duplicated")
            times[event.event_id] = event.decision_at
    return times


def _campaign_alerts(
    *, result: V5EvaluationResult, family: str, event_times: Mapping[str, datetime]
) -> tuple[V5CampaignAlert, ...]:
    by_campaign: dict[str, list[V5ArmRowEvidence]] = defaultdict(list)
    for row in result.row_evidence:
        if row.support.family == family and row.support.label == 1:
            by_campaign[row.support.campaign_id].append(row)
    alerts: list[V5CampaignAlert] = []
    for campaign_id, rows in sorted(by_campaign.items()):
        ordered = sorted(
            rows, key=lambda row: (event_times[row.support.event_id], row.support.event_id)
        )
        campaign_rows = tuple(
            row for row in result.row_evidence if row.support.campaign_id == campaign_id
        )
        first_decision = min(event_times[row.support.event_id] for row in campaign_rows)
        detected = [row for row in ordered if row.action in _DETECTION_ACTIONS]
        first = detected[0] if detected else None
        first_alert_at = event_times[first.support.event_id] if first is not None else None
        values = {
            "campaign_id": campaign_id,
            "family": family,
            "event_ids": tuple(row.support.event_id for row in ordered),
            "first_decision_at": first_decision,
            "first_alert_event_id": first.support.event_id if first is not None else None,
            "first_alert_at": first_alert_at,
            "time_to_first_alert_seconds": (
                (first_alert_at - first_decision).total_seconds()
                if first_alert_at is not None
                else None
            ),
            "detected": first is not None,
        }
        values["alert_sha256"] = _digest(values)
        alerts.append(V5CampaignAlert.model_validate(values))
    return tuple(alerts)


def _build_family_metrics(
    *,
    result: V5EvaluationResult,
    economics: V5EconomicEvidence,
    event_times: Mapping[str, datetime],
) -> tuple[V5FamilyMetrics, ...]:
    all_detected_legitimate = sum(
        row.support.label == 0 and row.action in _DETECTION_ACTIONS for row in result.row_evidence
    )
    families = sorted({row.support.family for row in result.row_evidence if row.support.label == 1})
    output: list[V5FamilyMetrics] = []
    for family in families:
        rows = tuple(
            row
            for row in result.row_evidence
            if row.support.family == family and row.support.label == 1
        )
        ids = tuple(row.support.event_id for row in rows)
        campaigns = tuple(sorted({row.support.campaign_id for row in rows}))
        detected_rows = sum(row.action in _DETECTION_ACTIONS for row in rows)
        alerts = _campaign_alerts(result=result, family=family, event_times=event_times)
        detected_campaigns = sum(alert.detected for alert in alerts)
        alert_seconds = tuple(
            alert.time_to_first_alert_seconds
            for alert in alerts
            if alert.time_to_first_alert_seconds is not None
        )
        time_metric = (
            _metric(
                name="time_to_first_alert_seconds",
                applicability=V5MetricApplicability.DEFINED,
                value=float(sum(alert_seconds) / len(alert_seconds)),
                numerator=float(sum(alert_seconds)),
                denominator=float(len(alert_seconds)),
                support_ids=ids,
                formula="sum(campaign_time_to_first_alert_seconds)/detected_campaigns",
            )
            if alert_seconds
            else _metric(
                name="time_to_first_alert_seconds",
                applicability=V5MetricApplicability.UNDEFINED,
                value=None,
                numerator=0.0,
                denominator=0.0,
                support_ids=ids,
                formula="time-to-first-alert requires a detected campaign",
            )
        )
        actions = tuple(
            (action.value, sum(row.action is action for row in rows)) for action in SentinelAction
        )
        values = {
            "family": family,
            "support_count": len(rows),
            "campaign_count": len(campaigns),
            "event_ids": ids,
            "campaign_ids": campaigns,
            "recall": _ratio_metric(
                name=f"family_recall:{family}",
                numerator=float(detected_rows),
                denominator=float(len(rows)),
                support_ids=ids,
                formula="detected_family_rows/family_rows",
            ),
            "precision": _ratio_metric(
                name=f"family_precision:{family}",
                numerator=float(detected_rows),
                denominator=float(detected_rows + all_detected_legitimate),
                support_ids=ids,
                formula="detected_family_rows/(detected_family_rows+detected_legitimate_rows)",
            ),
            "campaign_detection_rate": _ratio_metric(
                name=f"campaign_detection_rate:{family}",
                numerator=float(detected_campaigns),
                denominator=float(len(campaigns)),
                support_ids=ids,
                formula="detected_fraud_campaigns/fraud_campaigns",
            ),
            "time_to_first_alert": time_metric,
            "action_distribution": actions,
            "economic_strata": tuple(item for item in economics.by_family if item.family == family),
            "campaign_alerts": alerts,
        }
        values["family_sha256"] = _digest(
            {
                key: (
                    value.model_dump(mode="json")
                    if isinstance(value, BaseModel)
                    else [item.model_dump(mode="json") for item in value]
                    if isinstance(value, tuple) and value and isinstance(value[0], BaseModel)
                    else value
                )
                for key, value in values.items()
            }
        )
        output.append(V5FamilyMetrics.model_validate(values))
    return tuple(output)


def _percentile_interval(values: Sequence[float], confidence: float) -> tuple[float, float]:
    tail = (1.0 - confidence) / 2.0
    return (
        float(np.percentile(np.asarray(values, dtype=float), tail * 100.0)),
        float(np.percentile(np.asarray(values, dtype=float), (1.0 - tail) * 100.0)),
    )


def _bootstrap_metric_values(
    *,
    result: V5EvaluationResult,
    economics: V5EconomicEvidence,
    protocol: V5BootstrapProtocol,
    probability_applicable: bool,
    point_metrics: Mapping[str, V5MetricEstimate],
    family_metrics: Sequence[V5FamilyMetrics],
) -> V5BootstrapEvidence:
    rows_by_campaign: dict[str, tuple[int, ...]] = {}
    campaign_family: dict[str, str] = {}
    for campaign_id in sorted({row.support.campaign_id for row in result.row_evidence}):
        indices = tuple(
            index
            for index, row in enumerate(result.row_evidence)
            if row.support.campaign_id == campaign_id
        )
        fraud_families = {
            result.row_evidence[index].support.family
            for index in indices
            if result.row_evidence[index].support.label == 1
        }
        if len(fraud_families) > 1:
            raise ValueError("bootstrap campaign spans multiple fraud families")
        rows_by_campaign[campaign_id] = indices
        campaign_family[campaign_id] = next(iter(fraud_families), "legitimate")
    strata_map: dict[str, tuple[str, ...]] = {}
    for stratum in ("legitimate", *sorted(set(campaign_family.values()) - {"legitimate"})):
        campaigns = tuple(
            campaign for campaign, family in campaign_family.items() if family == stratum
        )
        if not campaigns:
            raise ValueError(f"bootstrap stratum {stratum} is empty")
        strata_map[stratum] = campaigns
    rng = np.random.Generator(np.random.PCG64(protocol.seed))
    samples: list[V5BootstrapSample] = []
    value_series: dict[tuple[str, str | None], list[float]] = defaultdict(list)
    payments_by_campaign: dict[str, tuple[V5PaymentEconomics, ...]] = {
        campaign: tuple(item for item in economics.payments if item.campaign_id == campaign)
        for campaign in rows_by_campaign
    }
    for replicate in range(protocol.replicates):
        drawn: list[str] = []
        for campaigns in strata_map.values():
            positions = rng.integers(0, len(campaigns), size=len(campaigns))
            drawn.extend(campaigns[int(position)] for position in positions)
        sampled_indices = tuple(index for campaign in drawn for index in rows_by_campaign[campaign])
        event_ids = tuple(result.row_evidence[index].support.event_id for index in sampled_indices)
        values = {
            "replicate": replicate,
            "campaign_ids": tuple(drawn),
            "event_ids": event_ids,
        }
        values["sample_sha256"] = _digest(values)
        samples.append(V5BootstrapSample.model_validate(values))
        labels = tuple(result.row_evidence[index].support.label for index in sampled_indices)
        actions = tuple(result.row_evidence[index].action for index in sampled_indices)
        detected = tuple(action in _DETECTION_ACTIONS for action in actions)
        fraud_count = sum(labels)
        legitimate_count = len(labels) - fraud_count
        if fraud_count:
            value_series[("recall", None)].append(
                sum(flag and label == 1 for flag, label in zip(detected, labels, strict=True))
                / fraud_count
            )
        if legitimate_count:
            value_series[("false_decline_rate", None)].append(
                sum(
                    action is SentinelAction.DECLINE_HOLD and label == 0
                    for action, label in zip(actions, labels, strict=True)
                )
                / legitimate_count
            )
            value_series[("challenge_rate", None)].append(
                sum(
                    action is SentinelAction.CHALLENGE and label == 0
                    for action, label in zip(actions, labels, strict=True)
                )
                / legitimate_count
            )
            value_series[("review_rate", None)].append(
                sum(
                    action is SentinelAction.REVIEW_HOLD and label == 0
                    for action, label in zip(actions, labels, strict=True)
                )
                / legitimate_count
            )
        fraud_campaign_draws = [
            campaign for campaign in drawn if campaign_family[campaign] != "legitimate"
        ]
        if fraud_campaign_draws:
            value_series[("campaign_detection_rate", None)].append(
                sum(
                    any(
                        result.row_evidence[index].support.label == 1
                        and result.row_evidence[index].action in _DETECTION_ACTIONS
                        for index in rows_by_campaign[campaign]
                    )
                    for campaign in fraud_campaign_draws
                )
                / len(fraud_campaign_draws)
            )
        selected_payments = tuple(
            payment
            for campaign in fraud_campaign_draws
            for payment in payments_by_campaign[campaign]
            if payment.is_fraud
        )
        currency_fractions: list[float] = []
        for currency in sorted({payment.currency for payment in selected_payments}):
            currency_payments = tuple(
                payment for payment in selected_payments if payment.currency == currency
            )
            malicious = sum((payment.amount for payment in currency_payments), Decimal(0))
            captured = sum(
                (payment.amount for payment in currency_payments if payment.captured),
                Decimal(0),
            )
            if malicious > 0:
                currency_fractions.append(float(captured / malicious))
        if currency_fractions:
            value_series[("captured_value_fraction", None)].append(
                sum(currency_fractions) / len(currency_fractions)
            )
        if probability_applicable:
            probabilities = tuple(
                result.row_evidence[index].probability for index in sampled_indices
            )
            total = len(labels)
            ece = 0.0
            for bin_index in range(10):
                lower = bin_index / 10
                upper = (bin_index + 1) / 10
                bin_positions = tuple(
                    index
                    for index, probability in enumerate(probabilities)
                    if probability >= lower
                    and (probability <= upper if bin_index == 9 else probability < upper)
                )
                if bin_positions:
                    mean_probability = sum(probabilities[index] for index in bin_positions) / len(
                        bin_positions
                    )
                    empirical = sum(labels[index] for index in bin_positions) / len(bin_positions)
                    ece += len(bin_positions) / total * abs(mean_probability - empirical)
            value_series[("expected_calibration_error", None)].append(ece)
        for family in family_metrics:
            draws = [campaign for campaign in drawn if campaign_family[campaign] == family.family]
            if not draws:
                continue
            family_indices = tuple(
                index for campaign in draws for index in rows_by_campaign[campaign]
            )
            fraud_indices = tuple(
                index for index in family_indices if result.row_evidence[index].support.label == 1
            )
            if fraud_indices:
                value_series[("recall", family.family)].append(
                    sum(
                        result.row_evidence[index].action in _DETECTION_ACTIONS
                        for index in fraud_indices
                    )
                    / len(fraud_indices)
                )
            value_series[("campaign_detection_rate", family.family)].append(
                sum(
                    any(
                        result.row_evidence[index].support.label == 1
                        and result.row_evidence[index].action in _DETECTION_ACTIONS
                        for index in rows_by_campaign[campaign]
                    )
                    for campaign in draws
                )
                / len(draws)
            )
    points: dict[tuple[str, str | None], V5MetricEstimate] = {
        (name, None): metric for name, metric in point_metrics.items()
    }
    for family in family_metrics:
        points[("recall", family.family)] = family.recall
        points[("campaign_detection_rate", family.family)] = family.campaign_detection_rate
    interval_keys = [(metric, None) for metric in protocol.metrics] + [
        (metric, family.family)
        for family in family_metrics
        for metric in ("recall", "campaign_detection_rate")
    ]
    intervals: list[V5BootstrapInterval] = []
    for metric_name, family_name in interval_keys:
        series = value_series.get((metric_name, family_name), [])
        point = points.get((metric_name, family_name))
        numeric_point: float | None
        interval_lower: float | None
        interval_upper: float | None
        if point is None or point.applicability is not V5MetricApplicability.DEFINED or not series:
            applicability = (
                point.applicability if point is not None else V5MetricApplicability.UNAVAILABLE
            )
            numeric_point = interval_lower = interval_upper = None
        else:
            applicability = V5MetricApplicability.DEFINED
            numeric_point = point.value
            interval_lower, interval_upper = _percentile_interval(series, protocol.confidence_level)
        values = {
            "metric": metric_name,
            "family": family_name,
            "applicability": applicability.value,
            "point": numeric_point,
            "lower": interval_lower,
            "upper": interval_upper,
            "defined_replicates": len(series),
            "sample_values_sha256": _digest(series),
        }
        values["interval_sha256"] = _digest(values)
        intervals.append(V5BootstrapInterval.model_validate(values))
    values = {
        "seed": protocol.seed,
        "replicates": protocol.replicates,
        "confidence_level": protocol.confidence_level,
        "interval_method": protocol.interval_method,
        "resampling_unit": "campaign",
        "stratification": protocol.stratification,
        "strata": tuple(sorted(strata_map.items())),
        "samples": tuple(samples),
        "intervals": tuple(intervals),
    }
    values["bootstrap_sha256"] = _digest(
        {
            **values,
            "samples": [sample.model_dump(mode="json") for sample in samples],
            "intervals": [interval.model_dump(mode="json") for interval in intervals],
        }
    )
    return V5BootstrapEvidence.model_validate(values)


def evaluate_v5_complete_result(
    *, result: V5EvaluationResult, protocol: V5EvidenceProtocol
) -> V5CompleteArmMetrics:
    """Compute the frozen complete metric contract from retained arm evidence."""
    if not result.row_evidence or not result.execution_artifacts or result.arm_spec is None:
        raise ValueError("complete evaluation requires retained arm execution evidence")
    arm = V5Arm(result.arm)
    support_ids = tuple(row.support.event_id for row in result.row_evidence)
    labels = tuple(row.support.label for row in result.row_evidence)
    actions = tuple(row.action for row in result.row_evidence)
    probability_applicable = arm is not V5Arm.RULES_ONLY
    probabilities = (
        tuple(row.probability for row in result.row_evidence) if probability_applicable else None
    )
    aggregate = dict(
        compute_v5_binary_metrics(
            labels=labels,
            actions=actions,
            probabilities=probabilities,
            support_ids=support_ids,
            probability_applicable=probability_applicable,
        )
    )
    for percentile in (50, 95, 99):
        aggregate[f"p{percentile}_latency_ms"] = _latency_metric(
            name=f"p{percentile}_latency_ms",
            percentile=float(percentile),
            rows=result.row_evidence,
            support_ids=support_ids,
        )
    calibration = compute_v5_calibration(
        labels=labels,
        probabilities=probabilities,
        boundaries=protocol.calibration.bin_boundaries,
        support_ids=support_ids,
        applicable=probability_applicable,
    )
    economics = reconcile_v5_economics(
        support=tuple(row.support for row in result.row_evidence),
        execution_artifacts=result.execution_artifacts,
        actions=actions,
        protocol=protocol.economics,
    )
    captured, escaped = _economic_fraction_metrics(economics, support_ids)
    aggregate["captured_value_fraction"] = captured
    aggregate["escaped_value_fraction"] = escaped
    for stratum in economics.by_currency:
        aggregate[f"captured_value:{stratum.currency}"] = _metric(
            name=f"captured_value:{stratum.currency}",
            applicability=V5MetricApplicability.DEFINED,
            value=float(stratum.captured_amount),
            numerator=float(stratum.captured_amount),
            denominator=1.0,
            support_ids=support_ids,
            formula=f"ledger_reconciled_captured_value_{stratum.currency}",
        )
        aggregate[f"escaped_value:{stratum.currency}"] = _metric(
            name=f"escaped_value:{stratum.currency}",
            applicability=V5MetricApplicability.DEFINED,
            value=float(stratum.escaped_amount),
            numerator=float(stratum.escaped_amount),
            denominator=1.0,
            support_ids=support_ids,
            formula=f"ledger_reconciled_escaped_value_{stratum.currency}",
        )
    event_times = _event_times(result.execution_artifacts)
    by_family = _build_family_metrics(
        result=result,
        economics=economics,
        event_times=event_times,
    )
    fraud_campaigns = {
        row.support.campaign_id for row in result.row_evidence if row.support.label == 1
    }
    detected_campaigns = {
        row.support.campaign_id
        for row in result.row_evidence
        if row.support.label == 1 and row.action in _DETECTION_ACTIONS
    }
    aggregate["campaign_detection_rate"] = _ratio_metric(
        name="campaign_detection_rate",
        numerator=float(len(detected_campaigns)),
        denominator=float(len(fraud_campaigns)),
        support_ids=support_ids,
        formula="detected_fraud_campaigns/fraud_campaigns",
    )
    bootstrap = _bootstrap_metric_values(
        result=result,
        economics=economics,
        protocol=protocol.bootstrap,
        probability_applicable=probability_applicable,
        point_metrics=aggregate,
        family_metrics=by_family,
    )
    values = {
        "arm": arm.value,
        "arm_result_sha256": result.result_sha256,
        "support_sha256": result.support_sha256,
        "aggregate": aggregate,
        "calibration": calibration,
        "economics": economics,
        "by_family": by_family,
        "bootstrap": bootstrap,
    }
    values["complete_metrics_sha256"] = _digest(
        {
            **values,
            "aggregate": {
                name: metric.model_dump(mode="json") for name, metric in aggregate.items()
            },
            "calibration": calibration.model_dump(mode="json"),
            "economics": economics.model_dump(mode="json"),
            "by_family": [family.model_dump(mode="json") for family in by_family],
            "bootstrap": bootstrap.model_dump(mode="json"),
        }
    )
    return V5CompleteArmMetrics.model_validate(values)


__all__ = [
    "V5BootstrapEvidence",
    "V5BootstrapInterval",
    "V5BootstrapSample",
    "V5CampaignAlert",
    "V5CalibrationBin",
    "V5CalibrationEvidence",
    "V5CompleteArmMetrics",
    "V5EconomicEvidence",
    "V5EconomicStratum",
    "V5FamilyMetrics",
    "V5MetricEstimate",
    "V5PaymentEconomics",
    "compute_v5_binary_metrics",
    "compute_v5_calibration",
    "evaluate_v5_complete_result",
    "reconcile_v5_economics",
]
