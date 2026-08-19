"""Closed hard gates and truthful champion/challenger selection."""

from __future__ import annotations

import hashlib
import math
from decimal import ROUND_HALF_EVEN, Context, Decimal, localcontext
from enum import StrEnum
from typing import Any, Literal, cast

from pydantic import Field, ValidationError, field_validator, model_validator

from apar.contracts._validation import ExternalContract
from apar.contracts.events import Rail
from apar.evaluation.contracts import Family
from apar.evaluation.regimes import RegimeKind
from apar.evaluation.splits import EntityCohort
from apar.runs.wire import WireContractError, canonical_json_bytes, strict_json_loads

_MONEY_CONTEXT = Context(prec=34, rounding=ROUND_HALF_EVEN)
_CENT = Decimal("0.01")
_MAX_RESULTS = 10_000
_SHA256_LENGTH = 64
_FAMILIES: tuple[Family, ...] = (
    "agentic_intent_abuse",
    "app_scam_mule",
    "card_testing_cnp",
    "synthetic_merchant_refund",
)


class GateContractError(ValueError):
    """Promotion evidence violates the closed competition contract."""


class DefenseArm(StrEnum):
    """The three matched champion/challenger defense arms."""

    RULES_ONLY = "rules_only"
    GBDT_ONLY = "gbdt_only"
    LAYERED_HYBRID = "layered_hybrid"


class EvaluationKind(StrEnum):
    """Closed evaluator-only robustness views."""

    CHRONOLOGICAL = "chronological"
    COLD_ENTITY = "cold_entity"
    HELD_FAMILY = "held_family"
    REGIME = "regime"
    HIDDEN = "hidden"


class EvaluationDescriptor(ExternalContract):
    """One exact evaluation view without labels or restricted references."""

    schema_version: Literal["1.0.0"] = "1.0.0"
    kind: EvaluationKind
    value: str

    @field_validator("value")
    @classmethod
    def value_is_bounded_text(cls, value: str) -> str:
        if type(value) is not str or not value or len(value) > 128:
            raise ValueError("evaluation value must be bounded nonempty text")
        return value

    @model_validator(mode="after")
    def value_matches_closed_kind(self) -> EvaluationDescriptor:
        allowed: dict[EvaluationKind, frozenset[str]] = {
            EvaluationKind.CHRONOLOGICAL: frozenset({"development"}),
            EvaluationKind.HIDDEN: frozenset({"hidden"}),
            EvaluationKind.HELD_FAMILY: frozenset(_FAMILIES),
            EvaluationKind.REGIME: frozenset(item.value for item in RegimeKind),
            EvaluationKind.COLD_ENTITY: frozenset(item.value for item in EntityCohort),
        }
        if self.value not in allowed[self.kind]:
            raise ValueError("evaluation value is outside its closed vocabulary")
        return self


class AssuranceEvidence(ExternalContract):
    """Binary non-averageable assurance evidence for one replay."""

    schema_version: Literal["1.0.0"] = "1.0.0"
    leakage_passed: bool
    parity_passed: bool
    artifact_signature_valid: bool
    rollback_available: bool
    hidden_access_clean: bool
    campaign_family_ownership_valid: bool

    @field_validator(
        "leakage_passed",
        "parity_passed",
        "artifact_signature_valid",
        "rollback_available",
        "hidden_access_clean",
        "campaign_family_ownership_valid",
        mode="before",
    )
    @classmethod
    def flags_are_exact_bools(cls, value: object) -> object:
        if type(value) is not bool:
            raise ValueError("assurance flags must be exact bools")
        return value

    @classmethod
    def passing(cls) -> AssuranceEvidence:
        """Return explicit all-pass evidence; callers must never infer it."""
        return cls(
            leakage_passed=True,
            parity_passed=True,
            artifact_signature_valid=True,
            rollback_available=True,
            hidden_access_clean=True,
            campaign_family_ownership_valid=True,
        )


class SlicePerformance(ExternalContract):
    """Aggregate-only recall for a named evaluator slice."""

    kind: Literal["family", "rail", "regime", "entity_cohort"]
    value: str
    recall: float | None

    @field_validator("value")
    @classmethod
    def value_is_bounded_text(cls, value: str) -> str:
        if type(value) is not str or not value or len(value) > 128:
            raise ValueError("slice value must be bounded nonempty text")
        return value

    @field_validator("recall", mode="before")
    @classmethod
    def recall_is_exact_finite_rate(cls, value: object) -> object:
        if value is not None and (
            type(value) is not float or not math.isfinite(value) or not 0.0 <= value <= 1.0
        ):
            raise ValueError("slice recall must be None or an exact finite rate")
        return value


class PromotionMetrics(ExternalContract):
    """Aggregate-only metrics needed by promotion, without evaluator truth."""

    schema_version: Literal["1.0.0"] = "1.0.0"
    row_count: int = Field(ge=1, le=1_000_000)
    recall: float | None
    ece: float | None
    p95_latency_ms: float | None
    preventable_settled_value: Decimal
    value_escaped: Decimal
    review_case_count: int = Field(ge=0)
    challenge_rate: float
    false_decline_rate: float
    review_case_rate: float
    slice_performance: tuple[SlicePerformance, ...]

    @field_validator("row_count", "review_case_count", mode="before")
    @classmethod
    def counts_are_exact_ints(cls, value: object) -> object:
        if type(value) is not int:
            raise ValueError("promotion counts must be exact integers")
        return value

    @field_validator(
        "recall",
        "ece",
        "p95_latency_ms",
        "challenge_rate",
        "false_decline_rate",
        "review_case_rate",
        mode="before",
    )
    @classmethod
    def numeric_values_are_exact_finite_floats(cls, value: object) -> object:
        if value is not None and (type(value) is not float or not math.isfinite(value)):
            raise ValueError("promotion metrics must be exact finite floats or None")
        return value

    @field_validator("preventable_settled_value", "value_escaped")
    @classmethod
    def money_is_exact_nonnegative_cents(cls, value: Decimal) -> Decimal:
        if type(value) is not Decimal or not value.is_finite() or value < 0:
            raise ValueError("promotion money must be a finite nonnegative Decimal")
        with localcontext(_MONEY_CONTEXT):
            if value.quantize(_CENT) != value:
                raise ValueError("promotion money must be cent denominated")
        return value

    @field_validator("slice_performance", mode="before")
    @classmethod
    def slices_are_exact_tuple(cls, value: object) -> object:
        if type(value) is not tuple:
            raise ValueError("slice performance must be an exact tuple")
        return value

    @model_validator(mode="after")
    def metrics_are_closed_and_ordered(self) -> PromotionMetrics:
        for name, value in (
            ("recall", self.recall),
            ("ECE", self.ece),
            ("challenge rate", self.challenge_rate),
            ("false-decline rate", self.false_decline_rate),
            ("review-case rate", self.review_case_rate),
        ):
            if value is not None and not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        if self.p95_latency_ms is not None and self.p95_latency_ms < 0.0:
            raise ValueError("p95 latency must be nonnegative")
        if self.review_case_count > self.row_count:
            raise ValueError("review-case count cannot exceed row count")
        keys = tuple((item.kind, item.value) for item in self.slice_performance)
        if keys != tuple(sorted(keys)) or len(keys) != len(set(keys)):
            raise ValueError("slice performance must be sorted and unique")
        expected_keys = tuple(
            sorted(
                (
                    *(("family", family) for family in _FAMILIES),
                    *(("rail", rail.value) for rail in Rail),
                    *(
                        ("regime", value)
                        for value in sorted(
                            ("baseline", *(item.value for item in RegimeKind))
                        )
                    ),
                    *(
                        ("entity_cohort", cohort.value)
                        for cohort in EntityCohort
                    ),
                )
            )
        )
        if keys != expected_keys:
            raise ValueError("promotion metrics require the complete closed slice vocabulary")
        return self


class ReplayFailure(ExternalContract):
    """Audited arm failure that receives no fallback credit."""

    code: Literal["MODEL_UNAVAILABLE", "MODEL_TIMEOUT"]
    failed_component_version: str

    @field_validator("failed_component_version")
    @classmethod
    def component_is_bounded_nonblank(cls, value: str) -> str:
        if type(value) is not str or not value or value.strip() != value or len(value) > 256:
            raise ValueError("failed component identity must be bounded nonblank text")
        return value


class ReplayResult(ExternalContract):
    """Aggregate public replay evidence; restricted truth never enters this model."""

    schema_version: Literal["1.0.0"] = "1.0.0"
    arm: DefenseArm
    evaluation: EvaluationDescriptor
    decision_event_ids: tuple[str, ...]
    decision_rows_digest: str
    common_integrity_digest: str
    action_digest: str
    score_digest: str
    threshold_report_digest: str
    threshold_set_digest: str
    bundle_manifest_digest: str
    case_callback_digest: str
    evaluation_context_digest: str
    hidden_release_receipt_digest: str | None
    metric_report_digest: str
    metrics: PromotionMetrics
    assurance: AssuranceEvidence
    failure: ReplayFailure | None = None
    fallback_count: int = Field(default=0, ge=0)
    mandatory_decline_count: int = Field(default=0, ge=0)
    result_digest: str

    @field_validator("decision_event_ids", mode="before")
    @classmethod
    def rows_are_exact_tuple(cls, value: object) -> object:
        if type(value) is not tuple:
            raise ValueError("decision event IDs must be an exact tuple")
        return value

    @field_validator("decision_event_ids")
    @classmethod
    def rows_are_canonical_unique(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if not value or any(type(item) is not str or not item for item in value):
            raise ValueError("decision event IDs must be nonempty exact text")
        if len(value) != len(set(value)):
            raise ValueError("decision event IDs must be unique")
        return value

    @field_validator(
        "decision_rows_digest",
        "common_integrity_digest",
        "action_digest",
        "score_digest",
        "threshold_report_digest",
        "threshold_set_digest",
        "bundle_manifest_digest",
        "case_callback_digest",
        "evaluation_context_digest",
        "hidden_release_receipt_digest",
        "metric_report_digest",
        "result_digest",
    )
    @classmethod
    def digests_are_sha256(cls, value: str | None) -> str | None:
        if value is not None:
            _validate_digest(value)
        return value

    @field_validator("fallback_count", "mandatory_decline_count", mode="before")
    @classmethod
    def counts_are_exact(cls, value: object) -> object:
        if type(value) is not int:
            raise ValueError("replay counts must be exact integers")
        return value

    @model_validator(mode="after")
    def result_is_self_consistent(self) -> ReplayResult:
        if self.metrics.row_count != len(self.decision_event_ids):
            raise ValueError("replay metrics and decision rows differ")
        if self.decision_rows_digest != _digest_document(list(self.decision_event_ids)):
            raise ValueError("decision row digest does not match exact ordered IDs")
        if max(self.fallback_count, self.mandatory_decline_count) > self.metrics.row_count:
            raise ValueError("replay counts exceed decision rows")
        if self.failure is not None and self.arm is not DefenseArm.GBDT_ONLY:
            raise ValueError("only GBDT-only may expose an audited model failure")
        if self.failure is not None and self.fallback_count:
            raise ValueError("failed GBDT-only replay cannot claim fallback")
        if self.evaluation.kind is EvaluationKind.HIDDEN:
            if self.assurance.hidden_access_clean != (
                self.hidden_release_receipt_digest is not None
            ):
                raise ValueError("hidden access evidence must bind an exact receipt")
        elif self.hidden_release_receipt_digest is not None:
            raise ValueError("non-hidden replay cannot claim hidden receipt evidence")
        expected = _digest_document(self.model_dump(mode="json", exclude={"result_digest"}))
        if self.result_digest != expected:
            raise ValueError("replay result digest is inconsistent")
        return self

    @classmethod
    def create(cls, **fields: object) -> ReplayResult:
        """Construct a canonical result and bind every aggregate field."""
        provisional = cast(Any, cls).model_construct(
            **fields, result_digest="0" * 64
        )
        document = provisional.model_dump(mode="json", exclude={"result_digest"})
        return cls.model_validate({**fields, "result_digest": _digest_document(document)})

    def rebuild(self, **updates: object) -> ReplayResult:
        """Return a fully revalidated result after an explicit test/evaluator update."""
        fields = {
            name: getattr(self, name)
            for name in type(self).model_fields
            if name != "result_digest"
        }
        fields.update(updates)
        return ReplayResult.create(**fields)

    def to_json(self) -> bytes:
        """Return canonical aggregate-only replay bytes."""
        if type(self) is not ReplayResult:
            raise GateContractError("replay result must be exact")
        checked = ReplayResult.model_validate(
            self.model_dump(mode="python", warnings=False), strict=True
        )
        payload = canonical_json_bytes(checked.model_dump(mode="json"))
        if len(payload) > 2_000_000:
            raise GateContractError("replay result payload exceeds resource cap")
        return payload

    @classmethod
    def from_json(cls, payload: bytes) -> ReplayResult:
        """Load bounded canonical replay aggregates and revalidate their digest."""
        if type(payload) is not bytes or len(payload) > 2_000_000:
            raise GateContractError("replay result payload is invalid")
        try:
            document = strict_json_loads(payload)
            if type(document) is not dict:
                raise GateContractError("replay result must be a JSON object")
            if type(document.get("decision_event_ids")) is list:
                document["decision_event_ids"] = tuple(document["decision_event_ids"])
            metrics = document.get("metrics")
            if type(metrics) is dict and type(metrics.get("slice_performance")) is list:
                metrics["slice_performance"] = tuple(metrics["slice_performance"])
            result = cls.model_validate(document)
            if result.to_json() != payload:
                raise GateContractError("replay result JSON is not canonical")
            return result
        except (ValidationError, WireContractError, ValueError) as error:
            raise GateContractError(str(error)) from error


class GateConfig(ExternalContract):
    """Frozen synthetic competition gate values."""

    schema_version: Literal["1.0.0"] = "1.0.0"
    challenge_rate_max: float = 0.02
    false_decline_rate_max: float = 0.001
    review_case_rate_max: float = 0.01
    minimum_family_recall: float = 0.50
    maximum_ece: float = 0.10
    maximum_p95_latency_ms: float = 50.0
    maximum_slice_recall_regression: float = 0.05
    minimum_value_improvement: Decimal = Decimal("0.01")

    @field_validator(
        "challenge_rate_max",
        "false_decline_rate_max",
        "review_case_rate_max",
        "minimum_family_recall",
        "maximum_ece",
        "maximum_p95_latency_ms",
        "maximum_slice_recall_regression",
        mode="before",
    )
    @classmethod
    def values_are_exact_finite_floats(cls, value: object) -> object:
        if type(value) is not float or not math.isfinite(value):
            raise ValueError("gate limits must be exact finite floats")
        return value

    @field_validator("minimum_value_improvement")
    @classmethod
    def improvement_is_exact_cent_decimal(cls, value: Decimal) -> Decimal:
        if type(value) is not Decimal or value != _CENT:
            raise ValueError("minimum value improvement must be the exact frozen cent")
        return value

    @model_validator(mode="after")
    def limits_are_closed(self) -> GateConfig:
        rates = (
            self.challenge_rate_max,
            self.false_decline_rate_max,
            self.review_case_rate_max,
            self.minimum_family_recall,
            self.maximum_ece,
            self.maximum_slice_recall_regression,
        )
        if any(not 0.0 <= value <= 1.0 for value in rates):
            raise ValueError("gate rates must be in [0, 1]")
        if self.maximum_p95_latency_ms < 0.0:
            raise ValueError("maximum p95 latency must be nonnegative")
        return self

    @classmethod
    def competition(cls) -> GateConfig:
        """Return the frozen competition profile."""
        return cls()


class ArmGateResult(ExternalContract):
    """Visible non-averageable failures for one arm."""

    arm: DefenseArm
    passed: bool
    failed_gate_codes: tuple[str, ...]

    @model_validator(mode="after")
    def failure_state_is_exact(self) -> ArmGateResult:
        if self.failed_gate_codes != tuple(sorted(set(self.failed_gate_codes))):
            raise ValueError("arm gate codes must be sorted and unique")
        if self.passed != (not self.failed_gate_codes):
            raise ValueError("arm pass state must match failed gate codes")
        return self


class ChampionStatus(StrEnum):
    PROMOTED = "promoted"
    RETAINED = "retained"
    NO_PROMOTION = "no_promotion"


class ChampionDecision(ExternalContract):
    """Canonical truthful outcome, including valid negative results."""

    schema_version: Literal["1.0.0"] = "1.0.0"
    status: ChampionStatus
    champion: DefenseArm | None
    failed_gate_codes: tuple[str, ...]
    arm_gate_results: tuple[ArmGateResult, ...]
    evaluated_result_digests: tuple[str, ...]
    decision_digest: str

    @model_validator(mode="after")
    def outcome_is_canonical(self) -> ChampionDecision:
        if self.failed_gate_codes != tuple(sorted(set(self.failed_gate_codes))):
            raise ValueError("decision gate codes must be sorted and unique")
        if tuple(item.arm for item in self.arm_gate_results) != tuple(DefenseArm):
            raise ValueError("arm gate results must be complete and ordered")
        if self.evaluated_result_digests != tuple(sorted(self.evaluated_result_digests)):
            raise ValueError("evaluated result digests must be sorted")
        if (self.status is ChampionStatus.NO_PROMOTION) != (self.champion is None):
            raise ValueError("no-promotion status must have no champion")
        if (
            self.status is ChampionStatus.PROMOTED
            and self.champion is not DefenseArm.LAYERED_HYBRID
        ):
            raise ValueError("only the layered hybrid can be promoted")
        if self.status is ChampionStatus.RETAINED and self.champion not in {
            DefenseArm.RULES_ONLY,
            DefenseArm.GBDT_ONLY,
        }:
            raise ValueError("retained champion must be a comparator")
        expected = _digest_document(self.model_dump(mode="json", exclude={"decision_digest"}))
        if self.decision_digest != expected:
            raise ValueError("champion decision digest is inconsistent")
        return self

    def to_json(self) -> bytes:
        """Serialize canonical aggregate-only decision evidence."""
        checked = ChampionDecision.model_validate(
            self.model_dump(mode="python", warnings=False), strict=True
        )
        return canonical_json_bytes(checked.model_dump(mode="json"))

    @classmethod
    def from_json(cls, payload: bytes) -> ChampionDecision:
        """Load only canonical, bounded, self-consistent decision bytes."""
        if type(payload) is not bytes or len(payload) > 2_000_000:
            raise GateContractError("champion decision payload is invalid")
        try:
            document = strict_json_loads(payload)
            if type(document) is not dict:
                raise GateContractError("champion decision must be a JSON object")
            decision = cls.model_validate(document)
            if decision.to_json() != payload:
                raise GateContractError("champion decision JSON is not canonical")
            return decision
        except (ValidationError, WireContractError) as error:
            raise GateContractError(str(error)) from error


def evaluate_promotion_gates(
    results: tuple[ReplayResult, ...], config: GateConfig
) -> ChampionDecision:
    """Apply every hard blocker before exact champion/challenger selection."""
    rows = _validated_results(results)
    checked_config = _exact_model(config, GateConfig, "gate config")
    by_arm: dict[DefenseArm, tuple[ReplayResult, ...]] = {
        arm: tuple(row for row in rows if row.arm is arm) for arm in DefenseArm
    }
    arm_codes: dict[DefenseArm, set[str]] = {
        arm: _hard_failure_codes(arm_rows, checked_config)
        for arm, arm_rows in by_arm.items()
    }
    if not _evaluation_lineage_is_exact(rows):
        for codes in arm_codes.values():
            codes.add("EVALUATION_LINEAGE")
    _apply_slice_regression(arm_codes, by_arm, checked_config)
    passing = {arm for arm in DefenseArm if not arm_codes[arm]}
    primary = {
        arm: next(
            (
                row
                for row in by_arm[arm]
                if row.evaluation.kind is EvaluationKind.CHRONOLOGICAL
            ),
            None,
        )
        for arm in DefenseArm
    }

    status: ChampionStatus
    champion: DefenseArm | None
    hybrid = primary[DefenseArm.LAYERED_HYBRID]
    rules_primary = primary[DefenseArm.RULES_ONLY]
    gbdt_primary = primary[DefenseArm.GBDT_ONLY]
    if hybrid is None or rules_primary is None or gbdt_primary is None:
        status = ChampionStatus.NO_PROMOTION
        champion = None
    elif DefenseArm.LAYERED_HYBRID in passing and _hybrid_qualifies(
        hybrid, (rules_primary, gbdt_primary), checked_config
    ):
        status = ChampionStatus.PROMOTED
        champion = DefenseArm.LAYERED_HYBRID
    else:
        comparators = (rules_primary, gbdt_primary)
        passing_comparators = tuple(
            row for row in comparators if row.arm in passing
        )
        if passing_comparators:
            best = min(
                passing_comparators,
                key=lambda row: (
                    -row.metrics.preventable_settled_value,
                    row.metrics.review_case_count,
                    tuple(DefenseArm).index(row.arm),
                ),
            )
            status = ChampionStatus.RETAINED
            champion = best.arm
        else:
            status = ChampionStatus.NO_PROMOTION
            champion = None

    arm_results = tuple(
        ArmGateResult(
            arm=arm,
            passed=not arm_codes[arm],
            failed_gate_codes=tuple(sorted(arm_codes[arm])),
        )
        for arm in DefenseArm
    )
    all_codes = tuple(sorted({code for codes in arm_codes.values() for code in codes}))
    fields: dict[str, object] = {
        "status": status,
        "champion": champion,
        "failed_gate_codes": all_codes,
        "arm_gate_results": arm_results,
        "evaluated_result_digests": tuple(sorted(row.result_digest for row in rows)),
    }
    provisional = cast(Any, ChampionDecision).model_construct(
        **fields, decision_digest="0" * 64
    )
    digest = _digest_document(provisional.model_dump(mode="json", exclude={"decision_digest"}))
    return ChampionDecision.model_validate({**fields, "decision_digest": digest})


def _validated_results(results: object) -> tuple[ReplayResult, ...]:
    if type(results) is not tuple or not results or len(results) > _MAX_RESULTS:
        raise GateContractError("replay results must be a bounded nonempty exact tuple")
    checked: list[ReplayResult] = []
    for row in cast(tuple[object, ...], results):
        checked.append(_exact_model(row, ReplayResult, "replay result"))
    keys = tuple((row.arm, row.evaluation.kind, row.evaluation.value) for row in checked)
    if len(keys) != len(set(keys)):
        raise GateContractError("replay arm/evaluation keys must be unique")
    return tuple(
        sorted(
            checked,
            key=lambda row: (
                tuple(EvaluationKind).index(row.evaluation.kind),
                row.evaluation.value,
                tuple(DefenseArm).index(row.arm),
            ),
        )
    )


def _hard_failure_codes(
    rows: tuple[ReplayResult, ...], config: GateConfig
) -> set[str]:
    codes: set[str] = set()
    expected = _required_descriptors()
    actual = frozenset((row.evaluation.kind, row.evaluation.value) for row in rows)
    if actual != expected:
        codes.add("EVALUATION_COVERAGE")
    for row in rows:
        assurance = row.assurance
        for passed, code in (
            (assurance.leakage_passed, "FEATURE_LEAKAGE"),
            (assurance.parity_passed, "FEATURE_PARITY"),
            (assurance.artifact_signature_valid, "ARTIFACT_SIGNATURE"),
            (assurance.rollback_available, "ROLLBACK_MISSING"),
            (assurance.hidden_access_clean, "HIDDEN_ACCESS"),
            (assurance.campaign_family_ownership_valid, "CAMPAIGN_FAMILY_OWNERSHIP"),
        ):
            if not passed:
                codes.add(code)
        if row.failure is not None:
            codes.add("MODEL_FAILURE")
        metrics = row.metrics
        if (
            metrics.challenge_rate > config.challenge_rate_max
            or metrics.false_decline_rate > config.false_decline_rate_max
            or metrics.review_case_rate > config.review_case_rate_max
        ):
            codes.add("OPERATING_BUDGET")
        if metrics.ece is None or metrics.ece > config.maximum_ece:
            codes.add("CALIBRATION_ECE")
        if metrics.p95_latency_ms is None or metrics.p95_latency_ms > config.maximum_p95_latency_ms:
            codes.add("P95_LATENCY")
        family_recall = {
            item.value: item.recall
            for item in metrics.slice_performance
            if item.kind == "family"
        }
        if row.evaluation.kind in {
            EvaluationKind.CHRONOLOGICAL,
            EvaluationKind.HIDDEN,
        }:
            families_to_check: tuple[str, ...] = _FAMILIES
        elif row.evaluation.kind is EvaluationKind.HELD_FAMILY:
            families_to_check = (row.evaluation.value,)
        else:
            families_to_check = ()
        if any(
            family_recall.get(family) is None
            or cast(float, family_recall[family]) < config.minimum_family_recall
            for family in families_to_check
        ):
            codes.add("PER_FAMILY_RECALL")
    return codes


def _evaluation_lineage_is_exact(rows: tuple[ReplayResult, ...]) -> bool:
    if len({row.bundle_manifest_digest for row in rows}) != 1:
        return False
    if len({row.threshold_set_digest for row in rows}) != 1:
        return False
    for arm in DefenseArm:
        arm_rows = tuple(row for row in rows if row.arm is arm)
        if len({row.threshold_report_digest for row in arm_rows}) > 1:
            return False
    descriptor_keys = {
        (row.evaluation.kind, row.evaluation.value) for row in rows
    }
    for key in descriptor_keys:
        descriptor_rows = tuple(
            row
            for row in rows
            if (row.evaluation.kind, row.evaluation.value) == key
        )
        if {row.arm for row in descriptor_rows} != set(DefenseArm):
            continue
        lineage = {
            (
                row.decision_event_ids,
                row.decision_rows_digest,
                row.common_integrity_digest,
                row.bundle_manifest_digest,
                row.threshold_set_digest,
                row.case_callback_digest,
                row.evaluation_context_digest,
                row.hidden_release_receipt_digest,
            )
            for row in descriptor_rows
        }
        if len(lineage) != 1:
            return False
    return True


def _apply_slice_regression(
    arm_codes: dict[DefenseArm, set[str]],
    by_arm: dict[DefenseArm, tuple[ReplayResult, ...]],
    config: GateConfig,
) -> None:
    lookup = {
        (row.arm, row.evaluation.kind, row.evaluation.value): row
        for rows in by_arm.values()
        for row in rows
    }
    hybrid_rows = by_arm[DefenseArm.LAYERED_HYBRID]
    for hybrid in hybrid_rows:
        comparators = tuple(
            lookup.get((arm, hybrid.evaluation.kind, hybrid.evaluation.value))
            for arm in (DefenseArm.RULES_ONLY, DefenseArm.GBDT_ONLY)
        )
        if any(item is None for item in comparators):
            continue
        comparison_by_key: dict[tuple[str, str], list[float]] = {}
        for comparator in cast(tuple[ReplayResult, ReplayResult], comparators):
            for item in comparator.metrics.slice_performance:
                if item.recall is not None:
                    comparison_by_key.setdefault((item.kind, item.value), []).append(item.recall)
        for item in hybrid.metrics.slice_performance:
            comparator_values = comparison_by_key.get((item.kind, item.value), [])
            if comparator_values and (
                item.recall is None
                or max(comparator_values) - item.recall
                > config.maximum_slice_recall_regression
            ):
                arm_codes[DefenseArm.LAYERED_HYBRID].add("SLICE_RECALL_REGRESSION")
                return


def _hybrid_qualifies(
    hybrid: ReplayResult,
    comparators: tuple[ReplayResult, ReplayResult],
    config: GateConfig,
) -> bool:
    with localcontext(_MONEY_CONTEXT):
        improvements = tuple(
            hybrid.metrics.preventable_settled_value
            - comparator.metrics.preventable_settled_value
            for comparator in comparators
        )
        if all(value >= config.minimum_value_improvement for value in improvements):
            return True
        best_value = max(
            comparator.metrics.preventable_settled_value for comparator in comparators
        )
        best_comparators = tuple(
            comparator
            for comparator in comparators
            if comparator.metrics.preventable_settled_value == best_value
        )
        within = (
            best_value - hybrid.metrics.preventable_settled_value
            <= config.minimum_value_improvement
        )
        lower_workload = all(
            hybrid.metrics.review_case_count < comparator.metrics.review_case_count
            for comparator in best_comparators
        )
        return within and lower_workload


def _required_descriptors() -> frozenset[tuple[EvaluationKind, str]]:
    return frozenset(
        {
            (EvaluationKind.CHRONOLOGICAL, "development"),
            (EvaluationKind.HIDDEN, "hidden"),
            *((EvaluationKind.HELD_FAMILY, family) for family in _FAMILIES),
            *((EvaluationKind.REGIME, item.value) for item in RegimeKind),
            *((EvaluationKind.COLD_ENTITY, item.value) for item in EntityCohort),
        }
    )


def _exact_model[T: ExternalContract](value: object, expected: type[T], label: str) -> T:
    if type(value) is not expected:
        raise GateContractError(f"{label} must have its exact contract type")
    try:
        return expected.model_validate(
            value.model_dump(mode="python", warnings=False), strict=True
        )
    except ValidationError as error:
        raise GateContractError(f"{label} failed semantic revalidation") from error


def _validate_digest(value: str) -> None:
    if type(value) is not str or len(value) != _SHA256_LENGTH:
        raise ValueError("digest must be lowercase SHA-256")
    if value != value.lower() or any(character not in "0123456789abcdef" for character in value):
        raise ValueError("digest must be lowercase SHA-256")


def _digest_document(document: object) -> str:
    return hashlib.sha256(canonical_json_bytes(document)).hexdigest()


__all__ = [
    "ArmGateResult",
    "AssuranceEvidence",
    "ChampionDecision",
    "ChampionStatus",
    "DefenseArm",
    "EvaluationDescriptor",
    "EvaluationKind",
    "GateConfig",
    "GateContractError",
    "PromotionMetrics",
    "ReplayFailure",
    "ReplayResult",
    "SlicePerformance",
    "evaluate_promotion_gates",
]
