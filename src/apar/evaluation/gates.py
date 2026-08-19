"""Closed hard gates and truthful champion/challenger selection."""

from __future__ import annotations

import base64
import binascii
import hashlib
import math
from collections.abc import Iterator
from decimal import ROUND_HALF_EVEN, Context, Decimal, localcontext
from enum import StrEnum
from typing import Any, Literal, Never, cast
from weakref import WeakKeyDictionary

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)
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


class EvaluationLineage(ExternalContract):
    """Aggregate descriptor provenance derived from evaluator-owned source receipts."""

    schema_version: Literal["1.0.0"] = "1.0.0"
    descriptor: EvaluationDescriptor
    decision_rows_digest: str
    decision_content_digest: str
    split_digest: str
    cohort_mapping_digest: str
    training_population_digest: str
    bundle_manifest_digest: str
    defender_top_ref_digest: str
    regime_parent_digest: str | None = None
    regime_output_digest: str | None = None
    regime_parameters_digest: str | None = None
    regime_truth_unchanged: bool | None = None
    held_family: Family | None = None
    training_exclusion_verified: bool = False
    lineage_digest: str

    @field_validator(
        "decision_rows_digest",
        "decision_content_digest",
        "split_digest",
        "cohort_mapping_digest",
        "training_population_digest",
        "bundle_manifest_digest",
        "defender_top_ref_digest",
        "regime_parent_digest",
        "regime_output_digest",
        "regime_parameters_digest",
        "lineage_digest",
    )
    @classmethod
    def lineage_digests_are_sha256(cls, value: str | None) -> str | None:
        if value is not None:
            _validate_digest(value)
        return value

    @field_validator("regime_truth_unchanged", "training_exclusion_verified", mode="before")
    @classmethod
    def lineage_flags_are_exact(cls, value: object) -> object:
        if value is not None and type(value) is not bool:
            raise ValueError("evaluation lineage flags must be exact bools or None")
        return value

    @model_validator(mode="after")
    def kind_specific_provenance_is_closed(self) -> EvaluationLineage:
        regime_values = (
            self.regime_parent_digest,
            self.regime_output_digest,
            self.regime_parameters_digest,
            self.regime_truth_unchanged,
        )
        if self.descriptor.kind is EvaluationKind.REGIME:
            if any(value is None for value in regime_values):
                raise ValueError("regime lineage requires complete derivation provenance")
        elif any(value is not None for value in regime_values):
            raise ValueError("non-regime lineage cannot claim regime provenance")
        if self.descriptor.kind is EvaluationKind.HELD_FAMILY:
            if (
                self.held_family != self.descriptor.value
                or not self.training_exclusion_verified
            ):
                raise ValueError("held-family lineage requires exact exclusion proof")
        elif self.held_family is not None or self.training_exclusion_verified:
            raise ValueError("non-held lineage cannot claim family exclusion")
        expected = _digest_document(
            self.model_dump(mode="json", exclude={"lineage_digest"})
        )
        if self.lineage_digest != expected:
            raise ValueError("evaluation lineage digest is inconsistent")
        return self

    @classmethod
    def create(cls, **fields: object) -> EvaluationLineage:
        provisional = cast(Any, cls).model_construct(
            **fields, lineage_digest="0" * 64
        )
        document = provisional.model_dump(mode="json", exclude={"lineage_digest"})
        return cls.model_validate(
            {**fields, "lineage_digest": _digest_document(document)}
        )


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


class RateEvidence(ExternalContract):
    """Exact rate derivation with explicit undefined-denominator semantics."""

    schema_version: Literal["1.0.0"] = "1.0.0"
    numerator: int = Field(ge=0, le=1_000_000)
    denominator: int = Field(ge=0, le=1_000_000)
    value: float | None
    defined: bool

    @field_validator("numerator", "denominator", mode="before")
    @classmethod
    def counts_are_exact_ints(cls, value: object) -> object:
        if type(value) is not int:
            raise ValueError("rate evidence counts must be exact integers")
        return value

    @field_validator("defined", mode="before")
    @classmethod
    def defined_is_exact_bool(cls, value: object) -> object:
        if type(value) is not bool:
            raise ValueError("rate evidence defined flag must be exact bool")
        return value

    @field_validator("value", mode="before")
    @classmethod
    def value_is_finite_rate_or_none(cls, value: object) -> object:
        if value is not None and (
            type(value) is not float or not math.isfinite(value) or not 0.0 <= value <= 1.0
        ):
            raise ValueError("rate evidence value must be None or an exact finite rate")
        return value

    @model_validator(mode="after")
    def derivation_is_exact(self) -> RateEvidence:
        if self.numerator > self.denominator:
            raise ValueError("rate evidence numerator cannot exceed denominator")
        expected_defined = self.denominator > 0
        if self.defined != expected_defined:
            raise ValueError("rate evidence defined flag disagrees with denominator")
        expected_value = (
            self.numerator / self.denominator if expected_defined else None
        )
        if self.value != expected_value:
            raise ValueError("rate evidence value disagrees with exact counts")
        return self

class PromotionMetrics(ExternalContract):
    """Aggregate-only metrics needed by promotion, without evaluator truth."""

    schema_version: Literal["1.1.0"] = "1.1.0"
    row_count: int = Field(ge=1, le=1_000_000)
    recall: float | None
    ece: float | None
    p95_latency_ms: float | None
    preventable_settled_value: Decimal
    value_escaped: Decimal
    review_case_count: int = Field(ge=0)
    challenge_rate: float
    false_decline: RateEvidence
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
            ("review-case rate", self.review_case_rate),
        ):
            if value is not None and not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        if self.p95_latency_ms is not None and self.p95_latency_ms < 0.0:
            raise ValueError("p95 latency must be nonnegative")
        if self.review_case_count > self.row_count:
            raise ValueError("review-case count cannot exceed row count")
        if self.false_decline.denominator > self.row_count:
            raise ValueError("false-decline denominator cannot exceed row count")
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


class CandidateBundleRole(StrEnum):
    """Closed candidate identity roles across robustness descriptors."""

    POOLED = "pooled"
    HELD_FAMILY_LOFO = "held_family_lofo"


class CandidateRoleEvidence(ExternalContract):
    """Exact bundle/top-ref/threshold identity for one candidate role."""

    schema_version: Literal["1.0.0"] = "1.0.0"
    role: CandidateBundleRole
    held_family: Family | None = None
    bundle_id: str
    bundle_manifest_digest: str
    defender_top_ref_digest: str
    threshold_set_digest: str
    role_digest: str

    @field_validator("bundle_id")
    @classmethod
    def bundle_id_is_bounded(cls, value: str) -> str:
        if type(value) is not str or not value or len(value) > 128:
            raise ValueError("candidate bundle ID must be bounded nonempty text")
        return value

    @field_validator(
        "bundle_manifest_digest",
        "defender_top_ref_digest",
        "threshold_set_digest",
        "role_digest",
    )
    @classmethod
    def role_digests_are_sha256(cls, value: str) -> str:
        _validate_digest(value)
        return value

    @model_validator(mode="after")
    def role_is_closed(self) -> CandidateRoleEvidence:
        if self.role is CandidateBundleRole.POOLED:
            if self.held_family is not None:
                raise ValueError("pooled candidate cannot claim a held family")
        elif self.held_family not in _FAMILIES:
            raise ValueError("LOFO candidate requires one strategic held family")
        expected = _digest_document(
            self.model_dump(mode="json", exclude={"role_digest"})
        )
        if self.role_digest != expected:
            raise ValueError("candidate role digest is inconsistent")
        return self

    @classmethod
    def create(cls, **fields: object) -> CandidateRoleEvidence:
        provisional = cast(Any, cls).model_construct(
            **fields, role_digest="0" * 64
        )
        document = provisional.model_dump(mode="json", exclude={"role_digest"})
        return cls.model_validate(
            {**fields, "role_digest": _digest_document(document)}
        )


class ReplayResult(ExternalContract):
    """Aggregate public replay evidence; restricted truth never enters this model."""

    schema_version: Literal["1.3.0"] = "1.3.0"
    arm: DefenseArm
    evaluation: EvaluationDescriptor
    evaluation_lineage: EvaluationLineage
    candidate_role: CandidateRoleEvidence
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
    hidden_public_proof_id: str | None
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
        "metric_report_digest",
        "result_digest",
    )
    @classmethod
    def digests_are_sha256(cls, value: str | None) -> str | None:
        if value is not None:
            _validate_digest(value)
        return value

    @field_validator("hidden_public_proof_id")
    @classmethod
    def hidden_proof_id_is_opaque(cls, value: str | None) -> str | None:
        if value is not None and (
            type(value) is not str
            or not value.startswith("hpf_")
            or len(value) != 36
            or any(character not in "0123456789abcdef" for character in value[4:])
        ):
            raise ValueError("hidden public proof ID is invalid")
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
        if (
            self.evaluation_lineage.descriptor != self.evaluation
            or self.evaluation_lineage.decision_rows_digest
            != self.decision_rows_digest
            or self.evaluation_lineage.bundle_manifest_digest
            != self.bundle_manifest_digest
        ):
            raise ValueError("replay result descriptor lineage is inconsistent")
        if (
            type(self.candidate_role) is not CandidateRoleEvidence
            or self.candidate_role.bundle_manifest_digest
            != self.bundle_manifest_digest
            or self.candidate_role.defender_top_ref_digest
            != self.evaluation_lineage.defender_top_ref_digest
            or self.candidate_role.threshold_set_digest != self.threshold_set_digest
        ):
            raise ValueError("replay candidate role identity is inconsistent")
        if self.evaluation.kind is EvaluationKind.HELD_FAMILY:
            if (
                self.candidate_role.role is not CandidateBundleRole.HELD_FAMILY_LOFO
                or self.candidate_role.held_family != self.evaluation.value
            ):
                raise ValueError("held-family replay requires its exact LOFO role")
        elif (
            self.candidate_role.role is not CandidateBundleRole.POOLED
            or self.candidate_role.held_family is not None
        ):
            raise ValueError("non-held replay requires the pooled candidate role")
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
                self.hidden_public_proof_id is not None
            ):
                raise ValueError("hidden access evidence must bind an opaque public proof")
        elif self.hidden_public_proof_id is not None:
            raise ValueError("non-hidden replay cannot claim hidden public proof")
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


class _SealedIdentityType(type):
    """Prevent runtime replacement of pinned evaluator identity behavior."""

    def __setattr__(cls, name: str, value: object) -> None:
        del cls, name, value
        raise TypeError("evaluator identity types are sealed")

    def __delattr__(cls, name: str) -> None:
        del cls, name
        raise TypeError("evaluator identity types are sealed")


def _signer_identity_store() -> tuple[Any, Any]:
    issued: WeakKeyDictionary[object, bool] = WeakKeyDictionary()

    def register(instance: object) -> None:
        issued[instance] = True

    def contains(instance: object) -> bool:
        return issued.get(instance, False)

    return register, contains


_register_signer_identity, _is_registered_signer = _signer_identity_store()


class EvaluatorSigningIdentity(metaclass=_SealedIdentityType):
    """Explicit Ed25519 identity used only for evaluator aggregate evidence."""

    __slots__ = ("__weakref__",)

    def __new__(cls, *args: object, **kwargs: object) -> EvaluatorSigningIdentity:
        del cls, args, kwargs
        raise TypeError("evaluator signing identities require the private-byte factory")

    @property
    def key_id(self) -> str:
        raise TypeError("unbound evaluator signing identity")

    @property
    def public_key_base64(self) -> str:
        raise TypeError("unbound evaluator signing identity")

    def _private_key(self) -> Ed25519PrivateKey:
        raise TypeError("unbound evaluator signing identity")

    def _private_seed_bytes(self) -> bytes:
        raise TypeError("unbound evaluator signing identity")

    @classmethod
    def from_private_bytes(cls, private_seed: bytes) -> EvaluatorSigningIdentity:
        if type(private_seed) is not bytes or len(private_seed) != 32:
            raise ValueError("evaluator private seed must be exactly 32 bytes")
        return _new_evaluator_signer(bytes(private_seed))

    @classmethod
    def is_exact(cls, value: object) -> bool:
        del cls
        return _is_evaluator_signer(value)

    def sign_batch(self, results: tuple[ReplayResult, ...]) -> VerifiedReplayBatch:
        return VerifiedReplayBatch.create(results=results, signer=self)

    def _sign(self, document: object) -> str:
        return base64.b64encode(
            self._private_key().sign(canonical_json_bytes(document))
        ).decode("ascii")

    def _worker_private_bytes(self) -> bytes:
        """Return an isolated-process copy; callers never receive restricted data."""
        return self._private_seed_bytes()

    def __setattr__(self, name: str, value: object) -> None:
        del name, value
        raise TypeError("evaluator signing identity is immutable")

    def __delattr__(self, name: str) -> None:
        del name
        raise TypeError("evaluator signing identity is immutable")

    def __copy__(self) -> Never:
        raise TypeError("evaluator signing identity cannot be copied")

    def __deepcopy__(self, memo: object) -> Never:
        del memo
        raise TypeError("evaluator signing identity cannot be copied")

    def __reduce__(self) -> Never:
        raise TypeError("evaluator signing identity cannot be serialized")


def _new_evaluator_signer(private_seed: bytes) -> EvaluatorSigningIdentity:
    key = Ed25519PrivateKey.from_private_bytes(private_seed)
    public = key.public_key().public_bytes(
        serialization.Encoding.Raw, serialization.PublicFormat.Raw
    )
    key_id = hashlib.sha256(public).hexdigest()
    public_key_base64 = base64.b64encode(public).decode("ascii")
    instance: EvaluatorSigningIdentity | None = None

    class _BoundEvaluatorSigner(EvaluatorSigningIdentity):
        __slots__ = ()

        def __new__(cls, *args: object, **kwargs: object) -> _BoundEvaluatorSigner:
            del cls, args, kwargs
            raise TypeError("bound evaluator signer cannot be constructed")

        def __init__(self, *args: object, **kwargs: object) -> None:
            del self, args, kwargs
            raise TypeError("evaluator signing identity cannot be reinitialized")

        @property
        def key_id(self) -> str:
            return key_id

        @property
        def public_key_base64(self) -> str:
            return public_key_base64

        def _private_key(self) -> Ed25519PrivateKey:
            if self is not instance:
                raise TypeError("evaluator signer identity is invalid")
            return key

        def _private_seed_bytes(self) -> bytes:
            if self is not instance:
                raise TypeError("evaluator signer identity is invalid")
            return key.private_bytes_raw()

    instance = cast(EvaluatorSigningIdentity, object.__new__(_BoundEvaluatorSigner))
    _register_signer_identity(instance)
    return instance


def _is_evaluator_signer(value: object) -> bool:
    if not isinstance(value, EvaluatorSigningIdentity) or not _is_registered_signer(value):
        return False
    try:
        return len(value._worker_private_bytes()) == 32
    except (AttributeError, TypeError, ValueError):
        return False


def _verifier_state_store() -> tuple[Any, Any]:
    states: WeakKeyDictionary[object, tuple[Ed25519PublicKey, str, str]] = (
        WeakKeyDictionary()
    )

    def register(
        instance: object,
        state: tuple[Ed25519PublicKey, str, str],
    ) -> None:
        if instance in states:
            raise TypeError("evaluator verifier cannot be reinitialized")
        states[instance] = state

    def get(instance: object) -> tuple[Ed25519PublicKey, str, str]:
        try:
            return states[instance]
        except KeyError as error:
            raise TypeError("evaluator verifier identity is invalid") from error

    return register, get


_register_verifier_state, _get_verifier_state = _verifier_state_store()


class EvaluatorReplayVerifier(metaclass=_SealedIdentityType):
    """Separately pinned, externally immutable evaluator verification identity."""

    __slots__ = ("__weakref__",)

    def __init__(
        self,
        *,
        signer_key_id: str,
        public_key_base64: str,
        _register: Any = _register_verifier_state,
        _get: Any = _get_verifier_state,
    ) -> None:
        if type(self) is not EvaluatorReplayVerifier:
            raise TypeError("evaluator verifier must have its exact type")
        try:
            _get(self)
        except TypeError:
            pass
        else:
            raise TypeError("evaluator verifier cannot be reinitialized")
        public = self._validate_key_fields(signer_key_id, public_key_base64)
        _register(
            self,
            (
                Ed25519PublicKey.from_public_bytes(public),
                signer_key_id,
                public_key_base64,
            ),
        )

    @property
    def key_id(self, _get: Any = _get_verifier_state) -> str:
        return cast(tuple[Ed25519PublicKey, str, str], _get(self))[1]

    @property
    def public_key_base64(self, _get: Any = _get_verifier_state) -> str:
        return cast(tuple[Ed25519PublicKey, str, str], _get(self))[2]

    def _public_key(self, _get: Any = _get_verifier_state) -> Ed25519PublicKey:
        return cast(tuple[Ed25519PublicKey, str, str], _get(self))[0]

    @staticmethod
    def _validate_key_fields(
        signer_key_id: str, public_key_base64: str
    ) -> bytes:
        _validate_digest(signer_key_id)
        if type(public_key_base64) is not str:
            raise GateContractError("evaluator public key encoding is invalid")
        try:
            public = base64.b64decode(public_key_base64, validate=True)
        except (ValueError, binascii.Error) as error:
            raise GateContractError("evaluator public key encoding is invalid") from error
        if len(public) != 32 or hashlib.sha256(public).hexdigest() != signer_key_id:
            raise GateContractError("evaluator public key identity is inconsistent")
        return public

    @classmethod
    def from_signer(cls, signer: EvaluatorSigningIdentity) -> EvaluatorReplayVerifier:
        if not EvaluatorSigningIdentity.is_exact(signer):
            raise GateContractError("evaluator signer must have its exact type")
        return cls(
            signer_key_id=signer.key_id,
            public_key_base64=signer.public_key_base64,
        )

    def verify_batch(self, batch: VerifiedReplayBatch) -> bool:
        if type(batch) is not VerifiedReplayBatch or batch.signer_key_id != self.key_id:
            return False
        try:
            signature = base64.b64decode(batch.signature_base64, validate=True)
            self._public_key().verify(
                signature, canonical_json_bytes(batch.unsigned_document())
            )
        except (InvalidSignature, TypeError, ValueError, binascii.Error):
            return False
        expected = _digest_document(
            {**batch.unsigned_document(), "signature_base64": batch.signature_base64}
        )
        return expected == batch.batch_digest

    def verify_document(self, document: object, signature_base64: str) -> bool:
        """Verify a canonical evaluator evidence document under the pinned key."""
        if type(signature_base64) is not str:
            return False
        try:
            signature = base64.b64decode(signature_base64, validate=True)
            self._public_key().verify(signature, canonical_json_bytes(document))
        except (InvalidSignature, TypeError, ValueError, binascii.Error):
            return False
        return True

    def verify_public_proof(self, proof: HiddenPublicProof) -> bool:
        if type(proof) is not HiddenPublicProof or proof.signer_key_id != self.key_id:
            return False
        try:
            signature = base64.b64decode(proof.signature_base64, validate=True)
            self._public_key().verify(
                signature, canonical_json_bytes(proof.unsigned_document())
            )
        except (InvalidSignature, TypeError, ValueError, binascii.Error):
            return False
        expected = _digest_document(
            {**proof.unsigned_document(), "signature_base64": proof.signature_base64}
        )
        return expected == proof.proof_digest

    def verify_promotion_envelope(self, envelope: VerifiedPromotionEnvelope) -> bool:
        if (
            type(envelope) is not VerifiedPromotionEnvelope
            or envelope.signer_key_id != self.key_id
        ):
            return False
        if not self.verify_document(
            envelope.unsigned_document(), envelope.signature_base64
        ):
            return False
        expected = _digest_document(
            {
                **envelope.unsigned_document(),
                "signature_base64": envelope.signature_base64,
            }
        )
        return expected == envelope.envelope_digest

    def __setattr__(self, name: str, value: object) -> None:
        del name, value
        raise TypeError("evaluator verifier is immutable")

    def __delattr__(self, name: str) -> None:
        del name
        raise TypeError("evaluator verifier is immutable")

    def __copy__(self) -> Never:
        raise TypeError("evaluator verifier cannot be copied")

    def __deepcopy__(self, memo: object) -> Never:
        del memo
        raise TypeError("evaluator verifier cannot be copied")

    def __reduce__(self) -> Never:
        raise TypeError("evaluator verifier cannot be serialized")


class VerifiedReplayBatch(ExternalContract):
    """Evaluator-signed exact replay results; raw result tuples are never trusted."""

    schema_version: Literal["1.1.0"] = "1.1.0"
    results: tuple[ReplayResult, ...]
    batch_content_digest: str
    signer_key_id: str
    signature_base64: str
    batch_digest: str

    @field_validator("results", mode="before")
    @classmethod
    def results_are_exact_tuple(cls, value: object) -> object:
        if type(value) is not tuple:
            raise ValueError("verified replay results must be an exact tuple")
        return value

    @field_validator("batch_content_digest", "signer_key_id", "batch_digest")
    @classmethod
    def batch_digests_are_sha256(cls, value: str) -> str:
        _validate_digest(value)
        return value

    @model_validator(mode="after")
    def batch_is_closed(self) -> VerifiedReplayBatch:
        if not self.results or len(self.results) > _MAX_RESULTS:
            raise ValueError("verified replay batch must be bounded and nonempty")
        if any(type(row) is not ReplayResult for row in self.results):
            raise ValueError("verified replay batch contains a nonexact result")
        keys = tuple(
            (row.evaluation.kind, row.evaluation.value, row.arm) for row in self.results
        )
        expected_keys = tuple(
            sorted(
                keys,
                key=lambda item: (
                    tuple(EvaluationKind).index(item[0]),
                    item[1],
                    tuple(DefenseArm).index(item[2]),
                ),
            )
        )
        if keys != expected_keys or len(keys) != len(set(keys)):
            raise ValueError("verified replay batch order or keys are invalid")
        expected_content = _batch_content_digest(self.results)
        if self.batch_content_digest != expected_content:
            raise ValueError("verified replay batch content digest is inconsistent")
        expected_digest = _digest_document(
            {**self.unsigned_document(), "signature_base64": self.signature_base64}
        )
        if self.batch_digest != expected_digest:
            raise ValueError("verified replay batch digest is inconsistent")
        return self

    @classmethod
    def create(
        cls,
        *,
        results: tuple[ReplayResult, ...],
        signer: EvaluatorSigningIdentity,
    ) -> VerifiedReplayBatch:
        if not EvaluatorSigningIdentity.is_exact(signer):
            raise GateContractError("verified replay batch requires exact evaluator signer")
        checked_results = _validated_results(results)
        unsigned = {
            "schema_version": "1.1.0",
            "results": [row.model_dump(mode="json") for row in checked_results],
            "batch_content_digest": _batch_content_digest(checked_results),
            "signer_key_id": signer.key_id,
        }
        signature = signer._sign(unsigned)
        digest = _digest_document({**unsigned, "signature_base64": signature})
        return cls(
            results=checked_results,
            batch_content_digest=cast(str, unsigned["batch_content_digest"]),
            signer_key_id=signer.key_id,
            signature_base64=signature,
            batch_digest=digest,
        )

    def unsigned_document(self) -> dict[str, object]:
        return self.model_dump(
            mode="json", exclude={"signature_base64", "batch_digest"}
        )

    def to_json(self) -> bytes:
        if type(self) is not VerifiedReplayBatch:
            raise GateContractError("verified replay batch must be exact")
        checked = VerifiedReplayBatch.model_validate(
            self.model_dump(mode="python", warnings=False), strict=True
        )
        payload = canonical_json_bytes(checked.model_dump(mode="json"))
        if len(payload) > 32_000_000:
            raise GateContractError("verified replay batch exceeds its resource cap")
        return payload

    @classmethod
    def from_json(cls, payload: bytes) -> VerifiedReplayBatch:
        if type(payload) is not bytes or not 0 < len(payload) <= 32_000_000:
            raise GateContractError("verified replay batch payload is invalid")
        try:
            document = strict_json_loads(payload)
            if type(document) is not dict or type(document.get("results")) is not list:
                raise GateContractError("verified replay batch must be an object")
            document["results"] = tuple(
                ReplayResult.from_json(canonical_json_bytes(item))
                for item in document["results"]
            )
            batch = cls.model_validate(document)
            if batch.to_json() != payload:
                raise GateContractError("verified replay batch JSON is not canonical")
            return batch
        except (ValidationError, WireContractError, ValueError) as error:
            raise GateContractError(str(error)) from error

    def __iter__(self) -> Iterator[ReplayResult]:  # type: ignore[override]
        return iter(self.results)

    def __len__(self) -> int:
        return len(self.results)

    def __getitem__(self, index: int) -> ReplayResult:
        return self.results[index]


class HiddenPublicProof(ExternalContract):
    """Opaque public proof joining aggregate hidden results to evaluator authority."""

    schema_version: Literal["1.2.0"] = "1.2.0"
    proof_id: str
    batch_content_digest: str
    decision_bindings_digest: str
    bundle_manifest_digest: str
    defender_top_ref_digest: str
    worker_manifest_digest: str
    evaluator_context_token: str
    cohort_mapping_token: str
    issued_at: str
    signer_key_id: str
    signature_base64: str
    proof_digest: str

    @field_validator("proof_id")
    @classmethod
    def proof_id_is_opaque(cls, value: str) -> str:
        checked = ReplayResult.hidden_proof_id_is_opaque(value)
        assert checked is not None
        return checked

    @field_validator(
        "batch_content_digest",
        "decision_bindings_digest",
        "bundle_manifest_digest",
        "defender_top_ref_digest",
        "worker_manifest_digest",
        "evaluator_context_token",
        "cohort_mapping_token",
        "signer_key_id",
        "proof_digest",
    )
    @classmethod
    def proof_digests_are_sha256(cls, value: str) -> str:
        _validate_digest(value)
        return value

    @field_validator("issued_at")
    @classmethod
    def issued_at_is_canonical_utc(cls, value: str) -> str:
        if (
            type(value) is not str
            or len(value) > 40
            or not value.endswith("Z")
            or "T" not in value
        ):
            raise ValueError("hidden public proof timestamp is invalid")
        return value

    @model_validator(mode="after")
    def proof_is_self_consistent(self) -> HiddenPublicProof:
        expected = _digest_document(
            {**self.unsigned_document(), "signature_base64": self.signature_base64}
        )
        if self.proof_digest != expected:
            raise ValueError("hidden public proof digest is inconsistent")
        return self

    @classmethod
    def create(
        cls,
        *,
        proof_id: str,
        batch_content_digest: str,
        decision_bindings_digest: str,
        bundle_manifest_digest: str,
        defender_top_ref_digest: str,
        worker_manifest_digest: str,
        evaluator_context_token: str,
        cohort_mapping_token: str,
        issued_at: str,
        signer: EvaluatorSigningIdentity,
    ) -> HiddenPublicProof:
        if not EvaluatorSigningIdentity.is_exact(signer):
            raise GateContractError("hidden public proof requires exact evaluator signer")
        fields: dict[str, object] = {
            "schema_version": "1.2.0",
            "proof_id": proof_id,
            "batch_content_digest": batch_content_digest,
            "decision_bindings_digest": decision_bindings_digest,
            "bundle_manifest_digest": bundle_manifest_digest,
            "defender_top_ref_digest": defender_top_ref_digest,
            "worker_manifest_digest": worker_manifest_digest,
            "evaluator_context_token": evaluator_context_token,
            "cohort_mapping_token": cohort_mapping_token,
            "issued_at": issued_at,
            "signer_key_id": signer.key_id,
        }
        signature = signer._sign(fields)
        digest = _digest_document({**fields, "signature_base64": signature})
        return cls(
            schema_version="1.2.0",
            proof_id=proof_id,
            batch_content_digest=batch_content_digest,
            decision_bindings_digest=decision_bindings_digest,
            bundle_manifest_digest=bundle_manifest_digest,
            defender_top_ref_digest=defender_top_ref_digest,
            worker_manifest_digest=worker_manifest_digest,
            evaluator_context_token=evaluator_context_token,
            cohort_mapping_token=cohort_mapping_token,
            issued_at=issued_at,
            signer_key_id=signer.key_id,
            signature_base64=signature,
            proof_digest=digest,
        )

    def unsigned_document(self) -> dict[str, object]:
        return self.model_dump(
            mode="json", exclude={"signature_base64", "proof_digest"}
        )

    def to_json(self) -> bytes:
        if type(self) is not HiddenPublicProof:
            raise GateContractError("hidden public proof must be exact")
        checked = HiddenPublicProof.model_validate(
            self.model_dump(mode="python", warnings=False), strict=True
        )
        payload = canonical_json_bytes(checked.model_dump(mode="json"))
        if len(payload) > 64_000:
            raise GateContractError("hidden public proof exceeds resource cap")
        return payload

    @classmethod
    def from_json(cls, payload: bytes) -> HiddenPublicProof:
        if type(payload) is not bytes or not 0 < len(payload) <= 64_000:
            raise GateContractError("hidden public proof payload is invalid")
        try:
            document = strict_json_loads(payload)
            proof = cls.model_validate(document)
            if proof.to_json() != payload:
                raise GateContractError("hidden public proof JSON is not canonical")
            return proof
        except (ValidationError, WireContractError, ValueError) as error:
            raise GateContractError(str(error)) from error


class VerifiedPromotionEnvelope(ExternalContract):
    """Evaluator-signed complete matrix joined to exact hidden public proofs."""

    schema_version: Literal["1.0.0"] = "1.0.0"
    component_batches: tuple[VerifiedReplayBatch, ...]
    hidden_proofs: tuple[HiddenPublicProof, ...]
    combined_batch: VerifiedReplayBatch
    signer_key_id: str
    signature_base64: str
    envelope_digest: str

    @field_validator("component_batches", "hidden_proofs", mode="before")
    @classmethod
    def collections_are_exact_tuples(cls, value: object) -> object:
        if type(value) is not tuple:
            raise ValueError("promotion envelope collections must be exact tuples")
        return value

    @field_validator("signer_key_id", "envelope_digest")
    @classmethod
    def envelope_digests_are_sha256(cls, value: str) -> str:
        _validate_digest(value)
        return value

    @model_validator(mode="after")
    def envelope_is_closed(self) -> VerifiedPromotionEnvelope:
        if not self.component_batches or len(self.component_batches) > 128:
            raise ValueError("promotion envelope component count is invalid")
        if len(self.hidden_proofs) > 8:
            raise ValueError("promotion envelope hidden proof count is invalid")
        if any(type(item) is not VerifiedReplayBatch for item in self.component_batches):
            raise ValueError("promotion envelope component batch is not exact")
        if any(type(item) is not HiddenPublicProof for item in self.hidden_proofs):
            raise ValueError("promotion envelope hidden proof is not exact")
        if type(self.combined_batch) is not VerifiedReplayBatch:
            raise ValueError("promotion envelope combined batch is not exact")
        expected = _digest_document(
            {**self.unsigned_document(), "signature_base64": self.signature_base64}
        )
        if self.envelope_digest != expected:
            raise ValueError("promotion envelope digest is inconsistent")
        return self

    @classmethod
    def create(
        cls,
        *,
        component_batches: tuple[VerifiedReplayBatch, ...],
        hidden_proofs: tuple[HiddenPublicProof, ...],
        signer: EvaluatorSigningIdentity,
        hidden_proof_verifier: EvaluatorReplayVerifier,
    ) -> VerifiedPromotionEnvelope:
        if not EvaluatorSigningIdentity.is_exact(signer):
            raise GateContractError("promotion envelope requires exact evaluator signer")
        verifier = EvaluatorReplayVerifier.from_signer(signer)
        if type(hidden_proof_verifier) is not EvaluatorReplayVerifier:
            raise GateContractError("promotion envelope requires pinned hidden verifier")
        checked_components, checked_proofs = _validate_envelope_components(
            component_batches,
            hidden_proofs,
            evaluator_verifier=verifier,
            hidden_proof_verifier=hidden_proof_verifier,
        )
        combined = signer.sign_batch(
            tuple(row for batch in checked_components for row in batch.results)
        )
        fields: dict[str, object] = {
            "schema_version": "1.0.0",
            "component_batches": [
                item.model_dump(mode="json") for item in checked_components
            ],
            "hidden_proofs": [item.model_dump(mode="json") for item in checked_proofs],
            "combined_batch": combined.model_dump(mode="json"),
            "signer_key_id": signer.key_id,
        }
        signature = signer._sign(fields)
        digest = _digest_document({**fields, "signature_base64": signature})
        return cls(
            component_batches=checked_components,
            hidden_proofs=checked_proofs,
            combined_batch=combined,
            signer_key_id=signer.key_id,
            signature_base64=signature,
            envelope_digest=digest,
        )

    def unsigned_document(self) -> dict[str, object]:
        return self.model_dump(
            mode="json", exclude={"signature_base64", "envelope_digest"}
        )

    def to_json(self) -> bytes:
        if type(self) is not VerifiedPromotionEnvelope:
            raise GateContractError("promotion envelope must be exact")
        checked = VerifiedPromotionEnvelope.model_validate(
            self.model_dump(mode="python", warnings=False), strict=True
        )
        payload = canonical_json_bytes(checked.model_dump(mode="json"))
        if len(payload) > 64_000_000:
            raise GateContractError("promotion envelope exceeds resource cap")
        return payload

    @classmethod
    def from_json(cls, payload: bytes) -> VerifiedPromotionEnvelope:
        if type(payload) is not bytes or not 0 < len(payload) <= 64_000_000:
            raise GateContractError("promotion envelope payload is invalid")
        try:
            document = strict_json_loads(payload)
            if type(document) is not dict:
                raise GateContractError("promotion envelope must be an object")
            raw_components = document.get("component_batches")
            raw_proofs = document.get("hidden_proofs")
            raw_combined = document.get("combined_batch")
            if (
                type(raw_components) is not list
                or type(raw_proofs) is not list
                or type(raw_combined) is not dict
            ):
                raise GateContractError("promotion envelope field types are invalid")
            document["component_batches"] = tuple(
                VerifiedReplayBatch.from_json(canonical_json_bytes(item))
                for item in raw_components
            )
            document["hidden_proofs"] = tuple(
                HiddenPublicProof.from_json(canonical_json_bytes(item))
                for item in raw_proofs
            )
            document["combined_batch"] = VerifiedReplayBatch.from_json(
                canonical_json_bytes(raw_combined)
            )
            envelope = cls.model_validate(document)
            if envelope.to_json() != payload:
                raise GateContractError("promotion envelope JSON is not canonical")
            return envelope
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


def _validate_envelope_components(
    component_batches: object,
    hidden_proofs: object,
    *,
    evaluator_verifier: EvaluatorReplayVerifier,
    hidden_proof_verifier: EvaluatorReplayVerifier,
) -> tuple[tuple[VerifiedReplayBatch, ...], tuple[HiddenPublicProof, ...]]:
    if type(component_batches) is not tuple or not component_batches:
        raise GateContractError("promotion components must be a nonempty exact tuple")
    if type(hidden_proofs) is not tuple:
        raise GateContractError("promotion hidden proofs must be an exact tuple")
    components = cast(tuple[object, ...], component_batches)
    proofs = cast(tuple[object, ...], hidden_proofs)
    if len(components) > 128 or len(proofs) > 8:
        raise GateContractError("promotion evidence exceeds resource caps")
    checked_components: list[VerifiedReplayBatch] = []
    descriptor_keys: list[tuple[EvaluationKind, str]] = []
    for item in components:
        if type(item) is not VerifiedReplayBatch:
            raise GateContractError("promotion component type is invalid")
        rows = item.results
        if not rows:
            raise GateContractError("promotion component rows are empty")
        component_verifier = (
            hidden_proof_verifier
            if rows[0].evaluation.kind is EvaluationKind.HIDDEN
            else evaluator_verifier
        )
        if not component_verifier.verify_batch(item):
            raise GateContractError("promotion component signature is invalid")
        if not rows or len(rows) > len(DefenseArm) or len({row.arm for row in rows}) != len(rows):
            raise GateContractError("promotion component arm membership is invalid")
        descriptors = {(row.evaluation.kind, row.evaluation.value) for row in rows}
        if len(descriptors) != 1:
            raise GateContractError("promotion component spans multiple descriptors")
        descriptor_keys.append(next(iter(descriptors)))
        checked_components.append(item)
    ordered = tuple(
        sorted(
            zip(descriptor_keys, checked_components, strict=True),
            key=lambda item: (tuple(EvaluationKind).index(item[0][0]), item[0][1]),
        )
    )
    if tuple(zip(descriptor_keys, checked_components, strict=True)) != ordered:
        raise GateContractError("promotion components are not canonically ordered")
    if len(descriptor_keys) != len(set(descriptor_keys)):
        raise GateContractError("promotion component descriptors are duplicated")
    checked_proofs: list[HiddenPublicProof] = []
    for item in proofs:
        if (
            type(item) is not HiddenPublicProof
            or not hidden_proof_verifier.verify_public_proof(item)
        ):
            raise GateContractError("promotion hidden proof signature is invalid")
        checked_proofs.append(item)
    if tuple(item.proof_id for item in checked_proofs) != tuple(
        sorted(item.proof_id for item in checked_proofs)
    ) or len({item.proof_id for item in checked_proofs}) != len(checked_proofs):
        raise GateContractError("promotion hidden proofs are not canonical")
    hidden_batches = tuple(
        batch
        for key, batch in ordered
        if key[0] is EvaluationKind.HIDDEN
    )
    if len(hidden_batches) != len(checked_proofs):
        raise GateContractError("each hidden descriptor requires one exact public proof")
    proofs_by_batch = {item.batch_content_digest: item for item in checked_proofs}
    if len(proofs_by_batch) != len(checked_proofs):
        raise GateContractError("hidden proof batch bindings are duplicated")
    for batch in hidden_batches:
        proof = proofs_by_batch.get(batch.batch_content_digest)
        if proof is None:
            raise GateContractError("hidden proof does not bind its descriptor batch")
        for row in batch.results:
            if (
                row.hidden_public_proof_id != proof.proof_id
                or row.evaluation_context_digest != proof.evaluator_context_token
                or row.evaluation_lineage.cohort_mapping_digest
                != proof.cohort_mapping_token
                or row.bundle_manifest_digest != proof.bundle_manifest_digest
                or row.evaluation_lineage.defender_top_ref_digest
                != proof.defender_top_ref_digest
                or not row.assurance.hidden_access_clean
            ):
                raise GateContractError("hidden proof aggregate bindings are invalid")
    return tuple(checked_components), tuple(checked_proofs)


def _validated_promotion_envelope(
    envelope: object,
    *,
    evaluator_verifier: EvaluatorReplayVerifier,
    hidden_proof_verifier: EvaluatorReplayVerifier,
) -> VerifiedReplayBatch:
    if (
        type(envelope) is not VerifiedPromotionEnvelope
        or not evaluator_verifier.verify_promotion_envelope(envelope)
    ):
        raise GateContractError("promotion requires a verified evaluator envelope")
    components, _ = _validate_envelope_components(
        envelope.component_batches,
        envelope.hidden_proofs,
        evaluator_verifier=evaluator_verifier,
        hidden_proof_verifier=hidden_proof_verifier,
    )
    combined = envelope.combined_batch
    if not evaluator_verifier.verify_batch(combined):
        raise GateContractError("promotion combined replay batch is invalid")
    expected = tuple(row for batch in components for row in batch.results)
    if _validated_results(expected) != combined.results:
        raise GateContractError("promotion envelope matrix differs from its components")
    return combined


def evaluate_promotion_gates(
    envelope: VerifiedPromotionEnvelope,
    config: GateConfig,
    *,
    evaluator_verifier: EvaluatorReplayVerifier | None = None,
    hidden_proof_verifier: EvaluatorReplayVerifier | None = None,
) -> ChampionDecision:
    """Apply every hard blocker before exact champion/challenger selection."""
    if (
        type(evaluator_verifier) is not EvaluatorReplayVerifier
        or type(hidden_proof_verifier) is not EvaluatorReplayVerifier
    ):
        raise GateContractError("promotion requires exact pinned evaluator verifiers")
    batch = _validated_promotion_envelope(
        envelope,
        evaluator_verifier=evaluator_verifier,
        hidden_proof_verifier=hidden_proof_verifier,
    )
    rows = _validated_results(batch.results)
    checked_config = _exact_model(config, GateConfig, "gate config")
    by_arm: dict[DefenseArm, tuple[ReplayResult, ...]] = {
        arm: tuple(row for row in rows if row.arm is arm) for arm in DefenseArm
    }
    arm_codes: dict[DefenseArm, set[str]] = {
        arm: _hard_failure_codes(arm_rows, checked_config)
        for arm, arm_rows in by_arm.items()
    }
    matrix_complete = _evaluation_matrix_is_complete(rows)
    if not matrix_complete:
        for codes in arm_codes.values():
            codes.add("EVALUATION_COVERAGE")
    if not _evaluation_lineage_is_exact(rows):
        for codes in arm_codes.values():
            codes.add("EVALUATION_LINEAGE")
    if not _candidate_roles_are_exact(rows):
        for codes in arm_codes.values():
            codes.add("CANDIDATE_ROLE")
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
    if not matrix_complete or hybrid is None or rules_primary is None or gbdt_primary is None:
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
        if not metrics.false_decline.defined:
            codes.add("FALSE_DECLINE_COVERAGE")
        if (
            metrics.challenge_rate > config.challenge_rate_max
            or (
                metrics.false_decline.value is not None
                and metrics.false_decline.value > config.false_decline_rate_max
            )
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
        family_floor_failed: bool
        if row.evaluation.kind in {
            EvaluationKind.CHRONOLOGICAL,
            EvaluationKind.HIDDEN,
        }:
            families_to_check: tuple[str, ...] = _FAMILIES
            family_floor_failed = any(
                family_recall.get(family) is None
                or cast(float, family_recall[family])
                < config.minimum_family_recall
                for family in families_to_check
            )
        elif row.evaluation.kind is EvaluationKind.HELD_FAMILY:
            families_to_check = (row.evaluation.value,)
            family_floor_failed = any(
                family_recall.get(family) is None
                or cast(float, family_recall[family])
                < config.minimum_family_recall
                for family in families_to_check
            )
        else:
            family_floor_failed = any(
                recall is not None and recall < config.minimum_family_recall
                for recall in family_recall.values()
            )
        if family_floor_failed:
            codes.add("PER_FAMILY_RECALL")
    return codes


def _evaluation_lineage_is_exact(rows: tuple[ReplayResult, ...]) -> bool:
    threshold_groups: dict[tuple[DefenseArm, str, str], set[str]] = {}
    for row in rows:
        threshold_groups.setdefault(
            (row.arm, row.bundle_manifest_digest, row.threshold_set_digest), set()
        ).add(row.threshold_report_digest)
    if any(len(digests) != 1 for digests in threshold_groups.values()):
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
            return False
        lineage = {
            (
                row.decision_event_ids,
                row.decision_rows_digest,
                row.evaluation_lineage.lineage_digest,
                row.candidate_role.role_digest,
                row.common_integrity_digest,
                row.bundle_manifest_digest,
                row.threshold_set_digest,
                row.case_callback_digest,
                row.evaluation_context_digest,
                row.hidden_public_proof_id,
            )
            for row in descriptor_rows
        }
        if len(lineage) != 1:
            return False
    return True


def _candidate_roles_are_exact(rows: tuple[ReplayResult, ...]) -> bool:
    pooled_identities: set[tuple[str, str, str, str]] = set()
    lofo_identities: dict[Family, set[tuple[str, str, str, str]]] = {
        family: set() for family in _FAMILIES
    }
    for row in rows:
        evidence = row.candidate_role
        identity = (
            evidence.bundle_id,
            evidence.bundle_manifest_digest,
            evidence.defender_top_ref_digest,
            evidence.threshold_set_digest,
        )
        if row.evaluation.kind is EvaluationKind.HELD_FAMILY:
            family = cast(Family, row.evaluation.value)
            if (
                evidence.role is not CandidateBundleRole.HELD_FAMILY_LOFO
                or evidence.held_family != family
            ):
                return False
            lofo_identities[family].add(identity)
        else:
            if (
                evidence.role is not CandidateBundleRole.POOLED
                or evidence.held_family is not None
            ):
                return False
            pooled_identities.add(identity)
    if len(pooled_identities) != 1 or any(
        len(identities) != 1 for identities in lofo_identities.values()
    ):
        return False
    pooled = next(iter(pooled_identities))
    lofo = tuple(next(iter(lofo_identities[family])) for family in _FAMILIES)
    return len(set(lofo)) == len(_FAMILIES) and pooled not in set(lofo)


def _evaluation_matrix_is_complete(rows: tuple[ReplayResult, ...]) -> bool:
    expected = {
        (arm, kind, value)
        for arm in DefenseArm
        for kind, value in _required_descriptors()
    }
    actual = {
        (row.arm, row.evaluation.kind, row.evaluation.value) for row in rows
    }
    return actual == expected


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


def _batch_content_digest(results: tuple[ReplayResult, ...]) -> str:
    return _digest_document(
        {
            "schema_version": "1.0.0",
            "results": [item.model_dump(mode="json") for item in results],
        }
    )


__all__ = [
    "ArmGateResult",
    "AssuranceEvidence",
    "CandidateBundleRole",
    "CandidateRoleEvidence",
    "ChampionDecision",
    "ChampionStatus",
    "DefenseArm",
    "EvaluationDescriptor",
    "EvaluationLineage",
    "EvaluationKind",
    "EvaluatorReplayVerifier",
    "EvaluatorSigningIdentity",
    "GateConfig",
    "GateContractError",
    "HiddenPublicProof",
    "PromotionMetrics",
    "RateEvidence",
    "ReplayFailure",
    "ReplayResult",
    "SlicePerformance",
    "VerifiedReplayBatch",
    "VerifiedPromotionEnvelope",
    "evaluate_promotion_gates",
]
