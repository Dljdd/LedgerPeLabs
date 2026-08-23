"""Development result reporting for Defend v5."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from apar.evaluation.v5_evaluation import V5Arm, V5EvaluationResult
from apar.evaluation.v5_hardening import V5HardeningResult
from apar.evaluation.v5_population import V5Corpus
from apar.evaluation.v5_protocol import V5DevelopmentProtocol

_VALID_STATUSES = {
    "development_ready", "development_not_ready", "invalid_corpus", "smoke",
}
_FORBIDDEN_CLAIMS = {
    "winner", "production_ready", "competition_validated", "confirmatory_supported",
}
_REQUIRED_ARMS = (
    V5Arm.RULES_ONLY.value,
    V5Arm.ENSEMBLE_NO_GRAPH.value,
    V5Arm.ENSEMBLE_WITH_GRAPH.value,
    V5Arm.FULL_SENTINEL.value,
)


def _digest(document: object) -> str:
    return hashlib.sha256(
        json.dumps(
            document,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


class V5DevelopmentResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    status: str
    profile: str = ""
    protocol_sha256: str = ""
    catalog_sha256: str = ""
    corpus_sha256: str = ""
    fidelity_status: str = ""
    failed_gates: tuple[str, ...] = ()
    arms: dict[str, V5EvaluationResult] = Field(default_factory=dict)
    hardening: V5HardeningResult | None = None
    result_sha256: str = ""

    @field_validator("status")
    @classmethod
    def status_is_valid(cls, value: str) -> str:
        if value in _FORBIDDEN_CLAIMS:
            raise ValueError(f"forbidden status claim: {value}")
        if value not in _VALID_STATUSES:
            raise ValueError(f"invalid status: {value}")
        return value

    @model_validator(mode="after")
    def final_evidence_is_complete(self) -> V5DevelopmentResult:
        if tuple(self.arms) != _REQUIRED_ARMS:
            raise ValueError("development result requires the exact four ordered arms")
        if any(name != result.arm for name, result in self.arms.items()):
            raise ValueError("development result arm key and result name mismatch")
        if len({result.support_sha256 for result in self.arms.values()}) != 1:
            raise ValueError("development arms do not share common ordered support")
        if any(
            result.arm_spec is None
            or result.arm_spec.protocol_sha256 != self.protocol_sha256
            or result.arm_spec.catalog_sha256 != self.catalog_sha256
            or result.arm_spec.spec_sha256 != result.arm_spec_sha256
            for result in self.arms.values()
        ):
            raise ValueError("development arm protocol/catalog/spec binding mismatch")
        if len({result.arm_spec_sha256 for result in self.arms.values()}) != len(_REQUIRED_ARMS):
            raise ValueError("development result contains cloned arm specifications")
        specs = tuple(
            result.arm_spec
            for result in self.arms.values()
            if result.arm_spec is not None
        )
        shared_bindings = {
            (
                spec.arm_config_sha256,
                spec.implementation_version,
                spec.implementation_sha256,
                spec.bootstrap_seed,
                spec.catalog_feature_names,
                spec.catalog_feature_groups,
                tuple(
                    json.dumps(
                        partition.model_dump(mode="json"),
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                    for partition in spec.training_partitions
                ),
            )
            for spec in specs
        }
        if len(shared_bindings) != 1:
            raise ValueError("development result contains mixed arm training provenance")
        if self.status == "development_ready" and (
            self.profile != "production"
            or self.fidelity_status != "pass"
            or self.failed_gates
        ):
            raise ValueError("development_ready requires complete passing production evidence")
        expected = _digest(self.model_dump(mode="json", exclude={"result_sha256"}))
        if self.result_sha256 != expected:
            raise ValueError("development result digest mismatch")
        return self


def build_v5_development_result(
    *,
    protocol: V5DevelopmentProtocol,
    corpus: V5Corpus,
    arms: Mapping[str, object] | None = None,
    catalog_sha256: str = "",
) -> V5DevelopmentResult:
    """Build a development evidence artifact from the completed pipeline."""
    from apar.evaluation.v5_fidelity import audit_v5_fidelity

    audit = audit_v5_fidelity(corpus)

    if not corpus.is_production:
        status = "smoke"
    elif audit.overall_status != "pass":
        status = "invalid_corpus"
    else:
        status = "development_not_ready"

    from apar.evaluation.v5_evaluation import V5EvaluationResult

    if tuple((arms or {}).keys()) != _REQUIRED_ARMS:
        raise ValueError("development result requires the exact four ordered arms")
    parsed_arms = {
        name: V5EvaluationResult.model_validate(data)
        for name, data in (arms or {}).items()
    }

    failed_gates: list[str] = []
    for arm_result in parsed_arms.values():
        if (
            arm_result.recall is not None
            and arm_result.recall < protocol.readiness.family_recall_min
        ):
            failed_gates.append("family_recall_min")
        if (
            arm_result.false_decline_rate is not None
            and arm_result.false_decline_rate > protocol.readiness.false_decline_rate_max
        ):
            failed_gates.append("false_decline_rate_max")
        if (
            arm_result.challenge_rate is not None
            and arm_result.challenge_rate > protocol.readiness.challenge_rate_max
        ):
            failed_gates.append("challenge_rate_max")
        if (
            arm_result.review_rate is not None
            and arm_result.review_rate > protocol.readiness.manual_review_rate_max
        ):
            failed_gates.append("manual_review_rate_max")
        if (
            arm_result.expected_calibration_error is not None
            and arm_result.expected_calibration_error
            > protocol.readiness.expected_calibration_error_max
        ):
            failed_gates.append("expected_calibration_error_max")
        if (
            arm_result.captured_value_fraction is not None
            and arm_result.captured_value_fraction < protocol.readiness.captured_value_fraction_min
        ):
            failed_gates.append("captured_value_fraction_min")
        if arm_result.p95_latency_ms is None:
            failed_gates.append("latency_missing")
        elif arm_result.p95_latency_ms > protocol.readiness.p95_decision_latency_ms_max:
            failed_gates.append("p95_decision_latency_ms_max")

    if not corpus.is_production:
        status = "smoke"
    elif audit.overall_status != "pass":
        status = "invalid_corpus"
    elif not failed_gates:
        status = "development_ready"
    else:
        status = "development_not_ready"

    document = {
        "status": status,
        "profile": corpus.profile.value,
        "protocol_sha256": protocol.protocol_sha256,
        "catalog_sha256": catalog_sha256,
        "corpus_sha256": corpus.corpus_sha256,
        "fidelity_status": audit.overall_status,
        "failed_gates": tuple(sorted(set(failed_gates))),
        "arms": {
            name: result.model_dump(mode="json") for name, result in parsed_arms.items()
        },
        "hardening": None,
    }
    document["result_sha256"] = _digest(document)
    return V5DevelopmentResult.model_validate(document)


__all__ = ["V5DevelopmentResult", "build_v5_development_result"]
