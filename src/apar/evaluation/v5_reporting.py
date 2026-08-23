"""Development result reporting for Defend v5."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator

from apar.evaluation.v5_evaluation import V5EvaluationResult
from apar.evaluation.v5_hardening import V5HardeningResult
from apar.evaluation.v5_population import V5Corpus
from apar.evaluation.v5_protocol import V5DevelopmentProtocol

_VALID_STATUSES = {
    "development_ready", "development_not_ready", "invalid_corpus", "smoke",
}
_FORBIDDEN_CLAIMS = {
    "winner", "production_ready", "competition_validated", "confirmatory_supported",
}


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

    @field_validator("status")
    @classmethod
    def status_is_valid(cls, value: str) -> str:
        if value in _FORBIDDEN_CLAIMS:
            raise ValueError(f"forbidden status claim: {value}")
        if value not in _VALID_STATUSES:
            raise ValueError(f"invalid status: {value}")
        return value


def build_v5_development_result(
    *,
    protocol: V5DevelopmentProtocol,
    corpus: V5Corpus,
    arms: dict | None = None,
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

    parsed_arms = {
        name: V5EvaluationResult.model_validate(data)
        for name, data in (arms or {}).items()
        if isinstance(data, dict) and "arm" in data
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
            arm_result.captured_value_fraction is not None
            and arm_result.captured_value_fraction < protocol.readiness.captured_value_fraction_min
        ):
            failed_gates.append("captured_value_fraction_min")

    if not corpus.is_production:
        status = "smoke"
    elif audit.overall_status != "pass":
        status = "invalid_corpus"
    elif failed_gates or not parsed_arms:
        status = "development_not_ready"
    else:
        status = "development_not_ready"

    return V5DevelopmentResult(
        status=status,
        profile=corpus.profile.value,
        protocol_sha256=protocol.protocol_sha256,
        catalog_sha256=catalog_sha256,
        corpus_sha256=corpus.corpus_sha256,
        fidelity_status=audit.overall_status,
        arms=dict(parsed_arms),
        failed_gates=tuple(sorted(set(failed_gates))),
    )


__all__ = ["V5DevelopmentResult", "build_v5_development_result"]
