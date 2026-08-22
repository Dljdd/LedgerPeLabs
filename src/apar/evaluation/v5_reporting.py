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

    return V5DevelopmentResult(
        status=status,
        profile=corpus.profile.value,
        protocol_sha256=protocol.protocol_sha256,
        corpus_sha256=corpus.corpus_sha256,
        fidelity_status=audit.overall_status,
    )


__all__ = ["V5DevelopmentResult", "build_v5_development_result"]
