"""Compile approved threat evidence into bounded synthetic scenarios."""

from urllib.parse import urlparse

from apar.compiler.errors import CompilerError
from apar.contracts.scenarios import FeedbackField, ScenarioBundle, ScenarioConfig
from apar.registry.models import ThreatCard

_SAFE_CLASS = "synthetic_only"
_SAFE_EXPORT_LEVEL = "sanitized"
_SUPPORTED_FEEDBACK = frozenset(field.value for field in FeedbackField)


def _has_direct_source(card: ThreatCard) -> bool:
    for evidence in card.evidence:
        direct_source_url = getattr(evidence, "direct_source_url", None)
        if not isinstance(direct_source_url, str):
            continue
        parsed = urlparse(direct_source_url)
        if parsed.scheme in {"http", "https"} and parsed.netloc:
            return True
    return False


def _is_positive_integer(value: object) -> bool:
    return type(value) is int and value > 0


def _is_supported_feedback(value: object) -> bool:
    return isinstance(value, str) and value in _SUPPORTED_FEEDBACK


def _reject_invalid_card(card: ThreatCard, config: ScenarioConfig) -> None:
    if not _has_direct_source(card):
        raise CompilerError("MISSING_EVIDENCE", "at least one direct HTTP source is required")

    if config.rail not in card.rails or not card.observables:
        raise CompilerError(
            "UNSUPPORTED_RAIL",
            "the selected rail must be approved and expose at least one observable",
        )

    if (
        card.status != "approved"
        or card.safety_class != _SAFE_CLASS
        or config.export_level != _SAFE_EXPORT_LEVEL
    ):
        raise CompilerError(
            "UNSAFE_EXPORT", "only approved, synthetic-only cards may produce sanitized exports"
        )

    if (
        not config.feedback
        or any(not _is_supported_feedback(value) for value in config.feedback)
        or not _is_positive_integer(config.benign_entity_count)
        or not _is_positive_integer(config.illicit_entity_count)
        or not _is_positive_integer(config.duration_hours)
        or type(config.seed) is not int
    ):
        raise CompilerError(
            "INVALID_FEEDBACK",
            "feedback and deterministic execution bounds must be supported and non-empty",
        )


def compile_scenario(card: ThreatCard, config: ScenarioConfig) -> ScenarioBundle:
    """Compile a reviewed card without adding undocumented scenario assumptions."""
    _reject_invalid_card(card, config)
    return ScenarioBundle(
        schema_version=config.schema_version,
        scenario_id=config.scenario_id,
        version=config.version,
        threat_card_ref=f"{card.threat_id}@{card.version}",
        rail=config.rail,
        viewpoint=config.viewpoint,
        genai_capability=card.genai_capability,
        attacker_mode=config.attacker_mode,
        attacker_objective=config.attacker_objective,
        query_budget=config.query_budget,
        feedback=config.feedback,
        benign_entity_count=config.benign_entity_count,
        illicit_entity_count=config.illicit_entity_count,
        duration_hours=config.duration_hours,
        seed=config.seed,
        economics=config.economics,
        lifecycle=config.lifecycle,
        hidden_validity=config.hidden_validity,
        safety={"synthetic_only": True, "export_level": config.export_level},
        extensions=config.extensions,
    )
