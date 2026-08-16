"""Compile approved threat evidence into bounded synthetic scenarios."""

from urllib.parse import urlparse

from pydantic import ValidationError

from apar.compiler.errors import CompilerError
from apar.contracts.scenarios import (
    FeedbackField,
    ReplayManifest,
    ScenarioBundle,
    ScenarioConfig,
)
from apar.registry.models import ThreatCard

_SAFE_CLASS = "synthetic_only"
_SAFE_EXPORT_LEVEL = "sanitized"
_SUPPORTED_FEEDBACK = frozenset(field.value for field in FeedbackField)
_EVIDENCE_FIELDS = frozenset({"direct_source_url", "evidence", "evidence_id", "threat_id"})
_RAIL_FIELDS = frozenset({"observables", "rail", "rails", "viewpoint"})
_SAFETY_FIELDS = frozenset({"export_level", "safety_class", "status"})


def _validation_code(error: ValidationError, default: str) -> str:
    fields = {
        part
        for detail in error.errors()
        for part in detail["loc"]
        if isinstance(part, str)
    }
    if fields & _EVIDENCE_FIELDS:
        return "MISSING_EVIDENCE"
    if fields & _RAIL_FIELDS:
        return "UNSUPPORTED_RAIL"
    if fields & _SAFETY_FIELDS:
        return "UNSAFE_EXPORT"
    return default


def _validated_card(card: ThreatCard) -> ThreatCard:
    try:
        fields = card.model_dump(mode="python", round_trip=True, warnings=False)
        return ThreatCard.model_validate(fields)
    except ValidationError as error:
        code = _validation_code(error, "MISSING_EVIDENCE")
        raise CompilerError(code, "threat card failed validation") from None


def _validated_config(config: ScenarioConfig) -> ScenarioConfig:
    try:
        fields = config.model_dump(mode="python", round_trip=True, warnings=False)
        return ScenarioConfig.model_validate(fields)
    except ValidationError as error:
        code = _validation_code(error, "INVALID_FEEDBACK")
        raise CompilerError(code, "scenario configuration failed validation") from None


def _has_direct_source(card: ThreatCard) -> bool:
    for evidence in card.evidence:
        direct_source_url = getattr(evidence, "direct_source_url", None)
        if not isinstance(direct_source_url, str):
            continue
        try:
            parsed = urlparse(direct_source_url)
        except ValueError:
            continue
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
    card = _validated_card(card)
    config = _validated_config(config)
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
        campaign_stages=config.campaign_stages,
        transition_rules=config.transition_rules,
        economics=config.economics,
        lifecycle=config.lifecycle,
        hidden_validity=config.hidden_validity,
        defender_knowledge_boundary=card.defender_knowledge_boundary,
        replay_manifest=ReplayManifest(
            scenario_id=config.scenario_id,
            scenario_version=config.version,
            threat_card_ref=f"{card.threat_id}@{card.version}",
            random_seed=config.replay.random_seed,
            simulation_start=config.replay.simulation_start,
            generator_version=config.replay.generator_version,
            event_ordering=config.replay.event_ordering,
        ),
        safety={"synthetic_only": True, "export_level": config.export_level},
        extensions=config.extensions,
    )
