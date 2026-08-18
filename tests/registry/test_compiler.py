from collections.abc import Callable

import pytest

from apar.compiler.compiler import compile_scenario
from apar.compiler.errors import CompilerError
from apar.contracts.events import Rail
from apar.contracts.scenarios import FeedbackField, ScenarioConfig
from apar.registry.models import ThreatCard
from tests.factories import make_threat_card

CardMutation = Callable[[ThreatCard], ThreatCard]
ConfigMutation = Callable[[ScenarioConfig], ScenarioConfig]


def _same_card(card: ThreatCard) -> ThreatCard:
    return card


def _same_config(config: ScenarioConfig) -> ScenarioConfig:
    return config


@pytest.mark.parametrize(
    ("case", "mutate_card", "mutate_config", "expected_code"),
    [
        (
            "no evidence records",
            lambda card: card.model_copy(update={"evidence": []}),
            _same_config,
            "MISSING_EVIDENCE",
        ),
        (
            "no direct source URL",
            lambda card: card.model_copy(
                update={
                    "evidence": [
                        card.evidence[0].model_copy(update={"direct_source_url": ""})
                    ]
                }
            ),
            _same_config,
            "MISSING_EVIDENCE",
        ),
        (
            "evidence record is structurally malformed",
            lambda card: card.model_copy(update={"evidence": [{}]}),
            _same_config,
            "MISSING_EVIDENCE",
        ),
        (
            "rail not approved by card",
            _same_card,
            lambda config: config.model_copy(update={"rail": Rail.CARD}),
            "UNSUPPORTED_RAIL",
        ),
        (
            "no network observables",
            lambda card: card.model_copy(update={"observables": []}),
            _same_config,
            "UNSUPPORTED_RAIL",
        ),
        (
            "safety class omitted",
            lambda card: card.model_copy(update={"safety_class": None}),
            _same_config,
            "UNSAFE_EXPORT",
        ),
        (
            "card is not approved",
            lambda card: card.model_copy(update={"status": "under_review"}),
            _same_config,
            "UNSAFE_EXPORT",
        ),
        (
            "safety class permits non-synthetic execution",
            lambda card: card.model_copy(update={"safety_class": "live_targeting"}),
            _same_config,
            "UNSAFE_EXPORT",
        ),
        (
            "export is not sanitized",
            _same_card,
            lambda config: config.model_copy(update={"export_level": "raw"}),
            "UNSAFE_EXPORT",
        ),
        (
            "feedback field is not supported",
            _same_card,
            lambda config: config.model_copy(update={"feedback": ["model_gradient"]}),
            "INVALID_FEEDBACK",
        ),
        (
            "feedback field is structurally malformed",
            _same_card,
            lambda config: config.model_copy(update={"feedback": [{"gradient": True}]}),
            "INVALID_FEEDBACK",
        ),
        (
            "feedback is empty",
            _same_card,
            lambda config: config.model_copy(update={"feedback": []}),
            "INVALID_FEEDBACK",
        ),
        (
            "benign population is not positive",
            _same_card,
            lambda config: config.model_copy(update={"benign_entity_count": 0}),
            "INVALID_FEEDBACK",
        ),
        (
            "illicit population is not positive",
            _same_card,
            lambda config: config.model_copy(update={"illicit_entity_count": 0}),
            "INVALID_FEEDBACK",
        ),
        (
            "duration is not positive",
            _same_card,
            lambda config: config.model_copy(update={"duration_hours": 0}),
            "INVALID_FEEDBACK",
        ),
        (
            "random seed is missing",
            _same_card,
            lambda config: config.model_copy(update={"seed": None}),
            "INVALID_FEEDBACK",
        ),
    ],
    ids=lambda value: value if isinstance(value, str) else None,
)
def test_compiler_rejection_matrix_uses_stable_codes(
    case: str,
    mutate_card: CardMutation,
    mutate_config: ConfigMutation,
    expected_code: str,
) -> None:
    """Catches invalid evidence-to-scenario inputs escaping their stable error category."""
    del case
    card = mutate_card(make_threat_card())
    config = mutate_config(card.default_config)

    with pytest.raises(CompilerError) as error:
        compile_scenario(card, config)

    assert error.value.code == expected_code


def test_compiler_revalidates_copied_threat_identity() -> None:
    """Catches an invalid copied threat slug entering compiled provenance."""
    card = make_threat_card()
    invalid = card.model_copy(update={"threat_id": "NOT A SLUG"})

    with pytest.raises(CompilerError) as error:
        compile_scenario(invalid, invalid.default_config)

    assert error.value.code == "MISSING_EVIDENCE"


def test_compiler_round_trip_revalidates_copied_nested_evidence_identity() -> None:
    """Catches a copied nested evidence slug bypassing field-mapping revalidation."""
    card = make_threat_card()
    invalid_evidence = card.evidence[0].model_copy(update={"evidence_id": "NOT A SLUG"})
    invalid = card.model_copy(update={"evidence": [invalid_evidence]})

    with pytest.raises(CompilerError) as error:
        compile_scenario(invalid, invalid.default_config)

    assert error.value.code == "MISSING_EVIDENCE"


@pytest.mark.parametrize(
    ("field_name", "invalid_value", "expected_code"),
    [
        ("schema_version", "9.0.0", "INVALID_FEEDBACK"),
        ("version", "not-semver", "INVALID_FEEDBACK"),
        ("query_budget", 0, "INVALID_FEEDBACK"),
        ("rail", "cash", "UNSUPPORTED_RAIL"),
    ],
)
def test_compiler_translates_copied_invalid_scenario_fields(
    field_name: str, invalid_value: object, expected_code: str
) -> None:
    """Catches Pydantic validation failures leaking through the compiler API."""
    card = make_threat_card()
    invalid = card.default_config.model_copy(update={field_name: invalid_value})

    with pytest.raises(CompilerError) as error:
        compile_scenario(card, invalid)

    assert error.value.code == expected_code


def test_compiler_treats_malformed_url_syntax_as_missing_evidence() -> None:
    """Catches URL parser failures escaping instead of using the evidence error code."""
    card = make_threat_card()
    malformed_evidence = card.evidence[0].model_copy(
        update={"direct_source_url": "https://["}
    )
    invalid = card.model_copy(update={"evidence": [malformed_evidence]})

    with pytest.raises(CompilerError) as error:
        compile_scenario(invalid, invalid.default_config)

    assert error.value.code == "MISSING_EVIDENCE"


def test_compiler_emits_a_traceable_bounded_scenario_bundle() -> None:
    """Catches compilation that drops provenance, bounds, safety, or replay parameters."""
    card = make_threat_card()

    bundle = compile_scenario(card, card.default_config)

    assert bundle.scenario_id == "app-mule-personalized-v1"
    assert bundle.threat_card_ref == "app-personalized-mule@2"
    assert bundle.rail is Rail.A2A
    assert bundle.feedback == [
        FeedbackField.APPROVE,
        FeedbackField.CHALLENGE,
        FeedbackField.DECLINE,
        FeedbackField.REALIZED_VALUE,
    ]
    assert bundle.safety == {"export_level": "sanitized", "synthetic_only": True}
    assert bundle.benign_entity_count == 5000
    assert bundle.illicit_entity_count == 60
    assert bundle.duration_hours == 24
    assert bundle.seed == 260816
    assert bundle.campaign_stages == card.default_config.campaign_stages
    assert bundle.transition_rules == card.default_config.transition_rules
    assert bundle.defender_knowledge_boundary == card.defender_knowledge_boundary
    assert bundle.replay_manifest.scenario_id == bundle.scenario_id
    assert bundle.replay_manifest.scenario_version == bundle.version
    assert bundle.replay_manifest.threat_card_ref == bundle.threat_card_ref
    assert bundle.replay_manifest.random_seed == bundle.seed
