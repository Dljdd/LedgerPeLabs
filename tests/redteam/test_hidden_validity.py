"""Independent hidden-campaign and validity-oracle behavior."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from apar.evaluation_hidden import HiddenCampaignGenerator, HiddenValidityOracle


def _scan_imports(root: Path, *, prefix: str) -> list[str]:
    findings: list[str] = []
    for path in sorted(root.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            names: tuple[str, ...]
            if isinstance(node, ast.Import):
                names = tuple(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                names = (node.module or "",)
            else:
                continue
            findings.extend(
                name
                for name in names
                if name == prefix or name.startswith(f"{prefix}.")
            )
    return findings


def test_hidden_package_does_not_import_defender() -> None:
    """Catch the hidden evaluator sharing production defender implementation."""
    assert _scan_imports(Path("src/apar/evaluation_hidden"), prefix="apar.defense") == []


def test_hidden_generator_does_not_import_main_generator() -> None:
    """Catch a circular hidden benchmark implemented with the public generator."""
    assert _scan_imports(Path("src/apar/evaluation_hidden"), prefix="apar.generators") == []


@pytest.mark.parametrize(
    ("family", "expected_rail"),
    (
        ("agentic_intent_abuse", "agentic"),
        ("app_scam_mule", "a2a"),
        ("card_testing_cnp", "card"),
        ("synthetic_merchant_refund", "card"),
    ),
)
def test_independent_hidden_families_are_deterministic_and_valid(
    family: str,
    expected_rail: str,
) -> None:
    """Catch missing motifs, unstable leaves, or an oracle rejecting its independent corpus."""
    first = HiddenCampaignGenerator().generate(family, seed=260_816, count=8)
    second = HiddenCampaignGenerator().generate(family, seed=260_816, count=8)

    assert first == second
    assert len(first) >= 8
    assert {event.rail.value for event in first} == {expected_rail}
    assert HiddenValidityOracle().evaluate(first).model_dump() == {"valid": True}


def test_hidden_oracle_returns_only_boolean_to_the_attacker_path() -> None:
    """Catch detailed hidden rejection reasons leaking through the nominal result."""
    events = HiddenCampaignGenerator().generate("app_scam_mule", seed=11, count=8)
    result = HiddenValidityOracle().evaluate(events[1:])

    assert result.model_dump() == {"valid": False}
    assert not hasattr(result, "reasons")
    assert not hasattr(result, "metrics")


def test_hidden_oracle_rejects_a_disconnected_campaign() -> None:
    """Catch a validity oracle that checks rows but not campaign graph connectivity."""
    events = HiddenCampaignGenerator().generate("card_testing_cnp", seed=29, count=8)
    last_payment = events[-1].rail_data["payment_id"]
    isolated = tuple(
        event.model_copy(
            update={
                "actor_id": "00000000-0000-4000-8000-000000000091",
                "counterparty_id": "00000000-0000-4000-8000-000000000092",
            }
        )
        if event.rail_data["payment_id"] == last_payment
        else event
        for event in events
    )

    assert not HiddenValidityOracle().evaluate(isolated).valid


def test_hidden_oracle_rejects_conflicting_opening_balance_evidence() -> None:
    """Catch per-event opening declarations changing after an entity was first observed."""
    events = HiddenCampaignGenerator().generate("card_testing_cnp", seed=31, count=8)
    changed = list(events)
    second_payment = changed[1]
    changed[1] = second_payment.model_copy(
        update={
            "party_refs": {
                **second_payment.party_refs,
                "actor_opening_balance": "2499.00",
            }
        }
    )

    assert not HiddenValidityOracle().evaluate(tuple(changed)).valid


def test_hidden_generator_rejects_unbounded_inputs() -> None:
    """Catch undeclared families or resource-amplifying count values reaching generation."""
    generator = HiddenCampaignGenerator()

    with pytest.raises(ValueError, match="unsupported hidden campaign family"):
        generator.generate("unknown", seed=1, count=8)
    with pytest.raises(ValueError, match="count"):
        generator.generate("card_testing_cnp", seed=1, count=0)
    with pytest.raises(TypeError, match="seed"):
        generator.generate("card_testing_cnp", seed=True, count=8)
