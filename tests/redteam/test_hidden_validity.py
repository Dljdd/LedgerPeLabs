"""Independent hidden-campaign and validity-oracle behavior."""

from __future__ import annotations

import ast
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

import pytest

from apar.contracts.events import EventKind
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


def test_hidden_generator_uses_the_domain_separated_numpy_stream() -> None:
    """Pin a hand-derived PCG64 leaf and schedule, independent of global RNG state."""
    events = HiddenCampaignGenerator().generate(
        "app_scam_mule", seed=260_816, count=8
    )
    source = Path("src/apar/evaluation_hidden/generator.py").read_text(encoding="utf-8")
    imports = _scan_imports(Path("src/apar/evaluation_hidden"), prefix="random")

    assert events[0].amount == Decimal("88.25")
    assert events[0].event_time == datetime(2026, 8, 16, 0, 22, 9, tzinfo=UTC)
    assert events[0].lineage["hidden_generator"] == "independent-numpy-v2"
    assert imports == []
    assert "np.random.Generator" in source


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


@pytest.mark.parametrize(
    "family",
    ("app_scam_mule", "card_testing_cnp", "synthetic_merchant_refund"),
)
def test_hidden_oracle_requires_independent_fee_and_account_evidence(
    family: str,
) -> None:
    """Catch principal-only conservation that ignores fees and ledger account roles."""
    events = HiddenCampaignGenerator().generate(family, seed=41, count=8)
    opening = events[0]
    assert {
        "fee_amount",
        "payer_account",
        "payee_account",
        "fee_account",
    } <= set(opening.rail_data)
    assert {
        "payer_opening_balance",
        "payee_opening_balance",
        "fee_opening_balance",
    } <= set(opening.party_refs)

    tampered = list(events)
    tampered[0] = opening.model_copy(
        update={"rail_data": {**opening.rail_data, "fee_amount": "-0.01"}}
    )

    assert HiddenValidityOracle().evaluate(tuple(tampered)).model_dump() == {
        "valid": False
    }


def test_hidden_card_oracle_covers_reversal_and_rejects_impossible_transition() -> None:
    """Pin authorization reversal and reject a reversal after clearing without raising."""
    events = HiddenCampaignGenerator().generate("card_testing_cnp", seed=43, count=8)
    assert EventKind.REVERSAL in {event.event_type for event in events}
    assert HiddenValidityOracle().evaluate(events).valid

    settlement_index = next(
        index
        for index, event in enumerate(events)
        if event.event_type is EventKind.SETTLEMENT
    )
    changed = list(events)
    changed[settlement_index] = changed[settlement_index].model_copy(
        update={"event_type": EventKind.REVERSAL}
    )

    assert not HiddenValidityOracle().evaluate(tuple(changed)).valid


def test_hidden_corpora_cover_refund_chargeback_recovery_return_and_freeze() -> None:
    """Pin every fee-bearing terminal lifecycle used by the independent ledger."""
    card_testing = HiddenCampaignGenerator().generate(
        "card_testing_cnp", seed=44, count=8
    )
    refunds = HiddenCampaignGenerator().generate(
        "synthetic_merchant_refund", seed=45, count=8
    )
    a2a = HiddenCampaignGenerator().generate("app_scam_mule", seed=46, count=8)
    kinds = {event.event_type for event in (*card_testing, *refunds, *a2a)}

    assert {
        EventKind.REFUND,
        EventKind.CHARGEBACK,
        EventKind.RECOVERY,
        EventKind.TRANSFER_RETURNED,
        EventKind.FUNDS_FROZEN,
    } <= kinds
    assert all(
        HiddenValidityOracle().evaluate(events).valid
        for events in (card_testing, refunds, a2a)
    )


def test_hidden_oracle_returns_false_for_out_of_order_negative_gaps() -> None:
    """Catch negative timing gaps reaching logarithms and escaping as exceptions."""
    events = HiddenCampaignGenerator().generate("app_scam_mule", seed=47, count=8)

    assert HiddenValidityOracle().evaluate(tuple(reversed(events))).model_dump() == {
        "valid": False
    }


def test_hidden_oracle_rejects_mid_lifecycle_fee_or_account_mutations() -> None:
    """Catch self-balancing but undeclared economics changing after authorization."""
    events = HiddenCampaignGenerator().generate(
        "synthetic_merchant_refund", seed=53, count=8
    )
    settlement_index = next(
        index
        for index, event in enumerate(events)
        if event.event_type is EventKind.SETTLEMENT
    )
    settlement = events[settlement_index]
    changed_fee = list(events)
    changed_fee[settlement_index] = settlement.model_copy(
        update={
            "rail_data": {
                **settlement.rail_data,
                "fee_amount": str(
                    Decimal(str(settlement.rail_data["fee_amount"])) + Decimal("0.01")
                ),
            }
        }
    )
    changed_account = list(events)
    changed_account[settlement_index] = settlement.model_copy(
        update={
            "rail_data": {
                **settlement.rail_data,
                "payee_account": "hidden:card:substituted-payee",
            }
        }
    )

    assert not HiddenValidityOracle().evaluate(tuple(changed_fee)).valid
    assert not HiddenValidityOracle().evaluate(tuple(changed_account)).valid


def test_hidden_generator_rejects_unbounded_inputs() -> None:
    """Catch undeclared families or resource-amplifying count values reaching generation."""
    generator = HiddenCampaignGenerator()

    with pytest.raises(ValueError, match="unsupported hidden campaign family"):
        generator.generate("unknown", seed=1, count=8)
    with pytest.raises(ValueError, match="count"):
        generator.generate("card_testing_cnp", seed=1, count=0)
    with pytest.raises(TypeError, match="seed"):
        generator.generate("card_testing_cnp", seed=True, count=8)
