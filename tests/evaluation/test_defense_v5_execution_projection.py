"""Fail-closed execution-evidence projection tests."""

from __future__ import annotations

from dataclasses import replace
from decimal import Decimal

import pytest

from apar.contracts.decisions import ReasonCode
from apar.contracts.events import Rail
from apar.evaluation.v5_execution import (
    V5ExecutionEvidence,
    build_execution_evidence,
    project_execution_evidence,
)
from apar.evaluation.v5_population import V5DecisionRow
from apar.simulator.ledger import LedgerEntry
from tests.evaluation.test_defense_v5_execution_evidence import (
    FAMILIES_AND_RAILS,
    _execute_campaign,
)


def _evidence(family: str, rail: Rail) -> V5ExecutionEvidence:
    commands, events, engine, population, params, bundle, campaign_evidence = (
        _execute_campaign(family, rail)
    )
    return build_execution_evidence(
        family=family,
        commands=commands,
        campaign_evidence=campaign_evidence,
        events=events,
        ledger_entries=engine.ledger.entries,
        opening_balances=population.opening_balances,
    )


class TestExecutionProjection:
    @pytest.mark.parametrize(("family", "rail"), FAMILIES_AND_RAILS)
    def test_projects_executed_events_to_decision_rows(self, family: str, rail: Rail) -> None:
        evidence = _evidence(family, rail)
        rows = project_execution_evidence(evidence)

        assert len(rows) > 0, f"no decision rows projected for {family}"
        assert len(rows) == len(evidence.events) == len(evidence.lineage)
        for row, lineage in zip(rows, evidence.lineage, strict=True):
            assert isinstance(row, V5DecisionRow)
            assert row.source_command_id == lineage.command_id
            assert row.source_event_id == lineage.event_id
            assert row.execution_evidence_sha256 == evidence.evidence_sha256
            assert row.amount > 0

    def test_row_amounts_match_event_amounts(self) -> None:
        evidence = _evidence("card_testing_cnp", Rail.CARD)
        rows = project_execution_evidence(evidence)
        events_by_id = {event.event_id: event for event in evidence.events}

        for row in rows:
            assert row.amount == events_by_id[row.source_event_id].amount

    @pytest.mark.parametrize("field", ["campaign_id", "payment_id"])
    def test_rejects_command_event_identity_mismatch(self, field: str) -> None:
        commands, events, engine, population, params, bundle, campaign_evidence = (
            _execute_campaign("card_testing_cnp", Rail.CARD)
        )
        first = events[0]
        if field == "campaign_id":
            tampered = first.model_copy(
                update={"campaign_id": "00000000-0000-4000-8000-000000000999"}
            )
        else:
            rail_data = dict(first.rail_data)
            rail_data["payment_id"] = "tampered-payment"
            tampered = first.model_copy(update={"rail_data": rail_data})

        with pytest.raises(ValueError, match=field):
            build_execution_evidence(
                family="card_testing_cnp",
                commands=commands,
                campaign_evidence=campaign_evidence,
                events=(tampered, *events[1:]),
                ledger_entries=engine.ledger.entries,
                opening_balances=population.opening_balances,
            )

    def test_rejects_removed_or_out_of_order_lifecycle_event(self) -> None:
        commands, events, engine, population, params, bundle, campaign_evidence = (
            _execute_campaign("synthetic_merchant_refund", Rail.CARD)
        )
        shared = {
            "family": "synthetic_merchant_refund",
            "commands": commands,
            "campaign_evidence": campaign_evidence,
            "ledger_entries": engine.ledger.entries,
            "opening_balances": population.opening_balances,
        }

        with pytest.raises(ValueError, match="lifecycle"):
            build_execution_evidence(events=events[:-1], **shared)
        with pytest.raises(ValueError, match="order|scheduled|lifecycle"):
            build_execution_evidence(
                events=(events[1], events[0], *events[2:]),
                **shared,
            )

    def test_rejects_event_id_and_source_command_id_tampering(self) -> None:
        commands, events, engine, population, params, bundle, campaign_evidence = (
            _execute_campaign("app_scam_mule", Rail.A2A)
        )
        tampered_event = events[0].model_copy(
            update={"event_id": "00000000-0000-4000-8000-000000000998"}
        )
        with pytest.raises(ValueError, match="event|ledger|lineage"):
            build_execution_evidence(
                family="app_scam_mule",
                commands=commands,
                campaign_evidence=campaign_evidence,
                events=(tampered_event, *events[1:]),
                ledger_entries=engine.ledger.entries,
                opening_balances=population.opening_balances,
            )

        evidence = _evidence("app_scam_mule", Rail.A2A)
        bad_lineage = replace(evidence.lineage[0], command_id="sha256:" + "0" * 64)
        with pytest.raises(ValueError, match="command_id"):
            replace(evidence, lineage=(bad_lineage, *evidence.lineage[1:]))

    def test_rejects_ledger_posting_mismatch(self) -> None:
        commands, events, engine, population, params, bundle, campaign_evidence = (
            _execute_campaign("card_testing_cnp", Rail.CARD)
        )
        entries = engine.ledger.entries
        first = entries[0]
        amount = sum(first.debit.values(), Decimal("0")) + Decimal("1.00")
        bad_entry = LedgerEntry(
            first.entry_id,
            {next(iter(first.debit)): amount},
            {next(iter(first.credit)): amount},
            first.currency,
        )

        with pytest.raises(ValueError, match="ledger"):
            build_execution_evidence(
                family="card_testing_cnp",
                commands=commands,
                campaign_evidence=campaign_evidence,
                events=events,
                ledger_entries=(bad_entry, *entries[1:]),
                opening_balances=population.opening_balances,
            )

    def test_rejects_opening_ledger_evidence_tampering(self) -> None:
        evidence = _evidence("card_testing_cnp", Rail.CARD)
        account, amount = evidence.opening_balances[0]
        tampered_opening = (
            (account, amount + Decimal("1.00")),
            *evidence.opening_balances[1:],
        )

        with pytest.raises(ValueError, match="digest|ledger"):
            replace(evidence, opening_balances=tampered_opening)

    def test_rejects_duplicate_opening_ledger_accounts(self) -> None:
        evidence = _evidence("app_scam_mule", Rail.A2A)

        with pytest.raises(ValueError, match="opening.*unique|canonical"):
            replace(
                evidence,
                opening_balances=(
                    *evidence.opening_balances,
                    evidence.opening_balances[0],
                ),
            )

    @pytest.mark.parametrize(
        "reason",
        [
            ReasonCode.SIGNATURE_INVALID,
            ReasonCode.MANDATE_SCOPE_VIOLATION,
            ReasonCode.AUTHENTICATION_EVIDENCE_REPLAY,
        ],
    )
    def test_rejects_tampered_invalid_agentic_verdict(self, reason: ReasonCode) -> None:
        commands, events, engine, population, params, bundle, campaign_evidence = (
            _execute_campaign("agentic_intent_abuse", Rail.AGENTIC)
        )
        index = next(
            i
            for i, event in enumerate(events)
            if event.rail_data.get("reason_code") == reason.value
        )
        rail_data = dict(events[index].rail_data)
        rail_data["integrity"] = "pass"
        tampered = events[index].model_copy(update={"rail_data": rail_data})

        with pytest.raises(ValueError, match="verifier|integrity"):
            build_execution_evidence(
                family="agentic_intent_abuse",
                commands=commands,
                campaign_evidence=campaign_evidence,
                events=(*events[:index], tampered, *events[index + 1 :]),
                ledger_entries=engine.ledger.entries,
                opening_balances=population.opening_balances,
            )
