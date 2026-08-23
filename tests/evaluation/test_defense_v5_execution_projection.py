"""Execution-evidence to V5DecisionRow projection tests."""

from __future__ import annotations

import pytest

from apar.contracts.events import Rail
from apar.evaluation.v5_execution import project_evidence_to_rows
from apar.evaluation.v5_population import V5DecisionRow
from tests.evaluation.test_defense_v5_execution_evidence import (
    FAMILIES_AND_RAILS,
    _execute_campaign,
)


class TestExecutionProjection:
    @pytest.mark.parametrize(("family", "rail"), FAMILIES_AND_RAILS)
    def test_projects_executed_events_to_decision_rows(self, family: str, rail: Rail) -> None:
        commands, events, engine, population, params, bundle = _execute_campaign(family, rail)
        rows = project_evidence_to_rows(
            commands=commands,
            events=events,
            family=family,
            campaign_id=params.campaign_id,
        )
        assert len(rows) > 0, f"no decision rows projected for {family}"
        for row in rows:
            assert isinstance(row, V5DecisionRow)
            assert row.source_event_id != "", "source_event_id must reference real event"
            assert row.amount > 0

    def test_row_amounts_match_event_amounts(self) -> None:
        family, rail = "card_testing_cnp", Rail.CARD
        commands, events, engine, population, params, bundle = _execute_campaign(family, rail)
        rows = project_evidence_to_rows(
            commands=commands, events=events,
            family=family, campaign_id=params.campaign_id,
        )
        event_amounts = {str(e.event_id): e for e in events}
        for row in rows[:3]:
            matching = [e for e in events if str(e.event_id) == row.source_event_id]
            if matching:
                assert abs(float(matching[0].amount) - float(row.amount)) < 0.01
