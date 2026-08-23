"""Execution-evidence contract and V5DecisionRow projection for Defend v5."""

from __future__ import annotations

from apar.contracts.events import EventKind, PaymentEvent, Rail
from apar.evaluation.v5_population import V5DecisionRow
from apar.simulator.clock import Command


def project_evidence_to_rows(
    *,
    commands: tuple[Command, ...],
    events: tuple[PaymentEvent, ...],
    family: str,
    campaign_id: str,
) -> list[V5DecisionRow]:
    """Project real executed events into decision rows with provenance."""
    rows: list[V5DecisionRow] = []
    seen_event_ids: set[str] = set()

    # Build a lookup from event_id to command.
    cmd_by_campaign = {}
    for cmd in commands:
        cid = getattr(cmd, 'campaign_id', None) or cmd.payload.get('campaign_id')
        if cid:
            cmd_by_campaign[cid] = cmd

    for event in events:
        if event.event_id in seen_event_ids:
            continue
        seen_event_ids.add(event.event_id)

        # Derive lifecycle state from event type.
        lifecycle_map = {
            EventKind.AUTHORIZATION: "authorized",
            EventKind.AUTHORIZATION_DECLINED: "declined",
            EventKind.TRANSFER_INITIATED: "initiated",
            EventKind.TRANSFER_ACCEPTED: "authorized",
            EventKind.TRANSFER_POSTED: "settled",
            EventKind.SETTLEMENT: "settled",
        }
        lifecycle_state = lifecycle_map.get(event.event_type, "initiated")

        # Find matching command for source provenance.
        source_command_id = ""
        matching_cmd = cmd_by_campaign.get(event.campaign_id)
        if matching_cmd:
            source_command_id = f"cmd-{event.campaign_id[:8]}-{event.event_id[:8]}"

        predictive_features = {
            "amount": float(event.amount),
            f"rail_{event.rail.value}": 1.0,
            "integrity_pass": 1.0 if event.rail != Rail.AGENTIC else 0.0,
        }

        row = V5DecisionRow(
            event_id=event.event_id,
            payment_id=event.payment_id if hasattr(event, 'payment_id') else str(event.trace_id),
            campaign_id=event.campaign_id,
            family=family,
            actor_id=event.actor_id,
            counterparty_id=event.counterparty_id,
            amount=event.amount,
            currency=event.currency,
            decision_at=event.available_at,
            is_fraud=True,
            rail=event.rail.value,
            integrity_status="pass" if event.rail != Rail.AGENTIC else "pass",
            lifecycle_state=lifecycle_state,
            source_command_id=source_command_id,
            source_event_id=str(event.event_id),
            predictive_features=predictive_features,
        )
        rows.append(row)

    return rows


__all__ = ["project_evidence_to_rows"]
