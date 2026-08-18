"""Task7-owned deterministic trust configuration for agentic candidate replay."""

from __future__ import annotations

import hashlib
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal
from typing import cast

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat

from apar.contracts.decisions import Action
from apar.contracts.events import PaymentEvent, Rail
from apar.contracts.scenarios import ScenarioBundle
from apar.generators import CampaignGenerator, CampaignParams, Population, campaign_bytes
from apar.simulator.engine import SimulationEngine
from apar.simulator.ledger import AccountReference, LedgerEntry
from apar.simulator.rails.agentic import AgenticPaymentCommand, AgenticRailAdapter
from apar.trust import (
    AgentMandate,
    AgentPaymentRequest,
    AuthenticationEvidence,
    AuthenticationOutcome,
    ReceiptOutcome,
    TrustVerifier,
)


@dataclass(frozen=True, slots=True)
class AgenticReplayResult:
    """Concrete commands, trust inputs, schedule, events, and ledger from one replay."""

    commands: tuple[AgenticPaymentCommand, ...]
    events: tuple[PaymentEvent, ...]
    entries: tuple[LedgerEntry, ...]
    command_sha256: str
    event_sha256: str
    approved_event_count: int


def _event_bytes(events: tuple[PaymentEvent, ...]) -> bytes:
    import json

    return json.dumps(
        [event.model_dump(mode="json", round_trip=True) for event in events],
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _base_mandate(requests: tuple[AgentPaymentRequest, ...]) -> AgentMandate:
    by_bytes = {request.mandate.canonical_bytes(): request.mandate for request in requests}
    counts = Counter(request.mandate.canonical_bytes() for request in requests)
    winner = min(counts, key=lambda raw: (-counts[raw], raw))
    return by_bytes[winner]


def _base_identity(requests: tuple[AgentPaymentRequest, ...]) -> tuple[str, str]:
    counts = Counter((request.agent_id, request.key_id) for request in requests)
    return min(counts, key=lambda value: (-counts[value], value))


def _schedule(
    bundle: ScenarioBundle,
    requests: tuple[AgentPaymentRequest, ...],
) -> tuple[datetime, ...]:
    cursor = bundle.replay_manifest.simulation_start
    scheduled = []
    for request in requests:
        cursor = max(cursor + timedelta(seconds=1), request.created_at + timedelta(seconds=1))
        scheduled.append(cursor)
    return tuple(scheduled)


def _reissue(
    *,
    bundle: ScenarioBundle,
    seed: int,
    commands: tuple[object, ...],
) -> tuple[
    tuple[AgenticPaymentCommand, ...],
    tuple[datetime, ...],
    bytes,
    AgentMandate,
    tuple[AuthenticationEvidence, ...],
    tuple[str, str],
]:
    if not commands or any(type(command) is not AgenticPaymentCommand for command in commands):
        raise TypeError("agentic replay requires exact public AgenticPaymentCommand values")
    typed_commands = cast(tuple[AgenticPaymentCommand, ...], commands)
    requests = tuple(command.request for command in typed_commands)
    mandate = _base_mandate(requests)
    identity = _base_identity(requests)
    schedule = _schedule(bundle, requests)
    private_seed = hashlib.sha256(
        f"apar-task7-agentic-replay-v1:{seed}:{requests[0].campaign_id}".encode()
    ).digest()
    private_key = Ed25519PrivateKey.from_private_bytes(private_seed)
    public_key = private_key.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw)

    evidence_by_ref: dict[str, AuthenticationEvidence] = {}
    for request, timestamp in zip(requests, schedule, strict=True):
        reference = request.authentication_evidence_ref
        if reference is None or "missing" in reference or reference in evidence_by_ref:
            continue
        evidence_by_ref[reference] = AuthenticationEvidence(
            evidence_id=reference,
            agent_id=request.agent_id,
            user_ref=mandate.user_ref,
            mandate_id=mandate.mandate_id,
            nonce=request.nonce,
            payment_intent_hash=request.payment_intent_hash,
            request_id=request.request_id,
            outcome=AuthenticationOutcome.STEP_UP_VERIFIED,
            issued_at=timestamp - timedelta(seconds=1),
            expires_at=timestamp + timedelta(seconds=30),
        )
    evidence = tuple(evidence_by_ref[key] for key in sorted(evidence_by_ref))

    chain = TrustVerifier(
        registered_agents={identity: public_key},
        mandates={mandate.mandate_id: mandate},
        authentication_evidence={item.evidence_id: item for item in evidence},
    )
    reissued: list[AgenticPaymentCommand] = []
    previous_receipt = ""
    for command, timestamp in zip(typed_commands, schedule, strict=True):
        original = command.request
        prior = "f" * 64 if original.prior_receipt_hash == "f" * 64 else previous_receipt
        unsigned = original.model_copy(update={"prior_receipt_hash": prior, "signature": b""})
        signature = (
            bytes(64)
            if original.signature == bytes(64)
            else private_key.sign(unsigned.signing_bytes())
        )
        request = unsigned.model_copy(update={"signature": signature})
        reissued.append(
            AgenticPaymentCommand(
                request,
                payer_account=cast(str, command.payload["payer_account"]),
                payee_account=cast(str, command.payload["payee_account"]),
            )
        )
        preview = chain.preview(request, timestamp)
        if preview.allowed:
            receipt = chain.commit(request, preview, ReceiptOutcome.APPROVE, timestamp)
            previous_receipt = receipt.receipt_hash
    return tuple(reissued), schedule, public_key, mandate, evidence, identity


def replay_agentic_candidate(
    *,
    bundle: ScenarioBundle,
    population: Population,
    params: CampaignParams,
    seed: int,
) -> AgenticReplayResult:
    """Generate and replay one candidate under the same deterministic public trust inputs."""
    if type(bundle) is not ScenarioBundle or bundle.rail is not Rail.AGENTIC:
        raise TypeError("agentic replay requires an exact agentic ScenarioBundle")
    if type(population) is not Population or type(params) is not CampaignParams:
        raise TypeError("agentic replay requires exact public generation inputs")
    generated = CampaignGenerator(seed=seed).generate(
        "agentic_intent_abuse", population, params
    )
    commands, schedule, public_key, mandate, evidence, identity = _reissue(
        bundle=bundle,
        seed=seed,
        commands=cast(tuple[object, ...], generated),
    )

    def factory() -> AgenticRailAdapter:
        verifier = TrustVerifier(
            registered_agents={identity: public_key},
            mandates={mandate.mandate_id: mandate},
            authentication_evidence={item.evidence_id: item for item in evidence},
        )
        return AgenticRailAdapter(verifier, lambda _request, _receipt: Action.APPROVE)

    engine = SimulationEngine(
        bundle,
        {Rail.AGENTIC: factory},
        opening_balances=cast(
            dict[AccountReference, Decimal], dict(population.opening_balances)
        ),
    )
    for priority, (timestamp, command) in enumerate(zip(schedule, commands, strict=True)):
        engine.schedule(timestamp, priority, command)
    events = engine.run()
    engine.ledger.assert_conserved()
    return AgenticReplayResult(
        commands=commands,
        events=events,
        entries=engine.ledger.entries,
        command_sha256=hashlib.sha256(campaign_bytes(commands)).hexdigest(),
        event_sha256=hashlib.sha256(_event_bytes(events)).hexdigest(),
        approved_event_count=sum(event.event_type.value == "authorization" for event in events),
    )


__all__ = ["AgenticReplayResult", "replay_agentic_candidate"]
