"""Mixed population builder with real executed evidence for Sentinel v5."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from types import MappingProxyType, SimpleNamespace
from typing import Any, Self, cast
from uuid import NAMESPACE_URL, uuid5

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from apar.contracts.decisions import Action
from apar.contracts.events import EventKind, PaymentEvent, Rail
from apar.contracts.scenarios import (
    AttackerMode,
    CampaignStage,
    FeedbackField,
    ReplayManifest,
    ReplayOrdering,
    ScenarioBundle,
    StageTransition,
)
from apar.evaluation.v5_protocol import V5DevelopmentProtocol, V5Family, V5Profile
from apar.generators.campaigns import (
    AGENTIC_INTENT_ABUSE_MOTIF,
    APP_SCAM_MULE_MOTIF,
    CARD_TESTING_CNP_MOTIF,
    SYNTHETIC_MERCHANT_REFUND_MOTIF,
    AgenticFixture,
    CampaignEvidence,
    CampaignParams,
    _CampaignEvaluator,
)
from apar.generators.population import Population, PopulationEntity, PopulationGenerator
from apar.simulator.clock import Command
from apar.simulator.engine import SimulationEngine
from apar.simulator.ledger import AccountReference, Ledger, LedgerEntry
from apar.simulator.rails import (
    A2ARailAdapter,
    AgenticRailAdapter,
    CardRailAdapter,
)
from apar.simulator.rails.a2a import (
    AcceptA2A,
    FreezeA2AFunds,
    InitiateA2A,
    PostA2A,
    RecoverA2A,
    RejectA2A,
    ReportA2AFraud,
    ReturnA2A,
)
from apar.simulator.rails.base import AdapterFactory
from apar.simulator.rails.card import (
    AuthorizeCard,
    ChargebackCard,
    ClearCard,
    DeclineCardAuthorization,
    OpenCardDispute,
    RecoverCard,
    RefundCard,
    ReportCardFraud,
    ReverseCardAuthorization,
    SettleCard,
)
from apar.trust.verifier import TrustVerifier


class V5DecisionRow(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    event_id: str
    payment_id: str
    campaign_id: str
    family: str
    actor_id: str
    counterparty_id: str
    amount: Decimal
    currency: str = "USD"
    decision_at: datetime
    is_fraud: bool
    rail: str = "card"
    integrity_status: str = "not_applicable"
    lifecycle_state: str = ""
    source_command_id: str = ""
    source_event_id: str = ""
    execution_evidence_sha256: str = ""
    predictive_features: dict[str, float] = Field(default_factory=dict)

    @field_validator("decision_at", mode="before")
    @classmethod
    def require_utc(cls, value: object) -> object:
        if isinstance(value, datetime) and value.tzinfo is None:
            raise ValueError("decision_at must be UTC")
        return value

    @model_validator(mode="after")
    def validate_row(self) -> Self:
        if type(self.is_fraud) is not bool:
            raise TypeError("is_fraud must be an exact boolean")
        if not self.amount.is_finite() or self.amount <= 0:
            raise ValueError("decision amount must be finite and positive")
        if self.rail not in {rail.value for rail in Rail}:
            raise ValueError("decision rail is unknown")
        if self.integrity_status not in {"pass", "fail", "not_applicable"}:
            raise ValueError("decision integrity status is unknown")
        if self.rail == Rail.AGENTIC.value:
            if self.integrity_status == "not_applicable":
                raise ValueError("agentic row must contain a verifier result")
        elif self.integrity_status != "not_applicable":
            raise ValueError("non-agentic row cannot claim a verifier result")
        if any(not math.isfinite(value) for value in self.predictive_features.values()):
            raise ValueError("predictive features must be finite")
        return self


class V5LineageManifest(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    command_id: str
    command_type: str
    command_name: str
    event_id: str
    payment_id: str
    actor_id: str
    counterparty_id: str
    lifecycle_position: int = Field(ge=0)
    is_fraud: bool
    command_payload_json: str
    trace_id: str
    scheduled_at: datetime


class V5EventRecord(BaseModel):
    """Bounded canonical event facts retained without a live simulator object."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    event_id: str
    payment_id: str
    event_type: str
    amount: Decimal
    currency: str
    decision_at: datetime
    rail_data_json: str
    lineage_json: str
    event_json: str


class V5LedgerPosting(BaseModel):
    """Canonical double-entry posting retained for independent economic replay."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    entry_id: str
    debit: tuple[tuple[str, Decimal], ...]
    credit: tuple[tuple[str, Decimal], ...]
    currency: str


class V5TrustRecord(BaseModel):
    """Bounded public verifier inputs and verdict for a projected agentic event."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    command_id: str
    event_id: str
    request_id: str
    agent_id: str
    key_id: str
    mandate_id: str
    authentication_evidence_id: str | None = None
    request_json: str
    mandate_json: str
    authentication_evidence_json: str | None = None
    public_key_hex: str
    receipt_hash: str
    request_hash: str
    signature_hash: str
    allowed: bool
    reason_code: str | None = None
    outcome: str


class V5TrustRegistry(BaseModel):
    """Canonical public verifier registry needed for independent replay."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    agent_id: str
    key_id: str
    public_key_hex: str
    mandate_json: str
    authentication_evidence_json: tuple[str, ...]


class V5ExecutionManifest(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    evidence_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    artifact_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    campaign_id: str
    family: str
    rail: str
    lineage: tuple[V5LineageManifest, ...]
    ledger_entry_ids: tuple[str, ...]
    trust_request_ids: tuple[str, ...]
    trust_failure_event_ids: tuple[str, ...]
    account_ids: tuple[str, ...]
    opening_balances: tuple[tuple[str, Decimal], ...]
    device_ids: tuple[str, ...]
    credential_ids: tuple[str, ...]
    merchant_ids: tuple[str, ...]
    payee_ids: tuple[str, ...]
    agent_ids: tuple[str, ...]
    key_ids: tuple[str, ...]
    mandate_ids: tuple[str, ...]
    authentication_evidence_ids: tuple[str, ...]
    event_records: tuple[V5EventRecord, ...]
    ledger_postings: tuple[V5LedgerPosting, ...]
    trust_records: tuple[V5TrustRecord, ...]
    trust_registry: V5TrustRegistry | None = None

    def artifact_digest(self) -> str:
        """Digest all retained immutable facts so serialized artifacts cannot drift."""
        document = self.model_dump(mode="json")
        document.pop("artifact_sha256", None)
        return hashlib.sha256(
            json.dumps(document, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        ).hexdigest()

    @model_validator(mode="after")
    def validate_manifest(self) -> Self:
        if self.family not in {family.value for family in V5Family} | {"legitimate"}:
            raise ValueError("execution manifest family is unknown")
        if self.rail not in {rail.value for rail in Rail}:
            raise ValueError("execution manifest rail is unknown")
        if not self.lineage:
            raise ValueError("execution manifest lineage must not be empty")
        event_ids = tuple(item.event_id for item in self.lineage)
        command_ids = tuple(item.command_id for item in self.lineage)
        from apar.evaluation.v5_execution import _command_id_from_facts

        expected_command_ids = tuple(
            _command_id_from_facts(
                command_type=item.command_type,
                command_name=item.command_name,
                command_payload=cast(
                    Mapping[str, object],
                    _decode_canonical_json(item.command_payload_json),
                ),
            )
            for item in self.lineage
        )
        if command_ids != expected_command_ids:
            raise ValueError("execution manifest canonical command ID disagrees with payload")
        if len(event_ids) != len(set(event_ids)):
            raise ValueError("execution manifest event IDs must be unique")
        if len(command_ids) != len(set(command_ids)):
            raise ValueError("execution manifest command IDs must be unique")
        if tuple(record.event_id for record in self.event_records) != event_ids:
            raise ValueError("execution manifest canonical event facts disagree with lineage")
        if any(
            event.payment_id != link.payment_id
            or json.loads(event.rail_data_json).get("payment_id") != link.payment_id
            for event, link in zip(self.event_records, self.lineage, strict=True)
        ):
            raise ValueError("execution manifest event payment lineage disagrees")
        self._validate_manifest_event_facts()
        if tuple(posting.entry_id for posting in self.ledger_postings) != self.ledger_entry_ids:
            raise ValueError("execution manifest ledger posting IDs disagree with ledger index")
        if tuple(record.request_id for record in self.trust_records) != self.trust_request_ids:
            raise ValueError("execution manifest verifier request IDs disagree with trust index")
        if tuple(
            record.event_id for record in self.trust_records if not record.allowed
        ) != self.trust_failure_event_ids:
            raise ValueError("execution manifest verifier failures disagree with trust index")
        if tuple(sorted(account for account, _amount in self.opening_balances)) != tuple(
            account for account, _amount in self.opening_balances
        ):
            raise ValueError("opening balances must be canonically ordered")
        if len({account for account, _amount in self.opening_balances}) != len(
            self.opening_balances
        ):
            raise ValueError("opening balance accounts must be unique")
        private_opening_accounts = tuple(
            account
            for account, _amount in self.opening_balances
            if account.startswith("acct:")
        )
        if private_opening_accounts != self.account_ids:
            raise ValueError("execution manifest account domain disagrees with opening balances")
        for domain_name in (
            "device_ids",
            "credential_ids",
            "merchant_ids",
            "payee_ids",
            "agent_ids",
            "key_ids",
            "mandate_ids",
            "authentication_evidence_ids",
        ):
            domain_values = getattr(self, domain_name)
            if domain_values != tuple(sorted(set(domain_values))):
                raise ValueError(f"{domain_name} must be unique and canonical")
        if any(
            tuple(sorted(posting.debit)) != posting.debit
            or tuple(sorted(posting.credit)) != posting.credit
            for posting in self.ledger_postings
        ):
            raise ValueError("ledger postings must use canonical account order")
        self._validate_manifest_ledger()
        trust_event_ids = tuple(record.event_id for record in self.trust_records)
        if self.rail == Rail.AGENTIC.value:
            if trust_event_ids != event_ids:
                raise ValueError("agentic manifest must retain one verifier record per event")
            if self.trust_registry is None:
                raise ValueError("agentic manifest must retain verifier registry facts")
            lineage_by_event = {link.event_id: link for link in self.lineage}
            if any(
                record.command_id != lineage_by_event[record.event_id].command_id
                for record in self.trust_records
            ):
                raise ValueError("agentic verifier command lineage disagrees with event")
            if any(
                record.allowed
                and (
                    record.agent_id != self.trust_registry.agent_id
                    or record.key_id != self.trust_registry.key_id
                    or record.mandate_id
                    != json.loads(self.trust_registry.mandate_json)["mandate_id"]
                )
                for record in self.trust_records
            ):
                raise ValueError("allowed verifier records disagree with registry facts")
            expected_domains = {
                "agent_ids": tuple(sorted({record.agent_id for record in self.trust_records})),
                "key_ids": tuple(sorted({record.key_id for record in self.trust_records})),
                "mandate_ids": tuple(sorted({record.mandate_id for record in self.trust_records})),
                "authentication_evidence_ids": tuple(
                    sorted(
                        {
                            record.authentication_evidence_id
                            for record in self.trust_records
                            if record.authentication_evidence_id is not None
                        }
                    )
                ),
            }
            if any(getattr(self, name) != values for name, values in expected_domains.items()):
                raise ValueError("agentic identity domains disagree with verifier records")
        elif self.trust_records:
            raise ValueError("non-agentic manifest cannot retain verifier records")
        elif self.trust_registry is not None:
            raise ValueError("non-agentic manifest cannot retain verifier registry facts")
        elif any(
            getattr(self, domain)
            for domain in (
                "agent_ids",
                "key_ids",
                "mandate_ids",
                "authentication_evidence_ids",
            )
        ):
            raise ValueError("non-agentic manifest cannot retain verifier identity domains")
        if not set(self.trust_failure_event_ids) <= set(event_ids):
            raise ValueError("trust failures must reference manifest events")
        self._validate_manifest_trust()
        if self.evidence_sha256 != self.evidence_digest():
            raise ValueError("execution manifest evidence digest disagrees with retained facts")
        if self.artifact_sha256 != self.artifact_digest():
            raise ValueError("execution manifest artifact digest was tampered")
        return self

    def evidence_digest(self) -> str:
        """Recompute the execution-evidence digest solely from retained raw facts."""
        from apar.evaluation.v5_execution import _command_id_from_facts

        document = {
            "domain": "apar.sentinel-v5.execution-evidence.v1",
            "family": self.family,
            "campaign_id": self.campaign_id,
            "rail": self.rail,
            "lineage": [
                {
                    "command_id": _command_id_from_facts(
                        command_type=link.command_type,
                        command_name=link.command_name,
                        command_payload=cast(
                            Mapping[str, object],
                            _decode_canonical_json(link.command_payload_json),
                        ),
                    ),
                    "command_name": link.command_name,
                    "event_id": link.event_id,
                    "campaign_id": self.campaign_id,
                    "payment_id": link.payment_id,
                    "actor_id": link.actor_id,
                    "counterparty_id": link.counterparty_id,
                    "rail": self.rail,
                    "scheduled_at": link.scheduled_at.isoformat().replace("+00:00", "Z"),
                    "lifecycle_position": link.lifecycle_position,
                    "is_fraud": link.is_fraud,
                }
                for link in self.lineage
            ],
            "events": [json.loads(record.event_json) for record in self.event_records],
            "ledger_entries": [
                {
                    "entry_id": posting.entry_id,
                    "debit": {account: str(amount) for account, amount in posting.debit},
                    "credit": {account: str(amount) for account, amount in posting.credit},
                    "currency": posting.currency,
                }
                for posting in self.ledger_postings
            ],
            "opening_balances": [
                {"account": account, "amount": str(amount)}
                for account, amount in self.opening_balances
            ],
            "trust": [
                {
                    "command_id": record.command_id,
                    "event_id": record.event_id,
                    "request_id": record.request_id,
                    "authentication_evidence_id": (
                        record.authentication_evidence_id
                        if record.authentication_evidence_json is not None
                        else None
                    ),
                    "authentication_evidence": (
                        json.loads(record.authentication_evidence_json)
                        if record.authentication_evidence_json is not None
                        else None
                    ),
                    "receipt_hash": record.receipt_hash,
                    "allowed": record.allowed,
                    "reason_code": record.reason_code,
                    "outcome": record.outcome,
                }
                for record in self.trust_records
            ],
        }
        return hashlib.sha256(
            json.dumps(document, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        ).hexdigest()

    def _validate_manifest_event_facts(self) -> None:
        """Cross-bind every serialized event to its command, lifecycle, and raw event."""
        from apar.evaluation.v5_execution import _COMMAND_EVENT_KINDS, _expected_kind

        opening_states: dict[str, dict[str, object]] = {}
        previous_events: dict[str, str] = {}
        trust_by_event = {record.event_id: record for record in self.trust_records}
        for link, event in zip(self.lineage, self.event_records, strict=True):
            payload = cast(dict[str, object], _decode_canonical_json(link.command_payload_json))
            raw_event = cast(dict[str, object], json.loads(event.event_json))
            canonical_decision_at = event.decision_at.isoformat().replace("+00:00", "Z")
            canonical_scheduled_at = link.scheduled_at.isoformat().replace("+00:00", "Z")
            if (
                raw_event.get("decision_at") != canonical_decision_at
                or raw_event.get("available_at") != canonical_decision_at
            ):
                raise ValueError("event decision timestamp disagrees with retained raw event")
            if raw_event.get("event_time") != canonical_scheduled_at:
                raise ValueError("event scheduled timestamp disagrees with retained raw event")
            if (
                raw_event.get("event_id") != event.event_id
                or raw_event.get("campaign_id") != self.campaign_id
                or raw_event.get("rail") != self.rail
                or raw_event.get("event_type") != event.event_type
                or raw_event.get("amount") != str(event.amount)
                or raw_event.get("currency") != event.currency
                or raw_event.get("rail_data") != json.loads(event.rail_data_json)
                or raw_event.get("lineage") != json.loads(event.lineage_json)
            ):
                raise ValueError("event facts disagree with retained raw event")
            if payload.get("payment_id") != link.payment_id:
                raise ValueError("event facts disagree with command payment")
            if link.command_name in {
                "a2a.initiate",
                "card.authorize",
                "card.decline",
                "agentic.pay",
            }:
                opening_states[link.payment_id] = payload
            state = opening_states.get(link.payment_id)
            if state is None:
                raise ValueError("event facts lack an opening command state")
            if (
                event.amount != state.get("amount")
                or event.currency != state.get("currency")
                or link.actor_id != state.get("actor_id")
                or link.counterparty_id != state.get("counterparty_id")
                or link.trace_id != state.get("trace_id")
            ):
                raise ValueError("event facts disagree with command state")
            expected_previous = previous_events.get(link.payment_id, "")
            if json.loads(event.lineage_json).get("previous_event_id", "") != expected_previous:
                raise ValueError("event facts disagree with lifecycle lineage")
            previous_events[link.payment_id] = event.event_id
            if link.command_name == "agentic.pay":
                record = trust_by_event.get(event.event_id)
                if record is None:
                    raise ValueError("agentic event facts lack a verifier record")
                request = _agent_request_from_json(record.request_json)
                request_fields = {
                    "request_id": request.request_id,
                    "payment_id": request.payment_id,
                    "amount": request.amount,
                    "currency": request.currency,
                    "campaign_id": request.campaign_id,
                    "trace_id": request.trace_id,
                    "actor_id": request.actor_id,
                    "counterparty_id": request.counterparty_id,
                    "nonce": request.nonce,
                    "consent_ref": request.consent_ref,
                    "cart_hash": request.cart_hash,
                    "payment_intent_hash": request.payment_intent_hash,
                    "signature": request.signature,
                }
                if any(payload.get(name) != value for name, value in request_fields.items()):
                    raise ValueError("agentic command facts disagree with signed request")
                if (
                    event.amount != request.amount
                    or event.currency != request.currency
                    or link.payment_id != request.payment_id
                    or link.actor_id != request.actor_id
                    or link.counterparty_id != request.counterparty_id
                    or link.trace_id != request.trace_id
                    or json.loads(event.rail_data_json).get("request_id") != request.request_id
                ):
                    raise ValueError("event facts disagree with signed request")
                expected_kind = _expected_kind(
                    cast(Any, SimpleNamespace(name=link.command_name)),
                    cast(
                        Any,
                        SimpleNamespace(
                            rail_data=json.loads(event.rail_data_json),
                        ),
                    ),
                )
            else:
                try:
                    expected_kind = _COMMAND_EVENT_KINDS[link.command_name]
                except KeyError as error:
                    raise ValueError("event facts have an unknown command lifecycle") from error
            if event.event_type != expected_kind.value:
                raise ValueError("event facts disagree with command lifecycle")

    def _validate_manifest_ledger(self) -> None:
        """Recompute postings from retained source commands and emitted event facts."""
        from apar.evaluation.v5_execution import _expected_ledger_entry

        states: dict[str, dict[str, object]] = {}
        expected: list[V5LedgerPosting] = []
        opening = dict(self.opening_balances)
        for link, event in zip(self.lineage, self.event_records, strict=True):
            state = cast(dict[str, object], _decode_canonical_json(link.command_payload_json))
            if state.get("payment_id") != link.payment_id:
                raise ValueError("command payload payment lineage disagrees")
            if link.command_name in {
                "a2a.initiate",
                "card.authorize",
                "card.decline",
                "agentic.pay",
            }:
                states[link.payment_id] = state
            source = states.get(link.payment_id)
            if source is None:
                raise ValueError("retained event lacks opening command facts")
            try:
                event_type = EventKind(event.event_type)
            except ValueError as error:
                raise ValueError("retained event type is unknown") from error
            expected_entry = _expected_ledger_entry(
                cast(
                    PaymentEvent,
                    SimpleNamespace(
                        event_id=event.event_id,
                        rail=Rail(self.rail),
                        event_type=event_type,
                    ),
                ),
                source,
            )
            if expected_entry is not None:
                expected.append(
                    V5LedgerPosting(
                        entry_id=expected_entry.entry_id,
                        debit=tuple(sorted(expected_entry.debit.items())),
                        credit=tuple(sorted(expected_entry.credit.items())),
                        currency=expected_entry.currency,
                    )
                )
        if tuple(expected) != self.ledger_postings:
            raise ValueError("ledger postings do not reconcile to event facts")
        try:
            ledger = Ledger(cast(dict[AccountReference, Decimal], opening))
            for posting in self.ledger_postings:
                private_accounts = {
                    account
                    for account, _amount in (*posting.debit, *posting.credit)
                    if account.startswith("acct:")
                }
                if not private_accounts <= set(opening):
                    raise ValueError("ledger posting references an account without opening facts")
                ledger.post(
                    LedgerEntry(
                        posting.entry_id,
                        dict(posting.debit),
                        dict(posting.credit),
                        posting.currency,
                    )
                )
            ledger.assert_conserved()
        except (AssertionError, TypeError, ValueError) as error:
            raise ValueError("ledger postings do not conserve opening balances") from error

    def _validate_manifest_trust(self) -> None:
        """Re-run verifier records from retained registry, request, and receipt facts."""
        if self.rail != Rail.AGENTIC.value:
            return
        assert self.trust_registry is not None
        from apar.trust.verifier import (
            IntegrityReceipt,
            ReceiptOutcome,
            TrustCommitPlan,
            TrustVerifier,
        )

        registry_mandate = _agent_mandate_from_json(self.trust_registry.mandate_json)
        registry_evidence = tuple(
            _authentication_evidence_from_json(item)
            for item in self.trust_registry.authentication_evidence_json
        )
        registry_evidence_by_id = {
            item.evidence_id: item for item in registry_evidence
        }
        verifier = TrustVerifier(
            registered_agents={
                (self.trust_registry.agent_id, self.trust_registry.key_id): bytes.fromhex(
                    self.trust_registry.public_key_hex
                )
            },
            mandates={registry_mandate.mandate_id: registry_mandate},
            authentication_evidence={item.evidence_id: item for item in registry_evidence},
        )
        event_by_id = {record.event_id: record for record in self.event_records}
        for record in self.trust_records:
            request = _agent_request_from_json(record.request_json)
            if (
                request.request_id != record.request_id
                or request.agent_id != record.agent_id
                or request.key_id != record.key_id
                or request.mandate.mandate_id != record.mandate_id
                or request.authentication_evidence_ref != record.authentication_evidence_id
                or request.mandate.canonical_bytes()
                != _agent_mandate_from_json(record.mandate_json).canonical_bytes()
                or record.public_key_hex != self.trust_registry.public_key_hex
            ):
                raise ValueError("verifier record identifiers disagree with retained request facts")
            expected_authentication = registry_evidence_by_id.get(
                request.authentication_evidence_ref or ""
            )
            if expected_authentication is None:
                if record.authentication_evidence_json is not None:
                    raise ValueError("verifier record retains unexpected authentication evidence")
            elif (
                record.authentication_evidence_json is None
                or _authentication_evidence_from_json(
                    record.authentication_evidence_json
                )
                != expected_authentication
            ):
                raise ValueError("verifier authentication evidence disagrees with registry facts")
            event = event_by_id[record.event_id]
            now = event.decision_at.astimezone(UTC).replace(tzinfo=UTC)
            preview = verifier.preview(request, now)
            if preview.allowed:
                rail_data = cast(
                    dict[str, object], _decode_canonical_json(event.rail_data_json)
                )
                action = cast(str, rail_data["action"])
                try:
                    outcome = ReceiptOutcome(action)
                except ValueError as error:
                    raise ValueError("verifier replay has unknown emitted action") from error
                prepared = verifier.prepare_commit(request, preview, outcome, now)
                if type(prepared) is TrustCommitPlan:
                    receipt = verifier.apply_commit(prepared)
                else:
                    receipt = cast(IntegrityReceipt, prepared)
            else:
                receipt = preview
            if type(receipt) is not IntegrityReceipt:
                raise TypeError("TrustVerifier returned an unknown verdict")
            reason_code = receipt.reason_code.value if receipt.reason_code is not None else None
            if (
                receipt.request_id != record.request_id
                or receipt.receipt_hash != record.receipt_hash
                or receipt.request_hash != record.request_hash
                or receipt.signature_hash != record.signature_hash
                or receipt.allowed is not record.allowed
                or reason_code != record.reason_code
                or receipt.outcome.value != record.outcome
            ):
                raise ValueError("verifier replay disagrees with retained receipt facts")


class V5PartitionCorpus(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    partition_name: str
    decisions: tuple[V5DecisionRow, ...]
    executions: tuple[V5ExecutionManifest, ...] = ()

    @model_validator(mode="after")
    def every_campaign_row_has_real_execution_evidence(self) -> Self:
        manifests = {item.evidence_sha256: item for item in self.executions}
        if len(manifests) != len(self.executions):
            raise ValueError("real execution evidence digests must be unique")
        manifested_events: set[tuple[str, str]] = set()
        observed_events: set[tuple[str, str]] = set()
        for manifest in self.executions:
            for link in manifest.lineage:
                key = (manifest.evidence_sha256, link.event_id)
                if key in manifested_events:
                    raise ValueError("real execution evidence events must be unique")
                manifested_events.add(key)

        for row in self.decisions:
            manifest_candidate = manifests.get(row.execution_evidence_sha256)
            if manifest_candidate is None:
                raise ValueError("decision row lacks real execution evidence")
            manifest = manifest_candidate
            if (
                manifest.campaign_id != row.campaign_id
                or manifest.family != row.family
                or manifest.rail != row.rail
            ):
                raise ValueError("campaign row disagrees with real execution evidence")
            matching = [item for item in manifest.lineage if item.event_id == row.source_event_id]
            if len(matching) != 1:
                raise ValueError("campaign row source event lacks real execution evidence")
            link = matching[0]
            if (
                row.event_id != link.event_id
                or row.source_command_id != link.command_id
                or row.payment_id != link.payment_id
                or row.actor_id != link.actor_id
                or row.counterparty_id != link.counterparty_id
                or row.is_fraud is not link.is_fraud
            ):
                raise ValueError("campaign row lineage disagrees with real execution evidence")
            expected_integrity = (
                "fail"
                if row.event_id in manifest.trust_failure_event_ids
                else "pass"
                if row.rail == Rail.AGENTIC.value
                else "not_applicable"
            )
            if row.integrity_status != expected_integrity:
                raise ValueError("campaign row trust status disagrees with verifier evidence")
            observed_events.add((manifest.evidence_sha256, row.event_id))
        if observed_events != manifested_events:
            raise ValueError("real execution evidence and projected campaign rows are incomplete")
        return self

    @property
    def fraud_count(self) -> int:
        return sum(1 for row in self.decisions if row.is_fraud)

    @property
    def benign_count(self) -> int:
        return sum(1 for row in self.decisions if not row.is_fraud)


class V5Corpus(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    profile: V5Profile
    partitions: dict[str, V5PartitionCorpus]
    corpus_sha256: str
    is_production: bool


_PARTITION_SEED_KEYS = (
    "train",
    "calibration",
    "threshold",
    "development_test",
    "hardening_train",
    "adaptive_holdout",
)
_FAMILY_RAILS = {
    V5Family.AGENTIC_INTENT_ABUSE.value: Rail.AGENTIC,
    V5Family.APP_SCAM_MULE.value: Rail.A2A,
    V5Family.CARD_TESTING_CNP.value: Rail.CARD,
    V5Family.SYNTHETIC_MERCHANT_REFUND.value: Rail.CARD,
}
_FAMILY_MOTIFS = {
    V5Family.AGENTIC_INTENT_ABUSE.value: AGENTIC_INTENT_ABUSE_MOTIF,
    V5Family.APP_SCAM_MULE.value: APP_SCAM_MULE_MOTIF,
    V5Family.CARD_TESTING_CNP.value: CARD_TESTING_CNP_MOTIF,
    V5Family.SYNTHETIC_MERCHANT_REFUND.value: SYNTHETIC_MERCHANT_REFUND_MOTIF,
}
_BASE_START = datetime(2026, 1, 1, tzinfo=UTC)
_PARTITION_OFFSETS_DAYS = {
    "train": 0,
    "calibration": 10,
    "threshold": 20,
    "development_test": 30,
    "hardening_train": 40,
    "adaptive_holdout": 50,
}
_MAX_LEGITIMATE_EVENTS_PER_MANIFEST = 96
_LEGITIMATE_BATCH_START_SECONDS = 8 * 60 * 60
_LEGITIMATE_BATCH_STRIDE_SECONDS = 60
_LEGITIMATE_EVENT_STRIDE_MILLISECONDS = 500
_LEGITIMATE_MANIFEST_BASE_ESTIMATE_BYTES = 32_768
_LEGITIMATE_EVENT_ESTIMATE_BYTES = 3_000
_AGENTIC_MANIFEST_BASE_ESTIMATE_BYTES = 65_536
_AGENTIC_EVENT_ESTIMATE_BYTES = 12_000
_FRAUD_EVENT_COUNTS = {
    V5Family.AGENTIC_INTENT_ABUSE.value: 25,
    V5Family.APP_SCAM_MULE.value: 36,
    V5Family.CARD_TESTING_CNP.value: 26,
    V5Family.SYNTHETIC_MERCHANT_REFUND.value: 46,
}
_FRAUD_EVENT_ESTIMATE_BYTES = {
    V5Family.AGENTIC_INTENT_ABUSE.value: 12_000,
    V5Family.APP_SCAM_MULE.value: 3_000,
    V5Family.CARD_TESTING_CNP.value: 3_000,
    V5Family.SYNTHETIC_MERCHANT_REFUND.value: 3_000,
}
_MAX_SINGLE_EXECUTION_ARTIFACT_BYTES = 16 * 1024 * 1024
_MAX_AGGREGATE_EXECUTION_ARTIFACT_BYTES = 256 * 1024 * 1024


@dataclass(frozen=True, slots=True)
class _LegitimateExecutionBatch:
    campaign_index: int
    rail: Rail
    event_count: int
    start_offset_seconds: int
    duration_seconds: int
    estimated_payload_bytes: int


@dataclass(frozen=True, slots=True)
class _PlannedExecutionArtifact:
    category: str
    rail: Rail
    event_count: int
    estimated_payload_bytes: int


@dataclass(frozen=True, slots=True)
class _DevelopmentExecutionArtifactPlan:
    artifacts: tuple[_PlannedExecutionArtifact, ...]
    legitimate_event_count: int

    @property
    def artifact_counts_by_category(self) -> Mapping[str, int]:
        counts: dict[str, int] = {}
        for artifact in self.artifacts:
            counts[artifact.category] = counts.get(artifact.category, 0) + 1
        return MappingProxyType(dict(sorted(counts.items())))

    @property
    def artifact_count(self) -> int:
        return len(self.artifacts)

    @property
    def max_artifact_payload_bytes(self) -> int:
        return max(artifact.estimated_payload_bytes for artifact in self.artifacts)

    @property
    def aggregate_payload_bytes(self) -> int:
        return sum(artifact.estimated_payload_bytes for artifact in self.artifacts)


def _estimate_execution_shape_payload_bytes(
    *,
    family: str,
    rail: Rail,
    event_count: int,
) -> int:
    """Conservatively bound canonical retained facts for one known execution shape."""
    if type(event_count) is not int or event_count <= 0:
        raise ValueError("execution artifact event count must be positive")
    if family == "legitimate":
        if rail is Rail.AGENTIC:
            return (
                _AGENTIC_MANIFEST_BASE_ESTIMATE_BYTES
                + event_count * _AGENTIC_EVENT_ESTIMATE_BYTES
            )
        return (
            _LEGITIMATE_MANIFEST_BASE_ESTIMATE_BYTES
            + event_count * _LEGITIMATE_EVENT_ESTIMATE_BYTES
        )
    try:
        per_event = _FRAUD_EVENT_ESTIMATE_BYTES[family]
    except KeyError as error:
        raise ValueError("execution artifact family is unknown") from error
    base = (
        _AGENTIC_MANIFEST_BASE_ESTIMATE_BYTES
        if rail is Rail.AGENTIC
        else _LEGITIMATE_MANIFEST_BASE_ESTIMATE_BYTES
    )
    return base + event_count * per_event


def _estimate_execution_artifact_payload_bytes(
    manifest: V5ExecutionManifest,
) -> int:
    """Estimate one validated canonical manifest without trusting serialized byte size."""
    if type(manifest) is not V5ExecutionManifest:
        raise TypeError("artifact estimator requires an exact execution manifest")
    return _estimate_execution_shape_payload_bytes(
        family=manifest.family,
        rail=Rail(manifest.rail),
        event_count=len(manifest.lineage),
    )


def _plan_legitimate_filler_batches(
    count: int,
) -> tuple[_LegitimateExecutionBatch, ...]:
    """Plan exact bounded real-execution batches without executing a profile."""
    if type(count) is not int or count < 0:
        raise ValueError("legitimate filler count must be a non-negative integer")
    batches: list[_LegitimateExecutionBatch] = []
    remaining = count
    campaign_index = 0
    while remaining:
        event_count = min(_MAX_LEGITIMATE_EVENTS_PER_MANIFEST, remaining)
        start_offset = (
            _LEGITIMATE_BATCH_START_SECONDS
            + campaign_index * _LEGITIMATE_BATCH_STRIDE_SECONDS
        )
        duration_milliseconds = (
            event_count - 1
        ) * _LEGITIMATE_EVENT_STRIDE_MILLISECONDS
        duration = (duration_milliseconds + 999) // 1_000
        if start_offset + duration >= 24 * 60 * 60:
            raise ValueError("legitimate batch plan exceeds its partition time window")
        batches.append(
            _LegitimateExecutionBatch(
                campaign_index=campaign_index,
                rail=Rail.CARD if campaign_index % 2 == 0 else Rail.A2A,
                event_count=event_count,
                start_offset_seconds=start_offset,
                duration_seconds=duration,
                estimated_payload_bytes=(
                    _LEGITIMATE_MANIFEST_BASE_ESTIMATE_BYTES
                    + event_count * _LEGITIMATE_EVENT_ESTIMATE_BYTES
                ),
            )
        )
        remaining -= event_count
        campaign_index += 1
    return tuple(batches)


def _plan_production_development_execution_artifacts(
    protocol: V5DevelopmentProtocol,
) -> _DevelopmentExecutionArtifactPlan:
    """Statically bound every declared production development-test execution artifact."""
    if type(protocol) is not V5DevelopmentProtocol:
        raise TypeError("artifact planning requires an exact v5 protocol")
    base_shapes = (
        (Rail.CARD, 12),
        (Rail.A2A, 10),
        (Rail.AGENTIC, 2),
    )
    base_event_count = sum(event_count for _rail, event_count in base_shapes)
    legitimate_target = protocol.production_dev_test_legitimate
    if legitimate_target < base_event_count:
        raise ValueError("production legitimate target cannot cover base rail traffic")
    artifacts = [
        _PlannedExecutionArtifact(
            category="legitimate_base",
            rail=rail,
            event_count=event_count,
            estimated_payload_bytes=_estimate_execution_shape_payload_bytes(
                family="legitimate",
                rail=rail,
                event_count=event_count,
            ),
        )
        for rail, event_count in base_shapes
    ]
    for batch in _plan_legitimate_filler_batches(
        legitimate_target - base_event_count
    ):
        artifacts.append(
            _PlannedExecutionArtifact(
                category="legitimate_filler",
                rail=batch.rail,
                event_count=batch.event_count,
                estimated_payload_bytes=batch.estimated_payload_bytes,
            )
        )
    campaign_counts = protocol.production_profile.campaigns_per_family
    if set(campaign_counts) != set(_FRAUD_EVENT_COUNTS):
        raise ValueError("production campaign families differ from artifact planner")
    for family in sorted(campaign_counts):
        rail = _FAMILY_RAILS[family]
        event_count = _FRAUD_EVENT_COUNTS[family]
        estimate = _estimate_execution_shape_payload_bytes(
            family=family,
            rail=rail,
            event_count=event_count,
        )
        artifacts.extend(
            _PlannedExecutionArtifact(
                category=family,
                rail=rail,
                event_count=event_count,
                estimated_payload_bytes=estimate,
            )
            for _ in range(campaign_counts[family])
        )
    plan = _DevelopmentExecutionArtifactPlan(
        artifacts=tuple(artifacts),
        legitimate_event_count=legitimate_target,
    )
    if plan.artifact_count > 4_096:
        raise ValueError("development execution artifact count exceeds profile limit")
    if plan.max_artifact_payload_bytes >= _MAX_SINGLE_EXECUTION_ARTIFACT_BYTES:
        raise ValueError("development execution artifact exceeds individual byte limit")
    if plan.aggregate_payload_bytes >= _MAX_AGGREGATE_EXECUTION_ARTIFACT_BYTES:
        raise ValueError("development execution artifacts exceed aggregate byte limit")
    return plan


class _PopulationIsolationError(ValueError):
    """Raised when corpus identities, time, or provenance cross partitions."""


def _domain_seed(base_seed: int, *labels: object) -> int:
    document = ":".join(
        ["apar", "sentinel-v5", "development", str(base_seed), *(str(item) for item in labels)]
    )
    derived = int.from_bytes(hashlib.sha256(document.encode()).digest()[:8], "big")
    derived &= 2**63 - 1
    return derived or 1


def _scenario_bundle(
    *,
    partition_name: str,
    family: str,
    campaign_index: int,
    seed: int,
    rail: Rail,
) -> ScenarioBundle:
    scenario_id = f"sentinel-v5-{partition_name}-{family}-{campaign_index:04d}"
    simulation_start = _BASE_START + timedelta(
        days=_PARTITION_OFFSETS_DAYS[partition_name],
        minutes=campaign_index,
    )
    stages = [
        CampaignStage(stage_id="generated", description="Generated payment campaign"),
        CampaignStage(stage_id="executed", description="Executed on the real rail adapter"),
    ]
    transitions = [
        StageTransition(
            from_stage="generated",
            to_stage="executed",
            condition="commands_scheduled",
        )
    ]
    return ScenarioBundle(
        schema_version="1.0.0",
        scenario_id=scenario_id,
        version="1.0.0",
        threat_card_ref=f"sentinel-v5-{family}@1",
        rail=rail,
        viewpoint="network_with_bank_enrichment",
        genai_capability={"synthetic_campaign": True},
        attacker_mode=AttackerMode.DECISION_ONLY,
        attacker_objective="development_execution_evidence",
        query_budget=40,
        feedback=[
            FeedbackField.APPROVE,
            FeedbackField.CHALLENGE,
            FeedbackField.DECLINE,
            FeedbackField.REALIZED_VALUE,
        ],
        benign_entity_count=40,
        illicit_entity_count=16,
        duration_hours=24,
        seed=seed,
        campaign_stages=stages,
        transition_rules=transitions,
        economics={"currency": "USD"},
        lifecycle={"execution_evidence": 1},
        hidden_validity={},
        defender_knowledge_boundary="decision-available payment signals only",
        replay_manifest=ReplayManifest(
            scenario_id=scenario_id,
            scenario_version="1.0.0",
            threat_card_ref=f"sentinel-v5-{family}@1",
            random_seed=seed,
            simulation_start=simulation_start,
            generator_version="0.1.0",
            event_ordering=ReplayOrdering.EVENT_TIME_THEN_EVENT_ID,
        ),
        safety={"synthetic_only": True, "export_level": "sanitized"},
        extensions={},
    )


def _campaign_params(
    *,
    partition_name: str,
    family: str,
    campaign_index: int,
    seed: int,
) -> CampaignParams:
    namespace = uuid5(
        NAMESPACE_URL,
        f"apar:sentinel-v5:{partition_name}:{family}:{seed}",
    )
    campaign_id = str(uuid5(namespace, f"campaign:{campaign_index}"))
    values: dict[str, object] = {
        "campaign_id": campaign_id,
        "seed": seed,
        "payment_count": 10,
        "target_illicit_rate": Decimal("0.70"),
        "class_rate_tolerance": Decimal("0.05"),
        "target_value_total": Decimal("500.00"),
        "value_tolerance": Decimal("0.01"),
        "min_amount": Decimal("10.00"),
        "max_amount": Decimal("90.00"),
        "currency": "USD",
        "duration_hours": 12,
        "query_budget": 40,
        "min_delay_seconds": 1,
        "max_delay_seconds": 300,
        "expected_motif": _FAMILY_MOTIFS[family],
    }
    if family == V5Family.AGENTIC_INTENT_ABUSE.value:
        values.update(
            {
                "payment_count": 25,
                "target_illicit_rate": Decimal("0.92"),
                "class_rate_tolerance": Decimal("0.01"),
            }
        )
    return CampaignParams(**values)  # type: ignore[arg-type]


def _adapter_factory(
    *,
    rail: Rail,
    campaign_evidence: object,
) -> AdapterFactory:
    if rail is Rail.CARD:

        def card_factory() -> CardRailAdapter:
            return CardRailAdapter()

        return card_factory
    if rail is Rail.A2A:

        def a2a_factory() -> A2ARailAdapter:
            return A2ARailAdapter()

        return a2a_factory

    fixture = getattr(campaign_evidence, "agentic_fixture", None)
    if fixture is None:
        raise ValueError("agentic campaign is missing TrustVerifier inputs")
    verifier = TrustVerifier(
        registered_agents={(fixture.agent_id, fixture.key_id): fixture.public_key},
        mandates={fixture.mandate.mandate_id: fixture.mandate},
        authentication_evidence={
            item.evidence_id: item for item in fixture.authentication_evidence
        },
    )

    def agentic_factory() -> AgenticRailAdapter:
        return AgenticRailAdapter(
            verifier,
            lambda _request, _receipt: Action.APPROVE,
        )

    return agentic_factory


def _canonical_fact(value: object) -> object:
    """Encode retained execution facts without preserving mutable runtime objects."""
    if value is None or type(value) in (bool, int, str):
        return value
    if type(value) is Decimal:
        return {"decimal": str(value)}
    if type(value) is bytes:
        return {"bytes": value.hex()}
    if type(value) is datetime:
        return {"datetime": value.isoformat().replace("+00:00", "Z")}
    raw_value = getattr(value, "value", None)
    if type(raw_value) is str:
        return raw_value
    if isinstance(value, tuple):
        return [_canonical_fact(item) for item in value]
    if isinstance(value, Mapping):
        return {str(key): _canonical_fact(item) for key, item in sorted(value.items())}
    raise TypeError(f"unsupported retained execution fact: {type(value).__name__}")


def _canonical_fact_json(value: object) -> str:
    return json.dumps(
        _canonical_fact(value),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _decode_canonical_fact(value: object) -> object:
    """Restore the closed canonical representation retained in a manifest."""
    if isinstance(value, list):
        return tuple(_decode_canonical_fact(item) for item in value)
    if isinstance(value, dict):
        if set(value) == {"decimal"}:
            return Decimal(cast(str, value["decimal"]))
        if set(value) == {"bytes"}:
            return bytes.fromhex(cast(str, value["bytes"]))
        if set(value) == {"datetime"}:
            parsed = datetime.fromisoformat(
                cast(str, value["datetime"]).replace("Z", "+00:00")
            )
            return parsed.astimezone(UTC).replace(tzinfo=UTC)
        return {str(key): _decode_canonical_fact(item) for key, item in value.items()}
    return value


def _decode_canonical_json(value: str) -> object:
    if type(value) is not str:
        raise TypeError("canonical retained fact must be JSON text")
    return _decode_canonical_fact(json.loads(value))


def _agent_mandate_from_json(value: str) -> Any:
    from apar.trust.verifier import AgentMandate, AuthenticationRequirement

    document = cast(dict[str, object], _decode_canonical_json(value))
    document["required_authentication"] = AuthenticationRequirement(
        cast(str, document["required_authentication"])
    )
    return AgentMandate(**cast(Any, document))


def _authentication_evidence_from_json(value: str) -> Any:
    from apar.trust.verifier import AuthenticationEvidence, AuthenticationOutcome

    document = cast(dict[str, object], _decode_canonical_json(value))
    document["outcome"] = AuthenticationOutcome(cast(str, document["outcome"]))
    return AuthenticationEvidence(**cast(Any, document))


def _agent_request_from_json(value: str) -> Any:
    from apar.trust.verifier import AgentPaymentRequest

    document = cast(dict[str, object], _decode_canonical_json(value))
    mandate_data = document.pop("mandate")
    document["mandate"] = _agent_mandate_from_json(_canonical_fact_json(mandate_data))
    return AgentPaymentRequest(**cast(Any, document))


def _manifest_from_evidence(
    evidence: object,
    *,
    population: Population,
) -> V5ExecutionManifest:
    from apar.evaluation.v5_execution import V5ExecutionEvidence

    if type(evidence) is not V5ExecutionEvidence:
        raise TypeError("manifest requires exact validated execution evidence")
    account_ids = {
        account
        for account, _amount in evidence.opening_balances
        if type(account) is str and account.startswith("acct:")
    }
    trust_failures = tuple(
        record.event_id for record in evidence.trust_evidence if not record.receipt.allowed
    )
    fixture = evidence.campaign_evidence.agentic_fixture
    if evidence.rail is Rail.AGENTIC and fixture is None:
        raise ValueError("agentic evidence must retain verifier fixture facts")
    candidate = V5ExecutionManifest.model_construct(
        evidence_sha256=evidence.evidence_sha256,
        artifact_sha256="0" * 64,
        campaign_id=evidence.campaign_id,
        family=evidence.family,
        rail=evidence.rail.value,
        lineage=tuple(
            V5LineageManifest(
                command_id=item.command_id,
                command_type=type(command).__qualname__,
                command_name=item.command_name,
                event_id=item.event_id,
                payment_id=item.payment_id,
                actor_id=item.actor_id,
                counterparty_id=item.counterparty_id,
                lifecycle_position=item.lifecycle_position,
                is_fraud=item.is_fraud,
                command_payload_json=_canonical_fact_json(command.payload),
                trace_id=event.trace_id,
                scheduled_at=item.scheduled_at,
            )
            for item, command, event in zip(
                evidence.lineage, evidence.commands, evidence.events, strict=True
            )
        ),
        ledger_entry_ids=tuple(entry.entry_id for entry in evidence.ledger_entries),
        trust_request_ids=tuple(
            record.request.request_id for record in evidence.trust_evidence
        ),
        trust_failure_event_ids=trust_failures,
        account_ids=tuple(sorted(account_ids)),
        opening_balances=tuple(
            (account, amount)
            for account, amount in evidence.opening_balances
            if type(account) is str
        ),
        device_ids=tuple(
            sorted(
                {
                    entity.entity_id
                    for entity in population.entities
                    if entity.role == "device"
                }
                | {activity.device_id for activity in population.benign_activities}
            )
        ),
        credential_ids=tuple(
            sorted({record.request.credential_id for record in evidence.trust_evidence})
        ),
        merchant_ids=tuple(
            sorted(
                {record.request.merchant_id for record in evidence.trust_evidence}
                | {
                    link.counterparty_id
                    for link in evidence.lineage
                    if link.rail is Rail.CARD
                }
            )
        ),
        payee_ids=tuple(
            sorted(
                {record.request.payee_id for record in evidence.trust_evidence}
                | {
                    cast(str, command.payload["payee_account"])
                    for command in evidence.commands
                    if "payee_account" in command.payload
                }
            )
        ),
        agent_ids=tuple(sorted({record.request.agent_id for record in evidence.trust_evidence})),
        key_ids=tuple(sorted({record.request.key_id for record in evidence.trust_evidence})),
        mandate_ids=tuple(
            sorted({record.request.mandate.mandate_id for record in evidence.trust_evidence})
        ),
        authentication_evidence_ids=tuple(
            sorted(
                {
                    record.request.authentication_evidence_ref
                    for record in evidence.trust_evidence
                    if record.request.authentication_evidence_ref is not None
                }
            )
        ),
        event_records=tuple(
            V5EventRecord(
                event_id=event.event_id,
                payment_id=str(event.rail_data["payment_id"]),
                event_type=event.event_type.value,
                amount=event.amount,
                currency=event.currency,
                decision_at=cast(datetime, event.decision_at),
                rail_data_json=_canonical_fact_json(event.rail_data),
                lineage_json=_canonical_fact_json(event.lineage),
                event_json=json.dumps(
                    event.model_dump(mode="json", warnings=False),
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                ),
            )
            for event in evidence.events
        ),
        ledger_postings=tuple(
            V5LedgerPosting(
                entry_id=entry.entry_id,
                debit=tuple(sorted(entry.debit.items())),
                credit=tuple(sorted(entry.credit.items())),
                currency=entry.currency,
            )
            for entry in evidence.ledger_entries
        ),
        trust_records=tuple(
            V5TrustRecord(
                command_id=record.command_id,
                event_id=record.event_id,
                request_id=record.request.request_id,
                agent_id=record.request.agent_id,
                key_id=record.request.key_id,
                mandate_id=record.request.mandate.mandate_id,
                authentication_evidence_id=record.request.authentication_evidence_ref,
                request_json=_canonical_fact_json(asdict(record.request)),
                mandate_json=_canonical_fact_json(asdict(record.request.mandate)),
                authentication_evidence_json=(
                    _canonical_fact_json(asdict(record.authentication_evidence))
                    if record.authentication_evidence is not None
                    else None
                ),
                public_key_hex=fixture.public_key.hex() if fixture is not None else "",
                receipt_hash=record.receipt.receipt_hash,
                request_hash=record.receipt.request_hash,
                signature_hash=record.receipt.signature_hash,
                allowed=record.receipt.allowed,
                reason_code=(
                    record.receipt.reason_code.value
                    if record.receipt.reason_code is not None
                    else None
                ),
                outcome=record.receipt.outcome.value,
            )
            for record in evidence.trust_evidence
        ),
        trust_registry=(
            V5TrustRegistry(
                agent_id=fixture.agent_id,
                key_id=fixture.key_id,
                public_key_hex=fixture.public_key.hex(),
                mandate_json=_canonical_fact_json(asdict(fixture.mandate)),
                authentication_evidence_json=tuple(
                    _canonical_fact_json(asdict(item))
                    for item in fixture.authentication_evidence
                ),
            )
            if fixture is not None
            else None
        ),
    )
    document = candidate.model_dump(mode="python")
    document["artifact_sha256"] = candidate.artifact_digest()
    return V5ExecutionManifest.model_validate(document)


def _execute_campaign(
    *,
    partition_name: str,
    family: str,
    campaign_index: int,
    partition_seed: int,
) -> tuple[list[V5DecisionRow], V5ExecutionManifest]:
    from apar.evaluation.v5_execution import (
        build_execution_evidence,
        project_execution_evidence,
    )

    rail = _FAMILY_RAILS[family]
    seed = _domain_seed(
        partition_seed,
        partition_name,
        family,
        campaign_index,
        "real-execution",
    )
    bundle = _scenario_bundle(
        partition_name=partition_name,
        family=family,
        campaign_index=campaign_index,
        seed=seed,
        rail=rail,
    )
    population = PopulationGenerator(seed=seed).generate(bundle)
    params = _campaign_params(
        partition_name=partition_name,
        family=family,
        campaign_index=campaign_index,
        seed=seed,
    )
    commands, campaign_evidence = _CampaignEvaluator(seed=seed).generate(
        family,
        population,
        params,
    )
    factory = _adapter_factory(rail=rail, campaign_evidence=campaign_evidence)
    opening_balances: dict[AccountReference, Decimal] = {
        cast(AccountReference, account): amount
        for account, amount in population.opening_balances.items()
    }
    engine = SimulationEngine(
        bundle,
        {rail: factory},
        opening_balances=opening_balances,
    )
    for priority, (scheduled_at, command) in enumerate(
        zip(campaign_evidence.schedule, commands, strict=True)
    ):
        engine.schedule(scheduled_at, priority, command)
    events = engine.run()
    evidence = build_execution_evidence(
        family=family,
        commands=commands,
        campaign_evidence=campaign_evidence,
        events=events,
        ledger_entries=engine.ledger.entries,
        opening_balances=opening_balances,
    )
    return project_execution_evidence(evidence), _manifest_from_evidence(
        evidence,
        population=population,
    )


def _legitimate_campaign_evidence(
    *,
    campaign_id: str,
    commands: tuple[Command, ...],
    schedule: tuple[datetime, ...],
    declared_entity_ids: tuple[str, ...],
    account_ids: tuple[str, ...],
    fixture: AgenticFixture | None = None,
) -> CampaignEvidence:
    """Create the explicit all-legitimate ground-truth contract for real traffic."""
    opening_count = sum(
        command.name in {"a2a.initiate", "card.authorize", "card.decline", "agentic.pay"}
        for command in commands
    )
    if opening_count == 0:
        raise ValueError("legitimate execution requires at least one payment opening")
    if len(commands) != len(schedule):
        raise ValueError("legitimate execution schedule must cover every command")
    payload = {
        "campaign_id": campaign_id,
        "commands": [command.name for command in commands],
        "schedule": [time.isoformat().replace("+00:00", "Z") for time in schedule],
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    attempted_value = sum(
        (
            cast(Decimal, command.payload["amount"])
            for command in commands
            if command.name in {"a2a.initiate", "card.authorize", "card.decline", "agentic.pay"}
        ),
        Decimal("0"),
    )
    return CampaignEvidence(
        family="legitimate",
        campaign_id=campaign_id,
        motif_signature="legitimate_operational_traffic.v1",
        payment_count=opening_count,
        command_count=len(commands),
        illicit_count=0,
        illicit_rate=Decimal("0"),
        value_total=attempted_value,
        attempted_value=attempted_value,
        unique_attempted_value=attempted_value,
        settled_value=Decimal("0"),
        schedule=schedule,
        graph_digest=digest,
        schedule_digest=digest,
        declared_entity_ids=tuple(sorted(declared_entity_ids)),
        account_ids=tuple(sorted(account_ids)),
        class_labels=tuple(False for _ in range(opening_count)),
        mutation_kinds=(),
        dependencies=(),
        observed_reasons=tuple(None for _ in range(opening_count)),
        valid_control_count=opening_count,
        replay_succeeded=False,
        ledger_conserved=False,
        attempts=1,
        agentic_fixture=fixture,
    )


def _realize_legitimate_campaign_evidence(
    contract: CampaignEvidence,
    *,
    commands: tuple[Command, ...],
    events: tuple[PaymentEvent, ...],
    ledger_entries: tuple[LedgerEntry, ...],
    opening_balances: dict[AccountReference, Decimal],
) -> CampaignEvidence:
    """Derive lifecycle-net economics; execution evidence validates the actual ledger."""
    if contract.family != "legitimate":
        raise ValueError("only an all-legitimate ground-truth contract can be realized")
    attempted_value = sum(
        (
            cast(Decimal, command.payload["amount"])
            for command in commands
            if command.name in {"a2a.initiate", "card.authorize", "card.decline", "agentic.pay"}
        ),
        Decimal("0"),
    )
    del ledger_entries, opening_balances
    settled_value = Decimal("0")
    for event in events:
        if (
            event.event_type in {EventKind.SETTLEMENT, EventKind.TRANSFER_POSTED}
            or event.rail is Rail.AGENTIC
            and event.event_type is EventKind.AUTHORIZATION
        ):
            settled_value += event.amount
        elif (
            event.event_type in {EventKind.REFUND, EventKind.CHARGEBACK}
            or event.rail is Rail.A2A
            and event.event_type in {EventKind.TRANSFER_RETURNED, EventKind.RECOVERY}
        ):
            settled_value -= event.amount
    return replace(
        contract,
        value_total=attempted_value,
        attempted_value=attempted_value,
        unique_attempted_value=attempted_value,
        settled_value=settled_value,
        # These flags remain false until a full build_execution_evidence call
        # revalidates actual engine events and ledger postings below the generator.
        replay_succeeded=False,
        ledger_conserved=False,
    )


def _legitimate_population(
    *,
    partition_name: str,
    rail: Rail,
    partition_seed: int,
    campaign_index: int = 0,
) -> tuple[Population, int]:
    seed = _domain_seed(
        partition_seed,
        partition_name,
        "legitimate",
        rail.value,
        campaign_index,
    )
    bundle = _scenario_bundle(
        partition_name=partition_name,
        family=f"legitimate-{rail.value}",
        campaign_index=campaign_index,
        seed=seed,
        rail=rail,
    )
    return PopulationGenerator(seed=seed).generate(bundle), seed


def _population_parties(
    population: Population,
) -> tuple[PopulationEntity, PopulationEntity, PopulationEntity]:
    entities = getattr(population, "entities", ())
    consumers = [entity for entity in entities if entity.role == "consumer"]
    merchants = [entity for entity in entities if entity.role == "merchant"]
    beneficiaries = [entity for entity in entities if entity.role == "beneficiary"]
    if not consumers or not merchants or not beneficiaries:
        raise ValueError("legitimate population is missing required operational parties")
    if any(entity.account_id is None for entity in (consumers[0], merchants[0], beneficiaries[0])):
        raise ValueError("legitimate population party is missing an account")
    return consumers[0], merchants[0], beneficiaries[0]


def _command_identity(
    *,
    partition_name: str,
    rail: Rail,
    seed: int,
    campaign_index: int = 0,
) -> tuple[str, Callable[[str], str], Callable[[str], str]]:
    namespace = uuid5(
        NAMESPACE_URL,
        "apar:sentinel-v5:legitimate:"
        f"{partition_name}:{rail.value}:{seed}:{campaign_index}",
    )
    campaign_id = str(uuid5(namespace, "campaign"))
    return (
        campaign_id,
        lambda label: f"payment:{uuid5(namespace, f'payment:{label}')}",
        lambda label: str(uuid5(namespace, f"trace:{label}")),
    )


def _lifecycle_key(name: str, payment_id: str, campaign_id: str) -> str:
    return f"{name}:{payment_id}:campaign:{campaign_id}"


def _card_legitimate_commands(
    *,
    partition_name: str,
    partition_seed: int,
) -> tuple[tuple[Command, ...], tuple[datetime, ...], Population, CampaignEvidence]:
    population, seed = _legitimate_population(
        partition_name=partition_name,
        rail=Rail.CARD,
        partition_seed=partition_seed,
    )
    consumer, merchant, _beneficiary = _population_parties(population)
    campaign_id, payment, trace = _command_identity(
        partition_name=partition_name,
        rail=Rail.CARD,
        seed=seed,
    )
    command_rows: list[Command] = []
    def authorize(label: str, amount: str) -> AuthorizeCard:
        return AuthorizeCard(
            payment(label),
            amount=Decimal(amount), currency="USD",
            payer_account=cast(str, consumer.account_id),
            payee_account=cast(str, merchant.account_id),
            actor_id=consumer.entity_id, counterparty_id=merchant.entity_id,
            campaign_id=campaign_id, trace_id=trace(label), fee=Decimal("0.30"),
        )

    for label, amount, followups in (
        ("refund", "58.40", ("clear", "settle", "refund")),
        (
            "chargeback_recovery",
            "57.70",
            ("clear", "settle", "report", "dispute", "chargeback", "recover"),
        ),
    ):
        opening = authorize(label, amount)
        command_rows.append(opening)
        for operation in followups:
            constructors: dict[str, Callable[..., Command]] = {
                "clear": ClearCard, "settle": SettleCard, "refund": RefundCard,
                "chargeback": ChargebackCard, "recover": RecoverCard,
                "reverse": ReverseCardAuthorization, "dispute": OpenCardDispute,
                "report": ReportCardFraud,
            }
            command_rows.append(
                constructors[operation](
                    opening.payment_id,
                    idempotency_key=_lifecycle_key(
                        f"card.{operation}", opening.payment_id, campaign_id
                    ),
                )
            )
    declined_payment = payment("declined")
    command_rows.append(
        DeclineCardAuthorization(
            declined_payment,
            amount=Decimal("45.60"), currency="USD",
            payer_account=cast(str, consumer.account_id),
            payee_account=cast(str, merchant.account_id),
            actor_id=consumer.entity_id, counterparty_id=merchant.entity_id,
            campaign_id=campaign_id, trace_id=trace("declined"), fee=Decimal("0.30"),
            idempotency_key=_lifecycle_key("card.decline", declined_payment, campaign_id),
        )
    )
    start = _BASE_START + timedelta(days=_PARTITION_OFFSETS_DAYS[partition_name], hours=2)
    schedule = tuple(start + timedelta(minutes=index) for index in range(len(command_rows)))
    commands = tuple(command_rows)
    evidence = _legitimate_campaign_evidence(
        campaign_id=campaign_id,
        commands=commands,
        schedule=schedule,
        declared_entity_ids=(consumer.entity_id, merchant.entity_id),
        account_ids=(cast(str, consumer.account_id), cast(str, merchant.account_id)),
    )
    return commands, schedule, population, evidence


def _a2a_legitimate_commands(
    *,
    partition_name: str,
    partition_seed: int,
) -> tuple[tuple[Command, ...], tuple[datetime, ...], Population, CampaignEvidence]:
    population, seed = _legitimate_population(
        partition_name=partition_name,
        rail=Rail.A2A,
        partition_seed=partition_seed,
    )
    consumer, _merchant, beneficiary = _population_parties(population)
    campaign_id, payment, trace = _command_identity(
        partition_name=partition_name,
        rail=Rail.A2A,
        seed=seed,
    )
    command_rows: list[Command] = []
    def initiate(label: str, amount: str) -> InitiateA2A:
        return InitiateA2A(
            payment(label), amount=Decimal(amount), currency="USD",
            payer_account=cast(str, consumer.account_id),
            payee_account=cast(str, beneficiary.account_id),
            actor_id=consumer.entity_id, counterparty_id=beneficiary.entity_id,
            campaign_id=campaign_id, trace_id=trace(label), fee=Decimal("0.20"),
        )

    for label, amount, followups in (
        ("return", "64.90", ("accept", "post", "return")),
        ("recovery", "81.10", ("accept", "post", "report", "freeze", "recover")),
    ):
        opening = initiate(label, amount)
        command_rows.append(opening)
        constructors: dict[str, Callable[..., Command]] = {
            "accept": AcceptA2A, "post": PostA2A, "return": ReturnA2A,
            "reject": RejectA2A, "report": ReportA2AFraud,
            "freeze": FreezeA2AFunds, "recover": RecoverA2A,
        }
        for operation in followups:
            command_rows.append(
                constructors[operation](
                    opening.payment_id,
                    idempotency_key=_lifecycle_key(
                        f"a2a.{operation}", opening.payment_id, campaign_id
                    ),
                )
            )
    start = _BASE_START + timedelta(days=_PARTITION_OFFSETS_DAYS[partition_name], hours=4)
    schedule = tuple(start + timedelta(minutes=index) for index in range(len(command_rows)))
    commands = tuple(command_rows)
    evidence = _legitimate_campaign_evidence(
        campaign_id=campaign_id,
        commands=commands,
        schedule=schedule,
        declared_entity_ids=(consumer.entity_id, beneficiary.entity_id),
        account_ids=(cast(str, consumer.account_id), cast(str, beneficiary.account_id)),
    )
    return commands, schedule, population, evidence


def _agentic_legitimate_commands(
    *,
    partition_name: str,
    partition_seed: int,
) -> tuple[tuple[Command, ...], tuple[datetime, ...], Population, CampaignEvidence]:
    population, seed = _legitimate_population(
        partition_name=partition_name,
        rail=Rail.AGENTIC,
        partition_seed=partition_seed,
    )
    params = _campaign_params(
        partition_name=partition_name,
        family=V5Family.AGENTIC_INTENT_ABUSE.value,
        campaign_index=10_000,
        seed=seed,
    )
    generated, source = _CampaignEvaluator(seed=seed).generate(
        V5Family.AGENTIC_INTENT_ABUSE.value,
        population,
        params,
    )
    selected = tuple(
        (command, scheduled_at)
        for command, scheduled_at, label in zip(
            generated, source.schedule, source.class_labels, strict=True
        )
        if not label
    )
    if len(selected) < 2 or source.agentic_fixture is None:
        raise ValueError("agentic generator did not provide valid authorization controls")
    commands = tuple(command for command, _scheduled_at in selected)
    schedule = tuple(scheduled_at for _command, scheduled_at in selected)
    first = commands[0]
    evidence = _legitimate_campaign_evidence(
        campaign_id=cast(str, first.payload["campaign_id"]),
        commands=commands,
        schedule=schedule,
        declared_entity_ids=source.declared_entity_ids,
        account_ids=source.account_ids,
        fixture=source.agentic_fixture,
    )
    return commands, schedule, population, evidence


def _scaled_legitimate_commands(
    *,
    partition_name: str,
    partition_seed: int,
    batch: _LegitimateExecutionBatch,
) -> tuple[tuple[Command, ...], tuple[datetime, ...], Population, CampaignEvidence]:
    """Build one bounded operational batch from generated parties and real commands."""
    population, seed = _legitimate_population(
        partition_name=partition_name,
        rail=batch.rail,
        partition_seed=partition_seed,
        campaign_index=batch.campaign_index + 1,
    )
    consumers = [entity for entity in population.entities if entity.role == "consumer"]
    counterparties = [
        entity
        for entity in population.entities
        if entity.role == ("merchant" if batch.rail is Rail.CARD else "beneficiary")
    ]
    if not consumers or not counterparties or any(
        entity.account_id is None for entity in (*consumers, *counterparties)
    ):
        raise ValueError("scaled legitimate population lacks operational parties")
    campaign_id, payment, trace = _command_identity(
        partition_name=partition_name,
        rail=batch.rail,
        seed=seed,
        campaign_index=batch.campaign_index + 1,
    )
    commands_list: list[Command] = []
    used_entities: set[str] = set()
    used_accounts: set[str] = set()
    recipe_index = 0
    remaining = batch.event_count
    card_recipes = (
        ("refund", ("clear", "settle", "refund")),
        ("settle", ("clear", "settle")),
        ("reverse", ("reverse",)),
        ("decline", ()),
    )
    a2a_recipes = (
        ("return", ("accept", "post", "return")),
        ("post", ("accept", "post")),
        ("reject", ("reject",)),
        ("initiate", ()),
    )
    recipes = card_recipes if batch.rail is Rail.CARD else a2a_recipes
    while remaining:
        ordered = (
            recipes[(batch.campaign_index + recipe_index) % len(recipes) :]
            + recipes[: (batch.campaign_index + recipe_index) % len(recipes)]
        )
        recipe_name, followups = next(
            recipe for recipe in ordered if 1 + len(recipe[1]) <= remaining
        )
        consumer = consumers[recipe_index % len(consumers)]
        counterparty = counterparties[
            (batch.campaign_index + recipe_index) % len(counterparties)
        ]
        label = f"batch-{batch.campaign_index:04d}-{recipe_index:04d}-{recipe_name}"
        payment_id = payment(label)
        amount = (
            Decimal("20.00")
            + Decimal(
                (batch.campaign_index * 1703 + recipe_index * 1301) % 7000
            )
            / Decimal("100")
        ).quantize(Decimal("0.01"))
        payer_account = cast(str, consumer.account_id)
        payee_account = cast(str, counterparty.account_id)
        used_entities.update((consumer.entity_id, counterparty.entity_id))
        used_accounts.update((payer_account, payee_account))
        if batch.rail is Rail.CARD:
            opening: Command
            if recipe_name == "decline":
                opening = DeclineCardAuthorization(
                    payment_id,
                    amount=amount,
                    currency="USD",
                    payer_account=payer_account,
                    payee_account=payee_account,
                    actor_id=consumer.entity_id,
                    counterparty_id=counterparty.entity_id,
                    campaign_id=campaign_id,
                    trace_id=trace(label),
                    fee=Decimal("0.30"),
                    idempotency_key=_lifecycle_key(
                        "card.decline", payment_id, campaign_id
                    ),
                )
            else:
                opening = AuthorizeCard(
                    payment_id,
                    amount=amount,
                    currency="USD",
                    payer_account=payer_account,
                    payee_account=payee_account,
                    actor_id=consumer.entity_id,
                    counterparty_id=counterparty.entity_id,
                    campaign_id=campaign_id,
                    trace_id=trace(label),
                    fee=Decimal("0.30"),
                )
            commands_list.append(opening)
            constructors: dict[str, Callable[..., Command]] = {
                "clear": ClearCard,
                "settle": SettleCard,
                "refund": RefundCard,
                "reverse": ReverseCardAuthorization,
            }
        else:
            commands_list.append(
                InitiateA2A(
                    payment_id,
                    amount=amount,
                    currency="USD",
                    payer_account=payer_account,
                    payee_account=payee_account,
                    actor_id=consumer.entity_id,
                    counterparty_id=counterparty.entity_id,
                    campaign_id=campaign_id,
                    trace_id=trace(label),
                    fee=Decimal("0.20"),
                )
            )
            constructors = {
                "accept": AcceptA2A,
                "post": PostA2A,
                "return": ReturnA2A,
                "reject": RejectA2A,
            }
        for operation in followups:
            commands_list.append(
                constructors[operation](
                    payment_id,
                    idempotency_key=_lifecycle_key(
                        f"{batch.rail.value}.{operation}", payment_id, campaign_id
                    ),
                )
            )
        remaining -= 1 + len(followups)
        recipe_index += 1
    commands = tuple(commands_list)
    start = _BASE_START + timedelta(
        days=_PARTITION_OFFSETS_DAYS[partition_name],
        seconds=batch.start_offset_seconds,
    )
    schedule = tuple(
        start
        + timedelta(
            milliseconds=index * _LEGITIMATE_EVENT_STRIDE_MILLISECONDS
        )
        for index in range(batch.event_count)
    )
    evidence = _legitimate_campaign_evidence(
        campaign_id=campaign_id,
        commands=commands,
        schedule=schedule,
        declared_entity_ids=tuple(sorted(used_entities)),
        account_ids=tuple(sorted(used_accounts)),
    )
    return commands, schedule, population, evidence


def _execute_legitimate_traffic(
    *,
    partition_name: str,
    partition_seed: int,
    requested_decisions: int,
) -> tuple[list[V5DecisionRow], list[V5ExecutionManifest]]:
    """Execute real legitimate operations through the same rails and evidence boundary."""
    from apar.evaluation.v5_execution import (
        build_execution_evidence,
        project_execution_evidence,
    )

    executions: list[V5ExecutionManifest] = []
    rows: list[V5DecisionRow] = []
    builders: tuple[
        tuple[
            Rail,
            Callable[
                ...,
                tuple[tuple[Command, ...], tuple[datetime, ...], Population, CampaignEvidence],
            ],
        ],
        ...
    ] = (
        (Rail.CARD, _card_legitimate_commands),
        (Rail.A2A, _a2a_legitimate_commands),
        (Rail.AGENTIC, _agentic_legitimate_commands),
    )
    for rail, builder in builders:
        commands, schedule, population, campaign_evidence = builder(
            partition_name=partition_name,
            partition_seed=partition_seed,
        )
        factory = _adapter_factory(rail=rail, campaign_evidence=campaign_evidence)
        opening_balances = {
            cast(AccountReference, account): amount
            for account, amount in population.opening_balances.items()
        }
        engine = SimulationEngine(
            population.bundle,
            {rail: factory},
            opening_balances=opening_balances,
        )
        for priority, (scheduled_at, command) in enumerate(
            zip(schedule, commands, strict=True)
        ):
            engine.schedule(scheduled_at, priority, command)
        events = engine.run()
        realized_evidence = _realize_legitimate_campaign_evidence(
            campaign_evidence,
            commands=commands,
            events=events,
            ledger_entries=engine.ledger.entries,
            opening_balances=opening_balances,
        )
        evidence = build_execution_evidence(
            family="legitimate",
            commands=commands,
            campaign_evidence=realized_evidence,
            events=events,
            ledger_entries=engine.ledger.entries,
            opening_balances=opening_balances,
        )
        rows.extend(project_execution_evidence(evidence))
        executions.append(_manifest_from_evidence(evidence, population=population))
    remaining = requested_decisions - len(rows)
    if remaining < 0:
        raise ValueError("requested legitimate cardinality is below required rail coverage")
    for batch in _plan_legitimate_filler_batches(remaining):
        commands, schedule, population, campaign_evidence = _scaled_legitimate_commands(
            partition_name=partition_name,
            partition_seed=partition_seed,
            batch=batch,
        )
        opening_balances = {
            cast(AccountReference, account): amount
            for account, amount in population.opening_balances.items()
        }
        engine = SimulationEngine(
            population.bundle,
            {
                batch.rail: _adapter_factory(
                    rail=batch.rail,
                    campaign_evidence=campaign_evidence,
                )
            },
            opening_balances=opening_balances,
        )
        for priority, (scheduled_at, command) in enumerate(
            zip(schedule, commands, strict=True)
        ):
            engine.schedule(scheduled_at, priority, command)
        events = engine.run()
        realized_evidence = _realize_legitimate_campaign_evidence(
            campaign_evidence,
            commands=commands,
            events=events,
            ledger_entries=engine.ledger.entries,
            opening_balances=opening_balances,
        )
        evidence = build_execution_evidence(
            family="legitimate",
            commands=commands,
            campaign_evidence=realized_evidence,
            events=events,
            ledger_entries=engine.ledger.entries,
            opening_balances=opening_balances,
        )
        rows.extend(project_execution_evidence(evidence))
        executions.append(_manifest_from_evidence(evidence, population=population))
    return rows, executions


def _partition_legitimate_target(
    protocol: V5DevelopmentProtocol,
    *,
    profile: V5Profile,
    partition_name: str,
) -> int:
    if profile is V5Profile.PRODUCTION and partition_name == "development_test":
        return protocol.production_dev_test_legitimate
    counts = protocol.smoke_profile if profile is V5Profile.SMOKE else protocol.production_profile
    if partition_name in {"train", "calibration", "threshold", "development_test"}:
        return counts.legitimate_decisions // 4
    return counts.legitimate_decisions // 8


def _retained_reference_domains(execution: V5ExecutionManifest) -> dict[str, set[str]]:
    """Extract every retained identifier/security reference that must not cross splits."""
    domains = {
        "command": {link.command_id for link in execution.lineage},
        "event": {link.event_id for link in execution.lineage},
        "trust_request": set(execution.trust_request_ids),
        "trace": {link.trace_id for link in execution.lineage},
        "consent": set(),
        "nonce": set(),
        "cart_hash": set(),
        "payment_intent_hash": set(),
        "public_key": set(),
        "receipt_hash": set(),
        "request_hash": set(),
        "signature_hash": set(),
        "prior_receipt_hash": set(),
        "authentication_fact": set(),
    }
    for record in execution.trust_records:
        request = cast(dict[str, object], _decode_canonical_json(record.request_json))
        mandate = cast(dict[str, object], request["mandate"])
        for domain, value in (
            ("consent", request["consent_ref"]),
            ("nonce", request["nonce"]),
            ("cart_hash", request["cart_hash"]),
            ("cart_hash", mandate["cart_hash"]),
            ("payment_intent_hash", request["payment_intent_hash"]),
            ("payment_intent_hash", mandate["payment_intent_hash"]),
            ("public_key", record.public_key_hex),
            ("receipt_hash", record.receipt_hash),
            ("request_hash", record.request_hash),
            ("signature_hash", record.signature_hash),
            ("prior_receipt_hash", request["prior_receipt_hash"]),
        ):
            if type(value) is str and value:
                domains[domain].add(value)
        if record.authentication_evidence_json is not None:
            domains["authentication_fact"].add(
                hashlib.sha256(record.authentication_evidence_json.encode()).hexdigest()
            )
    if execution.trust_registry is not None:
        domains["public_key"].add(execution.trust_registry.public_key_hex)
        domains["authentication_fact"].update(
            hashlib.sha256(value.encode()).hexdigest()
            for value in execution.trust_registry.authentication_evidence_json
        )
    return domains


def _validate_partition_isolation(partitions: dict[str, V5PartitionCorpus]) -> None:
    identity_extractors: dict[str, Callable[[V5DecisionRow], str]] = {
        "actor": lambda row: row.actor_id,
        "counterparty": lambda row: row.counterparty_id,
        "campaign": lambda row: row.campaign_id,
        "payment": lambda row: row.payment_id,
    }
    names = list(_PARTITION_SEED_KEYS)
    for identity_name, extractor in identity_extractors.items():
        values = {
            name: {extractor(row) for row in partitions[name].decisions}
            for name in names
        }
        for index, left in enumerate(names):
            for right in names[index + 1 :]:
                if values[left] & values[right]:
                    raise _PopulationIsolationError(
                        f"{identity_name} identity overlap between {left} and {right}"
                    )
    manifest_domains = (
        "account_ids",
        "device_ids",
        "credential_ids",
        "merchant_ids",
        "payee_ids",
        "agent_ids",
        "key_ids",
        "mandate_ids",
        "authentication_evidence_ids",
    )
    for domain in manifest_domains:
        values = {
            name: {
                identity
                for execution in partitions[name].executions
                for identity in getattr(execution, domain)
            }
            for name in names
        }
        for index, left in enumerate(names):
            for right in names[index + 1 :]:
                if values[left] & values[right]:
                    raise _PopulationIsolationError(
                        f"{domain.removesuffix('_ids')} identity overlap between {left} and {right}"
                    )
    reference_domains = {
        name: {
            reference: {
                value
                for execution in partitions[name].executions
                for value in _retained_reference_domains(execution)[reference]
            }
            for reference in _retained_reference_domains(
                partitions[name].executions[0]
            )
        }
        for name in names
    }
    all_references = set().union(
        *(set(domains) for domains in reference_domains.values())
    )
    for reference in all_references:
        for index, left in enumerate(names):
            for right in names[index + 1 :]:
                if reference_domains[left][reference] & reference_domains[right][reference]:
                    raise _PopulationIsolationError(
                        f"{reference} identity overlap between {left} and {right}"
                    )
    for left, right in zip(names, names[1:], strict=False):
        if max(row.decision_at for row in partitions[left].decisions) >= min(
            row.decision_at for row in partitions[right].decisions
        ):
            raise _PopulationIsolationError(
                f"partition chronology overlaps between {left} and {right}"
            )


def build_v5_corpus(
    protocol: V5DevelopmentProtocol,
    *,
    profile: V5Profile,
) -> V5Corpus:
    """Build every partition with separate benign rows and real executed campaigns."""
    profile_counts = (
        protocol.smoke_profile
        if profile is V5Profile.SMOKE
        else protocol.production_profile
    )
    seed_map = {
        "train": protocol.seeds.train,
        "calibration": protocol.seeds.calibration,
        "threshold": protocol.seeds.threshold,
        "development_test": protocol.seeds.development_test,
        "hardening_train": protocol.seeds.hardening_train,
        "adaptive_holdout": protocol.seeds.adaptive_holdout,
    }
    campaigns_for_profile = profile_counts.campaigns_per_family
    partition_models: dict[str, V5PartitionCorpus] = {}
    for partition_name in _PARTITION_SEED_KEYS:
        seed = seed_map[partition_name]
        rows, executions = _execute_legitimate_traffic(
            partition_name=partition_name,
            partition_seed=seed,
            requested_decisions=_partition_legitimate_target(
                protocol,
                profile=profile,
                partition_name=partition_name,
            ),
        )
        for family in sorted(campaigns_for_profile):
            for campaign_index in range(campaigns_for_profile[family]):
                projected, execution = _execute_campaign(
                    partition_name=partition_name,
                    family=family,
                    campaign_index=campaign_index,
                    partition_seed=seed,
                )
                rows.extend(projected)
                executions.append(execution)
        partition_models[partition_name] = V5PartitionCorpus(
            partition_name=partition_name,
            decisions=tuple(sorted(rows, key=lambda row: (row.decision_at, row.event_id))),
            executions=tuple(executions),
        )

    _validate_partition_isolation(partition_models)
    digest_content = json.dumps(
        {
            name: {
                "decisions": [
                    row.model_dump(mode="json") for row in partition.decisions
                ],
                "executions": [
                    execution.model_dump(mode="json")
                    for execution in partition.executions
                ],
            }
            for name, partition in partition_models.items()
        },
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    corpus_sha256 = hashlib.sha256(digest_content).hexdigest()
    return V5Corpus(
        profile=profile,
        partitions=partition_models,
        corpus_sha256=corpus_sha256,
        is_production=(profile is V5Profile.PRODUCTION),
    )


__all__ = [
    "V5Corpus",
    "V5DecisionRow",
    "V5EventRecord",
    "V5ExecutionManifest",
    "V5LedgerPosting",
    "V5LineageManifest",
    "V5PartitionCorpus",
    "V5TrustRecord",
    "V5TrustRegistry",
    "build_v5_corpus",
]
