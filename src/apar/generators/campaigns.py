"""Causal, bounded generators for the four executable campaign families."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from decimal import ROUND_HALF_EVEN, Decimal
from types import MappingProxyType
from typing import cast
from uuid import NAMESPACE_URL, UUID, uuid5

import numpy as np
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat

from apar.generators.population import Population, PopulationEntity
from apar.simulator.clock import Command
from apar.simulator.rails.a2a import (
    A2ACommand,
    AcceptA2A,
    FreezeA2AFunds,
    InitiateA2A,
    PostA2A,
    RecoverA2A,
    ReportA2AFraud,
    ReturnA2A,
)
from apar.simulator.rails.agentic import AgenticPaymentCommand
from apar.simulator.rails.card import (
    AuthorizeCard,
    CardCommand,
    ChargebackCard,
    ClearCard,
    DeclineCardAuthorization,
    OpenCardDispute,
    RecoverCard,
    RefundCard,
    ReportCardFraud,
    SettleCard,
)
from apar.trust.verifier import (
    AgentMandate,
    AgentPaymentRequest,
    AuthenticationEvidence,
    AuthenticationOutcome,
    AuthenticationRequirement,
)

APP_SCAM_MULE_MOTIF = "a2a:fan_in>layer>fan_out>cash_out"
CARD_TESTING_CNP_MOTIF = "card:probe>success>escalate>burst"
SYNTHETIC_MERCHANT_REFUND_MOTIF = "card:authorize>clear>settle>refund|dispute>chargeback>recovery"
AGENTIC_INTENT_ABUSE_MOTIF = "agentic:valid_control>delegated_binding_mutations>nonce_replay"

_MOTIFS: Mapping[str, str] = MappingProxyType(
    {
        "app_scam_mule": APP_SCAM_MULE_MOTIF,
        "card_testing_cnp": CARD_TESTING_CNP_MOTIF,
        "synthetic_merchant_refund": SYNTHETIC_MERCHANT_REFUND_MOTIF,
        "agentic_intent_abuse": AGENTIC_INTENT_ABUSE_MOTIF,
    }
)
_CENT = Decimal("0.01")


def _text(label: str, value: object) -> str:
    if type(value) is not str:
        raise TypeError(f"{label} must be an exact string")
    if not value:
        raise ValueError(f"{label} must not be empty")
    return value


def _uuid_text(label: str, value: object) -> str:
    text = _text(label, value)
    try:
        identifier = UUID(text)
    except ValueError as error:
        raise ValueError(f"{label} must be a UUID string") from error
    if str(identifier) != text:
        raise ValueError(f"{label} must be a canonical UUID string")
    return text


def _money(label: str, value: object, *, positive: bool) -> Decimal:
    if type(value) is not Decimal:
        raise TypeError(f"{label} must be an exact Decimal")
    amount = value
    if not amount.is_finite() or (amount <= 0 if positive else amount < 0):
        qualifier = "positive" if positive else "non-negative"
        raise ValueError(f"{label} must be finite and {qualifier}")
    if amount != amount.quantize(_CENT, rounding=ROUND_HALF_EVEN):
        raise ValueError(f"{label} must be canonically quantized for USD")
    return amount


def _decimal_unit(label: str, value: object) -> Decimal:
    if type(value) is not Decimal:
        raise TypeError(f"{label} must be an exact Decimal")
    number = value
    if not number.is_finite() or number < 0 or number > 1:
        raise ValueError(f"{label} must be between zero and one")
    return number


@dataclass(frozen=True, slots=True)
class CampaignParams:
    """Visible, bounded campaign variables available to later attacker policies."""

    campaign_id: str
    seed: int
    payment_count: int
    target_illicit_rate: Decimal
    class_rate_tolerance: Decimal
    target_value_total: Decimal
    value_tolerance: Decimal
    min_amount: Decimal
    max_amount: Decimal
    currency: str
    duration_hours: int
    query_budget: int
    min_delay_seconds: int
    max_delay_seconds: int
    expected_motif: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "campaign_id", _uuid_text("campaign_id", self.campaign_id))
        if type(self.seed) is not int:
            raise TypeError("seed must be an exact integer")
        for label in ("payment_count", "duration_hours", "query_budget", "min_delay_seconds"):
            value = getattr(self, label)
            if type(value) is not int or value <= 0:
                raise TypeError(f"{label} must be a positive exact integer")
        if (
            type(self.max_delay_seconds) is not int
            or self.max_delay_seconds < self.min_delay_seconds
        ):
            raise ValueError(
                "max_delay_seconds must be an exact integer at least min_delay_seconds"
            )
        object.__setattr__(
            self,
            "target_illicit_rate",
            _decimal_unit("target_illicit_rate", self.target_illicit_rate),
        )
        tolerance = _decimal_unit("class_rate_tolerance", self.class_rate_tolerance)
        object.__setattr__(self, "class_rate_tolerance", tolerance)
        target = _money("target_value_total", self.target_value_total, positive=True)
        value_tolerance = _money("value_tolerance", self.value_tolerance, positive=False)
        minimum = _money("min_amount", self.min_amount, positive=True)
        maximum = _money("max_amount", self.max_amount, positive=True)
        if maximum < minimum:
            raise ValueError("max_amount must be at least min_amount")
        object.__setattr__(self, "target_value_total", target)
        object.__setattr__(self, "value_tolerance", value_tolerance)
        object.__setattr__(self, "min_amount", minimum)
        object.__setattr__(self, "max_amount", maximum)
        if _text("currency", self.currency) != "USD":
            raise ValueError("campaign generator currently supports USD only")
        object.__setattr__(self, "expected_motif", _text("expected_motif", self.expected_motif))


@dataclass(frozen=True, slots=True)
class AgenticFixture:
    """Public verifier inputs for synthetic requests; the private key never escapes."""

    agent_id: str
    key_id: str
    public_key: bytes
    mandate: AgentMandate
    authentication_evidence: tuple[AuthenticationEvidence, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "agent_id", _text("agent_id", self.agent_id))
        object.__setattr__(self, "key_id", _text("key_id", self.key_id))
        if type(self.public_key) is not bytes or len(self.public_key) != 32:
            raise ValueError("public_key must be 32 exact bytes")
        if type(self.mandate) is not AgentMandate:
            raise TypeError("mandate must be an exact AgentMandate")
        if type(self.authentication_evidence) is not tuple or any(
            type(item) is not AuthenticationEvidence for item in self.authentication_evidence
        ):
            raise TypeError("authentication_evidence must contain exact records")


@dataclass(frozen=True, slots=True)
class CampaignDependency:
    """A downstream payment's explicit dependency on earlier settled inflows."""

    payment_id: str
    upstream_payment_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "payment_id", _text("payment_id", self.payment_id))
        if type(self.upstream_payment_ids) is not tuple or not self.upstream_payment_ids:
            raise TypeError("upstream_payment_ids must be a non-empty exact tuple")
        checked = tuple(
            _text("upstream payment_id", payment_id) for payment_id in self.upstream_payment_ids
        )
        if len(set(checked)) != len(checked):
            raise ValueError("upstream_payment_ids must be unique")


@dataclass(frozen=True, slots=True)
class CampaignEvidence:
    """Deterministic public proof that a generated campaign satisfies its bounds."""

    family: str
    campaign_id: str
    motif_signature: str
    payment_count: int
    command_count: int
    illicit_count: int
    illicit_rate: Decimal
    value_total: Decimal
    schedule: tuple[datetime, ...]
    graph_digest: str
    schedule_digest: str
    declared_entity_ids: tuple[str, ...]
    account_ids: tuple[str, ...]
    class_labels: tuple[bool, ...]
    mutation_kinds: tuple[str, ...]
    dependencies: tuple[CampaignDependency, ...]
    attempts: int
    agentic_fixture: AgenticFixture | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "family", _text("family", self.family))
        object.__setattr__(self, "campaign_id", _uuid_text("campaign_id", self.campaign_id))
        object.__setattr__(
            self,
            "motif_signature",
            _text("motif_signature", self.motif_signature),
        )
        for label in ("payment_count", "command_count", "attempts"):
            value = getattr(self, label)
            if type(value) is not int or value <= 0:
                raise TypeError(f"{label} must be a positive exact integer")
        if self.attempts > 100:
            raise ValueError("attempts must not exceed the rejection budget")
        if type(self.illicit_count) is not int or not 0 <= self.illicit_count <= self.payment_count:
            raise ValueError("illicit_count must be an exact bounded integer")
        _decimal_unit("illicit_rate", self.illicit_rate)
        _money("value_total", self.value_total, positive=True)
        if type(self.schedule) is not tuple or len(self.schedule) != self.command_count:
            raise ValueError("schedule must contain one timestamp per command")
        for timestamp in self.schedule:
            if type(timestamp) is not datetime or timestamp.tzinfo is not UTC:
                raise ValueError("schedule timestamps must be exact UTC datetimes")
        for label in ("graph_digest", "schedule_digest"):
            digest = getattr(self, label)
            if (
                type(digest) is not str
                or len(digest) != 64
                or any(character not in "0123456789abcdef" for character in digest)
            ):
                raise ValueError(f"{label} must be a lowercase SHA-256 digest")
        for label in ("declared_entity_ids", "account_ids", "mutation_kinds"):
            values = getattr(self, label)
            if type(values) is not tuple or any(type(item) is not str for item in values):
                raise TypeError(f"{label} must be an exact tuple of exact strings")
        if type(self.class_labels) is not tuple or any(
            type(item) is not bool for item in self.class_labels
        ):
            raise TypeError("class_labels must be an exact tuple of exact booleans")
        if len(self.class_labels) != self.payment_count:
            raise ValueError("class_labels must contain one value per payment")
        if type(self.dependencies) is not tuple or any(
            type(item) is not CampaignDependency for item in self.dependencies
        ):
            raise TypeError("dependencies must contain exact CampaignDependency records")
        if len({item.payment_id for item in self.dependencies}) != len(self.dependencies):
            raise ValueError("dependency payment IDs must be unique")
        if self.agentic_fixture is not None and type(self.agentic_fixture) is not AgenticFixture:
            raise TypeError("agentic_fixture must be an exact AgenticFixture or None")

    def canonical_bytes(self) -> bytes:
        """Return canonical JSON containing public evidence and public keys only."""
        fixture: dict[str, object] | None = None
        if self.agentic_fixture is not None:
            fixture = {
                "agent_id": self.agentic_fixture.agent_id,
                "key_id": self.agentic_fixture.key_id,
                "public_key": self.agentic_fixture.public_key.hex(),
                "mandate_hash": hashlib.sha256(
                    self.agentic_fixture.mandate.canonical_bytes()
                ).hexdigest(),
                "evidence_ids": [
                    item.evidence_id for item in self.agentic_fixture.authentication_evidence
                ],
            }
        value = {
            "family": self.family,
            "campaign_id": self.campaign_id,
            "motif_signature": self.motif_signature,
            "payment_count": self.payment_count,
            "command_count": self.command_count,
            "illicit_count": self.illicit_count,
            "illicit_rate": str(self.illicit_rate),
            "value_total": str(self.value_total),
            "schedule": [item.isoformat().replace("+00:00", "Z") for item in self.schedule],
            "graph_digest": self.graph_digest,
            "schedule_digest": self.schedule_digest,
            "declared_entity_ids": self.declared_entity_ids,
            "account_ids": self.account_ids,
            "class_labels": self.class_labels,
            "mutation_kinds": self.mutation_kinds,
            "dependencies": [
                {
                    "payment_id": item.payment_id,
                    "upstream_payment_ids": item.upstream_payment_ids,
                }
                for item in self.dependencies
            ],
            "attempts": self.attempts,
            "agentic_fixture": fixture,
        }
        return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


class GenerationConstraintError(RuntimeError):
    """Stable all-or-nothing failure after the fixed rejection budget."""

    def __init__(self, attempts: int = 100) -> None:
        self.code = "GENERATION_CONSTRAINT_UNSATISFIED"
        self.attempts = attempts
        super().__init__(self.code)


@dataclass(frozen=True, slots=True)
class _PaymentPlan:
    payment_id: str
    actor: PopulationEntity
    counterparty: PopulationEntity
    payer_account: str
    payee_account: str
    illicit: bool
    stages: tuple[str, ...]
    mutation_kind: str = ""


def _json_value(value: object) -> object:
    if value is None or type(value) in (bool, int, float, str):
        return value
    if type(value) is Decimal:
        return {"decimal": str(value)}
    if type(value) is bytes:
        return {"bytes": value.hex()}
    if type(value) is datetime:
        return {"datetime": value.isoformat().replace("+00:00", "Z")}
    if isinstance(value, Mapping):
        return {cast(str, key): _json_value(item) for key, item in sorted(value.items())}
    if type(value) in (tuple, list):
        return [_json_value(item) for item in cast(Sequence[object], value)]
    raise TypeError(f"unsupported campaign serialization type: {type(value).__name__}")


def campaign_bytes(commands: tuple[Command, ...]) -> bytes:
    """Canonicalize a command sequence for replay and reproducibility checks."""
    if type(commands) is not tuple or any(not isinstance(command, Command) for command in commands):
        raise TypeError("commands must be an exact tuple of Command values")
    value = [
        {"name": command.name, "payload": _json_value(command.payload)} for command in commands
    ]
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def motif_signature(commands: tuple[Command, ...]) -> str:
    """Recognize a deep family from public lifecycle and graph structure."""
    if type(commands) is not tuple or not commands:
        raise ValueError("commands must be a non-empty exact tuple")
    if all(isinstance(command, A2ACommand) for command in commands):
        opens = [command for command in commands if type(command) is InitiateA2A]
        payees = [cast(str, command.payload["payee_account"]) for command in opens]
        payers = [cast(str, command.payload["payer_account"]) for command in opens]
        if len(set(payees)) < len(payees) and set(payees) & set(payers):
            return APP_SCAM_MULE_MOTIF
    if all(isinstance(command, CardCommand) for command in commands):
        types = {type(command) for command in commands}
        if RefundCard in types and ChargebackCard in types and RecoverCard in types:
            return SYNTHETIC_MERCHANT_REFUND_MOTIF
        if DeclineCardAuthorization in types and SettleCard in types:
            return CARD_TESTING_CNP_MOTIF
    if all(isinstance(command, AgenticPaymentCommand) for command in commands):
        requests = [cast(AgenticPaymentCommand, command).request for command in commands]
        if len({request.nonce for request in requests}) < len(requests):
            return AGENTIC_INTENT_ABUSE_MOTIF
    raise ValueError("campaign commands do not satisfy a supported deep motif")


class CampaignGenerator:
    """Build a graph and legal schedule first, then sample bounded leaf values."""

    def __init__(self, *, seed: int) -> None:
        if type(seed) is not int:
            raise TypeError("seed must be an exact integer")
        self._seed = seed
        self._rng = np.random.default_rng(seed)
        self._last_evidence: CampaignEvidence | None = None

    @property
    def last_evidence(self) -> CampaignEvidence:
        if self._last_evidence is None:
            raise RuntimeError("no successful campaign generation evidence is available")
        return self._last_evidence

    def generate(
        self,
        family: str,
        population: Population,
        params: CampaignParams,
    ) -> tuple[Command, ...]:
        """Return public rail commands or fail atomically after exactly 100 attempts."""
        checked_family = _text("family", family)
        if checked_family not in _MOTIFS:
            raise ValueError(f"unsupported campaign family: {checked_family}")
        if type(population) is not Population:
            raise TypeError("population must be an exact Population")
        if type(params) is not CampaignParams:
            raise TypeError("params must be an exact CampaignParams")
        self._last_evidence = None
        prior_state = deepcopy(self._rng.bit_generator.state)
        campaign_tail = int(params.campaign_id.replace("-", "")[-16:], 16)
        composite_seed = self._seed ^ params.seed ^ campaign_tail
        self._rng = np.random.default_rng(composite_seed)
        for attempt in range(1, 101):
            try:
                plans = self._build_plans(checked_family, population, params)
                schedule = self._build_schedule(
                    checked_family,
                    plans,
                    population,
                    params,
                )
                amounts = self._sample_amounts(params)
                if checked_family == "card_testing_cnp":
                    amounts = tuple(sorted(amounts))
                commands, fixture = self._materialize(
                    checked_family, plans, amounts, population, params
                )
                evidence = self._validate_candidate(
                    checked_family,
                    commands,
                    plans,
                    amounts,
                    schedule,
                    population,
                    params,
                    attempt,
                    fixture,
                )
            except (ArithmeticError, TypeError, ValueError):
                continue
            self._last_evidence = evidence
            return commands
        self._rng.bit_generator.state = prior_state
        raise GenerationConstraintError(100)

    def _build_plans(
        self,
        family: str,
        population: Population,
        params: CampaignParams,
    ) -> tuple[_PaymentPlan, ...]:
        namespace = uuid5(NAMESPACE_URL, f"apar:campaign:{params.campaign_id}:{params.seed}")
        illicit_count = round(Decimal(params.payment_count) * params.target_illicit_rate)
        labels = tuple(index < illicit_count for index in range(params.payment_count))
        victims = self._accounts(population, "victim", "consumer", "organization")
        merchants = self._accounts(population, "merchant", "beneficiary")
        mules = self._accounts(population, "mule")
        attackers = self._accounts(population, "attacker", "synthetic_merchant")
        agents = self._accounts(population, "agent")
        synthetic_merchants = self._accounts(population, "synthetic_merchant")
        plans: list[_PaymentPlan] = []

        for index in range(params.payment_count):
            payment_id = f"{family}:{uuid5(namespace, f'payment:{index}')}"
            illicit = labels[index]
            if family == "app_scam_mule":
                incoming_count = max(2, (params.payment_count * 2) // 3)
                if index < incoming_count:
                    actor = victims[index % len(victims)]
                    counterparty = mules[0]
                    stages: tuple[str, ...] = ("initiate", "accept", "post")
                    if index == 0:
                        stages += ("report", "freeze", "recover")
                else:
                    actor = mules[0]
                    counterparty = attackers[index % len(attackers)]
                    stages = ("initiate", "accept", "post")
                    if index == params.payment_count - 1:
                        stages += ("return",)
            elif family == "card_testing_cnp":
                actor = victims[index % min(2, len(victims))]
                counterparty = merchants[0]
                stages = (
                    ("decline",)
                    if index < max(2, params.payment_count // 3)
                    else (
                        "authorize",
                        "clear",
                        "settle",
                    )
                )
            elif family == "synthetic_merchant_refund":
                actor = victims[index % len(victims)]
                counterparty = synthetic_merchants[0]
                stages = (
                    ("authorize", "clear", "settle", "refund")
                    if index % 2 == 0
                    else (
                        "authorize",
                        "clear",
                        "settle",
                        "report",
                        "dispute",
                        "chargeback",
                        "recover",
                    )
                )
            else:
                mutation_cycle = (
                    "valid_control",
                    "consent_scope",
                    "merchant",
                    "payee",
                    "cart",
                    "intent",
                    "credential",
                    "authentication_evidence",
                    "valid_control",
                    "nonce_replay",
                )
                kind = mutation_cycle[index % len(mutation_cycle)]
                actor = agents[0]
                counterparty = attackers[0] if kind == "payee" else merchants[0]
                plans.append(
                    _PaymentPlan(
                        payment_id,
                        actor,
                        counterparty,
                        cast(str, actor.account_id),
                        cast(str, counterparty.account_id),
                        illicit,
                        ("agentic.pay",),
                        kind,
                    )
                )
                continue
            plans.append(
                _PaymentPlan(
                    payment_id,
                    actor,
                    counterparty,
                    cast(str, actor.account_id),
                    cast(str, counterparty.account_id),
                    illicit,
                    stages,
                )
            )
        return tuple(plans)

    @staticmethod
    def _accounts(population: Population, *roles: str) -> tuple[PopulationEntity, ...]:
        entities = tuple(
            entity
            for entity in population.entities
            if entity.role in roles and entity.account_id is not None
        )
        if not entities:
            raise ValueError(f"population lacks required roles: {roles}")
        return entities

    def _build_schedule(
        self,
        family: str,
        plans: tuple[_PaymentPlan, ...],
        population: Population,
        params: CampaignParams,
    ) -> tuple[datetime, ...]:
        schedule: list[datetime] = []
        current = population.generated_at
        for plan in plans:
            for _stage in plan.stages:
                delay_low = params.min_delay_seconds
                delay_high = params.max_delay_seconds
                if family == "card_testing_cnp":
                    span = params.max_delay_seconds - params.min_delay_seconds
                    if plan.stages[0] == "decline":
                        delay_low = params.min_delay_seconds + (span * 2) // 3
                    else:
                        delay_high = params.min_delay_seconds + max(1, span // 4)
                delay = int(
                    self._rng.integers(
                        delay_low,
                        delay_high + 1,
                    )
                )
                current += timedelta(seconds=delay)
                schedule.append(current)
        return tuple(schedule)

    def _sample_amounts(self, params: CampaignParams) -> tuple[Decimal, ...]:
        low = int(params.min_amount / _CENT)
        high = int(params.max_amount / _CENT)
        target = int(params.target_value_total / _CENT)
        raw = cast(
            np.ndarray[tuple[int], np.dtype[np.int64]],
            self._rng.integers(low, high + 1, size=params.payment_count),
        )
        current = int(raw.sum())
        remainder = target - current
        order = [int(item) for item in self._rng.permutation(params.payment_count)]
        while remainder and order:
            progressed = False
            for index in order:
                if remainder > 0:
                    room = high - int(raw[index])
                    change = min(room, remainder)
                else:
                    room = int(raw[index]) - low
                    change = -min(room, -remainder)
                if change:
                    raw[index] += change
                    remainder -= change
                    progressed = True
                if remainder == 0:
                    break
            if not progressed:
                break
        return tuple((Decimal(int(value)) * _CENT).quantize(_CENT) for value in raw)

    def _materialize(
        self,
        family: str,
        plans: tuple[_PaymentPlan, ...],
        amounts: tuple[Decimal, ...],
        population: Population,
        params: CampaignParams,
    ) -> tuple[tuple[Command, ...], AgenticFixture | None]:
        if family == "agentic_intent_abuse":
            return self._materialize_agentic(plans, amounts, population, params)
        commands: list[Command] = []
        for index, (plan, amount) in enumerate(zip(plans, amounts, strict=True)):
            trace_id = self._derived_uuid(params, f"trace:{index}")
            if family == "app_scam_mule":
                commands.extend(self._a2a_commands(plan, amount, params, trace_id))
            else:
                commands.extend(self._card_commands(plan, amount, params, trace_id))
        return tuple(commands), None

    @staticmethod
    def _a2a_commands(
        plan: _PaymentPlan,
        amount: Decimal,
        params: CampaignParams,
        trace_id: str,
    ) -> tuple[Command, ...]:
        opening = InitiateA2A(
            plan.payment_id,
            amount=amount,
            currency=params.currency,
            payer_account=plan.payer_account,
            payee_account=plan.payee_account,
            actor_id=plan.actor.entity_id,
            counterparty_id=plan.counterparty.entity_id,
            campaign_id=params.campaign_id,
            trace_id=trace_id,
        )

        def followup(stage: str) -> A2ACommand:
            key = f"a2a.{stage}:{plan.payment_id}:campaign:{params.campaign_id}"
            if stage == "accept":
                return AcceptA2A(plan.payment_id, idempotency_key=key)
            if stage == "post":
                return PostA2A(plan.payment_id, idempotency_key=key)
            if stage == "report":
                return ReportA2AFraud(plan.payment_id, idempotency_key=key)
            if stage == "freeze":
                return FreezeA2AFunds(plan.payment_id, idempotency_key=key)
            if stage == "recover":
                return RecoverA2A(plan.payment_id, idempotency_key=key)
            if stage == "return":
                return ReturnA2A(plan.payment_id, idempotency_key=key)
            raise ValueError(f"unsupported A2A stage: {stage}")

        return (opening, *tuple(followup(stage) for stage in plan.stages[1:]))

    @staticmethod
    def _card_commands(
        plan: _PaymentPlan,
        amount: Decimal,
        params: CampaignParams,
        trace_id: str,
    ) -> tuple[Command, ...]:
        opening_type = DeclineCardAuthorization if plan.stages[0] == "decline" else AuthorizeCard
        opening = opening_type(
            plan.payment_id,
            amount=amount,
            currency=params.currency,
            payer_account=plan.payer_account,
            payee_account=plan.payee_account,
            actor_id=plan.actor.entity_id,
            counterparty_id=plan.counterparty.entity_id,
            campaign_id=params.campaign_id,
            trace_id=trace_id,
        )

        def followup(stage: str) -> CardCommand:
            key = f"card.{stage}:{plan.payment_id}:campaign:{params.campaign_id}"
            if stage == "clear":
                return ClearCard(plan.payment_id, idempotency_key=key)
            if stage == "settle":
                return SettleCard(plan.payment_id, idempotency_key=key)
            if stage == "refund":
                return RefundCard(plan.payment_id, idempotency_key=key)
            if stage == "report":
                return ReportCardFraud(plan.payment_id, idempotency_key=key)
            if stage == "dispute":
                return OpenCardDispute(plan.payment_id, idempotency_key=key)
            if stage == "chargeback":
                return ChargebackCard(plan.payment_id, idempotency_key=key)
            if stage == "recover":
                return RecoverCard(plan.payment_id, idempotency_key=key)
            raise ValueError(f"unsupported card stage: {stage}")

        return (opening, *tuple(followup(stage) for stage in plan.stages[1:]))

    def _materialize_agentic(
        self,
        plans: tuple[_PaymentPlan, ...],
        amounts: tuple[Decimal, ...],
        population: Population,
        params: CampaignParams,
    ) -> tuple[tuple[Command, ...], AgenticFixture]:
        private_key = Ed25519PrivateKey.from_private_bytes(self._rng.bytes(32))
        public_key = private_key.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw)
        agent_id = f"synthetic-agent-{plans[0].actor.entity_id}"
        key_id = "synthetic-key-1"
        merchant = next(
            entity
            for entity in population.entities
            if entity.role == "merchant" and entity.account_id
        )
        mandate = AgentMandate(
            mandate_id=f"mandate-{params.campaign_id}",
            version=1,
            agent_id=agent_id,
            user_ref=plans[0].payer_account,
            user_entity_id=plans[0].actor.entity_id,
            beneficiary_entity_id=merchant.entity_id,
            consent_ref="synthetic-consent-1",
            merchant_id=merchant.entity_id,
            payee_id=cast(str, merchant.account_id),
            cart_hash=hashlib.sha256(b"synthetic-cart-v1").hexdigest(),
            payment_intent_hash=hashlib.sha256(b"synthetic-intent-v1").hexdigest(),
            permitted_categories=("TRAVEL",),
            permitted_products=("synthetic-flight",),
            credential_id="synthetic-token-1",
            credential_scope="single_merchant_single_use",
            required_authentication=AuthenticationRequirement.STEP_UP,
            max_amount=params.max_amount,
            currency=params.currency,
            issued_at=population.generated_at - timedelta(hours=1),
            expires_at=population.generated_at + timedelta(hours=params.duration_hours + 1),
        )
        commands: list[Command] = []
        evidence: list[AuthenticationEvidence] = []
        first_request: AgentPaymentRequest | None = None
        for index, (plan, amount) in enumerate(zip(plans, amounts, strict=True)):
            if plan.mutation_kind == "nonce_replay" and first_request is not None:
                commands.append(
                    AgenticPaymentCommand(
                        first_request,
                        payer_account=first_request.mandate.user_ref,
                        payee_account=first_request.payee_id,
                    )
                )
                continue
            request_id = f"agentic-request-{index}"
            nonce = f"synthetic-nonce-{index}"
            merchant_id = mandate.merchant_id
            payee_id = mandate.payee_id
            cart_hash = mandate.cart_hash
            intent_hash = mandate.payment_intent_hash
            credential_id = mandate.credential_id
            credential_scope = mandate.credential_scope
            consent_ref = mandate.consent_ref
            auth_ref: str | None = f"synthetic-auth-{index}"
            effective_amount = amount
            if plan.mutation_kind == "consent_scope":
                consent_ref = "synthetic-consent-substituted"
            elif plan.mutation_kind == "merchant":
                merchant_id = plan.counterparty.entity_id
            elif plan.mutation_kind == "payee":
                payee_id = plan.payee_account
            elif plan.mutation_kind == "cart":
                cart_hash = hashlib.sha256(f"mutated-cart-{index}".encode()).hexdigest()
            elif plan.mutation_kind == "intent":
                intent_hash = hashlib.sha256(f"mutated-intent-{index}".encode()).hexdigest()
            elif plan.mutation_kind == "credential":
                credential_id = "synthetic-token-substituted"
                credential_scope = "unbounded"
            elif plan.mutation_kind == "authentication_evidence":
                auth_ref = "synthetic-auth-missing"
            unsigned = AgentPaymentRequest(
                request_id=request_id,
                payment_id=plan.payment_id,
                agent_id=agent_id,
                key_id=key_id,
                mandate=mandate,
                amount=effective_amount,
                currency=params.currency,
                merchant_id=merchant_id,
                payee_id=payee_id,
                cart_hash=cart_hash,
                payment_intent_hash=intent_hash,
                category="TRAVEL",
                product_id="synthetic-flight",
                credential_id=credential_id,
                credential_scope=credential_scope,
                consent_ref=consent_ref,
                authentication_evidence_ref=auth_ref,
                nonce=nonce,
                created_at=population.generated_at + timedelta(minutes=index),
                expires_at=population.generated_at + timedelta(minutes=index + 5),
                prior_receipt_hash="",
                campaign_id=params.campaign_id,
                trace_id=self._derived_uuid(params, f"agentic-trace:{index}"),
                actor_id=plan.actor.entity_id,
                counterparty_id=plan.counterparty.entity_id,
                signature=b"",
            )
            request = unsigned.model_copy(
                update={"signature": private_key.sign(unsigned.signing_bytes())}
            )
            if first_request is None:
                first_request = request
            if auth_ref is not None and plan.mutation_kind != "authentication_evidence":
                evidence.append(
                    AuthenticationEvidence(
                        evidence_id=auth_ref,
                        agent_id=agent_id,
                        user_ref=mandate.user_ref,
                        mandate_id=mandate.mandate_id,
                        nonce=nonce,
                        payment_intent_hash=intent_hash,
                        request_id=request_id,
                        outcome=AuthenticationOutcome.STEP_UP_VERIFIED,
                        issued_at=unsigned.created_at - timedelta(seconds=5),
                        expires_at=unsigned.created_at + timedelta(minutes=2),
                    )
                )
            commands.append(
                AgenticPaymentCommand(
                    request,
                    payer_account=request.mandate.user_ref,
                    payee_account=request.payee_id,
                )
            )
        return tuple(commands), AgenticFixture(
            agent_id,
            key_id,
            public_key,
            mandate,
            tuple(evidence),
        )

    def _validate_candidate(
        self,
        family: str,
        commands: tuple[Command, ...],
        plans: tuple[_PaymentPlan, ...],
        amounts: tuple[Decimal, ...],
        schedule: tuple[datetime, ...],
        population: Population,
        params: CampaignParams,
        attempt: int,
        fixture: AgenticFixture | None,
    ) -> CampaignEvidence:
        if (
            params.expected_motif != _MOTIFS[family]
            or motif_signature(commands) != params.expected_motif
        ):
            raise ValueError("motif constraint not satisfied")
        rate = Decimal(sum(plan.illicit for plan in plans)) / Decimal(len(plans))
        value_total = sum(amounts, Decimal("0.00"))
        if abs(rate - params.target_illicit_rate) > params.class_rate_tolerance:
            raise ValueError("class rate constraint not satisfied")
        if abs(value_total - params.target_value_total) > params.value_tolerance:
            raise ValueError("value total constraint not satisfied")
        if any(amount < params.min_amount or amount > params.max_amount for amount in amounts):
            raise ValueError("amount bounds not satisfied")
        if len(schedule) != len(commands) or schedule != tuple(sorted(schedule)):
            raise ValueError("command schedule is not total and ordered")
        if not schedule or schedule[-1] - population.generated_at > timedelta(
            hours=params.duration_hours
        ):
            raise ValueError("timestamp horizon not satisfied")
        declared = {entity.entity_id for entity in population.entities}
        referenced = {plan.actor.entity_id for plan in plans} | {
            plan.counterparty.entity_id for plan in plans
        }
        accounts = {plan.payer_account for plan in plans} | {plan.payee_account for plan in plans}
        for command in commands:
            for key, value in command.payload.items():
                if key.endswith("_account") and type(value) is str:
                    accounts.add(value)
        if not referenced <= declared or not accounts <= set(population.opening_balances):
            raise ValueError("entity or account references are not population-owned")
        self._validate_lifecycle(plans, commands)
        dependencies = self._dependencies(family, plans)
        graph_document = [
            [
                plan.payment_id,
                plan.actor.entity_id,
                plan.counterparty.entity_id,
                plan.payer_account,
                plan.payee_account,
                str(plan.illicit),
            ]
            for plan in plans
        ]
        graph_document.extend(
            [dependency.payment_id, *dependency.upstream_payment_ids] for dependency in dependencies
        )
        schedule_document = [
            [command.name, timestamp.isoformat().replace("+00:00", "Z")]
            for command, timestamp in zip(commands, schedule, strict=True)
        ]
        return CampaignEvidence(
            family=family,
            campaign_id=params.campaign_id,
            motif_signature=params.expected_motif,
            payment_count=len(plans),
            command_count=len(commands),
            illicit_count=sum(plan.illicit for plan in plans),
            illicit_rate=rate,
            value_total=value_total,
            schedule=schedule,
            graph_digest=hashlib.sha256(
                json.dumps(graph_document, separators=(",", ":")).encode()
            ).hexdigest(),
            schedule_digest=hashlib.sha256(
                json.dumps(schedule_document, separators=(",", ":")).encode()
            ).hexdigest(),
            declared_entity_ids=tuple(sorted(referenced)),
            account_ids=tuple(sorted(accounts)),
            class_labels=tuple(plan.illicit for plan in plans),
            mutation_kinds=tuple(
                dict.fromkeys(plan.mutation_kind for plan in plans if plan.mutation_kind)
            ),
            dependencies=dependencies,
            attempts=attempt,
            agentic_fixture=fixture,
        )

    @staticmethod
    def _validate_lifecycle(plans: tuple[_PaymentPlan, ...], commands: tuple[Command, ...]) -> None:
        expected = [
            stage.split(".", 1)[1] if stage.startswith("agentic.") else stage
            for plan in plans
            for stage in plan.stages
        ]
        actual = [command.name.split(".", 1)[1] for command in commands]
        if actual != expected:
            raise ValueError("rail lifecycle order does not match the causal schedule")

    @staticmethod
    def _dependencies(
        family: str,
        plans: tuple[_PaymentPlan, ...],
    ) -> tuple[CampaignDependency, ...]:
        if family != "app_scam_mule":
            return ()
        dependencies: list[CampaignDependency] = []
        for index, plan in enumerate(plans):
            upstream = tuple(
                candidate.payment_id
                for candidate in plans[:index]
                if candidate.payee_account == plan.payer_account and "post" in candidate.stages
            )
            if upstream:
                dependencies.append(CampaignDependency(plan.payment_id, upstream))
        return tuple(dependencies)

    @staticmethod
    def _derived_uuid(params: CampaignParams, label: str) -> str:
        namespace = uuid5(
            NAMESPACE_URL,
            f"apar:campaign-identifiers:{params.campaign_id}:{params.seed}",
        )
        return str(uuid5(namespace, label))


__all__ = [
    "AGENTIC_INTENT_ABUSE_MOTIF",
    "APP_SCAM_MULE_MOTIF",
    "CARD_TESTING_CNP_MOTIF",
    "SYNTHETIC_MERCHANT_REFUND_MOTIF",
    "AgenticFixture",
    "CampaignDependency",
    "CampaignEvidence",
    "CampaignGenerator",
    "CampaignParams",
    "GenerationConstraintError",
    "campaign_bytes",
    "motif_signature",
]
