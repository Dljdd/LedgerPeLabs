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

from apar.contracts.decisions import Action, ReasonCode
from apar.contracts.events import EventKind, PaymentEvent, Rail
from apar.generators.population import Population, PopulationEntity
from apar.simulator.clock import Command
from apar.simulator.engine import SimulationEngine
from apar.simulator.ledger import AccountReference
from apar.simulator.rails import A2ARailAdapter, AgenticRailAdapter, CardRailAdapter
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
from apar.simulator.rails.base import AdapterFactory
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
    ReceiptOutcome,
    TrustVerifier,
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
_AGENTIC_MUTATIONS = (
    "identity",
    "signature",
    "mandate",
    "authority_identity",
    "amount",
    "currency",
    "merchant",
    "payee",
    "category",
    "product",
    "cart",
    "intent",
    "credential",
    "token_scope",
    "consent",
    "mandate_time",
    "expiry",
    "auth_missing",
    "auth_mismatch",
    "auth_expired",
    "nonce_replay",
    "receipt_chain",
    "auth_replay",
)


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


class CampaignParameterError(ValueError):
    """Stable rejection for undeclared, ill-typed, or unbounded policy inputs."""

    def __init__(self, message: str) -> None:
        self.code = "CAMPAIGN_PARAMETER_INVALID"
        super().__init__(f"{self.code}: {message}")


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
    merchant_concentration: Decimal = Decimal("0.70")
    device_reuse_rate: Decimal = Decimal("0.60")
    retry_intensity: int = 2
    mule_count: int = 2
    mule_layers: int = 1
    mule_fanout: int = 2
    cash_out_fraction: Decimal = Decimal("0.30")
    cash_out_strategy: str = "staged"
    cash_out_delay_seconds: int = 600
    recovery_probability: Decimal = Decimal("0.25")
    agentic_mutations: tuple[str, ...] = _AGENTIC_MUTATIONS
    agentic_attack_mix: Decimal = Decimal("0.92")

    def __post_init__(self) -> None:
        try:
            object.__setattr__(
                self,
                "campaign_id",
                _uuid_text("campaign_id", self.campaign_id),
            )
            if type(self.seed) is not int:
                raise TypeError("seed must be an exact integer")
            integer_bounds = {
                "payment_count": (1, 256),
                "duration_hours": (1, 720),
                "query_budget": (1, 1000),
                "min_delay_seconds": (1, 3600),
                "max_delay_seconds": (1, 3600),
                "retry_intensity": (0, 10),
                "mule_count": (2, 16),
                "mule_layers": (1, 5),
                "mule_fanout": (1, 16),
                "cash_out_delay_seconds": (1, 86_400),
            }
            for label, (minimum_bound, maximum_bound) in integer_bounds.items():
                value = getattr(self, label)
                if type(value) is not int or not minimum_bound <= value <= maximum_bound:
                    raise TypeError(
                        f"{label} must be an exact integer in "
                        f"[{minimum_bound}, {maximum_bound}]"
                    )
            if self.max_delay_seconds < self.min_delay_seconds:
                raise ValueError("max_delay_seconds must be at least min_delay_seconds")
            for label in (
                "target_illicit_rate",
                "class_rate_tolerance",
                "merchant_concentration",
                "device_reuse_rate",
                "cash_out_fraction",
                "recovery_probability",
                "agentic_attack_mix",
            ):
                object.__setattr__(self, label, _decimal_unit(label, getattr(self, label)))
            target = _money("target_value_total", self.target_value_total, positive=True)
            value_tolerance = _money("value_tolerance", self.value_tolerance, positive=False)
            minimum = _money("min_amount", self.min_amount, positive=True)
            maximum = _money("max_amount", self.max_amount, positive=True)
            if target > Decimal("1000000.00"):
                raise ValueError("target_value_total exceeds the visible cap")
            if maximum > Decimal("100000.00") or maximum < minimum:
                raise ValueError("amount bounds are invalid or exceed the visible cap")
            object.__setattr__(self, "target_value_total", target)
            object.__setattr__(self, "value_tolerance", value_tolerance)
            object.__setattr__(self, "min_amount", minimum)
            object.__setattr__(self, "max_amount", maximum)
            if _text("currency", self.currency) != "USD":
                raise ValueError("campaign generator currently supports USD only")
            object.__setattr__(
                self,
                "expected_motif",
                _text("expected_motif", self.expected_motif),
            )
            if self.cash_out_strategy not in {"staged", "burst", "delayed"}:
                raise ValueError("cash_out_strategy is undeclared")
            if type(self.agentic_mutations) is not tuple or any(
                type(value) is not str for value in self.agentic_mutations
            ):
                raise TypeError("agentic_mutations must be an exact tuple of exact strings")
            if (
                len(set(self.agentic_mutations)) != len(self.agentic_mutations)
                or not set(self.agentic_mutations) <= set(_AGENTIC_MUTATIONS)
            ):
                raise ValueError("agentic_mutations contains duplicates or undeclared values")
        except (TypeError, ValueError) as error:
            if isinstance(error, CampaignParameterError):
                raise
            raise CampaignParameterError(str(error)) from error

    @classmethod
    def from_mapping(cls, values: Mapping[str, object]) -> CampaignParams:
        """Validate policy output without accepting undeclared dimensions."""
        if type(values) is not dict:
            raise CampaignParameterError("campaign parameters must be an exact mapping")
        declared = set(cls.__dataclass_fields__)
        unknown = set(values) - declared
        if unknown:
            raise CampaignParameterError(f"undeclared fields: {sorted(unknown)}")
        try:
            return cls(**values)
        except TypeError as error:
            raise CampaignParameterError(str(error)) from error


@dataclass(frozen=True, slots=True)
class AgenticFixture:
    """Evaluator-owned verifier inputs; the synthetic private key never escapes."""

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
    """Evaluator evidence linking downstream payments to settled inflows."""

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
    """Evaluator-owned proof that a generated campaign satisfies concrete bounds."""

    family: str
    campaign_id: str
    motif_signature: str
    payment_count: int
    command_count: int
    illicit_count: int
    illicit_rate: Decimal
    value_total: Decimal
    attempted_value: Decimal
    unique_attempted_value: Decimal
    settled_value: Decimal
    schedule: tuple[datetime, ...]
    graph_digest: str
    schedule_digest: str
    declared_entity_ids: tuple[str, ...]
    account_ids: tuple[str, ...]
    class_labels: tuple[bool, ...]
    mutation_kinds: tuple[str, ...]
    dependencies: tuple[CampaignDependency, ...]
    observed_reasons: tuple[ReasonCode | None, ...]
    valid_control_count: int
    replay_succeeded: bool
    ledger_conserved: bool
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
        _money("attempted_value", self.attempted_value, positive=True)
        _money("unique_attempted_value", self.unique_attempted_value, positive=True)
        _money("settled_value", self.settled_value, positive=False)
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
        if type(self.observed_reasons) is not tuple or any(
            reason is not None and type(reason) is not ReasonCode
            for reason in self.observed_reasons
        ):
            raise TypeError("observed_reasons must contain exact ReasonCode values or None")
        if type(self.valid_control_count) is not int or self.valid_control_count < 0:
            raise TypeError("valid_control_count must be a non-negative exact integer")
        if type(self.replay_succeeded) is not bool or type(self.ledger_conserved) is not bool:
            raise TypeError("replay flags must be exact booleans")
        if self.agentic_fixture is not None and type(self.agentic_fixture) is not AgenticFixture:
            raise TypeError("agentic_fixture must be an exact AgenticFixture or None")

    def canonical_bytes(self) -> bytes:
        """Return canonical evaluator JSON containing no private signing key."""
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
            "attempted_value": str(self.attempted_value),
            "unique_attempted_value": str(self.unique_attempted_value),
            "settled_value": str(self.settled_value),
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
            "observed_reasons": [
                reason.value if reason is not None else None
                for reason in self.observed_reasons
            ],
            "valid_control_count": self.valid_control_count,
            "replay_succeeded": self.replay_succeeded,
            "ledger_conserved": self.ledger_conserved,
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
        histories = _operation_histories(commands)
        if not opens or any(
            operations[:3] != ("initiate", "accept", "post")
            for operations in histories.values()
        ):
            raise ValueError("A2A motif requires complete initiated-to-posted lifecycles")
        payees = [cast(str, command.payload["payee_account"]) for command in opens]
        payers = [cast(str, command.payload["payer_account"]) for command in opens]
        bridges = set(payees) & set(payers)
        layer_edges = [
            command
            for command in opens
            if command.payload["payer_account"] in bridges
            and command.payload["payee_account"] in bridges
        ]
        fan_in = max((payees.count(account) for account in bridges), default=0)
        fan_out = max((payers.count(account) for account in bridges), default=0)
        if len(bridges) >= 2 and layer_edges and fan_in >= 2 and fan_out >= 2:
            return APP_SCAM_MULE_MOTIF
    if all(isinstance(command, CardCommand) for command in commands):
        histories = _operation_histories(commands)
        card_opens = [
            cast(CardCommand, command)
            for command in commands
            if type(command) in {AuthorizeCard, DeclineCardAuthorization}
        ]
        if not card_opens:
            raise ValueError("card motif requires opening attempts")
        operations = tuple(histories.values())
        refund_path = ("authorize", "clear", "settle", "refund")
        chargeback_path = (
            "authorize",
            "clear",
            "settle",
            "report",
            "dispute",
            "chargeback",
            "recover",
        )
        if refund_path in operations and chargeback_path in operations and all(
            path in {refund_path, chargeback_path} for path in operations
        ):
            return SYNTHETIC_MERCHANT_REFUND_MOTIF
        declines = [
            command
            for command in card_opens
            if type(command) is DeclineCardAuthorization
        ]
        successes = [
            command
            for command in card_opens
            if type(command) is AuthorizeCard
        ]
        success_path = ("authorize", "clear", "settle")
        if (
            declines
            and successes
            and all(histories[command.payment_id] == ("decline",) for command in declines)
            and all(histories[command.payment_id] == success_path for command in successes)
            and max(cast(Decimal, command.payload["amount"]) for command in declines)
            < min(cast(Decimal, command.payload["amount"]) for command in successes)
            and len({command.payload["payer_account"] for command in card_opens}) >= 2
            and len({command.payload["payee_account"] for command in card_opens})
            < len(card_opens)
        ):
            return CARD_TESTING_CNP_MOTIF
    if all(isinstance(command, AgenticPaymentCommand) for command in commands):
        requests = [cast(AgenticPaymentCommand, command).request for command in commands]
        if (
            len(requests) >= 3
            and len({request.request_id for request in requests}) == len(requests)
            and len({request.nonce for request in requests}) < len(requests)
            and len({request.signature for request in requests}) == len(requests)
        ):
            return AGENTIC_INTENT_ABUSE_MOTIF
    raise ValueError("campaign commands do not satisfy a supported deep motif")


def _operation_histories(commands: tuple[Command, ...]) -> dict[str, tuple[str, ...]]:
    histories: dict[str, list[str]] = {}
    for command in commands:
        payment_id = getattr(command, "payment_id", None)
        if type(payment_id) is not str:
            raise ValueError("campaign command lacks a payment_id")
        operation = command.name.split(".", 1)[1]
        histories.setdefault(payment_id, []).append(operation)
    return {payment_id: tuple(operations) for payment_id, operations in histories.items()}


class CampaignGenerator:
    """Build a graph and legal schedule first, then sample bounded leaf values."""

    def __init__(self, *, seed: int) -> None:
        if type(seed) is not int:
            raise TypeError("seed must be an exact integer")
        self._seed = seed
        self._rng = np.random.default_rng(seed)

    def generate(
        self,
        family: str,
        population: Population,
        params: CampaignParams,
    ) -> tuple[Command, ...]:
        """Return only commands; evaluator audit data is deliberately discarded."""
        commands, _audit = self._generate_audited(family, population, params)
        return commands

    def _generate_audited(
        self,
        family: str,
        population: Population,
        params: CampaignParams,
    ) -> tuple[tuple[Command, ...], CampaignEvidence]:
        """Evaluator-owned generation path, outside the policy-facing surface."""
        checked_family = _text("family", family)
        if checked_family not in _MOTIFS:
            raise ValueError(f"unsupported campaign family: {checked_family}")
        if type(population) is not Population:
            raise TypeError("population must be an exact Population")
        if type(params) is not CampaignParams:
            raise TypeError("params must be an exact CampaignParams")
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
                if checked_family == "app_scam_mule":
                    amounts = self._app_amounts(plans, params)
                elif checked_family == "card_testing_cnp":
                    amounts = tuple(sorted(amounts))
                elif checked_family == "agentic_intent_abuse":
                    amounts = self._agentic_amounts(plans, params)
                commands, fixture = self._materialize(
                    checked_family, plans, amounts, schedule, population, params
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
            except (ArithmeticError, RuntimeError, TypeError, ValueError):
                continue
            return commands, evidence
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
        attackers = self._accounts(population, "attacker")
        agents = self._accounts(population, "agent")
        synthetic_merchants = self._accounts(population, "synthetic_merchant")
        plans: list[_PaymentPlan] = []

        for index in range(params.payment_count):
            payment_id = f"{family}:{uuid5(namespace, f'payment:{index}')}"
            illicit = labels[index]
            if family == "app_scam_mule":
                attack_count = illicit_count
                selected_mules = mules[: params.mule_count]
                layer_count = params.mule_layers
                fanout_count = params.mule_fanout
                incoming_count = attack_count - layer_count - fanout_count
                if (
                    len(selected_mules) < layer_count + 1
                    or incoming_count < 2
                    or attack_count > params.payment_count
                ):
                    raise ValueError("APP motif cannot satisfy declared mule topology")
                if index < incoming_count:
                    actor = victims[index % len(victims)]
                    counterparty = selected_mules[0]
                    stages: tuple[str, ...] = ("initiate", "accept", "post")
                elif index < incoming_count + layer_count:
                    layer_index = index - incoming_count
                    actor = selected_mules[layer_index]
                    counterparty = selected_mules[layer_index + 1]
                    stages = ("initiate", "accept", "post")
                elif index < attack_count:
                    fanout_index = index - incoming_count - layer_count
                    actor = selected_mules[
                        fanout_index % (layer_count + 1)
                    ]
                    counterparty = attackers[index % len(attackers)]
                    stages = ("initiate", "accept", "post")
                    if index == attack_count - 1:
                        stages += ("report", "freeze", "recover")
                else:
                    actor = victims[index % len(victims)]
                    counterparty = merchants[index % len(merchants)]
                    stages = ("initiate", "accept", "post", "return")
            elif family == "card_testing_cnp":
                if illicit:
                    actor_span = max(
                        2,
                        min(
                            len(attackers),
                            round(
                                2
                                + (len(attackers) - 2)
                                * float(1 - params.device_reuse_rate)
                            ),
                        ),
                    )
                    actor = attackers[index % actor_span]
                    concentrated = max(
                        1,
                        round(
                            len(merchants)
                            * float(1 - params.merchant_concentration)
                        ),
                    )
                    counterparty = merchants[index % concentrated]
                    decline_count = max(
                        1,
                        min(illicit_count - 1, max(1, params.retry_intensity)),
                    )
                    stages = (
                        ("decline",)
                        if index < decline_count
                        else ("authorize", "clear", "settle")
                    )
                else:
                    actor = victims[index % len(victims)]
                    counterparty = merchants[
                        (index - illicit_count) % len(merchants)
                    ]
                    stages = ("authorize", "clear", "settle")
            elif family == "synthetic_merchant_refund":
                actor = (
                    attackers[index % len(attackers)]
                    if illicit
                    else victims[index % len(victims)]
                )
                counterparty = (
                    synthetic_merchants[0] if illicit else merchants[index % len(merchants)]
                )
                if illicit:
                    recovered_count = max(
                        1,
                        min(
                            illicit_count - 1,
                            round(
                                Decimal(illicit_count)
                                * params.recovery_probability
                            ),
                        ),
                    )
                    stages = (
                        (
                            "authorize",
                            "clear",
                            "settle",
                            "report",
                            "dispute",
                            "chargeback",
                            "recover",
                        )
                        if index < recovered_count
                        else ("authorize", "clear", "settle", "refund")
                    )
                else:
                    stages = ("authorize", "clear", "settle", "refund")
            else:
                isolated = tuple(
                    mutation
                    for mutation in params.agentic_mutations
                    if mutation not in {"nonce_replay", "auth_replay"}
                )
                mutation_cycle = (
                    "valid_control",
                    *isolated,
                    "valid_control",
                    "nonce_replay",
                    "auth_replay",
                )
                kind = (
                    mutation_cycle[index]
                    if index < len(mutation_cycle)
                    else "valid_control"
                )
                actor = agents[0]
                counterparty = merchants[0]
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
        horizon = min(
            population.horizon_end,
            population.generated_at + timedelta(hours=params.duration_hours),
        )
        for plan in plans:
            for stage_index, _stage in enumerate(plan.stages):
                delay_low = params.min_delay_seconds
                delay_high = params.max_delay_seconds
                if family == "card_testing_cnp":
                    span = params.max_delay_seconds - params.min_delay_seconds
                    if plan.stages[0] == "decline":
                        delay_low = params.min_delay_seconds + (span * 2) // 3
                    else:
                        delay_high = params.min_delay_seconds + max(1, span // 4)
                if (
                    family == "app_scam_mule"
                    and stage_index == 0
                    and plan.actor.role == "mule"
                    and plan.counterparty.role in {"attacker", "synthetic_merchant"}
                ):
                    if params.cash_out_strategy == "burst":
                        delay_low = params.min_delay_seconds
                        delay_high = params.min_delay_seconds
                    elif params.cash_out_strategy == "delayed":
                        delay_low = params.cash_out_delay_seconds
                        delay_high = params.cash_out_delay_seconds
                    else:
                        delay_low = max(
                            params.min_delay_seconds,
                            params.cash_out_delay_seconds // params.mule_fanout,
                        )
                        delay_high = params.cash_out_delay_seconds
                delay = int(
                    self._rng.integers(
                        delay_low,
                        delay_high + 1,
                    )
                )
                current += timedelta(seconds=delay)
                if current > horizon:
                    raise ValueError("campaign schedule exceeds a declared horizon")
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

    def _app_amounts(
        self,
        plans: tuple[_PaymentPlan, ...],
        params: CampaignParams,
    ) -> tuple[Decimal, ...]:
        """Make the visible cash-out fraction alter concrete economic flow."""
        cash_indices = tuple(
            index
            for index, plan in enumerate(plans)
            if plan.actor.role == "mule" and plan.counterparty.role == "attacker"
        )
        cash_set = set(cash_indices)
        layer_indices = tuple(
            index
            for index, plan in enumerate(plans)
            if plan.actor.role == "mule" and plan.counterparty.role == "mule"
        )
        inbound_indices = tuple(
            index
            for index, plan in enumerate(plans)
            if plan.actor.role != "mule" and plan.counterparty.role == "mule"
        )
        benign_indices = tuple(
            index
            for index in range(len(plans))
            if index not in cash_set
            and index not in set(layer_indices)
            and index not in set(inbound_indices)
        )
        if not cash_indices or not layer_indices or not inbound_indices:
            raise ValueError("APP topology requires cash-out and funding payments")
        cash_target = (params.target_value_total * params.cash_out_fraction).quantize(
            _CENT,
            rounding=ROUND_HALF_EVEN,
        )
        if not (
            params.min_amount * len(cash_indices)
            <= cash_target
            <= params.max_amount * len(cash_indices)
        ):
            raise ValueError("APP cash-out fraction is infeasible under amount bounds")
        values = [params.min_amount for _plan in plans]
        cash_remainder = cash_target - params.min_amount * len(cash_indices)
        for index in cash_indices:
            change = min(params.max_amount - values[index], cash_remainder)
            values[index] += change
            cash_remainder -= change

        # Fund every downstream mule constructively, walking layer edges backwards.
        required_outflow: dict[str, Decimal] = {}
        for index in cash_indices:
            account = plans[index].payer_account
            required_outflow[account] = (
                required_outflow.get(account, Decimal("0.00"))
                + values[index]
            )
        for index in reversed(layer_indices):
            plan = plans[index]
            required = required_outflow.get(plan.payee_account, Decimal("0.00"))
            values[index] = max(params.min_amount, required)
            if values[index] > params.max_amount:
                raise ValueError("APP layer cannot fund declared downstream cash-out")
            required_outflow[plan.payer_account] = (
                required_outflow.get(plan.payer_account, Decimal("0.00"))
                + values[index]
            )

        root_account = plans[inbound_indices[0]].payee_account
        root_required = required_outflow.get(root_account, Decimal("0.00"))
        inbound_remainder = root_required - params.min_amount * len(inbound_indices)
        for index in inbound_indices:
            if inbound_remainder <= 0:
                break
            change = min(params.max_amount - values[index], inbound_remainder)
            values[index] += change
            inbound_remainder -= change
        if inbound_remainder > 0:
            raise ValueError("APP fan-in cannot fund declared mule outflows")

        remainder = params.target_value_total - sum(values, Decimal("0.00"))
        for index in (*benign_indices, *inbound_indices):
            if remainder <= 0:
                break
            change = min(params.max_amount - values[index], remainder)
            values[index] += change
            remainder -= change
        if remainder != 0:
            raise ValueError("APP exact value target is infeasible after economic funding")
        return tuple(value.quantize(_CENT) for value in values)

    def _agentic_amounts(
        self,
        plans: tuple[_PaymentPlan, ...],
        params: CampaignParams,
    ) -> tuple[Decimal, ...]:
        """Allocate one isolated over-limit leaf while preserving the exact total."""
        amount_index = next(
            (index for index, plan in enumerate(plans) if plan.mutation_kind == "amount"),
            None,
        )
        if amount_index is None or params.max_amount - _CENT < params.min_amount:
            raise ValueError("agentic amount mutation requires a visible bounded interval")
        cap = params.max_amount - _CENT
        values = [params.min_amount for _plan in plans]
        values[amount_index] = params.max_amount
        remainder = params.target_value_total - sum(values, Decimal("0.00"))
        for index in self._rng.permutation(len(values)):
            checked_index = int(index)
            if checked_index == amount_index or remainder <= 0:
                continue
            change = min(cap - values[checked_index], remainder)
            values[checked_index] += change
            remainder -= change
        if remainder != 0:
            raise ValueError("agentic value target is infeasible under isolated amount bounds")
        return tuple(value.quantize(_CENT) for value in values)

    def _materialize(
        self,
        family: str,
        plans: tuple[_PaymentPlan, ...],
        amounts: tuple[Decimal, ...],
        schedule: tuple[datetime, ...],
        population: Population,
        params: CampaignParams,
    ) -> tuple[tuple[Command, ...], AgenticFixture | None]:
        if family == "agentic_intent_abuse":
            return self._materialize_agentic(plans, amounts, schedule, population, params)
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
        schedule: tuple[datetime, ...],
        population: Population,
        params: CampaignParams,
    ) -> tuple[tuple[Command, ...], AgenticFixture]:
        if len(plans) < 25 or len(schedule) != len(plans):
            raise ValueError("agentic matrix requires 23 isolated attacks and two controls")
        private_key = Ed25519PrivateKey.from_private_bytes(self._rng.bytes(32))
        public_key = private_key.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw)
        agent_id = f"synthetic-agent-{plans[0].actor.entity_id}"
        key_id = "synthetic-key-1"
        merchant = next(
            entity
            for entity in population.entities
            if entity.role == "merchant" and entity.account_id
        )
        attackers = self._accounts(population, "attacker")
        mandate_cap = params.max_amount - _CENT
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
            max_amount=mandate_cap,
            currency=params.currency,
            issued_at=population.generated_at,
            expires_at=min(
                population.horizon_end,
                population.generated_at + timedelta(hours=params.duration_hours),
            ),
        )
        base_created = tuple(timestamp - timedelta(seconds=1) for timestamp in schedule)
        base_expires = tuple(
            min(timestamp + timedelta(seconds=30), mandate.expires_at)
            for timestamp in schedule
        )
        evidence: list[AuthenticationEvidence] = []
        first_control_index = next(
            index for index, plan in enumerate(plans) if plan.mutation_kind == "valid_control"
        )
        first_auth_ref = f"synthetic-auth-{first_control_index}"
        first_nonce = f"synthetic-nonce-{first_control_index}"
        for index, plan in enumerate(plans):
            kind = plan.mutation_kind
            if kind == "auth_missing":
                continue
            request_id = f"agentic-request-{index}"
            nonce = first_nonce if kind == "nonce_replay" else f"synthetic-nonce-{index}"
            auth_ref = first_auth_ref if kind == "auth_replay" else f"synthetic-auth-{index}"
            if kind == "auth_replay":
                continue
            intent_hash = (
                hashlib.sha256(f"mutated-intent-{index}".encode()).hexdigest()
                if kind == "intent"
                else mandate.payment_intent_hash
            )
            issued_at = base_created[index] - timedelta(seconds=1)
            expires_at = base_expires[index]
            evidence_nonce = "mismatched-nonce" if kind == "auth_mismatch" else nonce
            if kind == "auth_expired":
                issued_at = schedule[index] - timedelta(seconds=2)
                expires_at = schedule[index] - timedelta(seconds=1)
            evidence.append(
                AuthenticationEvidence(
                    evidence_id=auth_ref,
                    agent_id=agent_id,
                    user_ref=mandate.user_ref,
                    mandate_id=mandate.mandate_id,
                    nonce=evidence_nonce,
                    payment_intent_hash=intent_hash,
                    request_id=request_id,
                    outcome=AuthenticationOutcome.STEP_UP_VERIFIED,
                    issued_at=issued_at,
                    expires_at=expires_at,
                )
            )

        chain_verifier = TrustVerifier(
            registered_agents={(agent_id, key_id): public_key},
            mandates={mandate.mandate_id: mandate},
            authentication_evidence={item.evidence_id: item for item in evidence},
        )
        commands: list[Command] = []
        previous_receipt = ""
        for index, (plan, amount) in enumerate(zip(plans, amounts, strict=True)):
            kind = plan.mutation_kind
            request_id = f"agentic-request-{index}"
            nonce = first_nonce if kind == "nonce_replay" else f"synthetic-nonce-{index}"
            auth_ref = first_auth_ref if kind == "auth_replay" else f"synthetic-auth-{index}"
            request_mandate = (
                mandate.model_copy(update={"consent_ref": "substituted-mandate-consent"})
                if kind == "mandate"
                else mandate
            )
            request_agent = "unregistered-synthetic-agent" if kind == "identity" else agent_id
            request_key = "unregistered-key" if kind == "identity" else key_id
            actor_id = (
                attackers[0].entity_id
                if kind == "authority_identity"
                else plan.actor.entity_id
            )
            merchant_id = attackers[0].entity_id if kind == "merchant" else mandate.merchant_id
            payee_id = cast(str, attackers[0].account_id) if kind == "payee" else mandate.payee_id
            created_at = base_created[index]
            expires_at = base_expires[index]
            if kind == "mandate_time":
                created_at = mandate.issued_at - timedelta(seconds=2)
                expires_at = mandate.issued_at + timedelta(seconds=1)
            elif kind == "expiry":
                expires_at = schedule[index]
            prior_hash = "f" * 64 if kind == "receipt_chain" else previous_receipt
            unsigned = AgentPaymentRequest(
                request_id=request_id,
                payment_id=plan.payment_id,
                agent_id=request_agent,
                key_id=request_key,
                mandate=request_mandate,
                amount=amount,
                currency="EUR" if kind == "currency" else params.currency,
                merchant_id=merchant_id,
                payee_id=payee_id,
                cart_hash=(
                    hashlib.sha256(f"mutated-cart-{index}".encode()).hexdigest()
                    if kind == "cart"
                    else mandate.cart_hash
                ),
                payment_intent_hash=(
                    hashlib.sha256(f"mutated-intent-{index}".encode()).hexdigest()
                    if kind == "intent"
                    else mandate.payment_intent_hash
                ),
                category="RETAIL" if kind == "category" else "TRAVEL",
                product_id="substituted-product" if kind == "product" else "synthetic-flight",
                credential_id=(
                    "synthetic-token-substituted"
                    if kind == "credential"
                    else mandate.credential_id
                ),
                credential_scope=(
                    "multi_merchant_reusable"
                    if kind == "token_scope"
                    else mandate.credential_scope
                ),
                consent_ref=(
                    "synthetic-consent-substituted"
                    if kind == "consent"
                    else mandate.consent_ref
                ),
                authentication_evidence_ref=(
                    "synthetic-auth-missing" if kind == "auth_missing" else auth_ref
                ),
                nonce=nonce,
                created_at=created_at,
                expires_at=expires_at,
                prior_receipt_hash=prior_hash,
                campaign_id=params.campaign_id,
                trace_id=self._derived_uuid(params, f"agentic-trace:{index}"),
                actor_id=actor_id,
                counterparty_id=merchant.entity_id,
                signature=b"",
            )
            request = unsigned.model_copy(
                update={"signature": private_key.sign(unsigned.signing_bytes())}
            )
            if kind == "signature":
                request = request.model_copy(update={"signature": bytes(64)})
            if kind == "valid_control":
                preview = chain_verifier.preview(request, schedule[index])
                if not preview.allowed:
                    raise ValueError("agentic control failed during chain construction")
                receipt = chain_verifier.commit(
                    request,
                    preview,
                    ReceiptOutcome.APPROVE,
                    schedule[index],
                )
                if not receipt.allowed:
                    raise ValueError("agentic control could not commit its receipt")
                previous_receipt = receipt.receipt_hash
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

    @staticmethod
    def _dry_replay(
        family: str,
        commands: tuple[Command, ...],
        schedule: tuple[datetime, ...],
        population: Population,
        fixture: AgenticFixture | None,
    ) -> tuple[PaymentEvent, ...]:
        """Execute a candidate through a fresh production rail before acceptance."""
        factory: AdapterFactory
        if family == "app_scam_mule":
            rail = Rail.A2A

            def a2a_factory() -> A2ARailAdapter:
                return A2ARailAdapter()

            factory = a2a_factory
        elif family in {"card_testing_cnp", "synthetic_merchant_refund"}:
            rail = Rail.CARD

            def card_factory() -> CardRailAdapter:
                return CardRailAdapter()

            factory = card_factory
        else:
            rail = Rail.AGENTIC
            if fixture is None:
                raise ValueError("agentic replay requires evaluator-owned verifier inputs")

            def agentic_factory() -> AgenticRailAdapter:
                verifier = TrustVerifier(
                    registered_agents={(fixture.agent_id, fixture.key_id): fixture.public_key},
                    mandates={fixture.mandate.mandate_id: fixture.mandate},
                    authentication_evidence={
                        item.evidence_id: item for item in fixture.authentication_evidence
                    },
                )
                return AgenticRailAdapter(
                    verifier,
                    lambda _request, _receipt: Action.APPROVE,
                )

            factory = agentic_factory

        bundle = population.bundle.model_copy(update={"rail": rail})
        engine = SimulationEngine(
            bundle,
            {rail: factory},
            opening_balances=cast(
                Mapping[AccountReference, Decimal],
                population.opening_balances,
            ),
        )
        for priority, (timestamp, command) in enumerate(zip(schedule, commands, strict=True)):
            engine.schedule(timestamp, priority, command)
        events = engine.run()
        engine.ledger.assert_conserved()
        return events

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
        events = self._dry_replay(
            family,
            commands,
            schedule,
            population,
            fixture,
        )
        opening_commands = tuple(
            command
            for command in commands
            if command.name
            in {"a2a.initiate", "card.authorize", "card.decline", "agentic.pay"}
        )
        if len(opening_commands) != params.payment_count:
            raise ValueError("payment_count must be derived from concrete opening commands")
        attempted_value = sum(
            (cast(Decimal, command.payload["amount"]) for command in opening_commands),
            Decimal("0.00"),
        )
        unique_attempted: dict[str, Decimal] = {}
        for command in opening_commands:
            payment_id = cast(str, command.payload["payment_id"])
            unique_attempted.setdefault(payment_id, cast(Decimal, command.payload["amount"]))
        unique_attempted_value = sum(unique_attempted.values(), Decimal("0.00"))
        settled_value = sum(
            (
                event.amount
                for event in events
                if event.event_type in {EventKind.SETTLEMENT, EventKind.TRANSFER_POSTED}
                or (
                    event.rail is Rail.AGENTIC
                    and event.event_type is EventKind.AUTHORIZATION
                    and event.rail_data.get("integrity") == "pass"
                    and event.rail_data.get("action") == Action.APPROVE.value
                )
            ),
            Decimal("0.00"),
        )
        observed_reasons = tuple(
            ReasonCode(cast(str, event.rail_data["reason_code"]))
            if event.rail_data.get("reason_code")
            else None
            for event in events
            if event.rail is Rail.AGENTIC
        )
        entity_illicit = {entity.entity_id: entity.illicit for entity in population.entities}
        if family == "agentic_intent_abuse":
            if len(observed_reasons) != len(opening_commands):
                raise ValueError("agentic commands must each produce one observed outcome")
            class_labels = tuple(reason is not None for reason in observed_reasons)
        else:
            class_labels = tuple(
                entity_illicit[cast(str, command.payload["actor_id"])]
                or entity_illicit[cast(str, command.payload["counterparty_id"])]
                for command in opening_commands
            )
        rate = Decimal(sum(class_labels)) / Decimal(len(class_labels))
        value_total = attempted_value
        if params.expected_motif != _MOTIFS[family]:
            raise ValueError("declared motif does not match the selected family")
        benign_card_control = family == "card_testing_cnp" and rate == 0
        if not benign_card_control and motif_signature(commands) != params.expected_motif:
            raise ValueError("motif constraint not satisfied")
        if family == "card_testing_cnp" and rate > 0:
            attack_actors = {
                cast(str, command.payload["actor_id"])
                for command, label in zip(opening_commands, class_labels, strict=True)
                if label
            }
            shared_targets: dict[str, set[str]] = {}
            for relationship in population.relationships:
                if (
                    relationship.source_id in attack_actors
                    and relationship.relation == "uses_shared_device"
                ):
                    shared_targets.setdefault(relationship.target_id, set()).add(
                        relationship.source_id
                    )
            if max((len(actors) for actors in shared_targets.values()), default=0) < 2:
                raise ValueError("card-testing motif lacks a shared-device attack graph")
        if abs(rate - params.target_illicit_rate) > params.class_rate_tolerance:
            raise ValueError("class rate constraint not satisfied")
        if (
            family == "agentic_intent_abuse"
            and abs(rate - params.agentic_attack_mix) > params.class_rate_tolerance
        ):
            raise ValueError("agentic attack mix constraint not satisfied")
        if abs(value_total - params.target_value_total) > params.value_tolerance:
            raise ValueError("value total constraint not satisfied")
        if any(
            cast(Decimal, command.payload["amount"]) < params.min_amount
            or cast(Decimal, command.payload["amount"]) > params.max_amount
            for command in opening_commands
        ):
            raise ValueError("amount bounds not satisfied")
        if len(schedule) != len(commands) or schedule != tuple(sorted(schedule)):
            raise ValueError("command schedule is not total and ordered")
        if (
            not schedule
            or schedule[-1]
            > population.generated_at + timedelta(hours=params.duration_hours)
            or schedule[-1] > population.horizon_end
        ):
            raise ValueError("timestamp horizon not satisfied")
        declared = {entity.entity_id for entity in population.entities}
        referenced = {
            cast(str, command.payload[field])
            for command in opening_commands
            for field in ("actor_id", "counterparty_id")
        }
        accounts: set[str] = set()
        for command in commands:
            for key, value in command.payload.items():
                if key.endswith("_account") and type(value) is str:
                    accounts.add(value)
        if not referenced <= declared or not accounts <= set(population.opening_balances):
            raise ValueError("entity or account references are not population-owned")
        dependencies = self._command_dependencies(family, opening_commands)
        graph_document = [
            [
                command.payload["payment_id"],
                command.payload["actor_id"],
                command.payload["counterparty_id"],
                command.payload["payer_account"],
                command.payload["payee_account"],
                str(class_labels[index]),
            ]
            for index, command in enumerate(opening_commands)
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
            illicit_count=sum(class_labels),
            # All metrics below are recomputed from concrete commands and outcomes.
            illicit_rate=rate,
            value_total=value_total,
            attempted_value=attempted_value,
            unique_attempted_value=unique_attempted_value,
            settled_value=settled_value,
            schedule=schedule,
            graph_digest=hashlib.sha256(
                json.dumps(graph_document, separators=(",", ":")).encode()
            ).hexdigest(),
            schedule_digest=hashlib.sha256(
                json.dumps(schedule_document, separators=(",", ":")).encode()
            ).hexdigest(),
            declared_entity_ids=tuple(sorted(referenced)),
            account_ids=tuple(sorted(accounts)),
            class_labels=class_labels,
            mutation_kinds=tuple(
                dict.fromkeys(
                    reason.value for reason in observed_reasons if reason is not None
                )
            ),
            dependencies=dependencies,
            observed_reasons=observed_reasons,
            valid_control_count=sum(reason is None for reason in observed_reasons),
            replay_succeeded=True,
            ledger_conserved=True,
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
    def _command_dependencies(
        family: str,
        opening_commands: tuple[Command, ...],
    ) -> tuple[CampaignDependency, ...]:
        if family != "app_scam_mule":
            return ()
        dependencies: list[CampaignDependency] = []
        for index, command in enumerate(opening_commands):
            payer_account = cast(str, command.payload["payer_account"])
            upstream = tuple(
                cast(str, candidate.payload["payment_id"])
                for candidate in opening_commands[:index]
                if candidate.payload["payee_account"] == payer_account
            )
            if upstream:
                dependencies.append(
                    CampaignDependency(cast(str, command.payload["payment_id"]), upstream)
                )
        return tuple(dependencies)

    @staticmethod
    def _derived_uuid(params: CampaignParams, label: str) -> str:
        namespace = uuid5(
            NAMESPACE_URL,
            f"apar:campaign-identifiers:{params.campaign_id}:{params.seed}",
        )
        return str(uuid5(namespace, label))


class _CampaignEvaluator:
    """Evaluator-owned closure path; policies receive neither audit state nor fixtures."""

    def __init__(self, *, seed: int) -> None:
        self.__generator = CampaignGenerator(seed=seed)

    def generate(
        self,
        family: str,
        population: Population,
        params: CampaignParams,
    ) -> tuple[tuple[Command, ...], CampaignEvidence]:
        return self.__generator._generate_audited(family, population, params)

    def validate(
        self,
        family: str,
        commands: tuple[Command, ...],
        population: Population,
        params: CampaignParams,
    ) -> None:
        """Fail closed when a concrete externally supplied candidate is incomplete."""
        try:
            if motif_signature(commands) != _MOTIFS[family]:
                raise ValueError("family motif mismatch")
            horizon = min(
                population.horizon_end,
                population.generated_at + timedelta(hours=params.duration_hours),
            )
            step = max(1, params.min_delay_seconds)
            schedule = tuple(
                population.generated_at + timedelta(seconds=step * (index + 1))
                for index in range(len(commands))
            )
            if not schedule or schedule[-1] > horizon:
                raise ValueError("candidate exceeds evaluator horizon")
            if family == "agentic_intent_abuse":
                raise ValueError("agentic candidates require evaluator-owned fixture closure")
            self.__generator._dry_replay(family, commands, schedule, population, None)
        except (KeyError, RuntimeError, TypeError, ValueError) as error:
            raise GenerationConstraintError(100) from error


__all__ = [
    "AGENTIC_INTENT_ABUSE_MOTIF",
    "APP_SCAM_MULE_MOTIF",
    "CARD_TESTING_CNP_MOTIF",
    "SYNTHETIC_MERCHANT_REFUND_MOTIF",
    "CampaignGenerator",
    "CampaignParameterError",
    "CampaignParams",
    "GenerationConstraintError",
    "campaign_bytes",
    "motif_signature",
]
