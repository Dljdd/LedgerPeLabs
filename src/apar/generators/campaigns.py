"""Causal, bounded generators for the four executable campaign families."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from decimal import ROUND_HALF_EVEN, Decimal, DecimalException, localcontext
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
_PROBABILITY_QUANTUM = Decimal("0.000000000000000001")
_COMPATIBLE_RECOVERY_PROBABILITY = Decimal("0.25")
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
_AGENTIC_REASON_COVERAGE = frozenset(
    {
        ReasonCode.AGENT_IDENTITY_MISMATCH,
        ReasonCode.SIGNATURE_INVALID,
        ReasonCode.MANDATE_SCOPE_VIOLATION,
        ReasonCode.AUTHORITY_IDENTITY_MISMATCH,
        ReasonCode.AMOUNT_LIMIT_EXCEEDED,
        ReasonCode.CURRENCY_MISMATCH,
        ReasonCode.MERCHANT_BINDING_MISMATCH,
        ReasonCode.PAYEE_BINDING_MISMATCH,
        ReasonCode.CATEGORY_SCOPE_VIOLATION,
        ReasonCode.PRODUCT_SCOPE_VIOLATION,
        ReasonCode.CART_HASH_MISMATCH,
        ReasonCode.PAYMENT_INTENT_HASH_MISMATCH,
        ReasonCode.CREDENTIAL_BINDING_MISMATCH,
        ReasonCode.TOKEN_SCOPE_VIOLATION,
        ReasonCode.CONSENT_BINDING_MISMATCH,
        ReasonCode.MANDATE_TIME_SCOPE_VIOLATION,
        ReasonCode.MANDATE_EXPIRED,
        ReasonCode.AUTHENTICATION_EVIDENCE_MISSING,
        ReasonCode.AUTHENTICATION_EVIDENCE_MISMATCH,
        ReasonCode.AUTHENTICATION_EVIDENCE_EXPIRED,
        ReasonCode.NONCE_REPLAY,
        ReasonCode.RECEIPT_CHAIN_BROKEN,
        ReasonCode.AUTHENTICATION_EVIDENCE_REPLAY,
    }
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
    try:
        if amount != amount.quantize(_CENT, rounding=ROUND_HALF_EVEN):
            raise ValueError(f"{label} must be canonically quantized for USD")
    except DecimalException as error:
        raise ValueError(f"{label} cannot be represented as bounded USD") from error
    return amount


def _decimal_unit(label: str, value: object) -> Decimal:
    if type(value) is not Decimal:
        raise TypeError(f"{label} must be an exact Decimal")
    number = value
    if not number.is_finite() or number < 0 or number > 1:
        raise ValueError(f"{label} must be between zero and one")
    return number


def _canonical_recovery_probability(recovery_count: int, eligible_count: int) -> Decimal:
    """Return one stable public probability for one discrete recovery count."""
    compatible_count = int(
        Decimal(eligible_count) * _COMPATIBLE_RECOVERY_PROBABILITY
    )
    if Decimal(eligible_count) * _COMPATIBLE_RECOVERY_PROBABILITY > compatible_count:
        compatible_count += 1
    if recovery_count == compatible_count:
        return _COMPATIBLE_RECOVERY_PROBABILITY
    with localcontext() as context:
        context.prec = 50
        return (Decimal(recovery_count) / Decimal(eligible_count)).quantize(
            _PROBABILITY_QUANTUM,
            rounding=ROUND_HALF_EVEN,
        )


def _recovery_count(
    probability: Decimal,
    eligible_count: int,
    *,
    require_unrecovered: bool,
) -> int:
    """Decode a canonical discrete probability without ceiling aliases."""
    maximum = eligible_count - 1 if require_unrecovered else eligible_count
    if eligible_count <= 0 or maximum <= 0:
        raise ValueError("recovery motif has no eligible discrete level")
    for recovery_count in range(1, maximum + 1):
        if probability == _canonical_recovery_probability(
            recovery_count,
            eligible_count,
        ):
            return recovery_count
    raise ValueError("recovery_probability is not canonical for the eligible count")


def _card_delay_regions(minimum: int, maximum: int) -> tuple[int, int]:
    """Derive disjoint probe and success regions or fail closed."""
    span = maximum - minimum
    probe_floor = minimum + (span * 2) // 3
    success_ceiling = minimum + span // 4
    if probe_floor <= success_ceiling:
        raise ValueError("card-testing delay regions are not strictly separated")
    return probe_floor, success_ceiling


def _ratio(numerator: int, denominator: int, *, precision: int = 18) -> Decimal:
    if denominator <= 0:
        raise ValueError("discrete adaptive denominator must be positive")
    with localcontext() as context:
        context.prec = max(precision + 8, 36)
        quantum = Decimal(1).scaleb(-precision)
        return (Decimal(numerator) / Decimal(denominator)).quantize(
            quantum,
            rounding=ROUND_HALF_EVEN,
        )


def _merchant_target_count(concentration: Decimal, available: int) -> int:
    if available <= 0:
        raise ValueError("merchant concentration has no eligible merchants")
    compatible = max(
        1,
        int(
            (
                Decimal(available) * (Decimal(1) - Decimal("0.70"))
            ).to_integral_value(rounding=ROUND_HALF_EVEN)
        ),
    )
    for count in range(1, available + 1):
        if count == compatible:
            candidate = Decimal("0.70")
        elif count == 1:
            candidate = Decimal(1)
        elif count == available:
            candidate = Decimal(0)
        else:
            candidate = Decimal(1) - _ratio(count, available)
        if concentration == candidate:
            return count
    raise ValueError("merchant_concentration is not canonical for the population")


def _card_actor_span(reuse_rate: Decimal, available: int) -> int:
    if available < 2:
        raise ValueError("device reuse requires at least two eligible actors")
    compatible = max(
        2,
        min(
            available,
            int(
                (
                    Decimal(2)
                    + Decimal(available - 2) * (Decimal(1) - Decimal("0.60"))
                ).to_integral_value(rounding=ROUND_HALF_EVEN)
            ),
        ),
    )
    for count in range(2, available + 1):
        if count == compatible:
            candidate = Decimal("0.60")
        elif count == 2:
            candidate = Decimal(1)
        elif count == available:
            candidate = Decimal(0)
        else:
            candidate = _ratio(available - count, available - 2)
        if reuse_rate == candidate:
            return count
    raise ValueError("device_reuse_rate is not canonical for the population")


def _cash_out_target(fraction: Decimal, total: Decimal) -> Decimal:
    cash_target = (total * fraction).quantize(_CENT, rounding=ROUND_HALF_EVEN)
    compatible_target = (total * Decimal("0.30")).quantize(
        _CENT,
        rounding=ROUND_HALF_EVEN,
    )
    if cash_target == compatible_target:
        candidate = Decimal("0.30")
    else:
        total_cents = int(total / _CENT)
        candidate = _ratio(int(cash_target / _CENT), total_cents)
    if fraction != candidate:
        raise ValueError("cash_out_fraction is not canonical for the realized cents")
    return cash_target


def _agentic_attack_count(attack_mix: Decimal, payment_count: int) -> int:
    for count in range(23, payment_count - 1):
        with localcontext() as context:
            context.prec = 28
            candidate = Decimal(count) / Decimal(payment_count)
        if attack_mix == candidate:
            return count
    raise ValueError("agentic_attack_mix is not canonical for the payment count")


def _agentic_extras(
    mutations: tuple[str, ...],
    additional_attacks: int,
) -> tuple[str, ...]:
    if additional_attacks == 0:
        if mutations != _AGENTIC_MUTATIONS:
            raise ValueError("agentic mutation selection has no adaptive slot")
        return ()
    if mutations != _AGENTIC_MUTATIONS and len(mutations) > additional_attacks:
        raise ValueError("agentic mutation selection contains unused values")
    extras = tuple(
        mutations[index % len(mutations)] for index in range(additional_attacks)
    )
    default_extras = tuple(
        _AGENTIC_MUTATIONS[index % len(_AGENTIC_MUTATIONS)]
        for index in range(additional_attacks)
    )
    if mutations != _AGENTIC_MUTATIONS and extras == default_extras:
        raise ValueError("agentic mutation selection aliases the default")
    return extras


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
    cash_out_delay_seconds: int = 300
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
            if type(self.seed) is not int or not 0 <= self.seed < 2**63:
                raise TypeError("seed must be an exact integer in [0, 2**63)")
            integer_bounds = {
                "payment_count": (1, 256),
                "duration_hours": (1, 720),
                "query_budget": (1, 1000),
                "min_delay_seconds": (1, 3600),
                "max_delay_seconds": (1, 3600),
                "retry_intensity": (0, 10),
                "mule_count": (2, 16),
                "mule_layers": (1, 15),
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
            if value_tolerance > Decimal("10000.00") or value_tolerance > target:
                raise ValueError("value_tolerance exceeds its visible cap")
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
                not self.agentic_mutations
                or len(set(self.agentic_mutations)) != len(self.agentic_mutations)
                or not set(self.agentic_mutations) <= set(_AGENTIC_MUTATIONS)
            ):
                raise ValueError("agentic_mutations contains duplicates or undeclared values")
        except (DecimalException, OverflowError, TypeError, ValueError) as error:
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
        posted_path = ("initiate", "accept", "post")
        return_path = (*posted_path, "return")
        recovery_path = (*posted_path, "report", "freeze", "recover")
        if not opens or any(
            operations not in {posted_path, return_path, recovery_path}
            for operations in histories.values()
        ):
            raise ValueError("A2A motif requires complete legal terminal lifecycles")
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
        if (
            len(bridges) >= 2
            and layer_edges
            and fan_in >= 2
            and fan_out >= 2
            and return_path in histories.values()
            and recovery_path in histories.values()
        ):
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
        raise ValueError(
            "agentic deep motif requires evaluator execution evidence"
        )
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
        if type(seed) is not int or not 0 <= seed < 2**63:
            raise TypeError("seed must be an exact integer in [0, 2**63)")
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
                    schedule,
                    population,
                    params,
                    attempt,
                    fixture,
                )
            except (
                ArithmeticError,
                DecimalException,
                KeyError,
                RuntimeError,
                TypeError,
                ValueError,
            ):
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
        agentic_kinds: tuple[str, ...] = ()
        if family == "agentic_intent_abuse":
            desired_attacks = _agentic_attack_count(
                params.agentic_attack_mix,
                params.payment_count,
            )
            if (
                params.payment_count < 25
                or desired_attacks < 23
                or params.payment_count - desired_attacks < 2
            ):
                raise ValueError("agentic matrix requires 23 attacks and two controls")
            mandatory_non_replay = tuple(
                mutation
                for mutation in _AGENTIC_MUTATIONS
                if mutation not in {"nonce_replay", "auth_replay"}
            )
            mandatory = (
                "valid_control",
                *mandatory_non_replay,
                "valid_control",
                "nonce_replay",
                "auth_replay",
            )
            additional_attacks = desired_attacks - 23
            extras = _agentic_extras(
                params.agentic_mutations,
                additional_attacks,
            )
            controls = ("valid_control",) * (
                params.payment_count - len(mandatory) - len(extras)
            )
            agentic_kinds = (*mandatory, *extras, *controls)

        for index in range(params.payment_count):
            payment_id = f"payment:{uuid5(namespace, f'payment:{index}')}"
            illicit = labels[index]
            if family == "app_scam_mule":
                attack_count = illicit_count
                selected_mules = mules[: params.mule_count]
                layer_count = params.mule_layers
                fanout_count = params.mule_fanout
                recovered_count = _recovery_count(
                    params.recovery_probability,
                    fanout_count,
                    require_unrecovered=False,
                )
                incoming_count = attack_count - layer_count - fanout_count
                if (
                    params.mule_count != layer_count + 1
                    or len(selected_mules) != params.mule_count
                    or incoming_count < 2
                    or recovered_count == 0
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
                    if fanout_index >= fanout_count - recovered_count:
                        stages += ("report", "freeze", "recover")
                else:
                    actor = victims[index % len(victims)]
                    counterparty = merchants[index % len(merchants)]
                    stages = ("initiate", "accept", "post", "return")
            elif family == "card_testing_cnp":
                if illicit:
                    if params.retry_intensity == 0:
                        raise ValueError("card-testing requires at least one probe retry")
                    if params.retry_intensity > illicit_count - 1:
                        raise ValueError("retry intensity exceeds eligible probes")
                    actor_span = _card_actor_span(
                        params.device_reuse_rate,
                        min(len(attackers), illicit_count),
                    )
                    actor = attackers[index % actor_span]
                    concentrated = _merchant_target_count(
                        params.merchant_concentration,
                        min(len(merchants), illicit_count),
                    )
                    counterparty = merchants[index % concentrated]
                    decline_count = params.retry_intensity
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
                    recovered_count = _recovery_count(
                        params.recovery_probability,
                        illicit_count,
                        require_unrecovered=True,
                    )
                    if recovered_count == 0 or recovered_count >= illicit_count:
                        raise ValueError(
                            "synthetic merchant motif requires recovery and refund paths"
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
                kind = agentic_kinds[index]
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
        card_delay_regions = (
            _card_delay_regions(
                params.min_delay_seconds,
                params.max_delay_seconds,
            )
            if family == "card_testing_cnp"
            else None
        )
        cash_payment_ids = tuple(
            plan.payment_id
            for plan in plans
            if plan.actor.role == "mule" and plan.counterparty.role == "attacker"
        )
        if family == "app_scam_mule" and params.cash_out_strategy == "staged":
            if len(cash_payment_ids) < 2:
                raise ValueError("staged cash-out requires at least two payments")
            if params.cash_out_delay_seconds <= params.min_delay_seconds:
                raise ValueError("staged cash-out requires a distinct delay witness")
        if (
            family == "app_scam_mule"
            and params.cash_out_strategy == "delayed"
            and params.cash_out_delay_seconds <= params.min_delay_seconds
        ):
            raise ValueError("delayed cash-out requires a distinct delay witness")
        horizon = min(
            population.horizon_end,
            population.generated_at + timedelta(hours=params.duration_hours),
        )
        for plan in plans:
            for stage_index, _stage in enumerate(plan.stages):
                delay_low = params.min_delay_seconds
                delay_high = params.max_delay_seconds
                if family == "card_testing_cnp":
                    assert card_delay_regions is not None
                    probe_floor, success_ceiling = card_delay_regions
                    if plan.stages[0] == "decline":
                        delay_low = probe_floor
                    else:
                        delay_high = success_ceiling
                if (
                    family == "app_scam_mule"
                    and stage_index == 0
                    and plan.actor.role == "mule"
                    and plan.counterparty.role in {"attacker", "synthetic_merchant"}
                ):
                    if not (
                        params.min_delay_seconds
                        <= params.cash_out_delay_seconds
                        <= params.max_delay_seconds
                    ):
                        raise ValueError("APP cash-out delay exceeds command delay bounds")
                    if params.cash_out_strategy == "burst":
                        if params.cash_out_delay_seconds != params.min_delay_seconds:
                            raise ValueError(
                                "burst cash-out delay must equal the minimum delay"
                            )
                        delay_low = params.min_delay_seconds
                        delay_high = params.min_delay_seconds
                    elif params.cash_out_strategy == "delayed":
                        delay_low = params.cash_out_delay_seconds
                        delay_high = params.cash_out_delay_seconds
                    else:
                        if plan.payment_id == cash_payment_ids[0]:
                            delay_low = params.min_delay_seconds
                            delay_high = params.min_delay_seconds
                        elif plan.payment_id == cash_payment_ids[-1]:
                            delay_low = params.cash_out_delay_seconds
                            delay_high = params.cash_out_delay_seconds
                        else:
                            delay_low = params.min_delay_seconds
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
        cash_target = _cash_out_target(
            params.cash_out_fraction,
            params.target_value_total,
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
        amount_indices = tuple(
            index for index, plan in enumerate(plans) if plan.mutation_kind == "amount"
        )
        if not amount_indices or params.max_amount - _CENT < params.min_amount:
            raise ValueError("agentic amount mutation requires a visible bounded interval")
        cap = params.max_amount - _CENT
        values = [params.min_amount for _plan in plans]
        for index in amount_indices:
            values[index] = params.max_amount
        remainder = params.target_value_total - sum(values, Decimal("0.00"))
        for index in self._rng.permutation(len(values)):
            checked_index = int(index)
            if checked_index in amount_indices or remainder <= 0:
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
        key_id = f"synthetic-key-{params.campaign_id}"
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
            consent_ref=f"synthetic-consent-{params.campaign_id}",
            merchant_id=merchant.entity_id,
            payee_id=cast(str, merchant.account_id),
            cart_hash=hashlib.sha256(b"synthetic-cart-v1").hexdigest(),
            payment_intent_hash=hashlib.sha256(b"synthetic-intent-v1").hexdigest(),
            permitted_categories=("TRAVEL",),
            permitted_products=("synthetic-flight",),
            credential_id=f"synthetic-token-{params.campaign_id}",
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
        first_auth_ref = f"synthetic-auth-{params.campaign_id}-{first_control_index}"
        first_nonce = f"synthetic-nonce-{params.campaign_id}-{first_control_index}"
        for index, plan in enumerate(plans):
            kind = plan.mutation_kind
            if kind == "auth_missing":
                continue
            request_id = f"agentic-request-{params.campaign_id}-{index}"
            nonce = (
                first_nonce
                if kind == "nonce_replay"
                else f"synthetic-nonce-{params.campaign_id}-{index}"
            )
            auth_ref = (
                first_auth_ref
                if kind == "auth_replay"
                else f"synthetic-auth-{params.campaign_id}-{index}"
            )
            if kind == "auth_replay":
                continue
            intent_hash = (
                hashlib.sha256(f"mutated-intent-{index}".encode()).hexdigest()
                if kind == "intent"
                else mandate.payment_intent_hash
            )
            issued_at = base_created[index] - timedelta(seconds=1)
            expires_at = base_expires[index]
            evidence_nonce = (
                f"mismatched-nonce-{params.campaign_id}"
                if kind == "auth_mismatch"
                else nonce
            )
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
            request_id = f"agentic-request-{params.campaign_id}-{index}"
            nonce = (
                first_nonce
                if kind == "nonce_replay"
                else f"synthetic-nonce-{params.campaign_id}-{index}"
            )
            auth_ref = (
                first_auth_ref
                if kind == "auth_replay"
                else f"synthetic-auth-{params.campaign_id}-{index}"
            )
            request_mandate = (
                mandate.model_copy(
                    update={
                        "consent_ref": f"substituted-mandate-consent-{params.campaign_id}"
                    }
                )
                if kind == "mandate"
                else mandate
            )
            request_agent = (
                f"unregistered-synthetic-agent-{params.campaign_id}"
                if kind == "identity"
                else agent_id
            )
            request_key = (
                f"unregistered-key-{params.campaign_id}"
                if kind == "identity"
                else key_id
            )
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
                    f"synthetic-token-substituted-{params.campaign_id}"
                    if kind == "credential"
                    else mandate.credential_id
                ),
                credential_scope=(
                    "multi_merchant_reusable"
                    if kind == "token_scope"
                    else mandate.credential_scope
                ),
                consent_ref=(
                    f"synthetic-consent-substituted-{params.campaign_id}"
                    if kind == "consent"
                    else mandate.consent_ref
                ),
                authentication_evidence_ref=(
                    f"synthetic-auth-missing-{params.campaign_id}"
                    if kind == "auth_missing"
                    else auth_ref
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
        schedule: tuple[datetime, ...],
        population: Population,
        params: CampaignParams,
        attempt: int,
        fixture: AgenticFixture | None,
    ) -> CampaignEvidence:
        if type(commands) is not tuple or not commands:
            raise TypeError("candidate commands must be a non-empty exact tuple")
        if type(schedule) is not tuple or len(schedule) != len(commands):
            raise ValueError("schedule must contain one timestamp per command")
        if params.expected_motif != _MOTIFS[family]:
            raise ValueError("declared motif does not match the selected family")
        if family == "agentic_intent_abuse":
            desired_attacks = _agentic_attack_count(
                params.agentic_attack_mix,
                params.payment_count,
            )
            _agentic_extras(
                params.agentic_mutations,
                desired_attacks - 23,
            )
        expected_command_type: type[Command]
        if family == "app_scam_mule":
            expected_command_type = A2ACommand
        elif family in {"card_testing_cnp", "synthetic_merchant_refund"}:
            expected_command_type = CardCommand
        else:
            expected_command_type = AgenticPaymentCommand
        if any(not isinstance(command, expected_command_type) for command in commands):
            raise TypeError("candidate contains a command from the wrong rail")
        if any(getattr(command, "campaign_id", None) != params.campaign_id for command in commands):
            raise ValueError("every command must retain the exact declared campaign_id")

        previous = population.generated_at
        horizon = min(
            population.horizon_end,
            population.generated_at + timedelta(hours=params.duration_hours),
        )
        for timestamp in schedule:
            if type(timestamp) is not datetime or timestamp.tzinfo is not UTC:
                raise ValueError("schedule timestamps must be exact UTC datetimes")
            delay = timestamp - previous
            if not (
                timedelta(seconds=params.min_delay_seconds)
                <= delay
                <= timedelta(seconds=params.max_delay_seconds)
            ):
                raise ValueError("every command delay must satisfy exact visible bounds")
            if timestamp > horizon:
                raise ValueError("timestamp exceeds a population or parameter horizon")
            previous = timestamp

        closed_payments: set[str] = set()
        prior_payment_id: str | None = None
        for command in commands:
            payment_id = getattr(command, "payment_id", None)
            if type(payment_id) is not str:
                raise ValueError("every command must expose a payment_id")
            if payment_id != prior_payment_id:
                if payment_id in closed_payments:
                    raise ValueError("payment lifecycle commands must remain contiguous")
                if prior_payment_id is not None:
                    closed_payments.add(prior_payment_id)
                prior_payment_id = payment_id

        opening_commands = tuple(
            command
            for command in commands
            if command.name
            in {"a2a.initiate", "card.authorize", "card.decline", "agentic.pay"}
        )
        if len(opening_commands) != params.payment_count:
            raise ValueError("payment_count must equal concrete opening command count")
        payment_ids = tuple(
            cast(str, command.payload["payment_id"]) for command in opening_commands
        )
        if len(set(payment_ids)) != len(payment_ids):
            raise ValueError("each opening command must use a unique payment_id")
        payment_namespace = uuid5(
            NAMESPACE_URL,
            f"apar:campaign:{params.campaign_id}:{params.seed}",
        )
        expected_payment_ids = tuple(
            f"payment:{uuid5(payment_namespace, f'payment:{index}')}"
            for index in range(params.payment_count)
        )
        if payment_ids != expected_payment_ids:
            raise ValueError("payment IDs or opening order differ from canonical lineage")
        payment_positions = {
            payment_id: index for index, payment_id in enumerate(payment_ids)
        }
        for command in commands:
            payment_id = cast(str, command.payload["payment_id"])
            position = payment_positions[payment_id]
            if command.name in {"a2a.initiate", "card.authorize", "card.decline"}:
                expected_key = f"{command.name}:{payment_id}"
                if command.payload.get("idempotency_key") != expected_key:
                    raise ValueError("opening idempotency key differs from canonical lineage")
            elif command.name != "agentic.pay":
                expected_key = (
                    f"{command.name}:{payment_id}:campaign:{params.campaign_id}"
                )
                if command.payload.get("idempotency_key") != expected_key:
                    raise ValueError("lifecycle idempotency key differs from canonical lineage")
            if command in opening_commands:
                trace_label = (
                    f"agentic-trace:{position}"
                    if family == "agentic_intent_abuse"
                    else f"trace:{position}"
                )
                if command.payload.get("trace_id") != self._derived_uuid(
                    params, trace_label
                ):
                    raise ValueError("trace ID differs from canonical lineage")
                if family == "agentic_intent_abuse" and command.payload.get(
                    "request_id"
                ) != f"agentic-request-{params.campaign_id}-{position}":
                    raise ValueError("agentic request ID differs from canonical lineage")

        entity_by_id = {entity.entity_id: entity for entity in population.entities}
        owner_by_account = {
            account.account_id: account.owner_entity_id for account in population.accounts
        }
        referenced: set[str] = set()
        accounts: set[str] = set()
        for command in commands:
            for key, value in command.payload.items():
                if key.endswith("_account") and type(value) is str:
                    accounts.add(value)
        if not accounts <= set(population.opening_balances):
            raise ValueError("candidate references an account outside the population")

        attempted_value = Decimal("0.00")
        unique_attempted: dict[str, Decimal] = {}
        for command in opening_commands:
            amount = command.payload.get("amount")
            currency = command.payload.get("currency")
            if type(amount) is not Decimal or not amount.is_finite():
                raise TypeError("every opening amount must be an exact finite Decimal")
            if amount < params.min_amount or amount > params.max_amount:
                raise ValueError("opening amount exceeds visible bounds")
            if type(currency) is not str:
                raise TypeError("opening currency must be an exact string")
            actor_id = cast(str, command.payload["actor_id"])
            counterparty_id = cast(str, command.payload["counterparty_id"])
            payer_account = cast(str, command.payload["payer_account"])
            payee_account = cast(str, command.payload["payee_account"])
            if actor_id not in entity_by_id or counterparty_id not in entity_by_id:
                raise ValueError("candidate references an undeclared entity")
            if payer_account not in owner_by_account or payee_account not in owner_by_account:
                raise ValueError("payer and payee must be population-owned accounts")
            if family != "agentic_intent_abuse" and (
                owner_by_account[payer_account] != actor_id
                or owner_by_account[payee_account] != counterparty_id
            ):
                raise ValueError("entity identities must own their declared payment accounts")
            if family != "agentic_intent_abuse" and currency != params.currency:
                raise ValueError("opening currency differs from campaign currency")
            attempted_value += amount
            unique_attempted.setdefault(cast(str, command.payload["payment_id"]), amount)
            referenced.update((actor_id, counterparty_id))

        if family == "agentic_intent_abuse":
            requests = [
                cast(AgenticPaymentCommand, command).request for command in commands
            ]
            if (
                len(requests) < 25
                or len({request.request_id for request in requests}) != len(requests)
                or len({request.nonce for request in requests}) == len(requests)
                or len({request.signature for request in requests}) < 24
            ):
                raise ValueError("agentic command structure is incomplete")
        elif motif_signature(commands) != params.expected_motif:
            raise ValueError("candidate does not satisfy its deep structural motif")
        events = self._dry_replay(
            family,
            commands,
            schedule,
            population,
            fixture,
        )
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
        if family == "agentic_intent_abuse":
            if len(observed_reasons) != len(opening_commands):
                raise ValueError("agentic commands must each produce one observed outcome")
            class_labels = tuple(reason is not None for reason in observed_reasons)
            observed_coverage = frozenset(
                reason for reason in observed_reasons if reason is not None
            )
            if not observed_coverage >= _AGENTIC_REASON_COVERAGE:
                raise ValueError("agentic campaign lacks mandatory Task4 reason coverage")
            if sum(reason is None for reason in observed_reasons) < 2:
                raise ValueError("agentic campaign requires two valid receipt-chain controls")
            for command, reason in zip(
                opening_commands, observed_reasons, strict=True
            ):
                actor_id = cast(str, command.payload["actor_id"])
                counterparty_id = cast(str, command.payload["counterparty_id"])
                payer_account = cast(str, command.payload["payer_account"])
                payee_account = cast(str, command.payload["payee_account"])
                if (
                    owner_by_account[payer_account] != actor_id
                    and reason is not ReasonCode.AUTHORITY_IDENTITY_MISMATCH
                ):
                    raise ValueError("agentic actor/account mismatch is not causally isolated")
                if (
                    owner_by_account[payee_account] != counterparty_id
                    and reason is not ReasonCode.PAYEE_BINDING_MISMATCH
                ):
                    raise ValueError("agentic payee/account mismatch is not causally isolated")
                if (
                    command.payload["currency"] != params.currency
                    and reason is not ReasonCode.CURRENCY_MISMATCH
                ):
                    raise ValueError("agentic currency drift is not causally isolated")
        else:
            histories = _operation_histories(commands)
            causal_labels: list[bool] = []
            for command in opening_commands:
                actor = entity_by_id[cast(str, command.payload["actor_id"])]
                counterparty = entity_by_id[
                    cast(str, command.payload["counterparty_id"])
                ]
                path = histories[cast(str, command.payload["payment_id"])]
                if family == "app_scam_mule":
                    posted = ("initiate", "accept", "post")
                    app_recovered = (*posted, "report", "freeze", "recover")
                    returned = (*posted, "return")
                    if (
                        (
                            actor.role in {"victim", "consumer", "organization"}
                            and counterparty.role == "mule"
                            and path == posted
                        )
                        or (
                            actor.role == "mule"
                            and counterparty.role == "mule"
                            and path == posted
                        )
                        or (
                            actor.role == "mule"
                            and counterparty.role == "attacker"
                            and path in {posted, app_recovered}
                        )
                    ):
                        causal_illicit = True
                    elif (
                        actor.role in {"victim", "consumer", "organization"}
                        and counterparty.role in {"merchant", "beneficiary"}
                        and path == returned
                    ):
                        causal_illicit = False
                    else:
                        raise ValueError(
                            "APP roles and terminal lifecycle are causally inconsistent"
                        )
                elif family == "card_testing_cnp":
                    declined = ("decline",)
                    settled = ("authorize", "clear", "settle")
                    if (
                        actor.role == "attacker"
                        and counterparty.role in {"merchant", "beneficiary"}
                        and path in {declined, settled}
                    ):
                        causal_illicit = True
                    elif (
                        actor.role in {"victim", "consumer", "organization"}
                        and counterparty.role in {"merchant", "beneficiary"}
                        and path == settled
                    ):
                        causal_illicit = False
                    else:
                        raise ValueError(
                            "card-testing roles and lifecycle are causally inconsistent"
                        )
                else:
                    refunded = ("authorize", "clear", "settle", "refund")
                    card_recovered = (
                        "authorize",
                        "clear",
                        "settle",
                        "report",
                        "dispute",
                        "chargeback",
                        "recover",
                    )
                    if (
                        actor.role == "attacker"
                        and counterparty.role == "synthetic_merchant"
                        and path in {refunded, card_recovered}
                    ):
                        causal_illicit = True
                    elif (
                        actor.role in {"victim", "consumer", "organization"}
                        and counterparty.role in {"merchant", "beneficiary"}
                        and path == refunded
                    ):
                        causal_illicit = False
                    else:
                        raise ValueError(
                            "synthetic-merchant roles and lifecycle are causally inconsistent"
                        )
                if causal_illicit != (actor.illicit or counterparty.illicit):
                    raise ValueError("causal class conflicts with population role ownership")
                causal_labels.append(causal_illicit)
            class_labels = tuple(causal_labels)
        rate = Decimal(sum(class_labels)) / Decimal(len(class_labels))
        value_total = attempted_value
        if family == "card_testing_cnp":
            if rate == 0:
                raise ValueError("zero-illicit campaigns cannot claim card-testing motif")
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
            decline_count = sum(command.name == "card.decline" for command in opening_commands)
            if params.retry_intensity > sum(class_labels) - 1:
                raise ValueError("retry intensity exceeds eligible concrete probes")
            expected_declines = params.retry_intensity
            if decline_count != expected_declines or decline_count == 0:
                raise ValueError("card probe count differs from visible retry intensity")
            available_attackers = sum(
                entity.role == "attacker" for entity in population.entities
            )
            expected_actor_span = _card_actor_span(
                params.device_reuse_rate,
                min(available_attackers, sum(class_labels)),
            )
            illicit_commands = tuple(
                command
                for command, label in zip(
                    opening_commands, class_labels, strict=True
                )
                if label
            )
            if len(
                {command.payload["actor_id"] for command in illicit_commands}
            ) != min(len(illicit_commands), expected_actor_span):
                raise ValueError("card actor reuse differs from visible device reuse")
            merchant_count = sum(
                entity.role in {"merchant", "beneficiary"}
                and entity.account_id is not None
                for entity in population.entities
            )
            concentrated = _merchant_target_count(
                params.merchant_concentration,
                min(merchant_count, sum(class_labels)),
            )
            if len(
                {command.payload["payee_account"] for command in illicit_commands}
            ) != min(len(illicit_commands), concentrated):
                raise ValueError("merchant distribution differs from visible concentration")
            if any(
                command.name != "card.decline"
                for command in opening_commands[:decline_count]
            ) or any(
                command.name == "card.decline"
                for command in opening_commands[decline_count:]
            ):
                raise ValueError("card probes must precede every success escalation")
            probe_actor_ids = {
                command.payload["actor_id"]
                for command in opening_commands[:decline_count]
            }
            attack_success_ids = {
                command.payload["actor_id"]
                for command, label in zip(
                    opening_commands[decline_count:],
                    class_labels[decline_count:],
                    strict=True,
                )
                if label
            }
            if not probe_actor_ids & attack_success_ids:
                raise ValueError("probe and escalation accounts lack causal reuse")
            probe_delay_floor, success_delay_ceiling = _card_delay_regions(
                params.min_delay_seconds,
                params.max_delay_seconds,
            )
            command_positions = {
                id(command): index for index, command in enumerate(commands)
            }
            probe_delays: list[int] = []
            for command in opening_commands[:decline_count]:
                index = command_positions[id(command)]
                prior_timestamp = (
                    population.generated_at if index == 0 else schedule[index - 1]
                )
                probe_delay = int(
                    (schedule[index] - prior_timestamp).total_seconds()
                )
                probe_delays.append(probe_delay)
                if probe_delay < probe_delay_floor:
                    raise ValueError("card probe delay is outside its temporal region")
            success_payment_ids = {
                cast(str, command.payload["payment_id"])
                for command, label in zip(
                    opening_commands[decline_count:],
                    class_labels[decline_count:],
                    strict=True,
                )
                if label
            }
            success_delays: list[int] = []
            for index, command in enumerate(commands):
                if cast(str, command.payload["payment_id"]) not in success_payment_ids:
                    continue
                prior_timestamp = (
                    population.generated_at if index == 0 else schedule[index - 1]
                )
                success_delay = int(
                    (schedule[index] - prior_timestamp).total_seconds()
                )
                success_delays.append(success_delay)
                if success_delay > success_delay_ceiling:
                    raise ValueError("card success burst exceeds its temporal region")
            if (
                not probe_delays
                or not success_delays
                or min(probe_delays) <= max(success_delays)
            ):
                raise ValueError("card probe delays do not exceed the success burst")
        if abs(rate - params.target_illicit_rate) > params.class_rate_tolerance:
            raise ValueError("class rate constraint not satisfied")
        if (
            family == "agentic_intent_abuse"
            and abs(rate - params.agentic_attack_mix) > params.class_rate_tolerance
        ):
            raise ValueError("agentic attack mix constraint not satisfied")
        if abs(value_total - params.target_value_total) > params.value_tolerance:
            raise ValueError("value total constraint not satisfied")
        dependencies = self._command_dependencies(family, opening_commands)
        if family == "app_scam_mule":
            mule_ids = {
                entity_id
                for command in opening_commands
                for entity_id in (
                    cast(str, command.payload["actor_id"]),
                    cast(str, command.payload["counterparty_id"]),
                )
                if entity_by_id[entity_id].role == "mule"
            }
            if len(mule_ids) != params.mule_count:
                raise ValueError("APP topology does not use the declared mule count")
            layer_count = sum(
                entity_by_id[cast(str, command.payload["actor_id"])].role == "mule"
                and entity_by_id[
                    cast(str, command.payload["counterparty_id"])
                ].role
                == "mule"
                for command in opening_commands
            )
            if layer_count != params.mule_layers:
                raise ValueError("APP topology does not use the declared layer count")
            cash_commands = tuple(
                command
                for command in opening_commands
                if entity_by_id[cast(str, command.payload["actor_id"])].role == "mule"
                and entity_by_id[
                    cast(str, command.payload["counterparty_id"])
                ].role
                == "attacker"
            )
            if len(cash_commands) != params.mule_fanout:
                raise ValueError("APP topology does not use the declared fan-out")
            expected_recoveries = _recovery_count(
                params.recovery_probability,
                params.mule_fanout,
                require_unrecovered=False,
            )
            actual_recoveries = sum(
                "recover" in histories[cast(str, command.payload["payment_id"])]
                for command in cash_commands
            )
            if expected_recoveries == 0 or actual_recoveries != expected_recoveries:
                raise ValueError("APP recovery count differs from visible probability")
            cash_total = sum(
                (cast(Decimal, command.payload["amount"]) for command in cash_commands),
                Decimal("0.00"),
            )
            expected_cash = _cash_out_target(
                params.cash_out_fraction,
                params.target_value_total,
            )
            if cash_total != expected_cash:
                raise ValueError("APP cash-out fraction differs from concrete flow")
            dependency_ids = {dependency.payment_id for dependency in dependencies}
            mule_outflow_ids = {
                cast(str, command.payload["payment_id"])
                for command in opening_commands
                if entity_by_id[cast(str, command.payload["actor_id"])].role == "mule"
            }
            if not mule_outflow_ids <= dependency_ids:
                raise ValueError("APP mule outflow lacks a prior concrete funding dependency")
            cash_positions = tuple(
                index
                for index, command in enumerate(commands)
                if command.name == "a2a.initiate"
                and entity_by_id[cast(str, command.payload["actor_id"])].role
                == "mule"
                and entity_by_id[
                    cast(str, command.payload["counterparty_id"])
                ].role
                == "attacker"
            )
            if params.cash_out_strategy == "staged" and (
                len(cash_positions) < 2
                or params.cash_out_delay_seconds <= params.min_delay_seconds
            ):
                raise ValueError("APP staged delay parameter is incompatible")
            if (
                params.cash_out_strategy == "delayed"
                and params.cash_out_delay_seconds <= params.min_delay_seconds
            ):
                raise ValueError("APP delayed delay parameter is incompatible")
            for cash_order, index in enumerate(cash_positions):
                timestamp = schedule[index]
                prior_timestamp = (
                    population.generated_at if index == 0 else schedule[index - 1]
                )
                delay_seconds = int((timestamp - prior_timestamp).total_seconds())
                if (
                    params.cash_out_strategy == "burst"
                    and params.cash_out_delay_seconds != params.min_delay_seconds
                ):
                    raise ValueError("APP burst delay parameter is incompatible")
                if (
                    params.cash_out_strategy == "burst"
                    and delay_seconds != params.min_delay_seconds
                ):
                    raise ValueError("APP burst cash-out timing drifted")
                if (
                    params.cash_out_strategy == "delayed"
                    and delay_seconds != params.cash_out_delay_seconds
                ):
                    raise ValueError("APP delayed cash-out timing drifted")
                if (
                    params.cash_out_strategy == "staged"
                    and delay_seconds > params.cash_out_delay_seconds
                ):
                    raise ValueError("APP staged cash-out timing drifted")
                if (
                    params.cash_out_strategy == "staged"
                    and cash_order == 0
                    and delay_seconds != params.min_delay_seconds
                ):
                    raise ValueError("APP staged cash-out lacks its opening witness")
                if (
                    params.cash_out_strategy == "staged"
                    and cash_order == len(cash_positions) - 1
                    and delay_seconds != params.cash_out_delay_seconds
                ):
                    raise ValueError("APP staged cash-out lacks its delay witness")
        elif family == "synthetic_merchant_refund":
            illicit_count = sum(class_labels)
            expected_recoveries = _recovery_count(
                params.recovery_probability,
                illicit_count,
                require_unrecovered=True,
            )
            actual_recoveries = sum(command.name == "card.recover" for command in commands)
            if (
                expected_recoveries == 0
                or expected_recoveries >= illicit_count
                or actual_recoveries != expected_recoveries
            ):
                raise ValueError("recovery lifecycle count differs from visible probability")
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
            payment_count=len(opening_commands),
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
        *,
        schedule: tuple[datetime, ...] | None = None,
        fixture: AgenticFixture | None = None,
    ) -> CampaignEvidence:
        """Fail closed when a concrete externally supplied candidate is incomplete."""
        try:
            if schedule is None:
                raise ValueError("external validation requires the concrete schedule")
            return self.__generator._validate_candidate(
                family,
                commands,
                schedule,
                population,
                params,
                1,
                fixture,
            )
        except (
            ArithmeticError,
            DecimalException,
            KeyError,
            RuntimeError,
            TypeError,
            ValueError,
        ) as error:
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
