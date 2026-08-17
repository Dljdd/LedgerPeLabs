"""Deterministic synthetic population and institution graph generation."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from types import MappingProxyType
from typing import cast
from uuid import NAMESPACE_URL, UUID, uuid5

import numpy as np

from apar.contracts.scenarios import ScenarioBundle
from apar.simulator.clock import Command
from apar.simulator.rails.a2a import AcceptA2A, InitiateA2A, PostA2A

_BENIGN_ROLES = (
    "victim",
    "consumer",
    "merchant",
    "beneficiary",
    "device",
    "agent",
    "organization",
    "institution",
    "merchant_location",
)
_ILLICIT_ROLES = ("mule", "synthetic_merchant", "attacker", "compromised_credential")
_CHANNELS = ("mobile", "web", "branch", "agent")
_COUNTRIES = ("GB", "IN", "SG", "US")


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


def _utc(label: str, value: object) -> datetime:
    if type(value) is not datetime or value.tzinfo is not UTC:
        raise ValueError(f"{label} must be an exact UTC datetime")
    timestamp = value
    return datetime(
        timestamp.year,
        timestamp.month,
        timestamp.day,
        timestamp.hour,
        timestamp.minute,
        timestamp.second,
        timestamp.microsecond,
        tzinfo=UTC,
        fold=timestamp.fold,
    )


def _closed_text_mapping(label: str, values: object) -> Mapping[str, str]:
    if type(values) not in (dict, type(MappingProxyType({}))):
        raise TypeError(f"{label} must be an exact mapping")
    owned: dict[str, str] = {}
    for key, value in cast(Mapping[object, object], values).items():
        checked_key = _text(f"{label} key", key)
        owned[checked_key] = _text(f"{label} value", value)
    return MappingProxyType(dict(sorted(owned.items())))


def _canonical_command_payload(command: Command) -> dict[str, object]:
    payload: dict[str, object] = {}
    for key, value in command.payload.items():
        if type(value) is Decimal:
            payload[key] = {"decimal": str(value)}
        elif type(value) is datetime:
            payload[key] = value.isoformat().replace("+00:00", "Z")
        elif type(value) in (str, int, float, bool, bytes) or value is None:
            payload[key] = value.hex() if type(value) is bytes else value
        elif type(value) is tuple:
            payload[key] = list(value)
        else:
            raise TypeError("benign command payload contains unsupported value")
    return payload


@dataclass(frozen=True, slots=True)
class PopulationEntity:
    """One pseudonymous actor, endpoint, institution, or technical entity."""

    entity_id: str
    role: str
    account_id: str | None
    illicit: bool
    attributes: Mapping[str, str]

    def __post_init__(self) -> None:
        object.__setattr__(self, "entity_id", _uuid_text("entity_id", self.entity_id))
        object.__setattr__(self, "role", _text("role", self.role))
        if self.account_id is not None:
            object.__setattr__(self, "account_id", _text("account_id", self.account_id))
        if type(self.illicit) is not bool:
            raise TypeError("illicit must be an exact boolean")
        object.__setattr__(
            self,
            "attributes",
            _closed_text_mapping("attributes", self.attributes),
        )


@dataclass(frozen=True, slots=True)
class PopulationRelationship:
    """One explicit, typed edge in the synthetic entity graph."""

    source_id: str
    target_id: str
    relation: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_id", _uuid_text("source_id", self.source_id))
        object.__setattr__(self, "target_id", _uuid_text("target_id", self.target_id))
        object.__setattr__(self, "relation", _text("relation", self.relation))


@dataclass(frozen=True, slots=True)
class PopulationAccount:
    """Explicit account ownership and institution binding."""

    account_id: str
    owner_entity_id: str
    institution_entity_id: str
    currency: str
    opening_balance: Decimal

    def __post_init__(self) -> None:
        object.__setattr__(self, "account_id", _text("account_id", self.account_id))
        object.__setattr__(
            self,
            "owner_entity_id",
            _uuid_text("owner_entity_id", self.owner_entity_id),
        )
        object.__setattr__(
            self,
            "institution_entity_id",
            _uuid_text("institution_entity_id", self.institution_entity_id),
        )
        if self.currency != "USD":
            raise ValueError("population accounts currently support USD only")
        if (
            type(self.opening_balance) is not Decimal
            or not self.opening_balance.is_finite()
            or self.opening_balance < 0
            or self.opening_balance != self.opening_balance.quantize(Decimal("0.01"))
        ):
            raise ValueError("opening_balance must be a non-negative canonical Decimal")


@dataclass(frozen=True, slots=True)
class BenignActivity:
    """One scheduled benign shift tied to real population entities and commands."""

    activity_id: str
    shift: str
    scheduled_at: datetime
    actor_id: str
    counterparty_id: str
    device_id: str
    merchant_location_id: str
    payment_id: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "activity_id", _uuid_text("activity_id", self.activity_id))
        object.__setattr__(self, "shift", _text("shift", self.shift))
        object.__setattr__(self, "scheduled_at", _utc("scheduled_at", self.scheduled_at))
        for field_name in (
            "actor_id",
            "counterparty_id",
            "device_id",
            "merchant_location_id",
        ):
            object.__setattr__(
                self,
                field_name,
                _uuid_text(field_name, getattr(self, field_name)),
            )
        object.__setattr__(self, "payment_id", _text("payment_id", self.payment_id))


@dataclass(frozen=True, slots=True)
class Population:
    """Canonical graph, balances, and benign controls consumed by campaigns."""

    scenario_id: str
    bundle: ScenarioBundle
    seed: int
    generated_at: datetime
    horizon_end: datetime
    entities: tuple[PopulationEntity, ...]
    relationships: tuple[PopulationRelationship, ...]
    accounts: tuple[PopulationAccount, ...]
    opening_balances: Mapping[str, Decimal]
    benign_controls: tuple[str, ...]
    benign_activities: tuple[BenignActivity, ...]
    benign_commands: tuple[Command, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "scenario_id", _text("scenario_id", self.scenario_id))
        if type(self.bundle) is not ScenarioBundle or self.bundle.scenario_id != self.scenario_id:
            raise TypeError("bundle must be the exact source ScenarioBundle")
        if type(self.seed) is not int:
            raise TypeError("seed must be an exact integer")
        started = _utc("generated_at", self.generated_at)
        ended = _utc("horizon_end", self.horizon_end)
        if ended <= started:
            raise ValueError("horizon_end must be after generated_at")
        object.__setattr__(self, "generated_at", started)
        object.__setattr__(self, "horizon_end", ended)
        if type(self.entities) is not tuple or not self.entities:
            raise TypeError("entities must be a non-empty exact tuple")
        if any(type(entity) is not PopulationEntity for entity in self.entities):
            raise TypeError("entities must contain exact PopulationEntity records")
        identifiers = {entity.entity_id for entity in self.entities}
        if len(identifiers) != len(self.entities):
            raise ValueError("population entity IDs must be unique")
        if type(self.relationships) is not tuple:
            raise TypeError("relationships must be an exact tuple")
        for edge in self.relationships:
            if type(edge) is not PopulationRelationship:
                raise TypeError("relationships must contain exact relationship records")
            if edge.source_id not in identifiers or edge.target_id not in identifiers:
                raise ValueError("relationships must reference declared entities")
        if type(self.accounts) is not tuple or any(
            type(account) is not PopulationAccount for account in self.accounts
        ):
            raise TypeError("accounts must contain exact PopulationAccount records")
        for account in self.accounts:
            if account.owner_entity_id not in identifiers:
                raise ValueError("account owner must reference a declared entity")
            if account.institution_entity_id not in identifiers:
                raise ValueError("account institution must reference a declared entity")
        balances: dict[str, Decimal] = {}
        if type(self.opening_balances) not in (dict, type(MappingProxyType({}))):
            raise TypeError("opening_balances must be an exact mapping")
        for balance_account, amount in self.opening_balances.items():
            checked_account = _text("opening balance account", balance_account)
            if type(amount) is not Decimal:
                raise TypeError("opening balances must be exact Decimals")
            if not amount.is_finite() or amount < 0 or amount != amount.quantize(Decimal("0.01")):
                raise ValueError("opening balances must be non-negative canonical USD amounts")
            balances[checked_account] = amount
        declared_accounts = {
            entity.account_id for entity in self.entities if entity.account_id is not None
        }
        if not declared_accounts <= set(balances):
            raise ValueError("every entity account must have an opening balance")
        object.__setattr__(
            self,
            "opening_balances",
            MappingProxyType(dict(sorted(balances.items()))),
        )
        if type(self.benign_controls) is not tuple:
            raise TypeError("benign_controls must be an exact tuple")
        controls = tuple(_text("benign control", item) for item in self.benign_controls)
        if controls != tuple(sorted(set(controls))):
            raise ValueError("benign_controls must be unique and sorted")
        if type(self.benign_activities) is not tuple or any(
            type(activity) is not BenignActivity for activity in self.benign_activities
        ):
            raise TypeError("benign_activities must contain exact BenignActivity records")
        for activity in self.benign_activities:
            if not {
                activity.actor_id,
                activity.counterparty_id,
                activity.device_id,
                activity.merchant_location_id,
            } <= identifiers:
                raise ValueError("benign activity must reference declared entities")
            if not started <= activity.scheduled_at <= ended:
                raise ValueError("benign activity must stay within the population horizon")
        if type(self.benign_commands) is not tuple or any(
            not isinstance(command, Command) for command in self.benign_commands
        ):
            raise TypeError("benign_commands must be an exact tuple of Commands")

    def by_role(self, role: str) -> tuple[PopulationEntity, ...]:
        """Return entities with an exact declared role in canonical order."""
        checked_role = _text("role", role)
        return tuple(entity for entity in self.entities if entity.role == checked_role)

    def canonical_bytes(self) -> bytes:
        """Serialize the public graph without interpreter-specific object details."""
        document = {
            "scenario_id": self.scenario_id,
            "bundle_version": self.bundle.version,
            "bundle_rail": self.bundle.rail.value,
            "seed": self.seed,
            "generated_at": self.generated_at.isoformat().replace("+00:00", "Z"),
            "horizon_end": self.horizon_end.isoformat().replace("+00:00", "Z"),
            "entities": [
                {
                    "entity_id": entity.entity_id,
                    "role": entity.role,
                    "account_id": entity.account_id,
                    "illicit": entity.illicit,
                    "attributes": dict(entity.attributes),
                }
                for entity in self.entities
            ],
            "relationships": [
                {
                    "source_id": edge.source_id,
                    "target_id": edge.target_id,
                    "relation": edge.relation,
                }
                for edge in self.relationships
            ],
            "accounts": [
                {
                    "account_id": account.account_id,
                    "owner_entity_id": account.owner_entity_id,
                    "institution_entity_id": account.institution_entity_id,
                    "currency": account.currency,
                    "opening_balance": str(account.opening_balance),
                }
                for account in self.accounts
            ],
            "opening_balances": {
                account: str(amount) for account, amount in self.opening_balances.items()
            },
            "benign_controls": self.benign_controls,
            "benign_activities": [
                {
                    "activity_id": activity.activity_id,
                    "shift": activity.shift,
                    "scheduled_at": activity.scheduled_at.isoformat().replace(
                        "+00:00", "Z"
                    ),
                    "actor_id": activity.actor_id,
                    "counterparty_id": activity.counterparty_id,
                    "device_id": activity.device_id,
                    "merchant_location_id": activity.merchant_location_id,
                    "payment_id": activity.payment_id,
                }
                for activity in self.benign_activities
            ],
            "benign_commands": [
                {"name": command.name, "payload": _canonical_command_payload(command)}
                for command in self.benign_commands
            ],
        }
        return json.dumps(document, sort_keys=True, separators=(",", ":")).encode()


class PopulationGenerator:
    """Build entity identities and graph structure before sampling leaf attributes."""

    def __init__(self, *, seed: int) -> None:
        if type(seed) is not int:
            raise TypeError("seed must be an exact integer")
        self._seed = seed
        self._rng = np.random.default_rng(seed)

    def generate(self, bundle: ScenarioBundle) -> Population:
        """Generate one closed population from visible scenario bounds."""
        if type(bundle) is not ScenarioBundle:
            raise TypeError("bundle must be an exact ScenarioBundle")
        if bundle.benign_entity_count < 9 or bundle.illicit_entity_count < 8:
            raise ValueError("population requires at least 9 benign and 8 illicit entities")
        composite_seed = int(
            np.random.SeedSequence([self._seed, bundle.seed]).generate_state(1)[0]
        )
        self._rng = np.random.default_rng(composite_seed)
        namespace = uuid5(NAMESPACE_URL, f"apar:population:{bundle.scenario_id}:{composite_seed}")
        entities: list[PopulationEntity] = []

        # Identity graph and account assignment come first.
        for illicit, count, roles, prefix in (
            (False, bundle.benign_entity_count, _BENIGN_ROLES, "benign"),
            (True, bundle.illicit_entity_count, _ILLICIT_ROLES, "illicit"),
        ):
            for index in range(count):
                role = roles[index % len(roles)]
                entity_id = str(uuid5(namespace, f"{prefix}:entity:{index}"))
                account_id = (
                    None if role in {"device", "compromised_credential"} else f"acct:{entity_id}"
                )
                # Leaf values are conditioned on the already-selected role.
                channel_pool = (
                    _CHANNELS[:2] if role in {"victim", "consumer", "mule"} else _CHANNELS
                )
                channel = channel_pool[int(self._rng.integers(0, len(channel_pool)))]
                country = _COUNTRIES[int(self._rng.integers(0, len(_COUNTRIES)))]
                entities.append(
                    PopulationEntity(
                        entity_id,
                        role,
                        account_id,
                        illicit,
                        {"channel": channel, "country": country, "synthetic": "true"},
                    )
                )

        relationships: list[PopulationRelationship] = []
        for index, entity in enumerate(entities):
            relationships.append(
                PopulationRelationship(
                    entity.entity_id,
                    entities[(index + 1) % len(entities)].entity_id,
                    "population_link",
                )
            )
        benign = [entity for entity in entities if not entity.illicit]
        device = next(entity for entity in benign if entity.role == "device")
        devices = [entity for entity in benign if entity.role == "device"]
        beneficiary = next(entity for entity in benign if entity.role == "beneficiary")
        merchants = [entity for entity in benign if entity.role == "merchant"]
        locations = [entity for entity in benign if entity.role == "merchant_location"]
        payers = [
            entity
            for entity in benign
            if entity.account_id and entity.role in {"victim", "consumer"}
        ]
        for entity in payers[:2]:
            relationships.append(
                PopulationRelationship(
                    entity.entity_id,
                    device.entity_id,
                    "shares_device",
                )
            )
        illicit_attackers = [
            entity for entity in entities if entity.illicit and entity.role == "attacker"
        ]
        for entity in illicit_attackers[:2]:
            relationships.append(
                PopulationRelationship(
                    entity.entity_id,
                    device.entity_id,
                    "uses_shared_device",
                )
            )
        relationships.extend(
            (
                PopulationRelationship(
                    merchants[0].entity_id,
                    locations[0].entity_id,
                    "located_at",
                ),
                PopulationRelationship(
                    merchants[-1].entity_id,
                    locations[-1].entity_id,
                    "new_merchant_location",
                ),
                PopulationRelationship(
                    payers[2].entity_id,
                    locations[-1].entity_id,
                    "travels_to",
                ),
            )
        )
        for entity in payers[2:4]:
            relationships.append(
                PopulationRelationship(
                    entity.entity_id,
                    beneficiary.entity_id,
                    "pays_shared_beneficiary",
                )
            )

        balances: dict[str, Decimal] = {}
        for entity in entities:
            if entity.account_id is None:
                continue
            if entity.role == "mule":
                amount = Decimal("0.00")
            elif entity.illicit:
                amount = Decimal("5000.00")
            else:
                amount = Decimal("10000.00")
            balances[entity.account_id] = amount
        for system_account in (
            "a2a:fees",
            "a2a:frozen",
            "card:holds",
            "card:fees",
            "card:chargebacks",
        ):
            balances[system_account] = Decimal("0.00")

        institution = next(entity for entity in benign if entity.role == "institution")
        accounts = tuple(
            PopulationAccount(
                entity.account_id,
                entity.entity_id,
                institution.entity_id,
                "USD",
                balances[entity.account_id],
            )
            for entity in entities
            if entity.account_id is not None
        )

        started = bundle.replay_manifest.simulation_start
        owned_start = datetime(
            started.year,
            started.month,
            started.day,
            started.hour,
            started.minute,
            started.second,
            started.microsecond,
            tzinfo=UTC,
        )
        benign_campaign_id = str(uuid5(namespace, "benign:campaign"))
        activities: list[BenignActivity] = []
        benign_commands: list[Command] = []
        activity_specs = (
            ("shared_device", payers[0], beneficiary, device, locations[0]),
            ("shared_beneficiary", payers[1], beneficiary, devices[-1], locations[0]),
            ("travel", payers[2], merchants[0], devices[-1], locations[-1]),
            ("new_merchant", payers[3], merchants[-1], device, locations[-1]),
        )
        for index, (shift, actor, counterparty, activity_device, location) in enumerate(
            activity_specs
        ):
            activity_id = str(uuid5(namespace, f"benign:activity:{index}"))
            payment_id = f"benign:{uuid5(namespace, f'benign:payment:{index}')}"
            scheduled_at = owned_start + timedelta(
                minutes=10 * (index + 1) + int(self._rng.integers(0, 5))
            )
            amount = Decimal(5 + int(self._rng.integers(0, 16))).quantize(Decimal("0.01"))
            trace_id = str(uuid5(namespace, f"benign:trace:{index}"))
            opening = InitiateA2A(
                payment_id,
                amount=amount,
                currency="USD",
                payer_account=cast(str, actor.account_id),
                payee_account=cast(str, counterparty.account_id),
                actor_id=actor.entity_id,
                counterparty_id=counterparty.entity_id,
                campaign_id=benign_campaign_id,
                trace_id=trace_id,
            )
            benign_commands.extend(
                (
                    opening,
                    AcceptA2A(
                        payment_id,
                        idempotency_key=(
                            f"a2a.accept:{payment_id}:campaign:{benign_campaign_id}"
                        ),
                    ),
                    PostA2A(
                        payment_id,
                        idempotency_key=(
                            f"a2a.post:{payment_id}:campaign:{benign_campaign_id}"
                        ),
                    ),
                )
            )
            activities.append(
                BenignActivity(
                    activity_id,
                    shift,
                    scheduled_at,
                    actor.entity_id,
                    counterparty.entity_id,
                    activity_device.entity_id,
                    location.entity_id,
                    payment_id,
                )
            )
        return Population(
            scenario_id=bundle.scenario_id,
            bundle=bundle,
            seed=self._seed,
            generated_at=owned_start,
            horizon_end=owned_start + timedelta(hours=bundle.duration_hours),
            entities=tuple(entities),
            relationships=tuple(relationships),
            accounts=accounts,
            opening_balances=balances,
            benign_controls=("new_merchant", "shared_beneficiary", "shared_device", "travel"),
            benign_activities=tuple(activities),
            benign_commands=tuple(benign_commands),
        )


__all__ = [
    "Population",
    "BenignActivity",
    "PopulationAccount",
    "PopulationEntity",
    "PopulationGenerator",
    "PopulationRelationship",
]
