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

_BENIGN_ROLES = (
    "victim",
    "consumer",
    "merchant",
    "beneficiary",
    "device",
    "agent",
    "organization",
    "institution",
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
class Population:
    """Canonical graph, balances, and benign controls consumed by campaigns."""

    scenario_id: str
    seed: int
    generated_at: datetime
    horizon_end: datetime
    entities: tuple[PopulationEntity, ...]
    relationships: tuple[PopulationRelationship, ...]
    opening_balances: Mapping[str, Decimal]
    benign_controls: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "scenario_id", _text("scenario_id", self.scenario_id))
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
        balances: dict[str, Decimal] = {}
        if type(self.opening_balances) not in (dict, type(MappingProxyType({}))):
            raise TypeError("opening_balances must be an exact mapping")
        for account, amount in self.opening_balances.items():
            checked_account = _text("opening balance account", account)
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

    def by_role(self, role: str) -> tuple[PopulationEntity, ...]:
        """Return entities with an exact declared role in canonical order."""
        checked_role = _text("role", role)
        return tuple(entity for entity in self.entities if entity.role == checked_role)

    def canonical_bytes(self) -> bytes:
        """Serialize the public graph without interpreter-specific object details."""
        document = {
            "scenario_id": self.scenario_id,
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
            "opening_balances": {
                account: str(amount) for account, amount in self.opening_balances.items()
            },
            "benign_controls": self.benign_controls,
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
        if bundle.benign_entity_count < 8 or bundle.illicit_entity_count < 4:
            raise ValueError("population requires at least 8 benign and 4 illicit entities")
        composite_seed = self._seed ^ bundle.seed
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
        beneficiary = next(entity for entity in benign if entity.role == "beneficiary")
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
        return Population(
            scenario_id=bundle.scenario_id,
            seed=self._seed,
            generated_at=owned_start,
            horizon_end=owned_start + timedelta(hours=bundle.duration_hours),
            entities=tuple(entities),
            relationships=tuple(relationships),
            opening_balances=balances,
            benign_controls=("new_merchant", "shared_beneficiary", "shared_device", "travel"),
        )


__all__ = [
    "Population",
    "PopulationEntity",
    "PopulationGenerator",
    "PopulationRelationship",
]
