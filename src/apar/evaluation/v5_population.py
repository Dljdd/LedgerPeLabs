"""Mixed, group-disjoint population builder for Defend v5."""

from __future__ import annotations

import hashlib
import json
import math
from datetime import UTC, datetime, timedelta
from decimal import Decimal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from apar.evaluation.v5_protocol import V5DevelopmentProtocol, V5Family, V5Profile


class V5DecisionRow(BaseModel):
    model_config = ConfigDict(frozen=True)

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
    predictive_features: dict[str, float] = Field(default_factory=dict)

    @field_validator("decision_at", mode="before")
    @classmethod
    def require_utc(cls, value: object) -> object:
        if isinstance(value, datetime) and value.tzinfo is None:
            raise ValueError("decision_at must be UTC")
        return value


class V5PartitionCorpus(BaseModel):
    model_config = ConfigDict(frozen=True)

    partition_name: str
    decisions: tuple[V5DecisionRow, ...]

    @property
    def fraud_count(self) -> int:
        return sum(1 for row in self.decisions if row.is_fraud)

    @property
    def benign_count(self) -> int:
        return sum(1 for row in self.decisions if not row.is_fraud)


class V5Corpus(BaseModel):
    model_config = ConfigDict(frozen=True)

    profile: V5Profile
    partitions: dict[str, V5PartitionCorpus]
    corpus_sha256: str
    is_production: bool


_OPERATIONAL_PARTITIONS = ("train", "calibration", "threshold", "development_test")
_FAMILY_PREFIXES = {
    V5Family.AGENTIC_INTENT_ABUSE.value: "agentic",
    V5Family.APP_SCAM_MULE.value: "appscam",
    V5Family.CARD_TESTING_CNP.value: "cardtest",
    V5Family.SYNTHETIC_MERCHANT_REFUND.value: "synthrefund",
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
_PARTITION_SEED_KEYS = (
    "train", "calibration", "threshold", "development_test",
    "hardening_train", "adaptive_holdout",
)
_LEGITIMATE_ACTORS_PER_PARTITION = 40
_CAMPAIGNS_PER_FAMILY_SMOKE = 2


class _PopulationIsolationError(ValueError):
    """Raised when the proposed corpus violates isolation or completeness rules."""


def _build_benign_partition(
    partition_name: str,
    count: int,
    seed_value: int,
) -> list[V5DecisionRow]:
    import random

    rng = random.Random(seed_value)
    offset_days = _PARTITION_OFFSETS_DAYS.get(partition_name, 0)
    rows: list[V5DecisionRow] = []
    for i in range(count):
        actor_index = i % _LEGITIMATE_ACTORS_PER_PARTITION
        actor_id = f"benign-{partition_name}-actor-{actor_index:04d}"
        counterparty_id = f"benign-{partition_name}-counterparty-{i % 60:04d}"
        amount_cents = rng.randint(100, 50_000)
        hour = rng.randint(0, 23)
        minute = rng.randint(0, 59)
        day_offset = offset_days + (i // _LEGITIMATE_ACTORS_PER_PARTITION)
        decision_at = _BASE_START + timedelta(days=day_offset, hours=hour, minutes=minute)
        event_id = f"{partition_name}-benign-event-{i:06d}"
        rows.append(V5DecisionRow(
            event_id=event_id,
            payment_id=f"payment-{event_id}",
            campaign_id=f"benign-base-{partition_name}",
            family="legitimate",
            actor_id=actor_id,
            counterparty_id=counterparty_id,
            amount=Decimal(amount_cents) / Decimal("100"),
            currency="USD",
            decision_at=decision_at,
            is_fraud=False,
            rail=("card", "a2a", "agentic")[i % 3],
            integrity_status="pass" if i % 3 == 2 else "not_applicable",
            predictive_features={
                "amount": float(amount_cents) / 100.0,
                "rail_card": float(i % 3 == 0),
                "rail_a2a": float(i % 3 == 1),
                "rail_agentic": float(i % 3 == 2),
                "integrity_pass": float(i % 3 == 2),
                "txn_hour_sin": math.sin(2 * math.pi * hour / 24),
                "txn_hour_cos": math.cos(2 * math.pi * hour / 24),
            },
        ))
    return rows


def _build_fraud_campaigns_for_partition(
    partition_name: str,
    campaigns_per_family: dict[str, int],
    seed_value: int,
) -> list[V5DecisionRow]:
    import random

    domain_input = f"fraud_campaign:{partition_name}".encode()
    partition_offset = int.from_bytes(hashlib.sha256(domain_input).digest()[:4], "big") % 10000
    rng = random.Random(seed_value + partition_offset)
    offset_days = _PARTITION_OFFSETS_DAYS.get(partition_name, 0)
    rows: list[V5DecisionRow] = []
    campaign_counter = 0
    for family_value in sorted(campaigns_per_family):
        prefix = _FAMILY_PREFIXES.get(family_value, family_value[:8])
        for c in range(campaigns_per_family[family_value]):
            campaign_id = f"campaign-{partition_name}-{prefix}-{c:04d}"
            actor_id = f"fraud-{partition_name}-{prefix}-actor-{campaign_counter:05d}"
            counterparty_id = f"fraud-{partition_name}-{prefix}-cp-{campaign_counter:05d}"
            decisions_in_campaign = rng.randint(3, 8)
            start_hour = rng.randint(6, 22)
            start_minute = rng.randint(0, 59)
            base_amount = rng.randint(200, 80_000)
            for d in range(decisions_in_campaign):
                decision_at = _BASE_START + timedelta(
                    days=offset_days,
                    hours=start_hour,
                    minutes=start_minute + d * rng.randint(1, 15),
                )
                amount_cents = base_amount + d * rng.randint(-50, 500)
                event_id = f"{campaign_id}-event-{d:03d}"
                rail = {
                    "agentic": "agentic",
                    "appscam": "a2a",
                    "cardtest": "card",
                    "synthrefund": "card",
                }.get(prefix, "card")
                integrity = (
                    "fail"
                    if (family_value == "agentic_intent_abuse" and d == 0)
                    else "pass"
                )
                rows.append(V5DecisionRow(
                    event_id=event_id,
                    payment_id=f"payment-{event_id}",
                    campaign_id=campaign_id,
                    family=family_value,
                    actor_id=actor_id,
                    counterparty_id=counterparty_id,
                    amount=Decimal(max(amount_cents, 50)) / Decimal("100"),
                    currency="USD",
                    decision_at=decision_at,
                    is_fraud=True,
                    rail=rail,
                    integrity_status=integrity if rail == "agentic" else "not_applicable",
                    predictive_features={
                        "amount": max(amount_cents, 50) / 100.0,
                        f"rail_{rail}": 1.0,
                        "integrity_pass": float(integrity == "pass") if rail == "agentic" else 0.0,
                        "txn_hour_sin": math.sin(2 * math.pi * decision_at.hour / 24),
                        "txn_hour_cos": math.cos(2 * math.pi * decision_at.hour / 24),
                    },
                ))
            campaign_counter += 1
    return rows


def build_v5_corpus(
    protocol: V5DevelopmentProtocol,
    *,
    profile: V5Profile,
) -> V5Corpus:
    """Build all partitions using real generators, rails, and strict group isolation."""
    profile_counts = (
        protocol.smoke_profile
        if profile is V5Profile.SMOKE
        else protocol.production_profile
    )
    legitimate_total = (
        min(profile_counts.legitimate_decisions, 500)
        if profile is V5Profile.SMOKE
        else profile_counts.legitimate_decisions
    )
    per_operational = legitimate_total // len(_OPERATIONAL_PARTITIONS)

    partitions: dict[str, list[V5DecisionRow]] = {}
    all_actors: dict[str, set[str]] = {name: set() for name in _PARTITION_SEED_KEYS}
    all_campaigns: dict[str, set[str]] = {name: set() for name in _PARTITION_SEED_KEYS}

    seed_map = {
        "train": protocol.seeds.train,
        "calibration": protocol.seeds.calibration,
        "threshold": protocol.seeds.threshold,
        "development_test": protocol.seeds.development_test,
        "hardening_train": protocol.seeds.hardening_train,
        "adaptive_holdout": protocol.seeds.adaptive_holdout,
    }

    campaigns_for_profile = (
        {f: _CAMPAIGNS_PER_FAMILY_SMOKE for f in _FAMILY_PREFIXES}
        if profile is V5Profile.SMOKE
        else profile_counts.campaigns_per_family
    )

    for partition_name in _PARTITION_SEED_KEYS:
        seed = seed_map[partition_name]
        benign_rows = _build_benign_partition(partition_name, per_operational, seed)
        fraud_rows = _build_fraud_campaigns_for_partition(
            partition_name, campaigns_for_profile, seed
        )
        partitions[partition_name] = benign_rows + fraud_rows
        all_actors[partition_name] = {r.actor_id for r in partitions[partition_name]}
        all_campaigns[partition_name] = {r.campaign_id for r in partitions[partition_name]}

    names = list(_PARTITION_SEED_KEYS)
    for i, left in enumerate(names):
        for right in names[i + 1:]:
            actor_overlap = all_actors[left] & all_actors[right]
            campaign_overlap = all_campaigns[left] & all_campaigns[right]
            if actor_overlap:
                raise _PopulationIsolationError(
                    f"actor identity overlap between {left} and {right}"
                )
            if campaign_overlap:
                raise _PopulationIsolationError(
                    f"campaign overlap between {left} and {right}"
                )

    partition_models = {
        name: V5PartitionCorpus(
            partition_name=name,
            decisions=tuple(sorted(rows, key=lambda r: (r.decision_at, r.event_id))),
        )
        for name, rows in partitions.items()
    }

    digest_content = json.dumps(
        {
            name: [row.model_dump(mode="json") for row in part.decisions]
            for name, part in partition_models.items()
        },
        sort_keys=True,
    ).encode()
    corpus_sha256 = hashlib.sha256(digest_content).hexdigest()

    return V5Corpus(
        profile=profile,
        partitions=partition_models,
        corpus_sha256=corpus_sha256,
        is_production=(profile is V5Profile.PRODUCTION),
    )


__all__ = ["V5Corpus", "V5DecisionRow", "V5PartitionCorpus", "build_v5_corpus"]
