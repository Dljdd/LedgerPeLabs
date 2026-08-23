"""Mixed population builder with real executed fraud evidence for Sentinel v5."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Self, cast
from uuid import NAMESPACE_URL, uuid5

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from apar.contracts.decisions import Action
from apar.contracts.events import Rail
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
    CampaignParams,
    _CampaignEvaluator,
)
from apar.generators.population import PopulationGenerator
from apar.simulator.engine import SimulationEngine
from apar.simulator.ledger import AccountReference
from apar.simulator.rails import (
    A2ARailAdapter,
    AgenticRailAdapter,
    CardRailAdapter,
)
from apar.simulator.rails.base import AdapterFactory
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
    command_name: str
    event_id: str
    payment_id: str
    actor_id: str
    counterparty_id: str
    lifecycle_position: int = Field(ge=0)
    is_fraud: bool


class V5ExecutionManifest(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    evidence_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    campaign_id: str
    family: str
    rail: str
    lineage: tuple[V5LineageManifest, ...]
    ledger_entry_ids: tuple[str, ...]
    trust_request_ids: tuple[str, ...]
    trust_failure_event_ids: tuple[str, ...]
    account_ids: tuple[str, ...]

    @model_validator(mode="after")
    def validate_manifest(self) -> Self:
        if self.family not in {family.value for family in V5Family}:
            raise ValueError("execution manifest family is unknown")
        if self.rail not in {rail.value for rail in Rail}:
            raise ValueError("execution manifest rail is unknown")
        if not self.lineage:
            raise ValueError("execution manifest lineage must not be empty")
        event_ids = tuple(item.event_id for item in self.lineage)
        command_ids = tuple(item.command_id for item in self.lineage)
        if len(event_ids) != len(set(event_ids)):
            raise ValueError("execution manifest event IDs must be unique")
        if len(command_ids) != len(set(command_ids)):
            raise ValueError("execution manifest command IDs must be unique")
        if not set(self.trust_failure_event_ids) <= set(event_ids):
            raise ValueError("trust failures must reference manifest events")
        return self


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
            if row.family == "legitimate":
                if row.execution_evidence_sha256:
                    raise ValueError(
                        "legitimate construction cannot claim fraud execution evidence"
                    )
                continue
            manifest_candidate = manifests.get(row.execution_evidence_sha256)
            if manifest_candidate is None:
                raise ValueError("campaign row lacks real execution evidence")
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


_OPERATIONAL_PARTITIONS = ("train", "calibration", "threshold", "development_test")
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
_LEGITIMATE_ACTORS_PER_PARTITION = 40


class _PopulationIsolationError(ValueError):
    """Raised when corpus identities, time, or provenance cross partitions."""


def _domain_seed(base_seed: int, *labels: object) -> int:
    document = ":".join(
        ["apar", "sentinel-v5", "development", str(base_seed), *(str(item) for item in labels)]
    )
    derived = int.from_bytes(hashlib.sha256(document.encode()).digest()[:8], "big")
    derived &= 2**63 - 1
    return derived or 1


def _build_benign_partition(
    partition_name: str,
    count: int,
    seed_value: int,
) -> list[V5DecisionRow]:
    import random

    rng = random.Random(_domain_seed(seed_value, partition_name, "benign"))
    partition_start = _BASE_START + timedelta(
        days=_PARTITION_OFFSETS_DAYS[partition_name]
    )
    window_seconds = 8 * 24 * 60 * 60
    slot_seconds = max(window_seconds // max(count, 1), 1)
    rows: list[V5DecisionRow] = []
    for i in range(count):
        actor_index = i % _LEGITIMATE_ACTORS_PER_PARTITION
        actor_id = f"benign-{partition_name}-actor-{actor_index:04d}"
        counterparty_id = f"benign-{partition_name}-counterparty-{i % 60:04d}"
        amount_cents = rng.randint(100, 50_000)
        decision_at = partition_start + timedelta(seconds=i * slot_seconds)
        event_id = f"{partition_name}-benign-event-{i:06d}"
        rail = ("card", "a2a")[i % 2]
        rows.append(
            V5DecisionRow(
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
                rail=rail,
                integrity_status="not_applicable",
                predictive_features={
                    "amount": float(amount_cents) / 100.0,
                    "rail_card": float(rail == "card"),
                    "rail_a2a": float(rail == "a2a"),
                    "rail_agentic": 0.0,
                    "integrity_pass": 0.0,
                    "txn_hour_sin": math.sin(2 * math.pi * decision_at.hour / 24),
                    "txn_hour_cos": math.cos(2 * math.pi * decision_at.hour / 24),
                },
            )
        )
    return rows


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


def _manifest_from_evidence(evidence: object) -> V5ExecutionManifest:
    from apar.evaluation.v5_execution import V5ExecutionEvidence

    if type(evidence) is not V5ExecutionEvidence:
        raise TypeError("manifest requires exact validated execution evidence")
    account_ids = {
        value
        for command in evidence.commands
        for key, value in command.payload.items()
        if key.endswith("_account")
        and type(value) is str
        and value.startswith("acct:")
    }
    trust_failures = tuple(
        record.event_id for record in evidence.trust_evidence if not record.receipt.allowed
    )
    return V5ExecutionManifest(
        evidence_sha256=evidence.evidence_sha256,
        campaign_id=evidence.campaign_id,
        family=evidence.family,
        rail=evidence.rail.value,
        lineage=tuple(
            V5LineageManifest(
                command_id=item.command_id,
                command_name=item.command_name,
                event_id=item.event_id,
                payment_id=item.payment_id,
                actor_id=item.actor_id,
                counterparty_id=item.counterparty_id,
                lifecycle_position=item.lifecycle_position,
                is_fraud=item.is_fraud,
            )
            for item in evidence.lineage
        ),
        ledger_entry_ids=tuple(entry.entry_id for entry in evidence.ledger_entries),
        trust_request_ids=tuple(
            record.request.request_id for record in evidence.trust_evidence
        ),
        trust_failure_event_ids=trust_failures,
        account_ids=tuple(sorted(account_ids)),
    )


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
    return project_execution_evidence(evidence), _manifest_from_evidence(evidence)


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
    accounts = {
        name: {
            account
            for execution in partitions[name].executions
            for account in execution.account_ids
        }
        for name in names
    }
    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            if accounts[left] & accounts[right]:
                raise _PopulationIsolationError(
                    f"account identity overlap between {left} and {right}"
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
    if profile is V5Profile.SMOKE:
        smoke_base = min(profile_counts.legitimate_decisions, 500)
        smoke_per = max(smoke_base // len(_OPERATIONAL_PARTITIONS), 1)
        partition_legitimate_counts = {
            "train": smoke_per,
            "calibration": smoke_per,
            "threshold": smoke_per,
            "development_test": smoke_per,
            "hardening_train": max(smoke_per // 2, 1),
            "adaptive_holdout": max(smoke_per // 2, 1),
        }
    else:
        prod_legit = protocol.production_profile.legitimate_decisions
        partition_legitimate_counts = {
            "train": prod_legit // 4,
            "calibration": prod_legit // 8,
            "threshold": prod_legit // 8,
            "development_test": protocol.production_dev_test_legitimate,
            "hardening_train": prod_legit // 8,
            "adaptive_holdout": prod_legit // 8,
        }

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
        rows = _build_benign_partition(
            partition_name,
            partition_legitimate_counts[partition_name],
            seed,
        )
        executions: list[V5ExecutionManifest] = []
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
    "V5ExecutionManifest",
    "V5LineageManifest",
    "V5PartitionCorpus",
    "build_v5_corpus",
]
