"""Evaluator-owned Task 5 composition, fresh replay, and frozen defender benchmark.

Importing :mod:`apar.redteam` does not import this module.  It is intentionally the only
Task 6 module allowed to depend on campaign generators, rail adapters, trust verifier
fixtures, and simulator audit artifacts.
"""

from __future__ import annotations

import hashlib
import inspect
import itertools
import json
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass, fields
from decimal import ROUND_HALF_EVEN, Decimal, localcontext
from pathlib import Path
from types import MappingProxyType
from typing import cast

from apar.contracts.decisions import Action
from apar.contracts.events import EventKind, PaymentEvent, Rail
from apar.generators import (
    AGENTIC_INTENT_ABUSE_MOTIF,
    APP_SCAM_MULE_MOTIF,
    CARD_TESTING_CNP_MOTIF,
    SYNTHETIC_MERCHANT_REFUND_MOTIF,
    CampaignParams,
    Population,
    campaign_bytes,
)
from apar.generators.campaigns import (
    AgenticFixture,
    CampaignEvidence,
    GenerationConstraintError,
    _CampaignEvaluator,
)
from apar.redteam.policies import (
    PUBLIC_REASON_FAMILIES,
    AdaptiveParameter,
    AdaptiveVector,
    AttackCandidate,
    DomainKind,
    Feedback,
    ParameterBounds,
    ParameterDomain,
    reconstruct_candidate,
)
from apar.redteam.search import (
    DisclosureProfile,
    EvaluationContract,
    EvaluatorCapability,
    SearchAuthority,
)
from apar.simulator.clock import Command
from apar.simulator.engine import SimulationEngine
from apar.simulator.ledger import AccountReference, LedgerEntry
from apar.simulator.rails import A2ARailAdapter, AgenticRailAdapter, CardRailAdapter
from apar.simulator.rails.base import AdapterFactory
from apar.trust import TrustVerifier

_MOTIFS = MappingProxyType(
    {
        "agentic_intent_abuse": AGENTIC_INTENT_ABUSE_MOTIF,
        "app_scam_mule": APP_SCAM_MULE_MOTIF,
        "card_testing_cnp": CARD_TESTING_CNP_MOTIF,
        "synthetic_merchant_refund": SYNTHETIC_MERCHANT_REFUND_MOTIF,
    }
)
_OPENINGS = frozenset({"a2a.initiate", "card.authorize", "card.decline", "agentic.pay"})
_SEVERITY = MappingProxyType({Action.APPROVE: 0, Action.CHALLENGE: 1, Action.DECLINE: 2})
_EVALUATOR_VERSION = "task6-artifact-defender-v1"
_OBSERVABLE_FEATURE_NAMES = (
    "declined_authorizations",
    "max_distinct_payees_per_actor",
    "maximum_payee_share",
    "minimum_opening_gap_seconds",
    "opening_attempts",
)


class BenchmarkConfigurationError(ValueError):
    """Stable failure for mismatched hidden family/template/background configuration."""


def _canonical_bytes(document: object) -> bytes:
    return json.dumps(
        document,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode()


def _digest(document: object) -> str:
    return hashlib.sha256(_canonical_bytes(document)).hexdigest()


def _money(value: Decimal) -> Decimal:
    return value.quantize(Decimal("0.01"), rounding=ROUND_HALF_EVEN)


def _evaluator_dependency_digest() -> str:
    """Hash the exact evaluator, Task 5, replay, rail, ledger, and trust modules."""
    paths: set[Path] = set()
    for implementation in (
        _CampaignEvaluator,
        SimulationEngine,
        LedgerEntry,
        A2ARailAdapter,
        AgenticRailAdapter,
        CardRailAdapter,
        TrustVerifier,
        _fresh_replay,
    ):
        source = inspect.getsourcefile(implementation)
        if source is None:
            raise RuntimeError("evaluator dependency has no inspectable source")
        paths.add(Path(source).resolve())
    return _digest(
        {
            "modules": [
                {
                    "path": path.name,
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
                for path in sorted(paths)
            ]
        }
    )


def _campaign_document(params: CampaignParams) -> dict[str, object]:
    document: dict[str, object] = {}
    for item in fields(CampaignParams):
        value = getattr(params, item.name)
        if type(value) is Decimal:
            document[item.name] = str(value)
        elif type(value) is tuple:
            document[item.name] = list(value)
        else:
            document[item.name] = value
    return document


def _vector(values: Mapping[str, object]) -> AdaptiveVector:
    return AdaptiveVector(
        entries=tuple(AdaptiveParameter(name=name, value=values[name]) for name in sorted(values))
    )


def _recovery_probability(recovery_count: int, eligible_count: int) -> Decimal:
    compatible = int(Decimal(eligible_count) * Decimal("0.25"))
    if Decimal(eligible_count) * Decimal("0.25") > compatible:
        compatible += 1
    if recovery_count == compatible:
        return Decimal("0.25")
    with localcontext() as context:
        context.prec = 50
        return (Decimal(recovery_count) / Decimal(eligible_count)).quantize(
            Decimal("0.000000000000000001"),
            rounding=ROUND_HALF_EVEN,
        )


def _recovery_values(
    eligible_count: int,
    *,
    require_unrecovered: bool,
) -> tuple[Decimal, ...]:
    maximum = eligible_count - 1 if require_unrecovered else eligible_count
    return tuple(
        sorted({_recovery_probability(count, eligible_count) for count in range(1, maximum + 1)})
    )


def _candidate_space(
    family: str,
    template: CampaignParams,
) -> tuple[tuple[tuple[str, DomainKind, tuple[object, ...]], ...], tuple[dict[str, object], ...]]:
    illicit_count = round(Decimal(template.payment_count) * template.target_illicit_rate)
    raw_domains: tuple[tuple[str, DomainKind, tuple[object, ...]], ...]
    combinations: list[dict[str, object]] = []
    if family == "card_testing_cnp":
        raw_domains = (
            (
                "device_reuse_rate",
                DomainKind.LINEAR,
                tuple(sorted({Decimal(0), template.device_reuse_rate, Decimal(1)})),
            ),
            (
                "merchant_concentration",
                DomainKind.LINEAR,
                tuple(sorted({Decimal(0), template.merchant_concentration, Decimal(1)})),
            ),
            (
                "retry_intensity",
                DomainKind.DISCRETE,
                tuple(range(1, min(10, illicit_count - 1) + 1)),
            ),
        )
        for values in itertools.product(*(item[2] for item in raw_domains)):
            combinations.append(
                {item[0]: value for item, value in zip(raw_domains, values, strict=True)}
            )
    elif family == "app_scam_mule":
        if template.mule_count != template.mule_layers + 1:
            raise BenchmarkConfigurationError("APP hidden template topology is not canonical")
        maximum_fanout = min(16, illicit_count - template.mule_layers - 2)
        fanouts = tuple(range(2, maximum_fanout + 1))
        delays = tuple(
            sorted(
                {
                    template.min_delay_seconds,
                    template.cash_out_delay_seconds,
                    template.max_delay_seconds,
                }
            )
        )
        cash_fractions = tuple(
            value
            for value in (Decimal("0.20"), Decimal("0.30"))
            if Decimal(0) < value <= Decimal(1)
        )
        recoveries = tuple(
            sorted(
                {
                    value
                    for fanout in fanouts
                    for value in _recovery_values(
                        fanout,
                        require_unrecovered=False,
                    )
                }
            )
        )
        raw_domains = (
            ("cash_out_delay_seconds", DomainKind.LOG, delays),
            ("cash_out_fraction", DomainKind.LINEAR, cash_fractions),
            (
                "cash_out_strategy",
                DomainKind.CATEGORICAL,
                ("burst", "delayed", "staged"),
            ),
            ("mule_fanout", DomainKind.DISCRETE, fanouts),
            ("recovery_probability", DomainKind.DISCRETE, recoveries),
        )
        for delay, fraction, strategy, fanout, recovery in itertools.product(
            *(item[2] for item in raw_domains)
        ):
            checked_delay = cast(int, delay)
            checked_strategy = cast(str, strategy)
            compatible_delay = (
                checked_delay == template.min_delay_seconds
                if checked_strategy == "burst"
                else checked_delay > template.min_delay_seconds
            )
            if not compatible_delay or recovery not in _recovery_values(
                cast(int, fanout), require_unrecovered=False
            ):
                continue
            combinations.append(
                {
                    "cash_out_delay_seconds": delay,
                    "cash_out_fraction": fraction,
                    "cash_out_strategy": strategy,
                    "mule_fanout": fanout,
                    "recovery_probability": recovery,
                }
            )
    elif family == "synthetic_merchant_refund":
        values = _recovery_values(illicit_count, require_unrecovered=True)
        raw_domains = (("recovery_probability", DomainKind.DISCRETE, values),)
        combinations = [{"recovery_probability": value} for value in values]
    else:
        mixes: list[Decimal] = []
        for attack_count in range(23, template.payment_count - 1):
            with localcontext() as context:
                context.prec = 28
                mix = Decimal(attack_count) / Decimal(template.payment_count)
            if abs(mix - template.target_illicit_rate) <= template.class_rate_tolerance:
                mixes.append(mix)
        mutation_values: tuple[object, ...] = (
            template.agentic_mutations,
            *(tuple([mutation]) for mutation in template.agentic_mutations[1:]),
        )
        raw_domains = (
            ("agentic_attack_mix", DomainKind.DISCRETE, tuple(mixes)),
            ("agentic_mutations", DomainKind.CATEGORICAL, mutation_values),
        )
        for mix_value, mutations in itertools.product(*(item[2] for item in raw_domains)):
            checked_mix = cast(Decimal, mix_value)
            additional = int(checked_mix * Decimal(template.payment_count)) - 23
            if additional == 0 and mutations != template.agentic_mutations:
                continue
            if additional > 0 and mutations != template.agentic_mutations:
                mutation_tuple = cast(tuple[str, ...], mutations)
                if len(mutation_tuple) > additional:
                    continue
                default = tuple(
                    template.agentic_mutations[index % len(template.agentic_mutations)]
                    for index in range(additional)
                )
                actual = tuple(
                    mutation_tuple[index % len(mutation_tuple)] for index in range(additional)
                )
                if actual == default:
                    continue
            combinations.append({"agentic_attack_mix": checked_mix, "agentic_mutations": mutations})
    return raw_domains, tuple(combinations)


def _compose(template: CampaignParams, vector: AdaptiveVector) -> CampaignParams:
    values = {item.name: getattr(template, item.name) for item in fields(CampaignParams)}
    values.update({entry.name: entry.value for entry in vector.entries})
    return CampaignParams.from_mapping(values)


def _public_bounds(
    family: str,
    population: Population,
    template: CampaignParams,
    *,
    generator_seed: int,
) -> ParameterBounds:
    raw_domains, combinations = _candidate_space(family, template)
    feasible: list[dict[str, object]] = []
    for updates in combinations:
        full = _vector(updates)
        params = _compose(template, full)
        try:
            _CampaignEvaluator(seed=generator_seed).generate(family, population, params)
        except (GenerationConstraintError, ArithmeticError, TypeError, ValueError):
            continue
        feasible.append(updates)
    if not feasible:
        raise BenchmarkConfigurationError("active Task 5 configuration has no feasible vector")
    default_values = {name: getattr(template, name) for name, _kind, _values in raw_domains}
    if not any(
        all(values[name] == default_values[name] for name in default_values) for values in feasible
    ):
        raise BenchmarkConfigurationError("hidden Task 5 default is not feasible")
    active = tuple(
        name
        for name, _kind, _values in raw_domains
        if len({_digest(_tagged_for_benchmark(values[name])) for values in feasible}) > 1
    )
    projected = (_vector({name: values[name] for name in active}) for values in feasible)
    by_fingerprint = {vector.fingerprint: vector for vector in projected}
    vectors = tuple(by_fingerprint[key] for key in sorted(by_fingerprint))
    defaults = _vector({name: default_values[name] for name in active})
    domains: list[ParameterDomain] = []
    for name, kind, raw_values in raw_domains:
        if name not in active:
            continue
        represented = tuple(vector.get(name) for vector in vectors)
        values = tuple(
            value
            for value in raw_values
            if any(
                value == candidate and type(value) is type(candidate) for candidate in represented
            )
        )
        domains.append(ParameterDomain(name=name, kind=kind, values=values))
    return ParameterBounds(
        family=family,
        defaults=defaults,
        domains=tuple(domains),
        feasible_vectors=vectors,
    )


def _tagged_for_benchmark(value: object) -> object:
    if type(value) is Decimal:
        return {"decimal": str(value)}
    if type(value) is tuple:
        return {"tuple": list(value)}
    return {type(value).__name__: value}


@dataclass(frozen=True, slots=True)
class DefenderRule:
    """One frozen rule over an evaluator-derived observable feature."""

    family: str
    feature: str
    threshold: Decimal
    action: Action
    reason_family: str

    def __post_init__(self) -> None:
        if type(self.family) is not str or self.family not in _MOTIFS:
            raise BenchmarkConfigurationError("defender rule family is unsupported")
        if type(self.feature) is not str or not self.feature:
            raise TypeError("defender feature must be an exact non-empty string")
        if type(self.threshold) is not Decimal or not self.threshold.is_finite():
            raise TypeError("defender threshold must be an exact finite Decimal")
        if type(self.action) is not Action or self.action is Action.APPROVE:
            raise TypeError("defender rule action must be exact challenge or decline")
        if (
            type(self.reason_family) is not str
            or self.reason_family not in PUBLIC_REASON_FAMILIES
            or self.reason_family == "approved"
        ):
            raise ValueError("defender rule needs a public non-approval reason")

    def document(self) -> dict[str, object]:
        return {
            "family": self.family,
            "feature": self.feature,
            "threshold": str(self.threshold),
            "action": self.action.value,
            "reason_family": self.reason_family,
        }


@dataclass(frozen=True, slots=True)
class DefenderRuleSet:
    """Order-independent frozen defender declared before any policy trials."""

    version: str
    rules: tuple[DefenderRule, ...]

    def __post_init__(self) -> None:
        if type(self.version) is not str or not self.version:
            raise TypeError("defender version must be an exact non-empty string")
        if type(self.rules) is not tuple or any(
            type(rule) is not DefenderRule for rule in self.rules
        ):
            raise TypeError("defender rules must be an exact tuple")
        ordered = tuple(
            sorted(
                self.rules,
                key=lambda rule: (
                    rule.family,
                    rule.feature,
                    rule.threshold,
                    rule.action.value,
                    rule.reason_family,
                ),
            )
        )
        if len({_digest(rule.document()) for rule in ordered}) != len(ordered):
            raise ValueError("defender rules must be unique")
        object.__setattr__(self, "rules", ordered)

    @property
    def defender_digest(self) -> str:
        return _digest({"version": self.version, "rules": [rule.document() for rule in self.rules]})

    def decide(
        self,
        family: str,
        features: Mapping[str, Decimal],
    ) -> tuple[Action, str]:
        triggered = tuple(
            rule
            for rule in self.rules
            if rule.family == family
            and rule.feature in features
            and features[rule.feature] >= rule.threshold
        )
        if not triggered:
            return Action.APPROVE, "approved"
        winner = min(
            triggered,
            key=lambda rule: (-_SEVERITY[rule.action], rule.reason_family, rule.feature),
        )
        return winner.action, winner.reason_family


def default_defender_rules() -> DefenderRuleSet:
    """Return the preregisterable Task 6 artifact defender without policy knowledge."""
    return DefenderRuleSet(
        version="artifact-defender-1.0.0",
        rules=(
            DefenderRule(
                family="app_scam_mule",
                feature="max_distinct_payees_per_actor",
                threshold=Decimal(4),
                action=Action.DECLINE,
                reason_family="entity",
            ),
            DefenderRule(
                family="app_scam_mule",
                feature="max_distinct_payees_per_actor",
                threshold=Decimal(3),
                action=Action.CHALLENGE,
                reason_family="entity",
            ),
            DefenderRule(
                family="card_testing_cnp",
                feature="declined_authorizations",
                threshold=Decimal(4),
                action=Action.DECLINE,
                reason_family="velocity",
            ),
            DefenderRule(
                family="card_testing_cnp",
                feature="declined_authorizations",
                threshold=Decimal(2),
                action=Action.CHALLENGE,
                reason_family="velocity",
            ),
        ),
    )


@dataclass(frozen=True, slots=True)
class BenchmarkObservation:
    """Evaluator-owned lossless audit trace for one candidate evaluation."""

    family: str
    candidate_id: str
    candidate_document_digest: str
    artifact_digest: str
    command_count: int
    command_type_counts: tuple[tuple[str, int], ...]
    event_digest: str
    event_count: int
    event_type_counts: tuple[tuple[str, int], ...]
    ledger_digest: str
    ledger_entry_count: int
    feature_values: tuple[tuple[str, Decimal], ...]
    fresh_replay_succeeded: bool
    ledger_conserved: bool
    matched_rule: DefenderRule | None
    decision_action: Action
    decision_reason_family: str
    role_bound_value_components: tuple[tuple[str, Decimal, Decimal, Decimal], ...]
    realized_settled_illicit_value: Decimal
    feedback_realized_value: Decimal | None

    def document(self) -> dict[str, object]:
        """Return the exact canonical public evidence record for this evaluation."""
        return {
            "family": self.family,
            "candidate_id": self.candidate_id,
            "candidate_document_digest": self.candidate_document_digest,
            "command_digest": self.artifact_digest,
            "command_count": self.command_count,
            "command_type_counts": [
                {"name": name, "count": count}
                for name, count in self.command_type_counts
            ],
            "event_digest": self.event_digest,
            "event_count": self.event_count,
            "event_type_counts": [
                {"event_type": event_type, "count": count}
                for event_type, count in self.event_type_counts
            ],
            "ledger_digest": self.ledger_digest,
            "ledger_entry_count": self.ledger_entry_count,
            "fresh_replay_succeeded": self.fresh_replay_succeeded,
            "ledger_conserved": self.ledger_conserved,
            "derived_feature_vector": [
                {"name": name, "value": str(value)}
                for name, value in self.feature_values
            ],
            "matched_defender_rule": (
                None if self.matched_rule is None else self.matched_rule.document()
            ),
            "decision": {
                "action": self.decision_action.value,
                "reason_family": self.decision_reason_family,
            },
            "role_bound_value_components": [
                {
                    "payment_id": payment_id,
                    "positive_value": str(positive),
                    "removed_value": str(removed),
                    "outstanding_value": str(outstanding),
                    "outstanding_minor_units": int(outstanding * 100),
                }
                for payment_id, positive, removed, outstanding in (
                    self.role_bound_value_components
                )
            ],
            "executed_role_bound_value": str(self.realized_settled_illicit_value),
            "feedback_realized_value": (
                None
                if self.feedback_realized_value is None
                else str(self.feedback_realized_value)
            ),
        }


def _ledger_digest(entries: tuple[LedgerEntry, ...]) -> str:
    document = [
        {
            "entry_id": entry.entry_id,
            "currency": entry.currency,
            "debit": {
                account: str(amount) for account, amount in sorted(entry.debit.items())
            },
            "credit": {
                account: str(amount) for account, amount in sorted(entry.credit.items())
            },
        }
        for entry in entries
    ]
    return _digest(document)


def _matched_rule(
    defender: DefenderRuleSet,
    family: str,
    features: Mapping[str, Decimal],
) -> DefenderRule | None:
    triggered = tuple(
        rule
        for rule in defender.rules
        if rule.family == family
        and rule.feature in features
        and features[rule.feature] >= rule.threshold
    )
    if not triggered:
        return None
    return min(
        triggered,
        key=lambda rule: (-_SEVERITY[rule.action], rule.reason_family, rule.feature),
    )


def _role_bound_value_components(
    events: tuple[PaymentEvent, ...],
    illicit_entity_ids: frozenset[str],
) -> tuple[tuple[str, Decimal, Decimal, Decimal], ...]:
    positive = {EventKind.SETTLEMENT, EventKind.TRANSFER_POSTED}
    negative = {
        EventKind.RECOVERY,
        EventKind.CHARGEBACK,
        EventKind.TRANSFER_RETURNED,
        EventKind.REFUND,
    }
    values: dict[str, tuple[Decimal, Decimal, Decimal]] = {}
    for event in events:
        if not {event.actor_id, event.counterparty_id} & illicit_entity_ids:
            continue
        raw_payment_id = event.rail_data.get("payment_id", event.trace_id)
        if type(raw_payment_id) is not str:
            continue
        positive_value, removed_value, outstanding = values.get(
            raw_payment_id,
            (Decimal(0), Decimal(0), Decimal(0)),
        )
        if event.event_type in positive:
            positive_value += event.amount
            outstanding += event.amount
        elif event.event_type in negative:
            applied = min(outstanding, event.amount)
            removed_value += applied
            outstanding -= applied
        values[raw_payment_id] = (
            _money(positive_value),
            _money(removed_value),
            _money(outstanding),
        )
    return tuple(
        (payment_id, positive_value, removed_value, outstanding)
        for payment_id, (positive_value, removed_value, outstanding) in sorted(values.items())
    )


def _fresh_replay(
    family: str,
    commands: tuple[Command, ...],
    evidence: CampaignEvidence,
    population: Population,
) -> tuple[tuple[PaymentEvent, ...], tuple[LedgerEntry, ...]]:
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
        fixture = evidence.agentic_fixture
        if type(fixture) is not AgenticFixture:
            raise BenchmarkConfigurationError("agentic replay requires an exact fixture")

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
    opening: dict[AccountReference, Decimal] = {
        cast(AccountReference, account): amount
        for account, amount in population.opening_balances.items()
    }
    engine = SimulationEngine(bundle, {rail: factory}, opening_balances=opening)
    for priority, (timestamp, command) in enumerate(zip(evidence.schedule, commands, strict=True)):
        engine.schedule(timestamp, priority, command)
    events = engine.run()
    engine.ledger.assert_conserved()
    return events, engine.ledger.entries


def _observable_features(
    commands: tuple[Command, ...],
    evidence: CampaignEvidence,
) -> Mapping[str, Decimal]:
    openings = tuple(command for command in commands if getattr(command, "name", "") in _OPENINGS)
    actor_payees: dict[str, set[str]] = {}
    payee_counts: dict[str, int] = {}
    for command in openings:
        payload = command.payload
        actor = cast(str, payload["actor_id"])
        payee = cast(str, payload["counterparty_id"])
        actor_payees.setdefault(actor, set()).add(payee)
        payee_counts[payee] = payee_counts.get(payee, 0) + 1
    opening_indices = tuple(
        index for index, command in enumerate(commands) if getattr(command, "name", "") in _OPENINGS
    )
    gaps = tuple(
        Decimal(int((evidence.schedule[right] - evidence.schedule[left]).total_seconds()))
        for left, right in zip(opening_indices, opening_indices[1:], strict=False)
    )
    maximum_payee_share = Decimal(0)
    if openings and payee_counts:
        with localcontext() as context:
            context.prec = 28
            maximum_payee_share = Decimal(max(payee_counts.values())) / Decimal(len(openings))
    return MappingProxyType(
        {
            "declined_authorizations": Decimal(
                sum(getattr(command, "name", "") == "card.decline" for command in openings)
            ),
            "max_distinct_payees_per_actor": Decimal(
                max((len(payees) for payees in actor_payees.values()), default=0)
            ),
            "maximum_payee_share": maximum_payee_share,
            "minimum_opening_gap_seconds": min(gaps, default=Decimal(0)),
            "opening_attempts": Decimal(len(openings)),
        }
    )


def role_bound_settled_value(
    events: tuple[PaymentEvent, ...],
    illicit_entity_ids: frozenset[str],
) -> Decimal:
    """Return outstanding illicit principal, subtracting each principal at most once."""
    if type(events) is not tuple or any(type(event) is not PaymentEvent for event in events):
        raise TypeError("events must be an exact tuple of PaymentEvent records")
    if type(illicit_entity_ids) is not frozenset or any(
        type(entity_id) is not str for entity_id in illicit_entity_ids
    ):
        raise TypeError("illicit_entity_ids must be an exact frozenset of strings")
    positive = {EventKind.SETTLEMENT, EventKind.TRANSFER_POSTED}
    negative = {
        EventKind.RECOVERY,
        EventKind.CHARGEBACK,
        EventKind.TRANSFER_RETURNED,
        EventKind.REFUND,
    }
    outstanding: dict[str, Decimal] = {}
    for event in events:
        if not {event.actor_id, event.counterparty_id} & illicit_entity_ids:
            continue
        raw_payment_id = event.rail_data.get("payment_id", event.trace_id)
        if type(raw_payment_id) is not str:
            raise ValueError("payment lifecycle identity must be an exact string")
        current = outstanding.get(raw_payment_id, Decimal(0))
        if event.event_type in positive:
            outstanding[raw_payment_id] = current + event.amount
        elif event.event_type in negative:
            outstanding[raw_payment_id] = max(Decimal(0), current - event.amount)
    return _money(sum(outstanding.values(), Decimal(0)))


class CampaignBenchmark:
    """Policy-independent evaluator bound to one hidden Task 5 background."""

    __slots__ = (
        "_defender",
        "_family",
        "_generator_seed",
        "_population",
        "_template",
        "_evaluation_traces",
        "evaluation_contract",
        "public_bounds",
    )

    def __init__(
        self,
        *,
        family: str,
        population: Population,
        hidden_template: CampaignParams,
        defender: DefenderRuleSet,
        disclosure_profile: DisclosureProfile,
        generator_seed: int,
    ) -> None:
        if type(family) is not str or family not in _MOTIFS:
            raise BenchmarkConfigurationError("benchmark family is unsupported")
        if type(population) is not Population:
            raise TypeError("population must be an exact Population")
        if type(hidden_template) is not CampaignParams:
            raise TypeError("hidden_template must be an exact CampaignParams")
        if hidden_template.expected_motif != _MOTIFS[family]:
            raise BenchmarkConfigurationError("hidden template motif does not match family")
        if type(defender) is not DefenderRuleSet:
            raise TypeError("defender must be an exact DefenderRuleSet")
        if type(disclosure_profile) is not DisclosureProfile:
            raise TypeError("disclosure_profile must be exact")
        disclosure_profile.assert_pristine()
        if type(generator_seed) is not int or not 0 <= generator_seed < 2**63:
            raise TypeError("generator_seed must be an exact bounded integer")
        public_bounds = _public_bounds(
            family,
            population,
            hidden_template,
            generator_seed=generator_seed,
        )
        population_digest = hashlib.sha256(population.canonical_bytes()).hexdigest()
        background_digest = hashlib.sha256(campaign_bytes(population.benign_commands)).hexdigest()
        template_digest = _digest(_campaign_document(hidden_template))
        evaluator_digest = _digest(
            {
                "version": _EVALUATOR_VERSION,
                "feature_names": list(_OBSERVABLE_FEATURE_NAMES),
                "feature_semantics": "command-schedule-opening-observations-v1",
                "replay_semantics": "fresh-production-rail-and-conserved-ledger-v1",
                "value_semantics": "role-bound-net-settlement-v1",
            }
        )
        self._family = family
        self._population = population
        self._template = hidden_template
        self._defender = defender
        self._generator_seed = generator_seed
        self._evaluation_traces: list[BenchmarkObservation] = []
        self.public_bounds = public_bounds
        self.evaluation_contract = EvaluationContract(
            family=family,
            bounds_digest=public_bounds.bounds_digest,
            hidden_template_digest=template_digest,
            background_digest=background_digest,
            population_digest=population_digest,
            evaluator_digest=evaluator_digest,
            defender_digest=defender.defender_digest,
            disclosure_profile=disclosure_profile,
        )

    def compose(self, candidate: AttackCandidate) -> CampaignParams:
        checked = reconstruct_candidate(candidate)
        vector = self.public_bounds.validate_vector(checked.params)
        params = _compose(self._template, vector)
        if params.expected_motif != _MOTIFS[self._family]:
            raise BenchmarkConfigurationError("composed campaign motif changed")
        return params

    def evaluate_with_observation(
        self,
        candidate: AttackCandidate,
    ) -> tuple[Feedback, BenchmarkObservation]:
        params = self.compose(candidate)
        checked_candidate = reconstruct_candidate(candidate)
        candidate_document = checked_candidate.model_dump(mode="json", round_trip=True)
        try:
            commands, evidence = _CampaignEvaluator(seed=self._generator_seed).generate(
                self._family,
                self._population,
                params,
            )
            events, _entries = _fresh_replay(
                self._family,
                commands,
                evidence,
                self._population,
            )
        except (GenerationConstraintError, ArithmeticError, RuntimeError, TypeError, ValueError):
            feedback = Feedback(
                action=Action.DECLINE,
                reason_family="invalid_candidate",
                realized_value=None,
            )
            observation = BenchmarkObservation(
                family=self._family,
                candidate_id=checked_candidate.candidate_id,
                candidate_document_digest=_digest(candidate_document),
                artifact_digest="0" * 64,
                command_count=0,
                command_type_counts=(),
                event_digest="0" * 64,
                event_count=0,
                event_type_counts=(),
                ledger_digest="0" * 64,
                ledger_entry_count=0,
                feature_values=(),
                fresh_replay_succeeded=False,
                ledger_conserved=False,
                matched_rule=None,
                decision_action=feedback.action,
                decision_reason_family=feedback.reason_family,
                role_bound_value_components=(),
                realized_settled_illicit_value=Decimal("0.00"),
                feedback_realized_value=feedback.realized_value,
            )
            return feedback, observation
        features = _observable_features(commands, evidence)
        action, reason = self._defender.decide(self._family, features)
        matched_rule = _matched_rule(self._defender, self._family, features)
        illicit_entities = frozenset(
            entity.entity_id for entity in self._population.entities if entity.illicit
        )
        executed_value = role_bound_settled_value(events, illicit_entities)
        value_components = _role_bound_value_components(events, illicit_entities)
        realized = executed_value if action is Action.APPROVE else Decimal("0.00")
        feedback = Feedback(
            action=action,
            reason_family="approved" if action is Action.APPROVE else reason,
            realized_value=realized,
        )
        event_document = [event.model_dump(mode="json", round_trip=True) for event in events]
        observation = BenchmarkObservation(
            family=self._family,
            candidate_id=checked_candidate.candidate_id,
            candidate_document_digest=_digest(candidate_document),
            artifact_digest=hashlib.sha256(campaign_bytes(commands)).hexdigest(),
            command_count=len(commands),
            command_type_counts=tuple(
                sorted(Counter(command.name for command in commands).items())
            ),
            event_digest=_digest(event_document),
            event_count=len(events),
            event_type_counts=tuple(
                sorted(Counter(event.event_type.value for event in events).items())
            ),
            ledger_digest=_ledger_digest(_entries),
            ledger_entry_count=len(_entries),
            feature_values=tuple(sorted(features.items())),
            fresh_replay_succeeded=True,
            ledger_conserved=True,
            matched_rule=matched_rule,
            decision_action=feedback.action,
            decision_reason_family=feedback.reason_family,
            role_bound_value_components=value_components,
            realized_settled_illicit_value=executed_value,
            feedback_realized_value=feedback.realized_value,
        )
        return feedback, observation

    def evaluate(self, candidate: AttackCandidate) -> Feedback:
        feedback, observation = self.evaluate_with_observation(candidate)
        self._evaluation_traces.append(observation)
        return feedback

    def take_evaluation_traces(self) -> tuple[BenchmarkObservation, ...]:
        """Drain evaluator-owned traces without exposing them to policy code."""
        traces = tuple(self._evaluation_traces)
        self._evaluation_traces.clear()
        return traces

    def issue_evaluator_capability(
        self,
        authority: SearchAuthority,
    ) -> EvaluatorCapability:
        if type(authority) is not SearchAuthority:
            raise TypeError("authority must be an exact SearchAuthority")
        return authority.register_evaluator(
            owner=self,
            bounds=self.public_bounds,
            evaluation_contract=self.evaluation_contract,
            evaluate=self.evaluate,
            dependency_digest=_evaluator_dependency_digest(),
        )


__all__ = [
    "BenchmarkConfigurationError",
    "BenchmarkObservation",
    "CampaignBenchmark",
    "DefenderRule",
    "DefenderRuleSet",
    "default_defender_rules",
    "role_bound_settled_value",
]
