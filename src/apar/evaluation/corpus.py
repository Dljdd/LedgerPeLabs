"""Authenticated assembly of the defender-visible development corpus."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from datetime import timedelta
from decimal import Decimal
from typing import Literal, cast

from pydantic import ValidationError

from apar.contracts.events import EventKind, PaymentEvent, Rail
from apar.defense.contracts import scrub_event
from apar.evaluation.contracts import (
    CorpusManifest,
    CorpusProfile,
    EvaluationTruthRow,
    FrozenCorpus,
)
from apar.runs import RunManifest, RunRunner
from apar.runs.wire import WireContractError, strict_json_loads
from apar.storage.artifacts import ArtifactStore

_PUBLIC_ARTIFACTS = ("events", "population", "summary")
_SETTLEMENT_EVENTS = frozenset({EventKind.SETTLEMENT, EventKind.TRANSFER_POSTED})
_REVERSING_EVENTS = frozenset(
    {
        EventKind.REVERSAL,
        EventKind.TRANSFER_RETURNED,
        EventKind.REFUND,
        EventKind.CHARGEBACK,
        EventKind.RECOVERY,
    }
)
_FAMILY_RAILS = {
    "agentic_intent_abuse": Rail.AGENTIC,
    "app_scam_mule": Rail.A2A,
    "card_testing_cnp": Rail.CARD,
    "synthetic_merchant_refund": Rail.CARD,
}


class CorpusVerificationError(ValueError):
    """The authenticated corpus boundary failed closed."""


def assemble_verified_corpus(
    manifests: Sequence[RunManifest],
    runner: RunRunner,
    store: ArtifactStore,
    profile: CorpusProfile,
) -> FrozenCorpus:
    """Reverify runs, then derive observations and truth from public artifacts only."""
    if not isinstance(store, ArtifactStore):
        raise TypeError("store must be an ArtifactStore")
    if type(runner) is not RunRunner:
        raise TypeError("runner must be an exact RunRunner")
    if type(profile) is not CorpusProfile:
        raise TypeError("profile must be an exact CorpusProfile")
    if not manifests:
        raise CorpusVerificationError("corpus requires at least one authenticated run")

    parsed: list[tuple[RunManifest, tuple[PaymentEvent, ...], dict[str, bool], str]] = []
    seen_run_ids: set[str] = set()
    for manifest in manifests:
        try:
            authenticated = type(manifest) is RunManifest and runner.verify_run(manifest)
        except (AttributeError, TypeError, ValueError):
            authenticated = False
        if not authenticated:
            raise CorpusVerificationError("corpus requires an authenticated run")
        if manifest.run_id in seen_run_ids:
            raise CorpusVerificationError("duplicate run ID in corpus")
        seen_run_ids.add(manifest.run_id)
        parsed.append((manifest, *_parse_public_artifacts(manifest, store, profile)))

    events = tuple(event for _, run_events, _, _ in parsed for event in run_events)
    _validate_event_identities(events)

    observations = tuple(scrub_event(event) for event in events)
    observed_by_event = {observation.event_id: observation for observation in observations}
    groups: dict[str, list[PaymentEvent]] = defaultdict(list)
    for event in events:
        groups[_payment_id(event)].append(event)

    truth: list[EvaluationTruthRow] = []
    for payment_id, lifecycle in sorted(groups.items()):
        openings = [
            event for event in lifecycle if observed_by_event[event.event_id].is_decision_point
        ]
        if len(openings) != 1:
            raise CorpusVerificationError(
                f"payment {payment_id} lifecycle must resolve to exactly one opening"
            )
        opening = openings[0]
        manifest, population_truth, family = next(
            (run, population, run_family)
            for run, run_events, population, run_family in parsed
            if opening in run_events
        )
        del manifest
        truth.append(
            _truth_row(
                opening,
                lifecycle,
                population_truth,
                family,
                profile.label_delay_days,
            )
        )

    ordered_observations = tuple(sorted(observations, key=lambda observation: observation.event_id))
    ordered_truth = tuple(sorted(truth, key=lambda row: row.event_id))
    return FrozenCorpus(
        observations=ordered_observations,
        truth=ordered_truth,
        manifest=CorpusManifest(
            profile_id=profile.profile_id,
            run_ids=tuple(manifest.run_id for manifest, _, _, _ in parsed),
            run_lineage_digests=tuple(manifest.lineage_digest for manifest, _, _, _ in parsed),
            observation_count=len(ordered_observations),
            truth_count=len(ordered_truth),
        ),
    )


def _parse_public_artifacts(
    manifest: RunManifest, store: ArtifactStore, profile: CorpusProfile
) -> tuple[tuple[PaymentEvent, ...], dict[str, bool], str]:
    try:
        references = {name: manifest.artifacts[name] for name in _PUBLIC_ARTIFACTS}
    except KeyError as error:
        raise CorpusVerificationError(
            "authenticated run is missing a public corpus artifact"
        ) from error
    try:
        event_document = strict_json_loads(store.read(references["events"]))
        population_document = strict_json_loads(store.read(references["population"]))
        summary_document = strict_json_loads(store.read(references["summary"]))
    except (ValueError, WireContractError) as error:
        raise CorpusVerificationError("public corpus artifact is not canonical") from error
    if type(event_document) is not list:
        raise CorpusVerificationError("events artifact must be a list")
    try:
        events = tuple(PaymentEvent.model_validate(item) for item in event_document)
    except (ValidationError, TypeError, ValueError) as error:
        raise CorpusVerificationError(
            "events artifact violates the payment-event contract"
        ) from error
    if type(population_document) is not dict or type(summary_document) is not dict:
        raise CorpusVerificationError("population and summary artifacts must be objects")
    family = summary_document.get("family")
    if family not in profile.families or type(family) is not str:
        raise CorpusVerificationError("run family is not declared by the corpus profile")
    family_literal = cast(
        Literal[
            "agentic_intent_abuse",
            "app_scam_mule",
            "card_testing_cnp",
            "synthetic_merchant_refund",
        ],
        family,
    )
    expected_rail = _FAMILY_RAILS[family_literal]
    if any(event.rail is not expected_rail for event in events):
        raise CorpusVerificationError("run declared rail does not match its family")
    entities = population_document.get("entities")
    if type(entities) is not list:
        raise CorpusVerificationError("population artifact has no entity truth")
    population_truth: dict[str, bool] = {}
    for entity in entities:
        if type(entity) is not dict:
            raise CorpusVerificationError("population entity is invalid")
        entity_id = entity.get("entity_id")
        illicit = entity.get("illicit")
        if type(entity_id) is not str or type(illicit) is not bool:
            raise CorpusVerificationError("population entity truth is invalid")
        if entity_id in population_truth:
            raise CorpusVerificationError("population contains duplicate entity IDs")
        population_truth[entity_id] = illicit
    missing_entity_truth = sorted(
        {
            entity_id
            for event in events
            for entity_id in (event.actor_id, event.counterparty_id)
            if entity_id not in population_truth
        }
    )
    if missing_entity_truth:
        raise CorpusVerificationError("population truth is missing a referenced entity")
    return events, population_truth, family


def _validate_event_identities(events: tuple[PaymentEvent, ...]) -> None:
    event_ids: set[str] = set()
    payment_campaigns: dict[str, str] = {}
    opening_by_payment: dict[str, str] = {}
    for event in events:
        if event.event_id in event_ids:
            raise CorpusVerificationError("duplicate event ID in corpus")
        event_ids.add(event.event_id)
        if event.lineage.get("synthetic") is not True:
            raise CorpusVerificationError("corpus accepts synthetic events only")
        payment_id = _payment_id(event)
        campaign = payment_campaigns.setdefault(payment_id, event.campaign_id)
        if campaign != event.campaign_id:
            raise CorpusVerificationError("payment ID reused across campaigns")
        if _opening(event):
            previous = opening_by_payment.setdefault(payment_id, event.event_id)
            if previous != event.event_id:
                raise CorpusVerificationError("duplicate payment opening")


def _payment_id(event: PaymentEvent) -> str:
    payment_id = event.rail_data.get("payment_id")
    if type(payment_id) is not str or not payment_id:
        raise CorpusVerificationError("event has no payment ID")
    return payment_id


def _opening(event: PaymentEvent) -> bool:
    return scrub_event(event).is_decision_point


def _truth_row(
    opening: PaymentEvent,
    lifecycle: list[PaymentEvent],
    population_truth: dict[str, bool],
    family: str,
    label_delay_days: int,
) -> EvaluationTruthRow:
    observed = scrub_event(opening)
    ordered = tuple(sorted(lifecycle, key=lambda event: (event.event_time, event.event_id)))
    settlements = [event for event in ordered if event.event_type in _SETTLEMENT_EVENTS]
    net_settled = sum((event.amount for event in settlements), Decimal("0")) - sum(
        (event.amount for event in ordered if event.event_type in _REVERSING_EVENTS), Decimal("0")
    )
    first_settlement_at = settlements[0].event_time if settlements else None
    delayed = (opening.decision_at or opening.available_at) + timedelta(days=label_delay_days)
    label_mature_at = max(delayed, first_settlement_at) if first_settlement_at else delayed
    if observed.integrity_status == "fail":
        is_fraud = True
        label_source: Literal["population_truth", "integrity_truth", "hidden_truth"] = (
            "integrity_truth"
        )
    else:
        is_fraud = population_truth[opening.actor_id] or population_truth[opening.counterparty_id]
        label_source = "population_truth"
    return EvaluationTruthRow(
        event_id=opening.event_id,
        payment_id=observed.payment_id,
        campaign_id=opening.campaign_id,
        family=cast(
            Literal[
                "agentic_intent_abuse",
                "app_scam_mule",
                "card_testing_cnp",
                "synthetic_merchant_refund",
            ],
            family,
        ),
        viewpoint="development",
        is_fraud=is_fraud,
        label_source=label_source,
        label_mature_at=label_mature_at,
        first_settlement_at=first_settlement_at,
        net_settled_value=net_settled,
        lifecycle_event_ids=tuple(event.event_id for event in ordered),
    )
