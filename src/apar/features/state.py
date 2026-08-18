"""Strict knowledge-time feature state and canonical checkpoints."""

from __future__ import annotations

import hashlib
import json
import math
import statistics
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal
from itertools import groupby
from operator import attrgetter
from typing import Literal, cast

from pydantic import field_validator

from apar.contracts._validation import ExternalContract, validate_utc_timestamp
from apar.contracts.events import EventKind, Rail
from apar.defense.contracts import ObservedEvent
from apar.features.catalog import FeatureCatalog, FeatureDefinition, audit_feature_catalog

_CHECKPOINT_SCHEMA_VERSION = "1.0.0"
_SENTINEL = -1.0
_DAY = timedelta(hours=24)
_HOUR = timedelta(hours=1)
_EXPECTED_OPTIONAL_REFS = 7
_SUSPICIOUS_EVENTS = frozenset(
    {
        EventKind.AUTHORIZATION_DECLINED,
        EventKind.AUTHENTICATION_CHALLENGE,
        EventKind.TRANSFER_REJECTED,
        EventKind.TRANSFER_RETURNED,
        EventKind.FUNDS_FROZEN,
        EventKind.REFUND,
        EventKind.DISPUTE_OPENED,
        EventKind.CHARGEBACK,
    }
)


class FeatureStateError(ValueError):
    """Feature state cannot preserve its causal or checkpoint contract."""


class FeatureVector(ExternalContract):
    """One ordered model row with historical-source provenance."""

    schema_version: Literal["1.0.0"] = "1.0.0"
    event_id: str
    decision_at: datetime
    source_event_ids: tuple[str, ...]
    max_source_available_at: datetime | None
    catalog_digest: str
    values: dict[str, float]

    @field_validator("decision_at", "max_source_available_at")
    @classmethod
    def timestamps_are_utc(cls, value: datetime | None) -> datetime | None:
        return None if value is None else validate_utc_timestamp(value)

    @property
    def decision_time(self) -> datetime:
        """Compatibility alias for consumers that name the decision timestamp generically."""
        return self.decision_at


@dataclass(frozen=True, slots=True)
class _GraphEdge:
    event_id: str
    payment_id: str
    actor_id: str
    counterparty_id: str
    event_time: datetime
    available_at: datetime
    suspicious: bool


def feature_catalog_digest(catalog: FeatureCatalog) -> str:
    """Hash the canonical, complete catalog contract."""
    audit_feature_catalog(catalog)
    document = catalog.model_dump(mode="json")
    return hashlib.sha256(_canonical_json(document)).hexdigest()


class CausalFeatureState:
    """Incremental state that admits historical observations strictly before decisions."""

    def __init__(self, catalog: FeatureCatalog) -> None:
        audit_feature_catalog(catalog)
        self.catalog = catalog
        self.catalog_digest = feature_catalog_digest(catalog)
        self._known: dict[str, ObservedEvent] = {}
        self._admitted_ids: set[str] = set()
        self._emitted_decision_ids: set[str] = set()
        self._late_event_ids: set[str] = set()
        self._late_event_watermarks: dict[str, datetime] = {}
        self._opening_payment_ids: set[str] = set()
        self._watermark: datetime | None = None
        self._actor_history: dict[str, list[str]] = defaultdict(list)
        self._counterparty_history: dict[str, list[str]] = defaultdict(list)
        self._pair_history: dict[tuple[str, str], list[str]] = defaultdict(list)
        self._outcome_history: dict[EventKind, list[str]] = defaultdict(list)
        self._actor_amount_history: dict[str, list[str]] = defaultdict(list)
        self._counterparty_amount_history: dict[str, list[str]] = defaultdict(list)
        self._edges: list[_GraphEdge] = []

    def process(self, events: Sequence[ObservedEvent]) -> tuple[FeatureVector, ...]:
        """Compute unseen decision rows deterministically without double-admitting events."""
        self._remember(events)
        decisions = sorted(
            (
                event
                for event in self._known.values()
                if event.is_decision_point and event.event_id not in self._emitted_decision_ids
            ),
            key=_decision_sort_key,
        )
        output: list[FeatureVector] = []
        for decision_at, grouped in groupby(decisions, key=attrgetter("decision_at")):
            if decision_at is None:
                raise FeatureStateError("decision-point event must declare decision_at")
            if self._watermark is not None and decision_at <= self._watermark:
                raise FeatureStateError("decision belongs to a closed decision timestamp")
            batch = tuple(grouped)
            batch_ids = {event.event_id for event in batch}
            self._admit_sources_strictly_before(decision_at, excluding=batch_ids)
            output.extend(self._compute(event, decision_at) for event in batch)
            self._emitted_decision_ids.update(batch_ids)
            self._watermark = decision_at
        return tuple(output)

    def checkpoint(self) -> bytes:
        """Return a canonical JSON checkpoint protected by a self-digest."""
        histories, adjacency = self._state_views()
        document: dict[str, object] = {
            "schema_version": _CHECKPOINT_SCHEMA_VERSION,
            "catalog_digest": self.catalog_digest,
            "watermark": _timestamp(self._watermark),
            "known_events": [
                self._known[event_id].model_dump(mode="json") for event_id in sorted(self._known)
            ],
            "admitted_event_ids": sorted(self._admitted_ids),
            "emitted_decision_ids": sorted(self._emitted_decision_ids),
            "late_event_ids": sorted(self._late_event_ids),
            "late_event_watermarks": {
                event_id: _timestamp(watermark)
                for event_id, watermark in sorted(self._late_event_watermarks.items())
            },
            "opening_payment_ids": sorted(self._opening_payment_ids),
            "histories": histories,
            "adjacency": adjacency,
        }
        document["self_digest"] = hashlib.sha256(_canonical_json(document)).hexdigest()
        return _canonical_json(document)

    @classmethod
    def restore(cls, payload: bytes, catalog: FeatureCatalog) -> CausalFeatureState:
        """Restore a verified canonical JSON checkpoint for exactly this catalog."""
        if type(payload) is not bytes:
            raise FeatureStateError("checkpoint payload must be bytes")
        try:
            decoded = json.loads(payload.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise FeatureStateError("checkpoint payload is not valid JSON") from error
        if type(decoded) is not dict or _canonical_json(decoded) != payload:
            raise FeatureStateError("checkpoint payload is not canonical JSON")
        document = cast(dict[str, object], decoded)
        supplied_digest = document.get("self_digest")
        unsigned = {key: value for key, value in document.items() if key != "self_digest"}
        expected_digest = hashlib.sha256(_canonical_json(unsigned)).hexdigest()
        if type(supplied_digest) is not str or supplied_digest != expected_digest:
            raise FeatureStateError("checkpoint self-digest mismatch")
        if document.get("schema_version") != _CHECKPOINT_SCHEMA_VERSION:
            raise FeatureStateError("checkpoint schema version mismatch")

        state = cls(catalog)
        if document.get("catalog_digest") != state.catalog_digest:
            raise FeatureStateError("checkpoint catalog digest mismatch")
        try:
            raw_events = _string_keyed_list(document, "known_events", require_dicts=True)
            known_events = tuple(ObservedEvent.model_validate(item) for item in raw_events)
            if len({event.event_id for event in known_events}) != len(known_events):
                raise ValueError("duplicate known event ID")
            if [event.event_id for event in known_events] != sorted(
                event.event_id for event in known_events
            ):
                raise ValueError("known events must be ordered by event ID")
            state._known = {event.event_id: event for event in known_events}
            admitted = set(_string_list(document, "admitted_event_ids"))
            emitted = set(_string_list(document, "emitted_decision_ids"))
            late = set(_string_list(document, "late_event_ids"))
            late_watermarks = _timestamp_mapping(document, "late_event_watermarks")
            if not admitted <= state._known.keys() or not emitted <= state._known.keys():
                raise ValueError("checkpoint references an unknown event")
            if not late <= state._known.keys():
                raise ValueError("checkpoint late state references an unknown event")
            if late != late_watermarks.keys():
                raise ValueError("checkpoint late state has inconsistent provenance")
            for event in sorted(
                (state._known[event_id] for event_id in admitted),
                key=lambda item: (item.available_at, item.event_id),
            ):
                state._admit_event(event)
            state._emitted_decision_ids = emitted
            state._late_event_ids = late
            state._late_event_watermarks = late_watermarks
            raw_watermark = document.get("watermark")
            if raw_watermark is not None and type(raw_watermark) is not str:
                raise ValueError("invalid watermark")
            state._watermark = (
                None
                if raw_watermark is None
                else validate_utc_timestamp(datetime.fromisoformat(raw_watermark))
            )
            emitted_events = tuple(state._known[event_id] for event_id in emitted)
            if any(
                not event.is_decision_point or event.decision_at is None
                for event in emitted_events
            ):
                raise ValueError("emitted state references a non-decision event")
            emitted_decision_times = {
                cast(datetime, event.decision_at) for event in emitted_events
            }
            expected_watermark = (
                max(emitted_decision_times) if emitted_decision_times else None
            )
            if state._watermark != expected_watermark:
                raise ValueError("watermark does not match emitted decisions")
            known_decision_ids = {
                event.event_id for event in known_events if event.is_decision_point
            }
            if emitted != known_decision_ids:
                raise ValueError("emitted state does not match known decisions")
            if state._watermark is None and admitted:
                raise ValueError("admitted state requires a decision watermark")
            if state._watermark is not None and any(
                state._known[event_id].available_at >= state._watermark
                for event_id in admitted
            ):
                raise ValueError("admitted state violates strict knowledge time")
            if state._watermark is None and late:
                raise ValueError("late state requires a decision watermark")
            if state._watermark is not None and any(
                state._known[event_id].available_at >= late_watermark
                or late_watermark > state._watermark
                or late_watermark not in emitted_decision_times
                for event_id, late_watermark in late_watermarks.items()
            ):
                raise ValueError("late state has impossible arrival timing")
        except (KeyError, TypeError, ValueError) as error:
            raise FeatureStateError(f"checkpoint state is invalid: {error}") from error

        histories, adjacency = state._state_views()
        if document.get("histories") != histories or document.get("adjacency") != adjacency:
            raise FeatureStateError("checkpoint histories or adjacency mismatch")
        opening_ids = document.get("opening_payment_ids")
        if opening_ids != sorted(state._opening_payment_ids):
            raise FeatureStateError("checkpoint opening-payment state mismatch")
        return state

    def _remember(self, events: Sequence[ObservedEvent]) -> None:
        staged: dict[str, ObservedEvent] = {}
        for event in events:
            if type(event) is not ObservedEvent:
                raise TypeError("events must contain exact ObservedEvent instances")
            previous = self._known.get(event.event_id, staged.get(event.event_id))
            if previous is not None:
                if previous != event:
                    raise FeatureStateError("duplicate event ID has conflicting observations")
                continue
            if event.is_decision_point and event.decision_at is None:
                raise FeatureStateError("decision-point event must declare decision_at")
            if (
                event.is_decision_point
                and self._watermark is not None
                and cast(datetime, event.decision_at) <= self._watermark
            ):
                raise FeatureStateError("decision belongs to a closed decision timestamp")
            staged[event.event_id] = event
        for event in staged.values():
            if self._watermark is not None and event.available_at < self._watermark:
                self._late_event_ids.add(event.event_id)
                self._late_event_watermarks[event.event_id] = self._watermark
            self._known[event.event_id] = event

    def _admit_sources_strictly_before(
        self, decision_at: datetime, *, excluding: set[str]
    ) -> None:
        candidates = sorted(
            (
                event
                for event in self._known.values()
                if event.event_id not in self._admitted_ids
                and event.event_id not in excluding
                and event.available_at < decision_at
            ),
            key=lambda event: (event.available_at, event.event_id),
        )
        for event in candidates:
            self._admit_event(event)

    def _admit_event(self, event: ObservedEvent) -> None:
        if event.event_id in self._admitted_ids:
            return
        self._admitted_ids.add(event.event_id)
        self._actor_history[event.actor_id].append(event.event_id)
        self._counterparty_history[event.counterparty_id].append(event.event_id)
        self._pair_history[(event.actor_id, event.counterparty_id)].append(event.event_id)
        self._outcome_history[event.event_type].append(event.event_id)
        if event.is_decision_point:
            self._actor_amount_history[event.actor_id].append(event.event_id)
            self._counterparty_amount_history[event.counterparty_id].append(event.event_id)
            if event.payment_id not in self._opening_payment_ids:
                self._opening_payment_ids.add(event.payment_id)
                self._edges.append(
                    _GraphEdge(
                        event_id=event.event_id,
                        payment_id=event.payment_id,
                        actor_id=event.actor_id,
                        counterparty_id=event.counterparty_id,
                        event_time=event.event_time,
                        available_at=event.available_at,
                        suspicious=(
                            event.event_type in _SUSPICIOUS_EVENTS
                            or event.integrity_status == "fail"
                        ),
                    )
                )

    def _compute(self, event: ObservedEvent, decision_at: datetime) -> FeatureVector:
        source_ids: set[str] = set()
        values: dict[str, float] = {}
        context = _FeatureContext(self, event, decision_at, source_ids)
        for definition in self.catalog.features:
            values[definition.name] = context.derive(definition)
        ordered_sources = tuple(sorted(source_ids))
        max_source = (
            max(self._known[event_id].available_at for event_id in ordered_sources)
            if ordered_sources
            else None
        )
        if max_source is not None and not max_source < decision_at:
            raise FeatureStateError("historical source violates strict knowledge time")
        return FeatureVector(
            event_id=event.event_id,
            decision_at=decision_at,
            source_event_ids=ordered_sources,
            max_source_available_at=max_source,
            catalog_digest=self.catalog_digest,
            values=values,
        )

    def _state_views(self) -> tuple[dict[str, object], dict[str, list[str]]]:
        histories: dict[str, object] = {
            "actor": _serialize_history(self._actor_history),
            "counterparty": _serialize_history(self._counterparty_history),
            "pair": {
                _pair_key(pair): list(event_ids)
                for pair, event_ids in sorted(self._pair_history.items())
            },
            "outcome": {
                outcome.value: list(event_ids)
                for outcome, event_ids in sorted(
                    self._outcome_history.items(), key=lambda item: item[0].value
                )
            },
            "actor_amount": _serialize_history(self._actor_amount_history),
            "counterparty_amount": _serialize_history(self._counterparty_amount_history),
        }
        adjacency_sets: dict[str, set[str]] = defaultdict(set)
        for edge in self._edges:
            adjacency_sets[edge.actor_id].add(edge.counterparty_id)
            adjacency_sets[edge.counterparty_id].add(edge.actor_id)
        adjacency = {
            node: sorted(neighbors) for node, neighbors in sorted(adjacency_sets.items())
        }
        return histories, adjacency


class _FeatureContext:
    """One decision's immutable view over already-admitted state."""

    def __init__(
        self,
        state: CausalFeatureState,
        event: ObservedEvent,
        decision_at: datetime,
        source_ids: set[str],
    ) -> None:
        self.state = state
        self.event = event
        self.decision_at = decision_at
        self.source_ids = source_ids

    def derive(self, definition: FeatureDefinition) -> float:
        method = getattr(self, f"_feature_{definition.name}", None)
        if method is None:
            raise FeatureStateError(f"feature has no executable definition: {definition.name}")
        value = method()
        if not math.isfinite(value):
            return _SENTINEL if definition.missing_behavior == "sentinel" else 0.0
        return float(value)

    def _events(
        self,
        event_ids: Iterable[str],
        *,
        window: timedelta | None = None,
        outcomes: frozenset[EventKind] | None = None,
    ) -> tuple[ObservedEvent, ...]:
        lower = self.event.event_time - window if window is not None else None
        selected = tuple(
            event
            for event in (self.state._known[event_id] for event_id in event_ids)
            if event.event_time <= self.event.event_time
            and (lower is None or event.event_time >= lower)
            and (outcomes is None or event.event_type in outcomes)
        )
        self.source_ids.update(event.event_id for event in selected)
        return selected

    def _actor(self, window: timedelta | None = None) -> tuple[ObservedEvent, ...]:
        return self._events(self.state._actor_history.get(self.event.actor_id, ()), window=window)

    def _counterparty(self, window: timedelta | None = None) -> tuple[ObservedEvent, ...]:
        return self._events(
            self.state._counterparty_history.get(self.event.counterparty_id, ()), window=window
        )

    def _pair(self, window: timedelta | None = None) -> tuple[ObservedEvent, ...]:
        return self._events(
            self.state._pair_history.get(
                (self.event.actor_id, self.event.counterparty_id), ()
            ),
            window=window,
        )

    def _amount_events(self, *, actor: bool) -> tuple[ObservedEvent, ...]:
        history = (
            self.state._actor_amount_history.get(self.event.actor_id, ())
            if actor
            else self.state._counterparty_amount_history.get(self.event.counterparty_id, ())
        )
        return self._events(history, window=_DAY)

    def _graph_edges(self, window: timedelta) -> tuple[_GraphEdge, ...]:
        lower = self.event.event_time - window
        selected = tuple(
            edge
            for edge in self.state._edges
            if lower <= edge.event_time <= self.event.event_time
        )
        self.source_ids.update(edge.event_id for edge in selected)
        return selected

    def _feature_txn_log_amount(self) -> float:
        return math.log1p(float(self.event.amount))

    def _feature_txn_rail_card(self) -> float:
        return float(self.event.rail is Rail.CARD)

    def _feature_txn_rail_a2a(self) -> float:
        return float(self.event.rail is Rail.A2A)

    def _feature_txn_rail_agentic(self) -> float:
        return float(self.event.rail is Rail.AGENTIC)

    def _feature_txn_hour_sin(self) -> float:
        return math.sin(2.0 * math.pi * _fractional_hour(self.event.event_time) / 24.0)

    def _feature_txn_hour_cos(self) -> float:
        return math.cos(2.0 * math.pi * _fractional_hour(self.event.event_time) / 24.0)

    def _feature_txn_integrity_pass(self) -> float:
        return float(self.event.rail is Rail.AGENTIC and self.event.integrity_status == "pass")

    def _feature_txn_optional_ref_count(self) -> float:
        return float(len(self.event.optional_refs))

    def _feature_actor_count_1m(self) -> float:
        return float(len(self._actor(timedelta(minutes=1))))

    def _feature_actor_count_10m(self) -> float:
        return float(len(self._actor(timedelta(minutes=10))))

    def _feature_actor_count_1h(self) -> float:
        return float(len(self._actor(_HOUR)))

    def _feature_actor_count_24h(self) -> float:
        return float(len(self._actor(_DAY)))

    def _feature_actor_amount_1h(self) -> float:
        events = self._events(
            self.state._actor_amount_history.get(self.event.actor_id, ()), window=_HOUR
        )
        return _sum_amounts(events)

    def _feature_actor_amount_24h(self) -> float:
        return _sum_amounts(self._amount_events(actor=True))

    def _feature_counterparty_count_1h(self) -> float:
        return float(len(self._counterparty(_HOUR)))

    def _feature_counterparty_count_24h(self) -> float:
        return float(len(self._counterparty(_DAY)))

    def _feature_counterparty_amount_24h(self) -> float:
        return _sum_amounts(self._amount_events(actor=False))

    def _actor_outcomes(self, kinds: frozenset[EventKind], window: timedelta) -> float:
        events = self._events(
            self.state._actor_history.get(self.event.actor_id, ()),
            window=window,
            outcomes=kinds,
        )
        return float(len(events))

    def _feature_actor_prior_decline_1h(self) -> float:
        return self._actor_outcomes(frozenset({EventKind.AUTHORIZATION_DECLINED}), _HOUR)

    def _feature_actor_prior_challenge_1h(self) -> float:
        return self._actor_outcomes(frozenset({EventKind.AUTHENTICATION_CHALLENGE}), _HOUR)

    def _feature_actor_prior_return_24h(self) -> float:
        return self._actor_outcomes(frozenset({EventKind.TRANSFER_RETURNED}), _DAY)

    def _feature_counterparty_prior_refund_24h(self) -> float:
        events = self._events(
            self.state._counterparty_history.get(self.event.counterparty_id, ()),
            window=_DAY,
            outcomes=frozenset({EventKind.REFUND}),
        )
        return float(len(events))

    def _elapsed(self, events: tuple[ObservedEvent, ...], *, first: bool) -> float:
        if not events:
            return _SENTINEL
        times = [event.event_time for event in events]
        source_time = min(times) if first else max(times)
        return max(0.0, (self.event.event_time - source_time).total_seconds())

    def _feature_actor_seconds_since_first(self) -> float:
        return self._elapsed(self._actor(), first=True)

    def _feature_actor_seconds_since_last(self) -> float:
        return self._elapsed(self._actor(), first=False)

    def _feature_counterparty_seconds_since_first(self) -> float:
        return self._elapsed(self._counterparty(), first=True)

    def _feature_counterparty_seconds_since_last(self) -> float:
        return self._elapsed(self._counterparty(), first=False)

    def _feature_pair_seconds_since_first(self) -> float:
        return self._elapsed(self._pair(), first=True)

    def _feature_pair_seconds_since_last(self) -> float:
        return self._elapsed(self._pair(), first=False)

    def _feature_actor_distinct_counterparties_24h(self) -> float:
        return float(len({event.counterparty_id for event in self._actor(_DAY)}))

    def _feature_counterparty_distinct_actors_24h(self) -> float:
        return float(len({event.actor_id for event in self._counterparty(_DAY)}))

    def _zscore(self, history: tuple[ObservedEvent, ...]) -> float:
        if not history:
            return _SENTINEL
        amounts = [float(event.amount) for event in history]
        deviation = statistics.pstdev(amounts)
        if deviation == 0.0:
            return 0.0
        return (float(self.event.amount) - statistics.fmean(amounts)) / deviation

    def _feature_actor_amount_zscore_24h(self) -> float:
        return self._zscore(self._amount_events(actor=True))

    def _feature_counterparty_amount_zscore_24h(self) -> float:
        return self._zscore(self._amount_events(actor=False))

    def _feature_pair_prior_count(self) -> float:
        return float(len(self._pair()))

    def _feature_graph_actor_fanout(self) -> float:
        return float(
            len(
                {
                    edge.counterparty_id
                    for edge in self._graph_edges(_DAY)
                    if edge.actor_id == self.event.actor_id
                }
            )
        )

    def _feature_graph_counterparty_fanin(self) -> float:
        return float(
            len(
                {
                    edge.actor_id
                    for edge in self._graph_edges(_DAY)
                    if edge.counterparty_id == self.event.counterparty_id
                }
            )
        )

    def _neighbors(self, edges: tuple[_GraphEdge, ...]) -> dict[str, set[str]]:
        neighbors: dict[str, set[str]] = defaultdict(set)
        for edge in edges:
            neighbors[edge.actor_id].add(edge.counterparty_id)
            neighbors[edge.counterparty_id].add(edge.actor_id)
        return neighbors

    def _feature_graph_shared_neighbor_count(self) -> float:
        neighbors = self._neighbors(self._graph_edges(_DAY))
        return float(
            len(
                neighbors.get(self.event.actor_id, set())
                & neighbors.get(self.event.counterparty_id, set())
            )
        )

    def _feature_graph_two_hop_reach(self) -> float:
        neighbors = self._neighbors(self._graph_edges(_DAY))
        direct = neighbors.get(self.event.actor_id, set())
        two_hop = set().union(*(neighbors.get(node, set()) for node in direct)) if direct else set()
        two_hop.difference_update(direct)
        two_hop.discard(self.event.actor_id)
        return float(len(two_hop))

    def _component(self, edges: tuple[_GraphEdge, ...]) -> tuple[set[str], set[tuple[str, str]]]:
        neighbors = self._neighbors(edges)
        seeds = {
            node
            for node in (self.event.actor_id, self.event.counterparty_id)
            if node in neighbors
        }
        seen: set[str] = set()
        frontier = list(seeds)
        while frontier:
            node = frontier.pop()
            if node in seen:
                continue
            seen.add(node)
            frontier.extend(neighbors[node] - seen)
        component_edges = {
            (min(edge.actor_id, edge.counterparty_id), max(edge.actor_id, edge.counterparty_id))
            for edge in edges
            if edge.actor_id in seen and edge.counterparty_id in seen
        }
        return seen, component_edges

    def _feature_graph_component_size(self) -> float:
        nodes, _ = self._component(self._graph_edges(_DAY))
        return float(len(nodes))

    def _feature_graph_edge_density(self) -> float:
        nodes, edges = self._component(self._graph_edges(_DAY))
        possible = len(nodes) * (len(nodes) - 1) / 2
        return 0.0 if possible == 0 else len(edges) / possible

    def _feature_graph_repeated_edge(self) -> float:
        return float(
            any(
                edge.actor_id == self.event.actor_id
                and edge.counterparty_id == self.event.counterparty_id
                for edge in self._graph_edges(_DAY)
            )
        )

    def _feature_graph_burst_motif(self) -> float:
        return float(
            sum(
                edge.actor_id == self.event.actor_id
                or edge.counterparty_id == self.event.counterparty_id
                for edge in self._graph_edges(_HOUR)
            )
        )

    def _feature_graph_prior_suspicious_count(self) -> float:
        return float(
            sum(
                edge.suspicious and edge.actor_id == self.event.actor_id
                for edge in self._graph_edges(_DAY)
            )
        )

    def _feature_dq_missing_optional_count(self) -> float:
        return float(max(0, _EXPECTED_OPTIONAL_REFS - len(self.event.optional_refs)))

    def _feature_dq_current_availability_lag_ms(self) -> float:
        return max(0.0, (self.event.available_at - self.event.event_time).total_seconds() * 1000)

    def _history_24h(self) -> tuple[ObservedEvent, ...]:
        return self._events(self.state._admitted_ids, window=_DAY)

    def _feature_dq_mean_history_lag_ms(self) -> float:
        history = self._history_24h()
        if not history:
            return _SENTINEL
        return statistics.fmean(
            max(0.0, (event.available_at - event.event_time).total_seconds() * 1000)
            for event in history
        )

    def _feature_dq_late_event_count(self) -> float:
        late = self._events(self.state._late_event_ids, window=_DAY)
        return float(len(late))

    def _feature_dq_history_count(self) -> float:
        return float(len(self._history_24h()))

    def _feature_dq_history_age_seconds(self) -> float:
        history = self._events(self.state._admitted_ids)
        if not history:
            return _SENTINEL
        first_event_time = min(item.event_time for item in history)
        return max(0.0, (self.event.event_time - first_event_time).total_seconds())

    def _feature_dq_degraded_state(self) -> float:
        history = self._events(self.state._admitted_ids)
        late = self._events(self.state._late_event_ids)
        return float(not history or bool(late))


def _decision_sort_key(event: ObservedEvent) -> tuple[datetime, str]:
    if event.decision_at is None:
        raise FeatureStateError("decision-point event must declare decision_at")
    return event.decision_at, event.event_id


def _fractional_hour(value: datetime) -> float:
    return value.hour + value.minute / 60 + value.second / 3600 + value.microsecond / 3_600_000_000


def _sum_amounts(events: Iterable[ObservedEvent]) -> float:
    return float(sum((event.amount for event in events), Decimal("0")))


def _canonical_json(document: object) -> bytes:
    return json.dumps(document, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
        "utf-8"
    )


def _timestamp(value: datetime | None) -> str | None:
    return None if value is None else value.isoformat()


def _serialize_history(history: Mapping[str, Sequence[str]]) -> dict[str, list[str]]:
    return {key: list(event_ids) for key, event_ids in sorted(history.items())}


def _pair_key(pair: tuple[str, str]) -> str:
    return json.dumps(pair, separators=(",", ":"))


def _string_list(document: Mapping[str, object], key: str) -> tuple[str, ...]:
    value = document[key]
    if type(value) is not list or any(type(item) is not str for item in value):
        raise ValueError(f"{key} must be a string list")
    items = cast(list[str], value)
    if items != sorted(items) or len(items) != len(set(items)):
        raise ValueError(f"{key} must be sorted and unique")
    return tuple(items)


def _string_keyed_list(
    document: Mapping[str, object], key: str, *, require_dicts: bool
) -> tuple[dict[str, object], ...]:
    value = document[key]
    if type(value) is not list or (require_dicts and any(type(item) is not dict for item in value)):
        raise ValueError(f"{key} must be an object list")
    return tuple(cast(list[dict[str, object]], value))


def _timestamp_mapping(document: Mapping[str, object], key: str) -> dict[str, datetime]:
    value = document[key]
    if type(value) is not dict or any(
        type(item_key) is not str or type(item_value) is not str
        for item_key, item_value in value.items()
    ):
        raise ValueError(f"{key} must map event IDs to timestamps")
    raw = cast(dict[str, str], value)
    return {
        event_id: validate_utc_timestamp(datetime.fromisoformat(timestamp))
        for event_id, timestamp in raw.items()
    }
