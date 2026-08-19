"""Deterministic, truth-blind investigation-case grouping."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from enum import StrEnum
from typing import Literal, cast

import numpy as np
from numpy.typing import NDArray
from pydantic import Field, ValidationError, field_validator, model_validator

from apar.contracts._validation import ExternalContract, validate_utc_timestamp
from apar.contracts.decisions import Action
from apar.contracts.events import EventKind, Rail
from apar.defense.contracts import ObservedEvent
from apar.defense.policy import DefenseDecision
from apar.defense.rules import DefenseReason
from apar.runs.wire import canonical_json_bytes

_CASE_ID_PREFIX = "case-"
_ESTIMATED_MINUTES = 20
_VALUE_SCALE = 1_000.0
_MAX_GROUPING_ROWS = 100_000
_MAX_IDENTIFIER_LENGTH = 4_096


class CaseContractError(ValueError):
    """Case inputs violate the closed causal-grouping contract."""


class CaseMotif(StrEnum):
    """Frozen defender-visible reason that an alert joined its causal case."""

    ISOLATED = "isolated"
    SHARED_ACTOR = "shared_actor"
    SHARED_COUNTERPARTY = "shared_counterparty"
    TRANSITIVE = "transitive"


class CaseAlertEvidence(ExternalContract):
    """Immutable facts available when one alert was assigned to a case."""

    schema_version: Literal["1.0.0"] = "1.0.0"
    event_id: str
    decision_at: datetime
    actor_id: str
    counterparty_id: str
    motif: CaseMotif
    visible_value_before_alert: Decimal = Field(ge=Decimal(0))
    latest_graph_evidence_at: datetime | None
    score: float = Field(ge=0.0, le=1.0)
    action: Action
    evidence_source_ids: tuple[str, ...]

    @field_validator("event_id", "actor_id", "counterparty_id")
    @classmethod
    def identifier_is_exact_text(cls, value: str) -> str:
        if type(value) is not str or not value or len(value) > _MAX_IDENTIFIER_LENGTH:
            raise ValueError("case evidence identifiers must be bounded non-empty text")
        return value

    @field_validator("decision_at")
    @classmethod
    def decision_time_is_utc(cls, value: datetime) -> datetime:
        return validate_utc_timestamp(value)

    @field_validator("latest_graph_evidence_at")
    @classmethod
    def latest_evidence_time_is_utc(cls, value: datetime | None) -> datetime | None:
        if value is not None:
            return validate_utc_timestamp(value)
        return None

    @field_validator("visible_value_before_alert")
    @classmethod
    def visible_value_is_finite(cls, value: Decimal) -> Decimal:
        if not value.is_finite():
            raise ValueError("visible value must be finite")
        return value

    @field_validator("score")
    @classmethod
    def score_is_finite(cls, value: float) -> float:
        if type(value) is not float or not math.isfinite(value):
            raise ValueError("case evidence score must be an exact finite float")
        return value

    @field_validator("evidence_source_ids")
    @classmethod
    def sources_are_canonical(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if (
            not value
            or any(
                type(item) is not str
                or not item
                or len(item) > _MAX_IDENTIFIER_LENGTH
                for item in value
            )
            or value != tuple(sorted(set(value)))
        ):
            raise ValueError("case evidence sources must be bounded, sorted, and unique")
        return value

    @model_validator(mode="after")
    def chronology_and_binding_are_causal(self) -> CaseAlertEvidence:
        if self.event_id not in self.evidence_source_ids:
            raise ValueError("case evidence must bind its alert event")
        if (
            self.latest_graph_evidence_at is not None
            and self.latest_graph_evidence_at >= self.decision_at
        ):
            raise ValueError("case graph evidence must be strictly earlier than the alert")
        return self


class InvestigationCase(ExternalContract):
    """An immutable past-only view of one grouped investigation."""

    schema_version: Literal["1.0.0"] = "1.0.0"
    case_id: str
    opened_at: datetime
    event_ids: tuple[str, ...]
    actor_ids: tuple[str, ...]
    counterparty_ids: tuple[str, ...]
    first_alert_at: datetime
    priority: float = Field(ge=0.0, le=100.0)
    estimated_minutes: int = Field(default=_ESTIMATED_MINUTES, ge=1)
    first_evidence_ids: tuple[str, ...]
    alert_evidence: tuple[CaseAlertEvidence, ...]

    @field_validator("opened_at", "first_alert_at")
    @classmethod
    def timestamps_are_utc(cls, value: datetime) -> datetime:
        return validate_utc_timestamp(value)

    @field_validator("case_id")
    @classmethod
    def case_id_is_content_derived(cls, value: str) -> str:
        digest = value.removeprefix(_CASE_ID_PREFIX)
        if not value.startswith(_CASE_ID_PREFIX) or len(digest) != 64:
            raise ValueError("case_id must contain a lowercase SHA-256 digest")
        try:
            int(digest, 16)
        except ValueError as error:
            raise ValueError("case_id must contain a lowercase SHA-256 digest") from error
        if digest != digest.lower():
            raise ValueError("case_id must contain a lowercase SHA-256 digest")
        return value

    @field_validator(
        "event_ids", "actor_ids", "counterparty_ids", "first_evidence_ids"
    )
    @classmethod
    def identifiers_are_canonical(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if not value or any(
            type(item) is not str
            or not item
            or len(item) > _MAX_IDENTIFIER_LENGTH
            for item in value
        ):
            raise ValueError("case identifier sets must contain exact non-empty strings")
        if value != tuple(sorted(set(value))):
            raise ValueError("case identifier sets must be sorted and unique")
        return value

    @field_validator("priority")
    @classmethod
    def priority_is_finite(cls, value: float) -> float:
        if type(value) is not float or not math.isfinite(value):
            raise ValueError("case priority must be finite")
        return value

    @field_validator("estimated_minutes", mode="before")
    @classmethod
    def estimated_minutes_is_bounded_exact_int(cls, value: object) -> object:
        if type(value) is not int or not 1 <= value <= 1_000_000:
            raise ValueError("estimated minutes must be a bounded exact integer")
        return value

    @field_validator("alert_evidence")
    @classmethod
    def alert_evidence_is_exact_tuple(cls, value: object) -> object:
        if type(value) is not tuple or any(
            type(item) is not CaseAlertEvidence for item in cast(tuple[object, ...], value)
        ):
            raise ValueError("alert evidence must contain exact CaseAlertEvidence rows")
        return value

    @model_validator(mode="after")
    def chronology_is_coherent(self) -> InvestigationCase:
        if self.opened_at != self.first_alert_at:
            raise ValueError("opened_at must equal first_alert_at")
        if not self.alert_evidence:
            raise ValueError("investigation cases require causal alert evidence")
        expected = tuple(
            sorted(self.alert_evidence, key=lambda row: (row.decision_at, row.event_id))
        )
        if self.alert_evidence != expected:
            raise ValueError("case alert evidence must use canonical causal order")
        evidence_ids = tuple(row.event_id for row in self.alert_evidence)
        if len(set(evidence_ids)) != len(evidence_ids):
            raise ValueError("case alert evidence event IDs must be unique")
        if self.event_ids != tuple(sorted(evidence_ids)):
            raise ValueError("case event IDs must exactly bind alert evidence")
        opening_ids = tuple(
            sorted(row.event_id for row in self.alert_evidence if row.decision_at == self.opened_at)
        )
        if not set(self.first_evidence_ids).issubset(opening_ids):
            raise ValueError("first evidence IDs must bind an opening alert batch")
        if self.case_id != _case_id(self.first_evidence_ids):
            raise ValueError("case_id must be content-derived from first evidence")
        if not set(row.actor_id for row in self.alert_evidence).issubset(self.actor_ids):
            raise ValueError("case actor IDs must contain every alert actor")
        if not set(row.counterparty_id for row in self.alert_evidence).issubset(
            self.counterparty_ids
        ):
            raise ValueError("case counterparty IDs must contain every alert counterparty")
        return self


@dataclass(slots=True)
class _CaseState:
    case_id: str
    opened_at: datetime
    event_ids: set[str]
    actor_ids: set[str]
    counterparty_ids: set[str]
    priority: float
    first_evidence_ids: tuple[str, ...]
    alert_evidence: list[CaseAlertEvidence]


class ReviewCaseCounter(ExternalContract):
    """Frozen adapter from candidate action vectors to causal review-case counts."""

    observations: tuple[ObservedEvent, ...]
    decisions: tuple[DefenseDecision, ...]
    as_of: datetime

    @field_validator("observations", mode="before")
    @classmethod
    def observations_are_exact(cls, value: object) -> object:
        if type(value) is not tuple or any(
            type(row) is not ObservedEvent for row in cast(tuple[object, ...], value)
        ):
            raise ValueError("observations must contain exact ObservedEvent rows")
        return tuple(
            _revalidate_observation(row)
            for row in cast(tuple[ObservedEvent, ...], value)
        )

    @field_validator("decisions", mode="before")
    @classmethod
    def decisions_are_exact(cls, value: object) -> object:
        if type(value) is not tuple or any(
            type(row) is not DefenseDecision for row in cast(tuple[object, ...], value)
        ):
            raise ValueError("decisions must contain exact DefenseDecision rows")
        return tuple(
            _revalidate_decision(row)
            for row in cast(tuple[DefenseDecision, ...], value)
        )

    @field_validator("as_of", mode="before")
    @classmethod
    def as_of_is_utc(cls, value: object) -> datetime:
        if type(value) is not datetime:
            raise ValueError("as_of must be an exact UTC datetime")
        return validate_utc_timestamp(value)

    @model_validator(mode="after")
    def rows_are_exact_and_canonical(self) -> ReviewCaseCounter:
        _validated_rows(self.observations, self.decisions, self.as_of)
        expected = tuple(
            sorted(
                self.observations,
                key=lambda row: (cast(datetime, row.decision_at), row.event_id),
            )
        )
        if self.observations != expected:
            raise ValueError("review-case observations must use canonical decision order")
        if tuple(row.event_id for row in self.decisions) != tuple(
            row.event_id for row in self.observations
        ):
            raise ValueError("review-case decisions must align with canonical observations")
        return self

    def __call__(self, actions: NDArray[np.object_]) -> int:
        """Count grouped interventions without observing labels or action severity."""
        try:
            if type(actions) is not np.ndarray:
                raise TypeError("actions must be an exact numpy.ndarray")
            if actions.ndim != 1 or actions.dtype != np.dtype(object):
                raise TypeError("actions must be a one-dimensional array with dtype object")
            if len(actions) != len(self.decisions):
                raise ValueError("action vector length must equal bound decision rows")
            candidate_actions: list[Action] = []
            for action in actions:
                if type(action) is not Action:
                    raise TypeError("every action must be an exact Action")
                candidate_actions.append(action)
            return len(
                _group_validated(
                    self.observations,
                    self.decisions,
                    actions=tuple(candidate_actions),
                )
            )
        except (ArithmeticError, MemoryError, OverflowError) as error:
            raise CaseContractError("case counting exceeded frozen resource bounds") from error


def bind_review_case_counter(
    observations: Sequence[ObservedEvent],
    decisions: Sequence[DefenseDecision],
    *,
    as_of: datetime,
) -> ReviewCaseCounter:
    """Bind exact canonical decision rows for threshold selection."""
    try:
        observation_rows, decision_rows = _validated_rows(observations, decisions, as_of)
        expected_observations = tuple(
            sorted(
                observation_rows,
                key=lambda row: (cast(datetime, row.decision_at), row.event_id),
            )
        )
        if observation_rows != expected_observations:
            raise CaseContractError("review-case observations must use canonical decision order")
        if tuple(row.event_id for row in decision_rows) != tuple(
            row.event_id for row in observation_rows
        ):
            raise CaseContractError("review-case decisions must align with observations")
        return ReviewCaseCounter(
            observations=observation_rows,
            decisions=decision_rows,
            as_of=as_of,
        )
    except CaseContractError:
        raise
    except (ArithmeticError, MemoryError, OverflowError) as error:
        raise CaseContractError("case counter binding exceeded frozen resource bounds") from error


def group_cases(
    observations: Sequence[ObservedEvent],
    decisions: Sequence[DefenseDecision],
    as_of: datetime,
) -> tuple[InvestigationCase, ...]:
    """Group alerts using only graph edges strictly available at each decision time.

    The priority frozen when a case first opens is::

        100 * (0.45 * max_score
             + 0.30 * risk_amount / (risk_amount + 1000)
             + 0.15 * min(max(entity_count - 2, 0) / 8, 1)
             + 0.10 * 1 / (1 + minutes_since_latest_graph_evidence))

    The recency term is zero when there is no prior graph evidence. Later alerts
    may extend or merge a case, but the canonical first-evidence ID and priority
    of the earliest case remain unchanged.
    """
    try:
        observation_rows, decision_rows = _validated_rows(observations, decisions, as_of)
        return _group_validated(observation_rows, decision_rows)
    except CaseContractError:
        raise
    except (ArithmeticError, MemoryError, OverflowError) as error:
        raise CaseContractError("case grouping exceeded frozen resource bounds") from error


def _validated_rows(
    observations: Sequence[ObservedEvent],
    decisions: Sequence[DefenseDecision],
    as_of: datetime,
) -> tuple[tuple[ObservedEvent, ...], tuple[DefenseDecision, ...]]:
    _utc(as_of, label="as_of")
    if type(observations) is not tuple or type(decisions) is not tuple:
        raise TypeError("observations and decisions must be exact tuples")
    observation_rows = cast(tuple[ObservedEvent, ...], observations)
    decision_rows = cast(tuple[DefenseDecision, ...], decisions)
    if len(observation_rows) != len(decision_rows):
        raise CaseContractError("observations and decisions must have equal lengths")
    if len(observation_rows) > _MAX_GROUPING_ROWS:
        raise CaseContractError("case grouping row count exceeds frozen resource cap")
    revalidated_observations: list[ObservedEvent] = []
    for observation_row in observation_rows:
        if type(observation_row) is not ObservedEvent:
            raise TypeError("observations must contain exact ObservedEvent rows")
        validated_observation = _revalidate_observation(observation_row)
        _validate_observation(validated_observation, as_of)
        revalidated_observations.append(validated_observation)
    revalidated_decisions: list[DefenseDecision] = []
    for decision_row in decision_rows:
        if type(decision_row) is not DefenseDecision:
            raise TypeError("decisions must contain exact DefenseDecision rows")
        validated_decision = _revalidate_decision(decision_row)
        if type(validated_decision.action) is not Action:
            raise CaseContractError("decision action must be an exact Action")
        revalidated_decisions.append(validated_decision)
    canonical_observations = tuple(revalidated_observations)
    canonical_decisions = tuple(revalidated_decisions)
    observation_ids = tuple(row.event_id for row in canonical_observations)
    decision_ids = tuple(row.event_id for row in canonical_decisions)
    if len(set(observation_ids)) != len(observation_ids):
        raise CaseContractError("duplicate observation event_id")
    if len(set(decision_ids)) != len(decision_ids):
        raise CaseContractError("duplicate decision event_id")
    if set(observation_ids) != set(decision_ids):
        raise CaseContractError("observation and decision event IDs must match bijectively")
    return canonical_observations, canonical_decisions


def _validate_observation(row: ObservedEvent, as_of: datetime) -> None:
    if not row.is_decision_point or row.decision_at is None:
        raise CaseContractError("every observation must be an exact decision-point row")
    for timestamp_label, timestamp_value in (
        ("event_time", row.event_time),
        ("available_at", row.available_at),
        ("decision_at", row.decision_at),
    ):
        _utc(timestamp_value, label=f"observation {timestamp_label}")
    if row.event_time > row.available_at:
        raise CaseContractError("observation event_time must not exceed available_at")
    if row.available_at > row.decision_at:
        raise CaseContractError("observation available_at must not exceed decision_at")
    if row.event_time > as_of or row.available_at > as_of or row.decision_at > as_of:
        raise CaseContractError("observation or decision occurs after as_of")
    for identifier_label, identifier_value in (
        ("event_id", row.event_id),
        ("actor_id", row.actor_id),
        ("counterparty_id", row.counterparty_id),
    ):
        if (
            type(identifier_value) is not str
            or not identifier_value
            or len(identifier_value) > _MAX_IDENTIFIER_LENGTH
        ):
            raise CaseContractError(
                f"observation {identifier_label} must be exact non-empty text"
            )
    if not row.amount.is_finite() or row.amount < Decimal(0):
        raise CaseContractError("observation amount must be finite and nonnegative")


def _revalidate_observation(row: ObservedEvent) -> ObservedEvent:
    if (
        type(row.event_id) is not str
        or type(row.payment_id) is not str
        or type(row.rail) is not Rail
        or type(row.event_type) is not EventKind
        or type(row.amount) is not Decimal
        or type(row.currency) is not str
        or type(row.event_time) is not datetime
        or type(row.available_at) is not datetime
        or (row.decision_at is not None and type(row.decision_at) is not datetime)
        or type(row.actor_id) is not str
        or type(row.counterparty_id) is not str
        or type(row.optional_refs) is not dict
        or any(
            type(key) is not str or type(value) is not str
            for key, value in row.optional_refs.items()
        )
        or type(row.integrity_status) is not str
        or (row.integrity_reason is not None and type(row.integrity_reason) is not str)
        or type(row.is_decision_point) is not bool
    ):
        raise CaseContractError("ObservedEvent contains non-exact field values")
    try:
        return ObservedEvent.model_validate(
            row.model_dump(mode="python", warnings=False), strict=True
        )
    except (AttributeError, TypeError, ValidationError, ValueError) as error:
        raise CaseContractError(
            "ObservedEvent failed deterministic semantic revalidation"
        ) from error


def _revalidate_decision(row: DefenseDecision) -> DefenseDecision:
    if (
        type(row.event_id) is not str
        or type(row.action) is not Action
        or type(row.score) is not float
        or type(row.rule_score) is not float
        or (row.calibrated_score is not None and type(row.calibrated_score) is not float)
        or type(row.reason_codes) is not tuple
        or any(type(reason) is not DefenseReason for reason in row.reason_codes)
        or type(row.evidence_source_ids) is not tuple
        or any(type(source) is not str for source in row.evidence_source_ids)
        or type(row.fallback_used) is not bool
        or (
            row.fallback_reason is not None
            and type(row.fallback_reason) is not DefenseReason
        )
        or (
            row.failed_component_version is not None
            and type(row.failed_component_version) is not str
        )
        or type(row.latency_ms) is not float
        or type(row.policy_version) is not str
    ):
        raise CaseContractError("DefenseDecision contains non-exact field values")
    try:
        return DefenseDecision.model_validate(
            row.model_dump(mode="python", warnings=False), strict=True
        )
    except (AttributeError, TypeError, ValidationError, ValueError) as error:
        raise CaseContractError(
            "DefenseDecision failed deterministic semantic revalidation"
        ) from error


def _utc(value: datetime, *, label: str) -> None:
    if type(value) is not datetime:
        raise TypeError(f"{label} must be an exact datetime")
    try:
        validate_utc_timestamp(value)
    except ValueError as error:
        raise CaseContractError(f"{label} must be timezone-aware UTC") from error


@dataclass(slots=True)
class _GraphComponent:
    nodes: set[str]
    case_ids: set[str]
    total_value: Decimal = Decimal(0)
    first_event_id: str | None = None
    latest_evidence: tuple[datetime, str] | None = None


class _IncrementalGraph:
    """Union-find graph that consumes each available observation exactly once."""

    __slots__ = ("_components", "_parent", "_size")

    def __init__(self) -> None:
        self._parent: dict[str, str] = {}
        self._size: dict[str, int] = {}
        self._components: dict[str, _GraphComponent] = {}

    def _ensure(self, node: str) -> str:
        if node not in self._parent:
            self._parent[node] = node
            self._size[node] = 1
            self._components[node] = _GraphComponent(nodes={node}, case_ids=set())
        return node

    def find(self, node: str) -> str:
        self._ensure(node)
        root = node
        while self._parent[root] != root:
            root = self._parent[root]
        while self._parent[node] != node:
            parent = self._parent[node]
            self._parent[node] = root
            node = parent
        return root

    def _union(self, left: str, right: str) -> str:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return left_root
        left_key = (self._size[left_root], left_root)
        right_key = (self._size[right_root], right_root)
        if left_key < right_key:
            left_root, right_root = right_root, left_root
        self._parent[right_root] = left_root
        self._size[left_root] += self._size.pop(right_root)
        target = self._components[left_root]
        source = self._components.pop(right_root)
        target.nodes.update(source.nodes)
        target.case_ids.update(source.case_ids)
        target.total_value += source.total_value
        if target.first_event_id is None or (
            source.first_event_id is not None
            and source.first_event_id < target.first_event_id
        ):
            target.first_event_id = source.first_event_id
        if source.latest_evidence is not None and (
            target.latest_evidence is None
            or source.latest_evidence > target.latest_evidence
        ):
            target.latest_evidence = source.latest_evidence
        return left_root

    def add_observation(self, row: ObservedEvent) -> None:
        root = self._union(_actor(row.actor_id), _counterparty(row.counterparty_id))
        component = self._components[root]
        component.total_value += row.amount
        if component.first_event_id is None or row.event_id < component.first_event_id:
            component.first_event_id = row.event_id
        evidence_key = (row.available_at, row.event_id)
        if component.latest_evidence is None or evidence_key > component.latest_evidence:
            component.latest_evidence = evidence_key

    def roots_for(self, row: ObservedEvent) -> frozenset[str]:
        return frozenset(
            {
                self.find(_actor(row.actor_id)),
                self.find(_counterparty(row.counterparty_id)),
            }
        )

    def canonical_roots(self, roots: set[str] | frozenset[str]) -> frozenset[str]:
        return frozenset(self.find(root) for root in roots)

    def bind_case(self, case_id: str, roots: set[str] | frozenset[str]) -> None:
        for root in self.canonical_roots(roots):
            self._components[root].case_ids = {case_id}

    def case_ids(self, roots: set[str] | frozenset[str]) -> set[str]:
        result: set[str] = set()
        for root in self.canonical_roots(roots):
            result.update(self._components[root].case_ids)
        return result

    def entity_count(self, roots: set[str] | frozenset[str]) -> int:
        return sum(
            len(self._components[root].nodes) for root in self.canonical_roots(roots)
        )

    def visible_value(self, roots: set[str] | frozenset[str]) -> Decimal:
        return sum(
            (self._components[root].total_value for root in self.canonical_roots(roots)),
            start=Decimal(0),
        )

    def latest_evidence_at(
        self, roots: set[str] | frozenset[str]
    ) -> datetime | None:
        evidence = tuple(
            item
            for root in self.canonical_roots(roots)
            if (item := self._components[root].latest_evidence) is not None
        )
        return max(evidence)[0] if evidence else None

    def bounded_source_ids(
        self, roots: set[str] | frozenset[str], *, event_id: str
    ) -> tuple[str, ...]:
        sources = {event_id}
        for root in self.canonical_roots(roots):
            component = self._components[root]
            if component.first_event_id is not None:
                sources.add(component.first_event_id)
            if component.latest_evidence is not None:
                sources.add(component.latest_evidence[1])
        return tuple(sorted(sources))


def _group_validated(
    observations: tuple[ObservedEvent, ...],
    decisions: tuple[DefenseDecision, ...],
    *,
    actions: tuple[Action, ...] | None = None,
) -> tuple[InvestigationCase, ...]:
    if not observations:
        return ()
    if actions is not None and len(actions) != len(decisions):
        raise CaseContractError("candidate actions must align with decisions")
    observation_by_id = {row.event_id: row for row in observations}
    decision_by_id = {row.event_id: row for row in decisions}
    action_by_id = {
        row.event_id: row.action if actions is None else actions[index]
        for index, row in enumerate(decisions)
    }
    alerts = tuple(
        sorted(
            (
                (cast(datetime, observation_by_id[event_id].decision_at), event_id)
                for event_id, action in action_by_id.items()
                if action is not Action.APPROVE
            )
        )
    )
    if not alerts:
        return ()

    graph = _IncrementalGraph()
    edges = tuple(sorted(observations, key=lambda row: (row.available_at, row.event_id)))
    edge_position = 0
    states: dict[str, _CaseState] = {}
    case_aliases: dict[str, str] = {}

    def resolve_case(case_id: str) -> str:
        trail: list[str] = []
        while case_id in case_aliases:
            trail.append(case_id)
            case_id = case_aliases[case_id]
        for stale in trail:
            case_aliases[stale] = case_id
        return case_id

    position = 0
    while position < len(alerts):
        decision_time = alerts[position][0]
        while edge_position < len(edges) and edges[edge_position].available_at < decision_time:
            graph.add_observation(edges[edge_position])
            edge_position += 1
        end = position
        while end < len(alerts) and alerts[end][0] == decision_time:
            end += 1
        batch_ids = tuple(event_id for _, event_id in alerts[position:end])
        batch_roots = tuple(
            graph.roots_for(observation_by_id[event_id]) for event_id in batch_ids
        )
        for members in _batch_components(batch_roots):
            member_ids = tuple(sorted(batch_ids[index] for index in members))
            member_roots: set[str] = set()
            for index in members:
                member_roots.update(batch_roots[index])
            member_roots = set(graph.canonical_roots(member_roots))
            matching_ids = {
                resolve_case(case_id) for case_id in graph.case_ids(member_roots)
            }
            matching = tuple(
                sorted(
                    (states[case_id] for case_id in matching_ids if case_id in states),
                    key=lambda state: (state.opened_at, state.case_id),
                )
            )
            actor_counts: dict[str, int] = {}
            counterparty_counts: dict[str, int] = {}
            for event_id in member_ids:
                row = observation_by_id[event_id]
                actor_counts[row.actor_id] = actor_counts.get(row.actor_id, 0) + 1
                counterparty_counts[row.counterparty_id] = (
                    counterparty_counts.get(row.counterparty_id, 0) + 1
                )
            evidence_rows: list[CaseAlertEvidence] = []
            for event_id in member_ids:
                row = observation_by_id[event_id]
                motif = _motif_for_alert(
                    row,
                    matching,
                    actor_counts=actor_counts,
                    counterparty_counts=counterparty_counts,
                    connected_batch_size=len(member_ids),
                )
                row_roots = graph.roots_for(row)
                evidence_rows.append(
                    CaseAlertEvidence(
                        event_id=event_id,
                        decision_at=decision_time,
                        actor_id=row.actor_id,
                        counterparty_id=row.counterparty_id,
                        motif=motif,
                        visible_value_before_alert=graph.visible_value(row_roots),
                        latest_graph_evidence_at=graph.latest_evidence_at(row_roots),
                        score=decision_by_id[event_id].score,
                        action=action_by_id[event_id],
                        evidence_source_ids=graph.bounded_source_ids(
                            row_roots, event_id=event_id
                        ),
                    )
                )
            if matching:
                anchor = matching[0]
                for merged in matching[1:]:
                    anchor.event_ids.update(merged.event_ids)
                    anchor.actor_ids.update(merged.actor_ids)
                    anchor.counterparty_ids.update(merged.counterparty_ids)
                    anchor.alert_evidence.extend(merged.alert_evidence)
                    case_aliases[merged.case_id] = anchor.case_id
                    del states[merged.case_id]
                anchor.event_ids.update(member_ids)
                anchor.alert_evidence.extend(evidence_rows)
            else:
                priority = _priority(
                    member_ids,
                    entity_count=graph.entity_count(member_roots),
                    latest_graph_evidence_at=graph.latest_evidence_at(member_roots),
                    observation_by_id=observation_by_id,
                    decision_by_id=decision_by_id,
                    decision_time=decision_time,
                )
                case_id = _case_id(member_ids)
                anchor = _CaseState(
                    case_id=case_id,
                    opened_at=decision_time,
                    event_ids=set(member_ids),
                    actor_ids=set(),
                    counterparty_ids=set(),
                    priority=priority,
                    first_evidence_ids=member_ids,
                    alert_evidence=evidence_rows,
                )
                states[case_id] = anchor
            for event_id in member_ids:
                row = observation_by_id[event_id]
                anchor.actor_ids.add(row.actor_id)
                anchor.counterparty_ids.add(row.counterparty_id)
            graph.bind_case(anchor.case_id, member_roots)
        position = end

    return tuple(
        InvestigationCase(
            case_id=state.case_id,
            opened_at=state.opened_at,
            event_ids=tuple(sorted(state.event_ids)),
            actor_ids=tuple(sorted(state.actor_ids)),
            counterparty_ids=tuple(sorted(state.counterparty_ids)),
            first_alert_at=state.opened_at,
            priority=state.priority,
            estimated_minutes=_ESTIMATED_MINUTES,
            first_evidence_ids=state.first_evidence_ids,
            alert_evidence=tuple(
                sorted(state.alert_evidence, key=lambda row: (row.decision_at, row.event_id))
            ),
        )
        for state in sorted(states.values(), key=lambda item: (item.opened_at, item.case_id))
    )


def _batch_components(
    root_sets: tuple[frozenset[str], ...],
) -> tuple[tuple[int, ...], ...]:
    parent = list(range(len(root_sets)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[max(left_root, right_root)] = min(left_root, right_root)

    first_by_root: dict[str, int] = {}
    for index, roots in enumerate(root_sets):
        for root in roots:
            if root in first_by_root:
                union(index, first_by_root[root])
            else:
                first_by_root[root] = index
    members: dict[int, list[int]] = {}
    for index in range(len(root_sets)):
        members.setdefault(find(index), []).append(index)
    return tuple(tuple(group) for _, group in sorted(members.items()))


def _motif_for_alert(
    row: ObservedEvent,
    matching: tuple[_CaseState, ...],
    *,
    actor_counts: dict[str, int],
    counterparty_counts: dict[str, int],
    connected_batch_size: int,
) -> CaseMotif:
    if any(row.actor_id in state.actor_ids for state in matching) or actor_counts.get(
        row.actor_id, 0
    ) > 1:
        return CaseMotif.SHARED_ACTOR
    if any(
        row.counterparty_id in state.counterparty_ids for state in matching
    ) or counterparty_counts.get(row.counterparty_id, 0) > 1:
        return CaseMotif.SHARED_COUNTERPARTY
    if matching or connected_batch_size > 1:
        return CaseMotif.TRANSITIVE
    return CaseMotif.ISOLATED


def _priority(
    event_ids: tuple[str, ...],
    *,
    entity_count: int,
    latest_graph_evidence_at: datetime | None,
    observation_by_id: dict[str, ObservedEvent],
    decision_by_id: dict[str, DefenseDecision],
    decision_time: datetime,
) -> float:
    max_score = max(decision_by_id[event_id].score for event_id in event_ids)
    risk_amount = math.fsum(
        float(observation_by_id[event_id].amount) * decision_by_id[event_id].score
        for event_id in event_ids
    )
    value_term = risk_amount / (risk_amount + _VALUE_SCALE) if risk_amount else 0.0
    coverage = min(max((entity_count - 2) / 8.0, 0.0), 1.0)
    if latest_graph_evidence_at is not None:
        age_minutes = max(
            0.0,
            (decision_time - latest_graph_evidence_at).total_seconds() / 60.0,
        )
        recency = 1.0 / (1.0 + age_minutes)
    else:
        recency = 0.0
    priority = 100.0 * (
        0.45 * max_score + 0.30 * value_term + 0.15 * coverage + 0.10 * recency
    )
    if not math.isfinite(priority):
        raise CaseContractError("case priority inputs must be finite")
    return round(min(max(priority, 0.0), 100.0), 6)


def _case_id(first_evidence: tuple[str, ...]) -> str:
    payload = canonical_json_bytes(
        {"domain": "apar-investigation-case-v1", "first_evidence": list(first_evidence)}
    )
    return _CASE_ID_PREFIX + hashlib.sha256(payload).hexdigest()


def _actor(identifier: str) -> str:
    return f"actor:{identifier}"


def _counterparty(identifier: str) -> str:
    return f"counterparty:{identifier}"
