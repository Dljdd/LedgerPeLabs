"""Deterministic, truth-blind investigation-case grouping."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import Literal, cast

import numpy as np
from numpy.typing import NDArray
from pydantic import Field, field_validator, model_validator

from apar.contracts._validation import ExternalContract, validate_utc_timestamp
from apar.contracts.decisions import Action
from apar.defense.contracts import ObservedEvent
from apar.defense.policy import DefenseDecision
from apar.runs.wire import canonical_json_bytes

_CASE_ID_PREFIX = "case-"
_ESTIMATED_MINUTES = 20
_VALUE_SCALE = 1_000.0


class CaseContractError(ValueError):
    """Case inputs violate the closed causal-grouping contract."""


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

    @field_validator("event_ids", "actor_ids", "counterparty_ids")
    @classmethod
    def identifiers_are_canonical(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if not value or any(type(item) is not str or not item for item in value):
            raise ValueError("case identifier sets must contain exact non-empty strings")
        if value != tuple(sorted(set(value))):
            raise ValueError("case identifier sets must be sorted and unique")
        return value

    @field_validator("priority")
    @classmethod
    def priority_is_finite(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("case priority must be finite")
        return value

    @model_validator(mode="after")
    def chronology_is_coherent(self) -> InvestigationCase:
        if self.opened_at != self.first_alert_at:
            raise ValueError("opened_at must equal first_alert_at")
        return self


@dataclass(slots=True)
class _CaseState:
    case_id: str
    opened_at: datetime
    event_ids: set[str]
    actor_ids: set[str]
    counterparty_ids: set[str]
    priority: float

    @property
    def nodes(self) -> set[str]:
        return {_actor(value) for value in self.actor_ids} | {
            _counterparty(value) for value in self.counterparty_ids
        }


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
        return value

    @field_validator("decisions", mode="before")
    @classmethod
    def decisions_are_exact(cls, value: object) -> object:
        if type(value) is not tuple or any(
            type(row) is not DefenseDecision for row in cast(tuple[object, ...], value)
        ):
            raise ValueError("decisions must contain exact DefenseDecision rows")
        return value

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
        if type(actions) is not np.ndarray:
            raise TypeError("actions must be an exact numpy.ndarray")
        if actions.ndim != 1 or actions.dtype != np.dtype(object):
            raise TypeError("actions must be a one-dimensional array with dtype object")
        if len(actions) != len(self.decisions):
            raise ValueError("action vector length must equal bound decision rows")
        candidate: list[DefenseDecision] = []
        for template, action in zip(self.decisions, actions, strict=True):
            if type(action) is not Action:
                raise TypeError("every action must be an exact Action")
            document = template.model_dump(mode="python")
            document["action"] = action
            candidate.append(DefenseDecision.model_validate(document))
        return len(group_cases(self.observations, tuple(candidate), as_of=self.as_of))


def bind_review_case_counter(
    observations: Sequence[ObservedEvent],
    decisions: Sequence[DefenseDecision],
    *,
    as_of: datetime,
) -> ReviewCaseCounter:
    """Bind exact canonical decision rows for threshold selection."""
    observation_rows, decision_rows = _validated_rows(observations, decisions, as_of)
    expected_observations = tuple(
        sorted(observation_rows, key=lambda row: (cast(datetime, row.decision_at), row.event_id))
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
    observation_rows, decision_rows = _validated_rows(observations, decisions, as_of)
    if not observation_rows:
        return ()
    observation_by_id = {row.event_id: row for row in observation_rows}
    decision_by_id = {row.event_id: row for row in decision_rows}
    alerts = tuple(
        sorted(
            (
                (cast(datetime, observation_by_id[event_id].decision_at), event_id)
                for event_id, item in decision_by_id.items()
                if item.action is not Action.APPROVE
            ),
        )
    )
    states: dict[str, _CaseState] = {}
    position = 0
    while position < len(alerts):
        decision_time = alerts[position][0]
        end = position
        while end < len(alerts) and alerts[end][0] == decision_time:
            end += 1
        batch_ids = tuple(event_id for _, event_id in alerts[position:end])
        graph = _graph_components(observation_rows, before=decision_time)
        node_sets = tuple(
            _nodes_for_alert(observation_by_id[event_id], graph) for event_id in batch_ids
        )
        for members in _connected_batch_members(node_sets):
            member_ids = tuple(sorted(batch_ids[index] for index in members))
            member_nodes: set[str] = set()
            for index in members:
                member_nodes.update(node_sets[index])
            matching = tuple(
                sorted(
                    (state for state in states.values() if state.nodes & member_nodes),
                    key=lambda state: (state.opened_at, state.case_id),
                )
            )
            if matching:
                anchor = matching[0]
                for merged in matching[1:]:
                    anchor.event_ids.update(merged.event_ids)
                    anchor.actor_ids.update(merged.actor_ids)
                    anchor.counterparty_ids.update(merged.counterparty_ids)
                    del states[merged.case_id]
                anchor.event_ids.update(member_ids)
                _add_nodes(anchor, member_nodes)
            else:
                priority = _priority(
                    member_ids,
                    member_nodes,
                    observation_by_id,
                    decision_by_id,
                    observation_rows,
                    decision_time,
                )
                case_id = _case_id(member_ids)
                actors, counterparties = _split_nodes(member_nodes)
                states[case_id] = _CaseState(
                    case_id=case_id,
                    opened_at=decision_time,
                    event_ids=set(member_ids),
                    actor_ids=actors,
                    counterparty_ids=counterparties,
                    priority=priority,
                )
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
        )
        for state in sorted(states.values(), key=lambda item: (item.opened_at, item.case_id))
    )


def _validated_rows(
    observations: Sequence[ObservedEvent],
    decisions: Sequence[DefenseDecision],
    as_of: datetime,
) -> tuple[tuple[ObservedEvent, ...], tuple[DefenseDecision, ...]]:
    _utc(as_of, label="as_of")
    if not isinstance(observations, (tuple, list)) or not isinstance(decisions, (tuple, list)):
        raise TypeError("observations and decisions must be exact row sequences")
    observation_rows = tuple(observations)
    decision_rows = tuple(decisions)
    if len(observation_rows) != len(decision_rows):
        raise CaseContractError("observations and decisions must have equal lengths")
    for row in observation_rows:
        if type(row) is not ObservedEvent:
            raise TypeError("observations must contain exact ObservedEvent rows")
        _validate_observation(row, as_of)
    for row in decision_rows:
        if type(row) is not DefenseDecision:
            raise TypeError("decisions must contain exact DefenseDecision rows")
    observation_ids = tuple(row.event_id for row in observation_rows)
    decision_ids = tuple(row.event_id for row in decision_rows)
    if len(set(observation_ids)) != len(observation_ids):
        raise CaseContractError("duplicate observation event_id")
    if len(set(decision_ids)) != len(decision_ids):
        raise CaseContractError("duplicate decision event_id")
    if set(observation_ids) != set(decision_ids):
        raise CaseContractError("observation and decision event IDs must match bijectively")
    return observation_rows, decision_rows


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
        if type(identifier_value) is not str or not identifier_value:
            raise CaseContractError(
                f"observation {identifier_label} must be exact non-empty text"
            )
    if not row.amount.is_finite() or row.amount < Decimal(0):
        raise CaseContractError("observation amount must be finite and nonnegative")


def _utc(value: datetime, *, label: str) -> None:
    if type(value) is not datetime:
        raise TypeError(f"{label} must be an exact datetime")
    try:
        validate_utc_timestamp(value)
    except ValueError as error:
        raise CaseContractError(f"{label} must be timezone-aware UTC") from error


def _graph_components(
    observations: tuple[ObservedEvent, ...], *, before: datetime
) -> dict[str, frozenset[str]]:
    parent: dict[str, str] = {}

    def find(node: str) -> str:
        parent.setdefault(node, node)
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    def union(left: str, right: str) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root == right_root:
            return
        low, high = sorted((left_root, right_root))
        parent[high] = low

    for row in sorted(observations, key=lambda item: (item.available_at, item.event_id)):
        if row.available_at < before:
            union(_actor(row.actor_id), _counterparty(row.counterparty_id))
    members: dict[str, set[str]] = {}
    for node in tuple(parent):
        members.setdefault(find(node), set()).add(node)
    result: dict[str, frozenset[str]] = {}
    for component in members.values():
        frozen = frozenset(component)
        for node in component:
            result[node] = frozen
    return result


def _nodes_for_alert(row: ObservedEvent, graph: dict[str, frozenset[str]]) -> set[str]:
    actor = _actor(row.actor_id)
    counterparty = _counterparty(row.counterparty_id)
    return set(graph.get(actor, (actor,))) | set(graph.get(counterparty, (counterparty,)))


def _connected_batch_members(node_sets: tuple[set[str], ...]) -> tuple[tuple[int, ...], ...]:
    remaining = set(range(len(node_sets)))
    groups: list[tuple[int, ...]] = []
    while remaining:
        seed = min(remaining)
        remaining.remove(seed)
        group = {seed}
        nodes = set(node_sets[seed])
        changed = True
        while changed:
            changed = False
            for index in sorted(remaining):
                if node_sets[index] & nodes:
                    remaining.remove(index)
                    group.add(index)
                    nodes.update(node_sets[index])
                    changed = True
        groups.append(tuple(sorted(group)))
    return tuple(groups)


def _priority(
    event_ids: tuple[str, ...],
    nodes: set[str],
    observation_by_id: dict[str, ObservedEvent],
    decision_by_id: dict[str, DefenseDecision],
    observations: tuple[ObservedEvent, ...],
    decision_time: datetime,
) -> float:
    max_score = max(decision_by_id[event_id].score for event_id in event_ids)
    risk_amount = math.fsum(
        float(observation_by_id[event_id].amount) * decision_by_id[event_id].score
        for event_id in event_ids
    )
    value_term = risk_amount / (risk_amount + _VALUE_SCALE) if risk_amount else 0.0
    coverage = min(max((len(nodes) - 2) / 8.0, 0.0), 1.0)
    prior_times = tuple(
        row.available_at
        for row in observations
        if row.available_at < decision_time
        and ({_actor(row.actor_id), _counterparty(row.counterparty_id)} & nodes)
    )
    if prior_times:
        age_minutes = max(
            0.0, (decision_time - max(prior_times)).total_seconds() / 60.0
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


def _split_nodes(nodes: set[str]) -> tuple[set[str], set[str]]:
    return (
        {value.removeprefix("actor:") for value in nodes if value.startswith("actor:")},
        {
            value.removeprefix("counterparty:")
            for value in nodes
            if value.startswith("counterparty:")
        },
    )


def _add_nodes(state: _CaseState, nodes: set[str]) -> None:
    actors, counterparties = _split_nodes(nodes)
    state.actor_ids.update(actors)
    state.counterparty_ids.update(counterparties)
