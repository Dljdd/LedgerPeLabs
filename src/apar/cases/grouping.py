"""Deterministic, truth-blind investigation-case grouping."""

from __future__ import annotations

import hashlib
import math
from bisect import bisect_right
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from enum import StrEnum
from typing import Any, Literal, cast

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
_MAX_TOPOLOGY_INTERVAL_REFS = 4_000_000
_MAX_TOPOLOGY_MEMBERSHIP_ENTRIES = 5_000_000
_BITSET_TOPOLOGY_LIMIT = 4_096
_MAX_PRIORITY_DECIMAL_ADJUSTED = 308


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
            any(
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


@dataclass(frozen=True, slots=True)
class _CaseView:
    case_id: str
    opened_at: datetime
    actor_ids: frozenset[str]
    counterparty_ids: frozenset[str]


@dataclass(frozen=True, slots=True)
class _MemberTree:
    positions: tuple[int, ...] = ()
    left: _MemberTree | None = None
    right: _MemberTree | None = None


@dataclass(frozen=True, slots=True)
class _CausalRow:
    event_id: str
    actor_id: str
    counterparty_id: str
    actor_node: str
    counterparty_node: str
    amount: Decimal
    available_at: datetime


@dataclass(frozen=True, slots=True)
class _ComponentSnapshot:
    root: str
    members: int | tuple[int, ...]
    entity_count: int
    total_value: Decimal
    first_event_id: str | None
    latest_evidence: tuple[datetime, str] | None


@dataclass(frozen=True, slots=True)
class _CausalAlert:
    decision_position: int
    row_position: int
    event_id: str
    score: float
    base_action: Action
    evidence_source_ids: tuple[str, ...]
    components: tuple[_ComponentSnapshot, ...]


@dataclass(frozen=True, slots=True)
class _CausalBatch:
    decision_at: datetime
    alerts: tuple[_CausalAlert, ...]


@dataclass(frozen=True, slots=True)
class _CausalTopology:
    rows: tuple[_CausalRow, ...]
    batches: tuple[_CausalBatch, ...]
    decision_count: int
    uses_bitsets: bool


class _RollbackTopologyDsu:
    """Rollback union-find with immutable causal component aggregates."""

    __slots__ = (
        "_first",
        "_history",
        "_latest",
        "_members",
        "_minimum",
        "_nodes",
        "_parent",
        "_size",
        "_total",
    )

    def __init__(
        self,
        nodes: tuple[str, ...],
        positions_by_node: dict[str, tuple[int, ...]],
        *,
        uses_bitsets: bool,
    ) -> None:
        self._nodes = nodes
        self._parent = list(range(len(nodes)))
        self._size = [1] * len(nodes)
        self._minimum = list(nodes)
        self._total = [Decimal(0)] * len(nodes)
        self._first: list[str | None] = [None] * len(nodes)
        self._latest: list[tuple[datetime, str] | None] = [None] * len(nodes)
        if uses_bitsets:
            self._members: list[int | _MemberTree] = [
                sum(1 << position for position in positions_by_node.get(node, ()))
                for node in nodes
            ]
        else:
            self._members = [
                _MemberTree(positions=positions_by_node.get(node, ())) for node in nodes
            ]
        self._history: list[tuple[object, ...]] = []

    def mark(self) -> int:
        return len(self._history)

    def find(self, node: int) -> int:
        while self._parent[node] != node:
            node = self._parent[node]
        return node

    def add_edge(self, left: int, right: int, row: _CausalRow) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            self._history.append(
                (
                    "edge",
                    left_root,
                    self._total[left_root],
                    self._first[left_root],
                    self._latest[left_root],
                )
            )
            self._add_evidence(left_root, row)
            return
        if (self._size[left_root], self._minimum[left_root]) < (
            self._size[right_root],
            self._minimum[right_root],
        ):
            left_root, right_root = right_root, left_root
        self._history.append(
            (
                "union",
                left_root,
                right_root,
                self._size[left_root],
                self._minimum[left_root],
                self._total[left_root],
                self._first[left_root],
                self._latest[left_root],
                self._members[left_root],
            )
        )
        self._parent[right_root] = left_root
        self._size[left_root] += self._size[right_root]
        self._minimum[left_root] = min(
            self._minimum[left_root], self._minimum[right_root]
        )
        self._total[left_root] += self._total[right_root]
        source_first = self._first[right_root]
        target_first = self._first[left_root]
        if target_first is None or (
            source_first is not None and source_first < target_first
        ):
            self._first[left_root] = source_first
        source_latest = self._latest[right_root]
        target_latest = self._latest[left_root]
        if source_latest is not None and (
            target_latest is None or source_latest > target_latest
        ):
            self._latest[left_root] = source_latest
        left_members = self._members[left_root]
        right_members = self._members[right_root]
        if type(left_members) is int and type(right_members) is int:
            self._members[left_root] = left_members | right_members
        else:
            assert type(left_members) is _MemberTree
            assert type(right_members) is _MemberTree
            self._members[left_root] = _MemberTree(
                left=left_members,
                right=right_members,
            )
        self._add_evidence(left_root, row)

    def _add_evidence(self, root: int, row: _CausalRow) -> None:
        self._total[root] += row.amount
        current_first = self._first[root]
        if current_first is None or row.event_id < current_first:
            self._first[root] = row.event_id
        evidence = (row.available_at, row.event_id)
        current_latest = self._latest[root]
        if current_latest is None or evidence > current_latest:
            self._latest[root] = evidence

    def rollback(self, mark: int) -> None:
        while len(self._history) > mark:
            record = self._history.pop()
            if record[0] == "edge":
                _, root, total, first, latest = record
                root_index = cast(int, root)
                self._total[root_index] = cast(Decimal, total)
                self._first[root_index] = cast(str | None, first)
                self._latest[root_index] = cast(
                    tuple[datetime, str] | None, latest
                )
                continue
            (
                _,
                left_root,
                right_root,
                size,
                minimum,
                total,
                first,
                latest,
                members,
            ) = record
            left_index = cast(int, left_root)
            right_index = cast(int, right_root)
            self._parent[right_index] = right_index
            self._size[left_index] = cast(int, size)
            self._minimum[left_index] = cast(str, minimum)
            self._total[left_index] = cast(Decimal, total)
            self._first[left_index] = cast(str | None, first)
            self._latest[left_index] = cast(tuple[datetime, str] | None, latest)
            self._members[left_index] = cast(int | _MemberTree, members)

    def snapshot(
        self,
        node: int,
        *,
        prior_limit: int,
        membership_budget: list[int],
    ) -> _ComponentSnapshot:
        root = self.find(node)
        raw_members = self._members[root]
        if isinstance(raw_members, int):
            masked_members = raw_members & ((1 << prior_limit) - 1)
            members: int | tuple[int, ...] = masked_members
            membership_budget[0] += masked_members.bit_count()
        else:
            positions: set[int] = set()
            stack: list[_MemberTree] = [raw_members]
            while stack:
                tree = stack.pop()
                positions.update(
                    position for position in tree.positions if position < prior_limit
                )
                if tree.left is not None:
                    stack.append(tree.left)
                if tree.right is not None:
                    stack.append(tree.right)
            members = tuple(sorted(positions))
            membership_budget[0] += len(members)
        if membership_budget[0] > _MAX_TOPOLOGY_MEMBERSHIP_ENTRIES:
            raise CaseContractError(
                "causal topology exceeds the frozen membership resource cap"
            )
        return _ComponentSnapshot(
            root=self._minimum[root],
            members=members,
            entity_count=self._size[root],
            total_value=self._total[root],
            first_event_id=self._first[root],
            latest_evidence=self._latest[root],
        )


def _build_causal_topology(
    observations: tuple[ObservedEvent, ...],
    decisions: tuple[DefenseDecision, ...],
) -> _CausalTopology:
    available_rows = tuple(
        sorted(observations, key=lambda row: (row.available_at, row.event_id))
    )
    rows = tuple(
        _CausalRow(
            event_id=row.event_id,
            actor_id=row.actor_id,
            counterparty_id=row.counterparty_id,
            actor_node=_actor(row.actor_id),
            counterparty_node=_counterparty(row.counterparty_id),
            amount=row.amount,
            available_at=row.available_at,
        )
        for row in available_rows
    )
    row_position_by_id = {row.event_id: index for index, row in enumerate(rows)}
    observation_by_id = {row.event_id: row for row in observations}
    decision_by_id = {row.event_id: row for row in decisions}
    ordered_decisions = tuple(
        sorted(
            decisions,
            key=lambda row: (
                cast(datetime, observation_by_id[row.event_id].decision_at),
                row.event_id,
            ),
        )
    )
    raw_batches: list[tuple[datetime, tuple[int, ...]]] = []
    decision_batch_by_id: dict[str, int] = {}
    position = 0
    while position < len(ordered_decisions):
        decision_at = cast(
            datetime, observation_by_id[ordered_decisions[position].event_id].decision_at
        )
        end = position + 1
        while (
            end < len(ordered_decisions)
            and observation_by_id[ordered_decisions[end].event_id].decision_at
            == decision_at
        ):
            end += 1
        positions = tuple(range(position, end))
        batch_index = len(raw_batches)
        for decision_position in positions:
            decision_batch_by_id[ordered_decisions[decision_position].event_id] = batch_index
        raw_batches.append((decision_at, positions))
        position = end
    if not raw_batches:
        return _CausalTopology(
            rows=rows,
            batches=(),
            decision_count=0,
            uses_bitsets=True,
        )

    nodes = tuple(
        sorted(
            {
                node
                for row in rows
                for node in (row.actor_node, row.counterparty_node)
            }
        )
    )
    node_position = {node: index for index, node in enumerate(nodes)}
    positions_by_node_lists: dict[str, list[int]] = {node: [] for node in nodes}
    for decision_position, decision_row in enumerate(ordered_decisions):
        row = rows[row_position_by_id[decision_row.event_id]]
        positions_by_node_lists[row.actor_node].append(decision_position)
        positions_by_node_lists[row.counterparty_node].append(decision_position)
    positions_by_node = {
        node: tuple(sorted(set(positions)))
        for node, positions in positions_by_node_lists.items()
    }
    uses_bitsets = len(ordered_decisions) <= _BITSET_TOPOLOGY_LIMIT
    dsu = _RollbackTopologyDsu(
        nodes,
        positions_by_node,
        uses_bitsets=uses_bitsets,
    )
    batch_times = tuple(item[0] for item in raw_batches)
    leaf_count = 1
    while leaf_count < len(raw_batches):
        leaf_count *= 2
    intervals: list[list[int]] = [[] for _ in range(leaf_count * 2)]
    interval_refs = 0

    def add_interval(start: int, stop: int, row_position: int) -> None:
        nonlocal interval_refs
        if start >= stop:
            return
        left = start + leaf_count
        right = stop + leaf_count
        while left < right:
            if left % 2:
                intervals[left].append(row_position)
                interval_refs += 1
                left += 1
            if right % 2:
                right -= 1
                intervals[right].append(row_position)
                interval_refs += 1
            if interval_refs > _MAX_TOPOLOGY_INTERVAL_REFS:
                raise CaseContractError(
                    "causal topology exceeds the frozen interval resource cap"
                )
            left //= 2
            right //= 2

    for row_position, row in enumerate(rows):
        visible_from = bisect_right(batch_times, row.available_at)
        own_batch = decision_batch_by_id.get(row.event_id)
        if own_batch is None:
            add_interval(visible_from, len(raw_batches), row_position)
        else:
            add_interval(visible_from, own_batch, row_position)
            add_interval(max(visible_from, own_batch + 1), len(raw_batches), row_position)

    built_batches: list[_CausalBatch | None] = [None] * len(raw_batches)
    membership_budget = [0]

    def visit(tree_position: int, start: int, stop: int) -> None:
        mark = dsu.mark()
        for row_position in intervals[tree_position]:
            row = rows[row_position]
            dsu.add_edge(
                node_position[row.actor_node],
                node_position[row.counterparty_node],
                row,
            )
        if stop - start == 1:
            if start < len(raw_batches):
                decision_at, decision_positions = raw_batches[start]
                component_cache: dict[int, _ComponentSnapshot] = {}
                alerts: list[_CausalAlert] = []
                prior_limit = decision_positions[0]
                for decision_position in decision_positions:
                    decision_row = ordered_decisions[decision_position]
                    row_position = row_position_by_id[decision_row.event_id]
                    row = rows[row_position]
                    components: list[_ComponentSnapshot] = []
                    for node in (row.actor_node, row.counterparty_node):
                        root = dsu.find(node_position[node])
                        component = component_cache.get(root)
                        if component is None:
                            component = dsu.snapshot(
                                root,
                                prior_limit=prior_limit,
                                membership_budget=membership_budget,
                            )
                            component_cache[root] = component
                        if component not in components:
                            components.append(component)
                    alerts.append(
                        _CausalAlert(
                            decision_position=decision_position,
                            row_position=row_position,
                            event_id=decision_row.event_id,
                            score=decision_row.score,
                            base_action=decision_by_id[decision_row.event_id].action,
                            evidence_source_ids=decision_row.evidence_source_ids,
                            components=tuple(sorted(components, key=lambda item: item.root)),
                        )
                    )
                built_batches[start] = _CausalBatch(
                    decision_at=decision_at,
                    alerts=tuple(alerts),
                )
        else:
            middle = (start + stop) // 2
            visit(tree_position * 2, start, middle)
            visit(tree_position * 2 + 1, middle, stop)
        dsu.rollback(mark)

    visit(1, 0, leaf_count)
    return _CausalTopology(
        rows=rows,
        batches=tuple(cast(_CausalBatch, batch) for batch in built_batches),
        decision_count=len(ordered_decisions),
        uses_bitsets=uses_bitsets,
    )


class _ReviewCaseCounterInput(ExternalContract):
    """Transient validator for the immutable callback constructor."""

    observations: tuple[ObservedEvent, ...]
    decisions: tuple[DefenseDecision, ...]
    as_of: datetime

    @field_validator("observations", mode="before")
    @classmethod
    def observations_are_exact(cls, value: object) -> object:
        if type(value) is tuple and len(value) > _MAX_GROUPING_ROWS:
            raise ValueError("review-case counter exceeds the grouping row resource cap")
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
    def rows_are_exact_and_canonical(self) -> _ReviewCaseCounterInput:
        _validated_rows(self.observations, self.decisions, self.as_of)
        expected = tuple(
            sorted(
                self.observations,
                key=lambda row: (row.available_at, row.event_id),
            )
        )
        if self.observations != expected:
            raise ValueError("review-case observations must use canonical availability order")
        observation_by_id = {row.event_id: row for row in self.observations}
        expected_decisions = tuple(
            sorted(
                self.decisions,
                key=lambda row: (
                    cast(datetime, observation_by_id[row.event_id].decision_at),
                    row.event_id,
                ),
            )
        )
        if self.decisions != expected_decisions:
            raise ValueError("review-case decisions must use canonical decision order")
        if len(self.observations) > _MAX_GROUPING_ROWS:
            raise ValueError("review-case counter exceeds the grouping row resource cap")
        return self

        return self


class ReviewCaseCounter(tuple[_CausalTopology]):
    """Intrinsically immutable adapter from action vectors to causal case counts."""

    __slots__ = ()

    def __new__(
        cls,
        *,
        observations: tuple[ObservedEvent, ...],
        decisions: tuple[DefenseDecision, ...],
        as_of: datetime,
    ) -> ReviewCaseCounter:
        binding = _ReviewCaseCounterInput(
            observations=observations,
            decisions=decisions,
            as_of=as_of,
        )
        topology = _build_causal_topology(binding.observations, binding.decisions)
        return tuple.__new__(cls, (topology,))

    @property
    def _topology(self) -> _CausalTopology:
        return tuple.__getitem__(self, 0)

    def __setattr__(self, name: str, value: Any) -> None:
        del name, value
        raise TypeError("review-case counter is immutable")

    def __delattr__(self, name: str) -> None:
        del name
        raise TypeError("review-case counter is immutable")

    def __reduce__(
        self,
    ) -> tuple[
        Callable[[_CausalTopology], ReviewCaseCounter],
        tuple[_CausalTopology],
    ]:
        return _restore_review_case_counter, (self._topology,)

    def __call__(self, actions: NDArray[np.object_]) -> int:
        """Count grouped interventions without observing labels or action severity."""
        try:
            if type(actions) is not np.ndarray:
                raise TypeError("actions must be an exact numpy.ndarray")
            if actions.ndim != 1 or actions.dtype != np.dtype(object):
                raise TypeError("actions must be a one-dimensional array with dtype object")
            if len(actions) != self._topology.decision_count:
                raise ValueError("action vector length must equal bound decision rows")
            intervention_mask = bytearray(len(actions))
            for action in actions:
                if type(action) is not Action:
                    raise TypeError("every action must be an exact Action")
            for index, action in enumerate(actions):
                intervention_mask[index] = int(action is not Action.APPROVE)
            return cast(
                int,
                _run_case_engine(
                    self._topology,
                    bytes(intervention_mask),
                    materialize=False,
                ),
            )
        except (ArithmeticError, MemoryError, OverflowError) as error:
            raise CaseContractError("case counting exceeded frozen resource bounds") from error


def _restore_review_case_counter(topology: _CausalTopology) -> ReviewCaseCounter:
    if type(topology) is not _CausalTopology:
        raise CaseContractError("review-case topology restore requires an exact snapshot")
    return tuple.__new__(ReviewCaseCounter, (topology,))


def bind_review_case_counter(
    observations: Sequence[ObservedEvent],
    decisions: Sequence[DefenseDecision],
    *,
    as_of: datetime,
) -> ReviewCaseCounter:
    """Bind exact canonical decision rows for threshold selection."""
    try:
        if type(observations) is tuple and len(observations) > _MAX_GROUPING_ROWS:
            raise CaseContractError(
                "review-case counter exceeds the grouping row resource cap"
            )
        observation_rows, decision_rows = _validated_rows(observations, decisions, as_of)
        expected_observations = tuple(
            sorted(
                observation_rows,
                key=lambda row: (row.available_at, row.event_id),
            )
        )
        if observation_rows != expected_observations:
            raise CaseContractError(
                "review-case observations must use canonical availability order"
            )
        observation_by_id = {row.event_id: row for row in observation_rows}
        expected_decisions = tuple(
            sorted(
                decision_rows,
                key=lambda row: (
                    cast(datetime, observation_by_id[row.event_id].decision_at),
                    row.event_id,
                ),
            )
        )
        if decision_rows != expected_decisions:
            raise CaseContractError("review-case decisions must use canonical decision order")
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

        current_expected_value = score * current transaction amount

        100 * (0.45 * score
             + 0.30 * current_expected_value / (current_expected_value + 1000)
             + 0.15 * min(max(entity_count - 2, 0) / 8, 1)
             + 0.10 * 1 / (1 + minutes_since_latest_graph_evidence))

    Current-batch rows are admitted only after every decision at that exact time,
    so self/peer amounts and edges cannot affect historical evidence, grouping,
    recency, or entity coverage. The explicit current expected-value term is the
    sole use of the current amount. The recency term is zero when there is no prior
    graph evidence. Later alerts may extend or merge a case, but the canonical
    first-evidence ID and priority of the earliest case remain unchanged.
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
    if (
        len(observation_rows) > _MAX_GROUPING_ROWS
        or len(decision_rows) > _MAX_GROUPING_ROWS
    ):
        raise CaseContractError("case grouping row count exceeds frozen resource cap")
    if any(type(row) is not ObservedEvent for row in observation_rows):
        raise TypeError("observations must contain exact ObservedEvent rows")
    if any(type(row) is not DefenseDecision for row in decision_rows):
        raise TypeError("decisions must contain exact DefenseDecision rows")
    raw_observation_ids = tuple(row.event_id for row in observation_rows)
    raw_decision_ids = tuple(row.event_id for row in decision_rows)
    if all(type(item) is str for item in raw_observation_ids) and len(
        set(raw_observation_ids)
    ) != len(raw_observation_ids):
        raise CaseContractError("duplicate observation event_id")
    if all(type(item) is str for item in raw_decision_ids) and len(
        set(raw_decision_ids)
    ) != len(raw_decision_ids):
        raise CaseContractError("duplicate decision event_id")
    revalidated_observations: list[ObservedEvent] = []
    for observation_row in observation_rows:
        validated_observation = _revalidate_observation(observation_row)
        _validate_observation(validated_observation, as_of)
        revalidated_observations.append(validated_observation)
    revalidated_decisions: list[DefenseDecision] = []
    for decision_row in decision_rows:
        validated_decision = _revalidate_decision(decision_row)
        if type(validated_decision.action) is not Action:
            raise CaseContractError("decision action must be an exact Action")
        revalidated_decisions.append(validated_decision)
    canonical_observations = tuple(revalidated_observations)
    canonical_decisions = tuple(revalidated_decisions)
    decision_ids = tuple(row.event_id for row in canonical_decisions)
    observation_by_id = {row.event_id: row for row in canonical_observations}
    decision_point_ids = {
        row.event_id for row in canonical_observations if row.is_decision_point
    }
    if set(decision_ids) != decision_point_ids:
        raise CaseContractError(
            "decision event IDs must match decision-point observations bijectively"
        )
    if any(observation_by_id[event_id].decision_at is None for event_id in decision_ids):
        raise CaseContractError("decision observations require an exact decision_at")
    return canonical_observations, canonical_decisions


def _validate_observation(row: ObservedEvent, as_of: datetime) -> None:
    for timestamp_label, timestamp_value in (
        ("event_time", row.event_time),
        ("available_at", row.available_at),
    ):
        _utc(timestamp_value, label=f"observation {timestamp_label}")
    if row.is_decision_point:
        if row.decision_at is None:
            raise CaseContractError("decision-point observations require decision_at")
        _utc(row.decision_at, label="observation decision_at")
    elif row.decision_at is not None:
        raise CaseContractError("nondecision observations must not declare decision_at")
    if row.event_time > row.available_at:
        raise CaseContractError("observation event_time must not exceed available_at")
    if row.decision_at is not None and row.available_at > row.decision_at:
        raise CaseContractError("observation available_at must not exceed decision_at")
    if (
        row.event_time > as_of
        or row.available_at > as_of
        or (row.decision_at is not None and row.decision_at > as_of)
    ):
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
class _EngineCase:
    case_id: str
    opened_at: datetime
    members: int | set[int]
    state: _CaseState | None


@dataclass(frozen=True, slots=True)
class _EnginePlan:
    alert: _CausalAlert
    action: Action
    matching_case_ids: tuple[str, ...]
    evidence: CaseAlertEvidence | None
    priority: float | None


def _run_case_engine(
    topology: _CausalTopology,
    intervention_mask: bytes,
    *,
    materialize: bool,
    actions_by_event_id: dict[str, Action] | None = None,
) -> int | tuple[InvestigationCase, ...]:
    """Run the one shared exact case-state machine over frozen causal topology."""
    if len(intervention_mask) != topology.decision_count:
        raise CaseContractError("intervention mask must align with causal topology")
    assignments: list[str | None] = [None] * topology.decision_count
    active: dict[str, _EngineCase] = {}
    aliases: dict[str, str] = {}

    def resolve(case_id: str) -> str:
        trail: list[str] = []
        while case_id in aliases:
            trail.append(case_id)
            case_id = aliases[case_id]
        for stale in trail:
            aliases[stale] = case_id
        return case_id

    def matching_ids(component: _ComponentSnapshot) -> tuple[str, ...]:
        matches: set[str] = set()
        members = component.members
        if isinstance(members, int):
            if members.bit_count() <= len(active):
                remaining = members
                while remaining:
                    least = remaining & -remaining
                    position = least.bit_length() - 1
                    assigned = assignments[position]
                    if assigned is not None:
                        active_id = resolve(assigned)
                        if active_id in active:
                            matches.add(active_id)
                    remaining ^= least
            else:
                for case_id, engine_case in active.items():
                    case_members = engine_case.members
                    assert type(case_members) is int
                    if case_members & members:
                        matches.add(case_id)
        elif len(members) <= len(active):
            for position in members:
                assigned = assignments[position]
                if assigned is not None:
                    active_id = resolve(assigned)
                    if active_id in active:
                        matches.add(active_id)
        else:
            member_set = set(members)
            for case_id, engine_case in active.items():
                case_members = engine_case.members
                assert type(case_members) is set
                if not case_members.isdisjoint(member_set):
                    matches.add(case_id)
        return tuple(
            sorted(matches, key=lambda item: (active[item].opened_at, item))
        )

    for batch in topology.batches:
        plans: list[_EnginePlan] = []
        for alert in batch.alerts:
            if intervention_mask[alert.decision_position] == 0:
                continue
            action = (
                alert.base_action
                if actions_by_event_id is None
                else actions_by_event_id[alert.event_id]
            )
            candidates: list[
                tuple[
                    tuple[datetime, str, str],
                    _ComponentSnapshot,
                    tuple[str, ...],
                ]
            ] = []
            for component in alert.components:
                matches = matching_ids(component)
                if matches:
                    first = matches[0]
                    candidates.append(
                        (
                            (active[first].opened_at, first, component.root),
                            component,
                            matches,
                        )
                    )
            selected = min(candidates, key=lambda item: item[0]) if candidates else None
            matching = () if selected is None else selected[2]
            evidence: CaseAlertEvidence | None = None
            priority: float | None = None
            if materialize:
                row = topology.rows[alert.row_position]
                visible_value = sum(
                    (component.total_value for component in alert.components),
                    start=Decimal(0),
                )
                evidence_keys = tuple(
                    component.latest_evidence
                    for component in alert.components
                    if component.latest_evidence is not None
                )
                latest_evidence_at = (
                    max(evidence_keys)[0]
                    if evidence_keys
                    else None
                )
                sources = {
                    source
                    for component in alert.components
                    for source in (
                        component.first_event_id,
                        component.latest_evidence[1]
                        if component.latest_evidence is not None
                        else None,
                    )
                    if source is not None
                }
                matching_views = tuple(
                    _CaseView(
                        case_id=case_id,
                        opened_at=active[case_id].opened_at,
                        actor_ids=frozenset(
                            cast(_CaseState, active[case_id].state).actor_ids
                        ),
                        counterparty_ids=frozenset(
                            cast(_CaseState, active[case_id].state).counterparty_ids
                        ),
                    )
                    for case_id in matching
                )
                evidence = CaseAlertEvidence(
                    event_id=alert.event_id,
                    decision_at=batch.decision_at,
                    actor_id=row.actor_id,
                    counterparty_id=row.counterparty_id,
                    motif=_motif_for_alert_from_ids(
                        row.actor_id,
                        row.counterparty_id,
                        matching_views,
                    ),
                    visible_value_before_alert=visible_value,
                    latest_graph_evidence_at=latest_evidence_at,
                    score=alert.score,
                    action=action,
                    evidence_source_ids=tuple(sorted(sources)),
                )
                if not matching:
                    priority = _priority(
                        score=alert.score,
                        current_amount=row.amount,
                        entity_count=sum(
                            component.entity_count for component in alert.components
                        ),
                        latest_graph_evidence_at=latest_evidence_at,
                        decision_time=batch.decision_at,
                    )
            plans.append(
                _EnginePlan(
                    alert=alert,
                    action=action,
                    matching_case_ids=matching,
                    evidence=evidence,
                    priority=priority,
                )
            )

        for plan in plans:
            active_matching = tuple(
                sorted(
                    {
                        resolve(case_id)
                        for case_id in plan.matching_case_ids
                        if resolve(case_id) in active
                    },
                    key=lambda item: (active[item].opened_at, item),
                )
            )
            if active_matching:
                case_id = active_matching[0]
                engine_case = active[case_id]
                for merged_id in active_matching[1:]:
                    merged = active.pop(merged_id)
                    aliases[merged_id] = case_id
                    if isinstance(engine_case.members, int):
                        assert isinstance(merged.members, int)
                        engine_case.members |= merged.members
                    else:
                        assert isinstance(merged.members, set)
                        engine_case.members.update(merged.members)
                    if materialize:
                        target = cast(_CaseState, engine_case.state)
                        source = cast(_CaseState, merged.state)
                        target.event_ids.update(source.event_ids)
                        target.actor_ids.update(source.actor_ids)
                        target.counterparty_ids.update(source.counterparty_ids)
                        target.alert_evidence.extend(source.alert_evidence)
            else:
                case_id = _case_id((plan.alert.event_id,))
                state: _CaseState | None = None
                if materialize:
                    assert plan.priority is not None
                    state = _CaseState(
                        case_id=case_id,
                        opened_at=batch.decision_at,
                        event_ids=set(),
                        actor_ids=set(),
                        counterparty_ids=set(),
                        priority=plan.priority,
                        first_evidence_ids=(plan.alert.event_id,),
                        alert_evidence=[],
                    )
                engine_case = _EngineCase(
                    case_id=case_id,
                    opened_at=batch.decision_at,
                    members=0 if topology.uses_bitsets else set(),
                    state=state,
                )
                active[case_id] = engine_case
            decision_position = plan.alert.decision_position
            assignments[decision_position] = case_id
            if isinstance(engine_case.members, int):
                engine_case.members |= 1 << decision_position
            else:
                engine_case.members.add(decision_position)
            if materialize:
                row = topology.rows[plan.alert.row_position]
                state = cast(_CaseState, engine_case.state)
                state.event_ids.add(plan.alert.event_id)
                state.actor_ids.add(row.actor_id)
                state.counterparty_ids.add(row.counterparty_id)
                assert plan.evidence is not None
                state.alert_evidence.append(plan.evidence)

    if not materialize:
        return len(active)
    states = tuple(
        cast(_CaseState, engine_case.state) for engine_case in active.values()
    )
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
                sorted(
                    state.alert_evidence,
                    key=lambda row: (row.decision_at, row.event_id),
                )
            ),
        )
        for state in sorted(states, key=lambda item: (item.opened_at, item.case_id))
    )


def _group_validated(
    observations: tuple[ObservedEvent, ...],
    decisions: tuple[DefenseDecision, ...],
    *,
    actions: tuple[Action, ...] | None = None,
) -> tuple[InvestigationCase, ...]:
    if actions is not None and len(actions) != len(decisions):
        raise CaseContractError("candidate actions must align with decisions")
    actions_by_event_id = {
        decision.event_id: decision.action if actions is None else actions[index]
        for index, decision in enumerate(decisions)
    }
    topology = _build_causal_topology(observations, decisions)
    intervention_mask = bytearray(topology.decision_count)
    for batch in topology.batches:
        for alert in batch.alerts:
            action = actions_by_event_id[alert.event_id]
            if type(action) is not Action:
                raise CaseContractError("candidate action must be an exact Action")
            intervention_mask[alert.decision_position] = int(action is not Action.APPROVE)
    return cast(
        tuple[InvestigationCase, ...],
        _run_case_engine(
            topology,
            bytes(intervention_mask),
            materialize=True,
            actions_by_event_id=actions_by_event_id,
        ),
    )


def _motif_for_alert_from_ids(
    actor_id: str,
    counterparty_id: str,
    matching: tuple[_CaseView, ...],
) -> CaseMotif:
    if any(actor_id in state.actor_ids for state in matching):
        return CaseMotif.SHARED_ACTOR
    if any(counterparty_id in state.counterparty_ids for state in matching):
        return CaseMotif.SHARED_COUNTERPARTY
    if matching:
        return CaseMotif.TRANSITIVE
    return CaseMotif.ISOLATED


def _priority(
    *,
    score: float,
    current_amount: Decimal,
    entity_count: int,
    latest_graph_evidence_at: datetime | None,
    decision_time: datetime,
) -> float:
    if current_amount and current_amount.adjusted() > _MAX_PRIORITY_DECIMAL_ADJUSTED:
        raise CaseContractError("case priority input exceeds frozen numeric bounds")
    score_decimal = Decimal(str(score))
    current_expected_value = current_amount * score_decimal
    value_scale = Decimal(str(_VALUE_SCALE))
    value_term = (
        float(current_expected_value / (current_expected_value + value_scale))
        if current_expected_value
        else 0.0
    )
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
        0.45 * score + 0.30 * value_term + 0.15 * coverage + 0.10 * recency
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
