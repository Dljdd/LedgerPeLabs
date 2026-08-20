"""Independent provenance audits and behavioral feature invariants."""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from itertools import groupby
from operator import attrgetter

from apar.defense.contracts import ObservedEvent
from apar.features.builders import FeatureMatrix, build_feature_matrix
from apar.features.catalog import FeatureCatalog, FeatureCatalogError, audit_feature_catalog
from apar.features.state import CausalFeatureState, FeatureVector, feature_catalog_digest

_FORBIDDEN_SEMANTICS = (
    "fraud",
    "illicit",
    "label",
    "target",
    "disposition",
    "chargeback_truth",
    "outcome_truth",
    "campaign",
    "family",
    "threat",
    "scenario",
    "regime",
    "seed",
    "generator",
    "hidden",
    "policy",
    "role",
    "viewpoint",
    "attack",
    "objective",
    "post_decision",
)
_AMOUNT_TOTAL_FEATURES = frozenset(
    {"actor_amount_1h", "actor_amount_24h", "counterparty_amount_24h"}
)
_AMOUNT_SCALE_INVARIANT_FEATURES = frozenset(
    {"actor_amount_zscore_24h", "counterparty_amount_zscore_24h"}
)
_OPTIONAL_REFERENCE_FEATURES = frozenset(
    {"txn_optional_ref_count", "dq_missing_optional_count"}
)
_OPTIONAL_REFERENCE_SLOTS = 7


class FeatureLeakageError(ValueError):
    """A feature matrix violates its semantic, identity, or knowledge-time boundary."""


@dataclass(frozen=True, slots=True)
class FeatureAuditReport:
    """Independent results for the feature catalog and historical provenance gates."""

    catalog_valid: bool
    strictly_past_only: bool
    source_ids_resolve: bool
    feature_order_matches: bool
    forbidden_sources: tuple[str, ...]

    @property
    def passed(self) -> bool:
        return (
            self.catalog_valid
            and self.strictly_past_only
            and self.source_ids_resolve
            and self.feature_order_matches
            and not self.forbidden_sources
        )


def audit_feature_matrix(
    events: Sequence[ObservedEvent],
    matrix: FeatureMatrix,
    catalog: FeatureCatalog,
    *,
    allow_decision_event_subset: bool = False,
) -> FeatureAuditReport:
    """Audit a matrix without trusting feature-state histories or embedded observations."""
    forbidden_sources = _forbidden_catalog_sources(matrix.catalog)
    if forbidden_sources:
        raise FeatureLeakageError(
            "forbidden feature provenance: " + ", ".join(forbidden_sources)
        )

    try:
        audit_feature_catalog(catalog)
        audit_feature_catalog(matrix.catalog)
        expected_digest = feature_catalog_digest(catalog)
    except FeatureCatalogError as error:
        raise FeatureLeakageError(f"feature catalog is invalid: {error}") from error

    availability, observations = _independent_event_indexes(events)
    source_ids_resolve = True
    strictly_past_only = True
    maximums_match = True
    decisions_match = True
    for row in matrix.rows:
        decision = observations.get(row.event_id)
        if (
            decision is None
            or not decision.is_decision_point
            or decision.decision_at is None
            or row.decision_at != decision.decision_at
        ):
            decisions_match = False

        resolved_times: list[datetime] = []
        for source_id in row.source_event_ids:
            source_available_at = availability.get(source_id)
            if source_available_at is None:
                source_ids_resolve = False
                continue
            resolved_times.append(source_available_at)
            if source_available_at >= row.decision_at:
                strictly_past_only = False
        expected_maximum = max(resolved_times) if resolved_times else None
        if source_ids_resolve and row.max_source_available_at != expected_maximum:
            maximums_match = False

    replay_row_ids = frozenset(row.event_id for row in matrix.rows)
    expected_rows = tuple(
        event.event_id
        for event in sorted(
            (
                event
                for event in observations.values()
                if event.is_decision_point and event.event_id in replay_row_ids
            ),
            key=_decision_sort_key,
        )
    )
    expected_matrix_events = tuple(
        sorted(
            (
                observations[event_id]
                for event_id in replay_row_ids
                if event_id in observations
            ),
            key=attrgetter("event_id"),
        )
    )
    full_context_events = tuple(
        sorted(observations.values(), key=attrgetter("event_id"))
    )
    neutralized_matrix_events = tuple(
        sorted(
            (
                event
                if event.event_id in replay_row_ids
                or (not event.is_decision_point and event.decision_at is None)
                else event.model_copy(
                    update={"is_decision_point": False, "decision_at": None}
                )
                for event in matrix.events
            ),
            key=attrgetter("event_id"),
        )
    )
    event_contract_matches = matrix.events == full_context_events or (
        allow_decision_event_subset
        and (
            matrix.events == expected_matrix_events
            or neutralized_matrix_events == full_context_events
        )
    )
    feature_order_matches = (
        matrix.catalog == catalog
        and matrix.catalog_digest == expected_digest
        and event_contract_matches
        and tuple(row.event_id for row in matrix.rows) == expected_rows
        and all(row.catalog_digest == expected_digest for row in matrix.rows)
        and all(tuple(row.values) == catalog.names for row in matrix.rows)
        and decisions_match
    )

    if not source_ids_resolve:
        raise FeatureLeakageError("historical source event IDs do not resolve")
    if not strictly_past_only:
        raise FeatureLeakageError("historical sources must be available strictly before decisions")
    if not maximums_match:
        raise FeatureLeakageError("maximum source availability does not match provenance")
    if not feature_order_matches:
        raise FeatureLeakageError("catalog digest or feature order does not match")

    report = FeatureAuditReport(
        catalog_valid=True,
        strictly_past_only=True,
        source_ids_resolve=True,
        feature_order_matches=True,
        forbidden_sources=(),
    )
    if not report.passed:
        raise FeatureLeakageError("feature audit failed")
    return report


def assert_online_offline_parity(
    events: Sequence[ObservedEvent], catalog: FeatureCatalog
) -> None:
    """Require cohort-partitioned online replay to equal one-shot offline construction."""
    unique_events = _unique_events(events)
    offline = build_feature_matrix(unique_events, catalog)
    audit_feature_matrix(unique_events, offline, catalog)

    online_state = CausalFeatureState(catalog)
    online_rows: list[FeatureVector] = []
    sent_non_decisions: set[str] = set()
    decisions = sorted(
        (event for event in unique_events if event.is_decision_point),
        key=_decision_sort_key,
    )
    for decision_at, cohort_iterator in groupby(decisions, key=attrgetter("decision_at")):
        if decision_at is None:
            raise FeatureLeakageError("decision-point event must declare decision_at")
        cohort = tuple(cohort_iterator)
        newly_available = tuple(
            event
            for event in unique_events
            if not event.is_decision_point
            and event.event_id not in sent_non_decisions
            and event.available_at <= decision_at
        )
        sent_non_decisions.update(event.event_id for event in newly_available)
        process_batch = tuple(
            sorted((*newly_available, *cohort), key=attrgetter("event_id"))
        )
        online_rows.extend(online_state.process(process_batch))

    if tuple(online_rows) != offline.rows:
        raise AssertionError("online and offline feature vectors differ")


def assert_future_append_invariant(
    prefix: Sequence[ObservedEvent],
    future: Sequence[ObservedEvent],
    catalog: FeatureCatalog,
) -> None:
    """Require a strictly future append to preserve prior value and provenance bytes."""
    prefix_matrix = build_feature_matrix(prefix, catalog)
    if not prefix_matrix.rows:
        raise ValueError("future-append prefix must contain a decision")
    boundary = max(row.decision_at for row in prefix_matrix.rows)
    if any(event.available_at <= boundary for event in future):
        raise ValueError("appended events must be strictly future observations")
    if any(
        event.is_decision_point
        and (event.decision_at is None or event.decision_at <= boundary)
        for event in future
    ):
        raise ValueError("appended decisions must be strictly future decisions")

    combined = build_feature_matrix((*prefix, *future), catalog)
    combined_by_id = {row.event_id: row for row in combined.rows}
    for original in prefix_matrix.rows:
        appended = combined_by_id.get(original.event_id)
        if appended is None:
            raise AssertionError("future append removed a prior feature row")
        if _model_value_bytes(original) != _model_value_bytes(appended):
            raise AssertionError("future append changed prior model-value bytes")
        if _historical_provenance_bytes(original) != _historical_provenance_bytes(appended):
            raise AssertionError("future append changed prior historical provenance bytes")


def assert_row_permutation_invariant(
    events: Sequence[ObservedEvent], catalog: FeatureCatalog
) -> None:
    """Require arbitrary observation-row permutation to preserve the matrix."""
    reference = build_feature_matrix(events, catalog)
    permuted = build_feature_matrix(tuple(reversed(events)), catalog)
    if permuted != reference:
        raise AssertionError("observation-row permutation changed the feature matrix")


def assert_equal_time_permutation_invariant(
    events: Sequence[ObservedEvent], catalog: FeatureCatalog
) -> None:
    """Require a complete equal-time decision cohort to be permutation invariant."""
    decision_times = {
        event.decision_at for event in events if event.is_decision_point
    }
    if len(decision_times) != 1 or None in decision_times:
        raise ValueError("events must contain one complete equal-time decision cohort")
    reference = build_feature_matrix(events, catalog)
    permuted = build_feature_matrix(tuple(reversed(events)), catalog)
    if permuted.rows != reference.rows:
        raise AssertionError("equal-time decision permutation changed feature vectors")


def assert_synthetic_id_bijection_invariant(
    events: Sequence[ObservedEvent], catalog: FeatureCatalog
) -> None:
    """Require consistent synthetic ID renaming to preserve values and mapped lineage."""
    unique_events = _unique_events(events)
    event_ids = _bijection((event.event_id for event in unique_events), "renamed-event")
    payment_ids = _bijection((event.payment_id for event in unique_events), "renamed-payment")
    entity_ids = _bijection(
        (
            identifier
            for event in unique_events
            for identifier in (
                event.actor_id,
                event.counterparty_id,
                *event.optional_refs.values(),
            )
        ),
        "renamed-entity",
    )
    renamed = tuple(
        event.model_copy(
            update={
                "event_id": event_ids[event.event_id],
                "payment_id": payment_ids[event.payment_id],
                "actor_id": entity_ids[event.actor_id],
                "counterparty_id": entity_ids[event.counterparty_id],
                "optional_refs": {
                    name: entity_ids[value] for name, value in event.optional_refs.items()
                },
            }
        )
        for event in unique_events
    )
    original_rows = {row.event_id: row for row in build_feature_matrix(unique_events, catalog).rows}
    renamed_rows = {row.event_id: row for row in build_feature_matrix(renamed, catalog).rows}

    for original_id, original in original_rows.items():
        transformed = renamed_rows.get(event_ids[original_id])
        if transformed is None:
            raise AssertionError("ID bijection removed a decision row")
        if transformed.values != original.values:
            raise AssertionError("ID bijection changed invariant model values")
        expected_sources = tuple(sorted(event_ids[source] for source in original.source_event_ids))
        if transformed.source_event_ids != expected_sources:
            raise AssertionError("ID bijection did not rename provenance consistently")
        if transformed.max_source_available_at != original.max_source_available_at:
            raise AssertionError("ID bijection changed provenance availability")


def assert_duplicate_event_ids_invariant(
    events: Sequence[ObservedEvent], catalog: FeatureCatalog
) -> None:
    """Require identical duplicate deliveries to remain idempotent."""
    reference = build_feature_matrix(events, catalog)
    duplicated = build_feature_matrix((*events, *events), catalog)
    if duplicated.rows != reference.rows:
        raise AssertionError("identical duplicate event IDs changed feature vectors")


def assert_missing_optional_references_invariant(
    events: Sequence[ObservedEvent], catalog: FeatureCatalog
) -> None:
    """Require optional-reference removal to affect only its declared quality columns."""
    unique_events = _unique_events(events)
    missing = tuple(event.model_copy(update={"optional_refs": {}}) for event in unique_events)
    baseline_rows = {row.event_id: row for row in build_feature_matrix(unique_events, catalog).rows}
    missing_rows = {row.event_id: row for row in build_feature_matrix(missing, catalog).rows}
    observations = {event.event_id: event for event in unique_events}

    for event_id, baseline in baseline_rows.items():
        transformed = missing_rows[event_id]
        if baseline.source_event_ids != transformed.source_event_ids:
            raise AssertionError("optional-reference removal changed historical provenance")
        for name, value in baseline.values.items():
            transformed_value = transformed.values[name]
            if name not in _OPTIONAL_REFERENCE_FEATURES and transformed_value != value:
                raise AssertionError("optional-reference removal changed an unrelated feature")
        optional_count = len(observations[event_id].optional_refs)
        if baseline.values["txn_optional_ref_count"] != float(optional_count):
            raise AssertionError("optional-reference count does not match the observation")
        if baseline.values["dq_missing_optional_count"] != float(
            max(0, _OPTIONAL_REFERENCE_SLOTS - optional_count)
        ):
            raise AssertionError("optional-reference missingness does not match the observation")
        if transformed.values["txn_optional_ref_count"] != 0.0:
            raise AssertionError("removed optional references still contribute to their count")
        if transformed.values["dq_missing_optional_count"] != float(
            _OPTIONAL_REFERENCE_SLOTS
        ):
            raise AssertionError("removed optional references do not activate missingness")


def assert_economic_scaling_invariant(
    events: Sequence[ObservedEvent],
    catalog: FeatureCatalog,
    *,
    factor: Decimal,
) -> None:
    """Require consistent positive amount scaling to preserve economic relationships."""
    if not factor.is_finite() or factor <= 0:
        raise ValueError("economic scaling factor must be positive and finite")
    unique_events = _unique_events(events)
    scaled_events = tuple(
        event.model_copy(update={"amount": event.amount * factor}) for event in unique_events
    )
    baseline_rows = {row.event_id: row for row in build_feature_matrix(unique_events, catalog).rows}
    scaled_rows = {row.event_id: row for row in build_feature_matrix(scaled_events, catalog).rows}
    observations = {event.event_id: event for event in unique_events}
    multiplier = float(factor)

    for event_id, baseline in baseline_rows.items():
        scaled = scaled_rows[event_id]
        if _historical_provenance_bytes(baseline) != _historical_provenance_bytes(scaled):
            raise AssertionError("economic scaling changed historical provenance")
        for name, value in baseline.values.items():
            scaled_value = scaled.values[name]
            if name == "txn_log_amount":
                expected = math.log1p(float(observations[event_id].amount * factor))
                _require_close(scaled_value, expected, message="current log amount did not scale")
            elif name in _AMOUNT_TOTAL_FEATURES:
                _require_close(
                    scaled_value,
                    value * multiplier,
                    message=f"amount total {name} did not scale",
                )
            elif name in _AMOUNT_SCALE_INVARIANT_FEATURES:
                _require_close(
                    scaled_value,
                    value,
                    message=f"standardized amount {name} changed under scaling",
                )
            elif scaled_value != value:
                raise AssertionError(f"economic scaling changed unrelated feature {name}")


def assert_checkpoint_restart_invariant(
    prefix: Sequence[ObservedEvent],
    continuation: Sequence[ObservedEvent],
    catalog: FeatureCatalog,
) -> None:
    """Require checkpoint restore to reproduce uninterrupted cohort-boundary replay."""
    prefix_decisions = tuple(event for event in prefix if event.is_decision_point)
    continuation_decisions = tuple(event for event in continuation if event.is_decision_point)
    if prefix_decisions and continuation_decisions:
        prefix_boundary = max(_required_decision_at(event) for event in prefix_decisions)
        continuation_boundary = min(
            _required_decision_at(event) for event in continuation_decisions
        )
        if continuation_boundary <= prefix_boundary:
            raise ValueError("checkpoint split must preserve complete decision-time cohorts")

    uninterrupted = CausalFeatureState(catalog)
    before = uninterrupted.process(prefix)
    payload = uninterrupted.checkpoint()
    restored = CausalFeatureState.restore(payload, catalog)
    uninterrupted_after = uninterrupted.process(continuation)
    restored_after = restored.process(continuation)
    if restored_after != uninterrupted_after:
        raise AssertionError("checkpoint restart changed continuation vectors")

    offline = build_feature_matrix((*prefix, *continuation), catalog)
    if (*before, *restored_after) != offline.rows:
        raise AssertionError("checkpoint restart differs from complete-cohort offline replay")


def _independent_event_indexes(
    events: Sequence[ObservedEvent],
) -> tuple[dict[str, datetime], dict[str, ObservedEvent]]:
    observations: dict[str, ObservedEvent] = {}
    for event in events:
        if type(event) is not ObservedEvent:
            raise TypeError("events must contain exact ObservedEvent instances")
        previous = observations.get(event.event_id)
        if previous is not None and previous != event:
            raise FeatureLeakageError("duplicate event ID has conflicting observations")
        observations[event.event_id] = event
    return (
        {event_id: event.available_at for event_id, event in observations.items()},
        observations,
    )


def _unique_events(events: Sequence[ObservedEvent]) -> tuple[ObservedEvent, ...]:
    _, observations = _independent_event_indexes(events)
    return tuple(observations.values())


def _forbidden_catalog_sources(catalog: FeatureCatalog) -> tuple[str, ...]:
    forbidden = {
        source_path
        for definition in catalog.features
        for source_path in definition.source_paths
        if any(term in source_path.lower() for term in _FORBIDDEN_SEMANTICS)
    }
    return tuple(sorted(forbidden))


def _decision_sort_key(event: ObservedEvent) -> tuple[datetime, str]:
    return _required_decision_at(event), event.event_id


def _required_decision_at(event: ObservedEvent) -> datetime:
    if event.decision_at is None:
        raise FeatureLeakageError("decision-point event must declare decision_at")
    return event.decision_at


def _bijection(values: Iterable[str], prefix: str) -> Mapping[str, str]:
    return {
        value: f"{prefix}-{index:04d}"
        for index, value in enumerate(sorted(set(values)), start=1)
    }


def _model_value_bytes(row: FeatureVector) -> bytes:
    return json.dumps(row.values, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _historical_provenance_bytes(row: FeatureVector) -> bytes:
    document = {
        "max_source_available_at": (
            None
            if row.max_source_available_at is None
            else row.max_source_available_at.isoformat()
        ),
        "source_event_ids": row.source_event_ids,
    }
    return json.dumps(document, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _require_close(actual: float, expected: float, *, message: str) -> None:
    if not math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12):
        raise AssertionError(message)
