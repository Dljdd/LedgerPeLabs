"""Causal Sentinel feature projection for Defend v5."""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from apar.evaluation.v5_population import V5DecisionRow

_FORBIDDEN_FIELDS = frozenset(
    {"family", "campaign_id", "scenario_id", "seed", "split", "is_fraud",
     "generator", "label", "outcome"}
)


def _set_if_in_catalog(
    values: dict[str, float],
    catalog: SentinelFeatureCatalog,
    name: str,
    value: float,
) -> None:
    if name in catalog.feature_names:
        values[name] = value


class SentinelFeatureCatalog(BaseModel):
    model_config = ConfigDict(frozen=True)

    feature_names: tuple[str, ...]
    catalog_sha256: str

    @classmethod
    def from_config(cls, path: Path) -> SentinelFeatureCatalog:
        document = json.loads(path.read_bytes())
        names = tuple(f["name"] for f in document.get("features", []))
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        return cls(feature_names=names, catalog_sha256=digest)

    @classmethod
    def default(cls) -> SentinelFeatureCatalog:
        root = Path(__file__).resolve().parents[3]
        return cls.from_config(root / "config/defense/feature-catalog-v5.json")

    @model_validator(mode="after")
    def no_forbidden_fields(self) -> Self:
        leaks = _FORBIDDEN_FIELDS & set(self.feature_names)
        if leaks:
            raise ValueError(f"catalog contains forbidden predictive fields: {leaks}")
        return self


class SentinelFeatureBatch(BaseModel):
    model_config = ConfigDict(frozen=True)

    rows: tuple[dict[str, float], ...]
    matrix: list[list[float]] = Field(default_factory=list)
    batch_sha256: str
    catalog_sha256: str


def build_sentinel_features(
    rows: Sequence[V5DecisionRow],
    *,
    catalog: SentinelFeatureCatalog,
) -> SentinelFeatureBatch:
    """Build a causal feature matrix using strict history-only cohort processing."""
    feature_rows: list[dict[str, float]] = []
    matrix: list[list[float]] = []

    # Build history-only indexes.
    actor_events: dict[str, list[V5DecisionRow]] = defaultdict(list)
    counterparty_events: dict[str, list[V5DecisionRow]] = defaultdict(list)
    pair_events: dict[tuple[str, str], list[V5DecisionRow]] = defaultdict(list)
    all_edges_seen = 0
    actor_nodes_seen: set[str] = set()
    cp_nodes_seen: set[str] = set()

    # Sort rows into canonical timestamp cohorts.
    sorted_rows = sorted(rows, key=lambda r: (r.decision_at, r.event_id))
    cohorts: list[list[V5DecisionRow]] = []
    current_time = None
    current_cohort: list[V5DecisionRow] = []
    for row in sorted_rows:
        if row.decision_at != current_time:
            if current_cohort:
                cohorts.append(current_cohort)
            current_cohort = [row]
            current_time = row.decision_at
        else:
            current_cohort.append(row)
    if current_cohort:
        cohorts.append(current_cohort)

    for cohort in cohorts:
        # Phase 1: emit features for every event in this cohort using prior state only.
        for row in cohort:
            now = row.decision_at
            values = dict(row.predictive_features)
            total_minutes = row.decision_at.hour * 60 + row.decision_at.minute
            values["txn_hour_sin"] = math.sin(2 * math.pi * total_minutes / (24 * 60))
            values["txn_hour_cos"] = math.cos(2 * math.pi * total_minutes / (24 * 60))

            # Actor velocity from strictly prior events.
            prior_actor = [
                r for r in actor_events.get(row.actor_id, [])
                if r.decision_at < now and r.event_id != row.event_id
            ]
            velocity_windows = [
                (60, "1m"), (300, "5m"), (3600, "1h"),
                (86400, "24h"), (604800, "7d"),
            ]
            for window_s, name in velocity_windows:
                features_key = f"actor_count_{name}"
                if features_key in catalog.feature_names:
                    count = sum(
                        1 for r in prior_actor
                        if (now - r.decision_at).total_seconds() <= window_s
                    )
                    values[features_key] = float(count)

            actor_amount_1h = sum(
                float(r.amount) for r in prior_actor
                if (now - r.decision_at).total_seconds() <= 3600
            )
            _set_if_in_catalog(values, catalog, "actor_amount_1h", actor_amount_1h)
            actor_amount_24h = sum(
                float(r.amount) for r in prior_actor
                if (now - r.decision_at).total_seconds() <= 86400
            )
            _set_if_in_catalog(values, catalog, "actor_amount_24h", actor_amount_24h)

            actor_history = actor_events.get(row.actor_id, [])
            first_seen = min((r.decision_at for r in actor_history), default=now)
            last_prior = max((r.decision_at for r in prior_actor), default=None)
            seconds_since_first = (now - first_seen).total_seconds()
            _set_if_in_catalog(values, catalog, "actor_seconds_since_first", seconds_since_first)
            seconds_since_last = (
                (now - last_prior).total_seconds() if last_prior else -1.0
            )
            _set_if_in_catalog(values, catalog, "actor_seconds_since_last", seconds_since_last)

            distinct_cps = len({r.counterparty_id for r in prior_actor})
            _set_if_in_catalog(
                values, catalog,
                "actor_distinct_counterparties_24h", float(distinct_cps),
            )
            _set_if_in_catalog(values, catalog, "graph_actor_fanout", float(distinct_cps))

            prior_cp = [
                r for r in counterparty_events.get(row.counterparty_id, [])
                if r.decision_at < now
                and r.event_id != row.event_id
                and (now - r.decision_at).total_seconds() <= 86400
            ]
            _set_if_in_catalog(values, catalog, "counterparty_count_1h",
                float(sum(1 for r in prior_cp if (now - r.decision_at).total_seconds() <= 3600)))
            _set_if_in_catalog(values, catalog, "counterparty_count_24h", float(len(prior_cp)))
            _set_if_in_catalog(values, catalog, "counterparty_amount_24h",
                sum(float(r.amount) for r in prior_cp))
            distinct_actors = len({
                r.actor_id for r in prior_cp
                if (now - r.decision_at).total_seconds() <= 86400
            })
            _set_if_in_catalog(
                values, catalog,
                "counterparty_distinct_actors_24h", float(distinct_actors),
            )
            _set_if_in_catalog(values, catalog, "graph_counterparty_fanin", float(distinct_actors))

            pair_key = (row.actor_id, row.counterparty_id)
            prior_pair = [
                r for r in pair_events.get(pair_key, [])
                if r.decision_at < now and r.event_id != row.event_id
            ]
            _set_if_in_catalog(values, catalog, "pair_prior_count", float(len(prior_pair)))
            pair_seconds_since_last = (
                (now - max(r.decision_at for r in prior_pair)).total_seconds()
                if prior_pair else -1.0
            )
            _set_if_in_catalog(
                values, catalog, "pair_seconds_since_last", pair_seconds_since_last
            )
            repeated_edge = float(min(len(prior_pair), 5.0))
            _set_if_in_catalog(values, catalog, "graph_repeated_edge", repeated_edge)

            amounts_24h = [
                float(r.amount) for r in prior_actor
                if (now - r.decision_at).total_seconds() <= 86400
            ]
            current_amount = float(row.amount)
            if len(amounts_24h) >= 3:
                mean_amt = sum(amounts_24h) / len(amounts_24h)
                variance = sum((a - mean_amt) ** 2 for a in amounts_24h) / len(amounts_24h)
                std = max(math.sqrt(max(variance, 0.01)), 0.01)
                zscore = (current_amount - mean_amt) / std
                _set_if_in_catalog(values, catalog, "actor_amount_zscore_24h", zscore)
            else:
                _set_if_in_catalog(values, catalog, "actor_amount_zscore_24h", 0.0)

            # Graph features: snapshots of the PRIOR graph only.
            cp_actors_prior = {
                r.actor_id for r in counterparty_events.get(row.counterparty_id, [])
                if r.decision_at < now
            }
            shared = sum(
                1 for other_actor in cp_actors_prior
                if other_actor != row.actor_id
            )
            _set_if_in_catalog(values, catalog, "graph_shared_neighbor_count", float(shared))
            _set_if_in_catalog(values, catalog, "graph_two_hop_reach", float(min(shared * 2, 20.0)))

            recent_burst = [
                r for r in prior_actor
                if (now - r.decision_at).total_seconds() <= 60
            ]
            burst_motif = float(min(len(recent_burst), 10.0))
            _set_if_in_catalog(values, catalog, "graph_burst_motif", burst_motif)
            prior_graph_size = len(actor_nodes_seen | cp_nodes_seen)
            _set_if_in_catalog(values, catalog, "graph_component_size", float(prior_graph_size))
            graph_nodes = max(len(actor_nodes_seen) * len(cp_nodes_seen), 1)
            _set_if_in_catalog(values, catalog, "graph_edge_density", all_edges_seen / graph_nodes)

            vector = [values.get(name, 0.0) for name in catalog.feature_names]
            if not all(math.isfinite(v) for v in vector):
                raise ValueError(f"non-finite feature value in event {row.event_id}")
            feature_rows.append(values)
            matrix.append(vector)

        # Phase 2: update state with all events from this cohort AFTER emitting features.
        for row in cohort:
            actor_events[row.actor_id].append(row)
            counterparty_events[row.counterparty_id].append(row)
            pair_events[(row.actor_id, row.counterparty_id)].append(row)
            actor_nodes_seen.add(row.actor_id)
            cp_nodes_seen.add(row.counterparty_id)
            all_edges_seen += 1

    content = json.dumps(
        {"rows": matrix, "names": list(catalog.feature_names)},
        sort_keys=True,
    ).encode()
    return SentinelFeatureBatch(
        rows=tuple(feature_rows),
        matrix=matrix,
        batch_sha256=hashlib.sha256(content).hexdigest(),
        catalog_sha256=catalog.catalog_sha256,
)


__all__ = ["SentinelFeatureBatch", "SentinelFeatureCatalog", "build_sentinel_features"]
