"""Causal Sentinel feature projection for Defend v5."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from apar.evaluation.v5_population import V5DecisionRow

_FORBIDDEN_FIELDS = frozenset(
    {"family", "campaign_id", "scenario_id", "seed", "split", "is_fraud",
     "generator", "label", "outcome"}
)


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
    """Build a causal feature matrix from decision rows."""
    feature_rows: list[dict[str, float]] = []
    matrix: list[list[float]] = []
    for row in rows:
        values = dict(row.predictive_features)
        hour = row.decision_at.hour
        minute = row.decision_at.minute
        total_minutes = hour * 60 + minute
        values["txn_hour_sin"] = math.sin(2 * math.pi * total_minutes / (24 * 60))
        values["txn_hour_cos"] = math.cos(2 * math.pi * total_minutes / (24 * 60))
        vector = [values.get(name, 0.0) for name in catalog.feature_names]
        if not all(math.isfinite(v) for v in vector):
            raise ValueError(f"non-finite feature value in event {row.event_id}")
        feature_rows.append(values)
        matrix.append(vector)

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
