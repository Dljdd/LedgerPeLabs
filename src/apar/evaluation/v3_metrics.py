"""Metric projection and bootstrap bridge for Defend v3.

Reuses v2 metric and selection types without changing their definitions.
Produces aggregate, strata, family, regime, cohort, and held-out-family evidence
plus exactly 2,000 two-level bootstrap replicates.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import date

from apar.evaluation.metrics import MetricReport
from apar.evaluation.v2_selection import (
    BoundedMetric,
    BootstrapMetricContribution,
    V2BootstrapBlock,
    V2MetricSet,
    bootstrap_v2_metrics,
)
from apar.cases.v2_workload import ActionWorkload
from apar.v3_protocol import V3ProtocolError


class V3MetricsError(V3ProtocolError):
    """Metric projection or bootstrap evidence is incomplete or inconsistent."""


def project_v2_metrics(
    report: MetricReport,
    workload: ActionWorkload,
) -> V2MetricSet:
    """Project a v2-compatible metric report and workload into exact evidence."""
    if type(report) is not MetricReport:
        raise V3MetricsError("metric report must be an exact MetricReport")
    if type(workload) is not ActionWorkload:
        raise V3MetricsError("workload must be an exact ActionWorkload")
    return V2MetricSet.from_metric_report(report, workload)


def build_bootstrap_block(
    *,
    day: date,
    block_id: str,
    metrics: Mapping[str, BootstrapMetricContribution],
) -> V2BootstrapBlock:
    """Build one day/case bootstrap block with validated metric contributions."""
    if type(day) is not date:
        raise V3MetricsError("bootstrap block day must be an exact date")
    if type(block_id) is not str or not block_id:
        raise V3MetricsError("bootstrap block ID must be nonempty")
    if type(metrics) is not dict or not metrics:
        raise V3MetricsError("bootstrap block requires nonempty metric contributions")
    return V2BootstrapBlock(day=day, block_id=block_id, metrics=dict(metrics))


def bootstrap_metrics(
    blocks: Sequence[V2BootstrapBlock],
    *,
    seed: int,
    replicates: int = 2_000,
) -> Mapping[str, BoundedMetric]:
    """Run exactly the preregistered two-level day/case-block bootstrap."""
    if replicates != 2_000:
        raise V3MetricsError("v3 requires exactly 2,000 bootstrap replicates")
    if type(seed) is not int:
        raise V3MetricsError("bootstrap seed must be an exact integer")
    return bootstrap_v2_metrics(tuple(blocks), seed=seed)


__all__ = [
    "V3MetricsError",
    "bootstrap_metrics",
    "build_bootstrap_block",
    "project_v2_metrics",
]
