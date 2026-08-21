"""Metric projection and bootstrap bridge tests for Defend v3."""

from __future__ import annotations

from datetime import date

import pytest

from apar.evaluation.v2_selection import BootstrapMetricContribution, V2BootstrapBlock
from apar.evaluation.v3_metrics import (
    V3MetricsError,
    bootstrap_metrics,
    build_bootstrap_block,
)


def _block(day: date, block_id: str) -> V2BootstrapBlock:
    return build_bootstrap_block(
        day=day,
        block_id=block_id,
        metrics={
            "recall": BootstrapMetricContribution(kind="ratio", numerator=1.0, denominator=2.0),
        },
    )


def test_bootstrap_block_construction() -> None:
    block = _block(date(2026, 3, 1), "case-a")
    assert block.day == date(2026, 3, 1)
    assert block.block_id == "case-a"
    assert "recall" in block.metrics


def test_bootstrap_requires_exactly_2000_replicates() -> None:
    blocks = (_block(date(2026, 3, 1), "case-a"), _block(date(2026, 3, 2), "case-b"))
    with pytest.raises(V3MetricsError, match="exactly 2,000"):
        bootstrap_metrics(blocks, seed=7, replicates=100)


def test_bootstrap_produces_bounded_metrics() -> None:
    blocks = (
        _block(date(2026, 3, 1), "case-a"),
        _block(date(2026, 3, 1), "case-b"),
        _block(date(2026, 3, 2), "case-c"),
    )
    result = bootstrap_metrics(blocks, seed=7)
    assert "recall" in result
    metric = result["recall"]
    assert metric.point is not None
    assert metric.lower is not None
    assert metric.upper is not None
    assert metric.bootstrap_replicates == 2_000


def test_bootstrap_is_deterministic_for_same_seed() -> None:
    blocks = (
        _block(date(2026, 3, 1), "case-a"),
        _block(date(2026, 3, 2), "case-b"),
    )
    first = bootstrap_metrics(blocks, seed=42)
    second = bootstrap_metrics(blocks, seed=42)
    assert first["recall"].lower == second["recall"].lower
    assert first["recall"].upper == second["recall"].upper


def test_empty_blocks_rejected() -> None:
    with pytest.raises(ValueError, match="at least one case block"):
        bootstrap_metrics((), seed=7)
