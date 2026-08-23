"""Runner feature-binding regression tests."""

from __future__ import annotations

import inspect
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "scripts" / "run_defense_v5_development.py"


class TestRunnerFeatureBinding:
    def test_runner_imports_sentinel_feature_catalog(self) -> None:
        source = RUNNER.read_text()
        assert "SentinelFeatureCatalog" in source, (
            "runner must import SentinelFeatureCatalog"
        )

    def test_runner_calls_build_sentinel_features(self) -> None:
        source = RUNNER.read_text()
        assert "build_sentinel_features" in source, (
            "runner must call build_sentinel_features"
        )

    def test_runner_does_not_build_dynamic_feature_union(self) -> None:
        source = RUNNER.read_text()
        assert "predictive_features.keys()" not in source, (
            "runner must not build dynamic feature union from row.predictive_features.keys()"
        )
        assert "_FORBIDDEN_FEATURE_NAMES" not in source.replace(
            "_FORBIDDEN_FEATURE_NAMES = {", ""
        ).replace("_FORBIDDEN_FEATURE_NAMES}", ""), (
            "runner must not maintain its own forbidden-feature list"
        )
