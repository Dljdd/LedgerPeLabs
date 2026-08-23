"""Derived readiness regression tests."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REPORTING = ROOT / "src" / "apar" / "evaluation" / "v5_reporting.py"


class TestDerivedReadiness:
    def test_reporting_does_not_hardcode_not_ready(self) -> None:
        """The build function must derive status from gates, not hard-code it."""
        source = REPORTING.read_text()
        # Find the build function body.
        build_start = source.find("def build_v5_development_result")
        assert build_start >= 0, "build_v5_development_result not found"
        build_body = source[build_start:]
        # The function must contain conditional status derivation (not just one assignment).
        ready_count = build_body.count('"development_ready"')
        not_ready_count = build_body.count('"development_not_ready"')
        assert ready_count > 0, (
            "build function never assigns development_ready; it is hard-coded to fail"
        )
        assert not_ready_count > 0, "build function must also handle development_not_ready"

    def test_reporting_checks_latency_gate(self) -> None:
        source = REPORTING.read_text()
        build_start = source.find("def build_v5_development_result")
        build_body = source[build_start:]
        assert "p95_latency_ms" in build_body or "latency_missing" in build_body, (
            "build function must check latency gates"
        )
