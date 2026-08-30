from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.run_apar_console import score_live_trace, verify_console_assets

ROOT = Path(__file__).resolve().parents[2]


def test_fixed_fallback_trace_is_hash_bound_to_portable_bundle() -> None:
    report = verify_console_assets(ROOT)

    assert report["status"] == "ready"
    assert report["portable_arm"] == "ensemble_with_graph"
    assert report["fallback_replay_verified"] is True
    assert report["fallback_trace_sha256"] == (
        "207b832ffe4d1fad2c19bb5dceb45861746eebdc5674a97f3ba9ac2f668c625a"
    )


def test_fixed_fallback_rejects_changed_trace(tmp_path: Path) -> None:
    trace = json.loads((ROOT / "web/public/data/verified-trace.json").read_text())
    trace["traces"][0]["latency_ms"] += 1
    altered = tmp_path / "verified-trace.json"
    altered.write_text(json.dumps(trace))

    with pytest.raises(ValueError, match="fallback trace hash differs"):
        verify_console_assets(ROOT, trace_path=altered)


def test_live_worker_calls_portable_graph_ensemble(tmp_path: Path) -> None:
    output = tmp_path / "live-trace.json"

    report = score_live_trace(ROOT, output)

    assert report["replay_verified"] is True
    assert len(report["traces"]) == 12
    assert {item["arm"] for item in report["traces"]} == {"ensemble_with_graph"}
    assert output.is_file()
