"""Deterministic projection of the portable model replay for UI fallback review."""

from __future__ import annotations

from typing import Any, cast

from scripts.submission.model import ReleaseError, canonical_json, sha256_bytes


def build_fallback_trace(report: dict[str, Any], *, web_status: str) -> dict[str, Any]:
    """Drop wall-clock fields and bind every replayed prediction/action."""
    raw_traces = report.get("traces")
    if report.get("replay_verified") is not True or not isinstance(raw_traces, list):
        raise ReleaseError("portable replay report is not verified")
    predictions: list[dict[str, Any]] = []
    for raw_trace in raw_traces:
        if not isinstance(raw_trace, dict):
            raise ReleaseError("portable replay trace is malformed")
        trace = cast(dict[str, Any], raw_trace)
        predictions.append(
            {
                "action": trace.get("final_action"),
                "event_id": trace.get("event_id"),
                "probability": trace.get("calibrated_probability"),
                "replay_probability_abs_error": trace.get("replay_probability_abs_error"),
            }
        )
    prediction_sha256 = sha256_bytes(canonical_json(predictions))
    fallback_reason = (
        "web_artifact_not_integrated" if web_status == "pending" else "web_runtime_unavailable"
    )
    result: dict[str, Any] = {
        "bundle_manifest_sha256": report.get("bundle_manifest_sha256"),
        "fallback_from": "web_prototype",
        "fallback_reason": fallback_reason,
        "fallback_to": "portable_sentinel_cli",
        "prediction_sha256": prediction_sha256,
        "predictions": predictions,
        "replay_verified": True,
        "scenario_count": len(predictions),
        "schema_version": "apar-submission-fallback-trace/1",
        "web_status": web_status,
    }
    result["trace_sha256"] = sha256_bytes(canonical_json(result))
    return result
