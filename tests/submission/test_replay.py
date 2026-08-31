from __future__ import annotations

from scripts.submission.replay import build_fallback_trace


def test_fallback_trace_excludes_runtime_timings_and_binds_all_predictions() -> None:
    """Wall-clock fields would make the clean-room fallback trace nondeterministic."""
    report = {
        "bundle_manifest_sha256": "5" * 64,
        "model_load_ms": 42.0,
        "replay_verified": True,
        "scoring_wall_ms": 7.0,
        "traces": [
            {
                "calibrated_probability": 0.25,
                "event_id": "event-1",
                "final_action": "challenge",
                "latency_ms": 1.5,
                "replay_probability_abs_error": 0.0,
            },
            {
                "calibrated_probability": 0.75,
                "event_id": "event-2",
                "final_action": "review_hold",
                "latency_ms": 2.5,
                "replay_probability_abs_error": 0.0,
            },
        ],
    }

    trace = build_fallback_trace(report, web_status="pending")

    assert trace == {
        "bundle_manifest_sha256": "5" * 64,
        "fallback_from": "web_prototype",
        "fallback_reason": "web_artifact_not_integrated",
        "fallback_to": "portable_sentinel_cli",
        "prediction_sha256": trace["prediction_sha256"],
        "predictions": [
            {
                "action": "challenge",
                "event_id": "event-1",
                "probability": 0.25,
                "replay_probability_abs_error": 0.0,
            },
            {
                "action": "review_hold",
                "event_id": "event-2",
                "probability": 0.75,
                "replay_probability_abs_error": 0.0,
            },
        ],
        "replay_verified": True,
        "scenario_count": 2,
        "schema_version": "apar-submission-fallback-trace/1",
        "trace_sha256": trace["trace_sha256"],
        "web_status": "pending",
    }
    assert trace["prediction_sha256"] == (
        "860a263639f2f326962af90dc174df28476d64dab5352c9c4a404a64afee358b"
    )
    assert trace["trace_sha256"] == (
        "6355bcccf8697d540b9641c7681ad7633ab2b99522d7b9152bc3399a5d2f342f"
    )
