from __future__ import annotations

import hashlib
import json
from base64 import b64encode
from pathlib import Path

import numpy as np
import pytest
from catboost import CatBoostClassifier

from apar.demo.sentinel_v5_portable import (
    export_portable_bundle,
    load_portable_bundle,
    run_portable_scenarios,
    score_portable_scenario,
    select_demo_rows,
)


def _canonical(document: object) -> bytes:
    return json.dumps(
        document,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode()


def _write_bundle(root: Path) -> None:
    model = CatBoostClassifier(
        allow_writing_files=False,
        iterations=4,
        depth=2,
        learning_rate=0.2,
        random_seed=404,
        verbose=False,
    )
    model.fit(
        np.asarray([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]),
        np.asarray([0, 0, 1, 1]),
    )
    model_path = root / "models" / "member-00.json"
    model_path.parent.mkdir(parents=True)
    model.save_model(str(model_path), format="json")
    calibrator_path = root / "calibrators" / "member-00.json"
    calibrator_path.parent.mkdir(parents=True)
    calibrator_path.write_bytes(
        _canonical(
            {
                "artifact_sha256": "0" * 64,
                "out_of_bounds": "clip",
                "x_thresholds": [0.0, 1.0],
                "y_thresholds": [0.95, 0.95],
            }
        )
    )
    spec_path = root / "spec.json"
    spec_path.write_bytes(
        _canonical(
            {
                "arm": "ensemble_with_graph",
                "arm_spec_sha256": "1" * 64,
                "catalog_feature_names": ["amount_log", "graph_risk"],
                "feature_names": ["amount_log", "graph_risk"],
                "model_artifacts": [
                    {
                        "artifact_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
                        "path": "models/member-00.json",
                    }
                ],
                "calibrators": [
                    {
                        "artifact_sha256": hashlib.sha256(calibrator_path.read_bytes()).hexdigest(),
                        "path": "calibrators/member-00.json",
                    }
                ],
                "thresholds": {
                    "model_challenge": 0.40,
                    "model_decline": 0.80,
                    "model_review": 0.60,
                },
                "switches": {
                    "disagreement": False,
                    "graph": True,
                    "model": True,
                    "novelty": False,
                    "rules": False,
                    "trust": False,
                },
            }
        )
    )
    manifest = {
        "accepted_capacity_evidence": False,
        "authoritative": False,
        "bundle_files": {
            path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (model_path, calibrator_path, spec_path)
        },
        "demo_only": True,
        "schema_version": "apar-sentinel-v5-portable-demo-bundle/1",
        "source_checkpoint_manifest_sha256": "2" * 64,
    }
    manifest["manifest_sha256"] = hashlib.sha256(_canonical(manifest)).hexdigest()
    (root / "manifest.json").write_bytes(_canonical(manifest))


def test_portable_bundle_replays_real_catboost_with_frozen_calibration(tmp_path: Path) -> None:
    """A missing model/calibrator replay would stop reproducing accepted probabilities."""
    _write_bundle(tmp_path)
    bundle = load_portable_bundle(tmp_path)

    trace = score_portable_scenario(
        bundle,
        {
            "event_id": "demo-positive",
            "features": [1.0, 1.0],
            "presentation_ground_truth": {"label": 1, "family": "card_testing_cnp"},
        },
    )

    assert trace["calibrated_probability"] == pytest.approx(0.95)
    assert trace["model_action"] == "decline_hold"
    assert trace["final_action"] == "decline_hold"
    assert trace["event_id"] == "demo-positive"


def test_portable_bundle_rejects_a_tampered_model_file(tmp_path: Path) -> None:
    """Removing file-hash validation would allow an unbound model to be demonstrated."""
    _write_bundle(tmp_path)
    model_path = tmp_path / "models" / "member-00.json"
    model_path.write_bytes(model_path.read_bytes() + b"\n")

    with pytest.raises(ValueError, match="bundle file digest mismatch"):
        load_portable_bundle(tmp_path)


def test_demo_row_selection_covers_families_benign_rails_and_friction_actions() -> None:
    """A first-N sampler would omit required families and proportional interventions."""
    rows = [
        _row("fraud-agent", 1, "agentic_intent_abuse", "agentic", "decline_hold"),
        _row("fraud-app", 1, "app_scam_mule", "a2a", "decline_hold"),
        _row("fraud-card", 1, "card_testing_cnp", "card", "decline_hold"),
        _row("fraud-merchant", 1, "synthetic_merchant_refund", "card", "review_hold"),
        _row("benign-agent", 0, "benign", "agentic", "approve"),
        _row("benign-a2a", 0, "benign", "a2a", "approve"),
        _row("benign-card", 0, "benign", "card", "approve"),
        _row("ambiguous-challenge", 0, "benign", "card", "challenge"),
        _row("ambiguous-review", 0, "benign", "a2a", "review_hold"),
        _row("filler-1", 0, "benign", "card", "approve"),
        _row("filler-2", 0, "benign", "card", "approve"),
        _row("filler-3", 0, "benign", "card", "approve"),
    ]

    selected = select_demo_rows(rows, limit=12)

    assert {row["support"]["family"] for row in selected if row["support"]["label"]} == {
        "agentic_intent_abuse",
        "app_scam_mule",
        "card_testing_cnp",
        "synthetic_merchant_refund",
    }
    assert {row["support"]["rail"] for row in selected if not row["support"]["label"]} >= {
        "a2a",
        "agentic",
        "card",
    }
    assert {row["probability_action"] for row in selected} >= {
        "approve",
        "challenge",
        "review_hold",
    }


def test_exporter_splits_embedded_artifacts_and_binds_real_scenarios(tmp_path: Path) -> None:
    """Keeping payloads embedded or omitting scenario bindings would make export unusable."""
    source = tmp_path / "source"
    source.mkdir()
    _write_bundle(source)
    model_payload = (source / "models" / "member-00.json").read_bytes()
    calibrator = json.loads((source / "calibrators" / "member-00.json").read_bytes())
    calibrator_without_digest = {
        key: value for key, value in calibrator.items() if key != "artifact_sha256"
    }
    calibrator["artifact_sha256"] = hashlib.sha256(
        _canonical(calibrator_without_digest)
    ).hexdigest()
    rows = [
        {
            **_row(
                f"scenario-{index:02d}",
                1 if index < 4 else 0,
                (
                    (
                        "agentic_intent_abuse",
                        "app_scam_mule",
                        "card_testing_cnp",
                        "synthetic_merchant_refund",
                    )[index]
                    if index < 4
                    else "benign"
                ),
                ("agentic", "a2a", "card")[index % 3],
                ("challenge" if index == 7 else "review_hold" if index == 8 else "approve"),
            ),
            "action": "decline_hold" if index < 4 else "approve",
            "catalog_feature_values": [1.0, 1.0],
            "model_calibrated_scores": [0.95],
            "model_raw_scores": [0.75],
            "probability": 0.95,
            "rule_components": [],
            "rule_score": None,
            "subset_feature_values": [1.0, 1.0],
            "trust_routed": False,
        }
        for index in range(12)
    ]
    arm_spec = {
        "arm": "ensemble_with_graph",
        "catalog_feature_groups": ["behavior", "graph"],
        "catalog_feature_names": ["amount_log", "graph_risk"],
        "feature_names": ["amount_log", "graph_risk"],
        "model_artifacts": [
            {
                "artifact_sha256": hashlib.sha256(model_payload).hexdigest(),
                "payload_base64": b64encode(model_payload).decode(),
                "serialization": "catboost-json-canonical-v1",
            }
        ],
        "calibrator_manifests": [calibrator],
        "threshold_digest": "3" * 64,
        "threshold_values": [
            ["model_challenge", 0.40],
            ["model_decline", 0.80],
            ["model_review", 0.60],
        ],
        "model": True,
        "graph": True,
        "rules": False,
        "trust": False,
        "novelty": False,
        "disagreement": False,
        "spec_sha256": "4" * 64,
    }
    output = tmp_path / "exported"

    export_portable_bundle(
        arm_spec=arm_spec,
        rows=rows,
        output_root=output,
        source_checkpoint_manifest_sha256="5" * 64,
        source_commit="40fb4a131da36556d2b8a04564cb62d73152c7c1",
        scenario_limit=12,
    )

    exported_manifest = json.loads((output / "manifest.json").read_bytes())
    scenarios = json.loads((output / "scenarios.json").read_bytes())
    exported_spec = json.loads((output / "spec.json").read_bytes())
    assert exported_manifest["demo_only"] is True
    assert exported_manifest["authoritative"] is False
    assert exported_manifest["accepted_capacity_evidence"] is False
    assert len(scenarios["scenarios"]) == 12
    assert "payload_base64" not in json.dumps(exported_spec)
    assert (output / "models" / "member-00.json").read_bytes() == model_payload
    assert load_portable_bundle(output).spec["arm"] == "ensemble_with_graph"


def test_batch_runner_requires_replay_and_emits_hash_bound_metrics(tmp_path: Path) -> None:
    """A presentation runner must fail closed rather than drift from accepted evidence."""
    _write_bundle(tmp_path)
    scenario_path = tmp_path / "scenarios.json"
    scenario_path.write_bytes(
        _canonical(
            {
                "arm": "ensemble_with_graph",
                "scenarios": [
                    {
                        "accepted_checkpoint_evidence": {
                            "probability": 0.95,
                            "probability_action": "decline_hold",
                        },
                        "event_id": "captured-fraud",
                        "features": [1.0, 1.0],
                        "presentation_ground_truth": {
                            "amount": 125.0,
                            "family": "card_testing_cnp",
                            "label": 1,
                            "rail": "card",
                        },
                    }
                ],
                "schema_version": "apar-sentinel-v5-portable-demo-scenarios/1",
            }
        )
    )
    output_path = tmp_path / "trace.json"

    report = run_portable_scenarios(
        bundle_root=tmp_path,
        scenario_path=scenario_path,
        output_path=output_path,
    )

    assert report["replay_verified"] is True
    assert report["metrics"]["scenario_recall"] == 1.0
    assert report["metrics"]["captured_value_proxy"] == 1.0
    assert report["traces"][0]["final_action"] == "decline_hold"
    claimed_digest = report.pop("trace_sha256")
    assert claimed_digest == hashlib.sha256(_canonical(report)).hexdigest()
    assert json.loads(output_path.read_bytes())["trace_sha256"] == claimed_digest


def _row(event_id: str, label: int, family: str, rail: str, action: str) -> dict[str, object]:
    return {
        "probability_action": action,
        "support": {
            "event_id": event_id,
            "family": family,
            "label": label,
            "rail": rail,
        },
    }
