"""Independent verification for exported non-authoritative v5 rescue evidence."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from apar.evaluation.v5_rescue_verifier import verify_v5_rescue_artifacts

ARM_ORDER = (
    "rules_only",
    "ensemble_no_graph",
    "ensemble_with_graph",
    "full_sentinel",
)
STAGE_ORDER = (
    "00_authorize",
    "10_corpus",
    "20_features",
    "30_arms",
    "40_label_shuffle",
    "50_identity_rename",
    "51_future_causality",
    "52_equal_time_isolation",
    "53_feature_leakage",
    "60_single_class_controls",
)
ROOT = Path(__file__).resolve().parents[2]
VERIFIER = ROOT / "scripts/verify_defense_v5_kaggle_non_authoritative_rescue.py"
RECOVERED_EVIDENCE = ROOT / "evidence/sentinel-v5-recovered-metrics"


def _canonical(document: object) -> bytes:
    return json.dumps(
        document,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode()


def _digest(document: object) -> str:
    return hashlib.sha256(_canonical(document)).hexdigest()


def _write_valid_bundle(root: Path) -> None:
    root.mkdir()
    arm_root = root / "compact-arm-receipts"
    arm_root.mkdir()
    arm_documents: list[dict[str, object]] = []
    support_sha256 = "9" * 64
    readiness: dict[str, object] = {
        "schema_version": "apar-sentinel-v5-non-authoritative-compact-readiness/1",
        "deterministic_controls": {"controls_sha256": "8" * 64},
        "deterministic_readiness": {"readiness_sha256": "7" * 64},
        "observational_readiness": {
            "evaluated_arm": "full_sentinel",
            "status": "not_ready",
            "gates": [
                {
                    "metric": "false_decline_rate",
                    "family": None,
                    "passed": False,
                    "point": 0.87,
                    "target": 0.001,
                }
            ],
            "qualifying_controls": [["benign_only", False, "6" * 64]],
            "readiness_sha256": "5" * 64,
        },
    }
    readiness["readiness_bundle_sha256"] = _digest(readiness)
    for index, arm in enumerate(ARM_ORDER):
        aggregate = {
            "recall": {"value": 0.90 + index / 100, "metric_sha256": "1" * 64},
            "precision": {"value": 0.80 + index / 100, "metric_sha256": "2" * 64},
            "false_decline_rate": {"value": 0.10 + index / 100},
            "challenge_rate": {"value": 0.01 + index / 1000},
            "review_rate": {"value": 0.001 + index / 10000},
            "captured_value_fraction": {"value": 0.95},
            "p95_latency_ms": {"value": 10.0 + index},
        }
        result_digest = f"{index + 1:x}" * 64
        metric_core: dict[str, object] = {
            "arm": arm,
            "deterministic_result_sha256": result_digest,
            "support_sha256": support_sha256,
            "aggregate": aggregate,
            "calibration_sha256": "a" * 64,
            "economics_sha256": "b" * 64,
            "family_sha256": ["c" * 64],
            "bootstrap_sha256": "d" * 64,
        }
        metric_core["deterministic_complete_metrics_sha256"] = _digest(metric_core)
        metric_observation: dict[str, object] = {
            "schema_version": ("apar-sentinel-v5-non-authoritative-compact-arm-metric/1"),
            "arm": arm,
            "arm_result_sha256": "e" * 64,
            "support_sha256": support_sha256,
            "complete_metrics_sha256": "f" * 64,
            "aggregate": aggregate,
            "calibration_sha256": "a" * 64,
            "economics_sha256": "b" * 64,
            "families": [],
            "bootstrap": {"bootstrap_sha256": "d" * 64},
        }
        metric_observation["compact_observation_sha256"] = _digest(metric_observation)
        arm_document: dict[str, object] = {
            "schema_version": ("apar-sentinel-v5-non-authoritative-compact-arm-receipt/1"),
            "authoritative": False,
            "accepted_capacity_evidence": False,
            "arm": arm,
            "deterministic_result_sha256": result_digest,
            "metric_core": metric_core,
            "metric_observation": metric_observation,
            "readiness_bundle": readiness if arm == "full_sentinel" else None,
        }
        arm_document["receipt_sha256"] = _digest(arm_document)
        (arm_root / f"{index:02d}-{arm}.json").write_bytes(_canonical(arm_document))
        arm_documents.append(arm_document)

    artifact: dict[str, object] = {
        "schema_version": "apar-sentinel-v5-non-authoritative-compact-rescue/1",
        "authoritative": False,
        "accepted_capacity_evidence": False,
        "rescue_reason": "official_stage_70_memory_exhaustion",
        "approved_source_commit": "40fb4a131da36556d2b8a04564cb62d73152c7c1",
        "execution_mode": "kaggle_capacity_validation",
        "official_predecessor_stage_manifests": [
            [stage, f"{index:x}" * 64] for index, stage in enumerate(STAGE_ORDER, 1)
        ],
        "run_binding_sha256": "a" * 64,
        "attempt_receipt_sha256": "b" * 64,
        "arm_metric_receipts": [
            {
                "arm": document["arm"],
                "deterministic_result_sha256": document["deterministic_result_sha256"],
                "metric_core": document["metric_core"],
                "metric_observation": document["metric_observation"],
                "receipt_sha256": document["receipt_sha256"],
            }
            for document in arm_documents
        ],
        "readiness_bundle": readiness,
    }
    artifact["rescue_compact_sha256"] = _digest(artifact)
    artifact_path = root / "70_metrics_non_authoritative_compact_rescue.json"
    artifact_path.write_bytes(_canonical(artifact))
    receipt: dict[str, object] = {
        "schema_version": ("apar-sentinel-v5-non-authoritative-compact-rescue-receipt/1"),
        "authoritative": False,
        "accepted_capacity_evidence": False,
        "artifact": artifact_path.name,
        "artifact_size_bytes": artifact_path.stat().st_size,
        "artifact_sha256": hashlib.sha256(artifact_path.read_bytes()).hexdigest(),
        "rescue_compact_sha256": artifact["rescue_compact_sha256"],
        "arm_receipt_sha256": [document["receipt_sha256"] for document in arm_documents],
        "lineage_terminal_manifest_sha256": "a" * 64,
    }
    receipt["receipt_sha256"] = _digest(receipt)
    (root / "rescue-receipt.json").write_bytes(_canonical(receipt))


def test_verifier_reports_exact_non_authoritative_lineage_and_metrics(
    tmp_path: Path,
) -> None:
    bundle = tmp_path / "rescue"
    _write_valid_bundle(bundle)

    report = verify_v5_rescue_artifacts(bundle)

    assert report["authoritative"] is False
    assert report["accepted_capacity_evidence"] is False
    assert report["first_missing_official_stage"] == "70_metrics"
    assert report["official_predecessor_stage_manifests"][-1] == [
        "60_single_class_controls",
        "a" * 64,
    ]
    assert [item["arm"] for item in report["arms"]] == list(ARM_ORDER)
    assert report["arms"][-1]["aggregate"]["recall"] == 0.93
    assert report["readiness"]["status"] == "not_ready"
    assert report["verification_sha256"] == _digest(
        {key: value for key, value in report.items() if key != "verification_sha256"}
    )


def test_verifier_rejects_an_arm_receipt_tamper(tmp_path: Path) -> None:
    bundle = tmp_path / "rescue"
    _write_valid_bundle(bundle)
    arm_path = bundle / "compact-arm-receipts/03-full_sentinel.json"
    document = json.loads(arm_path.read_bytes())
    document["accepted_capacity_evidence"] = True
    arm_path.write_bytes(_canonical(document))

    with pytest.raises(ValueError, match="arm receipt"):
        verify_v5_rescue_artifacts(bundle)


def test_verifier_cli_writes_the_canonical_verification_report(tmp_path: Path) -> None:
    bundle = tmp_path / "rescue"
    report_path = tmp_path / "verified.json"
    _write_valid_bundle(bundle)

    completed = subprocess.run(
        [
            sys.executable,
            str(VERIFIER),
            "--artifact-root",
            str(bundle),
            "--report",
            str(report_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    report = json.loads(report_path.read_bytes())
    assert json.loads(completed.stdout) == report
    assert report["first_missing_official_stage"] == "70_metrics"
    assert report["verification_sha256"] == _digest(
        {key: value for key, value in report.items() if key != "verification_sha256"}
    )


def test_committed_recovered_metrics_evidence_is_self_bound() -> None:
    report = json.loads((RECOVERED_EVIDENCE / "verified-report.json").read_bytes())
    receipt = json.loads((RECOVERED_EVIDENCE / "source-rescue-receipt.json").read_bytes())

    assert report["authoritative"] is False
    assert report["accepted_capacity_evidence"] is False
    assert report["verification_sha256"] == (
        "92b0add77fb41f34c8072b553c1c45e17dccc0b9a1387552252f0a42dde4e9a0"
    )
    assert report["verification_sha256"] == _digest(
        {key: value for key, value in report.items() if key != "verification_sha256"}
    )
    assert receipt["receipt_sha256"] == (
        "758309bbb554feae7fbea3550170bf1b86927582e204176fb5388fcfbeaea1b3"
    )
    assert receipt["receipt_sha256"] == _digest(
        {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    )
    assert receipt["artifact_sha256"] == report["source_artifact_sha256"]
