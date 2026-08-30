"""Independent verifier for non-authoritative Sentinel v5 rescue artifacts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import cast

_APPROVED_SOURCE_COMMIT = "40fb4a131da36556d2b8a04564cb62d73152c7c1"
_ARM_ORDER = (
    "rules_only",
    "ensemble_no_graph",
    "ensemble_with_graph",
    "full_sentinel",
)
_PREDECESSOR_STAGE_ORDER = (
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
_SUMMARY_METRICS = (
    "recall",
    "precision",
    "f1",
    "false_decline_rate",
    "challenge_rate",
    "review_rate",
    "captured_value_fraction",
    "p95_latency_ms",
)


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


def _load_object(path: Path, *, label: str) -> dict[str, object]:
    try:
        raw = json.loads(path.read_bytes())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is missing or malformed") from error
    if not isinstance(raw, dict) or any(not isinstance(key, str) for key in raw):
        raise ValueError(f"{label} is not an object")
    return cast(dict[str, object], raw)


def _mapping(value: object, *, label: str) -> dict[str, object]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise ValueError(f"{label} is not an object")
    return cast(dict[str, object], value)


def _sequence(value: object, *, label: str) -> list[object]:
    if not isinstance(value, list):
        raise ValueError(f"{label} is not an array")
    return cast(list[object], value)


def _sha256(value: object, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} is not a SHA-256 digest")
    return value


def _verify_self_digest(document: dict[str, object], *, field: str, label: str) -> str:
    claimed = _sha256(document.get(field), label=f"{label} {field}")
    unsigned = dict(document)
    unsigned.pop(field, None)
    if _digest(unsigned) != claimed:
        raise ValueError(f"{label} self-binding differs")
    return claimed


def _verify_non_authoritative(document: dict[str, object], *, label: str) -> None:
    if (
        document.get("authoritative") is not False
        or document.get("accepted_capacity_evidence") is not False
    ):
        raise ValueError(f"{label} authority boundary differs")


def _metric_projection(observation: dict[str, object]) -> dict[str, object]:
    aggregate = _mapping(observation.get("aggregate"), label="arm aggregate")
    projected: dict[str, object] = {}
    for name in _SUMMARY_METRICS:
        raw = aggregate.get(name)
        if raw is None:
            continue
        estimate = _mapping(raw, label=f"aggregate metric {name}")
        value = estimate.get("value")
        if value is not None and not isinstance(value, (int, float)):
            raise ValueError(f"aggregate metric {name} value differs")
        projected[name] = value
    return projected


def _verify_arm_receipt(
    *,
    path: Path,
    arm: str,
    embedded: dict[str, object],
    readiness: dict[str, object],
) -> dict[str, object]:
    document = _load_object(path, label=f"{arm} arm receipt")
    if (
        document.get("schema_version") != "apar-sentinel-v5-non-authoritative-compact-arm-receipt/1"
        or document.get("arm") != arm
    ):
        raise ValueError(f"{arm} arm receipt identity differs")
    _verify_non_authoritative(document, label=f"{arm} arm receipt")
    receipt_sha256 = _verify_self_digest(
        document,
        field="receipt_sha256",
        label=f"{arm} arm receipt",
    )
    core = _mapping(document.get("metric_core"), label=f"{arm} metric core")
    observation = _mapping(
        document.get("metric_observation"),
        label=f"{arm} metric observation",
    )
    if (
        core.get("arm") != arm
        or observation.get("arm") != arm
        or core.get("support_sha256") != observation.get("support_sha256")
        or document.get("deterministic_result_sha256") != core.get("deterministic_result_sha256")
    ):
        raise ValueError(f"{arm} arm receipt metric binding differs")
    _sha256(core.get("support_sha256"), label=f"{arm} support digest")
    _verify_self_digest(
        core,
        field="deterministic_complete_metrics_sha256",
        label=f"{arm} metric core",
    )
    _verify_self_digest(
        observation,
        field="compact_observation_sha256",
        label=f"{arm} metric observation",
    )
    expected_readiness: object = readiness if arm == "full_sentinel" else None
    if document.get("readiness_bundle") != expected_readiness:
        raise ValueError(f"{arm} arm receipt readiness binding differs")
    expected_embedded = {
        "arm": document["arm"],
        "deterministic_result_sha256": document["deterministic_result_sha256"],
        "metric_core": core,
        "metric_observation": observation,
        "receipt_sha256": receipt_sha256,
    }
    if embedded != expected_embedded:
        raise ValueError(f"{arm} embedded arm receipt differs")
    return {
        "arm": arm,
        "deterministic_result_sha256": _sha256(
            document.get("deterministic_result_sha256"),
            label=f"{arm} result digest",
        ),
        "receipt_sha256": receipt_sha256,
        "support_sha256": core["support_sha256"],
        "complete_metrics_sha256": _sha256(
            observation.get("complete_metrics_sha256"),
            label=f"{arm} complete metrics digest",
        ),
        "aggregate": _metric_projection(observation),
    }


def _readiness_projection(readiness: dict[str, object]) -> dict[str, object]:
    _verify_self_digest(
        readiness,
        field="readiness_bundle_sha256",
        label="readiness bundle",
    )
    observation = _mapping(
        readiness.get("observational_readiness"),
        label="observational readiness",
    )
    if observation.get("evaluated_arm") != "full_sentinel" or observation.get("status") not in {
        "ready",
        "not_ready",
    }:
        raise ValueError("observational readiness identity differs")
    return {
        "status": observation["status"],
        "evaluated_arm": observation["evaluated_arm"],
        "readiness_sha256": _sha256(
            observation.get("readiness_sha256"),
            label="readiness digest",
        ),
        "gates": _sequence(observation.get("gates"), label="readiness gates"),
        "qualifying_controls": _sequence(
            observation.get("qualifying_controls"),
            label="qualifying controls",
        ),
    }


def verify_v5_rescue_artifacts(root: Path) -> dict[str, object]:
    """Verify a downloaded rescue directory and return a hash-bound compact report."""
    root = root.resolve()
    receipt_path = root / "rescue-receipt.json"
    receipt = _load_object(receipt_path, label="rescue receipt")
    if (
        receipt.get("schema_version")
        != "apar-sentinel-v5-non-authoritative-compact-rescue-receipt/1"
    ):
        raise ValueError("rescue receipt schema differs")
    _verify_non_authoritative(receipt, label="rescue receipt")
    receipt_sha256 = _verify_self_digest(
        receipt,
        field="receipt_sha256",
        label="rescue receipt",
    )
    artifact_name = receipt.get("artifact")
    if (
        not isinstance(artifact_name, str)
        or Path(artifact_name).name != artifact_name
        or artifact_name != "70_metrics_non_authoritative_compact_rescue.json"
    ):
        raise ValueError("rescue artifact name differs")
    artifact_path = root / artifact_name
    try:
        artifact_bytes = artifact_path.read_bytes()
    except OSError as error:
        raise ValueError("rescue artifact is missing") from error
    artifact_sha256 = hashlib.sha256(artifact_bytes).hexdigest()
    if (
        receipt.get("artifact_size_bytes") != len(artifact_bytes)
        or receipt.get("artifact_sha256") != artifact_sha256
    ):
        raise ValueError("rescue artifact file binding differs")
    artifact = _load_object(artifact_path, label="rescue artifact")
    if (
        artifact.get("schema_version") != "apar-sentinel-v5-non-authoritative-compact-rescue/1"
        or artifact.get("approved_source_commit") != _APPROVED_SOURCE_COMMIT
        or artifact.get("execution_mode") != "kaggle_capacity_validation"
        or artifact.get("rescue_reason") != "official_stage_70_memory_exhaustion"
    ):
        raise ValueError("rescue artifact identity differs")
    _verify_non_authoritative(artifact, label="rescue artifact")
    compact_sha256 = _verify_self_digest(
        artifact,
        field="rescue_compact_sha256",
        label="rescue artifact",
    )
    if receipt.get("rescue_compact_sha256") != compact_sha256:
        raise ValueError("rescue compact receipt binding differs")

    raw_lineage = _sequence(
        artifact.get("official_predecessor_stage_manifests"),
        label="predecessor lineage",
    )
    lineage: list[list[str]] = []
    for expected_stage, raw_entry in zip(
        _PREDECESSOR_STAGE_ORDER,
        raw_lineage,
        strict=True,
    ):
        entry = _sequence(raw_entry, label=f"{expected_stage} lineage entry")
        if len(entry) != 2 or entry[0] != expected_stage:
            raise ValueError("predecessor lineage order differs")
        lineage.append([expected_stage, _sha256(entry[1], label=f"{expected_stage} manifest")])
    if len(raw_lineage) != len(_PREDECESSOR_STAGE_ORDER):
        raise ValueError("predecessor lineage length differs")
    if receipt.get("lineage_terminal_manifest_sha256") != lineage[-1][1]:
        raise ValueError("terminal predecessor receipt binding differs")

    readiness = _mapping(artifact.get("readiness_bundle"), label="readiness bundle")
    readiness_report = _readiness_projection(readiness)
    embedded_arms = _sequence(
        artifact.get("arm_metric_receipts"),
        label="embedded arm receipts",
    )
    claimed_arm_receipts = _sequence(
        receipt.get("arm_receipt_sha256"),
        label="claimed arm receipts",
    )
    if len(embedded_arms) != len(_ARM_ORDER) or len(claimed_arm_receipts) != len(_ARM_ORDER):
        raise ValueError("arm receipt count differs")
    arms: list[dict[str, object]] = []
    for index, arm in enumerate(_ARM_ORDER):
        embedded = _mapping(embedded_arms[index], label=f"{arm} embedded receipt")
        arm_report = _verify_arm_receipt(
            path=root / "compact-arm-receipts" / f"{index:02d}-{arm}.json",
            arm=arm,
            embedded=embedded,
            readiness=readiness,
        )
        if claimed_arm_receipts[index] != arm_report["receipt_sha256"]:
            raise ValueError(f"{arm} arm receipt list binding differs")
        arms.append(arm_report)
    if len({item["support_sha256"] for item in arms}) != 1:
        raise ValueError("arm metric support differs")

    report: dict[str, object] = {
        "schema_version": "apar-sentinel-v5-recovered-metrics-verification/1",
        "authoritative": False,
        "accepted_capacity_evidence": False,
        "official_chain_status": "incomplete",
        "recovered_metrics_status": "non_authoritative",
        "first_missing_official_stage": "70_metrics",
        "approved_source_commit": _APPROVED_SOURCE_COMMIT,
        "source_artifact": artifact_name,
        "source_artifact_sha256": artifact_sha256,
        "source_receipt_sha256": receipt_sha256,
        "source_rescue_compact_sha256": compact_sha256,
        "run_binding_sha256": _sha256(artifact.get("run_binding_sha256"), label="run binding"),
        "attempt_receipt_sha256": _sha256(
            artifact.get("attempt_receipt_sha256"), label="attempt receipt"
        ),
        "official_predecessor_stage_manifests": lineage,
        "arms": arms,
        "readiness": readiness_report,
    }
    report["verification_sha256"] = _digest(report)
    return cast(dict[str, object], json.loads(_canonical(report)))


__all__ = ["verify_v5_rescue_artifacts"]
