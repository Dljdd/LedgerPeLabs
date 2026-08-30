"""Portable, hash-bound Sentinel v5 scorer for competition demonstrations.

This module consumes an exported accepted-checkpoint bundle.  It does not train,
adapt, publish, or promote a model and it never treats demo evidence as official
capacity evidence.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from base64 import b64decode
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, cast

import numpy as np
from catboost import CatBoostClassifier  # type: ignore[import-untyped]

from apar.defense.sentinel import SentinelAction

_BUNDLE_SCHEMA = "apar-sentinel-v5-portable-demo-bundle/1"
_FRAUD_FAMILIES = (
    "agentic_intent_abuse",
    "app_scam_mule",
    "card_testing_cnp",
    "synthetic_merchant_refund",
)
_BENIGN_RAILS = ("card", "a2a", "agentic")
_FRICTION_ACTIONS = (
    SentinelAction.CHALLENGE.value,
    SentinelAction.REVIEW_HOLD.value,
)


def _canonical(document: object) -> bytes:
    return json.dumps(
        document,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode()


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _load_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        document = json.loads(path.read_bytes())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is not valid JSON") from error
    if not isinstance(document, dict):
        raise ValueError(f"{label} must be a JSON object")
    return cast(dict[str, Any], document)


def _write_canonical(path: Path, document: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_canonical(document))


def _thresholds_from_spec(arm_spec: dict[str, Any]) -> dict[str, float]:
    raw_thresholds = arm_spec.get("threshold_values")
    try:
        if isinstance(raw_thresholds, dict):
            thresholds = {str(name): float(value) for name, value in raw_thresholds.items()}
        elif isinstance(raw_thresholds, list):
            thresholds = {
                str(item[0]): float(item[1])
                for item in raw_thresholds
                if isinstance(item, list) and len(item) == 2
            }
            if len(thresholds) != len(raw_thresholds):
                raise ValueError("accepted arm thresholds are malformed")
        else:
            raise ValueError("accepted arm thresholds are absent")
    except (TypeError, ValueError) as error:
        raise ValueError("accepted arm thresholds are malformed") from error
    required = {"model_challenge", "model_review", "model_decline"}
    if not required.issubset(thresholds):
        raise ValueError("accepted arm model thresholds are incomplete")
    if not all(math.isfinite(value) for value in thresholds.values()):
        raise ValueError("accepted arm thresholds are not finite")
    return thresholds


def _scenario_from_row(row: dict[str, Any]) -> dict[str, Any]:
    support = row.get("support")
    features = row.get("subset_feature_values")
    if not isinstance(support, dict) or not isinstance(features, list):
        raise ValueError("accepted row lacks support or model features")
    event_id = support.get("event_id")
    if not isinstance(event_id, str) or not event_id:
        raise ValueError("accepted row event identifier is absent")
    presentation_ground_truth = {
        key: support[key]
        for key in (
            "amount",
            "currency",
            "family",
            "label",
            "rail",
            "scenario",
        )
        if key in support
    }
    evidence = {
        key: row[key]
        for key in (
            "action",
            "deterministic_row_sha256",
            "model_calibrated_scores",
            "model_raw_scores",
            "novelty_action",
            "novelty_score",
            "probability",
            "probability_action",
            "rule_components",
            "rule_score",
            "trust_routed",
        )
        if key in row
    }
    return {
        "accepted_checkpoint_evidence": evidence,
        "event_id": event_id,
        "features": features,
        "presentation_ground_truth": presentation_ground_truth,
    }


def export_portable_bundle(
    *,
    arm_spec: dict[str, Any],
    rows: Iterable[dict[str, Any]],
    output_root: Path,
    source_checkpoint_manifest_sha256: str,
    source_commit: str,
    scenario_limit: int = 12,
) -> Path:
    """Export one accepted arm as an immutable, explicitly demo-only bundle."""
    if output_root.exists() and any(output_root.iterdir()):
        raise ValueError("portable bundle output directory is not empty")
    output_root.mkdir(parents=True, exist_ok=True)
    raw_models = arm_spec.get("model_artifacts")
    raw_calibrators = arm_spec.get("calibrator_manifests")
    feature_names = arm_spec.get("feature_names")
    if (
        not isinstance(raw_models, list)
        or not raw_models
        or not isinstance(raw_calibrators, list)
        or len(raw_calibrators) != len(raw_models)
        or not isinstance(feature_names, list)
        or not feature_names
    ):
        raise ValueError("accepted arm model specification is incomplete")

    bundle_files: dict[str, str] = {}
    model_entries: list[dict[str, str]] = []
    calibrator_entries: list[dict[str, str]] = []
    for index, (raw_model, raw_calibrator) in enumerate(
        zip(raw_models, raw_calibrators, strict=True)
    ):
        if not isinstance(raw_model, dict) or not isinstance(raw_calibrator, dict):
            raise ValueError("accepted model member is malformed")
        encoded = raw_model.get("payload_base64")
        claimed_model_sha256 = raw_model.get("artifact_sha256")
        if not isinstance(encoded, str) or not isinstance(claimed_model_sha256, str):
            raise ValueError("accepted CatBoost payload is absent")
        try:
            model_payload = b64decode(encoded, validate=True)
        except ValueError as error:
            raise ValueError("accepted CatBoost payload is not valid base64") from error
        if _sha256(model_payload) != claimed_model_sha256:
            raise ValueError("accepted CatBoost artifact digest mismatch")

        calibrator_without_digest = {
            key: value for key, value in raw_calibrator.items() if key != "artifact_sha256"
        }
        claimed_calibrator_sha256 = raw_calibrator.get("artifact_sha256")
        if claimed_calibrator_sha256 != _sha256(_canonical(calibrator_without_digest)):
            raise ValueError("accepted calibrator artifact digest mismatch")

        model_name = f"models/member-{index:02d}.json"
        calibrator_name = f"calibrators/member-{index:02d}.json"
        model_path = output_root / model_name
        model_path.parent.mkdir(parents=True, exist_ok=True)
        model_path.write_bytes(model_payload)
        _write_canonical(output_root / calibrator_name, raw_calibrator)
        model_file_sha256 = _sha256(model_path.read_bytes())
        calibrator_file_sha256 = _sha256((output_root / calibrator_name).read_bytes())
        bundle_files[model_name] = model_file_sha256
        bundle_files[calibrator_name] = calibrator_file_sha256
        model_entries.append({"artifact_sha256": model_file_sha256, "path": model_name})
        calibrator_entries.append(
            {"artifact_sha256": calibrator_file_sha256, "path": calibrator_name}
        )

    thresholds = _thresholds_from_spec(arm_spec)
    switches = {
        name: bool(arm_spec.get(name, False))
        for name in ("disagreement", "graph", "model", "novelty", "rules", "trust")
    }
    spec = {
        "arm": str(arm_spec.get("arm")),
        "arm_spec_sha256": str(arm_spec.get("spec_sha256")),
        "calibrators": calibrator_entries,
        "catalog_feature_groups": arm_spec.get("catalog_feature_groups", []),
        "catalog_feature_names": arm_spec.get("catalog_feature_names", []),
        "feature_names": feature_names,
        "model_artifacts": model_entries,
        "switches": switches,
        "threshold_digest": str(arm_spec.get("threshold_digest")),
        "thresholds": thresholds,
    }
    _write_canonical(output_root / "spec.json", spec)
    bundle_files["spec.json"] = _sha256((output_root / "spec.json").read_bytes())

    selected_rows = select_demo_rows(rows, limit=scenario_limit)
    scenarios = {
        "arm": spec["arm"],
        "scenarios": [_scenario_from_row(row) for row in selected_rows],
        "schema_version": "apar-sentinel-v5-portable-demo-scenarios/1",
    }
    _write_canonical(output_root / "scenarios.json", scenarios)
    bundle_files["scenarios.json"] = _sha256((output_root / "scenarios.json").read_bytes())

    manifest = {
        "accepted_capacity_evidence": False,
        "authoritative": False,
        "bundle_files": dict(sorted(bundle_files.items())),
        "demo_only": True,
        "schema_version": _BUNDLE_SCHEMA,
        "source_checkpoint_manifest_sha256": source_checkpoint_manifest_sha256,
        "source_commit": source_commit,
    }
    manifest["manifest_sha256"] = _sha256(_canonical(manifest))
    _write_canonical(output_root / "manifest.json", manifest)
    return output_root


@dataclass(frozen=True, slots=True)
class PortableSentinelBundle:
    """Validated portable artifacts and loaded CatBoost members."""

    root: Path
    manifest: dict[str, Any]
    spec: dict[str, Any]
    models: tuple[CatBoostClassifier, ...]
    calibrators: tuple[dict[str, Any], ...]


def load_portable_bundle(root: Path) -> PortableSentinelBundle:
    """Load a demo bundle only after verifying its self-hash and every file."""
    root = root.resolve()
    manifest = _load_object(root / "manifest.json", label="bundle manifest")
    claimed_manifest_sha256 = manifest.pop("manifest_sha256", None)
    if claimed_manifest_sha256 != _sha256(_canonical(manifest)):
        raise ValueError("bundle manifest digest mismatch")
    manifest["manifest_sha256"] = claimed_manifest_sha256
    if (
        manifest.get("schema_version") != _BUNDLE_SCHEMA
        or manifest.get("demo_only") is not True
        or manifest.get("authoritative") is not False
        or manifest.get("accepted_capacity_evidence") is not False
    ):
        raise ValueError("bundle safety flags differ")
    raw_files = manifest.get("bundle_files")
    if not isinstance(raw_files, dict) or not raw_files:
        raise ValueError("bundle file index is empty")
    for raw_name, expected_digest in raw_files.items():
        if not isinstance(raw_name, str) or not isinstance(expected_digest, str):
            raise ValueError("bundle file index is malformed")
        relative = PurePosixPath(raw_name)
        if relative.is_absolute() or ".." in relative.parts or relative.as_posix() != raw_name:
            raise ValueError("bundle file path is unsafe")
        path = root.joinpath(*relative.parts)
        if path.is_symlink() or not path.is_file():
            raise ValueError("bundle file is missing")
        if _sha256(path.read_bytes()) != expected_digest:
            raise ValueError("bundle file digest mismatch")

    spec = _load_object(root / "spec.json", label="portable arm specification")
    model_entries = spec.get("model_artifacts")
    calibrator_entries = spec.get("calibrators")
    if (
        not isinstance(model_entries, list)
        or not model_entries
        or not isinstance(calibrator_entries, list)
        or len(calibrator_entries) != len(model_entries)
    ):
        raise ValueError("portable model member index is incomplete")
    models: list[CatBoostClassifier] = []
    calibrators: list[dict[str, Any]] = []
    for model_entry, calibrator_entry in zip(model_entries, calibrator_entries, strict=True):
        if not isinstance(model_entry, dict) or not isinstance(calibrator_entry, dict):
            raise ValueError("portable member entry is malformed")
        model_path = root / str(model_entry.get("path"))
        calibrator_path = root / str(calibrator_entry.get("path"))
        model = CatBoostClassifier()
        model.load_model(str(model_path), format="json")
        calibrator = _load_object(calibrator_path, label="portable calibrator")
        x_thresholds = calibrator.get("x_thresholds")
        y_thresholds = calibrator.get("y_thresholds")
        if (
            not isinstance(x_thresholds, list)
            or not isinstance(y_thresholds, list)
            or len(x_thresholds) < 2
            or len(x_thresholds) != len(y_thresholds)
        ):
            raise ValueError("portable calibrator knots are invalid")
        models.append(model)
        calibrators.append(calibrator)
    return PortableSentinelBundle(
        root=root,
        manifest=manifest,
        spec=spec,
        models=tuple(models),
        calibrators=tuple(calibrators),
    )


def _probability_action(probability: float, thresholds: dict[str, float]) -> SentinelAction:
    if probability >= thresholds["model_decline"]:
        return SentinelAction.DECLINE_HOLD
    if probability >= thresholds["model_review"]:
        return SentinelAction.REVIEW_HOLD
    if probability >= thresholds["model_challenge"]:
        return SentinelAction.CHALLENGE
    return SentinelAction.APPROVE


def score_portable_scenario(
    bundle: PortableSentinelBundle,
    scenario: dict[str, Any],
) -> dict[str, Any]:
    """Run the accepted CatBoost members and frozen calibration for one scenario."""
    started = time.perf_counter_ns()
    raw_features = scenario.get("features")
    feature_names = bundle.spec.get("feature_names")
    if not isinstance(raw_features, list) or not isinstance(feature_names, list):
        raise ValueError("scenario features or bundle feature order is absent")
    features = np.asarray(raw_features, dtype=np.float64)
    if features.ndim != 1 or len(features) != len(feature_names) or not np.isfinite(features).all():
        raise ValueError("scenario feature vector differs from the bundle")
    raw_scores = tuple(
        float(model.predict_proba(features.reshape(1, -1))[0, 1]) for model in bundle.models
    )
    calibrated_scores = tuple(
        float(
            np.interp(
                raw_score,
                np.asarray(calibrator["x_thresholds"], dtype=np.float64),
                np.asarray(calibrator["y_thresholds"], dtype=np.float64),
            )
        )
        for raw_score, calibrator in zip(raw_scores, bundle.calibrators, strict=True)
    )
    probability = float(np.mean(calibrated_scores))
    disagreement = float(np.std(calibrated_scores))
    raw_thresholds = bundle.spec.get("thresholds")
    if not isinstance(raw_thresholds, dict):
        raise ValueError("portable model thresholds are absent")
    thresholds = {str(name): float(value) for name, value in raw_thresholds.items()}
    model_action = _probability_action(probability, thresholds)
    latency_ms = (time.perf_counter_ns() - started) / 1_000_000
    if not math.isfinite(latency_ms):
        raise RuntimeError("portable scoring latency is not finite")
    return {
        "arm": str(bundle.spec.get("arm")),
        "calibrated_member_scores": list(calibrated_scores),
        "calibrated_probability": probability,
        "disagreement": disagreement,
        "event_id": str(scenario.get("event_id")),
        "final_action": model_action.value,
        "latency_ms": latency_ms,
        "model_action": model_action.value,
        "raw_member_scores": list(raw_scores),
        "reason_codes": [f"model_probability_{model_action.value}"],
    }


def _require_probability_replay(
    *,
    trace: dict[str, Any],
    evidence: dict[str, Any],
    probability_tolerance: float,
) -> None:
    expected_probability = evidence.get("probability")
    expected_action = evidence.get("probability_action")
    if not isinstance(expected_probability, (int, float)) or not isinstance(expected_action, str):
        raise ValueError("scenario accepted replay evidence is incomplete")
    if (
        abs(float(trace["calibrated_probability"]) - float(expected_probability))
        > probability_tolerance
    ):
        raise ValueError("portable probability replay differs from accepted evidence")
    if trace["model_action"] != expected_action:
        raise ValueError("portable action replay differs from accepted evidence")
    for evidence_name, trace_name in (
        ("model_raw_scores", "raw_member_scores"),
        ("model_calibrated_scores", "calibrated_member_scores"),
    ):
        expected_scores = evidence.get(evidence_name)
        if expected_scores is None:
            continue
        actual_scores = trace[trace_name]
        if (
            not isinstance(expected_scores, list)
            or len(expected_scores) != len(actual_scores)
            or any(
                abs(float(actual) - float(expected)) > probability_tolerance
                for actual, expected in zip(actual_scores, expected_scores, strict=True)
            )
        ):
            raise ValueError(f"portable {evidence_name} replay differs")


def _scenario_metrics(
    scenarios: list[dict[str, Any]], traces: list[dict[str, Any]]
) -> dict[str, Any]:
    fraud_count = 0
    fraud_intervened = 0
    benign_count = 0
    benign_intervened = 0
    fraud_value = 0.0
    captured_value = 0.0
    action_counts = {action.value: 0 for action in SentinelAction}
    for scenario, trace in zip(scenarios, traces, strict=True):
        truth = scenario.get("presentation_ground_truth")
        if not isinstance(truth, dict) or truth.get("label") not in (0, 1):
            raise ValueError("scenario presentation ground truth is incomplete")
        label = int(truth["label"])
        action = str(trace["final_action"])
        if action not in action_counts:
            raise ValueError("portable trace action is unknown")
        action_counts[action] += 1
        intervened = action != SentinelAction.APPROVE.value
        amount = float(truth.get("amount", 0.0))
        if not math.isfinite(amount) or amount < 0:
            raise ValueError("scenario presentation amount is invalid")
        if label == 1:
            fraud_count += 1
            fraud_value += amount
            fraud_intervened += int(intervened)
            captured_value += amount if intervened else 0.0
        else:
            benign_count += 1
            benign_intervened += int(intervened)
    return {
        "action_counts": action_counts,
        "benign_scenarios": benign_count,
        "captured_value_proxy": (captured_value / fraud_value if fraud_value > 0 else None),
        "fraud_scenarios": fraud_count,
        "legitimate_friction_rate": (benign_intervened / benign_count if benign_count else None),
        "scenario_recall": (fraud_intervened / fraud_count if fraud_count else None),
    }


def run_portable_scenarios(
    *,
    bundle_root: Path,
    scenario_path: Path,
    output_path: Path | None = None,
    probability_tolerance: float = 1e-12,
) -> dict[str, Any]:
    """Verify a bundle and fail closed unless every accepted scenario replays."""
    if probability_tolerance < 0 or not math.isfinite(probability_tolerance):
        raise ValueError("portable replay tolerance is invalid")
    load_started = time.perf_counter_ns()
    bundle = load_portable_bundle(bundle_root)
    model_load_ms = (time.perf_counter_ns() - load_started) / 1_000_000
    scenario_document = _load_object(scenario_path, label="portable scenarios")
    scenarios = scenario_document.get("scenarios")
    if (
        scenario_document.get("schema_version") != "apar-sentinel-v5-portable-demo-scenarios/1"
        or scenario_document.get("arm") != bundle.spec.get("arm")
        or not isinstance(scenarios, list)
        or not scenarios
        or any(not isinstance(scenario, dict) for scenario in scenarios)
    ):
        raise ValueError("portable scenarios differ from the bundle")
    typed_scenarios = cast(list[dict[str, Any]], scenarios)
    scoring_started = time.perf_counter_ns()
    traces: list[dict[str, Any]] = []
    for scenario in typed_scenarios:
        trace = score_portable_scenario(bundle, scenario)
        evidence = scenario.get("accepted_checkpoint_evidence")
        if not isinstance(evidence, dict):
            raise ValueError("scenario accepted replay evidence is absent")
        _require_probability_replay(
            trace=trace,
            evidence=evidence,
            probability_tolerance=probability_tolerance,
        )
        trace["presentation_ground_truth"] = scenario.get("presentation_ground_truth")
        trace["replay_probability_abs_error"] = abs(
            float(trace["calibrated_probability"]) - float(evidence["probability"])
        )
        traces.append(trace)
    scoring_wall_ms = (time.perf_counter_ns() - scoring_started) / 1_000_000
    report: dict[str, Any] = {
        "accepted_capacity_evidence": False,
        "authoritative": False,
        "bundle_manifest_sha256": bundle.manifest["manifest_sha256"],
        "demo_only": True,
        "metrics": _scenario_metrics(typed_scenarios, traces),
        "model_load_ms": model_load_ms,
        "probability_tolerance": probability_tolerance,
        "replay_verified": True,
        "schema_version": "apar-sentinel-v5-portable-demo-trace/1",
        "scoring_wall_ms": scoring_wall_ms,
        "traces": traces,
    }
    report["trace_sha256"] = _sha256(_canonical(report))
    if output_path is not None:
        _write_canonical(output_path, report)
    return report


def select_demo_rows(rows: Iterable[dict[str, Any]], *, limit: int = 12) -> list[dict[str, Any]]:
    """Select deterministic accepted rows with required family/rail/action coverage."""
    if limit < len(_FRAUD_FAMILIES) + len(_BENIGN_RAILS) + len(_FRICTION_ACTIONS):
        raise ValueError("demo row limit cannot satisfy required coverage")
    requirements = tuple(
        [("family", family) for family in _FRAUD_FAMILIES]
        + [("rail", rail) for rail in _BENIGN_RAILS]
        + [("action", action) for action in _FRICTION_ACTIONS]
    )
    pools: dict[tuple[str, str], list[dict[str, Any]]] = {
        requirement: [] for requirement in requirements
    }
    fillers: list[dict[str, Any]] = []

    def retain(pool: list[dict[str, Any]], row: dict[str, Any]) -> None:
        pool.append(row)
        pool.sort(key=lambda item: str(item["support"]["event_id"]))
        del pool[limit:]

    for row in rows:
        support = row.get("support")
        if not isinstance(support, dict) or not isinstance(support.get("event_id"), str):
            raise ValueError("demo row support is malformed")
        retain(fillers, row)
        label = int(support.get("label", -1))
        family = str(support.get("family"))
        rail = str(support.get("rail")).lower()
        action = str(row.get("probability_action"))
        if label == 1 and ("family", family) in pools:
            retain(pools[("family", family)], row)
        if label == 0 and action == SentinelAction.APPROVE.value and ("rail", rail) in pools:
            retain(pools[("rail", rail)], row)
        if ("action", action) in pools:
            retain(pools[("action", action)], row)

    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()

    def take(requirement: tuple[str, str]) -> None:
        for row in pools[requirement]:
            event_id = str(row["support"]["event_id"])
            if event_id not in selected_ids:
                selected.append(row)
                selected_ids.add(event_id)
                return
        raise ValueError(f"demo rows do not satisfy {requirement[0]}={requirement[1]}")

    for requirement in requirements:
        take(requirement)
    for row in fillers:
        if len(selected) >= limit:
            break
        event_id = str(row["support"]["event_id"])
        if event_id not in selected_ids:
            selected.append(row)
            selected_ids.add(event_id)
    if len(selected) < limit:
        raise ValueError("demo row source is smaller than requested limit")
    return selected


__all__ = [
    "PortableSentinelBundle",
    "export_portable_bundle",
    "load_portable_bundle",
    "run_portable_scenarios",
    "score_portable_scenario",
    "select_demo_rows",
]
