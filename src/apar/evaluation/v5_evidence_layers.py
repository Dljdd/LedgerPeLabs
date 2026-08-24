"""Canonical deterministic-core and observational-latency evidence layers."""

from __future__ import annotations

import hashlib
import json
import platform
import sys
import time
from importlib.metadata import version
from typing import Any, Final, cast

import numpy as np

DETERMINISTIC_CORE_SCHEMA: Final = "apar-sentinel-v5-deterministic-core/1"
LOCKED_DETERMINISTIC_CORE_SCHEMA: Final = (
    "apar-sentinel-v5-locked-deterministic-core/2"
)
OBSERVATIONAL_LATENCY_SCHEMA: Final = "apar-sentinel-v5-observational-latency/1"
OBSERVATIONAL_MEASUREMENT_METHOD: Final = "time.perf_counter_ns-elapsed-v1"

DETERMINISTIC_CORE_EXCLUSION_SCHEMA: Final = (
    (
        "arm_results[*].row_evidence[*]",
        ("latency_ms", "row_output_sha256"),
        "deterministic_row_sha256",
    ),
    (
        "arm_results[*]",
        (
            "p50_latency_ms",
            "p95_latency_ms",
            "p99_latency_ms",
            "score_sha256",
            "result_sha256",
        ),
        "deterministic_result_sha256",
    ),
    (
        "complete_metrics[*].aggregate",
        ("p50_latency_ms", "p95_latency_ms", "p99_latency_ms"),
        "observational_latency.arm_observations",
    ),
    (
        "complete_metrics[*]",
        ("arm_result_sha256", "complete_metrics_sha256"),
        "deterministic_complete_metrics_sha256",
    ),
    (
        "controls.controls[*]",
        ("control_sha256", "row_evidence_sha256"),
        "deterministic_control_sha256",
    ),
    (
        "controls.invariance.row_evidence",
        ("before_score_sha256", "after_score_sha256"),
        "before_score_signature+after_score_signature",
    ),
    (
        "controls.workload.row_evidence",
        ("full_score_sha256", "arms[*].score_sha256", "arms[*].rows[*].latency_ms"),
        "deterministic_control_row_evidence_sha256",
    ),
    (
        "controls.workload.measurements",
        ("*p95_latency_ms",),
        "observational_latency.control_observations",
    ),
    (
        "controls",
        ("suite_sha256",),
        "deterministic_control_suite_sha256",
    ),
    (
        "readiness",
        (
            "gates[p95_latency_ms]",
            "qualifying_controls[*].control_sha256",
            "readiness_sha256",
        ),
        "deterministic_readiness_sha256+observational_latency.latency_gate",
    ),
    (
        "payload",
        ("observational_latency", "payload_sha256"),
        "deterministic_core.core_sha256",
    ),
)
LOCKED_DETERMINISTIC_CORE_EXCLUSION_SCHEMA: Final = (
    *DETERMINISTIC_CORE_EXCLUSION_SCHEMA,
    (
        "locked_payload",
        ("attempt_receipt_sha256",),
        "durable_attempt_receipt.receipt_sha256",
    ),
)

_ARMS: Final = (
    "rules_only",
    "ensemble_no_graph",
    "ensemble_with_graph",
    "full_sentinel",
)
_WORKLOAD_CONTROLS: Final = ("benign_only", "fraud_only_diagnostic")
_INVARIANCE_CONTROLS: Final = (
    "identity_rename",
    "future_causality",
    "equal_time_isolation",
    "feature_leakage",
)
_ROW_FIELDS: Final = (
    "support",
    "catalog_feature_values",
    "subset_feature_values",
    "catalog_feature_sha256",
    "subset_feature_sha256",
    "model_raw_scores",
    "model_calibrated_scores",
    "threshold_trace",
    "rule_components",
    "rule_manifest_sha256",
    "rule_vector_sha256",
    "rule_source_event_ids",
    "rule_max_source_available_at",
    "rule_evidence_source_ids",
    "action",
    "probability",
    "probability_action",
    "model_action",
    "rule_action",
    "trust_action",
    "rule_score",
    "trust_routed",
    "novelty_score",
    "novelty_raw_score",
    "novelty_overridden",
    "disagreement",
    "novelty_routed",
    "disagreement_routed",
    "latency_ms",
    "arm_spec_sha256",
    "row_output_sha256",
)
_RESULT_FIELDS: Final = (
    "arm",
    "recall",
    "precision",
    "f1",
    "pr_auc",
    "roc_auc",
    "brier",
    "expected_calibration_error",
    "false_decline_rate",
    "challenge_rate",
    "review_rate",
    "captured_value_fraction",
    "escaped_value_fraction",
    "p50_latency_ms",
    "p95_latency_ms",
    "p99_latency_ms",
    "support_total",
    "support_fraud",
    "support_legitimate",
    "arm_spec_sha256",
    "support_sha256",
    "feature_count",
    "arm_spec",
    "execution_artifact_refs",
    "row_evidence",
    "score_sha256",
    "result_sha256",
)
_CONTROL_FIELDS: Final = (
    "name",
    "executed",
    "qualifies_for_readiness",
    "spec_json",
    "spec_sha256",
    "input_support_ids",
    "input_support_sha256",
    "input_artifact_ids",
    "input_artifact_sha256",
    "permutation_seed",
    "executed_arm_spec_sha256",
    "measurements",
    "criterion",
    "passed",
    "row_evidence_json",
    "row_evidence_sha256",
    "implementation_sha256",
    "control_sha256",
)
_CONTROL_SUITE_FIELDS: Final = (
    "controls",
    "evidence_protocol_sha256",
    "support_sha256",
    "implementation_sha256",
    "suite_sha256",
)


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode()


def _digest(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _mapping(value: object, label: str) -> dict[str, Any]:
    if type(value) is not dict or any(type(key) is not str for key in value):
        raise ValueError(f"{label} must be an object")
    return cast(dict[str, Any], value)


def _exact_fields(document: dict[str, Any], fields: tuple[str, ...], label: str) -> None:
    if set(document) != set(fields):
        raise ValueError(f"{label} schema fields differ")


def _stable_row(row: object) -> dict[str, Any]:
    document = _mapping(row, "arm row")
    _exact_fields(document, _ROW_FIELDS, "arm row")
    stable = {
        field: document[field]
        for field in _ROW_FIELDS
        if field not in {"latency_ms", "row_output_sha256"}
    }
    stable["deterministic_row_sha256"] = _digest(stable)
    return stable


def _stable_result(result: object) -> dict[str, Any]:
    document = _mapping(result, "retained arm result")
    _exact_fields(document, _RESULT_FIELDS, "retained arm result")
    stable = {
        field: document[field]
        for field in _RESULT_FIELDS
        if field
        not in {
            "arm_spec",
            "row_evidence",
            "p50_latency_ms",
            "p95_latency_ms",
            "p99_latency_ms",
            "score_sha256",
            "result_sha256",
        }
    }
    stable_rows = [_stable_row(row) for row in document["row_evidence"]]
    stable["row_evidence_addresses"] = [
        {
            "event_id": row["support"]["event_id"],
            "deterministic_row_sha256": row["deterministic_row_sha256"],
        }
        for row in stable_rows
    ]
    stable["deterministic_row_stream_sha256"] = _digest(stable_rows)
    stable["deterministic_result_sha256"] = _digest(stable)
    return stable


def _stable_complete_metrics(
    metrics: object, *, deterministic_result_sha256: str
) -> dict[str, Any]:
    document = _mapping(metrics, "complete metrics")
    expected = (
        "arm",
        "arm_result_sha256",
        "support_sha256",
        "aggregate",
        "calibration",
        "economics",
        "by_family",
        "bootstrap",
        "complete_metrics_sha256",
    )
    _exact_fields(document, expected, "complete metrics")
    aggregate = _mapping(document["aggregate"], "aggregate metrics")
    latency_names = {"p50_latency_ms", "p95_latency_ms", "p99_latency_ms"}
    if not latency_names <= set(aggregate):
        raise ValueError("complete metrics lack exact observational latency metrics")
    stable = {
        "arm": document["arm"],
        "deterministic_result_sha256": deterministic_result_sha256,
        "support_sha256": document["support_sha256"],
        "aggregate": {
            name: value for name, value in aggregate.items() if name not in latency_names
        },
        "calibration_sha256": document["calibration"]["calibration_sha256"],
        "economics_sha256": document["economics"]["economics_sha256"],
        "family_sha256": [item["family_sha256"] for item in document["by_family"]],
        "bootstrap_sha256": document["bootstrap"]["bootstrap_sha256"],
    }
    stable["deterministic_complete_metrics_sha256"] = _digest(stable)
    return stable


def _stable_control_row_evidence(name: str, raw_json: object) -> dict[str, Any]:
    if type(raw_json) is not str:
        raise ValueError("control row evidence must be canonical JSON")
    try:
        document = _mapping(json.loads(raw_json), "control row evidence")
    except json.JSONDecodeError as error:
        raise ValueError("control row evidence must be JSON") from error
    if raw_json != json.dumps(document, sort_keys=True):
        raise ValueError("control row evidence must be canonical JSON")
    if name in _INVARIANCE_CONTROLS:
        required = {"before_score_sha256", "after_score_sha256"}
        if not required <= set(document):
            raise ValueError("invariance control score bindings are incomplete")
        return {key: value for key, value in document.items() if key not in required}
    if name in _WORKLOAD_CONTROLS:
        if set(document) != {"arms", "full_score_sha256"}:
            raise ValueError("workload control row evidence schema differs")
        arms = _mapping(document["arms"], "workload control arms")
        if set(arms) != set(_ARMS):
            raise ValueError("workload control arm set differs")
        stable_arms: dict[str, Any] = {}
        for arm in _ARMS:
            arm_document = _mapping(arms[arm], "workload control arm")
            if "score_sha256" not in arm_document or "rows" not in arm_document:
                raise ValueError("workload control arm evidence is incomplete")
            rows = []
            for raw_row in arm_document["rows"]:
                row = _mapping(raw_row, "workload control row")
                if set(row) != {"event_id", "action", "probability", "latency_ms"}:
                    raise ValueError("workload control row schema differs")
                rows.append(
                    {
                        "event_id": row["event_id"],
                        "action": row["action"],
                        "probability": row["probability"],
                    }
                )
            stable_arms[arm] = {
                key: value
                for key, value in arm_document.items()
                if key not in {"score_sha256", "rows"}
            }
            stable_arms[arm]["rows"] = rows
        return {"arms": stable_arms}
    if name == "label_shuffle":
        return document
    raise ValueError(f"unknown executed control: {name}")


def _stable_controls(controls: object) -> tuple[dict[str, Any], dict[str, str]]:
    suite = _mapping(controls, "executed controls")
    _exact_fields(suite, _CONTROL_SUITE_FIELDS, "executed controls")
    stable_controls = []
    digests: dict[str, str] = {}
    for raw in suite["controls"]:
        control = _mapping(raw, "executed control")
        _exact_fields(control, _CONTROL_FIELDS, "executed control")
        name = str(control["name"])
        stable = {
            field: control[field]
            for field in _CONTROL_FIELDS
            if field
            not in {
                "control_sha256",
                "row_evidence_json",
                "row_evidence_sha256",
                "measurements",
            }
        }
        stable["measurements"] = [
            measurement
            for measurement in control["measurements"]
            if not str(measurement["name"]).endswith("p95_latency_ms")
        ]
        row_evidence = _stable_control_row_evidence(
            name, control["row_evidence_json"]
        )
        stable["deterministic_row_evidence_sha256"] = _digest(row_evidence)
        stable["deterministic_control_sha256"] = _digest(stable)
        digests[name] = stable["deterministic_control_sha256"]
        stable_controls.append(stable)
    stable_suite = {
        "controls": stable_controls,
        "evidence_protocol_sha256": suite["evidence_protocol_sha256"],
        "support_sha256": suite["support_sha256"],
        "implementation_sha256": suite["implementation_sha256"],
    }
    stable_suite["deterministic_control_suite_sha256"] = _digest(stable_suite)
    return stable_suite, digests


def _stable_readiness(
    readiness: object, *, deterministic_control_digests: dict[str, str]
) -> dict[str, Any]:
    document = _mapping(readiness, "readiness")
    expected = {
        "evaluated_arm",
        "gates",
        "qualifying_controls",
        "status",
        "readiness_sha256",
    }
    if set(document) != expected:
        raise ValueError("readiness schema differs")
    deterministic_gates = [
        gate for gate in document["gates"] if gate["metric"] != "p95_latency_ms"
    ]
    if len(deterministic_gates) + 1 != len(document["gates"]):
        raise ValueError("readiness must contain exactly one observational latency gate")
    controls = []
    for name, passed, _observed_digest in document["qualifying_controls"]:
        if name not in deterministic_control_digests:
            raise ValueError("readiness references an unknown qualifying control")
        controls.append([name, passed, deterministic_control_digests[name]])
    deterministic_status = (
        "ready"
        if all(gate["passed"] for gate in deterministic_gates)
        and all(item[1] for item in controls)
        else "not_ready"
    )
    stable = {
        "evaluated_arm": document["evaluated_arm"],
        "gates": deterministic_gates,
        "qualifying_controls": controls,
        "deterministic_status": deterministic_status,
    }
    stable["deterministic_readiness_sha256"] = _digest(stable)
    return stable


def build_deterministic_core_document(
    *,
    safe_seed: int,
    evidence_protocol: object,
    catalog_sha256: str,
    execution_artifacts: list[dict[str, Any]],
    arm_results: list[dict[str, Any]],
    complete_metrics: list[dict[str, Any]],
    controls: dict[str, Any],
    readiness: dict[str, Any],
) -> dict[str, Any]:
    """Build the exact versioned semantic commitment without observed timing."""
    if safe_seed != 404:
        raise ValueError("deterministic safe core requires seed 404")
    if len(arm_results) != len(_ARMS) or len(complete_metrics) != len(_ARMS):
        raise ValueError("deterministic safe core requires exact four arms")
    stable_results = [_stable_result(result) for result in arm_results]
    if [result["arm"] for result in stable_results] != list(_ARMS):
        raise ValueError("deterministic safe core arm order differs")
    stable_metrics = [
        _stable_complete_metrics(
            metrics,
            deterministic_result_sha256=result["deterministic_result_sha256"],
        )
        for metrics, result in zip(complete_metrics, stable_results, strict=True)
    ]
    stable_controls, control_digests = _stable_controls(controls)
    stable_readiness = _stable_readiness(
        readiness, deterministic_control_digests=control_digests
    )
    artifact_addresses = []
    for artifact in execution_artifacts:
        if set(artifact) != {
            "evidence_sha256",
            "artifact_sha256",
            "payload_sha256",
            "payload_json",
        }:
            raise ValueError("execution artifact schema differs")
        artifact_addresses.append(
            {
                "evidence_sha256": artifact["evidence_sha256"],
                "artifact_sha256": artifact["artifact_sha256"],
                "payload_sha256": artifact["payload_sha256"],
            }
        )
    return {
        "schema_version": DETERMINISTIC_CORE_SCHEMA,
        "exclusion_schema": DETERMINISTIC_CORE_EXCLUSION_SCHEMA,
        "safe_seed": safe_seed,
        "evidence_protocol": evidence_protocol,
        "catalog_sha256": catalog_sha256,
        "execution_artifact_addresses": artifact_addresses,
        "arm_results": stable_results,
        "complete_metrics": stable_metrics,
        "controls": stable_controls,
        "readiness": stable_readiness,
    }


def deterministic_core_sha256(**kwargs: object) -> str:
    """Return the canonical deterministic-core content address."""
    return _digest(build_deterministic_core_document(**kwargs))  # type: ignore[arg-type]


def build_locked_deterministic_core_document(
    *,
    run_binding: dict[str, Any],
    evidence_protocol: object,
    catalog_sha256: str,
    execution_artifacts: list[dict[str, Any]],
    arm_results: list[dict[str, Any]],
    complete_metrics: list[dict[str, Any]],
    controls: dict[str, Any],
    readiness: dict[str, Any],
) -> dict[str, Any]:
    """Build the versioned locked core with the full immutable run contract."""
    if (
        run_binding.get("mode") != "locked_development"
        or run_binding.get("profile") != "production"
        or run_binding.get("development_test_seed") != 2404
    ):
        raise ValueError("locked deterministic core requires production seed 2404")
    if len(arm_results) != len(_ARMS) or len(complete_metrics) != len(_ARMS):
        raise ValueError("locked deterministic core requires exact four arms")
    stable_results = [_stable_result(result) for result in arm_results]
    if [result["arm"] for result in stable_results] != list(_ARMS):
        raise ValueError("locked deterministic core arm order differs")
    stable_metrics = [
        _stable_complete_metrics(
            metrics,
            deterministic_result_sha256=result["deterministic_result_sha256"],
        )
        for metrics, result in zip(complete_metrics, stable_results, strict=True)
    ]
    stable_controls, control_digests = _stable_controls(controls)
    stable_readiness = _stable_readiness(
        readiness, deterministic_control_digests=control_digests
    )
    artifact_addresses = []
    for artifact in execution_artifacts:
        if set(artifact) != {
            "evidence_sha256",
            "artifact_sha256",
            "payload_sha256",
            "payload_json",
        }:
            raise ValueError("locked execution artifact schema differs")
        artifact_addresses.append(
            {
                "evidence_sha256": artifact["evidence_sha256"],
                "artifact_sha256": artifact["artifact_sha256"],
                "payload_sha256": artifact["payload_sha256"],
            }
        )
    return {
        "schema_version": LOCKED_DETERMINISTIC_CORE_SCHEMA,
        "exclusion_schema": LOCKED_DETERMINISTIC_CORE_EXCLUSION_SCHEMA,
        "run_binding": run_binding,
        "evidence_protocol": evidence_protocol,
        "catalog_sha256": catalog_sha256,
        "execution_artifact_addresses": artifact_addresses,
        "arm_results": stable_results,
        "complete_metrics": stable_metrics,
        "controls": stable_controls,
        "readiness": stable_readiness,
    }


def current_latency_environment() -> dict[str, Any]:
    """Declare the exact runtime and monotonic timer used for observations."""
    clock = time.get_clock_info("perf_counter")
    return {
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
        "platform_system": platform.system(),
        "platform_release": platform.release(),
        "platform_machine": platform.machine(),
        "executable_abi": sys.implementation.cache_tag,
        "catboost_version": version("catboost"),
        "numpy_version": version("numpy"),
        "scikit_learn_version": version("scikit-learn"),
        "clock": {
            "implementation": clock.implementation,
            "monotonic": clock.monotonic,
            "adjustable": clock.adjustable,
            "resolution_seconds": clock.resolution,
        },
    }


def _percentiles(samples: list[float]) -> tuple[float, float, float]:
    values = np.asarray(samples, dtype=float)
    return (
        float(np.percentile(values, 50)),
        float(np.percentile(values, 95)),
        float(np.percentile(values, 99)),
    )


def build_observational_latency_document(
    *,
    deterministic_core_sha256_value: str,
    arm_results: list[dict[str, Any]],
    complete_metrics: list[dict[str, Any]],
    controls: dict[str, Any],
    readiness: dict[str, Any],
) -> dict[str, Any]:
    """Retain real aligned timing samples, metrics, environment, and latency gate."""
    observations = []
    for result, metrics in zip(arm_results, complete_metrics, strict=True):
        arm = str(result["arm"])
        rows = result["row_evidence"]
        samples = [
            {
                "row_index": index,
                "event_id": row["support"]["event_id"],
                "latency_ms": row["latency_ms"],
            }
            for index, row in enumerate(rows)
        ]
        values = [float(sample["latency_ms"]) for sample in samples]
        if len(values) < 2 or any(value <= 0.0 for value in values):
            raise ValueError("observational arm latency samples are invalid")
        if len(set(values)) == 1:
            raise ValueError("constant arm latency samples are not observational evidence")
        p50, p95, p99 = _percentiles(values)
        aggregate = metrics["aggregate"]
        observations.append(
            {
                "arm": arm,
                "support_sha256": result["support_sha256"],
                "samples": samples,
                "p50_latency_ms": p50,
                "p95_latency_ms": p95,
                "p99_latency_ms": p99,
                "reported_metric_sha256": [
                    aggregate[name]["metric_sha256"]
                    for name in (
                        "p50_latency_ms",
                        "p95_latency_ms",
                        "p99_latency_ms",
                    )
                ],
            }
        )
    if [item["arm"] for item in observations] != list(_ARMS):
        raise ValueError("observational arm order differs")

    control_suite = _mapping(controls, "executed controls")
    by_name = {str(item["name"]): item for item in control_suite["controls"]}
    control_observations = []
    for control_name in _WORKLOAD_CONTROLS:
        control = by_name.get(control_name)
        if control is None:
            raise ValueError("workload latency control is missing")
        evidence = json.loads(control["row_evidence_json"])
        for arm in _ARMS:
            rows = evidence["arms"][arm]["rows"]
            samples = [
                {
                    "row_index": index,
                    "event_id": row["event_id"],
                    "latency_ms": row["latency_ms"],
                }
                for index, row in enumerate(rows)
            ]
            values = [float(sample["latency_ms"]) for sample in samples]
            if len(values) < 2 or any(value <= 0.0 for value in values):
                raise ValueError("observational control latency samples are invalid")
            if len(set(values)) == 1:
                raise ValueError(
                    "constant control latency samples are not observational evidence"
                )
            control_observations.append(
                {
                    "control": control_name,
                    "arm": arm,
                    "samples": samples,
                    "p95_latency_ms": float(np.percentile(values, 95)),
                }
            )

    latency_gates = [
        gate for gate in readiness["gates"] if gate["metric"] == "p95_latency_ms"
    ]
    if len(latency_gates) != 1:
        raise ValueError("observational latency gate is missing or duplicated")
    environment = current_latency_environment()
    document = {
        "schema_version": OBSERVATIONAL_LATENCY_SCHEMA,
        "measurement_method": OBSERVATIONAL_MEASUREMENT_METHOD,
        "synthetic_values": False,
        "deterministic_core_sha256": deterministic_core_sha256_value,
        "environment": environment,
        "environment_sha256": _digest(environment),
        "arm_observations": observations,
        "control_observations": control_observations,
        "latency_gate": latency_gates[0],
    }
    document["observational_latency_sha256"] = _digest(document)
    return document


__all__ = [
    "DETERMINISTIC_CORE_EXCLUSION_SCHEMA",
    "DETERMINISTIC_CORE_SCHEMA",
    "LOCKED_DETERMINISTIC_CORE_SCHEMA",
    "LOCKED_DETERMINISTIC_CORE_EXCLUSION_SCHEMA",
    "OBSERVATIONAL_LATENCY_SCHEMA",
    "build_deterministic_core_document",
    "build_locked_deterministic_core_document",
    "build_observational_latency_document",
    "current_latency_environment",
    "deterministic_core_sha256",
]
