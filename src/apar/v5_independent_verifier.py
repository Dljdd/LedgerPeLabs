"""Offline independent verifier for retained Sentinel v5 evidence.

This module intentionally does not import the production runner, evaluation,
metric, gate, simulator, rail, or TrustVerifier implementations.
"""

from __future__ import annotations

import base64
import hashlib
import json
import math
import platform
import stat
import subprocess
import sys
import time
import zlib
from collections import defaultdict
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from decimal import Decimal
from importlib.metadata import version
from pathlib import Path
from typing import Any, NoReturn, cast

import numpy as np
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

_ARMS = (
    "rules_only",
    "ensemble_no_graph",
    "ensemble_with_graph",
    "full_sentinel",
)
_ACTIONS = ("approve", "challenge", "review_hold", "decline_hold")
_DETECTION = frozenset({"challenge", "review_hold", "decline_hold"})
_SEVERITY = {action: index for index, action in enumerate(_ACTIONS)}
_FAMILIES = (
    "agentic_intent_abuse",
    "app_scam_mule",
    "card_testing_cnp",
    "synthetic_merchant_refund",
)
_EVENT_KIND = {
    "a2a.initiate": "transfer_initiated",
    "a2a.accept": "transfer_accepted",
    "a2a.reject": "transfer_rejected",
    "a2a.post": "transfer_posted",
    "a2a.report": "fraud_reported",
    "a2a.freeze": "funds_frozen",
    "a2a.recover": "recovery",
    "a2a.return": "transfer_returned",
    "card.authorize": "authorization",
    "card.decline": "authorization_declined",
    "card.clear": "clearing",
    "card.settle": "settlement",
    "card.reverse": "reversal",
    "card.report": "fraud_reported",
    "card.dispute": "dispute_opened",
    "card.chargeback": "chargeback",
    "card.recover": "recovery",
    "card.refund": "refund",
}
_EXECUTION_ARTIFACT_FIELDS = {
    "evidence_sha256",
    "artifact_sha256",
    "payload_sha256",
    "payload_json",
}
_MANIFEST_FIELDS = {
    "evidence_sha256",
    "artifact_sha256",
    "campaign_id",
    "family",
    "rail",
    "lineage",
    "ledger_entry_ids",
    "trust_request_ids",
    "trust_failure_event_ids",
    "account_ids",
    "opening_balances",
    "device_ids",
    "credential_ids",
    "merchant_ids",
    "payee_ids",
    "agent_ids",
    "key_ids",
    "mandate_ids",
    "authentication_evidence_ids",
    "event_records",
    "ledger_postings",
    "trust_records",
    "trust_registry",
}
_SUPPORT_FIELDS = {
    "event_id",
    "label",
    "payment_id",
    "campaign_id",
    "actor_id",
    "counterparty_id",
    "amount",
    "currency",
    "family",
    "rail",
    "integrity_status",
    "source_command_id",
    "source_event_id",
    "execution_evidence_sha256",
}
_TRAINING_FIELDS = {
    "partition",
    "ordered_event_ids",
    "labels",
    "feature_names",
    "feature_matrix",
    "support_records",
    "execution_artifacts",
    "catalog_sha256",
    "feature_batch_sha256",
    "feature_batch_payload_json",
    "feature_matrix_sha256",
    "ordered_rows_sha256",
    "ordered_support_sha256",
}
_SPEC_FIELDS = {
    "arm",
    "catalog_feature_names",
    "catalog_feature_groups",
    "feature_names",
    "graph_feature_names",
    "non_graph_feature_names",
    "catalog_sha256",
    "model_seeds",
    "calibration_method",
    "threshold_source_partition",
    "threshold_method",
    "threshold_digest",
    "threshold_values",
    "component_parameters",
    "bootstrap_seed",
    "execution_bound",
    "training_partitions",
    "model_artifact_sha256",
    "model_artifacts",
    "calibrator_artifact_sha256",
    "calibrator_manifests",
    "novelty_artifact_sha256",
    "novelty_manifest",
    "model",
    "graph",
    "rules",
    "trust",
    "novelty",
    "disagreement",
    "implementation_version",
    "implementation_sha256",
    "arm_config_sha256",
    "protocol_sha256",
    "spec_sha256",
}
_ROW_FIELDS = {
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
}
_RESULT_FIELDS = {
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
    "execution_artifacts",
    "row_evidence",
    "score_sha256",
    "result_sha256",
}
_DETERMINISTIC_CORE_SCHEMA = "apar-sentinel-v5-deterministic-core/1"
_OBSERVATIONAL_LATENCY_SCHEMA = "apar-sentinel-v5-observational-latency/1"
_OBSERVATIONAL_MEASUREMENT_METHOD = "time.perf_counter_ns-elapsed-v1"
_DETERMINISTIC_CORE_EXCLUSION_SCHEMA = (
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
_LOCKED_DETERMINISTIC_CORE_EXCLUSION_SCHEMA = (
    *_DETERMINISTIC_CORE_EXCLUSION_SCHEMA,
    (
        "locked_payload",
        ("attempt_receipt_sha256",),
        "durable_attempt_receipt.receipt_sha256",
    ),
)
_WORKLOAD_CONTROLS = ("benign_only", "fraud_only_diagnostic")
_INVARIANCE_CONTROLS = (
    "identity_rename",
    "future_causality",
    "equal_time_isolation",
    "feature_leakage",
)


class IndependentVerificationError(ValueError):
    """Stable fail-closed error for a rejected serialized artifact."""


def _fail(message: str) -> NoReturn:
    raise IndependentVerificationError(message)


def _canonical_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode()
    except (TypeError, ValueError) as error:
        raise IndependentVerificationError("evidence is not canonical JSON data") from error


def _digest(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _mapping(value: object, label: str) -> dict[str, Any]:
    if type(value) is not dict or any(type(key) is not str for key in value):
        _fail(f"{label} must be an object")
    return cast(dict[str, Any], value)


def _sequence(value: object, label: str) -> list[Any]:
    if type(value) is not list:
        _fail(f"{label} must be an array")
    return value


def _exact_fields(document: Mapping[str, object], fields: set[str], label: str) -> None:
    if set(document) != fields:
        _fail(f"{label} schema fields differ")


def _finite(value: object, label: str) -> float:
    if type(value) not in {int, float}:
        _fail(f"{label} must be finite")
    numeric = cast(int | float, value)
    if not math.isfinite(float(numeric)):
        _fail(f"{label} must be finite")
    return float(numeric)


def _close(left: object, right: object, label: str) -> None:
    if left is None or right is None:
        if left is not right:
            _fail(f"{label} undefined semantics differ")
        return
    if not math.isclose(_finite(left, label), _finite(right, label), rel_tol=1e-12, abs_tol=1e-12):
        _fail(f"{label} failed independent recomputation")


def _stable_core_row(value: object) -> dict[str, Any]:
    row = _mapping(value, "core arm row")
    _exact_fields(row, _ROW_FIELDS, "core arm row")
    stable = {
        key: row[key]
        for key in _ROW_FIELDS
        if key not in {"latency_ms", "row_output_sha256"}
    }
    stable["deterministic_row_sha256"] = _digest(stable)
    return stable


def _stable_core_result(value: object) -> dict[str, Any]:
    result = _mapping(value, "retained core arm result")
    fields = (_RESULT_FIELDS - {"execution_artifacts"}) | {"execution_artifact_refs"}
    _exact_fields(result, fields, "retained core arm result")
    excluded = {
        "arm_spec",
        "row_evidence",
        "p50_latency_ms",
        "p95_latency_ms",
        "p99_latency_ms",
        "score_sha256",
        "result_sha256",
    }
    stable = {key: result[key] for key in fields if key not in excluded}
    stable_rows = [_stable_core_row(row) for row in result["row_evidence"]]
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


def _stable_core_complete(
    value: object, *, deterministic_result_sha256: str
) -> dict[str, Any]:
    metrics = _mapping(value, "core complete metrics")
    fields = {
        "arm",
        "arm_result_sha256",
        "support_sha256",
        "aggregate",
        "calibration",
        "economics",
        "by_family",
        "bootstrap",
        "complete_metrics_sha256",
    }
    _exact_fields(metrics, fields, "core complete metrics")
    aggregate = _mapping(metrics["aggregate"], "core aggregate metrics")
    latency = {"p50_latency_ms", "p95_latency_ms", "p99_latency_ms"}
    if not latency <= set(aggregate):
        _fail("core complete metrics lack exact latency fields")
    stable = {
        "arm": metrics["arm"],
        "deterministic_result_sha256": deterministic_result_sha256,
        "support_sha256": metrics["support_sha256"],
        "aggregate": {
            name: metric for name, metric in aggregate.items() if name not in latency
        },
        "calibration_sha256": metrics["calibration"]["calibration_sha256"],
        "economics_sha256": metrics["economics"]["economics_sha256"],
        "family_sha256": [item["family_sha256"] for item in metrics["by_family"]],
        "bootstrap_sha256": metrics["bootstrap"]["bootstrap_sha256"],
    }
    stable["deterministic_complete_metrics_sha256"] = _digest(stable)
    return stable


def _stable_core_control_rows(name: str, raw_json: object) -> dict[str, Any]:
    if type(raw_json) is not str:
        _fail("core control row evidence must be JSON")
    try:
        rows = _mapping(json.loads(raw_json), "core control row evidence")
    except json.JSONDecodeError as error:
        raise IndependentVerificationError("core control row evidence is invalid") from error
    if raw_json != json.dumps(rows, sort_keys=True):
        _fail("core control row evidence is not canonical")
    if name in _INVARIANCE_CONTROLS:
        excluded = {"before_score_sha256", "after_score_sha256"}
        if not excluded <= set(rows):
            _fail("core invariance score bindings are incomplete")
        return {key: value for key, value in rows.items() if key not in excluded}
    if name in _WORKLOAD_CONTROLS:
        if set(rows) != {"arms", "full_score_sha256"}:
            _fail("core workload row evidence schema differs")
        arms = _mapping(rows["arms"], "core workload arms")
        if set(arms) != set(_ARMS):
            _fail("core workload arm set differs")
        stable_arms: dict[str, Any] = {}
        for arm in _ARMS:
            arm_rows = _mapping(arms[arm], "core workload arm")
            if "score_sha256" not in arm_rows or "rows" not in arm_rows:
                _fail("core workload arm evidence is incomplete")
            stable_rows = []
            for raw_row in arm_rows["rows"]:
                row = _mapping(raw_row, "core workload row")
                if set(row) != {"event_id", "action", "probability", "latency_ms"}:
                    _fail("core workload row schema differs")
                stable_rows.append(
                    {
                        "event_id": row["event_id"],
                        "action": row["action"],
                        "probability": row["probability"],
                    }
                )
            stable_arms[arm] = {
                key: item
                for key, item in arm_rows.items()
                if key not in {"score_sha256", "rows"}
            }
            stable_arms[arm]["rows"] = stable_rows
        return {"arms": stable_arms}
    if name == "label_shuffle":
        return rows
    _fail("core executed control name differs")


def _stable_core_controls(value: object) -> tuple[dict[str, Any], dict[str, str]]:
    suite = _mapping(value, "core controls")
    suite_fields = {
        "controls",
        "evidence_protocol_sha256",
        "support_sha256",
        "implementation_sha256",
        "suite_sha256",
    }
    _exact_fields(suite, suite_fields, "core controls")
    control_fields = {
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
    }
    stable_controls = []
    digests: dict[str, str] = {}
    for raw in suite["controls"]:
        control = _mapping(raw, "core control")
        _exact_fields(control, control_fields, "core control")
        name = str(control["name"])
        excluded = {
            "control_sha256",
            "row_evidence_json",
            "row_evidence_sha256",
            "measurements",
        }
        stable = {key: control[key] for key in control_fields if key not in excluded}
        stable["measurements"] = [
            measurement
            for measurement in control["measurements"]
            if not str(measurement["name"]).endswith("p95_latency_ms")
        ]
        stable_rows = _stable_core_control_rows(name, control["row_evidence_json"])
        stable["deterministic_row_evidence_sha256"] = _digest(stable_rows)
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


def _stable_core_readiness(
    value: object, *, deterministic_control_digests: Mapping[str, str]
) -> dict[str, Any]:
    readiness = _mapping(value, "core readiness")
    fields = {
        "evaluated_arm",
        "gates",
        "qualifying_controls",
        "status",
        "readiness_sha256",
    }
    _exact_fields(readiness, fields, "core readiness")
    gates = [gate for gate in readiness["gates"] if gate["metric"] != "p95_latency_ms"]
    if len(gates) + 1 != len(readiness["gates"]):
        _fail("core readiness latency gate is missing or duplicated")
    controls = []
    for name, passed, _digest_value in readiness["qualifying_controls"]:
        if name not in deterministic_control_digests:
            _fail("core readiness qualifying control is unknown")
        controls.append([name, passed, deterministic_control_digests[name]])
    status = (
        "ready"
        if all(gate["passed"] for gate in gates) and all(item[1] for item in controls)
        else "not_ready"
    )
    stable = {
        "evaluated_arm": readiness["evaluated_arm"],
        "gates": gates,
        "qualifying_controls": controls,
        "deterministic_status": status,
    }
    stable["deterministic_readiness_sha256"] = _digest(stable)
    return stable


def _independent_core_document(
    *,
    payload: Mapping[str, Any],
    artifacts: Sequence[Mapping[str, Any]],
    retained_results: Sequence[Mapping[str, Any]],
    complete: Sequence[Mapping[str, Any]],
    controls: Mapping[str, Any],
    readiness: Mapping[str, Any],
) -> dict[str, Any]:
    results = [_stable_core_result(result) for result in retained_results]
    if [result["arm"] for result in results] != list(_ARMS):
        _fail("deterministic core arm order differs")
    metrics = [
        _stable_core_complete(
            item, deterministic_result_sha256=result["deterministic_result_sha256"]
        )
        for item, result in zip(complete, results, strict=True)
    ]
    stable_controls, control_digests = _stable_core_controls(controls)
    stable_readiness = _stable_core_readiness(
        readiness, deterministic_control_digests=control_digests
    )
    addresses = []
    for artifact in artifacts:
        _exact_fields(artifact, _EXECUTION_ARTIFACT_FIELDS, "core execution artifact")
        addresses.append(
            {
                "evidence_sha256": artifact["evidence_sha256"],
                "artifact_sha256": artifact["artifact_sha256"],
                "payload_sha256": artifact["payload_sha256"],
            }
        )
    return {
        "schema_version": _DETERMINISTIC_CORE_SCHEMA,
        "exclusion_schema": _DETERMINISTIC_CORE_EXCLUSION_SCHEMA,
        "safe_seed": payload["safe_seed"],
        "evidence_protocol": payload["evidence_protocol"],
        "catalog_sha256": payload["catalog_sha256"],
        "execution_artifact_addresses": addresses,
        "arm_results": results,
        "complete_metrics": metrics,
        "controls": stable_controls,
        "readiness": stable_readiness,
    }


def _current_latency_environment() -> dict[str, Any]:
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


def _independent_observational_document(
    *,
    core_sha256: str,
    retained_results: Sequence[Mapping[str, Any]],
    complete: Sequence[Mapping[str, Any]],
    controls: Mapping[str, Any],
    readiness: Mapping[str, Any],
) -> dict[str, Any]:
    arm_observations = []
    for result, metrics in zip(retained_results, complete, strict=True):
        samples = [
            {
                "row_index": index,
                "event_id": row["support"]["event_id"],
                "latency_ms": row["latency_ms"],
            }
            for index, row in enumerate(result["row_evidence"])
        ]
        values = [_finite(item["latency_ms"], "arm latency sample") for item in samples]
        if len(values) < 2 or any(value <= 0.0 for value in values):
            _fail("observational arm latency samples are invalid")
        if len(set(values)) == 1:
            _fail("constant arm latency samples are not observational evidence")
        aggregate = metrics["aggregate"]
        arm_observations.append(
            {
                "arm": result["arm"],
                "support_sha256": result["support_sha256"],
                "samples": samples,
                "p50_latency_ms": float(np.percentile(values, 50)),
                "p95_latency_ms": float(np.percentile(values, 95)),
                "p99_latency_ms": float(np.percentile(values, 99)),
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
    if [item["arm"] for item in arm_observations] != list(_ARMS):
        _fail("observational arm order differs")
    controls_by_name = {str(item["name"]): item for item in controls["controls"]}
    control_observations = []
    for control_name in _WORKLOAD_CONTROLS:
        control = controls_by_name.get(control_name)
        if control is None:
            _fail("observational workload control is missing")
        evidence = _mapping(
            json.loads(str(control["row_evidence_json"])),
            "observational control rows",
        )
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
            values = [
                _finite(item["latency_ms"], "control latency sample") for item in samples
            ]
            if len(values) < 2 or any(value <= 0.0 for value in values):
                _fail("observational control latency samples are invalid")
            if len(set(values)) == 1:
                _fail("constant control latency samples are not observational evidence")
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
        _fail("observational latency gate is missing or duplicated")
    environment = _current_latency_environment()
    document = {
        "schema_version": _OBSERVATIONAL_LATENCY_SCHEMA,
        "measurement_method": _OBSERVATIONAL_MEASUREMENT_METHOD,
        "synthetic_values": False,
        "deterministic_core_sha256": core_sha256,
        "environment": environment,
        "environment_sha256": _digest(environment),
        "arm_observations": arm_observations,
        "control_observations": control_observations,
        "latency_gate": latency_gates[0],
    }
    document["observational_latency_sha256"] = _digest(document)
    return document


def _parse_envelope(serialized: bytes) -> tuple[dict[str, Any], dict[str, Any]]:
    if len(serialized) > 750_000_000:
        _fail("serialized envelope exceeds the frozen bound")
    try:
        envelope = _mapping(json.loads(serialized), "envelope")
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise IndependentVerificationError("envelope is not valid JSON") from error
    _exact_fields(
        envelope,
        {
            "schema_version",
            "compression",
            "payload_base64",
            "payload_sha256",
            "compressed_sha256",
            "uncompressed_bytes",
            "compressed_bytes",
            "envelope_sha256",
        },
        "envelope",
    )
    if envelope["schema_version"] != "apar-sentinel-v5-evidence-envelope/2":
        _fail("unknown envelope schema")
    if envelope["compression"] != "zlib-9":
        _fail("unknown envelope compression")
    expected_envelope = _digest(
        {key: value for key, value in envelope.items() if key != "envelope_sha256"}
    )
    if envelope["envelope_sha256"] != expected_envelope:
        _fail("envelope digest mismatch")
    try:
        compressed = base64.b64decode(str(envelope["payload_base64"]), validate=True)
        if len(compressed) > 536_870_912:
            _fail("compressed payload exceeds the frozen bound")
        decompressor = zlib.decompressobj()
        raw = decompressor.decompress(compressed, 536_870_913)
        if len(raw) > 536_870_912 or decompressor.unconsumed_tail or not decompressor.eof:
            _fail("uncompressed payload exceeds the frozen bound")
    except IndependentVerificationError:
        raise
    except (ValueError, zlib.error) as error:
        raise IndependentVerificationError("envelope compression is invalid") from error
    if (
        len(compressed) != envelope["compressed_bytes"]
        or len(raw) != envelope["uncompressed_bytes"]
    ):
        _fail("envelope byte count mismatch")
    if hashlib.sha256(compressed).hexdigest() != envelope["compressed_sha256"]:
        _fail("compressed payload digest mismatch")
    if hashlib.sha256(raw).hexdigest() != envelope["payload_sha256"]:
        _fail("payload digest mismatch")
    try:
        payload = _mapping(json.loads(raw), "payload")
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise IndependentVerificationError("payload is not valid JSON") from error
    if raw != _canonical_bytes(payload):
        _fail("payload is not canonical JSON")
    return envelope, payload


def _unpack_document(
    value: object,
    *,
    expected_kind: str,
    max_uncompressed_bytes: int = 536_870_912,
) -> dict[str, Any]:
    packed = _mapping(value, f"packed {expected_kind}")
    _exact_fields(
        packed,
        {
            "kind",
            "compression",
            "content_base64",
            "content_sha256",
            "compressed_sha256",
            "uncompressed_bytes",
            "compressed_bytes",
            "packed_sha256",
        },
        f"packed {expected_kind}",
    )
    if packed["kind"] != expected_kind or packed["compression"] != "zlib-9":
        _fail(f"packed {expected_kind} type/compression differs")
    if packed["packed_sha256"] != _digest(
        {key: item for key, item in packed.items() if key != "packed_sha256"}
    ):
        _fail(f"packed {expected_kind} digest mismatch")
    declared_raw = packed["uncompressed_bytes"]
    declared_compressed = packed["compressed_bytes"]
    if (
        type(declared_raw) is not int
        or type(declared_compressed) is not int
        or not 0 < declared_raw <= max_uncompressed_bytes
        or not 0 < declared_compressed <= 536_870_912
    ):
        _fail(f"packed {expected_kind} byte bounds differ")
    try:
        compressed = base64.b64decode(str(packed["content_base64"]), validate=True)
        decompressor = zlib.decompressobj()
        raw = decompressor.decompress(compressed, declared_raw + 1)
    except (ValueError, zlib.error) as error:
        raise IndependentVerificationError(
            f"packed {expected_kind} compression is invalid"
        ) from error
    if (
        len(raw) != declared_raw
        or len(compressed) != declared_compressed
        or decompressor.unconsumed_tail
        or not decompressor.eof
    ):
        _fail(f"packed {expected_kind} byte counts differ")
    if hashlib.sha256(raw).hexdigest() != packed["content_sha256"]:
        _fail(f"packed {expected_kind} content digest mismatch")
    if hashlib.sha256(compressed).hexdigest() != packed["compressed_sha256"]:
        _fail(f"packed {expected_kind} compressed digest mismatch")
    try:
        document = _mapping(json.loads(raw), expected_kind)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise IndependentVerificationError(f"packed {expected_kind} is not valid JSON") from error
    if raw != _canonical_bytes(document):
        _fail(f"packed {expected_kind} is not canonical JSON")
    return document


def _expand_retained_result(
    document: Mapping[str, Any],
    artifact_pool: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, Any], set[str]]:
    expanded = json.loads(_canonical_bytes(document))
    used: set[str] = set()

    def resolve(value: object) -> list[dict[str, Any]]:
        refs = _sequence(value, "execution artifact references")
        if any(type(item) is not str for item in refs) or len(set(refs)) != len(refs):
            _fail("execution artifact references are invalid or duplicated")
        try:
            artifacts = [artifact_pool[str(item)] for item in refs]
        except KeyError as error:
            raise IndependentVerificationError("execution artifact reference is missing") from error
        used.update(str(item) for item in refs)
        return cast(list[dict[str, Any]], json.loads(_canonical_bytes(artifacts)))

    if "execution_artifacts" in expanded:
        _fail("retained arm result bypasses the content-addressed artifact pool")
    expanded["execution_artifacts"] = resolve(expanded.pop("execution_artifact_refs", None))
    spec = _mapping(expanded.get("arm_spec"), "retained arm specification")
    for partition in _sequence(spec.get("training_partitions"), "retained training partitions"):
        partition = _mapping(partition, "retained training partition")
        if "execution_artifacts" in partition:
            _fail("retained training partition bypasses the artifact pool")
        partition["execution_artifacts"] = resolve(partition.pop("execution_artifact_refs", None))
    return expanded, used


def _verify_protocol(payload: Mapping[str, Any], root: Path) -> dict[str, Any]:
    protocol = _mapping(payload.get("evidence_protocol"), "evidence protocol")
    if protocol.get("schema_version") != "1.2.0":
        _fail("evidence protocol schema differs")
    if payload.get("safe_seed") != 404 or protocol.get("safe_development_test_seed") != 404:
        _fail("safe evidence seed must equal 404")
    if protocol.get("locked_development_test_seed") != 2404:
        _fail("locked seed binding differs from 2404")
    if payload.get("safe_seed") == protocol.get("locked_development_test_seed"):
        _fail("safe evidence aliases the locked seed")
    expected_controls = {
        "label_shuffle": {
            "permutation_seed": 1707,
            "max_roc_auc": 0.70,
            "max_pr_auc_excess_over_prevalence": 0.20,
            "min_roc_auc_delta": 0.05,
        },
        "identity_rename": {
            "namespace": "apar.sentinel-v5.control.identity-rename.v1",
            "require_exact_numeric_invariance": True,
        },
        "future_causality": {
            "offset_seconds": 86_400,
            "require_exact_invariance": True,
        },
        "equal_time": {"peer_count": 2, "require_exact_invariance": True},
        "benign_only": {
            "false_decline_rate_max": 0.001,
            "challenge_rate_max": 0.02,
            "review_rate_max": 0.01,
            "recall_semantics": "undefined",
        },
        "fraud_only": {
            "qualifies_for_readiness": False,
            "workload_semantics": "undefined",
        },
        "feature_leakage": {
            "forbidden_fields": [
                "is_fraud",
                "family",
                "campaign_id",
                "split",
                "seed",
                "generator",
                "lifecycle_state",
                "final_outcome",
            ],
            "require_exact_numeric_invariance": True,
        },
    }
    if protocol.get("controls") != expected_controls:
        _fail("executed-control protocol differs from frozen semantics")
    if protocol.get("calibration") != {
        "bin_boundaries": [index / 10 for index in range(11)],
        "interval_closure": "left_closed_right_open_final_closed",
        "rules_only": "not_applicable",
    }:
        _fail("calibration protocol differs from frozen semantics")
    expected_bootstrap = {
        "replicates": 2000,
        "seed": 707,
        "confidence_level": 0.95,
        "interval_method": "percentile",
        "fraud_unit": "campaign",
        "legitimate_unit": "campaign",
        "stratification": "legitimate_and_each_fraud_family",
        "metrics": [
            "recall",
            "false_decline_rate",
            "challenge_rate",
            "review_rate",
            "captured_value_fraction",
            "campaign_detection_rate",
            "expected_calibration_error",
        ],
    }
    if protocol.get("bootstrap") != expected_bootstrap:
        _fail("bootstrap protocol differs from frozen semantics")
    economics = _mapping(protocol.get("economics"), "economic protocol")
    if (
        economics.get("intervention_actions") != ["challenge", "review_hold", "decline_hold"]
        or economics.get("rail_movement_events")
        != {
            "a2a": ["transfer_posted"],
            "agentic": ["authorization"],
            "card": ["settlement"],
        }
        or economics.get("authorization_events") != ["authorization", "transfer_accepted"]
        or economics.get("value_reversal_events") != ["transfer_returned", "refund", "recovery"]
        or economics.get("family_rails")
        != {
            "agentic_intent_abuse": "agentic",
            "app_scam_mule": "a2a",
            "card_testing_cnp": "card",
            "synthetic_merchant_refund": "card",
        }
        or economics.get("aggregate_value_fraction") != "unweighted_mean_of_currency_fractions"
        or economics.get("capture_semantics")
        != "first_intervention_at_or_before_value_movement_or_lifecycle_reversal"
        or economics.get("escape_semantics")
        != "malicious_value_moved_without_prior_intervention_and_without_lifecycle_reversal"
    ):
        _fail("economic protocol differs from frozen semantics")
    expected_gates = [
        ["family_recall", "lower_bound_gte", 0.75],
        ["false_decline_rate", "upper_bound_lte", 0.001],
        ["manual_review_rate", "upper_bound_lte", 0.01],
        ["challenge_rate", "upper_bound_lte", 0.02],
        ["captured_value_fraction", "lower_bound_gte", 0.70],
        ["expected_calibration_error", "upper_bound_lte", 0.10],
        ["p95_decision_latency_ms", "point_lte", 50.0],
        ["campaign_detection_rate", "defined_interval", None],
    ]
    if [
        [gate["metric"], gate["comparison"], gate["target"]] for gate in protocol.get("gates", [])
    ] != expected_gates:
        _fail("readiness gates differ from frozen semantics")
    if protocol.get("bounds") != {
        "max_rows": 100_000,
        "max_execution_artifacts": 4_096,
        "max_single_execution_bytes": 16_777_216,
        "max_aggregate_execution_bytes": 536_870_912,
        "max_control_rows": 100_000,
        "max_bootstrap_replicates": 10_000,
        "max_serialized_evidence_bytes": 536_870_912,
    }:
        _fail("evidence bounds differ from frozen semantics")
    if protocol.get("run_modes") != {
        "safe_validation": {
            "profile": "smoke",
            "development_test_seed": 404,
            "repeatable": True,
            "authorization_required": False,
        },
        "locked_development": {
            "profile": "production",
            "development_test_seed": 2404,
            "repeatable": False,
            "authorization_required": True,
        },
    }:
        _fail("closed evidence run-mode protocol differs")
    if protocol.get("locked_artifact_storage") != {
        "schema_version": "apar-sentinel-v5-chunked-evidence/2",
        "attempt_receipt_path": (
            "docs/experiments/defense-v5-locked-development-attempt.json"
        ),
        "candidate_manifest_path": (
            "docs/experiments/defense-v5-locked-development-candidate.manifest.json"
        ),
        "judge_summary_path": (
            "docs/experiments/defense-v5-locked-development-summary.json"
        ),
        "chunk_size_bytes": 67_108_864,
        "expected_envelope_upper_bound_bytes": 805_306_368,
        "maximum_envelope_bytes": 1_073_741_824,
        "maximum_chunk_count": 16,
        "normal_git_blob_limit_bytes": 104_857_600,
        "publication": "content_chunks_then_atomic_exclusive_manifest",
        "attempt_publication": (
            "canonical_temp_fsync_link_no_replace_parent_fsync"
        ),
    }:
        _fail("locked artifact storage protocol differs")
    prior_result = root / str(protocol.get("existing_development_result_path"))
    if not prior_result.is_file() or hashlib.sha256(
        prior_result.read_bytes()
    ).hexdigest() != protocol.get("existing_development_result_sha256"):
        _fail("existing development result bytes differ from frozen evidence")
    derived = {
        key: value
        for key, value in protocol.items()
        if key
        not in {
            "evidence_protocol_sha256",
            "base_protocol_sha256",
            "arm_protocol_sha256",
            "implementation_sha256",
        }
    }
    if protocol.get("evidence_protocol_sha256") != _digest(derived):
        _fail("evidence protocol digest mismatch")
    evidence_protocol_source = root / "config/defense/defense-v5-evidence.json"
    if (
        not evidence_protocol_source.is_file()
        or json.loads(evidence_protocol_source.read_bytes()) != derived
    ):
        _fail("embedded evidence protocol differs from frozen source config")
    base = root / str(protocol.get("base_protocol_path"))
    arms = root / str(protocol.get("arm_protocol_path"))
    if not base.is_file() or not arms.is_file():
        _fail("bound protocol configuration is missing")
    if hashlib.sha256(base.read_bytes()).hexdigest() != protocol.get("base_protocol_sha256"):
        _fail("base protocol source digest mismatch")
    if hashlib.sha256(arms.read_bytes()).hexdigest() != protocol.get("arm_protocol_sha256"):
        _fail("arm protocol source digest mismatch")
    paths = _sequence(protocol.get("implementation_paths"), "implementation paths")
    if paths != sorted(set(paths)):
        _fail("implementation paths are not unique and canonical")
    implementation: list[list[str]] = []
    for relative in paths:
        path = root / str(relative)
        if not path.is_file():
            _fail("bound implementation source is missing")
        implementation.append([str(relative), hashlib.sha256(path.read_bytes()).hexdigest()])
    if protocol.get("implementation_sha256") != _digest(implementation):
        _fail("implementation source digest mismatch")
    base_document = json.loads(base.read_bytes())
    arm_document = json.loads(arms.read_bytes())
    catalog_path = root / str(base_document["feature_catalog_path"])
    if hashlib.sha256(catalog_path.read_bytes()).hexdigest() != payload.get("catalog_sha256"):
        _fail("feature catalog source digest mismatch")
    catalog_document = json.loads(catalog_path.read_bytes())
    arm_sources = [
        [relative, hashlib.sha256((root / relative).read_bytes()).hexdigest()]
        for relative in arm_document["implementation_paths"]
    ]
    arm_implementation_sha256 = _digest(
        {
            "version": arm_document["implementation_version"],
            "sources": arm_sources,
        }
    )
    safe_base_document = json.loads(_canonical_bytes(base_document))
    safe_base_document["seeds"]["development_test"] = 404
    safe_base_document.pop("protocol_sha256", None)
    safe_protocol_sha256 = _digest(safe_base_document)
    return {
        **protocol,
        "_arm_config_sha256": hashlib.sha256(arms.read_bytes()).hexdigest(),
        "_arm_implementation_sha256": arm_implementation_sha256,
        "_base_protocol_sha256": safe_protocol_sha256,
        "_model_seeds": base_document["seeds"]["catboost_seeds"],
        "_bootstrap_seed": base_document["seeds"]["bootstrap"],
        "_arm_entries": arm_document["arms"],
        "_catalog_feature_names": [item["name"] for item in catalog_document["features"]],
        "_catalog_feature_groups": [item["group"] for item in catalog_document["features"]],
    }


def _command_id(link: Mapping[str, Any]) -> str:
    try:
        payload = json.loads(str(link["command_payload_json"]))
    except json.JSONDecodeError as error:
        raise IndependentVerificationError("command payload is not JSON") from error
    document = {
        "domain": "apar.sentinel-v5.generated-command.v1",
        "type": link["command_type"],
        "name": link["command_name"],
        "payload": payload,
    }
    return f"sha256:{_digest(document)}"


def _expected_ledger(
    *, event_id: str, rail: str, event_type: str, state: Mapping[str, Any]
) -> dict[str, Any] | None:
    amount = Decimal(_unwrap(state["amount"], "decimal"))
    currency = str(state["currency"])
    payer = str(state["payer_account"])
    payee = str(state["payee_account"])
    debit: dict[str, Decimal]
    credit: dict[str, Decimal]
    suffix: str
    if rail == "card":
        fee = Decimal(_unwrap(state["fee"], "decimal"))
        hold = str(state["hold_account"])
        fee_account = str(state["fee_account"])
        chargeback = str(state["chargeback_account"])
        if event_type == "authorization":
            suffix, debit, credit = "hold", {payer: amount}, {hold: amount}
        elif event_type == "reversal":
            suffix, debit, credit = "release", {hold: amount}, {payer: amount}
        elif event_type == "settlement":
            suffix = "settle"
            debit, credit = {hold: amount}, {payee: amount - fee, fee_account: fee}
        elif event_type == "refund":
            suffix = "refund"
            debit, credit = {payee: amount - fee, fee_account: fee}, {payer: amount}
        elif event_type == "chargeback":
            suffix = "chargeback"
            debit, credit = {payee: amount - fee, fee_account: fee}, {chargeback: amount}
        elif event_type == "recovery":
            suffix, debit, credit = "recovery", {chargeback: amount}, {payer: amount}
        else:
            return None
    elif rail == "a2a":
        fee = Decimal(_unwrap(state["fee"], "decimal"))
        fee_account = str(state["fee_account"])
        frozen = str(state["frozen_account"])
        if event_type == "transfer_posted":
            suffix = "post"
            debit, credit = {payer: amount + fee}, {payee: amount, fee_account: fee}
        elif event_type == "transfer_returned":
            suffix, debit, credit = "return", {payee: amount}, {payer: amount}
        elif event_type == "funds_frozen":
            suffix, debit, credit = "freeze", {payee: amount}, {frozen: amount}
        elif event_type == "recovery":
            suffix, debit, credit = "recovery", {frozen: amount}, {payer: amount}
        else:
            return None
    elif rail == "agentic" and event_type == "authorization":
        suffix, debit, credit = "agentic-payment", {payer: amount}, {payee: amount}
    else:
        return None
    return {
        "entry_id": f"{event_id}:{suffix}",
        "debit": [[key, str(value)] for key, value in sorted(debit.items())],
        "credit": [[key, str(value)] for key, value in sorted(credit.items())],
        "currency": currency,
    }


def _unwrap(value: object, key: str) -> str:
    document = _mapping(value, key)
    if set(document) != {key} or type(document[key]) is not str:
        _fail(f"invalid wrapped {key}")
    return cast(str, document[key])


def _canonical_domain_bytes(domain: str, values: Sequence[Sequence[str]]) -> bytes:
    return json.dumps(
        [["domain", domain], *values], separators=(",", ":"), ensure_ascii=True
    ).encode()


def _mandate_bytes(mandate: Mapping[str, Any]) -> bytes:
    return _canonical_domain_bytes(
        "apar.agent-mandate.v1",
        [
            ["mandate_id", str(mandate["mandate_id"])],
            ["version", str(mandate["version"])],
            ["agent_id", str(mandate["agent_id"])],
            ["user_ref", str(mandate["user_ref"])],
            ["user_entity_id", str(mandate["user_entity_id"])],
            ["beneficiary_entity_id", str(mandate["beneficiary_entity_id"])],
            ["consent_ref", str(mandate["consent_ref"])],
            ["merchant_id", str(mandate["merchant_id"])],
            ["payee_id", str(mandate["payee_id"])],
            ["cart_hash", str(mandate["cart_hash"])],
            ["payment_intent_hash", str(mandate["payment_intent_hash"])],
            ["permitted_categories", json.dumps(mandate["permitted_categories"])],
            ["permitted_products", json.dumps(mandate["permitted_products"])],
            ["credential_id", str(mandate["credential_id"])],
            ["credential_scope", str(mandate["credential_scope"])],
            ["required_authentication", str(mandate["required_authentication"])],
            ["max_amount", _unwrap(mandate["max_amount"], "decimal")],
            ["currency", str(mandate["currency"])],
            ["issued_at", _unwrap(mandate["issued_at"], "datetime")],
            ["expires_at", _unwrap(mandate["expires_at"], "datetime")],
        ],
    )


def _request_signing_bytes(request: Mapping[str, Any]) -> tuple[bytes, bytes]:
    mandate = _mapping(request["mandate"], "request mandate")
    signature = bytes.fromhex(_unwrap(request["signature"], "bytes"))
    mandate_hash = hashlib.sha256(_mandate_bytes(mandate)).hexdigest()
    values = [
        ["request_id", str(request["request_id"])],
        ["payment_id", str(request["payment_id"])],
        ["agent_id", str(request["agent_id"])],
        ["key_id", str(request["key_id"])],
        ["mandate_hash", mandate_hash],
        ["amount", _unwrap(request["amount"], "decimal")],
        ["currency", str(request["currency"])],
        ["merchant_id", str(request["merchant_id"])],
        ["payee_id", str(request["payee_id"])],
        ["cart_hash", str(request["cart_hash"])],
        ["payment_intent_hash", str(request["payment_intent_hash"])],
        ["category", str(request["category"])],
        ["product_id", str(request["product_id"])],
        ["credential_id", str(request["credential_id"])],
        ["credential_scope", str(request["credential_scope"])],
        ["consent_ref", str(request["consent_ref"])],
        ["authentication_evidence_ref", str(request.get("authentication_evidence_ref") or "")],
        ["nonce", str(request["nonce"])],
        ["created_at", _unwrap(request["created_at"], "datetime")],
        ["expires_at", _unwrap(request["expires_at"], "datetime")],
        ["prior_receipt_hash", str(request["prior_receipt_hash"])],
        ["campaign_id", str(request["campaign_id"])],
        ["trace_id", str(request["trace_id"])],
        ["actor_id", str(request["actor_id"])],
        ["counterparty_id", str(request["counterparty_id"])],
    ]
    return _canonical_domain_bytes("apar.agent-payment-request.v1", values), signature


def _receipt_hash(
    *, request: Mapping[str, Any], request_hash: str, signature_hash: str, outcome: str
) -> str:
    if outcome == "rejected":
        return ""
    return hashlib.sha256(
        _canonical_domain_bytes(
            "apar.synthetic-integrity-receipt.v1",
            [
                ["agent_id", str(request["agent_id"])],
                ["nonce", str(request["nonce"])],
                [
                    "authentication_evidence_ref",
                    str(request.get("authentication_evidence_ref") or ""),
                ],
                ["request_hash", request_hash],
                ["signature_hash", signature_hash],
                ["previous_receipt_hash", str(request["prior_receipt_hash"])],
                ["outcome", outcome],
            ],
        )
    ).hexdigest()


def _wrapped_datetime(value: object) -> datetime:
    return datetime.fromisoformat(_unwrap(value, "datetime").replace("Z", "+00:00"))


def _expected_trust_reason(
    *,
    request: Mapping[str, Any],
    registry: Mapping[str, Any],
    approved: Mapping[str, Any],
    evidence_registry: Mapping[str, Mapping[str, Any]],
    signature_valid: bool,
    now: datetime,
    used_nonces: set[str],
    used_evidence_refs: set[str],
    last_receipt: str,
) -> str | None:
    if request["agent_id"] != registry["agent_id"] or request["key_id"] != registry["key_id"]:
        return "AGENT_IDENTITY_MISMATCH"
    if not signature_valid:
        return "SIGNATURE_INVALID"
    mandate = _mapping(request["mandate"], "request mandate")
    if mandate != approved or mandate["agent_id"] != request["agent_id"]:
        return "MANDATE_SCOPE_VIOLATION"
    if (
        request["actor_id"] != approved["user_entity_id"]
        or request["counterparty_id"] != approved["beneficiary_entity_id"]
    ):
        return "AUTHORITY_IDENTITY_MISMATCH"
    if Decimal(_unwrap(request["amount"], "decimal")) > Decimal(
        _unwrap(approved["max_amount"], "decimal")
    ):
        return "AMOUNT_LIMIT_EXCEEDED"
    comparisons = (
        ("currency", "currency", "CURRENCY_MISMATCH"),
        ("merchant_id", "merchant_id", "MERCHANT_BINDING_MISMATCH"),
        ("payee_id", "payee_id", "PAYEE_BINDING_MISMATCH"),
    )
    for request_field, mandate_field, reason in comparisons:
        if request[request_field] != approved[mandate_field]:
            return reason
    if request["category"] not in approved["permitted_categories"]:
        return "CATEGORY_SCOPE_VIOLATION"
    if request["product_id"] not in approved["permitted_products"]:
        return "PRODUCT_SCOPE_VIOLATION"
    for request_field, mandate_field, reason in (
        ("cart_hash", "cart_hash", "CART_HASH_MISMATCH"),
        (
            "payment_intent_hash",
            "payment_intent_hash",
            "PAYMENT_INTENT_HASH_MISMATCH",
        ),
        ("credential_id", "credential_id", "CREDENTIAL_BINDING_MISMATCH"),
        ("credential_scope", "credential_scope", "TOKEN_SCOPE_VIOLATION"),
        ("consent_ref", "consent_ref", "CONSENT_BINDING_MISMATCH"),
    ):
        if request[request_field] != approved[mandate_field]:
            return reason
    created_at = _wrapped_datetime(request["created_at"])
    expires_at = _wrapped_datetime(request["expires_at"])
    issued_at = _wrapped_datetime(approved["issued_at"])
    mandate_expires_at = _wrapped_datetime(approved["expires_at"])
    if created_at < issued_at or expires_at > mandate_expires_at:
        return "MANDATE_TIME_SCOPE_VIOLATION"
    if now < created_at or now < issued_at or now >= expires_at or now >= mandate_expires_at:
        return "MANDATE_EXPIRED"
    if str(request["nonce"]) in used_nonces:
        return "NONCE_REPLAY"
    if request["prior_receipt_hash"] != last_receipt:
        return "RECEIPT_CHAIN_BROKEN"
    if approved["required_authentication"] == "step_up":
        evidence_ref = str(request.get("authentication_evidence_ref") or "")
        if evidence_ref in used_evidence_refs:
            return "AUTHENTICATION_EVIDENCE_REPLAY"
        evidence = evidence_registry.get(evidence_ref)
        if evidence is None:
            return "AUTHENTICATION_EVIDENCE_MISSING"
        if (
            evidence["agent_id"] != request["agent_id"]
            or evidence["user_ref"] != approved["user_ref"]
            or evidence["mandate_id"] != approved["mandate_id"]
            or evidence["nonce"] != request["nonce"]
            or evidence["payment_intent_hash"] != request["payment_intent_hash"]
            or evidence["request_id"] != request["request_id"]
            or evidence["outcome"] != "step_up_verified"
        ):
            return "AUTHENTICATION_EVIDENCE_MISMATCH"
        if now < _wrapped_datetime(evidence["issued_at"]) or now >= _wrapped_datetime(
            evidence["expires_at"]
        ):
            return "AUTHENTICATION_EVIDENCE_EXPIRED"
    return None


def _verify_trust(manifest: Mapping[str, Any]) -> None:
    records = _sequence(manifest["trust_records"], "trust records")
    if manifest["rail"] != "agentic":
        if records or manifest.get("trust_registry") is not None:
            _fail("non-agentic manifest retains trust records")
        return
    registry = _mapping(manifest.get("trust_registry"), "trust registry")
    approved = _mapping(json.loads(str(registry["mandate_json"])), "registry mandate")
    evidence_registry = {
        str(evidence["evidence_id"]): evidence
        for evidence in (
            _mapping(json.loads(str(raw)), "registry authentication evidence")
            for raw in registry["authentication_evidence_json"]
        )
    }
    event_times = {
        str(record["event_id"]): datetime.fromisoformat(
            str(record["decision_at"]).replace("Z", "+00:00")
        )
        for record in manifest["event_records"]
    }
    last_receipt = ""
    used_nonces: set[str] = set()
    used_evidence_refs: set[str] = set()
    for raw in records:
        record = _mapping(raw, "trust record")
        request = _mapping(json.loads(str(record["request_json"])), "trust request")
        mandate = _mapping(json.loads(str(record["mandate_json"])), "trust mandate")
        if request["mandate"] != mandate:
            _fail("trust request mandate differs from retained mandate")
        if (
            record["request_id"] != request["request_id"]
            or record["agent_id"] != request["agent_id"]
            or record["key_id"] != request["key_id"]
            or record["mandate_id"] != mandate["mandate_id"]
            or record["public_key_hex"] != registry["public_key_hex"]
        ):
            _fail("trust record identity fields differ from request/registry")
        signing, signature = _request_signing_bytes(request)
        request_hash = hashlib.sha256(signing).hexdigest()
        signature_hash = hashlib.sha256(signature).hexdigest()
        if request_hash != record["request_hash"] or signature_hash != record["signature_hash"]:
            _fail("trust request/signature hash mismatch")
        signature_valid = True
        try:
            Ed25519PublicKey.from_public_bytes(
                bytes.fromhex(str(registry["public_key_hex"]))
            ).verify(signature, signing)
        except (InvalidSignature, ValueError):
            signature_valid = False
        expected_reason = _expected_trust_reason(
            request=request,
            registry=registry,
            approved=approved,
            evidence_registry=evidence_registry,
            signature_valid=signature_valid,
            now=event_times[str(record["event_id"])],
            used_nonces=used_nonces,
            used_evidence_refs=used_evidence_refs,
            last_receipt=last_receipt,
        )
        if (expected_reason is None) != bool(record["allowed"]):
            _fail("trust allowed verdict differs from independent ordered checks")
        if record["reason_code"] != expected_reason:
            _fail("trust reason code differs from independent ordered checks")
        evidence_ref = str(request.get("authentication_evidence_ref") or "")
        retained_auth = (
            _mapping(
                json.loads(str(record["authentication_evidence_json"])),
                "retained authentication evidence",
            )
            if record["authentication_evidence_json"] is not None
            else None
        )
        if retained_auth != evidence_registry.get(evidence_ref):
            _fail("trust record authentication evidence differs from registry")
        if record["allowed"]:
            expected_receipt = _receipt_hash(
                request=request,
                request_hash=request_hash,
                signature_hash=signature_hash,
                outcome=str(record["outcome"]),
            )
            if expected_receipt != record["receipt_hash"]:
                _fail("allowed trust receipt is not reproducible")
            used_nonces.add(str(request["nonce"]))
            if evidence_ref:
                used_evidence_refs.add(evidence_ref)
            last_receipt = expected_receipt
        else:
            reason = str(expected_reason)
            failure_hash = hashlib.sha256(
                _canonical_domain_bytes(
                    "apar.synthetic-integrity-failure.v1",
                    [
                        ["request_id", str(request["request_id"])],
                        ["request_hash", request_hash],
                        ["signature_hash", signature_hash],
                        ["reason_code", reason],
                        ["prior_receipt_hash", str(request["prior_receipt_hash"])],
                        ["outcome", "rejected"],
                    ],
                )
            ).hexdigest()
            if failure_hash != record["receipt_hash"] or record["outcome"] != "rejected":
                _fail("rejected trust receipt is not reproducible")


def _manifest_evidence_digest(manifest: Mapping[str, Any]) -> str:
    lineage = _sequence(manifest["lineage"], "manifest lineage")
    document = {
        "domain": "apar.sentinel-v5.execution-evidence.v1",
        "family": manifest["family"],
        "campaign_id": manifest["campaign_id"],
        "rail": manifest["rail"],
        "lineage": [
            {
                "command_id": _command_id(_mapping(link, "lineage")),
                "command_name": link["command_name"],
                "event_id": link["event_id"],
                "campaign_id": manifest["campaign_id"],
                "payment_id": link["payment_id"],
                "actor_id": link["actor_id"],
                "counterparty_id": link["counterparty_id"],
                "rail": manifest["rail"],
                "scheduled_at": link["scheduled_at"],
                "lifecycle_position": link["lifecycle_position"],
                "is_fraud": link["is_fraud"],
            }
            for link in lineage
        ],
        "events": [
            json.loads(str(record["event_json"]))
            for record in _sequence(manifest["event_records"], "event records")
        ],
        "ledger_entries": [
            {
                "entry_id": posting["entry_id"],
                "debit": {account: str(amount) for account, amount in posting["debit"]},
                "credit": {account: str(amount) for account, amount in posting["credit"]},
                "currency": posting["currency"],
            }
            for posting in _sequence(manifest["ledger_postings"], "ledger postings")
        ],
        "opening_balances": [
            {"account": account, "amount": str(amount)}
            for account, amount in manifest["opening_balances"]
        ],
        "trust": [
            {
                "command_id": record["command_id"],
                "event_id": record["event_id"],
                "request_id": record["request_id"],
                "authentication_evidence_id": (
                    record["authentication_evidence_id"]
                    if record["authentication_evidence_json"] is not None
                    else None
                ),
                "authentication_evidence": (
                    json.loads(record["authentication_evidence_json"])
                    if record["authentication_evidence_json"] is not None
                    else None
                ),
                "receipt_hash": record["receipt_hash"],
                "allowed": record["allowed"],
                "reason_code": record["reason_code"],
                "outcome": record["outcome"],
            }
            for record in manifest["trust_records"]
        ],
    }
    return _digest(document)


def _verify_manifest(artifact: Mapping[str, Any]) -> dict[str, Any]:
    _exact_fields(artifact, _EXECUTION_ARTIFACT_FIELDS, "execution artifact")
    payload_json = str(artifact["payload_json"])
    try:
        manifest = _mapping(json.loads(payload_json), "execution manifest")
    except json.JSONDecodeError as error:
        raise IndependentVerificationError("execution manifest is not JSON") from error
    _exact_fields(manifest, _MANIFEST_FIELDS, "execution manifest")
    if payload_json.encode() != _canonical_bytes(manifest):
        _fail("execution manifest payload is not canonical")
    if hashlib.sha256(payload_json.encode()).hexdigest() != artifact["payload_sha256"]:
        _fail("execution payload digest mismatch")
    if (
        artifact["evidence_sha256"] != manifest["evidence_sha256"]
        or artifact["artifact_sha256"] != manifest["artifact_sha256"]
    ):
        _fail("execution artifact index differs from manifest")
    if (
        _digest({key: value for key, value in manifest.items() if key != "artifact_sha256"})
        != manifest["artifact_sha256"]
    ):
        _fail("execution artifact digest mismatch")
    if _manifest_evidence_digest(manifest) != manifest["evidence_sha256"]:
        _fail("execution evidence digest mismatch")
    lineage = [_mapping(item, "lineage") for item in manifest["lineage"]]
    records = [_mapping(item, "event record") for item in manifest["event_records"]]
    for item in lineage:
        _exact_fields(
            item,
            {
                "command_id",
                "command_type",
                "command_name",
                "event_id",
                "payment_id",
                "actor_id",
                "counterparty_id",
                "lifecycle_position",
                "is_fraud",
                "command_payload_json",
                "trace_id",
                "scheduled_at",
            },
            "execution lineage",
        )
    for item in records:
        _exact_fields(
            item,
            {
                "event_id",
                "payment_id",
                "event_type",
                "amount",
                "currency",
                "decision_at",
                "rail_data_json",
                "lineage_json",
                "event_json",
            },
            "execution event record",
        )
    for item in manifest["ledger_postings"]:
        _exact_fields(
            _mapping(item, "ledger posting"),
            {"entry_id", "debit", "credit", "currency"},
            "ledger posting",
        )
    for item in manifest["trust_records"]:
        _exact_fields(
            _mapping(item, "trust record"),
            {
                "command_id",
                "event_id",
                "request_id",
                "agent_id",
                "key_id",
                "mandate_id",
                "authentication_evidence_id",
                "request_json",
                "mandate_json",
                "authentication_evidence_json",
                "public_key_hex",
                "receipt_hash",
                "request_hash",
                "signature_hash",
                "allowed",
                "reason_code",
                "outcome",
            },
            "trust record",
        )
    if manifest["trust_registry"] is not None:
        _exact_fields(
            _mapping(manifest["trust_registry"], "trust registry"),
            {
                "agent_id",
                "key_id",
                "public_key_hex",
                "mandate_json",
                "authentication_evidence_json",
            },
            "trust registry",
        )
    if not lineage or len(lineage) != len(records):
        _fail("execution lineage and event records do not align")
    if [item["event_id"] for item in lineage] != [item["event_id"] for item in records]:
        _fail("execution event record order differs from lineage")
    if len({item["event_id"] for item in lineage}) != len(lineage):
        _fail("execution lineage event IDs are duplicated")
    states: dict[str, dict[str, Any]] = {}
    previous: dict[str, str] = {}
    expected_postings: list[dict[str, Any]] = []
    positions: dict[str, list[int]] = defaultdict(list)
    for link, record in zip(lineage, records, strict=True):
        if link["command_id"] != _command_id(link):
            _fail("canonical command ID mismatch")
        command = _mapping(json.loads(str(link["command_payload_json"])), "command payload")
        event = _mapping(json.loads(str(record["event_json"])), "raw payment event")
        if (
            event.get("event_id") != link["event_id"]
            or event.get("campaign_id") != manifest["campaign_id"]
            or event.get("rail") != manifest["rail"]
            or event.get("event_type") != record["event_type"]
            or event.get("amount") != str(record["amount"])
            or event.get("currency") != record["currency"]
            or event.get("decision_at") != record["decision_at"]
            or event.get("available_at") != record["decision_at"]
            or event.get("event_time") != link["scheduled_at"]
        ):
            _fail("raw event facts disagree with execution lineage")
        if (
            command.get("payment_id") != link["payment_id"]
            or record["payment_id"] != link["payment_id"]
        ):
            _fail("execution payment lineage mismatch")
        if link["command_name"] in {
            "a2a.initiate",
            "card.authorize",
            "card.decline",
            "agentic.pay",
        }:
            states[str(link["payment_id"])] = command
        state = states.get(str(link["payment_id"]))
        if state is None:
            _fail("lifecycle event lacks an opening command")
        if (
            str(record["amount"]) != _unwrap(state["amount"], "decimal")
            or record["currency"] != state["currency"]
            or link["actor_id"] != state["actor_id"]
            or link["counterparty_id"] != state["counterparty_id"]
        ):
            _fail("event economics differ from opening command")
        lineage_json = _mapping(json.loads(str(record["lineage_json"])), "event lineage")
        if lineage_json.get("previous_event_id", "") != previous.get(str(link["payment_id"]), ""):
            _fail("event lifecycle previous-event lineage mismatch")
        previous[str(link["payment_id"])] = str(link["event_id"])
        positions[str(link["payment_id"])].append(int(link["lifecycle_position"]))
        expected_kind = _EVENT_KIND.get(str(link["command_name"]))
        if link["command_name"] == "agentic.pay":
            rail_data = _mapping(json.loads(str(record["rail_data_json"])), "agentic rail data")
            expected_kind = (
                "authorization_declined"
                if rail_data.get("integrity") == "fail" or rail_data.get("action") == "decline"
                else "authentication_challenge"
                if rail_data.get("integrity") == "pass" and rail_data.get("action") == "challenge"
                else "authorization"
                if rail_data.get("integrity") == "pass" and rail_data.get("action") == "approve"
                else None
            )
        if expected_kind is None or record["event_type"] != expected_kind:
            _fail("unknown or mismatched lifecycle event type")
        posting = _expected_ledger(
            event_id=str(record["event_id"]),
            rail=str(manifest["rail"]),
            event_type=str(record["event_type"]),
            state=state,
        )
        if posting is not None:
            expected_postings.append(posting)
    if any(values != list(range(len(values))) for values in positions.values()):
        _fail("lifecycle positions are missing, duplicated, or out of order")
    if expected_postings != manifest["ledger_postings"]:
        _fail("ledger postings do not reconcile to lifecycle events")
    if [item["entry_id"] for item in expected_postings] != manifest["ledger_entry_ids"]:
        _fail("ledger posting index mismatch")
    for posting in expected_postings:
        debit = sum((Decimal(amount) for _account, amount in posting["debit"]), Decimal(0))
        credit = sum((Decimal(amount) for _account, amount in posting["credit"]), Decimal(0))
        if debit != credit:
            _fail("ledger posting is not double-entry conserving")
    _verify_trust(manifest)
    return manifest


def _interp(value: float, xs: Sequence[object], ys: Sequence[object]) -> float:
    x = [_finite(item, "calibrator x knot") for item in xs]
    y = [_finite(item, "calibrator y knot") for item in ys]
    if (
        len(x) != len(y)
        or len(x) < 2
        or any(left >= right for left, right in zip(x, x[1:], strict=False))
    ):
        _fail("calibrator knots are invalid")
    if value <= x[0]:
        return y[0]
    if value >= x[-1]:
        return y[-1]
    for index in range(1, len(x)):
        if value <= x[index]:
            fraction = (value - x[index - 1]) / (x[index] - x[index - 1])
            return y[index - 1] + fraction * (y[index] - y[index - 1])
    _fail("calibrator interpolation failed")


def _model_action(probability: float, thresholds: Mapping[str, float]) -> str:
    if probability >= thresholds["model_decline"]:
        return "decline_hold"
    if probability >= thresholds["model_review"]:
        return "review_hold"
    if probability >= thresholds["model_challenge"]:
        return "challenge"
    return "approve"


def _rule_action(score: float, thresholds: Mapping[str, float]) -> str:
    if score >= thresholds["rules_decline"]:
        return "decline_hold"
    if score >= thresholds["rules_challenge"]:
        return "challenge"
    return "approve"


def _full_route(
    probability: float, disagreement: float, novelty: float, thresholds: Mapping[str, float]
) -> tuple[str, bool, bool]:
    if (
        probability >= thresholds["model_decline"]
        and disagreement < thresholds["disagreement_review"]
    ):
        return "decline_hold", False, False
    if probability >= thresholds["model_review"]:
        return "review_hold", False, False
    if (
        disagreement >= thresholds["disagreement_review"]
        and probability >= thresholds["model_challenge"]
    ):
        return "review_hold", True, False
    if novelty >= thresholds["novelty_review"] and probability >= 0.3:
        return "review_hold", False, True
    if probability >= thresholds["model_challenge"]:
        return "challenge", False, False
    if novelty >= thresholds["novelty_challenge"]:
        return "challenge", False, True
    return "approve", False, False


def _expected_rule_components(
    *, row: Mapping[str, Any], spec: Mapping[str, Any], event: Mapping[str, Any]
) -> list[list[object]]:
    names = _sequence(spec["catalog_feature_names"], "catalog feature names")
    values = _sequence(row["catalog_feature_values"], "catalog feature values")
    feature: dict[str, float] = {}
    for name, value in zip(names, values, strict=True):
        feature[str(name)] = float(value)

    def score(value: float, threshold: float) -> float | None:
        if value < threshold:
            return None
        return min(1.0, 0.60 + 0.20 * (value / threshold - 1.0))

    components: dict[str, float] = {}
    rail = str(event["rail"])
    kind = str(event["event_type"])
    integrity = str(row["support"]["integrity_status"])
    allowed = {
        "card": {"authorization", "authorization_declined"},
        "a2a": {"transfer_initiated"},
        "agentic": {"authorization", "authentication_challenge", "authorization_declined"},
    }
    coherent = kind in allowed.get(rail, set()) and integrity in (
        {"pass", "fail"} if rail == "agentic" else {"not_applicable"}
    )
    if rail == "agentic" and coherent and integrity == "fail":
        components["INTEGRITY_FAILURE"] = 1.0
    if not coherent:
        components["REQUIRED_DATA_MISSING"] = 1.0
    actor_scores = tuple(
        item
        for item in (
            score(feature.get("actor_count_1m", 0.0), 4.0),
            score(feature.get("actor_count_10m", 0.0), 8.0),
        )
        if item is not None
    )
    if actor_scores:
        components["ACTOR_VELOCITY"] = max(actor_scores)
    rules = (
        ("GRAPH_FAN_IN", feature.get("graph_counterparty_fanin", 0.0), 5.0),
        ("GRAPH_FAN_OUT", feature.get("graph_actor_fanout", 0.0), 5.0),
        (
            "AMOUNT_DEVIATION",
            max(
                abs(feature.get("actor_amount_zscore_24h", 0.0)),
                abs(feature.get("counterparty_amount_zscore_24h", 0.0)),
            ),
            4.0,
        ),
        ("GRAPH_SHARED_NEIGHBOR", feature.get("graph_shared_neighbor_count", 0.0), 3.0),
        ("COUNTERPARTY_VELOCITY", feature.get("pair_prior_count", 0.0), 4.0),
    )
    for reason, value, threshold in rules:
        result = score(value, threshold)
        if result is not None:
            components[reason] = result
    if feature.get("dq_degraded_state", 0.0) >= 1.0:
        components["FEATURE_STATE_DEGRADED"] = 0.60
    return [[name, components[name]] for name in sorted(components)]


def _feature_vector_digest(
    *, row: Mapping[str, Any], spec: Mapping[str, Any], event: Mapping[str, Any]
) -> str:
    feature_values = dict(
        zip(
            spec["catalog_feature_names"],
            row["catalog_feature_values"],
            strict=True,
        )
    )
    document = {
        "schema_version": "1.0.0",
        "event_id": row["support"]["event_id"],
        "decision_at": event["decision_at"],
        "source_event_ids": row["rule_source_event_ids"],
        "max_source_available_at": row["rule_max_source_available_at"],
        "catalog_digest": spec["catalog_sha256"],
        "ordered_values": sorted(feature_values.items()),
    }
    return _digest(document)


def _expected_rule_sources(
    *,
    target: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    event_times: Mapping[str, datetime],
    catalog_names: Sequence[str],
) -> list[str]:
    target_support = _mapping(target["support"], "target support")
    target_time = event_times[str(target_support["event_id"])]
    prior = [row for row in rows if event_times[str(row["support"]["event_id"])] < target_time]
    actor = [row for row in prior if row["support"]["actor_id"] == target_support["actor_id"]]
    counterparty = [
        row
        for row in prior
        if row["support"]["counterparty_id"] == target_support["counterparty_id"]
    ]
    pair = [
        row
        for row in actor
        if row["support"]["counterparty_id"] == target_support["counterparty_id"]
    ]
    sources: set[str] = set()

    def add(selected: Sequence[Mapping[str, Any]]) -> None:
        sources.update(str(row["support"]["event_id"]) for row in selected)

    names = set(catalog_names)
    if "actor_count_1m" in names:
        add(
            [
                row
                for row in actor
                if (target_time - event_times[str(row["support"]["event_id"])]).total_seconds()
                <= 60
            ]
        )
    if "actor_count_10m" in names:
        add(
            [
                row
                for row in actor
                if (target_time - event_times[str(row["support"]["event_id"])]).total_seconds()
                <= 600
            ]
        )
    if "graph_actor_fanout" in names:
        add(actor)
    if "actor_amount_zscore_24h" in names:
        add(
            [
                row
                for row in actor
                if (target_time - event_times[str(row["support"]["event_id"])]).total_seconds()
                <= 86_400
            ]
        )
    if "graph_counterparty_fanin" in names:
        add(
            [
                row
                for row in counterparty
                if (target_time - event_times[str(row["support"]["event_id"])]).total_seconds()
                <= 86_400
            ]
        )
    if "graph_shared_neighbor_count" in names:
        add(counterparty)
    if "pair_prior_count" in names:
        add(pair)
    return sorted(sources)


def _isolation_raw_score(manifest: Mapping[str, Any], features: Sequence[float]) -> float:
    if len(features) != int(manifest["feature_count"]):
        _fail("novelty feature count mismatch")
    values = np.asarray(features, dtype=np.float32)
    total_depth = 0.0
    for raw_tree in manifest["trees"]:
        tree = _mapping(raw_tree, "isolation tree")
        left = [int(value) for value in tree["children_left"]]
        right = [int(value) for value in tree["children_right"]]
        split_features = [int(value) for value in tree["feature"]]
        thresholds = [float(value) for value in tree["threshold"]]
        estimator_features = [int(value) for value in tree["estimator_features"]]
        decision_lengths = [float(value) for value in tree["decision_path_lengths"]]
        average_lengths = [float(value) for value in tree["average_path_lengths"]]
        node = 0
        traversed = 0
        while left[node] != -1:
            feature = split_features[node]
            source = estimator_features[feature]
            node = left[node] if values[source] <= thresholds[node] else right[node]
            traversed += 1
            if node < 0 or node >= len(left) or traversed > len(left):
                _fail("isolation tree traversal is invalid")
        total_depth += decision_lengths[node] + average_lengths[node] - 1.0
    samples = int(manifest["max_samples"])
    if samples <= 1:
        average_path = 0.0
    elif samples == 2:
        average_path = 1.0
    else:
        average_path = 2.0 * (math.log(samples - 1.0) + float(np.euler_gamma)) - (
            2.0 * (samples - 1.0) / samples
        )
    denominator = len(manifest["trees"]) * average_path
    anomaly = 1.0 if denominator == 0.0 else 2.0 ** (-total_depth / denominator)
    return -anomaly - float(manifest["offset"])


def _verify_training_partition(document: Mapping[str, Any]) -> None:
    _exact_fields(document, _TRAINING_FIELDS, "training partition")
    event_ids = _sequence(document["ordered_event_ids"], "training event IDs")
    labels = _sequence(document["labels"], "training labels")
    names = _sequence(document["feature_names"], "training feature names")
    matrix = _sequence(document["feature_matrix"], "training feature matrix")
    support = _sequence(document["support_records"], "training support")
    if (
        not event_ids
        or len(event_ids) > 100_000
        or len(event_ids) != len(labels)
        or len(matrix) != len(labels)
        or len(support) != len(labels)
    ):
        _fail("training partition rows do not align")
    if len(set(event_ids)) != len(event_ids) or set(labels) != {0, 1}:
        _fail("training partition IDs/classes are invalid")
    if [row["event_id"] for row in support] != event_ids or [
        row["label"] for row in support
    ] != labels:
        _fail("training support order/labels mismatch")
    if any(len(row) != len(names) for row in matrix):
        _fail("training feature matrix shape mismatch")
    if len(matrix) * len(names) > 10_000_000:
        _fail("training feature matrix exceeds the frozen cell bound")
    if document["feature_matrix_sha256"] != _digest(matrix):
        _fail("training feature matrix digest mismatch")
    expected_rows = [
        {"event_id": event_id, "label": label}
        for event_id, label in zip(event_ids, labels, strict=True)
    ]
    if document["ordered_rows_sha256"] != _digest(expected_rows):
        _fail("training ordered rows digest mismatch")
    if document["ordered_support_sha256"] != _digest(support):
        _fail("training support digest mismatch")
    batch_json = str(document["feature_batch_payload_json"])
    batch = json.loads(batch_json)
    if batch_json != json.dumps(batch, sort_keys=True):
        _fail("training feature batch is not canonical")
    if batch != {"names": names, "rows": matrix}:
        _fail("training feature batch differs from matrix")
    if hashlib.sha256(batch_json.encode()).hexdigest() != document["feature_batch_sha256"]:
        _fail("training feature batch digest mismatch")
    for artifact in document["execution_artifacts"]:
        _verify_manifest(_mapping(artifact, "training execution artifact"))


def _verify_spec(spec: Mapping[str, Any], catalog_sha256: str, bindings: Mapping[str, Any]) -> None:
    _exact_fields(spec, _SPEC_FIELDS, "arm specification")
    if spec["catalog_sha256"] != catalog_sha256:
        _fail("arm specification catalog digest mismatch")
    if spec["spec_sha256"] != _digest(
        {key: value for key, value in spec.items() if key != "spec_sha256"}
    ):
        _fail("arm specification digest mismatch")
    if spec["execution_bound"] is not True:
        _fail("arm specification is not execution-bound")
    if (
        spec["arm_config_sha256"] != bindings["_arm_config_sha256"]
        or spec["implementation_sha256"] != bindings["_arm_implementation_sha256"]
        or spec["protocol_sha256"] != bindings["_base_protocol_sha256"]
        or spec["bootstrap_seed"] != bindings["_bootstrap_seed"]
        or spec["catalog_feature_names"] != bindings["_catalog_feature_names"]
        or spec["catalog_feature_groups"] != bindings["_catalog_feature_groups"]
    ):
        _fail("arm specification source/config/protocol binding mismatch")
    entries = {str(item["arm"]): item for item in bindings["_arm_entries"]}
    entry = entries.get(str(spec["arm"]))
    if entry is None or any(
        spec[field] != entry[field]
        for field in (
            "model",
            "graph",
            "rules",
            "trust",
            "novelty",
            "disagreement",
            "calibration_method",
            "threshold_method",
        )
    ):
        _fail("arm specification component semantics differ from frozen config")
    names = list(bindings["_catalog_feature_names"])
    groups = list(bindings["_catalog_feature_groups"])
    approved_non_graph = [
        name
        for name, group in zip(names, groups, strict=True)
        if group not in {"graph", "integrity"}
    ]
    approved_graph = [
        name for name, group in zip(names, groups, strict=True) if group not in {"integrity"}
    ]
    rules = [
        "actor_count_1m",
        "actor_count_10m",
        "graph_counterparty_fanin",
        "graph_actor_fanout",
        "actor_amount_zscore_24h",
        "counterparty_amount_zscore_24h",
        "graph_shared_neighbor_count",
        "pair_prior_count",
        "dq_degraded_state",
    ]
    expected_features = (
        rules
        if spec["arm"] == "rules_only"
        else approved_non_graph
        if spec["arm"] == "ensemble_no_graph"
        else approved_graph
    )
    if spec["feature_names"] != expected_features:
        _fail("arm specification feature subset differs from frozen semantics")
    expected_seeds = bindings["_model_seeds"] if spec["model"] else []
    if spec["model_seeds"] != expected_seeds:
        _fail("arm specification model seeds differ from protocol")
    if [item["partition"] for item in spec["training_partitions"]] != [
        "train",
        "calibration",
        "threshold",
    ]:
        _fail("arm training partition order mismatch")
    for partition in spec["training_partitions"]:
        _verify_training_partition(_mapping(partition, "training partition"))
    model = bool(spec["model"])
    if model:
        seeds = _sequence(spec["model_seeds"], "model seeds")
        artifacts = _sequence(spec["model_artifacts"], "model artifacts")
        calibrators = _sequence(spec["calibrator_manifests"], "calibrators")
        if (
            not 3 <= len(seeds) <= 5
            or len(artifacts) != len(seeds)
            or len(calibrators) != len(seeds)
        ):
            _fail("model member evidence count mismatch")
        for artifact, expected in zip(artifacts, spec["model_artifact_sha256"], strict=True):
            artifact = _mapping(artifact, "model artifact")
            _exact_fields(
                artifact,
                {"serialization", "payload_base64", "artifact_sha256"},
                "model artifact",
            )
            if artifact["serialization"] != "catboost-json-canonical-v1":
                _fail("model artifact serialization differs")
            raw = base64.b64decode(artifact["payload_base64"], validate=True)
            if (
                hashlib.sha256(raw).hexdigest() != artifact["artifact_sha256"]
                or expected != artifact["artifact_sha256"]
            ):
                _fail("model artifact digest mismatch")
            try:
                model_document = _mapping(json.loads(raw), "canonical CatBoost model")
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise IndependentVerificationError(
                    "model artifact is not canonical JSON"
                ) from error
            if raw != _canonical_bytes(model_document):
                _fail("model artifact JSON is not canonical")
            model_info = _mapping(model_document.get("model_info"), "CatBoost model_info")
            if {"model_guid", "train_finish_time"} & set(model_info):
                _fail("model artifact retains volatile metadata")
        for manifest, expected in zip(calibrators, spec["calibrator_artifact_sha256"], strict=True):
            if (
                manifest["artifact_sha256"]
                != _digest(
                    {key: value for key, value in manifest.items() if key != "artifact_sha256"}
                )
                or expected != manifest["artifact_sha256"]
            ):
                _fail("calibrator artifact digest mismatch")
    elif spec["model_artifacts"] or spec["calibrator_manifests"]:
        _fail("rules-only arm retains model artifacts")
    if spec["novelty"]:
        novelty = _mapping(spec["novelty_manifest"], "novelty manifest")
        if (
            novelty["artifact_sha256"]
            != _digest({key: value for key, value in novelty.items() if key != "artifact_sha256"})
            or novelty["artifact_sha256"] != spec["novelty_artifact_sha256"]
        ):
            _fail("novelty artifact digest mismatch")
        if (
            novelty["feature_count"] != len(spec["feature_names"])
            or int(novelty["max_samples"]) <= 0
            or not 1 <= len(novelty["trees"]) <= 512
            or not math.isfinite(float(novelty["offset"]))
        ):
            _fail("novelty artifact dimensions are invalid")
        total_nodes = 0
        expected_estimator_features = list(range(int(novelty["feature_count"])))
        for raw_tree in novelty["trees"]:
            tree = _mapping(raw_tree, "isolation tree")
            arrays = [
                tree[name]
                for name in (
                    "children_left",
                    "children_right",
                    "feature",
                    "threshold",
                    "decision_path_lengths",
                    "average_path_lengths",
                )
            ]
            lengths = {len(array) for array in arrays}
            if len(lengths) != 1 or not lengths or 0 in lengths:
                _fail("isolation tree arrays do not align")
            total_nodes += len(arrays[0])
            if tree["estimator_features"] != expected_estimator_features:
                _fail("isolation tree estimator features are not exact")
            child_nodes: list[int] = []
            for node, (left, right, feature) in enumerate(
                zip(
                    tree["children_left"],
                    tree["children_right"],
                    tree["feature"],
                    strict=True,
                )
            ):
                leaf = left == -1 and right == -1
                if leaf:
                    if feature != -2:
                        _fail("isolation tree leaf feature is invalid")
                    continue
                if (
                    left <= node
                    or right <= node
                    or left >= len(arrays[0])
                    or right >= len(arrays[0])
                    or left == right
                    or feature < 0
                    or feature >= len(expected_estimator_features)
                ):
                    _fail("isolation tree child/split evidence is invalid")
                child_nodes.extend((int(left), int(right)))
            if sorted(child_nodes) != list(range(1, len(arrays[0]))):
                _fail("isolation tree nodes do not form one rooted tree")
        if total_nodes > 262_144:
            _fail("novelty artifact exceeds the frozen node bound")


def _verify_arm_result(
    result: Mapping[str, Any], catalog_sha256: str, bindings: Mapping[str, Any]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, dict[str, Any]]]:
    _exact_fields(result, _RESULT_FIELDS, "arm result")
    spec = _mapping(result["arm_spec"], "arm specification")
    _verify_spec(spec, catalog_sha256, bindings)
    if result["arm"] != spec["arm"] or result["arm_spec_sha256"] != spec["spec_sha256"]:
        _fail("arm result specification binding mismatch")
    artifacts = [_mapping(item, "execution artifact") for item in result["execution_artifacts"]]
    if not 1 <= len(artifacts) <= 4_096:
        _fail("arm execution artifact count exceeds the frozen bound")
    artifact_sizes = [len(str(artifact["payload_json"]).encode()) for artifact in artifacts]
    if any(size > 16_777_216 for size in artifact_sizes) or sum(artifact_sizes) > 268_435_456:
        _fail("arm execution artifact bytes exceed the frozen bound")
    manifests = {artifact["evidence_sha256"]: _verify_manifest(artifact) for artifact in artifacts}
    if len(manifests) != len(artifacts):
        _fail("execution artifact evidence IDs are duplicated")
    event_map: dict[str, dict[str, Any]] = {}
    artifact_event_ids: list[str] = []
    for manifest in manifests.values():
        for record in manifest["event_records"]:
            event = _mapping(json.loads(record["event_json"]), "payment event")
            event_id = str(event["event_id"])
            if event_id in event_map:
                _fail("execution event IDs are duplicated across artifacts")
            event_map[event_id] = event
            artifact_event_ids.append(event_id)
    rows = [_mapping(item, "arm row") for item in result["row_evidence"]]
    if not 1 <= len(rows) <= 100_000:
        _fail("arm row count exceeds the frozen bound")
    support = [_mapping(row["support"], "row support") for row in rows]
    for row, support_row in zip(rows, support, strict=True):
        _exact_fields(row, _ROW_FIELDS, "arm row")
        _exact_fields(support_row, _SUPPORT_FIELDS, "arm support row")
    event_ids = [str(item["event_id"]) for item in support]
    if len(set(event_ids)) != len(event_ids) or set(event_ids) != set(artifact_event_ids):
        _fail("arm rows do not uniquely and exhaustively cover execution lineage")
    if result["support_sha256"] != _digest(support):
        _fail("arm support digest mismatch")
    catalog_names = list(spec["catalog_feature_names"])
    claimed_catalog_indices: set[int] = set()
    subset_indices: list[int] = []
    for feature_name in spec["feature_names"]:
        for index, catalog_name in enumerate(catalog_names):
            if catalog_name == feature_name and index not in claimed_catalog_indices:
                claimed_catalog_indices.add(index)
                subset_indices.append(index)
                break
        else:
            _fail("arm feature is absent from the bound catalog")
    thresholds = {str(name): float(value) for name, value in spec["threshold_values"]}
    calibrators = list(spec["calibrator_manifests"])
    trust_failures: set[str] = {
        str(event_id)
        for manifest in manifests.values()
        for event_id in manifest["trust_failure_event_ids"]
    }
    all_event_times = {
        event_id: datetime.fromisoformat(str(event["available_at"]).replace("Z", "+00:00"))
        for event_id, event in event_map.items()
    }
    for row, row_support in zip(rows, support, strict=True):
        if row["arm_spec_sha256"] != spec["spec_sha256"]:
            _fail("row arm specification digest mismatch")
        if row["row_output_sha256"] != _digest(
            {key: value for key, value in row.items() if key != "row_output_sha256"}
        ):
            _fail("arm row digest mismatch")
        catalog_values = row["catalog_feature_values"]
        if row["catalog_feature_sha256"] != _digest(catalog_values):
            _fail("catalog row digest mismatch")
        expected_subset = [catalog_values[index] for index in subset_indices]
        if row["subset_feature_values"] != expected_subset or row[
            "subset_feature_sha256"
        ] != _digest(expected_subset):
            _fail("arm subset feature evidence mismatch")
        evidence = manifests.get(row_support["execution_evidence_sha256"])
        if evidence is None:
            _fail("row execution evidence cannot be resolved")
        links = [
            item for item in evidence["lineage"] if item["event_id"] == row_support["event_id"]
        ]
        if len(links) != 1:
            _fail("row lineage cannot be resolved exactly once")
        link = links[0]
        if any(
            row_support[field] != link[source]
            for field, source in (
                ("source_command_id", "command_id"),
                ("source_event_id", "event_id"),
                ("payment_id", "payment_id"),
                ("actor_id", "actor_id"),
                ("counterparty_id", "counterparty_id"),
            )
        ) or row_support["label"] != int(link["is_fraud"]):
            _fail("row support disagrees with command/event lineage")
        event = event_map[str(row_support["event_id"])]
        if (
            row_support["campaign_id"] != evidence["campaign_id"]
            or row_support["family"] != evidence["family"]
            or row_support["rail"] != evidence["rail"]
            or str(row_support["amount"]) != str(float(Decimal(str(event["amount"]))))
            or row_support["currency"] != event["currency"]
        ):
            _fail("row support facts differ from execution event")
        source_ids = list(row["rule_source_event_ids"])
        if source_ids != sorted(set(source_ids)):
            _fail("rule causal source IDs are not canonical")
        expected_source_ids = _expected_rule_sources(
            target=row,
            rows=rows,
            event_times=all_event_times,
            catalog_names=catalog_names,
        )
        if source_ids != expected_source_ids:
            _fail("rule causal source IDs are not exact")
        target_time = datetime.fromisoformat(str(event["decision_at"]).replace("Z", "+00:00"))
        source_times: list[datetime] = []
        for source_id in source_ids:
            if source_id not in all_event_times or all_event_times[source_id] >= target_time:
                _fail("rule causal source is missing or not strictly prior")
            source_times.append(all_event_times[source_id])
        expected_max = (
            max(source_times).isoformat().replace("+00:00", "Z") if source_times else None
        )
        if row["rule_max_source_available_at"] != expected_max:
            _fail("rule causal maximum availability mismatch")
        if spec["rules"]:
            expected_components = _expected_rule_components(row=row, spec=spec, event=event)
            if row["rule_components"] != expected_components:
                _fail("RuleEngine components failed independent replay")
            expected_rule_score = 1.0 - math.prod(
                (
                    1.0 - _finite(value, "rule component score")
                    for _reason, value in expected_components
                ),
                start=1.0,
            )
            _close(row["rule_score"], expected_rule_score, "rule score")
            if row["rule_vector_sha256"] != _feature_vector_digest(row=row, spec=spec, event=event):
                _fail("rule feature-vector digest mismatch")
            rule_manifest = {
                "schema_version": "1.0.0",
                "version": "1.0.0",
                "actor_count_1m": 4.0,
                "actor_count_10m": 8.0,
                "counterparty_fanin": 5.0,
                "actor_fanout": 5.0,
                "amount_zscore": 4.0,
                "shared_neighbors": 3.0,
                "repeated_pair_count": 4.0,
                "degraded_state": 1.0,
                "threshold_score": 0.6,
            }
            if row["rule_manifest_sha256"] != _digest(rule_manifest):
                _fail("rule manifest digest mismatch")
            expected_evidence = (
                sorted({str(row_support["event_id"]), *source_ids}) if expected_components else []
            )
            if row["rule_evidence_source_ids"] != expected_evidence:
                _fail("rule evidence source lineage mismatch")
            rule_action = _rule_action(expected_rule_score, thresholds)
            if row["rule_action"] != rule_action:
                _fail("rule action trace mismatch")
        else:
            rule_action = None
            if row["rule_score"] is not None or row["rule_components"]:
                _fail("rules-disabled arm retains rule outputs")
        trust_failure = str(row_support["event_id"]) in trust_failures
        expected_integrity = (
            "fail"
            if trust_failure
            else "pass"
            if row_support["rail"] == "agentic"
            else "not_applicable"
        )
        if row_support["integrity_status"] != expected_integrity:
            _fail("row integrity status differs from retained verifier verdict")
        trust_action = "decline_hold" if spec["trust"] and trust_failure else None
        if row["trust_action"] != trust_action or row["trust_routed"] != (trust_action is not None):
            _fail("trust routing trace mismatch")
        raw_scores = [float(value) for value in row["model_raw_scores"]]
        calibrated = [float(value) for value in row["model_calibrated_scores"]]
        if spec["model"]:
            if len(raw_scores) != len(calibrators) or len(calibrated) != len(calibrators):
                _fail("model member score count mismatch")
            expected_calibrated = [
                max(0.0, min(1.0, _interp(raw, manifest["x_thresholds"], manifest["y_thresholds"])))
                for raw, manifest in zip(raw_scores, calibrators, strict=True)
            ]
            if any(
                not math.isclose(left, right, rel_tol=1e-12, abs_tol=1e-12)
                for left, right in zip(calibrated, expected_calibrated, strict=True)
            ):
                _fail("calibrated model trace mismatch")
            probability = sum(calibrated) / len(calibrated)
            _close(row["probability"], probability, "ensemble probability")
            probability_action = _model_action(probability, thresholds)
            if row["probability_action"] != probability_action:
                _fail("probability action trace mismatch")
        else:
            probability = 1.0 if trust_action else float(row["rule_score"])
            _close(row["probability"], probability, "rules probability")
            probability_action = None
            if raw_scores or calibrated or row["probability_action"] is not None:
                _fail("rules-only arm retains model traces")
        if spec["novelty"]:
            disagreement = float(np.std(np.asarray(calibrated, dtype=float)))
            _close(row["disagreement"], disagreement, "model disagreement")
            novelty_raw = _finite(row["novelty_raw_score"], "novelty raw score")
            novelty_manifest = _mapping(spec["novelty_manifest"], "novelty manifest")
            _close(
                novelty_raw,
                _isolation_raw_score(novelty_manifest, expected_subset),
                "novelty raw artifact replay",
            )
            novelty = max(0.0, min(1.0, 0.5 - novelty_raw))
            _close(row["novelty_score"], novelty, "novelty score")
            model_action, disagreement_routed, novelty_routed = _full_route(
                probability, disagreement, novelty, thresholds
            )
            if (
                row["model_action"] != model_action
                or row["disagreement_routed"] != disagreement_routed
                or row["novelty_routed"] != novelty_routed
            ):
                _fail("full Sentinel component routing mismatch")
        elif spec["model"]:
            model_action = cast(str, probability_action)
            if (
                row["model_action"] != model_action
                or row["novelty_score"] is not None
                or row["disagreement"] is not None
            ):
                _fail("model-only component trace mismatch")
        else:
            model_action = None
        expected_action = (
            trust_action
            if trust_action is not None
            else max((model_action, rule_action), key=lambda action: _SEVERITY[str(action)])
            if spec["arm"] == "full_sentinel"
            else rule_action
            if spec["rules"]
            else model_action
        )
        if row["action"] != expected_action:
            _fail("final row action failed component replay")
    score_document = {
        "spec": spec,
        "support_sha256": result["support_sha256"],
        "execution_artifacts": result["execution_artifacts"],
        "rows": result["row_evidence"],
    }
    if result["score_sha256"] != _digest(score_document):
        _fail("arm score digest mismatch")
    if result["result_sha256"] != _digest(
        {key: value for key, value in result.items() if key != "result_sha256"}
    ):
        _fail("arm result digest mismatch")
    return rows, artifacts, manifests


def _ratio(numerator: float, denominator: float) -> float | None:
    return None if denominator == 0 else numerator / denominator


def _average_precision(labels: Sequence[int], probabilities: Sequence[float]) -> float:
    positives = sum(labels)
    if positives == 0:
        _fail("average precision requires positives")
    true_positive = 0
    false_positive = 0
    previous_recall = 0.0
    area = 0.0
    for probability in sorted(set(probabilities), reverse=True):
        positions = [index for index, value in enumerate(probabilities) if value == probability]
        true_positive += sum(labels[index] == 1 for index in positions)
        false_positive += sum(labels[index] == 0 for index in positions)
        recall = true_positive / positives
        precision = true_positive / (true_positive + false_positive)
        area += (recall - previous_recall) * precision
        previous_recall = recall
    return area


def _roc_auc(labels: Sequence[int], probabilities: Sequence[float]) -> float:
    positives = sum(labels)
    negatives = len(labels) - positives
    if not positives or not negatives:
        _fail("ROC-AUC requires both classes")
    wins = 0.0
    for positive, label in enumerate(labels):
        if label != 1:
            continue
        for negative, other in enumerate(labels):
            if other != 0:
                continue
            wins += (
                1.0
                if probabilities[positive] > probabilities[negative]
                else 0.5
                if probabilities[positive] == probabilities[negative]
                else 0.0
            )
    return wins / (positives * negatives)


def _percentile(values: Sequence[float], percentile: float) -> float:
    if not values:
        _fail("percentile requires values")
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile / 100.0
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] + fraction * (ordered[upper] - ordered[lower])


def _verify_metric(
    document: Mapping[str, Any],
    *,
    value: float | None,
    numerator: float | None,
    denominator: float | None,
    applicability: str,
    support_ids: Sequence[str],
) -> None:
    _exact_fields(
        document,
        {
            "name",
            "applicability",
            "value",
            "numerator",
            "denominator",
            "support_count",
            "support_sha256",
            "formula",
            "metric_sha256",
        },
        "metric estimate",
    )
    if document["metric_sha256"] != _digest(
        {key: item for key, item in document.items() if key != "metric_sha256"}
    ):
        _fail("metric digest mismatch")
    if document["applicability"] != applicability:
        _fail(f"metric {document['name']} applicability mismatch")
    _close(document["value"], value, f"metric {document['name']} value")
    _close(document["numerator"], numerator, f"metric {document['name']} numerator")
    _close(document["denominator"], denominator, f"metric {document['name']} denominator")
    if document["support_count"] != len(support_ids) or document["support_sha256"] != _digest(
        list(support_ids)
    ):
        # Production hashes tuples; JSON canonical form is the same array.
        _fail(f"metric {document['name']} support lineage mismatch")


def _metric_expectations(
    *, labels: Sequence[int], actions: Sequence[str], probabilities: Sequence[float] | None
) -> dict[str, tuple[float | None, float | None, float | None, str]]:
    detected = [action in _DETECTION for action in actions]
    fraud = sum(labels)
    legitimate = len(labels) - fraud
    true_positive = sum(flag and label == 1 for flag, label in zip(detected, labels, strict=True))
    false_positive = sum(flag and label == 0 for flag, label in zip(detected, labels, strict=True))
    false_negative = fraud - true_positive

    def ratio_metric(
        numerator: float, denominator: float
    ) -> tuple[float | None, float, float, str]:
        return (
            _ratio(numerator, denominator),
            numerator,
            denominator,
            "defined" if denominator else "undefined",
        )

    output: dict[str, tuple[float | None, float | None, float | None, str]] = {
        "recall": ratio_metric(float(true_positive), float(fraud)),
        "precision": ratio_metric(float(true_positive), float(true_positive + false_positive)),
        "f1": ratio_metric(
            float(2 * true_positive), float(2 * true_positive + false_positive + false_negative)
        ),
        "false_decline_rate": ratio_metric(
            float(
                sum(
                    action == "decline_hold" and label == 0
                    for action, label in zip(actions, labels, strict=True)
                )
            ),
            float(legitimate),
        ),
        "challenge_rate": ratio_metric(
            float(
                sum(
                    action == "challenge" and label == 0
                    for action, label in zip(actions, labels, strict=True)
                )
            ),
            float(legitimate),
        ),
        "review_rate": ratio_metric(
            float(
                sum(
                    action == "review_hold" and label == 0
                    for action, label in zip(actions, labels, strict=True)
                )
            ),
            float(legitimate),
        ),
        "decline_rate": ratio_metric(
            float(sum(action == "decline_hold" for action in actions)), float(len(labels))
        ),
    }
    if probabilities is None:
        for name in (
            "pr_auc",
            "roc_auc",
            "brier",
            "expected_calibration_error",
            "maximum_calibration_error",
        ):
            output[name] = (None, None, None, "not_applicable")
        return output
    if fraud and legitimate:
        pr = _average_precision(labels, probabilities)
        roc = _roc_auc(labels, probabilities)
        output["pr_auc"] = (pr, pr, 1.0, "defined")
        output["roc_auc"] = (roc, roc, 1.0, "defined")
    else:
        output["pr_auc"] = (None, float(fraud), 0.0, "undefined")
        output["roc_auc"] = (None, float(fraud), 0.0, "undefined")
    squared = sum(
        (probability - label) ** 2 for probability, label in zip(probabilities, labels, strict=True)
    )
    output["brier"] = (squared / len(labels), squared, float(len(labels)), "defined")
    gaps: list[tuple[int, float]] = []
    for index in range(10):
        lower, upper = index / 10, (index + 1) / 10
        positions = [
            position
            for position, probability in enumerate(probabilities)
            if probability >= lower
            and (probability <= upper if index == 9 else probability < upper)
        ]
        if positions:
            mean = sum(probabilities[position] for position in positions) / len(positions)
            empirical = sum(labels[position] for position in positions) / len(positions)
            gaps.append((len(positions), abs(mean - empirical)))
    ece = sum(count / len(labels) * gap for count, gap in gaps)
    mce = max((gap for _count, gap in gaps), default=0.0)
    output["expected_calibration_error"] = (ece, ece * len(labels), float(len(labels)), "defined")
    output["maximum_calibration_error"] = (mce, mce, 1.0, "defined")
    return output


def _verify_calibration(
    document: Mapping[str, Any],
    *,
    labels: Sequence[int],
    probabilities: Sequence[float] | None,
    support_ids: Sequence[str],
) -> None:
    _exact_fields(
        document,
        {
            "applicability",
            "boundaries",
            "bins",
            "expected_calibration_error",
            "maximum_calibration_error",
            "calibration_sha256",
        },
        "calibration evidence",
    )
    if document["calibration_sha256"] != _digest(
        {key: value for key, value in document.items() if key != "calibration_sha256"}
    ):
        _fail("calibration evidence digest mismatch")
    if probabilities is None:
        if document["applicability"] != "not_applicable" or document["bins"]:
            _fail("rules-only calibration semantics are not applicable")
        return
    if document["boundaries"] != [index / 10 for index in range(11)]:
        _fail("calibration bin boundaries differ from frozen values")
    bins = _sequence(document["bins"], "calibration bins")
    if len(bins) != 10:
        _fail("calibration bin count mismatch")
    for index, raw in enumerate(bins):
        bin_document = _mapping(raw, "calibration bin")
        _exact_fields(
            bin_document,
            {
                "lower",
                "upper",
                "final_closed",
                "count",
                "mean_probability",
                "empirical_rate",
                "absolute_gap",
                "event_ids",
                "bin_sha256",
            },
            "calibration bin",
        )
        if bin_document["bin_sha256"] != _digest(
            {key: value for key, value in bin_document.items() if key != "bin_sha256"}
        ):
            _fail("calibration bin digest mismatch")
        lower, upper = index / 10, (index + 1) / 10
        positions = [
            position
            for position, probability in enumerate(probabilities)
            if probability >= lower
            and (probability <= upper if index == 9 else probability < upper)
        ]
        event_ids = [support_ids[position] for position in positions]
        if bin_document["count"] != len(positions) or bin_document["event_ids"] != event_ids:
            _fail("calibration bin membership mismatch")
        mean = (
            sum(probabilities[position] for position in positions) / len(positions)
            if positions
            else None
        )
        empirical = (
            sum(labels[position] for position in positions) / len(positions) if positions else None
        )
        gap = abs(mean - empirical) if mean is not None and empirical is not None else None
        _close(bin_document["mean_probability"], mean, "calibration mean probability")
        _close(bin_document["empirical_rate"], empirical, "calibration empirical rate")
        _close(bin_document["absolute_gap"], gap, "calibration absolute gap")


def _verify_economics(
    document: Mapping[str, Any],
    *,
    rows: Sequence[Mapping[str, Any]],
    manifests: Mapping[str, Mapping[str, Any]],
) -> None:
    _exact_fields(
        document,
        {
            "currencies",
            "payment_count",
            "ledger_debit_by_currency",
            "ledger_credit_by_currency",
            "ledger_conserved",
            "payments",
            "by_currency",
            "by_campaign",
            "by_family",
            "by_rail",
            "economics_sha256",
        },
        "economic evidence",
    )
    if document["economics_sha256"] != _digest(
        {key: value for key, value in document.items() if key != "economics_sha256"}
    ):
        _fail("economic evidence digest mismatch")
    support_by_event = {str(row["support"]["event_id"]): row for row in rows}
    by_payment: dict[
        str,
        list[tuple[Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]]],
    ] = defaultdict(list)
    debit: dict[str, Decimal] = defaultdict(Decimal)
    credit: dict[str, Decimal] = defaultdict(Decimal)
    all_events: set[str] = set()
    for manifest in manifests.values():
        records = {item["event_id"]: item for item in manifest["event_records"]}
        for link in manifest["lineage"]:
            event_id = str(link["event_id"])
            if event_id in all_events:
                _fail("economic event lineage is duplicated")
            all_events.add(event_id)
            by_payment[str(link["payment_id"])].append((link, records[event_id], manifest))
        for posting in manifest["ledger_postings"]:
            currency = str(posting["currency"])
            debit[currency] += sum(
                (Decimal(str(amount)) for _account, amount in posting["debit"]), Decimal(0)
            )
            credit[currency] += sum(
                (Decimal(str(amount)) for _account, amount in posting["credit"]), Decimal(0)
            )
    if set(support_by_event) != all_events:
        _fail("economic support is not exhaustive")
    if debit != credit or document["ledger_conserved"] is not True:
        _fail("economic ledger totals do not conserve")
    if document["ledger_debit_by_currency"] != [
        [currency, str(value)] for currency, value in sorted(debit.items())
    ]:
        _fail("economic debit totals mismatch")
    if document["ledger_credit_by_currency"] != [
        [currency, str(value)] for currency, value in sorted(credit.items())
    ]:
        _fail("economic credit totals mismatch")
    expected_payments: list[dict[str, Any]] = []
    movement = {"card": {"settlement"}, "a2a": {"transfer_posted"}, "agentic": {"authorization"}}
    reversals = {"transfer_returned", "refund", "recovery"}
    authorization = {"authorization", "transfer_accepted"}
    for payment_id, facts in sorted(by_payment.items()):
        ordered = sorted(facts, key=lambda item: int(item[0]["lifecycle_position"]))
        if [item[0]["lifecycle_position"] for item in ordered] != list(range(len(ordered))):
            _fail("economic lifecycle positions are incomplete")
        links = [item[0] for item in ordered]
        payment_records = [item[1] for item in ordered]
        manifest = ordered[0][2]
        events = [json.loads(record["event_json"]) for record in payment_records]
        if (
            len({str(record["amount"]) for record in payment_records}) != 1
            or len({record["currency"] for record in payment_records}) != 1
        ):
            _fail("economic payment amount/currency changes across lifecycle")
        movement_times = [
            datetime.fromisoformat(str(event["decision_at"]).replace("Z", "+00:00"))
            for event in events
            if event["event_type"] in movement[str(manifest["rail"])]
        ]
        movement_time = min(movement_times) if movement_times else None
        intervention_times = [
            datetime.fromisoformat(str(event["decision_at"]).replace("Z", "+00:00"))
            for event in events
            if support_by_event[str(event["event_id"])]["action"] in _DETECTION
        ]
        first_intervention = min(intervention_times) if intervention_times else None
        intervened = first_intervention is not None and (
            movement_time is None or first_intervention <= movement_time
        )
        moved = movement_time is not None
        reversed_value = any(event["event_type"] in reversals for event in events)
        is_fraud = bool(links[0]["is_fraud"])
        captured = is_fraud and (not moved or reversed_value or intervened)
        escaped = is_fraud and moved and not captured
        values: dict[str, Any] = {
            "payment_id": payment_id,
            "campaign_id": manifest["campaign_id"],
            "family": manifest["family"],
            "rail": manifest["rail"],
            "currency": payment_records[0]["currency"],
            "is_fraud": is_fraud,
            "amount": str(payment_records[0]["amount"]),
            "authorized": any(event["event_type"] in authorization for event in events),
            "moved": moved,
            "reversed_or_recovered": reversed_value,
            "intervened_before_movement": intervened,
            "captured": captured,
            "escaped": escaped,
            "event_ids": [str(event["event_id"]) for event in events],
        }
        values["payment_sha256"] = _digest(values)
        expected_payments.append(values)
    if document["payments"] != expected_payments or document["payment_count"] != len(
        expected_payments
    ):
        _fail("payment-level economics failed lifecycle recomputation")
    for payment in document["payments"]:
        _exact_fields(
            _mapping(payment, "payment economics"),
            {
                "payment_id",
                "campaign_id",
                "family",
                "rail",
                "currency",
                "is_fraud",
                "amount",
                "authorized",
                "moved",
                "reversed_or_recovered",
                "intervened_before_movement",
                "captured",
                "escaped",
                "event_ids",
                "payment_sha256",
            },
            "payment economics",
        )
    # Every reported aggregate is rebuilt from its exact listed payment IDs.
    payment_map = {payment["payment_id"]: payment for payment in expected_payments}
    for group_name in ("by_currency", "by_campaign", "by_family", "by_rail"):
        for raw in document[group_name]:
            item = _mapping(raw, f"economic {group_name} stratum")
            _exact_fields(
                item,
                {
                    "family",
                    "rail",
                    "currency",
                    "payment_count",
                    "attempted_amount",
                    "malicious_amount",
                    "authorized_amount",
                    "settled_or_posted_amount",
                    "returned_refunded_recovered_amount",
                    "prevented_amount",
                    "captured_amount",
                    "escaped_amount",
                    "payment_ids",
                    "stratum_sha256",
                },
                "economic stratum",
            )
            selected = [payment_map[payment_id] for payment_id in item["payment_ids"]]
            if len(selected) != item["payment_count"] or len(set(item["payment_ids"])) != len(
                selected
            ):
                _fail("economic stratum payment coverage mismatch")
            expected = {
                "attempted_amount": sum(
                    (Decimal(payment["amount"]) for payment in selected), Decimal(0)
                ),
                "malicious_amount": sum(
                    (Decimal(payment["amount"]) for payment in selected if payment["is_fraud"]),
                    Decimal(0),
                ),
                "authorized_amount": sum(
                    (Decimal(payment["amount"]) for payment in selected if payment["authorized"]),
                    Decimal(0),
                ),
                "settled_or_posted_amount": sum(
                    (Decimal(payment["amount"]) for payment in selected if payment["moved"]),
                    Decimal(0),
                ),
                "returned_refunded_recovered_amount": sum(
                    (
                        Decimal(payment["amount"])
                        for payment in selected
                        if payment["reversed_or_recovered"]
                    ),
                    Decimal(0),
                ),
                "prevented_amount": sum(
                    (
                        Decimal(payment["amount"])
                        for payment in selected
                        if payment["is_fraud"]
                        and (not payment["moved"] or payment["intervened_before_movement"])
                    ),
                    Decimal(0),
                ),
                "captured_amount": sum(
                    (Decimal(payment["amount"]) for payment in selected if payment["captured"]),
                    Decimal(0),
                ),
                "escaped_amount": sum(
                    (Decimal(payment["amount"]) for payment in selected if payment["escaped"]),
                    Decimal(0),
                ),
            }
            if any(str(value) != str(item[name]) for name, value in expected.items()):
                _fail("economic stratum totals mismatch")
            if item["stratum_sha256"] != _digest(
                {key: value for key, value in item.items() if key != "stratum_sha256"}
            ):
                _fail("economic stratum digest mismatch")


def _verify_complete_metrics(
    complete: Mapping[str, Any],
    *,
    result: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    manifests: Mapping[str, Mapping[str, Any]],
) -> None:
    _exact_fields(
        complete,
        {
            "arm",
            "arm_result_sha256",
            "support_sha256",
            "aggregate",
            "calibration",
            "economics",
            "by_family",
            "bootstrap",
            "complete_metrics_sha256",
        },
        "complete arm metrics",
    )
    if complete["complete_metrics_sha256"] != _digest(
        {key: value for key, value in complete.items() if key != "complete_metrics_sha256"}
    ):
        _fail("complete metrics digest mismatch")
    if complete["arm"] != result["arm"] or complete["arm_result_sha256"] != result["result_sha256"]:
        _fail("complete metrics arm/result binding mismatch")
    support_ids = [str(row["support"]["event_id"]) for row in rows]
    labels = [int(row["support"]["label"]) for row in rows]
    actions = [str(row["action"]) for row in rows]
    probabilities = (
        None if result["arm"] == "rules_only" else [float(row["probability"]) for row in rows]
    )
    expectations = _metric_expectations(labels=labels, actions=actions, probabilities=probabilities)
    aggregate = _mapping(complete["aggregate"], "aggregate metrics")
    for name, expected in expectations.items():
        _verify_metric(
            _mapping(aggregate[name], f"aggregate {name}"),
            value=expected[0],
            numerator=expected[1],
            denominator=expected[2],
            applicability=expected[3],
            support_ids=support_ids,
        )
    latencies = [float(row["latency_ms"]) for row in rows]
    for percentile in (50, 95, 99):
        value = _percentile(latencies, percentile)
        _verify_metric(
            _mapping(aggregate[f"p{percentile}_latency_ms"], "latency metric"),
            value=value,
            numerator=value,
            denominator=1.0,
            applicability="defined",
            support_ids=support_ids,
        )
    _verify_calibration(
        _mapping(complete["calibration"], "calibration"),
        labels=labels,
        probabilities=probabilities,
        support_ids=support_ids,
    )
    economics = _mapping(complete["economics"], "economics")
    _verify_economics(economics, rows=rows, manifests=manifests)
    malicious_strata = [
        item for item in economics["by_currency"] if Decimal(str(item["malicious_amount"])) > 0
    ]
    captured_fractions = [
        float(Decimal(str(item["captured_amount"])) / Decimal(str(item["malicious_amount"])))
        for item in malicious_strata
    ]
    escaped_fractions = [
        float(Decimal(str(item["escaped_amount"])) / Decimal(str(item["malicious_amount"])))
        for item in malicious_strata
    ]
    for name, fractions in (
        ("captured_value_fraction", captured_fractions),
        ("escaped_value_fraction", escaped_fractions),
    ):
        _verify_metric(
            _mapping(aggregate[name], name),
            value=sum(fractions) / len(fractions) if fractions else None,
            numerator=sum(fractions) if fractions else 0.0,
            denominator=float(len(fractions)),
            applicability="defined" if fractions else "undefined",
            support_ids=support_ids,
        )
    fraud_campaigns = {
        row["support"]["campaign_id"] for row in rows if row["support"]["label"] == 1
    }
    detected_campaigns = {
        row["support"]["campaign_id"]
        for row in rows
        if row["support"]["label"] == 1 and row["action"] in _DETECTION
    }
    campaign_metric = _mapping(aggregate["campaign_detection_rate"], "campaign detection")
    _verify_metric(
        campaign_metric,
        value=_ratio(float(len(detected_campaigns)), float(len(fraud_campaigns))),
        numerator=float(len(detected_campaigns)),
        denominator=float(len(fraud_campaigns)),
        applicability="defined" if fraud_campaigns else "undefined",
        support_ids=support_ids,
    )
    _verify_family_metrics(
        _sequence(complete["by_family"], "family metrics"),
        rows=rows,
        economics=economics,
        manifests=manifests,
    )
    _verify_bootstrap(
        _mapping(complete["bootstrap"], "bootstrap"),
        rows=rows,
        economics=economics,
        aggregate=aggregate,
        family_metrics=_sequence(complete["by_family"], "family metrics"),
        probability_applicable=probabilities is not None,
    )


def _event_time_map(
    manifests: Mapping[str, Mapping[str, Any]],
) -> dict[str, datetime]:
    return {
        str(record["event_id"]): datetime.fromisoformat(
            str(record["decision_at"]).replace("Z", "+00:00")
        )
        for manifest in manifests.values()
        for record in manifest["event_records"]
    }


def _verify_family_metrics(
    documents: Sequence[object],
    *,
    rows: Sequence[Mapping[str, Any]],
    economics: Mapping[str, Any],
    manifests: Mapping[str, Mapping[str, Any]],
) -> None:
    families = [_mapping(item, "family metric") for item in documents]
    if [item["family"] for item in families] != list(_FAMILIES):
        _fail("family metric set/order differs from frozen families")
    times = _event_time_map(manifests)
    detected_legitimate = sum(
        row["support"]["label"] == 0 and row["action"] in _DETECTION for row in rows
    )
    reported_economics = _sequence(economics["by_family"], "family economics")
    for document in families:
        _exact_fields(
            document,
            {
                "family",
                "support_count",
                "campaign_count",
                "event_ids",
                "campaign_ids",
                "recall",
                "precision",
                "campaign_detection_rate",
                "time_to_first_alert",
                "action_distribution",
                "economic_strata",
                "campaign_alerts",
                "family_sha256",
            },
            "family metrics",
        )
        family = str(document["family"])
        selected = [
            row
            for row in rows
            if row["support"]["family"] == family and row["support"]["label"] == 1
        ]
        event_ids = [str(row["support"]["event_id"]) for row in selected]
        campaigns = sorted({str(row["support"]["campaign_id"]) for row in selected})
        if (
            document["support_count"] != len(selected)
            or document["campaign_count"] != len(campaigns)
            or document["event_ids"] != event_ids
            or document["campaign_ids"] != campaigns
        ):
            _fail("family support/campaign evidence mismatch")
        detected = sum(row["action"] in _DETECTION for row in selected)
        _verify_metric(
            _mapping(document["recall"], "family recall"),
            value=detected / len(selected),
            numerator=float(detected),
            denominator=float(len(selected)),
            applicability="defined",
            support_ids=event_ids,
        )
        precision_denominator = detected + detected_legitimate
        _verify_metric(
            _mapping(document["precision"], "family precision"),
            value=_ratio(float(detected), float(precision_denominator)),
            numerator=float(detected),
            denominator=float(precision_denominator),
            applicability="defined" if precision_denominator else "undefined",
            support_ids=event_ids,
        )
        alerts = [_mapping(item, "campaign alert") for item in document["campaign_alerts"]]
        if [item["campaign_id"] for item in alerts] != campaigns:
            _fail("campaign alert coverage mismatch")
        alert_seconds: list[float] = []
        detected_campaigns = 0
        for alert, campaign in zip(alerts, campaigns, strict=True):
            _exact_fields(
                alert,
                {
                    "campaign_id",
                    "family",
                    "event_ids",
                    "first_decision_at",
                    "first_alert_event_id",
                    "first_alert_at",
                    "time_to_first_alert_seconds",
                    "detected",
                    "alert_sha256",
                },
                "campaign alert",
            )
            fraud_rows = sorted(
                [row for row in selected if row["support"]["campaign_id"] == campaign],
                key=lambda row: (
                    times[str(row["support"]["event_id"])],
                    str(row["support"]["event_id"]),
                ),
            )
            campaign_rows = [row for row in rows if row["support"]["campaign_id"] == campaign]
            first_decision = min(times[str(row["support"]["event_id"])] for row in campaign_rows)
            detected_rows = [row for row in fraud_rows if row["action"] in _DETECTION]
            first = detected_rows[0] if detected_rows else None
            first_at = times[str(first["support"]["event_id"])] if first else None
            seconds = (first_at - first_decision).total_seconds() if first_at else None
            expected = {
                "campaign_id": campaign,
                "family": family,
                "event_ids": [str(row["support"]["event_id"]) for row in fraud_rows],
                "first_decision_at": first_decision.isoformat().replace("+00:00", "Z"),
                "first_alert_event_id": str(first["support"]["event_id"]) if first else None,
                "first_alert_at": first_at.isoformat().replace("+00:00", "Z") if first_at else None,
                "time_to_first_alert_seconds": seconds,
                "detected": first is not None,
            }
            if alert["alert_sha256"] != _digest(
                {key: value for key, value in alert.items() if key != "alert_sha256"}
            ) or any(alert[key] != value for key, value in expected.items()):
                _fail("campaign first-alert evidence mismatch")
            if first:
                detected_campaigns += 1
                assert seconds is not None
                alert_seconds.append(seconds)
        _verify_metric(
            _mapping(document["campaign_detection_rate"], "family campaign detection"),
            value=detected_campaigns / len(campaigns),
            numerator=float(detected_campaigns),
            denominator=float(len(campaigns)),
            applicability="defined",
            support_ids=event_ids,
        )
        _verify_metric(
            _mapping(document["time_to_first_alert"], "time to first alert"),
            value=(sum(alert_seconds) / len(alert_seconds) if alert_seconds else None),
            numerator=(float(sum(alert_seconds)) if alert_seconds else 0.0),
            denominator=float(len(alert_seconds)),
            applicability="defined" if alert_seconds else "undefined",
            support_ids=event_ids,
        )
        expected_actions = [
            [action, sum(row["action"] == action for row in selected)] for action in _ACTIONS
        ]
        if document["action_distribution"] != expected_actions:
            _fail("family action distribution mismatch")
        expected_strata = [item for item in reported_economics if item["family"] == family]
        if document["economic_strata"] != expected_strata:
            _fail("family ledger economics mismatch")
        if document["family_sha256"] != _digest(
            {key: value for key, value in document.items() if key != "family_sha256"}
        ):
            _fail("family metric digest mismatch")


def _bootstrap_series(
    *,
    rows: Sequence[Mapping[str, Any]],
    economics: Mapping[str, Any],
    samples: Sequence[Sequence[str]],
    campaign_rows: Mapping[str, Sequence[int]],
    campaign_family: Mapping[str, str],
    probability_applicable: bool,
    families: Sequence[str],
) -> dict[tuple[str, str | None], list[float]]:
    series: dict[tuple[str, str | None], list[float]] = defaultdict(list)
    payments_by_campaign: dict[str, list[Mapping[str, Any]]] = {
        campaign: [
            payment for payment in economics["payments"] if payment["campaign_id"] == campaign
        ]
        for campaign in campaign_rows
    }
    for drawn in samples:
        indices = [index for campaign in drawn for index in campaign_rows[campaign]]
        labels = [int(rows[index]["support"]["label"]) for index in indices]
        actions = [str(rows[index]["action"]) for index in indices]
        fraud_count = sum(labels)
        legitimate_count = len(labels) - fraud_count
        if fraud_count:
            series[("recall", None)].append(
                sum(
                    action in _DETECTION and label == 1
                    for action, label in zip(actions, labels, strict=True)
                )
                / fraud_count
            )
        if legitimate_count:
            for metric, action_name in (
                ("false_decline_rate", "decline_hold"),
                ("challenge_rate", "challenge"),
                ("review_rate", "review_hold"),
            ):
                series[(metric, None)].append(
                    sum(
                        action == action_name and label == 0
                        for action, label in zip(actions, labels, strict=True)
                    )
                    / legitimate_count
                )
        fraud_draws = [campaign for campaign in drawn if campaign_family[campaign] != "legitimate"]
        if fraud_draws:
            series[("campaign_detection_rate", None)].append(
                sum(
                    any(
                        rows[index]["support"]["label"] == 1 and rows[index]["action"] in _DETECTION
                        for index in campaign_rows[campaign]
                    )
                    for campaign in fraud_draws
                )
                / len(fraud_draws)
            )
        selected_payments = [
            payment
            for campaign in fraud_draws
            for payment in payments_by_campaign[campaign]
            if payment["is_fraud"]
        ]
        currency_fractions: list[float] = []
        for currency in sorted({str(payment["currency"]) for payment in selected_payments}):
            currency_payments = [
                payment for payment in selected_payments if payment["currency"] == currency
            ]
            malicious = sum(
                (Decimal(str(payment["amount"])) for payment in currency_payments),
                Decimal(0),
            )
            captured = sum(
                (
                    Decimal(str(payment["amount"]))
                    for payment in currency_payments
                    if payment["captured"]
                ),
                Decimal(0),
            )
            if malicious:
                currency_fractions.append(float(captured / malicious))
        if currency_fractions:
            series[("captured_value_fraction", None)].append(
                sum(currency_fractions) / len(currency_fractions)
            )
        if probability_applicable:
            probabilities = [float(rows[index]["probability"]) for index in indices]
            ece = 0.0
            for bin_index in range(10):
                lower = bin_index / 10
                upper = (bin_index + 1) / 10
                positions = [
                    position
                    for position, probability in enumerate(probabilities)
                    if probability >= lower
                    and (probability <= upper if bin_index == 9 else probability < upper)
                ]
                if positions:
                    mean_probability = sum(probabilities[position] for position in positions) / len(
                        positions
                    )
                    empirical = sum(labels[position] for position in positions) / len(positions)
                    ece += len(positions) / len(labels) * abs(mean_probability - empirical)
            series[("expected_calibration_error", None)].append(ece)
        for family in families:
            family_draws = [campaign for campaign in drawn if campaign_family[campaign] == family]
            if not family_draws:
                continue
            fraud_indices = [
                index
                for campaign in family_draws
                for index in campaign_rows[campaign]
                if rows[index]["support"]["label"] == 1
            ]
            if fraud_indices:
                series[("recall", family)].append(
                    sum(rows[index]["action"] in _DETECTION for index in fraud_indices)
                    / len(fraud_indices)
                )
            series[("campaign_detection_rate", family)].append(
                sum(
                    any(
                        rows[index]["support"]["label"] == 1 and rows[index]["action"] in _DETECTION
                        for index in campaign_rows[campaign]
                    )
                    for campaign in family_draws
                )
                / len(family_draws)
            )
    return series


def _verify_bootstrap(
    document: Mapping[str, Any],
    *,
    rows: Sequence[Mapping[str, Any]],
    economics: Mapping[str, Any],
    aggregate: Mapping[str, Any],
    family_metrics: Sequence[object],
    probability_applicable: bool,
) -> None:
    _exact_fields(
        document,
        {
            "seed",
            "replicates",
            "confidence_level",
            "interval_method",
            "resampling_unit",
            "stratification",
            "strata",
            "samples",
            "intervals",
            "bootstrap_sha256",
        },
        "bootstrap evidence",
    )
    if (
        document["seed"] != 707
        or document["replicates"] != 2000
        or document["confidence_level"] != 0.95
        or document["interval_method"] != "percentile"
        or document["resampling_unit"] != "campaign"
        or document["stratification"] != "legitimate_and_each_fraud_family"
    ):
        _fail("bootstrap design differs from frozen campaign design")
    if document["bootstrap_sha256"] != _digest(
        {key: value for key, value in document.items() if key != "bootstrap_sha256"}
    ):
        _fail("bootstrap evidence digest mismatch")
    unsorted_campaign_rows: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        unsorted_campaign_rows[str(row["support"]["campaign_id"])].append(index)
    campaign_rows = {
        campaign: unsorted_campaign_rows[campaign] for campaign in sorted(unsorted_campaign_rows)
    }
    campaign_family: dict[str, str] = {}
    for campaign, indices in campaign_rows.items():
        families = {
            str(rows[index]["support"]["family"])
            for index in indices
            if rows[index]["support"]["label"] == 1
        }
        if len(families) > 1:
            _fail("bootstrap campaign spans fraud families")
        campaign_family[campaign] = next(iter(families), "legitimate")
    strata: dict[str, list[str]] = {}
    for family in ("legitimate", *_FAMILIES):
        selected = [campaign for campaign, value in campaign_family.items() if value == family]
        if not selected:
            _fail("bootstrap stratum is empty")
        strata[family] = selected
    if document["strata"] != [[name, campaigns] for name, campaigns in sorted(strata.items())]:
        _fail("bootstrap strata membership mismatch")
    rng = np.random.Generator(np.random.PCG64(707))
    expected_draws: list[list[str]] = []
    samples = _sequence(document["samples"], "bootstrap samples")
    if len(samples) != 2000:
        _fail("bootstrap sample count mismatch")
    for replicate, raw in enumerate(samples):
        sample = _mapping(raw, "bootstrap sample")
        _exact_fields(
            sample,
            {"replicate", "campaign_ids", "event_ids", "sample_sha256"},
            "bootstrap sample",
        )
        drawn: list[str] = []
        for campaigns in strata.values():
            positions = rng.integers(0, len(campaigns), size=len(campaigns))
            drawn.extend(campaigns[int(position)] for position in positions)
        event_ids = [
            str(rows[index]["support"]["event_id"])
            for campaign in drawn
            for index in campaign_rows[campaign]
        ]
        if (
            sample["replicate"] != replicate
            or sample["campaign_ids"] != drawn
            or sample["event_ids"] != event_ids
        ):
            _fail("bootstrap deterministic campaign sample mismatch")
        if sample["sample_sha256"] != _digest(
            {key: value for key, value in sample.items() if key != "sample_sha256"}
        ):
            _fail("bootstrap sample digest mismatch")
        expected_draws.append(drawn)
    family_documents = [_mapping(item, "family metric") for item in family_metrics]
    family_names = [str(item["family"]) for item in family_documents]
    series = _bootstrap_series(
        rows=rows,
        economics=economics,
        samples=expected_draws,
        campaign_rows=campaign_rows,
        campaign_family=campaign_family,
        probability_applicable=probability_applicable,
        families=family_names,
    )
    family_points = {
        ("recall", str(item["family"])): item["recall"] for item in family_documents
    } | {
        ("campaign_detection_rate", str(item["family"])): item["campaign_detection_rate"]
        for item in family_documents
    }
    for raw in document["intervals"]:
        interval = _mapping(raw, "bootstrap interval")
        _exact_fields(
            interval,
            {
                "metric",
                "family",
                "applicability",
                "point",
                "lower",
                "upper",
                "defined_replicates",
                "sample_values_sha256",
                "interval_sha256",
            },
            "bootstrap interval",
        )
        key = (str(interval["metric"]), interval["family"])
        values = series.get(key, [])
        point_document = family_points.get(key) if key[1] is not None else aggregate.get(key[0])
        if point_document is None:
            _fail("bootstrap interval lacks a point metric")
        applicable = point_document["applicability"] == "defined" and bool(values)
        expected_point = point_document["value"] if applicable else None
        expected_lower = _percentile(values, 2.5) if applicable else None
        expected_upper = _percentile(values, 97.5) if applicable else None
        expected_applicability = point_document["applicability"] if not applicable else "defined"
        if interval["applicability"] != expected_applicability or interval[
            "defined_replicates"
        ] != len(values):
            _fail("bootstrap interval applicability/sample count mismatch")
        _close(interval["point"], expected_point, "bootstrap interval point")
        _close(interval["lower"], expected_lower, "bootstrap lower bound")
        _close(interval["upper"], expected_upper, "bootstrap upper bound")
        if interval["sample_values_sha256"] != _digest(values):
            _fail(f"bootstrap sample value digest mismatch for {key!r}")
        if interval["interval_sha256"] != _digest(
            {name: value for name, value in interval.items() if name != "interval_sha256"}
        ):
            _fail("bootstrap interval digest mismatch")


def _verify_measurement(measurement: Mapping[str, Any], *, support_sha256: str) -> None:
    _exact_fields(
        measurement,
        {
            "name",
            "applicability",
            "before",
            "after",
            "delta",
            "numerator",
            "denominator",
            "support_sha256",
        },
        "control measurement",
    )
    if measurement["support_sha256"] != support_sha256:
        _fail("control measurement support mismatch")
    applicability = measurement["applicability"]
    numeric = [
        measurement.get("before"),
        measurement.get("after"),
        measurement.get("delta"),
        measurement.get("numerator"),
        measurement.get("denominator"),
    ]
    if any(value is not None and not math.isfinite(float(value)) for value in numeric):
        _fail("control measurement contains non-finite values")
    if applicability == "defined":
        if (
            measurement["denominator"] is None
            or measurement["denominator"] <= 0
            or measurement["numerator"] is None
        ):
            _fail("defined control measurement lacks a positive denominator")
        if measurement["before"] is not None and measurement["after"] is not None:
            _close(
                measurement["delta"],
                float(measurement["after"]) - float(measurement["before"]),
                "control measurement delta",
            )
    elif applicability == "undefined":
        if measurement["denominator"] != 0 or any(
            measurement[name] is not None for name in ("before", "after", "delta")
        ):
            _fail("undefined control measurement semantics are invalid")
    elif any(value is not None for value in numeric):
        _fail("non-applicable control measurement claims values")


def _verify_controls(
    suite: Mapping[str, Any],
    *,
    protocol: Mapping[str, Any],
    support_ids: Sequence[str],
    execution_manifests: Mapping[str, Mapping[str, Any]],
    reference_rows: Sequence[Mapping[str, Any]],
) -> None:
    _exact_fields(
        suite,
        {
            "controls",
            "evidence_protocol_sha256",
            "support_sha256",
            "implementation_sha256",
            "suite_sha256",
        },
        "executed control suite",
    )
    if suite["suite_sha256"] != _digest(
        {key: value for key, value in suite.items() if key != "suite_sha256"}
    ):
        _fail("control suite digest mismatch")
    if (
        suite["evidence_protocol_sha256"] != protocol["evidence_protocol_sha256"]
        or suite["implementation_sha256"] != protocol["implementation_sha256"]
    ):
        _fail("control suite implementation/protocol binding mismatch")
    if suite["support_sha256"] != _digest(list(support_ids)):
        _fail("control suite support mismatch")
    controls = [_mapping(item, "executed control") for item in suite["controls"]]
    expected_names = [
        "label_shuffle",
        "identity_rename",
        "future_causality",
        "equal_time_isolation",
        "benign_only",
        "fraud_only_diagnostic",
        "feature_leakage",
    ]
    if [item["name"] for item in controls] != expected_names:
        _fail("executed control set/order mismatch")
    baseline_artifact_ids = list(execution_manifests)
    event_to_artifact = {
        str(link["event_id"]): evidence_sha256
        for evidence_sha256, manifest in execution_manifests.items()
        for link in manifest["lineage"]
    }
    base_event_times = {
        str(record["event_id"]): datetime.fromisoformat(
            str(record["decision_at"]).replace("Z", "+00:00")
        )
        for manifest in execution_manifests.values()
        for record in manifest["event_records"]
    }
    control_baseline_digests = controls[1]["executed_arm_spec_sha256"]
    if [item[0] for item in control_baseline_digests] != list(_ARMS):
        _fail("control baseline arm specification set/order mismatch")
    for control in controls:
        _exact_fields(
            control,
            {
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
            },
            "executed control",
        )
        if control["executed"] is not True:
            _fail("descriptive control is not executed")
        spec = json.loads(str(control["spec_json"]))
        rows = json.loads(str(control["row_evidence_json"]))
        if control["spec_json"] != json.dumps(spec, sort_keys=True) or control[
            "row_evidence_json"
        ] != json.dumps(rows, sort_keys=True):
            _fail("control JSON evidence is not canonical")
        if control["spec_sha256"] != _digest(spec) or control["row_evidence_sha256"] != _digest(
            rows
        ):
            _fail("control specification/row digest mismatch")
        if control["input_support_ids"] != list(support_ids) and control["name"] not in {
            "benign_only",
            "fraud_only_diagnostic",
        }:
            _fail("control input support differs from evaluation support")
        if control["input_support_sha256"] != _digest(control["input_support_ids"]):
            _fail("control input support digest mismatch")
        if control["input_artifact_sha256"] != _digest(control["input_artifact_ids"]):
            _fail("control artifact digest mismatch")
        if control["implementation_sha256"] != protocol["implementation_sha256"]:
            _fail("control implementation digest mismatch")
        for measurement in control["measurements"]:
            _verify_measurement(
                _mapping(measurement, "control measurement"),
                support_sha256=str(control["input_support_sha256"]),
            )
        if control["control_sha256"] != _digest(
            {key: value for key, value in control.items() if key != "control_sha256"}
        ):
            _fail("executed control digest mismatch")
        name = str(control["name"])
        if control["qualifies_for_readiness"] != (name != "fraud_only_diagnostic"):
            _fail("control readiness qualification differs from frozen semantics")
        expected_artifacts = sorted(
            {event_to_artifact[event_id] for event_id in control["input_support_ids"]}
        )
        if name != "future_causality" and control["input_artifact_ids"] != expected_artifacts:
            _fail("control input artifacts do not exactly cover its support")
        if name == "label_shuffle":
            labels = [int(value) for value in rows["labels"]]
            prevalence = sum(labels) / len(labels)
            if not math.isclose(rows["prevalence"], prevalence, rel_tol=1e-12, abs_tol=1e-12):
                _fail("label-shuffle prevalence mismatch")
            if control["permutation_seed"] != 1707:
                _fail("label-shuffle seed mismatch")
            permutations = rows["permutation_indices"]
            if rows["permutation_indices_sha256"] != _digest(permutations):
                _fail("label-shuffle permutation digest mismatch")
            shuffled_labels = rows["permuted_labels"]
            partition_labels = rows["partition_labels"]
            if not (len(permutations) == len(shuffled_labels) == len(partition_labels) == 3):
                _fail("label-shuffle partition evidence is incomplete")
            for original, indices, values in zip(
                partition_labels, permutations, shuffled_labels, strict=True
            ):
                if len(original) != len(indices) or sorted(indices) != list(range(len(indices))):
                    _fail("label-shuffle permutation is not bijective")
                if values != [original[index] for index in indices]:
                    _fail("label-shuffle values do not follow retained permutation")
            passed = True
            roc_before: list[float] = []
            roc_after: list[float] = []
            pr_before: list[float] = []
            pr_after: list[float] = []
            for arm in rows["arms"]:
                baseline = [float(value) for value in arm["baseline_probabilities"]]
                shuffled = [float(value) for value in arm["shuffled_probabilities"]]
                if len(baseline) != len(labels) or len(shuffled) != len(labels):
                    _fail("label-shuffle probability support mismatch")
                if arm["baseline_probability_sha256"] != _digest(baseline) or arm[
                    "shuffled_probability_sha256"
                ] != _digest(shuffled):
                    _fail("label-shuffle probability digest mismatch")
                before_roc = _roc_auc(labels, baseline)
                after_roc = _roc_auc(labels, shuffled)
                before_pr = _average_precision(labels, baseline)
                after_pr = _average_precision(labels, shuffled)
                for field, expected in (
                    ("baseline_roc_auc", before_roc),
                    ("shuffled_roc_auc", after_roc),
                    ("baseline_pr_auc", before_pr),
                    ("shuffled_pr_auc", after_pr),
                ):
                    _close(arm[field], expected, f"label-shuffle {field}")
                passed &= (
                    after_roc <= 0.70
                    and after_pr <= prevalence + 0.20
                    and before_roc - after_roc >= 0.05
                )
                roc_before.append(before_roc)
                roc_after.append(after_roc)
                pr_before.append(before_pr)
                pr_after.append(after_pr)
            if control["executed_arm_spec_sha256"] != [
                [item["arm"], item["control_arm_sha256"]] for item in rows["arms"]
            ]:
                _fail("label-shuffle executed model digest mismatch")
            if control["passed"] != passed:
                _fail("label-shuffle criterion mismatch")
            measurements = {item["name"]: item for item in control["measurements"]}
            _close(
                measurements["roc_auc"]["before"],
                sum(roc_before) / len(roc_before),
                "label ROC before",
            )
            _close(
                measurements["roc_auc"]["after"], sum(roc_after) / len(roc_after), "label ROC after"
            )
            _close(
                measurements["pr_auc"]["before"], sum(pr_before) / len(pr_before), "label PR before"
            )
            _close(measurements["pr_auc"]["after"], sum(pr_after) / len(pr_after), "label PR after")
        elif name in {
            "identity_rename",
            "future_causality",
            "equal_time_isolation",
            "feature_leakage",
        }:
            before_matrix = rows["before_matrix"]
            after_matrix = rows["after_matrix"]
            before_scores = rows["before_score_signature"]
            after_scores = rows["after_score_signature"]
            if (
                rows["before_matrix_sha256"] != _digest(before_matrix)
                or rows["after_matrix_sha256"] != _digest(after_matrix)
                or rows["before_score_sha256"] != _digest(before_scores)
                or rows["after_score_sha256"] != _digest(after_scores)
            ):
                _fail(f"{name} retained before/after digest mismatch")
            if (
                len(before_matrix) != len(support_ids)
                or len(after_matrix) != len(support_ids)
                or any(
                    not math.isfinite(float(value))
                    for matrix in (before_matrix, after_matrix)
                    for matrix_row in matrix
                    for value in matrix_row
                )
            ):
                _fail(f"{name} retained feature matrices are invalid")
            matrix_equal = before_matrix == after_matrix
            score_equal = before_scores == after_scores
            invariance_measurements = {item["name"]: item for item in control["measurements"]}
            for metric, invariant in (
                ("numeric_feature_rows", matrix_equal),
                ("prediction_rows", score_equal),
            ):
                measurement = invariance_measurements[metric]
                expected_after = float(len(support_ids)) if invariant else 0.0
                _close(
                    measurement["before"],
                    float(len(support_ids)),
                    f"{name} {metric} baseline",
                )
                _close(
                    measurement["after"],
                    expected_after,
                    f"{name} {metric} transformed",
                )
                _close(
                    measurement["numerator"],
                    expected_after,
                    f"{name} {metric} numerator",
                )
                _close(
                    measurement["denominator"],
                    float(len(support_ids)),
                    f"{name} {metric} denominator",
                )
            extra = True
            if name == "identity_rename":
                mapping = dict(rows["bijection"])
                if len(mapping) != rows["bijection_size"] or len(set(mapping.values())) != len(
                    mapping
                ):
                    _fail("identity-rename mapping is not a bijection")
                domains: dict[str, set[str]] = {
                    name: set()
                    for name in (
                        "account",
                        "actor",
                        "authentication_evidence",
                        "campaign",
                        "command",
                        "counterparty",
                        "credential",
                        "device",
                        "event",
                        "evidence",
                        "merchant",
                        "payee",
                        "payment",
                        "request",
                    )
                }
                for manifest in execution_manifests.values():
                    domains["campaign"].add(str(manifest["campaign_id"]))
                    domains["evidence"].update(
                        {
                            str(manifest["evidence_sha256"]),
                            str(manifest["artifact_sha256"]),
                        }
                    )
                    for source, target in (
                        ("account_ids", "account"),
                        ("authentication_evidence_ids", "authentication_evidence"),
                        ("credential_ids", "credential"),
                        ("device_ids", "device"),
                        ("merchant_ids", "merchant"),
                        ("payee_ids", "payee"),
                        ("trust_request_ids", "request"),
                    ):
                        domains[target].update(str(value) for value in manifest[source])
                    for link in manifest["lineage"]:
                        for field, target in (
                            ("actor_id", "actor"),
                            ("command_id", "command"),
                            ("counterparty_id", "counterparty"),
                            ("event_id", "event"),
                            ("payment_id", "payment"),
                        ):
                            domains[target].add(str(link[field]))
                    for trust in manifest["trust_records"]:
                        domains["request"].add(str(trust["request_id"]))
                        domains["evidence"].update(
                            {
                                str(trust["receipt_hash"]),
                                str(trust["request_hash"]),
                                str(trust["signature_hash"]),
                            }
                        )
                expected_domains = {
                    domain: {
                        "count": len(values),
                        "original_sha256": _digest(sorted(values)),
                        "renamed_sha256": _digest(sorted(mapping[value] for value in values)),
                    }
                    for domain, values in sorted(domains.items())
                }
                if set(mapping) != set().union(*domains.values()):
                    _fail("identity-rename mapping does not exactly cover identity domains")
                if rows["identity_domains"] != expected_domains:
                    _fail("identity-rename domain evidence mismatch")
                original = rows["original_relationships"]
                renamed = [[mapping[value] for value in relationship] for relationship in original]
                if renamed != rows["renamed_relationships"]:
                    _fail("identity-rename relationships are not preserved")
                if (
                    rows["original_relationship_sha256"] != _digest(original)
                    or rows["renamed_relationship_sha256"] != _digest(renamed)
                    or rows["bijection_sha256"] != _digest(mapping)
                ):
                    _fail("identity-rename relationship/mapping digest mismatch")
            elif name == "future_causality":
                inserted_artifacts = [
                    _mapping(item, "future execution artifact")
                    for item in rows["inserted_execution_artifacts"]
                ]
                future_manifests = {
                    artifact["evidence_sha256"]: _verify_manifest(artifact)
                    for artifact in inserted_artifacts
                }
                if len(future_manifests) != len(inserted_artifacts):
                    _fail("future control artifacts are duplicated")
                future_event_map = {
                    str(record["event_id"]): (evidence_sha256, record)
                    for evidence_sha256, manifest in future_manifests.items()
                    for record in manifest["event_records"]
                }
                if any(event_id not in base_event_times for event_id in support_ids):
                    _fail("future control baseline event is unresolved")
                baseline_times = [base_event_times[event_id] for event_id in support_ids]
                baseline_max = max(baseline_times)
                if (
                    datetime.fromisoformat(
                        str(rows["baseline_max_decision_at"]).replace("Z", "+00:00")
                    )
                    != baseline_max
                ):
                    _fail("future control baseline decision bound mismatch")
                inserted_ids: list[str] = []
                for raw_inserted in rows["inserted_events"]:
                    inserted = _mapping(raw_inserted, "inserted future event")
                    event_id = str(inserted["event_id"])
                    resolved = future_event_map.get(event_id)
                    if resolved is None:
                        _fail("future control event is absent from execution evidence")
                    evidence_sha256, record = resolved
                    decision_at = datetime.fromisoformat(
                        str(record["decision_at"]).replace("Z", "+00:00")
                    )
                    if (
                        inserted["execution_evidence_sha256"] != evidence_sha256
                        or datetime.fromisoformat(
                            str(inserted["decision_at"]).replace("Z", "+00:00")
                        )
                        != decision_at
                        or (decision_at - baseline_max).total_seconds()
                        < int(protocol["controls"]["future_causality"]["offset_seconds"])
                    ):
                        _fail("future control inserted event bound mismatch")
                    inserted_ids.append(event_id)
                expected_future_ids = list(future_manifests)
                extra = bool(
                    rows["future_rows_are_retained_execution_evidence"]
                    and rows["inserted_event_ids"] == inserted_ids
                    and rows["inserted_execution_evidence_sha256"] == sorted(expected_future_ids)
                    and control["input_artifact_ids"]
                    == [*baseline_artifact_ids, *expected_future_ids]
                )
            elif name == "equal_time_isolation":
                peers = set(rows["peer_event_ids"])
                if not peers <= set(base_event_times):
                    _fail("equal-time peer event is unresolved")
                peer_times = {base_event_times[event_id] for event_id in peers}
                source_by_event = {
                    str(row["support"]["event_id"]): row["rule_source_event_ids"]
                    for row in reference_rows
                }
                extra = (
                    len(peers) == 2
                    and len(peer_times) == 1
                    and datetime.fromisoformat(str(rows["peer_decision_at"]).replace("Z", "+00:00"))
                    in peer_times
                    and rows["peer_source_event_ids"]
                    == {event_id: source_by_event[event_id] for event_id in peers}
                    and bool(rows["peers_do_not_observe_each_other"])
                ) and all(
                    not (set(source_ids) & peers)
                    for source_ids in rows["peer_source_event_ids"].values()
                )
            else:
                mutated = rows["mutated_rows"]
                extra = (
                    rows["mutated_fields"]
                    == protocol["controls"]["feature_leakage"]["forbidden_fields"]
                    and len(mutated) == len(reference_rows)
                    and all(
                        item["event_id"] == original["support"]["event_id"]
                        and item["is_fraud"] != bool(original["support"]["label"])
                        and item["family"] != original["support"]["family"]
                        and item["campaign_id"] != original["support"]["campaign_id"]
                        and item["lifecycle_state"] == "mutated-final-outcome"
                        for item, original in zip(mutated, reference_rows, strict=True)
                    )
                )
            if control["passed"] != (matrix_equal and score_equal and extra):
                _fail(f"{name} invariant criterion mismatch")
        else:
            arms = rows["arms"]
            if set(arms) != set(_ARMS):
                _fail("single-class control arm set mismatch")
            measurements = {item["name"]: item for item in control["measurements"]}
            for arm, evidence in arms.items():
                evidence_rows = evidence["rows"]
                if not evidence_rows:
                    _fail("single-class control lacks row evidence")
                if [item["event_id"] for item in evidence_rows] != control["input_support_ids"]:
                    _fail("single-class control row order/support mismatch")
                actions = [str(item["action"]) for item in evidence_rows]
                probabilities = [float(item["probability"]) for item in evidence_rows]
                latencies = [float(item["latency_ms"]) for item in evidence_rows]
                if any(
                    action not in _ACTIONS
                    or not 0.0 <= probability <= 1.0
                    or not math.isfinite(latency)
                    or latency < 0.0
                    for action, probability, latency in zip(
                        actions, probabilities, latencies, strict=True
                    )
                ):
                    _fail("single-class control row values are invalid")
                if arm == "rules_only" and evidence.get("probability_quantiles") is not None:
                    _fail("rules-only control invents probability calibration")
                if arm != "rules_only":
                    expected_quantiles = [
                        _percentile(probabilities, percentile) for percentile in (5, 50, 95)
                    ]
                    if evidence.get("probability_quantiles") != expected_quantiles:
                        _fail("single-class probability distribution mismatch")
                p95 = _percentile(latencies, 95)
                latency_measurement = measurements[f"{arm}.p95_latency_ms"]
                _close(latency_measurement["before"], p95, "control p95 latency")
                _close(latency_measurement["after"], p95, "control p95 latency")
                _close(latency_measurement["numerator"], p95, "control p95 numerator")
                if name == "benign_only":
                    decline = sum(action == "decline_hold" for action in actions)
                    challenge = sum(action == "challenge" for action in actions)
                    review = sum(action == "review_hold" for action in actions)
                    if (
                        evidence["false_decline_numerator"] != decline
                        or evidence["challenge_numerator"] != challenge
                        or evidence["review_numerator"] != review
                        or evidence["legitimate_denominator"] != len(actions)
                    ):
                        _fail("benign-only workload counts mismatch")
                    for metric, numerator in (
                        ("false_decline_rate", decline),
                        ("challenge_rate", challenge),
                        ("review_rate", review),
                    ):
                        measurement = measurements[f"{arm}.{metric}"]
                        rate = numerator / len(actions)
                        _close(measurement["before"], rate, metric)
                        _close(measurement["after"], rate, metric)
                        _close(measurement["numerator"], float(numerator), metric)
                        _close(
                            measurement["denominator"],
                            float(len(actions)),
                            metric,
                        )
                    if arm != "rules_only":
                        expected_brier = sum(value**2 for value in probabilities) / len(
                            probabilities
                        )
                        _close(evidence["brier"], expected_brier, "benign Brier")
                else:
                    interventions = sum(action != "approve" for action in actions)
                    if evidence["recall_numerator"] != interventions or evidence[
                        "recall_denominator"
                    ] != len(actions):
                        _fail("fraud-only diagnostic counts mismatch")
                    measurement = measurements[f"{arm}.recall"]
                    recall = interventions / len(actions)
                    _close(measurement["before"], recall, "fraud diagnostic recall")
                    _close(measurement["after"], recall, "fraud diagnostic recall")
                    _close(
                        measurement["numerator"],
                        float(interventions),
                        "fraud diagnostic recall numerator",
                    )
            if name == "fraud_only_diagnostic":
                if control["qualifies_for_readiness"] is not False or control["passed"] is not True:
                    _fail("fraud-only diagnostic improperly qualifies readiness")
            elif control["passed"] != all(
                sum(item["action"] == "decline_hold" for item in evidence["rows"])
                / len(evidence["rows"])
                <= 0.001
                and sum(item["action"] == "challenge" for item in evidence["rows"])
                / len(evidence["rows"])
                <= 0.02
                and sum(item["action"] == "review_hold" for item in evidence["rows"])
                / len(evidence["rows"])
                <= 0.01
                for evidence in arms.values()
            ):
                _fail("benign-only workload criterion mismatch")
        if (
            name != "label_shuffle"
            and control["executed_arm_spec_sha256"] != control_baseline_digests
        ):
            _fail("executed controls do not share one frozen baseline arm set")


def _verify_readiness(
    readiness: Mapping[str, Any],
    *,
    complete_metrics: Mapping[str, Any],
    controls: Mapping[str, Any],
) -> None:
    _exact_fields(
        readiness,
        {
            "evaluated_arm",
            "gates",
            "qualifying_controls",
            "status",
            "readiness_sha256",
        },
        "readiness evidence",
    )
    if readiness["evaluated_arm"] != "full_sentinel":
        _fail("readiness is not bound to full_sentinel")
    if readiness["readiness_sha256"] != _digest(
        {key: value for key, value in readiness.items() if key != "readiness_sha256"}
    ):
        _fail("readiness evidence digest mismatch")
    intervals = {
        (item["metric"], item["family"]): item
        for item in complete_metrics["bootstrap"]["intervals"]
    }
    aggregate = complete_metrics["aggregate"]
    expected_gates = [
        ("family_recall", family, "lower_bound_gte", 0.75) for family in _FAMILIES
    ] + [
        ("false_decline_rate", None, "upper_bound_lte", 0.001),
        ("manual_review_rate", None, "upper_bound_lte", 0.01),
        ("challenge_rate", None, "upper_bound_lte", 0.02),
        ("captured_value_fraction", None, "lower_bound_gte", 0.70),
        ("expected_calibration_error", None, "upper_bound_lte", 0.10),
        ("p95_latency_ms", None, "point_lte", 50.0),
        ("campaign_detection_rate", None, "defined_interval", None),
    ]
    observed_gates = [
        (
            gate["metric"],
            gate["family"],
            gate["comparison"],
            gate["target"],
        )
        for gate in readiness["gates"]
    ]
    if observed_gates != expected_gates:
        _fail("readiness gate set/order differs from frozen semantics")
    for raw in readiness["gates"]:
        gate = _mapping(raw, "readiness gate")
        _exact_fields(
            gate,
            {
                "metric",
                "family",
                "comparison",
                "target",
                "applicability",
                "point",
                "lower",
                "upper",
                "passed",
                "source_sha256",
                "gate_sha256",
            },
            "readiness gate",
        )
        if gate["gate_sha256"] != _digest(
            {key: value for key, value in gate.items() if key != "gate_sha256"}
        ):
            _fail("readiness gate digest mismatch")
        metric = str(gate["metric"])
        if metric == "family_recall":
            source = intervals[("recall", gate["family"])]
        elif metric == "manual_review_rate":
            source = intervals[("review_rate", None)]
        elif metric == "p95_latency_ms":
            source = aggregate["p95_latency_ms"]
        else:
            source = intervals[(metric, None)]
        expected_source = source.get("interval_sha256", source.get("metric_sha256"))
        if gate["source_sha256"] != expected_source:
            _fail("readiness gate source binding mismatch")
        expected_point = source.get("point", source.get("value"))
        expected_lower = source.get("lower")
        expected_upper = source.get("upper")
        expected_applicability = source["applicability"]
        _close(gate["point"], expected_point, "readiness point")
        _close(gate["lower"], expected_lower, "readiness lower")
        _close(gate["upper"], expected_upper, "readiness upper")
        if gate["applicability"] != expected_applicability:
            _fail("readiness gate applicability mismatch")
        passed = False
        if expected_applicability == "defined":
            if gate["comparison"] == "lower_bound_gte":
                passed = expected_lower is not None and expected_lower >= gate["target"]
            elif gate["comparison"] == "upper_bound_lte":
                passed = expected_upper is not None and expected_upper <= gate["target"]
            elif gate["comparison"] == "point_lte":
                passed = expected_point is not None and expected_point <= gate["target"]
            else:
                passed = expected_lower is not None and expected_upper is not None
        if gate["passed"] != passed:
            _fail("readiness gate result mismatch")
    qualifying = [
        [control["name"], control["passed"], control["control_sha256"]]
        for control in controls["controls"]
        if control["qualifies_for_readiness"]
    ]
    if readiness["qualifying_controls"] != qualifying:
        _fail("readiness control evidence mismatch")
    expected_status = (
        "ready"
        if all(item["passed"] for item in readiness["gates"])
        and all(item[1] for item in qualifying)
        else "not_ready"
    )
    if readiness["status"] != expected_status:
        _fail("final readiness status mismatch")


def _verify_evidence_bytes(serialized: bytes, *, root: Path) -> dict[str, object]:
    envelope, payload = _parse_envelope(serialized)
    _exact_fields(
        payload,
        {
            "schema_version",
            "safe_seed",
            "evidence_protocol",
            "catalog_sha256",
            "execution_artifact_pool",
            "arm_results",
            "complete_metrics",
            "controls",
            "readiness",
            "deterministic_core",
            "observational_latency",
            "payload_sha256",
        },
        "payload",
    )
    if payload["schema_version"] != "1.1.0":
        _fail("unknown evidence payload schema")
    if payload["payload_sha256"] != _digest(
        {key: value for key, value in payload.items() if key != "payload_sha256"}
    ):
        _fail("inner payload digest mismatch")
    protocol = _verify_protocol(payload, root.resolve())
    packed_artifacts = _sequence(payload["execution_artifact_pool"], "execution artifact pool")
    if not 1 <= len(packed_artifacts) <= 4_096:
        _fail("execution artifact pool count exceeds the frozen bound")
    artifact_pool: dict[str, dict[str, Any]] = {}
    aggregate_artifact_bytes = 0
    for item in packed_artifacts:
        packed_artifact = _mapping(item, "packed execution artifact")
        artifact = _unpack_document(
            packed_artifact,
            expected_kind="execution_artifact",
            max_uncompressed_bytes=16_777_216,
        )
        evidence_id = artifact.get("evidence_sha256")
        if type(evidence_id) is not str or evidence_id in artifact_pool:
            _fail("execution artifact pool identifiers are invalid or duplicated")
        artifact_pool[evidence_id] = artifact
        aggregate_artifact_bytes += int(packed_artifact["uncompressed_bytes"])
    if aggregate_artifact_bytes > int(protocol["bounds"]["max_aggregate_execution_bytes"]):
        _fail("execution artifact pool exceeds the frozen aggregate bound")
    packed_results = _sequence(payload["arm_results"], "packed arm results")
    results: list[dict[str, Any]] = []
    retained_results: list[dict[str, Any]] = []
    used_artifact_ids: set[str] = set()
    for item in packed_results:
        retained = _unpack_document(item, expected_kind="arm_result")
        retained_results.append(retained)
        expanded, used = _expand_retained_result(retained, artifact_pool)
        results.append(expanded)
        used_artifact_ids.update(used)
    if used_artifact_ids != set(artifact_pool):
        _fail("execution artifact pool contains missing or cherry-picked support")
    complete = [
        _unpack_document(item, expected_kind="complete_metrics")
        for item in _sequence(payload["complete_metrics"], "packed complete metrics")
    ]
    if [item["arm"] for item in results] != list(_ARMS) or [
        item["arm"] for item in complete
    ] != list(_ARMS):
        _fail("artifact does not contain exact ordered four arms")
    verified_rows: list[list[dict[str, Any]]] = []
    verified_manifests: list[dict[str, dict[str, Any]]] = []
    for result in results:
        rows, _artifacts, manifests = _verify_arm_result(
            result, str(payload["catalog_sha256"]), protocol
        )
        verified_rows.append(rows)
        verified_manifests.append(manifests)
    reference_support = [row["support"] for row in verified_rows[0]]
    reference_features = [row["catalog_feature_values"] for row in verified_rows[0]]
    for rows in verified_rows[1:]:
        if [row["support"] for row in rows] != reference_support or [
            row["catalog_feature_values"] for row in rows
        ] != reference_features:
            _fail("arms use different ordered support or full-catalog features")
    if any(manifests != verified_manifests[0] for manifests in verified_manifests[1:]):
        _fail("arms retain different execution artifacts")
    for result, metrics, rows, manifests in zip(
        results, complete, verified_rows, verified_manifests, strict=True
    ):
        _verify_complete_metrics(metrics, result=result, rows=rows, manifests=manifests)
    controls = _unpack_document(payload["controls"], expected_kind="executed_controls")
    _verify_controls(
        controls,
        protocol=protocol,
        support_ids=[str(item["event_id"]) for item in reference_support],
        execution_manifests=verified_manifests[0],
        reference_rows=verified_rows[0],
    )
    readiness = _mapping(payload["readiness"], "readiness")
    _verify_readiness(readiness, complete_metrics=complete[-1], controls=controls)
    core_binding = _mapping(payload["deterministic_core"], "deterministic core binding")
    _exact_fields(
        core_binding,
        {"schema_version", "exclusion_schema", "core_sha256"},
        "deterministic core binding",
    )
    if core_binding["schema_version"] != _DETERMINISTIC_CORE_SCHEMA:
        _fail("unknown deterministic core schema")
    expected_exclusions = json.loads(_canonical_bytes(_DETERMINISTIC_CORE_EXCLUSION_SCHEMA))
    if core_binding["exclusion_schema"] != expected_exclusions:
        _fail("deterministic core exclusion schema differs")
    core_document = _independent_core_document(
        payload=payload,
        artifacts=list(artifact_pool.values()),
        retained_results=retained_results,
        complete=complete,
        controls=controls,
        readiness=readiness,
    )
    expected_core_sha256 = _digest(core_document)
    if core_binding["core_sha256"] != expected_core_sha256:
        _fail("deterministic core digest mismatch")
    observational = _unpack_document(
        payload["observational_latency"], expected_kind="observational_latency"
    )
    expected_observational = _independent_observational_document(
        core_sha256=expected_core_sha256,
        retained_results=retained_results,
        complete=complete,
        controls=controls,
        readiness=readiness,
    )
    if observational != expected_observational:
        _fail("observational latency evidence differs from retained samples")
    return {
        "verified": True,
        "status": readiness["status"],
        "safe_seed": 404,
        "arm_count": 4,
        "support_count": len(reference_support),
        "payload_sha256": payload["payload_sha256"],
        "envelope_sha256": envelope["envelope_sha256"],
        "deterministic_core_sha256": expected_core_sha256,
        "observational_latency_sha256": observational[
            "observational_latency_sha256"
        ],
        "observational_environment_sha256": observational["environment_sha256"],
        "evidence_protocol_sha256": protocol["evidence_protocol_sha256"],
        "base_protocol_sha256": protocol["base_protocol_sha256"],
        "arm_protocol_sha256": protocol["arm_protocol_sha256"],
        "implementation_sha256": protocol["implementation_sha256"],
        "catalog_sha256": payload["catalog_sha256"],
    }


def verify_evidence_bytes(serialized: bytes, *, root: Path) -> dict[str, object]:
    """Verify one artifact offline and return a concise machine-readable report."""
    try:
        return _verify_evidence_bytes(serialized, root=root)
    except IndependentVerificationError:
        raise
    except (
        KeyError,
        IndexError,
        OSError,
        StopIteration,
        TypeError,
        ValueError,
    ) as error:
        raise IndependentVerificationError(
            "serialized evidence is malformed or internally inconsistent"
        ) from error


def verify_locked_evidence_payload_bytes(
    serialized: bytes, *, root: Path
) -> dict[str, object]:
    """Independently replay one complete locked-development payload."""
    try:
        payload = _mapping(json.loads(serialized), "locked payload")
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise IndependentVerificationError("locked payload schema is invalid") from error
    if payload.get("schema_version") != (
        "apar-sentinel-v5-locked-development-payload/2"
    ):
        _fail("locked payload schema differs")
    _exact_fields(
        payload,
        {
            "schema_version",
            "run_binding",
            "attempt_receipt_sha256",
            "evidence_protocol",
            "catalog_sha256",
            "execution_artifact_pool",
            "arm_results",
            "complete_metrics",
            "controls",
            "readiness",
            "deterministic_core",
            "observational_latency",
            "payload_sha256",
        },
        "locked payload fields",
    )
    if serialized != _canonical_bytes(payload):
        _fail("locked payload is not canonical JSON")
    if payload["payload_sha256"] != _digest(
        {key: value for key, value in payload.items() if key != "payload_sha256"}
    ):
        _fail("locked payload digest mismatch")
    binding = _mapping(payload["run_binding"], "locked run binding")
    if (
        binding.get("mode") != "locked_development"
        or binding.get("profile") != "production"
        or binding.get("development_test_seed") != 2404
    ):
        _fail("locked run mode/profile/seed differs")
    return _verify_locked_payload_document(payload, root=root.resolve())


def _locked_legitimate_plan(count: int) -> tuple[int, int]:
    remaining = count - 24
    if remaining < 0:
        _fail("locked legitimate support cannot cover all base rails")
    full, final = divmod(remaining, 96)
    artifact_count = 3 + full + (1 if final else 0)
    estimate = 221_072 + full * 320_768
    if final:
        estimate += 32_768 + final * 3_000
    return artifact_count, estimate


def _independent_locked_support_plan(base: Mapping[str, Any]) -> dict[str, Any]:
    production = _mapping(base.get("production_profile"), "production profile")
    campaign_counts = _mapping(
        production.get("campaigns_per_family"), "production campaign counts"
    )
    expected_campaigns = {family: 100 for family in _FAMILIES}
    if campaign_counts != expected_campaigns or base.get(
        "production_dev_test_campaigns_per_family"
    ) != expected_campaigns:
        _fail("locked production campaign plan differs")
    if production.get("legitimate_decisions") != 50_000 or base.get(
        "production_dev_test_legitimate"
    ) != 50_000:
        _fail("locked production legitimate plan differs")
    event_rows = {
        "agentic_intent_abuse": 25,
        "app_scam_mule": 36,
        "card_testing_cnp": 26,
        "synthetic_merchant_refund": 46,
    }
    artifact_estimates = {
        "agentic_intent_abuse": 365_536,
        "app_scam_mule": 140_768,
        "card_testing_cnp": 110_768,
        "synthetic_merchant_refund": 170_768,
    }
    partitions: list[dict[str, Any]] = []
    for partition in ("train", "calibration", "threshold", "development_test"):
        legitimate = 50_000 if partition == "development_test" else 12_500
        legitimate_artifacts, legitimate_estimate = _locked_legitimate_plan(
            legitimate
        )
        fraud: list[list[Any]] = [
            [family, int(campaign_counts[family]) * event_rows[family]]
            for family in sorted(_FAMILIES)
        ]
        partitions.append(
            {
                "partition": partition,
                "legitimate_rows": legitimate,
                "fraud_rows_by_family": fraud,
                "total_rows": legitimate + sum(int(item[1]) for item in fraud),
                "execution_artifacts": legitimate_artifacts
                + sum(int(value) for value in campaign_counts.values()),
                "execution_payload_estimate_bytes": legitimate_estimate
                + sum(
                    int(campaign_counts[family]) * artifact_estimates[family]
                    for family in _FAMILIES
                ),
            }
        )
    values = {
        "mode": "locked_development",
        "profile": "production",
        "partitions": partitions,
        "retained_execution_artifacts": sum(
            int(item["execution_artifacts"]) for item in partitions
        ),
        "retained_execution_payload_estimate_bytes": sum(
            int(item["execution_payload_estimate_bytes"]) for item in partitions
        ),
    }
    values["support_plan_sha256"] = _digest(values)
    return values


def _independent_locked_core_document(
    *,
    payload: Mapping[str, Any],
    artifacts: Sequence[Mapping[str, Any]],
    retained_results: Sequence[Mapping[str, Any]],
    complete: Sequence[Mapping[str, Any]],
    controls: Mapping[str, Any],
    readiness: Mapping[str, Any],
) -> dict[str, Any]:
    proxy = dict(payload)
    proxy["safe_seed"] = 404
    document = _independent_core_document(
        payload=proxy,
        artifacts=artifacts,
        retained_results=retained_results,
        complete=complete,
        controls=controls,
        readiness=readiness,
    )
    document["schema_version"] = (
        "apar-sentinel-v5-locked-deterministic-core/2"
    )
    document["exclusion_schema"] = _LOCKED_DETERMINISTIC_CORE_EXCLUSION_SCHEMA
    document.pop("safe_seed")
    document["run_binding"] = payload["run_binding"]
    return document


def _git_value(root: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        _fail("locked source/preregistration Git binding cannot be resolved")
    return completed.stdout.strip()


def _locked_preregistration_source_paths(root: Path) -> tuple[str, ...]:
    arms = _mapping(
        json.loads((root / "config/defense/defense-v5-arms.json").read_bytes()),
        "locked arm protocol source",
    )
    evidence = _mapping(
        json.loads(
            (root / "config/defense/defense-v5-evidence.json").read_bytes()
        ),
        "locked evidence protocol source",
    )
    paths = {
        "config/defense/defense-v5-arms.json",
        "config/defense/defense-v5-development.json",
        "config/defense/defense-v5-evidence.json",
        "config/defense/feature-catalog-v5.json",
        "docs/superpowers/plans/2026-08-22-apar-sentinel-v5.md",
        "docs/superpowers/specs/2026-08-22-apar-sentinel-v5-design.md",
        *[str(item) for item in _sequence(arms["implementation_paths"], "arm paths")],
        *[
            str(item)
            for item in _sequence(evidence["implementation_paths"], "evidence paths")
        ],
    }
    return tuple(sorted(paths))


def _independent_locked_source_records(
    root: Path, source_commit: str
) -> list[list[str]]:
    records: list[list[str]] = []
    for relative in _locked_preregistration_source_paths(root):
        path = root / relative
        if not path.is_file():
            _fail(f"locked SOURCE file is missing: {relative}")
        tree_record = _git_value(root, "ls-tree", source_commit, "--", relative)
        if not tree_record or "\t" not in tree_record:
            _fail(f"locked SOURCE file is not tracked: {relative}")
        mode = tree_record.split(maxsplit=1)[0]
        if mode not in {"100644", "100755"}:
            _fail(f"locked SOURCE file mode differs: {relative}")
        records.append([relative, mode, hashlib.sha256(path.read_bytes()).hexdigest()])
    return records


def _verify_locked_preregistration_document(
    *,
    preregistration: Mapping[str, Any],
    protocol: Mapping[str, Any],
    payload: Mapping[str, Any],
    expected_plan: Mapping[str, Any],
    root: Path,
    source_commit: str,
) -> None:
    _exact_fields(
        preregistration,
        {
            "schema_version",
            "source_commit",
            "source_tree_oid",
            "preregistration_path",
            "source_bindings",
            "source_files",
            "safe_validation",
            "closed_run_mode",
            "production_support_plan",
            "frozen_definitions",
            "output_contract",
            "exact_command",
            "manifest_sha256",
        },
        "locked preregistration fields",
    )
    if preregistration["schema_version"] != (
        "apar-sentinel-v5-locked-preregistration/2"
    ):
        _fail("locked preregistration schema differs")
    if preregistration["manifest_sha256"] != _digest(
        {
            key: value
            for key, value in preregistration.items()
            if key != "manifest_sha256"
        }
    ):
        _fail("locked preregistration digest differs")
    if preregistration["source_files"] != _independent_locked_source_records(
        root, source_commit
    ):
        _fail("locked SOURCE file set/modes/content hashes differ")
    source_bindings = {
        "base_protocol_sha256": protocol["base_protocol_sha256"],
        "arm_protocol_sha256": protocol["arm_protocol_sha256"],
        "evidence_protocol_sha256": protocol["evidence_protocol_sha256"],
        "implementation_sha256": protocol["implementation_sha256"],
        "catalog_sha256": payload["catalog_sha256"],
        "verifier_sha256": hashlib.sha256(
            (root / "src/apar/v5_independent_verifier.py").read_bytes()
        ).hexdigest(),
    }
    if preregistration["source_bindings"] != source_bindings:
        _fail("locked preregistration source bindings differ")
    if preregistration["safe_validation"] != {
        "historical_deterministic_core_sha256": (
            "784a762fd90a65219a233e87df35290ac87c8fe8e4b9024de46564568f633719"
        ),
        "approved_deterministic_core_sha256": _mapping(
            preregistration["safe_validation"], "locked safe validation"
        ).get("approved_deterministic_core_sha256"),
        "approved_observational_environment_sha256": _mapping(
            preregistration["safe_validation"], "locked safe validation"
        ).get("approved_observational_environment_sha256"),
    }:
        _fail("locked preregistration safe validation fields differ")
    safe = _mapping(preregistration["safe_validation"], "locked safe validation")
    historical_safe_freeze = _mapping(
        json.loads(
            (
                root / "config/defense/defense-v5-safe-core-freeze.json"
            ).read_bytes()
        ),
        "historical safe-core freeze",
    )
    if historical_safe_freeze.get("approved_deterministic_core_sha256") != (
        "784a762fd90a65219a233e87df35290ac87c8fe8e4b9024de46564568f633719"
    ):
        _fail("historical approved safe core differs")
    for field in (
        "approved_deterministic_core_sha256",
        "approved_observational_environment_sha256",
    ):
        value = safe[field]
        if type(value) is not str or len(value) != 64 or any(
            character not in "0123456789abcdef" for character in value
        ):
            _fail("locked approved safe evidence digest differs")
    if preregistration["closed_run_mode"] != {
        "mode": "locked_development",
        "profile": "production",
        "development_test_seed": 2404,
        "repeatable": False,
        "authorization_required": True,
    }:
        _fail("locked preregistration closed run mode differs")
    if preregistration["production_support_plan"] != expected_plan:
        _fail("locked preregistration support plan differs")
    if preregistration["frozen_definitions"] != {
        "controls": protocol["controls"],
        "calibration": protocol["calibration"],
        "bootstrap": protocol["bootstrap"],
        "economics": protocol["economics"],
        "metric_definitions": protocol["metric_definitions"],
        "gates": protocol["gates"],
    }:
        _fail("locked preregistration frozen definitions differ")
    storage = _mapping(protocol["locked_artifact_storage"], "locked storage")
    if preregistration["output_contract"] != {
        **storage,
        "attempt_receipt_must_be_absent": True,
        "candidate_must_be_absent": True,
        "historical_result_path": protocol["existing_development_result_path"],
        "historical_result_sha256": protocol[
            "existing_development_result_sha256"
        ],
        "one_time_no_resume_or_retry": True,
        "legacy_summary_is_not_evidence": True,
    }:
        _fail("locked preregistration output/storage contract differs")
    if preregistration["exact_command"] != (
        ".venv/bin/python scripts/run_defense_v5_locked_development.py --root . "
        "--safe-evidence /private/tmp/apar-v5-approved-safe-evidence.json "
        "--approved-commit HEAD --authorize-exactly-once"
    ):
        _fail("locked preregistration exact command differs")


def _verify_locked_run_binding(
    *,
    binding: Mapping[str, Any],
    protocol: Mapping[str, Any],
    payload: Mapping[str, Any],
    root: Path,
) -> dict[str, Any]:
    _exact_fields(
        binding,
        {
            "schema_version",
            "mode",
            "profile",
            "development_test_seed",
            "source_commit",
            "source_tree_oid",
            "preregistration_commit",
            "preregistration_path",
            "preregistration_sha256",
            "base_protocol_sha256",
            "arm_protocol_sha256",
            "evidence_protocol_sha256",
            "implementation_sha256",
            "catalog_sha256",
            "support_plan",
            "candidate_manifest_path",
            "storage_schema_version",
            "payload_schema_version",
            "run_binding_sha256",
        },
        "locked run binding",
    )
    if (
        binding["schema_version"] != "apar-sentinel-v5-locked-run-binding/1"
        or binding["mode"] != "locked_development"
        or binding["profile"] != "production"
        or binding["development_test_seed"] != 2404
    ):
        _fail("locked run mode/profile/seed differs")
    if binding["run_binding_sha256"] != _digest(
        {key: value for key, value in binding.items() if key != "run_binding_sha256"}
    ):
        _fail("locked run-binding digest differs")
    for binding_field, protocol_field in (
        ("base_protocol_sha256", "base_protocol_sha256"),
        ("arm_protocol_sha256", "arm_protocol_sha256"),
        ("evidence_protocol_sha256", "evidence_protocol_sha256"),
        ("implementation_sha256", "implementation_sha256"),
    ):
        if binding[binding_field] != protocol[protocol_field]:
            _fail(f"locked {binding_field} differs")
    if binding["catalog_sha256"] != payload["catalog_sha256"]:
        _fail("locked catalog binding differs")
    storage = _mapping(protocol["locked_artifact_storage"], "locked storage")
    if (
        binding["candidate_manifest_path"] != storage["candidate_manifest_path"]
        or binding["storage_schema_version"] != storage["schema_version"]
        or binding["payload_schema_version"]
        != "apar-sentinel-v5-locked-development-payload/2"
    ):
        _fail("locked output/storage schema binding differs")
    base_path = root / str(protocol["base_protocol_path"])
    base = _mapping(json.loads(base_path.read_bytes()), "locked base protocol")
    expected_plan = _independent_locked_support_plan(base)
    if binding["support_plan"] != expected_plan:
        _fail("locked production support plan differs")
    prereg_path = str(binding["preregistration_path"])
    if prereg_path != (
        "config/defense/defense-v5-locked-development-preregistration.json"
    ):
        _fail("locked preregistration path differs")
    preregistration_path = root / prereg_path
    if not preregistration_path.is_file():
        _fail("locked preregistration is missing")
    preregistration_bytes = preregistration_path.read_bytes()
    if hashlib.sha256(preregistration_bytes).hexdigest() != binding[
        "preregistration_sha256"
    ]:
        _fail("locked preregistration digest differs")
    preregistration = _mapping(
        json.loads(preregistration_bytes), "locked preregistration"
    )
    if (
        preregistration.get("source_commit") != binding["source_commit"]
        or preregistration.get("source_tree_oid") != binding["source_tree_oid"]
        or preregistration.get("production_support_plan") != expected_plan
    ):
        _fail("locked preregistration source/support binding differs")
    _verify_locked_preregistration_document(
        preregistration=preregistration,
        protocol=protocol,
        payload=payload,
        expected_plan=expected_plan,
        root=root,
        source_commit=str(binding["source_commit"]),
    )
    source_commit = str(binding["source_commit"])
    preregistration_commit = str(binding["preregistration_commit"])
    if _git_value(root, "rev-parse", f"{source_commit}^{{tree}}") != binding[
        "source_tree_oid"
    ]:
        _fail("locked SOURCE tree differs")
    parents = _git_value(
        root, "rev-list", "--parents", "-n", "1", preregistration_commit
    ).split()
    if len(parents) != 2 or parents[1] != source_commit:
        _fail("locked PREREGISTRATION is not the single SOURCE child")
    changed = tuple(
        line
        for line in _git_value(
            root,
            "diff-tree",
            "--no-commit-id",
            "--name-only",
            "-r",
            preregistration_commit,
        ).splitlines()
        if line
    )
    if changed != (prereg_path,):
        _fail("locked PREREGISTRATION commit is not manifest-only")
    return expected_plan


def _support_distribution(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[int, dict[str, int]]:
    legitimate = 0
    fraud: dict[str, int] = {}
    for row in rows:
        support = _mapping(row["support"], "locked support row")
        if support["label"] == 0:
            legitimate += 1
        elif support["label"] == 1:
            family = str(support["family"])
            fraud[family] = fraud.get(family, 0) + 1
        else:
            _fail("locked support label differs")
    return legitimate, dict(sorted(fraud.items()))


def _verify_locked_support(
    *,
    expected_plan: Mapping[str, Any],
    result: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    artifact_count: int,
) -> None:
    plans = {
        str(item["partition"]): _mapping(item, "locked partition plan")
        for item in _sequence(expected_plan["partitions"], "locked partitions")
    }
    development = plans["development_test"]
    legitimate, fraud = _support_distribution(rows)
    if (
        legitimate != development["legitimate_rows"]
        or fraud != dict(development["fraud_rows_by_family"])
        or len(rows) != development["total_rows"]
    ):
        _fail("locked development-test production support differs")
    arm_spec = _mapping(result["arm_spec"], "locked arm specification")
    for partition in _sequence(
        arm_spec["training_partitions"], "locked training partitions"
    ):
        partition = _mapping(partition, "locked training partition")
        expected = plans[str(partition["partition"])]
        support_rows = [
            {"support": item}
            for item in _sequence(
                partition["support_records"], "locked training support"
            )
        ]
        legitimate, fraud = _support_distribution(support_rows)
        if (
            legitimate != expected["legitimate_rows"]
            or fraud != dict(expected["fraud_rows_by_family"])
            or len(support_rows) != expected["total_rows"]
        ):
            _fail(f"locked {partition['partition']} production support differs")
    if artifact_count != expected_plan["retained_execution_artifacts"]:
        _fail("locked retained execution artifact count differs")


def _verify_locked_protocol(
    payload: Mapping[str, Any], root: Path
) -> dict[str, Any]:
    proxy = dict(payload)
    proxy["safe_seed"] = 404
    protocol = _verify_protocol(proxy, root)
    base = _mapping(
        json.loads((root / str(protocol["base_protocol_path"])).read_bytes()),
        "locked base protocol",
    )
    if _mapping(base["seeds"], "locked seeds")["development_test"] != 2404:
        _fail("locked base protocol seed differs from 2404")
    protocol["_base_protocol_sha256"] = _digest(base)
    return protocol


def _read_locked_attempt_receipt_document(path: Path) -> dict[str, Any]:
    """Read the attempt receipt independently of production storage code."""
    try:
        metadata = path.lstat()
    except OSError as error:
        raise IndependentVerificationError(
            "locked attempt receipt is missing"
        ) from error
    if (
        not stat.S_ISREG(metadata.st_mode)
        or metadata.st_nlink != 1
        or metadata.st_size > 65_536
    ):
        _fail("locked attempt receipt topology/size differs")
    raw = path.read_bytes()
    try:
        receipt = _mapping(json.loads(raw), "locked attempt receipt")
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise IndependentVerificationError(
            "locked attempt receipt is not JSON"
        ) from error
    if raw != _canonical_bytes(receipt):
        _fail("locked attempt receipt is not canonical JSON")
    _exact_fields(
        receipt,
        {
            "schema_version",
            "run_binding_sha256",
            "preregistration_commit",
            "preregistration_sha256",
            "source_commit",
            "source_tree_oid",
            "approved_safe_deterministic_core_sha256",
            "approved_safe_observational_environment_sha256",
            "authorization_sha256",
            "exact_command",
            "started_at_utc",
            "receipt_sha256",
        },
        "locked attempt receipt",
    )
    if receipt["schema_version"] != (
        "apar-sentinel-v5-locked-attempt-receipt/1"
    ):
        _fail("locked attempt receipt schema differs")
    if receipt["receipt_sha256"] != _digest(
        {
            key: value
            for key, value in receipt.items()
            if key != "receipt_sha256"
        }
    ):
        _fail("locked attempt receipt digest differs")
    timestamp = receipt["started_at_utc"]
    if type(timestamp) is not str or not timestamp.endswith("Z"):
        _fail("locked attempt receipt timestamp differs")
    try:
        started = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    except ValueError as error:
        raise IndependentVerificationError(
            "locked attempt receipt timestamp is invalid"
        ) from error
    if started.tzinfo != UTC:
        _fail("locked attempt receipt timestamp is not UTC")
    return receipt


def _verify_locked_attempt_receipt(
    *,
    payload: Mapping[str, Any],
    protocol: Mapping[str, Any],
    binding: Mapping[str, Any],
    root: Path,
) -> dict[str, Any]:
    storage = _mapping(protocol["locked_artifact_storage"], "locked storage")
    attempt_path = str(storage["attempt_receipt_path"])
    if attempt_path != (
        "docs/experiments/defense-v5-locked-development-attempt.json"
    ):
        _fail("locked attempt receipt path differs")
    receipt = _read_locked_attempt_receipt_document(root / attempt_path)
    preregistration = _mapping(
        json.loads(
            (root / str(binding["preregistration_path"])).read_bytes()
        ),
        "locked preregistration",
    )
    safe = _mapping(preregistration["safe_validation"], "locked safe validation")
    expected_authorization_sha256 = _digest(
        {
            "authorization": "execute-exactly-once-locked-development",
            "preregistration_commit": binding["preregistration_commit"],
            "run_binding_sha256": binding["run_binding_sha256"],
            "exact_command": preregistration["exact_command"],
        }
    )
    expected = {
        "run_binding_sha256": binding["run_binding_sha256"],
        "preregistration_commit": binding["preregistration_commit"],
        "preregistration_sha256": binding["preregistration_sha256"],
        "source_commit": binding["source_commit"],
        "source_tree_oid": binding["source_tree_oid"],
        "approved_safe_deterministic_core_sha256": safe[
            "approved_deterministic_core_sha256"
        ],
        "approved_safe_observational_environment_sha256": safe[
            "approved_observational_environment_sha256"
        ],
        "authorization_sha256": expected_authorization_sha256,
        "exact_command": preregistration["exact_command"],
    }
    if any(receipt[field] != value for field, value in expected.items()):
        _fail("locked attempt receipt binding differs")
    if payload["attempt_receipt_sha256"] != receipt["receipt_sha256"]:
        _fail("locked payload attempt receipt binding differs")
    return receipt


def _verify_locked_payload_document(
    payload: Mapping[str, Any], *, root: Path
) -> dict[str, object]:
    protocol = _verify_locked_protocol(payload, root)
    binding = _mapping(payload["run_binding"], "locked run binding")
    expected_plan = _verify_locked_run_binding(
        binding=binding,
        protocol=protocol,
        payload=payload,
        root=root,
    )
    attempt_receipt = _verify_locked_attempt_receipt(
        payload=payload,
        protocol=protocol,
        binding=binding,
        root=root,
    )
    packed_artifacts = _sequence(
        payload["execution_artifact_pool"], "locked execution artifact pool"
    )
    if not 1 <= len(packed_artifacts) <= 4_096:
        _fail("locked execution artifact pool count exceeds bound")
    artifact_pool: dict[str, dict[str, Any]] = {}
    aggregate_artifact_bytes = 0
    for item in packed_artifacts:
        packed = _mapping(item, "locked packed execution artifact")
        artifact = _unpack_document(
            packed,
            expected_kind="execution_artifact",
            max_uncompressed_bytes=16_777_216,
        )
        evidence_id = artifact.get("evidence_sha256")
        if type(evidence_id) is not str or evidence_id in artifact_pool:
            _fail("locked execution artifact identifiers differ")
        artifact_pool[evidence_id] = artifact
        aggregate_artifact_bytes += int(packed["uncompressed_bytes"])
    if aggregate_artifact_bytes > 1_073_741_824:
        _fail("locked execution artifact pool exceeds production bound")
    results: list[dict[str, Any]] = []
    retained_results: list[dict[str, Any]] = []
    used_artifact_ids: set[str] = set()
    for item in _sequence(payload["arm_results"], "locked arm results"):
        retained = _unpack_document(item, expected_kind="arm_result")
        retained_results.append(retained)
        expanded, used = _expand_retained_result(retained, artifact_pool)
        results.append(expanded)
        used_artifact_ids.update(used)
    if used_artifact_ids != set(artifact_pool):
        _fail("locked artifact pool contains cherry-picked support")
    complete = [
        _unpack_document(item, expected_kind="complete_metrics")
        for item in _sequence(payload["complete_metrics"], "locked complete metrics")
    ]
    if [item["arm"] for item in results] != list(_ARMS) or [
        item["arm"] for item in complete
    ] != list(_ARMS):
        _fail("locked payload does not contain exact ordered four arms")
    verified_rows: list[list[dict[str, Any]]] = []
    verified_manifests: list[dict[str, dict[str, Any]]] = []
    for result in results:
        rows, _artifacts, manifests = _verify_arm_result(
            result, str(payload["catalog_sha256"]), protocol
        )
        verified_rows.append(rows)
        verified_manifests.append(manifests)
    reference_support = [row["support"] for row in verified_rows[0]]
    reference_features = [row["catalog_feature_values"] for row in verified_rows[0]]
    for rows in verified_rows[1:]:
        if [row["support"] for row in rows] != reference_support or [
            row["catalog_feature_values"] for row in rows
        ] != reference_features:
            _fail("locked arms use different ordered support/features")
    if any(
        manifests != verified_manifests[0]
        for manifests in verified_manifests[1:]
    ):
        _fail("locked arms retain different execution artifacts")
    _verify_locked_support(
        expected_plan=expected_plan,
        result=results[0],
        rows=verified_rows[0],
        artifact_count=len(artifact_pool),
    )
    for result, metrics, rows, manifests in zip(
        results, complete, verified_rows, verified_manifests, strict=True
    ):
        _verify_complete_metrics(metrics, result=result, rows=rows, manifests=manifests)
    controls = _unpack_document(
        payload["controls"], expected_kind="executed_controls"
    )
    _verify_controls(
        controls,
        protocol=protocol,
        support_ids=[str(item["event_id"]) for item in reference_support],
        execution_manifests=verified_manifests[0],
        reference_rows=verified_rows[0],
    )
    readiness = _mapping(payload["readiness"], "locked readiness")
    _verify_readiness(readiness, complete_metrics=complete[-1], controls=controls)
    core_binding = _mapping(
        payload["deterministic_core"], "locked deterministic core binding"
    )
    _exact_fields(
        core_binding,
        {"schema_version", "exclusion_schema", "core_sha256"},
        "locked deterministic core binding",
    )
    if core_binding["schema_version"] != (
        "apar-sentinel-v5-locked-deterministic-core/2"
    ):
        _fail("locked deterministic core schema differs")
    if core_binding["exclusion_schema"] != json.loads(
        _canonical_bytes(_LOCKED_DETERMINISTIC_CORE_EXCLUSION_SCHEMA)
    ):
        _fail("locked deterministic core exclusion schema differs")
    core_document = _independent_locked_core_document(
        payload=payload,
        artifacts=list(artifact_pool.values()),
        retained_results=retained_results,
        complete=complete,
        controls=controls,
        readiness=readiness,
    )
    core_sha256 = _digest(core_document)
    if core_binding["core_sha256"] != core_sha256:
        _fail("locked deterministic core digest differs")
    observational = _unpack_document(
        payload["observational_latency"], expected_kind="observational_latency"
    )
    expected_observational = _independent_observational_document(
        core_sha256=core_sha256,
        retained_results=retained_results,
        complete=complete,
        controls=controls,
        readiness=readiness,
    )
    if observational != expected_observational:
        _fail("locked observational latency evidence differs")
    return {
        "verified": True,
        "run_mode": "locked_development",
        "profile": "production",
        "development_test_seed": 2404,
        "status": readiness["status"],
        "arm_count": 4,
        "support_count": len(reference_support),
        "payload_sha256": payload["payload_sha256"],
        "deterministic_core_sha256": core_sha256,
        "observational_latency_sha256": observational[
            "observational_latency_sha256"
        ],
        "observational_environment_sha256": observational["environment_sha256"],
        "run_binding_sha256": binding["run_binding_sha256"],
        "attempt_receipt_sha256": attempt_receipt["receipt_sha256"],
        "support_plan_sha256": expected_plan["support_plan_sha256"],
        "evidence_protocol_sha256": protocol["evidence_protocol_sha256"],
        "base_protocol_sha256": protocol["base_protocol_sha256"],
        "arm_protocol_sha256": protocol["arm_protocol_sha256"],
        "implementation_sha256": protocol["implementation_sha256"],
        "catalog_sha256": payload["catalog_sha256"],
    }


def read_locked_evidence_storage_bytes(
    *,
    target_manifest: Path,
    attempt_receipt_path: Path,
    chunk_size_bytes: int,
    maximum_envelope_bytes: int,
    maximum_chunk_count: int,
    normal_git_blob_limit_bytes: int,
) -> bytes:
    """Independently authenticate and reconstruct a manifest-last artifact."""
    target_manifest = target_manifest.absolute()
    try:
        manifest_metadata = target_manifest.lstat()
    except OSError as error:
        raise IndependentVerificationError("locked storage manifest is missing") from error
    if (
        not stat.S_ISREG(manifest_metadata.st_mode)
        or manifest_metadata.st_nlink != 1
        or manifest_metadata.st_size > 1_048_576
    ):
        _fail("locked storage manifest topology/size differs")
    raw_manifest = target_manifest.read_bytes()
    try:
        manifest = _mapping(json.loads(raw_manifest), "locked storage manifest")
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise IndependentVerificationError("locked storage manifest is not JSON") from error
    if raw_manifest != _canonical_bytes(manifest):
        _fail("locked storage manifest is not canonical JSON")
    _exact_fields(
        manifest,
        {
            "schema_version",
            "content_encoding",
            "publication",
            "chunks_directory",
            "chunk_size_bytes",
            "maximum_envelope_bytes",
            "maximum_chunk_count",
            "normal_git_blob_limit_bytes",
            "payload_sha256",
            "payload_bytes",
            "run_binding_sha256",
            "attempt_receipt_path",
            "attempt_receipt_sha256",
            "completion_receipt",
            "chunks",
            "manifest_sha256",
        },
        "locked storage manifest",
    )
    if (
        manifest["schema_version"] != "apar-sentinel-v5-chunked-evidence/2"
        or manifest["content_encoding"] != "opaque-locked-evidence-bytes"
        or manifest["publication"]
        != "content_chunks_then_atomic_exclusive_manifest"
    ):
        _fail("locked storage schema/publication differs")
    if manifest["manifest_sha256"] != _digest(
        {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    ):
        _fail("locked storage manifest digest differs")
    if (
        manifest["chunk_size_bytes"] != chunk_size_bytes
        or manifest["maximum_envelope_bytes"] != maximum_envelope_bytes
        or manifest["maximum_chunk_count"] != maximum_chunk_count
        or manifest["normal_git_blob_limit_bytes"] != normal_git_blob_limit_bytes
        or not 0 < chunk_size_bytes < normal_git_blob_limit_bytes
    ):
        _fail("locked storage limits differ")
    chunks = _sequence(manifest["chunks"], "locked storage chunks")
    if not 0 < len(chunks) <= maximum_chunk_count:
        _fail("locked storage chunk count differs")
    expected_names = [f"part-{index:04d}.bin" for index in range(len(chunks))]
    for index, raw_chunk in enumerate(chunks):
        chunk = _mapping(raw_chunk, "locked storage chunk")
        _exact_fields(chunk, {"index", "filename", "bytes", "sha256"}, "chunk")
        if chunk["index"] != index or chunk["filename"] != expected_names[index]:
            _fail("locked storage chunk order differs")
        if (
            type(chunk["bytes"]) is not int
            or not 0 < chunk["bytes"] <= chunk_size_bytes
            or (index < len(chunks) - 1 and chunk["bytes"] != chunk_size_bytes)
        ):
            _fail("locked storage chunk size differs")
    chunks_name = manifest["chunks_directory"]
    if chunks_name != f"{target_manifest.name}.chunks":
        _fail("locked storage chunk directory name differs")
    chunks_directory = target_manifest.parent / str(chunks_name)
    try:
        directory_metadata = chunks_directory.lstat()
    except OSError as error:
        raise IndependentVerificationError("locked storage chunk directory is missing") from error
    if not stat.S_ISDIR(directory_metadata.st_mode):
        _fail("locked storage chunk directory topology differs")
    if {path.name for path in chunks_directory.iterdir()} != set(expected_names):
        _fail("locked storage chunk set differs")
    content = bytearray()
    for raw_chunk in chunks:
        chunk = _mapping(raw_chunk, "locked storage chunk")
        path = chunks_directory / str(chunk["filename"])
        try:
            metadata = path.lstat()
        except OSError as error:
            raise IndependentVerificationError("locked storage chunk is missing") from error
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_nlink != 1
            or metadata.st_size != chunk["bytes"]
        ):
            _fail("locked storage chunk topology/size differs")
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != chunk["sha256"]:
            _fail("locked storage chunk digest differs")
        content.extend(raw)
        if len(content) > maximum_envelope_bytes:
            _fail("locked storage payload exceeds the maximum envelope")
    payload = bytes(content)
    if (
        manifest["payload_bytes"] != len(payload)
        or manifest["payload_sha256"] != hashlib.sha256(payload).hexdigest()
    ):
        _fail("locked storage payload digest/size differs")
    declared_attempt_path = Path(str(manifest["attempt_receipt_path"]))
    if (
        declared_attempt_path.is_absolute()
        or ".." in declared_attempt_path.parts
        or tuple(attempt_receipt_path.absolute().parts[-len(declared_attempt_path.parts) :])
        != declared_attempt_path.parts
    ):
        _fail("locked attempt receipt path differs")
    attempt = _read_locked_attempt_receipt_document(
        attempt_receipt_path.absolute()
    )
    completion = _mapping(
        manifest["completion_receipt"], "locked completion receipt"
    )
    _exact_fields(
        completion,
        {
            "schema_version",
            "attempt_receipt_sha256",
            "completed_at_utc",
            "elapsed_ms",
            "payload_sha256",
            "payload_bytes",
            "receipt_sha256",
        },
        "locked completion receipt",
    )
    if completion.get("schema_version") != (
        "apar-sentinel-v5-locked-completion-receipt/2"
    ):
        _fail("locked completion receipt schema differs")
    if completion.get("receipt_sha256") != _digest(
        {key: value for key, value in completion.items() if key != "receipt_sha256"}
    ):
        _fail("locked completion receipt digest differs")
    if (
        attempt.get("run_binding_sha256") != manifest["run_binding_sha256"]
        or manifest.get("attempt_receipt_sha256")
        != attempt.get("receipt_sha256")
        or completion.get("attempt_receipt_sha256")
        != attempt.get("receipt_sha256")
        or completion.get("payload_sha256") != manifest["payload_sha256"]
        or completion.get("payload_bytes") != manifest["payload_bytes"]
    ):
        _fail("locked storage receipt lineage differs")
    return payload


def verify_locked_judge_summary(
    *,
    summary_path: Path,
    target_manifest: Path,
    attempt_receipt_path: Path,
    verification: Mapping[str, object],
    candidate_manifest_path: str,
    declared_attempt_receipt_path: str,
) -> dict[str, Any]:
    """Independently bind the compact summary to verified durable evidence."""
    try:
        metadata = summary_path.lstat()
    except OSError as error:
        raise IndependentVerificationError("locked judge summary is missing") from error
    if (
        not stat.S_ISREG(metadata.st_mode)
        or metadata.st_nlink != 1
        or metadata.st_size > 1_048_576
    ):
        _fail("locked judge summary topology/size differs")
    raw = summary_path.read_bytes()
    try:
        summary = _mapping(json.loads(raw), "locked judge summary")
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise IndependentVerificationError(
            "locked judge summary is not JSON"
        ) from error
    if raw != _canonical_bytes(summary):
        _fail("locked judge summary is not canonical JSON")
    _exact_fields(
        summary,
        {
            "schema_version",
            "candidate_manifest_path",
            "manifest_sha256",
            "payload_sha256",
            "run_binding_sha256",
            "attempt_receipt_path",
            "attempt_receipt_sha256",
            "verification",
            "summary_sha256",
        },
        "locked judge summary",
    )
    if summary["schema_version"] != (
        "apar-sentinel-v5-locked-judge-summary/2"
    ):
        _fail("locked judge summary schema differs")
    if summary["summary_sha256"] != _digest(
        {
            key: value
            for key, value in summary.items()
            if key != "summary_sha256"
        }
    ):
        _fail("locked judge summary digest differs")
    manifest = _mapping(
        json.loads(target_manifest.read_bytes()), "locked storage manifest"
    )
    attempt = _read_locked_attempt_receipt_document(attempt_receipt_path)
    expected = {
        "candidate_manifest_path": candidate_manifest_path,
        "manifest_sha256": manifest["manifest_sha256"],
        "payload_sha256": manifest["payload_sha256"],
        "run_binding_sha256": manifest["run_binding_sha256"],
        "attempt_receipt_path": declared_attempt_receipt_path,
        "attempt_receipt_sha256": attempt["receipt_sha256"],
        "verification": dict(verification),
    }
    if any(summary[field] != value for field, value in expected.items()):
        _fail("locked judge summary evidence binding differs")
    return summary


__all__ = [
    "IndependentVerificationError",
    "read_locked_evidence_storage_bytes",
    "verify_locked_judge_summary",
    "verify_evidence_bytes",
    "verify_locked_evidence_payload_bytes",
]
