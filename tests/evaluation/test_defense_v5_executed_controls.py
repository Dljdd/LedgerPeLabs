"""Executed Sentinel v5 control evidence."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from apar.evaluation.v5_controls import (
    V5ControlMeasurement,
    V5ExecutedControlResult,
    V5ExecutedControlSuite,
    execute_v5_controls,
)
from apar.evaluation.v5_evaluation import V5Arm, load_v5_arm_configuration
from apar.evaluation.v5_evidence_protocol import (
    V5MetricApplicability,
    load_v5_evidence_protocol,
)
from apar.evaluation.v5_population import build_v5_corpus
from apar.evaluation.v5_protocol import V5Profile
from apar.features.sentinel import SentinelFeatureCatalog
from tests.evaluation.v5_safe_protocol import load_safe_v5_test_protocol

ROOT = Path(__file__).resolve().parents[2]


def _digest(value: object) -> str:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    if isinstance(value, dict):
        value = {key: _jsonable(item) for key, item in value.items()}
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _jsonable(value: object) -> object:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    return value


def _control(name: str, *, qualifies: bool = True) -> V5ExecutedControlResult:
    support_ids = ("event:1", "event:2")
    artifact_ids = ("artifact:1",)
    spec_json = json.dumps({"name": name, "version": "1"}, sort_keys=True)
    row_evidence_json = json.dumps(
        [{"event_id": event_id, "before": 2.0, "after": 2.0} for event_id in support_ids],
        sort_keys=True,
    )
    measurement = V5ControlMeasurement(
        name="invariant_rows",
        applicability=V5MetricApplicability.DEFINED,
        before=2.0,
        after=2.0,
        delta=0.0,
        numerator=2.0,
        denominator=2.0,
        support_sha256=_digest(support_ids),
    )
    document = {
        "name": name,
        "executed": True,
        "qualifies_for_readiness": qualifies,
        "spec_json": spec_json,
        "spec_sha256": _digest(json.loads(spec_json)),
        "input_support_ids": support_ids,
        "input_support_sha256": _digest(support_ids),
        "input_artifact_ids": artifact_ids,
        "input_artifact_sha256": _digest(artifact_ids),
        "permutation_seed": 1707 if name == "label_shuffle" else None,
        "executed_arm_spec_sha256": (("full_sentinel", "4" * 64),),
        "measurements": (measurement,),
        "criterion": "exact equality",
        "passed": True,
        "row_evidence_json": row_evidence_json,
        "row_evidence_sha256": _digest(json.loads(row_evidence_json)),
        "implementation_sha256": "6" * 64,
    }
    document["control_sha256"] = _digest(document)
    return V5ExecutedControlResult.model_validate(document)


def test_executed_control_contract_rejects_descriptive_or_unbound_results() -> None:
    """A prose-only control cannot satisfy the executed evidence schema."""
    valid = _control("identity_rename")
    assert valid.executed is True
    assert valid.measurements[0].delta == 0.0

    with pytest.raises(ValueError, match="executed"):
        V5ExecutedControlResult.model_validate(
            valid.model_copy(update={"executed": False}).model_dump(mode="json")
        )
    with pytest.raises(ValueError, match="digest|sha256"):
        V5ExecutedControlResult.model_validate(
            valid.model_copy(update={"control_sha256": "0" * 64}).model_dump(mode="json")
        )


def test_control_suite_requires_exact_order_and_diagnostic_nonqualification() -> None:
    """Missing controls or a readiness-qualifying fraud-only diagnostic fail closed."""
    controls = (
        _control("label_shuffle"),
        _control("identity_rename"),
        _control("future_causality"),
        _control("equal_time_isolation"),
        _control("benign_only"),
        _control("fraud_only_diagnostic", qualifies=False),
        _control("feature_leakage"),
    )
    document = {
        "controls": controls,
        "evidence_protocol_sha256": "7" * 64,
        "support_sha256": "8" * 64,
        "implementation_sha256": "6" * 64,
    }
    document["suite_sha256"] = _digest(
        {
            key: [item.model_dump(mode="json") for item in value]
            if key == "controls"
            else value
            for key, value in document.items()
        }
    )
    suite = V5ExecutedControlSuite.model_validate(document)
    assert tuple(control.name for control in suite.controls) == (
        "label_shuffle",
        "identity_rename",
        "future_causality",
        "equal_time_isolation",
        "benign_only",
        "fraud_only_diagnostic",
        "feature_leakage",
    )

    with pytest.raises(ValueError, match="exact ordered controls"):
        suite.model_copy(update={"controls": suite.controls[:-1]}).validate_suite()
    fraud = suite.controls[5].model_copy(update={"qualifies_for_readiness": True})
    with pytest.raises(ValueError, match="fraud-only.*readiness"):
        suite.model_copy(
            update={"controls": (*suite.controls[:5], fraud, *suite.controls[6:])}
        ).validate_suite()


def test_safe_corpus_executes_all_controls_with_measured_evidence() -> None:
    """Replacing the executor with descriptive strings must break this integration."""
    protocol = load_safe_v5_test_protocol(ROOT)
    evidence_protocol = load_v5_evidence_protocol(
        ROOT / "config/defense/defense-v5-evidence.json",
        root=ROOT,
    )
    catalog = SentinelFeatureCatalog.default()
    configuration = load_v5_arm_configuration(
        ROOT / "config/defense/defense-v5-arms.json",
        catalog=catalog,
        protocol=protocol,
    )
    corpus = build_v5_corpus(protocol, profile=V5Profile.SMOKE)

    suite = execute_v5_controls(
        protocol=protocol,
        evidence_protocol=evidence_protocol,
        corpus=corpus,
        catalog=catalog,
        configuration=configuration,
    )

    assert all(control.executed for control in suite.controls)
    assert all(control.measurements for control in suite.controls)
    assert all(json.loads(control.row_evidence_json) for control in suite.controls)
    by_name = {control.name: control for control in suite.controls}
    label = by_name["label_shuffle"]
    assert label.permutation_seed == 1707
    assert {name for name, _digest_value in label.executed_arm_spec_sha256} == {
        V5Arm.ENSEMBLE_NO_GRAPH.value,
        V5Arm.ENSEMBLE_WITH_GRAPH.value,
        V5Arm.FULL_SENTINEL.value,
    }
    assert {measurement.name for measurement in label.measurements} >= {
        "roc_auc",
        "pr_auc",
        "roc_auc_delta",
    }

    for name in (
        "identity_rename",
        "future_causality",
        "equal_time_isolation",
        "feature_leakage",
    ):
        assert by_name[name].passed is True
        assert all(measurement.delta == 0.0 for measurement in by_name[name].measurements)

    identity_evidence = json.loads(by_name["identity_rename"].row_evidence_json)
    assert set(identity_evidence["identity_domains"]) >= {
        "account",
        "actor",
        "campaign",
        "command",
        "counterparty",
        "event",
        "evidence",
        "payment",
        "request",
    }
    assert all(
        domain["original_sha256"] != domain["renamed_sha256"]
        for domain in identity_evidence["identity_domains"].values()
        if domain["count"]
    )
    future_evidence = json.loads(by_name["future_causality"].row_evidence_json)
    assert future_evidence["future_rows_are_retained_execution_evidence"] is True
    assert future_evidence["inserted_execution_evidence_sha256"]

    benign = by_name["benign_only"]
    recall = next(item for item in benign.measurements if item.name == "recall")
    assert recall.applicability is V5MetricApplicability.UNDEFINED
    assert recall.denominator == 0
    assert {item.name for item in benign.measurements} >= {
        "false_decline_rate",
        "challenge_rate",
        "review_rate",
        "p95_latency_ms",
    }
    benign_evidence = json.loads(benign.row_evidence_json)
    assert set(benign_evidence["arms"]) == {
        V5Arm.RULES_ONLY.value,
        V5Arm.ENSEMBLE_NO_GRAPH.value,
        V5Arm.ENSEMBLE_WITH_GRAPH.value,
        V5Arm.FULL_SENTINEL.value,
    }
    assert all(benign_evidence["arms"][arm]["rows"] for arm in benign_evidence["arms"])

    fraud = by_name["fraud_only_diagnostic"]
    assert fraud.qualifies_for_readiness is False
    assert fraud.passed is True
    assert set(json.loads(fraud.row_evidence_json)["arms"]) == set(
        benign_evidence["arms"]
    )
