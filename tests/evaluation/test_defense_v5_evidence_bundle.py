"""Complete Sentinel v5 evidence bundle and readiness semantics."""

from __future__ import annotations

import json

import pytest

from apar.evaluation.v5_evidence_bundle import V5EvidenceEnvelope, V5PackedDocument
from apar.v5_independent_verifier import verify_evidence_bytes
from tests.evaluation.v5_safe_evidence_fixture import (
    ROOT,
    safe_v5_evidence_bytes,
)


def test_packed_document_is_bounded_content_addressed_and_fail_closed() -> None:
    document = {"rows": [{"event_id": f"event-{index}", "value": index} for index in range(32)]}
    packed = V5PackedDocument.pack(kind="test", document=document, max_uncompressed_bytes=16_384)
    assert packed.document() == document
    assert packed.uncompressed_bytes == len(
        json.dumps(document, sort_keys=True, separators=(",", ":")).encode()
    )
    with pytest.raises(ValueError, match="digest"):
        type(packed).model_validate({**packed.model_dump(mode="json"), "content_sha256": "0" * 64})


def test_real_safe_evidence_bundle_binds_controls_metrics_intervals_and_readiness() -> None:
    """A partial point-estimate result must not satisfy the complete bundle contract."""
    serialized = safe_v5_evidence_bytes()
    envelope = V5EvidenceEnvelope.model_validate_json(serialized)
    payload = envelope.payload()
    assert payload.execution_artifact_pool
    assert all(item.kind == "execution_artifact" for item in payload.execution_artifact_pool)
    assert all(result.kind == "arm_result" for result in payload.arm_results)
    assert all(metrics.kind == "complete_metrics" for metrics in payload.complete_metrics)
    assert payload.controls.kind == "executed_controls"
    assert tuple(result.arm for result in payload.arm_results) == (
        "rules_only",
        "ensemble_no_graph",
        "ensemble_with_graph",
        "full_sentinel",
    )
    assert len(payload.complete_metrics) == 4
    assert payload.controls.suite_sha256
    assert payload.readiness.status in {"ready", "not_ready"}
    assert all(gate.source_sha256 for gate in payload.readiness.gates)
    assert any(gate.family is not None for gate in payload.readiness.gates)
    assert (
        envelope.uncompressed_bytes < payload.evidence_protocol.bounds.max_serialized_evidence_bytes
    )
    assert envelope.compressed_bytes < envelope.uncompressed_bytes
    assert envelope == type(envelope).model_validate_json(envelope.model_dump_json())
    report = verify_evidence_bytes(serialized, root=ROOT)
    assert report["verified"] is True
    assert report["status"] == payload.readiness.status


def test_safe_evidence_separates_deterministic_core_from_observed_latency() -> None:
    """Removing either authenticated layer must collapse the two-layer commitment."""
    envelope = V5EvidenceEnvelope.model_validate_json(safe_v5_evidence_bytes())
    payload = envelope.payload()
    document = payload.model_dump(mode="json")
    assert "deterministic_core" in document
    assert "observational_latency" in document
    assert document["deterministic_core"]["schema_version"] == (
        "apar-sentinel-v5-deterministic-core/1"
    )
    assert document["deterministic_core"]["core_sha256"]
    assert document["observational_latency"]["kind"] == "observational_latency"
    latency = document["observational_latency"]
    assert latency["content_sha256"] != document["deterministic_core"]["core_sha256"]
    report = verify_evidence_bytes(envelope.serialized_bytes(), root=ROOT)
    assert report["deterministic_core_sha256"] == document["deterministic_core"][
        "core_sha256"
    ]
    assert report["observational_latency_sha256"]
