"""Fresh-process defender runtime adapter tests."""

from __future__ import annotations

import hashlib
import json

import pytest

from apar.evaluation.v3_isolation import build_isolation_manifest
from apar.evaluation.v3_runtime import (
    DefenderRequest,
    V3RuntimeError,
    run_defender_arm,
)


def _request(arm: str = "rules_only") -> DefenderRequest:
    document = {
        "arm": arm,
        "protocol_id": "apar-defend-v3",
        "execution_nonce": "a" * 64,
    }
    payload = json.dumps(document, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return DefenderRequest(
        arm=arm,
        protocol_id="apar-defend-v3",
        execution_nonce="a" * 64,
        input_payload=payload,
    )


def test_successful_subprocess_execution() -> None:
    manifest = build_isolation_manifest(protocol_id="apar-defend-v3", timeout_seconds=15.0)
    response = run_defender_arm(_request(), manifest=manifest)
    assert response.arm == "rules_only"
    assert response.output_sha256 == hashlib.sha256(response.output_payload).hexdigest()


def test_invalid_arm_rejected() -> None:
    manifest = build_isolation_manifest(protocol_id="apar-defend-v3")
    with pytest.raises(V3RuntimeError, match="invalid defender arm"):
        run_defender_arm(_request("unknown"), manifest=manifest)


def test_protocol_mismatch_rejected() -> None:
    manifest = build_isolation_manifest(protocol_id="apar-defend-v3")
    request = DefenderRequest(
        arm="rules_only",
        protocol_id="other-protocol",
        execution_nonce="a" * 64,
        input_payload=b"{}",
    )
    with pytest.raises(V3RuntimeError, match="protocol mismatch"):
        run_defender_arm(request, manifest=manifest)


def test_input_digest_mismatch_rejected() -> None:
    manifest = build_isolation_manifest(protocol_id="apar-defend-v3")
    request = DefenderRequest(
        arm="rules_only",
        protocol_id="apar-defend-v3",
        execution_nonce="a" * 64,
        input_payload=b"tampered",
    )
    with pytest.raises(V3RuntimeError, match="input digest mismatch"):
        run_defender_arm(request, manifest=manifest)
