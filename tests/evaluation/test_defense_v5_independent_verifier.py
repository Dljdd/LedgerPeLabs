"""Adversarial and process-isolation checks for the v5 offline verifier."""

from __future__ import annotations

import ast
import base64
import hashlib
import json
import os
import subprocess
import sys
import zlib
from collections.abc import Callable
from pathlib import Path

import pytest

from apar.v5_independent_verifier import (
    IndependentVerificationError,
    verify_evidence_bytes,
)
from tests.evaluation.v5_safe_evidence_fixture import (
    ROOT,
    safe_v5_evidence_bytes,
)


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode()


def _digest(value: object) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _unpack(packed: dict[str, object]) -> dict[str, object]:
    return json.loads(zlib.decompress(base64.b64decode(packed["content_base64"])))


def _repack(kind: str, document: dict[str, object]) -> dict[str, object]:
    raw = _canonical(document)
    compressed = zlib.compress(raw, level=9)
    packed: dict[str, object] = {
        "kind": kind,
        "compression": "zlib-9",
        "content_base64": base64.b64encode(compressed).decode("ascii"),
        "content_sha256": hashlib.sha256(raw).hexdigest(),
        "compressed_sha256": hashlib.sha256(compressed).hexdigest(),
        "uncompressed_bytes": len(raw),
        "compressed_bytes": len(compressed),
    }
    packed["packed_sha256"] = _digest(packed)
    return packed


def _mutate_packed(
    payload: dict[str, object],
    field: str,
    index: int | None,
    mutator: Callable[[dict[str, object]], None],
) -> None:
    packed = payload[field] if index is None else payload[field][index]
    document = _unpack(packed)
    mutator(document)
    rebound = _repack(str(packed["kind"]), document)
    if index is None:
        payload[field] = rebound
    else:
        payload[field][index] = rebound


def _mutated_envelope(mutator: Callable[[dict[str, object]], None]) -> bytes:
    envelope = json.loads(safe_v5_evidence_bytes())
    payload = json.loads(zlib.decompress(base64.b64decode(envelope["payload_base64"])))
    mutator(payload)
    payload["payload_sha256"] = _digest(
        {key: value for key, value in payload.items() if key != "payload_sha256"}
    )
    raw = _canonical(payload)
    compressed = zlib.compress(raw, level=9)
    rebound = {
        "schema_version": "apar-sentinel-v5-evidence-envelope/1",
        "compression": "zlib-9",
        "payload_base64": base64.b64encode(compressed).decode("ascii"),
        "payload_sha256": hashlib.sha256(raw).hexdigest(),
        "compressed_sha256": hashlib.sha256(compressed).hexdigest(),
        "uncompressed_bytes": len(raw),
        "compressed_bytes": len(compressed),
    }
    rebound["envelope_sha256"] = _digest(rebound)
    return _canonical(rebound)


def _mutate_event(payload: dict[str, object]) -> None:
    def mutate(artifact: dict[str, object]) -> None:
        manifest = json.loads(artifact["payload_json"])
        event = json.loads(manifest["event_records"][0]["event_json"])
        event["event_type"] = "unknown_future_outcome"
        manifest["event_records"][0]["event_json"] = json.dumps(
            event, sort_keys=True, separators=(",", ":")
        )
        artifact["payload_json"] = _canonical(manifest).decode()

    _mutate_packed(payload, "execution_artifact_pool", 0, mutate)


def _mutate_ledger(payload: dict[str, object]) -> None:
    for index, packed in enumerate(payload["execution_artifact_pool"]):
        artifact = _unpack(packed)
        manifest = json.loads(artifact["payload_json"])
        if manifest["ledger_postings"]:
            manifest["ledger_postings"][0]["debit"][0][1] = "999999.00"
            artifact["payload_json"] = _canonical(manifest).decode()
            payload["execution_artifact_pool"][index] = _repack("execution_artifact", artifact)
            return
    raise AssertionError("safe fixture lacks a ledger posting")


def _mutate_trust(payload: dict[str, object]) -> None:
    for index, packed in enumerate(payload["execution_artifact_pool"]):
        artifact = _unpack(packed)
        manifest = json.loads(artifact["payload_json"])
        if manifest["trust_records"]:
            manifest["trust_records"][0]["receipt_hash"] = "0" * 64
            artifact["payload_json"] = _canonical(manifest).decode()
            payload["execution_artifact_pool"][index] = _repack("execution_artifact", artifact)
            return
    raise AssertionError("safe fixture lacks trust evidence")


def _packed_mutation(
    field: str,
    index: int | None,
    mutator: Callable[[dict[str, object]], None],
) -> Callable[[dict[str, object]], None]:
    return lambda payload: _mutate_packed(payload, field, index, mutator)


_MUTATIONS: tuple[tuple[str, Callable[[dict[str, object]], None]], ...] = (
    (
        "metric",
        _packed_mutation(
            "complete_metrics",
            3,
            lambda item: item["aggregate"]["recall"].__setitem__("value", 0.123),
        ),
    ),
    (
        "denominator",
        _packed_mutation(
            "complete_metrics",
            3,
            lambda item: item["aggregate"]["precision"].__setitem__("denominator", 1.0),
        ),
    ),
    (
        "family",
        _packed_mutation(
            "complete_metrics",
            3,
            lambda item: item["by_family"][0].__setitem__("family", "forged_family"),
        ),
    ),
    ("event", _mutate_event),
    ("ledger", _mutate_ledger),
    ("trust", _mutate_trust),
    (
        "calibration",
        _packed_mutation(
            "complete_metrics",
            3,
            lambda item: item["calibration"]["bins"][0].__setitem__("count", 999),
        ),
    ),
    (
        "bootstrap",
        _packed_mutation(
            "complete_metrics",
            3,
            lambda item: item["bootstrap"]["samples"][0]["campaign_ids"].pop(),
        ),
    ),
    (
        "control",
        _packed_mutation(
            "controls",
            None,
            lambda item: item["controls"][0].__setitem__(
                "passed", not item["controls"][0]["passed"]
            ),
        ),
    ),
    (
        "arm",
        _packed_mutation("arm_results", 1, lambda item: item.__setitem__("arm", "full_sentinel")),
    ),
    (
        "spec",
        _packed_mutation(
            "arm_results",
            3,
            lambda item: item["arm_spec"]["threshold_values"][0].__setitem__(1, 0.999),
        ),
    ),
    (
        "row_order",
        _packed_mutation(
            "arm_results",
            0,
            lambda item: item["row_evidence"].__setitem__(
                slice(0, 2), list(reversed(item["row_evidence"][:2]))
            ),
        ),
    ),
    (
        "support",
        _packed_mutation(
            "arm_results",
            0,
            lambda item: item["row_evidence"][1]["support"].__setitem__(
                "event_id", item["row_evidence"][0]["support"]["event_id"]
            ),
        ),
    ),
    (
        "gate",
        lambda p: p["readiness"]["gates"][0].__setitem__(
            "passed", not p["readiness"]["gates"][0]["passed"]
        ),
    ),
    (
        "status",
        lambda p: p["readiness"].__setitem__(
            "status", "ready" if p["readiness"]["status"] == "not_ready" else "not_ready"
        ),
    ),
)


@pytest.mark.parametrize(("layer", "mutator"), _MUTATIONS, ids=[item[0] for item in _MUTATIONS])
def test_independent_verifier_rejects_rebound_layer_mutation(
    layer: str, mutator: Callable[[dict[str, object]], None]
) -> None:
    del layer
    with pytest.raises(IndependentVerificationError):
        verify_evidence_bytes(_mutated_envelope(mutator), root=ROOT)


def test_independent_verifier_has_a_static_production_import_boundary() -> None:
    source_path = ROOT / "src/apar/v5_independent_verifier.py"
    tree = ast.parse(source_path.read_text())
    imported = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    } | {node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)}
    forbidden = (
        "apar.evaluation.v5_arms",
        "apar.evaluation.v5_controls",
        "apar.evaluation.v5_evaluation",
        "apar.evaluation.v5_evidence_bundle",
        "apar.evaluation.v5_metrics",
        "apar.evaluation.v5_population",
        "apar.simulator",
        "apar.trust",
    )
    assert not any(
        module == prefix or module.startswith(f"{prefix}.")
        for module in imported
        for prefix in forbidden
    )


def test_independent_verifier_cli_is_deterministic_across_hash_seeds(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "safe-evidence.json"
    artifact.write_bytes(safe_v5_evidence_bytes())
    outputs: list[str] = []
    for seed in ("1", "987654"):
        environment = {**os.environ, "PYTHONHASHSEED": seed}
        completed = subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts/verify_defense_v5_evidence.py"),
                "--root",
                str(ROOT),
                str(artifact),
            ],
            cwd=ROOT,
            env=environment,
            check=False,
            capture_output=True,
            text=True,
        )
        assert completed.returncode == 0, completed.stderr
        outputs.append(completed.stdout)
    assert outputs[0] == outputs[1]
    assert json.loads(outputs[0])["verified"] is True
