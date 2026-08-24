"""Locked-development payload and deterministic run-binding contracts."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from apar.evaluation.v5_evidence_protocol import load_v5_evidence_protocol
from apar.evaluation.v5_protocol import load_v5_development_protocol
from apar.evaluation.v5_run_mode import (
    V5RunMode,
    build_v5_run_support_plan,
)

ROOT = Path(__file__).resolve().parents[2]


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode()


def _locked_binding_values() -> dict[str, object]:
    from apar.evaluation.v5_run_mode import V5LockedEvidenceRunBinding

    evidence = load_v5_evidence_protocol(
        ROOT / "config/defense/defense-v5-evidence.json", root=ROOT
    )
    development = load_v5_development_protocol(
        ROOT / "config/defense/defense-v5-development.json"
    )
    plan = build_v5_run_support_plan(
        mode=V5RunMode.LOCKED_DEVELOPMENT,
        evidence_protocol=evidence,
        development_protocol=development,
    )
    values: dict[str, object] = {
        "schema_version": "apar-sentinel-v5-locked-run-binding/1",
        "mode": "locked_development",
        "profile": "production",
        "development_test_seed": 2404,
        "source_commit": "1" * 40,
        "source_tree_oid": "2" * 40,
        "preregistration_commit": "5" * 40,
        "preregistration_path": (
            "config/defense/defense-v5-locked-development-preregistration.json"
        ),
        "preregistration_sha256": "3" * 64,
        "base_protocol_sha256": evidence.base_protocol_sha256,
        "arm_protocol_sha256": evidence.arm_protocol_sha256,
        "evidence_protocol_sha256": evidence.evidence_protocol_sha256,
        "implementation_sha256": evidence.implementation_sha256,
        "catalog_sha256": "4" * 64,
        "support_plan": plan.model_dump(mode="json"),
        "candidate_manifest_path": (
            evidence.locked_artifact_storage.candidate_manifest_path
        ),
        "storage_schema_version": (
            evidence.locked_artifact_storage.schema_version
        ),
        "payload_schema_version": (
            "apar-sentinel-v5-locked-development-payload/2"
        ),
    }
    values["run_binding_sha256"] = V5LockedEvidenceRunBinding.compute_digest(values)
    return values


def test_locked_evidence_contract_module_exists() -> None:
    """Without a distinct locked payload, the legacy summary remains the only output."""
    assert importlib.util.find_spec("apar.evaluation.v5_locked_evidence") is not None


def test_locked_run_binding_is_content_addressed_and_closed() -> None:
    """Mutating mode, profile, seed, or source bindings must invalidate the run."""
    from apar.evaluation.v5_run_mode import V5LockedEvidenceRunBinding

    values = _locked_binding_values()
    binding = V5LockedEvidenceRunBinding.model_validate(values)
    assert binding.mode is V5RunMode.LOCKED_DEVELOPMENT
    assert binding.profile.value == "production"
    assert binding.development_test_seed == 2404
    assert binding.preregistration_commit == "5" * 40
    assert binding.run_binding_sha256 == V5LockedEvidenceRunBinding.compute_digest(
        binding.model_dump(mode="json", exclude={"run_binding_sha256"})
    )

    for field, replacement in (
        ("mode", "safe_validation"),
        ("profile", "smoke"),
        ("development_test_seed", 404),
        ("source_commit", "f" * 40),
        ("preregistration_commit", "f" * 40),
    ):
        mutated = {**values, field: replacement}
        with pytest.raises((ValueError, ValidationError), match="binding|mode|profile|seed|digest"):
            V5LockedEvidenceRunBinding.model_validate(mutated)


def test_locked_payload_rejects_legacy_or_partial_documents() -> None:
    """Accepting missing controls/metrics would make the legacy summary verifiable."""
    from apar.evaluation.v5_locked_evidence import V5LockedEvidencePayload

    with pytest.raises(ValidationError):
        V5LockedEvidencePayload.model_validate(
            {
                "schema_version": "apar-sentinel-v5-locked-development-payload/2",
                "run_binding": _locked_binding_values(),
                "arm_results": [],
            }
        )


def test_independent_locked_payload_verifier_exists() -> None:
    """The locked path must not depend on the legacy summary readiness verifier."""
    from apar import v5_independent_verifier

    assert callable(
        getattr(v5_independent_verifier, "verify_locked_evidence_payload_bytes", None)
    )


def test_locked_payload_builder_requires_exact_ordered_four_arms() -> None:
    """A partial arm set must fail before a production payload can be serialized."""
    from apar.evaluation.v5_locked_evidence import build_v5_locked_evidence_payload

    evidence = load_v5_evidence_protocol(
        ROOT / "config/defense/defense-v5-evidence.json", root=ROOT
    )
    with pytest.raises(ValueError, match="exact ordered four arm"):
        build_v5_locked_evidence_payload(
            run_binding=_locked_binding_values(),
            attempt_receipt_sha256="6" * 64,
            evidence_protocol=evidence,
            catalog_sha256="4" * 64,
            arm_results=(),
            controls=None,
        )


def test_locked_payload_builder_rejects_smoke_sized_production_support() -> None:
    """A closed mode tag cannot make a smoke corpus satisfy production support."""
    from apar.evaluation.v5_locked_evidence import _validate_locked_support
    from apar.evaluation.v5_run_mode import V5LockedEvidenceRunBinding

    binding = V5LockedEvidenceRunBinding.model_validate(_locked_binding_values())
    support = SimpleNamespace(label=0, family="legitimate")
    rows = tuple(SimpleNamespace(support=support) for _index in range(200))
    partitions = tuple(
        SimpleNamespace(partition=name, support_records=rows)
        for name in ("train", "calibration", "threshold")
    )
    result = SimpleNamespace(
        row_evidence=rows,
        arm_spec=SimpleNamespace(training_partitions=partitions),
    )
    with pytest.raises(ValueError, match="development-test support"):
        _validate_locked_support(results=(result,) * 4, run_binding=binding)


def test_independent_locked_support_rejects_partial_production_rows() -> None:
    """Independent replay must reject an otherwise ordered smoke-sized support."""
    from apar import v5_independent_verifier as verifier

    evidence = load_v5_evidence_protocol(
        ROOT / "config/defense/defense-v5-evidence.json", root=ROOT
    )
    development = load_v5_development_protocol(
        ROOT / "config/defense/defense-v5-development.json"
    )
    expected = build_v5_run_support_plan(
        mode=V5RunMode.LOCKED_DEVELOPMENT,
        evidence_protocol=evidence,
        development_protocol=development,
    ).model_dump(mode="json")
    rows = [
        {"support": {"label": 0, "family": "legitimate"}}
        for _index in range(200)
    ]
    result = {
        "arm_spec": {
            "training_partitions": [
                {"partition": name, "support_records": [item["support"] for item in rows]}
                for name in ("train", "calibration", "threshold")
            ]
        }
    }
    with pytest.raises(
        verifier.IndependentVerificationError,
        match="development-test production support",
    ):
        verifier._verify_locked_support(
            expected_plan=expected,
            result=result,
            rows=rows,
            artifact_count=1,
        )


def test_locked_verifier_rejects_legacy_summary_and_safe_envelope() -> None:
    """Dispatching either legacy schema into locked verification would weaken evidence."""
    from apar.v5_independent_verifier import (
        IndependentVerificationError,
        verify_locked_evidence_payload_bytes,
    )

    legacy = (
        ROOT / "docs/experiments/defense-v5-development-result.json"
    ).read_bytes()
    with pytest.raises(IndependentVerificationError, match="locked payload schema"):
        verify_locked_evidence_payload_bytes(legacy, root=ROOT)

    safe_envelope = json.dumps(
        {
            "schema_version": "apar-sentinel-v5-evidence-envelope/2",
            "compression": "zlib-9",
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    with pytest.raises(IndependentVerificationError, match="locked payload schema"):
        verify_locked_evidence_payload_bytes(safe_envelope, root=ROOT)


def test_safe_verifier_rejects_a_locked_payload_schema() -> None:
    """Locked evidence cannot be relabeled as repeatable safe validation."""
    from apar.v5_independent_verifier import (
        IndependentVerificationError,
        verify_evidence_bytes,
    )

    locked = _canonical_bytes(
        {"schema_version": "apar-sentinel-v5-locked-development-payload/2"}
    )
    with pytest.raises(IndependentVerificationError, match="envelope"):
        verify_evidence_bytes(locked, root=ROOT)


def test_locked_verifier_rejects_partial_payload_before_evidence_replay() -> None:
    """A schema tag alone must never qualify as complete locked evidence."""
    from apar.v5_independent_verifier import (
        IndependentVerificationError,
        verify_locked_evidence_payload_bytes,
    )

    partial = _canonical_bytes(
        {"schema_version": "apar-sentinel-v5-locked-development-payload/2"}
    )
    with pytest.raises(IndependentVerificationError, match="locked payload fields"):
        verify_locked_evidence_payload_bytes(partial, root=ROOT)


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("mode", "safe_validation"),
        ("profile", "smoke"),
        ("development_test_seed", 404),
    ],
)
def test_locked_verifier_rejects_raw_run_relabeling(
    field: str, replacement: object
) -> None:
    """Independent verification must not rely on the production Pydantic model."""
    from apar.v5_independent_verifier import (
        IndependentVerificationError,
        verify_locked_evidence_payload_bytes,
    )

    run_binding = _locked_binding_values()
    run_binding[field] = replacement
    payload: dict[str, object] = {
        "schema_version": "apar-sentinel-v5-locked-development-payload/2",
        "run_binding": run_binding,
        "attempt_receipt_sha256": "6" * 64,
        "evidence_protocol": {},
        "catalog_sha256": "4" * 64,
        "execution_artifact_pool": [],
        "arm_results": [],
        "complete_metrics": [],
        "controls": {},
        "readiness": {},
        "deterministic_core": {},
        "observational_latency": {},
    }
    payload["payload_sha256"] = hashlib.sha256(_canonical_bytes(payload)).hexdigest()
    with pytest.raises(
        IndependentVerificationError, match="locked run mode/profile/seed"
    ):
        verify_locked_evidence_payload_bytes(_canonical_bytes(payload), root=ROOT)
