"""Tests for the sealed, pre-execution Defend v2 admission contract."""

from __future__ import annotations

import hashlib

import pytest

from apar.evaluation.v2_preregistration import (
    ExecutionReceipt,
    V2Preregistration,
    V2PreregistrationError,
    admit_v2_execution,
    sign_v2_preregistration,
)
from apar.runs.wire import canonical_json_bytes
from tests.evaluation.v2_authority import ephemeral_v2_authority


def test_missing_seed_commitments_is_rejected() -> None:
    """Removing evaluator-held seed bindings must make a preregistration unusable."""
    payload = complete_preregistration_payload()
    del payload["seed_commitments"]

    with pytest.raises(V2PreregistrationError, match="seed_commitments"):
        V2Preregistration.model_validate(payload)


def test_missing_committed_protocol_profile_binding_is_rejected() -> None:
    """A digest-only preregistration cannot float free of the frozen V2 profile."""
    payload = complete_preregistration_payload()
    del payload["protocol_profile_sha256"]
    signer = ephemeral_v2_authority().evaluator

    with pytest.raises(V2PreregistrationError, match="protocol_profile_sha256"):
        sign_v2_preregistration(payload, signer=signer)


def test_signature_and_manifest_bindings_cover_every_required_contract() -> None:
    """Changing one bound source digest must invalidate the signed contract."""
    preregistration = signed_preregistration()
    tampered = preregistration.model_copy(update={"source_manifest_sha256": digest("other")})

    assert preregistration.verify_signature() is True
    assert preregistration.verify_manifest_bindings() is True
    assert tampered.verify_signature() is False
    assert tampered.verify_manifest_bindings() is True


def test_second_admission_is_denied() -> None:
    """One completed confirmatory receipt must consume the only admission."""
    preregistration = signed_preregistration()
    receipt = ExecutionReceipt(
        preregistration_id=preregistration.preregistration_id,
        execution_nonce=preregistration.execution_nonce,
    )

    admission = admit_v2_execution(
        preregistration,
        existing_receipts=(receipt,),
        sealed_preregistration=preregistration,
    )

    assert (admission.admitted, admission.reason) == (
        False,
        "maximum_confirmatory_attempts_exhausted",
    )


def test_empty_general_sequence_is_admitted() -> None:
    """An empty non-list Sequence must leave the confirmatory admission available."""
    preregistration = signed_preregistration()

    admission = admit_v2_execution(
        preregistration,
        existing_receipts=range(0),
        sealed_preregistration=preregistration,
    )

    assert (admission.admitted, admission.reason, admission.execution_nonce) == (
        True,
        None,
        preregistration.execution_nonce,
    )


def test_nonempty_general_sequence_exhausts_admission() -> None:
    """A prior receipt represented by a non-list Sequence must consume the attempt."""
    preregistration = signed_preregistration()
    admission = admit_v2_execution(
        preregistration,
        existing_receipts=range(1),
        sealed_preregistration=preregistration,
    )

    assert (admission.admitted, admission.reason) == (
        False,
        "maximum_confirmatory_attempts_exhausted",
    )


def test_caller_approval_flag_is_not_an_admission_path() -> None:
    """An unbound approval boolean must be rejected instead of authorizing execution."""
    payload = complete_preregistration_payload()
    payload["approved"] = True

    with pytest.raises(V2PreregistrationError, match="approved"):
        V2Preregistration.model_validate(payload)


def signed_preregistration() -> V2Preregistration:
    return ephemeral_v2_authority().preregistration


def complete_preregistration_payload() -> dict[str, object]:
    synthetic_scope = (
        "Synthetic-only evaluation; not a real-world prevalence or external-validity claim."
    )
    return {
        "schema_version": "1.0.0",
        "preregistration_id": "apar-defend-v2",
        "protocol_profile_sha256": digest("protocol-profile"),
        "manifest_registry_sha256": digest("manifest-registry"),
        "source_manifest_sha256": digest("source"),
        "feature_manifest_sha256": digest("feature"),
        "candidate_grid_sha256": digest("candidate-grid"),
        "population_manifest_sha256": digest("population"),
        "seed_commitments": (
            {"name": "operating_population", "commitment_sha256": "1" * 64},
            {"name": "campaign_injection", "commitment_sha256": "2" * 64},
        ),
        "evaluator_capability_sha256": digest("evaluator-capability"),
        "metrics_manifest_sha256": digest("metrics"),
        "bootstrap_manifest_sha256": digest("bootstrap"),
        "controls_manifest_sha256": digest("controls"),
        "budget_manifest_sha256": digest("budget"),
        "reporting_schema_sha256": digest("reporting"),
        "fidelity_validation_bundle_sha256": digest("fidelity"),
        "synthetic_scope": synthetic_scope,
        "synthetic_scope_sha256": hashlib.sha256(canonical_json_bytes(synthetic_scope)).hexdigest(),
        "execution_nonce": digest("one-confirmatory-attempt"),
        "maximum_confirmatory_attempts": 1,
    }


def digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()
