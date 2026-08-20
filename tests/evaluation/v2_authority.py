"""Ephemeral V2 authority material used only by tests."""

from __future__ import annotations

import hashlib
import secrets
from dataclasses import dataclass

from apar.evaluation.gates import EvaluatorSigningIdentity
from apar.evaluation.v2_preregistration import (
    SYNTHETIC_NON_CLAIM,
    V2Preregistration,
    V2VerifiedAuthority,
    _issue_verified_v2_authority,
    sign_v2_preregistration,
)
from apar.runs import RunSigningIdentity
from apar.runs.wire import canonical_json_bytes


@dataclass(frozen=True, slots=True)
class EphemeralV2Authority:
    evaluator: EvaluatorSigningIdentity
    publisher: RunSigningIdentity
    preregistration: V2Preregistration
    verified_authority: V2VerifiedAuthority


def ephemeral_v2_authority() -> EphemeralV2Authority:
    private_seed = secrets.token_bytes(32)
    evaluator = EvaluatorSigningIdentity.from_private_bytes(private_seed)
    publisher = RunSigningIdentity.from_private_bytes(private_seed)
    synthetic_digest = hashlib.sha256(canonical_json_bytes(SYNTHETIC_NON_CLAIM)).hexdigest()
    payload: dict[str, object] = {
        "schema_version": "1.0.0",
        "preregistration_id": f"test-defend-v2-{secrets.token_hex(8)}",
        "protocol_profile_sha256": _digest("protocol-profile"),
        "manifest_registry_sha256": _digest("manifest-registry"),
        "source_manifest_sha256": _digest("source"),
        "feature_manifest_sha256": _digest("feature"),
        "candidate_grid_sha256": _digest("candidate-grid"),
        "population_manifest_sha256": _digest("population"),
        "seed_commitments": (
            {"name": "operating_population", "commitment_sha256": _digest("seed-one")},
            {"name": "campaign_injection", "commitment_sha256": _digest("seed-two")},
        ),
        "evaluator_capability_sha256": _digest("evaluator-capability"),
        "metrics_manifest_sha256": _digest("metrics"),
        "bootstrap_manifest_sha256": _digest("bootstrap"),
        "controls_manifest_sha256": _digest("controls"),
        "budget_manifest_sha256": _digest("budget"),
        "reporting_schema_sha256": _digest("reporting"),
        "fidelity_validation_bundle_sha256": _digest("fidelity"),
        "synthetic_scope": SYNTHETIC_NON_CLAIM,
        "synthetic_scope_sha256": synthetic_digest,
        "execution_nonce": secrets.token_hex(32),
        "maximum_confirmatory_attempts": 1,
    }
    preregistration = sign_v2_preregistration(payload, signer=evaluator)
    verified_authority = _issue_verified_v2_authority(preregistration)
    return EphemeralV2Authority(evaluator, publisher, preregistration, verified_authority)


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()
