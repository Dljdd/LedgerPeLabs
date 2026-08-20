"""Ephemeral V2 authority material used only by tests."""

from __future__ import annotations

import hashlib
import secrets
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from tempfile import TemporaryDirectory

from apar.evaluation.gates import EvaluatorSigningIdentity
from apar.evaluation.v2_preexecution import V2VerifiedAuthority, verify_v2_preexecution
from apar.evaluation.v2_preregistration import (
    SYNTHETIC_NON_CLAIM,
    V2Preregistration,
    sign_v2_preregistration,
)
from apar.runs import RunSigningIdentity
from apar.runs.wire import canonical_json_bytes

ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True, slots=True)
class EphemeralV2Authority:
    evaluator: EvaluatorSigningIdentity
    publisher: RunSigningIdentity
    preregistration: V2Preregistration
    verified_authority: object | None = None
    verification_root: Path | None = None
    _verification_directory: TemporaryDirectory[str] | None = field(
        default=None, repr=False, compare=False
    )


def ephemeral_v2_authority(*, verified: bool = False) -> EphemeralV2Authority:
    private_seed = secrets.token_bytes(32)
    evaluator = EvaluatorSigningIdentity.from_private_bytes(private_seed)
    publisher = RunSigningIdentity.from_private_bytes(private_seed)
    if verified:
        preregistration = _signed_pinned_preregistration(evaluator)
        directory = TemporaryDirectory(prefix="apar-v2-authority-")
        verification_root = Path(directory.name).resolve()
        _copy_preexecution_inputs(verification_root, preregistration)
        report = verify_v2_preexecution(verification_root, preregistration)
        if not report.admissible:
            raise RuntimeError(f"test authority preexecution failed: {report.codes}")
        authority = V2VerifiedAuthority.from_preregistration(preregistration)
        return EphemeralV2Authority(
            evaluator,
            publisher,
            preregistration,
            authority,
            verification_root,
            directory,
        )

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
    return EphemeralV2Authority(evaluator, publisher, preregistration)


def _signed_pinned_preregistration(
    evaluator: EvaluatorSigningIdentity,
) -> V2Preregistration:
    raw = (ROOT / "config/defense/competition-v2-preregistration.json").read_bytes()
    sealed = V2Preregistration.from_json(raw[:-1] if raw.endswith(b"\n") else raw)
    payload = sealed.unsigned_document()
    payload.pop("evaluator_key_id")
    payload.pop("evaluator_public_key_base64")
    return sign_v2_preregistration(payload, signer=evaluator)


def _copy_preexecution_inputs(
    verification_root: Path, preregistration: V2Preregistration
) -> None:
    shutil.copytree(ROOT / "src", verification_root / "src")
    for relative in (
        "config/defense/competition-v2-profile.json",
        "config/defense/competition-v2-manifests.json",
        "config/defense/feature-catalog.json",
        "docs/experiments/defense-v1-preregistration.json",
        "docs/experiments/defense-v1-result.json",
        "docs/experiments/defense-v1-run-manifests.json",
        "fixtures/defense/v1/hash-manifest.json",
        "fixtures/defense/v1/defender-bundle.json",
        "fixtures/defense/v1/calibration.json",
        "fixtures/defense/v1/thresholds.json",
        "fixtures/defense/v1/split-manifest.json",
    ):
        destination = verification_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, destination)
    sealed_path = verification_root / "config/defense/competition-v2-preregistration.json"
    sealed_path.write_bytes(preregistration.canonical_bytes() + b"\n")


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()
