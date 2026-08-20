"""Read-only admission checks for the sealed Defend v2 protocol."""

from __future__ import annotations

import hashlib
from pathlib import Path

from apar.evaluation.gates import EvaluatorSigningIdentity
from apar.evaluation.v2_preexecution import verify_v2_preexecution
from apar.evaluation.v2_preregistration import V2Preregistration, sign_v2_preregistration
from apar.runs.wire import canonical_json_bytes

ROOT = Path(__file__).resolve().parents[2]


def test_signed_preregistration_and_frozen_v1_roots_are_not_executed() -> None:
    """A sealed, unconsumed protocol is reported without starting any work."""
    report = verify_v2_preexecution(ROOT, signed_preregistration())

    assert (report.status, report.codes) == ("not_executed", ())


def test_hidden_import_in_defender_fails_preexecution(tmp_path: Path) -> None:
    """Defender code must not gain a path to evaluator-only modules."""
    (tmp_path / "src/apar/defense").mkdir(parents=True)
    (tmp_path / "src/apar/defense/bad.py").write_text(
        "from apar.evaluation_hidden import worker\n", encoding="utf-8"
    )

    report = verify_v2_preexecution(tmp_path, signed_preregistration())

    assert "HIDDEN_IMPORT_BOUNDARY" in report.codes


def test_existing_v2_receipt_fails_preexecution(tmp_path: Path) -> None:
    """A consumed confirmatory attempt cannot be represented as pre-execution."""
    (tmp_path / ".apar/defense-v2").mkdir(parents=True)
    (tmp_path / ".apar/defense-v2/execution-receipt.json").write_text(
        '{"preregistration_id":"apar-defend-v2"}', encoding="utf-8"
    )

    report = verify_v2_preexecution(tmp_path, signed_preregistration())

    assert "V2_EXECUTION_RECEIPT_PRESENT" in report.codes


def test_invalid_signature_fails_preexecution() -> None:
    """Unsafe model copies cannot turn an invalid admission into a pass."""
    preregistration = signed_preregistration().model_copy(
        update={"signature_base64": ""}
    )

    report = verify_v2_preexecution(ROOT, preregistration)

    assert "PREREGISTRATION_INVALID" in report.codes


def signed_preregistration() -> V2Preregistration:
    signer = EvaluatorSigningIdentity.from_private_bytes(b"v" * 32)
    return sign_v2_preregistration(_preregistration_payload(), signer=signer)


def _preregistration_payload() -> dict[str, object]:
    synthetic_scope = (
        "Synthetic-only evaluation; not a real-world prevalence or external-validity claim."
    )
    return {
        "schema_version": "1.0.0",
        "preregistration_id": "apar-defend-v2",
        "source_manifest_sha256": _digest("source"),
        "feature_manifest_sha256": _digest("feature"),
        "candidate_grid_sha256": _digest("candidate-grid"),
        "population_manifest_sha256": _digest("population"),
        "seed_commitments": (
            {"name": "operating_population", "commitment_sha256": _digest("population-seed")},
            {"name": "campaign_injection", "commitment_sha256": _digest("injection-seed")},
        ),
        "evaluator_capability_sha256": _digest("evaluator-capability"),
        "metrics_manifest_sha256": _digest("metrics"),
        "bootstrap_manifest_sha256": _digest("bootstrap"),
        "controls_manifest_sha256": _digest("controls"),
        "reporting_schema_sha256": _digest("reporting"),
        "fidelity_validation_bundle_sha256": _digest("fidelity"),
        "synthetic_scope": synthetic_scope,
        "synthetic_scope_sha256": hashlib.sha256(
            canonical_json_bytes(synthetic_scope)
        ).hexdigest(),
        "execution_nonce": _digest("one-confirmatory-attempt"),
        "maximum_confirmatory_attempts": 1,
    }


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()
