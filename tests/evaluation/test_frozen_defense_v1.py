"""Immutable checks for the preregistered 200-campaign defense-v1 result."""

from __future__ import annotations

import base64
import hashlib
import json
import math
from pathlib import Path

import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]

from apar.defense import orchestration
from apar.defense.contracts import ObservedEvent
from apar.defense.gbdt import CatBoostScorer, TrainingReceipt
from apar.evaluation.reporting import PublicArtifactVerifier
from apar.features.builders import build_feature_matrix
from apar.features.catalog import load_feature_catalog
from apar.runs.wire import canonical_json_bytes

ROOT = Path(__file__).resolve().parents[2]
FIXTURE_ROOT = ROOT / "fixtures" / "defense" / "v1"
RUN_LEDGER = ROOT / "docs" / "experiments" / "defense-v1-run-manifests.json"
RESULT = ROOT / "docs" / "experiments" / "defense-v1-result.json"
HASH_MANIFEST = FIXTURE_ROOT / "hash-manifest.json"
PREREGISTRATION = orchestration.load_defense_v1_preregistration(
    ROOT / "docs" / "experiments" / "defense-v1-preregistration.json"
)
PUBLICATION_AUTHORITY = PREREGISTRATION.authority_identities["publication"]

EXPECTED_ARTIFACTS = {
    "calibration.csv",
    "calibration.json",
    "corpus-manifest.json",
    "data-card.md",
    "defender-bundle.json",
    "defense-scorecard.json",
    "defense-scorecard.md",
    "evaluation-truth.parquet",
    "feature-manifest.json",
    "features.parquet",
    "latency-evidence.json",
    "leaderboard.csv",
    "limitations.md",
    "model-card.md",
    "model.cbm",
    "observations.parquet",
    "rules.json",
    "slice-metrics.csv",
    "split-manifest.json",
    "thresholds.json",
    "training-receipt.json",
    "value-workload.csv",
}


def _load_json(path: Path) -> dict[str, object]:
    payload = path.read_bytes()
    document = json.loads(payload)
    assert canonical_json_bytes(document) == payload
    assert isinstance(document, dict)
    return document


def _verify_signed(document: dict[str, object]) -> None:
    verifier = PublicArtifactVerifier(
        signer_key_id=PUBLICATION_AUTHORITY.key_id,
        public_key_base64=PUBLICATION_AUTHORITY.public_key_base64,
    )
    signature = document["signature_base64"]
    assert isinstance(signature, str)
    unsigned = {
        key: value for key, value in document.items() if key != "signature_base64"
    }
    assert document["signer_key_id"] == verifier.key_id
    assert document["public_key_base64"] == verifier.public_key_base64
    assert verifier.verify(unsigned, signature)


def test_frozen_negative_result_and_hash_manifest_are_exact_and_signed() -> None:
    result = _load_json(RESULT)
    _verify_signed(result)
    assert result["status"] == "no_promotion"
    assert result["failure_stage"] == "threshold_selection"
    assert result["failure_reason"] == "operating_budget_infeasible"
    assert result["champion"] is None
    assert result["defender_frozen_at"] is None
    assert result["hidden_released_at"] is None
    assert (
        result["hidden_release_status"]
        == "not_attempted_frozen_defender_unavailable"
    )
    assert result["retuned_after_failure"] is False
    assert result["campaign_count"] == 200
    assert result["synthetic_only"] is True
    assert result["authority_identities"] == {
        role: {
            "key_id": identity.key_id,
            "public_key_base64": identity.public_key_base64,
        }
        for role, identity in PREREGISTRATION.authority_identities.items()
    }

    manifest = _load_json(HASH_MANIFEST)
    _verify_signed(manifest)
    hashes = manifest["artifact_sha256"]
    assert isinstance(hashes, dict)
    assert set(hashes) == EXPECTED_ARTIFACTS
    assert {path.name for path in FIXTURE_ROOT.iterdir()} == {
        *EXPECTED_ARTIFACTS,
        "hash-manifest.json",
    }
    for name, digest in hashes.items():
        assert isinstance(name, str)
        assert isinstance(digest, str)
        assert hashlib.sha256((FIXTURE_ROOT / name).read_bytes()).hexdigest() == digest


def test_frozen_threshold_failure_is_mathematically_load_bearing() -> None:
    thresholds = _load_json(FIXTURE_ROOT / "thresholds.json")
    _verify_signed(thresholds)
    report = thresholds["report"]
    assert isinstance(report, dict)
    assert thresholds["status"] == "no_promotion"
    assert report["feasible"] is False
    assert report["reason"] == "no_candidate_satisfies_operating_budget"
    assert report["row_count"] == 336
    assert report["legitimate_count"] == 84
    assert report["fraud_count"] == 252
    assert report["minimum_review_case_count"] == 6
    assert report["minimum_review_case_rate"] == 6 / 336
    budget = report["budget"]
    assert isinstance(budget, dict)
    assert report["minimum_review_case_rate"] > budget["review_case_rate_max"]
    assert report["minimum_challenge_rate"] == 0.0
    assert report["minimum_false_decline_rate"] == 0.0
    assert report["feasible_candidate_count"] == 0
    assert report["candidate_count"] == 28

    alias = orchestration._load_defense_v1_alias_public(
        FIXTURE_ROOT / "defender-bundle.json",
        expected_kind="infeasible_candidate",
        expected_profile_sha256=orchestration._DEFENSE_V1_PROFILE_SHA256,
        signer_key_id=PUBLICATION_AUTHORITY.key_id,
        public_key_base64=PUBLICATION_AUTHORITY.public_key_base64,
    )
    assert alias.campaign_count == 200
    assert len(alias.authenticated_run_ids) == len(set(alias.authenticated_run_ids)) == 200
    assert alias.export_metadata["failure_reason"] == "operating_budget_infeasible"
    assert alias.export_metadata["threshold_report_digest"] == hashlib.sha256(
        (FIXTURE_ROOT / "thresholds.json").read_bytes()
    ).hexdigest()


def test_frozen_model_and_corpus_reload_without_claiming_a_champion() -> None:
    receipt = TrainingReceipt.model_validate_json(
        (FIXTURE_ROOT / "training-receipt.json").read_bytes()
    )
    scorer = CatBoostScorer.from_bytes(
        (FIXTURE_ROOT / "model.cbm").read_bytes(), receipt
    )
    feature_table = pq.read_table(
        pa.BufferReader((FIXTURE_ROOT / "features.parquet").read_bytes())
    )
    observation_table = pq.read_table(
        pa.BufferReader((FIXTURE_ROOT / "observations.parquet").read_bytes())
    )
    observations = tuple(
        ObservedEvent.model_validate_json(payload)
        for payload in observation_table.column("row_json").to_pylist()
    )
    catalog = load_feature_catalog(FIXTURE_ROOT / "feature-manifest.json")
    full_matrix = build_feature_matrix(observations, catalog)
    event_ids = tuple(feature_table.column("event_id").to_pylist())
    matrix = orchestration._matrix_subset(full_matrix, event_ids)
    by_id = {row.event_id: row for row in matrix.rows}
    for index, event_id in enumerate(event_ids):
        assert all(
            float(feature_table.column(name)[index].as_py())
            == by_id[event_id].values[name]
            for name in catalog.names
        )
    assert feature_table.schema.metadata[b"classification"] == b"defender_visible"
    assert feature_table.schema.metadata[b"catalog_digest"].decode() == (
        matrix.catalog_digest
    )
    scores = scorer.predict(matrix)
    assert len(scores) == len(matrix.rows) > 0
    assert all(math.isfinite(float(value)) and 0.0 <= value <= 1.0 for value in scores)

    ledger = orchestration._load_defense_v1_alias_public(
        RUN_LEDGER,
        expected_kind="run_ledger",
        expected_profile_sha256=orchestration._DEFENSE_V1_PROFILE_SHA256,
        signer_key_id=PUBLICATION_AUTHORITY.key_id,
        public_key_base64=PUBLICATION_AUTHORITY.public_key_base64,
    )
    corpus = orchestration._load_defense_v1_alias_public(
        FIXTURE_ROOT / "corpus-manifest.json",
        expected_kind="corpus_envelope",
        expected_profile_sha256=orchestration._DEFENSE_V1_PROFILE_SHA256,
        signer_key_id=PUBLICATION_AUTHORITY.key_id,
        public_key_base64=PUBLICATION_AUTHORITY.public_key_base64,
    )
    assert ledger.authenticated_run_ids == corpus.authenticated_run_ids
    assert len(ledger.authenticated_run_ids) == 200
    assert _load_json(RESULT)["run_ledger_sha256"] == ledger.artifact["sha256"]

    observation_bytes = (FIXTURE_ROOT / "observations.parquet").read_bytes()
    truth_bytes = (FIXTURE_ROOT / "evaluation-truth.parquet").read_bytes()
    observations = pq.read_table(pa.BufferReader(observation_bytes))
    truth = pq.read_table(pa.BufferReader(truth_bytes))
    assert observations.schema.metadata[b"classification"] == b"defender_visible"
    assert truth.schema.metadata[b"classification"] == b"restricted_evaluator_only"
    assert observations.num_rows > 0
    assert truth.num_rows > 0


def test_frozen_public_files_do_not_expose_restricted_rows_or_hidden_evidence() -> None:
    forbidden = (
        b"input_labels_digest",
        b"input_mandatory_actions_digest",
        b"row_is_fraud",
        b"row_net_settled_values",
        b"hidden_context_ref",
        b"hidden_proof_digest",
        b"net_settled_value_totals",
    )
    public_paths = tuple(
        path
        for path in FIXTURE_ROOT.iterdir()
        if path.name != "evaluation-truth.parquet"
    ) + (RUN_LEDGER, RESULT)
    for path in public_paths:
        payload = path.read_bytes()
        lowered = payload.lower()
        assert not any(token in lowered for token in forbidden)
        assert b"private_key" not in lowered
        assert b"hidden_seed" not in lowered

    split = _load_json(FIXTURE_ROOT / "split-manifest.json")
    assert "row_is_fraud" not in split
    result = _load_json(RESULT)
    public_artifacts = result["public_artifacts"]
    assert isinstance(public_artifacts, dict)
    assert set(public_artifacts) == EXPECTED_ARTIFACTS
    for name, reference in public_artifacts.items():
        assert isinstance(name, str)
        assert isinstance(reference, dict)
        assert hashlib.sha256((FIXTURE_ROOT / name).read_bytes()).hexdigest() == reference[
            "sha256"
        ]
        assert len(base64.b64decode(PUBLICATION_AUTHORITY.public_key_base64)) == 32
