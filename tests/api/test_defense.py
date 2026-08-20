"""Public Defend API privacy, integrity, and idempotency contracts."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from fastapi.testclient import TestClient

from apar.api.app import create_app
from apar.config import Settings
from apar.evaluation.gates import EvaluatorReplayVerifier, EvaluatorSigningIdentity
from apar.evaluation.service import EvaluationExecutor
from apar.storage.artifacts import ArtifactStore
from tests.evaluation.test_reporting import FORBIDDEN_PUBLIC_TOKENS, _request


class _FixtureExecutor(EvaluationExecutor):
    def __init__(self, request) -> None:
        self.request = request
        self.calls = 0
        self.timeouts: list[float] = []

    def execute(self, *, corpus_ref, defender_ref, timeout_seconds):
        self.calls += 1
        self.timeouts.append(timeout_seconds)
        assert corpus_ref.sha256 == self.request.corpus_artifact_digest
        assert defender_ref.sha256 == self.request.defender_artifact_digest
        return self.request


def _client(tmp_path: Path):
    settings = Settings.from_root(tmp_path)
    store = ArtifactStore(settings.artifact_root)
    corpus_ref = store.put_bytes(
        b'{"schema_version":"1.0.0","synthetic_only":true}',
        "application/vnd.apar.frozen-corpus+json",
    )
    defender_ref = store.put_bytes(
        b'{"kind":"fixture-defender","schema_version":"1.0.0"}',
        "application/vnd.apar.defender-bundle+json",
    )
    evaluator_signer = EvaluatorSigningIdentity.from_private_bytes(b"e" * 32)
    evaluator_verifier = EvaluatorReplayVerifier.from_signer(evaluator_signer)
    request, _ = _request(
        corpus_digest=corpus_ref.sha256,
        defender_digest=defender_ref.sha256,
        evaluator_signer=evaluator_signer,
        evaluator_verifier=evaluator_verifier,
    )
    executor = _FixtureExecutor(request)
    app = create_app(
        settings,
        defense_executor=executor,
        evaluator_verifier=evaluator_verifier,
        hidden_proof_verifier=evaluator_verifier,
    )
    client = TestClient(app)
    client.__enter__()
    return client, executor, request, corpus_ref, defender_ref


def test_create_get_and_artifact_routes_are_atomic_idempotent_and_public(
    tmp_path: Path,
) -> None:
    """Catch duplicate execution, partial reads, or privacy leakage on the golden API path."""
    client, executor, _, corpus_ref, defender_ref = _client(tmp_path)
    try:
        body = {
            "corpus_artifact_digest": corpus_ref.sha256,
            "defender_artifact_digest": defender_ref.sha256,
        }
        with ThreadPoolExecutor(max_workers=4) as pool:
            responses = tuple(
                pool.map(
                    lambda _: client.post("/api/v1/defense/evaluations", json=body),
                    range(4),
                )
            )
        created = responses[0]
        assert {response.status_code for response in responses} == {201}
        assert {response.content for response in responses} == {created.content}
        assert executor.calls == 1
        assert executor.timeouts == [900.0]
        scorecard = created.json()
        fetched = client.get(
            f"/api/v1/defense/evaluations/{scorecard['evaluation_id']}"
        )
        assert fetched.status_code == 200
        assert fetched.content == created.content
        payload = json.dumps(scorecard, sort_keys=True).lower()
        for token in FORBIDDEN_PUBLIC_TOKENS:
            assert token not in payload

        artifact = client.get(
            f"/api/v1/defense/evaluations/{scorecard['evaluation_id']}"
            "/artifacts/leaderboard.csv"
        )
        assert artifact.status_code == 200
        assert artifact.headers["content-type"].startswith("text/csv")
        assert artifact.headers["etag"].startswith('"')
        assert b"rules_only" in artifact.content
    finally:
        client.__exit__(None, None, None)


def test_defense_api_rejects_nonexact_requests_and_uniformly_hides_restricted_names(
    tmp_path: Path,
) -> None:
    """Catch coercion, traversal aliases, or restricted-artifact existence disclosure."""
    client, _, _, corpus_ref, defender_ref = _client(tmp_path)
    try:
        valid = {
            "corpus_artifact_digest": corpus_ref.sha256,
            "defender_artifact_digest": defender_ref.sha256,
        }
        assert client.post(
            "/api/v1/defense/evaluations", json={**valid, "extra": True}
        ).status_code == 422
        assert client.post(
            "/api/v1/defense/evaluations",
            json={**valid, "corpus_artifact_digest": corpus_ref.sha256.upper()},
        ).status_code == 422
        created = client.post("/api/v1/defense/evaluations", json=valid).json()
        base = f"/api/v1/defense/evaluations/{created['evaluation_id']}/artifacts"
        guesses = (
            "evaluation_truth",
            "per_decision_predictions",
            "restricted_hidden",
            "hidden-public-proof.json",
            "..%2Fleaderboard.csv",
            "%2e%2e%2fleaderboard.csv",
            "LEADERBOARD.CSV",
            "leaderboard%2Ecsv",
        )
        responses = [client.get(f"{base}/{name}") for name in guesses]
        assert {response.status_code for response in responses} == {404}
        assert {response.content for response in responses} == {
            b'{"detail":{"code":"DEFENSE_ARTIFACT_NOT_FOUND",'
            b'"message":"public artifact not found"}}'
        }
    finally:
        client.__exit__(None, None, None)


def test_missing_invalid_and_unavailable_evaluations_have_typed_nonleaking_errors(
    tmp_path: Path,
) -> None:
    """Catch storage/executor exceptions escaping through the public boundary."""
    client, _, _, _, _ = _client(tmp_path)
    try:
        missing = client.post(
            "/api/v1/defense/evaluations",
            json={
                "corpus_artifact_digest": "0" * 64,
                "defender_artifact_digest": "1" * 64,
            },
        )
        unknown = client.get("/api/v1/defense/evaluations/" + "0" * 64)
        assert missing.status_code == unknown.status_code == 404
        for response in (missing, unknown):
            assert "/" not in response.json()["detail"]["message"]
            assert "trace" not in response.text.lower()
    finally:
        client.__exit__(None, None, None)


def test_service_rejects_a_non_synthetic_corpus_before_executor_access(
    tmp_path: Path,
) -> None:
    """Catch public-data or live-data references crossing the synthetic-only boundary."""
    client, executor, _, _, defender_ref = _client(tmp_path)
    try:
        store = ArtifactStore(tmp_path / ".apar" / "artifacts")
        unsafe = store.put_bytes(
            b'{"schema_version":"1.0.0","synthetic_only":false}',
            "application/vnd.apar.frozen-corpus+json",
        )
        response = client.post(
            "/api/v1/defense/evaluations",
            json={
                "corpus_artifact_digest": unsafe.sha256,
                "defender_artifact_digest": defender_ref.sha256,
            },
        )
        assert response.status_code == 422
        assert executor.calls == 0
    finally:
        client.__exit__(None, None, None)


def test_every_get_revalidates_all_content_addressed_public_artifacts(
    tmp_path: Path,
) -> None:
    """Catch cached scorecards that stay readable after public artifact corruption."""
    client, _, _, corpus_ref, defender_ref = _client(tmp_path)
    try:
        created = client.post(
            "/api/v1/defense/evaluations",
            json={
                "corpus_artifact_digest": corpus_ref.sha256,
                "defender_artifact_digest": defender_ref.sha256,
            },
        ).json()
        leaderboard = next(
            item
            for item in created["public_artifacts"]["entries"]
            if item["name"] == "leaderboard.csv"
        )
        payload_path = (
            tmp_path / ".apar" / "artifacts" / leaderboard["sha256"] / "payload"
        )
        payload_path.write_bytes(b"tampered\n")

        fetched = client.get(
            f"/api/v1/defense/evaluations/{created['evaluation_id']}"
        )
        assert fetched.status_code == 422
        assert fetched.json() == {
            "detail": {
                "code": "DEFENSE_ARTIFACT_INVALID",
                "message": "published defense artifact failed validation",
            }
        }
    finally:
        client.__exit__(None, None, None)
