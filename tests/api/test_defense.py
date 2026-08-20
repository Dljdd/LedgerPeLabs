"""Public Defend API privacy, integrity, isolation, and durability contracts."""

from __future__ import annotations

import copy
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from apar.api.app import create_app
from apar.config import Settings
from apar.evaluation.defender_attestation import DefenderBundleVerifier
from apar.evaluation.publication_inputs import (
    VerifiedEvaluationInputs,
    publish_corpus_attestation,
    verify_evaluation_inputs,
)
from apar.evaluation.reporting import PublicArtifactVerifier, ScorecardPublicationRequest
from apar.evaluation.service import EvaluationExecutor
from apar.storage.artifacts import ArtifactStore
from tests.evaluation.test_replay import _corpus_evidence
from tests.evaluation.test_reporting import (
    EVALUATOR_SIGNER,
    EVALUATOR_VERIFIER,
    FORBIDDEN_PUBLIC_TOKENS,
    HIDDEN_VERIFIER,
    _request,
)

pytest_plugins = ("tests.defense.test_bundle",)


@dataclass(frozen=True, slots=True)
class _FixtureWorker:
    payload: bytes
    marker_path: str
    delay_seconds: float = 0.0

    def __call__(self, inputs: VerifiedEvaluationInputs) -> ScorecardPublicationRequest:
        assert type(inputs) is VerifiedEvaluationInputs
        descriptor = os.open(
            self.marker_path,
            os.O_WRONLY | os.O_CREAT | os.O_APPEND | os.O_NOFOLLOW,
            0o600,
        )
        try:
            os.write(descriptor, f"{os.getpid()}\n".encode("ascii"))
        finally:
            os.close(descriptor)
        if self.delay_seconds:
            time.sleep(self.delay_seconds)
        return ScorecardPublicationRequest.from_worker_json(self.payload)


@dataclass(frozen=True, slots=True)
class _ExecutionTracker:
    marker: Path
    timeout_seconds: float

    @property
    def process_ids(self) -> tuple[int, ...]:
        if not self.marker.exists():
            return ()
        return tuple(int(value) for value in self.marker.read_text(encoding="ascii").splitlines())

    @property
    def calls(self) -> int:
        return len(self.process_ids)


def _client(
    tmp_path: Path,
    bundle_fixture,
    *,
    marker_name: str = "defense-executor.calls",
    timeout_seconds: float = 900.0,
    delay_seconds: float = 0.0,
):
    _manifest, defender_ref = bundle_fixture.publisher.freeze(**bundle_fixture.kwargs)
    _attestation, corpus_ref = publish_corpus_attestation(
        _corpus_evidence(bundle_fixture),
        artifact_store=bundle_fixture.store,
        signer=EVALUATOR_SIGNER,
    )
    defender_verifier = DefenderBundleVerifier(
        bundle_fixture.store,
        signer_key_id=bundle_fixture.signer.key_id,
        public_key_base64=bundle_fixture.signer.public_key_base64,
    )
    verified = verify_evaluation_inputs(
        corpus_ref=corpus_ref,
        defender_ref=defender_ref,
        artifact_store=bundle_fixture.store,
        evaluator_verifier=EVALUATOR_VERIFIER,
        defender_verifier=defender_verifier,
    )
    request, _decision = _request(
        defender_digest=defender_ref.sha256,
        defender_bundle_id=verified.defender.bundle_id,
        split_digest=verified.corpus.split_digest,
    )
    marker = tmp_path / marker_name
    executor = EvaluationExecutor.from_worker(
        _FixtureWorker(request.to_worker_json(), str(marker), delay_seconds),
        timeout_seconds=timeout_seconds,
    )
    settings = Settings(
        root=tmp_path.resolve(),
        database_path=tmp_path / ".apar" / "state.db",
        artifact_root=bundle_fixture.store.validated_worker_root(),
    )
    publication_verifier = PublicArtifactVerifier.from_signer(bundle_fixture.signer)
    app = create_app(
        settings,
        defense_executor=executor,
        evaluator_verifier=EVALUATOR_VERIFIER,
        hidden_proof_verifier=HIDDEN_VERIFIER,
        publication_signer=bundle_fixture.signer,
        publication_verifier=publication_verifier,
        defender_signer_key_id=bundle_fixture.signer.key_id,
        defender_public_key_base64=bundle_fixture.signer.public_key_base64,
    )
    client = TestClient(app)
    client.__enter__()
    return (
        client,
        _ExecutionTracker(marker, executor.timeout_seconds),
        request,
        corpus_ref,
        defender_ref,
        settings,
    )


def test_create_get_and_artifact_routes_are_atomic_idempotent_and_public(
    tmp_path: Path, bundle_fixture
) -> None:
    """Catch duplicate execution, partial reads, or privacy leakage on the golden path."""
    client, tracker, _, corpus_ref, defender_ref, _settings = _client(tmp_path, bundle_fixture)
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
        assert tracker.calls == 1
        assert tracker.timeout_seconds == 900.0
        assert tracker.process_ids != (os.getpid(),)
        scorecard = created.json()
        fetched = client.get(f"/api/v1/defense/evaluations/{scorecard['evaluation_id']}")
        assert fetched.status_code == 200
        assert fetched.content == created.content
        payload = json.dumps(scorecard, sort_keys=True).lower()
        for token in FORBIDDEN_PUBLIC_TOKENS:
            assert token not in payload

        artifact = client.get(
            f"/api/v1/defense/evaluations/{scorecard['evaluation_id']}/artifacts/leaderboard.csv"
        )
        assert artifact.status_code == 200
        assert artifact.headers["content-type"].startswith("text/csv")
        assert artifact.headers["etag"].startswith('"')
        assert b"rules_only" in artifact.content
    finally:
        client.__exit__(None, None, None)


def test_defense_api_rejects_nonexact_requests_and_hides_restricted_names(
    tmp_path: Path, bundle_fixture
) -> None:
    client, _, _, corpus_ref, defender_ref, _settings = _client(tmp_path, bundle_fixture)
    try:
        valid = {
            "corpus_artifact_digest": corpus_ref.sha256,
            "defender_artifact_digest": defender_ref.sha256,
        }
        assert (
            client.post("/api/v1/defense/evaluations", json={**valid, "extra": True}).status_code
            == 422
        )
        assert (
            client.post(
                "/api/v1/defense/evaluations",
                json={**valid, "corpus_artifact_digest": corpus_ref.sha256.upper()},
            ).status_code
            == 422
        )
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


def test_missing_inputs_and_evaluations_have_typed_nonleaking_errors(
    tmp_path: Path, bundle_fixture
) -> None:
    client, _, _, _, _, _ = _client(tmp_path, bundle_fixture)
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


def test_unauthenticated_inputs_are_rejected_before_executor(
    tmp_path: Path, bundle_fixture
) -> None:
    client, tracker, _, _corpus_ref, _defender_ref, settings = _client(tmp_path, bundle_fixture)
    try:
        store = ArtifactStore(settings.artifact_root)
        unsafe = store.put_bytes(
            b'{"schema_version":"1.0.0","synthetic_only":false}',
            "application/vnd.apar.frozen-corpus+json",
        )
        fake_defender = store.put_bytes(
            b'{"kind":"fixture-defender","schema_version":"1.0.0"}',
            "application/vnd.apar.defender-bundle+json",
        )
        responses = (
            client.post(
                "/api/v1/defense/evaluations",
                json={
                    "corpus_artifact_digest": unsafe.sha256,
                    "defender_artifact_digest": _defender_ref.sha256,
                },
            ),
            client.post(
                "/api/v1/defense/evaluations",
                json={
                    "corpus_artifact_digest": _corpus_ref.sha256,
                    "defender_artifact_digest": fake_defender.sha256,
                },
            ),
        )
        assert {response.status_code for response in responses} == {422}
        assert tracker.calls == 0
    finally:
        client.__exit__(None, None, None)


def test_defense_routes_are_503_without_three_independent_pinned_identities(
    tmp_path: Path,
) -> None:
    with TestClient(create_app(Settings.from_root(tmp_path))) as client:
        response = client.post(
            "/api/v1/defense/evaluations",
            json={
                "corpus_artifact_digest": "0" * 64,
                "defender_artifact_digest": "1" * 64,
            },
        )
        health = client.get("/api/v1/health")
    assert response.status_code == 503
    assert health.status_code == 200


def test_executor_and_trust_capabilities_are_exact_sealed_and_independent(
    tmp_path: Path, bundle_fixture
) -> None:
    client, _tracker, request, _, _, settings = _client(tmp_path, bundle_fixture)
    client.__exit__(None, None, None)
    worker = _FixtureWorker(request.to_worker_json(), str(tmp_path / "sealed.calls"))
    executor = EvaluationExecutor.from_worker(worker)
    with pytest.raises(TypeError):
        executor.__init__()
    with pytest.raises(TypeError):
        executor.timeout_seconds = 1.0  # type: ignore[misc]
    with pytest.raises(TypeError):
        copy.deepcopy(executor)
    publication_verifier = PublicArtifactVerifier.from_signer(bundle_fixture.signer)
    with pytest.raises(TypeError, match="independent"):
        create_app(
            settings,
            defense_executor=executor,
            evaluator_verifier=EVALUATOR_VERIFIER,
            hidden_proof_verifier=EVALUATOR_VERIFIER,
            publication_signer=bundle_fixture.signer,
            publication_verifier=publication_verifier,
            defender_signer_key_id=bundle_fixture.signer.key_id,
            defender_public_key_base64=bundle_fixture.signer.public_key_base64,
        )


def test_defense_post_body_is_capped_before_parser_or_executor(
    tmp_path: Path, bundle_fixture
) -> None:
    client, tracker, _, _, _, _ = _client(tmp_path, bundle_fixture)
    try:
        response = client.post(
            "/api/v1/defense/evaluations",
            content=(chunk for chunk in (b"{", b"x" * 100_000, b"}")),
            headers={"content-type": "application/json"},
        )
        assert response.status_code == 413
        assert response.json()["detail"]["code"] == "DEFENSE_REQUEST_TOO_LARGE"
        assert tracker.calls == 0
    finally:
        client.__exit__(None, None, None)


def test_signed_evaluation_index_survives_service_restart(tmp_path: Path, bundle_fixture) -> None:
    client, tracker, _, corpus_ref, defender_ref, _settings = _client(
        tmp_path, bundle_fixture, marker_name="first.calls"
    )
    body = {
        "corpus_artifact_digest": corpus_ref.sha256,
        "defender_artifact_digest": defender_ref.sha256,
    }
    try:
        created = client.post("/api/v1/defense/evaluations", json=body)
        assert created.status_code == 201
        evaluation_id = created.json()["evaluation_id"]
    finally:
        client.__exit__(None, None, None)

    restarted, restarted_tracker, _, _, _, _ = _client(
        tmp_path, bundle_fixture, marker_name="second.calls"
    )
    try:
        fetched = restarted.get(f"/api/v1/defense/evaluations/{evaluation_id}")
        repeated = restarted.post("/api/v1/defense/evaluations", json=body)
        assert fetched.status_code == 200
        assert repeated.status_code == 201
        assert fetched.content == created.content == repeated.content
        assert restarted_tracker.calls == 0
        assert tracker.calls == 1
    finally:
        restarted.__exit__(None, None, None)


def test_restart_revalidates_signed_index_without_breaking_health(
    tmp_path: Path, bundle_fixture
) -> None:
    client, _, _, corpus_ref, defender_ref, settings = _client(tmp_path, bundle_fixture)
    try:
        created = client.post(
            "/api/v1/defense/evaluations",
            json={
                "corpus_artifact_digest": corpus_ref.sha256,
                "defender_artifact_digest": defender_ref.sha256,
            },
        )
        assert created.status_code == 201
        evaluation_id = created.json()["evaluation_id"]
    finally:
        client.__exit__(None, None, None)
    index_root = settings.artifact_root / ".defense-evaluation-index-v1"
    pointer = next(path for path in index_root.iterdir() if path.name != "index.lock")
    pointer.write_bytes(b"{}")

    restarted, _, _, _, _, _ = _client(tmp_path, bundle_fixture, marker_name="invalid-index.calls")
    try:
        fetched = restarted.get(f"/api/v1/defense/evaluations/{evaluation_id}")
        health = restarted.get("/api/v1/health")
        assert fetched.status_code == 422
        assert fetched.json()["detail"]["code"] == "DEFENSE_ARTIFACT_INVALID"
        assert health.status_code == 200
    finally:
        restarted.__exit__(None, None, None)


def test_timed_out_executor_cannot_publish_a_pointer(tmp_path: Path, bundle_fixture) -> None:
    client, tracker, _, corpus_ref, defender_ref, _settings = _client(
        tmp_path,
        bundle_fixture,
        timeout_seconds=0.05,
        delay_seconds=1.0,
    )
    try:
        response = client.post(
            "/api/v1/defense/evaluations",
            json={
                "corpus_artifact_digest": corpus_ref.sha256,
                "defender_artifact_digest": defender_ref.sha256,
            },
        )
        assert response.status_code == 409
        assert tracker.calls == 1
        index_root = bundle_fixture.store.validated_worker_root()
        pointer_names = tuple(
            name.name
            for name in (index_root / ".defense-evaluation-index-v1").iterdir()
            if name.name != "index.lock"
        )
        assert pointer_names == ()
    finally:
        client.__exit__(None, None, None)


def test_interrupted_index_temporary_is_never_a_visible_evaluation(
    tmp_path: Path, bundle_fixture
) -> None:
    client, _, _, _, _, settings = _client(tmp_path, bundle_fixture)
    client.__exit__(None, None, None)
    index_root = settings.artifact_root / ".defense-evaluation-index-v1"
    temporary = index_root / (".tmp-1-1-" + "a" * 64)
    temporary.write_bytes(b"partial")
    temporary.chmod(0o600)

    restarted, tracker, _, _, _, _ = _client(
        tmp_path, bundle_fixture, marker_name="crash-restart.calls"
    )
    try:
        unknown = restarted.get("/api/v1/defense/evaluations/" + "0" * 64)
        assert unknown.status_code == 404
        assert not temporary.exists()
        assert tracker.calls == 0
    finally:
        restarted.__exit__(None, None, None)


def test_every_get_revalidates_all_content_addressed_public_artifacts(
    tmp_path: Path, bundle_fixture
) -> None:
    client, _, _, corpus_ref, defender_ref, settings = _client(tmp_path, bundle_fixture)
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
        payload_path = settings.artifact_root / leaderboard["sha256"] / "payload"
        payload_path.write_bytes(b"tampered\n")

        fetched = client.get(f"/api/v1/defense/evaluations/{created['evaluation_id']}")
        assert fetched.status_code == 422
        assert fetched.json() == {
            "detail": {
                "code": "DEFENSE_ARTIFACT_INVALID",
                "message": "published defense artifact failed validation",
            }
        }
    finally:
        client.__exit__(None, None, None)
