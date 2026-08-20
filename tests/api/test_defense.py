"""Public Defend API privacy, integrity, isolation, and durability contracts."""

from __future__ import annotations

import base64
import copy
import inspect
import json
import multiprocessing
import os
import resource
import socket
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from apar.api.app import create_app
from apar.config import Settings
from apar.evaluation.defender_attestation import DefenderBundleVerifier
from apar.evaluation.publication_inputs import publish_corpus_attestation, verify_evaluation_inputs
from apar.evaluation.reporting import PublicArtifactVerifier
from apar.evaluation.service import (
    MAX_EXECUTOR_CAPABILITY_BYTES,
    EvaluationExecutor,
    _apply_child_limits,
    _close_inherited_fds,
    _index_payload,
    _IndexRecord,
    _input_key,
    _install_child_audit_hook,
    _parse_index_payload,
)
from apar.evaluation.v2_preregistration import ExecutionReceipt
from apar.evaluation.v2_reporting import (
    DefenseV2GateReport,
    V2ArmScorecard,
    not_executed_result,
    render_v2_scorecard,
)
from apar.evaluation.v2_selection import V2GateOutcome
from apar.runs import RunSigningIdentity
from apar.runs.wire import canonical_json_bytes
from apar.storage.artifacts import ArtifactStore
from tests.evaluation.test_replay import (
    _corpus_evidence,
    _selection_binding,
    _thresholds,
)
from tests.evaluation.test_reporting import (
    EVALUATOR_SIGNER,
    EVALUATOR_VERIFIER,
    FORBIDDEN_PUBLIC_TOKENS,
    HIDDEN_VERIFIER,
    _request,
)

pytest_plugins = ("tests.defense.test_bundle",)


WORKER_SOURCE = Path(__file__).parents[1] / "fixtures" / "defense_evaluator_worker.py"


def _probe_closed_inherited_fd(connection) -> None:
    descriptor = os.open(os.devnull, os.O_RDONLY)
    try:
        os.dup2(descriptor, 100)
    finally:
        os.close(descriptor)
    _close_inherited_fds(connection.fileno())
    try:
        os.fstat(100)
    except OSError:
        connection.send_bytes(b"closed")
    else:
        connection.send_bytes(b"open")
    finally:
        connection.close()


def _probe_python_audit_denials(connection) -> None:
    _install_child_audit_hook()
    denied: list[bool] = []
    for operation in (
        lambda: socket.socket(),
        lambda: os.system("true"),
        lambda: os.open(os.devnull, os.O_RDONLY),
    ):
        try:
            operation()
        except PermissionError:
            denied.append(True)
        else:
            denied.append(False)
    connection.send_bytes(json.dumps(denied).encode("ascii"))
    connection.close()


@dataclass(frozen=True, slots=True)
class _ExecutionTracker:
    marker: Path
    timeout_seconds: float
    capability: object
    initial_calls: int = 0

    @property
    def process_ids(self) -> tuple[int, ...]:
        if not self.marker.exists():
            return ()
        return tuple(int(value) for value in self.marker.read_text(encoding="ascii").splitlines())

    @property
    def calls(self) -> int:
        return max(0, len(self.process_ids) - self.initial_calls)


def _client(
    tmp_path: Path,
    bundle_fixture,
    *,
    marker_name: str = "defense-executor.calls",
    timeout_seconds: float = 900.0,
    delay_seconds: float = 0.0,
):
    evidence = _corpus_evidence(bundle_fixture)
    kwargs = {
        **bundle_fixture.kwargs,
        "lineage": bundle_fixture.kwargs["lineage"].model_copy(
            update={"corpus_digest": evidence.corpus_digest}
        ),
    }
    _manifest, defender_ref = bundle_fixture.publisher.freeze(**kwargs)
    _attestation, corpus_ref = publish_corpus_attestation(
        evidence,
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
    loaded = bundle_fixture.publisher.load(defender_ref)
    threshold_set = _thresholds(bundle_fixture, loaded, _selection_binding(loaded))
    request, _decision = _request(
        defender_digest=defender_ref.sha256,
        defender_bundle_id=verified.defender.bundle_id,
        corpus_digest=verified.corpus.corpus_digest,
        split_digest=verified.corpus.split_digest,
        threshold_set=threshold_set,
    )
    marker = tmp_path / marker_name
    initial_calls = len(marker.read_text(encoding="ascii").splitlines()) if marker.exists() else 0
    executor = EvaluationExecutor.from_signed_source(
        source_path=WORKER_SOURCE,
        callable_qualname="evaluate",
        version="1.0.0",
        config={
            "delay_iterations": 100_000_000 if delay_seconds else 0,
            "request_base64": base64.b64encode(request.to_worker_json()).decode("ascii"),
        },
        signer=EVALUATOR_SIGNER,
        timeout_seconds=timeout_seconds,
        execution_receipt_path=marker,
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
        _ExecutionTracker(marker, executor.timeout_seconds, executor, initial_calls),
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
    executor = EvaluationExecutor.from_signed_source(
        source_path=WORKER_SOURCE,
        callable_qualname="evaluate",
        version="1.0.0",
        config={
            "delay_iterations": 0,
            "request_base64": base64.b64encode(request.to_worker_json()).decode("ascii"),
        },
        signer=EVALUATOR_SIGNER,
        execution_receipt_path=tmp_path / "sealed.calls",
    )
    assert not hasattr(EvaluationExecutor, "from_worker")
    assert repr(executor) == "<sealed evaluator worker capability>"
    with pytest.raises(TypeError):
        executor.__init__()
    with pytest.raises(TypeError):
        executor.timeout_seconds = 1.0  # type: ignore[misc]
    with pytest.raises(TypeError):
        copy.deepcopy(executor)
    with pytest.raises(TypeError):
        EvaluationExecutor.from_signed_source = classmethod(lambda *args: None)  # type: ignore[misc]

    forbidden_source = tmp_path / "forbidden_worker.py"
    forbidden_source.write_text(
        "import socket\ndef evaluate(inputs, config):\n    return socket.socket()\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="forbidden capability"):
        EvaluationExecutor.from_signed_source(
            source_path=forbidden_source,
            callable_qualname="evaluate",
            version="1.0.0",
            config={"fixture": True},
            signer=EVALUATOR_SIGNER,
        )
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


def test_executor_capability_is_self_contained_and_service_ignores_class_dispatch(
    tmp_path: Path, bundle_fixture
) -> None:
    client, _tracker, request, corpus_ref, defender_ref, _settings = _client(
        tmp_path, bundle_fixture
    )
    factory = EvaluationExecutor.from_signed_source.__func__
    closure_values = inspect.getclosurevars(factory).nonlocals.values()
    assert isinstance(_tracker.capability, bytes)
    assert not any(type(value).__name__ == "WeakKeyDictionary" for value in closure_values)

    called = False

    def replaced_execute(*args: object, **kwargs: object):
        nonlocal called
        del args, kwargs
        called = True
        return request

    original = EvaluationExecutor.__dict__.get("execute")
    type.__setattr__(EvaluationExecutor, "execute", replaced_execute)
    try:
        response = client.post(
            "/api/v1/defense/evaluations",
            json={
                "corpus_artifact_digest": corpus_ref.sha256,
                "defender_artifact_digest": defender_ref.sha256,
            },
        )
        assert response.status_code == 201
        assert called is False
    finally:
        if original is None:
            type.__delattr__(EvaluationExecutor, "execute")
        else:
            type.__setattr__(EvaluationExecutor, "execute", original)
        client.__exit__(None, None, None)


def test_executor_factory_enforces_final_canonical_capability_size() -> None:
    accepted = EvaluationExecutor.from_signed_source(
        source_path=WORKER_SOURCE,
        callable_qualname="evaluate",
        version="1.0.0",
        config={"padding": "x" * 3_143_849},
        signer=EVALUATOR_SIGNER,
        execution_receipt_path=Path("/a/bc"),
    )
    assert len(accepted) == MAX_EXECUTOR_CAPABILITY_BYTES

    with pytest.raises(ValueError, match="capability exceeds its cap"):
        EvaluationExecutor.from_signed_source(
            source_path=WORKER_SOURCE,
            callable_qualname="evaluate",
            version="1.0.0",
            config={"padding": "x" * 3_143_849},
            signer=EVALUATOR_SIGNER,
            execution_receipt_path=Path("/a/bcd"),
        )


def test_executor_factory_rejects_reviewer_four_million_character_config() -> None:
    with pytest.raises(ValueError, match="capability exceeds its cap"):
        EvaluationExecutor.from_signed_source(
            source_path=WORKER_SOURCE,
            callable_qualname="evaluate",
            version="1.0.0",
            config={"padding": "x" * 4_000_000},
            signer=EVALUATOR_SIGNER,
        )


@pytest.mark.parametrize(
    "source",
    (
        "import _socket\ndef evaluate(inputs, config):\n    return _socket.socket()\n",
        "import os\ndef evaluate(inputs, config):\n    return os.system('true')\n",
        "def evaluate(inputs, config):\n    return open('/tmp/escape', 'wb')\n",
        "import ctypes\ndef evaluate(inputs, config):\n    return config['request_base64']\n",
    ),
)
def test_worker_source_rejects_prebound_network_process_and_file_capabilities(
    tmp_path: Path, source: str
) -> None:
    path = tmp_path / "unsafe_worker.py"
    path.write_text(source, encoding="utf-8")
    with pytest.raises(ValueError, match="forbidden|prebound|file"):
        EvaluationExecutor.from_signed_source(
            source_path=path,
            callable_qualname="evaluate",
            version="1.0.0",
            config={"request_base64": "AA=="},
            signer=EVALUATOR_SIGNER,
        )


def test_required_child_rlimit_failure_is_not_suppressed(monkeypatch) -> None:
    def fail(_kind: int, _limits: tuple[int, int]) -> None:
        raise OSError("setrlimit denied")

    monkeypatch.setattr(resource, "setrlimit", fail)
    with pytest.raises(OSError, match="setrlimit denied"):
        _apply_child_limits()


def test_child_closes_inherited_fd_100_before_worker_execution() -> None:
    context = multiprocessing.get_context("spawn")
    parent, child = context.Pipe(duplex=False)
    process = context.Process(target=_probe_closed_inherited_fd, args=(child,))
    process.start()
    child.close()
    assert parent.poll(5)
    assert parent.recv_bytes() == b"closed"
    process.join(timeout=5)
    assert process.exitcode == 0
    parent.close()


def test_python_audit_hook_denies_prebound_network_process_and_open() -> None:
    context = multiprocessing.get_context("spawn")
    parent, child = context.Pipe(duplex=False)
    process = context.Process(target=_probe_python_audit_denials, args=(child,))
    process.start()
    child.close()
    assert parent.poll(5)
    assert json.loads(parent.recv_bytes()) == [True, True, True]
    process.join(timeout=5)
    assert process.exitcode == 0
    parent.close()


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
        tmp_path, bundle_fixture, marker_name="first.calls"
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

    restarted, _, _, _, _, _ = _client(tmp_path, bundle_fixture)
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
        assert tracker.calls <= 1
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

    restarted, tracker, _, _, _, _ = _client(tmp_path, bundle_fixture)
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


def test_index_rejects_duplicate_evaluation_id_for_a_different_input_pair(
    tmp_path: Path, bundle_fixture
) -> None:
    """The durable index is a bijection, not a first-match multimap."""
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
    finally:
        client.__exit__(None, None, None)
    index_root = settings.artifact_root / ".defense-evaluation-index-v1"
    pointer = next(path for path in index_root.iterdir() if path.name != "index.lock")
    verifier = PublicArtifactVerifier.from_signer(bundle_fixture.signer)
    original = _parse_index_payload(pointer.read_bytes(), verifier)
    other_corpus = "a" * 64
    duplicate = _IndexRecord(
        _input_key(other_corpus, defender_ref.sha256),
        other_corpus,
        defender_ref.sha256,
        original.evaluation_id,
        original.bundle_ref,
        original.receipt_ref,
    )
    duplicate_path = index_root / f"{duplicate.input_key}.json"
    duplicate_path.write_bytes(_index_payload(duplicate, bundle_fixture.signer))
    duplicate_path.chmod(0o600)

    restarted, _, _, _, _, _ = _client(tmp_path, bundle_fixture)
    try:
        fetched = restarted.get(
            f"/api/v1/defense/evaluations/{created.json()['evaluation_id']}"
        )
        assert fetched.status_code == 422
        assert fetched.json()["detail"]["code"] == "DEFENSE_ARTIFACT_INVALID"
    finally:
        restarted.__exit__(None, None, None)


def test_every_get_reauthenticates_original_corpus_and_defender_inputs(
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
        (settings.artifact_root / corpus_ref.relative_path).write_bytes(b"tampered")
        fetched = client.get(f"/api/v1/defense/evaluations/{created.json()['evaluation_id']}")
        assert fetched.status_code == 422
        assert fetched.json()["detail"]["code"] == "DEFENSE_ARTIFACT_INVALID"
    finally:
        client.__exit__(None, None, None)


def test_get_validates_only_the_exact_selected_record(tmp_path: Path, bundle_fixture) -> None:
    """GET scans signed metadata globally but fully loads only its unique target."""
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
        index_root = settings.artifact_root / ".defense-evaluation-index-v1"
        pointer = next(path for path in index_root.iterdir() if path.name != "index.lock")
        verifier = PublicArtifactVerifier.from_signer(bundle_fixture.signer)
        original = _parse_index_payload(pointer.read_bytes(), verifier)
        other_corpus = "a" * 64
        other_defender = "b" * 64
        unselected = _IndexRecord(
            _input_key(other_corpus, other_defender),
            other_corpus,
            other_defender,
            "c" * 64,
            original.bundle_ref,
            original.receipt_ref,
        )
        other_path = index_root / f"{unselected.input_key}.json"
        other_path.write_bytes(_index_payload(unselected, bundle_fixture.signer))
        other_path.chmod(0o600)

        fetched = client.get(f"/api/v1/defense/evaluations/{created.json()['evaluation_id']}")
        assert fetched.status_code == 200
        assert fetched.content == created.content
    finally:
        client.__exit__(None, None, None)


def test_v2_scorecard_exposes_only_public_not_executed_contract(tmp_path: Path) -> None:
    """The public v2 read does not need a configured evaluation executor."""
    with TestClient(create_app(Settings.from_root(tmp_path))) as client:
        response = client.get("/defense/v2/scorecard")

    assert response.status_code == 200
    assert response.json()["status"] == "not_executed"
    assert "hidden_seed" not in response.text


def test_v2_scorecard_reads_verified_current_state_after_receipt(tmp_path: Path) -> None:
    """A durable receipt makes the verified signed result the only current status."""
    state = tmp_path / ".apar/defense-v2"
    state.mkdir(parents=True)
    signer = RunSigningIdentity.from_private_bytes(b"s" * 32)
    arms = tuple(
        V2ArmScorecard(
            arm=arm,
            status="no_promotion",
            gate=DefenseV2GateReport(
                arm=arm,
                outcome=V2GateOutcome(passed=False, codes=("CONTROL_INVALID",)),
            ),
        )
        for arm in ("rules_only", "gbdt_only", "layered_hybrid")
    )
    result = not_executed_result().model_copy(update={"status": "no_promotion", "arms": arms})
    card, _ = render_v2_scorecard(result, signer=signer)
    (state / "defense-v2-scorecard.json").write_bytes(card.to_json())
    receipt = ExecutionReceipt(
        preregistration_id="apar-defend-v2",
        execution_nonce="receipt-present",
    )
    (state / "execution-receipt.json").write_bytes(
        canonical_json_bytes(receipt.model_dump(mode="json"))
    )

    with TestClient(create_app(Settings.from_root(tmp_path))) as client:
        response = client.get("/defense/v2/scorecard")

    assert response.status_code == 200
    assert response.json() == json.loads(card.to_json())
    assert response.json()["status"] == "no_promotion"


def test_v2_scorecard_fails_closed_when_receipt_has_no_signed_result(tmp_path: Path) -> None:
    """A consumed execution cannot be hidden behind the initial compiled-in status."""
    state = tmp_path / ".apar/defense-v2"
    state.mkdir(parents=True)
    receipt = ExecutionReceipt(
        preregistration_id="apar-defend-v2",
        execution_nonce="receipt-present",
    )
    (state / "execution-receipt.json").write_bytes(
        canonical_json_bytes(receipt.model_dump(mode="json"))
    )

    with TestClient(create_app(Settings.from_root(tmp_path))) as client:
        response = client.get("/defense/v2/scorecard")

    assert response.status_code == 422
