"""Process-isolated policy execution and signed immutable run evidence."""

from __future__ import annotations

import base64
import copy
import hashlib
import json
import os
import stat
import subprocess
from concurrent.futures import ThreadPoolExecutor
from decimal import Decimal
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from pydantic import ValidationError

import apar.evaluation_hidden.validity as hidden_validity_module
import apar.runs.runner as runner_module
from apar.compiler import compile_scenario
from apar.contracts.decisions import Action
from apar.contracts.events import PaymentEvent, Rail
from apar.contracts.scenarios import AttackerMode, FeedbackField, ScenarioBundle
from apar.evaluation_hidden import HiddenCampaignGenerator, HiddenValidityOracle
from apar.generators import Population
from apar.redteam import Feedback, FixedPolicy, VisibleTrial, visible_objective
from apar.runs import (
    AttackerPolicy,
    AttackerPolicyKind,
    PolicyWorkerClient,
    PolicyWorkerError,
    RunExecutionError,
    RunRunner,
    RunSigningIdentity,
    SignedRunReceipt,
    bind_scenario_for_run,
)
from apar.runs.wire import history_to_wire, strict_json_loads
from apar.storage.artifacts import ArtifactStore
from tests.factories import make_scenario_config, make_threat_card
from tests.redteam.conftest import campaign_benchmark


def _signer() -> RunSigningIdentity:
    return RunSigningIdentity.from_private_bytes(bytes(range(32)))


def _bound_app_bundle(population: Population) -> ScenarioBundle:
    return bind_scenario_for_run(
        population.bundle,
        threat_family="app_scam_mule",
    )


def _resign_with_replaced_provenance(
    *,
    runner: RunRunner,
    store: ArtifactStore,
    signer: RunSigningIdentity,
    manifest: runner_module.RunManifest,
    provenance: dict[str, object],
) -> runner_module.RunManifest:
    provenance_ref = store.put_json(provenance)
    authorization = SignedRunReceipt.model_validate_json(
        store.read(manifest.artifacts["authorization_receipt"])
    )
    authorization_values = authorization.unsigned_document()
    authorization_digests = dict(authorization.artifact_digests)
    authorization_digests["provenance"] = provenance_ref.sha256
    authorization_values["artifact_digests"] = authorization_digests
    authorization_values["subject_sha256"] = runner_module._digest_document(
        dict(sorted(authorization_digests.items()))
    )
    forged_authorization = SignedRunReceipt.model_validate(
        {
            **authorization_values,
            "signature_base64": signer.sign(authorization_values),
        }
    )
    authorization_ref = store.put_json(forged_authorization)
    completion = SignedRunReceipt.model_validate_json(
        store.read(manifest.artifacts["completion_receipt"])
    )
    completion_values = completion.unsigned_document()
    completion_values["previous_receipt_sha256"] = authorization_ref.sha256
    forged_completion = SignedRunReceipt.model_validate(
        {
            **completion_values,
            "signature_base64": signer.sign(completion_values),
        }
    )
    artifacts = {
        **manifest.artifacts,
        "authorization_receipt": authorization_ref,
        "completion_receipt": store.put_json(forged_completion),
        "provenance": provenance_ref,
    }
    draft = manifest.model_copy(
        update={
            "artifacts": artifacts,
            "lineage_digest": runner._lineage_digest(artifacts),
        }
    )
    return draft.model_copy(
        update={"signature_base64": signer.sign(draft.unsigned_document())}
    )


def test_durable_signer_reuses_a_restrictive_non_symlink_identity(tmp_path: Path) -> None:
    """Catch ephemeral run signatures or a private key persisted with permissive mode."""
    key_path = tmp_path / "state" / "run-signing-key.ed25519"

    first = RunSigningIdentity.load_or_create(key_path)
    second = RunSigningIdentity.load_or_create(key_path)

    metadata = key_path.lstat()
    assert first.key_id == second.key_id
    assert first.public_key_base64 == second.public_key_base64
    assert stat.S_ISREG(metadata.st_mode)
    assert not key_path.is_symlink()
    assert stat.S_IMODE(metadata.st_mode) == 0o600
    assert metadata.st_size == 32


def test_durable_signer_completes_short_low_level_writes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Catch a short OS write silently persisting a truncated signing identity."""
    original_write = os.write

    def short_write(descriptor: int, content: bytes) -> int:
        return original_write(descriptor, content[:3])

    monkeypatch.setattr(os, "write", short_write)

    identity = RunSigningIdentity.load_or_create(tmp_path / "key.ed25519")

    assert len(base64.b64decode(identity.public_key_base64)) == 32
    assert (tmp_path / "key.ed25519").stat().st_size == 32


def test_durable_signer_never_publishes_a_partial_key_after_a_crash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Catch writing private key bytes directly into their final pathname."""
    key_path = tmp_path / "state" / "run-signing-key.ed25519"

    def interrupted_write(descriptor: int, content: bytes) -> None:
        os.write(descriptor, content[:3])
        raise OSError("simulated crash")

    monkeypatch.setattr(runner_module, "_write_all", interrupted_write)

    with pytest.raises(OSError, match="simulated crash"):
        RunSigningIdentity.load_or_create(key_path)

    assert not key_path.exists()
    assert list(key_path.parent.iterdir()) == []


def test_durable_signer_rejects_symlink_parent_and_hardlinked_key(tmp_path: Path) -> None:
    """Catch private signing material entering a redirected or shared namespace."""
    target = tmp_path / "target"
    target.mkdir(mode=0o700)
    linked_parent = tmp_path / "linked-state"
    linked_parent.symlink_to(target, target_is_directory=True)
    with pytest.raises(ValueError, match="parent"):
        RunSigningIdentity.load_or_create(linked_parent / "key.ed25519")

    key_path = tmp_path / "private" / "key.ed25519"
    RunSigningIdentity.load_or_create(key_path)
    os.link(key_path, tmp_path / "stolen-key-link")
    with pytest.raises(ValueError, match="link"):
        RunSigningIdentity.load_or_create(key_path)


def test_run_index_publication_is_private_atomic_and_crash_safe(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Catch direct final-path writes leaving a truncated append-only index entry."""
    index_root = tmp_path / "runs"
    runner = RunRunner(ArtifactStore(tmp_path / "artifacts"), _signer(), index_root)
    reference = ArtifactStore(tmp_path / "manifests").put_bytes(
        b"manifest", "application/json"
    )
    run_id = "run-0123456789abcdef0123456789abcdef"

    def interrupted_write(descriptor: int, content: bytes) -> None:
        os.write(descriptor, content[:3])
        raise OSError("simulated crash")

    monkeypatch.setattr(runner_module, "_write_all", interrupted_write)
    with pytest.raises(OSError, match="simulated crash"):
        runner._publish_index(run_id, reference)

    assert not (index_root / f"{run_id}.json").exists()
    assert list(index_root.glob(".*.tmp")) == []


def test_run_index_rejects_symlink_root_hardlinks_and_non_private_modes(
    tmp_path: Path,
) -> None:
    """Catch redirected roots and shared append-only index material."""
    target = tmp_path / "target"
    target.mkdir(mode=0o700)
    linked = tmp_path / "linked-runs"
    linked.symlink_to(target, target_is_directory=True)
    with pytest.raises(ValueError, match="run index root"):
        RunRunner(ArtifactStore(tmp_path / "artifacts-a"), _signer(), linked)

    index_root = tmp_path / "runs"
    runner = RunRunner(ArtifactStore(tmp_path / "artifacts-b"), _signer(), index_root)
    reference = runner.artifact_store.put_bytes(b"not-a-manifest", "application/json")
    run_id = "run-0123456789abcdef0123456789abcdef"
    runner._publish_index(run_id, reference)
    index_path = index_root / f"{run_id}.json"
    metadata = index_path.stat()
    assert stat.S_IMODE(metadata.st_mode) == 0o600
    assert metadata.st_nlink == 1
    os.link(index_path, tmp_path / "second-index-name")

    with pytest.raises(RunExecutionError, match="run index entry"):
        runner.get(run_id)
    (tmp_path / "second-index-name").unlink()
    index_path.chmod(0o644)
    with pytest.raises(RunExecutionError, match="run index entry"):
        runner.get(run_id)


def test_signer_and_index_concurrent_publishers_converge_without_replacement(
    tmp_path: Path,
) -> None:
    """Catch check-then-create races selecting multiple keys or replacing an index."""
    key_path = tmp_path / "state" / "key.ed25519"
    with ThreadPoolExecutor(max_workers=8) as pool:
        identities = tuple(
            pool.map(lambda _: RunSigningIdentity.load_or_create(key_path), range(8))
        )
    assert len({identity.key_id for identity in identities}) == 1
    assert key_path.stat().st_nlink == 1

    index_root = tmp_path / "runs"
    store = ArtifactStore(tmp_path / "artifacts-concurrent")
    reference = store.put_bytes(b"manifest", "application/json")
    runners = tuple(RunRunner(store, _signer(), index_root) for _ in range(8))
    run_id = "run-0123456789abcdef0123456789abcdef"
    with ThreadPoolExecutor(max_workers=8) as pool:
        tuple(pool.map(lambda runner: runner._publish_index(run_id, reference), runners))
    index_path = index_root / f"{run_id}.json"
    assert index_path.stat().st_nlink == 1
    assert stat.S_IMODE(index_path.stat().st_mode) == 0o600


def test_private_roots_reject_foreign_ownership(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Catch correct modes masking state owned by a different local principal."""
    store = ArtifactStore(tmp_path / "artifacts-owned")
    actual_uid = os.geteuid()
    monkeypatch.setattr("apar.runs.runner.os.geteuid", lambda: actual_uid + 1)

    with pytest.raises(ValueError, match="owned"):
        RunRunner(store, _signer(), tmp_path / "foreign-runs")


def test_tracked_task6_evidence_is_atomically_admitted_to_private_state(
    tmp_path: Path,
) -> None:
    """Catch requiring an unrepresentable tracked 0600 mode or retaining a public copy."""
    source = tmp_path / "task6-result.json"
    raw = b'{"frozen":true}'
    source.write_bytes(raw)
    source.chmod(0o644)
    state = tmp_path / "state"
    state_fd = runner_module._open_private_directory(state, "test state")
    try:
        observed_mode = runner_module._admit_private_evidence(
            source,
            state_fd,
            ".task6-result.private",
            hashlib.sha256(raw).hexdigest(),
        )
        runner_module._admit_private_evidence(
            source,
            state_fd,
            ".task6-result.private",
            hashlib.sha256(raw).hexdigest(),
        )
    finally:
        os.close(state_fd)

    private = state / ".task6-result.private"
    metadata = private.stat()
    assert observed_mode == 0o644
    assert stat.S_IMODE(source.stat().st_mode) == 0o644
    assert private.read_bytes() == raw
    assert stat.S_IMODE(metadata.st_mode) == 0o600
    assert metadata.st_nlink == 1


def test_policy_selection_rejects_callables_and_paths() -> None:
    """Catch API callers smuggling executable code or filesystem authority to a worker."""
    with pytest.raises(ValidationError, match="extra_forbidden"):
        AttackerPolicy.model_validate(
            {
                "family": "card_testing_cnp",
                "attacker_mode": "decision_only",
                "kind": "fixed",
                "query_budget": 1,
                "worker_timeout_ms": 1000,
                "callable": FixedPolicy(),
                "path": "/tmp/policy.py",
            }
        )


def test_policy_selection_rejects_caller_controlled_feedback_disclosure() -> None:
    """Catch an API caller granting itself realized-value visibility."""
    with pytest.raises(ValidationError, match="extra_forbidden"):
        AttackerPolicy.model_validate(
            {
                "family": "app_scam_mule",
                "attacker_mode": "decision_only",
                "kind": "adaptive",
                "query_budget": 2,
                "worker_timeout_ms": 1000,
                "expose_realized_value": True,
            }
        )


def test_standalone_runner_rejects_policy_attacker_mode_outside_reviewed_binding(
    tmp_path: Path,
    benchmark_population: Population,
) -> None:
    """Catch family-only pairing that ignores the reviewed attacker capability mode."""
    runner = RunRunner(ArtifactStore(tmp_path / "artifacts"), _signer(), tmp_path / "runs")
    policy = AttackerPolicy(
        attacker_mode=AttackerMode.ADAPTIVE,
        family="app_scam_mule",
        kind=AttackerPolicyKind.FIXED,
        query_budget=1,
        worker_timeout_ms=2_000,
    )

    with pytest.raises(RunExecutionError, match="attacker mode"):
        runner.execute(_bound_app_bundle(benchmark_population), policy)


def test_worker_history_contains_only_scenario_declared_feedback(
    benchmark_population: Population,
) -> None:
    """Catch evaluator objective/reasons crossing the disposable-worker wire."""
    benchmark = campaign_benchmark("app_scam_mule", benchmark_population)
    candidate = FixedPolicy().propose((), benchmark.public_bounds, np.random.default_rng(1))
    feedback = Feedback(
        action=Action.APPROVE,
        reason_family="approved",
        realized_value=Decimal("12.34"),
    )
    trial = VisibleTrial(
        candidate=candidate,
        feedback=feedback,
        objective_value=visible_objective(feedback),
    )

    golden = history_to_wire(
        (trial,),
        feedback_fields=(
            FeedbackField.APPROVE,
            FeedbackField.CHALLENGE,
            FeedbackField.DECLINE,
            FeedbackField.REALIZED_VALUE,
        ),
    )
    decision_only = history_to_wire(
        (trial,),
        feedback_fields=(
            FeedbackField.APPROVE,
            FeedbackField.CHALLENGE,
            FeedbackField.DECLINE,
        ),
    )

    assert set(golden[0]) == {"candidate", "feedback"}
    assert golden[0]["feedback"] == {
        "action": "approve",
        "realized_value": "12.34",
    }
    assert decision_only[0]["feedback"] == {"action": "approve"}
    assert "objective" not in str(golden)
    assert "approved" not in str(golden)


def test_clean_policy_worker_blocks_hidden_imports_reflection_and_network() -> None:
    """Catch a policy process importing or reaching evaluator state or opening a socket."""
    report = PolicyWorkerClient().probe(timeout_ms=2_000)

    assert report.clean_start
    assert report.filesystem_blocked
    assert report.forbidden_import_blocked
    assert report.reflection_import_blocked
    assert report.network_blocked
    assert report.hidden_modules_absent
    assert report.input_hidden_fields_absent
    assert report.orchestrator_modules_absent


def test_clean_policy_worker_blocks_real_process_and_native_escape_attempts() -> None:
    """Catch audit coverage that omits executable, fork, spawn, signal, or native escapes."""
    report = PolicyWorkerClient().probe(timeout_ms=2_000)

    assert report.exec_blocked
    assert report.fork_blocked
    assert report.spawn_blocked
    assert report.process_signal_blocked
    assert report.native_blocked


def test_policy_worker_spawn_has_no_parent_preexec_hook(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Catch Python child hooks that can deadlock a multi-threaded API before monitoring."""
    original = subprocess.Popen
    observed: list[dict[str, object]] = []

    def checked_popen(*args: Any, **kwargs: Any) -> subprocess.Popen[bytes]:
        observed.append(dict(kwargs))
        return original(*args, **kwargs)

    monkeypatch.setattr("apar.runs.runner.subprocess.Popen", checked_popen)

    assert PolicyWorkerClient().probe(timeout_ms=2_000).clean_start
    assert observed and "preexec_fn" not in observed[0]


def test_policy_worker_bounds_hung_startup_while_stdin_is_backpressured() -> None:
    """Catch a blocking parent write occurring outside the worker deadline."""
    with pytest.raises(PolicyWorkerError, match="deadline"):
        PolicyWorkerClient().probe_startup_hang(timeout_ms=50, request_bytes=2_000_000)


def test_policy_worker_converts_spawn_failure_to_fail_closed_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Catch OS startup errors escaping the stable execution boundary."""

    def fail_spawn(*_: object, **__: object) -> object:
        raise OSError("synthetic spawn failure")

    monkeypatch.setattr("apar.runs.runner.subprocess.Popen", fail_spawn)

    with pytest.raises(PolicyWorkerError, match="could not start"):
        PolicyWorkerClient().probe(timeout_ms=2_000)


def test_non_returning_policy_worker_is_killed_at_the_deadline() -> None:
    """Catch an in-process or cooperative timeout that cannot terminate a hung policy."""
    with pytest.raises(PolicyWorkerError, match="deadline"):
        PolicyWorkerClient().probe_hang(timeout_ms=50)


def test_policy_worker_is_killed_when_parent_rss_watchdog_exceeds_cap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Catch a platform-safe memory watchdog that observes but does not terminate."""

    def over_limit(_: int) -> int:
        return 805_306_369

    monkeypatch.setattr(runner_module, "_resident_bytes", over_limit)

    with pytest.raises(PolicyWorkerError, match="resident-memory limit"):
        PolicyWorkerClient().probe(timeout_ms=2_000)


def test_policy_worker_fails_closed_when_rss_cannot_be_observed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Catch an unavailable platform RSS reader silently leaving memory unbounded."""

    def unavailable(_: int) -> None:
        return None

    monkeypatch.setattr(runner_module, "_resident_bytes", unavailable)

    with pytest.raises(PolicyWorkerError, match="memory watchdog unavailable"):
        PolicyWorkerClient().probe(timeout_ms=2_000)


def test_worker_reconstructs_only_public_bounds_and_returns_a_candidate(
    benchmark_population: Population,
) -> None:
    """Catch an isolated worker that needs evaluator templates or emits untyped output."""
    benchmark = campaign_benchmark("card_testing_cnp", benchmark_population)

    candidate = PolicyWorkerClient().propose(
        kind=AttackerPolicyKind.FIXED,
        bounds=benchmark.public_bounds,
        history=(),
        feedback_fields=(FeedbackField.APPROVE, FeedbackField.DECLINE),
        seed=17,
        timeout_ms=2_000,
    )

    assert candidate.generation == 0
    assert candidate.parent_id is None
    assert candidate.params == benchmark.public_bounds.defaults


def test_cached_worker_uses_the_real_cache_only_planner_with_zero_network(
    benchmark_population: Population,
) -> None:
    """Catch a round-robin object being labelled as the cached LLM policy."""
    benchmark = campaign_benchmark("app_scam_mule", benchmark_population)
    worker = PolicyWorkerClient()

    candidate = worker.propose(
        kind=AttackerPolicyKind.CACHED_LLM,
        bounds=benchmark.public_bounds,
        history=(),
        feedback_fields=(FeedbackField.APPROVE, FeedbackField.DECLINE),
        seed=17,
        timeout_ms=2_000,
    )

    assert candidate.generation == 0
    assert [record.model_dump() for record in worker.take_audit_records()] == [
        {
            "cache_hit": True,
            "cache_source": "task6-v3-frozen-replay",
            "network_call_count": 0,
            "policy_kind": "cached_llm",
            "proposal_seed": 17,
        }
    ]


def test_completed_run_manifest_resolves_and_authenticates_every_artifact(
    tmp_path: Path,
    benchmark_population: Population,
) -> None:
    """Catch incomplete freezing, unsigned lineage, or an index naming missing bytes."""
    store = ArtifactStore(tmp_path / "artifacts")
    runner = RunRunner(
        artifact_store=store,
        signer=_signer(),
        run_index_root=tmp_path / "runs",
    )
    policy = AttackerPolicy(
        attacker_mode=AttackerMode.DECISION_ONLY,
        family="app_scam_mule",
        kind=AttackerPolicyKind.FIXED,
        query_budget=1,
        worker_timeout_ms=2_000,
    )

    manifest = runner.execute(_bound_app_bundle(benchmark_population), policy)

    assert set(manifest.artifacts) == {
        "authorization_receipt",
        "completion_receipt",
        "events",
        "feedback",
        "policy",
        "population",
        "provenance",
        "restricted_evaluation_audit",
        "restricted_evaluation_input",
        "restricted_hidden_evaluation_events",
        "restricted_validity",
        "scenario",
        "summary",
    }
    assert all(store.read(ref) for ref in manifest.artifacts.values())
    assert runner.verify_manifest(manifest)
    assert runner.verify_run(manifest)
    assert runner.get(manifest.run_id) == manifest
    events = json.loads(store.read(manifest.artifacts["events"]))
    assert events
    assert all(
        {
            "fee_account",
            "fee_amount",
            "frozen_account",
            "payee_account",
            "payer_account",
        }
        <= set(event["rail_data"])
        for event in events
    )
    assert all(
        {
            "fee_opening_balance",
            "frozen_opening_balance",
            "payee_opening_balance",
            "payer_opening_balance",
        }
        <= set(event["party_refs"])
        for event in events
    )


def test_detailed_validity_requires_the_exact_owning_completed_runner_capability(
    tmp_path: Path,
    benchmark_population: Population,
) -> None:
    """Catch direct, copied, forged, pre-completion, or cross-runner detail release."""
    assert "RestrictedValidityReport" not in hidden_validity_module.__all__
    assert not hasattr(hidden_validity_module, "RestrictedValidityReport")
    assert not hasattr(hidden_validity_module, "_evaluate_completed_run")
    assert not hasattr(hidden_validity_module, "_CAPABILITY_STATES")
    assert not hasattr(hidden_validity_module, "_RunnerCompletionCapability")
    assert not hasattr(HiddenValidityOracle(), "_report")

    first_runner = RunRunner(
        ArtifactStore(tmp_path / "first-artifacts"),
        _signer(),
        tmp_path / "first-runs",
    )
    second_runner = RunRunner(
        ArtifactStore(tmp_path / "second-artifacts"),
        _signer(),
        tmp_path / "second-runs",
    )
    capability = hidden_validity_module._issue_runner_completion_capability(first_runner)
    events = HiddenCampaignGenerator().generate("app_scam_mule", seed=11, count=8)

    with pytest.raises(TypeError, match="cannot be copied"):
        copy.copy(capability)
    forged = object.__new__(type(capability))
    with pytest.raises(PermissionError, match="capability"):
        hidden_validity_module._release_completed_runner_report(
            first_runner, forged, events
        )
    with pytest.raises(PermissionError, match="owning runner"):
        hidden_validity_module._release_completed_runner_report(
            second_runner, capability, events
        )
    with pytest.raises(PermissionError, match="not completed"):
        hidden_validity_module._release_completed_runner_report(
            first_runner, capability, events
        )

    manifest = first_runner.execute(
        _bound_app_bundle(benchmark_population),
        AttackerPolicy(
            attacker_mode=AttackerMode.DECISION_ONLY,
            family="app_scam_mule",
            kind=AttackerPolicyKind.FIXED,
            query_budget=1,
            worker_timeout_ms=5_000,
        ),
    )
    detail = strict_json_loads(
        first_runner.artifact_store.read(manifest.artifacts["restricted_validity"])
    )
    assert type(detail) is dict
    assert detail["valid"] is True
    assert type(detail["reason_codes"]) is list


def test_signed_provenance_inventories_and_revalidates_exact_execution_bytes(
    tmp_path: Path,
    benchmark_population: Population,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Catch dirty, relabelled, omitted, added, or dependency-unpinned execution code."""
    signer = _signer()
    store = ArtifactStore(tmp_path / "artifacts")
    runner = RunRunner(store, signer, tmp_path / "runs")
    manifest = runner.execute(
        _bound_app_bundle(benchmark_population),
        AttackerPolicy(
            attacker_mode=AttackerMode.DECISION_ONLY,
            family="app_scam_mule",
            kind=AttackerPolicyKind.FIXED,
            query_budget=1,
            worker_timeout_ms=5_000,
        ),
    )
    loaded = strict_json_loads(store.read(manifest.artifacts["provenance"]))
    assert type(loaded) is dict
    provenance = loaded
    inventory = provenance["execution_inventory"]
    assert type(inventory) is dict
    assert set(inventory) == {
        "dependencies",
        "entries",
        "environment",
        "inventory_sha256",
        "schema_version",
    }
    entries = inventory["entries"]
    assert type(entries) is list
    paths = [entry["path"] for entry in entries]
    expected_apar_sources = {
        path.as_posix() for path in Path("src/apar").rglob("*.py")
    }
    assert expected_apar_sources <= set(paths)
    assert {
        "pyproject.toml",
        "scripts/run_task6_holdout.py",
        "scripts/verify_g0.py",
        "scripts/verify_g1_g2.py",
    } <= set(paths)
    assert paths == sorted(paths)
    assert all(
        set(entry)
        == {"filesystem_mode", "git", "path", "sha256", "size_bytes", "type"}
        and entry["type"] == "regular_file"
        and type(entry["git"]) is dict
        and set(entry["git"])
        == {
            "index_blob_oid",
            "index_mode",
            "tracked",
            "working_blob_oid",
            "working_matches_index",
        }
        for entry in entries
    )
    dependencies = inventory["dependencies"]
    assert type(dependencies) is list
    assert {dependency["name"] for dependency in dependencies} >= {
        "cryptography",
        "fastapi",
        "numpy",
        "pandas",
        "pydantic",
        "pyarrow",
    }
    unsigned_inventory = {
        key: value for key, value in inventory.items() if key != "inventory_sha256"
    }
    assert inventory["inventory_sha256"] == runner_module._digest_document(
        unsigned_inventory
    )

    def changed_sha(document: dict[str, object]) -> None:
        checked = document["entries"]
        assert type(checked) is list and type(checked[0]) is dict
        checked[0]["sha256"] = "0" * 64

    def relabelled_path(document: dict[str, object]) -> None:
        checked = document["entries"]
        assert type(checked) is list and type(checked[0]) is dict
        checked[0]["path"] = "src/apar/relabelled.py"

    def omitted_entry(document: dict[str, object]) -> None:
        checked = document["entries"]
        assert type(checked) is list
        checked.pop()

    def added_entry(document: dict[str, object]) -> None:
        checked = document["entries"]
        assert type(checked) is list and type(checked[0]) is dict
        checked.append({**checked[0], "path": "src/apar/untracked-extra.py"})

    def changed_dependency(document: dict[str, object]) -> None:
        checked = document["dependencies"]
        assert type(checked) is list and type(checked[0]) is dict
        checked[0]["version"] = "0.invalid"

    for mutation in (
        changed_sha,
        relabelled_path,
        omitted_entry,
        added_entry,
        changed_dependency,
    ):
        forged_provenance = copy.deepcopy(provenance)
        forged_inventory = forged_provenance["execution_inventory"]
        assert type(forged_inventory) is dict
        mutation(forged_inventory)
        unsigned = {
            key: value
            for key, value in forged_inventory.items()
            if key != "inventory_sha256"
        }
        forged_inventory["inventory_sha256"] = runner_module._digest_document(unsigned)
        forged = _resign_with_replaced_provenance(
            runner=runner,
            store=store,
            signer=signer,
            manifest=manifest,
            provenance=forged_provenance,
        )
        assert not runner.verify_run(forged)

    forged_provenance = copy.deepcopy(provenance)
    forged_inventory = forged_provenance["execution_inventory"]
    assert type(forged_inventory) is dict
    forged_inventory["inventory_sha256"] = "f" * 64
    forged = _resign_with_replaced_provenance(
        runner=runner,
        store=store,
        signer=signer,
        manifest=manifest,
        provenance=forged_provenance,
    )
    assert not runner.verify_run(forged)

    original_inventory = runner_module._execution_inventory

    def changed_current_inventory() -> dict[str, object]:
        current = copy.deepcopy(original_inventory())
        current["inventory_sha256"] = "e" * 64
        return current

    monkeypatch.setattr(runner_module, "_execution_inventory", changed_current_inventory)
    assert not runner.verify_run(manifest)
    with pytest.raises(RunExecutionError, match="authenticated verification"):
        runner.get(manifest.run_id)


def test_git_source_relationship_fails_closed_but_inventory_supports_no_git_install(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Catch a present-but-broken Git index being silently labelled as no Git."""

    def broken_git(*_arguments: str, binary: bool = False) -> bytes | str:
        del binary
        raise RunExecutionError("broken Git metadata")

    monkeypatch.setattr(runner_module, "_git", broken_git)
    with pytest.raises(RunExecutionError, match="broken Git metadata"):
        runner_module._git_inventory(("src/apar/runs/runner.py",))

    installed_root = tmp_path / "installed-apar"
    installed_root.mkdir()
    monkeypatch.setattr(runner_module, "_REPOSITORY_ROOT", installed_root)
    assert runner_module._git_inventory(("apar/runs/runner.py",)) == ("sha256", {})

    def admit_without_git(
        source: Path,
        state_fd: int,
        private_name: str,
        expected_sha256: str,
    ) -> int:
        del source, state_fd, private_name, expected_sha256
        return 0o644

    def read_admitted(
        directory_fd: int,
        name: str,
        *,
        label: str,
        max_bytes: int,
    ) -> bytes:
        del directory_fd, name, label, max_bytes
        return b"pinned-task6-result"

    def read_pinned(
        path: Path,
        *,
        expected_mode: int,
        expected_sha256: str,
    ) -> bytes:
        del path, expected_mode, expected_sha256
        return b"pinned-current-file"

    monkeypatch.setattr(runner_module, "_admit_private_evidence", admit_without_git)
    monkeypatch.setattr(runner_module, "_read_private_file_at", read_admitted)
    monkeypatch.setattr(runner_module, "_strict_regular_file", read_pinned)
    monkeypatch.setattr(
        runner_module,
        "_execution_inventory",
        lambda: {"inventory_sha256": "self-contained"},
    )
    provenance = runner_module._provenance_document(1)
    task6 = provenance["task6_result"]
    assert type(task6) is dict
    assert task6["historical_verification"] == "embedded_pin_without_git"
    assert provenance["current_source_head"] == "git-unavailable"


def test_manifest_signature_detects_relabelled_lineage(
    tmp_path: Path,
    benchmark_population: Population,
) -> None:
    """Catch a content-addressed artifact set being accepted under forged run lineage."""
    runner = RunRunner(
        artifact_store=ArtifactStore(tmp_path / "artifacts"),
        signer=_signer(),
        run_index_root=tmp_path / "runs",
    )
    manifest = runner.execute(
        _bound_app_bundle(benchmark_population),
        AttackerPolicy(
            attacker_mode=AttackerMode.DECISION_ONLY,
            family="app_scam_mule",
            kind=AttackerPolicyKind.FIXED,
            query_budget=1,
            worker_timeout_ms=2_000,
        ),
    )

    forged = manifest.model_copy(update={"lineage_digest": "0" * 64})

    assert not runner.verify_manifest(forged)


def test_get_rechecks_the_durable_append_only_index_instead_of_memory(
    tmp_path: Path,
    benchmark_population: Population,
) -> None:
    """Catch same-process retrieval masking a replaced durable run-index entry."""
    index_root = tmp_path / "runs"
    runner = RunRunner(
        artifact_store=ArtifactStore(tmp_path / "artifacts"),
        signer=_signer(),
        run_index_root=index_root,
    )
    manifest = runner.execute(
        _bound_app_bundle(benchmark_population),
        AttackerPolicy(
            attacker_mode=AttackerMode.DECISION_ONLY,
            family="app_scam_mule",
            kind=AttackerPolicyKind.FIXED,
            query_budget=1,
            worker_timeout_ms=2_000,
        ),
    )
    (index_root / f"{manifest.run_id}.json").write_text("{}", encoding="utf-8")

    with pytest.raises(RunExecutionError, match="run index entry"):
        runner.get(manifest.run_id)


def test_get_rejects_non_hex_run_ids_before_any_index_lookup(tmp_path: Path) -> None:
    """Catch prefix/length-only run IDs reaching an attacker-selected index name."""
    store = ArtifactStore(tmp_path / "artifacts-invalid-run-id")
    runner = RunRunner(store, _signer())
    invalid_run_id = f"run-{'g' * 32}"
    runner._memory_index[invalid_run_id] = store.put_json({"forged": True})

    with pytest.raises(KeyError, match="does not exist"):
        runner.get(invalid_run_id)


def test_get_converts_malformed_index_bytes_to_a_fail_closed_error(
    tmp_path: Path,
    benchmark_population: Population,
) -> None:
    """Catch malformed durable bytes escaping the authenticated retrieval boundary."""
    index_root = tmp_path / "runs"
    runner = RunRunner(
        artifact_store=ArtifactStore(tmp_path / "artifacts"),
        signer=_signer(),
        run_index_root=index_root,
    )
    manifest = runner.execute(
        _bound_app_bundle(benchmark_population),
        AttackerPolicy(
            attacker_mode=AttackerMode.DECISION_ONLY,
            family="app_scam_mule",
            kind=AttackerPolicyKind.FIXED,
            query_budget=1,
            worker_timeout_ms=2_000,
        ),
    )
    (index_root / f"{manifest.run_id}.json").write_bytes(b"{not-json")

    with pytest.raises(RunExecutionError, match="run index entry"):
        runner.get(manifest.run_id)


def test_get_rejects_a_symlink_swap_at_the_index_descriptor_open(
    tmp_path: Path,
    benchmark_population: Population,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Catch a check/open race that follows a replacement symlink after ``lstat``."""
    index_root = tmp_path / "runs"
    runner = RunRunner(
        artifact_store=ArtifactStore(tmp_path / "artifacts"),
        signer=_signer(),
        run_index_root=index_root,
    )
    manifest = runner.execute(
        _bound_app_bundle(benchmark_population),
        AttackerPolicy(
            attacker_mode=AttackerMode.DECISION_ONLY,
            family="app_scam_mule",
            kind=AttackerPolicyKind.FIXED,
            query_budget=1,
            worker_timeout_ms=2_000,
        ),
    )
    index_path = index_root / f"{manifest.run_id}.json"
    replacement = tmp_path / "replacement-index.json"
    replacement.write_bytes(index_path.read_bytes())
    replacement.chmod(0o600)
    original_open = os.open
    replaced = False

    def replace_before_open(
        path: str | os.PathLike[str],
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        nonlocal replaced
        if not replaced and path == f"{manifest.run_id}.json" and dir_fd is not None:
            replaced = True
            index_path.unlink()
            index_path.symlink_to(replacement)
        return original_open(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(os, "open", replace_before_open)

    with pytest.raises(RunExecutionError, match="run index entry"):
        runner.get(manifest.run_id)


def test_authenticated_receipt_chain_is_bound_to_the_manifest_run_id(
    tmp_path: Path,
    benchmark_population: Population,
) -> None:
    """Catch valid same-authority receipts being relabelled under another manifest run."""
    signer = _signer()
    store = ArtifactStore(tmp_path / "artifacts")
    runner = RunRunner(store, signer, tmp_path / "runs")
    manifest = runner.execute(
        _bound_app_bundle(benchmark_population),
        AttackerPolicy(
            attacker_mode=AttackerMode.DECISION_ONLY,
            family="app_scam_mule",
            kind=AttackerPolicyKind.FIXED,
            query_budget=1,
            worker_timeout_ms=2_000,
        ),
    )
    wrong_run_id = "run-ffffffffffffffffffffffffffffffff"
    authorization = SignedRunReceipt.model_validate_json(
        store.read(manifest.artifacts["authorization_receipt"])
    )
    authorization_values = authorization.unsigned_document()
    authorization_values["run_id"] = wrong_run_id
    forged_authorization = SignedRunReceipt.model_validate(
        {
            **authorization_values,
            "signature_base64": signer.sign(authorization_values),
        }
    )
    authorization_ref = store.put_json(forged_authorization)

    completion = SignedRunReceipt.model_validate_json(
        store.read(manifest.artifacts["completion_receipt"])
    )
    completion_values = completion.unsigned_document()
    completion_values["run_id"] = wrong_run_id
    completion_values["previous_receipt_sha256"] = authorization_ref.sha256
    forged_completion = SignedRunReceipt.model_validate(
        {
            **completion_values,
            "signature_base64": signer.sign(completion_values),
        }
    )
    artifacts = {
        **manifest.artifacts,
        "authorization_receipt": authorization_ref,
        "completion_receipt": store.put_json(forged_completion),
    }
    draft = manifest.model_copy(
        update={
            "artifacts": artifacts,
            "lineage_digest": RunRunner._lineage_digest(artifacts),
        }
    )
    forged_manifest = draft.model_copy(
        update={"signature_base64": signer.sign(draft.unsigned_document())}
    )

    assert not runner.verify_run(forged_manifest)


def test_seeded_runs_are_byte_identical_without_exposing_private_signing_material(
    tmp_path: Path,
    benchmark_population: Population,
) -> None:
    """Catch timestamps/process IDs destabilizing reruns or private key bytes entering artifacts."""
    encoded_private = base64.b64encode(bytes(range(32)))
    policy = AttackerPolicy(
        attacker_mode=AttackerMode.DECISION_ONLY,
        family="app_scam_mule",
        kind=AttackerPolicyKind.FIXED,
        query_budget=1,
        worker_timeout_ms=2_000,
    )
    first_store = ArtifactStore(tmp_path / "first-artifacts")
    second_store = ArtifactStore(tmp_path / "second-artifacts")
    first = RunRunner(first_store, _signer(), tmp_path / "first-runs").execute(
        _bound_app_bundle(benchmark_population),
        policy,
    )
    second = RunRunner(second_store, _signer(), tmp_path / "second-runs").execute(
        _bound_app_bundle(benchmark_population),
        policy,
    )

    assert first == second
    assert {
        name: first_store.read(ref) for name, ref in first.artifacts.items()
    } == {
        name: second_store.read(ref) for name, ref in second.artifacts.items()
    }
    assert all(encoded_private not in first_store.read(ref) for ref in first.artifacts.values())


def test_restricted_bytes_do_not_influence_public_identity_or_policy_randomness(
    tmp_path: Path,
    benchmark_population: Population,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Catch evaluator-only commitments influencing worker seeds or public run identity."""
    bundle = _bound_app_bundle(benchmark_population)
    policy = AttackerPolicy(
        attacker_mode=AttackerMode.DECISION_ONLY,
        family="app_scam_mule",
        kind=AttackerPolicyKind.RANDOM,
        query_budget=3,
        worker_timeout_ms=5_000,
    )
    signer = _signer()
    first_store = ArtifactStore(tmp_path / "first-artifacts")
    first_runner = RunRunner(first_store, signer, tmp_path / "first-runs")
    first = first_runner.execute(bundle, policy)

    original_template_document = runner_module._campaign_template_document
    original_generate = HiddenCampaignGenerator.generate

    def alternate_template_document(params: object) -> dict[str, object]:
        return {
            **original_template_document(params),
            "evaluator_variant": "alternate-restricted-template",
        }

    def alternate_hidden_corpus(
        generator: HiddenCampaignGenerator,
        family: str,
        seed: int,
        count: int,
    ) -> tuple[PaymentEvent, ...]:
        return tuple(
            event.model_copy(
                update={
                    "privacy": {
                        **event.privacy,
                        "evaluator_variant": "alternate-restricted-corpus",
                    }
                }
            )
            for event in original_generate(generator, family, seed, count)
        )

    monkeypatch.setattr(
        runner_module, "_campaign_template_document", alternate_template_document
    )
    monkeypatch.setattr(HiddenCampaignGenerator, "generate", alternate_hidden_corpus)
    second_store = ArtifactStore(tmp_path / "second-artifacts")
    second_runner = RunRunner(second_store, signer, tmp_path / "second-runs")
    second = second_runner.execute(bundle, policy)

    def proposal_seeds(store: ArtifactStore, manifest: runner_module.RunManifest) -> list[int]:
        audit = strict_json_loads(
            store.read(manifest.artifacts["restricted_evaluation_audit"])
        )
        assert type(audit) is dict
        proposals = audit["policy_worker_proposals"]
        assert type(proposals) is list
        return [proposal["proposal_seed"] for proposal in proposals]

    assert first.run_id == second.run_id
    assert proposal_seeds(first_store, first) == proposal_seeds(second_store, second)
    assert first_store.read(first.artifacts["feedback"]) == second_store.read(
        second.artifacts["feedback"]
    )
    assert first_store.read(first.artifacts["events"]) == second_store.read(
        second.artifacts["events"]
    )
    assert first_runner.public_view(first) == second_runner.public_view(second)
    assert (
        first.artifacts["restricted_evaluation_input"].sha256
        != second.artifacts["restricted_evaluation_input"].sha256
    )
    assert (
        first.artifacts["restricted_hidden_evaluation_events"].sha256
        != second.artifacts["restricted_hidden_evaluation_events"].sha256
    )
    assert first.lineage_digest != second.lineage_digest


def test_agentic_final_artifact_replays_each_policy_winner_and_keeps_hidden_corpus_separate(
    tmp_path: Path,
) -> None:
    """Catch substituting a seed-only hidden corpus for the selected production candidate."""
    config = make_scenario_config(
        rail=Rail.AGENTIC,
        query_budget=2,
        feedback=[
            FeedbackField.APPROVE,
            FeedbackField.CHALLENGE,
            FeedbackField.DECLINE,
        ],
        seed=971,
        replay=make_scenario_config().replay.model_copy(update={"random_seed": 971}),
        benign_entity_count=40,
        illicit_entity_count=16,
    )
    card = make_threat_card(
        threat_id="agentic-winner-replay",
        family="agentic_intent_abuse",
        rails=[Rail.AGENTIC],
        default_config=config,
    )
    bundle: ScenarioBundle = bind_scenario_for_run(
        compile_scenario(card, config), threat_family=card.family
    )
    store = ArtifactStore(tmp_path / "artifacts")
    runner = RunRunner(store, _signer(), tmp_path / "runs")

    manifests = {
        kind: runner.execute(
            bundle,
            AttackerPolicy(
                attacker_mode=AttackerMode.DECISION_ONLY,
                family=card.family,
                kind=kind,
                query_budget=2,
                worker_timeout_ms=5_000,
            ),
        )
        for kind in (AttackerPolicyKind.FIXED, AttackerPolicyKind.ADAPTIVE)
    }
    summaries = {
        kind: strict_json_loads(store.read(manifest.artifacts["summary"]))
        for kind, manifest in manifests.items()
    }

    assert all(
        "restricted_hidden_evaluation_events" in manifest.artifacts
        for manifest in manifests.values()
    )
    assert all(
        summary["event_source"] == "task5_winner_commands_task4_agentic_replay"
        for summary in summaries.values()
        if type(summary) is dict
    )
    assert all(
        summary["selected_evaluation_event_sha256"]
        == manifests[kind].artifacts["events"].sha256
        for kind, summary in summaries.items()
        if type(summary) is dict
    )
    assert all(
        summary["production_approved_event_count"] >= 1
        for summary in summaries.values()
        if type(summary) is dict
    )
    assert (
        manifests[AttackerPolicyKind.FIXED].artifacts["events"].sha256
        != manifests[AttackerPolicyKind.ADAPTIVE].artifacts["events"].sha256
    )
    assert (
        manifests[AttackerPolicyKind.FIXED]
        .artifacts["restricted_hidden_evaluation_events"]
        .sha256
        == manifests[AttackerPolicyKind.ADAPTIVE]
        .artifacts["restricted_hidden_evaluation_events"]
        .sha256
    )
