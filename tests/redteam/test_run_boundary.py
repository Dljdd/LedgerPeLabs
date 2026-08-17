"""Process-isolated policy execution and signed immutable run evidence."""

from __future__ import annotations

import base64
import os
import stat
from pathlib import Path

import pytest
from pydantic import ValidationError

import apar.runs.runner as runner_module
from apar.generators import Population
from apar.redteam import FixedPolicy
from apar.runs import (
    AttackerPolicy,
    AttackerPolicyKind,
    PolicyWorkerClient,
    PolicyWorkerError,
    RunExecutionError,
    RunRunner,
    RunSigningIdentity,
    SignedRunReceipt,
)
from apar.storage.artifacts import ArtifactStore
from tests.redteam.conftest import campaign_benchmark


def _signer() -> RunSigningIdentity:
    return RunSigningIdentity.from_private_bytes(bytes(range(32)))


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


def test_policy_selection_rejects_callables_and_paths() -> None:
    """Catch API callers smuggling executable code or filesystem authority to a worker."""
    with pytest.raises(ValidationError, match="extra_forbidden"):
        AttackerPolicy.model_validate(
            {
                "family": "card_testing_cnp",
                "kind": "fixed",
                "query_budget": 1,
                "worker_timeout_ms": 1000,
                "callable": FixedPolicy(),
                "path": "/tmp/policy.py",
            }
        )


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
        seed=17,
        timeout_ms=2_000,
    )

    assert candidate.generation == 0
    assert candidate.parent_id is None
    assert candidate.params == benchmark.public_bounds.defaults


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
        family="app_scam_mule",
        kind=AttackerPolicyKind.FIXED,
        query_budget=1,
        worker_timeout_ms=2_000,
    )

    manifest = runner.execute(benchmark_population.bundle, policy)

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
        "restricted_validity",
        "scenario",
        "summary",
    }
    assert all(store.read(ref) for ref in manifest.artifacts.values())
    assert runner.verify_manifest(manifest)
    assert runner.verify_run(manifest)
    assert runner.get(manifest.run_id) == manifest


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
        benchmark_population.bundle,
        AttackerPolicy(
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
        benchmark_population.bundle,
        AttackerPolicy(
            family="app_scam_mule",
            kind=AttackerPolicyKind.FIXED,
            query_budget=1,
            worker_timeout_ms=2_000,
        ),
    )
    (index_root / f"{manifest.run_id}.json").write_text("{}", encoding="utf-8")

    with pytest.raises(RunExecutionError, match="run index entry"):
        runner.get(manifest.run_id)


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
        benchmark_population.bundle,
        AttackerPolicy(
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
        benchmark_population.bundle,
        AttackerPolicy(
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
        if not replaced and Path(path) == index_path:
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
        benchmark_population.bundle,
        AttackerPolicy(
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
        family="app_scam_mule",
        kind=AttackerPolicyKind.FIXED,
        query_budget=1,
        worker_timeout_ms=2_000,
    )
    first_store = ArtifactStore(tmp_path / "first-artifacts")
    second_store = ArtifactStore(tmp_path / "second-artifacts")
    first = RunRunner(first_store, _signer(), tmp_path / "first-runs").execute(
        benchmark_population.bundle,
        policy,
    )
    second = RunRunner(second_store, _signer(), tmp_path / "second-runs").execute(
        benchmark_population.bundle,
        policy,
    )

    assert first == second
    assert {
        name: first_store.read(ref) for name, ref in first.artifacts.items()
    } == {
        name: second_store.read(ref) for name, ref in second.artifacts.items()
    }
    assert all(encoded_private not in first_store.read(ref) for ref in first.artifacts.values())
