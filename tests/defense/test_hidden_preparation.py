"""Authenticated, evaluator-only hidden-source preparation contracts."""

from __future__ import annotations

import base64
import hashlib
import inspect
import os
import secrets
from datetime import datetime, timedelta
from pathlib import Path

import pytest

import apar.evaluation.competition as competition
from apar.defense import orchestration
from apar.evaluation.contracts import CorpusManifest, FrozenCorpus
from apar.evaluation.gates import DefenseArm, EvaluationDescriptor, EvaluationKind
from apar.evaluation.hidden_source import HiddenSourceReceipt, ordered_ids_digest
from apar.evaluation.metrics import LatencySample, SliceAssignment, SliceManifest
from apar.evaluation.replay import (
    ReplayEvaluationContext,
    ReplayFeatureAssurance,
    ReplayLatencySamples,
)
from apar.evaluation.splits import EntityCohort
from apar.evaluation_hidden.authority_core import (
    HiddenBoundaryError,
    _hidden_source_request_document,
    _snapshot_hidden_source_binding,
    _verify_hidden_source_worker,
)
from apar.runs import RunManifest, RunRunner, RunSigningIdentity
from apar.runs.wire import canonical_json_bytes
from apar.storage.artifacts import ArtifactRef, ArtifactStore

ROOT = Path(__file__).resolve().parents[2]
PROFILE = ROOT / "config" / "defense" / "competition-profile.json"


def test_hidden_source_public_identity_is_descriptor_pinned_and_closes_fds(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = (tmp_path / "runtime").resolve()
    root.mkdir(mode=0o700)
    signer = RunSigningIdentity.from_private_bytes(b"u" * 32)
    public_path = root / "hidden-run-signing.pub"
    public_path.write_bytes(
        base64.b64decode(signer.public_key_base64, validate=True)
    )
    public_path.chmod(0o600)

    assert orchestration._load_competition_hidden_run_public_identity(root) == (
        signer.key_id,
        signer.public_key_base64,
    )
    public_path.chmod(0o644)
    with pytest.raises(orchestration.CliContractError, match="owner-only"):
        orchestration._load_competition_hidden_run_public_identity(root)
    public_path.unlink()
    target = root / "attacker.pub"
    target.write_bytes(b"a" * 32)
    target.chmod(0o600)
    public_path.symlink_to(target)
    with pytest.raises(orchestration.CliContractError, match="cannot be read"):
        orchestration._load_competition_hidden_run_public_identity(root)
    public_path.unlink()
    public_path.write_bytes(b"u" * 32)
    public_path.chmod(0o600)

    opened: list[int] = []
    closed: list[int] = []
    original_open = os.open
    original_close = os.close

    def recording_open(*args: object, **kwargs: object) -> int:
        descriptor = original_open(*args, **kwargs)  # type: ignore[arg-type]
        opened.append(descriptor)
        return descriptor

    def recording_close(descriptor: int) -> None:
        closed.append(descriptor)
        original_close(descriptor)

    monkeypatch.setattr(os, "open", recording_open)
    monkeypatch.setattr(os, "close", recording_close)
    monkeypatch.setattr(os, "read", lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError()))
    with pytest.raises(orchestration.CliContractError, match="cannot be read"):
        orchestration._load_competition_hidden_run_public_identity(root)
    assert opened
    assert set(opened) <= set(closed)


def test_hidden_authority_snapshots_source_binding_before_low_level_mutation() -> None:
    source = RunSigningIdentity.from_private_bytes(secrets.token_bytes(32))
    digest = "1" * 64
    receipt_ref = ArtifactRef(
        digest,
        orchestration.HIDDEN_SOURCE_RECEIPT_MEDIA_TYPE,
        1,
        f"{digest}/payload",
    )
    development_run_ids = tuple(f"run-{index:032x}" for index in range(200))
    binding = orchestration.HiddenSourceWorkerBinding(
        receipt_ref=receipt_ref,
        source_signer_key_id=source.key_id,
        source_public_key_base64=source.public_key_base64,
        development_run_ids=development_run_ids,
        development_event_ids=("event",),
        development_payment_ids=("payment",),
        development_campaign_ids=("campaign",),
    )
    snapshot = _snapshot_hidden_source_binding(binding)
    expected = _hidden_source_request_document(snapshot)
    attacker = RunSigningIdentity.from_private_bytes(b"w" * 32)
    object.__setattr__(binding, "source_signer_key_id", attacker.key_id)
    object.__setattr__(binding, "source_public_key_base64", attacker.public_key_base64)
    object.__setattr__(
        binding,
        "receipt_ref",
        ArtifactRef("2" * 64, receipt_ref.media_type, 1, f"{'2' * 64}/payload"),
    )

    assert _hidden_source_request_document(snapshot) == expected
    assert expected is not None
    assert expected["source_signer_key_id"] == source.key_id
    assert expected["receipt_ref"] == orchestration._reference_document(receipt_ref)


def test_hidden_source_seeds_are_unique_and_disjoint_from_preregistration() -> None:
    profile = orchestration.load_competition_profile(PROFILE, competition=True)
    valid = (900_001, 900_002, 900_003, 900_004)
    orchestration._require_independent_hidden_seeds(profile, valid)
    with pytest.raises(orchestration.CliContractError, match="not independent"):
        orchestration._require_independent_hidden_seeds(
            profile, (valid[0], valid[0], valid[2], valid[3])
        )
    with pytest.raises(orchestration.CliContractError, match="not independent"):
        orchestration._require_independent_hidden_seeds(
            profile,
            (
                profile.campaign_seed(profile.families[0], 0),
                valid[1],
                valid[2],
                valid[3],
            ),
        )


@pytest.fixture(scope="module")
def authenticated_hidden_runs(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[
    orchestration.CompetitionProfile,
    ArtifactStore,
    RunSigningIdentity,
    RunRunner,
    tuple[ArtifactRef, ...],
    object,
]:
    root = tmp_path_factory.mktemp("authenticated-hidden-runs")
    profile = orchestration.load_competition_profile(PROFILE, competition=True)
    store = ArtifactStore(root / "artifacts")
    signer = RunSigningIdentity.from_private_bytes(b"h" * 32)
    runner = RunRunner(store, signer, root / "run-index")
    minimum_start = profile.campaign_start(profile.families[0], 49) + timedelta(days=8)
    references: list[ArtifactRef] = []
    for position, family in enumerate(profile.families, start=1):
        manifest = orchestration._run_one_campaign(
            family,
            index=0,
            profile=profile,
            runner=runner,
            fixture=False,
            start_override=minimum_start + timedelta(days=position),
            seed_override=900_000 + position,
        )
        references.append(
            store.put_bytes(
                canonical_json_bytes(manifest.model_dump(mode="json")),
                "application/json",
            )
        )
    return profile, store, signer, runner, tuple(references), minimum_start


def _assemble(
    inputs: tuple[
        orchestration.CompetitionProfile,
        ArtifactStore,
        RunSigningIdentity,
        RunRunner,
        tuple[ArtifactRef, ...],
        object,
    ],
    **updates: object,
) -> orchestration._AuthenticatedHiddenSource:
    profile, store, signer, runner, references, minimum_start = inputs
    arguments: dict[str, object] = {
        "store": store,
        "runner": runner,
        "signer": signer,
        "profile": profile,
        "manifest_refs": references,
        "development_run_ids": tuple(f"run-{index:032x}" for index in range(200)),
        "development_event_ids": ("development-event",),
        "development_payment_ids": ("development-payment",),
        "development_campaign_ids": ("development-campaign",),
        "minimum_simulation_start": minimum_start,
    }
    arguments.update(updates)
    return orchestration._assemble_authenticated_hidden_source(**arguments)  # type: ignore[arg-type]


def test_hidden_source_starts_from_four_real_authenticated_manifest_refs(
    authenticated_hidden_runs: tuple[
        orchestration.CompetitionProfile,
        ArtifactStore,
        RunSigningIdentity,
        RunRunner,
        tuple[ArtifactRef, ...],
        object,
    ],
) -> None:
    source = _assemble(authenticated_hidden_runs)

    assert source.families == orchestration._FAMILIES
    assert source.corpus.manifest.profile_id == "defense-hidden-authority-v1"
    assert source.corpus.manifest.run_ids == tuple(
        item.run_id for item in source.manifests
    )
    assert all(source.manifests[index].policy_kind.value == "fixed" for index in range(4))
    assert not hasattr(competition, "build_competition_hidden_context")
    parameters = inspect.signature(
        orchestration._assemble_authenticated_hidden_source
    ).parameters
    assert "corpus" not in parameters
    assert "source_lineage_digest" not in parameters


def test_hidden_source_rejects_fixture_profile_order_signer_and_forged_manifest(
    authenticated_hidden_runs: tuple[
        orchestration.CompetitionProfile,
        ArtifactStore,
        RunSigningIdentity,
        RunRunner,
        tuple[ArtifactRef, ...],
        object,
    ],
) -> None:
    profile, store, _signer, _runner, references, _minimum_start = (
        authenticated_hidden_runs
    )
    with pytest.raises(orchestration.CliContractError, match="inputs"):
        _assemble(authenticated_hidden_runs, profile=orchestration.CompetitionProfile.fixture())
    with pytest.raises(orchestration.CliContractError, match="ordered"):
        _assemble(
            authenticated_hidden_runs,
            manifest_refs=(references[1], references[0], *references[2:]),
        )
    with pytest.raises(orchestration.CliContractError, match="authenticated"):
        _assemble(
            authenticated_hidden_runs,
            signer=RunSigningIdentity.from_private_bytes(b"a" * 32),
        )

    legitimate = RunManifest.model_validate_json(store.read(references[0]))
    attacker = RunSigningIdentity.from_private_bytes(b"x" * 32)
    forged_unsigned = {
        **legitimate.model_dump(mode="json", exclude={"signature_base64"}),
        "public_key_base64": attacker.public_key_base64,
        "signer_key_id": attacker.key_id,
    }
    forged = RunManifest.model_validate(
        {**forged_unsigned, "signature_base64": attacker.sign(forged_unsigned)}
    )
    forged_ref = store.put_bytes(
        canonical_json_bytes(forged.model_dump(mode="json")), "application/json"
    )
    with pytest.raises(orchestration.CliContractError, match="authenticated"):
        _assemble(
            authenticated_hidden_runs,
            manifest_refs=(forged_ref, *references[1:]),
        )

    forged_corpus = FrozenCorpus(
        observations=(),
        truth=(),
        manifest=CorpusManifest(
            profile_id="defense-hidden-authority-v1",
            run_ids=("a", "b", "c", "d"),
            run_lineage_digests=("1" * 64, "2" * 64, "3" * 64, "4" * 64),
            observation_count=0,
            truth_count=0,
        ),
    )
    with pytest.raises(TypeError):
        orchestration._assemble_authenticated_hidden_source(  # type: ignore[call-arg]
            corpus=forged_corpus,
            source_lineage_digest="5" * 64,
        )


def test_isolated_worker_publicly_verifies_source_receipt_and_context_binding(
    authenticated_hidden_runs: tuple[
        orchestration.CompetitionProfile,
        ArtifactStore,
        RunSigningIdentity,
        RunRunner,
        tuple[ArtifactRef, ...],
        object,
    ],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profile, store, signer, _runner, references, minimum_start_object = (
        authenticated_hidden_runs
    )
    minimum_start = minimum_start_object
    assert type(minimum_start) is datetime
    source = _assemble(authenticated_hidden_runs)
    hidden_truth = tuple(
        row.model_copy(update={"viewpoint": "hidden", "label_source": "hidden_truth"})
        for row in source.corpus.truth
    )
    authority_as_of = max(row.label_mature_at for row in hidden_truth) + timedelta(days=1)
    context = ReplayEvaluationContext(
        evaluation=EvaluationDescriptor(kind=EvaluationKind.HIDDEN, value="hidden"),
        truth=hidden_truth,
        observations=source.corpus.observations,
        as_of=authority_as_of,
        slice_assignments=tuple(
            SliceAssignment(
                event_id=row.event_id,
                regime="baseline",
                entity_cohorts=(EntityCohort.COLD_PAIR,),
            )
            for row in hidden_truth
        ),
        slice_manifest=SliceManifest.closed(),
        latency_samples=tuple(
            ReplayLatencySamples(
                arm=arm,
                samples=tuple(
                    LatencySample(
                        event_id=row.event_id,
                        feature_ms=1.0,
                        rules_ms=1.0,
                        model_ms=0.0 if arm is DefenseArm.RULES_ONLY else 1.0,
                        calibration_policy_ms=1.0,
                        end_to_end_ms=3.0 if arm is DefenseArm.RULES_ONLY else 4.0,
                    )
                    for row in hidden_truth
                ),
            )
            for arm in DefenseArm
        ),
        feature_assurance=ReplayFeatureAssurance(
            leakage_passed=True,
            parity_passed=True,
            leakage_evidence_digest="1" * 64,
            parity_evidence_digest="2" * 64,
        ),
    )
    restricted_ref = store.put_bytes(
        context.to_json(), "application/vnd.apar.hidden-evaluation-context+json"
    )
    development_run_ids = tuple(f"run-{index:032x}" for index in range(200))
    development_event_ids: tuple[str, ...] = ()
    development_payment_ids: tuple[str, ...] = ()
    development_campaign_ids: tuple[str, ...] = ()
    development_corpus = FrozenCorpus(
        observations=(),
        truth=(),
        manifest=CorpusManifest(
            profile_id="defense-competition-v1",
            run_ids=development_run_ids,
            run_lineage_digests=tuple(
                hashlib.sha256(item.encode()).hexdigest()
                for item in development_run_ids
            ),
            observation_count=0,
            truth_count=0,
        ),
    )
    ensemble_ref = store.put_bytes(
        b"ensemble", orchestration._DEFENDER_ENSEMBLE_MEDIA_TYPE
    )
    hidden_view = FrozenCorpus(
        observations=source.corpus.observations,
        truth=hidden_truth,
        manifest=source.corpus.manifest,
    )
    unsigned = {
        "authority_as_of": authority_as_of.isoformat().replace("+00:00", "Z"),
        "development_campaign_ids_digest": ordered_ids_digest(
            development_campaign_ids
        ),
        "development_corpus_digest": competition.frozen_corpus_digest(
            development_corpus
        ),
        "development_event_ids_digest": ordered_ids_digest(development_event_ids),
        "development_payment_ids_digest": ordered_ids_digest(
            development_payment_ids
        ),
        "development_run_ids_digest": ordered_ids_digest(development_run_ids),
        "ensemble_ref_sha256": ensemble_ref.sha256,
        "families": list(orchestration._FAMILIES),
        "hidden_context_digest": restricted_ref.sha256,
        "hidden_corpus_digest": competition.frozen_corpus_digest(hidden_view),
        "kind": "authenticated_independent_hidden_runs",
        "manifest_refs": [orchestration._reference_document(item) for item in references],
        "minimum_simulation_start": minimum_start.isoformat().replace("+00:00", "Z"),
        "profile_sha256": hashlib.sha256(profile.to_json()).hexdigest(),
        "public_key_base64": signer.public_key_base64,
        "run_ids": list(source.corpus.manifest.run_ids),
        "run_lineage_digests": list(source.corpus.manifest.run_lineage_digests),
        "schema_version": "1.0.0",
        "signer_key_id": signer.key_id,
    }
    receipt = HiddenSourceReceipt.model_validate(
        {**unsigned, "signature_base64": signer.sign(unsigned)}
    )
    receipt_ref = store.put_bytes(
        receipt.to_json(),
        "application/vnd.apar.restricted-hidden-source-receipt+json",
    )
    source_document = {
        "development_campaign_ids": list(development_campaign_ids),
        "development_event_ids": list(development_event_ids),
        "development_payment_ids": list(development_payment_ids),
        "development_run_ids": list(development_run_ids),
        "receipt_ref": orchestration._reference_document(receipt_ref),
        "source_public_key_base64": signer.public_key_base64,
        "source_signer_key_id": signer.key_id,
    }

    forbidden_refs = {
        manifest.artifacts[name].sha256
        for manifest in source.manifests
        for name in manifest.artifacts
        if name
        not in {"authorization_receipt", "completion_receipt", "policy", "scenario"}
    }
    original_read = ArtifactStore.read

    def guarded_read(self: ArtifactStore, reference: ArtifactRef) -> bytes:
        if self is store and reference.sha256 in forbidden_refs:
            pytest.fail("parent verifier opened a restricted hidden run payload")
        return original_read(self, reference)

    monkeypatch.setattr(ArtifactStore, "read", guarded_read)
    binding = orchestration._verify_hidden_source_metadata(
        store=store,
        profile=profile,
        source_signer_key_id=signer.key_id,
        source_public_key_base64=signer.public_key_base64,
        source_receipt_ref=receipt_ref,
        ensemble_ref=ensemble_ref,
        development_corpus=development_corpus,
        restricted_context_ref=restricted_ref,
        authority_as_of=authority_as_of,
        maximum_frozen_at=minimum_start,
    )
    assert binding.receipt_ref == receipt_ref

    legitimate = source.manifests[0]
    changed_artifacts = {
        **legitimate.artifacts,
        "source_authority_extra": legitimate.artifacts["policy"],
    }
    changed_lineage = hashlib.sha256(
        canonical_json_bytes(
            {
                "artifacts": {
                    name: reference.sha256
                    for name, reference in sorted(changed_artifacts.items())
                },
                "authorization_receipt": changed_artifacts[
                    "authorization_receipt"
                ].sha256,
                "completion_receipt": changed_artifacts[
                    "completion_receipt"
                ].sha256,
            }
        )
    ).hexdigest()
    draft = legitimate.model_copy(
        update={"artifacts": changed_artifacts, "lineage_digest": changed_lineage}
    )
    changed_manifest = draft.model_copy(
        update={"signature_base64": signer.sign(draft.unsigned_document())}
    )
    changed_manifest_ref = store.put_bytes(
        canonical_json_bytes(changed_manifest.model_dump(mode="json")),
        "application/json",
    )
    changed_source = {
        **unsigned,
        "manifest_refs": [
            orchestration._reference_document(changed_manifest_ref),
            *(orchestration._reference_document(item) for item in references[1:]),
        ],
        "run_lineage_digests": [
            changed_lineage,
            *source.corpus.manifest.run_lineage_digests[1:],
        ],
    }
    changed_source_receipt = HiddenSourceReceipt.model_validate(
        {**changed_source, "signature_base64": signer.sign(changed_source)}
    )
    changed_source_ref = store.put_bytes(
        changed_source_receipt.to_json(),
        orchestration.HIDDEN_SOURCE_RECEIPT_MEDIA_TYPE,
    )
    with pytest.raises(orchestration.CliContractError, match="manifest signature"):
        orchestration._verify_hidden_source_metadata(
            store=store,
            profile=profile,
            source_signer_key_id=signer.key_id,
            source_public_key_base64=signer.public_key_base64,
            source_receipt_ref=changed_source_ref,
            ensemble_ref=ensemble_ref,
            development_corpus=development_corpus,
            restricted_context_ref=restricted_ref,
            authority_as_of=authority_as_of,
            maximum_frozen_at=minimum_start,
        )

    _verify_hidden_source_worker(
        store=store,
        restricted_ref=restricted_ref,
        restricted_payload=context.to_json(),
        source_document=source_document,
        sealed_at_wire=authority_as_of.isoformat().replace("+00:00", "Z"),
    )

    attacker = RunSigningIdentity.from_private_bytes(b"q" * 32)
    with pytest.raises(HiddenBoundaryError, match="identity|receipt"):
        _verify_hidden_source_worker(
            store=store,
            restricted_ref=restricted_ref,
            restricted_payload=context.to_json(),
            source_document={
                **source_document,
                "source_public_key_base64": attacker.public_key_base64,
                "source_signer_key_id": attacker.key_id,
            },
            sealed_at_wire=authority_as_of.isoformat().replace("+00:00", "Z"),
        )
    changed = {**unsigned, "hidden_corpus_digest": "f" * 64}
    changed_receipt = HiddenSourceReceipt.model_validate(
        {**changed, "signature_base64": signer.sign(changed)}
    )
    changed_ref = store.put_bytes(
        changed_receipt.to_json(),
        "application/vnd.apar.restricted-hidden-source-receipt+json",
    )
    with pytest.raises(HiddenBoundaryError, match="corpus"):
        _verify_hidden_source_worker(
            store=store,
            restricted_ref=restricted_ref,
            restricted_payload=context.to_json(),
            source_document={
                **source_document,
                "receipt_ref": orchestration._reference_document(changed_ref),
            },
            sealed_at_wire=authority_as_of.isoformat().replace("+00:00", "Z"),
        )

    changed_truth = (
        hidden_truth[0].model_copy(update={"is_fraud": not hidden_truth[0].is_fraud}),
        *hidden_truth[1:],
    )
    changed_context = context.model_copy(update={"truth": changed_truth})
    with pytest.raises(HiddenBoundaryError, match="context digest"):
        _verify_hidden_source_worker(
            store=store,
            restricted_ref=restricted_ref,
            restricted_payload=changed_context.to_json(),
            source_document=source_document,
            sealed_at_wire=authority_as_of.isoformat().replace("+00:00", "Z"),
        )
