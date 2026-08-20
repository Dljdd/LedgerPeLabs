"""Competition-export contracts that must fail closed before Task 14 ships."""

from __future__ import annotations

import importlib.util
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any, cast

import pytest

from apar.contracts.events import EventKind, Rail
from apar.contracts.scenarios import ScenarioBundle
from apar.defense import orchestration as orchestration
from apar.defense.contracts import ObservedEvent
from apar.evaluation.contracts import Family
from apar.features.builders import build_feature_matrix
from apar.features.catalog import load_feature_catalog
from apar.runs import (
    AttackerPolicy,
    AttackerPolicyKind,
    RunManifest,
    RunRunner,
    RunSigningIdentity,
)
from apar.runs.wire import canonical_json_bytes
from apar.storage.artifacts import ArtifactRef, ArtifactStore

ROOT = Path(__file__).resolve().parents[2]
PROFILE = ROOT / "config" / "defense" / "competition-profile.json"
CATALOG = ROOT / "config" / "defense" / "feature-catalog.json"
FAMILIES: tuple[Family, ...] = (
    "agentic_intent_abuse",
    "app_scam_mule",
    "card_testing_cnp",
    "synthetic_merchant_refund",
)


def _contract(name: str) -> Callable[..., Any]:
    value = getattr(orchestration, name, None)
    assert callable(value), f"Task 14 must provide orchestration.{name}"
    return cast(Callable[..., Any], value)


def _reference(reference: ArtifactRef) -> dict[str, object]:
    return {
        "media_type": reference.media_type,
        "relative_path": reference.relative_path,
        "sha256": reference.sha256,
        "size_bytes": reference.size_bytes,
    }


def _event(
    event_id: str,
    decision_at: datetime,
    *,
    integrity_status: str = "pass",
) -> ObservedEvent:
    return ObservedEvent(
        event_id=event_id,
        payment_id=f"payment-{event_id}",
        rail=Rail.AGENTIC,
        event_type=EventKind.AUTHORIZATION,
        amount=Decimal("100.00"),
        currency="USD",
        event_time=decision_at,
        available_at=decision_at,
        decision_at=decision_at,
        actor_id=f"actor-{event_id}",
        counterparty_id=f"counterparty-{event_id}",
        optional_refs={},
        integrity_status=cast(Any, integrity_status),
        integrity_reason=(
            "receipt_failed" if integrity_status == "fail" else None
        ),
        is_decision_point=True,
    )


def _load_verifier() -> ModuleType:
    path = ROOT / "scripts" / "verify_g3.py"
    spec = importlib.util.spec_from_file_location("verify_g3_contracts", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_authenticated_ledger_entry_is_bound_to_profile_and_signed_run_contents(
    tmp_path: Path,
) -> None:
    """Catch trusting outer ledger claims instead of the authenticated artifacts."""
    verify_entry = _contract("_verify_ledger_entry")
    profile = orchestration.CompetitionProfile.fixture()
    store = ArtifactStore(tmp_path / "artifacts")
    signer = RunSigningIdentity.from_private_bytes(b"l" * 32)
    runner = RunRunner(store, signer, tmp_path / "runs")
    expected_family: Family = "agentic_intent_abuse"
    expected_index = 0
    manifest = orchestration._run_one_campaign(
        expected_family,
        index=expected_index,
        profile=profile,
        runner=runner,
        fixture=True,
    )
    entry = orchestration.RunLedgerEntry(
        family=expected_family,
        campaign_index=expected_index,
        seed=profile.campaign_seed(expected_family, expected_index),
        simulation_start_utc=profile.campaign_start(expected_family, expected_index),
        run_id=manifest.run_id,
        manifest=_reference(store.put_json(manifest)),
    )

    checked = verify_entry(
        entry=entry,
        expected_family=expected_family,
        expected_index=expected_index,
        profile=profile,
        runner=runner,
        store=store,
    )

    assert checked == manifest
    policy = AttackerPolicy.model_validate_json(store.read(checked.artifacts["policy"]))
    scenario = ScenarioBundle.model_validate_json(store.read(checked.artifacts["scenario"]))
    assert policy.kind is AttackerPolicyKind.FIXED
    assert policy.query_budget == 1
    assert policy.family == expected_family
    assert scenario.query_budget == 1
    assert scenario.rail is Rail.AGENTIC
    assert scenario.threat_card_ref == "agentic-payee-substitution@1"

    mutations = (
        entry.model_copy(update={"campaign_index": 1}),
        entry.model_copy(update={"seed": entry.seed + 1}),
        entry.model_copy(
            update={
                "simulation_start_utc": entry.simulation_start_utc
                + timedelta(seconds=1)
            }
        ),
        entry.model_copy(update={"family": "app_scam_mule"}),
    )
    for changed in mutations:
        with pytest.raises(orchestration.CliContractError):
            verify_entry(
                entry=changed,
                expected_family=expected_family,
                expected_index=expected_index,
                profile=profile,
                runner=runner,
                store=store,
            )

    other_manifest = orchestration._run_one_campaign(
        "app_scam_mule",
        index=0,
        profile=profile,
        runner=runner,
        fixture=True,
    )
    wrong_scenario = entry.model_copy(
        update={
            "run_id": other_manifest.run_id,
            "manifest": _reference(store.put_json(other_manifest)),
        }
    )
    with pytest.raises(orchestration.CliContractError):
        verify_entry(
            entry=wrong_scenario,
            expected_family=expected_family,
            expected_index=expected_index,
            profile=profile,
            runner=runner,
            store=store,
        )

    changed_population = ScenarioBundle.model_validate(
        scenario.model_copy(
            update={"benign_entity_count": scenario.benign_entity_count + 1}
        ).model_dump(mode="json")
    )
    population_manifest = runner.execute(changed_population, policy)
    population_entry = entry.model_copy(
        update={
            "run_id": population_manifest.run_id,
            "manifest": _reference(store.put_json(population_manifest)),
        }
    )
    with pytest.raises(orchestration.CliContractError):
        verify_entry(
            entry=population_entry,
            expected_family=expected_family,
            expected_index=expected_index,
            profile=profile,
            runner=runner,
            store=store,
        )

    timeout_manifest = runner.execute(
        scenario, policy.model_copy(update={"worker_timeout_ms": 4_999})
    )
    timeout_entry = entry.model_copy(
        update={
            "run_id": timeout_manifest.run_id,
            "manifest": _reference(store.put_json(timeout_manifest)),
        }
    )
    with pytest.raises(orchestration.CliContractError):
        verify_entry(
            entry=timeout_entry,
            expected_family=expected_family,
            expected_index=expected_index,
            profile=profile,
            runner=runner,
            store=store,
        )

    two_query_scenario = ScenarioBundle.model_validate(
        scenario.model_copy(update={"query_budget": 2}).model_dump(mode="json")
    )
    two_query_manifest = runner.execute(
        two_query_scenario,
        policy.model_copy(update={"query_budget": 2}),
    )
    wrong_query_budget = entry.model_copy(
        update={
            "run_id": two_query_manifest.run_id,
            "manifest": _reference(store.put_json(two_query_manifest)),
        }
    )
    with pytest.raises(orchestration.CliContractError):
        verify_entry(
            entry=wrong_query_budget,
            expected_family=expected_family,
            expected_index=expected_index,
            profile=profile,
            runner=runner,
            store=store,
        )

    random_manifest = runner.execute(
        scenario,
        policy.model_copy(update={"kind": AttackerPolicyKind.RANDOM}),
    )
    wrong_policy = entry.model_copy(
        update={
            "run_id": random_manifest.run_id,
            "manifest": _reference(store.put_json(random_manifest)),
        }
    )
    with pytest.raises(orchestration.CliContractError):
        verify_entry(
            entry=wrong_policy,
            expected_family=expected_family,
            expected_index=expected_index,
            profile=profile,
            runner=runner,
            store=store,
        )


def test_verify_g3_validates_the_canonical_profile_before_starting_any_child(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Catch a verifier that runs expensive checks before profile attestation."""
    verifier = _load_verifier()
    invalid = tmp_path / "competition-profile.json"
    invalid.write_bytes(canonical_json_bytes({"schema_version": "1.0.0"}))
    calls: list[tuple[str, list[str]]] = []

    monkeypatch.setattr(verifier, "COMPETITION_PROFILE", invalid, raising=False)
    monkeypatch.setattr(verifier, "CHECKS", (("MUST_NOT_RUN", ["false"]),))

    def record_child(label: str, argv: list[str]) -> int:
        calls.append((label, argv))
        return 0

    monkeypatch.setattr(verifier, "_run_check", record_child)

    assert verifier.main() != 0
    assert calls == []


def test_production_run_generation_requires_an_existing_pinned_signer_before_runs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Catch silently creating a competition signing identity mid-export."""
    root = tmp_path / "evidence"
    root.mkdir(mode=0o700)
    signer_path = root / "run-signing.key"
    calls: list[object] = []

    def unexpected_run(*args: object, **kwargs: object) -> RunManifest:
        calls.append((args, kwargs))
        raise AssertionError("campaign execution started before signer validation")

    monkeypatch.setattr(orchestration, "_run_one_campaign", unexpected_run)

    with pytest.raises(orchestration.CliContractError, match="signing identity"):
        orchestration._generate_competition_runs(
            profile=orchestration.CompetitionProfile.fixture(),
            root=root,
            signer_path=signer_path,
            output_name="ledger.json",
        )
    assert calls == []
    assert not signer_path.exists()


def test_same_preexisting_signing_key_has_the_same_identity_across_roots(
    tmp_path: Path,
) -> None:
    """Catch root-path-derived or regenerated competition signer identities."""
    identities = []
    for name in ("first", "second"):
        root = tmp_path / name
        root.mkdir(mode=0o700)
        key = root / "run-signing.key"
        key.write_bytes(b"s" * 32)
        key.chmod(0o600)
        identities.append(orchestration._load_standard_signer(root))

    assert identities[0].key_id == identities[1].key_id
    assert identities[0].public_key_base64 == identities[1].public_key_base64


def test_corpus_envelope_separates_observations_from_unopened_restricted_truth(
    tmp_path: Path,
) -> None:
    """Catch putting feature inputs and evaluator truth behind one readable ref."""
    load_observations = _contract("_load_observation_dataset")
    store = ArtifactStore(tmp_path / "artifacts")
    observed = (_event("observation-1", datetime(2026, 1, 1, tzinfo=UTC)),)
    observations_ref = store.put_bytes(
        canonical_json_bytes([row.model_dump(mode="json") for row in observed]),
        "application/vnd.apar.observations+json",
    )
    missing_truth_ref = ArtifactRef(
        sha256="f" * 64,
        media_type="application/vnd.apar.restricted-truth+json",
        size_bytes=1,
        relative_path=f"{'f' * 64}/payload",
    )
    envelope = orchestration.CorpusEnvelope.model_validate(
        {
            "schema_version": "1.0.0",
            "profile_sha256": "1" * 64,
            "run_ledger_sha256": "2" * 64,
            "observations": _reference(observations_ref),
            "restricted_truth": _reference(missing_truth_ref),
            "observation_digest": "3" * 64,
            "restricted_truth_digest": "4" * 64,
            "corpus_digest": "5" * 64,
            "campaign_count": 200,
            "family_campaign_counts": {family: 50 for family in FAMILIES},
        }
    )

    loaded = load_observations(store=store, envelope=envelope)

    assert loaded == observed
    assert not (tmp_path / "artifacts" / ("f" * 64) / "payload").exists()


def test_training_derives_mandatory_rule_ids_for_model_exclusion() -> None:
    """Catch production training that sends mandatory declines into CatBoost."""
    derive = _contract("_derive_training_exclusions")
    now = datetime(2026, 1, 1, tzinfo=UTC)
    matrix = build_feature_matrix(
        (
            _event("mandatory", now, integrity_status="fail"),
            _event("eligible", now + timedelta(seconds=1)),
        ),
        load_feature_catalog(CATALOG),
    )

    mandatory_ids = derive(
        matrix=matrix,
        training_ids=("mandatory", "eligible"),
    )

    assert mandatory_ids == ("mandatory",)


def test_rolling_folds_keep_equal_time_campaign_cohorts_whole_and_causal() -> None:
    """Catch index-based folds that split simultaneous campaigns or one class."""
    build_folds = _contract("_rolling_campaign_folds")
    start = datetime(2026, 1, 1, tzinfo=UTC)
    campaigns = tuple(f"campaign-{group}-{member}" for group in range(5) for member in range(3))
    row_ids = tuple(f"row-{campaign}" for campaign in campaigns)
    row_campaigns = dict(zip(row_ids, campaigns, strict=True))
    campaign_starts = {
        campaign: start + timedelta(days=int(campaign.split("-")[1]))
        for campaign in campaigns
    }
    labels = {
        row_id: (index % 3 == 1)
        for index, row_id in enumerate(row_ids)
    }
    split = SimpleNamespace(
        campaigns={"train": campaigns},
        row_campaigns=row_campaigns,
        row_is_fraud=labels,
        training_row_ids=row_ids,
    )

    folds = build_folds(
        split,
        campaign_start_times=campaign_starts,
    )

    assert len(folds) >= 2
    for fold in folds:
        fit_times = {
            campaign_starts[row_campaigns[row_id]] for row_id in fold.fit_ids
        }
        validation_times = {
            campaign_starts[row_campaigns[row_id]]
            for row_id in fold.validation_ids
        }
        assert max(fit_times) < min(validation_times)
        assert fit_times.isdisjoint(validation_times)
        assert {labels[row_id] for row_id in fold.fit_ids} == {False, True}
        assert {labels[row_id] for row_id in fold.validation_ids} == {False, True}


def test_production_ensemble_requires_pooled_and_four_distinct_lofo_roles() -> None:
    """Catch publishing one pooled model under five evaluation role labels."""
    build = _contract("_build_defender_ensemble")
    pooled = "a" * 64
    lofo = {
        family: f"{index + 2:x}" * 64
        for index, family in enumerate(FAMILIES)
    }
    exclusions = {"pooled": ()} | {family: (family,) for family in FAMILIES}

    ensemble = build(
        pooled_ref=pooled,
        lofo_refs=lofo,
        training_exclusions=exclusions,
    )

    assert ensemble.pooled_ref == pooled
    assert ensemble.lofo_refs == lofo
    assert ensemble.training_exclusions == exclusions
    assert len({pooled, *lofo.values()}) == 5

    duplicate = dict(lofo)
    duplicate["synthetic_merchant_refund"] = duplicate["card_testing_cnp"]
    with pytest.raises(orchestration.CliContractError):
        build(
            pooled_ref=pooled,
            lofo_refs=duplicate,
            training_exclusions=exclusions,
        )


def test_secure_root_rejects_relative_symlink_and_group_or_world_access(
    tmp_path: Path,
) -> None:
    with pytest.raises(orchestration.CliContractError):
        orchestration._secure_root(Path("relative-root"))
    for mode in (0o755, 0o777):
        root = tmp_path / f"mode-{mode:o}"
        root.mkdir(mode=0o700)
        root.chmod(mode)
        with pytest.raises(orchestration.CliContractError, match="pinned"):
            orchestration._secure_root(root)
    target = tmp_path / "target"
    target.mkdir(mode=0o700)
    link = tmp_path / "link"
    link.symlink_to(target, target_is_directory=True)
    with pytest.raises(orchestration.CliContractError, match="pinned"):
        orchestration._secure_root(link)


def test_json_publication_is_atomic_canonical_and_never_overwrites(
    tmp_path: Path,
) -> None:
    root = tmp_path / "root"
    root.mkdir(mode=0o700)
    document = {"z": 1, "a": [2, 3]}
    orchestration._publish_json_file(root, "evidence.json", document)
    output = root / "evidence.json"
    assert output.read_bytes() == canonical_json_bytes(document)
    with pytest.raises(orchestration.CliContractError, match="never overwritten"):
        orchestration._publish_json_file(root, "evidence.json", {"changed": True})
    assert output.read_bytes() == canonical_json_bytes(document)
    assert tuple(root.glob(".apar-*.tmp")) == ()
    for name in ("../escape.json", "nested/evidence.json", ".", ".."):
        with pytest.raises(orchestration.CliContractError):
            orchestration._publish_json_file(root, name, document)


def test_success_stdout_is_one_canonical_public_document(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    root = tmp_path / "root"
    root.mkdir(mode=0o700)
    artifact = ArtifactRef(
        "a" * 64,
        "application/vnd.apar.corpus-envelope+json",
        123,
        f"{'a' * 64}/payload",
    )
    monkeypatch.setattr(
        orchestration,
        "_build_competition_corpus",
        lambda **_kwargs: artifact,
    )
    code = orchestration.command_main(
        "build_defense_corpus",
        [
            "--profile",
            str(PROFILE),
            "--run-manifests",
            "b" * 64,
            "--root",
            str(root),
            "--output-manifest",
            "corpus.json",
        ],
    )
    captured = capsys.readouterr()
    assert code == 0
    assert captured.err == ""
    assert captured.out.count("\n") == 1
    parsed = orchestration.strict_json_loads(captured.out.removesuffix("\n").encode())
    assert captured.out.encode() == canonical_json_bytes(parsed) + b"\n"
    lowered = captured.out.lower()
    assert str(root).lower() not in lowered
    assert "private" not in lowered
    assert "signing.key" not in lowered
