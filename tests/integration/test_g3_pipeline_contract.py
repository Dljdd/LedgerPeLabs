"""Cross-task contracts for the reduced G3 smoke pipeline.

The fixture is intentionally small, but every artifact asserted here is produced
and reloaded through the production Task 9, 12, and 13 boundaries.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace

import pytest

import apar.evaluation.competition as competition
from apar.contracts.events import EventKind, Rail
from apar.defense import orchestration
from apar.defense.bundle import DefenderBundlePublisher
from apar.defense.contracts import ObservedEvent
from apar.defense.orchestration import (
    G3FixtureResult,
    command_main,
    load_competition_profile,
    run_g3_fixture,
)
from apar.evaluation.contracts import EvaluationTruthRow
from apar.evaluation.gates import (
    ChampionDecision,
    ChampionStatus,
    DefenseArm,
    EvaluationDescriptor,
    EvaluationKind,
    EvaluatorSigningIdentity,
)
from apar.evaluation.metrics import LatencySample, SliceAssignment, SliceManifest
from apar.evaluation.replay import (
    ReplayEvaluationContext,
    ReplayFeatureAssurance,
    ReplayLatencySamples,
    ReplayThresholdSet,
)
from apar.evaluation.reporting import (
    DefenseScorecard,
    EvaluationArtifactBundle,
    PublicArtifactVerifier,
    load_evaluation_bundle,
)
from apar.evaluation.splits import EntityCohort
from apar.runs import RunSigningIdentity
from apar.runs.wire import strict_json_loads
from apar.storage.artifacts import ArtifactRef, ArtifactStore

ROOT = Path(__file__).resolve().parents[2]
PROFILE = ROOT / "config" / "defense" / "competition-profile.json"
_FIXTURE_SIGNER_SEED = hashlib.sha256(b"apar-g3-fixture-signer-v1").digest()


@dataclass(frozen=True, slots=True)
class _G3ContractRun:
    root: Path
    result: G3FixtureResult
    replay_batches: tuple[object, ...]
    gate_decisions: tuple[ChampionDecision, ...]
    published_pairs: tuple[tuple[DefenseScorecard, EvaluationArtifactBundle], ...]


@pytest.fixture(scope="module")
def g3_contract(tmp_path_factory: pytest.TempPathFactory) -> _G3ContractRun:
    """Run once while preserving the real replay, gate, and publication calls."""
    root = tmp_path_factory.mktemp("g3-pipeline-contract")
    replay_batches: list[object] = []
    gate_decisions: list[ChampionDecision] = []
    published_pairs: list[tuple[DefenseScorecard, EvaluationArtifactBundle]] = []

    real_replay = competition.replay_defense_arms
    real_gate = competition.evaluate_promotion_gates
    real_publish = competition.publish_scorecard

    def replay_spy(*args: object, **kwargs: object) -> object:
        batch = real_replay(*args, **kwargs)
        replay_batches.append(batch)
        return batch

    def gate_spy(*args: object, **kwargs: object) -> ChampionDecision:
        decision = real_gate(*args, **kwargs)
        assert type(decision) is ChampionDecision
        gate_decisions.append(decision)
        return decision

    def publish_spy(
        *args: object, **kwargs: object
    ) -> tuple[DefenseScorecard, EvaluationArtifactBundle]:
        published = real_publish(*args, **kwargs)
        assert type(published[0]) is DefenseScorecard
        assert type(published[1]) is EvaluationArtifactBundle
        published_pairs.append(published)
        return published

    patch = pytest.MonkeyPatch()
    patch.setattr(competition, "replay_defense_arms", replay_spy)
    patch.setattr(competition, "evaluate_promotion_gates", gate_spy)
    patch.setattr(competition, "publish_scorecard", publish_spy)
    try:
        result = run_g3_fixture(root)
    finally:
        patch.undo()

    return _G3ContractRun(
        root=root,
        result=result,
        replay_batches=tuple(replay_batches),
        gate_decisions=tuple(gate_decisions),
        published_pairs=tuple(published_pairs),
    )


def _threshold_set(payload: bytes) -> ReplayThresholdSet:
    document = strict_json_loads(payload)
    assert type(document) is dict
    reports = document.get("reports")
    assert type(reports) is list
    document["reports"] = tuple(reports)
    return ReplayThresholdSet.model_validate(document)


def _fixture_signer() -> RunSigningIdentity:
    return RunSigningIdentity.from_private_bytes(_FIXTURE_SIGNER_SEED)


def _signed_hidden_context() -> ReplayEvaluationContext:
    event_time = datetime(2027, 1, 20, tzinfo=UTC)
    observation = ObservedEvent(
        event_id="independent-hidden-event",
        payment_id="independent-hidden-payment",
        rail=Rail.CARD,
        event_type=EventKind.AUTHORIZATION,
        amount=Decimal("10.00"),
        currency="USD",
        event_time=event_time,
        available_at=event_time,
        decision_at=event_time,
        actor_id="hidden-actor",
        counterparty_id="hidden-counterparty",
        integrity_status="not_applicable",
        is_decision_point=True,
    )
    truth = EvaluationTruthRow(
        event_id=observation.event_id,
        payment_id=observation.payment_id,
        campaign_id="independent-hidden-campaign",
        family="card_testing_cnp",
        viewpoint="hidden",
        is_fraud=True,
        label_source="hidden_truth",
        label_mature_at=event_time,
        first_settlement_at=None,
        net_settled_value=Decimal("0.00"),
        lifecycle_event_ids=(observation.event_id,),
    )
    return ReplayEvaluationContext(
        evaluation=EvaluationDescriptor(kind=EvaluationKind.HIDDEN, value="hidden"),
        truth=(truth,),
        observations=(observation,),
        as_of=datetime(2027, 2, 1, tzinfo=UTC),
        slice_assignments=(
            SliceAssignment(
                event_id=observation.event_id,
                regime="baseline",
                entity_cohorts=(EntityCohort.COLD_PAIR,),
            ),
        ),
        slice_manifest=SliceManifest.closed(),
        latency_samples=tuple(
            ReplayLatencySamples(
                arm=arm,
                samples=(
                    LatencySample(
                        event_id=observation.event_id,
                        feature_ms=1.0,
                        rules_ms=1.0,
                        model_ms=0.0 if arm is DefenseArm.RULES_ONLY else 1.0,
                        calibration_policy_ms=0.0,
                        end_to_end_ms=(
                            2.0 if arm is DefenseArm.RULES_ONLY else 3.0
                        ),
                    ),
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


def test_reduced_g3_reloads_real_defender_thresholds_replay_and_publication(
    g3_contract: _G3ContractRun,
) -> None:
    """Catch a shortcut that fabricates a scorecard instead of using Tasks 9-13."""
    result = g3_contract.result
    store = ArtifactStore(g3_contract.root / "artifacts")

    assert result.run_manifests_verified == 4
    assert result.arms == tuple(arm.value for arm in DefenseArm)
    assert result.ensemble_mode == "reduced_pooled_only"
    assert result.champion_status == ChampionStatus.NO_PROMOTION.value
    assert g3_contract.replay_batches
    assert g3_contract.gate_decisions
    assert g3_contract.published_pairs
    assert g3_contract.gate_decisions[-1].status is ChampionStatus.NO_PROMOTION

    signer = _fixture_signer()
    assert signer.key_id == result.signer_key_id
    with DefenderBundlePublisher(store, signer, ROOT) as publisher:
        defender = publisher.load(result.defender_ref)
        defender.verify_reload()

    thresholds = _threshold_set(store.read(result.threshold_set_ref))
    assert tuple(item.arm for item in thresholds.reports) == tuple(DefenseArm)
    assert (
        thresholds.report_for(DefenseArm.LAYERED_HYBRID).report_digest
        == defender.threshold_report.report_digest
    )

    raw_bundle = strict_json_loads(store.read(result.evaluation_bundle_ref))
    assert type(raw_bundle) is dict
    verifier = PublicArtifactVerifier(
        signer_key_id=raw_bundle["signer_key_id"],
        public_key_base64=raw_bundle["public_key_base64"],
    )
    bundle = load_evaluation_bundle(
        result.evaluation_bundle_ref,
        artifact_store=store,
        verifier=verifier,
    )
    scorecard = bundle.scorecard(artifact_store=store, verifier=verifier)
    assert bundle.bundle_ref() == result.evaluation_bundle_ref
    assert bundle.public_artifacts["defense-scorecard.json"].sha256 == (
        result.scorecard_ref.sha256
    )
    assert scorecard.champion_decision.status is ChampionStatus.NO_PROMOTION
    assert tuple(row.arm for row in scorecard.leaderboard) == tuple(DefenseArm)

    public_payloads = [bundle.to_json(), scorecard.to_json()]
    public_payloads.extend(
        store.read(reference.as_artifact_ref())
        for reference in bundle.public_artifacts.values()
    )
    forbidden = (
        b"evaluation_truth",
        b"per_decision_predictions",
        b"restricted_hidden",
        b"metricderivationevidence",
        b"campaign_id",
        b"payment_id",
        b"event_id",
    )
    for payload in public_payloads:
        lowered = payload.lower()
        assert all(token not in lowered for token in forbidden)


def test_hidden_cli_rejects_missing_or_random_development_scorecard(
    g3_contract: _G3ContractRun,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Catch accepting any resolvable artifact as hidden-release authorization."""
    result = g3_contract.result
    store = ArtifactStore(g3_contract.root / "artifacts")
    random_ref = store.put_bytes(b"{}", "application/json")
    profile = load_competition_profile(PROFILE, competition=True)
    profile_digest = hashlib.sha256(profile.to_json()).hexdigest()
    hidden_context_ref = competition.seal_hidden_context(
        store=store,
        signer=RunSigningIdentity.from_private_bytes(b"h" * 32),
        context=_signed_hidden_context(),
        profile_sha256=profile_digest,
        development_corpus_digest="d" * 64,
    )
    corpus_envelope_ref = store.put_bytes(
        b"{}", "application/vnd.apar.corpus-envelope+json"
    )
    reference = orchestration._reference_document(result.defender_ref)
    fake_ensemble = SimpleNamespace(
        pooled_ref=reference,
        held_family_refs={family: reference for family in profile.families},
        corpus_envelope_ref=orchestration._reference_document(corpus_envelope_ref),
    )
    hidden_authority_opened = False

    def forbidden_hidden_identity(*_args: object) -> object:
        nonlocal hidden_authority_opened
        hidden_authority_opened = True
        raise AssertionError("invalid development receipt opened hidden authority")

    monkeypatch.setattr(orchestration, "_load_standard_signer", lambda _root: _fixture_signer())
    monkeypatch.setattr(
        orchestration, "_load_defender_ensemble", lambda **_kwargs: fake_ensemble
    )
    monkeypatch.setattr(
        orchestration,
        "_load_corpus_envelope",
        lambda *_args: (SimpleNamespace(run_ledger_sha256="e" * 64), object()),
    )
    monkeypatch.setattr(orchestration, "make_evaluation_split", lambda *_args: object())
    monkeypatch.setattr(
        orchestration,
        "_load_competition_evaluator_identity",
        lambda _root: EvaluatorSigningIdentity.from_private_bytes(b"e" * 32),
    )
    monkeypatch.setattr(
        orchestration, "_load_competition_hidden_identity", forbidden_hidden_identity
    )
    common = [
        "--phase",
        "hidden",
        "--defender",
        result.defender_ref.sha256,
        "--profile",
        str(PROFILE),
        "--root",
        str(g3_contract.root),
    ]

    assert command_main("evaluate_defender", common) == 2
    first = capsys.readouterr()
    assert first.out == ""
    assert "completed development scorecard" in first.err.lower()

    with_random = [
        *common,
        "--development-scorecard",
        random_ref.sha256,
        "--hidden-corpus",
        hidden_context_ref.sha256,
    ]
    assert command_main("evaluate_defender", with_random) == 2
    second = capsys.readouterr()
    assert second.out == ""
    assert "completed development receipt" in second.err.lower()
    assert not hidden_authority_opened


def test_competition_ensemble_requires_pooled_and_four_distinct_lofo_candidates() -> None:
    """Catch a production manifest that silently relabels pooled-only as LOFO."""
    profile = load_competition_profile(PROFILE, competition=True)
    signer = RunSigningIdentity.from_private_bytes(b"g" * 32)
    hidden_source = RunSigningIdentity.from_private_bytes(b"h" * 32)

    def ref(character: str) -> ArtifactRef:
        digest = character * 64
        return ArtifactRef(
            digest,
            "application/vnd.apar.defender-bundle+json",
            1,
            f"{digest}/payload",
        )

    held = {
        family: ref(character)
        for family, character in zip(profile.families, "2345", strict=True)
    }
    ensemble = orchestration._build_defender_ensemble(
        profile=profile,
        pooled_ref=ref("1"),
        held_family_refs=held,
        signer=signer,
        hidden_source_signer_key_id=hidden_source.key_id,
        hidden_source_public_key_base64=hidden_source.public_key_base64,
    )

    assert ensemble.mode == "competition_full"
    assert ensemble.pooled_ref["sha256"] == "1" * 64
    assert ensemble.held_family_refs == {
        family: {
            "media_type": reference.media_type,
            "relative_path": reference.relative_path,
            "sha256": reference.sha256,
            "size_bytes": reference.size_bytes,
        }
        for family, reference in held.items()
    }
    assert ensemble.held_family_training_exclusions == {
        family: (family,) for family in profile.families
    }

    missing_family = dict(held)
    missing_family.pop(profile.families[-1])
    with pytest.raises((ValueError, orchestration.CliContractError)):
        orchestration._build_defender_ensemble(
            profile=profile,
            pooled_ref=ref("1"),
            held_family_refs=missing_family,
            signer=signer,
            hidden_source_signer_key_id=hidden_source.key_id,
            hidden_source_public_key_base64=hidden_source.public_key_base64,
        )
