"""Fail-closed production evaluation release contracts for Task 14."""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace

import pytest

import apar.evaluation.competition as competition
from apar.contracts.events import EventKind, Rail
from apar.defense import orchestration
from apar.defense.contracts import ObservedEvent
from apar.evaluation.competition import (
    DevelopmentCompletionReceipt,
    RestrictedHiddenContextEnvelope,
    publish_competition_evaluation,
    publish_reduced_g3_evaluation,
    seal_development_completion,
    seal_hidden_context,
    verify_development_completion,
    verify_hidden_context,
)
from apar.evaluation.contracts import (
    CorpusManifest,
    EvaluationTruthRow,
    FrozenCorpus,
)
from apar.evaluation.gates import (
    DefenseArm,
    EvaluationDescriptor,
    EvaluationKind,
    EvaluatorReplayVerifier,
    EvaluatorSigningIdentity,
)
from apar.evaluation.metrics import LatencySample, SliceAssignment, SliceManifest
from apar.evaluation.regimes import RegimeKind
from apar.evaluation.replay import (
    ReplayEvaluationContext,
    ReplayFeatureAssurance,
    ReplayLatencySamples,
)
from apar.evaluation.splits import EntityCohort
from apar.runs import RunSigningIdentity
from apar.storage.artifacts import ArtifactRef, ArtifactStore


def _ref(character: str, media_type: str = "application/json") -> ArtifactRef:
    digest = character * 64
    return ArtifactRef(digest, media_type, 1, f"{digest}/payload")


def _empty_hidden_context() -> ReplayEvaluationContext:
    event_time = datetime(2027, 1, 20, tzinfo=UTC)
    observation = ObservedEvent(
        event_id="hidden-event",
        payment_id="hidden-payment",
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
        campaign_id="hidden-campaign",
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
                        calibration_policy_ms=1.0,
                        end_to_end_ms=3.0 if arm is DefenseArm.RULES_ONLY else 4.0,
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


def test_reduced_publication_rejects_a_competition_corpus_before_artifact_access(
    tmp_path: Path,
) -> None:
    """The pooled-only publisher must never become a production escape hatch."""
    corpus = FrozenCorpus(
        observations=(),
        truth=(),
        manifest=CorpusManifest(
            profile_id="competition-v1",
            run_ids=(),
            run_lineage_digests=(),
            observation_count=0,
            truth_count=0,
        ),
    )
    with pytest.raises(ValueError, match="fixture-only"):
        publish_reduced_g3_evaluation(
            store=ArtifactStore(tmp_path / "artifacts"),
            publication_signer=RunSigningIdentity.from_private_bytes(b"p" * 32),
            defender_ref=_ref("1", "application/vnd.apar.defender-bundle+json"),
            corpus=corpus,
            split=object(),  # rejected before a reduced split can be trusted
            profile_sha256="2" * 64,
            authenticated_run_ids=(),
        )


def test_hidden_context_envelope_is_signed_distinct_and_profile_bound(
    tmp_path: Path,
) -> None:
    store = ArtifactStore(tmp_path / "artifacts")
    signer = RunSigningIdentity.from_private_bytes(b"h" * 32)
    context = _empty_hidden_context()
    envelope_ref = seal_hidden_context(
        store=store,
        signer=signer,
        context=context,
        profile_sha256="a" * 64,
        development_corpus_digest="b" * 64,
    )

    envelope, loaded, restricted_ref = verify_hidden_context(
        store=store,
        envelope_ref=envelope_ref,
        signer=signer,
        profile_sha256="a" * 64,
        development_corpus_digest="b" * 64,
        development_event_ids=("development-event",),
    )

    assert type(envelope) is RestrictedHiddenContextEnvelope
    assert loaded == context.observations
    assert restricted_ref.sha256 == envelope.context_sha256
    with pytest.raises(ValueError, match="profile"):
        verify_hidden_context(
            store=store,
            envelope_ref=envelope_ref,
            signer=signer,
            profile_sha256="c" * 64,
            development_corpus_digest="b" * 64,
            development_event_ids=("development-event",),
        )


def test_development_completion_rejects_stale_hidden_or_mismatched_lineage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = ArtifactStore(tmp_path / "artifacts")
    signer = RunSigningIdentity.from_private_bytes(b"d" * 32)
    descriptors = tuple(
        sorted(
            (
                "chronological:development",
                *(f"cold_entity:{item.value}" for item in EntityCohort),
                *(
                    f"held_family:{family}"
                    for family in (
                        "agentic_intent_abuse",
                        "app_scam_mule",
                        "card_testing_cnp",
                        "synthetic_merchant_refund",
                    )
                ),
                *(f"regime:{item.value}" for item in RegimeKind),
            )
        )
    )
    scorecard_ref = store.put_bytes(
        b"scorecard", "application/vnd.apar.defense-scorecard+json"
    )
    evaluation_bundle_ref = store.put_bytes(
        b"bundle", "application/vnd.apar.evaluation-artifact-bundle+json"
    )
    development_evidence_ref = store.put_bytes(
        b"development-evidence",
        "application/vnd.apar.restricted-development-evidence+json",
    )
    pooled_ref = _ref("a", "application/vnd.apar.defender-bundle+json")
    corpus_ref = _ref(
        "c", "application/vnd.apar.verified-corpus-attestation+json"
    )
    corpus_evidence_ref = _ref(
        "f", "application/vnd.apar.replay-corpus-evidence+json"
    )
    split = SimpleNamespace(split_digest="5" * 64)
    attestation_document = {
        "top_ref": {
            "media_type": pooled_ref.media_type,
            "relative_path": pooled_ref.relative_path,
            "sha256": pooled_ref.sha256,
            "size_bytes": pooled_ref.size_bytes,
        }
    }
    attestation_payload = orchestration.canonical_json_bytes(attestation_document)
    pooled_attestation = SimpleNamespace(
        top_ref=pooled_ref,
        to_json=lambda: attestation_payload,
    )
    pooled_runtime = SimpleNamespace(attestation=pooled_attestation)
    threshold_set = SimpleNamespace(
        threshold_set_digest="6" * 64,
        model_dump=lambda **_kwargs: {
            "schema_version": "1.0.0",
            "threshold_set_digest": "6" * 64,
        },
    )
    champion = SimpleNamespace(decision_digest="8" * 64)
    monkeypatch.setattr(
        competition,
        "DefenseScorecard",
        SimpleNamespace(
            from_json=lambda *args, **kwargs: SimpleNamespace(
                evaluation_id="evaluation-id"
            )
        ),
    )
    monkeypatch.setattr(
        competition,
        "load_evaluation_bundle",
        lambda *args, **kwargs: SimpleNamespace(
            scorecard_sha256=scorecard_ref.sha256,
            evaluation_id="evaluation-id",
            bundle_digest="b" * 64,
        ),
    )
    fake_batches = tuple(
        SimpleNamespace(
            results=(
                SimpleNamespace(
                    evaluation=EvaluationDescriptor(
                        kind=EvaluationKind(kind), value=value
                    )
                ),
            )
        )
        for kind, value in (item.split(":", 1) for item in descriptors)
    )
    primary_results = tuple(
        SimpleNamespace(
            arm=arm,
            evaluation=EvaluationDescriptor(
                kind=EvaluationKind.CHRONOLOGICAL, value="development"
            ),
            evaluation_lineage=SimpleNamespace(
                corpus_digest="9" * 64,
                split_digest=split.split_digest,
            ),
        )
        for arm in DefenseArm
    )
    fake_request = SimpleNamespace(
        promotion_envelope=SimpleNamespace(
            envelope_digest="7" * 64,
            component_batches=fake_batches,
            combined_batch=SimpleNamespace(results=primary_results),
            hidden_proofs=(),
        ),
        champion_decision=champion,
        threshold_set=threshold_set,
        metric_evidence=(),
    )
    monkeypatch.setattr(
        competition,
        "ScorecardPublicationRequest",
        SimpleNamespace(from_worker_json=lambda _payload: fake_request),
    )
    monkeypatch.setattr(
        competition, "_candidate_runtime", lambda **_kwargs: pooled_runtime
    )
    monkeypatch.setattr(
        competition, "make_leave_one_family_out", lambda split, _family: split
    )
    monkeypatch.setattr(
        competition, "_verify_frozen_development_request", lambda *_a, **_k: None
    )
    monkeypatch.setattr(
        competition,
        "load_corpus_attestation",
        lambda *_a, **_k: SimpleNamespace(
            top_ref=corpus_ref,
            corpus_digest="9" * 64,
            split_digest=split.split_digest,
            evidence_ref=corpus_evidence_ref,
        ),
    )
    evaluator = EvaluatorSigningIdentity.from_private_bytes(b"e" * 32)
    evaluator_verifier = EvaluatorReplayVerifier.from_signer(evaluator)
    held_refs = {
        family: _ref(character, "application/vnd.apar.defender-bundle+json")
        for family, character in zip(
            (
                "agentic_intent_abuse",
                "app_scam_mule",
                "card_testing_cnp",
                "synthetic_merchant_refund",
            ),
            "bcde",
            strict=True,
        )
    }
    restricted_fields = {
        "schema_version": "1.0.0",
        "privacy_classification": "restricted_evaluation_evidence",
        "evaluation_id": "evaluation-id",
        "corpus_attestation_ref": competition._ref_document(corpus_ref),
        "corpus_evidence_ref": competition._ref_document(corpus_evidence_ref),
        "corpus_content_digest": "9" * 64,
        "split_digest": split.split_digest,
        "defender_attestation": attestation_document,
        "promotion_envelope_digest": "7" * 64,
        "champion_decision_digest": champion.decision_digest,
        "threshold_set_digest": threshold_set.threshold_set_digest,
        "threshold_set": threshold_set.model_dump(mode="json"),
        "metric_lineage": [],
        "public_bundle_digest": "b" * 64,
        "public_bundle_ref": competition._ref_document(evaluation_bundle_ref),
        "signer_key_id": signer.key_id,
        "public_key_base64": signer.public_key_base64,
    }
    def store_restricted(fields: dict[str, object]) -> ArtifactRef:
        signed = {**fields, "signature_base64": signer.sign(fields)}
        return store.put_bytes(
            orchestration.canonical_json_bytes(
                {
                    **signed,
                    "receipt_digest": hashlib.sha256(
                        orchestration.canonical_json_bytes(signed)
                    ).hexdigest(),
                }
            ),
            "application/vnd.apar.restricted-publication-receipt+json",
        )

    restricted_ref = store_restricted(restricted_fields)
    receipt_ref = seal_development_completion(
        store=store,
        signer=signer,
        ensemble_ref=_ref("1", "application/vnd.apar.defender-ensemble+json"),
        profile_sha256="2" * 64,
        corpus_envelope_ref=_ref("3", "application/vnd.apar.corpus-envelope+json"),
        run_ledger_sha256="4" * 64,
        scorecard_ref=scorecard_ref,
        evaluation_bundle_ref=evaluation_bundle_ref,
        development_evidence_ref=development_evidence_ref,
        restricted_publication_receipt_ref=restricted_ref,
        promotion_envelope_digest="7" * 64,
        descriptor_scope=descriptors,
    )
    receipt = verify_development_completion(
        store=store,
        receipt_ref=receipt_ref,
        signer=signer,
        ensemble_ref=_ref("1", "application/vnd.apar.defender-ensemble+json"),
        profile_sha256="2" * 64,
        corpus_envelope_ref=_ref("3", "application/vnd.apar.corpus-envelope+json"),
        run_ledger_sha256="4" * 64,
        evaluator_verifier=evaluator_verifier,
        pooled_ref=pooled_ref,
        held_family_refs=held_refs,  # type: ignore[arg-type]
        split=split,  # type: ignore[arg-type]
    )
    assert type(receipt) is DevelopmentCompletionReceipt
    assert receipt.hidden_included is False

    coordinated_mutations: tuple[dict[str, object], ...] = (
        {"champion_decision_digest": "0" * 64},
        {"threshold_set_digest": "0" * 64},
        {"threshold_set": {"schema_version": "forged"}},
        {"public_bundle_digest": "0" * 64},
        {"corpus_content_digest": "0" * 64},
        {"split_digest": "0" * 64},
        {"defender_attestation": {"top_ref": "forged"}},
        {"metric_lineage": [{"arm": "forged"}]},
        {
            "corpus_attestation_ref": competition._ref_document(
                _ref("0", "application/vnd.apar.verified-corpus-attestation+json")
            )
        },
        {
            "corpus_evidence_ref": competition._ref_document(
                _ref("0", "application/vnd.apar.replay-corpus-evidence+json")
            )
        },
        {
            "public_bundle_ref": competition._ref_document(
                _ref("0", "application/vnd.apar.evaluation-artifact-bundle+json")
            )
        },
    )
    for mutation in coordinated_mutations:
        forged_restricted_ref = store_restricted(
            {**restricted_fields, **mutation}
        )
        forged_completion_ref = seal_development_completion(
            store=store,
            signer=signer,
            ensemble_ref=_ref("1", "application/vnd.apar.defender-ensemble+json"),
            profile_sha256="2" * 64,
            corpus_envelope_ref=_ref(
                "3", "application/vnd.apar.corpus-envelope+json"
            ),
            run_ledger_sha256="4" * 64,
            scorecard_ref=scorecard_ref,
            evaluation_bundle_ref=evaluation_bundle_ref,
            development_evidence_ref=development_evidence_ref,
            restricted_publication_receipt_ref=forged_restricted_ref,
            promotion_envelope_digest="7" * 64,
            descriptor_scope=descriptors,
        )
        with pytest.raises(ValueError, match="restricted publication receipt"):
            verify_development_completion(
                store=store,
                receipt_ref=forged_completion_ref,
                signer=signer,
                ensemble_ref=_ref(
                    "1", "application/vnd.apar.defender-ensemble+json"
                ),
                profile_sha256="2" * 64,
                corpus_envelope_ref=_ref(
                    "3", "application/vnd.apar.corpus-envelope+json"
                ),
                run_ledger_sha256="4" * 64,
                evaluator_verifier=evaluator_verifier,
                pooled_ref=pooled_ref,
                held_family_refs=held_refs,  # type: ignore[arg-type]
                split=split,  # type: ignore[arg-type]
            )

    for update, message in (
        ({"hidden_included": True}, "hidden"),
        ({"ensemble_ref": {
            "media_type": "application/json",
            "relative_path": f"{'8' * 64}/payload",
            "sha256": "8" * 64,
            "size_bytes": 1,
        }}, "reference"),
        ({"profile_sha256": "9" * 64}, "profile"),
    ):
        document = receipt.unsigned_document()
        document.update(update)
        forged = {**document, "signature_base64": signer.sign(document)}
        payload = orchestration.canonical_json_bytes(forged)
        forged_ref = store.put_bytes(
            payload, "application/vnd.apar.development-completion+json"
        )
        with pytest.raises(ValueError, match=message):
            verify_development_completion(
                store=store,
                receipt_ref=forged_ref,
                signer=signer,
                ensemble_ref=_ref("1", "application/vnd.apar.defender-ensemble+json"),
                profile_sha256="2" * 64,
                corpus_envelope_ref=_ref("3", "application/vnd.apar.corpus-envelope+json"),
                run_ledger_sha256="4" * 64,
                evaluator_verifier=evaluator_verifier,
                pooled_ref=pooled_ref,
                held_family_refs=held_refs,  # type: ignore[arg-type]
                split=split,  # type: ignore[arg-type]
            )


def test_competition_evaluator_identities_reject_fixture_and_shared_keys(
    tmp_path: Path,
) -> None:
    root = tmp_path / "root"
    root.mkdir(mode=0o700)
    (root / "run-signing.key").write_bytes(b"p" * 32)
    (root / "evaluator-signing.key").write_bytes(
        hashlib.sha256(b"apar-g3-development-evaluator-v1").digest()
    )
    (root / "hidden-evaluator-signing.key").write_bytes(b"h" * 32)
    for path in root.iterdir():
        path.chmod(0o600)

    with pytest.raises(orchestration.CliContractError, match="fixture"):
        orchestration._load_competition_evaluator_identities(root)

    (root / "evaluator-signing.key").write_bytes(b"e" * 32)
    (root / "hidden-evaluator-signing.key").write_bytes(b"e" * 32)
    with pytest.raises(orchestration.CliContractError, match="distinct"):
        orchestration._load_competition_evaluator_identities(root)

    development_fixture = hashlib.sha256(
        b"apar-g3-development-evaluator-v1"
    ).digest()
    hidden_fixture = hashlib.sha256(b"apar-g3-hidden-evaluator-v1").digest()
    (root / "evaluator-signing.key").write_bytes(hidden_fixture)
    (root / "hidden-evaluator-signing.key").write_bytes(b"h" * 32)
    with pytest.raises(orchestration.CliContractError, match="fixture"):
        orchestration._load_competition_evaluator_identities(root)
    (root / "evaluator-signing.key").write_bytes(b"e" * 32)
    (root / "hidden-evaluator-signing.key").write_bytes(development_fixture)
    with pytest.raises(orchestration.CliContractError, match="fixture"):
        orchestration._load_competition_evaluator_identities(root)


def test_development_identity_loading_never_opens_hidden_private_key(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "root"
    root.mkdir(mode=0o700)
    for name, seed in (
        ("run-signing.key", b"p" * 32),
        ("evaluator-signing.key", b"e" * 32),
        ("hidden-evaluator-signing.key", b"h" * 32),
    ):
        path = root / name
        path.write_bytes(seed)
        path.chmod(0o600)
    opened: list[Path] = []
    real_read = Path.read_bytes

    def read_spy(path: Path) -> bytes:
        opened.append(path)
        if path.name == "hidden-evaluator-signing.key":
            raise AssertionError("development opened hidden authority state")
        return real_read(path)

    monkeypatch.setattr(Path, "read_bytes", read_spy)
    identity = orchestration._load_competition_evaluator_identity(root)
    assert identity.key_id
    assert root / "hidden-evaluator-signing.key" not in opened


def test_competition_publisher_executes_exact_matrix_then_extends_frozen_development(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exercise production routing; hidden must append, never rerun development."""
    store = ArtifactStore(tmp_path / "artifacts")
    publication_signer = RunSigningIdentity.from_private_bytes(b"p" * 32)
    evaluator_signer = EvaluatorSigningIdentity.from_private_bytes(b"e" * 32)
    hidden_signer = EvaluatorSigningIdentity.from_private_bytes(b"h" * 32)
    hidden_context_signer = RunSigningIdentity.from_private_bytes(b"h" * 32)
    source_context = _empty_hidden_context()
    corpus = FrozenCorpus(
        observations=source_context.observations,
        truth=source_context.truth,
        manifest=CorpusManifest(
            profile_id="competition-v1",
            run_ids=("run-development",),
            run_lineage_digests=("9" * 64,),
            observation_count=1,
            truth_count=1,
        ),
    )
    development_ids = ("development-1", "development-2")
    cohorts = tuple(EntityCohort)
    split = SimpleNamespace(
        row_ids={"development": development_ids},
        entity_cohorts={
            "development-1": cohorts[:-1],
            "development-2": (),
        },
        config=SimpleNamespace(),
    )
    pooled_ref = _ref("1", "application/vnd.apar.defender-bundle+json")
    held_refs = {
        family: _ref(character, "application/vnd.apar.defender-bundle+json")
        for family, character in zip(
            (
                "agentic_intent_abuse",
                "app_scam_mule",
                "card_testing_cnp",
                "synthetic_merchant_refund",
            ),
            "2345",
            strict=True,
        )
    }
    candidate_by_digest: dict[str, object] = {}
    for reference in (pooled_ref, *held_refs.values()):
        threshold = SimpleNamespace(
            threshold_set_digest=hashlib.sha256(reference.sha256.encode()).hexdigest(),
            model_dump=lambda **_kwargs: {"schema_version": "1.0.0"},
        )
        candidate_by_digest[reference.sha256] = SimpleNamespace(
            reference=reference,
            thresholds=threshold,
            threshold_labels=object(),
            threshold_values=None,
            attestation=object(),
            defender=SimpleNamespace(
                catalog=object(),
                manifest=SimpleNamespace(
                    frozen_at=datetime(2027, 1, 1, tzinfo=UTC)
                ),
            ),
        )

    replay_calls: list[tuple[EvaluationDescriptor, str, str]] = []
    empty_calls: list[EntityCohort] = []
    envelopes: list[tuple[object, ...]] = []
    published_scopes: list[tuple[str, ...]] = []
    development_builds = 0

    def family_split(_split: object, family: str) -> object:
        return SimpleNamespace(
            row_ids={"development": development_ids},
            entity_cohorts=split.entity_cohorts,
            held_out_evaluation_row_ids=(f"held-{family}",),
            config=split.config,
        )

    def batch(descriptor: EvaluationDescriptor) -> object:
        return SimpleNamespace(results=(SimpleNamespace(evaluation=descriptor),))

    def replay_component(**kwargs: object) -> tuple[object, object]:
        descriptor = kwargs["descriptor"]
        candidate = kwargs["candidate"]
        assert type(descriptor) is EvaluationDescriptor
        reference = candidate.reference  # type: ignore[attr-defined]
        threshold = candidate.thresholds  # type: ignore[attr-defined]
        replay_calls.append(
            (descriptor, reference.sha256, threshold.threshold_set_digest)
        )
        return batch(descriptor), SimpleNamespace(observations=(), as_of=datetime.now(UTC))

    def empty_component(**kwargs: object) -> object:
        cohort = kwargs["cohort"]
        assert type(cohort) is EntityCohort
        empty_calls.append(cohort)
        return batch(
            EvaluationDescriptor(kind=EvaluationKind.COLD_ENTITY, value=cohort.value)
        )

    real_development_components = competition._competition_development_components

    def development_components(**kwargs: object) -> object:
        nonlocal development_builds
        development_builds += 1
        return real_development_components(**kwargs)  # type: ignore[arg-type]

    fake_matrix = SimpleNamespace(
        rows=tuple(SimpleNamespace(event_id=item) for item in development_ids)
    )
    class FakeEnvelope:
        def __init__(self, components: tuple[object, ...], proofs: tuple[object, ...]):
            self.component_batches = components
            self.hidden_proofs = proofs
            self.envelope_digest = hashlib.sha256(
                str(len(components)).encode()
            ).hexdigest()

        @classmethod
        def create(cls, **kwargs: object) -> FakeEnvelope:
            components = kwargs["component_batches"]
            proofs = kwargs["hidden_proofs"]
            assert type(components) is tuple and type(proofs) is tuple
            envelopes.append(components)
            return cls(components, proofs)

    class FakeRequest:
        loaded: FakeRequest | None = None

        def __init__(self, **kwargs: object):
            self.promotion_envelope = kwargs["promotion_envelope"]
            self.champion_decision = kwargs["champion_decision"]
            self.metric_evidence = kwargs["metric_evidence"]
            self.threshold_set = kwargs["threshold_set"]
            FakeRequest.loaded = self

        def to_worker_json(self) -> bytes:
            return b"frozen-development-request"

        @classmethod
        def from_worker_json(cls, _payload: bytes) -> FakeRequest:
            assert cls.loaded is not None
            return cls.loaded

    monkeypatch.setattr(competition, "DefenderBundleVerifier", lambda *a, **k: object())
    monkeypatch.setattr(
        competition,
        "_candidate_runtime",
        lambda **kwargs: candidate_by_digest[kwargs["reference"].sha256],
    )
    monkeypatch.setattr(competition, "make_leave_one_family_out", family_split)
    monkeypatch.setattr(competition, "_replay_component", replay_component)
    monkeypatch.setattr(competition, "_empty_cold_component", empty_component)
    monkeypatch.setattr(competition, "_evaluation_matrix", lambda **_kwargs: fake_matrix)
    monkeypatch.setattr(competition, "_benign_control_corpus", lambda *_a: corpus)
    monkeypatch.setattr(
        competition,
        "_regime_specs",
        lambda *_a: tuple(SimpleNamespace(kind=item) for item in RegimeKind),
    )
    monkeypatch.setattr(competition, "derive_regime", lambda *_a, **_k: (corpus, object()))
    monkeypatch.setattr(competition, "make_evaluation_split", lambda *_a: split)
    monkeypatch.setattr(
        competition,
        "ReplayRegimeEvidence",
        SimpleNamespace(create=lambda **_kwargs: object()),
    )
    monkeypatch.setattr(
        competition, "_competition_development_components", development_components
    )
    monkeypatch.setattr(competition, "VerifiedPromotionEnvelope", FakeEnvelope)
    decision = SimpleNamespace(status=SimpleNamespace(value="no_promotion"))
    monkeypatch.setattr(competition, "evaluate_promotion_gates", lambda *_a, **_k: decision)
    monkeypatch.setattr(competition, "bind_replay_case_counter", lambda *_a, **_k: object())
    monkeypatch.setattr(competition, "_metric_publication_evidence", lambda **_k: ())
    monkeypatch.setattr(
        competition,
        "ReplayCorpusEvidence",
        SimpleNamespace(create=lambda **_kwargs: object()),
    )
    corpus_ref = store.put_bytes(b"corpus-attestation", "application/json")
    monkeypatch.setattr(
        competition,
        "publish_corpus_attestation",
        lambda *_a, **_k: (object(), corpus_ref),
    )
    monkeypatch.setattr(competition, "verify_evaluation_inputs", lambda **_k: object())
    monkeypatch.setattr(competition, "ScorecardPublicationRequest", FakeRequest)
    monkeypatch.setattr(competition, "publish_scorecard", lambda *_a, **_k: (object(), object()))
    receipt_ref = store.put_bytes(b"receipt", "application/json")
    monkeypatch.setattr(
        competition, "store_restricted_publication_receipt", lambda *_a, **_k: receipt_ref
    )

    def published_result(*_args: object, **kwargs: object) -> object:
        scope = kwargs["descriptor_scope"]
        assert type(scope) is tuple
        published_scopes.append(scope)
        return SimpleNamespace(
            development_evidence_ref=kwargs["development_evidence_ref"]
        )

    monkeypatch.setattr(competition, "_published_result", published_result)

    development = publish_competition_evaluation(
        store=store,
        publication_signer=publication_signer,
        evaluator_signer=evaluator_signer,
        hidden_signer=None,
        pooled_ref=pooled_ref,
        held_family_refs=held_refs,  # type: ignore[arg-type]
        corpus=corpus,
        split=split,  # type: ignore[arg-type]
        profile_sha256="a" * 64,
        authenticated_run_ids=corpus.manifest.run_ids,
    )

    assert development_builds == 1
    assert len(envelopes[-1]) == 16
    assert len(published_scopes[-1]) == 16
    assert empty_calls == [cohorts[-1]]
    for descriptor, reference_digest, threshold_digest in replay_calls:
        expected_ref = (
            held_refs[descriptor.value]  # type: ignore[index]
            if descriptor.kind is EvaluationKind.HELD_FAMILY
            else pooled_ref
        )
        expected_candidate = candidate_by_digest[expected_ref.sha256]
        assert reference_digest == expected_ref.sha256
        assert threshold_digest == expected_candidate.thresholds.threshold_set_digest  # type: ignore[attr-defined]

    hidden_observation = source_context.observations[0].model_copy(
        update={"event_id": "isolated-hidden", "payment_id": "isolated-hidden-payment"}
    )
    hidden_batch = batch(EvaluationDescriptor(kind=EvaluationKind.HIDDEN, value="hidden"))

    class FakeHiddenOutcome:
        def __init__(self) -> None:
            self.batch = hidden_batch
            self.public_proof = object()

    monkeypatch.setattr(competition, "HiddenReplayOutcome", FakeHiddenOutcome)
    monkeypatch.setattr(
        competition,
        "_verify_frozen_development_request",
        lambda *_a, **_k: None,
    )
    monkeypatch.setattr(
        competition,
        "verify_hidden_context",
        lambda **kwargs: (
            SimpleNamespace(as_of=datetime(2027, 2, 1, tzinfo=UTC)),
            (hidden_observation,),
            _ref("8", "application/vnd.apar.hidden-evaluation-context+json"),
        ),
    )
    monkeypatch.setattr(
        competition,
        "build_feature_matrix",
        lambda *_a: SimpleNamespace(
            rows=(SimpleNamespace(event_id=hidden_observation.event_id),)
        ),
    )

    class FakeAuthority:
        def __init__(self, *_args: object):
            pass

        def freeze_and_issue(self, *_args: object, **_kwargs: object) -> object:
            return object()

    monkeypatch.setattr(competition, "HiddenEvaluationAuthority", FakeAuthority)
    monkeypatch.setattr(
        competition, "replay_defense_arms", lambda **_kwargs: FakeHiddenOutcome()
    )
    hidden_context_ref = _ref(
        "7", "application/vnd.apar.restricted-hidden-context-envelope+json"
    )
    publish_competition_evaluation(
        store=store,
        publication_signer=publication_signer,
        evaluator_signer=evaluator_signer,
        hidden_signer=hidden_signer,
        pooled_ref=pooled_ref,
        held_family_refs=held_refs,  # type: ignore[arg-type]
        corpus=corpus,
        split=split,  # type: ignore[arg-type]
        profile_sha256="a" * 64,
        authenticated_run_ids=corpus.manifest.run_ids,
        hidden_context_ref=hidden_context_ref,
        hidden_context_signer=hidden_context_signer,
        development_evidence_ref=development.development_evidence_ref,
    )
    assert development_builds == 1
    assert len(envelopes[-1]) == 17
    assert len(published_scopes[-1]) == 17
