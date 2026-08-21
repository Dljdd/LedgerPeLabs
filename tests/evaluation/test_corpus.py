"""Verified development-corpus boundary behavior."""

from __future__ import annotations

import json
from datetime import timedelta
from decimal import Decimal
from pathlib import Path

import pytest

from apar.compiler import compile_scenario
from apar.contracts.events import EventKind, Rail
from apar.contracts.scenarios import AttackerMode
from apar.evaluation.contracts import CorpusProfile
from apar.evaluation.corpus import CorpusVerificationError, assemble_verified_corpus
from apar.runs import (
    AttackerPolicy,
    AttackerPolicyKind,
    RunManifest,
    RunRunner,
    RunSigningIdentity,
    bind_scenario_for_run,
)
from apar.storage.artifacts import ArtifactRef, ArtifactStore
from tests.factories import NOW, make_payment_event, make_scenario_config, make_threat_card


class RecordingArtifactStore(ArtifactStore):
    """Real artifact verification with a record of corpus-parser reads."""

    def __init__(self, root: Path) -> None:
        super().__init__(root)
        self.read_names: list[str] = []
        self._names_by_digest: dict[str, str] = {}

    def name(self, name: str, ref: ArtifactRef) -> None:
        self._names_by_digest[ref.sha256] = name

    def read(self, ref: ArtifactRef) -> bytes:
        self.read_names.append(self._names_by_digest.get(ref.sha256, "unknown"))
        return super().read(ref)


def _manifest(
    store: ArtifactStore,
    *,
    events: list[dict[str, object]],
    entities: list[dict[str, object]],
    family: str = "card_testing_cnp",
    run_id: str = "run-fixture",
) -> RunManifest:
    artifacts = {
        "events": store.put_bytes(
            json.dumps(events, sort_keys=True, separators=(",", ":")).encode(),
            "application/vnd.apar.events+json",
        ),
        "population": store.put_bytes(
            json.dumps({"entities": entities}, sort_keys=True, separators=(",", ":")).encode(),
            "application/vnd.apar.population+json",
        ),
        "summary": store.put_json({"family": family}),
    }
    return RunManifest.model_construct(
        run_id=run_id,
        scenario_id="scenario-fixture",
        artifacts=artifacts,
        lineage_digest="0" * 64,
    )


def _wire_event(**updates: object) -> dict[str, object]:
    return (
        make_payment_event(
            rail_data={"payment_id": "pay-1"},
            lineage={"synthetic": True},
        )
        .model_copy(update=updates)
        .model_dump(mode="json")
    )


def _population_entities() -> list[dict[str, object]]:
    event = make_payment_event()
    return [
        {"entity_id": event.actor_id, "illicit": True},
        {"entity_id": event.counterparty_id, "illicit": False},
    ]


def _completed_manifest(tmp_path: Path) -> tuple[ArtifactStore, RunRunner, RunManifest]:
    store = ArtifactStore(tmp_path / "artifacts")
    runner = RunRunner(
        store,
        RunSigningIdentity.from_private_bytes(bytes(range(32))),
        tmp_path / "runs",
    )
    config = make_scenario_config(
        rail=Rail.A2A,
        query_budget=1,
        seed=960,
        replay=make_scenario_config().replay.model_copy(update={"random_seed": 960}),
        benign_entity_count=40,
        illicit_entity_count=16,
    )
    card = make_threat_card(rails=[Rail.A2A], default_config=config)
    bundle = bind_scenario_for_run(compile_scenario(card, config), threat_family=card.family)
    manifest = runner.execute(
        bundle,
        AttackerPolicy(
            attacker_mode=AttackerMode.DECISION_ONLY,
            family="app_scam_mule",
            kind=AttackerPolicyKind.FIXED,
            query_budget=1,
            worker_timeout_ms=5_000,
        ),
    )
    return store, runner, manifest


def _fixture_runner(
    store: ArtifactStore, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> RunRunner:
    runner = RunRunner(
        store,
        RunSigningIdentity.from_private_bytes(bytes(range(32))),
        tmp_path / "runs",
    )
    monkeypatch.setattr(runner, "verify_run", lambda _manifest: True)
    return runner


def test_corpus_rejects_a_manifest_that_no_longer_verifies(
    tmp_path: Path,
) -> None:
    store, runner, valid = _completed_manifest(tmp_path)
    changed = valid.model_copy(update={"lineage_digest": "0" * 64})

    with pytest.raises(CorpusVerificationError, match="authenticated run"):
        assemble_verified_corpus([changed], runner, store, CorpusProfile.fixture())


def test_development_corpus_parser_reads_only_declared_public_artifacts(tmp_path: Path) -> None:
    _authoritative_store, runner, manifest = _completed_manifest(tmp_path)
    parser_store = RecordingArtifactStore(tmp_path / "artifacts")
    for name, ref in manifest.artifacts.items():
        parser_store.name(name, ref)

    assemble_verified_corpus([manifest], runner, parser_store, CorpusProfile.fixture())

    assert set(parser_store.read_names) == {"events", "population", "summary"}


def test_corpus_rejects_duplicate_event_ids_and_payment_openings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = ArtifactStore(tmp_path / "artifacts")
    runner = _fixture_runner(store, tmp_path, monkeypatch)
    first = _wire_event()
    duplicate_event = _wire_event(event_id=first["event_id"], rail_data={"payment_id": "pay-2"})
    manifest = _manifest(
        store,
        events=[first, duplicate_event],
        entities=_population_entities(),
    )

    with pytest.raises(CorpusVerificationError, match="duplicate event ID"):
        assemble_verified_corpus([manifest], runner, store, CorpusProfile.fixture())

    second_opening = _wire_event(
        event_id="00000000-0000-4000-8000-000000000006",
        rail_data={"payment_id": "pay-1"},
    )
    opening_manifest = _manifest(
        store,
        events=[first, second_opening],
        entities=_population_entities(),
    )
    with pytest.raises(CorpusVerificationError, match="duplicate payment opening"):
        assemble_verified_corpus([opening_manifest], runner, store, CorpusProfile.fixture())


def test_corpus_rejects_payment_ids_reused_across_campaigns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = ArtifactStore(tmp_path / "artifacts")
    runner = _fixture_runner(store, tmp_path, monkeypatch)
    first = _manifest(
        store,
        events=[_wire_event()],
        entities=_population_entities(),
    )
    second_event = _wire_event(
        event_id="00000000-0000-4000-8000-000000000006",
        campaign_id="00000000-0000-4000-8000-000000000007",
    )
    second = _manifest(
        store,
        events=[second_event],
        entities=_population_entities(),
        run_id="run-fixture-2",
    )

    with pytest.raises(CorpusVerificationError, match="payment ID reused across campaigns"):
        assemble_verified_corpus([first, second], runner, store, CorpusProfile.fixture())


def test_corpus_rejects_lifecycle_events_without_exactly_one_opening(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = ArtifactStore(tmp_path / "artifacts")
    runner = _fixture_runner(store, tmp_path, monkeypatch)
    lifecycle = _wire_event(
        event_type=EventKind.SETTLEMENT,
        decision_at=None,
        event_id="00000000-0000-4000-8000-000000000006",
    )
    manifest = _manifest(
        store,
        events=[lifecycle],
        entities=_population_entities(),
    )

    with pytest.raises(CorpusVerificationError, match="exactly one opening"):
        assemble_verified_corpus([manifest], runner, store, CorpusProfile.fixture())


def test_corpus_rejects_non_synthetic_events(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = ArtifactStore(tmp_path / "artifacts")
    runner = _fixture_runner(store, tmp_path, monkeypatch)
    manifest = _manifest(
        store,
        events=[_wire_event(lineage={"synthetic": False})],
        entities=_population_entities(),
    )

    with pytest.raises(CorpusVerificationError, match="synthetic"):
        assemble_verified_corpus([manifest], runner, store, CorpusProfile.fixture())


def test_corpus_rejects_missing_population_truth_for_a_referenced_entity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = ArtifactStore(tmp_path / "artifacts")
    runner = _fixture_runner(store, tmp_path, monkeypatch)
    manifest = _manifest(
        store,
        events=[_wire_event()],
        entities=[{"entity_id": make_payment_event().counterparty_id, "illicit": False}],
    )

    with pytest.raises(CorpusVerificationError, match="population truth"):
        assemble_verified_corpus([manifest], runner, store, CorpusProfile.fixture())


def test_corpus_rejects_mixed_rail_events_under_one_family(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = ArtifactStore(tmp_path / "artifacts")
    runner = _fixture_runner(store, tmp_path, monkeypatch)
    a2a_event = _wire_event(
        event_id="00000000-0000-4000-8000-000000000006",
        rail=Rail.A2A,
        event_type=EventKind.TRANSFER_INITIATED,
        rail_data={"payment_id": "pay-2"},
    )
    manifest = _manifest(
        store,
        events=[_wire_event(), a2a_event],
        entities=[
            {"entity_id": make_payment_event().actor_id, "illicit": True},
            {"entity_id": make_payment_event().counterparty_id, "illicit": False},
        ],
    )

    with pytest.raises(CorpusVerificationError, match="declared rail"):
        assemble_verified_corpus([manifest], runner, store, CorpusProfile.fixture())


def test_corpus_derives_isolated_truth_from_lifecycle_and_label_delay(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = ArtifactStore(tmp_path / "artifacts")
    runner = _fixture_runner(store, tmp_path, monkeypatch)
    opening = _wire_event()
    settlement = _wire_event(
        event_id="00000000-0000-4000-8000-000000000006",
        event_type=EventKind.SETTLEMENT,
        decision_at=None,
        event_time=NOW + timedelta(days=9),
        ingested_at=NOW + timedelta(days=9, milliseconds=25),
        available_at=NOW + timedelta(days=9, milliseconds=25),
        amount=Decimal("10.00"),
    )
    refund = _wire_event(
        event_id="00000000-0000-4000-8000-000000000007",
        event_type=EventKind.REFUND,
        decision_at=None,
        event_time=NOW + timedelta(days=10),
        ingested_at=NOW + timedelta(days=10, milliseconds=25),
        available_at=NOW + timedelta(days=10, milliseconds=25),
        amount=Decimal("3.00"),
    )
    manifest = _manifest(
        store,
        events=[opening, settlement, refund],
        entities=_population_entities(),
    )

    corpus = assemble_verified_corpus([manifest], runner, store, CorpusProfile.fixture())

    assert len(corpus.observations) == 3
    assert len(corpus.truth) == 1
    assert {"is_fraud", "campaign_id", "family", "label_source"}.isdisjoint(
        corpus.observations[0].model_dump()
    )
    assert corpus.truth[0].net_settled_value == Decimal("7.00")
    assert corpus.truth[0].first_settlement_at == NOW + timedelta(days=9)
    assert corpus.truth[0].label_mature_at == NOW + timedelta(days=9)
