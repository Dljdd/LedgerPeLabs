"""Task 15 named-export compatibility must preserve Task 14's signed refs."""

from __future__ import annotations

import ast
import base64
import hashlib
import inspect
import json
import os
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]
import pytest

from apar.contracts.events import EventKind, Rail
from apar.defense import orchestration
from apar.defense.contracts import ObservedEvent
from apar.evaluation.contracts import (
    CorpusManifest,
    EvaluationTruthRow,
    FrozenCorpus,
)
from apar.evaluation.gates import EvaluatorSigningIdentity, HiddenPublicProof
from apar.evaluation.hidden_source import HiddenSourceWorkerBinding
from apar.evaluation.reporting import PublicArtifactVerifier
from apar.runs import RunSigningIdentity
from apar.runs.wire import canonical_json_bytes
from apar.storage.artifacts import ArtifactRef, ArtifactStore

ROOT = Path(__file__).resolve().parents[2]
PREREGISTRATION = ROOT / "docs" / "experiments" / "defense-v1-preregistration.json"
PROFILE = ROOT / "config" / "defense" / "competition-profile.json"


def test_authority_provisioning_is_one_shot_private_and_secret_silent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    repository = (tmp_path / "repository").resolve()
    root = repository / ".apar" / "defense-v1"
    root.mkdir(mode=0o700, parents=True)
    monkeypatch.setattr(orchestration, "_REPOSITORY_ROOT", repository)

    code = orchestration.command_main(
        "provision_defense_authorities", ["--root", ".apar/defense-v1"]
    )

    assert code == 0
    output = capsys.readouterr().out
    assert output.count("\n") == 1
    document = json.loads(output)
    assert set(document) == {"authority_identities", "schema_version"}
    identities = document["authority_identities"]
    assert set(identities) == {
        "development_evaluator",
        "hidden_evaluator",
        "hidden_source",
        "publication",
    }
    assert all(
        set(identity) == {"key_id", "public_key_base64"}
        for identity in identities.values()
    )
    assert len({identity["key_id"] for identity in identities.values()}) == 4
    expected_files = {
        "evaluator-signing.key",
        "hidden-evaluator-signing.key",
        "hidden-run-signing.key",
        "hidden-run-signing.pub",
        "run-signing.key",
    }
    assert {path.name for path in root.iterdir()} == expected_files
    before = {path.name: path.read_bytes() for path in root.iterdir()}
    for path in root.iterdir():
        assert path.stat().st_mode & 0o777 == 0o600
        assert len(path.read_bytes()) == 32
        assert path.read_bytes() not in output.encode()
    role_files = {
        "development_evaluator": "evaluator-signing.key",
        "hidden_evaluator": "hidden-evaluator-signing.key",
        "hidden_source": "hidden-run-signing.key",
        "publication": "run-signing.key",
    }
    for role, filename in role_files.items():
        identity = RunSigningIdentity.from_private_bytes(before[filename])
        public = base64.b64decode(
            identities[role]["public_key_base64"], validate=True
        )
        assert len(public) == 32
        assert hashlib.sha256(public).hexdigest() == identities[role]["key_id"]
        assert identity.key_id == identities[role]["key_id"]
        assert identity.public_key_base64 == identities[role]["public_key_base64"]
    assert base64.b64decode(
        identities["hidden_source"]["public_key_base64"], validate=True
    ) == before["hidden-run-signing.pub"]

    assert (
        orchestration.command_main(
            "provision_defense_authorities", ["--root", ".apar/defense-v1"]
        )
        == 2
    )
    assert "already provisioned" in capsys.readouterr().err
    assert {path.name: path.read_bytes() for path in root.iterdir()} == before

    alternate = (tmp_path / "alternate").resolve()
    alternate.mkdir(mode=0o700)
    assert (
        orchestration.command_main(
            "provision_defense_authorities", ["--root", str(alternate)]
        )
        == 2
    )
    assert tuple(alternate.iterdir()) == ()


def test_authority_provisioning_preflights_every_final_name(tmp_path: Path) -> None:
    root = (tmp_path / "authorities").resolve()
    root.mkdir(mode=0o700)
    occupied = root / "hidden-run-signing.pub"
    occupied.write_bytes(b"occupied")
    occupied.chmod(0o600)

    with pytest.raises(orchestration.CliContractError, match="already provisioned"):
        orchestration.provision_defense_authorities(root)

    assert {path.name for path in root.iterdir()} == {"hidden-run-signing.pub"}
    assert occupied.read_bytes() == b"occupied"


def test_authority_provisioning_cleans_temps_and_finals_on_every_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepublish_root = (tmp_path / "prepublish").resolve()
    prepublish_root.mkdir(mode=0o700)
    original_write = os.write
    failed = False

    def failing_write(descriptor: int, payload: bytes) -> int:
        nonlocal failed
        if not failed:
            failed = True
            raise OSError("injected prepublish failure")
        return original_write(descriptor, payload)

    monkeypatch.setattr(orchestration.os, "write", failing_write)
    with pytest.raises(orchestration.CliContractError, match="failed closed"):
        orchestration.provision_defense_authorities(prepublish_root)
    assert tuple(prepublish_root.iterdir()) == ()

    monkeypatch.setattr(orchestration.os, "write", original_write)
    publish_root = (tmp_path / "publish").resolve()
    publish_root.mkdir(mode=0o700)
    original_publish = ArtifactStore.publish_no_replace_at
    calls = 0

    def failing_publish(directory_fd: int, source: str, target: str) -> None:
        nonlocal calls
        calls += 1
        if calls == 3:
            raise OSError("injected publish failure")
        original_publish(directory_fd, source, target)

    monkeypatch.setattr(
        ArtifactStore, "publish_no_replace_at", staticmethod(failing_publish)
    )
    with pytest.raises(orchestration.CliContractError, match="failed closed"):
        orchestration.provision_defense_authorities(publish_root)
    assert tuple(publish_root.iterdir()) == ()


def test_private_authority_loaders_are_descriptor_pinned_and_close_fds(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = (tmp_path / "runtime").resolve()
    root.mkdir(mode=0o700)
    key = root / "run-signing.key"
    key.write_bytes(b"s" * 32)
    key.chmod(0o600)
    assert orchestration._load_standard_signer(root).key_id == (
        RunSigningIdentity.from_private_bytes(b"s" * 32).key_id
    )

    key.chmod(0o644)
    with pytest.raises(orchestration.CliContractError, match="owner-only"):
        orchestration._load_standard_signer(root)
    key.unlink()
    target = root / "target.key"
    target.write_bytes(b"t" * 32)
    target.chmod(0o600)
    key.symlink_to(target)
    with pytest.raises(orchestration.CliContractError, match="cannot be read"):
        orchestration._load_standard_signer(root)
    key.unlink()
    key.write_bytes(b"s" * 32)
    key.chmod(0o600)

    original_read = os.read
    swapped = False

    def swap_during_read(descriptor: int, size: int) -> bytes:
        nonlocal swapped
        payload = original_read(descriptor, size)
        if payload and not swapped:
            swapped = True
            key.rename(root / "old.key")
            key.write_bytes(b"a" * 32)
            key.chmod(0o600)
        return payload

    monkeypatch.setattr(orchestration.os, "read", swap_during_read)
    with pytest.raises(orchestration.CliContractError, match="changed during read"):
        orchestration._load_standard_signer(root)
    monkeypatch.setattr(orchestration.os, "read", original_read)

    evaluator = root / "evaluator-signing.key"
    evaluator.write_bytes(b"e" * 32)
    evaluator.chmod(0o600)
    assert orchestration._load_evaluator_seed(root, evaluator.name) == b"e" * 32
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

    monkeypatch.setattr(orchestration.os, "open", recording_open)
    monkeypatch.setattr(orchestration.os, "close", recording_close)
    monkeypatch.setattr(
        orchestration.os,
        "read",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("injected")),
    )
    with pytest.raises(orchestration.CliContractError, match="cannot be read"):
        orchestration._load_evaluator_seed(root, evaluator.name)
    assert opened
    assert set(opened) <= set(closed)


def test_production_authority_keys_have_no_path_read_bypass() -> None:
    tree = ast.parse(inspect.getsource(orchestration))
    authority_names = {
        "evaluator-signing.key",
        "hidden-evaluator-signing.key",
        "hidden-run-signing.key",
        "hidden-run-signing.pub",
        "run-signing.key",
    }
    violations = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "read_bytes"
        ):
            rendered = ast.unparse(node.func.value)
            if any(name in rendered for name in authority_names):
                violations.append(rendered)
    assert violations == []


def test_unpreregistered_publication_key_blocks_before_first_campaign(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = (tmp_path / "runtime").resolve()
    root.mkdir(mode=0o700)
    key = root / "run-signing.key"
    key.write_bytes(b"x" * 32)
    key.chmod(0o600)
    campaign_calls: list[object] = []

    def unexpected_campaign(*args: object, **kwargs: object) -> object:
        campaign_calls.append((args, kwargs))
        raise AssertionError("campaign started before authority verification")

    monkeypatch.setattr(orchestration, "_run_one_campaign", unexpected_campaign)
    with pytest.raises(orchestration.CliContractError, match="preregistration"):
        orchestration._generate_competition_runs(
            profile=orchestration.CompetitionProfile.fixture(),
            root=root,
            signer_path=key,
            output_name="ledger.json",
            enforce_preregistered_authorities=True,
        )
    assert campaign_calls == []
    assert {path.name for path in root.iterdir()} == {"run-signing.key", "artifacts"}


def test_preregistration_is_exact_canonical_and_result_free(tmp_path: Path) -> None:
    preregistration = orchestration.load_defense_v1_preregistration(PREREGISTRATION)

    assert preregistration.preregistration_id == "defense-v1"
    assert preregistration.profile_sha256 == (
        "f91c36e0329ef46631826a84d33b46282567069410cc9dc2c17694fe7463d7b1"
    )
    assert len(preregistration.campaigns) == 200
    assert [campaign.seed for campaign in preregistration.campaigns[:50]] == list(
        range(260000, 260050)
    )
    assert preregistration.campaigns[-1].seed == 263049
    assert preregistration.result_fields_forbidden is True
    assert {
        role: {
            "key_id": identity.key_id,
            "public_key_base64": identity.public_key_base64,
        }
        for role, identity in preregistration.authority_identities.items()
    } == {
        "development_evaluator": {
            "key_id": "dd0ec0d9cbcf54f1b31a60f413a652e4bd19c7e342b32fb6fbb179d99bdc5823",
            "public_key_base64": "+Qo1uRXaRg1qcNourulW1HWGmtE33TGZgbK/72n5BTU=",
        },
        "hidden_evaluator": {
            "key_id": "b1dfc2c14cbe5f00a68dd4ec86344b73334ff804ec88f3879be9055876418bc2",
            "public_key_base64": "JirafB3LSBGts17ubm3CBsL4gE/w7Muy0sRCTZvkITs=",
        },
        "hidden_source": {
            "key_id": "a8e5587cb75a5216916de427efbc945dc4dd24dc1bd82bf56167571323ac665e",
            "public_key_base64": "ONFSJ5inhXjgmLssoWgttBzBFwXubNabdglWwWCdvEE=",
        },
        "publication": {
            "key_id": "35291f5d85b3b5ab3da466597aad30222f49a5cbf03ebcbcac5d43c6ab7235e2",
            "public_key_base64": "q2hK2MIxev3jIooJW1Ir2QujTExk6suy+zjO+UwNHIg=",
        },
    }

    document = json.loads(PREREGISTRATION.read_text(encoding="utf-8"))
    attacks = (
        {**document, "profile_sha256": "0" * 64},
        {
            **document,
            "campaigns": [
                {**document["campaigns"][0], "seed": 999999},
                *document["campaigns"][1:],
            ],
        },
        {
            **document,
            "model_selection": {
                **document["model_selection"],
                "fpr_probability_threshold": 0.51,
            },
        },
        {
            **document,
            "threshold_selection": {
                **document["threshold_selection"],
                "tie_breaks": ["higher_challenge_threshold"],
            },
        },
        {
            **document,
            "gates": {**document["gates"], "maximum_ece": 0.11},
        },
        {
            **document,
            "campaigns": [
                {
                    **document["campaigns"][0],
                    "simulation_start_utc": "2026-01-01T00:00:01Z",
                },
                *document["campaigns"][1:],
            ],
        },
        {
            **document,
            "artifact_paths": [*document["artifact_paths"], "results/escape.json"],
        },
        {
            **document,
            "authority_identities": {
                **document["authority_identities"],
                "publication": document["authority_identities"][
                    "development_evaluator"
                ],
            },
        },
        {**document, "observed_result": {"recall": 1.0}},
    )
    for index, attack in enumerate(attacks):
        path = tmp_path / f"attack-{index}.json"
        path.write_text(
            json.dumps(attack, sort_keys=True, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )
        with pytest.raises(orchestration.CliContractError, match="preregistration"):
            orchestration.load_defense_v1_preregistration(path)

    noncanonical = tmp_path / "pretty.json"
    noncanonical.write_text(json.dumps(document, indent=2), encoding="utf-8")
    with pytest.raises(orchestration.CliContractError, match="preregistration"):
        orchestration.load_defense_v1_preregistration(noncanonical)


def test_named_task15_arguments_are_exact_and_cannot_mix_with_digest_surface() -> None:
    generate = orchestration._parser("generate_defense_runs")
    generated = generate.parse_args(
        [
            "--preregistration",
            str(PREREGISTRATION),
            "--root",
            "/tmp/defense-v1",
            "--signer",
            "/tmp/defense-v1/run-signing-key.ed25519",
            "--output",
            "docs/experiments/defense-v1-run-manifests.json",
        ]
    )
    assert generated.preregistration == str(PREREGISTRATION)
    assert generated.output == "docs/experiments/defense-v1-run-manifests.json"

    build = orchestration._parser("build_defense_corpus")
    built = build.parse_args(
        [
            "--profile",
            str(PROFILE),
            "--run-manifests",
            "docs/experiments/defense-v1-run-manifests.json",
            "--root",
            "/tmp/defense-v1",
            "--output-manifest",
            "fixtures/defense/v1/corpus-manifest.json",
        ]
    )
    assert built.output_manifest == "fixtures/defense/v1/corpus-manifest.json"

    prepare = orchestration._parser("prepare_hidden_context")
    prepared = prepare.parse_args(
        [
            "--corpus",
            "fixtures/defense/v1/corpus-manifest.json",
            "--defender",
            "fixtures/defense/v1/defender-bundle.json",
            "--profile",
            "config/defense/competition-profile.json",
            "--root",
            ".apar/defense-v1",
        ]
    )
    assert prepared.corpus == "fixtures/defense/v1/corpus-manifest.json"
    assert prepared.defender == "fixtures/defense/v1/defender-bundle.json"

    train = orchestration._parser("train_defender")
    trained = train.parse_args(
        [
            "--corpus",
            "fixtures/defense/v1/corpus-manifest.json",
            "--catalog",
            "config/defense/feature-catalog.json",
            "--profile",
            str(PROFILE),
            "--rollback-ref",
            "rules-v1",
            "--root",
            "/tmp/defense-v1",
            "--export",
            "fixtures/defense/v1",
        ]
    )
    assert trained.corpus == "fixtures/defense/v1/corpus-manifest.json"
    assert trained.export == "fixtures/defense/v1"

    with pytest.raises(SystemExit):
        train.parse_args(
            [
                "--corpus",
                "fixtures/defense/v1/corpus-manifest.json",
                "--development-corpus",
                "0" * 64,
                "--catalog",
                "config/defense/feature-catalog.json",
                "--profile",
                str(PROFILE),
                "--rollback-ref",
                "rules-v1",
                "--root",
                "/tmp/defense-v1",
                "--export",
                "fixtures/defense/v1",
            ]
        )


def test_named_artifact_alias_resolves_only_the_signed_pointer(
    tmp_path: Path,
) -> None:
    export = tmp_path / "run-manifests.json"
    signer = RunSigningIdentity.from_private_bytes(b"p" * 32)
    signed_ref = {
        "media_type": "application/json",
        "relative_path": f"{'a' * 64}/payload",
        "sha256": "a" * 64,
        "size_bytes": 123,
    }
    unsigned = {
        "artifact": signed_ref,
        "authenticated_run_ids": [f"run-{index:03d}" for index in range(200)],
        "campaign_count": 200,
        "export_metadata": {},
        "family_counts": {
            "agentic_intent_abuse": 50,
            "app_scam_mule": 50,
            "card_testing_cnp": 50,
            "synthetic_merchant_refund": 50,
        },
        "kind": "run_ledger",
        "preregistration_sha256": (
            "95b460d2bb125e4fd3d432a6d52eb196a8c7e3d72bf5f0cf7d022cf2d9c8b428"
        ),
        "profile_sha256": (
            "f91c36e0329ef46631826a84d33b46282567069410cc9dc2c17694fe7463d7b1"
        ),
        "public_key_base64": signer.public_key_base64,
        "schema_version": "1.0.0",
        "signer_key_id": signer.key_id,
    }
    export.write_text(
        json.dumps(
            {**unsigned, "signature_base64": signer.sign(unsigned)},
            sort_keys=True,
            separators=(",", ":"),
        ),
        encoding="utf-8",
    )

    resolved = orchestration.resolve_defense_v1_alias(
        export,
        expected_kind="run_ledger",
        expected_profile_sha256=(
            "f91c36e0329ef46631826a84d33b46282567069410cc9dc2c17694fe7463d7b1"
        ),
        signer=signer,
    )
    assert resolved.sha256 == "a" * 64

    changed = json.loads(export.read_text(encoding="utf-8"))
    changed["artifact"] = {
        **changed["artifact"],
        "sha256": "b" * 64,
        "relative_path": f"{'b' * 64}/payload",
    }
    attacker = RunSigningIdentity.from_private_bytes(b"q" * 32)
    changed["signer_key_id"] = attacker.key_id
    changed["public_key_base64"] = attacker.public_key_base64
    changed["signature_base64"] = attacker.sign(
        {key: value for key, value in changed.items() if key != "signature_base64"}
    )
    export.write_text(
        json.dumps(changed, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )
    with pytest.raises(orchestration.CliContractError, match="alias"):
        orchestration.resolve_defense_v1_alias(
            export,
            expected_kind="run_ledger",
            expected_profile_sha256=(
                "f91c36e0329ef46631826a84d33b46282567069410cc9dc2c17694fe7463d7b1"
            ),
            signer=signer,
        )


def test_generate_named_command_executes_task14_core_and_publishes_signed_alias(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    repository = tmp_path / "repository"
    output = repository / "docs" / "experiments" / "defense-v1-run-manifests.json"
    output.parent.mkdir(parents=True)
    preregistration = output.parent / "defense-v1-preregistration.json"
    preregistration.write_bytes(PREREGISTRATION.read_bytes())
    runtime = repository / ".apar" / "defense-v1"
    runtime.mkdir(mode=0o700, parents=True)
    seed = b"z" * 32
    path = runtime / "run-signing.key"
    path.write_bytes(seed)
    path.chmod(0o600)
    signer = RunSigningIdentity.from_private_bytes(seed)
    core_calls: list[dict[str, Any]] = []

    def fake_generate(**kwargs: Any) -> object:
        core_calls.append(kwargs)
        profile = kwargs["profile"]
        store = ArtifactStore(runtime / "artifacts")
        entries = tuple(
            orchestration.RunLedgerEntry(
                family=family,
                campaign_index=index,
                seed=profile.campaign_seed(family, index),
                simulation_start_utc=profile.campaign_start(family, index),
                run_id=f"run-{entries_so_far:032x}",
                manifest={
                    "media_type": "application/json",
                    "relative_path": f"{'a' * 64}/payload",
                    "sha256": "a" * 64,
                    "size_bytes": 1,
                },
            )
            for entries_so_far, (family, index) in enumerate(
                (family, index)
                for family in profile.families
                for index in range(profile.campaigns_per_family)
            )
        )
        ledger = orchestration.CompetitionRunLedger(
            profile_sha256=orchestration._DEFENSE_V1_PROFILE_SHA256,
            signer_key_id=signer.key_id,
            public_key_base64=signer.public_key_base64,
            entries=entries,
        )
        return store.put_bytes(
            canonical_json_bytes(ledger.model_dump(mode="json")), "application/json"
        )

    monkeypatch.setattr(orchestration, "_REPOSITORY_ROOT", repository)
    monkeypatch.setattr(orchestration, "_generate_competition_runs", fake_generate)
    code = orchestration.command_main(
        "generate_defense_runs",
        [
            "--preregistration",
            "docs/experiments/defense-v1-preregistration.json",
            "--root",
            ".apar/defense-v1",
            "--signer",
            ".apar/defense-v1/run-signing-key.ed25519",
            "--output",
            "docs/experiments/defense-v1-run-manifests.json",
        ],
    )

    assert code == 0
    assert len(core_calls) == 1
    assert core_calls[0]["output_name"] == "defense-v1-run-ledger.json"
    assert core_calls[0]["enforce_preregistered_authorities"] is True
    assert output.exists()
    alias_ref = orchestration.resolve_defense_v1_alias(
        output,
        expected_kind="run_ledger",
        expected_profile_sha256=orchestration._DEFENSE_V1_PROFILE_SHA256,
        signer=signer,
    )
    store = ArtifactStore(runtime / "artifacts")
    assert store.read(alias_ref)
    stdout = capsys.readouterr().out
    assert stdout.endswith("\n") and stdout.count("\n") == 1
    assert str(runtime) not in stdout
    assert os.fspath(output) not in stdout


def test_train_named_command_resolves_signed_corpus_and_calls_task14_freeze(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = tmp_path / "repository"
    runtime = repository / ".apar" / "defense-v1"
    runtime.mkdir(mode=0o700, parents=True)
    seed = b"t" * 32
    key = runtime / "run-signing.key"
    key.write_bytes(seed)
    key.chmod(0o600)
    signer = RunSigningIdentity.from_private_bytes(seed)
    profile_path = repository / "config" / "defense" / "competition-profile.json"
    catalog_path = repository / "config" / "defense" / "feature-catalog.json"
    profile_path.parent.mkdir(parents=True)
    profile_path.write_bytes(PROFILE.read_bytes())
    catalog_path.write_bytes(
        (ROOT / "config" / "defense" / "feature-catalog.json").read_bytes()
    )
    store = ArtifactStore(runtime / "artifacts")
    corpus_ref = store.put_bytes(b"{}", orchestration._CORPUS_ENVELOPE_MEDIA_TYPE)
    corpus_path = repository / "fixtures" / "defense" / "v1" / "corpus-manifest.json"
    corpus_path.parent.mkdir(parents=True)
    run_ids = tuple(f"run-{index:032x}" for index in range(200))
    orchestration._publish_defense_v1_alias(
        path=corpus_path,
        kind="corpus_envelope",
        reference=corpus_ref,
        signer=signer,
        authenticated_run_ids=run_ids,
        export_metadata={
            "observation_dataset": {
                "classification": "defender_visible",
                "content_digest": "1" * 64,
                "file_sha256": "2" * 64,
                "row_count": 1,
            },
            "truth_dataset": {
                "classification": "restricted_evaluator_only",
                "content_digest": "3" * 64,
                "file_sha256": "4" * 64,
                "row_count": 1,
            },
        },
    )
    ensemble_ref = store.put_bytes(
        b'{"ensemble":true}', orchestration._DEFENDER_ENSEMBLE_MEDIA_TYPE
    )
    calls: list[dict[str, Any]] = []
    exports: list[dict[str, Any]] = []

    def fake_train(**kwargs: Any) -> object:
        calls.append(kwargs)
        return ensemble_ref

    def fake_export(**kwargs: Any) -> None:
        exports.append(kwargs)

    monkeypatch.setattr(orchestration, "_REPOSITORY_ROOT", repository)
    monkeypatch.setattr(orchestration, "_train_competition_defender", fake_train)
    monkeypatch.setattr(
        orchestration, "_export_defense_v1_defender", fake_export, raising=False
    )
    code = orchestration.command_main(
        "train_defender",
        [
            "--corpus",
            "fixtures/defense/v1/corpus-manifest.json",
            "--catalog",
            "config/defense/feature-catalog.json",
            "--profile",
            "config/defense/competition-profile.json",
            "--rollback-ref",
            "rules-v1",
            "--root",
            ".apar/defense-v1",
            "--export",
            "fixtures/defense/v1",
        ],
    )

    assert code == 0
    assert calls[0]["corpus_envelope_digest"] == corpus_ref.sha256
    assert calls[0]["rollback_ref"] == "rules-v1"
    assert calls[0]["enforce_preregistered_authorities"] is True
    assert len(exports) == 1
    assert exports[0]["reference"] == ensemble_ref


def test_build_named_command_resolves_signed_ledger_before_task14_assembly(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = tmp_path / "repository"
    runtime = repository / ".apar" / "defense-v1"
    runtime.mkdir(mode=0o700, parents=True)
    seed = b"b" * 32
    key = runtime / "run-signing.key"
    key.write_bytes(seed)
    key.chmod(0o600)
    signer = RunSigningIdentity.from_private_bytes(seed)
    profile_path = repository / "config" / "defense" / "competition-profile.json"
    profile_path.parent.mkdir(parents=True)
    profile_path.write_bytes(PROFILE.read_bytes())
    store = ArtifactStore(runtime / "artifacts")
    ledger_ref = store.put_bytes(b"{}", "application/json")
    ledger_path = repository / "docs" / "experiments" / "defense-v1-run-manifests.json"
    ledger_path.parent.mkdir(parents=True)
    run_ids = tuple(f"run-{index:032x}" for index in range(200))
    orchestration._publish_defense_v1_alias(
        path=ledger_path,
        kind="run_ledger",
        reference=ledger_ref,
        signer=signer,
        authenticated_run_ids=run_ids,
    )
    corpus_ref = store.put_bytes(b"{}", orchestration._CORPUS_ENVELOPE_MEDIA_TYPE)
    calls: list[dict[str, Any]] = []
    exports: list[dict[str, Any]] = []

    monkeypatch.setattr(orchestration, "_REPOSITORY_ROOT", repository)
    monkeypatch.setattr(
        orchestration,
        "_load_run_ledger_reference",
        lambda **_kwargs: SimpleNamespace(entries=()),
    )

    def fake_build(**kwargs: Any) -> object:
        calls.append(kwargs)
        return corpus_ref

    monkeypatch.setattr(orchestration, "_build_competition_corpus", fake_build)
    monkeypatch.setattr(
        orchestration,
        "_load_corpus_envelope",
        lambda *_args: (SimpleNamespace(), SimpleNamespace()),
    )
    monkeypatch.setattr(
        orchestration,
        "_export_defense_v1_corpus",
        lambda **kwargs: exports.append(kwargs),
    )
    code = orchestration.command_main(
        "build_defense_corpus",
        [
            "--profile",
            "config/defense/competition-profile.json",
            "--run-manifests",
            "docs/experiments/defense-v1-run-manifests.json",
            "--root",
            ".apar/defense-v1",
            "--output-manifest",
            "fixtures/defense/v1/corpus-manifest.json",
        ],
    )

    assert code == 0
    assert calls == [
        {
            "profile": orchestration.load_competition_profile(PROFILE, competition=True),
            "root": runtime,
            "ledger_digest": ledger_ref.sha256,
            "output_name": "defense-v1-corpus-envelope.json",
            "enforce_preregistered_authorities": True,
        }
    ]
    assert len(exports) == 1
    assert exports[0]["reference"] == corpus_ref


def test_prepare_hidden_named_command_uses_only_signed_frozen_inputs_and_returns_digest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    repository = tmp_path / "repository"
    runtime = repository / ".apar" / "defense-v1"
    runtime.mkdir(mode=0o700, parents=True)
    seed = b"p" * 32
    key = runtime / "run-signing.key"
    key.write_bytes(seed)
    key.chmod(0o600)
    signer = RunSigningIdentity.from_private_bytes(seed)
    profile_path = repository / "config" / "defense" / "competition-profile.json"
    profile_path.parent.mkdir(parents=True)
    profile_path.write_bytes(PROFILE.read_bytes())
    store = ArtifactStore(runtime / "artifacts")
    corpus_ref = store.put_bytes(b"{}", orchestration._CORPUS_ENVELOPE_MEDIA_TYPE)
    ensemble_ref = store.put_bytes(
        b'{"ensemble":true}', orchestration._DEFENDER_ENSEMBLE_MEDIA_TYPE
    )
    hidden_ref = store.put_bytes(
        b"hidden-envelope", "application/vnd.apar.hidden-context-envelope+json"
    )
    run_ids = tuple(f"run-{index:032x}" for index in range(200))
    fixture_root = repository / "fixtures" / "defense" / "v1"
    fixture_root.mkdir(parents=True)
    orchestration._publish_defense_v1_alias(
        path=fixture_root / "corpus-manifest.json",
        kind="corpus_envelope",
        reference=corpus_ref,
        signer=signer,
        authenticated_run_ids=run_ids,
        export_metadata={
            "observation_dataset": {
                "classification": "defender_visible",
                "content_digest": "1" * 64,
                "file_sha256": "2" * 64,
                "row_count": 1,
            },
            "truth_dataset": {
                "classification": "restricted_evaluator_only",
                "content_digest": "3" * 64,
                "file_sha256": "4" * 64,
                "row_count": 1,
            },
        },
    )
    orchestration._publish_defense_v1_alias(
        path=fixture_root / "defender-bundle.json",
        kind="defender_ensemble",
        reference=ensemble_ref,
        signer=signer,
        authenticated_run_ids=run_ids,
        export_metadata={
            "held_family_refs": {},
            "pooled_manifest": {},
            "pooled_ref": {},
            "portable_artifacts": {},
            "split_projections": {},
        },
    )
    calls: list[dict[str, Any]] = []

    def fake_prepare(**kwargs: Any) -> object:
        calls.append(kwargs)
        return hidden_ref

    monkeypatch.setattr(orchestration, "_REPOSITORY_ROOT", repository)
    monkeypatch.setattr(
        orchestration, "_prepare_competition_hidden_context", fake_prepare
    )
    code = orchestration.command_main(
        "prepare_hidden_context",
        [
            "--corpus",
            "fixtures/defense/v1/corpus-manifest.json",
            "--defender",
            "fixtures/defense/v1/defender-bundle.json",
            "--profile",
            "config/defense/competition-profile.json",
            "--root",
            ".apar/defense-v1",
        ],
    )

    assert code == 0
    assert len(calls) == 1
    assert calls[0]["corpus_envelope_ref"] == corpus_ref
    assert calls[0]["ensemble_ref"] == ensemble_ref
    assert calls[0]["development_run_ids"] == run_ids
    assert calls[0]["enforce_preregistered_authorities"] is True
    output = capsys.readouterr().out
    assert output.count("\n") == 1
    assert json.loads(output) == {
        "hidden_context_pointer_sha256": hidden_ref.sha256,
        "profile_sha256": orchestration._DEFENSE_V1_PROFILE_SHA256,
        "schema_version": "1.0.0",
    }
    assert "seed" not in output and "run-" not in output and str(runtime) not in output


def test_hidden_context_pointer_is_one_shot_and_rejects_alternate_valid_context(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = (tmp_path / "runtime").resolve()
    root.mkdir(mode=0o700)
    store = ArtifactStore(root / "artifacts")
    signer = RunSigningIdentity.from_private_bytes(b"p" * 32)
    profile = orchestration.load_competition_profile(PROFILE, competition=True)
    ensemble_ref = store.put_bytes(
        b"ensemble", orchestration._DEFENDER_ENSEMBLE_MEDIA_TYPE
    )
    corpus_ref = store.put_bytes(b"corpus", orchestration._CORPUS_ENVELOPE_MEDIA_TYPE)
    source_ref = store.put_bytes(
        b"source", orchestration.HIDDEN_SOURCE_RECEIPT_MEDIA_TYPE
    )
    hidden_refs = tuple(
        store.put_bytes(
            f"hidden-{index}".encode(),
            "application/vnd.apar.restricted-hidden-context-envelope+json",
        )
        for index in range(2)
    )

    def pointer_for(hidden_ref: object) -> orchestration.HiddenContextPointer:
        assert isinstance(hidden_ref, ArtifactRef)
        unsigned = {
            "development_corpus_ref": orchestration._reference_document(corpus_ref),
            "ensemble_ref": orchestration._reference_document(ensemble_ref),
            "hidden_context_ref": orchestration._reference_document(hidden_ref),
            "kind": "competition_hidden_context_pointer",
            "profile_sha256": hashlib.sha256(profile.to_json()).hexdigest(),
            "public_key_base64": signer.public_key_base64,
            "schema_version": "1.0.0",
            "signer_key_id": signer.key_id,
            "source_receipt_ref": orchestration._reference_document(source_ref),
        }
        return orchestration.HiddenContextPointer.model_validate(
            {**unsigned, "signature_base64": signer.sign(unsigned)}
        )

    first = pointer_for(hidden_refs[0])
    first_ref = store.put_bytes(
        canonical_json_bytes(first.model_dump(mode="json")),
        orchestration._HIDDEN_CONTEXT_POINTER_MEDIA_TYPE,
    )
    orchestration._publish_json_file(
        root,
        orchestration._HIDDEN_CONTEXT_POINTER_NAME,
        first.model_dump(mode="json"),
    )
    assert orchestration._load_hidden_context_pointer(
        root=root,
        store=store,
        digest=first_ref.sha256,
        signer=signer,
        profile_sha256=hashlib.sha256(profile.to_json()).hexdigest(),
        ensemble_ref=ensemble_ref,
        corpus_envelope_ref=corpus_ref,
    ) == first

    alternate = pointer_for(hidden_refs[1])
    alternate_ref = store.put_bytes(
        canonical_json_bytes(alternate.model_dump(mode="json")),
        orchestration._HIDDEN_CONTEXT_POINTER_MEDIA_TYPE,
    )
    with pytest.raises(orchestration.CliContractError, match="pointer reference"):
        orchestration._load_hidden_context_pointer(
            root=root,
            store=store,
            digest=alternate_ref.sha256,
            signer=signer,
            profile_sha256=hashlib.sha256(profile.to_json()).hexdigest(),
            ensemble_ref=ensemble_ref,
            corpus_envelope_ref=corpus_ref,
        )

    monkeypatch.setattr(
        orchestration,
        "_load_standard_signer",
        lambda *_args, **_kwargs: pytest.fail("second prep reached signer access"),
    )
    with pytest.raises(orchestration.CliContractError, match="already immutably selected"):
        orchestration._prepare_competition_hidden_context(
            profile=profile,
            root=root,
            corpus_envelope_ref=corpus_ref,
            ensemble_ref=ensemble_ref,
            development_run_ids=tuple(
                f"run-{index:032x}" for index in range(200)
            ),
        )


def test_development_named_command_uses_signed_aliases_and_never_loads_hidden_authority(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = tmp_path / "repository"
    runtime = repository / ".apar" / "defense-v1"
    runtime.mkdir(mode=0o700, parents=True)
    seed = b"e" * 32
    key = runtime / "run-signing.key"
    key.write_bytes(seed)
    key.chmod(0o600)
    signer = RunSigningIdentity.from_private_bytes(seed)
    evaluator = EvaluatorSigningIdentity.from_private_bytes(b"d" * 32)
    hidden_evaluator = EvaluatorSigningIdentity.from_private_bytes(b"h" * 32)
    profile_path = repository / "config" / "defense" / "competition-profile.json"
    profile_path.parent.mkdir(parents=True)
    profile_path.write_bytes(PROFILE.read_bytes())
    export = repository / "fixtures" / "defense" / "v1"
    export.mkdir(parents=True)
    store = ArtifactStore(runtime / "artifacts")
    corpus_ref = store.put_bytes(b"corpus", orchestration._CORPUS_ENVELOPE_MEDIA_TYPE)
    candidate_refs = tuple(
        store.put_bytes(
            f"candidate-{index}".encode(), orchestration._DEFENDER_BUNDLE_MEDIA_TYPE
        )
        for index in range(5)
    )
    profile = orchestration.load_competition_profile(PROFILE, competition=True)
    hidden_source = RunSigningIdentity.from_private_bytes(b"s" * 32)
    ensemble = orchestration._build_defender_ensemble(
        pooled_ref=candidate_refs[0],
        profile=profile,
        held_family_refs={
            family: candidate_refs[index + 1]
            for index, family in enumerate(profile.families)
        },
        signer=signer,
        corpus_envelope_ref=corpus_ref,
        hidden_source_signer_key_id=hidden_source.key_id,
        hidden_source_public_key_base64=hidden_source.public_key_base64,
    )
    assert isinstance(ensemble, orchestration.DefenderEnsembleEnvelope)
    top_ref = store.put_bytes(
        canonical_json_bytes(ensemble.model_dump(mode="json")),
        orchestration._DEFENDER_ENSEMBLE_MEDIA_TYPE,
    )
    run_ids = tuple(f"run-{index:032x}" for index in range(200))
    orchestration._publish_defense_v1_alias(
        path=export / "corpus-manifest.json",
        kind="corpus_envelope",
        reference=corpus_ref,
        signer=signer,
        authenticated_run_ids=run_ids,
        export_metadata={
            "observation_dataset": {
                "classification": "defender_visible",
                "content_digest": "1" * 64,
                "file_sha256": "2" * 64,
                "row_count": 1,
            },
            "truth_dataset": {
                "classification": "restricted_evaluator_only",
                "content_digest": "3" * 64,
                "file_sha256": "4" * 64,
                "row_count": 1,
            },
        },
    )
    orchestration._publish_defense_v1_alias(
        path=export / "defender-bundle.json",
        kind="defender_ensemble",
        reference=top_ref,
        signer=signer,
        authenticated_run_ids=run_ids,
        export_metadata={
            "held_family_refs": ensemble.held_family_refs,
            "pooled_manifest": {},
            "pooled_ref": ensemble.pooled_ref,
            "portable_artifacts": {},
            "split_projections": {},
        },
    )
    loaded = SimpleNamespace(verify_reload=lambda: None, manifest=SimpleNamespace())

    class FakePublisher:
        def __init__(self, *_args: object) -> None:
            pass

        def load(self, _ref: object) -> object:
            return loaded

        def close(self) -> None:
            pass

    published = SimpleNamespace(
        scorecard_ref=store.put_bytes(b"score", "application/json"),
        evaluation_bundle_ref=store.put_bytes(b"bundle", "application/json"),
        development_evidence_ref=store.put_bytes(b"evidence", "application/json"),
        restricted_publication_receipt_ref=store.put_bytes(b"receipt", "application/json"),
        promotion_envelope_digest="5" * 64,
        descriptor_scope=("chronological",),
        champion_decision=SimpleNamespace(status=SimpleNamespace(value="NO_PROMOTION")),
    )
    publish_calls: list[dict[str, Any]] = []
    completion_ref = store.put_bytes(
        b"completion", "application/vnd.apar.development-completion+json"
    )
    completion_calls: list[dict[str, Any]] = []
    authority_checks: list[str] = []
    original_authority_check = orchestration._assert_preregistered_authority

    monkeypatch.setattr(orchestration, "_REPOSITORY_ROOT", repository)
    monkeypatch.setattr(
        orchestration,
        "_preregistered_authority",
        lambda role: {
            "publication": orchestration.PreregisteredAuthorityIdentity(
                signer.key_id, signer.public_key_base64
            ),
            "development_evaluator": orchestration.PreregisteredAuthorityIdentity(
                evaluator.key_id, evaluator.public_key_base64
            ),
            "hidden_evaluator": orchestration.PreregisteredAuthorityIdentity(
                hidden_evaluator.key_id, hidden_evaluator.public_key_base64
            ),
            "hidden_source": orchestration.PreregisteredAuthorityIdentity(
                hidden_source.key_id, hidden_source.public_key_base64
            ),
        }[role],
    )
    monkeypatch.setattr(
        orchestration,
        "_assert_preregistered_authority",
        lambda role, **kwargs: (
            authority_checks.append(role),
            original_authority_check(role, **kwargs),
        )[-1],
    )
    monkeypatch.setattr(orchestration, "DefenderBundlePublisher", FakePublisher)
    monkeypatch.setattr(
        orchestration,
        "_load_corpus_envelope",
        lambda *_args: (
            SimpleNamespace(run_ledger_sha256="6" * 64),
            SimpleNamespace(manifest=SimpleNamespace(run_ids=run_ids)),
        ),
    )
    monkeypatch.setattr(orchestration, "make_evaluation_split", lambda *_args: object())
    monkeypatch.setattr(
        orchestration,
        "_load_competition_evaluator_identity",
        lambda _root: evaluator,
    )
    monkeypatch.setattr(
        orchestration,
        "_load_competition_hidden_identity",
        lambda *_args: pytest.fail("development opened hidden authority"),
    )
    import apar.evaluation.competition as competition

    monkeypatch.setattr(
        competition,
        "publish_competition_evaluation",
        lambda **kwargs: (publish_calls.append(kwargs), published)[1],
    )
    monkeypatch.setattr(
        competition,
        "seal_development_completion",
        lambda **kwargs: (completion_calls.append(kwargs), completion_ref)[1],
    )
    code = orchestration.command_main(
        "evaluate_defender",
        [
            "--phase",
            "development",
            "--corpus",
            "fixtures/defense/v1/corpus-manifest.json",
            "--defender",
            "fixtures/defense/v1/defender-bundle.json",
            "--profile",
            "config/defense/competition-profile.json",
            "--root",
            ".apar/defense-v1",
            "--export",
            "fixtures/defense/v1",
        ],
    )

    assert code == 0
    assert authority_checks == [
        "publication",
        "development_evaluator",
        "hidden_source",
    ]
    assert len(publish_calls) == 1
    assert publish_calls[0]["pooled_ref"] == candidate_refs[0]
    assert publish_calls[0]["corpus"].manifest.run_ids == run_ids
    assert len(completion_calls) == 1


def test_hidden_named_command_verifies_completion_before_public_only_source_and_exports(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    repository = tmp_path / "repository"
    runtime = repository / ".apar" / "defense-v1"
    runtime.mkdir(mode=0o700, parents=True)
    publication = RunSigningIdentity.from_private_bytes(b"j" * 32)
    publication_key = runtime / "run-signing.key"
    publication_key.write_bytes(b"j" * 32)
    publication_key.chmod(0o600)
    evaluator = EvaluatorSigningIdentity.from_private_bytes(b"k" * 32)
    hidden = EvaluatorSigningIdentity.from_private_bytes(b"l" * 32)
    hidden_context_signer = RunSigningIdentity.from_private_bytes(b"l" * 32)
    hidden_source = RunSigningIdentity.from_private_bytes(b"m" * 32)
    profile_path = repository / "config" / "defense" / "competition-profile.json"
    profile_path.parent.mkdir(parents=True)
    profile_path.write_bytes(PROFILE.read_bytes())
    export = repository / "fixtures" / "defense" / "v1"
    export.mkdir(parents=True)
    result_path = repository / "docs" / "experiments" / "defense-v1-result.json"
    result_path.parent.mkdir(parents=True)
    store = ArtifactStore(runtime / "artifacts")
    corpus_ref = store.put_bytes(b"corpus", orchestration._CORPUS_ENVELOPE_MEDIA_TYPE)
    candidate_refs = tuple(
        store.put_bytes(
            f"candidate-{index}".encode(), orchestration._DEFENDER_BUNDLE_MEDIA_TYPE
        )
        for index in range(5)
    )
    profile = orchestration.load_competition_profile(PROFILE, competition=True)
    ensemble = orchestration._build_defender_ensemble(
        pooled_ref=candidate_refs[0],
        profile=profile,
        held_family_refs={
            family: candidate_refs[index + 1]
            for index, family in enumerate(profile.families)
        },
        signer=publication,
        corpus_envelope_ref=corpus_ref,
        hidden_source_signer_key_id=hidden_source.key_id,
        hidden_source_public_key_base64=hidden_source.public_key_base64,
    )
    assert isinstance(ensemble, orchestration.DefenderEnsembleEnvelope)
    top_ref = store.put_bytes(
        canonical_json_bytes(ensemble.model_dump(mode="json")),
        orchestration._DEFENDER_ENSEMBLE_MEDIA_TYPE,
    )
    run_ids = tuple(f"run-{index:032x}" for index in range(200))
    for name, kind, reference, metadata in (
        (
            "corpus-manifest.json",
            "corpus_envelope",
            corpus_ref,
            {
                "observation_dataset": {
                    "classification": "defender_visible",
                    "content_digest": "1" * 64,
                    "file_sha256": "2" * 64,
                    "row_count": 0,
                },
                "truth_dataset": {
                    "classification": "restricted_evaluator_only",
                    "content_digest": "3" * 64,
                    "file_sha256": "4" * 64,
                    "row_count": 0,
                },
            },
        ),
        (
            "defender-bundle.json",
            "defender_ensemble",
            top_ref,
            {
                "held_family_refs": ensemble.held_family_refs,
                "pooled_manifest": {},
                "pooled_ref": ensemble.pooled_ref,
                "portable_artifacts": {},
                "split_projections": {},
            },
        ),
    ):
        orchestration._publish_defense_v1_alias(
            path=export / name,
            kind=kind,
            reference=reference,
            signer=publication,
            authenticated_run_ids=run_ids,
            export_metadata=metadata,
        )
    completion_ref = store.put_bytes(
        b"completion", "application/vnd.apar.development-completion+json"
    )
    scorecard_ref = store.put_bytes(b"scorecard", "application/json")
    evaluation_ref = store.put_bytes(b"evaluation", "application/json")
    evidence_ref = store.put_bytes(b"evidence", "application/json")
    source_ref = store.put_bytes(
        b"source", orchestration.HIDDEN_SOURCE_RECEIPT_MEDIA_TYPE
    )
    context_ref = store.put_bytes(
        b"context", "application/vnd.apar.restricted-hidden-context-envelope+json"
    )
    restricted_ref = store.put_bytes(
        b"restricted", "application/vnd.apar.hidden-evaluation-context+json"
    )
    pointer_ref = store.put_bytes(
        b"pointer", orchestration._HIDDEN_CONTEXT_POINTER_MEDIA_TYPE
    )
    corpus = FrozenCorpus(
        observations=(),
        truth=(),
        manifest=CorpusManifest(
            profile_id="defense-competition-v1",
            run_ids=run_ids,
            run_lineage_digests=tuple(
                hashlib.sha256(item.encode()).hexdigest() for item in run_ids
            ),
            observation_count=0,
            truth_count=0,
        ),
    )
    frozen_at = datetime(2026, 8, 18, 13, tzinfo=UTC)
    loaded = SimpleNamespace(
        verify_reload=lambda: None,
        manifest=SimpleNamespace(bundle_id="bundle", frozen_at=frozen_at),
    )

    class FakePublisher:
        def __init__(self, *_args: object) -> None:
            pass

        def load(self, _reference: object) -> object:
            return loaded

        def close(self) -> None:
            pass

    order: list[str] = []
    completion = SimpleNamespace(
        scorecard_ref=orchestration._reference_document(scorecard_ref),
        evaluation_bundle_ref=orchestration._reference_document(evaluation_ref),
        development_evidence_ref=orchestration._reference_document(evidence_ref),
    )
    pointer = SimpleNamespace(
        hidden_context_ref=orchestration._reference_document(context_ref),
        source_receipt_ref=orchestration._reference_document(source_ref),
    )
    source_binding = HiddenSourceWorkerBinding(
        receipt_ref=source_ref,
        source_signer_key_id=hidden_source.key_id,
        source_public_key_base64=hidden_source.public_key_base64,
        development_run_ids=run_ids,
        development_event_ids=(),
        development_payment_ids=(),
        development_campaign_ids=(),
    )
    hidden_release = datetime(2026, 8, 20, 12, tzinfo=UTC)
    published = SimpleNamespace(
        scorecard_ref=scorecard_ref,
        evaluation_bundle_ref=evaluation_ref,
        threshold_set_ref=store.put_bytes(b"threshold", "application/json"),
        development_evidence_ref=None,
        restricted_publication_receipt_ref=store.put_bytes(b"receipt", "application/json"),
        promotion_envelope_digest="5" * 64,
        descriptor_scope=("hidden:hidden",),
        champion_decision=SimpleNamespace(status=SimpleNamespace(value="NO_PROMOTION")),
        hidden_released_at=hidden_release,
        hidden_public_proof=SimpleNamespace(),
        public_artifacts={},
    )
    publish_calls: list[dict[str, Any]] = []
    export_calls: list[dict[str, Any]] = []
    authority_checks: list[str] = []
    original_authority_check = orchestration._assert_preregistered_authority
    monkeypatch.setattr(orchestration, "_REPOSITORY_ROOT", repository)
    monkeypatch.setattr(
        orchestration,
        "_preregistered_authority",
        lambda role: {
            "publication": orchestration.PreregisteredAuthorityIdentity(
                publication.key_id, publication.public_key_base64
            ),
            "development_evaluator": orchestration.PreregisteredAuthorityIdentity(
                evaluator.key_id, evaluator.public_key_base64
            ),
            "hidden_evaluator": orchestration.PreregisteredAuthorityIdentity(
                hidden.key_id, hidden.public_key_base64
            ),
            "hidden_source": orchestration.PreregisteredAuthorityIdentity(
                hidden_source.key_id, hidden_source.public_key_base64
            ),
        }[role],
    )
    monkeypatch.setattr(
        orchestration,
        "_assert_preregistered_authority",
        lambda role, **kwargs: (
            authority_checks.append(role),
            original_authority_check(role, **kwargs),
        )[-1],
    )
    monkeypatch.setattr(orchestration, "DefenderBundlePublisher", FakePublisher)
    monkeypatch.setattr(
        orchestration,
        "_load_corpus_envelope",
        lambda *_args: (SimpleNamespace(run_ledger_sha256="6" * 64), corpus),
    )
    monkeypatch.setattr(orchestration, "make_evaluation_split", lambda *_args: object())
    monkeypatch.setattr(
        orchestration, "_load_competition_evaluator_identity", lambda _root: evaluator
    )

    def verify_completion(**_kwargs: object) -> object:
        order.append("completion")
        return completion

    def load_hidden(*_args: object) -> tuple[object, object]:
        assert order == ["completion"]
        order.append("hidden_identity")
        return hidden, hidden_context_signer

    def load_pointer(**kwargs: object) -> object:
        assert order == ["completion", "hidden_identity"]
        assert kwargs["digest"] == pointer_ref.sha256
        order.append("pointer")
        return pointer

    monkeypatch.setattr(
        orchestration, "_load_competition_hidden_identity", load_hidden
    )
    monkeypatch.setattr(
        orchestration,
        "_load_competition_hidden_run_identity",
        lambda *_args, **_kwargs: pytest.fail("hidden evaluate opened source private key"),
    )
    monkeypatch.setattr(
        orchestration,
        "_load_competition_hidden_run_public_identity",
        lambda *_args, **_kwargs: pytest.fail("hidden evaluate reopened source public key"),
    )
    monkeypatch.setattr(orchestration, "_load_hidden_context_pointer", load_pointer)
    monkeypatch.setattr(
        orchestration,
        "_verify_hidden_source_metadata",
        lambda **_kwargs: (order.append("source_metadata"), source_binding)[1],
    )
    monkeypatch.setattr(
        orchestration.DefenseScorecard,
        "from_json",
        lambda *_args, **_kwargs: SimpleNamespace(
            bundle_summary=SimpleNamespace(bundle_id="bundle")
        ),
    )
    monkeypatch.setattr(
        orchestration,
        "load_evaluation_bundle",
        lambda *_args, **_kwargs: SimpleNamespace(scorecard_sha256=scorecard_ref.sha256),
    )
    monkeypatch.setattr(
        orchestration,
        "_build_hidden_release_attestation",
        lambda **_kwargs: SimpleNamespace(),
    )
    monkeypatch.setattr(
        orchestration,
        "_export_defense_v1_evaluation",
        lambda **kwargs: export_calls.append(kwargs),
    )
    import apar.evaluation.competition as competition

    monkeypatch.setattr(competition, "verify_development_completion", verify_completion)
    monkeypatch.setattr(
        competition,
        "verify_hidden_context",
        lambda **_kwargs: (
            SimpleNamespace(
                as_of=hidden_release,
                source_lineage_digest=source_ref.sha256,
            ),
            (),
            restricted_ref,
        ),
    )
    monkeypatch.setattr(
        competition,
        "publish_competition_evaluation",
        lambda **kwargs: (publish_calls.append(kwargs), published)[1],
    )
    code = orchestration.command_main(
        "evaluate_defender",
        [
            "--phase",
            "hidden",
            "--corpus",
            "fixtures/defense/v1/corpus-manifest.json",
            "--defender",
            "fixtures/defense/v1/defender-bundle.json",
            "--profile",
            "config/defense/competition-profile.json",
            "--root",
            ".apar/defense-v1",
            "--export",
            "fixtures/defense/v1",
            "--hash-manifest",
            "fixtures/defense/v1/hash-manifest.json",
            "--result",
            "docs/experiments/defense-v1-result.json",
            "--development-scorecard",
            completion_ref.sha256,
            "--hidden-corpus",
            pointer_ref.sha256,
        ],
    )
    assert code == 0
    assert authority_checks == [
        "publication",
        "development_evaluator",
        "hidden_source",
        "hidden_evaluator",
    ]
    assert order == ["completion", "hidden_identity", "pointer", "source_metadata"]
    assert len(publish_calls) == 1
    assert publish_calls[0]["pooled_ref"] == candidate_refs[0]
    assert publish_calls[0]["corpus"] == corpus
    assert publish_calls[0]["hidden_source_binding"] == source_binding
    assert len(export_calls) == 1
    output = capsys.readouterr().out
    assert output.count("\n") == 1
    assert "source" not in output and "hidden-corpus" not in output


def test_competition_ensemble_requires_five_real_predecessor_rollbacks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "runtime"
    root.mkdir(mode=0o700)
    store = ArtifactStore(root / "artifacts")
    corpus_ref = store.put_bytes(b"corpus", orchestration._CORPUS_ENVELOPE_MEDIA_TYPE)
    candidate_refs = tuple(
        store.put_bytes(
            f"bundle-{index}".encode(), orchestration._DEFENDER_BUNDLE_MEDIA_TYPE
        )
        for index in range(5)
    )
    signer = RunSigningIdentity.from_private_bytes(b"r" * 32)
    hidden_source = RunSigningIdentity.from_private_bytes(b"s" * 32)
    (root / "hidden-run-signing.pub").write_bytes(
        base64.b64decode(hidden_source.public_key_base64)
    )
    (root / "hidden-run-signing.pub").chmod(0o600)
    profile = orchestration.load_competition_profile(PROFILE, competition=True)
    freeze_calls: list[dict[str, Any]] = []

    def fake_freeze(**kwargs: Any) -> object:
        freeze_calls.append(kwargs)
        return candidate_refs[len(freeze_calls) - 1]

    class FakeVerifier:
        available = True

        def __init__(self, *_args: object, **_kwargs: object) -> None:
            pass

        def attest(self, _reference: object) -> object:
            return SimpleNamespace(rollback_available=self.available)

    monkeypatch.setattr(orchestration, "_freeze_competition_candidate", fake_freeze)
    monkeypatch.setattr(orchestration, "_load_standard_signer", lambda _root: signer)
    monkeypatch.setattr(orchestration, "DefenderBundleVerifier", FakeVerifier)

    top = orchestration._train_competition_defender(
        profile=profile,
        root=root,
        corpus_envelope_digest=corpus_ref.sha256,
        catalog_path=ROOT / "config" / "defense" / "feature-catalog.json",
        rollback_ref="rules-v1",
    )

    assert top.media_type == orchestration._DEFENDER_ENSEMBLE_MEDIA_TYPE
    assert len(freeze_calls) == 5
    assert [call["held_out_family"] for call in freeze_calls] == [
        None,
        *profile.families,
    ]
    assert {call["rollback_ref"] for call in freeze_calls} == {"rules-v1"}

    FakeVerifier.available = False
    freeze_calls.clear()
    with pytest.raises(orchestration.CliContractError, match="rollback is unavailable"):
        orchestration._train_competition_defender(
            profile=profile,
            root=root,
            corpus_envelope_digest=corpus_ref.sha256,
            catalog_path=ROOT / "config" / "defense" / "feature-catalog.json",
            rollback_ref="rules-v1",
        )


def test_defender_export_hydrates_five_roles_without_runtime_or_split_truth(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = (tmp_path / "runtime").resolve()
    runtime.mkdir(mode=0o700)
    publication = RunSigningIdentity.from_private_bytes(b"n" * 32)
    publication_key = runtime / "run-signing.key"
    publication_key.write_bytes(b"n" * 32)
    publication_key.chmod(0o600)
    hidden_source = RunSigningIdentity.from_private_bytes(b"o" * 32)
    (runtime / "hidden-run-signing.pub").write_bytes(
        base64.b64decode(hidden_source.public_key_base64, validate=True)
    )
    (runtime / "hidden-run-signing.pub").chmod(0o600)
    production_profile = orchestration.load_competition_profile(
        PROFILE, competition=True
    )
    profile_document = production_profile.model_dump(mode="json")
    profile_document["gbdt"] = {
        "depths": [4],
        "iterations": 20,
        "l2_leaf_regs": [3.0],
        "learning_rates": [0.03],
    }
    profile_document["calibration"]["minimum_class_count"] = 2
    profile = orchestration.CompetitionProfile.model_validate(profile_document)
    assert profile.fixture_only
    reduced_profile_sha256 = hashlib.sha256(profile.to_json()).hexdigest()
    monkeypatch.setattr(
        orchestration.CompetitionProfile,
        "fixture_only",
        property(lambda _self: False),
    )
    monkeypatch.setattr(
        orchestration, "_DEFENSE_V1_PROFILE_SHA256", reduced_profile_sha256
    )
    observations: list[ObservedEvent] = []
    truth: list[EvaluationTruthRow] = []
    run_ids: list[str] = []
    run_lineages: list[str] = []
    for family in profile.families:
        for index in range(profile.campaigns_per_family):
            campaign_id = f"portable-{family}-{index:02d}"
            run_id = f"run-{hashlib.sha256(campaign_id.encode()).hexdigest()[:32]}"
            run_ids.append(run_id)
            run_lineages.append(hashlib.sha256(run_id.encode()).hexdigest())
            start = profile.campaign_start(family, index)
            for slot in range(2):
                event_id = f"event-{family}-{index:02d}-{slot}"
                payment_id = f"payment-{family}-{index:02d}-{slot}"
                decision_at = start + timedelta(seconds=slot + 1)
                observations.append(
                    ObservedEvent(
                        event_id=event_id,
                        payment_id=payment_id,
                        rail=Rail.CARD,
                        event_type=EventKind.AUTHORIZATION,
                        amount=Decimal("100.00") + Decimal(slot),
                        currency="USD",
                        event_time=decision_at,
                        available_at=decision_at,
                        decision_at=decision_at,
                        actor_id=f"actor-{family}-{index:02d}-{slot % 2}",
                        counterparty_id=(
                            f"counterparty-{family}-{index:02d}-{slot % 3}"
                        ),
                        optional_refs={},
                        integrity_status="not_applicable",
                        is_decision_point=True,
                    )
                )
                truth.append(
                    EvaluationTruthRow(
                        event_id=event_id,
                        payment_id=payment_id,
                        campaign_id=campaign_id,
                        family=family,
                        viewpoint="development",
                        is_fraud=slot % 2 == 1,
                        label_source="population_truth",
                        label_mature_at=decision_at
                        + timedelta(days=profile.label_delay_days),
                        first_settlement_at=None,
                        net_settled_value=Decimal("100.00"),
                        lifecycle_event_ids=(event_id,),
                    )
                )
    corpus = FrozenCorpus(
        observations=tuple(observations),
        truth=tuple(truth),
        manifest=CorpusManifest(
            profile_id="defense-competition-v1",
            run_ids=tuple(run_ids),
            run_lineage_digests=tuple(run_lineages),
            observation_count=len(observations),
            truth_count=len(truth),
        ),
    )
    store = ArtifactStore(runtime / "artifacts")
    observation_payload = canonical_json_bytes(
        [item.model_dump(mode="json") for item in corpus.observations]
    )
    truth_payload = canonical_json_bytes(
        {
            "manifest": corpus.manifest.model_dump(mode="json"),
            "truth": [item.model_dump(mode="json") for item in corpus.truth],
        }
    )
    observations_ref = store.put_bytes(
        observation_payload, "application/vnd.apar.observations+json"
    )
    truth_ref = store.put_bytes(
        truth_payload, "application/vnd.apar.restricted-truth+json"
    )
    unsigned_envelope = {
        "campaign_count": profile.campaign_count,
        "corpus_digest": orchestration.frozen_corpus_digest(corpus),
        "family_campaign_counts": {
            family: profile.campaigns_per_family for family in profile.families
        },
        "observation_digest": orchestration._lineage_digest(
            "observations", observation_payload
        ),
        "observations": orchestration._reference_document(observations_ref),
        "profile_sha256": hashlib.sha256(profile.to_json()).hexdigest(),
        "public_key_base64": publication.public_key_base64,
        "restricted_truth": orchestration._reference_document(truth_ref),
        "restricted_truth_digest": orchestration._lineage_digest(
            "truth", truth_payload
        ),
        "run_ledger_sha256": "7" * 64,
        "schema_version": "1.0.0",
        "signer_key_id": publication.key_id,
    }
    envelope = orchestration.CorpusEnvelope.model_validate(
        {
            **unsigned_envelope,
            "signature_base64": publication.sign(unsigned_envelope),
        }
    )
    corpus_ref = store.put_bytes(
        canonical_json_bytes(envelope.model_dump(mode="json")),
        orchestration._CORPUS_ENVELOPE_MEDIA_TYPE,
    )

    top_ref = orchestration._train_competition_defender(
        profile=profile,
        root=runtime,
        corpus_envelope_digest=corpus_ref.sha256,
        catalog_path=ROOT / "config" / "defense" / "feature-catalog.json",
        rollback_ref="rules-v1",
    )
    export = tmp_path / "export"
    export.mkdir()
    orchestration._export_defense_v1_defender(
        directory=export,
        reference=top_ref,
        root=runtime,
        profile=profile,
        signer=publication,
        authenticated_run_ids=tuple(run_ids),
    )
    ensemble = orchestration._load_defender_ensemble(
        store=store,
        top_ref=top_ref,
        profile=profile,
        signer=publication,
    )
    assert ensemble is not None
    role_refs = {
        "pooled": orchestration._artifact_ref(ensemble.pooled_ref),
        **{
            family: orchestration._artifact_ref(ensemble.held_family_refs[family])
            for family in profile.families
        },
    }
    original_predictions: dict[str, tuple[float, ...]] = {}
    original_predecessors: set[str] = set()
    publisher = orchestration.DefenderBundlePublisher(store, publication, ROOT)
    try:
        for role, role_ref in role_refs.items():
            candidate = publisher.load(role_ref)
            candidate.verify_reload()
            original_predictions[role] = tuple(
                float(item) for item in candidate.scorer.predict(candidate.reload_matrix)
            )
            predecessor_ref = store.resolve(candidate.manifest.rollback_ref)
            predecessor = publisher.load(predecessor_ref)
            predecessor.verify_reload()
            assert predecessor.manifest.rollback_ref == orchestration.GENESIS_ROLLBACK_REF
            original_predecessors.add(predecessor_ref.sha256)
    finally:
        publisher.close()
    assert len(original_predecessors) == 5
    alias_path = export / "defender-bundle.json"
    alias = json.loads(alias_path.read_bytes())
    portable_records = alias["export_metadata"]["portable_artifacts"]
    assert isinstance(portable_records, dict)
    split_projections = alias["export_metadata"]["split_projections"]
    assert isinstance(split_projections, dict)
    assert len(split_projections) == 10
    bundle_record_digests = {
        digest
        for digest, record in portable_records.items()
        if record["media_type"] == orchestration._DEFENDER_BUNDLE_MEDIA_TYPE
    }
    assert bundle_record_digests == {
        *(reference.sha256 for reference in role_refs.values()),
        *original_predecessors,
    }
    forbidden_keys = {
        "fraud_prevalence",
        "is_fraud",
        "labels",
        "label_source",
        "net_settled_value",
        "net_settled_value_totals",
        "row_is_fraud",
        "row_net_settled_values",
        "target",
        "targets",
    }

    def assert_truth_private(value: object) -> None:
        pending = [value]
        while pending:
            current = pending.pop()
            if isinstance(current, dict):
                assert forbidden_keys.isdisjoint(current)
                pending.extend(current.values())
            elif isinstance(current, (list, tuple)):
                pending.extend(current)
            elif isinstance(current, str):
                normalized = current.casefold()
                assert not any(
                    token in normalized
                    for token in (
                        "fraud_prevalence",
                        "label_source",
                        "net_settled_value",
                        "row_is_fraud",
                        "row_net_settled_values",
                    )
                )

    assert_truth_private(alias)
    for record in portable_records.values():
        assert isinstance(record, dict)
        payload = base64.b64decode(record["payload_base64"], validate=True)
        media_type = record["media_type"]
        if media_type == "application/vnd.apache.parquet":
            table = pq.read_table(pa.BufferReader(payload))
            assert forbidden_keys.isdisjoint(table.schema.names)
            assert_truth_private(table.to_pylist())
            assert_truth_private(
                {
                    key.decode("utf-8", errors="strict"): value.decode(
                        "utf-8", errors="strict"
                    )
                    for key, value in (table.schema.metadata or {}).items()
                }
            )
        elif media_type.endswith("+json"):
            assert_truth_private(json.loads(payload))

    unavailable_runtime = tmp_path / "runtime-unavailable"
    runtime.rename(unavailable_runtime)
    fresh_store = ArtifactStore(tmp_path / "fresh-store")
    portable = orchestration.hydrate_defense_v1_defender(
        alias_path,
        store=fresh_store,
        source_root=ROOT,
        signer_key_id=publication.key_id,
        public_key_base64=publication.public_key_base64,
        expected_profile_sha256=reduced_profile_sha256,
    )
    assert set(portable.candidates) == {"pooled", *profile.families}
    assert len({item.manifest.bundle_id for item in portable.candidates.values()}) == 5
    assert set(portable.candidates) == set(original_predictions)
    for role, candidate in portable.candidates.items():
        candidate.verify_reload()
        scores = candidate.scorer.predict(candidate.reload_matrix)
        assert scores.shape == (len(candidate.reload_matrix.rows),)
        assert tuple(float(item) for item in scores) == original_predictions[role]
        split_component = candidate.manifest.component("split")
        with pytest.raises(ValueError, match="does not exist"):
            fresh_store.resolve(split_component.sha256)


def test_hidden_export_writes_exact_public_graph_result_then_hash_last(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    directory = tmp_path / "fixtures"
    directory.mkdir()
    store = ArtifactStore(tmp_path / "store")
    signer = RunSigningIdentity.from_private_bytes(b"v" * 32)
    hidden_signer = EvaluatorSigningIdentity.from_private_bytes(b"h" * 32)
    development_signer = EvaluatorSigningIdentity.from_private_bytes(b"d" * 32)
    hidden_source_signer = RunSigningIdentity.from_private_bytes(b"s" * 32)
    authority_identities = {
        "development_evaluator": orchestration.PreregisteredAuthorityIdentity(
            development_signer.key_id, development_signer.public_key_base64
        ),
        "hidden_evaluator": orchestration.PreregisteredAuthorityIdentity(
            hidden_signer.key_id, hidden_signer.public_key_base64
        ),
        "hidden_source": orchestration.PreregisteredAuthorityIdentity(
            hidden_source_signer.key_id, hidden_source_signer.public_key_base64
        ),
        "publication": orchestration.PreregisteredAuthorityIdentity(
            signer.key_id, signer.public_key_base64
        ),
    }
    monkeypatch.setattr(
        orchestration,
        "_preregistered_authority",
        lambda role: authority_identities[role],
    )
    public_artifacts = {
        name: store.put_bytes(
            f"public:{name}".encode(),
            "application/octet-stream",
        )
        for name in orchestration._DEFENSE_V1_PUBLIC_REPORT_FILES
    }
    for name in sorted(
        orchestration._DEFENSE_V1_FIXTURE_FILES
        - orchestration._DEFENSE_V1_PUBLIC_REPORT_FILES
    ):
        (directory / name).write_bytes(f"frozen:{name}".encode())
    evaluation_bundle_ref = store.put_bytes(b"evaluation-bundle", "application/json")
    threshold_set_ref = store.put_bytes(b"threshold-set", "application/json")
    ensemble_ref = store.put_bytes(
        b"ensemble", orchestration._DEFENDER_ENSEMBLE_MEDIA_TYPE
    )
    pooled_ref = store.put_bytes(b"pooled", orchestration._DEFENDER_BUNDLE_MEDIA_TYPE)
    held_refs = {
        family: store.put_bytes(
            f"held:{family}".encode(),
            orchestration._DEFENDER_BUNDLE_MEDIA_TYPE,
        )
        for family in orchestration._FAMILIES
    }
    corpus_ref = store.put_bytes(
        b"corpus", orchestration._CORPUS_ENVELOPE_MEDIA_TYPE
    )
    frozen_at = datetime(2027, 1, 1, tzinfo=UTC)
    released_at = frozen_at + timedelta(days=1)
    proof = HiddenPublicProof.create(
        proof_id="hpf_" + "1" * 32,
        batch_content_digest="1" * 64,
        decision_bindings_digest="2" * 64,
        bundle_manifest_digest=pooled_ref.sha256,
        defender_top_ref_digest=pooled_ref.sha256,
        worker_manifest_digest="5" * 64,
        evaluator_context_token="6" * 64,
        cohort_mapping_token="7" * 64,
        issued_at="2027-01-02T00:00:00Z",
        signer=hidden_signer,
    )
    release_attestation = orchestration._build_hidden_release_attestation(
        proof=proof,
        signer=hidden_signer,
        ensemble_ref=ensemble_ref,
        pooled_ref=pooled_ref,
        held_family_refs=held_refs,
        evaluation_bundle_ref=evaluation_bundle_ref,
        scorecard_ref=public_artifacts["defense-scorecard.json"],
        promotion_envelope_digest="9" * 64,
        defender_frozen_at=frozen_at,
    )
    class PublicReference:
        def __init__(self, reference: object) -> None:
            self.reference = reference

        def as_artifact_ref(self) -> object:
            return self.reference

    fake_bundle = SimpleNamespace(
        public_artifacts={
            name: PublicReference(reference)
            for name, reference in public_artifacts.items()
        },
        scorecard=lambda **_kwargs: SimpleNamespace(
            champion_decision=SimpleNamespace(
                status=SimpleNamespace(value="no_promotion"),
                model_dump=lambda **_kwargs: {
                    "status": "no_promotion",
                    "failed_gate_codes": ["SYNTHETIC_ONLY"],
                },
            )
        ),
    )
    monkeypatch.setattr(
        orchestration,
        "load_evaluation_bundle",
        lambda *_args, **_kwargs: fake_bundle,
    )
    attestation_attacks = (
        {"authority_issued_at": "2027-01-03T00:00:00Z"},
        {"ensemble_top_ref_digest": "a" * 64},
        {"pooled_defender_ref_digest": "b" * 64},
        {"candidate_roster_digest": "c" * 64},
        {"profile_sha256": "d" * 64},
        {"evaluation_bundle_digest": "e" * 64},
        {"scorecard_digest": "f" * 64},
        {"promotion_envelope_digest": "0" * 64},
    )
    for update in attestation_attacks:
        unsigned_attack = {
            **release_attestation.unsigned_document(),
            **update,
        }
        attacked = orchestration.HiddenReleaseAttestation.model_validate(
            {
                **unsigned_attack,
                "signature_base64": hidden_signer._sign(unsigned_attack),
            }
        )
        with pytest.raises(orchestration.CliContractError, match="inputs differ"):
            orchestration._export_defense_v1_evaluation(
                directory=directory,
                result_path=tmp_path / "attack-result.json",
                hash_manifest_path=directory / "hash-manifest.json",
                store=store,
                signer=signer,
                public_artifacts=public_artifacts,
                evaluation_bundle_ref=evaluation_bundle_ref,
                threshold_set_ref=threshold_set_ref,
                ensemble_ref=ensemble_ref,
                pooled_ref=pooled_ref,
                held_family_refs=held_refs,
                corpus_envelope_ref=corpus_ref,
                run_ledger_sha256="8" * 64,
                promotion_envelope_digest="9" * 64,
                descriptor_scope=("hidden:hidden",),
                status="no_promotion",
                defender_frozen_at=frozen_at,
                hidden_released_at=released_at,
                hidden_release_attestation=attacked,
                hidden_signer_key_id=hidden_signer.key_id,
                hidden_signer_public_key_base64=hidden_signer.public_key_base64,
            )
    attacker = EvaluatorSigningIdentity.from_private_bytes(b"i" * 32)
    attacker_unsigned = {
        **release_attestation.unsigned_document(),
        "signer_key_id": attacker.key_id,
        "public_key_base64": attacker.public_key_base64,
    }
    attacker_attestation = orchestration.HiddenReleaseAttestation.model_validate(
        {
            **attacker_unsigned,
            "signature_base64": attacker._sign(attacker_unsigned),
        }
    )
    with pytest.raises(orchestration.CliContractError, match="inputs differ"):
        orchestration._export_defense_v1_evaluation(
            directory=directory,
            result_path=tmp_path / "attacker-result.json",
            hash_manifest_path=directory / "hash-manifest.json",
            store=store,
            signer=signer,
            public_artifacts=public_artifacts,
            evaluation_bundle_ref=evaluation_bundle_ref,
            threshold_set_ref=threshold_set_ref,
            ensemble_ref=ensemble_ref,
            pooled_ref=pooled_ref,
            held_family_refs=held_refs,
            corpus_envelope_ref=corpus_ref,
            run_ledger_sha256="8" * 64,
            promotion_envelope_digest="9" * 64,
            descriptor_scope=("hidden:hidden",),
            status="no_promotion",
            defender_frozen_at=frozen_at,
            hidden_released_at=released_at,
            hidden_release_attestation=attacker_attestation,
            hidden_signer_key_id=hidden_signer.key_id,
            hidden_signer_public_key_base64=hidden_signer.public_key_base64,
        )
    for collision in ("late-report", "result", "hash"):
        collision_directory = tmp_path / collision
        collision_directory.mkdir()
        for name in sorted(
            orchestration._DEFENSE_V1_FIXTURE_FILES
            - orchestration._DEFENSE_V1_PUBLIC_REPORT_FILES
        ):
            (collision_directory / name).write_bytes(f"frozen:{name}".encode())
        collision_result = tmp_path / f"{collision}-result.json"
        collision_hash = collision_directory / "hash-manifest.json"
        if collision == "late-report":
            (collision_directory / "value-workload.csv").write_bytes(b"occupied")
        elif collision == "result":
            collision_result.write_bytes(b"occupied")
        else:
            collision_hash.write_bytes(b"occupied")
        before = {
            path: path.read_bytes()
            for path in (*collision_directory.iterdir(),)
            if path.is_file()
        }
        with pytest.raises(orchestration.CliContractError, match="preflight"):
            orchestration._export_defense_v1_evaluation(
                directory=collision_directory,
                result_path=collision_result,
                hash_manifest_path=collision_hash,
                store=store,
                signer=signer,
                public_artifacts=public_artifacts,
                evaluation_bundle_ref=evaluation_bundle_ref,
                threshold_set_ref=threshold_set_ref,
                ensemble_ref=ensemble_ref,
                pooled_ref=pooled_ref,
                held_family_refs=held_refs,
                corpus_envelope_ref=corpus_ref,
                run_ledger_sha256="8" * 64,
                promotion_envelope_digest="9" * 64,
                descriptor_scope=("hidden:hidden",),
                status="no_promotion",
                defender_frozen_at=frozen_at,
                hidden_released_at=released_at,
                hidden_release_attestation=release_attestation,
                hidden_signer_key_id=hidden_signer.key_id,
                hidden_signer_public_key_base64=hidden_signer.public_key_base64,
            )
        after = {
            path: path.read_bytes()
            for path in (*collision_directory.iterdir(),)
            if path.is_file()
        }
        assert after == before
        assert not any(
            (collision_directory / name).exists()
            for name in orchestration._DEFENSE_V1_PUBLIC_REPORT_FILES
            if name != "value-workload.csv" or collision != "late-report"
        )
    result_path = tmp_path / "result.json"
    hash_path = directory / "hash-manifest.json"

    orchestration._export_defense_v1_evaluation(
        directory=directory,
        result_path=result_path,
        hash_manifest_path=hash_path,
        store=store,
        signer=signer,
        public_artifacts=public_artifacts,
        evaluation_bundle_ref=evaluation_bundle_ref,
        threshold_set_ref=threshold_set_ref,
        ensemble_ref=ensemble_ref,
        pooled_ref=pooled_ref,
        held_family_refs=held_refs,
        corpus_envelope_ref=corpus_ref,
        run_ledger_sha256="8" * 64,
        promotion_envelope_digest="9" * 64,
        descriptor_scope=("hidden:hidden",),
        status="no_promotion",
        defender_frozen_at=frozen_at,
        hidden_released_at=released_at,
        hidden_release_attestation=release_attestation,
        hidden_signer_key_id=hidden_signer.key_id,
        hidden_signer_public_key_base64=hidden_signer.public_key_base64,
    )

    assert {path.name for path in directory.iterdir()} == {
        *orchestration._DEFENSE_V1_FIXTURE_FILES,
        "hash-manifest.json",
    }
    result = json.loads(result_path.read_bytes())
    signature = result.pop("signature_base64")
    verifier = PublicArtifactVerifier.from_signer(signer)
    assert verifier.verify(result, signature)
    assert result["hidden_released_at"] == proof.issued_at
    assert result["hidden_release_attestation"] == release_attestation.model_dump(
        mode="json"
    )
    assert result["authority_identities"] == {
        role: {
            "key_id": identity.key_id,
            "public_key_base64": identity.public_key_base64,
        }
        for role, identity in authority_identities.items()
    }
    hidden_verifier = orchestration.EvaluatorReplayVerifier(
        signer_key_id=release_attestation.signer_key_id,
        public_key_base64=release_attestation.public_key_base64,
    )
    assert hidden_verifier.verify_document(
        release_attestation.unsigned_document(),
        release_attestation.signature_base64,
    )
    hash_manifest = json.loads(hash_path.read_bytes())
    hash_signature = hash_manifest.pop("signature_base64")
    assert verifier.verify(hash_manifest, hash_signature)
    assert set(hash_manifest["artifact_sha256"]) == (
        orchestration._DEFENSE_V1_FIXTURE_FILES
    )
    with pytest.raises(orchestration.CliContractError, match="preflight"):
        orchestration._export_defense_v1_evaluation(
            directory=directory,
            result_path=result_path,
            hash_manifest_path=hash_path,
            store=store,
            signer=signer,
            public_artifacts=public_artifacts,
            evaluation_bundle_ref=evaluation_bundle_ref,
            threshold_set_ref=threshold_set_ref,
            ensemble_ref=ensemble_ref,
            pooled_ref=pooled_ref,
            held_family_refs=held_refs,
            corpus_envelope_ref=corpus_ref,
            run_ledger_sha256="8" * 64,
            promotion_envelope_digest="9" * 64,
            descriptor_scope=("hidden:hidden",),
            status="no_promotion",
            defender_frozen_at=frozen_at,
            hidden_released_at=released_at,
            hidden_release_attestation=release_attestation,
            hidden_signer_key_id=hidden_signer.key_id,
            hidden_signer_public_key_base64=hidden_signer.public_key_base64,
        )

    for bad_release in (frozen_at, released_at + timedelta(seconds=1)):
        fresh = tmp_path / f"bad-{bad_release.day}-{bad_release.second}"
        fresh.mkdir()
        for name in sorted(
            orchestration._DEFENSE_V1_FIXTURE_FILES
            - orchestration._DEFENSE_V1_PUBLIC_REPORT_FILES
        ):
            (fresh / name).write_bytes(b"frozen")
        with pytest.raises(orchestration.CliContractError, match="inputs differ"):
            orchestration._export_defense_v1_evaluation(
                directory=fresh,
                result_path=tmp_path / f"bad-{bad_release.day}-{bad_release.second}.json",
                hash_manifest_path=fresh / "hash-manifest.json",
                store=store,
                signer=signer,
                public_artifacts=public_artifacts,
                evaluation_bundle_ref=evaluation_bundle_ref,
                threshold_set_ref=threshold_set_ref,
                ensemble_ref=ensemble_ref,
                pooled_ref=pooled_ref,
                held_family_refs=held_refs,
                corpus_envelope_ref=corpus_ref,
                run_ledger_sha256="8" * 64,
                promotion_envelope_digest="9" * 64,
                descriptor_scope=("hidden:hidden",),
                status="no_promotion",
                defender_frozen_at=frozen_at,
                hidden_released_at=bad_release,
                hidden_release_attestation=release_attestation,
                hidden_signer_key_id=hidden_signer.key_id,
                hidden_signer_public_key_base64=hidden_signer.public_key_base64,
            )
