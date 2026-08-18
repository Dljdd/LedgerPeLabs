"""Signed, closed, native-only defender bundle contract tests."""

from __future__ import annotations

import hashlib
import inspect
import platform
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import cast

import catboost  # type: ignore[import-untyped]
import numpy as np
import pyarrow  # type: ignore[import-untyped]
import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]
import pytest
import sklearn  # type: ignore[import-untyped]
from pydantic import ValidationError

import apar.defense.bundle as bundle_module
from apar.contracts.decisions import Action
from apar.contracts.events import EventKind, Rail
from apar.defense.bundle import (
    GENESIS_ROLLBACK_REF,
    BundleContractError,
    BundleLineage,
    DefenderBundleManifest,
    DefenderBundlePublisher,
    EnvironmentLock,
    SourceInventory,
    SourceInventoryEntry,
    current_environment_lock,
)
from apar.defense.calibration import ProbabilityCalibrator, select_calibrator
from apar.defense.contracts import ObservedEvent
from apar.defense.gbdt import (
    CatBoostScorer,
    GbdtTrainingConfig,
    RollingFold,
    train_gbdt,
)
from apar.defense.policy import OperatingBudget
from apar.defense.rules import RuleManifest
from apar.defense.thresholds import ThresholdReport, select_policy_thresholds
from apar.features.builders import FeatureMatrix
from apar.features.catalog import EXPECTED_FEATURE_NAMES, load_feature_catalog
from apar.features.state import FeatureVector, feature_catalog_digest
from apar.runs.runner import RunSigningIdentity
from apar.runs.wire import canonical_json_bytes, strict_json_loads
from apar.storage.artifacts import ArtifactRef, ArtifactStore

T0 = datetime(2026, 8, 1, tzinfo=UTC)
CATALOG_PATH = Path(__file__).resolve().parents[2] / "config" / "defense" / "feature-catalog.json"


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("ascii")).hexdigest()


def _matrix(count: int = 24) -> FeatureMatrix:
    catalog = load_feature_catalog(CATALOG_PATH)
    catalog_digest = feature_catalog_digest(catalog)
    events: list[ObservedEvent] = []
    rows: list[FeatureVector] = []
    for index in range(count):
        event_id = f"bundle-row-{index:03d}"
        decision_at = T0 + timedelta(hours=index)
        events.append(
            ObservedEvent(
                event_id=event_id,
                payment_id=f"bundle-payment-{index:03d}",
                rail=Rail.CARD,
                event_type=EventKind.AUTHORIZATION,
                amount=Decimal(index + 1),
                currency="USD",
                event_time=decision_at,
                available_at=decision_at,
                decision_at=decision_at,
                actor_id=f"actor-{index % 5}",
                counterparty_id=f"counterparty-{index % 7}",
                optional_refs={},
                integrity_status="not_applicable",
                is_decision_point=True,
            )
        )
        values = {
            name: float(((index + 1) * (column + 3)) % 17) / 17.0
            for column, name in enumerate(catalog.names)
        }
        rows.append(
            FeatureVector(
                event_id=event_id,
                decision_at=decision_at,
                source_event_ids=(),
                max_source_available_at=None,
                catalog_digest=catalog_digest,
                values=values,
            )
        )
    return FeatureMatrix(
        events=tuple(events), catalog=catalog, catalog_digest=catalog_digest, rows=tuple(rows)
    )


def _train(matrix: FeatureMatrix) -> CatBoostScorer:
    labels = {row.event_id: int(index % 3 == 0) for index, row in enumerate(matrix.rows[:20])}
    ids = tuple(labels)
    folds = (
        RollingFold(name="fold-1", fit_ids=ids[:8], validation_ids=ids[8:12]),
        RollingFold(name="fold-2", fit_ids=ids[:12], validation_ids=ids[12:16]),
    )
    return train_gbdt(
        matrix,
        labels,
        ids,
        folds,
        GbdtTrainingConfig(depths=(2,), learning_rates=(0.1,), l2_leaf_regs=(3.0,), iterations=8),
        training_cutoff=matrix.rows[19].decision_at,
    )


def _calibrator() -> ProbabilityCalibrator:
    return select_calibrator(
        np.array([0.05, 0.2, 0.7, 0.9] * 3, dtype=np.float64),
        np.array([0, 0, 1, 1] * 3, dtype=np.int8),
        np.array([0.1, 0.3, 0.6, 0.8] * 2, dtype=np.float64),
        np.array([0, 0, 1, 1] * 2, dtype=np.int8),
        min_class_count=50,
    )


def _threshold_report() -> ThresholdReport:
    scores = np.array([0.1, 0.3, 0.7, 0.9], dtype=np.float64)
    return select_policy_thresholds(
        scores,
        np.array([0, 0, 1, 1], dtype=np.int8),
        np.array([Action.APPROVE] * 4, dtype=object),
        lambda actions: int(sum(action is not Action.APPROVE for action in actions)),
        OperatingBudget(
            challenge_rate_max=1.0,
            false_decline_rate_max=1.0,
            review_case_rate_max=1.0,
        ),
    )


@dataclass(frozen=True)
class BundleFixture:
    store: ArtifactStore
    signer: RunSigningIdentity
    publisher: DefenderBundlePublisher
    scorer: CatBoostScorer
    training_matrix: FeatureMatrix
    reload_matrix: FeatureMatrix
    kwargs: dict[str, object]


@pytest.fixture(scope="module")
def trained() -> tuple[FeatureMatrix, CatBoostScorer]:
    matrix = _matrix()
    return matrix, _train(matrix)


@pytest.fixture
def bundle_fixture(tmp_path: Path, trained: tuple[FeatureMatrix, CatBoostScorer]) -> BundleFixture:
    matrix, scorer = trained
    store = ArtifactStore(tmp_path / "artifacts")
    signer = RunSigningIdentity.from_private_bytes(b"b" * 32)
    publisher = DefenderBundlePublisher(store, signer)
    reload_matrix = matrix.model_copy(
        update={"events": matrix.events[20:], "rows": matrix.rows[20:]}
    )
    lineage = BundleLineage(
        corpus_digest=_sha("corpus"),
        observation_dataset_digest=_sha("observations"),
        evaluator_truth_digest=_sha("truth"),
        split_manifest_digest=_sha("split"),
        feature_provenance_digest=_sha("provenance"),
        hyperparameter_digest=hashlib.sha256(
            canonical_json_bytes(scorer.receipt.selected_params.model_dump(mode="json"))
        ).hexdigest(),
        reason_code_mapping_digest=_sha("reasons"),
    )
    inventory = SourceInventory(
        entries=(
            SourceInventoryEntry(path="src/apar/defense/bundle.py", sha256=_sha("bundle.py")),
            SourceInventoryEntry(path="src/apar/defense/gbdt.py", sha256=_sha("gbdt.py")),
        )
    )
    kwargs: dict[str, object] = {
        "scorer": scorer,
        "catalog": matrix.catalog,
        "training_matrix": matrix.model_copy(
            update={"events": matrix.events[:20], "rows": matrix.rows[:20]}
        ),
        "rule_manifest": RuleManifest.default(),
        "calibrator": _calibrator(),
        "threshold_report": _threshold_report(),
        "lineage": lineage,
        "environment_lock": current_environment_lock(),
        "source_inventory": inventory,
        "reload_matrix": reload_matrix,
        "bundle_id": "12345678-1234-5678-9234-567812345678",
        "frozen_at": datetime(2026, 8, 18, 12, tzinfo=UTC),
        "rollback_ref": GENESIS_ROLLBACK_REF,
    }
    return BundleFixture(store, signer, publisher, scorer, matrix, reload_matrix, kwargs)


def _resign(
    manifest: DefenderBundleManifest,
    signer: RunSigningIdentity,
    **updates: object,
) -> DefenderBundleManifest:
    changed = manifest.model_copy(update={**updates, "signature_base64": "A" * 88})
    return changed.model_copy(update={"signature_base64": signer.sign(changed.unsigned_document())})


def test_public_api_is_closed_and_immutable(bundle_fixture: BundleFixture) -> None:
    parameters = inspect.signature(DefenderBundlePublisher.freeze).parameters
    assert tuple(parameters) == (
        "self",
        "scorer",
        "catalog",
        "training_matrix",
        "rule_manifest",
        "calibrator",
        "threshold_report",
        "lineage",
        "environment_lock",
        "source_inventory",
        "reload_matrix",
        "bundle_id",
        "frozen_at",
        "rollback_ref",
    )
    with pytest.raises(ValidationError):
        cast(EnvironmentLock, current_environment_lock()).python_version = "forged"  # type: ignore[misc]

    class StoreSubclass(ArtifactStore):
        pass

    with pytest.raises(BundleContractError, match="exact ArtifactStore"):
        DefenderBundlePublisher(StoreSubclass(Path("unused")), bundle_fixture.signer)


def test_valid_publish_load_is_deterministic_and_reproduces_all_scores(
    bundle_fixture: BundleFixture,
) -> None:
    manifest, ref = bundle_fixture.publisher.freeze(**bundle_fixture.kwargs)
    repeated_manifest, repeated_ref = bundle_fixture.publisher.freeze(**bundle_fixture.kwargs)
    loaded = bundle_fixture.publisher.load(ref)

    assert manifest == repeated_manifest
    assert ref == repeated_ref
    assert loaded.manifest == manifest
    assert loaded.catalog == bundle_fixture.training_matrix.catalog
    assert loaded.rule_manifest == bundle_fixture.kwargs["rule_manifest"]
    assert loaded.training_matrix == bundle_fixture.kwargs["training_matrix"]
    assert loaded.reload_matrix == bundle_fixture.reload_matrix
    assert bundle_fixture.publisher.verify(manifest) is True
    np.testing.assert_allclose(
        loaded.scorer.predict(loaded.reload_matrix),
        bundle_fixture.scorer.predict(bundle_fixture.reload_matrix),
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_allclose(
        loaded.scorer.predict_raw(loaded.reload_matrix),
        bundle_fixture.scorer.predict_raw(bundle_fixture.reload_matrix),
        rtol=0.0,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        loaded.scorer.contributions(loaded.reload_matrix),
        bundle_fixture.scorer.contributions(bundle_fixture.reload_matrix),
        rtol=0.0,
        atol=1e-12,
    )
    expected_calibrated = cast(ProbabilityCalibrator, bundle_fixture.kwargs["calibrator"]).predict(
        bundle_fixture.scorer.predict(bundle_fixture.reload_matrix)
    )
    np.testing.assert_allclose(
        loaded.calibrator.predict(loaded.scorer.predict(loaded.reload_matrix)),
        expected_calibrated,
        rtol=0.0,
        atol=0.0,
    )
    assert strict_json_loads(bundle_fixture.store.read(ref)) == manifest.model_dump(mode="json")


@pytest.mark.parametrize(
    "field",
    (
        "feature_catalog_digest",
        "training_matrix_digest",
        "rule_manifest_digest",
        "model_digest",
        "training_receipt_digest",
        "calibration_digest",
        "threshold_digest",
        "environment_digest",
        "source_inventory_digest",
        "reload_matrix_digest",
        "reload_fixture_digest",
    ),
)
def test_every_component_address_tamper_fails_closed(
    bundle_fixture: BundleFixture, field: str
) -> None:
    manifest, _ = bundle_fixture.publisher.freeze(**bundle_fixture.kwargs)
    changed = _resign(manifest, bundle_fixture.signer, **{field: _sha(f"changed-{field}")})
    assert bundle_fixture.publisher.verify(changed) is False


@pytest.mark.parametrize(
    "field",
    (
        "feature_catalog_digest",
        "training_matrix_digest",
        "rule_manifest_digest",
        "model_digest",
        "training_receipt_digest",
        "calibration_digest",
        "threshold_digest",
        "environment_digest",
        "source_inventory_digest",
        "reload_matrix_digest",
        "reload_fixture_digest",
    ),
)
def test_every_component_media_type_substitution_fails_closed(
    bundle_fixture: BundleFixture, field: str
) -> None:
    manifest, _ = bundle_fixture.publisher.freeze(**bundle_fixture.kwargs)
    wrong = bundle_fixture.store.put_bytes(
        f"wrong-media-{field}".encode("ascii"), "application/octet-stream"
    )
    changed = _resign(manifest, bundle_fixture.signer, **{field: wrong.sha256})
    assert bundle_fixture.publisher.verify(changed) is False


@pytest.mark.parametrize(
    ("field", "media_type"),
    (
        ("feature_catalog_digest", "application/vnd.apar.feature-catalog+json"),
        ("training_matrix_digest", "application/vnd.apache.parquet"),
        ("rule_manifest_digest", "application/vnd.apar.rule-manifest+json"),
        ("model_digest", "application/vnd.apar.catboost-model"),
        ("training_receipt_digest", "application/vnd.apar.training-receipt+json"),
        ("calibration_digest", "application/vnd.apar.calibration+json"),
        ("threshold_digest", "application/vnd.apar.threshold-report+json"),
        ("environment_digest", "application/vnd.apar.environment-lock+json"),
        ("source_inventory_digest", "application/vnd.apar.source-inventory+json"),
        ("reload_matrix_digest", "application/vnd.apache.parquet"),
        ("reload_fixture_digest", "application/vnd.apar.reload-fixture+json"),
    ),
)
def test_every_component_payload_substitution_fails_closed(
    bundle_fixture: BundleFixture, field: str, media_type: str
) -> None:
    manifest, _ = bundle_fixture.publisher.freeze(**bundle_fixture.kwargs)
    replacement = bundle_fixture.store.put_bytes(
        f"hostile-payload-{field}".encode("ascii"), media_type
    )
    changed = _resign(manifest, bundle_fixture.signer, **{field: replacement.sha256})
    assert bundle_fixture.publisher.verify(changed) is False


def test_parquet_schema_metadata_feature_order_and_reload_score_substitution_fail(
    bundle_fixture: BundleFixture,
) -> None:
    manifest, _ = bundle_fixture.publisher.freeze(**bundle_fixture.kwargs)
    original_ref = bundle_fixture.store.resolve(manifest.training_matrix_digest)
    original = bundle_fixture.store.read(original_ref)
    table = pq.read_table(pa.BufferReader(original))
    changed_table = table.replace_schema_metadata({b"pandas": b"forbidden"})
    sink = pa.BufferOutputStream()
    pq.write_table(changed_table, sink, compression="NONE", use_dictionary=False)
    changed_parquet = bundle_fixture.store.put_bytes(
        sink.getvalue().to_pybytes(), "application/vnd.apache.parquet"
    )
    parquet_attack = _resign(
        manifest,
        bundle_fixture.signer,
        training_matrix_digest=changed_parquet.sha256,
    )
    assert bundle_fixture.publisher.verify(parquet_attack) is False

    catalog_document = cast(
        dict[str, object],
        strict_json_loads(
            bundle_fixture.store.read(bundle_fixture.store.resolve(manifest.feature_catalog_digest))
        ),
    )
    features = cast(list[object], catalog_document["features"])
    catalog_document["features"] = list(reversed(features))
    changed_catalog = bundle_fixture.store.put_bytes(
        canonical_json_bytes(catalog_document),
        "application/vnd.apar.feature-catalog+json",
    )
    catalog_attack = _resign(
        manifest,
        bundle_fixture.signer,
        feature_catalog_digest=changed_catalog.sha256,
    )
    assert bundle_fixture.publisher.verify(catalog_attack) is False

    fixture_document = cast(
        dict[str, object],
        strict_json_loads(
            bundle_fixture.store.read(bundle_fixture.store.resolve(manifest.reload_fixture_digest))
        ),
    )
    probability_scores = cast(list[float], fixture_document["probability_scores"])
    probability_scores[0] = min(1.0, probability_scores[0] + 0.01)
    changed_fixture = bundle_fixture.store.put_bytes(
        canonical_json_bytes(fixture_document),
        "application/vnd.apar.reload-fixture+json",
    )
    reload_attack = _resign(
        manifest,
        bundle_fixture.signer,
        reload_fixture_digest=changed_fixture.sha256,
    )
    assert bundle_fixture.publisher.verify(reload_attack) is False


def test_noncanonical_catalog_columns_and_forbidden_source_payload_fail_closed(
    bundle_fixture: BundleFixture,
) -> None:
    manifest, _ = bundle_fixture.publisher.freeze(**bundle_fixture.kwargs)
    forbidden_source = bundle_fixture.store.put_bytes(
        canonical_json_bytes(
            {
                "schema_version": "1.0.0",
                "entries": [{"path": "src/apar/evaluation_hidden/private.py", "sha256": _sha("x")}],
            }
        ),
        "application/vnd.apar.source-inventory+json",
    )
    attack = _resign(
        manifest,
        bundle_fixture.signer,
        source_inventory_digest=forbidden_source.sha256,
    )
    assert bundle_fixture.publisher.verify(attack) is False

    assert tuple(bundle_module._matrix_schema().names[-48:]) == EXPECTED_FEATURE_NAMES
    assert not any(
        forbidden in bundle_module._matrix_schema().names
        for forbidden in ("label", "family", "campaign", "truth", "hidden")
    )


def test_signature_authority_and_self_signed_attacker_are_rejected(
    bundle_fixture: BundleFixture,
) -> None:
    manifest, _ = bundle_fixture.publisher.freeze(**bundle_fixture.kwargs)
    assert (
        bundle_fixture.publisher.verify(manifest.model_copy(update={"signature_base64": "A" * 88}))
        is False
    )
    attacker = RunSigningIdentity.from_private_bytes(b"a" * 32)
    changed = manifest.model_copy(
        update={
            "signer_key_id": attacker.key_id,
            "public_key_base64": attacker.public_key_base64,
            "signature_base64": "A" * 88,
        }
    )
    attack = changed.model_copy(
        update={"signature_base64": attacker.sign(changed.unsigned_document())}
    )
    assert attacker.verify(attack.unsigned_document(), attack.signature_base64)
    assert bundle_fixture.publisher.verify(attack) is False


def test_real_rollback_chain_and_missing_or_forged_predecessors(
    bundle_fixture: BundleFixture,
) -> None:
    first, first_ref = bundle_fixture.publisher.freeze(**bundle_fixture.kwargs)
    second_kwargs = {
        **bundle_fixture.kwargs,
        "bundle_id": "22345678-1234-5678-9234-567812345678",
        "frozen_at": datetime(2026, 8, 18, 13, tzinfo=UTC),
        "rollback_ref": first_ref.sha256,
    }
    second, second_ref = bundle_fixture.publisher.freeze(**second_kwargs)
    assert bundle_fixture.publisher.load(second_ref).manifest == second
    assert second.frozen_at > first.frozen_at

    missing = _resign(second, bundle_fixture.signer, rollback_ref=_sha("missing"))
    forged = _resign(second, bundle_fixture.signer, rollback_ref=second.model_digest)
    assert bundle_fixture.publisher.verify(missing) is False
    assert bundle_fixture.publisher.verify(forged) is False

    attacker = RunSigningIdentity.from_private_bytes(b"a" * 32)
    attacker_publisher = DefenderBundlePublisher(bundle_fixture.store, attacker)
    attacker_kwargs = {
        **bundle_fixture.kwargs,
        "bundle_id": "32345678-1234-5678-9234-567812345678",
        "frozen_at": datetime(2026, 8, 18, 11, tzinfo=UTC),
    }
    _, attacker_ref = attacker_publisher.freeze(**attacker_kwargs)
    forged_predecessor = _resign(second, bundle_fixture.signer, rollback_ref=attacker_ref.sha256)
    assert bundle_fixture.publisher.verify(forged_predecessor) is False


def test_manifest_tamper_and_hostile_verify_never_raise(bundle_fixture: BundleFixture) -> None:
    manifest, _ = bundle_fixture.publisher.freeze(**bundle_fixture.kwargs)
    assert (
        bundle_fixture.publisher.verify(manifest.model_copy(update={"rollback_ref": _sha("x")}))
        is False
    )
    for hostile in (None, 1, {}, b"not-json", object()):
        assert bundle_fixture.publisher.verify(hostile) is False


def test_environment_source_and_threshold_contracts_fail_before_publication(
    bundle_fixture: BundleFixture,
) -> None:
    bad_environment = current_environment_lock().model_copy(update={"pyarrow_version": "0.0.0"})
    with pytest.raises(BundleContractError, match="environment"):
        bundle_fixture.publisher.freeze(
            **{**bundle_fixture.kwargs, "environment_lock": bad_environment}
        )
    with pytest.raises(ValidationError, match="private|hidden|restricted"):
        SourceInventory(
            entries=(
                SourceInventoryEntry(path="apar/evaluation_hidden/private.py", sha256=_sha("x")),
            )
        )
    infeasible = cast(ThresholdReport, bundle_fixture.kwargs["threshold_report"]).model_copy(
        update={"feasible": False}
    )
    with pytest.raises(BundleContractError, match="contract|feasible"):
        bundle_fixture.publisher.freeze(**{**bundle_fixture.kwargs, "threshold_report": infeasible})
    bad_lineage = cast(BundleLineage, bundle_fixture.kwargs["lineage"]).model_copy(
        update={"hyperparameter_digest": _sha("wrong-hyperparameters")}
    )
    with pytest.raises(BundleContractError, match="hyperparameters"):
        bundle_fixture.publisher.freeze(**{**bundle_fixture.kwargs, "lineage": bad_lineage})


def test_load_normalizes_top_reference_and_json_failures(bundle_fixture: BundleFixture) -> None:
    _, ref = bundle_fixture.publisher.freeze(**bundle_fixture.kwargs)
    wrong_media = ArtifactRef(ref.sha256, "application/json", ref.size_bytes, ref.relative_path)
    with pytest.raises(BundleContractError, match="bundle"):
        bundle_fixture.publisher.load(wrong_media)
    noncanonical = bundle_fixture.store.put_bytes(b'{"schema_version": "1.0.0"}', ref.media_type)
    duplicate = bundle_fixture.store.put_bytes(b'{"a":1,"a":1}', ref.media_type)
    for hostile_ref in (noncanonical, duplicate):
        with pytest.raises(BundleContractError):
            bundle_fixture.publisher.load(hostile_ref)


def test_manifest_validators_reject_wrong_types_digests_uuid_time_and_signature(
    bundle_fixture: BundleFixture,
) -> None:
    manifest, _ = bundle_fixture.publisher.freeze(**bundle_fixture.kwargs)
    document = manifest.model_dump(mode="python")
    mutations = (
        {"bundle_id": "not-a-uuid"},
        {"frozen_at": datetime(2026, 8, 18)},
        {"model_digest": "A" * 64},
        {"signature_base64": "AA=="},
        {"public_key_base64": "AA=="},
        {"signer_key_id": True},
    )
    for mutation in mutations:
        with pytest.raises(ValidationError):
            DefenderBundleManifest.model_validate({**document, **mutation})


def test_bundle_uses_native_safe_media_and_leaves_no_catboost_side_effects(
    bundle_fixture: BundleFixture, tmp_path: Path
) -> None:
    manifest, _ = bundle_fixture.publisher.freeze(**bundle_fixture.kwargs)
    assert bundle_fixture.store.resolve(manifest.model_digest).media_type == (
        "application/vnd.apar.catboost-model"
    )
    assert bundle_fixture.store.resolve(manifest.training_matrix_digest).media_type == (
        "application/vnd.apache.parquet"
    )
    payloads = tuple(
        bundle_fixture.store.read(bundle_fixture.store.resolve(digest))
        for digest in manifest.component_digests()
    )
    assert all(not payload.startswith(b"\x80\x04") for payload in payloads)
    assert not (tmp_path / "catboost_info").exists()


def test_current_environment_lock_is_exact_and_complete() -> None:
    lock = current_environment_lock()
    assert lock == EnvironmentLock(
        python_version=platform.python_version(),
        platform=platform.platform(),
        catboost_version=catboost.__version__,
        scikit_learn_version=sklearn.__version__,
        numpy_version=np.__version__,
        pyarrow_version=pyarrow.__version__,
        apar_schema_version="1.0.0",
    )


def test_no_pickle_or_arbitrary_python_deserialization_surface() -> None:
    source = Path(inspect.getfile(DefenderBundlePublisher)).read_text(encoding="utf-8")
    forbidden = ("import pickle", "from pickle", "joblib", "cloudpickle", "dill")
    assert all(token not in source for token in forbidden)
    assert "application/vnd.apar.catboost-model" in source
    assert "application/vnd.apache.parquet" in source
    assert canonical_json_bytes({"safe": True}) == b'{"safe":true}'
