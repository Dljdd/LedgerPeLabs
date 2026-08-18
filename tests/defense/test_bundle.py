"""Signed, closed, native-only defender bundle contract tests."""

from __future__ import annotations

import hashlib
import inspect
import os
import platform
import sys
import sysconfig
from dataclasses import FrozenInstanceError, dataclass
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
    CalibrationBindingReceipt,
    DefenderBundleManifest,
    DefenderBundlePublisher,
    EnvironmentLock,
    InstalledDistribution,
    SourceInventory,
    SourceInventoryEntry,
    ThresholdBindingReceipt,
    TrainingBindingReceipt,
    build_source_inventory,
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
from apar.evaluation.contracts import CorpusManifest, EvaluationTruthRow, FrozenCorpus
from apar.evaluation.splits import EvaluationSplit, SplitConfig, make_evaluation_split
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


def _matrix(count: int = 32) -> FeatureMatrix:
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


def _train(
    matrix: FeatureMatrix,
    mandatory_excluded_row_ids: tuple[str, ...] = (),
) -> CatBoostScorer:
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
        mandatory_row_ids=mandatory_excluded_row_ids,
    )


def _labels_from_scores(scores: np.ndarray) -> np.ndarray:
    labels = np.zeros(len(scores), dtype=np.int64)
    labels[np.argsort(scores)[len(scores) // 2 :]] = 1
    return labels


def _training_labels() -> dict[str, int]:
    return {f"bundle-row-{index:03d}": int(index % 3 == 0) for index in range(20)}


def _review_case_counter(actions: np.ndarray) -> int:
    return int(sum(action is not Action.APPROVE for action in actions))


def _split(matrix: FeatureMatrix, scorer: CatBoostScorer) -> EvaluationSplit:
    training_labels = _training_labels()
    fraud: dict[str, bool] = {event_id: bool(value) for event_id, value in training_labels.items()}
    for start, end in ((20, 24), (24, 28), (28, 32)):
        subset = matrix.model_copy(
            update={"events": matrix.events[start:end], "rows": matrix.rows[start:end]}
        )
        labels = _labels_from_scores(scorer.predict(subset))
        fraud.update(
            {row.event_id: bool(value) for row, value in zip(subset.rows, labels, strict=True)}
        )
    families = (
        "agentic_intent_abuse",
        "app_scam_mule",
        "card_testing_cnp",
        "synthetic_merchant_refund",
    )
    truth = tuple(
        EvaluationTruthRow(
            event_id=event.event_id,
            payment_id=event.payment_id,
            campaign_id=f"campaign-{index:03d}",
            family=families[index % len(families)],
            viewpoint="development",
            is_fraud=fraud[event.event_id],
            label_source="population_truth",
            label_mature_at=T0,
            first_settlement_at=None,
            net_settled_value=Decimal(index + 1),
            lifecycle_event_ids=(event.event_id,),
        )
        for index, event in enumerate(matrix.events)
    )
    corpus = FrozenCorpus(
        observations=matrix.events,
        truth=truth,
        manifest=CorpusManifest(
            profile_id="bundle-split",
            run_ids=("run-bundle",),
            run_lineage_digests=(_sha("run-bundle"),),
            observation_count=len(matrix.events),
            truth_count=len(truth),
        ),
    )
    return make_evaluation_split(
        corpus,
        SplitConfig(
            train_end=T0 + timedelta(hours=19),
            calibrator_fit_end=T0 + timedelta(hours=23),
            threshold_end=T0 + timedelta(hours=27),
            development_end=T0 + timedelta(hours=31),
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
    split: EvaluationSplit
    source_root: Path
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
    source_root = tmp_path / "source"
    source_root.mkdir()
    (source_root / "bundle.py").write_bytes(b"bundle-source\n")
    (source_root / "gbdt.py").write_bytes(b"gbdt-source\n")
    publisher = DefenderBundlePublisher(store, signer, source_root)
    split = _split(matrix, scorer)
    training_matrix = matrix.model_copy(
        update={"events": matrix.events[:20], "rows": matrix.rows[:20]}
    )
    fit_matrix = matrix.model_copy(
        update={"events": matrix.events[20:24], "rows": matrix.rows[20:24]}
    )
    selection_matrix = matrix.model_copy(
        update={"events": matrix.events[24:28], "rows": matrix.rows[24:28]}
    )
    reload_matrix = matrix.model_copy(
        update={"events": matrix.events[28:], "rows": matrix.rows[28:]}
    )
    fit_labels = np.asarray(
        [int(split.row_is_fraud[row.event_id]) for row in fit_matrix.rows], dtype=np.int64
    )
    selection_labels = np.asarray(
        [int(split.row_is_fraud[row.event_id]) for row in selection_matrix.rows],
        dtype=np.int64,
    )
    calibrator = select_calibrator(
        scorer.predict(fit_matrix),
        fit_labels,
        scorer.predict(selection_matrix),
        selection_labels,
        min_class_count=50,
    )
    threshold_values = np.asarray(
        [float(split.row_net_settled_values[row.event_id]) for row in selection_matrix.rows],
        dtype=np.float64,
    )
    mandatory_actions = np.array([Action.APPROVE] * len(selection_matrix.rows), dtype=object)
    threshold_report = select_policy_thresholds(
        calibrator.predict(scorer.predict(selection_matrix)),
        selection_labels,
        mandatory_actions,
        _review_case_counter,
        OperatingBudget(
            challenge_rate_max=1.0,
            false_decline_rate_max=1.0,
            review_case_rate_max=1.0,
        ),
        threshold_values,
    )
    lineage = BundleLineage(
        corpus_digest=_sha("corpus"),
        observation_dataset_digest=_sha("observations"),
        evaluator_truth_digest=_sha("truth"),
        split_manifest_digest=split.split_digest,
        feature_provenance_digest=_sha("provenance"),
        hyperparameter_digest=hashlib.sha256(
            canonical_json_bytes(scorer.receipt.selected_params.model_dump(mode="json"))
        ).hexdigest(),
        reason_code_mapping_digest=_sha("reasons"),
    )
    inventory = build_source_inventory(source_root, ("bundle.py", "gbdt.py"))
    kwargs: dict[str, object] = {
        "scorer": scorer,
        "catalog": matrix.catalog,
        "split": split,
        "training_matrix": training_matrix,
        "calibration_fit_matrix": fit_matrix,
        "calibration_fit_labels": fit_labels,
        "calibration_selection_matrix": selection_matrix,
        "calibration_selection_labels": selection_labels,
        "threshold_matrix": selection_matrix,
        "threshold_labels": selection_labels,
        "threshold_mandatory_actions": mandatory_actions,
        "threshold_values": threshold_values,
        "review_case_counter": _review_case_counter,
        "rule_manifest": RuleManifest.default(),
        "calibrator": calibrator,
        "threshold_report": threshold_report,
        "lineage": lineage,
        "environment_lock": current_environment_lock(),
        "source_inventory": inventory,
        "reload_matrix": reload_matrix,
        "bundle_id": "12345678-1234-5678-9234-567812345678",
        "frozen_at": datetime(2026, 8, 18, 12, tzinfo=UTC),
        "rollback_ref": GENESIS_ROLLBACK_REF,
    }
    return BundleFixture(
        store,
        signer,
        publisher,
        scorer,
        matrix,
        reload_matrix,
        split,
        source_root,
        kwargs,
    )


def _resign(
    manifest: DefenderBundleManifest,
    signer: RunSigningIdentity,
    **updates: object,
) -> DefenderBundleManifest:
    changed = manifest.model_copy(update={**updates, "signature_base64": "A" * 88})
    return changed.model_copy(update={"signature_base64": signer.sign(changed.unsigned_document())})


def _resign_component(
    manifest: DefenderBundleManifest,
    signer: RunSigningIdentity,
    name: str,
    ref: ArtifactRef,
) -> DefenderBundleManifest:
    field_name = bundle_module._COMPONENT_FIELD_MEDIA[name][0]
    components = tuple(
        component.model_copy(
            update={
                "sha256": ref.sha256,
                "media_type": ref.media_type,
                "size_bytes": ref.size_bytes,
            }
        )
        if component.name == name
        else component
        for component in manifest.components
    )
    return _resign(manifest, signer, **{field_name: ref.sha256, "components": components})


def test_public_api_is_closed_and_immutable(bundle_fixture: BundleFixture) -> None:
    parameters = inspect.signature(DefenderBundlePublisher.freeze).parameters
    assert tuple(parameters) == (
        "self",
        "scorer",
        "catalog",
        "split",
        "training_matrix",
        "mandatory_excluded_row_ids",
        "calibration_fit_matrix",
        "calibration_fit_labels",
        "calibration_selection_matrix",
        "calibration_selection_labels",
        "threshold_matrix",
        "threshold_labels",
        "threshold_mandatory_actions",
        "threshold_values",
        "review_case_counter",
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
        DefenderBundlePublisher(
            StoreSubclass(Path("unused")), bundle_fixture.signer, bundle_fixture.source_root
        )


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
    assert len(manifest.components) == 18
    assert len({component.name for component in manifest.components}) == 18
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


def test_loaded_mutable_contracts_are_copy_on_access_and_cannot_change_signed_state(
    bundle_fixture: BundleFixture,
) -> None:
    manifest, ref = bundle_fixture.publisher.freeze(**bundle_fixture.kwargs)
    loaded = bundle_fixture.publisher.load(ref)
    feature = EXPECTED_FEATURE_NAMES[0]
    original_value = loaded.training_matrix.rows[0].values[feature]

    first_training = loaded.training_matrix
    first_training.rows[0].values[feature] = original_value + 999.0
    first_reload = loaded.reload_matrix
    first_reload.events[0].optional_refs["device_id"] = "mutated"
    first_calibrator = loaded.calibrator
    first_calibrator.__dict__["artifact"] = object()

    assert loaded.training_matrix.rows[0].values[feature] == original_value
    assert "device_id" not in loaded.reload_matrix.events[0].optional_refs
    assert not hasattr(loaded, "split")
    assert loaded.calibrator == bundle_fixture.kwargs["calibrator"]
    loaded.verify_reload()
    assert bundle_fixture.publisher.verify(manifest)


def test_loaded_scorer_mutation_cannot_change_private_signed_runtime(
    bundle_fixture: BundleFixture,
) -> None:
    _, ref = bundle_fixture.publisher.freeze(**bundle_fixture.kwargs)
    loaded = bundle_fixture.publisher.load(ref)
    expected = loaded.scorer.predict(loaded.reload_matrix)
    external_scorer = loaded.scorer

    external_scorer._model.set_scale_and_bias(2.0, 0.0)

    np.testing.assert_allclose(
        loaded.scorer.predict(loaded.reload_matrix),
        expected,
        rtol=0.0,
        atol=0.0,
    )
    loaded.verify_reload()


def test_loaded_runtime_is_sealed_and_retains_no_evaluator_truth(
    bundle_fixture: BundleFixture,
) -> None:
    _, ref = bundle_fixture.publisher.freeze(**bundle_fixture.kwargs)
    loaded = bundle_fixture.publisher.load(ref)
    pending: list[object] = [loaded]
    seen: set[int] = set()
    slot_names: list[str] = []
    retained_bytes: list[bytes] = []
    while pending:
        value = pending.pop()
        if id(value) in seen:
            continue
        seen.add(id(value))
        for base in type(value).__mro__:
            for name in cast(tuple[str, ...], getattr(base, "__slots__", ())):
                slot_names.append(name)
                retained = getattr(value, name)
                if isinstance(retained, bytes):
                    retained_bytes.append(retained)
                elif type(retained).__module__ == bundle_module.__name__:
                    pending.append(retained)
    truth_tokens = (
        b'"row_is_fraud"',
        b'"row_families"',
        b'"row_campaigns"',
        b'"row_net_settled_values"',
        b"agentic_intent_abuse",
        b"campaign-024",
    )

    assert "_split_bytes" not in slot_names
    assert not hasattr(loaded, "__dict__")
    assert all(token not in payload for token in truth_tokens for payload in retained_bytes)
    for name in (
        "_model_bytes",
        "_threshold_bytes",
        "_reload_bytes",
        "_manifest",
        "_snapshot",
    ):
        with pytest.raises((FrozenInstanceError, AttributeError)):
            setattr(loaded, name, getattr(loaded, name, b"forged"))
    with pytest.raises((FrozenInstanceError, AttributeError)):
        loaded._snapshot.model_bytes = b"forged"


def test_freeze_rederives_calibration_thresholds_and_enforces_split_roles(
    bundle_fixture: BundleFixture,
) -> None:
    kwargs = bundle_fixture.kwargs
    fit = cast(FeatureMatrix, kwargs["calibration_fit_matrix"])
    selection = cast(FeatureMatrix, kwargs["calibration_selection_matrix"])
    fit_labels = cast(np.ndarray, kwargs["calibration_fit_labels"])
    selection_labels = cast(np.ndarray, kwargs["calibration_selection_labels"])
    unrelated = select_calibrator(
        np.array([0.1, 0.2, 0.8, 0.9], dtype=np.float64),
        np.array([0, 0, 1, 1], dtype=np.int64),
        np.array([0.15, 0.25, 0.75, 0.85], dtype=np.float64),
        np.array([0, 0, 1, 1], dtype=np.int64),
        min_class_count=50,
    )
    with pytest.raises(BundleContractError, match="calibrator"):
        bundle_fixture.publisher.freeze(**{**kwargs, "calibrator": unrelated})
    with pytest.raises(BundleContractError, match="calibration fit matrix rows"):
        bundle_fixture.publisher.freeze(
            **{
                **kwargs,
                "calibration_fit_matrix": selection,
                "calibration_fit_labels": selection_labels,
                "calibration_selection_matrix": fit,
                "calibration_selection_labels": fit_labels,
            }
        )

    reversed_threshold = selection.model_copy(
        update={
            "events": tuple(reversed(selection.events)),
            "rows": tuple(reversed(selection.rows)),
        }
    )
    with pytest.raises(BundleContractError, match="threshold matrix rows"):
        bundle_fixture.publisher.freeze(**{**kwargs, "threshold_matrix": reversed_threshold})
    with pytest.raises(BundleContractError, match="reload matrix rows"):
        bundle_fixture.publisher.freeze(**{**kwargs, "reload_matrix": fit})

    unrelated_report = select_policy_thresholds(
        np.array([0.1, 0.2, 0.8, 0.9], dtype=np.float64),
        np.array([0, 0, 1, 1], dtype=np.int64),
        np.array([Action.APPROVE] * 4, dtype=object),
        _review_case_counter,
        cast(ThresholdReport, kwargs["threshold_report"]).budget,
        cast(np.ndarray, kwargs["threshold_values"]),
    )
    with pytest.raises(BundleContractError, match="threshold report"):
        bundle_fixture.publisher.freeze(**{**kwargs, "threshold_report": unrelated_report})


def test_binding_receipts_are_signed_without_exposing_split_truth(
    bundle_fixture: BundleFixture,
) -> None:
    manifest, ref = bundle_fixture.publisher.freeze(**bundle_fixture.kwargs)
    loaded = bundle_fixture.publisher.load(ref)
    assert type(loaded.calibration_binding) is CalibrationBindingReceipt
    assert type(loaded.threshold_binding) is ThresholdBindingReceipt
    assert not hasattr(loaded, "split")
    assert loaded.calibration_binding.split_artifact_digest == manifest.split_artifact_digest
    assert loaded.threshold_binding.threshold_report_digest == loaded.threshold_report.report_digest
    assert loaded.threshold_binding.mandatory_actions == (Action.APPROVE,) * 4


@pytest.mark.parametrize(
    ("component_name", "field_name"),
    (
        ("calibration_binding", "fit_labels_digest"),
        ("calibration_binding", "selection_labels_digest"),
        ("threshold_binding", "labels_digest"),
        ("threshold_binding", "mandatory_actions_digest"),
        ("threshold_binding", "values_digest"),
    ),
)
def test_resigned_binding_receipt_cannot_substitute_evaluator_inputs(
    bundle_fixture: BundleFixture,
    component_name: str,
    field_name: str,
) -> None:
    manifest, _ = bundle_fixture.publisher.freeze(**bundle_fixture.kwargs)
    descriptor = manifest.component(component_name)
    document = cast(
        dict[str, object],
        strict_json_loads(
            bundle_fixture.store.read(bundle_fixture.store.resolve(descriptor.sha256))
        ),
    )
    document[field_name] = _sha(f"substituted-{field_name}")
    changed_ref = bundle_fixture.store.put_bytes(
        canonical_json_bytes(document), descriptor.media_type
    )
    attack = _resign_component(
        manifest,
        bundle_fixture.signer,
        component_name,
        changed_ref,
    )

    assert bundle_fixture.publisher.verify(attack) is False


def test_threshold_binding_preserves_optional_no_value_objective(
    bundle_fixture: BundleFixture,
) -> None:
    kwargs = bundle_fixture.kwargs
    scorer = bundle_fixture.scorer
    calibrator = cast(ProbabilityCalibrator, kwargs["calibrator"])
    matrix = cast(FeatureMatrix, kwargs["threshold_matrix"])
    labels = cast(np.ndarray, kwargs["threshold_labels"])
    actions = cast(np.ndarray, kwargs["threshold_mandatory_actions"])
    report = select_policy_thresholds(
        calibrator.predict(scorer.predict(matrix)),
        labels,
        actions,
        _review_case_counter,
        cast(ThresholdReport, kwargs["threshold_report"]).budget,
        None,
    )

    manifest, ref = bundle_fixture.publisher.freeze(
        **{
            **kwargs,
            "threshold_values": None,
            "threshold_report": report,
            "bundle_id": "42345678-1234-5678-9234-567812345678",
        }
    )

    loaded = bundle_fixture.publisher.load(ref)
    assert manifest.threshold_binding_digest == loaded.manifest.threshold_binding_digest
    assert loaded.threshold_binding.values_digest is None
    assert loaded.threshold_report.input_values_digest is None


def _kwargs_for_excluded_scorer(
    bundle_fixture: BundleFixture,
    scorer: CatBoostScorer,
    excluded: tuple[str, ...],
) -> dict[str, object]:
    kwargs = bundle_fixture.kwargs
    fit = cast(FeatureMatrix, kwargs["calibration_fit_matrix"])
    selection = cast(FeatureMatrix, kwargs["calibration_selection_matrix"])
    fit_labels = cast(np.ndarray, kwargs["calibration_fit_labels"])
    selection_labels = cast(np.ndarray, kwargs["calibration_selection_labels"])
    actions = cast(np.ndarray, kwargs["threshold_mandatory_actions"])
    values = cast(np.ndarray, kwargs["threshold_values"])
    calibrator = select_calibrator(
        scorer.predict(fit),
        fit_labels,
        scorer.predict(selection),
        selection_labels,
        min_class_count=50,
    )
    threshold_report = select_policy_thresholds(
        calibrator.predict(scorer.predict(selection)),
        selection_labels,
        actions,
        _review_case_counter,
        cast(ThresholdReport, kwargs["threshold_report"]).budget,
        values,
    )
    lineage = cast(BundleLineage, kwargs["lineage"]).model_copy(
        update={
            "hyperparameter_digest": hashlib.sha256(
                canonical_json_bytes(scorer.receipt.selected_params.model_dump(mode="json"))
            ).hexdigest()
        }
    )
    return {
        **kwargs,
        "scorer": scorer,
        "mandatory_excluded_row_ids": excluded,
        "calibrator": calibrator,
        "threshold_report": threshold_report,
        "lineage": lineage,
        "bundle_id": "62345678-1234-5678-9234-567812345678",
    }


def test_training_binding_supports_one_mandatory_exclusion(
    bundle_fixture: BundleFixture,
) -> None:
    training = cast(FeatureMatrix, bundle_fixture.kwargs["training_matrix"])
    excluded = ("bundle-row-005",)
    scorer = _train(training, excluded)
    kwargs = _kwargs_for_excluded_scorer(bundle_fixture, scorer, excluded)

    manifest, ref = bundle_fixture.publisher.freeze(**kwargs)
    loaded = bundle_fixture.publisher.load(ref)
    binding = loaded.training_binding

    assert scorer.receipt.requested_training_count == 20
    assert scorer.receipt.mandatory_excluded_count == 1
    assert scorer.receipt.final_training_count == 19
    assert type(binding) is TrainingBindingReceipt
    assert binding.requested_row_ids == tuple(row.event_id for row in training.rows)
    assert binding.excluded_row_ids == excluded
    assert binding.final_fit_row_ids == tuple(
        row.event_id for row in training.rows if row.event_id not in excluded
    )
    assert binding.training_receipt_digest == manifest.training_receipt_digest


def test_training_binding_rejects_excluded_id_order_count_and_digest_tamper(
    bundle_fixture: BundleFixture,
) -> None:
    training = cast(FeatureMatrix, bundle_fixture.kwargs["training_matrix"])
    excluded = ("bundle-row-005",)
    scorer = _train(training, excluded)
    kwargs = _kwargs_for_excluded_scorer(bundle_fixture, scorer, excluded)
    with pytest.raises(BundleContractError, match="excluded"):
        bundle_fixture.publisher.freeze(
            **{**kwargs, "mandatory_excluded_row_ids": ("bundle-row-006",)}
        )

    ordered = ("bundle-row-005", "bundle-row-006")
    scorer_two = _train(training, ordered)
    kwargs_two = _kwargs_for_excluded_scorer(bundle_fixture, scorer_two, ordered)
    with pytest.raises(BundleContractError, match="order"):
        bundle_fixture.publisher.freeze(
            **{**kwargs_two, "mandatory_excluded_row_ids": tuple(reversed(ordered))}
        )

    manifest, _ = bundle_fixture.publisher.freeze(**kwargs)
    descriptor = manifest.component("training_binding")
    original = cast(
        dict[str, object],
        strict_json_loads(
            bundle_fixture.store.read(bundle_fixture.store.resolve(descriptor.sha256))
        ),
    )
    for field_name, replacement in (
        ("excluded_count", 2),
        ("excluded_row_ids_digest", _sha("wrong-excluded-digest")),
        ("final_fit_row_ids_digest", _sha("wrong-final-digest")),
    ):
        document = {**original, field_name: replacement}
        changed_ref = bundle_fixture.store.put_bytes(
            canonical_json_bytes(document), descriptor.media_type
        )
        attack = _resign_component(
            manifest,
            bundle_fixture.signer,
            "training_binding",
            changed_ref,
        )
        assert bundle_fixture.publisher.verify(attack) is False


@pytest.mark.parametrize(
    "field",
    (
        "feature_catalog_digest",
        "split_artifact_digest",
        "training_matrix_digest",
        "training_binding_digest",
        "calibration_fit_matrix_digest",
        "calibration_selection_matrix_digest",
        "threshold_matrix_digest",
        "rule_manifest_digest",
        "model_digest",
        "training_receipt_digest",
        "calibration_digest",
        "calibration_binding_digest",
        "threshold_digest",
        "threshold_binding_digest",
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
        "split_artifact_digest",
        "training_matrix_digest",
        "training_binding_digest",
        "calibration_fit_matrix_digest",
        "calibration_selection_matrix_digest",
        "threshold_matrix_digest",
        "rule_manifest_digest",
        "model_digest",
        "training_receipt_digest",
        "calibration_digest",
        "calibration_binding_digest",
        "threshold_digest",
        "threshold_binding_digest",
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
        ("split_artifact_digest", "application/vnd.apar.evaluation-split+json"),
        ("training_matrix_digest", "application/vnd.apache.parquet"),
        ("training_binding_digest", "application/vnd.apar.training-binding+json"),
        ("calibration_fit_matrix_digest", "application/vnd.apache.parquet"),
        ("calibration_selection_matrix_digest", "application/vnd.apache.parquet"),
        ("threshold_matrix_digest", "application/vnd.apache.parquet"),
        ("rule_manifest_digest", "application/vnd.apar.rule-manifest+json"),
        ("model_digest", "application/vnd.apar.catboost-model"),
        ("training_receipt_digest", "application/vnd.apar.training-receipt+json"),
        ("calibration_digest", "application/vnd.apar.calibration+json"),
        ("calibration_binding_digest", "application/vnd.apar.calibration-binding+json"),
        ("threshold_digest", "application/vnd.apar.threshold-report+json"),
        ("threshold_binding_digest", "application/vnd.apar.threshold-binding+json"),
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
    attacker_publisher = DefenderBundlePublisher(
        bundle_fixture.store, attacker, bundle_fixture.source_root
    )
    attacker_kwargs = {
        **bundle_fixture.kwargs,
        "bundle_id": "32345678-1234-5678-9234-567812345678",
        "frozen_at": datetime(2026, 8, 18, 11, tzinfo=UTC),
    }
    _, attacker_ref = attacker_publisher.freeze(**attacker_kwargs)
    forged_predecessor = _resign(second, bundle_fixture.signer, rollback_ref=attacker_ref.sha256)
    assert bundle_fixture.publisher.verify(forged_predecessor) is False


def test_rollback_ancestry_is_manifest_only_and_resource_bounded(
    bundle_fixture: BundleFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, first_ref = bundle_fixture.publisher.freeze(**bundle_fixture.kwargs)
    second_kwargs = {
        **bundle_fixture.kwargs,
        "bundle_id": "52345678-1234-5678-9234-567812345678",
        "frozen_at": datetime(2026, 8, 18, 13, tzinfo=UTC),
        "rollback_ref": first_ref.sha256,
    }
    second, second_ref = bundle_fixture.publisher.freeze(**second_kwargs)
    original_read = ArtifactStore.read
    reads: list[str] = []

    def tracked_read(store: ArtifactStore, ref: ArtifactRef) -> bytes:
        reads.append(ref.sha256)
        return original_read(store, ref)

    monkeypatch.setattr(ArtifactStore, "read", tracked_read)
    bundle_fixture.publisher.load(second_ref)
    assert reads.count(second.model_digest) == 1

    monkeypatch.setattr(bundle_module, "_MAX_ROLLBACK_DEPTH", 0)
    with pytest.raises(BundleContractError, match="depth"):
        bundle_fixture.publisher.load(second_ref)

    monkeypatch.setattr(bundle_module, "_MAX_ROLLBACK_DEPTH", 32)
    monkeypatch.setattr(bundle_module, "_MAX_ROLLBACK_MANIFEST_BYTES", 1)
    with pytest.raises(BundleContractError, match="byte budget"):
        bundle_fixture.publisher.load(second_ref)


def test_manifest_tamper_and_hostile_verify_never_raise(bundle_fixture: BundleFixture) -> None:
    manifest, _ = bundle_fixture.publisher.freeze(**bundle_fixture.kwargs)
    assert (
        bundle_fixture.publisher.verify(manifest.model_copy(update={"rollback_ref": _sha("x")}))
        is False
    )
    for hostile in (None, 1, {}, b"not-json", object()):
        assert bundle_fixture.publisher.verify(hostile) is False


def test_manifest_revalidates_distinct_lineage_independent_of_signature(
    bundle_fixture: BundleFixture,
) -> None:
    manifest, _ = bundle_fixture.publisher.freeze(**bundle_fixture.kwargs)
    attack = _resign(
        manifest,
        bundle_fixture.signer,
        corpus_digest=manifest.observation_dataset_digest,
    )

    assert bundle_fixture.publisher.verify(attack) is False


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


def test_source_inventory_hashes_are_verified_at_freeze_verify_and_load(
    bundle_fixture: BundleFixture,
) -> None:
    inventory = cast(SourceInventory, bundle_fixture.kwargs["source_inventory"])
    bogus_entry = inventory.entries[0].model_copy(update={"sha256": _sha("bogus")})
    bogus = inventory.model_copy(update={"entries": (bogus_entry, *inventory.entries[1:])})
    with pytest.raises(BundleContractError, match="hash mismatch"):
        bundle_fixture.publisher.freeze(**{**bundle_fixture.kwargs, "source_inventory": bogus})

    manifest, ref = bundle_fixture.publisher.freeze(**bundle_fixture.kwargs)
    source = bundle_fixture.source_root / "bundle.py"
    original = source.read_bytes()
    source.write_bytes(b"changed-after-freeze\n")
    try:
        assert bundle_fixture.publisher.verify(manifest) is False
        with pytest.raises(BundleContractError, match="source inventory"):
            bundle_fixture.publisher.load(ref)
    finally:
        source.write_bytes(original)
    assert bundle_fixture.publisher.verify(manifest)


def test_source_root_rejects_symlinks_aliases_and_noncanonical_paths(tmp_path: Path) -> None:
    root = tmp_path / "source"
    root.mkdir()
    (root / "real.py").write_bytes(b"safe\n")
    (root / "directory").mkdir()
    (root / "directory" / "nested.py").write_bytes(b"nested\n")
    (root / "link.py").symlink_to(root / "real.py")
    (root / "parent-link").symlink_to(root / "directory", target_is_directory=True)

    with pytest.raises(BundleContractError, match="symlink"):
        build_source_inventory(root, ("link.py",))
    with pytest.raises(BundleContractError, match="symlink"):
        build_source_inventory(root, ("parent-link/nested.py",))
    for path in ("../real.py", "/real.py", "directory\\nested.py", "e\u0301.py"):
        with pytest.raises((BundleContractError, ValidationError)):
            build_source_inventory(root, (path,))
    with pytest.raises(ValidationError, match="casefold"):
        SourceInventory(
            entries=(
                SourceInventoryEntry(path="A.py", sha256=_sha("a")),
                SourceInventoryEntry(path="a.py", sha256=_sha("b")),
            )
        )

    root_link = tmp_path / "source-link"
    root_link.symlink_to(root, target_is_directory=True)
    with pytest.raises(BundleContractError, match="non-symlink"):
        DefenderBundlePublisher(
            ArtifactStore(tmp_path / "artifacts"),
            RunSigningIdentity.from_private_bytes(b"s" * 32),
            root_link,
        )


def test_source_reader_rejects_parent_swap_between_check_and_final_open(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "source"
    parent = root / "parent"
    parent.mkdir(parents=True)
    (parent / "tracked.py").write_bytes(b"original\n")
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "tracked.py").write_bytes(b"escaped\n")
    held = root / "held-parent"
    original_open = os.open
    attacked = False
    opened_fds: list[int] = []

    def attacking_open(
        path: str | bytes | os.PathLike[str] | os.PathLike[bytes],
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        nonlocal attacked
        if not attacked and Path(os.fsdecode(path)).name == "tracked.py":
            parent.rename(held)
            parent.symlink_to(outside, target_is_directory=True)
            attacked = True
        if dir_fd is None:
            descriptor = original_open(path, flags, mode)
        else:
            descriptor = original_open(path, flags, mode, dir_fd=dir_fd)
        opened_fds.append(descriptor)
        return descriptor

    monkeypatch.setattr(bundle_module.os, "open", attacking_open)
    try:
        with pytest.raises(BundleContractError, match="changed|source"):
            build_source_inventory(root, ("parent/tracked.py",))
    finally:
        if parent.is_symlink():
            parent.unlink()
        if held.exists():
            held.rename(parent)
    for descriptor in opened_fds:
        with pytest.raises(OSError):
            os.fstat(descriptor)


def test_publisher_root_fd_context_and_closed_state_are_explicit(
    bundle_fixture: BundleFixture,
) -> None:
    publisher = bundle_fixture.publisher
    manifest, ref = publisher.freeze(**bundle_fixture.kwargs)
    root_fd = publisher._source_root_fd
    os.fstat(root_fd)

    publisher.close()
    publisher.close()

    with pytest.raises(OSError):
        os.fstat(root_fd)
    with pytest.raises(BundleContractError, match="closed"):
        publisher.freeze(**bundle_fixture.kwargs)
    with pytest.raises(BundleContractError, match="closed"):
        publisher.load(ref)
    assert publisher.verify(manifest) is False

    context_publisher = DefenderBundlePublisher(
        bundle_fixture.store,
        bundle_fixture.signer,
        bundle_fixture.source_root,
    )
    context_fd = context_publisher._source_root_fd
    with context_publisher as entered:
        assert entered is context_publisher
        os.fstat(context_fd)
    with pytest.raises(OSError):
        os.fstat(context_fd)


@pytest.mark.parametrize(
    ("field", "replacement"),
    (
        ("pydantic_version", "0.0.0"),
        ("cryptography_version", "0.0.0"),
        ("pandas_version", "0.0.0"),
        ("python_cache_tag", "forged-cache"),
        ("python_soabi", "forged-soabi"),
    ),
)
def test_environment_closure_rejects_core_and_abi_tamper(
    bundle_fixture: BundleFixture, field: str, replacement: str
) -> None:
    environment = current_environment_lock().model_copy(update={field: replacement})
    with pytest.raises(BundleContractError, match="environment"):
        bundle_fixture.publisher.freeze(
            **{**bundle_fixture.kwargs, "environment_lock": environment}
        )


def test_environment_distribution_inventory_rejects_tamper_and_duplicates(
    bundle_fixture: BundleFixture,
) -> None:
    environment = current_environment_lock()
    first = environment.installed_distributions[0]
    changed = first.model_copy(update={"version": "0.0.0"})
    tampered = environment.model_copy(
        update={"installed_distributions": (changed, *environment.installed_distributions[1:])}
    )
    with pytest.raises(BundleContractError, match="environment"):
        bundle_fixture.publisher.freeze(**{**bundle_fixture.kwargs, "environment_lock": tampered})
    with pytest.raises(ValidationError, match="unique"):
        EnvironmentLock.model_validate(
            environment.model_dump(mode="python") | {"installed_distributions": (first, first)}
        )


def test_component_and_top_reference_size_limits_fail_before_load(
    bundle_fixture: BundleFixture,
) -> None:
    manifest, ref = bundle_fixture.publisher.freeze(**bundle_fixture.kwargs)
    oversized_ref = ArtifactRef(ref.sha256, ref.media_type, 2 * 1024 * 1024, ref.relative_path)
    with pytest.raises(BundleContractError, match="size"):
        bundle_fixture.publisher.load(oversized_ref)
    environment_component = manifest.component("environment").model_copy(
        update={"size_bytes": 17 * 1024 * 1024}
    )
    components = tuple(
        environment_component if item.name == "environment" else item
        for item in manifest.components
    )
    changed = _resign(manifest, bundle_fixture.signer, components=components)
    assert bundle_fixture.publisher.verify(changed) is False


@pytest.mark.parametrize("compression", ("ZSTD", "GZIP"))
def test_parquet_resource_contract_rejects_compression_and_extra_row_groups(
    bundle_fixture: BundleFixture, compression: str
) -> None:
    manifest, _ = bundle_fixture.publisher.freeze(**bundle_fixture.kwargs)
    payload = bundle_fixture.store.read(
        bundle_fixture.store.resolve(manifest.training_matrix_digest)
    )
    table = pq.read_table(pa.BufferReader(payload))
    sink = pa.BufferOutputStream()
    pq.write_table(table, sink, compression=compression, row_group_size=1)
    changed_ref = bundle_fixture.store.put_bytes(
        sink.getvalue().to_pybytes(), "application/vnd.apache.parquet"
    )
    attack = _resign_component(manifest, bundle_fixture.signer, "training_matrix", changed_ref)
    assert bundle_fixture.publisher.verify(attack) is False


@pytest.mark.parametrize(
    "bound_name",
    ("_MAX_PARQUET_ROWS", "_MAX_PARQUET_DECODED_BYTES"),
)
def test_parquet_resource_contract_enforces_row_and_decoded_byte_budgets(
    bundle_fixture: BundleFixture,
    monkeypatch: pytest.MonkeyPatch,
    bound_name: str,
) -> None:
    manifest, ref = bundle_fixture.publisher.freeze(**bundle_fixture.kwargs)
    monkeypatch.setattr(bundle_module, bound_name, 1)

    assert bundle_fixture.publisher.verify(manifest) is False
    with pytest.raises(BundleContractError, match="Parquet|bundle load"):
        bundle_fixture.publisher.load(ref)


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
    assert lock.python_version == platform.python_version()
    assert lock.platform == platform.platform()
    assert lock.catboost_version == catboost.__version__
    assert lock.scikit_learn_version == sklearn.__version__
    assert lock.numpy_version == np.__version__
    assert lock.pyarrow_version == pyarrow.__version__
    assert lock.python_cache_tag == sys.implementation.cache_tag
    assert lock.python_soabi == sysconfig.get_config_var("SOABI")
    assert lock.installed_distributions
    assert type(lock.installed_distributions[0]) is InstalledDistribution
    names = tuple(distribution.name for distribution in lock.installed_distributions)
    assert names == tuple(sorted(set(names)))


def test_no_pickle_or_arbitrary_python_deserialization_surface() -> None:
    source = Path(inspect.getfile(DefenderBundlePublisher)).read_text(encoding="utf-8")
    forbidden = ("import pickle", "from pickle", "joblib", "cloudpickle", "dill")
    assert all(token not in source for token in forbidden)
    assert "application/vnd.apar.catboost-model" in source
    assert "application/vnd.apache.parquet" in source
    assert canonical_json_bytes({"safe": True}) == b'{"safe":true}'
