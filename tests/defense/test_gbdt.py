"""Closed-contract and real CatBoost tests for the synthetic defense baseline."""

from __future__ import annotations

import inspect
import json
import math
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

import numpy as np
import pytest
from catboost import CatBoostClassifier, Pool
from pydantic import ValidationError

from apar.contracts.events import EventKind, Rail
from apar.defense.contracts import ObservedEvent
from apar.defense.gbdt import (
    CatBoostScorer,
    FoldResult,
    GbdtTrainingConfig,
    HyperParameters,
    ModelContractError,
    RollingFold,
    TrainingReceipt,
    _classifier,
    _digest_bytes,
    _load_native_model,
    _parameter_grid,
    _save_native_model,
    _selection_key,
    train_gbdt,
)
from apar.features.builders import FeatureMatrix
from apar.features.catalog import EXPECTED_FEATURE_NAMES, FeatureCatalog, load_feature_catalog
from apar.features.state import FeatureVector, feature_catalog_digest

T0 = datetime(2026, 1, 1, tzinfo=UTC)
CATALOG_PATH = Path(__file__).resolve().parents[2] / "config" / "defense" / "feature-catalog.json"


def _matrix(count: int = 24) -> FeatureMatrix:
    catalog = load_feature_catalog(CATALOG_PATH)
    digest = feature_catalog_digest(catalog)
    events: list[ObservedEvent] = []
    rows: list[FeatureVector] = []
    for index in range(count):
        event_id = f"row-{index:03d}"
        decision_at = T0 + timedelta(hours=index)
        events.append(
            ObservedEvent(
                event_id=event_id,
                payment_id=f"payment-{index:03d}",
                rail=Rail.CARD,
                event_type=EventKind.AUTHORIZATION,
                amount=Decimal(index + 1),
                currency="USD",
                event_time=decision_at,
                available_at=decision_at,
                decision_at=decision_at,
                actor_id=f"actor-{index % 5}",
                counterparty_id=f"counterparty-{index % 7}",
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
                catalog_digest=digest,
                values=values,
            )
        )
    return FeatureMatrix(
        events=tuple(events), catalog=catalog, catalog_digest=digest, rows=tuple(rows)
    )


def _labels() -> dict[str, int]:
    return {f"row-{index:03d}": int(index % 3 == 0) for index in range(20)}


def _folds() -> tuple[RollingFold, ...]:
    return (
        RollingFold(
            name="fold-1",
            fit_ids=tuple(f"row-{index:03d}" for index in range(8)),
            validation_ids=tuple(f"row-{index:03d}" for index in range(8, 12)),
        ),
        RollingFold(
            name="fold-2",
            fit_ids=tuple(f"row-{index:03d}" for index in range(12)),
            validation_ids=tuple(f"row-{index:03d}" for index in range(12, 16)),
        ),
    )


def _config(*, full_grid: bool = False) -> GbdtTrainingConfig:
    return GbdtTrainingConfig(
        depths=(4, 6) if full_grid else (2,),
        learning_rates=(0.03, 0.08) if full_grid else (0.1,),
        l2_leaf_regs=(3.0, 8.0) if full_grid else (3.0,),
        iterations=4 if full_grid else 8,
    )


def _train(*, config: GbdtTrainingConfig | None = None) -> CatBoostScorer:
    return train_gbdt(
        _matrix(),
        _labels(),
        tuple(f"row-{index:03d}" for index in range(20)),
        _folds(),
        _config() if config is None else config,
        training_cutoff=T0 + timedelta(hours=19),
    )


def _subset(matrix: FeatureMatrix, start: int, end: int) -> FeatureMatrix:
    row_ids = {f"row-{index:03d}" for index in range(start, end)}
    return matrix.model_copy(
        update={
            "events": tuple(event for event in matrix.events if event.event_id in row_ids),
            "rows": tuple(row for row in matrix.rows if row.event_id in row_ids),
        }
    )


def _semantic_catalog_mutation(matrix: FeatureMatrix) -> FeatureMatrix:
    mutated_feature = matrix.catalog.features[0].model_copy(update={"missing_behavior": "zero"})
    catalog = matrix.catalog.model_copy(
        update={"features": (mutated_feature, *matrix.catalog.features[1:])}
    )
    digest = feature_catalog_digest(catalog)
    return matrix.model_copy(
        update={
            "catalog": catalog,
            "catalog_digest": digest,
            "rows": tuple(row.model_copy(update={"catalog_digest": digest}) for row in matrix.rows),
        }
    )


def _forge_native_payload(
    scorer: CatBoostScorer, mutation: str, tmp_path: Path
) -> tuple[bytes, TrainingReceipt]:
    matrix = _matrix()
    train_ids = tuple(f"row-{index:03d}" for index in range(20))
    rows = {row.event_id: row for row in matrix.rows}
    data = np.asarray(
        [[rows[row_id].values[name] for name in EXPECTED_FEATURE_NAMES] for row_id in train_ids],
        dtype=np.float64,
    )
    labels = np.asarray([_labels()[row_id] for row_id in train_ids], dtype=np.int64)
    selected = scorer.receipt.selected_params
    settings: dict[str, object] = {
        "loss_function": "Logloss",
        "iterations": selected.iterations,
        "depth": selected.depth,
        "learning_rate": selected.learning_rate,
        "l2_leaf_reg": selected.l2_leaf_reg,
        "class_weights": list(scorer.receipt.class_weights),
        "random_seed": scorer.receipt.seed,
        "task_type": "CPU",
        "thread_count": 1,
        "allow_writing_files": False,
        "bootstrap_type": "No",
        "random_strength": 0,
        "verbose": False,
        "metadata": {"apar_training_contract_digest": scorer.receipt.training_contract_digest},
    }
    overrides: dict[str, dict[str, object]] = {
        "random_seed": {"random_seed": 999},
        "random_strength": {"random_strength": 1.0},
        "bootstrap_type": {"bootstrap_type": "Bernoulli"},
        "loss_function": {"loss_function": "MultiClass"},
        "thread_count": {"thread_count": 2},
        "allow_writing_files": {
            "allow_writing_files": True,
            "train_dir": str(tmp_path / "forged-catboost-info"),
        },
        "verbose": {"verbose": 1},
    }
    settings.update(overrides[mutation])
    model = CatBoostClassifier(**settings)
    model.fit(Pool(data=data, label=labels, feature_names=list(EXPECTED_FEATURE_NAMES)))
    payload = _save_native_model(model)
    importance = tuple(
        float(value)
        for value in np.asarray(
            model.get_feature_importance(type="PredictionValuesChange"), dtype=np.float64
        )
    )
    receipt = scorer.receipt.model_copy(
        update={
            "model_payload_digest": _digest_bytes(payload),
            "global_feature_importance": importance,
        }
    )
    return payload, receipt


def test_production_defaults_and_parameter_grid_are_exact() -> None:
    config = GbdtTrainingConfig()
    assert config.seed == 260816
    assert config.depths == (4, 6)
    assert config.learning_rates == (0.03, 0.08)
    assert config.l2_leaf_regs == (3.0, 8.0)
    assert config.iterations == 300
    assert config.fpr_probability_threshold == 0.5
    assert _parameter_grid(config) == tuple(
        HyperParameters(depth=depth, learning_rate=rate, l2_leaf_reg=l2, iterations=300)
        for depth in (4, 6)
        for rate in (0.03, 0.08)
        for l2 in (3.0, 8.0)
    )


@pytest.mark.parametrize("threshold", (0.0, 0.49, 0.5000001, 1.0))
def test_fpr_tie_break_threshold_is_immutably_frozen_at_half(threshold: float) -> None:
    with pytest.raises(ValidationError, match="FPR|fpr_probability_threshold"):
        GbdtTrainingConfig(fpr_probability_threshold=threshold)


def test_classifier_freezes_deterministic_cpu_settings() -> None:
    params = HyperParameters(depth=4, learning_rate=0.03, l2_leaf_reg=3.0, iterations=300)
    settings = _classifier(params, (0.75, 1.5), 260816).get_params()
    assert settings == {
        "loss_function": "Logloss",
        "iterations": 300,
        "depth": 4,
        "learning_rate": 0.03,
        "l2_leaf_reg": 3.0,
        "class_weights": [0.75, 1.5],
        "random_seed": 260816,
        "task_type": "CPU",
        "thread_count": 1,
        "allow_writing_files": False,
        "bootstrap_type": "No",
        "random_strength": 0,
        "verbose": False,
    }


def test_native_model_persists_every_declared_deterministic_setting() -> None:
    scorer = _train()
    model = _load_native_model(scorer.to_bytes())
    params = model.get_all_params()
    assert {
        "loss_function": params["loss_function"],
        "random_seed": params["random_seed"],
        "random_strength": params["random_strength"],
        "bootstrap_type": params["bootstrap_type"],
        "task_type": params["task_type"],
    } == {
        "loss_function": "Logloss",
        "random_seed": 260816,
        "random_strength": 0,
        "bootstrap_type": "No",
        "task_type": "CPU",
    }
    native_params = json.loads(model.get_metadata()["params"])
    flat = native_params["flat_params"]
    assert flat["thread_count"] == 1
    assert flat["allow_writing_files"] is False
    assert flat["verbose"] == 0
    assert model.get_metadata()["model_guid"] == scorer.receipt.training_contract_digest
    assert model.get_metadata()["train_finish_time"] == "2026-01-01T19:00:00Z"


@pytest.mark.parametrize(
    "mutation",
    (
        "random_seed",
        "random_strength",
        "bootstrap_type",
        "loss_function",
        "thread_count",
        "allow_writing_files",
        "verbose",
    ),
)
def test_loader_rejects_forged_native_deterministic_settings(mutation: str, tmp_path: Path) -> None:
    scorer = _train()
    payload, forged_receipt = _forge_native_payload(scorer, mutation, tmp_path)
    with pytest.raises(ModelContractError, match="native model|deterministic|setting"):
        CatBoostScorer.from_bytes(payload, forged_receipt)


@pytest.mark.parametrize(
    "update",
    (
        {"seed": -1},
        {"depths": ()},
        {"depths": (4, 4)},
        {"depths": (0,)},
        {"learning_rates": (0.0,)},
        {"learning_rates": (math.inf,)},
        {"l2_leaf_regs": (-1.0,)},
        {"iterations": 0},
        {"fpr_probability_threshold": 1.1},
    ),
)
def test_invalid_training_config_fails_closed(update: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        GbdtTrainingConfig(**update)


def test_full_declared_eight_candidate_search_is_executed() -> None:
    scorer = _train(config=_config(full_grid=True))
    assert len(scorer.receipt.fold_results) == 8 * len(_folds())
    assert {result.params for result in scorer.receipt.fold_results} == set(
        _parameter_grid(_config(full_grid=True))
    )


def test_seeded_cpu_training_and_native_reload_reproduce_scores() -> None:
    matrix = _subset(_matrix(), 20, 24)
    first = _train()
    second = _train()
    np.testing.assert_allclose(first.predict(matrix), second.predict(matrix), rtol=0.0, atol=1e-12)
    assert first.receipt.fold_results == second.receipt.fold_results
    assert first.to_bytes() == second.to_bytes()
    assert first.receipt == second.receipt

    payload = first.to_bytes()
    assert payload.startswith(b"CBM1")
    assert not payload.startswith(b"\x80\x04")
    restored = CatBoostScorer.from_bytes(payload, first.receipt)
    np.testing.assert_array_equal(first.predict(matrix), restored.predict(matrix))
    np.testing.assert_array_equal(first.predict_raw(matrix), restored.predict_raw(matrix))


def test_scores_and_shap_contributions_are_finite_and_reconstruct_logits() -> None:
    scorer = _train()
    matrix = _subset(_matrix(), 20, 24)
    scores = scorer.predict(matrix)
    raw = scorer.predict_raw(matrix)
    contributions = scorer.contributions(matrix)
    assert scores.shape == raw.shape == (4,)
    assert np.isfinite(scores).all() and np.isfinite(raw).all()
    assert ((scores >= 0.0) & (scores <= 1.0)).all()
    assert contributions.shape == (4, len(EXPECTED_FEATURE_NAMES) + 1)
    assert np.isfinite(contributions).all()
    np.testing.assert_allclose(
        contributions[:, :-1].sum(axis=1) + contributions[:, -1],
        raw,
        rtol=0.0,
        atol=1e-12,
    )
    importance = scorer.global_feature_importance()
    assert tuple(importance) == EXPECTED_FEATURE_NAMES
    assert all(math.isfinite(value) and value >= 0.0 for value in importance.values())


def test_native_payload_receipt_and_environment_tampering_is_rejected() -> None:
    scorer = _train()
    payload = scorer.to_bytes()
    changed = bytearray(payload)
    changed[-1] ^= 1
    with pytest.raises(ModelContractError, match="digest"):
        CatBoostScorer.from_bytes(bytes(changed), scorer.receipt)
    incompatible = scorer.receipt.model_copy(update={"catboost_version": "9.0.0"})
    with pytest.raises(ModelContractError, match="receipt|CatBoost version"):
        CatBoostScorer.from_bytes(payload, incompatible)
    wrong_catalog = scorer.receipt.model_copy(update={"catalog_digest": "0" * 64})
    with pytest.raises(ModelContractError, match="receipt|catalog"):
        CatBoostScorer.from_bytes(payload, wrong_catalog)
    wrong_params = scorer.receipt.model_copy(
        update={"selected_params": scorer.receipt.selected_params.model_copy(update={"depth": 6})}
    )
    with pytest.raises(ModelContractError, match="receipt|depth|training contract"):
        CatBoostScorer.from_bytes(payload, wrong_params)
    wrong_cutoff = scorer.receipt.model_copy(update={"training_cutoff": datetime(2026, 1, 1)})
    with pytest.raises(ModelContractError, match="receipt"):
        CatBoostScorer.from_bytes(payload, wrong_cutoff)
    changed_importance = list(scorer.receipt.global_feature_importance)
    changed_importance[0] += 1.0
    wrong_importance = scorer.receipt.model_copy(
        update={"global_feature_importance": tuple(changed_importance)}
    )
    with pytest.raises(ModelContractError, match="importance"):
        CatBoostScorer.from_bytes(payload, wrong_importance)


def test_scorer_is_immutable_after_publication() -> None:
    scorer = _train()
    with pytest.raises(AttributeError):
        scorer.receipt = scorer.receipt.model_copy(update={"seed": 0})


def test_training_records_environment_cutoff_weights_digests_and_exclusions() -> None:
    matrix = _matrix()
    train_ids = tuple(f"row-{index:03d}" for index in range(20))
    folds = (
        RollingFold(
            name="fold-1",
            fit_ids=tuple(f"row-{index:03d}" for index in range(7)),
            validation_ids=tuple(f"row-{index:03d}" for index in range(8, 12)),
        ),
        RollingFold(
            name="fold-2",
            fit_ids=tuple(f"row-{index:03d}" for index in range(1, 12)),
            validation_ids=tuple(f"row-{index:03d}" for index in range(12, 16)),
        ),
    )
    scorer = train_gbdt(
        matrix,
        _labels(),
        train_ids,
        folds,
        _config(),
        training_cutoff=T0 + timedelta(hours=19),
        mandatory_row_ids=("row-000",),
    )
    receipt = scorer.receipt
    assert receipt.requested_training_count == 20
    assert receipt.mandatory_excluded_count == 1
    assert receipt.final_training_count == 19
    assert receipt.final_training_row_ids_digest != receipt.requested_training_row_ids_digest
    assert receipt.training_cutoff == T0 + timedelta(hours=19)
    assert receipt.feature_order == EXPECTED_FEATURE_NAMES
    assert receipt.class_weights[0] > 0.0 and receipt.class_weights[1] > 0.0
    assert receipt.python_version and receipt.platform and receipt.catboost_version
    assert receipt.scikit_learn_version
    encoded = receipt.model_dump_json().lower()
    assert all(token not in encoded for token in ("campaign", "family", "scenario", "hidden"))


def test_mandatory_ids_must_be_requested_and_are_filtered_from_every_fit() -> None:
    with pytest.raises(ModelContractError, match="mandatory.*subset"):
        train_gbdt(
            _matrix(),
            _labels(),
            tuple(f"row-{index:03d}" for index in range(20)),
            _folds(),
            _config(),
            training_cutoff=T0 + timedelta(hours=19),
            mandatory_row_ids=("row-999",),
        )
    scorer = _train()
    assert scorer.receipt.mandatory_excluded_count == 0


def test_per_fold_class_weights_ignore_labels_outside_that_fit_set() -> None:
    baseline = _train()
    changed_labels = _labels()
    changed_labels["row-019"] = 1 - changed_labels["row-019"]
    changed = train_gbdt(
        _matrix(),
        changed_labels,
        tuple(f"row-{index:03d}" for index in range(20)),
        _folds(),
        _config(),
        training_cutoff=T0 + timedelta(hours=19),
    )
    baseline_fold = next(
        result for result in baseline.receipt.fold_results if result.fold_name == "fold-1"
    )
    changed_fold = next(
        result for result in changed.receipt.fold_results if result.fold_name == "fold-1"
    )
    assert baseline_fold.class_weights == changed_fold.class_weights
    assert baseline_fold.fit_ids_digest == changed_fold.fit_ids_digest
    assert baseline.receipt.class_weights != changed.receipt.class_weights


def test_mandatory_row_label_cannot_affect_folds_final_model_or_scores() -> None:
    labels = _labels()
    folds = (
        RollingFold(
            name="fold-1",
            fit_ids=tuple(f"row-{index:03d}" for index in range(8)),
            validation_ids=tuple(f"row-{index:03d}" for index in range(8, 12)),
        ),
        RollingFold(
            name="fold-2",
            fit_ids=tuple(f"row-{index:03d}" for index in range(12)),
            validation_ids=tuple(f"row-{index:03d}" for index in range(12, 16)),
        ),
    )
    kwargs = {
        "matrix": _matrix(),
        "train_ids": tuple(f"row-{index:03d}" for index in range(20)),
        "folds": folds,
        "config": _config(),
        "training_cutoff": T0 + timedelta(hours=19),
        "mandatory_row_ids": ("row-000",),
    }
    baseline = train_gbdt(labels=labels, **kwargs)
    labels["row-000"] = 1 - labels["row-000"]
    changed = train_gbdt(labels=labels, **kwargs)
    assert baseline.receipt.class_weights == changed.receipt.class_weights
    assert baseline.receipt.fold_results == changed.receipt.fold_results
    np.testing.assert_allclose(
        baseline.predict(_subset(_matrix(), 20, 24)),
        changed.predict(_subset(_matrix(), 20, 24)),
        rtol=0.0,
        atol=1e-12,
    )


def test_selection_ties_use_mean_ap_then_lower_fpr_then_lexicographic_params() -> None:
    small = HyperParameters(depth=4, learning_rate=0.03, l2_leaf_reg=3.0, iterations=300)
    large = HyperParameters(depth=6, learning_rate=0.08, l2_leaf_reg=8.0, iterations=300)
    assert _selection_key(
        small, mean_average_precision=0.8, mean_legitimate_fpr=0.2
    ) < _selection_key(large, mean_average_precision=0.7, mean_legitimate_fpr=0.0)
    assert _selection_key(
        small, mean_average_precision=0.8, mean_legitimate_fpr=0.1
    ) < _selection_key(large, mean_average_precision=0.8, mean_legitimate_fpr=0.2)
    assert _selection_key(
        small, mean_average_precision=0.8, mean_legitimate_fpr=0.1
    ) < _selection_key(large, mean_average_precision=0.8, mean_legitimate_fpr=0.1)


def test_fold_result_rejects_undefined_or_nonfinite_metrics() -> None:
    params = HyperParameters(depth=4, learning_rate=0.03, l2_leaf_reg=3.0, iterations=300)
    with pytest.raises(ValidationError):
        FoldResult(
            fold_name="f",
            params=params,
            average_precision=math.nan,
            legitimate_fpr=0.0,
            fit_count=2,
            validation_count=2,
            class_weights=(1.0, 1.0),
            fit_ids_digest="0" * 64,
            validation_ids_digest="1" * 64,
        )


def test_training_normalizes_machine_precision_average_precision_roundoff(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep an exact perfect PR-AUC representable inside its closed unit interval."""
    import apar.defense.gbdt as gbdt_module

    monkeypatch.setattr(
        gbdt_module,
        "average_precision_score",
        lambda *_args, **_kwargs: 1.0000000000000002,
    )

    scorer = _train()

    assert {row.average_precision for row in scorer.receipt.fold_results} == {1.0}


@pytest.mark.parametrize("mutation", ("missing", "extra", "reordered", "duplicate", "nonfinite"))
def test_training_rejects_any_feature_matrix_contract_mutation(mutation: str) -> None:
    matrix = _matrix()
    row = matrix.rows[0]
    values = dict(row.values)
    catalog: FeatureCatalog = matrix.catalog
    if mutation == "missing":
        values.pop(EXPECTED_FEATURE_NAMES[-1])
    elif mutation == "extra":
        values["campaign_alias"] = 1.0
    elif mutation == "reordered":
        values = dict(reversed(tuple(values.items())))
    elif mutation == "duplicate":
        catalog = catalog.model_copy(
            update={"features": (*catalog.features[:-1], catalog.features[0])}
        )
    else:
        values[EXPECTED_FEATURE_NAMES[0]] = math.inf
    changed = matrix.model_copy(
        update={
            "catalog": catalog,
            "rows": (row.model_copy(update={"values": values}), *matrix.rows[1:]),
        }
    )
    with pytest.raises(ModelContractError, match="feature|finite|catalog"):
        train_gbdt(
            changed,
            _labels(),
            tuple(f"row-{index:03d}" for index in range(20)),
            _folds(),
            _config(),
            training_cutoff=T0 + timedelta(hours=19),
        )


def test_training_rejects_audit_valid_semantic_catalog_substitution() -> None:
    changed = _semantic_catalog_mutation(_matrix())
    assert changed.catalog.names == EXPECTED_FEATURE_NAMES
    assert changed.catalog_digest == feature_catalog_digest(changed.catalog)
    with pytest.raises(ModelContractError, match="frozen competition catalog"):
        train_gbdt(
            changed,
            _labels(),
            tuple(f"row-{index:03d}" for index in range(20)),
            _folds(),
            _config(),
            training_cutoff=T0 + timedelta(hours=19),
        )


def test_scoring_rejects_audit_valid_semantic_catalog_substitution() -> None:
    scorer = _train()
    changed = _semantic_catalog_mutation(_subset(_matrix(), 20, 24))
    forged_receipt = scorer.receipt.model_copy(update={"catalog_digest": changed.catalog_digest})
    forged_scorer = CatBoostScorer(
        _load_native_model(scorer.to_bytes()), scorer.to_bytes(), forged_receipt
    )
    with pytest.raises(ModelContractError, match="frozen competition catalog"):
        forged_scorer.predict(changed)


def test_scorer_rejects_a_scoring_matrix_catalog_or_row_mutation() -> None:
    scorer = _train()
    matrix = _subset(_matrix(), 20, 24)
    reordered = matrix.rows[0].model_copy(
        update={"values": dict(reversed(tuple(matrix.rows[0].values.items())))}
    )
    with pytest.raises(ModelContractError, match="feature order"):
        scorer.predict(matrix.model_copy(update={"rows": (reordered, *matrix.rows[1:])}))
    with pytest.raises(ModelContractError, match="catalog"):
        scorer.predict(matrix.model_copy(update={"catalog_digest": "0" * 64}))


def test_duplicate_or_mismatched_matrix_row_identity_is_rejected() -> None:
    matrix = _matrix()
    duplicate = matrix.model_copy(update={"rows": (*matrix.rows[:-1], matrix.rows[0])})
    mismatch = matrix.model_copy(update={"events": matrix.events[:-1]})
    kwargs = dict(
        labels=_labels(),
        train_ids=tuple(f"row-{index:03d}" for index in range(20)),
        folds=_folds(),
        config=_config(),
        training_cutoff=T0 + timedelta(hours=19),
    )
    with pytest.raises(ModelContractError, match="duplicate"):
        train_gbdt(duplicate, **kwargs)
    with pytest.raises(ModelContractError, match="row IDs"):
        train_gbdt(mismatch, **kwargs)


@pytest.mark.parametrize(
    ("labels", "match"),
    (
        ({**_labels(), "row-999": 0}, "label IDs"),
        ({key: value for key, value in _labels().items() if key != "row-019"}, "label IDs"),
        ({**_labels(), "row-000": 2}, "binary"),
        ({**_labels(), "row-000": 0.0}, "binary"),
    ),
)
def test_training_rejects_misaligned_or_nonbinary_labels(
    labels: dict[str, object], match: str
) -> None:
    with pytest.raises(ModelContractError, match=match):
        train_gbdt(
            _matrix(),
            labels,
            tuple(f"row-{index:03d}" for index in range(20)),
            _folds(),
            _config(),
            training_cutoff=T0 + timedelta(hours=19),
        )


def test_train_ids_must_be_unique_known_chronological_and_before_cutoff() -> None:
    normal = tuple(f"row-{index:03d}" for index in range(20))
    for ids, cutoff, match in (
        ((*normal[:-1], normal[0]), T0 + timedelta(hours=19), "duplicate"),
        ((*normal[:-1], "row-999"), T0 + timedelta(hours=19), "matrix"),
        ((normal[1], normal[0], *normal[2:]), T0 + timedelta(hours=19), "chronological"),
        (normal, T0 + timedelta(hours=18), "cutoff"),
    ):
        with pytest.raises(ModelContractError, match=match):
            train_gbdt(_matrix(), _labels(), ids, _folds(), _config(), training_cutoff=cutoff)


def test_folds_reject_overlap_unknown_ids_bad_order_and_degenerate_classes() -> None:
    train_ids = tuple(f"row-{index:03d}" for index in range(20))
    with pytest.raises(ValidationError):
        RollingFold(name="overlap", fit_ids=("row-000",), validation_ids=("row-000",))
    bad_folds = (
        RollingFold(
            name="late-fit", fit_ids=("row-008", "row-009"), validation_ids=("row-001", "row-003")
        ),
    )
    with pytest.raises(ModelContractError, match="earlier"):
        train_gbdt(
            _matrix(),
            _labels(),
            train_ids,
            bad_folds,
            _config(),
            training_cutoff=T0 + timedelta(hours=19),
        )
    unknown = (
        RollingFold(
            name="unknown", fit_ids=("row-000", "row-001"), validation_ids=("row-998", "row-999")
        ),
    )
    with pytest.raises(ModelContractError, match="training IDs"):
        train_gbdt(
            _matrix(),
            _labels(),
            train_ids,
            unknown,
            _config(),
            training_cutoff=T0 + timedelta(hours=19),
        )
    all_legitimate = (
        RollingFold(
            name="one-class", fit_ids=("row-001", "row-002"), validation_ids=("row-004", "row-005")
        ),
    )
    with pytest.raises(ModelContractError, match="both classes"):
        train_gbdt(
            _matrix(),
            _labels(),
            train_ids,
            all_legitimate,
            _config(),
            training_cutoff=T0 + timedelta(hours=19),
        )
    duplicated_validation = (
        _folds()[0],
        RollingFold(
            name="again",
            fit_ids=tuple(f"row-{i:03d}" for i in range(8)),
            validation_ids=("row-008", "row-010"),
        ),
    )
    with pytest.raises(ModelContractError, match="validation.*overlap"):
        train_gbdt(
            _matrix(),
            _labels(),
            train_ids,
            duplicated_validation,
            _config(),
            training_cutoff=T0 + timedelta(hours=19),
        )


def test_training_boundary_has_no_evaluator_grouping_arguments_or_private_receipt_fields() -> None:
    parameters = inspect.signature(train_gbdt).parameters
    assert (
        "mandatory_row_ids" in parameters
        and parameters["mandatory_row_ids"].kind is inspect.Parameter.KEYWORD_ONLY
    )
    assert not ({"truth", "family", "campaign", "scenario", "regime", "hidden"} & set(parameters))


def test_training_never_creates_catboost_info(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    _train()
    assert not (tmp_path / "catboost_info").exists()
