"""Comparison arms regression tests."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

import apar.evaluation.v5_evaluation as v5_evaluation
from apar.defense.sentinel import SentinelAction
from apar.evaluation.v5_evaluation import (
    V5Arm,
    V5ArmScore,
    V5ArmSpecification,
    build_v5_arm_support_rows,
    build_v5_execution_artifacts,
    build_v5_training_partition_evidence,
)
from apar.evaluation.v5_population import V5DecisionRow, build_v5_corpus
from apar.evaluation.v5_protocol import V5Profile, load_v5_development_protocol
from apar.features.sentinel import SentinelFeatureCatalog
from tests.evaluation.v5_safe_protocol import load_safe_v5_test_protocol

ROOT = Path(__file__).resolve().parents[2]


class TestComparisonArms:
    def test_all_four_arm_values_exist(self) -> None:
        expected = {"rules_only", "ensemble_no_graph", "ensemble_with_graph", "full_sentinel"}
        actual = {arm.value for arm in V5Arm}
        assert expected <= actual, f"missing arms: {expected - actual}"

    def test_hardened_sentinel_not_in_current_round(self) -> None:
        """hardened_sentinel is a future arm; it must not be evaluated this round."""
        assert V5Arm.HARDENED_SENTINEL.value == "hardened_sentinel"


def test_arm_configuration_freezes_real_component_semantics() -> None:
    """Removing the declarative loader would permit enum-only comparison arms."""
    loader = getattr(v5_evaluation, "load_v5_arm_configuration", None)
    assert callable(loader), "v5 arm configuration loader is missing"

    protocol = load_v5_development_protocol(
        ROOT / "config/defense/defense-v5-development.json"
    )
    catalog = SentinelFeatureCatalog.default()
    configuration = loader(
        ROOT / "config/defense/defense-v5-arms.json",
        catalog=catalog,
        protocol=protocol,
    )

    by_arm = {template.arm: template for template in configuration.arms}
    assert set(by_arm) == {
        V5Arm.RULES_ONLY,
        V5Arm.ENSEMBLE_NO_GRAPH,
        V5Arm.ENSEMBLE_WITH_GRAPH,
        V5Arm.FULL_SENTINEL,
    }
    assert (by_arm[V5Arm.RULES_ONLY].rules, by_arm[V5Arm.RULES_ONLY].trust) == (
        True,
        True,
    )
    assert by_arm[V5Arm.RULES_ONLY].model is False
    assert by_arm[V5Arm.ENSEMBLE_NO_GRAPH].graph is False
    assert by_arm[V5Arm.ENSEMBLE_NO_GRAPH].novelty is False
    assert by_arm[V5Arm.ENSEMBLE_WITH_GRAPH].graph is True
    assert by_arm[V5Arm.ENSEMBLE_WITH_GRAPH].novelty is False
    assert (
        by_arm[V5Arm.ENSEMBLE_NO_GRAPH].feature_names
        == by_arm[V5Arm.ENSEMBLE_WITH_GRAPH].non_graph_feature_names
    )
    assert by_arm[V5Arm.FULL_SENTINEL].rules is True
    assert by_arm[V5Arm.FULL_SENTINEL].trust is True
    assert by_arm[V5Arm.FULL_SENTINEL].novelty is True
    assert by_arm[V5Arm.FULL_SENTINEL].disagreement is True
    assert all(template.catalog_sha256 == catalog.catalog_sha256 for template in by_arm.values())
    assert all(len(template.spec_sha256) == 64 for template in by_arm.values())


def test_four_arms_train_and_score_independent_component_paths() -> None:
    """A full-only scorer or cloned result dictionaries cannot satisfy this contract."""
    from apar.evaluation import v5_arms

    train = getattr(v5_arms, "train_v5_arm_set", None)
    score = getattr(v5_arms, "score_v5_arm_set", None)
    support_type = getattr(v5_evaluation, "V5ArmSupportRow", None)
    assert callable(train), "real comparison-arm trainer is missing"
    assert callable(score), "real comparison-arm scorer is missing"
    assert support_type is not None, "ordered arm support contract is missing"

    protocol = load_safe_v5_test_protocol(ROOT)
    corpus = build_v5_corpus(protocol, profile=V5Profile.SMOKE)
    catalog = SentinelFeatureCatalog.default()
    configuration = v5_evaluation.load_v5_arm_configuration(
        ROOT / "config/defense/defense-v5-arms.json",
        catalog=catalog,
        protocol=protocol,
    )
    graph_index = next(
        index for index, group in enumerate(catalog.feature_groups) if group == "graph"
    )
    integrity_index = catalog.feature_groups.index("integrity")
    rng = np.random.RandomState(817)

    def partition(
        name: str,
        size: int,
        *,
        trust_failure_first: bool = False,
    ) -> tuple[np.ndarray, np.ndarray, tuple[V5DecisionRow, ...]]:
        available = corpus.partitions[name].decisions
        selected: list[V5DecisionRow] = []
        if trust_failure_first:
            selected.append(
                next(
                    row
                    for row in available
                    if row.rail == "agentic" and row.integrity_status == "fail"
                )
            )
        benign = [row for row in available if not row.is_fraud and row not in selected]
        fraud = [row for row in available if row.is_fraud and row not in selected]
        for benign_row, fraud_row in zip(benign, fraud, strict=False):
            if len(selected) < size:
                selected.append(benign_row)
            if len(selected) < size:
                selected.append(fraud_row)
            if len(selected) == size:
                break
        if len(selected) != size:
            raise AssertionError(f"{name} lacks balanced test support")
        labels = np.array([int(row.is_fraud) for row in selected], dtype=int)
        matrix = rng.normal(0.0, 0.05, (size, len(catalog.feature_names)))
        matrix[:, graph_index] = labels * 8.0
        matrix[:, integrity_index] = labels * 9.0
        return matrix, labels, tuple(selected)

    x_train, y_train, train_rows = partition("train", 32)
    x_cal, y_cal, calibration_rows = partition("calibration", 16)
    x_threshold, y_threshold, threshold_rows_source = partition("threshold", 16)
    x_test, y_test, test_rows = partition(
        "development_test", 8, trust_failure_first=True
    )
    x_test[1, catalog.feature_names.index("actor_count_1m")] = 5.0

    def independent_digest(document: object) -> str:
        return hashlib.sha256(
            json.dumps(
                document,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            ).encode()
        ).hexdigest()

    def provenance(
        name: str,
        matrix: np.ndarray,
        labels: np.ndarray,
        rows: tuple[V5DecisionRow, ...],
    ):
        event_ids = tuple(row.event_id for row in rows)
        evidence_support = build_v5_arm_support_rows(rows)
        return build_v5_training_partition_evidence(
            partition=name,
            event_ids=event_ids,
            labels=labels,
            support=evidence_support,
            feature_batch_sha256=hashlib.sha256(
                json.dumps(
                    {"rows": matrix.tolist(), "names": list(catalog.feature_names)},
                    sort_keys=True,
                ).encode()
            ).hexdigest(),
            feature_matrix=matrix,
            feature_names=catalog.feature_names,
            catalog_sha256=catalog.catalog_sha256,
            execution_manifests=corpus.partitions[name].executions,
        )

    train_provenance = provenance("train", x_train, y_train, train_rows)
    calibration_provenance = provenance(
        "calibration", x_cal, y_cal, calibration_rows
    )
    threshold_provenance = provenance(
        "threshold", x_threshold, y_threshold, threshold_rows_source
    )
    tampered_catalog = catalog.model_copy(update={"catalog_sha256": "0" * 64})
    with pytest.raises(ValueError, match="catalog digest"):
        train(
            configuration=configuration,
            catalog=tampered_catalog,
            x_train=x_train,
            y_train=y_train,
            x_calibration=x_cal,
            y_calibration=y_cal,
            x_threshold=x_threshold,
            y_threshold=y_threshold,
            bootstrap_seed=protocol.seeds.bootstrap,
            train_evidence=train_provenance,
            calibration_evidence=calibration_provenance,
            threshold_evidence=threshold_provenance,
        )
    with pytest.raises(ValueError, match="swapped"):
        train(
            configuration=configuration,
            catalog=catalog,
            x_train=x_train,
            y_train=y_train,
            x_calibration=x_cal,
            y_calibration=y_cal,
            x_threshold=x_threshold,
            y_threshold=y_threshold,
            bootstrap_seed=protocol.seeds.bootstrap,
            train_evidence=calibration_provenance,
            calibration_evidence=train_provenance,
            threshold_evidence=threshold_provenance,
        )
    with pytest.raises(ValueError, match="bootstrap seed"):
        train(
            configuration=configuration,
            catalog=catalog,
            x_train=x_train,
            y_train=y_train,
            x_calibration=x_cal,
            y_calibration=y_cal,
            x_threshold=x_threshold,
            y_threshold=y_threshold,
            bootstrap_seed=protocol.seeds.bootstrap + 1,
            train_evidence=train_provenance,
            calibration_evidence=calibration_provenance,
            threshold_evidence=threshold_provenance,
        )
    trained = train(
        configuration=configuration,
        catalog=catalog,
        x_train=x_train,
        y_train=y_train,
        x_calibration=x_cal,
        y_calibration=y_cal,
        x_threshold=x_threshold,
        y_threshold=y_threshold,
        bootstrap_seed=protocol.seeds.bootstrap,
        train_evidence=train_provenance,
        calibration_evidence=calibration_provenance,
        threshold_evidence=threshold_provenance,
    )
    feature_semantics_forge = deepcopy(
        trained.by_arm[V5Arm.FULL_SENTINEL].spec.model_dump(mode="json")
    )
    forged_training = feature_semantics_forge["training_partitions"][0]
    forged_training["feature_names"][0] = "is_fraud"
    feature_payload = json.loads(forged_training["feature_batch_payload_json"])
    feature_payload["names"][0] = "is_fraud"
    forged_training["feature_batch_payload_json"] = json.dumps(
        feature_payload, sort_keys=True
    )
    forged_training["feature_batch_sha256"] = hashlib.sha256(
        forged_training["feature_batch_payload_json"].encode()
    ).hexdigest()
    feature_semantics_forge["spec_sha256"] = independent_digest(
        {
            key: value
            for key, value in feature_semantics_forge.items()
            if key != "spec_sha256"
        }
    )
    with pytest.raises(ValueError, match="full catalog feature names|feature semantics"):
        V5ArmSpecification.model_validate(feature_semantics_forge)

    support = build_v5_arm_support_rows(test_rows)
    execution_artifacts = build_v5_execution_artifacts(
        corpus.partitions["development_test"].executions
    )
    trust_failures = [row.integrity_status == "fail" for row in test_rows]
    with pytest.raises(ValueError, match="evaluation.*overlap|training partition"):
        score(
            trained=trained,
            catalog=catalog,
            features_matrix=x_train,
            support=build_v5_arm_support_rows(train_rows),
            execution_artifacts=build_v5_execution_artifacts(
                corpus.partitions["train"].executions
            ),
            trust_failures=[row.integrity_status == "fail" for row in train_rows],
        )
    scored = score(
        trained=trained,
        catalog=catalog,
        features_matrix=x_test,
        support=support,
        execution_artifacts=execution_artifacts,
        trust_failures=trust_failures,
    )

    assert tuple(scored.by_arm) == (
        V5Arm.RULES_ONLY,
        V5Arm.ENSEMBLE_NO_GRAPH,
        V5Arm.ENSEMBLE_WITH_GRAPH,
        V5Arm.FULL_SENTINEL,
    )
    assert len({result.support_sha256 for result in scored.by_arm.values()}) == 1
    assert all(
        tuple(row.support.event_id for row in result.rows)
        == tuple(item.event_id for item in support)
        for result in scored.by_arm.values()
    )
    assert trained.by_arm[V5Arm.RULES_ONLY].defender is None
    assert len(trained.by_arm[V5Arm.RULES_ONLY].spec.feature_names) == 6
    assert len(trained.by_arm[V5Arm.ENSEMBLE_NO_GRAPH].spec.feature_names) == 35
    assert len(trained.by_arm[V5Arm.ENSEMBLE_WITH_GRAPH].spec.feature_names) == 46
    assert len(trained.by_arm[V5Arm.FULL_SENTINEL].spec.feature_names) == 46
    assert all(arm.spec.execution_bound for arm in trained.arms)
    assert all(
        tuple(item.partition for item in arm.spec.training_partitions)
        == ("train", "calibration", "threshold")
        for arm in trained.arms
    )
    for arm in trained.arms:
        threshold_rows = arm.spec.training_partitions[2]
        expected_threshold_digest = independent_digest(
            {
                "source_partition": arm.spec.threshold_source_partition,
                "method": arm.spec.threshold_method,
                "threshold_ordered_rows_sha256": threshold_rows.ordered_rows_sha256,
                "threshold_support_sha256": threshold_rows.ordered_support_sha256,
                "threshold_feature_batch_sha256": threshold_rows.feature_batch_sha256,
                "threshold_feature_matrix_sha256": threshold_rows.feature_matrix_sha256,
                "threshold_values": arm.spec.threshold_values,
            }
        )
        assert arm.spec.threshold_digest == expected_threshold_digest
        if arm.spec.model:
            assert len(arm.spec.model_artifact_sha256) == len(arm.spec.model_seeds)
            assert len(arm.spec.calibrator_artifact_sha256) == len(arm.spec.model_seeds)
        else:
            assert arm.spec.model_artifact_sha256 == ()
            assert arm.spec.calibrator_artifact_sha256 == ()
    assert trained.by_arm[V5Arm.FULL_SENTINEL].spec.novelty_artifact_sha256
    assert all(
        arm.spec.novelty_artifact_sha256 is None
        for arm in trained.arms
        if arm.spec.arm is not V5Arm.FULL_SENTINEL
    )
    rule_spec_tamper = deepcopy(
        trained.by_arm[V5Arm.RULES_ONLY].spec.model_dump(mode="json")
    )
    rule_spec_tamper["component_parameters"][0][1] += 1.0
    rule_spec_tamper["spec_sha256"] = independent_digest(
        {
            key: value
            for key, value in rule_spec_tamper.items()
            if key != "spec_sha256"
        }
    )
    with pytest.raises(ValueError, match="rule component parameters"):
        V5ArmSpecification.model_validate(rule_spec_tamper)

    rule_threshold_tamper = deepcopy(
        trained.by_arm[V5Arm.RULES_ONLY].spec.model_dump(mode="json")
    )
    threshold_map = dict(rule_threshold_tamper["threshold_values"])
    threshold_map["rules_challenge"] = 0.1
    rule_threshold_tamper["threshold_values"] = sorted(threshold_map.items())
    threshold_rows = rule_threshold_tamper["training_partitions"][2]
    rule_threshold_tamper["threshold_digest"] = independent_digest(
        {
            "source_partition": rule_threshold_tamper["threshold_source_partition"],
            "method": rule_threshold_tamper["threshold_method"],
            "threshold_ordered_rows_sha256": threshold_rows["ordered_rows_sha256"],
            "threshold_support_sha256": threshold_rows["ordered_support_sha256"],
            "threshold_feature_batch_sha256": threshold_rows["feature_batch_sha256"],
            "threshold_feature_matrix_sha256": threshold_rows["feature_matrix_sha256"],
            "threshold_values": rule_threshold_tamper["threshold_values"],
        }
    )
    rule_threshold_tamper["spec_sha256"] = independent_digest(
        {
            key: value
            for key, value in rule_threshold_tamper.items()
            if key != "spec_sha256"
        }
    )
    with pytest.raises(ValueError, match="rule threshold"):
        V5ArmSpecification.model_validate(rule_threshold_tamper)
    assert scored.by_arm[V5Arm.RULES_ONLY].rows[0].trust_routed is True
    assert scored.by_arm[V5Arm.FULL_SENTINEL].rows[0].trust_routed is True
    assert scored.by_arm[V5Arm.ENSEMBLE_NO_GRAPH].rows[0].trust_routed is False
    assert scored.by_arm[V5Arm.ENSEMBLE_WITH_GRAPH].rows[0].trust_routed is False
    assert any(
        row.rule_score is not None and row.rule_score > 0.0
        for row in scored.by_arm[V5Arm.RULES_ONLY].rows
    )
    assert all(
        row.rule_score is None
        for arm in (V5Arm.ENSEMBLE_NO_GRAPH, V5Arm.ENSEMBLE_WITH_GRAPH)
        for row in scored.by_arm[arm].rows
    )
    assert all(
        row.arm_spec_sha256 == result.spec.spec_sha256
        for result in scored.by_arm.values()
        for row in result.rows
    )
    full_defender = trained.by_arm[V5Arm.FULL_SENTINEL].defender
    assert full_defender is not None
    thresholds = full_defender.thresholds
    challenge_action, challenge_disagreement, challenge_novelty = (
        v5_arms.route_full_sentinel_components(
            probability=thresholds.challenge_threshold,
            disagreement=0.0,
            novelty=0.0,
            thresholds=thresholds,
        )
    )
    disagreement_action, disagreement_routed, _ = (
        v5_arms.route_full_sentinel_components(
            probability=thresholds.challenge_threshold,
            disagreement=thresholds.disagreement_review_threshold,
            novelty=0.0,
            thresholds=thresholds,
        )
    )
    assert challenge_action == SentinelAction.CHALLENGE
    assert challenge_disagreement is False and challenge_novelty is False
    assert disagreement_action == SentinelAction.REVIEW_HOLD
    assert disagreement_routed is True
    clear_novelty, _, clear_novelty_routed = v5_arms.route_full_sentinel_components(
        probability=0.0,
        disagreement=0.0,
        novelty=0.0,
        thresholds=thresholds,
    )
    routed_novelty, _, novelty_routed = v5_arms.route_full_sentinel_components(
        probability=0.0,
        disagreement=0.0,
        novelty=1.0,
        thresholds=thresholds,
    )
    assert clear_novelty == SentinelAction.APPROVE and clear_novelty_routed is False
    assert routed_novelty == SentinelAction.REVIEW_HOLD and novelty_routed is True

    graph_mutation = x_test.copy()
    graph_mutation[:, graph_index] = 100.0 - graph_mutation[:, graph_index]
    mutated = score(
        trained=trained,
        catalog=catalog,
        features_matrix=graph_mutation,
        support=support,
        execution_artifacts=execution_artifacts,
        trust_failures=trust_failures,
    )
    baseline = score(
        trained=trained,
        catalog=catalog,
        features_matrix=x_test,
        support=support,
        execution_artifacts=execution_artifacts,
        trust_failures=trust_failures,
    )

    def probabilities(result: V5ArmScore) -> tuple[float, ...]:
        return tuple(row.probability for row in result.rows)

    assert probabilities(
        mutated.by_arm[V5Arm.ENSEMBLE_NO_GRAPH]
    ) == probabilities(baseline.by_arm[V5Arm.ENSEMBLE_NO_GRAPH])
    assert probabilities(mutated.by_arm[V5Arm.RULES_ONLY]) == probabilities(
        baseline.by_arm[V5Arm.RULES_ONLY]
    )
    assert tuple(row.action for row in mutated.by_arm[V5Arm.RULES_ONLY].rows) == tuple(
        row.action for row in baseline.by_arm[V5Arm.RULES_ONLY].rows
    )
    assert probabilities(
        mutated.by_arm[V5Arm.ENSEMBLE_WITH_GRAPH]
    ) != probabilities(baseline.by_arm[V5Arm.ENSEMBLE_WITH_GRAPH])
    assert tuple(
        row.action for row in scored.by_arm[V5Arm.ENSEMBLE_NO_GRAPH].rows
    ) == tuple(row.action for row in baseline.by_arm[V5Arm.ENSEMBLE_NO_GRAPH].rows)

    rule_mutation = x_test.copy()
    rule_mutation[1, catalog.feature_names.index("actor_count_1m")] = 100.0
    rule_mutated = score(
        trained=trained,
        catalog=catalog,
        features_matrix=rule_mutation,
        support=support,
        execution_artifacts=execution_artifacts,
        trust_failures=trust_failures,
    )
    full_rule_row = rule_mutated.by_arm[V5Arm.FULL_SENTINEL].rows[1]
    assert full_rule_row.rule_action == SentinelAction.DECLINE_HOLD
    assert full_rule_row.model_action is not None
    assert full_rule_row.action == SentinelAction.DECLINE_HOLD
    assert full_rule_row.action.severity > full_rule_row.model_action.severity

    invalid = scored.model_dump(mode="json")
    invalid["by_arm"][V5Arm.FULL_SENTINEL.value]["rows"] = list(
        reversed(invalid["by_arm"][V5Arm.FULL_SENTINEL.value]["rows"])
    )
    with pytest.raises(ValueError, match="support|order"):
        type(scored).model_validate(invalid)

    def rebind_score(document: dict) -> None:
        for row in document["rows"]:
            row["row_output_sha256"] = independent_digest(
                {key: value for key, value in row.items() if key != "row_output_sha256"}
            )
        document["score_sha256"] = independent_digest(
            {key: value for key, value in document.items() if key != "score_sha256"}
        )

    for threshold_name, forged_value in (
        ("disagreement_review", 0.150000001),
        ("novelty_challenge", 0.700000001),
        ("novelty_review", 0.900000001),
    ):
        full_threshold_forge = deepcopy(
            scored.by_arm[V5Arm.FULL_SENTINEL].model_dump(mode="json")
        )
        forged_thresholds = dict(full_threshold_forge["spec"]["threshold_values"])
        forged_thresholds[threshold_name] = forged_value
        full_threshold_forge["spec"]["threshold_values"] = sorted(
            forged_thresholds.items()
        )
        threshold_partition = full_threshold_forge["spec"]["training_partitions"][2]
        full_threshold_forge["spec"]["threshold_digest"] = independent_digest(
            {
                "source_partition": "threshold",
                "method": "sentinel_percentile_v1",
                "threshold_ordered_rows_sha256": threshold_partition[
                    "ordered_rows_sha256"
                ],
                "threshold_support_sha256": threshold_partition[
                    "ordered_support_sha256"
                ],
                "threshold_feature_batch_sha256": threshold_partition[
                    "feature_batch_sha256"
                ],
                "threshold_feature_matrix_sha256": threshold_partition[
                    "feature_matrix_sha256"
                ],
                "threshold_values": full_threshold_forge["spec"]["threshold_values"],
            }
        )
        full_threshold_forge["spec"]["spec_sha256"] = independent_digest(
            {
                key: value
                for key, value in full_threshold_forge["spec"].items()
                if key != "spec_sha256"
            }
        )
        for row in full_threshold_forge["rows"]:
            row["arm_spec_sha256"] = full_threshold_forge["spec"]["spec_sha256"]
            row["threshold_trace"].update(forged_thresholds)
        rebind_score(full_threshold_forge)
        with pytest.raises(ValueError, match="fixed full sentinel threshold"):
            V5ArmScore.model_validate(full_threshold_forge)

    immutable_row = scored.by_arm[V5Arm.FULL_SENTINEL].rows[0]
    with pytest.raises(TypeError):
        immutable_row.threshold_trace["novelty_review"] = 0.1
    with pytest.raises(TypeError):
        scored.by_arm[V5Arm.RULES_ONLY] = scored.by_arm[V5Arm.RULES_ONLY]
    assert scored.model_dump_json() == scored.model_dump_json()

    full_score = scored.by_arm[V5Arm.FULL_SENTINEL]
    source_spec = full_score.spec.model_dump(mode="json")

    too_many_members = deepcopy(source_spec)
    too_many_members["model_seeds"] = list(range(1_101, 1_107))
    too_many_members["model_artifacts"] = [source_spec["model_artifacts"][0]] * 6
    too_many_members["model_artifact_sha256"] = [
        source_spec["model_artifact_sha256"][0]
    ] * 6
    too_many_members["calibrator_manifests"] = [
        source_spec["calibrator_manifests"][0]
    ] * 6
    too_many_members["calibrator_artifact_sha256"] = [
        source_spec["calibrator_artifact_sha256"][0]
    ] * 6
    too_many_members["spec_sha256"] = independent_digest(
        {
            key: value
            for key, value in too_many_members.items()
            if key != "spec_sha256"
        }
    )
    with pytest.raises(ValueError, match="model member count.*production profile"):
        V5ArmSpecification.model_validate(too_many_members)

    too_few_members = deepcopy(source_spec)
    too_few_members["model_seeds"] = [1_101, 1_102]
    too_few_members["model_artifacts"] = source_spec["model_artifacts"][:2]
    too_few_members["model_artifact_sha256"] = source_spec[
        "model_artifact_sha256"
    ][:2]
    too_few_members["calibrator_manifests"] = source_spec[
        "calibrator_manifests"
    ][:2]
    too_few_members["calibrator_artifact_sha256"] = source_spec[
        "calibrator_artifact_sha256"
    ][:2]
    too_few_members["spec_sha256"] = independent_digest(
        {
            key: value
            for key, value in too_few_members.items()
            if key != "spec_sha256"
        }
    )
    with pytest.raises(ValueError, match="model member count.*production profile"):
        V5ArmSpecification.model_validate(too_few_members)

    knot_count = 20_001
    aggregate_calibrator = {
        "x_thresholds": [
            index / (knot_count - 1) for index in range(knot_count)
        ],
        "y_thresholds": [
            index / (knot_count - 1) for index in range(knot_count)
        ],
        "out_of_bounds": "clip",
    }
    aggregate_calibrator["artifact_sha256"] = independent_digest(
        aggregate_calibrator
    )
    excessive_aggregate_knots = deepcopy(source_spec)
    excessive_aggregate_knots["model_seeds"] = list(range(1_201, 1_206))
    excessive_aggregate_knots["model_artifacts"] = [
        source_spec["model_artifacts"][0]
    ] * 5
    excessive_aggregate_knots["model_artifact_sha256"] = [
        source_spec["model_artifact_sha256"][0]
    ] * 5
    excessive_aggregate_knots["calibrator_manifests"] = [
        aggregate_calibrator
    ] * 5
    excessive_aggregate_knots["calibrator_artifact_sha256"] = [
        aggregate_calibrator["artifact_sha256"]
    ] * 5
    excessive_aggregate_knots["spec_sha256"] = independent_digest(
        {
            key: value
            for key, value in excessive_aggregate_knots.items()
            if key != "spec_sha256"
        }
    )
    with pytest.raises(ValueError, match="aggregate calibrator knot count.*limit"):
        V5ArmSpecification.model_validate(excessive_aggregate_knots)

    synchronized_novelty_expansion = deepcopy(source_spec)
    novelty_document = synchronized_novelty_expansion["novelty_manifest"]
    original_feature_count = novelty_document["feature_count"]
    novelty_document["feature_count"] = original_feature_count + 1
    for tree in novelty_document["trees"]:
        tree["estimator_features"].append(original_feature_count)
    novelty_document["artifact_sha256"] = independent_digest(
        {
            key: value
            for key, value in novelty_document.items()
            if key != "artifact_sha256"
        }
    )
    synchronized_novelty_expansion["novelty_artifact_sha256"] = novelty_document[
        "artifact_sha256"
    ]
    synchronized_novelty_expansion["spec_sha256"] = independent_digest(
        {
            key: value
            for key, value in synchronized_novelty_expansion.items()
            if key != "spec_sha256"
        }
    )
    with pytest.raises(ValueError, match="novelty feature count.*arm features"):
        V5ArmSpecification.model_validate(synchronized_novelty_expansion)

    oversized_rows = full_score.model_copy(
        update={"rows": (full_score.rows[0],) * 100_001}
    )
    with pytest.raises(ValueError, match="score row.*limit"):
        oversized_rows.rows_match_specification()
    largest_execution = max(
        execution_artifacts, key=lambda artifact: len(artifact.payload_json.encode())
    )
    oversized_execution_count = full_score.model_copy(
        update={"execution_artifacts": (largest_execution,) * 4_097}
    )
    with pytest.raises(ValueError, match="execution artifact count.*limit"):
        oversized_execution_count.rows_match_specification()
    artifact_repetitions = (
        268_435_456 // len(largest_execution.payload_json.encode()) + 1
    )
    assert artifact_repetitions < 4_096
    oversized_execution_bytes = full_score.model_copy(
        update={
            "execution_artifacts": (largest_execution,) * artifact_repetitions
        }
    )
    with pytest.raises(ValueError, match="execution artifact bytes.*limit"):
        oversized_execution_bytes.rows_match_specification()
    first_feature_row = train_provenance.feature_matrix[0]
    matrix_repetitions = 10_000_000 // len(first_feature_row) + 1
    oversized_matrix = train_provenance.model_copy(
        update={"feature_matrix": (first_feature_row,) * matrix_repetitions}
    )
    with pytest.raises(ValueError, match="feature matrix cell.*limit"):
        oversized_matrix.ordered_provenance_is_complete()
    novelty_manifest = full_score.spec.novelty_manifest
    assert novelty_manifest is not None
    source_tree = novelty_manifest.trees[0]
    large_tree = source_tree.model_copy(
        update={
            "children_left": (-1,) * 1_025,
            "children_right": (-1,) * 1_025,
            "feature": (-2,) * 1_025,
            "threshold": (0.0,) * 1_025,
            "decision_path_lengths": (1.0,) * 1_025,
            "average_path_lengths": (0.0,) * 1_025,
        }
    )
    oversized_forest = novelty_manifest.model_copy(
        update={"trees": (large_tree,) * 256}
    )
    with pytest.raises(ValueError, match="isolation forest node.*limit"):
        oversized_forest.forest_is_content_addressed()
    model_artifact = full_score.spec.model_artifacts[0]
    model_repetitions = 67_108_864 // len(model_artifact.payload()) + 1
    oversized_models = full_score.spec.model_copy(
        update={"model_artifacts": (model_artifact,) * model_repetitions}
    )
    with pytest.raises(ValueError, match="CatBoost artifact bytes.*limit"):
        oversized_models.specification_is_bound()

    for result in scored.by_arm.values():
        assert result.score_sha256 == independent_digest(
            result.model_dump(mode="json", exclude={"score_sha256"})
        )
        assert all(
            row.row_output_sha256
            == independent_digest(row.model_dump(mode="json", exclude={"row_output_sha256"}))
            for row in result.rows
        )

    calibrated_tamper = deepcopy(
        scored.by_arm[V5Arm.ENSEMBLE_WITH_GRAPH].model_dump(mode="json")
    )
    original_score = calibrated_tamper["rows"][0]["model_calibrated_scores"][0]
    calibrated_tamper["rows"][0]["model_calibrated_scores"][0] = (
        original_score - 0.125 if original_score > 0.5 else original_score + 0.125
    )
    rebind_score(calibrated_tamper)
    with pytest.raises(ValueError, match="probability|disagreement|calibrator"):
        V5ArmScore.model_validate(calibrated_tamper)

    raw_score_tamper = deepcopy(
        scored.by_arm[V5Arm.ENSEMBLE_WITH_GRAPH].model_dump(mode="json")
    )
    original_raw = raw_score_tamper["rows"][0]["model_raw_scores"][0]
    raw_score_tamper["rows"][0]["model_raw_scores"][0] = 1.0 - original_raw
    rebind_score(raw_score_tamper)
    with pytest.raises(ValueError, match="raw model|artifact replay|calibrator|calibrated"):
        V5ArmScore.model_validate(raw_score_tamper)

    feature_score_forge = deepcopy(
        scored.by_arm[V5Arm.ENSEMBLE_WITH_GRAPH].model_dump(mode="json")
    )
    graph_name = catalog.feature_names[graph_index]
    subset_graph_index = feature_score_forge["spec"]["feature_names"].index(graph_name)
    feature_score_forge["rows"][0]["catalog_feature_values"][graph_index] -= 100.0
    feature_score_forge["rows"][0]["subset_feature_values"][subset_graph_index] -= 100.0
    feature_score_forge["rows"][0]["catalog_feature_sha256"] = independent_digest(
        feature_score_forge["rows"][0]["catalog_feature_values"]
    )
    feature_score_forge["rows"][0]["subset_feature_sha256"] = independent_digest(
        feature_score_forge["rows"][0]["subset_feature_values"]
    )
    rebind_score(feature_score_forge)
    with pytest.raises(ValueError, match="raw model|model artifact|replay"):
        V5ArmScore.model_validate(feature_score_forge)

    novelty_forge = deepcopy(scored.by_arm[V5Arm.FULL_SENTINEL].model_dump(mode="json"))
    novelty_row = next(
        row for row in novelty_forge["rows"] if not row["trust_routed"]
    )
    novelty_row["novelty_raw_score"] += 0.000001
    novelty_row["novelty_score"] = max(
        0.0, min(1.0, 0.5 - novelty_row["novelty_raw_score"])
    )
    rebind_score(novelty_forge)
    with pytest.raises(ValueError, match="novelty artifact|novelty replay"):
        V5ArmScore.model_validate(novelty_forge)

    trust_forge = deepcopy(scored.by_arm[V5Arm.FULL_SENTINEL].model_dump(mode="json"))
    trust_row = next(row for row in trust_forge["rows"] if not row["trust_routed"])
    trust_row["trust_action"] = SentinelAction.DECLINE_HOLD.value
    trust_row["trust_routed"] = True
    trust_row["action"] = SentinelAction.DECLINE_HOLD.value
    rebind_score(trust_forge)
    with pytest.raises(ValueError, match="trust evidence|verifier"):
        V5ArmScore.model_validate(trust_forge)

    feature_stream_forge = scored.model_dump(mode="json")
    rules_document = feature_stream_forge["by_arm"][V5Arm.RULES_ONLY.value]
    non_rule_index = next(
        index
        for index, name in enumerate(rules_document["spec"]["catalog_feature_names"])
        if name not in rules_document["spec"]["feature_names"]
    )
    rules_document["rows"][0]["catalog_feature_values"][non_rule_index] += 0.25
    rules_document["rows"][0]["catalog_feature_sha256"] = independent_digest(
        rules_document["rows"][0]["catalog_feature_values"]
    )
    rebind_score(rules_document)
    with pytest.raises(ValueError, match="catalog feature|feature stream"):
        type(scored).model_validate(feature_stream_forge)

    disabled_tamper = deepcopy(
        scored.by_arm[V5Arm.ENSEMBLE_NO_GRAPH].model_dump(mode="json")
    )
    disabled_tamper["rows"][0]["novelty_score"] = 0.5
    disabled_tamper["rows"][0]["novelty_raw_score"] = 0.1
    rebind_score(disabled_tamper)
    with pytest.raises(ValueError, match="disabled novelty"):
        V5ArmScore.model_validate(disabled_tamper)

    feature_tamper = deepcopy(
        scored.by_arm[V5Arm.ENSEMBLE_NO_GRAPH].model_dump(mode="json")
    )
    feature_tamper["rows"][0]["catalog_feature_values"][0] += 1.0
    feature_tamper["rows"][0]["catalog_feature_sha256"] = independent_digest(
        feature_tamper["rows"][0]["catalog_feature_values"]
    )
    rebind_score(feature_tamper)
    with pytest.raises(ValueError, match="subset feature"):
        V5ArmScore.model_validate(feature_tamper)

    action_tamper = deepcopy(scored.by_arm[V5Arm.RULES_ONLY].model_dump(mode="json"))
    action_tamper["rows"][0]["action"] = SentinelAction.APPROVE.value
    rebind_score(action_tamper)
    with pytest.raises(ValueError, match="final action"):
        V5ArmScore.model_validate(action_tamper)
