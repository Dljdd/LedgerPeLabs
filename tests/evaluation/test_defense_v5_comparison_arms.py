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
    build_v5_training_partition_evidence,
)
from apar.evaluation.v5_protocol import load_v5_development_protocol
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

    def partition(size: int) -> tuple[np.ndarray, np.ndarray]:
        labels = np.array(([0, 1] * ((size + 1) // 2))[:size], dtype=int)
        matrix = rng.normal(0.0, 0.05, (size, len(catalog.feature_names)))
        matrix[:, graph_index] = labels * 8.0
        matrix[:, integrity_index] = labels * 9.0
        return matrix, labels

    x_train, y_train = partition(32)
    x_cal, y_cal = partition(16)
    x_threshold, y_threshold = partition(16)
    x_test, y_test = partition(8)
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

    def provenance(name: str, matrix: np.ndarray, labels: np.ndarray, ordinal: int):
        event_ids = tuple(f"{name}-event-{index}" for index in range(len(labels)))
        evidence_support = tuple(
            support_type(
                event_id=event_id,
                label=int(labels[index]),
                campaign_id=f"{name}-campaign-{index // 2}",
                amount=float(index + 1),
                family="legitimate" if not labels[index] else "card_testing_cnp",
                execution_evidence_sha256=f"{ordinal + index + 1:064x}",
            )
            for index, event_id in enumerate(event_ids)
        )
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
            catalog_sha256=catalog.catalog_sha256,
        )

    train_provenance = provenance("train", x_train, y_train, 100)
    calibration_provenance = provenance("calibration", x_cal, y_cal, 200)
    threshold_provenance = provenance("threshold", x_threshold, y_threshold, 300)
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
    support = tuple(
        support_type(
            event_id=f"event-{index}",
            label=int(y_test[index]),
            campaign_id=f"campaign-{index // 2}",
            amount=float(index + 1),
            family="legitimate" if not y_test[index] else "card_testing_cnp",
            execution_evidence_sha256=f"{index + 1:064x}",
        )
        for index in range(len(x_test))
    )
    trust_failures = [True] + [False] * (len(x_test) - 1)
    scored = score(
        trained=trained,
        catalog=catalog,
        features_matrix=x_test,
        support=support,
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
        trust_failures=[False] * len(x_test),
        novelty_scores=[1.0] * len(x_test),
    )
    original_no_trust = score(
        trained=trained,
        catalog=catalog,
        features_matrix=x_test,
        support=support,
        trust_failures=[False] * len(x_test),
        novelty_scores=[0.0] * len(x_test),
    )

    def probabilities(result: V5ArmScore) -> tuple[float, ...]:
        return tuple(row.probability for row in result.rows)

    assert probabilities(
        mutated.by_arm[V5Arm.ENSEMBLE_NO_GRAPH]
    ) == probabilities(original_no_trust.by_arm[V5Arm.ENSEMBLE_NO_GRAPH])
    assert probabilities(mutated.by_arm[V5Arm.RULES_ONLY]) == probabilities(
        original_no_trust.by_arm[V5Arm.RULES_ONLY]
    )
    assert tuple(row.action for row in mutated.by_arm[V5Arm.RULES_ONLY].rows) == tuple(
        row.action for row in original_no_trust.by_arm[V5Arm.RULES_ONLY].rows
    )
    assert probabilities(
        mutated.by_arm[V5Arm.ENSEMBLE_WITH_GRAPH]
    ) != probabilities(original_no_trust.by_arm[V5Arm.ENSEMBLE_WITH_GRAPH])
    assert tuple(
        row.action for row in scored.by_arm[V5Arm.ENSEMBLE_NO_GRAPH].rows
    ) == tuple(row.action for row in original_no_trust.by_arm[V5Arm.ENSEMBLE_NO_GRAPH].rows)

    novelty_mutated = score(
        trained=trained,
        catalog=catalog,
        features_matrix=x_test,
        support=support,
        trust_failures=[False] * len(x_test),
        novelty_scores=[1.0] * len(x_test),
    )
    for arm in (V5Arm.ENSEMBLE_NO_GRAPH, V5Arm.ENSEMBLE_WITH_GRAPH):
        assert probabilities(novelty_mutated.by_arm[arm]) == probabilities(
            original_no_trust.by_arm[arm]
        )
        assert tuple(row.action for row in novelty_mutated.by_arm[arm].rows) == tuple(
            row.action for row in original_no_trust.by_arm[arm].rows
        )
    original_full_row = original_no_trust.by_arm[V5Arm.FULL_SENTINEL].rows[0]
    novelty_full_row = novelty_mutated.by_arm[V5Arm.FULL_SENTINEL].rows[0]
    assert original_full_row.novelty_routed is False
    assert novelty_full_row.novelty_routed is True
    assert novelty_full_row.action.severity > original_full_row.action.severity

    rule_mutation = x_test.copy()
    rule_mutation[0, catalog.feature_names.index("actor_count_1m")] = 100.0
    rule_mutated = score(
        trained=trained,
        catalog=catalog,
        features_matrix=rule_mutation,
        support=support,
        trust_failures=[False] * len(x_test),
        novelty_scores=[0.0] * len(x_test),
    )
    full_rule_row = rule_mutated.by_arm[V5Arm.FULL_SENTINEL].rows[0]
    assert full_rule_row.rule_action == SentinelAction.DECLINE_HOLD
    assert full_rule_row.model_action is not None
    assert full_rule_row.action == SentinelAction.DECLINE_HOLD
    assert full_rule_row.action.severity > full_rule_row.model_action.severity

    trust_mutation_matrix = x_test.copy()
    trust_mutation_matrix[:, integrity_index] = 100.0 - trust_mutation_matrix[:, integrity_index]
    trust_mutated = score(
        trained=trained,
        catalog=catalog,
        features_matrix=trust_mutation_matrix,
        support=support,
        trust_failures=[True] * len(x_test),
    )
    for arm in (V5Arm.ENSEMBLE_NO_GRAPH, V5Arm.ENSEMBLE_WITH_GRAPH):
        assert probabilities(trust_mutated.by_arm[arm]) == probabilities(
            original_no_trust.by_arm[arm]
        )
        assert tuple(row.action for row in trust_mutated.by_arm[arm].rows) == tuple(
            row.action for row in original_no_trust.by_arm[arm].rows
        )

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
    with pytest.raises(ValueError, match="calibrator|calibrated"):
        V5ArmScore.model_validate(raw_score_tamper)

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
