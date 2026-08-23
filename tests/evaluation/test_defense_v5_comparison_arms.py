"""Comparison arms regression tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import apar.evaluation.v5_evaluation as v5_evaluation
from apar.evaluation.v5_evaluation import V5Arm, V5ArmScore
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
    assert len(trained.by_arm[V5Arm.RULES_ONLY].spec.feature_names) == 9
    assert len(trained.by_arm[V5Arm.ENSEMBLE_NO_GRAPH].spec.feature_names) == 35
    assert len(trained.by_arm[V5Arm.ENSEMBLE_WITH_GRAPH].spec.feature_names) == 46
    assert len(trained.by_arm[V5Arm.FULL_SENTINEL].spec.feature_names) == 46
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
