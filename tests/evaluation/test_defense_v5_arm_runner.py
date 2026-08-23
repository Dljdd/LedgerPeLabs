"""Real-corpus integration for the four frozen Sentinel v5 arms."""

from __future__ import annotations

from pathlib import Path

from apar.evaluation.v5_evaluation import V5Arm, load_v5_arm_configuration
from apar.evaluation.v5_population import build_v5_corpus
from apar.evaluation.v5_protocol import V5Profile
from apar.features.sentinel import SentinelFeatureCatalog
from scripts import run_defense_v5_development as runner
from tests.evaluation.v5_safe_protocol import load_safe_v5_test_protocol

ROOT = Path(__file__).resolve().parents[2]


def test_runner_scores_four_arms_over_identical_real_execution_support() -> None:
    """A full-only or cloned runner cannot produce independent bound row evidence."""
    score_all = getattr(runner, "_score_all_arms_and_evaluate", None)
    assert callable(score_all), "runner four-arm integration is missing"

    protocol = load_safe_v5_test_protocol(ROOT)
    catalog = SentinelFeatureCatalog.default()
    configuration = load_v5_arm_configuration(
        ROOT / "config/defense/defense-v5-arms.json",
        catalog=catalog,
        protocol=protocol,
    )
    corpus = build_v5_corpus(protocol, profile=V5Profile.SMOKE)
    output = score_all(
        train_decisions=corpus.partitions["train"].decisions,
        calibration_decisions=corpus.partitions["calibration"].decisions,
        threshold_decisions=corpus.partitions["threshold"].decisions,
        dev_test_decisions=corpus.partitions["development_test"].decisions,
        catalog=catalog,
        configuration=configuration,
        bootstrap_seed=protocol.seeds.bootstrap,
    )

    results = output["arm_results"]
    assert tuple(results) == (
        V5Arm.RULES_ONLY.value,
        V5Arm.ENSEMBLE_NO_GRAPH.value,
        V5Arm.ENSEMBLE_WITH_GRAPH.value,
        V5Arm.FULL_SENTINEL.value,
    )
    assert len({result["support_sha256"] for result in results.values()}) == 1
    expected_event_ids = tuple(
        row.event_id for row in corpus.partitions["development_test"].decisions
    )
    assert all(
        tuple(item["support"]["event_id"] for item in result["row_evidence"])
        == expected_event_ids
        for result in results.values()
    )
    assert any(
        row["trust_routed"]
        for row in results[V5Arm.FULL_SENTINEL.value]["row_evidence"]
    )
    assert not any(
        row["trust_routed"]
        for arm in (V5Arm.ENSEMBLE_NO_GRAPH, V5Arm.ENSEMBLE_WITH_GRAPH)
        for row in results[arm.value]["row_evidence"]
    )
