"""Static dependency and obsolete-code regression tests."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
POPULATION = ROOT / "src" / "apar" / "evaluation" / "v5_population.py"
FEATURES = ROOT / "src" / "apar" / "features" / "sentinel.py"


class TestNoObsoleteLeakingCode:
    def test_population_does_not_contain_enrich_features(self) -> None:
        source = POPULATION.read_text()
        assert "_enrich_features" not in source, (
            "v5_population.py must not contain _enrich_features"
        )

    def test_population_does_not_compute_graph_features(self) -> None:
        source = POPULATION.read_text()
        for forbidden in (
            "graph_component_size", "graph_edge_density",
            "graph_shared_neighbor", "actor_count_", "pair_prior_count",
        ):
            assert forbidden not in source, (
                f"v5_population.py must not compute '{forbidden}'"
            )

    def test_feature_module_does_not_import_evaluator_labels(self) -> None:
        source = FEATURES.read_text()
        forbidden_imports = [
            "apar.evaluation.v5_reporting",
            "apar.evaluation_hidden",
        ]
        for forbidden in forbidden_imports:
            assert forbidden not in source, (
                f"feature module must not reference '{forbidden}'"
            )
