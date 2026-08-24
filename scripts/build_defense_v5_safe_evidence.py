"""Build a non-published seed-404 Sentinel v5 evidence fixture."""

from __future__ import annotations

import argparse
from pathlib import Path

from apar.evaluation.v5_controls import execute_v5_controls
from apar.evaluation.v5_evaluation import (
    V5Arm,
    V5EvaluationResult,
    load_v5_arm_configuration,
)
from apar.evaluation.v5_evidence_bundle import build_v5_evidence_envelope
from apar.evaluation.v5_evidence_protocol import load_v5_evidence_protocol
from apar.evaluation.v5_population import build_v5_corpus
from apar.evaluation.v5_protocol import (
    V5DevelopmentProtocol,
    V5Profile,
    load_v5_development_protocol,
    v5_protocol_digest,
)
from apar.features.sentinel import SentinelFeatureCatalog
from scripts.run_defense_v5_development import _score_all_arms_and_evaluate


def _safe_protocol(root: Path) -> V5DevelopmentProtocol:
    locked = load_v5_development_protocol(
        root / "config/defense/defense-v5-development.json"
    )
    if locked.seeds.development_test != 2404:
        raise ValueError("locked development-test seed binding changed")
    safe = locked.model_copy(
        update={
            "seeds": locked.seeds.model_copy(update={"development_test": 404}),
            "protocol_sha256": "",
        }
    )
    return safe.model_copy(update={"protocol_sha256": v5_protocol_digest(safe)})


def build_safe_evidence(root: Path) -> bytes:
    """Execute only the frozen smoke profile with the isolated safe seed."""
    protocol = _safe_protocol(root)
    evidence_protocol = load_v5_evidence_protocol(
        root / "config/defense/defense-v5-evidence.json", root=root
    )
    catalog = SentinelFeatureCatalog.from_config(root / protocol.feature_catalog_path)
    configuration = load_v5_arm_configuration(
        root / "config/defense/defense-v5-arms.json",
        catalog=catalog,
        protocol=protocol,
    )
    corpus = build_v5_corpus(protocol, profile=V5Profile.SMOKE)
    scored = _score_all_arms_and_evaluate(
        train_decisions=corpus.partitions["train"].decisions,
        train_executions=corpus.partitions["train"].executions,
        calibration_decisions=corpus.partitions["calibration"].decisions,
        calibration_executions=corpus.partitions["calibration"].executions,
        threshold_decisions=corpus.partitions["threshold"].decisions,
        threshold_executions=corpus.partitions["threshold"].executions,
        dev_test_decisions=corpus.partitions["development_test"].decisions,
        dev_test_executions=corpus.partitions["development_test"].executions,
        catalog=catalog,
        configuration=configuration,
        bootstrap_seed=protocol.seeds.bootstrap,
    )
    controls = execute_v5_controls(
        protocol=protocol,
        evidence_protocol=evidence_protocol,
        corpus=corpus,
        catalog=catalog,
        configuration=configuration,
    )
    results = tuple(
        V5EvaluationResult.model_validate(scored["arm_results"][arm.value])
        for arm in (
            V5Arm.RULES_ONLY,
            V5Arm.ENSEMBLE_NO_GRAPH,
            V5Arm.ENSEMBLE_WITH_GRAPH,
            V5Arm.FULL_SENTINEL,
        )
    )
    return build_v5_evidence_envelope(
        seed=404,
        evidence_protocol=evidence_protocol,
        catalog_sha256=catalog.catalog_sha256,
        arm_results=results,
        controls=controls,
    ).serialized_bytes()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    output = args.output.resolve()
    if output.exists():
        raise SystemExit("refusing to overwrite an existing safe evidence fixture")
    output.write_bytes(build_safe_evidence(root))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
