"""Execute the complete Sentinel v5 evidence workload for one closed run mode."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Any

from apar.evaluation.v5_controls import (
    V5ExecutedControlSuite,
    execute_v5_controls,
)
from apar.evaluation.v5_evaluation import (
    V5Arm,
    V5EvaluationResult,
    load_v5_arm_configuration,
)
from apar.evaluation.v5_evidence_protocol import V5EvidenceProtocol, load_v5_evidence_protocol
from apar.evaluation.v5_population import V5Corpus, build_v5_corpus
from apar.evaluation.v5_protocol import (
    V5DevelopmentProtocol,
    load_v5_development_protocol,
    v5_protocol_digest,
)
from apar.evaluation.v5_run_mode import (
    V5LockedEvidenceRunBinding,
    V5RunMode,
    resolve_v5_run_mode,
)
from apar.features.sentinel import SentinelFeatureCatalog

_runner_module: Any = import_module(
    f"{__package__}.run_defense_v5_development"
    if __package__
    else "run_defense_v5_development"
)
_score_all_arms_and_evaluate: Any = _runner_module._score_all_arms_and_evaluate

_ARM_ORDER = (
    V5Arm.RULES_ONLY,
    V5Arm.ENSEMBLE_NO_GRAPH,
    V5Arm.ENSEMBLE_WITH_GRAPH,
    V5Arm.FULL_SENTINEL,
)
_LOCKED_CAPABILITY_SEAL = object()


@dataclass(frozen=True)
class _LockedExecutionCapability:
    run_binding_sha256: str
    seal: object


def _issue_locked_execution_capability(
    binding: V5LockedEvidenceRunBinding,
) -> _LockedExecutionCapability:
    """Issue the in-process capability only after the runner's preflight."""
    return _LockedExecutionCapability(
        run_binding_sha256=binding.run_binding_sha256,
        seal=_LOCKED_CAPABILITY_SEAL,
    )


@dataclass(frozen=True)
class V5CompleteEvidenceExecution:
    """Immutable products required by either complete evidence envelope."""

    mode: V5RunMode
    protocol: V5DevelopmentProtocol
    evidence_protocol: V5EvidenceProtocol
    catalog: SentinelFeatureCatalog
    corpus: V5Corpus
    arm_results: tuple[V5EvaluationResult, ...]
    controls: V5ExecutedControlSuite


def _protocol_for_mode(
    *, root: Path, mode: V5RunMode, evidence_protocol: V5EvidenceProtocol
) -> tuple[V5DevelopmentProtocol, object]:
    locked = load_v5_development_protocol(
        root / "config/defense/defense-v5-development.json"
    )
    binding = resolve_v5_run_mode(
        mode=mode,
        evidence_protocol=evidence_protocol,
        development_protocol=locked,
    )
    if mode is V5RunMode.LOCKED_DEVELOPMENT:
        return locked, binding
    safe = locked.model_copy(
        update={
            "seeds": locked.seeds.model_copy(
                update={"development_test": binding.development_test_seed}
            ),
            "protocol_sha256": "",
        }
    )
    return safe.model_copy(update={"protocol_sha256": v5_protocol_digest(safe)}), binding


def execute_v5_complete_evidence(
    *,
    root: Path,
    mode: V5RunMode,
    locked_capability: _LockedExecutionCapability | None = None,
) -> V5CompleteEvidenceExecution:
    """Build corpus, all four arms, and all controls for one closed mode."""
    if mode is V5RunMode.LOCKED_DEVELOPMENT and (
        locked_capability is None
        or locked_capability.seal is not _LOCKED_CAPABILITY_SEAL
        or len(locked_capability.run_binding_sha256) != 64
    ):
        raise PermissionError("locked execution requires verified preflight capability")
    if mode is V5RunMode.SAFE_VALIDATION and locked_capability is not None:
        raise ValueError("safe validation cannot consume locked execution authority")
    root = root.resolve()
    evidence_protocol = load_v5_evidence_protocol(
        root / "config/defense/defense-v5-evidence.json", root=root
    )
    protocol, raw_binding = _protocol_for_mode(
        root=root, mode=mode, evidence_protocol=evidence_protocol
    )
    profile = raw_binding.profile  # type: ignore[attr-defined]
    catalog = SentinelFeatureCatalog.from_config(root / protocol.feature_catalog_path)
    configuration = load_v5_arm_configuration(
        root / "config/defense/defense-v5-arms.json",
        catalog=catalog,
        protocol=protocol,
    )
    corpus = build_v5_corpus(protocol, profile=profile)
    partitions = corpus.partitions
    scored = _score_all_arms_and_evaluate(
        train_decisions=partitions["train"].decisions,
        train_executions=partitions["train"].executions,
        calibration_decisions=partitions["calibration"].decisions,
        calibration_executions=partitions["calibration"].executions,
        threshold_decisions=partitions["threshold"].decisions,
        threshold_executions=partitions["threshold"].executions,
        dev_test_decisions=partitions["development_test"].decisions,
        dev_test_executions=partitions["development_test"].executions,
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
        mode=mode,
    )
    results = tuple(
        V5EvaluationResult.model_validate(scored["arm_results"][arm.value])
        for arm in _ARM_ORDER
    )
    return V5CompleteEvidenceExecution(
        mode=mode,
        protocol=protocol,
        evidence_protocol=evidence_protocol,
        catalog=catalog,
        corpus=corpus,
        arm_results=results,
        controls=controls,
    )


__all__ = ["V5CompleteEvidenceExecution", "execute_v5_complete_evidence"]
