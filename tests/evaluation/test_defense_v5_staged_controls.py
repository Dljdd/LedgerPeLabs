"""Frozen grouped-control contracts for Sentinel v5 staged execution."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

import pytest

from apar.evaluation.v5_checkpoint_storage import (
    V5CheckpointInput,
    V5CheckpointObservation,
    publish_v5_checkpoint,
)
from apar.evaluation.v5_controls import (
    V5ControlGroup,
    V5ExecutedControlGroup,
    V5ExecutedControlSuite,
    assemble_v5_control_suite,
    execute_v5_control_group,
    execute_v5_controls,
)
from apar.evaluation.v5_evaluation import (
    V5Arm,
    V5EvaluationResult,
    load_v5_arm_configuration,
)
from apar.evaluation.v5_evidence_bundle import build_v5_readiness_evidence
from apar.evaluation.v5_evidence_protocol import load_v5_evidence_protocol
from apar.evaluation.v5_kaggle_protocol import (
    V5KaggleEnvironmentBinding,
    V5KaggleMode,
    V5KaggleStage,
    V5KaggleSupportPlan,
    load_v5_kaggle_protocol,
)
from apar.evaluation.v5_locked_evidence import (
    V5CheckpointChainBinding,
    V5StagedEvidencePayload,
    build_v5_staged_evidence_payload,
)
from apar.evaluation.v5_metrics import evaluate_v5_complete_result
from apar.evaluation.v5_population import V5Corpus, build_v5_corpus
from apar.evaluation.v5_protocol import V5Profile
from apar.evaluation.v5_run_mode import V5PartitionSupportPlan, V5RunMode
from apar.evaluation.v5_staged_evidence import (
    _control_group_core_and_observation,
    _issue_stage_capability,
    _iter_v5_corpus_records,
    _metric_stage_core_and_observation,
    _restore_control_group,
    _restore_metric_stage_evidence,
    build_v5_metric_stage_evidence,
    execute_v5_control_stage,
    load_v5_control_group_checkpoint,
)
from apar.features.sentinel import SentinelFeatureCatalog
from scripts.run_defense_v5_development import _score_all_arms_and_evaluate
from tests.evaluation.v5_safe_protocol import load_safe_v5_test_protocol

ROOT = Path(__file__).resolve().parents[2]
KAGGLE_CONFIG = ROOT / "config/defense/defense-v5-kaggle-recovery.json"

_EXPECTED_GROUPS = (
    V5ControlGroup.LABEL_SHUFFLE,
    V5ControlGroup.INVARIANCE,
    V5ControlGroup.SINGLE_CLASS,
)
_ExecutedEvidence = tuple[
    V5ExecutedControlSuite,
    tuple[V5ExecutedControlGroup, ...],
    V5Corpus,
    tuple[V5EvaluationResult, ...],
]


def _digest(document: object) -> str:
    return hashlib.sha256(
        json.dumps(document, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _capacity_support_plan(
    results: tuple[V5EvaluationResult, ...],
) -> V5KaggleSupportPlan:
    reference = results[0]
    assert reference.arm_spec is not None
    partitions: list[V5PartitionSupportPlan] = []
    artifact_ids: set[str] = set()
    partition_rows = {
        item.partition: item.support_records for item in reference.arm_spec.training_partitions
    }
    partition_artifacts = {
        item.partition: item.execution_artifacts for item in reference.arm_spec.training_partitions
    }
    partition_rows["development_test"] = tuple(item.support for item in reference.row_evidence)
    partition_artifacts["development_test"] = reference.execution_artifacts
    for name in ("train", "calibration", "threshold", "development_test"):
        support = partition_rows[name]
        fraud = Counter(item.family for item in support if item.label == 1)
        artifacts = partition_artifacts[name]
        artifact_ids.update(item.evidence_sha256 for item in artifacts)
        legitimate = sum(item.label == 0 for item in support)
        partitions.append(
            V5PartitionSupportPlan(
                partition=name,
                legitimate_rows=legitimate,
                fraud_rows_by_family=tuple(sorted(fraud.items())),
                total_rows=len(support),
                execution_artifacts=len(artifacts),
                execution_payload_estimate_bytes=max(1, len(artifacts)),
            )
        )
    values = {
        "mode": V5KaggleMode.CAPACITY_VALIDATION,
        "profile": "production",
        "partitions": tuple(partitions),
        "retained_execution_artifacts": len(artifact_ids),
        "retained_execution_payload_estimate_bytes": len(artifact_ids),
    }
    values["support_plan_sha256"] = _digest(
        {
            **values,
            "mode": V5KaggleMode.CAPACITY_VALIDATION.value,
            "partitions": [item.model_dump(mode="json") for item in partitions],
        }
    )
    return V5KaggleSupportPlan.model_validate(values)


def _chain() -> V5CheckpointChainBinding:
    stages = tuple(stage.value for stage in tuple(V5KaggleStage)[:-1])
    values = {
        "schema_version": "apar-sentinel-v5-checkpoint-chain/1",
        "attempt_receipt_sha256": "5" * 64,
        "predecessor_stage_manifest_sha256": tuple(
            (stage, f"{index:x}" * 64) for index, stage in enumerate(stages, start=1)
        ),
    }
    values["predecessor_chain_root_sha256"] = _digest(values)
    return V5CheckpointChainBinding.model_validate(values)


def _environment() -> V5KaggleEnvironmentBinding:
    return V5KaggleEnvironmentBinding.bind(
        provider="kaggle",
        image="python-cpu-test",
        image_sha256="1" * 64,
        python_version="3.12.5",
        architecture="x86_64",
        cpu_count=4,
        dependency_manifest_sha256="2" * 64,
        source_archive_sha256="3" * 64,
        notebook_sha256="4" * 64,
        internet_enabled=False,
        accelerator="none",
        file_fsync_supported=True,
        directory_fsync_supported=True,
        hardlink_no_replace_supported=True,
    )


def _observation() -> V5CheckpointObservation:
    return V5CheckpointObservation(
        schema_version="apar-sentinel-v5-kaggle-observation/1",
        started_at_utc="2026-08-24T00:00:00Z",
        completed_at_utc="2026-08-24T00:00:01Z",
        wall_seconds=1.0,
        rss_samples_bytes=(100_000,),
        host_available_samples_bytes=(8_000_000,),
        peak_rss_bytes=100_000,
        environment=_environment(),
    )


def _publish(
    *,
    output_root: Path,
    stage: V5KaggleStage,
    predecessor: object | None,
    records: object,
) -> object:
    protocol = load_v5_kaggle_protocol(KAGGLE_CONFIG, root=ROOT)
    return publish_v5_checkpoint(
        output_root=output_root,
        stage=stage,
        run_binding_sha256=protocol.run_binding_sha256(V5KaggleMode.CAPACITY_VALIDATION),
        attempt_receipt_sha256="5" * 64,
        predecessor=predecessor,
        records=records,
        environment=_environment(),
        observation=_observation(),
        limits=protocol.resources,
    )


def _stable_row_evidence(name: str, raw_json: str) -> object:
    document = json.loads(raw_json)
    if name in {
        "identity_rename",
        "future_causality",
        "equal_time_isolation",
        "feature_leakage",
    }:
        document.pop("before_score_sha256")
        document.pop("after_score_sha256")
    elif name in {"benign_only", "fraud_only_diagnostic"}:
        document.pop("full_score_sha256")
        for arm in document["arms"].values():
            arm.pop("score_sha256")
            for row in arm["rows"]:
                row.pop("latency_ms")
    return document


def _stable_control(control: object) -> dict[str, object]:
    document = control.model_dump(mode="json")
    document.pop("control_sha256")
    document.pop("row_evidence_sha256")
    raw_json = document.pop("row_evidence_json")
    document["row_evidence"] = _stable_row_evidence(document["name"], raw_json)
    document["measurements"] = [
        item for item in document["measurements"] if not item["name"].endswith("p95_latency_ms")
    ]
    return document


@pytest.fixture(scope="module")
def executed_group_evidence() -> _ExecutedEvidence:
    protocol = load_safe_v5_test_protocol(ROOT)
    evidence_protocol = load_v5_evidence_protocol(
        ROOT / "config/defense/defense-v5-evidence.json", root=ROOT
    )
    catalog = SentinelFeatureCatalog.default()
    configuration = load_v5_arm_configuration(
        ROOT / "config/defense/defense-v5-arms.json",
        catalog=catalog,
        protocol=protocol,
    )
    corpus = build_v5_corpus(protocol, profile=V5Profile.SMOKE)
    suite = execute_v5_controls(
        protocol=protocol,
        evidence_protocol=evidence_protocol,
        corpus=corpus,
        catalog=catalog,
        configuration=configuration,
    )
    groups = tuple(
        execute_v5_control_group(
            group=group,
            protocol=protocol,
            evidence_protocol=evidence_protocol,
            corpus=corpus,
            catalog=catalog,
            configuration=configuration,
            mode=V5RunMode.SAFE_VALIDATION,
        )
        for group in _EXPECTED_GROUPS
    )
    partitions = corpus.partitions
    direct = _score_all_arms_and_evaluate(
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
    arm_results = tuple(
        V5EvaluationResult.model_validate(direct["arm_results"][arm.value])
        for arm in (
            V5Arm.RULES_ONLY,
            V5Arm.ENSEMBLE_NO_GRAPH,
            V5Arm.ENSEMBLE_WITH_GRAPH,
            V5Arm.FULL_SENTINEL,
        )
    )
    return suite, groups, corpus, arm_results


def test_grouped_controls_match_the_existing_one_shot_suite_deterministically(
    executed_group_evidence: _ExecutedEvidence,
) -> None:
    """Splitting jobs may change timings but cannot change any control semantics."""
    one_shot, groups, _corpus, _arm_results = executed_group_evidence
    assembled = assemble_v5_control_suite(groups)

    assert tuple(control.name for control in assembled.controls) == tuple(
        control.name for control in one_shot.controls
    )
    assert [_stable_control(control) for control in assembled.controls] == [
        _stable_control(control) for control in one_shot.controls
    ]


def test_control_group_contract_rejects_missing_duplicate_and_reordered_groups(
    executed_group_evidence: _ExecutedEvidence,
) -> None:
    """The suite assembler must accept exactly the three frozen groups once."""
    _one_shot, groups, _corpus, _arm_results = executed_group_evidence
    with pytest.raises(ValueError, match="exact ordered control groups"):
        assemble_v5_control_suite(groups[:-1])
    with pytest.raises(ValueError, match="exact ordered control groups"):
        assemble_v5_control_suite((groups[0], groups[0], groups[2]))
    with pytest.raises(ValueError, match="exact ordered control groups"):
        assemble_v5_control_suite((groups[1], groups[0], groups[2]))


def test_control_group_digest_and_membership_fail_closed(
    executed_group_evidence: _ExecutedEvidence,
) -> None:
    """A group cannot be relabeled, rebound, or have a control moved into it."""
    _one_shot, groups, _corpus, _arm_results = executed_group_evidence
    label = groups[0]
    with pytest.raises(ValueError, match="control group digest"):
        V5ExecutedControlGroup.model_validate(
            label.model_copy(update={"group_sha256": "0" * 64}).model_dump(mode="json")
        )
    with pytest.raises(ValueError, match="exact controls"):
        V5ExecutedControlGroup.model_validate(
            label.model_copy(
                update={"controls": (*label.controls, groups[1].controls[0])}
            ).model_dump(mode="json")
        )


def test_control_groups_bind_closed_run_mode(
    executed_group_evidence: _ExecutedEvidence,
) -> None:
    """Safe control evidence cannot be relabeled as locked evidence."""
    _one_shot, groups, _corpus, _arm_results = executed_group_evidence
    label = groups[0]
    with pytest.raises(ValueError, match="control group digest|run mode"):
        V5ExecutedControlGroup.model_validate(
            label.model_copy(update={"run_mode": V5RunMode.LOCKED_DEVELOPMENT}).model_dump(
                mode="json"
            )
        )


def test_control_checkpoint_layers_recombine_and_reject_each_layer_mutation(
    executed_group_evidence: _ExecutedEvidence,
) -> None:
    """Deterministic controls and observed timing must be separately authenticated."""
    _one_shot, groups, _corpus, _arm_results = executed_group_evidence
    for group in groups:
        core, observation = _control_group_core_and_observation(group)
        assert _restore_control_group(core=core, observation=observation) == group

        changed_core = dict(core)
        changed_core["support_sha256"] = "0" * 64
        with pytest.raises(ValueError, match="deterministic control group"):
            _restore_control_group(core=changed_core, observation=observation)

        changed_observation = dict(observation)
        changed_observation["observational_group_sha256"] = "0" * 64
        with pytest.raises(ValueError, match="observational control group"):
            _restore_control_group(core=core, observation=changed_observation)


def test_stage_40_50_60_publish_exact_groups_in_predecessor_order(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    executed_group_evidence: _ExecutedEvidence,
) -> None:
    """The three jobs cannot skip, reorder, or combine frozen control groups."""
    _one_shot, groups, corpus, _arm_results = executed_group_evidence
    protocol = load_v5_kaggle_protocol(KAGGLE_CONFIG, root=ROOT)
    dummy = lambda name: iter(  # noqa: E731
        (
            V5CheckpointInput(
                kind="stage_placeholder",
                key=name,
                canonical_bytes=json.dumps({"stage": name}, sort_keys=True).encode(),
            ),
        )
    )
    authorization = _publish(
        output_root=tmp_path / "00",
        stage=V5KaggleStage.AUTHORIZE,
        predecessor=None,
        records=dummy("authorize"),
    )
    corpus_root = tmp_path / "10"
    corpus_manifest = _publish(
        output_root=corpus_root,
        stage=V5KaggleStage.CORPUS,
        predecessor=authorization,
        records=_iter_v5_corpus_records(
            corpus=corpus,
            mode=V5KaggleMode.CAPACITY_VALIDATION,
            development_test_seed=404,
            support_plan_sha256="6" * 64,
        ),
    )
    features = _publish(
        output_root=tmp_path / "20",
        stage=V5KaggleStage.FEATURES,
        predecessor=corpus_manifest,
        records=dummy("features"),
    )
    predecessor = _publish(
        output_root=tmp_path / "30",
        stage=V5KaggleStage.ARMS,
        predecessor=features,
        records=dummy("arms"),
    )
    by_group = {group.group: group for group in groups}

    def frozen_group(**kwargs: object) -> V5ExecutedControlGroup:
        return by_group[kwargs["group"]]

    monkeypatch.setattr(
        "apar.evaluation.v5_staged_evidence.execute_v5_control_group",
        frozen_group,
    )
    loaded: list[V5ExecutedControlGroup] = []
    for stage, expected_group in zip(
        (
            V5KaggleStage.LABEL_SHUFFLE,
            V5KaggleStage.INVARIANCE_CONTROLS,
            V5KaggleStage.SINGLE_CLASS_CONTROLS,
        ),
        _EXPECTED_GROUPS,
        strict=True,
    ):
        capability = _issue_stage_capability(
            protocol=protocol,
            mode=V5KaggleMode.CAPACITY_VALIDATION,
            attempt_receipt_sha256="5" * 64,
            predecessor=predecessor,
        )
        assert capability.stage is stage
        output = tmp_path / stage.value
        predecessor = _publish(
            output_root=output,
            stage=stage,
            predecessor=predecessor,
            records=execute_v5_control_stage(
                root=ROOT,
                capability=capability,
                corpus_checkpoint_root=corpus_root,
            ),
        )
        assert predecessor.record_count == 1
        assert predecessor.observational_record_count == 1
        loaded.append(
            load_v5_control_group_checkpoint(
                checkpoint_root=output,
                limits=protocol.resources,
            )
        )
        assert loaded[-1].group is expected_group
    assert assemble_v5_control_suite(tuple(loaded)).controls


def test_metric_stage_matches_existing_complete_metrics_and_readiness(
    executed_group_evidence: _ExecutedEvidence,
) -> None:
    """Stage boundaries cannot alter metrics, economics, bootstrap, or gates."""
    _suite, groups, _corpus, arm_results = executed_group_evidence
    evidence_protocol = load_v5_evidence_protocol(
        ROOT / "config/defense/defense-v5-evidence.json", root=ROOT
    )
    staged = build_v5_metric_stage_evidence(
        arm_results=arm_results,
        control_groups=groups,
        evidence_protocol=evidence_protocol,
    )
    expected_metrics = tuple(
        evaluate_v5_complete_result(result=result, protocol=evidence_protocol)
        for result in arm_results
    )
    expected_readiness = build_v5_readiness_evidence(
        metrics=expected_metrics[-1], controls=assemble_v5_control_suite(groups)
    )

    assert staged.complete_metrics == expected_metrics
    assert staged.controls == assemble_v5_control_suite(groups)
    assert staged.readiness == expected_readiness
    assert all(item.economics.economics_sha256 for item in staged.complete_metrics)
    assert all(item.bootstrap.bootstrap_sha256 for item in staged.complete_metrics)
    assert all(item.calibration.calibration_sha256 for item in staged.complete_metrics)

    core, observation = _metric_stage_core_and_observation(evidence=staged, arm_results=arm_results)
    assert _restore_metric_stage_evidence(core=core, observation=observation) == staged
    changed_core = dict(core)
    changed_core["deterministic_metric_stage_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="deterministic metric stage"):
        _restore_metric_stage_evidence(core=changed_core, observation=observation)
    changed_observation = dict(observation)
    changed_observation["observational_metric_stage_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="observational metric stage"):
        _restore_metric_stage_evidence(core=core, observation=changed_observation)


def test_final_payload_retains_complete_staged_semantics_and_chain(
    executed_group_evidence: _ExecutedEvidence,
) -> None:
    """Stage 80 must retain all metrics/controls while binding the predecessor chain."""
    _suite, groups, _corpus, arm_results = executed_group_evidence
    evidence_protocol = load_v5_evidence_protocol(
        ROOT / "config/defense/defense-v5-evidence.json", root=ROOT
    )
    controls = assemble_v5_control_suite(groups)
    payload_bytes = build_v5_staged_evidence_payload(
        mode=V5KaggleMode.CAPACITY_VALIDATION,
        run_binding_sha256="a" * 64,
        support_plan=_capacity_support_plan(arm_results),
        chain=_chain(),
        evidence_protocol=evidence_protocol,
        catalog_sha256=SentinelFeatureCatalog.default().catalog_sha256,
        arm_results=arm_results,
        controls=controls,
    )
    payload = V5StagedEvidencePayload.model_validate_json(payload_bytes)
    metric_evidence = build_v5_metric_stage_evidence(
        arm_results=arm_results,
        control_groups=groups,
        evidence_protocol=evidence_protocol,
    )

    assert payload.mode is V5KaggleMode.CAPACITY_VALIDATION
    assert payload.development_test_seed == 404
    assert payload.checkpoint_chain == _chain()
    assert tuple(item.document() for item in payload.complete_metrics) == tuple(
        item.model_dump(mode="json") for item in metric_evidence.complete_metrics
    )
    assert payload.controls.document() == controls.model_dump(mode="json")
    assert payload.readiness == metric_evidence.readiness
    assert payload.deterministic_core.core_sha256
    assert (
        payload.observational_latency.document()["deterministic_core_sha256"]
        == payload.deterministic_core.core_sha256
    )
