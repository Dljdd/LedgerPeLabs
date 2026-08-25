"""Independent offline verification for staged Sentinel v5 checkpoints."""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from apar.evaluation.v5_checkpoint_storage import (
    V5CheckpointInput,
    publish_v5_checkpoint,
)
from apar.evaluation.v5_controls import assemble_v5_control_suite
from apar.evaluation.v5_evidence_protocol import load_v5_evidence_protocol
from apar.evaluation.v5_kaggle_protocol import (
    V5KaggleMode,
    V5KaggleStage,
    load_v5_kaggle_protocol,
)
from apar.evaluation.v5_locked_evidence import (
    V5CheckpointChainBinding,
    V5StagedEvidencePayload,
    build_v5_staged_evidence_payload,
)
from apar.evaluation.v5_staged_evidence import (
    _arm_core_and_observation,
    _control_group_core_and_observation,
    _issue_stage_capability,
    _iter_prepared_partition_records,
    _iter_v5_corpus_records,
    _metric_stage_core_and_observation,
    _prepare_partition,
    build_v5_metric_stage_evidence,
    execute_v5_authorization_stage,
)
from apar.features.sentinel import SentinelFeatureCatalog
from apar.v5_kaggle_independent_verifier import (
    V5KaggleIndependentVerificationError,
    _verify_v5_kaggle_test_fixture,
    verify_v5_kaggle_evidence,
    verify_v5_kaggle_prefix,
)
from tests.evaluation import test_defense_v5_staged_controls as staged_fixtures

ROOT = Path(__file__).resolve().parents[2]
VERIFIER = ROOT / "src/apar/v5_kaggle_independent_verifier.py"
CLI = ROOT / "scripts/verify_defense_v5_kaggle_evidence.py"


def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text())
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
    return names


def test_independent_verifier_module_and_import_boundary() -> None:
    assert importlib.util.find_spec("apar.v5_kaggle_independent_verifier") is not None
    forbidden = {
        "apar.evaluation.v5_staged_evidence",
        "apar.evaluation.v5_checkpoint_storage",
        "apar.evaluation.v5_controls",
        "apar.evaluation.v5_metrics",
        "apar.evaluation.v5_evaluation",
        "apar.evaluation.v5_runner",
        "apar.evaluation.v5_population",
        "apar.simulator",
        "apar.rails",
        "apar.trust",
    }
    assert _imports(VERIFIER).isdisjoint(forbidden)


def test_public_prefix_verifier_accepts_nonexecuting_authorization_stage(
    tmp_path: Path,
) -> None:
    protocol = load_v5_kaggle_protocol(
        ROOT / "config/defense/defense-v5-kaggle-recovery.json", root=ROOT
    )
    attempt = "5" * 64
    capability = _issue_stage_capability(
        protocol=protocol,
        mode=V5KaggleMode.CAPACITY_VALIDATION,
        attempt_receipt_sha256=attempt,
        predecessor=None,
    )
    base_environment = staged_fixtures._environment()
    notebook_sha = hashlib.sha256(
        (ROOT / "kaggle/defense_v5/00_authorize.ipynb").read_bytes()
    ).hexdigest()
    environment = type(base_environment).bind(
        provider="kaggle",
        image=base_environment.image,
        image_sha256=base_environment.image_sha256,
        python_version="3.12.5",
        architecture="x86_64",
        cpu_count=base_environment.cpu_count,
        dependency_manifest_sha256=base_environment.dependency_manifest_sha256,
        source_archive_sha256=base_environment.source_archive_sha256,
        notebook_sha256=notebook_sha,
        internet_enabled=False,
        accelerator="none",
        file_fsync_supported=True,
        directory_fsync_supported=True,
        hardlink_no_replace_supported=True,
    )
    checkpoint = tmp_path / "chain" / V5KaggleStage.AUTHORIZE.value
    publish_v5_checkpoint(
        output_root=checkpoint,
        stage=V5KaggleStage.AUTHORIZE,
        run_binding_sha256=protocol.run_binding_sha256(
            V5KaggleMode.CAPACITY_VALIDATION
        ),
        attempt_receipt_sha256=attempt,
        predecessor=None,
        records=execute_v5_authorization_stage(root=ROOT, capability=capability),
        environment=environment,
        observation=staged_fixtures._observation().model_copy(
            update={"environment": environment}
        ),
        limits=protocol.resources,
    )
    report = verify_v5_kaggle_prefix(
        root=ROOT,
        checkpoint_roots=(checkpoint,),
        expected_mode="kaggle_capacity_validation",
    )
    assert report.verified_stage_ids == (V5KaggleStage.AUTHORIZE.value,)
    completed = subprocess.run(
        [
            sys.executable,
            str(CLI),
            "--root",
            str(ROOT),
            "--mode",
            "kaggle_capacity_validation",
            "--chain-root",
            str(tmp_path / "chain"),
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr


def _digest(document: object) -> str:
    return hashlib.sha256(
        json.dumps(document, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


@pytest.fixture(scope="module")
def complete_safe_evidence() -> object:
    return staged_fixtures.executed_group_evidence.__wrapped__()


def _materialize_complete_chain(
    tmp_path: Path, evidence: object, *, semantic_mutation: str | None = None
) -> Path:
    _suite, groups, corpus, arm_results = evidence
    protocol = load_v5_kaggle_protocol(
        ROOT / "config/defense/defense-v5-kaggle-recovery.json", root=ROOT
    )
    run_binding = protocol.run_binding_sha256(V5KaggleMode.CAPACITY_VALIDATION)
    attempt = "5" * 64
    predecessor = None
    chain_root = tmp_path / "chain"
    chain_root.mkdir(parents=True)
    manifests = []
    support_plan = staged_fixtures._capacity_support_plan(arm_results)
    evidence_protocol = load_v5_evidence_protocol(
        ROOT / "config/defense/defense-v5-evidence.json", root=ROOT
    )
    controls = assemble_v5_control_suite(groups)
    metric_evidence = build_v5_metric_stage_evidence(
        arm_results=arm_results,
        control_groups=groups,
        evidence_protocol=evidence_protocol,
    )
    metric_core, metric_observation = _metric_stage_core_and_observation(
        evidence=metric_evidence, arm_results=arm_results
    )
    group_by_stage = {
        V5KaggleStage.LABEL_SHUFFLE: groups[0],
        V5KaggleStage.INVARIANCE_CONTROLS: groups[1],
        V5KaggleStage.SINGLE_CLASS_CONTROLS: groups[2],
    }
    for stage in tuple(V5KaggleStage)[:-1]:
        if stage is V5KaggleStage.AUTHORIZE:
            capability = _issue_stage_capability(
                protocol=protocol,
                mode=V5KaggleMode.CAPACITY_VALIDATION,
                attempt_receipt_sha256=attempt,
                predecessor=None,
            )
            records = execute_v5_authorization_stage(
                root=ROOT, capability=capability
            )
        elif stage is V5KaggleStage.CORPUS:
            records = _iter_v5_corpus_records(
                corpus=corpus,
                mode=V5KaggleMode.CAPACITY_VALIDATION,
                development_test_seed=404,
                support_plan_sha256=support_plan.support_plan_sha256,
            )
        elif stage is V5KaggleStage.FEATURES:
            catalog = SentinelFeatureCatalog.default()
            partition_order = (
                "train",
                "calibration",
                "threshold",
                "development_test",
            )
            feature_records = [
                V5CheckpointInput(
                    kind="feature_header",
                    key="features",
                    canonical_bytes=json.dumps(
                        {
                            "schema_version": "apar-sentinel-v5-kaggle-features/1",
                            "partition_order": partition_order,
                            "catalog_sha256": catalog.catalog_sha256,
                            "feature_names": catalog.feature_names,
                            "corpus_sha256": corpus.corpus_sha256,
                        },
                        sort_keys=True,
                        separators=(",", ":"),
                    ).encode(),
                )
            ]
            for partition_name in partition_order:
                feature_records.extend(
                    _iter_prepared_partition_records(
                        _prepare_partition(
                            partition_name=partition_name,
                            partition=corpus.partitions[partition_name],
                            catalog=catalog,
                        )
                    )
                )
            records = tuple(feature_records)
        elif stage is V5KaggleStage.ARMS:
            cores_and_observations = tuple(
                _arm_core_and_observation(result) for result in arm_results
            )
            records = (
                V5CheckpointInput(
                    kind="arm_header",
                    key="arms",
                    canonical_bytes=json.dumps(
                        {
                            "schema_version": "apar-sentinel-v5-kaggle-arms/1",
                            "arm_order": [item.arm for item in arm_results],
                            "support_event_ids": [
                                row.support.event_id for row in arm_results[0].row_evidence
                            ],
                            "support_sha256": arm_results[0].support_sha256,
                            "deterministic_result_sha256": [
                                core["deterministic_result_sha256"]
                                for core, _observation in cores_and_observations
                            ],
                        },
                        sort_keys=True,
                        separators=(",", ":"),
                    ).encode(),
                ),
                *tuple(
                    V5CheckpointInput(
                        kind="arm_result",
                        key=result.arm,
                        canonical_bytes=json.dumps(
                            core, sort_keys=True, separators=(",", ":")
                        ).encode(),
                    )
                    for result, (core, _observation) in zip(
                        arm_results, cores_and_observations, strict=True
                    )
                ),
                *tuple(
                    V5CheckpointInput(
                        kind="arm_latency",
                        key=result.arm,
                        canonical_bytes=json.dumps(
                            observation, sort_keys=True, separators=(",", ":")
                        ).encode(),
                        layer="observational",
                    )
                    for result, (_core, observation) in zip(
                        arm_results, cores_and_observations, strict=True
                    )
                ),
            )
        elif stage in group_by_stage:
            group = group_by_stage[stage]
            core, observation = _control_group_core_and_observation(group)
            records = (
                V5CheckpointInput(
                    kind="control_group",
                    key=group.group,
                    canonical_bytes=json.dumps(
                        core, sort_keys=True, separators=(",", ":")
                    ).encode(),
                ),
                V5CheckpointInput(
                    kind="control_observation",
                    key=group.group,
                    canonical_bytes=json.dumps(
                        observation, sort_keys=True, separators=(",", ":")
                    ).encode(),
                    layer="observational",
                ),
            )
        elif stage is V5KaggleStage.METRICS:
            records = (
                V5CheckpointInput(
                    kind="metric_evidence",
                    key="complete",
                    canonical_bytes=json.dumps(
                        metric_core, sort_keys=True, separators=(",", ":")
                    ).encode(),
                ),
                V5CheckpointInput(
                    kind="metric_observation",
                    key="complete",
                    canonical_bytes=json.dumps(
                        metric_observation, sort_keys=True, separators=(",", ":")
                    ).encode(),
                    layer="observational",
                ),
            )
        else:
            raise AssertionError(f"unexpected fixture stage: {stage}")
        records = tuple(records)
        if semantic_mutation == "corpus" and stage is V5KaggleStage.CORPUS:
            mutated = list(records)
            index = next(
                offset for offset, item in enumerate(mutated) if item.kind == "decision_row"
            )
            wrapper = json.loads(mutated[index].canonical_bytes)
            wrapper["decision"]["family"] = "mutated-family"
            mutated[index] = V5CheckpointInput(
                kind=mutated[index].kind,
                key=mutated[index].key,
                canonical_bytes=json.dumps(
                    wrapper, sort_keys=True, separators=(",", ":")
                ).encode(),
            )
            records = tuple(mutated)
        if semantic_mutation == "feature" and stage is V5KaggleStage.FEATURES:
            mutated = list(records)
            index = next(
                offset
                for offset, item in enumerate(mutated)
                if item.kind == "feature_matrix"
            )
            content = bytearray(mutated[index].canonical_bytes)
            content[0] ^= 1
            mutated[index] = V5CheckpointInput(
                kind=mutated[index].kind,
                key=mutated[index].key,
                canonical_bytes=bytes(content),
            )
            records = tuple(mutated)
        if semantic_mutation == "arm" and stage is V5KaggleStage.ARMS:
            mutated = list(records)
            index = next(
                offset for offset, item in enumerate(mutated) if item.kind == "arm_result"
            )
            core = json.loads(mutated[index].canonical_bytes)
            core["support_total"] += 1
            mutated[index] = V5CheckpointInput(
                kind=mutated[index].kind,
                key=mutated[index].key,
                canonical_bytes=json.dumps(
                    core, sort_keys=True, separators=(",", ":")
                ).encode(),
            )
            records = tuple(mutated)
        predecessor = publish_v5_checkpoint(
            output_root=chain_root / stage.value,
            stage=stage,
            run_binding_sha256=run_binding,
            attempt_receipt_sha256=attempt,
            predecessor=predecessor,
            records=records,
            environment=staged_fixtures._environment(),
            observation=staged_fixtures._observation(),
            limits=protocol.resources,
        )
        manifests.append(predecessor)
    chain_values = {
        "schema_version": "apar-sentinel-v5-checkpoint-chain/1",
        "attempt_receipt_sha256": attempt,
        "predecessor_stage_manifest_sha256": tuple(
            (item.stage.value, item.manifest_sha256) for item in manifests
        ),
    }
    chain_values["predecessor_chain_root_sha256"] = _digest(chain_values)
    chain = V5CheckpointChainBinding.model_validate(chain_values)
    payload_bytes = build_v5_staged_evidence_payload(
        mode=V5KaggleMode.CAPACITY_VALIDATION,
        run_binding_sha256=run_binding,
        support_plan=support_plan,
        chain=chain,
        evidence_protocol=evidence_protocol,
        catalog_sha256=SentinelFeatureCatalog.default().catalog_sha256,
        arm_results=arm_results,
        controls=controls,
    )
    payload = V5StagedEvidencePayload.model_validate_json(payload_bytes)
    core = {
        "schema_version": "apar-sentinel-v5-kaggle-final-core/1",
        "mode": V5KaggleMode.CAPACITY_VALIDATION,
        "run_binding_sha256": run_binding,
        "support_plan_sha256": payload.support_plan.support_plan_sha256,
        "deterministic_core_sha256": payload.deterministic_core.core_sha256,
    }
    core["final_core_sha256"] = _digest(core)
    publish_v5_checkpoint(
        output_root=chain_root / V5KaggleStage.FINALIZE.value,
        stage=V5KaggleStage.FINALIZE,
        run_binding_sha256=run_binding,
        attempt_receipt_sha256=attempt,
        predecessor=predecessor,
        records=(
            V5CheckpointInput(
                kind="final_core",
                key="complete",
                canonical_bytes=json.dumps(core, sort_keys=True, separators=(",", ":")).encode(),
            ),
            V5CheckpointInput(
                kind="final_payload",
                key="complete",
                canonical_bytes=payload_bytes,
                layer="observational",
            ),
        ),
        environment=staged_fixtures._environment(),
        observation=staged_fixtures._observation(),
        limits=protocol.resources,
    )
    return chain_root


def test_independent_verifier_replays_complete_safe_chain_and_rejects_tamper(
    tmp_path: Path, complete_safe_evidence: object
) -> None:
    chain_root = _materialize_complete_chain(tmp_path, complete_safe_evidence)
    roots = tuple(chain_root / stage.value for stage in V5KaggleStage)
    report = _verify_v5_kaggle_test_fixture(
        root=ROOT,
        checkpoint_roots=roots[:-1],
        final_root=roots[-1],
    )
    assert report.valid is True
    assert report.verified_stage_ids == tuple(stage.value for stage in V5KaggleStage)

    final_chunk = roots[-1] / "observational-chunks" / "part-0000.bin"
    content = bytearray(final_chunk.read_bytes())
    content[-1] ^= 1
    final_chunk.write_bytes(content)
    with pytest.raises(
        V5KaggleIndependentVerificationError,
        match="chunk digest|compressed stream",
    ):
        _verify_v5_kaggle_test_fixture(
            root=ROOT,
            checkpoint_roots=roots[:-1],
            final_root=roots[-1],
        )

    pristine = _materialize_complete_chain(tmp_path / "pristine", complete_safe_evidence)
    pristine_roots = tuple(pristine / stage.value for stage in V5KaggleStage)
    with pytest.raises(
        V5KaggleIndependentVerificationError,
        match="production support plan",
    ):
        verify_v5_kaggle_evidence(
            root=ROOT,
            checkpoint_roots=pristine_roots[:-1],
            final_root=pristine_roots[-1],
            expected_mode="kaggle_capacity_validation",
        )


@pytest.mark.parametrize("layer", ("corpus", "feature", "arm"))
def test_semantically_mutated_stage_stream_is_rejected_even_with_valid_storage(
    tmp_path: Path,
    complete_safe_evidence: object,
    layer: str,
) -> None:
    chain_root = _materialize_complete_chain(
        tmp_path,
        complete_safe_evidence,
        semantic_mutation=layer,
    )
    roots = tuple(chain_root / stage.value for stage in V5KaggleStage)
    with pytest.raises(V5KaggleIndependentVerificationError):
        _verify_v5_kaggle_test_fixture(
            root=ROOT,
            checkpoint_roots=roots[:-1],
            final_root=roots[-1],
        )


def test_verifier_cli_failure_is_deterministic_across_hash_seeds(
    tmp_path: Path, complete_safe_evidence: object
) -> None:
    chain_root = _materialize_complete_chain(tmp_path, complete_safe_evidence)
    outputs = []
    for hash_seed in ("1", "777"):
        environment = dict(os.environ)
        environment["PYTHONHASHSEED"] = hash_seed
        completed = subprocess.run(
            [
                sys.executable,
                str(CLI),
                "--root",
                str(ROOT),
                "--mode",
                "kaggle_capacity_validation",
                "--chain-root",
                str(chain_root),
            ],
            cwd=ROOT,
            env=environment,
            check=False,
            capture_output=True,
            text=True,
        )
        assert completed.returncode == 1
        outputs.append(completed.stderr)
    assert outputs[0] == outputs[1]


def test_chain_topology_environment_and_cross_stage_mutations_fail_closed(
    tmp_path: Path, complete_safe_evidence: object
) -> None:
    master = _materialize_complete_chain(tmp_path, complete_safe_evidence)
    master_roots = tuple(master / stage.value for stage in V5KaggleStage)

    with pytest.raises(V5KaggleIndependentVerificationError):
        _verify_v5_kaggle_test_fixture(
            root=ROOT,
            checkpoint_roots=master_roots[:3] + master_roots[4:-1],
            final_root=master_roots[-1],
        )
    reordered = list(master_roots[:-1])
    reordered[3], reordered[4] = reordered[4], reordered[3]
    with pytest.raises(V5KaggleIndependentVerificationError):
        _verify_v5_kaggle_test_fixture(
            root=ROOT,
            checkpoint_roots=reordered,
            final_root=master_roots[-1],
        )
    with pytest.raises(V5KaggleIndependentVerificationError):
        verify_v5_kaggle_evidence(
            root=ROOT,
            checkpoint_roots=master_roots[:-1],
            final_root=master_roots[-1],
            expected_mode="kaggle_locked_successor",
        )

    environment_copy = tmp_path / "environment-copy"
    shutil.copytree(master, environment_copy)
    observation_path = environment_copy / V5KaggleStage.ARMS.value / "observational.json"
    observation = json.loads(observation_path.read_bytes())
    observation["environment"]["image"] = "mutated-image"
    observation_path.write_bytes(
        json.dumps(observation, sort_keys=True, separators=(",", ":")).encode()
    )
    copied_roots = tuple(environment_copy / stage.value for stage in V5KaggleStage)
    with pytest.raises(V5KaggleIndependentVerificationError):
        _verify_v5_kaggle_test_fixture(
            root=ROOT,
            checkpoint_roots=copied_roots[:-1],
            final_root=copied_roots[-1],
        )

    substitution = tmp_path / "substitution-copy"
    shutil.copytree(master, substitution)
    source = (
        substitution / V5KaggleStage.LABEL_SHUFFLE.value / "observational-chunks" / "part-0000.bin"
    )
    target = (
        substitution
        / V5KaggleStage.INVARIANCE_CONTROLS.value
        / "observational-chunks"
        / "part-0000.bin"
    )
    target.write_bytes(source.read_bytes())
    substituted_roots = tuple(substitution / stage.value for stage in V5KaggleStage)
    with pytest.raises(V5KaggleIndependentVerificationError):
        _verify_v5_kaggle_test_fixture(
            root=ROOT,
            checkpoint_roots=substituted_roots[:-1],
            final_root=substituted_roots[-1],
        )
