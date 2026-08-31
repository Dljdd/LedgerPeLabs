"""Typed stage capability and payload contracts for Sentinel v5 checkpoints."""

from __future__ import annotations

import gc
import json
import weakref
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest

from apar.evaluation import v5_kaggle_rescue as rescue
from apar.evaluation import v5_staged_evidence as staged
from apar.evaluation.v5_checkpoint_storage import (
    V5CheckpointInput,
    V5CheckpointObservation,
    iter_v5_checkpoint_records,
    publish_v5_checkpoint,
)
from apar.evaluation.v5_evaluation import (
    V5Arm,
    V5EvaluationResult,
    load_v5_arm_configuration,
)
from apar.evaluation.v5_evidence_layers import _stable_complete_metrics
from apar.evaluation.v5_evidence_protocol import load_v5_evidence_protocol
from apar.evaluation.v5_kaggle_protocol import (
    V5KaggleEnvironmentBinding,
    V5KaggleMode,
    V5KaggleStage,
    build_v5_kaggle_support_plan,
    load_v5_kaggle_protocol,
)
from apar.evaluation.v5_metrics import evaluate_v5_complete_result
from apar.evaluation.v5_population import V5Corpus, build_v5_corpus
from apar.evaluation.v5_protocol import V5Profile, load_v5_development_protocol
from apar.features.sentinel import SentinelFeatureCatalog, build_sentinel_features
from scripts.run_defense_v5_development import _score_all_arms_and_evaluate

_CURRENT_ARMS = (
    V5Arm.RULES_ONLY,
    V5Arm.ENSEMBLE_NO_GRAPH,
    V5Arm.ENSEMBLE_WITH_GRAPH,
    V5Arm.FULL_SENTINEL,
)

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "config/defense/defense-v5-kaggle-recovery.json"


def _protocol() -> object:
    return load_v5_kaggle_protocol(CONFIG, root=ROOT)


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


def test_arm_section_records_split_before_checkpoint_record_limit() -> None:
    """Large arm evidence must be partitioned before V5CheckpointInput validation."""
    assert hasattr(staged, "_iter_bounded_arm_section_records"), (
        "arm checkpoint sections are not yet bounded"
    )
    items = tuple(
        {"event_id": f"event-{index}", "value": "x" * 180}
        for index in range(6)
    )

    records = tuple(
        staged._iter_bounded_arm_section_records(
            kind="arm_result_rows",
            arm="full_sentinel",
            section="row_evidence",
            items=items,
            max_record_bytes=1_024,
        )
    )

    assert len(records) > 1
    assert all(len(record.canonical_bytes) <= 1_024 for record in records)
    documents = [json.loads(record.canonical_bytes) for record in records]
    assert [document["index"] for document in documents] == list(range(len(records)))
    assert [document["start"] for document in documents] == [
        sum(len(previous["items"]) for previous in documents[:index])
        for index in range(len(documents))
    ]
    assert [item for document in documents for item in document["items"]] == list(items)


@pytest.fixture(scope="module")
def safe_smoke_corpus() -> V5Corpus:
    protocol = load_v5_development_protocol(
        ROOT / "config/defense/defense-v5-development.json"
    )
    safe = protocol.model_copy(
        update={
            "seeds": protocol.seeds.model_copy(update={"development_test": 404}),
            "protocol_sha256": "",
        }
    )
    return build_v5_corpus(safe, profile=V5Profile.SMOKE)


def _publish_stage(
    *,
    output_root: Path,
    stage: V5KaggleStage,
    records: Iterator[V5CheckpointInput],
    predecessor: object | None,
) -> object:
    protocol = _protocol()
    return publish_v5_checkpoint(
        output_root=output_root,
        stage=stage,
        run_binding_sha256=protocol.run_binding_sha256(
            V5KaggleMode.CAPACITY_VALIDATION
        ),
        attempt_receipt_sha256="5" * 64,
        predecessor=predecessor,
        records=records,
        environment=_environment(),
        observation=_observation(),
        limits=protocol.resources,
    )


def _build_feature_chain(
    *, tmp_path: Path, corpus: V5Corpus
) -> tuple[Path, object]:
    protocol = _protocol()
    authorization_capability = staged._issue_stage_capability(
        protocol=protocol,
        mode=V5KaggleMode.CAPACITY_VALIDATION,
        attempt_receipt_sha256="5" * 64,
        predecessor=None,
    )
    authorization_manifest = _publish_stage(
        output_root=tmp_path / "authorization",
        stage=V5KaggleStage.AUTHORIZE,
        records=staged.execute_v5_authorization_stage(
            root=ROOT, capability=authorization_capability
        ),
        predecessor=None,
    )
    corpus_root = tmp_path / "corpus"
    corpus_manifest = _publish_stage(
        output_root=corpus_root,
        stage=V5KaggleStage.CORPUS,
        records=staged._iter_v5_corpus_records(
            corpus=corpus,
            mode=V5KaggleMode.CAPACITY_VALIDATION,
            development_test_seed=404,
            support_plan_sha256="6" * 64,
        ),
        predecessor=authorization_manifest,
    )
    feature_capability = staged._issue_stage_capability(
        protocol=protocol,
        mode=V5KaggleMode.CAPACITY_VALIDATION,
        attempt_receipt_sha256="5" * 64,
        predecessor=corpus_manifest,
    )
    feature_root = tmp_path / "features"
    feature_manifest = _publish_stage(
        output_root=feature_root,
        stage=V5KaggleStage.FEATURES,
        records=staged.execute_v5_feature_stage(
            root=ROOT,
            capability=feature_capability,
            corpus_checkpoint_root=corpus_root,
        ),
        predecessor=corpus_manifest,
    )
    return feature_root, feature_manifest


def _deterministic_arm_document(result: V5EvaluationResult) -> dict[str, object]:
    document = result.model_dump(mode="json")
    for field in (
        "p50_latency_ms",
        "p95_latency_ms",
        "p99_latency_ms",
        "score_sha256",
        "result_sha256",
    ):
        document.pop(field)
    rows = document["row_evidence"]
    assert isinstance(rows, list)
    for row in rows:
        assert isinstance(row, dict)
        row.pop("latency_ms")
        row.pop("row_output_sha256")
    return document


def test_capacity_support_plan_is_production_sized_without_population_execution() -> None:
    """Capacity rehearsal support must equal production support without using seed 2404."""
    capacity = build_v5_kaggle_support_plan(
        root=ROOT,
        protocol=_protocol(),
        mode=V5KaggleMode.CAPACITY_VALIDATION,
    )
    locked = build_v5_kaggle_support_plan(
        root=ROOT,
        protocol=_protocol(),
        mode=V5KaggleMode.LOCKED_SUCCESSOR,
    )

    assert capacity.profile == "production"
    assert capacity.mode is V5KaggleMode.CAPACITY_VALIDATION
    assert locked.mode is V5KaggleMode.LOCKED_SUCCESSOR
    assert capacity.support_plan_sha256 != locked.support_plan_sha256
    assert [item.model_dump(mode="json") for item in capacity.partitions] == [
        item.model_dump(mode="json") for item in locked.partitions
    ]
    assert tuple(item.total_rows for item in capacity.partitions) == (
        25_800,
        25_800,
        25_800,
        63_300,
    )
    assert capacity.retained_execution_artifacts == 2_523
    assert capacity.retained_execution_payload_estimate_bytes == 608_083_936


def test_capacity_support_plan_batches_from_executed_base_decisions() -> None:
    """Artifact planning must use the actual 24 decisions emitted by base traffic."""
    from apar.evaluation.v5_population import _execute_legitimate_traffic

    rows, executions = _execute_legitimate_traffic(
        partition_name="train",
        partition_seed=404,
        requested_decisions=24,
    )
    assert len(rows) == 24
    assert tuple((manifest.rail, len(manifest.lineage)) for manifest in executions) == (
        ("card", 12),
        ("a2a", 10),
        ("agentic", 2),
    )

    capacity = build_v5_kaggle_support_plan(
        root=ROOT,
        protocol=_protocol(),
        mode=V5KaggleMode.CAPACITY_VALIDATION,
    )

    assert tuple(item.execution_artifacts for item in capacity.partitions) == (
        533,
        533,
        533,
        924,
    )
    assert capacity.retained_execution_artifacts == 2_523


def test_support_plan_label_split_matches_executed_ground_truth(
    safe_smoke_corpus: V5Corpus,
) -> None:
    """The static production plan must count benign controls emitted by attack campaigns."""
    protocol = load_v5_development_protocol(
        ROOT / "config/defense/defense-v5-development.json"
    )
    support = build_v5_kaggle_support_plan(
        root=ROOT,
        protocol=_protocol(),
        mode=V5KaggleMode.CAPACITY_VALIDATION,
    ).partitions[-1]
    smoke = safe_smoke_corpus.partitions["development_test"]

    expected_fraud: dict[str, int] = {}
    expected_campaign_controls = 0
    for family, production_campaigns in sorted(
        protocol.production_dev_test_campaigns_per_family.items()
    ):
        smoke_campaigns = protocol.smoke_profile.campaigns_per_family[family]
        family_rows = tuple(row for row in smoke.decisions if row.family == family)
        fraud_per_campaign = sum(row.is_fraud for row in family_rows) // smoke_campaigns
        benign_per_campaign = sum(not row.is_fraud for row in family_rows) // smoke_campaigns
        expected_fraud[family] = fraud_per_campaign * production_campaigns
        expected_campaign_controls += benign_per_campaign * production_campaigns

    assert dict(support.fraud_rows_by_family) == expected_fraud
    assert support.legitimate_rows == (
        protocol.production_dev_test_legitimate + expected_campaign_controls
    )


def test_support_mismatch_reports_exact_observed_and_expected_counts(
    safe_smoke_corpus: V5Corpus,
) -> None:
    """A rejected staged corpus must identify the exact immutable support delta."""
    support = build_v5_kaggle_support_plan(
        root=ROOT,
        protocol=_protocol(),
        mode=V5KaggleMode.CAPACITY_VALIDATION,
    )

    with pytest.raises(ValueError, match='"partition":"train"') as raised:
        staged._validate_corpus_support(
            corpus=safe_smoke_corpus,
            support_plan=support,
        )

    assert '"observed"' in str(raised.value)
    assert '"expected"' in str(raised.value)
    assert '"execution_artifacts"' in str(raised.value)


def test_authorization_capability_emits_one_canonical_closed_record() -> None:
    """Stage 00 must bind mode, support, source, recovery, and attempt authority."""
    protocol = _protocol()
    capability = staged._issue_stage_capability(
        protocol=protocol,
        mode=V5KaggleMode.CAPACITY_VALIDATION,
        attempt_receipt_sha256="5" * 64,
        predecessor=None,
    )
    records = tuple(
        staged.execute_v5_authorization_stage(root=ROOT, capability=capability)
    )

    assert len(records) == 1
    record = records[0]
    assert record.kind == "authorization"
    assert record.key == "00_authorize"
    document = json.loads(record.canonical_bytes)
    assert record.canonical_bytes == json.dumps(
        document,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    assert document["schema_version"] == (
        "apar-sentinel-v5-kaggle-authorization/1"
    )
    assert document["stage"] == "00_authorize"
    assert document["mode"] == "kaggle_capacity_validation"
    assert document["development_test_seed"] == 404
    assert document["profile"] == "production"
    assert document["attempt_receipt_sha256"] == "5" * 64
    assert document["run_binding_sha256"] == capability.run_binding_sha256
    assert document["support_plan"]["retained_execution_artifacts"] == 2_523
    assert document["recovery"]["retry_permitted"] is False
    assert set(document).isdisjoint(
        {"labels", "probabilities", "actions", "metrics", "readiness"}
    )


def test_forged_capability_fails_before_authorization_evidence_is_read(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A constructed capability cannot reach even the non-executing Stage-00 loader."""
    reached = False

    def forbidden(*args: object, **kwargs: object) -> object:
        nonlocal reached
        reached = True
        raise AssertionError("protocol loader reached")

    monkeypatch.setattr(staged, "load_v5_kaggle_protocol", forbidden)
    forged = staged.V5StageCapability(
        stage=V5KaggleStage.AUTHORIZE,
        mode=V5KaggleMode.CAPACITY_VALIDATION,
        run_binding_sha256="4" * 64,
        attempt_receipt_sha256="5" * 64,
        predecessor_manifest_sha256=None,
        seal=object(),
    )
    with pytest.raises(PermissionError, match="capability"):
        tuple(staged.execute_v5_authorization_stage(root=ROOT, capability=forged))
    assert reached is False


def test_capability_stage_is_derived_from_predecessor() -> None:
    """Capability issuance cannot select or skip a stage."""
    protocol = _protocol()
    authorization_capability = staged._issue_stage_capability(
        protocol=protocol,
        mode=V5KaggleMode.CAPACITY_VALIDATION,
        attempt_receipt_sha256="5" * 64,
        predecessor=None,
    )
    assert authorization_capability.stage is V5KaggleStage.AUTHORIZE


def test_authorization_modes_bind_different_seed_and_run_documents() -> None:
    """A capacity authorization cannot be relabeled as the locked successor."""
    protocol = _protocol()
    safe_capability = staged._issue_stage_capability(
        protocol=protocol,
        mode=V5KaggleMode.CAPACITY_VALIDATION,
        attempt_receipt_sha256="5" * 64,
        predecessor=None,
    )
    locked_capability = staged._issue_stage_capability(
        protocol=protocol,
        mode=V5KaggleMode.LOCKED_SUCCESSOR,
        attempt_receipt_sha256="6" * 64,
        predecessor=None,
    )
    safe = tuple(
        staged.execute_v5_authorization_stage(
            root=ROOT, capability=safe_capability
        )
    )[0]
    locked = tuple(
        staged.execute_v5_authorization_stage(
            root=ROOT, capability=locked_capability
        )
    )[0]
    safe_document = json.loads(safe.canonical_bytes)
    locked_document = json.loads(locked.canonical_bytes)

    assert safe_capability.run_binding_sha256 != locked_capability.run_binding_sha256
    assert safe_document["development_test_seed"] == 404
    assert locked_document["development_test_seed"] == 2404
    assert safe_document["support_plan"]["support_plan_sha256"] != (
        locked_document["support_plan"]["support_plan_sha256"]
    )


def test_stage_protocol_copy_uses_production_profile_and_only_the_closed_seed() -> None:
    """Stage 10 cannot accept a caller-selected seed or a smoke production run."""
    protocol = _protocol()
    safe, safe_profile = staged._development_protocol_for_stage(
        root=ROOT,
        protocol=protocol,
        mode=V5KaggleMode.CAPACITY_VALIDATION,
    )
    locked, locked_profile = staged._development_protocol_for_stage(
        root=ROOT,
        protocol=protocol,
        mode=V5KaggleMode.LOCKED_SUCCESSOR,
    )

    assert safe_profile is V5Profile.PRODUCTION
    assert safe.seeds.development_test == 404
    assert locked_profile is V5Profile.PRODUCTION
    assert locked.seeds.development_test == 2404
    assert safe.seeds.train == locked.seeds.train == 101
    assert safe.seeds.calibration == locked.seeds.calibration == 202
    assert safe.seeds.threshold == locked.seeds.threshold == 303


def test_corpus_checkpoint_round_trips_real_execution_evidence(
    tmp_path: Path,
    safe_smoke_corpus: V5Corpus,
) -> None:
    """Rows reloaded from a checkpoint remain bound to real event, ledger, and trust facts."""
    records = staged._iter_v5_corpus_records(
        corpus=safe_smoke_corpus,
        mode=V5KaggleMode.CAPACITY_VALIDATION,
        development_test_seed=404,
        support_plan_sha256="6" * 64,
    )
    output = tmp_path / "corpus"
    _publish_stage(
        output_root=output,
        stage=V5KaggleStage.AUTHORIZE,
        records=records,
        predecessor=None,
    )
    loaded = staged.load_v5_corpus_checkpoint(
        checkpoint_root=output,
        limits=_protocol().resources,
    )

    assert loaded == safe_smoke_corpus
    for partition in loaded.partitions.values():
        assert partition.decisions
        assert partition.executions
        manifest_by_digest = {
            item.evidence_sha256: item for item in partition.executions
        }
        assert any(manifest.ledger_postings for manifest in partition.executions)
        for row in partition.decisions:
            manifest = manifest_by_digest[row.execution_evidence_sha256]
            assert row.source_event_id in {
                link.event_id for link in manifest.lineage
            }
            assert row.source_command_id in {
                link.command_id for link in manifest.lineage
            }
            if row.rail == "agentic":
                assert manifest.trust_records


def test_corpus_checkpoint_rejects_source_lineage_tampering(
    tmp_path: Path,
    safe_smoke_corpus: V5Corpus,
) -> None:
    """Changing a projected source command cannot survive checkpoint reload."""
    records = list(
        staged._iter_v5_corpus_records(
            corpus=safe_smoke_corpus,
            mode=V5KaggleMode.CAPACITY_VALIDATION,
            development_test_seed=404,
            support_plan_sha256="6" * 64,
        )
    )
    index = next(
        position
        for position, record in enumerate(records)
        if record.kind == "decision_row"
    )
    document = json.loads(records[index].canonical_bytes)
    document["decision"]["source_command_id"] = "tampered-command"
    records[index] = V5CheckpointInput(
        kind=records[index].kind,
        key=records[index].key,
        canonical_bytes=json.dumps(
            document, sort_keys=True, separators=(",", ":")
        ).encode(),
    )
    output = tmp_path / "tampered-corpus"
    _publish_stage(
        output_root=output,
        stage=V5KaggleStage.AUTHORIZE,
        records=iter(records),
        predecessor=None,
    )
    with pytest.raises(ValueError, match="lineage|source event"):
        staged.load_v5_corpus_checkpoint(
            checkpoint_root=output,
            limits=_protocol().resources,
        )


def test_feature_checkpoint_rebuilds_causal_matrices_and_trust_order(
    tmp_path: Path,
    safe_smoke_corpus: V5Corpus,
) -> None:
    """Stage 20 must derive features from checkpointed rows, never direct labels."""
    protocol = _protocol()
    feature_root, feature_manifest = _build_feature_chain(
        tmp_path=tmp_path, corpus=safe_smoke_corpus
    )
    prepared = staged.load_v5_feature_checkpoint(
        checkpoint_root=feature_root,
        limits=protocol.resources,
    )

    assert feature_manifest.record_count > 0
    assert tuple(prepared) == (
        "train",
        "calibration",
        "threshold",
        "development_test",
    )
    catalog = SentinelFeatureCatalog.from_config(
        ROOT / "config/defense/feature-catalog-v5.json"
    )
    for partition_name, item in prepared.items():
        decisions = safe_smoke_corpus.partitions[partition_name].decisions
        rebuilt = build_sentinel_features(decisions, catalog=catalog)
        assert item.event_ids == tuple(row.event_id for row in decisions)
        assert np.array_equal(item.matrix, np.asarray(rebuilt.matrix, dtype="<f8"))
        assert item.labels.tolist() == [int(row.is_fraud) for row in decisions]
        assert item.trust_failures == tuple(
            row.integrity_status == "fail" if row.rail == "agentic" else False
            for row in decisions
        )
        assert tuple(item.feature_batch.provenance) == rebuilt.provenance
        assert item.feature_batch.batch_sha256 == rebuilt.batch_sha256
        if partition_name == "development_test":
            assert item.training_evidence is None
        else:
            assert item.training_evidence is not None


def test_arm_stage_matches_existing_safe_path_except_real_latency(
    tmp_path: Path,
    safe_smoke_corpus: V5Corpus,
) -> None:
    """Checkpoint boundaries cannot change any four-arm model or decision semantics."""
    protocol = _protocol()
    feature_root, feature_manifest = _build_feature_chain(
        tmp_path=tmp_path, corpus=safe_smoke_corpus
    )
    arm_capability = staged._issue_stage_capability(
        protocol=protocol,
        mode=V5KaggleMode.CAPACITY_VALIDATION,
        attempt_receipt_sha256="5" * 64,
        predecessor=feature_manifest,
    )
    arm_root = tmp_path / "arms"
    arm_manifest = _publish_stage(
        output_root=arm_root,
        stage=V5KaggleStage.ARMS,
        records=staged.execute_v5_arm_stage(
            root=ROOT,
            capability=arm_capability,
            corpus_checkpoint_root=tmp_path / "corpus",
            feature_checkpoint_root=feature_root,
        ),
        predecessor=feature_manifest,
    )
    staged_results = staged.load_v5_arm_checkpoint(
        checkpoint_root=arm_root,
        limits=protocol.resources,
    )

    development, _profile = staged._development_protocol_for_stage(
        root=ROOT,
        protocol=protocol,
        mode=V5KaggleMode.CAPACITY_VALIDATION,
    )
    catalog = SentinelFeatureCatalog.from_config(
        ROOT / development.feature_catalog_path
    )
    configuration = load_v5_arm_configuration(
        ROOT / "config/defense/defense-v5-arms.json",
        catalog=catalog,
        protocol=development,
    )
    partitions = safe_smoke_corpus.partitions
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
        bootstrap_seed=development.seeds.bootstrap,
    )
    direct_results = tuple(
        V5EvaluationResult.model_validate(direct["arm_results"][arm.value])
        for arm in _CURRENT_ARMS
    )

    deterministic_records = tuple(
        iter_v5_checkpoint_records(
            output_root=arm_root,
            limits=protocol.resources,
        )
    )
    assert arm_manifest.record_count > 5
    assert all(len(record.canonical_bytes) <= 16_777_216 for record in deterministic_records)
    assert "arm_result" not in {record.kind for record in deterministic_records}
    assert {"arm_result_meta", "arm_execution_artifacts", "arm_result_rows"} <= {
        record.kind for record in deterministic_records
    }
    assert tuple(result.arm for result in staged_results) == tuple(
        arm.value for arm in _CURRENT_ARMS
    )
    assert [_deterministic_arm_document(item) for item in staged_results] == [
        _deterministic_arm_document(item) for item in direct_results
    ]
    support_orders = {
        tuple(row.support.event_id for row in result.row_evidence)
        for result in staged_results
    }
    assert len(support_orders) == 1
    by_arm = {result.arm: result for result in staged_results}
    assert not any(
        name.startswith("graph_")
        for name in by_arm[V5Arm.ENSEMBLE_NO_GRAPH.value].arm_spec.feature_names
    )
    assert any(
        name.startswith("graph_")
        for name in by_arm[V5Arm.ENSEMBLE_WITH_GRAPH.value].arm_spec.feature_names
    )
    assert all(
        not row.model_raw_scores
        for row in by_arm[V5Arm.RULES_ONLY.value].row_evidence
    )


def test_non_authoritative_rescue_stream_releases_each_exact_arm(
    tmp_path: Path,
    safe_smoke_corpus: V5Corpus,
) -> None:
    """The OOM rescue must restore exactly one arm before advancing to the next."""
    protocol = _protocol()
    feature_root, feature_manifest = _build_feature_chain(
        tmp_path=tmp_path, corpus=safe_smoke_corpus
    )
    arm_capability = staged._issue_stage_capability(
        protocol=protocol,
        mode=V5KaggleMode.CAPACITY_VALIDATION,
        attempt_receipt_sha256="5" * 64,
        predecessor=feature_manifest,
    )
    arm_root = tmp_path / "arms"
    arm_manifest = _publish_stage(
        output_root=arm_root,
        stage=V5KaggleStage.ARMS,
        records=staged.execute_v5_arm_stage(
            root=ROOT,
            capability=arm_capability,
            corpus_checkpoint_root=tmp_path / "corpus",
            feature_checkpoint_root=feature_root,
        ),
        predecessor=feature_manifest,
    )
    expected = staged.load_v5_arm_checkpoint(
        checkpoint_root=arm_root,
        limits=protocol.resources,
    )
    candidate_loader = getattr(staged, "load_v5_metric_worker_arm_result", None)
    assert candidate_loader is not None, "candidate one-arm Stage 70 loader is missing"
    for expected_index, expected_result in enumerate(expected):
        isolated = candidate_loader(
            checkpoint_root=arm_root,
            limits=protocol.resources,
            target_arm=expected_result.arm,
        )
        assert isolated.arm_index == expected_index
        assert isolated.arm == expected_result.arm
        assert isolated.result.model_dump(mode="json") == expected_result.model_dump(
            mode="json"
        )
        assert isolated.deterministic_result_sha256 == (
            staged._arm_core_and_observation(expected_result)[0][
                "deterministic_result_sha256"
            ]
        )
    evidence_protocol = load_v5_evidence_protocol(
        ROOT / "config/defense/defense-v5-evidence.json",
        root=ROOT,
    )
    worker_receipt = staged._run_v5_metric_arm_worker_subprocess(
        root=ROOT,
        capability=arm_capability,
        arm_checkpoint_root=arm_root,
        arm_manifest=arm_manifest,
        target_arm=V5Arm.RULES_ONLY,
        evidence_protocol=evidence_protocol,
        limits=protocol.resources,
        timeout_seconds=180.0,
    )
    assert worker_receipt.arm is V5Arm.RULES_ONLY
    assert worker_receipt.resource_telemetry.fresh_interpreter is True
    assert worker_receipt.metric_summary.stable_document() == _stable_complete_metrics(
        evaluate_v5_complete_result(
            result=expected[0],
            protocol=evidence_protocol,
        ).model_dump(mode="json"),
        deterministic_result_sha256=worker_receipt.deterministic_result_sha256,
    )

    streamed = rescue.iter_non_authoritative_rescue_arm_results(
        checkpoint_root=arm_root,
        limits=protocol.resources,
    )
    first = next(streamed)
    assert first.arm == expected[0].arm
    assert first.result.model_dump(mode="json") == expected[0].model_dump(mode="json")
    assert (
        first.deterministic_result_sha256
        == staged._arm_core_and_observation(expected[0])[0]["deterministic_result_sha256"]
    )
    first_reference = weakref.ref(first.result)
    del first

    second = next(streamed)
    gc.collect()
    assert first_reference() is None
    observed = [second]
    observed.extend(streamed)
    assert tuple(item.arm for item in observed) == tuple(result.arm for result in expected[1:])
    assert [item.result.model_dump(mode="json") for item in observed] == [
        result.model_dump(mode="json") for result in expected[1:]
    ]
    for expected_result in expected:
        isolated = rescue.load_non_authoritative_rescue_arm_result(
            checkpoint_root=arm_root,
            limits=protocol.resources,
            target_arm=expected_result.arm,
        )
        assert isolated.arm == expected_result.arm
        assert isolated.result.model_dump(mode="json") == expected_result.model_dump(mode="json")
