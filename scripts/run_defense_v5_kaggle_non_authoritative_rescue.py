"""Run the explicitly non-authoritative Stage 70 OOM rescue.

The coordinator intentionally imports only the standard library. Each arm metric and
the compact finalization run in separate fresh Python interpreters so the operating
system can reclaim every restored arm before the next one starts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import traceback
from pathlib import Path
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from apar.evaluation.v5_checkpoint_storage import V5CheckpointManifest
    from apar.evaluation.v5_kaggle_protocol import (
        V5KaggleProtocol,
        V5KaggleResourceGates,
    )

_EXPECTED_SOURCE_COMMIT = "40fb4a131da36556d2b8a04564cb62d73152c7c1"
_CAPACITY_VALIDATION_MODE = "kaggle_capacity_validation"
_ARM_ORDER = (
    "rules_only",
    "ensemble_no_graph",
    "ensemble_with_graph",
    "full_sentinel",
)
_INTERNAL_GUARD = "apar-sentinel-v5-non-authoritative-rescue/1"
_INTERNAL_ACTIONS = ("arm", "finalize")


def _canonical(document: object) -> bytes:
    return json.dumps(
        document,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode()


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _write_atomic(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _load_bound_arm_receipt(*, path: Path, expected_arm: str) -> dict[str, object]:
    document = json.loads(path.read_bytes())
    if not isinstance(document, dict):
        raise RuntimeError("compact rescue arm receipt is not an object")
    claimed = document.pop("receipt_sha256", None)
    metric_core = document.get("metric_core")
    metric_observation = document.get("metric_observation")
    if not isinstance(metric_core, dict) or not isinstance(metric_observation, dict):
        raise RuntimeError("compact rescue arm metric documents are absent")
    core_without_digest = dict(metric_core)
    core_claimed = core_without_digest.pop(
        "deterministic_complete_metrics_sha256",
        None,
    )
    observation_without_digest = dict(metric_observation)
    observation_claimed = observation_without_digest.pop(
        "compact_observation_sha256",
        None,
    )
    if (
        set(document)
        != {
            "schema_version",
            "authoritative",
            "accepted_capacity_evidence",
            "arm",
            "deterministic_result_sha256",
            "metric_core",
            "metric_observation",
            "readiness_bundle",
        }
        or document.get("schema_version")
        != "apar-sentinel-v5-non-authoritative-compact-arm-receipt/1"
        or document.get("authoritative") is not False
        or document.get("accepted_capacity_evidence") is not False
        or document.get("arm") != expected_arm
        or not isinstance(document.get("deterministic_result_sha256"), str)
        or claimed != _sha256_bytes(_canonical(document))
        or metric_core.get("arm") != expected_arm
        or metric_core.get("deterministic_result_sha256")
        != document.get("deterministic_result_sha256")
        or core_claimed != _sha256_bytes(_canonical(core_without_digest))
        or metric_observation.get("arm") != expected_arm
        or metric_observation.get("support_sha256") != metric_core.get("support_sha256")
        or observation_claimed != _sha256_bytes(_canonical(observation_without_digest))
        or (expected_arm == "full_sentinel") != isinstance(document.get("readiness_bundle"), dict)
    ):
        raise RuntimeError("compact rescue arm receipt binding differs")
    document["receipt_sha256"] = claimed
    return cast(dict[str, object], document)


def _load_bound_receipt(path: Path) -> dict[str, object]:
    receipt = json.loads(path.read_bytes())
    if not isinstance(receipt, dict):
        raise RuntimeError("rescue receipt is not an object")
    claimed = receipt.pop("receipt_sha256", None)
    if (
        set(receipt)
        != {
            "schema_version",
            "authoritative",
            "accepted_capacity_evidence",
            "artifact",
            "artifact_size_bytes",
            "artifact_sha256",
            "rescue_compact_sha256",
            "arm_receipt_sha256",
            "lineage_terminal_manifest_sha256",
        }
        or receipt.get("schema_version")
        != "apar-sentinel-v5-non-authoritative-compact-rescue-receipt/1"
        or receipt.get("authoritative") is not False
        or receipt.get("accepted_capacity_evidence") is not False
        or claimed != _sha256_bytes(_canonical(receipt))
    ):
        raise RuntimeError("non-authoritative rescue receipt binding differs")
    receipt["receipt_sha256"] = claimed
    return cast(dict[str, object], receipt)


def _run_fresh_action(*, action: str, environment: dict[str, str]) -> None:
    if action not in _INTERNAL_ACTIONS:
        raise RuntimeError("unknown non-authoritative rescue action")
    subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "--internal-action", action],
        check=True,
        env=environment,
        text=True,
    )


def _base_child_environment(
    *,
    root: Path,
    predecessor_chain: Path,
    output_root: Path,
    approved_source_commit: str,
    execution_mode: str,
) -> dict[str, str]:
    environment = os.environ.copy()
    environment.update(
        {
            "APAR_V5_RESCUE_INTERNAL_GUARD": _INTERNAL_GUARD,
            "APAR_V5_RESCUE_SOURCE_ROOT": str(root),
            "APAR_V5_RESCUE_PREDECESSOR_CHAIN": str(predecessor_chain),
            "APAR_V5_RESCUE_OUTPUT_ROOT": str(output_root),
            "APAR_V5_RESCUE_APPROVED_COMMIT": approved_source_commit,
            "APAR_V5_RESCUE_EXECUTION_MODE": execution_mode,
        }
    )
    return environment


def _arm_metric_worker_from_environment() -> None:
    from apar.evaluation.v5_controls import assemble_v5_control_suite
    from apar.evaluation.v5_evidence_bundle import build_v5_readiness_evidence
    from apar.evaluation.v5_evidence_layers import _stable_controls, _stable_readiness
    from apar.evaluation.v5_evidence_protocol import load_v5_evidence_protocol
    from apar.evaluation.v5_kaggle_protocol import (
        V5KaggleStage,
        load_v5_kaggle_protocol,
    )
    from apar.evaluation.v5_kaggle_rescue import (
        build_non_authoritative_compact_arm_metric_documents,
        load_non_authoritative_rescue_arm_result,
    )
    from apar.evaluation.v5_metrics import evaluate_v5_complete_result
    from apar.evaluation.v5_staged_evidence import load_v5_control_group_checkpoint

    root = Path(os.environ["APAR_V5_RESCUE_SOURCE_ROOT"])
    predecessor_chain = Path(os.environ["APAR_V5_RESCUE_PREDECESSOR_CHAIN"])
    arm = os.environ["APAR_V5_RESCUE_TARGET_ARM"]
    output_path = Path(os.environ["APAR_V5_RESCUE_ARM_OUTPUT_PATH"])
    protocol = load_v5_kaggle_protocol(
        root / "config/defense/defense-v5-kaggle-recovery.json",
        root=root,
    )
    evidence_protocol = load_v5_evidence_protocol(
        root / protocol.source_bindings.evidence_protocol_path,
        root=root,
    )
    restored = load_non_authoritative_rescue_arm_result(
        checkpoint_root=predecessor_chain / V5KaggleStage.ARMS.value,
        limits=protocol.resources,
        target_arm=arm,
    )
    metric = evaluate_v5_complete_result(
        result=restored.result,
        protocol=evidence_protocol,
    )
    metric_core, metric_observation = build_non_authoritative_compact_arm_metric_documents(
        metric=metric,
        deterministic_result_sha256=restored.deterministic_result_sha256,
    )
    readiness_bundle: dict[str, object] | None = None
    if arm == "full_sentinel":
        control_stages = (
            V5KaggleStage.LABEL_SHUFFLE,
            V5KaggleStage.IDENTITY_RENAME,
            V5KaggleStage.FUTURE_CAUSALITY,
            V5KaggleStage.EQUAL_TIME_ISOLATION,
            V5KaggleStage.FEATURE_LEAKAGE,
            V5KaggleStage.SINGLE_CLASS_CONTROLS,
        )
        groups = tuple(
            load_v5_control_group_checkpoint(
                checkpoint_root=predecessor_chain / stage.value,
                limits=protocol.resources,
            )
            for stage in control_stages
        )
        controls = assemble_v5_control_suite(groups)
        readiness = build_v5_readiness_evidence(metrics=metric, controls=controls)
        stable_controls, control_digests = _stable_controls(controls.model_dump(mode="json"))
        stable_readiness = _stable_readiness(
            readiness.model_dump(mode="json"),
            deterministic_control_digests=control_digests,
        )
        readiness_bundle = {
            "schema_version": "apar-sentinel-v5-non-authoritative-compact-readiness/1",
            "deterministic_controls": stable_controls,
            "deterministic_readiness": stable_readiness,
            "observational_readiness": readiness.model_dump(mode="json"),
        }
        readiness_bundle["readiness_bundle_sha256"] = _sha256_bytes(_canonical(readiness_bundle))
    document: dict[str, object] = {
        "schema_version": "apar-sentinel-v5-non-authoritative-compact-arm-receipt/1",
        "authoritative": False,
        "accepted_capacity_evidence": False,
        "arm": restored.arm,
        "deterministic_result_sha256": restored.deterministic_result_sha256,
        "metric_core": metric_core,
        "metric_observation": metric_observation,
        "readiness_bundle": readiness_bundle,
    }
    document["receipt_sha256"] = _sha256_bytes(_canonical(document))
    _write_atomic(output_path, _canonical(document))


def _validated_predecessors(
    *, chain_root: Path, protocol: V5KaggleProtocol
) -> tuple[V5CheckpointManifest, ...]:
    from itertools import pairwise

    from apar.evaluation.v5_checkpoint_storage import read_v5_checkpoint_manifest
    from apar.evaluation.v5_kaggle_protocol import V5KaggleMode, V5KaggleStage

    predecessor_stages = tuple(V5KaggleStage)[:-2]
    manifests = tuple(
        read_v5_checkpoint_manifest(
            output_root=chain_root / stage.value,
            limits=protocol.resources,
        )
        for stage in predecessor_stages
    )
    if (
        tuple(item.stage for item in manifests) != predecessor_stages
        or any(
            current.predecessor_manifest_sha256 != previous.manifest_sha256
            for previous, current in pairwise(manifests)
        )
        or len({item.run_binding_sha256 for item in manifests}) != 1
        or len({item.attempt_receipt_sha256 for item in manifests}) != 1
        or manifests[0].run_binding_sha256
        != protocol.run_binding_sha256(V5KaggleMode.CAPACITY_VALIDATION)
    ):
        raise RuntimeError("accepted predecessor checkpoint lineage differs")
    return manifests


def _checkpoint_arm_digests(
    *, checkpoint_root: Path, limits: V5KaggleResourceGates
) -> tuple[str, ...]:
    from apar.evaluation.v5_checkpoint_storage import iter_v5_checkpoint_records
    from apar.evaluation.v5_staged_evidence import (
        _ARM_HEADER_SCHEMA,
        _read_json_record,
    )

    records = iter(iter_v5_checkpoint_records(output_root=checkpoint_root, limits=limits))
    try:
        header_record = next(records)
    except StopIteration as error:
        raise RuntimeError("arm checkpoint is empty") from error
    header = _read_json_record(header_record, label="arm header")
    digests = header.get("deterministic_result_sha256")
    if (
        header_record.kind != "arm_header"
        or header_record.key != "arms"
        or header.get("schema_version") != _ARM_HEADER_SCHEMA
        or header.get("arm_order") != list(_ARM_ORDER)
        or not isinstance(digests, list)
        or len(digests) != len(_ARM_ORDER)
        or any(not isinstance(item, str) for item in digests)
    ):
        raise RuntimeError("arm checkpoint compact binding differs")
    return tuple(cast(list[str], digests))


def _finalize_from_environment() -> None:
    from apar.evaluation.v5_kaggle_protocol import V5KaggleStage, load_v5_kaggle_protocol

    root = Path(os.environ["APAR_V5_RESCUE_SOURCE_ROOT"])
    predecessor_chain = Path(os.environ["APAR_V5_RESCUE_PREDECESSOR_CHAIN"])
    output_root = Path(os.environ["APAR_V5_RESCUE_OUTPUT_ROOT"])
    approved_source_commit = os.environ["APAR_V5_RESCUE_APPROVED_COMMIT"]
    execution_mode = os.environ["APAR_V5_RESCUE_EXECUTION_MODE"]
    protocol = load_v5_kaggle_protocol(
        root / "config/defense/defense-v5-kaggle-recovery.json",
        root=root,
    )
    manifests = _validated_predecessors(
        chain_root=predecessor_chain,
        protocol=protocol,
    )
    arm_output_root = output_root / "compact-arm-receipts"
    arm_documents = tuple(
        _load_bound_arm_receipt(
            path=arm_output_root / f"{index:02d}-{arm}.json",
            expected_arm=arm,
        )
        for index, arm in enumerate(_ARM_ORDER)
    )
    result_digests = tuple(
        cast(str, document["deterministic_result_sha256"]) for document in arm_documents
    )
    checkpoint_digests = _checkpoint_arm_digests(
        checkpoint_root=predecessor_chain / V5KaggleStage.ARMS.value,
        limits=protocol.resources,
    )
    if result_digests != checkpoint_digests:
        raise RuntimeError("resumed arm metrics differ from the accepted arm checkpoint")
    support_digests = {
        cast(dict[str, object], document["metric_core"])["support_sha256"]
        for document in arm_documents
    }
    if len(support_digests) != 1:
        raise RuntimeError("isolated rescue metric support differs")
    readiness_bundle = arm_documents[-1].get("readiness_bundle")
    if not isinstance(readiness_bundle, dict):
        raise RuntimeError("full_sentinel compact readiness is absent")
    readiness_claimed = readiness_bundle.get("readiness_bundle_sha256")
    readiness_without_digest = dict(readiness_bundle)
    readiness_without_digest.pop("readiness_bundle_sha256", None)
    if readiness_claimed != _sha256_bytes(_canonical(readiness_without_digest)):
        raise RuntimeError("full_sentinel compact readiness binding differs")

    envelope: dict[str, object] = {
        "schema_version": "apar-sentinel-v5-non-authoritative-compact-rescue/1",
        "authoritative": False,
        "accepted_capacity_evidence": False,
        "rescue_reason": "official_stage_70_memory_exhaustion",
        "approved_source_commit": approved_source_commit,
        "execution_mode": execution_mode,
        "official_predecessor_stage_manifests": tuple(
            (item.stage.value, item.manifest_sha256) for item in manifests
        ),
        "run_binding_sha256": manifests[0].run_binding_sha256,
        "attempt_receipt_sha256": manifests[0].attempt_receipt_sha256,
        "arm_metric_receipts": tuple(
            {
                "arm": document["arm"],
                "deterministic_result_sha256": document["deterministic_result_sha256"],
                "metric_core": document["metric_core"],
                "metric_observation": document["metric_observation"],
                "receipt_sha256": document["receipt_sha256"],
            }
            for document in arm_documents
        ),
        "readiness_bundle": readiness_bundle,
    }
    envelope["rescue_compact_sha256"] = _sha256_bytes(_canonical(envelope))
    envelope_path = output_root / "70_metrics_non_authoritative_compact_rescue.json"
    _write_atomic(envelope_path, _canonical(envelope))

    receipt: dict[str, object] = {
        "schema_version": "apar-sentinel-v5-non-authoritative-compact-rescue-receipt/1",
        "authoritative": False,
        "accepted_capacity_evidence": False,
        "artifact": envelope_path.name,
        "artifact_size_bytes": envelope_path.stat().st_size,
        "artifact_sha256": _sha256_bytes(envelope_path.read_bytes()),
        "rescue_compact_sha256": envelope["rescue_compact_sha256"],
        "arm_receipt_sha256": tuple(
            cast(str, document["receipt_sha256"]) for document in arm_documents
        ),
        "lineage_terminal_manifest_sha256": manifests[-1].manifest_sha256,
    }
    receipt["receipt_sha256"] = _sha256_bytes(_canonical(receipt))
    _write_atomic(output_root / "rescue-receipt.json", _canonical(receipt))


def run_non_authoritative_rescue(
    *,
    root: Path,
    predecessor_chain: Path,
    output_root: Path,
    approved_source_commit: str,
    execution_mode: str,
) -> dict[str, object]:
    """Run each heavy action in a fresh interpreter and emit a rescue-only receipt."""
    if approved_source_commit != _EXPECTED_SOURCE_COMMIT:
        raise RuntimeError("rescue source commit differs from the failed frozen run")
    if execution_mode != _CAPACITY_VALIDATION_MODE:
        raise RuntimeError("rescue permits capacity validation mode only")
    if output_root.name == "70_metrics":
        raise RuntimeError("rescue output cannot use the official Stage 70 path")
    if not predecessor_chain.is_dir():
        raise RuntimeError("accepted Stage 60 predecessor chain is absent")

    output_root.mkdir(parents=True, exist_ok=True)
    arm_output_root = output_root / "compact-arm-receipts"
    arm_output_root.mkdir(exist_ok=True)
    base_environment = _base_child_environment(
        root=root,
        predecessor_chain=predecessor_chain,
        output_root=output_root,
        approved_source_commit=approved_source_commit,
        execution_mode=execution_mode,
    )

    for index, arm in enumerate(_ARM_ORDER):
        output_path = arm_output_root / f"{index:02d}-{arm}.json"
        if output_path.is_file():
            _load_bound_arm_receipt(path=output_path, expected_arm=arm)
            print(
                json.dumps(
                    {"rescue_arm_reused": arm, "fresh_interpreter": True},
                    sort_keys=True,
                ),
                flush=True,
            )
            continue
        environment = dict(base_environment)
        environment["APAR_V5_RESCUE_TARGET_ARM"] = arm
        environment["APAR_V5_RESCUE_ARM_OUTPUT_PATH"] = str(output_path)
        _run_fresh_action(action="arm", environment=environment)
        _load_bound_arm_receipt(path=output_path, expected_arm=arm)
        print(
            json.dumps(
                {"rescue_arm_complete": arm, "fresh_interpreter": True},
                sort_keys=True,
            ),
            flush=True,
        )

    receipt_path = output_root / "rescue-receipt.json"
    if not receipt_path.is_file():
        _run_fresh_action(action="finalize", environment=dict(base_environment))
    return _load_bound_receipt(receipt_path)


def _run_internal_action(action: str) -> None:
    if os.environ.get("APAR_V5_RESCUE_INTERNAL_GUARD") != _INTERNAL_GUARD:
        raise RuntimeError("non-authoritative rescue internal guard differs")
    if os.environ.get("APAR_V5_RESCUE_APPROVED_COMMIT") != _EXPECTED_SOURCE_COMMIT:
        raise RuntimeError("non-authoritative rescue approved commit differs")
    if os.environ.get("APAR_V5_RESCUE_EXECUTION_MODE") != _CAPACITY_VALIDATION_MODE:
        raise RuntimeError("non-authoritative rescue execution mode differs")
    if action == "arm":
        _arm_metric_worker_from_environment()
    elif action == "finalize":
        _finalize_from_environment()
    else:  # pragma: no cover - argparse constrains this branch.
        raise RuntimeError("unknown non-authoritative rescue action")


def _main() -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--internal-action", choices=_INTERNAL_ACTIONS, required=True)
    arguments = parser.parse_args()
    try:
        _run_internal_action(arguments.internal_action)
    except BaseException:  # pragma: no cover - child failure is surfaced by exit status.
        traceback.print_exc()
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_main())
