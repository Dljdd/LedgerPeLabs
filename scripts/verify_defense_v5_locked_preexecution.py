"""Fail-closed audit for the one-time locked Sentinel v5 evidence run."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import subprocess
import sys
from pathlib import Path
from typing import Any

from apar.evaluation.v5_evidence_protocol import (
    V5EvidenceProtocol,
    load_v5_evidence_protocol,
)
from apar.evaluation.v5_protocol import load_v5_development_protocol
from apar.evaluation.v5_run_mode import (
    V5LockedEvidenceRunBinding,
    V5RunMode,
    V5RunSupportPlan,
    build_v5_run_support_plan,
    resolve_v5_run_mode,
)
from apar.features.sentinel import SentinelFeatureCatalog
from apar.v5_independent_verifier import (
    IndependentVerificationError,
    verify_evidence_bytes,
)

_PREREGISTRATION_PATH = (
    "config/defense/defense-v5-locked-development-preregistration.json"
)
_HISTORICAL_SAFE_FREEZE_PATH = "config/defense/defense-v5-safe-core-freeze.json"
_HISTORICAL_SAFE_CORE_SHA256 = (
    "784a762fd90a65219a233e87df35290ac87c8fe8e4b9024de46564568f633719"
)
_EXACT_COMMAND = (
    ".venv/bin/python scripts/run_defense_v5_locked_development.py --root . "
    "--safe-evidence /private/tmp/apar-v5-approved-safe-evidence.json "
    "--approved-commit HEAD --authorize-exactly-once"
)


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode()


def _digest(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _git(root: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise ValueError(completed.stderr.strip() or "git command failed")
    return completed.stdout.strip()


def _regular_single_link(path: Path, label: str) -> os.stat_result:
    metadata = path.lstat()
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
        raise ValueError(f"{label} must be a single-link regular file")
    return metadata


def _source_paths(root: Path) -> tuple[str, ...]:
    arms = json.loads((root / "config/defense/defense-v5-arms.json").read_bytes())
    evidence = json.loads(
        (root / "config/defense/defense-v5-evidence.json").read_bytes()
    )
    paths = {
        "config/defense/defense-v5-arms.json",
        "config/defense/defense-v5-development.json",
        "config/defense/defense-v5-evidence.json",
        "config/defense/feature-catalog-v5.json",
        *arms["implementation_paths"],
        *evidence["implementation_paths"],
    }
    return tuple(sorted(paths))


def _source_mode(root: Path, commit: str, relative: str) -> str:
    record = _git(root, "ls-tree", commit, "--", relative)
    if not record or "\t" not in record:
        raise ValueError(f"SOURCE file is not tracked: {relative}")
    mode = record.split(maxsplit=1)[0]
    if mode not in {"100644", "100755"}:
        raise ValueError(f"SOURCE file mode is unsupported: {relative}")
    return mode


def _source_records(root: Path, commit: str) -> list[list[str]]:
    records: list[list[str]] = []
    for relative in _source_paths(root):
        path = root / relative
        _regular_single_link(path, f"SOURCE file {relative}")
        records.append(
            [
                relative,
                _source_mode(root, commit, relative),
                hashlib.sha256(path.read_bytes()).hexdigest(),
            ]
        )
    return records


def _frozen_definitions(protocol: V5EvidenceProtocol) -> dict[str, object]:
    return {
        "controls": protocol.controls.model_dump(mode="json"),
        "calibration": protocol.calibration.model_dump(mode="json"),
        "bootstrap": protocol.bootstrap.model_dump(mode="json"),
        "economics": protocol.economics.model_dump(mode="json"),
        "metric_definitions": protocol.metric_definitions.model_dump(mode="json"),
        "gates": [item.model_dump(mode="json") for item in protocol.gates],
    }


def _manifest_without_digest(
    *,
    root: Path,
    source_commit: str,
    safe_verification: dict[str, object],
) -> dict[str, object]:
    evidence = load_v5_evidence_protocol(
        root / "config/defense/defense-v5-evidence.json", root=root
    )
    development = load_v5_development_protocol(
        root / "config/defense/defense-v5-development.json"
    )
    mode = resolve_v5_run_mode(
        mode=V5RunMode.LOCKED_DEVELOPMENT,
        evidence_protocol=evidence,
        development_protocol=development,
    )
    plan = build_v5_run_support_plan(
        mode=V5RunMode.LOCKED_DEVELOPMENT,
        evidence_protocol=evidence,
        development_protocol=development,
    )
    catalog = SentinelFeatureCatalog.from_config(
        root / development.feature_catalog_path
    )
    storage = evidence.locked_artifact_storage
    return {
        "schema_version": "apar-sentinel-v5-locked-preregistration/1",
        "source_commit": source_commit,
        "source_tree_oid": _git(root, "rev-parse", f"{source_commit}^{{tree}}"),
        "preregistration_path": _PREREGISTRATION_PATH,
        "source_bindings": {
            "base_protocol_sha256": evidence.base_protocol_sha256,
            "arm_protocol_sha256": evidence.arm_protocol_sha256,
            "evidence_protocol_sha256": evidence.evidence_protocol_sha256,
            "implementation_sha256": evidence.implementation_sha256,
            "catalog_sha256": catalog.catalog_sha256,
            "verifier_sha256": hashlib.sha256(
                (root / "src/apar/v5_independent_verifier.py").read_bytes()
            ).hexdigest(),
        },
        "source_files": _source_records(root, source_commit),
        "safe_validation": {
            "historical_deterministic_core_sha256": _HISTORICAL_SAFE_CORE_SHA256,
            "approved_deterministic_core_sha256": safe_verification[
                "deterministic_core_sha256"
            ],
            "approved_observational_environment_sha256": safe_verification[
                "observational_environment_sha256"
            ],
        },
        "closed_run_mode": mode.model_dump(mode="json"),
        "production_support_plan": plan.model_dump(mode="json"),
        "frozen_definitions": _frozen_definitions(evidence),
        "output_contract": {
            **storage.model_dump(mode="json"),
            "candidate_must_be_absent": True,
            "historical_result_path": evidence.existing_development_result_path,
            "historical_result_sha256": (
                evidence.existing_development_result_sha256
            ),
            "one_time_no_resume_or_retry": True,
            "legacy_summary_is_not_evidence": True,
        },
        "exact_command": _EXACT_COMMAND,
    }


def build_locked_preregistration_manifest(
    *, root: Path, safe_evidence: Path, source_commit: str
) -> dict[str, object]:
    """Create the only document allowed in the PREREGISTRATION commit."""
    root = root.resolve()
    if _git(root, "status", "--porcelain", "--untracked-files=all"):
        raise ValueError("SOURCE worktree is not clean")
    if _git(root, "rev-parse", "HEAD") != source_commit or len(source_commit) != 40:
        raise ValueError("HEAD does not equal the exact SOURCE commit")
    safe_verification = verify_evidence_bytes(safe_evidence.read_bytes(), root=root)
    values = _manifest_without_digest(
        root=root,
        source_commit=source_commit,
        safe_verification=safe_verification,
    )
    values["manifest_sha256"] = _digest(values)
    return values


def verify_locked_commit_chronology(
    *, root: Path, approved_commit: str, source_commit: str, preregistration_path: str
) -> None:
    """Require one preregistration-only child over the exact SOURCE commit."""
    head = _git(root, "rev-parse", "HEAD")
    if approved_commit == "HEAD":
        approved_commit = head
    if head != approved_commit or len(approved_commit) != 40:
        raise ValueError("HEAD does not equal the exact approved PREREGISTRATION commit")
    parents = _git(root, "rev-list", "--parents", "-n", "1", head).split()
    if len(parents) != 2 or parents[1] != source_commit:
        raise ValueError("PREREGISTRATION commit is not the single SOURCE child")
    changed = tuple(
        line
        for line in _git(
            root, "diff-tree", "--no-commit-id", "--name-only", "-r", head
        ).splitlines()
        if line
    )
    if changed != (preregistration_path,):
        raise ValueError("PREREGISTRATION commit must add only its manifest")


def _validate_manifest(
    *, root: Path, document: dict[str, object], safe_verification: dict[str, object]
) -> tuple[V5EvidenceProtocol, V5RunSupportPlan, SentinelFeatureCatalog]:
    expected_fields = {
        "schema_version",
        "source_commit",
        "source_tree_oid",
        "preregistration_path",
        "source_bindings",
        "source_files",
        "safe_validation",
        "closed_run_mode",
        "production_support_plan",
        "frozen_definitions",
        "output_contract",
        "exact_command",
        "manifest_sha256",
    }
    if set(document) != expected_fields:
        raise ValueError("locked preregistration schema differs")
    if document["schema_version"] != "apar-sentinel-v5-locked-preregistration/1":
        raise ValueError("locked preregistration version differs")
    if document["preregistration_path"] != _PREREGISTRATION_PATH:
        raise ValueError("locked preregistration path differs")
    if document["exact_command"] != _EXACT_COMMAND:
        raise ValueError("locked one-time command differs")
    if document["manifest_sha256"] != _digest(
        {key: value for key, value in document.items() if key != "manifest_sha256"}
    ):
        raise ValueError("locked preregistration digest differs")
    source_commit = str(document["source_commit"])
    if len(source_commit) != 40 or _git(
        root, "rev-parse", f"{source_commit}^{{tree}}"
    ) != document["source_tree_oid"]:
        raise ValueError("locked SOURCE commit/tree differs")
    evidence = load_v5_evidence_protocol(
        root / "config/defense/defense-v5-evidence.json", root=root
    )
    development = load_v5_development_protocol(
        root / "config/defense/defense-v5-development.json"
    )
    catalog = SentinelFeatureCatalog.from_config(
        root / development.feature_catalog_path
    )
    expected_bindings = {
        "base_protocol_sha256": evidence.base_protocol_sha256,
        "arm_protocol_sha256": evidence.arm_protocol_sha256,
        "evidence_protocol_sha256": evidence.evidence_protocol_sha256,
        "implementation_sha256": evidence.implementation_sha256,
        "catalog_sha256": catalog.catalog_sha256,
        "verifier_sha256": hashlib.sha256(
            (root / "src/apar/v5_independent_verifier.py").read_bytes()
        ).hexdigest(),
    }
    if document["source_bindings"] != expected_bindings:
        raise ValueError("locked source/config/verifier bindings differ")
    if document["source_files"] != _source_records(root, source_commit):
        raise ValueError("locked SOURCE file set/modes/content hashes differ")
    expected_mode = resolve_v5_run_mode(
        mode=V5RunMode.LOCKED_DEVELOPMENT,
        evidence_protocol=evidence,
        development_protocol=development,
    )
    expected_plan = build_v5_run_support_plan(
        mode=V5RunMode.LOCKED_DEVELOPMENT,
        evidence_protocol=evidence,
        development_protocol=development,
    )
    if document["closed_run_mode"] != expected_mode.model_dump(mode="json"):
        raise ValueError("locked closed run mode differs")
    if document["production_support_plan"] != expected_plan.model_dump(mode="json"):
        raise ValueError("locked production support plan differs")
    if document["frozen_definitions"] != _frozen_definitions(evidence):
        raise ValueError("locked controls/metrics/bootstrap/economics differ")
    output = document["output_contract"]
    expected_output = {
        **evidence.locked_artifact_storage.model_dump(mode="json"),
        "candidate_must_be_absent": True,
        "historical_result_path": evidence.existing_development_result_path,
        "historical_result_sha256": evidence.existing_development_result_sha256,
        "one_time_no_resume_or_retry": True,
        "legacy_summary_is_not_evidence": True,
    }
    if output != expected_output:
        raise ValueError("locked output/storage/no-overwrite contract differs")
    safe = document["safe_validation"]
    if not isinstance(safe, dict) or safe != {
        "historical_deterministic_core_sha256": _HISTORICAL_SAFE_CORE_SHA256,
        "approved_deterministic_core_sha256": safe_verification[
            "deterministic_core_sha256"
        ],
        "approved_observational_environment_sha256": safe_verification[
            "observational_environment_sha256"
        ],
    }:
        raise ValueError("locked safe-core validation binding differs")
    return evidence, expected_plan, catalog


def _assert_absent(path: Path, label: str) -> None:
    if os.path.lexists(path):
        raise ValueError(f"{label} must be absent before the one-time run")


def verify_locked_preexecution(
    *, root: Path, safe_evidence: Path, approved_commit: str
) -> dict[str, object]:
    """Audit the complete frozen plan without executing any experiment workload."""
    root = root.resolve()
    if _git(root, "status", "--porcelain", "--untracked-files=all"):
        raise ValueError("worktree is not clean")
    preregistration_path = root / _PREREGISTRATION_PATH
    _regular_single_link(preregistration_path, "locked preregistration")
    raw_preregistration = preregistration_path.read_bytes()
    document = json.loads(raw_preregistration)
    if not isinstance(document, dict):
        raise ValueError("locked preregistration must be a JSON object")
    source_commit = str(document.get("source_commit", ""))
    verify_locked_commit_chronology(
        root=root,
        approved_commit=approved_commit,
        source_commit=source_commit,
        preregistration_path=_PREREGISTRATION_PATH,
    )
    safe_path = safe_evidence.resolve()
    _regular_single_link(safe_path, "approved safe evidence")
    safe_verification = verify_evidence_bytes(safe_path.read_bytes(), root=root)
    evidence, plan, catalog = _validate_manifest(
        root=root, document=document, safe_verification=safe_verification
    )
    historical_freeze = json.loads(
        (root / _HISTORICAL_SAFE_FREEZE_PATH).read_bytes()
    )
    if historical_freeze.get("approved_deterministic_core_sha256") != (
        _HISTORICAL_SAFE_CORE_SHA256
    ):
        raise ValueError("historical approved safe core differs")
    result_path = root / evidence.existing_development_result_path
    _regular_single_link(result_path, "historical development result")
    if hashlib.sha256(result_path.read_bytes()).hexdigest() != (
        evidence.existing_development_result_sha256
    ):
        raise ValueError("historical development result bytes changed")
    storage = evidence.locked_artifact_storage
    candidate = root / storage.candidate_manifest_path
    _assert_absent(candidate, "locked candidate manifest")
    _assert_absent(candidate.with_name(f"{candidate.name}.chunks"), "locked chunks")
    _assert_absent(root / storage.judge_summary_path, "locked judge summary")
    head = _git(root, "rev-parse", "HEAD")
    if approved_commit != "HEAD" and approved_commit != head:
        raise ValueError("approved commit differs from current PREREGISTRATION")
    preregistration_sha256 = hashlib.sha256(raw_preregistration).hexdigest()
    binding_values: dict[str, Any] = {
        "schema_version": "apar-sentinel-v5-locked-run-binding/1",
        "mode": "locked_development",
        "profile": "production",
        "development_test_seed": 2404,
        "source_commit": source_commit,
        "source_tree_oid": document["source_tree_oid"],
        "preregistration_commit": head,
        "preregistration_path": _PREREGISTRATION_PATH,
        "preregistration_sha256": preregistration_sha256,
        "base_protocol_sha256": evidence.base_protocol_sha256,
        "arm_protocol_sha256": evidence.arm_protocol_sha256,
        "evidence_protocol_sha256": evidence.evidence_protocol_sha256,
        "implementation_sha256": evidence.implementation_sha256,
        "catalog_sha256": catalog.catalog_sha256,
        "support_plan": plan,
        "candidate_manifest_path": storage.candidate_manifest_path,
        "storage_schema_version": storage.schema_version,
        "payload_schema_version": (
            "apar-sentinel-v5-locked-development-payload/1"
        ),
    }
    binding_values["run_binding_sha256"] = (
        V5LockedEvidenceRunBinding.compute_digest(binding_values)
    )
    binding = V5LockedEvidenceRunBinding.model_validate(binding_values)
    return {
        "verified": True,
        "approved_commit": head,
        "source_commit": source_commit,
        "worktree_clean": True,
        "safe_seed_executed": 404,
        "locked_seed_asserted_only": 2404,
        "safe_deterministic_core_sha256": safe_verification[
            "deterministic_core_sha256"
        ],
        "historical_safe_core_sha256": _HISTORICAL_SAFE_CORE_SHA256,
        "historical_result_sha256": evidence.existing_development_result_sha256,
        "candidate_result_absent": True,
        "production_support_plan_sha256": plan.support_plan_sha256,
        "preregistration_sha256": preregistration_sha256,
        "run_binding": binding.model_dump(mode="json"),
        "exact_command": _EXACT_COMMAND,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--safe-evidence", type=Path, required=True)
    parser.add_argument("--approved-commit", required=True)
    args = parser.parse_args()
    try:
        report = verify_locked_preexecution(
            root=args.root,
            safe_evidence=args.safe_evidence,
            approved_commit=args.approved_commit,
        )
    except (
        IndependentVerificationError,
        json.JSONDecodeError,
        OSError,
        ValueError,
    ) as error:
        print(
            json.dumps(
                {"verified": False, "error": str(error)},
                sort_keys=True,
                separators=(",", ":"),
            ),
            file=sys.stderr,
        )
        return 1
    print(json.dumps(report, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
