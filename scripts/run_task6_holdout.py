"""Verify or explicitly execute the frozen Task 6 v3.3 confirmatory experiment.

The local result-file check is an accidental-rerun guard, not cryptographic exactly-once
enforcement. Durable append-only execution receipts and cross-process verification remain
a Task 7 responsibility. The historical v2 runner is preserved at commit ``10bb4c4``.
"""

from __future__ import annotations

import argparse
import errno
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import tempfile
from collections.abc import Callable
from decimal import Decimal, localcontext
from pathlib import Path
from typing import Any, cast

ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = ROOT / "src"
for import_root in (SOURCE_ROOT, ROOT):
    rendered = str(import_root)
    if rendered not in sys.path:
        sys.path.insert(0, rendered)

from apar.contracts.decisions import Action  # noqa: E402
from apar.redteam import (  # noqa: E402
    AdaptiveSearch,
    AdaptiveTournamentPolicy,
    EvaluatorCapability,
    FamilyThreshold,
    FixedPolicy,
    LLMPlannerPolicy,
    Policy,
    PolicyCapability,
    PolicyMetrics,
    PrimaryOutcome,
    RandomPolicy,
    RunGroupCapability,
    SearchAuthority,
    SearchResult,
    capability_delta_report,
)
from apar.redteam.task6_experiment import (  # noqa: E402
    Task6Experiment,
    build_task6_experiment,
)

PREREGISTRATION_PATH = ROOT / "docs/experiments/task6-v3.3-holdout-preregistration.json"
CACHE_PATH = ROOT / "docs/experiments/task6-v3-cached-llm-replay.json"
CANCELLATION_PATH = ROOT / "docs/experiments/task6-v3-cancellation.json"
CANCELLED_RESULT_PATH = ROOT / "docs/experiments/task6-v3-holdout-result.json"
V31_CANCELLATION_PATH = ROOT / "docs/experiments/task6-v3.1-cancellation.json"
V31_RESULT_PATH = ROOT / "docs/experiments/task6-v3.1-holdout-result.json"
V32_CANCELLATION_PATH = ROOT / "docs/experiments/task6-v3.2-cancellation.json"
V32_RESULT_PATH = ROOT / "docs/experiments/task6-v3.2-holdout-result.json"
RESULT_PATH = ROOT / "docs/experiments/task6-v3.3-holdout-result.json"
_LOCK_FILES = ("uv.lock", "poetry.lock", "Pipfile.lock", "requirements.lock")
_HEX = frozenset("0123456789abcdef")
_SOURCE_STATUS = "final_v3_3_frozen_before_confirmatory_execution"
_PURPOSE = (
    "Maximum one final confirmatory evaluation of the unchanged family-agnostic "
    "finite-lattice frontier/UCB policy under an externally approved exact freeze."
)
_STOPPING_RULE = (
    "If v3.3 fails either preregistered target threshold or its confirmatory validity "
    "hard gate, no further confirmatory holdout will be opened; later work is "
    "exploratory or Task 7 evaluation."
)
_EXECUTION_BOUNDARY = {
    "local_approval_is_durable_authenticity": False,
    "approval_scope": (
        "The supplied commit and SHA-256 are out-of-band controller/reviewer "
        "authorization, not self-authenticating caller input."
    ),
    "task7_responsibility": (
        "Durably sign approval, isolate and terminate workers, and retain an "
        "append-only execution receipt."
    ),
}


class ResultPublicationDurabilityError(RuntimeError):
    """The result name is visible but its directory durability is unconfirmed."""

    def __init__(self, path: Path, error: OSError) -> None:
        self.path = path
        self.target_published = True
        super().__init__(
            f"result was published at {path}, but directory durability failed; "
            "inspect the existing result before retrying"
        )
        self.__cause__ = error


_UNSUPPORTED_DIRECTORY_FSYNC = frozenset(
    value
    for value in (
        errno.EINVAL,
        getattr(errno, "ENOTSUP", None),
        getattr(errno, "EOPNOTSUPP", None),
    )
    if value is not None
)


class _NoNetworkClient:
    provider = "fixture"
    model_id = "cached-default-v1"

    def __init__(self) -> None:
        self.calls = 0

    def complete(self, _request: dict[str, object]) -> dict[str, object]:
        self.calls += 1
        raise AssertionError("v3.3 cached planner attempted network transport")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _load_exact_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if type(value) is not dict or any(type(key) is not str for key in value):
        raise TypeError(f"{path.name} must contain an exact JSON object")
    return value


def _canonical_digest(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode()
    return _sha256_bytes(encoded)


def _assert_exact_json(value: object, *, path: str = "protocol") -> None:
    if value is None or type(value) in {bool, int, str}:
        return
    if type(value) is list:
        for index, item in enumerate(value):
            _assert_exact_json(item, path=f"{path}[{index}]")
        return
    if type(value) is dict:
        if any(type(key) is not str for key in value):
            raise TypeError(f"{path} has a non-exact string key")
        for key, item in value.items():
            _assert_exact_json(item, path=f"{path}.{key}")
        return
    raise TypeError(f"{path} contains a non-exact JSON value")


def _expected_protocol() -> dict[str, object]:
    """Return the source-bound experiment protocol, independent of artifact input."""
    return {
        "protocol_version": "1.0.0",
        "experiment_id": "task6-v3.3-confirmatory",
        "seeds": [503, 607, 709, 811, 907, 1009, 1103, 1201],
        "budgets": {
            "proposal": 24,
            "query": 24,
            "logical_time": 24,
            "wall_time_ms": 120000,
        },
        "maximum_confirmatory_attempts": 1,
        "policies": {
            "fixed": "1.0.0",
            "random": "1.0.0",
            "adaptive": "3.0.0",
            "cached_llm": "1.0.0",
        },
        "metrics": {
            "app_scam_mule": {
                "primary_outcome": "net_settled_value_rate",
                "minimum_delta": "0.10",
                "no_metric_switching": True,
            },
            "card_testing_cnp": {
                "primary_outcome": "valid_yield",
                "minimum_delta": "0.10",
                "no_metric_switching": True,
            },
        },
        "negative_control": {
            "family": "agentic_intent_abuse",
            "policy_cells": ["fixed", "random", "adaptive"],
            "primary_outcome": "valid_yield",
            "minimum_delta": "0.10",
            "expected_observed_delta": "0",
            "support_expected": False,
            "included_in_supported_family_count": False,
            "same_seeds_and_budgets_as_targets": True,
        },
        "network": {
            "allowed_calls": 0,
            "cached_llm_required": True,
        },
        "uncertainty": {
            "method": "exact_paired_sign_resampling_reference_interval",
            "reported_values": [
                "per_seed_deltas",
                "mean_per_seed_delta",
                "reference_interval_95",
            ],
            "role": "descriptive_only",
            "post_hoc_gate": False,
        },
        "fairness": {
            "shared_default_first_proposal": True,
            "shared_default_policies": ["random", "adaptive"],
            "identical_proposal_query_logical_caps": True,
            "identical_wall_time_caps": True,
        },
        "stopping_rule": _STOPPING_RULE,
        "approval_boundary": dict(_EXECUTION_BOUNDARY),
    }


def _validate_protocol(value: object) -> dict[str, object]:
    _assert_exact_json(value)
    if type(value) is not dict or value != _expected_protocol():
        raise RuntimeError("preregistration protocol differs from the frozen source protocol")
    return cast(dict[str, object], value)


def _exact_hex(label: str, value: object, *, length: int) -> str:
    if type(value) is not str or len(value) != length or not set(value) <= _HEX:
        raise TypeError(f"{label} must be exact lowercase hexadecimal text")
    return value


def _require_exact_fields(
    label: str,
    value: object,
    expected: frozenset[str],
) -> dict[str, Any]:
    if type(value) is not dict or set(value) != expected:
        raise RuntimeError(f"{label} field set is not exact")
    if any(type(key) is not str for key in value):
        raise TypeError(f"{label} keys must be exact strings")
    return cast(dict[str, Any], value)


def _atomic_publish_result(
    path: Path,
    payload: bytes,
    *,
    before_publish: Callable[[], None] | None = None,
) -> None:
    """Durably publish once without replacing a concurrently-created result."""
    if not isinstance(path, Path):
        raise TypeError("result path must be a pathlib.Path")
    if type(payload) is not bytes:
        raise TypeError("result payload must be exact bytes")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    descriptor_open = True
    try:
        with os.fdopen(descriptor, "wb") as stream:
            descriptor_open = False
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        if before_publish is not None:
            before_publish()
        # A same-directory hard link is atomic and fails with EEXIST instead of replacing.
        os.link(temporary, path)
        temporary.unlink()
        try:
            directory_descriptor = os.open(path.parent, os.O_RDONLY)
        except OSError as error:
            if error.errno not in _UNSUPPORTED_DIRECTORY_FSYNC:
                raise ResultPublicationDurabilityError(path, error) from error
            return
        try:
            # Some filesystems do not expose directory fsync.
            try:
                os.fsync(directory_descriptor)
            except OSError as error:
                if error.errno not in _UNSUPPORTED_DIRECTORY_FSYNC:
                    raise ResultPublicationDurabilityError(path, error) from error
        finally:
            os.close(directory_descriptor)
    finally:
        if descriptor_open:
            os.close(descriptor)
        temporary.unlink(missing_ok=True)


def _environment_document() -> dict[str, object]:
    installed = sorted(
        (
            {
                "name": (distribution.metadata.get("Name") or "").lower().replace("_", "-"),
                "version": distribution.version,
            }
            for distribution in importlib.metadata.distributions()
        ),
        key=lambda item: (item["name"], item["version"]),
    )
    return {
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "python_cache_tag": sys.implementation.cache_tag,
        "platform": platform.platform(),
        "installed_distributions": installed,
        "installed_distributions_digest": _canonical_digest(installed),
    }


def _git_output(arguments: list[str], *, text: bool = True) -> str | bytes:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=text,
    )
    output: object = completed.stdout
    if type(output) not in {str, bytes}:
        raise TypeError("git output was not exact text or bytes")
    return cast(str | bytes, output)


def _tracked_paths(commit: str) -> tuple[str, ...]:
    return tuple(_git_tree_records(commit))


def _git_tree_records(commit: str) -> dict[str, dict[str, str]]:
    output = cast(
        bytes,
        _git_output(["ls-tree", "-r", "-z", "--full-tree", commit], text=False),
    )
    records: dict[str, dict[str, str]] = {}
    for record in output.split(b"\0"):
        if not record:
            continue
        metadata, separator, encoded_path = record.partition(b"\t")
        if separator != b"\t":
            raise RuntimeError("Git tree record is malformed")
        parts = metadata.decode("ascii").split(" ")
        if len(parts) != 3:
            raise RuntimeError("Git tree metadata is malformed")
        mode, object_type, object_id = parts
        path = encoded_path.decode("utf-8")
        if path in records:
            raise RuntimeError("Git tree contains a duplicate path")
        records[path] = {
            "git_mode": mode,
            "object_type": object_type,
            "git_object_id": object_id,
        }
    return records


def _is_behavior_affecting_path(path: str) -> bool:
    pure = Path(path)
    name = pure.name
    root_config = (
        "/" not in path
        and (
            path == "pyproject.toml"
            or path in _LOCK_FILES
            or path == ".python-version"
            or path.startswith("requirements")
            or pure.suffix in {".cfg", ".ini", ".toml", ".yaml", ".yml"}
        )
    )
    return (
        path.startswith("src/")
        or path.startswith("scripts/")
        or path.startswith("fixtures/")
        or root_config
        or name in {"sitecustomize.py", "usercustomize.py"}
        or pure.suffix == ".pth"
        or path
        in {
            "docs/experiments/task6-v3-cached-llm-replay.json",
            "docs/experiments/task6-v3-cancellation.json",
            "docs/experiments/task6-v3.1-cancellation.json",
            "docs/experiments/task6-v3.2-cancellation.json",
        }
    )


def _behavior_manifest_document(commit: str) -> dict[str, object]:
    records = _git_tree_records(commit)
    entries: dict[str, dict[str, str]] = {}
    for path, record in records.items():
        if not _is_behavior_affecting_path(path):
            continue
        if record["object_type"] != "blob" or record["git_mode"] not in {
            "100644",
            "100755",
        }:
            raise RuntimeError(
                f"behavior path must be a regular tracked file, not symlink/submodule: {path}"
            )
        content = cast(
            bytes,
            _git_output(["cat-file", "blob", record["git_object_id"]], text=False),
        )
        entries[path] = {**record, "content_sha256": _sha256_bytes(content)}
    return {"entries": entries, "digest": _canonical_digest(entries)}


def _validate_manifest_document(value: object) -> dict[str, dict[str, str]]:
    document = _require_exact_fields(
        "behavior manifest",
        value,
        frozenset({"entries", "digest"}),
    )
    entries = document["entries"]
    if type(entries) is not dict or any(type(path) is not str for path in entries):
        raise RuntimeError("behavior manifest path set is not exact")
    checked: dict[str, dict[str, str]] = {}
    for path, raw_entry in entries.items():
        entry = _require_exact_fields(
            f"behavior manifest entry {path}",
            raw_entry,
            frozenset(
                {"git_mode", "object_type", "git_object_id", "content_sha256"}
            ),
        )
        if entry["git_mode"] not in {"100644", "100755"}:
            raise RuntimeError(f"behavior path is not a regular file mode: {path}")
        if entry["object_type"] != "blob":
            raise RuntimeError(f"behavior path is not a regular blob: {path}")
        _exact_hex("Git object ID", entry["git_object_id"], length=40)
        _exact_hex("content SHA-256", entry["content_sha256"], length=64)
        checked[path] = cast(dict[str, str], entry)
    expected_digest = _canonical_digest(entries)
    if document["digest"] != expected_digest:
        raise RuntimeError("behavior manifest digest does not match its exact entries")
    return checked


def _validate_matching_manifests(source: object, observed: object) -> None:
    source_entries = _validate_manifest_document(source)
    observed_entries = _validate_manifest_document(observed)
    if source_entries != observed_entries:
        raise RuntimeError("execution behavior manifest path, mode, type, or hash changed")


def _filesystem_behavior_paths(root: Path) -> set[str]:
    ignored_directories = {
        ".git",
        ".venv",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        "__pycache__",
        ".superpowers",
        ".codex",
        ".agents",
        "tests",
        "validation_spike",
    }
    paths: set[str] = set()
    for current, directories, files in os.walk(root, followlinks=False):
        current_path = Path(current)
        kept: list[str] = []
        for directory in directories:
            directory_path = current_path / directory
            relative = directory_path.relative_to(root).as_posix()
            if directory in ignored_directories or relative.split("/")[0] in ignored_directories:
                continue
            if directory_path.is_symlink() and (
                relative in {"src", "scripts", "fixtures"}
                or relative.startswith(("src/", "scripts/", "fixtures/"))
            ):
                raise RuntimeError(f"behavior directory is a filesystem symlink: {relative}")
            kept.append(directory)
        directories[:] = kept
        for name in files:
            if name.endswith((".pyc", ".pyo")):
                continue
            path = current_path / name
            relative = path.relative_to(root).as_posix()
            if _is_behavior_affecting_path(relative):
                paths.add(relative)
    return paths


def _validate_filesystem_behavior_paths(
    entries: object,
    *,
    root: Path = ROOT,
) -> None:
    if type(entries) is not dict or any(type(path) is not str for path in entries):
        raise RuntimeError("filesystem behavior path set is not exact")
    expected_paths = set(entries)
    observed_paths = _filesystem_behavior_paths(root)
    if observed_paths != expected_paths:
        raise RuntimeError("filesystem behavior path set differs from approved manifest")
    for relative, raw_entry in entries.items():
        path = root / relative
        if path.is_symlink() or not path.is_file():
            raise RuntimeError(f"behavior path is not a regular filesystem file: {relative}")
        entry = cast(dict[str, str], raw_entry)
        expected_executable = entry["git_mode"] == "100755"
        observed_executable = bool(path.stat().st_mode & 0o100)
        if observed_executable != expected_executable:
            raise RuntimeError(f"behavior path filesystem mode changed: {relative}")
        if _sha256_file(path) != entry["content_sha256"]:
            raise RuntimeError(f"behavior path content changed: {relative}")


def _validate_python_customization_modules(
    entries: object,
    *,
    root: Path = ROOT,
) -> None:
    if type(entries) is not dict:
        raise RuntimeError("Python customization manifest is not exact")
    resolved_root = root.resolve()
    for module_name in ("sitecustomize", "usercustomize"):
        module = sys.modules.get(module_name)
        if module is None:
            continue
        source = getattr(module, "__file__", None)
        if type(source) is not str:
            raise RuntimeError(f"loaded {module_name} has no exact source path")
        path = Path(source)
        try:
            relative = path.resolve().relative_to(resolved_root).as_posix()
        except ValueError as error:
            raise RuntimeError(
                f"loaded {module_name} is outside the approved customization manifest"
            ) from error
        if relative not in entries or path.is_symlink() or not path.is_file():
            raise RuntimeError(f"loaded {module_name} is not an approved regular file")
        entry = cast(dict[str, str], entries[relative])
        if _sha256_file(path) != entry["content_sha256"]:
            raise RuntimeError(f"loaded {module_name} content is not approved")


def _source_freeze_document(source_commit: str) -> dict[str, object]:
    if type(source_commit) is not str or len(source_commit) != 40:
        raise TypeError("source commit must be exact full Git object ID")
    manifest = _behavior_manifest_document(source_commit)
    entries = cast(dict[str, dict[str, str]], manifest["entries"])
    present_locks = tuple(path for path in _LOCK_FILES if path in entries)
    if present_locks:
        lock_file: dict[str, object] = {
            "path": present_locks[0],
            "status": "tracked",
            "sha256": entries[present_locks[0]]["content_sha256"],
        }
    else:
        lock_file = {"path": None, "status": "explicitly_absent"}
    git_tree = cast(str, _git_output(["rev-parse", f"{source_commit}^{{tree}}"])).strip()
    return {
        "source_commit": source_commit,
        "git_tree": git_tree,
        "behavior_manifest": manifest,
        "lock_file": lock_file,
        "environment": _environment_document(),
    }


def _head_commit() -> str:
    return cast(str, _git_output(["rev-parse", "HEAD"])).strip()


def _require_clean_worktree() -> None:
    status = cast(str, _git_output(["status", "--porcelain"])).strip()
    if status:
        raise RuntimeError("v3.3 verification requires a clean Git worktree")


def _verify_source_freeze(artifact: dict[str, Any]) -> None:
    source = _require_exact_fields(
        "v3.3 source freeze",
        artifact.get("source_freeze"),
        frozenset(
            {"source_commit", "git_tree", "behavior_manifest", "lock_file", "environment"}
        ),
    )
    source_commit = source.get("source_commit")
    checked_source_commit = _exact_hex(
        "v3.3 source commit", source_commit, length=40
    )
    ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", checked_source_commit, "HEAD"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if ancestor.returncode != 0:
        raise RuntimeError("v3.3 source commit is not an ancestor of this checkout")
    observed = _source_freeze_document(checked_source_commit)
    if observed != source:
        raise RuntimeError("v3.3 source tree, manifest, or environment binding changed")
    execution_manifest = _behavior_manifest_document(_head_commit())
    _validate_matching_manifests(source["behavior_manifest"], execution_manifest)
    source_entries = _validate_manifest_document(source["behavior_manifest"])
    _validate_filesystem_behavior_paths(source_entries)
    _validate_python_customization_modules(source_entries)
    _require_clean_worktree()


def _verify_cancelled_predecessors() -> None:
    v3 = _load_exact_json(CANCELLATION_PATH)
    v31 = _load_exact_json(V31_CANCELLATION_PATH)
    v32 = _load_exact_json(V32_CANCELLATION_PATH)
    if v3.get("status") != "cancelled_before_execution":
        raise RuntimeError("v3 cancellation record is not canonical")
    if v31.get("status") != "cancelled_before_execution":
        raise RuntimeError("v3.1 cancellation record is not canonical")
    if v32.get("status") != "cancelled_before_execution":
        raise RuntimeError("v3.2 cancellation record is not canonical")
    if (
        CANCELLED_RESULT_PATH.exists()
        or V31_RESULT_PATH.exists()
        or V32_RESULT_PATH.exists()
    ):
        raise RuntimeError("a cancelled confirmatory result must remain absent")


def _require_recording_preconditions() -> None:
    _require_clean_worktree()
    if RESULT_PATH.exists():
        raise RuntimeError("local v3.3 result exists; refusing an accidental rerun")


def _validate_external_approval(
    *,
    approved_freeze_commit: object,
    approved_prereg_sha256: object,
    preregistration_path: Path = PREREGISTRATION_PATH,
) -> dict[str, str]:
    commit = _exact_hex("approved freeze commit", approved_freeze_commit, length=40)
    preregistration_sha = _exact_hex(
        "approved preregistration SHA-256",
        approved_prereg_sha256,
        length=64,
    )
    if _head_commit() != commit:
        raise RuntimeError("HEAD is not the exact approved freeze commit")
    if not preregistration_path.is_file() or preregistration_path.is_symlink():
        raise RuntimeError("approved preregistration must be an exact regular file")
    if _sha256_file(preregistration_path) != preregistration_sha:
        raise RuntimeError("preregistration bytes do not match the approved SHA-256")
    return {
        "approved_freeze_commit": commit,
        "approved_prereg_sha256": preregistration_sha,
    }


_PROVENANCE_FIELDS = frozenset(
    {
        "evaluator_code_digest",
        "contract_digest",
        "bounds_digest",
        "hidden_template_digest",
        "background_digest",
        "population_digest",
        "evaluator_digest",
        "defender_digest",
        "disclosure_profile_digest",
    }
)


def _validate_preregistration_schema(artifact: object) -> dict[str, Any]:
    _assert_exact_json(artifact, path="preregistration")
    document = _require_exact_fields(
        "v3.3 preregistration",
        artifact,
        frozenset(
            {
                "schema_version",
                "status",
                "purpose",
                "source_freeze",
                "cancellation_record",
                "protocol",
                "policy_bindings",
                "frozen_benchmark",
                "families",
                "negative_control_provenance",
                "cached_replay",
            }
        ),
    )
    if document["schema_version"] != "1.0.0":
        raise RuntimeError("v3.3 preregistration schema version changed")
    if document["status"] != _SOURCE_STATUS or document["purpose"] != _PURPOSE:
        raise RuntimeError("v3.3 preregistration status or purpose changed")
    _validate_protocol(document["protocol"])
    cancellation = _require_exact_fields(
        "v3.3 cancellation binding",
        document["cancellation_record"],
        frozenset(
            {
                "v3_path",
                "v3_sha256",
                "v3_1_path",
                "v3_1_sha256",
                "v3_2_path",
                "v3_2_sha256",
                "cancelled_results_absent",
                "reserved_seeds_unused",
            }
        ),
    )
    if cancellation["cancelled_results_absent"] is not True or cancellation[
        "reserved_seeds_unused"
    ] is not True:
        raise RuntimeError("v3.3 cancellation state is not exact")
    expected_cancellation_paths = {
        "v3_path": "docs/experiments/task6-v3-cancellation.json",
        "v3_1_path": "docs/experiments/task6-v3.1-cancellation.json",
        "v3_2_path": "docs/experiments/task6-v3.2-cancellation.json",
    }
    for field, expected in expected_cancellation_paths.items():
        if cancellation[field] != expected:
            raise RuntimeError("v3.3 cancellation path binding changed")
    for field in ("v3_sha256", "v3_1_sha256", "v3_2_sha256"):
        _exact_hex(field, cancellation[field], length=64)

    policies = _require_exact_fields(
        "v3.3 policy bindings",
        document["policy_bindings"],
        frozenset({"fixed", "random", "adaptive", "cached_llm"}),
    )
    expected_versions = cast(dict[str, str], _expected_protocol()["policies"])
    for name, raw_binding in policies.items():
        binding = _require_exact_fields(
            f"v3.3 policy binding {name}",
            raw_binding,
            frozenset({"version", "code_digest", "callable_digest"}),
        )
        if binding["version"] != expected_versions[name]:
            raise RuntimeError("v3.3 policy version binding changed")
        _exact_hex("policy code digest", binding["code_digest"], length=64)
        _exact_hex("policy callable digest", binding["callable_digest"], length=64)

    benchmark = _require_exact_fields(
        "v3.3 benchmark binding",
        document["frozen_benchmark"],
        frozenset(
            {"population_digest", "defender_digest", "disclosure_profile_digest"}
        ),
    )
    for value in benchmark.values():
        _exact_hex("benchmark digest", value, length=64)

    families = _require_exact_fields(
        "v3.3 family bindings",
        document["families"],
        frozenset({"app_scam_mule", "card_testing_cnp"}),
    )
    for family, raw_family in families.items():
        family_document = _require_exact_fields(
            f"v3.3 family binding {family}",
            raw_family,
            frozenset({"provenance"}),
        )
        provenance = _require_exact_fields(
            f"v3.3 family provenance {family}",
            family_document["provenance"],
            _PROVENANCE_FIELDS,
        )
        for value in provenance.values():
            _exact_hex("family provenance digest", value, length=64)
    negative = _require_exact_fields(
        "v3.3 negative-control binding",
        document["negative_control_provenance"],
        frozenset({"provenance"}),
    )
    negative_provenance = _require_exact_fields(
        "v3.3 negative-control provenance",
        negative["provenance"],
        _PROVENANCE_FIELDS,
    )
    for value in negative_provenance.values():
        _exact_hex("negative-control provenance digest", value, length=64)

    cache = _require_exact_fields(
        "v3.3 cached replay binding",
        document["cached_replay"],
        frozenset(
            {"path", "file_sha256", "preparation_seed", "preparation_budget"}
        ),
    )
    if (
        cache["path"] != "docs/experiments/task6-v3-cached-llm-replay.json"
        or cache["preparation_seed"] != 4
        or cache["preparation_budget"] != 24
    ):
        raise RuntimeError("v3.3 cached replay protocol changed")
    _exact_hex("cached replay digest", cache["file_sha256"], length=64)
    return document


def _runtime() -> tuple[
    dict[str, Any],
    Task6Experiment,
    SearchAuthority,
    RunGroupCapability,
    dict[str, EvaluatorCapability],
    EvaluatorCapability,
    dict[str, PolicyCapability],
    LLMPlannerPolicy,
    _NoNetworkClient,
]:
    artifact = _validate_preregistration_schema(_load_exact_json(PREREGISTRATION_PATH))
    cache_artifact = _load_exact_json(CACHE_PATH)
    if _sha256_file(CACHE_PATH) != artifact["cached_replay"]["file_sha256"]:
        raise RuntimeError("v3.3 cached replay artifact digest changed")
    protocol = _expected_protocol()
    if cache_artifact["development_seed"] in cast(list[int], protocol["seeds"]):
        raise RuntimeError("cache preparation seed overlaps the v3.3 holdout")

    experiment = build_task6_experiment(ROOT)
    authority = SearchAuthority()
    run_group = authority.issue_run_group("task6-v3.3-confirmatory")
    evaluators = {
        family: benchmark.issue_evaluator_capability(authority)
        for family, benchmark in experiment.benchmarks.items()
    }
    negative_evaluator = experiment.negative_control.issue_evaluator_capability(authority)
    no_network = _NoNetworkClient()
    cached_policy = LLMPlannerPolicy(
        no_network,
        replay_cache=cache_artifact["records"],
        require_cached_replay=True,
    )
    policy_objects: dict[str, Policy] = {
        "fixed": FixedPolicy(),
        "random": RandomPolicy(),
        "adaptive": AdaptiveTournamentPolicy(),
        "cached_llm": cast(Policy, cached_policy),
    }
    versions = cast(dict[str, str], protocol["policies"])
    policies = {
        name: authority.register_policy(policy, name=name, version=versions[name])
        for name, policy in policy_objects.items()
    }
    return (
        artifact,
        experiment,
        authority,
        run_group,
        evaluators,
        negative_evaluator,
        policies,
        cached_policy,
        no_network,
    )


def _verify_frozen_bindings(
    artifact: dict[str, Any],
    experiment: Task6Experiment,
    authority: SearchAuthority,
    evaluators: dict[str, EvaluatorCapability],
    negative_evaluator: EvaluatorCapability,
    policies: dict[str, PolicyCapability],
) -> None:
    _validate_preregistration_schema(artifact)
    _verify_cancelled_predecessors()
    _verify_source_freeze(artifact)
    if experiment.population_digest != artifact["frozen_benchmark"]["population_digest"]:
        raise RuntimeError("frozen v3.3 population changed")
    cancellation = artifact["cancellation_record"]
    cancellation_files = {
        "v3_sha256": CANCELLATION_PATH,
        "v3_1_sha256": V31_CANCELLATION_PATH,
        "v3_2_sha256": V32_CANCELLATION_PATH,
    }
    if any(
        _sha256_file(path) != cancellation[field]
        for field, path in cancellation_files.items()
    ):
        raise RuntimeError("frozen v3.3 cancellation record digest changed")
    for name, expected in artifact["policy_bindings"].items():
        policy = authority.policy_binding(policies[name])
        observed = {
            "version": policy.version,
            "code_digest": policy.code_digest,
            "callable_digest": policy.callable_digest,
        }
        if observed != expected:
            raise RuntimeError(
                f"frozen v3.3 policy binding changed: {name}; "
                f"expected={expected!r}; observed={observed!r}"
            )
    for family, expected in artifact["families"].items():
        evaluator = evaluators[family]
        contract = evaluator.evaluation_contract
        observed = {
            "evaluator_code_digest": evaluator.evaluator_code_digest,
            "contract_digest": contract.contract_digest,
            "bounds_digest": contract.bounds_digest,
            "hidden_template_digest": contract.hidden_template_digest,
            "background_digest": contract.background_digest,
            "population_digest": contract.population_digest,
            "evaluator_digest": contract.evaluator_digest,
            "defender_digest": contract.defender_digest,
            "disclosure_profile_digest": contract.disclosure_profile_digest,
        }
        if observed != expected["provenance"]:
            raise RuntimeError(f"frozen v3.3 evaluator provenance changed: {family}")
    negative_contract = negative_evaluator.evaluation_contract
    negative_observed = {
        "evaluator_code_digest": negative_evaluator.evaluator_code_digest,
        "contract_digest": negative_contract.contract_digest,
        "bounds_digest": negative_contract.bounds_digest,
        "hidden_template_digest": negative_contract.hidden_template_digest,
        "background_digest": negative_contract.background_digest,
        "population_digest": negative_contract.population_digest,
        "evaluator_digest": negative_contract.evaluator_digest,
        "defender_digest": negative_contract.defender_digest,
        "disclosure_profile_digest": negative_contract.disclosure_profile_digest,
    }
    if negative_observed != artifact["negative_control_provenance"]["provenance"]:
        raise RuntimeError("frozen v3.3 negative-control provenance changed")


def _valid_yield(results: tuple[SearchResult, ...]) -> Decimal:
    proposal_count = sum(len(result.trials) for result in results)
    if proposal_count == 0:
        return Decimal(0)
    approved_count = sum(
        trial.feedback.action is Action.APPROVE
        for result in results
        for trial in result.trials
    )
    return Decimal(approved_count) / Decimal(proposal_count)


def _run_negative_control(
    *,
    authority: SearchAuthority,
    run_group: RunGroupCapability,
    evaluator: EvaluatorCapability,
    policies: dict[str, PolicyCapability],
    seeds: tuple[int, ...],
    budget: int,
    wall_time_budget_ms: int,
) -> dict[str, object]:
    policy_names = ("fixed", "random", "adaptive")
    results = {
        name: tuple(
            AdaptiveSearch(
                evaluator_capability=evaluator,
                policy_capability=policies[name],
                run_group=run_group,
            ).search(
                seed=seed,
                budget=budget,
                wall_time_budget_ms=wall_time_budget_ms,
            )
            for seed in seeds
        )
        for name in policy_names
    }
    all_results = tuple(
        result for name in policy_names for result in results[name]
    )
    matched = bool(all_results) and all(
        result.proposal_budget == budget
        and result.query_budget == budget
        and result.logical_time_budget == budget
        and result.wall_time_budget_ms == wall_time_budget_ms
        and result.proposals_used == budget
        and result.queries_used == budget
        and result.logical_time_used == budget
        and not result.wall_time_exhausted
        and result.wall_time_overrun_ms == 0
        for result in all_results
    )
    random_yield = _valid_yield(results["random"])
    adaptive_yield = _valid_yield(results["adaptive"])
    observed_delta = adaptive_yield - random_yield
    return {
        "family": evaluator.evaluation_contract.family,
        "primary_outcome": "valid_yield",
        "minimum_delta": "0.10",
        "included_in_supported_family_count": False,
        "matched_budgets": matched,
        "network_call_count": 0,
        "random_valid_yield": str(random_yield),
        "adaptive_valid_yield": str(adaptive_yield),
        "observed_valid_yield_delta": str(observed_delta),
        "supported": matched and observed_delta >= Decimal("0.10"),
        "result_bindings": {
            name: [
                {
                    "seed": result.seed,
                    "result_id": result.result_id,
                    "result_seal": result.result_seal,
                    "canonical_document_digest": _canonical_digest(
                        result.canonical_document()
                    ),
                }
                for result in runs
            ]
            for name, runs in results.items()
        },
    }


def _confirmatory_gate(
    *,
    target_cells_bound: bool,
    target_matched_budgets: bool,
    target_network_call_count: int,
    target_supported_family_count: int,
    target_adaptive_claim: str,
    negative_control: dict[str, object],
) -> dict[str, object]:
    """Derive every support claim from targets and the frozen zero-delta control."""
    target_valid = (
        type(target_cells_bound) is bool
        and target_cells_bound
        and type(target_matched_budgets) is bool
        and target_matched_budgets
        and type(target_network_call_count) is int
        and target_network_call_count == 0
        and type(target_supported_family_count) is int
        and 0 <= target_supported_family_count <= 2
        and type(target_adaptive_claim) is str
        and target_adaptive_claim in {"supported", "not_supported"}
    )
    control_valid = (
        type(negative_control) is dict
        and negative_control.get("matched_budgets") is True
        and type(negative_control.get("network_call_count")) is int
        and negative_control.get("network_call_count") == 0
        and type(negative_control.get("observed_valid_yield_delta")) is str
        and negative_control.get("observed_valid_yield_delta") == "0"
        and negative_control.get("supported") is False
    )
    confirmatory_valid = target_valid and control_valid
    supported_count = target_supported_family_count if confirmatory_valid else 0
    claim_supported = (
        confirmatory_valid
        and supported_count == 2
        and target_adaptive_claim == "supported"
    )
    return {
        "confirmatory_valid": confirmatory_valid,
        "criterion_met": claim_supported,
        "supported_family_count": supported_count,
        "adaptive_claim": "supported" if claim_supported else "not_supported",
    }


def _metrics_document(metrics: PolicyMetrics) -> dict[str, object]:
    return {
        "proposal_count": metrics.proposal_count,
        "approved_count": metrics.approved_count,
        "valid_yield": str(metrics.valid_yield),
        "net_settled_value": str(metrics.net_settled_value),
        "adaptation_speed": str(metrics.adaptation_speed),
        "campaign_scale": metrics.campaign_scale,
    }


def _primary_value(result: SearchResult, outcome: PrimaryOutcome) -> Decimal:
    approved = tuple(
        trial for trial in result.trials if trial.feedback.action is Action.APPROVE
    )
    if outcome is PrimaryOutcome.VALID_YIELD:
        if not result.trials:
            return Decimal(0)
        return Decimal(len(approved)) / Decimal(len(result.trials))
    return sum(
        (trial.feedback.realized_value or Decimal(0) for trial in approved),
        Decimal(0),
    )


def _paired_delta(
    adaptive: Decimal,
    random: Decimal,
    outcome: PrimaryOutcome,
) -> Decimal:
    if outcome is not PrimaryOutcome.NET_SETTLED_VALUE_RATE:
        return adaptive - random
    if random == 0:
        return Decimal(0) if adaptive == 0 else Decimal(1)
    with localcontext() as context:
        context.prec = 28
        return (adaptive - random) / random


def _descriptive_uncertainty(
    adaptive: tuple[SearchResult, ...],
    random: tuple[SearchResult, ...],
    outcome: PrimaryOutcome,
) -> dict[str, object]:
    deltas = tuple(
        _paired_delta(
            _primary_value(adaptive_run, outcome),
            _primary_value(random_run, outcome),
            outcome,
        )
        for adaptive_run, random_run in zip(adaptive, random, strict=True)
    )
    with localcontext() as context:
        context.prec = 28
        observed_mean = sum(deltas, Decimal(0)) / Decimal(len(deltas))
        sign_means = sorted(
            sum(
                (
                    delta if mask & (1 << index) else -delta
                    for index, delta in enumerate(deltas)
                ),
                Decimal(0),
            )
            / Decimal(len(deltas))
            for mask in range(2 ** len(deltas))
        )
    lower = sign_means[int(Decimal("0.025") * Decimal(len(sign_means) - 1))]
    upper = sign_means[int(Decimal("0.975") * Decimal(len(sign_means) - 1))]
    return {
        "method": "exact_paired_sign_resampling_reference_interval",
        "role": "descriptive_only",
        "per_seed_deltas": [str(delta) for delta in deltas],
        "mean_per_seed_delta": str(observed_mean),
        "reference_interval_95": [str(lower), str(upper)],
    }


def _execute(
    artifact: dict[str, Any],
    authority: SearchAuthority,
    group: RunGroupCapability,
    evaluators: dict[str, EvaluatorCapability],
    negative_evaluator: EvaluatorCapability,
    policies: dict[str, PolicyCapability],
    cached_policy: LLMPlannerPolicy,
    no_network: _NoNetworkClient,
    external_approval: dict[str, str],
) -> dict[str, object]:
    protocol = _expected_protocol()
    seeds = tuple(cast(list[int], protocol["seeds"]))
    budgets = cast(dict[str, int], protocol["budgets"])
    budget = budgets["proposal"]
    if not (budget == budgets["query"] == budgets["logical_time"]):
        raise RuntimeError("source-bound discrete budgets do not match")
    wall_budget = budgets["wall_time_ms"]
    metrics = cast(dict[str, dict[str, object]], protocol["metrics"])
    thresholds = tuple(
        FamilyThreshold(
            family=family,
            primary_outcome=PrimaryOutcome(cast(str, details["primary_outcome"])),
            minimum_delta=Decimal(cast(str, details["minimum_delta"])),
            evaluation_contract=evaluators[family].evaluation_contract,
            evaluator_capability_id=evaluators[family].capability_id,
            evaluator_code_digest=evaluators[family].evaluator_code_digest,
        )
        for family, details in sorted(metrics.items())
    )
    issued_preregistration = authority.issue_preregistration(
        run_group=group,
        seeds=seeds,
        budget=budget,
        wall_time_budget_ms=wall_budget,
        thresholds=thresholds,
        policies=tuple(policies[name] for name in sorted(policies)),
    )
    results = {
        family: {
            name: tuple(
                AdaptiveSearch(
                    evaluator_capability=evaluators[family],
                    policy_capability=policies[name],
                    run_group=group,
                ).search(
                    seed=seed,
                    budget=budget,
                    wall_time_budget_ms=wall_budget,
                )
                for seed in seeds
            )
            for name in ("fixed", "random", "adaptive", "cached_llm")
        }
        for family in sorted(evaluators)
    }
    report = capability_delta_report(
        issued_preregistration,
        results,
        authority=authority,
    )
    audit = cached_policy.take_audit_records()
    if no_network.calls != 0 or len(audit) != 2 * len(seeds) * budget:
        raise RuntimeError("v3.3 cached LLM zero-network audit is incomplete")
    if any(record.call_status != "cache_success" for record in audit):
        raise RuntimeError("v3.3 cached LLM contains a replay miss")
    negative_control = _run_negative_control(
        authority=authority,
        run_group=group,
        evaluator=negative_evaluator,
        policies=policies,
        seeds=seeds,
        budget=budget,
        wall_time_budget_ms=wall_budget,
    )
    target_cells_bound = (
        {metric.family for metric in report.family_metrics}
        == set(artifact["families"])
        and set(results) == set(artifact["families"])
        and all(
            set(cells) == {"fixed", "random", "adaptive", "cached_llm"}
            and all(len(runs) == len(seeds) for runs in cells.values())
            for cells in results.values()
        )
    )
    confirmatory_claim = _confirmatory_gate(
        target_cells_bound=target_cells_bound,
        target_matched_budgets=report.matched_budgets,
        target_network_call_count=no_network.calls,
        target_supported_family_count=report.supported_family_count,
        target_adaptive_claim=report.adaptive_claim,
        negative_control=negative_control,
    )

    return {
        "schema_version": "1.0.0",
        "external_approval": external_approval,
        "preregistration_commit": external_approval["approved_freeze_commit"],
        "preregistration_file_sha256": _sha256_file(PREREGISTRATION_PATH),
        "preregistration_canonical_digest": _canonical_digest(artifact),
        "protocol": protocol,
        "matched_budgets": report.matched_budgets,
        **confirmatory_claim,
        "negative_control": negative_control,
        "families": {
            metric.family: {
                "primary_outcome": metric.primary_outcome.value,
                "minimum_delta": str(metric.minimum_delta),
                "observed_delta": str(metric.observed_delta),
                "supported": metric.supported
                and bool(confirmatory_claim["confirmatory_valid"]),
                "fixed": _metrics_document(metric.fixed),
                "random": _metrics_document(metric.random),
                "adaptive": _metrics_document(metric.adaptive),
                "cached_llm": _metrics_document(metric.cached_llm),
                "uncertainty": _descriptive_uncertainty(
                    results[metric.family]["adaptive"],
                    results[metric.family]["random"],
                    metric.primary_outcome,
                ),
            }
            for metric in report.family_metrics
        },
        "cached_llm_audit": {
            "attempt_count": len(audit),
            "cache_success_count": sum(
                record.call_status == "cache_success" for record in audit
            ),
            "network_call_count": no_network.calls,
            "audit_digest": _canonical_digest(
                [record.model_dump(mode="json") for record in audit]
            ),
        },
        "result_bindings": {
            family: {
                name: [
                    {
                        "seed": result.seed,
                        "result_id": result.result_id,
                        "result_seal": result.result_seal,
                        "canonical_document_digest": _canonical_digest(
                            result.canonical_document()
                        ),
                    }
                    for result in runs
                ]
                for name, runs in cells.items()
            }
            for family, cells in results.items()
        },
    }


def _parse_args(arguments: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--verify-only", action="store_true")
    mode.add_argument("--verify-source-only", action="store_true")
    mode.add_argument("--execute-confirmatory", action="store_true")
    parser.add_argument("--approved-freeze-commit")
    parser.add_argument("--approved-prereg-sha256")
    args = parser.parse_args(arguments)
    supplied_commit = args.approved_freeze_commit is not None
    supplied_preregistration = args.approved_prereg_sha256 is not None
    if supplied_commit != supplied_preregistration:
        parser.error("both external approval values must be supplied together")
    if args.execute_confirmatory and not supplied_commit:
        parser.error(
            "execute-confirmatory requires --approved-freeze-commit and "
            "--approved-prereg-sha256"
        )
    if args.verify_source_only and supplied_commit:
        parser.error("source-only verification does not accept freeze approval")
    return args


def main() -> None:
    args = _parse_args()
    if args.execute_confirmatory and not PREREGISTRATION_PATH.exists():
        raise RuntimeError("v3.3 preregistration is absent; execution is forbidden")
    if args.verify_source_only or not PREREGISTRATION_PATH.exists():
        _verify_cancelled_predecessors()
        if RESULT_PATH.exists():
            raise RuntimeError("v3.3 result must remain absent before preregistration")
        print(
            "v3.1 cancelled; v3.2 source is cancelled; verified Task 6 v3.3 source stage; "
            "awaiting external approval values --approved-freeze-commit and "
            "--approved-prereg-sha256; no holdout trial executed"
        )
        return
    external_approval: dict[str, str] | None = None
    if args.approved_freeze_commit is not None:
        external_approval = _validate_external_approval(
            approved_freeze_commit=args.approved_freeze_commit,
            approved_prereg_sha256=args.approved_prereg_sha256,
        )
    (
        artifact,
        experiment,
        authority,
        group,
        evaluators,
        negative_evaluator,
        policies,
        cached_policy,
        no_network,
    ) = _runtime()
    _verify_frozen_bindings(
        artifact,
        experiment,
        authority,
        evaluators,
        negative_evaluator,
        policies,
    )
    if args.verify_only:
        if RESULT_PATH.exists():
            raise RuntimeError("v3.3 result must be absent during freeze verification")
        if external_approval is None:
            print(
                "v3.1 and v3.2 cancelled; verified frozen Task 6 v3.3 bindings; "
                "awaiting external approval: "
                f"--approved-freeze-commit {_head_commit()} "
                f"--approved-prereg-sha256 {_sha256_file(PREREGISTRATION_PATH)}; "
                "no holdout trial executed"
            )
        else:
            print(
                "v3.1 and v3.2 cancelled; verified frozen Task 6 v3.3 bindings and "
                "supplied external approval; no holdout trial executed"
            )
        return

    if external_approval is None:
        raise RuntimeError("execute mode cannot self-select external approval values")
    _require_recording_preconditions()
    external_approval = _validate_external_approval(
        approved_freeze_commit=args.approved_freeze_commit,
        approved_prereg_sha256=args.approved_prereg_sha256,
    )
    document = _execute(
        artifact,
        authority,
        group,
        evaluators,
        negative_evaluator,
        policies,
        cached_policy,
        no_network,
        external_approval,
    )
    _atomic_publish_result(
        RESULT_PATH,
        (json.dumps(document, sort_keys=True, indent=2) + "\n").encode("utf-8"),
    )
    print(json.dumps(document["families"], sort_keys=True, indent=2))
    print(f"supported_family_count={document['supported_family_count']}")


if __name__ == "__main__":
    main()
