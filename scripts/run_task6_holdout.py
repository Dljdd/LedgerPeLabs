"""Prepare, verify, or explicitly execute Task 6 v3.4 evidence replication.

V3.4 changes instrumentation and evidence persistence only. Proposal algorithms, frozen
defender behavior, primary metrics, and thresholds remain those used by v3.3. The v3.3
artifact is preserved but canonically rejected because it omitted independently auditable
raw evidence.
"""

from __future__ import annotations

import argparse
import ast
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
)
from apar.redteam.benchmark import CampaignBenchmark, default_defender_rules  # noqa: E402
from apar.redteam.task6_experiment import (  # noqa: E402
    Task6Experiment,
    build_task6_experiment,
)
from apar.redteam.task6_verifier import (  # noqa: E402
    build_result_bundle_document,
    build_search_cell_document,
    canonical_digest,
    canonical_json_bytes,
    derive_artifact_scoped_provenance,
    strict_json_loads,
    verify_result_bundle,
)

PREREGISTRATION_PATH = ROOT / "docs/experiments/task6-v3.4-holdout-preregistration.json"
CACHE_PATH = ROOT / "docs/experiments/task6-v3-cached-llm-replay.json"
CANCELLATION_PATH = ROOT / "docs/experiments/task6-v3-cancellation.json"
V31_CANCELLATION_PATH = ROOT / "docs/experiments/task6-v3.1-cancellation.json"
V31_RESULT_PATH = ROOT / "docs/experiments/task6-v3.1-holdout-result.json"
V32_CANCELLATION_PATH = ROOT / "docs/experiments/task6-v3.2-cancellation.json"
V32_RESULT_PATH = ROOT / "docs/experiments/task6-v3.2-holdout-result.json"
V33_PREREGISTRATION_PATH = ROOT / "docs/experiments/task6-v3.3-holdout-preregistration.json"
V33_RESULT_PATH = ROOT / "docs/experiments/task6-v3.3-holdout-result.json"
V33_REJECTION_PATH = ROOT / "docs/experiments/task6-v3.3-postexecution-rejection.json"
RESULT_PATH = ROOT / "docs/experiments/task6-v3.4-holdout-result.json"
_LOCK_FILES = ("uv.lock", "poetry.lock", "Pipfile.lock", "requirements.lock")
_HEX = frozenset("0123456789abcdef")
_SOURCE_STATUS = "final_v3_4_frozen_before_evidence_replication"
_PURPOSE = (
    "One instrumentation-only evidence replication of the unchanged v3.3 "
    "family-agnostic finite-lattice frontier/UCB policy."
)
_STOPPING_RULE = (
    "If v3.4 fails either preregistered target threshold or its evidence-validity hard "
    "gate, no further Task 6 confirmatory attempt will be opened; later work is "
    "exploratory or belongs to Task 7 evaluation."
)
_V33_BASELINE_COMMIT = "c513f263536330e7104c8b6eb1c0e5da4ccba0b4"
_V33_RESULT_SHA256 = "78cfa7a8352b41c2f4ca34b67cde939d2e9ffdefd8f8f3f91ccb24ee1e05d7fd"
_V33_PROPOSAL_SHA256 = {
    "src/apar/redteam/llm_policy.py": (
        "8105a6788041f7d73b1afa571482f4b0ff3b15980f6c28769b701ab350936622"
    ),
    "src/apar/redteam/policies.py": (
        "c97ab7b263a493978cf901140a97f15874a34f8ff2ce54c84253e7baa998fb82"
    ),
    "src/apar/redteam/search.py": (
        "ee05348ab07a9852a68a3f6a477eeec7ad6837d94f187e6fbb97767220f60e89"
    ),
}
_V33_GENERATOR_SHA256 = {
    "src/apar/generators/__init__.py": (
        "c4b3cdb979f1ec154cd6d55b40317495f024be7e09f7d9b6f7a93101b99d2886"
    ),
    "src/apar/generators/campaigns.py": (
        "670b4a3ec358f82d88f9655bd41d878fbee11d4841ff264655554bae31c3b31a"
    ),
    "src/apar/generators/population.py": (
        "2e54862322980414098c17930ec95bd268372da8968a78384a5bd661bfdaa2e5"
    ),
    "src/apar/redteam/task6_experiment.py": (
        "a1367a8bb4310eeea2812a7d118ccb738ae1d9c32bfbc21c87413b1a869ce056"
    ),
}
_V33_DEFENDER_AST_SHA256 = (
    "e38ceaeea1c859f4281d415072d5a52476d2ce5f1390626c93ce8842ff2b5c19"
)
_DEFENDER_AST_NAMES = frozenset(
    {
        "DefenderRule",
        "DefenderRuleSet",
        "default_defender_rules",
        "_observable_features",
        "role_bound_settled_value",
    }
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
        "experiment_id": "task6-v3.4-evidence-replication",
        "replication_kind": "instrumentation_only_evidence_replication",
        "algorithm_retuned_after_v3_3": False,
        "seeds": [2601, 2707, 2801, 2903, 3001, 3109, 3203, 3301],
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
        "evidence_limits": {
            "expected_cell_count": 88,
            "trials_per_complete_cell": 24,
            "maximum_total_trials": 2112,
            "maximum_cached_llm_attempts": 384,
            "maximum_bundle_bytes": 33_554_432,
            "lossless": True,
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


def _defender_ast_digest(path: Path) -> str:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    selected = [
        node
        for node in tree.body
        if getattr(node, "name", None) in _DEFENDER_AST_NAMES
    ]
    if {getattr(node, "name", None) for node in selected} != set(_DEFENDER_AST_NAMES):
        raise RuntimeError("frozen defender AST selection is incomplete")
    rendered = "\n".join(
        ast.dump(node, include_attributes=False) for node in selected
    ).encode("utf-8")
    return _sha256_bytes(rendered)


def _behavior_equivalence_document() -> dict[str, object]:
    proposal_hashes = {
        path: _sha256_file(ROOT / path) for path in sorted(_V33_PROPOSAL_SHA256)
    }
    generator_hashes = {
        path: _sha256_file(ROOT / path) for path in sorted(_V33_GENERATOR_SHA256)
    }
    defender_ast = _defender_ast_digest(ROOT / "src/apar/redteam/benchmark.py")
    equivalent = (
        proposal_hashes == _V33_PROPOSAL_SHA256
        and generator_hashes == _V33_GENERATOR_SHA256
        and defender_ast == _V33_DEFENDER_AST_SHA256
    )
    return {
        "baseline_commit": _V33_BASELINE_COMMIT,
        "proposal_implementation_sha256": proposal_hashes,
        "v3_3_proposal_implementation_sha256": dict(_V33_PROPOSAL_SHA256),
        "generator_implementation_sha256": generator_hashes,
        "v3_3_generator_implementation_sha256": dict(_V33_GENERATOR_SHA256),
        "defender_ast_sha256": defender_ast,
        "v3_3_defender_ast_sha256": _V33_DEFENDER_AST_SHA256,
        "defender_rules": [
            rule.document() for rule in default_defender_rules().rules
        ],
        "equivalent": equivalent,
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
            "docs/experiments/task6-v3.3-holdout-preregistration.json",
            "docs/experiments/task6-v3.3-holdout-result.json",
            "docs/experiments/task6-v3.3-postexecution-rejection.json",
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


def _commit_is_ancestor(ancestor: str, descendant: str) -> bool:
    completed = subprocess.run(
        ["git", "merge-base", "--is-ancestor", ancestor, descendant],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode not in {0, 1}:
        raise RuntimeError("unable to validate approved result ancestry")
    return completed.returncode == 0


def _require_clean_worktree() -> None:
    status = cast(str, _git_output(["status", "--porcelain"])).strip()
    if status:
        raise RuntimeError("v3.4 freeze verification requires a clean Git worktree")


def _validate_postexecution_worktree_status(raw_status: bytes) -> None:
    """Permit exactly the newly published untracked v3.4 result, and nothing else."""
    if type(raw_status) is not bytes:
        raise TypeError("post-execution Git status must be exact bytes")
    expected = (
        b"?? docs/experiments/task6-v3.4-holdout-result.json\0"
    )
    if raw_status != expected:
        raise RuntimeError(
            "post-execution worktree must contain exactly the untracked v3.4 result"
        )


def _validate_postcommit_chronology(
    *,
    approved_result_commit: object,
    approved_result_sha256: object,
    preregistration_commit: object,
    approved_artifacts: object,
) -> None:
    """Bind one result-only commit to its exact preregistration parent and blob."""
    result_commit = _exact_hex(
        "approved result commit", approved_result_commit, length=40
    )
    result_sha = _exact_hex(
        "approved result SHA-256", approved_result_sha256, length=64
    )
    prereg_commit = _exact_hex(
        "preregistration commit", preregistration_commit, length=40
    )
    expected_path = "docs/experiments/task6-v3.4-holdout-result.json"
    if approved_artifacts != {expected_path: result_sha}:
        raise RuntimeError("approved result artifact path or SHA-256 differs")
    current_head = _head_commit()
    if not _commit_is_ancestor(result_commit, current_head):
        raise RuntimeError("approved result commit is not an ancestor of HEAD")
    parents = cast(
        str,
        _git_output(["rev-list", "--parents", "-n", "1", result_commit]),
    ).strip().split()
    if parents != [result_commit, prereg_commit]:
        raise RuntimeError(
            "result commit chronology requires the exact preregistration commit as its only parent"
        )
    changed = cast(
        bytes,
        _git_output(
            [
                "diff-tree",
                "--no-commit-id",
                "--name-status",
                "-r",
                "-z",
                result_commit,
            ],
            text=False,
        ),
    )
    if changed != b"A\0" + expected_path.encode("utf-8") + b"\0":
        raise RuntimeError("result commit changed paths differ from the approved artifact")
    record = _git_tree_records(result_commit).get(expected_path)
    if record is None or record["git_mode"] != "100644" or record["object_type"] != "blob":
        raise RuntimeError("approved result artifact is not one regular non-executable blob")
    content = cast(
        bytes,
        _git_output(["cat-file", "blob", record["git_object_id"]], text=False),
    )
    if _sha256_bytes(content) != result_sha:
        raise RuntimeError("approved result artifact SHA-256 differs from its committed blob")
    if RESULT_PATH.is_symlink() or not RESULT_PATH.is_file():
        raise RuntimeError("current approved result artifact is not one regular file")
    current = RESULT_PATH.read_bytes()
    if len(current) != len(content) or _sha256_bytes(current) != result_sha:
        raise RuntimeError("current approved result artifact size or SHA-256 differs")


def _verify_source_freeze(
    artifact: dict[str, Any],
    *,
    require_clean: bool = True,
) -> None:
    source = _require_exact_fields(
        "v3.4 source freeze",
        artifact.get("source_freeze"),
        frozenset(
            {"source_commit", "git_tree", "behavior_manifest", "lock_file", "environment"}
        ),
    )
    source_commit = source.get("source_commit")
    checked_source_commit = _exact_hex(
        "v3.4 source commit", source_commit, length=40
    )
    ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", checked_source_commit, "HEAD"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if ancestor.returncode != 0:
        raise RuntimeError("v3.4 source commit is not an ancestor of this checkout")
    observed = _source_freeze_document(checked_source_commit)
    if observed != source:
        raise RuntimeError("v3.4 source tree, manifest, or environment binding changed")
    execution_manifest = _behavior_manifest_document(_head_commit())
    _validate_matching_manifests(source["behavior_manifest"], execution_manifest)
    source_entries = _validate_manifest_document(source["behavior_manifest"])
    _validate_filesystem_behavior_paths(source_entries)
    _validate_python_customization_modules(source_entries)
    if require_clean:
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
    if V31_RESULT_PATH.exists() or V32_RESULT_PATH.exists():
        raise RuntimeError("a cancelled confirmatory result must remain absent")
    rejection = _load_exact_json(V33_REJECTION_PATH)
    expected_rejection = {
        "schema_version": "1.0.0",
        "status": "rejected_unverifiable",
        "result_commit": _V33_BASELINE_COMMIT,
        "result_path": "docs/experiments/task6-v3.3-holdout-result.json",
        "result_sha256": _V33_RESULT_SHA256,
        "result_preserved_byte_for_byte": True,
        "reviewer_reason": (
            "The result omits raw SearchResult documents, evaluator-owned per-trial "
            "execution traces, and individual cached-LLM audit attempts, so its "
            "aggregate capability claims cannot be independently reconstructed."
        ),
        "rejected_claims": [
            "adaptive_claim",
            "confirmatory_valid",
            "criterion_met",
            "supported_family_count",
        ],
        "replacement_preregistration_path": (
            "docs/experiments/task6-v3.4-holdout-preregistration.json"
        ),
        "replacement_result_path": "docs/experiments/task6-v3.4-holdout-result.json",
        "replacement_kind": "instrumentation_only_evidence_replication",
    }
    if rejection != expected_rejection:
        raise RuntimeError("v3.3 post-execution rejection record is not canonical")
    if not V33_RESULT_PATH.is_file() or V33_RESULT_PATH.is_symlink():
        raise RuntimeError("the preserved v3.3 result must be an exact regular file")
    if _sha256_file(V33_RESULT_PATH) != _V33_RESULT_SHA256:
        raise RuntimeError("the preserved v3.3 result bytes changed")


def _require_recording_preconditions() -> None:
    _require_clean_worktree()
    if RESULT_PATH.exists():
        raise RuntimeError("local v3.4 result exists; refusing an accidental rerun")


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


def _contract_document(evaluator: EvaluatorCapability) -> dict[str, object]:
    contract = evaluator.evaluation_contract
    return {
        "family": contract.family,
        "bounds_digest": contract.bounds_digest,
        "hidden_template_digest": contract.hidden_template_digest,
        "background_digest": contract.background_digest,
        "population_digest": contract.population_digest,
        "evaluator_digest": contract.evaluator_digest,
        "defender_digest": contract.defender_digest,
        "disclosure_profile": {
            "profile_id": contract.disclosure_profile.profile_id,
            "expose_realized_value": contract.disclosure_profile.expose_realized_value,
            "profile_digest": contract.disclosure_profile_digest,
        },
        "contract_digest": contract.contract_digest,
    }


def _preregistered_evidence_context(
    benchmark: CampaignBenchmark,
    evaluator: EvaluatorCapability,
) -> dict[str, object]:
    defender = default_defender_rules()
    return {
        "public_bounds": benchmark.public_bounds.document(),
        "evaluation_contract": _contract_document(evaluator),
        "evaluator_code_digest": evaluator.evaluator_code_digest,
        "defender": {
            "version": defender.version,
            "rules": [rule.document() for rule in defender.rules],
            "defender_digest": defender.defender_digest,
        },
    }


def _runtime_evidence_context(
    preregistered: dict[str, object],
    evaluator: EvaluatorCapability,
) -> dict[str, object]:
    if preregistered.get("evaluator_code_digest") != evaluator.evaluator_code_digest:
        raise RuntimeError("runtime evaluator code differs from preregistered context")
    return {
        "public_bounds": preregistered["public_bounds"],
        "evaluation_contract": preregistered["evaluation_contract"],
        "evaluator_binding": {
            "capability_id": evaluator.capability_id,
            "code_digest": evaluator.evaluator_code_digest,
        },
        "defender": preregistered["defender"],
    }


def _runtime_verification_contexts(
    artifact: dict[str, Any],
    evaluators: dict[str, EvaluatorCapability],
    negative_evaluator: EvaluatorCapability,
) -> dict[str, object]:
    preregistered = artifact["evidence_contexts"]
    return {
        "targets": {
            family: _runtime_evidence_context(
                preregistered["targets"][family], evaluators[family]
            )
            for family in sorted(evaluators)
        },
        "negative_control": _runtime_evidence_context(
            preregistered["negative_control"], negative_evaluator
        ),
    }


def _bootstrap_runtime() -> tuple[
    Task6Experiment,
    SearchAuthority,
    RunGroupCapability,
    dict[str, EvaluatorCapability],
    EvaluatorCapability,
    dict[str, PolicyCapability],
    LLMPlannerPolicy,
    _NoNetworkClient,
    dict[str, Any],
]:
    cache_artifact = _load_exact_json(CACHE_PATH)
    protocol = _expected_protocol()
    if cache_artifact.get("development_seed") in cast(list[int], protocol["seeds"]):
        raise RuntimeError("cache preparation seed overlaps the v3.4 replication")
    experiment = build_task6_experiment(ROOT)
    authority = SearchAuthority()
    group = authority.issue_run_group("task6-v3.4-evidence-replication")
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
        experiment,
        authority,
        group,
        evaluators,
        negative_evaluator,
        policies,
        cached_policy,
        no_network,
        cache_artifact,
    )


def _predecessor_evidence_document() -> dict[str, object]:
    return {
        "cancelled_before_execution": {
            "v3": {
                "path": "docs/experiments/task6-v3-cancellation.json",
                "sha256": _sha256_file(CANCELLATION_PATH),
            },
            "v3_1": {
                "path": "docs/experiments/task6-v3.1-cancellation.json",
                "sha256": _sha256_file(V31_CANCELLATION_PATH),
            },
            "v3_2": {
                "path": "docs/experiments/task6-v3.2-cancellation.json",
                "sha256": _sha256_file(V32_CANCELLATION_PATH),
            },
            "cancelled_results_absent": True,
            "reserved_seeds_unused": True,
        },
        "v3_3_preregistration": {
            "path": "docs/experiments/task6-v3.3-holdout-preregistration.json",
            "sha256": _sha256_file(V33_PREREGISTRATION_PATH),
        },
        "v3_3_result": {
            "path": "docs/experiments/task6-v3.3-holdout-result.json",
            "commit": _V33_BASELINE_COMMIT,
            "sha256": _V33_RESULT_SHA256,
            "status": "preserved_but_rejected_unverifiable",
        },
        "v3_3_rejection": {
            "path": "docs/experiments/task6-v3.3-postexecution-rejection.json",
            "sha256": _sha256_file(V33_REJECTION_PATH),
        },
    }


def _cached_replay_document(cache_artifact: dict[str, Any]) -> dict[str, object]:
    records = cache_artifact.get("records")
    if type(records) is not dict:
        raise RuntimeError("cached replay records are not an exact object")
    return {
        "path": "docs/experiments/task6-v3-cached-llm-replay.json",
        "file_sha256": _sha256_file(CACHE_PATH),
        "canonical_digest": _canonical_digest(cache_artifact),
        "schema_version": cache_artifact.get("schema_version"),
        "preparation_seed": cache_artifact.get("development_seed"),
        "preparation_budget": cache_artifact.get("budget"),
        "record_counts": cache_artifact.get("record_counts"),
        "record_count": len(records),
        "records_digest": _canonical_digest(records),
        "provider": "fixture",
        "model_id": "cached-default-v1",
        "policy_version": "1.0.0",
        "require_cached_replay": True,
        "network_calls_allowed": 0,
    }


def _result_publication_document() -> dict[str, object]:
    verifier_path = "src/apar/redteam/task6_verifier.py"
    result_path = "docs/experiments/task6-v3.4-holdout-result.json"
    return {
        "approved_artifact_paths": [result_path],
        "canonical_json_required": True,
        "atomic_exclusive_no_replace": True,
        "directory_eio_reports_published_recovery_state": True,
        "postexecution_mode": "exactly_one_untracked_result_and_no_other_change",
        "postcommit_mode": "result_commit_parent_is_exact_preregistration_commit",
        "verifier_path": verifier_path,
        "verifier_sha256": _sha256_file(ROOT / verifier_path),
        "verifier_calls_policy_search": False,
        "verification_input": "complete_raw_cells_not_summary",
        "deterministic_evaluator_replay_predeclared": False,
        "process_local_issuance_seals_portable": False,
    }


def _build_preregistration_document(source_commit: str) -> dict[str, object]:
    """Create the preregistration only; this path never invokes policy search."""
    _exact_hex("source commit", source_commit, length=40)
    (
        experiment,
        authority,
        _group,
        evaluators,
        negative_evaluator,
        policies,
        _cached_policy,
        no_network,
        cache_artifact,
    ) = _bootstrap_runtime()
    if no_network.calls != 0:
        raise RuntimeError("preregistration construction attempted network transport")
    policy_bindings = {}
    for name, capability in policies.items():
        binding = authority.policy_binding(capability)
        policy_bindings[name] = {
            "version": binding.version,
            "code_digest": binding.code_digest,
            "callable_digest": binding.callable_digest,
            "capability_id_scope": "process_local_nonportable_not_preregistered",
        }
    target_contexts = {
        family: _preregistered_evidence_context(
            experiment.benchmarks[family], evaluators[family]
        )
        for family in sorted(evaluators)
    }
    negative_context = _preregistered_evidence_context(
        experiment.negative_control, negative_evaluator
    )
    behavior = _behavior_equivalence_document()
    if behavior["equivalent"] is not True:
        raise RuntimeError("v3.4 behavior is not equivalent to the frozen v3.3 behavior")
    return {
        "schema_version": "1.0.0",
        "status": _SOURCE_STATUS,
        "purpose": _PURPOSE,
        "source_freeze": _source_freeze_document(source_commit),
        "predecessor_evidence": _predecessor_evidence_document(),
        "behavior_equivalence": behavior,
        "protocol": _expected_protocol(),
        "policy_bindings": policy_bindings,
        "frozen_benchmark": {
            "population_digest": experiment.population_digest,
            "defender_digest": default_defender_rules().defender_digest,
            "evaluator_instrumentation": "lossless_per_trial_trace_v1",
            "generator_semantics_changed_after_v3_3": False,
        },
        "evidence_contexts": {
            "targets": target_contexts,
            "negative_control": negative_context,
        },
        "cached_replay": _cached_replay_document(cache_artifact),
        "result_publication": _result_publication_document(),
    }


def _validate_preregistration_schema(
    artifact: object,
    *,
    verify_current_bindings: bool = True,
) -> dict[str, Any]:
    _assert_exact_json(artifact, path="preregistration")
    document = _require_exact_fields(
        "v3.4 preregistration",
        artifact,
        frozenset(
            {
                "schema_version",
                "status",
                "purpose",
                "source_freeze",
                "predecessor_evidence",
                "behavior_equivalence",
                "protocol",
                "policy_bindings",
                "frozen_benchmark",
                "evidence_contexts",
                "cached_replay",
                "result_publication",
            }
        ),
    )
    if document["schema_version"] != "1.0.0":
        raise RuntimeError("v3.4 preregistration schema version changed")
    if verify_current_bindings:
        if document["status"] != _SOURCE_STATUS or document["purpose"] != _PURPOSE:
            raise RuntimeError("v3.4 preregistration status or purpose changed")
        _validate_protocol(document["protocol"])
    elif (
        type(document["status"]) is not str
        or type(document["purpose"]) is not str
        or type(document["protocol"]) is not dict
    ):
        raise RuntimeError("historical v3.4 status, purpose, or protocol is malformed")
    if verify_current_bindings and document[
        "predecessor_evidence"
    ] != _predecessor_evidence_document():
        raise RuntimeError("v3.4 predecessor evidence binding changed")
    behavior = _require_exact_fields(
        "v3.4 behavior equivalence",
        document["behavior_equivalence"],
        frozenset(
            {
                "baseline_commit",
                "proposal_implementation_sha256",
                "v3_3_proposal_implementation_sha256",
                "generator_implementation_sha256",
                "v3_3_generator_implementation_sha256",
                "defender_ast_sha256",
                "v3_3_defender_ast_sha256",
                "defender_rules",
                "equivalent",
            }
        ),
    )
    if behavior["equivalent"] is not True:
        raise RuntimeError("v3.4 behavior equivalence to v3.3 is not exact")
    if verify_current_bindings:
        if behavior != _behavior_equivalence_document():
            raise RuntimeError("v3.4 behavior equivalence to v3.3 is not exact")
    elif (
        type(behavior["baseline_commit"]) is not str
        or behavior["proposal_implementation_sha256"]
        != behavior["v3_3_proposal_implementation_sha256"]
        or behavior["generator_implementation_sha256"]
        != behavior["v3_3_generator_implementation_sha256"]
        or behavior["defender_ast_sha256"]
        != behavior["v3_3_defender_ast_sha256"]
    ):
        raise RuntimeError("historical v3.4 behavior-equivalence evidence differs")
    if not verify_current_bindings:
        _exact_hex(
            "historical v3.3 baseline commit",
            behavior["baseline_commit"],
            length=40,
        )

    policies = _require_exact_fields(
        "v3.4 policy bindings",
        document["policy_bindings"],
        frozenset({"fixed", "random", "adaptive", "cached_llm"}),
    )
    expected_versions = (
        cast(dict[str, str], _expected_protocol()["policies"])
        if verify_current_bindings
        else None
    )
    for name, raw_binding in policies.items():
        binding = _require_exact_fields(
            f"v3.4 policy binding {name}",
            raw_binding,
            frozenset(
                {
                    "version",
                    "code_digest",
                    "callable_digest",
                    "capability_id_scope",
                }
            ),
        )
        if (
            expected_versions is not None
            and binding["version"] != expected_versions[name]
        ):
            raise RuntimeError("v3.4 policy version binding changed")
        if type(binding["version"]) is not str or not binding["version"]:
            raise RuntimeError("v3.4 policy version binding is malformed")
        if binding["capability_id_scope"] != (
            "process_local_nonportable_not_preregistered"
        ):
            raise RuntimeError("v3.4 policy capability scope changed")
        _exact_hex("policy code digest", binding["code_digest"], length=64)
        _exact_hex("policy callable digest", binding["callable_digest"], length=64)

    benchmark = _require_exact_fields(
        "v3.4 benchmark binding",
        document["frozen_benchmark"],
        frozenset(
            {
                "population_digest",
                "defender_digest",
                "evaluator_instrumentation",
                "generator_semantics_changed_after_v3_3",
            }
        ),
    )
    _exact_hex("population digest", benchmark["population_digest"], length=64)
    _exact_hex("defender digest", benchmark["defender_digest"], length=64)
    if (
        benchmark["evaluator_instrumentation"] != "lossless_per_trial_trace_v1"
        or benchmark["generator_semantics_changed_after_v3_3"] is not False
    ):
        raise RuntimeError("v3.4 instrumentation or generator-equivalence binding changed")
    contexts = _require_exact_fields(
        "v3.4 evidence contexts",
        document["evidence_contexts"],
        frozenset({"targets", "negative_control"}),
    )
    targets = _require_exact_fields(
        "v3.4 target contexts",
        contexts["targets"],
        frozenset({"app_scam_mule", "card_testing_cnp"}),
    )
    all_contexts = [*targets.values(), contexts["negative_control"]]
    for raw_context in all_contexts:
        context = _require_exact_fields(
            "v3.4 evidence context",
            raw_context,
            frozenset(
                {
                    "public_bounds",
                    "evaluation_contract",
                    "evaluator_code_digest",
                    "defender",
                }
            ),
        )
        _exact_hex("evaluator code digest", context["evaluator_code_digest"], length=64)
        bounds = cast(dict[str, object], context["public_bounds"])
        contract = cast(dict[str, object], context["evaluation_contract"])
        defender = _require_exact_fields(
            "v3.4 context defender",
            context["defender"],
            frozenset({"version", "rules", "defender_digest"}),
        )
        if contract.get("bounds_digest") != _canonical_digest(bounds):
            raise RuntimeError("v3.4 context bounds digest differs")
        if defender["defender_digest"] != _canonical_digest(
            {"version": defender["version"], "rules": defender["rules"]}
        ) or contract.get("defender_digest") != defender["defender_digest"]:
            raise RuntimeError("v3.4 context defender provenance differs")

    if verify_current_bindings:
        cache_artifact = _load_exact_json(CACHE_PATH)
        if document["cached_replay"] != _cached_replay_document(cache_artifact):
            raise RuntimeError("v3.4 cached replay configuration or digest changed")
    else:
        cached = _require_exact_fields(
            "historical v3.4 cached replay",
            document["cached_replay"],
            frozenset(
                {
                    "path",
                    "file_sha256",
                    "canonical_digest",
                    "schema_version",
                    "preparation_seed",
                    "preparation_budget",
                    "record_counts",
                    "record_count",
                    "records_digest",
                    "provider",
                    "model_id",
                    "policy_version",
                    "require_cached_replay",
                    "network_calls_allowed",
                }
            ),
        )
        for name in ("file_sha256", "canonical_digest", "records_digest"):
            _exact_hex(f"historical cache {name}", cached[name], length=64)
    if verify_current_bindings:
        if document["result_publication"] != _result_publication_document():
            raise RuntimeError("v3.4 result publication/verifier binding changed")
    else:
        publication = _require_exact_fields(
            "historical v3.4 result publication",
            document["result_publication"],
            frozenset(
                {
                    "approved_artifact_paths",
                    "canonical_json_required",
                    "atomic_exclusive_no_replace",
                    "directory_eio_reports_published_recovery_state",
                    "postexecution_mode",
                    "postcommit_mode",
                    "verifier_path",
                    "verifier_sha256",
                    "verifier_calls_policy_search",
                    "verification_input",
                    "deterministic_evaluator_replay_predeclared",
                    "process_local_issuance_seals_portable",
                }
            ),
        )
        if (
            publication["approved_artifact_paths"]
            != ["docs/experiments/task6-v3.4-holdout-result.json"]
            or publication["verifier_path"] != "src/apar/redteam/task6_verifier.py"
            or publication["verifier_calls_policy_search"] is not False
            or publication["verification_input"] != "complete_raw_cells_not_summary"
            or publication["process_local_issuance_seals_portable"] is not False
        ):
            raise RuntimeError("historical v3.4 result publication binding changed")
        _exact_hex(
            "historical v3.4 verifier SHA-256",
            publication["verifier_sha256"],
            length=64,
        )
    source = _require_exact_fields(
        "v3.4 source freeze",
        document["source_freeze"],
        frozenset(
            {"source_commit", "git_tree", "behavior_manifest", "lock_file", "environment"}
        ),
    )
    _exact_hex("source commit", source["source_commit"], length=40)
    _exact_hex("source tree", source["git_tree"], length=40)
    _validate_manifest_document(source["behavior_manifest"])
    return document


def _validate_portable_preregistration_schema(artifact: object) -> dict[str, Any]:
    """Validate frozen bytes while retaining the historical verifier binding."""
    return _validate_preregistration_schema(
        artifact,
        verify_current_bindings=False,
    )


def _git_regular_blob(commit: str, path: str) -> bytes:
    record = _git_tree_records(commit).get(path)
    if record is None or record["git_mode"] != "100644" or record["object_type"] != "blob":
        raise RuntimeError(f"historical artifact is not one regular blob: {path}")
    return cast(
        bytes,
        _git_output(["cat-file", "blob", record["git_object_id"]], text=False),
    )


def _validate_historical_preregistration(
    artifact: dict[str, Any],
    *,
    raw: bytes,
    preregistration_commit: str,
    approved_sha256: str,
) -> None:
    preregistration_path = (
        "docs/experiments/task6-v3.4-holdout-preregistration.json"
    )
    source = cast(dict[str, object], artifact["source_freeze"])
    source_commit = _exact_hex(
        "historical source commit", source["source_commit"], length=40
    )
    preregistration_commit = _exact_hex(
        "historical preregistration commit", preregistration_commit, length=40
    )
    approved_sha256 = _exact_hex(
        "historical preregistration SHA-256", approved_sha256, length=64
    )
    if PREREGISTRATION_PATH.is_symlink() or not PREREGISTRATION_PATH.is_file():
        raise RuntimeError("current preregistration artifact is not one regular file")
    if _sha256_bytes(raw) != approved_sha256:
        raise RuntimeError("current preregistration SHA-256 differs from approval")
    committed = _git_regular_blob(preregistration_commit, preregistration_path)
    if len(committed) != len(raw) or _sha256_bytes(committed) != approved_sha256:
        raise RuntimeError("historical preregistration size or SHA-256 differs")
    parents = cast(
        str,
        _git_output(
            ["rev-list", "--parents", "-n", "1", preregistration_commit]
        ),
    ).strip().split()
    if parents != [preregistration_commit, source_commit]:
        raise RuntimeError("preregistration commit is not directly based on its source freeze")
    changed = cast(
        bytes,
        _git_output(
            [
                "diff-tree",
                "--no-commit-id",
                "--name-status",
                "-r",
                "-z",
                preregistration_commit,
            ],
            text=False,
        ),
    )
    if changed != b"A\0" + preregistration_path.encode("utf-8") + b"\0":
        raise RuntimeError("preregistration commit is not an exact artifact-only freeze")
    observed_tree = cast(
        str,
        _git_output(["rev-parse", f"{source_commit}^{{tree}}"]),
    ).strip()
    if observed_tree != source["git_tree"]:
        raise RuntimeError("historical source tree differs from preregistration")
    observed_manifest = _behavior_manifest_document(source_commit)
    if observed_manifest != source["behavior_manifest"]:
        raise RuntimeError("historical behavior manifest differs from preregistration")
    entries = _validate_manifest_document(source["behavior_manifest"])
    verifier_path = cast(str, artifact["result_publication"]["verifier_path"])
    if (
        verifier_path not in entries
        or entries[verifier_path]["content_sha256"]
        != artifact["result_publication"]["verifier_sha256"]
    ):
        raise RuntimeError("historical raw verifier differs from preregistration")


def _load_frozen_cache_records(artifact: dict[str, Any]) -> dict[str, object]:
    cached = cast(dict[str, object], artifact["cached_replay"])
    source = cast(dict[str, object], artifact["source_freeze"])
    source_commit = cast(str, source["source_commit"])
    path = cast(str, cached["path"])
    raw = _git_regular_blob(source_commit, path)
    if _sha256_bytes(raw) != cached["file_sha256"]:
        raise RuntimeError("historical cached-replay file SHA-256 differs")
    loaded = strict_json_loads(raw, require_canonical=False)
    if type(loaded) is not dict:
        raise RuntimeError("historical cached replay is not one exact object")
    cache_artifact = cast(dict[str, Any], loaded)
    records = cache_artifact.get("records")
    if type(records) is not dict:
        raise RuntimeError("historical cached replay records are not an exact object")
    observed = {
        "path": path,
        "file_sha256": _sha256_bytes(raw),
        "canonical_digest": _canonical_digest(cache_artifact),
        "schema_version": cache_artifact.get("schema_version"),
        "preparation_seed": cache_artifact.get("development_seed"),
        "preparation_budget": cache_artifact.get("budget"),
        "record_counts": cache_artifact.get("record_counts"),
        "record_count": len(records),
        "records_digest": _canonical_digest(records),
        "provider": "fixture",
        "model_id": "cached-default-v1",
        "policy_version": "1.0.0",
        "require_cached_replay": True,
        "network_calls_allowed": 0,
    }
    if observed != cached:
        raise RuntimeError("historical cached-replay stable provenance differs")
    return cast(dict[str, object], records)


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
    artifact = _validate_preregistration_schema(
        strict_json_loads(PREREGISTRATION_PATH.read_bytes(), require_canonical=True)
    )
    (
        experiment,
        authority,
        run_group,
        evaluators,
        negative_evaluator,
        policies,
        cached_policy,
        no_network,
        _cache_artifact,
    ) = _bootstrap_runtime()
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
    *,
    require_clean: bool = True,
) -> None:
    _validate_preregistration_schema(artifact)
    _verify_cancelled_predecessors()
    _verify_source_freeze(artifact, require_clean=require_clean)
    if experiment.population_digest != artifact["frozen_benchmark"]["population_digest"]:
        raise RuntimeError("frozen v3.4 population changed")
    if artifact["frozen_benchmark"]["defender_digest"] != (
        default_defender_rules().defender_digest
    ):
        raise RuntimeError("frozen v3.4 defender changed")
    for name, expected in artifact["policy_bindings"].items():
        policy = authority.policy_binding(policies[name])
        observed = {
            "version": policy.version,
            "code_digest": policy.code_digest,
            "callable_digest": policy.callable_digest,
            "capability_id_scope": "process_local_nonportable_not_preregistered",
        }
        if observed != expected:
            raise RuntimeError(
                f"frozen v3.4 policy binding changed: {name}; "
                f"expected={expected!r}; observed={observed!r}"
            )
    target_contexts = artifact["evidence_contexts"]["targets"]
    for family, evaluator in evaluators.items():
        observed_context = _preregistered_evidence_context(
            experiment.benchmarks[family], evaluator
        )
        if observed_context != target_contexts[family]:
            raise RuntimeError(f"frozen v3.4 evaluator context changed: {family}")
    negative_observed = _preregistered_evidence_context(
        experiment.negative_control, negative_evaluator
    )
    if negative_observed != artifact["evidence_contexts"]["negative_control"]:
        raise RuntimeError("frozen v3.4 negative-control context changed")


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
    experiment: Task6Experiment,
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
    # Keep the authority-level preregistration checks from v3.3, but raw evidence and
    # independent verification below are the only basis for exported claims.
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
    authority.issue_preregistration(
        run_group=group,
        seeds=seeds,
        budget=budget,
        wall_time_budget_ms=wall_budget,
        thresholds=thresholds,
        policies=tuple(policies[name] for name in sorted(policies)),
    )
    if cached_policy.take_audit_records():
        raise RuntimeError("cached LLM audit buffer was not empty before execution")
    for benchmark in (*experiment.benchmarks.values(), experiment.negative_control):
        if benchmark.take_evaluation_traces():
            raise RuntimeError("evaluator trace buffer was not empty before execution")

    cells: list[dict[str, object]] = []
    defender = default_defender_rules()
    for family in sorted(evaluators):
        benchmark = experiment.benchmarks[family]
        for name in ("fixed", "random", "adaptive", "cached_llm"):
            for seed in seeds:
                result = AdaptiveSearch(
                    evaluator_capability=evaluators[family],
                    policy_capability=policies[name],
                    run_group=group,
                ).search(
                    seed=seed,
                    budget=budget,
                    wall_time_budget_ms=wall_budget,
                )
                audits = (
                    cached_policy.take_audit_records() if name == "cached_llm" else ()
                )
                cells.append(
                    build_search_cell_document(
                        cell_kind="target",
                        result=result,
                        public_bounds=benchmark.public_bounds,
                        evaluation_contract=benchmark.evaluation_contract,
                        policy_binding=authority.policy_binding(policies[name]),
                        defender=defender,
                        evaluation_traces=benchmark.take_evaluation_traces(),
                        llm_audit_records=audits,
                    )
                )
    negative_benchmark = experiment.negative_control
    for name in ("fixed", "random", "adaptive"):
        for seed in seeds:
            result = AdaptiveSearch(
                evaluator_capability=negative_evaluator,
                policy_capability=policies[name],
                run_group=group,
            ).search(
                seed=seed,
                budget=budget,
                wall_time_budget_ms=wall_budget,
            )
            cells.append(
                build_search_cell_document(
                    cell_kind="negative_control",
                    result=result,
                    public_bounds=negative_benchmark.public_bounds,
                    evaluation_contract=negative_benchmark.evaluation_contract,
                    policy_binding=authority.policy_binding(policies[name]),
                    defender=defender,
                    evaluation_traces=negative_benchmark.take_evaluation_traces(),
                    llm_audit_records=(),
                )
            )
    if cached_policy.take_audit_records():
        raise RuntimeError("cached LLM audit buffer contains unbound attempts")
    if no_network.calls != 0:
        raise RuntimeError("v3.4 cached planner attempted network transport")
    expected_contexts = _runtime_verification_contexts(
        artifact,
        evaluators,
        negative_evaluator,
    )
    policy_bindings: dict[str, object] = {
        name: authority.policy_binding(capability).model_dump(mode="json")
        for name, capability in policies.items()
    }
    cache_records = cast(dict[str, object], _load_exact_json(CACHE_PATH)["records"])
    document = build_result_bundle_document(
        protocol=protocol,
        cells=cells,
        expected_contexts=expected_contexts,
        expected_policy_bindings=policy_bindings,
        expected_llm_cache=cache_records,
        external_approval=external_approval,
        preregistration_canonical_digest=canonical_digest(artifact),
        network_call_count=no_network.calls,
    )
    verify_result_bundle(
        document,
        expected_protocol=protocol,
        expected_contexts=expected_contexts,
        expected_policy_bindings=policy_bindings,
        expected_llm_cache=cache_records,
        expected_external_approval=external_approval,
        expected_preregistration_canonical_digest=canonical_digest(artifact),
    )
    return document


def _verify_published_result_portably(
    *,
    expected_freeze_commit: str,
) -> tuple[dict[str, object], dict[str, object]]:
    """Verify frozen raw evidence without issuing any live process capability."""
    expected_freeze_commit = _exact_hex(
        "portable expected freeze commit", expected_freeze_commit, length=40
    )
    if RESULT_PATH.is_symlink() or not RESULT_PATH.is_file():
        raise RuntimeError("v3.4 result must be one exact regular file")
    result_raw = RESULT_PATH.read_bytes()
    loaded_result = strict_json_loads(result_raw, require_canonical=True)
    if type(loaded_result) is not dict:
        raise RuntimeError("v3.4 result must contain one exact JSON object")
    document = cast(dict[str, object], loaded_result)
    if PREREGISTRATION_PATH.is_symlink() or not PREREGISTRATION_PATH.is_file():
        raise RuntimeError("v3.4 preregistration must be one exact regular file")
    preregistration_raw = PREREGISTRATION_PATH.read_bytes()
    loaded_preregistration = strict_json_loads(
        preregistration_raw,
        require_canonical=True,
    )
    artifact = _validate_portable_preregistration_schema(loaded_preregistration)
    preregistration_sha = _sha256_bytes(preregistration_raw)
    if (
        document.get("preregistration_commit") != expected_freeze_commit
        or document.get("preregistration_file_sha256") != preregistration_sha
    ):
        raise RuntimeError("v3.4 result differs from its approved preregistration")
    _validate_historical_preregistration(
        artifact,
        raw=preregistration_raw,
        preregistration_commit=expected_freeze_commit,
        approved_sha256=preregistration_sha,
    )
    cache_records = _load_frozen_cache_records(artifact)
    contexts, policies = derive_artifact_scoped_provenance(
        document,
        preregistered_contexts=artifact["evidence_contexts"],
        preregistered_policy_bindings=artifact["policy_bindings"],
    )
    expected_approval = {
        "approved_freeze_commit": expected_freeze_commit,
        "approved_prereg_sha256": preregistration_sha,
    }
    summary = verify_result_bundle(
        document,
        expected_protocol=artifact["protocol"],
        expected_contexts=contexts,
        expected_policy_bindings=policies,
        expected_llm_cache=cache_records,
        expected_external_approval=expected_approval,
        expected_preregistration_canonical_digest=canonical_digest(artifact),
    )
    return document, summary


def _parse_args(arguments: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--verify-only", action="store_true")
    mode.add_argument("--verify-source-only", action="store_true")
    mode.add_argument("--prepare-preregistration", action="store_true")
    mode.add_argument("--execute-confirmatory", action="store_true")
    mode.add_argument("--verify-postexecution", action="store_true")
    mode.add_argument("--verify-postcommit", action="store_true")
    parser.add_argument("--approved-freeze-commit")
    parser.add_argument("--approved-prereg-sha256")
    parser.add_argument("--approved-result-commit")
    parser.add_argument("--approved-result-sha256")
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
    supplied_result_commit = args.approved_result_commit is not None
    supplied_result_sha = args.approved_result_sha256 is not None
    if supplied_result_commit != supplied_result_sha:
        parser.error("both approved result values must be supplied together")
    if args.verify_postcommit and not supplied_result_commit:
        parser.error(
            "verify-postcommit requires --approved-result-commit and "
            "--approved-result-sha256"
        )
    if supplied_result_commit and not args.verify_postcommit:
        parser.error("approved result values are accepted only by verify-postcommit")
    if (args.verify_source_only or args.prepare_preregistration) and supplied_commit:
        parser.error("source preparation/verification does not accept freeze approval")
    if (args.verify_postexecution or args.verify_postcommit) and supplied_commit:
        parser.error("post-execution verification derives approval from the result")
    return args


def main() -> None:
    args = _parse_args()
    if args.execute_confirmatory and not PREREGISTRATION_PATH.exists():
        raise RuntimeError("v3.4 preregistration is absent; execution is forbidden")
    if args.prepare_preregistration:
        _require_clean_worktree()
        _verify_cancelled_predecessors()
        if PREREGISTRATION_PATH.exists() or RESULT_PATH.exists():
            raise RuntimeError(
                "v3.4 preregistration/result must be absent before one-time preparation"
            )
        source_commit = _head_commit()
        document = _build_preregistration_document(source_commit)
        _validate_preregistration_schema(document)
        payload = canonical_json_bytes(document)
        _atomic_publish_result(PREREGISTRATION_PATH, payload)
        print(
            "prepared Task 6 v3.4 preregistration only; "
            f"source_commit={source_commit}; sha256={_sha256_bytes(payload)}; "
            "no holdout trial executed"
        )
        return
    if args.verify_source_only:
        _verify_cancelled_predecessors()
        if RESULT_PATH.exists():
            raise RuntimeError("v3.4 result must remain absent during source verification")
        behavior = _behavior_equivalence_document()
        if behavior["equivalent"] is not True:
            raise RuntimeError("v3.4 proposal or defender behavior differs from v3.3")
        print(
            "v3.2 source lineage retained; v3.3 result preserved and rejected; "
            "verified Task 6 v3.4 source stage; "
            "no holdout trial executed"
        )
        return
    if not PREREGISTRATION_PATH.exists():
        if args.verify_postexecution or args.verify_postcommit:
            raise RuntimeError("v3.4 preregistration is absent; post verification is impossible")
        _verify_cancelled_predecessors()
        if RESULT_PATH.exists():
            raise RuntimeError("v3.4 result must remain absent before preregistration")
        behavior = _behavior_equivalence_document()
        if behavior["equivalent"] is not True:
            raise RuntimeError("v3.4 proposal or defender behavior differs from v3.3")
        print(
            "v3.1/v3.2 cancelled lineage retained; v3.3 result preserved and rejected; "
            "verified Task 6 v3.4 source stage; "
            "no holdout trial executed"
        )
        return

    if args.verify_postexecution:
        status = cast(
            bytes,
            _git_output(
                ["status", "--porcelain=v1", "-z", "--untracked-files=all"],
                text=False,
            ),
        )
        _validate_postexecution_worktree_status(status)
        _document, summary = _verify_published_result_portably(
            expected_freeze_commit=_head_commit(),
        )
        print(
            "verified exact untracked Task 6 v3.4 raw-evidence result before commit; "
            f"confirmatory_valid={summary['confirmatory_valid']}"
        )
        return

    if args.verify_postcommit:
        _require_clean_worktree()
        loaded = strict_json_loads(RESULT_PATH.read_bytes(), require_canonical=True)
        if type(loaded) is not dict:
            raise RuntimeError("v3.4 result must contain an exact object")
        raw_result = cast(dict[str, object], loaded)
        preregistration_commit = cast(str, raw_result["preregistration_commit"])
        approved_result_sha = _exact_hex(
            "approved result SHA-256", args.approved_result_sha256, length=64
        )
        _validate_postcommit_chronology(
            approved_result_commit=args.approved_result_commit,
            approved_result_sha256=approved_result_sha,
            preregistration_commit=preregistration_commit,
            approved_artifacts={
                "docs/experiments/task6-v3.4-holdout-result.json": approved_result_sha
            },
        )
        _document, summary = _verify_published_result_portably(
            expected_freeze_commit=preregistration_commit,
        )
        print(
            "verified Task 6 v3.4 result-only commit chronology and raw evidence; "
            f"confirmatory_valid={summary['confirmatory_valid']}"
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
            raise RuntimeError("v3.4 result must be absent during freeze verification")
        if external_approval is None:
            print(
                "v3.1/v3.2 cancelled; v3.3 rejected; verified frozen Task 6 v3.4 "
                "bindings; "
                "awaiting external approval: "
                f"--approved-freeze-commit {_head_commit()} "
                f"--approved-prereg-sha256 {_sha256_file(PREREGISTRATION_PATH)}; "
                "no holdout trial executed"
            )
        else:
            print(
                "v3.1/v3.2 cancelled; v3.3 rejected; verified frozen Task 6 v3.4 "
                "bindings and "
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
        experiment,
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
        canonical_json_bytes(document),
    )
    summary = cast(dict[str, object], document["summary"])
    print(json.dumps(summary["families"], sort_keys=True, indent=2))
    print(f"supported_family_count={summary['supported_family_count']}")


if __name__ == "__main__":
    main()
