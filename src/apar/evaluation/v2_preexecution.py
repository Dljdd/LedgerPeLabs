"""Read-only admission checks for a never-started Defend v2 evaluation.

This module only reads public protocol material and defender source.  It has no
dependency on evaluator workers, population builders, or result publication.
"""

from __future__ import annotations

import ast
import hashlib
from collections.abc import Iterable
from pathlib import Path
from typing import Literal

from pydantic import Field

from apar.contracts._validation import ExternalContract
from apar.evaluation.v2_preregistration import (
    SYNTHETIC_NON_CLAIM,
    ExecutionReceipt,
    V2Preregistration,
)
from apar.evaluation.v2_protocol import load_v2_protocol, verify_v1_roots
from apar.runs.wire import WireContractError, canonical_json_bytes, strict_json_loads

_PROTOCOL_ID = "apar-defend-v2"
_PROFILE_PATH = Path("config/defense/competition-v2-profile.json")
_MANIFEST_REGISTRY_PATH = Path("config/defense/competition-v2-manifests.json")
_MANIFEST_FIELDS = {
    "source": "source_manifest_sha256",
    "feature": "feature_manifest_sha256",
    "candidate_grid": "candidate_grid_sha256",
    "population": "population_manifest_sha256",
    "evaluator_capability": "evaluator_capability_sha256",
    "metrics": "metrics_manifest_sha256",
    "bootstrap": "bootstrap_manifest_sha256",
    "controls": "controls_manifest_sha256",
    "budget": "budget_manifest_sha256",
    "reporting_schema": "reporting_schema_sha256",
    "fidelity_validation": "fidelity_validation_bundle_sha256",
}
_PROTOCOL_DIGEST = hashlib.sha256(
    canonical_json_bytes({"protocol_id": _PROTOCOL_ID, "synthetic_scope": SYNTHETIC_NON_CLAIM})
).hexdigest()
_FROZEN_V1_EVALUATION_IMPORTS = frozenset(
    {
        ("bundle.py", "apar.evaluation.splits"),
        ("orchestration.py", "apar.evaluation.contracts"),
        ("orchestration.py", "apar.evaluation.competition"),
        ("orchestration.py", "apar.evaluation.corpus"),
        ("orchestration.py", "apar.evaluation.defender_attestation"),
        ("orchestration.py", "apar.evaluation.gates"),
        ("orchestration.py", "apar.evaluation.hidden_source"),
        ("orchestration.py", "apar.evaluation.regimes"),
        ("orchestration.py", "apar.evaluation.replay"),
        ("orchestration.py", "apar.evaluation.reporting"),
        ("orchestration.py", "apar.evaluation.splits"),
    }
)


class PreexecutionCheck(ExternalContract):
    """One fail-closed, read-only condition required before execution."""

    code: str = Field(min_length=1)
    passed: bool


class PreexecutionReport(ExternalContract):
    """Public status of the pre-execution boundary, never an execution result."""

    status: Literal["not_executed"] = "not_executed"
    admissible: bool
    codes: tuple[str, ...]

    @classmethod
    def from_checks(cls, checks: Iterable[PreexecutionCheck]) -> PreexecutionReport:
        checked = tuple(checks)
        failed = tuple(check.code for check in checked if not check.passed)
        return cls(admissible=not failed, codes=failed)


def verify_v2_preexecution(root: Path, preregistration: V2Preregistration) -> PreexecutionReport:
    """Validate public admission prerequisites without generating or executing anything."""
    return PreexecutionReport.from_checks(
        (
            _check_v1_roots(root),
            verify_protocol_digest(preregistration),
            verify_protocol_profile(root, preregistration),
            verify_manifest_registry(root, preregistration),
            verify_no_v2_execution_receipt(root),
            verify_import_boundary(
                root,
                forbidden="apar.evaluation_hidden",
                allowed_prefix="apar.evaluation.v2_",
            ),
            verify_preregistration(root, preregistration),
        )
    )


def verify_protocol_digest(preregistration: object) -> PreexecutionCheck:
    """Bind the signed admission to the sole public v2 protocol identifier."""
    if type(preregistration) is not V2Preregistration:
        return PreexecutionCheck(code="PROTOCOL_DIGEST_INVALID", passed=False)
    try:
        supplied = hashlib.sha256(
            canonical_json_bytes(
                {
                    "protocol_id": preregistration.preregistration_id,
                    "synthetic_scope": preregistration.synthetic_scope,
                }
            )
        ).hexdigest()
    except (AttributeError, TypeError, ValueError):
        supplied = ""
    return PreexecutionCheck(code="PROTOCOL_DIGEST_INVALID", passed=supplied == _PROTOCOL_DIGEST)


def verify_protocol_profile(root: Path, preregistration: object) -> PreexecutionCheck:
    """Bind the signed preregistration to the exact committed profile and budgets."""
    passed = False
    if type(preregistration) is V2Preregistration:
        try:
            protocol = load_v2_protocol(root / _PROFILE_PATH)
            budget_digest = hashlib.sha256(
                canonical_json_bytes(protocol.budgets.model_dump(mode="json"))
            ).hexdigest()
            passed = (
                protocol.profile_sha256 is not None
                and preregistration.protocol_profile_sha256 == protocol.profile_sha256
                and preregistration.preregistration_id == protocol.protocol_id
                and preregistration.seed_commitments == protocol.seed_commitments
                and preregistration.budget_manifest_sha256 == budget_digest
            )
        except (OSError, AttributeError, TypeError, ValueError):
            passed = False
    return PreexecutionCheck(code="PROTOCOL_PROFILE_INVALID", passed=passed)


def verify_manifest_registry(root: Path, preregistration: object) -> PreexecutionCheck:
    """Require every signed manifest digest to resolve in one canonical committed registry."""
    passed = False
    if type(preregistration) is V2Preregistration:
        try:
            raw = (root / _MANIFEST_REGISTRY_PATH).read_bytes()
            wire = raw[:-1] if raw.endswith(b"\n") else raw
            document = strict_json_loads(wire)
            if type(document) is not dict or set(document) != {"schema_version", "manifests"}:
                raise ValueError("manifest registry shape is invalid")
            canonical = canonical_json_bytes(document)
            if wire != canonical:
                raise ValueError("manifest registry is not canonical")
            if document.get("schema_version") != "1.0.0":
                raise ValueError("manifest registry version is invalid")
            manifests = document.get("manifests")
            if type(manifests) is not dict or set(manifests) != set(_MANIFEST_FIELDS):
                raise ValueError("manifest registry entries are incomplete")
            if hashlib.sha256(canonical).hexdigest() != preregistration.manifest_registry_sha256:
                raise ValueError("manifest registry digest mismatch")
            for name, field in _MANIFEST_FIELDS.items():
                expected = getattr(preregistration, field)
                if (
                    expected is None
                    or hashlib.sha256(canonical_json_bytes(manifests[name])).hexdigest() != expected
                ):
                    raise ValueError(f"manifest binding mismatch: {name}")
            _verify_frozen_manifest_inputs(root, manifests)
            passed = True
        except (OSError, AttributeError, TypeError, ValueError, WireContractError):
            passed = False
    return PreexecutionCheck(code="MANIFEST_BINDINGS_INVALID", passed=passed)


def _verify_frozen_manifest_inputs(root: Path, manifests: dict[str, object]) -> None:
    """Verify the exact bytes and complete source inventory named by the registry."""
    source = manifests.get("source")
    feature = manifests.get("feature")
    candidates = manifests.get("candidate_grid")
    population = manifests.get("population")
    if not all(type(item) is dict for item in (source, feature, candidates, population)):
        raise ValueError("frozen input manifest sections are invalid")
    assert isinstance(source, dict)
    assert isinstance(feature, dict)
    assert isinstance(candidates, dict)
    assert isinstance(population, dict)
    if set(source) != {"roots", "files"}:
        raise ValueError("source manifest must contain roots and files")
    roots = source["roots"]
    files = source["files"]
    if type(roots) is not list or roots != ["src/apar/defense", "src/apar/features"]:
        raise ValueError("source roots are not exact")
    if type(files) is not dict or not files:
        raise ValueError("source file inventory is missing")
    actual_source_files: set[str] = set()
    for relative_root in roots:
        source_root = _resolved_frozen_path(root, relative_root)
        if not source_root.is_dir() or source_root.is_symlink():
            raise ValueError("source root is not a regular directory")
        inventory = tuple(source_root.rglob("*"))
        if any(path.is_symlink() for path in inventory):
            raise ValueError("source inventory contains a symbolic link")
        actual_source_files.update(
            path.relative_to(root).as_posix()
            for path in inventory
            if path.suffix == ".py" and path.is_file()
        )
    if set(files) != actual_source_files:
        raise ValueError("source inventory differs from the frozen tree")
    for relative_path, expected_digest in files.items():
        _verify_frozen_file(root, relative_path, expected_digest)

    if set(feature) != {"catalog", "reachable_roots"}:
        raise ValueError("feature manifest does not bind its catalog")
    _verify_frozen_reference(
        root, feature["catalog"], expected_path="config/defense/feature-catalog.json"
    )

    if not {"arms", "defender_bundle", "threshold_artifacts"}.issubset(candidates):
        raise ValueError("candidate manifest omits frozen inputs")
    _verify_frozen_reference(
        root,
        candidates["defender_bundle"],
        expected_path="fixtures/defense/v1/defender-bundle.json",
    )
    thresholds = candidates["threshold_artifacts"]
    if type(thresholds) is not list or len(thresholds) != 2:
        raise ValueError("threshold artifact inventory is incomplete")
    expected_threshold_paths = (
        "fixtures/defense/v1/calibration.json",
        "fixtures/defense/v1/thresholds.json",
    )
    for reference, expected_path in zip(thresholds, expected_threshold_paths, strict=True):
        _verify_frozen_reference(root, reference, expected_path=expected_path)

    if "campaign_ledger" not in population:
        raise ValueError("population manifest omits its campaign ledger")
    _verify_frozen_reference(
        root,
        population["campaign_ledger"],
        expected_path="fixtures/defense/v1/split-manifest.json",
    )


def _verify_frozen_reference(root: Path, value: object, *, expected_path: str) -> None:
    if type(value) is not dict or set(value) != {"path", "sha256"}:
        raise ValueError("frozen content reference is invalid")
    if value["path"] != expected_path:
        raise ValueError("frozen content path is invalid")
    _verify_frozen_file(root, value["path"], value["sha256"])


def _verify_frozen_file(root: Path, relative_path: object, expected_digest: object) -> None:
    if type(relative_path) is not str or type(expected_digest) is not str:
        raise ValueError("frozen file binding is invalid")
    if len(expected_digest) != 64 or any(
        character not in "0123456789abcdef" for character in expected_digest
    ):
        raise ValueError("frozen file digest is invalid")
    path = _resolved_frozen_path(root, relative_path)
    if not path.is_file() or path.is_symlink():
        raise ValueError("frozen input is not a regular file")
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected_digest:
        raise ValueError("frozen input digest mismatch")


def _resolved_frozen_path(root: Path, relative_path: object) -> Path:
    if type(relative_path) is not str:
        raise ValueError("frozen input path is invalid")
    relative = Path(relative_path)
    if relative.is_absolute() or not relative.parts or ".." in relative.parts:
        raise ValueError("frozen input path escapes the repository")
    if root.is_symlink():
        raise ValueError("repository root cannot be a symbolic link")
    lexical = root
    for part in relative.parts:
        lexical = lexical / part
        if lexical.is_symlink():
            raise ValueError("frozen input path contains a symbolic link")
    resolved_root = root.resolve(strict=True)
    resolved = lexical.resolve(strict=True)
    if not resolved.is_relative_to(resolved_root):
        raise ValueError("frozen input path escapes the repository")
    return resolved


def verify_no_v2_execution_receipt(root: Path) -> PreexecutionCheck:
    """Scan the complete durable application store for a schema-valid v2 receipt."""
    receipt_store = root / ".apar"
    try:
        if not receipt_store.exists():
            return PreexecutionCheck(code="V2_EXECUTION_RECEIPT_PRESENT", passed=True)
        if not receipt_store.is_dir():
            return PreexecutionCheck(code="V2_EXECUTION_RECEIPT_UNVERIFIABLE", passed=False)
        for path in receipt_store.rglob("*"):
            if not path.is_file():
                continue
            try:
                document = strict_json_loads(path.read_bytes())
                receipt = ExecutionReceipt.model_validate(document)
            except (WireContractError, ValueError, TypeError):
                continue
            if receipt.preregistration_id == _PROTOCOL_ID:
                return PreexecutionCheck(code="V2_EXECUTION_RECEIPT_PRESENT", passed=False)
    except (OSError, ValueError):
        return PreexecutionCheck(code="V2_EXECUTION_RECEIPT_UNVERIFIABLE", passed=False)
    return PreexecutionCheck(code="V2_EXECUTION_RECEIPT_PRESENT", passed=True)


def verify_import_boundary(root: Path, *, forbidden: str, allowed_prefix: str) -> PreexecutionCheck:
    """Check all local modules reachable from defender code without importing them."""
    project_source = root / "src"
    defender_root = project_source / "apar" / "defense"
    try:
        inventory = tuple(defender_root.rglob("*"))
        if defender_root.is_symlink() or any(path.is_symlink() for path in inventory):
            raise ValueError("defender Python inventory contains a symbolic link")
        pending = [(path, False) for path in inventory if path.suffix == ".py" and path.is_file()]
        seen: dict[Path, bool] = {}
        while pending:
            path, traverse_dependencies = pending.pop()
            _require_regular_python_path(project_source, path)
            if path in seen and (seen[path] or not traverse_dependencies):
                continue
            seen[path] = seen.get(path, False) or traverse_dependencies
            if _contains_disallowed_import(
                path, project_source, defender_root, forbidden, allowed_prefix
            ):
                return PreexecutionCheck(code="HIDDEN_IMPORT_BOUNDARY", passed=False)
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for module in _local_import_targets(tree, path, project_source):
                if not (module == "apar" or module.startswith("apar.")):
                    continue
                if module == "apar.evaluation" or module.startswith("apar.evaluation."):
                    continue
                is_feature = module == "apar.features" or module.startswith("apar.features.")
                if traverse_dependencies or is_feature:
                    pending.extend(
                        (target, True) for target in _local_module_paths(project_source, module)
                    )
    except (OSError, UnicodeError, SyntaxError, ValueError):
        return PreexecutionCheck(code="HIDDEN_IMPORT_BOUNDARY", passed=False)
    return PreexecutionCheck(code="HIDDEN_IMPORT_BOUNDARY", passed=True)


def _require_regular_python_path(project_source: Path, path: Path) -> None:
    """Reject symbolic links anywhere in a defender-reachable Python path."""
    if project_source.is_symlink():
        raise ValueError("project source root cannot be a symbolic link")
    relative = path.relative_to(project_source)
    lexical = project_source
    for part in relative.parts:
        lexical = lexical / part
        if lexical.is_symlink():
            raise ValueError("defender-reachable Python path contains a symbolic link")
    if not lexical.is_file() or lexical.suffix != ".py":
        raise ValueError("defender-reachable Python path is not a regular Python file")


def verify_preregistration(root: Path, preregistration: object) -> PreexecutionCheck:
    """Require the exact sealed contract and its intact evaluator signature."""
    if type(preregistration) is not V2Preregistration:
        return PreexecutionCheck(code="PREREGISTRATION_INVALID", passed=False)
    try:
        raw = (root / "config/defense/competition-v2-preregistration.json").read_bytes()
        wire = raw[:-1] if raw.endswith(b"\n") else raw
        sealed = V2Preregistration.from_json(wire)
        valid = (
            preregistration.verify_manifest_bindings()
            and preregistration.verify_signature()
            and preregistration.matches_sealed_preregistration(sealed)
        )
    except (AttributeError, OSError, TypeError, ValueError):
        valid = False
    return PreexecutionCheck(code="PREREGISTRATION_INVALID", passed=valid)


def _check_v1_roots(root: Path) -> PreexecutionCheck:
    try:
        verify_v1_roots(root)
    except (OSError, ValueError, TypeError):
        return PreexecutionCheck(code="V1_ROOTS_INVALID", passed=False)
    return PreexecutionCheck(code="V1_ROOTS_INVALID", passed=True)


def _contains_disallowed_import(
    path: Path,
    project_source: Path,
    defender_root: Path,
    forbidden: str,
    allowed_prefix: str,
) -> bool:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    (
        importlib_aliases,
        builtins_aliases,
        import_function_aliases,
        getattr_aliases,
        vars_aliases,
    ) = _import_bindings(tree)
    for node in ast.walk(tree):
        if _has_untrusted_authority_reflection(
            node,
            importlib_aliases,
            builtins_aliases,
            getattr_aliases,
            vars_aliases,
        ):
            return True
        if isinstance(node, ast.Import) and any(
            _is_disallowed_module(name.name, forbidden, allowed_prefix)
            and not _is_frozen_v1_import(path, defender_root, name.name)
            for name in node.names
        ):
            return True
        if isinstance(node, ast.ImportFrom):
            for module in _resolved_import_targets(node, path, project_source):
                if _is_disallowed_module(
                    module, forbidden, allowed_prefix
                ) and not _is_frozen_v1_import(path, defender_root, module):
                    return True
        if isinstance(node, ast.Call) and _is_dynamic_import_call(
            node,
            importlib_aliases,
            builtins_aliases,
            import_function_aliases,
            getattr_aliases,
            vars_aliases,
        ):
            if not node.args or not isinstance(node.args[0], ast.Constant):
                return True
            module_value = node.args[0].value
            if not isinstance(module_value, str) or _is_disallowed_module(
                module_value, forbidden, allowed_prefix
            ):
                return True
    return False


def _has_untrusted_authority_reflection(
    node: ast.AST,
    importlib_aliases: set[str],
    builtins_aliases: set[str],
    getattr_aliases: set[str],
    vars_aliases: set[str],
) -> bool:
    """Reject reflective authority access unless static syntax proves it absent."""
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        if node.func.id in getattr_aliases and node.args:
            root = node.args[0]
            return _has_named_root(root, builtins_aliases, vars_aliases) or _has_importlib_root(
                root, importlib_aliases, vars_aliases
            )
        if node.func.id in vars_aliases and len(node.args) == 1:
            root = node.args[0]
            return _has_named_root(root, builtins_aliases, vars_aliases) or _has_importlib_root(
                root, importlib_aliases, vars_aliases
            )
    if isinstance(node, ast.Subscript):
        root = _reflection_mapping_root(node.value, vars_aliases)
        return _has_named_root(root, builtins_aliases, vars_aliases) or _has_importlib_root(
            root, importlib_aliases, vars_aliases
        )
    return False


def _import_bindings(
    tree: ast.AST,
) -> tuple[set[str], set[str], set[str], set[str], set[str]]:
    importlib_aliases: set[str] = set()
    builtins_aliases: set[str] = {"__builtins__"}
    import_function_aliases = {"__import__"}
    getattr_aliases = {"getattr"}
    vars_aliases = {"vars"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for name in node.names:
                if name.name == "importlib" or name.name.startswith("importlib."):
                    importlib_aliases.add(name.asname or name.name.split(".", 1)[0])
                elif name.name == "builtins":
                    builtins_aliases.add(name.asname or name.name)
        elif (
            isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("importlib")
        ):
            for name in node.names:
                if name.name in {"__import__", "import_module"}:
                    import_function_aliases.add(name.asname or name.name)
                elif node.module == "importlib":
                    importlib_aliases.add(name.asname or name.name)
        elif isinstance(node, ast.ImportFrom) and node.module == "builtins":
            for name in node.names:
                if name.name == "__import__":
                    import_function_aliases.add(name.asname or name.name)
                elif name.name == "getattr":
                    getattr_aliases.add(name.asname or name.name)
                elif name.name == "vars":
                    vars_aliases.add(name.asname or name.name)
    changed = True
    while changed:
        changed = False
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign) and _has_importlib_root(
                node.value, importlib_aliases, vars_aliases
            ):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id not in importlib_aliases:
                        importlib_aliases.add(target.id)
                        changed = True
            elif isinstance(node, ast.Assign) and _has_named_root(
                node.value, builtins_aliases, vars_aliases
            ):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id not in builtins_aliases:
                        builtins_aliases.add(target.id)
                        changed = True
            elif isinstance(node, ast.Assign) and _is_import_function_reference(
                node.value,
                importlib_aliases,
                builtins_aliases,
                import_function_aliases,
                getattr_aliases,
                vars_aliases,
            ):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id not in import_function_aliases:
                        import_function_aliases.add(target.id)
                        changed = True
            elif isinstance(node, ast.Assign) and _has_named_root(
                node.value, getattr_aliases, vars_aliases
            ):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id not in getattr_aliases:
                        getattr_aliases.add(target.id)
                        changed = True
            elif isinstance(node, ast.Assign) and _has_named_root(
                node.value, vars_aliases, vars_aliases
            ):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id not in vars_aliases:
                        vars_aliases.add(target.id)
                        changed = True
            elif (
                isinstance(node, ast.AnnAssign)
                and node.value is not None
                and isinstance(node.target, ast.Name)
                and _has_importlib_root(node.value, importlib_aliases, vars_aliases)
                and node.target.id not in importlib_aliases
            ):
                importlib_aliases.add(node.target.id)
                changed = True
            elif (
                isinstance(node, ast.AnnAssign)
                and node.value is not None
                and isinstance(node.target, ast.Name)
                and _has_named_root(node.value, builtins_aliases, vars_aliases)
                and node.target.id not in builtins_aliases
            ):
                builtins_aliases.add(node.target.id)
                changed = True
            elif (
                isinstance(node, ast.AnnAssign)
                and node.value is not None
                and isinstance(node.target, ast.Name)
                and _is_import_function_reference(
                    node.value,
                    importlib_aliases,
                    builtins_aliases,
                    import_function_aliases,
                    getattr_aliases,
                    vars_aliases,
                )
                and node.target.id not in import_function_aliases
            ):
                import_function_aliases.add(node.target.id)
                changed = True
            elif (
                isinstance(node, ast.AnnAssign)
                and node.value is not None
                and isinstance(node.target, ast.Name)
                and _has_named_root(node.value, getattr_aliases, vars_aliases)
                and node.target.id not in getattr_aliases
            ):
                getattr_aliases.add(node.target.id)
                changed = True
            elif (
                isinstance(node, ast.AnnAssign)
                and node.value is not None
                and isinstance(node.target, ast.Name)
                and _has_named_root(node.value, vars_aliases, vars_aliases)
                and node.target.id not in vars_aliases
            ):
                vars_aliases.add(node.target.id)
                changed = True
    return (
        importlib_aliases,
        builtins_aliases,
        import_function_aliases,
        getattr_aliases,
        vars_aliases,
    )


def _is_dynamic_import_call(
    node: ast.Call,
    importlib_aliases: set[str],
    builtins_aliases: set[str],
    import_function_aliases: set[str],
    getattr_aliases: set[str],
    vars_aliases: set[str],
) -> bool:
    return _is_import_function_reference(
        node.func,
        importlib_aliases,
        builtins_aliases,
        import_function_aliases,
        getattr_aliases,
        vars_aliases,
    )


def _is_disallowed_module(module: str | None, forbidden: str, allowed_prefix: str) -> bool:
    if not module:
        return False
    return (
        module == forbidden
        or module.startswith(f"{forbidden}.")
        or (
            (module == "apar.evaluation" or module.startswith("apar.evaluation."))
            and not module.startswith(allowed_prefix)
        )
    )


def _resolved_import_targets(
    node: ast.ImportFrom, path: Path, project_source: Path
) -> tuple[str, ...]:
    if node.level == 0:
        resolved = node.module
        base: tuple[str, ...] = ()
    else:
        package = _source_package(path, project_source)
        keep = len(package) - node.level + 1
        if keep <= 0:
            raise ValueError("relative import escapes the defender package")
        base = package[:keep]
        resolved = ".".join((*base, *(node.module or "").split("."))).rstrip(".")
    if node.module is None:
        return tuple(".".join((*base, name.name)) for name in node.names)
    if resolved in {"apar", "apar.evaluation"}:
        return tuple(f"{resolved}.{name.name}" for name in node.names)
    if resolved:
        return (resolved,)
    return ()


def _source_package(path: Path, project_source: Path) -> tuple[str, ...]:
    relative = path.relative_to(project_source)
    parents = () if relative.parent == Path(".") else relative.parent.parts
    return parents


def _local_import_targets(tree: ast.AST, path: Path, project_source: Path) -> tuple[str, ...]:
    modules: set[str] = set()
    (
        importlib_aliases,
        builtins_aliases,
        import_function_aliases,
        getattr_aliases,
        vars_aliases,
    ) = _import_bindings(tree)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(name.name for name in node.names)
        elif isinstance(node, ast.ImportFrom):
            bases = _resolved_import_targets(node, path, project_source)
            modules.update(bases)
            for base in bases:
                modules.update(f"{base}.{name.name}" for name in node.names if name.name != "*")
        elif (
            isinstance(node, ast.Call)
            and _is_dynamic_import_call(
                node,
                importlib_aliases,
                builtins_aliases,
                import_function_aliases,
                getattr_aliases,
                vars_aliases,
            )
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[0].value, str)
        ):
            modules.add(node.args[0].value)
    return tuple(
        sorted(module for module in modules if module == "apar" or module.startswith("apar."))
    )


def _local_module_paths(project_source: Path, module: str) -> tuple[Path, ...]:
    relative = Path(*module.split("."))
    candidates = (
        project_source / relative.with_suffix(".py"),
        project_source / relative / "__init__.py",
    )
    return tuple(path for path in candidates if path.is_file())


def _has_importlib_root(
    value: ast.expr, importlib_aliases: set[str], vars_aliases: set[str]
) -> bool:
    reflected = _reflection_mapping_root(value, vars_aliases)
    if reflected is not value:
        return _has_importlib_root(reflected, importlib_aliases, vars_aliases)
    if isinstance(value, ast.Name):
        return value.id in importlib_aliases
    return isinstance(value, ast.Attribute) and _has_importlib_root(
        value.value, importlib_aliases, vars_aliases
    )


def _is_import_function_reference(
    value: ast.expr,
    importlib_aliases: set[str],
    builtins_aliases: set[str],
    import_function_aliases: set[str],
    getattr_aliases: set[str],
    vars_aliases: set[str],
) -> bool:
    if isinstance(value, ast.Name):
        return value.id in import_function_aliases
    if isinstance(value, ast.Attribute):
        return (
            value.attr == "import_module"
            and _has_importlib_root(value.value, importlib_aliases, vars_aliases)
        ) or (
            value.attr == "__import__"
            and _has_named_root(value.value, builtins_aliases, vars_aliases)
        )
    if isinstance(value, ast.Call):
        if (
            isinstance(value.func, ast.Name)
            and value.func.id in getattr_aliases
            and len(value.args) >= 2
            and isinstance(value.args[1], ast.Constant)
        ):
            attribute = value.args[1].value
            return (
                attribute == "__import__"
                and _has_named_root(value.args[0], builtins_aliases, vars_aliases)
            ) or (
                attribute == "import_module"
                and _has_importlib_root(value.args[0], importlib_aliases, vars_aliases)
            )
        return False
    if not isinstance(value, ast.Subscript) or not isinstance(value.slice, ast.Constant):
        return False
    attribute = value.slice.value
    root = _reflection_mapping_root(value.value, vars_aliases)
    return (
        attribute == "__import__" and _has_named_root(root, builtins_aliases, vars_aliases)
    ) or (
        attribute == "import_module"
        and _has_importlib_root(root, importlib_aliases, vars_aliases)
    )


def _reflection_mapping_root(value: ast.expr, vars_aliases: set[str]) -> ast.expr:
    if isinstance(value, ast.Attribute) and value.attr == "__dict__":
        return value.value
    if (
        isinstance(value, ast.Call)
        and isinstance(value.func, ast.Name)
        and value.func.id in vars_aliases
        and len(value.args) == 1
    ):
        return value.args[0]
    return value


def _has_named_root(value: ast.expr, aliases: set[str], vars_aliases: set[str]) -> bool:
    reflected = _reflection_mapping_root(value, vars_aliases)
    if reflected is not value:
        return _has_named_root(reflected, aliases, vars_aliases)
    return isinstance(value, ast.Name) and value.id in aliases


def _is_frozen_v1_import(path: Path, source_root: Path, module: str | None) -> bool:
    try:
        relative = path.relative_to(source_root).as_posix()
    except ValueError:
        return False
    return (relative, module) in _FROZEN_V1_EVALUATION_IMPORTS


__all__ = [
    "PreexecutionCheck",
    "PreexecutionReport",
    "verify_import_boundary",
    "verify_no_v2_execution_receipt",
    "verify_preregistration",
    "verify_protocol_digest",
    "verify_v2_preexecution",
]
