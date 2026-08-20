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
            verify_preregistration(preregistration),
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
            passed = True
        except (OSError, AttributeError, TypeError, ValueError, WireContractError):
            passed = False
    return PreexecutionCheck(code="MANIFEST_BINDINGS_INVALID", passed=passed)


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
        pending = list(defender_root.rglob("*.py"))
        seen: set[Path] = set()
        while pending:
            path = pending.pop()
            if path in seen:
                continue
            seen.add(path)
            if _contains_disallowed_import(
                path, project_source, defender_root, forbidden, allowed_prefix
            ):
                return PreexecutionCheck(code="HIDDEN_IMPORT_BOUNDARY", passed=False)
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for module in _local_import_targets(tree, path, project_source):
                if not (
                    module == "apar.defense"
                    or module.startswith("apar.defense.")
                    or module == "apar.features"
                    or module.startswith("apar.features.")
                ):
                    continue
                pending.extend(_local_module_paths(project_source, module))
    except (OSError, UnicodeError, SyntaxError, ValueError):
        return PreexecutionCheck(code="HIDDEN_IMPORT_BOUNDARY", passed=False)
    return PreexecutionCheck(code="HIDDEN_IMPORT_BOUNDARY", passed=True)


def verify_preregistration(preregistration: object) -> PreexecutionCheck:
    """Require the exact sealed contract and its intact evaluator signature."""
    if type(preregistration) is not V2Preregistration:
        return PreexecutionCheck(code="PREREGISTRATION_INVALID", passed=False)
    try:
        valid = preregistration.verify_manifest_bindings() and preregistration.verify_signature()
    except (AttributeError, TypeError, ValueError):
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
    importlib_aliases, builtins_aliases, import_function_aliases = _import_bindings(tree)
    for node in ast.walk(tree):
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
        ):
            if not node.args or not isinstance(node.args[0], ast.Constant):
                return True
            module_value = node.args[0].value
            if not isinstance(module_value, str) or _is_disallowed_module(
                module_value, forbidden, allowed_prefix
            ):
                return True
    return False


def _import_bindings(tree: ast.AST) -> tuple[set[str], set[str], set[str]]:
    importlib_aliases: set[str] = set()
    builtins_aliases: set[str] = set()
    import_function_aliases = {"__import__"}
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
    changed = True
    while changed:
        changed = False
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign) and _has_importlib_root(node.value, importlib_aliases):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id not in importlib_aliases:
                        importlib_aliases.add(target.id)
                        changed = True
            elif isinstance(node, ast.Assign) and _is_builtin_import_reference(
                node.value, builtins_aliases, import_function_aliases
            ):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id not in import_function_aliases:
                        import_function_aliases.add(target.id)
                        changed = True
            elif (
                isinstance(node, ast.AnnAssign)
                and node.value is not None
                and isinstance(node.target, ast.Name)
                and _has_importlib_root(node.value, importlib_aliases)
                and node.target.id not in importlib_aliases
            ):
                importlib_aliases.add(node.target.id)
                changed = True
            elif (
                isinstance(node, ast.AnnAssign)
                and node.value is not None
                and isinstance(node.target, ast.Name)
                and _is_builtin_import_reference(
                    node.value, builtins_aliases, import_function_aliases
                )
                and node.target.id not in import_function_aliases
            ):
                import_function_aliases.add(node.target.id)
                changed = True
    return importlib_aliases, builtins_aliases, import_function_aliases


def _is_dynamic_import_call(
    node: ast.Call,
    importlib_aliases: set[str],
    builtins_aliases: set[str],
    import_function_aliases: set[str],
) -> bool:
    if isinstance(node.func, ast.Name):
        return node.func.id in import_function_aliases
    if not isinstance(node.func, ast.Attribute):
        return False
    if node.func.attr == "import_module":
        return _has_importlib_root(node.func.value, importlib_aliases)
    return (
        node.func.attr == "__import__"
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id in builtins_aliases
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
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(name.name for name in node.names)
        elif isinstance(node, ast.ImportFrom):
            bases = _resolved_import_targets(node, path, project_source)
            modules.update(bases)
            for base in bases:
                modules.update(f"{base}.{name.name}" for name in node.names if name.name != "*")
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


def _has_importlib_root(value: ast.expr, importlib_aliases: set[str]) -> bool:
    if isinstance(value, ast.Name):
        return value.id in importlib_aliases
    return isinstance(value, ast.Attribute) and _has_importlib_root(value.value, importlib_aliases)


def _is_builtin_import_reference(
    value: ast.expr, builtins_aliases: set[str], import_function_aliases: set[str]
) -> bool:
    if isinstance(value, ast.Name):
        return value.id in import_function_aliases
    if isinstance(value, ast.Attribute):
        return (
            value.attr == "__import__"
            and isinstance(value.value, ast.Name)
            and value.value.id in builtins_aliases
        )
    if isinstance(value, ast.Call):
        return (
            isinstance(value.func, ast.Name)
            and value.func.id == "getattr"
            and len(value.args) >= 2
            and isinstance(value.args[0], ast.Name)
            and value.args[0].id in builtins_aliases
            and isinstance(value.args[1], ast.Constant)
            and value.args[1].value == "__import__"
        )
    return (
        isinstance(value, ast.Subscript)
        and isinstance(value.value, ast.Name)
        and value.value.id in builtins_aliases
        and isinstance(value.slice, ast.Constant)
        and value.slice.value == "__import__"
    )


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
