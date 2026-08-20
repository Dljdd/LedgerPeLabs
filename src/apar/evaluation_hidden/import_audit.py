"""Fail-closed AST audit for the defender-to-hidden package boundary."""

from __future__ import annotations

import ast
import hashlib
from pathlib import Path
from typing import Literal, cast

from pydantic import field_validator, model_validator

from apar.contracts._validation import ExternalContract
from apar.evaluation_hidden.authority_core import HiddenBoundaryError
from apar.runs.wire import canonical_json_bytes

_MAX_SOURCE_BYTES = 2_000_000


class HiddenImportAudit(ExternalContract):
    """Static evidence that defender packages cannot import evaluator internals."""

    schema_version: Literal["1.0.0"] = "1.0.0"
    passed: bool
    scanned_files: tuple[str, ...]
    violations: tuple[str, ...]
    audit_digest: str

    @field_validator("scanned_files", "violations", mode="before")
    @classmethod
    def collections_are_exact_tuples(cls, value: object) -> object:
        if type(value) is not tuple:
            raise ValueError("hidden import audit collections must be exact tuples")
        return value

    @model_validator(mode="after")
    def evidence_is_canonical(self) -> HiddenImportAudit:
        if self.scanned_files != tuple(sorted(set(self.scanned_files))):
            raise ValueError("scanned hidden-boundary files must be sorted and unique")
        if self.violations != tuple(sorted(set(self.violations))):
            raise ValueError("hidden-boundary violations must be sorted and unique")
        if self.passed != (not self.violations):
            raise ValueError("hidden import audit pass state is inconsistent")
        expected = hashlib.sha256(
            canonical_json_bytes(
                self.model_dump(mode="json", exclude={"audit_digest"})
            )
        ).hexdigest()
        if self.audit_digest != expected:
            raise ValueError("hidden import audit digest is inconsistent")
        return self


def audit_hidden_import_boundary(apar_source_root: Path) -> HiddenImportAudit:
    """Audit imports and conservatively reject dynamic code/import primitives."""
    if not isinstance(apar_source_root, Path):
        raise HiddenBoundaryError("APAR source root must be an exact Path")
    root = apar_source_root.resolve(strict=True)
    if not root.is_dir() or root.is_symlink():
        raise HiddenBoundaryError("APAR source root must be a regular directory")
    scanned: list[str] = []
    violations: list[str] = []
    for package in ("defense", "features"):
        package_root = root / package
        if not package_root.is_dir() or package_root.is_symlink():
            raise HiddenBoundaryError(f"missing defender package: {package}")
        for path in sorted(package_root.rglob("*.py")):
            if path.is_symlink() or not path.is_file():
                raise HiddenBoundaryError("hidden import audit rejects non-regular source")
            relative = path.relative_to(root).as_posix()
            payload = path.read_bytes()
            if len(payload) > _MAX_SOURCE_BYTES:
                raise HiddenBoundaryError("hidden import audit source exceeds resource cap")
            try:
                tree = ast.parse(payload, filename=relative)
            except (SyntaxError, ValueError) as error:
                raise HiddenBoundaryError(
                    f"hidden import audit cannot parse {relative}"
                ) from error
            scanned.append(relative)
            violations.extend(
                f"{relative}:{line}:{module}" for line, module in _hidden_imports(tree)
            )
    fields: dict[str, object] = {
        "passed": not violations,
        "scanned_files": tuple(sorted(scanned)),
        "violations": tuple(sorted(violations)),
    }
    digest = hashlib.sha256(
        canonical_json_bytes(
            {
                "schema_version": "1.0.0",
                "passed": not violations,
                "scanned_files": list(cast(tuple[str, ...], fields["scanned_files"])),
                "violations": list(cast(tuple[str, ...], fields["violations"])),
            }
        )
    ).hexdigest()
    return HiddenImportAudit.model_validate({**fields, "audit_digest": digest})


def _hidden_imports(tree: ast.AST) -> tuple[tuple[int, str], ...]:
    found: list[tuple[int, str]] = []
    importlib_names = {"importlib"}
    builtins_names = {"builtins"}
    dynamic_import_names = {"__import__", "importlib.import_module"}
    reflection_names = {"getattr"}
    code_execution_names = {"eval", "exec", "compile"}
    namespace_reflection_names = {"globals", "locals", "vars"}
    import_mapping_names: set[str] = set()
    subscript_callable_names: set[str] = set()
    retrieval_accessor_names: set[str] = set()
    dynamic_callable_names: set[str] = set()
    retrieval_methods = {"get", "__getitem__", "pop", "setdefault", "values"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "importlib":
                    importlib_names.add(alias.asname or alias.name)
                elif alias.name == "builtins":
                    builtins_names.add(alias.asname or alias.name)
        elif isinstance(node, ast.ImportFrom) and node.module == "importlib":
            for alias in node.names:
                if alias.name == "import_module":
                    dynamic_import_names.add(alias.asname or alias.name)
        elif isinstance(node, ast.ImportFrom) and node.module == "builtins":
            for alias in node.names:
                target = alias.asname or alias.name
                if alias.name == "__import__":
                    dynamic_import_names.add(target)
                elif alias.name == "getattr":
                    reflection_names.add(target)
                elif alias.name in code_execution_names:
                    code_execution_names.add(target)
                elif alias.name in namespace_reflection_names:
                    namespace_reflection_names.add(target)
    dynamic_import_names.update(f"{name}.import_module" for name in importlib_names)
    dynamic_import_names.update(f"{name}.__import__" for name in builtins_names)
    changed = True
    while changed:
        changed = False
        for node in ast.walk(tree):
            for binding_target, binding_value in _binding_pairs(node):
                for target_leaf in _assignment_target_leaves(binding_target):
                    alias_name = _call_name(target_leaf)
                    source_name = _call_name(binding_value)
                    changed |= _propagate_assignment_provenance(
                        alias_name=alias_name,
                        source_name=source_name,
                        value=binding_value,
                        importlib_names=importlib_names,
                        builtins_names=builtins_names,
                        dynamic_import_names=dynamic_import_names,
                        reflection_names=reflection_names,
                        code_execution_names=code_execution_names,
                        namespace_reflection_names=namespace_reflection_names,
                        import_mapping_names=import_mapping_names,
                        subscript_callable_names=subscript_callable_names,
                        retrieval_accessor_names=retrieval_accessor_names,
                        dynamic_callable_names=dynamic_callable_names,
                        retrieval_methods=retrieval_methods,
                    )
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if _is_hidden_module(alias.name):
                    found.append((node.lineno, alias.name))
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if _is_hidden_module(module):
                found.append((node.lineno, module))
            elif module == "apar":
                if any(alias.name == "evaluation_hidden" for alias in node.names):
                    found.append((node.lineno, "apar.evaluation_hidden"))
            elif node.level and (
                module == "evaluation_hidden"
                or module.startswith("evaluation_hidden.")
                or (
                    not module
                    and any(alias.name == "evaluation_hidden" for alias in node.names)
                )
            ):
                found.append((node.lineno, "apar.evaluation_hidden"))
        elif isinstance(node, ast.Call):
            name = _call_name(node.func)
            if (
                isinstance(node.func, (ast.Call, ast.Subscript))
                or name in subscript_callable_names
                or name in dynamic_callable_names
            ):
                found.append((node.lineno, "<mapping-derived-callable>"))
                continue
            if name in namespace_reflection_names:
                found.append((node.lineno, "<namespace-reflection>"))
                continue
            if name in code_execution_names:
                found.append((node.lineno, "<dynamic-code-execution>"))
                continue
            if name in reflection_names:
                owner = _call_name(node.args[0]) if node.args else ""
                # Only the two audited production self-lookups are ordinary
                # reflection. Every other getattr chain is unresolved code
                # provenance and therefore fails closed.
                if owner != "self":
                    found.append((node.lineno, "<import-reflection>"))
                continue
            if name not in dynamic_import_names:
                continue
            argument = _import_argument(node)
            if isinstance(argument, ast.Constant) and type(argument.value) is str:
                found.append((node.lineno, f"<dynamic-import:{argument.value}>"))
            else:
                found.append((node.lineno, "<unresolved-dynamic-import>"))
        elif isinstance(node, ast.Attribute):
            exact_immutable_primitive = (
                _call_name(node.value) in {"object", "tuple"}
                and node.attr
                in {"__getitem__", "__len__", "__new__", "__setattr__"}
            )
            if not exact_immutable_primitive and (
                (
                node.attr.startswith("__")
                and node.attr.endswith("__")
                and node.attr
                not in {"__version__", "__name__", "__getattribute__", "__setattr__"}
                )
                or (
                    node.attr
                    in {"__dict__", "__globals__", "__builtins__", "__import__"}
                )
                or (
                    node.attr == "import_module"
                    and _call_name(node.value) in importlib_names
                )
            ):
                found.append((node.lineno, "<import-dunder-reflection>"))
        elif isinstance(node, ast.Subscript) and (
            _dangerous_mapping_key(node.slice)
            or _call_name(node.value) in import_mapping_names
            or (
                isinstance(node.value, ast.Attribute)
                and node.value.attr == "__dict__"
                and _call_name(node.value.value)
                in importlib_names | builtins_names
            )
        ):
            found.append((node.lineno, "<import-mapping-reflection>"))
    return tuple(sorted(set(found)))


def _binding_pairs(node: ast.AST) -> tuple[tuple[ast.expr, ast.expr], ...]:
    """Return value-to-target edges for every Python binding form with a value."""
    if isinstance(node, ast.Assign):
        return tuple((target, node.value) for target in node.targets)
    if isinstance(node, ast.AnnAssign):
        return () if node.value is None else ((node.target, node.value),)
    if isinstance(node, (ast.NamedExpr, ast.AugAssign)):
        return ((node.target, node.value),)
    if isinstance(node, (ast.For, ast.AsyncFor, ast.comprehension)):
        return ((node.target, node.iter),)
    if isinstance(node, ast.withitem):
        if node.optional_vars is None:
            return ()
        return ((node.optional_vars, node.context_expr),)
    if isinstance(node, ast.ExceptHandler):
        if node.name is None or node.type is None:
            return ()
        return ((ast.Name(id=node.name, ctx=ast.Store()), node.type),)
    if isinstance(node, ast.Match):
        return tuple(
            (ast.Name(id=name, ctx=ast.Store()), node.subject)
            for case in node.cases
            for name in _match_capture_names(case.pattern)
        )
    return ()


def _match_capture_names(pattern: ast.pattern) -> tuple[str, ...]:
    """Return every name captured by a structural-pattern binding."""
    if isinstance(pattern, ast.MatchAs):
        nested = () if pattern.pattern is None else _match_capture_names(pattern.pattern)
        return nested + (() if pattern.name is None else (pattern.name,))
    if isinstance(pattern, ast.MatchStar):
        return () if pattern.name is None else (pattern.name,)
    if isinstance(pattern, ast.MatchSequence):
        return tuple(
            name
            for nested_pattern in pattern.patterns
            for name in _match_capture_names(nested_pattern)
        )
    if isinstance(pattern, ast.MatchMapping):
        nested = tuple(
            name
            for nested_pattern in pattern.patterns
            for name in _match_capture_names(nested_pattern)
        )
        return nested + (() if pattern.rest is None else (pattern.rest,))
    if isinstance(pattern, ast.MatchClass):
        return tuple(
            name
            for nested_pattern in (*pattern.patterns, *pattern.kwd_patterns)
            for name in _match_capture_names(nested_pattern)
        )
    if isinstance(pattern, ast.MatchOr):
        return tuple(
            name
            for nested_pattern in pattern.patterns
            for name in _match_capture_names(nested_pattern)
        )
    return ()


def _assignment_target_leaves(target: ast.expr) -> tuple[ast.expr, ...]:
    """Return every bound name/attribute in arbitrarily nested unpacking."""
    if isinstance(target, (ast.Name, ast.Attribute)):
        return (target,)
    if isinstance(target, ast.Starred):
        return _assignment_target_leaves(target.value)
    if isinstance(target, (ast.Tuple, ast.List)):
        return tuple(
            leaf
            for element in target.elts
            for leaf in _assignment_target_leaves(element)
        )
    return ()


def _propagate_assignment_provenance(
    *,
    alias_name: str,
    source_name: str,
    value: ast.expr,
    importlib_names: set[str],
    builtins_names: set[str],
    dynamic_import_names: set[str],
    reflection_names: set[str],
    code_execution_names: set[str],
    namespace_reflection_names: set[str],
    import_mapping_names: set[str],
    subscript_callable_names: set[str],
    retrieval_accessor_names: set[str],
    dynamic_callable_names: set[str],
    retrieval_methods: set[str],
) -> bool:
    """Conservatively propagate executable provenance to one assignment leaf."""
    changed = False

    def add(target: set[str], name: str) -> None:
        nonlocal changed
        if name and name not in target:
            target.add(name)
            changed = True

    if isinstance(value, ast.Subscript):
        add(subscript_callable_names, alias_name)
    if isinstance(value, ast.Attribute) and value.attr in retrieval_methods:
        add(retrieval_accessor_names, alias_name)
    if isinstance(value, ast.Call) and (
        _call_name(value.func) in retrieval_accessor_names
        or (
            isinstance(value.func, ast.Attribute)
            and value.func.attr in retrieval_methods
        )
        or (
            _call_name(value.func) in reflection_names
            and not _is_allowed_feature_dispatch(value)
        )
        or isinstance(value.func, (ast.Call, ast.Subscript))
    ):
        add(dynamic_callable_names, alias_name)
    if source_name in retrieval_accessor_names:
        add(retrieval_accessor_names, alias_name)
    if source_name in dynamic_callable_names:
        add(dynamic_callable_names, alias_name)
    if (
        isinstance(value, ast.Attribute)
        and value.attr == "__dict__"
        and _call_name(value.value) in importlib_names | builtins_names
    ):
        add(import_mapping_names, alias_name)
    if (
        isinstance(value, ast.Subscript)
        and _call_name(value.value) in import_mapping_names
    ):
        add(dynamic_import_names, alias_name)
    if source_name in importlib_names:
        add(importlib_names, alias_name)
        add(dynamic_import_names, f"{alias_name}.import_module" if alias_name else "")
    if source_name in builtins_names:
        add(builtins_names, alias_name)
        add(dynamic_import_names, f"{alias_name}.__import__" if alias_name else "")
    if source_name in dynamic_import_names or _is_getattr_import(
        value, importlib_names, builtins_names, reflection_names
    ):
        add(dynamic_import_names, alias_name)
    if source_name in reflection_names:
        add(reflection_names, alias_name)
    if source_name in code_execution_names:
        add(code_execution_names, alias_name)
    if source_name in namespace_reflection_names:
        add(namespace_reflection_names, alias_name)
    return changed


def _dangerous_mapping_key(node: ast.expr) -> bool:
    """Reject mapping access that can recover Python's import machinery."""
    return (
        isinstance(node, ast.Constant)
        and type(node.value) is str
        and node.value in {"__builtins__", "__dict__", "__import__", "import_module"}
    )


def _is_allowed_feature_dispatch(node: ast.Call) -> bool:
    """Allow only the existing closed feature-name dispatch reflection shape."""
    if (
        _call_name(node.func) != "getattr"
        or len(node.args) != 3
        or _call_name(node.args[0]) != "self"
        or not isinstance(node.args[1], ast.JoinedStr)
        or not isinstance(node.args[2], ast.Constant)
        or node.args[2].value is not None
    ):
        return False
    parts = node.args[1].values
    return (
        len(parts) == 2
        and isinstance(parts[0], ast.Constant)
        and parts[0].value == "_feature_"
        and isinstance(parts[1], ast.FormattedValue)
        and _call_name(parts[1].value) == "definition.name"
    )


def _import_argument(node: ast.Call) -> ast.expr | None:
    if node.args:
        return node.args[0]
    for keyword in node.keywords:
        if keyword.arg in {"name", "module"}:
            return keyword.value
    return None


def _call_name(node: ast.expr) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = _call_name(node.value)
        return f"{prefix}.{node.attr}" if prefix else ""
    return ""


def _is_getattr_import(
    node: ast.expr,
    importlib_names: set[str],
    builtins_names: set[str],
    reflection_names: set[str],
) -> bool:
    if not isinstance(node, ast.Call) or _call_name(node.func) not in reflection_names:
        return False
    if not node.args:
        return False
    owner = _call_name(node.args[0])
    # Fail closed even when the second argument is a composed expression.
    if owner not in importlib_names and owner not in builtins_names:
        return False
    if len(node.args) < 2 or not isinstance(node.args[1], ast.Constant):
        return True
    attribute = node.args[1].value
    return (owner in importlib_names and attribute == "import_module") or (
        owner in builtins_names and attribute == "__import__"
    )


def _is_hidden_module(module: str) -> bool:
    return module == "apar.evaluation_hidden" or module.startswith(
        "apar.evaluation_hidden."
    )


__all__ = ["HiddenImportAudit", "audit_hidden_import_boundary"]
