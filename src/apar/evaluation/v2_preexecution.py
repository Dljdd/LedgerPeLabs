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
from apar.evaluation.v2_preregistration import SYNTHETIC_NON_CLAIM, V2Preregistration
from apar.evaluation.v2_protocol import verify_v1_roots
from apar.runs.wire import canonical_json_bytes

_PROTOCOL_ID = "apar-defend-v2"
_PROTOCOL_DIGEST = hashlib.sha256(
    canonical_json_bytes(
        {"protocol_id": _PROTOCOL_ID, "synthetic_scope": SYNTHETIC_NON_CLAIM}
    )
).hexdigest()


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


def verify_v2_preexecution(
    root: Path, preregistration: V2Preregistration
) -> PreexecutionReport:
    """Validate public admission prerequisites without generating or executing anything."""
    return PreexecutionReport.from_checks(
        (
            _check_v1_roots(root),
            verify_protocol_digest(preregistration),
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


def verify_no_v2_execution_receipt(root: Path) -> PreexecutionCheck:
    """Fail if any receipt-shaped file identifies an already-consumed v2 attempt."""
    try:
        for path in root.rglob("*receipt*"):
            if not path.is_file():
                continue
            relative = path.relative_to(root).as_posix().lower()
            payload = path.read_bytes()[:262_144].lower()
            if "v2" in relative or _PROTOCOL_ID.encode("ascii") in payload:
                return PreexecutionCheck(code="V2_EXECUTION_RECEIPT_PRESENT", passed=False)
    except (OSError, ValueError):
        return PreexecutionCheck(code="V2_EXECUTION_RECEIPT_UNVERIFIABLE", passed=False)
    return PreexecutionCheck(code="V2_EXECUTION_RECEIPT_PRESENT", passed=True)


def verify_import_boundary(
    root: Path, *, forbidden: str, allowed_prefix: str
) -> PreexecutionCheck:
    """Check defender modules statically, without importing any of them."""
    del allowed_prefix  # Public v2 contracts are separate from the forbidden namespace.
    source_root = root / "src" / "apar" / "defense"
    try:
        for path in source_root.rglob("*.py"):
            if _contains_forbidden_import(path, forbidden):
                return PreexecutionCheck(code="HIDDEN_IMPORT_BOUNDARY", passed=False)
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


def _contains_forbidden_import(path: Path, forbidden: str) -> bool:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import) and any(
            _is_forbidden(name.name, forbidden) for name in node.names
        ):
            return True
        if isinstance(node, ast.ImportFrom) and _is_forbidden(node.module, forbidden):
            return True
        if (
            isinstance(node, ast.Call)
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[0].value, str)
            and _is_forbidden(node.args[0].value, forbidden)
        ):
            return True
    return False


def _is_forbidden(module: str | None, forbidden: str) -> bool:
    return module == forbidden or bool(module and module.startswith(f"{forbidden}."))


__all__ = [
    "PreexecutionCheck",
    "PreexecutionReport",
    "verify_import_boundary",
    "verify_no_v2_execution_receipt",
    "verify_preregistration",
    "verify_protocol_digest",
    "verify_v2_preexecution",
]
