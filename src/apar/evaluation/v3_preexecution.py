"""Read-only admission checks for a never-started Defend v3 evaluation."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from pathlib import Path
from typing import Literal

from pydantic import model_validator

from apar.contracts._validation import ExternalContract
from apar.v3_protocol import V3ProtocolError, verify_v1_v2_roots


class PreexecutionCheck(ExternalContract):
    """One fail-closed, read-only condition required before execution."""

    code: str
    passed: bool


class PreexecutionReport(ExternalContract):
    """Public status of the pre-execution boundary, never an execution result."""

    admissible: bool
    codes: tuple[str, ...] = ()
    status: Literal["not_executed"] = "not_executed"

    @model_validator(mode="after")
    def codes_match_admissibility(self) -> Self:
        from typing import Self as _Self
        if self.admissible != (not self.codes):
            raise ValueError("admissibility must agree with codes")
        return self

    @classmethod
    def from_checks(cls, checks: Iterable[PreexecutionCheck]) -> PreexecutionReport:
        failed = tuple(check.code for check in checks if not check.passed)
        return PreexecutionReport(admissible=not failed, codes=failed)


def verify_v3_preexecution(root: Path) -> PreexecutionReport:
    """Verify all read-only pre-execution conditions without starting evaluation."""
    return PreexecutionReport.from_checks(
        (
            _check_v1_v2_roots(root),
            _check_no_v3_receipt(root),
        )
    )


def _check_v1_v2_roots(root: Path) -> PreexecutionCheck:
    try:
        verify_v1_v2_roots(root)
    except (OSError, ValueError, TypeError):
        return PreexecutionCheck(code="V1_V2_ROOTS_INVALID", passed=False)
    return PreexecutionCheck(code="V1_V2_ROOTS_INVALID", passed=True)


def _check_no_v3_receipt(root: Path) -> PreexecutionCheck:
    receipt_store = root / ".apar" / "defense-v3"
    if not receipt_store.exists():
        return PreexecutionCheck(code="V3_EXECUTION_RECEIPT_PRESENT", passed=True)
    if not receipt_store.is_dir():
        return PreexecutionCheck(code="V3_EXECUTION_RECEIPT_UNVERIFIABLE", passed=False)
    receipt_path = receipt_store / "execution-receipt.json"
    if receipt_path.is_file():
        return PreexecutionCheck(code="V3_EXECUTION_RECEIPT_PRESENT", passed=False)
    return PreexecutionCheck(code="V3_EXECUTION_RECEIPT_PRESENT", passed=True)


__all__ = [
    "PreexecutionCheck",
    "PreexecutionReport",
    "verify_v3_preexecution",
]
