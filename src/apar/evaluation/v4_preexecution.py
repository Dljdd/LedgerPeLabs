"""Read-only admission checks for a never-started Defend v4 evaluation."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Literal

from pydantic import model_validator

from apar.contracts._validation import ExternalContract
from apar.v4_protocol import V4ProtocolError, verify_prior_roots


class PreexecutionCheck(ExternalContract):
    code: str
    passed: bool


class PreexecutionReport(ExternalContract):
    admissible: bool
    codes: tuple[str, ...] = ()
    status: Literal["not_executed"] = "not_executed"

    @model_validator(mode="after")
    def codes_match_admissibility(self) -> Self:
        if self.admissible != (not self.codes):
            raise ValueError("admissibility must agree with codes")
        return self

    @classmethod
    def from_checks(cls, checks: Iterable[PreexecutionCheck]) -> PreexecutionReport:
        failed = tuple(check.code for check in checks if not check.passed)
        return PreexecutionReport(admissible=not failed, codes=failed)


def verify_v4_preexecution(root: Path) -> PreexecutionReport:
    return PreexecutionReport.from_checks(
        (
            _check_prior_roots(root),
            _check_no_v4_receipt(root),
        )
    )


def _check_prior_roots(root: Path) -> PreexecutionCheck:
    try:
        verify_prior_roots(root)
    except (OSError, ValueError, TypeError):
        return PreexecutionCheck(code="PRIOR_ROOTS_INVALID", passed=False)
    return PreexecutionCheck(code="PRIOR_ROOTS_INVALID", passed=True)


def _check_no_v4_receipt(root: Path) -> PreexecutionCheck:
    receipt_store = root / ".apar" / "defense-v4"
    if not receipt_store.exists():
        return PreexecutionCheck(code="V4_EXECUTION_RECEIPT_PRESENT", passed=True)
    if not receipt_store.is_dir():
        return PreexecutionCheck(code="V4_EXECUTION_RECEIPT_UNVERIFIABLE", passed=False)
    receipt_path = receipt_store / "execution-receipt.json"
    if receipt_path.is_file():
        return PreexecutionCheck(code="V4_EXECUTION_RECEIPT_PRESENT", passed=False)
    return PreexecutionCheck(code="V4_EXECUTION_RECEIPT_PRESENT", passed=True)


__all__ = [
    "PreexecutionCheck",
    "PreexecutionReport",
    "verify_v4_preexecution",
]
