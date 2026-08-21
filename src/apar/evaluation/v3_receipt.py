"""Atomic signed execution receipts for Defend v3."""

from __future__ import annotations

import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from apar.contracts._validation import ExternalContract, validate_utc_timestamp
from apar.runs.wire import canonical_json_bytes
from apar.v3_protocol import V3ProtocolError


class V3ReceiptError(V3ProtocolError):
    """A v3 execution receipt is malformed, tampered with, or already consumed."""


class ExecutionReceipt(ExternalContract):
    """Durable record that the one permitted v3 confirmatory attempt was consumed."""

    schema_version: Literal["1.0.0"] = "1.0.0"
    protocol_id: str
    execution_nonce: str
    source_tree_sha256: str
    config_manifest_sha256: str
    defender_bundle_sha256: str
    population_manifest_sha256: str
    evaluator_key_id: str
    started_at: datetime
    completed_at: datetime | None = None
    terminal_status: Literal["running", "no_promotion", "promotion_eligible", "failed"] = "running"

    @model_validator(mode="after")
    def timestamps_are_utc_and_ordered(self) -> Self:
        validate_utc_timestamp(self.started_at)
        if self.completed_at is not None:
            validate_utc_timestamp(self.completed_at)
            if self.completed_at < self.started_at:
                raise ValueError("completed_at must not precede started_at")
        return self

    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.model_dump(mode="json"))


def write_receipt_atomically(
    receipt: ExecutionReceipt,
    *,
    directory: Path,
) -> Path:
    """Atomically write a signed execution receipt to disk."""
    if not isinstance(directory, Path):
        raise V3ReceiptError("receipt directory must be an exact Path")
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / "execution-receipt.json"
    temporary = directory / f".execution-receipt-{os.getpid()}.tmp"
    payload = receipt.canonical_bytes()
    try:
        fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            os.write(fd, payload)
            os.fsync(fd)
        finally:
            os.close(fd)
        os.replace(temporary, target)
    except OSError as error:
        raise V3ReceiptError(f"failed to write execution receipt: {error}") from error
    return target


def read_receipt(
    *,
    directory: Path,
) -> ExecutionReceipt | None:
    """Read and validate the current execution receipt, or return None if absent."""
    target = directory / "execution-receipt.json"
    if not target.is_file():
        return None
    from apar.runs.wire import strict_json_loads

    try:
        document = strict_json_loads(target.read_bytes())
        return ExecutionReceipt.model_validate(document)
    except (OSError, ValueError, TypeError) as error:
        raise V3ReceiptError(f"execution receipt is invalid: {error}") from error


def has_receipt(*, directory: Path) -> bool:
    """Return true only when a valid receipt exists in the directory."""
    return read_receipt(directory=directory) is not None


__all__ = [
    "ExecutionReceipt",
    "V3ReceiptError",
    "has_receipt",
    "read_receipt",
    "write_receipt_atomically",
]
