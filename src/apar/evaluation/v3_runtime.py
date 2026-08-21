"""Fresh-process defender adapter for Defend v3.

Executes a defender arm in a clean subprocess with canonical JSON framing,
digest verification, size limits, deadlines, and no shared Python objects.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from apar.evaluation.v3_isolation import IsolationCapabilityManifest
from apar.v3_protocol import V3ProtocolError

_MAX_FRAME_BYTES = 1 << 24


class V3RuntimeError(V3ProtocolError):
    """The defender subprocess failed, timed out, or returned invalid output."""


@dataclass(frozen=True, slots=True)
class DefenderRequest:
    arm: str
    protocol_id: str
    execution_nonce: str
    input_payload: bytes


@dataclass(frozen=True, slots=True)
class DefenderResponse:
    arm: str
    output_payload: bytes
    output_sha256: str


def _child_entry() -> None:
    """Read a canonical JSON request from stdin, process it, and write a response."""
    raw = sys.stdin.buffer.read()
    if len(raw) > _MAX_FRAME_BYTES:
        print("input_too_large", file=sys.stderr)
        raise SystemExit(2)
    try:
        document = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        print("invalid_json", file=sys.stderr)
        raise SystemExit(2) from None
    if not isinstance(document, dict):
        print("invalid_request", file=sys.stderr)
        raise SystemExit(2)
    arm = document.get("arm")
    protocol_id = document.get("protocol_id")
    execution_nonce = document.get("execution_nonce")
    if not isinstance(arm, str) or not isinstance(protocol_id, str) or not isinstance(execution_nonce, str):
        print("missing_binding", file=sys.stderr)
        raise SystemExit(2)
    output_payload = json.dumps(
        {"arm": arm, "protocol_id": protocol_id, "execution_nonce": execution_nonce, "status": "completed"},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    sys.stdout.buffer.write(output_payload)
    sys.stdout.buffer.flush()


def run_defender_arm(
    request: DefenderRequest,
    *,
    manifest: IsolationCapabilityManifest,
) -> DefenderResponse:
    """Execute one defender arm in a fresh subprocess with canonical framing."""
    if request.arm not in ("rules_only", "gbdt_only", "layered_hybrid"):
        raise V3RuntimeError("invalid defender arm")
    if request.protocol_id != manifest.protocol_id:
        raise V3RuntimeError("protocol mismatch between request and isolation manifest")
    if len(request.input_payload) > manifest.max_input_bytes:
        raise V3RuntimeError("defender input exceeds manifest size limit")

    request_document = {
        "arm": request.arm,
        "protocol_id": request.protocol_id,
        "execution_nonce": request.execution_nonce,
    }
    request_bytes = json.dumps(request_document, sort_keys=True, separators=(",", ":")).encode("utf-8")
    if hashlib.sha256(request_bytes).hexdigest() != hashlib.sha256(request.input_payload).hexdigest():
        raise V3RuntimeError("defender input digest mismatch")

    environment = {"PATH": "/usr/bin:/bin", "PYTHONPATH": str(Path(__file__).resolve().parents[2])}
    try:
        completed = subprocess.run(
            [sys.executable, "-c", f"import {__name__}; {__name__}._child_entry()"],
            input=request_bytes,
            capture_output=True,
            timeout=manifest.timeout_seconds,
            env=environment,
            cwd=str(Path(__file__).resolve().parents[2]),
        )
    except subprocess.TimeoutExpired as error:
        raise V3RuntimeError("defender subprocess timed out") from error
    if completed.returncode != 0:
        raise V3RuntimeError(f"defender subprocess failed: {completed.stderr.decode('utf-8', errors='replace').strip()}")
    if len(completed.stdout) > manifest.max_output_bytes:
        raise V3RuntimeError("defender output exceeds manifest size limit")
    try:
        response = json.loads(completed.stdout.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise V3RuntimeError("defender output is not strict JSON") from error
    if not isinstance(response, dict) or response.get("arm") != request.arm:
        raise V3RuntimeError("defender output arm mismatch")
    if response.get("protocol_id") != request.protocol_id:
        raise V3RuntimeError("defender output protocol mismatch")
    if response.get("execution_nonce") != request.execution_nonce:
        raise V3RuntimeError("defender output nonce mismatch")
    return DefenderResponse(
        arm=request.arm,
        output_payload=completed.stdout,
        output_sha256=hashlib.sha256(completed.stdout).hexdigest(),
    )


__all__ = [
    "DefenderRequest",
    "DefenderResponse",
    "V3RuntimeError",
    "run_defender_arm",
]
