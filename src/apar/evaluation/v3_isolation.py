"""Process-isolation capability manifest for Defend v3."""

from __future__ import annotations

import hashlib
from typing import Literal

from pydantic import Field, model_validator

from apar.contracts._validation import ExternalContract
from apar.v3_protocol import V3ProtocolError

_FORBIDDEN_MODULES = (
    "apar.evaluation",
    "apar.evaluation_hidden",
    "apar.runs",
    "apar.redteam",
)


class V3IsolationError(V3ProtocolError):
    """The defender subprocess violates the v3 isolation capability contract."""


class IsolationCapabilityManifest(ExternalContract):
    """Declares and binds the exact fresh-process isolation requirements."""

    schema_version: Literal["1.0.0"] = "1.0.0"
    protocol_id: str
    forbidden_modules: tuple[str, ...]
    no_network: bool = True
    no_shared_memory: bool = True
    no_pickle: bool = True
    no_signing_key: bool = True
    no_seed_material: bool = True
    no_receipt_store: bool = True
    canonical_io_only: bool = True
    max_input_bytes: int = Field(gt=0, le=1 << 24)
    max_output_bytes: int = Field(gt=0, le=1 << 24)
    timeout_seconds: float = Field(gt=0, le=300)
    manifest_digest: str

    @model_validator(mode="after")
    def forbidden_modules_are_exact(self) -> Self:
        if tuple(self.forbidden_modules) != _FORBIDDEN_MODULES:
            raise ValueError("isolation manifest must list exact forbidden modules")
        if not all(
            (
                self.no_network,
                self.no_shared_memory,
                self.no_pickle,
                self.no_signing_key,
                self.no_seed_material,
                self.no_receipt_store,
                self.canonical_io_only,
            )
        ):
            raise ValueError("all isolation flags must be enabled")
        return self


def build_isolation_manifest(
    *,
    protocol_id: str,
    max_input_bytes: int = 1 << 20,
    max_output_bytes: int = 1 << 20,
    timeout_seconds: float = 30.0,
) -> IsolationCapabilityManifest:
    """Build a signed-capability manifest for one v3 execution context."""
    if type(protocol_id) is not str or not protocol_id:
        raise V3IsolationError("protocol_id must be nonempty")
    unsigned = {
        "schema_version": "1.0.0",
        "protocol_id": protocol_id,
        "forbidden_modules": list(_FORBIDDEN_MODULES),
        "no_network": True,
        "no_shared_memory": True,
        "no_pickle": True,
        "no_signing_key": True,
        "no_seed_material": True,
        "no_receipt_store": True,
        "canonical_io_only": True,
        "max_input_bytes": max_input_bytes,
        "max_output_bytes": max_output_bytes,
        "timeout_seconds": timeout_seconds,
    }
    digest = hashlib.sha256(
        __import__("json").dumps(unsigned, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return IsolationCapabilityManifest.model_validate({**unsigned, "manifest_digest": digest})


def verify_loaded_modules(loaded_module_names: set[str]) -> None:
    """Reject any child process that has loaded a forbidden evaluator module."""
    for name in loaded_module_names:
        for root in _FORBIDDEN_MODULES:
            if name == root or name.startswith(root + "."):
                raise V3IsolationError(f"forbidden evaluator module loaded: {name}")


__all__ = [
    "IsolationCapabilityManifest",
    "V3IsolationError",
    "build_isolation_manifest",
    "verify_loaded_modules",
]
