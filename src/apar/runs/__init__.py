"""Authenticated deterministic run orchestration."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from apar.runs.runner import (
        AttackerPolicy,
        AttackerPolicyKind,
        PolicyWorkerBoundaryReport,
        PolicyWorkerClient,
        PolicyWorkerError,
        PublicRunManifest,
        RunExecutionError,
        RunManifest,
        RunRunner,
        RunSigningIdentity,
        ScenarioRunBinding,
        SignedRunReceipt,
        bind_scenario_for_run,
    )

__all__ = [
    "AttackerPolicy",
    "AttackerPolicyKind",
    "PolicyWorkerBoundaryReport",
    "PolicyWorkerClient",
    "PolicyWorkerError",
    "PublicRunManifest",
    "RunExecutionError",
    "RunManifest",
    "RunRunner",
    "RunSigningIdentity",
    "SignedRunReceipt",
    "ScenarioRunBinding",
    "bind_scenario_for_run",
]


def __getattr__(name: str) -> object:
    """Load parent orchestration only when a caller asks for a public runner symbol."""
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(import_module("apar.runs.runner"), name)
