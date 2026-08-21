"""Process-isolation capability manifest tests."""

from __future__ import annotations

import pytest

from apar.evaluation.v3_isolation import (
    V3IsolationError,
    build_isolation_manifest,
    verify_loaded_modules,
)


def test_manifest_binds_exact_forbidden_modules() -> None:
    manifest = build_isolation_manifest(protocol_id="apar-defend-v3")
    assert manifest.forbidden_modules == (
        "apar.evaluation",
        "apar.evaluation_hidden",
        "apar.runs",
        "apar.redteam",
    )


def test_all_isolation_flags_enabled() -> None:
    manifest = build_isolation_manifest(protocol_id="apar-defend-v3")
    assert manifest.no_network
    assert manifest.no_shared_memory
    assert manifest.no_pickle
    assert manifest.no_signing_key
    assert manifest.no_seed_material
    assert manifest.no_receipt_store
    assert manifest.canonical_io_only


def test_partial_flags_rejected() -> None:
    base = build_isolation_manifest(protocol_id="apar-defend-v3").model_dump(mode="python")
    base["no_network"] = False
    with pytest.raises(ValueError, match="all isolation flags"):
        __import__("apar.evaluation.v3_isolation", fromlist=["IsolationCapabilityManifest"]).IsolationCapabilityManifest.model_validate(base)


def test_loaded_evaluator_module_rejected() -> None:
    with pytest.raises(V3IsolationError, match="forbidden evaluator module"):
        verify_loaded_modules({"apar.evaluation.v2_controls"})


def test_clean_module_set_accepted() -> None:
    verify_loaded_modules({"json", "os", "apar.defense.rules"})
