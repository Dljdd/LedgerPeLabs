"""Frozen-only hidden release and static package-boundary attacks."""

from __future__ import annotations

import copy
from datetime import UTC, datetime
from pathlib import Path

import pytest

from apar.defense.bundle import DefenderBundleManifest
from apar.evaluation_hidden.defense_authority import (
    HiddenBoundaryError,
    HiddenEvaluationAuthority,
    audit_hidden_import_boundary,
)
from apar.storage.artifacts import ArtifactStore
from tests.defense.test_bundle import BundleFixture

pytest_plugins = ("tests.defense.test_bundle",)

ISSUED_AT = datetime(2026, 8, 19, 12, tzinfo=UTC)


def _freeze(fixture: BundleFixture):
    manifest, top_ref = fixture.publisher.freeze(**fixture.kwargs)
    return manifest, top_ref


def test_hidden_reference_cannot_resolve_before_bundle_freeze(
    bundle_fixture: BundleFixture,
) -> None:
    restricted_ref = bundle_fixture.store.put_bytes(b"hidden truth", "application/json")
    authority = HiddenEvaluationAuthority(bundle_fixture.publisher, bundle_fixture.store)

    with pytest.raises(HiddenBoundaryError, match="frozen defender"):
        authority.resolve(None, restricted_ref)


def test_verified_top_ref_is_sealed_before_restricted_resolution(
    bundle_fixture: BundleFixture,
) -> None:
    manifest, top_ref = _freeze(bundle_fixture)
    restricted_ref = bundle_fixture.store.put_bytes(b"hidden truth", "application/json")
    authority = HiddenEvaluationAuthority(bundle_fixture.publisher, bundle_fixture.store)

    capability = authority.freeze_and_issue(manifest, top_ref, issued_at=ISSUED_AT)

    assert authority.resolve(capability, restricted_ref) == b"hidden truth"
    assert capability.bundle_manifest_digest == top_ref.sha256
    assert capability.bundle_id == manifest.bundle_id
    assert capability.issued_at == ISSUED_AT
    assert "hidden truth" not in repr(capability)


def test_manifest_substitution_invalid_signature_and_wrong_top_ref_fail_closed(
    bundle_fixture: BundleFixture,
) -> None:
    manifest, top_ref = _freeze(bundle_fixture)
    changed = manifest.model_copy(update={"corpus_digest": "f" * 64})

    with pytest.raises(HiddenBoundaryError, match="verified signed frozen defender"):
        HiddenEvaluationAuthority(
            bundle_fixture.publisher, bundle_fixture.store
        ).freeze_and_issue(changed, top_ref, issued_at=ISSUED_AT)
    with pytest.raises(HiddenBoundaryError, match="verified signed frozen defender"):
        HiddenEvaluationAuthority(
            bundle_fixture.publisher, bundle_fixture.store
        ).freeze_and_issue(manifest, manifest.component("rules"), issued_at=ISSUED_AT)


def test_capability_is_immutable_unforgeable_single_authority_and_nonreplayable(
    bundle_fixture: BundleFixture,
) -> None:
    manifest, top_ref = _freeze(bundle_fixture)
    authority = HiddenEvaluationAuthority(bundle_fixture.publisher, bundle_fixture.store)
    capability = authority.freeze_and_issue(manifest, top_ref, issued_at=ISSUED_AT)

    assert copy.copy(capability) is capability
    assert copy.deepcopy(capability) is capability
    with pytest.raises((TypeError, HiddenBoundaryError)):
        type(capability)(capability.to_bytes())
    with pytest.raises((TypeError, HiddenBoundaryError, AttributeError)):
        capability.__init__(capability.to_bytes())
    with pytest.raises((TypeError, AttributeError)):
        capability.bundle_id = "substituted"
    with pytest.raises((TypeError, HiddenBoundaryError, AttributeError)):
        authority._restricted_store = bundle_fixture.store
    with pytest.raises((TypeError, HiddenBoundaryError)):
        authority.__init__(bundle_fixture.publisher, bundle_fixture.store)
    with pytest.raises(HiddenBoundaryError, match="already frozen"):
        authority.freeze_and_issue(manifest, top_ref, issued_at=ISSUED_AT)

    other = HiddenEvaluationAuthority(bundle_fixture.publisher, bundle_fixture.store)
    ref = bundle_fixture.store.put_bytes(b"restricted", "application/json")
    with pytest.raises(HiddenBoundaryError, match="frozen defender|capability"):
        other.resolve(capability, ref)


def test_wrong_restricted_store_and_forged_ref_do_not_resolve(
    tmp_path: Path, bundle_fixture: BundleFixture
) -> None:
    manifest, top_ref = _freeze(bundle_fixture)
    authority = HiddenEvaluationAuthority(bundle_fixture.publisher, bundle_fixture.store)
    capability = authority.freeze_and_issue(manifest, top_ref, issued_at=ISSUED_AT)
    other_store = ArtifactStore(tmp_path / "wrong-store")
    wrong_ref = other_store.put_bytes(b"hidden truth", "application/json")

    with pytest.raises(HiddenBoundaryError, match="restricted reference"):
        authority.resolve(capability, wrong_ref)


def test_hidden_authority_rejects_nonexact_manifest_and_noncanonical_issue_time(
    bundle_fixture: BundleFixture,
) -> None:
    manifest, top_ref = _freeze(bundle_fixture)
    authority = HiddenEvaluationAuthority(bundle_fixture.publisher, bundle_fixture.store)
    constructed = manifest.model_dump()
    constructed["threshold_digest"] = "f" * 64

    with pytest.raises(HiddenBoundaryError):
        authority.freeze_and_issue(
            DefenderBundleManifest.model_construct(**constructed),
            top_ref,
            issued_at=ISSUED_AT,
        )
    with pytest.raises(HiddenBoundaryError):
        authority.freeze_and_issue(
            manifest,
            top_ref,
            issued_at=datetime(2026, 8, 19, 12),
        )


def test_defense_and_features_cannot_import_hidden_package(tmp_path: Path) -> None:
    source_root = Path(__file__).resolve().parents[2] / "src" / "apar"
    assert audit_hidden_import_boundary(source_root).passed

    apar_root = tmp_path / "apar"
    (apar_root / "defense").mkdir(parents=True)
    (apar_root / "features").mkdir()
    (apar_root / "defense" / "bad.py").write_text(
        "from apar.evaluation_hidden import defense_authority\n"
    )
    (apar_root / "features" / "dynamic.py").write_text(
        "from importlib import import_module as load\n"
        "load('apar.evaluation_hidden.generator')\n"
    )
    (apar_root / "features" / "from_apar.py").write_text(
        "from apar import evaluation_hidden\n"
    )

    result = audit_hidden_import_boundary(apar_root)

    assert not result.passed
    assert len(result.violations) == 3


def test_hidden_authority_does_not_translate_bundle_integrity_failures_to_success(
    bundle_fixture: BundleFixture,
) -> None:
    manifest, top_ref = _freeze(bundle_fixture)
    bundle_fixture.publisher.close()

    with pytest.raises(HiddenBoundaryError, match="verified signed frozen defender"):
        HiddenEvaluationAuthority(
            bundle_fixture.publisher, bundle_fixture.store
        ).freeze_and_issue(manifest, top_ref, issued_at=ISSUED_AT)
