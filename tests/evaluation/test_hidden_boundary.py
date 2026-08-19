"""Frozen-only hidden release and static package-boundary attacks."""

from __future__ import annotations

import base64
import copy
import gc
import pickle
import weakref
from datetime import UTC, datetime
from pathlib import Path

import pytest

from apar.evaluation.defender_attestation import (
    DefenderAttestationError,
    DefenderBundleVerifier,
    VerifiedDefenderAttestation,
)
from apar.evaluation_hidden.defense_authority import (
    HiddenBoundaryError,
    HiddenEvaluationAuthority,
    audit_hidden_import_boundary,
)
from apar.runs.wire import canonical_json_bytes
from tests.defense.test_bundle import BundleFixture

pytest_plugins = ("tests.defense.test_bundle",)

ISSUED_AT = datetime(2026, 8, 19, 12, tzinfo=UTC)


def _freeze(fixture: BundleFixture):
    manifest, top_ref = fixture.publisher.freeze(**fixture.kwargs)
    return manifest, top_ref


def _verifier(fixture: BundleFixture) -> DefenderBundleVerifier:
    return DefenderBundleVerifier(
        fixture.store,
        signer_key_id=fixture.signer.key_id,
        public_key_base64=fixture.signer.public_key_base64,
    )


def _attested(fixture: BundleFixture):
    manifest, top_ref = _freeze(fixture)
    verifier = _verifier(fixture)
    return manifest, top_ref, verifier, verifier.attest(top_ref)


def test_verified_top_ref_is_sealed_into_an_opaque_capability(
    bundle_fixture: BundleFixture,
) -> None:
    manifest, top_ref, verifier, attestation = _attested(bundle_fixture)
    authority = HiddenEvaluationAuthority(verifier, bundle_fixture.store)

    capability = authority.freeze_and_issue(attestation, issued_at=ISSUED_AT)

    assert capability.bundle_manifest_digest == top_ref.sha256
    assert capability.bundle_id == manifest.bundle_id
    assert capability.issued_at == ISSUED_AT
    assert "hidden truth" not in repr(capability)


def test_manifest_substitution_invalid_signature_and_wrong_top_ref_fail_closed(
    bundle_fixture: BundleFixture,
) -> None:
    _, _, verifier, attestation = _attested(bundle_fixture)
    authority = HiddenEvaluationAuthority(verifier, bundle_fixture.store)

    with pytest.raises(HiddenBoundaryError, match="exact verified signed"):
        authority.freeze_and_issue(object(), issued_at=ISSUED_AT)  # type: ignore[arg-type]
    altered = attestation.to_json()[:-1] + b"0"
    with pytest.raises((DefenderAttestationError, HiddenBoundaryError)):
        VerifiedDefenderAttestation.from_json(altered, verifier=verifier)


def test_capability_is_immutable_unforgeable_single_authority_and_nonreplayable(
    bundle_fixture: BundleFixture,
) -> None:
    _, _, verifier, attestation = _attested(bundle_fixture)
    authority = HiddenEvaluationAuthority(verifier, bundle_fixture.store)
    capability = authority.freeze_and_issue(attestation, issued_at=ISSUED_AT)

    assert not isinstance(capability, bytes)
    with pytest.raises(TypeError):
        bytes(capability)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        tuple(capability)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        copy.copy(capability)
    with pytest.raises(TypeError):
        copy.deepcopy(capability)
    with pytest.raises(TypeError):
        pickle.dumps(capability)
    with pytest.raises((TypeError, HiddenBoundaryError)):
        type(capability)()
    with pytest.raises((TypeError, HiddenBoundaryError, AttributeError)):
        capability.__init__()
    with pytest.raises((TypeError, AttributeError)):
        capability.bundle_id = "substituted"
    with pytest.raises((TypeError, HiddenBoundaryError, AttributeError)):
        authority._restricted_store = bundle_fixture.store
    with pytest.raises((TypeError, HiddenBoundaryError, AttributeError)):
        object.__setattr__(authority, "_verifier", verifier)
    with pytest.raises((TypeError, HiddenBoundaryError)):
        authority.__init__(verifier, bundle_fixture.store)
    with pytest.raises(HiddenBoundaryError, match="already frozen"):
        authority.freeze_and_issue(attestation, issued_at=ISSUED_AT)
    with pytest.raises((TypeError, HiddenBoundaryError, AttributeError)):
        object.__setattr__(authority, "_active_capability_digest", None)
    with pytest.raises(TypeError, match="types are sealed"):
        type(authority).freeze_and_issue = lambda *args, **kwargs: capability
    with pytest.raises(TypeError, match="types are sealed"):
        type(capability).bundle_id = "substituted"
    with pytest.raises(HiddenBoundaryError, match="already frozen"):
        authority.freeze_and_issue(attestation, issued_at=ISSUED_AT)

    other = HiddenEvaluationAuthority(verifier, bundle_fixture.store)
    with pytest.raises(HiddenBoundaryError, match="already frozen"):
        authority.freeze_and_issue(attestation, issued_at=ISSUED_AT)
    assert other is not authority


def test_hidden_authority_rejects_nonexact_manifest_and_noncanonical_issue_time(
    bundle_fixture: BundleFixture,
) -> None:
    _, _, verifier, attestation = _attested(bundle_fixture)
    authority = HiddenEvaluationAuthority(verifier, bundle_fixture.store)

    with pytest.raises(HiddenBoundaryError):
        authority.freeze_and_issue(object(), issued_at=ISSUED_AT)  # type: ignore[arg-type]
    with pytest.raises(HiddenBoundaryError):
        authority.freeze_and_issue(
            attestation,
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
    (apar_root / "defense" / "relative.py").write_text(
        "from .. import evaluation_hidden\n"
    )
    (apar_root / "defense" / "assigned_alias.py").write_text(
        "import importlib as il\n"
        "load = il.import_module\n"
        "load('apar.evaluation_hidden.generator')\n"
    )
    (apar_root / "features" / "builtin_alias.py").write_text(
        "load = __import__\n"
        "load('apar.evaluation_hidden')\n"
    )
    (apar_root / "features" / "unresolved.py").write_text(
        "import importlib\n"
        "target = choose_module()\n"
        "importlib.import_module(target)\n"
    )
    (apar_root / "defense" / "nested_alias.py").write_text(
        "import importlib as il\n"
        "def load_hidden():\n"
        "    nested = il.import_module\n"
        "    nested('apar.evaluation_hidden.nested')\n"
    )
    (apar_root / "features" / "module_alias.py").write_text(
        "import importlib as il\n"
        "module_api = il\n"
        "load = module_api.import_module\n"
        "load('apar.evaluation_hidden.composed')\n"
    )
    (apar_root / "defense" / "getattr_alias.py").write_text(
        "import importlib as il\n"
        "load = getattr(il, 'import_module')\n"
        "load('apar.evaluation_hidden.getattr')\n"
    )
    (apar_root / "features" / "composed_reflection.py").write_text(
        "import importlib as il\n"
        "reflect = getattr\n"
        "attribute = 'import_' + 'module'\n"
        "load = reflect(il, attribute)\n"
        "load('apar.evaluation_hidden.composed_reflection')\n"
    )
    (apar_root / "defense" / "code_execution.py").write_text(
        "runner = compile\n"
        "runner('import apar.evaluation_hidden', '<attack>', 'exec')\n"
    )
    (apar_root / "features" / "nested_reflection.py").write_text(
        "loader = getattr(__import__('importlib'), 'import_module')\n"
        "loader('apar.evaluation_hidden.nested_reflection')\n"
    )

    result = audit_hidden_import_boundary(apar_root)

    assert not result.passed
    assert len(result.violations) >= 12
    assert any("composed_reflection.py" in item for item in result.violations)
    assert any("code_execution.py" in item for item in result.violations)
    assert any("nested_reflection.py" in item for item in result.violations)


def test_hidden_authority_state_registry_expires_without_reachable_mutable_state(
    bundle_fixture: BundleFixture,
) -> None:
    _, _, verifier, attestation = _attested(bundle_fixture)
    authority = HiddenEvaluationAuthority(verifier, bundle_fixture.store)
    capability = authority.freeze_and_issue(attestation, issued_at=ISSUED_AT)
    authority_ref = weakref.ref(authority)

    del authority
    gc.collect()

    assert authority_ref() is None
    assert capability.bundle_id == attestation.bundle_id


def test_hidden_authority_rejects_structurally_compatible_fake_loader(
    bundle_fixture: BundleFixture,
) -> None:
    class FakeLoader:
        def load(self, top_ref: object) -> object:
            del top_ref
            return object()

    with pytest.raises(HiddenBoundaryError, match="exact neutral verifier"):
        HiddenEvaluationAuthority(FakeLoader(), bundle_fixture.store)


def test_neutral_attestation_verifies_signature_top_ref_and_roundtrip(
    bundle_fixture: BundleFixture,
) -> None:
    from apar.evaluation.defender_attestation import (
        DefenderAttestationError,
        DefenderBundleVerifier,
    )

    manifest, top_ref = _freeze(bundle_fixture)
    verifier = DefenderBundleVerifier(
        bundle_fixture.store,
        signer_key_id=bundle_fixture.signer.key_id,
        public_key_base64=bundle_fixture.signer.public_key_base64,
    )
    attestation = verifier.attest(top_ref)

    assert verifier.attestation_from_json(attestation.to_json()) == attestation
    assert attestation.bundle_manifest_digest == top_ref.sha256
    with pytest.raises((AttributeError, TypeError)):
        object.__setattr__(verifier[0], "store", object())
    with pytest.raises((TypeError, DefenderAttestationError)):
        type(attestation)(attestation.to_json())
    with pytest.raises((AttributeError, TypeError, DefenderAttestationError)):
        type(attestation).model_construct(bundle_manifest_digest=top_ref.sha256)

    changed = manifest.model_copy(update={"corpus_digest": "f" * 64})
    invalid_ref = bundle_fixture.store.put_bytes(
        canonical_json_bytes(changed.model_dump(mode="json")), top_ref.media_type
    )
    with pytest.raises(DefenderAttestationError, match="signature"):
        verifier.attest(invalid_ref)


def test_neutral_attestation_derives_loadable_rollback_evidence(
    bundle_fixture: BundleFixture,
) -> None:
    from apar.evaluation.defender_attestation import (
        DefenderAttestationError,
        DefenderBundleVerifier,
    )

    manifest, predecessor_ref = _freeze(bundle_fixture)
    verifier = DefenderBundleVerifier(
        bundle_fixture.store,
        signer_key_id=bundle_fixture.signer.key_id,
        public_key_base64=bundle_fixture.signer.public_key_base64,
    )
    assert verifier.attest(predecessor_ref).rollback_available is False

    successor_kwargs = {
        **bundle_fixture.kwargs,
        "bundle_id": "22345678-1234-5678-9234-567812345678",
        "frozen_at": datetime(2026, 8, 19, 11, tzinfo=UTC),
        "rollback_ref": predecessor_ref.sha256,
    }
    _, successor_ref = bundle_fixture.publisher.freeze(**successor_kwargs)
    successor = verifier.attest(successor_ref)
    assert successor.rollback_available is True
    assert successor.rollback_predecessor_digest == predecessor_ref.sha256

    unsigned = manifest.model_copy(
        update={
            "rollback_ref": "f" * 64,
            "rollback_size_bytes": predecessor_ref.size_bytes,
            "signature_base64": base64.b64encode(b"\x00" * 64).decode("ascii"),
        }
    )
    missing = unsigned.model_copy(
        update={
            "signature_base64": bundle_fixture.signer.sign(
                unsigned.unsigned_document()
            )
        }
    )
    missing_ref = bundle_fixture.store.put_bytes(
        canonical_json_bytes(missing.model_dump(mode="json")),
        predecessor_ref.media_type,
    )
    with pytest.raises(DefenderAttestationError, match="rollback"):
        verifier.attest(missing_ref)


def test_public_hidden_boundary_never_exposes_restricted_payload_bytes() -> None:
    """Removing the resolver must make payload exfiltration impossible by API use."""
    import apar.evaluation_hidden.defense_authority as authority_module

    assert not hasattr(authority_module, "resolve_hidden_release")
    assert not hasattr(authority_module, "ResolvedHiddenEvaluation")
    assert not hasattr(authority_module, "_AUTHORITIES")
    assert not hasattr(authority_module, "_CAPABILITIES")
    assert not hasattr(authority_module, "_REQUESTS")


def test_verified_attestation_is_not_a_bytes_subtype(
    bundle_fixture: BundleFixture,
) -> None:
    """A bytes constructor bypass must not produce the verifier's proof type."""
    _, _, verifier, attestation = _attested(bundle_fixture)

    assert not isinstance(attestation, bytes)
    forged = bytes(attestation.to_json())
    assert not verifier.verify(forged)
