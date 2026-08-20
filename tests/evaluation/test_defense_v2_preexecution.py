"""Read-only admission checks for the sealed Defend v2 protocol."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pytest

from apar.evaluation.gates import EvaluatorSigningIdentity
from apar.evaluation.v2_preexecution import verify_manifest_registry, verify_v2_preexecution
from apar.evaluation.v2_preregistration import V2Preregistration, sign_v2_preregistration
from apar.runs.wire import canonical_json_bytes

ROOT = Path(__file__).resolve().parents[2]
PROFILE = json.loads((ROOT / "config/defense/competition-v2-profile.json").read_bytes())
MANIFEST_REGISTRY = json.loads((ROOT / "config/defense/competition-v2-manifests.json").read_bytes())


def test_readme_makes_no_v2_efficacy_claim() -> None:
    text = (ROOT / "README.md").read_text(encoding="utf-8").lower()

    assert "defend v2: protocol sealed; evaluation not executed" in text
    assert "defend v2 achieved" not in text


def test_signed_preregistration_and_frozen_v1_roots_are_not_executed() -> None:
    """A sealed, unconsumed protocol is reported without starting any work."""
    report = verify_v2_preexecution(ROOT, signed_preregistration())

    assert (report.status, report.codes) == ("not_executed", ())


def test_hidden_import_in_defender_fails_preexecution(tmp_path: Path) -> None:
    """Defender code must not gain a path to evaluator-only modules."""
    (tmp_path / "src/apar/defense").mkdir(parents=True)
    (tmp_path / "src/apar/defense/bad.py").write_text(
        "from apar.evaluation_hidden import worker\n", encoding="utf-8"
    )

    report = verify_v2_preexecution(tmp_path, signed_preregistration())

    assert "HIDDEN_IMPORT_BOUNDARY" in report.codes


def test_non_v2_evaluator_import_in_defender_fails_preexecution(tmp_path: Path) -> None:
    """Only explicitly versioned public evaluator contracts are defender-visible."""
    _write_defender_source(tmp_path, "from apar.evaluation import service\n")

    report = verify_v2_preexecution(tmp_path, signed_preregistration())

    assert "HIDDEN_IMPORT_BOUNDARY" in report.codes


def test_dynamic_hidden_import_expression_fails_preexecution(tmp_path: Path) -> None:
    """A computed module name cannot evade the static evaluator boundary."""
    _write_defender_source(tmp_path, "__import__('apar.' + 'evaluation_hidden')\n")

    report = verify_v2_preexecution(tmp_path, signed_preregistration())

    assert "HIDDEN_IMPORT_BOUNDARY" in report.codes


def test_dynamic_importlib_module_expression_fails_preexecution(tmp_path: Path) -> None:
    """Importlib aliases cannot make a computed evaluator module admissible."""
    _write_defender_source(
        tmp_path,
        "import importlib as loader\n"
        "module = 'apar.evaluation_hidden'\n"
        "loader.import_module(module)\n",
    )

    report = verify_v2_preexecution(tmp_path, signed_preregistration())

    assert "HIDDEN_IMPORT_BOUNDARY" in report.codes


def test_importlib_metadata_binds_the_importlib_root(tmp_path: Path) -> None:
    """A submodule import still exposes the importlib root capability."""
    _write_defender_source(
        tmp_path,
        "import importlib.metadata\n"
        "module = 'apar.evaluation_hidden'\n"
        "importlib.import_module(module)\n",
    )

    report = verify_v2_preexecution(tmp_path, signed_preregistration())

    assert "HIDDEN_IMPORT_BOUNDARY" in report.codes


def test_importlib_metadata_alias_cannot_import_a_computed_target(tmp_path: Path) -> None:
    """Aliases of importlib submodules remain dynamic import capabilities."""
    _write_defender_source(
        tmp_path,
        "import importlib.metadata as loader\n"
        "module = 'apar.evaluation_hidden'\n"
        "loader.import_module(module)\n",
    )

    report = verify_v2_preexecution(tmp_path, signed_preregistration())

    assert "HIDDEN_IMPORT_BOUNDARY" in report.codes


def test_importlib_root_assignment_cannot_import_a_computed_target(tmp_path: Path) -> None:
    """Assignments retain the dynamic import capability of an importlib root."""
    _write_defender_source(
        tmp_path,
        "import importlib\n"
        "loader = importlib\n"
        "module = 'apar.evaluation_hidden'\n"
        "loader.import_module(module)\n",
    )

    report = verify_v2_preexecution(tmp_path, signed_preregistration())

    assert "HIDDEN_IMPORT_BOUNDARY" in report.codes


def test_builtins_import_alias_cannot_import_a_computed_target(tmp_path: Path) -> None:
    """Aliasing the built-in import function cannot evade dynamic-import checks."""
    _write_defender_source(
        tmp_path,
        "from builtins import __import__ as load\n"
        "module = 'apar.evaluation_hidden'\n"
        "load(module)\n",
    )

    report = verify_v2_preexecution(tmp_path, signed_preregistration())

    assert "HIDDEN_IMPORT_BOUNDARY" in report.codes


def test_assigned_builtins_import_capability_is_rejected(tmp_path: Path) -> None:
    """Assignments of builtins.__import__ remain dynamic-import capabilities."""
    _write_defender_source(
        tmp_path,
        "import builtins as runtime\n"
        "load = runtime.__import__\n"
        "module = 'apar.evaluation_hidden'\n"
        "load(module)\n",
    )

    report = verify_v2_preexecution(tmp_path, signed_preregistration())

    assert "HIDDEN_IMPORT_BOUNDARY" in report.codes


def test_builtins_import_attribute_cannot_import_a_computed_target(tmp_path: Path) -> None:
    """A direct builtins.__import__ call remains a dynamic import operation."""
    _write_defender_source(
        tmp_path,
        "import builtins as runtime\n"
        "module = 'apar.evaluation_hidden'\n"
        "runtime.__import__(module)\n",
    )

    report = verify_v2_preexecution(tmp_path, signed_preregistration())

    assert "HIDDEN_IMPORT_BOUNDARY" in report.codes


def test_getattr_builtins_import_alias_is_rejected(tmp_path: Path) -> None:
    """Reflective aliases of builtins.__import__ cannot bypass the boundary."""
    _write_defender_source(
        tmp_path,
        "import builtins as runtime\n"
        "load = getattr(runtime, '__import__')\n"
        "module = 'apar.evaluation_hidden'\n"
        "load(module)\n",
    )

    report = verify_v2_preexecution(tmp_path, signed_preregistration())

    assert "HIDDEN_IMPORT_BOUNDARY" in report.codes


@pytest.mark.parametrize(
    "source",
    (
        "import builtins\nmodule = 'apar.evaluation_hidden'\n"
        "getattr(builtins, '__import__')(module)\n",
        "import builtins\nmodule = 'apar.evaluation_hidden'\n"
        "builtins.__dict__['__import__'](module)\n",
        "import importlib\nmodule = 'apar.evaluation_hidden'\n"
        "getattr(importlib, 'import_module')(module)\n",
        "import importlib\nmodule = 'apar.evaluation_hidden'\n"
        "importlib.__dict__['import_module'](module)\n",
        "import builtins\nruntime = builtins\nmodule = 'apar.evaluation_hidden'\n"
        "runtime.__import__(module)\n",
        "import importlib\nnamespace = importlib.__dict__\n"
        "module = 'apar.evaluation_hidden'\nnamespace['import_module'](module)\n",
        "import builtins\nnamespace = builtins.__dict__\n"
        "module = 'apar.evaluation_hidden'\nnamespace['__import__'](module)\n",
        "import importlib\nnamespace = vars(importlib)\n"
        "module = 'apar.evaluation_hidden'\nnamespace['import_module'](module)\n",
    ),
)
def test_reflective_dynamic_import_capabilities_fail_closed(tmp_path: Path, source: str) -> None:
    """Reflective access to either import capability cannot evade inspection."""
    _write_defender_source(tmp_path, source)

    report = verify_v2_preexecution(tmp_path, signed_preregistration())

    assert "HIDDEN_IMPORT_BOUNDARY" in report.codes


def test_transitive_defender_feature_import_is_scanned(tmp_path: Path) -> None:
    """A hidden import in a defender-reachable feature module must fail closed."""
    _write_defender_source(tmp_path, "from apar.features import bridge\n")
    features = tmp_path / "src/apar/features"
    features.mkdir(parents=True)
    (features / "__init__.py").write_text("", encoding="utf-8")
    (features / "bridge.py").write_text(
        "from apar.features import hidden_bridge\n", encoding="utf-8"
    )
    (features / "hidden_bridge.py").write_text(
        "from apar.evaluation_hidden import worker\n", encoding="utf-8"
    )

    report = verify_v2_preexecution(tmp_path, signed_preregistration())

    assert "HIDDEN_IMPORT_BOUNDARY" in report.codes


def test_constant_dynamic_feature_import_enters_transitive_scan(tmp_path: Path) -> None:
    """A literal dynamic feature import must not hide that module's imports."""
    _write_defender_source(tmp_path, "__import__('apar.features.dynamic_bridge')\n")
    features = tmp_path / "src/apar/features"
    features.mkdir(parents=True)
    (features / "dynamic_bridge.py").write_text(
        "from apar.evaluation_hidden import worker\n", encoding="utf-8"
    )

    report = verify_v2_preexecution(tmp_path, signed_preregistration())

    assert "HIDDEN_IMPORT_BOUNDARY" in report.codes


def test_feature_reachable_local_package_enters_transitive_scan(tmp_path: Path) -> None:
    """Feature dependencies outside apar.features remain defender-reachable code."""
    _write_defender_source(tmp_path, "from apar.features import bridge\n")
    features = tmp_path / "src/apar/features"
    features.mkdir(parents=True)
    (features / "bridge.py").write_text("from apar.shared import bridge\n", encoding="utf-8")
    shared = tmp_path / "src/apar/shared"
    shared.mkdir(parents=True)
    (shared / "bridge.py").write_text(
        "from apar.evaluation_hidden import worker\n", encoding="utf-8"
    )

    report = verify_v2_preexecution(tmp_path, signed_preregistration())

    assert "HIDDEN_IMPORT_BOUNDARY" in report.codes


def test_relative_evaluator_import_fails_preexecution(tmp_path: Path) -> None:
    """Relative imports cannot reach an evaluator module outside the v2 namespace."""
    _write_defender_source(tmp_path, "from ..evaluation import hidden_source\n")

    report = verify_v2_preexecution(tmp_path, signed_preregistration())

    assert "HIDDEN_IMPORT_BOUNDARY" in report.codes


def test_relative_evaluation_package_import_fails_preexecution(tmp_path: Path) -> None:
    """Importing the evaluator package itself bypasses no version boundary."""
    _write_defender_source(tmp_path, "from .. import evaluation\n")

    report = verify_v2_preexecution(tmp_path, signed_preregistration())

    assert "HIDDEN_IMPORT_BOUNDARY" in report.codes


def test_apar_package_evaluator_import_fails_preexecution(tmp_path: Path) -> None:
    """A package-level evaluator import cannot bypass the versioned namespace."""
    _write_defender_source(tmp_path, "from apar import evaluation\n")

    report = verify_v2_preexecution(tmp_path, signed_preregistration())

    assert "HIDDEN_IMPORT_BOUNDARY" in report.codes


def test_relative_versioned_evaluator_import_remains_admissible(tmp_path: Path) -> None:
    """Relative syntax may still name an explicitly versioned public contract."""
    _write_defender_source(tmp_path, "from ..evaluation import v2_preexecution\n")

    report = verify_v2_preexecution(tmp_path, signed_preregistration())

    assert "HIDDEN_IMPORT_BOUNDARY" not in report.codes


def test_versioned_public_evaluator_import_remains_admissible(tmp_path: Path) -> None:
    """The sealed pre-execution public contract remains an approved dependency."""
    _write_defender_source(
        tmp_path, "from apar.evaluation.v2_preexecution import PreexecutionReport\n"
    )

    report = verify_v2_preexecution(tmp_path, signed_preregistration())

    assert "HIDDEN_IMPORT_BOUNDARY" not in report.codes


def test_existing_v2_receipt_fails_preexecution(tmp_path: Path) -> None:
    """A consumed confirmatory attempt cannot be represented as pre-execution."""
    (tmp_path / ".apar/defense-v2").mkdir(parents=True)
    (tmp_path / ".apar/defense-v2/execution-receipt.json").write_text(
        '{"execution_nonce":"nonce","preregistration_id":"apar-defend-v2","schema_version":"1.0.0"}',
        encoding="utf-8",
    )

    report = verify_v2_preexecution(tmp_path, signed_preregistration())

    assert "V2_EXECUTION_RECEIPT_PRESENT" in report.codes


def test_v2_receipt_schema_is_found_without_a_receipt_filename(tmp_path: Path) -> None:
    """Durable v2 receipts are identified by schema anywhere in application state."""
    stored = tmp_path / ".apar" / "unrelated" / "nested" / "state.bin"
    stored.parent.mkdir(parents=True)
    stored.write_text(
        '{"execution_nonce":"nonce","preregistration_id":"apar-defend-v2","schema_version":"1.0.0"}',
        encoding="utf-8",
    )

    report = verify_v2_preexecution(tmp_path, signed_preregistration())

    assert "V2_EXECUTION_RECEIPT_PRESENT" in report.codes


def test_invalid_signature_fails_preexecution() -> None:
    """Unsafe model copies cannot turn an invalid admission into a pass."""
    preregistration = signed_preregistration().model_copy(update={"signature_base64": ""})

    report = verify_v2_preexecution(ROOT, preregistration)

    assert "PREREGISTRATION_INVALID" in report.codes


def test_profile_digest_mismatch_fails_preexecution() -> None:
    """A valid signature cannot substitute a different V2 profile."""
    payload = _preregistration_payload()
    payload["protocol_profile_sha256"] = _digest("substituted-profile")
    preregistration = sign_v2_preregistration(
        payload,
        signer=EvaluatorSigningIdentity.from_private_bytes(b"v" * 32),
    )

    report = verify_v2_preexecution(ROOT, preregistration)

    assert "PROTOCOL_PROFILE_INVALID" in report.codes


def test_digest_shaped_bindings_without_real_manifest_fail_preexecution() -> None:
    """A signature over arbitrary digest strings is not manifest evidence."""
    payload = _preregistration_payload()
    payload["manifest_registry_sha256"] = _digest("missing-manifest-registry")
    preregistration = sign_v2_preregistration(
        payload,
        signer=EvaluatorSigningIdentity.from_private_bytes(b"v" * 32),
    )

    report = verify_v2_preexecution(ROOT, preregistration)

    assert "MANIFEST_BINDINGS_INVALID" in report.codes


def test_seed_commitments_must_match_the_committed_profile() -> None:
    """Re-signing different evaluator seed commitments cannot authorize V2."""
    payload = _preregistration_payload()
    payload["seed_commitments"] = (
        {"name": "operating_population", "commitment_sha256": _digest("other-seed")},
        {"name": "campaign_injection", "commitment_sha256": "2" * 64},
    )
    preregistration = sign_v2_preregistration(
        payload,
        signer=EvaluatorSigningIdentity.from_private_bytes(b"v" * 32),
    )

    report = verify_v2_preexecution(ROOT, preregistration)

    assert "PROTOCOL_PROFILE_INVALID" in report.codes


def test_budget_binding_must_match_the_committed_profile() -> None:
    """A signed substitute budget cannot weaken the frozen profile limits."""
    payload = _preregistration_payload()
    payload["budget_manifest_sha256"] = _digest("weaker-budget")
    preregistration = sign_v2_preregistration(
        payload,
        signer=EvaluatorSigningIdentity.from_private_bytes(b"v" * 32),
    )

    report = verify_v2_preexecution(ROOT, preregistration)

    assert "PROTOCOL_PROFILE_INVALID" in report.codes
    assert "MANIFEST_BINDINGS_INVALID" in report.codes


def test_each_required_component_manifest_is_load_bearing() -> None:
    """A signed control-manifest substitution must differ from the real registry."""
    payload = _preregistration_payload()
    payload["controls_manifest_sha256"] = _digest("substituted-controls")
    preregistration = sign_v2_preregistration(
        payload,
        signer=EvaluatorSigningIdentity.from_private_bytes(b"v" * 32),
    )

    report = verify_v2_preexecution(ROOT, preregistration)

    assert "MANIFEST_BINDINGS_INVALID" in report.codes


@pytest.mark.parametrize(
    "relative_path",
    (
        "src/apar/defense/policy.py",
        "config/defense/feature-catalog.json",
        "fixtures/defense/v1/defender-bundle.json",
        "fixtures/defense/v1/split-manifest.json",
        "fixtures/defense/v1/thresholds.json",
    ),
)
def test_manifest_registry_rejects_modified_frozen_input(
    tmp_path: Path, relative_path: str
) -> None:
    """Every named source, catalog, bundle, ledger, and threshold byte is load-bearing."""
    _copy_manifest_inputs(tmp_path)
    target = tmp_path / relative_path
    target.write_bytes(target.read_bytes() + b"\n")

    check = verify_manifest_registry(tmp_path, signed_preregistration())

    assert check.passed is False


def signed_preregistration() -> V2Preregistration:
    signer = EvaluatorSigningIdentity.from_private_bytes(b"v" * 32)
    return sign_v2_preregistration(_preregistration_payload(), signer=signer)


def _preregistration_payload() -> dict[str, object]:
    synthetic_scope = (
        "Synthetic-only evaluation; not a real-world prevalence or external-validity claim."
    )
    return {
        "schema_version": "1.0.0",
        "preregistration_id": "apar-defend-v2",
        "protocol_profile_sha256": PROFILE["profile_sha256"],
        "manifest_registry_sha256": _manifest_registry_digest(),
        "source_manifest_sha256": _manifest_digest("source"),
        "feature_manifest_sha256": _manifest_digest("feature"),
        "candidate_grid_sha256": _manifest_digest("candidate_grid"),
        "population_manifest_sha256": _manifest_digest("population"),
        "seed_commitments": (
            {"name": "operating_population", "commitment_sha256": "1" * 64},
            {"name": "campaign_injection", "commitment_sha256": "2" * 64},
        ),
        "evaluator_capability_sha256": _manifest_digest("evaluator_capability"),
        "metrics_manifest_sha256": _manifest_digest("metrics"),
        "bootstrap_manifest_sha256": _manifest_digest("bootstrap"),
        "controls_manifest_sha256": _manifest_digest("controls"),
        "budget_manifest_sha256": _manifest_digest("budget"),
        "reporting_schema_sha256": _manifest_digest("reporting_schema"),
        "fidelity_validation_bundle_sha256": _manifest_digest("fidelity_validation"),
        "synthetic_scope": synthetic_scope,
        "synthetic_scope_sha256": hashlib.sha256(canonical_json_bytes(synthetic_scope)).hexdigest(),
        "execution_nonce": _digest("one-confirmatory-attempt"),
        "maximum_confirmatory_attempts": 1,
    }


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _manifest_registry_digest() -> str:
    return hashlib.sha256(canonical_json_bytes(MANIFEST_REGISTRY)).hexdigest()


def _manifest_digest(name: str) -> str:
    return hashlib.sha256(canonical_json_bytes(MANIFEST_REGISTRY["manifests"][name])).hexdigest()


def _write_defender_source(root: Path, source: str) -> None:
    path = root / "src/apar/defense/bad.py"
    path.parent.mkdir(parents=True)
    path.write_text(source, encoding="utf-8")


def _copy_manifest_inputs(root: Path) -> None:
    for directory in ("src/apar/defense", "src/apar/features"):
        shutil.copytree(ROOT / directory, root / directory)
    for relative in (
        "config/defense/competition-v2-manifests.json",
        "config/defense/feature-catalog.json",
        "fixtures/defense/v1/defender-bundle.json",
        "fixtures/defense/v1/split-manifest.json",
        "fixtures/defense/v1/thresholds.json",
        "fixtures/defense/v1/calibration.json",
    ):
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, target)
