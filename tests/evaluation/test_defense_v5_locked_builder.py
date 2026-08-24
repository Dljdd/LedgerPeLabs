"""Authorization and wiring contracts for the one-time locked builder."""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

from apar.evaluation.v5_evidence_protocol import load_v5_evidence_protocol
from apar.evaluation.v5_evidence_storage import V5LockedAttemptReceipt
from apar.evaluation.v5_protocol import load_v5_development_protocol
from apar.evaluation.v5_run_mode import (
    V5LockedEvidenceRunBinding,
    V5RunMode,
    build_v5_run_support_plan,
)

ROOT = Path(__file__).resolve().parents[2]


def _binding() -> V5LockedEvidenceRunBinding:
    evidence = load_v5_evidence_protocol(
        ROOT / "config/defense/defense-v5-evidence.json", root=ROOT
    )
    development = load_v5_development_protocol(
        ROOT / "config/defense/defense-v5-development.json"
    )
    values: dict[str, object] = {
        "schema_version": "apar-sentinel-v5-locked-run-binding/1",
        "mode": "locked_development",
        "profile": "production",
        "development_test_seed": 2404,
        "source_commit": "1" * 40,
        "source_tree_oid": "2" * 40,
        "preregistration_commit": "3" * 40,
        "preregistration_path": (
            "config/defense/defense-v5-locked-development-preregistration.json"
        ),
        "preregistration_sha256": "4" * 64,
        "base_protocol_sha256": evidence.base_protocol_sha256,
        "arm_protocol_sha256": evidence.arm_protocol_sha256,
        "evidence_protocol_sha256": evidence.evidence_protocol_sha256,
        "implementation_sha256": evidence.implementation_sha256,
        "catalog_sha256": "5" * 64,
        "support_plan": build_v5_run_support_plan(
            mode=V5RunMode.LOCKED_DEVELOPMENT,
            evidence_protocol=evidence,
            development_protocol=development,
        ).model_dump(mode="json"),
        "candidate_manifest_path": (
            evidence.locked_artifact_storage.candidate_manifest_path
        ),
        "storage_schema_version": evidence.locked_artifact_storage.schema_version,
        "payload_schema_version": (
            "apar-sentinel-v5-locked-development-payload/2"
        ),
    }
    values["run_binding_sha256"] = V5LockedEvidenceRunBinding.compute_digest(values)
    return V5LockedEvidenceRunBinding.model_validate(values)


class _SyntheticAuthority:
    """An isolated authority that cannot invoke the real execution engine."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, object]] = []
        self.binding = _binding()

    def preflight(
        self, *, root: Path, safe_evidence: Path, approved_commit: str
    ) -> object:
        from scripts.run_defense_v5_locked_development import (
            _LockedPreexecutionAuthorization,
        )

        self.calls.append(("preflight", (root, safe_evidence, approved_commit)))
        return _LockedPreexecutionAuthorization(
            binding=self.binding,
            approved_safe_deterministic_core_sha256="7" * 64,
            approved_safe_observational_environment_sha256="8" * 64,
            exact_command="locked-test-command",
        )

    def build_payload(
        self,
        *,
        root: Path,
        binding: V5LockedEvidenceRunBinding,
        attempt_receipt: V5LockedAttemptReceipt,
    ) -> bytes:
        storage = load_v5_evidence_protocol(
            ROOT / "config/defense/defense-v5-evidence.json", root=ROOT
        ).locked_artifact_storage
        assert (root / storage.attempt_receipt_path).is_file()
        assert attempt_receipt.run_binding_sha256 == binding.run_binding_sha256
        self.calls.append(("build_payload", binding.mode))
        return json.dumps(
            {
                "schema_version": "test-only-opaque-payload/1",
                "run_binding_sha256": binding.run_binding_sha256,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()

    def verify_payload(self, *, root: Path, payload: bytes) -> dict[str, object]:
        self.calls.append(("verify_payload", hashlib.sha256(payload).hexdigest()))
        return {"verified": True, "status": "test_only"}

    def storage(self, *, root: Path) -> object:
        self.calls.append(("storage", root))
        return load_v5_evidence_protocol(
            ROOT / "config/defense/defense-v5-evidence.json", root=ROOT
        ).locked_artifact_storage

    def verify_published(
        self, *, root: Path, target: Path, storage: object
    ) -> dict[str, object]:
        self.calls.append(("verify_published", target))
        return {"verified": True, "status": "test_only_published"}

    def verify_summary(
        self,
        *,
        root: Path,
        summary_target: Path,
        target: Path,
        storage: object,
        published: dict[str, object],
    ) -> dict[str, object]:
        assert summary_target.is_file()
        self.calls.append(("verify_summary", target))
        return {"verified": True, "status": "test_only_summary"}


class _CrashingAuthority(_SyntheticAuthority):
    """A test authority that fails only after the workload boundary is entered."""

    def __init__(self) -> None:
        super().__init__()
        self.build_calls = 0

    def build_payload(
        self,
        *,
        root: Path,
        binding: V5LockedEvidenceRunBinding,
        attempt_receipt: V5LockedAttemptReceipt,
    ) -> bytes:
        self.build_calls += 1
        self.calls.append(("build_payload", binding.mode))
        raise RuntimeError("simulated production crash")


def test_complete_evidence_engine_exists_for_both_closed_modes() -> None:
    """A separate locked runner must not reimplement or omit complete evidence."""
    module = importlib.util.find_spec("scripts.v5_complete_evidence_execution")
    assert module is not None


def test_locked_complete_engine_rejects_calls_without_preflight_capability() -> None:
    """Seed 2404 must not reach population through the shared internal engine alone."""
    from scripts.v5_complete_evidence_execution import execute_v5_complete_evidence

    with pytest.raises(PermissionError, match="verified preflight capability"):
        execute_v5_complete_evidence(root=ROOT, mode=V5RunMode.LOCKED_DEVELOPMENT)


def test_locked_builder_cli_has_no_seed_profile_output_or_test_injection() -> None:
    """The production CLI must expose only frozen inputs and explicit authorization."""
    path = ROOT / "scripts/run_defense_v5_locked_development.py"
    tree = ast.parse(path.read_text())
    arguments = {
        node.args[0].value
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "add_argument"
        and node.args
        and isinstance(node.args[0], ast.Constant)
        and isinstance(node.args[0].value, str)
    }
    assert arguments == {
        "--root",
        "--safe-evidence",
        "--approved-commit",
        "--authorize-exactly-once",
    }
    assert "--seed" not in path.read_text()
    assert "--profile" not in path.read_text()
    assert "--output" not in path.read_text()
    assert "test_authority" not in path.read_text()


def test_test_only_authority_proves_fixed_locked_publication_wiring(
    tmp_path: Path,
) -> None:
    """The test seam must publish without importing or calling the real workload."""
    from scripts.run_defense_v5_locked_development import (
        _execute_locked_development_once,
    )

    authority = _SyntheticAuthority()
    safe_evidence = tmp_path / "safe.json"
    safe_evidence.write_bytes(b"not-read-by-test-authority")
    manifest = _execute_locked_development_once(
        root=tmp_path,
        safe_evidence=safe_evidence,
        approved_commit="6" * 40,
        authorization_granted=True,
        authority=authority,
    )
    target = tmp_path / authority.binding.candidate_manifest_path
    assert target.is_file()
    assert (
        tmp_path
        / load_v5_evidence_protocol(
            ROOT / "config/defense/defense-v5-evidence.json", root=ROOT
        ).locked_artifact_storage.attempt_receipt_path
    ).is_file()
    assert (
        tmp_path
        / load_v5_evidence_protocol(
            ROOT / "config/defense/defense-v5-evidence.json", root=ROOT
        ).locked_artifact_storage.judge_summary_path
    ).is_file()
    assert manifest.run_binding_sha256 == authority.binding.run_binding_sha256
    assert [name for name, _value in authority.calls] == [
        "preflight",
        "storage",
        "build_payload",
        "verify_payload",
        "verify_published",
        "verify_summary",
    ]
    assert authority.calls[2] == ("build_payload", V5RunMode.LOCKED_DEVELOPMENT)


def test_locked_builder_requires_explicit_authorization_before_preflight(
    tmp_path: Path,
) -> None:
    """Without the exact opt-in, even the read-only preflight must not run."""
    from scripts.run_defense_v5_locked_development import (
        _execute_locked_development_once,
    )

    authority = _SyntheticAuthority()
    with pytest.raises(PermissionError, match="explicit one-time authorization"):
        _execute_locked_development_once(
            root=tmp_path,
            safe_evidence=tmp_path / "safe.json",
            approved_commit="6" * 40,
            authorization_granted=False,
            authority=authority,
        )
    assert authority.calls == []


def test_locked_builder_refuses_existing_candidate_before_execution(
    tmp_path: Path,
) -> None:
    """An existing manifest must stop the one-time path before building a payload."""
    from scripts.run_defense_v5_locked_development import (
        _execute_locked_development_once,
    )

    authority = _SyntheticAuthority()
    target = tmp_path / authority.binding.candidate_manifest_path
    target.parent.mkdir(parents=True)
    target.write_bytes(b"historical-candidate")
    with pytest.raises(FileExistsError, match="candidate manifest already exists"):
        _execute_locked_development_once(
            root=tmp_path,
            safe_evidence=tmp_path / "safe.json",
            approved_commit="6" * 40,
            authorization_granted=True,
            authority=authority,
        )
    assert [name for name, _value in authority.calls] == [
        "preflight",
        "storage",
    ]
    assert target.read_bytes() == b"historical-candidate"


def test_locked_builder_crash_consumes_attempt_before_workload(
    tmp_path: Path,
) -> None:
    """A crash after entry must leave a marker that blocks every later attempt."""
    from scripts.run_defense_v5_locked_development import (
        _execute_locked_development_once,
    )

    authority = _CrashingAuthority()
    safe_evidence = tmp_path / "safe.json"
    safe_evidence.write_bytes(b"not-read-by-test-authority")
    attempt = (
        tmp_path
        / "docs/experiments/defense-v5-locked-development-attempt.json"
    )
    with pytest.raises(RuntimeError, match="simulated production crash"):
        _execute_locked_development_once(
            root=tmp_path,
            safe_evidence=safe_evidence,
            approved_commit="6" * 40,
            authorization_granted=True,
            authority=authority,
        )
    first_marker_visible = attempt.is_file()
    with pytest.raises((FileExistsError, RuntimeError)):
        _execute_locked_development_once(
            root=tmp_path,
            safe_evidence=safe_evidence,
            approved_commit="6" * 40,
            authorization_granted=True,
            authority=authority,
        )
    assert (first_marker_visible, authority.build_calls) == (True, 1)


def test_locked_builder_does_not_enter_workload_when_directory_fsync_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The execution authority must remain unreachable until directory fsync."""
    from apar.evaluation import v5_evidence_storage as storage_module
    from scripts.run_defense_v5_locked_development import (
        _execute_locked_development_once,
    )

    authority = _SyntheticAuthority()

    def fail_directory_fsync(_path: Path) -> None:
        raise OSError("simulated directory fsync failure")

    monkeypatch.setattr(
        storage_module, "_fsync_directory", fail_directory_fsync
    )
    with pytest.raises(OSError, match="directory fsync failure"):
        _execute_locked_development_once(
            root=tmp_path,
            safe_evidence=tmp_path / "safe.json",
            approved_commit="6" * 40,
            authorization_granted=True,
            authority=authority,
        )
    assert [name for name, _value in authority.calls] == [
        "preflight",
        "storage",
    ]
    configured = load_v5_evidence_protocol(
        ROOT / "config/defense/defense-v5-evidence.json", root=ROOT
    ).locked_artifact_storage
    assert (tmp_path / configured.attempt_receipt_path).is_file()
