"""Crash-safe one-attempt receipt contracts for locked Sentinel v5."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest


def _receipt() -> object:
    from apar.evaluation.v5_evidence_storage import (
        build_v5_locked_attempt_receipt,
    )

    return build_v5_locked_attempt_receipt(
        run_binding_sha256="1" * 64,
        preregistration_commit="2" * 40,
        preregistration_sha256="3" * 64,
        source_commit="4" * 40,
        source_tree_oid="5" * 40,
        approved_safe_deterministic_core_sha256="6" * 64,
        approved_safe_observational_environment_sha256="7" * 64,
        authorization_sha256="8" * 64,
        exact_command="locked-test-command",
        started_at_utc="2026-08-24T00:00:00Z",
    )


def _canonical(document: object) -> bytes:
    return json.dumps(
        document,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode()


def test_attempt_receipt_is_exclusive_canonical_and_self_addressed(
    tmp_path: Path,
) -> None:
    from apar.evaluation.v5_evidence_storage import (
        publish_v5_locked_attempt_receipt,
        read_v5_locked_attempt_receipt,
    )

    target = tmp_path / "attempt.json"
    receipt = _receipt()
    published = publish_v5_locked_attempt_receipt(
        target=target, receipt=receipt
    )
    assert target.read_bytes() == _canonical(published.model_dump(mode="json"))
    assert target.stat().st_nlink == 1
    assert read_v5_locked_attempt_receipt(
        target=target,
        expected_run_binding_sha256="1" * 64,
    ) == published
    with pytest.raises(FileExistsError, match="attempt receipt already exists"):
        publish_v5_locked_attempt_receipt(target=target, receipt=receipt)


@pytest.mark.parametrize(
    "existing_kind", ["file", "directory", "symlink", "hardlink"]
)
def test_attempt_receipt_publication_rejects_every_existing_target(
    tmp_path: Path, existing_kind: str
) -> None:
    from apar.evaluation.v5_evidence_storage import (
        publish_v5_locked_attempt_receipt,
    )

    target = tmp_path / "attempt.json"
    if existing_kind == "file":
        target.write_text("preserve")
    elif existing_kind == "directory":
        target.mkdir()
    elif existing_kind == "symlink":
        source = tmp_path / "source"
        source.write_text("preserve")
        target.symlink_to(source)
    else:
        source = tmp_path / "source"
        source.write_text("preserve")
        os.link(source, target)
    with pytest.raises(FileExistsError, match="attempt receipt already exists"):
        publish_v5_locked_attempt_receipt(target=target, receipt=_receipt())


@pytest.mark.parametrize(
    "mutation",
    ["missing", "malformed", "symlink", "hardlink", "tampered", "rebound"],
)
def test_attempt_receipt_reader_rejects_topology_and_content_mutations(
    tmp_path: Path, mutation: str
) -> None:
    from apar.evaluation.v5_evidence_storage import (
        publish_v5_locked_attempt_receipt,
        read_v5_locked_attempt_receipt,
    )

    target = tmp_path / "attempt.json"
    if mutation != "missing":
        publish_v5_locked_attempt_receipt(target=target, receipt=_receipt())
    if mutation == "malformed":
        target.write_bytes(b"not-json")
    elif mutation == "symlink":
        raw = target.read_bytes()
        target.unlink()
        source = tmp_path / "source"
        source.write_bytes(raw)
        target.symlink_to(source)
    elif mutation == "hardlink":
        alias = tmp_path / "alias"
        os.link(target, alias)
    elif mutation in {"tampered", "rebound"}:
        document = json.loads(target.read_bytes())
        document["run_binding_sha256"] = "9" * 64
        if mutation == "rebound":
            document["receipt_sha256"] = hashlib.sha256(
                _canonical(
                    {
                        key: value
                        for key, value in document.items()
                        if key != "receipt_sha256"
                    }
                )
            ).hexdigest()
        target.write_bytes(_canonical(document))
    with pytest.raises((FileNotFoundError, ValueError)):
        read_v5_locked_attempt_receipt(
            target=target,
            expected_run_binding_sha256="1" * 64,
        )


def test_attempt_receipt_directory_fsync_failure_consumes_attempt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from apar.evaluation import v5_evidence_storage as storage

    target = tmp_path / "attempt.json"

    def fail_directory_fsync(_path: Path) -> None:
        raise OSError("simulated directory fsync failure")

    monkeypatch.setattr(storage, "_fsync_directory", fail_directory_fsync)
    with pytest.raises(OSError, match="directory fsync failure"):
        storage.publish_v5_locked_attempt_receipt(
            target=target, receipt=_receipt()
        )
    assert target.exists()
    with pytest.raises(FileExistsError, match="attempt receipt already exists"):
        storage.publish_v5_locked_attempt_receipt(
            target=target, receipt=_receipt()
        )


@pytest.mark.parametrize(
    "mutation",
    [
        "run_binding",
        "preregistration_commit",
        "preregistration_sha256",
        "source_commit",
        "source_tree_oid",
        "safe_core",
        "environment",
        "authorization",
        "exact_command",
        "payload_digest",
    ],
)
def test_independent_verifier_rejects_rebound_attempt_receipt(
    tmp_path: Path, mutation: str
) -> None:
    from apar import v5_independent_verifier as verifier
    from apar.evaluation.v5_evidence_storage import (
        build_v5_locked_attempt_receipt,
        publish_v5_locked_attempt_receipt,
    )

    command = "locked-test-command"
    binding = {
        "run_binding_sha256": "1" * 64,
        "preregistration_commit": "2" * 40,
        "preregistration_sha256": "3" * 64,
        "source_commit": "4" * 40,
        "source_tree_oid": "5" * 40,
        "preregistration_path": "config/preregistration.json",
    }
    authorization = hashlib.sha256(
        _canonical(
            {
                "authorization": "execute-exactly-once-locked-development",
                "preregistration_commit": binding["preregistration_commit"],
                "run_binding_sha256": binding["run_binding_sha256"],
                "exact_command": command,
            }
        )
    ).hexdigest()
    receipt = build_v5_locked_attempt_receipt(
        run_binding_sha256=str(binding["run_binding_sha256"]),
        preregistration_commit=str(binding["preregistration_commit"]),
        preregistration_sha256=str(binding["preregistration_sha256"]),
        source_commit=str(binding["source_commit"]),
        source_tree_oid=str(binding["source_tree_oid"]),
        approved_safe_deterministic_core_sha256="6" * 64,
        approved_safe_observational_environment_sha256="7" * 64,
        authorization_sha256=authorization,
        exact_command=command,
        started_at_utc="2026-08-24T00:00:00Z",
    )
    attempt_relative = (
        "docs/experiments/defense-v5-locked-development-attempt.json"
    )
    attempt_path = tmp_path / attempt_relative
    publish_v5_locked_attempt_receipt(
        target=attempt_path, receipt=receipt
    )
    preregistration_path = tmp_path / str(binding["preregistration_path"])
    preregistration_path.parent.mkdir(parents=True)
    preregistration_path.write_bytes(
        _canonical(
            {
                "exact_command": command,
                "safe_validation": {
                    "approved_deterministic_core_sha256": "6" * 64,
                    "approved_observational_environment_sha256": "7" * 64,
                },
            }
        )
    )
    protocol = {
        "locked_artifact_storage": {
            "attempt_receipt_path": attempt_relative
        }
    }
    payload = {"attempt_receipt_sha256": receipt.receipt_sha256}
    assert verifier._verify_locked_attempt_receipt(
        payload=payload,
        protocol=protocol,
        binding=binding,
        root=tmp_path,
    )["receipt_sha256"] == receipt.receipt_sha256

    if mutation == "payload_digest":
        payload["attempt_receipt_sha256"] = "9" * 64
    else:
        document = json.loads(attempt_path.read_bytes())
        field = {
            "run_binding": "run_binding_sha256",
            "preregistration_commit": "preregistration_commit",
            "preregistration_sha256": "preregistration_sha256",
            "source_commit": "source_commit",
            "source_tree_oid": "source_tree_oid",
            "safe_core": "approved_safe_deterministic_core_sha256",
            "environment": (
                "approved_safe_observational_environment_sha256"
            ),
            "authorization": "authorization_sha256",
            "exact_command": "exact_command",
        }[mutation]
        document[field] = (
            "rebound-command"
            if mutation == "exact_command"
            else "9" * 40
            if mutation in {
                "preregistration_commit",
                "source_commit",
                "source_tree_oid",
            }
            else "9" * 64
        )
        document["receipt_sha256"] = hashlib.sha256(
            _canonical(
                {
                    key: value
                    for key, value in document.items()
                    if key != "receipt_sha256"
                }
            )
        ).hexdigest()
        attempt_path.write_bytes(_canonical(document))
        payload["attempt_receipt_sha256"] = document["receipt_sha256"]
    with pytest.raises(
        verifier.IndependentVerificationError,
        match="attempt receipt.*binding",
    ):
        verifier._verify_locked_attempt_receipt(
            payload=payload,
            protocol=protocol,
            binding=binding,
            root=tmp_path,
        )
