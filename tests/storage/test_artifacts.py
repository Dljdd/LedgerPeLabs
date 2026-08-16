"""Tests for immutable, content-addressed local artifacts."""

import ctypes
import errno
import hashlib
import json
import os
from dataclasses import replace
from datetime import date
from decimal import Decimal
from enum import Enum
from pathlib import Path

import pytest
from pydantic import BaseModel, ValidationError

import apar.storage.artifacts as artifacts
from apar.storage.artifacts import ArtifactRef, ArtifactStore


class Stage(Enum):
    REVIEWED = "reviewed"


class CanonicalPayload(BaseModel):
    amount: Decimal
    published_on: date
    stage: Stage


class NestedCanonicalPayload(BaseModel):
    payload: CanonicalPayload


def _write_complete_artifact(root: Path, payload: bytes, media_type: str) -> ArtifactRef:
    digest = hashlib.sha256(payload).hexdigest()
    ref = ArtifactRef(digest, media_type, len(payload), f"{digest}/payload")
    artifact_dir = root / digest
    artifact_dir.mkdir()
    (artifact_dir / "payload").write_bytes(payload)
    (artifact_dir / "manifest.json").write_text(
        json.dumps(
            {
                "media_type": ref.media_type,
                "relative_path": ref.relative_path,
                "sha256": ref.sha256,
                "size_bytes": ref.size_bytes,
            },
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return ref


def test_json_hash_is_key_order_independent(tmp_path: Path) -> None:
    """Catches non-canonical JSON changing a content address."""
    store = ArtifactStore(tmp_path)

    first = store.put_json({"b": 2, "a": 1})
    second = store.put_json({"a": 1, "b": 2})

    assert first.sha256 == second.sha256
    assert first.relative_path == second.relative_path


def test_put_json_canonicalizes_pydantic_json_types(tmp_path: Path) -> None:
    """Catches Pydantic Decimal, date, or enum values losing JSON compatibility."""
    store = ArtifactStore(tmp_path)
    payload = CanonicalPayload(
        amount=Decimal("12.50"), published_on=date(2026, 8, 16), stage=Stage.REVIEWED
    )

    ref = store.put_json(payload)

    expected = b'{"amount":"12.50","published_on":"2026-08-16","stage":"reviewed"}'
    assert ref.sha256 == hashlib.sha256(expected).hexdigest()
    assert store.read(ref) == expected


def test_put_json_rejects_a_copied_invalid_model(tmp_path: Path) -> None:
    """Catches model_copy bypass values being frozen as trusted JSON artifacts."""
    store = ArtifactStore(tmp_path)
    valid = CanonicalPayload(
        amount=Decimal("12.50"), published_on=date(2026, 8, 16), stage=Stage.REVIEWED
    )
    invalid = valid.model_copy(update={"amount": "not-a-decimal"})

    with pytest.raises(ValidationError, match="amount"):
        store.put_json(invalid)


def test_put_json_rejects_a_copied_invalid_nested_model(tmp_path: Path) -> None:
    """Catches recursive Pydantic validation being skipped before canonical storage."""
    valid = CanonicalPayload(
        amount=Decimal("12.50"), published_on=date(2026, 8, 16), stage=Stage.REVIEWED
    )
    invalid_child = valid.model_copy(update={"published_on": "not-a-date"})
    parent = NestedCanonicalPayload(payload=invalid_child)

    with pytest.raises(ValidationError, match="published_on"):
        ArtifactStore(tmp_path).put_json(parent)


def test_put_json_valid_model_matches_its_json_payload(tmp_path: Path) -> None:
    """Catches model revalidation changing valid canonical artifact bytes."""
    store = ArtifactStore(tmp_path)
    model = CanonicalPayload(
        amount=Decimal("12.50"), published_on=date(2026, 8, 16), stage=Stage.REVIEWED
    )

    model_ref = store.put_json(model)
    dict_ref = store.put_json(model.model_dump(mode="json"))

    assert model_ref == dict_ref
    assert store.read(model_ref) == store.read(dict_ref)


def test_existing_digest_cannot_be_overwritten(tmp_path: Path) -> None:
    """Catches duplicate writes creating mutable artifact contents."""
    store = ArtifactStore(tmp_path)
    ref = store.put_bytes(b"evidence", "application/octet-stream")

    duplicate = store.put_bytes(b"evidence", "application/octet-stream")

    assert duplicate == ref
    assert store.read(ref) == b"evidence"
    assert len(list((tmp_path / ref.sha256).iterdir())) == 2


def test_existing_digest_with_corrupt_manifest_is_rejected(tmp_path: Path) -> None:
    """Catches accepting a digest directory whose metadata was modified after creation."""
    store = ArtifactStore(tmp_path)
    ref = store.put_bytes(b"evidence", "application/octet-stream")
    (tmp_path / ref.sha256 / "manifest.json").write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match="manifest"):
        store.put_bytes(b"evidence", "application/octet-stream")


def test_read_rejects_forged_or_traversing_references(tmp_path: Path) -> None:
    """Catches callers escaping the configured artifact root with a constructed ref."""
    store = ArtifactStore(tmp_path)
    ref = store.put_bytes(b"evidence", "application/octet-stream")

    forged = replace(ref, relative_path="../../outside")

    with pytest.raises(ValueError, match="relative path"):
        store.read(forged)
    with pytest.raises(ValueError, match="digest"):
        store.read(
            ArtifactRef(
                sha256="../" + ref.sha256,
                media_type=ref.media_type,
                size_bytes=ref.size_bytes,
                relative_path=ref.relative_path,
            )
        )


def test_interrupted_write_removes_temporary_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Catches failed atomic publication leaving a reusable partial artifact behind."""
    store = ArtifactStore(tmp_path)

    def fail_publish(source_name: str, destination_name: str) -> None:
        raise OSError("simulated interrupted rename")

    monkeypatch.setattr(store, "_publish_no_replace", fail_publish)

    with pytest.raises(OSError, match="simulated interrupted rename"):
        store.put_bytes(b"interrupted", "application/octet-stream")

    assert list(tmp_path.iterdir()) == []


def test_empty_destination_appearing_at_publication_is_never_replaced(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Catches a final rename overwriting an empty competing digest directory."""
    store = ArtifactStore(tmp_path)
    payload = b"racing artifact"
    digest = hashlib.sha256(payload).hexdigest()
    publish = store._publish_no_replace

    def publish_after_empty_destination(source_name: str, destination_name: str) -> None:
        (tmp_path / digest).mkdir()
        publish(source_name, destination_name)

    monkeypatch.setattr(store, "_publish_no_replace", publish_after_empty_destination)

    with pytest.raises(ValueError, match="invalid contents"):
        store.put_bytes(payload, "application/octet-stream")

    assert (tmp_path / digest).is_dir()
    assert list((tmp_path / digest).iterdir()) == []


def test_concurrent_winner_is_verified_and_reused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Catches a losing publisher failing instead of converging on a valid winner."""
    store = ArtifactStore(tmp_path)
    payload = b"racing artifact"
    media_type = "application/octet-stream"
    publish = store._publish_no_replace
    winner: ArtifactRef | None = None

    def publish_after_winner(source_name: str, destination_name: str) -> None:
        nonlocal winner
        winner = _write_complete_artifact(tmp_path, payload, media_type)
        publish(source_name, destination_name)

    monkeypatch.setattr(store, "_publish_no_replace", publish_after_winner)

    ref = store.put_bytes(payload, media_type)

    assert winner is not None
    assert ref == winner
    assert store.read(ref) == payload


def test_read_rejects_digest_directory_symlink_even_when_target_is_inside_root(
    tmp_path: Path,
) -> None:
    """Catches accepting a digest-directory symlink merely because its target is in-root."""
    store = ArtifactStore(tmp_path)
    ref = store.put_bytes(b"evidence", "application/octet-stream")
    artifact_dir = tmp_path / ref.sha256
    safe_target = tmp_path / "safe-target"
    artifact_dir.rename(safe_target)
    artifact_dir.symlink_to(safe_target, target_is_directory=True)

    with pytest.raises(ValueError, match="directory"):
        store.read(ref)


def test_read_rejects_payload_swap_before_descriptor_open(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Catches a payload pathname being swapped for a symlink before it is opened."""
    store = ArtifactStore(tmp_path)
    ref = store.put_bytes(b"evidence", "application/octet-stream")
    payload_path = tmp_path / ref.sha256 / "payload"
    external_payload = tmp_path / "external-payload"
    external_payload.write_bytes(b"external")
    original_open = os.open
    swapped = False

    def swap_before_open(
        path: str | os.PathLike[str], *args: object, **kwargs: object
    ) -> int:
        nonlocal swapped
        if path == "payload" and not swapped:
            payload_path.unlink()
            payload_path.symlink_to(external_payload)
            swapped = True
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr("apar.storage.artifacts.os.open", swap_before_open)

    with pytest.raises(ValueError, match="file"):
        store.read(ref)

    assert swapped


def test_exclusive_rename_selects_darwin_implementation(monkeypatch: pytest.MonkeyPatch) -> None:
    """Catches Darwin dispatch accidentally using an overwrite-capable fallback."""
    calls: list[tuple[int, str, str]] = []

    def rename_on_darwin(root_fd: int, source_name: str, destination_name: str) -> None:
        calls.append((root_fd, source_name, destination_name))

    monkeypatch.setattr(artifacts.sys, "platform", "darwin")
    monkeypatch.setattr(
        ArtifactStore, "_darwin_rename_no_replace", staticmethod(rename_on_darwin)
    )

    ArtifactStore._rename_directory_no_replace(7, "source", "destination")

    assert calls == [(7, "source", "destination")]


def test_exclusive_rename_selects_linux_implementation(monkeypatch: pytest.MonkeyPatch) -> None:
    """Catches Linux publication being rejected before calling renameat2."""
    calls: list[tuple[int, str, str]] = []

    def rename_on_linux(root_fd: int, source_name: str, destination_name: str) -> None:
        calls.append((root_fd, source_name, destination_name))

    monkeypatch.setattr(artifacts.sys, "platform", "linux")
    monkeypatch.setattr(
        ArtifactStore, "_linux_rename_no_replace", staticmethod(rename_on_linux)
    )

    ArtifactStore._rename_directory_no_replace(7, "source", "destination")

    assert calls == [(7, "source", "destination")]


def test_linux_exclusive_rename_translates_eexist(monkeypatch: pytest.MonkeyPatch) -> None:
    """Catches Linux EEXIST bypassing the concurrent-winner convergence path."""

    class RenameAt2:
        argtypes: object
        restype: object

        def __call__(self, *args: object) -> int:
            ctypes.set_errno(errno.EEXIST)
            return -1

    class LibC:
        renameat2 = RenameAt2()

    monkeypatch.setattr(artifacts.ctypes, "CDLL", lambda *args, **kwargs: LibC())

    with pytest.raises(FileExistsError):
        ArtifactStore._linux_rename_no_replace(7, "source", "destination")


def test_exclusive_rename_rejects_unsupported_platform(monkeypatch: pytest.MonkeyPatch) -> None:
    """Catches an unsupported platform silently falling back to normal rename."""
    monkeypatch.setattr(artifacts.sys, "platform", "freebsd")

    with pytest.raises(RuntimeError, match="unsupported platform"):
        ArtifactStore._rename_directory_no_replace(7, "source", "destination")
