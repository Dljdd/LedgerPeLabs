"""Tests for immutable, content-addressed local artifacts."""

import hashlib
from dataclasses import replace
from datetime import date
from decimal import Decimal
from enum import Enum
from pathlib import Path

import pytest
from pydantic import BaseModel

from apar.storage.artifacts import ArtifactRef, ArtifactStore


class Stage(Enum):
    REVIEWED = "reviewed"


class CanonicalPayload(BaseModel):
    amount: Decimal
    published_on: date
    stage: Stage


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

    def fail_publish(source: Path, destination: Path) -> None:
        raise OSError("simulated interrupted rename")

    monkeypatch.setattr("apar.storage.artifacts.os.rename", fail_publish)

    with pytest.raises(OSError, match="simulated interrupted rename"):
        store.put_bytes(b"interrupted", "application/octet-stream")

    assert list(tmp_path.iterdir()) == []
