"""Immutable local content-addressed artifact storage."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import cast

from pydantic import BaseModel

_DIGEST_LENGTH = 64
_MANIFEST_FILENAME = "manifest.json"
_PAYLOAD_FILENAME = "payload"


@dataclass(frozen=True, slots=True)
class ArtifactRef:
    """A typed reference to one immutable payload beneath an artifact root."""

    sha256: str
    media_type: str
    size_bytes: int
    relative_path: str


class ArtifactStore:
    """Persist payloads once under their SHA-256 digest."""

    def __init__(self, root: Path) -> None:
        self._root = root.resolve()
        self._root.mkdir(parents=True, exist_ok=True)
        if not self._root.is_dir():
            raise ValueError(f"artifact root is not a directory: {self._root}")

    def put_bytes(self, payload: bytes, media_type: str) -> ArtifactRef:
        """Store bytes atomically, returning their immutable content reference."""
        if not isinstance(payload, bytes):
            raise TypeError("artifact payload must be bytes")
        self._validate_media_type(media_type)

        digest = hashlib.sha256(payload).hexdigest()
        artifact_dir = self._root / digest
        if artifact_dir.exists():
            existing_ref, existing_payload = self._load_verified(digest)
            if existing_payload != payload:
                raise ValueError(f"artifact payload does not match digest directory: {digest}")
            return existing_ref

        ref = ArtifactRef(
            sha256=digest,
            media_type=media_type,
            size_bytes=len(payload),
            relative_path=self._relative_payload_path(digest),
        )
        temporary_dir = Path(tempfile.mkdtemp(prefix=f".{digest}.", dir=self._root))
        try:
            self._write_durable_file(temporary_dir / _PAYLOAD_FILENAME, payload)
            self._write_durable_file(temporary_dir / _MANIFEST_FILENAME, self._manifest_bytes(ref))
            self._fsync_directory(temporary_dir)

            # A pre-existing target is always inspected rather than replaced.
            if artifact_dir.exists():
                existing_ref, existing_payload = self._load_verified(digest)
                if existing_payload != payload:
                    raise ValueError(f"artifact payload does not match digest directory: {digest}")
                return existing_ref

            os.rename(temporary_dir, artifact_dir)
            self._fsync_directory(self._root)
            return ref
        finally:
            if temporary_dir.exists():
                shutil.rmtree(temporary_dir)

    def put_json(self, payload: BaseModel | dict[str, object]) -> ArtifactRef:
        """Canonicalize a validated model or JSON object before immutable storage."""
        json_compatible: object
        if isinstance(payload, BaseModel):
            json_compatible = payload.model_dump(mode="json")
        elif isinstance(payload, dict):
            json_compatible = payload
        else:
            raise TypeError("JSON artifact payload must be a Pydantic model or dict")
        return self.put_bytes(self._canonical_json(json_compatible), "application/json")

    def read(self, ref: ArtifactRef) -> bytes:
        """Return a verified payload, rejecting forged or malformed references."""
        self._validate_ref_shape(ref)
        stored_ref, payload = self._load_verified(ref.sha256)
        if stored_ref != ref:
            raise ValueError("artifact reference does not match stored manifest")
        return payload

    def _load_verified(self, digest: str) -> tuple[ArtifactRef, bytes]:
        self._validate_digest(digest)
        artifact_dir = self._root / digest
        self._require_directory_below_root(artifact_dir)
        expected_entries = {_PAYLOAD_FILENAME, _MANIFEST_FILENAME}
        entries = {entry.name for entry in artifact_dir.iterdir()}
        if entries != expected_entries:
            raise ValueError(f"artifact directory has invalid contents: {digest}")

        payload_path = artifact_dir / _PAYLOAD_FILENAME
        manifest_path = artifact_dir / _MANIFEST_FILENAME
        self._require_regular_file_below_root(payload_path)
        self._require_regular_file_below_root(manifest_path)
        payload = payload_path.read_bytes()
        if hashlib.sha256(payload).hexdigest() != digest:
            raise ValueError(f"artifact payload digest mismatch: {digest}")

        try:
            loaded_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError(f"artifact manifest is unreadable: {digest}") from error
        if not isinstance(loaded_manifest, dict):
            raise ValueError(f"artifact manifest must be an object: {digest}")
        manifest = cast(dict[str, object], loaded_manifest)
        ref = self._ref_from_manifest(manifest, digest, len(payload))
        if manifest_path.read_bytes() != self._manifest_bytes(ref):
            raise ValueError(f"artifact manifest is not canonical: {digest}")
        return ref, payload

    def _ref_from_manifest(
        self, manifest: dict[str, object], digest: str, payload_size: int
    ) -> ArtifactRef:
        expected_keys = {"sha256", "media_type", "size_bytes", "relative_path"}
        if set(manifest) != expected_keys:
            raise ValueError(f"artifact manifest has invalid fields: {digest}")
        sha256 = manifest["sha256"]
        media_type = manifest["media_type"]
        size_bytes = manifest["size_bytes"]
        relative_path = manifest["relative_path"]
        if (
            not isinstance(sha256, str)
            or not isinstance(media_type, str)
            or isinstance(size_bytes, bool)
            or not isinstance(size_bytes, int)
            or not isinstance(relative_path, str)
        ):
            raise ValueError(f"artifact manifest has invalid field types: {digest}")
        ref = ArtifactRef(sha256, media_type, size_bytes, relative_path)
        try:
            self._validate_ref_shape(ref)
        except (TypeError, ValueError) as error:
            raise ValueError(f"artifact manifest has invalid reference: {digest}") from error
        if ref.sha256 != digest or ref.size_bytes != payload_size:
            raise ValueError(f"artifact manifest does not match payload: {digest}")
        return ref

    @staticmethod
    def _canonical_json(payload: object) -> bytes:
        return json.dumps(
            payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True
        ).encode("utf-8")

    @staticmethod
    def _manifest_bytes(ref: ArtifactRef) -> bytes:
        return ArtifactStore._canonical_json(asdict(ref))

    @staticmethod
    def _relative_payload_path(digest: str) -> str:
        return f"{digest}/{_PAYLOAD_FILENAME}"

    @staticmethod
    def _validate_media_type(media_type: str) -> None:
        if not isinstance(media_type, str) or not media_type:
            raise ValueError("artifact media type must be a non-empty string")

    def _validate_ref_shape(self, ref: ArtifactRef) -> None:
        if not isinstance(ref, ArtifactRef):
            raise TypeError("artifact reference must be an ArtifactRef")
        self._validate_digest(ref.sha256)
        self._validate_media_type(ref.media_type)
        if (
            isinstance(ref.size_bytes, bool)
            or not isinstance(ref.size_bytes, int)
            or ref.size_bytes < 0
        ):
            raise ValueError("artifact reference has invalid size")
        if ref.relative_path != self._relative_payload_path(ref.sha256):
            raise ValueError("artifact reference has invalid relative path")

    @staticmethod
    def _validate_digest(digest: str) -> None:
        if (
            not isinstance(digest, str)
            or len(digest) != _DIGEST_LENGTH
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ValueError("artifact reference has invalid digest")

    def _require_directory_below_root(self, path: Path) -> None:
        try:
            resolved = path.resolve(strict=True)
        except OSError as error:
            raise ValueError(f"artifact directory is unavailable: {path.name}") from error
        if not resolved.is_relative_to(self._root) or not resolved.is_dir():
            raise ValueError(f"artifact directory is invalid: {path.name}")

    def _require_regular_file_below_root(self, path: Path) -> None:
        try:
            resolved = path.resolve(strict=True)
            mode = path.lstat().st_mode
        except OSError as error:
            raise ValueError(f"artifact file is unavailable: {path.name}") from error
        if not resolved.is_relative_to(self._root) or not stat.S_ISREG(mode):
            raise ValueError(f"artifact file is invalid: {path.name}")

    @staticmethod
    def _write_durable_file(path: Path, content: bytes) -> None:
        with path.open("wb") as file:
            file.write(content)
            file.flush()
            os.fsync(file.fileno())

    @staticmethod
    def _fsync_directory(path: Path) -> None:
        descriptor = os.open(path, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
