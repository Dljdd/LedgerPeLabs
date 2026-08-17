"""Immutable local content-addressed artifact storage."""

from __future__ import annotations

import ctypes
import errno
import hashlib
import json
import os
import shutil
import stat
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import cast

from pydantic import BaseModel

_DIGEST_LENGTH = 64
_MANIFEST_FILENAME = "manifest.json"
_PAYLOAD_FILENAME = "payload"
_RENAME_EXCL = 0x00000004
_RENAME_NOREPLACE = 0x00000001


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
        existing = self._load_if_present(digest)
        if existing is not None:
            return self._reuse_verified_artifact(digest, payload, existing)

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

            try:
                self._publish_no_replace(temporary_dir.name, digest)
            except FileExistsError:
                existing = self._load_if_present(digest)
                if existing is None:
                    raise ValueError(f"concurrent artifact disappeared: {digest}") from None
                return self._reuse_verified_artifact(digest, payload, existing)
            self._fsync_directory(self._root)
            return ref
        finally:
            if temporary_dir.exists():
                shutil.rmtree(temporary_dir)

    def put_json(self, payload: BaseModel | dict[str, object]) -> ArtifactRef:
        """Canonicalize a validated model or JSON object before immutable storage."""
        json_compatible: object
        if isinstance(payload, BaseModel):
            fields = payload.model_dump(mode="python", round_trip=True, warnings=False)
            validated = type(payload).model_validate(fields)
            json_compatible = validated.model_dump(mode="json")
        elif isinstance(payload, dict):
            json_compatible = payload
        else:
            raise TypeError("JSON artifact payload must be a Pydantic model or dict")
        return self.put_bytes(self._canonical_json(json_compatible), "application/json")

    def read(self, ref: ArtifactRef) -> bytes:
        """Return a verified payload, rejecting forged or malformed references."""
        self._validate_ref_shape(ref)
        try:
            stored_ref, payload = self._load_verified(ref.sha256)
        except FileNotFoundError as error:
            raise ValueError(f"artifact does not exist: {ref.sha256}") from error
        if stored_ref != ref:
            raise ValueError("artifact reference does not match stored manifest")
        return payload

    def resolve(self, digest: str) -> ArtifactRef:
        """Resolve one content address only after verifying its payload and manifest."""
        try:
            ref, _ = self._load_verified(digest)
        except FileNotFoundError as error:
            raise ValueError(f"artifact does not exist: {digest}") from error
        return ref

    def _load_if_present(self, digest: str) -> tuple[ArtifactRef, bytes] | None:
        try:
            return self._load_verified(digest)
        except FileNotFoundError:
            return None

    @staticmethod
    def _reuse_verified_artifact(
        digest: str, payload: bytes, existing: tuple[ArtifactRef, bytes]
    ) -> ArtifactRef:
        existing_ref, existing_payload = existing
        if existing_payload != payload:
            raise ValueError(f"artifact payload does not match digest directory: {digest}")
        return existing_ref

    def _load_verified(self, digest: str) -> tuple[ArtifactRef, bytes]:
        self._validate_digest(digest)
        root_fd = self._open_root_directory()
        try:
            artifact_fd = self._open_directory_at(root_fd, digest, digest)
            try:
                expected_entries = {_PAYLOAD_FILENAME, _MANIFEST_FILENAME}
                entries = set(os.listdir(artifact_fd))
                if entries != expected_entries:
                    raise ValueError(f"artifact directory has invalid contents: {digest}")
                payload = self._read_regular_file_at(artifact_fd, _PAYLOAD_FILENAME, digest)
                manifest_bytes = self._read_regular_file_at(
                    artifact_fd, _MANIFEST_FILENAME, digest
                )
            finally:
                os.close(artifact_fd)
        finally:
            os.close(root_fd)

        if hashlib.sha256(payload).hexdigest() != digest:
            raise ValueError(f"artifact payload digest mismatch: {digest}")
        try:
            loaded_manifest = json.loads(manifest_bytes.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError(f"artifact manifest is unreadable: {digest}") from error
        if not isinstance(loaded_manifest, dict):
            raise ValueError(f"artifact manifest must be an object: {digest}")
        manifest = cast(dict[str, object], loaded_manifest)
        ref = self._ref_from_manifest(manifest, digest, len(payload))
        if manifest_bytes != self._manifest_bytes(ref):
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

    def _publish_no_replace(self, source_name: str, destination_name: str) -> None:
        """Atomically rename a completed temporary directory only when destination is absent."""
        root_fd = self._open_root_directory()
        try:
            self._rename_directory_no_replace(root_fd, source_name, destination_name)
        finally:
            os.close(root_fd)

    @staticmethod
    def _rename_directory_no_replace(root_fd: int, source_name: str, destination_name: str) -> None:
        """Dispatch to a platform's native exclusive directory rename, never a fallback."""
        if sys.platform == "darwin":
            ArtifactStore._darwin_rename_no_replace(root_fd, source_name, destination_name)
            return
        if sys.platform == "linux":
            ArtifactStore._linux_rename_no_replace(root_fd, source_name, destination_name)
            return
        raise RuntimeError(
            "artifact publication has no safe implementation on unsupported platform"
        )

    @staticmethod
    def _darwin_rename_no_replace(root_fd: int, source_name: str, destination_name: str) -> None:
        """Call Darwin renameatx_np with RENAME_EXCL."""
        try:
            renameatx_np = ctypes.CDLL(None, use_errno=True).renameatx_np
        except AttributeError as error:
            raise RuntimeError(
                "artifact publication requires Darwin renameatx_np support"
            ) from error
        renameatx_np.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        renameatx_np.restype = ctypes.c_int
        result = renameatx_np(
            root_fd,
            os.fsencode(source_name),
            root_fd,
            os.fsencode(destination_name),
            _RENAME_EXCL,
        )
        if result == 0:
            return
        error_number = ctypes.get_errno()
        if error_number == errno.EEXIST:
            raise FileExistsError(error_number, os.strerror(error_number), destination_name)
        raise OSError(error_number, os.strerror(error_number), destination_name)

    @staticmethod
    def _linux_rename_no_replace(root_fd: int, source_name: str, destination_name: str) -> None:
        """Call Linux libc renameat2 with RENAME_NOREPLACE."""
        try:
            renameat2 = ctypes.CDLL(None, use_errno=True).renameat2
        except AttributeError as error:
            raise RuntimeError(
                "artifact publication requires Linux libc renameat2 support"
            ) from error
        renameat2.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        renameat2.restype = ctypes.c_int
        result = renameat2(
            root_fd,
            os.fsencode(source_name),
            root_fd,
            os.fsencode(destination_name),
            _RENAME_NOREPLACE,
        )
        if result == 0:
            return
        error_number = ctypes.get_errno()
        if error_number == errno.EEXIST:
            raise FileExistsError(error_number, os.strerror(error_number), destination_name)
        unsupported_errors = {errno.ENOSYS, errno.EINVAL, errno.EOPNOTSUPP, errno.ENOTSUP}
        if error_number in unsupported_errors:
            raise RuntimeError(
                "Linux filesystem does not support renameat2 RENAME_NOREPLACE"
            ) from OSError(error_number, os.strerror(error_number), destination_name)
        raise OSError(error_number, os.strerror(error_number), destination_name)

    def _open_root_directory(self) -> int:
        return self._open_directory_at(None, self._root, "root")

    @staticmethod
    def _open_directory_at(parent_fd: int | None, name: str | Path, label: str) -> int:
        flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
        try:
            if parent_fd is None:
                descriptor = os.open(name, flags)
            else:
                descriptor = os.open(name, flags, dir_fd=parent_fd)
        except FileNotFoundError:
            raise
        except OSError as error:
            raise ValueError(f"artifact directory is invalid: {label}") from error
        try:
            if not stat.S_ISDIR(os.fstat(descriptor).st_mode):
                raise ValueError(f"artifact directory is invalid: {label}")
            return descriptor
        except BaseException:
            os.close(descriptor)
            raise

    @staticmethod
    def _read_regular_file_at(directory_fd: int, name: str, digest: str) -> bytes:
        try:
            descriptor = os.open(name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=directory_fd)
        except OSError as error:
            raise ValueError(f"artifact file is invalid: {digest}/{name}") from error
        try:
            if not stat.S_ISREG(os.fstat(descriptor).st_mode):
                raise ValueError(f"artifact file is invalid: {digest}/{name}")
            chunks: list[bytes] = []
            while chunk := os.read(descriptor, 1024 * 1024):
                chunks.append(chunk)
            return b"".join(chunks)
        finally:
            os.close(descriptor)

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
