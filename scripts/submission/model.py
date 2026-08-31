"""Shared immutable release-tooling types and canonical encodings."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Any


class ReleaseError(RuntimeError):
    """A fail-closed submission policy or verification failure."""


@dataclass(frozen=True, slots=True)
class PolicyEntry:
    source: str
    archive: str
    required: bool


@dataclass(frozen=True, slots=True)
class ReleasePolicy:
    archive_root: str
    allowed_extensions: frozenset[str]
    entries: tuple[PolicyEntry, ...]
    extensionless_paths: frozenset[str]
    max_file_bytes: int
    max_total_bytes: int
    release: dict[str, Any]
    scan_allowed_emails: frozenset[str]
    scan_exemptions: frozenset[tuple[str, str]]
    web_entries: tuple[PolicyEntry, ...]
    web_status: str


@dataclass(frozen=True, slots=True)
class BuildResult:
    archive_path: str
    archive_sha256: str
    deterministic_core_sha256: str
    source_commit: str
    source_tree: str


@dataclass(frozen=True, slots=True)
class ScanReport:
    exemption_count: int
    files_scanned: int
    total_bytes: int


def canonical_json(document: object) -> bytes:
    """Encode a JSON document without platform- or whitespace-dependent bytes."""
    return json.dumps(
        document,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def require_safe_relative_path(raw_path: object, *, label: str) -> str:
    """Return a normalized POSIX path or reject absolute/traversing spellings."""
    if not isinstance(raw_path, str) or not raw_path or "\\" in raw_path:
        raise ReleaseError(f"{label} is an unsafe relative path")
    path = PurePosixPath(raw_path)
    if (
        path.is_absolute()
        or ".." in path.parts
        or "." in path.parts
        or path.as_posix() != raw_path
        or raw_path.startswith("/")
    ):
        raise ReleaseError(f"{label} is an unsafe relative path: {raw_path!r}")
    return raw_path
