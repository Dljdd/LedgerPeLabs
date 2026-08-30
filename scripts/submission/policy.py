"""Strict loader for the explicit submission allowlist."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

from scripts.submission.model import (
    PolicyEntry,
    ReleaseError,
    ReleasePolicy,
    require_safe_relative_path,
)

_SCHEMA = "apar-submission-policy/1"
_WEB_STATUSES = frozenset({"pending", "ready"})


def _object(value: object, *, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ReleaseError(f"{label} must be a JSON object")
    return cast(dict[str, Any], value)


def _string_list(value: object, *, label: str) -> list[str]:
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise ReleaseError(f"{label} must be a string list")
    return cast(list[str], value)


def _positive_integer(value: object, *, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ReleaseError(f"{label} must be a positive integer")
    return value


def _entries(value: object, *, label: str) -> tuple[PolicyEntry, ...]:
    if not isinstance(value, list):
        raise ReleaseError(f"{label} must be a list")
    entries: list[PolicyEntry] = []
    archives: set[str] = set()
    for index, item in enumerate(value):
        raw = _object(item, label=f"{label}[{index}]")
        source = require_safe_relative_path(raw.get("source"), label="entry source")
        archive = require_safe_relative_path(raw.get("archive"), label="entry archive")
        required = raw.get("required")
        if not isinstance(required, bool):
            raise ReleaseError("entry required flag must be boolean")
        if archive in archives:
            raise ReleaseError(f"duplicate archive allowlist path: {archive}")
        archives.add(archive)
        entries.append(PolicyEntry(source=source, archive=archive, required=required))
    return tuple(entries)


def load_policy(path: Path) -> ReleasePolicy:
    """Load and validate an exact allowlist policy; wildcards are never accepted."""
    try:
        document = json.loads(path.read_bytes())
    except (OSError, json.JSONDecodeError) as error:
        raise ReleaseError("submission policy is not valid JSON") from error
    raw = _object(document, label="submission policy")
    if raw.get("schema_version") != _SCHEMA:
        raise ReleaseError("submission policy schema differs")
    archive_root = require_safe_relative_path(raw.get("archive_root"), label="archive root")
    if "/" in archive_root:
        raise ReleaseError("archive root must be one path segment")
    entries = _entries(raw.get("entries"), label="entries")
    if not entries:
        raise ReleaseError("submission allowlist is empty")
    allowed_extensions = frozenset(
        _string_list(raw.get("allowed_extensions"), label="allowed_extensions")
    )
    if any(not item.startswith(".") or "/" in item for item in allowed_extensions):
        raise ReleaseError("allowed extensions must be exact suffixes")
    extensionless_paths = frozenset(
        require_safe_relative_path(item, label="extensionless path")
        for item in _string_list(raw.get("extensionless_paths"), label="extensionless_paths")
    )
    scan = _object(raw.get("scan"), label="scan")
    allowed_emails = frozenset(
        _string_list(scan.get("allowed_emails"), label="scan.allowed_emails")
    )
    raw_exemptions = scan.get("exemptions")
    if not isinstance(raw_exemptions, list):
        raise ReleaseError("scan.exemptions must be a list")
    exemptions: set[tuple[str, str]] = set()
    for index, item in enumerate(raw_exemptions):
        exemption = _object(item, label=f"scan.exemptions[{index}]")
        exempt_path = require_safe_relative_path(
            exemption.get("path"), label="scan exemption exact path"
        )
        if any(character in exempt_path for character in "*?[]"):
            raise ReleaseError("scan exemption must name one exact path")
        rule = exemption.get("rule")
        reason = exemption.get("reason")
        if not isinstance(rule, str) or not rule or any(character in rule for character in "*?[]"):
            raise ReleaseError("scan exemption rule must be exact")
        if not isinstance(reason, str) or len(reason.strip()) < 12:
            raise ReleaseError("scan exemption requires a specific reason")
        exemptions.add((exempt_path, rule))
    web = _object(raw.get("web"), label="web")
    web_status = web.get("status")
    if web_status not in _WEB_STATUSES:
        raise ReleaseError("web status must be pending or ready")
    web_entries = _entries(web.get("entries"), label="web.entries")
    if web_status == "ready" and not web_entries:
        raise ReleaseError("ready web integration has no exact allowlist entries")
    release = _object(raw.get("release"), label="release")
    return ReleasePolicy(
        archive_root=archive_root,
        allowed_extensions=allowed_extensions,
        entries=entries,
        extensionless_paths=extensionless_paths,
        max_file_bytes=_positive_integer(raw.get("max_file_bytes"), label="max_file_bytes"),
        max_total_bytes=_positive_integer(raw.get("max_total_bytes"), label="max_total_bytes"),
        release=release,
        scan_allowed_emails=allowed_emails,
        scan_exemptions=frozenset(exemptions),
        web_entries=web_entries,
        web_status=cast(str, web_status),
    )
