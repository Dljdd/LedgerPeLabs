"""Dependency lock, CycloneDX inventory, and notice consistency checks."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, cast

from scripts.submission.model import ReleaseError

_PIN = re.compile(r"^([A-Za-z0-9._-]+)==([^\s;]+)(?:\s*;.*)?$")


def _normalize(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _requirements(path: Path) -> dict[str, str]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise ReleaseError("judge requirements lock is missing") from error
    result: dict[str, str] = {}
    for raw_line in lines:
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        match = _PIN.fullmatch(line)
        if match is None:
            raise ReleaseError(f"judge requirement is not exactly pinned: {line}")
        name, version = match.groups()
        result[_normalize(name)] = version
    if not result:
        raise ReleaseError("judge requirements lock is empty")
    return result


def _load_sbom(path: Path) -> dict[str, Any]:
    try:
        document = json.loads(path.read_bytes())
    except (OSError, json.JSONDecodeError) as error:
        raise ReleaseError("dependency SBOM is not valid JSON") from error
    if not isinstance(document, dict):
        raise ReleaseError("dependency SBOM must be a JSON object")
    return cast(dict[str, Any], document)


def _properties(metadata: object) -> dict[str, str]:
    if not isinstance(metadata, dict):
        raise ReleaseError("SBOM metadata is absent")
    raw_properties = metadata.get("properties")
    if not isinstance(raw_properties, list):
        raise ReleaseError("SBOM metadata properties are absent")
    result: dict[str, str] = {}
    for item in raw_properties:
        if not isinstance(item, dict):
            raise ReleaseError("SBOM metadata property is malformed")
        name = item.get("name")
        value = item.get("value")
        if not isinstance(name, str) or not isinstance(value, str):
            raise ReleaseError("SBOM metadata property is malformed")
        result[name] = value
    return result


def validate_dependency_inventory(
    *,
    requirements_path: Path,
    sbom_path: Path,
    notice_path: Path,
    web_status: str,
) -> None:
    """Require exact lock/SBOM equality, license IDs, and honest web/project scope."""
    requirements = _requirements(requirements_path)
    sbom = _load_sbom(sbom_path)
    if sbom.get("bomFormat") != "CycloneDX" or sbom.get("specVersion") != "1.6":
        raise ReleaseError("dependency SBOM format differs")
    raw_components = sbom.get("components")
    if not isinstance(raw_components, list):
        raise ReleaseError("dependency SBOM components are absent")
    components: dict[str, str] = {}
    for item in raw_components:
        if not isinstance(item, dict):
            raise ReleaseError("dependency SBOM component is malformed")
        name = item.get("name")
        version = item.get("version")
        licenses = item.get("licenses")
        if not isinstance(name, str) or not isinstance(version, str):
            raise ReleaseError("dependency SBOM component identity is malformed")
        if not isinstance(licenses, list) or not licenses:
            raise ReleaseError(f"dependency SBOM license is absent: {name}")
        components[_normalize(name)] = version
    if components != requirements:
        raise ReleaseError("SBOM components differ from the exact requirements lock")
    properties = _properties(sbom.get("metadata"))
    expected_web = "false" if web_status == "pending" else "true"
    if properties.get("apar:web-dependencies-shipped") != expected_web:
        raise ReleaseError("web dependency scope differs from the integration status")
    if properties.get("apar:project-license-status") != "unspecified":
        raise ReleaseError("project license status is not explicit")
    try:
        notice = notice_path.read_text(encoding="utf-8")
    except OSError as error:
        raise ReleaseError("third-party notice is missing") from error
    if "Project license status: unspecified." not in notice:
        raise ReleaseError("third-party notice omits the project license status")
