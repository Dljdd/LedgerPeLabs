"""Validation primitives shared by external APAR contracts."""

import re
from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict

_SEMVER = re.compile(r"^(?P<major>0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)$")
_SUPPORTED_SCHEMA_MAJOR = "1"


class ExternalContract(BaseModel):
    """Base configuration for immutable, closed external payloads."""

    model_config = ConfigDict(extra="forbid", frozen=True, revalidate_instances="always")


def validate_schema_version(value: str) -> str:
    """Return a supported semantic schema version or reject incompatibility."""
    validate_semantic_version(value, field_name="schema_version")
    match = _SEMVER.fullmatch(value)
    assert match is not None
    if match.group("major") != _SUPPORTED_SCHEMA_MAJOR:
        raise ValueError("unsupported schema major version")
    return value


def validate_semantic_version(value: str, *, field_name: str) -> str:
    """Return a semantic version without applying envelope compatibility policy."""
    if _SEMVER.fullmatch(value) is None:
        raise ValueError(f"{field_name} must be a semantic version")
    return value


def validate_uuid(value: str) -> str:
    """Ensure public identifiers are UUID strings while preserving their wire form."""
    try:
        UUID(value)
    except (TypeError, ValueError) as error:
        raise ValueError("must be a UUID string") from error
    return value


def validate_utc_timestamp(value: datetime) -> datetime:
    """Require timestamps that are explicitly timezone-aware and in UTC."""
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("timestamp must be timezone-aware and UTC")
    if value.tzname() != "UTC":
        raise ValueError("timestamp must be timezone-aware and UTC")
    return value
