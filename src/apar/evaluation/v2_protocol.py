"""Closed, canonical protocol contracts for synthetic Defend v2."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from apar.contracts._validation import ExternalContract
from apar.runs.wire import WireContractError, canonical_json_bytes, strict_json_loads

_HEX = set("0123456789abcdef")
_V1_ROOTS = {
    "docs/experiments/defense-v1-preregistration.json": "95b460d2bb125e4fd3d432a6d52eb196a8c7e3d72bf5f0cf7d022cf2d9c8b428",
    "docs/experiments/defense-v1-result.json": "6a8512fa8e51b9552629957ff7646e748ea381dff602166950b0b9a6c09eccc0",
    "docs/experiments/defense-v1-run-manifests.json": "eb11ca98912b124a845bdd173a515effb97b83a232b1b6b3937b810933c6e07e",
    "fixtures/defense/v1/hash-manifest.json": "158d5562eb7723a45b7ef5c2c1eededa1378aebe11645e00e2cd6ffcb58bd941",
}


class V2ProtocolError(ValueError):
    """Raised when a protocol or frozen evidence root is not admissible."""


def _digest(value: str, *, field: str) -> str:
    if len(value) != 64 or set(value) - _HEX:
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")
    return value


class PrevalenceStratum(ExternalContract):
    name: Literal["low", "medium", "high"]
    transaction_count: int = Field(gt=0)
    fraud_transaction_count: int = Field(ge=0)
    family_transaction_counts: tuple[int, int, int, int]

    @model_validator(mode="after")
    def allocation_is_exact(self) -> "PrevalenceStratum":
        if len(self.family_transaction_counts) != 4 or any(
            type(value) is not int or value < 0 for value in self.family_transaction_counts
        ):
            raise ValueError("invalid frozen family allocation")
        if sum(self.family_transaction_counts) != self.fraud_transaction_count:
            raise ValueError("invalid frozen family allocation")
        if len(set(self.family_transaction_counts)) != 1:
            raise ValueError("invalid equal family allocation")
        if self.fraud_transaction_count > self.transaction_count:
            raise ValueError("fraud transactions exceed denominator")
        return self

    @classmethod
    def fixture(cls, transaction_count: int = 100, fraud_transaction_count: int = 20) -> "PrevalenceStratum":
        if fraud_transaction_count % 4:
            raise ValueError("fixture fraud count must be divisible by four")
        per_family = fraud_transaction_count // 4
        counts = (per_family,) * 4
        counts = (*counts[:-1], fraud_transaction_count - sum(counts[:-1]))
        return cls(name="low", transaction_count=transaction_count, fraud_transaction_count=fraud_transaction_count, family_transaction_counts=counts)


class OperatingPopulationProfile(ExternalContract):
    transaction_count: int = Field(gt=0)
    day_count: int = Field(gt=0)
    family_names: tuple[str, str, str, str]


class V2Budget(ExternalContract):
    challenge_rate_max: float = Field(ge=0, le=1)
    false_decline_rate_max: float = Field(ge=0, le=1)
    review_case_rate_max: float = Field(ge=0, le=1)


class SeedCommitment(ExternalContract):
    name: str = Field(min_length=1)
    commitment_sha256: str

    _valid_digest = model_validator(mode="after")(
        lambda self: self if _digest(self.commitment_sha256, field="commitment_sha256") else self
    )


class V2Protocol(ExternalContract):
    schema_version: str = "1.0.0"
    protocol_id: str = "apar-defend-v2"
    synthetic_only: bool = True
    fixture_only: bool = False
    operating: OperatingPopulationProfile
    strata: tuple[PrevalenceStratum, PrevalenceStratum, PrevalenceStratum]
    budgets: V2Budget
    seed_commitments: tuple[SeedCommitment, ...]
    v1_roots: dict[str, str] = Field(default_factory=lambda: dict(_V1_ROOTS))
    profile_sha256: str | None = None

    @model_validator(mode="after")
    def validate_contract(self) -> "V2Protocol":
        if not self.synthetic_only:
            raise ValueError("v2 protocol must be synthetic-only")
        names = tuple(item.name for item in self.strata)
        if len(set(names)) != 3 or set(names) != {"low", "medium", "high"}:
            raise ValueError("strata must contain unique low, medium, and high entries")
        if not self.fixture_only:
            expected = (("low", 100), ("medium", 500), ("high", 1_000))
            actual = tuple((s.name, s.fraud_transaction_count) for s in self.strata)
            if actual != expected:
                raise ValueError("invalid frozen production strata")
            if self.operating.transaction_count != 100_000:
                raise ValueError("invalid frozen production denominator")
            if self.operating.day_count != 28:
                raise ValueError("production profile requires 28 synthetic days")
            if any(item.transaction_count != 100_000 for item in self.strata):
                raise ValueError("invalid production stratum denominator")
            if self.v1_roots != _V1_ROOTS:
                raise ValueError("invalid frozen v1 root mapping")
        for path, digest in self.v1_roots.items():
            _digest(digest, field=f"v1_roots[{path}]")
        return self

    @classmethod
    def fixture(cls, *, transaction_count: int = 100) -> "V2Protocol":
        return cls(
            fixture_only=True,
            operating=OperatingPopulationProfile(
                transaction_count=transaction_count, day_count=2,
                family_names=("a", "b", "c", "d"),
            ),
            strata=(
                PrevalenceStratum.fixture(transaction_count, 4),
                PrevalenceStratum(name="medium", transaction_count=transaction_count, fraud_transaction_count=8, family_transaction_counts=(2, 2, 2, 2)),
                PrevalenceStratum(name="high", transaction_count=transaction_count, fraud_transaction_count=12, family_transaction_counts=(3, 3, 3, 3)),
            ),
            budgets=V2Budget(challenge_rate_max=.02, false_decline_rate_max=.001, review_case_rate_max=.01),
            seed_commitments=(),
        )

    def canonical_bytes(self) -> bytes:
        if self.fixture_only:
            raise V2ProtocolError("fixture-only protocol cannot be serialized as preregistration input")
        document = self.model_dump(mode="json", exclude_none=True)
        return canonical_json_bytes(document)

    def to_json(self) -> bytes:
        return self.canonical_bytes()


def load_v2_protocol(path: Path) -> V2Protocol:
    try:
        raw = path.read_bytes()
        document = strict_json_loads(raw)
        if not isinstance(document, dict):
            raise V2ProtocolError("protocol profile must be an object")
        if canonical_json_bytes(document) != raw:
            raise V2ProtocolError("protocol profile is not canonical JSON")
        supplied = document.get("profile_sha256")
        if not isinstance(supplied, str):
            raise V2ProtocolError("profile digest is missing")
        unsigned = dict(document)
        unsigned.pop("profile_sha256", None)
        expected = hashlib.sha256(canonical_json_bytes(unsigned)).hexdigest()
        if supplied != expected:
            raise V2ProtocolError("profile digest mismatch")
        protocol = V2Protocol.model_validate(document)
        if protocol.fixture_only:
            raise V2ProtocolError("fixture-only protocol cannot be loaded as preregistration input")
        return protocol
    except (OSError, WireContractError, ValueError, TypeError) as error:
        if isinstance(error, V2ProtocolError):
            raise
        raise V2ProtocolError(str(error)) from error


def verify_v1_roots(root: Path) -> None:
    for relative, expected in _V1_ROOTS.items():
        path = root / relative
        try:
            actual = hashlib.sha256(path.read_bytes()).hexdigest()
        except OSError as error:
            raise V2ProtocolError(f"frozen v1 root missing: {relative}") from error
        if actual != expected:
            raise V2ProtocolError(f"frozen v1 root mismatch: {relative}")


__all__ = [
    "OperatingPopulationProfile", "PrevalenceStratum", "SeedCommitment",
    "V2Budget", "V2Protocol", "V2ProtocolError", "load_v2_protocol",
    "verify_v1_roots",
]
