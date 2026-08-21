"""Defender-safe public protocol contracts for synthetic Defend v3.

V3 is a separately versioned execution protocol. It reuses v2's fixed metrics,
budgets, gates, stopping rules, arm definitions, and tie-break order without
alteration. This module must not import evaluator packages.
"""

from __future__ import annotations

import hashlib
import math
import json
from pathlib import Path
from typing import Literal, Self, cast

from pydantic import Field, model_validator

from apar.contracts._validation import ExternalContract

PROTOCOL_ID = "apar-defend-v3"
SCHEMA_VERSION = "1.0.0"
MAX_CONFIRMATORY_ATTEMPTS = 1
SYNTHETIC_NON_CLAIM = (
    "Synthetic-only evaluation; not a real-world prevalence or external-validity claim."
)

_SEED_NAMES = (
    "benign_operating_generation",
    "campaign_injection",
    "adversarial_efficacy_generation",
    "public_training",
    "public_calibration",
    "sealed_threshold_selection",
    "model_training",
    "calibration_fitting",
    "threshold_candidate_generation",
    "bootstrap",
    "benign_only_control",
    "score_permutation_control",
    "hidden_evaluation",
)

_V2_PROTOCOL_IDS = ("apar-defend-v2",)
_V1_PROTOCOL_IDS = ("apar-defend-v1",)

_HEX = frozenset("0123456789abcdef")


class V3ProtocolError(ValueError):
    """The isolated public protocol contract or its frozen roots are invalid."""


def _strict_tree(value: object, *, label: str) -> None:
    if value is None or type(value) in {bool, int, str}:
        return
    if type(value) is float:
        if not math.isfinite(value):
            raise V3ProtocolError(f"{label} contains a non-finite number")
        return
    if type(value) is list:
        for item in cast(list[object], value):
            _strict_tree(item, label=label)
        return
    if type(value) is dict:
        for key, item in cast(dict[object, object], value).items():
            if type(key) is not str:
                raise V3ProtocolError(f"{label} keys must be exact strings")
            _strict_tree(item, label=label)
        return
    raise V3ProtocolError(f"{label} contains a non-JSON value")


def _canonical_json_bytes(value: object) -> bytes:
    _strict_tree(value, label="protocol document")
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _digest(value: str, *, field: str) -> str:
    if len(value) != 64 or set(value) - _HEX or value != value.lower():
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")
    return value


class SeedCommitment(ExternalContract):
    name: str = Field(min_length=1)
    commitment_sha256: str

    @model_validator(mode="after")
    def commitment_is_a_digest(self) -> Self:
        _digest(self.commitment_sha256, field="commitment_sha256")
        return self


class V3Budget(ExternalContract):
    challenge_rate_max: float = Field(ge=0, le=1)
    false_decline_rate_max: float = Field(ge=0, le=1)
    review_case_rate_max: float = Field(ge=0, le=1)


class V3GateValues(ExternalContract):
    family_recall_min: float = 0.50
    calibration_ece_max: float = 0.10
    challenge_rate_max: float = 0.02
    false_decline_rate_max: float = 0.001
    review_case_rate_max: float = 0.01
    p95_decision_latency_ms_max: float = 50.0
    captured_value_min: float = 0.50
    escaped_value_max: float = 0.50
    p95_time_to_alert_seconds_max: float = 300.0


class V3Protocol(ExternalContract):
    schema_version: Literal["1.0.0"] = SCHEMA_VERSION
    protocol_id: str = PROTOCOL_ID
    synthetic_only: bool = True
    fixture_only: bool = False
    budgets: V3Budget
    gates: V3GateValues
    seed_commitments: tuple[SeedCommitment, ...]
    maximum_confirmatory_attempts: Literal[1] = MAX_CONFIRMATORY_ATTEMPTS
    profile_sha256: str | None = None

    @model_validator(mode="after")
    def validate_contract(self) -> Self:
        if not self.synthetic_only:
            raise ValueError("v3 protocol must be synthetic-only")
        if self.protocol_id in _V1_PROTOCOL_IDS or self.protocol_id in _V2_PROTOCOL_IDS:
            raise ValueError("v3 protocol must not reuse a v1 or v2 protocol identifier")
        names = tuple(item.name for item in self.seed_commitments)
        if tuple(names) != _SEED_NAMES:
            raise ValueError("v3 seed commitments must contain all thirteen named seeds")
        for item in self.seed_commitments:
            _digest(item.commitment_sha256, field="seed_commitments")
        return self

    @classmethod
    def fixture(
        cls,
        *,
        seed_values: dict[str, int] | None = None,
    ) -> V3Protocol:
        """Return a fixture-only protocol with deterministic seed commitments."""
        seeds = seed_values or {name: index + 1 for index, name in enumerate(_SEED_NAMES)}
        return cls(
            fixture_only=True,
            budgets=V3Budget(
                challenge_rate_max=0.02,
                false_decline_rate_max=0.001,
                review_case_rate_max=0.01,
            ),
            gates=V3GateValues(),
            seed_commitments=tuple(
                SeedCommitment(
                    name=name,
                    commitment_sha256=hashlib.sha256(
                        _canonical_json_bytes({"name": name, "seed": seeds[name]})
                    ).hexdigest(),
                )
                for name in _SEED_NAMES
            ),
        )

    def canonical_bytes(self) -> bytes:
        if self.fixture_only:
            raise V3ProtocolError(
                "fixture-only protocol cannot be serialized as preregistration input"
            )
        document = self.model_dump(mode="json", exclude_none=True)
        return _canonical_json_bytes(document)

    def to_json(self) -> bytes:
        return self.canonical_bytes()


def verify_v1_v2_roots(root: Path) -> None:
    """Verify frozen v1 and v2 evidence roots remain byte-for-byte unchanged."""
    v1_roots = {
        "docs/experiments/defense-v1-preregistration.json": (
            "95b460d2bb125e4fd3d432a6d52eb196a8c7e3d72bf5f0cf7d022cf2d9c8b428"
        ),
        "docs/experiments/defense-v1-result.json": (
            "6a8512fa8e51b9552629957ff7646e748ea381dff602166950b0b9a6c09eccc0"
        ),
        "docs/experiments/defense-v1-run-manifests.json": (
            "eb11ca98912b124a845bdd173a515effb97b83a232b1b6b3937b810933c6e07e"
        ),
        "fixtures/defense/v1/hash-manifest.json": (
            "158d5562eb7723a45b7ef5c2c1eededa1378aebe11645e00e2cd6ffcb58bd941"
        ),
    }
    v2_roots = {
        "config/defense/competition-v2-preregistration.json": (
            "77dd571642d757bbfa41b792df812f63892c2365c59f8da3067cde34dd742a4b"
        ),
        "config/defense/competition-v2-profile.json": (
            "86f8a9f75134661da12d4b6790fc3409fb2d417c4870650bdb03b5f8664e9430"
        ),
        "config/defense/competition-v2-manifests.json": (
            "84da523c41bac361b000dde5f74b5f6195cc5b0cc9175378ed12c73cb7cc92a7"
        ),
    }
    for path_str, expected in {**v1_roots, **v2_roots}.items():
        path = root / path_str
        try:
            actual = hashlib.sha256(path.read_bytes()).hexdigest()
        except OSError as error:
            raise V3ProtocolError(f"frozen root missing: {path_str}") from error
        if actual != expected:
            raise V3ProtocolError(f"frozen root mismatch: {path_str}")


__all__ = [
    "MAX_CONFIRMATORY_ATTEMPTS",
    "PROTOCOL_ID",
    "SYNTHETIC_NON_CLAIM",
    "SCHEMA_VERSION",
    "SeedCommitment",
    "V3Budget",
    "V3GateValues",
    "V3Protocol",
    "V3ProtocolError",
    "verify_v1_v2_roots",
]
