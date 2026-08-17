"""Closed public adaptive vectors and decision-only attacker policies.

This module is deliberately independent of campaign generators, simulator rails, trust
verification, and evaluator objects. Evaluator-owned code composes an ``AdaptiveVector``
with a hidden campaign template outside this import boundary.
"""

from __future__ import annotations

import hashlib
import json
from decimal import Decimal
from enum import StrEnum
from typing import Protocol, Self, cast

import numpy as np
from pydantic import ConfigDict, PrivateAttr, field_validator, model_validator

from apar.contracts._validation import ExternalContract
from apar.contracts.decisions import Action

PUBLIC_REASON_FAMILIES = frozenset(
    {
        "amount",
        "approved",
        "authentication",
        "authorization",
        "entity",
        "evaluation_failure",
        "integrity",
        "invalid_candidate",
        "other",
        "policy",
        "recovery",
        "velocity",
    }
)
PUBLIC_CAMPAIGN_FAMILIES = frozenset(
    {
        "agentic_intent_abuse",
        "app_scam_mule",
        "card_testing_cnp",
        "synthetic_merchant_refund",
    }
)
_HEX = frozenset("0123456789abcdef")


class CandidateContractError(ValueError):
    """Stable rejection for an invalid public candidate or bounds object."""


def _exact_text(label: str, value: object) -> str:
    if type(value) is not str:
        raise TypeError(f"{label} must be an exact string")
    if not value:
        raise ValueError(f"{label} must not be empty")
    return value


def _exact_non_negative_int(label: str, value: object, *, maximum: int | None = None) -> int:
    if type(value) is not int or value < 0:
        raise TypeError(f"{label} must be an exact non-negative integer")
    if maximum is not None and value > maximum:
        raise ValueError(f"{label} exceeds {maximum}")
    return value


def _adaptive_value(value: object) -> object:
    if type(value) is Decimal:
        if not value.is_finite():
            raise ValueError("adaptive Decimal must be finite")
        return value
    if type(value) in {int, str}:
        return value
    if type(value) is tuple and value and all(type(item) is str for item in value):
        return value
    raise TypeError("adaptive value must be an exact int, string, finite Decimal, or string tuple")


def _tagged_value(value: object) -> dict[str, object]:
    checked = _adaptive_value(value)
    if type(checked) is Decimal:
        return {"type": "decimal", "value": str(checked)}
    if type(checked) is int:
        return {"type": "integer", "value": checked}
    if type(checked) is str:
        return {"type": "string", "value": checked}
    return {"type": "string_tuple", "value": list(cast(tuple[str, ...], checked))}


def _json_value(value: object) -> object:
    checked = _adaptive_value(value)
    if type(checked) is Decimal:
        return str(checked)
    if type(checked) is tuple:
        return list(cast(tuple[str, ...], checked))
    return checked


def _same_value(left: object, right: object) -> bool:
    if type(left) is not type(right):
        return False
    if type(left) is Decimal:
        assert type(right) is Decimal
        return left.as_tuple() == right.as_tuple()
    return left == right


def _no_op_alias(left: object, right: object) -> bool:
    if type(left) is Decimal and type(right) is Decimal:
        return left == right
    return _same_value(left, right)


def _canonical_bytes(document: object) -> bytes:
    return json.dumps(
        document,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode()


def _digest(document: object) -> str:
    return hashlib.sha256(_canonical_bytes(document)).hexdigest()


def _assert_digest(label: str, value: object) -> str:
    text = _exact_text(label, value)
    if len(text) != 64 or not set(text) <= _HEX:
        raise ValueError(f"{label} must be lowercase SHA-256 hex")
    return text


def _set_integrity_seal(contract: ExternalContract, seal: str) -> None:
    private = contract.__pydantic_private__
    if private is None:
        raise RuntimeError("contract private storage is unavailable")
    private["_integrity_seal"] = seal


class DomainKind(StrEnum):
    """Declared sampling semantics for a finite canonical domain."""

    CATEGORICAL = "categorical"
    DISCRETE = "discrete"
    LINEAR = "linear"
    LOG = "log"


class AdaptiveParameter(ExternalContract):
    """One named, exact policy-visible adaptive value."""

    model_config = ConfigDict(
        extra="forbid", frozen=True, revalidate_instances="always", arbitrary_types_allowed=True
    )
    name: str
    value: object
    _integrity_seal: str = PrivateAttr(default="")

    @field_validator("name", mode="before")
    @classmethod
    def name_is_exact(cls, value: object) -> object:
        return _exact_text("parameter name", value)

    @field_validator("value", mode="before")
    @classmethod
    def value_is_exact(cls, value: object) -> object:
        return _adaptive_value(value)

    def model_post_init(self, _context: object) -> None:
        _set_integrity_seal(self, _digest(self.document()))

    def document(self) -> dict[str, object]:
        return {"name": self.name, "value": _tagged_value(self.value)}

    def assert_pristine(self) -> None:
        if type(self) is not AdaptiveParameter:
            raise CandidateContractError("adaptive parameter subclasses are forbidden")
        if self.__pydantic_extra__ or set(self.__dict__) != set(type(self).model_fields):
            raise CandidateContractError("adaptive parameter field set is not exact")
        if self._integrity_seal != _digest(self.document()):
            raise CandidateContractError("adaptive parameter integrity seal changed")


class AdaptiveVector(ExternalContract):
    """Sanitized immutable replacement for evaluator-owned campaign parameters."""

    model_config = ConfigDict(
        extra="forbid", frozen=True, revalidate_instances="always", arbitrary_types_allowed=True
    )
    entries: tuple[AdaptiveParameter, ...]
    _integrity_seal: str = PrivateAttr(default="")

    @field_validator("entries", mode="before")
    @classmethod
    def entries_are_exact(cls, value: object) -> object:
        if type(value) is not tuple or any(type(item) is not AdaptiveParameter for item in value):
            raise TypeError("entries must be an exact tuple of AdaptiveParameter records")
        for entry in value:
            entry.assert_pristine()
        names = tuple(entry.name for entry in value)
        if names != tuple(sorted(set(names))):
            raise ValueError("adaptive parameter names must be unique and sorted")
        return value

    def model_post_init(self, _context: object) -> None:
        _set_integrity_seal(self, self.fingerprint)

    @classmethod
    def from_mapping(cls, values: dict[str, object]) -> AdaptiveVector:
        if type(values) is not dict or any(type(key) is not str for key in values):
            raise CandidateContractError("adaptive mapping needs exact string keys")
        return cls(
            entries=tuple(
                AdaptiveParameter(name=name, value=values[name]) for name in sorted(values)
            )
        )

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(entry.name for entry in self.entries)

    @property
    def fingerprint(self) -> str:
        private = self.__pydantic_private__
        if private is not None and private.get("_integrity_seal"):
            return cast(str, private["_integrity_seal"])
        return _digest(self.document())

    def document(self) -> list[dict[str, object]]:
        return [entry.document() for entry in self.entries]

    def json_mapping(self) -> dict[str, object]:
        return {entry.name: _json_value(entry.value) for entry in self.entries}

    def get(self, name: str) -> object:
        checked = _exact_text("parameter name", name)
        for entry in self.entries:
            if entry.name == checked:
                return entry.value
        raise CandidateContractError(f"undeclared adaptive parameter: {checked}")

    def assert_pristine(self) -> None:
        if type(self) is not AdaptiveVector:
            raise CandidateContractError("adaptive vector subclasses are forbidden")
        if self.__pydantic_extra__ or set(self.__dict__) != set(type(self).model_fields):
            raise CandidateContractError("adaptive vector field set is not exact")
        for entry in self.entries:
            entry.assert_pristine()
        if self._integrity_seal != _digest(self.document()):
            raise CandidateContractError("adaptive vector integrity seal changed")


class ParameterDomain(ExternalContract):
    """One exact finite domain whose values are feasible in at least one vector."""

    model_config = ConfigDict(
        extra="forbid", frozen=True, revalidate_instances="always", arbitrary_types_allowed=True
    )
    name: str
    kind: DomainKind
    values: tuple[object, ...]
    _integrity_seal: str = PrivateAttr(default="")

    @field_validator("name", mode="before")
    @classmethod
    def name_is_exact(cls, value: object) -> object:
        return _exact_text("domain name", value)

    @field_validator("kind", mode="before")
    @classmethod
    def kind_is_exact(cls, value: object) -> object:
        if type(value) is not DomainKind:
            raise TypeError("domain kind must be an exact DomainKind")
        return value

    @field_validator("values", mode="before")
    @classmethod
    def values_are_exact(cls, value: object) -> object:
        if type(value) is not tuple or not value:
            raise TypeError("domain values must be a non-empty exact tuple")
        for item in value:
            _adaptive_value(item)
        if any(
            _no_op_alias(left, right)
            for index, left in enumerate(value)
            for right in value[index + 1 :]
        ):
            raise ValueError("domain values contain numerically equal no-op aliases")
        return value

    def model_post_init(self, _context: object) -> None:
        _set_integrity_seal(self, _digest(self.document()))

    def document(self) -> dict[str, object]:
        return {
            "name": self.name,
            "kind": self.kind.value,
            "values": [_tagged_value(value) for value in self.values],
        }

    def alternatives(self, current: object) -> tuple[object, ...]:
        return tuple(value for value in self.values if not _same_value(value, current))

    def assert_pristine(self) -> None:
        if type(self) is not ParameterDomain:
            raise CandidateContractError("parameter domain subclasses are forbidden")
        if self.__pydantic_extra__ or set(self.__dict__) != set(type(self).model_fields):
            raise CandidateContractError("parameter domain field set is not exact")
        if self._integrity_seal != _digest(self.document()):
            raise CandidateContractError("parameter domain integrity seal changed")


class ParameterBounds(ExternalContract):
    """Only public adaptive defaults, domains, and mutually feasible vectors."""

    model_config = ConfigDict(
        extra="forbid", frozen=True, revalidate_instances="always", arbitrary_types_allowed=True
    )
    family: str
    defaults: AdaptiveVector
    domains: tuple[ParameterDomain, ...]
    feasible_vectors: tuple[AdaptiveVector, ...]
    _integrity_seal: str = PrivateAttr(default="")

    @field_validator("family", mode="before")
    @classmethod
    def family_is_public(cls, value: object) -> object:
        checked = _exact_text("family", value)
        if checked not in PUBLIC_CAMPAIGN_FAMILIES:
            raise ValueError("unsupported public campaign family")
        return checked

    @field_validator("defaults", mode="before")
    @classmethod
    def defaults_are_exact(cls, value: object) -> object:
        if type(value) is not AdaptiveVector:
            raise TypeError("defaults must be an exact AdaptiveVector")
        value.assert_pristine()
        return value

    @field_validator("domains", mode="before")
    @classmethod
    def domains_are_exact(cls, value: object) -> object:
        if type(value) is not tuple or any(type(item) is not ParameterDomain for item in value):
            raise TypeError("domains must be an exact tuple of ParameterDomain records")
        for domain in value:
            domain.assert_pristine()
        return value

    @field_validator("feasible_vectors", mode="before")
    @classmethod
    def feasible_vectors_are_exact(cls, value: object) -> object:
        if (
            type(value) is not tuple
            or not value
            or any(type(item) is not AdaptiveVector for item in value)
        ):
            raise TypeError("feasible_vectors must be a non-empty exact tuple")
        for vector in value:
            vector.assert_pristine()
        return value

    @model_validator(mode="after")
    def bounds_are_closed(self) -> Self:
        names = tuple(domain.name for domain in self.domains)
        if names != tuple(sorted(set(names))) or self.defaults.names != names:
            raise ValueError("domains and defaults must have exact unique sorted names")
        fingerprints = tuple(vector.fingerprint for vector in self.feasible_vectors)
        if fingerprints != tuple(sorted(set(fingerprints))):
            raise ValueError("feasible vectors must be unique and fingerprint-sorted")
        for vector in self.feasible_vectors:
            if vector.names != names:
                raise ValueError("every feasible vector must contain exactly the domain names")
            for domain in self.domains:
                if not any(
                    _same_value(vector.get(domain.name), allowed) for allowed in domain.values
                ):
                    raise ValueError(f"vector value is outside domain: {domain.name}")
        if self.defaults.fingerprint not in set(fingerprints):
            raise ValueError("default vector must be feasible")
        for domain in self.domains:
            represented = tuple(vector.get(domain.name) for vector in self.feasible_vectors)
            if any(
                not any(_same_value(value, observed) for observed in represented)
                for value in domain.values
            ):
                raise ValueError(f"domain advertises an infeasible value: {domain.name}")
        return self

    def model_post_init(self, _context: object) -> None:
        _set_integrity_seal(self, self.bounds_digest)

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(domain.name for domain in self.domains)

    @property
    def bounds_digest(self) -> str:
        private = self.__pydantic_private__
        if private is not None and private.get("_integrity_seal"):
            return cast(str, private["_integrity_seal"])
        return _digest(self.document())

    @property
    def has_non_no_op_mutation(self) -> bool:
        return len(self.feasible_vectors) > 1

    def document(self) -> dict[str, object]:
        return {
            "family": self.family,
            "defaults": self.defaults.document(),
            "domains": [domain.document() for domain in self.domains],
            "feasible_vectors": [vector.document() for vector in self.feasible_vectors],
        }

    def schema_document(self) -> dict[str, object]:
        return {
            "family": self.family,
            "parameters": [
                {
                    "name": domain.name,
                    "kind": domain.kind.value,
                    "allowed_values": [_json_value(value) for value in domain.values],
                }
                for domain in self.domains
            ],
            "required": list(self.names),
            "additional_parameters": False,
        }

    def defaults_document(self) -> dict[str, object]:
        return self.defaults.json_mapping()

    def domain(self, name: str) -> ParameterDomain:
        checked = _exact_text("domain name", name)
        for domain in self.domains:
            if domain.name == checked:
                return domain
        raise CandidateContractError(f"undeclared adaptive parameter: {checked}")

    def validate_vector(self, vector: AdaptiveVector) -> AdaptiveVector:
        rebuilt = reconstruct_vector(vector)
        if rebuilt.names != self.names:
            raise CandidateContractError("candidate fields do not match public bounds")
        if rebuilt.fingerprint not in {
            candidate.fingerprint for candidate in self.feasible_vectors
        }:
            raise CandidateContractError("candidate is not a declared feasible vector")
        return rebuilt

    def decode_updates(self, document: object) -> AdaptiveVector:
        if type(document) is not dict or any(type(key) is not str for key in document):
            raise CandidateContractError("planner params must have exact string keys")
        if set(document) != set(self.names):
            unknown = sorted(set(document) - set(self.names))
            missing = sorted(set(self.names) - set(document))
            raise CandidateContractError(
                f"planner params have undeclared parameter={unknown} missing={missing} fields"
            )
        values: dict[str, object] = {}
        for name in self.names:
            raw = document[name]
            example = self.domain(name).values[0]
            if type(example) is Decimal:
                if type(raw) is not str:
                    raise CandidateContractError(f"{name} must be a canonical Decimal string")
                try:
                    decoded: object = Decimal(raw)
                except Exception as error:
                    raise CandidateContractError(f"{name} is not a Decimal") from error
                if not any(
                    type(value) is Decimal and str(value) == raw
                    for value in self.domain(name).values
                ):
                    raise CandidateContractError(f"{name} is outside its canonical domain")
            elif type(example) is tuple:
                if type(raw) is not list or not raw or any(type(item) is not str for item in raw):
                    raise CandidateContractError(f"{name} must be an exact string array")
                decoded = tuple(raw)
            elif type(raw) is not type(example):
                raise CandidateContractError(f"{name} has the wrong exact type")
            else:
                decoded = raw
            values[name] = decoded
        return self.validate_vector(AdaptiveVector.from_mapping(values))

    def changed_field_count(self, left: AdaptiveVector, right: AdaptiveVector) -> int:
        first = self.validate_vector(left)
        second = self.validate_vector(right)
        return sum(not _same_value(first.get(name), second.get(name)) for name in self.names)

    def mutations(self, base: AdaptiveVector) -> tuple[AdaptiveVector, ...]:
        checked = self.validate_vector(base)
        return tuple(
            vector
            for vector in self.feasible_vectors
            if 1 <= self.changed_field_count(checked, vector) <= 3
        )

    def assert_pristine(self) -> None:
        if type(self) is not ParameterBounds:
            raise CandidateContractError("parameter bounds subclasses are forbidden")
        if self.__pydantic_extra__ or set(self.__dict__) != set(type(self).model_fields):
            raise CandidateContractError("parameter bounds field set is not exact")
        self.defaults.assert_pristine()
        for domain in self.domains:
            domain.assert_pristine()
        for vector in self.feasible_vectors:
            vector.assert_pristine()
        if self._integrity_seal != _digest(self.document()):
            raise CandidateContractError("parameter bounds integrity seal changed")


def reconstruct_vector(value: AdaptiveVector) -> AdaptiveVector:
    if type(value) is not AdaptiveVector:
        raise CandidateContractError("candidate vector must be exact")
    value.assert_pristine()
    return AdaptiveVector(
        entries=tuple(
            AdaptiveParameter(name=entry.name, value=entry.value) for entry in value.entries
        )
    )


def reconstruct_bounds(value: ParameterBounds) -> ParameterBounds:
    if type(value) is not ParameterBounds:
        raise CandidateContractError("bounds must be exact")
    value.assert_pristine()
    return ParameterBounds(
        family=value.family,
        defaults=reconstruct_vector(value.defaults),
        domains=tuple(
            ParameterDomain(name=item.name, kind=item.kind, values=item.values)
            for item in value.domains
        ),
        feasible_vectors=tuple(reconstruct_vector(vector) for vector in value.feasible_vectors),
    )


class AttackCandidate(ExternalContract):
    """A canonical public candidate containing no hidden campaign configuration."""

    model_config = ConfigDict(
        extra="forbid", frozen=True, revalidate_instances="always", arbitrary_types_allowed=True
    )
    params: AdaptiveVector
    parent_id: str | None
    generation: int
    _integrity_seal: str = PrivateAttr(default="")

    @field_validator("params", mode="before")
    @classmethod
    def params_are_exact(cls, value: object) -> object:
        if type(value) is not AdaptiveVector:
            raise TypeError("params must be an exact AdaptiveVector")
        value.assert_pristine()
        return value

    @field_validator("parent_id", mode="before")
    @classmethod
    def parent_is_canonical(cls, value: object) -> object:
        return value if value is None else _assert_digest("parent_id", value)

    @field_validator("generation", mode="before")
    @classmethod
    def generation_is_exact(cls, value: object) -> object:
        return _exact_non_negative_int("generation", value, maximum=1000)

    def model_post_init(self, _context: object) -> None:
        _set_integrity_seal(self, self.candidate_id)

    @property
    def fingerprint(self) -> str:
        return self.params.fingerprint

    @property
    def candidate_id(self) -> str:
        private = self.__pydantic_private__
        if private is not None and private.get("_integrity_seal"):
            return cast(str, private["_integrity_seal"])
        return self._computed_candidate_id()

    def _computed_candidate_id(self) -> str:
        return _digest(
            {
                "fingerprint": self.fingerprint,
                "generation": self.generation,
                "parent_id": self.parent_id,
            }
        )

    def assert_pristine(self) -> None:
        if type(self) is not AttackCandidate:
            raise CandidateContractError("candidate subclasses are forbidden")
        if self.__pydantic_extra__ or set(self.__dict__) != set(type(self).model_fields):
            raise CandidateContractError("candidate field set is not exact")
        self.params.assert_pristine()
        if self._integrity_seal != self._computed_candidate_id():
            raise CandidateContractError("candidate integrity seal changed")


def reconstruct_candidate(value: AttackCandidate) -> AttackCandidate:
    if type(value) is not AttackCandidate:
        raise CandidateContractError("candidate must be exact")
    value.assert_pristine()
    return AttackCandidate(
        params=reconstruct_vector(value.params),
        parent_id=value.parent_id,
        generation=value.generation,
    )


class Feedback(ExternalContract):
    """The complete and intentionally coarse attacker observation."""

    action: Action
    reason_family: str
    realized_value: Decimal | None
    _integrity_seal: str = PrivateAttr(default="")

    @field_validator("action", mode="before")
    @classmethod
    def action_is_exact(cls, value: object) -> object:
        if type(value) is not Action:
            raise ValueError("action must be an exact Action")
        return value

    @field_validator("reason_family", mode="before")
    @classmethod
    def reason_is_public(cls, value: object) -> object:
        checked = _exact_text("reason_family", value)
        if checked not in PUBLIC_REASON_FAMILIES:
            raise ValueError("reason_family is not in the public allowlist")
        return checked

    @field_validator("realized_value", mode="before")
    @classmethod
    def realized_value_is_exact(cls, value: object) -> object:
        if value is None:
            return value
        if type(value) is not Decimal:
            raise TypeError("realized_value must be an exact Decimal or None")
        if not value.is_finite() or value < 0:
            raise ValueError("realized_value must be finite and non-negative")
        try:
            canonical = value.quantize(Decimal("0.01"))
        except Exception as error:
            raise ValueError("realized_value cannot be canonicalized") from error
        if value.as_tuple().exponent != -2 or value != canonical:
            raise ValueError("realized_value must be canonically quantized")
        return value

    @model_validator(mode="after")
    def action_and_reason_are_consistent(self) -> Self:
        if (self.action is Action.APPROVE) != (self.reason_family == "approved"):
            raise ValueError("approved reason_family must exactly match approve action")
        if (
            self.reason_family in {"evaluation_failure", "invalid_candidate"}
            and self.action is not Action.DECLINE
        ):
            raise ValueError("failure reason families must decline")
        return self

    def model_post_init(self, _context: object) -> None:
        _set_integrity_seal(self, self._computed_seal())

    def _computed_seal(self) -> str:
        return _digest(
            {
                "action": self.action.value,
                "reason_family": self.reason_family,
                "realized_value": None if self.realized_value is None else str(self.realized_value),
            }
        )

    def assert_pristine(self) -> None:
        if type(self) is not Feedback:
            raise CandidateContractError("feedback subclasses are forbidden")
        if self.__pydantic_extra__ or set(self.__dict__) != set(type(self).model_fields):
            raise CandidateContractError("feedback field set is not exact")
        if self._integrity_seal != self._computed_seal():
            raise CandidateContractError("feedback integrity seal changed")


def reconstruct_feedback(value: Feedback) -> Feedback:
    if type(value) is not Feedback:
        raise CandidateContractError("feedback must be exact")
    value.assert_pristine()
    return Feedback(
        action=value.action, reason_family=value.reason_family, realized_value=value.realized_value
    )


def visible_objective(feedback: Feedback) -> Decimal:
    """Compute the only objective permitted in policy-visible history."""
    checked = reconstruct_feedback(feedback)
    penalty = {
        Action.APPROVE: Decimal(0),
        Action.CHALLENGE: Decimal("0.25"),
        Action.DECLINE: Decimal(1),
    }[checked.action]
    return (checked.realized_value or Decimal(0)) - penalty


class VisibleTrial(ExternalContract):
    """Past-only policy history containing no evaluator feature or hidden reason."""

    model_config = ConfigDict(
        extra="forbid", frozen=True, revalidate_instances="always", arbitrary_types_allowed=True
    )
    candidate: AttackCandidate
    feedback: Feedback
    objective_value: Decimal
    _integrity_seal: str = PrivateAttr(default="")

    @field_validator("candidate", mode="before")
    @classmethod
    def candidate_is_exact(cls, value: object) -> object:
        if type(value) is not AttackCandidate:
            raise TypeError("candidate must be exact")
        value.assert_pristine()
        return value

    @field_validator("feedback", mode="before")
    @classmethod
    def feedback_is_exact(cls, value: object) -> object:
        if type(value) is not Feedback:
            raise TypeError("feedback must be exact")
        value.assert_pristine()
        return value

    @field_validator("objective_value", mode="before")
    @classmethod
    def objective_is_exact(cls, value: object) -> object:
        if type(value) is not Decimal or not value.is_finite():
            raise TypeError("objective_value must be an exact finite Decimal")
        return value

    @model_validator(mode="after")
    def objective_is_public(self) -> Self:
        if self.objective_value != visible_objective(self.feedback):
            raise ValueError("objective_value must be derived only from public feedback")
        return self

    def model_post_init(self, _context: object) -> None:
        _set_integrity_seal(self, self._computed_seal())

    def _computed_seal(self) -> str:
        return _digest(
            {
                "candidate_id": self.candidate.candidate_id,
                "feedback": self.feedback._computed_seal(),
                "objective": str(self.objective_value),
            }
        )

    def assert_pristine(self) -> None:
        if type(self) is not VisibleTrial:
            raise CandidateContractError("visible trial subclasses are forbidden")
        if self.__pydantic_extra__ or set(self.__dict__) != set(type(self).model_fields):
            raise CandidateContractError("visible trial field set is not exact")
        self.candidate.assert_pristine()
        self.feedback.assert_pristine()
        if self._integrity_seal != self._computed_seal():
            raise CandidateContractError("visible trial integrity seal changed")


def reconstruct_history(value: tuple[VisibleTrial, ...]) -> tuple[VisibleTrial, ...]:
    if type(value) is not tuple or any(type(item) is not VisibleTrial for item in value):
        raise CandidateContractError("history must be an exact tuple of VisibleTrial records")
    for item in value:
        item.assert_pristine()
    ordered = tuple(sorted(value, key=lambda item: item.candidate.generation))
    if tuple(item.candidate.generation for item in ordered) != tuple(range(len(ordered))):
        raise CandidateContractError("visible history generations must be contiguous")
    seen: set[str] = set()
    for index, trial in enumerate(ordered):
        candidate = trial.candidate
        if index == 0 and candidate.parent_id is not None:
            raise CandidateContractError("root candidate cannot have a parent")
        if index > 0 and candidate.parent_id not in seen:
            raise CandidateContractError("candidate parent must reference earlier history")
        seen.add(candidate.candidate_id)
    return ordered


def normalize_internal_history(
    value: tuple[VisibleTrial, ...],
) -> tuple[VisibleTrial, ...]:
    """Normalize evaluator-created immutable history without rebuilding old trials."""
    if type(value) is not tuple or any(type(item) is not VisibleTrial for item in value):
        raise CandidateContractError("history must be an exact tuple of VisibleTrial records")
    ordered = tuple(sorted(value, key=lambda item: item.candidate.generation))
    if tuple(item.candidate.generation for item in ordered) != tuple(range(len(ordered))):
        raise CandidateContractError("visible history generations must be contiguous")
    seen: set[str] = set()
    for index, trial in enumerate(ordered):
        candidate = trial.candidate
        if index == 0 and candidate.parent_id is not None:
            raise CandidateContractError("root candidate cannot have a parent")
        if index > 0 and candidate.parent_id not in seen:
            raise CandidateContractError("candidate parent must reference earlier history")
        seen.add(candidate.candidate_id)
    return ordered


def validate_candidate_lineage(
    candidate: AttackCandidate, history: tuple[VisibleTrial, ...]
) -> AttackCandidate:
    checked = reconstruct_candidate(candidate)
    visible = reconstruct_history(history)
    if checked.generation != len(visible):
        raise CandidateContractError("candidate generation must equal visible history length")
    if not visible and checked.parent_id is not None:
        raise CandidateContractError("root candidate cannot have a parent")
    if visible and checked.parent_id not in {trial.candidate.candidate_id for trial in visible}:
        raise CandidateContractError("candidate parent must reference visible history")
    return checked


class Policy(Protocol):
    policy_name: str
    policy_version: str

    def propose(
        self, history: tuple[VisibleTrial, ...], bounds: ParameterBounds, rng: np.random.Generator
    ) -> AttackCandidate: ...


def _inputs(
    history: tuple[VisibleTrial, ...], bounds: ParameterBounds, rng: np.random.Generator
) -> tuple[tuple[VisibleTrial, ...], ParameterBounds]:
    if type(rng) is not np.random.Generator:
        raise TypeError("rng must be an exact numpy.random.Generator")
    return normalize_internal_history(history), bounds


def _parent(history: tuple[VisibleTrial, ...]) -> str | None:
    return None if not history else history[-1].candidate.candidate_id


class FixedPolicy:
    policy_name = "fixed"
    policy_version = "1.0.0"

    def propose(
        self, history: tuple[VisibleTrial, ...], bounds: ParameterBounds, rng: np.random.Generator
    ) -> AttackCandidate:
        visible, public_bounds = _inputs(history, bounds, rng)
        return AttackCandidate(
            params=public_bounds.defaults, parent_id=_parent(visible), generation=len(visible)
        )


class RandomPolicy:
    policy_name = "random"
    policy_version = "1.0.0"

    def propose(
        self, history: tuple[VisibleTrial, ...], bounds: ParameterBounds, rng: np.random.Generator
    ) -> AttackCandidate:
        visible, public_bounds = _inputs(history, bounds, rng)
        vector = public_bounds.feasible_vectors[
            int(rng.integers(0, len(public_bounds.feasible_vectors)))
        ]
        return AttackCandidate(params=vector, parent_id=_parent(visible), generation=len(visible))


class AdaptiveTournamentPolicy:
    policy_name = "adaptive"
    policy_version = "1.0.0"

    def propose(
        self, history: tuple[VisibleTrial, ...], bounds: ParameterBounds, rng: np.random.Generator
    ) -> AttackCandidate:
        visible, public_bounds = _inputs(history, bounds, rng)
        if not visible:
            alternatives = public_bounds.mutations(public_bounds.defaults)
            vector = (
                public_bounds.defaults
                if not alternatives
                else alternatives[int(rng.integers(0, len(alternatives)))]
            )
            return AttackCandidate(params=vector, parent_id=None, generation=0)
        eligible = tuple(
            sorted(
                (
                    trial
                    for trial in visible
                    if trial.feedback.reason_family
                    not in {"evaluation_failure", "invalid_candidate"}
                ),
                key=lambda trial: trial.candidate.candidate_id,
            )
        )
        if not eligible:
            parent = visible[-1]
        else:
            sample_size = min(3, len(eligible))
            indices = tuple(
                int(index) for index in rng.choice(len(eligible), size=sample_size, replace=False)
            )
            contenders = tuple(eligible[index] for index in indices)
            parent = min(
                contenders, key=lambda trial: (-trial.objective_value, trial.candidate.candidate_id)
            )
        alternatives = public_bounds.mutations(parent.candidate.params)
        if not alternatives:
            vector = parent.candidate.params
        else:
            guided = self._guided(alternatives, parent, public_bounds)
            choices = guided or alternatives
            seen = {trial.candidate.fingerprint for trial in visible}
            novel = tuple(vector for vector in choices if vector.fingerprint not in seen)
            pool = novel or choices
            vector = pool[int(rng.integers(0, len(pool)))]
        return AttackCandidate(
            params=vector, parent_id=parent.candidate.candidate_id, generation=len(visible)
        )

    @staticmethod
    def _guided(
        alternatives: tuple[AdaptiveVector, ...], parent: VisibleTrial, bounds: ParameterBounds
    ) -> tuple[AdaptiveVector, ...]:
        reason = parent.feedback.reason_family
        parameter: str | None = None
        if reason == "velocity" and "retry_intensity" in bounds.names:
            parameter = "retry_intensity"
        elif reason == "entity" and "mule_fanout" in bounds.names:
            parameter = "mule_fanout"
        elif reason == "amount" and "cash_out_fraction" in bounds.names:
            parameter = "cash_out_fraction"
        if parameter is None:
            return ()
        current = parent.candidate.params.get(parameter)
        if type(current) not in {int, Decimal}:
            return ()
        return tuple(
            vector
            for vector in alternatives
            if type(vector.get(parameter)) is type(current)
            and cast(int | Decimal, vector.get(parameter)) < cast(int | Decimal, current)
        )


__all__ = [
    "PUBLIC_CAMPAIGN_FAMILIES",
    "PUBLIC_REASON_FAMILIES",
    "AdaptiveParameter",
    "AdaptiveTournamentPolicy",
    "AdaptiveVector",
    "AttackCandidate",
    "CandidateContractError",
    "DomainKind",
    "Feedback",
    "FixedPolicy",
    "ParameterBounds",
    "ParameterDomain",
    "Policy",
    "RandomPolicy",
    "VisibleTrial",
    "reconstruct_bounds",
    "reconstruct_candidate",
    "reconstruct_feedback",
    "reconstruct_history",
    "reconstruct_vector",
    "normalize_internal_history",
    "validate_candidate_lineage",
    "visible_objective",
]
