"""Evaluator-owned orchestration, provenance, deadlines, and capability metrics."""

from __future__ import annotations

import hashlib
import hmac
import inspect
import json
import marshal
import secrets
import time
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass, field
from decimal import Decimal, localcontext
from enum import StrEnum
from types import ModuleType
from typing import Self, cast

import numpy as np
from pydantic import ConfigDict, PrivateAttr, field_validator, model_validator

from apar.contracts._validation import ExternalContract
from apar.contracts.decisions import Action
from apar.redteam.policies import (
    AttackCandidate,
    CandidateContractError,
    Feedback,
    ParameterBounds,
    Policy,
    VisibleTrial,
    normalize_internal_history,
    reconstruct_bounds,
    reconstruct_candidate,
    reconstruct_feedback,
    reconstruct_history,
    validate_candidate_lineage,
    visible_objective,
)

_POLICY_NAMES = ("adaptive", "cached_llm", "fixed", "random")
_HEX = frozenset("0123456789abcdef")


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


def _implementation_digest(owner: object, entrypoint: Callable[..., object]) -> str:
    """Bind the exact registered type and all Python code reachable on that type."""
    if not inspect.ismethod(entrypoint) or entrypoint.__self__ is not owner:
        raise TypeError("registered entrypoint must be an exact bound owner method")
    records: list[dict[str, str]] = []
    for implementation_type in reversed(type(owner).__mro__):
        if implementation_type is object:
            continue
        for name, member in sorted(implementation_type.__dict__.items()):
            functions: tuple[Callable[..., object], ...] = ()
            if inspect.isfunction(member):
                functions = (member,)
            elif isinstance(member, (staticmethod, classmethod)):
                functions = (member.__func__,)
            elif isinstance(member, property):
                functions = tuple(
                    function
                    for function in (member.fget, member.fset, member.fdel)
                    if function is not None
                )
            for index, function in enumerate(functions):
                records.append(
                    {
                        "owner": (
                            f"{implementation_type.__module__}."
                            f"{implementation_type.__qualname__}"
                        ),
                        "member": f"{name}:{index}",
                        "code": hashlib.sha256(marshal.dumps(function.__code__)).hexdigest(),
                    }
                )
    return _digest(
        {
            "registered_type": f"{type(owner).__module__}.{type(owner).__qualname__}",
            "entrypoint": entrypoint.__func__.__qualname__,
            "implementation_records": records,
        }
    )


def _bound_callable_digest(owner: object, entrypoint: Callable[..., object]) -> str:
    """Digest the exact stored bound callable without a dynamic attribute lookup."""
    if not inspect.ismethod(entrypoint) or entrypoint.__self__ is not owner:
        raise TypeError("registered callable must remain bound to the exact owner")
    return _digest(
        {
            "owner_type": f"{type(owner).__module__}.{type(owner).__qualname__}",
            "callable": entrypoint.__func__.__qualname__,
            "code": hashlib.sha256(marshal.dumps(entrypoint.__func__.__code__)).hexdigest(),
        }
    )


def _source_name(value: object) -> str:
    source: str | None = None
    if callable(value):
        with suppress(OSError, TypeError):
            source = inspect.getsourcefile(cast(Callable[..., object], value))
    if source is None and isinstance(value, type):
        for member in value.__dict__.values():
            function = (
                member.__func__
                if isinstance(member, (staticmethod, classmethod))
                else member
            )
            if inspect.isfunction(function):
                source = function.__code__.co_filename
                break
    if source is None:
        source = getattr(value, "__module__", type(value).__module__)
    return source.replace("\\", "/").rsplit("/", maxsplit=1)[-1]


def _type_name(value: object) -> str:
    value_type = value if isinstance(value, type) else type(value)
    return f"{_source_name(value_type)}:{value_type.__qualname__}"


def _literal_document(value: object) -> object | None:
    if value is None or type(value) in {bool, int, str}:
        return {"type": type(value).__name__, "value": value}
    if type(value) is float:
        return {"type": "float", "value": value.hex()}
    if type(value) is bytes:
        return {"type": "bytes", "value": value.hex()}
    if type(value) is Decimal:
        return {"type": "Decimal", "value": str(value)}
    if type(value) in {tuple, frozenset}:
        items = cast(tuple[object, ...] | frozenset[object], value)
        documents = [_literal_document(item) for item in items]
        if any(document is None for document in documents):
            return None
        if type(value) is frozenset:
            documents.sort(key=_canonical_bytes)
        return {"type": type(value).__name__, "items": documents}
    return None


def _function_document(
    function: Callable[..., object],
    seen: set[int],
) -> dict[str, object]:
    if not inspect.isfunction(function):
        raise TypeError("runtime function dependency must be a Python function")
    code_digest = hashlib.sha256(marshal.dumps(function.__code__)).hexdigest()
    reference = {
        "source": _source_name(function),
        "qualname": function.__qualname__,
        "code": code_digest,
    }
    if id(function) in seen:
        return {"function_reference": reference}
    seen.add(id(function))
    dependencies: dict[str, object] = {}
    for name in sorted(set(function.__code__.co_names)):
        if name not in function.__globals__:
            continue
        dependency = _global_dependency_document(function.__globals__[name], seen)
        if dependency is not None:
            dependencies[name] = dependency
    defaults = _literal_document(function.__defaults__)
    keyword_defaults = _literal_document(
        None
        if function.__kwdefaults__ is None
        else tuple(sorted(function.__kwdefaults__.items()))
    )
    return {
        "function": reference,
        "defaults": defaults,
        "keyword_defaults": keyword_defaults,
        "globals": dependencies,
    }


def _callable_document(value: object, seen: set[int]) -> dict[str, object]:
    if inspect.ismethod(value):
        return {
            "kind": "bound_method",
            "owner_type": _type_name(value.__self__),
            "function": _function_document(value.__func__, seen),
        }
    if inspect.isfunction(value):
        return {"kind": "function", "function": _function_document(value, seen)}
    if inspect.isbuiltin(value):
        return {
            "kind": "builtin",
            "module": getattr(value, "__module__", None),
            "qualname": getattr(value, "__qualname__", getattr(value, "__name__", None)),
        }
    call = inspect.getattr_static(type(value), "__call__", None)
    if inspect.isfunction(call):
        return {
            "kind": "callable_instance",
            "owner_type": _type_name(value),
            "function": _function_document(call, seen),
        }
    return {"kind": "callable", "owner_type": _type_name(value)}


def _global_dependency_document(value: object, seen: set[int]) -> object | None:
    literal = _literal_document(value)
    if literal is not None:
        return {"kind": "constant", "document": literal}
    if inspect.isfunction(value):
        return {"kind": "function", "document": _function_document(value, seen)}
    if inspect.isbuiltin(value):
        return {"kind": "callable", "document": _callable_document(value, seen)}
    if isinstance(value, ModuleType):
        return {"kind": "module", "name": value.__name__}
    if isinstance(value, type):
        return {"kind": "type", "name": _type_name(value)}
    return None


def _policy_instance_items(owner: object) -> tuple[tuple[str, object], ...]:
    values: dict[str, object] = {}
    instance_dict = getattr(owner, "__dict__", None)
    if type(instance_dict) is dict:
        values.update(instance_dict)
    for implementation_type in type(owner).__mro__:
        slots = implementation_type.__dict__.get("__slots__", ())
        if type(slots) is str:
            slots = (slots,)
        for name in slots:
            if name in {"__dict__", "__weakref__"}:
                continue
            try:
                values[name] = object.__getattribute__(owner, name)
            except AttributeError:
                continue
    # Policy self-description is never authoritative; registration metadata is.
    values.pop("policy_name", None)
    values.pop("policy_version", None)
    return tuple(sorted(values.items()))


def _instance_value_document(name: str, value: object, seen: set[int]) -> object:
    literal = _literal_document(value)
    if literal is not None:
        return {"kind": "constant", "document": literal}
    if callable(value):
        return {"kind": "callable", "document": _callable_document(value, seen)}
    if name == "_client":
        complete = getattr(value, "complete", None)
        if not callable(complete):
            raise TypeError("registered LLM client must retain complete")
        return {
            "kind": "pinned_client",
            "type": _type_name(value),
            "provider": _literal_document(getattr(value, "provider", None)),
            "model_id": _literal_document(getattr(value, "model_id", None)),
            "complete": _callable_document(complete, seen),
        }
    if type(value) in {dict, list, set}:
        return {"kind": "mutable_state", "type": _type_name(value)}
    return {"kind": "instance_dependency", "type": _type_name(value)}


def _policy_runtime_document(
    owner: object,
    entrypoint: Callable[..., object],
    *,
    name: str,
    version: str,
) -> dict[str, object]:
    if not inspect.ismethod(entrypoint) or entrypoint.__self__ is not owner:
        raise TypeError("registered policy entrypoint must stay bound to its exact owner")
    seen: set[int] = set()
    class_records: list[dict[str, object]] = []
    for implementation_type in reversed(type(owner).__mro__):
        if implementation_type is object:
            continue
        member_records: dict[str, object] = {}
        for member_name, member in sorted(implementation_type.__dict__.items()):
            functions: tuple[Callable[..., object], ...] = ()
            if inspect.isfunction(member):
                functions = (member,)
            elif isinstance(member, (staticmethod, classmethod)):
                functions = (member.__func__,)
            elif isinstance(member, property):
                functions = tuple(
                    function
                    for function in (member.fget, member.fset, member.fdel)
                    if function is not None
                )
            if functions:
                member_records[member_name] = [
                    _function_document(function, seen) for function in functions
                ]
                continue
            literal = _literal_document(member)
            if literal is not None and not member_name.startswith("__"):
                member_records[member_name] = {"class_constant": literal}
        class_records.append(
            {
                "type": _type_name(implementation_type),
                "members": member_records,
            }
        )
    instance_records = {
        item_name: _instance_value_document(item_name, value, seen)
        for item_name, value in _policy_instance_items(owner)
    }
    return {
        "registered_type": _type_name(owner),
        "registered_metadata": {"name": name, "version": version},
        "stored_entrypoint": _function_document(entrypoint.__func__, seen),
        "class_records": class_records,
        "instance_records": instance_records,
    }


def _policy_runtime_digest(
    owner: object,
    entrypoint: Callable[..., object],
    *,
    name: str,
    version: str,
) -> str:
    return _digest(
        _policy_runtime_document(owner, entrypoint, name=name, version=version)
    )


def _exact_text(label: str, value: object) -> str:
    if type(value) is not str or not value:
        raise TypeError(f"{label} must be an exact non-empty string")
    return value


def _exact_digest(label: str, value: object) -> str:
    text = _exact_text(label, value)
    if len(text) != 64 or not set(text) <= _HEX:
        raise ValueError(f"{label} must be lowercase SHA-256 hex")
    return text


def _exact_non_negative_int(
    label: str,
    value: object,
    *,
    maximum: int | None = None,
) -> int:
    if type(value) is not int or value < 0:
        raise TypeError(f"{label} must be an exact non-negative integer")
    if maximum is not None and value > maximum:
        raise ValueError(f"{label} exceeds {maximum}")
    return value


def _exact_decimal(label: str, value: object, *, non_negative: bool = False) -> Decimal:
    if type(value) is not Decimal:
        raise TypeError(f"{label} must be an exact Decimal")
    if not value.is_finite() or (non_negative and value < 0):
        raise ValueError(f"{label} must be finite")
    return value


def _set_seal(contract: ExternalContract, seal: str) -> None:
    private = contract.__pydantic_private__
    if private is None:
        raise RuntimeError("contract private storage is unavailable")
    private["_integrity_seal"] = seal


class DisclosureProfile(ExternalContract):
    """Immutable scenario-owned feedback disclosure configuration."""

    profile_id: str
    expose_realized_value: bool
    _integrity_seal: str = PrivateAttr(default="")

    @field_validator("profile_id", mode="before")
    @classmethod
    def profile_is_exact(cls, value: object) -> object:
        return _exact_text("profile_id", value)

    @field_validator("expose_realized_value", mode="before")
    @classmethod
    def exposure_is_exact(cls, value: object) -> object:
        if type(value) is not bool:
            raise TypeError("expose_realized_value must be an exact bool")
        return value

    def model_post_init(self, _context: object) -> None:
        _set_seal(self, self.profile_digest)

    @property
    def profile_digest(self) -> str:
        return _digest(
            {
                "profile_id": self.profile_id,
                "expose_realized_value": self.expose_realized_value,
            }
        )

    def assert_pristine(self) -> None:
        if type(self) is not DisclosureProfile:
            raise CandidateContractError("disclosure profile subclasses are forbidden")
        if self.__pydantic_extra__ or set(self.__dict__) != set(type(self).model_fields):
            raise CandidateContractError("disclosure profile field set is not exact")
        if self._integrity_seal != self.profile_digest:
            raise CandidateContractError("disclosure profile integrity seal changed")


def _reconstruct_disclosure(value: DisclosureProfile) -> DisclosureProfile:
    if type(value) is not DisclosureProfile:
        raise CandidateContractError("disclosure profile must be exact")
    value.assert_pristine()
    return DisclosureProfile(
        profile_id=value.profile_id,
        expose_realized_value=value.expose_realized_value,
    )


class EvaluationContract(ExternalContract):
    """Digest-only evaluator provenance, never passed to an attacker policy."""

    family: str
    bounds_digest: str
    hidden_template_digest: str
    background_digest: str
    population_digest: str
    evaluator_digest: str
    defender_digest: str
    disclosure_profile: DisclosureProfile
    _integrity_seal: str = PrivateAttr(default="")

    @field_validator("family", mode="before")
    @classmethod
    def family_is_exact(cls, value: object) -> object:
        return _exact_text("family", value)

    @field_validator(
        "bounds_digest",
        "hidden_template_digest",
        "background_digest",
        "population_digest",
        "evaluator_digest",
        "defender_digest",
        mode="before",
    )
    @classmethod
    def digests_are_exact(cls, value: object) -> object:
        return _exact_digest("evaluation provenance digest", value)

    @field_validator("disclosure_profile", mode="before")
    @classmethod
    def disclosure_is_exact(cls, value: object) -> object:
        if type(value) is not DisclosureProfile:
            raise TypeError("disclosure_profile must be exact")
        value.assert_pristine()
        return value

    def model_post_init(self, _context: object) -> None:
        _set_seal(self, self.contract_digest)

    @property
    def disclosure_profile_digest(self) -> str:
        return self.disclosure_profile.profile_digest

    @property
    def contract_digest(self) -> str:
        return _digest(
            {
                "family": self.family,
                "bounds_digest": self.bounds_digest,
                "hidden_template_digest": self.hidden_template_digest,
                "background_digest": self.background_digest,
                "population_digest": self.population_digest,
                "evaluator_digest": self.evaluator_digest,
                "defender_digest": self.defender_digest,
                "disclosure_profile_digest": self.disclosure_profile_digest,
            }
        )

    def assert_pristine(self) -> None:
        if type(self) is not EvaluationContract:
            raise CandidateContractError("evaluation contract subclasses are forbidden")
        if self.__pydantic_extra__ or set(self.__dict__) != set(type(self).model_fields):
            raise CandidateContractError("evaluation contract field set is not exact")
        self.disclosure_profile.assert_pristine()
        if self._integrity_seal != self.contract_digest:
            raise CandidateContractError("evaluation contract integrity seal changed")


def reconstruct_evaluation_contract(value: EvaluationContract) -> EvaluationContract:
    if type(value) is not EvaluationContract:
        raise CandidateContractError("evaluation contract must be exact")
    value.assert_pristine()
    return EvaluationContract(
        family=value.family,
        bounds_digest=value.bounds_digest,
        hidden_template_digest=value.hidden_template_digest,
        background_digest=value.background_digest,
        population_digest=value.population_digest,
        evaluator_digest=value.evaluator_digest,
        defender_digest=value.defender_digest,
        disclosure_profile=_reconstruct_disclosure(value.disclosure_profile),
    )


@dataclass(frozen=True, slots=True)
class EvaluatorCapability:
    """Exact process-local authority for one evaluator implementation and contract."""

    authority_id: str
    capability_id: str
    evaluation_contract: EvaluationContract
    bounds: ParameterBounds
    evaluator_code_digest: str
    _authority: SearchAuthority = field(repr=False, compare=False)
    _owner: object = field(repr=False, compare=False)
    _evaluate: Callable[[AttackCandidate], Feedback] = field(repr=False, compare=False)


@dataclass(frozen=True, slots=True)
class PolicyCapability:
    """Minimal opaque process-local handle; executable bindings remain authority-private."""

    capability_id: str


@dataclass(frozen=True, slots=True)
class _PolicyRuntimeBinding:
    capability: PolicyCapability
    name: str
    version: str
    policy_code_digest: str
    policy_callable_digest: str
    registered_type: type[object]
    instance_items: tuple[tuple[str, object], ...] = field(repr=False, compare=False)
    policy: Policy = field(repr=False, compare=False)
    propose: Callable[
        [tuple[VisibleTrial, ...], ParameterBounds, np.random.Generator],
        AttackCandidate,
    ] = field(repr=False, compare=False)


@dataclass(frozen=True, slots=True)
class RunGroupCapability:
    """Process-local comparison group preventing cross-run result mixing."""

    authority_id: str
    capability_id: str
    label: str
    _authority: SearchAuthority = field(repr=False, compare=False)


class SearchResult(ExternalContract):
    """Complete result bound to exact policy, environment, disclosure, and budgets."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        revalidate_instances="always",
        arbitrary_types_allowed=True,
    )

    family: str
    bounds_digest: str
    hidden_template_digest: str
    background_digest: str
    population_digest: str
    evaluator_digest: str
    defender_digest: str
    disclosure_profile_digest: str
    evaluation_contract_digest: str
    authority_id: str
    evaluator_capability_id: str
    evaluator_code_digest: str
    policy_capability_id: str
    policy_name: str
    policy_version: str
    policy_code_digest: str
    policy_callable_digest: str
    run_group_id: str
    result_id: str
    result_seal: str
    seed: int
    proposals: tuple[AttackCandidate, ...]
    trials: tuple[VisibleTrial, ...]
    objective_values: tuple[Decimal, ...]
    winner: AttackCandidate | None
    proposal_budget: int
    query_budget: int
    logical_time_budget: int
    wall_time_budget_ms: int
    proposals_used: int
    queries_used: int
    logical_time_used: int
    wall_time_elapsed_ms: int
    wall_time_exhausted: bool
    wall_time_overrun_ms: int

    @field_validator("family", "policy_name", "policy_version", mode="before")
    @classmethod
    def text_is_exact(cls, value: object) -> object:
        return _exact_text("result text", value)

    @field_validator(
        "bounds_digest",
        "hidden_template_digest",
        "background_digest",
        "population_digest",
        "evaluator_digest",
        "defender_digest",
        "disclosure_profile_digest",
        "evaluation_contract_digest",
        "authority_id",
        "evaluator_capability_id",
        "evaluator_code_digest",
        "policy_capability_id",
        "policy_code_digest",
        "policy_callable_digest",
        "run_group_id",
        "result_id",
        "result_seal",
        mode="before",
    )
    @classmethod
    def result_digests_are_exact(cls, value: object) -> object:
        return _exact_digest("result provenance digest", value)

    @field_validator("seed", mode="before")
    @classmethod
    def seed_is_exact(cls, value: object) -> object:
        return _exact_non_negative_int("seed", value, maximum=2**63 - 1)

    @field_validator("proposals", mode="before")
    @classmethod
    def proposals_are_exact(cls, value: object) -> object:
        if type(value) is not tuple or any(type(item) is not AttackCandidate for item in value):
            raise TypeError("proposals must be an exact tuple of candidates")
        return value

    @field_validator("trials", mode="before")
    @classmethod
    def trials_are_exact(cls, value: object) -> object:
        if type(value) is not tuple or any(type(item) is not VisibleTrial for item in value):
            raise TypeError("trials must be an exact tuple of visible trials")
        return value

    @field_validator("objective_values", mode="before")
    @classmethod
    def objectives_are_exact(cls, value: object) -> object:
        if type(value) is not tuple or any(
            type(item) is not Decimal or not item.is_finite() for item in value
        ):
            raise TypeError("objective values must be exact finite Decimals")
        return value

    @field_validator("winner", mode="before")
    @classmethod
    def winner_is_exact(cls, value: object) -> object:
        if value is not None and type(value) is not AttackCandidate:
            raise TypeError("winner must be an exact candidate or None")
        return value

    @field_validator(
        "proposal_budget",
        "query_budget",
        "logical_time_budget",
        "proposals_used",
        "queries_used",
        "logical_time_used",
        mode="before",
    )
    @classmethod
    def search_counts_are_exact(cls, value: object) -> object:
        return _exact_non_negative_int("search count", value, maximum=1000)

    @field_validator(
        "wall_time_budget_ms",
        "wall_time_elapsed_ms",
        "wall_time_overrun_ms",
        mode="before",
    )
    @classmethod
    def wall_counts_are_exact(cls, value: object) -> object:
        return _exact_non_negative_int("wall time", value, maximum=3_600_000)

    @field_validator("wall_time_exhausted", mode="before")
    @classmethod
    def exhausted_is_exact(cls, value: object) -> object:
        if type(value) is not bool:
            raise TypeError("wall_time_exhausted must be an exact bool")
        return value

    @model_validator(mode="after")
    def result_is_consistent(self) -> Self:
        self._assert_consistent()
        return self

    def _assert_consistent(self) -> None:
        size = len(self.proposals)
        if len(self.trials) != size or len(self.objective_values) != size:
            raise ValueError("proposal, trial, and objective sequences must align")
        if tuple(trial.candidate for trial in self.trials) != self.proposals:
            raise ValueError("trial candidates must preserve proposal order")
        if tuple(trial.objective_value for trial in self.trials) != self.objective_values:
            raise ValueError("trial objectives must preserve objective order")
        if not (
            self.proposal_budget == self.query_budget == self.logical_time_budget
            and self.proposals_used == self.queries_used == self.logical_time_used == size
            and size <= self.proposal_budget
        ):
            raise ValueError("proposal, query, and logical-time accounting must match")
        if not self.wall_time_exhausted and size != self.proposal_budget:
            raise ValueError("a non-exhausted search must use its complete discrete budget")
        expected_overrun = max(0, self.wall_time_elapsed_ms - self.wall_time_budget_ms)
        if self.wall_time_overrun_ms != expected_overrun:
            raise ValueError("wall time overrun must be derived from elapsed time")
        if (size == 0) != (self.winner is None):
            raise ValueError("winner presence must match non-empty proposals")
        if self.winner is not None and self.winner not in self.proposals:
            raise ValueError("winner must be a proposed candidate")
        reconstruct_history(self.trials)

    def canonical_document(self) -> dict[str, object]:
        """Return every result-visible field except the process-local HMAC itself."""
        return self.model_dump(mode="json", round_trip=True, exclude={"result_seal"})

    def assert_deep_pristine(self) -> None:
        if type(self) is not SearchResult:
            raise CandidateContractError("search result subclasses are forbidden")
        if self.__pydantic_extra__ or set(self.__dict__) != set(type(self).model_fields):
            raise CandidateContractError("search result field set is not exact")
        for candidate in self.proposals:
            candidate.assert_pristine()
        for trial in self.trials:
            trial.assert_pristine()
        if self.winner is not None:
            self.winner.assert_pristine()
        self._assert_consistent()


class SearchAuthority:
    """Trusted process-local issuer for executable capabilities and authentic results."""

    __slots__ = (
        "_authority_id",
        "_counter",
        "_evaluators",
        "_policies",
        "_preregistrations",
        "_results",
        "_run_groups",
        "_secret",
    )

    def __init__(self) -> None:
        secret = secrets.token_bytes(32)
        self._secret = secret
        self._authority_id = hashlib.sha256(b"apar-search-authority-v1" + secret).hexdigest()
        self._counter = 0
        self._evaluators: dict[str, EvaluatorCapability] = {}
        self._policies: dict[str, _PolicyRuntimeBinding] = {}
        self._run_groups: dict[str, RunGroupCapability] = {}
        self._results: dict[str, SearchResult] = {}
        self._preregistrations: dict[str, CapabilityPreregistration] = {}

    @property
    def authority_id(self) -> str:
        return self._authority_id

    def _issue_id(self, label: str) -> str:
        self._counter += 1
        return hmac.new(
            self._secret,
            f"{label}:{self._counter}".encode(),
            hashlib.sha256,
        ).hexdigest()

    def _seal(self, label: str, document: object) -> str:
        return hmac.new(
            self._secret,
            label.encode() + b":" + _canonical_bytes(document),
            hashlib.sha256,
        ).hexdigest()

    def register_evaluator(
        self,
        *,
        owner: object,
        bounds: ParameterBounds,
        evaluation_contract: EvaluationContract,
        evaluate: Callable[[AttackCandidate], Feedback],
        dependency_digest: str,
    ) -> EvaluatorCapability:
        public_bounds = reconstruct_bounds(bounds)
        contract = reconstruct_evaluation_contract(evaluation_contract)
        if contract.family != public_bounds.family:
            raise ValueError("evaluation contract family does not match public bounds")
        if contract.bounds_digest != public_bounds.bounds_digest:
            raise ValueError("evaluation contract does not bind these public bounds")
        checked_dependency = _exact_digest("dependency_digest", dependency_digest)
        code_digest = _digest(
            {
                "owner_implementation": _implementation_digest(owner, evaluate),
                "dependency_digest": checked_dependency,
            }
        )
        capability = EvaluatorCapability(
            authority_id=self._authority_id,
            capability_id=self._issue_id("evaluator"),
            evaluation_contract=contract,
            bounds=public_bounds,
            evaluator_code_digest=code_digest,
            _authority=self,
            _owner=owner,
            _evaluate=evaluate,
        )
        self._evaluators[capability.capability_id] = capability
        return capability

    def register_policy(
        self,
        policy: Policy,
        *,
        name: str,
        version: str,
    ) -> PolicyCapability:
        checked_name = _exact_text("registered policy name", name)
        checked_version = _exact_text("registered policy version", version)
        propose = getattr(policy, "propose", None)
        if not callable(propose):
            raise TypeError("registered policy must expose propose")
        capability = PolicyCapability(
            capability_id=self._issue_id("policy"),
        )
        binding = _PolicyRuntimeBinding(
            capability=capability,
            name=checked_name,
            version=checked_version,
            policy_code_digest=_policy_runtime_digest(
                policy,
                propose,
                name=checked_name,
                version=checked_version,
            ),
            policy_callable_digest=_bound_callable_digest(policy, propose),
            registered_type=type(policy),
            instance_items=_policy_instance_items(policy),
            policy=policy,
            propose=propose,
        )
        self._policies[capability.capability_id] = binding
        return capability

    def issue_run_group(self, label: str) -> RunGroupCapability:
        capability = RunGroupCapability(
            authority_id=self._authority_id,
            capability_id=self._issue_id("run-group"),
            label=_exact_text("run group label", label),
            _authority=self,
        )
        self._run_groups[capability.capability_id] = capability
        return capability

    def run_group(self, capability_id: str) -> RunGroupCapability:
        checked = _exact_digest("run_group_id", capability_id)
        try:
            return self._run_groups[checked]
        except KeyError as error:
            raise ValueError("run group was not issued by this authority") from error

    def evaluator_capability(self, capability_id: str) -> EvaluatorCapability:
        checked = _exact_digest("evaluator_capability_id", capability_id)
        try:
            return self._evaluators[checked]
        except KeyError as error:
            raise ValueError("evaluator capability was not issued by this authority") from error

    def policy_capability(self, capability_id: str) -> PolicyCapability:
        checked = _exact_digest("policy_capability_id", capability_id)
        try:
            return self._policies[checked].capability
        except KeyError as error:
            raise ValueError("policy capability was not issued by this authority") from error

    def policy_binding(self, capability: PolicyCapability) -> PolicyBinding:
        """Return immutable public provenance derived only from a private binding."""
        binding = self._validate_policy(capability)
        return PolicyBinding(
            name=binding.name,
            version=binding.version,
            capability_id=binding.capability.capability_id,
            code_digest=binding.policy_code_digest,
            callable_digest=binding.policy_callable_digest,
        )

    def _validate_evaluator(self, capability: EvaluatorCapability) -> EvaluatorCapability:
        if (
            type(capability) is not EvaluatorCapability
            or capability._authority is not self
            or capability.authority_id != self._authority_id
            or self._evaluators.get(capability.capability_id) is not capability
        ):
            raise ValueError("evaluator capability was not issued by this exact authority")
        return capability

    def _validate_policy(self, capability: PolicyCapability) -> _PolicyRuntimeBinding:
        if type(capability) is not PolicyCapability:
            raise ValueError("policy capability was not issued by this exact authority")
        binding = self._policies.get(capability.capability_id)
        if binding is None or binding.capability is not capability:
            raise ValueError("policy capability was not issued by this exact authority")
        if type(binding.policy) is not binding.registered_type:
            raise ValueError("policy runtime implementation type changed")
        current_items = _policy_instance_items(binding.policy)
        if tuple(name for name, _value in current_items) != tuple(
            name for name, _value in binding.instance_items
        ):
            raise ValueError("policy runtime instance integrity changed")
        for (name, expected), (_current_name, observed) in zip(
            binding.instance_items,
            current_items,
            strict=True,
        ):
            literal = _literal_document(expected)
            unchanged = (
                type(observed) is type(expected) and observed == expected
                if literal is not None
                else observed is expected
            )
            if not unchanged:
                raise ValueError(f"policy runtime instance integrity changed: {name}")
        try:
            observed_callable_digest = _bound_callable_digest(
                binding.policy,
                binding.propose,
            )
            observed_runtime_digest = _policy_runtime_digest(
                binding.policy,
                binding.propose,
                name=binding.name,
                version=binding.version,
            )
        except (TypeError, AttributeError, ValueError) as error:
            raise ValueError("policy callable binding is invalid") from error
        if not hmac.compare_digest(
            binding.policy_callable_digest,
            observed_callable_digest,
        ):
            raise ValueError("policy callable implementation digest changed")
        if not hmac.compare_digest(
            binding.policy_code_digest,
            observed_runtime_digest,
        ):
            raise ValueError("policy runtime implementation integrity changed")
        return binding

    def _validate_run_group(self, capability: RunGroupCapability) -> RunGroupCapability:
        if (
            type(capability) is not RunGroupCapability
            or capability._authority is not self
            or capability.authority_id != self._authority_id
            or self._run_groups.get(capability.capability_id) is not capability
        ):
            raise ValueError("run group was not issued by this exact authority")
        return capability

    def validate_search_capabilities(
        self,
        evaluator: EvaluatorCapability,
        policy: PolicyCapability,
        run_group: RunGroupCapability,
    ) -> tuple[EvaluatorCapability, PolicyCapability, RunGroupCapability]:
        checked_evaluator = self._validate_evaluator(evaluator)
        self._validate_policy(policy)
        checked_group = self._validate_run_group(run_group)
        return checked_evaluator, policy, checked_group

    def propose(
        self,
        capability: PolicyCapability,
        history: tuple[VisibleTrial, ...],
        bounds: ParameterBounds,
        rng: np.random.Generator,
    ) -> AttackCandidate:
        binding = self._validate_policy(capability)
        return binding.propose(history, bounds, rng)

    def evaluate(
        self,
        capability: EvaluatorCapability,
        candidate: AttackCandidate,
    ) -> Feedback:
        checked = self._validate_evaluator(capability)
        return checked._evaluate(candidate)

    def issue_result(self, **values: object) -> SearchResult:
        result_id = self._issue_id("result")
        result = SearchResult.model_validate(
            {
                **values,
                "authority_id": self._authority_id,
                "result_id": result_id,
                "result_seal": "0" * 64,
            }
        )
        seal = self._seal("search-result-v1", result.canonical_document())
        object.__setattr__(result, "result_seal", seal)
        self._results[result_id] = result
        return result

    def validate_result(self, result: SearchResult) -> SearchResult:
        if (
            type(result) is not SearchResult
            or result.authority_id != self._authority_id
            or self._results.get(result.result_id) is not result
        ):
            raise ValueError("search result was not issued by this exact authority")
        try:
            result.assert_deep_pristine()
            expected = self._seal("search-result-v1", result.canonical_document())
        except Exception as error:
            raise ValueError("search result is not deeply pristine") from error
        if not hmac.compare_digest(result.result_seal, expected):
            raise ValueError("search result authenticity seal does not match")
        return result

    def issue_preregistration(
        self,
        *,
        run_group: RunGroupCapability,
        seeds: tuple[int, ...],
        budget: int,
        wall_time_budget_ms: int,
        thresholds: tuple[FamilyThreshold, ...],
        policies: tuple[PolicyCapability, ...],
    ) -> CapabilityPreregistration:
        group = self._validate_run_group(run_group)
        if type(thresholds) is not tuple:
            raise TypeError("thresholds must be an exact tuple")
        checked_thresholds: list[FamilyThreshold] = []
        for threshold in thresholds:
            if type(threshold) is not FamilyThreshold:
                raise TypeError("thresholds must contain exact FamilyThreshold records")
            evaluator = self._evaluators.get(threshold.evaluator_capability_id)
            if evaluator is None or evaluator.evaluator_code_digest != (
                threshold.evaluator_code_digest
            ):
                raise ValueError("threshold evaluator capability was not issued by this authority")
            if evaluator.evaluation_contract.contract_digest != (
                threshold.evaluation_contract.contract_digest
            ):
                raise ValueError("threshold evaluator contract does not match capability")
            checked_thresholds.append(
                FamilyThreshold(
                    family=threshold.family,
                    primary_outcome=threshold.primary_outcome,
                    minimum_delta=threshold.minimum_delta,
                    evaluation_contract=threshold.evaluation_contract,
                    evaluator_capability_id=threshold.evaluator_capability_id,
                    evaluator_code_digest=threshold.evaluator_code_digest,
                )
            )
        if type(policies) is not tuple:
            raise TypeError("policies must be an exact tuple")
        checked_policies = tuple(self._validate_policy(policy) for policy in policies)
        bindings = tuple(
            sorted(
                (
                    PolicyBinding(
                        name=policy.name,
                        version=policy.version,
                        capability_id=policy.capability.capability_id,
                        code_digest=policy.policy_code_digest,
                        callable_digest=policy.policy_callable_digest,
                    )
                    for policy in checked_policies
                ),
                key=lambda binding: binding.name,
            )
        )
        issuance_id = self._issue_id("preregistration")
        preregistration = CapabilityPreregistration(
            authority_id=self._authority_id,
            run_group_id=group.capability_id,
            issuance_id=issuance_id,
            issuance_seal="0" * 64,
            seeds=seeds,
            budget=budget,
            wall_time_budget_ms=wall_time_budget_ms,
            thresholds=tuple(checked_thresholds),
            policy_bindings=bindings,
        )
        seal = self._seal("capability-preregistration-v1", preregistration.canonical_document())
        object.__setattr__(preregistration, "issuance_seal", seal)
        self._preregistrations[issuance_id] = preregistration
        return preregistration

    def validate_preregistration(
        self,
        preregistration: CapabilityPreregistration,
    ) -> CapabilityPreregistration:
        if (
            type(preregistration) is not CapabilityPreregistration
            or preregistration.authority_id != self._authority_id
            or self._preregistrations.get(preregistration.issuance_id) is not preregistration
        ):
            raise ValueError("preregistration was not issued by this exact authority")
        expected = self._seal(
            "capability-preregistration-v1",
            preregistration.canonical_document(),
        )
        if not hmac.compare_digest(preregistration.issuance_seal, expected):
            raise ValueError("preregistration authenticity seal does not match")
        return preregistration


class AdaptiveSearch:
    """Run only exact evaluator and policy capabilities issued by one authority."""

    __slots__ = (
        "_authority",
        "_bounds",
        "_clock_ns",
        "_evaluator",
        "_policy",
        "_run_group",
    )

    def __init__(
        self,
        *,
        evaluator_capability: EvaluatorCapability,
        policy_capability: PolicyCapability,
        run_group: RunGroupCapability,
        clock_ns: Callable[[], int] = time.monotonic_ns,
    ) -> None:
        if type(evaluator_capability) is not EvaluatorCapability:
            raise TypeError("evaluator_capability must be exact")
        authority = evaluator_capability._authority
        evaluator, policy, group = authority.validate_search_capabilities(
            evaluator_capability,
            policy_capability,
            run_group,
        )
        if not callable(clock_ns):
            raise TypeError("clock_ns must be callable")
        self._authority = authority
        self._evaluator = evaluator
        self._policy = policy
        self._run_group = group
        self._bounds = evaluator.bounds
        self._clock_ns = clock_ns

    def _clock(self, previous: int | None = None) -> int:
        value = self._clock_ns()
        if type(value) is not int or value < 0:
            raise TypeError("monotonic clock must return an exact non-negative integer")
        if previous is not None and value < previous:
            raise RuntimeError("monotonic clock moved backwards")
        return value

    def search(
        self,
        seed: int,
        budget: int,
        wall_time_budget_ms: int,
    ) -> SearchResult:
        checked_seed = _exact_non_negative_int("seed", seed, maximum=2**63 - 1)
        checked_budget = _exact_non_negative_int("budget", budget, maximum=1000)
        checked_wall = _exact_non_negative_int(
            "wall_time_budget_ms", wall_time_budget_ms, maximum=3_600_000
        )
        rng = np.random.default_rng(checked_seed)
        proposals: list[AttackCandidate] = []
        trials: list[VisibleTrial] = []
        objectives: list[Decimal] = []
        start = self._clock()
        latest = start
        deadline = start + checked_wall * 1_000_000
        exhausted = False
        for _ in range(checked_budget):
            latest = self._clock(latest)
            if latest >= deadline:
                exhausted = True
                break
            visible_history = normalize_internal_history(tuple(trials))
            public_bounds = self._bounds
            candidate = self._authority.propose(
                self._policy,
                visible_history,
                public_bounds,
                rng,
            )
            latest = self._clock(latest)
            if latest >= deadline:
                exhausted = True
                break
            checked_candidate = validate_candidate_lineage(candidate, visible_history)
            public_bounds.validate_vector(checked_candidate.params)
            expose_value = (
                self._evaluator.evaluation_contract.disclosure_profile.expose_realized_value
            )
            try:
                returned = reconstruct_feedback(
                    self._authority.evaluate(self._evaluator, checked_candidate)
                )
                feedback = Feedback(
                    action=returned.action,
                    reason_family=returned.reason_family,
                    realized_value=(
                        returned.realized_value
                        if expose_value else None
                    ),
                )
            except Exception:
                feedback = Feedback(
                    action=Action.DECLINE,
                    reason_family="evaluation_failure",
                    realized_value=None,
                )
            objective = visible_objective(feedback)
            proposal = reconstruct_candidate(checked_candidate)
            trial = VisibleTrial(
                candidate=proposal,
                feedback=feedback,
                objective_value=objective,
            )
            proposals.append(proposal)
            trials.append(trial)
            objectives.append(objective)
            latest = self._clock(latest)
            if latest >= deadline and len(proposals) < checked_budget:
                exhausted = True
                break
        end = self._clock(latest)
        elapsed_ns = end - start
        elapsed_ms = elapsed_ns // 1_000_000
        if elapsed_ns > checked_wall * 1_000_000 and elapsed_ms == checked_wall:
            elapsed_ms += 1
        exhausted = exhausted or end >= deadline
        winner = (
            None
            if not trials
            else min(
                trials,
                key=lambda trial: (-trial.objective_value, trial.candidate.candidate_id),
            ).candidate
        )
        contract = self._evaluator.evaluation_contract
        policy_binding = self._authority.policy_binding(self._policy)
        return self._authority.issue_result(
            family=contract.family,
            bounds_digest=contract.bounds_digest,
            hidden_template_digest=contract.hidden_template_digest,
            background_digest=contract.background_digest,
            population_digest=contract.population_digest,
            evaluator_digest=contract.evaluator_digest,
            defender_digest=contract.defender_digest,
            disclosure_profile_digest=contract.disclosure_profile_digest,
            evaluation_contract_digest=contract.contract_digest,
            evaluator_capability_id=self._evaluator.capability_id,
            evaluator_code_digest=self._evaluator.evaluator_code_digest,
            policy_capability_id=policy_binding.capability_id,
            policy_name=policy_binding.name,
            policy_version=policy_binding.version,
            policy_code_digest=policy_binding.code_digest,
            policy_callable_digest=policy_binding.callable_digest,
            run_group_id=self._run_group.capability_id,
            seed=checked_seed,
            proposals=tuple(proposals),
            trials=tuple(trials),
            objective_values=tuple(objectives),
            winner=winner,
            proposal_budget=checked_budget,
            query_budget=checked_budget,
            logical_time_budget=checked_budget,
            wall_time_budget_ms=checked_wall,
            proposals_used=len(proposals),
            queries_used=len(proposals),
            logical_time_used=len(proposals),
            wall_time_elapsed_ms=elapsed_ms,
            wall_time_exhausted=exhausted,
            wall_time_overrun_ms=max(0, elapsed_ms - checked_wall),
        )


class PrimaryOutcome(StrEnum):
    VALID_YIELD = "valid_yield"
    NET_SETTLED_VALUE = "net_settled_value"
    NET_SETTLED_VALUE_RATE = "net_settled_value_rate"
    ADAPTATION_SPEED = "adaptation_speed"
    CAMPAIGN_SCALE = "campaign_scale"


class FamilyThreshold(ExternalContract):
    """Preregistered family outcome, threshold, and exact evaluator provenance."""

    family: str
    primary_outcome: PrimaryOutcome
    minimum_delta: Decimal
    evaluation_contract: EvaluationContract
    evaluator_capability_id: str
    evaluator_code_digest: str

    @field_validator("family", mode="before")
    @classmethod
    def family_is_exact(cls, value: object) -> object:
        return _exact_text("family", value)

    @field_validator("primary_outcome", mode="before")
    @classmethod
    def outcome_is_exact(cls, value: object) -> object:
        if type(value) is not PrimaryOutcome:
            raise TypeError("primary_outcome must be exact")
        return value

    @field_validator("minimum_delta", mode="before")
    @classmethod
    def delta_is_exact(cls, value: object) -> object:
        checked = _exact_decimal("minimum_delta", value, non_negative=True)
        if checked <= 0:
            raise ValueError("minimum_delta must be strictly positive")
        return checked

    @field_validator("evaluation_contract", mode="before")
    @classmethod
    def contract_is_exact(cls, value: object) -> object:
        if type(value) is not EvaluationContract:
            raise TypeError("evaluation_contract must be exact")
        return reconstruct_evaluation_contract(value)

    @field_validator("evaluator_capability_id", "evaluator_code_digest", mode="before")
    @classmethod
    def evaluator_binding_is_exact(cls, value: object) -> object:
        return _exact_digest("threshold evaluator binding", value)

    @model_validator(mode="after")
    def family_matches_contract(self) -> Self:
        if self.family != self.evaluation_contract.family:
            raise ValueError("threshold family must match evaluation contract")
        return self


class PolicyBinding(ExternalContract):
    """Preregistered exact policy capability and implementation identity."""

    name: str
    version: str
    capability_id: str
    code_digest: str
    callable_digest: str

    @field_validator("name", "version", mode="before")
    @classmethod
    def text_is_exact(cls, value: object) -> object:
        return _exact_text("policy binding text", value)

    @field_validator("capability_id", "code_digest", "callable_digest", mode="before")
    @classmethod
    def digests_are_exact(cls, value: object) -> object:
        return _exact_digest("policy binding digest", value)


class CapabilityPreregistration(ExternalContract):
    authority_id: str
    run_group_id: str
    issuance_id: str
    issuance_seal: str
    seeds: tuple[int, ...]
    budget: int
    wall_time_budget_ms: int
    thresholds: tuple[FamilyThreshold, ...]
    policy_bindings: tuple[PolicyBinding, ...]

    @field_validator(
        "authority_id",
        "run_group_id",
        "issuance_id",
        "issuance_seal",
        mode="before",
    )
    @classmethod
    def issuance_digests_are_exact(cls, value: object) -> object:
        return _exact_digest("preregistration issuance digest", value)

    @field_validator("seeds", mode="before")
    @classmethod
    def seeds_are_exact(cls, value: object) -> object:
        if type(value) is not tuple or not value:
            raise TypeError("seeds must be a non-empty exact tuple")
        checked = tuple(_exact_non_negative_int("seed", item, maximum=2**63 - 1) for item in value)
        if checked != tuple(sorted(set(checked))):
            raise ValueError("seeds must be unique and sorted")
        return checked

    @field_validator("budget", mode="before")
    @classmethod
    def budget_is_exact(cls, value: object) -> object:
        return _exact_non_negative_int("budget", value, maximum=1000)

    @field_validator("wall_time_budget_ms", mode="before")
    @classmethod
    def wall_budget_is_exact(cls, value: object) -> object:
        return _exact_non_negative_int("wall time budget", value, maximum=3_600_000)

    @field_validator("thresholds", mode="before")
    @classmethod
    def thresholds_are_exact(cls, value: object) -> object:
        if (
            type(value) is not tuple
            or not value
            or any(type(item) is not FamilyThreshold for item in value)
        ):
            raise TypeError("thresholds must be a non-empty exact tuple")
        families = tuple(item.family for item in value)
        if families != tuple(sorted(set(families))):
            raise ValueError("threshold families must be unique and sorted")
        return value

    @field_validator("policy_bindings", mode="before")
    @classmethod
    def policy_bindings_are_exact(cls, value: object) -> object:
        if (
            type(value) is not tuple
            or any(type(item) is not PolicyBinding for item in value)
        ):
            raise TypeError("policy_bindings must be an exact tuple")
        names = tuple(item.name for item in value)
        if names != _POLICY_NAMES:
            raise ValueError("policy bindings must contain exact comparison policies")
        if len({item.capability_id for item in value}) != len(value):
            raise ValueError("policy binding capabilities must be unique")
        return value

    def canonical_document(self) -> dict[str, object]:
        return self.model_dump(mode="json", round_trip=True, exclude={"issuance_seal"})


class PolicyMetrics(ExternalContract):
    """Observed policy metrics with yield derived rather than caller supplied."""

    proposal_count: int
    approved_count: int
    net_settled_value: Decimal
    adaptation_speed: Decimal
    campaign_scale: int

    @field_validator("proposal_count", "approved_count", "campaign_scale", mode="before")
    @classmethod
    def counts_are_exact(cls, value: object) -> object:
        return _exact_non_negative_int("metric count", value)

    @field_validator("net_settled_value", "adaptation_speed", mode="before")
    @classmethod
    def decimals_are_exact(cls, value: object) -> object:
        return _exact_decimal("metric Decimal", value, non_negative=True)

    @model_validator(mode="after")
    def counts_are_consistent(self) -> Self:
        if self.approved_count > self.proposal_count:
            raise ValueError("approved_count cannot exceed proposal_count")
        if self.campaign_scale != self.approved_count:
            raise ValueError("campaign_scale must equal observed approved campaigns")
        return self

    @property
    def valid_yield(self) -> Decimal:
        if self.proposal_count == 0:
            return Decimal(0)
        with localcontext() as context:
            context.prec = 28
            return Decimal(self.approved_count) / Decimal(self.proposal_count)


def _observed_delta(
    outcome: PrimaryOutcome,
    adaptive: PolicyMetrics,
    random: PolicyMetrics,
) -> Decimal:
    if outcome is PrimaryOutcome.VALID_YIELD:
        return adaptive.valid_yield - random.valid_yield
    if outcome is PrimaryOutcome.NET_SETTLED_VALUE:
        return adaptive.net_settled_value - random.net_settled_value
    if outcome is PrimaryOutcome.NET_SETTLED_VALUE_RATE:
        if random.net_settled_value == 0:
            return Decimal(0) if adaptive.net_settled_value == 0 else Decimal(1)
        with localcontext() as context:
            context.prec = 28
            return (
                adaptive.net_settled_value - random.net_settled_value
            ) / random.net_settled_value
    if outcome is PrimaryOutcome.ADAPTATION_SPEED:
        return random.adaptation_speed - adaptive.adaptation_speed
    return Decimal(adaptive.campaign_scale - random.campaign_scale)


class FamilyCapabilityMetrics(ExternalContract):
    """Matched policy aggregates with delta and support derived internally."""

    family: str
    primary_outcome: PrimaryOutcome
    minimum_delta: Decimal
    evaluation_contract_digest: str
    fixed: PolicyMetrics
    random: PolicyMetrics
    adaptive: PolicyMetrics
    cached_llm: PolicyMetrics

    @field_validator("family", mode="before")
    @classmethod
    def family_is_exact(cls, value: object) -> object:
        return _exact_text("family", value)

    @field_validator("primary_outcome", mode="before")
    @classmethod
    def outcome_is_exact(cls, value: object) -> object:
        if type(value) is not PrimaryOutcome:
            raise TypeError("primary outcome must be exact")
        return value

    @field_validator("minimum_delta", mode="before")
    @classmethod
    def minimum_is_exact(cls, value: object) -> object:
        return _exact_decimal("minimum_delta", value, non_negative=True)

    @field_validator("evaluation_contract_digest", mode="before")
    @classmethod
    def contract_digest_is_exact(cls, value: object) -> object:
        return _exact_digest("evaluation contract digest", value)

    @field_validator("fixed", "random", "adaptive", "cached_llm", mode="before")
    @classmethod
    def metrics_are_exact(cls, value: object) -> object:
        if type(value) is not PolicyMetrics:
            raise TypeError("policy metrics must be exact")
        return value

    @property
    def observed_delta(self) -> Decimal:
        return _observed_delta(self.primary_outcome, self.adaptive, self.random)

    @property
    def supported(self) -> bool:
        return self.observed_delta >= self.minimum_delta


class CapabilityDeltaReport(ExternalContract):
    """Capability report whose counts and adaptive claim cannot be relabeled."""

    family_metrics: tuple[FamilyCapabilityMetrics, ...]
    matched_budgets: bool

    @field_validator("family_metrics", mode="before")
    @classmethod
    def metrics_are_exact(cls, value: object) -> object:
        if type(value) is not tuple or any(
            type(item) is not FamilyCapabilityMetrics for item in value
        ):
            raise TypeError("family_metrics must be an exact tuple")
        return value

    @field_validator("matched_budgets", mode="before")
    @classmethod
    def matched_is_true(cls, value: object) -> object:
        if value is not True or type(value) is not bool:
            raise ValueError("matched_budgets must be observed true")
        return value

    @property
    def supported_family_count(self) -> int:
        return sum(metric.supported for metric in self.family_metrics)

    @property
    def adaptive_net_value(self) -> Decimal:
        return sum(
            (metric.adaptive.net_settled_value for metric in self.family_metrics),
            Decimal(0),
        )

    @property
    def random_net_value(self) -> Decimal:
        return sum(
            (metric.random.net_settled_value for metric in self.family_metrics),
            Decimal(0),
        )

    @property
    def adaptive_claim(self) -> str:
        return "supported" if self.adaptive_net_value > self.random_net_value else "not_supported"


def _aggregate(results: tuple[SearchResult, ...]) -> PolicyMetrics:
    proposal_count = sum(len(result.trials) for result in results)
    approved = tuple(
        trial
        for result in results
        for trial in result.trials
        if trial.feedback.action is Action.APPROVE
    )
    net_value = sum(
        (trial.feedback.realized_value or Decimal(0) for trial in approved),
        Decimal(0),
    )
    first_successes = tuple(
        next(
            (
                index
                for index, trial in enumerate(result.trials, start=1)
                if trial.feedback.action is Action.APPROVE
            ),
            len(result.trials) + 1,
        )
        for result in results
    )
    with localcontext() as context:
        context.prec = 28
        speed = Decimal(sum(first_successes)) / Decimal(len(first_successes))
    return PolicyMetrics(
        proposal_count=proposal_count,
        approved_count=len(approved),
        net_settled_value=net_value,
        adaptation_speed=speed,
        campaign_scale=len(approved),
    )


def capability_delta_report(
    preregistration: CapabilityPreregistration,
    results: dict[str, dict[str, tuple[SearchResult, ...]]],
    *,
    authority: SearchAuthority,
) -> CapabilityDeltaReport:
    """Compute only preregistered outcomes from exact matched provenance cells."""
    if type(authority) is not SearchAuthority:
        raise TypeError("authority must be exact")
    authority.validate_preregistration(preregistration)
    if type(results) is not dict or any(type(key) is not str for key in results):
        raise TypeError("results must have exact string family keys")
    thresholds = {item.family: item for item in preregistration.thresholds}
    policy_bindings = {item.name: item for item in preregistration.policy_bindings}
    if set(results) != set(thresholds):
        raise ValueError("results must contain exactly preregistered families")
    metrics: list[FamilyCapabilityMetrics] = []
    disclosure_profiles: set[str] = set()
    for family in sorted(results):
        cells = results[family]
        if type(cells) is not dict or any(type(key) is not str for key in cells):
            raise TypeError("policy cells must have exact string keys")
        if tuple(sorted(cells)) != _POLICY_NAMES:
            raise ValueError("each family needs fixed, random, adaptive, and cached_llm")
        threshold = thresholds[family]
        contract = threshold.evaluation_contract
        aggregates: dict[str, PolicyMetrics] = {}
        for policy_name in _POLICY_NAMES:
            policy_binding = policy_bindings[policy_name]
            runs = cells[policy_name]
            if (
                type(runs) is not tuple
                or len(runs) != len(preregistration.seeds)
                or any(type(result) is not SearchResult for result in runs)
            ):
                raise TypeError("policy runs must match preregistered seeds")
            if tuple(result.seed for result in runs) != preregistration.seeds:
                raise ValueError("capability comparison seeds are not matched")
            for result in runs:
                authority.validate_result(result)
                if result.run_group_id != preregistration.run_group_id:
                    raise ValueError("capability result belongs to another run group")
                if result.policy_name != policy_name:
                    raise ValueError("capability result was relabeled under another policy")
                if (
                    result.policy_capability_id != policy_binding.capability_id
                    or result.policy_version != policy_binding.version
                    or result.policy_code_digest != policy_binding.code_digest
                    or result.policy_callable_digest != policy_binding.callable_digest
                ):
                    raise ValueError("capability result policy binding does not match")
                if result.family != family:
                    raise ValueError("capability result family was swapped")
                if (
                    result.evaluator_capability_id != threshold.evaluator_capability_id
                    or result.evaluator_code_digest != threshold.evaluator_code_digest
                ):
                    raise ValueError("capability result evaluator implementation does not match")
                result_provenance = (
                    result.bounds_digest,
                    result.hidden_template_digest,
                    result.background_digest,
                    result.population_digest,
                    result.evaluator_digest,
                    result.defender_digest,
                    result.disclosure_profile_digest,
                    result.evaluation_contract_digest,
                )
                contract_provenance = (
                    contract.bounds_digest,
                    contract.hidden_template_digest,
                    contract.background_digest,
                    contract.population_digest,
                    contract.evaluator_digest,
                    contract.defender_digest,
                    contract.disclosure_profile_digest,
                    contract.contract_digest,
                )
                if result_provenance != contract_provenance:
                    raise ValueError("capability result evaluator provenance does not match")
                if not (
                    result.proposal_budget
                    == result.query_budget
                    == result.logical_time_budget
                    == preregistration.budget
                    and result.wall_time_budget_ms == preregistration.wall_time_budget_ms
                ):
                    raise ValueError("capability comparison budgets are not matched")
                if not (
                    result.proposals_used
                    == result.queries_used
                    == result.logical_time_used
                    == preregistration.budget
                ):
                    raise ValueError("capability comparison actual usage is not matched")
                if result.wall_time_exhausted or result.wall_time_overrun_ms != 0:
                    raise ValueError("capability comparison contains an exhausted deadline")
                disclosure_profiles.add(result.disclosure_profile_digest)
            aggregates[policy_name] = _aggregate(runs)
        metrics.append(
            FamilyCapabilityMetrics(
                family=family,
                primary_outcome=threshold.primary_outcome,
                minimum_delta=threshold.minimum_delta,
                evaluation_contract_digest=contract.contract_digest,
                fixed=aggregates["fixed"],
                random=aggregates["random"],
                adaptive=aggregates["adaptive"],
                cached_llm=aggregates["cached_llm"],
            )
        )
    if len(disclosure_profiles) != 1:
        raise ValueError("capability comparison disclosure profiles are not comparable")
    return CapabilityDeltaReport(family_metrics=tuple(metrics), matched_budgets=True)


__all__ = [
    "AdaptiveSearch",
    "CapabilityDeltaReport",
    "CapabilityPreregistration",
    "DisclosureProfile",
    "EvaluationContract",
    "EvaluatorCapability",
    "FamilyCapabilityMetrics",
    "FamilyThreshold",
    "PolicyBinding",
    "PolicyCapability",
    "PolicyMetrics",
    "PrimaryOutcome",
    "RunGroupCapability",
    "SearchAuthority",
    "SearchResult",
    "capability_delta_report",
    "reconstruct_evaluation_contract",
]
