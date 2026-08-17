"""Strict optional LLM proposal transport with complete digest-only audit."""

from __future__ import annotations

import hashlib
import json
import math
import time
from collections.abc import Callable
from copy import deepcopy
from typing import Protocol

import numpy as np
from pydantic import field_validator, model_validator

from apar.contracts._validation import ExternalContract
from apar.redteam.policies import (
    AttackCandidate,
    CandidateContractError,
    ParameterBounds,
    VisibleTrial,
    reconstruct_bounds,
    reconstruct_history,
    validate_candidate_lineage,
)

_HEX = frozenset("0123456789abcdef")
_TRANSPORT_FIELDS = frozenset({"output", "latency_ms", "input_tokens", "output_tokens"})
_PLANNER_FIELDS = frozenset({"params", "parent_id", "generation"})
_CACHE_FIELDS = frozenset(
    {
        "transport",
        "response_digest",
        "provider",
        "model_id",
        "policy_version",
        "schema_digest",
    }
)
_PROTOTYPE_KEYS = frozenset({"__proto__", "constructor", "prototype"})
_CALL_STATUSES = frozenset({"online_success", "online_failure", "cache_success", "cache_failure"})
_FAILURE_FAMILIES = frozenset({"bounds", "cache", "lineage", "schema", "transport"})
_POLICY_VERSION = "1.0.0"


def _exact_text(label: str, value: object) -> str:
    if type(value) is not str or not value:
        raise TypeError(f"{label} must be an exact non-empty string")
    return value


def _count(label: str, value: object) -> int:
    if type(value) is not int or value < 0:
        raise TypeError(f"{label} must be an exact non-negative integer")
    return value


def _validate_json_tree(value: object, *, label: str) -> None:
    if value is None or type(value) in {bool, int, str}:
        return
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError(f"{label} contains a non-finite JSON number")
        return
    if type(value) is list:
        for item in value:
            _validate_json_tree(item, label=label)
        return
    if type(value) is dict:
        for key, item in value.items():
            if type(key) is not str:
                raise TypeError(f"{label} keys must be exact built-in strings")
            if key in _PROTOTYPE_KEYS:
                raise ValueError(f"{label} contains a prototype-like key")
            _validate_json_tree(item, label=label)
        return
    raise TypeError(f"{label} contains a non-strict JSON value")


def _canonical_bytes(document: object) -> bytes:
    _validate_json_tree(document, label="LLM document")
    return json.dumps(
        document,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode()


def _digest(document: object) -> str:
    return hashlib.sha256(_canonical_bytes(document)).hexdigest()


def _sentinel_digest(failure_family: str) -> str:
    return _digest({"unavailable_response": failure_family})


class LLMClient(Protocol):
    provider: str
    model_id: str

    def complete(self, request: dict[str, object]) -> dict[str, object]: ...


class LLMAuditRecord(ExternalContract):
    """One proposal-attempt audit containing attribution and digests, never bodies."""

    provider: str
    model_id: str
    policy_version: str
    schema_digest: str
    prompt_digest: str
    response_digest: str
    latency_ms: int
    input_tokens: int
    output_tokens: int
    call_status: str
    failure_family: str | None
    cache_hit: bool

    @field_validator("provider", "model_id", "policy_version", mode="before")
    @classmethod
    def identity_is_exact(cls, value: object) -> object:
        return _exact_text("LLM identity", value)

    @field_validator(
        "schema_digest",
        "prompt_digest",
        "response_digest",
        mode="before",
    )
    @classmethod
    def digest_is_exact(cls, value: object) -> object:
        text = _exact_text("LLM digest", value)
        if len(text) != 64 or not set(text) <= _HEX:
            raise ValueError("LLM digest must be lowercase SHA-256 hex")
        return text

    @field_validator("latency_ms", "input_tokens", "output_tokens", mode="before")
    @classmethod
    def counts_are_exact(cls, value: object) -> object:
        return _count("LLM audit count", value)

    @field_validator("call_status", mode="before")
    @classmethod
    def status_is_exact(cls, value: object) -> object:
        checked = _exact_text("call_status", value)
        if checked not in _CALL_STATUSES:
            raise ValueError("call_status is undeclared")
        return checked

    @field_validator("failure_family", mode="before")
    @classmethod
    def failure_is_coarse(cls, value: object) -> object:
        if value is None:
            return value
        checked = _exact_text("failure_family", value)
        if checked not in _FAILURE_FAMILIES:
            raise ValueError("failure_family is undeclared")
        return checked

    @field_validator("cache_hit", mode="before")
    @classmethod
    def cache_is_exact(cls, value: object) -> object:
        if type(value) is not bool:
            raise TypeError("cache_hit must be an exact bool")
        return value

    @model_validator(mode="after")
    def status_is_consistent(self) -> LLMAuditRecord:
        success = self.call_status.endswith("_success")
        if success != (self.failure_family is None):
            raise ValueError("LLM call status and failure family disagree")
        if self.cache_hit != self.call_status.startswith("cache_"):
            raise ValueError("cache_hit and call status disagree")
        return self


class LLMPlannerPolicy:
    """Propose exact bounded vectors from visible history and public schema only."""

    __slots__ = (
        "_audit_records",
        "_client",
        "_clock_ns",
        "_model_id",
        "_provider",
        "_replay_cache",
        "_require_cached_replay",
    )

    policy_version = _POLICY_VERSION

    def __init__(
        self,
        client: LLMClient,
        *,
        replay_cache: dict[str, dict[str, object]] | None = None,
        require_cached_replay: bool = False,
        clock_ns: Callable[[], int] = time.monotonic_ns,
    ) -> None:
        provider = _exact_text("provider", getattr(client, "provider", None))
        model_id = _exact_text("model_id", getattr(client, "model_id", None))
        if not callable(getattr(client, "complete", None)):
            raise TypeError("LLM client must expose complete")
        if replay_cache is not None and type(replay_cache) is not dict:
            raise TypeError("replay_cache must be an exact dict or None")
        if type(require_cached_replay) is not bool:
            raise TypeError("require_cached_replay must be an exact bool")
        if not callable(clock_ns):
            raise TypeError("clock_ns must be callable")
        checked_cache: dict[str, dict[str, object]] = {}
        for prompt_digest, record in (replay_cache or {}).items():
            if (
                type(prompt_digest) is not str
                or len(prompt_digest) != 64
                or not set(prompt_digest) <= _HEX
            ):
                raise ValueError("replay cache key must be a canonical digest")
            checked_cache[prompt_digest] = self._validate_cache_record(
                record,
                provider=provider,
                model_id=model_id,
            )
        self._client = client
        self._provider = provider
        self._model_id = model_id
        self._replay_cache = checked_cache
        self._audit_records: list[LLMAuditRecord] = []
        self._require_cached_replay = require_cached_replay
        self._clock_ns = clock_ns

    @property
    def policy_name(self) -> str:
        return "cached_llm" if self._require_cached_replay else "llm"

    @staticmethod
    def prompt_document(
        history: tuple[VisibleTrial, ...],
        bounds: ParameterBounds,
    ) -> dict[str, object]:
        visible = reconstruct_history(history)
        public_bounds = reconstruct_bounds(bounds)
        return {
            "protocol": "apar-decision-only-planner-v2",
            "bounds": public_bounds.schema_document(),
            "history": [
                {
                    "candidate_id": trial.candidate.candidate_id,
                    "parent_id": trial.candidate.parent_id,
                    "generation": trial.candidate.generation,
                    "params": trial.candidate.params.json_mapping(),
                    "feedback": {
                        "action": trial.feedback.action.value,
                        "reason_family": trial.feedback.reason_family,
                        "realized_value": (
                            None
                            if trial.feedback.realized_value is None
                            else str(trial.feedback.realized_value)
                        ),
                    },
                    "objective_value": str(trial.objective_value),
                }
                for trial in visible
            ],
            "response_contract": {
                "required": ["generation", "params", "parent_id"],
                "additional_fields": False,
            },
        }

    @staticmethod
    def _schema_digest(bounds: ParameterBounds) -> str:
        return _digest(
            {
                "protocol": "apar-decision-only-planner-v2",
                "bounds": bounds.schema_document(),
                "response_fields": sorted(_PLANNER_FIELDS),
            }
        )

    def _now(self) -> int:
        value = self._clock_ns()
        if type(value) is not int or value < 0:
            raise TypeError("LLM clock must return an exact non-negative integer")
        return value

    def _append_audit(
        self,
        *,
        schema_digest: str,
        prompt_digest: str,
        response_digest: str,
        latency_ms: int,
        input_tokens: int,
        output_tokens: int,
        call_status: str,
        failure_family: str | None,
    ) -> None:
        self._audit_records.append(
            LLMAuditRecord(
                provider=self._provider,
                model_id=self._model_id,
                policy_version=self.policy_version,
                schema_digest=schema_digest,
                prompt_digest=prompt_digest,
                response_digest=response_digest,
                latency_ms=latency_ms,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                call_status=call_status,
                failure_family=failure_family,
                cache_hit=call_status.startswith("cache_"),
            )
        )

    def propose(
        self,
        history: tuple[VisibleTrial, ...],
        bounds: ParameterBounds,
        rng: np.random.Generator | None = None,
    ) -> AttackCandidate:
        if rng is not None and type(rng) is not np.random.Generator:
            raise TypeError("rng must be an exact numpy.random.Generator or None")
        visible = reconstruct_history(history)
        public_bounds = reconstruct_bounds(bounds)
        request = self.prompt_document(visible, public_bounds)
        prompt_digest = _digest(request)
        schema_digest = self._schema_digest(public_bounds)
        cached = self._replay_cache.get(prompt_digest)
        if cached is None and self._require_cached_replay:
            self._append_audit(
                schema_digest=schema_digest,
                prompt_digest=prompt_digest,
                response_digest=_sentinel_digest("cache"),
                latency_ms=0,
                input_tokens=0,
                output_tokens=0,
                call_status="cache_failure",
                failure_family="cache",
            )
            raise ValueError("cached replay miss; network access remains disabled")

        cache_hit = cached is not None
        status_prefix = "cache" if cache_hit else "online"
        started = self._now()
        response_digest = _sentinel_digest("transport")
        latency_ms = 0
        input_tokens = 0
        output_tokens = 0
        failure_family: str | None = None
        try:
            if cached is None:
                try:
                    raw_transport: object = self._client.complete(deepcopy(request))
                except Exception as error:
                    failure_family = "transport"
                    raise ValueError("LLM transport failed") from error
            else:
                raw_transport = cached["transport"]
            try:
                transport = self._validate_transport(raw_transport)
            except (TypeError, ValueError) as error:
                failure_family = "schema"
                raise ValueError(str(error)) from error
            latency_ms = 0 if cache_hit else _count("latency_ms", transport["latency_ms"])
            input_tokens = _count("input_tokens", transport["input_tokens"])
            output_tokens = _count("output_tokens", transport["output_tokens"])
            output = transport["output"]
            response_digest = _digest(output)
            if cached is not None:
                if cached["schema_digest"] != schema_digest:
                    failure_family = "cache"
                    raise ValueError("cached schema digest does not match active bounds")
                if cached["response_digest"] != response_digest:
                    failure_family = "cache"
                    raise ValueError("cached response digest does not match output")
            try:
                candidate = self._decode_candidate(output, visible, public_bounds)
            except CandidateContractError as error:
                failure_family = "bounds"
                raise ValueError(str(error)) from error
            except (TypeError, ValueError) as error:
                failure_family = (
                    "lineage" if "parent" in str(error) or "generation" in str(error) else "schema"
                )
                raise ValueError(str(error)) from error
            if cached is None:
                self._replay_cache[prompt_digest] = {
                    "transport": deepcopy(transport),
                    "response_digest": response_digest,
                    "provider": self._provider,
                    "model_id": self._model_id,
                    "policy_version": self.policy_version,
                    "schema_digest": schema_digest,
                }
        except Exception:
            ended = self._now()
            if latency_ms == 0 and not cache_hit:
                latency_ms = max(0, (ended - started) // 1_000_000)
            family = failure_family or "schema"
            self._append_audit(
                schema_digest=schema_digest,
                prompt_digest=prompt_digest,
                response_digest=(
                    response_digest
                    if response_digest != _sentinel_digest("transport")
                    else _sentinel_digest(family)
                ),
                latency_ms=latency_ms,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                call_status=f"{status_prefix}_failure",
                failure_family=family,
            )
            raise
        self._append_audit(
            schema_digest=schema_digest,
            prompt_digest=prompt_digest,
            response_digest=response_digest,
            latency_ms=latency_ms,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            call_status=f"{status_prefix}_success",
            failure_family=None,
        )
        return candidate

    def take_audit_records(self) -> tuple[LLMAuditRecord, ...]:
        records = tuple(self._audit_records)
        self._audit_records.clear()
        return records

    def export_replay_cache(self) -> dict[str, dict[str, object]]:
        return deepcopy(self._replay_cache)

    @staticmethod
    def _validate_transport(transport: object) -> dict[str, object]:
        _validate_json_tree(transport, label="LLM transport")
        if type(transport) is not dict or set(transport) != _TRANSPORT_FIELDS:
            raise ValueError("LLM transport has missing or undeclared fields")
        if type(transport["output"]) is not dict:
            raise TypeError("LLM output must be an exact object")
        _count("latency_ms", transport["latency_ms"])
        _count("input_tokens", transport["input_tokens"])
        _count("output_tokens", transport["output_tokens"])
        return deepcopy(transport)

    @classmethod
    def _validate_cache_record(
        cls,
        record: object,
        *,
        provider: str,
        model_id: str,
    ) -> dict[str, object]:
        _validate_json_tree(record, label="LLM replay cache")
        if type(record) is not dict or set(record) != _CACHE_FIELDS:
            raise ValueError("replay cache record has missing or undeclared fields")
        if record["provider"] != provider or record["model_id"] != model_id:
            raise ValueError("replay cache provider or model does not match the client")
        if record["policy_version"] != _POLICY_VERSION:
            raise ValueError("replay cache policy version does not match")
        for field in ("response_digest", "schema_digest"):
            value = record[field]
            if type(value) is not str or len(value) != 64 or not set(value) <= _HEX:
                raise ValueError(f"cache {field} must be canonical SHA-256 hex")
        transport = cls._validate_transport(record["transport"])
        if _digest(transport["output"]) != record["response_digest"]:
            raise ValueError("replay cache response digest does not match output")
        return {
            "transport": transport,
            "response_digest": record["response_digest"],
            "provider": provider,
            "model_id": model_id,
            "policy_version": _POLICY_VERSION,
            "schema_digest": record["schema_digest"],
        }

    @staticmethod
    def _decode_candidate(
        output: object,
        history: tuple[VisibleTrial, ...],
        bounds: ParameterBounds,
    ) -> AttackCandidate:
        _validate_json_tree(output, label="LLM planner output")
        if type(output) is not dict:
            raise ValueError("planner output must be an exact object")
        unknown = set(output) - _PLANNER_FIELDS
        missing = _PLANNER_FIELDS - set(output)
        if unknown:
            raise ValueError(f"undeclared planner field: {sorted(unknown)}")
        if missing:
            raise ValueError(f"missing planner field: {sorted(missing)}")
        generation = output["generation"]
        if type(generation) is not int or generation != len(history):
            raise ValueError("generation must exactly match visible history length")
        parent_id = output["parent_id"]
        if not history:
            if parent_id is not None:
                raise ValueError("root parent_id must be null")
        elif type(parent_id) is not str or parent_id not in {
            trial.candidate.candidate_id for trial in history
        }:
            raise ValueError("parent_id must reference visible history")
        params = bounds.decode_updates(output["params"])
        return validate_candidate_lineage(
            AttackCandidate(
                params=params,
                parent_id=parent_id,
                generation=generation,
            ),
            history,
        )


__all__ = ["LLMAuditRecord", "LLMClient", "LLMPlannerPolicy"]
