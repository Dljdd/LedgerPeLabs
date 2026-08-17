"""Schema-constrained optional LLM planner with digest-only audit records."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from typing import Protocol

import numpy as np
from pydantic import field_validator

from apar.contracts._validation import ExternalContract
from apar.redteam.policies import (
    AttackCandidate,
    ParameterBounds,
    VisibleTrial,
)

_HEX = frozenset("0123456789abcdef")
_TRANSPORT_FIELDS = frozenset(
    {"output", "latency_ms", "input_tokens", "output_tokens"}
)
_PLANNER_FIELDS = frozenset({"params", "parent_id", "generation"})
_CACHE_FIELDS = frozenset({"transport", "response_digest", "provider", "model_id"})


def _exact_text(label: str, value: object) -> str:
    if type(value) is not str or not value:
        raise TypeError(f"{label} must be an exact non-empty string")
    return value


def _count(label: str, value: object) -> int:
    if type(value) is not int or value < 0:
        raise TypeError(f"{label} must be an exact non-negative integer")
    return value


def _canonical_bytes(document: object) -> bytes:
    try:
        return json.dumps(
            document,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode()
    except (TypeError, ValueError) as error:
        raise ValueError("LLM document must be strict canonical JSON") from error


def _digest(document: object) -> str:
    return hashlib.sha256(_canonical_bytes(document)).hexdigest()


class LLMClient(Protocol):
    """Minimal transport interface; it receives no evaluator capability."""

    provider: str
    model_id: str

    def complete(self, request: dict[str, object]) -> dict[str, object]: ...


class LLMAuditRecord(ExternalContract):
    """Evaluator-transferable audit metadata with no prompt or response body."""

    provider: str
    model_id: str
    prompt_digest: str
    response_digest: str
    latency_ms: int
    input_tokens: int
    output_tokens: int
    cache_hit: bool

    @field_validator("provider", "model_id", mode="before")
    @classmethod
    def text_is_exact(cls, value: object) -> object:
        return _exact_text("LLM identity", value)

    @field_validator("prompt_digest", "response_digest", mode="before")
    @classmethod
    def digest_is_canonical(cls, value: object) -> object:
        text = _exact_text("digest", value)
        if len(text) != 64 or not set(text) <= _HEX:
            raise ValueError("digest must be lowercase SHA-256 hex")
        return text

    @field_validator("latency_ms", "input_tokens", "output_tokens", mode="before")
    @classmethod
    def counts_are_exact(cls, value: object) -> object:
        return _count("LLM audit count", value)

    @field_validator("cache_hit", mode="before")
    @classmethod
    def cache_hit_is_exact(cls, value: object) -> object:
        if type(value) is not bool:
            raise TypeError("cache_hit must be an exact bool")
        return value


class LLMPlannerPolicy:
    """Propose only exact bounded parameters from visible decision history."""

    __slots__ = (
        "_audit_records",
        "_client",
        "_replay_cache",
        "_require_cached_replay",
    )

    def __init__(
        self,
        client: LLMClient,
        *,
        replay_cache: dict[str, dict[str, object]] | None = None,
        require_cached_replay: bool = False,
    ) -> None:
        provider = getattr(client, "provider", None)
        model_id = getattr(client, "model_id", None)
        _exact_text("provider", provider)
        _exact_text("model_id", model_id)
        if not callable(getattr(client, "complete", None)):
            raise TypeError("LLM client must expose complete")
        if replay_cache is not None and type(replay_cache) is not dict:
            raise TypeError("replay_cache must be an exact dict or None")
        if type(require_cached_replay) is not bool:
            raise TypeError("require_cached_replay must be an exact bool")
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
                provider=_exact_text("provider", provider),
                model_id=_exact_text("model_id", model_id),
            )
        self._client = client
        self._replay_cache = checked_cache
        self._audit_records: list[LLMAuditRecord] = []
        self._require_cached_replay = require_cached_replay

    @property
    def policy_name(self) -> str:
        return "cached_llm" if self._require_cached_replay else "llm"

    @staticmethod
    def prompt_document(
        history: tuple[VisibleTrial, ...],
        bounds: ParameterBounds,
    ) -> dict[str, object]:
        if type(history) is not tuple or any(type(item) is not VisibleTrial for item in history):
            raise TypeError("history must be an exact tuple of exact VisibleTrial records")
        if type(bounds) is not ParameterBounds:
            raise TypeError("bounds must be an exact ParameterBounds")
        return {
            "protocol": "apar-decision-only-planner-v1",
            "bounds": bounds.schema_document(),
            "history": [
                {
                    "candidate_id": trial.candidate.candidate_id,
                    "parent_id": trial.candidate.parent_id,
                    "generation": trial.candidate.generation,
                    "params": {
                        name: bounds.defaults_document()[name]
                        if getattr(trial.candidate.params, name)
                        == getattr(bounds.template, name)
                        else _json_adaptive_value(
                            getattr(trial.candidate.params, name)
                        )
                        for name in bounds.names
                    },
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
                for trial in history
            ],
            "response_contract": {
                "required": ["generation", "params", "parent_id"],
                "additional_fields": False,
            },
        }

    def propose(
        self,
        history: tuple[VisibleTrial, ...],
        bounds: ParameterBounds,
        rng: np.random.Generator | None = None,
    ) -> AttackCandidate:
        if rng is not None and type(rng) is not np.random.Generator:
            raise TypeError("rng must be an exact numpy.random.Generator or None")
        request = self.prompt_document(history, bounds)
        prompt_digest = _digest(request)
        cached_record = self._replay_cache.get(prompt_digest)
        if cached_record is None:
            if self._require_cached_replay:
                raise ValueError("cached replay miss; network access remains disabled")
            transport = self._validate_transport(
                self._client.complete(deepcopy(request))
            )
            output_digest = _digest(transport["output"])
            self._replay_cache[prompt_digest] = {
                "transport": deepcopy(transport),
                "response_digest": output_digest,
                "provider": _exact_text("provider", self._client.provider),
                "model_id": _exact_text("model_id", self._client.model_id),
            }
            cache_hit = False
            latency_ms = _count("latency_ms", transport["latency_ms"])
        else:
            transport = self._validate_transport(cached_record["transport"])
            cache_hit = True
            latency_ms = 0
        output = transport["output"]
        candidate = self._decode_candidate(output, history, bounds)
        response_digest = _digest(output)
        self._audit_records.append(
            LLMAuditRecord(
                provider=_exact_text("provider", self._client.provider),
                model_id=_exact_text("model_id", self._client.model_id),
                prompt_digest=prompt_digest,
                response_digest=response_digest,
                latency_ms=latency_ms,
                input_tokens=_count("input_tokens", transport["input_tokens"]),
                output_tokens=_count("output_tokens", transport["output_tokens"]),
                cache_hit=cache_hit,
            )
        )
        return candidate

    def take_audit_records(self) -> tuple[LLMAuditRecord, ...]:
        """Transfer digest-only records to the evaluator and clear policy state."""
        records = tuple(self._audit_records)
        self._audit_records.clear()
        return records

    def export_replay_cache(self) -> dict[str, dict[str, object]]:
        """Return the sanitized schema response cache for deterministic offline replay."""
        return deepcopy(self._replay_cache)

    @staticmethod
    def _validate_transport(transport: object) -> dict[str, object]:
        if type(transport) is not dict:
            raise TypeError("LLM transport result must be an exact object")
        if set(transport) != _TRANSPORT_FIELDS:
            raise ValueError("LLM transport result has missing or undeclared fields")
        if type(transport["output"]) is not dict:
            raise TypeError("LLM output must be an exact object")
        _count("latency_ms", transport["latency_ms"])
        _count("input_tokens", transport["input_tokens"])
        _count("output_tokens", transport["output_tokens"])
        _canonical_bytes(transport["output"])
        return deepcopy(transport)

    @classmethod
    def _validate_cache_record(
        cls,
        record: object,
        *,
        provider: str,
        model_id: str,
    ) -> dict[str, object]:
        if type(record) is not dict or set(record) != _CACHE_FIELDS:
            raise ValueError("replay cache record has missing or undeclared fields")
        record_provider = _exact_text("cache provider", record["provider"])
        record_model = _exact_text("cache model_id", record["model_id"])
        if record_provider != provider or record_model != model_id:
            raise ValueError("replay cache provider or model does not match the client")
        response_digest = _exact_text("response digest", record["response_digest"])
        if len(response_digest) != 64 or not set(response_digest) <= _HEX:
            raise ValueError("response digest must be canonical SHA-256 hex")
        transport = cls._validate_transport(record["transport"])
        if _digest(transport["output"]) != response_digest:
            raise ValueError("replay cache response digest does not match output")
        return {
            "transport": transport,
            "response_digest": response_digest,
            "provider": record_provider,
            "model_id": record_model,
        }

    @staticmethod
    def _decode_candidate(
        output: object,
        history: tuple[VisibleTrial, ...],
        bounds: ParameterBounds,
    ) -> AttackCandidate:
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
        if parent_id is not None:
            if type(parent_id) is not str:
                raise ValueError("parent_id must be an exact string or null")
            visible_ids = {trial.candidate.candidate_id for trial in history}
            if parent_id not in visible_ids:
                raise ValueError("parent_id must reference visible history")
        try:
            updates = bounds.decode_updates(output["params"])
            params = bounds.with_updates(bounds.template, updates)
        except (ArithmeticError, TypeError, ValueError) as error:
            raise ValueError(str(error)) from error
        return AttackCandidate(
            params=params,
            parent_id=parent_id,
            generation=generation,
        )


def _json_adaptive_value(value: object) -> object:
    from decimal import Decimal

    if type(value) is Decimal:
        if not value.is_finite():
            raise ValueError("adaptive Decimal must be finite")
        return str(value)
    if type(value) in {int, str}:
        return value
    if type(value) is tuple and all(type(item) is str for item in value):
        return list(value)
    raise TypeError("adaptive value is not strict JSON compatible")


__all__ = ["LLMAuditRecord", "LLMClient", "LLMPlannerPolicy"]
