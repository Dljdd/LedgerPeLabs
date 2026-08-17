"""Strict LLM planner schema, audit, and offline replay tests."""

from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest

from apar.redteam import LLMPlannerPolicy


class FakeLLM:
    provider = "fixture"
    model_id = "planner-v1"

    def __init__(self, response: object) -> None:
        self.response = response
        self.calls = 0

    def complete(self, _request: dict[str, object]) -> dict[str, object]:
        self.calls += 1
        return {
            "output": deepcopy(self.response),
            "latency_ms": 7,
            "input_tokens": 11,
            "output_tokens": 5,
        }


def valid_output(bounds) -> dict[str, object]:  # type: ignore[no-untyped-def]
    return {
        "params": bounds.defaults_document(),
        "parent_id": None,
        "generation": 0,
    }


def test_llm_rejects_undeclared_top_level_and_parameter_fields(card_bounds) -> None:  # type: ignore[no-untyped-def]
    undeclared_top = valid_output(card_bounds) | {"model_score": 0.9}
    with pytest.raises(ValueError, match="undeclared planner field"):
        LLMPlannerPolicy(FakeLLM(undeclared_top)).propose((), card_bounds)

    undeclared_param = valid_output(card_bounds)
    undeclared_param["params"] = undeclared_param["params"] | {"threshold": 0.2}  # type: ignore[operator]
    with pytest.raises(ValueError, match="undeclared parameter"):
        LLMPlannerPolicy(FakeLLM(undeclared_param)).propose((), card_bounds)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda output: output.pop("generation"),
        lambda output: output["params"].pop("retry_intensity"),
        lambda output: output["params"].__setitem__("retry_intensity", True),
        lambda output: output["params"].__setitem__("retry_intensity", 999),
    ],
)
def test_llm_rejects_missing_wrong_type_and_out_of_bounds_output(
    card_bounds, mutate  # type: ignore[no-untyped-def]
) -> None:
    output = valid_output(card_bounds)
    mutate(output)
    with pytest.raises(ValueError):
        LLMPlannerPolicy(FakeLLM(output)).propose((), card_bounds)


def test_llm_records_digest_only_audit_and_supports_zero_network_replay(card_bounds) -> None:  # type: ignore[no-untyped-def]
    client = FakeLLM(valid_output(card_bounds))
    online = LLMPlannerPolicy(client)
    candidate = online.propose((), card_bounds, np.random.default_rng(3))
    records = online.take_audit_records()
    cache = online.export_replay_cache()

    class NoNetwork(FakeLLM):
        def complete(self, _request: dict[str, object]) -> dict[str, object]:
            raise AssertionError("network path used during cached replay")

    replay = LLMPlannerPolicy(
        NoNetwork({}), replay_cache=cache, require_cached_replay=True
    )
    replayed = replay.propose((), card_bounds)
    replay_records = replay.take_audit_records()

    assert candidate == replayed
    assert client.calls == 1
    assert records[0].provider == "fixture"
    assert records[0].model_id == "planner-v1"
    assert records[0].latency_ms == 7
    assert records[0].input_tokens == 11
    assert records[0].output_tokens == 5
    assert records[0].cache_hit is False
    assert replay_records[0].cache_hit is True
    assert replay_records[0].latency_ms == 0
    assert records[0].prompt_digest == replay_records[0].prompt_digest
    assert records[0].response_digest == replay_records[0].response_digest
    assert replay.policy_name == "cached_llm"
    serialized = repr(records).lower()
    assert "retry_intensity" not in serialized
    assert "reasoning" not in serialized


def test_llm_prompt_contains_only_visible_history_and_adaptive_bounds(card_bounds) -> None:  # type: ignore[no-untyped-def]
    document = LLMPlannerPolicy.prompt_document((), card_bounds)
    text = repr(document)
    for forbidden in (
        "model_score",
        "feature",
        "threshold",
        "gradient",
        "label",
        "hidden",
        "CampaignEvidence",
        "mutation_reason",
        "campaign_id",
    ):
        assert forbidden not in text


def test_replay_cache_binds_response_digest_provider_and_model(card_bounds) -> None:  # type: ignore[no-untyped-def]
    online = LLMPlannerPolicy(FakeLLM(valid_output(card_bounds)))
    online.propose((), card_bounds)
    cache = online.export_replay_cache()
    prompt_digest = next(iter(cache))
    record = cache[prompt_digest]
    transport = record["transport"]
    assert type(transport) is dict
    output = transport["output"]
    assert type(output) is dict
    params = output["params"]
    assert type(params) is dict
    params["retry_intensity"] = 3
    with pytest.raises(ValueError, match="response digest"):
        LLMPlannerPolicy(FakeLLM({}), replay_cache=cache)

    clean = online.export_replay_cache()
    clean[prompt_digest]["provider"] = "different-provider"
    with pytest.raises(ValueError, match="provider or model"):
        LLMPlannerPolicy(FakeLLM({}), replay_cache=clean)


@pytest.mark.parametrize(
    "response",
    [
        {"params": {"retry_intensity": float("nan")}},
        {"params": {"retry_intensity": float("inf")}},
        {"params": {"retry_intensity": 2}, "reasoning": "private chain"},
    ],
)
def test_llm_rejects_nonfinite_and_reasoning_trace_outputs(card_bounds, response) -> None:  # type: ignore[no-untyped-def]
    with pytest.raises((TypeError, ValueError)):
        LLMPlannerPolicy(FakeLLM(response)).propose((), card_bounds)


def test_llm_rejects_mapping_and_scalar_subclasses(card_bounds) -> None:  # type: ignore[no-untyped-def]
    class DictSubclass(dict[str, object]):
        pass

    class IntSubclass(int):
        pass

    with pytest.raises((TypeError, ValueError)):
        LLMPlannerPolicy(FakeLLM(DictSubclass(valid_output(card_bounds)))).propose(
            (), card_bounds
        )
    output = valid_output(card_bounds)
    params = output["params"]
    assert type(params) is dict
    params["retry_intensity"] = IntSubclass(2)
    with pytest.raises((TypeError, ValueError)):
        LLMPlannerPolicy(FakeLLM(output)).propose((), card_bounds)


def test_llm_rejects_numerically_equal_noncanonical_decimal_text(card_bounds) -> None:  # type: ignore[no-untyped-def]
    output = valid_output(card_bounds)
    params = output["params"]
    assert type(params) is dict
    params["merchant_concentration"] = "0.700"
    with pytest.raises(ValueError, match="canonical domain"):
        LLMPlannerPolicy(FakeLLM(output)).propose((), card_bounds)


def test_cache_only_mode_fails_before_network_on_a_cache_miss(card_bounds) -> None:  # type: ignore[no-untyped-def]
    client = FakeLLM(valid_output(card_bounds))
    policy = LLMPlannerPolicy(client, require_cached_replay=True)
    with pytest.raises(ValueError, match="cached replay miss"):
        policy.propose((), card_bounds)
    assert client.calls == 0
