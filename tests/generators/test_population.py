"""Population generation is deterministic, connected, and free of secrets."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from datetime import UTC

import pytest

from apar.compiler.compiler import compile_scenario
from apar.generators.population import PopulationGenerator
from tests.factories import make_scenario_config, make_threat_card


def _bundle(seed: int):  # type: ignore[no-untyped-def]
    config = make_scenario_config(
        seed=seed,
        replay=make_scenario_config().replay.model_copy(update={"random_seed": seed}),
        benign_entity_count=36,
        illicit_entity_count=12,
    )
    return compile_scenario(make_threat_card(default_config=config), config)


@pytest.mark.parametrize("seed", [7, 19, 101, 260_816, 900_001])
def test_population_is_byte_reproducible_for_five_seeds(seed: int) -> None:
    first = PopulationGenerator(seed=seed).generate(_bundle(seed))
    second = PopulationGenerator(seed=seed).generate(_bundle(seed))

    assert first.canonical_bytes() == second.canonical_bytes()
    assert first.seed == seed
    assert first.generated_at.tzinfo is UTC


def test_population_declares_counts_accounts_and_closed_relationships() -> None:
    population = PopulationGenerator(seed=41).generate(_bundle(41))
    entity_ids = {entity.entity_id for entity in population.entities}
    account_ids = {entity.account_id for entity in population.entities if entity.account_id}

    assert len(population.entities) == 48
    assert sum(not entity.illicit for entity in population.entities) == 36
    assert sum(entity.illicit for entity in population.entities) == 12
    assert account_ids <= set(population.opening_balances)
    assert all(
        edge.source_id in entity_ids and edge.target_id in entity_ids
        for edge in population.relationships
    )
    assert population.by_role("mule")
    assert population.by_role("synthetic_merchant")
    assert population.by_role("agent")


def test_benign_population_contains_shift_controls_that_overlap_attack_novelty() -> None:
    population = PopulationGenerator(seed=53).generate(_bundle(53))

    assert {"shared_device", "shared_beneficiary", "new_merchant", "travel"} <= set(
        population.benign_controls
    )
    shared_device_edges = [
        edge for edge in population.relationships if edge.relation == "shares_device"
    ]
    shared_beneficiary_edges = [
        edge for edge in population.relationships if edge.relation == "pays_shared_beneficiary"
    ]
    assert len(shared_device_edges) >= 2
    assert len(shared_beneficiary_edges) >= 2


def test_population_records_are_immutable_and_do_not_export_private_keys() -> None:
    population = PopulationGenerator(seed=67).generate(_bundle(67))

    with pytest.raises(FrozenInstanceError):
        population.entities[0].role = "attacker"  # type: ignore[misc]
    with pytest.raises(TypeError):
        population.opening_balances["new"] = population.opening_balances[  # type: ignore[index]
            next(iter(population.opening_balances))
        ]
    serialized = population.canonical_bytes().lower()
    assert b"private_key" not in serialized
    assert b"secret_key" not in serialized
