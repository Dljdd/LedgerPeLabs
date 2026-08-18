"""Round-three causal binding, temporal, and canonical-lineage regressions."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta
from decimal import Decimal
from typing import cast

import pytest

from apar.compiler.compiler import compile_scenario
from apar.contracts.events import Rail
from apar.generators.campaigns import (
    AGENTIC_INTENT_ABUSE_MOTIF,
    APP_SCAM_MULE_MOTIF,
    CARD_TESTING_CNP_MOTIF,
    SYNTHETIC_MERCHANT_REFUND_MOTIF,
    CampaignParams,
    GenerationConstraintError,
    _CampaignEvaluator,
    campaign_bytes,
    motif_signature,
)
from apar.generators.population import PopulationGenerator
from apar.simulator.clock import Command
from apar.simulator.rails.a2a import (
    AcceptA2A,
    FreezeA2AFunds,
    InitiateA2A,
    PostA2A,
    RecoverA2A,
    ReportA2AFraud,
    ReturnA2A,
)
from apar.simulator.rails.card import (
    AuthorizeCard,
    ChargebackCard,
    ClearCard,
    OpenCardDispute,
    RecoverCard,
    RefundCard,
    ReportCardFraud,
    SettleCard,
)
from tests.factories import make_scenario_config, make_threat_card

_MOTIFS = {
    "app_scam_mule": APP_SCAM_MULE_MOTIF,
    "card_testing_cnp": CARD_TESTING_CNP_MOTIF,
    "synthetic_merchant_refund": SYNTHETIC_MERCHANT_REFUND_MOTIF,
    "agentic_intent_abuse": AGENTIC_INTENT_ABUSE_MOTIF,
}


def _bundle(seed: int):  # type: ignore[no-untyped-def]
    config = make_scenario_config(
        rail=Rail.A2A,
        seed=seed,
        replay=make_scenario_config().replay.model_copy(update={"random_seed": seed}),
        benign_entity_count=40,
        illicit_entity_count=32,
        duration_hours=24,
    )
    return compile_scenario(make_threat_card(rails=[Rail.A2A], default_config=config), config)


def _params(family: str, **updates: object) -> CampaignParams:
    values: dict[str, object] = {
        "campaign_id": "00000000-0000-4000-8000-000000000903",
        "seed": 303,
        "payment_count": 10,
        "target_illicit_rate": Decimal("0.70"),
        "class_rate_tolerance": Decimal("0.01"),
        "target_value_total": Decimal("500.00"),
        "value_tolerance": Decimal("0.01"),
        "min_amount": Decimal("10.00"),
        "max_amount": Decimal("90.00"),
        "currency": "USD",
        "duration_hours": 12,
        "query_budget": 40,
        "min_delay_seconds": 1,
        "max_delay_seconds": 300,
        "expected_motif": _MOTIFS[family],
    }
    if family == "agentic_intent_abuse":
        values.update(
            payment_count=25,
            target_illicit_rate=Decimal("0.92"),
            agentic_attack_mix=Decimal("0.92"),
        )
    values.update(updates)
    return CampaignParams(**values)  # type: ignore[arg-type]


def _groups(commands: tuple[Command, ...]) -> list[tuple[Command, ...]]:
    groups: list[tuple[Command, ...]] = []
    for command in commands:
        payment_id = cast(str, command.payload["payment_id"])
        if not groups or groups[-1][0].payload["payment_id"] != payment_id:
            groups.append((command,))
        else:
            groups[-1] = (*groups[-1], command)
    return groups


def _a2a_followups(
    payment_id: str, campaign_id: str, stages: tuple[str, ...]
) -> tuple[Command, ...]:
    types = {
        "accept": AcceptA2A,
        "post": PostA2A,
        "return": ReturnA2A,
        "report": ReportA2AFraud,
        "freeze": FreezeA2AFunds,
        "recover": RecoverA2A,
    }
    return tuple(
        types[stage](
            payment_id,
            idempotency_key=f"a2a.{stage}:{payment_id}:campaign:{campaign_id}",
        )
        for stage in stages
    )


def _card_followups(
    payment_id: str, campaign_id: str, stages: tuple[str, ...]
) -> tuple[Command, ...]:
    types = {
        "clear": ClearCard,
        "settle": SettleCard,
        "refund": RefundCard,
        "report": ReportCardFraud,
        "dispute": OpenCardDispute,
        "chargeback": ChargebackCard,
        "recover": RecoverCard,
    }
    return tuple(
        types[stage](
            payment_id,
            idempotency_key=f"card.{stage}:{payment_id}:campaign:{campaign_id}",
        )
        for stage in stages
    )


def test_app_terminals_are_causally_bound_to_role_paths() -> None:
    population = PopulationGenerator(seed=31).generate(_bundle(31))
    params = _params("app_scam_mule", seed=31)
    evaluator = _CampaignEvaluator(seed=31)
    commands, audit = evaluator.generate("app_scam_mule", population, params)
    groups = _groups(commands)
    mule_ids = {entity.entity_id for entity in population.by_role("mule")}
    attacker_ids = {entity.entity_id for entity in population.by_role("attacker")}
    cash_index = next(
        index
        for index, group in enumerate(groups)
        if group[0].payload["actor_id"] in mule_ids
        and group[0].payload["counterparty_id"] in attacker_ids
        and any(command.name == "a2a.recover" for command in group)
    )
    benign_index = next(
        index
        for index, group in enumerate(groups)
        if any(command.name == "a2a.return" for command in group)
    )
    cash_open = groups[cash_index][0]
    benign_open = groups[benign_index][0]
    cash_id = cast(str, cash_open.payload["payment_id"])
    benign_id = cast(str, benign_open.payload["payment_id"])
    groups[cash_index] = (
        cash_open,
        *_a2a_followups(cash_id, params.campaign_id, ("accept", "post", "return")),
    )
    groups[benign_index] = (
        benign_open,
        *_a2a_followups(
            benign_id,
            params.campaign_id,
            ("accept", "post", "report", "freeze", "recover"),
        ),
    )
    mutated = tuple(command for group in groups for command in group)

    with pytest.raises(GenerationConstraintError):
        evaluator.validate(
            "app_scam_mule",
            mutated,
            population,
            params,
            schedule=audit.schedule,
        )


def test_synthetic_merchant_terminals_are_causally_bound_to_role_paths() -> None:
    population = PopulationGenerator(seed=32).generate(_bundle(32))
    params = _params("synthetic_merchant_refund", seed=32)
    evaluator = _CampaignEvaluator(seed=32)
    commands, audit = evaluator.generate("synthetic_merchant_refund", population, params)
    groups = _groups(commands)
    entities = {entity.entity_id: entity for entity in population.entities}
    recovery_index = next(
        index
        for index, group in enumerate(groups)
        if any(command.name == "card.recover" for command in group)
    )
    benign_index = next(
        index
        for index, group in enumerate(groups)
        if not entities[cast(str, group[0].payload["actor_id"])].illicit
    )
    attack_open = groups[recovery_index][0]
    benign_open = groups[benign_index][0]
    attack_id = cast(str, attack_open.payload["payment_id"])
    benign_id = cast(str, benign_open.payload["payment_id"])
    groups[recovery_index] = (
        attack_open,
        *_card_followups(attack_id, params.campaign_id, ("clear", "settle", "refund")),
    )
    groups[benign_index] = (
        benign_open,
        *_card_followups(
            benign_id,
            params.campaign_id,
            ("clear", "settle", "report", "dispute", "chargeback", "recover"),
        ),
    )
    mutated = tuple(command for group in groups for command in group)

    with pytest.raises(GenerationConstraintError):
        evaluator.validate(
            "synthetic_merchant_refund",
            mutated,
            population,
            params,
            schedule=audit.schedule,
        )


def test_card_testing_requires_probe_then_tighter_success_temporal_regions() -> None:
    population = PopulationGenerator(seed=33).generate(_bundle(33))
    params = _params("card_testing_cnp", seed=33)
    evaluator = _CampaignEvaluator(seed=33)
    commands, audit = evaluator.generate("card_testing_cnp", population, params)
    groups = _groups(commands)
    success_index = next(
        index for index, group in enumerate(groups) if type(group[0]) is AuthorizeCard
    )
    reordered_groups = (
        groups[success_index],
        *groups[:success_index],
        *groups[success_index + 1 :],
    )
    reordered = tuple(command for group in reordered_groups for command in group)
    with pytest.raises(GenerationConstraintError):
        evaluator.validate(
            "card_testing_cnp",
            reordered,
            population,
            params,
            schedule=audit.schedule,
        )

    uniform = tuple(
        population.generated_at + timedelta(seconds=100 * (index + 1))
        for index in range(len(commands))
    )
    with pytest.raises(GenerationConstraintError):
        evaluator.validate(
            "card_testing_cnp",
            commands,
            population,
            params,
            schedule=uniform,
        )


def test_canonical_payment_trace_and_idempotency_lineage_rejects_coherent_drift() -> None:
    population = PopulationGenerator(seed=34).generate(_bundle(34))
    params = _params("app_scam_mule", seed=34)
    evaluator = _CampaignEvaluator(seed=34)
    commands, audit = evaluator.generate("app_scam_mule", population, params)
    groups = _groups(commands)
    first = cast(InitiateA2A, groups[0][0])
    opaque_id = "app_scam_mule:00000000-0000-4000-8000-000000000111"
    opaque_trace = "00000000-0000-4000-8000-000000000112"
    opening = InitiateA2A(
        opaque_id,
        amount=cast(Decimal, first.payload["amount"]),
        currency=cast(str, first.payload["currency"]),
        payer_account=cast(str, first.payload["payer_account"]),
        payee_account=cast(str, first.payload["payee_account"]),
        actor_id=cast(str, first.payload["actor_id"]),
        counterparty_id=cast(str, first.payload["counterparty_id"]),
        campaign_id=params.campaign_id,
        trace_id=opaque_trace,
    )
    stages = tuple(command.name.split(".", 1)[1] for command in groups[0][1:])
    groups[0] = (
        opening,
        *_a2a_followups(opaque_id, params.campaign_id, stages),
    )
    renamed = tuple(command for group in groups for command in group)
    with pytest.raises(GenerationConstraintError):
        evaluator.validate(
            "app_scam_mule",
            renamed,
            population,
            params,
            schedule=audit.schedule,
        )

    accept_index = next(
        index
        for index, command in enumerate(commands)
        if command.name == "a2a.accept"
    )
    accept = commands[accept_index]
    prefix_drift = AcceptA2A(
        cast(str, accept.payload["payment_id"]),
        idempotency_key=f"opaque-prefix:campaign:{params.campaign_id}",
    )
    changed = (*commands[:accept_index], prefix_drift, *commands[accept_index + 1 :])
    with pytest.raises(GenerationConstraintError):
        evaluator.validate(
            "app_scam_mule",
            changed,
            population,
            params,
            schedule=audit.schedule,
        )


def test_agentic_context_free_signature_never_claims_deep_verified_motif() -> None:
    population = PopulationGenerator(seed=35).generate(_bundle(35))
    commands, audit = _CampaignEvaluator(seed=35).generate(
        "agentic_intent_abuse",
        population,
        _params("agentic_intent_abuse", seed=35),
    )
    assert audit.motif_signature == AGENTIC_INTENT_ABUSE_MOTIF
    with pytest.raises(ValueError, match="requires evaluator execution evidence"):
        motif_signature(commands)


def test_adaptive_clipping_and_contradictory_regions_reject() -> None:
    population = PopulationGenerator(seed=36).generate(_bundle(36))
    card = _params("card_testing_cnp", seed=36)
    valid, _ = _CampaignEvaluator(seed=36).generate(
        "card_testing_cnp", population, replace(card, retry_intensity=6)
    )
    assert valid
    for retry in (7, 10):
        with pytest.raises(GenerationConstraintError):
            _CampaignEvaluator(seed=36).generate(
                "card_testing_cnp",
                population,
                replace(card, retry_intensity=retry),
            )

    for family in ("app_scam_mule", "synthetic_merchant_refund"):
        with pytest.raises(GenerationConstraintError):
            _CampaignEvaluator(seed=36).generate(
                family,
                population,
                replace(_params(family, seed=36), recovery_probability=Decimal("0")),
            )

    app = _params("app_scam_mule", seed=36, cash_out_strategy="burst")
    for delay in (10, 200):
        with pytest.raises(GenerationConstraintError):
            _CampaignEvaluator(seed=36).generate(
                "app_scam_mule",
                population,
                replace(app, cash_out_delay_seconds=delay),
            )


def test_card_testing_requires_a_strictly_separated_delay_region() -> None:
    population = PopulationGenerator(seed=37).generate(_bundle(37))
    evaluator = _CampaignEvaluator(seed=37)
    base = _params("card_testing_cnp", seed=37)

    for maximum in (1, 2):
        with pytest.raises(GenerationConstraintError) as caught:
            evaluator.generate(
                "card_testing_cnp",
                population,
                replace(
                    base,
                    min_delay_seconds=1,
                    max_delay_seconds=maximum,
                ),
            )
        assert caught.value.code == "GENERATION_CONSTRAINT_UNSATISFIED"
        assert caught.value.attempts == 100

    params = replace(base, min_delay_seconds=1, max_delay_seconds=3)
    commands, audit = evaluator.generate("card_testing_cnp", population, params)
    entity_roles = {entity.entity_id: entity.role for entity in population.entities}
    delays = tuple(
        int(
            (
                timestamp
                - (population.generated_at if index == 0 else audit.schedule[index - 1])
            ).total_seconds()
        )
        for index, timestamp in enumerate(audit.schedule)
    )
    probe_delays = tuple(
        delays[index]
        for index, command in enumerate(commands)
        if command.name == "card.decline"
        and entity_roles[cast(str, command.payload["actor_id"])] == "attacker"
    )
    illicit_success_ids = {
        cast(str, command.payload["payment_id"])
        for command in commands
        if command.name == "card.authorize"
        and entity_roles[cast(str, command.payload["actor_id"])] == "attacker"
    }
    success_delays = tuple(
        delays[index]
        for index, command in enumerate(commands)
        if cast(str, command.payload["payment_id"]) in illicit_success_ids
    )

    assert probe_delays
    assert success_delays
    assert min(probe_delays) > max(success_delays)
    assert min(probe_delays) >= 2
    assert max(success_delays) <= 1


@pytest.mark.parametrize("family", ["app_scam_mule", "synthetic_merchant_refund"])
@pytest.mark.parametrize("probability", [Decimal("0.01"), Decimal("0.10")])
def test_noncanonical_recovery_probabilities_reject(
    family: str,
    probability: Decimal,
) -> None:
    population = PopulationGenerator(seed=38).generate(_bundle(38))
    with pytest.raises(GenerationConstraintError) as caught:
        _CampaignEvaluator(seed=38).generate(
            family,
            population,
            replace(
                _params(family, seed=38),
                recovery_probability=probability,
            ),
        )
    assert caught.value.code == "GENERATION_CONSTRAINT_UNSATISFIED"
    assert caught.value.attempts == 100


def test_canonical_recovery_levels_realize_distinct_exact_counts() -> None:
    population = PopulationGenerator(seed=39).generate(_bundle(39))

    app = _params("app_scam_mule", seed=39)
    app_one, _ = _CampaignEvaluator(seed=39).generate(
        "app_scam_mule",
        population,
        replace(app, recovery_probability=Decimal("0.25")),
    )
    app_two, _ = _CampaignEvaluator(seed=39).generate(
        "app_scam_mule",
        population,
        replace(app, recovery_probability=Decimal("1")),
    )
    assert sum(command.name == "a2a.recover" for command in app_one) == 1
    assert sum(command.name == "a2a.recover" for command in app_two) == 2
    assert campaign_bytes(app_one) != campaign_bytes(app_two)
    with pytest.raises(GenerationConstraintError):
        _CampaignEvaluator(seed=39).generate(
            "app_scam_mule",
            population,
            replace(app, recovery_probability=Decimal("0.50")),
        )

    synthetic = _params("synthetic_merchant_refund", seed=39)
    synthetic_two, _ = _CampaignEvaluator(seed=39).generate(
        "synthetic_merchant_refund",
        population,
        replace(synthetic, recovery_probability=Decimal("0.25")),
    )
    synthetic_three, _ = _CampaignEvaluator(seed=39).generate(
        "synthetic_merchant_refund",
        population,
        replace(
            synthetic,
            recovery_probability=Decimal("0.428571428571428571"),
        ),
    )
    assert sum(command.name == "card.recover" for command in synthetic_two) == 2
    assert sum(command.name == "card.recover" for command in synthetic_three) == 3
    assert campaign_bytes(synthetic_two) != campaign_bytes(synthetic_three)
    with pytest.raises(GenerationConstraintError):
        _CampaignEvaluator(seed=39).generate(
            "synthetic_merchant_refund",
            population,
            replace(synthetic, recovery_probability=Decimal("0.30")),
        )


def test_staged_cash_out_delay_is_an_exact_external_schedule_witness() -> None:
    population = PopulationGenerator(seed=40).generate(_bundle(40))
    base = _params(
        "app_scam_mule",
        seed=40,
        cash_out_strategy="staged",
        min_delay_seconds=1,
        max_delay_seconds=300,
    )
    evaluator = _CampaignEvaluator(seed=40)
    commands_two, audit_two = evaluator.generate(
        "app_scam_mule",
        population,
        replace(base, cash_out_delay_seconds=2),
    )
    commands_three, audit_three = evaluator.generate(
        "app_scam_mule",
        population,
        replace(base, cash_out_delay_seconds=3),
    )

    entity_roles = {entity.entity_id: entity.role for entity in population.entities}

    def cash_positions(commands: tuple[Command, ...]) -> tuple[int, ...]:
        return tuple(
            index
            for index, command in enumerate(commands)
            if command.name == "a2a.initiate"
            and entity_roles[cast(str, command.payload["actor_id"])] == "mule"
            and entity_roles[cast(str, command.payload["counterparty_id"])] == "attacker"
        )

    two_positions = cash_positions(commands_two)
    three_positions = cash_positions(commands_three)
    assert len(two_positions) >= 2
    assert len(three_positions) >= 2

    def delay(schedule: tuple[datetime, ...], index: int) -> int:
        timestamp = schedule[index]
        prior = population.generated_at if index == 0 else schedule[index - 1]
        return int((timestamp - prior).total_seconds())

    assert delay(audit_two.schedule, two_positions[0]) == 1
    assert delay(audit_two.schedule, two_positions[-1]) == 2
    assert delay(audit_three.schedule, three_positions[0]) == 1
    assert delay(audit_three.schedule, three_positions[-1]) == 3
    assert audit_two.schedule != audit_three.schedule

    last_cash = three_positions[-1]
    mutated_schedule = tuple(
        timestamp if index < last_cash else timestamp - timedelta(seconds=1)
        for index, timestamp in enumerate(audit_three.schedule)
    )
    with pytest.raises(GenerationConstraintError):
        evaluator.validate(
            "app_scam_mule",
            commands_three,
            population,
            replace(base, cash_out_delay_seconds=3),
            schedule=mutated_schedule,
        )


@pytest.mark.parametrize(
    ("family", "updates"),
    [
        (
            "card_testing_cnp",
            {"merchant_concentration": Decimal("0.700000000000000001")},
        ),
        (
            "card_testing_cnp",
            {"merchant_concentration": Decimal("0.10")},
        ),
        (
            "card_testing_cnp",
            {"device_reuse_rate": Decimal("0.600000000000000001")},
        ),
        (
            "app_scam_mule",
            {"cash_out_fraction": Decimal("0.300000000000000001")},
        ),
        (
            "agentic_intent_abuse",
            {"agentic_attack_mix": Decimal("0.920000000000000001")},
        ),
        (
            "agentic_intent_abuse",
            {"agentic_mutations": ("amount",)},
        ),
    ],
)
def test_other_adaptive_aliases_reject_instead_of_becoming_no_ops(
    family: str,
    updates: dict[str, object],
) -> None:
    population = PopulationGenerator(seed=41).generate(_bundle(41))
    with pytest.raises(GenerationConstraintError) as caught:
        _CampaignEvaluator(seed=41).generate(
            family,
            population,
            _params(family, seed=41, **updates),
        )
    assert caught.value.code == "GENERATION_CONSTRAINT_UNSATISFIED"
    assert caught.value.attempts == 100


def test_cash_out_strategies_cannot_share_the_minimum_delay_schedule() -> None:
    population = PopulationGenerator(seed=42).generate(_bundle(42))
    params = _params(
        "app_scam_mule",
        seed=42,
        min_delay_seconds=1,
        max_delay_seconds=300,
        cash_out_delay_seconds=1,
        cash_out_strategy="delayed",
    )
    with pytest.raises(GenerationConstraintError):
        _CampaignEvaluator(seed=42).generate("app_scam_mule", population, params)
