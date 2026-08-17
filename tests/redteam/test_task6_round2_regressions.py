"""Task 6 round-two authority, authenticity, deadline, and value regressions."""

from __future__ import annotations

import copy
import inspect
from decimal import Decimal

import pytest

from apar.contracts.decisions import Action
from apar.redteam import (
    AdaptiveSearch,
    CapabilityPreregistration,
    FamilyThreshold,
    Feedback,
    FixedPolicy,
    PrimaryOutcome,
    SearchAuthority,
    capability_delta_report,
)
from apar.redteam.benchmark import CampaignBenchmark, role_bound_settled_value
from tests.redteam.test_round1_regressions import _evaluation_contract, _tiny_bounds
from tests.simulator.test_a2a_rail import (
    OTHER_PAYMENT_ID as OTHER_A2A_PAYMENT_ID,
)
from tests.simulator.test_a2a_rail import (
    PAYEE_ID as A2A_PAYEE_ID,
)
from tests.simulator.test_a2a_rail import (
    PAYER_ID as A2A_PAYER_ID,
)
from tests.simulator.test_a2a_rail import (
    PAYMENT_ID as A2A_PAYMENT_ID,
)
from tests.simulator.test_a2a_rail import _engine as _a2a_engine
from tests.simulator.test_a2a_rail import _initiate
from tests.simulator.test_a2a_rail import _schedule as _schedule_a2a
from tests.simulator.test_card_rail import OTHER_PAYMENT_ID as OTHER_CARD_PAYMENT_ID
from tests.simulator.test_card_rail import PAYEE_ID as CARD_PAYEE_ID
from tests.simulator.test_card_rail import PAYER_ID as CARD_PAYER_ID
from tests.simulator.test_card_rail import PAYMENT_ID as CARD_PAYMENT_ID
from tests.simulator.test_card_rail import _authorize
from tests.simulator.test_card_rail import _engine as _card_engine
from tests.simulator.test_card_rail import _schedule as _schedule_card


class _Clock:
    now = 0

    def __call__(self) -> int:
        return self.now


class _Evaluator:
    def __init__(self, clock: _Clock | None = None, advance_ns: int = 0) -> None:
        self.clock = clock
        self.advance_ns = advance_ns

    def evaluate(self, _candidate) -> Feedback:  # type: ignore[no-untyped-def]
        if self.clock is not None:
            self.clock.now += self.advance_ns
        return Feedback(
            action=Action.APPROVE,
            reason_family="approved",
            realized_value=None,
        )


def _issued_search(
    *,
    authority: SearchAuthority | None = None,
    clock: _Clock | None = None,
    advance_ns: int = 0,
    policy_name: str = "fixed",
    run_group=None,  # type: ignore[no-untyped-def]
):  # type: ignore[no-untyped-def]
    owner = authority or SearchAuthority()
    bounds = _tiny_bounds()
    evaluator = _Evaluator(clock, advance_ns)
    evaluator_capability = owner.register_evaluator(
        owner=evaluator,
        bounds=bounds,
        evaluation_contract=_evaluation_contract(bounds),
        evaluate=evaluator.evaluate,
        dependency_digest="6" * 64,
    )
    policy_capability = owner.register_policy(
        FixedPolicy(),
        name=policy_name,
        version="test-policy-v1",
    )
    group = run_group or owner.issue_run_group("round2-regression")
    search = AdaptiveSearch(
        evaluator_capability=evaluator_capability,
        policy_capability=policy_capability,
        run_group=group,
        clock_ns=clock or _Clock(),
    )
    return owner, evaluator_capability, policy_capability, group, search


def test_search_accepts_only_authority_issued_evaluator_and_policy_capabilities(
    card_benchmark: CampaignBenchmark,
) -> None:
    authority = SearchAuthority()
    evaluator = card_benchmark.issue_evaluator_capability(authority)
    policy = FixedPolicy()
    policy_capability = authority.register_policy(
        policy,
        name="fixed",
        version="1.0.0",
    )
    group = authority.issue_run_group("exact-authority")

    search = AdaptiveSearch(
        evaluator_capability=evaluator,
        policy_capability=policy_capability,
        run_group=group,
    )
    result = search.search(seed=1, budget=1, wall_time_budget_ms=60_000)

    assert "evaluate" not in inspect.signature(search.search).parameters
    assert result.evaluator_code_digest == evaluator.evaluator_code_digest
    assert result.policy_code_digest == policy_capability.policy_code_digest
    assert result.policy_name == "fixed"
    with pytest.raises(TypeError):
        search.search(
            seed=1,
            budget=1,
            wall_time_budget_ms=60_000,
            evaluate=card_benchmark.evaluate,  # type: ignore[call-arg]
        )
    with pytest.raises(ValueError, match="issued|authority|capability"):
        AdaptiveSearch(
            evaluator_capability=copy.copy(evaluator),
            policy_capability=policy_capability,
            run_group=group,
        )


def test_cross_authority_capabilities_and_mutable_policy_self_description_reject() -> None:
    first = SearchAuthority()
    second = SearchAuthority()
    _, evaluator, policy, group, _search = _issued_search(authority=first)
    foreign_policy = second.register_policy(
        FixedPolicy(),
        name="fixed",
        version="foreign-v1",
    )
    with pytest.raises(ValueError, match="authority"):
        AdaptiveSearch(
            evaluator_capability=evaluator,
            policy_capability=foreign_policy,
            run_group=group,
        )

    registered = FixedPolicy()
    capability = first.register_policy(
        registered,
        name="fixed",
        version="registered-v1",
    )
    registered.policy_name = "adaptive"  # type: ignore[misc]
    registered.policy_version = "forged"  # type: ignore[misc]
    exact_search = AdaptiveSearch(
        evaluator_capability=evaluator,
        policy_capability=capability,
        run_group=group,
        clock_ns=_Clock(),
    )
    result = exact_search.search(seed=1, budget=1, wall_time_budget_ms=100)
    assert result.policy_name == "fixed"
    assert result.policy_version == "registered-v1"


def _issued_comparison(
    *,
    authority: SearchAuthority | None = None,
    run_group=None,  # type: ignore[no-untyped-def]
    budget: int = 1,
) -> tuple[
    SearchAuthority,
    CapabilityPreregistration,
    dict[str, dict[str, tuple[object, ...]]],
]:
    owner = authority or SearchAuthority()
    bounds = _tiny_bounds()
    evaluator_owner = _Evaluator()
    evaluator = owner.register_evaluator(
        owner=evaluator_owner,
        bounds=bounds,
        evaluation_contract=_evaluation_contract(bounds),
        evaluate=evaluator_owner.evaluate,
        dependency_digest="7" * 64,
    )
    group = run_group or owner.issue_run_group("comparison")
    policy_capabilities = tuple(
        owner.register_policy(
            FixedPolicy(),
            name=name,
            version=f"{name}-test-v1",
        )
        for name in ("adaptive", "cached_llm", "fixed", "random")
    )
    threshold = FamilyThreshold(
        family=bounds.family,
        primary_outcome=PrimaryOutcome.VALID_YIELD,
        minimum_delta=Decimal("0.01"),
        evaluation_contract=evaluator.evaluation_contract,
        evaluator_capability_id=evaluator.capability_id,
        evaluator_code_digest=evaluator.evaluator_code_digest,
    )
    preregistration = owner.issue_preregistration(
        run_group=group,
        seeds=(1,),
        budget=budget,
        wall_time_budget_ms=100,
        thresholds=(threshold,),
        policies=policy_capabilities,
    )
    cells: dict[str, tuple[object, ...]] = {}
    for policy in policy_capabilities:
        result = AdaptiveSearch(
            evaluator_capability=evaluator,
            policy_capability=policy,
            run_group=group,
            clock_ns=_Clock(),
        ).search(seed=1, budget=budget, wall_time_budget_ms=100)
        cells[policy.name] = (result,)
    return owner, preregistration, {bounds.family: cells}


def test_report_rejects_model_copy_synthetic_and_deep_trial_mutation() -> None:
    authority, preregistration, results = _issued_comparison()
    fixed = results["card_testing_cnp"]["fixed"][0]
    forged = fixed.model_copy(update={"policy_name": "random"})  # type: ignore[union-attr]
    results["card_testing_cnp"]["fixed"] = (forged,)
    with pytest.raises(ValueError, match="issued|seal|authentic"):
        capability_delta_report(preregistration, results, authority=authority)  # type: ignore[arg-type]

    authority, preregistration, results = _issued_comparison()
    fixed = results["card_testing_cnp"]["fixed"][0]
    trial = fixed.trials[0]  # type: ignore[union-attr]
    object.__setattr__(trial, "objective_value", Decimal("999.00"))
    with pytest.raises(ValueError, match="seal|authentic|pristine"):
        capability_delta_report(preregistration, results, authority=authority)  # type: ignore[arg-type]


def test_report_rejects_cross_authority_and_cross_run_results() -> None:
    first, preregistration, results = _issued_comparison()
    _second, _foreign_preregistration, foreign_results = _issued_comparison()
    results["card_testing_cnp"]["fixed"] = foreign_results["card_testing_cnp"]["fixed"]
    with pytest.raises(ValueError, match="authority|issued"):
        capability_delta_report(preregistration, results, authority=first)  # type: ignore[arg-type]

    first, preregistration, results = _issued_comparison()
    foreign_group = first.issue_run_group("other-run")
    _, _, foreign_run_results = _issued_comparison(
        authority=first,
        run_group=foreign_group,
    )
    results["card_testing_cnp"]["fixed"] = foreign_run_results[
        "card_testing_cnp"
    ]["fixed"]
    with pytest.raises(ValueError, match="run group"):
        capability_delta_report(preregistration, results, authority=first)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("elapsed_ns", "elapsed_ms", "overrun_ms"),
    ((5_000_000, 5, 0), (6_000_000, 6, 1)),
)
def test_final_proposal_at_or_past_deadline_is_exhausted(
    elapsed_ns: int,
    elapsed_ms: int,
    overrun_ms: int,
) -> None:
    clock = _Clock()
    _authority, _evaluator, _policy, _group, search = _issued_search(
        clock=clock,
        advance_ns=elapsed_ns,
    )

    result = search.search(seed=1, budget=1, wall_time_budget_ms=5)

    assert result.proposals_used == 1
    assert result.wall_time_elapsed_ms == elapsed_ms
    assert result.wall_time_exhausted is True
    assert result.wall_time_overrun_ms == overrun_ms


def test_report_rejects_unequal_actual_opportunities_and_deadline_exhaustion() -> None:
    authority, preregistration, results = _issued_comparison(budget=2)
    evaluator = authority.evaluator_capability(
        preregistration.thresholds[0].evaluator_capability_id
    )
    adaptive_binding = next(
        binding
        for binding in preregistration.policy_bindings
        if binding.name == "adaptive"
    )
    adaptive = authority.policy_capability(adaptive_binding.capability_id)
    clock = _Clock()
    clock.now = 100_000_000
    exhausted = AdaptiveSearch(
        evaluator_capability=evaluator,
        policy_capability=adaptive,
        run_group=authority.run_group(preregistration.run_group_id),
        clock_ns=clock,
    ).search(seed=1, budget=2, wall_time_budget_ms=0)
    results["card_testing_cnp"]["adaptive"] = (exhausted,)

    with pytest.raises(ValueError, match="actual usage|exhausted|matched"):
        capability_delta_report(preregistration, results, authority=authority)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "terminal_commands",
    (
        ("chargeback",),
        ("chargeback", "recovery"),
        ("refund",),
    ),
)
def test_card_terminal_principal_is_subtracted_at_most_once(
    terminal_commands: tuple[str, ...],
) -> None:
    from apar.simulator.rails.card import (
        ChargebackCard,
        ClearCard,
        OpenCardDispute,
        RecoverCard,
        RefundCard,
        ReportCardFraud,
        SettleCard,
    )

    engine = _card_engine()
    first = [
        _authorize(),
        ClearCard(CARD_PAYMENT_ID),
        SettleCard(CARD_PAYMENT_ID),
    ]
    if "chargeback" in terminal_commands:
        first.extend(
            [
                ReportCardFraud(CARD_PAYMENT_ID),
                OpenCardDispute(CARD_PAYMENT_ID),
                ChargebackCard(CARD_PAYMENT_ID),
            ]
        )
    if "recovery" in terminal_commands:
        first.append(RecoverCard(CARD_PAYMENT_ID))
    if "refund" in terminal_commands:
        first.append(RefundCard(CARD_PAYMENT_ID))
    second = [
        _authorize(
            payment_id=OTHER_CARD_PAYMENT_ID,
            amount=Decimal("20.00"),
            idempotency_key="second-card-open",
        ),
        ClearCard(OTHER_CARD_PAYMENT_ID),
        SettleCard(OTHER_CARD_PAYMENT_ID),
    ]
    _schedule_card(engine, *(first + second))

    events = engine.run()

    assert role_bound_settled_value(
        events,
        frozenset({CARD_PAYER_ID, CARD_PAYEE_ID}),
    ) == Decimal("20.00")


def test_app_recovery_preserves_unrecovered_posted_principal() -> None:
    from apar.simulator.rails.a2a import (
        AcceptA2A,
        FreezeA2AFunds,
        PostA2A,
        RecoverA2A,
        ReportA2AFraud,
    )

    engine = _a2a_engine()
    _schedule_a2a(
        engine,
        _initiate(),
        AcceptA2A(A2A_PAYMENT_ID),
        PostA2A(A2A_PAYMENT_ID),
        ReportA2AFraud(A2A_PAYMENT_ID),
        FreezeA2AFunds(A2A_PAYMENT_ID),
        RecoverA2A(A2A_PAYMENT_ID),
        _initiate(
            payment_id=OTHER_A2A_PAYMENT_ID,
            amount=Decimal("20.00"),
            idempotency_key="second-a2a-open",
        ),
        AcceptA2A(OTHER_A2A_PAYMENT_ID),
        PostA2A(OTHER_A2A_PAYMENT_ID),
    )

    events = engine.run()

    assert role_bound_settled_value(
        events,
        frozenset({A2A_PAYER_ID, A2A_PAYEE_ID}),
    ) == Decimal("20.00")
