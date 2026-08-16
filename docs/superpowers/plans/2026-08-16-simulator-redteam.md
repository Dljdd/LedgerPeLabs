# Stateful Simulator and Adaptive Red-Team Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build rail-correct, stateful synthetic payment campaigns and bounded attackers that produce reproducible, economically valid event streams.

**Architecture:** A deterministic discrete-event engine owns clocks, queues, balances, lifecycle state, and campaign causality. Rail adapters implement card, A2A, and agentic semantics; attacker policies propose declared parameter changes and never mutate simulator state directly.

**Tech Stack:** Python 3.12, Pydantic v2, NumPy, pandas, Hypothesis, pytest

**Spec:** `SOLUTION_SPEC.md`, `docs/04-simulation-and-red-team.md`, `docs/05-defense-and-agentic-trust.md`, `docs/06-data-and-api-contracts.md`

## Global Constraints

- Consume the exact contracts defined in `2026-08-16-foundation-contracts.md`.
- Use a local `numpy.random.Generator` created from the scenario seed; never use global random state.
- Preserve event time, arrival time, and decision time as distinct UTC timestamps.
- Conserve value under declared fees, settlements, reversals, refunds, and transfer posting.
- Reject impossible lifecycle transitions before emitting an event.
- Give attackers only `action` and `reason_family` in the golden scenario.
- Keep hidden validity logic under `src/apar/evaluation_hidden/` and forbid imports from `src/apar/defense/`.
- Store every completed run as immutable scenario, event, decision-feedback, and manifest artifacts.

---

## Target file map

```text
src/apar/simulator/clock.py               Deterministic simulation clock and queue
src/apar/simulator/ledger.py              Balance and value-conservation ledger
src/apar/simulator/engine.py              Discrete-event orchestration
src/apar/simulator/rails/base.py          Rail-adapter protocol
src/apar/simulator/rails/card.py          Authorization-to-settlement lifecycle
src/apar/simulator/rails/a2a.py           Transfer initiation-to-posting lifecycle
src/apar/simulator/rails/agentic.py       Delegated payment lifecycle
src/apar/trust/verifier.py                Deterministic agentic integrity checks
src/apar/generators/population.py         Entity and benign-background generation
src/apar/generators/campaigns.py          Four deep campaign families
src/apar/redteam/policies.py              Fixed, random, and adaptive policy protocol
src/apar/redteam/search.py                Decision-only adaptive search
src/apar/redteam/llm_policy.py            Schema-constrained optional agent planner
src/apar/evaluation_hidden/generator.py   Separately implemented hidden campaigns
src/apar/evaluation_hidden/validity.py    Independent validity oracle
src/apar/runs/runner.py                   Run orchestration and artifact freezing
src/apar/api/routes/scenarios.py           Compilation endpoints
src/apar/api/routes/runs.py                Simulation and replay endpoints
tests/simulator/                           Lifecycle and property tests
tests/redteam/                             Search, visibility, and validity tests
tests/integration/test_g1_simulation.py    Multi-rail invariant gate
tests/integration/test_g2_adaptation.py    Adaptation and hidden-generator gate
```

### Task 1: Implement the deterministic clock, event queue, and ledger

**Files:**
- Create: `src/apar/simulator/__init__.py`
- Create: `src/apar/simulator/clock.py`
- Create: `src/apar/simulator/ledger.py`
- Create: `tests/simulator/test_clock.py`
- Create: `tests/simulator/test_ledger.py`

**Interfaces:**
- Produces: `SimulationClock.schedule(at: datetime, priority: int, command: Command) -> None`
- Produces: `SimulationClock.pop() -> ScheduledCommand`
- Produces: `Ledger.post(entry: LedgerEntry) -> None`
- Produces: `Ledger.assert_conserved() -> None`

- [ ] **Step 1: Write ordering and conservation tests**

```python
def test_queue_orders_by_time_priority_then_sequence(clock, now) -> None:
    clock.schedule(now, 2, Command("second"))
    clock.schedule(now, 1, Command("first"))
    clock.schedule(now, 1, Command("first-tie"))
    assert [clock.pop().command.name for _ in range(3)] == ["first", "first-tie", "second"]


def test_ledger_rejects_unbalanced_posting(ledger) -> None:
    with pytest.raises(ValueError, match="debits must equal credits"):
        ledger.post(LedgerEntry("e1", debit={"payer": Decimal("10")}, credit={"payee": Decimal("9")}))
```

- [ ] **Step 2: Run the tests and confirm missing simulator modules**

Run: `python -m pytest tests/simulator/test_clock.py tests/simulator/test_ledger.py -q`

Expected: collection fails because clock and ledger modules are absent.

- [ ] **Step 3: Implement stable queue ordering and double-entry posting**

Use a heap key `(at, priority, sequence)` with an incrementing integer sequence. The ledger must quantize each amount to the currency exponent, reject negative posting legs, require equal debit and credit totals, and keep append-only entries.

- [ ] **Step 4: Run deterministic and Hypothesis tests**

Run: `python -m pytest tests/simulator/test_clock.py tests/simulator/test_ledger.py -q`

Expected: ordering, duplicate-time, rounding, balanced-entry, and randomized conservation tests pass.

- [ ] **Step 5: Commit the simulation primitives**

```bash
git add src/apar/simulator tests/simulator/test_clock.py tests/simulator/test_ledger.py
git commit -m "feat: add deterministic simulation clock and ledger"
```

### Task 2: Define the engine and rail-adapter protocol

**Files:**
- Create: `src/apar/simulator/rails/__init__.py`
- Create: `src/apar/simulator/rails/base.py`
- Create: `src/apar/simulator/engine.py`
- Create: `tests/simulator/test_engine.py`

**Interfaces:**
- Consumes: `PaymentEvent`, `ScenarioBundle`, `SimulationClock`, `Ledger`
- Produces: `RailAdapter.initialize(engine: SimulationEngine) -> None`
- Produces: `RailAdapter.handle(command: Command, engine: SimulationEngine) -> list[PaymentEvent]`
- Produces: `SimulationEngine.run(until: datetime) -> tuple[PaymentEvent, ...]`

- [ ] **Step 1: Write engine determinism and duplicate-ID tests**

```python
def test_same_seed_produces_byte_identical_events(engine_factory) -> None:
    first = [event.model_dump(mode="json") for event in engine_factory(seed=260816).run()]
    second = [event.model_dump(mode="json") for event in engine_factory(seed=260816).run()]
    assert json.dumps(first, sort_keys=True, separators=(",", ":")) == json.dumps(second, sort_keys=True, separators=(",", ":"))


def test_duplicate_event_id_is_rejected(engine, event) -> None:
    engine.emit(event)
    with pytest.raises(ValueError, match="duplicate event_id"):
        engine.emit(event)
```

- [ ] **Step 2: Verify engine tests fail before implementation**

Run: `python -m pytest tests/simulator/test_engine.py -q`

Expected: collection fails because `SimulationEngine` does not exist.

- [ ] **Step 3: Implement run ownership and append-only emission**

The engine owns the clock, ledger, seeded generator, adapter map, entity state, emitted-event tuple, and seen-ID set. A rail adapter can request future commands and ledger entries through engine methods but cannot access the heap or artifact store directly.

- [ ] **Step 4: Run engine tests and serialization replay**

Run: `python -m pytest tests/simulator/test_engine.py -q`

Expected: determinism, duplicate rejection, monotonic queue time, unsupported rail, and serialize/replay tests pass.

- [ ] **Step 5: Commit the adapter boundary**

```bash
git add src/apar/simulator/rails src/apar/simulator/engine.py tests/simulator/test_engine.py
git commit -m "feat: add simulation engine and rail protocol"
```

### Task 3: Implement card and A2A payment lifecycles

**Files:**
- Create: `src/apar/simulator/rails/card.py`
- Create: `src/apar/simulator/rails/a2a.py`
- Create: `tests/simulator/test_card_rail.py`
- Create: `tests/simulator/test_a2a_rail.py`

**Interfaces:**
- Produces: `CardRailAdapter`
- Produces: `A2ARailAdapter`
- Card states: `created -> authorized -> cleared -> settled -> reported -> dispute -> chargeback -> recovery`, with reversal before settlement and refund after settlement
- A2A states: `created -> initiated -> accepted -> posted -> reported -> frozen -> recovered`, with rejection before posting and return after posting

- [ ] **Step 1: Write legal and illegal transition tests**

```python
def test_card_cannot_settle_before_clearing(card_engine) -> None:
    card_engine.schedule(SettleCard("p1"))
    with pytest.raises(LifecycleError) as error:
        card_engine.run()
    assert error.value.code == "CARD_SETTLE_BEFORE_CLEAR"


def test_a2a_posting_moves_value_once(a2a_engine) -> None:
    events = a2a_engine.run()
    assert [event.event_type.value for event in events] == ["transfer_initiated", "transfer_posted"]
    assert a2a_engine.ledger.balance("payer") == Decimal("90.00")
    assert a2a_engine.ledger.balance("payee") == Decimal("10.00")
```

- [ ] **Step 2: Run the rail tests and confirm adapters are missing**

Run: `python -m pytest tests/simulator/test_card_rail.py tests/simulator/test_a2a_rail.py -q`

Expected: collection fails for missing card and A2A adapters.

- [ ] **Step 3: Implement transition tables and ledger effects**

Represent each rail transition as a map from `(current_state, command_type)` to `(next_state, emitted_kind, ledger_effect)`. Keep declined authorizations and rejected transfers out of the value ledger. Reverse unsettled card holds without creating settlement value. Model reports, disputes, chargebacks, recoveries, returns, and refunds as new linked events rather than mutation.

- [ ] **Step 4: Run lifecycle property tests**

Run: `python -m pytest tests/simulator/test_card_rail.py tests/simulator/test_a2a_rail.py -q`

Expected: every legal path, illegal transition, idempotent duplicate command, fee, reversal, refund, rejection, posting, and return test passes.

- [ ] **Step 5: Commit two rail-correct adapters**

```bash
git add src/apar/simulator/rails/card.py src/apar/simulator/rails/a2a.py tests/simulator/test_card_rail.py tests/simulator/test_a2a_rail.py
git commit -m "feat: model card and A2A payment lifecycles"
```

### Task 4: Implement the agentic trust verifier and rail adapter

**Files:**
- Create: `src/apar/trust/__init__.py`
- Create: `src/apar/trust/verifier.py`
- Create: `src/apar/simulator/rails/agentic.py`
- Create: `tests/trust/test_verifier.py`
- Create: `tests/simulator/test_agentic_rail.py`

**Interfaces:**
- Produces: `AgentMandate`, `AgentPaymentRequest`, `IntegrityReceipt`
- Produces: `TrustVerifier.verify(request: AgentPaymentRequest, now: datetime) -> IntegrityReceipt`
- Check order: identity, signature, mandate, amount and currency, payee binding, cart binding, expiry, nonce replay, receipt chain
- Agentic adapter calls `verify` before any risk-score callback

- [ ] **Step 1: Write fail-closed and ordering tests**

```python
def test_payee_substitution_fails_before_risk_scoring(valid_request, agentic_adapter, scorer) -> None:
    tampered = valid_request.model_copy(update={"payee_id": "attacker"})
    decision = agentic_adapter.process(tampered, now=valid_request.created_at)
    assert decision.action == Action.DECLINE
    assert decision.reason_codes == ["PAYEE_BINDING_MISMATCH"]
    scorer.assert_not_called()


def test_nonce_replay_is_rejected(valid_request, verifier) -> None:
    assert verifier.verify(valid_request, valid_request.created_at).allowed
    second = verifier.verify(valid_request, valid_request.created_at)
    assert second.reason_code == "NONCE_REPLAY"
```

- [ ] **Step 2: Confirm trust and rail tests fail before implementation**

Run: `python -m pytest tests/trust tests/simulator/test_agentic_rail.py -q`

Expected: collection fails because the trust verifier is missing.

- [ ] **Step 3: Implement deterministic verification and receipts**

Use Ed25519 verification through `cryptography`. Hash the canonical mandate, cart, payee, and prior receipt into the signed request. Store consumed nonces in run state. Return a signed-looking synthetic receipt hash for the range without generating or exporting real payment credentials.

- [ ] **Step 4: Run integrity attack suites**

Run: `python -m pytest tests/trust tests/simulator/test_agentic_rail.py -q`

Expected: identity spoofing, mandate escalation, amount change, currency change, payee substitution, cart substitution, expiry, nonce replay, and receipt-chain break all reject with stable reason codes before the scorer is called.

- [ ] **Step 5: Commit the agentic trust plane**

```bash
git add src/apar/trust src/apar/simulator/rails/agentic.py tests/trust tests/simulator/test_agentic_rail.py
git commit -m "feat: add fail-closed agentic payment integrity"
```

### Task 5: Generate populations and four deep campaign families

**Files:**
- Create: `src/apar/generators/__init__.py`
- Create: `src/apar/generators/population.py`
- Create: `src/apar/generators/campaigns.py`
- Create: `tests/generators/test_population.py`
- Create: `tests/generators/test_campaigns.py`

**Interfaces:**
- Produces: `PopulationGenerator.generate(bundle: ScenarioBundle) -> Population`
- Produces: `CampaignGenerator.generate(family: str, population: Population, params: CampaignParams) -> tuple[Command, ...]`
- Families: `app_scam_mule`, `card_testing_cnp`, `synthetic_merchant_refund`, `agentic_intent_abuse`

- [ ] **Step 1: Write reproducibility, motif, and class-rate tests**

```python
@pytest.mark.parametrize("family", [
    "app_scam_mule", "card_testing_cnp", "synthetic_merchant_refund", "agentic_intent_abuse"
])
def test_campaign_has_declared_entities_and_motif(family, population, params) -> None:
    commands = CampaignGenerator(seed=260816).generate(family, population, params)
    assert commands
    assert all(command.campaign_id == params.campaign_id for command in commands)
    assert motif_signature(commands) == params.expected_motif
```

- [ ] **Step 2: Confirm generator tests fail before implementation**

Run: `python -m pytest tests/generators -q`

Expected: collection fails because generator modules are absent.

- [ ] **Step 3: Implement conditional leaf sampling under causal schedules**

Generate entity graphs and campaign schedules first. Sample amounts, delays, devices, channels, and merchant attributes conditionally within those schedules. Do not generate independent rows. Apply bounded rejection sampling until class rate, amount total, and graph motif constraints are inside the scenario tolerances or raise `GENERATION_CONSTRAINT_UNSATISFIED` after 100 attempts.

- [ ] **Step 4: Run multi-seed fidelity checks**

Run: `python -m pytest tests/generators -q`

Expected: five fixed seeds pass reproducibility, benign-shift, class-rate, value-total, campaign-motif, and rail-lifecycle assertions.

- [ ] **Step 5: Commit executable campaign generation**

```bash
git add src/apar/generators tests/generators
git commit -m "feat: generate stateful payment campaign families"
```

### Task 6: Add fixed, random, adaptive non-LLM, and schema-constrained LLM attackers

**Files:**
- Create: `src/apar/redteam/__init__.py`
- Create: `src/apar/redteam/policies.py`
- Create: `src/apar/redteam/search.py`
- Create: `src/apar/redteam/llm_policy.py`
- Create: `tests/redteam/test_policies.py`
- Create: `tests/redteam/test_search.py`
- Create: `tests/redteam/test_llm_policy.py`
- Create: `tests/redteam/test_capability_delta.py`

**Interfaces:**
- Produces: `AttackCandidate(params: CampaignParams, parent_id: str | None, generation: int)`
- Produces: `Feedback(action: Action, reason_family: str, realized_value: Decimal | None)`; `realized_value` is null unless the scenario explicitly exposes it
- Produces: `AdaptiveSearch.search(seed: int, budget: int, evaluate: Callable[[AttackCandidate], Feedback]) -> SearchResult`
- Produces: `LLMPlannerPolicy.propose(history: tuple[VisibleTrial, ...], bounds: ParameterBounds) -> AttackCandidate`
- Produces: `CapabilityDeltaReport(family_metrics, supported_family_count, matched_budgets)`
- Search result includes all proposed candidates, visible rejections, feedback, objective values, and winner

- [ ] **Step 1: Write feedback-isolation and matched-budget tests**

```python
def test_adaptive_search_cannot_read_scores_or_features() -> None:
    fields = set(Feedback.model_fields)
    assert fields == {"action", "reason_family", "realized_value"}


def test_random_and_adaptive_receive_equal_budget(search_factory) -> None:
    random_result = search_factory("random", budget=40)
    adaptive_result = search_factory("adaptive", budget=40)
    assert len(random_result.proposals) == len(adaptive_result.proposals) == 40


def test_llm_policy_rejects_undeclared_output(fake_llm, bounds) -> None:
    fake_llm.response = {"params": {"amount": 50}, "model_score": 0.9}
    with pytest.raises(ValueError, match="undeclared planner field"):
        LLMPlannerPolicy(fake_llm).propose((), bounds)


def test_at_least_two_families_show_a_measurable_capability_delta(ablation_report) -> None:
    assert ablation_report.matched_budgets is True
    assert ablation_report.supported_family_count >= 2


def test_adaptive_claim_matches_observed_result(ablation_report) -> None:
    if ablation_report.adaptive_net_value <= ablation_report.random_net_value:
        assert ablation_report.adaptive_claim == "not_supported"
    else:
        assert ablation_report.adaptive_claim == "supported"
```

- [ ] **Step 2: Run red-team tests and confirm missing policies**

Run: `python -m pytest tests/redteam -q`

Expected: collection fails for missing policy and search modules.

- [ ] **Step 3: Implement bounded mutation and decision-only selection**

Fixed policy returns declared defaults. Random policy samples each allowed parameter uniformly or log-uniformly within the scenario bounds. Adaptive non-LLM policy uses tournament selection over prior valid candidates, mutates one to three parameters, and optimizes settled illicit value minus visible rejection penalties. The optional LLM policy receives the same visible trial history and parameter JSON schema, records provider, model ID, prompt digest, response digest, latency, and token usage, and must support cached replay with no network. No planner receives model score, feature value, threshold, gradients, hidden validity reasons, or future outcomes.

- [ ] **Step 4: Run ablation and metamorphic tests**

Run: `python -m pytest tests/redteam -q`

Expected: fixed, random, adaptive non-LLM, and cached LLM runs are reproducible; undeclared parameter mutation is rejected; candidate ordering changes do not alter results; a feedback object with an extra field is rejected; matched query, wall-time, and proposal budgets are enforced; at least two deep families show a preregistered capability delta in valid attack yield, net settled value, adaptation speed, or campaign scale.

- [ ] **Step 5: Commit bounded adaptive search**

```bash
git add src/apar/redteam tests/redteam
git commit -m "feat: add bounded decision-only adaptive attackers"
```

### Task 7: Add an independent hidden validity oracle and freeze run artifacts

**Files:**
- Create: `src/apar/evaluation_hidden/__init__.py`
- Create: `src/apar/evaluation_hidden/generator.py`
- Create: `src/apar/evaluation_hidden/validity.py`
- Create: `src/apar/runs/__init__.py`
- Create: `src/apar/runs/runner.py`
- Create: `src/apar/api/routes/scenarios.py`
- Create: `src/apar/api/routes/runs.py`
- Modify: `src/apar/api/app.py`
- Create: `tests/redteam/test_hidden_validity.py`
- Create: `tests/integration/test_g1_simulation.py`
- Create: `tests/integration/test_g2_adaptation.py`
- Create: `scripts/verify_g1_g2.py`

**Interfaces:**
- Produces: `HiddenValidityOracle.evaluate(events: tuple[PaymentEvent, ...]) -> HiddenValidityResult`
- Produces: `HiddenCampaignGenerator.generate(family: str, seed: int, count: int) -> tuple[PaymentEvent, ...]`
- Produces: `RunRunner.execute(bundle: ScenarioBundle, policy: AttackerPolicy) -> RunManifest`
- Produces: `POST /api/v1/scenarios/compile`, `POST /api/v1/runs`, `GET /api/v1/runs/{run_id}`
- `RunManifest` references immutable scenario, population, events, feedback, validity, and summary artifacts

- [ ] **Step 1: Write hidden-boundary and artifact-manifest tests**

```python
def test_hidden_package_does_not_import_defender() -> None:
    forbidden = scan_imports(Path("src/apar/evaluation_hidden"), prefix="apar.defense")
    assert forbidden == []


def test_hidden_generator_does_not_import_main_generator() -> None:
    forbidden = scan_imports(Path("src/apar/evaluation_hidden"), prefix="apar.generators")
    assert forbidden == []


def test_completed_run_manifest_resolves_every_artifact(runner, bundle, fixed_policy) -> None:
    manifest = runner.execute(bundle, fixed_policy)
    for ref in manifest.artifacts.values():
        assert runner.artifact_store.read(ref)
```

- [ ] **Step 2: Run G1 and G2 tests before orchestration exists**

Run: `python -m pytest tests/integration/test_g1_simulation.py tests/integration/test_g2_adaptation.py -q`

Expected: collection fails because hidden validity and run orchestration are absent.

- [ ] **Step 3: Implement independent validity and run freezing**

Implement hidden campaigns from independent entity-motif templates, schedule code, parameter names, and leaf distributions under `evaluation_hidden/generator.py`; it must not import the main population or campaign generators. The hidden oracle checks value conservation, lifecycle legality, account balance feasibility, parameter bounds, campaign connectivity, declared actor roles, and maximum benign-distribution distance. Return only `valid: bool` to the attacker path; store detailed internal reasons in a restricted evaluation artifact after the run. Freeze all run inputs before execution and all outputs after completion. The API routes accept only compiled scenario artifact IDs and return typed run manifests, never raw hidden-oracle reasons.

- [ ] **Step 4: Execute the multi-rail G1/G2 gate**

Run: `python scripts/verify_g1_g2.py`

Expected output includes `G1 PASS` for three rail invariants and `G2 PASS` for four campaign families, matched random/adaptive budgets, hidden validity separation, and byte-identical seeded reruns.

Run: `python -m pytest tests/simulator tests/trust tests/generators tests/redteam tests/integration/test_g1_simulation.py tests/integration/test_g2_adaptation.py -q`

Expected: all tests pass.

- [ ] **Step 5: Commit the G1/G2 deliverable**

```bash
git add src/apar/evaluation_hidden src/apar/runs src/apar/api/routes/scenarios.py src/apar/api/routes/runs.py src/apar/api/app.py tests/redteam/test_hidden_validity.py tests/integration scripts/verify_g1_g2.py
git commit -m "test: establish G1 and G2 simulation gates"
```

## Plan completion gate

G1 and G2 are complete when the fixed seed produces byte-identical artifacts, all three rail lifecycles conserve value through report and recovery stages, all agentic integrity attacks fail closed, all four campaign families satisfy hidden validity, the independently implemented hidden generator passes its import boundary, at least two families pass the preregistered GenAI capability-delta test, and fixed, random, adaptive non-LLM, and cached LLM planners are evaluated at matched budgets without access to scores, features, gradients, or hidden rejection reasons. If adaptive search does not beat random search, the run report must mark the adaptive claim `not_supported` rather than hiding the result.
