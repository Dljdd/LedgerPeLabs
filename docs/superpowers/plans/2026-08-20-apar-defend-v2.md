# APAR Defend v2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a sealed, synthetic-only Defend v2 protocol and pre-execution assurance path for fair rules-only, GBDT-only, and layered-hybrid evaluation—without executing a v2 evaluation.

**Architecture:** V2 is additive and versioned beside frozen v1 evidence. Evaluator-owned population code creates independent benign operating bases and injects sealed campaign events. Arm replay feeds case-aware workload and conservative multi-stratum gates; signed contracts render only truthful `not_executed` output until a separately authorized execution.

**Tech Stack:** Python 3.12, Pydantic 2, NumPy, pandas, PyArrow, CatBoost CPU, cryptography, FastAPI, pytest

**Spec:** `docs/superpowers/specs/2026-08-20-apar-defend-v2-design.md`

## Global Constraints

- Do not modify `docs/experiments/defense-v1-*.json`, `fixtures/defense/v1/`, or `tests/evaluation/test_frozen_defense_v1.py`.
- All data and reports are synthetic-only. Every report contains: `Synthetic-only evaluation; not a real-world prevalence or external-validity claim.`
- Do not generate, execute, reveal, or publish a v2 hidden population/evaluation. The permitted terminal state is verified `not_executed`.
- Preserve `available_at < decision_at` causality and campaign, entity, time, and family isolation.
- `apar.defense`, feature, calibration, and policy code may not import `apar.evaluation_hidden` or receive evaluator truth/secret seeds.
- Use existing dependencies, canonical JSON bytes, SHA-256, and existing Ed25519 primitives; never use pickle.
- Arms have identical observations, vectors, partitions, case grouping, latency environment, candidate-grid shape, and budgets. Decision logic and selected threshold values are the only differences.
- Operating profiles have 100,000 decision transactions over 28 synthetic days. Fraud transaction counts are low=100, medium=500, high=1,000, distributed equally across four families.
- Gates are recall `>=0.50`, ECE `<=0.10`, challenge `<=0.02`, false decline `<=0.001`, review cases/transaction `<=0.01`, p95 decision latency `<=50ms`, captured preventable settled value `>=0.50`, escaped value `<=0.50`, and p95 time-to-alert `<=300s`.
- Bootstrap exactly 2,000 times by synthetic day then campaign/entity case block; gate on the conservative 95% bound. Undefined mandatory metrics fail closed.
- One confirmatory hidden attempt is allowed. Any failure produces `no_promotion`; there is no retry, retuning, seed replacement, or metric switch.
- Before every commit run `git status --short`, stage task-owned paths only, and commit as `Dylan Moraes <dylanmoraesdljdd@gmail.com>` with no AI attribution.

---

## Locked File Map

```text
config/defense/competition-v2-profile.json             Immutable public protocol profile
src/apar/evaluation/v2_protocol.py                     Closed profile, stratum, seed, and v1-root contracts
src/apar/evaluation/v2_population.py                   Evaluator-owned benign base and campaign injector
src/apar/cases/v2_workload.py                          Case grouping and action-specific workload aggregation
src/apar/evaluation/v2_selection.py                    Matched threshold selection, bootstrap, and gates
src/apar/evaluation/v2_controls.py                     Benign-only and score-permutation controls
src/apar/evaluation/v2_preregistration.py              Canonical signed preregistration and one-attempt admission
src/apar/evaluation/v2_reporting.py                    Signed JSON/CSV scorecard renderer
src/apar/evaluation/v2_preexecution.py                 Read-only pre-execution verifier
scripts/verify_defense_v2_preexecution.py              Verifier CLI; never starts evaluation
src/apar/api/routes/defense.py                         Read-only v2 scorecard endpoint
tests/evaluation/test_defense_v2_*.py                  Protocol, population, selector, controls, registration, reporting, verifier tests
tests/cases/test_v2_workload.py                        Case and transaction denominator tests
tests/integration/test_defense_v2_preexecution.py      Sealed not-executed golden path
tests/api/test_defense.py                              V2 endpoint regression test
```

### Task 1: Create the sealed v2 protocol contract

**Files:**
- Create: `config/defense/competition-v2-profile.json`
- Create: `src/apar/evaluation/v2_protocol.py`
- Create: `tests/evaluation/test_defense_v2_protocol.py`

**Interfaces:**
- Produces: `PrevalenceStratum`, `OperatingPopulationProfile`, `V2Budget`, `V2Protocol`, `SeedCommitment`
- Produces: `load_v2_protocol(path: Path) -> V2Protocol`
- Produces: `verify_v1_roots(root: Path) -> None`

- [ ] **Step 1: Write failing profile and v1-isolation tests.**

```python
def test_production_strata_are_exact() -> None:
    p = load_v2_protocol(PROFILE)
    assert p.operating.transaction_count == 100_000
    assert [(s.name, s.fraud_transaction_count) for s in p.strata] == [
        ("low", 100), ("medium", 500), ("high", 1_000)
    ]

def test_v1_hash_mismatch_fails_closed(tmp_path: Path) -> None:
    root = copy_v1_roots(tmp_path)
    (root / "docs/experiments/defense-v1-result.json").write_bytes(b"changed")
    with pytest.raises(V2ProtocolError, match="frozen v1 root"):
        verify_v1_roots(root)
```

- [ ] **Step 2: Run the test.**

Run: `.venv/bin/python -m pytest tests/evaluation/test_defense_v2_protocol.py -q`

Expected: collection failure for `apar.evaluation.v2_protocol`.

- [ ] **Step 3: Implement closed Pydantic contracts.**

```python
class PrevalenceStratum(ExternalContract):
    name: Literal["low", "medium", "high"]
    transaction_count: int = Field(gt=0)
    fraud_transaction_count: int = Field(ge=0)
    family_transaction_counts: tuple[int, int, int, int]

    @model_validator(mode="after")
    def allocation_is_exact(self) -> "PrevalenceStratum":
        if sum(self.family_transaction_counts) != self.fraud_transaction_count:
            raise ValueError("invalid frozen family allocation")
        return self

class V2Protocol(ExternalContract):
    fixture_only: bool = False
    strata: tuple[PrevalenceStratum, PrevalenceStratum, PrevalenceStratum]

    @model_validator(mode="after")
    def production_values_are_exact(self) -> "V2Protocol":
        expected = (("low", 100), ("medium", 500), ("high", 1_000))
        actual = tuple((s.name, s.fraud_transaction_count) for s in self.strata)
        if not self.fixture_only and actual != expected:
            raise ValueError("invalid frozen production strata")
        return self
```

Require canonical JSON, reject unknown/duplicate strata, validate profile digest,
and compare every named v1 evidence root against hard-coded SHA-256 values.
V2Protocol enforces the three production denominators and fraud counts when
fixture_only is false; the test factory sets fixture_only true and cannot be
serialized as a preregistration input.

- [ ] **Step 4: Run the new and frozen-evidence tests.**

Run: `.venv/bin/python -m pytest tests/evaluation/test_defense_v2_protocol.py tests/evaluation/test_frozen_defense_v1.py -q`

Expected: PASS.

- [ ] **Step 5: Commit.**

```bash
git add config/defense/competition-v2-profile.json src/apar/evaluation/v2_protocol.py tests/evaluation/test_defense_v2_protocol.py
git -c user.name='Dylan Moraes' -c user.email='dylanmoraesdljdd@gmail.com' commit -m "feat: add sealed defend v2 protocol"
```

### Task 2: Build independent operating populations

**Files:**
- Create: `src/apar/evaluation/v2_population.py`
- Create: `tests/evaluation/test_defense_v2_population.py`

**Interfaces:**
- Produces: `OperatingPopulation`, `CampaignInjection`, `PopulationManifest`
- Produces: `build_benign_base(protocol: V2Protocol, *, seed: int) -> OperatingPopulation`
- Produces: `inject_frozen_campaigns(base: OperatingPopulation, injections: tuple[CampaignInjection, ...], stratum: PrevalenceStratum, *, seed: int) -> OperatingPopulation`

- [ ] **Step 1: Write failing denominator and isolation tests.**

```python
def test_injection_keeps_exact_denominator() -> None:
    base = build_benign_base(V2Protocol.fixture(transaction_count=100), seed=11)
    result = inject_frozen_campaigns(
        base, campaign_injections(total_decisions=20),
        PrevalenceStratum.fixture(100, 20), seed=17,
    )
    assert len(result.observations) == 100
    assert sum(row.is_fraud for row in result.truth) == 20

def test_entity_overlap_is_rejected() -> None:
    base = build_benign_base(V2Protocol.fixture(transaction_count=100), seed=11)
    injection = CampaignInjection.fixture(entity_ids=(base.observations[0].actor_id,))
    with pytest.raises(PopulationIsolationError, match="entity overlap"):
        inject_frozen_campaigns(base, (injection,), PrevalenceStratum.fixture(), seed=17)
```

- [ ] **Step 2: Run `.venv/bin/python -m pytest tests/evaluation/test_defense_v2_population.py -q` and confirm collection failure.**

- [ ] **Step 3: Implement evaluator-owned population code.**

```python
@dataclass(frozen=True, slots=True)
class OperatingPopulation:
    observations: tuple[ObservedEvent, ...]
    truth: tuple[EvaluationTruthRow, ...]
    manifest: PopulationManifest

def inject_frozen_campaigns(
    base: OperatingPopulation,
    injections: tuple[CampaignInjection, ...],
    stratum: PrevalenceStratum,
    *,
    seed: int,
) -> OperatingPopulation:
    """Replace exactly stratum.fraud_transaction_count benign decisions."""
```

Generate benign observations independently, retain evaluator truth separately, replace exactly required decision rows, preserve campaign context, and reject duplicate IDs, non-benign bases, wrong family allocation, and campaign/entity/time overlap. Store seed commitments, never evaluator seeds, in manifests.

- [ ] **Step 4: Run `.venv/bin/python -m pytest tests/evaluation/test_defense_v2_population.py -q` and confirm PASS for low/medium/high allocations, 28 days, determinism, and disjointness.**

- [ ] **Step 5: Commit.**

```bash
git add src/apar/evaluation/v2_population.py tests/evaluation/test_defense_v2_population.py
git -c user.name='Dylan Moraes' -c user.email='dylanmoraesdljdd@gmail.com' commit -m "feat: add defend v2 operating populations"
```

### Task 3: Add case-aware workload metrics

**Files:**
- Create: `src/apar/cases/v2_workload.py`
- Create: `tests/cases/test_v2_workload.py`

**Interfaces:**
- Produces: `ReviewCase`, `ActionWorkload`
- Produces: `group_review_cases(events: Sequence[ObservedEvent], *, window: timedelta = timedelta(hours=24)) -> tuple[ReviewCase, ...]`
- Produces: `aggregate_action_workload(...) -> ActionWorkload`

- [ ] **Step 1: Write failing separate-denominator tests.**

```python
def test_two_transactions_in_one_window_are_one_review_case() -> None:
    events = (
        observed("a", actor="actor-1", at="2026-01-01T10:00:00Z"),
        observed("b", actor="actor-1", at="2026-01-01T10:05:00Z"),
    )
    m = aggregate_action_workload(100, group_review_cases(events), review_actions("a", "b"), truth_for(events))
    assert (m.review_case_count, m.reviewed_transaction_count, m.review_case_rate) == (1, 2, 0.01)

def test_false_decline_rate_uses_legitimate_denominator() -> None:
    m = aggregate_action_workload(100, (), declines("legitimate-1"), truth_with(legitimate=80))
    assert m.false_decline_rate == 1 / 80
```

- [ ] **Step 2: Run `.venv/bin/python -m pytest tests/cases/test_v2_workload.py -q` and confirm collection failure.**

- [ ] **Step 3: Implement immutable workload contracts.**

```python
class ActionWorkload(ExternalContract):
    total_transaction_count: int
    legitimate_transaction_count: int
    review_case_count: int
    reviewed_transaction_count: int
    challenge_count: int
    automatic_integrity_decline_count: int
    false_decline_count: int
    false_intervention_count: int
    review_case_rate: float
    challenge_rate: float
    false_decline_rate: float | None
    false_interventions_per_10k: float
```

Define a case by frozen entity key and a 24-hour UTC window. Count review by case, other interventions by transaction, and false intervention as legitimate challenge plus legitimate decline.

- [ ] **Step 4: Run `.venv/bin/python -m pytest tests/cases/test_v2_workload.py -q` and confirm PASS for boundaries, per-day volumes, action precedence, and zero legitimate denominator.**

- [ ] **Step 5: Commit.**

```bash
git add src/apar/cases/v2_workload.py tests/cases/test_v2_workload.py
git -c user.name='Dylan Moraes' -c user.email='dylanmoraesdljdd@gmail.com' commit -m "feat: add case-aware defend v2 workload metrics"
```

### Task 4: Select thresholds under matched conservative gates

**Files:**
- Create: `src/apar/evaluation/v2_selection.py`
- Create: `tests/evaluation/test_defense_v2_selection.py`

**Interfaces:**
- Produces: `BoundedMetric`, `V2MetricSet`, `ArmThresholdCandidate`, `V2GateOutcome`, `V2SelectionReport`
- Produces: `bootstrap_v2_metrics(...) -> Mapping[str, BoundedMetric]`
- Produces: `select_v2_thresholds(candidates: Sequence[ArmThresholdCandidate], protocol: V2Protocol) -> V2SelectionReport`

- [ ] **Step 1: Write failing all-strata and confidence-bound veto tests.**

```python
def test_high_stratum_review_failure_rejects_candidate() -> None:
    r = select_v2_thresholds(
        (candidate("safe"), candidate("high-review", high_review_rate=0.0101)),
        protocol(),
    )
    assert r.selected_candidate_id == "safe"

def test_upper_bound_vetoes_maximum_gate() -> None:
    outcome = evaluate_v2_gates(
        metrics_with(challenge=BoundedMetric(point=0.019, lower=0.017, upper=0.021)),
        protocol(),
    )
    assert outcome.passed is False
    assert "CHALLENGE_BUDGET" in outcome.codes
```

- [ ] **Step 2: Run `.venv/bin/python -m pytest tests/evaluation/test_defense_v2_selection.py -q` and confirm collection failure.**

- [ ] **Step 3: Implement bootstrap, gates, and tie-break.**

```python
class V2MetricSet(ExternalContract):
    precision: BoundedMetric
    recall: BoundedMetric
    f1: BoundedMetric
    pr_auc: BoundedMetric
    roc_auc: BoundedMetric
    ece: BoundedMetric
    fpr: BoundedMetric
    false_interventions_per_10k: BoundedMetric
    preventable_settled_value_fraction: BoundedMetric
    escaped_value_fraction: BoundedMetric
    time_to_alert_p95_seconds: BoundedMetric

def select_v2_thresholds(
    candidates: Sequence[ArmThresholdCandidate], protocol: V2Protocol
) -> V2SelectionReport:
    feasible = [c for c in candidates if c.gates.passed]
    if not feasible:
        return V2SelectionReport.no_promotion("no_candidate_satisfies_v2_constraints")
    return min(
        feasible,
        key=lambda c: (
            -c.minimum_family_captured_value_lower_bound,
            c.maximum_review_case_rate_upper_bound,
            c.maximum_false_decline_rate_upper_bound,
            c.maximum_challenge_rate_upper_bound,
            c.p95_decision_latency_upper_bound,
            c.threshold_tuple,
        ),
    )
```

Project existing classification, calibration, value, and alert evidence into
`V2MetricSet`; keep every numerator and denominator in the metric artifact.
Resample day and then campaign/entity case blocks for exactly 2,000 replicates.
Apply lower bounds to minimum gates and upper bounds to maximum gates. Reject any
undefined metric and any stratum/family/control failure.

- [ ] **Step 4: Run `.venv/bin/python -m pytest tests/evaluation/test_defense_v2_selection.py -q` and confirm PASS for deterministic ties, family/value/time gates, and no-promotion.**

- [ ] **Step 5: Commit.**

```bash
git add src/apar/evaluation/v2_selection.py tests/evaluation/test_defense_v2_selection.py
git -c user.name='Dylan Moraes' -c user.email='dylanmoraesdljdd@gmail.com' commit -m "feat: add defend v2 constrained selection gates"
```

### Task 5: Add mandatory negative controls

**Files:**
- Create: `src/apar/evaluation/v2_controls.py`
- Create: `tests/evaluation/test_defense_v2_controls.py`

**Interfaces:**
- Produces: `ControlResult`, `run_benign_only_control(...) -> ControlResult`, `run_score_permutation_control(...) -> ControlResult`

- [ ] **Step 1: Write failing control tests.**

```python
def test_benign_control_reports_interventions_without_true_positives() -> None:
    c = run_benign_only_control(actions=challenge_actions("row-1"), truth=benign_truth("row-1"))
    assert c.valid is True and c.true_positive_count == 0

def test_qualifying_permuted_scores_invalidates_run() -> None:
    c = run_score_permutation_control(scores=perfect_scores(), truth=truth_rows(), blocks=case_blocks(), seed=7)
    assert (c.valid, c.reason) == (False, "permuted_scores_qualified")
```

- [ ] **Step 2: Run `.venv/bin/python -m pytest tests/evaluation/test_defense_v2_controls.py -q` and confirm collection failure.**

- [ ] **Step 3: Implement block-preserving controls.**

```python
def run_score_permutation_control(
    *, scores: np.ndarray, truth: Sequence[EvaluationTruthRow],
    blocks: Sequence[str], seed: int
) -> ControlResult:
    permutation = np.random.default_rng(seed).permutation(np.unique(blocks))
    return evaluate_control_scores(permute_scores_by_block(scores, blocks, permutation), truth)
```

Keep time/case blocks intact. A malformed control or apparent permutation efficacy pass invalidates the entire run.

- [ ] **Step 4: Run `.venv/bin/python -m pytest tests/evaluation/test_defense_v2_controls.py -q` and confirm PASS.**

- [ ] **Step 5: Commit.**

```bash
git add src/apar/evaluation/v2_controls.py tests/evaluation/test_defense_v2_controls.py
git -c user.name='Dylan Moraes' -c user.email='dylanmoraesdljdd@gmail.com' commit -m "feat: add defend v2 negative controls"
```

### Task 6: Seal preregistration and one-attempt admission

**Files:**
- Create: `src/apar/evaluation/v2_preregistration.py`
- Create: `tests/evaluation/test_defense_v2_preregistration.py`

**Interfaces:**
- Produces: `V2Preregistration`, `ExecutionAdmission`
- Produces: `sign_v2_preregistration(...) -> V2Preregistration`
- Produces: `admit_v2_execution(preregistration: V2Preregistration, *, existing_receipts: Sequence[ExecutionReceipt]) -> ExecutionAdmission`

- [ ] **Step 1: Write failing completeness and retry tests.**

```python
def test_missing_seed_commitments_is_rejected() -> None:
    payload = complete_preregistration_payload()
    del payload["seed_commitments"]
    with pytest.raises(V2PreregistrationError, match="seed_commitments"):
        V2Preregistration.model_validate(payload)

def test_second_admission_is_denied() -> None:
    a = admit_v2_execution(signed_preregistration(), existing_receipts=(completed_receipt(),))
    assert (a.admitted, a.reason) == (False, "maximum_confirmatory_attempts_exhausted")
```

- [ ] **Step 2: Run `.venv/bin/python -m pytest tests/evaluation/test_defense_v2_preregistration.py -q` and confirm collection failure.**

- [ ] **Step 3: Implement canonical signing and admission.**

```python
def admit_v2_execution(
    preregistration: V2Preregistration, *,
    existing_receipts: Sequence[ExecutionReceipt],
) -> ExecutionAdmission:
    if len(existing_receipts) >= preregistration.maximum_confirmatory_attempts:
        return ExecutionAdmission.denied("maximum_confirmatory_attempts_exhausted")
    if not preregistration.verify_signature() or not preregistration.verify_manifest_bindings():
        return ExecutionAdmission.denied("invalid_preregistration")
    return ExecutionAdmission.admitted_once(preregistration.execution_nonce)
```

Bind source, feature, candidate-grid, population, seed, evaluator-key/capability, metric, bootstrap, control, schema, fidelity-validation bundle (if used), and synthetic-scope digests. Do not accept a caller-provided approval boolean.

- [ ] **Step 4: Run `.venv/bin/python -m pytest tests/evaluation/test_defense_v2_preregistration.py -q` and confirm PASS.**

- [ ] **Step 5: Commit.**

```bash
git add src/apar/evaluation/v2_preregistration.py tests/evaluation/test_defense_v2_preregistration.py
git -c user.name='Dylan Moraes' -c user.email='dylanmoraesdljdd@gmail.com' commit -m "feat: seal defend v2 preregistration"
```

### Task 7: Render signed stable not-executed scorecards

**Files:**
- Create: `src/apar/evaluation/v2_reporting.py`
- Create: `tests/evaluation/test_defense_v2_reporting.py`

**Interfaces:**
- Produces: `DefenseV2Scorecard`, `DefenseV2GateReport`
- Produces: `render_v2_scorecard(...) -> tuple[DefenseV2Scorecard, dict[str, bytes]]`

- [ ] **Step 1: Write failing visibility and CSV-schema tests.**

```python
def test_not_executed_lists_all_arms_without_hidden_values() -> None:
    card, files = render_v2_scorecard(not_executed_result(), signer=signer())
    assert card.status == "not_executed"
    assert [a.arm for a in card.arms] == ["rules_only", "gbdt_only", "layered_hybrid"]
    assert "hidden_seed" not in files["defense-v2-scorecard.json"].decode()

def test_workload_csv_has_both_denominators() -> None:
    _, files = render_v2_scorecard(not_executed_result(), signer=signer())
    assert b"review_cases,reviewed_transactions,review_case_rate,review_transaction_rate" in files["defense-v2-workload.csv"].splitlines()[0]
```

- [ ] **Step 2: Run `.venv/bin/python -m pytest tests/evaluation/test_defense_v2_reporting.py -q` and confirm collection failure.**

- [ ] **Step 3: Implement signed canonical public artifacts.**

```python
class DefenseV2Scorecard(ExternalContract):
    schema_version: Literal["2.0.0"] = "2.0.0"
    status: Literal["not_executed", "no_promotion", "promotion_eligible"]
    protocol_digest: str
    synthetic_scope: Literal["Synthetic-only evaluation; not a real-world prevalence or external-validity claim."]
    arms: tuple[V2ArmScorecard, V2ArmScorecard, V2ArmScorecard]
    signature_base64: str
```

Render `defense-v2-scorecard.json`, `defense-v2-arm-metrics.csv`, `defense-v2-workload.csv`, `defense-v2-gates.json`, and `defense-v2-limitations.md`. Keep all arm rows and headers visible. Never omit a failed gate.

- [ ] **Step 4: Run `.venv/bin/python -m pytest tests/evaluation/test_defense_v2_reporting.py -q` and confirm PASS.**

- [ ] **Step 5: Commit.**

```bash
git add src/apar/evaluation/v2_reporting.py tests/evaluation/test_defense_v2_reporting.py
git -c user.name='Dylan Moraes' -c user.email='dylanmoraesdljdd@gmail.com' commit -m "feat: add defend v2 scorecard contracts"
```

### Task 8: Add read-only pre-execution verification and public status

**Files:**
- Create: `src/apar/evaluation/v2_preexecution.py`
- Create: `scripts/verify_defense_v2_preexecution.py`
- Modify: `src/apar/api/routes/defense.py`
- Create: `tests/evaluation/test_defense_v2_preexecution.py`
- Create: `tests/integration/test_defense_v2_preexecution.py`
- Modify: `tests/api/test_defense.py`

**Interfaces:**
- Produces: `verify_v2_preexecution(root: Path, preregistration: V2Preregistration) -> PreexecutionReport`
- Produces: `GET /defense/v2/scorecard -> DefenseV2Scorecard`

- [ ] **Step 1: Write failing safety and endpoint tests.**

```python
def test_hidden_import_in_defender_fails_preexecution(tmp_path: Path) -> None:
    write_file(tmp_path / "src/apar/defense/bad.py", "from apar.evaluation_hidden import worker\n")
    assert "HIDDEN_IMPORT_BOUNDARY" in verify_v2_preexecution(tmp_path, signed_preregistration()).codes

def test_endpoint_exposes_only_public_not_executed_contract(client: TestClient) -> None:
    response = client.get("/defense/v2/scorecard")
    assert response.status_code == 200
    assert response.json()["status"] == "not_executed"
    assert "hidden_seed" not in response.text
```

- [ ] **Step 2: Run `.venv/bin/python -m pytest tests/evaluation/test_defense_v2_preexecution.py tests/integration/test_defense_v2_preexecution.py tests/api/test_defense.py -q` and confirm failure.**

- [ ] **Step 3: Implement a verifier that cannot execute evaluation.**

```python
def verify_v2_preexecution(root: Path, preregistration: V2Preregistration) -> PreexecutionReport:
    return PreexecutionReport.from_checks((
        verify_v1_roots(root),
        verify_protocol_digest(preregistration),
        verify_no_v2_execution_receipt(root),
        verify_import_boundary(root, forbidden="apar.evaluation_hidden", allowed_prefix="apar.evaluation.v2_"),
        verify_preregistration(preregistration),
    ))
```

The CLI may validate and render only `not_executed`; it must not import a hidden worker, call a generator, resolve hidden references, or write evaluation artifacts. The endpoint reads a signed scorecard and never starts work.

- [ ] **Step 4: Run the complete v2 safety suite.**

Run: `.venv/bin/python -m pytest tests/evaluation/test_defense_v2_protocol.py tests/evaluation/test_defense_v2_population.py tests/cases/test_v2_workload.py tests/evaluation/test_defense_v2_selection.py tests/evaluation/test_defense_v2_controls.py tests/evaluation/test_defense_v2_preregistration.py tests/evaluation/test_defense_v2_reporting.py tests/evaluation/test_defense_v2_preexecution.py tests/integration/test_defense_v2_preexecution.py tests/api/test_defense.py tests/evaluation/test_frozen_defense_v1.py -q`

Expected: PASS; verifier returns `not_executed`, every breach fails closed, and v1 remains byte-verified.

- [ ] **Step 5: Commit.**

```bash
git add src/apar/evaluation/v2_preexecution.py scripts/verify_defense_v2_preexecution.py src/apar/api/routes/defense.py tests/evaluation/test_defense_v2_preexecution.py tests/integration/test_defense_v2_preexecution.py tests/api/test_defense.py
git -c user.name='Dylan Moraes' -c user.email='dylanmoraesdljdd@gmail.com' commit -m "feat: verify defend v2 pre-execution gates"
```

### Task 9: Document verified non-execution and run repository checks

**Files:**
- Modify: `README.md`
- Modify: `docs/TRACEABILITY.md`
- Modify: `tests/evaluation/test_defense_v2_preexecution.py`

**Interfaces:**
- Produces: exact README status `Defend v2: protocol sealed; evaluation not executed.`

- [ ] **Step 1: Write the failing documentation-state test.**

```python
def test_readme_makes_no_v2_efficacy_claim() -> None:
    text = README.read_text().lower()
    assert "defend v2: protocol sealed; evaluation not executed" in text
    assert "defend v2 achieved" not in text
```

- [ ] **Step 2: Run `.venv/bin/python -m pytest tests/evaluation/test_defense_v2_preexecution.py::test_readme_makes_no_v2_efficacy_claim -q` and confirm failure.**

- [ ] **Step 3: Add only the approved status sentence and a traceability row.**

Add: `Defend v2: protocol sealed; evaluation not executed. Any future result remains synthetic-only and is not a real-world prevalence or external-validity claim.` Do not edit v1 status, hashes, conclusions, or result files.

- [ ] **Step 4: Run full verification without evaluation execution.**

Run sequentially:

```bash
.venv/bin/python -m pytest -q
.venv/bin/python scripts/verify_g3.py
.venv/bin/python scripts/verify_defense_v2_preexecution.py
```

Expected: PASS; v2 verifier reports `not_executed` and creates no evaluation receipt/result artifact.

- [ ] **Step 5: Commit.**

```bash
git add README.md docs/TRACEABILITY.md tests/evaluation/test_defense_v2_preexecution.py
git -c user.name='Dylan Moraes' -c user.email='dylanmoraesdljdd@gmail.com' commit -m "docs: record defend v2 pre-execution status"
```

## Plan Self-Review

| Requirement | Tasks |
| --- | --- |
| v1 byte preservation and truthful no-promotion | 1, 8, 9 |
| Independent efficacy/workload populations and frozen strata | 1, 2 |
| Action-specific case and transaction denominators | 3, 4, 7 |
| Matched three-arm selection and fixed budgets | 4 |
| Uncertainty, controls, and one attempt | 4, 5, 6 |
| Hidden authority separation and pre-execution gate | 2, 6, 8 |
| Stable judge-facing contracts and synthetic scope | 7, 8, 9 |
| Fidelity validation may bind only before signing | 6: validation-bundle digest is required if used |

No task generates or evaluates a v2 population. The only post-implementation command is a read-only pre-execution verifier.
