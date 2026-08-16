# Defense, Evaluation, and Governance Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build leakage-safe layered defenses, campaign investigation, hidden evaluation, and human-controlled promotion reports.

**Architecture:** Deterministic integrity and policy checks precede past-only feature computation, calibrated risk scoring, and action selection. A separate evaluation package freezes defenders, applies time and entity isolation, measures operational value, and produces immutable promotion evidence.

**Tech Stack:** Python 3.12, pandas, NumPy, scikit-learn, CatBoost, NetworkX, FastAPI, pytest, Hypothesis

**Spec:** `SOLUTION_SPEC.md`, `docs/05-defense-and-agentic-trust.md`, `docs/06-data-and-api-contracts.md`, `docs/07-evaluation-and-validation.md`, `docs/08-security-and-governance.md`

## Global Constraints

- Consume immutable run artifacts from the simulator plan; do not regenerate training events inside defender code.
- Compute decision features only from rows whose `source_timestamp < decision_timestamp`.
- Equal-time events must not observe one another.
- Fit preprocessors, graph statistics, calibrators, and thresholds on training data only.
- Keep campaign IDs intact across partitions and report returning-entity and cold-entity results separately.
- Evaluate rules, transaction features, temporal features, and graph features before accepting a neural challenger.
- Select actions under declared false-decline, challenge-rate, case-volume, and latency budgets.
- Average precision is diagnostic; primary metrics are preventable settled value, friction, workload, time to alert, and reconstruction quality.
- Promotion is impossible without a human approver identity and a rollback artifact.

---

## Target file map

```text
src/apar/features/state.py                  Past-only online state
src/apar/features/catalog.py                Availability, provenance, and privacy contract
src/apar/features/builders.py               Transaction, velocity, and graph features
src/apar/features/parity.py                 Online and offline parity checks
config/features/catalog.yaml                Completed feature availability matrix
src/apar/defense/rules.py                   Deterministic payment-risk rules
src/apar/defense/models.py                  Scorer protocol and CatBoost adapter
src/apar/defense/calibration.py             Training-only calibration
src/apar/defense/actions.py                 Budgeted action policy
src/apar/cases/grouping.py                  Campaign reconstruction
src/apar/cases/queue.py                     Investigator prioritization
src/apar/evaluation/splits.py               Time and entity isolation
src/apar/evaluation/metrics.py              Value, friction, case, and latency metrics
src/apar/evaluation/runner.py               Champion/challenger evaluation
src/apar/evaluation/gates.py                Promotion gates
src/apar/governance/report.py               Signed-off assurance report
src/apar/api/routes/runs.py                 Run and evaluation endpoints
src/apar/api/routes/reports.py              Report and promotion endpoints
tests/features/                              Leakage and parity tests
tests/defense/                               Baseline and action tests
tests/cases/                                 Reconstruction tests
tests/evaluation/                            Split, metric, and gate tests
tests/governance/                            Approval and rollback tests
```

### Task 1: Build strict past-only feature state

**Files:**
- Create: `src/apar/features/__init__.py`
- Create: `src/apar/features/state.py`
- Create: `src/apar/features/catalog.py`
- Create: `src/apar/features/builders.py`
- Create: `src/apar/features/parity.py`
- Create: `config/features/catalog.yaml`
- Create: `tests/features/test_state.py`
- Create: `tests/features/test_parity.py`

**Interfaces:**
- Produces: `FeatureDefinition(name, version, rails, source_event_types, entity_key, window, availability_rule, missing_behavior, freshness_sla_ms, online, privacy_purpose, forbidden_sources)`
- Produces: `FeatureVector(event_id: str, decision_time: datetime, max_source_timestamp: datetime, source_event_ids: tuple[str, ...], values: dict[str, float])`
- Produces: `FeatureState.observe(event: PaymentEvent) -> None`
- Produces: `FeatureState.compute(event: PaymentEvent, decision_time: datetime) -> FeatureVector`
- Produces: `FeatureState.checkpoint() -> bytes` and `FeatureState.restore(payload: bytes) -> FeatureState`
- Produces: `build_offline(events: Sequence[PaymentEvent], decision_times: Mapping[str, datetime]) -> list[FeatureVector]`

- [ ] **Step 1: Write future-append and equal-time tests**

```python
def test_future_append_cannot_change_prior_vector(event_stream, decision_times) -> None:
    before = build_offline(event_stream[:5], decision_times)
    after = build_offline(event_stream, decision_times)
    assert before == after[:5]


def test_equal_time_events_do_not_observe_each_other(feature_state, equal_time_events, now) -> None:
    vectors = [feature_state.compute(event, now) for event in equal_time_events]
    assert all(vector.values["actor_count_1m"] == 0 for vector in vectors)
```

- [ ] **Step 2: Confirm feature tests fail before state exists**

Run: `python -m pytest tests/features -q`

Expected: collection fails for missing feature modules.

- [ ] **Step 3: Implement two-phase timestamp processing**

For each timestamp group, compute every feature vector against state ending strictly before the timestamp, then observe the group. Implement amount deviation, actor and counterparty velocities over 1 minute, 10 minutes, 1 hour, and 24 hours, unique devices, unique counterparties, decline bursts, account age, prior trust failures, and prior graph-motif counts. Record the maximum source timestamp and every source event ID used for each vector. Complete the feature catalog row for every feature, reject forbidden label, campaign, scenario, and generator sources, and serialize checkpoints with schema version and last processed availability timestamp.

- [ ] **Step 4: Run leakage, ordering, and parity tests**

Run: `python -m pytest tests/features -q`

Expected: future append, equal time, out-of-order arrival, late correction, checkpoint plus replay, online/offline parity, completed availability matrix, source-event provenance, and feature-name allowlist tests pass.

- [ ] **Step 5: Commit the past-only feature boundary**

```bash
git add src/apar/features tests/features
git commit -m "feat: add past-only feature state and parity checks"
```

### Task 2: Implement deterministic rules and budgeted action policy

**Files:**
- Create: `src/apar/defense/__init__.py`
- Create: `src/apar/defense/rules.py`
- Create: `src/apar/defense/actions.py`
- Create: `tests/defense/test_rules.py`
- Create: `tests/defense/test_actions.py`

**Interfaces:**
- Produces: `RuleEngine.evaluate(event: PaymentEvent, features: FeatureVector) -> RuleResult`
- Produces: `ActionPolicy.choose(score: float, integrity: IntegrityReceipt, rules: RuleResult) -> Action`
- Stable actions: `approve`, `monitor`, `challenge`, `hold`, `decline`
- Stable reason families: `integrity`, `velocity`, `amount`, `device`, `counterparty`, `graph`, `policy`

- [ ] **Step 1: Write precedence and budget tests**

```python
def test_integrity_rejection_overrides_low_model_score(action_policy, failed_receipt) -> None:
    action = action_policy.choose(0.01, failed_receipt, RuleResult.clear())
    assert action == Action.DECLINE


def test_policy_respects_challenge_budget(calibration_rows) -> None:
    policy = ActionPolicy.fit(calibration_rows, challenge_rate_max=0.02, decline_rate_max=0.001)
    actions = [policy.choose(row.score, row.integrity, row.rules) for row in calibration_rows]
    assert fraction(actions, Action.CHALLENGE) <= 0.02
    assert fraction(actions, Action.DECLINE) <= 0.001
```

- [ ] **Step 2: Verify rule and action tests fail**

Run: `python -m pytest tests/defense/test_rules.py tests/defense/test_actions.py -q`

Expected: collection fails because defense modules are absent.

- [ ] **Step 3: Implement precedence and transparent rule outputs**

Apply integrity failure first, then hard policy constraints, then deterministic fraud rules, then calibrated score bands. Every non-approve decision includes one to five stable reason codes ordered by contribution and precedence. Fit thresholds on calibration rows only, under the configured challenge and decline caps. If scoring exceeds the configured timeout or the model artifact cannot be loaded, execute deterministic policy, set `fallback_used=True`, and retain the timeout or load-failure reason in the audit envelope.

- [ ] **Step 4: Run policy behavior and stability tests**

Run: `python -m pytest tests/defense/test_rules.py tests/defense/test_actions.py -q`

Expected: integrity precedence, rule ordering, threshold monotonicity, fixed-budget behavior, stable reason ordering, and empty-feature degraded-path tests pass.

- [ ] **Step 5: Commit the deterministic defense layers**

```bash
git add src/apar/defense/rules.py src/apar/defense/actions.py tests/defense
git commit -m "feat: add transparent rules and budgeted actions"
```

### Task 3: Train and calibrate strong GBDT baselines

**Files:**
- Create: `src/apar/defense/models.py`
- Create: `src/apar/defense/calibration.py`
- Create: `tests/defense/test_models.py`
- Create: `tests/defense/test_calibration.py`

**Interfaces:**
- Produces: `Scorer.fit(frame: DataFrame, labels: Series) -> Scorer`
- Produces: `Scorer.predict(frame: DataFrame) -> ndarray[float64]`
- Produces: `ModelArtifact(feature_names, train_end, library_version, seed, payload_ref)`
- Produces: `Calibrator.fit(raw_scores, labels) -> Calibrator`

- [ ] **Step 1: Write training-boundary and reproducibility tests**

```python
def test_model_rejects_forbidden_feature_names(training_frame) -> None:
    training_frame["future_chargeback"] = 1
    with pytest.raises(ValueError, match="forbidden feature"):
        CatBoostScorer(seed=260816).fit(training_frame, training_frame.pop("label"))


def test_seeded_model_predictions_are_reproducible(train_frame, test_frame) -> None:
    first = CatBoostScorer(seed=260816).fit(train_frame.x, train_frame.y).predict(test_frame)
    second = CatBoostScorer(seed=260816).fit(train_frame.x, train_frame.y).predict(test_frame)
    np.testing.assert_allclose(first, second, rtol=0, atol=1e-12)
```

- [ ] **Step 2: Confirm model tests fail before adapters exist**

Run: `python -m pytest tests/defense/test_models.py tests/defense/test_calibration.py -q`

Expected: collection fails for missing model and calibration modules.

- [ ] **Step 3: Implement three declared baselines**

Implement transaction/entity GBDT, temporal-feature GBDT, and graph-feature GBDT using CatBoost with fixed seed, class weights computed from training rows, and no automatic time-derived features. Add logistic and isotonic calibration, selecting the method by lower calibration-set Brier score. Freeze feature names and library versions with each model artifact.

- [ ] **Step 4: Run reproducibility and calibration tests**

Run: `python -m pytest tests/defense/test_models.py tests/defense/test_calibration.py -q`

Expected: forbidden-feature, seed, train-only preprocessing, serialization parity, score bounds, Brier improvement, and unseen-category tests pass.

- [ ] **Step 5: Commit the strong model baselines**

```bash
git add src/apar/defense/models.py src/apar/defense/calibration.py tests/defense/test_models.py tests/defense/test_calibration.py
git commit -m "feat: add calibrated GBDT baselines"
```

### Task 4: Reconstruct campaigns and prioritize investigator cases

**Files:**
- Create: `src/apar/cases/__init__.py`
- Create: `src/apar/cases/grouping.py`
- Create: `src/apar/cases/queue.py`
- Create: `tests/cases/test_grouping.py`
- Create: `tests/cases/test_queue.py`

**Interfaces:**
- Produces: `CampaignGrouper.group(events, decisions) -> tuple[Case, ...]`
- Produces: `Case(case_id, entity_ids, event_ids, first_alert_time, escaped_value, motif, priority)`
- Produces: `CaseQueue.rank(cases, analyst_capacity: int) -> tuple[Case, ...]`

- [ ] **Step 1: Write motif and analyst-capacity tests**

```python
def test_shared_device_and_payee_join_events_into_one_case(events, decisions) -> None:
    cases = CampaignGrouper().group(events, decisions)
    assert len(cases) == 1
    assert cases[0].motif == "shared_device_to_mule"
    assert set(cases[0].event_ids) == {event.event_id for event in events}


def test_case_queue_never_exceeds_capacity(cases) -> None:
    ranked = CaseQueue().rank(cases, analyst_capacity=3)
    assert len(ranked) == 3
    assert list(ranked) == sorted(ranked, key=lambda case: (-case.priority, case.case_id))
```

- [ ] **Step 2: Confirm case tests fail before grouping exists**

Run: `python -m pytest tests/cases -q`

Expected: collection fails for missing case modules.

- [ ] **Step 3: Implement causal graph grouping and value-aware ranking**

Build a typed NetworkX multigraph over accounts, devices, merchants, beneficiaries, agents, carts, and events. Add only edges known by the case decision time. Group connected suspicious components with deterministic union-find and derive named motifs. Rank cases using preventable value, confidence, entity coverage, recency, and analyst effort; break ties by `case_id`.

- [ ] **Step 4: Run reconstruction and workload tests**

Run: `python -m pytest tests/cases -q`

Expected: shared-device, fan-in, fan-out, cycle, refund-loop, unknown-entity, time cutoff, deterministic grouping, and capacity tests pass.

- [ ] **Step 5: Commit campaign-level investigation**

```bash
git add src/apar/cases tests/cases
git commit -m "feat: reconstruct and prioritize fraud campaigns"
```

### Task 5: Implement time-respecting splits and operational metrics

**Files:**
- Create: `src/apar/evaluation/__init__.py`
- Create: `src/apar/evaluation/splits.py`
- Create: `src/apar/evaluation/metrics.py`
- Create: `tests/evaluation/test_splits.py`
- Create: `tests/evaluation/test_metrics.py`

**Interfaces:**
- Produces: `EvaluationSplit(train_ids, calibration_ids, development_ids, hidden_ids, cohort_labels)`
- Produces: `make_split(events, train_end, calibration_end, development_end, cold_fraction, seed) -> EvaluationSplit`
- Produces: `compute_metrics(events, decisions, cases, budgets) -> MetricReport`

- [ ] **Step 1: Write isolation and value-metric tests**

```python
def test_campaign_ids_never_cross_partitions(split, campaign_by_event) -> None:
    partitions = [split.train_ids, split.calibration_ids, split.development_ids, split.hidden_ids]
    campaign_sets = [{campaign_by_event[event_id] for event_id in ids} for ids in partitions]
    assert all(left.isdisjoint(right) for index, left in enumerate(campaign_sets) for right in campaign_sets[index + 1:])


def test_preventable_settled_value_excludes_predecision_loss(metric_fixture) -> None:
    report = compute_metrics(**metric_fixture)
    assert report.preventable_settled_value == Decimal("80.00")
    assert report.value_moved_before_first_alert == Decimal("20.00")
```

- [ ] **Step 2: Confirm evaluation tests fail before split and metrics modules**

Run: `python -m pytest tests/evaluation/test_splits.py tests/evaluation/test_metrics.py -q`

Expected: collection fails for missing evaluation modules.

- [ ] **Step 3: Implement four partitions and cohort reporting**

Assign whole campaigns by their first event time. Within development and hidden periods, deterministically mark returning entities and cold accounts, devices, merchants, and beneficiaries. Compute preventable settled value, recall at action-specific false-positive caps, false declines, challenge rate, value moved before alert, time to alert, campaign reconstruction precision/recall, cases per 100,000 decisions, value per analyst-hour, Brier score, calibration error, and p50/p95/p99 latency.

- [ ] **Step 4: Run split mutation and metric oracle tests**

Run: `python -m pytest tests/evaluation/test_splits.py tests/evaluation/test_metrics.py -q`

Expected: campaign isolation, time boundaries, cold cohorts, deterministic seed, zero-denominator behavior, hand-calculated value, workload, latency, calibration, and reconstruction tests pass.

- [ ] **Step 5: Commit leakage-safe evaluation foundations**

```bash
git add src/apar/evaluation/splits.py src/apar/evaluation/metrics.py tests/evaluation
git commit -m "feat: add isolated splits and operational metrics"
```

### Task 6: Build champion/challenger evaluation and promotion gates

**Files:**
- Create: `src/apar/evaluation/runner.py`
- Create: `src/apar/evaluation/gates.py`
- Create: `tests/evaluation/test_runner.py`
- Create: `tests/evaluation/test_gates.py`

**Interfaces:**
- Produces: `EvaluationRunner.evaluate(run_refs, hidden_run_refs, defenders, split) -> EvaluationBundle`
- Produces: `PromotionGates.evaluate(bundle: EvaluationBundle) -> GateReport`
- Gates: value improvement, friction, workload, per-family minimum, leakage, hidden generator, calibration, reason stability, latency, rollback, human approval

- [ ] **Step 1: Write no-average-masking and freeze tests**

```python
def test_family_failure_blocks_promotion(passing_bundle) -> None:
    weakened = passing_bundle.model_copy(update={"family_recall": {**passing_bundle.family_recall, "agentic_intent_abuse": 0.1}})
    report = PromotionGates(min_family_recall=0.5).evaluate(weakened)
    assert report.promotable is False
    assert "PER_FAMILY_MINIMUM" in report.failed_gate_codes


def test_evaluator_rejects_unfrozen_model(evaluation_runner, mutable_model) -> None:
    with pytest.raises(ValueError, match="frozen model artifact"):
        evaluation_runner.evaluate([], [mutable_model], evaluation_runner.split)
```

- [ ] **Step 2: Confirm runner and gate tests fail**

Run: `python -m pytest tests/evaluation/test_runner.py tests/evaluation/test_gates.py -q`

Expected: collection fails for missing runner and gates.

- [ ] **Step 3: Implement baseline ladder and hard gates**

Evaluate simple transaction, rules, transaction/entity GBDT, temporal GBDT, and graph GBDT under identical splits and action budgets. After defender artifacts are frozen, request hidden campaigns only through `HiddenCampaignGenerator` artifacts and prevent the evaluator from disclosing their parameters to defender code. A challenger must beat rules and the strongest GBDT on at least one primary value metric without breaching any friction or workload cap. Any leakage, hidden-generator, per-family, latency, integrity, or rollback failure sets `promotable=False` regardless of average score.

- [ ] **Step 4: Run hidden and metamorphic evaluation tests**

Run: `python -m pytest tests/evaluation -q`

Expected: frozen artifacts, matched budgets, future append, label permutation, generator fingerprint, family minimum, hidden generator, reason stability, and promotion veto tests pass.

- [ ] **Step 5: Commit champion/challenger assurance gates**

```bash
git add src/apar/evaluation/runner.py src/apar/evaluation/gates.py tests/evaluation
git commit -m "feat: add hidden champion challenger gates"
```

### Task 7: Produce immutable governance reports and G3 evidence

**Files:**
- Create: `src/apar/governance/__init__.py`
- Create: `src/apar/governance/report.py`
- Modify: `src/apar/api/routes/runs.py`
- Create: `src/apar/api/routes/reports.py`
- Modify: `src/apar/api/app.py`
- Create: `tests/governance/test_report.py`
- Create: `tests/api/test_reports.py`
- Create: `tests/integration/test_g3_defense.py`
- Create: `scripts/verify_g3.py`
- Create: `scripts/benchmark_scoring.py`

**Interfaces:**
- Produces: `AssuranceReport(run_manifest, model_cards, data_card, threat_cards, metrics, gates, approver, decision, rollback_ref)`
- Produces: `POST /api/v1/runs/{run_id}/evaluate`
- Produces: `GET /api/v1/reports/{report_id}`
- Produces: `POST /api/v1/reports/{report_id}/promotion`

- [ ] **Step 1: Write human-approval and immutable-report tests**

```python
def test_promotion_requires_named_human_and_rollback(report_service, passing_gate_report) -> None:
    with pytest.raises(ValueError, match="approver_id"):
        report_service.promote(passing_gate_report, approver_id="", rollback_ref=None)


def test_report_digest_changes_when_metric_changes(report_factory) -> None:
    first = report_factory(preventable_value="100.00")
    second = report_factory(preventable_value="99.00")
    assert first.artifact_ref.sha256 != second.artifact_ref.sha256
```

- [ ] **Step 2: Confirm governance and API tests fail**

Run: `python -m pytest tests/governance tests/api/test_reports.py -q`

Expected: collection fails for missing governance report service.

- [ ] **Step 3: Implement report assembly and promotion audit entries**

Assemble reports only from artifact references, never mutable in-memory frames. Record evaluator version, split digest, defender digests, generator digest, metric configuration, gate results, timestamp, approver identity, explicit decision, and rollback reference. Promotion endpoints return 409 for failed gates or absent human approval and never modify an existing report artifact.

- [ ] **Step 4: Run the G3 gate**

Run: `python scripts/verify_g3.py`

Expected output ends with `G3 PASS: leakage-safe baselines, operational metrics, hidden gates, and human promotion control`.

Run: `python scripts/benchmark_scoring.py --events 100000 --warmup 5000`

Expected: the report contains p50, p95, and p99 for schema validation, integrity, feature lookup, rules, model scoring, action policy, and total synchronous latency; asynchronous graph and case processing is reported separately.

Run: `python -m pytest tests/features tests/defense tests/cases tests/evaluation tests/governance tests/integration/test_g3_defense.py -q`

Expected: all tests pass.

- [ ] **Step 5: Commit the G3 deliverable**

```bash
git add src/apar/governance src/apar/api/routes/runs.py src/apar/api/routes/reports.py src/apar/api/app.py tests/governance tests/api/test_reports.py tests/integration/test_g3_defense.py scripts/verify_g3.py scripts/benchmark_scoring.py
git commit -m "test: establish G3 defense assurance gate"
```

## Plan completion gate

G3 is complete when rules and all three GBDT baselines are evaluated on campaign-isolated temporal splits, earlier features remain unchanged under future append, operational budgets are enforced, hidden and per-family failures veto promotion, and an immutable report requires a named human approver plus a rollback artifact.
