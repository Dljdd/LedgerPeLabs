# APAR Competition-Grade Defend Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a reproducible synthetic-data Defend subsystem that trains a strong CatBoost baseline, compares rules/GBDT/hybrid defenses under strict causal evaluation, and publishes frozen judge-facing evidence through the real APAR replay boundary.

**Architecture:** Verified signed run artifacts are converted into a scrubbed observation corpus and an evaluator-only truth sidecar. A knowledge-time feature state feeds deterministic rules and a calibrated CatBoost scorer; a separate evaluator applies chronological, cold-entity, family-holdout, regime, and post-freeze hidden tests before publishing immutable scorecards.

**Tech Stack:** Python 3.12, Pydantic 2, NumPy, pandas, PyArrow, CatBoost CPU, scikit-learn, FastAPI, cryptography, pytest, Hypothesis

**Spec:** `docs/superpowers/specs/2026-08-18-apar-defend-pipeline-design.md`

## Global Constraints

- Use Python `>=3.12,<3.13`; run commands through `.venv/bin/python` after installing `-e '.[dev]'`.
- Use synthetic APAR artifacts only. Reject private, live, scraped, purchased, or unauthorized payment data.
- Do not modify `validation_spike/`, `docs/experiments/task6-*.json`, `scripts/run_task6_holdout.py`, or the Task 6-frozen generator, red-team, simulator, ledger, rail, and trust-verifier files listed in the design spec.
- Treat `PaymentEvent.available_at < decision_at` as the historical-source boundary. Equal-time events do not observe one another.
- Never expose label, campaign, family, scenario, regime, seed, role, generator, hidden, policy, post-decision, or raw entity-identity semantics to the model matrix.
- Require at least 50 campaigns from each of the four executable families in the competition corpus.
- Use the fixed competition budgets: challenge rate `<= 0.02`, legitimate false-decline rate `<= 0.001`, and review-case rate `<= 0.01`.
- Train CatBoost on CPU with `thread_count=1`, fixed seed, `allow_writing_files=False`, `bootstrap_type="No"`, and `random_strength=0`.
- Use no pickle for externally loaded models, calibrators, thresholds, or manifests.
- Freeze model, feature, rule, calibration, threshold, source-inventory, and environment artifacts before resolving hidden-event references.
- Do not credit a challenge as prevented value unless a frozen evaluator outcome establishes the counterfactual.
- Keep failures, undefined denominators, undetected campaigns, and negative results visible in scorecards.
- State in every data/model/scorecard report that synthetic results do not establish external validity or Mastercard production performance.
- Preserve unrelated user changes. Before every commit, inspect `git status --short` and stage only files owned by that task.
- Author every commit as `Dylan Moraes <dylanmoraesdljdd@gmail.com>`; do not add AI attribution or co-author trailers.

---

## Locked File Map

```text
config/defense/feature-catalog.json           Ordered decision-time feature contract
config/defense/competition-profile.json       Seeds, splits, budgets, gates, and regimes
src/apar/defense/contracts.py                 Observation and decision-policy contracts
src/apar/evaluation/contracts.py              Evaluator-only corpus, truth, split, and metric contracts
src/apar/evaluation/corpus.py                 Verified-run ingestion, scrubbing, and truth isolation
src/apar/features/catalog.py                  Feature allowlist and semantic provenance audit
src/apar/features/state.py                    Knowledge-time state, graph state, and checkpoints
src/apar/features/builders.py                 Offline feature matrix construction
src/apar/features/parity.py                   Leakage, provenance, and replay invariants
src/apar/evaluation/splits.py                 Chronological, maturity, entity, and family splits
src/apar/evaluation/regimes.py                Lineage-preserving derived regimes
src/apar/defense/rules.py                     Family-agnostic rule baseline
src/apar/defense/policy.py                    Arm scores, action precedence, budgets, and fallback
src/apar/defense/gbdt.py                      CatBoost search, training, serialization, and scoring
src/apar/defense/calibration.py               JSON-safe sigmoid/isotonic calibration
src/apar/defense/thresholds.py                Matched-budget threshold selection
src/apar/defense/bundle.py                    Signed frozen defender publication and loading
src/apar/cases/grouping.py                     Past-only deterministic case grouping
src/apar/cases/queue.py                        Capacity, service-time, backlog, and SLA simulation
src/apar/evaluation/metrics.py                 Classification, calibration, value, alert, workload, latency
src/apar/evaluation/replay.py                  Rules/GBDT/hybrid decision replay
src/apar/evaluation/gates.py                   Hard blockers and truthful champion selection
src/apar/evaluation/reporting.py               JSON, CSV, Markdown, data-card, and model-card artifacts
src/apar/evaluation/service.py                 Artifact-backed evaluation orchestration
src/apar/evaluation_hidden/defense_authority.py Frozen-only restricted-event release
src/apar/api/routes/defense.py                 Public evaluation endpoints
scripts/build_defense_corpus.py                Reproducible corpus build/export
scripts/generate_defense_runs.py               Execute the preregistered 200 signed APAR runs
scripts/train_defender.py                      Development-only training and freeze
scripts/evaluate_defender.py                   Development and separately authorized hidden phases
scripts/verify_g3.py                            One-command Defend verification
tests/defense/                                 Corpus, rules, model, calibration, threshold, bundle tests
tests/features/                                Catalog, state, parity, leakage, metamorphic tests
tests/cases/                                   Grouping and queue tests
tests/evaluation/                              Split, regime, metric, replay, gate, reporting tests
tests/api/test_defense.py                      Public/restricted API tests
tests/integration/test_g3_defense.py           Real APAR run-to-scorecard golden path
fixtures/defense/v1/                           Frozen corpus/model/threshold/scorecard export
docs/experiments/defense-v1-preregistration.json
docs/experiments/defense-v1-run-manifests.json
docs/experiments/defense-v1-result.json
```

### Task 1: Add defense contracts and the verified corpus boundary

**Files:**
- Create: `src/apar/defense/__init__.py`
- Create: `src/apar/defense/contracts.py`
- Create: `src/apar/evaluation/__init__.py`
- Create: `src/apar/evaluation/contracts.py`
- Create: `src/apar/evaluation/corpus.py`
- Create: `tests/defense/__init__.py`
- Create: `tests/defense/test_contracts.py`
- Create: `tests/evaluation/__init__.py`
- Create: `tests/evaluation/test_corpus.py`

**Interfaces:**
- Consumes: `PaymentEvent`, `RunManifest`, `RunRunner.verify_run(manifest)`, `ArtifactStore.read(ref)`
- Produces: `ObservedEvent`, `PolicyThresholds`, `EvaluationTruthRow`, `CorpusProfile`, `CorpusManifest`, `FrozenCorpus`
- Produces: `scrub_event(event: PaymentEvent) -> ObservedEvent`
- Produces: `assemble_verified_corpus(manifests: Sequence[RunManifest], runner: RunRunner, store: ArtifactStore, profile: CorpusProfile) -> FrozenCorpus`

- [ ] **Step 1: Write failing closed-contract and scrubbing tests**

```python
def test_scrub_event_removes_every_evaluator_semantic() -> None:
    event = make_payment_event(
        rail_data={"payment_id": "pay-1", "hidden_family": "app_scam_mule"},
        lineage={"synthetic": True, "campaign_role": "attack", "generator": "dev"},
        party_refs={"actor_role": "mule", "merchant_id": "merchant-1"},
    )
    observed = scrub_event(event)
    dumped = observed.model_dump(mode="json")
    encoded = json.dumps(dumped, sort_keys=True).lower()
    assert observed.optional_refs == {"merchant_id": "merchant-1"}
    assert all(token not in encoded for token in (
        "hidden_family", "campaign_role", "actor_role", "generator",
    ))


def test_corpus_rejects_a_manifest_that_no_longer_verifies(
    completed_manifest: RunManifest,
    run_runner: RunRunner,
    artifact_store: ArtifactStore,
) -> None:
    changed = completed_manifest.model_copy(update={"lineage_digest": "0" * 64})
    with pytest.raises(CorpusVerificationError, match="authenticated run"):
        assemble_verified_corpus(
            [changed], run_runner, artifact_store, CorpusProfile.fixture()
        )
```

- [ ] **Step 2: Run the focused tests and confirm the missing-module failure**

Run: `.venv/bin/python -m pytest tests/defense/test_contracts.py tests/evaluation/test_corpus.py -q`

Expected: collection fails because `apar.defense.contracts` and `apar.evaluation.corpus` do not exist.

- [ ] **Step 3: Implement the closed corpus contracts**

```python
class ObservedEvent(ExternalContract):
    schema_version: Literal["1.0.0"] = "1.0.0"
    event_id: str
    payment_id: str
    rail: Rail
    event_type: EventKind
    amount: Decimal
    currency: str
    event_time: datetime
    available_at: datetime
    decision_at: datetime | None
    actor_id: str
    counterparty_id: str
    optional_refs: dict[str, str] = Field(default_factory=dict)
    integrity_status: Literal["pass", "fail", "not_applicable"]
    integrity_reason: str | None = None
    is_decision_point: bool
    privacy_classification: Literal["synthetic"] = "synthetic"


class EvaluationTruthRow(ExternalContract):
    event_id: str
    payment_id: str
    campaign_id: str
    family: Literal[
        "agentic_intent_abuse",
        "app_scam_mule",
        "card_testing_cnp",
        "synthetic_merchant_refund",
    ]
    viewpoint: Literal["development", "hidden"]
    is_fraud: bool
    label_source: Literal["population_truth", "integrity_truth", "hidden_truth"]
    label_mature_at: datetime
    first_settlement_at: datetime | None
    net_settled_value: Decimal
    lifecycle_event_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class FrozenCorpus:
    observations: tuple[ObservedEvent, ...]
    truth: tuple[EvaluationTruthRow, ...]
    manifest: CorpusManifest


class PolicyThresholds(ExternalContract):
    challenge: float = Field(ge=0.0, le=1.0)
    decline: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def ordered(self) -> "PolicyThresholds":
        if self.challenge > self.decline:
            raise ValueError("challenge threshold must not exceed decline threshold")
        return self
```

`scrub_event` must allow only `merchant_id`, `payee_id`, `beneficiary_entity_id`,
`user_entity_id`, `device_id`, `institution_id`, and `agent_id` from `party_refs`.
It must set `payment_id` from `rail_data`, derive integrity status from the current
agentic receipt fields, and classify only the three opening-event rules in the
spec as decision points. `assemble_verified_corpus` must reverify every manifest,
parse canonical event/population/summary artifacts, group lifecycle events by
payment, derive truth outside the observation object, apply the profile's exact
label-delay days, and reject duplicate IDs or non-synthetic rows. A read-spy test
must prove development corpus assembly never reads
`restricted_hidden_evaluation_events`, `restricted_evaluation_input`,
`restricted_evaluation_audit`, or `restricted_validity`; only the frozen hidden
authority in Task 12 may resolve restricted event references.

- [ ] **Step 4: Run the contract/corpus tests**

Run: `.venv/bin/python -m pytest tests/defense/test_contracts.py tests/evaluation/test_corpus.py -q`

Expected: all tests pass, including manifest tamper rejection, non-synthetic input,
duplicate payment/event IDs, decision-point selection, truth isolation, lifecycle
netting, and label-maturity checks.

- [ ] **Step 5: Commit the corpus boundary**

```bash
git add src/apar/defense src/apar/evaluation tests/defense/test_contracts.py tests/evaluation/test_corpus.py
git -c user.name='Dylan Moraes' -c user.email='dylanmoraesdljdd@gmail.com' commit -m "feat: add verified defense corpus boundary"
```

### Task 2: Lock the feature catalog and semantic audit

**Files:**
- Create: `config/defense/feature-catalog.json`
- Create: `src/apar/features/__init__.py`
- Create: `src/apar/features/catalog.py`
- Create: `tests/features/__init__.py`
- Create: `tests/features/test_catalog.py`

**Interfaces:**
- Consumes: `ObservedEvent`
- Produces: `FeatureDefinition`, `FeatureCatalog`, `FeatureSource`
- Produces: `load_feature_catalog(path: Path) -> FeatureCatalog`
- Produces: `audit_feature_catalog(catalog: FeatureCatalog) -> None`

- [ ] **Step 1: Write failing catalog completeness and semantic-source tests**

```python
def test_catalog_has_the_exact_ordered_competition_features() -> None:
    catalog = load_feature_catalog(FEATURE_CATALOG_PATH)
    assert catalog.names == EXPECTED_FEATURE_NAMES
    assert len(catalog.names) == len(set(catalog.names)) == 48


def test_semantically_forbidden_source_is_rejected_after_an_innocent_rename() -> None:
    definition = FeatureDefinition(
        name="ordinary_count",
        family="temporal",
        rails=(Rail.CARD,),
        source_paths=("truth.is_fraud",),
        state_keys=(),
        window_seconds=3600,
        missing_behavior="zero",
    )
    with pytest.raises(FeatureCatalogError, match="forbidden source"):
        audit_feature_catalog(FeatureCatalog(version="1.0.0", features=(definition,)))
```

- [ ] **Step 2: Run the catalog tests and confirm failure**

Run: `.venv/bin/python -m pytest tests/features/test_catalog.py -q`

Expected: collection fails because `apar.features.catalog` does not exist.

- [ ] **Step 3: Implement the exact allowlisted catalog**

```python
class FeatureDefinition(ExternalContract):
    name: str
    family: Literal["transaction", "temporal", "entity", "graph", "data_quality"]
    rails: tuple[Rail, ...]
    source_paths: tuple[str, ...]
    state_keys: tuple[str, ...] = ()
    window_seconds: int | None = Field(default=None, ge=1)
    missing_behavior: Literal["zero", "sentinel", "indicator"]
```

The JSON file and `EXPECTED_FEATURE_NAMES` test fixture must contain this ordered
list, with source paths and rail applicability for every row:

```python
EXPECTED_FEATURE_NAMES = (
    "txn_log_amount", "txn_rail_card", "txn_rail_a2a", "txn_rail_agentic",
    "txn_hour_sin", "txn_hour_cos", "txn_integrity_pass",
    "txn_optional_ref_count", "actor_count_1m", "actor_count_10m",
    "actor_count_1h", "actor_count_24h", "actor_amount_1h",
    "actor_amount_24h", "counterparty_count_1h", "counterparty_count_24h",
    "counterparty_amount_24h", "actor_prior_decline_1h",
    "actor_prior_challenge_1h", "actor_prior_return_24h",
    "counterparty_prior_refund_24h", "actor_seconds_since_first",
    "actor_seconds_since_last", "counterparty_seconds_since_first",
    "counterparty_seconds_since_last", "pair_seconds_since_first",
    "pair_seconds_since_last", "actor_distinct_counterparties_24h",
    "counterparty_distinct_actors_24h", "actor_amount_zscore_24h",
    "counterparty_amount_zscore_24h", "pair_prior_count",
    "graph_actor_fanout", "graph_counterparty_fanin",
    "graph_shared_neighbor_count", "graph_two_hop_reach",
    "graph_component_size", "graph_edge_density", "graph_repeated_edge",
    "graph_burst_motif", "graph_prior_suspicious_count",
    "dq_missing_optional_count", "dq_current_availability_lag_ms",
    "dq_mean_history_lag_ms", "dq_late_event_count", "dq_history_count",
    "dq_history_age_seconds", "dq_degraded_state",
)
```

`audit_feature_catalog` must reject names or source paths containing any of:
`fraud`, `illicit`, `label`, `target`, `campaign`, `family`, `scenario`, `regime`,
`seed`, `generator`, `hidden`, `policy`, `role`, `viewpoint`, `chargeback_truth`,
or `post_decision`. It must also reject raw ID features, while allowing IDs only
as declared `state_keys` or provenance references.

- [ ] **Step 4: Run catalog and schema tests**

Run: `.venv/bin/python -m pytest tests/features/test_catalog.py tests/contracts/test_schema_snapshots.py -q`

Expected: all tests pass; the catalog has exactly 48 unique ordered features and
no forbidden source.

- [ ] **Step 5: Commit the feature contract**

```bash
git add config/defense/feature-catalog.json src/apar/features tests/features/test_catalog.py
git -c user.name='Dylan Moraes' -c user.email='dylanmoraesdljdd@gmail.com' commit -m "feat: lock defense feature catalog"
```

### Task 3: Implement knowledge-time feature state and checkpoints

**Files:**
- Create: `src/apar/features/state.py`
- Create: `src/apar/features/builders.py`
- Create: `tests/features/conftest.py`
- Create: `tests/features/test_state.py`
- Create: `tests/features/test_builders.py`

**Interfaces:**
- Consumes: `Sequence[ObservedEvent]`, `FeatureCatalog`
- Produces: `FeatureVector`, `FeatureMatrix`, `CausalFeatureState`
- Produces: `build_feature_matrix(events: Sequence[ObservedEvent], catalog: FeatureCatalog) -> FeatureMatrix`
- Produces: `CausalFeatureState.checkpoint() -> bytes`
- Produces: `CausalFeatureState.restore(payload: bytes, catalog: FeatureCatalog) -> CausalFeatureState`

- [ ] **Step 1: Write failing knowledge-time, equal-time, graph, and checkpoint tests**

```python
def test_equal_time_decisions_do_not_observe_one_another(
    equal_time_observations: tuple[ObservedEvent, ...],
    feature_catalog: FeatureCatalog,
) -> None:
    matrix = build_feature_matrix(equal_time_observations, feature_catalog)
    assert [row.values["actor_count_1m"] for row in matrix.rows] == [0.0, 0.0]
    assert all(row.max_source_available_at is None for row in matrix.rows)


def test_checkpoint_restore_reproduces_future_vectors(
    observed_stream: tuple[ObservedEvent, ...],
    feature_catalog: FeatureCatalog,
) -> None:
    first = CausalFeatureState(feature_catalog)
    before = first.process(observed_stream[:8])
    restored = CausalFeatureState.restore(first.checkpoint(), feature_catalog)
    assert restored.process(observed_stream[8:]) == first.process(observed_stream[8:])
    assert before
```

- [ ] **Step 2: Run state tests and confirm failure**

Run: `.venv/bin/python -m pytest tests/features/test_state.py tests/features/test_builders.py -q`

Expected: collection fails for missing state/builders modules.

- [ ] **Step 3: Implement two-phase knowledge-time processing**

```python
class CausalFeatureState:
    def process(self, events: Sequence[ObservedEvent]) -> tuple[FeatureVector, ...]:
        ordered = tuple(sorted(events, key=lambda e: (e.available_at, e.event_id)))
        decisions = sorted(
            (e for e in ordered if e.is_decision_point),
            key=lambda e: (e.decision_at, e.event_id),
        )
        output: list[FeatureVector] = []
        for decision_time, group in groupby(decisions, key=attrgetter("decision_at")):
            assert decision_time is not None
            self._admit_sources_strictly_before(ordered, decision_time)
            batch = tuple(group)
            output.extend(self._compute(event, decision_time) for event in batch)
        return tuple(output)
```

State must maintain timestamped actor, counterparty, pair, outcome, amount, and
adjacency histories. Prune rolling windows before each calculation; use stable
population standard deviation with a zero-variance result of `0.0`; update graph
edges once per payment opening; and derive all 48 catalog values. Every vector
must list only source event IDs actually used, set `max_source_available_at` to
their maximum or `None`, and record the catalog digest. Checkpoints must be
canonical JSON containing schema version, catalog digest, watermark, admitted
event IDs, histories, adjacency, and a self-digest.

- [ ] **Step 4: Run state, builder, and typing tests**

Run: `.venv/bin/python -m pytest tests/features/test_state.py tests/features/test_builders.py -q`

Expected: all tests pass for rolling windows, amount statistics, lifecycle
outcomes, graph motifs, null history, missing fields, checkpoint corruption,
equal-time batches, and stable row order.

- [ ] **Step 5: Commit causal feature state**

```bash
git add src/apar/features/state.py src/apar/features/builders.py tests/features
git -c user.name='Dylan Moraes' -c user.email='dylanmoraesdljdd@gmail.com' commit -m "feat: add causal defense feature state"
```

### Task 4: Add feature parity, provenance, and metamorphic gates

**Files:**
- Create: `src/apar/features/parity.py`
- Create: `tests/features/test_parity.py`
- Create: `tests/features/test_leakage.py`
- Create: `tests/features/test_metamorphic.py`

**Interfaces:**
- Consumes: `ObservedEvent`, `FeatureMatrix`, `FeatureCatalog`
- Produces: `FeatureAuditReport`
- Produces: `audit_feature_matrix(events, matrix, catalog) -> FeatureAuditReport`
- Produces: `assert_future_append_invariant(prefix, future, catalog) -> None`
- Produces: `assert_online_offline_parity(events, catalog) -> None`

- [ ] **Step 1: Write deliberate leakage and future-append failures**

```python
def test_future_append_leaves_prior_feature_bytes_identical(
    observed_stream: tuple[ObservedEvent, ...],
    feature_catalog: FeatureCatalog,
) -> None:
    prefix = observed_stream[:12]
    future = observed_stream[12:]
    assert_future_append_invariant(prefix, future, feature_catalog)


@pytest.mark.parametrize("source", (
    "truth.is_fraud", "lineage.campaign_role", "rail_data.hidden_family",
    "party_refs.actor_role", "scenario.seed", "event.viewpoint",
))
def test_forbidden_provenance_is_rejected_even_with_safe_name(source: str) -> None:
    matrix = matrix_with_injected_feature(name="safe_metric", source_path=source)
    with pytest.raises(FeatureLeakageError):
        audit_feature_matrix(matrix.events, matrix, matrix.catalog)
```

Define `matrix_with_injected_feature` in `tests/features/test_leakage.py` by
copying the deterministic clean fixture, appending one `FeatureDefinition` with
the supplied source path, and appending the matching constant column to every
row. The helper must not call production audit code while constructing the
deliberately invalid matrix.

- [ ] **Step 2: Run parity/leakage tests and confirm failure**

Run: `.venv/bin/python -m pytest tests/features/test_parity.py tests/features/test_leakage.py tests/features/test_metamorphic.py -q`

Expected: collection fails because `apar.features.parity` does not exist.

- [ ] **Step 3: Implement independent audits and metamorphic helpers**

```python
@dataclass(frozen=True, slots=True)
class FeatureAuditReport:
    catalog_valid: bool
    strictly_past_only: bool
    source_ids_resolve: bool
    feature_order_matches: bool
    forbidden_sources: tuple[str, ...]

    @property
    def passed(self) -> bool:
        return (
            self.catalog_valid and self.strictly_past_only
            and self.source_ids_resolve and self.feature_order_matches
            and not self.forbidden_sources
        )
```

The audit must rebuild a source-ID-to-availability map independently of feature
state, require every historical source to resolve and precede its decision, and
compare online incremental vectors with one-shot offline vectors. Add helpers for
row permutation, equal-time permutation, synthetic ID bijection, future append,
duplicate event IDs, missing optional references, consistent economic scaling,
and checkpoint/restart.

- [ ] **Step 4: Run the complete feature suite**

Run: `.venv/bin/python -m pytest tests/features -q`

Expected: every deliberate injection is rejected and every clean invariant passes.

- [ ] **Step 5: Commit feature assurance gates**

```bash
git add src/apar/features/parity.py tests/features
git -c user.name='Dylan Moraes' -c user.email='dylanmoraesdljdd@gmail.com' commit -m "test: enforce feature leakage and parity gates"
```

### Task 5: Add chronological, maturity, cold-entity, family, and regime manifests

**Files:**
- Create: `src/apar/evaluation/splits.py`
- Create: `src/apar/evaluation/regimes.py`
- Create: `tests/evaluation/test_splits.py`
- Create: `tests/evaluation/test_regimes.py`

**Interfaces:**
- Consumes: `FrozenCorpus`, `FeatureMatrix`
- Produces: `SplitConfig`, `EvaluationSplit`, `EntityCohort`, `RegimeSpec`, `DerivedRegimeManifest`
- Produces: `make_evaluation_split(corpus, config) -> EvaluationSplit`
- Produces: `make_leave_one_family_out(split, family) -> EvaluationSplit`
- Produces: `derive_regime(corpus, spec) -> tuple[FrozenCorpus, DerivedRegimeManifest]`

- [ ] **Step 1: Write campaign-isolation, maturity, cold-cohort, and truth-preservation tests**

```python
def test_campaign_never_crosses_chronological_partitions(corpus: FrozenCorpus) -> None:
    split = make_evaluation_split(corpus, fixed_split_config())
    campaign_sets = [set(split.campaigns[name]) for name in split.partition_names]
    assert all(
        left.isdisjoint(right)
        for index, left in enumerate(campaign_sets)
        for right in campaign_sets[index + 1:]
    )


def test_regime_transform_cannot_change_truth(corpus: FrozenCorpus) -> None:
    changed, manifest = derive_regime(corpus, RegimeSpec.missing_optional())
    assert changed.truth == corpus.truth
    assert manifest.parent_corpus_digest == corpus.manifest.corpus_digest
```

- [ ] **Step 2: Run split/regime tests and confirm failure**

Run: `.venv/bin/python -m pytest tests/evaluation/test_splits.py tests/evaluation/test_regimes.py -q`

Expected: collection fails for missing evaluation modules.

- [ ] **Step 3: Implement the exact split and transformation rules**

```python
class SplitConfig(ExternalContract):
    train_end: datetime
    calibrator_fit_end: datetime
    threshold_end: datetime
    development_end: datetime
    held_out_family: str | None = None


class EntityCohort(StrEnum):
    COLD_ACTOR = "cold_actor"
    COLD_COUNTERPARTY = "cold_counterparty"
    COLD_PAIR = "cold_pair"
    WARM_WITHIN_CAMPAIGN = "warm_within_campaign"
    RETURNING_PRIOR_CAMPAIGN = "returning_prior_campaign"
```

Assign a whole campaign by its first decision time. Permit training rows only
when `label_mature_at <= train_end`. Compute entity cohorts against identities
observed strictly earlier than the test decision. Implement the six exact regimes
from the spec: prevalence dilution with separately supplied control campaigns,
missing optional refs, non-decision source availability delay, consistent burst
compression, consistent benign amount scaling, and bijective cold-ID remapping.
Every derived manifest records parent digest, transformer version, parameters,
output digest, and a boolean proving truth bytes are unchanged.

- [ ] **Step 4: Run evaluation split/regime tests**

Run: `.venv/bin/python -m pytest tests/evaluation/test_splits.py tests/evaluation/test_regimes.py -q`

Expected: all split boundary, label maturity, cold/warm cohort, four-family LOFO,
truth immutability, deterministic transform, and economic consistency tests pass.

- [ ] **Step 5: Commit evaluation partitions and regimes**

```bash
git add src/apar/evaluation tests/evaluation/test_splits.py tests/evaluation/test_regimes.py
git -c user.name='Dylan Moraes' -c user.email='dylanmoraesdljdd@gmail.com' commit -m "feat: add isolated defense evaluation splits"
```

### Task 6: Implement family-agnostic rules and action precedence

**Files:**
- Create: `src/apar/defense/rules.py`
- Create: `src/apar/defense/policy.py`
- Create: `tests/defense/test_rules.py`
- Create: `tests/defense/test_policy.py`

**Interfaces:**
- Consumes: `ObservedEvent`, `FeatureVector`, `Action`
- Produces: `DefenseReason`, `RuleHit`, `RuleResult`, `RuleManifest`, `OperatingBudget`, `DefenseDecision`
- Produces: `RuleEngine.evaluate(event, vector) -> RuleResult`
- Produces: `ActionPolicy.choose(event, rule_result, calibrated_score, thresholds) -> DefenseDecision`

- [ ] **Step 1: Write failing integrity-precedence, family-blindness, and fallback tests**

```python
def test_integrity_failure_cannot_be_overridden_by_low_risk(
    failed_agentic_event: ObservedEvent,
    empty_vector: FeatureVector,
) -> None:
    decision = ActionPolicy.default().choose(
        failed_agentic_event,
        RuleResult.clear(),
        calibrated_score=0.0,
        thresholds=PolicyThresholds(challenge=0.8, decline=0.95),
    )
    assert decision.action is Action.DECLINE
    assert decision.reason_codes == (DefenseReason.INTEGRITY_FAILURE,)


def test_rules_have_no_family_argument(rule_engine: RuleEngine) -> None:
    assert tuple(inspect.signature(rule_engine.evaluate).parameters) == (
        "event", "vector",
    )
```

- [ ] **Step 2: Run rule/policy tests and confirm failure**

Run: `.venv/bin/python -m pytest tests/defense/test_rules.py tests/defense/test_policy.py -q`

Expected: collection fails for missing rules/policy modules.

- [ ] **Step 3: Implement stable rules, scores, reasons, and fallback**

```python
class DefenseReason(StrEnum):
    INTEGRITY_FAILURE = "INTEGRITY_FAILURE"
    REQUIRED_DATA_MISSING = "REQUIRED_DATA_MISSING"
    ACTOR_VELOCITY = "ACTOR_VELOCITY"
    COUNTERPARTY_VELOCITY = "COUNTERPARTY_VELOCITY"
    AMOUNT_DEVIATION = "AMOUNT_DEVIATION"
    NEW_COUNTERPARTY = "NEW_COUNTERPARTY"
    GRAPH_FAN_IN = "GRAPH_FAN_IN"
    GRAPH_FAN_OUT = "GRAPH_FAN_OUT"
    GRAPH_SHARED_NEIGHBOR = "GRAPH_SHARED_NEIGHBOR"
    FEATURE_STATE_DEGRADED = "FEATURE_STATE_DEGRADED"
    MODEL_UNAVAILABLE = "MODEL_UNAVAILABLE"
    MODEL_TIMEOUT = "MODEL_TIMEOUT"


@dataclass(frozen=True, slots=True)
class OperatingBudget:
    challenge_rate_max: float = 0.02
    false_decline_rate_max: float = 0.001
    review_case_rate_max: float = 0.01
```

Rule hits must have a score in `[0,1]`, severity, stable reason, evidence source
IDs, and rule version. Exact initial thresholds are actor counts `>=4` in 1m or
`>=8` in 10m, counterparty fan-in `>=5`, actor fan-out `>=5`, amount z-score
`>=4`, shared neighbors `>=3`, repeated pair count `>=4`, and degraded-state
challenge at score `0.60`. Integrity and required-data failures are mandatory;
all other hits contribute a family-blind continuous rule score. Sort reasons by
mandatory status, descending score, then reason string.

- [ ] **Step 4: Run rule/policy tests**

Run: `.venv/bin/python -m pytest tests/defense/test_rules.py tests/defense/test_policy.py -q`

Expected: mandatory precedence, exact thresholds, stable ordering, evidence refs,
no family access, score bounds, and hybrid rules-only fallback tests pass.

- [ ] **Step 5: Commit deterministic defense layers**

```bash
git add src/apar/defense/rules.py src/apar/defense/policy.py tests/defense
git -c user.name='Dylan Moraes' -c user.email='dylanmoraesdljdd@gmail.com' commit -m "feat: add transparent defense rules and policy"
```

### Task 7: Add deterministic CatBoost training and serialization

**Files:**
- Modify: `pyproject.toml`
- Create: `src/apar/defense/gbdt.py`
- Create: `tests/defense/test_gbdt.py`

**Interfaces:**
- Consumes: `FeatureMatrix`, training row IDs, binary labels
- Produces: `GbdtTrainingConfig`, `FoldResult`, `TrainingReceipt`, `CatBoostScorer`
- Produces: `train_gbdt(matrix, labels, train_ids, folds, config) -> CatBoostScorer`
- Produces: `CatBoostScorer.to_bytes() -> bytes`
- Produces: `CatBoostScorer.from_bytes(payload, receipt) -> CatBoostScorer`
- Produces: `CatBoostScorer.predict(matrix) -> np.ndarray`
- Produces: `CatBoostScorer.contributions(matrix) -> np.ndarray`

- [ ] **Step 1: Add the model dependencies and install the development environment**

Add these dependencies to `project.dependencies`:

```toml
"catboost>=1.2,<2.0",
"scikit-learn>=1.5,<2.0",
```

Run: `.venv/bin/python -m pip install -e '.[dev]'`

Expected: CatBoost and scikit-learn import successfully under Python 3.12.

- [ ] **Step 2: Write failing reproducibility, forbidden-column, and reload tests**

```python
def test_seeded_cpu_training_and_reload_reproduce_scores(
    training_fixture: TrainingFixture,
) -> None:
    first = train_gbdt(**training_fixture.kwargs)
    second = train_gbdt(**training_fixture.kwargs)
    np.testing.assert_allclose(
        first.predict(training_fixture.test_matrix),
        second.predict(training_fixture.test_matrix),
        rtol=0.0,
        atol=1e-12,
    )
    restored = CatBoostScorer.from_bytes(first.to_bytes(), first.receipt)
    np.testing.assert_array_equal(
        first.predict(training_fixture.test_matrix),
        restored.predict(training_fixture.test_matrix),
    )


def test_training_rejects_a_column_outside_the_feature_manifest(
    training_fixture: TrainingFixture,
) -> None:
    changed = training_fixture.with_column("campaign_alias", 1.0)
    with pytest.raises(ModelContractError, match="feature order"):
        train_gbdt(**changed.kwargs)


def test_contribution_rows_reconstruct_raw_model_scores(training_fixture) -> None:
    scorer = train_gbdt(**training_fixture.kwargs)
    contributions = scorer.contributions(training_fixture.test_matrix)
    reconstructed_logits = contributions[:, :-1].sum(axis=1) + contributions[:, -1]
    np.testing.assert_allclose(
        reconstructed_logits,
        scorer.predict_raw(training_fixture.test_matrix),
        rtol=0.0,
        atol=1e-12,
    )
```

- [ ] **Step 3: Run GBDT tests and confirm failure**

Run: `.venv/bin/python -m pytest tests/defense/test_gbdt.py -q`

Expected: collection fails because `apar.defense.gbdt` does not exist.

- [ ] **Step 4: Implement the bounded rolling-time search**

```python
@dataclass(frozen=True, slots=True)
class GbdtTrainingConfig:
    seed: int = 260816
    depths: tuple[int, ...] = (4, 6)
    learning_rates: tuple[float, ...] = (0.03, 0.08)
    l2_leaf_regs: tuple[float, ...] = (3.0, 8.0)
    iterations: int = 300


def _classifier(params: HyperParameters, class_weights: tuple[float, float], seed: int):
    return CatBoostClassifier(
        loss_function="Logloss",
        iterations=300,
        depth=params.depth,
        learning_rate=params.learning_rate,
        l2_leaf_reg=params.l2_leaf_reg,
        class_weights=list(class_weights),
        random_seed=seed,
        thread_count=1,
        allow_writing_files=False,
        bootstrap_type="No",
        random_strength=0,
        verbose=False,
    )
```

Evaluate every one of the eight parameter combinations on the frozen rolling
folds using mean average precision, then lower legitimate FPR, then lexicographic
parameters as tie-breakers. Compute class weights from training rows only. Save
the native CatBoost model through a private temporary file, store its bytes, and
record feature order, train IDs digest, fold results, library version, Python
version, platform, seed, weights, selected parameters, and training cutoff in the
receipt.

Exclude rows resolved by mandatory schema or integrity gates from risk-model
training and record their count in the receipt. Implement contributions with
CatBoost `ShapValues`; return one column per ordered feature plus the expected
value column, and verify raw-logit reconstruction before publishing a model.

- [ ] **Step 5: Run GBDT and dependency tests**

Run: `.venv/bin/python -m pytest tests/defense/test_gbdt.py -q`

Expected: deterministic search, training-only weights, fixed feature order,
finite bounded scores, native reload parity, and forbidden-column rejection pass.

- [ ] **Step 6: Commit the strong GBDT baseline**

```bash
git add pyproject.toml src/apar/defense/gbdt.py tests/defense/test_gbdt.py
git -c user.name='Dylan Moraes' -c user.email='dylanmoraesdljdd@gmail.com' commit -m "feat: add deterministic CatBoost baseline"
```

### Task 8: Add chronological calibration and matched-budget thresholds

**Files:**
- Create: `src/apar/defense/calibration.py`
- Create: `src/apar/defense/thresholds.py`
- Create: `tests/defense/test_calibration.py`
- Create: `tests/defense/test_thresholds.py`

**Interfaces:**
- Consumes: raw scores and labels from calibrator-fit and calibrator-selection windows
- Produces: `CalibrationKind`, `CalibrationArtifact`, `ProbabilityCalibrator`, `PolicyThresholds`, `ThresholdReport`
- Produces: `select_calibrator(fit_scores, fit_labels, selection_scores, selection_labels, min_class_count=50) -> ProbabilityCalibrator`
- Produces: `select_policy_thresholds(scores, labels, mandatory_actions, review_case_counter, budget, values=None) -> ThresholdReport`

- [ ] **Step 1: Write failing chronological-selection, JSON-roundtrip, and budget tests**

```python
def test_calibrator_selection_uses_a_later_window() -> None:
    calibrator = select_calibrator(
        fit_scores=np.array([0.05, 0.2, 0.7, 0.9] * 30),
        fit_labels=np.array([0, 0, 1, 1] * 30),
        selection_scores=np.array([0.1, 0.3, 0.6, 0.8] * 30),
        selection_labels=np.array([0, 0, 1, 1] * 30),
        min_class_count=50,
    )
    restored = ProbabilityCalibrator.from_json(calibrator.to_json())
    np.testing.assert_allclose(restored.predict(np.array([0.2, 0.8])), calibrator.predict(np.array([0.2, 0.8])))


def test_thresholds_respect_fixed_false_decline_and_challenge_budgets() -> None:
    def review_case_counter(actions: np.ndarray) -> int:
        return len({
            REVIEW_CASE_KEYS[index]
            for index, action in enumerate(actions)
            if action is not Action.APPROVE
        })

    report = select_policy_thresholds(
        SCORES,
        LABELS,
        MANDATORY_ACTIONS,
        review_case_counter,
        OperatingBudget(),
    )
    assert report.calibration_false_decline_rate <= 0.001
    assert report.calibration_challenge_rate <= 0.02
    assert report.calibration_review_case_rate <= 0.01
    assert 0.0 <= report.thresholds.challenge <= report.thresholds.decline <= 1.0
```

- [ ] **Step 2: Run calibration/threshold tests and confirm failure**

Run: `.venv/bin/python -m pytest tests/defense/test_calibration.py tests/defense/test_thresholds.py -q`

Expected: collection fails for missing modules.

- [ ] **Step 3: Implement JSON-safe sigmoid and isotonic calibrators**

```python
class CalibrationKind(StrEnum):
    SIGMOID = "sigmoid"
    ISOTONIC = "isotonic"


class CalibrationArtifact(ExternalContract):
    kind: CalibrationKind
    fit_row_digest: str
    selection_row_digest: str
    selection_brier: float
    sigmoid_coefficient: float | None = None
    sigmoid_intercept: float | None = None
    isotonic_x: tuple[float, ...] = ()
    isotonic_y: tuple[float, ...] = ()


def _sigmoid_inputs(scores: np.ndarray) -> np.ndarray:
    clipped = np.clip(scores.astype(float), 1e-8, 1.0 - 1e-8)
    return np.log(clipped / (1.0 - clipped)).reshape(-1, 1)
```

Fit sigmoid with `LogisticRegression(C=1e6, solver="lbfgs", random_state=260816)`
on logit-transformed raw scores. Store its coefficient/intercept as numeric JSON.
Fit isotonic with `IsotonicRegression(out_of_bounds="clip")` and store exact
threshold arrays. Isotonic is eligible only when each class has at least 50 fit
rows. Validate that exactly the fields for the selected kind are populated and
record both chronological row digests. Choose lower selection-window Brier
score; ties choose sigmoid. Clip final probabilities to `[1e-8, 1 - 1e-8]` so
the frozen `1.0` threshold is an unambiguous no-action sentinel.

- [ ] **Step 4: Implement exhaustive deterministic threshold selection**

For every unique score plus `0.0` and `1.0`, evaluate candidate decline and
challenge pairs. Mandatory declines are always applied first. Keep only pairs
with legitimate false-decline rate `<=0.001`, total challenge rate `<=0.02`, and
review-case rate `<=0.01`. `review_case_counter(actions)` is a deterministic,
past-only callback: threshold tests use the fixed hand oracle, while production
wires the case grouper implemented in Task 10. It returns the number of grouped
cases created by the candidate action vector and cannot access evaluator truth.
Choose the pair with highest fraud value captured when values are supplied,
otherwise highest fraud recall; tie-break by fewer false interventions, higher
decline threshold, higher challenge threshold. Return infeasible rather than
silently relaxing a budget.

Reuse the closed `PolicyThresholds` contract from Task 1. `ActionPolicy` applies
decline first and challenge second using `score >= threshold`; because calibrated
scores are clipped below `1.0`, a frozen threshold of `1.0` disables that action.

- [ ] **Step 5: Run calibration and threshold tests**

Run: `.venv/bin/python -m pytest tests/defense/test_calibration.py tests/defense/test_thresholds.py -q`

Expected: fit/selection separation, class-count eligibility, Brier tie-break,
JSON roundtrip, monotonic calibrated probabilities, exact budgets, deterministic
ties, and infeasible-budget tests pass.

- [ ] **Step 6: Commit calibration and thresholds**

```bash
git add src/apar/defense/calibration.py src/apar/defense/thresholds.py tests/defense
git -c user.name='Dylan Moraes' -c user.email='dylanmoraesdljdd@gmail.com' commit -m "feat: add calibrated defense operating points"
```

### Task 9: Publish and verify signed frozen defender bundles

**Files:**
- Create: `src/apar/defense/bundle.py`
- Create: `tests/defense/test_bundle.py`

**Interfaces:**
- Consumes: `ArtifactStore`, `RunSigningIdentity`, feature/rule/model/calibration/threshold artifacts
- Produces: `DefenderBundleManifest`, `LoadedDefenderBundle`, `DefenderBundlePublisher`
- Produces: `DefenderBundlePublisher.freeze(...) -> tuple[DefenderBundleManifest, ArtifactRef]`
- Produces: `DefenderBundlePublisher.load(ref: ArtifactRef) -> LoadedDefenderBundle`
- Produces: `DefenderBundlePublisher.verify(manifest) -> bool`

- [ ] **Step 1: Write failing signature, tamper, catalog, and reload tests**

```python
def test_frozen_bundle_rejects_threshold_tampering(bundle_fixture: BundleFixture) -> None:
    manifest, _ = bundle_fixture.publisher.freeze(**bundle_fixture.kwargs)
    changed = manifest.model_copy(update={"threshold_digest": "0" * 64})
    assert bundle_fixture.publisher.verify(changed) is False


def test_loaded_bundle_reproduces_frozen_scores(bundle_fixture: BundleFixture) -> None:
    manifest, ref = bundle_fixture.publisher.freeze(**bundle_fixture.kwargs)
    loaded = bundle_fixture.publisher.load(ref)
    assert loaded.manifest == manifest
    np.testing.assert_array_equal(loaded.scorer.predict(bundle_fixture.matrix), bundle_fixture.expected_scores)
```

- [ ] **Step 2: Run bundle tests and confirm failure**

Run: `.venv/bin/python -m pytest tests/defense/test_bundle.py -q`

Expected: collection fails because `apar.defense.bundle` does not exist.

- [ ] **Step 3: Implement content-addressed publication and signature verification**

```python
class DefenderBundleManifest(ExternalContract):
    schema_version: Literal["1.0.0"] = "1.0.0"
    bundle_id: str
    feature_catalog_digest: str
    training_matrix_digest: str
    split_manifest_digest: str
    rule_manifest_digest: str
    model_digest: str
    training_receipt_digest: str
    calibration_digest: str
    threshold_digest: str
    environment_digest: str
    source_inventory_digest: str
    rollback_ref: str
    signer_key_id: str
    public_key_base64: str
    signature_base64: str
    frozen_at: datetime
```

Publish model bytes as `application/vnd.apar.catboost-model`, all numeric
artifacts as canonical JSON, and the ordered training matrix as Parquet. Sign the
manifest's canonical unsigned document with `RunSigningIdentity`. Loading must
verify signature, every digest/media type, exact feature order/catalog digest,
schema versions, CatBoost major version, and reload-score fixture before returning
a scorer. Do not deserialize pickle or arbitrary Python objects.

- [ ] **Step 4: Run bundle and artifact-store tests**

Run: `.venv/bin/python -m pytest tests/defense/test_bundle.py tests/storage/test_artifacts.py -q`

Expected: valid publish/load passes; changed model, threshold, feature order,
environment, signature, media type, and rollback reference fail closed.

- [ ] **Step 5: Commit frozen defender bundles**

```bash
git add src/apar/defense/bundle.py tests/defense/test_bundle.py
git -c user.name='Dylan Moraes' -c user.email='dylanmoraesdljdd@gmail.com' commit -m "feat: freeze signed defender bundles"
```

### Task 10: Add past-only case grouping and review queue simulation

**Files:**
- Create: `src/apar/cases/__init__.py`
- Create: `src/apar/cases/grouping.py`
- Create: `src/apar/cases/queue.py`
- Create: `tests/cases/__init__.py`
- Create: `tests/cases/test_grouping.py`
- Create: `tests/cases/test_queue.py`

**Interfaces:**
- Consumes: `ObservedEvent`, `DefenseDecision`
- Produces: `InvestigationCase`, `CaseSnapshot`, `QueueConfig`, `QueueReport`
- Produces: `group_cases(observations, decisions, as_of) -> tuple[InvestigationCase, ...]`
- Produces: `simulate_case_queue(cases, config) -> QueueReport`

- [ ] **Step 1: Write failing grouping, future-extension, and capacity tests**

```python
def test_shared_counterparty_groups_alerts_without_future_evidence(
    suspicious_fixture: SuspiciousFixture,
) -> None:
    cases = group_cases(
        suspicious_fixture.observations,
        suspicious_fixture.decisions,
        as_of=suspicious_fixture.cutoff,
    )
    assert len(cases) == 1
    assert cases[0].event_ids == suspicious_fixture.expected_event_ids


def test_future_event_cannot_change_an_earlier_queue_priority(queue_fixture: QueueFixture) -> None:
    before = simulate_case_queue(queue_fixture.cases, queue_fixture.config)
    after = simulate_case_queue(queue_fixture.cases + (queue_fixture.future_case,), queue_fixture.config)
    assert after.snapshots[: len(before.snapshots)] == before.snapshots
```

- [ ] **Step 2: Run case tests and confirm failure**

Run: `.venv/bin/python -m pytest tests/cases -q`

Expected: collection fails for missing `apar.cases`.

- [ ] **Step 3: Implement deterministic causal grouping and queueing**

```python
class QueueConfig(ExternalContract):
    analyst_count: int = Field(default=2, ge=1)
    service_minutes_per_case: int = Field(default=20, ge=1)
    sla_minutes: int = Field(default=240, ge=1)
    bucket_minutes: int = Field(default=60, ge=1)


class InvestigationCase(ExternalContract):
    case_id: str
    opened_at: datetime
    event_ids: tuple[str, ...]
    actor_ids: tuple[str, ...]
    counterparty_ids: tuple[str, ...]
    first_alert_at: datetime
    priority: float
    estimated_minutes: int
```

Group non-approve decisions by connected actor/counterparty edges available by
`as_of`. Derive deterministic case IDs from the sorted first evidence set. Future
events can create a later snapshot but cannot rewrite an earlier one. Compute
priority from maximum score, score-times-current-amount expected value, graph
coverage, and recency using past-only inputs; never use evaluator truth. Tie-break
by case ID. The queue uses hourly
capacity `analyst_count * 60 / service_minutes_per_case`, carries backlog forward,
and records start, completion, wait, SLA breach, and analyst minutes.

- [ ] **Step 4: Run case tests**

Run: `.venv/bin/python -m pytest tests/cases -q`

Expected: shared actor/counterparty, isolated cases, deterministic IDs, future
extension, exact capacity, backlog, SLA, stable ties, and empty queue tests pass.

- [ ] **Step 5: Commit case and workload simulation**

```bash
git add src/apar/cases tests/cases
git -c user.name='Dylan Moraes' -c user.email='dylanmoraesdljdd@gmail.com' commit -m "feat: add defense case workload simulation"
```

### Task 11: Implement operational, value, calibration, alert-time, and latency metrics

**Files:**
- Create: `src/apar/evaluation/metrics.py`
- Create: `tests/evaluation/test_metrics_classification.py`
- Create: `tests/evaluation/test_metrics_value.py`
- Create: `tests/evaluation/test_metrics_operations.py`

**Interfaces:**
- Consumes: truth rows, defense decisions, cases, queue report, latency samples
- Produces: `MetricValue`, `ClassificationMetrics`, `CalibrationMetrics`, `ValueMetrics`, `AlertMetrics`, `OperationalMetrics`, `MetricReport`
- Produces: `compute_metric_report(...) -> MetricReport`
- Produces: `campaign_bootstrap(report_inputs, seed=260816, replicates=1000) -> ConfidenceIntervals`

- [ ] **Step 1: Write hand-calculated classification, value, and censored-alert tests**

```python
def test_hand_calculated_classification_and_false_interventions() -> None:
    report = compute_metric_report(**four_row_metric_fixture())
    assert report.classification.precision.value == pytest.approx(2 / 3)
    assert report.classification.recall.value == 1.0
    assert report.classification.f1.value == pytest.approx(0.8)
    assert report.operations.false_interventions_per_10k == 5000.0


def test_preventable_value_requires_a_pre_settlement_decline() -> None:
    report = compute_metric_report(**lifecycle_value_fixture())
    assert report.value.fraudulent_net_settled_value == Decimal("100.00")
    assert report.value.preventable_settled_value == Decimal("60.00")
    assert report.value.value_escaped == Decimal("40.00")
    assert report.value.challenge_credited_as_prevented == Decimal("0.00")


def test_undetected_campaign_remains_censored() -> None:
    report = compute_metric_report(**undetected_campaign_fixture())
    assert report.alerts.undetected_campaigns == 1
    assert report.alerts.p95_seconds.value is None
    assert report.alerts.p95_seconds.undefined_reason == "insufficient_detected_campaigns"
```

- [ ] **Step 2: Run metric tests and confirm failure**

Run: `.venv/bin/python -m pytest tests/evaluation/test_metrics_classification.py tests/evaluation/test_metrics_value.py tests/evaluation/test_metrics_operations.py -q`

Expected: collection fails for incomplete `apar.evaluation.metrics`.

- [ ] **Step 3: Implement exact metric semantics**

```python
class MetricValue(ExternalContract):
    value: float | None
    numerator: float
    denominator: float
    undefined_reason: str | None = None


class ValueMetrics(ExternalContract):
    fraudulent_net_settled_value: Decimal
    preventable_settled_value: Decimal
    value_escaped: Decimal
    value_before_first_alert: Decimal
    remaining_preventable_at_alert: Decimal
    challenge_credited_as_prevented: Decimal = Decimal("0.00")
```

Use scikit-learn only for `average_precision_score`, `roc_auc_score`, and the
calibration slope/intercept fit. Compute precision/recall/F1, rates, Brier, frozen
equal-frequency ECE bins, value lifecycle netting, false actions per 10,000,
challenge rate, review cases per 100,000, analyst minutes, backlog, SLA, time to
first alert, value before alert, and p50/p90/p95/p99 latencies explicitly. Return
`MetricValue(None, ..., reason)` for absent classes or insufficient detected
campaigns. Bootstrap whole campaigns with NumPy `PCG64(260816)` for exactly 1000
replicates and stable percentile interpolation.

- [ ] **Step 4: Run all metric tests**

Run: `.venv/bin/python -m pytest tests/evaluation/test_metrics_*.py -q`

Expected: all hand oracles, zero denominators, lifecycle de-duplication, recovery,
unsupported challenge, censored alert, queue workload, calibration, AUC, latency,
and deterministic bootstrap tests pass.

- [ ] **Step 5: Commit operational metrics**

```bash
git add src/apar/evaluation/metrics.py tests/evaluation/test_metrics_*.py
git -c user.name='Dylan Moraes' -c user.email='dylanmoraesdljdd@gmail.com' commit -m "feat: add defense operational metrics"
```

### Task 12: Replay three arms and enforce promotion/hidden gates

**Files:**
- Create: `src/apar/evaluation/replay.py`
- Create: `src/apar/evaluation/gates.py`
- Create: `src/apar/evaluation_hidden/defense_authority.py`
- Modify: `src/apar/evaluation_hidden/__init__.py`
- Create: `tests/evaluation/test_replay.py`
- Create: `tests/evaluation/test_gates.py`
- Create: `tests/evaluation/test_hidden_boundary.py`

**Interfaces:**
- Consumes: frozen corpus/split/features, rules, loaded defender, calibrator, per-arm thresholds
- Produces: `DefenseArm`, `ReplayResult`, `GateConfig`, `GateResult`, `ChampionDecision`
- Produces: `replay_defense_arms(...) -> tuple[ReplayResult, ...]`
- Produces: `evaluate_promotion_gates(results, config) -> ChampionDecision`
- Produces: `HiddenEvaluationAuthority.freeze_and_issue(manifest) -> HiddenEvaluationCapability`

- [ ] **Step 1: Write failing matched-row, common-integrity, hidden-freeze, and family-veto tests**

```python
def test_all_arms_score_the_exact_same_decision_rows(replay_fixture: ReplayFixture) -> None:
    results = replay_defense_arms(**replay_fixture.kwargs)
    assert {result.arm for result in results} == set(DefenseArm)
    assert len({result.decision_event_ids for result in results}) == 1


def test_hidden_reference_cannot_resolve_before_bundle_freeze(hidden_fixture: HiddenFixture) -> None:
    with pytest.raises(HiddenBoundaryError, match="frozen defender"):
        hidden_fixture.authority.resolve(hidden_fixture.restricted_ref)


def test_one_strategic_family_below_floor_vetoes_promotion(passing_results) -> None:
    changed = weaken_family_recall(passing_results, "agentic_intent_abuse", 0.49)
    decision = evaluate_promotion_gates(changed, GateConfig.competition())
    assert decision.status == "no_promotion"
    assert "PER_FAMILY_RECALL" in decision.failed_gate_codes
```

- [ ] **Step 2: Run replay/gate tests and confirm failure**

Run: `.venv/bin/python -m pytest tests/evaluation/test_replay.py tests/evaluation/test_gates.py tests/evaluation/test_hidden_boundary.py -q`

Expected: collection fails for missing replay/gate modules.

- [ ] **Step 3: Implement identical-row three-arm replay**

```python
class DefenseArm(StrEnum):
    RULES_ONLY = "rules_only"
    GBDT_ONLY = "gbdt_only"
    LAYERED_HYBRID = "layered_hybrid"


class GateConfig(ExternalContract):
    challenge_rate_max: float = 0.02
    false_decline_rate_max: float = 0.001
    review_case_rate_max: float = 0.01
    minimum_family_recall: float = 0.50
    maximum_ece: float = 0.10
    maximum_p95_latency_ms: float = 50.0
    maximum_slice_recall_regression: float = 0.05
    minimum_value_improvement: Decimal = Decimal("0.01")
```

Compute mandatory integrity decisions once and reuse them in all arms. Rules-only
uses `RuleResult.risk_score`; GBDT-only uses calibrated model score; hybrid uses
mandatory/hard rules first and `max(rule_score, calibrated_score)` for remaining
actions. Each arm uses its frozen matched-budget thresholds. A model error is an
audited GBDT-only failure and a declared rules-only hybrid fallback.

- [ ] **Step 4: Implement hard blockers and truthful champion selection**

Block promotion on leakage/parity failure, invalid artifact/signature, missing
rollback, hidden access violation, budget breach, ECE above `0.10`, p95 latency
above `50ms`, any family recall below `0.50`, or slice recall regression above
`0.05`. Promote hybrid only when it improves preventable settled value by at least
`0.01` over both comparators, or matches the best value within `0.01` while
strictly reducing review workload without another regression. Otherwise retain
the best passing comparator or return `no_promotion`; never average away a hard
failure.

The hidden authority must accept only a verified signed bundle manifest, seal its
digest into a capability, and permit the evaluator—not `apar.defense` or
`apar.features`—to resolve restricted event refs. Add an AST/import audit proving
those packages do not import `apar.evaluation_hidden`.

- [ ] **Step 5: Run replay, gate, and hidden-boundary tests**

Run: `.venv/bin/python -m pytest tests/evaluation/test_replay.py tests/evaluation/test_gates.py tests/evaluation/test_hidden_boundary.py -q`

Expected: exact-row fairness, common integrity, fallback, budget matching,
family/slice veto, negative-result `no_promotion`, signature freeze, and static
hidden import tests pass.

- [ ] **Step 6: Commit replay and promotion assurance**

```bash
git add src/apar/evaluation/replay.py src/apar/evaluation/gates.py src/apar/evaluation_hidden/defense_authority.py src/apar/evaluation_hidden/__init__.py tests/evaluation
git -c user.name='Dylan Moraes' -c user.email='dylanmoraesdljdd@gmail.com' commit -m "feat: add matched defense replay and gates"
```

### Task 13: Publish judge reports and expose the public API

**Files:**
- Create: `src/apar/evaluation/reporting.py`
- Create: `src/apar/evaluation/service.py`
- Create: `src/apar/api/routes/defense.py`
- Modify: `src/apar/api/dependencies.py`
- Modify: `src/apar/api/app.py`
- Create: `tests/evaluation/test_reporting.py`
- Create: `tests/api/test_defense.py`

**Interfaces:**
- Consumes: replay results, metric reports, gates, lineage refs, `ArtifactStore`
- Produces: `DefenseScorecard`, `EvaluationArtifactBundle`, `DefenseEvaluationService`
- Produces: `publish_scorecard(...) -> tuple[DefenseScorecard, EvaluationArtifactBundle]`
- API: `POST /api/v1/defense/evaluations`
- API: `GET /api/v1/defense/evaluations/{evaluation_id}`
- API: `GET /api/v1/defense/evaluations/{evaluation_id}/artifacts/{name}`

- [ ] **Step 1: Write failing report-lineage and restricted-API tests**

```python
def test_every_leaderboard_metric_resolves_to_an_artifact(report_fixture) -> None:
    scorecard, bundle = publish_scorecard(**report_fixture.kwargs)
    assert bundle.public_artifacts["defense-scorecard.json"].sha256
    assert scorecard.core_digest == canonical_digest(scorecard.core_document())
    assert all(row.metric_artifact_sha256 for row in scorecard.leaderboard)
    assert "synthetic" in scorecard.external_validity_statement.lower()


def test_public_api_never_returns_truth_predictions_or_hidden_refs(client, evaluation_id) -> None:
    response = client.get(f"/api/v1/defense/evaluations/{evaluation_id}")
    body = json.dumps(response.json(), sort_keys=True).lower()
    assert response.status_code == 200
    assert all(token not in body for token in (
        "evaluation_truth", "per_decision_predictions", "restricted_hidden",
    ))
```

- [ ] **Step 2: Run reporting/API tests and confirm failure**

Run: `.venv/bin/python -m pytest tests/evaluation/test_reporting.py tests/api/test_defense.py -q`

Expected: collection fails for missing reporting/service/route modules.

- [ ] **Step 3: Implement canonical report publication**

```python
class DefenseScorecard(ExternalContract):
    schema_version: Literal["1.0.0"] = "1.0.0"
    evaluation_id: str
    defender_bundle_id: str
    corpus_digest: str
    split_digest: str
    leaderboard: tuple[LeaderboardRow, ...]
    slice_summaries: tuple[SliceSummary, ...]
    gate_result: GateResult
    public_artifacts: dict[str, ArtifactRef]
    failed_checks: tuple[str, ...]
    limitations: tuple[str, ...]
    external_validity_statement: str
    core_digest: str
```

Publish canonical JSON, Markdown, leaderboard CSV, slice CSV, calibration CSV,
value/workload CSV, feature manifest, thresholds, data card, model card, and
limitations. Keep truth and per-decision predictions restricted and absent from
the public manifest. Exclude wall-clock timestamps and latency samples from the
byte-reproducible core digest; publish latency evidence separately with its
environment. `DefenseScorecard.public_artifacts` contains every public artifact
except the scorecard itself. `EvaluationArtifactBundle.public_artifacts` adds the
final `defense-scorecard.json` reference, avoiding any self-referential digest.

- [ ] **Step 4: Add the artifact-backed service and routes**

Initialize `DefenseEvaluationService` in the FastAPI lifespan using the existing
artifact store and run signer. POST accepts only 64-character corpus and defender
artifact digests, validates them, executes a frozen evaluation, and returns `201`
with the public scorecard. GET revalidates the scorecard digest. Named artifact
GET permits only the public allowlist and returns `404` for restricted names.
Map invalid digests to `404`, invalid artifacts to `422`, and gate/execution
failures to `409` without exposing paths or hidden reasons.

- [ ] **Step 5: Run report, API, and existing API regressions**

Run: `.venv/bin/python -m pytest tests/evaluation/test_reporting.py tests/api -q`

Expected: public reports validate, every displayed metric traces to an artifact,
restricted data remains absent, tampered reports fail, and existing health,
registry, scenario, and run endpoints remain green.

- [ ] **Step 6: Commit reporting and API handoff**

```bash
git add src/apar/evaluation/reporting.py src/apar/evaluation/service.py src/apar/api tests/evaluation/test_reporting.py tests/api/test_defense.py
git -c user.name='Dylan Moraes' -c user.email='dylanmoraesdljdd@gmail.com' commit -m "feat: publish defense judge scorecards"
```

### Task 14: Add reproducible CLIs and the G3 integration gate

**Files:**
- Create: `config/defense/competition-profile.json`
- Create: `scripts/generate_defense_runs.py`
- Create: `scripts/build_defense_corpus.py`
- Create: `scripts/train_defender.py`
- Create: `scripts/evaluate_defender.py`
- Create: `scripts/verify_g3.py`
- Create: `tests/integration/test_g3_defense.py`
- Create: `tests/defense/test_cli.py`

**Interfaces:**
- Consumes: compiled scenario refs, authenticated run refs, profile JSON, artifact root
- Produces: reproducible corpus, frozen defender, development evaluation, hidden evaluation, scorecard refs
- CLI outputs one canonical JSON document on stdout and diagnostics on stderr

- [ ] **Step 1: Write failing CLI determinism and real-run integration tests**

```python
def test_g3_fixture_consumes_real_authenticated_run_artifacts(tmp_path: Path) -> None:
    result = run_g3_fixture(tmp_path)
    assert result.run_manifests_verified == 4
    assert result.arms == ("rules_only", "gbdt_only", "layered_hybrid")
    assert result.scorecard_ref.sha256 == result.public_artifacts["defense-scorecard.json"].sha256


def test_hidden_phase_refuses_an_unfrozen_defender(tmp_path: Path) -> None:
    completed = subprocess.run(
        [sys.executable, "scripts/evaluate_defender.py", "--phase", "hidden",
         "--defender", "0" * 64, "--root", str(tmp_path)],
        cwd=ROOT, text=True, capture_output=True,
    )
    assert completed.returncode != 0
    assert "frozen defender" in completed.stderr.lower()
```

- [ ] **Step 2: Run CLI/integration tests and confirm failure**

Run: `.venv/bin/python -m pytest tests/defense/test_cli.py tests/integration/test_g3_defense.py -q`

Expected: tests fail because the commands do not exist.

- [ ] **Step 3: Implement strict profile-driven commands**

The competition profile must contain these exact values:

```json
{
  "schema_version": "1.0.0",
  "families": [
    "agentic_intent_abuse",
    "app_scam_mule",
    "card_testing_cnp",
    "synthetic_merchant_refund"
  ],
  "campaigns_per_family": 50,
  "seed_bases": {
    "agentic_intent_abuse": 260000,
    "app_scam_mule": 261000,
    "card_testing_cnp": 262000,
    "synthetic_merchant_refund": 263000
  },
  "simulation_start_utc": "2026-01-01T00:00:00Z",
  "campaign_spacing_days": 8,
  "partition_campaign_indices": {
    "train": [0, 24],
    "calibrator_fit": [25, 31],
    "threshold_selection": [32, 37],
    "development_test": [38, 49]
  },
  "label_delay_days": 7,
  "gbdt": {
    "depths": [4, 6],
    "learning_rates": [0.03, 0.08],
    "l2_leaf_regs": [3.0, 8.0],
    "iterations": 300
  },
  "calibration": {
    "candidates": ["sigmoid", "isotonic"],
    "minimum_class_count": 50,
    "ece_bins": 10
  },
  "model_seed": 260816,
  "bootstrap_seed": 260816,
  "bootstrap_replicates": 1000,
  "budgets": {
    "challenge_rate_max": 0.02,
    "false_decline_rate_max": 0.001,
    "review_case_rate_max": 0.01
  },
  "gates": {
    "minimum_family_recall": 0.5,
    "maximum_ece": 0.1,
    "maximum_p95_latency_ms": 50.0,
    "maximum_slice_recall_regression": 0.05
  },
  "regimes": [
    "prevalence_dilution",
    "missing_optional",
    "availability_delay",
    "compressed_bursts",
    "benign_amount_shift",
    "cold_id_remap"
  ]
}
```

`generate_defense_runs.py` accepts only the committed preregistration, artifact
root, signer path, and output-ledger path. It compiles the declared scenarios,
executes exactly 50 fixed-policy query-budget-one runs per family through
`RunRunner`, verifies every completed manifest, and atomically writes the ordered
run-ID/artifact-digest ledger. For campaign index `i` in `[0, 49]`, use seed
`seed_bases[family] + i` and simulation start
`simulation_start_utc + (i * campaign_spacing_days)`. The eight-day spacing is
a preregistered maturity embargo: with the seven-day label delay, every earlier
partition is label-mature before the next partition begins.
The four index ranges in `partition_campaign_indices` are inclusive and assign
whole campaigns; leave-one-family-out training repeats the same time partitions
while excluding the held-out family. `build_defense_corpus.py` accepts only
profile, run-manifest digest list, root, and output-manifest path.
`train_defender.py` accepts only development corpus, catalog, profile, root, and
rollback ref. `evaluate_defender.py --phase development`
must not resolve hidden refs; `--phase hidden` requires a verified frozen bundle
and completed development scorecard. Every command validates exact JSON schema,
uses atomic artifact publication, and never overwrites an existing result.

The G3 integration fixture must still consume four real authenticated APAR runs,
one per family, but use `CorpusProfile.fixture()` with
`calibration.minimum_class_count=2` and reduced fixture-only evidence counts.
The fixture profile is never accepted by the competition export commands.
`verify_g3.py` first validates the immutable competition-profile values above,
then runs the lightweight fixture path; Task 15 supplies the full 200-campaign
evidence gate.

- [ ] **Step 4: Implement the one-command G3 verifier**

`scripts/verify_g3.py` must run, in order:

```python
CHECKS = (
    ("G0", [sys.executable, "scripts/verify_g0.py"]),
    ("G1_G2", [sys.executable, "scripts/verify_g1_g2.py"]),
    ("FEATURES", [sys.executable, "-m", "pytest", "tests/features", "-q"]),
    ("DEFENSE", [sys.executable, "-m", "pytest", "tests/defense", "-q"]),
    ("CASES", [sys.executable, "-m", "pytest", "tests/cases", "-q"]),
    ("EVALUATION", [sys.executable, "-m", "pytest", "tests/evaluation", "-q"]),
    ("G3", [sys.executable, "-m", "pytest", "tests/integration/test_g3_defense.py", "-q"]),
)
```

It must exit nonzero on the first failure and print exactly this final line only
after every check passes:

`G3 PASS: causal features, rules/GBDT/hybrid, matched budgets, frozen hidden evaluation, and judge scorecards`

- [ ] **Step 5: Run the G3 fixture gate**

Run: `.venv/bin/python scripts/verify_g3.py`

Expected: the command ends with the exact G3 PASS line and preserves all frozen
Task 6 hash/equivalence tests.

- [ ] **Step 6: Commit the executable G3 gate**

```bash
git add config/defense/competition-profile.json scripts tests/defense/test_cli.py tests/integration/test_g3_defense.py
git -c user.name='Dylan Moraes' -c user.email='dylanmoraesdljdd@gmail.com' commit -m "test: establish G3 defense assurance gate"
```

### Task 15: Preregister and freeze the competition evidence bundle

**Files:**
- Create: `docs/experiments/defense-v1-preregistration.json`
- Create: `docs/experiments/defense-v1-run-manifests.json`
- Create: `fixtures/defense/v1/corpus-manifest.json`
- Create: `fixtures/defense/v1/hash-manifest.json`
- Create: `fixtures/defense/v1/observations.parquet`
- Create: `fixtures/defense/v1/evaluation-truth.parquet`
- Create: `fixtures/defense/v1/features.parquet`
- Create: `fixtures/defense/v1/split-manifest.json`
- Create: `fixtures/defense/v1/feature-manifest.json`
- Create: `fixtures/defense/v1/rules.json`
- Create: `fixtures/defense/v1/model.cbm`
- Create: `fixtures/defense/v1/training-receipt.json`
- Create: `fixtures/defense/v1/calibration.json`
- Create: `fixtures/defense/v1/thresholds.json`
- Create: `fixtures/defense/v1/defender-bundle.json`
- Create: `fixtures/defense/v1/defense-scorecard.json`
- Create: `fixtures/defense/v1/defense-scorecard.md`
- Create: `fixtures/defense/v1/leaderboard.csv`
- Create: `fixtures/defense/v1/slice-metrics.csv`
- Create: `fixtures/defense/v1/calibration.csv`
- Create: `fixtures/defense/v1/value-workload.csv`
- Create: `fixtures/defense/v1/model-card.md`
- Create: `fixtures/defense/v1/data-card.md`
- Create: `fixtures/defense/v1/limitations.md`
- Create: `docs/experiments/defense-v1-result.json`
- Create: `tests/evaluation/test_frozen_defense_v1.py`
- Modify: `README.md`
- Modify: `docs/TRACEABILITY.md`
- Modify: `docs/superpowers/plans/IMPLEMENTATION_TRACEABILITY.md`

**Interfaces:**
- Consumes: completed Tasks 1-14 and exactly 200 authenticated campaign runs
- Produces: a portable frozen dataset/model/threshold/report bundle and hash manifest
- Preserves: all Task 6 evidence and validation-spike bytes

- [ ] **Step 1: Write and validate the preregistration before producing results**

The preregistration must repeat the exact profile values from Task 14, list all
200 scenario/run seeds and intended chronological windows. The exact seeds are
`260000..260049`, `261000..261049`, `262000..262049`, and
`263000..263049` in the family order in the competition profile. Campaign index
`i` starts at `2026-01-01T00:00:00Z + (i * 8 days)`; indices `0..24` are training,
`25..31` calibrator fit, `32..37` threshold selection, and `38..49`
development test. The eight-day spacing is the fixed embargo for the seven-day
label delay. It must declare the CatBoost
grid, calibration selection rule, threshold tie-breaks, value semantics, gates,
hidden-release rule, artifact paths, stopping rule, and negative-result policy.
It must state that no model, feature, seed, threshold, or gate can change after
hidden evaluation begins.

Run: `.venv/bin/python -m json.tool docs/experiments/defense-v1-preregistration.json`

Expected: valid JSON is printed and the file contains no result metrics.

- [ ] **Step 2: Commit the preregistration separately**

```bash
git add docs/experiments/defense-v1-preregistration.json
git -c user.name='Dylan Moraes' -c user.email='dylanmoraesdljdd@gmail.com' commit -m "docs: preregister defense v1 evaluation"
```

- [ ] **Step 3: Build and freeze the development corpus**

Run:

```bash
.venv/bin/python scripts/generate_defense_runs.py \
  --preregistration docs/experiments/defense-v1-preregistration.json \
  --root .apar/defense-v1 \
  --signer .apar/defense-v1/run-signing-key.ed25519 \
  --output docs/experiments/defense-v1-run-manifests.json

.venv/bin/python scripts/build_defense_corpus.py \
  --profile config/defense/competition-profile.json \
  --run-manifests docs/experiments/defense-v1-run-manifests.json \
  --root .apar/defense-v1 \
  --output-manifest fixtures/defense/v1/corpus-manifest.json
```

Expected: exactly 200 verified campaigns, 50 per family; observation/truth bytes
are separate; catalog/provenance audits pass; exported corpus artifacts and
manifest hashes are written once.

- [ ] **Step 4: Train, calibrate, select thresholds, and freeze before hidden access**

Run:

```bash
.venv/bin/python scripts/train_defender.py \
  --corpus fixtures/defense/v1/corpus-manifest.json \
  --catalog config/defense/feature-catalog.json \
  --profile config/defense/competition-profile.json \
  --rollback-ref rules-v1 \
  --root .apar/defense-v1 \
  --export fixtures/defense/v1
```

Expected: model, receipt, calibration, thresholds, feature/split manifests, and
signed defender bundle exist; development test and hidden refs remain unread.

- [ ] **Step 5: Run development evaluation and inspect blockers without retuning**

Run:

```bash
.venv/bin/python scripts/evaluate_defender.py \
  --phase development \
  --corpus fixtures/defense/v1/corpus-manifest.json \
  --defender fixtures/defense/v1/defender-bundle.json \
  --profile config/defense/competition-profile.json \
  --root .apar/defense-v1 \
  --export fixtures/defense/v1
```

Expected: chronological, cold, four LOFO, and six regime results are published.
Any failure remains in the report; do not alter the preregistered bundle.

- [ ] **Step 6: Resolve hidden events only through the frozen authority and finalize results**

Run:

```bash
.venv/bin/python scripts/evaluate_defender.py \
  --phase hidden \
  --corpus fixtures/defense/v1/corpus-manifest.json \
  --defender fixtures/defense/v1/defender-bundle.json \
  --profile config/defense/competition-profile.json \
  --root .apar/defense-v1 \
  --export fixtures/defense/v1 \
  --hash-manifest fixtures/defense/v1/hash-manifest.json \
  --result docs/experiments/defense-v1-result.json
```

Expected: the result records hidden release after the signed bundle timestamp,
includes success and failure gates, states synthetic-only limitations, and never
contains restricted hidden parameters or per-decision truth. The command writes
`hash-manifest.json` last, after hashing every other exported fixture.

- [ ] **Step 7: Pin every exported artifact and test the frozen bundle**

```python
def test_frozen_defense_v1_hashes_and_scorecard_are_portable() -> None:
    manifest = load_frozen_manifest(FIXTURE_ROOT / "hash-manifest.json")
    for relative_path, expected_sha256 in manifest.artifact_sha256.items():
        payload = (FIXTURE_ROOT / relative_path).read_bytes()
        assert hashlib.sha256(payload).hexdigest() == expected_sha256
    scorecard = DefenseScorecard.model_validate_json(
        (FIXTURE_ROOT / "defense-scorecard.json").read_text()
    )
    assert scorecard.external_validity_statement.startswith("Synthetic APAR")
```

Run: `.venv/bin/python -m pytest tests/evaluation/test_frozen_defense_v1.py -q`

Expected: all artifact hashes, model reload scores, threshold budgets, scorecard
schema, restricted-field scan, and frozen-before-hidden timestamps pass.

- [ ] **Step 8: Update status and traceability without changing historical evidence**

Update README status to mark the Defend G3 subsystem implemented. Add G3 commands,
artifact paths, three-arm comparison, hidden-freeze evidence, operational metrics,
and synthetic limitations to both traceability files. Keep the validation spike
described as supporting historical evidence.

Run: `git diff --name-only codex/apar-foundation...HEAD`

Expected: no path under `validation_spike/`, no `docs/experiments/task6-*` path,
and no Task 6-frozen source path appears.

- [ ] **Step 9: Run final proportional verification**

Run:

```bash
.venv/bin/python scripts/verify_g3.py
.venv/bin/python -m pytest -q
.venv/bin/python -m ruff check src tests scripts
.venv/bin/python -m mypy src
```

Expected: G3 ends with the exact PASS line; the full suite, lint, and strict typing
pass; frozen v1 artifact tests pass without network access.

- [ ] **Step 10: Commit the frozen competition evidence and documentation**

```bash
git add fixtures/defense/v1 docs/experiments/defense-v1-run-manifests.json docs/experiments/defense-v1-result.json tests/evaluation/test_frozen_defense_v1.py README.md docs/TRACEABILITY.md docs/superpowers/plans/IMPLEMENTATION_TRACEABILITY.md
git -c user.name='Dylan Moraes' -c user.email='dylanmoraesdljdd@gmail.com' commit -m "test: freeze defense v1 competition evidence"
```

## Plan Completion Gate

Implementation is complete only when all fifteen tasks are committed, the full
test/lint/type suite passes, the 200-campaign corpus and defender artifacts are
portable and hash-pinned, the model is frozen before hidden access, all three
defense arms are compared under identical budgets, failures remain visible, the
judge scorecard resolves to immutable evidence, and no frozen Task 6 or validation
spike byte changes.
