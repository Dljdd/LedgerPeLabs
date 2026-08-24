# APAR Sentinel v5 Implementation Plan

> **For the implementing agent:** REQUIRED SUB-SKILL: use `superpowers:test-driven-development` for every behavior change and `superpowers:verification-before-completion` before reporting success. Use `superpowers:executing-plans` if it is available in the session.

**Goal:** Build a causally valid, campaign-aware, calibrated fraud defender and produce honest development evidence across all four APAR threat families without touching frozen prior evidence or running a sealed evaluation.

**Architecture:** A mixed benign/fraud simulator feeds strict history-only transaction and temporal-graph features into deterministic trust/rule checks, a calibrated three-seed CatBoost ensemble, and a novelty/uncertainty router. A four-action policy is evaluated chronologically, then hardened once with valid adaptive evasions and tested on untouched adaptive seeds.

**Tech stack:** Python 3.12, Pydantic v2, NumPy, pandas, scikit-learn, CatBoost, PyArrow, existing APAR simulator/rails/trust/red-team/evaluation packages, Pytest, Ruff, mypy.

**Design authority:** `docs/superpowers/specs/2026-08-22-apar-sentinel-v5-design.md`

---

## Global execution rules

1. Read the complete design authority and all referenced frozen v1/v2/Task 6/Task 7 contracts before editing code.
2. Start from `codex/apar-baseline` at or after `f1d1b355751d8e24d59a7a28e4a67e342d81cdf3` in a dedicated branch/worktree named `codex/apar-sentinel-v5`.
3. Preserve user changes. If the worktree is unexpectedly dirty, stop and report the exact paths before editing.
4. Write the failing test first, run it, record the intended RED, implement the smallest correct behavior, then run GREEN.
5. Never modify frozen v1/v2/Task 6/Task 7 evidence, preregistrations, result JSON, historical verifiers, or the detached v4 experiment.
6. Never run a sealed, hidden, one-shot, or confirmatory evaluation in this task.
7. Do not use family, campaign, seed, scenario, split, or outcome fields as predictive inputs.
8. Do not fake, coerce, fill, or suppress non-finite metrics. The readiness verdict must fail closed.
9. Do not add network calls or new dependencies unless the design cannot be implemented with the installed stack and Dylan explicitly approves the change.
10. Commit only under Dylan's identity:

```bash
git -c user.name="Dylan Moraes" \
    -c user.email="dylanmoraesdljdd@gmail.com" \
    commit -m "<message>"
```

Do not add AI co-author trailers or AI authorship metadata.

## Locked paths

Before the first code edit, record SHA-256 hashes for every existing file under these evidence paths and compare them again before completion:

```text
docs/experiments/task6-*
docs/experiments/defense-v1-*
docs/experiments/defense-v2-*
config/defense/defense-v1-*
config/defense/defense-v2-*
```

Also record the exact available paths rather than silently succeeding on an empty glob. The new v5 code may call historical portable verifiers but may not rewrite them.

## Planned file map

New production files:

```text
config/defense/defense-v5-development.json
config/defense/feature-catalog-v5.json
src/apar/evaluation/v5_protocol.py
src/apar/evaluation/v5_population.py
src/apar/evaluation/v5_fidelity.py
src/apar/features/sentinel.py
src/apar/defense/sentinel.py
src/apar/evaluation/v5_evaluation.py
src/apar/evaluation/v5_hardening.py
src/apar/evaluation/v5_reporting.py
scripts/run_defense_v5_development.py
scripts/verify_defense_v5_readiness.py
```

New tests:

```text
tests/evaluation/test_defense_v5_protocol.py
tests/evaluation/test_defense_v5_population.py
tests/evaluation/test_defense_v5_fidelity.py
tests/features/test_sentinel_features.py
tests/defense/test_sentinel_model.py
tests/evaluation/test_defense_v5_evaluation.py
tests/evaluation/test_defense_v5_controls.py
tests/evaluation/test_defense_v5_hardening.py
tests/evaluation/test_defense_v5_reporting.py
tests/evaluation/test_defense_v5_readiness.py
```

Development artifact written only after the implementation gates pass:

```text
docs/experiments/defense-v5-development-result.json
```

Do not modify the historical `config/defense/feature-catalog.json`; v5 owns a versioned catalog.

---

## Task 1: Freeze the v5 protocol and configuration

**Files:**

- Create: `src/apar/evaluation/v5_protocol.py`
- Create: `config/defense/defense-v5-development.json`
- Create: `tests/evaluation/test_defense_v5_protocol.py`

### Step 1: Write protocol RED tests

Define tests for exact immutable contracts:

- `V5Partition` with `TRAIN`, `CALIBRATION`, `THRESHOLD`, `DEVELOPMENT_TEST`, `HARDENING_TRAIN`, and `ADAPTIVE_HOLDOUT`;
- `V5Profile` with `SMOKE` and `PRODUCTION`;
- exact four-family enumeration matching existing Task 5 names;
- `V5ReadinessTargets` with recall `0.75`, false decline `0.001`, review `0.01`, challenge `0.02`, captured value `0.70`, ECE `0.10`, and p95 latency `50.0` ms;
- positive, bounded integer/decimal validation using exact built-in scalar types;
- unique, non-empty seed sets for every partition;
- production profile lower bounds of 50,000 legitimate untouched test decisions and 100 campaigns per family;
- at least 2,000 bootstrap replicates;
- strict rejection of unknown fields, numeric subclasses, booleans-as-integers, NaN, infinity, overlapping seed sets, weakened gates, and missing family entries;
- canonical JSON serialization and SHA-256 protocol digest stability.

The configuration loader must return an immutable `V5DevelopmentProtocol`. It must not accept command-line gate overrides.

### Step 2: Run RED

```bash
uv run --python 3.12.5 --with-editable . --extra dev \
  python -m pytest tests/evaluation/test_defense_v5_protocol.py -q
```

Expected RED: import/collection failure because `apar.evaluation.v5_protocol` does not exist.

### Step 3: Implement the protocol

Use frozen Pydantic models with `extra="forbid"` and explicit exact-type validators. The JSON config must declare:

- protocol ID and version;
- all partition seeds;
- family counts and legitimate counts per profile;
- CatBoost seed list;
- bootstrap seed/count;
- workload/readiness gates;
- permitted feature catalog path;
- development-only status;
- `sealed_evaluation_allowed: false`.

Expose one loader:

```python
def load_v5_development_protocol(path: Path) -> V5DevelopmentProtocol:
    """Load, validate, freeze, and digest the development protocol."""
```

No module-level mutable registries are allowed.

### Step 4: Run GREEN and static checks

```bash
uv run --python 3.12.5 --with-editable . --extra dev \
  python -m pytest tests/evaluation/test_defense_v5_protocol.py -q
uv run --python 3.12.5 --with-editable . --extra dev ruff check \
  src/apar/evaluation/v5_protocol.py tests/evaluation/test_defense_v5_protocol.py
uv run --python 3.12.5 --with-editable . --extra dev mypy --strict \
  src/apar/evaluation/v5_protocol.py
```

### Step 5: Commit

```bash
git add config/defense/defense-v5-development.json \
  src/apar/evaluation/v5_protocol.py \
  tests/evaluation/test_defense_v5_protocol.py
git -c user.name="Dylan Moraes" \
  -c user.email="dylanmoraesdljdd@gmail.com" \
  commit -m "feat: freeze sentinel v5 development protocol"
```

---

## Task 2: Build a mixed, group-disjoint population

**Files:**

- Create: `src/apar/evaluation/v5_population.py`
- Create: `tests/evaluation/test_defense_v5_population.py`
- Reuse: `src/apar/evaluation/v2_population.py`
- Reuse: `src/apar/generators/campaigns.py`
- Reuse: `src/apar/generators/population.py`

### Step 1: Write population RED tests

Tests must construct a small deterministic profile and prove:

- legitimate and fraudulent decisions coexist in every operational evaluation partition;
- all four families appear with exact requested campaign counts;
- every fraud row is derived from a Task 5 campaign command and real rail event, not from a target label alone;
- legitimate activity includes recurring, first-time, high-value, bursty, refund/return, challenge, and recovery cases where rail semantics permit them;
- actor, account, campaign, device/credential, and merchant/payee identities are disjoint across partitions;
- timestamps are monotonic within an entity history and partitions are chronologically ordered;
- split membership is deterministic for the same protocol and changes for a different development seed;
- synthetic ID renaming preserves labels, amounts, timing, topology, and all non-ID semantics;
- the predictive projection omits family, campaign, scenario, seed, split, generator, and future-outcome fields;
- impossible family counts, insufficient unique entities, duplicate commands, replay failure, ledger imbalance, or horizon overflow fail closed;
- `SMOKE` outputs are visibly marked and cannot satisfy production readiness.

### Step 2: Run RED

```bash
uv run --python 3.12.5 --with-editable . --extra dev \
  python -m pytest tests/evaluation/test_defense_v5_population.py -q
```

### Step 3: Implement population orchestration

Define immutable records for:

```python
class V5DecisionRow(BaseModel): ...
class V5EntityGroups(BaseModel): ...
class V5PartitionCorpus(BaseModel): ...
class V5PopulationManifest(BaseModel): ...
class V5Corpus(BaseModel): ...
```

Create:

```python
def build_v5_corpus(
    protocol: V5DevelopmentProtocol,
    *,
    profile: V5Profile,
) -> V5Corpus:
    """Build all partitions using real generators, rails, and strict group isolation."""
```

Implementation requirements:

- delegate campaign creation to existing Task 5 generators;
- delegate execution to the existing simulator/rail/ledger;
- reuse v2 benign-population mechanisms where valid;
- derive labels and economic outcomes from commands/events/ledger postings;
- use namespace/domain-separated seeds per partition and family;
- keep the raw audit projection separate from the sanitized model projection;
- reject any identity overlap and include exact overlap counts in the manifest;
- keep all timestamps UTC and canonical;
- produce a deterministic corpus digest.

### Step 4: Run GREEN and integration checks

```bash
uv run --python 3.12.5 --with-editable . --extra dev \
  python -m pytest \
  tests/evaluation/test_defense_v5_population.py \
  tests/generators -q
uv run --python 3.12.5 --with-editable . --extra dev ruff check \
  src/apar/evaluation/v5_population.py tests/evaluation/test_defense_v5_population.py
uv run --python 3.12.5 --with-editable . --extra dev mypy --strict \
  src/apar/evaluation/v5_population.py
```

### Step 5: Commit

```bash
git add src/apar/evaluation/v5_population.py \
  tests/evaluation/test_defense_v5_population.py
git -c user.name="Dylan Moraes" \
  -c user.email="dylanmoraesdljdd@gmail.com" \
  commit -m "feat: build mixed sentinel v5 population"
```

---

## Task 3: Enforce behavioral fidelity

**Files:**

- Create: `src/apar/evaluation/v5_fidelity.py`
- Create: `tests/evaluation/test_defense_v5_fidelity.py`

### Step 1: Write fidelity RED tests

Create deterministic valid and mutated corpora. Require checks for:

- amount and inter-arrival quantiles by legitimate/family stratum;
- hour/day distribution and relationship-newness rates;
- card probe-before-escalation lifecycle;
- APP fan-in/layer/fan-out/cash-out/return/freeze/recovery lifecycle;
- synthetic merchant purchase/settlement/refund/chargeback/recovery lifecycle;
- agentic trust-failure or valid-control lifecycle;
- graph degree, weighted degree, fan-in/fan-out, repeated edges, shared neighbors, component size, density, and motif counts;
- attempted/approved/settled/returned/recovered/net-attacker value reconciliation;
- mandatory failure for disconnected motifs, reversed causal order, missing terminal states, fragmented roles, target-only labels, or ledger imbalance;
- exact observed/reference/tolerance/pass fields for every check;
- `invalid_corpus` readiness status if any mandatory check fails.

Use metamorphic mutations that change one mechanism at a time. Do not assert only that a hash changed.

### Step 2: Run RED

```bash
uv run --python 3.12.5 --with-editable . --extra dev \
  python -m pytest tests/evaluation/test_defense_v5_fidelity.py -q
```

### Step 3: Implement the auditor

Define:

```python
class FidelityDimension(str, Enum):
    STATISTICAL = "statistical"
    TEMPORAL = "temporal"
    RELATIONAL = "relational"
    ECONOMIC = "economic"

class FidelityCheck(BaseModel): ...
class FidelityAudit(BaseModel): ...

def audit_v5_fidelity(
    corpus: V5Corpus,
    protocol: V5DevelopmentProtocol,
) -> FidelityAudit: ...
```

Post-run full-graph calculations must be confined to this audit module. They must not be imported by the online feature builder.

### Step 4: Run GREEN and static checks

```bash
uv run --python 3.12.5 --with-editable . --extra dev \
  python -m pytest \
  tests/evaluation/test_defense_v5_fidelity.py \
  tests/evaluation/test_defense_v5_population.py -q
uv run --python 3.12.5 --with-editable . --extra dev ruff check \
  src/apar/evaluation/v5_fidelity.py tests/evaluation/test_defense_v5_fidelity.py
uv run --python 3.12.5 --with-editable . --extra dev mypy --strict \
  src/apar/evaluation/v5_fidelity.py
```

### Step 5: Commit

```bash
git add src/apar/evaluation/v5_fidelity.py \
  tests/evaluation/test_defense_v5_fidelity.py
git -c user.name="Dylan Moraes" \
  -c user.email="dylanmoraesdljdd@gmail.com" \
  commit -m "feat: audit payment behavior fidelity"
```

---

## Task 4: Build the causal Sentinel feature projection

**Files:**

- Create: `config/defense/feature-catalog-v5.json`
- Create: `src/apar/features/sentinel.py`
- Create: `tests/features/test_sentinel_features.py`
- Reuse: `src/apar/features/state.py`

### Step 1: Write feature RED tests

Require a deterministic matrix containing the design-authority feature groups and prove:

- every historical source timestamp is strictly less than the decision timestamp;
- equal-time decisions do not observe one another;
- future insertion/permutation leaves earlier vectors byte-identical;
- input shuffling yields canonical row and column order;
- identifier renaming leaves all numeric features and model-ready categories unchanged;
- current decision outcomes, future lifecycle outcomes, labels, family, campaign, scenario, seed, split, and evaluator fields are absent;
- removing a probe, fan-in edge, shared device/credential, refund precursor, or prior trust failure changes only the expected downstream history features;
- cold/unseen categorical values use a declared unknown representation;
- malformed or non-finite features fail before model inference;
- catalog SHA-256 and matrix schema are deterministic across subprocesses.

### Step 2: Run RED

```bash
uv run --python 3.12.5 --with-editable . --extra dev \
  python -m pytest tests/features/test_sentinel_features.py -q
```

### Step 3: Implement catalog and builder

The catalog must version and type every feature. Add request/integrity, velocity, pair, entity, temporal-graph, and missingness features only where the current event contracts supply the needed public identifiers.

Expose:

```python
class SentinelFeatureBatch(BaseModel): ...

def build_sentinel_features(
    rows: Sequence[V5DecisionRow],
    *,
    catalog: SentinelFeatureCatalog,
) -> SentinelFeatureBatch: ...
```

Build features by walking canonical event time and updating `FeatureState` only after a timestamp cohort is emitted. Do not calculate online features from a completed full-corpus graph.

### Step 4: Run GREEN and existing feature regression

```bash
uv run --python 3.12.5 --with-editable . --extra dev \
  python -m pytest tests/features tests/features/test_sentinel_features.py -q
uv run --python 3.12.5 --with-editable . --extra dev ruff check \
  src/apar/features/sentinel.py tests/features/test_sentinel_features.py
uv run --python 3.12.5 --with-editable . --extra dev mypy --strict \
  src/apar/features/sentinel.py
```

### Step 5: Commit

```bash
git add config/defense/feature-catalog-v5.json \
  src/apar/features/sentinel.py \
  tests/features/test_sentinel_features.py
git -c user.name="Dylan Moraes" \
  -c user.email="dylanmoraesdljdd@gmail.com" \
  commit -m "feat: add causal campaign features"
```

---

## Task 5: Implement the calibrated ensemble and action policy

**Files:**

- Create: `src/apar/defense/sentinel.py`
- Create: `tests/defense/test_sentinel_model.py`
- Reuse: `src/apar/defense/gbdt.py`
- Reuse: existing calibration, rule, threshold, and policy modules under `src/apar/defense/`
- Reuse: `src/apar/trust/verifier.py`

### Step 1: Write model RED tests

Create a deterministic mixed fixture and prove:

- three independently seeded CatBoost members train on the same frozen feature schema;
- calibration is fit only on the calibration partition;
- ensemble probability is the mean of calibrated member probabilities;
- disagreement is deterministic and non-negative;
- Isolation Forest is fit on legitimate training rows only;
- novelty alone never produces `decline`;
- definitive trust failures cannot be overridden by a low model score;
- action ordering is approve < challenge < review/hold < decline/hold;
- thresholds are selected only from the threshold split under explicit false-decline, challenge, and review constraints;
- no feasible thresholds return an explicit infeasible result rather than silently relaxing constraints;
- save/load preserves predictions, actions, reasons, feature schema, and digests;
- inference rejects missing, extra, reordered, non-finite, or wrongly typed features;
- measured latency uses actual inference calls, not constants.

### Step 2: Run RED

```bash
uv run --python 3.12.5 --with-editable . --extra dev \
  python -m pytest tests/defense/test_sentinel_model.py -q
```

### Step 3: Implement the defender

Define immutable public contracts:

```python
class SentinelAction(str, Enum):
    APPROVE = "approve"
    CHALLENGE = "challenge"
    REVIEW_HOLD = "review_hold"
    DECLINE_HOLD = "decline_hold"

class SentinelDecision(BaseModel): ...
class SentinelThresholds(BaseModel): ...
class SentinelModelManifest(BaseModel): ...
class SentinelDefender: ...
```

The trainer must accept explicit train/calibration/threshold batches and explicit protocol seeds. It must return a frozen model bundle plus threshold-selection audit. Do not expose `y` through a generic feature dictionary.

Reason output must use stable reason families, not raw model internals or synthetic family names.

### Step 4: Run GREEN, deterministic replay, and static checks

```bash
uv run --python 3.12.5 --with-editable . --extra dev \
  python -m pytest \
  tests/defense/test_sentinel_model.py \
  tests/defense -q
uv run --python 3.12.5 --with-editable . --extra dev ruff check \
  src/apar/defense/sentinel.py tests/defense/test_sentinel_model.py
uv run --python 3.12.5 --with-editable . --extra dev mypy --strict \
  src/apar/defense/sentinel.py
```

### Step 5: Commit

```bash
git add src/apar/defense/sentinel.py tests/defense/test_sentinel_model.py
git -c user.name="Dylan Moraes" \
  -c user.email="dylanmoraesdljdd@gmail.com" \
  commit -m "feat: add calibrated sentinel ensemble"
```

---

## Task 6: Implement causal evaluation, controls, and ablations

**Files:**

- Create: `src/apar/evaluation/v5_evaluation.py`
- Create: `tests/evaluation/test_defense_v5_evaluation.py`
- Create: `tests/evaluation/test_defense_v5_controls.py`

### Step 1: Write evaluation RED tests

Test exact calculations for:

- confusion counts, precision, recall, F1, PR-AUC, ROC-AUC, ECE, and Brier score;
- false decline, challenge, and review rates using legitimate decisions as denominators;
- captured, escaped, returned, and recovered role-bound values from ledger evidence;
- campaign time-to-alert with `None` only when no alert occurs and an explicit censored flag;
- real p50/p95/p99 inference latency;
- per-family and aggregate strata;
- leave-one-family-out, held-parameter-region, cold-identity, and time-shift experiments;
- 2,000 campaign/group bootstrap replicates and deterministic intervals;
- explicit failure on non-finite mandatory metrics, empty legitimate denominators, missing families, duplicate decision IDs, or mismatched evidence.

Control tests must prove:

- label shuffling collapses discrimination on a known separable fixture;
- identity renaming preserves predictions;
- future and equal-time metamorphics preserve earlier predictions;
- family/campaign/split/generator columns never reach the model;
- graph, novelty, and rule ablations actually remove their target mechanism;
- benign-only control measures workload and cannot claim recall;
- fraud-only diagnostic is marked non-operational and cannot pass readiness;
- adaptive no-delta fixtures cannot claim improvement.

### Step 2: Run RED

```bash
uv run --python 3.12.5 --with-editable . --extra dev \
  python -m pytest \
  tests/evaluation/test_defense_v5_evaluation.py \
  tests/evaluation/test_defense_v5_controls.py -q
```

### Step 3: Implement evaluation

Expose:

```python
class V5Arm(str, Enum): ...
class V5MetricEstimate(BaseModel): ...
class V5StratumResult(BaseModel): ...
class V5EvaluationResult(BaseModel): ...

def evaluate_v5_arm(...) -> V5EvaluationResult: ...
def run_v5_controls(...) -> tuple[V5ControlResult, ...]: ...
def compare_v5_arms(...) -> V5ArmComparison: ...
```

Primary bootstrap sampling units are campaigns for fraud and legitimate actor/account groups for workload. Metric functions must report denominator and support alongside the estimate.

### Step 4: Run GREEN and evaluation regressions

```bash
uv run --python 3.12.5 --with-editable . --extra dev \
  python -m pytest \
  tests/evaluation/test_defense_v5_evaluation.py \
  tests/evaluation/test_defense_v5_controls.py \
  tests/evaluation/test_frozen_defense_v1.py -q
uv run --python 3.12.5 --with-editable . --extra dev ruff check \
  src/apar/evaluation/v5_evaluation.py \
  tests/evaluation/test_defense_v5_evaluation.py \
  tests/evaluation/test_defense_v5_controls.py
uv run --python 3.12.5 --with-editable . --extra dev mypy --strict \
  src/apar/evaluation/v5_evaluation.py
```

### Step 5: Commit

```bash
git add src/apar/evaluation/v5_evaluation.py \
  tests/evaluation/test_defense_v5_evaluation.py \
  tests/evaluation/test_defense_v5_controls.py
git -c user.name="Dylan Moraes" \
  -c user.email="dylanmoraesdljdd@gmail.com" \
  commit -m "feat: evaluate sentinel operating performance"
```

---

## Task 7: Add one-round adaptive hardening

**Files:**

- Create: `src/apar/evaluation/v5_hardening.py`
- Create: `tests/evaluation/test_defense_v5_hardening.py`
- Reuse: `src/apar/redteam/search.py`
- Reuse: `src/apar/redteam/policies.py`
- Reuse: `src/apar/redteam/benchmark.py`

### Step 1: Write hardening RED tests

Require:

- matched proposal, evaluation, network, and logical-time budgets across fixed, random, adaptive, and cached-LLM policies;
- candidates reconstructed from public Task 5-valid parameter bounds;
- production-rail replay and ledger conservation for every accepted candidate;
- only development-search seeds may influence hardening examples;
- adaptive-holdout seeds, campaign IDs, and results remain unavailable during retraining;
- successful evasions are deduplicated by canonical campaign semantics, not only IDs;
- the hardening subset is selected by a frozen rule before holdout execution;
- exactly one retraining/recalibration/threshold-selection round;
- baseline and hardened evaluation use identical holdout campaigns and decision ordering;
- negative-control/singleton search spaces yield no adaptive claim;
- all four families are reported even when an adaptive delta is unsupported;
- hardening cannot pass if legitimate false decline, challenge, review, calibration, or latency gates regress beyond protocol limits.

### Step 2: Run RED

```bash
uv run --python 3.12.5 --with-editable . --extra dev \
  python -m pytest tests/evaluation/test_defense_v5_hardening.py -q
```

### Step 3: Implement orchestration

Define:

```python
class V5HardeningCorpus(BaseModel): ...
class V5FamilyHardeningResult(BaseModel): ...
class V5HardeningResult(BaseModel): ...

def run_v5_adaptive_hardening(
    *,
    protocol: V5DevelopmentProtocol,
    baseline: SentinelDefender,
    corpus: V5Corpus,
) -> V5HardeningResult: ...
```

Do not modify Task 6 policies or search algorithms to favor the v5 defender. This module composes existing authority-issued search/evaluator capabilities and converts accepted development evasions into the sanitized hardening-training format.

### Step 4: Run GREEN and Task 6 regression

```bash
uv run --python 3.12.5 --with-editable . --extra dev \
  python -m pytest \
  tests/evaluation/test_defense_v5_hardening.py \
  tests/redteam -q
uv run --python 3.12.5 --with-editable . --extra dev ruff check \
  src/apar/evaluation/v5_hardening.py \
  tests/evaluation/test_defense_v5_hardening.py
uv run --python 3.12.5 --with-editable . --extra dev mypy --strict \
  src/apar/evaluation/v5_hardening.py
```

### Step 5: Commit

```bash
git add src/apar/evaluation/v5_hardening.py \
  tests/evaluation/test_defense_v5_hardening.py
git -c user.name="Dylan Moraes" \
  -c user.email="dylanmoraesdljdd@gmail.com" \
  commit -m "feat: harden sentinel against adaptive campaigns"
```

---

## Task 8: Produce reports and a fail-closed readiness verifier

**Files:**

- Create: `src/apar/evaluation/v5_reporting.py`
- Create: `scripts/run_defense_v5_development.py`
- Create: `scripts/verify_defense_v5_readiness.py`
- Create: `tests/evaluation/test_defense_v5_reporting.py`
- Create: `tests/evaluation/test_defense_v5_readiness.py`

### Step 1: Write reporting/verifier RED tests

Test that the canonical result contains:

- protocol, corpus, fidelity, feature, model, threshold, and code provenance digests;
- exact split supports and identity-overlap audit;
- all comparison arms, strata, metrics, denominators, intervals, and latency;
- all controls and ablations;
- baseline/hardened adaptive comparison;
- explicit failed readiness gates;
- `development_ready`, `development_not_ready`, or `invalid_corpus` only;
- a redacted presentation view with no hidden evaluator reasons, raw signatures, private identities, labels, seeds, or restricted refs.

The verifier must reject:

- missing/non-finite mandatory metrics;
- absent families or legitimate support;
- smoke results presented as production;
- failed fidelity with a ready verdict;
- weakened or changed targets;
- unsupported bootstrap unit/count;
- missing control/ablation results;
- digest mismatch or reordered/duplicated decisions;
- a result containing `winner`, `production_ready`, `competition_validated`, or `confirmatory_supported` claims;
- any indication that sealed evaluation was enabled or executed.

### Step 2: Run RED

```bash
uv run --python 3.12.5 --with-editable . --extra dev \
  python -m pytest \
  tests/evaluation/test_defense_v5_reporting.py \
  tests/evaluation/test_defense_v5_readiness.py -q
```

### Step 3: Implement reporting and scripts

`run_defense_v5_development.py` must support:

```text
--profile smoke
--profile production
--output <path>
--verify-only <existing-result>
```

It must not expose threshold, target, seed, result-status, or sealed-evaluation overrides.

`verify_defense_v5_readiness.py` must independently parse the JSON result, recompute every readiness gate from raw metric fields, verify all bound digests available in the artifact, and exit non-zero for invalid evidence. It must not import or call the training pipeline to decide the verdict.

### Step 4: Run GREEN and CLI negative tests

```bash
uv run --python 3.12.5 --with-editable . --extra dev \
  python -m pytest \
  tests/evaluation/test_defense_v5_reporting.py \
  tests/evaluation/test_defense_v5_readiness.py -q
uv run --python 3.12.5 --with-editable . --extra dev ruff check \
  src/apar/evaluation/v5_reporting.py \
  scripts/run_defense_v5_development.py \
  scripts/verify_defense_v5_readiness.py \
  tests/evaluation/test_defense_v5_reporting.py \
  tests/evaluation/test_defense_v5_readiness.py
uv run --python 3.12.5 --with-editable . --extra dev mypy --strict \
  src/apar/evaluation/v5_reporting.py \
  scripts/run_defense_v5_development.py \
  scripts/verify_defense_v5_readiness.py
```

### Step 5: Commit

```bash
git add src/apar/evaluation/v5_reporting.py \
  scripts/run_defense_v5_development.py \
  scripts/verify_defense_v5_readiness.py \
  tests/evaluation/test_defense_v5_reporting.py \
  tests/evaluation/test_defense_v5_readiness.py
git -c user.name="Dylan Moraes" \
  -c user.email="dylanmoraesdljdd@gmail.com" \
  commit -m "feat: report sentinel v5 development evidence"
```

---

## Task 9: Run smoke, then production development evidence

**Files:**

- Create only after successful execution: `docs/experiments/defense-v5-development-result.json`
- Modify only if needed to fix an observed implementation bug: new v5 files and their tests

### Step 1: Run the complete v5 test slice before any experiment

```bash
uv run --python 3.12.5 --with-editable . --extra dev \
  python -m pytest \
  tests/evaluation/test_defense_v5_protocol.py \
  tests/evaluation/test_defense_v5_population.py \
  tests/evaluation/test_defense_v5_fidelity.py \
  tests/features/test_sentinel_features.py \
  tests/defense/test_sentinel_model.py \
  tests/evaluation/test_defense_v5_evaluation.py \
  tests/evaluation/test_defense_v5_controls.py \
  tests/evaluation/test_defense_v5_hardening.py \
  tests/evaluation/test_defense_v5_reporting.py \
  tests/evaluation/test_defense_v5_readiness.py -q
```

### Step 2: Execute and verify smoke evidence

Write the complete seed-404 evidence envelope to an absent temporary path, not
`docs/experiments`. The direct-file command is canonical; module invocation is
an equivalent tested entrypoint and neither mode requires `PYTHONPATH` mutation:

```bash
.venv/bin/python scripts/build_defense_v5_safe_evidence.py \
  --root . \
  --output /private/tmp/apar-v5-safe-404.json
.venv/bin/python scripts/verify_defense_v5_evidence.py \
  --root . \
  /private/tmp/apar-v5-safe-404.json
.venv/bin/python -m scripts.build_defense_v5_safe_evidence \
  --root . \
  --output /private/tmp/apar-v5-safe-404-module.json
```

Expected behavior: independently repeated builds have the same
`deterministic_core_sha256`. Real latency samples, their authenticated
`observational_latency_sha256`, the payload SHA, and the envelope SHA may differ.
The complete serialized artifact is not claimed to be byte-reproducible.

### Step 3: Freeze the development code state

Record:

- HEAD commit and tree;
- protocol/config/catalog SHA-256;
- all v5 production and script file SHA-256 values;
- Python and dependency versions;
- a clean tracked worktree.

Commit behavior, source, tests, and documentation as the SOURCE commit. Build
and independently verify two safe artifacts at that clean commit. They must
share a deterministic core while retaining authentic, independently verified
observational latency. Then add only
`config/defense/defense-v5-locked-development-preregistration.json` in the
direct child PREREGISTRATION commit. The one-time pre-execution audit command
is:

```bash
.venv/bin/python scripts/verify_defense_v5_locked_preexecution.py \
  --root . \
  --safe-evidence /private/tmp/apar-v5-approved-safe-evidence.json \
  --approved-commit HEAD
```

The audit must reject any other deterministic core, non-manifest change,
descendant HEAD, changed SOURCE mode/content, changed historical result,
existing candidate/chunks/summary, altered production support, or changed
protocol/config/catalog/verifier binding. It may accept fresh, independently
verified latency observations from the pinned environment because latency is
not the safe-core commitment. It must report seed `404` as the only executed
seed and seed `2404` as asserted only.

Do not edit model, thresholds, targets, or population parameters after reading the production development-test result. If a genuine execution bug occurs, preserve the failed result outside the canonical path, add a failing regression test, fix the bug, and rerun the entire development experiment with a new explicitly recorded attempt ID.

### Step 4: Execute the production development run

```bash
.venv/bin/python scripts/run_defense_v5_locked_development.py \
  --root . \
  --safe-evidence /private/tmp/apar-v5-approved-safe-evidence.json \
  --approved-commit HEAD \
  --authorize-exactly-once
```

This exact command is a development run only. It must not be executed while
building the SOURCE or PREREGISTRATION commits. It has no arbitrary seed,
profile, output, retry, or resume surface. It writes chunks followed by exactly
one exclusive candidate manifest at
`docs/experiments/defense-v5-locked-development-candidate.manifest.json`; it
never writes or replaces the historical result.

### Step 5: Independently verify the result

```bash
.venv/bin/python scripts/verify_defense_v5_locked_evidence.py \
  --root . \
  docs/experiments/defense-v5-locked-development-candidate.manifest.json
```

The verifier reconstructs the bounded chunked payload and independently
recomputes support, lineage, arms, controls, metrics, economics, calibration,
bootstrap intervals, deterministic core, observational latency, and readiness.
Accept `development_not_ready` or an invalid result as honest outcomes. Do not
change thresholds, targets, labels, seeds, or verdict logic to obtain
`development_ready`.

### Step 6: Commit raw development evidence unchanged

Do not commit a monolithic payload. Preserve the manifest and its ordered
64 MiB chunks according to the preregistered durability contract, verify every
chunk remains below 100 MiB and the complete payload remains below 1 GiB, and
only then create a separate compact judge-facing summary. No summary claim may
exist before the raw candidate has been durably published and independently
verified.

---

## Task 10: Full verification, self-review, and handoff

**Files:**

- Modify: this plan only if exact executed evidence needs a factual completion appendix
- Do not modify: canonical development result after its evidence commit

### Step 1: Run the full repository gates

```bash
uv run --python 3.12.5 --with-editable . --extra dev python -m pytest -q
uv run --python 3.12.5 --with-editable . --extra dev ruff check src tests scripts
uv run --python 3.12.5 --with-editable . --extra dev mypy --strict src scripts
uv run --python 3.12.5 --with-editable . --extra dev python scripts/verify_g0.py
uv run --python 3.12.5 --with-editable . --extra dev python scripts/verify_g1_g2.py
```

Run the existing Task 6 portable postcommit verification with the exact frozen commit/SHA arguments documented by its current help/report. Do not call an execute or confirmatory mode.

### Step 2: Verify historical evidence isolation

Compare the pre-task locked-path manifest against current bytes. Fail if:

- a locked path was added, removed, renamed, or changed unexpectedly;
- the validation spike changed;
- the detached v4 worktree was merged;
- a historical result or preregistration changed.

Then run:

```bash
git diff --check
git status --short
```

### Step 3: Perform adversarial self-review

Review the diff as a skeptical fraud scientist, production risk owner, and competition judge. At minimum ask:

- Can any identifier, family marker, generator artifact, future outcome, or equal-time peer leak the label?
- Are benign and fraudulent denominators correct?
- Could a fraud-only or smoke result pass readiness?
- Are graph features causal rather than full-corpus features?
- Are undefined metrics surfaced rather than coerced?
- Is novelty ever able to decline by itself?
- Did any threshold or target change after a holdout was read?
- Are adaptive budgets and holdout opportunities matched?
- Is hardening evaluated on campaigns not used for retraining?
- Can a report claim readiness without fidelity, controls, ablations, latency, and uncertainty?
- Are vendor/research claims clearly distinguished from APAR evidence?
- Did any code path run a sealed experiment?

For every issue found, add a failing regression test before fixing it and rerun all affected gates.

### Step 4: Final handoff report

Report:

- branch and exact commits;
- files added/modified;
- RED and GREEN evidence per task;
- corpus sizes and split supports;
- fidelity verdict;
- all arm and family results with intervals;
- readiness verdict and exact failed gates;
- adaptive baseline/hardened result;
- complete gate counts;
- locked-evidence hash comparison;
- confirmation that no sealed/confirmatory run occurred;
- any remaining limitations and the next recommended task.

Do not call the system competition-winning, production-ready, or validated. The next task after a successful v5 development run is an independent review and a separately preregistered confirmatory protocol; the walkthrough/web prototype consumes only the redacted view.
