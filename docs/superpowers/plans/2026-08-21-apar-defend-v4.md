# APAR Defend v4 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> superpowers:subagent-driven-development (recommended) or
> superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Wire actual defender scoring, metric computation, gate evaluation,
and signed scorecard generation into the v3 execution boundary without modifying
v1, v2, or v3 evidence.

**Architecture:** V4 is additive. It reuses the v3 execution boundary (seed
ledger, populations, isolation, runtime, metrics bridge, controls, receipts)
unchanged and adds a scoring adapter that loads the frozen v1 defender bundle
and produces real decisions for each arm.

**Tech Stack:** Python 3.12, Pydantic 2, NumPy, pandas, CatBoost CPU,
cryptography, multiprocessing, pytest.

**Spec:** `docs/superpowers/specs/2026-08-21-apar-defend-v4-design.md`

## Global Constraints

- Do not modify any v1, v2, or v3 evidence file, config, implementation module,
  verifier, endpoint, or test. V4 imports prior contracts as read-only inputs.
- All data is synthetic-only. Every report contains:
  `Synthetic-only evaluation; not a real-world prevalence or external-validity claim.`
- Preserve strict `available_at < decision_at` causality and campaign, entity,
  time, family, and partition isolation.
- Defender code never receives labels, hidden seeds, stratum assignments,
  outcomes, evaluator keys, receipt stores, network access, or shared Python
  objects.
- Use canonical JSON bytes, SHA-256, Ed25519 signatures. Never use pickle across
  a boundary.
- Arms receive identical observations, vectors, partitions, case grouping,
  latency environment, candidate-grid shape, budgets, and stopping rules.
- Bootstrap exactly 2,000 times by synthetic day then campaign/entity case
  block; hard gates use the conservative 95% bound; undefined metrics fail.
- Exactly one confirmatory attempt after independent review and explicit user
  approval. Any failure produces truthful `no_promotion`.
- Before every commit run `git status --short`, stage task-owned paths only, and
  commit as `Dylan Moraes <dylanmoraesdljdd@gmail.com>` with no AI attribution.

## Locked File Map

```text
src/apar/v4_protocol.py                                Defender-safe public contract
src/apar/evaluation/v4_scoring.py                      Observation/feature loading and arm scoring
src/apar/evaluation/v4_gate_evaluation.py              Conservative gate evaluation
src/apar/evaluation/v4_publication.py                  Signed scorecard generation
src/apar/evaluation/v4_runner.py                       One-attempt confirmatory runner
src/apar/evaluation/v4_preexecution.py                 Read-only readiness verifier
scripts/run_defense_v4_confirmatory.py                Explicitly gated execution CLI
scripts/verify_defense_v4_preexecution.py             Read-only status CLI
tests/evaluation/test_defense_v4_*.py                  Focused v4 tests
tests/integration/test_defense_v4_preexecution.py      Sealed not-executed golden path
```

### Task 1: Create the v4 protocol contract

**Files:**
- Create: `src/apar/v4_protocol.py`
- Create: `tests/evaluation/test_defense_v4_protocol.py`

- [ ] Step 1: Write failing tests requiring distinct protocol ID (`apar-defend-v4`),
  rejection of v1/v2/v3 identifiers, thirteen named seed commitments inherited
  from v3, one maximum confirmatory attempt, exact synthetic non-claim, frozen
  v1/v2/v3 root hashes, and no evaluator package imports.

- [ ] Step 2: Run focused test and confirm collection failure.

- [ ] Step 3: Implement the public protocol contract reusing v3 seed commitment
  names and adding v3 result/receipt roots to the frozen evidence set.

- [ ] Step 4: Run focused test and verify it passes.

- [ ] Step 5: Commit only these paths.

### Task 2: Implement the scoring adapter

**Files:**
- Create: `src/apar/evaluation/v4_scoring.py`
- Create: `tests/evaluation/test_defense_v4_scoring.py`

- [ ] Step 1: Write failing tests for observation loading from v3 population
  builders, feature vector construction using the frozen v1 48-feature catalog,
  past-only causality verification, rules-only scoring from frozen v1 rules.json,
  GBDT-only scoring from frozen model.cbm + calibration.json, layered-hybrid
  scoring with rule-first then GBDT fallback, threshold candidate evaluation,
  and matched arm input identity.

- [ ] Step 2: Run focused test and confirm failure.

- [ ] Step 3: Implement the scoring adapter that runs inside the v3 isolated
  subprocess. Load the frozen defender bundle, construct features, apply each
  arm's decision path, and return actions/scores/latencies as canonical JSON.

- [ ] Step 4: Run focused test and verify all three arms produce valid decisions
  on fixture populations with correct action precedence and past-only features.

- [ ] Step 5: Commit.

### Task 3: Implement conservative gate evaluation

**Files:**
- Create: `src/apar/evaluation/v4_gate_evaluation.py`
- Create: `tests/evaluation/test_defense_v4_gate_evaluation.py`

- [ ] Step 1: Write failing tests projecting scored decisions into v2-compatible
  metric sets via the v3 metrics bridge, running 2,000 bootstrap replicates,
  evaluating all eight fixed gates against conservative bounds, selecting
  thresholds by the preregistered objective/tie-break, and producing complete
  gate outcomes for all arms × strata × families.

- [ ] Step 2: Run focused test and confirm failure.

- [ ] Step 3: Implement pure gate evaluation functions that reuse v2 selection
  types and the v3 bootstrap bridge without changing any threshold or bound.

- [ ] Step 4: Run focused test and verify undefined metrics fail closed,
  conservative bounds are load-bearing, and tie-break order is deterministic.

- [ ] Step 5: Commit.

### Task 4: Implement signed scorecard publication

**Files:**
- Create: `src/apar/evaluation/v4_publication.py`
- Create: `tests/evaluation/test_defense_v4_publication.py`

- [ ] Step 1: Write failing tests for rendering all seven required artifacts
  (scorecard JSON, arm-metrics CSV, workload CSV, gates JSON, limitations MD,
  artifact manifests, execution receipt) with all metrics visible, failed gates
  retained, synthetic non-claim present, Ed25519 signature verified, and
  promotion_eligible requiring every gate pass plus a frozen defender.

- [ ] Step 2: Run focused test and confirm failure.

- [ ] Step 3: Adapt the v3 reporting renderer to v4 bindings and add metric-rich
  CSV columns for precision/recall/F1/PR-AUC/ROC-AUC/ECE/Brier/FPR/challenge/
  false-decline/review-case/false-interventions/value/time-to-alert/latency.

- [ ] Step 4: Run focused test and verify stable JSON/CSV snapshots and signature
  integrity.

- [ ] Step 5: Commit.

### Task 5: Add one-attempt runner and pre-execution verifier

**Files:**
- Create: `src/apar/evaluation/v4_runner.py`
- Create: `src/apar/evaluation/v4_preexecution.py`
- Create: `scripts/run_defense_v4_confirmatory.py`
- Create: `scripts/verify_defense_v4_preexecution.py`
- Create: `tests/evaluation/test_defense_v4_runner.py`
- Create: `tests/evaluation/test_defense_v4_preexecution.py`
- Create: `tests/integration/test_defense_v4_preexecution.py`
- Modify: `README.md`
- Modify: `docs/TRACEABILITY.md`

- [ ] Step 1: Write failing tests for approval-token requirement, source/config/
  bundle/population digest binding, atomic receipt creation, duplicate execution
  rejection, full pipeline integration (scoring → metrics → gates → scorecard),
  crash consumption, and absence of partial result publication.

- [ ] Step 2: Run both focused suites and confirm failure.

- [ ] Step 3: Implement the CLI that refuses execution unless every pre-execution
  check passes and an explicit approval token matches the sealed freeze digest.
  Atomically write the receipt before running. Execute scoring through the v3
  isolated runtime, evaluate controls and gates, render scorecards, then publish
  once or record truthful `no_promotion`.

- [ ] Step 4: Run sequentially:

```bash
.venv/bin/python -m pytest -q
.venv/bin/python scripts/verify_g0.py
.venv/bin/python scripts/verify_g1_g2.py
.venv/bin/python scripts/verify_g3.py
.venv/bin/python scripts/verify_defense_v2_preexecution.py
.venv/bin/python scripts/verify_defense_v3_preexecution.py
.venv/bin/python scripts/verify_defense_v4_preexecution.py
```

Expected: full suite passes; G0--G3 pass; v2 remains admissible/not_executed;
v3 reports consumed attempt; v4 reports not_executed.

- [ ] Step 5: Request independent review before any execution approval.

- [ ] Step 6: Commit documentation and verifier paths.

## Plan Self-Review

| Requirement | Tasks |
| --- | --- |
| Separate versioned protocol | 1 |
| Actual defender scoring | 2 |
| Conservative gate evaluation | 3 |
| Signed evidence publication | 4 |
| One-attempt execution and readiness | 5 |

No task executes a v4 evaluation. Execution begins only after independent review
and a second explicit user approval at the Task 5 approval boundary.
