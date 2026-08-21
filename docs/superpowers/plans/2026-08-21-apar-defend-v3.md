# APAR Defend v3 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> superpowers:subagent-driven-development (recommended) or
> superpowers:executing-plans to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a separately versioned, evaluator-owned execution path for the
already sealed Defend v2 scientific protocol without modifying v1 or v2 evidence.

**Architecture:** V3 is additive. It supplies an encrypted seed ledger, sealed
adversarial and operating populations, process-isolated defender execution, a
one-attempt confirmatory runner, signed receipts, conservative gate evaluation,
and stable JSON/CSV publication. It reuses v2's fixed metrics, budgets, gates,
stopping rules, arm definitions, and tie-break order without alteration.

**Tech Stack:** Python 3.12, Pydantic 2, NumPy, pandas, PyArrow, CatBoost CPU,
cryptography, multiprocessing, pytest.

**Spec:** `docs/superpowers/specs/2026-08-21-apar-defend-v3-design.md`

## Global Constraints

- Do not modify any file under `docs/experiments/defense-v1-*`,
  `fixtures/defense/v1/`, Task 6 evidence, or existing G0--G3 evidence.
- Do not modify any v2 config, implementation module, verifier, endpoint, or
  test. V3 must import v2 contracts only as immutable read-only inputs.
- All data is synthetic-only. Every report contains:
  `Synthetic-only evaluation; not a real-world prevalence or external-validity claim.`
- Preserve strict `available_at < decision_at` causality and campaign, entity,
  time, family, and partition isolation.
- Defender code never receives labels, hidden seeds, stratum assignments,
  outcomes, evaluator keys, receipt stores, network access, or shared Python
  objects.
- Use canonical JSON bytes, SHA-256, Ed25519 signatures, and authenticated
  encryption for seed material. Never use pickle across a boundary.
- Arms receive identical observations, vectors, partitions, case grouping,
  latency environment, candidate-grid shape, budgets, and stopping rules.
- Operating strata remain exactly low=100 fraud / 100,000 transactions,
  medium=500 / 100,000, high=1,000 / 100,000, allocated equally by family.
- Bootstrap exactly 2,000 times by synthetic day then campaign/entity case
  block; hard gates use the conservative 95% bound; undefined metrics fail.
- Exactly one confirmatory attempt is allowed after independent review and
  explicit user approval. Any failure produces truthful `no_promotion`.
- Before every commit run `git status --short`, stage task-owned paths only, and
  commit as `Dylan Moraes <dylanmoraesdljdd@gmail.com>` with no AI attribution.

## Locked File Map

```text
config/defense/v3/profile.json                         Public v3 execution profile
config/defense/v3/preregistration.json                 Signed public preregistration
src/apar/v3_protocol.py                                Defender-safe public contract
src/apar/evaluation/v3_seed_ledger.py                  Encrypted evaluator seed ledger
src/apar/evaluation/v3_population.py                   Efficacy and operating populations
src/apar/evaluation/v3_isolation.py                    Process-isolation capability manifest
src/apar/evaluation/v3_runtime.py                      Fresh-process defender adapter
src/apar/evaluation/v3_replay.py                       Matched three-arm replay orchestration
src/apar/evaluation/v3_metrics.py                      Metric projection and bootstrap bridge
src/apar/evaluation/v3_controls.py                     Mandatory control runner
src/apar/evaluation/v3_runner.py                       One-attempt confirmatory runner
src/apar/evaluation/v3_receipt.py                      Atomic signed execution receipts
src/apar/evaluation/v3_reporting.py                    Completed scorecard renderer
src/apar/evaluation/v3_preexecution.py                 Read-only readiness verifier
scripts/run_defense_v3_confirmatory.py                 Explicitly gated execution CLI
scripts/verify_defense_v3_preexecution.py              Read-only status CLI
tests/evaluation/test_defense_v3_*.py                  Focused v3 tests
tests/integration/test_defense_v3_preexecution.py      Sealed not-executed golden path
```

### Task 1: Freeze the additive v3 boundary

**Files:**
- Create: `src/apar/v3_protocol.py`
- Create: `tests/evaluation/test_defense_v3_protocol.py`

- [ ] Step 1: Write failing tests requiring a distinct protocol ID, schema
  version, synthetic non-claim, all twelve named seed commitments, one maximum
  confirmatory attempt, exact v2 metric/budget/gate inheritance, and rejection of
  v1 or v2 identifiers as v3 inputs.

- [ ] Step 2: Run
  `.venv/bin/python -m pytest tests/evaluation/test_defense_v3_protocol.py -q`
  and confirm collection failure.

- [ ] Step 3: Implement a frozen public protocol contract with canonical digest
  helpers, immutable budget and gate values, seed commitment names, and explicit
  compatibility references to v2 contracts. The module must not import evaluator
  packages.

- [ ] Step 4: Run the focused test and verify it passes.

- [ ] Step 5: Commit only these paths:

```bash
git add src/apar/v3_protocol.py tests/evaluation/test_defense_v3_protocol.py
git -c user.name='Dylan Moraes' -c user.email='dylanmoraesdljdd@gmail.com' commit -m 'feat: add defend v3 protocol boundary'
```

### Task 2: Seal the encrypted seed ledger

**Files:**
- Create: `src/apar/evaluation/v3_seed_ledger.py`
- Create: `tests/evaluation/test_defense_v3_seed_ledger.py`

- [ ] Step 1: Write failing tests for authenticated encryption, key identity,
  nonce uniqueness, payload digest binding, exact seed names, commitment
  verification, tamper rejection, wrong-key rejection, replay rejection, and
  disclosure only after population completion.

- [ ] Step 2: Run the focused test and confirm failure.

- [ ] Step 3: Implement an AES-GCM or XChaCha20-Poly1305 ledger using
  cryptography primitives. Store no plaintext seed in any manifest. Derive each
  public commitment from the canonical seed document. Bind every seed to its
  population role and completion state.

- [ ] Step 4: Run the focused test and verify deterministic commitments and
  fail-closed decryption.

- [ ] Step 5: Commit:

```bash
git add src/apar/evaluation/v3_seed_ledger.py tests/evaluation/test_defense_v3_seed_ledger.py
git -c user.name='Dylan Moraes' -c user.email='dylanmoraesdljdd@gmail.com' commit -m 'feat: seal defend v3 seed ledger'
```

### Task 3: Build disjoint efficacy and operating populations

**Files:**
- Create: `src/apar/evaluation/v3_population.py`
- Create: `tests/evaluation/test_defense_v3_population.py`

- [ ] Step 1: Write failing tests for exact operating denominators, equal family
  allocation, benign replacement semantics, campaign coherence, past-only
  decisions, unique IDs, entity/time/campaign/family disjointness, chronological
  holdout, cold-entity cohorts, regime slices, held-out-family exclusion, and
  immutable manifests.

- [ ] Step 2: Run the focused test and confirm failure.

- [ ] Step 3: Reuse v2 operating population construction unchanged where
  possible. Add the adversarial efficacy builder and a separate disjointness
  auditor that emits signed per-partition manifests and a combined proof.

- [ ] Step 4: Run the focused test and verify all partitions are complete,
  isolated, causal, and deterministic under committed seeds.

- [ ] Step 5: Commit:

```bash
git add src/apar/evaluation/v3_population.py tests/evaluation/test_defense_v3_population.py
git -c user.name='Dylan Moraes' -c user.email='dylanmoraesdljdd@gmail.com' commit -m 'feat: build defend v3 sealed populations'
```

### Task 4: Enforce fresh-process defender isolation

**Files:**
- Create: `src/apar/evaluation/v3_isolation.py`
- Create: `src/apar/evaluation/v3_runtime.py`
- Create: `tests/evaluation/test_defense_v3_isolation.py`
- Create: `tests/evaluation/test_defense_v3_runtime.py`

- [ ] Step 1: Write failing tests proving a child process has no evaluator
  modules loaded, cannot import them, has no signing key or seed, cannot read the
  receipt store, cannot open sockets, rejects pickle/shared objects/callbacks,
  and receives only canonical framed bytes bound to protocol ID and nonce.

- [ ] Step 2: Run both focused tests and confirm failure.

- [ ] Step 3: Implement a capability manifest and subprocess adapter using a
  clean interpreter, minimal environment, closed descriptors, resource limits,
  audit hooks, input/output size limits, deadlines, and digest verification.
  Reject symlinked source and non-regular Python files.

- [ ] Step 4: Run both focused tests and verify timeout, crash, malformed frame,
  oversized payload, forbidden import, and network attempts fail closed.

- [ ] Step 5: Commit:

```bash
git add src/apar/evaluation/v3_isolation.py src/apar/evaluation/v3_runtime.py tests/evaluation/test_defense_v3_isolation.py tests/evaluation/test_defense_v3_runtime.py
git -c user.name='Dylan Moraes' -c user.email='dylanmoraesdljdd@gmail.com' commit -m 'feat: isolate defend v3 defender runtime'
```

### Task 5: Project matched metrics and bootstrap intervals

**Files:**
- Create: `src/apar/evaluation/v3_metrics.py`
- Create: `tests/evaluation/test_defense_v3_metrics.py`

- [ ] Step 1: Write failing tests projecting actions, scores, truth, workload,
  value, alert time, latency, campaign reconstruction, chronological slices,
  cold-entity cohorts, regimes, and held-out families into exact v2-compatible
  metric sets and day/case bootstrap blocks.

- [ ] Step 2: Run the focused test and confirm failure.

- [ ] Step 3: Implement pure projection functions that reuse v2 metric and
  selection types without changing their definitions. Produce aggregate, strata,
  family, regime, cohort, and held-out-family evidence plus exactly 2,000
  two-level bootstrap replicates.

- [ ] Step 4: Run the focused test and verify row permutation stability,
  undefined-denominator failure, conservative bounds, and deterministic seeds.

- [ ] Step 5: Commit:

```bash
git add src/apar/evaluation/v3_metrics.py tests/evaluation/test_defense_v3_metrics.py
git -c user.name='Dylan Moraes' -c user.email='dylanmoraesdljdd@gmail.com' commit -m 'feat: project defend v3 matched metrics'
```

### Task 6: Run mandatory controls

**Files:**
- Create: `src/apar/evaluation/v3_controls.py`
- Create: `tests/evaluation/test_defense_v3_controls.py`

- [ ] Step 1: Write failing tests for benign zero-fraud claims, intervention
  reporting, block-preserving score permutation, forged signature rejection,
  context replay rejection, malformed evaluator rejection, and apparently
  qualifying permuted efficacy invalidation.

- [ ] Step 2: Run the focused test and confirm failure.

- [ ] Step 3: Adapt v2 control primitives to v3 bindings without weakening
  signature, identity, block, or validity checks.

- [ ] Step 4: Run the focused test and verify both controls are load bearing.

- [ ] Step 5: Commit:

```bash
git add src/apar/evaluation/v3_controls.py tests/evaluation/test_defense_v3_controls.py
git -c user.name='Dylan Moraes' -c user.email='dylanmoraesdljdd@gmail.com' commit -m 'feat: run defend v3 negative controls'
```

### Task 7: Add atomic receipts and one-attempt runner

**Files:**
- Create: `src/apar/evaluation/v3_receipt.py`
- Create: `src/apar/evaluation/v3_runner.py`
- Create: `scripts/run_defense_v3_confirmatory.py`
- Create: `tests/evaluation/test_defense_v3_receipt.py`
- Create: `tests/evaluation/test_defense_v3_runner.py`

- [ ] Step 1: Write failing tests for approval-token requirement, source/config/
  bundle/population digest binding, atomic receipt creation, crash consumption,
  duplicate execution rejection, nonce mismatch rejection, malformed input
  termination, control failure termination, gate failure termination, and
  absence of partial result publication.

- [ ] Step 2: Run both focused tests and confirm failure.

- [ ] Step 3: Implement a CLI that refuses execution unless every pre-execution
  check passes and an explicit approval token matches the sealed freeze digest.
  Atomically write the signed receipt before running. Execute arms through the
  isolated runtime, evaluate controls and gates, then publish once or record
  truthful `no_promotion`.

- [ ] Step 4: Run both focused tests and verify exactly-once semantics and
  fail-closed behavior.

- [ ] Step 5: Commit:

```bash
git add src/apar/evaluation/v3_receipt.py src/apar/evaluation/v3_runner.py scripts/run_defense_v3_confirmatory.py tests/evaluation/test_defense_v3_receipt.py tests/evaluation/test_defense_v3_runner.py
git -c user.name='Dylan Moraes' -c user.email='dylanmoraesdljdd@gmail.com' commit -m 'feat: gate defend v3 confirmatory execution'
```

### Task 8: Render completed signed evidence

**Files:**
- Create: `src/apar/evaluation/v3_reporting.py`
- Create: `tests/evaluation/test_defense_v3_reporting.py`

- [ ] Step 1: Write failing tests for all required metrics, calibration bins,
  Brier score, workload reconstruction, campaign reconstruction, slice results,
  conservative gates, failed-arm visibility, synthetic limitations, canonical
  JSON/CSV snapshots, signature verification, and champion gating.

- [ ] Step 2: Run the focused test and confirm failure.

- [ ] Step 3: Render seven stable artifacts: scorecard JSON, arm metrics CSV,
  workload CSV, gates JSON, limitations Markdown, signed artifact manifests, and
  signed execution receipt. Preserve all three arms and every failed or undefined
  value.

- [ ] Step 4: Run the focused test and verify `promotion_eligible` requires every
  gate pass and a frozen defender; hidden output remains absent until then.

- [ ] Step 5: Commit:

```bash
git add src/apar/evaluation/v3_reporting.py tests/evaluation/test_defense_v3_reporting.py
git -c user.name='Dylan Moraes' -c user.email='dylanmoraesdljdd@gmail.com' commit -m 'feat: render defend v3 competition evidence'
```

### Task 9: Verify readiness and preserve prior evidence

**Files:**
- Create: `src/apar/evaluation/v3_preexecution.py`
- Create: `scripts/verify_defense_v3_preexecution.py`
- Create: `tests/evaluation/test_defense_v3_preexecution.py`
- Create: `tests/integration/test_defense_v3_preexecution.py`
- Modify: `README.md`
- Modify: `docs/TRACEABILITY.md`

- [ ] Step 1: Write failing tests requiring exact v1/v2 hashes, no v3 receipt,
  valid preregistration, complete manifests, source inventory match, runtime
  capability match, and truthful `not_executed` status.

- [ ] Step 2: Run both focused suites and confirm failure.

- [ ] Step 3: Implement a read-only verifier and CLI. Add only:
  `Defend v3: execution path drafted; evaluation not executed.` Do not alter any
  prior conclusion or hash.

- [ ] Step 4: Run sequentially:

```bash
.venv/bin/python -m pytest -q
.venv/bin/python scripts/verify_g0.py
.venv/bin/python scripts/verify_g1_g2.py
.venv/bin/python scripts/verify_g3.py
.venv/bin/python scripts/verify_defense_v2_preexecution.py
.venv/bin/python scripts/verify_defense_v3_preexecution.py
```

Expected: full suite passes; G0--G3 pass; v2 remains admissible and
`not_executed`; v3 reports `not_executed`.

- [ ] Step 5: Request independent review before any execution approval.

- [ ] Step 6: Commit documentation and verifier paths:

```bash
git add README.md docs/TRACEABILITY.md src/apar/evaluation/v3_preexecution.py scripts/verify_defense_v3_preexecution.py tests/evaluation/test_defense_v3_preexecution.py tests/integration/test_defense_v3_preexecution.py
git -c user.name='Dylan Moraes' -c user.email='dylanmoraesdljdd@gmail.com' commit -m 'docs: draft defend v3 execution readiness'
```

## Plan Self-Review

| Requirement | Tasks |
| --- | --- |
| Separate versioned protocol | 1 |
| Sealed seed authority | 2 |
| Complete populations and isolation proofs | 3 |
| Process boundary and canonical IO | 4 |
| Matched metrics and uncertainty | 5 |
| Mandatory controls | 6 |
| One-attempt execution and receipts | 7 |
| Stable signed publication | 8 |
| Readiness review and evidence preservation | 9 |

No task executes a v3 evaluation. Execution begins only after independent review
and a second explicit user approval at the Task 7 approval boundary.
