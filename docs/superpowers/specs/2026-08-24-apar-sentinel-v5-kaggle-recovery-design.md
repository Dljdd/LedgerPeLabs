# APAR Sentinel v5: Kaggle Staged Recovery Design

**Status:** Approved design; implementation and seed-2404 execution remain unauthorized

**Date:** 2026-08-24

**Owner:** Dylan Moraes (`@Dljdd`)

**Rejected execution baseline:** `141e19c754c7bf54514eece5f7ff498ba53cd945` plus the durable failed-attempt receipt

**Preserved implementation source:** `6c3f7bef7e66f32c9a7156e710ee2ac953d499db`

## 1. Decision

Replace the rejected single-process Sentinel v5 production runner with a private,
Kaggle-specific staged execution protocol. The successor execution will use nine
separately saved Kaggle notebook versions. Every completed stage produces an
immutable, content-addressed checkpoint which becomes the read-only input to the
next stage.

This is a successor attempt under a new preregistration, not a retry or resume of
the crashed local attempt. The prior receipt remains terminal evidence.

The staged protocol changes execution and storage orchestration only. It must not
change:

- the production profile or development-test seed 2404;
- population, campaign, simulator, rail, ledger, or TrustVerifier semantics;
- feature names, arm semantics, model seeds, calibration, or thresholds;
- the seven controls or their criteria;
- metric, economic, bootstrap, uncertainty, or gate definitions;
- historical results, rejection records, safe evidence, or prior freeze commits.

## 2. Motivation and failure boundary

The first authorized locked-development attempt wrote its durable receipt at
2026-08-24T13:32:54.948338Z and was interrupted by a macOS kernel watchdog panic
before any candidate manifest, evidence chunks, or judge summary were published.
At the panic boundary, the production Python process retained approximately
6.25 GiB on an 8 GiB machine. The old architecture held corpus generation,
training, scoring, controls, payload construction, verification, and publication
inside one process, so the host failure destroyed all in-memory progress.

The old receipt proves that attempt was consumed. It must never be removed,
replaced, repaired, or accepted as authority for a successor execution.

The recovery design has two goals:

1. move the workload to Kaggle's larger free CPU runtime; and
2. ensure a later platform interruption loses only the current incomplete stage,
   never a previously completed stage.

## 3. Authority and non-goals

### 3.1 Authorized design work

This design authorizes implementation, test-safe local execution, private Kaggle
setup, and two production-size seed-404 rehearsals after the implementation plan
is separately approved.

### 3.2 Work still requiring later authorization

The design does not authorize:

- reading seed 2404 outside validation of its frozen binding;
- generating any seed-2404 population or campaign;
- training, scoring, running controls, or computing metrics with seed 2404;
- publishing a development result;
- adaptive, sealed, confirmatory, or production-result workflows;
- exposing any intermediate or final outcome.

After the new preregistration and exact pre-execution audit pass, Stage 00 still
requires one explicit user authorization. Stage 10, where seed 2404 first enters
population generation, cannot start until Stage 00 is durably published and
verified.

### 3.3 Kaggle privacy

All source bundles, dependency bundles, notebooks, notebook outputs, datasets,
and checkpoint artifacts must remain private. No source, checkpoint, metric,
result, or judge summary may be published to a public Kaggle page.

Credentials and API tokens must never be written into notebook source, stdout,
checkpoint files, manifests, downloaded artifacts, or repository history.

## 4. Architecture

### 4.1 Platform model

Kaggle's saved notebook output is the durable boundary. Each stage runs as one
private `Save & Run All` job. A successful job saves its files from
`/kaggle/working`. The next notebook attaches that exact saved output as a
read-only input and independently validates its contents before doing work.

The protocol must not depend on an interactive kernel, an uncommitted notebook
cell, mutable Python variables, or scratch files surviving between sessions.

### 4.2 Stages

The closed stage order is:

| Index | Stage ID | Work | Seed-2404 access |
|---:|---|---|---|
| 0 | `00_authorize` | Bind source, preregistration, environment, capacity evidence, support plan, exact command, and successor authorization | Prohibited |
| 1 | `10_corpus` | Build production corpus through real campaign, simulator, rail, ledger, and TrustVerifier execution | Permitted after verified Stage 00 |
| 2 | `20_features` | Build approved causal feature matrices and training-partition evidence | No new population access |
| 3 | `30_arms` | Train and independently score all four frozen arms | No new population access |
| 4 | `40_label_shuffle` | Execute the fixed label-shuffle retraining control | No new population access |
| 5 | `50_invariance_controls` | Execute identity rename, future causality, equal-time isolation, and forbidden-field mutation | No new population access except frozen control-only lifecycle construction already required by the protocol |
| 6 | `60_single_class_controls` | Execute benign-only workload and fraud-only diagnostic controls | No new population access |
| 7 | `70_metrics` | Compute complete metrics, economics, calibration, campaign bootstrap, and readiness evidence | Prohibited |
| 8 | `80_finalize` | Assemble the final envelope, reconstruct every checkpoint, run the independent verifier, and publish the compact private summary | Prohibited |

The stage partition is frozen before either full-size rehearsal. A stage may not
be merged, split, reordered, or assigned different work after observing rehearsal
or development outcomes. A proven resource defect requires a new SOURCE and
preregistration chronology.

### 4.3 Data flow

Each stage consumes:

- the immutable private source bundle from the approved preregistration commit;
- the immutable dependency wheelhouse and environment declaration;
- the exact frozen protocol/config/catalog files; and
- either no predecessor for Stage 00 or the exact checkpoint manifest and chunks
  from the immediately preceding stage.

Each stage emits only its new checkpoint. It does not copy predecessor payloads
into its output. Stage 80 attaches and verifies all nine outputs, then assembles
the existing complete locked evidence payload plus the checkpoint-chain binding.

## 5. Checkpoint contract

### 5.1 Directory and filenames

Every stage writes to a fresh output root:

```text
/kaggle/working/apar-v5-checkpoint/
  checkpoint.manifest.json
  observational.json
  chunks/
    part-0000.bin
    part-0001.bin
    ...
```

The final stage additionally writes:

```text
/kaggle/working/apar-v5-final/
  candidate.manifest.json
  candidate.manifest.json.chunks/
  judge-summary.json
  verification.json
```

### 5.2 Canonical manifest

`checkpoint.manifest.json` is an immutable, strict-schema document containing:

- schema version and stage protocol version;
- closed run mode and exact stage ID/index;
- successor attempt receipt digest;
- run-binding, preregistration, SOURCE tree, protocol, config, catalog,
  implementation, dependency, and environment digests;
- predecessor stage ID and checkpoint digest, or explicit Stage-00 genesis;
- exact input support IDs and input artifact digests;
- deterministic-core schema, chunk order, sizes, and SHA-256 digests;
- observational document digest;
- output support IDs and bounded artifact counts;
- notebook source digest and recorded Kaggle notebook-version identity;
- start/completion UTC timestamps;
- checkpoint self-digest.

The manifest is written last with exclusive, no-replace semantics. Chunk files
are content-addressed, bounded, individually fsynced, and directory-fsynced where
the Kaggle filesystem supports those POSIX operations. The implementation must
probe and record those capabilities during rehearsal; it must fail closed if the
locked environment is weaker than the rehearsed environment.

### 5.3 Deterministic and observational layers

Deterministic checkpoint evidence excludes only exact, versioned observational
fields. Recursive name-based field omission is prohibited. Observational evidence
contains:

- real wall-clock start and completion times;
- real stage latency;
- sampled process RSS and host memory readings;
- peak RSS, available memory, CPU count/model, filesystem declaration, Python
  version, package-lock digest, and Kaggle image declaration;
- output byte counts and checkpoint publication latency.

The independent verifier validates both layers, recomputes their metrics, and
proves observational fields cannot influence model inputs, probabilities,
actions, thresholds, controls, non-latency metrics, economics, bootstrap, or
readiness.

## 6. Resume and terminal semantics

### 6.1 Completed stages

A stage is completed only when its saved Kaggle output contains a valid manifest,
all exact chunks, a valid observational document, and a self-consistent digest
chain. Once completed:

- it cannot be recomputed, replaced, overwritten, or skipped;
- a later stage must consume exactly its content digest;
- a second valid checkpoint for the same stage and attempt is a terminal conflict;
- changing the Kaggle notebook version without identical checkpoint bytes is a
  terminal conflict.

### 6.2 Incomplete stages

If a Kaggle job terminates without publishing any valid checkpoint manifest, the
current stage is incomplete. The same stage may be run again from the last valid
predecessor. Completed predecessors are not recomputed.

The retry admission is closed:

- the stage ID is inferred from the predecessor and cannot be supplied as an
  arbitrary public CLI value;
- source, config, environment, notebook source, and predecessor digests must be
  identical;
- no later checkpoint may exist;
- no valid or malformed manifest for the current stage may already be published;
- a malformed, partial, tampered, or semantically invalid published checkpoint is
  terminal evidence and does not permit another run.

The final Stage-80 publication remains one-time and non-resumable after its
candidate manifest becomes visible.

### 6.3 Outcome non-disclosure

Notebook stdout may contain only:

- stage ID and status;
- row/artifact counts allowed by the frozen support plan;
- byte sizes;
- deterministic and observational digests; and
- resource telemetry.

It must not print probabilities, actions, labels, family metrics, aggregate
metrics, economic values, controls, gates, readiness, or final status. Private
checkpoint bytes may contain the evidence required by later stages, but no
judge-facing result is produced before Stage 80.

## 7. Workload decomposition

### 7.1 Corpus checkpoint

Stage 10 serializes the immutable `V5Corpus` facts needed by all later stages:
decision rows, execution manifests, lineage, canonical event facts, ledger
postings, reconciliation facts, TrustVerifier inputs/registry/receipts, partition
bindings, support order, and corpus digest.

It must preserve the existing real execution path. Directly constructing fraud
rows, fabricating provenance, or substituting hand-built execution evidence
remains prohibited.

### 7.2 Feature checkpoint

Stage 20 derives features from the corpus checkpoint and emits:

- ordered event IDs and decision-time provenance;
- approved feature names/catalog digest;
- train, calibration, threshold, and development-test matrices;
- labels in a separate non-predictive evidence layer;
- trust-failure ordering;
- training support/evidence records;
- exact matrix and batch digests.

Forbidden fields remain excluded from predictive inputs.

### 7.3 Arm checkpoint

Stage 30 trains and scores the four existing arms over identical support. It
emits the same complete `V5EvaluationResult` evidence currently expected by the
locked payload. Models need not be retained after scoring unless deterministic
replay tests prove their serialized bytes are required. No non-full arm may read
full-sentinel outputs.

### 7.4 Control checkpoints

Stages 40, 50, and 60 refactor control orchestration into independently callable
closed groups without changing any control implementation or criterion. Stage 70
assembles the exact seven-control suite and rejects duplicates, omissions,
unexpected order, or mismatched arm/config/source bindings.

### 7.5 Metrics and finalization

Stage 70 computes the existing complete metrics, ledger-derived economics,
calibration bins, campaign bootstrap intervals, and readiness evidence from the
checkpointed arm/control artifacts.

Stage 80 independently recomputes those claims, reconstructs the execution pool,
and builds the existing deterministic-core plus observational-latency envelope.
The final payload additionally binds the nine-stage chain root and successor
attempt receipt.

## 8. Kaggle environment freeze

### 8.1 Private inputs

Before rehearsal, create private, versioned Kaggle inputs for:

- a canonical Git archive of the exact approved repository commit;
- a Linux x86-64 dependency wheelhouse installed with `--no-index`;
- the approved safe evidence fixture; and
- generated notebook sources and their digests.

Locked notebook stages run with internet disabled. The package environment is
created only from the frozen wheelhouse. Kaggle credentials are not required by
the stage process after inputs are attached.

### 8.2 Model execution

The staged run remains CPU-only. Switching CatBoost or any other learned
component to GPU/TPU execution is a model-semantic change and is prohibited.

### 8.3 Notebook generation

Human-edited notebook logic is prohibited. Repository code deterministically
generates nine minimal notebooks whose cells only:

1. locate the frozen private inputs;
2. verify their digests;
3. install from the wheelhouse;
4. invoke the closed stage entrypoint; and
5. emit the redacted stage receipt.

Direct and generated-notebook invocation paths must execute the same Python stage
entrypoint.

## 9. Capacity validation

The exact nine-stage pipeline must complete twice on Kaggle using:

- the production-size profile and support plan;
- test-safe development seed 404;
- the same notebook sources, dependency bundle, stage partition, and machine
  class intended for the locked run; and
- a separate repeatable `kaggle_capacity_validation` mode which cannot be
  relabeled as locked development.

The following criteria are frozen before either rehearsal:

| Criterion | Required value |
|---|---:|
| Peak process RSS per stage | less than 18 GiB |
| Wall time per stage | less than 6 hours |
| Saved output per stage | less than 10 GB |
| Deterministic stage digests | identical across both rehearsals |
| Checkpoint-chain verification | pass for every stage |
| Reconstructed safe final evidence | independently valid in both rehearsals |

If any criterion fails, Kaggle is rejected for the locked run. The criteria may
not be relaxed after observing rehearsal values. A proven resource implementation
defect may be corrected only through a new SOURCE commit and two new rehearsals.

## 10. Independent verifier

Create a separate offline verifier for the staged artifact. It must not import:

- production corpus generation;
- feature construction;
- model training or prediction;
- control execution;
- production metric, gate, readiness, storage, or stage-runner functions;
- live simulator, rail, ledger, or TrustVerifier calls.

The verifier consumes downloaded checkpoint outputs, the final output, and frozen
source/config/protocol bindings. It independently verifies:

- exact schemas, size limits, content hashes, chunk order, and self-digests;
- genesis authorization and predecessor chain;
- exact stage order and absence of duplicate or alternate checkpoints;
- run mode, seed, profile, support plan, environment, source, and notebook
  bindings;
- command/event/campaign/payment/ledger/trust lineage;
- feature/support ordering and forbidden-field exclusion evidence;
- four-arm identity and ordered support equality;
- all seven controls and criteria;
- aggregate/per-family metrics, calibration, economics, and campaign bootstrap;
- deterministic/observational separation and resource telemetry;
- final candidate, judge summary, readiness, and chain-root bindings.

It fails closed on missing, extra, reordered, substituted, cross-attempt,
cross-environment, malformed, partial, or tampered evidence. It emits concise
machine-readable output and a nonzero exit status on failure.

## 11. Recovery and freeze chronology

The design document commit is governance and precedes the evidence chronology.
Implementation then creates:

1. **RECOVERY** — adds only the prior attempt receipt and a terminal abort record.
   The abort record binds the receipt, rejected preregistration/source, panic UTC,
   watchdog reason, panic-log digest, absent candidate/chunk/summary paths, and
   unchanged historical-result digest.
2. **SOURCE3** — sole child of RECOVERY containing the staged runner, checkpoint
   contracts, independent verifier, deterministic notebook generator, configs,
   and tests. It contains no new preregistration or seed-2404 result.
3. **PREREGISTRATION3** — sole child of SOURCE3 changing only
   `config/defense/defense-v5-kaggle-preregistration.json`. It binds SOURCE3,
   private Kaggle environment/input/notebook digests, both seed-404 rehearsal
   chain roots, capacity measurements, closed stage semantics, seed 2404,
   production support, successor attempt path, exact stage commands, final output
   contract, and independent verifier digest.

The old SOURCE, preregistration, failed receipt, historical result, safe-core
freeze, rejection records, thresholds, seeds, and evidence remain byte-identical.

## 12. Browser and operator workflow

After SOURCE3 is green:

1. use the browser to create private Kaggle source/dependency inputs and the nine
   private rehearsal notebooks;
2. pause for the owner to complete Kaggle login or security confirmation without
   disclosing credentials;
3. run both seed-404 rehearsals stage by stage;
4. download and independently verify both checkpoint chains locally;
5. create and commit PREREGISTRATION3;
6. upload the final preregistration-bound private source input;
7. run the exact pre-execution audit and confirm no successor receipt/result;
8. request explicit authorization for Stage 00;
9. stop before Stage 10 unless Stage 00 is durably saved and verified.

After authorization, later stages are manually launched from the exact verified
predecessor. An incomplete stage may be relaunched; a completed or malformed
published stage may not.

## 13. TDD and adversarial coverage

Implementation begins with focused RED tests for the monolithic runner's inability
to resume. Required GREEN coverage includes:

- exact stage state transitions and capability boundaries;
- seed 2404 unreachable before Stage 10 authorization;
- completed-stage immutability and incomplete-stage continuation;
- malformed published checkpoint terminal behavior;
- missing, extra, reordered, duplicated, altered, and cross-attempt checkpoints;
- wrong predecessor, notebook, source, environment, config, protocol, support, or
  dependency digest;
- observational deletion, mutation, reordering, and deterministic contamination;
- resource-gate failure and refusal to relax gates;
- outcome/log leakage detection;
- control grouping equivalence with the existing seven-control suite;
- final payload equality with the existing complete evidence semantics;
- private notebook metadata and network-disabled locked execution;
- deterministic notebook generation;
- fresh-subprocess replay under different `PYTHONHASHSEED` values;
- static independent-verifier import boundaries;
- legacy runner rejection as staged evidence;
- browser-upload artifact digests matching local generated bytes.

Local gates include focused and full v5 tests, relevant rail/ledger/trust
regressions, Ruff, strict mypy on changed production files, `git diff --check`,
authorship checks, locked-evidence byte hashes, and a clean worktree.

## 14. Acceptance criteria

Implementation is ready for a later locked authorization only when:

- the crashed attempt is preserved as terminal evidence;
- all nine closed stages and their independent verifier are implemented;
- every checkpoint mutation test fails closed;
- no predictive or experiment semantics changed;
- two private production-size seed-404 Kaggle rehearsals satisfy every frozen
  capacity gate and share identical deterministic stage digests;
- both safe chains reconstruct and independently verify offline;
- PREREGISTRATION3 changes exactly one approved path and binds the complete
  environment and chain;
- the exact clean pre-execution audit passes;
- the successor attempt and final candidate paths are absent; and
- seed 2404 has been asserted as a binding only and has not entered population,
  training, scoring, controls, metrics, or finalization.

The system is not authorized to execute seed 2404 merely because this design is
approved or implemented.
