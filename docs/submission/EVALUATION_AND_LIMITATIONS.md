[Submission home](README.md) · [Pitch deck](APAR_COMPETITION_DECK.pdf) · [Model card](MODEL_CARD.md) · [Research journey](RESEARCH_AND_EXPERIMENT_JOURNEY.md)

# Evaluation, results, and limitations

> **Judge file 05 · Metrics and limits.** The four-arm numbers are verified
> recovered diagnostics and non-authoritative; the official chain remains
> incomplete at Stage 70.

---

## Experiment ladder

APAR deliberately treated failed experiments as evidence. Early versions
exposed workload infeasibility, missing calibration/latency metrics, leakage,
generator fingerprinting, and memory pressure. Each failure either produced a
rejection record or changed the protocol before a new run.

The most important v5 ablation is the recovered four-arm comparison:

| Arm | Recall | Precision | F1 | False-decline | Challenge | Review | p95 ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| Rules only | 85.949% | 14.878% | 25.366% | 87.436% | 2.639% | 0.000% | 4.062 |
| Ensemble, no graph | 99.745% | 94.756% | 97.186% | 0.000% | 0.802% | 0.209% | 3.382 |
| **Ensemble with graph** | **99.867%** | **95.876%** | **97.831%** | **0.0037%** | **0.572%** | **0.211%** | **3.544** |
| Full Sentinel hybrid | 99.929% | 16.846% | 28.831% | 87.437% | 2.802% | 0.118% | 19.742 |

All four arms report a 1.0 captured-value fraction on this synthetic support.
That ceiling is a generator/evaluation characteristic and should not be
extrapolated.

## Interpretation

Graph context adds a measurable but modest lift over the already strong
non-graph ensemble: +0.122 percentage points recall, +1.120 points precision,
+0.645 points F1, and -0.230 points challenge rate, for about +0.162 ms p95
latency. This is the empirical reason for selecting `ensemble_with_graph`.

The full hybrid demonstrates why recall alone is an unsafe objective: it obtains
the highest recall while its deterministic routing creates unacceptable benign
friction and much higher latency. Its readiness verdict is `not_ready`.

## Controls completed

- label-shuffle control;
- identity-renaming invariance;
- future-causality append test;
- equal-time isolation;
- feature/label leakage tests;
- single-class benign/fraud controls;
- rail/event/ledger projection tamper tests;
- deterministic bundle and trace hashing;
- independent portable replay.

## Evidence boundary

The recovered results are verified and replayable but explicitly
non-authoritative. They cannot be promoted into official Stage 70 capacity
evidence because the frozen in-memory four-arm run exhausted Kaggle memory and
the later memory-safe execution architecture necessarily changed frozen source
bindings. The official chain therefore stops after Stage 60.

The portable demo and recovered metrics use seed 404. An earlier local locked
development attempt started and irreversibly aborted; it emitted no candidate
manifest, candidate chunks, judge summary, or successful seed-2404 result. The
retained evidence does not prove which internal sub-step was reached, so no
stronger non-execution claim is made.

## What remains outside this competition package

- representative external or real-world validation;
- official Stage 70 and Stage 80 completion under a newly approved chain;
- production drift, fairness, resilience, throughput, and capacity evidence;
- privacy-preserving multi-institution evaluation;
- policy refinement before enabling the full hybrid.

These are deployment gates, not hidden omissions. The competition claim is a
working, deeply instrumented synthetic assurance prototype with a portable
model and honest evidence governance.
