# Implementation-plan traceability

This matrix records where each approved specification area is implemented and how completion is proven.

| Approved specification area | Implementation plan and task | Verification evidence |
|---|---|---|
| Executive decision and product thesis | Foundation Tasks 3 and 6; Prototype Tasks 3 through 6 | Evidence-backed threat compiles into a bounded campaign and reaches a human-owned assurance report |
| Competition pillars and feasibility | Plan index G0 through G5 | One continuous Identify, Generate, Defend, and Assure golden path |
| Goals, non-goals, and user jobs | Foundation global constraints; Prototype Tasks 2 through 5 | API, safety, judge timing, investigator, and product-owner tests |
| System context and logical architecture | All four target file maps | Import boundaries, API contracts, artifact references, and end-to-end gates |
| Threat identification breadth | Foundation Task 6 | At least 20 schema-valid, sourced threat cards across cards, A2A, merchant, identity, mule, and agentic families |
| Four executable scenarios | Simulator Task 5 | Multi-seed campaign tests for APP/mule, card testing/CNP, merchant/refund, and agentic intent abuse |
| Rail-specific scenario bundles | Foundation Tasks 2 and 3; Simulator Tasks 2 through 4 | Strict schemas and legal transition tests for card, A2A, and agentic rails |
| Payment lifecycle through recovery | Simulator Tasks 1 through 4 | Property tests for authorization, clearing, settlement, reporting, dispute, chargeback, return, freeze, and recovery |
| Conservation and causal timing | Simulator Tasks 1 through 3; Defense Task 1 | Double-entry ledger, source provenance, strict past-only and future-append tests |
| Benign realism and entity motifs | Simulator Task 5 | Class-rate, value-total, motif, benign-shift, and five-seed fidelity tests |
| Attacker observation boundary | Simulator Task 6 | Feedback schema contains only action, reason family, and realized value |
| Search space and objective | Simulator Task 6 | Bounded parameter mutation and visible objective tests |
| Fixed, random, adaptive, and LLM ablation | Simulator Task 6 | Matched query, proposal, wall-time, seed, and feedback budgets |
| Hidden validity | Simulator Task 7 | Restricted reasons and visible boolean-only result |
| Separately implemented hidden generator | Simulator Task 7; Defense Task 6 | Static import scan prevents main-generator and defender imports; frozen hidden-run artifacts |
| Layered defense order | Simulator Task 4; Defense Tasks 1 through 3 | Integrity precedence, timeout fallback, stable reasons, calibrated budget tests |
| Rules and strong GBDT baselines | Defense Tasks 2 and 3; frozen Task 15 evidence | Deterministic CatBoost reloads against the frozen 48-feature matrix; exhaustive matched-threshold selection records a truthful workload-infeasible `no_promotion` result |
| Online and asynchronous processing | Defense Tasks 1, 4, and 7 | Feature checkpoint/replay and separate synchronous versus graph/case latency reports |
| Agentic trust plane | Simulator Task 4 | Identity, signature, mandate, amount, currency, payee, cart, expiry, nonce, and receipt-chain rejection tests |
| Event and feature data contracts | Foundation Task 2; Defense Task 1 | Versioned event envelope, distinct timestamps, feature catalog, source-event IDs, and forbidden-source tests |
| Scoring and model output contract | Foundation Task 2; Defense Tasks 2 and 3 | Calibrated score, action, owner, reason, evidence, fallback, and latency contract tests |
| Temporal and entity-isolated evaluation | Defense Task 5 | Whole-campaign partitions and returning/cold-entity cohort tests |
| Operational metrics and promotion gates | Defense Tasks 5 through 7; Task 15 | Workload reconstruction is load-bearing: six mandatory cases over 336 rows exceed the preregistered 1% cap, vetoing freeze/hidden release without retuning |
| Safety, privacy, and governance | Foundation Task 4; Defense Task 7; Prototype Task 8 | Immutable artifacts, restricted exports, privacy scan, SBOM, license and secret checks |
| Six-view product experience | Prototype Tasks 1 through 5 | Component, accessibility, and interaction tests for all six routes |
| Five-minute judging narrative | Prototype Task 6 | Offline Playwright golden path and deterministic fallback recording |
| Reliability and degraded behavior | Simulator Tasks 1 and 7; Defense Tasks 1 and 2; Prototype Task 6 | Idempotency, watermark behavior, checkpoint/replay, model fallback, fixture, and recording checks |
| Implementation boundaries | Four target file maps | Focused package ownership and explicit interfaces in every task |
| Acceptance criteria | Plan index G0 through G5 | Each gate has a command, expected output, and independently reviewable commit |
| Key risks and responses | Global constraints and plan completion gates | Scope freeze, hidden evaluation, operational budgets, rail contracts, deterministic fixtures, and sanitized packaging |
| Sources and evidence policy | Foundation Task 6; Prototype Tasks 3 and 8 | Direct URLs, fact/inference separation, provenance rendering, and URL allowlist scan |
| Approval and change control | Defense Task 7; Prototype Task 5 | Named human decision, immutable report, rollback artifact, and failed-gate veto |

## Interface consistency decisions

- `PaymentEvent` is the shared event type from foundation through evaluation.
- `ScenarioBundle` is compiled once, frozen by digest, and consumed by the run orchestrator.
- `RunManifest` is the sole bridge from simulation into feature building and evaluation.
- `FeatureVector` always includes decision time, maximum source time, and source event IDs.
- `Decision` always includes score, action, ordered reason codes, model version, evidence references, fallback state, and latency.
- `EvaluationBundle` contains both development and hidden-run results but never exposes hidden parameters to defender code.
- `AssuranceReport` references artifacts by digest and cannot be mutated by promotion.
- The web client consumes versioned API models and never reads `.apar/` directly.
