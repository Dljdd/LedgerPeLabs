[Submission home](README.md) · [Pitch deck](APAR_COMPETITION_DECK.pdf) · [Walkthrough](FIVE_MINUTE_WALKTHROUGH.md) · [Model card](MODEL_CARD.md)

# Competition traceability

> **Judge file 03 · Requirement-to-proof map.** Read left to right: competition
> need, APAR response, demonstrable proof, then evidence status.

---

APAR answers the challenge as a working assurance product: identify emerging
GenAI-enabled payment threats, turn them into bounded synthetic campaigns,
replay them through rail-correct lifecycles, compare defenses, and preserve a
human promotion boundary.

| Competition need | APAR response | Demonstrable proof | Status |
|---|---|---|---|
| Identify novel payment attacks | Evidence-backed threat registry separates observed facts from modeled GenAI capability deltas | 20 typed threat cards; source and confidence fields; Overview route | Validated foundation |
| Generate realistic simulations | Stateful campaigns across card, A2A, and agentic rails with event time, lifecycle state, entities, and ledger conservation | Four executable families; deterministic scenario and campaign traces; Scenario and Replay routes | Implemented and tested synthetically |
| Defend accurately | Calibrated three-member CatBoost ensemble over 46 causal temporal, velocity, data-quality, and graph features | Hash-bound Stage 30 `ensemble_with_graph` bundle; exact 12-case replay | Demo-ready |
| Keep customer friction low | Selective actions—approve, challenge, review hold, decline hold—plus capacity gates | Recovered graph-arm diagnostics: 0.0037% false-decline and 0.572% challenge rate | Verified, non-authoritative diagnostic |
| Detect campaigns, not isolated rows | Pair history, fan-in/fan-out, shared-neighbor, two-hop, burst-motif, component, and density signals | Investigation graph and graph/no-graph ablation | Implemented |
| Address agentic commerce | Deterministic TrustVerifier checks identity, mandate, merchant/cart binding, scope, expiry, nonce, and replay before statistical risk | Agentic rail tests and Assurance route | Implemented integrity proof |
| Adapt to evolving attacks | Bounded campaign generator and adversarial search harness with fixed budgets and declared feedback | CampaignGenerator, simulator, adaptive-search evidence, negative control | Implemented research track |
| Make results auditable | Content-addressed artifacts, immutable evidence contracts, rejected-experiment records, fail-closed gates, no self-promotion | Assurance route, release manifest, independent replay and verifier tests | Implemented |
| Present a usable product | Six-route offline console with live scorer, fallback drill, keyboard/mobile coverage, and investigation workflow | Console screenshots, automated tests, five-minute script | Ready for recording |
| Enable practical adoption | Shadow-mode rail adapters, champion/challenger evaluation, human approval, and export boundary | Deployment plan and architecture documentation | Proposed production path |

## What judges can verify in five minutes

1. A sourced threat becomes a bounded synthetic campaign.
2. The campaign graph and payment lifecycle replay deterministically.
3. The real local `ensemble_with_graph` scorer returns calibrated probabilities
   and selective actions.
4. An investigator receives a linked case rather than disconnected alerts.
5. The Assurance route exposes hashes, controls, failed gates, and the explicit
   human promotion boundary.

## Evidence classes

| Evidence class | Permitted claim |
|---|---|
| Accepted Stage 30 portable bundle | The model and thresholds can be loaded and replayed exactly |
| Curated 12-scenario replay | The packaged model reproduces 12 bound predictions and actions |
| Recovered four-arm metrics | Directional architecture comparison on the synthetic development corpus |
| TrustVerifier tests | Deterministic authorization-integrity enforcement |
| Official Stage 70 | Not available; no official capacity/readiness claim |

The product intentionally exposes these distinctions instead of combining them
into a single inflated “accuracy” claim.
