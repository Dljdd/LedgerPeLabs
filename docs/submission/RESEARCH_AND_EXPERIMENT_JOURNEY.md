[Submission home](README.md) · [Pitch deck](APAR_COMPETITION_DECK.pdf) · [Evaluation](EVALUATION_AND_LIMITATIONS.md) · [Deployment plan](COMMERCIAL_AND_DEPLOYMENT_PLAN.md)

# Research and experiment journey

> **Judge file 07 · Design rationale.** The architecture is the result of
> rejected capacity, completeness, leakage, fidelity, and policy outcomes—not a
> post-hoc model story.

---

## The starting question

The challenge is not simply “can we classify a suspicious transaction?” GenAI
changes the attacker’s operating capability: it can personalize at scale,
iterate quickly, coordinate entities, and act through delegated software. The
defensive problem therefore spans campaign behavior, payment lifecycle,
customer friction, authorization integrity, and evidence governance.

That led to APAR’s central thesis:

> Detect the payment campaign, verify delegated intent, and continuously test
> the control system before claiming readiness.

## What the research suggested

### 1. Networks reveal coordination that rows hide

BIS Project Hertha found that payment-system analytics can supplement bank
controls by identifying complex coordinated criminal activity. Temporal Graph
Networks formalize dynamic graphs as sequences of timed events. We therefore
implemented cheap, interpretable past-only graph summaries—fan-in/fan-out,
shared neighbors, two-hop reach, burst motifs, component size, and edge
density—rather than requiring a heavy end-to-end GNN for the competition demo.

Sources: [BIS Project Hertha](https://www.bis.org/publications/project-hertha-identifying-financial-crime-patterns-real-time-retail-payment-systems),
[Temporal Graph Networks](https://arxiv.org/abs/2006.10637).

### 2. Strong tabular ensembles are a pragmatic real-time baseline

CatBoost’s ordered boosting was designed to reduce prediction shift and target
leakage in categorical settings. A three-seed ensemble offered a strong,
portable CPU model while keeping enough diversity for calibration. We froze the
feature order, model members, calibrators, thresholds, and hashes into a small
offline bundle.

Source: [CatBoost: unbiased boosting with categorical features](https://arxiv.org/abs/1706.09516).

### 3. A risk score is not an operational policy

Selective classification research treats abstention as a risk/coverage trade.
Payments similarly need multiple interventions: approve, challenge, hold for
review, or decline/hold. APAR evaluates false declines, challenge rate, review
rate, latency, and captured value alongside predictive metrics.

Source: [Selective Classification for Deep Neural Networks](https://arxiv.org/abs/1705.08500).

### 4. Agentic payments need deterministic authority checks

Mastercard describes verifiable intent, agent identity, permission, and secure
credentials as foundations for agentic commerce. NIST frames agent identity,
least privilege, delegation, auditable intent, non-repudiation, and prompt
injection as distinct authorization problems. We therefore placed a
TrustVerifier before statistical risk, binding identity, mandate, scope,
merchant, cart, expiry, and nonce/replay.

Sources: [Mastercard’s vision for trusted agentic commerce](https://www.mastercard.com/us/en/news-and-trends/stories/2026/mastercard-agentic-commerce-vision.html),
[NIST agent identity and authorization](https://www.nccoe.nist.gov/sites/default/files/2026-02/accelerating-the-adoption-of-software-and-ai-agent-identity-and-authorization-concept-paper.pdf),
[OpenID AuthZEN Authorization API](https://openid.net/wg/authzen/specifications/).

## What the experiments taught us

### V1: accuracy without capacity is not a winner

The first frozen defense produced useful modeling evidence but could not meet
the preregistered one-percent workload budget. Six mandatory review cases over
336 threshold rows implied a 1.7857% minimum. We did not relax the budget; the
outcome was `no_promotion`.

### V3–V4: fail-closed metrics matter

Incomplete scaffolding consumed an attempt in v3. V4 executed real frozen
models but calibration and time-to-alert were not computed, so gates correctly
failed. This stopped us from presenting partial evidence as readiness.

### V5 first pass: perfect synthetic scores exposed leakage

Feature enrichment initially produced perfect separation. Instead of treating
that as success, causal tests showed future graph leakage: component membership
could change after future events were inserted. We removed the path, added
future-append and equal-time isolation tests, and retained rejection records.
The corrected baseline fell to a plausible ROC-AUC near 0.82 before further
integration.

### V5 integration: evidence had to come from actual rails

We replaced hand-built fraud rows with real `CampaignGenerator` →
`SimulationEngine` → rail adapter → payment events → ledger → validated row
projection. We wired TrustVerifier failures into decisions and added tamper
tests for source IDs, event ordering, lifecycle states, ledger postings, opening
balances, and agentic verdicts.

### V5 comparison: graph context won; full routing did not

The recovered four-arm experiment showed:

- rules alone were too blunt;
- the non-graph ensemble was strong;
- causal graph summaries improved the precision/recall/friction frontier;
- the full deterministic hybrid maximized recall but catastrophically over-routed
  legitimate events and increased latency.

This is why the competition champion is `ensemble_with_graph`, not
`full_sentinel`.

### Packaging: reproducibility became part of the product

The selected Stage 30 model was exported as three CatBoost members, three
calibrators, 46 ordered features, thresholds, scenario records, and hashes. The
release builds deterministically, scans secrets/PII patterns, verifies every
payload, installs into a fresh environment, replays all 12 scenarios, and can
fall back to a separately hash-bound trace.

## Novelty that remains defensible

APAR is not novel because it uses a graph feature or an ensemble in isolation.
It stands out because it connects five normally separate disciplines:

1. evidence-backed threat modeling;
2. rail- and ledger-correct adversarial campaign simulation;
3. causal temporal and graph defense features;
4. deterministic authorization integrity for agentic payments;
5. fail-closed, content-addressed human promotion governance.

The result is a competition demo that shows not only a model that works, but a
credible process for discovering when it does not.
