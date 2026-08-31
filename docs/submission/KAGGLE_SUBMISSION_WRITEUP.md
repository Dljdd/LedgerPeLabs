# Kaggle submission entry — APAR

This file is formatted for direct transfer into a Kaggle Project Writeup. Add
the final video URL, upload the media-gallery images, preview every link, and
remove this introductory paragraph before submitting.

## Kaggle fields

**Project title**

```text
APAR — Adaptive Payment Assurance Range
```

**Project subtitle**

```text
Campaign-aware fraud defense and verifiable intent for AI-era payments
```

**Short project description**

```text
APAR turns emerging AI-enabled payment threats into deterministic synthetic
campaigns, replays them across card, account-to-account, and agentic-payment
rails, compares campaign-aware defenses, verifies delegated intent, and
produces auditable evidence for a human promotion decision.
```

**Video link**

```text
[ADD FINAL PUBLIC VIDEO URL]
```

**Project links**

- Working prototype: https://web-six-tau-bxhm7rwrzu.vercel.app/overview
- Public source: https://github.com/Dljdd/LedgerPeLabs
- Pitch deck: https://github.com/Dljdd/LedgerPeLabs/blob/codex/apar-final-submission/docs/submission/APAR_COMPETITION_DECK.pdf

**Suggested media gallery order**

1. `docs/demo/screenshots/apar-console-overview-desktop.png`
2. `docs/demo/screenshots/apar-console-replay-desktop.png`
3. `docs/demo/screenshots/apar-console-investigation-desktop.png`

---

# APAR — Adaptive Payment Assurance Range

## Assure the campaign. Verify the intent. Promote only the evidence.

Generative AI does not create a new payment rail; it changes the attacker's
speed, personalization, coordination, and autonomy. Fraud teams can iterate
faster, coordinate mule networks, probe controls, and act through delegated
software. Defenses that score one transaction at a time are therefore solving
only part of the problem.

APAR is a pre-production payment assurance range. It converts emerging threats
into bounded synthetic campaigns, executes them through rail-correct payment
lifecycles, compares champion and challenger controls, verifies delegated
agentic intent, and preserves an explicit human decision before promotion.

## The problem we chose to solve

Payment institutions need to answer more than “is this transaction risky?”
They also need to know:

- whether several individually plausible payments form a coordinated campaign;
- whether a defense catches fraud without overwhelming customers or operations;
- whether an AI agent actually possesses authority for the exact purchase;
- whether evaluation evidence is causal, reproducible, and free from leakage;
- and whether a control remains safe when attackers adapt.

APAR makes those questions executable before a control is placed in production.

## How APAR works

### 1. Evidence-backed threat modeling

Threat evidence is translated into typed, bounded scenario specifications. We
separate sourced observations from modeled AI capability changes and preserve
confidence, assumptions, and provenance.

### 2. Rail- and ledger-correct campaign simulation

The simulator generates stateful campaigns across card,
account-to-account, and agentic-payment contexts. It maintains event-time
ordering, payment lifecycle state, fixed adversarial budgets, and a conserved
synthetic ledger.

The working implementation covers four campaign families:

- AI-personalized APP scam and mule convergence;
- card-testing CNP bursts;
- synthetic merchant and refund abuse;
- agentic intent abuse.

Fraud evidence is projected from executed rail events and ledger records—not
from hand-authored “fraud rows.”

### 3. Campaign-aware defense

Our selected model is a portable three-member calibrated CatBoost ensemble over
46 frozen, past-only features. It combines transaction velocity, amount
deviation, prior pair behavior, counterparty fan-out, shared neighbors, two-hop
reach, burst motifs, component structure, and data-quality indicators.

Instead of one binary action, APAR supports a selective policy: approve,
challenge, review-hold, or decline-hold. Model selection therefore considers
false declines, challenge volume, review demand, latency, calibration, and
captured value alongside precision and recall.

### 4. Verifiable intent for agentic payments

Agentic commerce introduces an authorization problem that a fraud score cannot
solve. Before statistical risk, APAR's deterministic TrustVerifier checks:

- agent identity and the user's mandate;
- permitted scope and spending constraints;
- merchant and cart binding;
- expiry, nonce, and replay protection.

An invalid authority chain fails before model risk is considered.

### 5. Evidence governance

Models, scenarios, controls, results, and replay traces are content-addressed.
Leakage and tamper tests fail closed, rejected experiments remain visible, and
no model can promote itself. The final output is evidence for a human decision,
not an autonomous deployment.

## What our experiments changed

The final architecture was shaped by failures rather than a single successful
training run.

1. **Rules were too blunt.** They detected fraud but produced unacceptable
   benign friction.
2. **A non-graph ensemble was strong.** Calibrated tabular modeling created a
   practical real-time baseline.
3. **Perfect scores exposed leakage.** Future-append tests showed that an early
   graph implementation allowed future component membership to influence past
   decisions. We removed the path, added causal and equal-time isolation tests,
   and retained the rejected result.
4. **Real simulation mattered.** We replaced constructed fraud rows with
   CampaignGenerator → SimulationEngine → rail adapter → events → ledger →
   validated evidence.
5. **Causal graph context provided the best balance.** Graph summaries improved
   the precision/recall/friction frontier without requiring a heavyweight GNN
   in the real-time path.

The selected `ensemble_with_graph` model produced these recovered results on
the synthetic development corpus:

| Metric | Verified synthetic diagnostic |
|---|---:|
| Recall | 99.87% |
| Precision | 95.88% |
| F1 | 97.83% |
| False-decline rate | 0.0037% |
| Challenge rate | 0.572% |
| p95 scoring latency | 3.54 ms |

These measurements are deliberately described as **verified synthetic
diagnostics**. They support the architecture comparison, but they are not
production, real-cardholder, Mastercard-data, or external-validation claims.

## What makes APAR different

The novelty is not one model or one graph feature. It is the integration of
five disciplines that payment teams usually operate separately:

1. evidence-backed threat modeling;
2. rail- and ledger-correct adversarial campaign simulation;
3. causal temporal and graph defense features;
4. deterministic authorization integrity for agentic payments;
5. fail-closed, content-addressed, human-governed promotion evidence.

This allows APAR to show not only that a model works on a demonstration, but
also how the institution can discover when it does not.

## Working prototype

The public six-route console is available here:

**https://web-six-tau-bxhm7rwrzu.vercel.app/overview**

It lets judges inspect the threat framing, bounded scenario, synchronized
replay, connected investigation graph, defense comparison, and assurance
evidence. The hosted application uses a committed hash-bound trace so that it
remains accessible without a Python service.

The public source repository contains the real portable scorer. Running the
local console loads the packaged `ensemble_with_graph` model, scores the 12
curated cases, verifies the returned probabilities and actions, and exposes the
result through the same interface:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -e '.[dev]'
npm ci --prefix web
.venv/bin/python scripts/run_apar_console.py start
```

## Reproducibility and responsible claims

The repository includes the model members, calibrators, ordered feature
contract, thresholds, scenario records, payload hashes, deterministic fallback,
model and data cards, experiment narrative, clean-room release tooling, and
third-party inventory.

The 12 visible cases are curated replay demonstrations, not a population
estimate. APAR uses no real cardholder data and makes no production-readiness or
external-validation claim. Where evidence is absent—for example measured
analyst-time savings—the interface says **evidence pending**.

That boundary is intentional. Payment assurance should reward systems that can
explain both what their evidence proves and what it does not.

## Path to adoption

APAR is designed to move through controlled stages: offline replay, shadow-mode
feature generation, assisted operations with human-controlled friction,
limited challenger traffic with rollback, and finally governed scale with
continuous adversarial regression.

The result is a practical assurance layer for payment-risk, fraud-operations,
threat-intelligence, and agentic-commerce teams—a way to adapt defenses as
quickly as AI-enabled fraud while keeping authority and promotion human
controlled.

## Links

- **Working prototype:** https://web-six-tau-bxhm7rwrzu.vercel.app/overview
- **Public repository:** https://github.com/Dljdd/LedgerPeLabs
- **Pitch deck:** https://github.com/Dljdd/LedgerPeLabs/blob/codex/apar-final-submission/docs/submission/APAR_COMPETITION_DECK.pdf
- **Video:** https://www.youtube.com/watch?v=A_4Pe_A7iMg

## Team LedgerPeLabs

- Dylan Moraes
- Anuj Sharma
- Dhananjay Joshi
- Rahul Biradar
