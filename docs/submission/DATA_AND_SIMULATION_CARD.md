# Data and simulation card

## Purpose

APAR uses synthetic data to test whether payment controls remain effective when
GenAI increases campaign speed, personalization, coordination, or delegated
action. It does not attempt to infer whether a message was written by AI.

## Implemented scenario families

| Family | Rail | Modeled behavior | Key evidence |
|---|---|---|---|
| AI-personalized APP scam and mule | A2A | Victim-authorized fan-in, layering, fan-out, and cash-out | Transfer lifecycle and campaign graph |
| Adaptive card testing / CNP | Card | Pacing, amount, retry, merchant, and device variation | Authorization and decline events |
| Synthetic merchant / refund | Card | Merchant, payment, return/refund, and lifecycle manipulation | Ledger-backed lifecycle |
| Agentic intent abuse | Agentic | Identity, mandate, scope, cart, merchant, nonce, expiry, and replay failures | TrustVerifier plus agentic rail |

Campaigns execute through `CampaignGenerator`, `SimulationEngine`, a real rail
adapter, emitted payment events, and a conserved synthetic ledger before they
are projected into model rows. Unbacked campaign rows and inconsistent
event/ledger evidence are rejected.

## Fidelity invariants

- `decision_at < now` for historical features; no future or self-event access.
- Equal-time rows do not update one another.
- Explicit authorization, transfer, settlement, return/refund, and decline
  lifecycle states.
- Opening balances plus ledger postings reconcile.
- Entity, event, campaign, source, and rail identifiers remain bound through
  projection.
- Scenario generation is deterministic for a declared seed.
- No network access, live targets, real PANs, or personal data.

## Partitions and evaluation protocol

The v5 corpus contract separates training, calibration, threshold selection,
development test, operational controls, and locked development evidence. The
competition package ships only the accepted portable Stage 30 model, curated
replay scenarios, and clearly labeled recovered diagnostics. It does not expose
or claim an accepted official Stage 70 result.

## Known simulation gaps

Synthetic behavior cannot reproduce all production distributions, cross-bank
visibility, delayed labels, human response, merchant diversity, identity
resolution errors, or adaptive feedback loops. The APP graph shown in the UI is
a deterministic illustration; it is not linked row-for-row to the portable
model’s 12-scenario trace. The UI states this explicitly.

## Safe extension path

Real-world validation should begin with privacy-preserving, retrospective,
institution-authorized data; shadow-only decisions; partitioned event-time
replay; per-segment and per-rail error analysis; and explicit rollback. The
synthetic range remains valuable as a repeatable regression and adversarial
testing layer after production evidence becomes available.
