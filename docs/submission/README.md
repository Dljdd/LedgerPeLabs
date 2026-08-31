# APAR competition submission pack

This directory is the judge-facing handoff for the Mastercard Innovation
Challenge 2026. Start with the deck, then use the five-minute walkthrough to
demonstrate the console.

## Primary deliverables

- `APAR_COMPETITION_DECK.pptx` and `APAR_COMPETITION_DECK.pdf`: concise pitch.
- [Five-minute walkthrough](FIVE_MINUTE_WALKTHROUGH.md): recording script and
  shot list.
- [Research and experiment journey](RESEARCH_AND_EXPERIMENT_JOURNEY.md): why
  the architecture evolved and why the graph ensemble was selected.
- [Competition traceability](COMPETITION_TRACEABILITY.md): requirement-to-proof
  map.
- [Model card](MODEL_CARD.md), [data and simulation card](DATA_AND_SIMULATION_CARD.md),
  and [evaluation and limitations](EVALUATION_AND_LIMITATIONS.md): evidence
  boundaries.
- [Commercial and deployment plan](COMMERCIAL_AND_DEPLOYMENT_PLAN.md): practical
  path from prototype to network-scale assurance.
- [Release checklist](RELEASE_CHECKLIST.md): clean-machine instructions.

## One-command demo

```bash
.venv/bin/python scripts/run_apar_console.py start
```

Open `http://127.0.0.1:4173/overview`. The console calls the local portable
model and transparently switches to a hash-bound verified fallback if the model
worker is unavailable.

## Claim boundary

APAR is a synthetic, pre-production assurance prototype. The competition model
is the accepted Stage 30 `ensemble_with_graph` bundle. The 12-scenario replay is
a deterministic demonstration, not a population estimate. Four-arm recovered
metrics are verified diagnostics but non-authoritative; the official evaluation
chain remains incomplete at Stage 70. No real cardholder data, production
deployment, or external validation is claimed.
