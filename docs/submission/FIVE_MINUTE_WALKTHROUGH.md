# Five-minute APAR walkthrough

Target length: **4:40–4:55**. Record at 1080p, browser zoom 100%, with the console
already running at `http://127.0.0.1:4173/overview`. Keep the cursor still while
speaking and move only when the script calls for it.

## 0:00–0:35 — Open on Overview

**On screen:** APAR Overview, title and threat/evidence cards.

**Say:**

“Generative AI does not create a new payment rail—it changes the attacker’s
speed, personalization, coordination, and autonomy. APAR is an Adaptive Payment
Assurance Range: a pre-production system that turns emerging threats into safe,
replayable campaigns and tests whether payment controls still work. Our key
shift is from scoring isolated transactions to assuring the whole campaign.”

Point briefly to the card, A2A, and agentic rail contexts. Click **Inspect
scenario controls**.

## 0:35–1:10 — Scenario generation

**On screen:** Scenario route and campaign graph.

**Say:**

“This scenario is synthetic, deterministic, and bounded. It models an
AI-personalized authorized-push-payment scam converging on a mule network. The
simulator enforces event-time ordering, payment lifecycles, a fixed query
budget, and a conserved synthetic ledger. We also implement card testing,
synthetic merchant/refund abuse, and agentic intent abuse through real rail
adapters—not hand-built fraud rows.”

Select the three stage cards, then click **Start verified replay**.

## 1:10–2:15 — Live model replay

**On screen:** Replay route. Click **Play both streams**, then pause on a risky
A2A decision.

**Say:**

“The browser calls the real local Stage 30 `ensemble_with_graph` scorer. The
portable bundle contains three calibrated CatBoost members and 46 frozen,
past-only features: velocity, amount deviation, pair history, fan-out, shared
neighbors, two-hop reach, burst motifs, component size, and data-quality flags.

For each event we expose the calibrated probability, frozen thresholds, action,
latency, and reason code. The action policy can approve, challenge, review-hold,
or decline-hold. If the worker is unavailable, the UI switches visibly to a
hash-bound verified fallback; it never silently invents a result.

These 12 cases replay exactly and are curated demonstrations—not production
accuracy estimates.”

Point to the explicit “No payment-to-trace mapping asserted” notice. Click
**Investigation**.

## 2:15–2:55 — Campaign investigation

**On screen:** Investigation route. Select the central mule node.

**Say:**

“The investigator receives one connected case instead of disconnected alerts.
Here, 14 entities and 10 synthetic payment edges reveal fan-in, layering, and
fan-out around the selected mule. Linked payments, amounts, and decisions remain
traceable to the deterministic scenario. We deliberately mark analyst-time
benefit as evidence pending instead of fabricating an efficiency claim.”

Click **Defenses**.

## 2:55–3:55 — Research and arm comparison

**On screen:** Defenses route. Compare `rules_only`, `ensemble_no_graph`, and
`ensemble_with_graph`.

**Say:**

“Our experiments changed the architecture. Rules alone caught fraud but caused
extreme benign friction. A calibrated non-graph ensemble was strong. Adding
causal graph context produced the best balance: in verified recovered
diagnostics it reached 99.87 percent recall, 95.88 percent precision, 97.83
percent F1, a 0.0037 percent false-decline rate, and 3.54 millisecond p95
latency.

The full hybrid actually had slightly higher recall, but failed false-decline
and challenge gates. So we did not call it the champion. This is our core
lesson: in payments, recall without friction and capacity discipline is not a
winning system.”

Point to the non-authoritative qualifier. Click **Assurance**.

## 3:55–4:40 — Trust and governance

**On screen:** Assurance route. Select two artifacts and then the TrustVerifier
panel.

**Say:**

“APAR also treats agentic payments as an authorization problem. Before machine
learning, TrustVerifier checks agent identity, user mandate, scope, merchant and
cart binding, expiry, nonce, and replay. A model cannot compensate for invalid
authority.

Every model, scenario, control, and trace is hash-bound. Leakage and tamper tests
fail closed, rejected experiments remain visible, and no model can promote
itself. The accepted demo model is Stage 30; recovered four-arm metrics are
useful diagnostics but not official Stage 70 or production evidence.”

## 4:40–4:55 — Close

**On screen:** Return to Overview or hold on the human promotion gate.

**Say:**

“APAR’s promise is simple: synthetic evidence before assertion—campaign-aware
defense, verifiable intent, honest failure gates, and a human decision before
deployment. That is how payment security can adapt as quickly as AI-enabled
fraud.”

## Recording checklist

- Run `.venv/bin/python scripts/run_apar_console.py health` before recording.
- Use the real local scorer label; do not record in fallback-only mode.
- Say `ensemble_with_graph`, never `full_sentinel`, for the champion.
- Call the four-arm numbers “verified recovered diagnostics.”
- Do not call the 12 cases population, production, Mastercard, or external data.
- If asked about the locked run: “The official chain is incomplete at Stage 70;
  an earlier local locked attempt aborted without publishing a successful
  result.”
- End before five minutes; leave 5–10 seconds for platform trimming.
