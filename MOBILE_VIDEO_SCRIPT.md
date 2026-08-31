# APAR five-minute video script

Open this file on your phone in portrait mode. Turn on Do Not Disturb, increase
the text size, and prevent the screen from locking during the recording.

Text marked **DO** is a recording action. Read only the paragraphs marked
**SAY**.

## Before you press Record

**DO**

- Put the browser at 100% zoom.
- Open the public Overview page in the first tab:
  `https://web-six-tau-bxhm7rwrzu.vercel.app/overview`
- Open the local Overview page in the second tab:
  `http://127.0.0.1:4173/overview`
- Confirm the public tab says **Verified fallback · offline**.
- Confirm the local tab says **Local scorer · verified**.
- Return both tabs to Overview.

---

## 0:00–0:18 — Public Overview

**SCREEN**

Public Vercel URL — Overview.

**DO**

Keep the address bar and **Verified fallback · offline** label visible. Point
briefly to the label. Do not start the replay on this tab.

**SAY**

“This is the public APAR judge copy on Vercel. It is a static browser build
that demonstrates accessibility and replays a committed, hash-bound verified
trace. It does not run the Python scorer.”

---

## 0:18–0:32 — Switch to the local model

**SCREEN**

Local URL — Overview.

**DO**

Switch to the local browser tab. Briefly show `127.0.0.1` and point to
**Local scorer · verified**.

**SAY**

“I am now switching to the local console. This is where the packaged Python
`ensemble_with_graph` scorer runs. The green runtime label confirms that its
returned trace has been replay-verified.”

---

## 0:32–1:05 — Screen 1: Overview

**DO**

Point to **Synthetic only**, the three payment contexts, and the evidence
boundary. Select event **04** in the curated decision footprint. Then click
**Inspect scenario controls**.

**SAY**

“APAR is a pre-production assurance layer for adaptive payment risk. It turns
an emerging threat into a bounded synthetic campaign, replays that campaign
through payment controls, and preserves the evidence needed for a human
promotion decision.”

“Our selected path is an AI-personalized authorized-push-payment scam that
converges on a mule network. The goal is to detect the linked campaign, not
only isolated transactions.”

“Source facts remain separate from APAR’s modeling inference. The evidence
supports faster personalization and message iteration, but we make no claim of
autonomous settlement access.”

---

## 1:05–1:38 — Screen 2: Scenario

**DO**

Point to the synthetic label, campaign envelope, conserved ledger, and
fan-in-to-cash-out motif. Then click **Start verified replay**.

**SAY**

“This campaign is synthetic, deterministic, read-only, and ledger-conserved.
No real people, accounts, credentials, or payment instructions are used.”

“The campaign is bounded by declared timing, query, ordering, and value
controls. Its fan-in, layering, fan-out, and cash-out stages create a
repeatable campaign-level test.”

“The twelve portable cases are curated replay demonstrations bound to the
packaged checkpoint. They are not population, prevalence, or production
performance estimates.”

---

## 1:38–2:42 — Screen 3: Replay

**DO**

Point first to **LIVE LOCAL SCORER** and `ensemble_with_graph`.

Click **Play both streams**. Allow three or four steps, then click **Pause both
streams**.

Point to the active payment edge, calibrated probability, action thresholds,
final action, reason code, and feature count.

Select portable event **08** in the event ledger without changing the campaign
selector. Point to the separate ground-truth panel. Click **Reset both**, then
click **Investigation** in the left navigation.

**SAY**

“The local worker loads the packaged `ensemble_with_graph` model and scores the
twelve curated cases. Before rendering a result, the console verifies the
event identifiers, calibrated probabilities, actions, and model-bundle
binding.”

“The orange path shows value progressing through the synthetic campaign. The
decision panel shows the independently selected portable prediction, its
calibrated probability, bound thresholds, final action, reason code, and
feature count.”

“One presentation control advances the campaign graph and the portable
decision trace in their own repository order. This is a synchronized
walkthrough, not a claimed row-level join between the two evidence streams.”

“Post-event truth remains in a separate examination panel and was not provided
to the model as an input. The displayed local latency is also not a production
latency estimate.”

---

## 2:42–3:22 — Screen 4: Investigation

**DO**

Select the central mule node. Point to the linked payments, attempted-value
progression, case grouping, and **Evidence pending** analyst-time field. Then
click **Defenses**.

**SAY**

“The investigator receives the connected campaign rather than twelve
disconnected alerts. This deterministic synthetic graph contains fourteen
entities, ten payment edges, and a conserved five-hundred-dollar
attempted-value progression.”

“Selecting the central mule exposes its genuine linked payments and the shared
campaign case. Where analyst-time evidence does not exist, the console says
Evidence pending instead of inventing a productivity claim.”

---

## 3:22–4:02 — Screen 5: Defenses

**DO**

Keep the default focus on `ensemble_with_graph`. Point to the champion badge
and the non-authoritative qualifier. Do not hover across the alternative arm
cards. Then click **Assurance**.

**SAY**

“The accepted portable competition model is `ensemble_with_graph`: calibrated
member voting combined with bound graph features.”

“The recovered comparison shows zero-point-nine-seven-eight F1, a
zero-point-zero-zero-three-seven percent false-decline rate, and a
zero-point-five-seven-two percent challenge rate for this arm.”

“These numbers are verified synthetic diagnostics only. They are
non-authoritative, they are not accepted capacity evidence, and they are not
production estimates.”

---

## 4:02–4:48 — Screen 6: Assurance

**DO**

Select the first two artifact rows so their full hashes appear. Point to the
human promotion boundary and the separate authorization-integrity checks.

**SAY**

“Every demonstrated decision remains bound to inspectable artifacts, exact
replay checks, and an explicit human promotion boundary. No model can approve
itself.”

“The separate authorization checks verify identity, mandate, scope, cart
binding, expiry, nonce, and replay rejection. This is an authorization-
integrity proof, not a claim about fraud-model performance.”

“The result is designed to be inspectable: the interface exposes hashes,
runtime posture, evidence qualifiers, and the distinction between model
performance and deterministic integrity controls.”

---

## 4:48–5:00 — Closing line

**SCREEN**

Remain on Assurance. Stop moving the pointer.

**SAY**

“APAR demonstrates evidence before assertion: a working synthetic assurance
range, the portable `ensemble_with_graph` champion, honest diagnostics, and no
claim of production deployment, Mastercard data, or external validation.”

“Thank you.”

Hold the final frame silently for two seconds before stopping the recording.

---

## Emergency fallback wording

Use this only if the local page does not say **Local scorer verified**.

**SAY**

“The local worker is unavailable, so the console has switched visibly to its
committed hash-bound verified trace. I am demonstrating deterministic replay;
I am not claiming that the Python scorer ran in this take.”

Continue the same six-screen route, but do not use the words **live local
scorer**.

## Final claim checklist

Before recording, confirm that your script includes all of these points:

- APAR is a synthetic, pre-production assurance layer.
- The public Vercel build is an accessible verified fallback.
- The local console runs the packaged `ensemble_with_graph` scorer.
- The twelve cases are curated replay demonstrations.
- Campaign and portable streams are synchronized for presentation but are not
  asserted to be row-level mapped.
- Recovered numbers are verified synthetic diagnostics and non-authoritative.
- Authorization integrity is separate from model performance.
- Human approval remains required.
- There is no production, real-Mastercard-data, prevalence, or external-
  validation claim.
