# APAR final competition recording runbook

This is the canonical recording plan for the APAR competition walkthrough.
Target a five-minute, 1920×1080 video at 100% browser zoom. The public site is
the accessibility proof and verified static replay; the local console is the
only version that runs the packaged Python scorer.

## Truth boundary to keep on screen and in the narration

| Topic | Exact boundary |
|---|---|
| Public URL | `https://web-six-tau-bxhm7rwrzu.vercel.app/overview` is a static Vercel build. It uses the browser's hash-bound verified fallback and does not run Python. |
| Local URL | `http://127.0.0.1:4173/overview` calls the packaged Python scorer at `/api/score`. Use it for the real-model replay only when the UI says **Local scorer verified**. |
| Champion | The accepted portable competition model is `ensemble_with_graph`. |
| Recovered numbers | Verified synthetic diagnostics, non-authoritative, not accepted capacity evidence, and not production estimates. |
| Twelve cases | Curated replay demonstrations bound to the packaged checkpoint; they are not a population or prevalence estimate. |
| Data and validation | No real Mastercard or cardholder data, no production-deployment claim, and no external-validation claim. |
| Integrity proof | The authorization checks are a separate deterministic integrity proof, not evidence of fraud-model performance. |

Never let the narration outrun the visible runtime label. If the page says
**Verified fallback · offline**, describe a committed verified trace. Only say
the Python model ran when the local page says **Local scorer verified**.

## Deployment verification — 31 August 2026

- Vercel scope/project: `dljdds-projects/web` (the existing project was reused).
- Git source: `Dljdd/mastercard-innovation-challenge-2026`.
- Project root and build: `web/`; Node 24; `npm ci`; `npm run build`; output
  directory `dist`.
- Stable production URL:
  `https://web-six-tau-bxhm7rwrzu.vercel.app/overview`.
- Verified deployment URL:
  `https://web-rn3jntvu1-dljdds-projects.vercel.app`.
- Hosted scorer behavior: `POST /api/score` returns HTTP 405. The client then
  validates and renders `web/public/data/verified-trace.json`; no Python runtime
  is present in the Vercel build.
- Direct refresh succeeded on `/overview`, `/scenario`, `/replay`,
  `/investigation`, `/defenses`, and `/assurance`.
- All six routes rendered at 1920×1080 without horizontal overflow or browser
  console warnings. Replay visibly reported **HASH-BOUND VERIFIED FALLBACK**.
- Local verification completed with 8 Python prototype tests, 21 browser unit
  tests, 12 Playwright desktop/mobile tests, TypeScript checks, ESLint, and the
  production Vite build all passing.

## One-time machine preparation

Work from the final-submission checkout:

```bash
cd '/Users/dylanmoraes/.codex/worktrees/03b9/MasterCard Challenge'
```

If `.venv` or `web/node_modules` is missing, install once before recording:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -e '.[dev]'
npm ci --prefix web
```

Then run the focused preflight:

```bash
.venv/bin/python scripts/run_apar_console.py health
npm run build --prefix web
```

Do not continue unless health reports all of the following:

- `status` is `ready`;
- `portable_arm` is `ensemble_with_graph`;
- `event_count` is `12`;
- `fallback_replay_verified` is `true`;
- `recovered_authoritative` and `accepted_capacity_evidence` are `false`.

## Start the recording build

Reset only the generated local trace, then launch from the repository root:

```bash
.venv/bin/python scripts/run_apar_console.py reset
.venv/bin/python scripts/run_apar_console.py start
```

Leave that Terminal tab running. In a second clean Terminal tab, confirm the
served evidence and local scorer endpoint:

```bash
curl --fail --silent http://127.0.0.1:4173/api/health
curl --fail --silent --request POST http://127.0.0.1:4173/api/score
```

The second command must return a replay-verified trace for
`ensemble_with_graph`. Load `http://127.0.0.1:4173/overview` once before the
rehearsal and confirm the sidebar says **Local scorer · verified**. Hide
Terminal before recording; never show shell history, environment files,
deployment dashboards, tokens, or repository metadata.

## Prepare the Mac and browser

1. Connect power and use a quiet room. Disconnect unnecessary external
   displays so the pointer cannot leave the recorded screen.
2. Open **System Settings → Displays** and select a 1920×1080 mode when the
   display offers one. On a Retina-only display, record at native resolution
   and export the final copy at 1080p.
3. Open **Control Center → Focus → Do Not Disturb** and choose **For 1 hour**.
   Quit Mail, Messages, Slack, Calendar, and any app that can show a banner.
4. Hide the Dock with **Option–Command–D**. Use a neutral desktop without
   personal filenames, widgets, or account information.
5. Use one clean browser window. Close unrelated tabs and DevTools, hide the
   bookmarks bar with **Shift–Command–B**, and reset page zoom with
   **Command–0**. Keep the window maximized at a 16:9 size.
6. Prepare exactly two tabs in this order:
   - `https://web-six-tau-bxhm7rwrzu.vercel.app/overview` — title it mentally
     as the **public fallback** tab.
   - `http://127.0.0.1:4173/overview` — the **local scorer** tab.
7. Reload both tabs immediately before recording. Confirm the hosted tab says
   **Verified fallback · offline** and the local tab says
   **Local scorer · verified**.
8. Rehearse the complete path once. Reset the Replay page with **Reset both**,
   return both tabs to `/overview`, and place the pointer over empty space.

## Configure macOS Screenshot / Screen Recording

1. Press **Shift–Command–5**.
2. Choose **Record Entire Screen** for a 1920×1080 display. Otherwise choose
   **Record Selected Portion** and frame only the maximized browser.
3. Open **Options**:
   - Save to a dedicated empty folder such as `Desktop/APAR Recording`.
   - Choose the tested external microphone, or **MacBook Microphone**.
   - Select a 5-second timer if available.
   - Leave pointer-click highlighting off unless the competition explicitly
     requests it.
4. Record a 10-second microphone test first. Play it back at normal volume;
   speech should be clear without clipping, room echo, or fan noise.
5. Start the real recording. Hold still for two seconds before speaking. End
   with the macOS stop button in the menu bar or **Control–Command–Escape**,
   then leave two clean seconds of silence for trimming.

## Five-minute click-by-click choreography and spoken script

### 0:00–0:18 — Public accessibility proof

Show the hosted Overview tab with the address bar and fallback label visible.

Say:

> “This is the public APAR judge copy on Vercel. It is a static browser build:
> it verifies accessibility and replays a committed, hash-bound trace. It does
> not run the Python scorer.”

Point once to **Verified fallback · offline**. Do not start Replay here.

### 0:18–0:30 — Explicit transition to the real local scorer

Switch to the prepared local tab. Briefly keep `127.0.0.1` and the local
runtime label visible.

Say:

> “I am now switching to the local console. This is where the packaged Python
> `ensemble_with_graph` scorer runs, and the green label confirms the returned
> trace was replay-verified.”

This cut is mandatory: it keeps public accessibility and local execution as
two distinct claims.

### 0:30–1:02 — Overview

Say:

> “APAR is a pre-production assurance layer for adaptive payment risk. This
> selected synthetic APP-and-mule campaign tests whether controls detect the
> linked campaign, not only isolated transactions.”

Point to **Synthetic only**, the three payment contexts, and the evidence
boundary. Select event **04** in the curated decision footprint. Click
**Inspect scenario controls**.

### 1:02–1:34 — Scenario

Say:

> “The campaign is synthetic, deterministic, read-only, and ledger-conserved.
> The twelve portable cases are curated replay demonstrations bound to the
> packaged checkpoint—not estimates of prevalence or live performance.”

Point to the campaign envelope and motif. Do not autoplay the stage cards.
Click **Start verified replay**.

### 1:34–2:38 — Replay

First point to **LIVE LOCAL SCORER** and `ensemble_with_graph`.

Say:

> “The local worker scores the packaged cases and the console verifies the
> returned event IDs, probabilities, actions, and bundle binding before it
> renders them.”

Click **Play both streams**, allow three or four steps, then click **Pause both
streams**. Point to the moving payment edge, calibrated probability, bound
thresholds, final action, reason code, and feature count.

Say:

> “One presentation control advances the campaign graph and portable decision
> trace in their own repository order. It is a synchronized walkthrough, not a
> claimed row-level join between those evidence streams.”

Select portable event **08** in the event ledger without moving the campaign
selector. Point to the separate ground-truth examination panel, then click
**Reset both**. Click **Investigation** in the left navigation.

### 2:38–3:18 — Investigation

Say:

> “The investigator receives one deterministic campaign case: fourteen linked
> entities, ten payment edges, and a conserved five-hundred-dollar synthetic
> attempted-value progression.”

Select the central mule node and point to its linked payments. Mention that
**Evidence pending** is shown where analyst-time evidence does not exist. Click
**Defenses**.

### 3:18–4:00 — Defenses

Keep the default focus on the champion and point to the champion badge. Do not
hover across the alternative arm cards.

Say:

> “The accepted portable competition model is `ensemble_with_graph`. The
> recovered comparison shows 0.978 F1, 0.0037 percent false-decline, and a
> 0.572 percent challenge rate for this arm. These numbers are verified
> synthetic diagnostics only: non-authoritative, not accepted capacity
> evidence, and not production estimates.”

Point to the visible non-authoritative qualifier, then click **Assurance**.

### 4:00–4:48 — Assurance

Select the first two artifact rows to expose their hashes. Follow the human
promotion boundary and the separate authorization-integrity checks.

Say:

> “Every shown decision remains bound to inspectable artifacts, replay checks,
> and a human promotion boundary. The authorization checks separately verify
> identity, mandate, scope, cart binding, and replay rejection; they are not a
> fraud-model performance claim.”

Close with:

> “APAR demonstrates evidence before assertion: a working synthetic assurance
> range, the portable `ensemble_with_graph` champion, honest diagnostics, and
> no claim of production deployment, Mastercard data, or external validation.”

Hold the final Assurance frame without moving the pointer until 5:00.

## Recovery if the local scorer fails

If the local page says **Verified fallback · offline**, do not continue with
the local-model wording.

1. Stop recording and preserve the take only for rehearsal notes.
2. In Terminal, stop the server with **Control–C**, then run:

   ```bash
   cd '/Users/dylanmoraes/.codex/worktrees/03b9/MasterCard Challenge'
   .venv/bin/python scripts/run_apar_console.py health
   .venv/bin/python scripts/run_apar_console.py reset
   .venv/bin/python scripts/run_apar_console.py start
   ```

3. Reload `http://127.0.0.1:4173/replay` and confirm **LIVE LOCAL SCORER**.
4. If it still fails, use the deterministic fallback drill:

   ```bash
   .venv/bin/python scripts/run_apar_console.py start --fallback-only
   ```

   Record the hosted/static narrative only and say that the committed
   hash-bound trace is being replayed. Do not say the Python scorer ran.

Never repair the demo by changing evidence, regenerating protected artifacts,
training a model, or running a locked evaluation.

## Final export, compression, and upload checks

1. Open the `.mov` in QuickTime Player. Trim only the silent handles; do not
   cut off the hosted/local transition or any runtime label.
2. Use **File → Export As → 1080p**. Choose **Greater Compatibility (H.264)**
   when QuickTime offers that option.
3. If a smaller MP4 is required and `ffmpeg` is already installed, use:

   ```bash
   ffmpeg -i 'APAR-final.mov' -c:v libx264 -preset medium -crf 20 -pix_fmt yuv420p -c:a aac -b:a 160k -movflags +faststart 'APAR-final-1080p.mp4'
   ```

4. Inspect the exported file:

   ```bash
   mdls -name kMDItemDurationSeconds -name kMDItemPixelWidth -name kMDItemPixelHeight -name kMDItemFSSize 'APAR-final-1080p.mp4'
   ```

5. Watch the entire export once with headphones and once muted. Confirm:
   - 1920×1080 output, readable text, and no horizontal clipping;
   - clear microphone audio with no notification sounds;
   - the hosted fallback label and local-scorer label are both visible;
   - no bookmarks, personal notifications, Terminal history, tokens, or
     secrets appear;
   - the twelve cases are described as curated demonstrations;
   - recovered numbers are immediately qualified as verified synthetic
     diagnostics;
   - there is no production, Mastercard-data, or external-validation claim;
   - the video begins and ends cleanly and stays within the competition limit.

Keep the original `.mov` until the uploaded MP4 has been played back from the
competition portal. Uploading or submitting the video remains a user-owned
action.
