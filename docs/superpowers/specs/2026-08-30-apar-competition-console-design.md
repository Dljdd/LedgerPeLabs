# APAR competition assurance console design

**Date:** 2026-08-30  
**Baseline:** `d8764266459bada13707aa155fe43df90a3f50fd`  
**Branch:** `codex/apar-competition-console`  
**Status:** Approved design awaiting implementation-plan review

## 1. Objective

Build a judge-facing, offline-capable web prototype that explains the APAR assurance
loop in five minutes. The product is an institutional payment-risk assurance console,
not a general analytics dashboard. It must let a judge move from the reviewed threat,
through the bounded synthetic scenario and accepted portable-model replay, to an
investigation view and an explicit human promotion gate without external guidance.

The smallest acceptable vertical slice contains six persistent sections:

1. Overview
2. Scenario
3. Replay
4. Investigation
5. Defenses
6. Assurance

The app preserves one canonical selected context across all six sections and restores
that context on reload or reset.

## 2. Evidence and safety boundary

The prototype must not train, adapt, publish, promote, or alter a model. It must not
execute seed 2404 or any locked, production, sealed, confirmatory, or adaptive
experiment. It must not modify Kaggle recovery material, accepted checkpoints, frozen
evidence, thresholds, or historical result artifacts.

Three evidence lanes remain structurally and visually separate.

### 2.1 Portable model evidence

The following immutable repository artifacts are the only source for displayed model
probabilities, member scores, actions, thresholds, replay metrics, and model hashes:

- `demo/sentinel-v5/manifest.json`
- `demo/sentinel-v5/spec.json`
- `demo/sentinel-v5/scenarios.json`
- `demo/sentinel-v5/models/`
- `demo/sentinel-v5/calibrators/`
- output from `scripts/run_sentinel_v5_demo.py`

Every live prediction and trace is labeled `ensemble_with_graph`. The UI must never
call it `full_sentinel`, a complete hybrid, production evidence, accepted capacity
evidence, sealed evidence, or a production estimate.

The portable bundle's flags are immutable presentation requirements:

- `demo_only=true`
- `authoritative=false`
- `accepted_capacity_evidence=false`

The 12-case aggregate metrics are labeled curated synthetic replay descriptions. They
are never presented as expected production performance.

`presentation_ground_truth` is displayed in its own disclosure region, separated from
model input and output by heading, surface, and data contract. It is never included in
the scorer request.

### 2.2 Scenario and investigation context

The approved threat card at `fixtures/threats/app-personalized-mule.json` is the source
for title, confidence, source/inference distinction, rails, capability delta, safety
class, lifecycle, bounds, query budget, duration, and seed 260816.

A committed scenario-context artifact may be generated only from the existing public
compiler, population generator, APP campaign generator, rail simulator, case grouping,
and seed 260816. Its generation command and SHA-256 are recorded. It is labeled
`Deterministic synthetic scenario context` and is not represented as the hidden source
of the portable checkpoint rows.

The context artifact may provide:

- ordered synthetic rail events and stages;
- pseudonymous entity and account identifiers;
- actual generated graph nodes and edges;
- campaign graph and schedule digests;
- value flow and conservation evidence;
- actual case-grouping output when the repository case engine can bind it safely.

If first-alert, value-progression, or analyst-effort evidence cannot be generated and
bound to this artifact, the corresponding UI value reads `Evidence pending`. It must
not contain a placeholder number. Clearly labeled static product illustrations may be
used only for non-model layout or workflow explanation and must not be mixed into an
evidence table.

### 2.3 Agentic-integrity evidence

The agentic proof point uses `src/apar/trust/verifier.py` and a deterministic synthetic
fixture equivalent to the tested records in `tests/trust/test_verifier.py`. The
committed public proof artifact contains no private key or credential secret.

It shows the actual ordered outcomes for:

- registered identity and signature;
- mandate and amount/category scope;
- merchant, payee, cart, and payment-intent binding;
- expiry and authentication evidence;
- nonce and receipt-chain replay controls.

This panel is labeled deterministic trust verification and remains separate from the
`ensemble_with_graph` prediction claim.

### 2.4 Recovered diagnostic metrics

The following recovered-metrics artifacts may supply genuine four-arm comparison
values and readiness gates:

- `docs/demo/SENTINEL_V5_RECOVERED_METRICS.md`
- `evidence/sentinel-v5-recovered-metrics/verified-report.json`
- `evidence/sentinel-v5-recovered-metrics/source-rescue-receipt.json`

Every surface using these values displays the exact qualifier
`Recovered diagnostic evidence — non-authoritative`. The qualifier must remain visible
with the table or gate result at every supported breakpoint.

The required boundary is:

- `authoritative=false`;
- `accepted_capacity_evidence=false`;
- official chain status `incomplete`;
- first missing official stage `70_metrics`;
- readiness `not_ready`;
- the accepted portable `ensemble_with_graph` arm remains the live demo model;
- `full_sentinel` failed false-decline, challenge-rate, and `benign_only` gates.

The recovered four-arm metrics are diagnostic projections of verified self-hashed
documents. They are not accepted Stage 70 results and must not be combined with the
12-case portable replay metrics into one estimate. The strongest supported conclusion
is stated plainly: the graph ensemble is the currently usable competition model, while
deterministic full-hybrid routing requires policy refinement.

Seed and attempt wording must remain exact. The portable demo and recovered Kaggle
metrics use seed 404 only. No Kaggle locked-successor/seed-2404 chain was run. An
earlier local locked-development attempt was started and irreversibly aborted; it
published no candidate manifest, chunks, judge summary, or successful seed-2404 result.
No retry is permitted. The UI and walkthrough must not claim that the seed had no
prior execution history.

## 3. Technical architecture

### 3.1 Stack

Use a self-contained React 19.2 and TypeScript client built with Vite 8.1 and Tailwind
CSS 4.3. Use semantic app-owned components rather than adding a component suite. Use
Playwright 1.62 for end-to-end and screenshot tests, Vitest for component and data
contract tests, and Testing Library for interaction tests.

The stack was checked against official project documentation on 2026-08-30. Exact
patch versions are committed through `web/package-lock.json`.

No runtime dependency may load remote code, fonts, icons, images, telemetry, analytics,
or data. Icons are small app-owned inline SVGs with accessible names or hidden
decorative semantics.

### 3.2 Directory boundary

All new frontend source is isolated under `web/`:

```text
web/
  public/data/              committed canonical presentation artifacts
  src/app/                  shell, state, data loading, route handling
  src/components/           semantic reusable console primitives
  src/features/             six product sections
  src/styles/               tokens and global responsive rules
  tests/                    unit and component tests
  e2e/                      Playwright golden path, a11y, responsive, offline
```

Supporting prototype-only scripts are explicit:

```text
scripts/build_apar_console_fixture.py
scripts/run_apar_console.py
scripts/check_apar_console.py
```

Generated judge materials live under:

```text
submission/APAR_CONSOLE_WALKTHROUGH.md
submission/APAR_CONSOLE_GAPS.md
submission/screenshots/apar-console-desktop.png
submission/screenshots/apar-console-mobile.png
```

No video is produced.

### 3.3 Local server and scorer bridge

`scripts/run_apar_console.py` is the single start command after installation. It uses
Python's standard-library HTTP server to serve the built client and a small same-origin
JSON API. It does not require an application server package.

At startup the server:

1. verifies the committed presentation artifacts;
2. attempts to load and run the accepted portable scorer;
3. verifies exact probability/action replay;
4. exposes the verified trace at `/api/v1/replay`;
5. serves the app from `web/dist`;
6. exposes `/api/v1/health` and `/api/v1/reset`.

If the Python scoring worker cannot import, load, or verify the bundle, the server
continues in degraded mode and serves the committed fallback trace. The fallback file
must be direct output from `scripts/run_sentinel_v5_demo.py`, retain its original
`trace_sha256`, and pass the same manifest/scenario consistency checks before it can be
served. The top status strip visibly says `Verified live scorer` or
`Verified fixed fallback`; it never silently falls back.

The client never sends model features over a network. All communication is loopback,
same-origin, and JSON-only.

### 3.4 Client state and navigation

Use native browser history and a six-route route table rather than a router dependency:

- `/overview`
- `/scenario`
- `/replay`
- `/investigation`
- `/defenses`
- `/assurance`

Unknown routes redirect to `/overview`. The selected replay row is stored in client
state and reflected in the URL fragment for reloadability. Reset clears transient
state, selects the canonical APP event, restores replay position zero, and navigates to
Overview.

The shell must work without JavaScript-driven entrance completion: all primary content
is visible in the first rendered state.

## 4. Product sections

### 4.1 Overview

Question answered: Why does APAR matter?

The page presents:

- the APAR thesis in one sentence;
- the approved APP scam and mule threat title and confidence;
- source statements versus project inference;
- explicit GenAI capability delta: personalization and iteration speed;
- A2A and agentic rail coverage;
- a six-step golden-path map;
- a clear synthetic-only boundary.

No unbound market-size or fraud-loss number is shown.

### 4.2 Scenario

Question answered: What is being replayed, and under what constraints?

The page presents the approved card's lifecycle, rail/viewpoint, duration, population
bounds, attacker objective, query budget, feedback scope, seed 260816, configured-cost
markers, synthetic-only export level, and replay ordering. The primary control starts
or resets replay; it does not execute an adaptive search.

The lifecycle is a semantic ordered list. A compact visual connector may mirror the
same stage data but is never the only representation.

### 4.3 Replay

Question answered: What did the accepted portable arm decide?

The main view contains:

- an ordered table of the 12 genuine curated scenario records;
- a focused APP sequence derived from genuine APP event identifiers;
- calibrated probability and three calibrated member scores;
- model action/final action, reason code, disagreement, and measured latency;
- bundle manifest, arm spec, threshold, and trace evidence links;
- a collapsed ground-truth drawer in its own `Post-event truth` region.

The replay control advances one record at a time, pauses, and resets. Progress motion
is allowed because it explains state. Selecting by keyboard changes state immediately
without animated delay.

### 4.4 Investigation

Question answered: How does a transaction alert become an actionable campaign case?

The page contains an actual deterministic APP scenario graph and synchronized
relationship table. Node focus highlights immediate causal neighbors and updates an
accessible textual description. It includes generated actor/account roles, edge amount,
rail-event reference, first bound alert when available, cumulative value progression,
case grouping evidence, and analyst effort only when bound.

The graph is an SVG with fixed deterministic positions calculated by entity role and
causal layer. It has no force simulation, decorative particles, glow, or idle motion.
Mobile displays the relationship table first and the graph as an optional detail.

### 4.5 Defenses

Question answered: What does each defense arm contain, and what is actually evidenced?

Render a true comparison table for:

- rules;
- `ensemble_no_graph`;
- `ensemble_with_graph`;
- `full_sentinel`.

Rows cover deterministic rules, model, graph features, trust routing, novelty,
disagreement, available evidence, and live-demo status. The portable spec is the source
for `ensemble_with_graph` switches. Repository arm contracts are the source for the
other architecture rows.

The view may display the genuine four-arm recovered metrics from
`verified-report.json`, provided the complete table is headed
`Recovered diagnostic evidence — non-authoritative` and visibly binds the incomplete
official chain and `not_ready` status. The portable arm's 12-case replay metrics remain
in a separate block labeled curated synthetic replay descriptions. Any metric absent
from both sources reads `Evidence pending`; no placeholder bars, values, or rankings
are drawn.

The `full_sentinel` row and detail must expose its false-decline rate, challenge rate,
and failed gates without collapse or euphemism. It must not be styled as champion-ready.
The primary conclusion names `ensemble_with_graph` as the usable competition model and
names full-hybrid policy refinement as remaining work.

### 4.6 Assurance

Question answered: Can a reviewer reconstruct the claim, and who may promote it?

The page presents:

- manifest, source checkpoint, spec, threshold, scenario, and trace hashes;
- verified file and replay checks with textual pass/fail state;
- synthetic, demo-only, non-authoritative, and not-capacity-evidence badges;
- recovered report verification hash, incomplete official chain, first missing
  `70_metrics` stage, and `not_ready` readiness;
- all failed `full_sentinel` readiness gates alongside passed gates;
- exact seed-404 recovery and irreversibly aborted local-attempt wording, including the
  absence of published candidate artifacts or a successful seed-2404 result and the
  no-retry boundary;
- known limitations and remaining submission gaps;
- the TrustVerifier proof point;
- an explicit human promotion gate.

The gate is a non-mutating product demonstration. Its initial and reset state is
`Human review required`. A reviewer may inspect the required acknowledgement control,
but the prototype cannot mark the model approved or alter any artifact.

## 5. Visual system

### 5.1 Direction

The console uses a deep-neutral institutional palette and dense, opaque analytical
surfaces. Translucency is limited to the sticky navigation backplate where it improves
hierarchy. There are no purple hues, blue-to-purple gradients, decorative grids,
glows, fake application windows, floating-card compositions, resting hover lifts, or
content-gating entrance animations.

Mastercard-adjacent orange is an accent, not a brand imitation. Semantic red, amber,
and green always appear with text and a non-color cue.

### 5.2 Tokens

Color tokens target WCAG 2.2 AA in both default and forced/high-contrast contexts:

```text
--ink-950:          #080a0d
--surface-900:      #101318
--surface-850:      #171b21
--surface-800:      #1d2229
--line-700:         #303741
--text-100:         #f4f2ed
--text-300:         #c7c4bc
--text-500:         #97958e
--accent-orange:    #ff7a1a
--accent-orange-2:  #ff9b52
--signal-red:       #ff5d5d
--signal-amber:     #f5ba45
--signal-green:     #57c785
--focus:            #ffd0a8
```

Typography uses a local system stack. Headings are compact and decisive; tables,
probabilities, money, durations, and hashes use tabular or monospaced numerals.

```text
display:  clamp(2.25rem, 5vw, 4.75rem) / 0.98
h1:       clamp(1.75rem, 3vw, 3rem) / 1.05
h2:       clamp(1.2rem, 2vw, 1.65rem) / 1.2
body:     1rem / 1.55
small:    0.8125rem / 1.45
micro:    0.6875rem / 1.35, uppercase tracking
```

Spacing uses a four-pixel base with 4, 8, 12, 16, 24, 32, 48, and 64-pixel steps.
Radii are restrained: 4 pixels for controls, 8 pixels for surfaces, and 999 pixels
only for short status badges. Dense tables remain square enough to feel operational.

Elevation comes from surface contrast and one restrained navigation shadow. Analytical
cards have no decorative drop shadow.

### 5.3 Core components

- **Status strip:** global worker state, arm identity, synthetic boundary, and reset.
- **Section navigation:** visible labels, `aria-current`, progress index, horizontal
  overflow on narrow screens without clipping.
- **Evidence badge:** icon/text/color cue and accessible expansion.
- **Metric block:** label, tabular value, provenance link, and evidence qualifier.
- **Data table:** real table semantics, sticky headers on desktop, row focus, responsive
  overflow wrapper, and no card conversion that loses column relationships.
- **Decision inspector:** input/features, model output, and post-event truth in three
  separate regions.
- **Entity graph:** deterministic SVG plus equivalent relationship table.
- **Hash row:** full copyable value, visually abbreviated only where an adjacent reveal
  exposes the complete digest.
- **Notice:** loading, empty, degraded, failure, or evidence-pending state with cause and
  next safe action.

### 5.4 Motion contract

The product personality is crisp institutional operations. Motion communicates replay
causality, graph focus, and meaningful worker-state changes only.

Tokens:

```text
--ease-out:     cubic-bezier(0.23, 1, 0.32, 1)
--ease-in-out:  cubic-bezier(0.77, 0, 0.175, 1)
--duration-press: 140ms
--duration-fast:  160ms
--duration-state: 220ms
```

Rules:

- buttons scale to `0.97` on pointer press for 140ms;
- keyboard-triggered navigation and replay controls change immediately;
- graph focus uses opacity and transform for at most 220ms;
- replay progress uses transform with `--ease-in-out` for at most 220ms;
- status notices use interruptible CSS transitions, not keyframes;
- no `transition: all`, `ease-in`, `scale(0)`, layout-property animation, unguarded hover
  transform, or continuous ambient motion;
- `prefers-reduced-motion: reduce` removes positional motion while retaining immediate
  state and short opacity/color feedback.

Content is never hidden waiting for animation completion.

## 6. Responsive behavior

Supported widths are 360, 768, 1024, and 1440 pixels, plus 200 percent browser zoom.

- **Desktop:** sticky left context rail, horizontal section navigation, two-column
  evidence/detail layouts, wide comparison table, graph beside case evidence.
- **Tablet:** compact top context bar, single primary column with paired metrics where
  space permits, scroll-contained tables.
- **Mobile:** one column, touch targets at least 44 by 44 pixels, navigation in a labeled
  horizontal scroller, tables in named overflow regions, relationship table before SVG,
  hashes wrapping without page overflow.

No breakpoint may hide an evidence qualifier, worker state, arm identity, or human gate.

## 7. Accessibility

The implementation targets WCAG 2.2 AA:

- skip link and landmark structure;
- logical heading order;
- complete keyboard operation and visible three-pixel focus outline;
- no color-only status;
- minimum 44-pixel primary controls;
- accessible names for SVG and icon controls;
- status updates through restrained live regions;
- graph/table equivalence;
- reduced-motion support;
- focus restoration after closing details;
- no focus trap outside a true modal;
- readable content at 200 percent zoom without two-dimensional page scrolling.

Automated checks supplement keyboard and visual inspection; they do not replace it.

## 8. Loading, empty, degraded, and failure states

- **Loading:** deterministic text such as `Verifying portable bundle · step 2 of 4`;
  no indefinite unlabeled spinner.
- **Empty:** explains which artifact produced no rows and offers reset.
- **Degraded:** serves the verified fixed fallback, names the worker failure category,
  and preserves the complete walkthrough.
- **Failure:** blocks claims when neither live nor fallback trace verifies; Overview and
  static scenario evidence remain available while model-value surfaces fail closed.
- **Evidence pending:** shown for unbound arm metrics and investigation fields; never
  replaced with zero, an em dash without explanation, or a sample value.

## 9. Test-first implementation

Production behavior is added only after a focused failing test demonstrates it.

### 9.1 Data and Python tests

- fixture builder rejects seed 2404 and any locked/confirmatory mode;
- portable fallback retains exact source trace hash and flags;
- scenario-context hash is deterministic;
- truth is absent from scorer inputs;
- arm identity is exactly `ensemble_with_graph`;
- trust proof contains no private key material;
- launcher reports live, degraded, and fail-closed states correctly;
- reset returns the canonical selection.

### 9.2 Component tests

- every section renders source-bound qualifiers;
- Overview distinguishes fact from inference;
- Replay separates input/output from post-event truth;
- Defenses renders all four arms and `Evidence pending` where required;
- recovered metrics retain both non-authoritative flags and cannot be rendered without
  the exact recovered-evidence qualifier;
- `full_sentinel` failures and the incomplete Stage 70 chain remain visible;
- Investigation keeps graph and relationship table synchronized;
- Assurance cannot promote itself;
- loading, empty, degraded, and failure notices expose cause and action;
- keyboard selection, focus, and reduced motion work.

### 9.3 Playwright tests

- five-minute golden path follows Overview → Scenario → Replay → Investigation →
  Defenses → Assurance;
- runtime networking outside loopback is blocked;
- degraded worker still completes the path from fallback;
- reload/reset reproduces canonical state;
- desktop and mobile keyboard paths work;
- no horizontal page overflow or clipped controls at supported widths and 200 percent
  zoom;
- automated accessibility scan has no serious or critical violations;
- representative desktop and mobile screenshots are captured from deterministic state.

## 10. Commands and documentation

The root README documents exact clean-install, run, health, offline, reset, test, and
build commands. After dependencies are installed, the single start command is:

```bash
.venv/bin/python scripts/run_apar_console.py
```

Preflight is:

```bash
.venv/bin/python scripts/check_apar_console.py
```

The walkthrough document gives Dylan a timed five-minute narration with explicit click
targets and recovery notes. The gaps document distinguishes completed prototype work
from remaining submission work.

## 11. Verification and completion gate

Before the final commit, run fresh complete checks for:

- frontend unit/component tests;
- TypeScript;
- lint;
- production build;
- Playwright golden path, offline fallback, accessibility, keyboard, and responsive
  coverage;
- existing portable-model replay;
- relevant Python tests;
- preflight and health checks;
- `git diff --check`;
- secret patterns and absolute home paths;
- generated-asset allowlist;
- Git author and committer identity;
- clean worktree.

The final commit must be authored and committed as
`Dylan Moraes <dylanmoraesdljdd@gmail.com>`. Commit messages and metadata contain no
tool or assistant attribution.

## 12. Out of scope

- model retraining or threshold selection;
- adaptive search execution;
- seed 2404;
- locked, production, sealed, confirmatory, or publication evaluation;
- Kaggle recovery changes;
- promotion or model-registry mutation;
- production deployment;
- remote analytics, telemetry, fonts, or runtime APIs;
- walkthrough video creation.
