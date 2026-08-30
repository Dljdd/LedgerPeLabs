# APAR Competition Assurance Console Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a presentation-ready, offline judge console that walks through APAR threat evidence, a bounded synthetic scenario, accepted portable replay, genuine investigation context, recovered diagnostic comparisons, and a human assurance gate.

**Architecture:** A strict Vite/React/TypeScript client loads a committed canonical evidence document and a same-origin replay endpoint. A standard-library Python server verifies the immutable bundle, prefers the live `ensemble_with_graph` scorer, and fails over visibly to a committed hash-bound trace. All model, truth, recovered, investigation, and TrustVerifier evidence lanes remain distinct.

**Tech Stack:** React 19.2, TypeScript, Vite 8.1, Tailwind CSS 4.3, Vitest, Testing Library, Playwright 1.62, Python 3.12 standard-library HTTP server, existing APAR scorer/generator/trust modules.

**Spec:** `docs/superpowers/specs/2026-08-30-apar-competition-console-design.md`

## Global Constraints

- Start from merged commit `ce41fa903e988c8943d1c6c6b9aeb9bf340915a6`; do not rewrite or regenerate its recovered evidence.
- Never train or adapt a model, execute seed 2404, run a locked/Kaggle/confirmatory/adaptive experiment, or modify accepted bundles, thresholds, checkpoints, or historical results.
- Live predictions are always labeled `ensemble_with_graph`, never `full_sentinel` or complete hybrid.
- Portable and recovered evidence must retain `authoritative=false` and `accepted_capacity_evidence=false`.
- Every recovered comparison must display `Recovered diagnostic evidence — non-authoritative`, `official_chain_status=incomplete`, `first_missing_official_stage=70_metrics`, and `readiness=not_ready`.
- Display all `full_sentinel` failures: false-decline, challenge-rate, and `benign_only`.
- State that the portable demo and recovered Kaggle metrics use seed 404 only; no Kaggle locked-successor/seed-2404 chain was run; the earlier local locked-development attempt was started and irreversibly aborted without publishing a candidate manifest, chunks, judge summary, or successful seed-2404 result; no retry is permitted.
- Do not claim the seed had no prior execution history.
- All runtime assets are local; no telemetry, analytics, remote fonts, remote images, or external runtime requests.
- Motion is limited to replay progress, graph focus, pointer press feedback, and meaningful state changes; all content is visible without animation completion.
- Author and commit every commit as `Dylan Moraes <dylanmoraesdljdd@gmail.com>` with no tool or assistant attribution.

---

### Task 1: Build the canonical evidence document

**Files:**
- Create: `tests/prototype/test_console_evidence.py`
- Create: `scripts/build_apar_console_evidence.py`
- Create: `web/public/data/console-evidence.json`

**Interfaces:**
- Consumes: approved APP threat card, portable manifest/spec/scenarios, recovered verified report and receipt, public campaign generator, and TrustVerifier contracts.
- Produces: `build_console_evidence(root: Path) -> dict[str, object]` and canonical `console-evidence.json` with `document_sha256`.

- [ ] **Step 1: Write the failing evidence-boundary tests**

```python
def test_console_evidence_preserves_model_and_recovery_boundaries(repo_root: Path) -> None:
    document = build_console_evidence(repo_root)
    assert document["portable"]["arm"] == "ensemble_with_graph"
    assert document["portable"]["authoritative"] is False
    assert document["portable"]["accepted_capacity_evidence"] is False
    assert document["recovered"]["qualifier"] == (
        "Recovered diagnostic evidence — non-authoritative"
    )
    assert document["recovered"]["official_chain_status"] == "incomplete"
    assert document["recovered"]["first_missing_official_stage"] == "70_metrics"
    assert document["recovered"]["readiness"]["status"] == "not_ready"
    failed = {row["metric"] for row in document["recovered"]["failed_gates"]}
    assert {"false_decline_rate", "challenge_rate", "benign_only"} <= failed

def test_console_evidence_never_mixes_truth_into_model_input(repo_root: Path) -> None:
    document = build_console_evidence(repo_root)
    for row in document["portable"]["records"]:
        assert "presentation_ground_truth" not in row["model_input"]
        assert set(row["post_event_truth"]) == {"amount", "currency", "family", "label", "rail"}
```

- [ ] **Step 2: Run the tests and verify RED**

Run: `python -m pytest tests/prototype/test_console_evidence.py -q`

Expected: FAIL because `scripts.build_apar_console_evidence` does not exist.

- [ ] **Step 3: Implement canonical loading, SHA-256 verification, and presentation projection**

Implement exact-type JSON loaders, canonical JSON hashing, portable-manifest self-hash verification, recovered-report verification-hash checking through `apar.evaluation.v5_rescue_verifier`, and projection into independent keys:

```python
document = {
    "schema_version": "apar-console-evidence/1",
    "threat": threat_projection,
    "scenario_context": generated_app_context,
    "portable": portable_projection,
    "recovered": recovered_projection,
    "trust_proof": trust_projection,
    "copy_boundary": seed_and_attempt_copy,
}
document["document_sha256"] = sha256(canonical(document)).hexdigest()
```

Generate the APP context only with seed `260816`, the approved threat card, ten A2A payments, the public campaign generator, and deterministic role/layer layout. Project actual command IDs, pseudonymous account references, amounts, causal dependencies, graph digest, and schedule digest. If case-engine evidence cannot be bound, emit `{"status": "evidence_pending"}` rather than a number.

Build the TrustVerifier projection from deterministic public receipt fields only. Exclude private key bytes, signatures, and credentials; include actual pass/fail reason codes for identity, mandate, binding, expiry, and replay.

- [ ] **Step 4: Run the evidence tests and deterministic rebuild check**

Run: `python -m pytest tests/prototype/test_console_evidence.py -q`

Run the builder twice into separate temporary files and compare SHA-256 values. Expected: PASS and byte-identical output.

- [ ] **Step 5: Commit the evidence bridge**

```bash
git add tests/prototype/test_console_evidence.py scripts/build_apar_console_evidence.py web/public/data/console-evidence.json
git -c user.name="Dylan Moraes" -c user.email=dylanmoraesdljdd@gmail.com commit -m "feat: bind console evidence sources"
```

### Task 2: Bootstrap the strict offline React client

**Files:**
- Create: `web/package.json`
- Create: `web/package-lock.json`
- Create: `web/tsconfig.json`
- Create: `web/tsconfig.app.json`
- Create: `web/vite.config.ts`
- Create: `web/vitest.config.ts`
- Create: `web/playwright.config.ts`
- Create: `web/eslint.config.js`
- Create: `web/index.html`
- Create: `web/src/main.tsx`
- Create: `web/src/app/types.ts`
- Create: `web/src/app/evidence.ts`
- Create: `web/src/app/routes.ts`
- Create: `web/src/app/App.test.tsx`

**Interfaces:**
- Consumes: `/data/console-evidence.json`, `/api/v1/health`, `/api/v1/replay`.
- Produces: `ConsoleEvidence`, `ReplayResponse`, `loadConsoleEvidence()`, `loadReplay()`, and six native-history routes.

- [ ] **Step 1: Write failing schema and route tests**

```tsx
it("rejects a portable arm mislabeled as full_sentinel", () => {
  expect(() => parseEvidence({ ...fixture, portable: { ...fixture.portable, arm: "full_sentinel" } }))
    .toThrow("portable arm must be ensemble_with_graph");
});

it("redirects an unknown route to overview", () => {
  history.replaceState({}, "", "/unknown");
  render(<App />);
  expect(screen.getByRole("heading", { name: /adaptive payment assurance/i })).toBeVisible();
});
```

- [ ] **Step 2: Install dependencies and verify RED**

Run: `npm --prefix web install`

Run: `npm --prefix web test -- --run`

Expected: FAIL because app/data modules are absent.

- [ ] **Step 3: Implement strict types, explicit runtime narrowing, and history navigation**

Set `strict`, `noUncheckedIndexedAccess`, and `exactOptionalPropertyTypes`. Do not use `any`. Route labels are Overview, Scenario, Replay, Investigation, Defenses, and Assurance. Unknown paths call `history.replaceState({}, "", "/overview")`.

- [ ] **Step 4: Run typecheck, lint, and tests**

Run: `npm --prefix web run typecheck`

Run: `npm --prefix web run lint`

Run: `npm --prefix web test -- --run`

- [ ] **Step 5: Commit the client boundary**

```bash
git add web
git -c user.name="Dylan Moraes" -c user.email=dylanmoraesdljdd@gmail.com commit -m "build: bootstrap assurance console client"
```

### Task 3: Implement the institutional shell and design system

**Files:**
- Create: `web/src/styles/tokens.css`
- Create: `web/src/styles/global.css`
- Create: `web/src/components/AppShell.tsx`
- Create: `web/src/components/StatusStrip.tsx`
- Create: `web/src/components/SectionNav.tsx`
- Create: `web/src/components/EvidenceBadge.tsx`
- Create: `web/src/components/Metric.tsx`
- Create: `web/src/components/DataTable.tsx`
- Create: `web/src/components/Notice.tsx`
- Create: `web/src/components/HashRow.tsx`
- Create: `web/src/components/components.test.tsx`

**Interfaces:**
- Consumes: current route, health/replay source status, evidence qualifiers.
- Produces: accessible shell primitives used by every feature.

- [ ] **Step 1: Write failing semantic, focus, and motion tests**

```tsx
it("keeps the live arm and evidence boundary visible", () => {
  render(<StatusStrip source="fallback" />);
  expect(screen.getByText("ensemble_with_graph")).toBeVisible();
  expect(screen.getByText("Verified fixed fallback")).toBeVisible();
});

it("does not use animation to hide content", () => {
  const { container } = render(<Metric label="Recall" value="0.998673" qualifier="Recovered diagnostic evidence — non-authoritative" />);
  expect(container.querySelector('[style*="opacity: 0"]')).toBeNull();
});
```

- [ ] **Step 2: Verify RED**

Run: `npm --prefix web test -- --run src/components/components.test.tsx`

- [ ] **Step 3: Implement tokens and components**

Use spec colors and typography, a 4/8/12/16/24/32/48/64 spacing scale, tabular numerals, 44-pixel controls, and a three-pixel focus outline. Add exact motion tokens:

```css
--ease-out: cubic-bezier(0.23, 1, 0.32, 1);
--ease-in-out: cubic-bezier(0.77, 0, 0.175, 1);
--duration-press: 140ms;
--duration-fast: 160ms;
--duration-state: 220ms;
```

Buttons use pointer-only `transform: scale(.97)` press feedback. No `transition: all`, `ease-in`, `scale(0)`, hover lift, background grid, glow, purple, or layout-property animation. Reduced motion removes position changes.

- [ ] **Step 4: Run focused tests and static motion-law scan**

Run: `npm --prefix web test -- --run src/components/components.test.tsx`

Run: `rg -n 'transition:\s*all|ease-in(?!-out)|scale\(0\)|#[0-9a-fA-F]{6}' web/src --pcre2`

Expected: no forbidden motion patterns; color literals exist only in `tokens.css`.

- [ ] **Step 5: Commit the shell**

```bash
git add web/src/styles web/src/components web/src/app
git -c user.name="Dylan Moraes" -c user.email=dylanmoraesdljdd@gmail.com commit -m "feat: add institutional console shell"
```

### Task 4: Build Overview and Scenario

**Files:**
- Create: `web/src/features/overview/Overview.tsx`
- Create: `web/src/features/overview/Overview.test.tsx`
- Create: `web/src/features/scenario/Scenario.tsx`
- Create: `web/src/features/scenario/Lifecycle.tsx`
- Create: `web/src/features/scenario/Scenario.test.tsx`

**Interfaces:**
- Consumes: `ConsoleEvidence.threat`, `scenario_context`, and `copy_boundary`.
- Produces: first two golden-path sections and replay/reset navigation actions.

- [ ] **Step 1: Write failing source/inference, boundary, and keyboard tests**

Assert direct evidence and project inference have different labels; rails are A2A and agentic; seed 260816 and query budget 40 come from the card; synthetic-only is visible; the lifecycle is an ordered list; no prohibited execution-history claim appears.

- [ ] **Step 2: Verify RED**

Run: `npm --prefix web test -- --run src/features/overview src/features/scenario`

- [ ] **Step 3: Implement the restrained editorial overview and bounded scenario ledger**

Use a large thesis statement, one operational evidence rail, a compact six-step walkthrough, and an opaque constraint ledger. Source every value from evidence props. “Start replay” navigates without executing search or generation.

- [ ] **Step 4: Run focused tests**

Run: `npm --prefix web test -- --run src/features/overview src/features/scenario`

- [ ] **Step 5: Commit the first walkthrough segment**

```bash
git add web/src/features/overview web/src/features/scenario
git -c user.name="Dylan Moraes" -c user.email=dylanmoraesdljdd@gmail.com commit -m "feat: add threat and scenario walkthrough"
```

### Task 5: Build Replay and Investigation

**Files:**
- Create: `web/src/features/replay/Replay.tsx`
- Create: `web/src/features/replay/ReplayTimeline.tsx`
- Create: `web/src/features/replay/DecisionInspector.tsx`
- Create: `web/src/features/replay/Replay.test.tsx`
- Create: `web/src/features/investigation/Investigation.tsx`
- Create: `web/src/features/investigation/EntityGraph.tsx`
- Create: `web/src/features/investigation/RelationshipTable.tsx`
- Create: `web/src/features/investigation/Investigation.test.tsx`

**Interfaces:**
- Consumes: verified replay response and deterministic scenario context.
- Produces: replay selection/advance/reset and synchronized graph/table focus.

- [ ] **Step 1: Write failing replay-truth separation and graph-equivalence tests**

```tsx
it("keeps post-event truth outside model input and output", () => {
  render(<DecisionInspector record={record} />);
  expect(within(screen.getByRole("region", { name: "Model output" })).queryByText("Fraud label")).toBeNull();
  expect(within(screen.getByRole("region", { name: "Post-event truth" })).getByText("Fraud label")).toBeVisible();
});

it("renders every graph edge in the relationship table", () => {
  render(<Investigation context={context} />);
  expect(screen.getAllByRole("row")).toHaveLength(context.edges.length + 1);
});
```

- [ ] **Step 2: Verify RED**

Run: `npm --prefix web test -- --run src/features/replay src/features/investigation`

- [ ] **Step 3: Implement ordered replay, decision evidence, and deterministic SVG**

Order curated rows by genuine event identifier with the APP sequence called out. Show probability, action, reason, measured latency, member scores, disagreement, and trace/hash binding. Put truth in a closed-by-default `<details>` region.

Lay out graph nodes by generated causal layer. Node focus changes immediate-neighbor opacity over 220ms only for pointer interaction; keyboard focus updates immediately. The table repeats every node/edge relationship and is first on mobile. Unbound case effort reads `Evidence pending`.

- [ ] **Step 4: Run focused tests**

Run: `npm --prefix web test -- --run src/features/replay src/features/investigation`

- [ ] **Step 5: Commit replay and investigation**

```bash
git add web/src/features/replay web/src/features/investigation
git -c user.name="Dylan Moraes" -c user.email=dylanmoraesdljdd@gmail.com commit -m "feat: add replay and campaign investigation"
```

### Task 6: Build Defenses and Assurance

**Files:**
- Create: `web/src/features/defenses/Defenses.tsx`
- Create: `web/src/features/defenses/RecoveredMetricsTable.tsx`
- Create: `web/src/features/defenses/Defenses.test.tsx`
- Create: `web/src/features/assurance/Assurance.tsx`
- Create: `web/src/features/assurance/GateTable.tsx`
- Create: `web/src/features/assurance/TrustProof.tsx`
- Create: `web/src/features/assurance/HumanGate.tsx`
- Create: `web/src/features/assurance/Assurance.test.tsx`

**Interfaces:**
- Consumes: four-arm recovered metrics/readiness, portable hashes/flags, exact seed copy, and trust proof.
- Produces: honest comparison conclusion and non-mutating human-review gate.

- [ ] **Step 1: Write failing recovery-boundary and failure-visibility tests**

Assert the exact qualifier heads the recovered table; all four arms appear; `ensemble_with_graph` is labeled usable live arm; `full_sentinel` is `Policy refinement required`; exact false-decline/challenge values and `benign_only` failure are visible; official chain incomplete, Stage 70 missing, and `not_ready` appear; both non-authoritative flags appear; Human Gate never emits an approved state.

- [ ] **Step 2: Verify RED**

Run: `npm --prefix web test -- --run src/features/defenses src/features/assurance`

- [ ] **Step 3: Implement the recovered comparison and assurance ledger**

Use a real HTML table with aligned metric rows and six-decimal formatting from the evidence document. Present the conclusion exactly: `The graph ensemble is the currently usable competition model. Deterministic full-hybrid routing requires policy refinement.`

Render every recovered gate with pass/fail text and target. Bind hashes through `HashRow`. Trust proof checks identity, mandate, scope, binding, and replay in a separate region. Human Gate remains `Human review required`; acknowledgement reveals review responsibilities but cannot promote.

- [ ] **Step 4: Run focused tests**

Run: `npm --prefix web test -- --run src/features/defenses src/features/assurance`

- [ ] **Step 5: Commit defenses and assurance**

```bash
git add web/src/features/defenses web/src/features/assurance
git -c user.name="Dylan Moraes" -c user.email=dylanmoraesdljdd@gmail.com commit -m "feat: add recovered comparison and assurance"
```

### Task 7: Add the scorer server, fixed fallback, health check, and reset

**Files:**
- Create: `tests/prototype/test_console_server.py`
- Create: `scripts/run_apar_console.py`
- Create: `scripts/check_apar_console.py`
- Create: `web/public/data/sentinel-v5-demo-trace.json`
- Modify: `web/src/app/evidence.ts`
- Modify: `web/src/app/App.tsx`

**Interfaces:**
- Produces: `/api/v1/health`, `/api/v1/replay`, `/api/v1/reset`, static serving, and CLI `--host`, `--port`, `--check`, `--force-fallback`.

- [ ] **Step 1: Write failing live, fallback, and fail-closed tests**

Test that successful scoring returns `source=live`, worker import/load failure returns `source=fallback` only after validating fallback flags/hash, tampered fallback returns HTTP 503 with no model values, and reset returns canonical event ID and `/overview` path.

- [ ] **Step 2: Generate the committed fallback with the existing scorer**

Run:

```bash
.venv/bin/python scripts/run_sentinel_v5_demo.py \
  --scenario demo/sentinel-v5/scenarios.json \
  --output web/public/data/sentinel-v5-demo-trace.json
```

Do not edit the JSON after generation.

- [ ] **Step 3: Verify RED**

Run: `python -m pytest tests/prototype/test_console_server.py -q`

- [ ] **Step 4: Implement the standard-library server and preflight**

Validate self-hashes before listening. Build `web/dist` on startup only when absent or stale. Never make an external request. Use explicit cache headers (`no-store` for API; immutable for hashed build assets). `--force-fallback` is test/demo-only and names the degraded cause.

- [ ] **Step 5: Run server tests and health checks**

Run: `python -m pytest tests/prototype/test_console_server.py -q`

Run: `.venv/bin/python scripts/check_apar_console.py`

Run: `.venv/bin/python scripts/run_apar_console.py --check`

- [ ] **Step 6: Commit reliability tooling**

```bash
git add tests/prototype scripts/run_apar_console.py scripts/check_apar_console.py web/public/data/sentinel-v5-demo-trace.json web/src/app
git -c user.name="Dylan Moraes" -c user.email=dylanmoraesdljdd@gmail.com commit -m "feat: add verified offline console runtime"
```

### Task 8: Add golden-path, accessibility, responsive, and offline coverage

**Files:**
- Create: `web/e2e/golden-path.spec.ts`
- Create: `web/e2e/accessibility.spec.ts`
- Create: `web/e2e/responsive.spec.ts`
- Create: `web/e2e/offline-fallback.spec.ts`

**Interfaces:**
- Consumes: root console launcher.
- Produces: deterministic judge-path and visual verification.

- [ ] **Step 1: Write the failing end-to-end path**

Navigate Overview → Scenario → Replay → Investigation → Defenses → Assurance by accessible names. Assert the portable arm identity, one APP decision, graph/table evidence, recovered qualifier, all `full_sentinel` failures, incomplete Stage 70 chain, TrustVerifier checks, and Human review required.

- [ ] **Step 2: Verify RED**

Run: `npm --prefix web run e2e -- golden-path.spec.ts`

- [ ] **Step 3: Add keyboard, reduced-motion, viewport, zoom, and network-blocking cases**

Cover widths 360, 768, 1024, and 1440; 200 percent zoom; no document-level horizontal overflow; skip link and focus visibility; all routes via keyboard; `prefers-reduced-motion`; external network abort; and forced-fallback completion.

- [ ] **Step 4: Run the complete Playwright suite**

Run: `npm --prefix web run e2e`

- [ ] **Step 5: Commit browser verification**

```bash
git add web/e2e web/playwright.config.ts
git -c user.name="Dylan Moraes" -c user.email=dylanmoraesdljdd@gmail.com commit -m "test: cover offline judge walkthrough"
```

### Task 9: Document, capture, audit, and verify the deliverable

**Files:**
- Modify: `README.md`
- Create: `submission/APAR_CONSOLE_WALKTHROUGH.md`
- Create: `submission/APAR_CONSOLE_GAPS.md`
- Create: `submission/screenshots/apar-console-desktop.png`
- Create: `submission/screenshots/apar-console-mobile.png`
- Create under `animation-plans/` only if the read-only motion audit finds actionable defects.

**Interfaces:**
- Produces: exact install/run/offline/test commands, five-minute Dylan narration, screenshot assets, and explicit remaining gaps.

- [ ] **Step 1: Write documentation assertions**

Add a small Python test that checks the README contains the exact install, start, health, force-fallback, reset, build, and test commands and that walkthrough wording matches `copy_boundary`.

- [ ] **Step 2: Write README, walkthrough, and gaps**

The walkthrough uses six timed segments totaling no more than five minutes. It uses the exact recovered qualifier and attempt wording. The gaps file names official Stage 70/80 incompleteness, non-authoritative readiness, full-hybrid policy failures, and absent human promotion decision.

- [ ] **Step 3: Capture deterministic screenshots**

Use Playwright at 1440×1000 on Defenses and 390×844 on Replay. Store only the two requested PNGs under `submission/screenshots/`.

- [ ] **Step 4: Run the read-only motion audit**

Audit `transition`, `animation`, `@keyframes`, `prefers-reduced-motion`, hover transforms, keyboard-triggered motion, easing, duration, and animated properties. If defects exist, write precise plans under `animation-plans/`, implement them through the normal test-first workflow, and re-audit. If none exist, record that no correction plan was necessary in the gaps/verification notes.

- [ ] **Step 5: Run fresh full verification**

```bash
npm --prefix web run typecheck
npm --prefix web run lint
npm --prefix web test -- --run
npm --prefix web run build
npm --prefix web run e2e
.venv/bin/python -m pytest tests/prototype tests/demo/test_sentinel_v5_portable.py tests/evaluation/test_defense_v5_kaggle_rescue.py tests/evaluation/test_defense_v5_kaggle_rescue_verifier.py -q
.venv/bin/python scripts/run_sentinel_v5_demo.py --scenario demo/sentinel-v5/scenarios.json --output /tmp/apar-console-final-trace.json
.venv/bin/python scripts/check_apar_console.py
git diff --check
```

Scan tracked prototype files for credential patterns, external runtime URLs, `/Users/`, prohibited claim wording, and unqualified recovered metrics. Inspect screenshots at full resolution.

- [ ] **Step 6: Commit final materials and verify identity/cleanliness**

```bash
git add README.md submission web scripts tests docs animation-plans
git -c user.name="Dylan Moraes" -c user.email=dylanmoraesdljdd@gmail.com commit -m "feat: complete competition assurance console"
git status --short --branch
git log -1 --format=fuller
```

Expected: clean worktree and both Author/Commit fields are Dylan Moraes `<dylanmoraesdljdd@gmail.com>`.
