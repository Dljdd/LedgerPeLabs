# Prototype, Demo, and Submission Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deliver a polished offline-capable six-view web prototype, a deterministic five-minute judging walkthrough, and a reproducible competition submission archive.

**Architecture:** A React client consumes the localhost FastAPI boundary and renders immutable threat, run, evaluation, and report artifacts. One deterministic golden path is bundled for offline judging; live recomputation is available but never required for the core demonstration.

**Tech Stack:** React 19, TypeScript 5, Vite, TanStack Query, React Router, Vitest, React Testing Library, Playwright, FastAPI, Python 3.12

**Spec:** `SOLUTION_SPEC.md`, `docs/01-product-requirements.md`, `docs/09-prototype-demo-and-submission.md`, `docs/10-delivery-roadmap.md`, `AGENTS.md`

## Global Constraints

- Implement exactly six primary views: Threat Registry, Scenario Builder, Campaign Replay, Defense Comparison, Campaign Investigation, and Assurance Report.
- Complete the golden path in under five minutes on a clean machine without external network access.
- Keep all content visible by default; never gate text or controls behind entrance animation.
- Use a restrained warm-white, ink, Mastercard-red, and measured amber palette; do not use purple, blue-to-purple gradients, background glow, or gradient-filled text.
- Do not use hero pills, glowy buttons, floating cards, fake macOS windows, full-page grid backgrounds, icon tiles, decorative nav dots, or all-around shadows.
- Use real tables and graphs for real data. Do not use crude decorative charts or fake metrics.
- Use bare icons only when their meaning is established; do not place icons or logos in filled boxes.
- All text must meet WCAG 2.2 AA contrast and remain fully inside clipped or fixed-height regions.
- Corresponding rows, controls, and action areas in comparison layouts must share horizontal alignment.
- Support keyboard navigation, visible focus, reduced motion, 200 percent zoom, and widths from 360 to 1440 pixels.
- Do not include secrets, private data, operational attack recipes, or unsupported product claims in the bundle.

---

## Target file map

```text
web/package.json                          Frontend scripts and dependencies
web/vite.config.ts                       Vite and test configuration
web/src/app/App.tsx                      Router and application shell
web/src/app/api.ts                       Typed localhost API client
web/src/app/query.ts                     Query client and error policy
web/src/styles/tokens.css                Approved color, type, spacing, and focus tokens
web/src/styles/global.css                Reset, layout, tables, forms, and responsive rules
web/src/components/                      Focused reusable data components
web/src/views/ThreatRegistry.tsx         Evidence and coverage
web/src/views/ScenarioBuilder.tsx        Bounded scenario compilation
web/src/views/CampaignReplay.tsx         Event and lifecycle replay
web/src/views/DefenseComparison.tsx      Baseline and challenger comparison
web/src/views/CampaignInvestigation.tsx  Entity graph and case queue
web/src/views/AssuranceReport.tsx         Gates and human promotion
web/src/fixtures/golden-report.json       Offline deterministic artifact
web/tests/                               Component and interaction tests
web/e2e/                                 Golden-path and responsive Playwright tests
scripts/dev.py                           One-command local startup
scripts/build_demo_fixture.py            Rebuild offline fixture from artifacts
scripts/record_demo.mjs                   Deterministic fallback recording
scripts/package_submission.py            Allowlisted archive builder
submission/WALKTHROUGH.md                Five-minute narration and recovery path
submission/demo-fallback.mp4              Screen recording used only if live demo fails
submission/MANIFEST.md                   Included files, versions, digests, and licenses
```

### Task 1: Bootstrap the strict React client and typed API boundary

**Files:**
- Create: `web/package.json`
- Create: `web/tsconfig.json`
- Create: `web/vite.config.ts`
- Create: `web/index.html`
- Create: `web/src/main.tsx`
- Create: `web/src/app/App.tsx`
- Create: `web/src/app/api.ts`
- Create: `web/src/app/query.ts`
- Create: `web/tests/app.test.tsx`

**Interfaces:**
- Produces: `ApiClient` methods for health, threats, scenario compilation, runs, evaluations, reports, and promotion
- Produces routes: `/threats`, `/scenarios`, `/runs/:runId/replay`, `/runs/:runId/compare`, `/runs/:runId/investigate`, `/reports/:reportId`

- [ ] **Step 1: Write routing and API error tests**

```tsx
it("redirects the root to the threat registry", async () => {
  render(<App initialEntries={["/"]} />);
  expect(await screen.findByRole("heading", { name: "Threat registry" })).toBeVisible();
});

it("maps a typed API error without losing its code", async () => {
  mockFetch(409, { detail: { code: "FAILED_PROMOTION_GATE", message: "Hidden evaluation failed" } });
  await expect(api.promote("r1", { approverId: "judge" })).rejects.toMatchObject({
    code: "FAILED_PROMOTION_GATE",
  });
});
```

- [ ] **Step 2: Run frontend tests and confirm the app is absent**

Run: `npm --prefix web test -- --run`

Expected: the command fails because `web/package.json` is absent.

- [ ] **Step 3: Add strict TypeScript, router, query client, and API types**

Use `strict: true`, `noUncheckedIndexedAccess: true`, and `exactOptionalPropertyTypes: true`. The client base URL defaults to `http://127.0.0.1:8000/api/v1`. React-facing types use camelCase; `api.ts` performs the explicit snake_case mapping required by Pydantic requests and responses. Every API method validates response shape through explicit TypeScript narrowing and throws `ApiError { code, message, status }` for non-2xx responses.

- [ ] **Step 4: Run type, lint, and test gates**

Run: `npm --prefix web run typecheck`

Run: `npm --prefix web run lint`

Run: `npm --prefix web test -- --run`

Expected: all commands exit zero.

- [ ] **Step 5: Commit the frontend boundary**

```bash
git add web
git commit -m "build: bootstrap strict APAR web client"
```

### Task 2: Implement the application shell and design system

**Files:**
- Create: `web/src/styles/tokens.css`
- Create: `web/src/styles/global.css`
- Create: `web/src/components/AppHeader.tsx`
- Create: `web/src/components/PrimaryNav.tsx`
- Create: `web/src/components/StatusText.tsx`
- Create: `web/src/components/DataTable.tsx`
- Create: `web/src/components/Metric.tsx`
- Create: `web/tests/design-system.test.tsx`
- Create: `web/e2e/shell.spec.ts`

**Interfaces:**
- Produces semantic tokens: `--surface`, `--surface-raised`, `--ink`, `--ink-muted`, `--brand-red`, `--signal-amber`, `--success`, `--danger`, `--focus`
- Produces accessible header, navigation, status text, data table, and metric primitives

- [ ] **Step 1: Write semantic, keyboard, and visibility tests**

```tsx
it("uses text and aria-current for the active page", () => {
  render(<PrimaryNav currentPath="/threats" />);
  expect(screen.getByRole("link", { name: "Threats" })).toHaveAttribute("aria-current", "page");
});

it("renders content without animation state classes", () => {
  const { container } = render(<Metric label="Value saved" value="$82,400" />);
  expect(container.querySelector('[style*="opacity: 0"]')).toBeNull();
  expect(screen.getByText("$82,400")).toBeVisible();
});
```

- [ ] **Step 2: Confirm component and shell tests fail**

Run: `npm --prefix web test -- --run web/tests/design-system.test.tsx`

Expected: imports fail because the components do not exist.

- [ ] **Step 3: Implement the visual system from explicit tokens**

Use a neutral system sans stack, 16-pixel root size, 1.5 body line height, 44-pixel minimum control height, 3-pixel visible focus outline, square-to-8-pixel corner radii, no gradient, and no resting card shadow. Separate sections through spacing and surface tone, not hairline borders around every block. Use CSS grid only where content has a true comparison relationship.

- [ ] **Step 4: Run design-law and responsive shell checks**

Run: `npm --prefix web test -- --run web/tests/design-system.test.tsx`

Run: `npm --prefix web run e2e -- shell.spec.ts`

Expected: keyboard navigation, active-state type treatment, visible content, 360/768/1440 layouts, 200 percent zoom, focus visibility, and reduced-motion checks pass.

- [ ] **Step 5: Commit the non-generic application shell**

```bash
git add web/src/styles web/src/components web/tests/design-system.test.tsx web/e2e/shell.spec.ts
git commit -m "feat: add accessible APAR application shell"
```

### Task 3: Build Threat Registry and Scenario Builder views

**Files:**
- Create: `web/src/views/ThreatRegistry.tsx`
- Create: `web/src/views/ScenarioBuilder.tsx`
- Create: `web/src/components/EvidenceList.tsx`
- Create: `web/src/components/CoverageMatrix.tsx`
- Create: `web/src/components/ParameterField.tsx`
- Create: `web/tests/threat-registry.test.tsx`
- Create: `web/tests/scenario-builder.test.tsx`

**Interfaces:**
- Consumes: threat-card and compiler endpoints
- Produces: selected `threatId`, bounded `ScenarioConfig`, compiled `scenarioArtifactRef`
- The builder exposes only parameters allowed by the selected threat card

- [ ] **Step 1: Write evidence and bounded-input tests**

```tsx
it("shows provenance and separates fact from inference", async () => {
  render(<ThreatRegistry />);
  expect(await screen.findByText("Primary source")).toBeVisible();
  expect(screen.getByText("Project inference")).toBeVisible();
  expect(screen.getByRole("link", { name: /open source/i })).toHaveAttribute("href", expect.stringMatching(/^https:/));
});

it("prevents values outside the compiled parameter bounds", async () => {
  render(<ScenarioBuilder threatId="t1" />);
  const input = await screen.findByLabelText("Illicit entities");
  await userEvent.clear(input);
  await userEvent.type(input, "10000");
  expect(screen.getByText("Maximum 200")).toBeVisible();
  expect(screen.getByRole("button", { name: "Compile scenario" })).toBeDisabled();
});
```

- [ ] **Step 2: Run view tests and confirm missing views**

Run: `npm --prefix web test -- --run web/tests/threat-registry.test.tsx web/tests/scenario-builder.test.tsx`

Expected: imports fail because the views do not exist.

- [ ] **Step 3: Implement source-first registry and constrained compilation**

The registry table shows family, GenAI capability delta, affected rails, evidence confidence, implementation depth, and safety class. The detail region lists direct URLs, dates, fact statements, and project inferences. The builder renders numeric, enum, boolean, and duration controls from the server contract, displays compiler errors by stable code, and never accepts hidden or undeclared fields.

- [ ] **Step 4: Run interaction and accessibility tests**

Run: `npm --prefix web test -- --run web/tests/threat-registry.test.tsx web/tests/scenario-builder.test.tsx`

Expected: source links, filters, keyboard row selection, bound enforcement, compiler rejection, safe descriptions, and compiled-artifact navigation tests pass.

- [ ] **Step 5: Commit the Identify and Generate views**

```bash
git add web/src/views/ThreatRegistry.tsx web/src/views/ScenarioBuilder.tsx web/src/components/EvidenceList.tsx web/src/components/CoverageMatrix.tsx web/src/components/ParameterField.tsx web/tests
git commit -m "feat: add threat registry and scenario builder"
```

### Task 4: Build Campaign Replay and Defense Comparison views

**Files:**
- Create: `web/src/views/CampaignReplay.tsx`
- Create: `web/src/views/DefenseComparison.tsx`
- Create: `web/src/components/LifecycleTimeline.tsx`
- Create: `web/src/components/DecisionInspector.tsx`
- Create: `web/src/components/ComparisonTable.tsx`
- Create: `web/src/components/ActionFrontier.tsx`
- Create: `web/tests/campaign-replay.test.tsx`
- Create: `web/tests/defense-comparison.test.tsx`

**Interfaces:**
- Consumes: event, feedback, decision, defender, and metric artifacts
- Produces: synchronized selected event, lifecycle state, model decision, and reason-code inspection
- Comparison columns: rules, strongest GBDT champion, adaptive-robust challenger

- [ ] **Step 1: Write synchronization and aligned-comparison tests**

```tsx
it("keeps event, lifecycle, and decision inspection synchronized", async () => {
  render(<CampaignReplay runId="golden" />);
  await userEvent.click(await screen.findByRole("row", { name: /event e-17/i }));
  expect(screen.getByText("Transfer posted")).toBeVisible();
  expect(screen.getByText("Reason: beneficiary velocity")).toBeVisible();
});

it("renders the same metric rows for every defender", async () => {
  render(<DefenseComparison runId="golden" />);
  const table = await screen.findByRole("table", { name: "Defender comparison" });
  expect(within(table).getAllByRole("row")).toHaveLength(9);
  expect(within(table).getAllByRole("columnheader")).toHaveLength(4);
});
```

- [ ] **Step 2: Confirm replay and comparison tests fail**

Run: `npm --prefix web test -- --run web/tests/campaign-replay.test.tsx web/tests/defense-comparison.test.tsx`

Expected: imports fail because replay and comparison components are missing.

- [ ] **Step 3: Implement data-driven lifecycle and aligned metrics**

Render the lifecycle as a semantic ordered list with a compact SVG connector generated from real event positions. Show event time, arrival time, and decision time explicitly. Use a true HTML table for defender comparison so metric labels, values, and action budgets share rows. The action frontier plots real false-positive budget against preventable value and provides the same data in a table.

- [ ] **Step 4: Run event, table, and no-data tests**

Run: `npm --prefix web test -- --run web/tests/campaign-replay.test.tsx web/tests/defense-comparison.test.tsx`

Expected: timestamp synchronization, late arrival, equal-time grouping, reason-code order, aligned rows, keyboard selection, empty state, and reduced-motion tests pass.

- [ ] **Step 5: Commit the Defend comparison workflow**

```bash
git add web/src/views/CampaignReplay.tsx web/src/views/DefenseComparison.tsx web/src/components/LifecycleTimeline.tsx web/src/components/DecisionInspector.tsx web/src/components/ComparisonTable.tsx web/src/components/ActionFrontier.tsx web/tests
git commit -m "feat: add campaign replay and defense comparison"
```

### Task 5: Build Campaign Investigation and Assurance Report views

**Files:**
- Create: `web/src/views/CampaignInvestigation.tsx`
- Create: `web/src/views/AssuranceReport.tsx`
- Create: `web/src/components/EntityGraph.tsx`
- Create: `web/src/components/CaseQueue.tsx`
- Create: `web/src/components/GateTable.tsx`
- Create: `web/src/components/PromotionForm.tsx`
- Create: `web/tests/campaign-investigation.test.tsx`
- Create: `web/tests/assurance-report.test.tsx`

**Interfaces:**
- Consumes: case, entity graph, gate, provenance, approval, and rollback artifacts
- Produces: selected case and entity neighborhood
- Produces: explicit approve or reject action owned by a named human

- [ ] **Step 1: Write graph fallback and promotion-control tests**

```tsx
it("provides a tabular graph alternative", async () => {
  render(<CampaignInvestigation runId="golden" />);
  expect(await screen.findByRole("img", { name: "Campaign entity graph" })).toBeVisible();
  expect(screen.getByRole("table", { name: "Campaign relationships" })).toBeVisible();
});

it("cannot promote when a hard gate fails", async () => {
  render(<AssuranceReport reportId="failed-hidden" />);
  expect(await screen.findByText("Hidden generator: failed")).toBeVisible();
  expect(screen.getByRole("button", { name: "Approve promotion" })).toBeDisabled();
});
```

- [ ] **Step 2: Confirm investigation and assurance tests fail**

Run: `npm --prefix web test -- --run web/tests/campaign-investigation.test.tsx web/tests/assurance-report.test.tsx`

Expected: imports fail because the views do not exist.

- [ ] **Step 3: Implement evidence-linked investigation and human approval**

Draw the entity graph from typed nodes and edges with deterministic layout and no decorative animation. Include a synchronized relationship table for accessibility. The report shows primary metrics, family minimums, operational budgets, leakage checks, model and generator digests, source artifacts, rollback artifact, and gate reasons. The promotion form requires approver ID, decision, and confirmation text; it never changes failed gate state.

- [ ] **Step 4: Run investigation, provenance, and gate tests**

Run: `npm --prefix web test -- --run web/tests/campaign-investigation.test.tsx web/tests/assurance-report.test.tsx`

Expected: case ranking, graph/table synchronization, entity keyboard focus, provenance links, failed-gate veto, approval confirmation, rejection path, and immutable-report tests pass.

- [ ] **Step 5: Commit the investigation and assurance workflow**

```bash
git add web/src/views/CampaignInvestigation.tsx web/src/views/AssuranceReport.tsx web/src/components/EntityGraph.tsx web/src/components/CaseQueue.tsx web/src/components/GateTable.tsx web/src/components/PromotionForm.tsx web/tests
git commit -m "feat: add campaign investigation and assurance report"
```

### Task 6: Bundle the deterministic golden path and one-command startup

**Files:**
- Create: `scripts/dev.py`
- Create: `scripts/build_demo_fixture.py`
- Create: `scripts/record_demo.mjs`
- Create: `web/src/fixtures/golden-report.json`
- Create: `web/src/app/fixtureApi.ts`
- Create: `web/e2e/golden-path.spec.ts`
- Create: `submission/demo-fallback.mp4`
- Modify: `README.md`

**Interfaces:**
- Produces: `python scripts/dev.py` starting API on `127.0.0.1:8000` and Vite on `127.0.0.1:5173`
- Produces: fixture mode selected by `VITE_DEMO_SOURCE=fixture`
- Produces a complete Identify, Generate, Replay, Defend, Investigate, Assure walkthrough

- [ ] **Step 1: Write the five-minute golden-path test**

```ts
test("judge completes the complete golden path", async ({ page }) => {
  await page.goto("/threats");
  await page.getByRole("row", { name: /AI-personalized APP scam/i }).click();
  await page.getByRole("link", { name: "Build scenario" }).click();
  await page.getByRole("button", { name: "Compile scenario" }).click();
  await page.getByRole("link", { name: "Replay campaign" }).click();
  await page.getByRole("link", { name: "Compare defenses" }).click();
  await page.getByRole("link", { name: "Investigate campaign" }).click();
  await page.getByRole("link", { name: "Open assurance report" }).click();
  await expect(page.getByText("Human promotion required")).toBeVisible();
});
```

- [ ] **Step 2: Run the path before fixture and startup scripts exist**

Run: `npm --prefix web run e2e -- golden-path.spec.ts`

Expected: startup fails because fixture mode and scripts are absent.

- [ ] **Step 3: Build the fixture from immutable G3 artifacts**

The fixture builder reads one approved scenario artifact, fixed and adaptive run manifests, three defender evaluation bundles, reconstructed cases, and one unapproved assurance report. It emits stable IDs, strips restricted hidden-validity reasons and internal file paths, verifies no source timestamp violates decision time, and writes canonical JSON plus SHA-256. `record_demo.mjs` drives the same Playwright actions at 1440 by 900 pixels, records narration-free video, and writes `submission/demo-fallback.mp4` with the fixture digest in its adjacent manifest entry.

- [ ] **Step 4: Run startup, offline, and timing tests**

Run: `python scripts/dev.py --fixture --check`

Expected output: `API ready`, `web ready`, and `fixture digest verified`.

Run: `npm --prefix web run e2e -- golden-path.spec.ts`

Expected: golden path passes in Chromium with network requests blocked and completes in under 300 seconds.

Run: `node scripts/record_demo.mjs --verify`

Expected: the MP4 exists, follows the exact golden-path route order, lasts no more than five minutes, and its SHA-256 is recorded in `submission/MANIFEST.md`.

- [ ] **Step 5: Commit the G4 golden path**

```bash
git add scripts/dev.py scripts/build_demo_fixture.py scripts/record_demo.mjs web/src/fixtures web/src/app/fixtureApi.ts web/e2e/golden-path.spec.ts submission/demo-fallback.mp4 submission/MANIFEST.md README.md
git commit -m "test: establish offline five minute golden path"
```

### Task 7: Perform full design-law, accessibility, and responsive QA

**Files:**
- Create: `web/e2e/accessibility.spec.ts`
- Create: `web/e2e/responsive.spec.ts`
- Create: `web/e2e/visual.spec.ts`
- Create: `docs/qa/design-law-audit.md`
- Modify: affected `web/src/**/*.tsx` and `web/src/styles/*.css` files when the audit identifies a failure

**Interfaces:**
- Produces: an explicit pass/fail record for every workspace anti-slop rule applicable to the product
- Produces: Playwright screenshots at 360, 768, 1024, and 1440 pixels

- [ ] **Step 1: Add automated accessibility and clipping assertions**

```ts
for (const width of [360, 768, 1024, 1440]) {
  test(`no horizontal clipping at ${width}`, async ({ page }) => {
    await page.setViewportSize({ width, height: 900 });
    await page.goto("/reports/golden");
    const overflow = await page.evaluate(() => document.documentElement.scrollWidth - document.documentElement.clientWidth);
    expect(overflow).toBe(0);
  });
}
```

- [ ] **Step 2: Run QA to expose current defects**

Run: `npm --prefix web run e2e -- accessibility.spec.ts responsive.spec.ts visual.spec.ts`

Expected: any contrast, name, focus, clipping, alignment, or overflow defect fails with a saved screenshot.

- [ ] **Step 3: Audit every design-law item and fix each applicable failure**

Record each rule in `docs/qa/design-law-audit.md` as `Pass`, `Not applicable`, or `Fixed`, with the affected route and screenshot name. Inspect content visibility, gradients, glows, pill use, icon containers, fake windows, grid backgrounds, card shadows, hover lifts, clipped content, comparison alignment, text gutters, centering, logos, section color continuity, contrast, navigation state, and responsive behavior. Fix each failed item in the component or CSS file that owns it.

- [ ] **Step 4: Rerun the complete frontend verification**

Run: `npm --prefix web run typecheck && npm --prefix web run lint && npm --prefix web test -- --run && npm --prefix web run e2e`

Expected: every command exits zero, every page is visible without JavaScript animation completion, and screenshots show no clipping or misalignment.

- [ ] **Step 5: Commit the completed visual QA gate**

```bash
git add web docs/qa/design-law-audit.md
git commit -m "test: complete accessibility and design law audit"
```

### Task 8: Package the reproducible competition submission

**Files:**
- Create: `scripts/package_submission.py`
- Create: `submission/WALKTHROUGH.md`
- Create: `submission/MANIFEST.md`
- Create: `submission/THREAT_MODEL.md`
- Create: `submission/PRIVACY.md`
- Create: `submission/LICENSES.md`
- Create: `submission/SBOM.spdx.json`
- Create: `tests/submission/test_package.py`
- Modify: `README.md`

**Interfaces:**
- Produces: `dist/apar-mastercard-innovation-2026.zip`
- Archive includes source, fixture, documentation, walkthrough, licenses, lockfiles, and checksums
- Archive excludes `.git`, `.apar`, caches, raw hidden reasons, environment files, tokens, and local absolute paths

- [ ] **Step 1: Write allowlist, secret, and reproducibility tests**

```python
def test_archive_contains_only_allowlisted_roots(built_archive) -> None:
    roots = {name.split("/", 1)[0] for name in built_archive.names}
    assert roots <= {"src", "web", "scripts", "fixtures", "docs", "submission", "README.md", "pyproject.toml"}


def test_two_builds_have_identical_digest(build_submission) -> None:
    first = build_submission()
    second = build_submission()
    assert first.sha256 == second.sha256
```

- [ ] **Step 2: Confirm package tests fail before the builder exists**

Run: `python -m pytest tests/submission/test_package.py -q`

Expected: collection fails because the submission builder is absent.

- [ ] **Step 3: Implement deterministic archive construction and documents**

Walk the explicit allowlist, reject symlinks, normalize archive timestamps to `2026-08-16T00:00:00Z`, sort entries, and use fixed compression settings. Scan text for credential patterns, private keys, local home paths, forbidden operational attack terms, and unapproved external URLs. Generate `submission/SBOM.spdx.json` from the locked Python and npm dependency graphs, fail on missing or prohibited licenses, and generate the manifest with file digests, Python and Node versions, dependency licenses, startup command, fixture digest, fallback-video digest, test commands, and known limitations.

- [ ] **Step 4: Execute the complete G5 release gate**

Run: `python scripts/package_submission.py --verify`

Expected output ends with the archive path, SHA-256, file count, `privacy PASS`, `licenses PASS`, `secrets PASS`, `provenance PASS`, and `reproducibility PASS`.

Run: `python -m pytest -q && npm --prefix web test -- --run && npm --prefix web run e2e`

Expected: every backend, frontend, and end-to-end test passes.

- [ ] **Step 5: Commit the competition-ready release tooling**

```bash
git add scripts/package_submission.py submission tests/submission README.md
git commit -m "build: package reproducible competition submission"
```

## Plan completion gate

G4 and G5 are complete when a clean machine starts the fixture-backed product with one command, a judge completes all six views in under five minutes without network access, every applicable design-law and accessibility check passes, and two independently built submission archives have identical SHA-256 digests with no secrets, private data, hidden oracle details, or non-allowlisted files.
