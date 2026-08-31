# APAR console verification record

Verification date: 2026-08-31

## Implemented surface

The console implements six native routes: Overview, Scenario, Replay,
Investigation, Defenses, and Assurance. It uses committed local JSON only,
calls the real Python portable scorer when available, and retains a labeled,
hash-bound fixed trace for degraded operation.

The finalized Editorial Casefile treatment uses a serif narrative hierarchy,
warm ink surfaces, and graph-led replay. The 10 scenario payment edges and 12
portable decisions remain independently selectable, with an explicit statement
that no payment-to-trace record mapping is asserted.

## Representative screenshots

- `docs/demo/screenshots/apar-console-overview-desktop.png`
- `docs/demo/screenshots/apar-console-replay-desktop.png`
- `docs/demo/screenshots/apar-console-investigation-desktop.png`
- `docs/demo/screenshots/apar-console-overview-mobile.png`
- `docs/demo/screenshots/apar-console-assurance-mobile.png`

## Verified checks

- Console evidence builder: 5 tests passed.
- Console server and live scorer worker: 3 tests passed.
- Frontend unit and boundary checks: 11 tests passed.
- TypeScript project check: passed.
- ESLint with zero warnings: passed.
- Vite production build: passed.
- Playwright desktop/mobile suite: 8 tests passed.
- Playwright route smoke: all six routes, both breakpoints.
- Axe automated checks: no violations on Overview and Replay, desktop/mobile.
- Keyboard path: skip link, primary navigation, and replay step passed.
- Viewport check: no document-level horizontal overflow on any route at tested
  desktop/mobile breakpoints.
- Fixed trace hash, bundle binding, accepted action/probability replay, and real
  scorer path: passed.
- Motion audit: passed; see `animation-plans/apar-console-motion-audit.md`.
- Design Lab and preview routes: removed after final approval.

The exact commands are maintained in `web/README.md`. These check results are
engineering verification of the prototype, not new model-performance evidence.

## Honest remaining submission gaps

- Dylan has not recorded the walkthrough video; this task intentionally stops
  before video production.
- The official evidence chain remains incomplete at `70_metrics`; recovered
  diagnostics remain non-authoritative and not accepted capacity evidence.
- The deterministic `full_sentinel` routing policy still fails false-decline,
  challenge-rate, and benign-only gates and requires refinement.
- Browser automation currently covers Chromium desktop and mobile profiles;
  Firefox, WebKit, and physical assistive-technology testing remain pending.
- Axe is focused on Overview and Replay. A manual WCAG 2.2 AA audit of every
  surface, zoom mode, and assistive-technology combination remains pending.
- The prototype is a local competition console, not a hardened production
  deployment. Load testing, deployment threat modeling, operational monitoring,
  and disaster-recovery validation are not complete.
- Live and fixed-trace latency values are environment-specific and are not
  production service-level evidence.
- Analyst-time benefit has no bound evidence and remains visibly pending.
- Submission archive allowlisting, SBOM/license packaging, and independent
  two-person archive review remain to be completed.
