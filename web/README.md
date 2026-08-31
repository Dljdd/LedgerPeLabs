# APAR competition assurance console

This directory contains the judge-facing React and TypeScript prototype. It is
an institutional payment-risk assurance console, not a production decisioning
service. The live portable model is the accepted Stage 30
`ensemble_with_graph` arm. It must not be described as `full_sentinel` or as a
complete hybrid.

The finalized interface uses an Editorial Casefile design system: serif
narrative headings, warm ink surfaces, monospaced evidence, and a graph-led
Replay route. The scenario graph and portable trace have independent selectors;
the console does not assert a payment-to-trace record mapping.

## Prerequisites

- Python 3.12 with the repository installed in `.venv`
- Node.js 20.19 or newer
- npm 10 or newer

From a clean checkout, install once while package registries are available:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -e '.[dev]'
npm ci --prefix web
```

There are no remote fonts, analytics, telemetry, credentials, or runtime CDN
requests. After installation, all commands below work with external networking
disabled.

## Start from the repository root

```bash
.venv/bin/python scripts/run_apar_console.py start
```

Open `http://127.0.0.1:4173/overview`. The command performs the evidence
preflight, builds the frontend, serves all six native routes, and exposes the
local portable scorer at `/api/score`.

The browser first calls the real local scorer. If it is unavailable, the UI
fails over to `web/public/data/verified-trace.json`, labels it
“Hash-bound verified fallback,” and checks it against the accepted scenario
rows and bundle manifest before rendering model values.

## Preflight, reset, and fallback drill

```bash
.venv/bin/python scripts/run_apar_console.py health
.venv/bin/python scripts/run_apar_console.py reset
.venv/bin/python scripts/run_apar_console.py start --fallback-only
```

`health` verifies both embedded document hashes, bundle binding, all 12
accepted event/action/probability pairs, `authoritative=false`, and
`accepted_capacity_evidence=false`. `reset` removes only the generated local
live trace; reload or the Replay reset button returns the interface to event 1.
`--fallback-only` is the deterministic worker-failure drill.

## Focused verification

Run these commands from the repository root unless noted:

```bash
.venv/bin/python -m pytest tests/prototype -q
.venv/bin/ruff check scripts/run_apar_console.py scripts/build_apar_console_evidence.py tests/prototype
.venv/bin/mypy scripts/run_apar_console.py scripts/build_apar_console_evidence.py
npm run test --prefix web -- --run
npm run typecheck --prefix web
npm run lint --prefix web
npm run build --prefix web
APAR_PYTHON=../.venv/bin/python npm run e2e --prefix web
```

The Playwright command requires a locally installed Chromium browser but does
not require an external network. It covers desktop and mobile route smoke,
the golden path, viewport clipping, Axe checks, and keyboard operation.

Regenerate the committed representative screenshots while the console is
running:

```bash
npm run screenshots --prefix web
```

## Evidence boundaries

- The 12 cases are curated synthetic replay checks, not production estimates.
- The recovered four-arm diagnostics are labeled exactly “Recovered diagnostic
  evidence — non-authoritative.” The official chain is incomplete at
  `70_metrics`.
- `full_sentinel` remains a diagnostic architecture arm and fails its
  false-decline, challenge-rate, and benign-only gates.
- TrustVerifier identity, mandate, scope, binding, and replay checks are a
  separate deterministic integrity proof—not a graph-model performance claim.
- The campaign graph uses deterministic synthetic scenario seed `260816`. This
  is distinct from the portable demo and recovered Kaggle metric evidence,
  which use seed `404` only.

The five-minute judge script is in
[`docs/demo/APAR_CONSOLE_WALKTHROUGH.md`](../docs/demo/APAR_CONSOLE_WALKTHROUGH.md).
