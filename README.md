# Mastercard Innovation Challenge 2026

This repository specifies and validates an **Adaptive Payment Assurance Range** for testing payment-risk and agentic-commerce controls against emerging GenAI-enabled fraud.

The product is an assurance layer, not a replacement payment-decision engine. It turns sourced threats into constrained synthetic campaigns, evaluates champion and challenger defenses under hidden shifts, and produces evidence for a human promotion decision.

## Start here

- [Canonical solution specification](SOLUTION_SPEC.md)
- [Documentation index](docs/README.md)
- [Approved implementation plans](docs/superpowers/plans/README.md)
- [Diagram catalog](docs/diagrams/README.md)
- [Empirical validation spike](validation_spike/README.md)
- [Judge-facing APAR console](web/README.md)

## Run the judge-facing console

After installing the Python project and the frontend dependencies with
`npm ci --prefix web`, start the complete offline console from the repository
root:

```bash
.venv/bin/python scripts/run_apar_console.py start
```

Open `http://127.0.0.1:4173/overview`. The launcher verifies the committed
evidence and fallback trace, builds the client, and serves the real local
portable scorer. See [web/README.md](web/README.md) for preflight, reset,
fallback-only, test, and offline instructions.

## Install and verify the G0 foundation

The supported runtime is Python 3.12. The immutable artifact store is supported on
macOS (Darwin, using native `renameatx_np`) and Linux filesystems that provide native
`renameat2(..., RENAME_NOREPLACE)` support. Other platforms, including Windows, are
not currently supported by the artifact publication boundary.

From a clean checkout on a supported platform, use the virtual environment's
interpreter directly:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e '.[dev]'
```

Run the complete G0 contract verification with one command:

```bash
.venv/bin/python scripts/verify_g0.py
```

The command validates all 20 evidence-backed threat cards and exercises the real registry,
scenario compiler, API, and immutable artifact store from clean temporary state.

## Proposed product flow

`Threat evidence -> Threat registry -> Scenario compiler -> Stateful payment simulator -> Champion/challenger controls -> Operational policy -> Hidden evaluation -> Human promotion report`

## Repository status

| Area | Status |
|---|---|
| Product and architecture specification | Documented |
| Empirical falsification spike | Implemented |
| Threat registry and scenario compiler | G0 foundation implemented |
| Rail-specific simulator | Specified, not implemented |
| Adaptive red-team optimizer | Harness validated; adaptive optimizer not implemented |
| Defender service | Implemented; 200-campaign v1 evidence is signed and hash-pinned, with truthful `no_promotion` after the frozen 1% workload budget proved infeasible |
| Agentic trust plane | Specified, not implemented |
| Web prototype and walkthrough | Implemented and locally verified; walkthrough video not recorded |

The `validation_spike` is retained as supporting evidence. It must not be represented as the complete competition solution.

The Defend G3 implementation gate is available with `.venv/bin/python scripts/verify_g3.py`.
The full synthetic competition result is pinned in
`docs/experiments/defense-v1-result.json` and `fixtures/defense/v1/`. It contains
exactly 200 authenticated campaigns. The trained candidate reloads, but no
defender was frozen and hidden evaluation was not released: six mandatory review
cases over 336 threshold rows imply a 1.7857% minimum workload, above the
preregistered 1% cap. The budget was not relaxed and no champion is claimed.

Defend v2: protocol sealed; evaluation not executed. Any future result remains synthetic-only and is not a real-world prevalence or external-validity claim.

Defend v3: execution path drafted; evaluation not executed.

Defend v3 confirmatory attempt consumed on an incomplete scaffold; truthful `no_promotion` recorded. No competition evidence was produced. A v4 protocol revision is required.

Defend v4: execution path implemented; evaluation not executed.

Defend v4 confirmatory attempt consumed; truthful `no_promotion` recorded. Real frozen CatBoost/calibrator/rules were scored, but CALIBRATION and TIME_TO_ALERT gates always fail because those metrics are not yet computed (fail-closed None). No champion is claimed.

## Working title

**Adaptive Payment Assurance Range**  
Evidence-backed adversarial testing for card, account-to-account, and agentic-commerce controls.
