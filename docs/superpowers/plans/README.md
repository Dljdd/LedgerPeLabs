# Adaptive Payment Assurance Range implementation plans

The approved architecture is implemented through four independently reviewable plans. Execute them in this order:

1. [Foundation and contracts](2026-08-16-foundation-contracts.md)
2. [Stateful simulator and adaptive red-team](2026-08-16-simulator-redteam.md)
3. [Defense, evaluation, and governance](2026-08-16-defense-evaluation.md)
4. [Prototype, demo, and submission](2026-08-16-prototype-submission.md)

Use the [implementation traceability matrix](IMPLEMENTATION_TRACEABILITY.md) to map every approved specification area to its owning task and verification evidence.

## Locked technology choices

- Python 3.12 for contracts, simulation, scoring, evaluation, APIs, and reports.
- FastAPI and Pydantic v2 for the local application API and typed boundaries.
- SQLite for metadata and Parquet for immutable event, feature, and result artifacts.
- NumPy, pandas, scikit-learn, CatBoost, and NetworkX for baselines and graph features.
- React, TypeScript, and Vite for the six-view competition prototype.
- pytest, Hypothesis, Vitest, React Testing Library, and Playwright for verification.
- Ruff and mypy for Python quality gates; ESLint and TypeScript strict mode for the web client.

## Cross-plan invariants

- No decision may consume a source timestamp equal to or later than its decision timestamp.
- Payment lifecycles and balances must conserve value under declared fees and reversals.
- Campaign, entity, and scenario IDs must not cross evaluation partitions.
- The adaptive attacker receives only declared decision feedback.
- The hidden generator and hidden validity oracle must not import defender implementation modules.
- Agentic payment integrity checks execute before probabilistic scoring and fail closed when required evidence is missing.
- Promotion requires frozen artifacts, operational-budget compliance, per-family minimums, and human approval.
- The prototype must run offline with one deterministic golden-path fixture.
- Simulation and red-team code must remain synthetic-only and must not connect to or target live payment systems.
- UI content is visible by default and follows the workspace anti-slop design law.

## Execution gates

| Gate | Required evidence |
|---|---|
| G0 | Clean-machine bootstrap and all contract tests pass |
| G1 | Card, A2A, and agentic rail invariants pass under property testing |
| G2 | Fixed, random, adaptive, and hidden-generator campaigns run reproducibly |
| G3 | Rules and GBDT baselines pass time-respecting evaluation without leakage |
| G4 | Six-view prototype completes the golden path in under five minutes |
| G5 | Submission archive passes accessibility, privacy, license, secret, provenance, and reproducibility checks |
