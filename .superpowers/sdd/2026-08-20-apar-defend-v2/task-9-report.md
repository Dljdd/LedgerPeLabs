# Task 9 Report: Defend v2 pre-execution status

## Scope

Task 9 documents the sealed Defend v2 protocol without asserting efficacy or
executing an evaluation. The existing v1 status, hashes, conclusions, and
result files were not changed.

## RED

- Added `test_readme_makes_no_v2_efficacy_claim` before changing the README.
  `.venv/bin/python -m pytest tests/evaluation/test_defense_v2_preexecution.py::test_readme_makes_no_v2_efficacy_claim -q`
  failed because the required `Defend v2: protocol sealed; evaluation not
  executed` status was absent.
- The initial complete suite exposed the load-bearing OpenAPI integration
  regression: `tests/api/test_health.py::test_openapi_exposes_only_the_approved_api_paths`
  failed because its exact static allowlist did not include the already-approved
  `/defense/v2/scorecard` and `/api/v1/defense/v2/scorecard` routes.
- The focused OpenAPI test reproduced that failure directly, reporting exactly
  those two paths as unexpected additions.

## GREEN

- README now states: `Defend v2: protocol sealed; evaluation not executed.` It
  also carries the approved synthetic-only, no-external-validity non-claim.
- `docs/TRACEABILITY.md` records the v2 protocol as sealed and not executed,
  with the signed preregistration and read-only verifier as evidence.
- The README regression passes and rejects the prohibited `Defend v2 achieved`
  efficacy phrase.
- `tests/api/test_health.py` retains an immutable, explicit
  `APPROVED_OPENAPI_PATHS` allowlist and exact equality check against the live
  OpenAPI document. It includes exactly the two approved v2 scorecard paths;
  it does not derive allowed paths from the application.
- Focused checks passed:

  ```text
  .venv/bin/python -m pytest tests/evaluation/test_defense_v2_preexecution.py::test_readme_makes_no_v2_efficacy_claim -q
  1 passed

  .venv/bin/python -m pytest tests/api/test_health.py::test_openapi_exposes_only_the_approved_api_paths -q
  1 passed
  ```

## Full verification

```text
.venv/bin/python -m pytest -q
1809 passed, 1 skipped in 618.31s (0:10:18)

.venv/bin/python scripts/verify_g3.py
G3 PASS: causal features, rules/GBDT/hybrid, matched budgets, frozen hidden evaluation, and judge scorecards

.venv/bin/python scripts/verify_defense_v2_preexecution.py
{"admissible":true,"codes":[],"status":"not_executed"}
```

The G3 run passed its G0--G2 gates, then its remaining groups of 421, 92, 283,
and 2 tests. The pre-execution verifier and a durable-state scan found no v2
evaluation receipt, result artifact, or persisted `.apar` file.

## Commits

- `646f3d9 docs: record defend v2 pre-execution status`
- `5b650c2 test: approve defend v2 scorecard paths`
