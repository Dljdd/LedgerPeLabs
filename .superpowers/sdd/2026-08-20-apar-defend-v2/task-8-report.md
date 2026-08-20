# Task 8 Report: Read-only v2 pre-execution verification

## RED

- Added verifier, CLI, and public-scorecard tests before implementation.
- `.venv/bin/python -m pytest tests/evaluation/test_defense_v2_preexecution.py tests/integration/test_defense_v2_preexecution.py -q` failed at collection because `apar.evaluation.v2_preexecution` did not exist.
- Added the exact `/defense/v2/scorecard` API contract test; it failed with `404` before the public route was registered.

## GREEN

- Added `v2_preexecution.py`, containing only read-only protocol-root, preregistration, receipt, and static defender-import checks. It has no imports of `apar.evaluation_hidden`, worker code, or population generation code.
- Added a CLI that accepts an optional canonical signed preregistration, renders canonical `not_executed` JSON, and writes no evaluation artifact.
- Added immutable, self-verifying signed `not_executed` scorecard reads at `/defense/v2/scorecard` and the versioned `/api/v1/defense/v2/scorecard` alias. Neither route depends on an executor or starts work.
- The verifier fails closed with codes for invalid v1 roots, protocol binding, prior v2 receipt, hidden defender import, or invalid preregistration.

## Verification

- `ruff check` passed for all task files and the route registration.
- `mypy` passed for the verifier, CLI, and API route/application files.
- Focused verifier, integration CLI, and API tests passed: 22 tests.
- Required v2/frozen-v1 safety suite passed: 81 tests.
- `python scripts/verify_defense_v2_preexecution.py` outputs `{"admissible":true,"codes":[],"status":"not_executed"}`.
