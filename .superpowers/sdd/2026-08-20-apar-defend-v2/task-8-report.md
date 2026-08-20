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

## Review round 1/5 fixes

- RED: regression tests failed for `from apar.evaluation import service`, a computed `__import__('apar.' + 'evaluation_hidden')`, and a schema-valid v2 receipt stored under an arbitrary non-receipt filename.
- GREEN: defender imports now admit only the `apar.evaluation.v2_` public namespace, apart from an explicit frozen-v1 compatibility allowlist required for the pre-existing historical implementation. Computed module expressions passed to `__import__` or importlib aliases fail closed.
- Receipt detection now scans the complete durable `.apar` state store by `ExecutionReceipt` schema and v2 preregistration identifier; it does not depend on filenames, paths below that store, or a partial-byte inspection.
- Added coverage for an importlib alias with a variable module target and retained an allowed `apar.evaluation.v2_preexecution` import regression.
- Focused verifier, CLI, and API tests pass; Ruff and mypy pass.

## Review round 2/5 fixes

- RED: four new tests failed for `import importlib.metadata` binding the root, an `importlib.metadata` alias, `from ..evaluation import hidden_source`, and `from .. import evaluation`.
- GREEN: importlib submodule imports now register both their bound alias and the root capability, so any computed `__import__`/`import_module` target fails closed.
- Relative `ImportFrom` nodes are resolved from the current defender package before evaluator namespace validation. The same logic rejects both evaluator bypass forms while accepting `from ..evaluation import v2_preexecution`.
- Durable receipt scanning is unchanged and remains covered by the prior arbitrary-path schema regression.
- Ruff, mypy, focused tests, and the requested complete v2/frozen-v1 safety suite pass.
