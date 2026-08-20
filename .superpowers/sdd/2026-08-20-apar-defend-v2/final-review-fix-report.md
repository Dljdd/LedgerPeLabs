# Defend V2 final-review hardening report

## Scope and outcome

This change closes the seven final-review blockers as one V2-only hardening pass. It does not modify frozen V1 source or evidence, start a Defend V2 evaluation, or create a V2 execution receipt/result.

## RED evidence

The initial focused regression command was:

```text
.venv/bin/python -m pytest \
  tests/evaluation/test_defense_v2_preregistration.py::test_missing_committed_protocol_profile_binding_is_rejected \
  tests/evaluation/test_defense_v2_preexecution.py::test_builtins_import_alias_cannot_import_a_computed_target \
  tests/evaluation/test_defense_v2_preexecution.py::test_assigned_builtins_import_capability_is_rejected \
  tests/evaluation/test_defense_v2_preexecution.py::test_transitive_defender_feature_import_is_scanned \
  tests/evaluation/test_defense_v2_selection.py::test_forged_valid_control_cannot_pass_selection \
  tests/cases/test_v2_workload.py::test_rejects_challenged_transaction_missing_from_review_cases \
  tests/evaluation/test_defense_v2_population.py::test_injection_outside_declared_day_horizon_is_rejected \
  tests/api/test_defense.py::test_v2_scorecard_reads_verified_current_state_after_receipt \
  tests/api/test_defense.py::test_v2_scorecard_fails_closed_when_receipt_has_no_signed_result -q
```

It failed in every blocker area: no required profile field, builtins aliases and transitive features were not detected, a forged `ControlValidity(valid=True)` passed selection, missing case evidence produced zero workload, an outside-horizon campaign was accepted, and the API returned the compiled-in `not_executed` scorecard despite durable state.

Additional RED checks failed for:

- a validly re-signed substituted profile (`PROTOCOL_PROFILE_INVALID` absent);
- digest-shaped bindings with no committed manifest registry (`MANIFEST_BINDINGS_INVALID` absent);
- direct `builtins.__import__` and `getattr(builtins, "__import__")` paths;
- the public scorecard using the old protocol-ID/scope digest instead of the committed profile digest.

## Implemented hardening

1. **Preregistration/profile/manifests**
   - Added required signed `protocol_profile_sha256`, `manifest_registry_sha256`, and `budget_manifest_sha256` bindings.
   - Added committed canonical V2 manifest registry and signed preregistration files.
   - Preexecution now loads the real profile and registry, checks profile ID/digest, exact named seed commitments, frozen budget digest, and every source/feature/grid/population/evaluator/metric/bootstrap/control/reporting/fidelity manifest digest.

2. **Import boundary**
   - Propagates importlib and builtins import capabilities through aliases and annotated/ordinary assignments.
   - Rejects computed direct, aliased, subscripted, and reflective builtins/importlib import paths.
   - Traverses defender and defender-reachable `apar.features` modules transitively while retaining explicitly allowed `apar.evaluation.v2_*` imports and the frozen V1 compatibility allowlist.

3. **Mandatory controls**
   - Replaced caller-provided validity booleans with a capability-attested `ControlValidity` containing both exact `benign_only` and `score_permutation` producer results.
   - Selection emits `CONTROL_INVALID` for forged, absent, unattested, invalid, or not-run control evidence.

4. **Workload**
   - Every challenged transaction must be present in exactly one supplied review-case universe; incomplete case evidence now fails closed before workload rates are derived.

5. **Operating population**
   - Population manifests now declare exact UTC `horizon_start`/`horizon_end` values coherent with `day_count`.
   - Base and injected decisions must stay within that frozen interval, and injection preserves the base manifest horizon.

6. **Read-only API state**
   - The endpoint scans durable state read-only, verifies canonical signed scorecards, reconciles them with schema-valid V2 receipts, serves a completed current card after a receipt, and returns 422 rather than stale `not_executed` when receipt/result state is incomplete or inconsistent.
   - The compiled-in signed fallback and reporting default now bind the committed profile digest.

7. **Cosmetic**
   - Fixed the mis-indented score-permutation invalidation branch and formatted all touched Python files.

## GREEN evidence

Focused and safety verification:

```text
.venv/bin/python -m pytest tests/evaluation/test_defense_v2_protocol.py tests/evaluation/test_defense_v2_population.py tests/cases/test_v2_workload.py tests/evaluation/test_defense_v2_selection.py tests/evaluation/test_defense_v2_controls.py tests/evaluation/test_defense_v2_preregistration.py tests/evaluation/test_defense_v2_reporting.py tests/evaluation/test_defense_v2_preexecution.py tests/integration/test_defense_v2_preexecution.py tests/api/test_defense.py tests/evaluation/test_frozen_defense_v1.py -q
119 passed in 105.36s

.venv/bin/python -m pytest tests/evaluation/test_defense_v2_preexecution.py -q
24 passed in 1.49s

.venv/bin/ruff check <all changed Python source and tests>
All checks passed!

.venv/bin/mypy src/apar/evaluation/v2_preregistration.py src/apar/evaluation/v2_preexecution.py src/apar/evaluation/v2_controls.py src/apar/evaluation/v2_selection.py src/apar/evaluation/v2_population.py src/apar/cases/v2_workload.py src/apar/evaluation/v2_reporting.py src/apar/api/routes/defense.py scripts/verify_defense_v2_preexecution.py
Success: no issues found in 9 source files

.venv/bin/python scripts/verify_defense_v2_preexecution.py
{"admissible":true,"codes":[],"status":"not_executed"}

.venv/bin/python -m pytest -q
1823 passed, 1 skipped in 489.50s
```

Final exact-tree verification after the last seed, budget, and component-manifest regressions:

```text
.venv/bin/python -m pytest -q
1827 passed, 1 skipped in 481.88s (0:08:01)
```

## Final Important-findings hardening round

### Scope

This follow-up closes the remaining Important findings without modifying any
frozen V1 file and without starting V2 evaluation work.

- The committed manifest registry now contains a complete SHA-256 inventory of
  every Python file in the frozen defender/feature source trees and raw-byte
  bindings for the feature catalog, defender bundle, campaign ledger, calibration
  artifact, and threshold artifact. Preexecution rejects missing, added, symlinked,
  path-escaped, or content-modified inputs.
- The import scan now follows constant dynamic local imports and every local
  dependency reachable through a feature module. It recognizes direct, aliased,
  assigned, `getattr`, `__dict__`, subscript, and `__builtins__` import capabilities
  for both builtins and importlib while preserving explicit public
  `apar.evaluation.v2_*` imports.
- `ControlResult` is now canonical Ed25519 evidence signed by the committed V2
  evaluator authority. Admission revalidates the signature and exact content, so
  `model_copy` mutation cannot promote an invalid result. Both exact control kinds
  remain mandatory at selection.
- Durable API state now accepts only the exact committed preregistration ID and
  execution nonce. Current and fallback scorecards must be signed by the committed
  evaluator/publication authority; a fresh self-declared signing key is rejected.

### RED evidence

The initial preexecution regression run was:

```text
uv run pytest tests/evaluation/test_defense_v2_preexecution.py -q
11 failed, 27 passed in 2.93s
```

The failures covered four reflective builtins/importlib call shapes, a constant
dynamic feature import, a feature-reachable local package, and raw-byte mutations
of each required frozen input class (source, catalog, bundle, campaign ledger, and
threshold artifact).

The new control and API tests initially failed during collection because the
required `V2ControlError` and committed preregistration/nonce authority constants
did not exist. After exposing those interfaces, an additional alias probe was run:

```text
.venv/bin/pytest \
  tests/evaluation/test_defense_v2_preexecution.py::test_reflective_dynamic_import_capabilities_fail_closed -q
1 failed, 5 passed in 0.31s
```

The remaining failure was `runtime = builtins; runtime.__import__(module)`, proving
that builtins-root alias propagation was still absent before the final scanner
change.

A final reflective mapping probe then failed before implementation:

```text
.venv/bin/pytest \
  tests/evaluation/test_defense_v2_preexecution.py::test_reflective_dynamic_import_capabilities_fail_closed -q
2 failed, 6 passed in 0.40s
```

Those two failures covered import-function lookup through aliases of
`builtins.__dict__` and `vars(importlib)`.

### GREEN evidence

Focused contracts and integration:

```text
.venv/bin/pytest \
  tests/evaluation/test_defense_v2_preexecution.py \
  tests/evaluation/test_defense_v2_controls.py \
  tests/evaluation/test_defense_v2_selection.py \
  tests/evaluation/test_defense_v2_preregistration.py \
  tests/evaluation/test_defense_v2_reporting.py \
  tests/integration/test_defense_v2_preexecution.py -q
75 passed in 4.02s

.venv/bin/pytest tests/api/test_defense.py -q -k v2_scorecard
5 passed, 25 deselected in 1.29s
```

Static validation:

```text
.venv/bin/ruff check <all changed Python source and tests>
All checks passed!

.venv/bin/mypy \
  src/apar/evaluation/v2_preexecution.py \
  src/apar/evaluation/v2_preregistration.py \
  src/apar/evaluation/v2_controls.py \
  src/apar/evaluation/v2_selection.py \
  src/apar/evaluation/v2_reporting.py \
  src/apar/api/routes/defense.py \
  scripts/verify_defense_v2_preexecution.py
Success: no issues found in 7 source files
```

Read-only preexecution and complete suite:

```text
.venv/bin/python scripts/verify_defense_v2_preexecution.py
{"admissible":true,"codes":[],"status":"not_executed"}

.venv/bin/pytest -q
1846 passed, 1 skipped in 482.63s (0:08:02)
```

`git diff --exit-code` over `fixtures/defense/v1` and the three frozen V1
experiment documents returned zero, and no `.apar` V2 receipt/result file was
created.
