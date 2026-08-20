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
