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

## Final hardening round 2

### Scope and implementation

- Removed the repository-held deterministic V2 authority seed and every committed
  `b"v" * 32` signing literal. Production now contains only a canonical sealed
  preregistration, its pinned Ed25519 public key/key ID, and a scorecard signed by
  that public authority. Test code creates isolated random ephemeral authorities;
  no generated private seed was printed or persisted.
- Extended the import boundary to propagate aliases of `getattr` and `vars`, detect
  reflective builtins/importlib import capability through those aliases, and follow
  constant local imports through transitive feature dependencies.
- Added lexical symbolic-link rejection before resolution for every frozen content
  reference, every source inventory entry, and every defender-reachable Python path.
  Symlinked Python is rejected rather than omitted from inventory or scanned through.
- Bound each control signature to the signed preregistration ID, execution nonce,
  arm, candidate ID, exact input digest, and evaluator public identity. Selection
  derives its expected binding from an intact signed preregistration and rejects
  evidence replayed across any of those dimensions.
- Added preregistration ID and execution nonce to the scorecard's signed payload.
  The API verifies the card signer against the sealed preregistration and compares
  those signed fields to the exact durable receipt, creating one cryptographic
  linkage rather than accepting independent valid objects.
- The API factory accepts a preregistration/scorecard pair only for explicit test
  injection. Its production default remains the compiled public-only sealed pair,
  and the GET route remains read-only and unable to start evaluation work.

### RED evidence

The first scanner/manifest regression run failed exactly the new alias and symlink
probes:

```text
.venv/bin/pytest tests/evaluation/test_defense_v2_preexecution.py -q
2 failed, 42 passed
```

The failures were the transitive `from builtins import getattr as lookup, vars as
namespace` reflective import path and a same-content frozen-input symlink. Control
and selection tests initially failed collection because the exact binding contracts
`V2ControlBinding` and `V2ControlContext` did not yet exist.

After closing the frozen-file case, a separate transitive Python-inventory probe
demonstrated the remaining reachable-feature symlink bypass:

```text
.venv/bin/pytest tests/evaluation/test_defense_v2_preexecution.py \
  -q -k transitive_feature_python_symlink
1 failed, 44 deselected
```

API/reporting regressions also initially referenced the removed deterministic
authority constants and old unbound scorecard shape, so a card signature could not
yet be checked as one object bound to the receipt nonce.

### GREEN evidence

Focused final hardening contracts:

```text
.venv/bin/pytest \
  tests/evaluation/test_defense_v2_preregistration.py \
  tests/evaluation/test_defense_v2_preexecution.py \
  tests/evaluation/test_defense_v2_controls.py \
  tests/evaluation/test_defense_v2_selection.py \
  tests/evaluation/test_defense_v2_reporting.py \
  tests/api/test_defense.py \
  -q -k 'v2 or preregistration or control or selection'
87 passed, 25 deselected in 5.78s

.venv/bin/pytest tests/api/test_defense.py -q -k v2_scorecard
5 passed, 25 deselected in 2.51s
```

Static and read-only admission verification:

```text
.venv/bin/ruff check <all changed Python source and tests>
All checks passed!

.venv/bin/mypy \
  src/apar/evaluation/v2_controls.py \
  src/apar/evaluation/v2_preexecution.py \
  src/apar/evaluation/v2_preregistration.py \
  src/apar/evaluation/v2_reporting.py \
  src/apar/evaluation/v2_selection.py \
  src/apar/api/app.py src/apar/api/routes/defense.py
Success: no issues found in 7 source files

.venv/bin/python scripts/verify_defense_v2_preexecution.py
{"admissible":true,"codes":[],"status":"not_executed"}
```

Complete repository regression suite:

```text
.venv/bin/pytest -q
1852 passed, 1 skipped in 493.86s (0:08:13)
```

`git diff --exit-code` confirmed no change beneath `src/apar/defense`,
`src/apar/features`, `fixtures/defense/v1`, or the three frozen V1 experiment
documents. Repository searches found no remaining trusted V2 constants,
deterministic `b"v" * 32` seed, or production `from_private_bytes` construction.
No V2 evaluation or durable execution receipt/result was created.

## Final hardening round 3

### Scope and implementation

- The import scanner now rejects reflective access rooted at builtins or importlib
  itself, rather than waiting for a recognizable import call. This covers aliased
  `getattr` and `vars`, `__dict__`/subscript access, non-constant attribute and
  slice expressions, assigned namespace aliases, and transitive feature modules.
- `V2ControlContext` now names one exact candidate and can be constructed from a
  presented preregistration only when an independently supplied sealed
  preregistration matches canonically. The match explicitly rechecks the trusted
  evaluator key ID, public key, and execution nonce.
- Control admission independently receives the sealed preregistration and expected
  execution context. It revalidates both signed control results, revalidates the
  context, calls `matches_sealed_preregistration`, compares the attested public key
  to the trusted key, and checks the exact preregistration/nonce/arm/candidate/input
  binding before admission.
- Threshold selection now receives an exact candidate-indexed context map plus the
  independent sealed preregistration. It rejects missing, extra, mismatched, or
  caller-self-authenticated contexts instead of deriving expectations from the
  `ControlValidity` evidence under review.

### RED evidence

The variable-name reflection probes failed before implementation:

```text
.venv/bin/pytest tests/evaluation/test_defense_v2_preexecution.py \
  -q -k 'variable_reflective or transitive_feature_variable'
5 failed, 45 deselected in 0.47s
```

They cover `attribute_name`, `method_name`, and `key` variables used through direct
and aliased `getattr`, `vars`, and `__dict__` subscript paths, including a
defender-reachable feature module.

The first independent-control-context probe failed during collection because the
old context factory accepted no sealed preregistration and no candidate ID:

```text
.venv/bin/pytest tests/evaluation/test_defense_v2_selection.py \
  -q -k 'outsider or independent'
TypeError: V2ControlContext.from_preregistration() got an unexpected keyword \
argument 'sealed_preregistration'
```

This demonstrated that the old API could not express an independent trust root.

### GREEN evidence

Focused scanner, control, and selection verification:

```text
.venv/bin/pytest \
  tests/evaluation/test_defense_v2_controls.py \
  tests/evaluation/test_defense_v2_selection.py \
  tests/evaluation/test_defense_v2_preexecution.py -q
78 passed in 3.99s
```

Static and read-only admission verification:

```text
.venv/bin/ruff check <round-3 changed Python source and tests>
All checks passed!

.venv/bin/mypy \
  src/apar/evaluation/v2_controls.py \
  src/apar/evaluation/v2_preexecution.py \
  src/apar/evaluation/v2_selection.py
Success: no issues found in 3 source files

.venv/bin/python scripts/verify_defense_v2_preexecution.py
{"admissible":true,"codes":[],"status":"not_executed"}
```

Complete repository regression suite:

```text
.venv/bin/pytest -q
1861 passed, 1 skipped in 481.48s (0:08:01)
```

`git diff --exit-code` confirmed no change beneath the frozen V1 source, feature,
fixture, or experiment-document paths. `git diff --check` was clean. No V2
evaluation or durable execution receipt/result was created.

## Final hardening round 4

### Scope and implementation

- Replaced the caller-supplied sealed-preregistration parameter with
  `V2VerifiedAuthority`, an opaque process-local capability. Its public constructor
  raises, it has no serializable authority fields, and only live objects recorded
  by the verifier's private weak registry resolve to a trusted preregistration.
- Added `verify_v2_authority(root, preregistration)` as the production issuance
  boundary. It mints a capability only after the complete pinned preexecution suite
  verifies the committed authority, profile, manifests, source boundary, V1 roots,
  and unused execution admission.
- Control context construction, control admission, gate evaluation, and threshold
  selection now consume the opaque capability. Passing a self-signed
  preregistration as both alleged seal and context is no longer an API path and
  resolves to no authority; unregistered exact-type objects also carry no trust.
- Extended alias analysis to recognize qualified `builtins.getattr` and
  `builtins.vars` calls, qualified aliases such as `lookup = runtime.getattr` and
  `namespace = runtime.vars`, annotated aliases, and their transitive feature use.
  Non-constant attribute names and mapping keys fail closed.

### RED evidence

Qualified builtins reflection initially bypassed all exact probes:

```text
.venv/bin/pytest tests/evaluation/test_defense_v2_preexecution.py \
  -q -k qualified_builtins
5 failed, 50 deselected in 0.70s
```

The exact caller-self-sealing exploit also passed every gate before the capability
change:

```text
.venv/bin/pytest tests/evaluation/test_defense_v2_selection.py \
  -q -k own_selection_trust_root
1 failed, 16 deselected in 0.29s

AssertionError: assert 'CONTROL_INVALID' in ()
```

Finally, the first opacity/issuer probes failed collection because neither the
capability nor the trusted verifier issuance API existed:

```text
ImportError: cannot import name 'V2VerifiedAuthority'
ImportError: cannot import name 'verify_v2_authority'
2 errors during collection
```

### GREEN evidence

Focused V2 contracts and API regression:

```text
.venv/bin/pytest \
  tests/evaluation/test_defense_v2_preregistration.py \
  tests/evaluation/test_defense_v2_preexecution.py \
  tests/evaluation/test_defense_v2_controls.py \
  tests/evaluation/test_defense_v2_selection.py \
  tests/evaluation/test_defense_v2_reporting.py \
  tests/integration/test_defense_v2_preexecution.py \
  tests/api/test_defense.py -q
130 passed in 57.97s
```

Static and read-only admission verification:

```text
.venv/bin/ruff check <round-4 changed Python source and tests>
All checks passed!

.venv/bin/mypy \
  src/apar/evaluation/v2_preregistration.py \
  src/apar/evaluation/v2_preexecution.py \
  src/apar/evaluation/v2_controls.py \
  src/apar/evaluation/v2_selection.py
Success: no issues found in 4 source files

.venv/bin/python scripts/verify_defense_v2_preexecution.py
{"admissible":true,"codes":[],"status":"not_executed"}
```

Complete repository regression suite:

```text
.venv/bin/pytest -q
1869 passed, 1 skipped in 478.51s (0:07:58)
```

Frozen V1 source, feature, fixture, and experiment-document paths remain unchanged;
`git diff --check` is clean. No V2 evaluation or durable receipt/result was created.

## Final hardening round 5

### Scope and implementation

- Removed `V2VerifiedAuthority`, `_issue_verified_v2_authority`, and the mutable
  authority registry from `v2_preregistration`. The opaque capability class and
  weak registry now exist only inside a private closure owned by
  `v2_preexecution`; the only issuance function first runs the complete pinned
  `verify_v2_preexecution` boundary.
- Changed control admission and threshold selection to resolve capabilities through
  that verifier-owned closure. A caller cannot construct the capability, and no
  module-level registry is exposed for mutation.
- Replaced the tests' direct private issuer call with an isolated ephemeral
  authority fixture. Trusted test authorities copy the actual frozen source,
  catalog, manifest, bundle, campaign-ledger, threshold, and V1-root inputs into a
  temporary repository and obtain their capability through the same public
  preexecution verifier used by production.
- Extended the import scanner to fail closed when builtins/importlib reflection
  authority is retained through destructuring, assignment expressions, function
  defaults, containers, attribute targets, or other unsupported assignment forms.
  This covers qualified and aliased `getattr`, `vars`, `__import__`, and importlib
  roots, including defender-reachable transitive feature modules.

### RED evidence

The initial binding probes demonstrated that destructuring, walrus assignment,
defaults, containers, and attribute assignment were not rejected:

```text
.venv/bin/pytest tests/evaluation/test_defense_v2_preexecution.py \
  -q -k 'unsupported_reflection or destructured_reflection'
6 failed, 56 deselected in 0.75s
```

The preregistration module still exposed its private issuer before the move:

```text
.venv/bin/pytest tests/evaluation/test_defense_v2_preregistration.py \
  -q -k exposes_no_capability_issuer_or_registry
1 failed, 8 deselected
```

Additional exact importlib and `builtins.__import__` probes failed before the
scanner was widened from only `getattr`/`vars` references to every reflection
authority root:

```text
.venv/bin/pytest tests/evaluation/test_defense_v2_preexecution.py \
  -q -k 'unsupported_reflection or destructured_importlib'
5 failed, 5 passed, 57 deselected in 2.96s
```

### GREEN evidence

Focused issuer, preregistration, scanner, controls, and selection contracts:

```text
.venv/bin/pytest \
  tests/evaluation/test_defense_v2_preexecution.py \
  tests/evaluation/test_defense_v2_preregistration.py \
  tests/evaluation/test_defense_v2_controls.py \
  tests/evaluation/test_defense_v2_selection.py -q
105 passed in 7.40s
```

Static verification:

```text
.venv/bin/ruff check <round-5 changed Python source and tests>
All checks passed!

.venv/bin/mypy \
  src/apar/evaluation/v2_preexecution.py \
  src/apar/evaluation/v2_preregistration.py \
  src/apar/evaluation/v2_controls.py \
  src/apar/evaluation/v2_selection.py \
  tests/evaluation/v2_authority.py
Success: no issues found in 5 source files
```

Complete repository regression suite:

```text
.venv/bin/pytest -q
1881 passed, 1 skipped in 478.10s (0:07:58)
```

The final direct read-only check returned
`{"status":"not_executed","admissible":true,"codes":[]}`. No frozen V1 source,
feature, fixture, or experiment-document path changed; `git diff --check` is clean.
No V2 evaluation or durable receipt/result was created.

## Final hardening round 6

### Scope and implementation

- Replaced the closure-local class plus `WeakKeyDictionary` registry with a plain
  `V2VerifiedAuthority` evidence contract. The object carries the exact signed
  preregistration and its canonical digest; construction itself grants no trust.
  Every control-context and selection admission reparses the evidence, verifies the
  source-pinned production evaluator key ID/public key, verifies the Ed25519-signed
  preregistration, and reruns the complete fixed-root preexecution boundary.
- Removed the caller-selected root parameter from `verify_v2_authority`. Production
  issuance now uses only the deployment root derived from the installed verifier
  source and the pinned public evaluator identity. A copied repository with a
  caller-re-signed seal cannot issue or validate production authority.
- Kept test authority injection explicitly outside production APIs: isolated tests
  copy all frozen inputs, pass complete preexecution, and use pytest-scoped
  monkeypatching for their temporary root and ephemeral public identity. This is a
  privileged test harness, not a Python-opacity or capability security claim.
- Removed sensitive evaluator modules (`v2_preexecution`, `v2_preregistration`,
  `v2_controls`, `v2_selection`, and `v2_reporting`) from the defender import
  surface, preventing defender-reachable code from retargeting trust state. The
  versioned data-only `v2_protocol` contract remains importable in absolute and
  relative forms.
- Extended reflection scanning across call arguments, container mutation,
  subscripted callables, `for`/async-`for` destructuring, comprehensions, augmented
  assignments, and assignment expressions. Qualified builtins reflection fails
  closed, while static non-import importlib metadata calls remain admissible.

### RED evidence

The exact copied-root, registry/capability, append/index, and loop-destructuring
probes all demonstrated the prior bypasses:

```text
.venv/bin/pytest \
  tests/evaluation/test_defense_v2_preexecution.py \
  tests/evaluation/test_defense_v2_controls.py \
  -q -k 'caller_selected_repository or copied_root_self_signed or mutated_container'
4 failed, 79 deselected in 3.10s
```

Specifically, the caller-selected copied repository issued an authority without
raising, the copied-root self-signed context was accepted, and neither
`items.append(builtins.getattr); items[0]` nor `for lookup, ... in
((builtins.getattr, ...),)` produced `HIDDEN_IMPORT_BOUNDARY`.

### GREEN evidence

Exact adversarial trust, reflection, and sensitive-import probes:

```text
.venv/bin/pytest \
  tests/evaluation/test_defense_v2_preexecution.py \
  tests/evaluation/test_defense_v2_controls.py \
  -q -k 'caller_selected_repository or copied_root_self_signed or \
  mutated_container or sensitive_versioned or public_v2_protocol'
6 passed, 78 deselected in 2.59s
```

Focused V2 preregistration, preexecution, controls, and selection contracts:

```text
.venv/bin/pytest \
  tests/evaluation/test_defense_v2_preexecution.py \
  tests/evaluation/test_defense_v2_preregistration.py \
  tests/evaluation/test_defense_v2_controls.py \
  tests/evaluation/test_defense_v2_selection.py -q
111 passed in 63.00s (0:01:02)
```

Static verification:

```text
.venv/bin/ruff check <round-6 changed Python source and tests>
All checks passed!

.venv/bin/mypy \
  src/apar/evaluation/v2_preexecution.py \
  src/apar/evaluation/v2_preregistration.py \
  src/apar/evaluation/v2_controls.py \
  src/apar/evaluation/v2_selection.py \
  tests/evaluation/v2_authority.py
Success: no issues found in 5 source files
```

Complete repository regression suite:

```text
.venv/bin/pytest -q
1887 passed, 1 skipped in 540.62s (0:09:00)
```

The final read-only preexecution script returned
`{"admissible":true,"codes":[],"status":"not_executed"}`. Frozen V1 source,
feature, fixture, and experiment-document paths have no diff; `git diff --check`
is clean. No V2 evaluation or durable receipt/result was created.

## Final hardening round 7

### Scope and implementation

- Added explicit static tracking for direct and aliased `sys` module roots in every
  defender and defender-reachable feature module.
- Made interpreter import authority fail closed: `sys.modules`, `sys.meta_path`,
  `sys.path_hooks`, `sys.path_importer_cache`, `sys.__dict__`, `vars(sys)`,
  constant or computed `getattr(sys, ...)`, `from sys import modules`, and dynamic
  loading of `sys` are rejected.
- Closed dunder reflection paths through `sys.__getattribute__`, `sys.__getattr__`,
  and generic calls such as `object.__getattribute__(sys, ...)`.
- Propagated simple `sys` aliases while rejecting laundering through destructuring,
  assignment expressions, defaults, containers, loops, comprehensions, augmented
  assignments, and arbitrary call arguments. Legitimate frozen-V1 access to
  `sys.implementation`, `sys.byteorder`, stdout, and stderr remains admissible.

### RED evidence

The initial exact loaded-module/import-hook probes all bypassed the boundary:

```text
.venv/bin/pytest tests/evaluation/test_defense_v2_preexecution.py \
  -q -k 'sys_import_authority or transitive_feature_sys_modules'
7 failed, 71 deselected in 0.56s
```

After direct/aliased registry handling, two additional acquisition paths remained
and were captured before implementation:

```text
.venv/bin/pytest tests/evaluation/test_defense_v2_preexecution.py \
  -q -k sys_import_authority
2 failed, 6 passed, 72 deselected in 2.64s
```

The exact `sys.__getattribute__` and `sys.__getattr__` probes then demonstrated the
dunder-reflection gap before it was closed:

```text
.venv/bin/pytest tests/evaluation/test_defense_v2_preexecution.py \
  -q -k sys_import_authority
2 failed, 9 passed, 72 deselected in 0.37s
```

### GREEN evidence

Exact direct, aliased, dynamic, dunder, and transitive feature probes:

```text
.venv/bin/pytest tests/evaluation/test_defense_v2_preexecution.py \
  -q -k 'sys_import_authority or transitive_feature_sys_modules'
12 passed, 71 deselected in 0.30s
```

Complete preexecution scanner suite and static checks:

```text
.venv/bin/pytest tests/evaluation/test_defense_v2_preexecution.py -q
83 passed in 9.10s

.venv/bin/ruff check \
  src/apar/evaluation/v2_preexecution.py \
  tests/evaluation/test_defense_v2_preexecution.py
All checks passed!

.venv/bin/mypy src/apar/evaluation/v2_preexecution.py
Success: no issues found in 1 source file
```

Complete repository regression suite:

```text
.venv/bin/pytest -q
1899 passed, 1 skipped in 558.51s (0:09:18)
```

The final read-only preexecution script returned
`{"admissible":true,"codes":[],"status":"not_executed"}`. Frozen V1 source,
feature, fixture, and experiment-document paths have no diff; `git diff --check`
is clean. No V2 evaluation or durable receipt/result was created.

## Final hardening round 8

### Scope and implementation

- Rejected every `from sys import ...` form, including wildcard imports, because
  frozen V1 uses only ordinary `import sys` and needs no from-import exception.
- Added a fail-closed namespace/code-execution primitive boundary covering
  `eval`, `exec`, `compile`, `globals`, `locals`, `vars`, `__import__`, `getattr`,
  `setattr`, and `delattr`, including builtins imports, qualified builtins access,
  aliases, stored references, and reflective acquisition.
- Rejected frame recovery through `sys._getframe` and `sys._current_frames`, and
  rejected `inspect` imports/dynamic acquisition in defender-reachable code.
- Preserved the four necessary frozen-V1 `getattr` call sites only when the entire
  containing source file matches its committed SHA-256. Any edit, alternate file,
  or reachable feature loses the exemption and fails closed.

### RED evidence

The exact wildcard import bypass initially remained admissible:

```text
.venv/bin/pytest tests/evaluation/test_defense_v2_preexecution.py \
  -q -k sys_import_authority
1 failed, 11 passed, 72 deselected in 0.45s
```

The consolidated namespace/code-execution probes all bypassed before the
primitive-acquisition rule was added:

```text
.venv/bin/pytest tests/evaluation/test_defense_v2_preexecution.py \
  -q -k namespace_and_code_execution
10 failed, 84 deselected in 0.49s
```

These probes cover `globals()['__builtins__']['__import__']`, string `eval`, an
imported eval alias, compile/exec, locals, zero-argument vars, `sys._getframe`,
getattr, setattr, and delattr.

### GREEN evidence

Combined sys and namespace authority probes:

```text
.venv/bin/pytest tests/evaluation/test_defense_v2_preexecution.py \
  -q -k 'namespace_and_code_execution or sys_import_authority'
22 passed, 72 deselected in 0.35s
```

Complete preexecution scanner and static verification:

```text
.venv/bin/pytest tests/evaluation/test_defense_v2_preexecution.py -q
94 passed in 9.06s

.venv/bin/ruff check \
  src/apar/evaluation/v2_preexecution.py \
  tests/evaluation/test_defense_v2_preexecution.py
All checks passed!

.venv/bin/mypy src/apar/evaluation/v2_preexecution.py
Success: no issues found in 1 source file
```

Complete repository regression suite:

```text
.venv/bin/pytest -q
1910 passed, 1 skipped in 565.29s (0:09:25)
```

The final read-only preexecution script returned
`{"admissible":true,"codes":[],"status":"not_executed"}`. Frozen V1 source,
feature, fixture, and experiment-document paths have no diff; `git diff --check`
is clean. No V2 evaluation or durable receipt/result was created.

## Final hardening round 9

### Scope and implementation

- Replaced recovery-path patching for non-frozen defender code with a positive
  language capability policy. Only byte-exact path/SHA-256 source named by the
  sealed defender/feature manifest retains the audited compatibility surface.
- Every other defender-reachable module must consist solely of explicitly
  admitted AST node types, static imports, attributes, and call targets. Lambdas,
  comprehensions, dunder attributes, dynamic importing/code, arbitrary builtins,
  introspection modules such as `gc`, and every unlisted capability fail closed.
- Applied that strict policy through the existing transitive feature/local-package
  traversal, while retaining exact-hash frozen V1 evaluator-import compatibility.
- Documented the independent runtime boundary required before any future v2
  execution: defender code must run in a fresh OS process with no evaluator
  modules, signing/seed material, module cache, shared Python objects, writable
  evaluator source, or network. Only preregistration/nonce-bound canonical bytes
  may cross the process boundary. The read-only verifier does not claim this
  execution boundary has run.

### RED evidence

The exact runtime-object probes initially remained admissible:

```text
.venv/bin/pytest -q tests/evaluation/test_defense_v2_preexecution.py \
  -k 'runtime_object_graph'
3 failed, 94 deselected in 0.50s
```

Those failures were the direct `(lambda: None).__globals__` recovery, bare
`import gc; gc.get_objects()`, and the same lambda recovery in a transitively
reachable feature module. After the syntax/import policy was in place, a
separate arbitrary-builtin capability probe demonstrated that `open(...)` was
still accepted:

```text
.venv/bin/pytest -q tests/evaluation/test_defense_v2_preexecution.py \
  -k 'runtime_object_graph'
1 failed, 3 passed, 94 deselected in 0.35s
```

### GREEN evidence

Exact object-graph, introspection-module, file-capability, transitive feature,
and sealed-root probes:

```text
.venv/bin/pytest -q tests/evaluation/test_defense_v2_preexecution.py \
  -k 'runtime_object_graph or signed_preregistration_and_frozen_v1_roots_are_not_executed'
5 passed, 93 deselected in 1.20s
```

Complete preexecution scanner, static checks, and read-only verifier:

```text
.venv/bin/pytest -q tests/evaluation/test_defense_v2_preexecution.py
98 passed in 9.35s

.venv/bin/ruff check \
  src/apar/evaluation/v2_preexecution.py \
  tests/evaluation/test_defense_v2_preexecution.py
All checks passed!

.venv/bin/mypy src/apar/evaluation/v2_preexecution.py
Success: no issues found in 1 source file

.venv/bin/python scripts/verify_defense_v2_preexecution.py
{"admissible":true,"codes":[],"status":"not_executed"}
```

Complete repository regression suite:

```text
.venv/bin/pytest -q
1914 passed, 1 skipped in 557.97s (0:09:17)
```

`git diff --exit-code` confirmed no changes under `src/apar/defense`,
`src/apar/features`, `fixtures/defense/v1`, or the three frozen V1 experiment
documents. `git diff --check` is clean. No V2 evaluation or durable execution
receipt/result was created.

## Final hardening round 10

### Scope and implementation

- Narrowed the non-frozen local import surface from broad `apar.contracts.*`
  admission to the exact public `_validation`, `decisions`, `events`, and
  `scenarios` data-contract modules, the feature namespace, and public v2
  protocol contract.
- Made every admitted local contract/feature import enter recursive capability
  traversal. The traversal includes each executable parent-package
  `__init__.py`, the leaf module, and every subsequent local dependency.
- Path/SHA-pinned the audited current contract package and exact data-contract
  modules. Any new or modified contract loses compatibility status and must pass
  the strict positive AST/import/attribute/call policy.
- Kept evaluator namespaces terminal and forbidden rather than recursively
  scanning irrelevant evaluator implementation modules.

### RED evidence

A direct broadly admitted local contract initially executed without traversal:

```text
.venv/bin/pytest -q tests/evaluation/test_defense_v2_preexecution.py \
  -k direct_contract_dependency_runtime_recovery
1 failed, 98 deselected in 0.40s
```

An approved exact data-contract import also initially skipped its modified leaf,
and then its executable package initializer:

```text
.venv/bin/pytest -q tests/evaluation/test_defense_v2_preexecution.py \
  -k allowed_contract_dependency_is_scanned_transitively
1 failed, 99 deselected in 2.75s

.venv/bin/pytest -q tests/evaluation/test_defense_v2_preexecution.py \
  -k allowed_contract_package_initializer_is_scanned
1 failed, 100 deselected in 0.34s
```

### GREEN evidence

Complete preexecution scanner, static checks, and read-only verifier:

```text
.venv/bin/pytest -q tests/evaluation/test_defense_v2_preexecution.py
101 passed in 9.00s

.venv/bin/ruff check \
  src/apar/evaluation/v2_preexecution.py \
  tests/evaluation/test_defense_v2_preexecution.py
All checks passed!

.venv/bin/mypy src/apar/evaluation/v2_preexecution.py
Success: no issues found in 1 source file

.venv/bin/python scripts/verify_defense_v2_preexecution.py
{"admissible":true,"codes":[],"status":"not_executed"}
```

Complete repository regression suite:

```text
.venv/bin/pytest -q
1917 passed, 1 skipped in 563.17s (0:09:23)
```

Frozen V1 source, feature, fixture, and experiment-document paths have no diff;
`git diff --check` is clean. No V2 evaluation or durable receipt/result was
created.

## Final hardening round 11

### Scope and implementation

- Special-cased the sole permitted evaluator import,
  `apar.evaluation.v2_protocol`, so graph traversal includes both its leaf source
  and the executable `apar.evaluation` package initializer.
- Pinned both public evaluator files to verifier-owned SHA-256 values. Any
  present replacement bytes fail admission even if they otherwise use only safe
  AST capabilities.
- Retained static inspection of the exact current files. The audited package
  initializer may retain its byte-exact frozen V1 compatibility imports, while
  every other evaluator module remains a forbidden terminal node and is never
  admitted through recursive traversal.

### RED evidence

Both previously omitted executable files accepted lambda/globals recovery before
the public evaluator traversal was added:

```text
.venv/bin/pytest -q tests/evaluation/test_defense_v2_preexecution.py \
  -k public_v2_protocol_module_and_package_are_scanned
2 failed, 101 deselected in 0.40s
```

A syntactically capability-safe replacement protocol also demonstrated that AST
inspection alone did not bind the public contract bytes:

```text
.venv/bin/pytest -q tests/evaluation/test_defense_v2_preexecution.py \
  -k present_public_v2_protocol_must_match_pinned_bytes
1 failed, 103 deselected in 2.27s
```

### GREEN evidence

Complete preexecution scanner, static checks, and read-only verifier:

```text
.venv/bin/pytest -q tests/evaluation/test_defense_v2_preexecution.py
104 passed in 9.19s

.venv/bin/ruff check \
  src/apar/evaluation/v2_preexecution.py \
  tests/evaluation/test_defense_v2_preexecution.py
All checks passed!

.venv/bin/mypy src/apar/evaluation/v2_preexecution.py
Success: no issues found in 1 source file

.venv/bin/python scripts/verify_defense_v2_preexecution.py
{"admissible":true,"codes":[],"status":"not_executed"}
```

Complete repository regression suite:

```text
.venv/bin/pytest -q
1920 passed, 1 skipped in 562.45s (0:09:22)
```

Frozen V1 source, feature, fixture, and experiment-document paths have no diff;
`git diff --check` is clean. No V2 evaluation or durable receipt/result was
created.

## Final hardening round 12

### Scope and implementation

- Relocated the defender-visible protocol implementation to
  `apar.v2_protocol`, outside the evaluator package and behind the inert
  `apar` package initializer.
- Removed the public protocol's dependency on `apar.runs.wire` by giving the
  isolated contract a minimal strict canonical JSON codec. A fresh process can
  now import the protocol without loading any `apar.evaluation`, `apar.runs`, or
  `apar.redteam` module.
- Updated every v2 production consumer and test to the isolated path. The old
  `apar.evaluation.v2_protocol` module is an evaluator-only compatibility
  re-export and is explicitly forbidden to defender source.
- Recursively scans the new public module, inert package initializer, and local
  dependency leaves. The public module and initializer remain verifier-owned
  path/SHA pins, so replacement bytes fail regardless of otherwise-safe syntax.
- Updated the v2 design's process-isolation contract; no V1 source, fixture, or
  experiment artifact changed.

### RED evidence

The required module did not exist before relocation, so a fresh-process import
failed immediately:

```text
.venv/bin/pytest -q tests/evaluation/test_defense_v2_protocol.py \
  -k public_protocol_import_does_not_load
1 failed, 9 deselected in 0.83s
```

At the same time the legacy evaluator path remained admitted while the new
isolated path was rejected by the scanner:

```text
.venv/bin/pytest -q tests/evaluation/test_defense_v2_preexecution.py \
  -k 'legacy_evaluator_v2_protocol or public_v2_protocol_import_remains_admissible'
2 failed, 100 deselected in 0.50s
```

### GREEN evidence

Protocol isolation and complete preexecution regression:

```text
.venv/bin/pytest -q \
  tests/evaluation/test_defense_v2_protocol.py \
  tests/evaluation/test_defense_v2_preexecution.py
112 passed in 9.44s

PYTHONPATH=src .venv/bin/python -c \
  'import sys; import apar.v2_protocol; ...'
loaded forbidden modules: []
```

Static and read-only verification:

```text
.venv/bin/ruff check <all changed Python source and tests>
All checks passed!

.venv/bin/mypy \
  src/apar/v2_protocol.py \
  src/apar/evaluation/v2_protocol.py \
  src/apar/evaluation/v2_preregistration.py \
  src/apar/evaluation/v2_population.py \
  src/apar/evaluation/v2_selection.py \
  src/apar/evaluation/v2_preexecution.py
Success: no issues found in 6 source files

.venv/bin/python scripts/verify_defense_v2_preexecution.py
{"admissible":true,"codes":[],"status":"not_executed"}
```

Complete repository regression suite:

```text
.venv/bin/pytest -q
1919 passed, 1 skipped in 567.35s (0:09:27)
```

Frozen V1 source, feature, fixture, and experiment-document paths have no diff;
`git diff --check` is clean. No V2 evaluation or durable receipt/result was
created.
