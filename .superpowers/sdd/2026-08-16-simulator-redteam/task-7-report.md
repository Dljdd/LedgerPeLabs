# Task 7 implementation report

## Outcome

Task 7 completes the stateful simulator plan with four connected boundaries:

1. An independent hidden campaign generator and validity oracle under
   `apar.evaluation_hidden`. The package imports neither `apar.defense` nor
   `apar.generators`; it owns separate motifs, schedules, random leaves, lifecycle
   validation, role checks, balance feasibility, value conservation, connectivity,
   bounds, and benign-distance calculations.
2. A one-request-per-process policy worker. The parent serializes only public parameter
   bounds, visible history, and a typed policy kind. The worker is launched with an
   isolated interpreter, sanitized environment, temporary working directory, import and
audit restrictions, zero-network enforcement, hard resource limits, parent RSS and
output watchdogs, a hard deadline, and process-group termination.
3. A deterministic `RunRunner` that freezes scenario, policy, population, provenance,
   and restricted evaluator inputs before policy execution; executes the Task 5
   evaluator and production rails; freezes events, public feedback, restricted audit,
   restricted validity, and summary afterward; signs authorization and completion
   receipts; signs the exact manifest lineage; and publishes an append-only run index.
4. API routes that compile a registered threat card to an immutable scenario artifact,
   create a run from that artifact ID plus a closed policy selection, and retrieve a
   fully reverified signed manifest. The API accepts no file path, callable, uploaded
   policy, evaluator object, template, label, or raw hidden reason.

The final focused acceptance script reports three G1 production-rail invariants and six
G2 checks. After Fix Round 1 the final repository suite is `913 passed, 1 skipped`. The accepted Task 6
v3.4 result remains unchanged and is used only as frozen evidence. No experiment,
search, evaluator trial, cache preparation, seed, threshold, or policy tuning was run or
modified by Task 7.

## Binding architecture decisions

### Independent hidden evaluation

`HiddenCampaignGenerator.generate(family, seed, count)` supports exactly:

- `agentic_intent_abuse`;
- `app_scam_mule`;
- `card_testing_cnp`; and
- `synthetic_merchant_refund`.

It uses a locally constructed, domain-separated NumPy `Generator(PCG64)`, hidden entity
motifs, a separate causal
schedule, independent beta-distributed leaves, UUID5 identities, and no production
generator or defender import. It returns only `tuple[PaymentEvent, ...]`, as required by
the Task 7 ruling.

`HiddenValidityOracle.evaluate(...)` returns a frozen model with exactly one field,
`valid: bool`. Detailed stable reason codes and metrics exist only through a module-private
completed-run interface. `RunRunner` calls that interface only after policy execution and
stores the result in a restricted artifact. The attacker worker
cannot import the package, does not receive its module, template, constraints, labels,
audit records, digests, failures, or artifact references, and receives no validity bit
during proposal generation.

The oracle independently checks:

- exact, nonempty `PaymentEvent` ownership and canonical event ordering;
- unique event IDs and one campaign;
- card, A2A, and agentic lifecycle legality and causal predecessor lineage;
- stable amount, currency, actor, and counterparty through each payment lifecycle;
- declared actor/counterparty role pairs;
- consistent opening-balance evidence for every repeated entity;
- chronological balance feasibility and principal reversal at most once;
- USD/amount/payment-count/time bounds;
- attack-graph connectivity, excluding evaluator-labelled benign controls when present;
- a separately calculated amount/gap/role benign-distance bound.

Production APP replay initially exposed a useful independent disagreement: the oracle
rejected valid production events for missing public role pairs and because benign
controls form separate graph components. The final rule admits the concrete public
population roles and tests connectivity over the evaluator-labelled attack component,
while the fully independent hidden corpus remains evaluated without that label.

### Disposable policy process

`PolicyWorkerClient` accepts only `AttackerPolicyKind` values `fixed`, `random`,
`adaptive`, and `cached_llm`. The public `AttackerPolicy` is closed and frozen and holds
only reviewed family, attacker mode, policy kind, bounded query budget, and bounded worker
timeout. Feedback visibility comes only from the compiled scenario. Pydantic extra-field rejection prevents paths and
callables from entering the route or runner.

Each proposal launches the exact reviewed `policy_worker.py` through:

```text
<current-python> -I <absolute-reviewed-worker-path>
```

The worker gets a sanitized five-variable environment, a fresh temporary current
directory, closed inherited descriptors, and a new process session. Its canonical JSON
wire uses tagged public adaptive values and rejects duplicate keys, non-UTF-8 bytes,
non-finite JSON, noncanonical bytes, wrong field sets, wrong exact types, forged bounds,
history, candidate IDs, vector membership, or parent/generation lineage.

The worker package uses lazy public exports, so importing its public wire submodule does
not import the parent runner/orchestrator. Before installing the audit hook it loads the
reviewed NumPy RNG dependency needed by built-in policies; after that point every
filesystem `open` is denied. This prevents a policy from bypassing import restrictions
by reading hidden source, artifact, or signing-key bytes from local paths.

The worker installs both an import hook and a Python audit hook. Hidden evaluation,
production generators, defender/evaluator benchmark, simulator, trust verifier,
reflection/native/process modules, network sockets, subprocess/spawn/system operations,
native dynamic loading, and post-startup filesystem opens are denied. A fresh-process
probe proves forbidden imports, reflection imports, sockets, and filesystem reads fail;
hidden and parent-orchestrator modules are absent; and no hidden request fields exist.

The parent enforces:

- a 50--30,000 ms wall deadline;
- process-group `SIGKILL` on deadline or boundary failure;
- 1 MiB combined stdout/stderr and 1 MiB file-size limits;
- 8 MiB worker request read cap;
- hard core, CPU, file-descriptor, and process-count limits;
- a 768 MiB live RSS cap;
- Darwin RSS through `proc_pid_rusage` and Linux RSS through `/proc/<pid>/status`;
- fail-closed termination if RSS cannot be observed;
- no retained worker, evaluator, template, fixture, audit, or policy state after exit.

Darwin `RLIMIT_AS` was deliberately not used because a normal Python interpreter's
mapped address space exceeds a meaningful cap on that platform. The parent RSS watchdog
is platform-compatible, observes the actual resident set, is exercised against the real
worker, and has deterministic tests for both over-cap and unavailable-reader kill paths.

This is a boundary for the four reviewed built-in policy selections, not an API for
running arbitrary uploaded Python code. The cached-LLM selection in local Task 7 uses the
real `LLMPlannerPolicy` against the exact pinned Task 6 replay cache with a transport that
must never be called. The empirical cached-LLM claim is
carried only by the accepted frozen Task 6 v3.4 raw evidence, not by a Task 7 rerun.

### Signed immutable execution lineage

`RunSigningIdentity` is injected from exact 32-byte Ed25519 private material or loaded
from a durable mode-0600 regular non-symlink file. Creation uses exclusive/no-follow
open, complete short-write handling, file fsync, and directory fsync. API and artifacts
expose only the public Ed25519 bytes and their SHA-256 key ID. Private seed bytes never
enter an artifact, receipt, manifest, response, or log.

Before any worker executes, `RunRunner` stores and signs an authorization receipt over:

- policy;
- population;
- provenance;
- restricted evaluation input; and
- independent restricted hidden-evaluation events; and
- compiled scenario.

After the run, it stores and signs a completion receipt over:

- selected-winner production events;
- public feedback history;
- restricted production evaluation audit;
- restricted validity report; and
- summary.

The completion receipt contains the authorization receipt's artifact digest. The final
manifest contains six input artifacts, five output artifacts, and both receipts: thirteen
immutable references. Its signed lineage digest includes the complete named artifact
map and both receipt references.

Verification requires the configured durable signer, exact thirteen-name set, immutable
artifact reads, both receipt signatures, exact input/output digest maps, authorization
`previous=None`, completion-to-authorization chaining, identical receipt and manifest
run IDs, manifest lineage digest, and pinned provenance. An authentically signed receipt
chain from the same authority cannot be relabelled under another manifest run ID.

The durable run index is published with exclusive/no-follow mode-0600 creation and
complete short-write/fsync handling. When durable indexing is configured, every GET
rereads the filesystem entry rather than trusting an in-process cache. It opens the
entry with `O_NOFOLLOW`, validates regular-file/mode/size and stable metadata with
`fstat` on that same descriptor, bounds the read to 4 KiB, and requires exact canonical
`ArtifactRef` fields and types, a verified immutable manifest artifact, and complete
signature/receipt/provenance verification. Same-process replacement and a symlink swap
at descriptor open are therefore detected.

### Provenance and Task 6 mode ruling

Every run freezes a provenance artifact. Before admitting the pinned Task 6 files, the
runner opens them with `O_NOFOLLOW`, verifies a regular descriptor, checks the declared
current filesystem mode where POSIX exposes it, reads and hashes the descriptor, and
separately checks the historical Git object, mode, path, commit, and bytes.

The Task 6 v3.4 result is pinned as:

- path `docs/experiments/task6-v3.4-holdout-result.json`;
- current source mode `0600` or fresh-checkout `0644`, regular, owned, non-symlink, and single-link;
- private admitted execution mode `0600`;
- historical Git mode `100644`, object type `blob`;
- result commit `d6d3eecbfe2d871af8375e1455814cb5c48f2928`;
- SHA-256 `f82981a987651a7f7ebb10a9011df063b2dc54a56181cae5b838e31de5e658db`.

This closes the Task 6 deferred nuance: current filesystem mode and historical Git mode
are distinct facts and are verified separately.

### Production rail and hidden event ownership

For card and A2A families, the final event artifact is produced by public Task 5 campaign
commands replayed through `SimulationEngine` and the real public rail adapter with a
conserved ledger. The runner enriches immutable events with evaluator-owned public roles,
opening balances, and attack/control classification before independent validity.

For agentic attacks, Task 7 owns one deterministic public `TrustVerifier` configuration.
Every proposed candidate is generated through public Task 5 interfaces and replayed
through the real Task 4 agentic adapter under that configuration. The winning candidate
is regenerated and replayed under the identical configuration; its evaluated event
digest must equal the frozen production-event digest or the run fails closed. The
independent `HiddenCampaignGenerator` corpus is stored separately for validity evaluation.
No private Task 5 engine state or evaluator fixture is accessed.

G1 separately proves the production matrix: 25 commands yield 23 fail-closed declined
authorizations and two valid receipt-chain controls, with a conserved ledger. Task 5's
evaluator refuses to produce that observation unless its mandatory Task 4 reason
coverage and two controls succeed.

## API surface

The exact published paths are now:

- `GET /api/v1/health`;
- `GET /api/v1/threats`;
- `PUT /api/v1/threats/{threat_id}`;
- `GET /api/v1/threats/{threat_id}`;
- `POST /api/v1/scenarios/compile`;
- `POST /api/v1/runs`;
- `GET /api/v1/runs/{run_id}`.

Compilation accepts a registered `threat_id` and closed `ScenarioConfig`, compiles through
the existing public compiler, stores canonical immutable bytes, and returns only
`scenario_artifact_id` and `scenario_id`. Run creation resolves and verifies the content
address, reconstructs an exact `ScenarioBundle`, accepts a closed `AttackerPolicy`, and
returns a typed redacted `PublicRunManifest` backed by a reverified signed internal
manifest. No route returns restricted references, their hashes/sizes, their payloads, or
raw hidden reasons. Execution errors use a generic structured message and do
not disclose local paths or internal provenance failures.

Application lifespan owns one repository, artifact store, durable signer, and runner per
application instance. Existing dependency injection patterns are retained.

## Strict TDD evidence

### Independent hidden package

Initial RED:

```text
$ .venv/bin/python -m pytest tests/redteam/test_hidden_validity.py -q
ERROR collecting tests/redteam/test_hidden_validity.py
ModuleNotFoundError: No module named 'apar.evaluation_hidden'
```

Initial GREEN:

```text
$ .venv/bin/python -m pytest tests/redteam/test_hidden_validity.py -q
.........                                                                [100%]
9 passed in 0.30s
```

Later independent balance-evidence hardening RED:

```text
$ .venv/bin/python -m pytest tests/redteam/test_hidden_validity.py -q -k conflicting
FAILED test_hidden_oracle_rejects_conflicting_opening_balance_evidence
E AssertionError: assert not True
1 failed, 9 deselected in 0.14s
```

GREEN with the full hidden/G2 slice:

```text
$ .venv/bin/python -m pytest tests/redteam/test_hidden_validity.py \
    tests/integration/test_g2_adaptation.py -q
................                                                         [100%]
16 passed in 7.25s
```

### Worker, signer, receipts, and manifest

Initial RED:

```text
$ .venv/bin/python -m pytest tests/redteam/test_run_boundary.py -q
ERROR collecting tests/redteam/test_run_boundary.py
ModuleNotFoundError: No module named 'apar.runs'
```

First focused GREEN after strict wire/process implementation:

```text
$ .venv/bin/python -m pytest tests/redteam/test_run_boundary.py -q \
    -k 'durable or selection or clean_policy or non_returning or reconstructs'
.....                                                                    [100%]
5 passed, 3 deselected in 1.38s
```

First complete runner/manifest GREEN:

```text
$ .venv/bin/python -m pytest tests/redteam/test_run_boundary.py -q
........                                                                 [100%]
8 passed in 16.72s
```

The parent RSS watchdog's real over-cap termination test passed with the deadline and
clean-worker tests. A second fail-closed test began RED when an unavailable RSS reader
silently allowed the worker to finish:

```text
$ .venv/bin/python -m pytest tests/redteam/test_run_boundary.py -q -k rss_cannot
FAILED test_policy_worker_fails_closed_when_rss_cannot_be_observed
E Failed: DID NOT RAISE PolicyWorkerError
1 failed, 9 deselected in 0.34s
```

GREEN for actual clean worker, deadline kill, over-cap kill, and unavailable-reader kill:

```text
$ .venv/bin/python -m pytest tests/redteam/test_run_boundary.py -q \
    -k 'memory or rss or clean_policy or non_returning'
....                                                                     [100%]
4 passed, 6 deselected in 0.36s
```

Final filesystem/orchestrator isolation hardening began RED because the probe contract
had no filesystem result and the worker's `apar.runs.wire` import eagerly loaded the
parent runner:

```text
$ .venv/bin/python -m pytest tests/redteam/test_run_boundary.py -q -k clean_policy
FAILED test_clean_policy_worker_blocks_hidden_imports_reflection_and_network
E AttributeError: 'PolicyWorkerBoundaryReport' object has no attribute 'filesystem_blocked'
1 failed, 14 deselected in 0.33s
```

The first audit denial correctly closed filesystem reads but exposed a real lazy-load
dependency: NumPy's first RNG access attempted to open its random package after the hook,
so the proposal/runner slice failed closed (`2 failed, 4 passed, 9 deselected`). The
reviewed RNG entry point is now loaded before the audit hook, while every policy-time
open remains denied. The real worker/proposal/deadline/memory slice is GREEN:

```text
$ .venv/bin/python -m pytest tests/redteam/test_run_boundary.py -q \
    -k 'clean_policy or reconstructs or memory or rss or non_returning'
......                                                                   [100%]
6 passed, 9 deselected in 2.41s
```

Durable index regression RED, after correcting a test import:

```text
$ .venv/bin/python -m pytest tests/redteam/test_run_boundary.py -q -k rechecks
FAILED test_get_rechecks_the_durable_append_only_index_instead_of_memory
E Failed: DID NOT RAISE RunExecutionError
1 failed, 10 deselected in 1.62s
```

Same-authority receipt relabelling RED:

```text
$ .venv/bin/python -m pytest tests/redteam/test_run_boundary.py -q \
    -k receipt_chain_is_bound
FAILED test_authenticated_receipt_chain_is_bound_to_the_manifest_run_id
E AssertionError: assert not True
1 failed, 11 deselected in 1.58s
```

Combined GREEN:

```text
$ .venv/bin/python -m pytest tests/redteam/test_run_boundary.py -q \
    -k 'receipt_chain_is_bound or rechecks'
..                                                                       [100%]
2 passed, 10 deselected in 3.16s
```

A final descriptor-level self-review found that the original read path checked with
`lstat` and then reopened by name. The adversarial test swaps in a symlink exactly when
the descriptor is opened. It began RED because no descriptor open occurred:

```text
$ .venv/bin/python -m pytest tests/redteam/test_run_boundary.py -q -k symlink_swap
FAILED test_get_rejects_a_symlink_swap_at_the_index_descriptor_open
E Failed: DID NOT RAISE RunExecutionError
1 failed, 13 deselected in 2.62s
```

After moving validation and the bounded read onto one `O_NOFOLLOW` descriptor, both the
replacement and race tests are GREEN. A neighboring malformed-byte test first failed
with an escaping `WireContractError`; retrieval now converts parse/reference failures to
the same closed `RunExecutionError` boundary, and all three cases are GREEN:

```text
$ .venv/bin/python -m pytest tests/redteam/test_run_boundary.py -q -k malformed_index
FAILED test_get_converts_malformed_index_bytes_to_a_fail_closed_error
E apar.runs.wire.WireContractError: wire input is not strict UTF-8 JSON
1 failed, 14 deselected in 1.78s

$ .venv/bin/python -m pytest tests/redteam/test_run_boundary.py -q \
    -k 'malformed_index or symlink_swap or rechecks'
...                                                                      [100%]
3 passed, 12 deselected in 6.82s
```

Short low-level signing-key write RED and GREEN:

```text
$ .venv/bin/python -m pytest tests/redteam/test_run_boundary.py -q -k short_low
FAILED test_durable_signer_completes_short_low_level_writes
E ValueError: Ed25519 private seed must be exactly 32 bytes
1 failed, 12 deselected in 0.15s

$ .venv/bin/python -m pytest tests/redteam/test_run_boundary.py -q -k short_low
.                                                                        [100%]
1 passed, 12 deselected in 0.12s
```

Pre-self-review combined runner/API/artifact GREEN:

```text
$ .venv/bin/python -m pytest tests/redteam/test_run_boundary.py \
    tests/api/test_runs.py tests/storage/test_artifacts.py -q
..................................                                       [100%]
34 passed in 11.23s
```

Final combined boundary/API/artifact GREEN after descriptor and filesystem hardening:

```text
$ .venv/bin/python -m pytest tests/redteam/test_run_boundary.py \
    tests/api/test_runs.py tests/storage/test_artifacts.py -q
....................................                                     [100%]
36 passed in 15.77s
```

### Artifact lookup and API

Artifact resolution RED and GREEN:

```text
$ .venv/bin/python -m pytest tests/storage/test_artifacts.py -q -k resolve
FAILED test_resolve_returns_only_a_verified_content_address
E AttributeError: 'ArtifactStore' object has no attribute 'resolve'
1 failed, 17 deselected in 0.40s

$ .venv/bin/python -m pytest tests/storage/test_artifacts.py -q -k resolve
.                                                                        [100%]
1 passed, 17 deselected in 0.38s
```

Route/OpenAPI RED and GREEN:

```text
$ .venv/bin/python -m pytest tests/api/test_runs.py \
    tests/api/test_health.py::test_openapi_exposes_only_the_foundation_api_paths -q
FFFF                                                                     [100%]
4 failed in 0.92s

$ .venv/bin/python -m pytest tests/api/test_runs.py \
    tests/api/test_health.py::test_openapi_exposes_only_the_foundation_api_paths -q
....                                                                     [100%]
4 passed in 3.58s
```

Generic execution error RED and GREEN:

```text
$ .venv/bin/python -m pytest tests/api/test_runs.py -q -k typed_policy
FAILED test_run_accepts_only_a_compiled_id_and_typed_policy_then_is_gettable
E mismatch: internal runner message was exposed
1 failed, 2 deselected in 2.07s

$ .venv/bin/python -m pytest tests/api/test_runs.py -q -k typed_policy
.                                                                        [100%]
1 passed, 2 deselected in 2.12s
```

### G1/G2

G1 first RED caught an incorrect assumed package export:

```text
$ .venv/bin/python -m pytest tests/integration/test_g1_simulation.py -q
ERROR collecting tests/integration/test_g1_simulation.py
ImportError: cannot import name 'SimulationEngine' from 'apar.simulator'
```

The test was corrected to the existing public engine module; G1 GREEN:

```text
$ .venv/bin/python -m pytest tests/integration/test_g1_simulation.py -q
...                                                                      [100%]
3 passed in 0.49s
```

G2 first RED found the genuine independent validity mismatch:

```text
$ .venv/bin/python -m pytest tests/integration/test_g2_adaptation.py -q
....F.                                                                   [100%]
FAILED test_g2_disposable_policies_use_matched_budgets_and_seeded_bytes
E assert False is True
1 failed, 5 passed in 1.99s
```

The restricted reason artifact contained exactly
`ACTOR_ROLE_INVALID` and `CAMPAIGN_DISCONNECTED`. After the independent role/control
correction, G2 GREEN:

```text
$ .venv/bin/python -m pytest tests/integration/test_g2_adaptation.py -q
......                                                                   [100%]
6 passed in 7.63s
```

Final focused gate:

```text
$ .venv/bin/python scripts/verify_g1_g2.py
...                                                                      [100%]
3 passed in 0.39s
G1 PASS: card and A2A report/recovery conserve value; agentic 23-attack matrix fails closed with 2 controls
......                                                                   [100%]
6 passed in 8.66s
G2 PASS: 4 hidden families; fixed/random/adaptive/cached-LLM matched budgets; boolean-only validity; byte-identical seeded reruns; 2-family frozen capability evidence
```

## Final verification

Required simulator/trust/generator/red-team/G1/G2 command:

```text
$ .venv/bin/python -m pytest tests/simulator tests/trust tests/generators \
    tests/redteam tests/integration/test_g1_simulation.py \
    tests/integration/test_g2_adaptation.py -q
774 passed, 1 skipped in 145.06s (0:02:25)
```

Full repository:

```text
$ .venv/bin/python -m pytest -q
882 passed, 1 skipped in 147.02s (0:02:27)
```

Static analysis:

```text
$ .venv/bin/ruff check src tests scripts
All checks passed!

$ .venv/bin/mypy --strict <all Task 7 source and test targets>
Success: no issues found in 18 source files

$ .venv/bin/mypy src
Success: no issues found in 52 source files
```

Foundation gate:

```text
$ .venv/bin/python scripts/verify_g0.py
G0 PASS: 20 threat cards, contracts, registry, compiler, API, and artifact store
```

Hygiene and isolation:

```text
$ git diff --check
# exit 0, no output

$ git diff --exit-code 7534c0e38eaff568f390784113a0dd5992f8d048 -- validation_spike
# exit 0, no output
```

Frozen-file comparison to base `7534c0e` also exits 0 with no output for the v3.4
result/preregistration, policies, benchmark/defender, generator, LLM policy, search,
experiment, verifier, and `validation_spike` paths.

Current frozen hashes include:

```text
f82981a987651a7f7ebb10a9011df063b2dc54a56181cae5b838e31de5e658db  task6-v3.4-holdout-result.json
12bf24e081e97f3222bf1fc92fb1d441c36bba548184c6b503519590efc649a4  task6-v3.4-holdout-preregistration.json
c97ab7b263a493978cf901140a97f15874a34f8ff2ce54c84253e7baa998fb82  src/apar/redteam/policies.py
7996fcf20c85547a861afcfeb9da132dad534ad30f7fe1a07618e90e2faef519  src/apar/redteam/benchmark.py
670b4a3ec358f82d88f9655bd41d878fbee11d4841ff264655554bae31c3b31a  src/apar/generators/campaigns.py
2e54862322980414098c17930ec95bd268372da8968a78384a5bd661bfdaa2e5  src/apar/generators/population.py
8105a6788041f7d73b1afa571482f4b0ff3b15980f6c28769b701ab350936622  src/apar/redteam/llm_policy.py
ee05348ab07a9852a68a3f6a477eeec7ad6837d94f187e6fbb97767220f60e89  src/apar/redteam/search.py
a1367a8bb4310eeea2812a7d118ccb738ae1d9c32bfbc21c87413b1a869ce056  src/apar/redteam/task6_experiment.py
```

Historical Task 6 verification has an intentional two-stage record. Before commit, the
verifier refused with `RuntimeError: v3.4 freeze verification requires a clean Git
worktree`; this is the correct fail-closed behavior, not an evidence failure. The clean
postcommit command and result are recorded below after the Task 7 commit is created:

```text
$ .venv/bin/python scripts/run_task6_holdout.py --verify-postcommit \
    --approved-result-commit d6d3eecbfe2d871af8375e1455814cb5c48f2928 \
    --approved-result-sha256 f82981a987651a7f7ebb10a9011df063b2dc54a56181cae5b838e31de5e658db
verified Task 6 v3.4 result-only commit chronology and raw evidence; confirmatory_valid=True
```

## Honest claim boundary and residual scope

The accepted frozen v3.4 evidence reports matched budgets, zero cached-LLM network calls,
`supported_family_count=2`, `criterion_met=true`, and the exact preregistered adaptive
claim as supported. Task 7 verifies and cites those bytes; it does not recreate, retune,
or extend the experiment.

The capability result is evidence about this exact synthetic, replay-backed,
preregistered setup. It is not evidence of live-fraud detection or evasion effectiveness,
production safety, causal field performance, or generalization beyond the frozen
families, seeds, policies, fixtures, defender, and budgets.

The hostile boundary intentionally executes only reviewed built-in policy kinds. It is
not a general-purpose arbitrary-code sandbox and exposes no API for uploaded code. On
Darwin/Linux it fails closed if the parent cannot enforce or observe the declared process
limits. This narrower authority is the safe contract implemented by the typed API.

## Fix Round 1 — review rejection closure (2026-08-18)

This section supersedes every conflicting implementation statement or pre-fix count
above it. All Critical, Important, Warning/leakage, and requested Minor findings from the
Round 1 review were reproduced and addressed. No Task 6 experiment, search, evaluator
trial, cache preparation, threshold, seed, defender, generator, or policy was executed or
retuned.

### Final architecture after Round 1

- Feedback authority belongs only to `ScenarioBundle.feedback`. `AttackerPolicy` has no
  disclosure switch. The worker wire carries the candidate and only declared public
  action/reason/value fields; it carries no objective, model score, hidden validity,
  detailed reason, evaluator digest, or evaluator artifact reference.
- The worker remains a closed built-in-policy API. Resource limits are installed by the
  child before package/numeric imports. The parent begins the wall deadline before spawn,
  performs nonblocking stdin/stdout/stderr handling in one selector/watchdog loop, caps
  request/output bytes and RSS, and kills the process group on every boundary failure.
  Real `execv`, `fork`, `posix_spawn`, signal, native loader, socket, filesystem, hidden
  import, and reflection attempts are exercised and denied.
- Agentic evaluation and final production replay share one Task 7-owned deterministic
  public `TrustVerifier` construction. The selected evaluation event digest must equal the
  final frozen production event digest. The independent hidden corpus remains a separate
  restricted input artifact.
- Hidden generation uses only a locally constructed domain-separated NumPy generator.
  Hidden lifecycle validation independently replays explicit accounts, opening balances,
  fees, holds, frozen funds, chargeback accounts, and all declared card/A2A/agentic
  terminal paths.
- Artifact, signer, private evidence, and run-index access is rooted in stable no-follow
  directory descriptors. Files are owned mode-0600 regular single-link objects; private
  directories are owned mode-0700. Publication is file-fsync plus native atomic exclusive
  no-replace rename plus directory-fsync.
- A tracked Task 6 result may be a regular owned single-link source at current mode 0600
  or a fresh-checkout mode 0644. Its historical Git mode and bytes are pinned separately,
  and its execution copy is atomically admitted into private mode-0600 state.
- Scenario compilation adds the validated `apar_run_binding_v1` envelope after the frozen
  compiler returns. Standalone execution requires exact threat family, scenario attacker
  mode, policy attacker mode, threat-card reference, and rail pairing.
- The internal signed manifest retains every restricted reference. The public API returns
  a separately typed allowlisted view containing only scenario, policy, population,
  feedback, and production-event references plus the boolean validity result; no
  restricted hash, size, path, signature, reason, or other low-entropy identifier crosses
  the route.
- G2 uses three matched adaptive steps and the real `LLMPlannerPolicy` over the exact
  pinned Task 6 replay cache. Its transport raises if called. The one-command gate invokes
  the exact Task 6 postcommit verification-only recomputation instead of trusting summary
  fields.

### Finding-to-test map

| Review finding | Production closure | Regression evidence |
| --- | --- | --- |
| Critical 1 — golden feedback bypass | Removed caller disclosure authority; scenario-owned filtering in runner and wire | API extra-field rejection, golden and decision-only wire-shape tests |
| Critical 2 — process escape | Audit/import denial for exec/fork/spawn/signal/native/network/filesystem plus closed built-ins | Real worker escape probe test |
| Critical 3 — unbounded startup | No `preexec_fn`; deadline before spawn; child-installed limits; one nonblocking selector/watchdog | no-preexec, 2 MiB backpressure hang, spawn-error, deadline, RSS tests |
| Critical 4 — agentic winner ignored | Same verifier/config for evaluation and final replay; exact selected/final event digest equality | fixed-versus-adaptive winner-change and separate-hidden-corpus test |
| Important 1 — generator RNG | Domain-separated local NumPy `Generator(PCG64)` | literal first hidden amount and deterministic rerun test |
| Important 2 — validity economics/lifecycles | Independent explicit account/fee ledger for every rail lifecycle | reversal/refund/chargeback/recovery/return/freeze, mutation, negative-gap tests |
| Important 3 — atomic signer/index | 0600 temporary, fsync, native no-replace publication, directory fsync | concurrency, crash, partial, short-write, symlink, hardlink, owner/mode tests |
| Important 4 — real G2 | Three-step matched histories; exact frozen Task 6 cache and verifier command | G2 history/parent/cache audit and one-command script |
| Important 5 — tracked 0600 impossible | Accept verified 0600/0644 tracked source; admit private 0600 copy | fresh-checkout-mode private-admission test |
| Important 6 — artifact/index root admission | Stable no-follow root descriptors; exact owner/mode/link checks | root symlink/path-swap/file-swap/owner/mode/link tests |
| Important 7 — family pairing | API compile envelope plus standalone family/mode/card/rail equality | same-rail cross-family API rejection and attacker-mode standalone rejection |
| Warning/leakage gap | Redacted `PublicRunManifest`; private completed-run detail interface | exact public allowlist and restricted identifier absence assertions |
| Minor run ID | Full `run-[0-9a-f]{32}` match before lookup | non-hex same-length ID test |
| Minor invalid event order | Oracle returns boolean false without log-domain exception | out-of-order/negative-gap test |

### Strict TDD evidence by finding

#### Critical 1 — feedback disclosure

RED against the reviewed implementation:

```text
$ .venv/bin/python -m pytest \
    tests/redteam/test_run_boundary.py::test_policy_selection_rejects_caller_controlled_feedback_disclosure \
    tests/redteam/test_run_boundary.py::test_worker_history_contains_only_scenario_declared_feedback \
    tests/api/test_runs.py::test_run_accepts_only_a_compiled_id_and_typed_policy_then_is_gettable -q
FFF                                                                      [100%]
3 failed in 5.48s
```

GREEN after removing policy disclosure authority and filtering both directions of the
wire from the compiled feedback declaration:

```text
$ <same command>
...                                                                      [100%]
3 passed in 3.03s
```

The golden case explicitly proves `realized_value` is sent only when declared; the
decision-only case contains only candidate and declared public feedback. The public API
rejects both `expose_realized_value` and arbitrary path/callable fields.

#### Critical 2 and Critical 3 — process and startup boundary

The first real-escape/startup tests were RED:

```text
$ .venv/bin/python -m pytest \
    tests/redteam/test_run_boundary.py::test_clean_policy_worker_blocks_real_process_and_native_escape_attempts \
    tests/redteam/test_run_boundary.py::test_policy_worker_spawn_has_no_parent_preexec_hook \
    tests/redteam/test_run_boundary.py::test_policy_worker_bounds_hung_startup_while_stdin_is_backpressured \
    tests/redteam/test_run_boundary.py::test_policy_worker_converts_spawn_failure_to_fail_closed_error -q
FFFF                                                                     [100%]
4 failed in 0.54s
```

GREEN after child-side limit installation and the bounded selector/watchdog launch path:

```text
$ <same command>
....                                                                     [100%]
4 passed in 0.59s
```

The existing real deadline/RSS tests additionally prove process-group termination above
805,306,368 resident bytes and termination when RSS observation is unavailable. A later
full-suite run exposed only a test-wrapper runtime-cast defect (`1 failed, 798 passed,
1 skipped`); the wrapper was corrected without production changes and the complete
required integration suite then passed at 799/1.

#### Critical 4 — selected agentic winner replay

The winner-specific regression initially failed because the event artifact still named
the independent hidden corpus. After production events were separated, a stronger digest
test was RED until evaluation and final replay used the same deterministic verifier:

```text
$ .venv/bin/python -m pytest \
    tests/redteam/test_run_boundary.py::test_agentic_final_artifact_replays_each_policy_winner_and_keeps_hidden_corpus_separate -q
F                                                                        [100%]
1 failed in 3.69s
```

The first exact-replay GREEN was:

```text
$ <same command>
.                                                                        [100%]
1 passed in 4.73s
```

Final strengthening uses action-only feedback so fixed and adaptive choose different
winners, asserts their production event hashes differ, asserts each selected evaluation
hash equals its frozen production hash, requires at least one approved production event,
and asserts the independent hidden-corpus hash remains identical:

```text
$ <same command>
.                                                                        [100%]
1 passed in 4.50s
```

#### Important 1 — independent NumPy RNG

The literal hand-derived stream assertion was RED against `random.Random`:

```text
$ .venv/bin/python -m pytest tests/redteam/test_hidden_validity.py \
    -q -k 'numpy_stream or independent_hidden_families'
F....                                                                    [100%]
E AssertionError: expected first amount Decimal('88.25'), got Decimal('119.36')
1 failed, 4 passed in 0.16s
```

GREEN after local SHA-256 domain separation into `numpy.random.PCG64`:

```text
$ <same command>
.....                                                                    [100%]
5 passed in 0.16s
```

No global RNG state is read or written.

#### Important 2 and invalid-order Minor — lifecycle economics

Initial explicit lifecycle/account/fee regressions were RED:

```text
$ .venv/bin/python -m pytest tests/redteam/test_hidden_validity.py -q \
    -k 'requires_independent_fee or reversal or cover_refund or negative_gaps or mutations'
FFFFF                                                                    [100%]
5 failed in 0.19s
```

The first lifecycle implementation was GREEN at `5 passed in 0.16s`. A subsequent fee
mutation test reproduced a missing economic-contract check (`1 failed in 0.16s`) and
passed after binding fee/accounts across each payment (`1 passed in 0.33s`). Return-path
coverage then reproduced one omitted A2A terminal path (`1 failed in 0.16s`). Final hidden
validity coverage:

```text
$ .venv/bin/python -m pytest tests/redteam/test_hidden_validity.py -q
..................                                                       [100%]
18 passed in 0.56s
```

Impossible transitions, out-of-order events, and negative gaps add reason codes and
return `valid=False`; they do not raise from the distance/log calculation.

#### Important 3 and Important 6 — atomic private storage and root admission

The new signer/index/artifact adversarial tests began with nine failures:

```text
$ .venv/bin/python -m pytest tests/redteam/test_run_boundary.py \
    tests/storage/test_artifacts.py -q \
    -k 'partial_key or short_low or symlink_parent or crash_safe or concurrent_publishers or symlink_or_permissive_artifact_root or stable_root_descriptor or owner_and_link or payload_swap'
FFFFFFFFF                                                                [100%]
9 failed in 0.37s
```

Initial component GREEN was seven artifact tests plus six signer/index tests. The first
combined stress pass found one real hard-link publication race (`1 failed, 80 passed`).
Replacing link/unlink publication with native `renameatx_np(RENAME_EXCL)` on Darwin and
`renameat2(RENAME_NOREPLACE)` on Linux made the race-focused set GREEN:

```text
$ .venv/bin/python -m pytest tests/redteam/test_run_boundary.py \
    tests/storage/test_artifacts.py -q -k 'concurrent or exclusive_rename'
...                                                                      [100%]
3 passed in 0.43s
```

Unsupported platforms and unsupported filesystems fail closed; there is no overwrite
fallback. All reads use stable directory/file descriptors and reject symlink, wrong
owner, wrong mode, unexpected hard link, metadata change, and non-regular object.

#### Important 4 — real G2 and exact Task 6 verifier

The multi-step assertions were RED against the one-step review baseline: observed
generation history was `[0]`, so no adaptive parent/history change was exercised, and no
authenticated frozen-cache source existed in the worker audit. The replacement uses a
matched budget of three for fixed, random, adaptive, and cached LLM policies. The cached
path constructs the production `LLMPlannerPolicy` with the exact pinned replay records,
`require_cached_replay=True`, and a client whose transport raises if called.

Focused frozen-cache GREEN:

```text
$ .venv/bin/python -m pytest \
    tests/redteam/test_run_boundary.py::test_cached_worker_uses_the_real_cache_only_planner_with_zero_network \
    tests/integration/test_g2_adaptation.py::test_g2_disposable_policies_use_matched_budgets_and_seeded_bytes -q
..                                                                       [100%]
2 passed in 12.12s
```

Every cached proposal records `cache_source=task6-v3-frozen-replay`, `cache_hit=True`,
and `network_call_count=0`. `scripts/verify_g1_g2.py` now invokes the exact existing
`run_task6_holdout.py --verify-postcommit` command with approved commit and SHA. The
clean-HEAD gate result is recorded in the postcommit subsection below.

#### Important 5 — tracked mode and private admission

The fresh-checkout regression was RED because no private-admission helper existed:

```text
$ .venv/bin/python -m pytest \
    tests/redteam/test_run_boundary.py::test_tracked_task6_evidence_is_atomically_admitted_to_private_state -q
F                                                                        [100%]
1 failed
```

GREEN covers both source modes and private publication:

```text
$ .venv/bin/python -m pytest tests/redteam/test_run_boundary.py -q \
    -k 'tracked_task6_evidence or durable_signer_never_publishes'
..                                                                       [100%]
2 passed
```

The runner never chmods the tracked result. It verifies regular/non-symlink ownership,
single-link state, bounded stable descriptor metadata, source mode 0600 or 0644, exact
SHA, historical Git mode 100644, and historical bytes before reading the atomic private
0600 copy.

#### Important 7 — complete reviewed pairing

Family-envelope and same-rail cross-family tests were RED before the compile extension:

```text
$ .venv/bin/python -m pytest \
    tests/api/test_runs.py::test_compile_returns_a_verified_scenario_artifact_id \
    tests/api/test_runs.py::test_run_rejects_same_rail_policy_from_a_different_reviewed_family -q
FF                                                                       [100%]
2 failed in 1.70s
```

GREEN after `apar_run_binding_v1` and runner validation:

```text
$ <same command>
..                                                                       [100%]
2 passed in 0.97s
```

A later standalone RED showed policy mode itself was not yet typed (`extra_forbidden` on
`attacker_mode`). `AttackerPolicy.attacker_mode` is now required and execution requires
binding mode == scenario mode == policy mode:

```text
$ .venv/bin/python -m pytest \
    tests/redteam/test_run_boundary.py::test_standalone_runner_rejects_policy_attacker_mode_outside_reviewed_binding -q
.                                                                        [100%]
1 passed in 0.19s
```

The frozen G0 compiler implementation and hash were not changed.

#### Warning/leakage gap — public response and restricted completion detail

The original API regression was RED while a full internal manifest, including the
restricted validity reference, was returned. The final test asserts an exact five-name
public artifact allowlist and verifies the internal restricted SHA and relative path are
absent from the entire serialized response. It also asserts no `restricted_validity`,
`restricted_evaluation`, or `reasons` token is present. `evaluation_hidden.__init__`
exports neither the restricted report nor a detailed evaluator. The caller-controlled
`run_complete=True` switch no longer exists; only the runner imports the module-private
completed-run function after proposal execution.

```text
$ .venv/bin/python -m pytest tests/api/test_runs.py -q
....                                                                     [100%]
4 passed
```

#### Full-regex run ID Minor

The exact-length non-hex ID test was deliberately run against the prefix/length-only
implementation and reached artifact lookup instead of rejecting the identifier:

```text
$ .venv/bin/python -m pytest \
    tests/redteam/test_run_boundary.py::test_get_rejects_non_hex_run_ids_before_any_index_lookup -q
F                                                                        [100%]
E apar.runs.runner.RunExecutionError: stored run manifest is invalid
1 failed in 0.25s
```

GREEN after the full regex:

```text
$ <same command>
.                                                                        [100%]
1 passed in 0.44s
```

#### Self-review compatibility defect — pre-existing `.apar` root

The first full repository run found that G0 can initialize its general database parent at
0755 before API startup. Keeping a key directly in that parent correctly failed closed,
but broke existing API startup:

```text
$ .venv/bin/python -m pytest -q
FF
FAILED tests/integration/test_g0_contract_flow.py::test_golden_threat_is_available_through_real_api
FAILED tests/integration/test_g0_contract_flow.py::test_one_command_g0_verification
2 failed, 911 passed, 1 skipped in 145.74s
```

Signer and index state now live beneath a dedicated owned mode-0700
`.apar/private-run-state` directory. The database root is neither chmodded nor trusted as
private:

```text
$ .venv/bin/python -m pytest \
    tests/integration/test_g0_contract_flow.py::test_golden_threat_is_available_through_real_api \
    tests/integration/test_g0_contract_flow.py::test_one_command_g0_verification \
    tests/api/test_runs.py -q
......                                                                   [100%]
6 passed in 6.47s
```

### Final focused and repository gates

Final review-focused set:

```text
$ .venv/bin/python -m pytest tests/redteam/test_run_boundary.py \
    tests/redteam/test_hidden_validity.py tests/storage/test_artifacts.py \
    tests/api/test_runs.py tests/integration/test_g2_adaptation.py -q
........................................................................ [ 86%]
...........                                                              [100%]
83 passed in 34.29s
```

Required integration set:

```text
$ .venv/bin/python -m pytest tests/simulator tests/trust tests/generators \
    tests/redteam tests/integration/test_g1_simulation.py \
    tests/integration/test_g2_adaptation.py -q
799 passed, 1 skipped in 158.59s (0:02:38)
```

Full repository:

```text
$ .venv/bin/python -m pytest -q
913 passed, 1 skipped in 147.53s (0:02:27)
```

Static and foundation gates:

```text
$ .venv/bin/ruff check src tests scripts
All checks passed!

$ .venv/bin/mypy --strict <18 exact Task 7 source/test targets>
Success: no issues found in 18 source files

$ .venv/bin/mypy src
Success: no issues found in 53 source files

$ .venv/bin/python scripts/verify_g0.py
G0 PASS: 20 threat cards, contracts, registry, compiler, API, and artifact store
```

Hygiene and frozen isolation all exited zero with no diff output:

```text
$ git diff --check
$ git diff --exit-code 7534c0e38eaff568f390784113a0dd5992f8d048 -- validation_spike
$ git diff --exit-code 7534c0e38eaff568f390784113a0dd5992f8d048 -- \
    docs/experiments/task6-v3.4-holdout-result.json \
    docs/experiments/task6-v3.4-holdout-preregistration.json \
    src/apar/redteam/policies.py src/apar/redteam/benchmark.py \
    src/apar/generators/campaigns.py src/apar/generators/population.py \
    src/apar/redteam/llm_policy.py src/apar/redteam/search.py \
    src/apar/redteam/task6_experiment.py scripts/run_task6_holdout.py
```

Pinned current hashes:

```text
f82981a987651a7f7ebb10a9011df063b2dc54a56181cae5b838e31de5e658db  docs/experiments/task6-v3.4-holdout-result.json
12bf24e081e97f3222bf1fc92fb1d441c36bba548184c6b503519590efc649a4  docs/experiments/task6-v3.4-holdout-preregistration.json
c97ab7b263a493978cf901140a97f15874a34f8ff2ce54c84253e7baa998fb82  src/apar/redteam/policies.py
7996fcf20c85547a861afcfeb9da132dad534ad30f7fe1a07618e90e2faef519  src/apar/redteam/benchmark.py
670b4a3ec358f82d88f9655bd41d878fbee11d4841ff264655554bae31c3b31a  src/apar/generators/campaigns.py
2e54862322980414098c17930ec95bd268372da8968a78384a5bd661bfdaa2e5  src/apar/generators/population.py
8105a6788041f7d73b1afa571482f4b0ff3b15980f6c28769b701ab350936622  src/apar/redteam/llm_policy.py
ee05348ab07a9852a68a3f6a477eeec7ad6837d94f187e6fbb97767220f60e89  src/apar/redteam/search.py
a1367a8bb4310eeea2812a7d118ccb738ae1d9c32bfbc21c87413b1a869ce056  src/apar/redteam/task6_experiment.py
1b137c2de0bec6eb95acf34172217113c825ab5be9322e694f9ec9e458427efc  docs/experiments/task6-v3-cached-llm-replay.json
```

Historical result mode remains `100644 blob` at approved commit
`d6d3eecbfe2d871af8375e1455814cb5c48f2928`.

### Files changed in Fix Round 1

- `scripts/verify_g1_g2.py`
- `src/apar/api/app.py`
- `src/apar/api/routes/runs.py`
- `src/apar/api/routes/scenarios.py`
- `src/apar/evaluation_hidden/__init__.py`
- `src/apar/evaluation_hidden/generator.py`
- `src/apar/evaluation_hidden/validity.py`
- `src/apar/runs/__init__.py`
- `src/apar/runs/agentic_replay.py` (new)
- `src/apar/runs/policy_worker.py`
- `src/apar/runs/runner.py`
- `src/apar/runs/wire.py`
- `src/apar/storage/artifacts.py`
- `tests/api/test_runs.py`
- `tests/integration/test_g2_adaptation.py`
- `tests/redteam/test_hidden_validity.py`
- `tests/redteam/test_run_boundary.py`
- `tests/storage/test_artifacts.py`
- this report

### Self-review and residual scope

The complete diff was reviewed for hidden imports, worker request/response fields,
retained objects, subprocess setup, selector termination, platform rename behavior,
descriptor lifetime, source/private file modes, hard links, API response fields,
manifest/receipt name sets, scenario/policy pairing, agentic event ownership, and Task 6
frozen-path drift. The review itself found and fixed the concurrent hard-link publication
race, the exact run-ID validation gap, the fixed/random agentic test case that selected an
identical winner, the test-wrapper monkeypatch cast, and the pre-existing 0755 `.apar`
compatibility defect.

Residual boundary: native exclusive publication and RSS observation are implemented for
Darwin and Linux. An unsupported platform or filesystem fails closed. The worker remains
intentionally limited to the four reviewed built-in selections; it is not an arbitrary
uploaded-code sandbox. The Task 6 result is accepted synthetic frozen evidence only and
does not establish live-fraud effectiveness.

### Fix Round 1 commits and clean-HEAD postcommit gates

Implementation/report commit: `PENDING_FIX_ROUND_1_IMPLEMENTATION_COMMIT`

Postcommit verification-report commit: `PENDING_FIX_ROUND_1_REPORT_COMMIT`

The following outputs are populated only after the implementation commit, from a clean
tracked HEAD:

```text
$ .venv/bin/python scripts/verify_g1_g2.py
PENDING_CLEAN_HEAD_G1_G2_OUTPUT

$ .venv/bin/python scripts/run_task6_holdout.py --verify-postcommit \
    --approved-result-commit d6d3eecbfe2d871af8375e1455814cb5c48f2928 \
    --approved-result-sha256 f82981a987651a7f7ebb10a9011df063b2dc54a56181cae5b838e31de5e658db
PENDING_CLEAN_HEAD_TASK6_OUTPUT
```
