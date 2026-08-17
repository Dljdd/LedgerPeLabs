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
G2 checks. The final repository suite is `882 passed, 1 skipped`. The accepted Task 6
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

It uses Python's independent `random.Random`, hidden entity motifs, a separate causal
schedule, independent beta-distributed leaves, UUID5 identities, and no production
generator or defender import. It returns only `tuple[PaymentEvent, ...]`, as required by
the Task 7 ruling.

`HiddenValidityOracle.evaluate(...)` returns a frozen model with exactly one field,
`valid: bool`. Detailed stable reason codes and metrics exist only in
`evaluate_restricted(..., run_complete=True)`. `RunRunner` calls that method only after
policy execution and stores the result in a restricted artifact. The attacker worker
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
only family, policy kind, bounded query budget, bounded worker timeout, and the declared
realized-value disclosure flag. Pydantic extra-field rejection prevents paths and
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
running arbitrary uploaded Python code. The cached-LLM selection in local Task 7
orchestration is a deterministic zero-network fixture. The empirical cached-LLM claim is
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
- compiled scenario.

After the run, it stores and signs a completion receipt over:

- production/hidden evaluation events;
- public feedback history;
- restricted production evaluation audit;
- restricted validity report; and
- summary.

The completion receipt contains the authorization receipt's artifact digest. The final
manifest contains exactly those ten input/output artifacts plus both receipts: twelve
immutable references. Its signed lineage digest includes the complete named artifact
map and both receipt references.

Verification requires the configured durable signer, exact twelve-name set, immutable
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
- current mode `0600`, regular and non-symlink;
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

For agentic attacks, `CampaignBenchmark` performs the evaluator-owned Task 5 generation
and fresh real `AgenticRailAdapter`/`TrustVerifier` replay for every proposal; its
restricted audit freezes the production event/ledger digests and exact counts. Because
the public Task 5 command API intentionally does not expose its evaluator-only verifier
fixture, the manifest's independent validity corpus is then produced by
`HiddenCampaignGenerator`. The summary records this source explicitly. No private Task 5
engine state or evaluator fixture is accessed.

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
returns a typed signed `RunManifest`. Run retrieval returns the same reverified manifest.
Restricted artifact references may appear in signed lineage, but no route returns their
payloads or raw hidden reasons. Execution errors use a generic structured message and do
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
PENDING_CLEAN_POSTCOMMIT_VERIFICATION
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
