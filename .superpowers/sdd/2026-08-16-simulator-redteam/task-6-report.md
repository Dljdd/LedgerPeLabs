# Task 6 implementation report

## Outcome after fix round 1

Task 6 now provides fixed, random, adaptive non-LLM, and optional cached-LLM
attackers behind a closed public projection. The policy boundary contains only finite
public adaptive vectors, coarse feedback, visible history, and a local NumPy generator.
It contains no `CampaignParams`, hidden template, evaluator, generator evidence, trust
fixture, model feature, score, threshold, label, mutation reason, dependency, hidden
failure, or future outcome.

The evaluator-owned `CampaignBenchmark`, which is deliberately not imported by
`apar.redteam`, composes a public vector into a hidden Task 5 template. It then generates
Task 5 commands, performs a fresh rail and ledger replay, derives observable features
from commands, schedules, events, and graph roles, applies a frozen policy-independent
defender, and derives net settled illicit value from executed events and independently
bound population roles.

## Security-driven interface ruling

The plan sample declared `AttackCandidate.params: CampaignParams`. That type exposes the
hidden template, campaign ID, seed, expected motif, class/value targets, query budget, and
other evaluator-owned inputs. Task 6 therefore changes the public type to
`AttackCandidate.params: AdaptiveVector` and binds it to a sanitized `ParameterBounds`.
The evaluator alone owns composition back into `CampaignParams`.

This projection adds a downstream Task 7 adapter responsibility: the isolated runner must
send only the canonical public bounds/vector/history documents to a policy worker, retain
the bound hidden template and evaluation contract in the evaluator process, and compose
only after accepting and reconstructing the public response. Serializing `CampaignParams`
to the policy worker would violate this ruling.

## TDD evidence for the integrity correction

Fix round 1 began with 13 failing reproductions covering the critical synthetic-outcome
benchmark and the seven important plus three minor review findings. Separate deadline
tests were captured red before wall-time implementation. After the public API migration,
the capability provenance tests remained red in exactly three cases: tampered background,
evaluator, and disclosure digests were initially accepted because only the aggregate
contract digest was compared. The report now compares every bound provenance component.

The final red-team suite has 86 tests. It includes multiple seeds; zero, one, ordinary,
and 1,000-proposal budgets; exact bounds/domain exhaustion; order metamorphics; global RNG
isolation; evaluator exceptions; duplicate candidates; lineage attacks; slow policy and
evaluator clocks; LLM schema/prototype/subclass/non-finite attacks; cache digest mutation;
zero-network replay; matched wall/discrete budgets; defender permutation; negative/no
delta; and provenance-cell swaps.

## Closed public bounds

`ParameterBounds` contains exactly `family`, public defaults, public domains, and the
mutually feasible vector lattice. All contracts are frozen, exact-field, exact-type, and
canonical-document validated. A policy-facing bounds object has no hidden template or
reconstruction capability. Search reconstructs caller-supplied bounds, every returned
candidate, and every externally supplied visible contract before use; integrity seals
reject `model_copy(update=...)`, extra attributes, mutable aliases, and subclasses at the
boundary.

The evaluator preflights its advertised lattice with Task 5 before exposing it. For the
reviewed Task 5 fixtures the active spaces are:

| Family | Public adaptive fields | Feasible vectors |
| --- | ---: | ---: |
| APP scam/mule | 5 | 54 |
| Card testing CNP | 3 | 54 |
| Synthetic merchant refund | 1 | 6 |
| Agentic intent abuse, minimum 25-payment matrix | 0 | 1 |

Every categorical/discrete value and every numeric boundary is represented by at least one
feasible vector. Coupled APP recovery/fan-out and delay/strategy constraints are enforced
before evaluation. The minimum agentic matrix advertises no false mutation slot. A hidden
motif/family mismatch fails benchmark construction.

Candidate IDs and vector fingerprints are stable SHA-256 values over canonical documents.
Root proposals require `generation=0` and `parent_id=None`; subsequent proposals require
the exact visible-history generation and an actual visible parent ID.

## Policies and feedback

`Feedback` exposes exactly `action`, `reason_family`, and `realized_value`. Reasons come
from a coarse public allowlist and never contain a Task 4/5 hidden failure. Search erases
realized value unless a scenario-owned immutable `DisclosureProfile`, bound into the
`EvaluationContract`, enables it.

- Fixed returns the declared public default.
- Random samples the finite categorical/discrete/linear/log domains with a local RNG and
  then admits only an exact feasible vector.
- Adaptive uses a genuine sampled tournament over visible trials, without inserting the
  global best. Any sampled contender can win. It mutates one to three declared fields,
  uses only coarse visible feedback, and never clips or manufactures aliases.
- All policies preserve stable ordering semantics and leave global RNG state unchanged.

## Search, provenance, and deadlines

Every proposal receives one evaluation attempt, including an evaluator exception that is
collapsed to coarse `evaluation_failure`. Proposal, query, and logical-time budgets are
identical and explicitly recorded. Search results bind:

- family and bounds digest;
- hidden-template, background, population, evaluator, defender, and disclosure-profile
  digests;
- aggregate evaluation-contract digest;
- exact policy identity/version and seed;
- proposal/query/logical-time/wall-time budgets and usage;
- elapsed wall time, exhaustion, and overrun;
- every proposal, visible feedback, objective, and deterministic winner.

A real monotonic deadline includes policy proposal, LLM transport/cache work, and
evaluation time. In the trusted in-process Task 6 boundary, a callback that returns after
the deadline is charged and prevents further calls. Task 6 cannot preempt or kill a
hostile or indefinitely hanging Python callback; process termination and resource/network
controls remain a Task 7 responsibility.

Capability reporting rejects cross-family, cross-bounds, cross-template,
cross-background, cross-population, cross-evaluator, cross-defender, and cross-disclosure
cells. Metric counts validate `approved <= proposals`, campaign scale is derived from
approved trials, and valid yield, deltas, support, supported-family count, and the adaptive
claim are derived properties rather than caller-supplied labels.

## LLM behavior and audit

The optional planner receives the same public bounds and visible history as other policies.
Generated JSON has exactly `params`, `parent_id`, and `generation`, exact recursive built-in
string keys, strict finite JSON values, canonical decimal strings, exact field presence,
an actual visible parent, and an in-lattice vector. Prototype-like keys, subclasses,
missing/extra fields, score/reasoning fields, and out-of-bounds values reject.

Provider and model ID are pinned at planner construction and cache verification; later
client mutation cannot forge attribution. Every online or cache proposal attempt,
including transport, schema, bounds, lineage, and cache failures, emits a digest-only
evaluator-owned `LLMAuditRecord` with provider, model ID, policy/schema/prompt/response
digests, validated latency, token counts, call status, failure family, and cache-hit state.
No raw prompt, response, secret, or reasoning trace is exported. Required cached replay
performs zero network calls.

## Preregistered empirical capability delta

Before trials, the benchmark fixes:

- seeds `(4, 9)`;
- 8 proposals, queries, and logical-time units per policy and seed;
- a 60,000 ms wall budget in every cell;
- APP scam/mule and card testing CNP evaluator contracts;
- valid yield as the primary outcome;
- a minimum adaptive-minus-random delta of `0.10` for each family;
- frozen defender rules over artifact-derived APP distinct-payee and card declined-probe
  features.

The observed deterministic replay-backed results are:

| Family | Fixed yield | Random yield | Adaptive yield | Cached LLM yield | Delta | Supported |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| APP scam/mule | 1.0000 | 0.5625 | 0.5000 | 1.0000 | -0.0625 | no |
| Card testing CNP | 0.0000 | 0.0625 | 0.3750 | 0.0000 | 0.3125 | yes |

The honest `supported_family_count` is **1**, not the plan's hoped-for 2. No threshold,
outcome, seed, or label was retuned after observation. This is an exposed empirical failure
of the two-family capability target and should guide the next architecture iteration.

Aggregate role-bound settled illicit value was `3068.36` for adaptive versus `2098.06`
for random, so the independently derived `adaptive_claim` is `supported`. The claim is
about this exact replay-backed preregistration, not live fraud effectiveness. The
minimum-agentic singleton is a preregistered negative control: all policies observe zero
delta, zero supported families, and `adaptive_claim='not_supported'`. Reversing defender
rule declaration order leaves defender/evaluation digests and every APP result unchanged.

## Isolation boundary and Task 7 responsibility

Static AST tests reject evaluator/rail/trust imports, aliases, and dynamic
`eval`/`exec`/`import_module`/`__import__` patterns in policy-facing modules. A fresh-process
runtime sentinel proves importing `apar.redteam` does not load the evaluator benchmark,
generators, rail internals, or trust verifier. Search never passes or retains the evaluator
callable in a policy object.

These controls establish a closed ordinary-code capability boundary inside one trusted
Python process. They do not sandbox a deliberately hostile policy using frame inspection,
native extensions, debugger/process-global reflection, or a non-returning callback. Task 7
must add process isolation, narrow serialization, restricted imports/object graphs,
resource and network controls, timeout termination, and teardown before making a hostile
code isolation claim.

## Verification

- Task 6 red-team suite: `86 passed`.
- Simulator/trust/generator/red-team integration: `610 passed`.
- Full repository pytest: `714 passed`.
- Ruff over `src`, `tests`, and `scripts`: PASS.
- Strict mypy over `src/apar/redteam`: PASS.
- Project mypy over 42 source files: PASS.
- G0 verifier: PASS for 20 threat cards and reviewed contract flow.
- `git diff --check`: PASS.
- `validation_spike/`: unchanged from `6335bd9`.

## Deliberate tradeoffs

- Public adaptation uses finite canonical grids. This enables exact replay, feasibility,
  identity, and LLM validation but does not explore a continuous attack space.
- Task 5 preflight and fresh rail replay are more expensive than a toy closure. They are
  retained because capability evidence must come from executable artifacts and ledgers.
- The current coarse-reason tournament improves the card benchmark but underperforms
  random APP valid yield. The result is retained as evidence rather than hidden by a new
  post-hoc rule.
- In-process deadline enforcement can stop only after a slow callback returns. Task 7 must
  supply preemptive worker termination.

## Fix round 2: executable authority and authentic results

Fix round 2 began with 11 focused failing reproductions for the two critical, two
important, and one minor review findings. The implementation now has a single
process-local `SearchAuthority` that explicitly registers exact evaluator and policy
instances. An `EvaluatorCapability` binds the reconstructed public bounds, immutable
evaluation contract, exact bound evaluator method, owner implementation digest, and a
dependency digest covering the campaign generator, benchmark, replay engine, rails,
ledger, and trust verifier. A `PolicyCapability` binds the exact registered instance,
type implementation digest, and evaluator-assigned name/version. Search accepts only
those capabilities and invokes no caller-supplied evaluation callback.

Every issued search result contains the authority, evaluator capability/code, policy
capability/code, and run-group bindings. The authority deep-canonicalizes and HMAC-seals
the complete visible result, including nested proposals, feedback, objectives, winner,
usage, deadlines, and provenance, then retains exact-object issuance identity. Reporting
rejects copied or reconstructed results, `model_copy` relabeling, nested trial mutation,
synthetic results, cross-authority cells, and cross-run cells. The process-local HMAC is
appropriate for the trusted Task 6 process; Task 7 remains responsible for durable
cross-process signatures, artifact verification, worker isolation, and hostile-process
termination.

The monotonic deadline now marks exhaustion whenever actual elapsed time reaches or
exceeds the configured deadline, including a final proposal that consumes the complete
discrete budget. Exact fake-clock tests cover a 5 ms deadline reached at 5 ms and overrun
at 6 ms. Capability comparison requires equal configured budgets, equal actual
proposal/query/logical usage, equal disclosure, and no exhausted or overrun cell. It
rejects unequal opportunities instead of emitting a support claim.

Role-bound value accounting now tracks outstanding principal by payment lifecycle.
Settlement/posting adds outstanding principal, while refund, return, chargeback, and
recovery remove at most the remaining principal. Tests execute real card and A2A rail
lifecycle commands and prove chargeback followed by recovery cannot double-subtract the
same payment while a second unrecovered payment remains counted.

## Generic adaptive policy and development evidence

The old adaptive helper contained family-specific mappings from public reason families
to `retry_intensity`, `mule_fanout`, and `cash_out_fraction`. Version 2.0.0 removes those
mappings. It is a family-agnostic novelty-aware contextual UCB/tournament policy that:

- samples a real parent tournament, without forcing the global best into the sample;
- derives mutation directions only from declared public domain positions;
- learns public child-minus-parent objective gains by the parent's coarse reason context;
- gives exploration bonuses to unobserved directions and unseen vectors; and
- lets repeated negative direction gains suppress those directions.

It receives only the existing closed `VisibleTrial`, `Feedback`, `ParameterBounds`, and
local RNG contracts. Behavioral and static tests exercise arbitrary parameter names and
exclude family field names, hidden features, scores, gradients, defender rules, and
evaluator objects from the policy implementation.

On the already-used development seeds `(4, 9)` at budget 8, before freezing the holdout,
the generic policy produced:

| Family | Random primary result | Adaptive primary result | Preregistered-form delta |
| --- | ---: | ---: | ---: |
| APP scam/mule | net value `1860.00` | net value `2330.00` | `+25.2688%` |
| Card testing CNP | yield `1/16` | yield `3/16` | `+0.125` |

This was development evidence only. It finalized the generic policy; it was not labeled
as holdout evidence.

## Frozen holdout preregistration

Commit `10bb4c4c9634b1c5f3b0cc1deb8554c9b22d2547` precedes every holdout trial and contains
the finalized policy, integrity corrections, tests, preregistered runner, cached replay, and
machine-readable preregistration. The preregistration binds source and registered code
digests, exact evaluator contracts, bounds/template/background/population/evaluator/
defender/disclosure digests, untouched seeds `(101, 211, 307, 401)`, budget 16, equal
60,000 ms wall budgets, and exactly one primary outcome per target family:

- APP: adaptive role-bound net settled value improvement rate over random, minimum
  `0.10`;
- card: adaptive valid-yield minus random valid-yield, minimum `0.10`.

The minimum-agentic singleton remains the bound negative/no-delta control. The cached LLM
replay was prepared only on development seed 4, has no holdout-seed overlap, and is frozen
by file digest. The runner checked a clean preregistration tree, source/provenance/cache
digests, cache hits, and zero network calls. Its local result-file check was only a
procedural accidental-rerun guard, not durable exactly-once enforcement.

## Recorded untouched holdout result

One procedural execution is recorded from `10bb4c4` by commit chronology and its canonical
artifact; no policy, seed, budget, metric, threshold, benchmark, defender, or test was
changed after observing it. This evidence does not cryptographically exclude an unrecorded
local invocation. Append-only storage, an external CI receipt, durable signatures, and
cross-process verification remain Task 7 responsibilities. The recorded artifact SHA-256 is
`d73aa1fe59177e4968991383a7c7e648b688f25a2d0ba55d68adfd6b90271ae6`.

| Family | Random | Adaptive | Observed primary delta | Threshold | Supported |
| --- | ---: | ---: | ---: | ---: | --- |
| APP scam/mule net value | `3170.00` | `7390.00` | `+1.3312302839` (133.12%) | `0.10` | yes |
| Card testing CNP yield | `8/64 = 0.125` | `13/64 = 0.203125` | `+0.078125` | `0.10` | no |

All configured and actual proposal/query/logical budgets matched, no deadline exhausted or
overran, and all 128 cached-LLM attempts were cache successes with zero network calls.
Aggregate adaptive net value was `10484.78` versus random `5074.48`, so the derived
`adaptive_claim` is `supported` for aggregate value. The family acceptance result is
nevertheless **`supported_family_count = 1`**. The required two-family Task 6 criterion is
therefore **unmet**. Card improved materially but missed the fixed threshold; the result
is retained without metric switching or post-hoc tuning.

## Fix round 2 verification

- Focused Task 6 red-team suite: `100 passed` before the holdout.
- Simulator/trust/generator/red-team/G0-flow integration: `629 passed`.
- Full repository pytest: `728 passed`.
- Ruff over `src`, `tests`, and `scripts`: PASS.
- Strict mypy over `src/apar/redteam`: PASS.
- Project mypy over 41 source files: PASS.
- G0 verifier: PASS for 20 threat cards and reviewed contract flow.
- Frozen holdout binding verification: PASS without executing a trial before commit.
- `git diff --check`: PASS.
- `validation_spike/`: unchanged from `4d3c8f5`.

## Fix round 3 Phase A: exact callable execution

Rereview found that v2 registration hashed `policy.propose`, but execution later performed
a fresh `_policy.propose` lookup. An instance or class attribute replacement after
registration could therefore redirect execution away from the hashed method. Phase A
captured instance replacement, class replacement, and stored-callable tampering as RED
reproductions before changing the authority.

`PolicyCapability` now retains the exact bound callable observed at trusted registration,
its bound-callable implementation digest, the broader registered type implementation
digest, and the exact policy instance. `SearchAuthority` validates exact issuance,
owner/callable binding, and the callable digest, then invokes only that stored callable.
It never performs a dynamic `policy.propose` lookup during a search. Search results and
preregistration policy bindings include both implementation and callable digests. Copied
and cross-authority capabilities remain rejected. Documented policy state can still
mutate internally, so the cached LLM can append digest-only audit records without changing
its executable binding.

## V2 execution-record correction

The v2 result remains permanent at commit `dd55570` with artifact SHA-256
`d73aa1fe59177e4968991383a7c7e648b688f25a2d0ba55d68adfd6b90271ae6`. Its accurate
claim is that one procedural execution is recorded by preregistration/result commit
chronology and the canonical artifact. A local result-file absence check cannot
cryptographically enforce or prove exactly-once execution and cannot exclude an
unrecorded invocation. Task 7 owns append-only storage, an external CI/execution receipt,
durable signatures, cross-process verification, and hostile worker isolation.

The current runner is the v3 runner; the historical v2 runner is preserved at commit
`10bb4c4`. Direct `.venv/bin/python scripts/run_task6_holdout.py --verify-only` invocation
now bootstraps the production source path and imports no test fixtures. Production module
`apar.redteam.task6_experiment` reconstructs the exact Task 6 population and evaluator
contracts from a bound reviewed fixture.

## Final v3 generic frontier/UCB policy

V3 retains no family name, Task 5 field name, defender feature/threshold, v2 outcome
constant, hidden reason, or confirmatory seed. Static AST checks enforce that boundary.
Random and adaptive both use the public default as proposal zero, as preregistered for
fairness. Adaptive then:

- deterministically covers unique feasible one-field default-to-domain-boundary vectors;
- avoids duplicate vectors whenever an unseen feasible mutation exists;
- prefers unseen one-field directions before broader lattice moves;
- learns public parent-child objective deltas in the parent's coarse reason context;
- applies novelty/UCB exploration and repeated-bad-direction decay; and
- uses the existing genuine sampled tournament for later exploitation.

After implementation was fixed, it was evaluated only on the already-open v2 seeds
`(101, 211, 307, 401)` as development data at budget 24. APP adaptive value was `8870.00`
versus random `4720.00`; card adaptive yield was `27/96 = 0.28125` versus random
`14/96 = 0.145833...`. These observations are permanently development evidence, not v3
confirmatory evidence, and no further policy tuning followed them.

## Final v3 preregistration — result intentionally absent

`docs/experiments/task6-v3-holdout-preregistration.json` freezes:

- untouched seeds `(503, 607, 709, 811, 907, 1009, 1103, 1201)`;
- equal proposal/query/logical caps at budget 24 and equal wall-time caps at 120,000 ms;
- a maximum of one additional confirmatory attempt;
- zero network calls and a development-seed-only cached LLM replay;
- unchanged APP relative role-bound value improvement `>= 0.10`;
- unchanged card valid-yield delta `>= 0.10`;
- shared random/adaptive default proposal and otherwise identical opportunities;
- the negative minimum-agentic no-delta control;
- per-seed deltas and an exact paired sign-resampling reference interval as descriptive
  uncertainty only, never a post-hoc gate;
- exact policy code/callable, evaluator, bounds, template, background, population,
  defender, disclosure, runner, `llm_policy.py`, fixture, and source digests; and
- Python 3.12.13, CPython/cache tag, platform, pyproject digest, explicit absence of a
  lockfile, and exact installed dependency versions.

If v3 fails, no further confirmatory holdout will be opened; later work is exploratory or
Task 7 evaluation. The runner requires the explicit `--execute-confirmatory` flag, while
`--verify-only` validates every frozen binding without calling `AdaptiveSearch`.

Phase A did **not** execute a v3 holdout. The file
`docs/experiments/task6-v3-holdout-result.json` is absent, and none of the eight v3 seeds
was used in a policy/evaluator trial.

## Fix round 3 Phase A verification

- Callable substitution/tampering reproductions: `3 passed`.
- Complete Task 6 red-team suite: `109 passed`.
- Full repository pytest: `737 passed`.
- Ruff over `src`, `tests`, and `scripts`: PASS.
- Strict mypy over seven Task 6 source/runner files: PASS.
- Project mypy over 42 source files: PASS.
- G0 verifier: PASS for 20 threat cards and reviewed contract flow.
- Direct v3 verify-only invocation without `PYTHONPATH`: PASS.
- V3 result absence assertion: PASS.
- `git diff --check`: PASS.
- `validation_spike/`: unchanged from `dd55570`.

## Fix round 3 Phase B: canceled v3 freeze

Independent review after `cbeaeea` found that the public policy capability still carried
the executable, policy instance, and authoritative digests; its runtime check covered the
stored `propose` function but not every helper and module-global dependency. It also found
that the runner used a check followed by overwrite-capable `write_text`, and declared a
negative control that the confirmatory execution path never ran.

The canonical `docs/experiments/task6-v3-cancellation.json` record therefore cancels the
`cbeaeea` preregistration before execution. It preserves the original preregistration and
history, binds its SHA-256, records the review findings, and states that no v3 result was
created, the execute flag was not invoked, and none of its eight seeds was used. The
replacement uses distinct `task6-v3.1-holdout-preregistration.json` and
`task6-v3.1-holdout-result.json` paths.

## Authority-private policy execution

Phase B first captured nine RED reproductions spanning the opaque-handle boundary,
instance/class/global helper substitution, atomic publication, the missing negative
control, and missing cancellation/freeze artifacts. `PolicyCapability` now contains only
an opaque capability nonce. A separate immutable authority-private record retains the
exact issued handle identity, registered owner/type, stored bound callable, registration
metadata, runtime implementation digest, callable digest, and instance references.

Every proposal revalidates this private record. Its runtime digest covers all exact Python
methods on the registered policy class hierarchy, recursively referenced module-global
helper functions and immutable constants, callable instance slots, the LLM client's exact
object/type/transport callable, pinned provider/model values, and registered name/version.
Copies, synthetic handles, subclasses, cross-authority handles, coupled nonce tampering,
and instance/class/global dependency replacement reject before execution. Mutable LLM
audit and replay-cache contents remain usable only through their pinned original
containers; transport or pinned-client substitution rejects. Search result and
preregistration provenance is derived only from the authority-private record.

## Atomic exclusive result publication

The runner serializes the complete result to a same-directory temporary file, flushes and
fsyncs it, then publishes through an atomic hard link that fails if the target exists. It
removes the temporary link and fsyncs the directory where supported. A race regression
creates sentinel result bytes after the precheck but before publication: publication
raises `FileExistsError`, the sentinel bytes remain unchanged, and the temporary file is
cleaned. The pre-existing-result refusal remains as an early guard; the atomic link is the
actual local no-replace guarantee. Task 7 still owns durable append-only external receipts.

## Confirmatory negative control

The evaluator-owned experiment now includes the real minimum-agentic Task 5 family with a
singleton public adaptive space and hidden realized value. The v3.1 runner executes fixed,
random, and adaptive cells over the same confirmatory seeds, proposal/query/logical budget
24, and wall-time cap as the target cells. It derives matched-budget status and observed
valid-yield delta from authority-issued results, records zero network calls and sealed
result bindings, and marks this control as excluded from target `supported_family_count`.
A development-seed regression observes the preregistered zero delta without opening a
confirmatory seed.

## Final v3.1 freeze — results intentionally absent

V3.1 retains the exact v3 policy algorithm, untouched seeds
`(503, 607, 709, 811, 907, 1009, 1103, 1201)`, budget 24, 120,000 ms wall cap, zero-network
cached replay, and unchanged target gates: APP role-bound net settled value relative
improvement `>= 0.10` and card valid-yield delta `>= 0.10`. It binds the cancellation,
complete source/environment/cache/runtime-policy digests, all target and control evaluator
provenance, descriptive-only paired uncertainty, and the rule that no further confirmatory
holdout opens if v3.1 fails.

Phase B did not execute either confirmatory experiment. Both
`docs/experiments/task6-v3-holdout-result.json` and
`docs/experiments/task6-v3.1-holdout-result.json` are absent. Direct `--verify-only`
reconstructs and checks the complete v3.1 freeze without invoking `AdaptiveSearch`.

## Fix round 3 Phase B verification

- Initial review reproduction: `9 failed, 1 passed` before implementation.
- Authority, publication, control, cancellation, and freeze regressions: `11 passed`.
- Complete Task 6 red-team suite: `120 passed`.
- Full repository pytest: `748 passed`.
- Ruff over `src`, `tests`, and `scripts`: PASS.
- Strict mypy over seven Task 6 source/runner files: PASS.
- Project mypy over 42 source files: PASS.
- G0 verifier: PASS for 20 threat cards and reviewed contract flow.
- Direct v3.1 verify-only invocation: PASS.
- Canceled-v3 and v3.1 result absence assertions: PASS.
- `git diff --check`: PASS.
- `validation_spike/`: unchanged from `cbeaeea`.

## Fix round 3 Phase C: canceled v3.1 and source freeze

Independent review canceled the `6ad59cd` v3.1 preregistration before execution. The
canonical `task6-v3.1-cancellation.json` records the exact preregistration commit and file
digest, absent result, uninvoked confirmatory path, unused reserved seeds, review findings,
and distinct v3.2 replacement paths. The original v3 and v3.1 preregistrations remain in
history; neither result exists.

The policy authority now snapshots the exact referenced module objects plus referenced
behavior attributes, recursive Python globals, function code/defaults/keyword defaults/
closure cells, every policy class/MRO callable and immutable class datum, callable-object
type code and state, exact instance slots, registered metadata, and pinned LLM client
identity. Registration keeps exact module references in its private binding. Before each
proposal, it rederives the runtime document and rejects module replacement (including a
same-named fake), same-object module-attribute mutation, class/helper/global/default/
closure/callable-state mutation, or capability substitution before policy behavior runs.
The stored bound callable remains the only executable entrypoint.

This is layered mutation detection in a trusted Python process, not a hostile-code sandbox
and not a claim of mathematical completeness over every possible Python behavior. Task 7
still owns a clean policy worker, process isolation/termination, durable signatures, and
append-only execution receipts. V3.2 adds a complementary clean-process/full-tree control:
its preregistration must point to a preceding source commit and bind that commit's Git tree
and canonical manifest of every tracked `src/**/*.py`, Python runner/verification script,
fixture, cached replay/config/cancellation input, pyproject, explicit lockfile absence,
Python/platform identity, and the complete installed-distribution freeze.

Result publication continues to use a same-directory fsynced temporary file and atomic
exclusive hard-link publication. Directory fsync now suppresses only explicitly unsupported
errors (`EINVAL`/platform `ENOTSUP`); `EIO` and other failures propagate as a recovery error
that states the target is already published and must be inspected before retry. The target
bytes are never overwritten and the temporary link is cleaned.

The confirmatory result derives every support field through one hard gate. Exact target
cells, matched non-exhausted budgets, zero network calls, and the matched zero-delta/
unsupported negative control are all mandatory. If any condition fails, `confirmatory_valid`
and `criterion_met` are false, the reported supported-family count becomes zero, every
family support field is false, and `adaptive_claim` is `not_supported`.

Phase C uses two commits. The first is this source state and intentionally contains no v3.2
preregistration or result. A later preregistration-only commit may bind this exact source
commit. The runner supports source-only verification without constructing `AdaptiveSearch`;
after the preregistration exists, direct `--verify-only` must prove source ancestry, clean
worktree, exact current file hashes, source-tree/manifest/environment bindings, evaluator
and policy provenance, cancellation records, and result absence. Only the explicit
`--execute-confirmatory` mode can open the reserved seeds.

## Fix round 3 Phase C source-stage verification

- Initial Phase C regressions: `14 failed, 1 passed` before implementation.
- Phase C dependency, durability, gate, cancellation, and source tests: `19 passed`.
- Complete Task 6 red-team suite: `139 passed`.
- Simulator/trust/generator/red-team/G0-flow integration: `668 passed`.
- Full repository pytest: `767 passed`.
- Ruff over `src`, `tests`, and `scripts`: PASS.
- Strict mypy over seven Task 6 source/runner files: PASS.
- Project mypy over 42 source files: PASS.
- G0 verifier: PASS for 20 threat cards and reviewed contract flow.
- Direct v3.2 source-only invocation without `PYTHONPATH`: PASS.
- V3, v3.1, and v3.2 result absence plus v3.2 preregistration absence: PASS.
- `git diff --check`: PASS.
- `validation_spike/`: unchanged from `6ad59cd`.

No v3.2 confirmatory search was invoked, and reserved seeds
`(503, 607, 709, 811, 907, 1009, 1103, 1201)` remain untouched.

## Fix round 3 Phase D: canceled v3.2 and external approval anchor

Independent review canceled the `239617e` v3.2 preregistration before execution. The
canonical `task6-v3.2-cancellation.json` preserves its exact source commit,
preregistration commit and byte digest, the absent result, the uninvoked execution path,
and the untouched reserved seeds. V3.3 keeps the same policy, seeds, budgets, metrics,
thresholds, negative control, no-network rule, descriptive uncertainty, and one-attempt
stopping rule; this phase changes only the freeze and approval integrity boundary.

Confirmatory execution now requires two values supplied by an out-of-band controller or
reviewer: an exact approved freeze commit and the exact SHA-256 of the v3.3
preregistration bytes. The runner requires current `HEAD` to equal that commit, not merely
descend from it, and checks the artifact bytes immediately before execution. Verify-only
can validate an explicit pair or print the values awaiting approval; execute mode cannot
self-select them. The result, if a separately authorized controller later creates it,
records both approved values. This is a trusted-controller authorization anchor, not
durable authenticity against a malicious caller that supplies its own values. Task 7
still owns signed approval, process isolation, and an append-only execution receipt.

The experiment protocol is now reconstructed from source constants and compared against
an exact canonical preregistration schema before any evaluator or policy capability is
created. The frozen fields include every seed and proposal/query/logical/wall cap, all
policy versions, both target metrics and thresholds, the negative-control definition,
the zero-network contract, fairness, uncertainty, maximum attempt, stopping rule, and
approval boundary. Extra fields, scalar subclasses, changed types, and changed values
reject.

The source freeze records the exact behavior-affecting Git path set with each path's mode,
object type, blob object ID, and content SHA-256. It includes all tracked `src/`, `scripts/`,
fixtures, root package/environment configuration and locks, cached replay/cancellation
inputs, customization modules, and `.pth` inputs. Execution compares the source and exact
approved-HEAD sets and entries, then independently validates the filesystem. Added or
deleted behavior files, mode/type changes, Git or filesystem symlinks, submodules, changed
bytes, unexpected `sitecustomize`/`usercustomize`, and loaded external customization
modules reject before search.

Phase D again uses two commits. The source commit contains this code, cancellation,
regressions, and report, but intentionally no v3.3 preregistration or result. A subsequent
preregistration-only commit binds that exact source tree and manifest. No Phase D path
constructs or executes confirmatory search while producing either freeze commit.

## Fix round 3 Phase D source-stage verification

- Initial approval/protocol/manifest regressions: `26 failed, 1 passed` before
  implementation; two additional customization escape reproductions and the absent-
  preregistration execute reproduction were also captured RED.
- Phase D approval, protocol, cancellation, manifest, customization, and source tests:
  `30 passed`.
- Round-three regression suites: `69 passed`.
- Complete Task 6 red-team suite: `169 passed`.
- Simulator/trust/generator/red-team/G0-flow integration: `698 passed`.
- Full repository pytest: `797 passed`.
- Ruff over `src`, `tests`, and `scripts`: PASS.
- Strict mypy over seven Task 6 source/runner files: PASS.
- Project mypy over 44 source files: PASS.
- G0 verifier: PASS for 20 threat cards and reviewed contract flow.
- Direct v3.3 source-only invocation without `PYTHONPATH`: PASS.
- V3, v3.1, v3.2, and v3.3 result absence plus v3.3 preregistration absence: PASS.
- `git diff --check`: PASS.
- `validation_spike/`: unchanged from `239617e`.

No v3.3 confirmatory search was invoked, and reserved seeds
`(503, 607, 709, 811, 907, 1009, 1103, 1201)` remain untouched.
