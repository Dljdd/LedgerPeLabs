# Requirements traceability matrix

This matrix is the control surface for implementation status. Replace `Specified` with `Implemented` or `Validated` only when an evidence link exists.

| Requirement area | Requirement | Status | Planned evidence |
|---|---|---|---|
| Identify | At least 20 reviewed threat cards | Specified | Registry export and coverage report |
| Identify | Evidence, confidence, rail, lifecycle, and GenAI delta | Specified | Threat-card schema tests |
| Generate | Four deep scenario families | Specified | Scenario bundles and replay fixtures |
| Generate | Rail-correct lifecycle | Specified | State-transition tests |
| Generate | Value conservation | Specified | Reconciliation artifacts |
| Generate | Benign novelty and degraded data | Specified | Stress-suite report |
| Adapt | Candidate uses prior feedback | Specified | Candidate trace and counterfactual test |
| Adapt | Matched random, adaptive, and agent ablation | Specified | Red-team evaluation report |
| Adapt | Hidden economic and causal validity | Specified | Hidden-valid result summary |
| Defend | Tuned rules baseline | Implemented | Versioned rules and matched-threshold tests; competition evaluation correctly stopped before comparison |
| Defend | Strong GBDT baseline | Validated | Signed 200-campaign training receipt, deterministic native model, and frozen reload/prediction test |
| Defend | Past-only feature state | Validated | Causal source audit, future-append tests, and frozen 48-feature competition matrix |
| Defend | Campaign case grouping | Validated | Exact callback reconstruction and signed 6/336 minimum workload failure evidence |
| Defend | v2 sealed evaluation protocol | Protocol sealed, not executed | Signed preregistration and read-only pre-execution verifier; no evaluation receipt or result artifact exists |
| Agentic | Identity, mandate, scope, binding, replay | Specified | Integrity attack suite |
| Evaluate | Chronological and cold-entity splits | Specified | Dataset manifest and assertions |
| Evaluate | Independent hidden generator | Implemented, not released | Separate pinned authority and isolated worker; release was correctly withheld because no defender could freeze |
| Evaluate | Fixed operational budgets | Validated negative | Signed exhaustive frontier proves 1.7857% minimum review workload exceeds the frozen 1% cap; no relaxation |
| Evaluate | No future batch ranking | Specified | Online queue test |
| Governance | Immutable content-addressed runs | Validated | 200 signed run manifests plus signed corpus/result/hash aliases |
| Governance | Human promotion gate | Specified | Signed report and audit event |
| Safety | Synthetic-only, no live targeting | Specified | Network and input safety tests |
| Engineering | Portable one-command start | Specified | Clean-machine test log |
| Engineering | CI fails on invariant failure | Specified | CI run and mutation tests |
| Prototype | Five-minute offline golden path | Specified | Usability test and recording |
| Submission | Repository, walkthrough, web prototype | Specified | Final archive inventory |
| Validation spike | H1-H5 falsification spike | Validated with limitations | `validation_spike` reports and outputs |
| Defend v3 | Execution path drafted; evaluation not executed | Drafted | Separately versioned protocol; no v3 population or result exists |
| Defend v3 result | Confirmatory attempt consumed on incomplete scaffold | Truthful no_promotion | No scoring, metrics, or gates were evaluated; v4 protocol revision required |

## Status update rules

- `Implemented` requires code and an executable path.
- `Validated` requires an acceptance test result.
- A screenshot alone does not validate backend behavior.
- A same-code hidden regime does not satisfy the independent hidden-generator requirement.
- A fixed or random candidate schedule does not satisfy adaptive search.
- An architecture diagram does not satisfy an implementation requirement.
