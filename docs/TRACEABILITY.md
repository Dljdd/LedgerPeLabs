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
| Defend | Tuned rules baseline | Specified | Versioned rules and metrics |
| Defend | Strong GBDT baseline | Specified | Model bundle and comparison |
| Defend | Past-only feature state | Partially validated | Validation spike plus production tests |
| Defend | Campaign case grouping | Specified | Case output and reconstruction metrics |
| Agentic | Identity, mandate, scope, binding, replay | Specified | Integrity attack suite |
| Evaluate | Chronological and cold-entity splits | Specified | Dataset manifest and assertions |
| Evaluate | Independent hidden generator | Specified | Separate package hash and freeze record |
| Evaluate | Fixed operational budgets | Specified | Action-frontier report |
| Evaluate | No future batch ranking | Specified | Online queue test |
| Governance | Immutable content-addressed runs | Specified | Artifact-store tests |
| Governance | Human promotion gate | Specified | Signed report and audit event |
| Safety | Synthetic-only, no live targeting | Specified | Network and input safety tests |
| Engineering | Portable one-command start | Specified | Clean-machine test log |
| Engineering | CI fails on invariant failure | Specified | CI run and mutation tests |
| Prototype | Five-minute offline golden path | Specified | Usability test and recording |
| Submission | Repository, walkthrough, web prototype | Specified | Final archive inventory |
| Validation spike | H1-H5 falsification spike | Validated with limitations | `validation_spike` reports and outputs |

## Status update rules

- `Implemented` requires code and an executable path.
- `Validated` requires an acceptance test result.
- A screenshot alone does not validate backend behavior.
- A same-code hidden regime does not satisfy the independent hidden-generator requirement.
- A fixed or random candidate schedule does not satisfy adaptive search.
- An architecture diagram does not satisfy an implementation requirement.

