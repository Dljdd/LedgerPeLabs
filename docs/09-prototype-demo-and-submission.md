# Prototype, demo, and submission specification

## 1. Prototype objective

The prototype must make the assurance loop understandable and credible in five minutes. It is not a generic dashboard and does not attempt to expose every configuration option.

## 2. Information architecture

Primary navigation:

1. Threats
2. Scenarios
3. Replay
4. Defenses
5. Investigation
6. Assurance report

Each view shall answer one question and maintain the same selected run context.

## 3. Golden-path scenario

Use an AI-personalized APP scam and mule network as the main narrative because it demonstrates:

- GenAI capability delta.
- Individually plausible authorized payments.
- Coordinated network behavior.
- Money-flow conservation.
- Adaptive attack behavior.
- Customer-friction tradeoffs.
- Investigator case grouping.

Use the agentic-commerce integrity scenario as a short second proof point.

## 4. Screen requirements

### Threats

Show evidence, confidence, source versus inference, GenAI capability delta, affected rail, and implementation status. Do not present counts without coverage context.

### Scenarios

Show the approved scenario, constraints, lifecycle, economic objective, query budget, safety boundary, and matched control variants. Hide unsafe low-level details from judge-facing export.

### Replay

Show the current lifecycle stage, event chronology, actor or entity relationship, amount state, decision, and reason. Events must remain visible by default and not depend on entrance animations.

### Defenses

Compare rules, champion, and challenger at the same operating budget. Show value saved, false interventions, challenges, workload, time to alert, and calibration. Do not center the page on AP.

### Investigation

Show one campaign graph, first alert, value moved before and after alert, linked entities, case evidence, and estimated analyst effort.

### Assurance report

Show passed and failed gates, hidden evaluation, provenance, limitations, and the human promotion decision. The model cannot mark itself approved.

## 5. Demo sequence

```mermaid
sequenceDiagram
    participant J as Judge
    participant UI as Prototype
    participant R as Assurance range
    participant D as Layered defense
    participant E as Evaluator

    J->>UI: Open reviewed GenAI threat
    UI->>R: Compile approved scenario
    R->>D: Replay obvious campaign
    D-->>UI: Baseline catches obvious pattern
    J->>UI: Start bounded adaptation
    R->>D: Submit adaptive candidates
    D-->>R: Coarse decision feedback
    R-->>UI: Invalid candidates rejected; valid evasion found
    R->>D: Replay champion and challenger
    D->>E: Decisions, actions, evidence and costs
    E-->>UI: Hidden gates and business metrics
    J->>UI: Open agentic integrity attack
    UI-->>J: Deterministic rejection and audit chain
    J->>UI: Open promotion report
```

## 6. Demo script budget

| Segment | Time | Outcome |
|---|---:|---|
| Problem and threat | 35 sec | Establish GenAI capability delta |
| Compile and replay | 45 sec | Demonstrate rail-correct generation |
| Adaptive evasion | 70 sec | Show novelty and bounded feedback |
| Champion/challenger | 70 sec | Show defense and business metrics |
| Investigation | 35 sec | Show campaign-level actionability |
| Agentic integrity | 30 sec | Show deterministic trust control |
| Assurance report | 35 sec | Show hidden evaluation and governance |

Total target: 5 minutes.

## 7. Reliability requirements

- One documented start command.
- No external network dependency.
- Fixed golden-path fixture.
- Cached LLM or agent outputs with full replay trace.
- Seeded live mode as optional evidence, not the only demo.
- Preflight health check.
- Reset button returns to the canonical scenario.
- Backup screen recording and static PDF screenshots.
- Visible error state and fallback if any worker fails.

## 8. UI quality constraints

- Content visible by default.
- No decorative background grid, glow, floating cards, or fake app-window prop.
- No purple or blue-to-purple gradient system.
- No card hover lift or entrance animation gating content.
- Real graph and event data, not decorative charts.
- Corresponding comparison rows and actions align across models.
- All controls have adequate contrast, focus state, and readable labels.
- Text has consistent gutters and is never clipped.
- Motion is limited to replay progress and meaningful state changes.

## 9. Walkthrough document outline

1. Executive problem and product thesis.
2. Competition requirement mapping.
3. GenAI threat portfolio.
4. Architecture and rail viewpoint.
5. Main campaign walkthrough.
6. Adaptive attacker ablation.
7. Defense and business results.
8. Agentic trust scenario.
9. Hidden evaluation and scientific safeguards.
10. Commercial fit and scaling path.
11. Limitations and next steps.
12. Reproduction and repository instructions.

## 10. Submission repository checklist

- Root README and architecture summary.
- Portable start command.
- Complete source for Identify, Generate, and Defend.
- Synthetic demo fixtures.
- Model and data cards.
- Threat-card and evidence manifest.
- Evaluation report and reproducible command.
- Tests and CI configuration.
- License, notices, SBOM, and dependency lock.
- Walkthrough PDF, PPTX, or DOCX.
- Working web prototype.
- Screenshots or backup recording.
- Explicit limitations.

## 11. Archive safety checklist

- Project is an isolated repository.
- Archive is built from an explicit allowlist.
- No `.env`, credentials, caches, local databases, unrelated files, or absolute home paths.
- No unauthorized data or personal metadata.
- Archive is extracted and started in a clean directory.
- Hashes are recorded.
- Final contents are reviewed by two people where possible.

## 12. Commercial narrative

Position the product as adaptive assurance for existing payment controls.

Likely internal users:

- Fraud-product and payment-security teams.
- Threat intelligence.
- Model risk and validation.
- Issuer and acquirer assurance programs.
- Investigator and scam-intelligence teams.

Outputs are scenario packs, control-gap reports, challenger evidence, and governed promotion recommendations.

## 13. Demo acceptance tests

- New judge completes the workflow without guidance.
- Golden path completes in under five minutes.
- Every displayed metric links to a run artifact.
- App works with network disabled.
- Reload and reset reproduce the same state.
- Failure injection shows a usable fallback.
- Keyboard-only navigation reaches every primary action.
- No hidden, clipped, overlapping, or low-contrast content at supported widths.

