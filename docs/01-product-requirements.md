# Product requirements

## 1. Product definition

The Adaptive Payment Assurance Range is a synthetic, pre-production platform for challenging payment controls with evidence-backed, GenAI-enabled fraud campaigns.

It serves assurance, model-risk, fraud-product, threat-intelligence, and investigation users. It does not make live payment decisions in the competition prototype.

## 2. Outcome hierarchy

### North-star outcome

Increase confidence that layered payment controls remain effective under novel, economically valid, GenAI-enabled campaign shifts without exceeding customer-friction or operational-capacity limits.

### Product outcomes

- Threat evidence becomes a testable scenario.
- Scenarios retain provenance and assumptions.
- Campaigns obey payment lifecycle and economic constraints.
- Defenses are compared at the same operating budget.
- Unknown and shifted campaigns are evaluated without future information.
- Human reviewers receive an auditable promotion recommendation.

### Competition outcomes

- Demonstrate Identify, Generate, and Defend in one workflow.
- Show novelty that is not merely a new classifier.
- Show payment-domain and Mastercard-view credibility.
- Run a deterministic five-minute prototype.
- Deliver a complete, portable, safe submission package.

## 3. Functional requirements

### PR-IDENTIFY

- **PR-IDENTIFY-01:** Store at least 20 reviewed threat cards.
- **PR-IDENTIFY-02:** Every card shall include at least one evidence record or be labeled a hypothesis.
- **PR-IDENTIFY-03:** Every card shall identify affected rail, lifecycle stages, decision owner, observables, and GenAI capability delta.
- **PR-IDENTIFY-04:** Coverage shall be reported across rails, channels, attack stages, and GenAI capabilities.
- **PR-IDENTIFY-05:** Cards shall distinguish current evidence from future inference.

### PR-GENERATE

- **PR-GENERATE-01:** Compile an approved threat card into a typed scenario bundle.
- **PR-GENERATE-02:** Execute four deep scenario families.
- **PR-GENERATE-03:** Simulate event, ingestion, feature, and decision time separately.
- **PR-GENERATE-04:** Simulate authentication, authorization, settlement, report, recovery, and label maturity.
- **PR-GENERATE-05:** Enforce balance and flow conservation.
- **PR-GENERATE-06:** Preserve deterministic replay and complete lineage.
- **PR-GENERATE-07:** Include realistic benign novelty and degraded data.

### PR-ADAPT

- **PR-ADAPT-01:** Run fixed, random, adaptive non-LLM, and agent/LLM planners under matched budgets.
- **PR-ADAPT-02:** Candidate `n+1` shall depend on observations from prior candidates for an adaptive run.
- **PR-ADAPT-03:** The attacker shall not access model internals, hidden constraints, labels, or future events.
- **PR-ADAPT-04:** Hidden validity shall reject economically or causally invalid candidates.
- **PR-ADAPT-05:** Every proposal, observation, prompt, tool call, seed, and model version shall be replayable.

### PR-DEFEND

- **PR-DEFEND-01:** Provide deterministic rules and at least one strong GBDT baseline.
- **PR-DEFEND-02:** Use only decision-available features on the synchronous path.
- **PR-DEFEND-03:** Group transactions into campaigns and investigator cases.
- **PR-DEFEND-04:** Produce stable reason codes and evidence references.
- **PR-DEFEND-05:** Separate recommended action from the actor authorized to decide.
- **PR-DEFEND-06:** Support approve, challenge, review, delay, decline, monitor, and revoke as policy outputs where rail-appropriate.

### PR-AGENTIC

- **PR-AGENTIC-01:** Verify agent identity and request signature.
- **PR-AGENTIC-02:** Verify mandate scope, expiry, merchant, beneficiary, amount, currency, and category.
- **PR-AGENTIC-03:** Bind cart, payment intent, credential scope, consent, and execution receipt.
- **PR-AGENTIC-04:** Reject replay and substitution deterministically.
- **PR-AGENTIC-05:** Do not allow an ML score to override an integrity failure.

### PR-EVALUATE

- **PR-EVALUATE-01:** Evaluate chronologically and by family.
- **PR-EVALUATE-02:** Include returning-entity and cold-entity partitions.
- **PR-EVALUATE-03:** Use a separately implemented hidden generator after defender freeze.
- **PR-EVALUATE-04:** Enforce false-positive, false-decline, challenge, review, and latency budgets.
- **PR-EVALUATE-05:** Measure preventable settled value and value escaped before alert.
- **PR-EVALUATE-06:** Block promotion on any critical integrity or safety failure.

### PR-DEMO

- **PR-DEMO-01:** Complete the golden path in under five minutes.
- **PR-DEMO-02:** Start with one documented command on a clean machine.
- **PR-DEMO-03:** Run without external network access.
- **PR-DEMO-04:** Provide deterministic fixture and recording fallback.
- **PR-DEMO-05:** Produce a downloadable assurance report.

## 4. Non-functional requirements

### Reliability

- Decisions are idempotent for duplicate event IDs.
- Model timeout invokes a declared deterministic fallback.
- Missing enrichment is visible in the output.
- Every displayed result can be reconstructed.

### Performance

- Synchronous score latency is measured at p50, p95, and p99.
- Offline simulation and graph processing do not block synchronous scoring.
- The prototype publishes hardware, data size, throughput, and memory use.

### Security and privacy

- Synthetic, anonymized, or authorized data only.
- No live-system attack execution.
- Role-separated approval for scenario publication and defender promotion.
- Secret scan, license inventory, and SBOM before submission.

### Accessibility and usability

- Keyboard-accessible navigation and controls.
- Text and controls meet readable contrast.
- Status is never communicated by color alone.
- No critical content depends on an entrance animation.
- The demo remains readable at laptop and tablet widths.

## 5. Scope priorities

### Must have

- Threat registry.
- Four scenarios.
- Rail-specific contracts.
- Stateful lifecycle and conservation.
- Rules plus GBDT.
- Adaptive-search ablation.
- Agentic integrity scenario.
- Hidden generator.
- Past-only evaluation.
- Web prototype and walkthrough.

### Should have

- Campaign graph and investigator cases.
- Drift and workload dashboard.
- Signed promotion report.
- Replay checkpoints.
- GNN challenger.

### Deferred

- Production payment connectivity.
- Cloud-scale distributed deployment.
- Automated model promotion.
- Full commercial cryptographic protocol.
- Broad external-data ingestion.

## 6. Product success gates

The product is competition-ready only if all must-have requirements are evidenced in [TRACEABILITY.md](TRACEABILITY.md), no critical gate is waived, and the final archive passes the submission checklist.

