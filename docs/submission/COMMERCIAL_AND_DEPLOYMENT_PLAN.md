# Commercial and deployment plan

## Product position

APAR is an assurance layer around existing fraud stacks—not a rip-and-replace
authorization engine. It gives issuers, acquirers, payment networks, and model
risk teams a controlled way to translate emerging threats into replayable
campaigns, compare controls under fixed operational budgets, and preserve an
auditable human promotion decision.

## Customer value

- Threat teams convert narrative intelligence into executable tests.
- Fraud data scientists compare rules, tabular ensembles, graph signals, and
  integrity controls on the same campaign support.
- Operations teams see friction, review demand, captured value, and latency—not
  just ROC-AUC.
- Investigators receive linked campaign cases and entity graphs.
- Model-risk reviewers receive immutable inputs, controls, failure gates, and
  rejected-run records.
- Agentic-commerce teams validate identity, intent, scope, binding, and replay
  independently of probabilistic fraud risk.

## Integration architecture

1. Read authorized payment, entity, device, account, merchant, decision, and
   lifecycle signals through versioned rail adapters.
2. Maintain past-only temporal and graph state in an event-time feature service.
3. Run APAR in shadow mode beside the incumbent decision engine.
4. Group related events into campaigns and calculate action/capacity economics.
5. Compare champion and challenger results in the assurance range.
6. Require independent review and signed promotion before any export.

## Phased rollout

| Phase | Scope | Exit evidence |
|---|---|---|
| 1. Offline replay | Authorized historical samples; no live decisions | Data parity, leakage, calibration, segment and rail results |
| 2. Shadow mode | Real-time features and scores; incumbent remains authoritative | p95/p99 latency, drift, alert quality, investigation capacity |
| 3. Assisted operations | Low-risk challenges/reviews with human control | Measured customer friction and case productivity |
| 4. Controlled challenger | Limited traffic and explicit rollback | Stable economics, fairness, security, and resilience gates |
| 5. Governed scale | Multi-rail assurance cadence | Signed monitoring and periodic adversarial regression |

## Commercial model

APAR can be delivered as a managed assurance program or deployable control
plane. Pricing can follow protected payment volume plus campaign-evaluation
capacity, while the synthetic range and evidence governance remain shared
platform capabilities. The strongest initial wedge is pre-production assurance
for APP/mule and agentic-commerce launches, where existing transaction-centric
benchmarks are least representative.

## Enterprise controls

Required controls include data minimization, encryption and key separation,
role-based access, tenant isolation, regional processing, content-addressed
evidence, model inventory, incident response, rollback, audit retention, and
human approval. Any cross-institution graph work requires explicit lawful basis,
privacy-preserving design, and contractual governance.

## Why this matches industry direction

BIS Project Hertha reports that payment-system analytics can supplement banks
by detecting complex coordinated activity from a minimal set of data points.
Mastercard’s agentic-commerce work emphasizes verifiable intent, agent identity,
secure credentials, and user control. APAR turns both directions into one
testable assurance workflow: network context for fraud and deterministic intent
integrity for delegated commerce.

Sources: [BIS Project Hertha](https://www.bis.org/publications/project-hertha-identifying-financial-crime-patterns-real-time-retail-payment-systems),
[Mastercard trusted agentic commerce](https://www.mastercard.com/us/en/news-and-trends/stories/2026/mastercard-agentic-commerce-vision.html),
[NIST software and AI agent identity concept paper](https://www.nccoe.nist.gov/sites/default/files/2026-02/accelerating-the-adoption-of-software-and-ai-agent-identity-and-authorization-concept-paper.pdf).
