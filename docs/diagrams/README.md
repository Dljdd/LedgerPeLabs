# Diagram catalog

Editable Mermaid sources are stored in this directory so the architecture can be reused in Markdown, the walkthrough document, slides, and the web prototype.

| Diagram | Source | Purpose |
|---|---|---|
| System context | [01-system-context.mmd](01-system-context.mmd) | Product boundary and actors |
| Identify, Generate, Defend, Assure | [02-logical-architecture.mmd](02-logical-architecture.mmd) | Core component model |
| End-to-end sequence | [03-end-to-end-sequence.mmd](03-end-to-end-sequence.mmd) | Runtime and review flow |
| Deployment view | [04-deployment-view.mmd](04-deployment-view.mmd) | Competition deployment boundaries |
| Payment lifecycle | [05-payment-lifecycle.mmd](05-payment-lifecycle.mmd) | State transitions and outcomes |
| Adaptive red-team loop | [06-adaptive-loop.mmd](06-adaptive-loop.mmd) | Bounded feedback and validity |
| Agentic trust sequence | [07-agentic-trust-sequence.mmd](07-agentic-trust-sequence.mmd) | Intent verification before risk |
| Evaluation topology | [08-evaluation-topology.mmd](08-evaluation-topology.mmd) | Freeze and hidden evaluation |
| Data lineage | [09-data-lineage.mmd](09-data-lineage.mmd) | Provenance and train-serving controls |
| Investigation flow | [10-campaign-investigation.mmd](10-campaign-investigation.mmd) | Events to campaign case |
| Demo sequence | [11-demo-sequence.mmd](11-demo-sequence.mmd) | Five-minute judge narrative |
| Delivery roadmap | [12-delivery-roadmap.mmd](12-delivery-roadmap.mmd) | Competition critical path |

## Rendering

Any Mermaid-compatible Markdown renderer can display these files after wrapping their content in a `mermaid` code fence. For slide or document use, render to SVG to preserve text quality.

## Diagram rules

- Keep labels short and domain-specific.
- Show decision owner separately from recommended action.
- Do not imply that specified components are implemented.
- Keep hidden generator and evaluator visibly separated from development code.
- Keep the trust plane before probabilistic risk scoring.
- Preserve synchronous versus asynchronous boundaries.

