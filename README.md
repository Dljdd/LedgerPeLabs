# Mastercard Innovation Challenge 2026

This repository specifies and validates an **Adaptive Payment Assurance Range** for testing payment-risk and agentic-commerce controls against emerging GenAI-enabled fraud.

The product is an assurance layer, not a replacement payment-decision engine. It turns sourced threats into constrained synthetic campaigns, evaluates champion and challenger defenses under hidden shifts, and produces evidence for a human promotion decision.

## Start here

- [Canonical solution specification](SOLUTION_SPEC.md)
- [Documentation index](docs/README.md)
- [Approved implementation plans](docs/superpowers/plans/README.md)
- [Diagram catalog](docs/diagrams/README.md)
- [Empirical validation spike](validation_spike/README.md)

## Proposed product flow

`Threat evidence -> Threat registry -> Scenario compiler -> Stateful payment simulator -> Champion/challenger controls -> Operational policy -> Hidden evaluation -> Human promotion report`

## Repository status

| Area | Status |
|---|---|
| Product and architecture specification | Documented |
| Empirical falsification spike | Implemented |
| Threat registry and scenario compiler | Specified, not implemented |
| Rail-specific simulator | Specified, not implemented |
| Adaptive red-team optimizer | Harness validated; adaptive optimizer not implemented |
| Defender service | Specified; validation baseline only |
| Agentic trust plane | Specified, not implemented |
| Web prototype and walkthrough | Specified, not implemented |

The `validation_spike` is retained as supporting evidence. It must not be represented as the complete competition solution.

## Working title

**Adaptive Payment Assurance Range**  
Evidence-backed adversarial testing for card, account-to-account, and agentic-commerce controls.
