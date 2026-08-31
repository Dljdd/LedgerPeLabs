# Model card — Sentinel v5 graph ensemble

## Intended use

The competition model is a portable, CPU-scored `ensemble_with_graph` arm for
synthetic pre-production demonstration. It estimates campaign risk at decision
time and routes an event to `approve`, `challenge`, `review_hold`, or
`decline_hold`. It is not a production authorization model and must not be used
against real cardholders without an independently governed validation program.

## Model identity

| Field | Value |
|---|---|
| Architecture | Three-member calibrated CatBoost ensemble |
| Features | 46 frozen causal features |
| Selected arm | `ensemble_with_graph` |
| Accepted checkpoint | Stage `30_arms` |
| Source checkpoint manifest | `ae9c7a0a027d97739fd0556c5ae6d659f7d0846e31076541978b853af2b4b579` |
| Portable bundle manifest | `52ed4c31d88cd0fc9d3f7557e3addff2044e33bb8991cc7e12a5b70385dbc900` |
| Probability tolerance in replay | `1e-12` |
| Data class | Synthetic only |

The feature set combines transaction context with past-only velocity, amount
deviation, pair history, actor/counterparty fan-out, shared-neighbor, two-hop,
burst-motif, component, edge-density, lifecycle, and data-quality signals.
Rows at equal decision time are isolated and future events are excluded.

## Decision policy

The package uses the frozen thresholds in `demo/sentinel-v5/spec.json`:

- challenge at probability `0.1`;
- review hold at probability `0.5037653375170894`;
- decline hold at probability `1.0`.

For agentic transactions, a definitive TrustVerifier failure takes precedence
over statistical risk and produces `decline_hold`.

## Verified behavior

The portable runner reloads three model members and three calibrators, scores
12 hash-bound scenarios, and reproduces every stored probability and action.
The expected action mix is three approvals, one challenge, two review holds,
and six decline holds. On the curated examples, scenario recall and the
captured-value proxy are both 1.0. These are replay checks, not population
performance estimates.

The verified but non-authoritative recovered comparison reports the graph arm
at 99.867% recall, 95.876% precision, 97.831% F1, 0.0037% false-decline,
0.572% challenge, 0.211% review, and 3.544 ms p95 latency on its synthetic
development support. These values are useful for model selection; they are not
official Stage 70 or production evidence.

The official chain remains incomplete at Stage 70.

## Why this arm was selected

The rules-only arm created severe benign friction. The non-graph ensemble was
strong, but adding causal graph features improved recall, precision, F1, and
challenge rate with a small latency cost. The full deterministic hybrid caught
almost every synthetic fraud case but routed far too many legitimate cases to
decline/challenge, so it is retained as a diagnostic architecture—not called
the champion.

The full Sentinel hybrid is not the champion.

## Risks and controls

- Synthetic generator artifacts may overstate separation; graph motifs can be
  easier to distinguish in simulation than in real traffic.
- Population prevalence, behavior, and feedback loops are not externally
  validated.
- Model calibration can drift under new rails, geographies, and attacker
  behavior.
- Graph features require entity resolution and strict event-time semantics.
- A decision model cannot validate delegated payment authority; TrustVerifier
  remains a separate deterministic plane.

Required production steps are shadow deployment, representative backtesting,
fairness and segment analysis, calibration monitoring, latency/load testing,
human review capacity testing, security review, and accountable approval.
