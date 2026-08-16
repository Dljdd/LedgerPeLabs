# Preregistered validation plan

This plan was written before the experiment was executed and before hidden-regime metrics were observed. The experiment is a falsification spike for the proposed Adaptive Payment Security Range. It is not evidence of real-world effectiveness because every transaction in the spike is synthetic.

## Fixed experimental choices

- Seeds: `11, 23, 37, 51, 73`.
- Regime A: 40 simulated days, with training on days 0-24 and testing on days 25-39.
- Regime B: an independently generated 40-day hidden regime with changed benign distributions and altered attack mechanics. It is never used for fitting or parameter selection.
- Attack families: account takeover (`ato`), distributed card testing (`card_testing`), and mule routing (`mule`). Mule routing is excluded from classifier training.
- Hidden benign segment: `business_travel`, which occurs only in regime B and has higher amounts, burstier activity, more new merchants, and occasional new devices without cross-account device or beneficiary sharing.
- Transaction-only baseline features: log amount, channel, and cyclical hour.
- Event-time model: the baseline plus account velocity and amount, new device/beneficiary flags, device-account fan-out, beneficiary-account fan-in, and merchant/device decline bursts. All aggregates use only strictly earlier events.
- Classifier: an in-repository NumPy implementation of weighted L2-regularized logistic regression. No hidden-regime tuning is permitted.
- Operational threshold: the 99th percentile of scores on legitimate regime-A training events.
- Novelty signal: maximum robust standardized deviation over device-account fan-out, beneficiary-account fan-in, device decline burst, and merchant decline burst relative to legitimate regime-A training events. Amount and new-device status are deliberately excluded so that ordinary travel is not treated as campaign novelty.
- Review capacity: 2% of hidden-regime events.
- Novelty-aware priority: `0.65 * risk_percentile + 0.35 * novelty_percentile`. Risk-only and random policies use the same capacity.
- Campaign attacker: decision-only feedback, 40 queries per seed, minimum visible fraud value of 600 monetary units, positive amounts, valid entity references, and monotonically increasing timestamps.
- Hidden post-search constraints: minimum value 700, maximum attack execution duration 240 minutes, maximum 12 splits, warm-up no longer than 14 days, and declared capability for any trusted-device use.

## Hypotheses and decision rules

### H1

Temporal/campaign features outperform transaction-only features on campaign fraud in development regime A.

- **SUPPORTED** if mean average-precision gain is at least 0.05 and mean campaign-detection gain at the fixed threshold is at least 0.10.
- **PARTIALLY SUPPORTED** if both gains are positive but only one reaches its threshold.
- **NOT SUPPORTED** otherwise.

### H2

Gains substantially survive hidden regime B with different benign behavior and attack mechanisms.

- **SUPPORTED** if the hidden average-precision gain is positive and at least 50% of the development gain, and hidden campaign detection is no worse than the baseline.
- **PARTIALLY SUPPORTED** if the hidden average-precision gain remains positive but is below 50% of the development gain, or campaign detection regresses.
- **NOT SUPPORTED** if the hidden average-precision gain is non-positive.

### H3

Simple novelty-aware triage improves held-out-family fraud capture at a fixed review budget without having that budget consumed by unseen benign behavior.

- **SUPPORTED** if novelty-aware triage improves held-out mule fraud-value capture over risk-only by at least 0.05, exceeds random review, and its unseen-benign share of review slots is no more than 0.02 above risk-only.
- **PARTIALLY SUPPORTED** if held-out fraud capture improves but the benign-consumption condition or cross-seed consistency fails.
- **NOT SUPPORTED** if mean held-out fraud capture does not improve.

### H4

Constrained decision-only campaign search can find economically valid evasions under a fixed query budget, and hard hidden checks expose reward hacking.

- **SUPPORTED** if an evasion is found in at least 60% of seeds, at least one evasion passes all hidden constraints, and at least one superficially successful evasion is rejected by a hidden constraint.
- **PARTIALLY SUPPORTED** if evasions are found but either none passes hidden checks or no reward-hacked success is exposed.
- **NOT SUPPORTED** if the evasion rate is below 20%.

### H5

Event-time and metamorphic tests detect leakage or simulator fingerprints when deliberately injected.

- **SUPPORTED** if every deliberate forbidden feature is rejected, equal-time and future-source timestamps are rejected, and ID-permutation, future-event independence, metadata-removal, chronological-split, and campaign-isolation tests pass for all seeds.
- **PARTIALLY SUPPORTED** if deliberate leakage is caught but any clean invariant test fails.
- **NOT SUPPORTED** if any deliberate leakage is accepted.

## Negative-result policy

All seeds and all failed checks will be retained. Parameters and decision rules will not be changed after hidden metrics are seen. Code defects may be corrected, but any correction after a result-producing run must be recorded in `outputs/run_history.json`.

## Pre-run audit amendment, 2026-08-16

Before the first metric-producing run, an independent audit identified that the current event's decline outcome would not be available at a pre-authorization decision point. It was removed from both model feature sets. Prior-event decline bursts remain valid because they use strictly earlier events. The event-time contract was also tightened: temporal aggregates may use only events with `source_timestamp < decision_timestamp`. Equal-time events do not see one another, regardless of event-ID ordering. No experimental metric or hidden-regime result had been observed when this amendment was made.
