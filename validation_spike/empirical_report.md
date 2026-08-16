# Empirical falsification report

## Result

This preregistered synthetic spike produced the following verdicts across five fixed seeds:

| Hypothesis | Verdict |
|---|---|
| H1: temporal/campaign features in development | **SUPPORTED** |
| H2: survival under hidden regime shift | **SUPPORTED** |
| H3: novelty-aware fixed-budget triage | **PARTIALLY SUPPORTED** |
| H4: constrained decision-only evasion and hidden validity | **SUPPORTED** |
| H5: leakage and metamorphic defenses | **SUPPORTED** |

No thresholds, seeds, feature sets, or verdict rules were changed after hidden results were observed.

## Data volume

| Slice | Events, mean ± SD | Fraud rate, mean ± SD |
|---|---:|---:|
| Regime A, all | 6968.0 ± 0.0 | 11.022% ± 0.000% |
| Regime A, training | 4282.6 ± 38.9 | 9.808% ± 0.090% |
| Regime A, chronological test | 2685.4 ± 38.9 | 12.961% ± 0.186% |
| Hidden regime B | 6988.6 ± 14.0 | 6.782% ± 0.014% |

Hidden B contained 314.6 ± 14.0 events from the unseen business-travel segment.

## Measurements

### H1 and H2: detector comparison

| Regime | Model | Average precision | Campaign detection at A-trained threshold | False-positive rate |
|---|---|---:|---:|---:|
| Development A | Transaction only | 0.137 ± 0.009 | 0.000 ± 0.000 | 0.011 ± 0.003 |
| Development A | Event-time temporal/campaign | 0.879 ± 0.005 | 0.779 ± 0.039 | 0.011 ± 0.002 |
| Hidden B | Transaction only | 0.058 ± 0.003 | 0.000 ± 0.000 | 0.014 ± 0.005 |
| Hidden B | Event-time temporal/campaign | 0.439 ± 0.016 | 0.887 ± 0.014 | 0.050 ± 0.003 |

Development average-precision gain was 0.742; campaign-detection gain was 0.779. Hidden average-precision gain was 0.380. The hidden gain was 51.2% of the development gain.

### H3: 2% review capacity in hidden regime B

| Policy | Held-out mule value captured | Unseen travel share of reviews | Fraud share of reviews |
|---|---:|---:|---:|
| Risk only | 0.002 ± 0.003 | 0.476 ± 0.042 | 0.500 ± 0.043 |
| Novelty aware | 0.045 ± 0.024 | 0.037 ± 0.017 | 0.954 ± 0.012 |
| Random | 0.022 ± 0.016 | 0.046 ± 0.014 | 0.073 ± 0.024 |

The novelty-aware policy changed mule-value capture by +0.043 and changed unseen-travel review consumption by -0.438 relative to risk-only.

### H4: decision-only search

- Evasion found in 100% of seeds under 40 queries.
- Seeds with at least one hidden-valid evasion: 5/5.
- Seeds where hidden checks rejected at least one superficial success: 5/5.
- Mean number of evasions found per seed: 12.8.

The hidden checks cover minimum economic value, execution duration, split count, warm-up duration, and undeclared trusted-device capability. A rejected evasion is evidence that the search exploited the visible environment, not evidence of a successful fraud campaign.

This spike used a fixed, seed-specific schedule containing three declared candidates followed by reproducibly sampled candidates. It validates bounded decision-only candidate search and hidden post-search validity checks. It does not empirically validate an adaptive genetic or evolutionary optimizer.

### H5: test results

- `id_permutation_invariance`: 5/5 seeds passed
- `future_event_independence`: 5/5 seeds passed
- `metadata_removal_invariance`: 5/5 seeds passed
- `all_forbidden_features_rejected`: 5/5 seeds passed
- `equal_source_timestamp_rejected`: 5/5 seeds passed
- `future_source_timestamp_rejected`: 5/5 seeds passed
- `clean_event_time_check`: 5/5 seeds passed
- `chronological_split`: 5/5 seeds passed
- `campaign_isolation`: 5/5 seeds passed

## Architecture changes warranted

- Do not deploy a fixed novelty blend. Calibrate triage against benign novelty and review-capacity constraints, with a risk-only fallback.
- Make drift-aware threshold monitoring and recalibration a promotion requirement. Hidden-regime feature gains do not make a static regime-A threshold operationally safe.
- Report absolute held-out-family coverage, not only improvement over risk-only. Reserve investigation capacity for campaign-level exploration if transaction-level novelty capture remains low.
- Describe the implemented red team as bounded decision-only candidate search. Adaptive mutation or evolutionary optimization remains unvalidated.
- Benchmark against a strong GBDT and production-style rules before claiming model superiority; the transaction-only logistic baseline is deliberately weak.
- Require an independently implemented generator or authorized external dataset before claiming cross-simulator transfer.

## Negative results and limitations

- Non-supported hypotheses: H3.
- All development and hidden data were produced by code in this repository. Distinct mechanisms and parameters reduce, but do not eliminate, simulator circularity.
- The NumPy logistic model is intentionally simple. Results do not establish that the same ordering holds for CatBoost, LightGBM, GNNs, or production decision systems.
- The transaction-only comparator is deliberately weak and contains no production rule engine, entity profiles, or GBDT interactions. H1 is a mechanism check, not evidence of superiority over a mature fraud stack.
- Although H2 passes its preregistered relative-gain rule, the event-time model's false-positive rate increased from 1.078% in development to 5.010% in hidden B. A static regime-A threshold is therefore operationally unsafe under this simulated drift.
- Novelty-aware triage captured only 4.472% of held-out mule value at the 2% review budget. Its improvement over risk-only did not reach the preregistered five-point threshold, so H3 is not fully supported.
- The five seeds quantify simulator variance, not uncertainty over real payment populations.
- The hidden benign segment covers one type of novelty. Product launches, festivals, migrations, emergencies, and merchant-network changes could consume triage capacity differently.
- The campaign attacker searches a small declared parameter space. It is not a comprehensive adversarial-ML evaluation.
- Hidden B changes mechanisms and distributions but is implemented in the same source file as regime A. It is not an independent generator and cannot resolve simulator circularity.
- “Campaign detection” means at least one event crossed a threshold. It does not establish that every actor or flow was reconstructed.

## Reproducibility

Run the exact command in `README.md` from this directory. Exact per-seed results, selected attack candidates, hidden rejection reasons, environment versions, and code hashes are in `outputs/results.json`; flat metrics are in `outputs/metrics.csv`.
