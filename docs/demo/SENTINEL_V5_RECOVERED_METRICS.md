# Sentinel v5 recovered metrics

This document records the seed-404 Kaggle capacity-validation recovery. The recovery is **non-authoritative**: it is not an accepted Stage 70 checkpoint, it is not capacity evidence, and it does not complete the official safe chain.

## Exact status

- Portable competition model: ready at commit `d8764266459bada13707aa155fe43df90a3f50fd`; this recovery does not modify its bundle or scorer.
- Accepted official prefix: Stages 00, 10, 20, 30, 40, 50, 51, 52, 53, and 60.
- First missing official stage: `70_metrics`.
- Official Stage 70 saved run: script version `345957579`, failed after `9118.0s`, accelerator `None`, with Kaggle's `Your notebook tried to allocate more memory than is available.0` message. The wrapper reported `RuntimeError: closed checkpoint stage failed`; no Stage 70 checkpoint was published.
- Official Stage 80: absent/invalid because its required Stage 70 predecessor does not exist.
- Non-authoritative recovery: saved-successful script version `346022855`, `12225.0s`, accelerator `None`. All four arm workers reported `fresh_interpreter: true` before compact finalization.
- Recovered readiness: `not_ready`. This is diagnostic evidence only.
- This Kaggle recovery and every accepted safe checkpoint used development-test seed 404 only. No Kaggle locked-successor or seed-2404 chain was run.
- An earlier local locked-development attempt started at `2026-08-24T13:32:54.948338Z` and was irreversibly aborted by the host watchdog at `2026-08-24T13:48:11Z`. It published no candidate manifest, candidate chunks, or judge summary, and produced no successful seed-2404 result. The retained execution-boundary artifacts do not prove which internal sub-step the attempt reached, so this report makes no stronger claim.

The accepted terminal Stage 60 rerun is saved-successful script version `345946007` (`9911.9s`) with deterministic digest `a8caf985a2459fc9792b0de744defdd496f62654ed000108de93db7f6f8a0e5f`, manifest `d13122fc912aaec48574c67e3e3a0891abb133d76a247c90f337d9de2f1cfcfc`, and observational digest `c0d59d3bec2afb6179ae9615e8cb8ed2670656cc46809eaeb05fb2e06ffe5593`.

## Why the official chain cannot be completed in place

The frozen official notebook digest is `5cd7a8638ecab0ef50be0e959e76b9f49f0c7bbba3d7ca57b5944d1b83bf6adb`; the frozen runner digest is `98e869d79535b77bcaeab533509eb3acbcfdfb602e352c35ddf4c2d4c962685f`. Both still match approved source commit `40fb4a131da36556d2b8a04564cb62d73152c7c1`.

That runner unconditionally calls `load_v5_arm_checkpoint`, materializing the complete four-arm evidence tuple before metric construction. The exact frozen run exhausted Kaggle memory. The recovery avoids this by using one fresh interpreter per arm and retaining only compact metric documents. Putting that architecture into the official notebook would change the notebook and source bindings checked by the frozen authority. Publishing the recovered JSON directly would also bypass capability issuance, telemetry, and the exclusive checkpoint publisher. Either route would break the official evidence contract.

Consequently, an official Stage 70 requires a future source-bound rerun beginning at the protocol-defined blast radius; it cannot be retroactively manufactured from this recovery. No such rerun was started.

## Future source remediation status

A memory-bounded Stage 70/80 candidate is now implemented locally. It restores one Stage 30 arm in each fresh Python interpreter, writes only compact self-hashed metric summaries, enforces per-worker RSS/time/artifact gates, and lets Stage 80 index Stage 70 without restoring the four-arm checkpoint again. Focused tests prove that this candidate preserves the prior deterministic metric core and rejects receipt, support, and resource substitutions.

This is unexecuted source remediation, not an accepted checkpoint and not an extension of the accepted prefix. Its source and implementation bindings differ from commit `40fb4a131da36556d2b8a04564cb62d73152c7c1`, so any official use would require a newly approved chain beginning at Stage 00. The public verifier deliberately rejects compact Stage 70/80 evidence for official acceptance until an independent memory-bounded semantic replay of the Stage 30 rows, model artifacts, controls, and metric construction is implemented. Structural compact replay remains available only to the test fixture. No Kaggle run or Stage 00 restart was performed for this remediation.

## Verified predecessor lineage

| Stage | Manifest SHA-256 |
|---|---|
| 00_authorize | `82fbb30ce7e081a25dfae940aa9a78d46998edac88fc9558db0c8cbb76659f85` |
| 10_corpus | `c0ce0a7e40622432ce0c70a19b507321cbb5337fadc6cfe2aa3cf9bc1b442803` |
| 20_features | `09403f1caabbee2c698b81e99436fe2b6e828bda51410b3ffef29c80b56dad54` |
| 30_arms | `ae9c7a0a027d97739fd0556c5ae6d659f7d0846e31076541978b853af2b4b579` |
| 40_label_shuffle | `6296a4ee7e948c8984bfe86151d309966915be4fb73dd858f97113d765224bdc` |
| 50_identity_rename | `9fb98cd50b6697ed44f368e22ae3825ff70b93670ebf1bd2080bc976d6162e28` |
| 51_future_causality | `45f38eb676f8232e959576fb29d47ccca8cea2faaeed8ce02f747b8a850b20db` |
| 52_equal_time_isolation | `24911298299e49af307b010f00011b2bdd90853bbc377f254b4127326c894d0d` |
| 53_feature_leakage | `67c4719637a12faabe083702f1352771d4ecb01517d42d598bdeb97d946b10cd` |
| 60_single_class_controls | `d13122fc912aaec48574c67e3e3a0891abb133d76a247c90f337d9de2f1cfcfc` |

Every recovered arm uses support digest `c4ef086be6858088e9281bf0c6f8fb2fe963a0f035f814bbe377d3737cea6df5`.

## Recovered four-arm metrics

These values are projections of the self-hashed rescue documents, not fabricated estimates and not official capacity results.

| Arm | Recall | Precision | F1 | False decline | Challenge | Review | Captured value | p95 latency |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| rules_only | 0.859490 | 0.148782 | 0.253655 | 0.874355 | 0.026393 | 0.000000 | 1.000000 | 4.061952 ms |
| ensemble_no_graph | 0.997449 | 0.947557 | 0.971863 | 0.000000 | 0.008019 | 0.002093 | 1.000000 | 3.382065 ms |
| ensemble_with_graph | 0.998673 | 0.958758 | 0.978309 | 0.000037 | 0.005720 | 0.002112 | 1.000000 | 3.543675 ms |
| full_sentinel | 0.999286 | 0.168456 | 0.288309 | 0.874374 | 0.028019 | 0.001178 | 1.000000 | 19.741710 ms |

The recovered full-Sentinel readiness gate passes all four family-recall targets, captured-value, review-rate, calibration, latency, and campaign-detection checks. It fails:

- false-decline rate: `0.8743738318` against an upper target of `0.001`;
- challenge rate: `0.0280186916` against an upper target of `0.02`;
- `benign_only` qualifying control.

Those failures are why the diagnostic readiness status is `not_ready`.

## Evidence and independent verification

Committed evidence:

- [`verified-report.json`](../../evidence/sentinel-v5-recovered-metrics/verified-report.json) — independently rebuilt compact report; self-digest `92b0add77fb41f34c8072b553c1c45e17dccc0b9a1387552252f0a42dde4e9a0`.
- [`source-rescue-receipt.json`](../../evidence/sentinel-v5-recovered-metrics/source-rescue-receipt.json) — semantic copy of the Kaggle rescue receipt.

Original downloaded archive: `/Users/dylanmoraes/Downloads/results (2).zip`, SHA-256 `00f3f8baf97e7f29b2235fd1637515a43998e05d4fb44019edd2bce2d297add9`.

The original compact artifact is `15,468,572` bytes with file SHA-256 `a8b5f61bc5f6764c530bd7f8de8960c0046f3efef32f7e00120029b076471857`. Its rescue-compact self-digest is `398a895178b08a939b1f50df28fea4129245d2be0cf96bbe50488a10a444bfea`; the top receipt self-digest is `758309bbb554feae7fbea3550170bf1b86927582e204176fb5388fcfbeaea1b3`.

To independently verify an extracted Kaggle output directory:

```bash
.venv/bin/python scripts/verify_defense_v5_kaggle_non_authoritative_rescue.py \
  --artifact-root /path/to/apar-v5-rescue \
  --report /tmp/sentinel-v5-recovered-metrics.json
```

The verifier checks the artifact file hash and size, both top-level self-digests, all four arm receipt self-digests, metric core and observation self-digests, a single shared support digest, exact arm order, exact Stage 00–60 order, terminal Stage 60 binding, non-authoritative flags, and readiness binding before emitting a new self-hashed report.

## Portable model remains separate

The competition demo remains the immediate usable model deliverable. Run it independently with:

```bash
.venv/bin/python scripts/run_sentinel_v5_demo.py \
  --scenario demo/sentinel-v5/scenarios.json
```

Its predictions are replay-bound to the accepted Stage 30 export. The recovered metrics above do not modify its model, calibrators, thresholds, scenarios, or manifest.
