[Submission home](README.md) · [Pitch deck](APAR_COMPETITION_DECK.pdf) · [Walkthrough](FIVE_MINUTE_WALKTHROUGH.md) · [Submission notices](THIRD_PARTY_NOTICES.md)

# APAR judge release checklist

> **Judge file 09 · Reproducibility runbook.** Cold-start the archive, verify
> exact replay and hashes, then use the release gate to confirm the tracked
> evidence boundary.

---

This release is a deterministic, offline-after-install demonstration of the
accepted Stage 30 `ensemble_with_graph` portable Sentinel model. It does not
train, adapt, publish, promote, or evaluate a new model.

## Archive cold start

Assumptions: CPython 3.12, `venv`, `pip`, and network access to the judge's
configured Python package index for the first dependency install. The release
gate covers macOS and Linux on common 64-bit Intel/ARM machines. Windows is not
release-gate verified.

From the directory containing `APAR-submission.zip`:

```bash
python3.12 -c "import hashlib,pathlib; p=pathlib.Path('APAR-submission.zip'); print(hashlib.sha256(p.read_bytes()).hexdigest())"
python3.12 -m zipfile -e APAR-submission.zip apar-release
cd apar-release/APAR
python3.12 -m venv .venv
.venv/bin/python -m pip install --requirement release/requirements-judge.txt
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m scripts.submission.runtime_verify --root .
```

The verifier must report:

- `replay_verified: true`;
- `scenario_count: 12`;
- portable bundle manifest SHA-256
  `52ed4c31d88cd0fc9d3f7557e3addff2044e33bb8991cc7e12a5b70385dbc900`;
- prediction SHA-256
  `5c8e4f40d1fb0316ae2d4fdb235c0043f69deff2816d89b870657db08131444e`;
- fallback trace SHA-256
  `4a4ee2ac9964ed5555fe5d9024cb68021045e686688607a7a27722cd835a9aa0`.

The expected action counts are three approvals, one challenge, two review holds,
and six decline holds. Every probability and action is replayed against the
hash-bound accepted scenario record with a tolerance of `1e-12`.

Typical dependency setup takes 1-5 minutes depending on network/cache speed.
The 12-scenario replay normally completes in under 15 seconds after setup and
uses no large training or evaluation job.

## Offline behavior and fallback

The archive does not vendor wheels. The first dependency installation therefore
needs the configured package index or a pre-populated package cache. After those
exact locked packages are installed, model loading, scenario replay, payload
verification, and fallback trace generation are fully offline. No Kaggle,
cloud, API key, credential, service, database, or network call is used.

The submission can include the complete console source with `--include-web`.
The clean-room gate installs the exact frontend lock and runs a production
build from the extracted archive. After dependency installation, model replay,
console build, and fallback operation are offline. Use the portable CLI as the
minimal fallback:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python scripts/run_sentinel_v5_demo.py \
  --scenario demo/sentinel-v5/scenarios.json
```

The judge-side verifier emits the deterministic fallback trace as part of its
JSON output. The web source, embedded evidence, and launcher are hash-inventoried
in the archive; browser end-to-end tests remain maintainer-side because they
require a preinstalled Chromium binary.

## Archive contents

- `SUBMISSION_MANIFEST.json`: source commit/tree, file sizes and SHA-256 hashes,
  model/bundle hashes, authority flags, runtime, build command, scan summary,
  and web status. It contains no generated timestamp.
- `demo/sentinel-v5/`: three CatBoost JSON members, three frozen calibrators,
  the portable specification, bundle manifest, and 12 scenarios.
- `src/apar/`: only the Python modules required by the portable scorer.
- `scripts/run_sentinel_v5_demo.py`: direct demo CLI.
- `scripts/submission/`: judge-side payload/replay verifier and deterministic
  fallback projection.
- `web/`: six-route React/TypeScript console, exact dependency lock, and embedded
  verified evidence.
- `scripts/run_apar_console.py` and `scripts/build_apar_console_evidence.py`:
  offline launcher and evidence preflight.
- `docs/submission/`: deck companion documents and video walkthrough.
- `evidence/sentinel-v5-recovered-metrics/`: verified but explicitly
  non-authoritative recovery receipt/report.
- `docs/experiments/defense-v5-locked-development-abort.json`: retained abort
  boundary for the earlier local locked-development attempt.
- `release/requirements-judge.txt`, `release/dependency-sbom.cdx.json`, and this
  notice/checklist: exact dependency, license, and operating instructions.

## Evidence boundary and known limitations

- The accepted demo model is Stage 30 `ensemble_with_graph` and the portable
  demo/recovery uses development-test seed 404 only.
- The recovered metrics are verified but non-authoritative. They are not
  accepted capacity evidence. The official chain is incomplete at Stage 70.
- No Kaggle locked-successor/seed-2404 chain ran. An earlier local
  locked-development attempt started and irreversibly aborted. It published no
  candidate manifest, candidate chunks, judge summary, or successful seed-2404
  result. Its retained boundary artifacts do not prove which internal sub-step
  it reached, so this release makes no stronger seed-execution claim.
- `full_sentinel` is not the champion. Its recovered readiness is `not_ready`;
  it fails the false-decline and challenge-rate diagnostic gates.
- Scenario metrics describe 12 curated synthetic replay examples. They are not
  population, sealed, confirmatory, production, or real-cardholder estimates.
- There is no production-readiness claim and no real cardholder data is shipped.
- The project-level license status is unspecified; see the third-party notice.

## Maintainer release gate

From a clean committed worktree on Python 3.12:

```bash
.venv/bin/python -m scripts.submission.release_gate
```

The gate refuses dirty state or changes to the protected evidence/model paths,
runs focused tests, Ruff and mypy, validates the dependency/license inventory,
builds the archive twice, compares its bytes, performs strict extraction into a
new temporary directory, installs only the exact judge lock, replays all 12
scenarios, and checks that no command changed tracked files.
