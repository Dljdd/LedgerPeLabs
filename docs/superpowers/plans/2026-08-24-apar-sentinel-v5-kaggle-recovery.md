# APAR Sentinel v5 Kaggle Staged Recovery Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the consumed monolithic Sentinel v5 attempt with a private, nine-stage Kaggle checkpoint pipeline that supports verified continuation from the last completed stage without executing seed 2404 during implementation or rehearsal.

**Architecture:** A closed Kaggle recovery protocol binds one successor attempt to nine ordered, content-addressed notebook outputs. Production components emit immutable corpus, feature, arm, control, metric, and final evidence checkpoints; a separate offline verifier reconstructs the chain without importing runner, model, control, metric, or storage implementations.

**Tech Stack:** Python 3.12.5, Pydantic v2, NumPy, CatBoost, canonical JSON/JSONL, zlib streaming, Pytest, Ruff, strict mypy, Kaggle private notebooks and saved notebook outputs.

**Spec:** `docs/superpowers/specs/2026-08-24-apar-sentinel-v5-kaggle-recovery-design.md`

## Global Constraints

- Preserve commits `6c3f7bef7e66f32c9a7156e710ee2ac953d499db`, `141e19c754c7bf54514eece5f7ff498ba53cd945`, and every earlier result/freeze/rejection byte.
- Never execute seed 2404 during Tasks 1-13. It may be parsed and compared only as a frozen binding.
- Do not run the production, adaptive, sealed, confirmatory, or publication workflow.
- Keep all Kaggle inputs, notebooks, outputs, and datasets private.
- Do not change population, model, feature, threshold, control, metric, bootstrap, economic, or readiness semantics.
- Keep CatBoost and all learned arms on their existing CPU code path.
- Capacity rehearsal uses production-size support with seed 404, never a reduced smoke profile.
- Frozen Kaggle gates are peak RSS `< 18 GiB`, wall time `< 6 hours`, output `< 10 GB`, identical deterministic stage digests, and complete offline verification.
- A completed checkpoint cannot be recomputed. Only a stage with no published manifest may be attempted again from its exact predecessor.
- A malformed or tampered published checkpoint is terminal and cannot authorize another stage attempt.
- Notebook stdout must not contain labels, probabilities, actions, metrics, economics, controls, gates, readiness, or final status.
- All commits use `Dylan Moraes <dylanmoraesdljdd@gmail.com>` with no additional author or co-author metadata.
- The approved chronology is GOVERNANCE (design and implementation plan) → RECOVERY → SOURCE3 → PREREGISTRATION3. Do not create intermediate production commits between RECOVERY and SOURCE3.

## File Map

### New production files

- `config/defense/defense-v5-kaggle-recovery.json` — closed stage order, safe/locked bindings, storage limits, and capacity gates.
- `src/apar/evaluation/v5_kaggle_protocol.py` — strict recovery protocol and environment/run binding models.
- `src/apar/evaluation/v5_checkpoint_storage.py` — deterministic streaming chunks, observational telemetry, exclusive manifests, and readers.
- `src/apar/evaluation/v5_staged_evidence.py` — typed stage payloads and execution functions.
- `src/apar/v5_kaggle_independent_verifier.py` — independent chain and final-evidence recomputation.
- `scripts/run_defense_v5_kaggle_stage.py` — closed next-stage CLI and capability boundary.
- `scripts/build_defense_v5_kaggle_notebooks.py` — deterministic private notebook and metadata generator.
- `scripts/verify_defense_v5_kaggle_evidence.py` — offline verifier CLI.
- `scripts/verify_defense_v5_kaggle_preexecution.py` — final clean-commit/predecessor/source/environment audit.
- `kaggle/defense_v5/00_authorize.ipynb` through `kaggle/defense_v5/80_finalize.ipynb` — generated minimal notebook sources.
- `kaggle/defense_v5/*-metadata.json` — generated private notebook metadata with internet disabled.

### Modified production files

- `src/apar/evaluation/v5_controls.py` — expose three closed control groups while preserving `execute_v5_controls` behavior.
- `src/apar/evaluation/v5_locked_evidence.py` — bind the final payload to the checkpoint-chain root without changing metric/evidence semantics.
- `src/apar/evaluation/v5_evidence_protocol.py` — include new implementation paths only; do not modify gates or existing closed modes.

### New evidence/config files

- `docs/experiments/defense-v5-locked-development-attempt.json` — existing failed receipt, preserved verbatim in RECOVERY.
- `docs/experiments/defense-v5-locked-development-abort.json` — canonical terminal record for the watchdog-aborted attempt.
- `config/defense/defense-v5-kaggle-preregistration.json` — created only after both Kaggle rehearsals.

### New tests

- `tests/evaluation/test_defense_v5_kaggle_recovery_record.py`
- `tests/evaluation/test_defense_v5_kaggle_protocol.py`
- `tests/evaluation/test_defense_v5_checkpoint_storage.py`
- `tests/evaluation/test_defense_v5_staged_evidence.py`
- `tests/evaluation/test_defense_v5_staged_controls.py`
- `tests/evaluation/test_defense_v5_kaggle_runner.py`
- `tests/evaluation/test_defense_v5_kaggle_verifier.py`
- `tests/evaluation/test_defense_v5_kaggle_notebooks.py`
- `tests/evaluation/test_defense_v5_kaggle_preexecution.py`

---

### Task 1: Preserve the consumed attempt as terminal recovery evidence

**Files:**

- Add: `docs/experiments/defense-v5-locked-development-attempt.json`
- Create: `docs/experiments/defense-v5-locked-development-abort.json`

**Interfaces:**

- Consumes: raw receipt SHA-256 `c9093272309605293f6377699df1810485901e0e3c5dfa9f81226ddea31151e8` and receipt self-digest `2cd207fdef0b808a8623843152195495d25d40b5d7903c5e71fd936611a09b93`.
- Produces: immutable abort record consumed by `load_v5_kaggle_protocol()` and the independent verifier.

- [ ] **Step 1: Reconfirm preserved bytes and absent outputs**

Run:

```bash
openssl dgst -sha256 docs/experiments/defense-v5-locked-development-attempt.json
openssl dgst -sha256 docs/experiments/defense-v5-development-result.json
test ! -e docs/experiments/defense-v5-locked-development-candidate.manifest.json
test ! -e docs/experiments/defense-v5-locked-development-candidate.manifest.json.chunks
test ! -e docs/experiments/defense-v5-locked-development-summary.json
```

Expected: receipt `c9093272…51e8`, historical result `af326f3a…d185`, and all three candidate paths absent.

- [ ] **Step 2: Create the canonical abort record with `apply_patch`**

Use this exact schema and facts; calculate `record_sha256` over the document without that field using sorted compact JSON:

```json
{
  "schema_version": "apar-sentinel-v5-locked-attempt-abort/1",
  "status": "aborted_host_kernel_watchdog",
  "attempt_receipt_path": "docs/experiments/defense-v5-locked-development-attempt.json",
  "attempt_receipt_raw_sha256": "c9093272309605293f6377699df1810485901e0e3c5dfa9f81226ddea31151e8",
  "attempt_receipt_self_sha256": "2cd207fdef0b808a8623843152195495d25d40b5d7903c5e71fd936611a09b93",
  "rejected_preregistration_commit": "141e19c754c7bf54514eece5f7ff498ba53cd945",
  "rejected_source_commit": "6c3f7bef7e66f32c9a7156e710ee2ac953d499db",
  "started_at_utc": "2026-08-24T13:32:54.948338Z",
  "panic_at_utc": "2026-08-24T13:48:11Z",
  "panic_reason": "watchdog timeout: no checkins from watchdogd in 94 seconds",
  "panic_evidence_sha256": "af55a482c5d468a044ba3739084884ccdadb9ccc410d3d581221453c5944e595",
  "reset_counter_sha256": "4497ef9844ac4fa44f16d4942ed91521dd588b3dbf4b19aea03ae91b2a4f6d08",
  "candidate_manifest_published": false,
  "candidate_chunks_published": false,
  "judge_summary_published": false,
  "historical_result_sha256": "af326f3a0fcbbe12c9b8623fc7d82a1ba6d0f327ec9a80f462cacd4bea1dd185",
  "retry_permitted": false,
  "record_sha256": "dc0743f1fe93356ea1e06af188d7a0e08cf46f0fcea02a674dbc1b2ec63d94d8"
}
```

- [ ] **Step 3: Verify canonical digest and exact path set**

Run:

```bash
.venv/bin/python -c 'import hashlib,json,pathlib; p=pathlib.Path("docs/experiments/defense-v5-locked-development-abort.json"); d=json.loads(p.read_bytes()); claimed=d.pop("record_sha256"); actual=hashlib.sha256(json.dumps(d,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest(); assert claimed==actual; print(actual)'
git diff --check
git status --short
```

Expected: only the receipt and abort record are new besides this already committed plan/spec history.

- [ ] **Step 4: Commit RECOVERY under Dylan's identity**

```bash
git add docs/experiments/defense-v5-locked-development-attempt.json \
  docs/experiments/defense-v5-locked-development-abort.json
git -c user.name="Dylan Moraes" \
  -c user.email="dylanmoraesdljdd@gmail.com" \
  commit -m "chore: preserve aborted sentinel attempt"
```

Verify `git diff-tree --no-commit-id --name-only -r HEAD` lists exactly those two paths.

### Task 2: Freeze the closed Kaggle stage protocol

**Files:**

- Create: `src/apar/evaluation/v5_kaggle_protocol.py`
- Create: `config/defense/defense-v5-kaggle-recovery.json`
- Create: `tests/evaluation/test_defense_v5_kaggle_protocol.py`
- Create: `tests/evaluation/test_defense_v5_kaggle_recovery_record.py`

**Interfaces:**

- Consumes: existing `V5DevelopmentProtocol`, `V5EvidenceProtocol`, failed receipt, and abort record.
- Produces:

```python
class V5KaggleMode(str, Enum):
    CAPACITY_VALIDATION = "kaggle_capacity_validation"
    LOCKED_SUCCESSOR = "kaggle_locked_successor"

class V5KaggleStage(str, Enum):
    AUTHORIZE = "00_authorize"
    CORPUS = "10_corpus"
    FEATURES = "20_features"
    ARMS = "30_arms"
    LABEL_SHUFFLE = "40_label_shuffle"
    INVARIANCE_CONTROLS = "50_invariance_controls"
    SINGLE_CLASS_CONTROLS = "60_single_class_controls"
    METRICS = "70_metrics"
    FINALIZE = "80_finalize"

def load_v5_kaggle_protocol(path: Path, *, root: Path) -> V5KaggleProtocol: ...
def resolve_next_v5_kaggle_stage(predecessor: V5CheckpointManifest | None) -> V5KaggleStage: ...
```

- [ ] **Step 1: Write protocol RED tests**

```python
def test_kaggle_protocol_has_exact_closed_stage_order() -> None:
    protocol = load_v5_kaggle_protocol(CONFIG, root=ROOT)
    assert protocol.stage_order == tuple(V5KaggleStage)
    assert protocol.capacity.mode is V5KaggleMode.CAPACITY_VALIDATION
    assert protocol.capacity.development_test_seed == 404
    assert protocol.capacity.profile == "production"
    assert protocol.locked.mode is V5KaggleMode.LOCKED_SUCCESSOR
    assert protocol.locked.development_test_seed == 2404
    assert protocol.locked.profile == "production"
    assert protocol.resources.max_peak_rss_bytes == 18 * 1024**3
    assert protocol.resources.max_stage_seconds == 6 * 60 * 60
    assert protocol.resources.max_stage_output_bytes == 10_000_000_000
```

Add parametrized rejection for arbitrary seed/profile, reordered/duplicate stages, weakened resource gates, unknown fields, missing abort binding, safe↔locked relabeling, and changed existing result/safe-core hashes.

- [ ] **Step 2: Run RED**

```bash
.venv/bin/python -m pytest \
  tests/evaluation/test_defense_v5_kaggle_protocol.py \
  tests/evaluation/test_defense_v5_kaggle_recovery_record.py -q
```

Expected: collection failure because `v5_kaggle_protocol` does not exist.

- [ ] **Step 3: Implement strict frozen models and loader**

Use `ConfigDict(frozen=True, extra="forbid")`, exact literals, exact lowercase SHA-256 fields, and this resource model:

```python
class V5KaggleResourceGates(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    max_peak_rss_bytes: Literal[19327352832]
    max_stage_seconds: Literal[21600]
    max_stage_output_bytes: Literal[10000000000]
    max_checkpoint_chunk_bytes: Literal[67108864]
    max_checkpoint_chunks: Literal[160]
```

Validate the abort record's canonical digest and require `retry_permitted is False` before returning the protocol.

- [ ] **Step 4: Run GREEN and static checks**

```bash
.venv/bin/python -m pytest \
  tests/evaluation/test_defense_v5_kaggle_protocol.py \
  tests/evaluation/test_defense_v5_kaggle_recovery_record.py -q
.venv/bin/ruff check src/apar/evaluation/v5_kaggle_protocol.py \
  tests/evaluation/test_defense_v5_kaggle_protocol.py \
  tests/evaluation/test_defense_v5_kaggle_recovery_record.py
.venv/bin/mypy --strict src/apar/evaluation/v5_kaggle_protocol.py
```

Expected: all commands pass. Do not commit yet; SOURCE3 remains a single later commit.

### Task 3: Implement streaming checkpoint storage and resource telemetry

**Files:**

- Create: `src/apar/evaluation/v5_checkpoint_storage.py`
- Create: `tests/evaluation/test_defense_v5_checkpoint_storage.py`

**Interfaces:**

- Consumes: `V5KaggleProtocol`, exact stage payload record iterator, predecessor manifest, environment binding.
- Produces:

```python
@dataclass(frozen=True, slots=True)
class V5CheckpointInput:
    kind: str
    key: str
    canonical_bytes: bytes

class V5CheckpointManifest(BaseModel): ...
class V5CheckpointObservation(BaseModel): ...

def publish_v5_checkpoint(
    *, output_root: Path, stage: V5KaggleStage,
    run_binding_sha256: str, attempt_receipt_sha256: str,
    predecessor: V5CheckpointManifest | None,
    records: Iterable[V5CheckpointInput],
    environment: V5KaggleEnvironmentBinding,
    observation: V5CheckpointObservation,
    limits: V5KaggleResourceGates,
) -> V5CheckpointManifest: ...

def iter_v5_checkpoint_records(
    *, output_root: Path, limits: V5KaggleResourceGates
) -> Iterator[V5CheckpointInput]: ...
```

- [ ] **Step 1: Write storage RED tests**

```python
def test_checkpoint_is_manifest_last_content_addressed_and_reconstructable(tmp_path: Path) -> None:
    manifest = publish_v5_checkpoint(
        output_root=tmp_path / "out",
        stage=V5KaggleStage.CORPUS,
        run_binding_sha256="1" * 64,
        attempt_receipt_sha256="2" * 64,
        predecessor=_authorization_manifest(),
        records=(V5CheckpointInput("row", "event-1", b'{"x":1}'),),
        environment=_environment(), observation=_observation(), limits=_limits(),
    )
    assert manifest.stage == "10_corpus"
    assert [item.canonical_bytes for item in iter_v5_checkpoint_records(
        output_root=tmp_path / "out", limits=_limits()
    )] == [b'{"x":1}']
```

Parametrize deletion, byte mutation, reordered chunks, extra chunks, manifest rebinding, predecessor substitution, observational deletion/mutation, environment substitution, symlink/hardlink targets, manifest-first visibility, directory-fsync failure, malformed published manifests, duplicate stage manifests, output over 10 GB via injected limits, and peak/time gate violations.

- [ ] **Step 2: Run RED**

```bash
.venv/bin/python -m pytest tests/evaluation/test_defense_v5_checkpoint_storage.py -q
```

Expected: import failure for `v5_checkpoint_storage`.

- [ ] **Step 3: Implement deterministic streaming format**

Encode each record as a length-prefixed canonical envelope:

```python
header = json.dumps(
    {"kind": record.kind, "key": record.key, "bytes": len(record.canonical_bytes)},
    sort_keys=True, separators=(",", ":"), allow_nan=False,
).encode()
stream.write(len(header).to_bytes(8, "big"))
stream.write(header)
stream.write(record.canonical_bytes)
```

Feed the stream through `zlib.compressobj(level=9, wbits=31)` into fixed 64 MiB chunks. Use temporary files, file fsync, `os.link` no-replace publication, and parent-directory fsync. Publish `checkpoint.manifest.json` last. The reader independently recomputes every chunk and record digest and rejects trailing bytes.

Sample RSS on Linux from `/proc/self/status` and host availability from `/proc/meminfo`; allow an injected sampler only in tests. Store real samples in `observational.json` and exclude them through one exact tuple-of-paths schema.

- [ ] **Step 4: Run GREEN and static checks**

```bash
.venv/bin/python -m pytest tests/evaluation/test_defense_v5_checkpoint_storage.py -q
.venv/bin/ruff check src/apar/evaluation/v5_checkpoint_storage.py \
  tests/evaluation/test_defense_v5_checkpoint_storage.py
.venv/bin/mypy --strict src/apar/evaluation/v5_checkpoint_storage.py
```

### Task 4: Define typed stage payloads and authorization capability

**Files:**

- Create: `src/apar/evaluation/v5_staged_evidence.py`
- Create: `tests/evaluation/test_defense_v5_staged_evidence.py`

**Interfaces:**

- Consumes: checkpoint records, existing v5 corpus/features/arms/evidence types, and a sealed stage capability.
- Produces:

```python
@dataclass(frozen=True, slots=True)
class V5StageCapability:
    stage: V5KaggleStage
    mode: V5KaggleMode
    run_binding_sha256: str
    attempt_receipt_sha256: str
    predecessor_manifest_sha256: str | None
    seal: object

def execute_v5_authorization_stage(
    *, root: Path, capability: V5StageCapability,
) -> Iterator[V5CheckpointInput]: ...
```

- [ ] **Step 1: Write RED capability and schema tests**

```python
def test_seed_2404_cannot_reach_corpus_without_stage_10_capability(monkeypatch: pytest.MonkeyPatch) -> None:
    reached = False
    def forbidden(*args: object, **kwargs: object) -> object:
        nonlocal reached
        reached = True
        raise AssertionError("population boundary reached")
    monkeypatch.setattr(staged, "build_v5_corpus", forbidden)
    with pytest.raises(PermissionError):
        staged.execute_v5_corpus_stage(
            root=ROOT, capability=_forged_capability(V5KaggleStage.CORPUS)
        )
    assert reached is False
```

Add exact schema round-trips for authorization, corpus, feature, arm, each control group, metric, and final payload records. Reject pickle, live model objects, private keys, unordered support, unknown record kinds, labels embedded in feature records, and future stages reading unavailable predecessors.

- [ ] **Step 2: Run RED**

```bash
.venv/bin/python -m pytest tests/evaluation/test_defense_v5_staged_evidence.py -q
```

Expected: missing `v5_staged_evidence` module.

- [ ] **Step 3: Implement sealed capability issuance**

Keep the seal module-private. Issue capabilities only after validating the exact predecessor manifest and mode/stage transition:

```python
def _issue_stage_capability(
    *, protocol: V5KaggleProtocol, mode: V5KaggleMode,
    attempt_receipt_sha256: str,
    predecessor: V5CheckpointManifest | None,
) -> V5StageCapability:
    expected = resolve_next_v5_kaggle_stage(predecessor)
    return V5StageCapability(
        stage=expected, mode=mode,
        run_binding_sha256=protocol.run_binding_sha256(mode),
        attempt_receipt_sha256=attempt_receipt_sha256,
        predecessor_manifest_sha256=(None if predecessor is None else predecessor.manifest_sha256),
        seal=_STAGE_CAPABILITY_SEAL,
    )
```

Implement only `_issue_stage_capability` and `execute_v5_authorization_stage` in this task. Stage 00 emits the exact source, preregistration, recovery, safe-core, environment, support-plan, command, and authorization bindings. Tasks 5-8 add their named stage functions under focused RED tests; Task 9 adds the closed dispatcher only after every concrete stage function exists.

- [ ] **Step 4: Run GREEN and static checks**

```bash
.venv/bin/python -m pytest tests/evaluation/test_defense_v5_staged_evidence.py -q
.venv/bin/ruff check src/apar/evaluation/v5_staged_evidence.py \
  tests/evaluation/test_defense_v5_staged_evidence.py
.venv/bin/mypy --strict src/apar/evaluation/v5_staged_evidence.py
```

### Task 5: Implement corpus and causal-feature checkpoints

**Files:**

- Modify: `src/apar/evaluation/v5_staged_evidence.py`
- Modify: `tests/evaluation/test_defense_v5_staged_evidence.py`

**Interfaces:**

- Consumes: valid Stage-00 or Stage-10 predecessor and closed mode.
- Produces:

```python
def execute_v5_corpus_stage(
    *, root: Path, capability: V5StageCapability
) -> Iterator[V5CheckpointInput]: ...

def load_v5_corpus_checkpoint(
    *, checkpoint_root: Path, limits: V5KaggleResourceGates
) -> V5Corpus: ...

@dataclass(frozen=True, slots=True)
class V5PreparedPartition:
    partition: str
    matrix: NDArray[np.float64]
    labels: NDArray[np.int_]
    event_ids: tuple[str, ...]
    campaign_ids: tuple[str, ...]
    amounts: NDArray[np.float64]
    trust_failures: tuple[bool, ...]
    feature_batch: SentinelFeatureBatch
    training_evidence: V5TrainingPartitionEvidence | None

def execute_v5_feature_stage(
    *, root: Path, capability: V5StageCapability,
    corpus_checkpoint_root: Path,
) -> Iterator[V5CheckpointInput]: ...
```

- [ ] **Step 1: Write corpus RED tests with test-only seed 404**

```python
def test_corpus_stage_matches_real_production_profile_support_at_safe_seed() -> None:
    records = tuple(execute_v5_corpus_stage(
        root=ROOT, capability=_capacity_capability(V5KaggleStage.CORPUS)
    ))
    header = _record_json(records, "corpus_header")
    assert header["profile"] == "production"
    assert header["development_test_seed"] == 404
    assert header["support_plan_sha256"] == _capacity_support_plan_sha256()
    assert header["execution_path"] == [
        "PopulationGenerator", "CampaignGenerator", "SimulationEngine",
        "rail_adapter", "ledger", "execution_evidence", "V5DecisionRow",
    ]
```

Use a small injected protocol fixture for ordinary unit speed, then one guarded integration test marked `v5_capacity_local` which asserts production support counts without using seed 2404. Add round-trip/tamper tests for every decision/execution manifest, partition order, ledger/trust records, and support digest.

- [ ] **Step 2: Run corpus RED**

```bash
.venv/bin/python -m pytest \
  tests/evaluation/test_defense_v5_staged_evidence.py -k 'corpus' -q
```

Expected: `execute_v5_corpus_stage` is absent.

- [ ] **Step 3: Implement corpus records without changing generation**

Build the protocol by copying only `seeds.development_test` for capacity mode and recomputing `protocol_sha256`; always call `build_v5_corpus(protocol, profile=V5Profile.PRODUCTION)`. Locked mode requires the Stage-10 capability and exact seed 2404. Serialize records in canonical partition/decision/execution order:

```python
yield V5CheckpointInput("corpus_header", "corpus", canonical_bytes(header))
for partition_name in ("train", "calibration", "threshold", "development_test"):
    partition = corpus.partitions[partition_name]
    for row in partition.decisions:
        yield V5CheckpointInput("decision_row", f"{partition_name}:{row.event_id}", canonical_bytes(row.model_dump(mode="json")))
    for manifest in partition.executions:
        yield V5CheckpointInput("execution_manifest", f"{partition_name}:{manifest.evidence_sha256}", canonical_bytes(manifest.model_dump(mode="json")))
```

The loader validates `V5Corpus` from exact records and recomputes its digest rather than trusting the header.

- [ ] **Step 4: Write feature RED tests**

```python
def test_feature_stage_uses_checkpointed_corpus_and_preserves_order() -> None:
    records = tuple(execute_v5_feature_stage(
        root=ROOT,
        capability=_capacity_capability(V5KaggleStage.FEATURES),
        corpus_checkpoint_root=SAFE_CORPUS_CHECKPOINT,
    ))
    prepared = load_v5_feature_checkpoint(records)
    assert tuple(prepared) == ("train", "calibration", "threshold", "development_test")
    assert prepared["development_test"].event_ids == tuple(
        row.event_id for row in load_v5_corpus_checkpoint(
            checkpoint_root=SAFE_CORPUS_CHECKPOINT, limits=_limits()
        ).partitions["development_test"].decisions
    )
```

Add forbidden-field mutation, identity rename, equal-time order, trust-failure alignment, label separation, matrix dtype/shape, and batch-digest tests.

- [ ] **Step 5: Implement canonical numeric-array records**

Encode matrices as explicit little-endian C-order bytes, not pickle:

```python
def encode_f64_matrix(matrix: NDArray[np.float64]) -> tuple[dict[str, object], bytes]:
    canonical = np.ascontiguousarray(matrix, dtype="<f8")
    raw = canonical.tobytes(order="C")
    return ({"dtype": "<f8", "shape": list(canonical.shape),
             "sha256": hashlib.sha256(raw).hexdigest()}, raw)
```

Store labels as `<i8`, amounts as `<f8`, trust flags as one byte per row, and metadata as canonical JSON. Reconstruct with exact length checks and `allow_pickle=False` semantics. Build training evidence through existing `build_v5_training_partition_evidence` and verify feature catalog/provenance digests.

- [ ] **Step 6: Run GREEN and regression equivalence**

```bash
.venv/bin/python -m pytest \
  tests/evaluation/test_defense_v5_staged_evidence.py -k 'corpus or feature' -q
.venv/bin/python -m pytest \
  tests/evaluation/test_defense_v5_execution_evidence.py \
  tests/evaluation/test_defense_v5_population.py \
  tests/evaluation/test_defense_v5_runner_feature_binding.py -q
.venv/bin/ruff check src/apar/evaluation/v5_staged_evidence.py \
  tests/evaluation/test_defense_v5_staged_evidence.py
.venv/bin/mypy --strict src/apar/evaluation/v5_staged_evidence.py
```

### Task 6: Train and score the four arms from the feature checkpoint

**Files:**

- Modify: `src/apar/evaluation/v5_staged_evidence.py`
- Modify: `tests/evaluation/test_defense_v5_staged_evidence.py`

**Interfaces:**

- Consumes: Stage-20 prepared partitions and Stage-10 execution manifests.
- Produces:

```python
def execute_v5_arm_stage(
    *, root: Path, capability: V5StageCapability,
    corpus_checkpoint_root: Path,
    feature_checkpoint_root: Path,
) -> Iterator[V5CheckpointInput]: ...

def load_v5_arm_checkpoint(
    *, checkpoint_root: Path, limits: V5KaggleResourceGates
) -> tuple[V5EvaluationResult, V5EvaluationResult, V5EvaluationResult, V5EvaluationResult]: ...
```

- [ ] **Step 1: Write arm-stage RED tests**

```python
def test_arm_stage_matches_monolithic_safe_result_on_exact_support() -> None:
    staged = load_v5_arm_checkpoint(
        checkpoint_root=_run_safe_arm_stage(), limits=_limits()
    )
    monolithic = _run_existing_safe_arm_path()
    assert tuple(item.arm for item in staged) == (
        "rules_only", "ensemble_no_graph", "ensemble_with_graph", "full_sentinel"
    )
    assert [_deterministic_arm_document(item) for item in staged] == [
        _deterministic_arm_document(item) for item in monolithic
    ]
```

Add tests for identical ordered support, no graph features in `ensemble_no_graph`, graph features reaching `ensemble_with_graph`, trust routing, disabled-component invariance, no access to full-sentinel outputs by other arms, and exact arm-spec digests.

- [ ] **Step 2: Run RED**

```bash
.venv/bin/python -m pytest \
  tests/evaluation/test_defense_v5_staged_evidence.py -k 'arm_stage' -q
```

Expected: missing arm-stage function.

- [ ] **Step 3: Implement arm execution from prepared arrays**

Call the same existing functions used by the runner:

```python
trained = train_v5_arm_set(
    configuration=configuration, catalog=catalog,
    x_train=train.matrix, y_train=train.labels,
    x_calibration=calibration.matrix, y_calibration=calibration.labels,
    x_threshold=threshold.matrix, y_threshold=threshold.labels,
    bootstrap_seed=protocol.seeds.bootstrap,
    train_evidence=train.training_evidence,
    calibration_evidence=calibration.training_evidence,
    threshold_evidence=threshold.training_evidence,
)
scores = score_v5_arm_set(
    trained=trained, catalog=catalog,
    features_matrix=development.matrix,
    support=build_v5_arm_support_rows(development_rows),
    execution_artifacts=build_v5_execution_artifacts(development_executions),
    trust_failures=development.trust_failures,
    feature_provenance=development.feature_batch.provenance,
)
```

Bind each score with existing `evaluate_v5_arm` and `bind_v5_evaluation_result`. Emit exactly four ordered `arm_result` records. Do not serialize mutable model objects.

- [ ] **Step 4: Run GREEN and arm regressions**

```bash
.venv/bin/python -m pytest \
  tests/evaluation/test_defense_v5_staged_evidence.py -k 'arm_stage' -q
.venv/bin/python -m pytest \
  tests/evaluation/test_defense_v5_arm_runner.py \
  tests/evaluation/test_defense_v5_comparison_arms.py \
  tests/evaluation/test_defense_v5_arm_evidence_hardening.py -q
```

### Task 7: Split controls into three closed, equivalent groups

**Files:**

- Modify: `src/apar/evaluation/v5_controls.py`
- Modify: `src/apar/evaluation/v5_staged_evidence.py`
- Create: `tests/evaluation/test_defense_v5_staged_controls.py`

**Interfaces:**

- Consumes: checkpointed corpus and existing frozen control configuration.
- Produces:

```python
class V5ControlGroup(str, Enum):
    LABEL_SHUFFLE = "label_shuffle"
    INVARIANCE = "invariance"
    SINGLE_CLASS = "single_class"

class V5ExecutedControlGroup(BaseModel): ...

def execute_v5_control_group(
    *, group: V5ControlGroup, protocol: V5DevelopmentProtocol,
    evidence_protocol: V5EvidenceProtocol, corpus: V5Corpus,
    catalog: SentinelFeatureCatalog, configuration: V5ArmConfiguration,
    mode: V5RunMode,
) -> V5ExecutedControlGroup: ...

def assemble_v5_control_suite(
    groups: Sequence[V5ExecutedControlGroup],
) -> V5ExecutedControlSuite: ...

def execute_v5_control_stage(
    *, root: Path, capability: V5StageCapability,
    corpus_checkpoint_root: Path,
) -> Iterator[V5CheckpointInput]: ...
```

- [ ] **Step 1: Write RED equivalence and isolation tests**

```python
def test_grouped_controls_equal_existing_suite_except_observational_latency() -> None:
    grouped = assemble_v5_control_suite(tuple(
        execute_v5_control_group(group=group, **_safe_control_inputs())
        for group in V5ControlGroup
    ))
    existing = execute_v5_controls(**_safe_control_inputs())
    assert _deterministic_controls(grouped) == _deterministic_controls(existing)
    assert tuple(item.name for item in grouped.controls) == (
        "label_shuffle", "identity_rename", "future_causality",
        "equal_time_isolation", "feature_leakage",
        "benign_only", "fraud_only_diagnostic",
    )
```

Reject missing/duplicate/reordered groups, controls in the wrong group, different implementation/spec/arm digests, safe↔locked mode changes, and group access to an arm/control output it does not require.

- [ ] **Step 2: Run RED**

```bash
.venv/bin/python -m pytest tests/evaluation/test_defense_v5_staged_controls.py -q
```

Expected: `V5ControlGroup` import failure.

- [ ] **Step 3: Refactor without changing full-suite behavior**

Retain one `_ControlRuntime` builder and add an internal exact dispatch:

```python
_GROUP_NAMES = {
    V5ControlGroup.LABEL_SHUFFLE: ("label_shuffle",),
    V5ControlGroup.INVARIANCE: (
        "identity_rename", "future_causality",
        "equal_time_isolation", "feature_leakage",
    ),
    V5ControlGroup.SINGLE_CLASS: ("benign_only", "fraud_only_diagnostic"),
}
```

`execute_v5_controls` must continue to build one runtime and execute all seven controls in the original order. `execute_v5_control_group` builds a runtime for one staged job and executes only its frozen names.

- [ ] **Step 4: Add Stage 40/50/60 record emitters**

Implement `execute_v5_control_stage` in `v5_staged_evidence.py`. It infers the one allowed `V5ControlGroup` from `capability.stage`, validates the mode through existing control binding, and emits one immutable `control_group` record.

- [ ] **Step 5: Run GREEN and complete control regressions**

```bash
.venv/bin/python -m pytest \
  tests/evaluation/test_defense_v5_staged_controls.py \
  tests/evaluation/test_defense_v5_executed_controls.py \
  tests/evaluation/test_defense_v5_controls.py -q
.venv/bin/ruff check src/apar/evaluation/v5_controls.py \
  src/apar/evaluation/v5_staged_evidence.py \
  tests/evaluation/test_defense_v5_staged_controls.py
.venv/bin/mypy --strict src/apar/evaluation/v5_controls.py \
  src/apar/evaluation/v5_staged_evidence.py
```

### Task 8: Build metric and final payload checkpoints

**Files:**

- Modify: `src/apar/evaluation/v5_staged_evidence.py`
- Modify: `src/apar/evaluation/v5_locked_evidence.py`
- Modify: `tests/evaluation/test_defense_v5_staged_evidence.py`
- Modify: `tests/evaluation/test_defense_v5_locked_evidence_contract.py`

**Interfaces:**

- Consumes: exact arm checkpoint and all three control-group checkpoints.
- Produces:

```python
class V5CheckpointChainBinding(BaseModel):
    schema_version: Literal["apar-sentinel-v5-checkpoint-chain/1"]
    attempt_receipt_sha256: str
    predecessor_stage_manifest_sha256: tuple[tuple[str, str], ...]
    predecessor_chain_root_sha256: str

def execute_v5_metric_stage(
    *, root: Path, capability: V5StageCapability,
    corpus_checkpoint_root: Path,
    arm_checkpoint_root: Path,
    control_checkpoint_roots: Sequence[Path],
) -> Iterator[V5CheckpointInput]: ...
def execute_v5_finalize_stage(
    *, root: Path, capability: V5StageCapability,
    predecessor_checkpoint_roots: Sequence[Path],
) -> Iterator[V5CheckpointInput]: ...
def build_v5_staged_locked_evidence_payload(
    *, chain: V5CheckpointChainBinding,
    run_binding: V5LockedEvidenceRunBinding,
    evidence_protocol: V5EvidenceProtocol,
    catalog_sha256: str,
    arm_results: Sequence[V5EvaluationResult],
    controls: V5ExecutedControlSuite,
) -> bytes: ...
```

- [ ] **Step 1: Write RED metric/final equivalence tests**

```python
def test_staged_metrics_and_payload_match_existing_complete_semantics() -> None:
    staged = _build_safe_staged_payload()
    existing = _build_safe_monolithic_payload()
    assert _unpack_complete_metrics(staged) == _unpack_complete_metrics(existing)
    assert _unpack_controls(staged) == _unpack_controls(existing)
    assert _unpack_readiness(staged) == _unpack_readiness(existing)
    assert _chain_binding(staged).predecessor_stage_manifest_sha256[-1][0] == "70_metrics"
    assert _final_manifest(staged).stage == "80_finalize"
    assert _final_manifest(staged).predecessor_manifest_sha256 == _metric_manifest_sha256(staged)
```

Add per-family, ECE, numerator/denominator, ledger economics, campaign bootstrap, undefined semantics, support order, attempt receipt, and chain-root mutation tests.

- [ ] **Step 2: Run RED**

```bash
.venv/bin/python -m pytest \
  tests/evaluation/test_defense_v5_staged_evidence.py -k 'metric or final' \
  tests/evaluation/test_defense_v5_locked_evidence_contract.py -q
```

Expected: missing staged metric/final functions or chain field.

- [ ] **Step 3: Implement Stage 70 from existing metric functions**

Use `assemble_v5_control_suite`, then call `evaluate_v5_complete_result` for the four exact arm results and `build_v5_readiness_evidence` for full sentinel. Emit canonical `complete_metrics`, `executed_controls`, and `readiness` records; never read development outcomes to change definitions.

- [ ] **Step 4: Extend the locked payload with a non-self-referential chain binding**

Add `checkpoint_chain: V5CheckpointChainBinding` to a new payload schema version. Keep all existing execution artifacts, arm results, complete metrics, controls, readiness, deterministic core, and observational latency fields. The payload binds the completed Stage 00-70 predecessor manifests and their chain root. `execute_v5_finalize_stage` independently reads all eight predecessors, builds and verifies the payload, and emits it as Stage-80 records. The Stage-80 manifest then binds that payload and the Stage-70 predecessor; its manifest digest is the final nine-stage chain root. Never place the Stage-80 manifest digest inside its own payload or exclusion schema.

- [ ] **Step 5: Run GREEN and metric/economic regressions**

```bash
.venv/bin/python -m pytest \
  tests/evaluation/test_defense_v5_staged_evidence.py \
  tests/evaluation/test_defense_v5_locked_evidence_contract.py \
  tests/evaluation/test_defense_v5_complete_metrics.py \
  tests/evaluation/test_defense_v5_calibration.py \
  tests/evaluation/test_defense_v5_readiness_tamper.py -q
.venv/bin/ruff check src/apar/evaluation/v5_staged_evidence.py \
  src/apar/evaluation/v5_locked_evidence.py
.venv/bin/mypy --strict src/apar/evaluation/v5_staged_evidence.py \
  src/apar/evaluation/v5_locked_evidence.py
```

### Task 9: Add the closed next-stage runner and continuation rules

**Files:**

- Create: `scripts/run_defense_v5_kaggle_stage.py`
- Create: `tests/evaluation/test_defense_v5_kaggle_runner.py`
- Modify: `src/apar/evaluation/v5_staged_evidence.py`

**Interfaces:**

- Consumes: root, private input roots, safe evidence path, approved commit, mode-specific authorization, and exact predecessor output.
- Produces:

```python
def execute_next_v5_kaggle_stage(
    *, root: Path, input_root: Path, output_root: Path,
    safe_evidence: Path, approved_commit: str,
    authorization_granted: bool,
    authority: V5KaggleStageAuthority,
) -> V5CheckpointManifest: ...
```

- [ ] **Step 1: Capture monolithic RED and staged-runner RED**

```python
def test_legacy_locked_runner_cannot_accept_predecessor_or_checkpoint() -> None:
    arguments = _cli_arguments(ROOT / "scripts/run_defense_v5_locked_development.py")
    assert "--predecessor" not in arguments
    assert "--checkpoint" not in arguments

def test_completed_stage_advances_and_incomplete_stage_can_repeat(tmp_path: Path) -> None:
    authority = _SyntheticStageAuthority(fail_once_at=V5KaggleStage.FEATURES)
    _publish_authorization_and_corpus(tmp_path, authority)
    with pytest.raises(RuntimeError, match="simulated interruption"):
        _advance(tmp_path, authority)
    assert authority.executed == [V5KaggleStage.FEATURES]
    _advance(tmp_path, authority)
    assert authority.executed == [V5KaggleStage.FEATURES, V5KaggleStage.FEATURES]
```

Also test that a valid current-stage manifest advances once, a malformed visible manifest is terminal, a duplicate manifest is terminal, a later-stage file is terminal, and Stage 10 cannot start without Stage 00 authorization.

- [ ] **Step 2: Run RED**

```bash
.venv/bin/python -m pytest tests/evaluation/test_defense_v5_kaggle_runner.py -q
```

Expected: missing runner module.

- [ ] **Step 3: Implement exact CLI and stage inference**

Public arguments are exactly:

```text
--root
--input-root
--output-root
--safe-evidence
--approved-commit
--authorize-successor
```

There is no seed, profile, stage, resume, retry, force, delete, output-name, test-authority, or model override. Infer the next stage solely from the verified predecessor. `--authorize-successor` is accepted only for Stage 00 and rejected later.

The runner order is:

```text
preflight → verify environment/input topology → verify predecessor →
issue exact stage capability → start telemetry → execute stage →
stop telemetry → publish chunks → publish observation → publish manifest last →
independently read checkpoint → print redacted receipt
```

- [ ] **Step 4: Run GREEN and crash/mutation tests**

```bash
.venv/bin/python -m pytest \
  tests/evaluation/test_defense_v5_kaggle_runner.py \
  tests/evaluation/test_defense_v5_checkpoint_storage.py -q
.venv/bin/ruff check scripts/run_defense_v5_kaggle_stage.py \
  tests/evaluation/test_defense_v5_kaggle_runner.py
.venv/bin/mypy --strict scripts/run_defense_v5_kaggle_stage.py
```

### Task 10: Implement the independent staged-evidence verifier

**Files:**

- Create: `src/apar/v5_kaggle_independent_verifier.py`
- Create: `scripts/verify_defense_v5_kaggle_evidence.py`
- Create: `tests/evaluation/test_defense_v5_kaggle_verifier.py`

**Interfaces:**

- Consumes: the nine downloaded checkpoint roots, optional Stage-80 final root, frozen recovery/protocol/preregistration/config/source bindings, and no live execution objects.
- Produces:

```python
class V5KaggleVerificationReport(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    schema_version: Literal["apar-sentinel-v5-kaggle-verification/1"]
    valid: bool
    mode: Literal["kaggle_capacity_validation", "kaggle_locked_successor"]
    verified_stage_ids: tuple[str, ...]
    chain_root_sha256: str
    deterministic_stage_sha256: tuple[tuple[str, str], ...]
    observational_stage_sha256: tuple[tuple[str, str], ...]
    final_payload_sha256: str | None
    verifier_sha256: str

def verify_v5_kaggle_evidence(
    *, root: Path, checkpoint_roots: Sequence[Path],
    final_root: Path | None, expected_mode: str,
) -> V5KaggleVerificationReport: ...
```

- [ ] **Step 1: Write RED import-boundary and valid-chain tests**

```python
def test_independent_verifier_has_no_production_execution_imports() -> None:
    tree = ast.parse(VERIFIER.read_text(encoding="utf-8"))
    imports = _all_imports(tree)
    forbidden = {
        "apar.evaluation.v5_staged_evidence",
        "apar.evaluation.v5_checkpoint_storage",
        "apar.evaluation.v5_controls",
        "apar.evaluation.v5_metrics",
        "apar.evaluation.v5_evaluation",
        "apar.evaluation.v5_runner",
        "apar.evaluation.v5_population",
        "apar.simulator",
        "apar.rails",
        "apar.trust",
    }
    assert imports.isdisjoint(forbidden)

def test_independent_verifier_reconstructs_every_safe_stage(tmp_path: Path) -> None:
    chain = _materialize_synthetic_complete_chain(tmp_path)
    report = verify_v5_kaggle_evidence(
        root=ROOT, checkpoint_roots=chain.checkpoints,
        final_root=chain.final_root,
        expected_mode="kaggle_capacity_validation",
    )
    assert report.valid is True
    assert report.verified_stage_ids == tuple(stage.value for stage in V5KaggleStage)
```

Add adversarial parametrization for: missing/extra/reordered/duplicated stages; missing/extra/reordered/altered chunks; manifest, predecessor, chain-root, attempt, run-mode, seed, profile, support-plan, source, tree, config, protocol, catalog, implementation, dependency, environment, notebook, and safe-core mutations; cross-attempt, cross-rehearsal, and cross-environment substitution; malformed record lengths and trailing bytes; feature/support reordering; forbidden feature fields; unequal arm support; missing/reordered controls; metric numerator/denominator/family/calibration/economic/bootstrap mutation; trust/ledger lineage mutation; observational deletion/reordering/sample/environment mutation; resource-gate violations; final payload, summary, readiness, and verifier-report mutation; safe↔locked relabeling; and legacy-summary substitution.

- [ ] **Step 2: Run RED**

```bash
.venv/bin/python -m pytest tests/evaluation/test_defense_v5_kaggle_verifier.py -q
```

Expected: missing independent verifier module.

- [ ] **Step 3: Implement independent schemas, readers, and recomputation**

Duplicate the frozen canonical decoding, digest, metric, ECE, economic, campaign-bootstrap, control-criterion, arm-contribution, and readiness formulas inside the independent module. Do not call or import production readers, runners, metrics, gates, controls, model, simulator, rail, ledger, or TrustVerifier code. Read chunk bytes with bounded standard-library I/O, enforce the exact schema and size limits before allocation, and recompute all record and chain digests from bytes.

The verifier must reconstruct command→event→payment→campaign lineage, double-entry conservation, TrustVerifier request/evidence/verdict bindings, approved feature/support order, four-arm row evidence, all seven control measurements, complete aggregate/per-family/calibration/economic metrics, campaign-level bootstrap samples and intervals, readiness gates, final deterministic core, observational latency/resource evidence, compact summary, and checkpoint chain root.

- [ ] **Step 4: Add deterministic fresh-process verification**

Run the same synthetic chain through the CLI with different hash seeds and compare canonical reports:

```bash
PYTHONHASHSEED=1 .venv/bin/python scripts/verify_defense_v5_kaggle_evidence.py \
  --root . --mode kaggle_capacity_validation \
  --chain-root /private/tmp/apar-v5-chain-a
PYTHONHASHSEED=777 .venv/bin/python -m scripts.verify_defense_v5_kaggle_evidence \
  --root . --mode kaggle_capacity_validation \
  --chain-root /private/tmp/apar-v5-chain-a
```

The test captures stdout and asserts byte-identical canonical JSON. Both CLIs return nonzero and print only a bounded error code for every mutation.

- [ ] **Step 5: Run GREEN and static boundary checks**

```bash
.venv/bin/python -m pytest tests/evaluation/test_defense_v5_kaggle_verifier.py -q
.venv/bin/ruff check src/apar/v5_kaggle_independent_verifier.py \
  scripts/verify_defense_v5_kaggle_evidence.py \
  tests/evaluation/test_defense_v5_kaggle_verifier.py
.venv/bin/mypy --strict src/apar/v5_kaggle_independent_verifier.py \
  scripts/verify_defense_v5_kaggle_evidence.py
```

### Task 11: Generate the nine private Kaggle notebooks deterministically

**Files:**

- Create: `scripts/build_defense_v5_kaggle_notebooks.py`
- Create: `tests/evaluation/test_defense_v5_kaggle_notebooks.py`
- Create: `kaggle/defense_v5/00_authorize.ipynb`
- Create: `kaggle/defense_v5/10_corpus.ipynb`
- Create: `kaggle/defense_v5/20_features.ipynb`
- Create: `kaggle/defense_v5/30_arms.ipynb`
- Create: `kaggle/defense_v5/40_label_shuffle.ipynb`
- Create: `kaggle/defense_v5/50_invariance_controls.ipynb`
- Create: `kaggle/defense_v5/60_single_class_controls.ipynb`
- Create: `kaggle/defense_v5/70_metrics.ipynb`
- Create: `kaggle/defense_v5/80_finalize.ipynb`
- Create: `kaggle/defense_v5/00_authorize-metadata.json` through `kaggle/defense_v5/80_finalize-metadata.json`

**Interfaces:**

```python
def build_v5_kaggle_notebooks(
    *, root: Path, output_dir: Path, owner_slug: str,
    source_dataset_slug: str, wheelhouse_dataset_slug: str,
    safe_evidence_dataset_slug: str,
) -> tuple[V5GeneratedNotebook, ...]: ...
```

- [ ] **Step 1: Resolve and validate the private Kaggle owner slug**

Use the authenticated Kaggle account page only to read the visible owner slug; never read, print, or persist API credentials. Freeze that literal slug and the three private dataset slugs in `defense-v5-kaggle-recovery.json`. Reject uppercase aliases, URLs, whitespace, and any owner or dataset identifier that differs from the frozen literals.

- [ ] **Step 2: Write notebook RED tests**

```python
def test_generated_notebooks_are_private_cpu_and_network_disabled(tmp_path: Path) -> None:
    generated = build_v5_kaggle_notebooks(
        root=ROOT, output_dir=tmp_path,
        owner_slug=FROZEN_OWNER,
        source_dataset_slug=FROZEN_SOURCE,
        wheelhouse_dataset_slug=FROZEN_WHEELS,
        safe_evidence_dataset_slug=FROZEN_SAFE,
    )
    assert tuple(item.stage for item in generated) == tuple(V5KaggleStage)
    for item in generated:
        metadata = json.loads(item.metadata_path.read_bytes())
        assert metadata["is_private"] is True
        assert metadata["enable_internet"] is False
        assert metadata["accelerator"] == "none"
```

Add tests that each notebook has only generated bootstrap/install/invoke cells; uses `pip --no-index --find-links`; invokes only `scripts/run_defense_v5_kaggle_stage.py`; has no seed/profile/stage/resume/retry/force/output override; binds the exact source, wheelhouse, safe-evidence, predecessor, and notebook digest; imports no Kaggle API/token packages; has no secret-shaped values; emits only the redacted receipt; names only its exact predecessor input; and contains no result interpretation code.

- [ ] **Step 3: Run RED**

```bash
.venv/bin/python -m pytest tests/evaluation/test_defense_v5_kaggle_notebooks.py -q
```

Expected: notebook generator is absent.

- [ ] **Step 4: Implement canonical notebook generation**

Serialize notebooks with sorted compact JSON and a fixed notebook format/cell-ID scheme. Generate the exact direct and module commands in separate tests; both must enter `run_defense_v5_kaggle_stage.py`. Stage 00 has no predecessor input and includes the authorization flag only in locked-successor metadata. Stages 10-80 attach exactly the prior saved notebook output. Every metadata document declares private visibility, CPU, internet disabled, and the frozen input slugs.

- [ ] **Step 5: Prove two independent generations are byte-identical**

```bash
.venv/bin/python scripts/build_defense_v5_kaggle_notebooks.py \
  --root . --output /private/tmp/apar-v5-notebooks-a
.venv/bin/python -m scripts.build_defense_v5_kaggle_notebooks \
  --root . --output /private/tmp/apar-v5-notebooks-b
diff -ru /private/tmp/apar-v5-notebooks-a /private/tmp/apar-v5-notebooks-b
```

Expected: no diff. Compare every generated path against the committed `kaggle/defense_v5` bytes.

- [ ] **Step 6: Run GREEN and static checks**

```bash
.venv/bin/python -m pytest tests/evaluation/test_defense_v5_kaggle_notebooks.py -q
.venv/bin/ruff check scripts/build_defense_v5_kaggle_notebooks.py \
  tests/evaluation/test_defense_v5_kaggle_notebooks.py
.venv/bin/mypy --strict scripts/build_defense_v5_kaggle_notebooks.py
```

### Task 12: Add the SOURCE3 bundle and closed pre-execution audit

**Files:**

- Create: `scripts/verify_defense_v5_kaggle_preexecution.py`
- Create: `tests/evaluation/test_defense_v5_kaggle_preexecution.py`
- Modify: `src/apar/evaluation/v5_evidence_protocol.py`
- Create after SOURCE3 commit: `/private/tmp/apar-v5-source3.tar.gz`
- Create after SOURCE3 commit: `/private/tmp/apar-v5-wheelhouse/`

**Interfaces:**

```python
def verify_v5_kaggle_preexecution(
    *, root: Path, expected_head: str,
    rehearsal_chain_roots: Sequence[Path],
) -> V5KagglePreexecutionReport: ...
```

- [ ] **Step 1: Write RED chronology, topology, and binding tests**

Test rejection of dirty worktree; wrong HEAD/tree/parent; a descendant HEAD; missing or altered RECOVERY paths; changed historical result, safe fixture, old receipt, abort record, config, protocol, catalog, implementation, notebook, source-bundle, wheelhouse, environment, or verifier digest; public or internet-enabled notebook metadata; wrong source file modes/path set; absent/invalid rehearsals; unequal deterministic rehearsal digests; weakened capacity measurements; existing successor receipt/checkpoint/final result/summary; symlink/hardlink/partial output targets; legacy monolithic output; seed/profile/mode mutation; and any preregistration commit that changes more than the one allowed path.

Add an execution-boundary spy asserting the audit only loads/asserts seed 2404 and never calls population generation, feature construction, training, prediction, controls, metrics, or finalization.

- [ ] **Step 2: Run RED**

```bash
.venv/bin/python -m pytest tests/evaluation/test_defense_v5_kaggle_preexecution.py -q
```

Expected: pre-execution verifier is absent.

- [ ] **Step 3: Implement canonical offline source and dependency binding tools**

Implement creation of a Git archive from a supplied clean commit with normalized archive prefix, mtime, owner/group, gzip header, ordering, and no working-tree files. Implement a Linux x86-64 CPython-3.12 wheelhouse manifest containing sorted wheel filenames, sizes, SHA-256 digests, package versions, and a manifest self-digest. Tests use fixture commits and fixture wheels; the real SOURCE3 archive is created only after Task 13 commits SOURCE3. The audit recomputes the Git tree/path/mode/content manifest and wheelhouse manifest; it never trusts uploaded metadata alone. Extend `V5_EVIDENCE_IMPLEMENTATION_PATHS` with only the new production files and generated notebook/metadata paths; do not change any existing semantic binding.

- [ ] **Step 4: Implement two audit phases without weakening either**

`--phase source` accepts only clean SOURCE3, no preregistration, no successor outputs, exact RECOVERY parent, and the generated source/wheel/notebook bindings. `--phase frozen` accepts only clean PREREGISTRATION3, exact sole-parent SOURCE3, exactly one changed preregistration path, two independently verified capacity chains, and all successor output paths absent. Neither phase creates a receipt or output.

The exact frozen command is:

```bash
.venv/bin/python scripts/verify_defense_v5_kaggle_preexecution.py \
  --root . \
  --phase frozen \
  --expected-head-from-preregistration \
  --rehearsal-a /private/tmp/apar-v5-rehearsal-a \
  --rehearsal-b /private/tmp/apar-v5-rehearsal-b
```

- [ ] **Step 5: Run GREEN and direct/module subprocess tests**

```bash
.venv/bin/python -m pytest tests/evaluation/test_defense_v5_kaggle_preexecution.py -q
.venv/bin/ruff check scripts/verify_defense_v5_kaggle_preexecution.py \
  tests/evaluation/test_defense_v5_kaggle_preexecution.py
.venv/bin/mypy --strict scripts/verify_defense_v5_kaggle_preexecution.py
```

### Task 13: Run local gates and create the single SOURCE3 commit

**Files:** all production, config, generated notebook, and test files from Tasks 2-12; no preregistration and no successor result.

- [ ] **Step 1: Run all focused staged-recovery tests**

```bash
.venv/bin/python -m pytest \
  tests/evaluation/test_defense_v5_kaggle_recovery_record.py \
  tests/evaluation/test_defense_v5_kaggle_protocol.py \
  tests/evaluation/test_defense_v5_checkpoint_storage.py \
  tests/evaluation/test_defense_v5_staged_evidence.py \
  tests/evaluation/test_defense_v5_staged_controls.py \
  tests/evaluation/test_defense_v5_kaggle_runner.py \
  tests/evaluation/test_defense_v5_kaggle_verifier.py \
  tests/evaluation/test_defense_v5_kaggle_notebooks.py \
  tests/evaluation/test_defense_v5_kaggle_preexecution.py -q
```

- [ ] **Step 2: Run v5 and execution-boundary regressions**

```bash
.venv/bin/python -m pytest tests/evaluation -k 'defense_v5 or sentinel' -q
.venv/bin/python -m pytest \
  tests/generators tests/simulator tests/rails tests/ledger tests/trust -q
```

If the repository uses different directory names, enumerate the existing matching test files with `rg --files tests | rg '(generator|simulator|rail|ledger|trust)'` and run the resulting explicit list. Do not omit an existing boundary suite because a guessed directory is absent.

- [ ] **Step 3: Run complete static and repository-integrity gates**

```bash
.venv/bin/ruff check src/apar/evaluation/v5_kaggle_protocol.py \
  src/apar/evaluation/v5_checkpoint_storage.py \
  src/apar/evaluation/v5_staged_evidence.py \
  src/apar/evaluation/v5_controls.py \
  src/apar/evaluation/v5_locked_evidence.py \
  src/apar/evaluation/v5_evidence_protocol.py \
  src/apar/v5_kaggle_independent_verifier.py \
  scripts/run_defense_v5_kaggle_stage.py \
  scripts/build_defense_v5_kaggle_notebooks.py \
  scripts/verify_defense_v5_kaggle_evidence.py \
  scripts/verify_defense_v5_kaggle_preexecution.py \
  tests/evaluation/test_defense_v5_kaggle_*.py \
  tests/evaluation/test_defense_v5_checkpoint_storage.py \
  tests/evaluation/test_defense_v5_staged_*.py
.venv/bin/mypy --strict \
  src/apar/evaluation/v5_kaggle_protocol.py \
  src/apar/evaluation/v5_checkpoint_storage.py \
  src/apar/evaluation/v5_staged_evidence.py \
  src/apar/evaluation/v5_controls.py \
  src/apar/evaluation/v5_locked_evidence.py \
  src/apar/evaluation/v5_evidence_protocol.py \
  src/apar/v5_kaggle_independent_verifier.py \
  scripts/run_defense_v5_kaggle_stage.py \
  scripts/build_defense_v5_kaggle_notebooks.py \
  scripts/verify_defense_v5_kaggle_evidence.py \
  scripts/verify_defense_v5_kaggle_preexecution.py
git diff --check
```

- [ ] **Step 4: Reconfirm locked bytes and seed boundary**

```bash
openssl dgst -sha256 docs/experiments/defense-v5-development-result.json
openssl dgst -sha256 docs/experiments/defense-v5-locked-development-attempt.json
test ! -e docs/experiments/defense-v5-kaggle-successor-attempt.json
test ! -e docs/experiments/defense-v5-kaggle-development-candidate.manifest.json
test ! -e docs/experiments/defense-v5-kaggle-development-candidate.manifest.json.chunks
test ! -e docs/experiments/defense-v5-kaggle-development-summary.json
```

Expected: historical result `af326f3a…d185`, failed receipt `c9093272…51e8`, no successor artifacts, and execution-boundary tests prove seed 2404 never reached population, training, scoring, controls, metrics, or finalization.

- [ ] **Step 5: Commit SOURCE3 under Dylan's identity**

Stage every Task 2-12 production/config/notebook/test path, verify that no preregistration or successor output is staged, then:

```bash
git -c user.name="Dylan Moraes" \
  -c user.email="dylanmoraesdljdd@gmail.com" \
  commit -m "feat: add staged sentinel recovery"
```

Verify SOURCE3 is the sole child of RECOVERY, has Dylan as author and committer, and contains the full implementation without `config/defense/defense-v5-kaggle-preregistration.json`.

- [ ] **Step 6: Build and audit the exact clean SOURCE3 inputs**

From clean SOURCE3, generate `/private/tmp/apar-v5-source3.tar.gz` from `HEAD`, materialize the frozen Linux x86-64 CPython-3.12 wheelhouse, and run:

```bash
.venv/bin/python scripts/verify_defense_v5_kaggle_preexecution.py \
  --root . \
  --phase source \
  --expected-head HEAD \
  --source-archive /private/tmp/apar-v5-source3.tar.gz \
  --wheelhouse /private/tmp/apar-v5-wheelhouse
```

Expected: canonical machine-readable PASS, exact RECOVERY→SOURCE3 chronology, clean tree, no preregistration, no successor output, and seed 2404 asserted only.

### Task 14: Create private Kaggle inputs and run two seed-404 capacity rehearsals

**Files:**

- Read local generated artifacts from `kaggle/defense_v5/`.
- Download rehearsal A into `/private/tmp/apar-v5-rehearsal-a/`.
- Download rehearsal B into `/private/tmp/apar-v5-rehearsal-b/`.
- Do not write browser credentials or Kaggle tokens to the repository.

- [ ] **Step 1: Use the browser skill and verify private topology before upload**

Read `browser:control-in-app-browser` completely before browser actions and announce that the skill is controlling the private Kaggle setup. If Kaggle requests login, MFA, CAPTCHA, or another security confirmation, pause for Dylan to complete it personally. Confirm every source dataset, wheelhouse dataset, safe-evidence dataset, notebook, and notebook output is private before uploading any evidence.

- [ ] **Step 2: Upload and independently hash the SOURCE3 inputs**

Create private immutable inputs for the exact SOURCE3 Git archive, Linux x86-64 CPython-3.12 wheelhouse, and approved safe evidence. After upload, download each version and prove its byte digest equals the locally frozen digest. Reject an upload whose bytes, visibility, version, or metadata differ.

- [ ] **Step 3: Create the nine private capacity notebooks**

Upload the generated notebook and metadata pairs with internet disabled and CPU selected. Attach only the exact frozen inputs and, from Stage 10 onward, the saved output of the immediate predecessor. Confirm notebook source digests in Kaggle match the committed bytes. Do not edit notebook cells in the browser.

- [ ] **Step 4: Run rehearsal A stage by stage**

For each stage from `00_authorize` to `80_finalize`, use private **Save & Run All**, wait for completion, inspect only the redacted receipt/resource fields, save the output version, and attach that exact output to the next notebook. If the current job dies without a visible manifest, rerun only that stage from its exact predecessor. If a manifest is visible but invalid, stop and reject the rehearsal.

After every stage, download its saved output and run:

```bash
.venv/bin/python scripts/verify_defense_v5_kaggle_evidence.py \
  --root . --mode kaggle_capacity_validation \
  --chain-root /private/tmp/apar-v5-rehearsal-a
```

The verifier must accept the completed prefix and, after Stage 80, the full final evidence.

- [ ] **Step 5: Run rehearsal B from fresh notebook output versions**

Repeat all nine stages using the same committed sources, dependency/environment inputs, notebook bytes, production profile/support, and seed 404, but a separate capacity-attempt binding and fresh saved outputs. Download into `/private/tmp/apar-v5-rehearsal-b` and independently verify every prefix and the final chain.

- [ ] **Step 6: Enforce the frozen capacity comparison**

Run the offline comparison and require:

```text
each stage peak_rss_bytes < 19327352832
each stage wall_seconds < 21600
each stage output_bytes < 10000000000
rehearsal A deterministic stage digests == rehearsal B deterministic stage digests
both final safe payloads independently valid
observational digests authenticated and allowed to differ
```

If any condition fails, stop. Do not weaken a threshold, reduce support, change stage partitioning, switch to GPU, or proceed to preregistration. A code/resource defect requires a new SOURCE commit and two new rehearsals.

### Task 15: Freeze PREREGISTRATION3 and stop before the successor authorization

**Files:**

- Create: `config/defense/defense-v5-kaggle-preregistration.json`
- No other file may change in PREREGISTRATION3.

- [ ] **Step 1: Generate the canonical preregistration from verified evidence**

Bind exact SOURCE3 commit/tree/parent, source path/mode/content manifest, verifier digest, closed run/stage protocol, production profile, seed 2404, support plan, catalog/config/protocol/implementation digests, old terminal attempt/abort records, approved safe core, private Kaggle source/wheel/safe/notebook/environment version digests, both seed-404 rehearsal chain roots, per-stage deterministic digests, observed capacity maxima, resource gates, checkpoint schema/chunk limits, exact stage commands, successor attempt path, final storage/output paths, private visibility, outcome-redaction rules, no-overwrite rules, incomplete-stage admission, malformed-stage terminal behavior, and required later authorization.

The preregistration must state that Stage 00 creates the successor authorization checkpoint and Stage 10 is the first function allowed to pass seed 2404 to population generation.

- [ ] **Step 2: Validate path-only chronology before commit**

```bash
git diff --check
git diff --name-only
```

Expected: only `config/defense/defense-v5-kaggle-preregistration.json` differs from clean SOURCE3.

- [ ] **Step 3: Commit PREREGISTRATION3 under Dylan's identity**

```bash
git add config/defense/defense-v5-kaggle-preregistration.json
git -c user.name="Dylan Moraes" \
  -c user.email="dylanmoraesdljdd@gmail.com" \
  commit -m "chore: freeze staged sentinel execution"
```

Verify PREREGISTRATION3 is the sole child of SOURCE3, its diff contains exactly the one preregistration path, and both author and committer are Dylan.

- [ ] **Step 4: Run the exact clean frozen pre-execution audit**

```bash
.venv/bin/python scripts/verify_defense_v5_kaggle_preexecution.py \
  --root . \
  --phase frozen \
  --expected-head-from-preregistration \
  --rehearsal-a /private/tmp/apar-v5-rehearsal-a \
  --rehearsal-b /private/tmp/apar-v5-rehearsal-b
git status --short
```

Expected: canonical machine-readable PASS; exact two-commit chronology; both rehearsals valid and deterministic-core equal; worktree clean; historical bytes unchanged; successor receipt/checkpoints/final result absent; seed 2404 asserted only.

- [ ] **Step 5: Stop and request a new explicit authorization**

Show, but do not execute, the exact private Stage-00 Kaggle **Save & Run All** action and its bound command. Report SOURCE3 and PREREGISTRATION3 hashes, rehearsal chain roots, per-stage capacity maxima, verifier results, locked-file hashes, and absence of successor artifacts. A later explicit authorization permits Stage 00 only; Stage 10 still requires the saved Stage-00 output to verify successfully.

## Plan Completion Review

- [ ] Every requirement in the approved design maps to at least one RED test, production step, adversarial mutation, verification command, or frozen preregistration field above.
- [ ] Search the resulting implementation for unfinished markers, skipped tests, expected failures, empty production branches, and permissive defaults; none may remain.
- [ ] Confirm all public and serialized interfaces use exact frozen models and consistent types across producer, storage, runner, notebook, verifier, and preregistration boundaries.
- [ ] Confirm the browser rehearsal phase never runs seed 2404 and the final step stops before Stage 00.
- [ ] Confirm the implementation report distinguishes deterministic checkpoint equality from authenticated observational variation and does not claim byte-reproducible complete artifacts.
