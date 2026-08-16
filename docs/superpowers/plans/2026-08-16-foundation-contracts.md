# Foundation and Contracts Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a reproducible Python application foundation with typed payment, scenario, decision, registry, artifact, and API contracts.

**Architecture:** A Python package named `apar` owns domain contracts and local services. Pydantic models are the only data exchanged across subsystem boundaries, SQLite stores mutable metadata, and content-addressed files store immutable run artifacts.

**Tech Stack:** Python 3.12, FastAPI, Pydantic v2, SQLite, PyArrow, pytest, Hypothesis, Ruff, mypy

**Spec:** `SOLUTION_SPEC.md`, `docs/01-product-requirements.md`, `docs/02-system-architecture.md`, `docs/03-threat-registry.md`, `docs/06-data-and-api-contracts.md`

## Global Constraints

- Use Python 3.12 and require `python_requires = ">=3.12,<3.13"`.
- Use UUID strings for external IDs and timezone-aware UTC datetimes for event, arrival, and decision times.
- Reject any feature source with `source_timestamp >= decision_timestamp`.
- Keep mutable metadata in `.apar/state.db` and immutable artifacts in `.apar/artifacts/<sha256>/`.
- Serialize canonical JSON with sorted keys, UTF-8, and compact separators before hashing.
- Keep `validation_spike/` unchanged and outside the import path.
- Expose only localhost by default and make network-dependent integrations optional.
- Every task must pass Ruff, mypy, and its focused pytest selection before commit.

---

## Target file map

```text
pyproject.toml                         Python dependencies and quality commands
src/apar/__init__.py                  Package version
src/apar/config.py                    Local paths and runtime settings
src/apar/contracts/events.py          Payment and control event contracts
src/apar/contracts/scenarios.py       Scenario and attacker contracts
src/apar/contracts/decisions.py       Score, action, and reason contracts
src/apar/contracts/reports.py         Evaluation and promotion contracts
src/apar/registry/models.py           Threat-card and evidence contracts
src/apar/registry/repository.py       SQLite threat repository
src/apar/compiler/compiler.py         Threat-card to scenario compilation
src/apar/compiler/errors.py           Stable compiler error codes
src/apar/storage/database.py          SQLite schema and migrations
src/apar/storage/artifacts.py         Content-addressed artifact store
src/apar/api/app.py                   FastAPI application factory
src/apar/api/routes/health.py         Health and version endpoint
src/apar/api/routes/registry.py       Threat-card endpoints
tests/contracts/                      Contract and invariant tests
tests/factories.py                    Fully populated contract constructors for tests
tests/registry/                       Registry and compiler tests
tests/storage/                        Artifact-store tests
tests/api/                            API boundary tests
```

### Task 1: Bootstrap the Python package and quality gates

**Files:**
- Create: `pyproject.toml`
- Create: `src/apar/__init__.py`
- Create: `src/apar/config.py`
- Create: `tests/test_package.py`

**Interfaces:**
- Produces: `apar.__version__: str`
- Produces: `Settings.from_root(root: Path) -> Settings`

- [ ] **Step 1: Write the package import and path test**

```python
from pathlib import Path

from apar import __version__
from apar.config import Settings


def test_settings_are_root_relative(tmp_path: Path) -> None:
    settings = Settings.from_root(tmp_path)
    assert __version__ == "0.1.0"
    assert settings.database_path == tmp_path / ".apar" / "state.db"
    assert settings.artifact_root == tmp_path / ".apar" / "artifacts"
```

- [ ] **Step 2: Run the focused test and confirm the missing-package failure**

Run: `python -m pytest tests/test_package.py -q`

Expected: collection fails with `ModuleNotFoundError: No module named 'apar'`.

- [ ] **Step 3: Add the package metadata and settings implementation**

```python
# src/apar/config.py
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    root: Path
    database_path: Path
    artifact_root: Path

    @classmethod
    def from_root(cls, root: Path) -> "Settings":
        resolved = root.resolve()
        state_root = resolved / ".apar"
        return cls(resolved, state_root / "state.db", state_root / "artifacts")
```

Set `src/apar/__init__.py` to `__version__ = "0.1.0"`. Configure the build backend, runtime dependencies, `dev` optional dependencies, pytest paths, Ruff rules, and mypy strict mode in `pyproject.toml`.

- [ ] **Step 4: Run the foundation quality gate**

Run: `python -m pytest tests/test_package.py -q`

Expected: `1 passed`.

Run: `python -m ruff check src tests && python -m mypy src`

Expected: both commands exit zero.

- [ ] **Step 5: Commit the independently runnable package**

```bash
git add pyproject.toml src/apar/__init__.py src/apar/config.py tests/test_package.py
git commit -m "build: bootstrap APAR Python package"
```

### Task 2: Define event, scenario, decision, and report contracts

**Files:**
- Create: `src/apar/contracts/__init__.py`
- Create: `src/apar/contracts/events.py`
- Create: `src/apar/contracts/scenarios.py`
- Create: `src/apar/contracts/decisions.py`
- Create: `src/apar/contracts/reports.py`
- Create: `tests/factories.py`
- Create: `tests/contracts/test_models.py`

**Interfaces:**
- Produces: `PaymentEvent`, `EventKind`, `Rail`, `LifecycleState`
- Produces: `ScenarioBundle`, `AttackerMode`, `FeedbackField`
- Produces: `Decision`, `Action`, `ReasonCode`
- Produces: `EvaluationReport`, `PromotionDecision`

- [ ] **Step 1: Write strict contract tests**

```python
from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest
from pydantic import ValidationError

from apar.contracts.decisions import Action, Decision
from apar.contracts.events import EventKind, PaymentEvent, Rail


def test_event_rejects_ingestion_before_event_time() -> None:
    now = datetime(2026, 8, 16, tzinfo=UTC)
    with pytest.raises(ValidationError, match="ingested_at"):
        PaymentEvent(
            schema_version="1.0.0", event_id="e1", campaign_id="c1",
            trace_id="tr1", rail=Rail.CARD, viewpoint="network_native",
            event_type=EventKind.AUTHORIZATION, amount=Decimal("10.00"), currency="USD",
            event_time=now, ingested_at=now - timedelta(seconds=1), available_at=now,
            actor_id="a1", counterparty_id="m1",
        )


def test_decision_rejects_future_source() -> None:
    now = datetime(2026, 8, 16, tzinfo=UTC)
    with pytest.raises(ValidationError, match="strictly before"):
        Decision(
            decision_id="d1", event_id="e1", decision_time=now,
            max_source_timestamp=now, score=0.7, action=Action.CHALLENGE,
            reason_codes=["VELOCITY_1M"], model_version="rules-v1",
        )
```

- [ ] **Step 2: Verify both invalid cases fail before implementation**

Run: `python -m pytest tests/contracts/test_models.py -q`

Expected: collection fails because the contract modules do not exist.

- [ ] **Step 3: Implement exact enums and Pydantic models**

```python
# src/apar/contracts/events.py
from datetime import datetime
from decimal import Decimal
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Rail(StrEnum):
    CARD = "card"
    A2A = "a2a"
    AGENTIC = "agentic"


class EventKind(StrEnum):
    AUTHORIZATION = "authorization"
    CLEARING = "clearing"
    SETTLEMENT = "settlement"
    REVERSAL = "reversal"
    TRANSFER_INITIATED = "transfer_initiated"
    TRANSFER_POSTED = "transfer_posted"
    REFUND = "refund"
    FRAUD_REPORTED = "fraud_reported"
    DISPUTE_OPENED = "dispute_opened"
    CHARGEBACK = "chargeback"
    RECOVERY = "recovery"


class PaymentEvent(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    schema_version: str
    event_id: str
    campaign_id: str
    trace_id: str
    rail: Rail
    viewpoint: str
    event_type: EventKind
    amount: Decimal
    currency: str
    event_time: datetime
    ingested_at: datetime
    available_at: datetime
    decision_at: datetime | None = None
    actor_id: str
    counterparty_id: str
    party_refs: dict[str, str] = Field(default_factory=dict)
    rail_data: dict[str, str | int | float | bool] = Field(default_factory=dict)
    lineage: dict[str, str | bool] = Field(default_factory=dict)
    privacy: dict[str, str] = Field(default_factory=dict)
    extensions: dict[str, object] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_times_and_amount(self) -> "PaymentEvent":
        if self.ingested_at < self.event_time:
            raise ValueError("ingested_at must be at or after event_time")
        if self.available_at < self.ingested_at:
            raise ValueError("available_at must be at or after ingested_at")
        if self.decision_at is not None and self.decision_at < self.available_at:
            raise ValueError("decision_at must be at or after available_at")
        if self.amount < 0:
            raise ValueError("amount must be non-negative")
        return self
```

Implement the remaining models with `extra="forbid"`, frozen instances, semantic schema versions, UTC-aware timestamp validators, score bounds from 0 to 1, a non-empty reason-code list for non-approve actions, and `max_source_timestamp < decision_time`. Unknown optional data is retained only inside the explicit `extensions` object; an unknown major schema version is rejected. Define test constructors in `tests/factories.py`: `make_payment_event()`, `make_threat_card()`, `make_scenario_config()`, `make_decision()`, and `make_evaluation_report()`. Each constructor returns a fully valid object and accepts keyword overrides through `model_copy(update=overrides)`.

- [ ] **Step 4: Run contract tests and schema snapshots**

Run: `python -m pytest tests/contracts/test_models.py -q`

Expected: all contract tests pass.

Add `tests/contracts/test_schema_snapshots.py` asserting that `model_json_schema()` contains required keys for every external model, then run `python -m pytest tests/contracts -q`.

- [ ] **Step 5: Commit stable public contracts**

```bash
git add src/apar/contracts tests/contracts
git commit -m "feat: define payment assurance contracts"
```

### Task 3: Build the threat registry and scenario compiler

**Files:**
- Create: `src/apar/storage/database.py`
- Create: `src/apar/registry/__init__.py`
- Create: `src/apar/registry/models.py`
- Create: `src/apar/registry/repository.py`
- Create: `src/apar/compiler/__init__.py`
- Create: `src/apar/compiler/errors.py`
- Create: `src/apar/compiler/compiler.py`
- Create: `tests/registry/test_repository.py`
- Create: `tests/registry/test_compiler.py`

**Interfaces:**
- Consumes: `ScenarioBundle`, `Rail`, and `AttackerMode` from Task 2
- Produces: `ThreatCard`, `EvidenceRecord`, `ThreatRepository`
- Produces: `compile_scenario(card: ThreatCard, config: ScenarioConfig) -> ScenarioBundle`
- Produces stable errors: `MISSING_EVIDENCE`, `UNSUPPORTED_RAIL`, `UNSAFE_EXPORT`, `INVALID_FEEDBACK`

- [ ] **Step 1: Write persistence and rejection tests**

```python
from pathlib import Path

import pytest

from apar.compiler.compiler import compile_scenario
from apar.compiler.errors import CompilerError
from apar.registry.repository import ThreatRepository
from tests.factories import make_threat_card


def test_repository_round_trip(tmp_path: Path) -> None:
    threat_card = make_threat_card()
    repo = ThreatRepository(tmp_path / "state.db")
    repo.upsert(threat_card)
    assert repo.get(threat_card.threat_id) == threat_card


def test_compiler_rejects_missing_source() -> None:
    threat_card = make_threat_card()
    invalid = threat_card.model_copy(update={"evidence": []})
    with pytest.raises(CompilerError) as error:
        compile_scenario(invalid, invalid.default_config)
    assert error.value.code == "MISSING_EVIDENCE"
```

- [ ] **Step 2: Confirm registry and compiler tests fail**

Run: `python -m pytest tests/registry -q`

Expected: collection fails because registry and compiler modules are absent.

- [ ] **Step 3: Implement schema migration, repository, and compiler checks**

Use a single migration table with version `1`. Store the full validated threat card as canonical JSON and index `threat_id`, `family`, `confidence`, and `implementation_status`. The compiler must verify at least one direct source URL, allowed rail membership, non-empty observables, an explicit safety class, supported feedback fields, positive population size, positive duration, and a fixed random seed.

```python
class CompilerError(ValueError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
```

- [ ] **Step 4: Run registry tests and an explainable rejection matrix**

Run: `python -m pytest tests/registry -q`

Expected: repository round-trip passes and every invalid fixture maps to one stable compiler error code.

- [ ] **Step 5: Commit the evidence-to-scenario boundary**

```bash
git add src/apar/storage/database.py src/apar/registry src/apar/compiler tests/registry
git commit -m "feat: add threat registry and scenario compiler"
```

### Task 4: Implement the content-addressed artifact store

**Files:**
- Create: `src/apar/storage/artifacts.py`
- Create: `tests/storage/test_artifacts.py`

**Interfaces:**
- Produces: `ArtifactRef(sha256: str, media_type: str, size_bytes: int, relative_path: str)`
- Produces: `ArtifactStore.put_bytes(payload: bytes, media_type: str) -> ArtifactRef`
- Produces: `ArtifactStore.put_json(payload: BaseModel | dict[str, object]) -> ArtifactRef`
- Produces: `ArtifactStore.read(ref: ArtifactRef) -> bytes`

- [ ] **Step 1: Write immutability and canonicalization tests**

```python
from apar.storage.artifacts import ArtifactStore


def test_json_hash_is_key_order_independent(tmp_path) -> None:
    store = ArtifactStore(tmp_path)
    first = store.put_json({"b": 2, "a": 1})
    second = store.put_json({"a": 1, "b": 2})
    assert first.sha256 == second.sha256
    assert first.relative_path == second.relative_path


def test_existing_digest_cannot_be_overwritten(tmp_path) -> None:
    store = ArtifactStore(tmp_path)
    ref = store.put_bytes(b"evidence", "application/octet-stream")
    assert store.read(ref) == b"evidence"
    assert len(list((tmp_path / ref.sha256).iterdir())) == 2
```

- [ ] **Step 2: Verify the missing store fails**

Run: `python -m pytest tests/storage/test_artifacts.py -q`

Expected: collection fails with missing `apar.storage.artifacts`.

- [ ] **Step 3: Implement atomic writes and manifests**

Canonical JSON is `json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")`. Write payload and `manifest.json` to a temporary directory beside the digest directory, call `fsync`, then rename once. If the digest directory exists, verify byte equality and return its manifest without writing.

- [ ] **Step 4: Run storage and interrupted-write tests**

Run: `python -m pytest tests/storage -q`

Expected: canonicalization, duplicate writes, manifest validation, and simulated interrupted-write cleanup all pass.

- [ ] **Step 5: Commit immutable run storage**

```bash
git add src/apar/storage/artifacts.py tests/storage
git commit -m "feat: add content-addressed artifact store"
```

### Task 5: Expose health and threat-registry API boundaries

**Files:**
- Create: `src/apar/api/__init__.py`
- Create: `src/apar/api/app.py`
- Create: `src/apar/api/dependencies.py`
- Create: `src/apar/api/routes/__init__.py`
- Create: `src/apar/api/routes/health.py`
- Create: `src/apar/api/routes/registry.py`
- Create: `tests/api/test_health.py`
- Create: `tests/api/test_registry.py`

**Interfaces:**
- Consumes: `Settings`, `ThreatRepository`, and `ThreatCard`
- Produces: `create_app(settings: Settings) -> FastAPI`
- Produces: `GET /api/v1/health`, `GET /api/v1/threats`, `GET /api/v1/threats/{threat_id}`, `PUT /api/v1/threats/{threat_id}`

- [ ] **Step 1: Write API status and conflict tests**

```python
from fastapi.testclient import TestClient

from apar.api.app import create_app
from apar.config import Settings
from tests.factories import make_threat_card


def test_health_is_versioned(tmp_path) -> None:
    client = TestClient(create_app(Settings.from_root(tmp_path)))
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "version": "0.1.0"}


def test_path_and_payload_threat_ids_must_match(tmp_path) -> None:
    client = TestClient(create_app(Settings.from_root(tmp_path)))
    threat_payload = make_threat_card().model_dump(mode="json")
    response = client.put("/api/v1/threats/other-id", json=threat_payload)
    assert response.status_code == 409
    assert response.json()["detail"]["code"] == "THREAT_ID_MISMATCH"
```

- [ ] **Step 2: Confirm route tests fail before application creation**

Run: `python -m pytest tests/api -q`

Expected: collection fails because `apar.api.app` is missing.

- [ ] **Step 3: Implement the application factory and typed error envelope**

Bind only through the CLI in the later execution plan; `create_app` must not start a server. Store settings in `app.state`, initialize the database during lifespan, and return errors as `{"detail": {"code": str, "message": str}}`. Reject unknown JSON keys through the Pydantic contract.

- [ ] **Step 4: Run API tests and OpenAPI assertions**

Run: `python -m pytest tests/api -q`

Expected: all endpoint and 404/409/422 behavior tests pass.

Add an assertion that `/openapi.json` contains only `/api/v1/health` and the three registry paths at this stage.

- [ ] **Step 5: Commit the local API boundary**

```bash
git add src/apar/api tests/api
git commit -m "feat: expose health and threat registry API"
```

### Task 6: Add the 20-card threat portfolio, golden fixture, and G0 verification command

**Files:**
- Create: `fixtures/golden/threat-card.json`
- Create: `fixtures/golden/scenario-config.json`
- Create: `fixtures/threats/` with one JSON file per approved threat ID
- Create: `tests/integration/test_g0_contract_flow.py`
- Create: `tests/registry/test_portfolio.py`
- Create: `scripts/verify_g0.py`
- Modify: `README.md`

**Interfaces:**
- Consumes: registry, compiler, contracts, and artifact store from Tasks 2 through 5
- Produces: `python scripts/verify_g0.py` with exit code 0 only when the golden contract flow succeeds

- [ ] **Step 1: Write the end-to-end contract test**

```python
from pathlib import Path

from apar.compiler.compiler import compile_scenario
from apar.contracts.scenarios import ScenarioConfig
from apar.registry.models import ThreatCard
from apar.registry.repository import ThreatRepository
from apar.storage.artifacts import ArtifactStore


def test_threat_compiles_and_freezes(tmp_path) -> None:
    golden_threat = ThreatCard.model_validate_json(Path("fixtures/golden/threat-card.json").read_text())
    golden_config = ScenarioConfig.model_validate_json(Path("fixtures/golden/scenario-config.json").read_text())
    repo = ThreatRepository(tmp_path / "state.db")
    store = ArtifactStore(tmp_path / "artifacts")
    repo.upsert(golden_threat)
    bundle = compile_scenario(repo.get(golden_threat.threat_id), golden_config)
    ref = store.put_json(bundle)
    assert ref.sha256 == store.put_json(bundle).sha256
    assert bundle.seed == golden_config.seed
```

- [ ] **Step 2: Run the integration test before adding fixtures**

Run: `python -m pytest tests/integration/test_g0_contract_flow.py -q`

Expected: setup fails because the golden fixture files are absent.

- [ ] **Step 3: Add the threat portfolio and deterministic golden configuration**

Create at least these 20 evidence-backed IDs: `app-personalized-mule`, `voice-clone-app`, `remote-access-guidance`, `investment-persuasion`, `invoice-payee-substitution`, `adaptive-card-testing`, `cnp-checkout-automation`, `credential-stuffing-ato`, `synthetic-identity-bustout`, `first-party-dispute-automation`, `merchant-laundering`, `synthetic-merchant-refund`, `promotion-abuse`, `mule-recruitment`, `mule-fanout-layering`, `instant-payment-velocity`, `qr-social-engineering`, `agentic-payee-substitution`, `agentic-mandate-escalation`, and `agentic-cart-tampering`. Every card must include at least one authoritative or primary direct source URL, publication and access dates, source type, confidence, fact statements, project inferences, capability delta, affected rail, observables, simulation status, and safety class. Mark exactly four as `deep_scenario`: APP scam and mule, card testing/CNP, synthetic merchant/refund, and agentic intent abuse.

The golden APP-scam card uses `family="app_scam_mule"`, rails `a2a` and `agentic`, capability deltas `personalization` and `iteration_speed`, non-operational attack descriptions, observables, and safety class `synthetic_only`. Its scenario uses seed `260816`, a 24-hour simulated duration, 5,000 benign entities, 60 illicit entities, and decision feedback limited to `action` and `reason_family`.

- [ ] **Step 4: Run G0 verification from a clean local state**

Run: `python scripts/verify_g0.py`

Expected output ends with `G0 PASS: 20 threat cards, contracts, registry, compiler, API, and artifact store` and exit code 0.

Run: `python -m pytest -q && python -m ruff check src tests scripts && python -m mypy src scripts`

Expected: every command exits zero.

- [ ] **Step 5: Commit the G0 deliverable**

```bash
git add fixtures tests/integration tests/registry/test_portfolio.py scripts/verify_g0.py README.md
git commit -m "test: establish G0 golden contract flow"
```

## Plan completion gate

G0 is complete when a fresh environment can install the package, execute `python scripts/verify_g0.py`, load at least 20 evidence-backed threat cards, compile the golden threat into a frozen scenario artifact, and pass all contract, registry, storage, and API tests without reading or changing `validation_spike/`.
