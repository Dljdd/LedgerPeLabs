"""Reproducible, synthetic-only orchestration for the Defend evidence commands.

The public commands accept only the immutable competition profile.  A deliberately
small fixture profile is available only through :func:`run_g3_fixture`; it cannot be
selected from any export command.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import os
import secrets
import stat
import sys
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Literal, Never, cast

import numpy as np
from pydantic import Field, ValidationError, field_validator, model_validator

from apar.compiler import compile_scenario
from apar.contracts._validation import ExternalContract, validate_utc_timestamp
from apar.contracts.decisions import Action
from apar.contracts.scenarios import ScenarioBundle, ScenarioConfig
from apar.defense.bundle import (
    GENESIS_ROLLBACK_REF,
    BundleLineage,
    DefenderBundlePublisher,
    build_source_inventory,
    current_environment_lock,
)
from apar.defense.calibration import select_calibrator
from apar.defense.contracts import ObservedEvent
from apar.defense.gbdt import GbdtTrainingConfig, RollingFold, train_gbdt
from apar.defense.policy import OperatingBudget
from apar.defense.rules import RuleEngine, RuleManifest
from apar.defense.thresholds import select_policy_thresholds
from apar.evaluation.contracts import (
    CorpusManifest,
    CorpusProfile,
    EvaluationTruthRow,
    Family,
    FrozenCorpus,
)
from apar.evaluation.corpus import assemble_verified_corpus
from apar.evaluation.gates import (
    DefenseArm,
    EvaluatorReplayVerifier,
    EvaluatorSigningIdentity,
)
from apar.evaluation.regimes import frozen_corpus_digest
from apar.evaluation.replay import bind_replay_case_counter
from apar.evaluation.reporting import (
    DefenseScorecard,
    PublicArtifactVerifier,
    ReportingContractError,
    load_evaluation_bundle,
)
from apar.evaluation.splits import EvaluationSplit, SplitConfig, make_evaluation_split
from apar.features.builders import FeatureMatrix, build_feature_matrix
from apar.features.catalog import FeatureCatalog, load_feature_catalog
from apar.features.state import FeatureVector
from apar.registry.models import ThreatCard
from apar.runs import (
    AttackerPolicy,
    AttackerPolicyKind,
    RunManifest,
    RunRunner,
    RunSigningIdentity,
    bind_scenario_for_run,
)
from apar.runs.wire import WireContractError, canonical_json_bytes, strict_json_loads
from apar.storage.artifacts import ArtifactRef, ArtifactStore

_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
_COMMITTED_PROFILE = _REPOSITORY_ROOT / "config" / "defense" / "competition-profile.json"
_CATALOG = _REPOSITORY_ROOT / "config" / "defense" / "feature-catalog.json"
_CORPUS_ENVELOPE_MEDIA_TYPE = "application/vnd.apar.corpus-envelope+json"
_DEFENDER_ENSEMBLE_MEDIA_TYPE = "application/vnd.apar.defender-ensemble+json"
_HEX = frozenset("0123456789abcdef")
_FAMILIES: tuple[Family, ...] = (
    "agentic_intent_abuse",
    "app_scam_mule",
    "card_testing_cnp",
    "synthetic_merchant_refund",
)
_THREAT_CARD_BY_FAMILY: dict[Family, str] = {
    "agentic_intent_abuse": "agentic-payee-substitution.json",
    "app_scam_mule": "app-personalized-mule.json",
    "card_testing_cnp": "adaptive-card-testing.json",
    "synthetic_merchant_refund": "synthetic-merchant-refund.json",
}
_RAIL_BY_FAMILY = {
    "agentic_intent_abuse": "agentic",
    "app_scam_mule": "a2a",
    "card_testing_cnp": "card",
    "synthetic_merchant_refund": "card",
}
_THREAT_REF_BY_FAMILY = {
    family: f"{filename.removesuffix('.json')}@1"
    for family, filename in _THREAT_CARD_BY_FAMILY.items()
}
_PARTITIONS = ("train", "calibrator_fit", "threshold_selection", "development_test")
_REGIMES = (
    "prevalence_dilution",
    "missing_optional",
    "availability_delay",
    "compressed_bursts",
    "benign_amount_shift",
    "cold_id_remap",
)
_FIXTURE_SIGNER_SEED = hashlib.sha256(b"apar-g3-fixture-signer-v1").digest()
_MAX_JSON_BYTES = 64 * 1024 * 1024


class CliContractError(ValueError):
    """A CLI input or publication would violate the frozen evidence contract."""


class GbdtProfile(ExternalContract):
    depths: tuple[int, ...]
    learning_rates: tuple[float, ...]
    l2_leaf_regs: tuple[float, ...]
    iterations: int = Field(ge=1)


class CalibrationProfile(ExternalContract):
    candidates: tuple[Literal["sigmoid", "isotonic"], ...]
    minimum_class_count: int = Field(ge=1)
    ece_bins: int = Field(ge=2)


class BudgetProfile(ExternalContract):
    challenge_rate_max: float = Field(ge=0.0, le=1.0)
    false_decline_rate_max: float = Field(ge=0.0, le=1.0)
    review_case_rate_max: float = Field(ge=0.0, le=1.0)


class GateProfile(ExternalContract):
    minimum_family_recall: float = Field(ge=0.0, le=1.0)
    maximum_ece: float = Field(ge=0.0, le=1.0)
    maximum_p95_latency_ms: float = Field(gt=0.0)
    maximum_slice_recall_regression: float = Field(ge=0.0, le=1.0)


class CompetitionProfile(ExternalContract):
    """Exact preregistered competition values or an internal reduced fixture."""

    schema_version: Literal["1.0.0"] = "1.0.0"
    families: tuple[Family, ...]
    campaigns_per_family: int = Field(ge=1, le=50)
    seed_bases: dict[Family, int]
    simulation_start_utc: datetime
    campaign_spacing_days: int = Field(ge=1)
    partition_campaign_indices: dict[str, tuple[int, int]]
    label_delay_days: int = Field(ge=0)
    gbdt: GbdtProfile
    calibration: CalibrationProfile
    model_seed: int = Field(ge=0)
    bootstrap_seed: int = Field(ge=0)
    bootstrap_replicates: int = Field(ge=1)
    budgets: BudgetProfile
    gates: GateProfile
    regimes: tuple[str, ...]

    @field_validator("simulation_start_utc")
    @classmethod
    def start_is_utc(cls, value: datetime) -> datetime:
        return validate_utc_timestamp(value)

    @model_validator(mode="after")
    def profile_is_closed(self) -> CompetitionProfile:
        if self.families != _FAMILIES:
            raise ValueError("profile families must be complete and ordered")
        if set(self.seed_bases) != set(_FAMILIES):
            raise ValueError("profile seed bases must cover every family exactly")
        if set(self.partition_campaign_indices) != set(_PARTITIONS):
            raise ValueError("profile partitions must be complete and ordered")
        for bounds in self.partition_campaign_indices.values():
            if (
                type(bounds) is not tuple
                or len(bounds) != 2
                or any(type(item) is not int for item in bounds)
                or bounds[0] < 0
                or bounds[0] > bounds[1]
            ):
                raise ValueError("profile partition bounds must be inclusive integer pairs")
        if self.regimes != _REGIMES:
            raise ValueError("profile regimes must be complete and ordered")
        if self.calibration.candidates != ("sigmoid", "isotonic"):
            raise ValueError("profile calibration candidates must be exact and ordered")
        return self

    @property
    def campaign_count(self) -> int:
        return len(self.families) * self.campaigns_per_family

    @property
    def fixture_only(self) -> bool:
        return self.to_json() != _competition_profile_bytes()

    def campaign_seed(self, family: Family, index: int) -> int:
        if family not in self.families or type(index) is not int:
            raise CliContractError("campaign family or index is invalid")
        if not 0 <= index < self.campaigns_per_family:
            raise CliContractError("campaign index is outside the profile")
        return self.seed_bases[family] + index

    def campaign_start(self, family: Family, index: int) -> datetime:
        del family
        if type(index) is not int or not 0 <= index < self.campaigns_per_family:
            raise CliContractError("campaign index is outside the profile")
        return self.simulation_start_utc + timedelta(
            days=index * self.campaign_spacing_days
        )

    def to_json(self) -> bytes:
        return canonical_json_bytes(self.model_dump(mode="json"))

    @classmethod
    def fixture(cls) -> CompetitionProfile:
        document = _competition_profile_document()
        document["campaigns_per_family"] = 1
        document["partition_campaign_indices"] = {
            name: [0, 0] for name in _PARTITIONS
        }
        cast(dict[str, object], document["gbdt"])["depths"] = [2]
        cast(dict[str, object], document["gbdt"])["learning_rates"] = [0.1]
        cast(dict[str, object], document["gbdt"])["l2_leaf_regs"] = [3.0]
        cast(dict[str, object], document["gbdt"])["iterations"] = 8
        cast(dict[str, object], document["calibration"])["minimum_class_count"] = 2
        document["bootstrap_replicates"] = 16
        document["budgets"] = {
            "challenge_rate_max": 1.0,
            "false_decline_rate_max": 1.0,
            "review_case_rate_max": 1.0,
        }
        return cls.model_validate(document)


def _competition_profile_document() -> dict[str, object]:
    return {
        "schema_version": "1.0.0",
        "families": list(_FAMILIES),
        "campaigns_per_family": 50,
        "seed_bases": {
            "agentic_intent_abuse": 260000,
            "app_scam_mule": 261000,
            "card_testing_cnp": 262000,
            "synthetic_merchant_refund": 263000,
        },
        "simulation_start_utc": "2026-01-01T00:00:00Z",
        "campaign_spacing_days": 8,
        "partition_campaign_indices": {
            "train": [0, 24],
            "calibrator_fit": [25, 31],
            "threshold_selection": [32, 37],
            "development_test": [38, 49],
        },
        "label_delay_days": 7,
        "gbdt": {
            "depths": [4, 6],
            "learning_rates": [0.03, 0.08],
            "l2_leaf_regs": [3.0, 8.0],
            "iterations": 300,
        },
        "calibration": {
            "candidates": ["sigmoid", "isotonic"],
            "minimum_class_count": 50,
            "ece_bins": 10,
        },
        "model_seed": 260816,
        "bootstrap_seed": 260816,
        "bootstrap_replicates": 1000,
        "budgets": {
            "challenge_rate_max": 0.02,
            "false_decline_rate_max": 0.001,
            "review_case_rate_max": 0.01,
        },
        "gates": {
            "minimum_family_recall": 0.5,
            "maximum_ece": 0.1,
            "maximum_p95_latency_ms": 50.0,
            "maximum_slice_recall_regression": 0.05,
        },
        "regimes": list(_REGIMES),
    }


def _competition_profile_bytes() -> bytes:
    return canonical_json_bytes(_competition_profile_document())


def load_competition_profile(path: Path, *, competition: bool) -> CompetitionProfile:
    checked = _regular_file(path, label="competition profile", max_bytes=128 * 1024)
    payload = checked[:-1] if checked.endswith(b"\n") and not checked.endswith(b"\n\n") else checked
    try:
        document = strict_json_loads(payload)
        profile = CompetitionProfile.model_validate(document)
    except (ValidationError, WireContractError, TypeError, ValueError) as error:
        raise CliContractError("competition profile is invalid") from error
    if profile.to_json() != payload:
        raise CliContractError("competition profile must be canonical JSON")
    if competition and payload != _competition_profile_bytes():
        raise CliContractError("competition profile differs from the immutable preregistration")
    if competition:
        _validate_competition_partitions(profile)
    return profile


def _validate_competition_partitions(profile: CompetitionProfile) -> None:
    if profile.campaigns_per_family != 50:
        raise CliContractError("competition profile campaign count differs")
    flattened: list[int] = []
    for name in _PARTITIONS:
        low, high = profile.partition_campaign_indices[name]
        flattened.extend(range(low, high + 1))
    if flattened != list(range(50)):
        raise CliContractError("competition partitions must cover whole campaigns exactly")
    if profile.campaign_spacing_days <= profile.label_delay_days:
        raise CliContractError("competition partition label-maturity embargo is invalid")


@dataclass(frozen=True, slots=True)
class G3FixtureResult:
    run_manifests_verified: int
    arms: tuple[str, ...]
    scorecard_ref: ArtifactRef
    public_artifacts: dict[str, ArtifactRef]
    core_artifact_digests: tuple[str, ...]
    signer_key_id: str
    defender_ref: ArtifactRef
    threshold_set_ref: ArtifactRef
    evaluation_bundle_ref: ArtifactRef
    fixture_control_ref: ArtifactRef
    fixture_control_campaign_count: int
    ensemble_mode: Literal["reduced_pooled_only"] = "reduced_pooled_only"
    champion_status: Literal["no_promotion"] = "no_promotion"
    reduced_fixture_evidence: bool = True
    competition_evidence: bool = False


@dataclass(frozen=True, slots=True)
class _FixtureTraining:
    defender_ref: ArtifactRef
    split: EvaluationSplit
    corpus: FrozenCorpus


class RunLedgerEntry(ExternalContract):
    """One ordered authenticated campaign reference in the competition ledger."""

    family: Family
    campaign_index: int = Field(ge=0, le=49)
    seed: int = Field(ge=0)
    simulation_start_utc: datetime
    run_id: str
    manifest: dict[str, object]

    @field_validator("simulation_start_utc")
    @classmethod
    def ledger_start_is_utc(cls, value: datetime) -> datetime:
        return validate_utc_timestamp(value)

    @model_validator(mode="after")
    def reference_is_exact(self) -> RunLedgerEntry:
        _artifact_ref(self.manifest)
        if not self.run_id.startswith("run-") or len(self.run_id) != 36:
            raise ValueError("ledger run ID is invalid")
        return self


class FixtureControlReceipt(ExternalContract):
    """Signed disclosure for the reduced-only derived chronological control."""

    schema_version: Literal["1.0.0"] = "1.0.0"
    kind: Literal["fixture_class_support_control"] = "fixture_class_support_control"
    source_run_id: str
    source_campaign_id: str
    control_campaign_id: str
    source_corpus_digest: str
    control_observation_digest: str
    control_truth_digest: str
    signer_key_id: str
    public_key_base64: str
    signature_base64: str

    @model_validator(mode="after")
    def receipt_is_closed(self) -> FixtureControlReceipt:
        for value in (
            self.source_corpus_digest,
            self.control_observation_digest,
            self.control_truth_digest,
            self.signer_key_id,
        ):
            _digest(value)
        if not self.source_run_id.startswith("run-"):
            raise ValueError("fixture control source run is invalid")
        return self

    def unsigned_document(self) -> dict[str, object]:
        return self.model_dump(
            mode="json", exclude={"signature_base64"}
        )


class CompetitionRunLedger(ExternalContract):
    """Complete profile-bound ordered manifest ledger."""

    schema_version: Literal["1.0.0"] = "1.0.0"
    profile_sha256: str
    signer_key_id: str
    public_key_base64: str
    entries: tuple[RunLedgerEntry, ...]

    @model_validator(mode="after")
    def ledger_is_complete(self) -> CompetitionRunLedger:
        _digest(self.profile_sha256)
        _digest(self.signer_key_id)
        expected = tuple(
            (family, index)
            for family in _FAMILIES
            for index in range(50)
        )
        actual = tuple((entry.family, entry.campaign_index) for entry in self.entries)
        if actual != expected:
            raise ValueError("competition run ledger is not complete and ordered")
        if len({entry.run_id for entry in self.entries}) != len(self.entries):
            raise ValueError("competition run ledger contains duplicate runs")
        return self


class CorpusEnvelope(ExternalContract):
    """Content reference and public lineage for an authenticated frozen corpus."""

    schema_version: Literal["1.0.0"] = "1.0.0"
    profile_sha256: str
    run_ledger_sha256: str
    observations: dict[str, object]
    restricted_truth: dict[str, object]
    observation_digest: str
    restricted_truth_digest: str
    corpus_digest: str
    campaign_count: int = Field(ge=1)
    family_campaign_counts: dict[Family, int]
    signer_key_id: str | None = None
    public_key_base64: str | None = None
    signature_base64: str | None = None

    @model_validator(mode="after")
    def envelope_is_closed(self) -> CorpusEnvelope:
        _digest(self.profile_sha256)
        _digest(self.run_ledger_sha256)
        for value in (
            self.observation_digest,
            self.restricted_truth_digest,
            self.corpus_digest,
        ):
            _digest(value)
        _artifact_ref(self.observations)
        _artifact_ref(self.restricted_truth)
        if set(self.family_campaign_counts) != set(_FAMILIES):
            raise ValueError("corpus family counts are incomplete")
        signatures = (
            self.signer_key_id,
            self.public_key_base64,
            self.signature_base64,
        )
        if any(item is not None for item in signatures) and any(
            item is None for item in signatures
        ):
            raise ValueError("corpus envelope signature identity is incomplete")
        if self.signer_key_id is not None:
            _digest(self.signer_key_id)
        return self

    def unsigned_document(self) -> dict[str, object]:
        return self.model_dump(mode="json", exclude={"signature_base64"})


@dataclass(frozen=True, slots=True)
class DefenderEnsemblePlan:
    """Truth-free structural plan used before signed reference publication."""

    pooled_ref: str
    lofo_refs: dict[Family, str]
    training_exclusions: dict[str, tuple[Family, ...]]


class DefenderEnsembleEnvelope(ExternalContract):
    """Signed full competition roster: pooled plus one true LOFO per family."""

    schema_version: Literal["1.0.0"] = "1.0.0"
    mode: Literal["competition_full"] = "competition_full"
    profile_sha256: str
    pooled_ref: dict[str, object]
    held_family_refs: dict[Family, dict[str, object]]
    held_family_training_exclusions: dict[Family, tuple[Family, ...]]
    corpus_envelope_ref: dict[str, object] | None = None
    signer_key_id: str
    public_key_base64: str
    signature_base64: str

    @model_validator(mode="after")
    def ensemble_is_complete(self) -> DefenderEnsembleEnvelope:
        _digest(self.profile_sha256)
        _digest(self.signer_key_id)
        pooled = _artifact_ref(self.pooled_ref)
        if set(self.held_family_refs) != set(_FAMILIES):
            raise ValueError("held-family defender roster is incomplete")
        held = {
            family: _artifact_ref(reference)
            for family, reference in self.held_family_refs.items()
        }
        if self.corpus_envelope_ref is not None:
            _artifact_ref(self.corpus_envelope_ref)
        if len({pooled.sha256, *(item.sha256 for item in held.values())}) != 5:
            raise ValueError("competition defender roles must use distinct bundles")
        if self.held_family_training_exclusions != {
            family: (family,) for family in _FAMILIES
        }:
            raise ValueError("held-family training exclusions are incomplete")
        return self

    def unsigned_document(self) -> dict[str, object]:
        return self.model_dump(mode="json", exclude={"signature_base64"})


def _build_defender_ensemble(
    *,
    pooled_ref: str | ArtifactRef,
    lofo_refs: dict[Family, str] | None = None,
    training_exclusions: dict[str, tuple[Family, ...]] | None = None,
    profile: CompetitionProfile | None = None,
    held_family_refs: dict[Family, ArtifactRef] | None = None,
    signer: RunSigningIdentity | None = None,
    corpus_envelope_ref: ArtifactRef | None = None,
) -> DefenderEnsemblePlan | DefenderEnsembleEnvelope:
    """Close both the prepublication plan and the signed competition envelope."""
    if type(pooled_ref) is str:
        if lofo_refs is None or training_exclusions is None:
            raise CliContractError("defender ensemble plan is incomplete")
        expected_exclusions: dict[str, tuple[Family, ...]] = {
            "pooled": (),
            **{family: (family,) for family in _FAMILIES},
        }
        if (
            set(lofo_refs) != set(_FAMILIES)
            or training_exclusions != expected_exclusions
            or len({pooled_ref, *lofo_refs.values()}) != 5
        ):
            raise CliContractError("defender ensemble roles are incomplete or duplicated")
        for digest in (pooled_ref, *lofo_refs.values()):
            _digest(digest)
        return DefenderEnsemblePlan(pooled_ref, dict(lofo_refs), dict(training_exclusions))
    if (
        type(pooled_ref) is not ArtifactRef
        or type(profile) is not CompetitionProfile
        or held_family_refs is None
        or type(signer) is not RunSigningIdentity
        or profile.fixture_only
    ):
        raise CliContractError("signed competition defender ensemble is incomplete")
    unsigned = {
        "held_family_refs": {
            family: _reference_document(held_family_refs[family])
            for family in profile.families
            if family in held_family_refs
        },
        "held_family_training_exclusions": {
            family: [family] for family in profile.families
        },
        "corpus_envelope_ref": (
            None
            if corpus_envelope_ref is None
            else _reference_document(corpus_envelope_ref)
        ),
        "mode": "competition_full",
        "pooled_ref": _reference_document(pooled_ref),
        "profile_sha256": hashlib.sha256(profile.to_json()).hexdigest(),
        "public_key_base64": signer.public_key_base64,
        "schema_version": "1.0.0",
        "signer_key_id": signer.key_id,
    }
    try:
        envelope = DefenderEnsembleEnvelope.model_validate(
            {**unsigned, "signature_base64": signer.sign(unsigned)}
        )
    except (KeyError, ValidationError, ValueError) as error:
        raise CliContractError("signed competition defender ensemble is invalid") from error
    if not signer.verify(envelope.unsigned_document(), envelope.signature_base64):
        raise CliContractError("competition defender ensemble signature failed")
    return envelope


def run_g3_fixture(root: Path) -> G3FixtureResult:
    """Run four real families plus one disclosed reduced-only control artifact."""
    checked_root = _secure_root(root)
    store = ArtifactStore(checked_root / "artifacts")
    signer = RunSigningIdentity.from_private_bytes(_FIXTURE_SIGNER_SEED)
    runner = RunRunner(store, signer, checked_root / "runs")
    profile = CompetitionProfile.fixture()
    fixture_schedule: tuple[tuple[Family, int], ...] = (
        ("app_scam_mule", 1),
        ("synthetic_merchant_refund", 9),
        ("agentic_intent_abuse", 17),
        ("card_testing_cnp", 25),
    )
    manifests = tuple(
        _run_one_campaign(
            family,
            index=0,
            profile=profile,
            runner=runner,
            fixture=True,
            start_override=profile.simulation_start_utc + timedelta(days=day),
            seed_override=(profile.seed_bases[family] + schedule_index),
        )
        for schedule_index, (family, day) in enumerate(fixture_schedule)
    )
    if not all(runner.verify_run(manifest) for manifest in manifests):
        raise CliContractError("fixture run authentication failed")
    authenticated_corpus = assemble_verified_corpus(
        manifests,
        runner,
        store,
        CorpusProfile.fixture(),
    )
    corpus, control_ref = _add_fixture_class_support_control(
        authenticated_corpus,
        source_run_id=manifests[0].run_id,
        store=store,
        signer=signer,
    )
    training = _run_fixture_model_pipeline(corpus, profile, store, signer)
    for manifest in manifests:
        store.put_json(manifest)
    from apar.evaluation.competition import publish_reduced_g3_evaluation

    published = publish_reduced_g3_evaluation(
        store=store,
        publication_signer=signer,
        defender_ref=training.defender_ref,
        corpus=training.corpus,
        split=training.split,
        profile_sha256=hashlib.sha256(profile.to_json()).hexdigest(),
        authenticated_run_ids=tuple(manifest.run_id for manifest in manifests),
    )
    scorecard_ref = published.scorecard_ref
    public_artifacts = published.public_artifacts
    core = tuple(
        reference.sha256
        for _, reference in sorted(public_artifacts.items())
    )
    return G3FixtureResult(
        run_manifests_verified=len(manifests),
        arms=tuple(arm.value for arm in DefenseArm),
        scorecard_ref=scorecard_ref,
        public_artifacts=public_artifacts,
        core_artifact_digests=core,
        signer_key_id=signer.key_id,
        defender_ref=training.defender_ref,
        threshold_set_ref=published.threshold_set_ref,
        evaluation_bundle_ref=published.evaluation_bundle_ref,
        fixture_control_ref=control_ref,
        fixture_control_campaign_count=1,
    )


def _add_fixture_class_support_control(
    corpus: FrozenCorpus,
    *,
    source_run_id: str,
    store: ArtifactStore,
    signer: RunSigningIdentity,
) -> tuple[FrozenCorpus, ArtifactRef]:
    """Derive and sign one deterministic earlier cohort; never used by export CLIs."""
    source_truth = tuple(
        row for row in corpus.truth if row.family == "app_scam_mule"
    )
    source_campaigns = {row.campaign_id for row in source_truth}
    if not source_truth or len(source_campaigns) != 1:
        raise CliContractError("fixture control requires one authenticated APP campaign")
    source_campaign_id = next(iter(source_campaigns))
    source_payment_ids = {row.payment_id for row in source_truth}
    source_observations = tuple(
        row for row in corpus.observations if row.payment_id in source_payment_ids
    )
    shift = -timedelta(days=1)
    control_campaign_id = "fixture-control-" + hashlib.sha256(
        source_campaign_id.encode("utf-8")
    ).hexdigest()[:24]

    def mapped(label: str, value: str) -> str:
        return str(uuid.uuid5(uuid.NAMESPACE_URL, f"apar:g3:{label}:{value}"))

    event_ids = {
        row.event_id: mapped("control-event", row.event_id)
        for row in source_observations
    }
    payment_ids = {
        row.payment_id: f"fixture-control:{mapped('control-payment', row.payment_id)}"
        for row in source_observations
    }
    control_observations = tuple(
        sorted(
            (
                row.model_copy(
                    update={
                        "event_id": event_ids[row.event_id],
                        "payment_id": payment_ids[row.payment_id],
                        "event_time": row.event_time + shift,
                        "available_at": row.available_at + shift,
                        "decision_at": (
                            None
                            if row.decision_at is None
                            else row.decision_at + shift
                        ),
                        "actor_id": mapped("control-actor", row.actor_id),
                        "counterparty_id": mapped(
                            "control-counterparty", row.counterparty_id
                        ),
                    }
                )
                for row in source_observations
            ),
            key=lambda row: row.event_id,
        )
    )
    control_truth = tuple(
        sorted(
            (
                row.model_copy(
                    update={
                        "event_id": event_ids[row.event_id],
                        "payment_id": payment_ids[row.payment_id],
                        "campaign_id": control_campaign_id,
                        "label_mature_at": row.label_mature_at + shift,
                        "first_settlement_at": (
                            None
                            if row.first_settlement_at is None
                            else row.first_settlement_at + shift
                        ),
                        "lifecycle_event_ids": tuple(
                            event_ids[item] for item in row.lifecycle_event_ids
                        ),
                    }
                )
                for row in source_truth
            ),
            key=lambda row: row.event_id,
        )
    )
    source_digest = frozen_corpus_digest(corpus)
    observation_digest = hashlib.sha256(
        canonical_json_bytes(
            [row.model_dump(mode="json") for row in control_observations]
        )
    ).hexdigest()
    truth_digest = hashlib.sha256(
        canonical_json_bytes([row.model_dump(mode="json") for row in control_truth])
    ).hexdigest()
    unsigned = {
        "control_campaign_id": control_campaign_id,
        "control_observation_digest": observation_digest,
        "control_truth_digest": truth_digest,
        "kind": "fixture_class_support_control",
        "public_key_base64": signer.public_key_base64,
        "schema_version": "1.0.0",
        "signer_key_id": signer.key_id,
        "source_campaign_id": source_campaign_id,
        "source_corpus_digest": source_digest,
        "source_run_id": source_run_id,
    }
    receipt = FixtureControlReceipt.model_validate(
        {**unsigned, "signature_base64": signer.sign(unsigned)}
    )
    if not signer.verify(receipt.unsigned_document(), receipt.signature_base64):
        raise CliContractError("fixture control signature failed verification")
    receipt_ref = store.put_json(receipt)
    receipt_digest = hashlib.sha256(
        canonical_json_bytes(receipt.model_dump(mode="json"))
    ).hexdigest()
    observations = tuple(
        sorted((*corpus.observations, *control_observations), key=lambda row: row.event_id)
    )
    truth = tuple(sorted((*corpus.truth, *control_truth), key=lambda row: row.event_id))
    return (
        FrozenCorpus(
            observations=observations,
            truth=truth,
            manifest=CorpusManifest(
                profile_id="development-fixture-v1+signed-control-v1",
                run_ids=(*corpus.manifest.run_ids, "fixture-class-support-control-v1"),
                run_lineage_digests=(
                    *corpus.manifest.run_lineage_digests,
                    receipt_digest,
                ),
                observation_count=len(observations),
                truth_count=len(truth),
            ),
        ),
        receipt_ref,
    )


def _run_one_campaign(
    family: Family,
    *,
    index: int,
    profile: CompetitionProfile,
    runner: RunRunner,
    fixture: bool,
    start_override: datetime | None = None,
    seed_override: int | None = None,
) -> RunManifest:
    bundle, policy = _compile_campaign_inputs(
        family,
        index=index,
        profile=profile,
        fixture=fixture,
        start_override=start_override,
        seed_override=seed_override,
    )
    return runner.execute(bundle, policy)


def _compile_campaign_inputs(
    family: Family,
    *,
    index: int,
    profile: CompetitionProfile,
    fixture: bool,
    start_override: datetime | None = None,
    seed_override: int | None = None,
) -> tuple[ScenarioBundle, AttackerPolicy]:
    """Compile the one canonical scenario/policy pair admitted for a campaign slot."""
    path = _REPOSITORY_ROOT / "fixtures" / "threats" / _THREAT_CARD_BY_FAMILY[family]
    card = ThreatCard.model_validate_json(
        _regular_file(path, label="threat card", max_bytes=512_000)
    )
    seed = profile.campaign_seed(family, index) if seed_override is None else seed_override
    if type(seed) is not int or not 0 <= seed < 2**63:
        raise CliContractError("scenario seed override is invalid")
    start = (
        profile.campaign_start(family, index)
        if start_override is None
        else validate_utc_timestamp(start_override)
    )
    config = ScenarioConfig.model_validate(
        card.default_config.model_copy(
            update={
                "seed": seed,
                "query_budget": 1,
                "benign_entity_count": 40 if fixture else card.default_config.benign_entity_count,
                "illicit_entity_count": 16 if fixture else card.default_config.illicit_entity_count,
                "replay": card.default_config.replay.model_copy(
                    update={"random_seed": seed, "simulation_start": start}
                ),
            }
        ).model_dump(mode="json")
    )
    bundle = bind_scenario_for_run(
        compile_scenario(card, config), threat_family=family
    )
    return (
        bundle,
        AttackerPolicy(
            family=family,
            attacker_mode=config.attacker_mode,
            kind=AttackerPolicyKind.FIXED,
            query_budget=1,
            worker_timeout_ms=5_000,
        ),
    )


def _run_fixture_model_pipeline(
    corpus: FrozenCorpus,
    profile: CompetitionProfile,
    store: ArtifactStore,
    signer: RunSigningIdentity,
) -> _FixtureTraining:
    catalog = load_feature_catalog(_CATALOG)
    start = profile.simulation_start_utc
    split = make_evaluation_split(
        corpus,
        SplitConfig(
            train_end=start + timedelta(days=8, hours=1),
            calibrator_fit_end=start + timedelta(days=16, hours=1),
            threshold_end=start + timedelta(days=24, hours=1),
            development_end=start + timedelta(days=32, hours=1),
        ),
    )
    train_ids = split.training_row_ids
    fit_ids = split.row_ids["calibrator_fit"]
    selection_ids = split.row_ids["threshold"]
    evaluation_ids = split.row_ids["development"]
    if not all((train_ids, fit_ids, selection_ids, evaluation_ids)):
        raise CliContractError("fixture split must contain all four partitions")
    training_matrix = _partition_feature_matrix(corpus, split, "train", catalog)
    fit_matrix = _partition_feature_matrix(corpus, split, "calibrator_fit", catalog)
    selection_matrix = _partition_feature_matrix(corpus, split, "threshold", catalog)
    evaluation_matrix = _partition_feature_matrix(corpus, split, "development", catalog)
    labels = {item: int(split.row_is_fraud[item]) for item in train_ids}
    event_by_id = {event.event_id: event for event in training_matrix.events}
    row_by_id = {row.event_id: row for row in training_matrix.rows}
    rules = RuleEngine.default()
    mandatory_train_ids = tuple(
        item
        for item in train_ids
        if any(
            hit.mandatory
            for hit in rules.evaluate(event_by_id[item], row_by_id[item]).hits
        )
    )
    eligible_train_ids = tuple(item for item in train_ids if item not in set(mandatory_train_ids))
    folds = _fixture_folds(eligible_train_ids, labels)
    scorer = train_gbdt(
        training_matrix,
        labels,
        train_ids,
        folds,
        GbdtTrainingConfig(
            seed=profile.model_seed,
            depths=profile.gbdt.depths,
            learning_rates=profile.gbdt.learning_rates,
            l2_leaf_regs=profile.gbdt.l2_leaf_regs,
            iterations=profile.gbdt.iterations,
        ),
        training_cutoff=split.config.train_end,
        mandatory_row_ids=mandatory_train_ids,
    )
    fit_labels = np.asarray(
        [int(split.row_is_fraud[item]) for item in fit_ids], dtype=np.int64
    )
    selection_labels = np.asarray(
        [int(split.row_is_fraud[item]) for item in selection_ids], dtype=np.int64
    )
    calibrator = select_calibrator(
        scorer.predict(fit_matrix),
        fit_labels,
        scorer.predict(selection_matrix),
        selection_labels,
        min_class_count=profile.calibration.minimum_class_count,
    )
    event_by_id = {event.event_id: event for event in selection_matrix.events}
    row_by_id = {row.event_id: row for row in selection_matrix.rows}
    rule_results = tuple(
        rules.evaluate(event_by_id[item], row_by_id[item]) for item in selection_ids
    )
    rule_scores = np.asarray(
        [result.score for result in rule_results],
        dtype=np.float64,
    )
    model_scores = calibrator.predict(scorer.predict(selection_matrix))
    mandatory = np.asarray(
        [
            Action.DECLINE
            if any(hit.mandatory for hit in result.hits)
            else Action.APPROVE
            for result in rule_results
        ],
        dtype=object,
    )
    budget = OperatingBudget(
        challenge_rate_max=profile.budgets.challenge_rate_max,
        false_decline_rate_max=profile.budgets.false_decline_rate_max,
        review_case_rate_max=profile.budgets.review_case_rate_max,
    )

    selection_as_of = split.config.development_end + timedelta(days=profile.label_delay_days)
    case_observations = selection_matrix.events
    case_binding = bind_replay_case_counter(
        case_observations,
        selection_ids,
        as_of=selection_as_of,
    )
    case_count = case_binding.reconstruct(
        case_observations,
        selection_ids,
        selection_as_of,
    )

    reports = []
    for scores in (rule_scores, model_scores, np.maximum(rule_scores, model_scores)):
        reports.append(
            select_policy_thresholds(
                scores,
                selection_labels,
                mandatory,
                cast(Any, case_count),
                budget,
                None,
            )
        )
    if not all(report.feasible for report in reports):
        raise CliContractError("fixture matched-budget threshold selection is infeasible")
    corpus_bytes = _frozen_corpus_bytes(corpus)
    observation_bytes = canonical_json_bytes(
        [item.model_dump(mode="json") for item in corpus.observations]
    )
    truth_bytes = canonical_json_bytes([item.model_dump(mode="json") for item in corpus.truth])
    lineage = BundleLineage(
        corpus_digest=frozen_corpus_digest(corpus),
        observation_dataset_digest=_lineage_digest("observations", observation_bytes),
        evaluator_truth_digest=_lineage_digest("truth", truth_bytes),
        split_manifest_digest=split.split_digest,
        feature_provenance_digest=_lineage_digest(
            "features",
            canonical_json_bytes(
                [
                    item.model_dump(mode="json")
                    for item in (
                        training_matrix,
                        fit_matrix,
                        selection_matrix,
                        evaluation_matrix,
                    )
                ]
            ),
        ),
        hyperparameter_digest=hashlib.sha256(
            canonical_json_bytes(scorer.receipt.selected_params.model_dump(mode="json"))
        ).hexdigest(),
        reason_code_mapping_digest=_lineage_digest(
            "reasons", canonical_json_bytes(RuleManifest.default().model_dump(mode="json"))
        ),
    )
    source_paths = (
        "config/defense/competition-profile.json",
        "config/defense/feature-catalog.json",
        "src/apar/cases/grouping.py",
        "src/apar/defense/bundle.py",
        "src/apar/defense/calibration.py",
        "src/apar/defense/gbdt.py",
        "src/apar/defense/policy.py",
        "src/apar/defense/rules.py",
        "src/apar/defense/thresholds.py",
        "src/apar/defense/orchestration.py",
        "src/apar/evaluation/competition.py",
        "src/apar/evaluation/regimes.py",
        "src/apar/evaluation/replay.py",
        "src/apar/evaluation/splits.py",
        "src/apar/features/builders.py",
        "src/apar/features/catalog.py",
        "src/apar/features/state.py",
    )
    publisher = DefenderBundlePublisher(store, signer, _REPOSITORY_ROOT)
    try:
        _manifest, reference = publisher.freeze(
            scorer=scorer,
            catalog=catalog,
            split=split,
            training_matrix=training_matrix,
            mandatory_excluded_row_ids=mandatory_train_ids,
            calibration_fit_matrix=fit_matrix,
            calibration_fit_labels=fit_labels,
            calibration_selection_matrix=selection_matrix,
            calibration_selection_labels=selection_labels,
            threshold_matrix=selection_matrix,
            threshold_labels=selection_labels,
            threshold_mandatory_actions=mandatory,
            threshold_values=None,
            review_case_counter=cast(Any, case_count),
            rule_manifest=RuleManifest.default(),
            calibrator=calibrator,
            threshold_report=reports[-1],
            lineage=lineage,
            environment_lock=current_environment_lock(),
            source_inventory=build_source_inventory(_REPOSITORY_ROOT, source_paths),
            reload_matrix=evaluation_matrix,
            bundle_id=str(
                uuid.uuid5(
                    uuid.NAMESPACE_URL,
                    f"apar:g3-fixture:{hashlib.sha256(corpus_bytes).hexdigest()}",
                )
            ),
            frozen_at=split.config.development_end,
            rollback_ref=GENESIS_ROLLBACK_REF,
        )
        loaded = publisher.load(reference)
        loaded.verify_reload()
        return _FixtureTraining(reference, split, corpus)
    finally:
        publisher.close()


def _train_competition_defender(
    *,
    profile: CompetitionProfile,
    root: Path,
    corpus_envelope_digest: str,
    catalog_path: Path,
    rollback_ref: str,
) -> ArtifactRef:
    """Freeze pooled and all four true LOFO bundles before publishing one roster."""
    pooled = _freeze_competition_candidate(
        profile=profile,
        root=root,
        corpus_envelope_digest=corpus_envelope_digest,
        catalog_path=catalog_path,
        rollback_ref=rollback_ref,
        held_out_family=None,
    )
    held = {
        family: _freeze_competition_candidate(
            profile=profile,
            root=root,
            corpus_envelope_digest=corpus_envelope_digest,
            catalog_path=catalog_path,
            rollback_ref=rollback_ref,
            held_out_family=family,
        )
        for family in profile.families
    }
    signer = _load_standard_signer(root)
    envelope = _build_defender_ensemble(
        profile=profile,
        pooled_ref=pooled,
        held_family_refs=held,
        signer=signer,
        corpus_envelope_ref=ArtifactStore(root / "artifacts").resolve(
            _digest(corpus_envelope_digest)
        ),
    )
    if type(envelope) is not DefenderEnsembleEnvelope:
        raise CliContractError("competition defender ensemble type differs")
    return ArtifactStore(root / "artifacts").put_bytes(
        canonical_json_bytes(envelope.model_dump(mode="json")),
        _DEFENDER_ENSEMBLE_MEDIA_TYPE,
    )


def _fixture_windows(
    rows: tuple[FeatureVector, ...], labels: dict[str, int]
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    row_ids = tuple(row.event_id for row in rows)
    by_class = {
        value: [row_id for row_id in row_ids if labels[row_id] == value]
        for value in (0, 1)
    }
    if min(len(items) for items in by_class.values()) < 4:
        raise CliContractError("fixture needs at least four decision rows from each class")
    fit_ids = tuple(sorted((by_class[0][-3], by_class[0][-2], by_class[1][-3], by_class[1][-2])))
    selection_ids = tuple(sorted((by_class[0][-1], by_class[1][-1])))
    reserved = set((*fit_ids, *selection_ids))
    train_ids = tuple(item for item in row_ids if item not in reserved)
    if min(sum(labels[item] == value for item in train_ids) for value in (0, 1)) < 2:
        raise CliContractError("fixture training window needs both classes")
    evaluation_ids = selection_ids
    return train_ids, fit_ids, selection_ids, evaluation_ids


def _fixture_folds(
    train_ids: tuple[str, ...], labels: dict[str, int]
) -> tuple[RollingFold, ...]:
    ordered = tuple(train_ids)
    pair: tuple[tuple[str, ...], tuple[str, ...]] | None = None
    for boundary in range(len(ordered) - 2, 1, -1):
        candidate = (ordered[:boundary], ordered[boundary:])
        if all({labels[item] for item in window} == {0, 1} for window in candidate):
            pair = candidate
            break
    if pair is None:
        raise CliContractError("fixture rolling fold needs both chronological classes")
    fit, validation = pair
    return (RollingFold(name="fixture-fold", fit_ids=fit, validation_ids=validation),)


def _matrix_subset(matrix: FeatureMatrix, ids: tuple[str, ...]) -> FeatureMatrix:
    event_by_id = {item.event_id: item for item in matrix.events}
    row_by_id = {item.event_id: item for item in matrix.rows}
    return FeatureMatrix(
        events=tuple(event_by_id[item] for item in ids),
        catalog=matrix.catalog,
        catalog_digest=matrix.catalog_digest,
        rows=tuple(row_by_id[item] for item in ids),
    )


def _partition_feature_matrix(
    corpus: FrozenCorpus,
    split: EvaluationSplit,
    partition: Literal["train", "calibrator_fit", "threshold", "development"],
    catalog: FeatureCatalog,
) -> FeatureMatrix:
    row_ids = frozenset(split.row_ids[partition])
    truth_by_id = {row.event_id: row for row in corpus.truth}
    lifecycle_ids = {
        lifecycle_id
        for event_id in row_ids
        for lifecycle_id in truth_by_id[event_id].lifecycle_event_ids
    }
    events = tuple(
        event for event in corpus.observations if event.event_id in lifecycle_ids
    )
    complete = build_feature_matrix(events, catalog)
    matrix = _matrix_subset(complete, split.row_ids[partition])
    if tuple(row.event_id for row in complete.rows) != split.row_ids[partition]:
        raise CliContractError("partition feature rows differ from evaluator split")
    return matrix


def _case_observations(
    events: tuple[ObservedEvent, ...], decision_ids: tuple[str, ...]
) -> tuple[ObservedEvent, ...]:
    selected = frozenset(decision_ids)
    return tuple(
        event
        for event in events
        if not event.is_decision_point or event.event_id in selected
    )


def _frozen_corpus_bytes(corpus: FrozenCorpus) -> bytes:
    return canonical_json_bytes(
        {
            "manifest": corpus.manifest.model_dump(mode="json"),
            "observations": [item.model_dump(mode="json") for item in corpus.observations],
            "truth": [item.model_dump(mode="json") for item in corpus.truth],
        }
    )


def _load_frozen_corpus(payload: bytes) -> FrozenCorpus:
    from apar.evaluation.contracts import CorpusManifest, EvaluationTruthRow

    try:
        document = strict_json_loads(payload)
        if type(document) is not dict or set(document) != {
            "manifest",
            "observations",
            "truth",
        }:
            raise CliContractError("corpus artifact fields differ")
        observations = document["observations"]
        truth = document["truth"]
        if type(observations) is not list or type(truth) is not list:
            raise CliContractError("corpus row collections are invalid")
        corpus = FrozenCorpus(
            observations=tuple(ObservedEvent.model_validate(item) for item in observations),
            truth=tuple(EvaluationTruthRow.model_validate(item) for item in truth),
            manifest=CorpusManifest.model_validate(document["manifest"]),
        )
    except CliContractError:
        raise
    except (ValidationError, WireContractError, TypeError, ValueError) as error:
        raise CliContractError("corpus artifact is invalid") from error
    if _frozen_corpus_bytes(corpus) != payload:
        raise CliContractError("corpus artifact is not canonical")
    return corpus


def _artifact_ref(document: object) -> ArtifactRef:
    if type(document) is not dict or set(document) != {
        "sha256",
        "media_type",
        "size_bytes",
        "relative_path",
    }:
        raise CliContractError("artifact reference fields differ")
    try:
        reference = ArtifactRef(
            sha256=cast(str, document["sha256"]),
            media_type=cast(str, document["media_type"]),
            size_bytes=cast(int, document["size_bytes"]),
            relative_path=cast(str, document["relative_path"]),
        )
        _digest(reference.sha256)
        if (
            type(reference.media_type) is not str
            or not reference.media_type
            or type(reference.size_bytes) is not int
            or reference.size_bytes < 0
            or reference.relative_path != f"{reference.sha256}/payload"
        ):
            raise CliContractError("artifact reference is invalid")
        return reference
    except (KeyError, TypeError, ValueError) as error:
        raise CliContractError("artifact reference is invalid") from error


def _signer_path(root: Path, raw: str) -> Path:
    candidate = Path(raw)
    expected = root / "run-signing.key"
    if not candidate.is_absolute() or candidate != expected:
        raise CliContractError("signer must be the pinned root signing identity")
    return candidate


def _load_standard_signer(root: Path) -> RunSigningIdentity:
    path = root / "run-signing.key"
    return _load_existing_signer(path, root=root)


def _load_evaluator_seed(root: Path, filename: str) -> bytes:
    path = root / filename
    try:
        info = path.lstat()
        if (
            stat.S_ISLNK(info.st_mode)
            or not stat.S_ISREG(info.st_mode)
            or stat.S_IMODE(info.st_mode) & 0o077
        ):
            raise CliContractError(
                "competition evaluator signing identity must be a private regular file"
            )
        seed = path.read_bytes()
    except FileNotFoundError as error:
        raise CliContractError(
            "competition evaluator signing identity is unavailable"
        ) from error
    except OSError as error:
        raise CliContractError(
            "competition evaluator signing identity cannot be read"
        ) from error
    if len(seed) != 32:
        raise CliContractError(
            "competition evaluator signing identity must contain 32 bytes"
        )
    return seed


def _load_competition_evaluator_identity(root: Path) -> EvaluatorSigningIdentity:
    """Load only the development evaluator; never touch hidden authority state."""
    seed = _load_evaluator_seed(root, "evaluator-signing.key")
    fixture_seeds = {
        hashlib.sha256(b"apar-g3-development-evaluator-v1").digest(),
        hashlib.sha256(b"apar-g3-hidden-evaluator-v1").digest(),
    }
    if seed in fixture_seeds:
        raise CliContractError(
            "competition evaluator identity cannot use a deterministic fixture key"
        )
    publication_seed = (root / "run-signing.key").read_bytes()
    if seed == publication_seed:
        raise CliContractError(
            "competition publication and evaluator identities must be distinct"
        )
    try:
        return EvaluatorSigningIdentity.from_private_bytes(seed)
    except (TypeError, ValueError) as error:
        raise CliContractError("competition evaluator signing identity is invalid") from error


def _load_competition_hidden_identity(
    root: Path, evaluator: EvaluatorSigningIdentity
) -> tuple[EvaluatorSigningIdentity, RunSigningIdentity]:
    """Load the hidden private authority only after hidden-release verification."""
    seed = _load_evaluator_seed(root, "hidden-evaluator-signing.key")
    fixture_seeds = {
        hashlib.sha256(b"apar-g3-development-evaluator-v1").digest(),
        hashlib.sha256(b"apar-g3-hidden-evaluator-v1").digest(),
    }
    if seed in fixture_seeds:
        raise CliContractError(
            "competition evaluator identity cannot use a deterministic fixture key"
        )
    publication_seed = (root / "run-signing.key").read_bytes()
    try:
        hidden = EvaluatorSigningIdentity.from_private_bytes(seed)
        hidden_context = RunSigningIdentity.from_private_bytes(seed)
    except (TypeError, ValueError) as error:
        raise CliContractError("competition hidden signing identity is invalid") from error
    if seed == publication_seed or hidden.key_id == evaluator.key_id:
        raise CliContractError(
            "competition publication, evaluator, and hidden identities must be distinct"
        )
    return hidden, hidden_context


def _load_competition_evaluator_identities(
    root: Path,
) -> tuple[EvaluatorSigningIdentity, EvaluatorSigningIdentity]:
    """Compatibility helper that explicitly loads both competition authorities."""
    evaluator = _load_competition_evaluator_identity(root)
    hidden, _ = _load_competition_hidden_identity(root, evaluator)
    return evaluator, hidden


def _load_existing_signer(path: Path, *, root: Path) -> RunSigningIdentity:
    if path != root / "run-signing.key":
        raise CliContractError("pinned run signing identity path differs")
    try:
        info = path.lstat()
        if (
            stat.S_ISLNK(info.st_mode)
            or not stat.S_ISREG(info.st_mode)
            or stat.S_IMODE(info.st_mode) & 0o077
        ):
            raise CliContractError(
                "pinned run signing identity must be a private regular file"
            )
        raw = path.read_bytes()
    except FileNotFoundError as error:
        raise CliContractError("pinned run signing identity is unavailable") from error
    except OSError as error:
        raise CliContractError("pinned run signing identity cannot be read") from error
    if len(raw) != 32:
        raise CliContractError("pinned run signing identity must contain 32 bytes")
    try:
        return RunSigningIdentity.from_private_bytes(raw)
    except (TypeError, ValueError) as error:
        raise CliContractError("pinned run signing identity is invalid") from error


def _generate_competition_runs(
    *,
    profile: CompetitionProfile,
    root: Path,
    signer_path: Path,
    output_name: str,
) -> ArtifactRef:
    store = ArtifactStore(root / "artifacts")
    signer = _load_existing_signer(signer_path, root=root)
    runner = RunRunner(store, signer, root / "runs")
    entries: list[RunLedgerEntry] = []
    for family in profile.families:
        for index in range(profile.campaigns_per_family):
            manifest = _run_one_campaign(
                family,
                index=index,
                profile=profile,
                runner=runner,
                fixture=False,
            )
            if not runner.verify_run(manifest):
                raise CliContractError("completed run failed authenticated verification")
            reference = store.put_json(manifest)
            entries.append(
                RunLedgerEntry(
                    family=family,
                    campaign_index=index,
                    seed=profile.campaign_seed(family, index),
                    simulation_start_utc=profile.campaign_start(family, index),
                    run_id=manifest.run_id,
                    manifest=_reference_document(reference),
                )
            )
    ledger = CompetitionRunLedger(
        profile_sha256=hashlib.sha256(profile.to_json()).hexdigest(),
        signer_key_id=signer.key_id,
        public_key_base64=signer.public_key_base64,
        entries=tuple(entries),
    )
    reference = store.put_json(ledger)
    _publish_json_file(root, output_name, ledger.model_dump(mode="json"))
    return reference


def _load_run_ledger(
    store: ArtifactStore, digest: str, profile: CompetitionProfile
) -> tuple[CompetitionRunLedger, ArtifactRef]:
    try:
        reference = store.resolve(_digest(digest))
        payload = store.read(reference)
        document = strict_json_loads(payload)
        ledger = CompetitionRunLedger.model_validate(document)
    except (ValidationError, WireContractError, ValueError) as error:
        raise CliContractError("authenticated run ledger is invalid") from error
    if canonical_json_bytes(ledger.model_dump(mode="json")) != payload:
        raise CliContractError("authenticated run ledger is not canonical")
    if ledger.profile_sha256 != hashlib.sha256(profile.to_json()).hexdigest():
        raise CliContractError("run ledger profile lineage differs")
    return ledger, reference


def _verify_ledger_entry(
    *,
    entry: RunLedgerEntry,
    expected_family: Family,
    expected_index: int,
    profile: CompetitionProfile,
    runner: RunRunner,
    store: ArtifactStore,
) -> RunManifest:
    """Bind an outer claim to the signed manifest, fixed policy, and scenario bytes."""
    expected_seed = profile.campaign_seed(expected_family, expected_index)
    expected_start = profile.campaign_start(expected_family, expected_index)
    if (
        entry.family != expected_family
        or entry.campaign_index != expected_index
        or entry.seed != expected_seed
        or entry.simulation_start_utc != expected_start
    ):
        raise CliContractError("run ledger campaign claim differs from the profile")
    try:
        manifest_ref = _artifact_ref(entry.manifest)
        manifest_payload = store.read(manifest_ref)
        manifest = RunManifest.model_validate_json(manifest_payload)
        if canonical_json_bytes(manifest.model_dump(mode="json")) != manifest_payload:
            raise CliContractError("run manifest artifact is not canonical")
        if manifest.run_id != entry.run_id or not runner.verify_run(manifest):
            raise CliContractError("run manifest failed authenticated verification")
        policy_payload = store.read(manifest.artifacts["policy"])
        scenario_payload = store.read(manifest.artifacts["scenario"])
        policy = AttackerPolicy.model_validate_json(policy_payload)
        scenario = ScenarioBundle.model_validate_json(scenario_payload)
        expected_scenario, expected_policy = _compile_campaign_inputs(
            expected_family,
            index=expected_index,
            profile=profile,
            fixture=profile.fixture_only,
        )
    except CliContractError:
        raise
    except (KeyError, ValidationError, WireContractError, TypeError, ValueError) as error:
        raise CliContractError("run ledger authenticated contents are invalid") from error
    if (
        policy_payload
        != canonical_json_bytes(expected_policy.model_dump(mode="json"))
        or scenario_payload
        != canonical_json_bytes(expected_scenario.model_dump(mode="json"))
        or policy != expected_policy
        or scenario != expected_scenario
        or
        policy.kind is not AttackerPolicyKind.FIXED
        or policy.family != expected_family
        or policy.query_budget != 1
        or scenario.query_budget != 1
        or scenario.seed != expected_seed
        or scenario.replay_manifest.random_seed != expected_seed
        or scenario.replay_manifest.simulation_start != expected_start
        or scenario.rail.value != _RAIL_BY_FAMILY[expected_family]
        or scenario.threat_card_ref != _THREAT_REF_BY_FAMILY[expected_family]
        or scenario.replay_manifest.threat_card_ref
        != _THREAT_REF_BY_FAMILY[expected_family]
    ):
        raise CliContractError("run policy or scenario differs from the profile")
    return manifest


def _build_competition_corpus(
    *,
    profile: CompetitionProfile,
    root: Path,
    ledger_digest: str,
    output_name: str,
) -> ArtifactRef:
    store = ArtifactStore(root / "artifacts")
    signer = _load_standard_signer(root)
    runner = RunRunner(store, signer, root / "runs")
    ledger, ledger_ref = _load_run_ledger(store, ledger_digest, profile)
    if (
        ledger.signer_key_id != signer.key_id
        or ledger.public_key_base64 != signer.public_key_base64
    ):
        raise CliContractError("run ledger signer identity differs")
    manifests: list[RunManifest] = []
    for expected, item in zip(
        (
            (family, index)
            for family in profile.families
            for index in range(profile.campaigns_per_family)
        ),
        ledger.entries,
        strict=True,
    ):
        manifests.append(
            _verify_ledger_entry(
                entry=item,
                expected_family=expected[0],
                expected_index=expected[1],
                profile=profile,
                runner=runner,
                store=store,
            )
        )
    corpus = assemble_verified_corpus(
        manifests,
        runner,
        store,
        CorpusProfile(
            profile_id="defense-competition-v1",
            families=profile.families,
            label_delay_days=profile.label_delay_days,
            fixture_only=False,
        ),
    )
    ownership: dict[str, Family] = {}
    family_campaigns: dict[Family, set[str]] = {
        family: set() for family in profile.families
    }
    for row in corpus.truth:
        existing = ownership.setdefault(row.campaign_id, row.family)
        if existing != row.family:
            raise CliContractError("competition campaign has multiple family owners")
        family_campaigns[row.family].add(row.campaign_id)
    counts = {family: len(family_campaigns[family]) for family in profile.families}
    if counts != {family: 50 for family in profile.families}:
        raise CliContractError("competition corpus must contain 50 campaigns per family")
    corpus_ref = store.put_bytes(
        canonical_json_bytes(
            [item.model_dump(mode="json") for item in corpus.observations]
        ),
        "application/vnd.apar.observations+json",
    )
    truth_ref = store.put_bytes(
        canonical_json_bytes(
            {
                "manifest": corpus.manifest.model_dump(mode="json"),
                "truth": [item.model_dump(mode="json") for item in corpus.truth],
            }
        ),
        "application/vnd.apar.restricted-truth+json",
    )
    observation_digest = _lineage_digest("observations", store.read(corpus_ref))
    restricted_truth_digest = _lineage_digest("truth", store.read(truth_ref))
    unsigned_envelope = {
        "campaign_count": len(ownership),
        "corpus_digest": frozen_corpus_digest(corpus),
        "family_campaign_counts": counts,
        "observation_digest": observation_digest,
        "observations": _reference_document(corpus_ref),
        "profile_sha256": hashlib.sha256(profile.to_json()).hexdigest(),
        "public_key_base64": signer.public_key_base64,
        "restricted_truth": _reference_document(truth_ref),
        "restricted_truth_digest": restricted_truth_digest,
        "run_ledger_sha256": ledger_ref.sha256,
        "schema_version": "1.0.0",
        "signer_key_id": signer.key_id,
    }
    envelope = CorpusEnvelope.model_validate(
        {
            **unsigned_envelope,
            "signature_base64": signer.sign(unsigned_envelope),
        }
    )
    envelope_ref = store.put_bytes(
        canonical_json_bytes(envelope.model_dump(mode="json")),
        _CORPUS_ENVELOPE_MEDIA_TYPE,
    )
    _publish_json_file(root, output_name, envelope.model_dump(mode="json"))
    return envelope_ref


def _load_corpus_envelope(
    store: ArtifactStore,
    digest: str,
    profile: CompetitionProfile,
    signer: RunSigningIdentity,
) -> tuple[CorpusEnvelope, FrozenCorpus]:
    try:
        envelope_ref = store.resolve(_digest(digest))
        if envelope_ref.media_type != _CORPUS_ENVELOPE_MEDIA_TYPE:
            raise CliContractError("development corpus envelope media type differs")
        payload = store.read(envelope_ref)
        envelope = CorpusEnvelope.model_validate(strict_json_loads(payload))
        observations = _load_observation_dataset(store=store, envelope=envelope)
        truth_ref = _artifact_ref(envelope.restricted_truth)
        truth_payload = store.read(truth_ref)
        truth_document = strict_json_loads(truth_payload)
        if type(truth_document) is not dict or set(truth_document) != {"manifest", "truth"}:
            raise CliContractError("restricted truth artifact fields differ")
        truth_rows = truth_document["truth"]
        if type(truth_rows) is not list:
            raise CliContractError("restricted truth rows are invalid")
        manifest = CorpusManifest.model_validate(truth_document["manifest"])
        truth = tuple(EvaluationTruthRow.model_validate(item) for item in truth_rows)
        if canonical_json_bytes(
            {
                "manifest": manifest.model_dump(mode="json"),
                "truth": [item.model_dump(mode="json") for item in truth],
            }
        ) != truth_payload:
            raise CliContractError("restricted truth artifact is not canonical")
        corpus = FrozenCorpus(observations=observations, truth=truth, manifest=manifest)
    except (ValidationError, WireContractError, ValueError) as error:
        raise CliContractError("development corpus envelope is invalid") from error
    if canonical_json_bytes(envelope.model_dump(mode="json")) != payload:
        raise CliContractError("development corpus envelope is not canonical")
    if envelope.profile_sha256 != hashlib.sha256(profile.to_json()).hexdigest():
        raise CliContractError("development corpus profile lineage differs")
    if (
        envelope.signer_key_id != signer.key_id
        or envelope.public_key_base64 != signer.public_key_base64
        or envelope.signature_base64 is None
        or not signer.verify(envelope.unsigned_document(), envelope.signature_base64)
    ):
        raise CliContractError("development corpus envelope signature differs")
    if envelope.corpus_digest != frozen_corpus_digest(corpus):
        raise CliContractError("development corpus content lineage differs")
    observations_payload = store.read(_artifact_ref(envelope.observations))
    if (
        envelope.observation_digest
        != _lineage_digest("observations", observations_payload)
        or envelope.restricted_truth_digest
        != _lineage_digest("truth", truth_payload)
    ):
        raise CliContractError("development corpus component digests differ")
    campaign_owners: dict[str, Family] = {}
    family_campaigns: dict[Family, set[str]] = {
        family: set() for family in profile.families
    }
    for row in truth:
        owner = campaign_owners.setdefault(row.campaign_id, row.family)
        if owner != row.family:
            raise CliContractError("development corpus campaign ownership differs")
        family_campaigns[row.family].add(row.campaign_id)
    counts = {family: len(family_campaigns[family]) for family in profile.families}
    if (
        envelope.campaign_count != len(campaign_owners)
        or envelope.family_campaign_counts != counts
        or counts != {family: profile.campaigns_per_family for family in profile.families}
        or manifest.observation_count != len(observations)
        or manifest.truth_count != len(truth)
        or len(manifest.run_ids) != profile.campaign_count
        or len(set(manifest.run_ids)) != len(manifest.run_ids)
    ):
        raise CliContractError("development corpus counts or run lineage differ")
    return envelope, corpus


def _load_observation_dataset(
    *, store: ArtifactStore, envelope: CorpusEnvelope
) -> tuple[ObservedEvent, ...]:
    """Open only the truth-blind observation artifact for feature consumers."""
    try:
        reference = _artifact_ref(envelope.observations)
        payload = store.read(reference)
        document = strict_json_loads(payload)
        if type(document) is not list:
            raise CliContractError("observation dataset must be a list")
        observations = tuple(ObservedEvent.model_validate(item) for item in document)
    except CliContractError:
        raise
    except (ValidationError, WireContractError, TypeError, ValueError) as error:
        raise CliContractError("observation dataset is invalid") from error
    if canonical_json_bytes(
        [item.model_dump(mode="json") for item in observations]
    ) != payload:
        raise CliContractError("observation dataset is not canonical")
    return observations


def _competition_split(profile: CompetitionProfile) -> SplitConfig:
    def cutoff_after(high_index: int) -> datetime:
        return profile.simulation_start_utc + timedelta(
            days=(high_index + 1) * profile.campaign_spacing_days
        ) - timedelta(microseconds=1)

    return SplitConfig(
        train_end=cutoff_after(profile.partition_campaign_indices["train"][1]),
        calibrator_fit_end=cutoff_after(
            profile.partition_campaign_indices["calibrator_fit"][1]
        ),
        threshold_end=cutoff_after(
            profile.partition_campaign_indices["threshold_selection"][1]
        ),
        development_end=cutoff_after(
            profile.partition_campaign_indices["development_test"][1]
        ),
    )


def _derive_training_exclusions(
    *, matrix: FeatureMatrix, training_ids: tuple[str, ...]
) -> tuple[str, ...]:
    """Return exact mandatory integrity/schema decisions excluded from model fitting."""
    event_by_id = {row.event_id: row for row in matrix.events}
    feature_by_id = {row.event_id: row for row in matrix.rows}
    rules = RuleEngine.default()
    try:
        return tuple(
            event_id
            for event_id in training_ids
            if any(
                hit.mandatory
                for hit in rules.evaluate(
                    event_by_id[event_id], feature_by_id[event_id]
                ).hits
            )
        )
    except KeyError as error:
        raise CliContractError("training exclusion rows are not covered by features") from error


def _rolling_campaign_folds(
    split: EvaluationSplit,
    *,
    campaign_start_times: dict[str, datetime],
) -> tuple[RollingFold, ...]:
    campaigns = split.campaigns["train"]
    row_campaigns = split.row_campaigns
    training_ids = split.training_row_ids
    if set(campaigns) != set(campaign_start_times):
        raise CliContractError("competition campaign start-time lineage differs")
    if len(campaigns) < 4:
        raise CliContractError("competition training partition needs rolling campaigns")
    cohorts: list[tuple[datetime, tuple[str, ...]]] = []
    for start in sorted(set(campaign_start_times.values())):
        cohort = tuple(
            sorted(
                campaign
                for campaign in campaigns
                if campaign_start_times[campaign] == start
            )
        )
        cohorts.append((start, cohort))
    if len(cohorts) < 3:
        raise CliContractError("competition rolling folds need three timestamp cohorts")
    first = max(1, len(cohorts) // 2)
    second = max(first + 1, (len(cohorts) * 3) // 4)
    second = min(second, len(cohorts) - 1)
    windows = ((first, second), (second, len(cohorts)))
    folds: list[RollingFold] = []
    for number, (boundary, validation_end) in enumerate(windows, start=1):
        fit_campaigns = {
            campaign for _, cohort in cohorts[:boundary] for campaign in cohort
        }
        validation_campaigns = {
            campaign
            for _, cohort in cohorts[boundary:validation_end]
            for campaign in cohort
        }
        fit_ids = tuple(item for item in training_ids if row_campaigns[item] in fit_campaigns)
        validation_ids = tuple(
            item for item in training_ids if row_campaigns[item] in validation_campaigns
        )
        labels = split.row_is_fraud
        if (
            not fit_ids
            or not validation_ids
            or {labels[item] for item in fit_ids} != {False, True}
            or {labels[item] for item in validation_ids} != {False, True}
        ):
            raise CliContractError("competition rolling fold is empty")
        folds.append(
            RollingFold(
                name=f"campaign-fold-{number}",
                fit_ids=fit_ids,
                validation_ids=validation_ids,
            )
        )
    return tuple(folds)


def _lineage_digest(label: str, payload: bytes) -> str:
    return hashlib.sha256(label.encode("ascii") + b"\x00" + payload).hexdigest()


def _freeze_competition_candidate(
    *,
    profile: CompetitionProfile,
    root: Path,
    corpus_envelope_digest: str,
    catalog_path: Path,
    rollback_ref: str,
    held_out_family: Family | None,
) -> ArtifactRef:
    store = ArtifactStore(root / "artifacts")
    signer = _load_standard_signer(root)
    envelope, corpus = _load_corpus_envelope(
        store, corpus_envelope_digest, profile, signer
    )
    if catalog_path != _CATALOG:
        raise CliContractError("competition catalog must be the committed catalog")
    catalog = load_feature_catalog(catalog_path)
    matrix = build_feature_matrix(corpus.observations, catalog)
    split_config = _competition_split(profile).model_copy(
        update={"held_out_family": held_out_family}
    )
    split = make_evaluation_split(corpus, split_config)
    labels = {
        item: int(split.row_is_fraud[item]) for item in split.training_row_ids
    }
    training_matrix = _matrix_subset(matrix, split.training_row_ids)
    mandatory_train_ids = _derive_training_exclusions(
        matrix=training_matrix,
        training_ids=split.training_row_ids,
    )
    event_by_id = {row.event_id: row for row in corpus.observations}
    campaign_start_times = {
        campaign: min(
            cast(datetime, event_by_id[row_id].decision_at)
            for row_id in split.training_row_ids
            if split.row_campaigns[row_id] == campaign
        )
        for campaign in split.campaigns["train"]
    }
    scorer = train_gbdt(
        matrix,
        labels,
        split.training_row_ids,
        _rolling_campaign_folds(
            split, campaign_start_times=campaign_start_times
        ),
        GbdtTrainingConfig(
            seed=profile.model_seed,
            depths=profile.gbdt.depths,
            learning_rates=profile.gbdt.learning_rates,
            l2_leaf_regs=profile.gbdt.l2_leaf_regs,
            iterations=profile.gbdt.iterations,
        ),
        training_cutoff=split.config.train_end,
        mandatory_row_ids=mandatory_train_ids,
    )
    fit_ids = split.row_ids["calibrator_fit"]
    threshold_ids = split.row_ids["threshold"]
    reload_ids = split.row_ids["development"]
    if not all((fit_ids, threshold_ids, reload_ids)):
        raise CliContractError("competition split contains an empty fitting or replay window")
    fit_matrix = _matrix_subset(matrix, fit_ids)
    threshold_matrix = _matrix_subset(matrix, threshold_ids)
    reload_matrix = _matrix_subset(matrix, reload_ids)
    fit_labels = np.asarray(
        [int(split.row_is_fraud[item]) for item in fit_ids], dtype=np.int64
    )
    threshold_labels = np.asarray(
        [int(split.row_is_fraud[item]) for item in threshold_ids], dtype=np.int64
    )
    calibrator = select_calibrator(
        scorer.predict(fit_matrix),
        fit_labels,
        scorer.predict(threshold_matrix),
        threshold_labels,
        min_class_count=profile.calibration.minimum_class_count,
    )
    rule_engine = RuleEngine.default()
    threshold_event_by_id = {item.event_id: item for item in threshold_matrix.events}
    threshold_row_by_id = {item.event_id: item for item in threshold_matrix.rows}
    mandatory_actions = np.asarray(
        [
            Action.DECLINE
            if any(
                hit.mandatory
                for hit in rule_engine.evaluate(
                    threshold_event_by_id[item], threshold_row_by_id[item]
                ).hits
            )
            else Action.APPROVE
            for item in threshold_ids
        ],
        dtype=object,
    )
    case_as_of = split.config.development_end + timedelta(
        days=profile.label_delay_days
    )
    case_observations = _case_observations(matrix.events, threshold_ids)
    callback_binding = bind_replay_case_counter(
        case_observations,
        threshold_ids,
        as_of=case_as_of,
    )
    callback = callback_binding.reconstruct(
        case_observations,
        threshold_ids,
        case_as_of,
    )
    budget = OperatingBudget(
        challenge_rate_max=profile.budgets.challenge_rate_max,
        false_decline_rate_max=profile.budgets.false_decline_rate_max,
        review_case_rate_max=profile.budgets.review_case_rate_max,
    )
    rule_scores = np.asarray(
        [
            rule_engine.evaluate(
                threshold_event_by_id[item], threshold_row_by_id[item]
            ).score
            for item in threshold_ids
        ],
        dtype=np.float64,
    )
    model_scores = calibrator.predict(scorer.predict(threshold_matrix))
    threshold_report = select_policy_thresholds(
        np.maximum(rule_scores, model_scores),
        threshold_labels,
        mandatory_actions,
        cast(Any, callback),
        budget,
        None,
    )
    if not threshold_report.feasible:
        raise CliContractError("competition threshold selection found no feasible operating point")
    observation_bytes = canonical_json_bytes(
        [item.model_dump(mode="json") for item in corpus.observations]
    )
    truth_bytes = canonical_json_bytes(
        [item.model_dump(mode="json") for item in corpus.truth]
    )
    lineage = BundleLineage(
        corpus_digest=frozen_corpus_digest(corpus),
        observation_dataset_digest=_lineage_digest("observations", observation_bytes),
        evaluator_truth_digest=_lineage_digest("truth", truth_bytes),
        split_manifest_digest=split.split_digest,
        feature_provenance_digest=_lineage_digest(
            "features", canonical_json_bytes(matrix.model_dump(mode="json"))
        ),
        hyperparameter_digest=hashlib.sha256(
            canonical_json_bytes(scorer.receipt.selected_params.model_dump(mode="json"))
        ).hexdigest(),
        reason_code_mapping_digest=_lineage_digest(
            "reasons", canonical_json_bytes(RuleManifest.default().model_dump(mode="json"))
        ),
    )
    source_paths = (
        "config/defense/competition-profile.json",
        "config/defense/feature-catalog.json",
        "src/apar/cases/grouping.py",
        "src/apar/defense/bundle.py",
        "src/apar/defense/calibration.py",
        "src/apar/defense/gbdt.py",
        "src/apar/defense/policy.py",
        "src/apar/defense/rules.py",
        "src/apar/defense/thresholds.py",
        "src/apar/defense/orchestration.py",
        "src/apar/evaluation/competition.py",
        "src/apar/evaluation/regimes.py",
        "src/apar/evaluation/replay.py",
        "src/apar/evaluation/splits.py",
        "src/apar/features/builders.py",
        "src/apar/features/catalog.py",
        "src/apar/features/state.py",
    )
    inventory = build_source_inventory(_REPOSITORY_ROOT, source_paths)
    checked_rollback = (
        GENESIS_ROLLBACK_REF
        if rollback_ref == GENESIS_ROLLBACK_REF
        else _digest(rollback_ref)
    )
    publisher = DefenderBundlePublisher(store, signer, _REPOSITORY_ROOT)
    try:
        _manifest, reference = publisher.freeze(
            scorer=scorer,
            catalog=catalog,
            split=split,
            training_matrix=training_matrix,
            mandatory_excluded_row_ids=mandatory_train_ids,
            calibration_fit_matrix=fit_matrix,
            calibration_fit_labels=fit_labels,
            calibration_selection_matrix=threshold_matrix,
            calibration_selection_labels=threshold_labels,
            threshold_matrix=threshold_matrix,
            threshold_labels=threshold_labels,
            threshold_mandatory_actions=mandatory_actions,
            threshold_values=None,
            review_case_counter=cast(Any, callback),
            rule_manifest=RuleManifest.default(),
            calibrator=calibrator,
            threshold_report=threshold_report,
            lineage=lineage,
            environment_lock=current_environment_lock(),
            source_inventory=inventory,
            reload_matrix=reload_matrix,
            bundle_id=str(
                uuid.uuid5(
                    uuid.NAMESPACE_URL,
                    "apar:defense:"
                    f"{envelope.corpus_digest}:{profile.model_seed}:"
                    f"{held_out_family or 'pooled'}",
                )
            ),
            frozen_at=split.config.development_end,
            rollback_ref=checked_rollback,
        )
        loaded = publisher.load(reference)
        loaded.verify_reload()
        return reference
    finally:
        publisher.close()


def _regular_file(path: Path, *, label: str, max_bytes: int) -> bytes:
    if not isinstance(path, Path):
        raise CliContractError(f"{label} path must be exact")
    try:
        info = path.lstat()
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
            raise CliContractError(f"{label} must be a regular non-symlink file")
        payload = path.read_bytes()
    except OSError as error:
        raise CliContractError(f"{label} is unavailable") from error
    if not 0 < len(payload) <= max_bytes:
        raise CliContractError(f"{label} exceeds its resource cap")
    return payload


def _secure_root(root: Path) -> Path:
    if not isinstance(root, Path) or not root.is_absolute():
        raise CliContractError("artifact root must be an absolute path")
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    try:
        resolved = root.resolve(strict=True)
        info = root.lstat()
    except OSError as error:
        raise CliContractError("artifact root is unavailable") from error
    if (
        resolved != root
        or stat.S_ISLNK(info.st_mode)
        or not stat.S_ISDIR(info.st_mode)
        or stat.S_IMODE(info.st_mode) & 0o077
    ):
        raise CliContractError("artifact root must be a pinned non-symlink directory")
    return root


def _digest(value: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or value != value.lower()
        or any(character not in _HEX for character in value)
    ):
        raise CliContractError("reference must be lowercase SHA-256")
    return value


def _reference_document(reference: ArtifactRef) -> dict[str, object]:
    return asdict(reference)


def _publish_json_file(root: Path, relative_name: str, document: object) -> None:
    if (
        type(relative_name) is not str
        or not relative_name
        or relative_name != Path(relative_name).name
        or relative_name in {".", ".."}
    ):
        raise CliContractError("output name must be one relative filename")
    payload = canonical_json_bytes(document)
    directory_fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    temporary_name = f".apar-{secrets.token_hex(16)}.tmp"
    try:
        descriptor = os.open(
            temporary_name,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
            0o600,
            dir_fd=directory_fd,
        )
        try:
            offset = 0
            while offset < len(payload):
                written = os.write(descriptor, payload[offset:])
                if written <= 0:
                    raise OSError("publication made no progress")
                offset += written
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        ArtifactStore.publish_no_replace_at(
            directory_fd, temporary_name, relative_name
        )
        os.fsync(directory_fd)
    except FileExistsError as error:
        raise CliContractError(
            "output already exists; immutable evidence is never overwritten"
        ) from error
    finally:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(temporary_name, dir_fd=directory_fd)
        os.close(directory_fd)


def _json_stdout(document: object) -> None:
    sys.stdout.buffer.write(canonical_json_bytes(document) + b"\n")
    sys.stdout.flush()


def _load_defender_ensemble(
    *,
    store: ArtifactStore,
    top_ref: ArtifactRef,
    profile: CompetitionProfile,
    signer: RunSigningIdentity,
) -> DefenderEnsembleEnvelope | None:
    if top_ref.media_type != _DEFENDER_ENSEMBLE_MEDIA_TYPE:
        return None
    payload = store.read(top_ref)
    try:
        document = strict_json_loads(payload)
    except WireContractError as error:
        raise CliContractError("frozen defender envelope is invalid") from error
    if type(document) is not dict or document.get("mode") != "competition_full":
        return None
    try:
        envelope = DefenderEnsembleEnvelope.model_validate(document)
    except (ValidationError, TypeError, ValueError) as error:
        raise CliContractError("frozen defender ensemble is invalid") from error
    if (
        canonical_json_bytes(envelope.model_dump(mode="json")) != payload
        or envelope.profile_sha256 != hashlib.sha256(profile.to_json()).hexdigest()
        or envelope.signer_key_id != signer.key_id
        or envelope.public_key_base64 != signer.public_key_base64
        or not signer.verify(envelope.unsigned_document(), envelope.signature_base64)
    ):
        raise CliContractError("frozen defender ensemble lineage differs")
    return envelope


def _parser(command: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog=f"{command}.py", allow_abbrev=False)
    if command == "generate_defense_runs":
        parser.add_argument("--profile", required=True)
        parser.add_argument("--root", required=True)
        parser.add_argument("--signer", required=True)
        parser.add_argument("--output-ledger", required=True)
    elif command == "build_defense_corpus":
        parser.add_argument("--profile", required=True)
        parser.add_argument("--run-manifests", required=True)
        parser.add_argument("--root", required=True)
        parser.add_argument("--output-manifest", required=True)
    elif command == "train_defender":
        parser.add_argument("--development-corpus", required=True)
        parser.add_argument("--catalog", required=True)
        parser.add_argument("--profile", required=True)
        parser.add_argument("--root", required=True)
        parser.add_argument("--rollback-ref", required=True)
    elif command == "evaluate_defender":
        parser.add_argument("--phase", choices=("development", "hidden"), required=True)
        parser.add_argument("--defender", required=True)
        parser.add_argument("--development-scorecard")
        parser.add_argument("--hidden-corpus")
        parser.add_argument("--profile", required=True)
        parser.add_argument("--root", required=True)
    else:
        raise ValueError("unknown defense command")
    return parser


def command_main(command: str, argv: list[str] | None = None) -> int:
    """Run one strict export command and keep diagnostics path-free."""
    parser = _parser(command)
    args = parser.parse_args(argv)
    try:
        profile = load_competition_profile(Path(args.profile), competition=True)
        root = _secure_root(Path(args.root))
        profile_digest = hashlib.sha256(profile.to_json()).hexdigest()
        if command == "generate_defense_runs":
            reference = _generate_competition_runs(
                profile=profile,
                root=root,
                signer_path=_signer_path(root, args.signer),
                output_name=args.output_ledger,
            )
            _json_stdout(
                {
                    "artifact": _reference_document(reference),
                    "campaign_count": 200,
                    "profile_sha256": profile_digest,
                    "schema_version": "1.0.0",
                }
            )
            return 0
        if command == "build_defense_corpus":
            reference = _build_competition_corpus(
                profile=profile,
                root=root,
                ledger_digest=args.run_manifests,
                output_name=args.output_manifest,
            )
            _json_stdout(
                {
                    "artifact": _reference_document(reference),
                    "profile_sha256": profile_digest,
                    "schema_version": "1.0.0",
                }
            )
            return 0
        if command == "train_defender":
            reference = _train_competition_defender(
                profile=profile,
                root=root,
                corpus_envelope_digest=args.development_corpus,
                catalog_path=Path(args.catalog),
                rollback_ref=args.rollback_ref,
            )
            _json_stdout(
                {
                    "frozen_defender": _reference_document(reference),
                    "profile_sha256": profile_digest,
                    "schema_version": "1.0.0",
                }
            )
            return 0
        if command == "evaluate_defender":
            defender_digest = _digest(args.defender)
            store = ArtifactStore(root / "artifacts")
            try:
                signer = _load_standard_signer(root)
                top_ref = store.resolve(defender_digest)
                ensemble = _load_defender_ensemble(
                    store=store,
                    top_ref=top_ref,
                    profile=profile,
                    signer=signer,
                )
                defender_ref = (
                    top_ref
                    if ensemble is None
                    else _artifact_ref(ensemble.pooled_ref)
                )
                publisher = DefenderBundlePublisher(store, signer, _REPOSITORY_ROOT)
                try:
                    loaded = publisher.load(defender_ref)
                    loaded.verify_reload()
                finally:
                    publisher.close()
            except (TypeError, ValueError) as error:
                raise CliContractError("frozen defender is unavailable or unverified") from error
            if args.phase == "hidden" and (
                args.development_scorecard is None or args.hidden_corpus is None
            ):
                raise CliContractError(
                    "hidden evaluation requires a completed development scorecard "
                    "receipt and hidden corpus"
                )
            if args.phase == "development" and (
                args.development_scorecard is not None or args.hidden_corpus is not None
            ):
                raise CliContractError(
                    "development evaluation cannot resolve hidden release inputs"
                )
            if args.phase == "hidden" and (
                args.development_scorecard is None or args.hidden_corpus is None
            ):
                raise CliContractError(
                    "hidden evaluation requires a completed development receipt and corpus"
                )
            evaluator_signer = _load_competition_evaluator_identity(root)
            hidden_signer: EvaluatorSigningIdentity | None = None
            hidden_context_signer: RunSigningIdentity | None = None
            development_evidence_ref: ArtifactRef | None = None
            if ensemble is None or ensemble.corpus_envelope_ref is None:
                raise CliContractError(
                    "competition evaluation requires the signed full defender ensemble"
                )
            corpus_envelope_ref = _artifact_ref(ensemble.corpus_envelope_ref)
            corpus_envelope, corpus = _load_corpus_envelope(
                store,
                corpus_envelope_ref.sha256,
                profile,
                signer,
            )
            split = make_evaluation_split(corpus, _competition_split(profile))
            from apar.evaluation.competition import (
                publish_competition_evaluation,
                seal_development_completion,
                verify_development_completion,
            )

            if args.phase == "hidden":
                try:
                    loaded_completion_ref = store.resolve(
                        _digest(args.development_scorecard)
                    )
                    completion = verify_development_completion(
                        store=store,
                        receipt_ref=loaded_completion_ref,
                        signer=signer,
                        ensemble_ref=top_ref,
                        profile_sha256=profile_digest,
                        corpus_envelope_ref=corpus_envelope_ref,
                        run_ledger_sha256=corpus_envelope.run_ledger_sha256,
                        evaluator_verifier=EvaluatorReplayVerifier.from_signer(
                            evaluator_signer
                        ),
                        pooled_ref=defender_ref,
                        held_family_refs={
                            family: _artifact_ref(ensemble.held_family_refs[family])
                            for family in profile.families
                        },
                        split=split,
                    )
                    scorecard_ref = _artifact_ref(completion.scorecard_ref)
                    evaluation_bundle_ref = _artifact_ref(
                        completion.evaluation_bundle_ref
                    )
                    verifier = PublicArtifactVerifier.from_signer(signer)
                    scorecard = DefenseScorecard.from_json(
                        store.read(scorecard_ref),
                        artifact_store=store,
                        verifier=verifier,
                    )
                    public_bundle = load_evaluation_bundle(
                        evaluation_bundle_ref,
                        artifact_store=store,
                        verifier=verifier,
                    )
                    if (
                        public_bundle.scorecard_sha256 != scorecard_ref.sha256
                        or scorecard.bundle_summary.bundle_id
                        != loaded.manifest.bundle_id
                    ):
                        raise CliContractError(
                            "development completion public lineage differs"
                        )
                    development_evidence_ref = _artifact_ref(
                        completion.development_evidence_ref
                    )
                except (ReportingContractError, TypeError, ValueError) as error:
                    raise CliContractError(
                        "hidden evaluation requires a completed development receipt"
                    ) from error
                hidden_signer, hidden_context_signer = (
                    _load_competition_hidden_identity(root, evaluator_signer)
                )
            hidden_context_ref = (
                None
                if args.phase == "development"
                else store.resolve(_digest(args.hidden_corpus))
            )
            published = publish_competition_evaluation(
                store=store,
                publication_signer=signer,
                evaluator_signer=evaluator_signer,
                hidden_signer=hidden_signer,
                pooled_ref=defender_ref,
                held_family_refs={
                    family: _artifact_ref(ensemble.held_family_refs[family])
                    for family in profile.families
                },
                corpus=corpus,
                split=split,
                profile_sha256=profile_digest,
                authenticated_run_ids=corpus.manifest.run_ids,
                hidden_context_ref=hidden_context_ref,
                hidden_context_signer=(
                    None
                    if args.phase == "development"
                    else hidden_context_signer
                ),
                development_evidence_ref=development_evidence_ref,
            )
            completion_ref = None
            if args.phase == "development":
                completion_ref = seal_development_completion(
                    store=store,
                    signer=signer,
                    ensemble_ref=top_ref,
                    profile_sha256=profile_digest,
                    corpus_envelope_ref=corpus_envelope_ref,
                    run_ledger_sha256=corpus_envelope.run_ledger_sha256,
                    scorecard_ref=published.scorecard_ref,
                    evaluation_bundle_ref=published.evaluation_bundle_ref,
                    development_evidence_ref=cast(
                        ArtifactRef, published.development_evidence_ref
                    ),
                    restricted_publication_receipt_ref=(
                        published.restricted_publication_receipt_ref
                    ),
                    promotion_envelope_digest=published.promotion_envelope_digest,
                    descriptor_scope=published.descriptor_scope,
                )
            _json_stdout(
                {
                    "development_completion": (
                        None
                        if completion_ref is None
                        else _reference_document(completion_ref)
                    ),
                    "defender": top_ref.sha256,
                    "evaluation_bundle": _reference_document(
                        published.evaluation_bundle_ref
                    ),
                    "phase": args.phase,
                    "profile_sha256": profile_digest,
                    "scorecard": _reference_document(published.scorecard_ref),
                    "schema_version": "1.0.0",
                    "status": published.champion_decision.status.value,
                }
            )
            return 0
        raise CliContractError("unknown defense command")
    except CliContractError as error:
        sys.stderr.write(f"ERROR: {error}\n")
        return 2
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        del error
        sys.stderr.write("ERROR: command failed closed contract verification\n")
        return 2


def script_main(command: str) -> Never:
    raise SystemExit(command_main(command))
