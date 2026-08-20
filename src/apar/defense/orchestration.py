"""Reproducible, synthetic-only orchestration for the Defend evidence commands.

The public commands accept only the immutable competition profile.  A deliberately
small fixture profile is available only through :func:`run_g3_fixture`; it cannot be
selected from any export command.
"""

from __future__ import annotations

import argparse
import base64
import binascii
import contextlib
import hashlib
import os
import secrets
import stat
import sys
import uuid
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path
from types import MappingProxyType
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
    DefenderBundleManifest,
    DefenderBundlePublisher,
    DefenderBundleReader,
    LoadedDefenderBundle,
    build_source_inventory,
    current_environment_lock,
    load_verified_defender_bundle,
)
from apar.defense.calibration import select_calibrator
from apar.defense.contracts import ObservedEvent
from apar.defense.gbdt import GbdtTrainingConfig, RollingFold, train_gbdt
from apar.defense.policy import OperatingBudget
from apar.defense.rules import RuleEngine, RuleManifest
from apar.defense.thresholds import ThresholdReport, select_policy_thresholds
from apar.evaluation.contracts import (
    CorpusManifest,
    CorpusProfile,
    EvaluationTruthRow,
    Family,
    FrozenCorpus,
)
from apar.evaluation.corpus import assemble_verified_corpus
from apar.evaluation.defender_attestation import DefenderBundleVerifier
from apar.evaluation.gates import (
    DefenseArm,
    EvaluatorReplayVerifier,
    EvaluatorSigningIdentity,
    HiddenPublicProof,
)
from apar.evaluation.hidden_source import (
    HIDDEN_SOURCE_RECEIPT_MEDIA_TYPE,
    HiddenSourceReceipt,
    HiddenSourceWorkerBinding,
    ordered_ids_digest,
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
    SignedRunReceipt,
    bind_scenario_for_run,
)
from apar.runs.wire import WireContractError, canonical_json_bytes, strict_json_loads
from apar.storage.artifacts import ArtifactRef, ArtifactStore

_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
_COMMITTED_PROFILE = _REPOSITORY_ROOT / "config" / "defense" / "competition-profile.json"
_COMMITTED_PREREGISTRATION = (
    _REPOSITORY_ROOT / "docs" / "experiments" / "defense-v1-preregistration.json"
)
_CATALOG = _REPOSITORY_ROOT / "config" / "defense" / "feature-catalog.json"
_CORPUS_ENVELOPE_MEDIA_TYPE = "application/vnd.apar.corpus-envelope+json"
_DEFENDER_ENSEMBLE_MEDIA_TYPE = "application/vnd.apar.defender-ensemble+json"
_DEFENDER_BUNDLE_MEDIA_TYPE = "application/vnd.apar.defender-bundle+json"
_INFEASIBLE_CANDIDATE_MEDIA_TYPE = (
    "application/vnd.apar.infeasible-defender-candidate+json"
)
_HIDDEN_CONTEXT_MEDIA_TYPE = "application/vnd.apar.hidden-evaluation-context+json"
_HIDDEN_CONTEXT_POINTER_MEDIA_TYPE = (
    "application/vnd.apar.hidden-context-pointer+json"
)
_HIDDEN_CONTEXT_POINTER_NAME = "hidden-context-pointer.json"
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
_DEFENSE_V1_PREREGISTRATION_SHA256 = (
    "95b460d2bb125e4fd3d432a6d52eb196a8c7e3d72bf5f0cf7d022cf2d9c8b428"
)
_DEFENSE_V1_PROFILE_SHA256 = (
    "f91c36e0329ef46631826a84d33b46282567069410cc9dc2c17694fe7463d7b1"
)
_DEFENSE_V1_PREREGISTRATION_MAX_BYTES = 512_000
_DEFENSE_V1_PUBLIC_REPORT_FILES = frozenset(
    {
        "calibration.csv",
        "data-card.md",
        "defense-scorecard.json",
        "defense-scorecard.md",
        "feature-manifest.json",
        "latency-evidence.json",
        "leaderboard.csv",
        "limitations.md",
        "model-card.md",
        "slice-metrics.csv",
        "thresholds.json",
        "value-workload.csv",
    }
)
_DEFENSE_V1_FIXTURE_FILES = frozenset(
    {
        "calibration.csv",
        "calibration.json",
        "corpus-manifest.json",
        "data-card.md",
        "defender-bundle.json",
        "defense-scorecard.json",
        "defense-scorecard.md",
        "evaluation-truth.parquet",
        "feature-manifest.json",
        "features.parquet",
        "latency-evidence.json",
        "leaderboard.csv",
        "limitations.md",
        "model-card.md",
        "model.cbm",
        "observations.parquet",
        "rules.json",
        "slice-metrics.csv",
        "split-manifest.json",
        "thresholds.json",
        "training-receipt.json",
        "value-workload.csv",
    }
)


class CliContractError(ValueError):
    """A CLI input or publication would violate the frozen evidence contract."""


@dataclass(frozen=True, slots=True)
class _ThresholdFailureArtifacts:
    """Public-safe trained artifacts retained when no operating point is feasible."""

    model: bytes
    training_receipt: bytes
    calibration: bytes
    threshold_report: ThresholdReport
    feature_manifest: bytes
    features: bytes
    split_projection: bytes
    rules: bytes


class _CompetitionThresholdInfeasible(CliContractError):
    """Carry exact public-safe failure artifacts across the CLI boundary."""

    def __init__(self, artifacts: _ThresholdFailureArtifacts) -> None:
        self.args = ("competition operating budget is infeasible",)
        self.artifacts = artifacts


@dataclass(frozen=True, slots=True)
class PreregisteredCampaign:
    """One exact public campaign slot frozen before result production."""

    family: Family
    campaign_index: int
    seed: int
    simulation_start_utc: datetime
    partition: str


@dataclass(frozen=True, slots=True)
class PreregisteredAuthorityIdentity:
    """One public Ed25519 trust root frozen before any competition run."""

    key_id: str
    public_key_base64: str


@dataclass(frozen=True, slots=True)
class DefenseV1Preregistration:
    """Pinned result-free Task 15 declaration admitted by competition exports."""

    preregistration_id: str
    profile: CompetitionProfile
    profile_sha256: str
    campaigns: tuple[PreregisteredCampaign, ...]
    authority_identities: Mapping[str, PreregisteredAuthorityIdentity]
    result_fields_forbidden: bool
    raw_sha256: str


class DefenseV1SignedAlias(ExternalContract):
    """Signed public name pointing only to one immutable Task 14 artifact."""

    schema_version: Literal["1.0.0"] = "1.0.0"
    kind: Literal[
        "run_ledger",
        "corpus_envelope",
        "defender_ensemble",
        "development_completion",
        "infeasible_candidate",
    ]
    artifact: dict[str, object]
    export_metadata: dict[str, object]
    campaign_count: int = Field(ge=0, le=200)
    family_counts: dict[Family, int]
    authenticated_run_ids: tuple[str, ...]
    preregistration_sha256: str
    profile_sha256: str
    signer_key_id: str
    public_key_base64: str
    signature_base64: str

    @model_validator(mode="after")
    def alias_is_closed(self) -> DefenseV1SignedAlias:
        reference = _artifact_ref(self.artifact)
        expected_media = {
            "run_ledger": "application/json",
            "corpus_envelope": _CORPUS_ENVELOPE_MEDIA_TYPE,
            "defender_ensemble": _DEFENDER_ENSEMBLE_MEDIA_TYPE,
            "development_completion": "application/vnd.apar.development-completion+json",
            "infeasible_candidate": _INFEASIBLE_CANDIDATE_MEDIA_TYPE,
        }[self.kind]
        if reference.media_type != expected_media:
            raise ValueError("defense-v1 alias artifact media type differs")
        expected_metadata = {
            "run_ledger": frozenset(),
            "corpus_envelope": frozenset({"observation_dataset", "truth_dataset"}),
            "defender_ensemble": frozenset(
                {
                    "held_family_refs",
                    "pooled_manifest",
                    "pooled_ref",
                    "portable_artifacts",
                    "split_projections",
                }
            ),
            "development_completion": frozenset(
                {"development_evidence_ref", "evaluation_bundle_ref", "scorecard_ref"}
            ),
            "infeasible_candidate": frozenset(
                {"failure_reason", "public_artifacts", "threshold_report_digest"}
            ),
        }[self.kind]
        if (
            set(self.export_metadata) != expected_metadata
            or len(canonical_json_bytes(self.export_metadata)) > 256 * 1024 * 1024
        ):
            raise ValueError("defense-v1 alias export metadata differs")
        for value in (
            self.preregistration_sha256,
            self.profile_sha256,
            self.signer_key_id,
        ):
            _digest(value)
        if self.preregistration_sha256 != _DEFENSE_V1_PREREGISTRATION_SHA256:
            raise ValueError("defense-v1 alias preregistration differs")
        if self.profile_sha256 != _DEFENSE_V1_PROFILE_SHA256:
            raise ValueError("defense-v1 alias profile differs")
        if self.kind == "run_ledger":
            if (
                self.campaign_count != 200
                or self.family_counts != {family: 50 for family in _FAMILIES}
                or len(self.authenticated_run_ids) != 200
                or len(set(self.authenticated_run_ids)) != 200
            ):
                raise ValueError("defense-v1 run-ledger alias is incomplete")
        elif (
            self.campaign_count != 200
            or self.family_counts != {family: 50 for family in _FAMILIES}
            or len(self.authenticated_run_ids) != 200
            or len(set(self.authenticated_run_ids)) != 200
        ):
            raise ValueError("defense-v1 artifact alias lineage is incomplete")
        return self

    def unsigned_document(self) -> dict[str, object]:
        return self.model_dump(mode="json", exclude={"signature_base64"})


@dataclass(frozen=True, slots=True)
class PortableDefenseV1Defender:
    """Public-only hydrated competition ensemble and its five runnable roles."""

    alias: DefenseV1SignedAlias
    ensemble: DefenderEnsembleEnvelope
    candidates: Mapping[str, LoadedDefenderBundle]


class PublicSplitProjection(ExternalContract):
    """Truth-free public partition projection bound to one signed private split."""

    schema_version: Literal["1.0.0"] = "1.0.0"
    classification: Literal["defender_visible_partition_projection"] = (
        "defender_visible_partition_projection"
    )
    config: SplitConfig
    partition_names: tuple[str, ...]
    campaigns: dict[str, tuple[str, ...]]
    row_ids: dict[str, tuple[str, ...]]
    training_row_ids: tuple[str, ...]
    entity_cohorts: dict[str, tuple[str, ...]]
    row_families: dict[str, Family]
    row_campaigns: dict[str, str]
    label_maturity_cutoff: datetime
    sample_counts: dict[str, int]
    held_out_family: Family | None
    held_out_evaluation_row_ids: tuple[str, ...]
    split_digest: str
    split_semantic_digest: str
    split_artifact_digest: str

    @field_validator(
        "split_digest", "split_semantic_digest", "split_artifact_digest"
    )
    @classmethod
    def split_digests_are_sha256(cls, value: str) -> str:
        return _digest(value)

    @model_validator(mode="after")
    def projection_is_closed(self) -> PublicSplitProjection:
        expected_partitions = ("train", "calibrator_fit", "threshold", "development")
        partition_rows = tuple(
            row_id for name in expected_partitions for row_id in self.row_ids.get(name, ())
        )
        partition_campaigns = tuple(
            campaign
            for name in expected_partitions
            for campaign in self.campaigns.get(name, ())
        )
        if (
            self.partition_names != expected_partitions
            or set(self.campaigns) != set(expected_partitions)
            or set(self.row_ids) != set(expected_partitions)
            or set(self.sample_counts) != set(expected_partitions)
            or any(
                self.sample_counts[name] != len(self.row_ids[name])
                for name in expected_partitions
            )
            or self.held_out_family != self.config.held_out_family
            or len(partition_rows) != len(set(partition_rows))
            or len(partition_campaigns) != len(set(partition_campaigns))
            or not set(self.training_row_ids).issubset(self.row_ids["train"])
            or any(
                row_id not in self.row_families
                or row_id not in self.row_campaigns
                or row_id not in self.entity_cohorts
                for row_id in partition_rows
            )
            or any(
                self.row_campaigns[row_id] not in self.campaigns[name]
                for name in expected_partitions
                for row_id in self.row_ids[name]
            )
            or self.held_out_evaluation_row_ids
            != (
                ()
                if self.held_out_family is None
                else tuple(
                    row_id
                    for row_id in self.row_ids["development"]
                    if self.row_families[row_id] == self.held_out_family
                )
            )
        ):
            raise ValueError("public split projection differs")
        return self


class HiddenReleaseAttestation(ExternalContract):
    """Aggregate-only authority signature proving the post-freeze release time."""

    schema_version: Literal["1.0.0"] = "1.0.0"
    ensemble_top_ref_digest: str
    pooled_defender_ref_digest: str
    candidate_roster_digest: str
    profile_sha256: str
    evaluation_bundle_digest: str
    scorecard_digest: str
    promotion_envelope_digest: str
    hidden_proof_digest: str
    authority_issued_at: str
    signer_key_id: str
    public_key_base64: str
    signature_base64: str

    @field_validator(
        "ensemble_top_ref_digest",
        "pooled_defender_ref_digest",
        "candidate_roster_digest",
        "profile_sha256",
        "evaluation_bundle_digest",
        "scorecard_digest",
        "promotion_envelope_digest",
        "hidden_proof_digest",
        "signer_key_id",
    )
    @classmethod
    def attestation_digests_are_sha256(cls, value: str) -> str:
        return _digest(value)

    @field_validator("authority_issued_at")
    @classmethod
    def issued_time_is_canonical_utc(cls, value: str) -> str:
        if type(value) is not str:
            raise ValueError("hidden release time differs")
        checked = validate_utc_timestamp(
            datetime.fromisoformat(value.replace("Z", "+00:00"))
        )
        if value != checked.isoformat().replace("+00:00", "Z"):
            raise ValueError("hidden release time differs")
        return value

    def unsigned_document(self) -> dict[str, object]:
        return self.model_dump(mode="json", exclude={"signature_base64"})


def load_defense_v1_preregistration(path: Path) -> DefenseV1Preregistration:
    """Load only the separately committed exact canonical defense-v1 declaration."""
    payload = _regular_file(
        path,
        label="defense-v1 preregistration",
        max_bytes=_DEFENSE_V1_PREREGISTRATION_MAX_BYTES,
    )
    if (
        not payload.endswith(b"\n")
        or payload.count(b"\n") != 1
        or hashlib.sha256(payload).hexdigest() != _DEFENSE_V1_PREREGISTRATION_SHA256
    ):
        raise CliContractError("defense-v1 preregistration bytes differ")
    try:
        document = strict_json_loads(payload[:-1])
        if type(document) is not dict:
            raise CliContractError("defense-v1 preregistration must be an object")
        campaigns_document = document["campaigns"]
        if type(campaigns_document) is not list:
            raise CliContractError("defense-v1 preregistration campaigns differ")
        campaigns = tuple(
            PreregisteredCampaign(
                family=cast(Family, item["family"]),
                campaign_index=cast(int, item["campaign_index"]),
                seed=cast(int, item["seed"]),
                simulation_start_utc=validate_utc_timestamp(
                    datetime.fromisoformat(
                        cast(str, item["simulation_start_utc"]).replace("Z", "+00:00")
                    )
                ),
                partition=cast(str, item["partition"]),
            )
            for item in campaigns_document
            if type(item) is dict
        )
        profile = CompetitionProfile.model_validate(document["profile"])
        raw_authorities = document["authority_identities"]
        if type(raw_authorities) is not dict or set(raw_authorities) != {
            "publication",
            "development_evaluator",
            "hidden_evaluator",
            "hidden_source",
        }:
            raise CliContractError("defense-v1 authority identities differ")
        authorities: dict[str, PreregisteredAuthorityIdentity] = {}
        for role, raw_identity in raw_authorities.items():
            if type(raw_identity) is not dict or set(raw_identity) != {
                "key_id",
                "public_key_base64",
            }:
                raise CliContractError("defense-v1 authority identity fields differ")
            key_id = _digest(cast(str, raw_identity["key_id"]))
            public_key_base64 = cast(str, raw_identity["public_key_base64"])
            public_key = base64.b64decode(public_key_base64, validate=True)
            if len(public_key) != 32 or hashlib.sha256(public_key).hexdigest() != key_id:
                raise CliContractError("defense-v1 authority identity is inconsistent")
            authorities[role] = PreregisteredAuthorityIdentity(
                key_id=key_id,
                public_key_base64=public_key_base64,
            )
        if len({item.key_id for item in authorities.values()}) != 4:
            raise CliContractError("defense-v1 authority identities must be distinct")
    except (KeyError, TypeError, ValueError, ValidationError, WireContractError) as error:
        raise CliContractError("defense-v1 preregistration is invalid") from error
    expected = tuple(
        (family, index)
        for family in profile.families
        for index in range(profile.campaigns_per_family)
    )

    def partition_for(index: int) -> str:
        matches = tuple(
            name
            for name, bounds in profile.partition_campaign_indices.items()
            if bounds[0] <= index <= bounds[1]
        )
        if len(matches) != 1:
            raise CliContractError("defense-v1 preregistration partition differs")
        return matches[0]

    if (
        len(campaigns) != 200
        or tuple((item.family, item.campaign_index) for item in campaigns) != expected
        or any(
            item.seed != profile.campaign_seed(item.family, item.campaign_index)
            or item.simulation_start_utc
            != profile.campaign_start(item.family, item.campaign_index)
            or item.partition != partition_for(item.campaign_index)
            for item in campaigns
        )
        or document.get("preregistration_id") != "defense-v1"
        or document.get("profile_sha256") != _DEFENSE_V1_PROFILE_SHA256
        or document.get("result_fields_forbidden") is not True
        or hashlib.sha256(profile.to_json()).hexdigest() != _DEFENSE_V1_PROFILE_SHA256
    ):
        raise CliContractError("defense-v1 preregistration lineage differs")
    return DefenseV1Preregistration(
        preregistration_id="defense-v1",
        profile=profile,
        profile_sha256=_DEFENSE_V1_PROFILE_SHA256,
        campaigns=campaigns,
        authority_identities=MappingProxyType(authorities),
        result_fields_forbidden=True,
        raw_sha256=_DEFENSE_V1_PREREGISTRATION_SHA256,
    )


def _preregistered_authority(role: str) -> PreregisteredAuthorityIdentity:
    preregistration = load_defense_v1_preregistration(_COMMITTED_PREREGISTRATION)
    try:
        identity = preregistration.authority_identities[role]
    except KeyError as error:
        raise CliContractError("defense-v1 authority role differs") from error
    if type(identity) is not PreregisteredAuthorityIdentity:
        raise CliContractError("defense-v1 authority identity type differs")
    return identity


def _assert_preregistered_authority(
    role: str, *, key_id: str, public_key_base64: str
) -> None:
    expected = _preregistered_authority(role)
    if expected.key_id != key_id or expected.public_key_base64 != public_key_base64:
        raise CliContractError(f"defense-v1 {role} authority differs from preregistration")


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
    hidden_source_signer_key_id: str
    hidden_source_public_key_base64: str
    corpus_envelope_ref: dict[str, object] | None = None
    signer_key_id: str
    public_key_base64: str
    signature_base64: str

    @model_validator(mode="after")
    def ensemble_is_complete(self) -> DefenderEnsembleEnvelope:
        _digest(self.profile_sha256)
        _digest(self.signer_key_id)
        _digest(self.hidden_source_signer_key_id)
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
        try:
            source_public_key = base64.b64decode(
                self.hidden_source_public_key_base64, validate=True
            )
        except (TypeError, ValueError, binascii.Error) as error:
            raise ValueError("hidden source public identity is invalid") from error
        if (
            len(source_public_key) != 32
            or hashlib.sha256(source_public_key).hexdigest()
            != self.hidden_source_signer_key_id
            or self.hidden_source_signer_key_id == self.signer_key_id
        ):
            raise ValueError("hidden source public identity differs")
        return self

    def unsigned_document(self) -> dict[str, object]:
        return self.model_dump(mode="json", exclude={"signature_base64"})


class HiddenContextPointer(ExternalContract):
    """Immutable operator pointer closing the one-shot hidden source selection."""

    schema_version: Literal["1.0.0"] = "1.0.0"
    kind: Literal["competition_hidden_context_pointer"] = (
        "competition_hidden_context_pointer"
    )
    profile_sha256: str
    ensemble_ref: dict[str, object]
    development_corpus_ref: dict[str, object]
    hidden_context_ref: dict[str, object]
    source_receipt_ref: dict[str, object]
    signer_key_id: str
    public_key_base64: str
    signature_base64: str

    @model_validator(mode="after")
    def pointer_is_closed(self) -> HiddenContextPointer:
        _digest(self.profile_sha256)
        _digest(self.signer_key_id)
        ensemble = _artifact_ref(self.ensemble_ref)
        development = _artifact_ref(self.development_corpus_ref)
        hidden = _artifact_ref(self.hidden_context_ref)
        source = _artifact_ref(self.source_receipt_ref)
        if (
            ensemble.media_type != _DEFENDER_ENSEMBLE_MEDIA_TYPE
            or development.media_type != _CORPUS_ENVELOPE_MEDIA_TYPE
            or hidden.media_type
            != "application/vnd.apar.restricted-hidden-context-envelope+json"
            or source.media_type != HIDDEN_SOURCE_RECEIPT_MEDIA_TYPE
        ):
            raise ValueError("hidden context pointer media types differ")
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
    hidden_source_signer_key_id: str | None = None,
    hidden_source_public_key_base64: str | None = None,
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
        or type(hidden_source_signer_key_id) is not str
        or type(hidden_source_public_key_base64) is not str
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
        "hidden_source_public_key_base64": hidden_source_public_key_base64,
        "hidden_source_signer_key_id": hidden_source_signer_key_id,
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
    enforce_preregistered_authorities: bool = False,
) -> ArtifactRef:
    """Freeze pooled and all four true LOFO bundles before publishing one roster."""
    hidden_source_key_id, hidden_source_public_key = (
        _load_competition_hidden_run_public_identity(root)
    )
    signer = _load_standard_signer(root)
    if enforce_preregistered_authorities:
        _assert_preregistered_authority(
            "publication",
            key_id=signer.key_id,
            public_key_base64=signer.public_key_base64,
        )
        _assert_preregistered_authority(
            "hidden_source",
            key_id=hidden_source_key_id,
            public_key_base64=hidden_source_public_key,
        )
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
    store = ArtifactStore(root / "artifacts")
    verifier = DefenderBundleVerifier(
        store,
        signer_key_id=signer.key_id,
        public_key_base64=signer.public_key_base64,
    )
    for reference in (pooled, *(held[family] for family in profile.families)):
        if not verifier.attest(reference).rollback_available:
            raise CliContractError("competition defender rollback is unavailable")
    envelope = _build_defender_ensemble(
        profile=profile,
        pooled_ref=pooled,
        held_family_refs=held,
        signer=signer,
        corpus_envelope_ref=store.resolve(_digest(corpus_envelope_digest)),
        hidden_source_signer_key_id=hidden_source_key_id,
        hidden_source_public_key_base64=hidden_source_public_key,
    )
    if type(envelope) is not DefenderEnsembleEnvelope:
        raise CliContractError("competition defender ensemble type differs")
    return store.put_bytes(
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
        (
            event
            if event.is_decision_point or event.decision_at is None
            else event.model_copy(update={"decision_at": None})
        )
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


def resolve_defense_v1_alias(
    path: Path,
    *,
    expected_kind: Literal[
        "run_ledger", "corpus_envelope", "defender_ensemble", "development_completion"
    ],
    expected_profile_sha256: str,
    signer: RunSigningIdentity,
) -> ArtifactRef:
    """Authenticate one named Task 15 pointer without trusting its filesystem name."""
    alias = _load_defense_v1_alias(
        path,
        expected_kind=expected_kind,
        expected_profile_sha256=expected_profile_sha256,
        signer=signer,
    )
    return _artifact_ref(alias.artifact)


def _load_defense_v1_alias(
    path: Path,
    *,
    expected_kind: Literal[
        "run_ledger", "corpus_envelope", "defender_ensemble", "development_completion"
    ],
    expected_profile_sha256: str,
    signer: RunSigningIdentity,
) -> DefenseV1SignedAlias:
    if type(signer) is not RunSigningIdentity:
        raise CliContractError("defense-v1 alias requires the pinned signer")
    _digest(expected_profile_sha256)
    payload = _regular_file(path, label="defense-v1 alias", max_bytes=300 * 1024 * 1024)
    try:
        document = strict_json_loads(payload)
        if type(document) is not dict:
            raise CliContractError("defense-v1 alias must be an object")
        tuple_fields = document.get("authenticated_run_ids")
        if type(tuple_fields) is not list:
            raise CliContractError("defense-v1 alias run lineage differs")
        document["authenticated_run_ids"] = tuple(tuple_fields)
        alias = DefenseV1SignedAlias.model_validate(document)
    except (ValidationError, ValueError, WireContractError) as error:
        raise CliContractError("defense-v1 alias is invalid") from error
    if (
        canonical_json_bytes(alias.model_dump(mode="json")) != payload
        or alias.kind != expected_kind
        or alias.profile_sha256 != expected_profile_sha256
        or alias.signer_key_id != signer.key_id
        or alias.public_key_base64 != signer.public_key_base64
        or not signer.verify(alias.unsigned_document(), alias.signature_base64)
    ):
        raise CliContractError("defense-v1 alias signature or lineage differs")
    return alias


def _load_defense_v1_alias_public(
    path: Path,
    *,
    expected_kind: Literal[
        "run_ledger", "corpus_envelope", "defender_ensemble", "development_completion"
    ],
    expected_profile_sha256: str,
    signer_key_id: str,
    public_key_base64: str,
) -> DefenseV1SignedAlias:
    """Authenticate a committed alias using only its separately pinned public key."""
    try:
        verifier = PublicArtifactVerifier(
            signer_key_id=signer_key_id,
            public_key_base64=public_key_base64,
        )
        payload = _regular_file(
            path, label="defense-v1 alias", max_bytes=300 * 1024 * 1024
        )
        document = strict_json_loads(payload)
        if type(document) is not dict or type(document.get("authenticated_run_ids")) is not list:
            raise CliContractError("defense-v1 alias run lineage differs")
        document["authenticated_run_ids"] = tuple(document["authenticated_run_ids"])
        alias = DefenseV1SignedAlias.model_validate(document)
    except (ReportingContractError, ValidationError, ValueError, WireContractError) as error:
        raise CliContractError("defense-v1 alias is invalid") from error
    if (
        canonical_json_bytes(alias.model_dump(mode="json")) != payload
        or alias.kind != expected_kind
        or alias.profile_sha256 != expected_profile_sha256
        or alias.signer_key_id != verifier.key_id
        or alias.public_key_base64 != verifier.public_key_base64
        or not verifier.verify(alias.unsigned_document(), alias.signature_base64)
    ):
        raise CliContractError("defense-v1 alias signature or lineage differs")
    return alias


def _decode_portable_artifacts(
    value: object,
) -> dict[str, tuple[ArtifactRef, bytes]]:
    if type(value) is not dict or not value or len(value) > 256:
        raise CliContractError("portable defender artifact set differs")
    decoded: dict[str, tuple[ArtifactRef, bytes]] = {}
    allowed_media = {
        _DEFENDER_ENSEMBLE_MEDIA_TYPE,
        _DEFENDER_BUNDLE_MEDIA_TYPE,
        "application/vnd.apar.catboost-model",
        "application/vnd.apache.parquet",
        "application/vnd.apar.feature-catalog+json",
        "application/vnd.apar.rule-manifest+json",
        "application/vnd.apar.training-receipt+json",
        "application/vnd.apar.calibration+json",
        "application/vnd.apar.threshold-report+json",
        "application/vnd.apar.environment-lock+json",
        "application/vnd.apar.source-inventory+json",
        "application/vnd.apar.reload-fixture+json",
        "application/vnd.apar.training-binding+json",
        "application/vnd.apar.calibration-binding+json",
        "application/vnd.apar.threshold-binding+json",
    }
    total_size = 0
    for digest, raw in value.items():
        checked_digest = _digest(digest)
        if type(raw) is not dict or set(raw) != {
            "media_type",
            "payload_base64",
            "size_bytes",
        }:
            raise CliContractError("portable defender record fields differ")
        media_type = raw["media_type"]
        encoded = raw["payload_base64"]
        size_bytes = raw["size_bytes"]
        if (
            type(media_type) is not str
            or media_type not in allowed_media
            or type(encoded) is not str
            or type(size_bytes) is not int
            or size_bytes <= 0
            or size_bytes > 128 * 1024 * 1024
        ):
            raise CliContractError("portable defender record is invalid")
        try:
            payload = base64.b64decode(encoded, validate=True)
        except (ValueError, binascii.Error) as error:
            raise CliContractError("portable defender payload is invalid") from error
        total_size += len(payload)
        if (
            len(payload) != size_bytes
            or hashlib.sha256(payload).hexdigest() != checked_digest
            or total_size > 192 * 1024 * 1024
        ):
            raise CliContractError("portable defender payload lineage differs")
        decoded[checked_digest] = (
            ArtifactRef(
                checked_digest,
                media_type,
                size_bytes,
                f"{checked_digest}/payload",
            ),
            payload,
        )
    return decoded


def hydrate_defense_v1_defender(
    path: Path,
    *,
    store: ArtifactStore,
    source_root: Path,
    signer_key_id: str,
    public_key_base64: str,
    expected_profile_sha256: str = _DEFENSE_V1_PROFILE_SHA256,
) -> PortableDefenseV1Defender:
    """Hydrate and execute the self-contained five-role bundle without private state."""
    if type(store) is not ArtifactStore or not isinstance(source_root, Path):
        raise CliContractError("portable defender hydration inputs differ")
    alias = _load_defense_v1_alias_public(
        path,
        expected_kind="defender_ensemble",
        expected_profile_sha256=expected_profile_sha256,
        signer_key_id=signer_key_id,
        public_key_base64=public_key_base64,
    )
    decoded = _decode_portable_artifacts(alias.export_metadata["portable_artifacts"])
    raw_split_projections = alias.export_metadata["split_projections"]
    if type(raw_split_projections) is not dict or not raw_split_projections:
        raise CliContractError("portable split projection set differs")
    try:
        split_projections = {
            _digest(digest): PublicSplitProjection.model_validate(document)
            for digest, document in raw_split_projections.items()
        }
    except (TypeError, ValueError, ValidationError) as error:
        raise CliContractError("portable split projection is invalid") from error
    for declared, payload in decoded.values():
        stored = store.put_bytes(payload, declared.media_type)
        if stored != declared:
            raise CliContractError("portable defender store hydration differs")
    top_ref = _artifact_ref(alias.artifact)
    if top_ref.sha256 not in decoded or decoded[top_ref.sha256][0] != top_ref:
        raise CliContractError("portable defender top reference is absent")
    try:
        envelope_payload = store.read(top_ref)
        envelope = DefenderEnsembleEnvelope.model_validate(
            strict_json_loads(envelope_payload)
        )
        verifier = PublicArtifactVerifier(
            signer_key_id=signer_key_id,
            public_key_base64=public_key_base64,
        )
    except (ReportingContractError, ValidationError, ValueError, WireContractError) as error:
        raise CliContractError("portable defender ensemble is invalid") from error
    if (
        canonical_json_bytes(envelope.model_dump(mode="json")) != envelope_payload
        or envelope.profile_sha256 != expected_profile_sha256
        or envelope.signer_key_id != verifier.key_id
        or envelope.public_key_base64 != verifier.public_key_base64
        or not verifier.verify(envelope.unsigned_document(), envelope.signature_base64)
        or alias.export_metadata["pooled_ref"] != envelope.pooled_ref
        or alias.export_metadata["held_family_refs"] != envelope.held_family_refs
    ):
        raise CliContractError("portable defender ensemble lineage differs")
    role_refs: dict[str, ArtifactRef] = {
        "pooled": _artifact_ref(envelope.pooled_ref)
    }
    role_refs.update(
        {
            family: _artifact_ref(envelope.held_family_refs[family])
            for family in _FAMILIES
        }
    )
    reachable = {top_ref.sha256}
    reachable_bundle_refs: set[str] = set()
    loaded_roles: dict[str, LoadedDefenderBundle] = {}
    reader = DefenderBundleReader(
        store,
        signer_key_id=signer_key_id,
        public_key_base64=public_key_base64,
        source_root=source_root,
    )
    for role, reference in role_refs.items():
        loaded = load_verified_defender_bundle(reader, reference)
        if loaded.manifest.rollback_ref == GENESIS_ROLLBACK_REF:
            raise CliContractError("portable defender rollback is unavailable")
        candidate_split = split_projections.get(reference.sha256)
        if (
            candidate_split is None
            or candidate_split.split_artifact_digest
            != loaded.manifest.split_artifact_digest
            or candidate_split.split_digest != loaded.manifest.split_manifest_digest
            or candidate_split.split_semantic_digest
            != loaded.training_binding.split_semantic_digest
        ):
            raise CliContractError("portable defender split binding differs")
        expected_held_family: Family | None = (
            None if role == "pooled" else cast(Family, role)
        )
        if candidate_split.config.held_out_family != expected_held_family:
            raise CliContractError("portable defender role split differs")
        loaded_roles[role] = loaded
        current_ref = reference
        current = loaded
        while True:
            reachable.add(current_ref.sha256)
            reachable_bundle_refs.add(current_ref.sha256)
            reachable.update(
                component.sha256
                for component in current.manifest.components
                if component.name != "split"
            )
            current_split = split_projections.get(current_ref.sha256)
            if (
                current_split is None
                or current_split.split_artifact_digest
                != current.manifest.split_artifact_digest
                or current_split.split_digest != current.manifest.split_manifest_digest
                or current_split.split_semantic_digest
                != current.training_binding.split_semantic_digest
            ):
                raise CliContractError("portable rollback split binding differs")
            if current.manifest.rollback_ref == GENESIS_ROLLBACK_REF:
                break
            predecessor_ref = store.resolve(current.manifest.rollback_ref)
            predecessor = load_verified_defender_bundle(reader, predecessor_ref)
            final_contract = current.manifest.model_dump(
                mode="json",
                exclude={
                    "bundle_id",
                    "frozen_at",
                    "rollback_ref",
                    "rollback_size_bytes",
                    "signature_base64",
                },
            )
            predecessor_contract = predecessor.manifest.model_dump(
                mode="json",
                exclude={
                    "bundle_id",
                    "frozen_at",
                    "rollback_ref",
                    "rollback_size_bytes",
                    "signature_base64",
                },
            )
            if final_contract != predecessor_contract:
                raise CliContractError("portable rollback contract differs")
            current_ref = predecessor_ref
            current = predecessor
    if set(decoded) != reachable:
        raise CliContractError("portable defender records are missing or unreachable")
    if set(split_projections) != reachable_bundle_refs:
        raise CliContractError("portable split projection records differ")
    pooled_manifest = cast(dict[str, object], alias.export_metadata["pooled_manifest"])
    if pooled_manifest != loaded_roles["pooled"].manifest.model_dump(mode="json"):
        raise CliContractError("portable pooled manifest differs")
    return PortableDefenseV1Defender(
        alias=alias,
        ensemble=envelope,
        candidates=MappingProxyType(loaded_roles),
    )


def _signer_path(root: Path, raw: str) -> Path:
    candidate = Path(raw)
    expected = root / "run-signing.key"
    if not candidate.is_absolute() or candidate != expected:
        raise CliContractError("signer must be the pinned root signing identity")
    return candidate


def _load_standard_signer(root: Path) -> RunSigningIdentity:
    path = root / "run-signing.key"
    return _load_existing_signer(path, root=root)


def _read_pinned_identity_bytes(root: Path, filename: str, *, label: str) -> bytes:
    """Read one exact private/public authority through pinned directory descriptors."""
    if (
        not isinstance(root, Path)
        or not root.is_absolute()
        or root.resolve() != root
        or filename != Path(filename).name
    ):
        raise CliContractError(f"{label} root differs")
    directory_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW
    file_flags = os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW
    parent_fd: int | None = None
    root_fd: int | None = None
    identity_fd: int | None = None
    try:
        parent_fd = os.open(root.parent, directory_flags)
        parent_info = os.fstat(parent_fd)
        root_fd = os.open(root.name, directory_flags, dir_fd=parent_fd)
        root_info = os.fstat(root_fd)
        identity_fd = os.open(filename, file_flags, dir_fd=root_fd)
        before = os.fstat(identity_fd)
        if (
            not stat.S_ISDIR(parent_info.st_mode)
            or not stat.S_ISDIR(root_info.st_mode)
            or stat.S_IMODE(root_info.st_mode) != 0o700
            or root_info.st_uid != os.geteuid()
            or not stat.S_ISREG(before.st_mode)
            or stat.S_IMODE(before.st_mode) != 0o600
            or before.st_uid != os.geteuid()
            or before.st_nlink != 1
            or before.st_size != 32
        ):
            raise CliContractError(f"{label} must be an owner-only regular file")
        chunks: list[bytes] = []
        remaining = 33
        while remaining:
            chunk = os.read(identity_fd, remaining)
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        raw = b"".join(chunks)
        after = os.fstat(identity_fd)
        named_file = os.stat(filename, dir_fd=root_fd, follow_symlinks=False)
        named_root = os.stat(root.name, dir_fd=parent_fd, follow_symlinks=False)
        final_parent = os.stat(root.parent, follow_symlinks=False)
        if (
            len(raw) != 32
            or (
                before.st_dev,
                before.st_ino,
                before.st_size,
                before.st_mtime_ns,
                before.st_ctime_ns,
                before.st_mode,
                before.st_uid,
                before.st_nlink,
            )
            != (
                after.st_dev,
                after.st_ino,
                after.st_size,
                after.st_mtime_ns,
                after.st_ctime_ns,
                after.st_mode,
                after.st_uid,
                after.st_nlink,
            )
            or (before.st_dev, before.st_ino)
            != (named_file.st_dev, named_file.st_ino)
            or (root_info.st_dev, root_info.st_ino)
            != (named_root.st_dev, named_root.st_ino)
            or (parent_info.st_dev, parent_info.st_ino)
            != (final_parent.st_dev, final_parent.st_ino)
        ):
            raise CliContractError(f"{label} changed during read")
        return raw
    except CliContractError:
        raise
    except OSError as error:
        raise CliContractError(f"{label} cannot be read") from error
    finally:
        for descriptor in (identity_fd, root_fd, parent_fd):
            if descriptor is not None:
                os.close(descriptor)


def _load_evaluator_seed(root: Path, filename: str) -> bytes:
    return _read_pinned_identity_bytes(
        root, filename, label="competition evaluator signing identity"
    )


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
    try:
        evaluator = EvaluatorSigningIdentity.from_private_bytes(seed)
    except (TypeError, ValueError) as error:
        raise CliContractError("competition evaluator signing identity is invalid") from error
    if evaluator.key_id == _load_standard_signer(root).key_id:
        raise CliContractError(
            "competition publication and evaluator identities must be distinct"
        )
    return evaluator


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
    try:
        hidden = EvaluatorSigningIdentity.from_private_bytes(seed)
        hidden_context = RunSigningIdentity.from_private_bytes(seed)
    except (TypeError, ValueError) as error:
        raise CliContractError("competition hidden signing identity is invalid") from error
    if hidden.key_id in {_load_standard_signer(root).key_id, evaluator.key_id}:
        raise CliContractError(
            "competition publication, evaluator, and hidden identities must be distinct"
        )
    return hidden, hidden_context


def _load_competition_hidden_run_identity(
    root: Path,
    evaluator: EvaluatorSigningIdentity,
    hidden: EvaluatorSigningIdentity,
) -> RunSigningIdentity:
    """Load the independent hidden-run source authority from its fourth pinned key."""
    pinned_key_id, pinned_public_key = _load_competition_hidden_run_public_identity(
        root
    )
    seed = _load_evaluator_seed(root, "hidden-run-signing.key")
    fixture_seeds = {
        _FIXTURE_SIGNER_SEED,
        hashlib.sha256(b"apar-g3-development-evaluator-v1").digest(),
        hashlib.sha256(b"apar-g3-hidden-evaluator-v1").digest(),
    }
    if seed in fixture_seeds:
        raise CliContractError(
            "competition hidden-run identity cannot use a deterministic fixture key"
        )
    try:
        source = RunSigningIdentity.from_private_bytes(seed)
    except (TypeError, ValueError) as error:
        raise CliContractError("competition hidden-run identity is invalid") from error
    if (
        source.key_id
        in {_load_standard_signer(root).key_id, evaluator.key_id, hidden.key_id}
        or source.key_id != pinned_key_id
        or source.public_key_base64 != pinned_public_key
    ):
        raise CliContractError(
            "competition hidden-run identity must be distinct from all authorities"
        )
    return source


def _load_competition_hidden_run_public_identity(root: Path) -> tuple[str, str]:
    """Load the pre-provisioned public source trust root without private access."""
    public_key = _read_pinned_identity_bytes(
        root, "hidden-run-signing.pub", label="hidden-run public identity"
    )
    return (
        hashlib.sha256(public_key).hexdigest(),
        base64.b64encode(public_key).decode("ascii"),
    )


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
    raw = _read_pinned_identity_bytes(
        root, "run-signing.key", label="pinned run signing identity"
    )
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
    enforce_preregistered_authorities: bool = False,
) -> ArtifactRef:
    store = ArtifactStore(root / "artifacts")
    signer = _load_existing_signer(signer_path, root=root)
    if enforce_preregistered_authorities:
        _assert_preregistered_authority(
            "publication",
            key_id=signer.key_id,
            public_key_base64=signer.public_key_base64,
        )
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
    enforce_preregistered_authorities: bool = False,
) -> ArtifactRef:
    store = ArtifactStore(root / "artifacts")
    signer = _load_standard_signer(root)
    if enforce_preregistered_authorities:
        _assert_preregistered_authority(
            "publication",
            key_id=signer.key_id,
            public_key_base64=signer.public_key_base64,
        )
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


@dataclass(frozen=True, slots=True)
class _AuthenticatedHiddenSource:
    corpus: FrozenCorpus
    manifests: tuple[RunManifest, ...]
    manifest_refs: tuple[ArtifactRef, ...]
    families: tuple[Family, ...]


def _require_independent_hidden_seeds(
    profile: CompetitionProfile, seeds: tuple[int, ...]
) -> None:
    preregistered_seeds = {
        profile.campaign_seed(family, index)
        for family in _FAMILIES
        for index in range(profile.campaigns_per_family)
    }
    if (
        type(seeds) is not tuple
        or len(seeds) != 4
        or len(set(seeds)) != 4
        or set(seeds).intersection(preregistered_seeds)
    ):
        raise CliContractError("hidden source seeds are not independent")


def _assemble_authenticated_hidden_source(
    *,
    store: ArtifactStore,
    runner: RunRunner,
    signer: RunSigningIdentity,
    profile: CompetitionProfile,
    manifest_refs: tuple[ArtifactRef, ...],
    development_run_ids: tuple[str, ...],
    development_event_ids: tuple[str, ...],
    development_payment_ids: tuple[str, ...],
    development_campaign_ids: tuple[str, ...],
    minimum_simulation_start: datetime,
) -> _AuthenticatedHiddenSource:
    """Reauthenticate exact independent run refs before any hidden corpus exists."""
    if (
        type(store) is not ArtifactStore
        or type(runner) is not RunRunner
        or type(signer) is not RunSigningIdentity
        or type(profile) is not CompetitionProfile
        or profile.fixture_only
        or type(manifest_refs) is not tuple
        or len(manifest_refs) != 4
        or len({item.sha256 for item in manifest_refs}) != 4
        or type(development_run_ids) is not tuple
        or len(development_run_ids) != 200
        or len(set(development_run_ids)) != 200
        or type(development_event_ids) is not tuple
        or type(development_payment_ids) is not tuple
        or type(development_campaign_ids) is not tuple
    ):
        raise CliContractError("hidden source inputs are invalid")
    checked_start = validate_utc_timestamp(minimum_simulation_start)
    manifests: list[RunManifest] = []
    families: list[Family] = []
    source_seeds: list[int] = []
    for reference in manifest_refs:
        try:
            if type(reference) is not ArtifactRef or reference.media_type != "application/json":
                raise CliContractError("hidden run manifest reference differs")
            payload = store.read(reference)
            manifest = RunManifest.model_validate_json(payload)
            if (
                canonical_json_bytes(manifest.model_dump(mode="json")) != payload
                or manifest.signer_key_id != signer.key_id
                or manifest.public_key_base64 != signer.public_key_base64
                or runner.get(manifest.run_id) != manifest
                or not runner.verify_run(manifest)
                or manifest.run_id in development_run_ids
            ):
                raise CliContractError("hidden run failed authenticated verification")
            policy_payload = store.read(manifest.artifacts["policy"])
            scenario_payload = store.read(manifest.artifacts["scenario"])
            policy = AttackerPolicy.model_validate_json(policy_payload)
            scenario = ScenarioBundle.model_validate_json(scenario_payload)
            family = cast(Family, policy.family)
            if family not in _FAMILIES:
                raise CliContractError("hidden run family differs")
            scenario_start = validate_utc_timestamp(
                scenario.replay_manifest.simulation_start
            )
            expected_scenario, expected_policy = _compile_campaign_inputs(
                family,
                index=0,
                profile=profile,
                fixture=False,
                start_override=scenario_start,
                seed_override=scenario.seed,
            )
            if (
                scenario_start <= checked_start
                or policy_payload
                != canonical_json_bytes(expected_policy.model_dump(mode="json"))
                or scenario_payload
                != canonical_json_bytes(expected_scenario.model_dump(mode="json"))
                or policy != expected_policy
                or scenario != expected_scenario
                or policy.kind is not AttackerPolicyKind.FIXED
                or policy.query_budget != 1
                or policy.worker_timeout_ms != 5_000
            ):
                raise CliContractError("hidden run scenario or policy differs")
        except CliContractError:
            raise
        except (KeyError, TypeError, ValueError, ValidationError) as error:
            raise CliContractError("hidden run manifest contents are invalid") from error
        manifests.append(manifest)
        families.append(family)
        source_seeds.append(scenario.seed)
    _require_independent_hidden_seeds(profile, tuple(source_seeds))
    if (
        tuple(families) != _FAMILIES
    ):
        raise CliContractError("hidden source requires one ordered run per family")
    try:
        corpus = assemble_verified_corpus(
            tuple(manifests),
            runner,
            store,
            CorpusProfile(
                profile_id="defense-hidden-authority-v1",
                families=_FAMILIES,
                label_delay_days=profile.label_delay_days,
                fixture_only=False,
            ),
        )
    except (TypeError, ValueError) as error:
        raise CliContractError("hidden run corpus assembly failed") from error
    hidden_event_ids = {row.event_id for row in corpus.observations}
    hidden_payment_ids = {row.payment_id for row in corpus.truth}
    hidden_campaign_ids = {row.campaign_id for row in corpus.truth}
    family_campaigns = {
        family: {row.campaign_id for row in corpus.truth if row.family == family}
        for family in _FAMILIES
    }
    if (
        hidden_event_ids.intersection(development_event_ids)
        or hidden_payment_ids.intersection(development_payment_ids)
        or hidden_campaign_ids.intersection(development_campaign_ids)
        or any(len(family_campaigns[family]) != 1 for family in _FAMILIES)
        or len(hidden_campaign_ids) != 4
        or corpus.manifest.run_ids != tuple(item.run_id for item in manifests)
        or corpus.manifest.run_lineage_digests
        != tuple(item.lineage_digest for item in manifests)
    ):
        raise CliContractError("hidden source is not disjoint and family complete")
    return _AuthenticatedHiddenSource(
        corpus=corpus,
        manifests=tuple(manifests),
        manifest_refs=manifest_refs,
        families=tuple(families),
    )


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


def _prepare_competition_hidden_context(
    *,
    profile: CompetitionProfile,
    root: Path,
    corpus_envelope_ref: ArtifactRef,
    ensemble_ref: ArtifactRef,
    development_run_ids: tuple[str, ...],
    enforce_preregistered_authorities: bool = False,
) -> ArtifactRef:
    """Execute, authenticate, assemble, and seal four authority-owned hidden runs."""
    if (
        type(profile) is not CompetitionProfile
        or profile.fixture_only
        or not isinstance(root, Path)
        or type(corpus_envelope_ref) is not ArtifactRef
        or corpus_envelope_ref.media_type != _CORPUS_ENVELOPE_MEDIA_TYPE
        or type(ensemble_ref) is not ArtifactRef
        or ensemble_ref.media_type != _DEFENDER_ENSEMBLE_MEDIA_TYPE
        or type(development_run_ids) is not tuple
        or len(development_run_ids) != 200
        or len(set(development_run_ids)) != 200
    ):
        raise CliContractError("hidden authority preparation inputs differ")
    try:
        pointer_info = (root / _HIDDEN_CONTEXT_POINTER_NAME).lstat()
    except FileNotFoundError:
        pointer_info = None
    if pointer_info is not None:
        raise CliContractError("hidden context is already immutably selected")
    store = ArtifactStore(root / "artifacts")
    publication_signer = _load_standard_signer(root)
    evaluator = _load_competition_evaluator_identity(root)
    hidden_evaluator, hidden_context_signer = _load_competition_hidden_identity(
        root, evaluator
    )
    if enforce_preregistered_authorities:
        for role, identity in (
            ("publication", publication_signer),
            ("development_evaluator", evaluator),
            ("hidden_evaluator", hidden_evaluator),
        ):
            _assert_preregistered_authority(
                role,
                key_id=identity.key_id,
                public_key_base64=identity.public_key_base64,
            )
    envelope, development_corpus = _load_corpus_envelope(
        store, corpus_envelope_ref.sha256, profile, publication_signer
    )
    if (
        development_corpus.manifest.run_ids != development_run_ids
        or envelope.campaign_count != 200
        or envelope.family_campaign_counts != {family: 50 for family in _FAMILIES}
    ):
        raise CliContractError("hidden source development lineage differs")
    ensemble = _load_defender_ensemble(
        store=store,
        top_ref=ensemble_ref,
        profile=profile,
        signer=publication_signer,
    )
    if (
        ensemble is None
        or ensemble.corpus_envelope_ref is None
        or _artifact_ref(ensemble.corpus_envelope_ref) != corpus_envelope_ref
    ):
        raise CliContractError("hidden source defender lineage differs")
    hidden_run_signer = _load_competition_hidden_run_identity(
        root, evaluator, hidden_evaluator
    )
    if enforce_preregistered_authorities:
        _assert_preregistered_authority(
            "hidden_source",
            key_id=hidden_run_signer.key_id,
            public_key_base64=hidden_run_signer.public_key_base64,
        )
    if (
        ensemble.hidden_source_signer_key_id != hidden_run_signer.key_id
        or ensemble.hidden_source_public_key_base64
        != hidden_run_signer.public_key_base64
    ):
        raise CliContractError("hidden source authority differs from frozen defender")
    candidate_refs = (
        _artifact_ref(ensemble.pooled_ref),
        *(
            _artifact_ref(ensemble.held_family_refs[family])
            for family in _FAMILIES
        ),
    )
    publisher = DefenderBundlePublisher(store, publication_signer, _REPOSITORY_ROOT)
    loaded_candidates: list[LoadedDefenderBundle] = []
    try:
        for reference in candidate_refs:
            loaded = publisher.load(reference)
            loaded.verify_reload()
            loaded_candidates.append(loaded)
    except (TypeError, ValueError) as error:
        raise CliContractError("hidden source defender roster failed reload") from error
    finally:
        publisher.close()
    maximum_frozen_at = max(item.manifest.frozen_at for item in loaded_candidates)
    latest_development_time = max(
        maximum_frozen_at,
        *(row.available_at for row in development_corpus.observations),
        *(row.label_mature_at for row in development_corpus.truth),
    )
    used_seeds = {
        profile.campaign_seed(family, index)
        for family in _FAMILIES
        for index in range(profile.campaigns_per_family)
    }
    hidden_seeds: list[int] = []
    while len(hidden_seeds) != 4:
        candidate = secrets.randbelow(2**63)
        if candidate not in used_seeds and candidate not in hidden_seeds:
            hidden_seeds.append(candidate)
    runner = RunRunner(store, hidden_run_signer, root / "hidden-runs")
    manifest_refs: list[ArtifactRef] = []
    for position, family in enumerate(_FAMILIES, start=1):
        manifest = _run_one_campaign(
            family,
            index=0,
            profile=profile,
            runner=runner,
            fixture=False,
            start_override=latest_development_time + timedelta(days=position),
            seed_override=hidden_seeds[position - 1],
        )
        reference = store.put_bytes(
            canonical_json_bytes(manifest.model_dump(mode="json")),
            "application/json",
        )
        if runner.get(manifest.run_id) != manifest or not runner.verify_run(manifest):
            raise CliContractError("hidden run failed post-execution verification")
        manifest_refs.append(reference)
    development_event_ids = tuple(
        row.event_id for row in development_corpus.observations
    )
    development_payment_ids = tuple(
        row.payment_id for row in development_corpus.truth
    )
    development_campaign_ids = tuple(
        row.campaign_id for row in development_corpus.truth
    )
    source = _assemble_authenticated_hidden_source(
        store=store,
        runner=runner,
        signer=hidden_run_signer,
        profile=profile,
        manifest_refs=tuple(manifest_refs),
        development_run_ids=development_run_ids,
        development_event_ids=development_event_ids,
        development_payment_ids=development_payment_ids,
        development_campaign_ids=development_campaign_ids,
        minimum_simulation_start=latest_development_time,
    )
    from apar.evaluation.competition import (
        _build_competition_hidden_context,
        seal_hidden_context,
        verify_hidden_context,
    )
    context = _build_competition_hidden_context(
        corpus=source.corpus,
        defender=loaded_candidates[0],
    )
    if context.as_of <= maximum_frozen_at:
        raise CliContractError("hidden authority release must follow every defender freeze")
    context_ref = store.put_bytes(context.to_json(), _HIDDEN_CONTEXT_MEDIA_TYPE)
    development_corpus_digest = frozen_corpus_digest(development_corpus)
    hidden_view_corpus = FrozenCorpus(
        observations=source.corpus.observations,
        truth=tuple(
            row.model_copy(
                update={"viewpoint": "hidden", "label_source": "hidden_truth"}
            )
            for row in source.corpus.truth
        ),
        manifest=source.corpus.manifest,
    )
    run_ids_digest = ordered_ids_digest(development_run_ids)
    unsigned_source = {
        "authority_as_of": _utc_wire(context.as_of),
        "development_campaign_ids_digest": ordered_ids_digest(
            development_campaign_ids
        ),
        "development_corpus_digest": development_corpus_digest,
        "development_event_ids_digest": ordered_ids_digest(development_event_ids),
        "development_payment_ids_digest": ordered_ids_digest(
            development_payment_ids
        ),
        "development_run_ids_digest": run_ids_digest,
        "ensemble_ref_sha256": ensemble_ref.sha256,
        "families": list(_FAMILIES),
        "hidden_context_digest": context_ref.sha256,
        "hidden_corpus_digest": frozen_corpus_digest(hidden_view_corpus),
        "kind": "authenticated_independent_hidden_runs",
        "manifest_refs": [
            _reference_document(reference) for reference in source.manifest_refs
        ],
        "minimum_simulation_start": _utc_wire(latest_development_time),
        "profile_sha256": hashlib.sha256(profile.to_json()).hexdigest(),
        "public_key_base64": hidden_run_signer.public_key_base64,
        "run_ids": [item.run_id for item in source.manifests],
        "run_lineage_digests": [
            item.lineage_digest for item in source.manifests
        ],
        "schema_version": "1.0.0",
        "signer_key_id": hidden_run_signer.key_id,
    }
    source_receipt = HiddenSourceReceipt.model_validate(
        {
            **unsigned_source,
            "signature_base64": hidden_run_signer.sign(unsigned_source),
        }
    )
    if not hidden_run_signer.verify(
        source_receipt.unsigned_document(), source_receipt.signature_base64
    ):
        raise CliContractError("hidden source receipt signature failed")
    source_receipt_ref = store.put_bytes(
        canonical_json_bytes(source_receipt.model_dump(mode="json")),
        HIDDEN_SOURCE_RECEIPT_MEDIA_TYPE,
    )
    sealed_ref = seal_hidden_context(
        store=store,
        signer=hidden_context_signer,
        context=context,
        profile_sha256=hashlib.sha256(profile.to_json()).hexdigest(),
        development_corpus_digest=development_corpus_digest,
        source_lineage_digest=source_receipt_ref.sha256,
        source_run_count=4,
    )
    checked_envelope, _, checked_context_ref = verify_hidden_context(
        store=store,
        envelope_ref=sealed_ref,
        signer=hidden_context_signer,
        profile_sha256=hashlib.sha256(profile.to_json()).hexdigest(),
        development_corpus_digest=development_corpus_digest,
        development_event_ids=tuple(
            row.event_id for row in development_corpus.observations
        ),
    )
    if (
        checked_envelope.source_mode != "authenticated_independent_runs"
        or checked_envelope.source_run_count != 4
        or checked_envelope.source_lineage_digest != source_receipt_ref.sha256
        or checked_envelope.as_of != context.as_of
        or checked_context_ref != context_ref
    ):
        raise CliContractError("hidden context self-verification failed")
    unsigned_pointer = {
        "development_corpus_ref": _reference_document(corpus_envelope_ref),
        "ensemble_ref": _reference_document(ensemble_ref),
        "hidden_context_ref": _reference_document(sealed_ref),
        "kind": "competition_hidden_context_pointer",
        "profile_sha256": hashlib.sha256(profile.to_json()).hexdigest(),
        "public_key_base64": publication_signer.public_key_base64,
        "schema_version": "1.0.0",
        "signer_key_id": publication_signer.key_id,
        "source_receipt_ref": _reference_document(source_receipt_ref),
    }
    pointer = HiddenContextPointer.model_validate(
        {
            **unsigned_pointer,
            "signature_base64": publication_signer.sign(unsigned_pointer),
        }
    )
    pointer_payload = canonical_json_bytes(pointer.model_dump(mode="json"))
    pointer_ref = store.put_bytes(
        pointer_payload, _HIDDEN_CONTEXT_POINTER_MEDIA_TYPE
    )
    _publish_json_file(
        root, _HIDDEN_CONTEXT_POINTER_NAME, pointer.model_dump(mode="json")
    )
    return pointer_ref


def _load_hidden_context_pointer(
    *,
    root: Path,
    store: ArtifactStore,
    digest: str,
    signer: RunSigningIdentity,
    profile_sha256: str,
    ensemble_ref: ArtifactRef,
    corpus_envelope_ref: ArtifactRef,
) -> HiddenContextPointer:
    """Load the single fixed signed hidden pointer selected before development."""
    descriptor: int | None = None
    directory_fd: int | None = None
    try:
        directory_fd = os.open(
            root, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW
        )
        descriptor = os.open(
            _HIDDEN_CONTEXT_POINTER_NAME,
            os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW,
            dir_fd=directory_fd,
        )
        info = os.fstat(descriptor)
        if (
            not stat.S_ISREG(info.st_mode)
            or info.st_uid != os.geteuid()
            or stat.S_IMODE(info.st_mode) & 0o077
            or not 0 < info.st_size <= 256 * 1024
        ):
            raise CliContractError("hidden context pointer file differs")
        payload = os.read(descriptor, info.st_size + 1)
        if len(payload) != info.st_size:
            raise CliContractError("hidden context pointer changed during read")
        pointer_ref = store.resolve(_digest(digest))
        if (
            pointer_ref.media_type != _HIDDEN_CONTEXT_POINTER_MEDIA_TYPE
            or pointer_ref.sha256 != hashlib.sha256(payload).hexdigest()
            or store.read(pointer_ref) != payload
        ):
            raise CliContractError("hidden context pointer reference differs")
        pointer = HiddenContextPointer.model_validate_json(payload)
    except CliContractError:
        raise
    except (OSError, TypeError, ValueError, ValidationError) as error:
        raise CliContractError("hidden context pointer is invalid") from error
    finally:
        if descriptor is not None:
            os.close(descriptor)
        if directory_fd is not None:
            os.close(directory_fd)
    if (
        canonical_json_bytes(pointer.model_dump(mode="json")) != payload
        or pointer.signer_key_id != signer.key_id
        or pointer.public_key_base64 != signer.public_key_base64
        or not signer.verify(pointer.unsigned_document(), pointer.signature_base64)
        or pointer.profile_sha256 != _digest(profile_sha256)
        or _artifact_ref(pointer.ensemble_ref) != ensemble_ref
        or _artifact_ref(pointer.development_corpus_ref) != corpus_envelope_ref
    ):
        raise CliContractError("hidden context pointer lineage differs")
    return pointer


def _verify_hidden_source_metadata(
    *,
    store: ArtifactStore,
    profile: CompetitionProfile,
    source_signer_key_id: str,
    source_public_key_base64: str,
    source_receipt_ref: ArtifactRef,
    ensemble_ref: ArtifactRef,
    development_corpus: FrozenCorpus,
    restricted_context_ref: ArtifactRef,
    authority_as_of: datetime,
    maximum_frozen_at: datetime,
) -> HiddenSourceWorkerBinding:
    """Verify public/source metadata without opening hidden events, population, or truth."""
    if (
        type(store) is not ArtifactStore
        or type(profile) is not CompetitionProfile
        or profile.fixture_only
        or _digest(source_signer_key_id) != source_signer_key_id
        or type(source_public_key_base64) is not str
        or type(source_receipt_ref) is not ArtifactRef
        or source_receipt_ref.media_type != HIDDEN_SOURCE_RECEIPT_MEDIA_TYPE
        or type(ensemble_ref) is not ArtifactRef
        or type(development_corpus) is not FrozenCorpus
        or type(restricted_context_ref) is not ArtifactRef
        or restricted_context_ref.media_type != _HIDDEN_CONTEXT_MEDIA_TYPE
    ):
        raise CliContractError("hidden source metadata inputs differ")
    try:
        source_verifier = PublicArtifactVerifier(
            signer_key_id=source_signer_key_id,
            public_key_base64=source_public_key_base64,
        )
        payload = store.read(source_receipt_ref)
        receipt = HiddenSourceReceipt.model_validate_json(payload)
    except (ReportingContractError, TypeError, ValueError, ValidationError) as error:
        raise CliContractError("hidden source receipt is invalid") from error
    development_run_ids = development_corpus.manifest.run_ids
    development_event_ids = tuple(
        row.event_id for row in development_corpus.observations
    )
    development_payment_ids = tuple(
        row.payment_id for row in development_corpus.truth
    )
    development_campaign_ids = tuple(
        row.campaign_id for row in development_corpus.truth
    )
    expected_minimum_start = max(
        (
            validate_utc_timestamp(maximum_frozen_at),
            *(row.available_at for row in development_corpus.observations),
            *(row.label_mature_at for row in development_corpus.truth),
        )
    )
    if (
        canonical_json_bytes(receipt.model_dump(mode="json")) != payload
        or receipt.signer_key_id != source_verifier.key_id
        or receipt.public_key_base64 != source_verifier.public_key_base64
        or not source_verifier.verify(
            receipt.unsigned_document(), receipt.signature_base64
        )
        or receipt.profile_sha256 != hashlib.sha256(profile.to_json()).hexdigest()
        or receipt.ensemble_ref_sha256 != ensemble_ref.sha256
        or receipt.development_corpus_digest
        != frozen_corpus_digest(development_corpus)
        or receipt.development_run_ids_digest
        != ordered_ids_digest(development_run_ids)
        or receipt.development_event_ids_digest
        != ordered_ids_digest(development_event_ids)
        or receipt.development_payment_ids_digest
        != ordered_ids_digest(development_payment_ids)
        or receipt.development_campaign_ids_digest
        != ordered_ids_digest(development_campaign_ids)
        or receipt.hidden_context_digest != restricted_context_ref.sha256
        or receipt.authority_as_of != validate_utc_timestamp(authority_as_of)
        or receipt.minimum_simulation_start != expected_minimum_start
        or set(receipt.run_ids).intersection(development_run_ids)
    ):
        raise CliContractError("hidden source receipt lineage differs")
    source_seeds: list[int] = []
    input_names = {
        "policy",
        "population",
        "provenance",
        "restricted_evaluation_input",
        "restricted_hidden_evaluation_events",
        "scenario",
    }
    output_names = {
        "events",
        "feedback",
        "restricted_evaluation_audit",
        "restricted_validity",
        "summary",
    }
    exact_artifact_names = input_names | output_names | {
        "authorization_receipt",
        "completion_receipt",
    }
    for position, (family, reference) in enumerate(
        zip(_FAMILIES, receipt.references, strict=True)
    ):
        try:
            manifest_payload = store.read(reference)
            manifest = RunManifest.model_validate_json(manifest_payload)
            expected_lineage = hashlib.sha256(
                canonical_json_bytes(
                    {
                        "artifacts": {
                            name: artifact.sha256
                            for name, artifact in sorted(manifest.artifacts.items())
                        },
                        "authorization_receipt": manifest.artifacts[
                            "authorization_receipt"
                        ].sha256,
                        "completion_receipt": manifest.artifacts[
                            "completion_receipt"
                        ].sha256,
                    }
                )
            ).hexdigest()
            if (
                canonical_json_bytes(manifest.model_dump(mode="json"))
                != manifest_payload
                or manifest.schema_version != "1.0.0"
                or set(manifest.artifacts) != exact_artifact_names
                or manifest.run_id != receipt.run_ids[position]
                or manifest.lineage_digest
                != receipt.run_lineage_digests[position]
                or manifest.lineage_digest != expected_lineage
                or manifest.signer_key_id != source_verifier.key_id
                or manifest.public_key_base64 != source_verifier.public_key_base64
                or not source_verifier.verify(
                    manifest.unsigned_document(), manifest.signature_base64
                )
            ):
                raise CliContractError("hidden source manifest signature differs")
            policy_payload = store.read(manifest.artifacts["policy"])
            scenario_payload = store.read(manifest.artifacts["scenario"])
            policy = AttackerPolicy.model_validate_json(policy_payload)
            scenario = ScenarioBundle.model_validate_json(scenario_payload)
            scenario_start = validate_utc_timestamp(
                scenario.replay_manifest.simulation_start
            )
            expected_scenario, expected_policy = _compile_campaign_inputs(
                family,
                index=0,
                profile=profile,
                fixture=False,
                start_override=scenario_start,
                seed_override=scenario.seed,
            )
            authorization_payload = store.read(
                manifest.artifacts["authorization_receipt"]
            )
            completion_payload = store.read(manifest.artifacts["completion_receipt"])
            authorization = SignedRunReceipt.model_validate_json(
                authorization_payload
            )
            completion = SignedRunReceipt.model_validate_json(completion_payload)
            if (
                policy.family != family
                or manifest.scenario_id != scenario.scenario_id
                or manifest.policy_kind is not AttackerPolicyKind.FIXED
                or policy_payload
                != canonical_json_bytes(expected_policy.model_dump(mode="json"))
                or scenario_payload
                != canonical_json_bytes(expected_scenario.model_dump(mode="json"))
                or policy != expected_policy
                or scenario != expected_scenario
                or scenario_start <= receipt.minimum_simulation_start
                or canonical_json_bytes(authorization.model_dump(mode="json"))
                != authorization_payload
                or canonical_json_bytes(completion.model_dump(mode="json"))
                != completion_payload
                or authorization.run_id != manifest.run_id
                or completion.run_id != manifest.run_id
                or authorization.schema_version != "1.0.0"
                or completion.schema_version != "1.0.0"
                or authorization.receipt_kind != "authorization"
                or completion.receipt_kind != "completion"
                or authorization.signer_key_id != source_verifier.key_id
                or completion.signer_key_id != source_verifier.key_id
                or authorization.public_key_base64
                != source_verifier.public_key_base64
                or completion.public_key_base64
                != source_verifier.public_key_base64
                or not source_verifier.verify(
                    authorization.unsigned_document(), authorization.signature_base64
                )
                or not source_verifier.verify(
                    completion.unsigned_document(), completion.signature_base64
                )
                or authorization.artifact_digests
                != {
                    name: manifest.artifacts[name].sha256
                    for name in sorted(input_names)
                }
                or completion.artifact_digests
                != {
                    name: manifest.artifacts[name].sha256
                    for name in sorted(output_names)
                }
                or authorization.previous_receipt_sha256 is not None
                or completion.previous_receipt_sha256
                != manifest.artifacts["authorization_receipt"].sha256
            ):
                raise CliContractError("hidden source public run metadata differs")
            source_seeds.append(scenario.seed)
        except CliContractError:
            raise
        except (KeyError, TypeError, ValueError, ValidationError) as error:
            raise CliContractError("hidden source public run metadata is invalid") from error
    _require_independent_hidden_seeds(profile, tuple(source_seeds))
    return HiddenSourceWorkerBinding(
        receipt_ref=source_receipt_ref,
        source_signer_key_id=source_verifier.key_id,
        source_public_key_base64=source_verifier.public_key_base64,
        development_run_ids=development_run_ids,
        development_event_ids=development_event_ids,
        development_payment_ids=development_payment_ids,
        development_campaign_ids=development_campaign_ids,
    )


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
    campaign_end_times: dict[str, datetime] | None = None,
) -> tuple[RollingFold, ...]:
    campaigns = split.campaigns["train"]
    row_campaigns = split.row_campaigns
    training_ids = split.training_row_ids
    checked_end_times = (
        campaign_start_times if campaign_end_times is None else campaign_end_times
    )
    if (
        set(campaigns) != set(campaign_start_times)
        or set(campaigns) != set(checked_end_times)
        or any(checked_end_times[item] < campaign_start_times[item] for item in campaigns)
    ):
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
    eligible_boundaries: list[int] = []
    latest_fit_end = max(checked_end_times[item] for item in cohorts[0][1])
    for boundary in range(1, len(cohorts)):
        validation_start = cohorts[boundary][0]
        if latest_fit_end < validation_start:
            eligible_boundaries.append(boundary)
        latest_fit_end = max(
            latest_fit_end,
            *(checked_end_times[item] for item in cohorts[boundary][1]),
        )
    if len(eligible_boundaries) < 2:
        raise CliContractError(
            "competition rolling folds need two causal campaign boundaries"
        )
    first_target = len(cohorts) // 2
    first = min(
        eligible_boundaries[:-1],
        key=lambda boundary: (abs(boundary - first_target), boundary),
    )
    second_target = (len(cohorts) * 3) // 4
    second = min(
        (boundary for boundary in eligible_boundaries if boundary > first),
        key=lambda boundary: (abs(boundary - second_target), boundary),
    )
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
    campaign_decision_bounds = {
        campaign: (
            min(
                cast(datetime, event_by_id[row_id].decision_at)
                for row_id in split.training_row_ids
                if split.row_campaigns[row_id] == campaign
            ),
            max(
                cast(datetime, event_by_id[row_id].decision_at)
                for row_id in split.training_row_ids
                if split.row_campaigns[row_id] == campaign
            ),
        )
        for campaign in split.campaigns["train"]
    }
    scorer = train_gbdt(
        matrix,
        labels,
        split.training_row_ids,
        _rolling_campaign_folds(
            split,
            campaign_start_times={
                campaign: bounds[0]
                for campaign, bounds in campaign_decision_bounds.items()
            },
            campaign_end_times={
                campaign: bounds[1]
                for campaign, bounds in campaign_decision_bounds.items()
            },
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
        split_payload = canonical_json_bytes(split.model_dump(mode="json"))
        projection = _public_split_projection_from_digest(
            split,
            split_artifact_digest=hashlib.sha256(split_payload).hexdigest(),
        )
        raise _CompetitionThresholdInfeasible(
            _ThresholdFailureArtifacts(
                model=scorer.to_bytes(),
                training_receipt=canonical_json_bytes(
                    scorer.receipt.model_dump(mode="json")
                ),
                calibration=calibrator.to_json(),
                threshold_report=threshold_report,
                feature_manifest=canonical_json_bytes(catalog.model_dump(mode="json")),
                features=_portable_feature_matrix_parquet(training_matrix),
                split_projection=canonical_json_bytes(
                    projection.model_dump(mode="json")
                ),
                rules=canonical_json_bytes(
                    RuleManifest.default().model_dump(mode="json")
                ),
            )
        )
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
    publisher = DefenderBundlePublisher(store, signer, _REPOSITORY_ROOT)
    try:
        role = held_out_family or "pooled"

        def freeze_one(
            *, bundle_role: str, frozen_at: datetime, predecessor: str
        ) -> ArtifactRef:
            _manifest, frozen_reference = publisher.freeze(
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
                        f"{bundle_role}",
                    )
                ),
                frozen_at=frozen_at,
                rollback_ref=predecessor,
            )
            return frozen_reference

        if rollback_ref == "rules-v1":
            predecessor_ref = freeze_one(
                bundle_role=f"rules-v1:{role}",
                frozen_at=split.config.development_end - timedelta(microseconds=1),
                predecessor=GENESIS_ROLLBACK_REF,
            )
            checked_rollback = predecessor_ref.sha256
        else:
            checked_rollback = (
                GENESIS_ROLLBACK_REF
                if rollback_ref == GENESIS_ROLLBACK_REF
                else _digest(rollback_ref)
            )
        reference = freeze_one(
            bundle_role=role,
            frozen_at=split.config.development_end,
            predecessor=checked_rollback,
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


def provision_defense_authorities(root: Path) -> dict[str, dict[str, str]]:
    """One-shot provisioning for the four distinct defense-v1 trust roots."""
    checked_root = _secure_root(root)
    names = {
        "publication": "run-signing.key",
        "development_evaluator": "evaluator-signing.key",
        "hidden_evaluator": "hidden-evaluator-signing.key",
        "hidden_source": "hidden-run-signing.key",
    }
    final_names = (*names.values(), "hidden-run-signing.pub")
    directory_fd = os.open(
        checked_root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
    )
    try:
        metadata = os.fstat(directory_fd)
        if (
            not stat.S_ISDIR(metadata.st_mode)
            or stat.S_IMODE(metadata.st_mode) != 0o700
            or metadata.st_uid != os.geteuid()
        ):
            raise CliContractError("authority root must be an owned mode-0700 directory")
        for name in final_names:
            try:
                os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
            except FileNotFoundError:
                continue
            except OSError as error:
                raise CliContractError("authority provisioning preflight failed") from error
            raise CliContractError("defense authorities are already provisioned")

        seeds = {role: secrets.token_bytes(32) for role in names}
        identities = {
            role: RunSigningIdentity.from_private_bytes(seed)
            for role, seed in seeds.items()
        }
        if (
            len(set(seeds.values())) != len(seeds)
            or len({identity.key_id for identity in identities.values()}) != len(identities)
            or any(
                seed
                in {
                    _FIXTURE_SIGNER_SEED,
                    hashlib.sha256(b"apar-g3-development-evaluator-v1").digest(),
                    hashlib.sha256(b"apar-g3-hidden-evaluator-v1").digest(),
                }
                for seed in seeds.values()
            )
        ):
            raise CliContractError("authority identities are not distinct")
        public_source = base64.b64decode(
            identities["hidden_source"].public_key_base64, validate=True
        )
        payloads = {
            **{names[role]: seeds[role] for role in names},
            "hidden-run-signing.pub": public_source,
        }
        created: list[str] = []
        try:
            for name, payload in payloads.items():
                temporary = f".{name}.{secrets.token_hex(12)}.tmp"
                try:
                    descriptor = os.open(
                        temporary,
                        os.O_WRONLY
                        | os.O_CREAT
                        | os.O_EXCL
                        | os.O_NOFOLLOW
                        | os.O_CLOEXEC,
                        0o600,
                        dir_fd=directory_fd,
                    )
                    try:
                        offset = 0
                        while offset < len(payload):
                            written = os.write(descriptor, payload[offset:])
                            if written <= 0:
                                raise OSError("authority provisioning made no progress")
                            offset += written
                        os.fchmod(descriptor, 0o600)
                        os.fsync(descriptor)
                        file_metadata = os.fstat(descriptor)
                        if (
                            not stat.S_ISREG(file_metadata.st_mode)
                            or stat.S_IMODE(file_metadata.st_mode) != 0o600
                            or file_metadata.st_uid != os.geteuid()
                            or file_metadata.st_nlink != 1
                            or file_metadata.st_size != 32
                        ):
                            raise CliContractError(
                                "provisioned authority file is invalid"
                            )
                    finally:
                        os.close(descriptor)
                    ArtifactStore.publish_no_replace_at(
                        directory_fd, temporary, name
                    )
                finally:
                    with contextlib.suppress(FileNotFoundError):
                        os.unlink(temporary, dir_fd=directory_fd)
                created.append(name)
            os.fsync(directory_fd)
        except BaseException:
            for name in created:
                with contextlib.suppress(FileNotFoundError):
                    os.unlink(name, dir_fd=directory_fd)
            os.fsync(directory_fd)
            raise
        return {
            role: {
                "key_id": identities[role].key_id,
                "public_key_base64": identities[role].public_key_base64,
            }
            for role in sorted(identities)
        }
    except CliContractError:
        raise
    except (OSError, ValueError, binascii.Error) as error:
        raise CliContractError("authority provisioning failed closed") from error
    finally:
        os.close(directory_fd)


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


def _publish_bytes_file(path: Path, payload: bytes) -> None:
    """Publish one preregistered portable artifact durably without replacement."""
    if type(payload) is not bytes or not payload:
        raise CliContractError("portable artifact payload must be nonempty bytes")
    directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
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
                    raise OSError("portable publication made no progress")
                offset += written
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        ArtifactStore.publish_no_replace_at(
            directory_fd, temporary_name, path.name
        )
        os.fsync(directory_fd)
    except FileExistsError as error:
        raise CliContractError(
            "portable evidence already exists; frozen evidence is never overwritten"
        ) from error
    finally:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(temporary_name, dir_fd=directory_fd)
        os.close(directory_fd)


def _portable_rows_parquet(
    rows: tuple[ObservedEvent, ...] | tuple[EvaluationTruthRow, ...],
    *,
    classification: Literal["defender_visible", "restricted_evaluator_only"],
    content_digest: str,
) -> bytes:
    """Encode canonical row documents in a deterministic, language-neutral Parquet table."""
    import pyarrow as pa  # type: ignore[import-untyped]
    import pyarrow.parquet as pq  # type: ignore[import-untyped]

    _digest(content_digest)
    schema = pa.schema(
        (pa.field("row_json", pa.binary(), nullable=False),),
        metadata={
            b"classification": classification.encode("ascii"),
            b"content_digest": content_digest.encode("ascii"),
            b"row_contract": (
                b"apar.observed-event.v1"
                if classification == "defender_visible"
                else b"apar.evaluation-truth.v1"
            ),
            b"schema_version": b"1.0.0",
        },
    )
    table = pa.Table.from_arrays(
        [
            pa.array(
                [canonical_json_bytes(row.model_dump(mode="json")) for row in rows],
                type=pa.binary(),
            )
        ],
        schema=schema,
    )
    sink = pa.BufferOutputStream()
    pq.write_table(
        table,
        sink,
        compression="NONE",
        use_dictionary=False,
        write_statistics=False,
        row_group_size=max(1, len(rows)),
    )
    return cast(bytes, sink.getvalue().to_pybytes())


def _portable_feature_matrix_parquet(matrix: FeatureMatrix) -> bytes:
    """Encode the public 48-column training matrix without labels or truth."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    feature_names = tuple(matrix.catalog.names)
    fields = (
        pa.field("event_id", pa.string(), nullable=False),
        pa.field("decision_at", pa.string(), nullable=False),
        *(pa.field(name, pa.float64(), nullable=False) for name in feature_names),
    )
    schema = pa.schema(
        fields,
        metadata={
            b"catalog_digest": matrix.catalog_digest.encode("ascii"),
            b"classification": b"defender_visible",
            b"row_contract": b"apar.feature-vector.v1",
            b"schema_version": b"1.0.0",
        },
    )
    columns: list[object] = [
        [row.event_id for row in matrix.rows],
        [_utc_wire(row.decision_at) for row in matrix.rows],
    ]
    columns.extend(
        [float(row.values[name]) for row in matrix.rows] for name in feature_names
    )
    table = pa.Table.from_arrays(
        [
            pa.array(column, type=field.type)
            for column, field in zip(columns, fields, strict=True)
        ],
        schema=schema,
    )
    sink = pa.BufferOutputStream()
    pq.write_table(
        table,
        sink,
        compression="NONE",
        use_dictionary=False,
        write_statistics=False,
        row_group_size=max(1, len(matrix.rows)),
    )
    return cast(bytes, sink.getvalue().to_pybytes())


def _export_defense_v1_corpus(
    *,
    path: Path,
    reference: ArtifactRef,
    envelope: CorpusEnvelope,
    corpus: FrozenCorpus,
    signer: RunSigningIdentity,
) -> None:
    observation_path = path.parent / "observations.parquet"
    truth_path = path.parent / "evaluation-truth.parquet"
    observation_payload = _portable_rows_parquet(
        corpus.observations,
        classification="defender_visible",
        content_digest=envelope.observation_digest,
    )
    truth_payload = _portable_rows_parquet(
        corpus.truth,
        classification="restricted_evaluator_only",
        content_digest=envelope.restricted_truth_digest,
    )
    _publish_bytes_file(observation_path, observation_payload)
    _publish_bytes_file(truth_path, truth_payload)
    _publish_defense_v1_alias(
        path=path,
        kind="corpus_envelope",
        reference=reference,
        signer=signer,
        authenticated_run_ids=corpus.manifest.run_ids,
        export_metadata={
            "observation_dataset": {
                "classification": "defender_visible",
                "content_digest": envelope.observation_digest,
                "file_sha256": hashlib.sha256(observation_payload).hexdigest(),
                "row_count": len(corpus.observations),
            },
            "truth_dataset": {
                "classification": "restricted_evaluator_only",
                "content_digest": envelope.restricted_truth_digest,
                "file_sha256": hashlib.sha256(truth_payload).hexdigest(),
                "row_count": len(corpus.truth),
            },
        },
    )


def _public_split_projection(
    split: EvaluationSplit,
    manifest: DefenderBundleManifest,
) -> PublicSplitProjection:
    if split.split_digest != manifest.split_manifest_digest:
        raise CliContractError("public split projection lineage differs")
    return _public_split_projection_from_digest(
        split,
        split_artifact_digest=manifest.split_artifact_digest,
    )


def _public_split_projection_from_digest(
    split: EvaluationSplit,
    *,
    split_artifact_digest: str,
) -> PublicSplitProjection:
    """Build the truth-free public split view from an exact signed source digest."""
    _digest(split_artifact_digest)
    projection = PublicSplitProjection(
        config=split.config,
        partition_names=tuple(split.partition_names),
        campaigns=split.campaigns,
        row_ids=split.row_ids,
        training_row_ids=split.training_row_ids,
        entity_cohorts={
            event_id: tuple(cohort.value for cohort in cohorts)
            for event_id, cohorts in split.entity_cohorts.items()
        },
        row_families=split.row_families,
        row_campaigns=split.row_campaigns,
        label_maturity_cutoff=split.label_maturity_cutoff,
        sample_counts=split.sample_counts,
        held_out_family=split.held_out_family,
        held_out_evaluation_row_ids=split.held_out_evaluation_row_ids,
        split_digest=split.split_digest,
        split_semantic_digest=hashlib.sha256(
            canonical_json_bytes(
                split.model_dump(mode="json", exclude={"split_digest"})
            )
        ).hexdigest(),
        split_artifact_digest=split_artifact_digest,
    )
    payload = canonical_json_bytes(projection.model_dump(mode="json"))
    forbidden = (b"row_is_fraud", b"row_net_settled_values", b"fraud_prevalence")
    if any(item in payload for item in forbidden):
        raise CliContractError("public split projection contains restricted truth")
    return projection


def _export_defense_v1_defender(
    *,
    directory: Path,
    reference: ArtifactRef,
    root: Path,
    profile: CompetitionProfile,
    signer: RunSigningIdentity,
    authenticated_run_ids: tuple[str, ...],
) -> None:
    """Export exact pooled Task 9 bytes plus the signed five-role ensemble pointer."""
    store = ArtifactStore(root / "artifacts")
    ensemble = _load_defender_ensemble(
        store=store,
        top_ref=reference,
        profile=profile,
        signer=signer,
    )
    if ensemble is None:
        raise CliContractError("defense-v1 export requires the full competition ensemble")
    pooled_ref = _artifact_ref(ensemble.pooled_ref)
    publisher = DefenderBundlePublisher(store, signer, _REPOSITORY_ROOT)
    try:
        candidate_refs = (
            pooled_ref,
            *(
                _artifact_ref(ensemble.held_family_refs[family])
                for family in profile.families
            ),
        )
        loaded_candidates = tuple(publisher.load(item) for item in candidate_refs)
        for candidate in loaded_candidates:
            candidate.verify_reload()
        portable_loaded: list[tuple[ArtifactRef, object]] = list(
            zip(candidate_refs, loaded_candidates, strict=True)
        )
        for candidate in loaded_candidates:
            predecessor_digest = candidate.manifest.rollback_ref
            while predecessor_digest != GENESIS_ROLLBACK_REF:
                predecessor_ref = store.resolve(predecessor_digest)
                predecessor = publisher.load(predecessor_ref)
                predecessor.verify_reload()
                portable_loaded.append((predecessor_ref, predecessor))
                predecessor_digest = predecessor.manifest.rollback_ref
    finally:
        publisher.close()
    pooled = loaded_candidates[0]
    manifest = pooled.manifest
    component_files = {
        "calibration": "calibration.json",
        "model": "model.cbm",
        "receipt": "training-receipt.json",
        "rules": "rules.json",
        "training_matrix": "features.parquet",
    }
    for component_name, filename in component_files.items():
        component = manifest.component(component_name)
        component_ref = ArtifactRef(
            component.sha256,
            component.media_type,
            component.size_bytes,
            f"{component.sha256}/payload",
        )
        _publish_bytes_file(directory / filename, store.read(component_ref))
    split_projections: dict[str, dict[str, object]] = {}
    for candidate_ref, candidate_object in portable_loaded:
        candidate = cast(LoadedDefenderBundle, candidate_object)
        split_component = candidate.manifest.component("split")
        split_ref = ArtifactRef(
            split_component.sha256,
            split_component.media_type,
            split_component.size_bytes,
            f"{split_component.sha256}/payload",
        )
        split = EvaluationSplit.model_validate(strict_json_loads(store.read(split_ref)))
        projection = _public_split_projection(split, candidate.manifest)
        split_projections[candidate_ref.sha256] = projection.model_dump(mode="json")
    pooled_projection = PublicSplitProjection.model_validate(
        split_projections[pooled_ref.sha256]
    )
    _publish_bytes_file(
        directory / "split-manifest.json",
        canonical_json_bytes(pooled_projection.model_dump(mode="json")),
    )
    portable_refs: dict[str, ArtifactRef] = {reference.sha256: reference}
    for candidate_ref, candidate_object in portable_loaded:
        candidate = cast(Any, candidate_object)
        portable_refs[candidate_ref.sha256] = candidate_ref
        for component in candidate.manifest.components:
            if component.name == "split":
                continue
            portable_refs[component.sha256] = ArtifactRef(
                component.sha256,
                component.media_type,
                component.size_bytes,
                f"{component.sha256}/payload",
            )
    portable_artifacts = {
        digest: {
            "media_type": item.media_type,
            "payload_base64": base64.b64encode(store.read(item)).decode("ascii"),
            "size_bytes": item.size_bytes,
        }
        for digest, item in sorted(portable_refs.items())
    }
    _publish_defense_v1_alias(
        path=directory / "defender-bundle.json",
        kind="defender_ensemble",
        reference=reference,
        signer=signer,
        authenticated_run_ids=authenticated_run_ids,
        export_metadata={
            "held_family_refs": ensemble.held_family_refs,
            "pooled_manifest": manifest.model_dump(mode="json"),
            "pooled_ref": ensemble.pooled_ref,
            "portable_artifacts": portable_artifacts,
            "split_projections": split_projections,
        },
    )


def _utc_wire(value: datetime) -> str:
    checked = validate_utc_timestamp(value)
    return checked.isoformat().replace("+00:00", "Z")


def _build_hidden_release_attestation(
    *,
    proof: HiddenPublicProof,
    signer: EvaluatorSigningIdentity,
    ensemble_ref: ArtifactRef,
    pooled_ref: ArtifactRef,
    held_family_refs: Mapping[Family, ArtifactRef],
    evaluation_bundle_ref: ArtifactRef,
    scorecard_ref: ArtifactRef,
    promotion_envelope_digest: str,
    defender_frozen_at: datetime,
) -> HiddenReleaseAttestation:
    if (
        type(proof) is not HiddenPublicProof
        or not EvaluatorSigningIdentity.is_exact(signer)
        or set(held_family_refs) != set(_FAMILIES)
    ):
        raise CliContractError("hidden release attestation inputs differ")
    verifier = EvaluatorReplayVerifier.from_signer(signer)
    issued_at = validate_utc_timestamp(
        datetime.fromisoformat(proof.issued_at.replace("Z", "+00:00"))
    )
    if (
        not verifier.verify_public_proof(proof)
        or proof.defender_top_ref_digest != pooled_ref.sha256
        or proof.bundle_manifest_digest != pooled_ref.sha256
        or issued_at <= defender_frozen_at
    ):
        raise CliContractError("hidden public proof lineage differs")
    roster_digest = _candidate_roster_digest(pooled_ref, held_family_refs)
    unsigned: dict[str, object] = {
        "authority_issued_at": proof.issued_at,
        "candidate_roster_digest": roster_digest,
        "ensemble_top_ref_digest": ensemble_ref.sha256,
        "evaluation_bundle_digest": evaluation_bundle_ref.sha256,
        "hidden_proof_digest": proof.proof_digest,
        "pooled_defender_ref_digest": pooled_ref.sha256,
        "profile_sha256": _DEFENSE_V1_PROFILE_SHA256,
        "promotion_envelope_digest": _digest(promotion_envelope_digest),
        "public_key_base64": signer.public_key_base64,
        "schema_version": "1.0.0",
        "scorecard_digest": scorecard_ref.sha256,
        "signer_key_id": signer.key_id,
    }
    return HiddenReleaseAttestation.model_validate(
        {**unsigned, "signature_base64": signer._sign(unsigned)}
    )


def _candidate_roster_digest(
    pooled_ref: ArtifactRef,
    held_family_refs: Mapping[Family, ArtifactRef],
) -> str:
    if type(pooled_ref) is not ArtifactRef or set(held_family_refs) != set(_FAMILIES):
        raise CliContractError("candidate roster differs")
    return hashlib.sha256(
        canonical_json_bytes(
            {
                "pooled": _reference_document(pooled_ref),
                "held_family_refs": {
                    family: _reference_document(held_family_refs[family])
                    for family in _FAMILIES
                },
            }
        )
    ).hexdigest()


def _export_defense_v1_threshold_failure(
    *,
    directory: Path,
    result_path: Path,
    hash_manifest_path: Path,
    root: Path,
    signer: RunSigningIdentity,
    artifacts: _ThresholdFailureArtifacts,
    corpus_envelope_ref: ArtifactRef,
    authenticated_run_ids: tuple[str, ...],
    run_ledger_sha256: str,
) -> None:
    """Publish the preregistered, non-promotable operating-budget failure."""
    report = artifacts.threshold_report
    _assert_preregistered_authority(
        "publication",
        key_id=signer.key_id,
        public_key_base64=signer.public_key_base64,
    )
    _digest(run_ledger_sha256)
    expected_preexisting = {
        "corpus-manifest.json",
        "evaluation-truth.parquet",
        "observations.parquet",
    }
    try:
        present_entries = tuple(directory.iterdir())
        result_path.lstat()
    except FileNotFoundError:
        result_absent = True
    except OSError as error:
        raise CliContractError("defense-v1 failure export preflight failed") from error
    else:
        result_absent = False
    if (
        {item.name for item in present_entries} != expected_preexisting
        or any(item.is_symlink() or not item.is_file() for item in present_entries)
        or not result_absent
        or hash_manifest_path.parent != directory
        or hash_manifest_path.name != "hash-manifest.json"
        or report.feasible
        or report.reason != "no_candidate_satisfies_operating_budget"
        or report.minimum_review_case_rate <= report.budget.review_case_rate_max
        or len(authenticated_run_ids) != 200
        or len(set(authenticated_run_ids)) != 200
    ):
        raise CliContractError("defense-v1 failure export inputs differ")

    minimum_review_cases = round(report.minimum_review_case_rate * report.row_count)
    if minimum_review_cases / report.row_count != report.minimum_review_case_rate:
        raise CliContractError("failure workload evidence is not integral")
    public_report = {
        "budget": report.budget.model_dump(mode="json"),
        "candidate_count": report.candidate_count,
        "candidate_threshold_count": report.candidate_threshold_count,
        "failure_reason": "operating_budget_infeasible",
        "feasible": False,
        "feasible_candidate_count": 0,
        "fraud_count": report.fraud_count,
        "legitimate_count": report.legitimate_count,
        "minimum_challenge_rate": report.minimum_challenge_rate,
        "minimum_false_decline_rate": report.minimum_false_decline_rate,
        "minimum_review_case_count": minimum_review_cases,
        "minimum_review_case_rate": report.minimum_review_case_rate,
        "objective_kind": report.objective_kind,
        "reason": report.reason,
        "row_count": report.row_count,
    }
    unsigned_thresholds: dict[str, object] = {
        "arm": "layered_hybrid",
        "preregistration_sha256": _DEFENSE_V1_PREREGISTRATION_SHA256,
        "profile_sha256": _DEFENSE_V1_PROFILE_SHA256,
        "public_key_base64": signer.public_key_base64,
        "report": public_report,
        "schema_version": "1.0.0",
        "signer_key_id": signer.key_id,
        "status": "no_promotion",
    }
    thresholds_payload = canonical_json_bytes(
        {
            **unsigned_thresholds,
            "signature_base64": signer.sign(unsigned_thresholds),
        }
    )
    threshold_report_digest = hashlib.sha256(thresholds_payload).hexdigest()

    component_payloads: dict[str, tuple[bytes, str]] = {
        "calibration.json": (artifacts.calibration, "application/json"),
        "feature-manifest.json": (artifacts.feature_manifest, "application/json"),
        "features.parquet": (artifacts.features, "application/json"),
        "model.cbm": (artifacts.model, "application/vnd.catboost.model"),
        "rules.json": (artifacts.rules, "application/json"),
        "split-manifest.json": (artifacts.split_projection, "application/json"),
        "thresholds.json": (thresholds_payload, "application/json"),
        "training-receipt.json": (artifacts.training_receipt, "application/json"),
    }
    store = ArtifactStore(root / "artifacts")
    component_refs = {
        name: store.put_bytes(payload, media_type)
        for name, (payload, media_type) in component_payloads.items()
    }
    unsigned_candidate: dict[str, object] = {
        "components": {
            name: _reference_document(reference)
            for name, reference in sorted(component_refs.items())
        },
        "corpus_envelope": _reference_document(corpus_envelope_ref),
        "failure_reason": "operating_budget_infeasible",
        "preregistration_sha256": _DEFENSE_V1_PREREGISTRATION_SHA256,
        "profile_sha256": _DEFENSE_V1_PROFILE_SHA256,
        "public_key_base64": signer.public_key_base64,
        "schema_version": "1.0.0",
        "signer_key_id": signer.key_id,
        "status": "no_promotion",
        "threshold_report_digest": threshold_report_digest,
    }
    candidate_ref = store.put_bytes(
        canonical_json_bytes(
            {
                **unsigned_candidate,
                "signature_base64": signer.sign(unsigned_candidate),
            }
        ),
        _INFEASIBLE_CANDIDATE_MEDIA_TYPE,
    )
    alias_unsigned_metadata = {
        "failure_reason": "operating_budget_infeasible",
        "public_artifacts": {
            name: _reference_document(reference)
            for name, reference in sorted(component_refs.items())
        },
        "threshold_report_digest": threshold_report_digest,
    }
    alias_unsigned = {
        "artifact": _reference_document(candidate_ref),
        "authenticated_run_ids": list(authenticated_run_ids),
        "campaign_count": 200,
        "export_metadata": alias_unsigned_metadata,
        "family_counts": {family: 50 for family in _FAMILIES},
        "kind": "infeasible_candidate",
        "preregistration_sha256": _DEFENSE_V1_PREREGISTRATION_SHA256,
        "profile_sha256": _DEFENSE_V1_PROFILE_SHA256,
        "public_key_base64": signer.public_key_base64,
        "schema_version": "1.0.0",
        "signer_key_id": signer.key_id,
    }
    alias_payload = canonical_json_bytes(
        {**alias_unsigned, "signature_base64": signer.sign(alias_unsigned)}
    )
    component_payloads["defender-bundle.json"] = (alias_payload, "application/json")

    unsigned_scorecard: dict[str, object] = {
        "campaign_count": 200,
        "champion": None,
        "failed_gate": "operating_budget",
        "failure_reason": "operating_budget_infeasible",
        "preregistration_sha256": _DEFENSE_V1_PREREGISTRATION_SHA256,
        "profile_sha256": _DEFENSE_V1_PROFILE_SHA256,
        "public_key_base64": signer.public_key_base64,
        "schema_version": "1.0.0",
        "signer_key_id": signer.key_id,
        "status": "no_promotion",
        "threshold_report_digest": threshold_report_digest,
    }
    scorecard_payload = canonical_json_bytes(
        {
            **unsigned_scorecard,
            "signature_base64": signer.sign(unsigned_scorecard),
        }
    )
    component_payloads.update(
        {
            "calibration.csv": (
                b"status,calibrator_artifact\nnot_evaluated,calibration.json\n",
                "text/csv",
            ),
            "data-card.md": (
                b"# Data card\n\nSynthetic APAR corpus: 200 authenticated "
                b"campaigns; evaluator truth remains separately classified.\n",
                "text/markdown",
            ),
            "defense-scorecard.json": (scorecard_payload, "application/json"),
            "defense-scorecard.md": (
                b"# Defense scorecard\n\nStatus: **no promotion**. The frozen 1% "
                b"review-case budget is infeasible; no champion was selected.\n",
                "text/markdown",
            ),
            "latency-evidence.json": (
                canonical_json_bytes(
                    {
                        "evaluation_executed": False,
                        "reason": "operating_budget_infeasible",
                        "schema_version": "1.0.0",
                    }
                ),
                "application/json",
            ),
            "leaderboard.csv": (
                b"arm,status,reason\nrules_only,not_evaluated,operating_budget_infeasible\ngbdt_only,not_evaluated,operating_budget_infeasible\nlayered_hybrid,no_promotion,operating_budget_infeasible\n",
                "text/csv",
            ),
            "limitations.md": (
                b"# Limitations\n\nNo operating point satisfies the preregistered "
                b"review workload cap. Hidden evaluation was not released and no "
                b"performance claim is made.\n",
                "text/markdown",
            ),
            "model-card.md": (
                b"# Model card\n\nA deterministic synthetic-only candidate was "
                b"trained, but it was not frozen, promoted, or evaluated after "
                b"threshold infeasibility.\n",
                "text/markdown",
            ),
            "slice-metrics.csv": (
                b"slice,status,reason\nall,not_evaluated,operating_budget_infeasible\n",
                "text/csv",
            ),
            "value-workload.csv": (
                (
                    "metric,value,budget,status\n"
                    f"minimum_review_case_rate,{report.minimum_review_case_rate},"
                    f"{report.budget.review_case_rate_max},failed\n"
                    f"minimum_review_case_count,{minimum_review_cases},,observed\n"
                ).encode("ascii"),
                "text/csv",
            ),
        }
    )
    if set(component_payloads) != _DEFENSE_V1_FIXTURE_FILES - expected_preexisting:
        raise CliContractError("defense-v1 failure artifact set differs")
    for name, (payload, _media_type) in sorted(component_payloads.items()):
        _publish_bytes_file(directory / name, payload)

    all_refs = {
        name: store.put_bytes(
            _regular_file(
                directory / name,
                label="defense-v1 failure artifact",
                max_bytes=300 * 1024 * 1024,
            ),
            component_payloads.get(name, (b"", "application/octet-stream"))[1],
        )
        for name in sorted(_DEFENSE_V1_FIXTURE_FILES)
    }
    unsigned_result: dict[str, object] = {
        "authority_identities": {
            role: asdict(_preregistered_authority(role))
            for role in (
                "development_evaluator",
                "hidden_evaluator",
                "hidden_source",
                "publication",
            )
        },
        "campaign_count": 200,
        "champion": None,
        "corpus_envelope": _reference_document(corpus_envelope_ref),
        "defender_frozen_at": None,
        "failure_reason": "operating_budget_infeasible",
        "failure_stage": "threshold_selection",
        "failures_remain_public": True,
        "hidden_release_status": "not_attempted_frozen_defender_unavailable",
        "hidden_released_at": None,
        "preregistration_sha256": _DEFENSE_V1_PREREGISTRATION_SHA256,
        "profile_sha256": _DEFENSE_V1_PROFILE_SHA256,
        "public_artifacts": {
            name: _reference_document(reference)
            for name, reference in sorted(all_refs.items())
        },
        "public_key_base64": signer.public_key_base64,
        "retuned_after_failure": False,
        "run_ledger_sha256": run_ledger_sha256,
        "schema_version": "1.0.0",
        "signer_key_id": signer.key_id,
        "status": "no_promotion",
        "synthetic_only": True,
        "threshold_report": public_report,
        "threshold_report_digest": threshold_report_digest,
    }
    _publish_bytes_file(
        result_path,
        canonical_json_bytes(
            {**unsigned_result, "signature_base64": signer.sign(unsigned_result)}
        ),
    )
    artifact_sha256 = {
        name: hashlib.sha256(
            _regular_file(
                directory / name,
                label="defense-v1 fixture",
                max_bytes=300 * 1024 * 1024,
            )
        ).hexdigest()
        for name in sorted(_DEFENSE_V1_FIXTURE_FILES)
    }
    unsigned_hash_manifest: dict[str, object] = {
        "artifact_sha256": artifact_sha256,
        "preregistration_sha256": _DEFENSE_V1_PREREGISTRATION_SHA256,
        "profile_sha256": _DEFENSE_V1_PROFILE_SHA256,
        "public_key_base64": signer.public_key_base64,
        "schema_version": "1.0.0",
        "signer_key_id": signer.key_id,
    }
    _publish_bytes_file(
        hash_manifest_path,
        canonical_json_bytes(
            {
                **unsigned_hash_manifest,
                "signature_base64": signer.sign(unsigned_hash_manifest),
            }
        ),
    )


def _export_defense_v1_evaluation(
    *,
    directory: Path,
    result_path: Path,
    hash_manifest_path: Path,
    store: ArtifactStore,
    signer: RunSigningIdentity,
    public_artifacts: Mapping[str, ArtifactRef],
    evaluation_bundle_ref: ArtifactRef,
    threshold_set_ref: ArtifactRef,
    ensemble_ref: ArtifactRef,
    pooled_ref: ArtifactRef,
    held_family_refs: Mapping[Family, ArtifactRef],
    corpus_envelope_ref: ArtifactRef,
    run_ledger_sha256: str,
    promotion_envelope_digest: str,
    descriptor_scope: tuple[str, ...],
    status: str,
    defender_frozen_at: datetime,
    hidden_released_at: datetime,
    hidden_release_attestation: HiddenReleaseAttestation,
    hidden_signer_key_id: str,
    hidden_signer_public_key_base64: str,
) -> None:
    """Publish exact public Task13 bytes, signed result, and hash manifest last."""
    if type(hidden_release_attestation) is not HiddenReleaseAttestation:
        raise CliContractError("hidden release attestation inputs differ")
    _assert_preregistered_authority(
        "publication",
        key_id=signer.key_id,
        public_key_base64=signer.public_key_base64,
    )
    _assert_preregistered_authority(
        "hidden_evaluator",
        key_id=hidden_signer_key_id,
        public_key_base64=hidden_signer_public_key_base64,
    )
    try:
        hidden_verifier = EvaluatorReplayVerifier(
            signer_key_id=hidden_release_attestation.signer_key_id,
            public_key_base64=hidden_release_attestation.public_key_base64,
        )
    except (TypeError, ValueError) as error:
        raise CliContractError("hidden public authority identity differs") from error
    if (
        type(store) is not ArtifactStore
        or type(signer) is not RunSigningIdentity
        or set(public_artifacts) != _DEFENSE_V1_PUBLIC_REPORT_FILES
        or hidden_released_at <= defender_frozen_at
        or hidden_release_attestation.authority_issued_at
        != _utc_wire(hidden_released_at)
        or hidden_release_attestation.signer_key_id != hidden_signer_key_id
        or hidden_release_attestation.public_key_base64
        != hidden_signer_public_key_base64
        or hidden_release_attestation.ensemble_top_ref_digest != ensemble_ref.sha256
        or hidden_release_attestation.pooled_defender_ref_digest != pooled_ref.sha256
        or hidden_release_attestation.candidate_roster_digest
        != _candidate_roster_digest(pooled_ref, held_family_refs)
        or hidden_release_attestation.profile_sha256 != _DEFENSE_V1_PROFILE_SHA256
        or hidden_release_attestation.evaluation_bundle_digest
        != evaluation_bundle_ref.sha256
        or hidden_release_attestation.scorecard_digest
        != public_artifacts["defense-scorecard.json"].sha256
        or hidden_release_attestation.promotion_envelope_digest
        != promotion_envelope_digest
        or not hidden_verifier.verify_document(
            hidden_release_attestation.unsigned_document(),
            hidden_release_attestation.signature_base64,
        )
        or status not in {"promoted", "retained", "no_promotion"}
    ):
        raise CliContractError("defense-v1 evaluation export inputs differ")
    for value in (run_ledger_sha256, promotion_envelope_digest):
        _digest(value)
    expected_preexisting = (
        _DEFENSE_V1_FIXTURE_FILES - _DEFENSE_V1_PUBLIC_REPORT_FILES
    )
    try:
        present_entries = tuple(directory.iterdir())
        present_names = {path.name for path in present_entries}
        result_path.lstat()
    except FileNotFoundError:
        result_absent = True
    except OSError as error:
        raise CliContractError("defense-v1 export preflight failed") from error
    else:
        result_absent = False
    if (
        present_names != expected_preexisting
        or any(path.is_symlink() or not path.is_file() for path in present_entries)
        or not result_absent
        or hash_manifest_path.parent != directory
        or hash_manifest_path.name != "hash-manifest.json"
    ):
        raise CliContractError(
            "defense-v1 export preflight found a collision or incomplete training set"
        )
    verifier = PublicArtifactVerifier.from_signer(signer)
    try:
        verified_bundle = load_evaluation_bundle(
            evaluation_bundle_ref,
            artifact_store=store,
            verifier=verifier,
        )
        verified_public_artifacts = {
            name: reference.as_artifact_ref()
            for name, reference in verified_bundle.public_artifacts.items()
        }
        scorecard = verified_bundle.scorecard(
            artifact_store=store,
            verifier=verifier,
        )
    except ReportingContractError as error:
        raise CliContractError("defense-v1 evaluation bundle failed verification") from error
    if (
        verified_public_artifacts != dict(public_artifacts)
        or scorecard.champion_decision.status.value != status
    ):
        raise CliContractError("defense-v1 public evaluation lineage differs")
    for name, reference in sorted(public_artifacts.items()):
        if type(reference) is not ArtifactRef or reference.sha256 != _digest(reference.sha256):
            raise CliContractError("defense-v1 public artifact reference differs")
        _publish_bytes_file(directory / name, store.read(reference))
    evaluation_bundle_payload = store.read(evaluation_bundle_ref)
    unsigned_result: dict[str, object] = {
        "authority_identities": {
            role: asdict(_preregistered_authority(role))
            for role in (
                "development_evaluator",
                "hidden_evaluator",
                "hidden_source",
                "publication",
            )
        },
        "campaign_count": 200,
        "champion_decision": scorecard.champion_decision.model_dump(mode="json"),
        "corpus_envelope": _reference_document(corpus_envelope_ref),
        "defender_ensemble": _reference_document(ensemble_ref),
        "defender_frozen_at": _utc_wire(defender_frozen_at),
        "descriptor_scope": list(descriptor_scope),
        "evaluation_bundle": _reference_document(evaluation_bundle_ref),
        "evaluation_bundle_payload_base64": base64.b64encode(
            evaluation_bundle_payload
        ).decode("ascii"),
        "failures_remain_public": True,
        "hidden_released_at": _utc_wire(hidden_released_at),
        "hidden_release_attestation": hidden_release_attestation.model_dump(
            mode="json"
        ),
        "preregistration_sha256": _DEFENSE_V1_PREREGISTRATION_SHA256,
        "profile_sha256": _DEFENSE_V1_PROFILE_SHA256,
        "promotion_envelope_digest": promotion_envelope_digest,
        "public_artifacts": {
            name: _reference_document(reference)
            for name, reference in sorted(public_artifacts.items())
        },
        "public_key_base64": signer.public_key_base64,
        "retuned_after_hidden": False,
        "run_ledger_sha256": run_ledger_sha256,
        "schema_version": "1.0.0",
        "signer_key_id": signer.key_id,
        "status": status,
        "synthetic_only": True,
        "threshold_set": _reference_document(threshold_set_ref),
    }
    result_document = {
        **unsigned_result,
        "signature_base64": signer.sign(unsigned_result),
    }
    _publish_bytes_file(result_path, canonical_json_bytes(result_document))
    present = {
        path.name
        for path in directory.iterdir()
        if path.is_file() and not path.is_symlink()
    }
    if present != _DEFENSE_V1_FIXTURE_FILES:
        raise CliContractError("defense-v1 fixture set is incomplete before hashing")
    artifact_sha256 = {
        name: hashlib.sha256(
            _regular_file(
                directory / name,
                label="defense-v1 fixture",
                max_bytes=300 * 1024 * 1024,
            )
        ).hexdigest()
        for name in sorted(_DEFENSE_V1_FIXTURE_FILES)
    }
    unsigned_hash_manifest: dict[str, object] = {
        "artifact_sha256": artifact_sha256,
        "preregistration_sha256": _DEFENSE_V1_PREREGISTRATION_SHA256,
        "profile_sha256": _DEFENSE_V1_PROFILE_SHA256,
        "public_key_base64": signer.public_key_base64,
        "schema_version": "1.0.0",
        "signer_key_id": signer.key_id,
    }
    _publish_bytes_file(
        hash_manifest_path,
        canonical_json_bytes(
            {
                **unsigned_hash_manifest,
                "signature_base64": signer.sign(unsigned_hash_manifest),
            }
        ),
    )


def _defense_v1_repository_path(raw: str, expected: str) -> Path:
    """Resolve one exact preregistered public output below the pinned repository."""
    if type(raw) is not str or raw != expected or Path(raw).is_absolute():
        raise CliContractError("defense-v1 export path differs from preregistration")
    candidate = _REPOSITORY_ROOT / raw
    candidate.parent.mkdir(mode=0o755, parents=True, exist_ok=True)
    if candidate.parent.resolve(strict=True) != candidate.parent:
        raise CliContractError("defense-v1 export parent must not contain symlinks")
    return candidate


def _defense_v1_input_path(raw: str, expected: str) -> Path:
    if type(raw) is not str or raw != expected or Path(raw).is_absolute():
        raise CliContractError("defense-v1 input path differs from preregistration")
    candidate = _REPOSITORY_ROOT / raw
    try:
        if candidate.resolve(strict=True) != candidate:
            raise CliContractError("defense-v1 input must not contain symlinks")
    except OSError as error:
        raise CliContractError("defense-v1 input is unavailable") from error
    return candidate


def _defense_v1_root(raw: str) -> Path:
    if raw != ".apar/defense-v1":
        raise CliContractError("defense-v1 named root differs from preregistration")
    return _secure_root(_REPOSITORY_ROOT / raw)


def _defense_v1_export_directory(raw: str) -> Path:
    if raw != "fixtures/defense/v1":
        raise CliContractError("defense-v1 export directory differs from preregistration")
    directory = _REPOSITORY_ROOT / raw
    directory.mkdir(mode=0o755, parents=True, exist_ok=True)
    if directory.resolve(strict=True) != directory:
        raise CliContractError("defense-v1 export directory must not contain symlinks")
    return directory


def _defense_v1_signer_path(root: Path, raw: str) -> Path:
    if raw != ".apar/defense-v1/run-signing-key.ed25519":
        raise CliContractError("defense-v1 signer alias differs from preregistration")
    return _signer_path(root, str(root / "run-signing.key"))


def _load_run_ledger_reference(
    *,
    store: ArtifactStore,
    reference: ArtifactRef,
    profile: CompetitionProfile,
    signer: RunSigningIdentity,
) -> CompetitionRunLedger:
    try:
        payload = store.read(reference)
        document = strict_json_loads(payload)
        ledger = CompetitionRunLedger.model_validate(document)
    except (ValidationError, WireContractError, ValueError) as error:
        raise CliContractError("defense-v1 run ledger is invalid") from error
    if (
        canonical_json_bytes(ledger.model_dump(mode="json")) != payload
        or ledger.profile_sha256 != hashlib.sha256(profile.to_json()).hexdigest()
        or ledger.signer_key_id != signer.key_id
        or ledger.public_key_base64 != signer.public_key_base64
    ):
        raise CliContractError("defense-v1 run ledger lineage differs")
    return ledger


def _publish_defense_v1_alias(
    *,
    path: Path,
    kind: Literal[
        "run_ledger",
        "corpus_envelope",
        "defender_ensemble",
        "development_completion",
        "infeasible_candidate",
    ],
    reference: ArtifactRef,
    signer: RunSigningIdentity,
    authenticated_run_ids: tuple[str, ...],
    export_metadata: dict[str, object] | None = None,
) -> DefenseV1SignedAlias:
    unsigned = {
        "artifact": _reference_document(reference),
        "authenticated_run_ids": list(authenticated_run_ids),
        "campaign_count": 200,
        "export_metadata": {} if export_metadata is None else export_metadata,
        "family_counts": {family: 50 for family in _FAMILIES},
        "kind": kind,
        "preregistration_sha256": _DEFENSE_V1_PREREGISTRATION_SHA256,
        "profile_sha256": _DEFENSE_V1_PROFILE_SHA256,
        "public_key_base64": signer.public_key_base64,
        "schema_version": "1.0.0",
        "signer_key_id": signer.key_id,
    }
    alias = DefenseV1SignedAlias.model_validate(
        {**unsigned, "signature_base64": signer.sign(unsigned)}
    )
    _publish_json_file(path.parent, path.name, alias.model_dump(mode="json"))
    return alias


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
    if command == "provision_defense_authorities":
        parser.add_argument("--root", required=True)
    elif command == "generate_defense_runs":
        source = parser.add_mutually_exclusive_group(required=True)
        source.add_argument("--profile")
        source.add_argument("--preregistration")
        parser.add_argument("--root", required=True)
        parser.add_argument("--signer", required=True)
        output = parser.add_mutually_exclusive_group(required=True)
        output.add_argument("--output-ledger")
        output.add_argument("--output")
    elif command == "build_defense_corpus":
        parser.add_argument("--profile", required=True)
        parser.add_argument("--run-manifests", required=True)
        parser.add_argument("--root", required=True)
        parser.add_argument("--output-manifest", required=True)
    elif command == "train_defender":
        corpus = parser.add_mutually_exclusive_group(required=True)
        corpus.add_argument("--development-corpus")
        corpus.add_argument("--corpus")
        parser.add_argument("--catalog", required=True)
        parser.add_argument("--profile", required=True)
        parser.add_argument("--root", required=True)
        parser.add_argument("--rollback-ref", required=True)
        parser.add_argument("--export")
    elif command == "prepare_hidden_context":
        parser.add_argument("--corpus", required=True)
        parser.add_argument("--defender", required=True)
        parser.add_argument("--profile", required=True)
        parser.add_argument("--root", required=True)
    elif command == "evaluate_defender":
        parser.add_argument("--phase", choices=("development", "hidden"), required=True)
        parser.add_argument("--defender", required=True)
        parser.add_argument("--corpus")
        parser.add_argument("--export")
        parser.add_argument("--hash-manifest")
        parser.add_argument("--result")
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
        if command == "provision_defense_authorities":
            identities = provision_defense_authorities(_defense_v1_root(args.root))
            _json_stdout(
                {
                    "authority_identities": identities,
                    "schema_version": "1.0.0",
                }
            )
            return 0
        named_generate = (
            command == "generate_defense_runs" and args.preregistration is not None
        )
        named_build = (
            command == "build_defense_corpus"
            and args.run_manifests
            == "docs/experiments/defense-v1-run-manifests.json"
        )
        named_train = command == "train_defender" and args.corpus is not None
        named_prepare = command == "prepare_hidden_context"
        named_evaluate = command == "evaluate_defender" and (
            args.corpus is not None
            or args.export is not None
            or args.hash_manifest is not None
            or args.result is not None
        )
        named = (
            named_generate
            or named_build
            or named_train
            or named_prepare
            or named_evaluate
        )
        if named_generate:
            preregistration = load_defense_v1_preregistration(
                _defense_v1_input_path(
                    args.preregistration,
                    "docs/experiments/defense-v1-preregistration.json",
                )
            )
            profile = preregistration.profile
        else:
            profile_path = (
                _defense_v1_input_path(
                    args.profile, "config/defense/competition-profile.json"
                )
                if named
                else Path(args.profile)
            )
            profile = load_competition_profile(profile_path, competition=True)
        root = _defense_v1_root(args.root) if named else _secure_root(Path(args.root))
        profile_digest = hashlib.sha256(profile.to_json()).hexdigest()
        if command == "generate_defense_runs":
            if named_generate != (args.output is not None):
                raise CliContractError(
                    "defense-v1 preregistration and named output must be paired"
                )
            reference = _generate_competition_runs(
                profile=profile,
                root=root,
                signer_path=(
                    _defense_v1_signer_path(root, args.signer)
                    if named_generate
                    else _signer_path(root, args.signer)
                ),
                output_name=(
                    "defense-v1-run-ledger.json"
                    if named_generate
                    else args.output_ledger
                ),
                enforce_preregistered_authorities=named_generate,
            )
            if named_generate:
                store = ArtifactStore(root / "artifacts")
                signer = _load_standard_signer(root)
                ledger = _load_run_ledger_reference(
                    store=store,
                    reference=reference,
                    profile=profile,
                    signer=signer,
                )
                output_path = _defense_v1_repository_path(
                    args.output,
                    "docs/experiments/defense-v1-run-manifests.json",
                )
                _publish_defense_v1_alias(
                    path=output_path,
                    kind="run_ledger",
                    reference=reference,
                    signer=signer,
                    authenticated_run_ids=tuple(item.run_id for item in ledger.entries),
                )
            output_document: dict[str, object] = {
                "artifact": _reference_document(reference),
                "campaign_count": 200,
                "profile_sha256": profile_digest,
                "schema_version": "1.0.0",
            }
            if named_generate:
                output_document["preregistration_sha256"] = (
                    _DEFENSE_V1_PREREGISTRATION_SHA256
                )
            _json_stdout(output_document)
            return 0
        if command == "build_defense_corpus":
            if named_build:
                store = ArtifactStore(root / "artifacts")
                signer = _load_standard_signer(root)
                ledger_alias_path = _defense_v1_input_path(
                    args.run_manifests,
                    "docs/experiments/defense-v1-run-manifests.json",
                )
                ledger_ref = resolve_defense_v1_alias(
                    ledger_alias_path,
                    expected_kind="run_ledger",
                    expected_profile_sha256=profile_digest,
                    signer=signer,
                )
                _load_run_ledger_reference(
                    store=store,
                    reference=ledger_ref,
                    profile=profile,
                    signer=signer,
                )
                ledger_digest = ledger_ref.sha256
                output_name = "defense-v1-corpus-envelope.json"
            else:
                ledger_digest = args.run_manifests
                output_name = args.output_manifest
            reference = _build_competition_corpus(
                profile=profile,
                root=root,
                ledger_digest=ledger_digest,
                output_name=output_name,
                enforce_preregistered_authorities=named_build,
            )
            if named_build:
                envelope, corpus = _load_corpus_envelope(
                    ArtifactStore(root / "artifacts"),
                    reference.sha256,
                    profile,
                    _load_standard_signer(root),
                )
                output_path = _defense_v1_repository_path(
                    args.output_manifest,
                    "fixtures/defense/v1/corpus-manifest.json",
                )
                _export_defense_v1_corpus(
                    path=output_path,
                    reference=reference,
                    envelope=envelope,
                    corpus=corpus,
                    signer=_load_standard_signer(root),
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
            if named_train:
                if args.export is None:
                    raise CliContractError(
                        "defense-v1 named training requires the exact export directory"
                    )
                signer = _load_standard_signer(root)
                corpus_alias = _load_defense_v1_alias(
                    _defense_v1_input_path(
                        args.corpus, "fixtures/defense/v1/corpus-manifest.json"
                    ),
                    expected_kind="corpus_envelope",
                    expected_profile_sha256=profile_digest,
                    signer=signer,
                )
                corpus_reference = _artifact_ref(corpus_alias.artifact)
                ArtifactStore(root / "artifacts").read(corpus_reference)
                corpus_digest = corpus_reference.sha256
                catalog_path = _defense_v1_input_path(
                    args.catalog, "config/defense/feature-catalog.json"
                )
                if args.rollback_ref != "rules-v1":
                    raise CliContractError("defense-v1 rollback alias differs")
                rollback_ref = "rules-v1"
            else:
                if args.export is not None:
                    raise CliContractError(
                        "digest-addressed training cannot use named export arguments"
                    )
                corpus_digest = args.development_corpus
                catalog_path = Path(args.catalog)
                rollback_ref = args.rollback_ref
            try:
                reference = _train_competition_defender(
                    profile=profile,
                    root=root,
                    corpus_envelope_digest=corpus_digest,
                    catalog_path=catalog_path,
                    rollback_ref=rollback_ref,
                    enforce_preregistered_authorities=named_train,
                )
            except _CompetitionThresholdInfeasible as failure:
                if not named_train:
                    raise
                run_ledger_alias = _load_defense_v1_alias(
                    _defense_v1_input_path(
                        "docs/experiments/defense-v1-run-manifests.json",
                        "docs/experiments/defense-v1-run-manifests.json",
                    ),
                    expected_kind="run_ledger",
                    expected_profile_sha256=profile_digest,
                    signer=signer,
                )
                _export_defense_v1_threshold_failure(
                    directory=_defense_v1_export_directory(cast(str, args.export)),
                    result_path=_defense_v1_repository_path(
                        "docs/experiments/defense-v1-result.json",
                        "docs/experiments/defense-v1-result.json",
                    ),
                    hash_manifest_path=_defense_v1_repository_path(
                        "fixtures/defense/v1/hash-manifest.json",
                        "fixtures/defense/v1/hash-manifest.json",
                    ),
                    root=root,
                    signer=signer,
                    artifacts=failure.artifacts,
                    corpus_envelope_ref=corpus_reference,
                    authenticated_run_ids=corpus_alias.authenticated_run_ids,
                    run_ledger_sha256=_artifact_ref(run_ledger_alias.artifact).sha256,
                )
                _json_stdout(
                    {
                        "failure_reason": "operating_budget_infeasible",
                        "profile_sha256": profile_digest,
                        "schema_version": "1.0.0",
                        "status": "no_promotion",
                    }
                )
                return 0
            if named_train:
                _export_defense_v1_defender(
                    directory=_defense_v1_export_directory(cast(str, args.export)),
                    reference=reference,
                    root=root,
                    profile=profile,
                    signer=_load_standard_signer(root),
                    authenticated_run_ids=corpus_alias.authenticated_run_ids,
                )
            _json_stdout(
                {
                    "frozen_defender": _reference_document(reference),
                    "profile_sha256": profile_digest,
                    "schema_version": "1.0.0",
                }
            )
            return 0
        if command == "prepare_hidden_context":
            signer = _load_standard_signer(root)
            corpus_alias = _load_defense_v1_alias(
                _defense_v1_input_path(
                    args.corpus, "fixtures/defense/v1/corpus-manifest.json"
                ),
                expected_kind="corpus_envelope",
                expected_profile_sha256=profile_digest,
                signer=signer,
            )
            defender_alias = _load_defense_v1_alias(
                _defense_v1_input_path(
                    args.defender, "fixtures/defense/v1/defender-bundle.json"
                ),
                expected_kind="defender_ensemble",
                expected_profile_sha256=profile_digest,
                signer=signer,
            )
            if (
                corpus_alias.authenticated_run_ids
                != defender_alias.authenticated_run_ids
            ):
                raise CliContractError("hidden preparation run lineage differs")
            corpus_reference = _artifact_ref(corpus_alias.artifact)
            ensemble_reference = _artifact_ref(defender_alias.artifact)
            store = ArtifactStore(root / "artifacts")
            store.read(corpus_reference)
            store.read(ensemble_reference)
            reference = _prepare_competition_hidden_context(
                profile=profile,
                root=root,
                corpus_envelope_ref=corpus_reference,
                ensemble_ref=ensemble_reference,
                development_run_ids=corpus_alias.authenticated_run_ids,
                enforce_preregistered_authorities=True,
            )
            _json_stdout(
                {
                    "hidden_context_pointer_sha256": reference.sha256,
                    "profile_sha256": profile_digest,
                    "schema_version": "1.0.0",
                }
            )
            return 0
        if command == "evaluate_defender":
            store = ArtifactStore(root / "artifacts")
            named_corpus_ref: ArtifactRef | None = None
            if named_evaluate:
                if args.corpus is None or args.export is None:
                    raise CliContractError(
                        "defense-v1 named evaluation requires corpus and export"
                    )
                if args.phase == "development" and (
                    args.hash_manifest is not None
                    or args.result is not None
                    or args.development_scorecard is not None
                    or args.hidden_corpus is not None
                ):
                    raise CliContractError(
                        "development evaluation cannot resolve hidden release inputs"
                    )
                export_directory = _defense_v1_export_directory(cast(str, args.export))
                if args.phase == "hidden":
                    if args.hash_manifest is None or args.result is None:
                        raise CliContractError(
                            "hidden defense-v1 evaluation requires result and hash outputs"
                        )
                    hash_manifest_path = _defense_v1_repository_path(
                        args.hash_manifest,
                        "fixtures/defense/v1/hash-manifest.json",
                    )
                    result_path = _defense_v1_repository_path(
                        args.result,
                        "docs/experiments/defense-v1-result.json",
                    )
                signer = _load_standard_signer(root)
                defender_alias = _load_defense_v1_alias(
                    _defense_v1_input_path(
                        args.defender, "fixtures/defense/v1/defender-bundle.json"
                    ),
                    expected_kind="defender_ensemble",
                    expected_profile_sha256=profile_digest,
                    signer=signer,
                )
                corpus_alias = _load_defense_v1_alias(
                    _defense_v1_input_path(
                        args.corpus, "fixtures/defense/v1/corpus-manifest.json"
                    ),
                    expected_kind="corpus_envelope",
                    expected_profile_sha256=profile_digest,
                    signer=signer,
                )
                if (
                    defender_alias.authenticated_run_ids
                    != corpus_alias.authenticated_run_ids
                ):
                    raise CliContractError("defense-v1 alias run lineage differs")
                top_ref = _artifact_ref(defender_alias.artifact)
                named_corpus_ref = _artifact_ref(corpus_alias.artifact)
                store.read(top_ref)
                store.read(named_corpus_ref)
                defender_digest = top_ref.sha256
            else:
                defender_digest = _digest(args.defender)
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
            if named_evaluate:
                _assert_preregistered_authority(
                    "publication",
                    key_id=signer.key_id,
                    public_key_base64=signer.public_key_base64,
                )
                _assert_preregistered_authority(
                    "development_evaluator",
                    key_id=evaluator_signer.key_id,
                    public_key_base64=evaluator_signer.public_key_base64,
                )
                _assert_preregistered_authority(
                    "hidden_source",
                    key_id=ensemble.hidden_source_signer_key_id,
                    public_key_base64=ensemble.hidden_source_public_key_base64,
                )
            corpus_envelope_ref = _artifact_ref(ensemble.corpus_envelope_ref)
            if named_corpus_ref is not None and corpus_envelope_ref != named_corpus_ref:
                raise CliContractError("defense-v1 defender and corpus aliases differ")
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
                verify_hidden_context,
            )

            hidden_source_binding: HiddenSourceWorkerBinding | None = None
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
                if named_evaluate:
                    _assert_preregistered_authority(
                        "hidden_evaluator",
                        key_id=hidden_signer.key_id,
                        public_key_base64=hidden_signer.public_key_base64,
                    )
            hidden_pointer: HiddenContextPointer | None = None
            if args.phase == "development":
                hidden_context_ref = None
            else:
                hidden_pointer = _load_hidden_context_pointer(
                    root=root,
                    store=store,
                    digest=cast(str, args.hidden_corpus),
                    signer=signer,
                    profile_sha256=profile_digest,
                    ensemble_ref=top_ref,
                    corpus_envelope_ref=corpus_envelope_ref,
                )
                hidden_context_ref = _artifact_ref(
                    hidden_pointer.hidden_context_ref
                )
            if hidden_context_ref is not None:
                checked_hidden_envelope, _, restricted_hidden_ref = (
                    verify_hidden_context(
                        store=store,
                        envelope_ref=hidden_context_ref,
                        signer=cast(RunSigningIdentity, hidden_context_signer),
                        profile_sha256=profile_digest,
                        development_corpus_digest=frozen_corpus_digest(corpus),
                        development_event_ids=tuple(
                            row.event_id for row in corpus.observations
                        ),
                    )
                )
                hidden_candidate_refs = (
                    defender_ref,
                    *(
                        _artifact_ref(ensemble.held_family_refs[family])
                        for family in profile.families
                    ),
                )
                hidden_frozen_times: list[datetime] = []
                hidden_publisher = DefenderBundlePublisher(
                    store, signer, _REPOSITORY_ROOT
                )
                try:
                    for candidate_ref in hidden_candidate_refs:
                        candidate = hidden_publisher.load(candidate_ref)
                        candidate.verify_reload()
                        hidden_frozen_times.append(candidate.manifest.frozen_at)
                finally:
                    hidden_publisher.close()
                assert hidden_pointer is not None
                source_receipt_ref = _artifact_ref(
                    hidden_pointer.source_receipt_ref
                )
                if (
                    checked_hidden_envelope.source_lineage_digest
                    != source_receipt_ref.sha256
                ):
                    raise CliContractError(
                        "hidden context pointer source lineage differs"
                    )
                hidden_source_binding = _verify_hidden_source_metadata(
                    store=store,
                    profile=profile,
                    source_signer_key_id=ensemble.hidden_source_signer_key_id,
                    source_public_key_base64=(
                        ensemble.hidden_source_public_key_base64
                    ),
                    source_receipt_ref=source_receipt_ref,
                    ensemble_ref=top_ref,
                    development_corpus=corpus,
                    restricted_context_ref=restricted_hidden_ref,
                    authority_as_of=checked_hidden_envelope.as_of,
                    maximum_frozen_at=max(hidden_frozen_times),
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
                hidden_source_binding=hidden_source_binding,
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
            elif named_evaluate:
                if (
                    published.hidden_released_at is None
                    or published.hidden_public_proof is None
                ):
                    raise CliContractError("hidden release timing is unavailable")
                held_refs = {
                    family: _artifact_ref(ensemble.held_family_refs[family])
                    for family in profile.families
                }
                candidate_refs = (
                    defender_ref,
                    *(held_refs[family] for family in profile.families),
                )
                frozen_times: list[datetime] = []
                frozen_publisher = DefenderBundlePublisher(
                    store, signer, _REPOSITORY_ROOT
                )
                try:
                    for candidate_ref in candidate_refs:
                        candidate = frozen_publisher.load(candidate_ref)
                        candidate.verify_reload()
                        frozen_times.append(candidate.manifest.frozen_at)
                finally:
                    frozen_publisher.close()
                defender_frozen_at = max(frozen_times)
                hidden_release_attestation = _build_hidden_release_attestation(
                    proof=published.hidden_public_proof,
                    signer=cast(EvaluatorSigningIdentity, hidden_signer),
                    ensemble_ref=top_ref,
                    pooled_ref=defender_ref,
                    held_family_refs=held_refs,
                    evaluation_bundle_ref=published.evaluation_bundle_ref,
                    scorecard_ref=published.scorecard_ref,
                    promotion_envelope_digest=published.promotion_envelope_digest,
                    defender_frozen_at=defender_frozen_at,
                )
                _export_defense_v1_evaluation(
                    directory=export_directory,
                    result_path=result_path,
                    hash_manifest_path=hash_manifest_path,
                    store=store,
                    signer=signer,
                    public_artifacts=published.public_artifacts,
                    evaluation_bundle_ref=published.evaluation_bundle_ref,
                    threshold_set_ref=published.threshold_set_ref,
                    ensemble_ref=top_ref,
                    pooled_ref=defender_ref,
                    held_family_refs=held_refs,
                    corpus_envelope_ref=corpus_envelope_ref,
                    run_ledger_sha256=corpus_envelope.run_ledger_sha256,
                    promotion_envelope_digest=published.promotion_envelope_digest,
                    descriptor_scope=published.descriptor_scope,
                    status=published.champion_decision.status.value,
                    defender_frozen_at=defender_frozen_at,
                    hidden_released_at=published.hidden_released_at,
                    hidden_release_attestation=hidden_release_attestation,
                    hidden_signer_key_id=cast(
                        EvaluatorSigningIdentity, hidden_signer
                    ).key_id,
                    hidden_signer_public_key_base64=cast(
                        EvaluatorSigningIdentity, hidden_signer
                    ).public_key_base64,
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
