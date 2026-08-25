"""Closed execution-mode bindings for Sentinel v5 evidence runs."""

from __future__ import annotations

import hashlib
import json
from enum import StrEnum
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from apar.evaluation.v5_evidence_protocol import V5EvidenceProtocol
from apar.evaluation.v5_protocol import V5DevelopmentProtocol, V5Profile


class V5RunMode(StrEnum):
    SAFE_VALIDATION = "safe_validation"
    LOCKED_DEVELOPMENT = "locked_development"


class V5RunBinding(BaseModel):
    """One of the only two allowed seed/profile combinations."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    mode: V5RunMode
    profile: V5Profile
    development_test_seed: int
    repeatable: bool
    authorization_required: bool

    @model_validator(mode="after")
    def binding_is_closed(self) -> Self:
        expected = {
            V5RunMode.SAFE_VALIDATION: (V5Profile.SMOKE, 404, True, False),
            V5RunMode.LOCKED_DEVELOPMENT: (
                V5Profile.PRODUCTION,
                2404,
                False,
                True,
            ),
        }[self.mode]
        observed = (
            self.profile,
            self.development_test_seed,
            self.repeatable,
            self.authorization_required,
        )
        if observed != expected:
            raise ValueError("run mode profile/seed/authorization binding differs")
        return self


class V5PartitionSupportPlan(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    partition: str
    legitimate_rows: int = Field(gt=0)
    fraud_rows_by_family: tuple[tuple[str, int], ...]
    total_rows: int = Field(gt=0)
    execution_artifacts: int = Field(gt=0)
    execution_payload_estimate_bytes: int = Field(gt=0)

    @model_validator(mode="after")
    def totals_reconcile(self) -> Self:
        if self.total_rows != self.legitimate_rows + sum(
            count for _family, count in self.fraud_rows_by_family
        ):
            raise ValueError("partition support totals do not reconcile")
        return self


class V5RunSupportPlan(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    mode: V5RunMode
    profile: V5Profile
    partitions: tuple[V5PartitionSupportPlan, ...]
    retained_execution_artifacts: int = Field(gt=0)
    retained_execution_payload_estimate_bytes: int = Field(gt=0)
    support_plan_sha256: str

    @model_validator(mode="after")
    def digest_matches(self) -> Self:
        document = self.model_dump(mode="json", exclude={"support_plan_sha256"})
        expected = hashlib.sha256(
            json.dumps(
                document,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode()
        ).hexdigest()
        if self.support_plan_sha256 != expected:
            raise ValueError("run support-plan digest mismatch")
        return self


class V5LockedEvidenceRunBinding(BaseModel):
    """Content-addressed SOURCE/preregistration/run contract for one locked run."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: str
    mode: V5RunMode
    profile: V5Profile
    development_test_seed: int
    source_commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    source_tree_oid: str = Field(pattern=r"^[0-9a-f]{40}$")
    preregistration_commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    preregistration_path: str
    preregistration_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    base_protocol_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    arm_protocol_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    evidence_protocol_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    implementation_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    catalog_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    support_plan: V5RunSupportPlan
    candidate_manifest_path: str
    storage_schema_version: str
    payload_schema_version: str
    run_binding_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @staticmethod
    def compute_digest(values: object) -> str:
        if isinstance(values, BaseModel):
            document = values.model_dump(
                mode="json", exclude={"run_binding_sha256"}
            )
        elif isinstance(values, dict):
            document = {
                key: (
                    value.model_dump(mode="json")
                    if isinstance(value, BaseModel)
                    else value
                )
                for key, value in values.items()
                if key != "run_binding_sha256"
            }
        else:
            raise TypeError("run binding digest requires a model or object")
        return hashlib.sha256(
            json.dumps(
                document,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            ).encode()
        ).hexdigest()

    @field_validator("schema_version")
    @classmethod
    def schema_is_exact(cls, value: str) -> str:
        if value != "apar-sentinel-v5-locked-run-binding/1":
            raise ValueError("locked run-binding schema differs")
        return value

    @model_validator(mode="after")
    def locked_binding_is_exact(self) -> Self:
        if (
            self.mode is not V5RunMode.LOCKED_DEVELOPMENT
            or self.profile is not V5Profile.PRODUCTION
            or self.development_test_seed != 2404
            or self.support_plan.mode is not V5RunMode.LOCKED_DEVELOPMENT
            or self.support_plan.profile is not V5Profile.PRODUCTION
        ):
            raise ValueError("locked run mode/profile/seed/support binding differs")
        if self.preregistration_path != (
            "config/defense/defense-v5-locked-development-preregistration.json"
        ):
            raise ValueError("locked preregistration path differs")
        if self.candidate_manifest_path != (
            "docs/experiments/defense-v5-locked-development-candidate.manifest.json"
        ):
            raise ValueError("locked candidate manifest path differs")
        if self.storage_schema_version != "apar-sentinel-v5-chunked-evidence/2":
            raise ValueError("locked storage schema differs")
        if self.payload_schema_version != (
            "apar-sentinel-v5-locked-development-payload/2"
        ):
            raise ValueError("locked payload schema differs")
        if self.run_binding_sha256 != self.compute_digest(self):
            raise ValueError("locked run-binding digest mismatch")
        return self


def resolve_v5_run_mode(
    *,
    mode: V5RunMode,
    evidence_protocol: V5EvidenceProtocol,
    development_protocol: V5DevelopmentProtocol,
) -> V5RunBinding:
    """Resolve one closed run mode against the frozen seed sources."""
    if development_protocol.seeds.development_test != 2404:
        raise ValueError("development protocol locked seed differs from 2404")
    if (
        evidence_protocol.safe_development_test_seed != 404
        or evidence_protocol.locked_development_test_seed != 2404
    ):
        raise ValueError("evidence protocol safe/locked seed bindings differ")
    if mode is V5RunMode.SAFE_VALIDATION:
        return V5RunBinding(
            mode=mode,
            profile=V5Profile.SMOKE,
            development_test_seed=404,
            repeatable=True,
            authorization_required=False,
        )
    return V5RunBinding(
        mode=mode,
        profile=V5Profile.PRODUCTION,
        development_test_seed=2404,
        repeatable=False,
        authorization_required=True,
    )


_FAMILY_EVENT_ROWS = {
    "agentic_intent_abuse": 25,
    "app_scam_mule": 36,
    "card_testing_cnp": 26,
    "synthetic_merchant_refund": 46,
}
_FAMILY_FRAUD_EVENT_ROWS = {
    "agentic_intent_abuse": 23,
    "app_scam_mule": 24,
    "card_testing_cnp": 17,
    "synthetic_merchant_refund": 34,
}
_FAMILY_ARTIFACT_ESTIMATES = {
    "agentic_intent_abuse": 365_536,
    "app_scam_mule": 140_768,
    "card_testing_cnp": 110_768,
    "synthetic_merchant_refund": 170_768,
}
_RETAINED_PARTITIONS = ("train", "calibration", "threshold", "development_test")
_LEGITIMATE_BASE_DECISION_ROWS = 24


def _legitimate_artifact_plan(count: int) -> tuple[int, int]:
    # Real execution projects all 24 base decisions: 12 card, 10 A2A, and two
    # agentic. Filler batches consume the remaining requested decision support.
    base_decisions = _LEGITIMATE_BASE_DECISION_ROWS
    base_artifacts = 3
    base_estimate = 221_072
    remaining = count - base_decisions
    if remaining < 0:
        raise ValueError("legitimate support cannot cover the three base rails")
    full_batches, final_events = divmod(remaining, 96)
    filler_artifacts = full_batches + (1 if final_events else 0)
    filler_estimate = full_batches * 320_768
    if final_events:
        filler_estimate += 32_768 + final_events * 3_000
    return base_artifacts + filler_artifacts, base_estimate + filler_estimate


def build_v5_run_support_plan(
    *,
    mode: V5RunMode,
    evidence_protocol: V5EvidenceProtocol,
    development_protocol: V5DevelopmentProtocol,
) -> V5RunSupportPlan:
    """Derive the exact retained support without executing any population."""
    binding = resolve_v5_run_mode(
        mode=mode,
        evidence_protocol=evidence_protocol,
        development_protocol=development_protocol,
    )
    counts = (
        development_protocol.smoke_profile
        if mode is V5RunMode.SAFE_VALIDATION
        else development_protocol.production_profile
    )
    if (
        mode is V5RunMode.LOCKED_DEVELOPMENT
        and counts.campaigns_per_family
        != development_protocol.production_dev_test_campaigns_per_family
    ):
        raise ValueError("production campaign plan differs across frozen fields")
    partitions: list[V5PartitionSupportPlan] = []
    for partition in _RETAINED_PARTITIONS:
        operational_legitimate = (
            development_protocol.production_dev_test_legitimate
            if mode is V5RunMode.LOCKED_DEVELOPMENT
            and partition == "development_test"
            else counts.legitimate_decisions // 4
        )
        campaigns = counts.campaigns_per_family
        fraud = tuple(
            (family, campaigns[family] * fraud_rows)
            for family, fraud_rows in sorted(_FAMILY_FRAUD_EVENT_ROWS.items())
        )
        campaign_controls = sum(
            campaigns[family]
            * (_FAMILY_EVENT_ROWS[family] - _FAMILY_FRAUD_EVENT_ROWS[family])
            for family in _FAMILY_EVENT_ROWS
        )
        legitimate = operational_legitimate + campaign_controls
        legitimate_artifacts, legitimate_estimate = _legitimate_artifact_plan(
            operational_legitimate
        )
        fraud_artifacts = sum(campaigns.values())
        fraud_estimate = sum(
            campaigns[family] * estimate
            for family, estimate in _FAMILY_ARTIFACT_ESTIMATES.items()
        )
        partitions.append(
            V5PartitionSupportPlan(
                partition=partition,
                legitimate_rows=legitimate,
                fraud_rows_by_family=fraud,
                total_rows=legitimate + sum(value for _family, value in fraud),
                execution_artifacts=legitimate_artifacts + fraud_artifacts,
                execution_payload_estimate_bytes=legitimate_estimate
                + fraud_estimate,
            )
        )
    values = {
        "mode": mode,
        "profile": binding.profile,
        "partitions": tuple(partitions),
        "retained_execution_artifacts": sum(
            item.execution_artifacts for item in partitions
        ),
        "retained_execution_payload_estimate_bytes": sum(
            item.execution_payload_estimate_bytes for item in partitions
        ),
    }
    document = {
        key: (
            value.value
            if isinstance(value, StrEnum)
            else [item.model_dump(mode="json") for item in value]
            if isinstance(value, tuple)
            else value
        )
        for key, value in values.items()
    }
    values["support_plan_sha256"] = hashlib.sha256(
        json.dumps(
            document,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ).hexdigest()
    return V5RunSupportPlan.model_validate(values)


__all__ = [
    "V5PartitionSupportPlan",
    "V5LockedEvidenceRunBinding",
    "V5RunBinding",
    "V5RunMode",
    "V5RunSupportPlan",
    "build_v5_run_support_plan",
    "resolve_v5_run_mode",
]
