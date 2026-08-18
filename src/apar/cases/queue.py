"""Frozen-capacity, online review queue simulation."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from typing import Literal

from pydantic import Field, ValidationError, field_validator, model_validator

from apar.cases.grouping import InvestigationCase
from apar.contracts._validation import ExternalContract, validate_utc_timestamp
from apar.runs.wire import WireContractError, canonical_json_bytes, strict_json_loads


class QueueContractError(ValueError):
    """Queue inputs or serialized evidence violate the closed contract."""


class QueueConfig(ExternalContract):
    """Frozen synthetic analyst-capacity assumptions."""

    schema_version: Literal["1.0.0"] = "1.0.0"
    analyst_count: int = Field(default=2, ge=1)
    service_minutes_per_case: int = Field(default=20, ge=1)
    sla_minutes: int = Field(default=240, ge=1)
    bucket_minutes: int = Field(default=60, ge=1)

    @field_validator(
        "analyst_count",
        "service_minutes_per_case",
        "sla_minutes",
        "bucket_minutes",
        mode="before",
    )
    @classmethod
    def counts_are_exact_integers(cls, value: object) -> object:
        if type(value) is not int:
            raise ValueError("queue configuration counts must be exact integers")
        return value

    @model_validator(mode="after")
    def capacity_has_exact_service_slots(self) -> QueueConfig:
        if self.service_minutes_per_case > self.bucket_minutes:
            raise ValueError("service time must not exceed the capacity bucket")
        if self.bucket_minutes % self.service_minutes_per_case != 0:
            raise ValueError("capacity bucket must contain an exact number of service slots")
        return self


class CaseSnapshot(ExternalContract):
    """Immutable start/completion evidence for one review case."""

    schema_version: Literal["1.0.0"] = "1.0.0"
    case_id: str
    opened_at: datetime
    started_at: datetime
    completed_at: datetime
    wait_minutes: float = Field(ge=0.0)
    sla_breached: bool
    analyst_minutes: int = Field(ge=1)
    priority: float = Field(ge=0.0, le=100.0)

    @field_validator("opened_at", "started_at", "completed_at")
    @classmethod
    def timestamps_are_utc(cls, value: datetime) -> datetime:
        return validate_utc_timestamp(value)

    @field_validator("wait_minutes", "priority")
    @classmethod
    def metrics_are_finite(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("queue snapshot metrics must be finite")
        return value

    @model_validator(mode="after")
    def chronology_is_coherent(self) -> CaseSnapshot:
        if self.started_at < self.opened_at:
            raise ValueError("case cannot start before it opens")
        if self.completed_at != self.started_at + timedelta(minutes=self.analyst_minutes):
            raise ValueError("completion must equal start plus analyst minutes")
        expected_wait = (self.started_at - self.opened_at).total_seconds() / 60.0
        if self.wait_minutes != expected_wait:
            raise ValueError("wait_minutes must equal start minus opening time")
        return self


class QueueReport(ExternalContract):
    """Digest-bound deterministic review workload evidence."""

    schema_version: Literal["1.0.0"] = "1.0.0"
    config: QueueConfig
    snapshots: tuple[CaseSnapshot, ...]
    arrival_count: int = Field(ge=0)
    completed_count: int = Field(ge=0)
    analyst_minutes: int = Field(ge=0)
    peak_backlog_count: int = Field(ge=0)
    sla_breach_count: int = Field(ge=0)
    report_digest: str

    @field_validator("report_digest")
    @classmethod
    def digest_is_sha256(cls, value: str) -> str:
        if len(value) != 64 or value != value.lower():
            raise ValueError("queue report digest must be lowercase SHA-256")
        try:
            int(value, 16)
        except ValueError as error:
            raise ValueError("queue report digest must be lowercase SHA-256") from error
        return value

    @model_validator(mode="after")
    def evidence_is_consistent(self) -> QueueReport:
        case_ids = tuple(item.case_id for item in self.snapshots)
        if len(set(case_ids)) != len(case_ids):
            raise ValueError("queue snapshot case IDs must be unique")
        expected_order = tuple(
            sorted(
                self.snapshots,
                key=lambda item: (item.opened_at, -item.priority, item.case_id),
            )
        )
        if self.snapshots != expected_order:
            raise ValueError("queue snapshots must use canonical causal order")
        if any(
            item.analyst_minutes != self.config.service_minutes_per_case
            for item in self.snapshots
        ):
            raise ValueError("snapshot service time must match the frozen fixture")
        if self.arrival_count != len(self.snapshots):
            raise ValueError("arrival count must equal snapshot count")
        if self.completed_count != len(self.snapshots):
            raise ValueError("completed count must equal snapshot count")
        if self.analyst_minutes != sum(item.analyst_minutes for item in self.snapshots):
            raise ValueError("analyst minutes must equal snapshot service time")
        if self.sla_breach_count != sum(item.sla_breached for item in self.snapshots):
            raise ValueError("SLA breach count must equal snapshot breaches")
        if any(
            item.sla_breached != (item.wait_minutes > self.config.sla_minutes)
            for item in self.snapshots
        ):
            raise ValueError("snapshot SLA status must use the frozen strict boundary")
        if self.peak_backlog_count != _peak_backlog(self.snapshots):
            raise ValueError("peak backlog must match causal arrival/start evidence")
        if self.report_digest != _report_digest(self):
            raise ValueError("queue report digest is inconsistent")
        return self

    def to_json(self) -> bytes:
        """Return canonical digest-bound queue evidence."""
        return canonical_json_bytes(self.model_dump(mode="json"))

    @classmethod
    def from_json(cls, payload: bytes) -> QueueReport:
        """Load exact canonical JSON and revalidate every derived field."""
        try:
            document = strict_json_loads(payload)
            if type(document) is not dict:
                raise QueueContractError("queue report JSON must contain an object")
            return cls.model_validate(document)
        except (WireContractError, ValidationError) as error:
            raise QueueContractError(str(error)) from error


def simulate_case_queue(
    cases: Sequence[InvestigationCase], config: QueueConfig
) -> QueueReport:
    """Allocate immutable service slots in causal arrival batches.

    Batches are processed by ``opened_at``. Within the same arrival instant,
    higher frozen priority wins and ``case_id`` is the final tie-break. Later
    arrivals never reorder earlier backlog, so appending future cases preserves
    every already-emitted snapshot.
    """
    if not isinstance(cases, (tuple, list)):
        raise TypeError("cases must be an exact row sequence")
    if type(config) is not QueueConfig:
        raise TypeError("config must be an exact QueueConfig")
    rows = tuple(cases)
    for row in rows:
        if type(row) is not InvestigationCase:
            raise TypeError("cases must contain exact InvestigationCase rows")
        if row.estimated_minutes != config.service_minutes_per_case:
            raise QueueContractError(
                "case estimated_minutes must equal the frozen service fixture"
            )
    identifiers = tuple(row.case_id for row in rows)
    if len(set(identifiers)) != len(identifiers):
        raise QueueContractError("duplicate case_id in queue input")
    ordered = tuple(sorted(rows, key=lambda row: (row.opened_at, -row.priority, row.case_id)))
    snapshots: list[CaseSnapshot] = []
    last_slot: datetime | None = None
    slot_uses = 0
    for row in ordered:
        earliest = _first_slot_at_or_after(row.opened_at, config)
        if (
            last_slot is not None
            and row.opened_at <= last_slot
            and slot_uses < config.analyst_count
        ):
            started_at = last_slot
            slot_uses += 1
        else:
            next_after_last = (
                earliest
                if last_slot is None
                else last_slot + timedelta(minutes=config.service_minutes_per_case)
            )
            started_at = max(earliest, next_after_last)
            last_slot = started_at
            slot_uses = 1
        wait_minutes = (started_at - row.opened_at).total_seconds() / 60.0
        snapshots.append(
            CaseSnapshot(
                case_id=row.case_id,
                opened_at=row.opened_at,
                started_at=started_at,
                completed_at=started_at
                + timedelta(minutes=config.service_minutes_per_case),
                wait_minutes=wait_minutes,
                sla_breached=wait_minutes > config.sla_minutes,
                analyst_minutes=config.service_minutes_per_case,
                priority=row.priority,
            )
        )
    snapshot_rows = tuple(snapshots)
    analyst_minutes = len(rows) * config.service_minutes_per_case
    peak_backlog_count = _peak_backlog(snapshot_rows)
    sla_breach_count = sum(item.sla_breached for item in snapshot_rows)
    document = {
        "schema_version": "1.0.0",
        "config": config,
        "snapshots": snapshot_rows,
        "arrival_count": len(rows),
        "completed_count": len(rows),
        "analyst_minutes": analyst_minutes,
        "peak_backlog_count": peak_backlog_count,
        "sla_breach_count": sla_breach_count,
    }
    digest_document = _json_document(document)
    return QueueReport(
        config=config,
        snapshots=snapshot_rows,
        arrival_count=len(rows),
        completed_count=len(rows),
        analyst_minutes=analyst_minutes,
        peak_backlog_count=peak_backlog_count,
        sla_breach_count=sla_breach_count,
        report_digest=hashlib.sha256(canonical_json_bytes(digest_document)).hexdigest(),
    )


def _first_slot_at_or_after(opened_at: datetime, config: QueueConfig) -> datetime:
    epoch = datetime(1970, 1, 1, tzinfo=UTC)
    elapsed_since_epoch = opened_at - epoch
    elapsed_microseconds = (
        (elapsed_since_epoch.days * 86_400 + elapsed_since_epoch.seconds) * 1_000_000
        + elapsed_since_epoch.microseconds
    )
    bucket_microseconds = config.bucket_minutes * 60 * 1_000_000
    bucket_index = elapsed_microseconds // bucket_microseconds
    bucket_start = epoch + timedelta(
        microseconds=bucket_index * bucket_microseconds
    )
    elapsed_in_bucket = elapsed_microseconds - bucket_index * bucket_microseconds
    service_microseconds = config.service_minutes_per_case * 60 * 1_000_000
    wave = (elapsed_in_bucket + service_microseconds - 1) // service_microseconds
    waves_per_bucket = config.bucket_minutes // config.service_minutes_per_case
    if wave >= waves_per_bucket:
        return bucket_start + timedelta(minutes=config.bucket_minutes)
    return bucket_start + timedelta(minutes=wave * config.service_minutes_per_case)


def _peak_backlog(snapshots: tuple[CaseSnapshot, ...]) -> int:
    if not snapshots:
        return 0
    openings = sorted({item.opened_at for item in snapshots})
    return max(
        sum(item.opened_at <= instant for item in snapshots)
        - sum(item.started_at <= instant for item in snapshots)
        for instant in openings
    )


def _json_document(value: object) -> object:
    if isinstance(value, ExternalContract):
        return value.model_dump(mode="json")
    if type(value) is dict:
        return {str(key): _json_document(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_json_document(item) for item in value]
    return value


def _report_digest(report: QueueReport) -> str:
    document = report.model_dump(mode="json", exclude={"report_digest"})
    return hashlib.sha256(canonical_json_bytes(document)).hexdigest()
