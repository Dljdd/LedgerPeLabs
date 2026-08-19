"""Deterministic online review-capacity simulation contracts."""

from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime, timedelta
from time import perf_counter
from typing import Any

import pytest
from pydantic import ValidationError

from apar.cases import (
    CaseAlertEvidence,
    CaseMotif,
    CaseSnapshot,
    InvestigationCase,
    QueueConfig,
    QueueContractError,
    QueueReport,
    simulate_case_queue,
)
from apar.contracts.decisions import Action
from apar.runs.wire import canonical_json_bytes
from tests.cases.conftest import NOW


def case(
    index: int,
    *,
    opened_at: datetime = NOW,
    priority: float = 50.0,
    estimated_minutes: int = 20,
    **updates: Any,
) -> InvestigationCase:
    event_id = f"event-{index}"
    case_id = "case-" + hashlib.sha256(
        canonical_json_bytes(
            {"domain": "apar-investigation-case-v1", "first_evidence": [event_id]}
        )
    ).hexdigest()
    values: dict[str, Any] = {
        "case_id": case_id,
        "opened_at": opened_at,
        "event_ids": (event_id,),
        "actor_ids": (f"actor-{index}",),
        "counterparty_ids": (f"counterparty-{index}",),
        "first_alert_at": opened_at,
        "priority": priority,
        "estimated_minutes": estimated_minutes,
        "first_evidence_ids": (event_id,),
        "alert_evidence": (
            CaseAlertEvidence(
                event_id=event_id,
                decision_at=opened_at,
                actor_id=f"actor-{index}",
                counterparty_id=f"counterparty-{index}",
                motif=CaseMotif.ISOLATED,
                visible_value_before_alert="100.00",
                latest_graph_evidence_at=opened_at - timedelta(seconds=1),
                score=0.8,
                action=Action.CHALLENGE,
                evidence_source_ids=(event_id,),
            ),
        ),
    }
    values.update(updates)
    return InvestigationCase(**values)


def test_empty_queue_has_canonical_zero_report() -> None:
    report = simulate_case_queue((), QueueConfig())
    assert report.arrival_count == 0
    assert report.completed_count == 0
    assert report.snapshots == ()
    assert report.analyst_minutes == 0
    assert report.peak_backlog_count == 0
    assert report.sla_breach_count == 0
    assert QueueReport.from_json(report.to_json()) == report


def test_exact_hourly_capacity_carries_seventh_case_to_next_bucket() -> None:
    config = QueueConfig(analyst_count=2, service_minutes_per_case=20, bucket_minutes=60)
    report = simulate_case_queue(tuple(case(index) for index in range(7)), config)

    starts = tuple(snapshot.started_at for snapshot in report.snapshots)
    assert starts == (
        NOW,
        NOW,
        NOW + timedelta(minutes=20),
        NOW + timedelta(minutes=20),
        NOW + timedelta(minutes=40),
        NOW + timedelta(minutes=40),
        NOW + timedelta(minutes=60),
    )
    assert report.analyst_minutes == 140
    assert report.peak_backlog_count == 5


def test_same_time_priority_orders_work_and_case_id_breaks_ties() -> None:
    cases = (
        case(3, priority=30.0),
        case(2, priority=90.0),
        case(1, priority=90.0),
    )
    report = simulate_case_queue(cases, QueueConfig(analyst_count=1))
    assert tuple(snapshot.case_id for snapshot in report.snapshots) == (
        case(1).case_id,
        case(2).case_id,
        case(3).case_id,
    )


def test_later_case_cannot_change_any_earlier_snapshot() -> None:
    config = QueueConfig(analyst_count=1, service_minutes_per_case=20)
    earlier = tuple(case(index) for index in range(5))
    before = simulate_case_queue(earlier, config)
    future_case = case(99, opened_at=NOW + timedelta(minutes=5), priority=100.0)
    after = simulate_case_queue(earlier + (future_case,), config)
    assert after.snapshots[: len(before.snapshots)] == before.snapshots


def test_wait_and_sla_use_exact_strict_boundary() -> None:
    report = simulate_case_queue(
        tuple(case(index) for index in range(4)),
        QueueConfig(
            analyst_count=1,
            service_minutes_per_case=20,
            sla_minutes=40,
            bucket_minutes=60,
        ),
    )
    assert tuple(snapshot.wait_minutes for snapshot in report.snapshots) == (0, 20, 40, 60)
    assert tuple(snapshot.sla_breached for snapshot in report.snapshots) == (
        False,
        False,
        False,
        True,
    )
    assert report.sla_breach_count == 1


def test_arrival_on_service_boundary_uses_that_boundary() -> None:
    opened = NOW + timedelta(minutes=20)
    report = simulate_case_queue((case(1, opened_at=opened),), QueueConfig(analyst_count=1))
    assert report.snapshots[0].started_at == opened
    assert report.snapshots[0].wait_minutes == 0


def test_non_hour_bucket_alignment_is_anchored_in_utc_not_wall_clock_hour() -> None:
    opened = NOW + timedelta(hours=1, minutes=20)
    report = simulate_case_queue(
        (case(1, opened_at=opened, estimated_minutes=18),),
        QueueConfig(
            analyst_count=1,
            service_minutes_per_case=18,
            bucket_minutes=90,
        ),
    )
    assert report.snapshots[0].started_at == NOW + timedelta(hours=1, minutes=30)


def test_queue_is_input_permutation_stable() -> None:
    cases = tuple(case(index, priority=float(index)) for index in range(5))
    assert simulate_case_queue(cases, QueueConfig()) == simulate_case_queue(
        tuple(reversed(cases)), QueueConfig()
    )


def test_report_is_immutable_canonical_and_digest_bound() -> None:
    report = simulate_case_queue((case(1),), QueueConfig())
    with pytest.raises(ValidationError):
        report.completed_count = 2  # type: ignore[misc]
    with pytest.raises(ValidationError):
        report.snapshots[0].wait_minutes = 1  # type: ignore[misc]
    payload = report.to_json()
    assert payload == json.dumps(
        json.loads(payload), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    tampered = json.loads(payload)
    tampered["analyst_minutes"] += 1
    tampered_payload = json.dumps(
        tampered, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    with pytest.raises(QueueContractError, match="digest|analyst"):
        QueueReport.from_json(tampered_payload)


@pytest.mark.parametrize("mutation", ["duplicate", "reorder"])
def test_report_rejects_recomputed_digest_over_noncanonical_snapshots(mutation: str) -> None:
    report = simulate_case_queue(
        (case(1, priority=90.0), case(2, priority=50.0)), QueueConfig(analyst_count=1)
    )
    document = report.model_dump(mode="json")
    snapshots = list(document["snapshots"])
    if mutation == "duplicate":
        snapshots[1] = snapshots[0]
    else:
        snapshots.reverse()
    document["snapshots"] = snapshots
    document["report_digest"] = hashlib.sha256(
        canonical_json_bytes(
            {key: value for key, value in document.items() if key != "report_digest"}
        )
    ).hexdigest()

    with pytest.raises(ValidationError, match="unique|canonical"):
        QueueReport.model_validate(document)


@pytest.mark.parametrize(
    "updates",
    [
        {"analyst_count": 0},
        {"service_minutes_per_case": 0},
        {"sla_minutes": 0},
        {"bucket_minutes": 0},
        {"service_minutes_per_case": 40, "bucket_minutes": 60},
    ],
)
def test_queue_config_rejects_invalid_or_inexact_capacity(updates: dict[str, int]) -> None:
    with pytest.raises(ValidationError):
        QueueConfig(**updates)


def test_queue_config_rejects_boolean_integer_coercion() -> None:
    with pytest.raises(ValidationError):
        QueueConfig(analyst_count=True)  # type: ignore[arg-type]


def test_queue_rejects_duplicate_cases_or_service_fixture_mismatch() -> None:
    row = case(1)
    with pytest.raises(QueueContractError, match="duplicate case_id"):
        simulate_case_queue((row, row), QueueConfig())
    with pytest.raises(QueueContractError, match="estimated_minutes"):
        simulate_case_queue((case(2, estimated_minutes=10),), QueueConfig())


def test_case_and_snapshot_contracts_reject_nonfinite_or_incoherent_values() -> None:
    with pytest.raises(ValidationError):
        case(1, priority=math.nan)
    with pytest.raises(ValidationError):
        case(1, opened_at=datetime(2026, 8, 18, 12, 0))
    with pytest.raises(ValidationError):
        CaseSnapshot(
            case_id=f"case-{1:064x}",
            opened_at=NOW,
            started_at=NOW - timedelta(minutes=1),
            completed_at=NOW,
            wait_minutes=0,
            sla_breached=False,
            analyst_minutes=20,
            priority=50.0,
        )


def test_report_rejects_redigested_impossible_simultaneous_starts() -> None:
    report = simulate_case_queue(
        (case(1), case(2)),
        QueueConfig(analyst_count=1, service_minutes_per_case=20),
    )
    document = report.model_dump(mode="json")
    snapshots = list(document["snapshots"])
    snapshots[1]["started_at"] = snapshots[0]["started_at"]
    snapshots[1]["completed_at"] = snapshots[0]["completed_at"]
    snapshots[1]["wait_minutes"] = snapshots[0]["wait_minutes"]
    document["snapshots"] = snapshots
    document["peak_backlog_count"] = 0
    document["report_digest"] = hashlib.sha256(
        canonical_json_bytes(
            {key: value for key, value in document.items() if key != "report_digest"}
        )
    ).hexdigest()
    payload = json.dumps(document, sort_keys=True, separators=(",", ":")).encode()

    with pytest.raises(QueueContractError, match="schedule|reconstruct|snapshot"):
        QueueReport.from_json(payload)


def test_report_binds_case_arrivals_and_snapshot_identifiers() -> None:
    report = simulate_case_queue((case(1),), QueueConfig())
    assert report.case_inputs == (case(1),)
    document = report.model_dump(mode="json")
    document["snapshots"][0]["case_id"] = f"case-{9:064x}"
    document["report_digest"] = hashlib.sha256(
        canonical_json_bytes(
            {key: value for key, value in document.items() if key != "report_digest"}
        )
    ).hexdigest()

    with pytest.raises(ValidationError, match="case|snapshot|schedule"):
        QueueReport.model_validate(document)


def test_queue_revalidates_model_constructed_case_and_config() -> None:
    valid_case = case(1)
    forged_case = InvestigationCase.model_construct(
        **{**valid_case.model_dump(mode="python"), "priority": math.nan}
    )
    forged_config = QueueConfig.model_construct(
        analyst_count=True,
        service_minutes_per_case=20,
        sla_minutes=240,
        bucket_minutes=60,
    )

    with pytest.raises(QueueContractError, match="InvestigationCase|semantic|priority"):
        simulate_case_queue((forged_case,), QueueConfig())
    with pytest.raises(QueueContractError, match="QueueConfig|semantic|integer"):
        simulate_case_queue((valid_case,), forged_config)


def test_sweep_line_backlog_meets_frozen_benchmark_ceiling() -> None:
    row_count = 3_000
    rows = tuple(
        case(index, opened_at=NOW + timedelta(seconds=index))
        for index in range(row_count)
    )

    started = perf_counter()
    report = simulate_case_queue(rows, QueueConfig(analyst_count=2))
    elapsed = perf_counter() - started

    assert report.arrival_count == row_count
    assert elapsed < 2.5


def test_queue_normalizes_datetime_overflow_and_bounds_configuration() -> None:
    near_limit = datetime(9999, 12, 31, 23, 59, tzinfo=NOW.tzinfo)
    with pytest.raises(QueueContractError, match="resource|bounds|overflow"):
        simulate_case_queue((case(1, opened_at=near_limit),), QueueConfig())
    with pytest.raises(ValidationError, match="less than or equal"):
        QueueConfig(analyst_count=100_001)


def test_report_loader_rejects_payload_before_unbounded_json_parse(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from apar.cases import queue

    monkeypatch.setattr(queue, "_MAX_QUEUE_PAYLOAD_BYTES", 64)

    with pytest.raises(QueueContractError, match="payload|resource"):
        QueueReport.from_json(b"x" * 65)
