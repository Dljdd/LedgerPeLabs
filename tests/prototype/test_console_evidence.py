"""Evidence-boundary tests for the judge-facing console document."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, cast

from scripts.build_apar_console_evidence import build_console_evidence, write_console_evidence


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def test_console_evidence_preserves_model_and_recovery_boundaries() -> None:
    document = build_console_evidence(_repo_root())
    portable = cast(dict[str, Any], document["portable"])
    recovered = cast(dict[str, Any], document["recovered"])
    readiness = cast(dict[str, Any], recovered["readiness"])

    assert portable["arm"] == "ensemble_with_graph"
    assert portable["authoritative"] is False
    assert portable["accepted_capacity_evidence"] is False
    assert recovered["qualifier"] == "Recovered diagnostic evidence — non-authoritative"
    assert recovered["authoritative"] is False
    assert recovered["accepted_capacity_evidence"] is False
    assert recovered["official_chain_status"] == "incomplete"
    assert recovered["first_missing_official_stage"] == "70_metrics"
    assert readiness["status"] == "not_ready"

    failed = {
        str(row["metric"])
        for row in cast(list[dict[str, Any]], recovered["failed_gates"])
    }
    assert {"false_decline_rate", "challenge_rate", "benign_only"} <= failed


def test_console_evidence_separates_truth_from_model_input_and_output() -> None:
    document = build_console_evidence(_repo_root())
    portable = cast(dict[str, Any], document["portable"])
    records = cast(list[dict[str, Any]], portable["records"])

    assert len(records) == 12
    for record in records:
        assert set(record) == {
            "accepted_checkpoint_evidence",
            "event_id",
            "model_input",
            "post_event_truth",
        }
        assert "presentation_ground_truth" not in cast(dict[str, Any], record["model_input"])
        assert set(cast(dict[str, Any], record["post_event_truth"])) == {
            "amount",
            "currency",
            "family",
            "label",
            "rail",
        }


def test_console_evidence_contains_real_deterministic_scenario_graph() -> None:
    document = build_console_evidence(_repo_root())
    context = cast(dict[str, Any], document["scenario_context"])
    graph = cast(dict[str, Any], context["graph"])

    assert context["seed"] == 260816
    assert context["family"] == "app_scam_mule"
    assert context["synthetic"] is True
    assert len(cast(list[object], graph["nodes"])) >= 3
    assert len(cast(list[object], graph["edges"])) >= 3
    assert len(cast(str, graph["graph_sha256"])) == 64
    assert len(cast(str, context["schedule_sha256"])) == 64
    assert context["ledger_conserved"] is True


def test_console_evidence_uses_exact_attempt_boundary_and_public_trust_proof() -> None:
    document = build_console_evidence(_repo_root())
    copy_boundary = cast(dict[str, Any], document["copy_boundary"])
    trust = cast(dict[str, Any], document["trust_proof"])
    encoded = json.dumps(document, sort_keys=True).lower()

    assert copy_boundary["evidence_seed"] == 404
    assert copy_boundary["kaggle_locked_successor_run"] is False
    assert copy_boundary["local_locked_attempt"] == "started_and_irreversibly_aborted"
    assert copy_boundary["published_successful_seed_2404_result"] is False
    assert copy_boundary["retry_permitted"] is False
    assert "private_key" not in encoded
    assert "seed 2404 was never executed" not in encoded
    assert [row["check"] for row in cast(list[dict[str, Any]], trust["checks"])] == [
        "identity",
        "mandate",
        "scope",
        "binding",
        "replay",
    ]


def test_console_evidence_write_is_canonical_and_byte_reproducible(tmp_path: Path) -> None:
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"

    first_document = write_console_evidence(_repo_root(), first)
    second_document = write_console_evidence(_repo_root(), second)

    assert first.read_bytes() == second.read_bytes()
    assert first_document == second_document
    without_digest = {
        key: value for key, value in first_document.items() if key != "document_sha256"
    }
    expected = hashlib.sha256(
        json.dumps(
            without_digest,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode()
    ).hexdigest()
    assert first_document["document_sha256"] == expected
