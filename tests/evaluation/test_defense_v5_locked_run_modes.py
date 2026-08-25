"""Closed run-mode contracts for Sentinel v5 evidence execution."""

from __future__ import annotations

import ast
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def _legacy_profile_choices() -> tuple[str, ...]:
    """Return the literal argparse choices without executing the legacy runner."""
    source = ROOT / "scripts/run_defense_v5_development.py"
    tree = ast.parse(source.read_text())
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr != "add_argument" or not node.args:
            continue
        first = node.args[0]
        if not isinstance(first, ast.Constant) or first.value != "--profile":
            continue
        choices = next(
            (
                keyword.value
                for keyword in node.keywords
                if keyword.arg == "choices"
            ),
            None,
        )
        if not isinstance(choices, (ast.List, ast.Tuple)):
            raise AssertionError("legacy --profile choices must be a literal closed list")
        return tuple(
            str(item.value)
            for item in choices.elts
            if isinstance(item, ast.Constant)
        )
    raise AssertionError("legacy runner does not declare --profile")


def test_legacy_summary_runner_cannot_select_production() -> None:
    """Adding production back to the summary CLI would consume the locked seed."""
    assert _legacy_profile_choices() == ("smoke",)


def test_closed_run_mode_module_exists() -> None:
    """Without a closed contract, callers can still combine arbitrary seed/profile values."""
    assert importlib.util.find_spec("apar.evaluation.v5_run_mode") is not None


def test_closed_run_modes_bind_exact_profile_and_seed() -> None:
    """Changing either binding must prevent safe/locked mode relabeling."""
    from apar.evaluation.v5_evidence_protocol import load_v5_evidence_protocol
    from apar.evaluation.v5_protocol import load_v5_development_protocol
    from apar.evaluation.v5_run_mode import V5RunMode, resolve_v5_run_mode

    evidence = load_v5_evidence_protocol(
        ROOT / "config/defense/defense-v5-evidence.json", root=ROOT
    )
    development = load_v5_development_protocol(
        ROOT / "config/defense/defense-v5-development.json"
    )

    safe = resolve_v5_run_mode(
        mode=V5RunMode.SAFE_VALIDATION,
        evidence_protocol=evidence,
        development_protocol=development,
    )
    assert (safe.mode.value, safe.profile.value, safe.development_test_seed) == (
        "safe_validation",
        "smoke",
        404,
    )
    assert safe.repeatable is True
    assert safe.authorization_required is False

    locked = resolve_v5_run_mode(
        mode=V5RunMode.LOCKED_DEVELOPMENT,
        evidence_protocol=evidence,
        development_protocol=development,
    )
    assert (locked.mode.value, locked.profile.value, locked.development_test_seed) == (
        "locked_development",
        "production",
        2404,
    )
    assert locked.repeatable is False
    assert locked.authorization_required is True

    with pytest.raises(ValueError):
        V5RunMode("production")


def test_evidence_protocol_freezes_modes_and_chunked_storage() -> None:
    """Omitting these fields would leave production mode and artifact durability mutable."""
    from apar.evaluation.v5_evidence_protocol import load_v5_evidence_protocol

    evidence = load_v5_evidence_protocol(
        ROOT / "config/defense/defense-v5-evidence.json", root=ROOT
    )
    assert evidence.schema_version == "1.2.0"
    assert evidence.run_modes.safe_validation.model_dump(mode="json") == {
        "profile": "smoke",
        "development_test_seed": 404,
        "repeatable": True,
        "authorization_required": False,
    }
    assert evidence.run_modes.locked_development.model_dump(mode="json") == {
        "profile": "production",
        "development_test_seed": 2404,
        "repeatable": False,
        "authorization_required": True,
    }
    storage = evidence.locked_artifact_storage
    assert storage.schema_version == "apar-sentinel-v5-chunked-evidence/2"
    assert storage.attempt_receipt_path == (
        "docs/experiments/defense-v5-locked-development-attempt.json"
    )
    assert storage.candidate_manifest_path == (
        "docs/experiments/defense-v5-locked-development-candidate.manifest.json"
    )
    assert storage.judge_summary_path == (
        "docs/experiments/defense-v5-locked-development-summary.json"
    )
    assert storage.chunk_size_bytes == 64 * 1024 * 1024
    assert storage.expected_envelope_upper_bound_bytes == 768 * 1024 * 1024
    assert storage.maximum_envelope_bytes == 1024 * 1024 * 1024
    assert storage.maximum_chunk_count == 16
    assert storage.normal_git_blob_limit_bytes == 100 * 1024 * 1024
    assert storage.chunk_size_bytes < storage.normal_git_blob_limit_bytes


def test_locked_support_plan_is_exact_without_executing_population() -> None:
    """A smoke-sized or partial support plan must not qualify as production evidence."""
    from apar.evaluation.v5_evidence_protocol import load_v5_evidence_protocol
    from apar.evaluation.v5_protocol import load_v5_development_protocol
    from apar.evaluation.v5_run_mode import V5RunMode, build_v5_run_support_plan

    evidence = load_v5_evidence_protocol(
        ROOT / "config/defense/defense-v5-evidence.json", root=ROOT
    )
    development = load_v5_development_protocol(
        ROOT / "config/defense/defense-v5-development.json"
    )
    plan = build_v5_run_support_plan(
        mode=V5RunMode.LOCKED_DEVELOPMENT,
        evidence_protocol=evidence,
        development_protocol=development,
    )
    expected = {
        "train": (16_000, 25_800, 534),
        "calibration": (16_000, 25_800, 534),
        "threshold": (16_000, 25_800, 534),
        "development_test": (53_500, 63_300, 924),
    }
    assert {
        partition.partition: (
            partition.legitimate_rows,
            partition.total_rows,
            partition.execution_artifacts,
        )
        for partition in plan.partitions
    } == expected
    development_test = plan.partitions[-1]
    assert dict(development_test.fraud_rows_by_family) == {
        "agentic_intent_abuse": 2_300,
        "app_scam_mule": 2_400,
        "card_testing_cnp": 1_700,
        "synthetic_merchant_refund": 3_400,
    }
    assert plan.retained_execution_artifacts == 2_526
    assert plan.retained_execution_payload_estimate_bytes == 608_386_240
    assert len(plan.support_plan_sha256) == 64


def test_controls_accept_only_the_seed_for_the_closed_mode() -> None:
    """Keeping the safe-only guard would make a locked full-evidence run impossible."""
    from apar.evaluation.v5_controls import validate_v5_control_run_mode
    from apar.evaluation.v5_evidence_protocol import load_v5_evidence_protocol
    from apar.evaluation.v5_protocol import (
        load_v5_development_protocol,
        v5_protocol_digest,
    )
    from apar.evaluation.v5_run_mode import V5RunMode

    evidence = load_v5_evidence_protocol(
        ROOT / "config/defense/defense-v5-evidence.json", root=ROOT
    )
    locked = load_v5_development_protocol(
        ROOT / "config/defense/defense-v5-development.json"
    )
    safe = locked.model_copy(
        update={
            "seeds": locked.seeds.model_copy(update={"development_test": 404}),
            "protocol_sha256": "",
        }
    )
    safe = safe.model_copy(update={"protocol_sha256": v5_protocol_digest(safe)})

    validate_v5_control_run_mode(
        protocol=safe,
        evidence_protocol=evidence,
        mode=V5RunMode.SAFE_VALIDATION,
    )
    validate_v5_control_run_mode(
        protocol=locked,
        evidence_protocol=evidence,
        mode=V5RunMode.LOCKED_DEVELOPMENT,
    )
    with pytest.raises(ValueError, match="run mode|seed"):
        validate_v5_control_run_mode(
            protocol=locked,
            evidence_protocol=evidence,
            mode=V5RunMode.SAFE_VALIDATION,
        )
    with pytest.raises(ValueError, match="run mode|seed"):
        validate_v5_control_run_mode(
            protocol=safe,
            evidence_protocol=evidence,
            mode=V5RunMode.LOCKED_DEVELOPMENT,
        )


@pytest.mark.parametrize(
    ("mode", "profile", "seed"),
    [
        ("safe_validation", "production", 404),
        ("safe_validation", "smoke", 2404),
        ("locked_development", "smoke", 2404),
        ("locked_development", "production", 404),
    ],
)
def test_run_binding_rejects_safe_locked_relabeling(
    mode: str, profile: str, seed: int
) -> None:
    """Relaxing binding validation would let safe evidence masquerade as locked."""
    from apar.evaluation.v5_run_mode import V5RunBinding

    with pytest.raises(ValueError, match="run mode|profile|seed"):
        V5RunBinding(
            mode=mode,
            profile=profile,
            development_test_seed=seed,
            repeatable=mode == "safe_validation",
            authorization_required=mode == "locked_development",
        )
