"""Closed-mode and stage-order contracts for the Sentinel v5 Kaggle recovery."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pytest
from pydantic import ValidationError

from apar.evaluation.v5_kaggle_protocol import (
    V5KaggleEnvironmentBinding,
    V5KaggleMode,
    V5KaggleStage,
    load_v5_kaggle_protocol,
    resolve_next_v5_kaggle_stage,
)

ROOT = Path(__file__).resolve().parents[2]


def _environment(python_version: str) -> V5KaggleEnvironmentBinding:
    return V5KaggleEnvironmentBinding.bind(
        provider="kaggle",
        image="kaggle-cpu",
        image_sha256="1" * 64,
        python_version=python_version,
        architecture="x86_64",
        cpu_count=4,
        dependency_manifest_sha256="2" * 64,
        source_archive_sha256="3" * 64,
        notebook_sha256="4" * 64,
        internet_enabled=False,
        accelerator="none",
        file_fsync_supported=True,
        directory_fsync_supported=True,
        hardlink_no_replace_supported=True,
    )


def test_environment_binds_observed_cpython_312_patch_not_local_patch() -> None:
    assert _environment("3.12.13").python_version == "3.12.13"
    with pytest.raises(ValueError):
        _environment("3.13.0")


CONFIG = ROOT / "config/defense/defense-v5-kaggle-recovery.json"


@dataclass(frozen=True)
class _Predecessor:
    stage: V5KaggleStage


def _write_mutation(tmp_path: Path, mutate: str) -> Path:
    document = json.loads(CONFIG.read_bytes())
    if mutate == "arbitrary_capacity_seed":
        document["capacity"]["development_test_seed"] = 405
    elif mutate == "locked_smoke_profile":
        document["locked"]["profile"] = "smoke"
    elif mutate == "safe_locked_relabel":
        document["capacity"]["mode"] = "kaggle_locked_successor"
    elif mutate == "weaken_memory_gate":
        document["resources"]["max_peak_rss_bytes"] = 20 * 1024**3
    elif mutate == "weaken_time_gate":
        document["resources"]["max_stage_seconds"] = 7 * 60 * 60
    elif mutate == "weaken_output_gate":
        document["resources"]["max_stage_output_bytes"] = 11_000_000_000
    elif mutate == "reorder_stages":
        document["stage_order"][0], document["stage_order"][1] = (
            document["stage_order"][1],
            document["stage_order"][0],
        )
    elif mutate == "duplicate_stage":
        document["stage_order"][-1] = document["stage_order"][-2]
    elif mutate == "unknown_field":
        document["unfrozen"] = True
    else:
        raise AssertionError(f"unknown test mutation: {mutate}")
    path = tmp_path / "mutated-kaggle-protocol.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


def test_kaggle_protocol_has_exact_closed_stage_order_and_modes() -> None:
    """A wrong stage, seed, profile, or gate must make the run inadmissible."""
    protocol = load_v5_kaggle_protocol(CONFIG, root=ROOT)

    assert protocol.stage_order == tuple(V5KaggleStage)
    assert protocol.capacity.mode is V5KaggleMode.CAPACITY_VALIDATION
    assert protocol.capacity.development_test_seed == 404
    assert protocol.capacity.profile == "production"
    assert protocol.capacity.repeatable is True
    assert protocol.capacity.authorization_required is False
    assert protocol.locked.mode is V5KaggleMode.LOCKED_SUCCESSOR
    assert protocol.locked.development_test_seed == 2404
    assert protocol.locked.profile == "production"
    assert protocol.locked.repeatable is False
    assert protocol.locked.authorization_required is True
    assert protocol.resources.max_peak_rss_bytes == 18 * 1024**3
    assert protocol.resources.max_stage_seconds == 6 * 60 * 60
    assert protocol.resources.max_stage_output_bytes == 10_000_000_000
    assert len(protocol.protocol_sha256) == 64
    assert protocol.run_binding_sha256(V5KaggleMode.CAPACITY_VALIDATION) != (
        protocol.run_binding_sha256(V5KaggleMode.LOCKED_SUCCESSOR)
    )


def test_invariance_controls_are_independent_resource_bounded_stages() -> None:
    """Recombining all invariance work must not recreate the 6-hour Stage-50 failure."""
    assert tuple(stage.value for stage in V5KaggleStage) == (
        "00_authorize",
        "10_corpus",
        "20_features",
        "30_arms",
        "40_label_shuffle",
        "50_identity_rename",
        "51_future_causality",
        "52_equal_time_isolation",
        "53_feature_leakage",
        "60_single_class_controls",
        "70_metrics",
        "80_finalize",
    )


def test_next_stage_is_derived_only_from_the_verified_predecessor() -> None:
    """A caller cannot skip, repeat, or choose an arbitrary stage."""
    assert resolve_next_v5_kaggle_stage(None) is V5KaggleStage.AUTHORIZE
    stages = tuple(V5KaggleStage)
    for current, expected in zip(stages[:-1], stages[1:], strict=True):
        assert resolve_next_v5_kaggle_stage(_Predecessor(current)) is expected
    with pytest.raises(ValueError, match="final stage"):
        resolve_next_v5_kaggle_stage(_Predecessor(V5KaggleStage.FINALIZE))


@pytest.mark.parametrize(
    "mutation",
    [
        "arbitrary_capacity_seed",
        "locked_smoke_profile",
        "safe_locked_relabel",
        "weaken_memory_gate",
        "weaken_time_gate",
        "weaken_output_gate",
        "reorder_stages",
        "duplicate_stage",
        "unknown_field",
    ],
)
def test_kaggle_protocol_rejects_nonfrozen_mode_stage_and_resource_mutations(
    tmp_path: Path,
    mutation: str,
) -> None:
    """Changing any closed execution choice must fail before stage authority exists."""
    with pytest.raises((ValidationError, ValueError)):
        load_v5_kaggle_protocol(_write_mutation(tmp_path, mutation), root=ROOT)


def test_kaggle_protocol_models_are_immutable() -> None:
    """Runtime code cannot retarget a validated protocol in memory."""
    protocol = load_v5_kaggle_protocol(CONFIG, root=ROOT)
    with pytest.raises(ValidationError):
        protocol.locked.development_test_seed = 404
