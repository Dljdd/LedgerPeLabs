"""G2: independent validity, matched policies, replay, and frozen capability evidence."""

from __future__ import annotations

import hashlib
import json
import stat
from pathlib import Path
from typing import cast

import pytest

from apar.compiler import compile_scenario
from apar.contracts.events import Rail
from apar.contracts.scenarios import ScenarioBundle
from apar.evaluation_hidden import HiddenCampaignGenerator, HiddenValidityOracle
from apar.runs import AttackerPolicy, AttackerPolicyKind, RunRunner, RunSigningIdentity
from apar.storage.artifacts import ArtifactStore
from tests.factories import make_scenario_config, make_threat_card

ROOT = Path(__file__).resolve().parents[2]
TASK6_RESULT = ROOT / "docs/experiments/task6-v3.4-holdout-result.json"
TASK6_RESULT_SHA256 = "f82981a987651a7f7ebb10a9011df063b2dc54a56181cae5b838e31de5e658db"
FAMILIES = (
    "agentic_intent_abuse",
    "app_scam_mule",
    "card_testing_cnp",
    "synthetic_merchant_refund",
)


def _bundle() -> ScenarioBundle:
    config = make_scenario_config(
        rail=Rail.A2A,
        query_budget=1,
        seed=960,
        replay=make_scenario_config().replay.model_copy(update={"random_seed": 960}),
        benign_entity_count=40,
        illicit_entity_count=16,
    )
    return compile_scenario(
        make_threat_card(rails=[Rail.A2A], default_config=config),
        config,
    )


@pytest.mark.parametrize("family", FAMILIES)
def test_g2_all_independent_hidden_families_are_valid_and_replay_identically(
    family: str,
) -> None:
    """Prove four separately generated hidden motifs are stable and independently valid."""
    generator = HiddenCampaignGenerator()
    oracle = HiddenValidityOracle()

    first = generator.generate(family, seed=260_816, count=8)
    second = generator.generate(family, seed=260_816, count=8)

    assert first == second
    assert oracle.evaluate(first).model_dump() == {"valid": True}
    with pytest.raises(PermissionError, match="only after run completion"):
        oracle.evaluate_restricted(first, run_complete=False)


def test_g2_disposable_policies_use_matched_budgets_and_seeded_bytes(tmp_path: Path) -> None:
    """Prove all four typed planners consume equal public budgets without hidden feedback."""
    store = ArtifactStore(tmp_path / "artifacts")
    runner = RunRunner(
        artifact_store=store,
        signer=RunSigningIdentity.from_private_bytes(bytes(range(32))),
        run_index_root=tmp_path / "runs",
    )
    bundle = _bundle()
    manifests = {}
    for kind in AttackerPolicyKind:
        manifest = runner.execute(
            bundle,
            AttackerPolicy(
                family="app_scam_mule",
                kind=kind,
                query_budget=1,
                worker_timeout_ms=2_000,
            ),
        )
        manifests[kind] = manifest
        feedback = cast(
            dict[str, object],
            json.loads(store.read(manifest.artifacts["feedback"])),
        )
        summary = cast(
            dict[str, object],
            json.loads(store.read(manifest.artifacts["summary"])),
        )
        assert feedback == {
            "history": feedback["history"],
            "logical_time_used": 1,
            "proposal_budget": 1,
            "proposals_used": 1,
            "queries_used": 1,
            "query_budget": 1,
        }
        assert "reason_codes" not in feedback
        assert "hidden" not in feedback
        assert summary["hidden_valid"] is True

    repeated = runner.execute(
        bundle,
        AttackerPolicy(
            family="app_scam_mule",
            kind=AttackerPolicyKind.FIXED,
            query_budget=1,
            worker_timeout_ms=2_000,
        ),
    )

    assert repeated == manifests[AttackerPolicyKind.FIXED]
    assert len({manifest.policy_kind for manifest in manifests.values()}) == 4


def test_g2_uses_accepted_frozen_task6_capability_evidence_without_rerun() -> None:
    """Bind the two-family GenAI claim to the exact accepted v3.4 result bytes."""
    metadata = TASK6_RESULT.lstat()
    raw = TASK6_RESULT.read_bytes()

    assert stat.S_ISREG(metadata.st_mode)
    assert not TASK6_RESULT.is_symlink()
    assert stat.S_IMODE(metadata.st_mode) == 0o600
    assert hashlib.sha256(raw).hexdigest() == TASK6_RESULT_SHA256

    result = cast(dict[str, object], json.loads(raw))
    protocol = cast(dict[str, object], result["protocol"])
    budgets = cast(dict[str, object], protocol["budgets"])
    policies = cast(dict[str, object], protocol["policies"])
    summary = cast(dict[str, object], result["summary"])
    families = cast(dict[str, object], summary["families"])
    cached_audit = cast(dict[str, object], summary["cached_llm_audit"])

    assert result["status"] == "executed_evidence_replication"
    assert policies == {
        "adaptive": "3.0.0",
        "cached_llm": "1.0.0",
        "fixed": "1.0.0",
        "random": "1.0.0",
    }
    assert budgets == {
        "logical_time": 24,
        "proposal": 24,
        "query": 24,
        "wall_time_ms": 120_000,
    }
    assert summary["matched_budgets"] is True
    assert summary["criterion_met"] is True
    assert summary["confirmatory_valid"] is True
    assert summary["supported_family_count"] == 2
    assert set(families) == {"app_scam_mule", "card_testing_cnp"}
    assert all(cast(dict[str, object], value)["supported"] is True for value in families.values())
    assert cached_audit["network_call_count"] == 0
    assert cached_audit["cache_success_count"] == cached_audit["attempt_count"] == 384
