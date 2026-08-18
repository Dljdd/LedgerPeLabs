"""G2: independent validity, matched policies, replay, and frozen capability evidence."""

from __future__ import annotations

import hashlib
import json
import stat
import subprocess
from pathlib import Path
from typing import cast

import pytest

import apar.evaluation_hidden as hidden_evaluation
from apar.compiler import compile_scenario
from apar.contracts.events import Rail
from apar.contracts.scenarios import AttackerMode, ScenarioBundle
from apar.evaluation_hidden import HiddenCampaignGenerator, HiddenValidityOracle
from apar.runs import (
    AttackerPolicy,
    AttackerPolicyKind,
    RunRunner,
    RunSigningIdentity,
    bind_scenario_for_run,
)
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
        query_budget=3,
        seed=960,
        replay=make_scenario_config().replay.model_copy(update={"random_seed": 960}),
        benign_entity_count=40,
        illicit_entity_count=16,
    )
    card = make_threat_card(rails=[Rail.A2A], default_config=config)
    return bind_scenario_for_run(
        compile_scenario(card, config),
        threat_family=card.family,
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
    assert "RestrictedValidityReport" not in hidden_evaluation.__all__
    assert not hasattr(oracle, "evaluate_restricted")


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
    seed_sequences: dict[AttackerPolicyKind, list[int]] = {}
    for kind in AttackerPolicyKind:
        manifest = runner.execute(
            bundle,
            AttackerPolicy(
                attacker_mode=AttackerMode.DECISION_ONLY,
                family="app_scam_mule",
                kind=kind,
                query_budget=3,
                worker_timeout_ms=5_000,
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
            "logical_time_used": 3,
            "proposal_budget": 3,
            "proposals_used": 3,
            "queries_used": 3,
            "query_budget": 3,
        }
        history = cast(list[dict[str, object]], feedback["history"])
        assert [cast(dict[str, object], item["candidate"])["generation"] for item in history] == [
            0,
            1,
            2,
        ]
        assert cast(dict[str, object], history[1]["candidate"])["parent_id"] == cast(
            dict[str, object], history[0]["candidate"]
        )["candidate_id"]
        assert "reason_codes" not in feedback
        assert "hidden" not in feedback
        assert summary["hidden_valid"] is True
        audit = cast(
            dict[str, object],
            json.loads(store.read(manifest.artifacts["restricted_evaluation_audit"])),
        )
        proposals = cast(list[dict[str, object]], audit["policy_worker_proposals"])
        seed_sequences[kind] = [cast(int, proposal["proposal_seed"]) for proposal in proposals]
        if kind is AttackerPolicyKind.CACHED_LLM:
            assert proposals == [
                {
                    "cache_hit": True,
                    "cache_source": "task6-v3-frozen-replay",
                    "network_call_count": 0,
                    "policy_kind": "cached_llm",
                    "proposal_seed": 7_664_614_645_859_848_164,
                },
                {
                    "cache_hit": True,
                    "cache_source": "task6-v3-frozen-replay",
                    "network_call_count": 0,
                    "policy_kind": "cached_llm",
                    "proposal_seed": 6_479_548_443_684_787_591,
                },
                {
                    "cache_hit": True,
                    "cache_source": "task6-v3-frozen-replay",
                    "network_call_count": 0,
                    "policy_kind": "cached_llm",
                    "proposal_seed": 1_839_419_232_351_933_406,
                }
            ]

    assert set(map(tuple, seed_sequences.values())) == {
        (
            7_664_614_645_859_848_164,
            6_479_548_443_684_787_591,
            1_839_419_232_351_933_406,
        )
    }

    repeated = runner.execute(
        bundle,
        AttackerPolicy(
            attacker_mode=AttackerMode.DECISION_ONLY,
            family="app_scam_mule",
            kind=AttackerPolicyKind.FIXED,
            query_budget=3,
            worker_timeout_ms=5_000,
        ),
    )

    assert repeated == manifests[AttackerPolicyKind.FIXED]
    assert len({manifest.policy_kind for manifest in manifests.values()}) == 4


def test_g2_uses_accepted_frozen_task6_capability_evidence_without_rerun() -> None:
    """Pin historical bytes/mode; the one-command gate performs exact recomputation."""
    metadata = TASK6_RESULT.lstat()
    raw = TASK6_RESULT.read_bytes()

    assert stat.S_ISREG(metadata.st_mode)
    assert not TASK6_RESULT.is_symlink()
    assert stat.S_IMODE(metadata.st_mode) in {0o600, 0o644}
    assert hashlib.sha256(raw).hexdigest() == TASK6_RESULT_SHA256
    tree = subprocess.run(
        [
            "git",
            "ls-tree",
            "d6d3eecbfe2d871af8375e1455814cb5c48f2928",
            "--",
            "docs/experiments/task6-v3.4-holdout-result.json",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    assert tree.startswith("100644 blob ")
    historical = subprocess.run(
        [
            "git",
            "show",
            "d6d3eecbfe2d871af8375e1455814cb5c48f2928:"
            "docs/experiments/task6-v3.4-holdout-result.json",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
    ).stdout
    assert hashlib.sha256(historical).hexdigest() == TASK6_RESULT_SHA256
