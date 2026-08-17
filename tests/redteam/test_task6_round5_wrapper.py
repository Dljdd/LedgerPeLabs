"""Task 6 round-five portable raw-evidence wrapper regressions."""

from __future__ import annotations

import ast
import hashlib
import json
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import cast

import pytest

import apar.redteam.task6_verifier as evidence_verifier
import scripts.run_task6_holdout as holdout_runner
from apar.redteam.task6_verifier import (
    EvidenceVerificationError,
    canonical_digest,
    verify_result_bundle,
)

ROOT = Path(__file__).resolve().parents[2]
PREREGISTRATION = ROOT / "docs/experiments/task6-v3.4-holdout-preregistration.json"
RESULT = ROOT / "docs/experiments/task6-v3.4-holdout-result.json"
CACHE = ROOT / "docs/experiments/task6-v3-cached-llm-replay.json"


def _load(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert type(value) is dict
    return value


def _cells(bundle: dict[str, object]) -> list[dict[str, object]]:
    evidence = cast(dict[str, object], bundle["evidence"])
    return cast(list[dict[str, object]], evidence["cells"])


def _refresh_cell(cell: dict[str, object]) -> None:
    search = cast(dict[str, object], cell["search_result"])
    search["canonical_document_digest"] = canonical_digest(search["document"])
    core = {key: value for key, value in cell.items() if key != "cell_digest"}
    cell["cell_digest"] = canonical_digest(core)


def _refresh_bundle(bundle: dict[str, object]) -> None:
    evidence = cast(dict[str, object], bundle["evidence"])
    evidence["evidence_digest"] = canonical_digest(evidence["cells"])
    core = {key: value for key, value in bundle.items() if key != "bundle_digest"}
    bundle["bundle_digest"] = canonical_digest(core)


def _set_evaluator_id(cell: dict[str, object], capability_id: str) -> None:
    public = cast(dict[str, object], cell["public_context"])
    evaluator = cast(dict[str, object], public["evaluator_binding"])
    evaluator["capability_id"] = capability_id
    search = cast(dict[str, object], cell["search_result"])
    document = cast(dict[str, object], search["document"])
    document["evaluator_capability_id"] = capability_id
    _refresh_cell(cell)


def _set_policy_id(cell: dict[str, object], capability_id: str) -> None:
    public = cast(dict[str, object], cell["public_context"])
    policy = cast(dict[str, object], public["policy_binding"])
    policy["capability_id"] = capability_id
    search = cast(dict[str, object], cell["search_result"])
    document = cast(dict[str, object], search["document"])
    document["policy_capability_id"] = capability_id
    _refresh_cell(cell)


def _mutate_one_evaluator(bundle: dict[str, object]) -> None:
    _set_evaluator_id(_cells(bundle)[0], "1" * 64)


def _mutate_one_policy(bundle: dict[str, object]) -> None:
    _set_policy_id(_cells(bundle)[0], "2" * 64)


def _reuse_evaluator_across_families(bundle: dict[str, object]) -> None:
    cells = _cells(bundle)
    app_id = cast(
        dict[str, object],
        cast(dict[str, object], cells[0]["public_context"])["evaluator_binding"],
    )["capability_id"]
    assert type(app_id) is str
    for cell in cells:
        if cell["family"] == "card_testing_cnp":
            _set_evaluator_id(cell, app_id)


def _reuse_policy_id(bundle: dict[str, object]) -> None:
    cells = _cells(bundle)
    fixed = next(cell for cell in cells if cell["policy_name"] == "fixed")
    fixed_id = cast(
        dict[str, object],
        cast(dict[str, object], fixed["public_context"])["policy_binding"],
    )["capability_id"]
    assert type(fixed_id) is str
    for cell in cells:
        if cell["policy_name"] == "random":
            _set_policy_id(cell, fixed_id)


def _split_authority(bundle: dict[str, object]) -> None:
    cell = _cells(bundle)[0]
    search = cast(dict[str, object], cell["search_result"])
    cast(dict[str, object], search["document"])["authority_id"] = "3" * 64
    _refresh_cell(cell)


def _split_run_group(bundle: dict[str, object]) -> None:
    cell = _cells(bundle)[0]
    search = cast(dict[str, object], cell["search_result"])
    cast(dict[str, object], search["document"])["run_group_id"] = "4" * 64
    _refresh_cell(cell)


def _mismatch_embedded_policy(bundle: dict[str, object]) -> None:
    cell = _cells(bundle)[0]
    search = cast(dict[str, object], cell["search_result"])
    cast(dict[str, object], search["document"])["policy_capability_id"] = "5" * 64
    _refresh_cell(cell)


def _duplicate_result_id(bundle: dict[str, object]) -> None:
    first, second = _cells(bundle)[:2]
    first_search = cast(dict[str, object], first["search_result"])
    second_search = cast(dict[str, object], second["search_result"])
    first_id = cast(dict[str, object], first_search["document"])["result_id"]
    cast(dict[str, object], second_search["document"])["result_id"] = first_id
    _refresh_cell(second)


def _duplicate_cell_id(bundle: dict[str, object]) -> None:
    first, second = _cells(bundle)[:2]
    second["cell_id"] = first["cell_id"]
    _refresh_cell(second)


def _reuse_result_id_as_cell_id(bundle: dict[str, object]) -> None:
    first, second = _cells(bundle)[:2]
    first_search = cast(dict[str, object], first["search_result"])
    second["cell_id"] = cast(dict[str, object], first_search["document"])["result_id"]
    _refresh_cell(second)


def _mutate_evaluator_code(bundle: dict[str, object]) -> None:
    cell = _cells(bundle)[0]
    public = cast(dict[str, object], cell["public_context"])
    binding = cast(dict[str, object], public["evaluator_binding"])
    binding["code_digest"] = "6" * 64
    search = cast(dict[str, object], cell["search_result"])
    cast(dict[str, object], search["document"])["evaluator_code_digest"] = "6" * 64
    _refresh_cell(cell)


def _mutate_policy_code(bundle: dict[str, object]) -> None:
    cell = _cells(bundle)[0]
    public = cast(dict[str, object], cell["public_context"])
    binding = cast(dict[str, object], public["policy_binding"])
    binding["code_digest"] = "7" * 64
    search = cast(dict[str, object], cell["search_result"])
    cast(dict[str, object], search["document"])["policy_code_digest"] = "7" * 64
    _refresh_cell(cell)


def _mutate_policy_callable(bundle: dict[str, object]) -> None:
    cell = _cells(bundle)[0]
    public = cast(dict[str, object], cell["public_context"])
    binding = cast(dict[str, object], public["policy_binding"])
    binding["callable_digest"] = "8" * 64
    search = cast(dict[str, object], cell["search_result"])
    cast(dict[str, object], search["document"])["policy_callable_digest"] = "8" * 64
    _refresh_cell(cell)


def _mutate_policy_version(bundle: dict[str, object]) -> None:
    cell = _cells(bundle)[0]
    public = cast(dict[str, object], cell["public_context"])
    binding = cast(dict[str, object], public["policy_binding"])
    binding["version"] = "9.9.9"
    search = cast(dict[str, object], cell["search_result"])
    cast(dict[str, object], search["document"])["policy_version"] = "9.9.9"
    _refresh_cell(cell)


def _refresh_contract_and_search(cell: dict[str, object]) -> None:
    public = cast(dict[str, object], cell["public_context"])
    contract = cast(dict[str, object], public["evaluation_contract"])
    disclosure = cast(dict[str, object], contract["disclosure_profile"])
    disclosure["profile_digest"] = canonical_digest(
        {
            "profile_id": disclosure["profile_id"],
            "expose_realized_value": disclosure["expose_realized_value"],
        }
    )
    contract["contract_digest"] = canonical_digest(
        {
            "family": contract["family"],
            "bounds_digest": contract["bounds_digest"],
            "hidden_template_digest": contract["hidden_template_digest"],
            "background_digest": contract["background_digest"],
            "population_digest": contract["population_digest"],
            "evaluator_digest": contract["evaluator_digest"],
            "defender_digest": contract["defender_digest"],
            "disclosure_profile_digest": disclosure["profile_digest"],
        }
    )
    search = cast(dict[str, object], cell["search_result"])
    document = cast(dict[str, object], search["document"])
    for name in (
        "bounds_digest",
        "hidden_template_digest",
        "background_digest",
        "population_digest",
        "evaluator_digest",
        "defender_digest",
    ):
        document[name] = contract[name]
    document["disclosure_profile_digest"] = disclosure["profile_digest"]
    document["evaluation_contract_digest"] = contract["contract_digest"]
    _refresh_cell(cell)


def _mutate_contract_field(name: str) -> ArtifactMutation:
    def mutate(bundle: dict[str, object]) -> None:
        cell = _cells(bundle)[0]
        public = cast(dict[str, object], cell["public_context"])
        contract = cast(dict[str, object], public["evaluation_contract"])
        contract[name] = "9" * 64
        _refresh_contract_and_search(cell)

    return mutate


def _mutate_disclosure(bundle: dict[str, object]) -> None:
    cell = _cells(bundle)[0]
    public = cast(dict[str, object], cell["public_context"])
    contract = cast(dict[str, object], public["evaluation_contract"])
    disclosure = cast(dict[str, object], contract["disclosure_profile"])
    disclosure["profile_id"] = "forged-profile"
    _refresh_contract_and_search(cell)


def _mutate_bounds(bundle: dict[str, object]) -> None:
    cell = _cells(bundle)[0]
    public = cast(dict[str, object], cell["public_context"])
    bounds = cast(dict[str, object], public["public_bounds"])
    bounds["defaults"] = []
    contract = cast(dict[str, object], public["evaluation_contract"])
    contract["bounds_digest"] = canonical_digest(bounds)
    _refresh_contract_and_search(cell)


def _mutate_defender(bundle: dict[str, object]) -> None:
    cell = _cells(bundle)[0]
    public = cast(dict[str, object], cell["public_context"])
    defender = cast(dict[str, object], public["defender"])
    rules = cast(list[dict[str, object]], defender["rules"])
    rules[0]["threshold"] = "999"
    defender["defender_digest"] = canonical_digest(
        {"version": defender["version"], "rules": rules}
    )
    contract = cast(dict[str, object], public["evaluation_contract"])
    contract["defender_digest"] = defender["defender_digest"]
    _refresh_contract_and_search(cell)


ArtifactMutation = Callable[[dict[str, object]], None]


def test_artifact_scoped_provenance_verifies_the_frozen_bundle_without_runtime_ids() -> None:
    preregistration = _load(PREREGISTRATION)
    result = _load(RESULT)
    cache = _load(CACHE)
    derive = getattr(
        evidence_verifier,
        "derive_artifact_scoped_provenance",
        None,
    )
    assert callable(derive)

    contexts, policies = derive(
        result,
        preregistered_contexts=preregistration["evidence_contexts"],
        preregistered_policy_bindings=preregistration["policy_bindings"],
    )
    summary = verify_result_bundle(
        result,
        expected_protocol=preregistration["protocol"],
        expected_contexts=contexts,
        expected_policy_bindings=policies,
        expected_llm_cache=cache["records"],
        expected_external_approval=result["external_approval"],
        expected_preregistration_canonical_digest=result[
            "preregistration_canonical_digest"
        ],
    )

    assert summary == result["summary"]


def test_frozen_algorithm_evaluator_and_generator_files_are_byte_identical() -> None:
    preregistration = _load(PREREGISTRATION)
    behavior = cast(dict[str, object], preregistration["behavior_equivalence"])
    source = cast(dict[str, object], preregistration["source_freeze"])
    source_commit = cast(str, source["source_commit"])
    manifest = cast(dict[str, object], source["behavior_manifest"])
    entries = cast(dict[str, dict[str, str]], manifest["entries"])
    proposal_paths = set(
        cast(dict[str, str], behavior["proposal_implementation_sha256"])
    )
    generator_paths = set(
        cast(dict[str, str], behavior["generator_implementation_sha256"])
    )
    evaluator_paths = {
        "src/apar/redteam/benchmark.py",
        "src/apar/redteam/search.py",
        "src/apar/simulator/engine.py",
        "src/apar/simulator/ledger.py",
        "src/apar/simulator/rails/a2a.py",
        "src/apar/simulator/rails/agentic.py",
        "src/apar/simulator/rails/card.py",
        "src/apar/trust/verifier.py",
    }
    frozen_paths = proposal_paths | generator_paths | evaluator_paths

    for relative in sorted(frozen_paths):
        current = (ROOT / relative).read_bytes()
        historical = subprocess.run(
            ["git", "show", f"{source_commit}:{relative}"],
            cwd=ROOT,
            check=True,
            capture_output=True,
        ).stdout
        assert current == historical, relative
        assert hashlib.sha256(current).hexdigest() == entries[relative][
            "content_sha256"
        ]


def test_execution_and_search_paths_are_ast_identical_to_approved_result_head() -> None:
    approved_head = "d6d3eecbfe2d871af8375e1455814cb5c48f2928"
    historical = subprocess.run(
        ["git", "show", f"{approved_head}:scripts/run_task6_holdout.py"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    current_tree = ast.parse((ROOT / "scripts/run_task6_holdout.py").read_text())
    historical_tree = ast.parse(historical)

    def selected(tree: ast.Module) -> dict[str, str]:
        names = {
            "_bootstrap_runtime",
            "_runtime",
            "_confirmatory_gate",
            "_run_negative_control",
            "_execute",
        }
        functions = {
            node.name: ast.dump(node, include_attributes=False)
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name in names
        }
        assert set(functions) == names
        return functions

    assert selected(current_tree) == selected(historical_tree)


@pytest.mark.parametrize(
    "mutate",
    [
        _mutate_one_evaluator,
        _mutate_one_policy,
        _reuse_evaluator_across_families,
        _reuse_policy_id,
        _split_authority,
        _split_run_group,
        _mismatch_embedded_policy,
        _duplicate_result_id,
        _duplicate_cell_id,
        _reuse_result_id_as_cell_id,
    ],
    ids=[
        "one-cell-evaluator-relabel",
        "one-cell-policy-relabel",
        "cross-family-evaluator-reuse",
        "cross-policy-id-reuse",
        "authority-split",
        "run-group-split",
        "embedded-public-mismatch",
        "duplicate-result-id",
        "duplicate-cell-id",
        "cross-kind-id-reuse",
    ],
)
def test_artifact_scoped_provenance_rejects_recomputed_id_topology_tamper(
    mutate: ArtifactMutation,
) -> None:
    preregistration = _load(PREREGISTRATION)
    result = _load(RESULT)
    mutate(result)
    _refresh_bundle(result)

    with pytest.raises(EvidenceVerificationError, match="identity|capability|provenance"):
        evidence_verifier.derive_artifact_scoped_provenance(
            result,
            preregistered_contexts=preregistration["evidence_contexts"],
            preregistered_policy_bindings=preregistration["policy_bindings"],
        )


@pytest.mark.parametrize(
    "mutate",
    [
        _mutate_evaluator_code,
        _mutate_policy_code,
        _mutate_policy_callable,
        _mutate_policy_version,
        _mutate_bounds,
        _mutate_contract_field("hidden_template_digest"),
        _mutate_contract_field("background_digest"),
        _mutate_contract_field("population_digest"),
        _mutate_contract_field("evaluator_digest"),
        _mutate_disclosure,
        _mutate_defender,
    ],
    ids=[
        "evaluator-code",
        "policy-code",
        "policy-callable",
        "policy-version",
        "bounds",
        "template",
        "background",
        "population",
        "evaluator-contract",
        "disclosure",
        "defender",
    ],
)
def test_artifact_scoped_provenance_rejects_recomputed_stable_field_tamper(
    mutate: ArtifactMutation,
) -> None:
    preregistration = _load(PREREGISTRATION)
    result = _load(RESULT)
    mutate(result)
    _refresh_bundle(result)

    with pytest.raises(EvidenceVerificationError, match="preregistered|provenance"):
        evidence_verifier.derive_artifact_scoped_provenance(
            result,
            preregistered_contexts=preregistration["evidence_contexts"],
            preregistered_policy_bindings=preregistration["policy_bindings"],
        )


def test_portable_published_result_verification_uses_no_fresh_capabilities(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    verify = getattr(holdout_runner, "_verify_published_result_portably", None)
    assert callable(verify)

    def forbidden_runtime() -> None:
        raise AssertionError("portable verification constructed runtime capabilities")

    monkeypatch.setattr(holdout_runner, "_bootstrap_runtime", forbidden_runtime)
    monkeypatch.setattr(holdout_runner, "_expected_protocol", forbidden_runtime)
    document, summary = verify(
        expected_freeze_commit="52e8d795c9c2bc40fda1d40178cce50e33349b20"
    )

    assert document["summary"] == summary
    assert summary["confirmatory_valid"] is True
    assert summary["supported_family_count"] == 2


def test_postcommit_chronology_accepts_descendant_head_and_checks_current_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result_commit = "d" * 40
    preregistration_commit = "e" * 40
    descendant_head = "c" * 40
    result_path = "docs/experiments/task6-v3.4-holdout-result.json"
    approved_bytes = b"approved-result-bytes"
    approved_sha = hashlib.sha256(approved_bytes).hexdigest()
    local_result = tmp_path / "task6-v3.4-holdout-result.json"
    local_result.write_bytes(approved_bytes)

    monkeypatch.setattr(holdout_runner, "RESULT_PATH", local_result)
    monkeypatch.setattr(holdout_runner, "_head_commit", lambda: descendant_head)
    monkeypatch.setattr(
        holdout_runner,
        "_commit_is_ancestor",
        lambda ancestor, descendant: (ancestor, descendant)
        == (result_commit, descendant_head),
        raising=False,
    )

    def git_output(arguments: list[str], *, text: bool = True) -> str | bytes:
        if arguments == ["rev-list", "--parents", "-n", "1", result_commit]:
            return f"{result_commit} {preregistration_commit}\n"
        if arguments == [
            "diff-tree",
            "--no-commit-id",
            "--name-status",
            "-r",
            "-z",
            result_commit,
        ]:
            return b"A\0" + result_path.encode() + b"\0"
        if arguments == ["cat-file", "blob", "a" * 40]:
            return approved_bytes
        raise AssertionError(arguments)

    monkeypatch.setattr(holdout_runner, "_git_output", git_output)
    monkeypatch.setattr(
        holdout_runner,
        "_git_tree_records",
        lambda _commit: {
            result_path: {
                "git_mode": "100644",
                "object_type": "blob",
                "git_object_id": "a" * 40,
            }
        },
    )

    holdout_runner._validate_postcommit_chronology(
        approved_result_commit=result_commit,
        approved_result_sha256=approved_sha,
        preregistration_commit=preregistration_commit,
        approved_artifacts={result_path: approved_sha},
    )

    local_result.write_bytes(b"tampered-but-internally-self-consistent")
    with pytest.raises(RuntimeError, match="SHA|size|bytes|artifact"):
        holdout_runner._validate_postcommit_chronology(
            approved_result_commit=result_commit,
            approved_result_sha256=approved_sha,
            preregistration_commit=preregistration_commit,
            approved_artifacts={result_path: approved_sha},
        )


def test_approved_result_sha_rejects_recomputed_nonportable_seal_tamper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = _load(RESULT)
    first = _cells(result)[0]
    search = cast(dict[str, object], first["search_result"])
    search["process_local_issuance_seal"] = "a" * 64
    _refresh_cell(first)
    _refresh_bundle(result)
    tampered_path = tmp_path / RESULT.name
    tampered_path.write_bytes(evidence_verifier.canonical_json_bytes(result))
    monkeypatch.setattr(holdout_runner, "RESULT_PATH", tampered_path)

    with pytest.raises(RuntimeError, match="SHA|size|artifact"):
        holdout_runner._validate_postcommit_chronology(
            approved_result_commit="d6d3eecbfe2d871af8375e1455814cb5c48f2928",
            approved_result_sha256=(
                "f82981a987651a7f7ebb10a9011df063b2dc54a56181cae5b838e31de5e658db"
            ),
            preregistration_commit="52e8d795c9c2bc40fda1d40178cce50e33349b20",
            approved_artifacts={
                "docs/experiments/task6-v3.4-holdout-result.json": (
                    "f82981a987651a7f7ebb10a9011df063"
                    "b2dc54a56181cae5b838e31de5e658db"
                )
            },
        )


@pytest.mark.parametrize("mode", ["--verify-postexecution", "--verify-postcommit"])
def test_postverification_cli_routes_only_to_portable_artifact_verification(
    mode: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    freeze_commit = "52e8d795c9c2bc40fda1d40178cce50e33349b20"
    arguments = ["run_task6_holdout.py", mode]
    if mode == "--verify-postcommit":
        arguments.extend(
            [
                "--approved-result-commit",
                "d6d3eecbfe2d871af8375e1455814cb5c48f2928",
                "--approved-result-sha256",
                "f82981a987651a7f7ebb10a9011df063b2dc54a56181cae5b838e31de5e658db",
            ]
        )
    monkeypatch.setattr(sys, "argv", arguments)
    monkeypatch.setattr(holdout_runner, "_require_clean_worktree", lambda: None)
    monkeypatch.setattr(holdout_runner, "_head_commit", lambda: freeze_commit)

    def git_output(arguments: list[str], *, text: bool = True) -> str | bytes:
        if arguments[:2] == ["status", "--porcelain=v1"]:
            return b"?? docs/experiments/task6-v3.4-holdout-result.json\0"
        raise AssertionError(arguments)

    monkeypatch.setattr(holdout_runner, "_git_output", git_output)
    chronology_calls: list[dict[str, object]] = []
    monkeypatch.setattr(
        holdout_runner,
        "_validate_postcommit_chronology",
        lambda **kwargs: chronology_calls.append(kwargs),
    )
    portable_calls: list[str] = []

    def portable(*, expected_freeze_commit: str) -> tuple[dict[str, object], dict[str, object]]:
        portable_calls.append(expected_freeze_commit)
        return {}, {"confirmatory_valid": True}

    monkeypatch.setattr(
        holdout_runner,
        "_verify_published_result_portably",
        portable,
    )

    def forbidden_runtime() -> None:
        raise AssertionError("postverification constructed fresh capabilities")

    monkeypatch.setattr(holdout_runner, "_runtime", forbidden_runtime)

    holdout_runner.main()

    assert portable_calls == [freeze_commit]
    assert bool(chronology_calls) is (mode == "--verify-postcommit")
