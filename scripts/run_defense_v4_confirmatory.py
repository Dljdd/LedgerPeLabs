"""Explicitly gated one-attempt confirmatory execution CLI for Defend v4."""

from __future__ import annotations

import argparse
import hashlib
import json
import secrets
import sys
from pathlib import Path

from apar.defense.contracts import ObservedEvent
from apar.evaluation.v3_population import build_efficacy_population
from apar.evaluation.v4_preexecution import verify_v4_preexecution
from apar.evaluation.v4_publication import from_gate_results, render_v4_scorecard
from apar.evaluation.v4_runner import (
    V4ExecutionInputs,
    create_v4_receipt,
    execute_v4_arms,
    finalize_v4_receipt,
    verify_v4_approval,
)
from apar.runs.wire import canonical_json_bytes
from apar.v4_protocol import V4GateValues


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--approval-token", type=str, required=True)
    parser.add_argument("--campaign-count", type=int, default=2)
    parser.add_argument("--decisions-per-campaign", type=int, default=4)
    args = parser.parse_args()
    root = args.root.resolve()

    report = verify_v4_preexecution(root)
    if not report.admissible:
        print(f"pre-execution failed: {report.codes}", file=sys.stderr)
        return 1

    def file_digest(path: str) -> str:
        return hashlib.sha256((root / path).read_bytes()).hexdigest()

    source_tree = hashlib.sha256(
        b"".join(
            file_digest(str(p)).encode()
            for p in sorted((root / "src/apar").rglob("*.py"))
        )
    ).hexdigest()
    config_digest = file_digest("config/defense/competition-v2-manifests.json")
    bundle_digest = file_digest("fixtures/defense/v1/defender-bundle.json")

    population = build_efficacy_population(
        campaign_count_per_family=args.campaign_count,
        decisions_per_campaign=args.decisions_per_campaign,
        day_count=28,
        seed=7,
    )
    population_digest = hashlib.sha256(
        canonical_json_bytes(
            {
                "observations": population.manifest.observations_sha256,
                "truth": population.manifest.truth_sha256,
            }
        )
    ).hexdigest()

    freeze_document = json.dumps(
        {
            "protocol_id": "apar-defend-v4",
            "source_tree_sha256": source_tree,
            "config_manifest_sha256": config_digest,
            "defender_bundle_sha256": bundle_digest,
            "population_manifest_sha256": population_digest,
            "maximum_confirmatory_attempts": 1,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    freeze_digest = hashlib.sha256(freeze_document).hexdigest()

    execution_nonce = secrets.token_hex(32)
    inputs = V4ExecutionInputs(
        protocol_id="apar-defend-v4",
        execution_nonce=execution_nonce,
        source_tree_sha256=source_tree,
        config_manifest_sha256=config_digest,
        defender_bundle_sha256=bundle_digest,
        population_manifest_sha256=population_digest,
        evaluator_key_id=hashlib.sha256(b"v4-execution-authority").hexdigest(),
        approval_token=args.approval_token,
    )

    verify_v4_approval(inputs, expected_freeze_digest=freeze_digest)

    receipt_dir = root / ".apar" / "defense-v4"
    receipt = create_v4_receipt(inputs, directory=receipt_dir)
    print(f"RECEIPT_CREATED: {receipt.execution_nonce[:16]}...")

    try:
        gate_results = execute_v4_arms(
            inputs,
            observations=population.observations,
            observations_sha256=population.manifest.observations_sha256,
            truth_sha256=population.manifest.truth_sha256,
            gates=V4GateValues(),
        )
        render_result = from_gate_results(
            protocol_id="apar-defend-v4",
            execution_nonce=execution_nonce,
            results=gate_results,
        )
        for result in gate_results:
            print(f"{result.arm}: {'PASS' if result.gate_outcome.passed else 'FAIL'} {list(result.gate_outcome.codes)}")
        overall_status = render_result.status
    except Exception as error:
        print(f"EXECUTION_ERROR: {error}", file=sys.stderr)
        overall_status = "no_promotion"

    finalized = finalize_v4_receipt(receipt, directory=receipt_dir, status=overall_status)
    print(f"TERMINAL_STATUS: {finalized.terminal_status}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
