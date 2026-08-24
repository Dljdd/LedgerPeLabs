"""Independently reconstruct and verify locked Sentinel v5 evidence offline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from apar.v5_independent_verifier import (
    IndependentVerificationError,
    read_locked_evidence_storage_bytes,
    verify_locked_evidence_payload_bytes,
    verify_locked_judge_summary,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("candidate_manifest", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    try:
        raw = json.loads(
            (root / "config/defense/defense-v5-evidence.json").read_bytes()
        )
        storage = raw["locked_artifact_storage"]
        expected_target = (root / storage["candidate_manifest_path"]).resolve()
        if args.candidate_manifest.resolve() != expected_target:
            raise ValueError("locked evidence is not at the frozen candidate path")
        payload = read_locked_evidence_storage_bytes(
            target_manifest=args.candidate_manifest.resolve(),
            attempt_receipt_path=root / storage["attempt_receipt_path"],
            chunk_size_bytes=int(storage["chunk_size_bytes"]),
            maximum_envelope_bytes=int(storage["maximum_envelope_bytes"]),
            maximum_chunk_count=int(storage["maximum_chunk_count"]),
            normal_git_blob_limit_bytes=int(
                storage["normal_git_blob_limit_bytes"]
            ),
        )
        report = verify_locked_evidence_payload_bytes(payload, root=root)
        verify_locked_judge_summary(
            summary_path=root / storage["judge_summary_path"],
            target_manifest=args.candidate_manifest.resolve(),
            attempt_receipt_path=root / storage["attempt_receipt_path"],
            verification=report,
            candidate_manifest_path=storage["candidate_manifest_path"],
            declared_attempt_receipt_path=storage["attempt_receipt_path"],
        )
    except (
        IndependentVerificationError,
        json.JSONDecodeError,
        KeyError,
        OSError,
        TypeError,
        ValueError,
    ) as error:
        print(
            json.dumps(
                {"verified": False, "error": str(error)},
                sort_keys=True,
                separators=(",", ":"),
            ),
            file=sys.stderr,
        )
        return 1
    print(json.dumps(report, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
