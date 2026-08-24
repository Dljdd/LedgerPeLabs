"""Build a non-published seed-404 Sentinel v5 evidence fixture."""

from __future__ import annotations

import argparse
from importlib import import_module
from pathlib import Path

from apar.evaluation.v5_evidence_bundle import build_v5_evidence_envelope
from apar.evaluation.v5_run_mode import V5RunMode

_execution_module = import_module(
    f"{__package__}.v5_complete_evidence_execution"
    if __package__
    else "v5_complete_evidence_execution"
)


def build_safe_evidence(root: Path) -> bytes:
    """Execute only the frozen smoke profile with the isolated safe seed."""
    executed = _execution_module.execute_v5_complete_evidence(
        root=root, mode=V5RunMode.SAFE_VALIDATION
    )
    return build_v5_evidence_envelope(
        seed=404,
        evidence_protocol=executed.evidence_protocol,
        catalog_sha256=executed.catalog.catalog_sha256,
        arm_results=executed.arm_results,
        controls=executed.controls,
    ).serialized_bytes()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    output = args.output.resolve()
    if not (root / "config/defense/defense-v5-evidence.json").is_file():
        raise SystemExit(
            "repository root is missing the Sentinel v5 evidence configuration"
        )
    if output.exists():
        raise SystemExit("refusing to overwrite an existing safe evidence fixture")
    output.write_bytes(build_safe_evidence(root))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
