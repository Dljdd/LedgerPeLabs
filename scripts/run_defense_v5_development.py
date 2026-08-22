"""Run the Sentinel v5 development pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from apar.evaluation.v5_population import build_v5_corpus
from apar.evaluation.v5_protocol import V5Profile, load_v5_development_protocol
from apar.evaluation.v5_reporting import build_v5_development_result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=["smoke", "production"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    protocol = load_v5_development_protocol(root / "config/defense/defense-v5-development.json")
    profile = V5Profile(args.profile)
    corpus = build_v5_corpus(protocol, profile=profile)
    result = build_v5_development_result(protocol=protocol, corpus=corpus)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(json.loads(result.model_dump_json()), indent=2) + "\n")
    print(f"status={result.status} profile={result.profile} output={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
