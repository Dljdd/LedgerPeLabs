"""Cross-process corpus determinism regression tests."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

_SCRIPT = """
import hashlib, json, sys
sys.path.insert(0, {root!r})
from pathlib import Path
from apar.evaluation.v5_population import build_v5_corpus
from apar.evaluation.v5_protocol import V5Profile, load_v5_development_protocol
protocol = load_v5_development_protocol(Path({root!r}) / "config/defense/defense-v5-development.json")
corpus = build_v5_corpus(protocol, profile=V5Profile.SMOKE)
counts = {{name: len(p.decisions) for name, p in corpus.partitions.items()}}
print(json.dumps({{"digest": corpus.corpus_sha256, "counts": counts}}))
""".format(root=str(ROOT))


class TestCrossProcessDeterminism:
    def test_identical_across_hash_seeds(self) -> None:
        results = []
        for hash_seed in ("0", "42", "12345"):
            env = dict(__import__("os").environ)
            env["PYTHONHASHSEED"] = hash_seed
            proc = subprocess.run(
                [sys.executable, "-c", _SCRIPT],
                capture_output=True,
                text=True,
                cwd=str(ROOT),
                env=env,
                timeout=120,
            )
            assert proc.returncode == 0, f"PYTHONHASHSEED={hash_seed}: {proc.stderr}"
            results.append(json.loads(proc.stdout.strip()))

        digests = {r["digest"] for r in results}
        assert len(digests) == 1, f"corpus digest varies across PYTHONHASHSEED: {digests}"
        counts_list = [json.dumps(r["counts"], sort_keys=True) for r in results]
        assert len(set(counts_list)) == 1, f"partition counts vary: {counts_list}"
